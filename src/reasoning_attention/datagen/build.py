"""Stage 3: explained halves -> the AV / AR training parquets.

Renders the final training rows. The two halves get different shapes because the
two models are trained in opposite directions:

  **AV half (`h -> s`)** — `prompt` is the AV template carrying the `<INJECT>`
  literal, `response` is the summary wrapped in `<explanation>` tags. The literal
  is stored, not the real placeholder token: the training side swaps it for
  `NLAConfig.placeholder_token` at load time, so a dataset built today stays
  valid if the placeholder is ever repointed. `activation_vector` is the vector
  whose embedding slot the placeholder occupies.

  **AR half (`s -> h`)** — `prompt` is the AR template ending in the fixed
  `</text> <summary>` suffix, and there is no response: the AR is not a language
  model here. Training runs the prompt through the truncated backbone and reads
  the predicted activation at the final token, so the target is
  `activation_vector` and the suffix is what anchors the read position. The
  suffix token ids go in the sidecar for training to verify the tail before
  extracting.

Vectors stay RAW in both. Normalization is training-side.
"""

from __future__ import annotations

import argparse

import pyarrow as pa
import pyarrow.parquet as pq

from reasoning_attention.config import NLAConfig
from reasoning_attention.datagen.sidecar import (
    DatasetMeta,
    ExplainerMeta,
    ExtractionMeta,
    TokenMeta,
    read_sidecar,
    write_sidecar,
)
from reasoning_attention.nla.prompts import (
    AR_TEMPLATE,
    AV_TEMPLATE,
    INJECT_PLACEHOLDER,
    build_ar_prompt,
    build_av_content,
    wrap_explanation,
)

# The AR prompt's fixed tail. Training tokenizes the prompt, asserts these ids
# are its last tokens, then extracts the activation at tokens[-1].
AR_SUFFIX = "</text> <summary>"

# The reference's chat-messages column type (docs/design.md, "Parquet columns").
MESSAGES_TYPE = pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))


def build_av_table(table: pa.Table) -> pa.Table:
    """AV rows: constant prompt with the `<INJECT>` literal, summary as response."""
    n = table.num_rows
    prompt = build_av_content(INJECT_PLACEHOLDER)
    responses = [wrap_explanation(s) for s in table.column("summary").to_pylist()]
    return pa.table(
        {
            "prompt": pa.array([prompt] * n, type=pa.string()),
            "response": pa.array(responses, type=pa.string()),
            "activation_vector": table.column("activation_vector"),
            "doc_id": table.column("doc_id"),
            "n_raw_tokens": table.column("n_raw_tokens"),
            "activation_layer": table.column("activation_layer"),
        }
    )


def build_ar_table(table: pa.Table) -> pa.Table:
    """AR rows: suffix-anchored prompt, activation as the regression target."""
    prompts = [build_ar_prompt(s) for s in table.column("summary").to_pylist()]
    for p in prompts[:1]:
        assert p.endswith(AR_SUFFIX), (
            f"AR prompt does not end with {AR_SUFFIX!r} — the training side extracts "
            f"at tokens[-1] and would read the wrong position. Got: ...{p[-40:]!r}"
        )
    return pa.table(
        {
            "prompt": pa.array(prompts, type=pa.string()),
            "activation_vector": table.column("activation_vector"),
            "doc_id": table.column("doc_id"),
            "n_raw_tokens": table.column("n_raw_tokens"),
            "activation_layer": table.column("activation_layer"),
        }
    )


def build_rl_table(table: pa.Table) -> pa.Table:
    """RL rows: the AV prompt and the activation, no response.

    Stage 2 needs no summary — the AV generates the explanation during rollout
    and the AR turns it back into an activation to compute the reward. So this
    stage runs straight off a base parquet with no API spend.
    """
    n = table.num_rows
    prompt = build_av_content(INJECT_PLACEHOLDER)
    columns = {
        # list[dict] messages, NOT a bare string — unlike the AV-SFT table above.
        # This half is consumed by the reference's NLADataSource, which sets
        # apply_chat_template=False so the loss-mask generator keeps the message
        # list intact, and nla_generate then asserts on the type:
        #   "nla_generate requires list[dict] prompt (got str)".
        # Our own SFT loop reads a plain string and applies the chat template
        # itself, which is why the AV table stays a string.
        "prompt": pa.array([[{"role": "user", "content": prompt}]] * n, type=MESSAGES_TYPE),
        "activation_vector": table.column("activation_vector"),
        "doc_id": table.column("doc_id"),
        "n_raw_tokens": table.column("n_raw_tokens"),
        "activation_layer": table.column("activation_layer"),
    }
    if "source" in table.column_names:
        columns["source"] = table.column("source")
    return pa.table(columns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="explained half, or base parquet for rl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", required=True, choices=["av_sft", "ar_sft", "rl"])
    args = parser.parse_args()

    nla_cfg = NLAConfig()
    in_meta = read_sidecar(args.input)
    table = pq.read_table(args.input)
    if args.stage != "rl":
        assert "summary" in table.column_names, (
            f"{args.input} has no `summary` column — run the explain stage first"
        )

    if args.stage == "rl":
        out = build_rl_table(table)
        templates = {"av": AV_TEMPLATE}
        ar_suffix_ids: list[int] = []
    elif args.stage == "av_sft":
        out = build_av_table(table)
        templates = {"av": AV_TEMPLATE}
        ar_suffix_ids = []
    else:
        out = build_ar_table(table)
        templates = {"ar": AR_TEMPLATE}
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(nla_cfg.model_id)
        ar_suffix_ids = tokenizer(AR_SUFFIX, add_special_tokens=False)["input_ids"]

    pq.write_table(out, args.output)

    explainer = in_meta.get("explainer")
    meta = DatasetMeta(
        dataset_id=f"{in_meta['dataset_id']}__{args.stage}",
        stage=args.stage,
        row_count=out.num_rows,
        n_documents=in_meta["n_documents"],
        extraction=ExtractionMeta(**in_meta["extraction"]),
        created_by="reasoning_attention.datagen.build",
        # Both halves record the placeholder (the AV injects at it, and the
        # training side asserts the id against the live tokenizer); only the AR
        # carries the suffix ids it anchors its extraction on.
        tokens=TokenMeta(
            placeholder_token=nla_cfg.placeholder_token,
            placeholder_token_id=nla_cfg.placeholder_token_id,
            ar_suffix_ids=ar_suffix_ids,
        ),
        explainer=ExplainerMeta(**explainer) if explainer else None,
        prompt_templates=templates,
        parent_datasets=[in_meta["dataset_id"]],
    )
    print(f"{args.stage}: {out.num_rows} rows -> {args.output}")
    print(f"sidecar -> {write_sidecar(args.output, meta)}")


if __name__ == "__main__":
    main()
