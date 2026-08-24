"""Stage 2: half.parquet -> +`summary` column.

Calls the explainer model once per row to turn `context_text` into the natural
language summary `s`. That summary is the AV's SFT target (`h -> s`) and the AR's
SFT input (`s -> h`), so both halves go through this stage.

Written chunk-by-chunk to `{output}.chunks/`, and existing chunk files are
skipped on restart: a crash at chunk 300/500 costs that chunk's API calls, not
the 299 before it. At 500k rows the API bill is the expensive part of this
pipeline, so resumability is not optional.

Rows whose response arrives without `<analysis>` tags, or with fewer than
`MIN_FEATURES` features after cleanup, are dropped rather than kept as partial
explanations — training on half a thought is worse than training on less data.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from reasoning_attention.config import ExplainerConfig, load_project_env
from reasoning_attention.datagen.prompts import (
    EXPLAIN_INSTRUCTION,
    MIN_FEATURES,
    build_explain_prompt,
    count_features,
    extract_and_clean,
)
from reasoning_attention.datagen.providers import CompletionProvider, OpenAIProvider
from reasoning_attention.datagen.sidecar import (
    DatasetMeta,
    ExplainerMeta,
    ExtractionMeta,
    read_sidecar,
    write_sidecar,
)


def explain_chunk(chunk: pa.Table, provider: CompletionProvider) -> tuple[pa.Table, int]:
    """Add a `summary` column to one chunk, dropping unusable rows.

    Returns `(chunk_with_summaries, n_dropped)`.
    """
    texts = chunk.column("context_text").to_pylist()
    completions = provider.complete([build_explain_prompt(t) for t in texts])
    assert len(completions) == len(texts), (
        f"provider returned {len(completions)} completions for {len(texts)} prompts — "
        f"length mismatch violates the CompletionProvider contract"
    )

    keep: list[bool] = []
    summaries: list[str] = []
    for raw in completions:
        cleaned = extract_and_clean(raw) if raw is not None else None
        if cleaned is None or count_features(cleaned) < MIN_FEATURES:
            keep.append(False)
            continue
        keep.append(True)
        summaries.append(cleaned)

    n_dropped = keep.count(False)
    if n_dropped:
        chunk = chunk.filter(pa.array(keep, type=pa.bool_()))
    return chunk.append_column("summary", pa.array(summaries, type=pa.string())), n_dropped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="av_half.parquet or ar_half.parquet")
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-size", type=int, default=512, help="rows per API batch")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N rows")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="in-flight requests (hosted API measured at 7.38 calls/s at 128 vs 2.08 at 32)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible server to label against, e.g. http://127.0.0.1:8000/v1 "
        "for `vllm serve`. Omit to use the hosted API.",
    )
    parser.add_argument("--model", default=None, help="override the explainer model id")
    parser.add_argument(
        "--api-kind",
        default=None,
        choices=["responses", "chat"],
        help="'chat' for local servers (vLLM/SGLang serve /v1/chat/completions and have "
        "no reasoning-effort parameter); defaults to 'chat' when --base-url is given",
    )
    args = parser.parse_args()

    load_project_env()
    overrides: dict[str, object] = {}
    if args.base_url:
        # A local server speaks chat-completions and has no reasoning knob, so
        # default api_kind accordingly rather than making the caller remember.
        overrides["base_url"] = args.base_url
        overrides["api_kind"] = args.api_kind or "chat"
    elif args.api_kind:
        overrides["api_kind"] = args.api_kind
    if args.model:
        overrides["model"] = args.model
    if args.concurrency:
        overrides["concurrency"] = args.concurrency
    config = replace(ExplainerConfig(), **overrides)  # type: ignore[arg-type]
    print(
        f"explainer: {config.model} via "
        f"{config.base_url or 'hosted API'} ({config.api_kind}), "
        f"concurrency {config.concurrency}"
    )
    in_meta = read_sidecar(args.input)
    provider = OpenAIProvider(config)

    table = pq.read_table(args.input)
    if args.limit is not None:
        table = table.slice(0, args.limit)
    out_schema = table.schema.append(pa.field("summary", pa.string()))

    chunks_dir = Path(f"{args.output}.chunks")
    chunks_dir.mkdir(parents=True, exist_ok=True)

    starts = list(range(0, table.num_rows, args.chunk_size))
    chunk_paths = [chunks_dir / f"chunk_{s:08d}.parquet" for s in starts]
    n_dropped = 0
    n_resumed = 0
    for start, chunk_path in zip(starts, chunk_paths, strict=True):
        if chunk_path.exists():
            n_resumed += 1
            continue
        out_chunk, dropped = explain_chunk(table.slice(start, args.chunk_size), provider)
        n_dropped += dropped
        # tmp + rename so a kill mid-write can't leave a half-written chunk that
        # a later run would mistake for complete.
        tmp = chunk_path.with_suffix(".tmp")
        pq.write_table(out_chunk, tmp)
        tmp.rename(chunk_path)
        print(f"  chunk {start}: +{out_chunk.num_rows} rows, -{dropped} dropped")
    if n_resumed:
        print(f"resumed: skipped {n_resumed}/{len(starts)} completed chunks")

    row_count = 0
    with pq.ParquetWriter(args.output, out_schema) as writer:
        for chunk_path in tqdm(chunk_paths, desc="merging"):
            part = pq.read_table(chunk_path)
            writer.write_table(part)
            row_count += part.num_rows

    assert row_count > 0, (
        f"every row was dropped. Either responses never matched the <analysis> "
        f"pattern (truncated? raise ExplainerConfig.max_output_tokens) or they had "
        f"fewer than {MIN_FEATURES} features after cleanup."
    )

    meta = DatasetMeta(
        dataset_id=f"{in_meta['dataset_id']}__explained",
        stage=in_meta["stage"],
        row_count=row_count,
        n_documents=in_meta["n_documents"],
        extraction=ExtractionMeta(**in_meta["extraction"]),
        created_by="reasoning_attention.datagen.explain",
        explainer=ExplainerMeta(
            model=config.model,
            reasoning_effort=config.reasoning_effort,
            max_output_tokens=config.max_output_tokens,
            instruction_prompt=EXPLAIN_INSTRUCTION,
        ),
        parent_datasets=[in_meta["dataset_id"]],
    )
    print(f"wrote {row_count} rows -> {args.output}")
    if n_dropped:
        print(f"  DROPPED {n_dropped} rows (no <analysis> tags or < {MIN_FEATURES} features)")
    print(f"sidecar -> {write_sidecar(args.output, meta)}")


if __name__ == "__main__":
    main()
