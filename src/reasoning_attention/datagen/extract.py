"""Stage 0: activations -> base.parquet.

Forward the target model over Ultra-FineWeb, sample `positions_per_doc` token
positions per document, and store the RAW layer-`l` residual stream at each.

Two invariants carried over from the reference repo, both of which are silent
corruption if broken:

  - **Vectors are stored unnormalized** (`norm="none"` in the sidecar).
    Normalization is a training-time decision (injection scale / MSE scale).
  - **Per-document keyed RNG.** Positions are drawn from an RNG keyed on
    `(seed, doc_id)`, so the same document yields the same positions regardless
    of slice bounds, chunk size, or process count. Runs over disjoint document
    ranges therefore merge into a row-for-row identical dataset.

A forward hook on the single target layer is used rather than
`output_hidden_states=True`, which would materialize all 29 hidden-state tensors
and multiply activation memory by the layer count.

The corpus is read in **streaming** mode. Ultra-FineWeb's `en` split is ~1 TB, so
a non-streaming `load_dataset` downloads the entire corpus before yielding the
first document — it pulled 47 GB in ten minutes on this box before being killed.
Streaming fetches shards lazily, so a 100k-document slice costs roughly the bytes
of those 100k documents.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import random
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from reasoning_attention.config import D_MODEL, NLAConfig, WarmStartDataConfig, load_project_env
from reasoning_attention.datagen.sidecar import DatasetMeta, ExtractionMeta, write_sidecar


def render_chat(conversation: Any, tokenizer: Any) -> str:
    """Flatten a WildChat conversation into one string via the chat template.

    WildChat rows hold `conversation` as a list of `{role, content, ...}` turns.
    Rendering with the *target model's* template means the sampled activations come
    from text shaped the way the model actually sees dialogue, rather than from a
    bespoke concatenation. `add_generation_prompt=False` because we want the whole
    conversation as a document, not a prompt awaiting a reply.
    """
    messages = [
        {"role": str(turn["role"]), "content": str(turn["content"])}
        for turn in conversation
        if turn.get("content")
    ]
    if not messages:
        return ""
    rendered: str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return rendered


def document_text(row: dict[str, Any], column: str, kind: str, tokenizer: Any) -> str:
    """Extract one document's text, per corpus kind."""
    if kind == "chat":
        return render_chat(row[column], tokenizer)
    return str(row[column])


def build_schema(d_model: int) -> pa.Schema:
    """Row schema for base.parquet.

    `activation_vector` is a FixedSizeList, not a variable-length list: every
    vector is exactly `d_model` wide, and the fixed form has no offset array. A
    variable-length list overflows its int32 offsets (and, worse, silently
    corrupts `ChunkedArray.take()` past a 4 GiB values buffer) at the row counts
    this pipeline produces.
    """
    return pa.schema(
        [
            ("doc_id", pa.string()),
            ("n_raw_tokens", pa.int64()),
            ("context_text", pa.string()),
            ("activation_vector", pa.list_(pa.float32(), d_model)),
            ("activation_layer", pa.int64()),
            ("source", pa.string()),
        ]
    )


@dataclass
class ExtractionResult:
    """One document's layer-l hidden states plus the token ids they align to."""

    hidden_states: torch.Tensor  # [seq_len, d_model], float32, CPU, unpadded
    token_ids: list[int]


def sample_positions(
    token_ids: list[int],
    n_positions: int,
    special_ids: set[int],
    doc_id: str,
    seed: int,
    min_position: int,
) -> list[int]:
    """Draw up to `n_positions` distinct token positions for one document.

    Keyed on `(seed, doc_id)` so the draw is independent of how the corpus was
    sliced. Positions below `min_position` are excluded — too little
    left-context for the activation to mean anything — as are special tokens.
    Returns [] for a document with no valid candidates; the caller skips it.
    """
    rng = random.Random(hashlib.sha256(f"{seed}|{doc_id}".encode()).digest())
    candidates = [
        i for i, tid in enumerate(token_ids) if i >= min_position and tid not in special_ids
    ]
    if not candidates:
        return []
    return rng.sample(candidates, k=min(n_positions, len(candidates)))


class Extractor:
    """Runs the target model and captures one layer's output via a forward hook."""

    def __init__(
        self,
        model_id: str,
        layer_index: int,
        max_context_tokens: int,
        batch_size: int = 8,
        dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Both sides MUST be "right". We slice [:seq_len] to drop padding, so
        # left-padding would hand back pad-position activations; left-truncation
        # would mean token_ids[0] is not the document start and every position
        # index would refer to the wrong text.
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, device_map=device_map
        ).eval()
        self.layer_index = layer_index
        self.max_context_tokens = max_context_tokens
        self.batch_size = batch_size
        self.d_model = int(self.model.config.hidden_size)
        self._captured: torch.Tensor | None = None

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> None:
        # Decoder blocks return a tuple; the hidden state is the first element.
        # .clone() because .detach() alone shares storage and the buffer may be
        # reused before we move it to CPU.
        h = output[0] if isinstance(output, tuple) else output
        self._captured = h.detach().clone()

    @torch.no_grad()
    def extract(self, texts: list[str]) -> list[ExtractionResult]:
        layers = self.model.model.layers
        assert 0 <= self.layer_index < len(layers), (
            f"layer_index={self.layer_index} out of range for {len(layers)} layers"
        )
        handle = layers[self.layer_index].register_forward_hook(self._hook)
        try:
            out: list[ExtractionResult] = []
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                enc = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_context_tokens,
                )
                enc = {k: v.to(self.model.device) for k, v in enc.items()}
                self._captured = None
                self.model(**enc)
                assert self._captured is not None, "forward hook did not fire"
                hidden = self._captured.float().cpu()
                lengths = enc["attention_mask"].sum(dim=1).tolist()
                ids = enc["input_ids"].cpu().tolist()
                for row, seq_len in enumerate(lengths):
                    out.append(
                        ExtractionResult(
                            hidden_states=hidden[row, :seq_len],
                            token_ids=ids[row][:seq_len],
                        )
                    )
            return out
        finally:
            # Without this a mid-batch exception leaks the hook and the next
            # call double-registers, capturing the wrong layer.
            handle.remove()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="output base.parquet path")
    parser.add_argument("--n-documents", type=int, default=None, help="override doc count")
    parser.add_argument("--corpus-start", type=int, default=None)
    parser.add_argument("--positions-per-doc", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8, help="documents per forward")
    parser.add_argument("--chunk-size", type=int, default=256, help="documents per parquet write")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--corpus", default=None, help="override the corpus id")
    parser.add_argument("--corpus-config", default=None)
    parser.add_argument("--corpus-split", default=None)
    parser.add_argument("--text-column", default=None)
    parser.add_argument(
        "--corpus-kind",
        default="plain",
        choices=["plain", "chat"],
        help="'plain' reads a text column; 'chat' renders a conversation list "
        "through the target model's chat template (WildChat)",
    )
    parser.add_argument(
        "--source-tag", default=None, help="value for the `source` column (default: corpus id)"
    )
    args = parser.parse_args()

    load_project_env()
    data_cfg = WarmStartDataConfig()
    nla_cfg = NLAConfig()
    corpus = args.corpus or data_cfg.corpus
    corpus_config = args.corpus_config if args.corpus else data_cfg.corpus_config
    corpus_split = args.corpus_split or data_cfg.corpus_split
    text_column = args.text_column or data_cfg.text_column
    source_tag = args.source_tag or corpus
    n_documents = args.n_documents or data_cfg.n_documents
    corpus_start = data_cfg.corpus_start if args.corpus_start is None else args.corpus_start
    positions_per_doc = args.positions_per_doc or data_cfg.positions_per_doc
    seed = data_cfg.seed if args.seed is None else args.seed

    extractor = Extractor(
        model_id=nla_cfg.model_id,
        layer_index=nla_cfg.extraction_layer,
        max_context_tokens=data_cfg.max_context_tokens,
        batch_size=args.batch_size,
    )
    assert extractor.d_model == D_MODEL, (
        f"model reports d_model={extractor.d_model}, config says {D_MODEL}"
    )
    schema = build_schema(extractor.d_model)
    special_ids = set(extractor.tokenizer.all_special_ids)

    dataset = load_dataset(corpus, name=corpus_config, split=corpus_split, streaming=True)
    # IterableDataset has no .select(): skip/take bound the slice, and the docs
    # arrive as an iterator we consume in chunks.
    if corpus_start:
        dataset = dataset.skip(corpus_start)
    dataset = dataset.take(n_documents)

    def chunked(stream: Iterator[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
        while batch := list(itertools.islice(stream, size)):
            yield batch

    row_count = 0
    n_skipped = 0
    n_short = 0
    n_docs_seen = 0
    n_chunks = -(-n_documents // args.chunk_size)  # ceil, for the progress bar
    with pq.ParquetWriter(args.output, schema) as writer:
        for chunk in tqdm(chunked(iter(dataset), args.chunk_size), total=n_chunks, desc="chunks"):
            texts = [
                document_text(doc, text_column, args.corpus_kind, extractor.tokenizer)
                for doc in chunk
            ]
            # A chat row with no usable turns renders empty; drop it before the
            # forward rather than feeding the model a zero-length sequence.
            keep = [i for i, t in enumerate(texts) if t.strip()]
            n_skipped += len(texts) - len(keep)
            if not keep:
                n_docs_seen += len(chunk)
                continue
            results = extractor.extract([texts[i] for i in keep])

            rows: dict[str, list[Any]] = {name: [] for name in schema.names}
            for result_index, res in enumerate(results):
                doc_index = corpus_start + n_docs_seen + keep[result_index]
                doc_id = f"{corpus}:{corpus_split}:{doc_index}"
                positions = sample_positions(
                    res.token_ids,
                    positions_per_doc,
                    special_ids,
                    doc_id,
                    seed,
                    data_cfg.min_position,
                )
                if not positions:
                    n_skipped += 1
                    continue
                if len(positions) < positions_per_doc:
                    n_short += 1
                for pos in positions:
                    n_raw_tokens = pos + 1
                    rows["doc_id"].append(doc_id)
                    rows["n_raw_tokens"].append(n_raw_tokens)
                    rows["context_text"].append(
                        extractor.tokenizer.decode(
                            res.token_ids[:n_raw_tokens], skip_special_tokens=True
                        )
                    )
                    # Raw — normalization is training-side.
                    rows["activation_vector"].append(res.hidden_states[pos].tolist())
                    rows["activation_layer"].append(nla_cfg.extraction_layer)
                    rows["source"].append(source_tag)

            n_docs_seen += len(chunk)
            if rows["doc_id"]:
                writer.write_table(pa.Table.from_pydict(rows, schema=schema))
                row_count += len(rows["doc_id"])

    meta = DatasetMeta(
        dataset_id=(
            f"base_L{nla_cfg.extraction_layer}_{source_tag.replace('/', '-')}"
            f"_{corpus_start}_{n_documents}"
        ),
        stage="base",
        row_count=row_count,
        n_documents=n_documents,
        extraction=ExtractionMeta(
            base_model=nla_cfg.model_id,
            d_model=extractor.d_model,
            layer_index=nla_cfg.extraction_layer,
            hidden_states_index=nla_cfg.hidden_states_index,
            corpus=corpus,
            corpus_config=corpus_config,
            corpus_split=corpus_split,
            corpus_start=corpus_start,
            n_documents=n_documents,
            positions_per_doc=positions_per_doc,
            max_context_tokens=data_cfg.max_context_tokens,
            min_position=data_cfg.min_position,
        ),
        created_by="reasoning_attention.datagen.extract",
    )
    print(f"wrote {row_count} rows -> {args.output}")
    print(f"  skipped {n_skipped} docs (no valid position past {data_cfg.min_position})")
    print(f"  short-sampled {n_short} docs (< {positions_per_doc} valid positions)")
    print(f"sidecar -> {write_sidecar(args.output, meta)}")


if __name__ == "__main__":
    main()
