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
import os
import random
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from reasoning_attention.config import D_MODEL, NLAConfig, WarmStartDataConfig, load_project_env
from reasoning_attention.datagen.sidecar import DatasetMeta, ExtractionMeta, write_sidecar

from ..nla.arch import inner_transformer


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


# open-r1's traces were produced by an R1-family model and carry its markers.
# `<|begin_of_thought|>` maps onto Qwen3's `<think>`, but the solution markers
# have NO Qwen equivalent — Qwen3 writes the answer plainly after `</think>` —
# so they are deleted rather than mapped. Mapping them would teach the NLA to
# explain a token our target model never emits.
R1_MARKERS = {
    "<|begin_of_thought|>": "<think>",
    "<|end_of_thought|>": "</think>",
    "<|begin_of_solution|>": "",
    "<|end_of_solution|>": "",
}


def rewrite_r1_markers(text: str) -> str:
    for src, dst in R1_MARKERS.items():
        text = text.replace(src, dst)
    return text.strip()


def render_reasoning(conversation: Any, tokenizer: Any) -> str:
    """open-r1 `conversations` ([{from, value}, ...]) as Qwen3 sees a trace.

    The dataset's own `system` field is an R1-style prompt and is deliberately
    dropped: our rollouts were generated with no system turn, so including one
    would put the training activations in a context the study never probes.
    """
    turns = [t for t in (conversation or []) if t.get("value")]
    user = next((t["value"] for t in turns if t.get("from") == "user"), None)
    assistant = next((t["value"] for t in turns if t.get("from") != "user"), None)
    if not user or not assistant:
        return ""
    header: str = tokenizer.apply_chat_template(
        [{"role": "user", "content": str(user)}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return header + rewrite_r1_markers(str(assistant))


def render_trace(row: dict[str, Any], tokenizer: Any) -> str:
    """One of our own rollouts, rebuilt exactly as the model emitted it.

    Same construction as `scripts/onset_verbalize.py` and the study's probes:
    chat template with `add_generation_prompt=True`, then the raw response. The
    template does not emit `<think>` — the model does.
    """
    question, response = row.get("question"), row.get("response")
    if not question or not response:
        return ""
    header: str = tokenizer.apply_chat_template(
        [{"role": "user", "content": str(question)}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return header + str(response)


def document_text(row: dict[str, Any], column: str, kind: str, tokenizer: Any) -> str:
    """Extract one document's text, per corpus kind."""
    if kind == "chat":
        return render_chat(row[column], tokenizer)
    if kind == "reasoning":
        return render_reasoning(row[column], tokenizer)
    if kind == "trace":
        return render_trace(row, tokenizer)
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


# Byte-level BPE writes "\n" as "Ċ", so a token whose piece contains "ĊĊ" spans a
# blank line. Checked against the vocab rather than by decoding every token,
# which would cost a string op per token per document.
_BLANK_LINE_PIECE = "ĊĊ"


def block_boundary_positions(token_ids: list[int], tokenizer: Any, min_position: int) -> list[int]:
    """Positions whose token spans a blank line — the end of a reasoning block.

    Qwen3 merges the preceding punctuation into the break, so ".\n\n" is ONE
    token: the block's last token and the break are the same position. That is
    the same position `tokenview.block_boundaries` probes, so training and
    measurement land on the same states.
    """
    pieces = tokenizer.convert_ids_to_tokens(token_ids)
    return [i for i, piece in enumerate(pieces) if i >= min_position and _BLANK_LINE_PIECE in piece]


class Extractor:
    """Runs the target model and captures one layer's output via a forward hook."""

    def __init__(
        self,
        model_id: str,
        layer_index: int,
        max_context_tokens: int,
        batch_size: int = 8,
        token_budget: int | None = None,
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
        # Forward the INNER transformer, never the causal-LM wrapper: the wrapper
        # runs lm_head over every position, and at the 32k context the reasoning
        # corpora need that is a batch x 32768 x 151936 logit tensor (~79 GB at
        # batch 8) thrown away immediately. The layer hook fires either way.
        self.trunk: Any = inner_transformer(self.model)
        self.layer_index = layer_index
        self.max_context_tokens = max_context_tokens
        self.batch_size = batch_size
        self.token_budget = token_budget
        self.d_model = int(self.model.config.hidden_size)
        self._captured: torch.Tensor | None = None

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> None:
        # Decoder blocks return a tuple; the hidden state is the first element.
        # .clone() because .detach() alone shares storage and the buffer may be
        # reused before we move it to CPU.
        h = output[0] if isinstance(output, tuple) else output
        self._captured = h.detach().clone()

    def _batches(self, texts: list[str]) -> list[list[int]]:
        """Group document indices into forward passes.

        A fixed document count starves the GPU on long corpora and overruns it on
        short ones: open-r1 averages ~7.1k tokens against Ultra-FineWeb's ~180, so
        `--batch-size 2` there ran ~14k tokens per forward at ~37 TFLOPS on a card
        that does 150-200. Packing to a TOKEN budget keeps every forward the same
        size in the thing that actually costs.

        Sorting by length first is the other half: a batch pads to its longest
        member, so pairing a 28k document with a 2k one wastes 26k positions.
        Descending order means the first index in a batch is its longest, so the
        padded cost is exactly `len(batch) * lengths[batch[0]]`.

        `batch_size` stays a hard cap on the count so a run of tiny documents
        cannot assemble a batch of thousands.
        """
        if self.token_budget is None:
            return [
                list(range(i, min(i + self.batch_size, len(texts))))
                for i in range(0, len(texts), self.batch_size)
            ]
        encoded = self.tokenizer(texts, add_special_tokens=True)["input_ids"]
        lengths = [min(len(ids), self.max_context_tokens) for ids in encoded]
        order = sorted(range(len(texts)), key=lambda i: lengths[i], reverse=True)

        batches: list[list[int]] = []
        batch: list[int] = []
        for i in order:
            padded = (len(batch) + 1) * lengths[batch[0]] if batch else lengths[i]
            if batch and (padded > self.token_budget or len(batch) >= self.batch_size):
                batches.append(batch)
                batch = []
            batch.append(i)
        if batch:
            batches.append(batch)
        return batches

    @torch.no_grad()
    def extract(self, texts: list[str]) -> list[ExtractionResult]:
        layers = self.model.model.layers
        assert 0 <= self.layer_index < len(layers), (
            f"layer_index={self.layer_index} out of range for {len(layers)} layers"
        )
        handle = layers[self.layer_index].register_forward_hook(self._hook)
        try:
            # Indexed, not appended: length-sorted batches come back out of order
            # and the caller maps results to documents positionally.
            out: list[ExtractionResult | None] = [None] * len(texts)
            for index_batch in self._batches(texts):
                enc = self.tokenizer(
                    [texts[i] for i in index_batch],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_context_tokens,
                )
                enc = {k: v.to(self.model.device) for k, v in enc.items()}
                self._captured = None
                self.trunk(**enc)
                assert self._captured is not None, "forward hook did not fire"
                hidden = self._captured.float().cpu()
                lengths = enc["attention_mask"].sum(dim=1).tolist()
                ids = enc["input_ids"].cpu().tolist()
                for row, seq_len in enumerate(lengths):
                    out[index_batch[row]] = ExtractionResult(
                        hidden_states=hidden[row, :seq_len],
                        token_ids=ids[row][:seq_len],
                    )
            assert all(r is not None for r in out), "a document produced no result"
            return cast(list[ExtractionResult], out)
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
        choices=["plain", "chat", "reasoning", "trace"],
        help="'plain' reads a text column; 'chat' renders a conversation list "
        "through the chat template (WildChat); 'reasoning' does the same for "
        "open-r1 conversations with R1 markers rewritten; 'trace' rebuilds one "
        "of our own rollouts from its question + response",
    )
    parser.add_argument(
        "--source-tag", default=None, help="value for the `source` column (default: corpus id)"
    )
    parser.add_argument(
        "--corpus-file",
        default=None,
        help="read documents from a local jsonl instead of the Hub",
    )
    parser.add_argument(
        "--position-mode",
        default="random",
        choices=["random", "block"],
        help="'random' samples positions uniformly (the reference recipe); "
        "'block' samples only block boundaries, the positions this study probes "
        "and where reconstruction is measurably worst",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=None,
        help="pack each forward to this many padded tokens instead of a fixed count",
    )
    parser.add_argument(
        "--max-document-tokens",
        type=int,
        default=None,
        help="skip documents longer than this (measured, not truncated)",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=None,
        help="truncate documents to this length (default: DataGenConfig)",
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
    # A local corpus is not the Hub default: naming it "openbmb/Ultra-FineWeb"
    # would collide doc_ids with the web half of the same merged corpus.
    doc_namespace = f"file:{Path(args.corpus_file).name}" if args.corpus_file else corpus
    doc_split = "-" if args.corpus_file else corpus_split
    n_documents = args.n_documents or data_cfg.n_documents
    max_context_tokens = args.max_context_tokens or data_cfg.max_context_tokens
    corpus_start = data_cfg.corpus_start if args.corpus_start is None else args.corpus_start
    positions_per_doc = args.positions_per_doc or data_cfg.positions_per_doc
    seed = data_cfg.seed if args.seed is None else args.seed

    extractor = Extractor(
        model_id=nla_cfg.model_id,
        layer_index=nla_cfg.extraction_layer,
        max_context_tokens=max_context_tokens,
        batch_size=args.batch_size,
        token_budget=args.token_budget,
    )
    assert extractor.d_model == D_MODEL, (
        f"model reports d_model={extractor.d_model}, config says {D_MODEL}"
    )
    schema = build_schema(extractor.d_model)
    special_ids = set(extractor.tokenizer.all_special_ids)

    if args.corpus_file:
        # Our own rollouts live in a local jsonl, not on the Hub. Streaming a
        # file keeps the rest of the loop identical.
        dataset = load_dataset("json", data_files=args.corpus_file, split="train", streaming=True)
    else:
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
    n_too_long = 0
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
            if args.max_document_tokens is not None:
                # DROP the over-long ones rather than truncating them. Truncation
                # would keep the document at a length the batch must still be
                # sized for, and would sample positions from a prefix that is not
                # the document the rest of the corpus represents. Measured on the
                # raw text, so the count is the same one the percentiles report.
                measured = extractor.tokenizer([texts[i] for i in keep], add_special_tokens=True)[
                    "input_ids"
                ]
                kept = [
                    i
                    for i, ids in zip(keep, measured, strict=True)
                    if len(ids) <= args.max_document_tokens
                ]
                n_too_long += len(keep) - len(kept)
                keep = kept
            if not keep:
                n_docs_seen += len(chunk)
                continue
            results = extractor.extract([texts[i] for i in keep])

            rows: dict[str, list[Any]] = {name: [] for name in schema.names}
            for result_index, res in enumerate(results):
                doc_index = corpus_start + n_docs_seen + keep[result_index]
                doc_id = f"{doc_namespace}:{doc_split}:{doc_index}"
                if args.position_mode == "block":
                    cands = block_boundary_positions(
                        res.token_ids, extractor.tokenizer, data_cfg.min_position
                    )
                    # Deterministic in (seed, doc_id) like the random sampler, so
                    # a re-slice of the corpus yields the same positions.
                    rng = random.Random(hashlib.sha256(f"{seed}|{doc_id}|block".encode()).digest())
                    positions = (
                        rng.sample(cands, k=min(positions_per_doc, len(cands))) if cands else []
                    )
                else:
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
            corpus=doc_namespace,
            corpus_config=corpus_config,
            corpus_split=corpus_split,
            corpus_start=corpus_start,
            n_documents=n_documents,
            positions_per_doc=positions_per_doc,
            max_context_tokens=max_context_tokens,
            min_position=data_cfg.min_position,
        ),
        created_by="reasoning_attention.datagen.extract",
    )
    print(f"wrote {row_count} rows -> {args.output}")
    print(f"  skipped {n_skipped} docs (no valid position past {data_cfg.min_position})")
    print(f"  short-sampled {n_short} docs (< {positions_per_doc} valid positions)")
    if args.max_document_tokens is not None:
        print(f"  dropped {n_too_long} docs over {args.max_document_tokens} tokens")
    print(f"sidecar -> {write_sidecar(args.output, meta)}")


if __name__ == "__main__":
    main()
    # Hard exit rather than returning into interpreter finalization.
    #
    # The streaming dataset's aiohttp worker threads do not shut down cleanly:
    # finalization either raises `PyGILState_Release: ... no thread-state for
    # this thread` or, worse, HANGS — the process sits holding the GPU with the
    # parquet already written, so a chained build waits on it forever. Everything
    # is flushed by this point (the ParquetWriter closed with its `with` block
    # and the sidecar is on disk), so there is nothing for finalization to do.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
