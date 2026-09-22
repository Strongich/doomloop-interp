"""Stage 1: base.parquet -> the disjoint AV and AR halves.

The paper's appendix splits the ~500k `(context, summary)` pairs **evenly by
document** between the AV and AR warm-start sets. Both words matter:

  - **Evenly** — ~250k pairs each.
  - **By document** — the partition is over `doc_id`, never over rows. Stage 0
    draws 5 positions from each document, so a row-level split would put
    position 2 of a document in the AV half and position 4 in the AR half. Those
    two contexts share a prefix, which leaks one model's training text into the
    other's, and the halves stop being disjoint in any meaningful sense.

The consequence the appendix spells out: if `(h_17, s_17)` lands in `D_AV`, the
AV learns `h_17 -> s_17` and the AR never sees that pair at all. The AR instead
learns `s_j -> h_j` for pairs from its own half.

Streams row-group by row-group: only `doc_id` is read to compute the partition,
because reading `activation_vector` for the whole table at once overflows
pyarrow's int32 list offsets at 500k x 2048.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from reasoning_attention.config import WarmStartDataConfig, load_project_env
from reasoning_attention.datagen.sidecar import (
    DatasetMeta,
    ExtractionMeta,
    read_sidecar,
    write_sidecar,
)

# Rows per streamed batch. 65536 x 2048 floats stays comfortably under the
# int32 offset limit regardless of how row groups were laid out on disk.
_BATCH_ROWS = 65536


def read_half_docs(halves_dir: str) -> dict[str, str]:
    """`doc_id -> "av_half" | "ar_half"` from an existing split, or {} if absent.

    Used to EXTEND a corpus without disturbing the documents already labelled.
    The assignment must be preserved rather than recomputed, because a document
    that moves halves invalidates the positional chunk cache in `explain`: the
    half parquets stop being a prefix of their extended selves and every
    downstream chunk gets relabelled at full API/GPU cost.
    """
    assignment: dict[str, str] = {}
    for name in ("av_half", "ar_half"):
        path = Path(halves_dir) / f"{name}.parquet"
        if not path.exists():
            continue
        for doc in pq.ParquetFile(path).read(columns=["doc_id"]).column("doc_id").to_pylist():
            assignment[str(doc)] = name
    return assignment


def partition_documents(
    doc_ids: list[str],
    av_fraction: float,
    seed: int,
    preassigned: dict[str, str] | None = None,
) -> tuple[set[str], set[str]]:
    """Split unique document ids into (av_docs, ar_docs).

    `sorted()` before the shuffle is load-bearing: set iteration order varies
    with the hash seed, so without it the same `--seed` would produce different
    splits across runs and environments.

    `preassigned` pins documents to the half they already occupy; only the
    remainder is shuffled, and it is dealt so the WHOLE corpus lands on
    `av_fraction` rather than just the new part. With no preassignment this is
    bit-identical to the original single-shot partition.
    """
    unique = sorted(set(doc_ids))
    preassigned = preassigned or {}
    av = {d for d in unique if preassigned.get(d) == "av_half"}
    ar = {d for d in unique if preassigned.get(d) == "ar_half"}
    fresh = [d for d in unique if d not in av and d not in ar]
    random.Random(seed).shuffle(fresh)
    n_av = max(0, min(len(fresh), int(len(unique) * av_fraction) - len(av)))
    return av | set(fresh[:n_av]), ar | set(fresh[n_av:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        required=True,
        nargs="+",
        help="base.parquet shard(s) from the extract stage. Multiple shards are "
        "streamed IN THE GIVEN ORDER, so list the pre-existing one first when "
        "extending — the halves stay prefixes of their previous selves and "
        "`explain` resumes instead of relabelling.",
    )
    parser.add_argument(
        "--preserve-halves",
        default=None,
        help="directory of an existing split whose doc->half assignment to keep",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--av-fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    load_project_env()
    data_cfg = WarmStartDataConfig()
    av_fraction = data_cfg.av_fraction if args.av_fraction is None else args.av_fraction
    seed = data_cfg.seed if args.seed is None else args.seed
    assert 0.0 < av_fraction < 1.0, f"av_fraction must be in (0, 1), got {av_fraction}"

    metas = [read_sidecar(path) for path in args.base]
    for path, meta in zip(args.base, metas, strict=True):
        assert meta["stage"] == "base", f"{path}: expected stage=base, got {meta['stage']!r}"
    # Everything about the extraction except WHICH slice of the corpus was read
    # must agree, or the shards hold activations from different models/layers.
    volatile = {"corpus_start", "n_documents"}
    for path, meta in zip(args.base[1:], metas[1:], strict=True):
        differing = {
            k
            for k, v in meta["extraction"].items()
            if k not in volatile and metas[0]["extraction"].get(k) != v
        }
        assert not differing, f"{path}: extraction differs from {args.base[0]} in {differing}"

    files = [pq.ParquetFile(path) for path in args.base]
    for path, handle in zip(args.base[1:], files[1:], strict=True):
        assert handle.schema_arrow.equals(files[0].schema_arrow), f"{path}: schema mismatch"

    doc_ids = [
        str(d)
        for handle in files
        for d in handle.read(columns=["doc_id"]).column("doc_id").to_pylist()
    ]
    preassigned = read_half_docs(args.preserve_halves) if args.preserve_halves else {}
    av_docs, ar_docs = partition_documents(doc_ids, av_fraction, seed, preassigned)
    assert not (av_docs & ar_docs), "halves overlap — partition is broken"
    if preassigned:
        kept = len(set(doc_ids) & set(preassigned))
        print(f"preserved {kept} existing doc assignments, dealt {len(set(doc_ids)) - kept} new")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    buckets = {"av_half": av_docs, "ar_half": ar_docs}
    paths = {name: str(out_dir / f"{name}.parquet") for name in buckets}

    schema = files[0].schema_arrow
    writers = {name: pq.ParquetWriter(paths[name], schema) for name in buckets}
    row_counts = {name: 0 for name in buckets}
    try:
        for handle in files:
            for batch in handle.iter_batches(batch_size=_BATCH_ROWS):
                batch_docs = batch.column("doc_id").to_pylist()
                for name, bucket in buckets.items():
                    mask = pa.array([d in bucket for d in batch_docs], type=pa.bool_())
                    subset = batch.filter(mask)
                    if subset.num_rows:
                        writers[name].write_table(pa.Table.from_batches([subset]))
                        row_counts[name] += subset.num_rows
    finally:
        for writer in writers.values():
            writer.close()

    total = sum(row_counts.values())
    assert total == len(doc_ids), (
        f"row accounting mismatch: {total} written vs {len(doc_ids)} read — rows were lost"
    )

    # One merged extraction record: same everything, widened corpus window.
    merged = dict(metas[0]["extraction"])
    merged["corpus_start"] = min(m["extraction"]["corpus_start"] for m in metas)
    merged["n_documents"] = sum(m["extraction"]["n_documents"] for m in metas)
    extraction = ExtractionMeta(**merged)
    for name, bucket in buckets.items():
        half_meta = DatasetMeta(
            dataset_id=f"{metas[0]['dataset_id']}__{name}",
            stage=name,
            row_count=row_counts[name],
            n_documents=len(bucket),
            extraction=extraction,
            created_by="reasoning_attention.datagen.split",
            parent_datasets=[m["dataset_id"] for m in metas],
        )
        write_sidecar(paths[name], half_meta)
        print(f"{name}: {len(bucket)} docs -> {row_counts[name]} rows -> {paths[name]}")


if __name__ == "__main__":
    main()
