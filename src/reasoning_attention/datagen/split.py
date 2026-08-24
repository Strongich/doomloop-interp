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


def partition_documents(
    doc_ids: list[str], av_fraction: float, seed: int
) -> tuple[set[str], set[str]]:
    """Split unique document ids into (av_docs, ar_docs).

    `sorted()` before the shuffle is load-bearing: set iteration order varies
    with the hash seed, so without it the same `--seed` would produce different
    splits across runs and environments.
    """
    unique = sorted(set(doc_ids))
    random.Random(seed).shuffle(unique)
    n_av = int(len(unique) * av_fraction)
    return set(unique[:n_av]), set(unique[n_av:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="base.parquet from the extract stage")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--av-fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    load_project_env()
    data_cfg = WarmStartDataConfig()
    av_fraction = data_cfg.av_fraction if args.av_fraction is None else args.av_fraction
    seed = data_cfg.seed if args.seed is None else args.seed
    assert 0.0 < av_fraction < 1.0, f"av_fraction must be in (0, 1), got {av_fraction}"

    base_meta = read_sidecar(args.base)
    assert base_meta["stage"] == "base", f"expected stage=base, got {base_meta['stage']!r}"

    parquet = pq.ParquetFile(args.base)
    doc_ids = parquet.read(columns=["doc_id"]).column("doc_id").to_pylist()
    av_docs, ar_docs = partition_documents(doc_ids, av_fraction, seed)
    assert not (av_docs & ar_docs), "halves overlap — partition is broken"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    buckets = {"av_half": av_docs, "ar_half": ar_docs}
    paths = {name: str(out_dir / f"{name}.parquet") for name in buckets}

    schema = parquet.schema_arrow
    writers = {name: pq.ParquetWriter(paths[name], schema) for name in buckets}
    row_counts = {name: 0 for name in buckets}
    try:
        for batch in parquet.iter_batches(batch_size=_BATCH_ROWS):
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

    extraction = ExtractionMeta(**base_meta["extraction"])
    for name, bucket in buckets.items():
        meta = DatasetMeta(
            dataset_id=f"{base_meta['dataset_id']}__{name}",
            stage=name,
            row_count=row_counts[name],
            n_documents=len(bucket),
            extraction=extraction,
            created_by="reasoning_attention.datagen.split",
            parent_datasets=[base_meta["dataset_id"]],
        )
        write_sidecar(paths[name], meta)
        print(f"{name}: {len(bucket)} docs -> {row_counts[name]} rows -> {paths[name]}")


if __name__ == "__main__":
    main()
