"""Concatenate base parquets from different corpora into one RL prompt set.

The RL set is drawn from two sources (Ultra-FineWeb prose and WildChat dialogue),
extracted in separate passes because they need different text rendering. This
merges them and shuffles, so a training run does not see 200k web rows followed
by 200k chat rows — with a constant-LR RL schedule that ordering would be a
curriculum nobody asked for.

Row-level shuffling is correct here, unlike in the SFT split: there is no AV/AR
boundary to leak across, every row is an independent RL prompt, and positions
from one document carry no shared target.
"""

from __future__ import annotations

import argparse

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from reasoning_attention.datagen.sidecar import (
    DatasetMeta,
    ExtractionMeta,
    read_sidecar,
    write_sidecar,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="base parquets to merge")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--no-shuffle", action="store_true")
    args = parser.parse_args()

    tables = [pq.read_table(path) for path in args.inputs]
    schemas = {tuple(t.schema.names) for t in tables}
    assert len(schemas) == 1, (
        f"inputs have different schemas and cannot be concatenated: {schemas}. "
        f"Re-extract them with the same version of the extract stage."
    )
    merged = pa.concat_tables(tables)

    if not args.no_shuffle:
        order = np.random.default_rng(args.seed).permutation(merged.num_rows)
        merged = merged.take(pa.array(order))

    pq.write_table(merged, args.output)

    metas = [read_sidecar(path) for path in args.inputs]
    per_source = {}
    for path, table, source_meta in zip(args.inputs, tables, metas, strict=True):
        per_source[source_meta["extraction"]["corpus"]] = table.num_rows
        print(f"  {path}: {table.num_rows} rows ({source_meta['extraction']['corpus']})")

    # The merged set spans corpora, so the single `extraction` block can only
    # describe the first; per-source counts and ids go in prompt_templates-adjacent
    # provenance instead of pretending one corpus produced everything.
    merged_meta = DatasetMeta(
        dataset_id="rl_merged_" + "_".join(sorted(m["dataset_id"] for m in metas))[:80],
        stage="base",
        row_count=merged.num_rows,
        n_documents=sum(m["n_documents"] for m in metas),
        extraction=ExtractionMeta(**metas[0]["extraction"]),
        created_by="reasoning_attention.datagen.merge",
        parent_datasets=[m["dataset_id"] for m in metas],
        prompt_templates={f"rows:{k}": str(v) for k, v in per_source.items()},
    )
    print(f"merged {merged.num_rows} rows -> {args.output}")
    print(f"sidecar -> {write_sidecar(args.output, merged_meta)}")


if __name__ == "__main__":
    main()
