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

from reasoning_attention.config import load_project_env
from reasoning_attention.datagen.sidecar import (
    DatasetMeta,
    ExtractionMeta,
    read_sidecar,
    write_sidecar,
)


def _widen_strings(table: pa.Table) -> pa.Table:
    """Cast `string`/`binary` columns to their 64-bit-offset variants.

    Arrow's `string` addresses its data with int32 offsets, so one array caps at
    2 GB. `context_text` holds whole source documents, and concatenating ~1.1M of
    them overflows that — `pa.concat_tables` then dies with "offset overflow while
    concatenating arrays". Widening first is the fix Arrow's own error recommends,
    and it is free: the on-disk parquet encoding is unchanged either way.
    """
    fields = []
    for field in table.schema:
        if pa.types.is_string(field.type):
            fields.append(field.with_type(pa.large_string()))
        elif pa.types.is_binary(field.type):
            fields.append(field.with_type(pa.large_binary()))
        else:
            fields.append(field)
    target = pa.schema(fields)
    return table.cast(target) if target != table.schema else table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="base parquets to merge")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--block-size", type=int, default=50_000)
    args = parser.parse_args()

    load_project_env()
    tables = [_widen_strings(pq.read_table(path)) for path in args.inputs]
    schemas = {tuple(t.schema.names) for t in tables}
    assert len(schemas) == 1, (
        f"inputs have different schemas and cannot be concatenated: {schemas}. "
        f"Re-extract them with the same version of the extract stage."
    )
    merged = pa.concat_tables(tables)

    order = (
        np.arange(merged.num_rows)
        if args.no_shuffle
        else np.random.default_rng(args.seed).permutation(merged.num_rows)
    )
    # Take and write in blocks. A single take() over ~1.1M rows materialises every
    # column at once — with a 2048-float vector and the document text per row that
    # is tens of GB, on top of the tables already in memory. Blocks bound it.
    with pq.ParquetWriter(args.output, merged.schema) as writer:
        for start in range(0, len(order), args.block_size):
            block = merged.take(pa.array(order[start : start + args.block_size]))
            writer.write_table(block)

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
