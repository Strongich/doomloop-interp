"""Write down what the labeling model produces, in a form you can actually read.

The explain stage stores summaries in parquet next to 2048-float vectors, which
is the right format for training and a useless one for judging label quality.
This script emits the same content three ways:

  - `<out>.jsonl`  one record per row — context, summary, feature count, norms.
  - `<out>.md`     the same pairs formatted for reading, longest context last.
  - `<out>.stats.json`  aggregate counts: features per summary, word counts,
                        drop reasons, `h_l` norm distribution.

Two modes:

  --from-parquet PATH   read an already-explained parquet (no API calls, free).
  --from-base PATH      call the labeling model on rows of a base/half parquet,
                        write the results down, and do NOT persist a training
                        parquet — this is for inspecting labels before spending
                        money on all 500k of them.

Usage:
    uv run python scripts/dump_explanations.py --from-base out/halves/av_half.parquet --limit 20
    uv run python scripts/dump_explanations.py --from-parquet out/av_explained.parquet
    uv run python scripts/dump_explanations.py --from-base out/base.parquet --limit 8 --out /tmp/labels
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from reasoning_attention.config import ExplainerConfig
from reasoning_attention.datagen.prompts import (
    MIN_FEATURES,
    build_explain_prompt,
    count_features,
    extract_and_clean,
)
from reasoning_attention.datagen.providers import OpenAIProvider

console = Console(width=120)


def label_rows(texts: list[str], config: ExplainerConfig) -> list[dict[str, Any]]:
    """Call the labeling model on `texts`; one record per input, drops included.

    Records carry `status` so the dump shows what was rejected and why, not just
    the survivors — a 30% drop rate is something you want to see before the full
    run, and it is invisible if failures are silently filtered out.
    """
    provider = OpenAIProvider(config)
    completions = provider.complete([build_explain_prompt(t) for t in texts])
    records: list[dict[str, Any]] = []
    for text, raw in zip(texts, completions, strict=True):
        if raw is None:
            records.append({"context": text, "summary": None, "status": "no_completion"})
            continue
        cleaned = extract_and_clean(raw)
        if cleaned is None:
            records.append(
                {"context": text, "summary": None, "status": "no_tags", "raw": raw}
            )
            continue
        n_features = count_features(cleaned)
        records.append(
            {
                "context": text,
                "summary": cleaned,
                "n_features": n_features,
                "status": "ok" if n_features >= MIN_FEATURES else "too_few_features",
            }
        )
    return records


def summarize(records: list[dict[str, Any]], norms: list[float] | None) -> dict[str, Any]:
    """Aggregate stats over the dumped records."""
    ok = [r for r in records if r["status"] == "ok"]
    words = [len(r["summary"].split()) for r in ok]
    stats: dict[str, Any] = {
        "n_rows": len(records),
        "n_usable": len(ok),
        "status_counts": dict(Counter(r["status"] for r in records)),
        "features_per_summary": dict(Counter(r["n_features"] for r in ok)),
        "summary_words": {
            "min": min(words) if words else 0,
            "max": max(words) if words else 0,
            "mean": round(sum(words) / len(words), 1) if words else 0.0,
        },
        "context_words": {
            "min": min(len(r["context"].split()) for r in records) if records else 0,
            "max": max(len(r["context"].split()) for r in records) if records else 0,
        },
    }
    if norms:
        arr = np.asarray(norms, dtype=np.float64)
        stats["h_l_norm"] = {
            "min": round(float(arr.min()), 1),
            "max": round(float(arr.max()), 1),
            "mean": round(float(arr.mean()), 1),
        }
    return stats


def write_markdown(path: Path, records: list[dict[str, Any]], stats: dict[str, Any]) -> None:
    """Readable side-by-side dump, shortest context first."""
    lines = [
        "# Labeling-model explanations",
        "",
        f"`{stats['n_usable']}/{stats['n_rows']}` usable "
        f"(needs >= {MIN_FEATURES} features).",
        "",
    ]
    ordered = sorted(records, key=lambda r: len(r["context"]))
    for i, rec in enumerate(ordered):
        status = rec["status"]
        lines.append(f"## {i}. {status}" + (f" — {rec.get('n_features')} features" if rec.get("n_features") else ""))
        lines.append("")
        lines.append("**Context (tail):**")
        lines.append("")
        lines.append("> " + rec["context"][-600:].replace("\n", " ").strip())
        lines.append("")
        if rec["summary"]:
            lines.append("**Summary:**")
            lines.append("")
            for feature in rec["summary"].split("\n\n"):
                lines.append(f"- {feature}")
        else:
            lines.append(f"**No summary** ({status}).")
            if rec.get("raw"):
                lines.append("")
                lines.append("```")
                lines.append(rec["raw"][:800])
                lines.append("```")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--from-parquet", help="already-explained parquet (no API calls)")
    source.add_argument("--from-base", help="base/half parquet — calls the labeling model")
    parser.add_argument("--limit", type=int, default=20, help="rows to dump")
    parser.add_argument("--offset", type=int, default=0, help="row offset")
    parser.add_argument("--out", default="explanations", help="output path stem")
    args = parser.parse_args()

    path = args.from_parquet or args.from_base
    table = pq.read_table(path).slice(args.offset, args.limit)
    contexts = table.column("context_text").to_pylist()

    norms: list[float] | None = None
    if "activation_vector" in table.column_names:
        vectors = np.asarray(
            table.column("activation_vector").combine_chunks().values.to_numpy()
        ).reshape(table.num_rows, -1)
        norms = [float(n) for n in np.linalg.norm(vectors, axis=1)]

    if args.from_parquet:
        assert "summary" in table.column_names, (
            f"{path} has no `summary` column — pass --from-base to label it instead"
        )
        records = [
            {
                "context": c,
                "summary": s,
                "n_features": count_features(s),
                "status": "ok" if count_features(s) >= MIN_FEATURES else "too_few_features",
            }
            for c, s in zip(contexts, table.column("summary").to_pylist(), strict=True)
        ]
    else:
        config = ExplainerConfig()
        console.print(
            f"[dim]labeling {len(contexts)} rows with {config.model} "
            f"(effort={config.reasoning_effort})...[/dim]"
        )
        records = label_rows(contexts, config)

    if norms:
        for rec, norm in zip(records, norms, strict=True):
            rec["h_l_norm"] = round(norm, 1)

    stats = summarize(records, norms)
    stem = Path(args.out)
    stem.parent.mkdir(parents=True, exist_ok=True)

    jsonl_path = stem.with_suffix(".jsonl")
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    md_path = stem.with_suffix(".md")
    write_markdown(md_path, records, stats)
    stats_path = Path(f"{stem}.stats.json")
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    table_view = Table(show_header=False, box=None)
    table_view.add_row("rows", str(stats["n_rows"]))
    table_view.add_row("usable", f"{stats['n_usable']}/{stats['n_rows']}")
    table_view.add_row("status", json.dumps(stats["status_counts"]))
    table_view.add_row("features/summary", json.dumps(stats["features_per_summary"]))
    table_view.add_row("summary words", json.dumps(stats["summary_words"]))
    if "h_l_norm" in stats:
        table_view.add_row("h_l norm", json.dumps(stats["h_l_norm"]))
    console.print(Panel(table_view, title="label dump", expand=False))

    for rec in records[:2]:
        if rec["summary"]:
            console.print(
                Panel(
                    rec["summary"],
                    title=f"{rec['n_features']} features · context tail: "
                    f"...{rec['context'][-60:]!r}",
                    border_style="green",
                )
            )

    console.print(f"wrote [bold]{jsonl_path}[/bold]")
    console.print(f"wrote [bold]{md_path}[/bold]")
    console.print(f"wrote [bold]{stats_path}[/bold]")


if __name__ == "__main__":
    main()
