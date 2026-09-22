"""Smoke test: load the GPT endpoint and write one real explanation.

Verifies the whole explainer path before any of the 500k-pair pipeline runs on
top of it: the key loads from `.env`, `gpt-5.6-luna` answers at high reasoning
effort, the `<analysis>` tags survive, and the cleanup produces >= 2 features.

Usage:
    uv run python scripts/smoke_gpt_endpoint.py
    uv run python scripts/smoke_gpt_endpoint.py --list-models
    uv run python scripts/smoke_gpt_endpoint.py --snippet "Some prefix text..."
    uv run python scripts/smoke_gpt_endpoint.py --batch 4     # exercise concurrency
"""

from __future__ import annotations

import argparse

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
from reasoning_attention.datagen.providers import OpenAIProvider, load_api_key

console = Console(width=120)

_DEFAULT_SNIPPET = (
    "An ST-elevation myocardial infarction (STEMI) is a type of heart attack that mainly "
    "affects your heart's lower chambers. They are named for how they change the appearance "
    "of your heart's electrical activity on a test called an electrocardiogram. STEMIs are "
    "the most severe form of heart attack, and they happen when"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snippet", default=_DEFAULT_SNIPPET, help="text to explain")
    parser.add_argument("--model", default=None, help="override the configured model id")
    parser.add_argument("--effort", default=None, help="override reasoning effort")
    parser.add_argument(
        "--batch", type=int, default=1, help="send N copies to exercise concurrency"
    )
    parser.add_argument(
        "--list-models", action="store_true", help="list available model ids and exit"
    )
    args = parser.parse_args()

    config = ExplainerConfig()
    if args.model:
        config = ExplainerConfig(
            model=args.model,
            reasoning_effort=args.effort or config.reasoning_effort,
        )
    elif args.effort:
        config = ExplainerConfig(reasoning_effort=args.effort)

    if args.list_models:
        import openai

        load_api_key(config.env_file)
        ids = sorted(m.id for m in openai.OpenAI().models.list())
        console.print(f"[bold]{len(ids)} models[/bold]")
        console.print(", ".join(i for i in ids if i.startswith("gpt-5")))
        console.print(
            f"\nconfigured model {config.model!r} available: "
            f"{'[green]yes[/green]' if config.model in ids else '[red]NO[/red]'}"
        )
        return

    table = Table(show_header=False, box=None)
    table.add_row("model", config.model)
    table.add_row("reasoning effort", config.reasoning_effort)
    table.add_row("max output tokens", str(config.max_output_tokens))
    table.add_row("concurrency", str(config.concurrency))
    table.add_row("env file", str(config.env_file))
    table.add_row("prompts", str(args.batch))
    console.print(Panel(table, title="explainer config", expand=False))

    provider = OpenAIProvider(config)
    prompts = [build_explain_prompt(args.snippet)] * args.batch
    console.print(f"[dim]calling {config.model}...[/dim]")
    completions = provider.complete(prompts)

    n_ok = 0
    for i, raw in enumerate(completions):
        if raw is None:
            console.print(f"[red]prompt {i}: no completion (truncated or retries exhausted)[/red]")
            continue
        cleaned = extract_and_clean(raw)
        if cleaned is None:
            console.print(f"[red]prompt {i}: <analysis> tags missing — row would be dropped[/red]")
            console.print(Panel(raw, title=f"raw response {i}", border_style="red"))
            continue
        n_features = count_features(cleaned)
        ok = n_features >= MIN_FEATURES
        n_ok += ok
        console.print(
            Panel(
                cleaned,
                title=(
                    f"explanation {i} — {n_features} features, {len(cleaned.split())} words "
                    f"{'[green]OK[/green]' if ok else f'[red]< {MIN_FEATURES} — dropped[/red]'}"
                ),
                border_style="green" if ok else "red",
            )
        )

    console.print(f"\n[bold]{n_ok}/{len(prompts)} usable[/bold]")


if __name__ == "__main__":
    main()
