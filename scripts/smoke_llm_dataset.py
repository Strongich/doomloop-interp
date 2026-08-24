"""Temporary smoke test: one dataset record -> transformers model -> raw response.

Loads a single record from one of the prepared math datasets, runs it through
the plain-transformers model (`model.loader.load_model`), and prints the
generated text WITHOUT skipping special tokens — so you can see Qwen3's
<think>...</think> block and any special tokens verbatim.

Sampling params come from `config.SamplingDefaults` (the LLM sampling config).

Usage:
    uv run python scripts/smoke_llm_dataset.py
    uv run python scripts/smoke_llm_dataset.py --dataset amc23 --index 3
    uv run python scripts/smoke_llm_dataset.py --dataset aime2025 --no-thinking
"""

from __future__ import annotations

import argparse

import torch
from rich.console import Console
from rich.panel import Panel

from reasoning_attention.config import SamplingDefaults
from reasoning_attention.data.math_datasets import load_one, render_prompt
from reasoning_attention.model.loader import load_model

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="gsm8k",
        choices=["gsm8k", "aime2025", "amc23"],
        help="Which prepared dataset to pull a record from.",
    )
    parser.add_argument("--index", type=int, default=0, help="Record index to run.")
    parser.add_argument(
        "--no-thinking", action="store_true", help="Disable Qwen3 thinking mode."
    )
    args = parser.parse_args()

    # --- one record ---
    ds = load_one(args.dataset)
    record = ds[args.index]
    question = record["question"]
    gold = record["answer"]

    console.print(
        Panel(
            f"[bold]source[/]: {record['source']} / {record['subset']} / {record['split']}\n"
            f"[bold]question[/]: {question}\n"
            f"[bold green]gold answer[/]: {gold}",
            title=f"{args.dataset}[{args.index}]",
        )
    )

    # --- run through the transformers model ---
    loaded = load_model()
    model, tokenizer = loaded.model, loaded.tokenizer

    prompt = render_prompt(tokenizer, question, enable_thinking=not args.no_thinking)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    s = SamplingDefaults()  # LLM sampling params, from config
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            do_sample=True,
            temperature=s.temperature,
            top_p=s.top_p,
            top_k=s.top_k,
            max_new_tokens=s.max_tokens,
        )
    # Only the newly generated continuation (drop the prompt tokens).
    new_tokens = generated[0, inputs["input_ids"].shape[1]:]
    # skip_special_tokens=False so <think>…</think> and any special tokens show.
    text = tokenizer.decode(new_tokens, skip_special_tokens=False)

    console.rule("[bold]raw model output (special tokens NOT skipped)[/]")
    print(text)


if __name__ == "__main__":
    main()
