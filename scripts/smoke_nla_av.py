"""Temporary smoke test: the AV (activation verbalizer) injection mechanism.

Builds the NLA models, extracts a real layer-l activation (`h_l`) from the final
token of a text snippet, injects it at the `<INJECT>` placeholder position of the
AV prompt, and generates the explanation. Also sanity-checks that the placeholder
resolves to exactly one token and reports the injection scale.

The AV is UNTRAINED, so the explanation will tend to echo the prompt framing
rather than truly decode the vector — this tests the *mechanism*, not quality.

Usage:
    uv run python scripts/smoke_nla_av.py
    uv run python scripts/smoke_nla_av.py --snippet "Paris is the capital of France."
    uv run python scripts/smoke_nla_av.py --max-new-tokens 120 --explanation-only
"""

from __future__ import annotations

import argparse

import torch
from rich.console import Console
from rich.panel import Panel

from reasoning_attention.nla import NLA, INJECT_PLACEHOLDER
from reasoning_attention.nla.prompts import build_av_messages

console = Console()

_DEFAULT_SNIPPET = (
    "The mitochondria is the powerhouse of the cell, producing ATP through respiration."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snippet",
        default=_DEFAULT_SNIPPET,
        help="Text whose final-token layer-l activation is extracted and verbalized.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument(
        "--explanation-only",
        action="store_true",
        help="Return only the text inside <explanation> tags.",
    )
    parser.add_argument(
        "--thinking", action="store_true", help="Enable Qwen3 thinking mode in the AV."
    )
    args = parser.parse_args()

    nla = NLA.from_pretrained()
    cfg = nla.config

    # --- sanity: the placeholder resolves to exactly one token ---
    messages = build_av_messages(cfg.placeholder_token)
    prompt = nla.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=args.thinking
    )
    ids = nla.tokenizer(prompt, return_tensors="pt")["input_ids"][0]
    n_placeholder = int((ids == cfg.placeholder_token_id).sum())
    scale = cfg.resolve_injection_scale(nla.d_model)

    # --- extract a real h_l at the final token of the snippet ---
    enc = nla.tokenizer(args.snippet, return_tensors="pt").to(nla.av.device)
    with torch.no_grad():
        out = nla.av(**enc, output_hidden_states=True)
    h_l = out.hidden_states[cfg.hidden_states_index][0, -1]  # [d_model]

    console.print(
        Panel(
            f"[bold]INJECT placeholder[/]: {INJECT_PLACEHOLDER}  ->  token "
            f"{cfg.placeholder_token} (id {cfg.placeholder_token_id})\n"
            f"[bold]placeholder occurrences[/]: {n_placeholder} (expect 1)\n"
            f"[bold]extraction layer l[/]: {cfg.extraction_layer}  "
            f"(hidden_states[{cfg.hidden_states_index}])\n"
            f"[bold]injection scale[/]: {scale:.3f}  "
            f"(raw h_l L2 = {h_l.float().norm().item():.1f})\n"
            f"[bold]snippet[/]: {args.snippet}",
            title="AV injection setup",
        )
    )

    text = nla.verbalize(
        h_l,
        max_new_tokens=args.max_new_tokens,
        enable_thinking=args.thinking,
        return_explanation=args.explanation_only,
    )
    console.rule("[bold]AV output (untrained — tests mechanism, not quality)[/]")
    print(text)


if __name__ == "__main__":
    main()
