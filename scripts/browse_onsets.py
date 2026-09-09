#!/usr/bin/env python3
"""Gradio browser for the onset CSVs — step through explanations with arrows.

    uv run --group dev python scripts/browse_onsets.py
    uv run --group dev python scripts/browse_onsets.py --share

Reads whatever `data/*_correct.csv` / `*_incorrect.csv` pairs exist (plus any
single-file set like `loop_onsets.csv`) and offers each as a "set". For a
*paired* set the two arms of a pair are shown side by side — same question, one
rollout that recovered and one that did not — because that comparison is the
whole point of the paired frame and reading the two files separately loses it.

Keyboard: left/right arrows step, since clicking a button 684 times is not a
workflow. The prefix pane shows the tail of the assistant turn, so the token
`h_l` was read from is the last thing on screen.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# How much of the assistant turn to show. The onset is at the END of the prefix,
# so the tail is the informative part; the head is thousands of tokens of setup.
TAIL_CHARS = 2600


def discover(data_dir: Path) -> dict[str, dict[str, Path]]:
    """Group the CSVs into sets: {name: {"correct": path, "incorrect": path}}."""
    sets: dict[str, dict[str, Path]] = {}
    for path in sorted(data_dir.glob("*.csv")):
        stem = path.stem
        for suffix in ("_correct", "_incorrect"):
            if stem.endswith(suffix):
                sets.setdefault(stem[: -len(suffix)], {})[suffix[1:]] = path
                break
        else:
            sets.setdefault(stem, {})["all"] = path
    return sets


def load(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


class Store:
    """Loaded CSV sets, indexed for pairing."""

    def __init__(self, data_dir: Path, traces_dir: Path, base_model: str, av_path: Path) -> None:
        self.data_dir = data_dir
        self.traces_dir = traces_dir
        self.base_model = base_model
        self.av_path = av_path
        self.sets = discover(data_dir)
        self.cache: dict[str, dict[str, list[dict[str, Any]]]] = {}

    def rows(self, name: str) -> dict[str, list[dict[str, Any]]]:
        if name not in self.cache:
            self.cache[name] = {k: load(v) for k, v in self.sets[name].items()}
        return self.cache[name]

    def is_paired(self, name: str) -> bool:
        """True when both arms exist and line up row-for-row on question_id.

        Checked rather than inferred from the filename: a set sampled without
        --paired also has both files, and showing unrelated rows side by side
        while implying they are a pair would be actively misleading.
        """
        rows = self.rows(name)
        if not {"correct", "incorrect"} <= rows.keys():
            return False
        a, b = rows["correct"], rows["incorrect"]
        if len(a) != len(b):
            return False
        return all(x["question_id"] == y["question_id"] for x, y in zip(a, b, strict=True))


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def conversation(row: dict[str, Any]) -> list[dict[str, str]]:
    """The row's messages as chat turns, for the full-conversation panel.

    Rendered raw (`render_markdown=False` on the Chatbot): a reasoning trace is
    full of `\boxed{}`, `$...$`, `**` and stray underscores that a markdown pass
    silently eats or turns into emphasis, and the point of this panel is to show
    what the model actually emitted.
    """
    try:
        messages = json.loads(row.get("messages") or "[]")
    except json.JSONDecodeError:
        return []
    return [
        {"role": m["role"], "content": m.get("content", "")}
        for m in messages
        if m.get("role") in ("system", "user", "assistant")
    ]


def render_prefix(row: dict[str, Any]) -> str:
    """The assistant turn's tail, with the onset token called out at the end."""
    try:
        messages = json.loads(row.get("messages") or "[]")
    except json.JSONDecodeError:
        messages = []
    assistant = next((m["content"] for m in reversed(messages) if m.get("role") == "assistant"), "")
    if not assistant:
        return "<div class='pane muted'>no messages column in this CSV</div>"
    clipped = len(assistant) > TAIL_CHARS
    tail = assistant[-TAIL_CHARS:]
    head = "<span class='muted'>… earlier reasoning elided …</span><br><br>" if clipped else ""
    marker = esc(row.get("onset_token") or row.get("marker") or "")
    return (
        f"<div class='pane'>{head}<span class='ctx'>{esc(tail)}</span>"
        f"<span class='onset'>{marker}</span> <span class='muted'>&larr; h_l read here</span></div>"
    )


def render_meta(row: dict[str, Any]) -> str:
    label = row.get("label", "")
    outcome = row.get("outcome", "")
    chips = [
        f"<span class='chip {label}'>{esc(label)}</span>",
        f"<span class='chip'>{esc(outcome)}</span>",
        f"<span class='chip'>{esc(row.get('dataset', ''))}</span>",
    ]
    rank, count = row.get("marker_rank"), row.get("marker_count")
    if rank and count:
        chips.append(f"<span class='chip'>marker {esc(rank)}/{esc(count)}</span>")
    if row.get("loop_period_chars"):
        chips.append(f"<span class='chip'>loop period {esc(row['loop_period_chars'])}c</span>")
    facts = [
        ("rollout", f"{row.get('question_id', '')} #{row.get('rollout_index', '')}"),
        ("prefix", f"{row.get('prefix_tokens', '')} tokens"),
        ("‖h_l‖", row.get("h_norm", "")),
        ("gold", row.get("gold", "")),
        ("stop", row.get("stop_reason", "")),
    ]
    grid = "".join(
        f"<div><span class='k'>{esc(k)}</span><span class='v'>{esc(str(v))}</span></div>"
        for k, v in facts
    )
    return f"<div class='chips'>{''.join(chips)}</div><div class='facts'>{grid}</div>"


def render_explanation(row: dict[str, Any]) -> str:
    return f"<div class='pane expl'>{esc(row.get('explanation', ''))}</div>"


def filtered(rows: list[dict[str, Any]], dataset: str) -> list[int]:
    return [i for i, r in enumerate(rows) if dataset == "all" or r.get("dataset") == dataset]


# --------------------------------------------------------------------------- #
# trajectory tab
# --------------------------------------------------------------------------- #

TRAJ_METRICS = [
    ("recurrence", "recurrence — cos to the most similar earlier state", (0.4, 1.005)),
    ("velocity", "velocity — cos to the previous state", (0.2, 1.005)),
    ("eff_rank", "eff_rank — directions used per 384-token window", None),
]


def traj_rows(data_dir: Path) -> list[dict[str, Any]]:
    path = data_dir / "trajectories" / "summary.csv"
    return load(path) if path.exists() else []


def traj_key(row: dict[str, Any]) -> str:
    return f"{row['question_id']}#{row['rollout_index']}"


def traj_label(row: dict[str, Any]) -> str:
    tag = f"x{row['repeats']}" if row["kind"] == "loop" else "control"
    return f"{traj_key(row)}  [{tag}]  {row['outcome']}"


def traj_npz(data_dir: Path, row: dict[str, Any]) -> Any:
    import numpy as np

    name = f"{row['question_id'].replace(':', '_')}_{row['rollout_index']}.npz"
    return np.load(data_dir / "trajectories" / name)


def traj_plot(data_dir: Path, rows: list[dict[str, Any]], label: str) -> Any:
    """Three stacked panels, loop trace against its matched control.

    Plotted together on a shared x axis because the interesting quantity is the
    *gap*: any trace grows more self-similar as context accumulates, so a rise in
    recurrence only means something relative to a rollout of the same question
    that did not loop.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    chosen = next((r for r in rows if traj_label(r) == label), None)
    if chosen is None:
        return None
    series = [(chosen, "#dc2626" if chosen["kind"] == "loop" else "#0ea5e9")]
    mate = next(
        (r for r in rows if r["question_id"] == chosen["question_id"] and r is not chosen),
        None,
    )
    if mate is not None:
        series.append((mate, "#0ea5e9" if mate["kind"] == "control" else "#dc2626"))

    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    for ax, (metric, title, ylim) in zip(axes, TRAJ_METRICS, strict=True):
        for row, colour in series:
            npz = traj_npz(data_dir, row)
            y = npz[metric].astype(float)
            x = np.arange(len(y)) * int(row["stride"])
            tag = "loop" if row["kind"] == "loop" else "control"
            ax.plot(x, y, lw=0.8, color=colour, alpha=0.85, label=f"{traj_key(row)} ({tag})")
            if row["kind"] == "loop":
                ax.axvline(
                    int(row["onset"]) * int(row["stride"]),
                    color="#f59e0b",
                    lw=1.4,
                    ls="--",
                    label="loop onset" if metric == "recurrence" else None,
                )
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(alpha=0.25, lw=0.5)
        if ylim:
            ax.set_ylim(*ylim)
    axes[0].legend(fontsize=8, loc="lower right")
    axes[-1].set_xlabel("response token position")
    fig.tight_layout()
    return fig


def traj_facts(rows: list[dict[str, Any]], label: str) -> str:
    chosen = next((r for r in rows if traj_label(r) == label), None)
    if chosen is None:
        return ""
    group = [r for r in rows if r["question_id"] == chosen["question_id"]]
    head = (
        "<tr><th>rollout</th><th>kind</th><th>outcome</th><th>tokens</th>"
        "<th>recurrence pre&rarr;post</th><th>velocity pre&rarr;post</th>"
        "<th>eff_rank pre&rarr;post</th></tr>"
    )
    body = ""
    for r in group:
        strong = " class='hl'" if r is chosen else ""
        body += (
            f"<tr{strong}><td>{esc(traj_key(r))}</td><td>{esc(r['kind'])}</td>"
            f"<td>{esc(r['outcome'])}</td><td>{esc(r['n_tokens'])}</td>"
            f"<td>{float(r['recurrence_pre']):.3f} &rarr; {float(r['recurrence_post']):.3f}</td>"
            f"<td>{float(r['velocity_pre']):.3f} &rarr; {float(r['velocity_post']):.3f}</td>"
            f"<td>{float(r['eff_rank_pre']):.0f} &rarr; {float(r['eff_rank_post']):.0f}</td></tr>"
        )
    return f"<table class='tt'>{head}{body}</table>"


# --------------------------------------------------------------------------- #
# token explorer
# --------------------------------------------------------------------------- #

EXPLORER_CACHE = "explorer_traces.jsonl"
TOKEN_WINDOW = 900
# Distinct enough to read at a glance, and legible in both themes.
TOKEN_COLORS = {"doubt": "#f59e0b", "gold": "#10b981"}


def build_explorer_cache(data_dir: Path, traces_dir: Path) -> Path:
    """Extract the traces the explorer offers into one small jsonl.

    Scanning gsm8k.jsonl is 568 MB per lookup; the interesting traces are the 68
    from the trajectory run plus the 34 loopers, so pull them out once.
    """
    out = data_dir / EXPLORER_CACHE
    if out.exists():
        return out
    wanted: set[tuple[str, str]] = set()
    for name in ("trajectories/summary.csv", "loop_onsets.csv"):
        path = data_dir / name
        if path.exists():
            for row in load(path):
                wanted.add((row["question_id"], str(row["rollout_index"])))
    kept = []
    for path in sorted(traces_dir.glob("*.jsonl")):
        with path.open() as fh:
            for line in fh:
                row = json.loads(line)
                if (row["question_id"], str(row["rollout_index"])) in wanted:
                    kept.append(
                        {
                            k: row[k]
                            for k in (
                                "question_id",
                                "rollout_index",
                                "dataset",
                                "question",
                                "gold",
                                "response",
                                "outcome",
                                "is_correct",
                            )
                        }
                    )
    with out.open("w") as fh:
        for row in kept:
            fh.write(json.dumps(row) + "\n")
    return out


def explorer_traces(data_dir: Path, traces_dir: Path) -> list[dict[str, Any]]:
    path = build_explorer_cache(data_dir, traces_dir)
    if not path.exists():
        return []
    with path.open() as fh:
        return [json.loads(line) for line in fh]


def explorer_label(row: dict[str, Any]) -> str:
    tag = "correct" if str(row["is_correct"]).lower() in ("true", "1") else row["outcome"]
    return f"{row['question_id']}#{row['rollout_index']}  [{tag}]"


class Models:
    """Target model and AV, loaded on first use rather than at startup."""

    def __init__(self, base: str, av: Path) -> None:
        self.base = base
        self.av_path = av
        self.cache: Any = None
        self.nla: Any = None

    def ready(self) -> tuple[Any, Any]:
        if self.cache is None:
            from reasoning_attention.config import NLAConfig
            from reasoning_attention.tokenview import StateCache

            cfg = NLAConfig()
            print("loading target model for the token explorer...")
            self.cache = StateCache(self.base, cfg.extraction_layer)
        if self.nla is None:
            from reasoning_attention.nla.model import NLA

            print("loading AV...")
            self.nla = NLA.av_only(str(self.av_path))
            self.nla.av.eval()
        return self.cache, self.nla


def similarity_table(view: Any, index: int) -> str:
    """Cosine from the selected token to each anchor, against a matched null.

    The null is the point: arbitrary layer-20 states inside one trace already sit
    at ~0.88 cosine, so a raw similarity to the answer state says nothing. Each
    row reports how many sd above chance the pair is, for pairs the same distance
    apart.
    """
    from reasoning_attention.tokenview import baseline, cosine

    states = view.states
    rows = ""
    for anchor in view.anchors:
        sim = cosine(states, index, anchor.token_index)
        sep = abs(index - anchor.token_index)
        mu, sd = baseline(states, sep, region=view.think)
        z = (sim - mu) / sd if sd > 0 else float("nan")
        verdict = "above chance" if z > 2 else ("below chance" if z < -2 else "chance")
        rows += (
            f"<tr><td>{esc(anchor.label)}</td><td>{anchor.token_index}</td>"
            f"<td>{sim:.4f}</td><td>{mu:.4f} ± {sd:.4f}</td>"
            f"<td>{z:+.1f}σ</td><td>{verdict}</td></tr>"
        )
    head = (
        "<tr><th>anchor</th><th>token</th><th>cosine</th>"
        "<th>chance at same distance</th><th>z</th><th></th></tr>"
    )
    lo, hi = view.think
    note = (
        f"<div class='muted' style='font-size:11.5px;margin-top:6px'>null drawn from "
        f"random pairs inside &lt;think&gt; (tokens {lo}–{hi}) at the same separation"
        "</div>"
    )
    return f"<table class='tt'>{head}{rows}</table>{note}"


CSS = """
#wrap {max-width: 1500px; margin: 0 auto}
.pane {font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12.5px;
  line-height: 1.55; white-space: pre-wrap; word-break: break-word;
  background: var(--block-background-fill); border: 1px solid var(--border-color-primary);
  border-radius: 8px; padding: 12px 14px; max-height: 420px; overflow-y: auto}
.pane.expl {font-family: inherit; font-size: 14.5px; line-height: 1.65; max-height: 300px}
.ctx {opacity: .82}
.onset {background: #f59e0b33; border-bottom: 2px solid #f59e0b; font-weight: 700;
  padding: 1px 3px; border-radius: 3px}
.muted {opacity: .5}
.chips {display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 8px}
.chip {font-size: 11.5px; padding: 2px 9px; border-radius: 999px; font-weight: 600;
  background: var(--background-fill-secondary); border: 1px solid var(--border-color-primary)}
.chip.correct {background: #10b98126; border-color: #10b981; color: #059669}
.chip.incorrect {background: #ef444426; border-color: #ef4444; color: #dc2626}
.facts {display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 4px 14px; font-size: 12px; margin-bottom: 6px}
.facts .k {opacity: .55; margin-right: 6px}
.facts .v {font-family: ui-monospace, monospace}
table.tt {border-collapse: collapse; font-size: 12px; width: 100%}
table.tt th {text-align: left; font-weight: 600; opacity: .6; padding: 4px 10px 4px 0;
  border-bottom: 1px solid var(--border-color-primary)}
table.tt td {padding: 4px 10px 4px 0; font-family: ui-monospace, monospace}
table.tt tr.hl td {font-weight: 700; color: var(--body-text-color)}
.tokens {font-family: ui-monospace, monospace; font-size: 12.5px; line-height: 2.1;
  max-height: 560px; overflow-y: auto}
.tokens span {cursor: pointer}
.qbox {font-size: 13.5px; padding: 10px 12px; border-radius: 8px;
  background: var(--background-fill-secondary); border: 1px solid var(--border-color-primary)}
h3.sec {font-size: 12px; text-transform: uppercase; letter-spacing: .07em; opacity: .6;
  margin: 14px 0 6px}
"""

# Gradio has no key bindings, and 684 pairs is a lot of clicking.
JS = """
() => {
  document.addEventListener('keydown', (e) => {
    if (e.target.matches('input, textarea, select')) return;
    const id = e.key === 'ArrowLeft' ? 'btn-prev' : e.key === 'ArrowRight' ? 'btn-next' : null;
    if (!id) return;
    e.preventDefault();
    document.getElementById(id)?.querySelector('button')?.click();
  });
}
"""


def build(store: Store) -> gr.Blocks:
    names = sorted(store.sets)
    if not names:
        raise SystemExit("no CSVs found — run scripts/onset_verbalize.py first")

    # Gradio 6 moved css/theme from Blocks() to launch(); they are passed in main().
    trajectories = traj_rows(store.data_dir)
    explorer = explorer_traces(store.data_dir, store.traces_dir)
    models = Models(store.base_model, store.av_path)
    view_box: dict[str, Any] = {}

    with gr.Blocks(title="Onset explanations") as demo:
        with gr.Tabs():
            with gr.Tab("explanations"), gr.Column(elem_id="wrap"):
                gr.Markdown(
                    "## Activation explanations at the self-doubt onset\n"
                    "`h_l` (layer 20) read at the last token before the marker, "
                    "verbalized by the trained AV. **←/→** to step."
                )
                with gr.Row():
                    set_dd = gr.Dropdown(names, value=names[0], label="set", scale=3)
                    ds_dd = gr.Dropdown(["all"], value="all", label="dataset", scale=2)
                    arm_dd = gr.Dropdown(
                        ["correct", "incorrect"], value="correct", label="arm", scale=2
                    )
                with gr.Row():
                    prev_btn = gr.Button("← previous", elem_id="btn-prev")
                    pos_md = gr.Markdown("—")
                    next_btn = gr.Button("next →", elem_id="btn-next", variant="primary")
                idx_sl = gr.Slider(0, 1, step=1, value=0, label="index", interactive=True)

                gr.HTML("<h3 class='sec'>question</h3>")
                question = gr.HTML()
                panes: list[dict[str, Any]] = []
                cols = gr.Row()
                with cols:
                    for side in ("left", "right"):
                        with gr.Column():
                            title = gr.Markdown()
                            meta = gr.HTML()
                            gr.HTML("<h3 class='sec'>AV explanation</h3>")
                            expl = gr.HTML()
                            gr.HTML("<h3 class='sec'>context, ending at the onset</h3>")
                            pref = gr.HTML()
                            # Closed by default: a loop-onset prefix runs to 28k
                            # tokens, and mounting that on every arrow press would
                            # make navigation crawl.
                            with gr.Accordion(
                                "full conversation (assistant turn ends at the onset)",
                                open=False,
                            ):
                                # gradio 6.26: `type` and `show_copy_button` are
                                # gone (messages format is the only one now), and
                                # `<think>` would be swallowed into a reasoning
                                # accordion, so reasoning_tags is cleared — the
                                # think block IS the content here.
                                chat = gr.Chatbot(
                                    render_markdown=False,
                                    reasoning_tags=[],
                                    height=560,
                                    show_label=False,
                                    resizable=True,
                                )
                            panes.append(
                                {
                                    "title": title,
                                    "meta": meta,
                                    "expl": expl,
                                    "pref": pref,
                                    "chat": chat,
                                    "side": side,
                                }
                            )

                state = gr.State({"name": names[0], "dataset": "all", "arm": "correct", "i": 0})

            if trajectories:
                with gr.Tab("trajectories"), gr.Column(elem_id="wrap"):
                    gr.Markdown(
                        "## Does the residual stream stop moving?\n"
                        "Layer-20 state at every position. Each looping trace is drawn "
                        "against a non-looping rollout of the **same question** — "
                        "self-similarity rises with context in any trace, so only the "
                        "gap between the two curves means anything."
                    )
                    loops_first = sorted(
                        trajectories, key=lambda r: (r["kind"] != "loop", -int(r["repeats"] or 0))
                    )
                    traj_dd = gr.Dropdown(
                        [traj_label(r) for r in loops_first],
                        value=traj_label(loops_first[0]),
                        label="trace (loops first, most repeats first)",
                    )
                    traj_tbl = gr.HTML()
                    traj_fig = gr.Plot()

                    def show_traj(label: str) -> tuple[Any, Any]:
                        return (
                            traj_facts(trajectories, label),
                            traj_plot(store.data_dir, trajectories, label),
                        )

                    traj_dd.change(show_traj, traj_dd, [traj_tbl, traj_fig])
                    demo.load(show_traj, traj_dd, [traj_tbl, traj_fig])

            if explorer:
                with gr.Tab("token explorer"), gr.Column(elem_id="wrap"):
                    doubt_c, gold_c = TOKEN_COLORS["doubt"], TOKEN_COLORS["gold"]
                    gr.Markdown(
                        "## Click a token, read its activation\n"
                        "Layer-20 state at the clicked position, verbalized by "
                        f"the AV. <span style='color:{doubt_c}'>**self-doubt "
                        f"markers**</span> and <span style='color:{gold_c}'>**the "
                        "gold answer's first appearance**</span> are highlighted. "
                        "First click loads the models (~20s)."
                    )
                    ex_dd = gr.Dropdown(
                        [explorer_label(r) for r in explorer],
                        value=explorer_label(explorer[0]),
                        label="trace",
                    )
                    ex_info = gr.Markdown()
                    ex_slider = gr.Slider(
                        0, 1, step=TOKEN_WINDOW, value=0, label="token window start"
                    )
                    with gr.Row():
                        with gr.Column(scale=3):
                            ex_tokens = gr.HighlightedText(
                                color_map=TOKEN_COLORS,
                                show_legend=False,
                                combine_adjacent=False,
                                show_label=False,
                                elem_classes=["tokens"],
                            )
                        with gr.Column(scale=2):
                            ex_sel = gr.Markdown("*click a token*")
                            ex_sim = gr.HTML()
                            gr.HTML("<h3 class='sec'>AV explanation of this token's h_l</h3>")
                            ex_expl = gr.Markdown()

                    ex_state = gr.State({"label": None, "start": 0, "index": None})

                    def ex_load(label: str, start: float) -> tuple[Any, ...]:
                        row = next(r for r in explorer if explorer_label(r) == label)
                        cache, _ = models.ready()
                        from reasoning_attention.tokenview import build_view

                        key = (label,)
                        if view_box.get("key") != key:
                            view = build_view(
                                cache.tokenizer, row["question"], row["response"], row["gold"]
                            )
                            cache.fill(view)
                            view_box["key"] = key
                            view_box["view"] = view
                        view = view_box["view"]
                        n = len(view.ids)
                        start = int(min(max(0, start), max(0, n - 1)))
                        anchors = ", ".join(f"**{a.label}** @{a.token_index}" for a in view.anchors)
                        info = (
                            f"gold `{row['gold']}` · {n} response tokens · "
                            f"{sum(1 for x in view.labels if x == 'doubt')} doubt-marker tokens · "
                            f"anchors: {anchors}"
                        )
                        return (
                            info,
                            gr.update(minimum=0, maximum=max(n - TOKEN_WINDOW, 0), value=start),
                            view.tokens_for_display(start, start + TOKEN_WINDOW),
                            {"label": label, "start": start, "index": None},
                        )

                    def ex_pick(st: dict[str, Any], evt: gr.SelectData) -> tuple[Any, ...]:
                        view = view_box["view"]
                        index = int(st["start"]) + int(evt.index)
                        piece = view.piece(index)
                        return (
                            f"**token {index}** · {piece!r} · label `{view.labels[index]}`",
                            similarity_table(view, index),
                            {**st, "index": index},
                        )

                    def ex_explain(st: dict[str, Any]) -> str:
                        if st.get("index") is None:
                            return ""
                        _, nla = models.ready()
                        view = view_box["view"]
                        vec = view.states[int(st["index"])].float().cpu()
                        return nla.verbalize(
                            vec, max_new_tokens=256, return_explanation=True
                        ).strip()

                    ex_dd.change(
                        ex_load, [ex_dd, ex_slider], [ex_info, ex_slider, ex_tokens, ex_state]
                    )
                    ex_slider.release(
                        ex_load, [ex_dd, ex_slider], [ex_info, ex_slider, ex_tokens, ex_state]
                    )
                    # Two events: the fast one paints the selection and similarities
                    # immediately, the slow one queues the AV generation behind it, so
                    # clicking a new token never waits on the previous explanation.
                    ex_tokens.select(ex_pick, ex_state, [ex_sel, ex_sim, ex_state]).then(
                        lambda: "*generating…*", None, ex_expl
                    ).then(ex_explain, ex_state, ex_expl)

        outs = [question, pos_md, idx_sl, arm_dd, state]
        for p in panes:
            outs += [p["title"], p["meta"], p["expl"], p["pref"], p["chat"]]

        def view(st: dict[str, Any]) -> list[Any]:
            name = st["name"]
            rows_by_arm = store.rows(name)
            paired = store.is_paired(name)
            arm = "all" if "all" in rows_by_arm else st["arm"]
            primary = rows_by_arm[arm]
            keep = filtered(primary, st["dataset"])
            if not keep:
                blank: list[Any] = ["", "", "", "", []] * len(panes)
                return ["", "**no rows match**", gr.update(), gr.update(visible=False), st, *blank]
            i = max(0, min(st["i"], len(keep) - 1))
            st = {**st, "i": i}
            row = primary[keep[i]]

            sides: list[tuple[str, dict[str, Any]] | None] = [("this rollout", row)]
            if paired:
                sides = [
                    ("✅ answered correctly", rows_by_arm["correct"][keep[i]]),
                    ("❌ did not", rows_by_arm["incorrect"][keep[i]]),
                ]
            sides += [None] * (len(panes) - len(sides))

            out: list[Any] = [
                f"<div class='qbox'>{esc(row.get('question', ''))}</div>",
                f"**{i + 1} / {len(keep)}**" + ("  · paired" if paired else ""),
                gr.update(minimum=0, maximum=max(len(keep) - 1, 0), value=i),
                gr.update(visible=not paired and "all" not in rows_by_arm),
                st,
            ]
            for _p, spec in zip(panes, sides, strict=True):
                if spec is None:
                    out += ["", "", "", "", []]
                    continue
                head, r = spec
                out += [
                    f"**{head}**",
                    render_meta(r),
                    render_explanation(r),
                    render_prefix(r),
                    conversation(r),
                ]
            return out

        def step(st: dict[str, Any], delta: int) -> list[Any]:
            return view({**st, "i": st["i"] + delta})

        def reset(st: dict[str, Any], **kw: Any) -> list[Any]:
            return view({**st, **kw, "i": 0})

        prev_btn.click(lambda st: step(st, -1), state, outs)
        next_btn.click(lambda st: step(st, +1), state, outs)
        idx_sl.input(lambda st, v: view({**st, "i": int(v)}), [state, idx_sl], outs)
        arm_dd.change(lambda st, v: reset(st, arm=v), [state, arm_dd], outs)
        ds_dd.change(lambda st, v: reset(st, dataset=v), [state, ds_dd], outs)

        def switch(st: dict[str, Any], name: str) -> list[Any]:
            rows_by_arm = store.rows(name)
            first = next(iter(rows_by_arm.values()))
            datasets = ["all"] + sorted({r.get("dataset", "") for r in first} - {""})
            return [gr.update(choices=datasets, value="all")] + reset(st, name=name, dataset="all")

        set_dd.change(switch, [state, set_dd], [ds_dd] + outs)
        demo.load(switch, [state, set_dd], [ds_dd] + outs).then(None, None, None, js=JS)
    return demo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--traces", type=Path, default=Path("data/traces"))
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--av", type=Path, default=Path("checkpoints/av_rl_ep1"))
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    store = Store(args.data, args.traces, args.base, args.av)
    print("sets found:")
    for name, files in store.sets.items():
        print(f"  {name}: {', '.join(sorted(files))}")
    build(store).launch(
        server_port=args.port,
        share=args.share,
        inbrowser=False,
        css=CSS,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
