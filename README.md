# doomloop-interp

**What happens inside a reasoning model once it hits the "doom loop" tokens in its
own chain of thought?**

Small reasoning models (here `Qwen/Qwen3-1.7B` in thinking mode) frequently derail
mid-`<think>` into self-doubt boilerplate — *"Wait, let me verify"*, *"But wait,
that's not right"*, *"Hmm, let me reconsider"* — and then loop, sometimes verbatim,
until the token budget runs out. The object of study is not the text but the
**residual stream at those positions**: what representation is the model carrying
when it emits "Wait, let me verify", and how does it change as the loop tightens?

The approach is to make those activations *readable* by training a **Natural
Language Autoencoder** (NLA) on the target model, then verbalizing `h_l` at
doom-loop token positions along real reasoning traces (GSM8K / AIME2025 / AMC23)
and comparing them against healthy reasoning positions.

All training method comes from
[**Natural Language Autoencoders**](https://transformer-circuits.pub/2026/nla/index.html)
(Anthropic, 2026). Two components, both copies of the target model:

- **AV — Activation Verbalizer** (their *actor*): the full model, with `h_l`
  injected over one placeholder token's embedding; emits an explanation.
- **AR — Activation Reconstructor** (their *critic*): the model truncated to its
  first `l+1` blocks plus a learned affine map; reads the explanation and predicts
  `h_l` back.

Trained in two stages: an SFT warm-start on a summarization proxy task
(~0.3–0.4 FVE), then joint RL — AR by MSE, AV by GRPO with reconstruction as the
reward (0.6–0.8 FVE).

## Status

| | |
|---|---|
| Datagen (warm-start + RL sets) | works, verified end to end |
| Stage-1 SFT loops (AV + AR) | run; verified mechanically, **not trained to convergence** |
| Stage-2 joint RL (GRPO) | scripted, **not yet run** |
| The doom-loop study itself | not started — only the detectors in `metrics.py` |

No checkpoint in this repo is worth drawing conclusions from yet.

## Setup

```bash
uv sync                  # runtime deps
uv sync --group dev      # + ruff / mypy
```

The reference implementation is **not vendored** — clone it next to the source,
where `CLAUDE.md` and `scripts/setup_rl_stack.sh` expect it:

```bash
git clone https://github.com/kitft/natural_language_autoencoders.git
```

Explanation labeling needs an OpenAI key in `.env` (gitignored):

```
OPENAI_API_KEY=sk-...
```

> GPU note: local dev targets a Blackwell RTX 5070 Ti (sm_120), so `torch` is left
> unpinned and vLLM drives its version. Training targets 2× A100. See
> `implementation-notes.md`.

## Pipeline

```bash
# 0. verify the labeling endpoint
uv run python scripts/smoke_gpt_endpoint.py

# 1. warm-start data: 100k docs x 5 positions -> 500k pairs, split by document
DATA_DIR=/workspace/data/warmstart scripts/build_warmstart_data.sh

# 2. inspect what the labeler actually wrote
uv run python scripts/dump_explanations.py --from-parquet .../av_explained.parquet

# 3. Stage-1 SFT (AV and AR are fully independent — any order, or in parallel)
DATA_DIR=/workspace/data/warmstart scripts/train_sft.sh        # both
DATA_DIR=... scripts/train_sft.sh ar                           # one

# 4. Stage-2 RL data: 40k web + 40k chat docs -> 400k prompts, no API spend
DATA_DIR=/workspace/data/rl scripts/build_rl_data.sh

# 5. Stage-2 GRPO (needs the separate RL env — see below)
make rl-setup
RL_PARQUET=... ACTOR_SFT_CKPT=... CRITIC_SL_CKPT=... RUN_DIR=... scripts/train_grpo.sh
```

### Why the RL stack lives in its own venv

`sglang[all]` pins torch to a **cu12** build (2.9.1), which has no sm_120 kernels
— installing it into the project venv breaks the local Blackwell GPU and vLLM
along with it. `scripts/setup_rl_stack.sh` builds `.venv-rl` instead, from the
pinned `requirements/rl.txt` plus patched Miles + SGLang checkouts. The project
venv is never touched.

## Cluster

```bash
make start      # apply pod.yaml, wait, port-forward 2222:22, drop into zsh
make connect    # exec into the running pod
make stop
```

2× A100-SXM4-80GB on `malea-srv01`, PVC `doomloops-interp` mounted at
`/workspace`. Put datasets and caches on the PVC — the container layer is
ephemeral, and the `explain` stage's `.chunks/` directory is the only thing
standing between a crash and re-paying for the API calls.

## Layout

```
src/reasoning_attention/
  config.py       # every knob: model, NLA, explainer, warm-start + RL data recipes
  metrics.py      # doom-loop / repetition detectors
  model/          # download + plain-transformers loader (thinking mode)
  serving/        # vLLM engine for throughput
  data/           # gsm8k / aime2025 / amc23, one schema
  datagen/        # extract -> split -> explain -> build (+ merge for RL)
  nla/            # AV + AR construction, injection, prompt templates
  training/       # Stage-1 SFT: AV cross-entropy, AR MSE + FVE
scripts/          # smoke tests, data builders, launchers (NOT linted)
```

## Dev

```bash
make format   # ruff check --fix + ruff format
make lint     # mypy + ruff check + ruff format --check
```

`CLAUDE.md` carries the invariants that must not be broken silently.
`implementation-notes.md` is the decision log (D1…D25) — the record of *why*, and
what makes this work reconstructable.
