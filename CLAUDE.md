# doomloop-interp — instructions for Claude / AI assistants

## What this project is

This repo holds the code for the paper **VERS: Steering Reasoning Models by Editing Their
Activation Explanations** (Ihor Vitenko). The paper sources live in `paper/`, which is kept
local and is not in git.

The project started as an interpretability study. The question was what a reasoning model
(`Qwen/Qwen3-1.7B`, thinking mode) carries in its residual stream when its chain of thought
falls into self-doubt ("Wait, let me verify", "Hmm") and loops. To read those states, we
trained a **Natural Language Autoencoder (NLA)** on the model. The study then became a
steering method:

1. **Verbalize.** The AV writes a text explanation of the layer-20 state at each doubt
   paragraph break of one correct trace (35 donor sites).
2. **Edit.** The explanation is edited to remove the doubt.
3. **Reconstruct.** The AR turns the original and the edited text back into states.
4. **Steer.** The mean of the AR differences, normalized, is the direction **N**.

At generation time we add `α·‖h‖·N` to the layer-20 output at every paragraph break inside
`<think>`, starting after a delay of 256 generated tokens.

Main results (all in the paper):

- **Qwen3-1.7B.** On the full GSM8K test, VERS saves 45.8% of tokens at −0.82 pp. On
  MATH-500, it saves 31.3% at −3.05 pp.
- **Comparators.** At half strength, the trade-off lies between those of SEAL and DEER.
- **Transfer.** With fitted affine maps and a strength tuned per model, N transfers to
  Qwen3-8B, 30B-A3B and 235B-A22B and saves 30–35% there.
- **Limits.**
  - On competition problems (hard128), VERS loses 3–7 pp at every size.
  - A fixed direction does not separate needed checks from unneeded ones.
  - Neither the redundant-only direction nor probe-gated steering fixed that.
  - Reversing N (−N) adds doubt. The doubt-edit direction P does not add doubt.
- **Students.**
  - Fine-tuning on steered traces (Steered-SFT) teaches the shorter style without the hook,
    but it loses accuracy on hard questions.
  - DPO students keep accuracy but save less.

## Where the method comes from

The NLA training follows **<https://transformer-circuits.pub/2026/nla/index.html>**
("Natural Language Autoencoders"). Do not invent a different training scheme. For design
questions, that paper and its reference repo are the authority.

- **AV, Activation Verbalizer** (the paper's *actor*). A copy of the target model. Its fixed
  prompt contains one placeholder token, and the **embedding of that token is overwritten**
  by the layer-`l` activation. It writes an explanation inside `<explanation>` tags.
- **AR, Activation Reconstructor** (the paper's *critic*). The same model **cut to its first
  `l+1` blocks**, with the final norm and `lm_head` removed, plus a learned affine map. It
  reads an explanation and predicts `h_l` at the final token.
- **Stage 1, SFT warm start.**
  - The data is 100k Ultra-FineWeb documents × 5 positions, about 500k pairs.
  - The pairs are **split by document** into disjoint AV and AR halves.
  - A local Qwen3.8-27B wrote the summaries.
- **Stage 2, joint RL.**
  - The AR trains on MSE between normalized vectors.
  - The AV trains with GRPO. Its reward is `-log‖h_l - AR(z)‖²`, with a KL penalty toward
    the warm-start AV.
  - One epoch is 6,147 steps on 2× A100.

`natural_language_autoencoders/` is a clone of the reference implementation
(`kitft/natural_language_autoencoders`). It is not in git. **Treat it as read-only.** Its
own `CLAUDE.md` applies to that tree only. Our code lives in `src/reasoning_attention/`.

## Names used everywhere

- **Directions** (unit vectors, `{"unit": float32[d], ...}`):
  - `N` = `data/pool/dir_A_1trace.pt`, the deployed VERS direction. Its sha256 is
    `6efc8476…ad44`, locked in the protocol manifests.
  - `D` = `data/dom/dir_D_diffmeans.pt`, the difference of means between doubt and plain
    boundaries.
  - `P` = `data/vers/dir_P_doubt.pt`, the doubt-edit direction.
  - `negN` = `data/vers/dir_negN.pt`, which is exactly −N.
  - `R101/R202/R303` are random controls.
  - Mapped target directions are `NU<layer>`, `RU`, `DU`, `D8`, … in `data/xfer*/dirs*/`.
- **Policy names** (`reasoning_policy_vllm.py --policies`):
  - `N@a1.0d256` means direction N, α 1.0, delay 256.
  - A `g3` suffix adds the loop guard.
  - `base` means no steering. `brevityA` is the concision prompt.
  - `ban_all` and `ban_para_d256` are the doubt-word bans. `SEAL` is the SEAL baseline.
- **Cohorts:**
  - dev200 and dev400: policy selection and α tuning. dev400 includes 86 GSM8K test
    questions.
  - gsm997: GSM8K test questions that no selection touched.
  - gsm1319: the full GSM8K test, made of gsm997 plus touched322.
  - MATH-500, confirm400 (MATH test), hard128 (AIME/AMC/HMMT).
  - gate400 and sel400 (MATH L4–5), fresh1000.
  - `paper/FACTS.md` has the full registry.
- **Statistics:**
  - Question-clustered bootstrap.
  - Non-inferiority rule: the lower CI end of Δaccuracy must be ≥ −2 pp.
  - Tokens saved = −(steered tokens / base tokens − 1).
  - Doubt is counted by `doubt_blocks`: the 13 markers in the first sentence of a paragraph.

## Layout

```
src/reasoning_attention/
  config.py           # ModelConfig / NLAConfig / DataGenConfig / VLLMConfig / SamplingDefaults
  grading.py          # correct / wrong / no answer; gold canonicalization (D69)
  loops.py            # loop-onset detector (ported from Antidoom)
  metrics.py          # n-gram repetition
  tokenview.py        # per-token layer-20 states of one trace (token explorer)
  model/              # download (hf CLI), loader (transformers + thinking prompts)
  serving/
    vllm_server.py    #   build_llm(): throughput-tuned vLLM engine
    vllm_steering.py  #   the steering hook: norm-relative add at the decoder output, TP/PP/MoE
    vllm_ban.py       #   decode-time doubt-word ban (V1 logits processor)
  data/math_datasets.py  # gsm8k / aime2025 / amc23 / math, one schema
  datagen/            # extract -> split -> explain -> build (+ merge for RL)
  training/           # stage-1 SFT: AV cross-entropy, AR MSE + FVE
  nla/                # arch (truncation), injection, prompts (verbatim), model, sidecar_export
scripts/              # one script per experiment step, queues, reports (not linted)
  reasoning_policy_vllm.py   # every steered / baseline rollout goes through this
  gpu_queue.py               # job queue for multi-GPU pods; *_queue*.py write its jobs
  xfer_* / sel_* / gate_* / distill* / baselines_* / vers_*   # one family per experiment
paper/scripts/        # rebuild paper tables and figures from saved results (in git)
reproduce.mk          # `make <target>` for every paper result; `make repro-help`
EXPERIMENT-*.md       # locked protocols + execution logs, one per experiment
implementation-notes.md   # decision log D1…D82 — APPEND to it
data/, checkpoints/   # not in git (GBs); rebuilt by the pipeline
```

## Locked-in decisions (do not change these silently)

Each one has its reason in `implementation-notes.md`.

**NLA**

- **Target model:** `Qwen/Qwen3-1.7B`, 28 layers, `d_model=2048`, thinking mode on.
- **Activation:** the residual stream at the output of decoder layer `l=20` (0-indexed,
  about 71% depth), read at the final token. It is not post-LayerNorm, and it is not an
  attention or MLP sub-output.
- **HF indexing:** layer-`l` residual == `hidden_states[l+1]` == `hidden_states[21]`. Use
  `NLAConfig.hidden_states_index` / `ar_num_layers`. Do not write the ±1 by hand.
- **No vocabulary change.**
  - The placeholder is `<|image_pad|>` (id **151655**), a token the text model never emits.
  - Never resize the embedding matrix.
- **Injection scale = 1000.**
  - The reference picks "a round number a bit above the mean norm". Our `h_l` norms are
    782–1004.
  - `sqrt_d_model` (≈45) is only the reference's fallback default.
  - This scale is separate from the steering strength α.
- **AR truncation happens in the config, before `from_pretrained`.**
  - Never slice the `nn.ModuleList` afterwards.
  - The "unexpected weights" report for blocks 21–27 is the expected sign that it worked.
- **AR affine map is bias-free** (`ar_affine_bias=False`), as in the reference.
- **Datagen never normalizes.** Vectors are stored raw (`norm="none"`).
- **The AV/AR split is by document, never by row.**
- **The AR's MSE normalizes both prediction and target** (a direction-only loss).
- **AR prompts use `add_special_tokens=True`; AV prompts go through the chat template.**
- **FVE uses a fixed dataset-level denominator.** The paper reports `fve_nrm`: states and
  reconstructions are scaled to unit length, but the mean is not.
- **Full fine-tuning is the default** (A100). `--lora` is only for the 16 GB local card.
- **Warm-start corpus:** `openbmb/Ultra-FineWeb` (`default`, split `en`, column `content`).
  **Always stream it.**

**Steering and evaluation**

- **All rollouts run in vLLM 0.22 through `serving/vllm_steering.py`.**
  - It runs eagerly with synchronous scheduling, and prefix caching is off.
  - The hook stores only the vector and the schedule. It never calls the AV or the AR.
- **Steering is added at paragraph-break tokens:** any token with ≥ 2 newlines, after the
  prompt and before the first `</think>`, from generated token `d` on.
- **The 235B uses tensor parallelism only, never `--pp` ≥ 3.** Pipeline parallelism
  corrupts prefills in the V1 runner (D70).
- **Grading uses the corrected grader (D69).**
  - Old runs are regraded into `data/regraded/`. The original journals are never changed.
  - Report numbers from regraded data.
- **Steered runs repeat statistics, not exact tokens (D63).** Compare a rerun by its CI,
  not by diffing outputs.
- **Lock before running.**
  - Each experiment writes its protocol to `EXPERIMENT-*.md`, and a lock file hashes the
    cohorts and directions before the first rollout.
  - Do not change a locked protocol after results come in. Add a dated amendment instead.
- **Eval datasets carry a single USER turn.** The gold answer stays in the `answer`
  column.

**Environment**

- **Python is pinned to 3.11** (`.python-version`, `uv.lock`, mypy, `requires-python`).
  Keep all four in sync.
- **torch is not pinned.** vLLM drives it, because the local card is a Blackwell
  RTX 5070 Ti (sm_120) that needs cu12.8+ wheels.
- **Never install SGLang into the project venv** (D21).
  - It replaces torch with a cu12 build that has no sm_120 kernels, and it breaks vLLM.
  - The RL stack lives in `.venv-rl` (`make rl-setup`).
  - The sglang tag picks the transformers version of the RL stack. Use `SGLANG_PIN` ≥
    v0.5.15: v0.5.8 pins transformers 4.57.1, which breaks Qwen3 injection (D44).

## Known state and caveats

- **The work is complete and the paper is drafted.**
  - The NLA checkpoints are `checkpoints/{av,ar}_rl_ep1`, released as
    `Strongich/nla-qwen3-1.7b-L20-{av,ar}`.
  - New experiments need a new `EXPERIMENT-*.md` protocol.
- **The paper is local-only.**
  - `paper/` is gitignored except `paper/scripts/`.
  - The build is pdfLaTeX + BibTeX (natbib), run with `cd paper && latexmk paper.tex`.
  - The arXiv package is made with arxiv-latex-cleaner (`paper/README.md`).
  - When a number changes, update the section `.tex`, rebuild `paper.bbl`, then re-run the
    cleaner.
- **`FINDINGS.md` and `CLAIMS.md` are local notes and gitignored.** Never commit them.
- **Pod paths.** The multi-GPU runs used a pod with `/workspace` mounted. The queue
  scripts write jobs that source `/workspace/env.sh` and log to `/workspace/*_logs`.
- **Long runs belong to the user.**
  - Give the user the command (tmux or `run_in_background`) and launch it only when asked.
  - Runs resume from `rollouts.jsonl`, but only with the same arguments, because a manifest
    check enforces that.
- **The LLM judge** (`gpt-5.6-luna`, key in `.env`) was used for the trace audit and the
  reading test. That account is suspended, so a rerun needs a different judge.
- `datagen.extract` prints a `PyGILState_Release` fatal-error trace at shutdown, after its
  outputs are written. It comes from aiohttp threads, not data corruption.
- **Correctness check worth keeping.** The AR's raw layer-`l` residual is bit-identical to
  the full model's `hidden_states[21]` on the same input. If a refactor breaks this, the
  refactor is wrong.
- **Injection smoke test.** If injection fails silently, the AV free-associates off the
  literal placeholder token. Well-formed output is no proof of injection, so assert the
  site count.

## Workflow

```bash
uv sync                  # runtime deps
uv sync --group dev      # + ruff / mypy
make format              # ruff check --fix + ruff format
make lint                # mypy + ruff check + ruff format --check
make repro-help          # paper reproduction targets (reproduce.mk)
```

- Lint, format and type-check cover **`src/` only** (`LINT_DIRS`). `scripts/` is excluded on
  purpose.
- The tools are **ruff** and **mypy**. Do not add black, isort, flake8 or their configs.
- Line length is 100. `nla/prompts.py` has an `E501` exemption because the templates are
  verbatim.
- Run everything through `uv run`.
- **After a non-obvious decision or deviation, append a `D<n>` entry to
  `implementation-notes.md`.** That file records *why*.
- Write findings in simple English. Open each finding with a plain-words summary and one
  real worked example.
