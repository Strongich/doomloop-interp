# doomloop-interp

Code for **VERS: Steering Reasoning Models by Editing Their Activation Explanations**
(Ihor Vitenko).

Reasoning models write a long chain of thought, and many of its checks do not change the
answer. We train a natural language autoencoder (NLA) on Qwen3-1.7B. It turns a
residual-stream state into a text explanation and turns the text back into a state. VERS
(Verbalize, Edit, Reconstruct, Steer) edits the explanations at the doubt points of one
reasoning trace and turns these edits into one fixed direction. During generation, the
direction is added at paragraph breaks.

- On the full GSM8K test, VERS cuts generated tokens by 45.8% at −0.82 pp accuracy.
- On MATH-500, it cuts tokens by 31.3% at −3.05 pp.
- With fitted affine maps, the direction transfers to Qwen3-8B, 30B-A3B and 235B-A22B and
  saves 30–35% of tokens there.
- On competition problems, VERS loses 3–7 pp at every model size.

## Released checkpoints

The NLA used in every experiment (one epoch of joint GRPO training after the warm start).

| base model | layer | d_model | AV | AR |
|---|---|---|---|---|
| Qwen3-1.7B (thinking) | 20 / 28 | 2048 | [`Strongich/nla-qwen3-1.7b-L20-av`](https://huggingface.co/Strongich/nla-qwen3-1.7b-L20-av) | [`Strongich/nla-qwen3-1.7b-L20-ar`](https://huggingface.co/Strongich/nla-qwen3-1.7b-L20-ar) |

- The AV (activation verbalizer) writes an explanation of a layer-20 state.
- The AR (activation reconstructor) rebuilds the state from an explanation.

Each checkpoint has an `nla_meta.yaml` with the prompt template, the injection token and
the injection scale (1,000). Load these values from the file; do not hard-code them.

```bash
make checkpoints   # -> checkpoints/av_rl_ep1, checkpoints/ar_rl_ep1
```

## Setup

```bash
uv sync                    # Python 3.11, vLLM, transformers
uv sync --group dev        # + ruff / mypy
```

- **NLA training only:** the RL stack lives in a separate venv (`make rl-setup`, see
  `scripts/setup_rl_stack.sh`). It needs a clone of the reference implementation
  [`kitft/natural_language_autoencoders`](https://github.com/kitft/natural_language_autoencoders)
  at the repo root. Never install SGLang into the main venv: it replaces torch with a build
  that breaks vLLM.
- **LLM judge steps** (`trace-audit`, `transfer-reading`): put `OPENAI_API_KEY` in `.env`.

## Reproducing the paper

All targets are in `reproduce.mk`, which the `Makefile` includes. To list them:

```bash
make repro-help
```

Each target runs the commands that produced the paper numbers. Rollouts use vLLM with the
steering hook (`scripts/reasoning_policy_vllm.py`). Outputs go to `data/`, which is not in
git. Long runs are resumable: run the same target again after a crash.

### Order and hardware

Targets are listed in the order they depend on each other.

| step | targets | paper | hardware |
|---|---|---|---|
| 0. Inputs | `checkpoints` `traces` | — | 2 GPUs (vLLM) |
| 1. NLA (optional) | `nla-warmstart-data` `nla-sft` `nla-rl-data` `nla-rl` | §3, App. H | 2× A100-80GB, ~1 week |
| 2. Directions | `explanations` `construction` `dir-D` `dir-P` `local-edit` | §3.2, §4.1, App. A | 1 GPU |
| 3. Policy (1.7B) | `policy-select` `policy-controls` `policy-eval` `policy-regrade` | §4.2, App. B | 4× A100 |
| | `gsm8k-full` | §4.2 (full GSM8K) | 1 GPU, ~3.5 h |
| | `strength-sweep` `dom150` `kv-cache` `long-trace` `continuation` | §4.3, App. B–C | 1–4 GPUs |
| | `trace-audit` | §6.1 | API judge |
| 4. Transfer | `transfer-cohorts` `transfer-fit` `transfer-maps` `transfer-screen` `transfer-dev` `transfer-confirm` `transfer-controls` `transfer-regrade` | §5, App. E | 1 A100 per 8B/30B run; 4× A100 (tensor parallel) for 235B |
| | `nla-fve` `transfer-fidelity` `transfer-reading` `transfer-interface` | §3, §5, App. E | up to 6× A100 |
| 5. Students | `distill-1.7b` `distill-1.7b-dpo` `distill-1.7b-gated` `distill-8b` `distill-offline` | §7, App. D, App. G | 6× A100, ~1.5 days each |
| 6. Comparators | `baselines` `baselines-amend-e` `baselines-amend-f` `baselines-report` | §4.4, §6.1, App. D | 4× A100 queue |
| 7. Selectivity | `selective` `gated` `vers-p` `neg-n` | §6.2–6.3, App. F | 6× A100 queue; `neg-n` 1 GPU |
| 8. Figures | `figures` | all figures | CPU |

Some steps need outputs from earlier steps:

- `distill-1.7b` and `distill-8b` come before `baselines`. The SEAL vectors are extracted
  from the teacher generations.
- `selective` uses the 1.7B teacher generations as source traces.
- `distill-8b` and `baselines` use the transfer maps.

### Things to know

- **Same statistics, not the same tokens.** Steered generation in vLLM batches is not
  bit-exact (D63 in `implementation-notes.md`). A rerun repeats the statistics within their
  confidence intervals, not the exact outputs.
- **Pod paths.** The multi-GPU steps ran on a pod with `/workspace` mounted. The queue
  scripts (`scripts/*_queue*.py`, run by `scripts/gpu_queue.py`) write job commands that
  source `/workspace/env.sh` and log to `/workspace/*_logs`. Set `WS=` to change the log
  root, and provide an `env.sh` that sets `HF_HOME` and activates the venv.
- **Corrected grader.** Early runs were graded with an older answer parser. Every table uses
  the corrected grader (D69). `policy-regrade` and `transfer-regrade` regrade saved journals.
- **Judge.** The trace audit and the reading test used `gpt-5.6-luna` through the OpenAI API.
  That account is now suspended, so a rerun needs a different judge account.
- **Protocols.** `EXPERIMENT-*.md` files hold each locked protocol and its execution log.
  `implementation-notes.md` records why each choice was made (D-entries).

### Steps without a full script

These paper numbers or inputs were made with short inline code, not a committed script.
The table gives the source to rebuild each one.

| paper item | what is missing | source |
|---|---|---|
| Table 3 (transfer) CIs | the regraded paired bootstrap was inline | `transfer-regrade` runs `xfer_compare.py`; its CIs differ slightly (8B saving: [29.0, 35.1] vs [28.9, 34.9] in the paper) |
| AV word-share table (§3, 289 triples) | word shares and McNemar test were computed inline | `data/block_explanations_3way.csv` from `explanations` |
| Construction ablation table (App. A) | means of tokens, accuracy and doubt per arm | `data/pool/arms/*.csv` from `construction` |
| gsm997 values to two decimals | a manual call of `report_reasoning_policy.collapse/paired` | `data/eval_gsm8k_fresh` from `policy-eval` |
| Continuation table, commit-row CI | `prefix_report.py` prints the rate without a CI | `data/prefix_online_vllm/branches.csv` |
| Long-trace test (App. B) | the older 297-site direction `data/delta_suppress_mean.pt` has no builder | `steer_sweep.py --delta mean` computes this vector but does not save it |
| Lock files | `data/vers/lock.json` and `data/negn_gsm8k_lock.json` were written inline | `EXPERIMENT-vers-doubt.md`, `EXPERIMENT-negN-and-gsm8k-full.md` §6 |
| 1.7B DPO round (v3) cohorts | `data/distill_v3/{mathtest_fresh1000,easy,hard}.jsonl` have no builder | `EXPERIMENT-offline-distillation.md` (v3 sections) |
| 1.7B Gated-SFT training | jobs were added to the queue by hand | `distill_train.py --method sft --arm gated`, then `distill_eval.py --sets mathpipe`; see `EXPERIMENT-pipeline-sft.md` |
| 8B DPO-steered student, GSM8K | the re-evaluation was run by hand | `distill_eval.py` on `gsm8k_test`; see `EXPERIMENT-offline-distillation.md` |
| web3000 cohort for the maps | the builder is not in `scripts/` | `EXPERIMENT-transfer-claim4.md` |

Some commands in `transfer-fit`, `transfer-screen` and `transfer-fidelity` were rebuilt from
the saved run manifests, not from a logged command line:

- the 30B layer lists in `transfer-fit`;
- the `--spec` lists in `transfer-screen`;
- the target layers of `xfer_fidelity.py extract` in `transfer-fidelity`.

The 1.7B SEAL vector in `baselines` was extracted by hand before the queue started, and
its `--journals` glob was not logged. The target uses the 1.7B teacher generations, in the
same way the queue does for the 8B.

## Repository layout

```
src/reasoning_attention/   NLA (AV, AR), datagen, SFT, grading, vLLM serving
scripts/                   experiments, queues, reports (one script per step)
paper/scripts/             rebuild the paper tables and figures from saved results
reproduce.mk               paper reproduction targets
EXPERIMENT-*.md            locked protocols and execution logs
implementation-notes.md    decision log (D1 ...)
```

The LaTeX sources of the paper are not in this repository.

## License

MIT ([LICENSE](LICENSE)). The checkpoints are fine-tuned from Qwen3-1.7B and also follow
its license.
