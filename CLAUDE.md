# reasoning-attention — instructions for Claude / AI assistants

## What this project is investigating

**Research question: what actually happens inside a reasoning model once it hits
the "doom loop" tokens in its own chain of thought?**

Small reasoning models (here `Qwen/Qwen3-1.7B` in thinking mode) frequently
derail mid-`<think>` into self-doubt boilerplate — *"Wait, let me verify"*,
*"But wait, that's not right"*, *"Hmm, let me reconsider"* — and then loop,
sometimes verbatim, until the token budget runs out. The interesting object is
not the text but the **residual stream at those positions**: what representation
is the model actually carrying when it emits "Wait, let me verify", and how does
it change as the loop tightens?

The plan is to make those activations *readable* by training a **Natural
Language Autoencoder (NLA)** on the target model, then verbalizing `h_l` at
doom-loop token positions along real reasoning traces (math problems from
GSM8K / AIME2025 / AMC23) and comparing them against healthy reasoning
positions.

So the pipeline has two halves, and they are at very different maturity:
1. **NLA machinery** (the instrument) — mostly scaffolded, not yet trained.
2. **The doom-loop study** (the actual experiment) — not started; only the
   detectors in `metrics.py` and the eval datasets exist so far.

## Where the method comes from

**All training ideas are taken from
<https://transformer-circuits.pub/2026/nla/index.html>** ("Natural Language
Autoencoders"). Do not invent a different training scheme — when a design
question comes up, the paper (and the reference repo below) is the authority.

The paper's method, in the terminology this codebase uses:

- **AV — Activation Verbalizer** (the paper's *actor*): a copy of the target
  model. Fixed prompt containing one placeholder token whose **embedding is
  overwritten** by the layer-`l` activation; it emits a natural-language
  explanation inside `<explanation>` tags.
- **AR — Activation Reconstructor** (the paper's *critic*): the same model
  **truncated to its first `l+1` blocks**, final norm and `lm_head` stripped,
  plus a learned affine map. It reads the explanation and predicts `h_l` at the
  final token.
- **Stage 1 — SFT warm-start** on a text-summarization proxy task: extract
  `h_l` from truncated snippets, have a strong model write the summary, then SFT
  the AV on (activation → summary) and the AR on (summary → activation). Paper
  reports ~0.3–0.4 FVE at this point. Our recipe (implemented in `datagen/`):
  100k Ultra-FineWeb documents × 5 random positions ≈ 500k pairs, **split evenly
  by document** into disjoint halves — `D_AV` ≈ 250k, `D_AR` ≈ 250k. A pair is
  never seen by both models.
- **Stage 2 — joint RL**: AR trains by **MSE** against the true `h_l`; AV trains
  by **GRPO** with reconstruction as the reward, `r = -log‖h_l - AR(z)‖²`, plus a
  **KL penalty toward the AV init** to keep it fluent. Sampling at `T=1`. Paper
  reaches 0.6–0.8 FVE.

`natural_language_autoencoders/` is Anthropic's **reference implementation**,
cloned into the repo for guidance. **Treat it as read-only reference** — our code
lives in `src/reasoning_attention/`. It has its own `CLAUDE.md` describing *its*
invariants; those apply to that tree, not automatically to ours (it is
FSDP/Megatron/SGLang multi-GPU; we are single-GPU transformers).

## Layout

```
src/reasoning_attention/
  config.py           # ModelConfig / NLAConfig / DataGenConfig / VLLMConfig / SamplingDefaults
  metrics.py          # doom-loop + repetition detectors (the study's entry point)
  model/
    download.py       #   snapshot download via the `hf` CLI
    loader.py         #   plain-transformers init + thinking-mode prompts
  serving/
    vllm_server.py    #   build_llm(): throughput-tuned vLLM engine
  data/
    math_datasets.py  #   gsm8k / aime2025 / amc23, one unified schema
  datagen/            # datasets: extract -> split -> explain -> build (+ merge for RL)
    extract.py        #   stage 0: Ultra-FineWeb (streaming) -> raw h_l parquet
    split.py          #   stage 1: document-level 50/50 -> disjoint AV/AR halves
    explain.py        #   stage 2: gpt-5.6-luna writes each row's summary
    build.py          #   stage 3: AV/AR training parquets
    providers.py      #   OpenAIProvider (Responses API, reasoning effort)
    prompts.py        #   explainer instruction prompt (verbatim from the ref repo)
    merge.py          #   concat + shuffle the two RL corpora
    sidecar.py        #   the datagen <-> training metadata contract
  training/            # stage-1 SFT warm-start
    data.py           #   AV/AR parquet datasets + collators (normalization lives here)
    sft.py            #   the loop: AV cross-entropy, AR MSE + FVE
  nla/
    arch.py           #   truncate_config_layers / strip_lm_head / strip_final_norm
    injection.py      #   inject_at_placeholder + normalize_activation
    prompts.py        #   AV_TEMPLATE / AR_TEMPLATE (verbatim from the reference repo)
    model.py          #   NLA (AV + AR), ARModel, NLA.verbalize()
scripts/              # smoke tests + the forward-pass tracer (NOT linted, NOT type-checked)
  smoke_gpt_endpoint.py          #   verifies the explainer endpoint end to end
  dump_explanations.py           #   writes the labeler's output to jsonl/md/stats
  train_sft.sh                   #   launcher for the AV/AR warm-start
  build_rl_data.sh               #   200k web + 200k chat RL prompt set
  setup_rl_stack.sh              #   Miles + patched SGLang into .venv-rl (NOT the main venv)
  train_grpo.sh                  #   stage-2 GRPO, 2xA100
natural_language_autoencoders/   # reference implementation — read-only
implementation-notes.md          # running decision log (D1…D22) — APPEND to this
```

## Locked-in decisions (don't silently change these)

These were deliberate and are recorded with rationale in `implementation-notes.md`:

- **Target model** `Qwen/Qwen3-1.7B`, 28 layers, `d_model=2048`, thinking mode on.
- **Activation = residual stream at the output of decoder layer `l=20`**
  (0-indexed, ~71% depth), read at the **final token**. Not post-LayerNorm, not
  an attention/MLP sub-output.
- **HF indexing**: `hidden_states` has length `num_layers+1`; index 0 is the
  embedding. So layer-`l` residual == `hidden_states[l+1]` == `hidden_states[21]`.
  Use `NLAConfig.hidden_states_index` / `ar_num_layers` instead of open-coding
  the ±1.
- **No vocabulary change.** The placeholder is an existing token the text-only
  model never emits — `<|image_pad|>` (id **151655**) — because embedding-level
  injection makes the token's lexical identity irrelevant. Never resize the
  embedding matrix for this.
- **Injection scale** = **1000** by default. The reference's rule is empirical,
  not a function of `d_model`: *"picked as a round number a bit above the mean
  norm of the dataset's vectors"* (`docs/inference.md`). Their own values follow
  only that — 150 for Qwen2.5-7B (mean ~125), 80000 for Gemma-3-12B (its scaled
  embedding inflates norms ~500x), 30 for Llama-3.3-70B (below `sqrt(8192)`).
  Our `h_l` norms are 782-1004, mean ~900. `sqrt_d_model` (≈45.3) is the repo's
  *fallback* default, not its recipe, and would inject ~20x below distribution.
- **AR truncation happens in the config, before `from_pretrained`** — set
  `num_hidden_layers` and slice the per-layer arrays (`layer_types`, …). Never
  slice the `nn.ModuleList` post-hoc. The transformers "unexpected weights"
  report for blocks 21–27 is the *expected* confirmation it worked.
- **AR affine map is bias-free** (`ar_affine_bias=False`) to match the reference
  repo, even though the blog writes `A@x + b`. It's a knob, not an accident.
- **Eval datasets carry a single USER turn** — the gold answer is kept in the
  `answer` column and never appended as an assistant turn.
- **Explanations come from `gpt-5.6-luna` at `reasoning_effort="high"`**, not a
  local model. Key in `.env` (gitignored). Reasoning models reject
  `temperature`/`top_p` — don't add them back.
- **Warm-start corpus** is `openbmb/Ultra-FineWeb`, config `default`, split `en`,
  text in the **`content`** column. **Always stream it**: a non-streaming
  `load_dataset` pulls the whole ~1 TB split before yielding document 1.
- **Datagen never normalizes.** Vectors are stored raw with `norm="none"` in the
  sidecar; scaling happens at injection and loss time.
- **The AV/AR split is by document, never by row.** 5 positions from one document
  share a prefix, so a row-level split leaks context across the halves and they
  stop being disjoint.
- **The AR's MSE normalizes BOTH prediction and target** to the injection scale —
  it is a direction-only loss, matching the reference's `mse_scale`. Comparing a
  free-magnitude prediction against a fixed-norm target measures the wrong thing.
- **AR prompts tokenize with `add_special_tokens=True`** to match the extractor
  that produced the gold activations; **AV prompts go through the chat template**
  so training matches `NLA.verbalize()` at inference.
- **FVE uses a fixed dataset-level denominator** (`data.baseline_variance`), never
  a per-batch variance — otherwise the metric is not comparable across steps.
- **Full fine-tuning is the training default**, matching the reference (their
  actor SFT is the full 28-layer model under FSDP). It needs ~21 GB of optimizer
  state: fine on the A100-80GB training box, impossible on the 16 GB local card,
  where `--lora` is the fallback. Prefer full FT for the AR especially — it is the
  Stage-2 reward model, and a rank-limited reward model is one the AV can game.
  The AR's affine map is always trained in full, in fp32.
- **Python is pinned to 3.11** (`.python-version`, `uv.lock`, mypy,
  `requires-python`). Keep all four in sync.
- **torch is intentionally unpinned** — vLLM drives it, because the box is a
  Blackwell RTX 5070 Ti (sm_120) needing cu12.8+ wheels.

## Known state / caveats

- **The AV is untrained.** `NLA.verbalize()` produces well-formed
  `<explanation>` output that echoes the prompt framing rather than decoding the
  vector. The *mechanism* is verified end-to-end; quality needs the SFT
  warm-start. Don't report untrained verbalizations as findings.
- **Datagen and the Stage-1 SFT loops exist and run; Stage-2 RL does not.** No
  GRPO loop yet. The SFT loops are verified mechanically on a 6-row smoke set,
  *not* trained to convergence — no checkpoint here is worth drawing conclusions
  from.
- **SFT hyperparameters now come from their Qwen2.5-7B case study** (D19): LR
  2e-5 -> 2e-6 cosine, warmup 5%, identity-init AR affine, save every 500 steps.
  Batch size is ours. `injection_scale` was deliberately NOT copied and wants a
  sweep — theirs is 150 at d_model 3584, which is an absolute value (~2.5x
  sqrt(d)), not a ratio.
- **Never install SGLang into the project venv** (D21). It downgrades torch to a
  cu12 build with no sm_120 kernels, killing Blackwell support locally, and breaks
  vLLM. The RL stack lives in `.venv-rl` via `scripts/setup_rl_stack.sh`.
- `datagen.extract` prints a `PyGILState_Release` fatal-error trace at
  interpreter shutdown, *after* the parquet and sidecar are written. The output
  is intact — it's the streaming dataset's aiohttp threads at teardown, not the
  extraction path. Don't chase it as data corruption.
- **Correctness check worth preserving**: the AR's raw layer-`l` residual is
  bit-identical (max abs diff 0.0) to the full AV model's `hidden_states[21]` on
  the same input. If a refactor breaks that equality, the refactor is wrong.
- The reference repo's loudest smoke test carries over conceptually: if
  injection silently fails, the model free-associates off the *literal*
  placeholder token instead of the activation. Well-formed output is not proof
  that injection happened — assert the site count.

## Workflow

```bash
uv sync                 # runtime deps
uv sync --group dev      # + ruff / mypy
make format              # ruff check --fix + ruff format
make lint                # mypy + ruff check + ruff format --check
```

- Lint/format/type-check scope is **`src/` only** (`LINT_DIRS`); `scripts/` is
  deliberately excluded.
- Tooling is **ruff** (format + lint + import sorting) and **mypy**. There is no
  black/isort/flake8/autoflake — don't reintroduce them or their config files.
- Line length 100. `nla/prompts.py` has an `E501` exemption because the AV/AR
  templates are verbatim and must not be re-wrapped.
- Run everything through `uv run` so it uses the project venv.
- **After a non-obvious decision or deviation, append a `D<n>` entry to
  `implementation-notes.md`** — that file is the project's memory of *why*, and
  it is how this work stays reconstructable.
