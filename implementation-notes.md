# Implementation Notes

Running log of decisions, deviations, and tradeoffs made while implementing the
spec. Anything here was *not* explicitly in the spec, or was a judgement call.

## Spec (as I understood it)

1. `uv` venv pinned to Python 3.11 + `pyproject.toml`. Packages: transformers,
   matplotlib, pytorch, vllm.
2. Main project package with a model-initialization section
   (`Qwen/Qwen3-1.7B`, thinking mode). Download via the `hf` CLI.
3. A vLLM serving entry point returning a configured vLLM `LLM` instance, with
   the throughput params given in the spec.

## Environment (detected)

- uv venv to 3.11.
- `uv` 0.7.19 available.
- GPU: **NVIDIA RTX 5070 Ti** (Blackwell, compute capability sm_120).

## Decisions / deviations

### D1 — Blackwell (sm_120) GPU drives version floors
The RTX 5070 Ti is Blackwell. PyTorch only ships sm_120 kernels in the CUDA 12.8
(cu128) wheels, and vLLM only added Blackwell support in recent releases.
Consequence:
- We do **not** pin an old torch. vLLM pins its own compatible torch, so I let
  **vLLM drive the torch version** rather than pinning torch myself (pinning
  both almost always conflicts). `torch` is still listed as a direct dep per the
  spec, but unpinned so the resolver can satisfy vLLM's constraint.
- Require `vllm >= 0.8.5` (first releases with solid Blackwell/sm_120 support).
  If the resolver/install fails on this box, the fix is usually a newer vLLM +
  cu128 torch, not an older one.

### D2 — `pytorch` package name
On PyPI the import package `torch` is the distribution name; `pytorch` is a
dummy. pyproject lists `torch` (the real one).

### D3 — Project layout
`src/` layout with package `reasoning_attention`. Sections requested map to
sub-packages: `model/` (init + download) and `serving/` (vLLM endpoint). Keeps
the "sections" cleanly separated and importable.

### D4 — Thinking mode for Qwen3
Qwen3 exposes thinking mode two ways:
- **transformers**: `tokenizer.apply_chat_template(..., enable_thinking=True)`.
- **vLLM**: same flag is passed through `chat_template_kwargs={"enable_thinking": True}`
  on the sampling/serving call (not on `LLM()` construction).
So the `LLM` factory itself is mode-agnostic; thinking mode is a per-request
concern. I expose a helper for building the prompt with thinking enabled, and
document the vLLM-side flag. Default sampling params follow Qwen's official
recommendation for thinking mode (temp 0.6, top_p 0.95, top_k 20).

### D5 — Model download via `hf` CLI
Per spec, download uses the `hf` CLI (not `huggingface_hub.snapshot_download`).
`model/download.py` shells out to `hf download Qwen/Qwen3-1.7B`. It's idempotent
(hf skips already-cached files). I did not trigger an actual multi-GB download
as part of scaffolding — the function is wired and runnable on demand.

### D7 — Makefile + lint tooling (added per follow-up request)
Added a `Makefile` with `format` (isort + autoflake + black) and `lint` (mypy +
isort --check + black --check + flake8). Tooling lives in a `[dependency-groups]
dev` group, installable with `uv sync --group dev` (or `--only-group dev`). All
targets run via `uv run` so they use the project venv. Config:
- `LINT_DIRS := src` (only our package; the spec's snippet linted dir vars).
- black/flake8 line length aligned to 100; isort uses the black profile;
  `.flake8` ignores E203/W503 (black-compatible).
- One real mypy finding fixed: `build_sampling_params` built a `dict` mypy
  inferred as `dict[str, float]`, which broke `**params` into `SamplingParams`.
  Annotated it `dict[str, Any]`. `make lint` is green.

### Resolved versions (after `uv sync`)
The unpinned/loose constraints resolved to a current Blackwell-ready stack:
- torch **2.11.0+cu130** (CUDA 13.0), vLLM **0.22.0**, transformers **5.9.0**.
- `torch.cuda.is_available()` is True and device capability reports **(12, 0)**
  = sm_120, confirming the Blackwell wheels are correct on this box.
- Note transformers resolved to the 5.x line (spec floor was just >=4.51).

### D8 — `scripts/trace_forward.py` (forward-pass tracer)
Loads the model via `model.loader.load_model` and traces a single forward.
Two modes:
- **default (forward hooks)**: prints every nn.Module in true execution order
  with input/output shapes. Limitation: hooks only fire for registered modules,
  so the residual `+` and RoPE application are *invisible* (they're bare tensor
  ops, not modules). `--max-depth N` collapses deep leaves.
- **`--fx` (op-level)**: originally intended to use `transformers.utils.fx`
  `symbolic_trace`, but **transformers 5.x removed that module**, and HF models
  aren't cleanly `torch.fx`-traceable anyway (dynamic control flow + KV cache).
  Switched to a **`TorchDispatchMode`** that intercepts every ATen op during a
  real forward. This captures the residual adds (`aten.add.Tensor` on
  `(B,S,hidden)` tensors) — confirmed **56 = 2×28 layers**. `--all-ops` dumps
  all captured ops (2643 for a 19-token prompt), capped at 400 rows.
Console width pinned to 140 so the rich tables stay readable when piped.
`scripts/` is intentionally outside `LINT_DIRS` (src-only), so the tracer isn't
type-checked.

### D9 — NLA activation choice (layer + activation type)
Implementing the method from transformer-circuits.pub/2026/nla. Decisions:
- **Activation type = residual stream** = the *output of decoder layer l*
  (HF `hidden_states`), at the **final token**. Explicitly NOT post-LayerNorm
  and NOT an attention/MLP sub-output. This is the `(B, S, 2048)` tensor our
  tracer labels `model.layers.{l}` output.
- **Layer l = 20** (0-indexed), ~71% depth of 28 layers — "mid-to-late" per the
  paper. User-selected over the l=18 default (wanted a more processed/semantic
  representation; AR keeps 21 layers).
- **HF indexing gotcha**: `outputs.hidden_states` has length num_layers+1;
  index 0 is the embedding, index i is the output of layer i-1. So layer-l
  residual stream = `hidden_states[l+1]` = `hidden_states[21]`. Captured as
  `NLAConfig.hidden_states_index`.
- **AR truncation**: AR model = first l layers (+layer l) = 21 leading layers
  (`NLAConfig.ar_num_layers`).
- **Normalization**: L2-normalize h_l to unit norm (paper does this for
  stability), then scale by a fixed constant on injection. `injection_scale`
  defaults to 1.0 as a placeholder — needs tuning to match Qwen3's embedding
  norm scale (TODO once we wire injection).
- **Placeholder token** (task 1): **no vocabulary change.** Embedding-level
  injection makes the token's lexical identity irrelevant — the inner layers
  only ever see the activation vector we swap into its embedding slot, never the
  original token. So instead of adding a new token (and resizing the embedding
  matrix), we **repurpose an existing token the text model never emits**:
  `<|image_pad|>` (id **151655**), a multimodal padding placeholder that Qwen3-1.7B
  (text-only) never produces. Verified single-token. Stored as
  `NLAConfig.placeholder_token` / `placeholder_token_id`.
  - Rationale for *this* token over others: vision/image pad tokens already have
    real embedding rows, are guaranteed unused in text generation, and won't
    collide with the chat template (which inserts none of them in text mode).
    `<|endoftext|>`/`<unk>` were rejected because they *do* occur in normal use.
All of this lives in `config.NLAConfig`. No extraction code written yet — this
commit is just the locked-in decision + arch facts (NUM_HIDDEN_LAYERS=28,
D_MODEL=2048).

### D10 — NLA class: AV + AR scaffold (`nla/` package)
Built `reasoning_attention.nla` with `NLA`, `ARModel`, `AROutput`, modeled on the
reference repo `natural_language_autoencoders` (which the user cloned into the
project root for guidance). The repo's `NLACriticModel` is the AR; their *actor*
is the AV. Mapped to our single-GPU Qwen3 setup, dropping all FSDP / Megatron /
multimodal machinery we don't need.

- **AV** = full `AutoModelForCausalLM` (28 blocks). Untouched.
- **AR** = backbone truncated to first `ar_num_layers = l+1 = 21` blocks, with
  `lm_head → Identity` and final `norm → Identity`, plus a learned
  `Linear(2048, 2048, bias=False)` affine map (`ARModel.affine`).
- **Truncation method** (copied from repo, important): set
  `config.num_hidden_layers = l+1` **before** `from_pretrained` (so the loader
  reads only kept blocks) and slice per-layer config arrays (`layer_types` etc.)
  to match — never post-hoc `ModuleList` slicing. transformers prints an
  "UNEXPECTED" report for the discarded layers 21–27; that's the *expected*
  confirmation truncation worked.
- **Helpers** in `nla/arch.py`: `truncate_config_layers`, `inner_transformer`,
  `strip_lm_head`, `strip_final_norm` (distilled from repo
  `models.py`/`arch_adapters.py`).
- **Affine bias**: repo value head is bias-free; blog writes `A@x+b`. Defaulted
  to no bias (`NLAConfig.ar_affine_bias=False`) to match the repo, exposed as a
  knob. [[nla-blog-quote]]

**Correctness check (passed):** AR's raw layer-l residual is **bit-identical**
(max abs diff 0.0) to the AV full model's `hidden_states[21]` on the same input —
validates truncation + norm-strip + the `l+1` indexing end to end. Raw `h_l`
final-token L2 ≈ 832 (useful later for `injection_scale`).

Deferred (explicitly "later" per user): special-token injection, activation
extraction module, affine-map context translation, MSE training loop. The class
only *constructs* AV+AR for now.

Indexing note: the repo confirms our convention exactly — datagen
`layer_index=K` hooks `layers[K]` = `hidden_states[K+1]`, AR keeps `K+1` blocks.

### D11 — Math eval datasets + smoke test
Added `reasoning_attention.data.math_datasets` (single module, 3 datasets) and a
temporary smoke script. Structures confirmed via the HF Dataset Viewer skill:
- **openai/gsm8k** — config `main` only; train (7473) + test (1319) combined into
  one dataset under a `split` column (8792 rows). `answer` is a CoT ending in
  `#### <final>`; we parse and keep only the final value (e.g. "72"), stripping
  commas. (The socratic config is ignored per "one subset".)
- **opencompass/AIME2025** — both subsets `AIME2025-I` + `AIME2025-II` (all
  subsets), test split, concatenated (30 rows), `subset` column kept.
- **math-ai/amc23** — config `default`, single test split (40 rows). Extra cols
  (id, url) dropped during normalization.

Unified schema for all three: `question, answer, source, subset, split,
messages`. **`answer` is a top-level key**, and **`messages` is a single USER
turn only — the answer is never added as an assistant turn** (per request), so
the model must generate it. `render_prompt()` applies the chat template
(thinking mode, add_generation_prompt) when a tokenizer is available; `messages`
itself stays tokenizer-agnostic.

**LLM sampling params**: live in `config.SamplingDefaults` (= `VLLMConfig.sampling`):
temp 0.6 / top_p 0.95 / top_k 20 / max_tokens 8192 (Qwen thinking-mode rec).

**Smoke script** `scripts/smoke_llm_dataset.py`: loads one record, runs it
through the **plain-transformers model** (`model.loader.load_model`, NOT vLLM —
per user request), prints the newly generated continuation with
`skip_special_tokens=False` (so `<think>…</think>` and the trailing `<|im_end|>`
show verbatim). Sampling pulled from `config.SamplingDefaults`
(do_sample + temp/top_p/top_k/max_new_tokens). Verified on gsm8k[0] → model
reasons in a think block and outputs `\boxed{72}` == gold.

Note: using transformers (not vLLM) also sidesteps the headless-GPU 0.95
utilization issue — `device_map="auto"` coexists fine with the ~840 MiB display
allocation, so no `--gpu-mem-util` knob is needed here. (The `VLLMConfig`
0.95 default remains for the actual serving point.)

Added `datasets>=2.19` dep (resolved to 4.8.5).

### D12 — `<INJECT>` placeholder + AV injection mechanism + prompts
Recreated the reference repo's injection mechanism and BOTH prompt templates.

Found in the repo:
- **`<INJECT>` placeholder**: `nla/schema.py:INJECT_PLACEHOLDER = "<INJECT>"`. It's
  the literal slot in the AV prompt (inside `<concept>...</concept>`), swapped at
  load time for the real single-token injection char (their ㊗), whose embedding
  is then overwritten with the activation.
- **AV/actor template** + **AR/critic template**: verbatim defaults in
  `nla/datagen/stage3_build.py` (`_DEFAULT_ACTOR_TEMPLATE`,
  `_DEFAULT_CRITIC_TEMPLATE`). AV asks for 2-3 snippets inside `<explanation>`
  tags around a `<concept>{injection_char}</concept>` slot. AR is suffix-anchored:
  `Summary of the following text: <text>{explanation}</text> <summary>`.
- **Injection scale**: `nla/schema.py:normalize_activation` scales the vector to a
  target L2-norm; default `sqrt(d_model)` ("ambient residual-stream scale"),
  `None` → raw.

Recreated (single-GPU, no FSDP/distributed):
- `nla/prompts.py`: `AV_TEMPLATE`, `AR_TEMPLATE` (verbatim), `INJECT_PLACEHOLDER`,
  `build_av_messages`, `build_ar_prompt`, `wrap_explanation`,
  `extract_explanation`.
- `nla/injection.py`: `inject_at_placeholder` (port of `inject_at_marked_positions`,
  optional neighbor check, count assertion) + `normalize_activation`.
- `NLA.verbalize(activation)`: builds AV prompt with our placeholder token
  (`<|image_pad|>`), embeds the prompt, overwrites the placeholder row with the
  scaled activation, generates from `inputs_embeds`, optionally returns just the
  `<explanation>` payload.
- `NLAConfig.injection_scale` now matches repo semantics ("sqrt_d_model" default /
  None=raw / float) via `resolve_injection_scale`; dropped the redundant
  `l2_normalize` flag (scaling to a target norm subsumes it).

**Verified end-to-end**: placeholder → exactly 1 token; real `h_l` (norm 963.6)
extracted from a snippet, scaled to √2048≈45.3, injected; AV generated a valid
`<explanation>…</explanation>` with 2-3 snippets.

**Caveat (expected)**: the AV is **untrained**, so the explanation echoes the
prompt framing rather than truly decoding the vector — the *mechanism* is
correct and well-formed, but meaningful verbalization needs AV-SFT (deferred).
flake8: added a per-file `E501` ignore for `prompts.py` (verbatim templates must
not wrap).

### D6 — vLLM params
All throughput params from the spec are passed verbatim to `LLM(...)`. They're
also surfaced as a dataclass default so callers can override individually
without losing the tuned defaults.

### D13 — Tooling migration: ruff replaces isort/black/flake8/autoflake
Supersedes D7's toolchain (D7 itself is kept as history). `make format` is now
`ruff check --fix` + `ruff format`; `make lint` is `mypy` + `ruff check` +
`ruff format --check`. Config moved into `pyproject.toml` under `[tool.ruff]`
(line-length 100, target py311, rules `E,W,F,I,UP,B` — `I` subsumes isort, `F`
subsumes autoflake's unused-import/variable pass). `.flake8` deleted; its E501
per-file ignore for `nla/prompts.py` is now a `[tool.ruff.lint.per-file-ignores]`
entry, joined later by the same exemption for `datagen/prompts.py`.
Also fixed: `requires-python` said `==3.14.*` while `uv.lock`, `.python-version`
and mypy all said 3.11 — set to `==3.11.*`. 4 autofixes applied to `src/`
(`Callable`/`Sequence` from `collections.abc`, one redundant quoted annotation)
and two files reformatted.

### D14 — `AV_TEMPLATE` drift caught and reverted
`AV_TEMPLATE` had lost the words "a meticulous AI researcher" from its opening
sentence, so it was no longer verbatim against the reference repo's
`stage3_build._DEFAULT_ACTOR_TEMPLATE` (490 vs 520 chars). Restored. Verified by
byte-comparison: the only remaining difference is the documented
`{injection_char}` → `{placeholder}` rename (520 vs 517 chars = exactly that
3-char delta). `AR_TEMPLATE` was and is byte-identical.
Why this matters beyond tidiness: the AV prompt is part of what the model is
SFT'd and GRPO'd against, and the paper's FVE numbers are tied to their prompt.
Silent divergence here is a confound in any comparison to the paper.

### D15 — Explanations come from an API model, not a local one
Supersedes the `DataGenConfig` / `Qwen/Qwen3-30B-A3B-Instruct-2507` plan from the
original scaffolding: explanation authoring now runs through
**`gpt-5.6-luna` at `reasoning_effort="high"`** over the OpenAI Responses API.
`DataGenConfig` was replaced by `ExplainerConfig`; `DATAGEN_MODEL_ID` by
`EXPLAINER_MODEL_ID`.
- **Key handling**: `.env` copied from `SELF/lever/LEVER/.env` (holds only
  `OPENAI_API_KEY`), `chmod 600`, and `.env` added to `.gitignore`.
- **Reasoning models reject `temperature`/`top_p`**, so neither is exposed —
  `reasoning_effort` is the only quality knob. `max_output_tokens=4096` because
  the cap covers reasoning tokens *plus* the visible answer; a response that hits
  it comes back `status="incomplete"` and is dropped (no closing tag).
- **Error policy** mirrors the reference `AnthropicProvider`: rate-limit /
  timeout / connection / 5xx degrade to `None` (row dropped, count logged);
  anything else — auth, bad request, unknown model — raises, so a broken key
  can't masquerade as a data problem.
- Verified live: `gpt-5.6-luna` is present on the key, and
  `scripts/smoke_gpt_endpoint.py` produced a well-formed 3-feature, 57-word
  explanation.

### D16 — Warm-start datagen pipeline (`datagen/`, 4 stages)
Implements the paper's Stage-1 recipe as specified: **100k Ultra-FineWeb
documents × 5 random positions ≈ 500k `(context, summary, h_l)` pairs, split
evenly BY DOCUMENT into two disjoint halves** (`D_AV` ≈ 250k, `D_AR` ≈ 250k). If
`(h_17, s_17) ∈ D_AV` the AV trains `h_17 → s_17` and the AR never sees that
pair; the AR trains `s_j → h_j` on its own half.
Stages: `extract` → `split` → `explain` → `build`, plus `sidecar.py` (the
datagen↔training contract) and `providers.py`/`prompts.py`.
- **Corpus id**: `openbmb/Ultra-FineWeb`, config `default`, split `en`, text in
  the **`content`** column (not `text`). The `openbmb/UltraFineWeb` spelling is a
  redirect and the dataset-server rejects it.
- **Streaming is mandatory.** A non-streaming `load_dataset` on the `en` split
  downloads the whole ~1 TB corpus before yielding document 1 — it pulled 47 GB
  in ten minutes before being killed. `extract` now uses
  `streaming=True` + `.skip()`/`.take()`, so a 100k-doc slice costs ~those
  documents' bytes. (The 47 GB partial cache is still in
  `~/.cache/huggingface/hub/datasets--openbmb--Ultra-FineWeb` and is safe to
  delete — streaming does not read it.)
- **Split by document, never by row.** Stage 0 draws 5 positions per document, so
  a row-level split would put position 2 of a document in one half and position 4
  in the other; those contexts share a prefix, which leaks across the boundary.
  `partition_documents` sorts before shuffling because set iteration order varies
  with the hash seed — without `sorted()` the same `--seed` gives different splits
  across runs.
- **Invariants carried from the reference repo**: raw unnormalized vectors
  (`norm="none"` in the sidecar); per-document keyed RNG on `(seed, doc_id)` so
  the same doc yields the same positions regardless of slice/chunk/process count;
  `min_position=50`; right-padding AND right-truncation (left-padding would
  return pad-position activations, left-truncation would misalign every index);
  `FixedSizeList` for `activation_vector` (variable-length lists overflow int32
  offsets and silently corrupt `take()` past a 4 GiB values buffer);
  forward hook on the single target layer instead of `output_hidden_states=True`.
- **AV rows** carry the constant AV prompt with the literal `<INJECT>` (swapped
  for the real placeholder at load time, so the dataset survives repointing the
  placeholder) + `<explanation>`-wrapped response. **AR rows** carry the
  suffix-anchored prompt ending `</text> <summary>` and no response — the AR is a
  regressor here; `ar_suffix_ids` ([522, 1318, 29, 366, 1708, 29]) go in the
  sidecar so training can verify the tail before extracting at `tokens[-1]`.
- **Resumability**: `explain` writes per-chunk parquets and skips completed ones
  on restart (tmp+rename, so a kill mid-write can't leave a half-file that looks
  complete). At 500k rows the API spend is the expensive part.
- **Verified end-to-end on 20 docs**: 100 rows (=20×5), `h_l` L2 in 607–1052,
  halves 10/10 docs with an empty intersection, 6+6 real explanations, and final
  AV/AR parquets with the expected shapes.

**Known GIL-teardown noise**: `extract` prints
`PyGILState_Release: thread state ... must be current when releasing` at
interpreter shutdown, *after* the parquet and sidecar are written. Output is
intact (validated). It comes from the streaming dataset's aiohttp threads at
finalization, not from the extraction path.

**Not built yet**: the SFT training loops themselves (AV `h→s`, AR `s→h`), the
FVE metric, and Stage-2 joint RL (GRPO). D16 delivers the dataset those consume.

### D17 — Label inspection script (`scripts/dump_explanations.py`)
Deleted the 47 GB partial Ultra-FineWeb cache (per D16; streaming never reads it).
Added a script that writes the labeling model's output down in readable form:
`.jsonl` (one record per row), `.md` (context tail → features, shortest first),
and `.stats.json` (status counts, features/summary, word counts, `h_l` norm
range). Two modes: `--from-parquet` re-reads an explained parquet for free, and
`--from-base` calls the labeling model on un-labeled rows *without* writing a
training parquet — for judging label quality before paying for 500k of them.
Records carry a `status` field (`ok` / `no_completion` / `no_tags` /
`too_few_features`) so rejects show up in the dump instead of being silently
filtered; a 30% drop rate should be visible before the full run, not after.
Verified on 6 rows: 6/6 ok, 3 features each, 56-85 words, `h_l` norms 774-994.

### D18 — SFT trainer (`training/`) + launcher, reconciled against the reference
Built `training/data.py` (parquet datasets + collators) and `training/sft.py`
(the loop), plus `scripts/train_sft.sh`. Hyperparameters are deliberately untuned
placeholders. **Four correctness fixes came from reading the reference repo's own
SFT stages** (`nla/rollout/sft_actor.py`, `sft_critic.py`, `nla/loss.py`) — each
was wrong in my first pass:

1. **The AR MSE normalizes BOTH sides, not just the target.** `nla/loss.py`:
   "if a float, BOTH pred and gold are L2-normalized to that norm — direction-only
   MSE." I had been comparing a free-magnitude prediction against a fixed-norm
   target, which makes the loss dominated by norm error rather than direction.
2. **AR tokenization uses `add_special_tokens=True`** to match the extractor that
   produced the gold activations. The reference flags this explicitly: without it
   the backbone runs OOD and its layer-l means drift from the gold regime (they
   measured init `cos(mu_backbone, mu_gold)` ~0 vs ~0.9+). Qwen3 has
   `bos_token=None` so it is a no-op here, but it is the correct setting and
   survives a model change.
3. **The AV must be chat-templated.** The reference's mask generator templates the
   messages and supervises only the assistant turn. More importantly
   `NLA.verbalize()` calls `apply_chat_template(add_generation_prompt=True)` at
   inference, so training on the bare prompt string would have the model meet the
   `<|im_start|>assistant` scaffolding for the first time at eval.
4. **FVE uses a fixed dataset-level denominator**, not a per-batch variance —
   the reference carries it as `nla_baseline_rawvar` and reports
   `1 - mse/baseline`. A per-batch variance moves as batch composition changes and
   is not comparable across steps.

Also adopted from their `configs/TRAINING_NOTES.md`: warmup 5% (they use 50/1000
iters), cosine decay, and their production LR of **2e-5 at effective batch 256**
with the documented **sqrt(batch/256) scaling rule** for Adam-family optimizers.
`--full-finetune` re-derives the LR from that rule instead of using the LoRA
default. Their reported actor loss trajectory is 4.4 (step 0) → 2.9 (warmup end)
→ 1.5 (step 300); our AV starts at **4.67**, which is the sanity check that the
setup matches.

- **LoRA is the default because full fine-tuning does not fit.** Qwen3-1.7B bf16
  with AdamW needs ~3.4 GB weights + 3.4 GB grads + 13.8 GB fp32 moments ≈
  **20.7 GB against 16 GB** on this box. LoRA (r=16, the standard
  q/k/v/o/gate/up/down target set) trains 17.4M params = 1.0%.
  `--full-finetune` exists for a bigger GPU and will OOM here. The AR's affine map
  is always trained in full and in fp32 — it is newly initialized and has no
  pretrained weights to adapt.
- **Logged `pred_norm`** because the reference warns a direction-only loss is
  norm-neutral in activation space but *not* in weight space: `|pred|` drifts
  upward roughly linearly with steps under Adam. Their named mitigations are a
  lower LR or an explicit norm term.

Two bugs found by running it:
- **`inner_transformer` could not see through PEFT.** It assumed one `.model`
  hop, but a LoRA-wrapped backbone nests as
  `PeftModel.base_model.model.model`, so one hop landed on the causal LM, which
  has no `.layers`. Now descends until a module actually owns `.layers`. Safe for
  LoRA: PEFT replaces target submodules in place, so adapters still run when the
  inner module is called directly. **The bit-identity regression check still
  passes (max abs diff 0.0)** after the change.
- **fp32 affine vs bf16 backbone** raised a dtype mismatch; `ARModel.forward` now
  casts the residual into the affine's dtype instead of assuming they match.
  Also `load_activations` now copies the arrow buffer — `torch.from_numpy` on a
  read-only array is undefined behaviour, not an error.

Verified end-to-end on the 6-row smoke set: AV loss 4.67 → 3.79 with falling
perplexity; AR loss 2.02 → 1.34 with FVE rising −3.19 → −1.77 (negative is
correct at 6 rows with a randomly initialized affine). Checkpoints save as a PEFT
adapter for the AV, and `backbone/` + `affine.pt` for the AR (ARModel is not a
`PreTrainedModel`, so the two halves are saved separately and reload
independently). `train_log.json` records args, per-step metrics, and the eval
summary.

**Still not built**: Stage-2 joint RL (GRPO with `r = -log‖h_l - AR(z)‖²` and the
KL penalty to the AV init), and the doom-loop study itself.

### D19 — SFT hyperparameters copied from their Qwen2.5-7B case study
Source: `natural_language_autoencoders/configs/TRAINING_NOTES.md` (+ `actor_sft.sh`,
`critic_sft.sh`). Copied everything **except batch size** (theirs is 256 on
2xH100-80GB). Their caveat is carried into our docstrings verbatim in spirit:
"these are the settings we used, not settings we claim are optimal — we did not
sweep batch size, learning rate, or GRPO group size."

- `lr 2e-5`, `min_lr 2e-6`, cosine, warmup 5% (their 50/1000 iters). Same LR for
  AV and AR ("matched to actor — worked well").
- **`min_lr` needed a custom scheduler.** `transformers.get_cosine_schedule_with_warmup`
  decays to exactly 0; theirs floors at LR/10. Added `cosine_with_floor`.
- **Identity-init the AR affine map** — their ⚠️ *Critical* note. `nn.Linear`'s
  kaiming default scales the backbone output norm by ~1/sqrt(3), giving them
  `pred_norm ~48` against `backbone_norm ~83` at step 0 and an initial loss of
  1.94 vs 1.61 with the identity. **Confirmed on our smoke set: step-0 AR loss
  dropped 2.02 -> 1.42 (~30%).** The identity is the right prior anyway — the AR
  reads the same residual stream the AV wrote, so "change nothing" beats a random
  rotation. `ARModel(identity_init=True)` by default.
- **FVE denominator replaced.** Was the target variance; is now the *achievable
  predict-the-mean loss* (`predict_mean_baseline`), which is their
  `load_predict_mean_baselines`. Because the loss normalizes the prediction too, a
  signal-free model can only emit one constant direction, so that is the honest
  floor. Their numbers: baseline 0.938, trained 0.586 -> **FVE 37.5%**; their
  shuffled-target control scored 0.922 ~= baseline, as it should.
- `--save-interval 500` now actually checkpoints mid-run (`step_%07d/`).
- `--attn-implementation` exposed: theirs is `flash_attention_2` for the AV and
  `sdpa` for the AR, **without** gradient checkpointing (m16 fits and is 36%
  faster; it also deadlocks NCCL in their RL `update_weights()`). FA2 needs the
  `flash-attn` package, so the flag defaults to unset.
- **`injection_scale` left at our `sqrt_d_model`, not copied.** Theirs is 150 for
  Qwen2.5-7B (d_model 3584, sqrt = 59.9), so 150 is ~2.5x sqrt(d) — an absolute
  value, not a ratio. At our d_model 2048, copying the number gives 150 while
  copying the ratio gives ~113. Exposed as `--injection-scale` /
  `INJECTION_SCALE=` with both readings documented; **this one wants a sweep, not
  a guess.** Related: measured Qwen3 token-embedding norms are ~1.54, so even 45.3
  is already ~30x the ambient embedding scale.

### D20 — Stage-2 RL data: 200k web + 200k chat (scaled down from 500k)
`RLDataConfig`: **40k documents x 5 positions = 200k activations per source**,
400k total — a quarter of the paper's 500k-per-source, per the smaller run. Two
sources: Ultra-FineWeb prose and `allenai/WildChat-1M` dialogue.
- **No API spend.** RL needs no summaries: the AV generates the explanation during
  rollout and the AR scores it. `build.py --stage rl` runs straight off a base
  parquet.
- **WildChat needed a new corpus kind.** Its text lives in a `conversation` column
  as `{role, content}` turns, so `extract.py --corpus-kind chat` renders each
  conversation through the *target model's* chat template before sampling
  positions — activations then come from dialogue shaped the way the model
  actually sees it. Empty renders are dropped before the forward.
- **The web slice starts at document 100000**, past the warm-start slice, so RL
  activations come from documents the SFT stage never saw.
- `datagen/merge.py` concatenates and **shuffles at row level**. Row-level is
  correct here (unlike the SFT split): there is no AV/AR boundary to leak across
  and every row is an independent RL prompt. Unshuffled, a constant-LR run would
  see 200k web rows then 200k chat rows — an unintended curriculum.
- A `source` column is carried through so per-corpus behaviour stays auditable.
- Verified end to end on 8+8 documents: 40+40 rows, merged to 80, RL parquet has
  no `response` column and an even source split.

### D21 — RL stack goes in a SEPARATE env, not the project venv
The request was to add Miles and SGLang to the venv. **Doing that would break this
box.** `uv pip install --dry-run 'sglang[all]'` against our venv resolves to:

    torch        2.11.0+cu130 -> 2.9.1   (a cu12 build)
    transformers 5.13.0       -> 4.57.1
    torchvision / torchaudio / triton      all downgraded

cu12 wheels carry no sm_120 kernels, so torch stops working on the RTX 5070 Ti
(capability 12.0) — extraction, SFT, and every smoke test die — and vLLM 0.22 pins
torch 2.11, so it breaks too. The reference's own `docs/setup.md` says the same:
"An unpinned `pip install torch` may pull a cu130 build, which conflicts with
`sgl-kernel`'s cu12 wheels." Since the RL run targets 2xA100 (sm_80) where cu12 is
fine, the stack does not need to be here at all.

`scripts/setup_rl_stack.sh` therefore builds `.venv-rl` (upstream does the same
thing via `build_conda.sh`): torch from the cu124 index first, then Miles cloned
at the pinned commit **`radixark/miles@051cd15`** with
`nla/miles_patches/*.patch` applied (checking the pin out first is what makes
`git apply` succeed), then SGLang from a source checkout with
`patches/apply_sglang_patches.sh` — a wheel will not do, since training needs the
bf16-base64 transport, chunked-prefill slicing, and retract-path KV fixes. Ends
with `import miles, sglang, nla`. `--check` reports status without changing
anything. **Not executed yet — awaiting the go-ahead.**

### D22 — `scripts/train_grpo.sh`, their rl.sh adapted to 2xA100 + FSDP
Their hyperparameters kept: GRPO group `n-samples-per-prompt 8`, KL loss coef
**0.01** (`--use-kl-loss`; `--kl-coef` is a no-op for GRPO because
`get_grpo_returns` discards the kl tensor), response cap **150** (uncapped, a hot
critic rewards verbosity and response length drifts 123 -> 165+), constant LR,
`--sglang-disable-radix-cache`, `round_robin` router with the circuit breaker off,
`--group-rm` reward path, and `NLA_EMBED_DUMP_DIR=/dev/shm/nla`.
Changed for our hardware:
- **GPU layout**: theirs is 8 actor + 4 critic + 4 rollout = 16. On 2 GPUs all
  three roles must colocate (`ACTOR_GPUS=CRITIC_GPUS=ROLLOUT_GPUS=2`); Ray hangs
  on placement if the roles request more GPUs than exist.
- **Batch**: theirs is 128 prompts x 8 = 1024. Ours defaults to 16 x 8 = 128, with
  the LR **sqrt-scaled off their 1.41e-5 @ 1024** per their own rule. GRPO group
  size stays 8 — it is the advantage baseline, not a throughput knob.
- Warns if `/dev/shm` < 8 GiB (they write ~1 GB of embedding dumps per step).
- Keeps `--gradient-checkpointing` OFF, with their reason inline: it deadlocks
  NCCL in `update_weights()` (FSDP full-param gather changes, broadcast hangs,
  10-minute watchdog SIGABRT).

Note their radix-cache comment is a correctness issue, not a perf one, and is
reproduced in our script: the cache keys on token IDs, but we inject a *different*
activation at the same marker token every call, so a cache hit silently returns
another activation's output.

### D23 — Cluster dev pod (`pod.yaml` + Makefile targets)
Copied from `UNI/MASTERS/pod.yaml` (namespace `vitenko-thesis`, node
`malea-srv01`, `pytorch/pytorch:2.10.0-cuda13.0-cudnn9-devel`, the sshd + authorized_keys
block). Changes:
- **PVC is `doomloops-interp`** (1000Gi, longhorn, RWO) mounted at `/workspace`.
- **oh-my-zsh installed with `--unattended`.** Without that flag the installer runs
  `chsh` (which prompts) and then execs zsh, so the container script would never
  reach `sleep infinity` — the pod would come up Ready and hang. Guarded on
  `[ ! -d /root/.oh-my-zsh ]` so a restart doesn't reinstall.
- **`/dev/shm` raised to 8Gi** via an `emptyDir{medium: Memory}`. The container
  default is 64 MB; the reference RL loop writes ~1 GB of embedding dumps per step
  and their setup docs call this out explicitly.
- **`HF_HOME` and `UV_CACHE_DIR` moved onto the PVC.** The container layer is
  ephemeral, and model weights plus dataset shards are GBs that would be
  re-downloaded on every restart.
- **2 GPUs on `malea-srv01`**, pod named `reasoning-interp`. `malea-srv02` was
  requested but is **not usable**: it is Ready and untainted yet exposes no
  `nvidia.com/gpu` at all, so a GPU pod pinned there would sit Pending forever.
  Verified cluster GPU inventory: `malea-srv01` 8x A100-SXM4-80GB, `ki-srv01` 2x
  A100-80GB-PCIe, `ki-srv02`/`ki-srv03` 1x A100-PCIE-40GB each, everything free
  (12 GPUs idle, nothing running in `vitenko-thesis`). Chose `malea-srv01` over
  `ki-srv01` — the only other 2-GPU option — because SXM4 gives NVLink between the
  cards and FSDP is all-gather bound. RWO on the claim still means one pod at a
  time.
- Nested-heredoc bug caught before shipping: a `cat <<'RC'` inside the YAML block
  scalar can't terminate, because its delimiter ends up indented. Replaced with
  `printf`. The container script is now extracted and `bash -n`-checked.

`make start/stop/connect/forward/pod-logs` mirror the MASTERS Makefile and reuse
that repo's kubeconfig path. Added `make pod-secret` to push `OPENAI_API_KEY` in
as a k8s secret rather than baking it into an image.

**Server-validated.** The first attempt used
`UNI/MASTERS/kubeconfig.yaml`, whose token had expired (403,
`system:unauthenticated`). A fresh `kubeconfig.yaml` now lives in this project and
works; the Makefile points at `$(CURDIR)/kubeconfig.yaml` and it is gitignored
(bearer token). `kubectl apply --dry-run=server` returns
`pod/reasoning-interp created (server dry run)`, and `doomloops-interp` shows
Bound, 1000Gi, RWO, longhorn.

### D24 — `scripts/build_warmstart_data.sh` + where Stage-1 data lands
There was no single entry point for the Stage-1 pipeline (only the per-stage
modules and the error message in `train_sft.sh`). Added one. Nothing in the
pipeline writes to an implicit location — every stage takes `--output`, and the
script centralizes the layout under `$DATA_DIR` (default `data/warmstart`,
point it at `/workspace/...` on the cluster).

Measured sizes, extrapolated from real parquets rather than estimated:

| file | rows | bytes/row | size |
|---|---|---|---|
| `base.parquet` | 500k | 4118 | 2.06 GB |
| `halves/{av,ar}_half.parquet` | 250k each | 4118 | ~1.03 GB each |
| `av_sft.parquet` | 250k | 5407 | 1.35 GB |
| `ar_sft.parquet` | 250k | 5124 | 1.28 GB |
| `rl.parquet` (stage 2) | 400k | 3272 | 1.31 GB |

~9 GB for Stage 1 with intermediates, ~2.6 GB if the intermediates are deleted.
Trivial against the 1000Gi PVC. The `explain` stage additionally keeps
`{output}.chunks/chunk_*.parquet` — that directory is the resume state and the
only thing standing between a crash and re-paying for the API calls, so it must
live on the PVC too, not in a container-local temp dir.

### D25 — Pinned manifest for the RL env (`requirements/rl.in` + `rl.txt`)
The RL stack had only an imperative installer, so the env was reproducible from
whatever resolved on the day. Added `requirements/rl.in` (direct deps) compiled to
`requirements/rl.txt` — a full 681-line resolution produced with:

    uv pip compile requirements/rl.in --python-version 3.11 \
      --extra-index-url https://download.pytorch.org/whl/cu124 \
      --index-strategy unsafe-best-match -o requirements/rl.txt

Resolved pins confirm the earlier dry-run: **torch 2.9.1, torchvision 0.24.1,
torchaudio 2.9.1, sglang 0.5.9, sgl-kernel 0.3.21, transformers 4.57.1,
triton 3.5.1, ray 2.58.0, numpy 2.4.6.** That torch is a cu12 build — correct for
the A100 (sm_80) target and exactly why this must not touch the project venv
(cu12 has no sm_120 kernels for the local Blackwell card).

Miles, SGLang, the reference `nla` package and this project are deliberately *not*
in the manifest: all four install editable from patched git checkouts, so pip
cannot pin them. `setup_rl_stack.sh` still owns those steps and now installs the
wheel deps from the manifest, falling back to a fresh resolve with a warning if
the file is missing. `make rl-pins` regenerates it.

**Pod portability fixes found while doing this**: the
`pytorch/pytorch:...-devel` image has pip but **no uv**, so the setup script now
bootstraps uv (and `pod.yaml` installs it at boot). `pod.yaml` also adds
`build-essential` for the flash-attn source build and exports `CUDA_HOME` plus
`$HOME/.local/bin` on PATH in `.zshrc`. `make rl-setup` / `rl-check` wrap the
script.
