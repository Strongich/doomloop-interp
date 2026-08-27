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

### D26 — Labeling with a *thinking* model: uncapped output, strip the CoT

Labeling 200 rows against a locally served `Qwen/Qwen3.8-27B` dropped every row
with `AssertionError: every row was dropped`. Three independent bugs, all worth
recording because each one masked the next.

**1. The output cap truncated the chain of thought.** `CHAT_MAX_OUTPUT_TOKENS`
was set to 512 on the reasoning that a local chat model "emits no hidden
reasoning tokens, so the answer alone is ~150-250 tokens". That is false for
Qwen3, which serves in thinking mode by default: a measured response is **1511
completion tokens, ~695 words of which are thinking**. The cap cut the response
off mid-reasoning, so no `<analysis>` tag was ever emitted and every row failed
extraction. The cap is now `None` (uncapped) for `api_kind == "chat"`. The prompt
already bounds the *answer* to ~80-100 words; it is the thinking that must not be
truncated.

**2. Extraction matched a draft inside the reasoning.** vLLM here runs with no
reasoning parser (`reasoning_parser=''`), so the CoT is inline in `content`
rather than split into `reasoning_content` — and the chat template pre-fills the
opening `<think>`, so the response *begins* in reasoning and only the closing
`</think>` appears. The model routinely writes a trial `<analysis>` block while
planning, and `ANALYSIS_RE` matched that first, yielding 7 features of scratch
work ("Need produce final with..."). `extract_and_clean` now discards everything
through the final `</think>` before searching. This also makes the code correct
if a reasoning parser is ever enabled: `content` is then already answer-only and
the split is a no-op.

**3. An all-dropped chunk poisoned the resume cache.** `explain.py` wrote the
zero-row chunk to `.chunks/` anyway, and the resume check is
`if chunk_path.exists()`. A zero-row chunk is indistinguishable from a completed
one, so every later run "resumed" past it and merged nothing — the failure
resurfaced as a parsing assertion long after the real cause (the 400, then the
cap) had scrolled off. The stage now **refuses to cache a chunk that kept no
rows** and aborts on the first one, printing the raw text of up to 3 unusable
responses. Losing the raw responses was itself the reason diagnosis required
re-serving a 27B model just to see one completion.

**Throughput consequence.** `reasoning_effort` defaults to `xhigh` on this model.
Measured on one row: xhigh 5417 tok / 111 s, medium 1398, low 838, and thinking
disabled 85 tok/row. All produce well-formed 3-feature explanations; the
difference is grounding detail, not format. `chat_reasoning_effort` and
`chat_enable_thinking` are now plumbed through `ExplainerConfig` ->
`OpenAIProvider._one_chat` (via `extra_body.chat_template_kwargs`), exposed as
`--reasoning-effort` / `--no-thinking`. Defaults leave the server alone, so the
choice is explicit at the call site.

### D27 — Verify model shards before serving (`scripts/verify_weights.py`)

An interrupted **Xet** download reconstructs shards in place under their final
names instead of staging `blobs/*.incomplete`, so a killed transfer leaves
truncated files that the HF cache reports as present. All 18 shards of the 27B
were short — 14 GB on disk against 51.7 GiB upstream — and an earlier
"18 shards, 0 partials" check passed because it counted *files*. The only symptom
was `SafetensorError: incomplete metadata, file not fully covered`, five minutes
into engine startup, behind a `RuntimeError: Engine core initialization failed`.

Two fixes: re-download with `HF_HUB_DISABLE_XET=1` so partials stage as
resumable `.incomplete` blobs, and a preflight in `label_and_shutdown.sh` that
parses every cached shard header (cheap, offline) and exits with the re-download
command. Verified it fires on a deliberately truncated shard and passes on a
healthy model — a guard that never fires is worthless.

### D28 — Local labeling runs in non-thinking mode with Qwen3's sampling recipe

Settled after measuring all four modes on 192 real rows against the served 27B:

| mode | rows/s | usable | 200k ETA |
|---|---|---|---|
| thinking off | 5.55 | 88.5% | 10 h |
| effort low | 1.79 | 99.5% | 31 h |
| effort medium | — | — | ~38 h (extrapolated) |
| effort xhigh (Qwen3 default) | — | — | ~147 h (extrapolated) |

**Non-thinking is the faithful choice, not merely the cheap one.** The reference
labels with `claude-sonnet-4-6` through the Anthropic Messages API at
`max_tokens=300, temperature=1.0, concurrency=32` and passes **no `thinking`
parameter** (`nla/datagen/providers.py:67`). That is deliberate rather than an
omission: its response handler asserts `len(resp.content) == 1 and
resp.content[0].type == "text"`, which an extended-thinking response — a
`thinking` block plus a `text` block — would fail immediately. The reference also
tolerates `stop_reason == "max_tokens"` and lets truncated responses fail the
`</analysis>` regex, so **it drops rows too**; our ~11.5% drop rate at
`enable_thinking=False` is the same trade, not a regression.

Sampling follows Qwen3's published Instruct-mode recipe: `temperature=0.7`,
`top_p=0.80`, `top_k=20`, `min_p=0.0`, `repetition_penalty=1.0` — but with
**`presence_penalty=0.0` instead of the recommended 1.5**. Measured over 192 rows:

| presence_penalty | usable | no `<analysis>` tag | `<2 features` |
|---|---|---|---|
| 1.5 (recipe) | 70.3% | 0 | 57 |
| 0.0 | 90.1% | 0 | 19 |

The recipe is tuned for open-ended chat, where discouraging repetition improves
output. Here the repetition *is* the deliverable: the prompt asks for three
parallel feature lines inside one `<analysis>` block. The failure signature makes
the mechanism unambiguous — the tag is emitted in 100% of responses under both
settings, and every extra failure is `<2 features`, i.e. the penalty flattening
the list rather than breaking the format. Throughput is unaffected (5.9 vs 6.1
rows/s), so this is ~20 points of yield for free. Note the split — `temperature`, `top_p` and
`presence_penalty` are standard OpenAI chat fields, while `top_k`, `min_p` and
`repetition_penalty` are vLLM extensions that must travel in `extra_body`
alongside `chat_template_kwargs.enable_thinking`. All of it applies to the local
chat path only; the hosted reasoning API rejects `temperature`/`top_p` outright.

`chat_enable_thinking` therefore defaults to **False**, and `--reasoning-effort`
flips it back on (an effort level is meaningless with thinking disabled).

Quality at 20 rows, thinking off vs effort low: 19/20 vs 20/20 usable, 15 vs 19
summaries with exactly 3 features, mean 54 vs 74 words against a prompt asking
for ~80-100. The gap is detail, not format.

### D29 — `injection_scale` = 1000, from the reference's stated rule (not sqrt(d))

Reverses the earlier default of `sqrt_d_model` (≈45.3 at d_model 2048), which
was chosen by reading `resolve_target_scale` and taking its fallback for the
recipe. It is not the recipe. `docs/inference.md` states the rule outright:

> `injection_scale` is picked as a round number a bit above the mean norm of
> the dataset's vectors.

Their published values follow that and nothing else — no relation to `d_model`:

| model | d_model | layer | injection_scale | sqrt(d_model) | mean ‖h‖ |
|---|---|---|---|---|---|
| Qwen2.5-7B | 3584 | 20 | 150 | 59.9 | ~125 |
| Gemma-3-12B | 3840 | 32 | 80000 | 62.0 | ~74k |
| Llama-3.3-70B | 8192 | 53 | 30 | 90.5 | — |

Gemma is ~500x Qwen because `Gemma3TextScaledWordEmbedding.forward()` multiplies
by sqrt(hidden_size), inflating residual-stream norms; Llama-70B is *below*
sqrt(d). Any "scale = f(d_model)" reading is contradicted in both directions.

Our extracted `h_l` at layer 20 measures 782-1004, mean ≈ 900 (consistent with
the 774-994 seen at extraction), so the round number just above is **1000**.
`sqrt_d_model` would have injected at ~20x below the distribution the AV must
read — the same class of error as injecting raw, in the opposite direction.

No sweep was run (no spare compute); 1000 is the rule's value, not a tuned one.
`--injection-scale` still accepts a float, `sqrt_d_model`, or `raw`, so a sweep
later needs no code change. Note `mse_scale` is a **separate** knob in the
reference (`docs/design.md`): ours is the same `injection_scale` value used for
the direction-only AR loss, where the constant cancels through the mean and only
`None` vs not-None actually changes the objective.

### D30 — `mse_scale` is a separate knob from `injection_scale`

Found by the first SFT smoke run after D29. `ARDataset` and `ar_loss` were both
using `resolve_injection_scale()`, so raising `injection_scale` to 1000 also
raised the AR loss's normalization target — and those are different things.

The AR loss normalizes both sides, so the MSE carries a factor `s^2/d`; it is
exactly 1 only when `s = sqrt(d)` ("s cancels via mean", `nla/loss.py:77`). At
`s=1000, d=2048` the factor is ~488, which showed up as:

| | injection_scale reused | mse_scale = sqrt(d) | reference |
|---|---|---|---|
| fve_baseline | 345.87 | 0.7213 | 0.938 |
| step-1 loss | 600.39 | 1.2558 | 1.61 (step 0) |
| grad_norm | **10671** | 42.3 | — |

FVE survived (it is a ratio) but the gradients did not: `grad_norm ~1e4` against
`max_grad_norm=1.0` means every step is clipped ~10,000x and the effective LR is
whatever survives clipping. It also made the loss incomparable to their published
0.938 / 0.586 numbers, removing the main external check we have.

`docs/design.md` states these are independent, and their values differ in
practice (`injection_scale=150`, `mse_scale` defaulting to `sqrt_d_model`).
`NLAConfig` now has both, sharing one `_resolve_scale` so they accept the same
forms. `AVDataset` uses `injection_scale` (it injects); `ARDataset` and `ar_loss`
use `mse_scale` (the AR never injects — it predicts the vector).

Post-fix step-0 FVE is **-0.741** against their **-0.716** (= 1 - 1.61/0.938),
which is a useful independent-implementation check on the identity init.

**Still to watch**: `grad_norm` is 42 at step 1 and 7.9 at step 2 against
`max_grad_norm=1.0`, so early steps are clipped hard. Expected from identity init
(the affine map starts at exactly the wrong scale for the target), but if it does
not settle below ~1 within the warmup the clip is doing the LR's job.

### D31 — Labeling prompt v2: describe, never quote (the surface-form leak)

The AR warm-start reached **0.5848 held-out FVE**, well above the reference's
0.375. The masked-explanation ablation shows most of that was a shortcut:

| run | FVE @ step 130 |
|---|---|
| `sft-ar` (real) | 0.542 |
| `sft-ar-masked-both` (quotes + final feature stripped) | **0.141** |

**74% of the AR's FVE came from surface form, not semantics.** Measured over 4000
labels from `av_explained.parquet`:

| | |
|---|---|
| explanation contains a quoted span | 97.4% |
| that quote appears verbatim in the context | 82.0% |
| the quote lies in the context's **last 6 words** | 69.8% |

`h_l` is read at the context's final token and the AR is the same base model
truncated, so an explanation naming that token hands over the answer. Two lines
of the reference prompt invite exactly this: *"Feel free to include specific
textual examples inline"* and *"The final feature must describe the very end of
the presented sequence"*.

The paper names this failure — *"the AV could achieve good reconstruction by
reproducing the input context verbatim"* — lists it under limitations, and ships
no control for it. Their only guard is a Stage-2 KL penalty they call a partial
mitigation. The shuffled control does **not** catch it: there the explanation
still describes some real snippet, so both arms lose the hint equally.

**This is a deliberate deviation from the "prompts are verbatim from the
reference" invariant**, and the first place this project departs from their
recipe on purpose. v1 is kept as `EXPLAIN_INSTRUCTION_V1` for provenance and
reproduction. v2 keeps the task, the 2-3 feature structure, the `<analysis>`
tags and the ~80-100 word budget; it forbids reproducing any word, phrase, name,
number or punctuation from the text, and asks the final feature for the *role* of
the ending rather than its identity.

Instructions are not compliance, so `verbatim_overlap()` measures it: a quoted
span occurring in the context, or any shared 5-word shingle. `explain.py` reports
the count per chunk, and `--reject-verbatim` drops those rows.

**Cost**: the existing 177k pairs were labeled with v1 and carry the leak, so
acting on this means re-labeling (~10 h of GPU). The current SFT run and its
controls are being left to finish first — their numbers are still the evidence
for the size of the effect.

**Expect FVE to fall.** The residual 0.141 is the honest semantic signal under
v1; a good v2 should beat that but will not approach 0.58. A lower, leak-free
number is the better result for the doom-loop study, where the question is what
the model represents at a position, not which token sits there.

### D32 — Stage-1 results: v1 vs v2, and what the ablations showed

Warm-start complete on both prompt versions. All numbers held-out, ~2% tail.

| | reference (Qwen2.5-7B) | ours v1 (their prompt) | ours v2 (leak-free) |
|---|---|---|---|
| AR final loss | 0.586 | 0.3022 | 0.5587 |
| AR baseline | 0.938 | 0.7279 | 0.7285 |
| **AR FVE** | **0.375** | **0.5848** | **0.2331** |
| AV loss @ step 300 | 1.5 | — | 1.368 |
| AV held-out | not published | interrupted | 1.3609 (ppl 3.90) |
| `critic_rand` | ~0 (0.922/0.938) | not run | **~0** (-0.07..+0.02 @ step 300) |
| masked ablation | **not run** | 0.141 (-74%) | ~0.12 (-50%) |
| verbatim leak | undocumented | ~84% | **0** (0.56% caught + dropped) |

**Their 0.375 is not comparable to our 0.2331** — theirs used the leaky prompt.
The like-for-like number is our v1 **0.5848**, which exceeds their 0.375 (smaller
model, lower-dimensional target, and our baseline is lower, which makes FVE
harder to score, not easier). So we reproduce their recipe and beat it, then show
~60% of that score was surface-form lookup.

The two controls answer different questions and both are needed:
- `critic_rand` (shuffled pairing) ~0 confirms the AR needs the *correct*
  explanation — it is not scoring off a generic prior.
- The masked ablation confirms *why* it needs it. On v1 removing quotes cost 74%;
  on v2 removing the final-feature paragraph costs ~50%, which is not a leak —
  `h_l` is read AT the final token, so its description is legitimately the most
  informative feature, and the remaining ~50% coming from the other two means the
  signal is distributed rather than a single lookup.

**Epochs back to 1.** Their `--num-rollout 1000` is one pass over their half, and
the pass is what matters, not the step count — D28's 3-epoch reasoning had this
backwards. Measured at 1143 steps: AV train loss 1.3642@500 -> 1.4584@1143, AR FVE
0.2011@580 -> 0.1753@1143, held-out better than final train loss in both. Epochs
1-2 bought nothing and cost mild overfitting. One epoch is ~381 steps.

v1 artifacts are preserved for reproduction: `warmstart/v1_leaky/` (parquets +
sidecars + chunks) and `v1_leaky_checkpoints/`.

### D33 — The case study is recovered-vs-failed, not phrase-vs-phrase

First traces from the target model (`Qwen3-1.7B`, thinking mode, R1-style system
prompt, Qwen sampling temp 0.6 / top_p 0.95 / top_k 20, 8192-token budget) over 5
math problems spanning GSM8K -> AIME difficulty.

Doom looping reproduces exactly as described: both AIME-level problems consumed
the full 8192 tokens, never closed `<think>`, and produced no answer. The genuine
derailment in `aime-hard` starts at token 4281 (52% through) — the same modular
step re-derived 4x:
`'0 mod7 => N = -1000(1 - A) mod7 => N = 1000(A -'`

**But the obvious labelling scheme does not work.** Self-doubt marker density
does not separate success from failure — it inverts:

| trace | think tok | markers | density /1k | per-quarter | outcome |
|---|---|---|---|---|---|
| gsm8k-easy | 260 | 3 | **11.5** | [1,0,0,2] | correct |
| trap | 1994 | 16 | 8.0 | [6,3,3,4] | correct |
| amc23-ish | 4379 | 51 | **11.6** | [20,10,11,10] | correct |
| aime-ish | 8192 | 80 | 9.8 | [27,22,17,14] | **no answer** |
| aime-hard | 8192 | 52 | **6.3** | [7,12,18,15] | **no answer** |

The highest-density trace answered correctly; the failure had the lowest density.
The 4-gram repetition ratio inverts the same way (0.286 for the success vs 0.214
for the failure), which is why `metrics.is_degenerate` requires BOTH signals.

*"Wait, let me verify"* is therefore normal reasoning, not distress. What
separates the failure is the **trajectory**: `aime-hard` is the only trace whose
marker density rises toward the end ([7,12,18,15]) while every successful trace
front-loads and decays. The loop lands inside that rising region.

**Consequence for the study**: a doom-loop-vs-healthy split by phrase matching
would put the same `Hmm` in both buckets and measure nothing. The contrast is the
same marker phrase in a trace that **recovers** (correct `\boxed{}`) versus one
that **produces no answer**, and the NLA's job is to say what differs in `h_l`.

Caveats on this sample: n=5, and the reported loop in `aime-ish` at token 47 is a
false positive — the model restating the problem's equation 3x, surfaced only
because the probe used `min_repeats=3`. The project default is
`DOOM_LOOP_MIN_REPEATS = 20` and the trace pipeline should keep it.

**Next**: a trace-collection stage over the real GSM8K / AIME2025 / AMC23 sets
(~50+ problems) recording, per trace, the marker positions inside `<think>`,
whether `</think>` closed, whether a `\boxed{}` answer matched the gold, and the
loop span if any — producing a labelled position set for the AV to verbalize.

## D34 — the RL venv's torch is whatever `sglang[all]` picks, and that is fine

`setup_rl_stack.sh` step 1 seeds torch 2.6.0+cu124 from the PyTorch cu124 index
so nothing later can drag in a cu130 build. Step 2 then installs
`sglang[all]==0.5.8`, which resolves its own torch and wins: measured
`2.6.0+cu124 -> 2.9.1+cu128`, pulling `torchao`, `torchcodec`,
`torch-memory-saver` along with it. The verify block asserted `2.6.0` + `cu124`
and so failed a venv that was actually correct (`rl_setup5.log`, EXIT=1).

Resolution: **stop pinning a torch version in the check.** cu128 ships sm_80
kernels, the box is 2xA100 (sm_80), and `sgl-kernel 0.3.21` + `flashinfer 0.6.1`
were resolved against that same torch. The verify block now asserts only what has
actually broken a launch:

- **transformers is 4.x** — this is the load-bearing one, see D35.
- **no vLLM** — D21, in the other direction.
- **CUDA available, and `get_arch_list()` covers every device's compute
  capability** — a real portability check instead of a version-string match.

Final stack: torch 2.9.1+cu128, transformers 4.57.1, sglang 0.5.18 editable from
the patched v0.5.8 checkout (`0189f41c30` — the tag's `python/pyproject.toml`
carries a higher version string than the tag name; cosmetic), sgl-kernel 0.3.21,
flashinfer-python 0.6.1, miles 0.2.1, nla 0.1.0, no vLLM.

## D35 — the critic tokenizer must be RE-SAVED, not copied, across venvs

`load_nla_config` on `critic_rl` died with `AttributeError: 'list' object has no
attribute 'keys'` inside `_set_model_specific_special_tokens`. Cause: the
tokenizer files `convert_ar_to_critic.py` copied out of the AR checkpoint were
written by the **training** venv's transformers 5.9.0, and
`tokenizer_config.json` is not backward-readable across the major version —
`extra_special_tokens` serializes as a **dict** in 4.x and a **list** in 5.x
(5.x also adds `backend`, `is_local`, `local_files_only`). The RL venv pins
transformers 4.57.1, so it cannot read its own critic dir.

Fix: `convert_ar_to_critic.py` now round-trips the tokenizer through the
**running** `AutoTokenizer` (`from_pretrained` -> `padding_side = "right"` ->
`save_pretrained`) instead of `shutil.copy2`, so the emitted format always
matches the env that will consume it. A new `--tokenizer-from` points the load at
the base model when the AR's own config is from an incompatible major. The script
then re-reads the file it just wrote and refuses to continue if it sees the 5.x
shape, naming `.venv-rl/bin/python` as the fix — this failure belongs in the
converter, not 20 minutes into a GRPO launch.

`padding_side="right"` is copied from the reference's
`prepare_critic_checkpoint.py`: `critic_fwd` passes `attention_mask=None`
(causal-only), so left-pad tokens would be attended by the last real token.

Loader check after the fix, under `.venv-rl`:

```
d_model 2048 | critic_num_layers 21
injection_char '<|image_pad|>' 151655 | neighbors 29 / 522
critic_suffix_ids [1318, 29, 366, 1708, 29]
injection_scale 1000.0 | mse_scale 45.254833995939045
```

Neighbor IDs are verified against the live tokenizer by `load_nla_config` itself,
so this output is also the drift check passing.

## D36 — rollout batch is a GPU-count knob, not a memory knob; RL runs at 64x8

Reconsidered the GRPO batch after asking whether we should simply maximize
`--rollout-batch-size` against available VRAM. **We should not, because it is not
memory-bound.** `--rollout-batch-size` sets the number of *prompts* per RL step;
the resulting `--global-batch-size` samples are accumulated over
`global / (micro_batch * n_gpus)` micro-steps. Peak memory is set by
`--micro-batch-size` x sequence length. Raising the rollout batch buys lower
gradient noise at a linear wall-clock cost per step; it does not touch VRAM.

The reference's own two runs are the proof:

| | LR scan | production (released) |
|---|---|---|
| GPUs | 2 | 2 x 8 |
| prompts / rollout | 64 | 128 |
| **micro-batch** | **16** | **16** |
| global batch | 512 | 1024 |
| actor lr | 1e-5 | 1.41e-5 (= 1e-5 x sqrt2) |

Same micro-batch at both batch sizes. They scaled the rollout batch with **GPU
count** and rescaled LR by sqrt(batch).

So our target is their **2-GPU** config, since we have 2 GPUs:
`ROLLOUT_BATCH=64`, `SAMPLES_PER_PROMPT=8` -> global 512. Changed from 16 (global
128), which was 1/4 of the batch their smallest evidenced run used.

`ACTOR_MICRO` also moves **4 -> 16**. The 4 came from `configs/rl.sh`'s
`${ACTOR_MICRO:-4}` default, but `TRAINING_NOTES.md`'s RL section states what they
actually ran: "m16 is fine with resp_len capped at 150". Their sweep also shows
bigger is *not* automatically faster — m16 at 9.05s beat m64+checkpointing at
12.83s, because 8 microbatches of fwd+bwd cost less than 2 of fwd+recompute+bwd
(the extra FSDP gathers are cheaper than the recompute saved). Their memory
ceiling (7B, d_model 3584, 28 layers + a 152k lm_head) is far above ours, so we
have headroom to go higher, but the FLOP-equivalence argument is about ratios and
their measured optimum is the honest starting point.

Two hard constraints carried over:

- **`grad_accum` must be an integer.** `512 / (16 * 2) = 16` exactly. They
  measured a non-integer accum (5.33) at **479s/step** against ~9s — pathological,
  not a gentle slowdown.
- **Never pass `--gradient-checkpointing` in RL.** It deadlocks NCCL inside
  `update_weights()` (FSDP full-param gather changes behaviour, the broadcast
  hangs, the 10-minute watchdog SIGABRTs). Already annotated in `train_grpo.sh`.

The LR derivation is unchanged and now independently reproduces their number:
`1.41e-5 * sqrt(512/1024) = 9.97e-6` against their scan winner of 1e-5 at global
512. Critic at parity, "as they ran for most of training" — their scan's apparent
preference for a 5x hotter critic is flagged in their own notes as a 30-step
artifact that also rewards verbosity and courts OOM via length drift.

Standing caveat from the top of `TRAINING_NOTES.md`, worth repeating: "These are
the settings we used, not settings we claim are optimal. We did not sweep batch
size, learning rate, or GRPO group size for RL."

## D37 — flash-attn must be installed AFTER sglang, and needs a matching nvcc

The preflight never reached its first check:

```
ImportError: .../flash_attn_2_cuda.cpython-311-x86_64-linux-gnu.so:
  undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb
```

That symbol lives in `libc10_cuda`, so this is a torch C++ ABI mismatch, reached
through `miles.backends.fsdp_utils.actor` -> `ring_flash_attn` -> `flash_attn`.
An **ordering** bug in `setup_rl_stack.sh`: flash-attn was installed in step 2,
binding it to step 1's torch 2.6.0+cu124, and step 3 then replaced torch with
2.9.1+cu128. Step 2 didn't compile — it silently took the prebuilt
`cu12torch2.6` wheel, which is why nothing failed at install time.

Moved to a new **step 3b**, after sglang. Consequences worth recording:

- **torch 2.9.1 is correct, not an accident.** The `v0.5.8` tree (HEAD
  `0189f41c30`, tagged, 2026-01-23) hard-pins `torch==2.9.1` in
  `python/pyproject.toml`. D34 accepted this version on kernel-coverage grounds;
  it is in fact what sglang requires. `sgl-kernel 0.3.21` and
  `flashinfer-python 0.6.1` are resolved against the same torch. flash-attn was
  the only package out of step.
- **No prebuilt FA2 wheel exists for torch 2.9.** The v2.8.x releases top out at
  `cu12torch2.8`; everything newer on the releases page is FA4 beta with no cp311
  assets. So step 3b compiles from source.
- **The build needs an nvcc matching torch's CUDA.** The image ships CUDA **13.0**
  only, and torch's `_check_cuda_version` aborts: *"The detected CUDA version
  (13.0) mismatches the version that was used to compile PyTorch (12.8)"*.
  `nvidia-cuda-nvcc-cu12==12.8.*` from PyPI is NOT enough — it ships `ptxas` and
  `nvvm` but no `nvcc` driver binary. `apt-get install -y cuda-nvcc-12-8` supplies
  a real one at `/usr/local/cuda-12.8` (the CUDA apt repo is already configured in
  this image). `TORCH_CUDA_ARCH_LIST=8.0` keeps it to the one arch we run on.
- **`--no-deps` is mandatory on that install.** A first attempt with
  `--force-reinstall` and no `--no-deps` began re-resolving the whole tree and
  downloading `nvidia-nccl-cu13` + triton — D21's hazard from a new direction.
  Killed mid-download; the venv was verified intact afterwards (torch 2.9.1+cu128,
  transformers 4.57.1, sglang, sgl-kernel 0.3.21 all unchanged).

Also fixed alongside: `git clean -fdq` on the sglang clone becomes **`-fdqx`**.
setuptools_scm writes `python/sglang/_version.py`, which is gitignored, so a plain
clean left a stale one from the earlier v0.5.9 attempt and the correctly
checked-out v0.5.8 tree installed itself as **"sglang 0.5.18"**. Cosmetic here —
the code and its pyproject were genuinely v0.5.8 — but it is exactly the kind of
misreported version that sends you chasing the wrong dependency.

And the verify block now imports `miles.backends.fsdp_utils` rather than bare
`miles`, since that is the module that actually pulls flash_attn. `import miles`
alone passed cleanly against the broken extension.

## D38 — the critic sidecar's `extraction_layer_index` is K, not the layer count

`rl_preflight` check 2 failed immediately:

```
AssertionError: critic truncated to 21 layers but extraction layer_index=21
  -> want 22. Off-by-one in prepare_critic_checkpoint --num-layers.
```

`build_model_sidecar` wrote `critic.extraction_layer_index = cfg.ar_num_layers`
(21, the layer COUNT) where the reference means K, the extraction **layer index**
(20). Their loader reads the field into a variable named `critic_num_layers`,
which is what invited the confusion, but `rl_preflight` pins the semantics:
`assert n_layers == k + 1`. The same file's dataset sidecar already had it right
(`extraction.layer_index = cfg.extraction_layer`), so our two sidecars disagreed
with each other.

**The checkpoint itself was never wrong** — `critic_rl/config.json` holds
`num_hidden_layers: 21` = blocks 0..20, and block 20's output is exactly what
datagen captured. This was a metadata-only defect. But it is the metadata their
loader trusts, and their own comment prices the real version of this bug: *"v21's
critic was prepared with --num-layers 33 when extraction layer_index=32 ->
num_hidden_layers=34 (one too many). Head had to approximately undo block 33's
transform to hit the gold at block-32 output. SFT-FVE ceiling dropped to ~0.32."*

Worth flagging against my own earlier note: the `critic_num_layers 21` line in
D35's loader dump was me reading this wrong value back and reporting it as
confirmation the truncation was correct. It confirmed nothing — `load_nla_config`
echoes the sidecar without cross-checking `config.json`. Only `rl_preflight`
compares the two, which is the whole reason to run it before Ray starts.

This is the third instance in three days of the `hidden_states` index (21) being
substituted for the layer index (20). CLAUDE.md's rule — use
`NLAConfig.hidden_states_index` / `ar_num_layers` rather than open-coding the +-1
— is necessary but not sufficient: here BOTH helpers exist and the wrong one was
picked, because the reference's field NAME suggests a count. When exporting to
their schema, the check is what their reader asserts, not what our field is called.

## D39 — 2 GPUs forces `--colocate` and a 1 actor + 1 critic layout

First GRPO launch hung. No error, no timeout: `train.py` alive at ~7% CPU with
28s of CPU time over 7 minutes, Ray infra up, **zero worker actors, `nvidia-smi`
completely empty**. That is what an unsatisfiable Ray placement group looks like.

`miles/ray/placement_group.py:create_placement_groups` sizes ONE placement group
for every role before anything starts, and picks the total by branch:

| branch | GPUs requested | on 2 GPUs |
|---|---|---|
| default | actor + rollout + critic = 2+2+2 | **6 — hangs** |
| `--colocate` | actor + critic = 2+0+2 | **4 — still hangs** |
| `--colocate`, 1+1 | actor + critic = 1+0+1 | **2 — fits** |

`--colocate` folds the sglang engines onto the actor's GPUs and ignores
`--rollout-num-gpus`, but **the critic always gets its own devices** — there is no
flag that colocates it (the one mention at `placement_group.py:187` is
`colocate + debug_train_only` only). So on two devices the only viable layout is
**1 actor (sharing with rollout) + 1 critic**.

Our script's comment already said "colocation is not optional here" — but the flag
was never passed. The comment was right and the code did not implement it.

Consequences accepted:

- `--colocate` forces `--offload`, so the actor is swapped to CPU while sglang
  generates and back for the training pass, every step. That is the price of two
  devices, and it will show up in step time.
- `--num-gpus-per-node` must be set to 2. It defaults to 8, and miles' own help
  says: *"If you are going to use less than 8 gpus per node under colocate mode,
  you should set this number."*
- The actor no longer shards across 2 GPUs. Fine at 1.7B: full-FT Adam state is
  ~24 GB (bf16 weights + fp32 master + m/v) against 80 GB.
- `grad_accum` becomes `512 / (16 x 1) = 32` — still an exact integer, which D36
  flagged as load-bearing (their non-integer accum measured 479s/step vs ~9s).

Added a **preflight guard in the launcher** that recomputes the same arithmetic
and refuses to start when the layout exceeds visible GPUs, printing the per-role
breakdown. A silent forever-hang is the worst failure mode of the three, because
it looks exactly like slow startup — I reported this run as "launched and
initializing" before checking `nvidia-smi`.

Revised expectation for the 400-step probe: the ~47s/step figure from their notes
was measured at `rollout_batch=64x8` on their hardware without offload swapping,
so it is now a floor rather than an estimate. Real step time gets measured, not
predicted.

## D40 — miles' hardcoded cuda-compat path breaks CUDA in the sglang engines

With the placement group fixed (D39), the run got further and died in
`SGLangEngine.init()`:

```
RuntimeError: No accelerator (CUDA, XPU, HPU, NPU) is available.
  sglang/srt/server_args.py:825 _handle_missing_default_values -> get_device()
```

Misleading message: `nvidia-smi` was healthy, the driver was fine, and
`torch.cuda.is_available()` was True everywhere else. Not a GPU-visibility
problem either — miles deliberately sets `NOSET_VISIBLE_DEVICES_ENV_VARS_LIST`
so the engine sees every device and selects via `base_gpu_id`.

The cause is the `LD_LIBRARY_PATH` miles injects into each engine actor's Ray
`runtime_env` (`miles/ray/rollout.py`):

```python
"LD_LIBRARY_PATH": f"/usr/local/cuda/compat:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:{sys.prefix}/lib",
```

Its comment says "cuda-compat first so a forward-compat libcuda.so wins if
present". Forward-compat libs exist for a driver **older** than the toolkit. Here
the box is the other way round:

| | version |
|---|---|
| kernel driver (`/sys/module/nvidia/version`) | **580.105.08** |
| `/usr/local/cuda-13.0/compat/libcuda.so.*` | **580.65.06** |

so the stale compat `libcuda.so.1` shadowed the real one and CUDA init failed with
`Error 803: system has unsupported display driver / cuda driver combination`.

Bisected the path entry by entry:

| LD_LIBRARY_PATH | `torch.cuda.is_available()` |
|---|---|
| unset | True |
| miles' full string | **False** |
| `/usr/local/cuda/compat` alone | **False** |
| miles' string minus compat | True |

So `compat` is the entire problem; the CUDA-13 `lib64` alongside a cu128 torch is
harmless (torch uses its own bundled runtime).

Fix: `mv /usr/local/cuda-13.0/compat{,.disabled-mismatched-driver}`. We cannot
override this from our side — Ray `runtime_env` env_vars beat the launcher's — and
patching the pinned miles clone would be undone by the `reset --hard` in
`setup_rl_stack.sh`. Disabling the directory is also simply correct here: a compat
lib older than the running driver has no legitimate use.

`train_grpo.sh` now **detects** the mismatch (compares the compat lib's version
against the live driver with `sort -V`) and refuses to launch with the exact `mv`
command, rather than silently mutating `/usr/local` or letting the run die 90
seconds in with a message about accelerators that points nowhere near the cause.

## D41 — `--kl-loss-type k2`, and the reference clone is per-machine (keep it pulled)

Next failure, from the reference's own actor:

```
AssertionError: --kl-loss-type k1 with --use-kl-loss has zero expected gradient;
  use k2/k3/low_var_kl or add --use-unbiased-kl.
```

Correct and important: k1 = `log p - log p_ref` as a **direct loss term** has zero
expected gradient under the sampling distribution, so a KL penalty configured that
way does nothing at all while still logging a plausible `train/kl_loss`. miles
defaults `--kl-loss-type` to k1; their `configs/rl.sh` overrides it to **k2**. Our
launcher passed only `--use-kl-loss --kl-loss-coef`, so it inherited k1.

Fixed: `--kl-loss-type "${KL_LOSS_TYPE:-k2}"`.

**Process failure worth recording.** I first concluded the assertion lived in
*miles* and that our `MILES_PIN` had outrun their config — then that we should pick
k3 on GRPO-literature grounds. Both wrong. `natural_language_autoencoders/` is
**gitignored** (`.gitignore:226`), so it is a separate clone on every machine, and
the local one was behind the pod's. The pod's HEAD was `0577769 "Trim KL comments;
allow k1 with --use-unbiased-kl; note k2 in design/header"` — the commit that
answers this exact question — against a local `879282f`. Two searches for the
answer came back empty because the file I was grepping did not have it yet.

`git -C natural_language_autoencoders pull` before treating that tree as the
authority. It is documentation as much as code, and it is not pinned by our repo.

**Second constraint found in the same header** (and not previously in our script):

> One step per rollout: NLAFSDPActor refuses to start if
> `rollout_batch_size x n_samples_per_prompt != global_batch_size`; set
> `NLA_I_KNOW_WHAT_IM_DOING=1` to bypass. (A mismatch would not change the step
> count — the FSDP path forces one step — it silently rescales gradients via the
> loss normalizer instead.)

We satisfy it (64 x 8 = 512 = GLOBAL_BATCH) because GLOBAL_BATCH defaults to that
product, but it was luck rather than intent, and an override of any one of the
three would have broken it quietly. Now asserted in the launcher with that
reasoning attached. It also retroactively justifies D36's choice to scale
ROLLOUT_BATCH rather than decouple the batches: they are not independent knobs.

Their header is worth heeding on one more point — synchronous `train.py` with one
optimizer step per rollout is *"the ONLY configuration we have tested — all
released checkpoints were trained this way"*, and `train_async.py` overlaps
generation with training so samples come from stale weights. We are on `train.py`.
