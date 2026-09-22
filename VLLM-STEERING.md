# vLLM generation with the reasoning intervention

Implemented and locally checked on 2026-09-21 using the installed **vLLM 0.22.0**,
Qwen3-1.7B BF16, and RTX 5070 Ti. No dependency upgrades or vLLM source patches.

## Run

The corrected-prefix launcher now uses vLLM for **base, N, D, and exit**:

```bash
bash scripts/run_prefix_online.sh
```

It reads the existing `data/prefix_online/prefixes.jsonl` and writes new branches
to `data/prefix_online_vllm/`. The stopped Transformers results stay in their
original directory. It does not append vLLM rows to those results.

For a bounded run in a separate directory:

```bash
LIMIT=16 SEEDS=1 OUTDIR=data/prefix_online_vllm_pilot bash scripts/run_prefix_online.sh
```

Direct use, including throughput/checkpoint controls:

```bash
uv run python scripts/branch_continue.py \
  --backend vllm \
  --prefixes data/prefix_online/prefixes.jsonl \
  --outdir data/prefix_online_vllm \
  --arms base N D exit --seeds 2 \
  --batch 32 --checkpoint-every 128 \
  --max-new-tokens 12288 --exit-tokens 4096 \
  --gpu-memory-utilization 0.85 --max-num-batched-tokens 2048
```

`--batch` limits concurrent sequences; vLLM manages KV blocks and dynamically
admits/retires requests. `--checkpoint-every` is the number of submitted requests
per durable checkpoint, independently of concurrency. Smaller values reduce lost
work on interruption but leave less queued work to fill vacated batch slots.
`--kv-budget` and `--retire-every` apply only to the Transformers backend.

Use `--backend transformers` for the historical implementation. Select its old
output directory explicitly when intentionally resuming that engine. The vLLM
runner rejects existing unmanifested result directories and both dispatch paths
prevent accidental writing into the other backend's manifested directory.

## Exact intervention

The worker hook is in
[src/reasoning_attention/serving/vllm_steering.py](src/reasoning_attention/serving/vllm_steering.py).
At the output of decoder layer 20, vLLM returns separate MLP-output and residual
arrays. Their sum is the full residual used by the original Transformers hook:

```text
h = hidden_states + residual
h_edited = h + alpha * ||h||_2 * unit_direction
```

Only selected rows are changed. For those rows, the outgoing pair is
`(h_edited, 0)` so the next fused residual addition produces the edited state.
Unselected tensor pairs are preserved. Norms/directions use FP32 and the addition
uses the same BF16 rounding order as the existing hook.

The selector uses request IDs and absolute token positions from the bundled V1
model runner, not persistent batch-slot assumptions. It:

- Excludes every original prompt/prefix token, including the freeze boundary.
- Edits a newly generated paragraph-boundary token when it is consumed to predict
  the following token, matching the original decode loop.
- Stops at the first generated `</think>` and never steers the exit arm.
- Reapplies edits at the same absolute generated-token sites if KV preemption
  causes those tokens to be recomputed.
- Records actual unique sites and application counts. The driver independently
  derives expected sites from output token IDs and fails on any mismatch.

The NLA is not loaded during generation. The direction files are the existing
N/D/P vectors; their hashes are recorded with each run.

## Supported execution and limits

This implementation deliberately uses the **V1 GPU model runner bundled inside
vLLM 0.22.0**, eager execution, synchronous scheduling, a single GPU, and no prefix
cache or speculative decoding. Continuous batching and paged KV caching remain
active. CUDA graphs/compilation are disabled because dynamically installed Python
hooks must execute on every forward. Prefix caching is disabled because changing
directions must not reuse intervention-dependent cached states.

The factory selects `VLLM_USE_V2_MODEL_RUNNER=0` and
`VLLM_USE_FLASHINFER_SAMPLER=0` in the generation process. The installed default V2
runner failed during local warmup, and FlashInfer sampling failed during a
separate V1 startup. The native sampler and bundled V1 runner passed the tests.
These are execution settings, not package downgrades. FlashInfer may still be
used for supported attention kernels; the override concerns sampling.

The hook checks the exact installed version and dense Qwen3 architecture. MoE,
tensor/pipeline parallelism, speculative decoding, CUDA graphs, and asynchronous
scheduling require separate implementations/validation. This is a working backend
for the current experiment, not a claim of compatibility with every vLLM model.
All arms in a comparison use the same engine configuration.

## Reproducibility and resume

`run_manifest.json` records inputs/code/direction hashes, library versions,
model/tokenizer metadata, decoding, steering, and engine settings. A changed
manifest requires a new output directory. A hash of `(question_id, seed)` assigns
sampling seeds independently of checkpoint batching and identically across arms.
It does not imply bit-identical outputs across GPU architectures or engines.

`branches_vllm.jsonl` is the durable result journal, including complete token IDs,
text, grading, request seed, and the injection audit. CSV and text-only JSONL files
are derived snapshots. Completed runs resume as a no-op rather than regenerating
everything. An incomplete final journal write is dropped on resume; malformed
complete records raise an error. Interrupted in-flight requests are regenerated.

Two metrics are made explicit in the new backend:

- `injections` counts sites actually consumed and modified; a boundary emitted as
  the final capped token is not counted as an injection that never executed.
- Unclosed thinking uses its observed generated-token count, with `closed=0`;
  exit has zero thinking tokens. The old HF runner could store `-1` for these.

Do not silently merge historical HF and new vLLM results. Existing
`scripts/prefix_report.py` accepts the new CSV schema.

## Validation performed

```bash
uv run python scripts/test_vllm_steering.py
uv run python scripts/check_vllm_steering_parity.py collect --out /tmp/steering-parity.json
uv run python scripts/check_vllm_steering_parity.py replay --out /tmp/steering-parity.json
make lint
```

- Ten CPU regression checks cover prompt exclusion, closing-tag scope, recompute,
  chunking, request order, exact FP32/BF16 edits, zero-alpha identity, interrupted
  journal recovery, and deterministic per-question seeds.
- GPU checks cover all four arms, worker/output site agreement, zero-alpha exact
  greedy output identity, and observable changes from nonzero steering.
- Three requests with two slots and a 256-token prefill chunk budget exercise
  request turnover and chunked prefill. HF replays the exact vLLM tokens to avoid
  conflating numerical differences with diverging generated contexts.
- Across 288 baseline and 288 steered positions, greedy next-token agreement was
  100%. Mean absolute selected-token log-probability differences were 0.00582 and
  0.00602 respectively; maximum differences were 0.189 and 0.131. This is numerical
  agreement on a bounded test, not proof of identical long sampled trajectories.
- A 16-question/all-four-arm scheduling check completed with exact site audits.
  At eight concurrent sequences and caps of 768/384 tokens, measured generation
  throughput was approximately 650–691 output tokens/second after loading. These
  capped smoke runs are not accuracy experiments or a measured HF speedup.

Preemption selection is covered by CPU replay tests; a forced GPU-memory
preemption stress test and CUDA-graph support are not claimed here.
