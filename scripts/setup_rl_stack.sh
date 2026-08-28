#!/usr/bin/env bash
# Install the Stage-2 RL stack: Miles + patched SGLang + the reference nla package.
#
#   scripts/setup_rl_stack.sh            # build into .venv-rl
#   RL_ROOT=/opt/nla-rl scripts/setup_rl_stack.sh
#   scripts/setup_rl_stack.sh --check    # report what's installed, change nothing
#
# ---------------------------------------------------------------------------
# WHY A SEPARATE ENV, AND NOT THE PROJECT venv
#
# `sglang[all]` pins its own tested torch. Resolved against this project's venv it
# wants:
#     torch        2.11.0+cu130 -> 2.9.1   (a cu12 build)
#     transformers 5.13.0       -> 4.57.1
#     torchvision / torchaudio / triton    all downgraded
#
# Two of those are fatal here. cu12 wheels carry no sm_120 kernels, so torch stops
# working on this box's RTX 5070 Ti (capability 12.0) — extraction, SFT, and every
# smoke test die. And vLLM 0.22 pins torch 2.11, so it breaks too. The reference's
# own setup docs say the same thing: "An unpinned pip install torch may pull a
# cu130 build, which conflicts with sgl-kernel's cu12 wheels."
#
# The RL run targets 2xA100 (sm_80), where cu12 is fine. So the RL stack lives in
# its own env and the project venv stays intact. This mirrors upstream, whose
# build_conda.sh also creates a dedicated env.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT="$PWD"

RL_ROOT="${RL_ROOT:-$PROJECT_ROOT/.venv-rl}"
SRC_ROOT="${SRC_ROOT:-$PROJECT_ROOT/.rl-src}"
NLA_REPO="${NLA_REPO:-$PROJECT_ROOT/natural_language_autoencoders}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
# Pinned upstream commit from the reference repo, so its integration patch applies.
MILES_PIN="$(cut -d@ -f2 "$NLA_REPO/nla/miles_patches/UPSTREAM_PIN")"
MILES_REPO="${MILES_REPO:-https://github.com/radixark/miles.git}"
SGLANG_REPO="${SGLANG_REPO:-https://github.com/sgl-project/sglang.git}"
# PIN IT, and to v0.5.8 specifically. The NLA input_embeds patches anchor on exact
# source lines; an unpinned clone gets today's main and `apply_sglang_patches.sh`
# dies with "pattern matched 0 times — sglang source changed, patch manually".
# Tested per tag: the retract-fix anchor matches on 0.5.6 / 0.5.7 / 0.5.8 and
# breaks on 0.5.9, where three fields were appended to `reset_for_retract` between
# the anchor and `def offload_kv_cache`. 0.5.8 is therefore the newest usable tag.
# NOTE this disagrees with requirements/rl.txt, which resolved sglang==0.5.9 — the
# editable install from source is what actually gets used, and it wins because it
# is installed after the manifest. Regenerate the manifest (`make rl-pins`) if that
# divergence starts to matter.
# v0.5.8 pins transformers==4.57.1, which mis-computes Qwen3's forward when the
# prompt arrives as inputs_embeds: identical weights and tokens give a logit sum
# of -280k vs -42k under 5.9.0, and the AV degenerates (1/8 closed tags vs 8/8).
# transformers 5.9.0 and vLLM 0.22 agree with each other against it, so 4.57.1 is
# the outlier. sglang moved to transformers 5.x at v0.5.10 (5.3.0) and pins 5.12.1
# from v0.5.15 — which is why we track a newer tag. See D44.
SGLANG_PIN="${SGLANG_PIN:-v0.5.15}"
# cu124 per the reference's setup note; A100 is sm_80 so cu124 is fine.
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"

if [[ "${1:-}" == "--check" ]]; then
  echo "RL_ROOT   : $RL_ROOT $([[ -d $RL_ROOT ]] && echo '(exists)' || echo '(absent)')"
  echo "SRC_ROOT  : $SRC_ROOT"
  echo "miles pin : $MILES_PIN"
  if [[ -x "$RL_ROOT/bin/python" ]]; then
    "$RL_ROOT/bin/python" - <<'PY'
for m in ("torch", "sglang", "miles", "nla"):
    try:
        mod = __import__(m)
        print(f"  {m:12s} {getattr(mod, '__version__', 'ok')}")
    except Exception as exc:
        print(f"  {m:12s} MISSING ({type(exc).__name__})")
PY
  else
    echo "  (no interpreter yet — run without --check)"
  fi
  exit 0
fi

# The pod image ships pip but not uv, so bootstrap it rather than failing late.
if ! command -v uv >/dev/null 2>&1; then
  echo "=== uv not found — installing it ==="
  curl -fsSL https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null || { echo "ERROR: uv still not on PATH" >&2; exit 1; }

echo "=== 0. isolated venv at $RL_ROOT (python $PYTHON_VERSION) ==="
uv venv --python "$PYTHON_VERSION" "$RL_ROOT"
PIP=("uv" "pip" "install" "--python" "$RL_ROOT/bin/python")
mkdir -p "$SRC_ROOT"

PINS="$PROJECT_ROOT/requirements/rl.txt"

echo "=== 1. torch from $TORCH_INDEX (before anything can pull a cu130 build) ==="
"${PIP[@]}" --index-url "$TORCH_INDEX" torch torchvision torchaudio

echo "=== 1b. the rest of the wheel deps, from the pinned manifest ==="
# requirements/rl.txt is a full resolution (uv pip compile requirements/rl.in), so
# the env is rebuilt from a manifest rather than from whatever resolves today.
# Regenerate it with:
#   uv pip compile requirements/rl.in --python-version 3.11 \
#     --extra-index-url https://download.pytorch.org/whl/cu124 \
#     --index-strategy unsafe-best-match -o requirements/rl.txt
if [[ -f "$PINS" ]]; then
  "${PIP[@]}" --extra-index-url "$TORCH_INDEX" --index-strategy unsafe-best-match -r "$PINS"
else
  echo "  WARNING: $PINS missing — resolving sglang[all] fresh instead" >&2
  "${PIP[@]}" --extra-index-url "$TORCH_INDEX" --index-strategy unsafe-best-match "sglang[all]>=0.5.6"
fi

echo "=== 2. Miles @ $MILES_PIN + the NLA integration patch ==="
if [[ ! -d "$SRC_ROOT/miles/.git" ]]; then
  git clone "$MILES_REPO" "$SRC_ROOT/miles"
fi
git -C "$SRC_ROOT/miles" fetch --all --tags
git -C "$SRC_ROOT/miles" checkout "$MILES_PIN"
# Checking out the pin first is what makes `git apply` succeed — the patches are
# generated against exactly this commit.
for patch in "$NLA_REPO"/nla/miles_patches/*.patch; do
  if git -C "$SRC_ROOT/miles" apply --check "$patch" 2>/dev/null; then
    git -C "$SRC_ROOT/miles" apply "$patch"
    echo "  applied $(basename "$patch")"
  else
    echo "  SKIP $(basename "$patch") — already applied or does not apply"
  fi
done
# flash-attn is installed in step 3b — AFTER sglang, which pins torch.
# Our own patch, applied after theirs: ring-flash-attn 0.1.8 (its latest) imports
# a symbol transformers 5.x removed, and miles pulls it at module scope. Only
# context parallelism needs it, so make the import lazy. Without this,
# `import miles.backends.fsdp_utils` raises ImportError under transformers 5.12.1.
"$RL_ROOT/bin/python" "$PROJECT_ROOT/scripts/patch_miles_ring_attn.py" \
  --miles-src "$SRC_ROOT/miles" || echo "  (skipped — upstream moved it; the ring_flash_attn fix below is the load-bearing one)"

"${PIP[@]}" -e "$SRC_ROOT/miles"

# sglang >= v0.5.15 pulls build deps with no cp311 wheels (granian, and friends)
# that compile from source and need cargo. Without it the build dies with a bare
# "error: can't find Rust compiler" 300 lines into a wheel download log.
if ! command -v cargo >/dev/null 2>&1 && [[ -x "$HOME/.cargo/bin/cargo" ]]; then
  export PATH="$HOME/.cargo/bin:$PATH"
fi
if ! command -v cargo >/dev/null 2>&1; then
  echo "  WARNING: no cargo on PATH. If the sglang build fails on Rust, run:" >&2
  echo "    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y" >&2
fi
# sglang-grpc's build.rs shells out to protoc; without it cargo dies with
# 'Could not find `protoc`' after several minutes of compiling.
if ! command -v protoc >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    echo "  installing protobuf-compiler"
    apt-get install -y -qq protobuf-compiler || echo "  WARNING: protoc install failed" >&2
  else
    echo "  WARNING: no protoc; sglang-grpc will fail to build." >&2
  fi
fi

echo "=== 3. SGLang from source + the NLA input_embeds patches ==="
if [[ ! -d "$SRC_ROOT/sglang/.git" ]]; then
  git clone "$SGLANG_REPO" "$SRC_ROOT/sglang"
fi
git -C "$SRC_ROOT/sglang" fetch --all --tags --quiet
# Discard any previously-applied patches first: they leave the tree dirty and
# `checkout` then refuses ("local changes would be overwritten"). The clone is
# ours and disposable, and the patches are re-applied immediately below, so a hard
# reset is the right move and makes re-running this script idempotent.
git -C "$SRC_ROOT/sglang" reset --hard --quiet
# -x so IGNORED files go too. setuptools_scm writes python/sglang/_version.py,
# which is gitignored: a plain `clean -fdq` leaves a stale one behind and the
# rebuilt package then reports the PREVIOUS checkout's version (observed:
# a correct v0.5.8 tree installing as "sglang 0.5.18").
git -C "$SRC_ROOT/sglang" clean -fdqx
git -C "$SRC_ROOT/sglang" checkout --quiet "$SGLANG_PIN"
echo "  sglang at $SGLANG_PIN ($(git -C "$SRC_ROOT/sglang" rev-parse --short HEAD))"
# The reference's patches are cut against v0.5.8 and do NOT apply to newer tags.
# They are not load-bearing for us: input_embeds is already NATIVE in sglang
# (io_struct/tokenizer_manager carry it identically in v0.5.8 and v0.5.18), and
# nla_generate only uses the bf16-base64 transport when NLA_BF16_B64_EMBEDS=1,
# which we do not set — we ride the plain payload["input_embeds"] field. The
# retract fix is documented by them as a no-op for Qwen ("never retract anyway —
# KV headroom"). So apply what fits and report what does not, rather than aborting.
for patch in "$NLA_REPO"/patches/*.patch; do
  [[ -f "$patch" ]] || continue
  if git -C "$SRC_ROOT/sglang" apply --check "$patch" 2>/dev/null; then
    git -C "$SRC_ROOT/sglang" apply "$patch"
    echo "  applied $(basename "$patch")"
  else
    echo "  SKIP $(basename "$patch") — does not apply to $SGLANG_PIN (native path used)"
  fi
done
"${PIP[@]}" -e "$SRC_ROOT/sglang/python[all]"

echo "=== 3b. flash-attn, built against the torch sglang settled on ==="
# ORDER MATTERS. flash-attn ships a compiled extension bound to torch's C++ ABI,
# and step 2's install would bind it to step 1's torch — which step 3 then
# replaces (sglang v0.5.8 hard-pins torch==2.9.1). Symptom, at import inside
# miles' FSDP actor: "flash_attn_2_cuda...so: undefined symbol:
# _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb". So install it LAST.
#
# There is no prebuilt FA2 wheel for torch 2.9 (the v2.8.3 release tops out at
# cu12torch2.8), so this compiles. Two things make that work:
#   - CUDA_HOME must point at an nvcc whose major.minor matches torch's CUDA, or
#     torch's own _check_cuda_version aborts the build. The image ships 13.0
#     against a cu128 torch; `apt-get install -y cuda-nvcc-12-8` supplies 12.8.
#   - TORCH_CUDA_ARCH_LIST pins the single arch we run on, turning a very long
#     multi-arch build into a short one.
# Match torch's CUDA exactly — torch's own _check_cuda_version aborts the build
# otherwise ("detected CUDA version (13.0) mismatches ... (12.8)"). Derive it so a
# torch bump does not silently pick the wrong toolkit.
TORCH_CUDA="$("$RL_ROOT/bin/python" -c 'import torch;print(torch.version.cuda or "")')"
FA_CUDA_HOME="${FA_CUDA_HOME:-/usr/local/cuda-${TORCH_CUDA}}"
FA_ARCH_LIST="${FA_ARCH_LIST:-8.0}"   # A100 = sm_80
# --no-deps is NOT optional: `--force-reinstall` without it re-resolves the whole
# tree and pulls a cu13 torch, undoing steps 1-3 (this is D21's hazard again).
if [[ -x "$FA_CUDA_HOME/bin/nvcc" ]]; then
  CUDA_HOME="$FA_CUDA_HOME" PATH="$FA_CUDA_HOME/bin:$PATH" \
    TORCH_CUDA_ARCH_LIST="$FA_ARCH_LIST" MAX_JOBS="${FA_MAX_JOBS:-48}" \
    FLASH_ATTENTION_FORCE_BUILD=TRUE \
    "${PIP[@]}" flash-attn --no-build-isolation --no-deps --no-cache-dir || {
      echo "  WARNING: flash-attn build failed. Miles' ring_flash_attn imports it" >&2
      echo "  unconditionally in the FSDP actor, so training will not start." >&2
    }
else
  echo "  WARNING: no nvcc at $FA_CUDA_HOME/bin/nvcc — skipping flash-attn." >&2
  echo "  Install one matching torch's CUDA ($("$RL_ROOT/bin/python" -c 'import torch;print(torch.version.cuda)')):" >&2
  echo "    apt-get install -y cuda-nvcc-${TORCH_CUDA//./-}" >&2
fi

echo "=== 4. the reference nla package + ours ==="
"${PIP[@]}" -e "$NLA_REPO"
# --no-deps for OUR project. Its dependency tree includes vLLM, which pulls a
# cu130 torch and transformers 5.x and overwrites the pinned cu124 stack that
# Miles, sglang and the prebuilt sgl-kernel wheel were resolved against. Observed:
# torch 2.6.0+cu124 -> 2.13.0+cu130, transformers 4.57 -> 5.16. This is the same
# hazard D21 created this venv to avoid, just in the other direction. All we need
# here is the package importable for its config and prompt constants; every
# runtime dep it uses in this env (torch, transformers, pyarrow, yaml, numpy) is
# already pinned by the manifest.
"${PIP[@]}" -e "$PROJECT_ROOT" --no-deps

echo "=== 4a. ring-flash-attn transformers-5.x fallback ==="
# Must run AFTER every install step, since a reinstall restores the stock file.
"$RL_ROOT/bin/python" "$PROJECT_ROOT/scripts/patch_ring_flash_attn.py" --venv "$RL_ROOT"

echo "=== 4b. system shared libs sgl_kernel dlopens ==="
# sgl_kernel's compiled extension links libnuma. It is NOT a pip dependency, so
# the wheel installs clean and then fails at IMPORT with a message that blames
# the wheel and tells you to reinstall it:
#   ImportError: libnuma.so.1: cannot open shared object file
#   ModuleNotFoundError: No module named 'common_ops'
# Reinstalling never helps — the missing piece is a system package.
if ! ldconfig -p | grep -q libnuma; then
  if command -v apt-get >/dev/null 2>&1; then
    echo "  installing libnuma1"
    apt-get install -y -qq libnuma1 || echo "  WARNING: libnuma1 install failed" >&2
  else
    echo "  WARNING: libnuma.so.1 not found and no apt-get; install it for sgl_kernel." >&2
  fi
fi

echo "=== 5. verify ==="
EXPECTED_TRANSFORMERS="$(grep -oE '"transformers==[^"]+"' "$SRC_ROOT/sglang/python/pyproject.toml" \
  | head -1 | sed 's/.*transformers==//; s/"//')"
export EXPECTED_TRANSFORMERS
echo "  sglang declares transformers==${EXPECTED_TRANSFORMERS:-unknown}"
# Import miles.backends.fsdp_utils, not just miles: that is the module that
# pulls ring_flash_attn -> flash_attn, where a torch-ABI mismatch surfaces. A
# bare `import miles` passes happily with a broken flash_attn.
"$RL_ROOT/bin/python" -c "import miles.backends.fsdp_utils, sglang, sgl_kernel, nla, flash_attn; print('miles(fsdp) + sglang + sgl_kernel + nla + flash_attn import OK')"
# Assert the pinned stack actually survived. An import check alone passes happily
# on a torch that something upgraded underneath us, and the prebuilt sgl-kernel
# wheel is compiled against a specific torch ABI.
"$RL_ROOT/bin/python" - <<'PYCHECK'
import importlib.metadata as md
import os
import sys

import torch

# Step 1 seeds torch 2.6.0+cu124, but `sglang[all]==0.5.8` resolves its own
# torch and wins: measured 2.6.0+cu124 -> 2.9.1+cu128. That is FINE and is not
# worth fighting — cu128 ships sm_80 kernels, the A100 is sm_80, and sgl-kernel
# + flashinfer were resolved against that same torch. So do not pin a torch
# version here; assert only the things that actually broke a launch:
#   - transformers major version, because the critic checkpoint's
#     tokenizer_config.json is not readable across the 4.x/5.x boundary
#     (`extra_special_tokens` is a dict in 4.x, a list in 5.x)
#   - no vLLM, which drags a cu130 torch and transformers 5.x (D21)
#   - CUDA actually usable
problems = []
# Do not hardcode a version — assert the installed transformers matches what the
# sglang checkout itself declares. That keeps this honest across pin bumps, and
# catches the real hazard: something (vLLM, a stale manifest) quietly pulling a
# different major than the engine was built against.
want_tf = os.environ.get("EXPECTED_TRANSFORMERS", "")
have_tf = md.version("transformers")
if want_tf and have_tf != want_tf:
    problems.append(f"transformers is {have_tf}, but sglang pins {want_tf}")
if not torch.cuda.is_available():
    problems.append("torch.cuda.is_available() is False")
if torch.cuda.is_available():
    caps = {torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())}
    arch = {int(a.split("_")[1]) for a in torch.cuda.get_arch_list() if a.startswith("sm_")}
    missing = {c for c in caps if c[0] * 10 + c[1] not in arch}
    if missing:
        problems.append(f"torch {torch.__version__} has no kernels for device caps {missing}")
try:
    if md.version("vllm"):
        problems.append("vLLM is installed here — it will fight the pinned torch")
except md.PackageNotFoundError:
    pass
if problems:
    print("RL STACK VERIFY FAILED:", file=sys.stderr)
    for p in problems:
        print(f"  - {p}", file=sys.stderr)
    sys.exit(1)
print(
    f"stack verify OK: torch {torch.__version__}, "
    f"transformers {md.version('transformers')}, sglang {md.version('sglang')}"
)
PYCHECK

cat <<EOM

Done. The RL stack is in $RL_ROOT; the project venv is untouched.

Launch GRPO with:
  RL_PYTHON=$RL_ROOT/bin/python scripts/train_grpo.sh

Note the reference's environment requirements:
  - /dev/shm must be >= 8g (it writes ~1 GB of embedding dumps per step)
  - export CUDA_HOME if your toolkit is not at /usr/local/cuda
EOM
