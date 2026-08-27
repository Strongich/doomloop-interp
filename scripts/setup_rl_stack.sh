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
SGLANG_PIN="${SGLANG_PIN:-v0.5.8}"
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
# ring_flash_attn assumes flash-attn is present; needed when not using build_conda.sh.
"${PIP[@]}" flash-attn --no-build-isolation || {
  echo "  WARNING: flash-attn build failed. Miles' ring_flash_attn needs it, and" >&2
  echo "  the reference RL config runs --attn-implementation flash_attention_2." >&2
  echo "  Install a matching prebuilt wheel for your CUDA/torch before training." >&2
}
"${PIP[@]}" -e "$SRC_ROOT/miles"

echo "=== 3. SGLang from source + the NLA input_embeds patches ==="
if [[ ! -d "$SRC_ROOT/sglang/.git" ]]; then
  git clone "$SGLANG_REPO" "$SRC_ROOT/sglang"
fi
git -C "$SRC_ROOT/sglang" fetch --all --tags --quiet
git -C "$SRC_ROOT/sglang" checkout --quiet "$SGLANG_PIN"
echo "  sglang at $SGLANG_PIN ($(git -C "$SRC_ROOT/sglang" rev-parse --short HEAD))"
# Training needs the patched source (bf16-base64 transport, chunked-prefill
# slicing, retract-path KV fix) — a wheel will not do.
bash "$NLA_REPO/patches/apply_sglang_patches.sh" "$SRC_ROOT/sglang"
"${PIP[@]}" -e "$SRC_ROOT/sglang/python[all]"

echo "=== 4. the reference nla package + ours ==="
"${PIP[@]}" -e "$NLA_REPO"
"${PIP[@]}" -e "$PROJECT_ROOT"

echo "=== 5. verify ==="
"$RL_ROOT/bin/python" -c "import miles, sglang, nla; print('miles + sglang + nla import OK')"

cat <<EOM

Done. The RL stack is in $RL_ROOT; the project venv is untouched.

Launch GRPO with:
  RL_PYTHON=$RL_ROOT/bin/python scripts/train_grpo.sh

Note the reference's environment requirements:
  - /dev/shm must be >= 8g (it writes ~1 GB of embedding dumps per step)
  - export CUDA_HOME if your toolkit is not at /usr/local/cuda
EOM
