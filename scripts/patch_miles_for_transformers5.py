"""Adapt the pinned miles checkout to transformers 5.x + sglang v0.5.15.

miles 0.2.x targets transformers 4.x and sglang v0.5.8. We need transformers
5.12.1 because 4.57.1 mis-computes Qwen3's forward when the prompt arrives as
inputs_embeds (D44), and that version is chosen by sglang's own hard pin — so the
API drift has to be absorbed here. Three independent breakages, all mechanical:

1. `ring_flash_attn` imported at module scope (`fsdp_utils/actor.py`).
   Only context parallelism uses it, and cp_size defaults to 1. Made lazy.
   (The real fix for ring-flash-attn itself is scripts/patch_ring_flash_attn.py;
   this one just removes the needless module-scope dependency.)

2. `model._no_split_modules` became a **set** in transformers 5.x, and
   `apply_fsdp2` indexes it -> "TypeError: 'set' object is not subscriptable".

3. sglang renamed its ServerArgs fields `*_parallel_size` -> `*_size`, so
   `validate_args` died with "Namespace object has no attribute
   sglang_data_parallel_size".

`.rl-src/miles` is a disposable clone that setup_rl_stack.sh hard-resets before
re-applying patches, so this runs after that. Idempotent.

Usage:
    python scripts/patch_miles_for_transformers5.py --miles-src .rl-src/miles
"""

import argparse
import sys
from pathlib import Path

ACTOR = ("miles", "backends", "fsdp_utils", "actor.py")
SGL_ARGS = ("miles", "backends", "sglang_utils", "arguments.py")

RING_IMPORT = "from ring_flash_attn import update_ring_flash_attn_params\n"
RING_CALL = "                update_ring_flash_attn_params(cu_seqlens, self.cp_group)"
RING_LAZY = """                # Imported here, not at module scope: ring-flash-attn 0.1.8 reads a
                # transformers symbol that moved in 5.x. Only context parallelism
                # needs it, and cp_size > 1 is the branch we are already in.
                from ring_flash_attn import update_ring_flash_attn_params

                update_ring_flash_attn_params(cu_seqlens, self.cp_group)"""

NOSPLIT = """    layer_cls_to_wrap = model._no_split_modules
    assert len(layer_cls_to_wrap) > 0 and layer_cls_to_wrap[0] is not None"""
NOSPLIT_FIX = """    # transformers 5.x makes _no_split_modules a set, and this indexes it ->
    # "TypeError: 'set' object is not subscriptable". sorted() gives a stable list;
    # the membership test below is unaffected.
    layer_cls_to_wrap = sorted(model._no_split_modules or [])
    assert len(layer_cls_to_wrap) > 0 and layer_cls_to_wrap[0] is not None"""

PARALLEL = """    args.sglang_tp_size = args.rollout_num_gpus_per_engine
    args.sglang_dp_size = args.sglang_data_parallel_size
    args.sglang_pp_size = args.sglang_pipeline_parallel_size
    args.sglang_ep_size = args.sglang_expert_parallel_size"""
PARALLEL_FIX = """    args.sglang_tp_size = args.rollout_num_gpus_per_engine

    # sglang renamed its ServerArgs fields *_parallel_size -> *_size (by v0.5.15);
    # miles still reads the old names. Accept either.
    def _pick(*names, default=1):
        for n in names:
            v = getattr(args, n, None)
            if v is not None:
                return v
        return default

    args.sglang_dp_size = _pick("sglang_data_parallel_size", "sglang_dp_size")
    args.sglang_pp_size = _pick("sglang_pipeline_parallel_size", "sglang_pp_size")
    args.sglang_ep_size = _pick("sglang_expert_parallel_size", "sglang_ep_size")"""


def _edit(path: Path, name: str, done_marker: str, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if done_marker in text:
        print(f"  {name}: already applied")
        return
    if text.count(old) != 1:
        print(
            f"  WARNING {name}: anchor found {text.count(old)}x in {path} (want 1). "
            f"Upstream changed it — re-derive rather than forcing.",
            file=sys.stderr,
        )
        return
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"  {name}: applied")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--miles-src", default=".rl-src/miles")
    args = ap.parse_args()
    root = Path(args.miles_src)

    actor = root.joinpath(*ACTOR)
    sgl_args = root.joinpath(*SGL_ARGS)
    for p in (actor, sgl_args):
        assert p.is_file(), f"{p} not found — is --miles-src correct?"

    text = actor.read_text(encoding="utf-8")
    if RING_IMPORT in text:
        if text.count(RING_CALL) == 1:
            actor.write_text(
                text.replace(RING_IMPORT, "", 1).replace(RING_CALL, RING_LAZY, 1),
                encoding="utf-8",
            )
            print("  ring_flash_attn lazy import: applied")
        else:
            print("  WARNING ring_flash_attn: call site not found; skipped", file=sys.stderr)
    else:
        print("  ring_flash_attn lazy import: already applied")

    _edit(actor, "_no_split_modules set", "sorted(model._no_split_modules", NOSPLIT, NOSPLIT_FIX)
    _edit(sgl_args, "sglang *_size rename", "def _pick(", PARALLEL, PARALLEL_FIX)


if __name__ == "__main__":
    main()
