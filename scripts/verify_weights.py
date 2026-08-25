"""Preflight: fail fast if any cached shard of a model is truncated.

An interrupted Xet download reconstructs shards *in place* under their final
names, so the HF cache reports the model as present while the bytes are short.
Nothing notices until the server parses a header minutes into startup. Reading
just the safetensors headers here is cheap and turns that into an instant,
legible failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

from safetensors import safe_open


def main() -> int:
    repo_id, hf_home = sys.argv[1], Path(sys.argv[2])
    cache = hf_home / "hub" / ("models--" + repo_id.replace("/", "--"))
    snapshots = sorted((cache / "snapshots").glob("*")) if cache.exists() else []
    if not snapshots:
        print(f"verify: {repo_id} not cached — the server will download it")
        return 0

    shards = sorted(snapshots[-1].glob("*.safetensors"))
    if not shards:
        print(f"verify: no safetensors under {snapshots[-1]} — letting the server decide")
        return 0

    bad = []
    for shard in shards:
        try:
            with safe_open(shard, framework="pt"):
                pass
        except Exception as exc:  # noqa: BLE001 - any parse failure means unusable
            bad.append((shard.name, f"{shard.stat().st_size} bytes: {exc}"))

    if bad:
        print(f"verify: {len(bad)}/{len(shards)} shard(s) of {repo_id} are corrupt or truncated:")
        for name, why in bad:
            print(f"  {name} — {why}")
        print(f"\nRe-download with:\n  rm -rf {cache}\n  HF_HUB_DISABLE_XET=1 uv run hf download {repo_id} --max-workers 4")
        return 1

    print(f"verify: {len(shards)} shard(s) of {repo_id} OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
