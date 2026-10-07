#!/usr/bin/env python3
"""EXPERIMENT-reviewer-baselines.md §B4: lock before any Part C rollout. sha256 of the hard
set, the protocol file, every direction file used by any arm, and the git state."""
import hashlib, json, subprocess, sys, time
from pathlib import Path

OUT = Path("data/baselines/protocol_locked.json")
if OUT.exists():
    sys.exit(f"{OUT} exists")
FILES = ["data/baselines/hard128.jsonl", "data/baselines/hard128_build.json",
         "EXPERIMENT-reviewer-baselines.md", "data/baselines/ban_tokens.json",
         "data/pool/dir_A_1trace.pt", "data/dom/dir_D_diffmeans.pt",
         "data/xfer8b/dirs_clean/L26_NU.pt", "data/xfer8b/dirs_clean/L26_D8.pt",
         "data/xfer8b/dirs_clean/L26_DU.pt", "data/xfer8b/dirs_clean/L26_maps.pt",
         "data/xfer30b/dirs_L34/L34_NU.pt", "data/xfer30b/dirs_L34/L34_D8.pt",
         "data/xfer30b/dirs_L34/L34_DU.pt", "data/xfer30b/dirs_L34/L34_maps.pt",
         "data/xfer235b/dirs_clean/L67_NU.pt",
         "data/baselines/seal/seal_1p7b_L20.pt", "data/baselines/seal/seal_8b_L26.pt",
         "data/xfer8b/confirm_mathtest400.jsonl", "data/gated/mathtest_gate400.jsonl",
         "data/policy/dev400.jsonl"]
sh = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
code = ["scripts/reasoning_policy_vllm.py", "scripts/baselines_deer.py",
        "scripts/baselines_seal_vector.py", "src/reasoning_attention/serving/vllm_steering.py",
        "src/reasoning_attention/serving/vllm_ban.py", "src/reasoning_attention/grading.py"]
lock = {"locked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": {p: sh(p) for p in FILES}, "code": {p: sh(p) for p in code},
        "git": {"head": git("rev-parse", "HEAD"), "status": git("status", "--short").splitlines()}}
OUT.write_text(json.dumps(lock, indent=2) + "\n")
print(f"wrote {OUT}: {len(FILES)} files")
