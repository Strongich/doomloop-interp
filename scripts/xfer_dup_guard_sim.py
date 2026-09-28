#!/usr/bin/env python3
"""Offline check of a repeat guard: first paragraph inside <think> that is an EXACT token
repeat of an earlier paragraph (>= MINLEN tokens). Reports trigger rate and position by
arm and by capped/uncapped, from saved journals."""
import json, sys, glob, collections
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
isb = [tok.convert_ids_to_tokens(i).count("Ċ") >= 2 for i in range(len(tok))]
close = tok.convert_tokens_to_ids("</think>")
MINLEN = int(sys.argv[2]) if len(sys.argv) > 2 else 8
K = int(sys.argv[3]) if len(sys.argv) > 3 else 2  # trigger when a paragraph occurs the K-th time

def first_dup(ids):
    end = ids.index(close) if close in ids else len(ids)
    seen, start = collections.Counter(), 0
    for i in range(end):
        if isb[ids[i]]:
            para = tuple(ids[start:i + 1])
            if len(para) >= MINLEN:
                seen[para] += 1
                if seen[para] >= K:
                    return i + 1  # one-based generated index g of the boundary
            start = i + 1
    return None

stats = collections.defaultdict(lambda: [0, 0, []])
for path in glob.glob(sys.argv[1]):
    for line in open(path):
        r = json.loads(line)
        g = first_dup(r["token_ids"])
        k = (r["policy"], "capped" if r["capped"] else "ok")
        stats[k][0] += 1
        if g is not None:
            stats[k][1] += 1
            stats[k][2].append((g, len(r["token_ids"])))
for k in sorted(stats):
    n, t, gs = stats[k]
    frac = sorted(g / L for g, L in gs)
    med = frac[len(frac) // 2] if frac else float("nan")
    print(f"{k[0]:16s} {k[1]:7s} n={n:5d} triggered {t:5d} ({100*t/n:5.1f}%)  median trigger at {med:.2f} of length")
