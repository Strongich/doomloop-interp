#!/usr/bin/env python3
r"""Student training, LOCKED PROTOCOL v2 §L6: SFT or sigmoid DPO, LoRA or full, plain HF + peft.

Data are distill_build_sets.py outputs -- stored token ids only (prompt_ids +
completion ids), never re-rendered through the chat template.

  SFT   loss = CE on completion tokens only, token-mean over each global batch of 64
  DPO   loss = -logsigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r))), beta 0.1, mean over
        each global batch of 32 pairs; log-probs are sums over completion tokens; the
        reference is the untrained base (LoRA: adapters disabled; full: a frozen bf16 copy)

Common: 1 epoch, seed 0, bf16 autocast (fp32 master weights), gradient checkpointing,
max length 16,384, AdamW wd 0, cosine with 3% warmup to 0, grad-clip 1.0, token-bucketed
micro-batches inside each global batch. The final checkpoint is saved (LoRA merged into
bf16 weights, for vLLM). Logs to W&B project doomloop-nla-sft.

    CUDA_VISIBLE_DEVICES=0 uv run python scripts/distill_train.py --method sft \
        --arm short --peft lora
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen3-1.7B"
MAX_LEN = 16384
LR = {("sft", "lora"): 1e-4, ("sft", "full"): 1e-5, ("dpo", "lora"): 5e-6, ("dpo", "full"): 5e-7}
GLOBAL = {"sft": 64, "dpo": 32}
BETA = 0.1


def load(method: str, path: Path) -> list[dict]:
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    for r in rows:
        keys = ("completion_ids",) if method == "sft" else ("chosen_ids", "rejected_ids")
        for k in keys:
            if len(r["prompt_ids"]) + len(r[k]) > MAX_LEN:
                raise ValueError(f"{r['question_id']} {k} exceeds {MAX_LEN}")
    return rows


def buckets(lengths: list[int], budget: int) -> list[list[int]]:
    """Indices grouped so each group's padded size (max len x count) <= budget (>= 1 each)."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    out, cur, cur_max = [], [], 0
    for i in order:
        m = max(cur_max, lengths[i])
        if cur and m * (len(cur) + 1) > budget:
            out.append(cur)
            cur, m = [], lengths[i]
        cur.append(i)
        cur_max = m
    if cur:
        out.append(cur)
    return out


class Student:
    def __init__(self, peft: str, device: str, ckpt: bool = True, init: str = MODEL) -> None:
        from transformers import AutoModelForCausalLM

        self.peft = peft
        dtype = torch.bfloat16 if peft == "lora" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(init, dtype=dtype,
                                                     attn_implementation="sdpa").to(device)
        model.config.use_cache = False
        if peft == "lora":
            from peft import LoraConfig, get_peft_model

            model = get_peft_model(model, LoraConfig(
                r=64, lora_alpha=128, lora_dropout=0.05, target_modules="all-linear",
                task_type="CAUSAL_LM"))
            model.print_trainable_parameters()
        if ckpt:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model = model
        inner = model.get_base_model() if peft == "lora" else model
        self.backbone, self.lm_head = inner.model, inner.lm_head

    def seq_logps(self, seqs: list[list[int]], n_prompt: list[int], device: str,
                  reduce_sum: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-sequence summed completion log-prob and completion token counts.

        Right-padded batch; logits only at completion positions, in checkpointed chunks
        (the full 16k x 151k fp32 logit matrix would not fit)."""
        L = max(map(len, seqs))
        ids = torch.full((len(seqs), L), 0, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s)
            att[i, : len(s)] = 1
        ids, att = ids.to(device), att.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = self.backbone(input_ids=ids, attention_mask=att if not bool(att.all()) else None
                              ).last_hidden_state
        # position t predicts token t+1; completion tokens are [n_prompt, len)
        mask = torch.zeros((len(seqs), L), dtype=torch.bool, device=device)
        for i, (s, p) in enumerate(zip(seqs, n_prompt, strict=True)):
            mask[i, p - 1 : len(s) - 1] = True
        rows, cols = mask.nonzero(as_tuple=True)
        hs = h[rows, cols]
        tgt = ids[rows, cols + 1]
        w = self.lm_head.weight

        def chunk(hc: torch.Tensor, tc: torch.Tensor) -> torch.Tensor:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = hc @ w.T.to(hc.dtype)
            return -F.cross_entropy(logits.float(), tc, reduction="none")

        lp = torch.cat([checkpoint(chunk, hs[j : j + 4096], tgt[j : j + 4096], use_reentrant=False)
                        if torch.is_grad_enabled() else chunk(hs[j : j + 4096], tgt[j : j + 4096])
                        for j in range(0, len(tgt), 4096)])
        per = torch.zeros(len(seqs), device=device, dtype=lp.dtype).index_add_(0, rows, lp)
        cnt = torch.bincount(rows, minlength=len(seqs))
        return per, cnt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["sft", "dpo"], required=True)
    ap.add_argument("--arm", required=True, help="sft: ordinary|short|steered|steered_short; "
                                                 "dpo: natural|steered")
    ap.add_argument("--peft", choices=["lora", "full"], required=True)
    ap.add_argument("--sets", type=Path, default=ROOT / "data/distill/sets")
    ap.add_argument("--out", type=Path, default=ROOT / "data/distill/models")
    ap.add_argument("--micro-tokens", type=int, default=16384,
                    help="padded tokens per forward")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N examples")
    ap.add_argument("--name", default=None)
    ap.add_argument("--lr", type=float, default=None, help="override the §L6 LR (§L10 item 8)")
    ap.add_argument("--lr-from", type=Path, default=None,
                    help="json with the selected {'lr': ...} (distill_queue.py sweep)")
    ap.add_argument("--init", default=MODEL,
                    help="starting weights; also the DPO reference (v3: the twin's merged SFT model)")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="sweep proxy: stop after N steps, constant LR after warmup, no save")
    ap.add_argument("--heldout", type=int, default=0, help="sweep: held-out tail examples")
    ap.add_argument("--wandb-project", default="doomloop-nla-sft")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    name = args.name or f"{args.method}_{args.arm}_{args.peft}"
    out = args.out / name
    if (out / "DONE").exists():
        raise SystemExit(f"{out} already trained")
    device = "cuda"
    torch.manual_seed(0)
    random.seed(0)
    data_path = args.sets / f"{args.method}_{args.arm}.jsonl"
    rows = load(args.method, data_path)
    if args.limit:
        rows = rows[: args.limit]
    order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(0)).tolist()
    G = GLOBAL[args.method]
    batches = [order[i : i + G] for i in range(0, len(order), G)]
    steps = len(batches)
    warm = max(1, math.ceil(0.03 * steps))
    hold: list[int] = []
    if args.max_steps:
        # Sweep proxy (§L10 item 8): the first max_steps batches of the full run's order, the
        # full run's warmup, then constant LR; loss on a held-out tail never trained on here.
        hold = order[-args.heldout:]
        batches = batches[: args.max_steps]
        assert not set(hold) & {i for b in batches for i in b}, "held-out overlaps sweep batches"
    lr = LR[(args.method, args.peft)]
    if args.lr_from is not None:
        lr = float(json.loads(args.lr_from.read_text())["lr"])
    if args.lr is not None:
        lr = args.lr
    comp = sum(len(r["completion_ids"]) for r in rows) if args.method == "sft" else \
        sum(len(r["chosen_ids"]) + len(r["rejected_ids"]) for r in rows)
    config = {"method": args.method, "arm": args.arm, "peft": args.peft, "lr": lr,
              "global_batch": G, "steps": steps, "warmup_steps": warm, "examples": len(rows),
              "completion_tokens": comp, "max_len": MAX_LEN, "micro_tokens": args.micro_tokens,
              "beta": BETA if args.method == "dpo" else None, "seed": 0, "epochs": 1,
              "data": str(data_path), "model": MODEL, "init": args.init,
              "lora": {"r": 64, "alpha": 128, "dropout": 0.05, "targets": "all-linear"}
              if args.peft == "lora" else None, "grad_clip": 1.0, "weight_decay": 0.0,
              "sweep_max_steps": args.max_steps or None,
              "sweep_heldout": args.heldout if args.max_steps else None}
    print(json.dumps(config), flush=True)

    student = Student(args.peft, device, init=args.init)
    ref = None
    if args.method == "dpo" and args.peft == "full":
        # Same fp32-weights + bf16-autocast numerics as the policy, so the step-0 margin is 0.
        ref = Student("full", device, ckpt=False, init=args.init)
        ref.model.eval().requires_grad_(False)
    params = [p for p in student.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, fused=True)
    from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

    sched = get_cosine_schedule_with_warmup(opt, warm, steps) if not args.max_steps else \
        get_constant_schedule_with_warmup(opt, warm)
    run = None
    if not args.no_wandb:
        import wandb

        run = wandb.init(project=args.wandb_project, name=name, config=config,
                         group=f"{'sweep-' if args.max_steps else ''}{args.method}-{args.peft}",
                         tags=["protocol-v2", "1.7B"] + (["lr-sweep"] if args.max_steps else []))

    def batch_loss(b: list[int], train: bool) -> tuple[dict, int]:
        """One global batch: backward (train) or forward-only; returns (log, completion tokens)."""
        log: dict = {}
        if args.method == "sft":
            n_tok = sum(len(rows[i]["completion_ids"]) for i in b)
            tot = 0.0
            lens = [len(rows[i]["prompt_ids"]) + len(rows[i]["completion_ids"]) for i in b]
            for grp in buckets(lens, args.micro_tokens):
                idx = [b[j] for j in grp]
                per, cnt = student.seq_logps(
                    [rows[i]["prompt_ids"] + rows[i]["completion_ids"] for i in idx],
                    [len(rows[i]["prompt_ids"]) for i in idx], device)
                loss = -per.sum() / n_tok
                if train:
                    loss.backward()
                tot += float(loss.detach())
            log = {"train/loss": tot, "train/completion_tokens": n_tok}
        else:
            stats = {"loss": 0.0, "acc": 0.0, "margin": 0.0, "chosen": 0.0, "rejected": 0.0}
            by_type: dict[str, list[float]] = {}
            n_tok = sum(len(rows[i]["chosen_ids"]) + len(rows[i]["rejected_ids"]) for i in b)
            # groups of pairs whose two sequences together fit 2 x micro-tokens
            pair_len = [len(rows[i]["prompt_ids"]) * 2 + len(rows[i]["chosen_ids"])
                        + len(rows[i]["rejected_ids"]) for i in b]
            groups, cur, cur_tok = [], [], 0
            for j in sorted(range(len(b)), key=lambda j: -pair_len[j]):
                if cur and cur_tok + pair_len[j] > 2 * args.micro_tokens:
                    groups.append(cur)
                    cur, cur_tok = [], 0
                cur.append(j)
                cur_tok += pair_len[j]
            groups.append(cur)
            for grp in groups:
                idx = [b[j] for j in grp]
                seqs = [rows[i]["prompt_ids"] + rows[i][k] for k in ("chosen_ids", "rejected_ids")
                        for i in idx]
                npr = [len(rows[i]["prompt_ids"]) for _ in (0, 1) for i in idx]
                lens = list(map(len, seqs))
                pol = torch.zeros(len(seqs), device=device)
                refl = torch.zeros(len(seqs), device=device)
                parts = buckets(lens, args.micro_tokens)
                for part in parts:
                    per, _ = student.seq_logps([seqs[j] for j in part], [npr[j] for j in part],
                                               device)
                    pol = pol.index_put((torch.tensor(part, device=device),), per)
                    with torch.no_grad():
                        if ref is None:
                            with student.model.disable_adapter():
                                rper, _ = student.seq_logps([seqs[j] for j in part],
                                                            [npr[j] for j in part], device)
                        else:
                            rper, _ = ref.seq_logps([seqs[j] for j in part],
                                                    [npr[j] for j in part], device)
                    refl = refl.index_put((torch.tensor(part, device=device),), rper.float())
                n = len(idx)
                rc, rr = BETA * (pol[:n] - refl[:n]), BETA * (pol[n:] - refl[n:])
                margin = rc - rr
                loss = -F.logsigmoid(margin).sum() / len(b)
                if train:
                    loss.backward()
                stats["loss"] += float(loss.detach())
                stats["acc"] += float((margin > 0).float().sum()) / len(b)
                stats["margin"] += float(margin.sum()) / len(b)
                stats["chosen"] += float(rc.sum()) / len(b)
                stats["rejected"] += float(rr.sum()) / len(b)
                for i, m in zip(idx, margin.tolist(), strict=True):
                    by_type.setdefault(rows[i]["type"], []).append(m)
            log = {f"train/{k}": v for k, v in stats.items()}
            log.update({f"train/margin_{t}": sum(v) / len(v) for t, v in by_type.items()})
            log["train/completion_tokens"] = n_tok
        return log, n_tok

    def heldout_loss() -> dict:
        """Forward-only loss on the held-out tail (sweep mode), weighted like one big batch."""
        student.model.eval()
        tot: dict[str, float] = {}
        w_sum = 0.0
        with torch.no_grad():
            for i in range(0, len(hold), G):
                hb = hold[i : i + G]
                lg, nt = batch_loss(hb, train=False)
                w = nt if args.method == "sft" else len(hb)
                w_sum += w
                for k, v in lg.items():
                    if k.startswith("train/") and k != "train/completion_tokens":
                        key = "heldout/" + k.split("/", 1)[1]
                        tot[key] = tot.get(key, 0.0) + v * w
        student.model.train()
        return {k: v / w_sum for k, v in tot.items()}

    student.model.train()
    t0, seen_tok = time.monotonic(), 0
    sweep_log: list[dict] = []
    for step, b in enumerate(batches):
        ts = time.monotonic()
        opt.zero_grad(set_to_none=True)
        log, n_tok = batch_loss(b, train=True)
        seen_tok += n_tok
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        dt = time.monotonic() - ts
        log.update({"train/grad_norm": float(gn), "train/lr": sched.get_last_lr()[0],
                    "train/tokens_seen": seen_tok, "train/step_seconds": dt,
                    "train/tok_per_s": n_tok / dt, "train/epoch": (step + 1) / steps})
        if args.max_steps and (step + 1) in (args.max_steps // 2, args.max_steps):
            h = heldout_loss()
            log.update(h)
            sweep_log.append({"step": step + 1, **h})
        if run:
            run.log(log, step=step + 1)
        if step % 10 == 0 or step == len(batches) - 1:
            eta = (time.monotonic() - t0) / (step + 1) * (len(batches) - step - 1)
            print(f"step {step + 1}/{len(batches)} " + " ".join(
                f"{k.split('/')[1]}={v:.4g}" for k, v in log.items()
                if k in ("train/loss", "train/acc", "train/margin", "train/grad_norm",
                         "train/tok_per_s", "heldout/loss")) + f" eta {eta / 60:.0f}m", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    if args.max_steps:
        res = {**config, "heldout": sweep_log, "final_heldout_loss": sweep_log[-1]["heldout/loss"],
               "train_seconds": time.monotonic() - t0}
        (out / "sweep_result.json").write_text(json.dumps(res, indent=1) + "\n")
        (out / "DONE").write_text("sweep\n")
        if run:
            run.summary.update({"final_heldout_loss": res["final_heldout_loss"]})
            run.finish()
        print(f"sweep {name}: heldout loss {res['final_heldout_loss']:.5f}", flush=True)
        os._exit(0)
    from transformers import AutoTokenizer

    model = student.model
    if args.peft == "lora":
        model.save_pretrained(out / "adapter")
        model = model.merge_and_unload()
    model = model.to(torch.bfloat16)
    model.config.use_cache = True
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(MODEL).save_pretrained(out)
    summary = {**config, "train_seconds": time.monotonic() - t0, "tokens_seen": seen_tok}
    (out / "train_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (out / "DONE").write_text("ok\n")
    if run:
        run.summary.update({"train_seconds": summary["train_seconds"]})
        run.finish()
    print(f"saved {out}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
