"""Stage-1 SFT warm-start for the AV and AR.

    uv run python -m reasoning_attention.training.sft --stage av --data out/av_sft.parquet
    uv run python -m reasoning_attention.training.sft --stage ar --data out/ar_sft.parquet

**AV (`h -> s`)** — embed the prompt, overwrite the placeholder row with the
scaled activation, and take causal-LM cross-entropy on the summary tokens only.
The loss is reported as nats/token plus perplexity.

**AR (`s -> h`)** — run the truncated backbone over the summary prompt, read the
residual at each row's final real token, apply the affine map, and take MSE
against the target. Reported alongside **FVE** (fraction of variance explained,
`1 - MSE/Var`), which is the paper's metric — the warm-start is expected to reach
roughly 0.3-0.4 before RL.

Defaults are deliberately untuned placeholders (see `--help`); the point is a
loop that runs and reports honestly, not a tuned recipe.

**Full fine-tuning is the default**, matching the reference repo (their actor SFT
is the full 28-layer model under FSDP, no adapters). Qwen3-1.7B in bf16 with AdamW
needs ~3.5 GB weights + 3.5 GB grads + 14 GB of fp32 moments ~= 21 GB before
activations: comfortable on an A100-80GB, impossible on a 16 GB card. Pass
`--lora` on a small GPU — it trains ~1% of the parameters and fits, at the cost of
deviating from the method. Prefer full fine-tuning for the AR in particular: it is
the Stage-2 reward model, and a rank-limited reward model is one the AV can game.
The AR's affine map is always trained in full regardless — it is newly initialized
and has no pretrained weights to adapt.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from reasoning_attention.config import NLAConfig
from reasoning_attention.nla.injection import (
    inject_at_placeholder,
    normalize_activation,
)
from reasoning_attention.nla.model import NLA, ARModel
from reasoning_attention.training.data import (
    ARBatch,
    ARCollator,
    ARDataset,
    AVBatch,
    AVCollator,
    AVDataset,
    predict_mean_baseline,
)

# Untuned defaults. Every one of these is a placeholder chosen to make the loop
# run on 16 GB, not a tuned value.
# --- Reference hyperparameters, copied from the Qwen2.5-7B case study in
# --- natural_language_autoencoders/configs/TRAINING_NOTES.md.
# Batch size is deliberately NOT copied: theirs is 256 on 2xH100-80GB.
# Their note stands: "These are the settings we used, not settings we claim are
# optimal. We did not sweep batch size, learning rate, or GRPO group size."
REFERENCE_LR = 2e-5  # both AV and AR ("matched to actor - worked well")
REFERENCE_MIN_LR = 2e-6  # cosine floor, = LR/10
REFERENCE_BATCH = 256  # their --global-batch-size, and the anchor for the sqrt LR rule
# Their per-stage micro-batch (--micro-batch-size), which we reproduce as
# batch_size x grad_accum on one GPU. The two halves differ because the AV runs
# the full 28 layers plus a 152k-vocab lm_head while the AR stops at layer 20 and
# ends in a d x d head: at m32+ the AV OOMed on their 80GB cards, the AR did not.
# Keeping their *effective* batch at 256 is what matters here — it is the batch
# their LR was tuned at, so the sqrt rule below resolves to exactly 2e-5.
REFERENCE_MICRO_BATCH = {"av": 16, "ar": 64}
# Their --attn-implementation per stage. FA2 is their measured best for the AV
# (36% faster than sdpa+checkpointing at m16) but needs the flash-attn package;
# resolve_attn falls back to sdpa when it is missing rather than crashing.
REFERENCE_ATTN = {"av": "flash_attention_2", "ar": "sdpa"}
# They ran with NO gradient checkpointing on either half — the AV fits at m16
# without it, and recompute cost exceeded the FSDP gather it saved.
REFERENCE_WARMUP_RATIO = 0.05  # 50 warmup iters / 1000 rollouts
REFERENCE_INJECTION_SCALE = 150.0  # their --nla-injection-scale for Qwen2.5-7B
REFERENCE_SAVE_INTERVAL = 500

DEFAULTS = {
    # Full fine-tuning is the default (matching the reference), so the reference
    # LR applies directly. --lora needs a much hotter LR; see LORA_LEARNING_RATE.
    "learning_rate": REFERENCE_LR,
    "min_lr": REFERENCE_MIN_LR,
    # Resolved per stage from REFERENCE_MICRO_BATCH once --stage is known; these
    # are only the fallbacks if that lookup is bypassed.
    "batch_size": 16,
    "grad_accum": 16,
    # 1. Their run length (`--num-rollout 1000`, ~250k samples at batch 256) is one
    # pass over their half, and one pass is what matters — not the step count. We
    # measured 3 epochs on our 99k halves (1143 steps): both models plateaued
    # inside epoch 0 and then drifted the wrong way. AV train loss 1.3642@500 ->
    # 1.4584@1143; AR FVE 0.2011@580 -> 0.1753@1143, while held-out beat final
    # train loss in both. The extra passes bought nothing but mild overfitting.
    "epochs": 1,
    "warmup_ratio": REFERENCE_WARMUP_RATIO,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    # Measured on the built parquets: AV totals mean 186 / p99 233 / max 276,
    # AR 94 / 149 / 223. 384 clears the longest observed row with headroom, and
    # 1024 just padded to 4x the needed width. See resolve_max_length.
    "max_length": 384,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "log_every": 10,
    "eval_frac": 0.02,  # held-out tail for the metric
    "save_interval": REFERENCE_SAVE_INTERVAL,
}

# Attention/MLP projections — the standard LoRA target set for Llama-family.
# LoRA convention, only used when --lora is passed.
LORA_LEARNING_RATE = 1e-4

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class StepLog:
    step: int
    loss: float
    lr: float
    grad_norm: float
    seconds: float
    # AR only.
    fve: float | None = None
    # AV only.
    perplexity: float | None = None
    # AR only — watched because a direction-only loss lets |pred| drift upward.
    pred_norm: float | None = None


def _scale_key(stage: str) -> str:
    """Which scale knob a stage actually uses — they are NOT the same thing (D30).

    AV: injection_scale, the L2 norm the activation is mapped to before its
        embedding is overwritten. A hyperparameter; wrong value => the AV reads
        an out-of-distribution vector.
    AR: mse_scale, the norm BOTH prediction and target are mapped to before the
        MSE. Loss normalization only; the AR is never injected into.
    """
    return "injection_scale" if stage == "av" else "mse_scale"


def cosine_with_floor(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, floor_ratio: float
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine decay from `lr` to `min_lr`, matching their `--min-lr`.

    `transformers.get_cosine_schedule_with_warmup` decays to exactly 0, which is
    not what the reference ran: it used `--lr 2e-5 --min-lr 2e-6`, i.e. a floor at
    LR/10. Reproducing that needs an explicit lambda.
    """

    def scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return floor_ratio + (1.0 - floor_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _apply_lora(model: Any, args: argparse.Namespace) -> Any:
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGETS,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model


def av_loss(model: Any, batch: AVBatch, config: NLAConfig, device: torch.device) -> torch.Tensor:
    """Cross-entropy on the summary tokens, with the activation injected."""
    input_ids = batch.input_ids.to(device)
    embeddings = model.get_input_embeddings()(input_ids)
    vectors = batch.activations.to(device=device, dtype=embeddings.dtype)
    injected = inject_at_placeholder(input_ids, embeddings, vectors, config.placeholder_token_id)
    out = model(
        inputs_embeds=injected,
        attention_mask=batch.attention_mask.to(device),
        labels=batch.labels.to(device),
    )
    loss: torch.Tensor = out.loss
    return loss


def ar_loss(
    model: ARModel, batch: ARBatch, device: torch.device, scale: float | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MSE at each row's final real token; returns (loss, predictions, targets).

    **Both** prediction and target are normalized to `scale` before the MSE, so
    this is a direction-only loss. That is what the reference repo's `mse_scale`
    does ("if a float, BOTH pred and gold are L2-normalized to that norm"), and
    it matters: comparing a free-magnitude prediction against a fixed-norm target
    makes the loss dominated by norm error rather than direction.

    The reference also warns that with a normalized loss the gradient is
    tangent-to-the-sphere and norm-neutral in *activation* space, but not in
    weight space — `|pred|` drifts upward roughly linearly with steps under Adam.
    `pred_norm` is logged so that drift is visible; the mitigations they name are
    a lower LR or an explicit norm term.
    """
    out = model(
        input_ids=batch.input_ids.to(device),
        attention_mask=batch.attention_mask.to(device),
    )
    rows = torch.arange(batch.input_ids.shape[0], device=device)
    predictions = out.activation_pred[rows, batch.last_index.to(device)]
    targets = batch.targets.to(device=device, dtype=predictions.dtype)
    loss = torch.nn.functional.mse_loss(
        normalize_activation(predictions.float(), scale),
        normalize_activation(targets.float(), scale),
    )
    return loss, predictions, targets


def fraction_variance_explained(mse: float, baseline: float) -> float:
    """`1 - MSE/baseline` — the paper's FVE.

    `baseline` is the achievable predict-the-mean loss from
    `data.predict_mean_baseline`, fixed for the dataset, so the number is
    comparable across steps and runs. The reference's warm-start reached
    **37.5%** by this measure (0.586 against a 0.938 baseline); joint RL takes it
    to 0.6-0.8.
    """
    if baseline == 0.0:
        return float("nan")
    return 1.0 - (mse / baseline)


def build_av(args: argparse.Namespace, config: NLAConfig) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    extra = {"attn_implementation": args.attn_implementation} if args.attn_implementation else {}
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id, dtype=torch.bfloat16, device_map={"": 0}, **extra
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if args.lora:
        model = _apply_lora(model, args)
    return model, tokenizer


def build_ar(args: argparse.Namespace, config: NLAConfig) -> tuple[ARModel, Any]:
    """Truncated backbone + affine map, reusing the NLA loader for the truncation."""
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    backbone, d_model = NLA._load_truncated_backbone(  # noqa: SLF001 - same package
        config, torch.bfloat16, {"": 0}, True
    )
    backbone.config.use_cache = False
    if args.gradient_checkpointing:
        backbone.gradient_checkpointing_enable()
    if args.lora:
        backbone = _apply_lora(backbone, args)
    model = ARModel(backbone, d_model=d_model, bias=config.ar_affine_bias)
    # The affine map is new — always trained in full, whatever the backbone does.
    model.affine.float()
    for param in model.affine.parameters():
        param.requires_grad = True
    return model, tokenizer


def save_checkpoint(model: Any, tokenizer: Any, stage: str, out_dir: Path) -> None:
    """Persist one checkpoint. Called at `--save-interval` and at the end.

    The AV is a PeftModel (or a causal LM) and saves itself. `ARModel` is not a
    `PreTrainedModel`, so its backbone and its affine map are written separately
    and reload independently.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if stage == "av":
        model.save_pretrained(str(out_dir))
    else:
        model.backbone.save_pretrained(str(out_dir / "backbone"))
        torch.save(model.affine.state_dict(), out_dir / "affine.pt")
    tokenizer.save_pretrained(str(out_dir))


def resolve_attn(requested: str | None, stage: str) -> str | None:
    """Their per-stage attention choice, degraded to sdpa if FA2 is unavailable.

    flash_attention_2 is a separate package and a source build; asking for it on
    a box without it raises inside `from_pretrained`, which is a poor way to lose
    a training run. sdpa is always present and is what they used for the AR anyway.
    """
    if requested is not None:
        return requested
    choice = REFERENCE_ATTN.get(stage)
    if choice == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            print(
                "attn: flash_attention_2 unavailable (flash-attn not installed) -> sdpa. "
                "Their AV run measured FA2 at m16 as 36% faster; `uv pip install flash-attn "
                "--no-build-isolation` to match it."
            )
            return "sdpa"
    return choice


def init_wandb(args: argparse.Namespace, total_steps: int, effective_batch: int) -> Any:
    """Start a wandb run, or return None when disabled/unavailable.

    Off unless --wandb is passed: a training run must not fail because a metrics
    service is unreachable, and offline smoke runs should stay silent.
    """
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError:
        print("wandb: not installed — skipping (uv add wandb)")
        return None
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_name
        or f"sft-{args.stage}{'-rand' if args.shuffle_activations else ''}"
        f"{'-masked-' + args.mask_explanation if args.mask_explanation else ''}",
        config={
            **{k: getattr(args, k) for k in DEFAULTS},
            "stage": args.stage,
            "lora": args.lora,
            "shuffle_activations": args.shuffle_activations,
            "mask_explanation": args.mask_explanation,
            "effective_batch": effective_batch,
            "total_steps": total_steps,
            "attn_implementation": args.attn_implementation,
            "model_id": NLAConfig().model_id,
        },
    )
    print(f"wandb: logging to {run.url}")
    return run


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", required=True, choices=["av", "ar"])
    parser.add_argument("--data", required=True, help="stage-3 parquet for this half")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N rows")
    parser.add_argument(
        "--lora",
        action="store_true",
        help="train LoRA adapters instead of full fine-tuning. Only needed when the "
        "GPU cannot hold ~21 GB of optimizer state (e.g. a 16 GB card); on A100-80GB "
        "full fine-tuning fits and is what the reference repo does.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="reference ran WITHOUT this (m16 fits and is 36%% faster); it also "
        "deadlocks NCCL in their RL update_weights()",
    )
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help="reference used flash_attention_2 for the AV and sdpa for the AR; "
        "default None lets transformers choose (FA2 needs the flash-attn package)",
    )
    parser.add_argument(
        "--injection-scale",
        default=None,
        help=f"override NLAConfig.injection_scale. The reference picks a round number "
        f"just above the dataset's mean activation norm — {REFERENCE_INJECTION_SCALE} for "
        f"Qwen2.5-7B (mean ~125), 80000 for Gemma-3-12B, 30 for Llama-3.3-70B. Our h_l "
        f"norms average ~900, so we default to 1000. Accepts a float, 'sqrt_d_model', "
        f"or 'raw'.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mask-explanation",
        default=None,
        choices=["quotes", "last-feature", "both"],
        help="AR ABLATION: strip surface-form hints (quoted literal tokens, and/or the "
        "final feature the prompt reserves for describing the sequence end) before the "
        "AR reads the explanation. Tests whether FVE comes from semantics or from being "
        "told the last token's identity. AR only — for the AV the explanation is the "
        "target, not the input.",
    )
    parser.add_argument(
        "--shuffle-activations",
        action="store_true",
        help="RANDOM CONTROL: pair each explanation with another row's activation. "
        "Any score above chance is then generic prior, not vector-reading. The "
        "FVE baseline is unchanged, so real-vs-control FVE compares directly.",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="log metrics to wandb (off by default)"
    )
    parser.add_argument("--wandb-project", default="doomloop-nla-sft")
    parser.add_argument("--wandb-name", default=None, help="run name; defaults to sft-<stage>")
    for key, value in DEFAULTS.items():
        parser.add_argument(
            f"--{key.replace('_', '-')}", type=type(value), default=value, help=f"default {value}"
        )
    args = parser.parse_args()
    # Their micro-batch differs per half; reproduce it as batch x accum on one GPU
    # while holding the effective batch at their 256 so the LR rule stays valid.
    if args.batch_size == DEFAULTS["batch_size"] and args.grad_accum == DEFAULTS["grad_accum"]:
        micro = REFERENCE_MICRO_BATCH[args.stage]
        args.batch_size = micro
        args.grad_accum = max(1, REFERENCE_BATCH // micro)
    args.attn_implementation = resolve_attn(args.attn_implementation, args.stage)

    torch.manual_seed(args.seed)
    config = NLAConfig()
    if args.injection_scale is not None:
        try:
            override: float | str = float(args.injection_scale)
        except ValueError:
            override = args.injection_scale
        config = replace(config, injection_scale=override)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Same seed for both halves so the control is reproducible; None = real pairing.
    shuffle_seed = args.seed if args.shuffle_activations else None
    if shuffle_seed is not None:
        print(
            f"RANDOM CONTROL: activations shuffled (seed {shuffle_seed}) — text no longer "
            f"matches its vector. Expect FVE ~= 0 / a visibly worse loss than the real run."
        )
    if args.mask_explanation and args.stage == "av":
        raise SystemExit(
            "--mask-explanation applies to the AR only: for the AV the explanation is the "
            "generation target, not the input, so masking it changes the task rather than "
            "removing a shortcut."
        )
    if args.stage == "av":
        model, tokenizer = build_av(args, config)
        dataset: Any = AVDataset(
            args.data,
            tokenizer,
            config,
            max_length=args.max_length,
            limit=args.limit,
            shuffle_seed=shuffle_seed,
        )
        collate: Any = AVCollator(tokenizer.pad_token_id or 0, dataset.scale)
    else:
        model, tokenizer = build_ar(args, config)
        dataset = ARDataset(
            args.data,
            tokenizer,
            config,
            max_length=args.max_length,
            limit=args.limit,
            shuffle_seed=shuffle_seed,
            mask_mode=args.mask_explanation,
        )
        collate = ARCollator(tokenizer.pad_token_id or 0, dataset.scale)

    # Fixed FVE denominator for the AR — the achievable predict-the-mean loss,
    # the reference's `load_predict_mean_baselines`.
    baseline = (
        predict_mean_baseline(dataset.activations, dataset.scale) if args.stage == "ar" else 0.0
    )

    n_eval = max(1, int(len(dataset) * args.eval_frac)) if len(dataset) > 20 else 0
    n_train = len(dataset) - n_eval
    train_set = torch.utils.data.Subset(dataset, range(n_train))
    eval_set = torch.utils.data.Subset(dataset, range(n_train, len(dataset))) if n_eval else None

    loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate, drop_last=False
    )
    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = max(1, steps_per_epoch * args.epochs)

    effective_batch = args.batch_size * args.grad_accum
    if not args.lora and args.learning_rate == DEFAULTS["learning_rate"]:
        # Re-derive from the reference setting rather than assuming their batch.
        args.learning_rate = REFERENCE_LR * math.sqrt(effective_batch / REFERENCE_BATCH)
        print(
            f"full-finetune: LR re-derived to {args.learning_rate:.2e} "
            f"(reference {REFERENCE_LR:.0e} @ batch {REFERENCE_BATCH}, sqrt-scaled to "
            f"batch {effective_batch})"
        )

    if args.lora and args.learning_rate == DEFAULTS["learning_rate"]:
        # The reference LR is tuned for full fine-tuning and is ~5x too cold for
        # adapters, which see far fewer parameters per step.
        args.learning_rate = LORA_LEARNING_RATE
        print(f"lora: LR raised to {args.learning_rate:.2e} (full-finetune default is too cold)")

    # Their 500 means "halfway + final" at their 1000 steps. Keep the literal
    # value, but never let it silently never-fire on a shorter run: a save_interval
    # past the end of training yields no intermediate checkpoint at all, which is
    # only discovered when a run dies at step 900 and there is nothing to resume.
    if args.save_interval and args.save_interval >= total_steps:
        fallback = max(1, total_steps // 2)
        print(
            f"save_interval {args.save_interval} >= total_steps {total_steps} — would never "
            f"fire; using {fallback} (halfway + final, their intent)"
        )
        args.save_interval = fallback

    wandb_run = init_wandb(args, total_steps, effective_batch)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = cosine_with_floor(
        optimizer,
        warmup_steps=int(total_steps * args.warmup_ratio),
        total_steps=total_steps,
        floor_ratio=args.min_lr / args.learning_rate if args.learning_rate else 0.0,
    )

    n_trainable = sum(p.numel() for p in trainable)
    print(
        f"stage={args.stage} rows={len(dataset)} (train {n_train} / eval {n_eval}) "
        f"steps={total_steps} trainable={n_trainable / 1e6:.1f}M "
        f"lora={f'r{args.lora_r}' if args.lora else 'off (full finetune)'} "
        f"lr={args.learning_rate:.2e} eff_batch={effective_batch} "
        f"{_scale_key(args.stage)}={dataset.scale}"
        + (f" fve_baseline={baseline:.4f}" if args.stage == "ar" else "")
    )

    out_dir = Path(args.output_dir) / f"{args.stage}_sft"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs: list[StepLog] = []
    model.train()
    step = 0
    started = time.time()

    for epoch in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        for micro, batch in enumerate(loader):
            if args.stage == "av":
                loss = av_loss(model, batch, config, device)
                extra: dict[str, float | None] = {"perplexity": float(torch.exp(loss.detach()))}
            else:
                loss, predictions, targets = ar_loss(model, batch, device, dataset.scale)
                extra = {
                    "fve": fraction_variance_explained(float(loss.detach()), baseline),
                    "pred_norm": float(predictions.detach().float().norm(dim=-1).mean()),
                }
            (loss / args.grad_accum).backward()

            is_last = micro == len(loader) - 1
            if (micro + 1) % args.grad_accum == 0 or is_last:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if args.save_interval and step % args.save_interval == 0:
                    interim = out_dir / f"step_{step:07d}"
                    save_checkpoint(model, tokenizer, args.stage, interim)
                    print(f"  checkpoint -> {interim}")

                if step % args.log_every == 0 or step == 1 or step == total_steps:
                    entry = StepLog(
                        step=step,
                        loss=float(loss.detach()),
                        lr=float(scheduler.get_last_lr()[0]),
                        grad_norm=float(grad_norm),
                        seconds=round(time.time() - started, 1),
                        **extra,
                    )
                    logs.append(entry)
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                f"train/{k}": v
                                for k, v in asdict(entry).items()
                                if k != "step" and v is not None
                            },
                            step=entry.step,
                        )
                    metric = (
                        f"ppl {entry.perplexity:.2f}"
                        if entry.perplexity is not None
                        else f"FVE {entry.fve:.4f}"
                    )
                    print(
                        f"  epoch {epoch} step {step}/{total_steps} "
                        f"loss {entry.loss:.4f} {metric} "
                        f"lr {entry.lr:.2e} gnorm {entry.grad_norm:.2f} "
                        f"({entry.seconds}s)"
                    )

    # Held-out pass. For the AR this is the number that matters: the paper's
    # warm-start lands around 0.3-0.4 FVE.
    eval_summary: dict[str, float] = {}
    if eval_set is not None:
        model.eval()
        eval_loader = DataLoader(
            eval_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate
        )
        losses: list[float] = []
        preds: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in eval_loader:
                if args.stage == "av":
                    losses.append(float(av_loss(model, batch, config, device)))
                else:
                    loss, predictions, _ = ar_loss(model, batch, device, dataset.scale)
                    losses.append(float(loss))
                    preds.append(predictions.detach().float().cpu())
        eval_summary["loss"] = sum(losses) / len(losses)
        if args.stage == "av":
            eval_summary["perplexity"] = math.exp(eval_summary["loss"])
        else:
            eval_summary["fve"] = fraction_variance_explained(eval_summary["loss"], baseline)
            eval_summary["pred_norm"] = float(torch.cat(preds).norm(dim=-1).mean())
        print(f"eval: {json.dumps({k: round(v, 4) for k, v in eval_summary.items()})}")
        if wandb_run is not None:
            wandb_run.summary.update({f"eval/{k}": v for k, v in eval_summary.items()})

    if wandb_run is not None:
        # The dataset-level FVE denominator, so a run is interpretable on its own.
        wandb_run.summary["fve_baseline"] = baseline
        # Name the knob by what it IS for this stage. The AV injects at
        # injection_scale (1000); the AR injects nothing — its scale is mse_scale
        # (sqrt(d_model) = 45.25), the norm BOTH prediction and target are mapped to
        # before the loss. Logging both under one name has twice led to the AR's
        # 45.25 being read back as the AV's injection scale (see D30).
        wandb_run.summary[_scale_key(args.stage)] = dataset.scale
        wandb_run.finish()

    save_checkpoint(model, tokenizer, args.stage, out_dir)

    (out_dir / "train_log.json").write_text(
        json.dumps(
            {
                "stage": args.stage,
                "args": {k: v for k, v in vars(args).items()},
                "rows": len(dataset),
                "trainable_params": n_trainable,
                _scale_key(args.stage): dataset.scale,
                "fve_baseline": baseline,
                "steps": [asdict(entry) for entry in logs],
                "eval": eval_summary,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
