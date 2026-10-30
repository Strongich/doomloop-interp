# Targets that rebuild every result in the VERS paper. Included by the Makefile.
# `make repro-help` lists them. README.md ("Reproducing the paper") gives the order,
# the hardware, and the steps that have no script yet.
#
# Most steps ran on a pod with /workspace mounted and /workspace/env.sh sourced
# (HF_HOME, uv cache). The queue scripts write job commands that use these paths.

PY      ?= uv run python
RLPY    ?= .venv-rl/bin/python
WS      ?= /workspace
HFUSER  ?= Strongich
QUEUE    = $(PY) scripts/gpu_queue.py

# Rollout engine (vLLM with the steering hook) and the flags shared by every run.
RUN     = $(PY) scripts/reasoning_policy_vllm.py --stream --max-num-batched-tokens 8192
M17     = --batch 256 --gpu-memory-utilization 0.9
M8      = --model Qwen/Qwen3-8B --batch 128 --gpu-memory-utilization 0.9
M30     = --model Qwen/Qwen3-30B-A3B --batch 64 --gpu-memory-utilization 0.93
M235    = --model Qwen/Qwen3-235B-A22B-FP8 --tp 4 --batch 128 --checkpoint-every 32 \
          --gpu-memory-utilization 0.9

N_DIR   = data/pool/dir_A_1trace.pt
DEV400  = data/policy/dev400.jsonl
CONF400 = data/xfer8b/confirm_mathtest400.jsonl
HARD128 = data/baselines/hard128.jsonl
X8      = data/xfer8b/dirs_clean
X30     = data/xfer30b/dirs_L34
X235    = data/xfer235b/dirs_clean

.PHONY: repro-help checkpoints traces \
        nla-warmstart-data nla-sft nla-rl-data nla-rl nla-fve \
        explanations directions construction dir-D dir-P local-edit \
        policy-select policy-controls policy-eval policy-regrade gsm8k-full neg-n \
        dom150 kv-cache long-trace continuation strength-sweep trace-audit \
        transfer-cohorts transfer-fit transfer-maps transfer-screen transfer-dev \
        transfer-confirm transfer-controls transfer-regrade transfer-fidelity \
        transfer-reading transfer-interface \
        baselines baselines-amend-e baselines-amend-f baselines-report \
        selective gated vers-p \
        distill-1.7b distill-1.7b-dpo distill-1.7b-gated distill-8b distill-offline \
        figures

repro-help: ## List the paper reproduction targets
	@grep -hE '^[a-zA-Z0-9_.-]+:.*## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "} {printf "  %-22s %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# 0. Inputs
# ---------------------------------------------------------------------------
checkpoints: ## Download the released NLA (AV + AR, Qwen3-1.7B layer 20)
	hf download $(HFUSER)/nla-qwen3-1.7b-L20-av --local-dir checkpoints/av_rl_ep1
	hf download $(HFUSER)/nla-qwen3-1.7b-L20-ar --local-dir checkpoints/ar_rl_ep1

traces: ## 1.7B reasoning traces (all GSM8K, AIME 2025, AMC 23) and the correct-trace sample
	GSM8K_LIMIT= bash scripts/generate_traces.sh
	$(PY) scripts/build_trace_corpus.py --out data/traces_all.jsonl
	$(PY) scripts/build_trace_sample.py --only-correct --max-pairs 600 --out data/correct_sample.jsonl

# ---------------------------------------------------------------------------
# 1. NLA training (§3, App. H). Skip it and use `make checkpoints` instead.
#    2x A100-80GB. Labelling ~10 h per 200k rows, GRPO ~5.5 days.
# ---------------------------------------------------------------------------
nla-warmstart-data: ## Warm-start pairs: 40k docs, then extend to 100k (local Qwen3.8-27B labeller)
	DATA_DIR=$(WS)/data/warmstart INSTRUCTION=reference STEM=_ref NO_SHUTDOWN=1 \
		bash scripts/label_and_shutdown.sh
	DATA_DIR=$(WS)/data/warmstart N_DOCS=100000 INSTRUCTION=reference STEM=_ref NO_SHUTDOWN=1 \
		bash scripts/label_and_shutdown.sh

nla-sft: ## Warm-start SFT of the AV and the AR (one GPU each, effective batch 256)
	for m in av ar; do \
		gpu=$$([ $$m = av ] && echo 0 || echo 1); \
		CUDA_VISIBLE_DEVICES=$$gpu DATA_DIR=$(WS)/data/warmstart OUTPUT_DIR=$(WS)/data/checkpoints \
		EPOCHS=1 MAX_LENGTH=384 BATCH_SIZE=16 GRAD_ACCUM=16 \
		AV_DATA=$(WS)/data/warmstart/av_ref_sft.parquet AR_DATA=$(WS)/data/warmstart/ar_ref_sft.parquet \
		bash scripts/train_sft.sh $$m & \
	done; wait

nla-rl-data: ## Joint-training prompts: 393,439 web + chat positions
	DATA_DIR=$(WS)/data/rl bash scripts/build_rl_data.sh

nla-rl: ## GRPO joint training, one epoch (6,147 steps), then export av_rl_ep1 / ar_rl_ep1
	bash scripts/setup_rl_stack.sh
	$(RLPY) scripts/patch_nla_nonthinking.py
	$(RLPY) scripts/convert_ar_to_critic.py --ar-checkpoint $(WS)/data/checkpoints/ar_sft \
		--output $(WS)/data/checkpoints/critic_rl
	RL_PARQUET=$(WS)/data/rl/rl.parquet ACTOR_SFT_CKPT=$(WS)/data/checkpoints/av_sft \
		CRITIC_SL_CKPT=$(WS)/data/checkpoints/critic_rl RUN_DIR=$(WS)/data/runs/grpo_main \
		DISTRIBUTED_POST=1 NUM_ROLLOUT=3074 SAVE_INTERVAL=250 bash scripts/train_grpo.sh
	RL_PARQUET=$(WS)/data/rl/rl.parquet ACTOR_SFT_CKPT=$(WS)/data/checkpoints/av_sft \
		CRITIC_SL_CKPT=$(WS)/data/runs/grpo_main/critic/iter_0003074/hf RUN_DIR=$(WS)/data/runs/grpo_main \
		DISTRIBUTED_POST=1 NUM_ROLLOUT=6147 SAVE_INTERVAL=250 bash scripts/train_grpo.sh
	$(RLPY) .rl-src/miles/tools/convert_fsdp_to_hf.py \
		--input-dir $(WS)/data/runs/grpo_main/actor/iter_0006147 \
		--output-dir checkpoints/av_rl_ep1 --origin-hf-dir $(WS)/data/checkpoints/av_sft -f
	cp -r $(WS)/data/runs/grpo_main/critic/iter_0006147/hf checkpoints/ar_rl_ep1

nla-fve: transfer-fit ## Source FVE on web / chat / reasoning states (§3; needs the transfer pairs)
	$(PY) scripts/xfer_fidelity.py build
	$(PY) scripts/xfer_fidelity.py extract --model Qwen/Qwen3-1.7B --layer 20 --tag s
	$(PY) scripts/xfer_fidelity.py verbalize --which s
	$(PY) scripts/xfer_fidelity.py report

# ---------------------------------------------------------------------------
# 2. Directions and the local edit test (§3.2, §4.1, App. A)
# ---------------------------------------------------------------------------
explanations: ## AV explanations at 297 doubt / plain boundary pairs
	$(PY) scripts/verbalize_blocks.py --limit 300

construction: ## N (data/pool/dir_A_1trace.pt) and the construction ablation (Table A)
	bash scripts/run_tier01.sh

directions: construction dir-D dir-P ## N, D (difference of means) and P (doubt edit), -N

dir-D: ## D and the logistic-probe direction
	$(PY) scripts/build_dom_directions.py

dir-P: ## P, -N and the rebuilt N' (explanation figure)
	$(PY) scripts/vers_derive.py --out-dir data/vers

local-edit: ## §4.1 local edit test at alpha 0.25 (induction and suppression)
	$(PY) scripts/steer_sweep.py --limit 200 --samples 4 --alpha 0.25
	$(PY) scripts/steer_sweep.py --kind doubt_wait --mode suppress --limit 200 --samples 8 \
		--alpha 0.25 --out data/steer_suppress_a0.25.csv \
		--dump-examples data/steer_suppress_examples_a0.25.json --n-examples 8
	$(PY) scripts/steer_report.py data/steer_suppress_a0.25.csv

# ---------------------------------------------------------------------------
# 3. The steering policy on Qwen3-1.7B (§4, App. B, App. C)
# ---------------------------------------------------------------------------
policy-select: ## dev200 delay x strength screen, dev400 stage 2 (4x A100, ~2 h)
	$(PY) scripts/audit_question_exposure.py
	$(PY) scripts/build_policy_cohort.py
	LOGDIR=$(WS) bash scripts/run_policy_chain.sh

policy-controls: ## Random-direction controls on dev400 (costs used by policy-eval)
	SHARDS=3 STAGE=2 SHORTLIST="R101@a1.0d256 R202@a1.0d256 R303@a1.0d256" CONTROLS= \
		OUTROOT=data/reasoning_policy_v1_controls PLAN=data/policy/shards_controls.json \
		LOGDIR=$(WS) bash scripts/run_policy_sharded.sh

policy-eval: ## Frozen policy on gsm997 and MATH-500 (4x A100, ~6 h)
	$(PY) scripts/build_policy_cohort.py --math500
	$(PY) scripts/build_policy_cohort.py --fresh 997 --seed 20260922
	LOGROOT=$(WS)/eval_logs bash scripts/run_policy_eval.sh

policy-regrade: ## Regrade with the corrected grader; MATH-500 table and levels
	$(PY) scripts/regrade_journals.py 'data/eval_math500/**' 'data/eval_gsm8k_fresh/**'
	$(PY) paper/scripts/regrade_policy_tables.py

gsm8k-full: ## Full GSM8K test, 1,319 x 4 seeds: base, N, concision prompt (1 GPU, ~3.5 h)
	$(PY) scripts/build_policy_cohort.py --full-test
	$(RUN) $(M17) --cohort data/policy/gsm8k_test_1319.jsonl --outdir data/eval_gsm8k_full/run \
		--max-new-tokens 16384 --seeds 4 --policies base N@a1.0d256 brevityA
	$(PY) scripts/negn_gsm8k_report.py gsm8k

neg-n: vers-p ## -N on dev400 x 2 seeds against P (§6.3, App. F; 1 GPU, ~9 h)
	$(RUN) $(M17) --dir negN=data/vers/dir_negN.pt --cohort $(DEV400) --outdir data/vers/negN_dev \
		--max-new-tokens 16384 --seeds 2 --policies base negN@a0.25d256 negN@a0.5d256 negN@a1.0d256
	$(PY) scripts/negn_gsm8k_report.py negn

dom150: ## N vs D vs probe direction on dom150 at matched schedule (§4.3)
	$(PY) scripts/build_hard832.py
	bash scripts/run_tier3_step1.sh
	$(PY) scripts/build_dom150.py
	bash scripts/run_dom_stageA.sh

kv-cache: ## App. C.1 direct cache test: probability ratios at 1.93M tokens (needs dom150)
	$(PY) scripts/logprob_ratio.py --limit 789

long-trace: ## App. C.1 stability over long answers
	$(PY) scripts/suppress_answer.py --limit 120 --alpha 1.0 --batch 6 --max-new-tokens 6144 \
		--out data/suppress_answer_a1.0.csv --dump data/suppress_answer_texts_a1.0.jsonl
	$(PY) scripts/contamination_curve.py

continuation: ## Continuation value on cont387 (fixed-prefix branches, ~8 h)
	bash scripts/run_after_answer.sh
	bash scripts/run_prefix_online.sh
	$(PY) scripts/prefix_report.py --csv data/prefix_online_vllm/branches.csv

strength-sweep: ## alpha sweep on MATH-500 levels 4-5 (4x A100, ~1.7 h)
	$(PY) -c "import json; [print(l, end='') for l in open('data/policy/math500.jsonl') \
		if json.loads(l)['level'] >= 4]" > data/policy/math500_L45.jsonl
	SHARDS=4 STAGE=2 COHORT=data/policy/math500_L45.jsonl SEEDS=4 MAXNEW=32768 \
		SHORTLIST="N@a0.25d256 N@a0.5d256 N@a0.75d256" CONTROLS= REPLICATE=0 PAIR=0 \
		OUTROOT=data/eval_math500_alpha PLAN=data/policy/shards_math500_alpha.json \
		LOGDIR=$(WS) bash scripts/run_policy_sharded.sh
	$(PY) paper/scripts/regrade_policy_tables.py

trace-audit: ## Blind trace-quality audit, 200 pairs (needs OPENAI_API_KEY in .env)
	$(PY) scripts/audit_trace_quality.py build
	$(PY) scripts/audit_trace_quality.py judge --effort low
	$(PY) scripts/audit_trace_quality.py report

# ---------------------------------------------------------------------------
# 4. Transfer to Qwen3-8B, 30B-A3B, 235B-A22B (§5, App. E)
#    8B/30B: one A100 per run. 235B: 4x A100 with tensor parallelism (never --pp).
# ---------------------------------------------------------------------------
transfer-cohorts: ## dev250, fit400, web3000, confirm400
	$(PY) scripts/xfer_build_cohorts.py
	$(PY) scripts/xfer_build_confirm.py

transfer-fit: ## Base traces on fit400 and paired activations for the affine maps
	$(RUN) --cohort data/xfer8b/fit_math400.jsonl --outdir data/xfer8b/r17_fit_base --policies base \
		--seeds 1 --max-new-tokens 16384 --batch 192 --checkpoint-every 384
	$(RUN) $(M8) --cohort data/xfer8b/fit_math400.jsonl --outdir data/xfer8b/r8b_fit_base \
		--policies base --seeds 1 --max-new-tokens 16384 --checkpoint-every 256
	$(RUN) $(M30) --batch 48 --cohort data/xfer8b/fit_math400.jsonl --outdir data/xfer30b/r30_fit_base \
		--policies base --seeds 1 --max-new-tokens 16384
	$(PY) scripts/xfer_corruption_probe.py --pp 1 --tp 4 --batch 128 --stream
	$(RUN) $(M235) --cohort data/xfer8b/fit_math400.jsonl --outdir data/xfer235b/r235_fit_base \
		--policies base --seeds 1 --max-new-tokens 16384
	$(PY) scripts/xfer_extract_pairs.py --out data/xfer8b/pairs.pt
	$(PY) scripts/xfer_extract_pairs.py --big-model Qwen/Qwen3-30B-A3B \
		--layers 20 24 28 30 32 35 38 41 --out data/xfer30b/pairs.pt
	$(PY) scripts/xfer_extract_pairs.py --big-model Qwen/Qwen3-30B-A3B --layers 34 \
		--out data/xfer30b/pairs_L34.pt
	$(PY) scripts/xfer_extract_pairs_vllm.py --phase small --out data/xfer235b/pairs.pt \
		--target rtg=data/xfer235b/r235_fit_base/rollouts.jsonl
	$(PY) scripts/xfer_extract_pairs_vllm.py --phase big --out data/xfer235b/pairs.pt \
		--target rtg=data/xfer235b/r235_fit_base/rollouts.jsonl --pp 1 --tp 4

transfer-maps: ## Affine maps and mapped directions (NU, DU, RU, ...) per target layer
	$(PY) scripts/xfer_fit_maps.py --pairs data/xfer8b/pairs.pt --out $(X8)
	$(PY) scripts/xfer_fit_maps.py --pairs data/xfer30b/pairs_L34.pt --out $(X30)
	$(PY) scripts/xfer_fit_maps.py --pairs data/xfer235b/pairs.pt --out $(X235)

transfer-screen: ## Single-site screen that chose the layer (sites reused by transfer-reading)
	$(PY) scripts/xfer_local_injection.py --model Qwen/Qwen3-1.7B \
		--spec 20:N:$(N_DIR) 20:D:data/dom/dir_D_diffmeans.pt 20:R:data/dirs_random/dir_R101.pt \
		--alphas -1 -0.5 0.25 0.5 1 1.5 2 --out data/xfer8b/local/17b.jsonl
	$(PY) scripts/xfer_local_injection.py --model Qwen/Qwen3-8B \
		--spec $(foreach l,18 22 26 30,$(foreach d,NU NEt D8 RU,$(l):$(d):$(X8)/L$(l)_$(d).pt)) \
		--alphas -1 -0.5 0.25 0.5 1 1.5 2 --out data/xfer8b/local/8b.jsonl
	$(PY) scripts/xfer_local_injection.py --model Qwen/Qwen3-30B-A3B \
		--spec $(foreach d,NU NEt D8 RU,34:$(d):$(X30)/L34_$(d).pt) \
		--alphas -1 -0.5 0.25 0.5 1 1.5 --out data/xfer30b/local/30b_L34.jsonl
	$(PY) scripts/xfer_local_injection_vllm.py --model Qwen/Qwen3-235B-A22B-FP8 --layer 67 --tp 4 \
		--spec NU:$(X235)/L67_NU.pt RU:$(X235)/L67_RU.pt --alphas -0.5 0.25 0.5 1 \
		--traces data/xfer235b/r235_fit_base/rollouts.jsonl --out data/xfer235b/local/235b.jsonl

transfer-dev: ## Strength sweeps on dev400 (rule: dAcc >= -1 pp, caps <= 2x base)
	$(RUN) $(M8) --layer 22 --dir NU22=$(X8)/L22_NU.pt --cohort $(DEV400) \
		--outdir data/xfer8b/r8b_gdev_g0 --policies base NU22@a1.0d256 --seeds 4 --max-new-tokens 16384
	$(RUN) $(M8) --layer 22 --dir NU22=$(X8)/L22_NU.pt --cohort $(DEV400) \
		--outdir data/xfer8b/r8b_gdev_g1 --policies NU22@a1.0d256g3 NU22@a0.75d256 --seeds 4 \
		--max-new-tokens 16384
	$(RUN) $(M8) --batch 192 --layer 26 --dir NU26=$(X8)/L26_NU.pt --cohort $(DEV400) \
		--outdir data/xfer8b/r8b_gdev_L26 --policies NU26@a0.75d256 --seeds 4 --max-new-tokens 16384
	$(RUN) $(M30) --batch 48 --cohort $(DEV400) --outdir data/xfer30b/r30_gdev_base --policies base \
		--seeds 4 --max-new-tokens 16384
	for a in 0.5 0.75 1.0; do \
		$(RUN) $(M30) --layer 34 --dir NU34=$(X30)/L34_NU.pt --cohort $(DEV400) \
		--outdir data/xfer30b/r30_gdev_a$$(echo $$a | tr -d .)0 --policies NU34@a$${a}d256 \
		--seeds 2 --max-new-tokens 16384; done
	$(RUN) $(M235) --layer 67 --dir NU67=$(X235)/L67_NU.pt --cohort $(DEV400) \
		--outdir data/xfer235b/r235_gdev_sweep --seeds 2 --max-new-tokens 16384 \
		--policies base NU67@a0.75d256 NU67@a1.0d256 NU67@a1.5d256 NU67@a2.0d256

transfer-confirm: ## Transfer confirmation on confirm400 x 4 seeds (Table 3)
	for h in 0 1; do \
		$(RUN) --batch 192 --checkpoint-every 200 --cohort data/xfer8b/confirm_h$$h.jsonl \
		--outdir data/xfer8b/r17_confirm_h$$h --policies base N@a1.0d256 --seeds 4 --max-new-tokens 32768; \
		$(RUN) $(M8) --layer 22 --dir NU22=$(X8)/L22_NU.pt \
		--cohort data/xfer8b/confirm_h$$h.jsonl --outdir data/xfer8b/r8b_confirm_h$$h \
		--policies base NU22@a0.75d256 --seeds 4 --max-new-tokens 32768; done
	$(RUN) $(M8) --batch 192 --checkpoint-every 400 --layer 26 --dir NU26=$(X8)/L26_NU.pt \
		--cohort $(CONF400) --outdir data/xfer8b/r8b_confirm_L26 --policies NU26@a0.75d256 \
		--seeds 4 --max-new-tokens 32768
	$(RUN) $(M30) --layer 34 --dir NU34=$(X30)/L34_NU.pt --cohort $(CONF400) \
		--outdir data/xfer30b/r30_confirm_base --policies base --seeds 4 --max-new-tokens 32768
	$(RUN) $(M30) --layer 34 --dir NU34=$(X30)/L34_NU.pt --cohort $(CONF400) \
		--outdir data/xfer30b/r30_confirm_arm --policies NU34@a0.75d256 --seeds 4 --max-new-tokens 32768
	$(RUN) $(M235) --layer 67 --dir NU67=$(X235)/L67_NU.pt --cohort $(CONF400) \
		--outdir data/xfer235b/r235_confirm --policies base NU67@a1.5d256 --seeds 4 --max-new-tokens 32768

transfer-controls: ## Random and native-D rows of the transfer control table
	$(RUN) $(M8) --layer 26 --dir RU26=$(X8)/L26_RU.pt --cohort $(CONF400) \
		--outdir data/xfer8b/r8b_confirm_L26_ru --policies RU26@a0.75d256 --seeds 4 --max-new-tokens 32768
	$(RUN) $(M30) --layer 34 --dir RU34=$(X30)/L34_RU.pt --cohort $(CONF400) \
		--outdir data/xfer30b/r30_confirm_ru --policies RU34@a0.75d256 --seeds 4 --max-new-tokens 32768
	$(RUN) $(M235) --layer 67 --dir RU67=$(X235)/L67_RU.pt --cohort $(DEV400) \
		--outdir data/xfer235b/r235_gdev_ru --policies RU67@a1.5d256 --seeds 2 --max-new-tokens 16384
	$(RUN) $(M8) --layer 26 --dir D26=$(X8)/L26_D8.pt --cohort $(DEV400) --outdir data/xfer8b/d26_gdev_sweep \
		--policies D26@a0.5d256 D26@a0.75d256 D26@a1.0d256 --seeds 2 --max-new-tokens 16384
	$(RUN) $(M30) --layer 34 --dir D34=$(X30)/L34_D8.pt --cohort $(DEV400) --outdir data/xfer30b/d34_gdev_sweep \
		--policies D34@a0.5d256 D34@a0.75d256 D34@a1.0d256 --seeds 2 --max-new-tokens 16384
	$(RUN) $(M8) --layer 26 --dir D26=$(X8)/L26_D8.pt --cohort $(CONF400) --outdir data/xfer8b/d26_confirm \
		--policies D26@a0.5d256 --seeds 4 --max-new-tokens 32768
	$(RUN) $(M30) --layer 34 --dir D34=$(X30)/L34_D8.pt --cohort $(CONF400) --outdir data/xfer30b/d34_confirm \
		--policies D34@a0.5d256 --seeds 4 --max-new-tokens 32768

transfer-regrade: ## Corrected grader on the transfer runs; offline-collection numbers (App. D)
	$(PY) scripts/regrade_journals.py 'data/xfer8b/*' 'data/xfer30b/*'
	$(PY) scripts/xfer_compare.py --cohort $(CONF400) \
		--run "1.7B=data/regraded/data/xfer8b/r17_confirm_h*" "8B=data/regraded/data/xfer8b/r8b_confirm_*" \
		"30B=data/regraded/data/xfer30b/r30_confirm_*" "235B=data/xfer235b/r235_confirm" --by-level
	$(PY) paper/scripts/regrade_transfer_pipeline.py

transfer-fidelity: ## Fidelity table: refit maps, FVE through the maps, MLP ceiling (6x A100)
	$(PY) scripts/xfer_refit_mix.py build
	for t in 8b 30b; do \
		$(PY) scripts/xfer_refit_mix.py extract --target $$t && \
		$(PY) scripts/xfer_refit_mix.py fit --target $$t; done
	$(PY) scripts/xfer_fidelity.py build
	$(PY) scripts/xfer_fidelity.py extract --model Qwen/Qwen3-1.7B --layer 20 --tag s
	$(PY) scripts/xfer_fidelity.py extract --model Qwen/Qwen3-8B --layer 26 --tag 8b
	$(PY) scripts/xfer_fidelity.py extract --model Qwen/Qwen3-30B-A3B --layer 34 --tag 30b
	for w in s 8b 30b; do $(PY) scripts/xfer_fidelity.py verbalize --which $$w --variant mix; done
	$(PY) scripts/xfer_fidelity.py report --variant mix
	for t in 8b 30b; do $(PY) scripts/xfer_ceiling.py --target $$t; done

transfer-reading: transfer-screen ## Reading test: blind judge on target explanations (OPENAI_API_KEY)
	for t in 8b 30b; do for s in sites extract verbalize judge report; do \
		$(PY) scripts/xfer_disagree.py $$s --target $$t --variant mix || exit 1; done; done

transfer-interface: ## Direction built on the 8B itself (cos 0.958, 49.1% on dev400)
	$(PY) scripts/xfer_interface.py verbalize --which s
	$(PY) scripts/xfer_interface.py verbalize --which b
	$(PY) scripts/xfer_interface.py analyze

# ---------------------------------------------------------------------------
# 5. Published comparators and hard problems (§4.4, §6.1, App. D)
#    Runs through scripts/gpu_queue.py on 4x A100. Needs transfer-maps first.
# ---------------------------------------------------------------------------
baselines: ## SEAL, DEER, word ban, mapped D; confirm400, gate400, hard128 at 1.7B/8B/30B
	$(PY) scripts/baselines_ban_tokens.py
	$(PY) scripts/baselines_seal_vector.py --model Qwen/Qwen3-1.7B --layer 20 \
		--journals 'data/distill/gen/shard*/rollouts.jsonl' --questions 'data/distill/gen/shard*.jsonl' \
		--out data/baselines/seal/seal_1p7b_L20.pt
	$(PY) scripts/baselines_build_hard.py
	mkdir -p $(WS)/baselines_logs
	$(PY) scripts/baselines_queue.py > $(WS)/baselines_logs/jobs.jsonl
	$(QUEUE) --jobs $(WS)/baselines_logs/jobs.jsonl --gpus 0,1,2,3 --log $(WS)/baselines_logs

baselines-amend-e: ## Strength-matched and tuned SEAL (Amendment E)
	$(PY) scripts/baselines_queue_e.py > $(WS)/baselines_logs/jobs_e.jsonl
	$(QUEUE) --jobs $(WS)/baselines_logs/jobs_e.jsonl --gpus 0,1,2,3 --log $(WS)/baselines_logs

baselines-amend-f: ## 1.7B at alpha 0.5 on hard128 (Amendment F, 1x A100, ~5 h)
	$(RUN) $(M17) --cohort $(HARD128) --outdir data/baselines/f17_base_s01 --max-new-tokens 32768 \
		--seeds 2 --seed-start 0 --policies base
	for s in 0 2 4 6; do \
		$(RUN) $(M17) --cohort $(HARD128) --outdir data/baselines/f17_s$$s$$((s + 1)) \
		--max-new-tokens 32768 --seeds $$((s + 2)) --seed-start $$s --policies N@a0.5d256; done
	$(PY) scripts/amendf_report.py

baselines-report: ## Comparator, pairwise and hard128 tables (data/baselines/report.md)
	$(PY) scripts/baselines_report.py

# ---------------------------------------------------------------------------
# 6. Selective steering (§6.2-6.3, App. F). 6x A100 queue, ~1 day each.
#    Needs the 1.7B distillation generation (distill-1.7b) for the source traces.
# ---------------------------------------------------------------------------
selective: ## Redundant-only direction on sel400
	$(PY) scripts/sel_build.py
	for k in 0 1 2; do CUDA_VISIBLE_DEVICES=$$k $(PY) scripts/sel_probe.py \
		--journals 'data/distill/gen/shard*/rollouts.jsonl' --select data/selective/source1000.jsonl \
		--questions data/distill/train.jsonl --shard $$k/3 \
		--out data/selective/probes/source_s$$k.jsonl & done; wait
	$(PY) scripts/sel_label.py --probes 'data/selective/probes/source_s*.jsonl' \
		--out data/selective/episodes_source.jsonl
	for k in 0 1 2; do CUDA_VISIBLE_DEVICES=$$k $(PY) scripts/sel_states.py \
		--episodes data/selective/episodes_source.jsonl \
		--journals 'data/distill/gen/shard*/rollouts.jsonl' --questions data/distill/train.jsonl \
		--shard $$k/3 --out data/selective/states/source_s$$k.pt & done; wait
	$(PY) scripts/sel_gate.py
	$(PY) scripts/sel_directions.py
	$(PY) scripts/sel_queue_eval.py --jobs $(WS)/distill_logs/jobs.jsonl
	$(QUEUE) --jobs $(WS)/distill_logs/jobs.jsonl --log $(WS)/distill_logs

gated: ## Probe-gated steering on gate400
	$(PY) scripts/gate_build.py
	CUDA_VISIBLE_DEVICES=0 $(PY) scripts/gate_states.py
	$(PY) scripts/gate_probe.py
	$(PY) scripts/gate_build.py --lock
	$(PY) scripts/gate_queue.py --jobs $(WS)/distill_logs/gate_jobs.jsonl
	$(QUEUE) --jobs $(WS)/distill_logs/gate_jobs.jsonl --log $(WS)/distill_logs

vers-p: dir-P ## P (doubt-edit direction) on dev400 x 2 seeds, then the selection rule
	mkdir -p $(WS)/vers_logs
	$(PY) scripts/vers_queue.py
	$(QUEUE) --jobs $(WS)/vers_logs/jobs.jsonl --gpus 0 --log $(WS)/vers_logs

# ---------------------------------------------------------------------------
# 7. Distillation into students (§7, App. G). 6x A100, one job per GPU.
# ---------------------------------------------------------------------------
distill-1.7b: ## 1.7B students: Short-SFT, Steered-SFT, DPO (full and LoRA)
	$(PY) scripts/distill_build_pools.py
	SHARDS=6 LOGDIR=$(WS)/distill_logs bash scripts/run_distill_gen.sh
	$(PY) scripts/distill_queue.py

distill-1.7b-dpo: ## 1.7B Natural SFT -> DPO round (v3)
	bash scripts/run_distill_v3_gen.sh
	$(PY) scripts/distill_v3_queue.py
	bash scripts/distill_v3_finish_split.sh
	$(PY) scripts/distill_v3_report.py

distill-1.7b-gated: ## 1.7B Gated-SFT (pipeline traces)
	$(PY) scripts/pipe_build.py
	$(PY) scripts/pipe_report.py

distill-8b: transfer-maps ## 8B students (v4): teacher generation, proxies, 4 trainings, evals
	$(PY) scripts/distill8b_lock.py --shards 6
	for k in 0 1 2 3 4 5; do CUDA_VISIBLE_DEVICES=$$k $(RUN) --model Qwen/Qwen3-8B --layer 26 \
		--dir NU26=$(X8)/L26_NU.pt --cohort data/distill_8b/gen/shard$$k.jsonl \
		--outdir data/distill_8b/gen/shard$$k --policies base NU26@a0.75d256 --seeds 4 \
		--max-new-tokens 32768 --batch 512 --gpu-memory-utilization 0.90 --checkpoint-every 500 & \
	done; wait
	$(PY) scripts/distill8b_queue.py --phase A --jobs $(WS)/distill_logs/d8b_jobs.jsonl
	$(QUEUE) --jobs $(WS)/distill_logs/d8b_jobs.jsonl --log $(WS)/distill_logs
	$(PY) scripts/distill8b_queue.py --phase B --jobs $(WS)/distill_logs/d8b_jobs.jsonl
	$(QUEUE) --jobs $(WS)/distill_logs/d8b_jobs.jsonl --log $(WS)/distill_logs
	$(PY) scripts/distill8b_report.py

distill-offline: transfer-regrade ## Offline trace collection with the 8B / 30B teachers (App. D)
	$(PY) scripts/audit_trace_quality.py build --out data/xfer_claim4/t3b_8b_pooled --arm NU26@a0.75d256
	$(PY) scripts/audit_trace_quality.py judge --out data/xfer_claim4/t3b_8b_pooled
	$(PY) scripts/audit_trace_quality.py report --out data/xfer_claim4/t3b_8b_pooled

# ---------------------------------------------------------------------------
# 8. Figures
# ---------------------------------------------------------------------------
figures: ## All paper figures from saved results (CPU)
	$(PY) paper/scripts/generate_figures.py
