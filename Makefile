# Makefile for reasoning-attention
# Usage:
#   make format   # auto-fix import order, dead code, and style
#   make lint     # type-check + verify formatting/style (no writes)
#
# All tools run through `uv run` so they use the project's .venv.

LINT_DIRS := src

# --- Kubernetes dev pod (namespace vitenko-thesis, PVC doomloops-interp) ---
NAMESPACE  := vitenko-thesis
POD        := reasoning-interp
# This project's own kubeconfig (gitignored — it carries a bearer token).
KUBECONFIG_PATH := $(CURDIR)/kubeconfig.yaml
export KUBECONFIG = $(KUBECONFIG_PATH)

.PHONY: lint format start stop connect forward pod-logs rl-setup rl-check

lint:
	uv run mypy $(LINT_DIRS)
	uv run ruff check $(LINT_DIRS)
	uv run ruff format --check --diff $(LINT_DIRS)

format:
	uv run ruff check --fix $(LINT_DIRS)
	uv run ruff format $(LINT_DIRS)

# ---------------------------------------------------------------------------
# Pod lifecycle. Same flow as the MASTERS Makefile.
# ---------------------------------------------------------------------------
start:
	-kubectl delete pod $(POD) -n $(NAMESPACE) --ignore-not-found --wait
	kubectl apply -f pod.yaml
	kubectl wait --for=condition=Ready pod/$(POD) -n $(NAMESPACE) --timeout=600s
	@echo "Waiting for zsh + oh-my-zsh to finish installing..."
	@until kubectl exec $(POD) -n $(NAMESPACE) -- test -d /root/.oh-my-zsh > /dev/null 2>&1; do sleep 2; done
	kubectl port-forward pod/$(POD) -n $(NAMESPACE) 2222:22 &
	kubectl exec -it $(POD) -n $(NAMESPACE) -- zsh

stop:
	kubectl delete pod $(POD) -n $(NAMESPACE)

connect:
	kubectl exec -it $(POD) -n $(NAMESPACE) -- zsh

forward:
	kubectl port-forward pod/$(POD) -n $(NAMESPACE) 2222:22 7860:7860

pod-logs:
	kubectl logs -f pod/$(POD) -n $(NAMESPACE)

# Push the OpenAI key in as a secret rather than baking it into the image or
# committing it. The explain stage reads OPENAI_API_KEY from the environment
# when no .env file is present.
pod-secret:
	kubectl create secret generic openai --from-env-file=.env -n $(NAMESPACE) \
		--dry-run=client -o yaml | kubectl apply -f -
	@echo "Add to pod.yaml under the container to consume it:"
	@echo "      envFrom:"
	@echo "        - secretRef:"
	@echo "            name: openai"

# ---------------------------------------------------------------------------
# Stage-2 RL environment. Separate venv on purpose — sglang downgrades torch to a
# cu12 build, which has no sm_120 kernels and would break this box's GPU and vLLM.
# ---------------------------------------------------------------------------
rl-setup:
	bash scripts/setup_rl_stack.sh

rl-check:
	bash scripts/setup_rl_stack.sh --check

# Regenerate requirements/rl.txt from requirements/rl.in.
rl-pins:
	uv pip compile requirements/rl.in --python-version 3.11 \
		--extra-index-url https://download.pytorch.org/whl/cu124 \
		--index-strategy unsafe-best-match -o requirements/rl.txt
