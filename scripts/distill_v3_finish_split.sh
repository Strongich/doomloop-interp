#!/usr/bin/env bash
# For each v3 DPO model: once the main eval's partial export holds all 4,000 mathfresh
# rollouts, stop that eval (its math500/gsm8k work is being done in eval_split/), then merge
# mathfresh rows + the split runs into data/distill_v3/eval/<m>/rollouts.csv.
cd "$(dirname "$0")/.."
descend() { local p=$1; echo $p; for c in $(ps -o pid= --ppid $p); do descend $c; done; }
for m in dpo_natural dpo_nla; do
  until [ -f data/distill_v3/eval/$m/rollouts.csv ] || \
        [ "$(python3 -c "import csv;print(sum(r['set']=='mathfresh' for r in csv.DictReader(open('data/distill_v3/eval/$m/rollouts.csv.partial'))))" 2>/dev/null)" = "4000" ]; do sleep 120; done
  if [ ! -f data/distill_v3/eval/$m/rollouts.csv ]; then
    pid=$(ps -eo pid,args | awk -v m="--name $m " 'index($0, "distill_eval.py") && index($0, m) && index($0, "distill_v3/eval ") && $2 ~ /python/ {print $1; exit}')
    echo "$(date +%H:%M) $m mathfresh complete; stopping main eval pid $pid"
    [ -n "$pid" ] && kill -9 $(descend $pid)
  fi
done
for m in dpo_natural dpo_nla; do
  for s in math500 gsm8k_test; do
    until [ -f data/distill_v3/eval_split/${m}_$s/rollouts.csv ]; do sleep 120; done
  done
  [ -f data/distill_v3/eval/$m/rollouts.csv ] && continue
  python3 - "$m" <<'PY'
import csv, sys
m = sys.argv[1]
src = f"data/distill_v3/eval/{m}/rollouts.csv.partial"
rows = [r for r in csv.DictReader(open(src)) if r["set"] == "mathfresh"]
assert len(rows) == 4000, len(rows)
for s in ("math500", "gsm8k_test"):
    rows += list(csv.DictReader(open(f"data/distill_v3/eval_split/{m}_{s}/rollouts.csv")))
with open(f"data/distill_v3/eval/{m}/rollouts.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(m, "merged", len(rows))
PY
done
echo "$(date +%H:%M) all merged"
