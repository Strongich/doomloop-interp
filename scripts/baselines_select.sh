#!/bin/bash
# Part B alpha selection (§B3): writes data/baselines/<tag>_selected.txt (policy name or NONE).
set -euo pipefail
tag=$1; base=$2; arms=$3
python scripts/select_alpha.py --base "$base" --arms "$arms" | tee data/baselines/${tag}_select.log
sel=$(grep '^SELECTED' data/baselines/${tag}_select.log | awk '{print $2}')
echo "$sel" > data/baselines/${tag}_selected.txt
