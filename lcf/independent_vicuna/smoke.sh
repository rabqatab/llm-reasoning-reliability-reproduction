#!/usr/bin/env bash
# Tiny end-to-end smoke test on the H100 (a few minutes) to catch wiring bugs
# BEFORE launching the full run. Uses --limit everywhere.
set -euo pipefail
cd "$(dirname "$0")"
export LCF_SCRATCH="${LCF_SCRATCH:-$PWD/_scratch_smoke}"
CK="$LCF_SCRATCH/checkpoints"

python src/data_prep.py
python src/gen_valid.py --backend "${GEN_BACKEND:-local}" --limit 12
python src/extract.py --limit 12
python src/train.py --tag full --epochs 2
python src/eval_identification.py --ckpt "$CK/lcf_full.pt" --tag full --limit 12
python src/eval_generation.py --ckpt "$CK/lcf_full.pt" --judge none --tag full --limit 8
python src/aggregate.py
echo "SMOKE OK"
