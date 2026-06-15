#!/bin/bash
# Multi-model LCF sweep: per model, extract reps -> train LCF -> FAST fallacy-id
# eval (option-scoring DeltaProb/Acc, original vs +LCF; no slow generation).
# Tests whether the mixed Qwen3(+)/Llama2(-) pattern holds across more models.
# Run in the LOGIN shell (sparkq OOMs on GB10). Usage: run_multimodel.sh <model> [<model> ...]
set -uo pipefail
ROOT=/home/alphabridge/Study/reliableAI_final
cd "$ROOT/lcf/lcf_impl"
export PYTHONPATH=$ROOT WANDB_MODE=offline HF_HOME=/mnt/nfs/ssd1/huggingface_cache \
       HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NVIDIA_DISABLE_REQUIRE=1
OUT=$ROOT/lcf/multimodel_results.txt
for M in "$@"; do
  SAFE=$(echo "$M" | sed 's#.*/##')
  echo "### $M extract train $(date +%H:%M)"
  uv run python extract_reps.py --model "$M" --split train --out "$ROOT/lcf/data/reps_${SAFE}_train.pt" || { echo "FAIL_extract_$SAFE"; continue; }
  uv run python extract_reps.py --model "$M" --split val   --out "$ROOT/lcf/data/reps_${SAFE}_val.pt"   || true
  echo "### $M train LCF $(date +%H:%M)"
  uv run python train.py --model "$M" --reps "$ROOT/lcf/data/reps_${SAFE}_train.pt" --val-reps "$ROOT/lcf/data/reps_${SAFE}_val.pt" || { echo "FAIL_train_$SAFE"; continue; }
  echo "### $M fallacy-id eval $(date +%H:%M)"
  uv run python fallacy_eval.py --model "$M" --ckpt "$ROOT/lcf/checkpoints/$SAFE" 2>&1 | grep -E "RESULT|original|LCF" | tee -a "$OUT"
done
echo "### MULTIMODEL DONE $(date +%H:%M)"
