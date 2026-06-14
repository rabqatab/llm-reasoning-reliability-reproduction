#!/bin/bash
# Sequential LCF GPU pipeline for one base model: extract train reps -> extract val reps -> train LCF.
# Run via sparkq (HF offline, NVIDIA_DISABLE_REQUIRE set by submitter). Usage: run_lcf_pipeline.sh <hf_model_id>
set -euo pipefail
MODEL="${1:-Qwen/Qwen3-8B}"
SAFE=$(echo "$MODEL" | sed 's#.*/##')
IMPL=/home/alphabridge/Study/reliableAI_final/lcf/lcf_impl
DATA=/home/alphabridge/Study/reliableAI_final/lcf/data
TRAIN_REPS="$DATA/reps_${SAFE}_train.pt"
VAL_REPS="$DATA/reps_${SAFE}_val.pt"
export WANDB_MODE=offline
cd "$IMPL"

echo "### [1/3] extract train reps for $MODEL"
uv run python extract_reps.py --model "$MODEL" --split train --out "$TRAIN_REPS"
echo "### [2/3] extract val reps for $MODEL"
uv run python extract_reps.py --model "$MODEL" --split val --out "$VAL_REPS"
echo "### [3/3] train LCF for $MODEL"
uv run python train.py --model "$MODEL" --reps "$TRAIN_REPS" --val-reps "$VAL_REPS"
echo "### LCF PIPELINE DONE for $MODEL"
