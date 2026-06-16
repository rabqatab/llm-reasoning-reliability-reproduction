#!/bin/bash
# LCF generalization to the LEGAL domain (breadth experiment).
# Does the logic-validity LCF that helped Qwen3 on fallacies transfer to legal
# valid/invalid conclusion pairs? Runs extract->train->fallacy-id eval per domain
# on Qwen3-8B, with ISOLATED checkpoints (Qwen3-8B-legal / Qwen3-8B-kcc_legal) so
# the base fallacy checkpoint is never clobbered. LCF_DATA_DIR points the pipeline
# at lcf/<domain>/ without touching lcf/data/.
# Run in the LOGIN shell (sparkq OOMs on GB10 model-load transients). CPU eval too.
set -uo pipefail
ROOT=/home/alphabridge/Study/reliableAI_final
cd "$ROOT/lcf/lcf_impl"
export PYTHONPATH=$ROOT WANDB_MODE=offline HF_HOME=/mnt/nfs/ssd1/huggingface_cache \
       HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NVIDIA_DISABLE_REQUIRE=1
M=Qwen/Qwen3-8B
OUT=$ROOT/results/lcf_legal_results.txt
: > "$OUT"
for DOM in legal kcc_legal; do
  DDIR=$ROOT/lcf/$DOM
  TAG=Qwen3-8B-$DOM
  echo "### $DOM extract $(date +%H:%M)"
  LCF_DATA_DIR=$DDIR uv run python extract_reps.py --model "$M" --split train \
      --out "$ROOT/lcf/data/reps_${TAG}_train.pt" || { echo "FAIL_extract_$DOM"; continue; }
  LCF_DATA_DIR=$DDIR uv run python extract_reps.py --model "$M" --split val \
      --out "$ROOT/lcf/data/reps_${TAG}_val.pt" || true
  echo "### $DOM train $(date +%H:%M)"
  LCF_DATA_DIR=$DDIR uv run python train.py --model "$M" \
      --reps "$ROOT/lcf/data/reps_${TAG}_train.pt" \
      --val-reps "$ROOT/lcf/data/reps_${TAG}_val.pt" \
      --ckpt-name "$TAG" --no-wandb || { echo "FAIL_train_$DOM"; continue; }
  echo "### $DOM eval $(date +%H:%M)"
  echo "== $DOM ==" >> "$OUT"
  uv run python fallacy_eval.py --model "$M" --ckpt "$ROOT/lcf/checkpoints/$TAG" \
      --data "$DDIR/fallacy_id_test.jsonl" 2>&1 | grep -E "RESULT|original|\+LCF|==" | tee -a "$OUT"
done
echo "### LCF-LEGAL DONE $(date +%H:%M)"
