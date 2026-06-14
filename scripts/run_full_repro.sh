#!/bin/bash
# Full faithful reproduction grid for Paper A (RPC).
# 4 datasets x 3 models x 3 methods. CPU-only (operates on published reasoning paths).
cd "$(dirname "$0")"
export HF_HUB_DISABLE_PROGRESS_BARS=1
: > results_full.txt

DATASETS=(MATH MathOdyssey AIME OlympiadBench)
MODELS=(Deepseek-Math-RL-7B InternLM2-Math-Plus-1.8B InternLM2-Math-Plus-7B)
METHODS=(SC PPL RPC)

for ds in "${DATASETS[@]}"; do
  if [ "$ds" == "MATH" ]; then K=64; else K=128; fi
  for model in "${MODELS[@]}"; do
    for method in "${METHODS[@]}"; do
      echo ">>> $method $ds $model K=$K"
      uv run python main.py --dataset "$ds" --model "$model" --method "$method" --K "$K" \
        2>&1 | grep -E "^($method |Failed)" | tail -1 | tee -a results_full.txt
    done
  done
done
echo "ALL DONE" | tee -a results_full.txt
