#!/bin/bash
# Train + eval LCF baselines (SFT / ITI / RAHF) sequentially on one GPU node.
# Continue-on-failure so one bad baseline doesn't kill the rest. skip-gpt4:
# GPT-4 judging + ValidTrained are filled later by postprocess_judge.py.
set -uo pipefail
MODEL="${1:-Qwen/Qwen3-8B}"
ROOT=/home/alphabridge/Study/reliableAI_final
export PYTHONPATH=$ROOT:$ROOT/lcf
export WANDB_MODE=offline
cd "$ROOT/lcf/eval"
DATA="$ROOT/lcf/data"
DISC="$ROOT/lcf/eval/discriminator"
COMMON="--model $MODEL --data-dir $DATA --discriminator-dir $DISC --skip-gpt4 --delta-scale 100"

echo "### [SFT] train"
uv run python ../baselines/sft.py  --model "$MODEL" --epochs 10        || echo "FAIL sft-train"
echo "### [SFT] eval"
uv run python run_eval.py $COMMON --variant +SFT                       || echo "FAIL sft-eval"

echo "### [ITI] train"
uv run python ../baselines/iti.py  --model "$MODEL" --k 48 --alpha 15  || echo "FAIL iti-train"
echo "### [ITI] eval"
uv run python run_eval.py $COMMON --variant +ITI                       || echo "FAIL iti-eval"

echo "### [RAHF] train"
uv run python ../baselines/rahf.py --model "$MODEL" --epochs 5 --alpha 5 || echo "FAIL rahf-train"
echo "### [RAHF] eval"
uv run python run_eval.py $COMMON --variant +RAHF                      || echo "FAIL rahf-eval"

echo "### BASELINES DONE"
