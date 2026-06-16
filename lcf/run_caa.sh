#!/bin/bash
cd /home/alphabridge/Study/reliableAI_final/lcf/lcf_impl
export PYTHONPATH=/home/alphabridge/Study/reliableAI_final WANDB_MODE=offline \
  HF_HOME=/mnt/nfs/ssd1/huggingface_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NVIDIA_DISABLE_REQUIRE=1
echo "### CAA Qwen3-8B $(date +%H:%M)"
uv run python lcf_caa.py --model Qwen/Qwen3-8B --layer 12 --alphas 0,4,8 --n-dir 100
echo "### CAA Llama2 $(date +%H:%M)"
uv run python lcf_caa.py --model meta-llama/Llama-2-7b-chat-hf --layer 11 --alphas 0,4,8 --n-dir 100
echo "### CAA DONE $(date +%H:%M)"
