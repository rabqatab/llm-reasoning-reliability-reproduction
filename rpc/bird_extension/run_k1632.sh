#!/bin/bash
set -uo pipefail
ROOT=/home/alphabridge/Study/reliableAI_final
RPCPY=$ROOT/rpc/RPC/.venv/bin/python
cd "$ROOT/rpc/bird_extension"
rm -f "$ROOT/results/rpc_bird_K16.txt" "$ROOT/results/rpc_bird_K32.txt"
echo "===== BIRD K=16 (timeout=5 repeats=5) $(date +%H:%M:%S) ====="
"$RPCPY" run_bird.py --json bird_K32.json --meta bird_K32.meta.json --in_dir . \
  --K 16 --repeats 5 --timeout 5 --methods SC,PPL,RPC --out "$ROOT/results/rpc_bird_K16.txt" 2>&1
echo "===== BIRD K=32 (timeout=5 repeats=1; K=32=all paths => no sampling variance) $(date +%H:%M:%S) ====="
"$RPCPY" run_bird.py --json bird_K32.json --meta bird_K32.meta.json --in_dir . \
  --K 32 --repeats 1 --timeout 5 --methods SC,PPL,RPC --out "$ROOT/results/rpc_bird_K32.txt" 2>&1
echo "BIRD_K1632_DONE $(date +%H:%M:%S)"
