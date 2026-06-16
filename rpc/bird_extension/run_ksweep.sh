#!/bin/bash
# BIRD K-scaling sweep: SC/PPL/RPC at K=8,16,32 on the K=32 generated paths.
# timeout=5s repeats=5 — caps pathological SQLite queries (the O(K^2) pairwise
# exec-match with the default 30s timeout made K>=16 take hours).
set -uo pipefail
ROOT=/home/alphabridge/Study/reliableAI_final
RPCPY=$ROOT/rpc/RPC/.venv/bin/python
cd "$ROOT/rpc/bird_extension"
for K in 8 16 32; do
  echo "===== BIRD K=$K (timeout=5 repeats=5) $(date +%H:%M:%S) ====="
  "$RPCPY" run_bird.py --json bird_K32.json --meta bird_K32.meta.json \
    --in_dir . --K "$K" --repeats 5 --timeout 5 --methods SC,PPL,RPC \
    --out "$ROOT/results/rpc_bird_K${K}.txt" 2>&1
done
echo "BIRD_KSWEEP_DONE $(date +%H:%M:%S)"
