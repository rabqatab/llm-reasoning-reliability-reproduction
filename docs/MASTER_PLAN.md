# reliableAI_final — Master Plan & Progress

Two-paper reproduction + extension. Decisions (2026-06-14): paper models **+ local Qwen3**; reproduce paper datasets **+ extend to local BIRD**; **both papers in parallel**.

## Resources
- GPU: DGX Spark GB10, single-node (all models 7–14B). Submit via `sparkq` — **read `docs/sparkq_issues.md`** (unified-memory OOM rules; if a 7B load OOMs in sparkq but node is free, run directly in login shell with `nohup`). Env: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NVIDIA_DISABLE_REQUIRE=1 WANDB_MODE=offline`.
- API keys: `./.env` (copied from ~/Research/LexLink-ko-mcp/.env) → OPENAI (GPT-4 judge + GPT-3.5 valid-conclusion gen), ANTHROPIC, GOOGLE.
- HF_HOME=`/mnt/nfs/ssd1/huggingface_cache`. Pre-download models from LOGIN shell (NFS RO in jobs).
- Models: Qwen3-8B/14B/32B + Llama-2-7b-chat-hf + Vicuna/Mistral/ChatGLM3/Baichuan2 (downloaded). Llama-3.1 gated (skip).

## Paper A — RPC (test-time scaling)  [rpc/]
- [x] Clone, uv env, reproduction verified.
- [x] **Full 33-cell grid matches paper Table 2** → `results_full.txt`, `docs/RPC_reproduction_results.md`.
- [x] **BIRD extension** [rpc/bird_extension/]: Qwen3-8B K=8 SQL paths, exec-match equality. SC 28.5 > RPC 25.0 = PPL 25.0 (honest negative; RPC needs larger K). PPL overconfidence (ECE 73.6) reproduces.
- [ ] (optional) Larger-K BIRD; Verb baseline.

## Paper B — LCF (logic representation, from scratch)  [lcf/]
- [x] Spec `docs/LCF_implementation_spec.md`; LFUD + ITI + RAHF cloned.
- [x] **Core** [lcf/lcf_impl/]: model/losses/data/train/infer. Smoke-tested. Layer-distinctiveness selection matches paper (attn 10-19).
- [x] **Data**: LFUD split 540/60/204; 804 GPT-3.5 valid conclusions; fallacy_id items.
- [x] **Eval** [lcf/eval/]: GPT-4o judge, distilbert discriminator (deberta-v3 broken on transformers 5.x), metrics, postprocess_judge.py (cleans generations to first line, fills GPT-4 + ValidTrained).
- [x] **Qwen3-8B reproduced**: ΔProb 3.96→7.83 (matches paper); generation muted under greedy. → `docs/LCF_reproduction_results.md`.
- [~] **Llama-2-7b-chat (paper headline model)**: pipeline + LCF trained ✓, eval original/+LCF running (login-shell direct, `lcf/llama2_run.log`).
- [~] **Baselines**: SFT collator bug fixed (DataCollatorForSeq2Seq); ITI ran (Acc/ΔProb ≈ original); SFT/RAHF re-run pending. Driver `lcf/run_baselines.sh`.
- [ ] Multi-model table (Vicuna/Mistral/ChatGLM3/Baichuan2).

## Next
1. Finish Llama-2 eval → compare to paper Table 1 (70.58/58.84 → 83.82/96.56) → postprocess GPT-4.
2. Re-run Qwen3 + Llama2 baselines (SFT fixed) for full 5-variant tables.
3. Multi-model LCF; optional larger-K BIRD.

## Drivers
- `rpc/RPC/run_full_repro.sh` — RPC grid (CPU).
- `lcf/run_lcf_pipeline.sh <model>` — extract reps + train LCF.
- `lcf/run_lcf_full.sh <model>` — pipeline + eval original + +LCF (run ALONE on node).
- `lcf/run_baselines.sh <model>` — SFT/ITI/RAHF train+eval.
- `lcf/eval/postprocess_judge.py` — GPT-4 judge + ValidTrained on saved generations (login shell).
