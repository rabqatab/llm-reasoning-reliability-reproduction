# reliableAI_final — Master Plan & Progress

Two-paper reproduction + extension. Decisions (2026-06-14): paper models **+ local Qwen3**; reproduce paper datasets **+ extend to local BIRD**; **both papers in parallel**.

## Resources
- GPU: DGX Spark GB10, single-node (all models 7–14B). Submit via `sparkq`. Unified-memory rules: pre-download from login shell (NFS cache RO in jobs); ≤2× 7B/node; if a load OOMs in sparkq but the node is free, run directly in the login shell with `nohup`. Env: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NVIDIA_DISABLE_REQUIRE=1 WANDB_MODE=offline`.
- API keys: `./.env` (copied from ~/Research/LexLink-ko-mcp/.env) → OPENAI (GPT-4 judge + GPT-3.5 valid-conclusion gen), ANTHROPIC, GOOGLE.
- HF_HOME=`/mnt/nfs/ssd1/huggingface_cache`. Pre-download models from LOGIN shell (NFS RO in jobs).
- Models: Qwen3-8B/14B/32B + Llama-2-7b-chat-hf + Vicuna/Mistral/ChatGLM3/Baichuan2 (downloaded). Llama-3.1 gated (skip).

## Paper A — RPC (test-time scaling)  [rpc/]
- [x] Clone, uv env, reproduction verified. **Full 33-cell grid = paper Table 2** → `docs/RPC_reproduction_results.md`.
- [x] **4 local-dataset extensions** (Qwen3-8B, K=8, reuse RPC evaluators w/ domain equality):
  - [x] BIRD text-to-SQL (exec-match): SC 28.5 > RPC 25.0 (tie/lose at K=8).
  - [x] JurisNet legal extraction (set exact-match): **RPC 20.0/46.0 > SC 19.1/52.4 (wins Acc+ECE)**.
  - [x] KCC precedent-relevance (balanced): SC≈RPC≈PPL ~65 bal-acc (binary → no RPC edge; PPL best ECE 21.6).
  - [x] LFUD fallacy-id MCQ: all ~88% (model already strong → RPC≈SC; PPL best ECE 8.3). Connects both papers.
  - PPL over-confidence (ECE 73–93) reproduces in ALL domains.
- [x] **Batched generation** `rpc/_batched_gen.py` (num_return_sequences=K) — ~Kx faster than sequential.

## Paper B — LCF (logic representation, from scratch)  [lcf/]
- [x] Spec; core (model/losses/data/train/infer); LFUD data (540/60/204 + GPT-3.5 valid conclusions); eval (GPT-4o judge, distilbert discriminator, postprocess_judge).
- [x] **Reproduced (mixed/negative)**: Qwen3-8B ΔProb 3.96→7.83 (matches paper direction); **Llama-2-7b-chat DEGRADES** (4.85→2.44, opposite to paper). → `docs/LCF_reproduction_results.md`.
- [x] **CRITICAL ANALYSIS** → `docs/LCF_critical_analysis.md` (main contribution): logic-validity direction is **real but weak** (0.82 best single sub-layer vs **0.95 for suicide-risk** on MoodRisk; 0.52=chance pooled); **separability ≠ controllability** (v2 redesign gives no gain); flagship 96.56% relies on **unauditable discriminator**. Verdict: not reproducible / not model-agnostic — NOT fabrication.
- [x] **Model-agnostic v2** `lcf/lcf_impl/{probe_layers,lcf_v2,lcf_v2_eval}.py` — best-layer supervised direction + norm-relative intervention; evaluated (no consistent gain).
- [x] **MoodRisk control probe** `lcf/lcf_impl/moodrisk_probe.py` — risk-direction 0.95 (representation-editing premise holds for semantic attributes).
- [x] **Baselines**: SFT (collator fixed), ITI (ran ≈ original), RAHF — built; `lcf/run_baselines.sh`.
- [x] **Generalization data**: legal-LCF from JurisNet (`lcf/legal/`) + KCC (`lcf/kcc_legal/`) — valid/invalid legal conclusion pairs (GPT-built).

## Next (optional breadth)
1. (done) 5-domain RPC table complete: RPC wins when model uncertain+diverse (math, legal extraction), ties when confident/binary (MCQ, KCC).
2. Larger-K BIRD re-test; run LCF pipeline on legal/kcc_legal (expected weak per critical analysis).
3. Multi-model LCF (Vicuna/Mistral/ChatGLM3/Baichuan2 downloaded).

## Drivers
- `rpc/RPC/run_full_repro.sh` — RPC grid (CPU). `rpc/{bird,jurisnet,kcc,lfud_mcq}_ext` — generate_* (GPU) + run_* (CPU eval).
- `lcf/run_lcf_full.sh <model>` — extract→train→eval (ALONE on node). `lcf/run_baselines.sh <model>`.
- `lcf/lcf_impl/{probe_layers,lcf_v2,lcf_v2_eval,moodrisk_probe}.py` — critical-analysis experiments (reps-only / GPU).
- `lcf/eval/postprocess_judge.py` — GPT-4 judge + ValidTrained on saved generations (login shell).
- Node 2: `docker run -d nvcr.io/nvidia/pytorch:25.09-py3` + NFS cache mount (see sparkq debug doc).
