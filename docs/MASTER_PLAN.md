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
  - [x] KCC precedent-relevance (**4-class graded 0-3**, 320 balanced, chance 25): SC≈RPC≈PPL ~43% acc; RPC no edge even with answer diversity; PPL best ECE (45 vs 51). [CORRECTED from an earlier invalid binary framing that dropped grades 2,3 and inverted grade 1.]
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

## Breadth experiments (DONE)
1. [x] 5-domain RPC table: RPC wins under uncertainty+diversity+informative-confidence (math, legal extraction), ties when the model is confident (MCQ 88%) or confidence is uninformative (KCC 4-class graded).
2. [x] **Larger-K BIRD re-test → confirms Remark 6.** Regenerated BIRD K=32 (n=80), re-aggregated SC/PPL/RPC at K=8/16/32. **RPC's edge over SC GROWS with K**: ~tie at K=8 → RPC wins at K=16 → **+2.5 acc & ½ ECE at K=32**. → `docs/RPC_reproduction_results.md` (BIRD K-scaling). NB: fixed a thread-leak in `rpc/bird_extension/sql_exec.py` (O(K²) exec-match signal-killed the process at K≥16) via `conn.interrupt()`.
3. [x] **Multi-model LCF** — the mixed pattern holds: **Mistral-7B DEGRADES** (Acc 35.3→27.5, ΔProb 14.20→3.23) like Llama2; **Vicuna-7b** (independent cross-check by coauthor vanguard-gpt, separate codebase `lcf/independent_vicuna/`) **helps on no metric** (Acc 69.6→63.7, ΔProb 34.4→30.0; separability 0.66≈chance) → `docs/LCF_vicuna_independent.md`. `lcf/run_multimodel.sh`, `fallacy_eval.py`. (ChatGLM3/Baichuan2 still pending — low value, pattern shown on 4 models.)
4. [x] **LCF on legal domain** (`lcf/legal/` JurisNet, `lcf/kcc_legal/` KCC; Qwen3-8B, Node 2 docker). Inconsistent: legal Acc 72.5→62.5 (degrades), kcc_legal 60.0→67.5 (improves) — same model, opposite directions → reinforces weak/unreliable LCF signal. `lcf/run_legal.sh`, `results/lcf_legal_results.txt`. (n=40/domain.)

## Model-agnostic LCF investigation (DONE — prior-work-grounded) → `docs/LCF_model_agnostic.md`
Goal: fix LCF's model-dependence using the activation-steering literature (CAA, RepE, CAST/K-CAST, Valentino AAAI'26).
- [x] **v3 — conditional CAA** (`lcf_caa.py`): mean-diff dir + midpoint gate. Degrades both Qwen3 & Llama2; gate inert (cond≈static).
- [x] **v4 — faithful K-CAST** (`lcf_kcast.py`): kNN gate + LayerNavigator (Qwen3 L15 sep .885, Llama2 L22 sep .835) + signed α. Still degrades both; **kNN gate fires ~98%** (reference/task distribution mismatch) → cond≈static. Negative *explained*.
- [x] **v5 — formal-syllogism task** (`gen_syllogisms.py` 2×2 validity×believability; `lcf_syllogism_steer.py`; reference==task): **content-direction ablation debiases Qwen3 to 100% (gap 5→0) — first positive steering result**; but validity-add is harmful, Llama2 unrescuable (at chance, no capability), kNN gate still ≈ static.
- **Final verdict**: contrastive steering debiases content effects ONLY when distribution-matched + content(not validity) direction + a capable model. Not model-agnostic in general; K-CAST's conditional gate does not reproduce.
- GPU: Qwen3 on Node 1 (sparkq), Llama2 on Node 2 (docker, K-CAST/syllogism run in parallel). NB: a dropped NFS mount on Node 1 (`/mnt/nfs/ssd1`) was the real cause of an early weight-load stall — remount via `sudo mount -a`.

## Drivers
- `rpc/RPC/run_full_repro.sh` — RPC grid (CPU). `rpc/{bird,jurisnet,kcc,lfud_mcq}_ext` — generate_* (GPU) + run_* (CPU eval).
- `lcf/run_lcf_full.sh <model>` — extract→train→eval (ALONE on node). `lcf/run_baselines.sh <model>`.
- `lcf/lcf_impl/{probe_layers,lcf_v2,lcf_v2_eval,moodrisk_probe}.py` — critical-analysis experiments (reps-only / GPU).
- `lcf/eval/postprocess_judge.py` — GPT-4 judge + ValidTrained on saved generations (login shell).
- Node 2: `docker run -d nvcr.io/nvidia/pytorch:25.09-py3` + NFS cache mount (see sparkq debug doc).
