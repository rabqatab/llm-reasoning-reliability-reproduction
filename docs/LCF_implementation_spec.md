# LCF (Logic Control Framework) — Implementation-Ready Specification

Reproduction spec for *"Content-free Logical Modification of LLM by Disentangling and Modifying Logic Representation"* (Wu, Bu, Chen, Cai — AAAI 2025). Distilled from the main paper (`paper/LCF_AAAI2025.pdf`) + supplementary (`lcf/LCF_official/LCF_supplementary_materials.pdf`). The official repo has NO code, so this is a from-scratch build. Ambiguities flagged **[AMBIGUOUS]** with chosen defaults.

## 0. Concept
LCF = small trainable adapter inserted AFTER attention & MLP sub-modules of selected layers in a FROZEN base LLM. For hidden rep `R_input`: (1) project to content `R_content` + logic `R_logic`; (2) shift logic toward valid region `R_logic+ = R_logic + V`; (3) fuse via decoder → `R_+`; (4) nudge `R_input` toward `R_+` → `R_input+` replaces `R_input`. Only projectors+decoder train (~136 MB); base LLM frozen.

## A. Architecture
Base hidden size d=4096 (Llama2-7B). Best config [2048,1024].
- **Content Projector** (2-layer MLP): `Linear(4096→2048) → ReLU → Linear(2048→1024)` → `R_content ∈ ℝ^1024`
- **Logic Projector** (same shape, separate weights): → `R_logic ∈ ℝ^1024`
- **Decoder**: `Linear(1024→2048) → ReLU → Linear(2048→4096)` → ℝ^d
- **Fusion (Eq.3):** `R_+ = MLP(R_content + Attn(R_content, R_logic+))`, Attn = cross-attention Q=R_content, K=V=R_logic+. **[AMBIGUOUS]** single-head scaled dot-product, learnable W_Q/K/V/O ∈ ℝ^{1024×1024}, scale 1/√1024 (8-head variant OK).
- **Logic modification (Eq.1-2):** `R_logic+ = R_logic + V`, `V = C_pos_logic − C_neg_logic` (mean of all valid / all invalid logic reps). **[AMBIGUOUS]** EMA centroids during training (detached), frozen per-layer V at inference. Reverse modification `R_logic− = R_logic − V` to reduce validity.
- **Adjust input (Eq.4-5):** `D = R_+ − R_input`; `R_input+ = R_input + (D/||D||₂)·η`. **η = 0.5 (Conclusion Generation), 4.5 (Fallacy Identification)**.
- **Layer selection:** train on attn+MLP reps from layers **10–30** (randomly sample 2 layers per identical-token pair, all feed ONE shared LCF). Inference: apply to the **10 sub-layers with highest "distinctiveness"** = nearest-centroid separability of valid vs invalid reps on val set. Attn 10–20 & MLP 20–30 help most; 15–20 generally good.

## B. Losses  (total Eq.13: `L = L_rec + L_logic+ + L_logic− + L_content`)
- **Reconstruction (Eq.6-7):** `R̂ = Decoder(R_content, R_logic)` (UN-modified logic); `L_rec = MSE(R_input, R̂)`.
- **Logic contrastive InfoNCE (Eq.8-9):** valid set `S_logic+`, invalid set `S_logic−`, temp τ, sim=cosine. `L_logic+`: pull valid together, push from invalid; `L_logic−`: symmetric. **[AMBIGUOUS]** τ=0.1, cosine sim, SupCon-style over {valid,invalid} labels.
- **Content constraint (Eq.10-12):** for pair `(R_input+, R_input−)` w/ identical content opposite logic: `R̂_− = Decoder(R_content+, R_logic−)`, `R̂_+ = Decoder(R_content+, R_logic+)`, `L_content = MSE(R_input−,R̂_−) + MSE(R_input+,R̂_+)`. Forces content projector logic-independent.

## C. Training
- **Data:** each LFUD fallacious sample → invalid conclusion; GPT-3.5-turbo generates a valid conclusion from same premise → **540 valid + 540 invalid** (manually reviewed). Find identical tokens between (invalid,valid) conclusion pair; per token randomly sample 2 layers in [10,30] → `(R_input+, R_input−)` pairs. Per-model pair counts: Llama2 15956, Llama3 13400, Mistral 14684, Vicuna 15956, ChatGLM3 13608, Baichuan 13520.
- **Trainable:** Content+Logic Projector+Decoder (one shared LCF/model). **Frozen:** base LLM.
- **Optimizer:** AdamW, **lr 1e-3, 10 epochs**. **[AMBIGUOUS]** wd 0.01, batch ≥256, grad-clip 1.0.

## D. LFUD Dataset
- arXiv 2402.11100 (Li et al 2024c). **Repo: https://github.com/YandaGo/LFUD** (`LFUD.csv`).
- 12 fallacy types, 67 scenarios, 804 sentences → 4020 QA across 5 tasks. Fields: `proposition` (premise), `sentence`, `fallacy_type`, task1–5.
- **LCF split by scenario 45:5:17 → 540 train / 60 val / 204 test** (test scenarios disjoint).
- LCF tasks: **Conclusion Generation** (premise→conclusion, metric Valid%); **Fallacy Identification** (4-option: 1 valid / 2 invalid / "I have no comment").

## E. Baselines
| Baseline | Repo | Reuse |
|---|---|---|
| **ITI** | https://github.com/likenneth/honest_llama | probing for direction per attn head, shift top-K heads at inference; retarget "truthful"→"valid-logic" on LFUD |
| **RAHF** | https://github.com/LiuAmber/RAHF (ACL2024) | representation-control from preferred/dispreferred pairs; feed valid/invalid conclusion pairs |
| **SFT** | HF Trainer/LoRA | fine-tune base LLM on 540 valid conclusions (premise→valid). Strongest simple baseline (+20% vs LCF +38%) |

Table 5 (Llama2): Original 70.58/58.84 · ITI 69.60/62.25 · RAHF 71.56/46.56 · SFT 79.90/78.43 · **LCF 83.82/96.56** (Valid%GPT4 / Valid%Trained).

## F. Metrics (204-sample test, 6 LLMs)
- **Conclusion Generation:** Valid%(GPT-4 judge); Valid%(Trained = Llama-2 fallacy classifier fine-tuned on LFUD, must build); Perplexity (base LLM, fluency).
- **Fallacy Identification:** Accuracy; **Δ Probability** = mean(`P(correct) − mean(P(incorrect))`) from option probabilities.

## G. Sanity numbers (+LCF rows, main Table 1)
Llama2 83.82/96.56/12.12PPL/75.00Acc/6.29Δ · Llama3 82.84/93.13/17.76/76.96/5.12 · Vicuna 78.92/75.00/20.39/71.56/4.71 · Mistral 85.71/94.60/21.17/74.01/3.39 · ChatGLM3 77.94/93.62/42.47/73.03/3.02 · Baichuan 81.86/91.17/29.82/66.17/1.59.
Ablation (Llama2): w/o L_logic → Acc 51.47, ΔProb −1.83 (logic loss critical). Dims [2048,1024] best.

## Key ambiguities → defaults
ReLU activation; single-head cross-attn; τ=0.1 cosine; EMA centroids→frozen V; SupCon positives; AdamW wd0.01 batch≥256; PPL via base LLM; ΔProb = P(correct)−mean(P(incorrect)); Valid%(Trained) discriminator = self-trained Llama2 LFUD classifier.
