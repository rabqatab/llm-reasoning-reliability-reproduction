# reliableAI_final — Two-Paper Reproduction & Extension

**Authors:** Jimin Kwon ([vanguard-gpt](https://github.com/vanguard-gpt)), Minhan Cho

Reproduce two papers from scratch, compare against their baselines on the papers' datasets **and** a locally-available dataset (BIRD). Decisions made with the user: use paper models **+ local Qwen3**; reproduce paper datasets **+ extend RPC to local BIRD**; run **both papers in parallel**.

## The two papers (`paper/`)
| | Paper A — **RPC** | Paper B — **LCF** |
|---|---|---|
| File | `RPC_NeurIPS2025.pdf` | `LCF_AAAI2025.pdf` |
| Title | Bridging Internal Probability and Self-Consistency for LLM Reasoning | Content-free Logical Modification by Disentangling & Modifying Logic Representation |
| Idea | Test-time scaling: fuse perplexity + self-consistency (PC) and prune low-prob paths via Weibull mixture (RP) | Split hidden states into content/logic; push logic toward "valid" region via contrastive learning |
| Training? | **No** (inference-time aggregation) | **Yes** (projectors+decoder; base LLM frozen) |
| Official code | github.com/WNJXYK/RPC (complete) | github.com/wulidongdong/LCF (**empty** → built from scratch) |

## Status
- **RPC: reproduced exactly** (33-cell grid = paper Table 2) and **extended to 4 new domains** — BIRD SQL, JurisNet legal extraction, KCC precedent relevance, LFUD MCQ — to map *where* RPC helps. **BIRD K-scaling (K=8/16/32) confirms Remark 6**: RPC's edge over SC *grows with the sample budget* — RPC ties SC at K=8 but wins by **+2.5 acc & ½ the ECE at K=32**. → `docs/RPC_reproduction_results.md`
- **LCF: implemented from scratch + critically analysed.** Reproduction is mixed/negative, and we trace *why*: the logic-validity direction is **real but weak (0.82 best single layer vs 0.95 for a semantic attribute), diluted to chance when layers are pooled**, and **not causally controllable** by representation shifting; the paper's flagship 96.56% relies on an **unauditable, confoundable discriminator**. Verdict: not reproducible / not model-agnostic as published — *not* fabrication. → **`docs/LCF_critical_analysis.md`** (+ `docs/LCF_reproduction_results.md`)
- **Model-agnostic investigation (v2→v5), prior-work-grounded** → **`docs/LCF_model_agnostic.md`**: v2 (best-layer supervised dir) and v3 (CAA ± midpoint gate) and v4 (faithful **K-CAST** kNN gate + LayerNavigator + signed sweep) all **fail to steer logic-validity on the fallacy task** — the kNN gate fires ~98% of tokens (reference/task distribution mismatch). v5 builds a **formal-syllogism 2×2 (validity×believability)** task where reference==task: there, **content-direction ablation debiases Qwen3 to 100% (content-effect gap 5→0) — the first positive steering result** — but it is **still not model-agnostic** (no effect on Llama2, which lacks the capability) and the **conditional kNN gate never beats static**. Net: contrastive steering debiases content *only* when distribution-matched + content-direction + a capable model.
- **LCF baselines** (SFT/ITI/RAHF); multi-model LCF — Mistral degrades like Llama2, and **Vicuna-7b** (a second, independent re-implementation in a separate codebase `lcf/independent_vicuna/`) **helps on no metric** → `docs/LCF_vicuna_independent.md`; **legal-domain LCF** run (JurisNet degrades / KCC-legal improves on the same Qwen3 — inconsistent, reinforces the weak signal); MoodRisk risk-direction control probe 0.95.

## Layout
```
paper/                     the two PDFs (RPC_NeurIPS2025, LCF_AAAI2025)
rpc/RPC/                   cloned official RPC repo (uv venv, CPU-only); + run_full_repro.sh
rpc/_batched_gen.py        batched K-sample generation helper (shared by extensions)
rpc/bird_extension/        RPC on BIRD text-to-SQL (SQL exec-match)
rpc/jurisnet_ext/          RPC on JurisNet legal statute extraction (exact-match)
rpc/kcc_ext/               RPC on KCC precedent-relevance (4-class graded relevance 0-3)
rpc/lfud_mcq/              RPC on LFUD fallacy-identification MCQ (connects both papers)
lcf/lcf_impl/              from-scratch LCF: model/losses/data/train/infer
   probe_layers.py, lcf_v2.py, lcf_v2_eval.py   model-agnostic v2 + per-layer probe
   lcf_caa.py (v3), lcf_kcast.py (v4 K-CAST)    model-agnostic v3/v4 steering
   gen_syllogisms.py, lcf_syllogism_steer.py    v5 formal-syllogism content-effect test
   fallacy_eval.py         fast multi-model fallacy-id ΔProb eval
   moodrisk_probe.py       control: risk-direction probe on MoodRisk Mistral reps
lcf/eval/                  metrics (GPT-4 judge, discriminator), run_eval, postprocess_judge
lcf/baselines/             SFT / ITI / RAHF
lcf/legal/, lcf/kcc_legal/ legal-domain LCF data (valid/invalid conclusion pairs)
lcf/run_lcf_full.sh        one-shot: extract->train->eval (run ALONE on a node)
docs/   LCF_critical_analysis.md ← main finding · LCF_model_agnostic.md ← v2-v5 journey · RPC/LCF_reproduction_results.md · LCF_implementation_spec.md · MASTER_PLAN.md
results/                   raw result/probe/sweep logs
```

## Reproduce
**RPC (CPU only, no GPU):**
```bash
cd rpc/RPC && bash run_full_repro.sh        # -> results_full.txt
```
**LCF (GPU; note GB10 unified-memory limits — ≤2× 7B jobs/node, HF offline):**
```bash
# data (login shell, OpenAI key in ../.env for GPT-3.5 valid conclusions)
cd lcf/lcf_impl && uv run python lfud_data.py --model gpt-3.5-turbo
# full per-model run (extract reps -> train LCF -> eval original + +LCF)
bash lcf/run_lcf_full.sh meta-llama/Llama-2-7b-chat-hf
# fill GPT-4 + ValidTrained on the saved generations (login shell, has internet)
cd lcf/eval && uv run python postprocess_judge.py --judge-model gpt-4o
```

## Headline results (original-paper layout)

### Paper A — RPC · paper Table 2 layout (Accuracy↑ / ECE↓), InternLM2-Math-Plus-7B
Our reproduction on the authors' published reasoning paths (mean over 10 seeds):

| Method | MATH | MathOdyssey | OlympiadBench | AIME | **Avg** |
|:--|:--:|:--:|:--:|:--:|:--:|
| PPL | 46.99 / 48.99 | 27.35 / 67.70 | 7.27 / 86.90 | 5.96 / 88.98 | 21.89 / 73.14 |
| SC  | 50.57 / 6.71 | 28.25 / 12.23 | 11.07 / 20.20 | 9.40 / 14.35 | 24.82 / 13.37 |
| **RPC** | **51.94 / 6.41** | **31.77 / 9.69** | **11.12 / 18.89** | **9.75 / 14.30** | **26.15 / 12.32** |

Paper Table 2 averages: PPL 21.90/73.14 · SC 24.82/13.37 · RPC 26.11/12.37 → **our numbers match**. RPC wins on **both** accuracy and calibration; PPL is fairly accurate but badly over-confident (ECE up to ~89).

**Extensions — RPC on local datasets (Qwen3-8B, K=8).** Does RPC beat Self-Consistency outside math? Domain-dependent:

| Domain (Qwen3-8B, K=8) | SC Acc/ECE | PPL Acc/ECE | RPC Acc/ECE | RPC vs SC |
|---|---|---|---|---|
| math (paper) | — | over-confident | best | **RPC wins** |
| BIRD text-to-SQL (exec-match) | **28.5**/39.8 | 25.0/73.6 | 25.0/39.2 | lose |
| JurisNet legal extraction (exact-match) | 19.1/52.4 | 18.7/**78.8** | **20.0/46.0** | **RPC wins (Acc+ECE)** |
| KCC precedent relevance (4-class graded 0-3, chance 25) | 43.1/51.1 | 42.8/**45.1** | 43.3/51.1 | ~tie |
| LFUD fallacy MCQ | 88.0/12.1 | 87.0/**8.3** | 88.0/12.1 | ~tie |

Takeaway: **RPC beats SC only when the model is uncertain, answers are diverse, AND its confidence tracks correctness** (math, legal extraction — low accuracy). When the model is already confident/accurate (MCQ 88%), RPC ≈ SC. **KCC graded relevance is the telling case**: it *is* uncertain (43% on a 4-way task, chance 25) and answer-diverse, yet RPC ≈ SC and PPL is the *best*-calibrated (ECE 45 vs 51) — so diversity alone is **not sufficient**; RPC's perplexity-weighting + Weibull pruning only help when token-perplexity is informative about correctness. **PPL over-confidence is difficulty-dependent** — terrible on hard open tasks (ECE 49–93) but decent here (45) and on easy MCQ (8.3). A nuanced, faithful characterization of *when* RPC helps. See `docs/RPC_reproduction_results.md`.

**BIRD K-scaling (the budget effect, confirms Remark 6).** RPC *loses* to SC at K=8 only because K is below the paper's K=64–128 regime — re-aggregating the same paths at growing K shows RPC's advantage *grow monotonically*:

| K | SC Acc/ECE | RPC Acc/ECE | RPC vs SC |
|---|---|---|---|
| 8 | 27.5/37.7 | 27.0/**34.3** | ~tie acc, RPC best ECE |
| 16 | 28.4/36.4 | **28.8/31.3** | **RPC wins acc + ECE** |
| 32 | 27.5/38.2 | **30.0/26.7** | **RPC wins big (+2.5 acc, ½ ECE)** |

### Paper B — LCF · paper Table 1 layout
Conclusion Generation: Valid%(GPT4)↑ · Valid%(Trained)↑ · PPL↓ &nbsp;|&nbsp; Fallacy Identification: Acc↑ · ΔProb↑

| Model | | Valid%(GPT4) | Valid%(Trained) | PPL | Acc | ΔProb |
|:--|:--|:--:|:--:|:--:|:--:|:--:|
| **Qwen3-8B** (ours) | Original | 47.1 | 82.4 | 3.80 | 31.9 | 3.96 |
| | **+LCF** | 47.1 | 76.5 | **2.02** | 31.4 | **7.83** |
| **Llama-2-7b-chat** (ours) | Original | 35.3 | 100.0\* | 3.83 | 39.2 | 4.85 |
| | **+LCF** | 29.4 | 100.0\* | 2.26 | 27.0 | 2.44 |
| **Vicuna-7b** (ours, indep.)‡ | Original | 36.2 | — | 1.49 | 69.6 | 34.4 |
| | **+LCF** | 31–35 | — | 2.98 | 63.7 | 30.0 |
| _Llama2 (paper)_ | _Original_ | _70.58_ | _58.84_ | _21.08_ | _51.47_ | _−1.89_ |
| | _+LCF_ | _83.82_ | _96.56_ | _12.12_ | _75.00_ | _6.29_ |

\*Llama2 ValidTrained is degenerate (the distilbert judge marks all of Llama2's generations valid) — read ValidGPT4 instead.

‡ **A second, independent re-implementation** (separate codebase `lcf/independent_vicuna/`, Claude-Sonnet-4.6 judge, ΔProb ×100; PPL is greedy-text PPL, not comparable to other rows; sharing no code with the main pipeline, it controls for implementation bugs). On **Vicuna-7b, LCF helps on no metric** (η swept 0.25–8.0, neutral-to-harmful): the logic/content **disentanglement never forms** (t-SNE intermixed, separability 0.66 ≈ chance; quantization and label-quality controls rule those out). This **independently corroborates** our finding — LCF's one reproducible effect (ΔProb) is model-dependent, and the paper's headline gains reproduce on **no model we tested** (Qwen3, Llama2, Mistral, Vicuna). Details: `docs/LCF_vicuna_independent.md`.

**Mixed/negative — and we explain why (`docs/LCF_critical_analysis.md`).** On Qwen3-8B +LCF ~doubles ΔProb (3.96→7.83); on Llama-2-7b-chat the same recipe **degrades** it (4.85→2.44) — opposite to the paper. Root-cause analysis (the project's main contribution):
- **The logic-validity direction is real but weak.** A held-out probe separates valid/invalid at **0.82 at the single best sub-layer** (identical for both models) but **0.52 = chance when pooled** over the layers the paper mixes. Control: the *same* probe on **suicide-risk** (MoodRisk Mistral reps) hits **0.95 across all layers** — so representation editing works for semantic attributes; logic-validity is just weakly/locally encoded.
- **Separability ≠ controllability.** A model-agnostic redesign (best-layer supervised direction + norm-relative shift, `lcf_v2.py`) still gives no consistent gain at any strength — shifting along the direction does not causally steer logical behaviour.
- **The flagship metric is unauditable.** The paper's Valid%(Trained)=96.56 uses an unreleased self-trained discriminator; our analogue is degenerate; the auditable GPT-4 judge shows no gain.

**Verdict: not reproducible / not model-agnostic as published; the headline leans on an unauditable discriminator — but the premise is real and there is no evidence of fabrication.**

**Making it model-agnostic (v2→v5, prior-work-grounded; `docs/LCF_model_agnostic.md`).** We then tried to *fix* the model-dependence with the activation-steering literature (CAA, RepE, CAST/**K-CAST**, Valentino AAAI'26). v2 (supervised dir), v3 (CAA ± midpoint gate), v4 (faithful K-CAST kNN gate + LayerNavigator + signed sweep) all **fail to steer logic-validity on the fallacy task** — the kNN gate fires ~98% of tokens (reference/task distribution mismatch). On a purpose-built **formal-syllogism 2×2 (validity×believability)** task where reference==task (v5), **content-direction ablation debiases Qwen3 to 100% (content-effect gap 5→0) — the first positive steering result** — yet it is **still not model-agnostic** (Llama2, which can't do syllogisms at all, gets nothing) and the **conditional kNN gate never beats static**. So contrastive steering debiases content effects *only* when distribution-matched + targeting the content (not validity) direction + on a model that already has the capability.

**Cross-domain generalization (legal).** Running the LCF pipeline on legal valid/invalid conclusion pairs (Qwen3-8B, n=40/domain) is also inconsistent: **JurisNet degrades** (Acc 72.5→62.5, ΔProb 57.4→49.2) while **KCC-legal improves** (60.0→67.5, 50.3→57.6) — same model, opposite directions. Even *where* LCF helps is unstable, reinforcing the weak-signal verdict.

## Models & data
Paper RPC data = authors' published reasoning paths (auto-downloaded). LFUD = `github.com/YandaGo/LFUD`. Models: Qwen3-8B (local), Llama-2-7b-chat-hf + Vicuna/Mistral/ChatGLM3/Baichuan2 (downloaded); Llama-3.1 still HF-gated. BIRD at `/mnt/nfs/ssd2/bird_data`.

## ⚠️ GPU on DGX Spark GB10
Unified 128G memory shared CPU+GPU. Key rules: pre-download models from the login shell (NFS cache is read-only in jobs) and run with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NVIDIA_DISABLE_REQUIRE=1`; keep to ≤2× 7B jobs per node; if a load OOMs in the job scheduler while the node looks free, run it directly in the login shell.

---

# 한국어 요약

**저자:** 권지민 ([vanguard-gpt](https://github.com/vanguard-gpt)), 조민한

LLM 신뢰성(reliability) 분야의 **두 논문을 처음부터 구현·재현**하고, 논문이 쓴 데이터셋과 **로컬 가용 데이터셋(BIRD)** 으로 baseline과 비교하는 프로젝트입니다.

## 두 논문
| | Paper A — **RPC** | Paper B — **LCF** |
|---|---|---|
| 파일 | `paper/RPC_NeurIPS2025.pdf` | `paper/LCF_AAAI2025.pdf` |
| 학회 | NeurIPS 2025 | AAAI 2025 |
| 아이디어 | Test-time scaling: perplexity와 self-consistency를 결합(PC)하고, 저확률 추론경로를 Weibull 혼합분포로 가지치기(RP) | LLM hidden state를 **내용(content)/논리(logic)** 로 분리해, 논리 표현만 "타당한 영역"으로 이동(대조학습) |
| 학습 필요? | **아니오** (추론 시 집계만) | **예** (projector·decoder 학습, 기반 LLM은 동결) |
| 공식 코드 | github.com/WNJXYK/RPC (완비) | github.com/wulidongdong/LCF (**비어 있음** → 직접 구현) |

## 핵심 결과 (원 논문 레이아웃)

### Paper A — RPC · 논문 Table 2 형식 (Accuracy↑ / ECE↓), InternLM2-Math-Plus-7B

| 방법 | MATH | MathOdyssey | OlympiadBench | AIME | **평균** |
|:--|:--:|:--:|:--:|:--:|:--:|
| PPL | 46.99 / 48.99 | 27.35 / 67.70 | 7.27 / 86.90 | 5.96 / 88.98 | 21.89 / 73.14 |
| SC  | 50.57 / 6.71 | 28.25 / 12.23 | 11.07 / 20.20 | 9.40 / 14.35 | 24.82 / 13.37 |
| **RPC** | **51.94 / 6.41** | **31.77 / 9.69** | **11.12 / 18.89** | **9.75 / 14.30** | **26.15 / 12.32** |

→ 논문 평균(RPC 26.11/12.37 등)과 **일치**. RPC는 정확도·캘리브레이션(ECE)을 **동시에** 개선하고, PPL은 정확하지만 심하게 과신(ECE 최대 ~89)합니다.

**확장 — 로컬 데이터셋에 RPC 적용** (Qwen3-8B, K=8). RPC는 수학 밖에서도 SC를 이기는가? **도메인 의존적:**

| 도메인 (K=8) | SC Acc/ECE | PPL Acc/ECE | RPC Acc/ECE | RPC vs SC |
|---|---|---|---|---|
| 수학 (논문) | — | 과신 | best | **RPC 승** |
| BIRD text-to-SQL | **28.5**/39.8 | 25.0/73.6 | 25.0/39.2 | 패 |
| JurisNet 법률추출 | 19.1/52.4 | 18.7/**78.8** | **20.0/46.0** | **RPC 승(Acc+ECE)** |
| KCC 판례관련성 (4-class 등급 0-3, chance 25) | 43.1/51.1 | 42.8/**45.1** | 43.3/51.1 | ~무 |
| LFUD fallacy MCQ | 88.0/12.1 | 87.0/**8.3** | 88.0/12.1 | ~무 |

핵심: **RPC는 모델이 불확실하고·답이 다양하고·confidence가 정답을 잘 추종할 때만 SC를 이깁니다**(수학·법률추출, 저정확도). 모델이 이미 확신/정확하면(MCQ 88%) RPC≈SC. **KCC 등급분류가 결정적 사례**: 불확실하고(4-way 43%, chance 25) 답도 다양한데도 RPC≈SC이고 PPL이 오히려 best-calibrated(ECE 45 vs 51) — 즉 **답 다양성만으론 불충분**하고, RPC의 perplexity 가중+Weibull pruning은 token-perplexity가 정답성을 담을 때만 도움. **PPL 과신은 난이도 의존적** — 어려운 open 태스크 ECE 49–93, KCC는 45, MCQ는 8.3. "RPC가 *언제* 작동하는지"의 정밀한 특성화.

**BIRD K-scaling (예산 효과, Remark 6 확증).** RPC가 K=8에서 SC에 진 건 K가 논문의 K=64–128보다 작아서일 뿐 — 같은 경로를 K를 키워 재집계하면 RPC 우위가 **단조 증가**: K=8 (27.0/34.3, ~무) → K=16 (**28.8/31.3, RPC 승**) → K=32 (**30.0/26.7, RPC 대승 +2.5 acc, ECE 절반**). → `docs/RPC_reproduction_results.md`

### Paper B — LCF · 논문 Table 1 형식
Conclusion Generation: Valid%(GPT4)↑ · Valid%(Trained)↑ · PPL↓ &nbsp;|&nbsp; Fallacy Identification: Acc↑ · ΔProb↑

| 모델 | | Valid%(GPT4) | Valid%(Trained) | PPL | Acc | ΔProb |
|:--|:--|:--:|:--:|:--:|:--:|:--:|
| **Qwen3-8B** (구현) | Original | 47.1 | 82.4 | 3.80 | 31.9 | 3.96 |
| | **+LCF** | 47.1 | 76.5 | **2.02** | 31.4 | **7.83** |
| **Llama-2-7b-chat** (구현) | Original | 35.3 | 100.0\* | 3.83 | 39.2 | 4.85 |
| | **+LCF** | 29.4 | 100.0\* | 2.26 | 27.0 | 2.44 |
| **Vicuna-7b** (구현, 독립)‡ | Original | 36.2 | — | 1.49 | 69.6 | 34.4 |
| | **+LCF** | 31–35 | — | 2.98 | 63.7 | 30.0 |
| _Llama2 (논문)_ | _Original_ | _70.58_ | _58.84_ | _21.08_ | _51.47_ | _−1.89_ |
| | _+LCF_ | _83.82_ | _96.56_ | _12.12_ | _75.00_ | _6.29_ |

\*Llama2의 ValidTrained는 무의미합니다(distilbert 판정기가 Llama2 생성물을 전부 valid로 분류). **ValidGPT4**를 보세요.

‡ **두 번째 독립 재구현**(별도 코드베이스 `lcf/independent_vicuna/`, Claude-Sonnet-4.6 판별기, ΔProb는 ×100; PPL은 greedy-text PPL이라 다른 행과 비교 불가; 메인 파이프라인과 코드를 공유하지 않아 구현 버그를 통제). **Vicuna-7b에서 LCF는 어떤 지표도 개선 못 함**(η 0.25–8.0 중립~악화): 논리/내용 **disentanglement 미형성**(t-SNE 혼재, separability 0.66 ≈ 우연; 양자화·라벨품질 통제실험으로 기각). 본 프로젝트 결론(ΔProb 효과는 모델 의존적)을 **독립적으로 뒷받침** — 논문 헤드라인 gains는 우리가 시험한 **어떤 모델**(Qwen3·Llama2·Mistral·Vicuna)**에서도 재현되지 않음**. 상세: `docs/LCF_vicuna_independent.md`.

→ **혼재/부정 — 그리고 이유를 규명했습니다 (`docs/LCF_critical_analysis.md`, 본 프로젝트의 핵심 기여).** Qwen3는 ΔProb 2배(3.96→7.83), Llama2는 동일 recipe로 **악화**(4.85→2.44, 논문과 반대). 근본 원인:
- **logic 방향은 실재하나 약함**: held-out probe로 valid/invalid 분리도가 **best 단일 sub-layer 0.82**(두 모델 동일)이나 레이어를 섞으면 **0.52(chance)**. 대조군 — 같은 probe를 **자살위험**(MoodRisk Mistral reps)에 적용하면 **0.95(전 레이어)**. 즉 representation editing은 *의미적* 속성엔 작동하나, logic-validity는 약하게·국소적으로만 인코딩됨.
- **분리 ≠ 제어**: model-agnostic 재설계 v2(best-layer 지도방향 + norm-상대 개입)조차 어떤 강도에서도 일관된 개선 없음 — 방향으로 밀어도 논리 행동이 인과적으로 안 바뀜.
- **headline 지표 감사불가**: Valid%(Trained) 96.56은 미공개 self-trained discriminator 의존, 내 analogue은 degenerate, 감사가능한 GPT-4 judge는 개선 없음.

**판정: 논문대로 재현 불가 / model-agnostic 아님 / headline은 감사불가 discriminator 의존 — 단, 전제는 실재하고 날조 증거는 없음.**

**model-agnostic 개량 시도 (v2→v5, 선행연구 기반; `docs/LCF_model_agnostic.md`).** activation-steering 문헌(CAA, RepE, CAST/**K-CAST**, Valentino AAAI'26)에 근거해 모델 의존성을 *고치려* 시도. v2(지도방향)·v3(CAA±midpoint gate)·v4(충실한 K-CAST kNN gate + LayerNavigator + 부호 탐색) 모두 fallacy 과제에서 **logic-validity steering 실패** — kNN gate가 토큰의 ~98%에서 발화(reference/task 분포 불일치). reference==task인 **형식 삼단논법 2×2(validity×believability)** 과제(v5)에선 **content 방향 ablation이 Qwen3를 100%로 debias(content-effect gap 5→0) — 프로젝트 첫 positive 결과** — 그러나 **여전히 model-agnostic 아님**(삼단논법 능력 자체가 없는 Llama2엔 무효), **조건부 kNN gate는 static을 끝내 못 넘음**. 즉 contrastive steering은 *분포 일치 + content(아닌 validity) 방향 + 능력 보유 모델* 조건에서만 content effect를 debias함.

**도메인 일반화 (법률).** LCF 파이프라인을 법률 valid/invalid 결론쌍에 적용(Qwen3-8B, 도메인당 n=40)해도 비일관적: **JurisNet 악화**(Acc 72.5→62.5, ΔProb 57.4→49.2) vs **KCC-legal 개선**(60.0→67.5, 50.3→57.6) — 같은 모델, 반대 방향. LCF가 *어디서* 돕는지조차 불안정 → weak-signal 판정 재확인.

## 재현 방법
```bash
# Paper A (CPU만, GPU 불필요)
cd rpc/RPC && bash run_full_repro.sh

# Paper B (GPU; GB10 통합메모리 주의 — 노드당 7B 2개 이하, HF offline)
cd lcf/lcf_impl && uv run python lfud_data.py --model gpt-3.5-turbo   # 데이터(OpenAI 키 필요)
bash lcf/run_lcf_full.sh meta-llama/Llama-2-7b-chat-hf               # 추출→학습→평가
cd lcf/eval && uv run python postprocess_judge.py --judge-model gpt-4o   # GPT-4 판정 채우기
```

## 진행 현황
- ✅ **RPC 재현 완료**(33셀 = 논문 Table 2) + **4개 신규 도메인 확장**(BIRD·JurisNet·KCC·LFUD-MCQ) + **BIRD K-scaling**(K=8/16/32 → RPC 우위가 K와 함께 증가, Remark 6 확증)
- ✅ **LCF 직접 구현 + 비판적 검증 완료** — `docs/LCF_critical_analysis.md`: 전제는 실재(0.82)하나 약함, 논문 recipe가 신호 희석, **분리≠제어**(v2도 개선 없음), headline 지표 감사불가. 날조 아님.
- ✅ **model-agnostic 개량 v2→v5 완료** — `docs/LCF_model_agnostic.md`: v3(CAA)·v4(K-CAST)는 fallacy 과제 실패(gate ~98% 발화), v5 형식 삼단논법에선 content-ablation이 Qwen3를 100%로 debias(첫 positive). 단 여전히 non-agnostic + kNN gate는 static 못 넘음.
- ✅ **MoodRisk 대조 probe**(위험방향 0.95) + **multi-model LCF**(Mistral도 Llama2처럼 악화)
- ✅ LCF 베이스라인(SFT/ITI/RAHF) 빌드/평가 완료. 법률-LCF 데이터(JurisNet·KCC)는 빌드만 완료(미실행 — model-agnostic 스트림으로 대체).

> 핵심 문서: **`docs/LCF_critical_analysis.md`**(검증) · **`docs/LCF_model_agnostic.md`**(v2→v5 개량) · `docs/RPC_reproduction_results.md` · `docs/LCF_reproduction_results.md`. 외부 레포(RPC/LFUD/ITI/RAHF)·논문 PDF는 라이선스상 vendoring하지 않거나 별도이며 URL만 명시.
