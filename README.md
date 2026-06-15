# reliableAI_final — Two-Paper Reproduction & Extension

**Authors:** Jimin Kwon, Minhan Cho

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
- **RPC: reproduced exactly** (33-cell grid = paper Table 2) and **extended to 4 new domains** — BIRD SQL, JurisNet legal extraction, KCC precedent relevance, LFUD MCQ — to map *where* RPC helps. → `docs/RPC_reproduction_results.md`
- **LCF: implemented from scratch + critically analysed.** Reproduction is mixed/negative, and we trace *why*: the logic-validity direction is **real but weak (0.82 best single layer vs 0.95 for a semantic attribute), diluted to chance when layers are pooled**, and **not causally controllable** by representation shifting; the paper's flagship 96.56% relies on an **unauditable, confoundable discriminator**. Verdict: not reproducible / not model-agnostic as published — *not* fabrication. → **`docs/LCF_critical_analysis.md`** (+ `docs/LCF_reproduction_results.md`)
- **LCF baselines** (SFT/ITI/RAHF) + **model-agnostic redesign v2** (best-layer supervised direction + norm-relative intervention): built and evaluated — v2 also gives no consistent gain (separability ≠ controllability).
- **Generalization datasets built**: legal-LCF (JurisNet & KCC precedents → valid/invalid conclusion pairs), MoodRisk risk-direction probe (control showing the premise holds for semantic attributes).

## Layout
```
paper/                     the two PDFs (RPC_NeurIPS2025, LCF_AAAI2025)
rpc/RPC/                   cloned official RPC repo (uv venv, CPU-only); + run_full_repro.sh
rpc/_batched_gen.py        batched K-sample generation helper (shared by extensions)
rpc/bird_extension/        RPC on BIRD text-to-SQL (SQL exec-match)
rpc/jurisnet_ext/          RPC on JurisNet legal statute extraction (exact-match)
rpc/kcc_ext/               RPC on KCC precedent-relevance (balanced binary)
rpc/lfud_mcq/              RPC on LFUD fallacy-identification MCQ (connects both papers)
lcf/lcf_impl/              from-scratch LCF: model/losses/data/train/infer
   probe_layers.py, lcf_v2.py, lcf_v2_eval.py   model-agnostic v2 + per-layer probe
   moodrisk_probe.py       control: risk-direction probe on MoodRisk Mistral reps
lcf/eval/                  metrics (GPT-4 judge, discriminator), run_eval, postprocess_judge
lcf/baselines/             SFT / ITI / RAHF
lcf/legal/, lcf/kcc_legal/ legal-domain LCF data (valid/invalid conclusion pairs)
lcf/run_lcf_full.sh        one-shot: extract->train->eval (run ALONE on a node)
docs/   LCF_critical_analysis.md  ← main finding · RPC/LCF_reproduction_results.md · LCF_implementation_spec.md · MASTER_PLAN.md
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

| Domain (Qwen3-8B, K=8) | SC Acc | PPL Acc / ECE | RPC Acc / ECE | RPC vs SC |
|---|---|---|---|---|
| math (paper) | — | (over-confident) | — | RPC wins |
| BIRD text-to-SQL (exec-match) | **28.5** | 25.0 / 73.6 | 25.0 / 39.2 | ~tie/lose |
| JurisNet legal extraction (exact-match) | 19.1 / 52.4 | 18.7 / **78.8** | **20.0 / 46.0** | **RPC wins (Acc+ECE)** |
| KCC precedent relevance (balanced) | _running_ | _running_ | _running_ | _pending_ |
| LFUD fallacy MCQ | _running_ | _running_ | _running_ | _pending_ |

Takeaway: **PPL is badly over-confident everywhere (ECE 49–93)** — the paper's core point reproduces robustly. RPC's *accuracy* edge over SC is domain-dependent (clear on math & legal extraction; absent on BIRD SQL at the small K=8 the paper warns about). See `docs/RPC_reproduction_results.md`.

### Paper B — LCF · paper Table 1 layout
Conclusion Generation: Valid%(GPT4)↑ · Valid%(Trained)↑ · PPL↓ &nbsp;|&nbsp; Fallacy Identification: Acc↑ · ΔProb↑

| Model | | Valid%(GPT4) | Valid%(Trained) | PPL | Acc | ΔProb |
|:--|:--|:--:|:--:|:--:|:--:|:--:|
| **Qwen3-8B** (ours) | Original | 47.1 | 82.4 | 3.80 | 31.9 | 3.96 |
| | **+LCF** | 47.1 | 76.5 | **2.02** | 31.4 | **7.83** |
| **Llama-2-7b-chat** (ours) | Original | 35.3 | 100.0\* | 3.83 | 39.2 | 4.85 |
| | **+LCF** | 29.4 | 100.0\* | 2.26 | 27.0 | 2.44 |
| _Llama2 (paper)_ | _Original_ | _70.58_ | _58.84_ | _21.08_ | _51.47_ | _−1.89_ |
| | _+LCF_ | _83.82_ | _96.56_ | _12.12_ | _75.00_ | _6.29_ |

\*Llama2 ValidTrained is degenerate (the distilbert judge marks all of Llama2's generations valid) — read ValidGPT4 instead.

**Mixed/negative — and we explain why (`docs/LCF_critical_analysis.md`).** On Qwen3-8B +LCF ~doubles ΔProb (3.96→7.83); on Llama-2-7b-chat the same recipe **degrades** it (4.85→2.44) — opposite to the paper. Root-cause analysis (the project's main contribution):
- **The logic-validity direction is real but weak.** A held-out probe separates valid/invalid at **0.82 at the single best sub-layer** (identical for both models) but **0.52 = chance when pooled** over the layers the paper mixes. Control: the *same* probe on **suicide-risk** (MoodRisk Mistral reps) hits **0.95 across all layers** — so representation editing works for semantic attributes; logic-validity is just weakly/locally encoded.
- **Separability ≠ controllability.** A model-agnostic redesign (best-layer supervised direction + norm-relative shift, `lcf_v2.py`) still gives no consistent gain at any strength — shifting along the direction does not causally steer logical behaviour.
- **The flagship metric is unauditable.** The paper's Valid%(Trained)=96.56 uses an unreleased self-trained discriminator; our analogue is degenerate; the auditable GPT-4 judge shows no gain.

**Verdict: not reproducible / not model-agnostic as published; the headline leans on an unauditable discriminator — but the premise is real and there is no evidence of fabrication.**

## Models & data
Paper RPC data = authors' published reasoning paths (auto-downloaded). LFUD = `github.com/YandaGo/LFUD`. Models: Qwen3-8B (local), Llama-2-7b-chat-hf + Vicuna/Mistral/ChatGLM3/Baichuan2 (downloaded); Llama-3.1 still HF-gated. BIRD at `/mnt/nfs/ssd2/bird_data`.

## ⚠️ GPU on DGX Spark GB10
Unified 128G memory shared CPU+GPU. Key rules: pre-download models from the login shell (NFS cache is read-only in jobs) and run with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NVIDIA_DISABLE_REQUIRE=1`; keep to ≤2× 7B jobs per node; if a load OOMs in the job scheduler while the node looks free, run it directly in the login shell.

---

# 한국어 요약

**저자:** 권지민, 조민한

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

| 도메인 (K=8) | SC Acc | PPL Acc/ECE | RPC Acc/ECE | RPC vs SC |
|---|---|---|---|---|
| 수학 (논문) | — | 과신 | — | RPC 승 |
| BIRD text-to-SQL | **28.5** | 25.0 / 73.6 | 25.0 / 39.2 | 무/패 |
| JurisNet 법률추출 | 19.1 / 52.4 | 18.7 / **78.8** | **20.0 / 46.0** | **RPC 승(Acc+ECE)** |
| KCC 판례관련성 | _실행중_ | _실행중_ | _실행중_ | _대기_ |
| LFUD fallacy MCQ | _실행중_ | _실행중_ | _실행중_ | _대기_ |

핵심: **PPL은 전 도메인에서 심하게 과신(ECE 49–93)** — 논문 핵심 주장은 견고히 재현. RPC의 *정확도* 우위는 도메인 의존적(수학·법률추출은 승, BIRD SQL은 작은 K=8에서 무).

### Paper B — LCF · 논문 Table 1 형식
Conclusion Generation: Valid%(GPT4)↑ · Valid%(Trained)↑ · PPL↓ &nbsp;|&nbsp; Fallacy Identification: Acc↑ · ΔProb↑

| 모델 | | Valid%(GPT4) | Valid%(Trained) | PPL | Acc | ΔProb |
|:--|:--|:--:|:--:|:--:|:--:|:--:|
| **Qwen3-8B** (구현) | Original | 47.1 | 82.4 | 3.80 | 31.9 | 3.96 |
| | **+LCF** | 47.1 | 76.5 | **2.02** | 31.4 | **7.83** |
| **Llama-2-7b-chat** (구현) | Original | 35.3 | 100.0\* | 3.83 | 39.2 | 4.85 |
| | **+LCF** | 29.4 | 100.0\* | 2.26 | 27.0 | 2.44 |
| _Llama2 (논문)_ | _Original_ | _70.58_ | _58.84_ | _21.08_ | _51.47_ | _−1.89_ |
| | _+LCF_ | _83.82_ | _96.56_ | _12.12_ | _75.00_ | _6.29_ |

\*Llama2의 ValidTrained는 무의미합니다(distilbert 판정기가 Llama2 생성물을 전부 valid로 분류). **ValidGPT4**를 보세요.

→ **혼재/부정 — 그리고 이유를 규명했습니다 (`docs/LCF_critical_analysis.md`, 본 프로젝트의 핵심 기여).** Qwen3는 ΔProb 2배(3.96→7.83), Llama2는 동일 recipe로 **악화**(4.85→2.44, 논문과 반대). 근본 원인:
- **logic 방향은 실재하나 약함**: held-out probe로 valid/invalid 분리도가 **best 단일 sub-layer 0.82**(두 모델 동일)이나 레이어를 섞으면 **0.52(chance)**. 대조군 — 같은 probe를 **자살위험**(MoodRisk Mistral reps)에 적용하면 **0.95(전 레이어)**. 즉 representation editing은 *의미적* 속성엔 작동하나, logic-validity는 약하게·국소적으로만 인코딩됨.
- **분리 ≠ 제어**: model-agnostic 재설계 v2(best-layer 지도방향 + norm-상대 개입)조차 어떤 강도에서도 일관된 개선 없음 — 방향으로 밀어도 논리 행동이 인과적으로 안 바뀜.
- **headline 지표 감사불가**: Valid%(Trained) 96.56은 미공개 self-trained discriminator 의존, 내 analogue은 degenerate, 감사가능한 GPT-4 judge는 개선 없음.

**판정: 논문대로 재현 불가 / model-agnostic 아님 / headline은 감사불가 discriminator 의존 — 단, 전제는 실재하고 날조 증거는 없음.**

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
- ✅ **RPC 재현 완료**(33셀 = 논문 Table 2) + **4개 신규 도메인 확장**(BIRD·JurisNet·KCC·LFUD-MCQ; KCC/MCQ는 Node 2에서 생성 중)
- ✅ **LCF 직접 구현 + 비판적 검증 완료** — `docs/LCF_critical_analysis.md`: 전제는 실재(0.82)하나 약함, 논문 recipe가 신호 희석, **분리≠제어**(v2도 개선 없음), headline 지표 감사불가. 날조 아님.
- ✅ **MoodRisk 대조 probe**(위험방향 0.95) — representation-editing 전제가 의미적 속성엔 강함을 입증
- 🟡 LCF 베이스라인(SFT/ITI/RAHF)·법률 LCF 데이터(JurisNet·KCC): 빌드 완료, 일부 실행 대기

> 핵심 문서: **`docs/LCF_critical_analysis.md`**(검증), `docs/RPC_reproduction_results.md`, `docs/LCF_reproduction_results.md`. 외부 레포(RPC/LFUD/ITI/RAHF)·논문 PDF는 라이선스상 vendoring하지 않거나 별도이며 URL만 명시.
