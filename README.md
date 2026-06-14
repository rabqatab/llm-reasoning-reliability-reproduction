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
- **RPC: reproduced exactly.** Full 33-cell grid matches paper Table 2. → `docs/RPC_reproduction_results.md`
- **RPC BIRD extension: done** (honest negative at K=8). → same doc
- **LCF: implemented from scratch.** Mixed reproduction: +LCF ~doubles ΔProb on Qwen3-8B (matches paper) but **degrades** Llama-2-7b-chat (opposite to paper) — contrastive objective under-fits with the paper's hyperparameters. → `docs/LCF_reproduction_results.md`
- **LCF baselines** (SFT/ITI/RAHF): code built; ITI ran; SFT/RAHF re-run pending.

## Layout
```
paper/                  the two PDFs
rpc/RPC/                cloned official RPC repo (uv venv, CPU-only)
rpc/RPC/run_full_repro.sh   full reproduction grid -> results_full.txt
rpc/bird_extension/     RPC applied to BIRD text-to-SQL (Qwen3 gen + SQL exec-match)
lcf/lcf_impl/           from-scratch LCF: model, losses, data, train, infer
lcf/eval/               metrics (GPT-4 judge, discriminator), run_eval, postprocess_judge
lcf/baselines/          SFT / ITI / RAHF
lcf/data/               LFUD splits, conclusion_gen_*, fallacy_id_*, extracted reps
lcf/checkpoints/<model>/lcf.pt   trained LCF adapters
lcf/run_lcf_full.sh     one-shot: extract->train->eval original+ +LCF (run ALONE on a node)
docs/                   spec, results, sparkq issues, this plan
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

**Extension — RPC on local BIRD text-to-SQL** (Qwen3-8B, K=8, SQL execution-match): SC **28.5** > RPC 25.0 = PPL 25.0. Honest negative — RPC's Weibull path-pruning is unreliable at K=8 (paper used K=64–128); PPL over-confidence (ECE 73.6) still reproduces.

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

**Mixed, honest result.** On **Qwen3-8B**, +LCF reproduces the paper's central claim — it **~doubles ΔProb** (3.96 → 7.83), moving the logic representation so the model favors the *valid* conclusion. On **Llama-2-7b-chat** (the paper's headline model) the same code/hyperparameters **degrade** identification (ΔProb 4.85 → 2.44, Acc 39 → 27) — opposite to the paper. Diagnosis: the contrastive logic objective barely trains (InfoNCE plateaus at ~chance, ln(batch)≈5.5, for **both** models) with the paper's settings (lr 1e-3, 10 epochs), so the learned validity direction is weak; applying the strong η=4.5 nudge along it helps one model and hurts another. The paper likely needed stronger/tuned contrastive training and/or per-model η. Conclusion-generation gains are also muted under greedy decoding. So: core mechanism reproduced on Qwen3, **not** cleanly reproduced on Llama2 — under investigation.

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

**확장 — 로컬 BIRD text-to-SQL에 RPC 적용** (Qwen3-8B, K=8, SQL 실행결과 비교): SC **28.5** > RPC 25.0 = PPL 25.0. **정직한 negative** — K=8에서는 RPC의 Weibull 가지치기가 불안정합니다(논문은 K=64–128). PPL 과신(ECE 73.6)은 동일하게 재현됩니다.

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

→ **혼재된, 정직한 결과.** **Qwen3-8B**에서는 +LCF가 논문 핵심 주장을 재현 — **ΔProb를 약 2배**(3.96 → 7.83)로 올려 모델이 *타당한* 결론을 선호하게 만듭니다. 반면 **Llama-2-7b-chat**(논문 주력 모델)에서는 동일 코드·하이퍼파라미터인데도 식별 성능이 **악화**(ΔProb 4.85 → 2.44, Acc 39 → 27)되어 논문과 반대입니다. 진단: 논문 설정(lr 1e-3, 10 epoch)에서 **대조학습(InfoNCE)이 두 모델 모두 무작위 수준(ln(batch)≈5.5)에서 정체**해 학습된 "타당성 방향(V)"이 약하고, 강한 η=4.5 nudge의 효과가 모델마다 다르게 나타납니다. 논문은 더 강한/튜닝된 대조학습이나 모델별 η가 필요했을 가능성이 큽니다. 생성 태스크는 greedy 디코딩에서 효과가 약했습니다. 즉 **핵심 메커니즘은 Qwen3에서 재현, Llama2에서는 깔끔히 재현 안 됨 — 조사 중**입니다.

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
- ✅ **RPC 재현 완료** (전체 33셀이 논문 Table 2와 일치) + BIRD 확장 완료
- ✅ **LCF 직접 구현 완료** — 혼재된 재현: Qwen3-8B는 ΔProb 2배(논문 일치), **Llama-2-7b-chat은 악화**(논문과 반대) → 대조학습 under-fit이 원인으로 추정, 조사 중
- 🟡 LCF 베이스라인(SFT/ITI/RAHF): 코드 완비, ITI 실행됨, SFT/RAHF 재실행 예정

> 세부 결과는 `docs/RPC_reproduction_results.md`, `docs/LCF_reproduction_results.md` 참고. 외부 레포(RPC/LFUD/ITI/RAHF)는 라이선스상 vendoring하지 않으며 URL만 명시합니다.
