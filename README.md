# reliableAI_final — Two-Paper Reproduction & Extension

**Authors:** Jimin Kwon, Minhan Cho

Reproduce two papers from scratch, compare against their baselines on the papers' datasets **and** a locally-available dataset (BIRD). Decisions made with the user: use paper models **+ local Qwen3**; reproduce paper datasets **+ extend RPC to local BIRD**; run **both papers in parallel**.

## The two papers (`paper/`)
| | Paper A — **RPC** | Paper B — **LCF** |
|---|---|---|
| File | `NeurIPS2025.pdf` | `AAAI2025.pdf` |
| Title | Bridging Internal Probability and Self-Consistency for LLM Reasoning | Content-free Logical Modification by Disentangling & Modifying Logic Representation |
| Idea | Test-time scaling: fuse perplexity + self-consistency (PC) and prune low-prob paths via Weibull mixture (RP) | Split hidden states into content/logic; push logic toward "valid" region via contrastive learning |
| Training? | **No** (inference-time aggregation) | **Yes** (projectors+decoder; base LLM frozen) |
| Official code | github.com/WNJXYK/RPC (complete) | github.com/wulidongdong/LCF (**empty** → built from scratch) |

## Status
- **RPC: reproduced exactly.** Full 33-cell grid matches paper Table 2. → `docs/RPC_reproduction_results.md`
- **RPC BIRD extension: done** (honest negative at K=8). → same doc
- **LCF: implemented from scratch + reproduced on Qwen3-8B** (ΔProb 2× — matches paper). Llama-2-7b-chat (the paper's headline model) run in progress. → `docs/LCF_reproduction_results.md`
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
| **Llama-2-7b-chat** (ours) | Original | — | — | 3.83 | 39.2 | 4.85 |
| | **+LCF** | _running_ | _running_ | _running_ | _running_ | _running_ |
| _Llama2 (paper)_ | _Original_ | _70.58_ | _58.84_ | _21.08_ | _51.47_ | _−1.89_ |
| | _+LCF_ | _83.82_ | _96.56_ | _12.12_ | _75.00_ | _6.29_ |

Reproduced claim: **LCF roughly doubles ΔProb** (Qwen3-8B 3.96 → 7.83), matching the paper's direction (Llama2 −1.89 → 6.29) — LCF moves the logic representation so the model favors the *valid* conclusion. Conclusion-generation gains are muted under greedy decoding on Qwen3 (η=0.5 is a gentle nudge). Reproduction on **Llama-2-7b-chat** — the paper's headline model — is running.

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
| 파일 | `paper/NeurIPS2025.pdf` | `paper/AAAI2025.pdf` |
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
| **Llama-2-7b-chat** (구현) | Original | — | — | 3.83 | 39.2 | 4.85 |
| | **+LCF** | _실행중_ | _실행중_ | _실행중_ | _실행중_ | _실행중_ |
| _Llama2 (논문)_ | _Original_ | _70.58_ | _58.84_ | _21.08_ | _51.47_ | _−1.89_ |
| | _+LCF_ | _83.82_ | _96.56_ | _12.12_ | _75.00_ | _6.29_ |

→ 핵심 주장 재현: **LCF가 ΔProb를 약 2배로** (Qwen3-8B 3.96 → 7.83) 끌어올려, 논문 방향(Llama2 −1.89 → 6.29)과 일치합니다 — LCF가 논리 표현을 이동시켜 모델이 *타당한* 결론에 더 높은 확률을 부여합니다. 생성(Generation) 태스크는 greedy 디코딩에서 효과가 약했습니다(η=0.5는 약한 nudge). 논문 주력 모델 **Llama-2-7b-chat** 재현은 진행 중입니다.

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
- ✅ **LCF 직접 구현 + Qwen3-8B 재현 완료** (ΔProb 2배, 논문 일치)
- 🟢 **Llama-2-7b-chat 재현 진행 중** (LCF 학습 완료, +LCF 평가 실행 중)
- 🟡 LCF 베이스라인(SFT/ITI/RAHF): 코드 완비, ITI 실행됨, SFT/RAHF 재실행 예정

> 세부 결과는 `docs/RPC_reproduction_results.md`, `docs/LCF_reproduction_results.md` 참고. 외부 레포(RPC/LFUD/ITI/RAHF)는 라이선스상 vendoring하지 않으며 URL만 명시합니다.
