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

## Key results so far
- **RPC** InternLM2-Math-Plus-7B / MathOdyssey: PPL 27.4 (ECE 67.7, overconfident) · SC 28.3 · **RPC 31.8 (ECE 9.7)** — matches paper.
- **RPC BIRD** (Qwen3-8B, K=8): SC 28.5 > RPC 25.0 = PPL 25.0 — RPC needs larger K (honest).
- **LCF** Qwen3-8B: ΔProb **3.96 → 7.83** with LCF (fallacy identification) — matches paper's direction. Conclusion-generation muted under greedy decoding. Llama-2-7b-chat (paper model) running.

## Models & data
Paper RPC data = authors' published reasoning paths (auto-downloaded). LFUD = `github.com/YandaGo/LFUD`. Models: Qwen3-8B (local), Llama-2-7b-chat-hf + Vicuna/Mistral/ChatGLM3/Baichuan2 (downloaded); Llama-3.1 still HF-gated. BIRD at `/mnt/nfs/ssd2/bird_data`.

## ⚠️ GPU on DGX Spark GB10
Unified 128G memory shared CPU+GPU. Key rules: pre-download models from the login shell (NFS cache is read-only in jobs) and run with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NVIDIA_DISABLE_REQUIRE=1`; keep to ≤2× 7B jobs per node; if a load OOMs in the job scheduler while the node looks free, run it directly in the login shell.
