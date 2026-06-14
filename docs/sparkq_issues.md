# sparkq / DGX Spark GB10 — Issues Log & Fixes

Consolidated record of every sparkq / GPU error hit while running the reliableAI_final jobs on the DGX Spark GB10 nodes (2026-06-14), with root cause and the fix/workaround. Ordered roughly by how much pain they caused.

---

## 1. GB10 unified-memory OOM (the big one)

GB10 has **128 GB UNIFIED memory shared by CPU and GPU** — not a separate VRAM pool. Every memory issue below traces back to this.

### 1a. Concurrent 7B jobs exhaust unified memory
- **Symptom:** 3 jobs running (Qwen3 baselines 45G + Llama2 pipeline 35G + Llama2 eval 35G) → `torch.AcceleratorError: CUDA error: out of memory` at `model.to(device)`. `sparkq status` showed GPU 115G/128G **plus** 64G CPU-mem — combined > 128G physical.
- **Cause:** sparkq reports `--gpu-mem` and `--cpu-mem` as separate numbers, but on unified memory they draw from the **same** 128G. Desktop GUI (gnome-shell ~9G, gnome-control-center ~6G) permanently eats ~15G more.
- **Fix:** **Max 2× 7B jobs per node.** Treat (gpu-mem + cpu-mem + ~15G GUI) as the real footprint and keep the sum under ~110G.

### 1b. sparkq job OOMs where the login shell succeeds  ← most surprising
- **Symptom:** Loading `Llama-2-7b-chat-hf` to GPU OOMs **inside a sparkq job** at `model.to(device)`, but the **identical load runs fine from the login shell** (`torch.cuda.mem_get_info` 40.1G → 12.9G free, success).
- **Cause:** `from_pretrained` loads weights to **CPU RAM first (~14G)**, then `.to(device)` copies to GPU (+14–27G). On unified memory both live at once → ~40G transient peak. sparkq evidently runs the job under a cgroup whose limit (driven by `--gpu-mem`/`--cpu-mem`) doesn't account for this CPU+GPU overlap, so the job is capped below the real peak even though the node has the memory.
- **Workaround that worked:** **bypass sparkq for the heavy load** — run the pipeline directly in the login shell:
  ```bash
  cd lcf/lcf_impl
  export HF_HOME=/mnt/nfs/ssd1/huggingface_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
         NVIDIA_DISABLE_REQUIRE=1 PYTHONPATH=<root> WANDB_MODE=offline
  nohup bash run_lcf_full.sh meta-llama/Llama-2-7b-chat-hf > run.log 2>&1 &
  ```
  Past `model.to(device)` immediately. (If you must use sparkq, give very generous `--cpu-mem` so the transient CPU-side copy fits.)

### 1c. Killed-job memory not reclaimed instantly
- **Symptom:** Cancel a job, resubmit another within ~5s → new job OOMs at model load even though `sparkq status` shows the node "empty".
- **Cause:** A killed CUDA process takes seconds–tens-of-seconds to release unified memory; `nvidia-smi --query-compute-apps` may already show nothing while the driver is still reclaiming. `nvidia-smi --query-gpu=memory.*` returns **`[N/A]` on GB10** (can't read it that way) — use `python -c "import torch; torch.cuda.mem_get_info()"` instead.
- **Fix:** After cancelling, **wait and verify** `torch.cuda.mem_get_info()[0]` (free bytes) has recovered before resubmitting; don't trust an empty `sparkq status` alone.

### 1d. Dependent jobs submitted concurrently → race + OOM
- **Symptom:** Submitted the LCF eval at the same time as the LCF training pipeline; eval's `+LCF` step needs the checkpoint the pipeline is still producing, and both loading 7B at once OOM'd.
- **Fix:** **Chain dependent steps in ONE bash driver** (`run_lcf_full.sh` = extract → train → eval), submitted as a single job. sparkq has no job-dependency mechanism.

---

## 2. NFS Hugging Face cache is READ-ONLY inside sparkq jobs
- **Symptom:** `OSError: [Errno 30] Read-only file system: '/mnt/nfs/ssd1/huggingface_cache/hub/.locks/...lock'` — fails creating HF's download lock file.
- **Cause:** `HF_HOME` is on NFS (`/mnt/nfs/ssd1`); the sparkq job context mounts it read-only (the login shell has it RW).
- **Fix:** **Pre-download every model/dataset from the login shell**, then run jobs with `--env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1` so HF uses the cache without locking or network.

## 3. Cached model missing weight files (offline mode can't recover)
- **Symptom:** `microsoft/deberta-v3-small does not appear to have a file named pytorch_model.bin or model.safetensors` — config+tokenizer were cached, weights were not; offline mode can't fetch them.
- **Fix:** Before submitting an offline job, **verify weights exist** (`find <cache>/<model> -name '*.safetensors' -o -name '*.bin'`); if missing, `snapshot_download(model)` fully from the login shell first.

## 4. torch absent from the RPC venv
- **Symptom:** BIRD generation `ModuleNotFoundError: No module named 'torch'` — `rpc/RPC/.venv` is CPU-only (datasets/numpy/scipy/sympy).
- **Fix:** Run torch jobs with the **`lcf/lcf_impl` venv**: `--workdir lcf/lcf_impl` + `uv run python <abs path to script>`. `uv run --project <dir>` did **not** reliably resolve torch.

---

## 5. transformers 5.x API breaks (the venvs got transformers 5.12)
- `Trainer(tokenizer=...)` removed → **`Trainer(processing_class=...)`**. (broke discriminator + would break any HF Trainer code).
- `tok.apply_chat_template(..., return_tensors="pt")` returns a **BatchEncoding, not a tensor** → `.shape` fails. Use `return_dict=True` and index `["input_ids"]`.
- **deberta-v3 SentencePiece tokenizer is broken under transformers 5.x** → classifier trains to `eval_accuracy=0.5` / `eval_loss=0.693` (pure chance), predicts one class for everything. Fix: use a model with a fast tokenizer + cached `tokenizer.json` (we switched to **distilbert-base-uncased**, val acc 1.0).

## 6. SFT data collator can't pad pre-masked labels
- **Symptom:** `ValueError: expected sequence of length 21 at dim 1 (got 15)` / "features (`labels`) have excessive nesting... activate padding". `DataCollatorForLanguageModeling(mlm=False)` does not pad a pre-set variable-length `labels` field.
- **Fix:** **`DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)`** — pads input_ids and labels, preserving prompt masking.

## 7. Cross-agent interface mismatches (multi-agent build cost)
- `run_eval` imported module-level `generate_with_lcf` / `score_options_with_lcf`; `infer.py` only defined them as `LCFInference` methods → ImportError. Fix: added module-level **singleton wrappers** (configurable via `LCF_EVAL_MODEL` / `LCF_EVAL_CKPT`).
- `infer.py` used `from model import build_lcf` (works only when cwd==lcf_impl) → failed when imported as a package. Fix: `try: from model import ... except ImportError: from lcf.lcf_impl.model import ...`.
- For baseline evals run_eval does `import baselines.sft` AND `from lcf.lcf_impl.infer` → needs **`PYTHONPATH=<root>:<root>/lcf`** (both on path).

## 8. Qwen3 "thinking" mode
- **Symptom:** Generation ~30 min/problem; outputs full `<think>…` rambling instead of the answer; BIRD gen would've taken ~100h.
- **Fix:** `apply_chat_template(..., enable_thinking=False)`; for completion-style prompts, take the **first non-empty line** as the answer (strip the trailing ramble).

---

## 9. Operational / sparkq-CLI gotchas
- **Poller false-fire:** a transient empty `sparkq status` made a "job done?" poller fire early. Fix: require **2 consecutive empty** checks before declaring done.
- **`--eta` is NOT a hard kill:** jobs run well past their ETA (observed 1h26m on a 40m ETA). It's only a scheduling estimate.
- **GPU contention slowdown:** two generation jobs sharing one GB10 each run ~2× slower (lcf-eval + bird-gen took 1h38m / 1h43m vs ~40m alone).
- **Correct submit syntax:** `sparkq submit "<cmd>" --workdir <abs> --env K=V --node 1 --gpu-mem 45G --cpu-mem 24G --eta 90m --tag <t>` (positional cmd; `--allow-duplicates` to resubmit same cmd). NOT `--name/--gpus/--`.
- **Logs:** `sparkq log <id>` shows only recent output (truncated); the full log is at **`~/.sparkq/logs/<id>.log`** — grep that for errors that scrolled off.
- **History:** `sparkq history` shows TAG/RUNTIME/EXIT; exit `-15` = SIGTERM (cancelled), `1` = crash, `0` = success.

---

## TL;DR checklist before submitting a GPU job
1. Model/dataset fully downloaded from the **login shell** (weights present).
2. Submit with `--env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env NVIDIA_DISABLE_REQUIRE=1 --env WANDB_MODE=offline`.
3. torch jobs → `lcf/lcf_impl` venv (`uv run python <abs path>`), `--workdir lcf/lcf_impl`.
4. ≤ 2 concurrent 7B jobs; verify `torch.cuda.mem_get_info()` free memory recovered after any cancel.
5. Chain dependent steps in one driver script.
6. If a 7B load OOMs in sparkq but the node looks free → **run it directly in the login shell** (nohup) — sparkq's cgroup under-provisions the CPU+GPU transient on unified memory.
