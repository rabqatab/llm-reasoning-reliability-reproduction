# BIRD text-to-SQL extension for RPC

Applies the RPC / Self-Consistency / Perplexity confidence-aggregation methods
(NeurIPS 2025) to the BIRD text-to-SQL task with a local model. The original
RPC evaluators in `../RPC/` are reused **unchanged**; only the answer-equality
function is swapped from sympy math comparison to SQLite **execution-match**
(BIRD-style execution accuracy: order-insensitive equality of result sets).

Nothing under `../RPC/` is modified — this directory only imports/reads from it.

## Files

| File | Purpose |
|------|---------|
| `sql_exec.py` | `run_sql(db_path, sql, timeout)` -> frozenset of row-tuples or None; `exec_match(db_path, a, b)` -> bool. Stdlib `sqlite3` only. |
| `test_sql_exec.py` | CPU test of the above against real BIRD DBs (gold-vs-gold, gold-vs-broken, invalid). |
| `make_equal_funcs.py` | `SQLFuncFactory` / `make_funcs_for_problem` — per-problem `equal_func`/`check_equal` closures backed by `exec_match`, RPC-evaluator compatible. |
| `generate_paths.py` | GPU harness: loads an HF causal LM, samples K SQL candidates per question, computes `mean_logprob`, writes RPC-format JSON + meta sidecar. **Resumable.** |
| `run_bird.py` | CPU evaluator: runs SC/PPL/RPC on the generated JSON, injecting the SQL equal-funcs; writes `results_bird.txt`. |
| `test_run_bird.py` | CPU smoke test of the full eval pipeline on synthetic data. |

## Data format (RPC-compatible)

`bird_<model>.json` — all `[n_problems][K]` except `answer`:
- `predict`: extracted candidate SQL strings
- `completion`: raw model text
- `mean_logprob`: mean over generated tokens of the log-softmax prob of the
  chosen token. PPL uses `np.exp(mean_logprob)` as the path probability.
- `answer`: `[n_problems]` gold SQL strings

`bird_<model>.meta.json` — `[n_problems]` of `{question_id, db_id}` (needed to
locate each problem's SQLite DB for exec-match).

## Run order

### Step 1 — generate reasoning paths (GPU, via sparkq)

Do **not** run a 7B/8B model on a CPU box. Submit to a GPU node with sparkq:

```bash
sparkq submit --name bird-gen --gpus 1 -- \
  bash -lc 'cd /home/alphabridge/Study/reliableAI_final/rpc/RPC && \
    uv run --project . python ../bird_extension/generate_paths.py \
      --model Qwen/Qwen3-8B --n 200 --K 16 --out_dir ../bird_extension'
```

Notes:
- The model resolves from the HF cache (`HF_HOME`); pre-download with
  `huggingface-cli download Qwen/Qwen3-8B` on the GPU node if not cached.
- `transformers` + `torch` are required at generation time (not listed in the
  RPC `requirements.txt`); install them into the GPU env, e.g.
  `uv pip install torch transformers` (or use a transformers-ready env).
- Generation checkpoints to `bird_<model>.partial.json` after every question,
  so a re-submit resumes.
- Optional: `--difficulty simple|moderate|challenging`, `--temperature`,
  `--top_p`, `--max_new_tokens`. See `python generate_paths.py --help`.

Check status / logs:
```bash
sparkq status --all
sparkq log bird-gen
```

### Step 2 — evaluate (CPU only)

`sqlite3` is stdlib, so eval runs anywhere. Use the RPC uv project so the RPC
package's imports (scipy/sympy/fraction/eval/data_processing) resolve:

```bash
cd /home/alphabridge/Study/reliableAI_final/rpc/RPC
uv run --project . python ../bird_extension/run_bird.py \
  --model Qwen/Qwen3-8B --K 16 --repeats 10 --methods SC,PPL,RPC
```

Appends one line per method to `results_bird.txt`:
```
SC  BIRD Qwen/Qwen3-8B 16 {'Accuracy': '...', 'ECE': '...'}
PPL BIRD Qwen/Qwen3-8B 16 {'Accuracy': '...', 'ECE': '...'}
RPC BIRD Qwen/Qwen3-8B 16 {'Accuracy': '...', 'ECE': '...'}
```

## Tests (CPU, run now)

```bash
# SQL execution-match against real BIRD DBs
python3 test_sql_exec.py

# Full SC/PPL/RPC pipeline on synthetic data (needs RPC deps)
cd ../RPC && uv run --project . python ../bird_extension/test_run_bird.py
```

## Notes / gotchas

- **Result normalization**: results are compared as an order-insensitive
  `frozenset` of row tuples (standard BIRD exec-acc). This treats duplicate
  rows as one; queries differing only in row order or duplicate multiplicity
  are considered equal.
- **Encoding**: a few BIRD text columns contain non-UTF-8 bytes;
  `conn.text_factory` decodes with `errors="replace"` so a stray byte does not
  abort an otherwise valid query.
- **Slow gold queries**: 2 of 1534 dev gold queries (question_id 518 and 701)
  do not finish within the 30s timeout and return `None` (treated as a
  non-match). All other 1532 gold queries run and match themselves
  (gold-vs-gold = 100% on the tested samples; 1532/1534 = 99.87% run to
  completion overall). Raise `--timeout` if you need those two.
- **Why we reimplement `solve`**: the RPC `Evaluator.process`/`worker` hardcode
  `numberic_compare` and the math `check_equal`, and carry no problem index, so
  they cannot thread a per-problem DB path. `run_bird.py` copies the ~30-line
  solve loop and installs the correct `SQLFuncFactory.funcs(idx)` closure per
  problem, while calling the unmodified `prep_evaluator` / `sc_evaluator` /
  `wpc_evaluator`. The math equality cache is disabled (`cache_file=None`).
