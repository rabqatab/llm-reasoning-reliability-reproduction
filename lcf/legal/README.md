# LCF Legal-domain dataset (generalization test of LCF beyond fallacies)

A **legal-reasoning** analogue of the LFUD dataset, built so the LCF method
(Paper B, logic-representation editing) can be evaluated on a **second domain**
(Korean legal reasoning) in addition to logical fallacies. It is produced by
`legal_data.py` and emits files matching the **exact LCF data contract**
(`../data/SCHEMA.md`), so the existing `extract_reps.py` / `train.py` /
`run_eval.py` consume it unchanged.

## What the data is

| File | Schema |
|---|---|
| `split_scenarios.json` | `{train:[int…], val:[int…], test:[int…]}` — scenario ids = JurisNet `prec_seq` (a unique case id) |
| `valid_conclusions.jsonl` | cache, resumable: `{scenario_id, premise, valid_conclusion, invalid_conclusion}` |
| `conclusion_gen_{train,val,test}.jsonl` | `{scenario_id, premise, valid_conclusion, invalid_conclusion}` |
| `fallacy_id_{val,test}.jsonl` | `{scenario_id, premise, options:[4 str], answer_idx}` |

- **premise** = a Korean legal fact-pattern / statute `context` from
  `JurisNet-ko` (`data/benchmark/ko_ver/test.jsonl`, field `context`). Long
  contexts are truncated to 1200 chars; very short ones (<60 chars) skipped.
- **valid_conclusion** = a conclusion that follows *logically validly* from the
  premise (sound legal inference).
- **invalid_conclusion** = a plausible-sounding but *logically fallacious*
  inference from the **same** premise — same legal content, opposite validity
  (target fallacies: over-generalising a holding, affirming-the-consequent on
  statutory conditions, conflating necessary vs sufficient conditions, ignoring
  a stated exception, etc.).
- **fallacy_id** item: 4 options = the valid conclusion + 2 invalid ones (this
  case's fallacy + one same-split distractor) + an "I have no comment" option
  (`본 사안만으로는 단정할 수 없다.`). **`answer_idx` points at the VALID
  option** (task = *identify the logically valid legal conclusion*). This is the
  legal mirror of LFUD's "identify the fallacy" MCQ; the eval harness scores
  per-option logprobs and checks the gold index either way.

Scenario = case (`prec_seq`). A single case spans several `context` segments in
the source; we keep the **first segment per case** so each scenario is one case,
then scenario-split **70:10:20** (seed 42) into train/val/test. Premises stay in
their source language (Korean); conclusions are generated in Korean.

## How it was generated

`legal_data.py` calls the **OpenAI API** (default `gpt-4o-mini`) once per
premise with a Korean system prompt (see `GEN_SYS` in the script) asking for a
JSON object `{"valid_conclusion": …, "invalid_conclusion": …}` via
`response_format=json_object`. The key is read from `$OPENAI_API_KEY` or the
repo `../../.env` (same as `lfud_data.py`). The call uses only the Python
**stdlib (`urllib`)** — the `openai` package is **not** required.

Resumable: each generated triple is appended to `valid_conclusions.jsonl`; re-runs
skip cases already cached. `--no-api` builds the splits + (blank) files with no
network. Cost is reported per run (gpt-4o-mini: $0.15/1M in, $0.60/1M out).

```bash
# dry-run (no API): build splits + blank conclusion/fallacy files
python legal_data.py --no-api --n 200

# small validation batch with the API (prints 3 example triples + cost/time)
python legal_data.py --n 30 --model gpt-4o-mini

# full run
python legal_data.py --n 200 --model gpt-4o-mini
```

## Running the LCF pipeline on this legal data

`run_eval.py` accepts `--data-dir`, so eval reads this dir directly. NOTE:
`extract_reps.py` and `train.py` currently **hardcode** `DATA_DIR = lcf/data`
(no `--data-dir` flag), so for representation extraction either (a) point them at
this dir with a symlink, or (b) copy the four `conclusion_gen_*`/`split` files in.
Recommended (non-destructive): back up `lcf/data`, symlink the legal files, run,
then restore. Or simply copy this dir's contents into `lcf/data` for the legal run.

```bash
# from lcf/lcf_impl/ — make the hardcoded loader read the legal data
# (option b: copy; reversible — back up lcf/data first if it holds fallacy data)
cp ../legal/conclusion_gen_train.jsonl ../legal/conclusion_gen_val.jsonl \
   ../legal/conclusion_gen_test.jsonl ../legal/split_scenarios.json ../data/

# 1. extract paired (valid, invalid) reps from the frozen base model  [GPU]
python extract_reps.py --model Qwen/Qwen3-8B --split train
python extract_reps.py --model Qwen/Qwen3-8B --split val

# 2. train the LCF logic-representation editor                        [GPU]
python train.py --reps ../data/reps_Qwen3-8B_train.pt \
                --val-reps ../data/reps_Qwen3-8B_val.pt \
                --model Qwen/Qwen3-8B

# 3. evaluate on the legal test split (reads THIS dir via --data-dir)
cd ../eval
python run_eval.py --model Qwen/Qwen3-8B --variant +LCF \
                   --data-dir ../legal \
                   --ckpt ../checkpoints/Qwen3-8B/lcf.pt
python run_eval.py --model Qwen/Qwen3-8B --variant original \
                   --data-dir ../legal --skip-gpt4   # baseline
```

`run_eval.py --dry-run` exercises the full metric wiring on CPU with a stub
backend (no GPU / no model download) and is the quickest way to confirm the
legal files are well-formed.

## GPU note (sparkq / GB10)

`extract_reps.py` and `train.py` are the GPU steps. Per the project's sparkq
gotchas, **GB10 OOMs** on these; run GPU jobs in the **login shell** (or the
node-2 login shell), not via the sparkq GB10 queue. CPU/API steps (`legal_data.py`,
`run_eval.py --dry-run`) run anywhere.
