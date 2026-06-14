# LCF KCC Legal-domain dataset (precedent-holding variant; LCF generalization test)

A **SECOND** legal-reasoning analogue of the LFUD dataset, built so the LCF
method (Paper B, logic-representation editing) can be evaluated on a second,
**distinct** legal source. The first legal build (`../legal/`, from JurisNet
fact-pattern `context`s) is fact-pattern-based; this build is
**precedent-HOLDING-based**: each scenario is a Korean **KCC** civil Supreme
Court precedent and the premise is its holding (판결요지). Having two
independently-sourced legal datasets lets us check that LCF's gains are not an
artifact of one corpus.

Produced by `kcc_legal_data.py`; emits files matching the **exact LCF data
contract** (`../data/SCHEMA.md`), so `extract_reps.py` / `train.py` /
`run_eval.py` consume it unchanged.

## What the data is

| File | Schema |
|---|---|
| `split_scenarios.json` | `{train:[int…], val:[int…], test:[int…]}` — scenario ids = KCC precedent number (`*_precedentNumber`) |
| `valid_conclusions.jsonl` | cache, resumable: `{scenario_id, premise, valid_conclusion, invalid_conclusion}` |
| `conclusion_gen_{train,val,test}.jsonl` | `{scenario_id, premise, valid_conclusion, invalid_conclusion}` |
| `fallacy_id_{val,test}.jsonl` | `{scenario_id, premise, options:[4 str], answer_idx}` |

- **premise** = a Korean Supreme Court precedent **holding** (판결요지,
  `query_precedentNote`) from `KoCivSCMdataset` (`/home/alphabridge/Research/KCC/dataset/*.json`).
  Holdings longer than 1200 chars are truncated; very short ones (<40 chars) skipped.
- **valid_conclusion** = a conclusion that follows *logically validly* from the
  holding, **keeping the conditions the holding states** (sound legal inference).
- **invalid_conclusion** = a plausible-sounding but *logically fallacious*
  inference from the **same** holding — same legal content, opposite validity
  (target fallacies: over-generalising the holding / **dropping a stated
  condition or exception**, affirming-the-consequent, conflating necessary vs
  sufficient conditions, etc.).
- **fallacy_id** item: 4 options = the valid conclusion + 2 invalid ones (this
  precedent's fallacy + one same-split distractor) + an "I have no comment"
  option (`본 사안만으로는 단정할 수 없다.`). **`answer_idx` points at the VALID
  option** (task = *identify the logically valid legal conclusion*). This is the
  legal mirror of LFUD's "identify the fallacy" MCQ; the eval harness scores
  per-option logprobs and checks the gold index either way.

Scenario = precedent (deduped by precedent number), scenario-split **70:10:20**
(seed 42). For `--n 200` the split is **140 / 20 / 40** (train/val/test).
Premises stay in Korean; conclusions are generated in Korean.

### Source layout note (why query + candidate precedents)

Each `*.json` file in the KCC dataset is a dict keyed by record id, and every
record carries identical `query_*` fields plus a varying `candidate_*` pair —
i.e. **one query precedent per file (only 20 total)**, each compared against many
candidate precedents. Query and candidate precedents are Supreme Court precedents
of the **same structure** (a `precedentNumber` + a `precedentNote` holding). To
reach N=200 unique precedent scenarios, `kcc_legal_data.py` therefore treats both
as precedents and **dedupes by precedent number**: it emits the 20 unique query
precedents first (deterministic, one per file in sorted order), then fills up to
`--n` from the distinct candidate precedent holdings (≈2,583 unique available).
This honors the spec's intent — *each precedent is one scenario, premise = the
holding (Note)* — while still yielding 200 scenarios.

## How it was generated

`kcc_legal_data.py` calls the **OpenAI API** (default `gpt-4o-mini`) once per
holding with a Korean system prompt (see `GEN_SYS` in the script) asking for a
JSON object `{"valid_conclusion": …, "invalid_conclusion": …}` via
`response_format=json_object`. The key is read from `$OPENAI_API_KEY` or the repo
`../../.env`. The call uses only the Python **stdlib (`urllib`)** — the `openai`
package is **not** required.

Resumable: each generated triple is appended to `valid_conclusions.jsonl`;
re-runs skip precedents already cached. `--no-api` builds the splits + (blank)
files with no network. Cost is reported per run (gpt-4o-mini: $0.15/1M in,
$0.60/1M out).

**Observed cost/time at `--n 200`:** ~291 s total for 170 new calls (~1.7 s and
~$0.00014 per premise), ~$0.023 for the full 200-precedent build.

```bash
# dry-run (no API): build splits + blank conclusion/fallacy files
python kcc_legal_data.py --no-api --n 200

# small validation batch with the API (prints 3 example triples + cost/time)
python kcc_legal_data.py --n 30 --model gpt-4o-mini

# full run
python kcc_legal_data.py --n 200 --model gpt-4o-mini
```

## Running the LCF pipeline on this KCC legal data

`run_eval.py` accepts `--data-dir`, so eval reads this dir directly. **NOTE (from
the JurisNet legal build):** `extract_reps.py` and `train.py` currently
**hardcode** `DATA_DIR = lcf/data` (no `--data-dir` flag), so for representation
extraction / training you must **copy this dir's files into `../data` first**
(back up `lcf/data` beforehand if it holds the fallacy or JurisNet-legal data).

```bash
# from lcf/lcf_impl/ — make the hardcoded loader read the KCC legal data
# (copy workaround; reversible — back up lcf/data first if it holds other data)
cp ../kcc_legal/conclusion_gen_train.jsonl ../kcc_legal/conclusion_gen_val.jsonl \
   ../kcc_legal/conclusion_gen_test.jsonl ../kcc_legal/split_scenarios.json ../data/

# 1. extract paired (valid, invalid) reps from the frozen base model  [GPU]
python extract_reps.py --model Qwen/Qwen3-8B --split train
python extract_reps.py --model Qwen/Qwen3-8B --split val

# 2. train the LCF logic-representation editor                        [GPU]
python train.py --reps ../data/reps_Qwen3-8B_train.pt \
                --val-reps ../data/reps_Qwen3-8B_val.pt \
                --model Qwen/Qwen3-8B

# 3. evaluate on the KCC legal test split (reads THIS dir via --data-dir)
cd ../eval
python run_eval.py --model Qwen/Qwen3-8B --variant +LCF \
                   --data-dir ../kcc_legal \
                   --ckpt ../checkpoints/Qwen3-8B/lcf.pt
python run_eval.py --model Qwen/Qwen3-8B --variant original \
                   --data-dir ../kcc_legal --skip-gpt4   # baseline
```

`run_eval.py --dry-run` exercises the full metric wiring on CPU with a stub
backend (no GPU / no model download) and is the quickest way to confirm these
files are well-formed.

## GPU note (sparkq / GB10)

`extract_reps.py` and `train.py` are the GPU steps. Per the project's sparkq
gotchas, **GB10 OOMs** on these; run GPU jobs in the **login shell** (or the
node-2 login shell), not via the sparkq GB10 queue. CPU/API steps
(`kcc_legal_data.py`, `run_eval.py --dry-run`) run anywhere.
