"""Generation harness for the BIRD text-to-SQL RPC extension.

Loads a local HF causal LM, samples K candidate SQL queries per BIRD dev
question, computes the per-path mean token log-probability, extracts the SQL,
and writes an RPC-format JSON plus a sidecar meta file.

Output JSON (``bird_<model>.json``) keys, all shaped [n_problems][K]:
    predict       : extracted SQL strings
    completion    : raw generated text
    mean_logprob  : mean over generated tokens of log-softmax prob of the
                    chosen token  (PPL uses np.exp(mean_logprob) as the path
                    probability, exactly like the math datasets)
    answer        : list[n_problems] of gold SQL strings

Sidecar (``bird_<model>.meta.json``): list[n_problems] of
    {"question_id": int, "db_id": str}.

This script is GPU-bound. DO NOT run it on a CPU-only box; submit it to a GPU
node (see README.md / sparkq). It is resumable: completed problems are
checkpointed to ``bird_<model>.partial.json`` after each question and reloaded
on restart.

Example:
    python generate_paths.py --model Qwen/Qwen3-8B --n 200 --K 16 \
        --difficulty simple --out_dir .
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List, Optional

BIRD_ROOT_DEFAULT = "/mnt/nfs/ssd2/bird_data/dev_20240627"


# --------------------------------------------------------------------------- #
# Schema serialization
# --------------------------------------------------------------------------- #
def build_schema_strings(tables_json_path: str) -> Dict[str, str]:
    """Return {db_id: schema_text} from dev_tables.json.

    The schema text is a compact CREATE-TABLE-style listing of each table and
    its columns (original names), suitable for prompting.
    """
    with open(tables_json_path, "r", encoding="utf-8") as f:
        tables = json.load(f)

    schemas: Dict[str, str] = {}
    for db in tables:
        db_id = db["db_id"]
        table_names = db["table_names_original"]
        # column_names_original entries are [table_index, column_name].
        cols_by_table: List[List[str]] = [[] for _ in table_names]
        types = db.get("column_types", [])
        for ci, (tidx, cname) in enumerate(db["column_names_original"]):
            if tidx < 0:
                continue  # the '*' pseudo-column
            ctype = types[ci] if ci < len(types) else ""
            cols_by_table[tidx].append(f"`{cname}` {ctype}".strip())

        lines: List[str] = []
        for tidx, tname in enumerate(table_names):
            cols = ",\n  ".join(cols_by_table[tidx])
            lines.append(f"CREATE TABLE `{tname}` (\n  {cols}\n);")
        schemas[db_id] = "\n\n".join(lines)
    return schemas


def build_prompt(schema_text: str, question: str, evidence: str) -> str:
    """Build the instruction prompt for one BIRD question."""
    ev = f"\nExternal knowledge / evidence: {evidence}" if evidence else ""
    return (
        "You are an expert data analyst. Given a SQLite database schema and a "
        "question, write a single SQLite query that answers it.\n\n"
        "Database schema:\n"
        f"{schema_text}\n\n"
        f"Question: {question}{ev}\n\n"
        "Respond with exactly one SQLite query inside a ```sql code block.\n"
    )


# --------------------------------------------------------------------------- #
# SQL extraction
# --------------------------------------------------------------------------- #
_SQL_BLOCK = re.compile(r"```sql\s*(.*?)```", re.IGNORECASE | re.DOTALL)
# Fallback: a SELECT statement, or a WITH ... AS ( CTE that leads into a query.
_SELECT = re.compile(r"\bSELECT\b.*|\bWITH\b\s+\w+.*?\bAS\b\s*\(.*",
                     re.IGNORECASE | re.DOTALL)


def extract_sql(text: str) -> str:
    """Extract a single SQL query from model output.

    First tries the ```sql fenced block; falls back to the first SELECT/WITH
    statement found. Returns a stripped, semicolon-trimmed single string (empty
    if nothing SQL-like was produced).
    """
    m = _SQL_BLOCK.search(text)
    candidate = m.group(1) if m else None
    if candidate is None:
        m2 = _SELECT.search(text)
        candidate = m2.group(0) if m2 else ""
    candidate = candidate.strip()
    # Cut at the first statement terminator if multiple statements slipped in.
    if ";" in candidate:
        candidate = candidate.split(";")[0]
    # Collapse internal whitespace/newlines into single spaces for stable
    # exec-match keys (SQLite ignores the difference).
    candidate = " ".join(candidate.split())
    return candidate


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def load_dev(bird_root: str, n: int, difficulty: Optional[str]) -> List[dict]:
    with open(os.path.join(bird_root, "dev.json"), "r", encoding="utf-8") as f:
        dev = json.load(f)
    if difficulty:
        dev = [it for it in dev if it["difficulty"] == difficulty]
    return dev[:n]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sample K SQL candidates per BIRD dev question with a local "
        "HF causal LM and write RPC-format JSON (predict/completion/"
        "mean_logprob/answer) plus a meta sidecar. GPU required.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-8B",
                    help="HF model id or local path (resolved from HF_HOME cache).")
    ap.add_argument("--n", type=int, default=200, help="Number of dev questions.")
    ap.add_argument("--K", type=int, default=16, help="Samples per question.")
    ap.add_argument("--difficulty", type=str, default=None,
                    choices=["simple", "moderate", "challenging"],
                    help="Optional difficulty filter.")
    ap.add_argument("--bird_root", type=str, default=BIRD_ROOT_DEFAULT,
                    help="BIRD dev_20240627 directory.")
    ap.add_argument("--out_dir", type=str, default=".", help="Output directory.")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Heavy imports kept inside main so --help works on a CPU box without torch.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(args.out_dir, exist_ok=True)
    model_tag = args.model.replace("/", "_")
    out_json = os.path.join(args.out_dir, f"bird_{model_tag}.json")
    out_meta = os.path.join(args.out_dir, f"bird_{model_tag}.meta.json")
    partial = os.path.join(args.out_dir, f"bird_{model_tag}.partial.json")

    dev = load_dev(args.bird_root, args.n, args.difficulty)
    schemas = build_schema_strings(os.path.join(args.bird_root, "dev_tables.json"))

    torch.manual_seed(args.seed)
    print(f"Loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    # Resume from checkpoint if present.
    predict: List[List[str]] = []
    completion: List[List[str]] = []
    mean_logprob: List[List[float]] = []
    answer: List[str] = []
    meta: List[Dict] = []
    start_idx = 0
    if os.path.exists(partial):
        with open(partial, "r", encoding="utf-8") as f:
            ckpt = json.load(f)
        predict = ckpt["predict"]
        completion = ckpt["completion"]
        mean_logprob = ckpt["mean_logprob"]
        answer = ckpt["answer"]
        meta = ckpt["meta"]
        start_idx = len(predict)
        print(f"Resuming from checkpoint: {start_idx} problems already done.")

    for idx in range(start_idx, len(dev)):
        it = dev[idx]
        schema_text = schemas[it["db_id"]]
        prompt = build_prompt(schema_text, it["question"], it.get("evidence", ""))

        # Use chat template when available.
        if tok.chat_template:
            try:  # Qwen3 etc.: disable chain-of-thought so it emits SQL directly
                enc = tok.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True, return_tensors="pt", return_dict=True,
                    enable_thinking=False,
                )
            except TypeError:  # tokenizers without the enable_thinking kwarg
                enc = tok.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True, return_tensors="pt", return_dict=True,
                )
            input_ids = enc["input_ids"].to(model.device)
        else:
            input_ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
        prompt_len = input_ids.shape[1]

        p_preds, p_comps, p_lps = [], [], []
        for _ in range(args.K):
            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=tok.pad_token_id,
                )
            seq = out.sequences[0]
            gen_ids = seq[prompt_len:]
            # scores: tuple(len gen) of [1, vocab] logits for each step.
            logps = []
            for step, logits in enumerate(out.scores):
                if step >= gen_ids.shape[0]:
                    break
                logp = torch.log_softmax(logits[0].float(), dim=-1)
                tok_id = gen_ids[step]
                logps.append(logp[tok_id].item())
            mlp = float(sum(logps) / len(logps)) if logps else float("-inf")
            text = tok.decode(gen_ids, skip_special_tokens=True)
            p_preds.append(extract_sql(text))
            p_comps.append(text)
            p_lps.append(mlp)

        predict.append(p_preds)
        completion.append(p_comps)
        mean_logprob.append(p_lps)
        answer.append(it["SQL"])
        meta.append({"question_id": it["question_id"], "db_id": it["db_id"]})

        # Checkpoint after each problem.
        with open(partial, "w", encoding="utf-8") as f:
            json.dump({"predict": predict, "completion": completion,
                       "mean_logprob": mean_logprob, "answer": answer,
                       "meta": meta}, f)
        print(f"[{idx + 1}/{len(dev)}] qid={it['question_id']} db={it['db_id']} done")

    # Final outputs.
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"predict": predict, "completion": completion,
                   "mean_logprob": mean_logprob, "answer": answer}, f)
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {out_json} and {out_meta} ({len(predict)} problems, K={args.K}).")


if __name__ == "__main__":
    main()
