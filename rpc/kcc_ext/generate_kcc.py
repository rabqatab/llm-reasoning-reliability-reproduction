"""Generate K sampled CoT answers for the KCC civil-precedent-relevance task, RPC-format.

A FOURTH domain for the RPC test-time-scaling method (Paper A), after the paper's
math benchmarks, BIRD text-to-SQL, and JurisNet ko_ver legal extraction. The task
is BINARY legal-relevance classification: given a Korean query precedent's
판시사항/판결요지 and a candidate precedent's, decide whether the candidate is a
legally related 선례 (precedent) to the query — 0(무관) / 1(관련).

Each of K sampled completions reasons briefly in Korean, then ends with
'Answer: <0/1>'. We parse the last 0/1. Aggregation (SC/PPL/RPC) is done by
run_kcc.py with trivial integer-equality on the 0/1 label (the run_mcq.py pattern).

Output: kcc_<model>.json with keys
  predict       0/1 int  (-1 if unparseable)
  completion    raw model text
  mean_logprob  mean per-token log-prob (from output_scores)
  answer        gold 0/1 label
shaped [n_pairs][K]. Resumable via a .partial.json checkpoint.

GPU ONLY — do NOT run a 7-8B model on this box (no GPU). See README.md for the
exact login-shell command. CPU here is for editing / dry import checks only.
"""
from __future__ import annotations
import argparse, json, os, re

PROMPT = (
    "다음은 한국 민사 대법원 판례 두 건의 판시사항과 판결요지입니다.\n\n"
    "[질의 판례]\n{query}\n\n"
    "[후보 판례]\n{candidate}\n\n"
    "질문: 이 두 판례는 법적으로 관련된 선례 관계입니까? "
    "즉, 후보 판례가 질의 판례의 쟁점에 대한 법적으로 관련된 선례입니까?\n"
    "간단히 근거를 제시한 뒤, 마지막 줄에 'Answer: <N>' 형식으로 답하세요. "
    "<N>은 0(무관) 또는 1(관련)입니다.\n"
)


def build_prompt(item):
    return PROMPT.format(query=item["query_text"], candidate=item["candidate_text"])


def extract_label(text):
    """Last explicit 'Answer: 0/1', else last standalone 0/1 in the text, else -1."""
    m = re.findall(r"[Aa]nswer\s*[:\-]?\s*\(?([01])\b", text)
    if m:
        return int(m[-1])
    m2 = re.findall(r"\b([01])\b", text)  # fallback: last standalone 0/1
    if m2:
        return int(m2[-1])
    return -1  # unparseable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--data", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "kcc_subset.jsonl"))
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--out_dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    import torch, sys
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from _batched_gen import sample_k

    items = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    if args.n:
        items = items[: args.n]

    tag = args.model.replace("/", "_")
    out_path = os.path.join(args.out_dir, f"kcc_{tag}.json")
    partial = out_path.replace(".json", ".partial.json")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    res = {"predict": [], "completion": [], "mean_logprob": [], "answer": []}
    if os.path.exists(partial):
        res = json.load(open(partial))
    start = len(res["predict"])

    for qi in range(start, len(items)):
        it = items[qi]
        prompt = build_prompt(it)
        try:
            enc = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, return_tensors="pt",
                return_dict=True, enable_thinking=False,
            )
        except TypeError:
            enc = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, return_tensors="pt",
                return_dict=True,
            )
        input_ids = enc["input_ids"].to(model.device)
        comps, lps = sample_k(model, tok, input_ids, args.K, args.max_new_tokens,
                              args.temperature, args.top_p)
        preds = [extract_label(t) for t in comps]
        res["predict"].append(preds)
        res["completion"].append(comps)
        res["mean_logprob"].append(lps)
        res["answer"].append(int(it["label"]))
        if qi % 5 == 0:
            json.dump(res, open(partial, "w"), ensure_ascii=False)
            print(f"[gen] {qi + 1}/{len(items)}", flush=True)

    json.dump(res, open(out_path, "w"), ensure_ascii=False)
    if os.path.exists(partial):
        os.remove(partial)
    print(f"[gen] wrote {out_path}  ({len(res['predict'])} pairs, K={args.K})")


if __name__ == "__main__":
    main()
