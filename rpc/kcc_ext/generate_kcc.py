"""Generate K sampled CoT answers for the KCC civil-precedent-relevance task, RPC-format.

A FOURTH domain for the RPC test-time-scaling method (Paper A), after the paper's
math benchmarks, BIRD text-to-SQL, and JurisNet ko_ver legal extraction. The task
is the KCC **GRADED relevance** classification (labels 0-3, 3 = highly relevant):
given a Korean query precedent's 판시사항/판결요지 and a candidate's, judge the
relevance grade 0(무관)/1(약한)/2(상당)/3(매우 밀접). (An earlier version wrongly
binarized this to 0/1; see build_subset.py.)

Each of K sampled completions reasons briefly in Korean, then ends with
'Answer: <0-3>'. We parse the leading 0-3. Aggregation (SC/PPL/RPC) is done by
run_kcc.py with trivial integer-equality on the 0-3 label (the run_mcq.py pattern).

Output: kcc_<model>.json with keys
  predict       0-3 int  (-1 if unparseable)
  completion    raw model text
  mean_logprob  mean per-token log-prob (from output_scores)
  answer        gold 0-3 label
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
    "질문: 후보 판례가 질의 판례에 대한 선례로서 갖는 관련성 등급을 0~3으로 판정하세요.\n"
    "(0=무관, 1=약한 관련, 2=상당히 관련, 3=매우 밀접한 선례)\n"
    "반드시 첫 줄에 'Answer: 0' / 'Answer: 1' / 'Answer: 2' / 'Answer: 3' 중 하나만 쓰고, "
    "그 다음 줄에 한 문장으로 근거를 쓰세요.\n"
)


def build_prompt(item):
    return PROMPT.format(query=item["query_text"], candidate=item["candidate_text"])


def extract_label(text):
    """First explicit 'Answer: 0-3', else last standalone 0-3 in the text, else -1."""
    m = re.findall(r"[Aa]nswer\s*[:\-]?\s*\(?([0-3])\b", text)
    if m:
        return int(m[0])   # answer-first: take the leading 'Answer: N'
    m2 = re.findall(r"\b([0-3])\b", text)  # fallback: last standalone 0-3
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
