"""Generate K sampled CoT answers to LFUD fallacy-identification MCQ, RPC-format.

Applies the RPC test-time-scaling method (Paper A) to a logical-reasoning MCQ
(Paper B's LFUD fallacy-identification task) — connecting the two papers. Each
sampled reasoning path ends in a chosen option index; aggregation (SC/PPL/RPC)
is done by run_mcq.py with trivial integer-equality on option indices.

Output: mcq_<model>.json with keys predict (chosen idx), completion, mean_logprob,
answer (correct idx), shaped [n_questions][K].  Run on GPU (login shell / sparkq).
"""
from __future__ import annotations
import argparse, json, os, re

PROMPT = (
    "{premise}\n\nOptions:\n{opts}\n\n"
    "Reason briefly, then on the last line write 'Answer: <N>' where <N> is the "
    "option number (0-{maxi}) that the question asks for."
)


def build_prompt(item):
    opts = "\n".join(f"{i}. {o}" for i, o in enumerate(item["options"]))
    return PROMPT.format(premise=item["premise"], opts=opts, maxi=len(item["options"]) - 1)


def extract_idx(text, n_opts):
    m = re.findall(r"[Aa]nswer\s*[:\-]?\s*\(?([0-9])", text)
    if m:
        v = int(m[-1])
        if 0 <= v < n_opts:
            return v
    m2 = re.findall(r"\b([0-9])\b", text)  # fallback: last standalone digit
    for v in reversed(m2):
        if 0 <= int(v) < n_opts:
            return int(v)
    return -1  # unparseable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--data", default="/home/alphabridge/Study/reliableAI_final/lcf/data/fallacy_id_test.jsonl")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--out_dir", default="/home/alphabridge/Study/reliableAI_final/rpc/lfud_mcq")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    items = [json.loads(l) for l in open(args.data) if l.strip()]
    if args.n:
        items = items[: args.n]
    tag = args.model.replace("/", "_")
    out_path = os.path.join(args.out_dir, f"mcq_{tag}.json")
    partial = out_path.replace(".json", ".partial.json")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 device_map="auto").eval()

    res = {"predict": [], "completion": [], "mean_logprob": [], "answer": []}
    if os.path.exists(partial):
        res = json.load(open(partial))
    start = len(res["predict"])

    for qi in range(start, len(items)):
        it = items[qi]
        prompt = build_prompt(it)
        try:
            enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                          add_generation_prompt=True, return_tensors="pt",
                                          return_dict=True, enable_thinking=False)
        except TypeError:
            enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                          add_generation_prompt=True, return_tensors="pt",
                                          return_dict=True)
        input_ids = enc["input_ids"].to(model.device)
        plen = input_ids.shape[1]
        preds, comps, lps = [], [], []
        for _ in range(args.K):
            with torch.no_grad():
                out = model.generate(input_ids, do_sample=True, temperature=args.temperature,
                                     top_p=args.top_p, max_new_tokens=args.max_new_tokens,
                                     return_dict_in_generate=True, output_scores=True,
                                     pad_token_id=tok.pad_token_id)
            gen = out.sequences[0][plen:]
            logps = []
            for step, logit in enumerate(out.scores):
                if step >= gen.shape[0]:
                    break
                logps.append(torch.log_softmax(logit[0].float(), -1)[gen[step]].item())
            text = tok.decode(gen, skip_special_tokens=True)
            preds.append(extract_idx(text, len(it["options"])))
            comps.append(text)
            lps.append(sum(logps) / len(logps) if logps else float("-inf"))
        res["predict"].append(preds)
        res["completion"].append(comps)
        res["mean_logprob"].append(lps)
        res["answer"].append(it["answer_idx"])
        if qi % 10 == 0:
            json.dump(res, open(partial, "w"))
            print(f"[gen] {qi + 1}/{len(items)}", flush=True)

    json.dump(res, open(out_path, "w"))
    print(f"[gen] wrote {out_path}  ({len(res['predict'])} questions, K={args.K})")


if __name__ == "__main__":
    main()
