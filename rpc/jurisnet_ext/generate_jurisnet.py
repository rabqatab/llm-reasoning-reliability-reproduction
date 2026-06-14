"""Generate K sampled answers for the JurisNet ko_ver statute-version task, RPC-format.

Applies the RPC test-time-scaling method (Paper A) to a THIRD domain — Korean
legal statute-version extraction (JurisNet ko_ver) — alongside math (paper) and
BIRD text-to-SQL. Each sampled completion is the model's Korean answer listing
the applicable law_name + 제N조; we canonicalize it (normalize.canon_from_text)
to a frozenset of (law, article) pairs. Aggregation (SC/PPL/RPC) is done by
run_jurisnet.py with normalize.answer_match as the equality function.

Output: jurisnet_<model>.json with keys
  predict       canonical answer string  (normalize.canon_str)
  completion    raw model text
  mean_logprob  mean per-token log-prob (from output_scores)
  answer        gold canonical answer string
shaped [n_cases][K].  Resumable via a .partial.json checkpoint.

GPU ONLY — do NOT run a 7-8B model on this box (no GPU). See README.md for the
exact login-shell command. CPU here is for editing / dry import checks only.
"""
from __future__ import annotations
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import canon_from_extractions, canon_from_text, canon_str  # noqa: E402

PROMPT = (
    "다음은 한국 판례의 본문입니다. 이 판례에 적용된 법령과 조문을 추출하세요.\n\n"
    "판례 본문:\n{context}\n\n"
    "지시사항:\n"
    "- 적용 법령명과 조문을 '<법령명> 제N조' 형식으로 나열하세요.\n"
    "- 법령명에 '시행령', '시행규칙', '부칙' 등이 포함되면 그대로 적으세요.\n"
    "- 조문이 '제N조의M' 형태이면 그대로 적으세요.\n"
    "- 항/호는 무시하고 (법령명, 조문)만 적으세요.\n"
    "- 한 줄에 하나씩, 마지막에 'Answer:' 다음 줄부터 목록만 출력하세요.\n\n"
    "예시 출력:\n"
    "Answer:\n민법 제103조\n형법 제250조의2\n"
)


def build_prompt(item):
    return PROMPT.format(context=item["context"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument(
        "--data",
        default="/home/alphabridge/Research/JurisNet-ko/data/benchmark/ko_ver/test.jsonl",
    )
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--max_ctx_chars", type=int, default=4000,
                    help="truncate very long Korean case texts to bound prompt length")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--out_dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    items = []
    with open(args.data) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    if args.n:
        items = items[: args.n]
    # bound context length (token budget) by char-truncation
    for it in items:
        if len(it["context"]) > args.max_ctx_chars:
            it["context"] = it["context"][: args.max_ctx_chars]

    tag = args.model.replace("/", "_")
    out_path = os.path.join(args.out_dir, f"jurisnet_{tag}.json")
    partial = out_path.replace(".json", ".partial.json")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()

    res = {"predict": [], "completion": [], "mean_logprob": [], "answer": []}
    if os.path.exists(partial):
        res = json.load(open(partial))
    start = len(res["predict"])

    for qi in range(start, len(items)):
        it = items[qi]
        gold = canon_str(canon_from_extractions(it["extractions"]))
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
        plen = input_ids.shape[1]
        preds, comps, lps = [], [], []
        for _ in range(args.K):
            with torch.no_grad():
                out = model.generate(
                    input_ids, do_sample=True, temperature=args.temperature,
                    top_p=args.top_p, max_new_tokens=args.max_new_tokens,
                    return_dict_in_generate=True, output_scores=True,
                    pad_token_id=tok.pad_token_id,
                )
            gen = out.sequences[0][plen:]
            logps = []
            for step, logit in enumerate(out.scores):
                if step >= gen.shape[0]:
                    break
                logps.append(torch.log_softmax(logit[0].float(), -1)[gen[step]].item())
            text = tok.decode(gen, skip_special_tokens=True)
            preds.append(canon_str(canon_from_text(text)))
            comps.append(text)
            lps.append(sum(logps) / len(logps) if logps else float("-inf"))
        res["predict"].append(preds)
        res["completion"].append(comps)
        res["mean_logprob"].append(lps)
        res["answer"].append(gold)
        if qi % 5 == 0:
            json.dump(res, open(partial, "w"), ensure_ascii=False)
            print(f"[gen] {qi + 1}/{len(items)}", flush=True)

    json.dump(res, open(out_path, "w"), ensure_ascii=False)
    if os.path.exists(partial):
        os.remove(partial)
    print(f"[gen] wrote {out_path}  ({len(res['predict'])} cases, K={args.K})")


if __name__ == "__main__":
    main()
