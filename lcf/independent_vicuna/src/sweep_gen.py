"""Generation Valid% sweep with an Anthropic judge (paper's headline metric).
Loads the model once; baseline generated+judged once; LCF generated+judged per eta.

Usage: python src/sweep_gen.py [limit] [judge_model]
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CFG
from model_utils import load_llm
from inference import load_lcf, attach
from eval_generation import generate, perplexity, make_judge, make_local_judge

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 80
JUDGE_MODEL = sys.argv[2] if len(sys.argv) > 2 else "claude-sonnet-4-6"  # or "local"
ETAS = [0.5, 2.0, 4.5]

model, tok = load_llm(CFG)
lcf, ck = load_lcf(CFG.ckpt_dir / "lcf_full.pt")
steerer = attach(model, lcf, ck["top_taps"], eta=0.5)
judge = make_local_judge(model, tok) if JUDGE_MODEL == "local" else make_judge("anthropic", JUDGE_MODEL)
items = [json.loads(l) for l in open(CFG.data_dir / "pairs_test.jsonl")][:LIMIT]


def gen_all():
    return [(it["premise"], generate(model, tok, it["premise"])) for it in items]


def score(pairs):
    steerer.active = False  # perplexity under the unmodified base model
    ppl = [perplexity(model, tok, p, g) for p, g in pairs]
    ppl = [x for x in ppl if x == x]  # drop nan
    valid = sum(judge(p, g) for p, g in pairs) / len(pairs) * 100
    return valid, (sum(ppl) / len(ppl) if ppl else float("nan"))


out = {"n": len(items), "judge": JUDGE_MODEL}
steerer.active = False
base_pairs = gen_all()
bvalid, bppl = score(base_pairs)
out["baseline"] = {"valid_pct": bvalid, "ppl": bppl}
print(f"BASELINE  valid%={bvalid:.1f}  ppl={bppl:.2f}")

out["lcf"] = {}
for e in ETAS:
    steerer.active = True; steerer.eta = e
    pairs = gen_all()
    v, p = score(pairs)
    out["lcf"][str(e)] = {"valid_pct": v, "ppl": p,
                          "samples": [{"premise": pr, "base": bg, "lcf": lg}
                                      for (pr, bg), (_, lg) in zip(base_pairs, pairs)][:8]}
    print(f"eta={e:4}  LCF valid%={v:.1f}  ppl={p:.2f}   (dValid {v-bvalid:+.1f})")

json.dump(out, open(CFG.results_dir / "gen_valid_sweep.json", "w"), ensure_ascii=False, indent=2)
print(f"saved -> {CFG.results_dir/'gen_valid_sweep.json'}")
