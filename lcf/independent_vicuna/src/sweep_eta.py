"""Sweep the inference steering magnitude eta on the identification task,
loading the model once. Baseline computed once; LCF per eta."""
import sys, json
from pathlib import Path
import torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CFG
from model_utils import load_llm
from inference import load_lcf, attach
from eval_identification import option_logprob

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 120
model, tok = load_llm(CFG)
lcf, ck = load_lcf(CFG.ckpt_dir / "lcf_full.pt")
steerer = attach(model, lcf, ck["top_taps"], eta=0.5)

items = [json.loads(l) for l in open(CFG.data_dir / "identification.jsonl")][:LIMIT]
valid_map = json.load(open(CFG.data_dir / "valid_conclusions.json"))
rng = torch.Generator().manual_seed(CFG.seed)
prepared = []
for it in items:
    v = valid_map.get(str(it["index"]))
    if not v:
        continue
    opts = [v, it["invalid_A"], it["invalid_B"], it["no_comment"]]
    labels = [1, 0, 0, 0]
    perm = torch.randperm(4, generator=rng).tolist()
    opts = [opts[i] for i in perm]; labels = [labels[i] for i in perm]
    prefix = f"Premise: {it['premise']}\nSelect the logically valid conclusion.\nConclusion:"
    prepared.append((prefix, opts, labels.index(1)))


def run(active, eta):
    steerer.active = active; steerer.eta = eta
    acc = dp = 0
    for prefix, opts, ans in prepared:
        sc = torch.tensor([option_logprob(model, tok, prefix, " " + o) for o in opts])
        p = F.softmax(sc, 0)
        acc += int(int(sc.argmax()) == ans)
        dp += (p[ans] - (p.sum() - p[ans]) / 3).item()
    n = len(prepared)
    return acc / n * 100, dp / n


ba, bdp = run(False, 0)
print(f"n={len(prepared)}  BASELINE  acc={ba:.1f}  dprob={bdp:.3f}")
for e in [0.25, 0.5, 1.0, 2.0, 4.5, 8.0]:
    a, d = run(True, e)
    print(f"eta={e:5}  LCF  acc={a:.1f}  dprob={d:.3f}   (dAcc {a-ba:+.1f})")
