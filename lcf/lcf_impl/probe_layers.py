"""Per-sublayer supervised probe: where (and how strongly) is the valid/invalid
signal present, train->val held-out? Backbone for the falsification analysis and
the model-agnostic LCF v2 (pick the single best sub-layer + supervised direction).

Outputs, per model: held-out accuracy of a logistic probe at each (layer, kind),
the best sub-layer, and per-layer hidden-state norms (for norm-relative scaling).
"""
from __future__ import annotations
import argparse, torch, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DATA = "/home/alphabridge/Study/reliableAI_final/lcf/data"


def load(name, split):
    d = torch.load(f"{DATA}/reps_{name}_{split}.pt", weights_only=False)
    return (d["R_plus"].float().numpy(), d["R_minus"].float().numpy(),
            d["layer"].numpy(), d["kind"].numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-8B")
    args = ap.parse_args()
    Rp, Rm, La, Ka = load(args.model, "train")
    Vp, Vm, Lv, Kv = load(args.model, "val")
    print(f"== {args.model}  train={len(Rp)} val={len(Vp)} ==")
    print(f"{'layer':>5} {'kind':>4} {'n_tr':>5} {'n_val':>5} | {'val-acc':>7} | {'|h|':>7}")
    results = []
    for L in sorted(set(La)):
        for k in (0, 1):
            mtr = (La == L) & (Ka == k); mv = (Lv == L) & (Kv == k)
            if mtr.sum() < 60 or mv.sum() < 20:
                continue
            Xtr = np.r_[Rp[mtr], Rm[mtr]]; ytr = np.r_[np.ones(mtr.sum()), np.zeros(mtr.sum())]
            Xv = np.r_[Vp[mv], Vm[mv]]; yv = np.r_[np.ones(mv.sum()), np.zeros(mv.sum())]
            sc = StandardScaler().fit(Xtr)
            clf = LogisticRegression(max_iter=500, C=0.5).fit(sc.transform(Xtr), ytr)
            acc = clf.score(sc.transform(Xv), yv)
            hnorm = float(np.linalg.norm(np.r_[Rp[mtr], Rm[mtr]], axis=1).mean())
            results.append((acc, int(L), 'attn' if k == 0 else 'mlp', int(mtr.sum()), hnorm))
    results.sort(reverse=True)
    for acc, L, kind, ntr, hn in results:
        print(f"{L:>5} {kind:>4} {ntr:>5} {'':>5} | {acc:>7.3f} | {hn:>7.1f}")
    best = results[0]
    print(f"\nBEST sub-layer: L{best[1]} {best[2]}  held-out acc={best[0]:.3f}  |h|={best[4]:.1f}")
    print(f"mean held-out acc over sub-layers: {np.mean([r[0] for r in results]):.3f}  "
          f"(n_sublayers={len(results)})")


if __name__ == "__main__":
    main()
