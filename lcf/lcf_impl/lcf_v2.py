"""Model-agnostic LCF v2 — fit the logic-validity direction.

Diagnosis (docs/LCF_critical_analysis.md): the valid/invalid signal is real but
LAYER-LOCALIZED (~0.82 at the best single sub-layer, ~0.52 when pooled). The
original LCF mixes layers + uses a weak centroid V, diluting it. v2 fixes this
in a model-agnostic way:
  1. pick the SINGLE best sub-layer by held-out probe accuracy (per model),
  2. direction = supervised logistic weight vector at that layer (held-out validated),
  3. inference shifts h <- h + alpha*||h||*w_hat  (norm-RELATIVE => scale-free across models),
  4. optionally gate to reps the probe flags invalid.

This `fit` step is reps-only (no 7B load). Saves a tiny direction checkpoint.
"""
from __future__ import annotations
import argparse, json, torch, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DATA = "/home/alphabridge/Study/reliableAI_final/lcf/data"
OUT = "/home/alphabridge/Study/reliableAI_final/lcf/checkpoints"


def load(name, split):
    d = torch.load(f"{DATA}/reps_{name}_{split}.pt", weights_only=False)
    return (d["R_plus"].float().numpy(), d["R_minus"].float().numpy(),
            d["layer"].numpy(), d["kind"].numpy())


def fit(model_short):
    Rp, Rm, La, Ka = load(model_short, "train")
    Vp, Vm, Lv, Kv = load(model_short, "val")
    best = None
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
            if best is None or acc > best["acc"]:
                # direction in raw hidden space: probe weight de-standardized
                w = clf.coef_[0] / sc.scale_           # back to raw space
                w = w / (np.linalg.norm(w) + 1e-8)
                hnorm = float(np.linalg.norm(Xtr, axis=1).mean())
                best = {"acc": float(acc), "layer": int(L),
                        "kind": "attn" if k == 0 else "mlp",
                        "w": w.astype(np.float32), "h_norm": hnorm}
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="short name, e.g. Qwen3-8B")
    args = ap.parse_args()
    b = fit(args.model)
    import os
    os.makedirs(f"{OUT}/{args.model}", exist_ok=True)
    path = f"{OUT}/{args.model}/lcf_v2_direction.pt"
    torch.save({"layer": b["layer"], "kind": b["kind"],
                "w": torch.tensor(b["w"]), "h_norm": b["h_norm"],
                "probe_val_acc": b["acc"]}, path)
    print(f"[v2] {args.model}: best L{b['layer']} {b['kind']} "
          f"held-out acc={b['acc']:.3f} |h|={b['h_norm']:.1f} -> {path}")


if __name__ == "__main__":
    main()
