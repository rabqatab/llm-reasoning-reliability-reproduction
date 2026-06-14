"""t-SNE of the content vs logic spaces (paper Figure 3a).

Loads a trained LCF + the validation reps, projects valid/invalid samples into
content and logic spaces, and plots both 2-D embeddings side by side. The logic
space should show a clear valid/invalid boundary; the content space should not.

Usage: python src/analysis_tsne.py --ckpt _scratch/checkpoints/lcf_full.pt
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CFG
from inference import load_lcf, DEV


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CFG.ckpt_dir / "lcf_full.pt"))
    ap.add_argument("--max-per-class", type=int, default=600)
    args = ap.parse_args()
    import numpy as np
    from sklearn.manifold import TSNE
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lcf, ck = load_lcf(args.ckpt)
    val = torch.load(CFG.hidden_dir / "val_reps.pt")
    # pool all taps' val reps
    valid = torch.cat([d["valid"] for d in val.values()])[: args.max_per_class]
    invalid = torch.cat([d["invalid"] for d in val.values()])[: args.max_per_class]

    with torch.no_grad():
        cv, lv = lcf.project(valid.float().to(DEV))
        ci, li = lcf.project(invalid.float().to(DEV))
    content = torch.cat([cv, ci]).cpu().numpy()
    logic = torch.cat([lv, li]).cpu().numpy()
    y = np.array([1] * len(cv) + [0] * len(ci))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, X, title in [(axes[0], content, "Content space"), (axes[1], logic, "Logic space")]:
        emb = TSNE(n_components=2, perplexity=30, init="pca",
                   random_state=CFG.seed).fit_transform(X)
        ax.scatter(emb[y == 1, 0], emb[y == 1, 1], s=8, alpha=.6, label="valid", c="tab:blue")
        ax.scatter(emb[y == 0, 0], emb[y == 0, 1], s=8, alpha=.6, label="invalid", c="tab:red")
        ax.set_title(title); ax.legend(); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("LCF disentanglement: valid vs invalid in content / logic spaces")
    fig.tight_layout()
    out = CFG.results_dir / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "tsne_content_logic.png", dpi=150)
    print(f"saved -> {out / 'tsne_content_logic.png'}")


if __name__ == "__main__":
    main()
