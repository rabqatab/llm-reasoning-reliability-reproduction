"""Contrastive hyperparameter sweep for LCF (reps-only, no 7B load).

For each config, trains the LCF on pre-extracted reps and reports the final
train InfoNCE and the HELD-OUT (val reps) separability of the learned logic
projection — i.e. does the projector learn a generalizable valid/invalid
direction? Diagnoses whether the paper's contrastive objective is just
under-fit (fixable by stronger training) or fundamentally signal-limited.

Usage:  uv run python contrastive_sweep.py --model Qwen3-8B
"""
from __future__ import annotations
import argparse, itertools, torch, numpy as np
import torch.nn.functional as F
from model import build_lcf
from losses import total_loss

DATA = "/home/alphabridge/Study/reliableAI_final/lcf/data"


def load(name, split):
    d = torch.load(f"{DATA}/reps_{name}_{split}.pt", weights_only=False)
    return d["R_plus"].float(), d["R_minus"].float(), d["d"]


def heldout_sep(lcf, Rp, Rm, device):
    """Nearest-centroid separability of the logic projection on held-out reps,
    with centroids from a disjoint half (honest, not the optimistic in-sample)."""
    with torch.no_grad():
        _, zp = lcf.encode(Rp.to(device)); _, zm = lcf.encode(Rm.to(device))
        zp = F.normalize(zp, dim=-1).cpu(); zm = F.normalize(zm, dim=-1).cpu()
    n = len(zp); h = n // 2
    cpos, cneg = zp[:h].mean(0), zm[:h].mean(0)
    # classify the other half by nearest centroid
    def acc(z, lbl_pos):
        dp = (z - cpos).norm(dim=-1); dn = (z - cneg).norm(dim=-1)
        pred_pos = dp < dn
        return (pred_pos == lbl_pos).float().mean().item()
    return 0.5 * (acc(zp[h:], True) + acc(zm[h:], False))


def train_one(Rp, Rm, d, lr, epochs, batch, tau, device):
    lcf = build_lcf(d=d, dims=(2048, 1024)).to(device)
    opt = torch.optim.AdamW(lcf.parameters(), lr=lr, weight_decay=0.01)
    n = len(Rp); g = torch.Generator().manual_seed(0)
    last = {}
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            rp, rm = Rp[idx].to(device), Rm[idx].to(device)
            lcf.update_centroids(torch.cat([lcf.encode(rp)[1], lcf.encode(rm)[1]]).detach(),
                                 torch.cat([torch.ones(len(rp)), torch.zeros(len(rm))]).to(device))
            loss, comps = total_loss(lcf, rp, rm, tau=tau)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(lcf.parameters(), 1.0); opt.step()
        last = comps
    return lcf, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-8B")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Rp, Rm, d = load(args.model, "train")
    Vp, Vm, _ = load(args.model, "val")
    print(f"== {args.model}  train={len(Rp)} val={len(Vp)}  d={d} ==")
    print(f"{'lr':>7} {'ep':>4} {'bs':>5} {'tau':>5} | {'infoNCE+':>8} {'infoNCE-':>8} | {'val-sep':>7}")
    grid = list(itertools.product(
        [1e-3, 5e-3, 1e-2],   # lr (paper=1e-3)
        [10, 40],             # epochs (paper=10)
        [256, 1024],          # batch
        [0.1, 0.05],          # tau (paper=0.1)
    ))
    # keep it bounded: a representative subset
    picks = [(1e-3,10,256,0.1),(5e-3,40,256,0.05),(1e-2,40,1024,0.05),
             (1e-2,40,256,0.05),(5e-3,40,1024,0.1),(1e-2,10,256,0.05)]
    for lr, ep, bs, tau in picks:
        lcf, comps = train_one(Rp, Rm, d, lr, ep, bs, tau, device)
        sep = heldout_sep(lcf, Vp, Vm, device)
        print(f"{lr:>7.0e} {ep:>4} {bs:>5} {tau:>5} | "
              f"{comps['logic_pos']:>8.3f} {comps['logic_neg']:>8.3f} | {sep:>7.3f}")


if __name__ == "__main__":
    main()
