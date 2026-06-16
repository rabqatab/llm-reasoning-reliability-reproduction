"""Train the LCF module on extracted representations, then derive the steering
vector (eq 2) and the top-K most 'distinctive' taps to modify at inference.

Usage:
  python src/train.py                       # full objective -> ckpt 'lcf_full.pt'
  python src/train.py --no-content --tag no_content
  python src/train.py --no-logic   --tag no_logic
  python src/train.py --no-rec     --tag no_rec
  python src/train.py --no-content-proj --tag no_content_proj
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CFG
from lcf import LCF

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def compute_steering(lcf, valid, invalid, bs=4096):
    zv = torch.cat([lcf.project(valid[i:i+bs].float().to(DEV))[1].cpu()
                    for i in range(0, len(valid), bs)])
    zi = torch.cat([lcf.project(invalid[i:i+bs].float().to(DEV))[1].cpu()
                    for i in range(0, len(invalid), bs)])
    lcf.set_steering_vector(zv.to(DEV), zi.to(DEV))


@torch.no_grad()
def distinctiveness(lcf, val_data):
    """Per-tap nearest-center separability on the validation set (supplementary)."""
    scores = {}
    for (l, kind), d in val_data.items():
        _, zv = lcf.project(d["valid"].float().to(DEV))
        _, zi = lcf.project(d["invalid"].float().to(DEV))
        cpos, cneg = zv.mean(0), zi.mean(0)
        acc_v = ((zv - cpos).norm(dim=1) < (zv - cneg).norm(dim=1)).float().mean().item()
        acc_i = ((zi - cneg).norm(dim=1) < (zi - cpos).norm(dim=1)).float().mean().item()
        scores[(l, kind)] = (acc_v + acc_i) / 2
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-rec", action="store_true")
    ap.add_argument("--no-logic", action="store_true")
    ap.add_argument("--no-content", action="store_true")
    ap.add_argument("--no-content-proj", action="store_true")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--epochs", type=int, default=CFG.epochs)
    ap.add_argument("--lr", type=float, default=CFG.lr)
    ap.add_argument("--clip", type=float, default=1.0, help="grad-norm clip (0=off)")
    ap.add_argument("--tau", type=float, default=CFG.contrastive_tau)
    args = ap.parse_args()

    data = torch.load(CFG.hidden_dir / "train_reps.pt")
    valid, invalid = data["valid"], data["invalid"]
    print(f"train rep-pairs: {valid.shape}  on {DEV}")
    ds = TensorDataset(valid, invalid)
    bs = min(CFG.batch_size, len(ds))
    dl = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=len(ds) > bs)

    lcf = LCF(d_model=CFG.d_model, proj_hidden=CFG.proj_hidden, proj_out=CFG.proj_out,
              dec_hidden=CFG.dec_hidden, tau=args.tau,
              use_content_proj=not args.no_content_proj).to(DEV)
    opt = torch.optim.AdamW(lcf.parameters(), lr=args.lr, weight_decay=CFG.weight_decay)
    print(f"lr={args.lr} clip={args.clip} tau={args.tau} epochs={args.epochs}")

    for ep in range(args.epochs):
        agg = {"rec": 0, "logic": 0, "content": 0, "total": 0}
        for rv, ri in dl:
            rv, ri = rv.float().to(DEV), ri.float().to(DEV)
            opt.zero_grad()
            total, parts = lcf.losses(rv, ri, use_rec=not args.no_rec,
                                      use_logic=not args.no_logic,
                                      use_content=not args.no_content)
            total.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(lcf.parameters(), args.clip)
            opt.step()
            for k in agg:
                agg[k] += parts[k]
        n = max(1, len(dl))
        print(f"epoch {ep+1:2d}/{args.epochs}  " +
              "  ".join(f"{k}={agg[k]/n:.4f}" for k in agg))

    compute_steering(lcf, valid, invalid)

    val_data = torch.load(CFG.hidden_dir / "val_reps.pt")
    scores = distinctiveness(lcf, val_data)
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    if ranked:
        top = [[l, kind] for (l, kind), _ in ranked[: CFG.n_modify_layers]]
        print("\ntop taps by distinctiveness:")
        for (l, kind), s in ranked[: CFG.n_modify_layers]:
            print(f"  layer {l:2d} {kind:4} : {s:.3f}")
    else:  # fallback (e.g., no val coverage in a smoke run): mid-range attn taps
        mid = CFG.layer_lo + 4
        top = [[l, "attn"] for l in range(mid, mid + CFG.n_modify_layers)]
        print(f"\n[warn] no validation reps; falling back to taps {top}")

    ckpt = {
        "state_dict": lcf.state_dict(),
        "steering": lcf.steering.cpu(),
        "top_taps": top,
        "distinctiveness": {f"{l}_{k}": s for (l, k), s in scores.items()},
        "config": {"d_model": CFG.d_model, "proj_hidden": CFG.proj_hidden,
                   "proj_out": CFG.proj_out, "dec_hidden": CFG.dec_hidden,
                   "tau": CFG.contrastive_tau, "use_content_proj": not args.no_content_proj},
        "args": vars(args),
    }
    out = CFG.ckpt_dir / f"lcf_{args.tag}.pt"
    torch.save(ckpt, out)
    print(f"\nsaved checkpoint -> {out}")


if __name__ == "__main__":
    main()
