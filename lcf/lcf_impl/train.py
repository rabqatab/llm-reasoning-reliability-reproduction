"""Train ONE shared LCF adapter on extracted reps (spec section C).

AdamW lr 1e-3, 10 epochs, wd 0.01, grad-clip 1.0, batch >= 256.
Total loss = L_rec + L_logic+ + L_logic- + L_content.
wandb project: lcf-repro.

After training, select the 10 most-distinctive sub-layers on the val reps via
nearest-centroid separability and store them in config.json.

Save: lcf/checkpoints/<model>/lcf.pt + config.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from model import build_lcf
from losses import total_loss

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"

ETA_DEFAULTS = {"generation": 0.5, "identification": 4.5}


def load_reps(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    return d


def make_loader(reps, batch_size):
    ds = TensorDataset(reps["R_plus"], reps["R_minus"], reps["layer"], reps["kind"])
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)


def select_layers_by_distinctiveness(lcf, reps, top_k=10):
    """Nearest-centroid separability of valid vs invalid logic reps per sub-layer.

    For each (layer, kind) sub-layer, project R_plus/R_minus to logic space,
    compute class centroids, and score separability = fraction of samples closer
    to their own centroid (nearest-centroid accuracy). Return top_k sub-layers.
    """
    lcf.eval()
    R_plus, R_minus = reps["R_plus"], reps["R_minus"]
    layer, kind = reps["layer"], reps["kind"]
    with torch.no_grad():
        _, logic_p = lcf.encode(R_plus)
        _, logic_m = lcf.encode(R_minus)

    scores = {}
    sublayers = set((int(l), int(k)) for l, k in zip(layer, kind))
    for (li, ki) in sublayers:
        mask = (layer == li) & (kind == ki)
        lp = logic_p[mask]
        lm = logic_m[mask]
        if lp.numel() == 0 or lm.numel() == 0:
            continue
        c_pos = lp.mean(0)
        c_neg = lm.mean(0)
        # nearest-centroid accuracy
        def acc(x, own, other):
            d_own = (x - own).norm(dim=-1)
            d_other = (x - other).norm(dim=-1)
            return (d_own < d_other).float().mean().item()
        sep = 0.5 * (acc(lp, c_pos, c_neg) + acc(lm, c_neg, c_pos))
        scores[(li, ki)] = sep

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top = ranked[:top_k]
    return [{"layer": li, "kind": ("attn" if ki == 0 else "mlp"), "score": s}
            for (li, ki), s in top]


def freeze_centroids_from_reps(lcf, reps):
    """Compute global C_pos/C_neg over all train logic reps and freeze them."""
    lcf.eval()
    with torch.no_grad():
        _, logic_p = lcf.encode(reps["R_plus"])
        _, logic_m = lcf.encode(reps["R_minus"])
        lcf.set_centroids(logic_p.mean(0), logic_m.mean(0))


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reps = load_reps(args.reps)
    d = int(reps.get("d", reps["R_plus"].shape[1]))
    assert d == reps["R_plus"].shape[1]

    lcf = build_lcf(d=d, dims=tuple(args.dims)).to(device)
    loader = make_loader(reps, args.batch_size)
    opt = torch.optim.AdamW(lcf.parameters(), lr=args.lr, weight_decay=args.wd)

    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(project="lcf-repro", config=vars(args))
        except Exception as e:  # noqa
            print(f"[warn] wandb disabled: {e}")
            use_wandb = False

    step = 0
    for epoch in range(args.epochs):
        lcf.train()
        for Rp, Rm, lay, kin in loader:
            Rp, Rm = Rp.to(device), Rm.to(device)
            # EMA centroids from this batch's logic reps
            with torch.no_grad():
                _, lg_p = lcf.encode(Rp)
                _, lg_m = lcf.encode(Rm)
                lcf.update_centroids(
                    torch.cat([lg_p, lg_m]),
                    torch.cat([torch.ones(len(Rp)), torch.zeros(len(Rm))]).long(),
                )
            loss, comps = total_loss(lcf, Rp, Rm, tau=args.tau)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lcf.parameters(), args.grad_clip)
            opt.step()
            step += 1
            if use_wandb:
                import wandb
                wandb.log({f"train/{k}": float(v) for k, v in comps.items()}, step=step)
        print(f"epoch {epoch}: loss={float(comps['loss']):.4f} "
              f"rec={float(comps['rec']):.4f} pos={float(comps['logic_pos']):.4f} "
              f"neg={float(comps['logic_neg']):.4f} content={float(comps['content']):.4f}")

    # freeze global centroids for inference
    freeze_centroids_from_reps(lcf, {k: v.to(device) if torch.is_tensor(v) else v
                                     for k, v in reps.items()})

    # select distinctive sub-layers on val reps (fall back to train reps)
    val_reps = reps
    if args.val_reps and Path(args.val_reps).exists():
        val_reps = load_reps(args.val_reps)
    sel = select_layers_by_distinctiveness(lcf.cpu(), val_reps, top_k=10)
    lcf.to(device)

    short = args.model.split("/")[-1]
    out_dir = CKPT_DIR / short
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(lcf.state_dict(), out_dir / "lcf.pt")
    config = {
        "model": args.model,
        "d": d,
        "dims": list(args.dims),
        "proj_dim": args.dims[1],
        "eta_defaults": ETA_DEFAULTS,
        "selected_sublayers": sel,
        "tau": args.tau,
        "layer_range": [10, 30],
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"[train] saved -> {out_dir}/lcf.pt + config.json")
    print("[train] top sub-layers:", json.dumps(sel, indent=2))
    if use_wandb:
        import wandb
        wandb.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--reps", required=True, help="path to reps_<model>_train.pt")
    ap.add_argument("--val-reps", default=None, help="path to reps_<model>_val.pt")
    ap.add_argument("--dims", type=int, nargs=2, default=[2048, 1024])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
