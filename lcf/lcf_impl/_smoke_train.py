"""CPU smoke for train.py logic: synthetic reps -> train 2 epochs -> save ckpt
   -> verify distinctiveness selection + config, then load via infer plumbing
   (LCF state load only, no HF model)."""
import json
import tempfile
from pathlib import Path
import torch

import train as T
from model import build_lcf


def main():
    d = 16
    N = 300
    mu = torch.randn(d) * 2
    reps = {
        "R_plus": mu + 0.3 * torch.randn(N, d),
        "R_minus": -mu + 0.3 * torch.randn(N, d),
        "layer": torch.randint(10, 31, (N,)),
        "kind": torch.randint(0, 2, (N,)),
        "d": d,
        "model": "toy/Tiny",
    }
    tmp = Path(tempfile.mkdtemp())
    rp = tmp / "reps_train.pt"
    torch.save(reps, rp)

    class Args:
        model = "toy/Tiny"
        reps = str(rp)
        val_reps = None
        dims = [32, 8]
        lr = 1e-2
        wd = 0.01
        epochs = 2
        batch_size = 64
        grad_clip = 1.0
        tau = 0.1
        no_wandb = True

    # redirect checkpoint dir
    T.CKPT_DIR = tmp / "ckpt"
    T.train(Args())

    out = tmp / "ckpt" / "Tiny"
    cfg = json.load(open(out / "config.json"))
    assert cfg["d"] == d and cfg["dims"] == [32, 8]
    assert cfg["eta_defaults"] == {"generation": 0.5, "identification": 4.5}
    assert len(cfg["selected_sublayers"]) >= 1
    # separable data => top selected sublayer should have high distinctiveness
    top = cfg["selected_sublayers"][0]
    assert top["score"] > 0.8, cfg["selected_sublayers"][:3]

    # reload state dict into a fresh LCF (infer-style)
    lcf = build_lcf(d=d, dims=tuple(cfg["dims"]))
    state = torch.load(out / "lcf.pt", weights_only=True)
    lcf.load_state_dict(state)
    assert lcf.V.abs().sum() > 0  # centroids were frozen
    print("[smoke-train] OK  selected:", json.dumps(cfg["selected_sublayers"][:3]))
    print("[smoke-train] top distinctiveness score:", top["score"])


if __name__ == "__main__":
    main()
