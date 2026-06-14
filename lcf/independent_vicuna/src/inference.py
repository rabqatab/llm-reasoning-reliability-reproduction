"""Load a trained LCF checkpoint and attach it to a base LLM via forward hooks."""
from __future__ import annotations
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CFG
from lcf import LCF
from model_utils import get_taps, Steerer

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_lcf(ckpt_path):
    ck = torch.load(ckpt_path, map_location=DEV)
    c = ck["config"]
    lcf = LCF(d_model=c["d_model"], proj_hidden=c["proj_hidden"], proj_out=c["proj_out"],
              dec_hidden=c["dec_hidden"], tau=c["tau"],
              use_content_proj=c.get("use_content_proj", True)).to(DEV)
    lcf.load_state_dict(ck["state_dict"])
    lcf.eval()
    return lcf, ck


def attach(model, lcf, top_taps, eta, sign=1.0):
    """Register Steerer hooks on exactly the checkpoint's top taps."""
    want = {(int(l), kind) for l, kind in top_taps}
    all_taps = get_taps(model, 0, CFG.n_layers, CFG.tap_points)
    taps = {k: v for k, v in all_taps.items() if k in want}
    assert len(taps) == len(want), f"missing taps: {want - set(taps)}"
    return Steerer(lcf, taps, eta=eta, sign=sign)
