"""Extract paired (valid, invalid) hidden representations from a frozen base LLM.

For each (valid_conclusion, invalid_conclusion) pair sharing the same premise:
  1. Tokenize "premise + conclusion" for both.
  2. Forward pass (frozen), hooking the ATTENTION and MLP sub-layer OUTPUTS at
     layers 10-30 (spec section A/C).
  3. Find IDENTICAL tokens that appear in BOTH conclusions (token-id intersection
     over the conclusion span), and for each identical-token occurrence sample 2
     distinct sub-layers in [10,30].
  4. Save (R_input_plus, R_input_minus, layer, sublayer, label) tensors.

R_input_plus  = rep from the VALID conclusion  (label 1)
R_input_minus = rep from the INVALID conclusion (label 0)
A "sub-layer" is an (layer, kind) pair where kind in {attn, mlp}.

Output: lcf/data/reps_<model>.pt  -> dict with stacked tensors:
  R_plus  (N, d), R_minus (N, d), layer (N,), kind (N,) [0=attn,1=mlp]

Run on GPU via sparkq (see README). CPU smoke test: --smoke uses a tiny random
GPT2-like model is NOT loaded; instead pass --tiny to assert hook plumbing with a
2-layer toy model.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
# LCF_DATA_DIR lets the same pipeline run on alternative domains (e.g. legal
# valid/invalid conclusion pairs in lcf/legal/) without clobbering lcf/data/.
DATA_DIR = Path(os.environ.get("LCF_DATA_DIR", ROOT / "data"))

LAYER_LO, LAYER_HI = 10, 30  # inclusive range to sample from
SEED = 42


# --------------------------------------------------------------- hooking
class SubLayerCapture:
    """Registers forward hooks on attn + mlp submodules of decoder layers.

    Works with HF LlamaForCausalLM / Qwen3ForCausalLM style models where each
    decoder layer has `.self_attn` and `.mlp` submodules. Captures the OUTPUT of
    each. Stored as captures[(layer_idx, kind)] = tensor (B, T, d).
    """

    def __init__(self, model, layers):
        self.handles = []
        self.captures = {}
        self.layers = set(layers)
        decoder_layers = self._find_layers(model)
        for li, layer in enumerate(decoder_layers):
            if li not in self.layers:
                continue
            attn = getattr(layer, "self_attn", None)
            mlp = getattr(layer, "mlp", None)
            if attn is not None:
                self.handles.append(attn.register_forward_hook(self._mk_hook(li, "attn")))
            if mlp is not None:
                self.handles.append(mlp.register_forward_hook(self._mk_hook(li, "mlp")))

    @staticmethod
    def _find_layers(model):
        # transformers: model.model.layers (Llama/Qwen). Fall back to search.
        m = model
        for attr in ("model", "transformer"):
            if hasattr(m, attr):
                m = getattr(m, attr)
        if hasattr(m, "layers"):
            return m.layers
        if hasattr(m, "h"):
            return m.h
        raise RuntimeError("could not locate decoder layers")

    def _mk_hook(self, li, kind):
        def hook(_module, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            self.captures[(li, kind)] = t.detach()
        return hook

    def clear(self):
        self.captures = {}

    def remove(self):
        for h in self.handles:
            h.remove()


# --------------------------------------------------------------- token align
def conclusion_token_ids(tokenizer, premise, conclusion):
    """Return (input_ids tensor, conclusion_start_index).

    Encodes premise + " " + conclusion; the conclusion span starts after the
    premise tokens so we only align tokens within the conclusion.
    """
    full = (premise.rstrip() + " " + conclusion.strip())
    ids = tokenizer(full, return_tensors="pt").input_ids[0]
    prem_ids = tokenizer(premise.rstrip(), return_tensors="pt").input_ids[0]
    start = min(len(prem_ids), len(ids))  # conclusion span begins here
    return ids, start


def identical_token_positions(ids_v, start_v, ids_m, start_m):
    """Match identical token ids between the two conclusion spans.

    Returns list of (pos_v, pos_m) where the token id is identical. Greedy
    one-to-one matching over the multiset intersection (first-come).
    """
    conc_v = [(i, int(ids_v[i])) for i in range(start_v, len(ids_v))]
    conc_m = [(j, int(ids_m[j])) for j in range(start_m, len(ids_m))]
    used_m = set()
    pairs = []
    for (i, tid) in conc_v:
        for (j, tjd) in conc_m:
            if j in used_m:
                continue
            if tjd == tid:
                pairs.append((i, j))
                used_m.add(j)
                break
    return pairs


# --------------------------------------------------------------- extraction
def extract(model, tokenizer, pairs_data, device, layers, n_layers_per_token=2,
            max_pairs=None):
    rng = random.Random(SEED)
    cap = SubLayerCapture(model, layers)
    kinds = ["attn", "mlp"]
    sublayers = [(li, k) for li in layers for k in kinds]

    R_plus, R_minus, out_layer, out_kind = [], [], [], []
    model.eval()
    n_done = 0
    for rec in pairs_data:
        valid = rec.get("valid_conclusion", "").strip()
        invalid = rec.get("invalid_conclusion", "").strip()
        premise = rec.get("premise", "").strip()
        if not valid or not invalid:
            continue

        ids_v, sv = conclusion_token_ids(tokenizer, premise, valid)
        ids_m, sm = conclusion_token_ids(tokenizer, premise, invalid)

        cap.clear()
        with torch.no_grad():
            model(ids_v.unsqueeze(0).to(device))
        cap_v = {k: v.float().cpu() for k, v in cap.captures.items()}

        cap.clear()
        with torch.no_grad():
            model(ids_m.unsqueeze(0).to(device))
        cap_m = {k: v.float().cpu() for k, v in cap.captures.items()}

        tok_pairs = identical_token_positions(ids_v, sv, ids_m, sm)
        for (pi, pj) in tok_pairs:
            chosen = rng.sample(sublayers, k=min(n_layers_per_token, len(sublayers)))
            for (li, kind) in chosen:
                if (li, kind) not in cap_v or (li, kind) not in cap_m:
                    continue
                R_plus.append(cap_v[(li, kind)][0, pi])
                R_minus.append(cap_m[(li, kind)][0, pj])
                out_layer.append(li)
                out_kind.append(0 if kind == "attn" else 1)
        n_done += 1
        if max_pairs and n_done >= max_pairs:
            break

    cap.remove()
    if not R_plus:
        raise RuntimeError("no token pairs extracted (empty valid/invalid conclusions?)")
    return {
        "R_plus": torch.stack(R_plus),
        "R_minus": torch.stack(R_minus),
        "layer": torch.tensor(out_layer, dtype=torch.long),
        "kind": torch.tensor(out_kind, dtype=torch.long),
    }


def load_pairs(split):
    path = DATA_DIR / f"conclusion_gen_{split}.jsonl"
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--tiny", action="store_true",
                    help="CPU smoke: build a 2-layer toy model and assert hook plumbing")
    args = ap.parse_args()

    layers = list(range(LAYER_LO, LAYER_HI + 1))

    if args.tiny:
        _smoke_tiny(layers)
        return

    import torch as T
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = T.bfloat16 if T.cuda.is_available() else T.float32
    device = "cuda" if T.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    d = model.config.hidden_size
    print(f"[extract] model={args.model} d={d} layers={LAYER_LO}-{LAYER_HI} split={args.split}")

    pairs = load_pairs(args.split)
    out = extract(model, tok, pairs, device, layers, max_pairs=args.max_pairs)
    out["d"] = d
    out["model"] = args.model

    short = args.model.split("/")[-1]
    out_path = Path(args.out) if args.out else DATA_DIR / f"reps_{short}_{args.split}.pt"
    torch.save(out, out_path)
    print(f"[extract] saved {out['R_plus'].shape[0]} token pairs -> {out_path}")


# ------------------------------------------------------- CPU smoke (tiny)
def _smoke_tiny(layers):
    """Assert hooking + alignment logic with a 2-layer toy transformer."""
    import torch.nn as nn

    d = 16
    class ToyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.Linear(d, d)
            self.mlp = nn.Linear(d, d)
        def forward(self, x):
            return x + self.self_attn(x) + self.mlp(x)

    class ToyInner(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.layers = nn.ModuleList([ToyLayer() for _ in range(n)])
        def forward(self, ids):
            x = torch.randn(ids.shape[0], ids.shape[1], d)
            for layer in self.layers:
                x = layer(x)
            return x

    class ToyModel(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.model = ToyInner(n)
            class Cfg: hidden_size = d
            self.config = Cfg()
        def forward(self, ids):
            return self.model(ids)

    n = 32
    toy = ToyModel(n)
    cap = SubLayerCapture(toy, [10, 15, 20])
    toy(torch.zeros(1, 5, dtype=torch.long))
    keys = sorted(cap.captures.keys())
    assert (10, "attn") in cap.captures and (15, "mlp") in cap.captures, keys
    for k, v in cap.captures.items():
        assert v.shape == (1, 5, d), (k, v.shape)
    cap.remove()

    # token alignment
    ids_v = torch.tensor([1, 2, 3, 7, 8, 9])
    ids_m = torch.tensor([1, 2, 3, 7, 5, 9])
    pairs = identical_token_positions(ids_v, 3, ids_m, 3)
    # conclusion spans: v=[7,8,9] m=[7,5,9]; identical: 7->pos3,3 ; 9->pos5,5
    assert (3, 3) in pairs and (5, 5) in pairs and (4, 4) not in pairs, pairs
    print("[smoke-tiny] OK: captured", keys, "token pairs", pairs)


if __name__ == "__main__":
    main()
