"""Model-agnostic logic steering v4 — faithful K-CAST (kNN-conditional gate).

The v3 conditional gate (`lcf_caa.py`) was a midpoint projection on the steering
direction — essentially a 1-NN-to-centroid linear gate that fires on almost every
token, so it behaved like static steering and did not rescue any model. This v4
implements the gate the AAAI'26 result actually relies on: a **kNN classifier on
reference representations** (K-CAST / CAST). Added components vs v3:

  1. kNN gate: at layer L, classify each token's residual rep against a labelled
     reference set {valid, invalid}; steer (push toward valid) ONLY tokens the
     kNN votes "invalid-side". This is the conditional that, per Valentino AAAI'26,
     rescues unresponsive models — tested faithfully here.
  2. LayerNavigator: pick L per model as the layer with max held-out kNN
     valid/invalid separability (leave-one-out), among candidate layers.
  3. Signed alpha sweep: alpha in {-8,-4,4,8} (v3 only tried positive); the
     fallacy-identification target may need the opposite sign.
  4. Gate sanity: report the chosen layer's kNN separability and the gate firing
     rate, so we can tell whether the gate discriminates at all.

Direction is still CAA mean-difference v_L = mean(h_valid) - mean(h_invalid)
on the residual stream (model-agnostic by construction). Compares original vs
static-CAA vs kNN-CAST on fallacy identification (Acc, dProb), for any HF model.

Usage: uv run python lcf_kcast.py --model <id> [--layer L | --auto-layer] \
          --alphas -8,-4,4,8 --n-dir 100 --k 7
"""
from __future__ import annotations
import argparse, json, os
import numpy as np, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA = os.environ.get("LCF_DATA", "/home/alphabridge/Study/reliableAI_final/lcf/data")


def decoder_layers(model):
    m = model
    for a in ("model", "transformer"):
        if hasattr(m, a):
            m = getattr(m, a)
    return m.layers if hasattr(m, "layers") else m.h


@torch.no_grad()
def all_layer_reps(model, tok, texts, device, layers):
    """Mean residual-stream rep at each requested layer, one forward per text.
    Returns {L: (n, d) float cpu tensor}."""
    acc = {L: [] for L in layers}
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=128).to(device)
        out = model(**ids, output_hidden_states=True)
        for L in layers:
            acc[L].append(out.hidden_states[L + 1][0].mean(0).float().cpu())
    return {L: torch.stack(v) for L, v in acc.items()}


def knn_sep(Rv, Ri, k=7):
    """Leave-one-out kNN accuracy separating valid (Rv) from invalid (Ri) reps,
    cosine metric. ~0.5 = no signal, 1.0 = perfectly separable."""
    X = torch.cat([Rv, Ri]); y = torch.tensor([1] * len(Rv) + [0] * len(Ri))
    Xn = X / (X.norm(dim=-1, keepdim=True) + 1e-8)
    S = Xn @ Xn.T
    S.fill_diagonal_(-1e9)
    nn = S.topk(min(k, len(X) - 1), dim=-1).indices
    pred = (y[nn].float().mean(-1) > 0.5).long()
    return (pred == y).float().mean().item()


class Steer:
    """Forward hook on decoder layer L. mode: 'static' steers every token;
    'knn' steers only tokens the kNN classifies invalid-side."""
    def __init__(self, model, L, vhat, hnorm, refn, reflabel, k):
        self.vhat = vhat; self.hnorm = hnorm
        self.refn = refn; self.reflabel = reflabel; self.k = k
        self.alpha = 0.0; self.mode = "static"
        self.fire_num = 0.0; self.fire_den = 0.0      # gate firing-rate accounting
        self.h = decoder_layers(model)[L].register_forward_hook(self._hook)
        self.dev = None

    def _hook(self, mod, inp, out):
        if self.alpha == 0.0:
            return out
        t = out[0] if isinstance(out, tuple) else out          # (B,T,d)
        if self.dev is None or self.dev != t.device:
            self.vhat = self.vhat.to(t.device, t.dtype)
            self.refn = self.refn.to(t.device, t.dtype)
            self.reflabel = self.reflabel.to(t.device)
            self.dev = t.device
        push = self.alpha * self.hnorm * self.vhat               # CAA push toward valid
        if self.mode == "knn":
            tn = t / (t.norm(dim=-1, keepdim=True) + 1e-8)       # (B,T,d)
            sims = tn @ self.refn.T                               # (B,T,Nref) cosine
            nn = sims.topk(self.k, dim=-1).indices                # (B,T,k)
            inv_frac = (self.reflabel[nn] == 0).float().mean(-1)  # frac invalid neighbours
            gate = (inv_frac > 0.5).to(t.dtype).unsqueeze(-1)     # (B,T,1) — match residual dtype
            self.fire_num += float(gate.float().sum()); self.fire_den += gate.numel()
            add = gate * push
        else:
            add = push
        mod_out = t + add
        return (mod_out,) + tuple(out[1:]) if isinstance(out, tuple) else mod_out


@torch.no_grad()
def seq_lp(model, tok, prompt, cont, device):
    p = tok(prompt, return_tensors="pt").input_ids
    full = tok(prompt + " " + cont, return_tensors="pt").input_ids.to(device)
    n = p.shape[1]
    lp = F.log_softmax(model(full).logits[0, :-1].float(), -1)
    tgt = full[0, 1:]
    return (lp[range(tgt.shape[0]), tgt][n - 1:].sum() / max(1, tgt.shape[0] - n + 1)).item()


def acc_delta(model, tok, items, device):
    correct, deltas = 0, []
    for it in items:
        lps = [seq_lp(model, tok, it["premise"] + "\n", o, device) for o in it["options"]]
        p = F.softmax(torch.tensor(lps), 0).numpy(); ai = it["answer_idx"]
        correct += int(np.argmax(lps) == ai)
        deltas.append(p[ai] - float(np.mean([p[j] for j in range(len(p)) if j != ai])))
    return 100.0 * correct / len(items), 100.0 * float(np.mean(deltas))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=None, help="fixed steering layer; omit to auto-select")
    ap.add_argument("--auto-layer", action="store_true", help="pick layer by max kNN separability")
    ap.add_argument("--cand-layers", default=None, help="comma list of candidate layers for --auto-layer")
    ap.add_argument("--alphas", default="-8,-4,4,8")
    ap.add_argument("--n-dir", type=int, default=100)
    ap.add_argument("--k", type=int, default=7, help="kNN neighbours for the gate / separability")
    args = ap.parse_args()
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 device_map="cuda", trust_remote_code=True).eval()
    short = args.model.split("/")[-1]
    n_layers = len(decoder_layers(model))

    # Reference texts (valid / invalid conclusions).
    rows = [json.loads(l) for l in open(f"{DATA}/conclusion_gen_train.jsonl") if l.strip()]
    rows = [r for r in rows if r.get("valid_conclusion", "").strip() and r.get("invalid_conclusion", "").strip()][:args.n_dir]
    vtex = [r["valid_conclusion"] for r in rows]; itex = [r["invalid_conclusion"] for r in rows]

    # Candidate layers for LayerNavigator.
    if args.cand_layers:
        cands = [int(x) for x in args.cand_layers.split(",")]
    elif args.auto_layer or args.layer is None:
        cands = list(range(max(2, n_layers // 4), min(n_layers - 1, 3 * n_layers // 4), 2))
    else:
        cands = [args.layer]
    print(f"== K-CAST {short} (n_layers={n_layers}) candidate layers {cands} ==", flush=True)

    repsV = all_layer_reps(model, tok, vtex, device, cands)
    repsI = all_layer_reps(model, tok, itex, device, cands)
    seps = {L: knn_sep(repsV[L], repsI[L], args.k) for L in cands}
    for L in cands:
        print(f"   layer {L:>2}: kNN separability {seps[L]:.3f}", flush=True)
    L = args.layer if (args.layer is not None and not args.auto_layer) else max(seps, key=seps.get)
    if L not in repsV:  # fixed layer not among cands → extract it
        rr = all_layer_reps(model, tok, vtex, device, [L]); repsV[L] = rr[L]
        rr = all_layer_reps(model, tok, itex, device, [L]); repsI[L] = rr[L]
    print(f"== steering layer L={L} (separability {seps.get(L, knn_sep(repsV[L], repsI[L], args.k)):.3f}) ==", flush=True)

    # CAA direction + reference set at L.
    Rv, Ri = repsV[L], repsI[L]
    vhat = (Rv.mean(0) - Ri.mean(0)); vhat = vhat / (vhat.norm() + 1e-8)
    hnorm = float(torch.cat([Rv, Ri]).norm(dim=-1).mean())
    ref = torch.cat([Rv, Ri]); refn = ref / (ref.norm(dim=-1, keepdim=True) + 1e-8)
    reflabel = torch.tensor([1] * len(Rv) + [0] * len(Ri))

    items = [json.loads(l) for l in open(f"{DATA}/fallacy_id_test.jsonl") if l.strip()]
    st = Steer(model, L, vhat, hnorm, refn, reflabel, args.k)

    print(f"{'mode':>10} {'alpha':>6} | {'Acc':>6} {'dProb':>7} {'gate%':>6}")
    a0, d0 = acc_delta(model, tok, items, device)   # alpha=0 baseline
    print(f"{'original':>10} {0.0:>6.1f} | {a0:>6.2f} {d0:>7.3f} {'-':>6}", flush=True)
    for a in [float(x) for x in args.alphas.split(",")]:
        for mode in ("static", "knn"):
            st.alpha = a; st.mode = mode; st.fire_num = st.fire_den = 0.0
            acc, dp = acc_delta(model, tok, items, device)
            fire = (100.0 * st.fire_num / st.fire_den) if (mode == "knn" and st.fire_den) else float("nan")
            tag = "static-CAA" if mode == "static" else "kNN-CAST"
            print(f"{tag:>10} {a:>6.1f} | {acc:>6.2f} {dp:>7.3f} {fire:>6.1f}", flush=True)


if __name__ == "__main__":
    main()
