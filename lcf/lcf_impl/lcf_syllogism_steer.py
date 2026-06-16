"""K-CAST on the FORMAL SYLLOGISM task — the reference==task-distribution test
that LCF_model_agnostic.md S7 flagged as the one remaining open route.

On the fallacy task the kNN gate fired on ~98% of tokens because the steering
reference texts (short conclusion sentences) were a different distribution from the
task tokens. Here reference (train syllogisms) and task (test syllogisms) share the
same distribution, so the gate *can* be selective — we test whether it is, and
whether conditional steering then debiases the content effect.

Two steering directions:
  --direction content : v = mean(believable) - mean(unbelievable) reps; ABLATE it
      (h' = h - alpha*(h.v)v) on believable-side tokens. The faithful debiaser:
      remove the believability signal so the model judges on logical form.
  --direction validity: v = mean(valid) - mean(invalid) reps; ADD alpha*||h||*v on
      invalid-side tokens (the v4 recipe), for comparison.

Reports, original vs static vs kNN-CAST, per alpha:
  overall Acc, dProb(correct), gate%, and per-cell accuracy VB/VU/IB/IU plus the
  CONTENT-EFFECT GAP = acc(congruent: VB,IU) - acc(conflict: VU,IB). A working
  debiaser shrinks that gap (raises conflict-cell accuracy).

Usage: uv run python lcf_syllogism_steer.py --model <id> --direction content \
          --auto-layer --alphas 0.5,1,2 --k 7
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
    acc = {L: [] for L in layers}
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=160).to(device)
        out = model(**ids, output_hidden_states=True)
        for L in layers:
            acc[L].append(out.hidden_states[L + 1][0].mean(0).float().cpu())
    return {L: torch.stack(v) for L, v in acc.items()}


def knn_sep(Rpos, Rneg, k=7):
    X = torch.cat([Rpos, Rneg]); y = torch.tensor([1] * len(Rpos) + [0] * len(Rneg))
    Xn = X / (X.norm(dim=-1, keepdim=True) + 1e-8)
    S = Xn @ Xn.T; S.fill_diagonal_(-1e9)
    nn = S.topk(min(k, len(X) - 1), dim=-1).indices
    return ((y[nn].float().mean(-1) > 0.5).long() == y).float().mean().item()


class Steer:
    """mode static -> apply to every token; knn -> only tokens the kNN puts on the
    positive (steer-worthy) side. kind 'add' (validity) or 'ablate' (content)."""
    def __init__(self, model, L, vhat, hnorm, refn, reflabel, k, kind):
        self.vhat = vhat; self.hnorm = hnorm; self.kind = kind
        self.refn = refn; self.reflabel = reflabel; self.k = k
        self.alpha = 0.0; self.mode = "static"
        self.fire_num = self.fire_den = 0.0
        self.h = decoder_layers(model)[L].register_forward_hook(self._hook)
        self.dev = None

    def _hook(self, mod, inp, out):
        if self.alpha == 0.0:
            return out
        t = out[0] if isinstance(out, tuple) else out
        if self.dev is None or self.dev != t.device:
            self.vhat = self.vhat.to(t.device, t.dtype)
            self.refn = self.refn.to(t.device, t.dtype)
            self.reflabel = self.reflabel.to(t.device); self.dev = t.device
        if self.kind == "ablate":
            comp = (t * self.vhat).sum(-1, keepdim=True) * self.vhat   # projection onto vhat
            delta = -self.alpha * comp                                 # remove the component
        else:
            delta = (self.alpha * self.hnorm * self.vhat).expand_as(t)
        if self.mode == "knn":
            tn = t / (t.norm(dim=-1, keepdim=True) + 1e-8)
            sims = tn @ self.refn.T
            nn = sims.topk(self.k, dim=-1).indices
            pos_frac = (self.reflabel[nn] == 1).float().mean(-1)        # frac positive-side neighbours
            gate = (pos_frac > 0.5).to(t.dtype).unsqueeze(-1)
            self.fire_num += float(gate.float().sum()); self.fire_den += gate.numel()
            delta = gate * delta
        mod_out = t + delta
        return (mod_out,) + tuple(out[1:]) if isinstance(out, tuple) else mod_out


@torch.no_grad()
def seq_lp(model, tok, prompt, cont, device):
    p = tok(prompt, return_tensors="pt").input_ids
    full = tok(prompt + " " + cont, return_tensors="pt").input_ids.to(device)
    n = p.shape[1]
    lp = F.log_softmax(model(full).logits[0, :-1].float(), -1)
    tgt = full[0, 1:]
    return (lp[range(tgt.shape[0]), tgt][n - 1:].sum() / max(1, tgt.shape[0] - n + 1)).item()


def evaluate(model, tok, items, device):
    """Returns overall acc, dProb(correct)x100, and per-cell accuracy dict."""
    correct, deltas = 0, []
    by_cell = {}
    for it in items:
        lps = [seq_lp(model, tok, it["premise"], o, device) for o in it["options"]]
        p = F.softmax(torch.tensor(lps), 0).numpy(); ai = it["answer_idx"]
        ok = int(np.argmax(lps) == ai)
        correct += ok
        deltas.append(p[ai] - p[1 - ai])
        by_cell.setdefault(it["cell"], []).append(ok)
    acc = 100.0 * correct / len(items)
    dprob = 100.0 * float(np.mean(deltas))
    cell_acc = {c: 100.0 * np.mean(v) for c, v in by_cell.items()}
    return acc, dprob, cell_acc


def gap(cell_acc):
    cong = np.mean([cell_acc.get("VB", 0), cell_acc.get("IU", 0)])
    conf = np.mean([cell_acc.get("VU", 0), cell_acc.get("IB", 0)])
    return cong - conf, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--direction", choices=["content", "validity"], default="content")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--auto-layer", action="store_true")
    ap.add_argument("--cand-layers", default=None)
    ap.add_argument("--alphas", default=None, help="default 0.5,1,2 (content) / -8,-4,4,8 (validity)")
    ap.add_argument("--k", type=int, default=7)
    args = ap.parse_args()
    device = "cuda"
    kind = "ablate" if args.direction == "content" else "add"
    if args.alphas is None:
        args.alphas = "0.5,1,2" if args.direction == "content" else "-8,-4,4,8"

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 device_map="cuda", trust_remote_code=True).eval()
    short = args.model.split("/")[-1]; n_layers = len(decoder_layers(model))

    train = [json.loads(l) for l in open(f"{DATA}/syllogism_train.jsonl") if l.strip()]
    test = [json.loads(l) for l in open(f"{DATA}/syllogism_test.jsonl") if l.strip()]
    # Reference texts = the syllogism prompts; positive/negative side by direction.
    if args.direction == "content":
        pos = [r["premise"] for r in train if r["believable"]]
        neg = [r["premise"] for r in train if not r["believable"]]
        poslab = "believable"
    else:
        pos = [r["premise"] for r in train if r["valid"]]
        neg = [r["premise"] for r in train if not r["valid"]]
        poslab = "valid"

    cands = ([int(x) for x in args.cand_layers.split(",")] if args.cand_layers
             else ([args.layer] if (args.layer is not None and not args.auto_layer)
                   else list(range(max(2, n_layers // 4), min(n_layers - 1, 3 * n_layers // 4), 2))))
    print(f"== Syllogism K-CAST {short} | direction={args.direction} (steer={kind}, pos={poslab}) ==", flush=True)
    print(f"   candidate layers {cands}", flush=True)
    Rp = all_layer_reps(model, tok, pos, device, cands)
    Rn = all_layer_reps(model, tok, neg, device, cands)
    seps = {L: knn_sep(Rp[L], Rn[L], args.k) for L in cands}
    for L in cands:
        print(f"   layer {L:>2}: {poslab}/not kNN separability {seps[L]:.3f}", flush=True)
    L = args.layer if (args.layer is not None and not args.auto_layer) else max(seps, key=seps.get)
    if L not in Rp:
        Rp[L] = all_layer_reps(model, tok, pos, device, [L])[L]
        Rn[L] = all_layer_reps(model, tok, neg, device, [L])[L]
    print(f"== layer L={L} (separability {seps.get(L, knn_sep(Rp[L], Rn[L], args.k)):.3f}) ==", flush=True)

    vhat = (Rp[L].mean(0) - Rn[L].mean(0)); vhat = vhat / (vhat.norm() + 1e-8)
    hnorm = float(torch.cat([Rp[L], Rn[L]]).norm(dim=-1).mean())
    ref = torch.cat([Rp[L], Rn[L]]); refn = ref / (ref.norm(dim=-1, keepdim=True) + 1e-8)
    reflabel = torch.tensor([1] * len(Rp[L]) + [0] * len(Rn[L]))
    st = Steer(model, L, vhat, hnorm, refn, reflabel, args.k, kind)

    hdr = f"{'mode':>10} {'alpha':>6} | {'Acc':>5} {'dProb':>6} {'gate%':>5} | {'VB':>5} {'VU':>5} {'IB':>5} {'IU':>5} | {'conflictAcc':>11} {'gap':>6}"
    print(hdr, flush=True)
    a, dp, ca = evaluate(model, tok, test, device)
    g, conf = gap(ca)
    print(f"{'original':>10} {0.0:>6.2f} | {a:>5.1f} {dp:>6.2f} {'-':>5} | {ca.get('VB',0):>5.0f} {ca.get('VU',0):>5.0f} {ca.get('IB',0):>5.0f} {ca.get('IU',0):>5.0f} | {conf:>11.1f} {g:>6.1f}", flush=True)
    for av in [float(x) for x in args.alphas.split(",")]:
        for mode in ("static", "knn"):
            st.alpha = av; st.mode = mode; st.fire_num = st.fire_den = 0.0
            a, dp, ca = evaluate(model, tok, test, device)
            g, conf = gap(ca)
            fire = (100.0 * st.fire_num / st.fire_den) if (mode == "knn" and st.fire_den) else float("nan")
            tag = "static" if mode == "static" else "kNN-CAST"
            print(f"{tag:>10} {av:>6.2f} | {a:>5.1f} {dp:>6.2f} {fire:>5.1f} | {ca.get('VB',0):>5.0f} {ca.get('VU',0):>5.0f} {ca.get('IB',0):>5.0f} {ca.get('IU',0):>5.0f} | {conf:>11.1f} {g:>6.1f}", flush=True)


if __name__ == "__main__":
    main()
