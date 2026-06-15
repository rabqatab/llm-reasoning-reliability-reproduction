"""Model-agnostic logic steering v3 — CAA direction + CONDITIONAL gate.

Grounded in prior work (docs/LCF_model_agnostic.md): CAA (Rimsky et al. ACL'24)
mean-difference steering vector on the residual stream at a localized layer, applied
CONDITIONALLY (CAST/K-CAST; Valentino et al. AAAI'26 show static steering is model-
dependent, conditional rescues unresponsive models). Compares original vs static-CAA
vs conditional-CAA on fallacy identification (ΔProb/Acc), for any HF model.

Usage: uv run python lcf_caa.py --model <id> --layer 12 --alphas 0,2,4,8
"""
from __future__ import annotations
import argparse, json, os
import numpy as np, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA = "/home/alphabridge/Study/reliableAI_final/lcf/data"


def decoder_layers(model):
    m = model
    for a in ("model", "transformer"):
        if hasattr(m, a):
            m = getattr(m, a)
    return m.layers if hasattr(m, "layers") else m.h


@torch.no_grad()
def layer_rep(model, tok, text, L, device):
    """Mean residual-stream (decoder layer L output) over the text's tokens."""
    ids = tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    out = model(**ids, output_hidden_states=True)
    return out.hidden_states[L + 1][0].mean(0).float().cpu()   # hidden_states[0]=embeds


def build_caa(model, tok, L, device, n=200):
    rows = [json.loads(l) for l in open(f"{DATA}/conclusion_gen_train.jsonl") if l.strip()]
    rows = [r for r in rows if r.get("valid_conclusion", "").strip() and r.get("invalid_conclusion", "").strip()][:n]
    vp = torch.stack([layer_rep(model, tok, r["valid_conclusion"], L, device) for r in rows])
    vm = torch.stack([layer_rep(model, tok, r["invalid_conclusion"], L, device) for r in rows])
    mu_v, mu_i = vp.mean(0), vm.mean(0)
    v = mu_v - mu_i
    vhat = v / (v.norm() + 1e-8)
    mid = 0.5 * (mu_v + mu_i)                 # midpoint; proj>0 => valid side
    hnorm = float(torch.cat([vp, vm]).norm(dim=-1).mean())
    return vhat, mid, hnorm


class Steer:
    def __init__(self, model, L, vhat, mid, hnorm):
        self.vhat = vhat; self.mid = mid; self.hnorm = hnorm
        self.alpha = 0.0; self.conditional = False
        layers = decoder_layers(model)
        self.h = layers[L].register_forward_hook(self._hook)
        self.dev = None

    def _hook(self, mod, inp, out):
        if self.alpha == 0.0:
            return out
        t = out[0] if isinstance(out, tuple) else out          # (B,T,d) residual stream
        if self.dev is None or self.dev != t.device:
            self.vhat = self.vhat.to(t.device, t.dtype); self.mid = self.mid.to(t.device, t.dtype); self.dev = t.device
        step = self.alpha * self.hnorm * self.vhat               # CAA: norm-scaled push toward valid
        if self.conditional:
            proj = ((t - self.mid) * self.vhat).sum(-1, keepdim=True)   # >0 valid side
            gate = torch.relu(-proj)                              # only invalid-side tokens, by distance
            add = gate * (self.alpha * self.vhat)
        else:
            add = step
        mod = t + add
        return (mod,) + tuple(out[1:]) if isinstance(out, tuple) else mod


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
    ap.add_argument("--layer", type=int, required=True, help="residual-stream layer for steering")
    ap.add_argument("--alphas", default="0,2,4,8")
    ap.add_argument("--n-dir", type=int, default=150)
    args = ap.parse_args()
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 device_map="cuda", trust_remote_code=True).eval()
    short = args.model.split("/")[-1]
    print(f"== CAA steering {short} L{args.layer} ==", flush=True)
    vhat, mid, hnorm = build_caa(model, tok, args.layer, device, args.n_dir)
    items = [json.loads(l) for l in open(f"{DATA}/fallacy_id_test.jsonl") if l.strip()]
    st = Steer(model, args.layer, vhat, mid, hnorm)
    print(f"{'mode':>12} {'alpha':>6} | {'Acc':>6} {'ΔProb':>7}")
    for a in [float(x) for x in args.alphas.split(",")]:
        for cond in ([False] if a == 0 else [False, True]):
            st.alpha = a; st.conditional = cond
            acc, dp = acc_delta(model, tok, items, device)
            mode = "original" if a == 0 else ("cond-CAA" if cond else "static-CAA")
            print(f"{mode:>12} {a:>6.1f} | {acc:>6.2f} {dp:>7.3f}", flush=True)


if __name__ == "__main__":
    main()
