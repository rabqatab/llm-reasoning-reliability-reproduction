"""Evaluate model-agnostic LCF v2 on fallacy identification, sweeping the
norm-relative strength alpha (alpha=0 is the untouched baseline). Reports Acc +
DeltaProb so we can see whether the single-best-layer supervised direction gives
a CONSISTENT cross-model improvement (the model-agnostic claim). GPU.

Usage (login shell):
  HF_HOME=... HF_HUB_OFFLINE=1 uv run python lcf_v2_eval.py --model Qwen/Qwen3-8B
"""
from __future__ import annotations
import argparse, json, torch, numpy as np
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA = "/home/alphabridge/Study/reliableAI_final/lcf/data"
CKPT = "/home/alphabridge/Study/reliableAI_final/lcf/checkpoints"
FAL_PROMPT = "{premise}\n"


class V2Hook:
    def __init__(self, model, direction, alpha):
        self.w = direction["w"].float()
        self.alpha = alpha
        self.enabled = False
        layers = self._layers(model)
        sub = getattr(layers[direction["layer"]],
                      "self_attn" if direction["kind"] == "attn" else "mlp")
        self.handle = sub.register_forward_hook(self._hook)
        self.w_dev = None

    def _layers(self, m):
        for a in ("model", "transformer"):
            if hasattr(m, a):
                m = getattr(m, a)
        return m.layers if hasattr(m, "layers") else m.h

    def _hook(self, mod, inp, out):
        if not self.enabled:
            return out
        t = out[0] if isinstance(out, tuple) else out      # (B,T,d)
        if self.w_dev is None or self.w_dev.device != t.device:
            self.w_dev = self.w.to(t.device, t.dtype)
        wn = self.w_dev / (self.w_dev.norm() + 1e-8)
        hn = t.norm(dim=-1, keepdim=True)                  # per-token ||h||
        mod = t + self.alpha * hn * wn                      # norm-relative shift toward valid
        return (mod,) + tuple(out[1:]) if isinstance(out, tuple) else mod


@torch.no_grad()
def seq_logprob(model, tok, prompt, cont, device):
    p = tok(prompt, return_tensors="pt").input_ids
    full = tok(prompt + " " + cont, return_tensors="pt").input_ids.to(device)
    n = p.shape[1]
    lp = F.log_softmax(model(full).logits[0, :-1].float(), -1)
    tgt = full[0, 1:]
    return (lp[range(tgt.shape[0]), tgt][n - 1:].sum() / max(1, tgt.shape[0] - n + 1)).item()


def delta_acc(model, tok, items, device, scale=100.0):
    correct, deltas = 0, []
    for it in items:
        prompt = FAL_PROMPT.format(premise=it["premise"])
        lps = [seq_logprob(model, tok, prompt, o, device) for o in it["options"]]
        probs = F.softmax(torch.tensor(lps), 0).numpy()
        ai = it["answer_idx"]
        if int(np.argmax(lps)) == ai:
            correct += 1
        inc = [probs[j] for j in range(len(probs)) if j != ai]
        deltas.append(probs[ai] - float(np.mean(inc)))
    return 100.0 * correct / len(items), scale * float(np.mean(deltas))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--alphas", default="0,0.5,1,2,4")
    args = ap.parse_args()
    short = args.model.split("/")[-1]
    direction = torch.load(f"{CKPT}/{short}/lcf_v2_direction.pt", weights_only=False)
    items = [json.loads(l) for l in open(f"{DATA}/fallacy_id_test.jsonl") if l.strip()]
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 device_map="auto").eval()
    hook = V2Hook(model, direction, alpha=0.0)
    print(f"== LCF v2 {short}  best L{direction['layer']} {direction['kind']} "
          f"probe_acc={direction['probe_val_acc']:.3f} ==")
    print(f"{'alpha':>6} | {'Acc':>6} | {'DeltaProb':>9}")
    for a in [float(x) for x in args.alphas.split(",")]:
        hook.alpha = a; hook.enabled = (a != 0.0)
        acc, dp = delta_acc(model, tok, items, device)
        print(f"{a:>6.1f} | {acc:>6.2f} | {dp:>9.3f}", flush=True)


if __name__ == "__main__":
    main()
