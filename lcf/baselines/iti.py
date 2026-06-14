"""ITI baseline (adapted from honest_llama) — "valid-logic" head intervention.

Spec E: ITI = probe per attention head, shift top-K heads at inference; retarget
"truthful" -> "valid-logic" on LFUD. Table 5 (Llama2): ITI 69.60/62.25.

WHAT IS FULL-FAITHFUL vs REDUCED
--------------------------------
Faithful to honest_llama (reused logic, see lcf/baselines_honest_llama/utils.py):
  * Head-wise activation collection at each layer's attention output, reshaped to
    (num_layers, num_heads, head_dim)  — same scheme as get_activations.py.
  * Per-head linear probe (LogisticRegression) on valid vs invalid reps; rank heads
    by val accuracy; pick top-K  (train_probes / get_top_heads).
  * Center-of-mass direction  mean(valid) - mean(invalid)  per head, unit-normalized,
    scaled by alpha * std(projection)  (get_com_directions / get_interventions_dict).
  * Inference-time additive shift on the selected heads' output  (ITI_Intervener).

Reduced / re-implemented (documented):
  * honest_llama uses `baukit.TraceDict` + a pyvene/`head_out` module name specific
    to Llama. We instead register forward hooks on `self_attn.o_proj` INPUT (the
    concatenated per-head values before the output projection), which is architecture
    -agnostic (works for Qwen3, Llama, Mistral). This is the same intervention point
    semantically (per-head attention output) but via plain PyTorch hooks, not baukit.
  * Probing data is LFUD valid/invalid conclusion reps (not TruthfulQA mc2_targets).
  * We collect the LAST-token rep per sentence (honest_llama also uses last token).

Direction/probe fitting runs on val reps; intervention applied at inference.
CPU-smoke (--smoke) exercises the head reshape + COM-direction math on synthetic reps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
DEFAULT_VAL = REPO_ROOT / "lcf" / "data" / "conclusion_gen_val.jsonl"
DEFAULT_DIR = BASE_DIR / "iti_dirs"


# --------------------------------------------------------------------------- #
# honest_llama-style head math (reused, see utils.py:get_com_directions/get_top_heads)
# --------------------------------------------------------------------------- #
def flattened_idx_to_layer_head(idx, num_heads):
    return idx // num_heads, idx % num_heads


def layer_head_to_flattened_idx(layer, head, num_heads):
    return layer * num_heads + head


def com_directions(head_acts, labels, num_layers, num_heads):
    """Center-of-mass direction per (layer, head): mean(valid) - mean(invalid).

    head_acts: [N, num_layers, num_heads, head_dim]; labels: [N] (1=valid,0=invalid).
    Returns [num_layers*num_heads, head_dim]. (reused from honest_llama get_com_directions)
    """
    labels = np.asarray(labels)
    dirs = []
    for layer in range(num_layers):
        for head in range(num_heads):
            a = head_acts[:, layer, head, :]
            pos = a[labels == 1].mean(axis=0)
            neg = a[labels == 0].mean(axis=0)
            dirs.append(pos - neg)
    return np.array(dirs)


def rank_heads(head_acts, labels, num_layers, num_heads, seed=0):
    """Per-head logistic-probe val accuracy; returns sorted (acc, layer, head).

    (reused from honest_llama train_probes/get_top_heads, simplified to single split)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    labels = np.asarray(labels)
    accs = []
    for layer in range(num_layers):
        for head in range(num_heads):
            X = head_acts[:, layer, head, :]
            Xtr, Xva, ytr, yva = train_test_split(
                X, labels, test_size=0.3, random_state=seed, stratify=labels)
            clf = LogisticRegression(random_state=seed, max_iter=1000).fit(Xtr, ytr)
            accs.append((clf.score(Xva, yva), layer, head))
    return sorted(accs, key=lambda t: t[0], reverse=True)


def select_top_heads(head_acts, labels, num_layers, num_heads, k=48, alpha=15.0, seed=0):
    """Pick top-K heads by separability; build {(layer,head): (direction, scale)}."""
    ranked = rank_heads(head_acts, labels, num_layers, num_heads, seed)
    top = ranked[:k]
    coms = com_directions(head_acts, labels, num_layers, num_heads)
    labels = np.asarray(labels)
    interventions = {}
    for acc, layer, head in top:
        d = coms[layer_head_to_flattened_idx(layer, head, num_heads)]
        d = d / (np.linalg.norm(d) + 1e-8)
        proj = head_acts[:, layer, head, :] @ d
        interventions[(layer, head)] = (d.astype(np.float32),
                                        float(alpha * proj.std()))
    return interventions


# --------------------------------------------------------------------------- #
# Activation collection + intervention (architecture-agnostic via o_proj hooks)
# --------------------------------------------------------------------------- #
def _num_heads(model):
    cfg = model.config
    return getattr(cfg, "num_attention_heads")


def _num_layers(model):
    return getattr(model.config, "num_hidden_layers")


def collect_head_acts(model, tokenizer, texts, device="cuda"):
    """Return [N, num_layers, num_heads, head_dim] last-token attn-head outputs.

    Hooks the INPUT to each layer's self_attn.o_proj (= concatenated per-head outputs).
    """
    import torch
    L, H = _num_layers(model), _num_heads(model)
    hd = model.config.hidden_size // H
    captured = {}

    def mk_hook(li):
        def hook(module, inp, out):
            captured[li] = inp[0][0, -1].detach().float().cpu()  # [hidden]
        return hook

    handles = []
    for li, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.o_proj.register_forward_hook(mk_hook(li)))

    acts = np.zeros((len(texts), L, H, hd), dtype=np.float32)
    model.eval()
    for n, text in enumerate(texts):
        enc = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**enc)
        for li in range(L):
            acts[n, li] = captured[li].reshape(H, hd).numpy()
    for h in handles:
        h.remove()
    return acts


def fit(model_name="Qwen/Qwen3-8B", val_path=DEFAULT_VAL, out_dir=DEFAULT_DIR,
        k=48, alpha=15.0, device="cuda", seed=0):
    """Probe heads on LFUD val reps, save top-K directions+scales."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rows = [json.loads(l) for l in Path(val_path).read_text().splitlines() if l.strip()]
    texts, labels = [], []
    for r in rows:
        texts.append(f"{r['premise']} {r['valid_conclusion']}"); labels.append(1)
        texts.append(f"{r['premise']} {r['invalid_conclusion']}"); labels.append(0)

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    acts = collect_head_acts(model, tok, texts, device)
    interventions = select_top_heads(acts, labels, _num_layers(model),
                                     _num_heads(model), k=k, alpha=alpha, seed=seed)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    payload = {f"{l}_{h}": {"dir": d.tolist(), "scale": s}
               for (l, h), (d, s) in interventions.items()}
    (out_dir / f"iti_{model_name.replace('/', '-')}.json").write_text(
        json.dumps({"k": k, "alpha": alpha, "heads": payload}))
    print(f"[iti] fit {len(interventions)} heads -> {out_dir}")


class ITIBackend:
    """Base model with additive per-head shifts on selected heads (run_eval-compatible)."""

    def __init__(self, model_name, dirs_path, device="cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
        self.model.eval()
        self.device = device
        spec = json.loads(Path(dirs_path).read_text())
        H = _num_heads(self.model)
        hd = self.model.config.hidden_size // H
        # group shifts per layer into a single [hidden] additive vector
        self.layer_shift = {}
        for key, v in spec["heads"].items():
            layer, head = map(int, key.split("_"))
            vec = self.layer_shift.setdefault(layer, np.zeros(H * hd, dtype=np.float32))
            vec[head * hd:(head + 1) * hd] += np.asarray(v["dir"]) * v["scale"]
        self._handles = []
        self._register()

    def _register(self):
        import torch
        for li, layer in enumerate(self.model.model.layers):
            if li not in self.layer_shift:
                continue
            shift = torch.tensor(self.layer_shift[li], device=self.device,
                                 dtype=self.model.dtype)

            def pre_hook(module, args, shift=shift):
                x = args[0]
                x[:, -1] = x[:, -1] + shift  # shift last-token per-head outputs
                return (x,) + args[1:]
            self._handles.append(
                layer.self_attn.o_proj.register_forward_pre_hook(pre_hook))

    def generate(self, prompt, max_new_tokens=64, **kw):
        import torch
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens,
                                      do_sample=False, pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][enc.input_ids.shape[1]:],
                               skip_special_tokens=True).strip()

    def score_options(self, prompt, options, **kw):
        import torch
        scores = []
        for opt in options:
            full = prompt + " " + opt
            n_p = self.tok(prompt, return_tensors="pt").input_ids.shape[1]
            f = self.tok(full, return_tensors="pt").input_ids.to(self.device)
            with torch.no_grad():
                logits = self.model(f).logits.float()
            logp = torch.log_softmax(logits[0, :-1], dim=-1)
            tgt = f[0, 1:]
            tok_lp = logp[range(tgt.shape[0]), tgt][n_p - 1:]
            scores.append((tok_lp.sum() / max(1, tok_lp.numel())).item())
        return scores


def make_backend(model_name, ckpt=None):
    ckpt = ckpt or (DEFAULT_DIR / f"iti_{model_name.replace('/', '-')}.json")
    return ITIBackend(model_name, ckpt)


def _smoke():
    rng = np.random.default_rng(0)
    N, L, H, D = 60, 4, 8, 16
    # plant a separable direction in (layer1, head2) for the valid class
    acts = rng.normal(size=(N, L, H, D)).astype(np.float32)
    labels = np.array([1, 0] * (N // 2))
    acts[labels == 1, 1, 2, :] += 3.0  # valid reps offset on this head
    ranked = rank_heads(acts, labels, L, H, seed=0)
    top_acc, tl, th = ranked[0]
    print(f"[smoke] best head = (layer {tl}, head {th}) acc={top_acc:.3f} "
          f"(expected layer 1 head 2)")
    assert (tl, th) == (1, 2), (tl, th)
    iv = select_top_heads(acts, labels, L, H, k=4, alpha=10.0)
    (l, h), (d, s) = next(iter(iv.items()))
    print(f"[smoke] top intervention head=({l},{h}) |dir|={np.linalg.norm(d):.3f} "
          f"scale={s:.3f}")
    assert abs(np.linalg.norm(d) - 1.0) < 1e-4
    print("[smoke] ITI probe + COM-direction math OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--val-path", default=str(DEFAULT_VAL))
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--alpha", type=float, default=15.0)
    args = ap.parse_args()
    if args.smoke:
        _smoke()
    else:
        fit(model_name=args.model, val_path=args.val_path, k=args.k, alpha=args.alpha)
