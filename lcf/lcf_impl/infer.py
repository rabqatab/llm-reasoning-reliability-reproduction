"""Inference wrapper: apply a trained LCF as forward hooks on a frozen HF model.

The LCF is applied to the OUTPUTS of the selected sub-layers (attn / mlp of layers
in config.selected_sublayers). During forward/generate, each token rep R_input at
those sub-layers is replaced by R_input+ = LCF(R_input, eta, sign).

  eta = 0.5  -> conclusion generation (gentle nudge)
  eta = 4.5  -> fallacy identification (strong nudge)
  sign = +1  -> push toward VALID logic ; sign = -1 -> push toward INVALID

Public API (imported by the eval agent):
  wrapper = LCFInference(model_name, ckpt_dir, device)
  wrapper.generate_with_lcf(prompt, eta=0.5, sign=+1, **gen_kwargs) -> str
  wrapper.score_options_with_lcf(prompt, options, eta=4.5, sign=+1) -> list[float]
      (per-option total sequence logprob; eval agent uses for ΔProb / Accuracy)
  wrapper.set_lcf_enabled(bool)   # toggle hooks (baseline = original model)
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

try:  # works when cwd == lcf_impl (scripts) ...
    from model import build_lcf
except ImportError:  # ... and when imported as a package (eval harness)
    from lcf.lcf_impl.model import build_lcf

ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "checkpoints"


class LCFInference:
    def __init__(self, model_name, ckpt_dir=None, device=None, dtype=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32)

        short = model_name.split("/")[-1]
        ckpt_dir = Path(ckpt_dir) if ckpt_dir else CKPT_DIR / short
        with open(ckpt_dir / "config.json") as f:
            self.config = json.load(f)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=self.dtype)
        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        d = self.config["d"]
        self.lcf = build_lcf(d=d, dims=tuple(self.config["dims"]))
        state = torch.load(ckpt_dir / "lcf.pt", map_location="cpu", weights_only=True)
        self.lcf.load_state_dict(state)
        self.lcf.to(self.device).to(self.dtype).eval()

        self._enabled = True
        self._eta = self.config["eta_defaults"]["generation"]
        self._sign = 1
        self.handles = []
        self._install_hooks()

    # ---------------------------------------------------------------- hooks
    def _decoder_layers(self):
        m = self.model
        for attr in ("model", "transformer"):
            if hasattr(m, attr):
                m = getattr(m, attr)
        if hasattr(m, "layers"):
            return m.layers
        if hasattr(m, "h"):
            return m.h
        raise RuntimeError("could not locate decoder layers")

    def _install_hooks(self):
        layers = self._decoder_layers()
        targets = {(s["layer"], s["kind"]) for s in self.config["selected_sublayers"]}
        for li, layer in enumerate(layers):
            for kind, attr in (("attn", "self_attn"), ("mlp", "mlp")):
                if (li, kind) not in targets:
                    continue
                sub = getattr(layer, attr, None)
                if sub is None:
                    continue
                self.handles.append(sub.register_forward_hook(self._mk_hook()))

    def _mk_hook(self):
        def hook(_module, _inp, out):
            if not self._enabled:
                return out
            is_tuple = isinstance(out, tuple)
            t = out[0] if is_tuple else out  # (B,T,d)
            orig_dtype = t.dtype
            B, T, d = t.shape
            flat = t.reshape(B * T, d).to(self.lcf.C_pos.dtype)
            mod = self.lcf(flat, eta=self._eta, sign=self._sign)
            mod = mod.reshape(B, T, d).to(orig_dtype)
            if is_tuple:
                return (mod,) + tuple(out[1:])
            return mod
        return hook

    def set_lcf_enabled(self, flag: bool):
        self._enabled = bool(flag)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    # ---------------------------------------------------------------- API
    @torch.no_grad()
    def generate_with_lcf(self, prompt, eta=None, sign=1, max_new_tokens=128,
                          do_sample=False, **gen_kwargs):
        self._eta = eta if eta is not None else self.config["eta_defaults"]["generation"]
        self._sign = sign
        self._enabled = True
        ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **ids, max_new_tokens=max_new_tokens, do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id, **gen_kwargs)
        text = self.tokenizer.decode(out[0][ids.input_ids.shape[1]:],
                                     skip_special_tokens=True)
        return text

    @torch.no_grad()
    def score_options_with_lcf(self, prompt, options, eta=None, sign=1):
        """Return per-option total sequence logprob of option given prompt.

        Used by the eval agent for ΔProb / Accuracy. eta defaults to the
        identification value (4.5).
        """
        self._eta = eta if eta is not None else self.config["eta_defaults"]["identification"]
        self._sign = sign
        self._enabled = True
        scores = []
        for opt in options:
            scores.append(self._seq_logprob(prompt, opt))
        return scores

    @torch.no_grad()
    def _seq_logprob(self, prompt, continuation):
        p_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0]
        full = self.tokenizer(prompt + " " + continuation, return_tensors="pt").input_ids[0]
        cont_ids = full[len(p_ids):]
        if len(cont_ids) == 0:
            return float("-inf")
        inp = full.unsqueeze(0).to(self.device)
        logits = self.model(inp).logits[0]  # (T, V)
        logprobs = F.log_softmax(logits.float(), dim=-1)
        total = 0.0
        # token at position i is predicted by logits at i-1
        for k, tok in enumerate(cont_ids):
            pos = len(p_ids) + k - 1
            total += logprobs[pos, int(tok)].item()
        return total


def load_inference(model_name="Qwen/Qwen3-8B", ckpt_dir=None, **kw):
    return LCFInference(model_name, ckpt_dir=ckpt_dir, **kw)


# ---- module-level API expected by the eval harness ----------------------
# A lazily-built singleton; configured via env LCF_EVAL_MODEL / LCF_EVAL_CKPT
# (both default to the Qwen3-8B checkpoint under lcf/checkpoints/).
_SINGLETON = None


def _get_singleton():
    global _SINGLETON
    if _SINGLETON is None:
        import os
        model = os.environ.get("LCF_EVAL_MODEL", "Qwen/Qwen3-8B")
        ckpt = os.environ.get("LCF_EVAL_CKPT") or None
        _SINGLETON = LCFInference(model, ckpt_dir=ckpt)
    return _SINGLETON


def generate_with_lcf(prompt, eta=0.5, sign=1, **kw):
    return _get_singleton().generate_with_lcf(prompt, eta=eta, sign=sign, **kw)


def score_options_with_lcf(prompt, options, eta=4.5, sign=1):
    return _get_singleton().score_options_with_lcf(prompt, options, eta=eta, sign=sign)
