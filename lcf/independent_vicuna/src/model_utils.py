"""Shared helpers: load a 4-bit LLM, find attn/MLP tap points, register hooks.

Targets the Llama family (Llama2/Llama3/Mistral/Vicuna) whose decoder layers
live at `model.model.layers[l]` with `.self_attn` and `.mlp` submodules.
ChatGLM3/Baichuan use different attribute names; extend `get_decoder_layers`
if you reproduce those too.
"""
from __future__ import annotations
import torch


def load_llm(cfg):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(cfg.model_name, cache_dir=str(cfg.hf_cache))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = dict(cache_dir=str(cfg.hf_cache), torch_dtype=torch.float16, device_map="auto")
    if cfg.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **kwargs)
    model.eval()
    return model, tok


def get_decoder_layers(model):
    m = model
    for attr in ("model", "transformer", "gpt_neox"):
        if hasattr(m, attr):
            m = getattr(m, attr)
            break
    for attr in ("layers", "h", "encoder"):
        if hasattr(m, attr):
            return getattr(m, attr)
    raise AttributeError("Could not locate decoder layers for this architecture")


def get_taps(model, lo, hi, tap_points=("attn", "mlp")):
    """Return {(layer_idx, kind): submodule} for layers in [lo, hi)."""
    layers = get_decoder_layers(model)
    taps = {}
    for l in range(lo, min(hi, len(layers))):
        if "attn" in tap_points:
            taps[(l, "attn")] = layers[l].self_attn
        if "mlp" in tap_points:
            taps[(l, "mlp")] = layers[l].mlp
    return taps


def _split_out(out):
    """Module output may be a Tensor or a tuple; return (hidden, rebuild_fn)."""
    if isinstance(out, tuple):
        return out[0], (lambda h: (h,) + tuple(out[1:]))
    return out, (lambda h: h)


class Capture:
    """Forward hooks that stash each tap's hidden states for one forward pass."""

    def __init__(self):
        self.store = {}
        self.handles = []

    def register(self, taps):
        for key, mod in taps.items():
            self.handles.append(mod.register_forward_hook(self._mk(key)))
        return self

    def _mk(self, key):
        def fn(mod, inp, out):
            h, _ = _split_out(out)
            self.store[key] = h.detach()
        return fn

    def clear(self):
        self.store = {}

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


class Steerer:
    """Forward hooks that apply lcf.modify() to the chosen taps during inference.

    Toggle `.active` to compare baseline vs +LCF without re-registering hooks.
    `sign=-1` performs the paper's 'invalid modification' (bidirectional control).
    """

    def __init__(self, lcf, taps, eta, sign=1.0):
        self.lcf, self.eta, self.sign = lcf, eta, sign
        self.active = True
        self.handles = [m.register_forward_hook(self._mk()) for m in taps.values()]

    def _mk(self):
        def fn(mod, inp, out):
            if not self.active:
                return out
            h, rebuild = _split_out(out)
            h2 = self.lcf.modify(h.float(), self.eta, self.sign).to(h.dtype)
            return rebuild(h2)
        return fn

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []
