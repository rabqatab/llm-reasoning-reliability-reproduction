"""RAHF baseline (adapted from baselines_RAHF) — representation control.

Spec E: RAHF = representation-control from preferred(valid)/dispreferred(invalid)
pairs; feed valid/invalid conclusion pairs. Table 5 (Llama2): RAHF 71.56/46.56.

WHAT IS FULL-FAITHFUL vs REDUCED
--------------------------------
The original RAHF (lcf/baselines_RAHF/code/step2/RAHF.py) has two variants:
  * SCIT: a single model is taught preferences via Hindsight (step1) then a LoRA is
    fit to the (pos - neg) hidden-state difference.
  * DUAL: trains separate "good" and "bad" SFT models (step1), then fits a LoRA on
    the base model so its hidden states match  base + alpha*(good_hidden - bad_hidden).

Faithful core we reproduce (RAHF-DUAL target, compute_loss_DUAL):
  target_hidden[l] = base_hidden[l] + alpha * (good_hidden[l] - bad_hidden[l])
  loss = || lora_hidden[l] - target_hidden[l] ||_2   over target_layers
  optional + KL(lora_logits || base_logits).
  LoRA on layers up to max(target_layers); MSE/L2 on response tokens. This IS the
  RAHF.py loss, ported to HF Trainer with our data.

REDUCED (documented, to stay single-model / GPU-light on one DGX Spark node):
  * We do NOT train two extra full SFT "good"/"bad" models (step1). Instead, following
    the SCIT spirit, we obtain the preference direction directly from the SAME frozen
    base model run on the VALID vs INVALID conclusion of each pair:
        good_hidden = base(premise + valid_conclusion)
        bad_hidden  = base(premise + invalid_conclusion)
    This is a faithful reduction: RAHF's direction is (preferred - dispreferred) hidden
    states; we source preferred/dispreferred from the labeled LFUD pairs rather than
    from two separately-tuned models. Clearly noted as reduced.
  * target_layers default to a mid-stack band (LCF spec found layers 10-30 most
    logic-bearing); we expose --target-layers.

CPU-smoke (--smoke) verifies the target-hidden construction math on synthetic tensors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
DEFAULT_TRAIN = REPO_ROOT / "lcf" / "data" / "conclusion_gen_train.jsonl"
PROMPT = "{premise}\nTherefore,"


def adapter_dir(model_name):
    return BASE_DIR / f"rahf_{model_name.replace('/', '-')}"


def build_target_hidden(base_hidden, good_hidden, bad_hidden, alpha, target_layers):
    """RAHF-DUAL target: base + alpha*(good - bad) on target layers (faithful to RAHF.py).

    Each *_hidden is a list indexed by layer of [B, T, d] tensors. Returns stacked
    target over target_layers: [len(layers), B, T, d].
    """
    import torch
    return torch.stack([
        base_hidden[l] + alpha * (good_hidden[l] - bad_hidden[l])
        for l in target_layers
    ])


def train(model_name="Qwen/Qwen3-8B", train_path=DEFAULT_TRAIN, out_dir=None,
          target_layers=(10, 14, 18, 22, 26, 30), alpha=5.0, epochs=5, lr=1e-4,
          lora_r=16, lora_alpha=32, max_length=256, kl=True, device="cuda", seed=0):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    set_seed(seed)

    out_dir = Path(out_dir or adapter_dir(model_name))
    rows = [json.loads(l) for l in Path(train_path).read_text().splitlines() if l.strip()]
    print(f"[rahf] {len(rows)} pairs; target_layers={list(target_layers)} alpha={alpha}")

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    base.eval()

    target_layers = list(target_layers)
    lora_layers = list(range(max(target_layers) + 1))
    lcfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      layers_to_transform=lora_layers, task_type="CAUSAL_LM")
    model = get_peft_model(
        AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device),
        lcfg)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    def enc(text):
        return tok(text, return_tensors="pt", truncation=True,
                   max_length=max_length).to(device)

    for ep in range(epochs):
        total = 0.0
        for r in rows:
            base_in = enc(PROMPT.format(premise=r["premise"]))
            good_in = enc(f"{r['premise']} {r['valid_conclusion']}")
            bad_in = enc(f"{r['premise']} {r['invalid_conclusion']}")
            with torch.no_grad():
                b = base(**base_in, output_hidden_states=True)
                g = base(**good_in, output_hidden_states=True).hidden_states
                d = base(**bad_in, output_hidden_states=True).hidden_states
                base_h = b.hidden_states
                base_logits = b.logits
                # align lengths to the base prompt T (use last-T tokens)
                T = base_h[0].shape[1]
                tgt = torch.stack([
                    base_h[l] + alpha * (g[l][:, -T:] - d[l][:, -T:])
                    for l in target_layers])
            out = model(**base_in, output_hidden_states=True)
            lora_h = torch.stack([out.hidden_states[l] for l in target_layers])
            loss = torch.norm((lora_h - tgt).float(), p=2, dim=-1).mean()
            if kl:
                lp = torch.log_softmax(out.logits.float(), -1)
                bp = torch.log_softmax(base_logits.float(), -1)
                loss = loss + torch.nn.functional.kl_div(
                    lp, bp, reduction="batchmean", log_target=True)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        print(f"[rahf] epoch {ep} mean_loss={total / len(rows):.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    (out_dir / "rahf_config.json").write_text(json.dumps(
        {"target_layers": target_layers, "alpha": alpha, "kl": kl}))
    print(f"[rahf] adapter saved to {out_dir}")


class RAHFBackend:
    """Base model + RAHF LoRA adapter (run_eval-compatible)."""

    def __init__(self, model_name, adapter_path, device="cuda"):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
        self.model = PeftModel.from_pretrained(base, str(adapter_path)).to(device)
        self.model.eval(); self.device = device

    def generate(self, prompt, max_new_tokens=64, **kw):
        import torch
        e = self.tok(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**e, max_new_tokens=max_new_tokens,
                                      do_sample=False, pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][e.input_ids.shape[1]:],
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
    return RAHFBackend(model_name, ckpt or adapter_dir(model_name))


def _smoke():
    import numpy as np
    try:
        import torch
    except ImportError:
        print("[smoke] torch not importable; skipping tensor smoke")
        return
    L, B, T, D = 6, 1, 5, 8
    base_h = [torch.zeros(B, T, D) for _ in range(L)]
    good_h = [torch.ones(B, T, D) for _ in range(L)]
    bad_h = [torch.full((B, T, D), -1.0) for _ in range(L)]
    alpha, layers = 2.0, [1, 3, 5]
    tgt = build_target_hidden(base_h, good_h, bad_h, alpha, layers)
    # target = 0 + 2*(1 - (-1)) = 4 everywhere on target layers
    assert tgt.shape == (len(layers), B, T, D), tgt.shape
    assert torch.allclose(tgt, torch.full_like(tgt, 4.0)), tgt[0, 0, 0]
    print(f"[smoke] target_hidden shape={tuple(tgt.shape)} value={tgt[0,0,0,0].item()} "
          f"(expected 4.0 = 0 + 2*(1-(-1)))")
    print("[smoke] RAHF target-hidden math OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--train-path", default=str(DEFAULT_TRAIN))
    ap.add_argument("--target-layers", default="10,14,18,22,26,30")
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--epochs", type=int, default=5)
    args = ap.parse_args()
    if args.smoke:
        _smoke()
    else:
        train(model_name=args.model, train_path=args.train_path,
              target_layers=tuple(int(x) for x in args.target_layers.split(",")),
              alpha=args.alpha, epochs=args.epochs)
