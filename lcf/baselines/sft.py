"""SFT baseline: LoRA fine-tune the base LLM on the 540 valid conclusions.

Spec E: "fine-tune base LLM on 540 valid conclusions (premise -> valid). Strongest
simple baseline (+20% vs LCF +38%)." Table 5 (Llama2): SFT 79.90/78.43.

Trains a causal-LM LoRA adapter on (premise -> valid_conclusion) from
lcf/data/conclusion_gen_train.jsonl (produced by the other agent). Saves the adapter
to lcf/baselines/sft_<model>/. Exposes make_backend() with the SAME interface
run_eval expects: .generate(prompt) and .score_options(prompt, options).

Run on GPU via sparkq (see README). CPU-smoke the data/prompt formatting with --smoke.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
DEFAULT_TRAIN = REPO_ROOT / "lcf" / "data" / "conclusion_gen_train.jsonl"

# Same generation prompt used by run_eval so train/infer match.
SFT_PROMPT = "{premise}\nTherefore,"


def format_example(premise: str, valid_conclusion: str) -> dict:
    """Build a supervised (prompt, completion) pair from a premise/valid_conclusion."""
    prompt = SFT_PROMPT.format(premise=premise)
    # valid_conclusion typically starts with "Therefore," — strip to avoid doubling
    cont = valid_conclusion.strip()
    if cont.lower().startswith("therefore,"):
        cont = cont[len("therefore,"):].strip()
    return {"prompt": prompt, "completion": " " + cont}


def adapter_dir(model_name: str) -> Path:
    return BASE_DIR / f"sft_{model_name.replace('/', '-')}"


def train(model_name="Qwen/Qwen3-8B", train_path=DEFAULT_TRAIN, out_dir=None,
          epochs=10, lr=1e-4, batch_size=8, lora_r=16, lora_alpha=32,
          max_length=256, seed=0):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              DataCollatorForSeq2Seq, Trainer, TrainingArguments)

    out_dir = Path(out_dir or adapter_dir(model_name))
    rows = [json.loads(l) for l in Path(train_path).read_text().splitlines() if l.strip()]
    pairs = [format_example(r["premise"], r["valid_conclusion"]) for r in rows]
    print(f"[sft] {len(pairs)} training pairs from {train_path}")

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def tok_fn(ex):
        full = ex["prompt"] + ex["completion"]
        enc = tok(full, truncation=True, max_length=max_length)
        # mask the prompt tokens out of the loss
        p_len = len(tok(ex["prompt"], truncation=True, max_length=max_length).input_ids)
        labels = list(enc["input_ids"])
        for i in range(min(p_len, len(labels))):
            labels[i] = -100
        enc["labels"] = labels
        return enc

    ds = Dataset.from_list(pairs).map(tok_fn, remove_columns=["prompt", "completion"])

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
    lcfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(out_dir), num_train_epochs=epochs, learning_rate=lr,
        per_device_train_batch_size=batch_size, logging_steps=10, save_strategy="epoch",
        bf16=torch.cuda.is_available(), seed=seed, report_to=[])
    # Seq2Seq collator pads input_ids (pad token) AND our masked labels (-100),
    # preserving prompt masking; DataCollatorForLanguageModeling cannot pad
    # pre-set variable-length labels.
    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collator)
    trainer.train()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"[sft] adapter saved to {out_dir}")


class SFTBackend:
    """run_eval-compatible backend: base model + SFT LoRA adapter."""

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
        self.model.eval()
        self.device = device

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
    return SFTBackend(model_name, ckpt or adapter_dir(model_name))


def _smoke():
    sample = {"premise": "All men are mortal. Socrates is a man.",
              "valid_conclusion": "Therefore, Socrates is mortal."}
    ex = format_example(sample["premise"], sample["valid_conclusion"])
    print("[smoke] prompt    :", repr(ex["prompt"]))
    print("[smoke] completion:", repr(ex["completion"]))
    assert ex["completion"].strip().lower().startswith("socrates")
    print("[smoke] adapter dir:", adapter_dir("Qwen/Qwen3-8B"))
    print("[smoke] SFT formatting OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--train-path", default=str(DEFAULT_TRAIN))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    if args.smoke:
        _smoke()
    else:
        train(model_name=args.model, train_path=args.train_path,
              epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
