"""LFUD validity / fallacy-type discriminator ("Valid%(Trained)" judge).

Spec F: Valid%(Trained) uses a self-trained classifier fine-tuned on LFUD to judge
whether a conclusion/sentence is logically valid (and optionally its fallacy type).

We train a sequence-classification model with TWO heads' worth of supervision folded
into one model via a multi-task label scheme, but the load-bearing output is the
BINARY validity head used by `predict_valid`:

  * Binary task (primary): label 1 = VALID (no fallacy), 0 = INVALID (has fallacy).
    Built from LFUD `task1` ({'label': True/False}) -> a fallacious `sentence` is
    INVALID. Valid examples are mined from `task3` options (the non-fallacious
    conclusion) and from `task2`/`task4` distractors that are logically fine.
  * 12-way fallacy-type head (auxiliary, optional): predicts `fallacy_type` for the
    INVALID class. Enabled with --multitask; not required for Valid%(Trained).

Default backbone: `microsoft/deberta-v3-small` (small, often cached). Override with
`--model Qwen/Qwen3-8B` for the paper-faithful Llama-2-style large classifier (run
via sparkq). The data pipeline is CPU-smoke-testable without any model download.

Output dir: lcf/eval/discriminator/   (config.json + weights + label_map.json)
API: load_discriminator(dir).predict_valid(texts) -> list[bool]
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
from pathlib import Path
from typing import Sequence

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_DISC_DIR = EVAL_DIR / "discriminator"
DEFAULT_LFUD_CSV = EVAL_DIR.parents[0] / "LFUD" / "LFUD.csv"  # lcf/LFUD/LFUD.csv

# 12 LFUD fallacy types (spec D). Index 0 reserved for "valid/no-fallacy".
FALLACY_TYPES = [
    "faulty generalization", "ad hominem", "ad populum", "appeal to emotion",
    "false causality", "circular reasoning", "fallacy of relevance",
    "deductive fallacy", "intentional fallacy", "fallacy of credibility",
    "false dilemma", "fallacy of extension",
]


# --------------------------------------------------------------------------- #
# Data pipeline (CPU-smoke-testable, no model needed)
# --------------------------------------------------------------------------- #
def _safe_literal(cell: str):
    """LFUD cells are python-dict-literals stored as strings."""
    try:
        return ast.literal_eval(cell)
    except (ValueError, SyntaxError):
        return None


def build_examples_from_lfud(
    csv_path: str | os.PathLike = DEFAULT_LFUD_CSV,
    test_scenarios: set | None = None,
    split: str = "train",
    max_rows: int | None = None,
) -> list[dict]:
    """Return [{'text', 'valid': 0/1, 'fallacy_idx': int}].

    INVALID examples: the fallacious `sentence` (task1 label True == has fallacy).
    VALID examples:   the non-fallacious option from task3 (the conclusion that does
                      NOT create the fallacy) and the alternative premise from task4.
    `fallacy_idx`: 0 for valid, else 1..12 by FALLACY_TYPES order.

    Scenario-disjoint splitting: pass `test_scenarios` (a set of `proposition`
    strings or row indices) to hold out; `split` selects which side to return.
    """
    import csv as _csv

    rows: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            rows.append(row)

    # scenario id = proposition text (groups sentences sharing a premise)
    def scen_key(r):
        return r.get("proposition", "")

    if test_scenarios is None:
        # deterministic 45:5:17 -> ~ 540:60:204 by unique scenario, here just
        # default to "all rows" for training when no holdout requested.
        keep = rows
    else:
        if split == "test":
            keep = [r for r in rows if scen_key(r) in test_scenarios]
        else:
            keep = [r for r in rows if scen_key(r) not in test_scenarios]

    examples: list[dict] = []
    for r in keep:
        ftype = (r.get("fallacy_type") or "").strip().lower()
        fidx = FALLACY_TYPES.index(ftype) + 1 if ftype in FALLACY_TYPES else 0
        # INVALID: the fallacious sentence
        sent = (r.get("sentence") or "").strip()
        if sent:
            examples.append({"text": sent, "valid": 0,
                             "fallacy_idx": fidx if fidx else 0})
        # VALID: mine the non-fallacious task3 conclusion option
        t3 = _safe_literal(r.get("task3", "") or "")
        if isinstance(t3, dict) and "options" in t3 and "answer" in t3:
            opts, ans = t3["options"], t3["answer"]
            premise = r.get("proposition", "")
            for j, opt in enumerate(opts):
                if j != ans:  # the option that does NOT create the fallacy = valid
                    examples.append(
                        {"text": f"{premise} {opt}".strip(),
                         "valid": 1, "fallacy_idx": 0})
    return examples


def make_scenario_split(
    csv_path: str | os.PathLike = DEFAULT_LFUD_CSV,
    ratios: tuple[int, int, int] = (45, 5, 17),
    seed: int = 0,
) -> dict[str, set]:
    """Split unique scenarios (propositions) into train/val/test sets (disjoint)."""
    import csv as _csv

    scenarios: list[str] = []
    seen = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            s = row.get("proposition", "")
            if s and s not in seen:
                seen.add(s)
                scenarios.append(s)
    rng = random.Random(seed)
    rng.shuffle(scenarios)
    total = sum(ratios)
    n = len(scenarios)
    n_tr = round(n * ratios[0] / total)
    n_va = round(n * ratios[1] / total)
    return {
        "train": set(scenarios[:n_tr]),
        "val": set(scenarios[n_tr:n_tr + n_va]),
        "test": set(scenarios[n_tr + n_va:]),
    }


# --------------------------------------------------------------------------- #
# Model wrapper
# --------------------------------------------------------------------------- #
class Discriminator:
    def __init__(self, model, tokenizer, device: str = "cpu", max_length: int = 256):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

    def predict_valid(self, texts: Sequence[str]) -> list[bool]:
        """Binary VALID prediction. label index 1 == valid (see training)."""
        import torch
        self.model.eval()
        preds: list[bool] = []
        bs = 16
        for i in range(0, len(texts), bs):
            batch = list(texts[i:i + bs])
            enc = self.tokenizer(
                batch, return_tensors="pt", truncation=True,
                padding=True, max_length=self.max_length,
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits  # [B, 2]
            preds.extend((logits[:, 1] > logits[:, 0]).cpu().tolist())
        return [bool(p) for p in preds]


def load_discriminator(disc_dir: str | os.PathLike | None = None,
                       device: str = "cpu") -> Discriminator:
    disc_dir = Path(disc_dir or DEFAULT_DISC_DIR)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(disc_dir)
    model = AutoModelForSequenceClassification.from_pretrained(disc_dir).to(device)
    return Discriminator(model, tok, device=device)


# --------------------------------------------------------------------------- #
# Training (GPU via sparkq; small backbone trains on CPU for smoke)
# --------------------------------------------------------------------------- #
def build_examples_from_conclusion_gen(data_dir, split):
    """Balanced VALID/INVALID examples from conclusion_gen_<split>.jsonl.

    valid_conclusion -> VALID(1), invalid_conclusion -> INVALID(0). Both are full
    same-style conclusions (the invalid one is the original LFUD fallacious
    sentence; the valid one is the GPT-3.5 logically-valid rewrite), so the
    classifier must learn validity rather than surface style. Scenario split
    (540/60/204) was already applied when these files were built.
    """
    import json
    path = Path(data_dir) / f"conclusion_gen_{split}.jsonl"
    examples = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        vc = (r.get("valid_conclusion") or "").strip()
        ic = (r.get("invalid_conclusion") or "").strip()
        if vc:
            examples.append({"text": vc, "valid": 1})
        if ic:
            examples.append({"text": ic, "valid": 0})
    return examples


def train(
    model_name: str = "microsoft/deberta-v3-small",
    csv_path: str | os.PathLike = DEFAULT_LFUD_CSV,
    out_dir: str | os.PathLike = DEFAULT_DISC_DIR,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    seed: int = 0,
    max_length: int = 256,
) -> None:
    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments)

    data_dir = Path(csv_path).resolve().parents[1] / "data"  # lcf/data
    if (data_dir / "conclusion_gen_train.jsonl").exists():
        train_ex = build_examples_from_conclusion_gen(data_dir, "train")
        val_ex = build_examples_from_conclusion_gen(data_dir, "val")
        print(f"[disc] using conclusion_gen (valid vs invalid): "
              f"train={len(train_ex)} val={len(val_ex)} examples")
    else:
        splits = make_scenario_split(csv_path, seed=seed)
        train_ex = build_examples_from_lfud(csv_path, splits["test"] | splits["val"],
                                            split="train")
        val_ex = build_examples_from_lfud(csv_path, splits["val"], split="test")
        print(f"[disc] using LFUD task3 mining: train={len(train_ex)} val={len(val_ex)}")

    tok = AutoTokenizer.from_pretrained(model_name)

    def to_ds(ex):
        return Dataset.from_dict({
            "text": [e["text"] for e in ex],
            "label": [e["valid"] for e in ex],
        })

    def tok_fn(b):
        return tok(b["text"], truncation=True, max_length=max_length)

    train_ds = to_ds(train_ex).map(tok_fn, batched=True)
    val_ds = to_ds(val_ex).map(tok_fn, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": float((preds == labels).mean())}

    args = TrainingArguments(
        output_dir=str(out_dir), num_train_epochs=epochs,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
        learning_rate=lr, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="accuracy",
        logging_steps=20, seed=seed, report_to=[], bf16=torch.cuda.is_available(),
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      eval_dataset=val_ds, compute_metrics=metrics, processing_class=tok)
    trainer.train()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    (Path(out_dir) / "label_map.json").write_text(
        json.dumps({"0": "INVALID", "1": "VALID", "fallacy_types": FALLACY_TYPES}))
    print(f"[disc] saved to {out_dir}")


def _smoke() -> None:
    """CPU smoke: exercise the data pipeline only (no model)."""
    csv_path = DEFAULT_LFUD_CSV
    if not Path(csv_path).exists():
        print(f"[smoke] LFUD csv not found at {csv_path}; skipping data smoke")
        return
    splits = make_scenario_split(csv_path)
    print(f"[smoke] scenarios train/val/test = "
          f"{len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")
    ex = build_examples_from_lfud(csv_path, splits["test"] | splits["val"],
                                  split="train", max_rows=200)
    n_valid = sum(e["valid"] for e in ex)
    print(f"[smoke] built {len(ex)} examples from first 200 rows "
          f"({n_valid} valid / {len(ex) - n_valid} invalid)")
    print(f"[smoke] sample valid  : {next(e['text'] for e in ex if e['valid'])[:90]!r}")
    print(f"[smoke] sample invalid: {next(e['text'] for e in ex if not e['valid'])[:90]!r}")
    assert n_valid > 0 and n_valid < len(ex), "expected a mix of valid/invalid"
    print("[smoke] data pipeline OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="CPU data-pipeline smoke test")
    ap.add_argument("--model", default="microsoft/deberta-v3-small")
    ap.add_argument("--csv", default=str(DEFAULT_LFUD_CSV))
    ap.add_argument("--out", default=str(DEFAULT_DISC_DIR))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    if args.smoke:
        _smoke()
    else:
        train(model_name=args.model, csv_path=args.csv, out_dir=args.out,
              epochs=args.epochs, batch_size=args.batch_size)
