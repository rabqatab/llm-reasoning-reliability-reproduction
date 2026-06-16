"""Extract paired hidden representations R_input+ / R_input- for LCF training.

For each (premise, valid, invalid) triple:
  * run the LLM on  premise+valid  and  premise+invalid  (capturing attn/MLP taps
    in layers [lo, hi))
  * align identical tokens between the two conclusions (difflib)
  * for each aligned token, sample `layers_per_pair` taps and record the matched
    (valid_vec, invalid_vec) — same content, opposite logical validity.

Outputs (gitignored, under SCRATCH):
  hidden/train_reps.pt : {valid:(M,d), invalid:(M,d), layer:(M,), kind:(M,), tok:(M,)}
  hidden/val_reps.pt   : {(layer,kind): {valid:(n,d), invalid:(n,d)}}  (all taps; for distinctiveness)

Usage: python src/extract.py            # train + val
       python src/extract.py --limit 8  # smoke test
"""
from __future__ import annotations
import argparse, json, random, sys
from difflib import SequenceMatcher
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CFG
from model_utils import load_llm, get_taps, Capture


def build_ids(tok, premise, conclusion):
    pre = tok(premise, return_tensors="pt").input_ids[0]
    con = tok(conclusion, add_special_tokens=False, return_tensors="pt").input_ids[0]
    return torch.cat([pre, con]), len(pre), con  # full_ids, prefix_len, conclusion_ids


@torch.no_grad()
def tap_states(model, cap, ids):
    cap.clear()
    model(input_ids=ids.unsqueeze(0).to(model.device))
    # {(l,kind): (T, d) on cpu float16}
    return {k: v[0].to("cpu", torch.float16) for k, v in cap.store.items()}


def extract(model, tok, cap, pairs, valid_map, tap_keys, rng, random_taps, max_tok_pairs=None):
    rec_valid, rec_invalid, rec_layer, rec_kind, rec_tok = [], [], [], [], []
    val_per_tap = {k: {"valid": [], "invalid": []} for k in tap_keys}
    kept = 0
    for p in pairs:
        valid_text = valid_map.get(str(p["index"]))
        if not valid_text:
            continue
        v_ids, v_pre, v_con = build_ids(tok, p["premise"], valid_text)
        i_ids, i_pre, i_con = build_ids(tok, p["premise"], p["invalid"])
        # align identical conclusion tokens
        blocks = SequenceMatcher(None, v_con.tolist(), i_con.tolist()).get_matching_blocks()
        matches = [(v_pre + b.a + k, i_pre + b.b + k, int(v_con[b.a + k]))
                   for b in blocks for k in range(b.size)]
        if not matches:
            continue
        if max_tok_pairs:
            rng.shuffle(matches); matches = matches[:max_tok_pairs]
        v_states = tap_states(model, cap, v_ids)
        i_states = tap_states(model, cap, i_ids)
        for vpos, ipos, tid in matches:
            taps = rng.sample(tap_keys, CFG.layers_per_pair) if random_taps else tap_keys
            for (l, kind) in taps:
                vv = v_states[(l, kind)][vpos]
                iv = i_states[(l, kind)][ipos]
                if random_taps:
                    rec_valid.append(vv); rec_invalid.append(iv)
                    rec_layer.append(l); rec_kind.append(0 if kind == "attn" else 1); rec_tok.append(tid)
                else:
                    val_per_tap[(l, kind)]["valid"].append(vv)
                    val_per_tap[(l, kind)]["invalid"].append(iv)
        kept += 1
    if random_taps:
        return {
            "valid": torch.stack(rec_valid), "invalid": torch.stack(rec_invalid),
            "layer": torch.tensor(rec_layer), "kind": torch.tensor(rec_kind),
            "tok": torch.tensor(rec_tok),
        }, kept
    else:
        out = {}
        for k, d in val_per_tap.items():
            if d["valid"]:
                out[k] = {"valid": torch.stack(d["valid"]), "invalid": torch.stack(d["invalid"])}
        return out, kept


def load_pairs(split):
    return [json.loads(l) for l in open(CFG.data_dir / f"pairs_{split}.jsonl")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    valid_map = json.load(open(CFG.data_dir / "valid_conclusions.json"))
    model, tok = load_llm(CFG)
    taps = get_taps(model, CFG.layer_lo, CFG.layer_hi, CFG.tap_points)
    tap_keys = list(taps.keys())
    cap = Capture().register(taps)
    rng = random.Random(CFG.seed)

    train_pairs = load_pairs("train")
    val_pairs = load_pairs("val")
    if args.limit:
        train_pairs, val_pairs = train_pairs[: args.limit], val_pairs[: max(2, args.limit // 4)]

    print(f"extracting train reps from {len(train_pairs)} pairs ...")
    train_data, kt = extract(model, tok, cap, train_pairs, valid_map, tap_keys, rng, random_taps=True)
    torch.save(train_data, CFG.hidden_dir / "train_reps.pt")
    print(f"  kept {kt} pairs -> {train_data['valid'].shape[0]} rep-pairs "
          f"saved to {CFG.hidden_dir/'train_reps.pt'}")

    print(f"extracting val reps (all taps) from {len(val_pairs)} pairs ...")
    val_data, kv = extract(model, tok, cap, val_pairs, valid_map, tap_keys, rng,
                           random_taps=False, max_tok_pairs=8)
    torch.save(val_data, CFG.hidden_dir / "val_reps.pt")
    print(f"  kept {kv} pairs across {len(val_data)} taps -> {CFG.hidden_dir/'val_reps.pt'}")
    cap.remove()


if __name__ == "__main__":
    main()
