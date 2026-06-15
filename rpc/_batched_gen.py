"""Batched K-sample generation with per-sequence mean log-prob (RPC format).

Replaces the slow `for _ in range(K): model.generate(...)` loop with a SINGLE
batched generate (num_return_sequences via input expand) -> ~Kx faster. Returns
the K completion texts and their mean per-token log-probs.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


@torch.no_grad()
def sample_k(model, tok, input_ids, K, max_new_tokens, temperature=0.8, top_p=0.95):
    plen = input_ids.shape[1]
    batch = input_ids.expand(K, -1).contiguous()
    out = model.generate(
        batch, do_sample=True, temperature=temperature, top_p=top_p,
        max_new_tokens=max_new_tokens, num_return_sequences=1,
        return_dict_in_generate=True, output_scores=True,
        pad_token_id=tok.pad_token_id,
    )
    # out.sequences: [K, plen+gen]; out.scores: tuple(len gen) of [K, vocab]
    logprobs = F.log_softmax(torch.stack(out.scores, 1).float(), dim=-1)  # [K, gen, vocab]
    gen = out.sequences[:, plen:]                                         # [K, gen]
    eos = tok.eos_token_id
    texts, mlps = [], []
    for k in range(K):
        g = gen[k]
        lps = []
        for t in range(g.shape[0]):
            tokid = int(g[t])
            lps.append(logprobs[k, t, tokid].item())
            if tokid == eos:           # include up to first EOS, then stop (rest is pad)
                break
        texts.append(tok.decode(g, skip_special_tokens=True))
        mlps.append(sum(lps) / len(lps) if lps else float("-inf"))
    return texts, mlps
