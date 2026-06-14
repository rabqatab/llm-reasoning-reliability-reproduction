"""LCF training losses (spec section B, Eq.6-13).

Total: L = L_rec + L_logic+ + L_logic- + L_content

Conventions:
  - A "pair" is (R_input_plus, R_input_minus): same content, opposite logic.
    R_input_plus is the VALID-conclusion token rep (label 1),
    R_input_minus is the INVALID-conclusion token rep (label 0).
  - All reps are (B, d). Projections are (B, proj).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rec_loss(lcf, R_input: torch.Tensor) -> torch.Tensor:
    """Eq.6-7: R_hat = Decoder(content, UN-modified logic); MSE(R_input, R_hat)."""
    R_content, R_logic = lcf.encode(R_input)
    R_hat = lcf.reconstruct(R_content, R_logic)
    return F.mse_loss(R_hat, R_input)


def _supcon(logic: torch.Tensor, labels: torch.Tensor, target_label: int,
            tau: float = 0.1) -> torch.Tensor:
    """SupCon-style InfoNCE over {valid, invalid} logic reps.

    For each anchor whose label == target_label, positives are the OTHER samples
    sharing target_label; negatives are everything else. Cosine similarity / tau.
    Returns mean over valid anchors (0 if none).
    """
    z = F.normalize(logic, dim=-1)
    sim = (z @ z.t()) / tau  # (B,B)
    B = z.size(0)
    eye = torch.eye(B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, float("-inf"))  # exclude self

    labels = labels.to(z.device)
    anchor_mask = labels == target_label  # which rows are anchors
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~eye  # same-label pairs

    # log-softmax denominator over all non-self entries.
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)  # (B,B)
    # log_prob has -inf on the diagonal; zero it so 0*-inf doesn't become NaN.
    log_prob = log_prob.masked_fill(eye, 0.0)

    pos_per_anchor = pos_mask.float().sum(1)  # (B,)
    valid_anchor = anchor_mask & (pos_per_anchor > 0)
    if valid_anchor.sum() == 0:
        return logic.sum() * 0.0  # keep graph, zero loss

    mean_log_prob_pos = (pos_mask.float() * log_prob).sum(1) / pos_per_anchor.clamp_min(1)
    loss = -mean_log_prob_pos[valid_anchor].mean()
    return loss


def infonce_logic_pos(logic: torch.Tensor, labels: torch.Tensor, tau: float = 0.1):
    """L_logic+ : pull valid (label 1) logic reps together, push from invalid."""
    return _supcon(logic, labels, target_label=1, tau=tau)


def infonce_logic_neg(logic: torch.Tensor, labels: torch.Tensor, tau: float = 0.1):
    """L_logic- : symmetric, anchored on invalid (label 0)."""
    return _supcon(logic, labels, target_label=0, tau=tau)


def content_loss(lcf, R_plus: torch.Tensor, R_minus: torch.Tensor) -> torch.Tensor:
    """Eq.10-12: content must be logic-independent.

    Take content from the valid rep (R_plus), pair it with each logic, and require
    that the decoder reconstructs the corresponding input.
      R_hat_+ = Decoder(content+, logic+)  ~ R_plus
      R_hat_- = Decoder(content+, logic-)  ~ R_minus
    """
    content_p, logic_p = lcf.encode(R_plus)
    _, logic_m = lcf.encode(R_minus)
    R_hat_plus = lcf.reconstruct(content_p, logic_p)
    R_hat_minus = lcf.reconstruct(content_p, logic_m)
    return F.mse_loss(R_hat_plus, R_plus) + F.mse_loss(R_hat_minus, R_minus)


def total_loss(lcf, R_plus: torch.Tensor, R_minus: torch.Tensor, tau: float = 0.1):
    """Combine all losses over a paired batch.

    R_plus  (B,d): valid-conclusion token reps (label 1)
    R_minus (B,d): invalid-conclusion token reps (label 0)
    Returns (loss, dict of components).
    """
    R_all = torch.cat([R_plus, R_minus], dim=0)
    labels = torch.cat([
        torch.ones(R_plus.size(0), device=R_plus.device),
        torch.zeros(R_minus.size(0), device=R_minus.device),
    ]).long()

    # reconstruction over all reps
    l_rec = rec_loss(lcf, R_all)

    # logic contrastive over all reps
    _, logic_all = lcf.encode(R_all)
    l_pos = infonce_logic_pos(logic_all, labels, tau=tau)
    l_neg = infonce_logic_neg(logic_all, labels, tau=tau)

    # content constraint over pairs
    l_content = content_loss(lcf, R_plus, R_minus)

    loss = l_rec + l_pos + l_neg + l_content
    comps = {
        "loss": loss.detach(),
        "rec": l_rec.detach(),
        "logic_pos": l_pos.detach(),
        "logic_neg": l_neg.detach(),
        "content": l_content.detach(),
    }
    return loss, comps
