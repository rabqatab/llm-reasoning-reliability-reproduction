"""Logic Control Framework (LCF) module.

Faithful re-implementation of Wu et al. (AAAI-25),
"Content-free Logical Modification of LLMs by Disentangling and Modifying
Logic Representation."

Equation map (paper):
  (1) R_logic+ = R_logic + V                         -> apply_steering
  (2) V = C_pos - C_neg                              -> set_steering_vector
  (3) R+ = MLP(R_content + Attn(R_content, R_logic+))-> Decoder.forward
  (4) D = R+ - R_input
  (5) R_input+ = R_input + D/||D|| * eta             -> modify
  (6,7) L_rec = MSE(R_input, Decoder(R_c, R_l))      -> losses["rec"]
  (8,9) L_logic+/- : supervised contrastive          -> losses["logic"]
  (10-12) L_content: swap-logic reconstruction        -> losses["content"]
  (13) L = L_rec + L_logic+ + L_logic- + L_content

Works for both:
  * training  -> per-token vectors batched as (B, 1, d_model)
  * inference -> full sequences          (B, T, d_model)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    """Single-head cross-attention. Query = content, Key/Value = logic.

    Operates over the token (sequence) dimension. With T=1 (training on single
    tokens) it degenerates to a learned linear map of the logic vector, which is
    exactly what we want; with T>1 (inference over a sequence) it performs real
    attention of content tokens over logic tokens.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, content, logic):  # content,logic: (B, T, dim)
        q, k, v = self.q(content), self.k(logic), self.v(logic)
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)  # (B,T,T)
        return attn @ v  # (B, T, dim)


class Decoder(nn.Module):
    """Fuse content + (modified) logic back to the original hidden space. Eq (3)."""

    def __init__(self, proj_dim: int, hidden: int, d_model: int):
        super().__init__()
        self.cross = CrossAttention(proj_dim)
        self.mlp = MLP(proj_dim, hidden, d_model)

    def forward(self, r_content, r_logic):
        fused = r_content + self.cross(r_content, r_logic)  # (B,T,proj_dim)
        return self.mlp(fused)                              # (B,T,d_model)


class LCF(nn.Module):
    def __init__(self, d_model=4096, proj_hidden=2048, proj_out=1024,
                 dec_hidden=2048, tau=0.1, use_content_proj=True):
        super().__init__()
        self.proj_out = proj_out
        self.use_content_proj = use_content_proj  # ablation: drop content projector
        self.content_proj = MLP(d_model, proj_hidden, proj_out)
        self.logic_proj = MLP(d_model, proj_hidden, proj_out)
        self.decoder = Decoder(proj_out, dec_hidden, d_model)
        self.tau = tau
        # Steering vector V (eq 2), filled by set_steering_vector after training.
        self.register_buffer("steering", torch.zeros(proj_out))

    # ---- projections --------------------------------------------------------
    def project(self, r_input):
        r_logic = self.logic_proj(r_input)
        if self.use_content_proj:
            r_content = self.content_proj(r_input)
        else:  # ablation: no content path -> decoder relies on logic only
            r_content = torch.zeros_like(r_logic)
        return r_content, r_logic

    # ---- inference-time modification (eqs 1,3,4,5) --------------------------
    @torch.no_grad()
    def set_steering_vector(self, logic_valid, logic_invalid):
        """V = mean(valid logic reps) - mean(invalid logic reps). Eq (2)."""
        c_pos = logic_valid.mean(0)
        c_neg = logic_invalid.mean(0)
        self.steering.copy_((c_pos - c_neg).to(self.steering.dtype))

    def modify(self, r_input, eta: float, sign: float = 1.0):
        """Return R_input+ that nudges r_input toward the logically-valid anchor.

        sign=-1.0 reverses the steering -> "invalid modification" (bidirectional).
        """
        r_content, r_logic = self.project(r_input)
        r_logic_mod = r_logic + sign * self.steering            # eq (1)
        r_plus = self.decoder(r_content, r_logic_mod)           # eq (3)
        d = r_plus - r_input                                    # eq (4)
        d = d / (d.norm(dim=-1, keepdim=True) + 1e-8)
        return r_input + d * eta                                # eq (5)

    # ---- training losses ----------------------------------------------------
    def reconstruct(self, r_input):
        r_content, r_logic = self.project(r_input)
        return self.decoder(r_content, r_logic)                 # eq (6)

    def supervised_contrastive(self, z, labels):
        """L_logic+ + L_logic- as a symmetric supervised-contrastive loss (eqs 8,9).

        z: (N, proj_out) logic reps; labels: (N,) bool/int (1=valid, 0=invalid).
        Pull same-validity reps together, push opposite-validity apart.
        """
        z = F.normalize(z, dim=-1)
        sim = (z @ z.t()) / self.tau                            # (N, N)
        N = z.size(0)
        self_mask = torch.eye(N, dtype=torch.bool, device=z.device)
        labels = labels.view(-1)
        # numerical stability, then exclude self from the denominator (not via -inf
        # on sim, to avoid 0*-inf NaNs when weighting by the positive mask)
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()
        exp = torch.exp(sim).masked_fill(self_mask, 0.0)
        log_prob = sim - torch.log(exp.sum(1, keepdim=True) + 1e-12)  # finite everywhere
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask
        pos_counts = pos_mask.sum(1).clamp(min=1)
        loss = -(log_prob * pos_mask).sum(1) / pos_counts
        valid_anchor = pos_mask.any(1)
        return loss[valid_anchor].mean() if valid_anchor.any() else sim.new_zeros(())

    def content_constraint(self, r_valid, r_invalid):
        """Swap logic across a same-content pair; must flip validity. Eqs (10-12)."""
        c_v, l_v = self.project(r_valid)
        c_i, l_i = self.project(r_invalid)
        rhat_minus = self.decoder(c_v, l_i)   # valid content + invalid logic -> invalid
        rhat_plus = self.decoder(c_i, l_v)    # invalid content + valid logic -> valid
        return F.mse_loss(rhat_minus, r_invalid) + F.mse_loss(rhat_plus, r_valid)

    def losses(self, r_valid, r_invalid, use_rec=True, use_logic=True, use_content=True):
        """Full objective (eq 13). r_valid/r_invalid: (B, d_model) paired by content.

        Flags drop individual terms for ablation studies (Table 3).
        """
        r_all = torch.cat([r_valid, r_invalid], 0)
        labels = torch.cat([
            torch.ones(r_valid.size(0), device=r_all.device),
            torch.zeros(r_invalid.size(0), device=r_all.device),
        ])
        zero = r_all.new_zeros(())
        l_rec = F.mse_loss(self.reconstruct(r_all), r_all) if use_rec else zero
        if use_logic:
            _, z_all = self.project(r_all)
            l_logic = self.supervised_contrastive(z_all, labels)
        else:
            l_logic = zero
        l_content = self.content_constraint(r_valid, r_invalid) if use_content else zero
        total = l_rec + l_logic + l_content
        d = lambda t: float(t.detach())
        return total, {"rec": d(l_rec), "logic": d(l_logic),
                       "content": d(l_content), "total": d(total)}
