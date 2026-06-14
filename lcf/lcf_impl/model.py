"""LCF (Logic Control Framework) core module.

Implements the trainable adapter from "Content-free Logical Modification of LLM by
Disentangling and Modifying Logic Representation" (Wu et al., AAAI 2025).

Architecture (spec sections A):
  - ContentProjector / LogicProjector: Linear(d->2048)->ReLU->Linear(2048->1024)
  - Decoder: cross-attention fusion of content + (modified) logic, then
             Linear(1024->2048)->ReLU->Linear(2048->d)
  - LCF: holds projectors + decoder + EMA logic centroids (C_pos, C_neg),
         V = C_pos - C_neg. Modifies logic, fuses, and nudges R_input toward R_+.

All shapes assume a batch of token reps: R_input has shape (B, d).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentProjector(nn.Module):
    """MLP: Linear(d -> hidden) -> ReLU -> Linear(hidden -> proj)."""

    def __init__(self, d: int, hidden: int = 2048, proj: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(d, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, proj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# Logic projector has the same shape but separate weights.
class LogicProjector(ContentProjector):
    pass


class CrossAttention(nn.Module):
    """Single-head scaled dot-product cross-attention with learnable W_Q/K/V/O.

    Q from content, K=V from (modified) logic. Operates on (B, proj) vectors by
    treating each as a length-1 sequence, so attention reduces to a gated
    projection of the logic value -- which is exactly the fusion in Eq.3.
    """

    def __init__(self, proj: int = 1024):
        super().__init__()
        self.proj = proj
        self.W_Q = nn.Linear(proj, proj, bias=False)
        self.W_K = nn.Linear(proj, proj, bias=False)
        self.W_V = nn.Linear(proj, proj, bias=False)
        self.W_O = nn.Linear(proj, proj, bias=False)
        self.scale = proj ** -0.5

    def forward(self, content: torch.Tensor, logic: torch.Tensor) -> torch.Tensor:
        # content, logic: (B, proj). Treat as (B, 1, proj) sequences.
        q = self.W_Q(content).unsqueeze(1)  # (B,1,proj)
        k = self.W_K(logic).unsqueeze(1)    # (B,1,proj)
        v = self.W_V(logic).unsqueeze(1)    # (B,1,proj)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B,1,1)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (B,1,proj)
        out = self.W_O(out).squeeze(1)  # (B,proj)
        return out


class Decoder(nn.Module):
    """Fuse content + logic via cross-attention, then decode back to d.

    Eq.3:  R_+ = MLP(R_content + Attn(Q=R_content, K=V=R_logic))
    """

    def __init__(self, d: int, hidden: int = 2048, proj: int = 1024):
        super().__init__()
        self.attn = CrossAttention(proj)
        self.fc1 = nn.Linear(proj, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, d)

    def forward(self, content: torch.Tensor, logic: torch.Tensor) -> torch.Tensor:
        fused = content + self.attn(content, logic)  # (B, proj)
        return self.fc2(self.act(self.fc1(fused)))   # (B, d)


class LCF(nn.Module):
    """Logic Control Framework adapter.

    Holds projectors, decoder, and EMA logic centroids C_pos / C_neg (proj-dim).
    V = C_pos - C_neg is the validity direction in logic space.
    """

    def __init__(self, d: int, hidden: int = 2048, proj: int = 1024,
                 ema_momentum: float = 0.99):
        super().__init__()
        self.d = d
        self.proj_dim = proj
        self.ema_momentum = ema_momentum

        self.content_proj = ContentProjector(d, hidden, proj)
        self.logic_proj = LogicProjector(d, hidden, proj)
        self.decoder = Decoder(d, hidden, proj)

        # EMA centroids of valid (pos) and invalid (neg) logic reps.
        self.register_buffer("C_pos", torch.zeros(proj))
        self.register_buffer("C_neg", torch.zeros(proj))
        self.register_buffer("_initialized", torch.zeros(1))

    # ---- core ops -------------------------------------------------------
    def encode(self, R_input: torch.Tensor):
        """R_input (B,d) -> (R_content (B,proj), R_logic (B,proj))."""
        return self.content_proj(R_input), self.logic_proj(R_input)

    @property
    def V(self) -> torch.Tensor:
        return self.C_pos - self.C_neg

    def modify(self, R_logic: torch.Tensor, sign: int = 1) -> torch.Tensor:
        """R_logic +/- V  (Eq.1-2). sign=+1 -> toward valid, -1 -> toward invalid."""
        return R_logic + sign * self.V

    def reconstruct(self, R_content: torch.Tensor, R_logic: torch.Tensor) -> torch.Tensor:
        """Decoder on UN-modified logic (used for L_rec)."""
        return self.decoder(R_content, R_logic)

    def forward(self, R_input: torch.Tensor, eta: float, sign: int = 1) -> torch.Tensor:
        """Full modification pipeline (Eq.3-5).

        Returns R_input_plus = R_input + (D/||D||)*eta, where
        D = R_+ - R_input and R_+ = Decoder(content, modified_logic).
        """
        R_content, R_logic = self.encode(R_input)
        R_logic_mod = self.modify(R_logic, sign=sign)
        R_plus = self.decoder(R_content, R_logic_mod)  # (B,d)
        D = R_plus - R_input
        norm = D.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return R_input + (D / norm) * eta

    # ---- centroid bookkeeping ------------------------------------------
    @torch.no_grad()
    def update_centroids(self, R_logic: torch.Tensor, labels: torch.Tensor):
        """EMA-update C_pos/C_neg from a batch of logic reps.

        labels: 1 = valid (pos), 0 = invalid (neg). Detached.
        """
        R_logic = R_logic.detach()
        labels = labels.to(R_logic.device)
        m = self.ema_momentum
        first = self._initialized.item() < 0.5

        pos = R_logic[labels == 1]
        neg = R_logic[labels == 0]
        if pos.numel() > 0:
            mean_pos = pos.mean(0)
            self.C_pos.copy_(mean_pos if first else m * self.C_pos + (1 - m) * mean_pos)
        if neg.numel() > 0:
            mean_neg = neg.mean(0)
            self.C_neg.copy_(mean_neg if first else m * self.C_neg + (1 - m) * mean_neg)
        if first and pos.numel() > 0 and neg.numel() > 0:
            self._initialized.fill_(1.0)

    @torch.no_grad()
    def set_centroids(self, C_pos: torch.Tensor, C_neg: torch.Tensor):
        """Freeze centroids (e.g. computed once over full train set for inference)."""
        self.C_pos.copy_(C_pos.to(self.C_pos.device))
        self.C_neg.copy_(C_neg.to(self.C_neg.device))
        self._initialized.fill_(1.0)


def build_lcf(d: int, dims=(2048, 1024), ema_momentum: float = 0.99) -> LCF:
    hidden, proj = dims
    return LCF(d=d, hidden=hidden, proj=proj, ema_momentum=ema_momentum)
