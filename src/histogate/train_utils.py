"""Teacher-student distillation loss (plan P1/C3).

loss = CE(y) + w_kl * T^2 * KL(teacher_logits/T || student_logits/T)
               + w_mse * MSE(proj(student_feat) -> teacher_emb)

t_soft carries teacher LOGITS or log-probabilities (the temperature is
applied exactly once on each side — passing raw probabilities would
double-softmax and over-flatten the target; caught by
tests/test_distill_loss.py).
The T^2 factor is the standard Hinton scaling so KL gradients survive the
temperature softening. All terms are computed in float32 regardless of the
autocast context the caller used for the forward pass.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

TEMP = 2.0
W_KL = 1.0
W_MSE = 0.5


def distill_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    t_soft: torch.Tensor,
    t_emb: torch.Tensor,
    s_feat: torch.Tensor,
    proj: nn.Module,
    temp: float = TEMP,
    w_kl: float = W_KL,
    w_mse: float = W_MSE,
) -> tuple[torch.Tensor, dict]:
    """Weighted CE + KL + feature-MSE; returns (loss, float parts dict).

    t_soft must be teacher logits/log-probs (see module docstring)."""
    ce = F.cross_entropy(logits, y)
    kl = F.kl_div(
        F.log_softmax(logits / temp, dim=1),
        F.softmax(t_soft / temp, dim=1),
        reduction="batchmean",
    ) * (temp * temp)
    mse = F.mse_loss(proj(s_feat), t_emb)
    loss = ce + w_kl * kl + w_mse * mse
    return loss, {"ce": float(ce), "kl": float(kl), "mse": float(mse)}
