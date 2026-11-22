"""P1 Stage B: distillation loss composition (plan C3)."""
from __future__ import annotations

import torch
import torch.nn as nn


def test_distill_loss_reduces_to_ce_when_weights_zero() -> None:
    from histogate.train_utils import distill_loss

    torch.manual_seed(0)
    logits = torch.randn(8, 9)
    y = torch.randint(0, 9, (8,))
    t_soft = torch.softmax(torch.randn(8, 9), dim=1)
    t_emb = torch.randn(8, 32)
    s_feat = torch.randn(8, 16)
    proj = nn.Linear(16, 32)
    loss, parts = distill_loss(logits, y, t_soft, t_emb, s_feat, proj,
                               w_kl=0.0, w_mse=0.0)
    ce = torch.nn.functional.cross_entropy(logits, y)
    assert torch.isclose(loss, ce, atol=1e-6)
    assert parts["kl"] >= 0 and parts["mse"] >= 0


def test_distill_loss_perfect_terms_near_zero() -> None:
    from histogate.train_utils import distill_loss

    torch.manual_seed(0)
    y = torch.tensor([0, 1, 2, 3])
    logits = 8.0 * torch.eye(4) - 2.0         # confident, argmax == y
    # teacher soft = the SAME distribution: KL(teacher||student) must be 0
    # (a one-hot teacher at T=2 would be near-uniform — not "perfect")
    t_soft = logits.clone()  # identical logits -> KL == 0
    t_emb = torch.randn(4, 8)
    proj = nn.Linear(4, 8)
    with torch.no_grad():
        proj.weight.copy_(t_emb.T)             # perfect projection
        proj.bias.zero_()
    s_feat = torch.eye(4)
    loss, parts = distill_loss(logits, y, t_soft, t_emb, s_feat, proj)
    assert parts["mse"] == 0.0
    assert parts["kl"] < 1e-6
    assert loss.item() < 0.1, parts


def test_distill_loss_kl_penalizes_mismatch() -> None:
    from histogate.train_utils import distill_loss

    torch.manual_seed(0)
    y = torch.randint(0, 4, (4,))
    logits = 4.0 * torch.eye(4)[y]
    t_same = logits.clone()
    t_flip = logits.flip(1).clone()  # opposite teacher
    proj = nn.Linear(4, 8)
    _, p_same = distill_loss(logits, y, t_same, torch.zeros(4, 8),
                             torch.zeros(4, 4), proj)
    _, p_flip = distill_loss(logits, y, t_flip, torch.zeros(4, 8),
                             torch.zeros(4, 4), proj)
    assert p_flip["kl"] > p_same["kl"]


def test_distill_loss_grads_flow() -> None:
    from histogate.train_utils import distill_loss

    torch.manual_seed(0)
    logits = torch.randn(4, 9, requires_grad=True)
    s_feat = torch.randn(4, 16, requires_grad=True)
    proj = nn.Linear(16, 32)
    loss, _ = distill_loss(logits, torch.randint(0, 9, (4,)),
                           torch.softmax(torch.randn(4, 9), 1),
                           torch.randn(4, 32), s_feat, proj)
    loss.backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0
    assert s_feat.grad is not None and s_feat.grad.abs().sum() > 0
    assert proj.weight.grad is not None and proj.weight.grad.abs().sum() > 0
