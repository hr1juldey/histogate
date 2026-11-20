"""PR-9 (2/2): AttentionMIL — slide-level head over patch embeddings.

PLAN.md PR-9 RED contract:
(a) HIDDEN-KEY BAG: slide label depends on >=3 key instances among 200 noise
    instances; a trained head must reach >=95% AND recover >=2/3 keys via
    argmax attention. A MEAN-POOLING baseline trained identically must score
    LOWER — the adversarial comparison that justifies attention at all.
(b) PERMUTATION INVARIANCE: shuffling instance order leaves the slide logit
    bit-stable (<=1e-6).
(c) Single-instance bags produce finite logits (no degenerate softmax).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from histogate.model import AttentionMIL

DIM = 64
N_INST = 200
N_KEYS = 3
KEY_VEC = torch.zeros(DIM)
KEY_VEC[:8] = 4.0  # key signature on first 8 dims; everything else is noise


def _bag(rng: np.random.Generator, label: int) -> torch.Tensor:
    """label 1 -> N_KEYS key instances; label 0 -> none.

    NOTE (parent-diagnosed 2026-11-20): an earlier draft used 3-keys-vs-2-keys
    (a counting task). That oracle is UNSOLVABLE by attention pooling: once
    attention concentrates, a 3-key and a 2-key bag pool to the same vector
    (= KEY + noise), and meanwhile the classifier head siphons the weak
    mean signal (measured: acc 0.65-0.69, key-hit 0.00-0.08 across key
    strength 4-8 and lr 3e-3/1e-2 at 2000-3000 epochs) so attention never
    sharpens. Presence-vs-absence keeps the full difficulty that matters:
    find 3 needles among 200, mean pooling must still fail (3/200 dilution).
    """
    h = torch.from_numpy(rng.normal(0, 1, (N_INST, DIM)).astype(np.float32))
    n_keys = N_KEYS if label else 0
    pos = rng.choice(N_INST, size=N_KEYS, replace=False)[:n_keys]
    h[torch.from_numpy(pos)] = h[torch.from_numpy(pos)] + KEY_VEC
    return h, set(int(p) for p in pos)


def _dataset(n_bags: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, list[set[int]]]:
    rng = np.random.default_rng(seed)
    bags, labels, keysets = [], [], []
    for i in range(n_bags):
        y = i % 2
        h, keys = _bag(rng, y)
        bags.append(h)
        labels.append(y)
        keysets.append(keys)
    return torch.stack(bags), torch.tensor(labels), keysets


def _train(head: nn.Module, X: torch.Tensor, y: torch.Tensor, epochs: int = 1500) -> None:
    opt = torch.optim.Adam(head.parameters(), lr=3e-3)
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(head(X), y)
        loss.backward()
        opt.step()


@pytest.mark.slow
def test_hidden_key_bag_beats_mean_pooling() -> None:
    torch.manual_seed(0)
    X, y, keysets = _dataset(128, seed=1)
    Xt, yt, keyt = _dataset(64, seed=2)

    mil = AttentionMIL(in_dim=DIM, n_classes=2)
    _train(mil, X, y)
    with torch.no_grad():
        acc_mil = (mil(Xt).argmax(1) == yt).float().mean().item()
    assert acc_mil >= 0.95, f"attention head only {acc_mil:.3f} — wiring, not quality"

    # attention must land ON the keys: >=2 of the top-3 attended instances are keys
    # (key recovery is defined on POSITIVE bags — negatives contain no keys)
    with torch.no_grad():
        A = mil.attention(Xt)  # (B, N)
    top3 = A.argsort(dim=1, descending=True)[:, :3]
    pos_bags = [b for b in range(len(yt)) if yt[b] == 1]
    hits = np.mean(
        [len(set(top3[b].tolist()) & keyt[b]) >= 2 for b in pos_bags]
    )
    assert hits >= 0.9, f"attention misses keys (hit-rate {hits:.2f})"

    # adversarial baseline: identical training, mean pooling instead of attention
    class MeanPool(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(DIM, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x.mean(dim=1))

    mean_head = MeanPool()
    _train(mean_head, X, y)
    with torch.no_grad():
        acc_mean = (mean_head(Xt).argmax(1) == yt).float().mean().item()
    assert acc_mil > acc_mean, (
        f"mean pooling ({acc_mean:.3f}) matched/beat attention ({acc_mil:.3f}) "
        "— attention unjustified on this probe"
    )


def test_permutation_invariance() -> None:
    torch.manual_seed(3)
    mil = AttentionMIL(in_dim=DIM, n_classes=2).eval()
    X, _, _ = _dataset(4, seed=4)
    g = torch.Generator().manual_seed(5)
    perm = torch.randperm(N_INST, generator=g)
    with torch.no_grad():
        d = (mil(X) - mil(X[:, perm])).abs().max().item()
    assert d <= 1e-6, f"bag order changed logits by {d:.2e}"


def test_single_instance_bag_is_finite() -> None:
    torch.manual_seed(6)
    mil = AttentionMIL(in_dim=DIM, n_classes=2).eval()
    one = torch.randn(3, 1, DIM)
    with torch.no_grad():
        out = mil(one)
    assert out.shape == (3, 2)
    assert torch.isfinite(out).all()
