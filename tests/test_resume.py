"""Overnight-run hardening: per-epoch checkpoints + resume (CPU, fast models).

Adversarial angles: BEST checkpoint must track the best epoch (a run whose
val F1 peaks mid-run must not ship the last epoch as "best"); resume must
SKIP completed epochs (recomputing them wastes the night and lies about
history); resume without state must fail loudly (fail-closed, not epoch-0).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from histogate.config import TrainSettings
from histogate.model import Trainer


class _TinyDS(Dataset):
    def __init__(self, n: int = 64, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        self.x = torch.rand(n, 3, 224, 224, generator=g)
        self.y = torch.randint(0, 9, (n,), generator=g)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        return self.x[i], self.y[i]


def _trainer(tmp_path: Path, epochs: int, name: str = "hard") -> Trainer:
    cfg = TrainSettings(batch=64, epochs=epochs, lr=1e-4, workers=0,
                        backbone="convnext_tiny", val_holdout=0.3)
    return Trainer(cfg, run_dir=tmp_path / name, run_name=name, device="cpu",
                   tracker=_NoTracker())


class _NoTracker:
    enabled = False

    def run(self, name):
        import contextlib

        return contextlib.nullcontext()

    def log_params(self, *_): ...
    def log_epoch(self, *_): ...
    def log_step(self, *_): ...
    def log_final(self, *_): ...
    def log_artifacts(self, *_): ...
    def annotate(self, p):
        return p


def _tiny_model() -> torch.nn.Module:
    from histogate.model import PatchClassifier

    torch.manual_seed(0)
    return PatchClassifier(backbone="convnext_tiny", pretrained=False)


@pytest.mark.slow
def test_best_checkpoint_written_and_tracks_best(tmp_path: Path) -> None:
    t = _trainer(tmp_path, epochs=2)
    metrics = t.fit(_tiny_model(), _TinyDS(), _TinyDS(seed=1))
    best = tmp_path / "hard" / "best.ckpt.pt"
    last = tmp_path / "hard" / "last.ckpt.pt"
    state = tmp_path / "hard" / "state.json"
    assert best.exists() and last.exists() and state.exists()
    ck = torch.load(best, map_location="cpu", weights_only=False)
    f1s = [e["val_macro_f1"] for e in metrics["epochs"]]
    assert ck["metrics"]["final"]["val_macro_f1"] == pytest.approx(max(f1s)), (
        "best checkpoint must carry the MAX-f1 epoch, not the last"
    )


@pytest.mark.slow
def test_resume_skips_completed_epochs(tmp_path: Path) -> None:
    t1 = _trainer(tmp_path, epochs=2, name="resumable")
    t1.fit(_tiny_model(), _TinyDS(), _TinyDS(seed=1))

    t2 = _trainer(tmp_path, epochs=3, name="resumable")
    m2 = t2.fit(_tiny_model(), _TinyDS(), _TinyDS(seed=1), resume=True)
    assert m2["epochs"], "resumed run must record new epochs"
    assert m2["epochs"][0]["epoch"] == 2, (
        f"resume recomputed epoch {m2['epochs'][0]['epoch']} — completed epochs were not skipped"
    )


def test_resume_without_state_fails_closed(tmp_path: Path) -> None:
    t = _trainer(tmp_path, epochs=1, name="empty")
    with pytest.raises(FileNotFoundError, match="state"):
        t.fit(_tiny_model(), _TinyDS(), _TinyDS(), resume=True)
