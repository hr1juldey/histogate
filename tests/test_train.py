"""PR-5 RED-first tests: supervised training wiring (GPU-marked per TESTING §2).

Oracles / negative controls (PLAN PR-5 RED, TESTING §1.2):
- ONE-BATCH OVERFIT: 256 fixed real patches, <=50 optimizer steps -> >=0.99
  train accuracy. Probes the wiring end-to-end (data -> AMP -> loss ->
  optimizer -> weights), NOT dataset quality.
- CHANCE-CONTROL (negative): the same trainer on label-SHUFFLED data must land
  at chance macro-F1 <= 0.12 (PLAN PR-5 RED-a). If it "learns" shuffled
  labels, there is leakage or label corruption downstream of the loader.
- DETERMINISM: seed 42 twice -> byte-identical metrics JSON (sha256 of file
  bytes — TESTING §1.4 hash rule, not "close enough").
- CHECKPOINT SCHEMA: LLD §6 — {state_dict, config_hash, git_sha, metrics}.

CPU part (no gpu marker): PatchClassifier contract — 9 logits, 768-d features.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from histogate.config import TrainSettings
from histogate.data import CLASS_ORDER, build_cache, build_index

REPO = Path(__file__).resolve().parents[1]
TRAIN_DIR = REPO / "data" / "raw" / "NCT-CRC-HE-100K"

CHANCE_F1_MAX = 0.12  # PLAN.md PR-5 RED-a
OVERFIT_STEPS = 50  # PLAN.md PR-5 RED-b
OVERFIT_ACC = 0.99  # PLAN.md PR-5 RED-b


def _require_data() -> None:
    if not TRAIN_DIR.is_dir():
        pytest.skip("NCT-CRC-HE-100K not extracted yet")


def _subset_index(n: int):
    """Deterministic class-balanced round-robin subset of the real index."""
    idx = build_index(TRAIN_DIR)
    by_class: dict[str, list] = {c: [] for c in CLASS_ORDER}
    for e in idx.entries:
        if len(by_class[e.label]) < (n + len(CLASS_ORDER) - 1) // len(CLASS_ORDER):
            by_class[e.label].append(e)
    picked = []
    for i in range((n + len(CLASS_ORDER) - 1) // len(CLASS_ORDER)):
        for c in CLASS_ORDER:
            if len(by_class[c]) > i:
                picked.append(by_class[c][i])
    from histogate.data import PatchIndex

    return PatchIndex(picked[:n])


@pytest.fixture(scope="module")
def cache_256(tmp_path_factory):
    _require_data()
    return build_cache(_subset_index(256), tmp_path_factory.mktemp("c256") / "train.bin")


@pytest.fixture(scope="module")
def cache_3072(tmp_path_factory):
    _require_data()
    return build_cache(_subset_index(3072), tmp_path_factory.mktemp("c3072") / "train.bin")


class TestPatchClassifierContract:
    """CPU, runs in the default suite — model head/feature shapes (LLD §6)."""

    def test_nine_logits_and_768_features(self) -> None:
        from histogate.model import PatchClassifier

        model = PatchClassifier(pretrained=False, grad_checkpoint=True)
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits = model(x)
            feats = model.features(x)
        assert logits.shape == (2, 9)
        assert feats.shape == (2, 768)


@pytest.mark.gpu
class TestOneBatchOverfit:
    def test_256_patches_50_steps(self, tmp_path: Path, cache_256) -> None:
        """PLAN PR-5 RED-b: wiring probe — memorize one fixed batch fast."""
        from histogate.model import PatchClassifier, PatchDataset, Trainer

        cfg = TrainSettings(epochs=OVERFIT_STEPS, batch=256, workers=4, ema=0.999)
        ds = PatchDataset(cache_256, train=True, augment=False)  # FIXED patches per PLAN
        model = PatchClassifier(pretrained=True, grad_checkpoint=True)
        trainer = Trainer(cfg, run_dir=tmp_path / "overfit", run_name="overfit")
        trainer.fit(model, ds, ds)  # single-batch dataset: 1 step/epoch

        model.eval()
        loader = torch.utils.data.DataLoader(ds, batch_size=256, num_workers=0)
        correct = total = 0
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for x, y in loader:
                pred = model(x.to(trainer.device)).argmax(1).cpu()
                correct += int((pred == y).sum())
                total += len(y)
        acc = correct / total
        assert acc >= OVERFIT_ACC, f"one-batch overfit acc {acc:.4f} < {OVERFIT_ACC} in {OVERFIT_STEPS} steps"


@pytest.mark.gpu
class TestChanceControl:
    def test_shuffled_labels_land_at_chance(self, tmp_path: Path, cache_3072) -> None:
        """PLAN PR-5 RED-a: NEGATIVE control — shuffled labels must NOT be learnable."""
        from histogate.model import PatchClassifier, PatchDataset, Trainer

        cfg = TrainSettings(epochs=2, batch=256, workers=4)
        shuffled = PatchDataset(cache_3072, train=True, shuffle_labels=True, seed=cfg.seed)
        n_val = 512
        train_ds = torch.utils.data.Subset(shuffled, range(len(shuffled) - n_val))
        val_ds = torch.utils.data.Subset(shuffled, range(len(shuffled) - n_val, len(shuffled)))

        model = PatchClassifier(pretrained=False, grad_checkpoint=True)
        trainer = Trainer(cfg, run_dir=tmp_path / "chance", run_name="chance")
        metrics = trainer.fit(model, train_ds, val_ds)

        f1 = metrics["final"]["val_macro_f1"]
        assert f1 <= CHANCE_F1_MAX, (
            f"trainer reached macro-F1 {f1:.4f} on LABEL-SHUFFLED data "
            f"(chance ceiling {CHANCE_F1_MAX}) — leakage or label corruption"
        )


@pytest.mark.gpu
class TestDeterminism:
    def test_same_seed_twice_identical_metrics_bytes(self, tmp_path: Path, cache_3072) -> None:
        """PLAN PR-5 RED-c: seed 42 twice -> identical sha256 of metrics.json."""
        from histogate.model import PatchClassifier, PatchDataset, Trainer

        hashes = []
        for run in (1, 2):
            cfg = TrainSettings(epochs=2, batch=256, workers=4)
            ds = PatchDataset(cache_3072, train=True)
            train_ds = torch.utils.data.Subset(ds, range(len(ds) - 512))
            val_ds = torch.utils.data.Subset(ds, range(len(ds) - 512, len(ds)))
            model = PatchClassifier(pretrained=False, grad_checkpoint=True)
            run_dir = tmp_path / f"det{run}"
            trainer = Trainer(cfg, run_dir=run_dir, run_name="det")
            trainer.fit(model, train_ds, val_ds)
            h = hashlib.sha256(trainer.metrics_path.read_bytes()).hexdigest()
            hashes.append(h)

        assert hashes[0] == hashes[1], (
            f"seed-determinism violated: metrics.json sha256 differs between runs: {hashes}"
        )


@pytest.mark.gpu
class TestCheckpointSchema:
    def test_checkpoint_carries_lld6_fields(self, tmp_path: Path, cache_256) -> None:
        """LLD §6: checkpoint = {state_dict, config_hash, git_sha, metrics} — no orphans."""
        from histogate.model import PatchClassifier, PatchDataset, Trainer

        cfg = TrainSettings(epochs=1, batch=256, workers=4)
        ds = PatchDataset(cache_256, train=True)
        model = PatchClassifier(pretrained=False, grad_checkpoint=True)
        trainer = Trainer(cfg, run_dir=tmp_path / "ckpt", run_name="ckpt")
        trainer.fit(model, ds, ds)
        ckpt_path = trainer.ckpt_path
        assert ckpt_path.exists()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert set(ckpt) == {"state_dict", "config_hash", "git_sha", "metrics"}
        assert len(ckpt["config_hash"]) == 64  # sha256 hex
        assert len(ckpt["git_sha"]) == 40  # full git object sha
        torch.load(ckpt_path, map_location="cpu", weights_only=True)  # weights_only loadable
        assert isinstance(ckpt["metrics"], dict)
        assert "peak_vram_mb" in ckpt["metrics"]  # VRAM logged per task contract
