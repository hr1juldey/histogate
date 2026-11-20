"""PR-4 RED-first tests: data layer with patient-integrity enforcement.

Oracles:
- LEAKAGE PROBE: an index pair with a deliberately shared patient MUST raise.
  (The test constructs the failure it guards — the anti-tautological core.)
- COUNT/DISTRIBUTION: independently built fixture tree, counts recomputed in test.
- CACHE DETERMINISM: two cache builds -> identical sha256; read-back == source.
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from histogate.data import (
    PatientLeakError,
    assert_no_patient_overlap,
    build_cache,
    build_index,
    class_distribution,
)

CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]


def _make_tree(root: Path, split: str, per_class: int = 3, patients: dict[str, list[str]] | None = None) -> Path:
    """Fixture tree: <root>/<split>/<CLASS>/<CLASS>-<PATIENT>.tif"""
    split_dir = root / split
    split_dir.mkdir(parents=True)
    for c in CLASSES:
        d = split_dir / c
        d.mkdir()
        pats = (patients or {}).get(c, [f"P{i}" for i in range(per_class)])
        for p in pats:
            arr = np.zeros((8, 8, 3), dtype=np.uint8)
            arr[..., 0] = CLASSES.index(c) * 25
            import tifffile

            tifffile.imwrite(d / f"{c}-{p}.tif", arr)
    return split_dir


class TestIndex:
    def test_counts_and_labels(self, tmp_path: Path) -> None:
        _make_tree(tmp_path, "train", per_class=3)
        idx = build_index(tmp_path / "train")
        assert len(idx) == 27
        dist = class_distribution(idx)
        assert all(dist[c] == 3 for c in CLASSES)

    def test_patient_parsing_from_filename(self, tmp_path: Path) -> None:
        _make_tree(tmp_path, "train", per_class=2)
        idx = build_index(tmp_path / "train")
        patients = {e.patient_id for e in idx.entries}
        assert patients == {"P0", "P1"}  # same patient names reused across classes
        assert all(e.patient_id.startswith("P") for e in idx.entries)

    def test_label_mismatch_between_dir_and_filename_rejects(self, tmp_path: Path) -> None:
        d = tmp_path / "train" / "ADI"
        d.mkdir(parents=True)
        import tifffile

        tifffile.imwrite(d / "TUM-PPPP.tif", np.zeros((8, 8, 3), np.uint8))
        with pytest.raises(ValueError):
            build_index(tmp_path / "train")


class TestLeakageProbe:
    def test_shared_patient_must_raise(self, tmp_path: Path) -> None:
        """THE adversarial core: construct overlapping patients, expect the guard to fire."""
        _make_tree(tmp_path, "train", per_class=2, patients={"ADI": ["SHARED", "A2"]})
        _make_tree(tmp_path, "val", per_class=2, patients={"ADI": ["SHARED", "V1"]})
        train = build_index(tmp_path / "train")
        val = build_index(tmp_path / "val")
        with pytest.raises(PatientLeakError) as exc:
            assert_no_patient_overlap({"train": train, "val": val})
        assert "SHARED" in str(exc.value)

    def test_disjoint_patients_pass(self, tmp_path: Path) -> None:
        # every class gets a split-prefixed patient so NOTHING is shared
        trn = {c: ["TRN1"] for c in CLASSES}
        val = {c: ["VAL1"] for c in CLASSES}
        _make_tree(tmp_path, "train", per_class=1, patients=trn)
        _make_tree(tmp_path, "val", per_class=1, patients=val)
        train = build_index(tmp_path / "train")
        val_idx = build_index(tmp_path / "val")
        assert_no_patient_overlap({"train": train, "val": val_idx})  # no raise


class TestCache:
    def test_cache_determinism_and_readback(self, tmp_path: Path) -> None:
        _make_tree(tmp_path, "train", per_class=2)
        idx = build_index(tmp_path / "train")
        c1 = build_cache(idx, tmp_path / "cache1.bin", size=8)
        build_cache(idx, tmp_path / "cache2.bin", size=8)
        h1 = hashlib.sha256((tmp_path / "cache1.bin").read_bytes()).hexdigest()
        h2 = hashlib.sha256((tmp_path / "cache2.bin").read_bytes()).hexdigest()
        assert h1 == h2, "cache build is not deterministic"
        assert len(c1) == len(idx)
        img, label = c1[0]
        assert img.dtype == np.uint8  # data layer stays numpy; torch lives in PR-5
        assert img.shape == (8, 8, 3)
        assert label in range(9)


class TestIntegration:
    def test_real_dataset_counts(self, tmp_path: Path) -> None:
        """Runs only when the real data has landed (slow marker; tonight)."""
        root = Path("/home/riju279/Documents/code/AIRA/histogate/data/raw")
        train_dir = root / "NCT-CRC-HE-100K"
        val_dir = root / "CRC-VAL-HE-7K"
        if not (train_dir.exists() and val_dir.exists()):
            pytest.skip("real dataset not yet extracted")
        train = build_index(train_dir)
        val = build_index(val_dir)
        assert len(train) == 100000
        assert len(val) == 7180
        assert_no_patient_overlap({"train": train, "val": val})
