"""P0a: split-conformal Adaptive Prediction Sets (APS) — exact qhat math.

Deterministic (non-randomized) APS: score(x, y) = cumulative softmax mass
from the highest-probability class down to and including the true class.
qhat = the ceil((n+1)(1-alpha))-th smallest calibration score (capped at n).
"""
import numpy as np
import pytest


def test_aps_scores_exact():
    from histogate.conformal import _aps_scores
    cal_probs = np.array([
        [0.6, 0.3, 0.1],   # y=0 is rank 0 -> 0.6
        [0.5, 0.4, 0.1],   # y=1 is rank 1 -> 0.5+0.4 = 0.9
        [0.9, 0.05, 0.05],  # y=0 is rank 0 -> 0.9
        [0.34, 0.33, 0.33],  # y=2 is rank 2 (stable ties) -> 1.0
    ])
    cal_y = np.array([0, 1, 0, 2])
    s = _aps_scores(cal_probs, cal_y)
    assert s == pytest.approx([0.6, 0.9, 0.9, 1.0])


def test_aps_qhat_exact():
    from histogate.conformal import aps_qhat
    cal_probs = np.array([
        [0.6, 0.3, 0.1], [0.5, 0.4, 0.1], [0.9, 0.05, 0.05], [0.34, 0.33, 0.33],
    ])
    cal_y = np.array([0, 1, 0, 2])
    # alpha=0.25: k = ceil(5*0.75) = 4 -> 4th smallest score = 1.0
    assert aps_qhat(cal_probs, cal_y, alpha=0.25) == pytest.approx(1.0)
    # alpha=0.50: k = ceil(5*0.5) = 3 -> 3rd smallest = 0.9
    assert aps_qhat(cal_probs, cal_y, alpha=0.5) == pytest.approx(0.9)
    # alpha small (high coverage) can exceed n -> capped at n
    assert aps_qhat(cal_probs, cal_y, alpha=0.01) == pytest.approx(1.0)


def test_aps_set_deterministic():
    from histogate.conformal import aps_set
    classes = ["ADI", "BACK", "DEB"]
    # qhat exactly at top mass -> single-class set
    assert aps_set(np.array([0.9, 0.05, 0.05]), 0.9, classes) == ["ADI"]
    # qhat above top mass -> grows until covered
    s = aps_set(np.array([0.6, 0.3, 0.1]), 0.9, classes)
    assert s == ["ADI", "BACK"]


def test_aps_set_uses_class_order_default():
    from histogate.conformal import aps_set
    from histogate.data import CLASS_ORDER
    s = aps_set(np.array([0.7, 0.1, 0.05, 0.05, 0.03, 0.03, 0.02, 0.01, 0.01]), 0.7)
    assert s == [CLASS_ORDER[0]]


def test_qhat_monotone_in_alpha():
    from histogate.conformal import aps_qhat
    rng = np.random.default_rng(0)
    p = rng.dirichlet([1] * 3, size=200)
    y = rng.integers(0, 3, 200)
    q10 = aps_qhat(p, y, alpha=0.10)
    q25 = aps_qhat(p, y, alpha=0.25)
    assert q25 <= q10  # smaller alpha (higher coverage) -> larger threshold


def test_empirical_coverage_on_exchangeable_draws():
    """The guarantee itself: P(y in set) >= 1 - alpha on exchangeable data."""
    from histogate.conformal import aps_qhat, aps_set
    rng = np.random.default_rng(42)
    n_cal, n_test = 2000, 2000
    def draw(m):
        p = rng.dirichlet([2, 2, 2], size=m)
        y = np.array([rng.choice(3, p=pi) for pi in p])
        return p, y
    p_cal, y_cal = draw(n_cal)
    p_test, y_test = draw(n_test)
    q = aps_qhat(p_cal, y_cal, alpha=0.10)
    hits = 0
    for i in range(n_test):
        members = aps_set(p_test[i], q, classes=[0, 1, 2])
        if y_test[i] in members:
            hits += 1
    cov = hits / n_test
    assert cov >= 0.90, f"empirical coverage {cov:.3f} below the 1-alpha guarantee"


# ---------------------------------------------------------------- A4: served
ROOT_DIR = __import__("pathlib").Path(__file__).resolve().parents[1]


def test_analyzer_serves_conformal_set_when_artifact_present(tmp_path):
    """analyze() of an in-distribution tile must carry conformal_set (alpha .10)."""
    import json

    import pytest
    from PIL import Image

    ckpt = ROOT_DIR / "runs/pr16-mixed/best.ckpt.pt"
    conformal = ROOT_DIR / "runs/pr16-mixed/conformal.json"
    tile = ROOT_DIR / "data/raw/CRC-VAL-HE-7K/NORM/NORM-TCGA-AASSYQPA.tif"
    if not (ckpt.exists() and conformal.exists() and tile.exists()):
        pytest.skip("ship artifacts absent")
    calib = json.loads((ROOT_DIR / "runs/pr16-mixed/calibration.json").read_text())
    calib_path = tmp_path / "calibration.json"
    calib_path.write_text(json.dumps(calib))
    tmp_path.joinpath("ood_knn_bank.pt").write_bytes(
        (ROOT_DIR / "runs/pr16-mixed/ood_knn_bank.pt").read_bytes())

    from histogate.serve.patch_demo import PatchAnalyzer
    an = PatchAnalyzer(ckpt, calib_path)
    r = an.analyze(Image.open(tile).convert("RGB"))
    assert r["ood"] is False
    assert isinstance(r.get("conformal_set"), list) and r["conformal_set"]
    assert set(r["conformal_set"]) <= set(__import__("histogate.data", fromlist=["CLASS_ORDER"]).CLASS_ORDER)
    assert r["conformal_set"][0] == r["top_class"]  # ranked by prob desc


def test_analyzer_no_conformal_key_when_artifact_absent(tmp_path):
    """Without conformal.json the response shape is unchanged (no key at all)."""
    import json

    import pytest
    from PIL import Image

    ckpt = ROOT_DIR / "runs/pr16-mixed/best.ckpt.pt"
    tile = ROOT_DIR / "data/raw/CRC-VAL-HE-7K/NORM/NORM-TCGA-AASSYQPA.tif"
    if not (ckpt.exists() and tile.exists()):
        pytest.skip("ship artifacts absent")
    calib = json.loads((ROOT_DIR / "runs/pr16-mixed/calibration.json").read_text())
    calib_path = tmp_path / "calibration.json"
    calib_path.write_text(json.dumps(calib))
    tmp_path.joinpath("ood_knn_bank.pt").write_bytes(
        (ROOT_DIR / "runs/pr16-mixed/ood_knn_bank.pt").read_bytes())

    from histogate.serve.patch_demo import PatchAnalyzer
    an = PatchAnalyzer(ckpt, calib_path)
    r = an.analyze(Image.open(tile).convert("RGB"))
    assert "conformal_set" not in r
