"""PR-1 RED-first tests: the Pydantic contracts are the law of the system.

Oracles: the schema definitions in docs/LLD.md §2 (designed BEFORE code).
Adversarial core: invalid values MUST reject; inconsistent internal state
(a GateReport claiming PASS while a check failed) MUST reject.
"""

import pytest
from pydantic import ValidationError

from histogate.schemas.bench import BenchResult
from histogate.schemas.gate import GateCheck, GateReport
from histogate.schemas.report import (
    ArtifactFinding,
    ArtifactKind,
    PatchPrediction,
    SlideReport,
    TILResult,
    TissueClass,
    TSRResult,
)


def _tsr(**kw) -> dict:
    base = dict(
        stroma_fraction=0.62, tumor_fraction=0.38, tsr=0.38,
        group="stroma-high", method="patch-proxy-vanpelt-2018",
        disclosure="patch-level proxy of the van Pelt 2018 slide-margin protocol",
    )
    base.update(kw)
    return base


class TestTissueClass:
    def test_exactly_nine_classes(self) -> None:
        assert {c.value for c in TissueClass} == {
            "ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"
        }

    def test_unknown_class_rejects(self) -> None:
        with pytest.raises(ValidationError):
            PatchPrediction(tissue="BLOOD", prob=0.9, energy=-1.0,
                            temperature=1.0, is_ood=False)


class TestPatchPrediction:
    def test_valid(self) -> None:
        p = PatchPrediction(tissue=TissueClass.TUM, prob=0.97, energy=-2.1,
                            temperature=1.3, is_ood=False, artifact=None)
        assert p.tissue is TissueClass.TUM

    def test_prob_out_of_range_rejects(self) -> None:
        with pytest.raises(ValidationError):
            PatchPrediction(tissue=TissueClass.TUM, prob=1.5, energy=-2.1,
                            temperature=1.0, is_ood=False)

    def test_artifact_must_be_known_kind(self) -> None:
        with pytest.raises(ValidationError):
            PatchPrediction(tissue=TissueClass.STR, prob=0.9, energy=-1.0,
                            temperature=1.0, is_ood=True, artifact="coffee_stain")


class TestArtifacts:
    def test_exactly_eight_airaqc_kinds(self) -> None:
        assert len(ArtifactKind) == 8

    def test_stage_split(self) -> None:
        f = ArtifactFinding(kind=ArtifactKind.PEN_MARK, stage="processing", pixel_fraction=0.03)
        assert f.stage == "processing"
        with pytest.raises(ValidationError):
            ArtifactFinding(kind=ArtifactKind.PEN_MARK, stage="imaging", pixel_fraction=0.03)

    def test_pixel_fraction_bounds(self) -> None:
        with pytest.raises(ValidationError):
            ArtifactFinding(kind=ArtifactKind.AIR_BUBBLE, stage="processing", pixel_fraction=-0.1)


class TestTSR:
    def test_group_cut_follows_fraction(self) -> None:
        """Adversarial consistency: group label must match the 50% cut of stroma."""
        r = TSRResult(**_tsr(stroma_fraction=0.62, group="stroma-high"))
        assert r.group == "stroma-high"
        with pytest.raises(ValidationError):
            # stroma 0.62 > 0.50 must be stroma-high; claiming low must REJECT
            TSRResult(**_tsr(stroma_fraction=0.62, group="stroma-low"))

    def test_empty_disclosure_rejects(self) -> None:
        with pytest.raises(ValidationError):
            TSRResult(**_tsr(disclosure=""))

    def test_fraction_bounds(self) -> None:
        with pytest.raises(ValidationError):
            TSRResult(**_tsr(stroma_fraction=1.2))


class TestTIL:
    def test_density_nonnegative(self) -> None:
        ok = TILResult(intratumoral_density_mm2=142.0, stromal_density_mm2=88.5)
        assert ok.mpp == 0.5
        with pytest.raises(ValidationError):
            TILResult(intratumoral_density_mm2=-1.0, stromal_density_mm2=0.0)


class TestSlideReport:
    def _report(self, **kw) -> dict:
        base = dict(
            slide_id="slideX", n_patches=512, n_gated=37,
            compartment_fractions={c: 1.0 / 9 for c in TissueClass},
            artifacts=[ArtifactFinding(kind=ArtifactKind.TISSUE_FOLD, stage="processing",
                                       pixel_fraction=0.02).model_dump()],
            tsr=TSRResult(**_tsr()).model_dump(),
            til=TILResult(intratumoral_density_mm2=100.0, stromal_density_mm2=50.0).model_dump(),
            slide_label="tumor", slide_confidence=0.8,
            calibration_temperature=1.31, model_revision="abc123",
        )
        base.update(kw)
        return base

    def test_missing_ruo_rejects(self) -> None:
        with pytest.raises(ValidationError):
            SlideReport(**self._report())

    def test_ruo_literal_exact(self) -> None:
        r = SlideReport(**self._report(), ruo_disclaimer="Research Use Only. Not for use in diagnostic procedures.")
        assert "Research Use Only" in r.ruo_disclaimer
        with pytest.raises(ValidationError):
            SlideReport(**self._report(), ruo_disclaimer="for research use")

    def test_compartment_fractions_sum_leq_one(self) -> None:
        bad = {c: 0.2 for c in TissueClass}  # sums to 1.8
        with pytest.raises(ValidationError):
            SlideReport(**self._report(compartment_fractions=bad,
                                       ruo_disclaimer="Research Use Only. Not for use in diagnostic procedures."))

    def test_json_roundtrip_stable(self) -> None:
        r = SlideReport(**self._report(), ruo_disclaimer="Research Use Only. Not for use in diagnostic procedures.")
        r2 = SlideReport.model_validate_json(r.model_dump_json())
        assert r2 == r


class TestGate:
    def _check(self, passed: bool) -> GateCheck:
        return GateCheck(name="G-ACC", value=0.948, threshold=0.93,
                         comparator=">=", passed=passed, detail="worst seed")

    def test_overall_pass_must_equal_checks(self) -> None:
        """Adversarial: a report claiming overall PASS with a failed check must REJECT."""
        with pytest.raises(ValidationError):
            GateReport(overall_pass=True, checks=[self._check(False)],
                       model_revision="abc", dataset_revision="v1",
                       timestamp="2026-11-21T02:14:00")

    def test_valid_gate_report(self) -> None:
        g = GateReport(overall_pass=True, checks=[self._check(True)],
                       model_revision="abc", dataset_revision="v1",
                       timestamp="2026-11-21T02:14:00")
        assert g.overall_pass

    def test_comparator_literal(self) -> None:
        with pytest.raises(ValidationError):
            GateCheck(name="x", value=1, threshold=1, comparator="~=", passed=True, detail="")


class TestBench:
    def test_precision_literal(self) -> None:
        with pytest.raises(ValidationError):
            BenchResult(precision="fp8", macro_f1=0.9, ece=0.03,
                        patches_per_s=41.0, p95_latency_ms=12.0, peak_vram_mb=1500)

    def test_negative_throughput_rejects(self) -> None:
        with pytest.raises(ValidationError):
            BenchResult(precision="fp16", macro_f1=0.9, ece=0.03,
                        patches_per_s=-5, p95_latency_ms=12.0, peak_vram_mb=1500)
