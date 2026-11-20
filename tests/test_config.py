"""PR-1 RED-first tests: config is the single source of thresholds.

Oracle: docs/TESTING.md §3 (gate table) — thresholds designed before code.
Adversarial: inconsistent thresholds MUST reject at construction, and env
overrides MUST actually flow through (a config that ignores env is silent death).
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from histogate.config import GateSettings, Settings, load_settings

REPO = Path(__file__).resolve().parents[1]


class TestGateSettings:
    def test_defaults_match_testing_doc(self) -> None:
        g = GateSettings()
        assert g.f1_min == 0.93
        assert g.f1_drop_domain_max == 0.015
        assert g.ece_max == 0.05
        assert g.ood_auroc_min == 0.90
        assert g.tsr_drift_max == 0.05
        assert g.til_drift_rel_max == 0.10
        assert g.quant_agreement_min == 0.999
        assert g.throughput_min == 40.0
        assert g.vram_max_mb == 2048

    def test_nonsense_thresholds_reject(self) -> None:
        with pytest.raises(ValidationError):
            GateSettings(tsr_drift_max=0.9)  # outside (0, 0.2]
        with pytest.raises(ValidationError):
            GateSettings(f1_min=0.2)  # below sane floor
        with pytest.raises(ValidationError):
            GateSettings(throughput_min=-1)

    def test_internal_consistency(self) -> None:
        with pytest.raises(ValidationError):
            GateSettings(quant_agreement_min=0.99, f1_min=0.99)  # agreement must exceed f1 floor tolerance rule
        # rule: quant agreement floor must be >= f1_min (a model can't be trusted
        # to reproduce its own accuracy floor if quantized agreement is lower)


class TestSettings:
    def test_default_yaml_loads(self) -> None:
        s = load_settings(REPO / "configs" / "default.yaml")
        assert s.train.seed == 42
        assert s.train.batch == 256
        assert s.serve.precision in {"fp32", "fp16", "int8"}

    def test_env_override_flows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HISTOGATE_TRAIN__SEED", "7")
        s = Settings()
        assert s.train.seed == 7

    def test_seed_range(self) -> None:
        with pytest.raises(ValidationError):
            Settings(train={"seed": -3})
