"""The knob registry must make the 2026-06-22 meta_floor incident impossible."""
import pytest
from src.training.knob_registry import validate, KnobRegistryError


class Cfg(dict):
    """Stands in for an OmegaConf algorithm node (has .get and attribute access)."""
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e


def _ok_cfg(**over):
    import yaml, os
    reg = yaml.safe_load(open(os.path.join("core", "KNOBS.yaml")))
    lb = list((reg.get("load_bearing") or {}).keys())
    base = Cfg(dcpo_rmeta_source="pmi_shift", dcpo_ack_load_bearing=lb)
    base.update(over)
    return base


def test_clean_config_passes_and_returns_live_knobs():
    resolved = validate(_ok_cfg())
    assert resolved, "validate must report the resolved live knobs"
    assert all(isinstance(n, str) for n, _ in resolved)


def test_setting_a_dead_lineage_knob_is_rejected():
    """Reaching a retired reward generation must fail loudly."""
    with pytest.raises(KnobRegistryError, match="retired reward"):
        validate(_ok_cfg(dcpo_over_threshold=0.5))


def test_dead_lineage_knob_at_its_default_is_tolerated():
    """Presence alone is not activation; only deviation is."""
    validate(_ok_cfg(dcpo_over_threshold=1.0))


def test_unacknowledged_load_bearing_knob_is_rejected():
    """This is the meta_floor incident: a load-bearing knob inherited invisibly."""
    cfg = _ok_cfg()
    cfg["dcpo_ack_load_bearing"] = [
        k for k in cfg["dcpo_ack_load_bearing"] if k != "dcpo_meta_floor"
    ]
    with pytest.raises(KnobRegistryError, match="dcpo_meta_floor"):
        validate(cfg)


def test_missing_rmeta_source_is_rejected():
    cfg = _ok_cfg()
    del cfg["dcpo_rmeta_source"]
    with pytest.raises(KnobRegistryError, match="dcpo_rmeta_source is absent"):
        validate(cfg)


def test_negative_control_the_guard_can_actually_fail():
    """Without this, every test above could pass by never exercising the check."""
    with pytest.raises(KnobRegistryError):
        validate(Cfg())
