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


def test_removed_knob_is_rejected_even_at_its_old_default():
    """The rule that separates `removed` from `dead_lineage`.

    dcpo_pmi_agg's reader went with the dense-PMI scorer in 8ddae40. Its old default was
    'sum_clip'. Setting it to exactly that must still fail: there is no code left to read
    it, so the value is inert whatever it is, and a config that sets it tells the operator
    a lever is on when nothing can turn it on. Four archived configs do exactly this.
    """
    with pytest.raises(KnobRegistryError, match="8ddae40"):
        validate(_ok_cfg(dcpo_pmi_agg="sum_clip"))

    # and at any other value, for the same reason
    with pytest.raises(KnobRegistryError, match="was removed"):
        validate(_ok_cfg(dcpo_pmi_agg="mean"))


def test_removed_and_dead_lineage_are_governed_by_different_rules():
    """Negative control for the test above: prove the two sections really do differ.

    Without this, `test_removed_knob_is_rejected_even_at_its_old_default` would still pass
    if someone made the dead-lineage rule unconditional too — which would be a behaviour
    change, not a tightening, and would start rejecting dcpo_w_over on the live b3p launcher.
    """
    # dead_lineage AT its default: tolerated (reader still exists; presence != activation)
    validate(_ok_cfg(dcpo_over_threshold=1.0))
    # removed AT its default: rejected (no reader exists)
    with pytest.raises(KnobRegistryError):
        validate(_ok_cfg(dcpo_pmi_agg="sum_clip"))


def test_every_removed_entry_carries_the_sha_that_removed_it():
    """A removed entry without provenance cannot produce an actionable error message."""
    import yaml, os
    reg = yaml.safe_load(open(os.path.join("core", "KNOBS.yaml")))
    removed = reg.get("removed") or {}
    assert removed, "the removed section must exist"
    seen = 0
    for gen_name, gen in removed.items():
        for knob, entry in (gen.get("knobs") or {}).items():
            seen += 1
            assert entry.get("status") == "removed", f"{knob} in removed/{gen_name}"
            assert entry.get("removed_in"), f"{knob} has no removing sha"
            assert entry.get("removed_on"), f"{knob} has no removal date"
    assert seen == 35, f"expected the 35 knobs removed on 2026-08-03, found {seen}"


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
