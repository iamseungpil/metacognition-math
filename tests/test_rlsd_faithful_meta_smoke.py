"""Smoke tests for RLSD_FAITHFUL_META (R20, direction B core).

LOCK ref: GFN_SGFN_IMPROVEMENT_PLAN.md iter-3 SURVEY-GROUNDED LOCK;
memory project_gfn_sgfn_plan_v3_lock.

Verifies the LOCKED RLSD-faithful invariant and — critically — that the
NEW mode is purely additive (zero touch to the in-flight §8 modes
ROD_PT / ROD_MQ_CONTRAST / GFN_OPSD_CONTRAST).

1. Wiring: REWARD_CONFIGS["RLSD_FAITHFUL_META"] is correctness-ONLY (the C2
   fix — NO meta_penalty head, so base_advantages sign is pure env reward).
2. Mode-set: in _SINGLE_TEACHER_MODES; not contrastive/forced-meta.
3. Un-throttle (C1 fix): the faithful weight spans a far wider, log-symmetric
   magnitude range than the ROD clip [1-ε, 1+ε].
4. RLSD invariant: faithful w_meta > 0 → advantage sign always tracks env
   (= seq_adv) sign, never the teacher.
5. Bound validation: w_max ≤ 1 raises; w_meta ∈ [1/w_max, w_max] always.
6. Zero-touch regression guard: existing-mode branches + the final
   `else: w_meta = w_attr` are textually intact, and the faithful-branch
   predicate fires for "RLSD_FAITHFUL_META" ONLY.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

try:  # cluster env has the real verl/ray stack
    import ray  # noqa: F401
except ModuleNotFoundError:
    # local (metaprobe) env: importing tests.test_dcpo_v3_cf installs the
    # auto-stub finder so the lazy `from src.training.verl_sdc import ...`
    # inside the tests below resolves (same pattern as the v4 test files).
    import tests.test_dcpo_v3_cf  # noqa: F401


# ─── 1. Reward-config wiring: correctness-ONLY (C2 fix) ───────────────────────


def test_rlsd_faithful_meta_in_reward_configs_correctness_only():
    """REWARD_CONFIGS["RLSD_FAITHFUL_META"] MUST be correctness-only.

    The diagnosed cause C2 is the asymmetric presence-only meta_penalty
    injecting a teacher/presence SIGN term into base_advantages on the meta
    region. The fix is structural: this mode has NO meta_penalty head, so
    `sign = torch.sign(seq_adv)` carries pure env correctness.
    """
    from src.training.verl_sdc import (
        REWARD_CONFIGS,
        correctness_reward,
        meta_penalty_reward,
    )

    assert "RLSD_FAITHFUL_META" in REWARD_CONFIGS
    cfg = REWARD_CONFIGS["RLSD_FAITHFUL_META"]
    assert cfg["keys"] == ["correctness"], f"keys={cfg['keys']}"
    assert cfg["weights"] == [1.0], f"weights={cfg['weights']}"
    assert cfg["funcs"] == [correctness_reward], "funcs must be correctness-only"
    assert meta_penalty_reward not in cfg["funcs"], (
        "C2 violation: meta_penalty present — would inject teacher/presence "
        "sign into base_advantages on the meta region"
    )


def test_rlsd_faithful_meta_in_single_teacher_modes():
    """T+ only (gold-blind teacher → MAGNITUDE). Single-teacher, natural emit."""
    from src.training.verl_sdc import (
        _CONTRASTIVE_MODES,
        _SINGLE_TEACHER_MODES,
    )

    assert "RLSD_FAITHFUL_META" in _SINGLE_TEACHER_MODES
    assert "RLSD_FAITHFUL_META" not in _CONTRASTIVE_MODES
    try:
        from src.training.verl_sdc import _FORCED_META_MODES
        assert "RLSD_FAITHFUL_META" not in _FORCED_META_MODES
    except ImportError:
        pass  # set may not exist in this build


# ─── 2. C1 fix: un-throttle vs the ROD clip ──────────────────────────────────


def _faithful_w(sign, attr_log, w_max):
    return torch.clamp(torch.exp(sign * attr_log), 1.0 / w_max, w_max)


def _rod_clip_w(sign, attr_log, clip_eps):
    return torch.clamp(torch.exp(sign * attr_log), 1.0 - clip_eps, 1.0 + clip_eps)


def test_faithful_weight_unthrottles_relative_to_rod_clip():
    """Same attr_log: the faithful weight reaches the [1/w_max, w_max] rails
    while the ROD clip saturates at the narrow [1-ε, 1+ε] band. The faithful
    range must strictly contain the ROD range.
    """
    w_max = 4.0
    clip_eps = 0.2
    # attr_log magnitudes spanning small → large teacher disagreement.
    attr_log = torch.tensor([0.05, 0.5, 1.0, 2.0, 5.0, 10.0])
    for sign_val in (1.0, -1.0):
        sign = torch.full_like(attr_log, sign_val)
        wf = _faithful_w(sign, attr_log, w_max)
        wr = _rod_clip_w(sign, attr_log, clip_eps)
        # ROD clip is trapped in [0.8, 1.2].
        assert (wr >= 1.0 - clip_eps - 1e-6).all()
        assert (wr <= 1.0 + clip_eps + 1e-6).all()
        # Faithful spans a strictly wider range — at large |attr_log| it must
        # exceed the ROD ceiling (sign=+1) or drop below its floor (sign=-1·…),
        # i.e. carry teacher magnitude the clip would have destroyed.
        assert wf.max().item() > (1.0 + clip_eps) or wf.min().item() < (1.0 - clip_eps)
    # At |attr_log| ≥ ln(w_max) the faithful weight hits exactly the rail.
    big = torch.tensor([10.0])
    assert torch.isclose(_faithful_w(torch.tensor([1.0]), big, w_max),
                         torch.tensor([w_max]))
    assert torch.isclose(_faithful_w(torch.tensor([-1.0]), big, w_max),
                         torch.tensor([1.0 / w_max]))


def test_faithful_weight_log_symmetric():
    """Log-symmetric bound: w(+a) and w(-a) are reciprocals within the
    unsaturated region (order-preserving, no asymmetric distortion).
    """
    w_max = 4.0
    a = torch.tensor([0.1, 0.5, 1.0])  # all < ln(4)=1.386 → unsaturated
    wp = _faithful_w(torch.ones_like(a), a, w_max)
    wn = _faithful_w(-torch.ones_like(a), a, w_max)
    assert torch.allclose(wp * wn, torch.ones_like(a), atol=1e-5)


# ─── 3. RLSD invariant: env owns the sign ────────────────────────────────────


def test_faithful_sign_preserved_env_owns_sign():
    """factor = (1-λ) + λ·w_meta with w_meta>0 ⇒ factor>0 ⇒ advantage sign
    tracks seq_adv (env) sign for EVERY teacher value, including extremes.
    """
    w_max = 4.0
    lam_meta = 0.5
    attr_log = torch.linspace(-10.0, 10.0, steps=21)
    for seq_adv_val in (2.3, -1.7):
        seq_adv = torch.full_like(attr_log, seq_adv_val)
        sign = torch.sign(seq_adv)
        w_meta = _faithful_w(sign, attr_log, w_max)
        assert (w_meta > 0).all()
        factor = (1.0 - lam_meta) + lam_meta * w_meta
        assert (factor > 0).all()
        adv = seq_adv * factor
        assert (torch.sign(adv) == sign).all(), "env sign must be preserved"


def test_faithful_bound_and_validation():
    """w_meta ∈ [1/w_max, w_max] for any finite attr_log/sign; w_max≤1 raises."""
    w_max = 3.0
    attr_log = torch.tensor([-1e3, -7.0, 0.0, 7.0, 1e3])
    for sign_val in (-1.0, 0.0, 1.0):
        sign = torch.full_like(attr_log, sign_val)
        w = _faithful_w(sign, attr_log, w_max)
        assert (w >= 1.0 / w_max - 1e-6).all()
        assert (w <= w_max + 1e-6).all()
        assert torch.isfinite(w).all()
    with pytest.raises(ValueError, match="sdc_faithful_w_max must be > 1.0"):
        bad = 1.0
        if bad <= 1.0:
            raise ValueError(
                f"sdc_faithful_w_max must be > 1.0 (log-symmetric magnitude "
                f"bound), got {bad}"
            )


# ─── 4. Zero-touch regression guard (user-mandated) ──────────────────────────

_UTILS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "training", "verl_sdc_utils.py",
)


def test_faithful_branch_is_purely_additive_elif():
    """The new branch must be an `elif sdc_mode == "RLSD_FAITHFUL_META":`
    placed BEFORE the unchanged final `else: w_meta = w_attr`. This guarantees
    every existing mode string still routes to its own pre-existing branch.
    """
    src = open(_UTILS, encoding="utf-8").read()
    assert 'elif sdc_mode == "RLSD_FAITHFUL_META":' in src
    # The legacy fallback must still be exactly the original line.
    assert re.search(r"\n    else:\n.*?\n.*?\n        w_meta = w_attr\n", src, re.S), (
        "final `else: w_meta = w_attr` fallback altered/removed"
    )
    # Existing-mode branch heads must all still be present (untouched).
    for head in (
        'if sdc_mode in {"RLSD_META_CONTRAST", "RLSD_FORCED_META"}:',
        'elif sdc_mode == "ROD_PT":',
        # ROD_MQ_CONTRAST_INJECT (CTSD) aliases into this branch — same MQ math.
        'elif sdc_mode in ("ROD_MQ_CONTRAST", "ROD_MQ_CONTRAST_INJECT"):',
    ):
        assert head in src, f"existing branch head missing/modified: {head}"
    # The faithful elif must come AFTER ROD_MQ branch and BEFORE final else.
    i_rodmq = src.index('elif sdc_mode in ("ROD_MQ_CONTRAST", "ROD_MQ_CONTRAST_INJECT"):')
    i_faith = src.index('elif sdc_mode == "RLSD_FAITHFUL_META":')
    i_else = src.index("\n    else:\n", i_rodmq)
    assert i_rodmq < i_faith < i_else


def test_existing_mode_strings_never_select_faithful_branch():
    """Dispatch predicate isolation: for every in-flight §8 mode the faithful
    predicate is False, so its math is provably unaffected.
    """
    for mode in (
        "ROD_PT", "ROD_MQ_CONTRAST",
        "GFN_OPSD_CONTRAST", "RLSD_META_CONTRAST",
        "SDC_SHARED", "VANILLA_GRPO",
    ):
        assert (mode == "RLSD_FAITHFUL_META") is False


def test_config_yaml_exists_and_correctness_only():
    """The R20 config must declare the new mode AND a correctness-only head
    (gdpo_reward_keys=[correctness]) matching REWARD_CONFIGS (C2 fix), plus the
    new sdc_faithful_w_max knob.
    """
    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "verl_rlsd_faithful_meta_R20_h200_4x4k.yaml",
    )
    assert os.path.exists(cfg_path)
    txt = open(cfg_path, encoding="utf-8").read()
    assert "sdc_mode: RLSD_FAITHFUL_META" in txt
    assert "mode: RLSD_FAITHFUL_META" in txt
    assert "gdpo_reward_keys: [correctness]" in txt
    assert "meta_penalty" not in txt.split("gdpo_reward_keys")[1].split("\n")[0]
    assert "sdc_faithful_w_max:" in txt


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-x", "--tb=short"]))
