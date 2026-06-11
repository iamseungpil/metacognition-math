"""E.9 BCI-RLVR isolation regression.

Goal: prove the E.9 change is purely ADDITIVE — every pre-existing
REWARD_CONFIGS mode keeps its exact (func-names, weights, keys), and the new
BCI_RLVR entry exists with the spec'd heads.

verl is NOT installed locally, and verl_sdc.py imports verl/ray/torch/tensordict
at module top, so we STUB those modules in sys.modules before import. The reward
FUNCTIONS themselves come from the real (unstubbed) src.training.rewards, so the
`funcs` identities in REWARD_CONFIGS are genuine — we snapshot their __name__.
"""
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Every sys.modules name _install_verl_stubs may FORCE-replace. The fixture
# snapshots these before stubbing and restores them after the module's tests,
# so the stub torch (etc.) cannot leak into later test modules (real torch ops
# resolve `torch.SymFloat` through sys.modules at call time — a leaked stub
# breaks tensor indexing in every module that runs after this one).
_STUBBED_NAMES = (
    "torch", "ray", "tensordict", "numpy", "hydra",
    "verl", "verl.trainer", "verl.trainer.ppo", "verl.trainer.ppo.ray_trainer",
    "src.training.verl_sdc_utils", "src.training.verl_sdc",
)


def _install_verl_stubs():
    """Minimal fake modules so src.training.verl_sdc imports far enough to build
    REWARD_CONFIGS, without pulling in the real verl/ray/torch stack."""

    def _stub(name, force=False):
        # FORCE-replace for the heavy/partial verl stack: some local envs ship a
        # partial real `ray`/`verl` (e.g. ray without `.remote`) which breaks the
        # import for the WRONG reason. We want a clean controlled stub so the test
        # isolates REWARD_CONFIGS only. numpy/torch (used by real rewards? no) are
        # left alone via force=False.
        if force or name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
        return sys.modules[name]

    # torch — only attribute access at import time (Tensor type hints / decorators)
    torch = _stub("torch", force=True)
    torch.Tensor = type("Tensor", (), {})
    torch.no_grad = lambda *a, **k: (lambda f: f)

    # ray — FORCE stub (partial real ray in some envs lacks `.remote`)
    ray = _stub("ray", force=True)
    ray.remote = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
    ray.is_initialized = lambda: True
    ray.init = lambda *a, **k: None
    ray.get = lambda *a, **k: None

    # tensordict.TensorDict (FORCE — avoid pulling the real heavy dep)
    td = _stub("tensordict", force=True)
    td.TensorDict = type("TensorDict", (), {})

    # numpy (env may lack it; verl_sdc uses np at runtime only, not at import)
    if "numpy" not in sys.modules:
        try:
            import numpy  # noqa: F401
        except Exception:
            np = _stub("numpy")
            np.ndarray = type("ndarray", (), {})
            np.array = lambda *a, **k: None

    # verl package + the symbols verl_sdc imports by name (FORCE — local envs may
    # ship a partial real verl)
    verl = _stub("verl", force=True)
    verl.DataProto = type("DataProto", (), {})
    trainer = _stub("verl.trainer", force=True)
    ppo = _stub("verl.trainer.ppo", force=True)
    rt = _stub("verl.trainer.ppo.ray_trainer", force=True)

    class _RayPPOTrainer:  # base class verl_sdc subclasses
        def __init__(self, *a, **k):
            pass

    rt.RayPPOTrainer = _RayPPOTrainer
    rt.ResourcePoolManager = type("ResourcePoolManager", (), {})
    rt.Role = type("Role", (), {})
    verl.trainer = trainer
    trainer.ppo = ppo
    ppo.ray_trainer = rt

    # verl_sdc_utils imports verl/torch -> stub it with EVERY name verl_sdc pulls.
    # SYNC PAIR (round 2 IMPORTANT-4): this attribute list mirrors the
    # `from src.training.verl_sdc_utils import (...)` block in
    # src/training/verl_sdc.py — a name added there without a stub line here
    # makes THIS suite error at setup when run STANDALONE (the full suite hides
    # it via import order). test_stub_covers_verl_sdc_utils_import_list locks it.
    vsu = _stub("src.training.verl_sdc_utils", force=True)
    vsu.build_sdc_region_masks = lambda *a, **k: None
    vsu.compute_sdc_gdpo_advantage = lambda *a, **k: None
    vsu.dcpo_length_cost = lambda *a, **k: 0.0
    vsu.dcpo_w_meta_warmup_scale = lambda *a, **k: 1.0

    # postmeta_closure_reward is a REAL reward func used by SDC_SHARED's `funcs`.
    # It lives in verl_sdc_utils (torch/verl deps) so we can't import it here;
    # give the stub the correct __name__ so the funcs-name snapshot stays
    # faithful (we only ever assert on __name__, never call it).
    def postmeta_closure_reward(*a, **k):
        return None

    vsu.postmeta_closure_reward = postmeta_closure_reward

    # hydra — verl_sdc.py has a top-level `import hydra` + @hydra.main decorator
    # on main(). FORCE-stub so module import (and the decorator call) succeed.
    hydra = _stub("hydra", force=True)
    hydra.main = lambda *a, **k: (lambda f: f)


@pytest.fixture(scope="module")
def reward_configs():
    saved = {name: sys.modules.get(name) for name in _STUBBED_NAMES}
    _install_verl_stubs()
    from src.training.verl_sdc import REWARD_CONFIGS

    yield REWARD_CONFIGS
    for name, mod in saved.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod


def _snapshot(cfg):
    return {
        "funcs": [f.__name__ for f in cfg["funcs"]],
        "weights": list(cfg["weights"]),
        "keys": list(cfg["keys"]),
    }


# Hardcoded EXPECTED snapshot of every PRE-EXISTING mode (frozen pre-E.9).
EXPECTED = {
    "SDC_SHARED": {
        "funcs": ["correctness_reward", "outcome_calibration_reward",
                  "meta_structure_reward", "meta_commit_shape_reward",
                  "postmeta_closure_reward"],
        "weights": [1.0, 0.7, 0.25, 0.35, 0.45],
        "keys": ["correctness", "outcome_calibration", "meta_structure",
                 "meta_commit_shape", "postmeta_closure"],
    },
    "SDC_CORR_ONLY": {
        "funcs": ["correctness_reward"], "weights": [1.0], "keys": ["correctness"],
    },
    "SDC_CORR_META_PEN": {
        "funcs": ["correctness_reward", "meta_penalty_reward"],
        "weights": [1.0, 1.0], "keys": ["correctness", "meta_penalty"],
    },
    "VANILLA_GRPO": {
        "funcs": ["correctness_reward"], "weights": [1.0], "keys": ["correctness"],
    },
    "RLSD_META_ATTR": {
        "funcs": ["correctness_reward", "meta_penalty_reward"],
        "weights": [1.0, 1.0], "keys": ["correctness", "meta_penalty"],
    },
    "RLSD_META_CONTRAST": {
        "funcs": ["correctness_reward", "meta_penalty_reward"],
        "weights": [1.0, 1.0], "keys": ["correctness", "meta_penalty"],
    },
    "OPSD_META": {
        "funcs": ["correctness_reward", "meta_penalty_reward"],
        "weights": [1.0, 1.0], "keys": ["correctness", "meta_penalty"],
    },
    "RLSD_FORCED_META": {
        "funcs": ["correctness_reward", "meta_penalty_reward"],
        "weights": [1.0, 1.0], "keys": ["correctness", "meta_penalty"],
    },
    "ROD_PT": {
        "funcs": ["correctness_reward", "meta_penalty_reward"],
        "weights": [1.0, 1.0], "keys": ["correctness", "meta_penalty"],
    },
    "ROD_PT_DEGEN": {
        "funcs": ["correctness_reward", "meta_penalty_reward",
                  "degeneration_penalty_reward"],
        "weights": [1.0, 1.0, 0.3],
        "keys": ["correctness", "meta_penalty", "degeneration_penalty"],
    },
    "ROD_MQ": {
        "funcs": ["correctness_reward", "meta_penalty_reward"],
        "weights": [1.0, 1.0], "keys": ["correctness", "meta_penalty"],
    },
    "ROD_MQ_CONTRAST": {
        "funcs": ["correctness_reward", "meta_penalty_reward"],
        "weights": [1.0, 1.0], "keys": ["correctness", "meta_penalty"],
    },
    "ROD_MQ_CONTRAST_INJECT": {
        "funcs": ["correctness_reward", "meta_penalty_reward"],
        "weights": [1.0, 1.0], "keys": ["correctness", "meta_penalty"],
    },
    "GFN_OPSD_CONTRAST": {
        "funcs": ["correctness_reward", "meta_penalty_reward"],
        "weights": [1.0, 1.0], "keys": ["correctness", "meta_penalty"],
    },
    "RLSD_FAITHFUL_META": {
        "funcs": ["correctness_reward"], "weights": [1.0], "keys": ["correctness"],
    },
    "STABLE_GFN": {
        "funcs": ["correctness_reward", "meta_penalty_reward"],
        "weights": [1.0, 1.0], "keys": ["correctness", "meta_penalty"],
    },
    "ROD_PT2_E21CTRL": {
        "funcs": ["correctness_reward", "confidence_revision_reward",
                  "redirect_execution_reward", "verify_execution_reward",
                  "confidence_omission_floor", "meta_count_bonus",
                  "meta_penalty_adaptive_reward"],
        "weights": [1.0, 0.35, 0.30, 0.15, 0.5, 1.0, 1.0],
        "keys": ["correctness", "confidence_revision", "redirect_execution",
                 "verify_execution", "meta_floor", "meta_count_bonus",
                 "meta_penalty_adaptive"],
    },
    "STABLE_GFN_C2FIX": {
        "funcs": ["correctness_reward", "meta_penalty_adaptive_reward"],
        "weights": [1.0, 1.0], "keys": ["correctness", "meta_penalty_adaptive"],
    },
    "MATCHED_E21RV2": {
        "funcs": ["correctness_reward", "confidence_revision_reward",
                  "redirect_execution_reward", "verify_execution_reward",
                  "confidence_omission_floor", "meta_count_bonus"],
        "weights": [1.0, 0.35, 0.30, 0.15, 0.5, 1.0],
        "keys": ["correctness", "confidence_revision", "redirect_execution",
                 "verify_execution", "meta_floor", "meta_count_bonus"],
    },
    # tri-objective family (post-E.9 additions, snapshotted on registration)
    "TRIOBJ_META_V1": {
        "funcs": ["correctness_reward", "meta_revision_utility_reward",
                  "meta_commit_shape_reward"],
        "weights": [1.0, 0.5, 0.3],
        "keys": ["correctness", "meta_revision_utility", "meta_commit_shape"],
    },
    "TRIOBJ_DCPO_V2": {
        "funcs": ["correctness_region_reward", "meta_region_utility_reward",
                  "cal_region_reward"],
        "weights": [1.0, 0.5, 0.3],
        "keys": ["correctness", "meta_region_utility", "cal_region_reward"],
    },
    "TRIOBJ_DCPO_V3": {
        "funcs": ["correctness_region_reward", "meta_region_utility_reward",
                  "cal_region_reward", "meta_emission_reward",
                  "format_penalty_reward"],
        "weights": [1.0, 0.5, 0.3, 0.0, 0.1],
        "keys": ["correctness", "meta_region_utility", "cal_region_reward",
                 "meta_emission", "format_penalty"],
    },
    "TRIOBJ_DCPO_V4": {
        "funcs": ["correctness_region_reward", "meta_region_utility_reward",
                  "cal_region_reward", "meta_emission_reward",
                  "format_penalty_reward"],
        "weights": [1.0, 0.5, 0.3, 0.0, 0.1],
        "keys": ["correctness", "meta_region_utility", "cal_region_reward",
                 "meta_emission", "format_penalty"],
    },
}


def test_every_preexisting_mode_byte_identical(reward_configs):
    for mode, exp in EXPECTED.items():
        assert mode in reward_configs, f"pre-existing mode {mode} vanished"
        got = _snapshot(reward_configs[mode])
        assert got == exp, f"mode {mode} changed:\n  got={got}\n  exp={exp}"


def test_no_preexisting_mode_dropped(reward_configs):
    # every key in REWARD_CONFIGS is either a frozen pre-existing mode or BCI_RLVR
    known = set(EXPECTED) | {"BCI_RLVR"}
    extra = set(reward_configs) - known
    assert not extra, f"unexpected new modes (snapshot the test): {extra}"
    missing = set(EXPECTED) - set(reward_configs)
    assert not missing, f"pre-existing modes missing: {missing}"


def test_bci_rlvr_exists_with_spec_heads(reward_configs):
    assert "BCI_RLVR" in reward_configs
    got = _snapshot(reward_configs["BCI_RLVR"])
    assert got == {
        "funcs": ["correctness_reward", "outcome_calibration_reward"],
        "weights": [1.0, 0.5],
        "keys": ["correctness", "outcome_calibration"],
    }, got


def test_stub_covers_verl_sdc_utils_import_list(reward_configs):
    # Round 2 IMPORTANT-4 lock for the SYNC PAIR: every name verl_sdc.py pulls
    # from src.training.verl_sdc_utils must exist on the stub module, else this
    # suite errors at setup when run STANDALONE (the dcpo_length_cost regression
    # — 3 setup ERRORS hidden in the full suite by import order).
    import re

    src_path = os.path.join(os.path.dirname(__file__), "..",
                            "src", "training", "verl_sdc.py")
    with open(src_path) as f:
        src = f.read()
    m = re.search(r"from src\.training\.verl_sdc_utils import \(([^)]*)\)", src)
    assert m, "verl_sdc.py verl_sdc_utils import block not found"
    names = [n.strip().rstrip(",") for n in m.group(1).splitlines() if n.strip()]
    assert names, "empty verl_sdc_utils import list?"
    vsu = sys.modules["src.training.verl_sdc_utils"]  # the installed stub
    missing = [n for n in names if not hasattr(vsu, n)]
    assert not missing, (
        f"verl_sdc_utils stub out of sync with verl_sdc.py import list "
        f"(add stub lines in _install_verl_stubs): {missing}")
