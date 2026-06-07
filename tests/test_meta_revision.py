"""Unit tests for TRIOBJ_META_V1 meta-revision reward (pure python, no GPU/verl).

Run:
    /home/v-seungplee/miniconda3/envs/metaprobe/bin/python -m pytest tests/test_meta_revision.py -q

Covers the outcome-gated branches of meta_revision_utility_reward, including the
post-review anti-hacking guards:
  L1 — identical (un-revised) boxed answers earn nothing.
  L2 — the +1.0 wrong->right bonus requires the meta block to LOCALIZE the error
       (verify signal AND redirect/anomaly signal); otherwise discounted to +0.30.
  L3 — empty \\boxed{} does not count as a sincere attempt.
  L4 — genuine/localization signals are scanned inside the meta block only.
Plus a smoke that REWARD_CONFIGS['TRIOBJ_META_V1'] has 3 funcs/weights/keys.
"""

import ast
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.training.meta_revision_rewards import meta_revision_utility_reward

# Genuine verification meta, but NO error-localization (verify keyword only).
_META_VERIFY = (
    "<|meta|> Let me double-check my arithmetic by re-adding the terms; "
    "verifying the result. Confidence: 0.9 <|/meta|>"
)
# Genuine verification AND error-localization (verify + mistake/let-me-try + fails).
_META_LOC = (
    "<|meta|> Let me verify: this fails the check — I made a mistake; "
    "let me try the substitution again. Confidence: 0.8 <|/meta|>"
)
GOLD = "42"


def _wrap(text):
    """TRL completion format: list[ {role, content} ]."""
    return [{"role": "assistant", "content": text}]


def test_wrong_to_right_localized_full_bonus():
    # answer1 wrong, answer2 right, meta verifies AND localizes -> +1.0
    text = f"Prelim \\boxed{{7}}\n{_META_LOC}\nFinal \\boxed{{42}}"
    assert meta_revision_utility_reward([_wrap(text)], ground_truth=[GOLD]) == [1.0]


def test_wrong_to_right_no_localization_discounted():
    # L2: wrong->right but meta does NOT localize the error -> discounted +0.30
    text = f"Prelim \\boxed{{7}}\n{_META_VERIFY}\nFinal \\boxed{{42}}"
    assert meta_revision_utility_reward([_wrap(text)], ground_truth=[GOLD]) == [0.30]


def test_right_to_wrong_destructive_revision():
    # answer1 right, answer2 wrong -> -1.0 (SCoRe)
    text = f"Prelim \\boxed{{42}}\n{_META_VERIFY}\nFinal \\boxed{{7}}"
    assert meta_revision_utility_reward([_wrap(text)], ground_truth=[GOLD]) == [-1.0]


def test_right_to_right_revised_genuine_meta_confirmation():
    # right->right, REVISED form (42 -> 42.0, both correct), genuine meta -> +0.15
    text = f"Prelim \\boxed{{42}}\n{_META_VERIFY}\nFinal \\boxed{{42.0}}"
    assert meta_revision_utility_reward([_wrap(text)], ground_truth=[GOLD]) == [0.15]


def test_right_to_right_duplicate_box_no_credit():
    # L1: identical boxed answers (no real revision) -> 0.0 even with genuine meta
    text = f"Prelim \\boxed{{42}}\n{_META_VERIFY}\nFinal \\boxed{{42}}"
    assert meta_revision_utility_reward([_wrap(text)], ground_truth=[GOLD]) == [0.0]


def test_right_to_right_revised_no_meta_overcheck():
    # right->right, revised, NO genuine meta -> -0.10 (over-check)
    text = "Prelim \\boxed{42}\nyeah looks fine\nFinal \\boxed{42.0}"
    assert meta_revision_utility_reward([_wrap(text)], ground_truth=[GOLD]) == [-0.10]


def test_both_wrong_zero():
    text = f"Prelim \\boxed{{7}}\n{_META_LOC}\nFinal \\boxed{{9}}"
    assert meta_revision_utility_reward([_wrap(text)], ground_truth=[GOLD]) == [0.0]


def test_single_boxed_no_credit():
    text = f"{_META_LOC}\nFinal \\boxed{{42}}"
    assert meta_revision_utility_reward([_wrap(text)], ground_truth=[GOLD]) == [0.0]
    assert meta_revision_utility_reward([_wrap("no box here")], ground_truth=[GOLD]) == [0.0]


def test_empty_box_not_counted():
    # L3: an empty \boxed{} is dropped, leaving one real answer -> 0.0 (no two-pass)
    text = f"Prelim \\boxed{{}}\n{_META_LOC}\nFinal \\boxed{{42}}"
    assert meta_revision_utility_reward([_wrap(text)], ground_truth=[GOLD]) == [0.0]


def test_clip_and_batch_length():
    completions = [
        _wrap(f"\\boxed{{7}}\n{_META_LOC}\n\\boxed{{42}}"),     # +1.0 (localized)
        _wrap(f"\\boxed{{42}}\n{_META_VERIFY}\n\\boxed{{7}}"),  # -1.0
        _wrap("\\boxed{42}"),                                   #  0.0 (single)
    ]
    out = meta_revision_utility_reward(completions, ground_truth=[GOLD, GOLD, GOLD])
    assert len(out) == 3
    assert all(-1.0 <= s <= 1.0 for s in out)
    assert out == [1.0, -1.0, 0.0]


def test_no_ground_truth_does_not_crash():
    text = f"\\boxed{{7}}\n{_META_LOC}\n\\boxed{{42}}"
    out = meta_revision_utility_reward([_wrap(text)], ground_truth=None)
    assert len(out) == 1
    assert -1.0 <= out[0] <= 1.0


# --- REWARD_CONFIGS smoke --------------------------------------------------
def _reward_configs_entry_via_ast():
    path = os.path.join(REPO, "src", "training", "verl_sdc.py")
    with open(path) as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "REWARD_CONFIGS" and isinstance(node.value, ast.Dict):
                    for k, v in zip(node.value.keys, node.value.values):
                        if isinstance(k, ast.Constant) and k.value == "TRIOBJ_META_V1" and isinstance(v, ast.Dict):
                            entry = {}
                            for kk, vv in zip(v.keys, v.values):
                                if isinstance(kk, ast.Constant) and isinstance(vv, (ast.List, ast.Tuple)):
                                    entry[kk.value] = len(vv.elts)
                            return entry
    return None


def test_reward_configs_triobj_meta_v1_smoke():
    try:
        from src.training.verl_sdc import REWARD_CONFIGS  # needs verl
        entry = REWARD_CONFIGS["TRIOBJ_META_V1"]
        assert len(entry["funcs"]) == 3 and len(entry["weights"]) == 3 and len(entry["keys"]) == 3
        assert all(callable(fn) for fn in entry["funcs"])
        assert entry["weights"] == [1.0, 0.5, 0.3]
        assert entry["keys"] == ["correctness", "meta_revision_utility", "meta_commit_shape"]
        assert any(fn.__name__ == "meta_revision_utility_reward" for fn in entry["funcs"])
    except ImportError:
        entry = _reward_configs_entry_via_ast()
        assert entry is not None, "TRIOBJ_META_V1 not found in REWARD_CONFIGS"
        assert entry.get("funcs") == 3 and entry.get("weights") == 3 and entry.get("keys") == 3


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
