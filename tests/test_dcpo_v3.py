"""Unit tests for TRIOBJ_DCPO_V3 — counterfactual meta-ablation R_meta.

PURE PYTHON (runs under /home/v-seungplee/miniconda3/envs/metaprobe/bin/python).
Covers (spec §9):
  - first_meta_token_index: one / many / zero <|meta|> ids; respects response_mask.
  - R_meta = c_with - c_without for all 4 cases + the None (no-CF) case.
  - R_meta = 0 when cf_correct is None for a rollout.
  - R_cal Brier on parsed confidence (-(conf - c_with)^2).
  - region routing (compose_dcpo_region_advantage) unchanged.
  - cf_answer_from_prefix text fallback.
"""
import numpy as np
import torch

from src.training.dcpo_region import (
    first_meta_token_index,
    first_meta_index,
    cf_answer_from_prefix,
    dcpo_region_rewards,
    compose_dcpo_region_advantage,
    group_mean_subtract,
)

META_OPEN = 151669
META_CLOSE = 151670


# ── completion helper (TRL format) ──────────────────────────────────────────
def _c(text):
    return [{"content": text}]


# ═══════════════════════════════════════════════════════════════════════════
# first_meta_token_index
# ═══════════════════════════════════════════════════════════════════════════
def test_first_meta_single():
    ids = [1, 2, META_OPEN, 3, META_CLOSE, 4]
    assert first_meta_token_index(ids) == 2


def test_first_meta_many_returns_first():
    ids = [1, META_OPEN, 2, META_CLOSE, 3, META_OPEN, 4, META_CLOSE]
    assert first_meta_token_index(ids) == 1


def test_first_meta_none():
    ids = [1, 2, 3, 4]
    assert first_meta_token_index(ids) is None


def test_first_meta_respects_mask():
    # the first <|meta|> sits on a MASKED (pad) position → skipped; second is real.
    ids = [1, META_OPEN, 2, META_OPEN, 3]
    mask = [True, False, True, True, True]
    assert first_meta_token_index(ids, mask) == 3


def test_first_meta_mask_all_false_no_meta():
    ids = [META_OPEN, META_OPEN]
    mask = [False, False]
    assert first_meta_token_index(ids, mask) is None


def test_first_meta_accepts_numpy_and_tensor():
    ids = np.array([1, META_OPEN, 2])
    assert first_meta_token_index(ids) == 1
    t = torch.tensor([5, 6, META_OPEN])
    assert first_meta_token_index(t) == 2


def test_first_meta_alias():
    assert first_meta_index is first_meta_token_index


# ═══════════════════════════════════════════════════════════════════════════
# R_meta = c_with - c_without  (the 4 cases + None)
# ═══════════════════════════════════════════════════════════════════════════
# A single rollout with a meta block; we drive c_with via the main answer (matches
# gt or not) and c_without via the cf_correct array.
def _meta_text(answer):
    # main rollout: reasoning, a meta block (with conf so R_cal is exercised), boxed.
    return (
        f"reasoning <|meta|> let me verify; confidence: 0.80 <|/meta|> "
        f"The answer is \\boxed{{{answer}}}"
    )


def _rewards(main_answer, gt, cf_correct):
    return dcpo_region_rewards(
        [_c(_meta_text(main_answer))],
        ground_truth=[gt],
        group_index=["g"],
        cf_correct=[cf_correct],
    )


def test_rmeta_with_right_without_wrong_plus1():
    # main correct (c_with=1), counterfactual wrong (c_without=0) → +1
    out = _rewards("5", "5", cf_correct=0.0)
    assert out["R_meta"][0] == 1.0


def test_rmeta_both_right_zero():
    out = _rewards("5", "5", cf_correct=1.0)
    assert out["R_meta"][0] == 0.0


def test_rmeta_both_wrong_zero():
    out = _rewards("4", "5", cf_correct=0.0)
    assert out["R_meta"][0] == 0.0


def test_rmeta_with_wrong_without_right_minus1():
    # main wrong (c_with=0), counterfactual right (c_without=1) → -1
    out = _rewards("4", "5", cf_correct=1.0)
    assert out["R_meta"][0] == -1.0


def test_rmeta_cf_none_is_zero():
    # cf_correct None for the rollout AND no pre-meta answer to fall back on → R_meta 0.
    out = _rewards("5", "5", cf_correct=None)
    assert out["R_meta"][0] == 0.0


def test_rmeta_cf_correct_array_none_entry():
    # cf_correct array present but this entry None → R_meta 0 (no crash).
    out = dcpo_region_rewards(
        [_c(_meta_text("5"))],
        ground_truth=["5"],
        group_index=["g"],
        cf_correct=[None],
    )
    assert out["R_meta"][0] == 0.0


# ── v3b BUG-2 regression: np.float32 NaN sentinel must NOT read as c_without=True ──
def test_rmeta_npfloat32_nan_is_none_not_true():
    # The producer ships cf_correct as np.float32 with NaN for skipped rows.
    # np.float32 is NOT a python-float subclass, so isinstance-gated NaN checks
    # miss it and bool(nan)=True turned every skipped+wrong row into spurious -1.
    # NaN must behave exactly like None: no meta → R_meta 0 even when main is wrong.
    nan_arr = np.asarray([float("nan")], dtype=np.float32)
    out = dcpo_region_rewards(
        [_c("no meta here. The answer is \\boxed{4}")],   # wrong (gt=5), NO meta
        ground_truth=["5"],
        group_index=["g"],
        cf_correct=list(nan_arr),
    )
    assert out["R_meta"][0] == 0.0   # pre-fix this was -1.0 (the v3b artifact)


def test_rmeta_npfloat32_real_values_still_work():
    arr = np.asarray([0.0], dtype=np.float32)   # CF wrong, main right → +1
    out = dcpo_region_rewards(
        [_c(_meta_text("5"))], ground_truth=["5"], group_index=["g"],
        cf_correct=list(arr),
    )
    assert out["R_meta"][0] == 1.0


# ── v3b BUG-1 regression: the producer→consumer cf_texts handoff (cf_completions) ──
def test_rmeta_cf_completions_graded_with_real_gt():
    # CF text answers 4 (wrong vs gt=5), main answers 5 (right) → R_meta +1.
    # This is the deployed path now: producer stashes TEXTS, consumer grades here
    # with the real ground truth (the producer-side grade saw gt="" → c_without≡0).
    out = dcpo_region_rewards(
        [_c(_meta_text("5"))],
        ground_truth=["5"],
        group_index=["g"],
        cf_completions=["after more thought, The answer is \\boxed{4}"],
    )
    assert out["R_meta"][0] == 1.0


def test_rmeta_cf_completions_correct_cf_zero():
    # CF also right → meta made no causal difference → 0.
    out = dcpo_region_rewards(
        [_c(_meta_text("5"))],
        ground_truth=["5"],
        group_index=["g"],
        cf_completions=["The answer is \\boxed{5}"],
    )
    assert out["R_meta"][0] == 0.0


def test_rmeta_cf_completions_none_entry_falls_back():
    # None entry (skipped/no-CF row) + no meta in main → R_meta 0.
    out = dcpo_region_rewards(
        [_c("plain. The answer is \\boxed{4}")],
        ground_truth=["5"],
        group_index=["g"],
        cf_completions=[None],
    )
    assert out["R_meta"][0] == 0.0


def test_rmeta_no_cf_args_uses_text_fallback():
    # No cf_correct / cf_completions: text fallback grades the pre-meta prefix.
    # Pre-meta prefix here has a boxed answer that is WRONG; main is RIGHT → +1.
    text = "draft \\boxed{4} <|meta|> recheck; confidence: 0.70 <|/meta|> \\boxed{5}"
    out = dcpo_region_rewards([_c(text)], ground_truth=["5"], group_index=["g"])
    # c_with = correct(final=5) = 1 ; c_without = correct(prefix=4) = 0 → +1
    assert out["R_meta"][0] == 1.0


def test_rmeta_no_meta_rollout_zero():
    # No <|meta|> at all → text fallback returns None → R_meta 0 (no penalty).
    out = dcpo_region_rewards(
        [_c("just \\boxed{5}")], ground_truth=["5"], group_index=["g"]
    )
    assert out["R_meta"][0] == 0.0


def test_rmeta_all_four_cases_vectorized():
    # one group of 4 rollouts hitting each case; cf_correct supplied per row.
    comps = [_c(_meta_text(a)) for a in ("5", "5", "4", "4")]
    gts = ["5", "5", "5", "5"]
    cf = [0.0, 1.0, 0.0, 1.0]   # without: wrong, right, wrong, right
    out = dcpo_region_rewards(comps, ground_truth=gts, group_index=["g"] * 4, cf_correct=cf)
    # with: 1,1,0,0 → delta: +1, 0, 0, -1
    assert out["R_meta"] == [1.0, 0.0, 0.0, -1.0]


# ═══════════════════════════════════════════════════════════════════════════
# R_cal Brier on conf  ( -(conf - c_with)^2 )
# ═══════════════════════════════════════════════════════════════════════════
def test_rcal_brier_correct():
    # conf 0.80 (parsed), main correct → c_with=1 → -(0.8-1)^2 = -0.04
    out = _rewards("5", "5", cf_correct=0.0)
    assert abs(out["R_cal"][0] - (-(0.80 - 1.0) ** 2)) < 1e-9


def test_rcal_brier_wrong():
    # conf 0.80, main wrong → c_with=0 → -(0.8-0)^2 = -0.64
    out = _rewards("4", "5", cf_correct=0.0)
    assert abs(out["R_cal"][0] - (-(0.80 - 0.0) ** 2)) < 1e-9


def test_rcal_zero_when_no_conf():
    # meta block with NO confidence number → R_cal 0.
    text = "reasoning <|meta|> just a note, no number <|/meta|> \\boxed{5}"
    out = dcpo_region_rewards([_c(text)], ground_truth=["5"], group_index=["g"], cf_correct=[0.0])
    assert out["R_cal"][0] == 0.0


def test_rcal_independent_of_counterfactual():
    # R_cal uses c_with (main correctness), NOT the counterfactual. Same main answer
    # → same R_cal regardless of cf_correct.
    a = _rewards("5", "5", cf_correct=0.0)["R_cal"][0]
    b = _rewards("5", "5", cf_correct=1.0)["R_cal"][0]
    assert a == b


# ═══════════════════════════════════════════════════════════════════════════
# R_corr unchanged + diagnostics + stubs
# ═══════════════════════════════════════════════════════════════════════════
def test_rcorr_pm1():
    assert _rewards("5", "5", cf_correct=0.0)["R_corr"][0] == 1.0
    assert _rewards("4", "5", cf_correct=0.0)["R_corr"][0] == -1.0


def test_dropped_kwargs_ignored():
    # v2 carry-over kwargs are accepted-but-ignored (caller compat).
    out = dcpo_region_rewards(
        [_c(_meta_text("5"))],
        ground_truth=["5"],
        group_index=["g"],
        cf_correct=[0.0],
        eps=0.1, p_lo=0.2, p_hi=0.8, warmup_steps=200,
        sandbag_clamp=True, sandbag_floor=0.05,
        format_credit=0.05, format_penalty=0.05,
        some_unknown_future_knob=123,
    )
    assert out["R_meta"][0] == 1.0
    # constant stubs present so existing wandb keys stay alive.
    assert out["canary_pass1_acc"] == [1.0]
    assert out["sandbag_clamp"] == [1.0]
    assert "p_hat" in out and "group_acc" in out


# ═══════════════════════════════════════════════════════════════════════════
# cf_answer_from_prefix
# ═══════════════════════════════════════════════════════════════════════════
def test_cf_answer_from_prefix_extracts_premeta():
    text = "draft \\boxed{7} <|meta|> verify <|/meta|> \\boxed{5}"
    assert cf_answer_from_prefix(text) == "7"


def test_cf_answer_from_prefix_no_meta_none():
    assert cf_answer_from_prefix("just \\boxed{5}") is None


def test_cf_answer_from_prefix_meta_but_no_premeta_answer_none():
    # meta fires before any answer is written → no pre-meta answer → None (under-credit).
    assert cf_answer_from_prefix("<|meta|> verify <|/meta|> \\boxed{5}") is None


# ═══════════════════════════════════════════════════════════════════════════
# region routing UNCHANGED (compose_dcpo_region_advantage)
# ═══════════════════════════════════════════════════════════════════════════
def test_region_routing_unchanged_basic():
    # B=2 group; R_meta {+1, -1} routes ONLY to META_CONTENT; tag tokens get 0.
    # layout T=4: [ans, TAG, meta_c, conf]
    ans = [[1, 0, 0, 0], [0, 0, 0, 0]]
    meta_c = [[0, 0, 1, 1], [0, 0, 0, 0]]
    conf = [[0, 0, 0, 1], [0, 0, 0, 0]]
    rm = [[1, 1, 1, 1], [1, 1, 1, 1]]
    A, A2 = compose_dcpo_region_advantage(
        response_mask=torch.tensor(rm, dtype=torch.float32),
        index=["g", "g"],
        R_corr=np.asarray([1.0, -1.0], dtype=np.float32),
        R_meta=np.asarray([1.0, -1.0], dtype=np.float32),
        R_cal=np.asarray([0.0, 0.0], dtype=np.float32),
        answer_mask=torch.tensor(ans, dtype=torch.float32),
        meta_content_mask=torch.tensor(meta_c, dtype=torch.float32),
        conf_mask=torch.tensor(conf, dtype=torch.float32),
    )
    assert torch.equal(A, A2)
    row = A[0]
    # Â_meta = 1 - mean(1,-1) = 1 ; meta_content idx 2 → w_meta*Â_meta = 0.5
    assert torch.allclose(row[2], torch.tensor(0.5))
    # TAG idx 1 → 0 (neither answer nor content)
    assert row[1] == 0.0
    # answer idx 0 → w_corr*Â_corr = 1.0 (Â_corr = 1 - 0 = 1)
    assert torch.allclose(row[0], torch.tensor(1.0))


def test_group_mean_subtract_centers_group():
    out = group_mean_subtract(torch.tensor([1.0, 0.0, 0.0, -1.0]), ["g"] * 4).squeeze(1)
    assert abs(float(out.sum())) < 1e-6   # group-mean-subtract centers to ~0
