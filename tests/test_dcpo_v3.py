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


def test_meta_emission_reward_observability_func():
    # weight-0.0 val observability func: 1.0 iff <|meta|> present; never affects reward.
    import tests.test_dcpo_v3_cf  # installs the auto-stub finder (verl absent locally)
    from src.training.verl_sdc import meta_emission_reward, REWARD_CONFIGS
    out = meta_emission_reward([_c(_meta_text("5")), _c("plain answer 4")])
    assert out == [1.0, 0.0]
    cfg = REWARD_CONFIGS["TRIOBJ_DCPO_V3"]
    i = cfg["keys"].index("meta_emission")
    assert cfg["weights"][i] == 0.0          # MUST stay observability-only
    assert cfg["keys"][:3] == ["correctness", "meta_region_utility", "cal_region_reward"]


def test_v3_yaml_reward_lists_match_reward_configs():
    # REGRESSION (v3f boot crash 2026-06-10): main_task validates
    # len(yaml gdpo_reward_keys/weights) == len(REWARD_CONFIGS funcs). The v3e
    # release added meta_emission as a 4th func but left the yaml lists at 3,
    # killing the run at boot. Keep them in lockstep.
    import os
    import yaml as _yaml
    import tests.test_dcpo_v3_cf  # auto-stub
    from src.training.verl_sdc import REWARD_CONFIGS
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs",
                            "triobj_dcpo_v3_h100_4x4k.yaml")
    with open(cfg_path) as f:
        ycfg = _yaml.safe_load(f)
    alg = ycfg["algorithm"]
    rc = REWARD_CONFIGS["TRIOBJ_DCPO_V3"]
    assert alg["gdpo_reward_keys"] == rc["keys"]
    assert [float(w) for w in alg["gdpo_reward_weights"]] == [float(w) for w in rc["weights"]]
    assert len(rc["funcs"]) == len(rc["keys"]) == len(rc["weights"])


def test_populate_writes_every_gdpo_reward_key():
    # REGRESSION (v3g step-1 crash 2026-06-10): the async-path populator
    # (_populate_dcpo_region_keys) must write EVERY key in gdpo_reward_keys into
    # non_tensor_batch — the GDPO advantage assertion requires all of them, and
    # the RewardLoopWorker placeholders do not cover mode-specific extras like
    # meta_emission. Source-level check: each configured key appears as a
    # non_tensor_batch["<key>"] write in the populator.
    import inspect
    import tests.test_dcpo_v3_cf  # auto-stub
    from src.training.verl_sdc import _populate_dcpo_region_keys, REWARD_CONFIGS
    src = inspect.getsource(_populate_dcpo_region_keys)
    for key in REWARD_CONFIGS["TRIOBJ_DCPO_V3"]["keys"]:
        assert f'non_tensor_batch["{key}"]' in src, f"populator does not write {key!r}"


def test_trend_scalar_helper_never_raises():
    import tests.test_dcpo_v3_cf  # auto-stub
    from src.training.verl_sdc import _log_dcpo_trend_scalars
    heads = {"has_meta": [True, False], "R_meta": [1.0, 0.0], "c_with": [1.0, 0.0],
             "c_without": [0.0, float("nan")]}
    # wandb stubbed/absent -> must silently no-op, not raise.
    _log_dcpo_trend_scalars(step=3, heads=heads, cf_texts=["cf", None])


def test_diagnostic_keys_for_rollout_table():
    # The wandb rollout table reads c_with / c_without / conf / has_meta / answer.
    out = dcpo_region_rewards(
        [_c(_meta_text("5")), _c("plain. The answer is \\boxed{7}")],
        ground_truth=["5", "5"],
        group_index=["g", "g"],
        cf_completions=["The answer is \\boxed{4}", None],
    )
    assert out["c_with"] == [1.0, 0.0]
    assert out["c_without"][0] == 0.0          # CF graded wrong with real gt
    assert out["c_without"][1] != out["c_without"][1]  # NaN (no CF, no meta)
    assert abs(out["conf"][0] - 0.80) < 1e-9   # parsed from the meta block
    assert out["has_meta"] == [True, False]
    assert out["answer"] == ["5", "7"]


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


# ═══════════════════════════════════════════════════════════════════════════
# v3 FORMAT FIX — unclosed-meta mask clamp + gate + penalty + 4th routed head
# (live-run finding: 40% unclosed meta; the old unclosed-to-end rule put the
#  FINAL ANSWER inside META_CONTENT for 17% of rollouts → R_corr misrouted)
# ═══════════════════════════════════════════════════════════════════════════
from src.training.dcpo_region import build_dcpo_region_masks, THINK_CLOSE_DEFAULT

THINK_CLOSE = 151668

# Fake token vocab (decode = concatenation, exact char offsets — same pattern
# as tests/test_dcpo_region.py).
_VOCAB = {
    META_OPEN: "<|meta|>",
    META_CLOSE: "<|/meta|>",
    THINK_CLOSE: "</think>",
    1: "reason ",
    2: "confidence:",
    3: " 0",
    4: ".",
    5: "8",
    6: "final ",
    7: "\\boxed{5}",
}


def _decode(ids):
    return "".join(_VOCAB.get(int(t), "?") for t in ids)


def _m(ids):
    return build_dcpo_region_masks(ids, [True] * len(ids), _decode)


def test_think_close_default_id():
    assert THINK_CLOSE_DEFAULT == 151668


def test_mask_closed_span_unchanged():
    # CLOSED block: byte-identical to the pre-fix behaviour (content + conf parsed).
    ids = [1, META_OPEN, 2, 3, 4, 5, META_CLOSE, 6, 7]
    m = _m(ids)
    assert np.array_equal(np.where(m["META_CONTENT"])[0], np.array([2, 3, 4, 5]))
    assert np.array_equal(np.where(m["CONF"])[0], np.array([3, 4, 5]))
    assert np.array_equal(np.where(m["ANSWER_REGION"])[0], np.array([0, 7, 8]))
    assert not m["FORMAT_VIOLATION"].any()
    assert m["meta_unclosed"] is False
    assert m["meta_drift"] is False


def test_mask_drift_clamps_at_think_close():
    # open .. content .. </think> answer — UNCLOSED + DRIFT (case a).
    ids = [1, META_OPEN, 2, 3, THINK_CLOSE, 6, 7]
    m = _m(ids)
    # META_REGION clamped to open..(think_close-1) = idx 1..3
    assert np.array_equal(np.where(m["META_REGION"])[0], np.array([1, 2, 3]))
    # clamped block: REGION but NOT CONTENT (neutral) — and IS the violation span
    assert not m["META_CONTENT"].any()
    assert np.array_equal(np.where(m["FORMAT_VIOLATION"])[0], np.array([1, 2, 3]))
    # </think> + the answer tokens REVERT to ANSWER_REGION → R_corr reaches \boxed
    assert np.array_equal(np.where(m["ANSWER_REGION"])[0], np.array([0, 4, 5, 6]))
    assert not m["CONF"].any()   # gate: no conf parse for the clamped block
    assert m["meta_unclosed"] is True
    assert m["meta_drift"] is True


def test_mask_truncation_no_think_close():
    # open, no close, NO </think>, runs to end (case b): gated, NOT a violation.
    ids = [1, META_OPEN, 2, 3, 4, 5]
    m = _m(ids)
    assert np.array_equal(np.where(m["META_REGION"])[0], np.array([1, 2, 3, 4, 5]))
    assert not m["META_CONTENT"].any()
    assert not m["CONF"].any()       # gated — no conf span for a truncated block
    assert not m["FORMAT_VIOLATION"].any()
    assert m["meta_unclosed"] is True
    assert m["meta_drift"] is False


def test_mask_drift_invariants_hold():
    ids = [1, META_OPEN, 2, 3, THINK_CLOSE, 6, 7]
    m = _m(ids)
    rm = np.ones(len(ids), dtype=bool)
    assert np.all(m["CONF"] <= m["META_CONTENT"])
    assert np.all(m["META_CONTENT"] <= m["META_REGION"])
    assert np.all(m["FORMAT_VIOLATION"] <= m["META_REGION"])
    assert not np.any(m["FORMAT_VIOLATION"] & m["META_CONTENT"])
    assert not np.any(m["META_CONTENT"] & m["ANSWER_REGION"])
    assert np.array_equal(m["ANSWER_REGION"] | m["META_REGION"], rm)


def test_mask_dup_open_after_drift_clamps_previous_span():
    # REGRESSION (review finding): a drifted span "closed" by a DUPLICATE
    # <|meta|> open must NOT put the post-</think> ANSWER tokens into
    # META_CONTENT — `open…</think>…answer…open` is the same drift class as
    # case a and gets the same clamp/violation treatment.
    ids = [1, META_OPEN, 2, 3, THINK_CLOSE, 6, 7, 6, META_OPEN, 1, 1]
    m = _m(ids)
    # First span: clamped to open..(think_close-1) = [1..3], a violation.
    assert np.array_equal(np.where(m["FORMAT_VIOLATION"])[0], np.array([1, 2, 3]))
    # </think> + the answer tokens REVERT to ANSWER_REGION (R_corr reaches \boxed).
    assert np.array_equal(np.where(m["ANSWER_REGION"])[0], np.array([0, 4, 5, 6, 7]))
    # Second span: unclosed-to-end with NO </think> after → truncation-gated.
    assert np.array_equal(
        np.where(m["META_REGION"])[0], np.array([1, 2, 3, 8, 9, 10]))
    # BOTH spans gated: no content, no conf parse over answer text.
    assert not m["META_CONTENT"].any()
    assert not m["CONF"].any()
    assert m["meta_unclosed"] is True
    assert m["meta_drift"] is True


def test_mask_dup_open_inside_think_keeps_legacy_force_close():
    # Dup open with NO intervening </think>: pre-existing force-close at i-1
    # stays byte-identical (content + span kept, no violation).
    ids = [1, META_OPEN, 2, 3, 4, 5, META_OPEN, 1, META_CLOSE, 6, 7]
    m = _m(ids)
    assert np.array_equal(np.where(m["META_CONTENT"])[0], np.array([2, 3, 4, 5, 7]))
    assert not m["FORMAT_VIOLATION"].any()
    assert m["meta_unclosed"] is False
    assert m["meta_drift"] is False


# ── v2 byte-identical lock: clamp/gate are v3-ONLY ───────────────────────────
def test_mask_clamp_unclosed_false_keeps_legacy_v2_behaviour():
    # KARPATHY lock "v2 mode byte-identical": clamp_unclosed=False reproduces
    # the pre-v3 unclosed-to-end rule VERBATIM (content + conf span kept, no
    # violation/gate flags) — this is what the v2 populator paths request.
    ids = [1, META_OPEN, 2, 3, 4, 5]
    m = build_dcpo_region_masks(ids, [True] * len(ids), _decode, clamp_unclosed=False)
    assert np.array_equal(np.where(m["META_CONTENT"])[0], np.array([2, 3, 4, 5]))
    assert np.array_equal(np.where(m["CONF"])[0], np.array([3, 4, 5]))
    assert not m["FORMAT_VIOLATION"].any()
    assert m["meta_unclosed"] is False
    assert m["meta_drift"] is False
    # Drift pattern too: legacy keeps </think>+answer INSIDE the meta block.
    ids2 = [1, META_OPEN, 2, 3, THINK_CLOSE, 6, 7]
    m2 = build_dcpo_region_masks(ids2, [True] * len(ids2), _decode, clamp_unclosed=False)
    assert np.array_equal(np.where(m2["META_REGION"])[0], np.array([1, 2, 3, 4, 5, 6]))
    assert np.array_equal(np.where(m2["META_CONTENT"])[0], np.array([2, 3, 4, 5, 6]))
    assert not m2["FORMAT_VIOLATION"].any()
    assert m2["meta_unclosed"] is False


def test_rewards_gate_unclosed_false_keeps_legacy_v2_behaviour():
    # KARPATHY lock "v2 mode byte-identical": gate_unclosed=False disables the
    # unclosed R_meta gate AND the drift penalty — R_meta follows the plain cf
    # path exactly as before the v3 format fix.
    text = "reason <|meta|> verify; confidence: 0.80 </think> The answer is \\boxed{5}"
    out = dcpo_region_rewards(
        [_c(text)], ground_truth=["5"], group_index=["g"],
        cf_completions=["The answer is \\boxed{4}"],   # c_without=0, c_with=1
        gate_unclosed=False,
    )
    assert out["R_meta"][0] == 1.0                 # UNgated (legacy)
    assert out["format_penalty"][0] == 0.0         # no penalty head for v2
    assert out["meta_unclosed"][0] == 0.0


# ── rewards: the unclosed gate + the drift penalty ───────────────────────────
def test_rmeta_gated_unclosed_even_with_positive_cf():
    # DRIFT row whose cf grading would yield +1 → gate forces R_meta 0.
    text = "reason <|meta|> verify; confidence: 0.80 </think> The answer is \\boxed{5}"
    out = dcpo_region_rewards(
        [_c(text)], ground_truth=["5"], group_index=["g"],
        cf_completions=["The answer is \\boxed{4}"],   # would grade c_without=0 → +1
    )
    assert out["R_meta"][0] == 0.0
    assert out["meta_unclosed"][0] == 1.0
    assert out["format_penalty"][0] == -1.0


def test_rmeta_gated_truncation_even_with_positive_cf():
    # TRUNCATION row (no </think> after open): gated R_meta, but NO penalty.
    text = "draft \\boxed{5} <|meta|> verify but cut mid-stre"
    out = dcpo_region_rewards(
        [_c(text)], ground_truth=["5"], group_index=["g"],
        cf_correct=[0.0],   # ungated this would be +1 (c_with=1, c_without=0)
    )
    assert out["R_meta"][0] == 0.0
    assert out["meta_unclosed"][0] == 1.0
    assert out["format_penalty"][0] == 0.0


def test_format_penalty_only_for_drift_rows():
    drift = "a <|meta|> note </think> The answer is \\boxed{5}"
    trunc = "a \\boxed{5} <|meta|> note that never ends"
    closed = _meta_text("5")
    out = dcpo_region_rewards(
        [_c(drift), _c(trunc), _c(closed)],
        ground_truth=["5"] * 3, group_index=["g"] * 3,
        cf_correct=[0.0, 0.0, 0.0],
    )
    assert out["format_penalty"] == [-1.0, 0.0, 0.0]
    assert out["meta_unclosed"] == [1.0, 1.0, 0.0]
    # closed row stays UNgated: c_with=1, c_without=0 → +1
    assert out["R_meta"][2] == 1.0
    # has_meta semantics unchanged (emission tracking): all three emit the tag.
    assert out["has_meta"] == [True, True, True]


# ── compose: the 4th head routes ONLY onto FORMAT_VIOLATION ──────────────────
def test_compose_format_head_routes_only_on_violation_mask():
    rm = torch.ones(2, 3)
    ans = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    zeros = torch.zeros(2, 3)
    fv = torch.tensor([[0.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    A, A2 = compose_dcpo_region_advantage(
        response_mask=rm,
        index=["g", "g"],
        R_corr=np.zeros(2, dtype=np.float32),
        R_meta=np.zeros(2, dtype=np.float32),
        R_cal=np.zeros(2, dtype=np.float32),
        answer_mask=ans,
        meta_content_mask=zeros,
        conf_mask=zeros,
        R_format=np.asarray([-1.0, 0.0], dtype=np.float32),
        format_violation_mask=fv,
        w_format=0.1,
    )
    assert torch.equal(A, A2)
    # Â_format = [-0.5, +0.5]; routed ONLY onto fv → row0 idx 1,2 = 0.1 * -0.5
    assert torch.allclose(A[0], torch.tensor([0.0, -0.05, -0.05]))
    # row1 has NO violation tokens → the +0.5 centered head lands NOWHERE.
    assert torch.allclose(A[1], torch.zeros(3))


def test_compose_none_format_params_byte_identical():
    # None defaults → output byte-identical to the 3-head compose (v2 compat).
    kwargs = dict(
        response_mask=torch.ones(2, 4),
        index=["g", "g"],
        R_corr=np.asarray([1.0, -1.0], dtype=np.float32),
        R_meta=np.asarray([1.0, -1.0], dtype=np.float32),
        R_cal=np.asarray([0.5, -0.5], dtype=np.float32),
        answer_mask=torch.tensor([[1.0, 0, 0, 0], [0, 0, 0, 0]]),
        meta_content_mask=torch.tensor([[0.0, 0, 1, 1], [0, 0, 0, 0]]),
        conf_mask=torch.tensor([[0.0, 0, 0, 1], [0, 0, 0, 0]]),
    )
    A_old, _ = compose_dcpo_region_advantage(**kwargs)
    A_new, _ = compose_dcpo_region_advantage(
        **kwargs, R_format=None, format_violation_mask=None, w_format=0.1
    )
    assert torch.equal(A_old, A_new)


# ── the pure-text format_penalty_reward func (REWARD_CONFIGS 5th entry) ──────
def test_format_penalty_reward_text_cases():
    import tests.test_dcpo_v3_cf  # auto-stub (verl absent locally)
    from src.training.verl_sdc import format_penalty_reward, REWARD_CONFIGS
    closed = "<|meta|> ok <|/meta|> </think> \\boxed{1}"
    drift = "<|meta|> ok </think> \\boxed{1}"
    trunc = "<|meta|> ok but cut"          # truncation: no </think> after the open
    nometa = "plain </think> \\boxed{1}"
    out = format_penalty_reward([_c(closed), _c(drift), _c(trunc), _c(nometa)])
    assert out == [0.0, -1.0, 0.0, 0.0]
    # three-way sync: 5th entry wired with routing weight 0.1.
    cfg = REWARD_CONFIGS["TRIOBJ_DCPO_V3"]
    i = cfg["keys"].index("format_penalty")
    assert cfg["weights"][i] == 0.1
    assert len(cfg["funcs"]) == len(cfg["keys"]) == len(cfg["weights"]) == 5
