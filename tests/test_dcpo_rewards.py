"""Unit tests for dcpo_region_rewards — option B (format enforcement + transition).

PURE PYTHON. The B-rework:
  - answers extracted with the SAME lenient _extract_answer_fallback as correctness
    (handles "The answer is X" / last \\boxed / last number), NOT \\boxed-only;
  - PRELIMINARY answer = extracted from text BEFORE the first <|meta|>; FINAL = whole text;
  - two_pass = a preliminary answer exists before the meta block AND a final exists;
  - group difficulty p_hat = mean FINAL correctness (warranted iff 0.2<=p_hat<=0.8);
  - payoff: single-pass -> -format_penalty*warmup; two-pass no-revision -> +format_credit*warmup;
    two-pass revised wrong->right (warranted) -> +1; right->wrong -> -warmup; else -> +eps.

Tests pass sandbag_clamp=False to isolate the payoff from the canary circuit-breaker
(small synthetic groups have degenerate prelim accuracy); a dedicated test covers the clamp.
"""
import numpy as np

from src.training.dcpo_region import dcpo_region_rewards

GT = "4"


def _mk(a1, a2, conf=None, gt=GT):
    """Two-pass: \\boxed{a1} <|meta|>(conf)<|/meta|> ... \\boxed{a2}."""
    meta = "<|meta|>review"
    if conf is not None:
        meta += f" confidence: {conf}"
    meta += "<|/meta|>"
    return [{"content": f"\\boxed{{{a1}}} {meta} The final answer is \\boxed{{{a2}}}"}]


def _single(ans, conf=None):
    """Single-pass: a <|meta|> block but NO preliminary answer before it."""
    meta = "<|meta|>review"
    if conf is not None:
        meta += f" confidence: {conf}"
    meta += "<|/meta|>"
    return [{"content": f"Let me think. {meta} The answer is {ans}"}]


def run(comps, gts, step=300, sandbag_clamp=False, **cfg):
    return dcpo_region_rewards(comps, ground_truth=gts, group_index=["g"] * len(comps),
                              step=step, sandbag_clamp=sandbag_clamp, **cfg)


# ── R_corr (lenient final extraction, no \boxed required) ──────────────────
def test_r_corr_fallback():
    assert run([_single("4")], [GT])["R_corr"][0] == 1.0    # final right via fallback
    assert run([_single("7")], [GT])["R_corr"][0] == -1.0   # final wrong


# ── format enforcement ─────────────────────────────────────────────────────
def test_single_pass_format_penalty():
    # no preliminary answer before the meta -> single-pass -> -format_penalty*warmup
    r = run([_single("4")], [GT], step=200, warmup_steps=200, format_penalty=0.05)
    assert abs(r["R_meta"][0] - (-0.05)) < 1e-9


def test_two_pass_no_revision_format_credit():
    # group p_hat = 0.5 (one final right, one wrong) = warranted; rollout0 prelim==final.
    comps = [_mk(4, 4), _mk(7, 7)]
    r = run(comps, [GT, GT], step=200, warmup_steps=200, format_credit=0.05)
    assert abs(r["R_meta"][0] - 0.05) < 1e-9   # 2-pass structure, no revision


# ── transition reward ───────────────────────────────────────────────────────
def test_flip_wrong_to_right_warranted():
    # finals 4(right),7(wrong) -> p_hat=0.5 warranted; r0 prelim7(wrong)->final4(right).
    comps = [_mk(7, 4), _mk(7, 7)]
    assert run(comps, [GT, GT], step=300)["R_meta"][0] == 1.0


def test_flip_unwarranted_easy_denied():
    # all finals right -> p_hat=1.0 unwarranted -> staged flip earns 0 (anti-sandbag).
    comps = [_mk(7, 4)] + [_mk(4, 4)] * 5
    assert run(comps, [GT] * 6, step=300)["R_meta"][0] == 0.0


def test_destructive_right_to_wrong():
    # finals 7(wrong),4(right) -> p_hat=0.5; r0 prelim4(right)->final7(wrong) -> -warmup.
    comps = [_mk(4, 7), _mk(4, 4)]
    r = run(comps, [GT, GT], step=200, warmup_steps=200)
    assert r["R_meta"][0] == -1.0


def test_revised_other_warranted_eps():
    # warranted, revised but not a clean wrong->right or right->wrong: both wrong, changed.
    # finals 7(wrong),4(right) -> p_hat=0.5; r0 prelim 8(wrong)->final 7(wrong), revised.
    comps = [_mk(8, 7), _mk(9, 4)]
    r = run(comps, [GT, GT], step=300, eps=0.1)
    assert abs(r["R_meta"][0] - 0.1) < 1e-9


# ── R_cal Brier on FINAL (revives now that final is extractable) ───────────
def test_r_cal_brier_on_final():
    r = run([_mk(7, 4, conf="0.9")], [GT])   # final 4 right, conf 0.9 -> -(0.9-1)^2
    assert abs(r["R_cal"][0] - (-(0.9 - 1.0) ** 2)) < 1e-9
    r2 = run([_mk(4, 7, conf="0.9")], [GT])  # final 7 wrong -> -(0.9-0)^2
    assert abs(r2["R_cal"][0] - (-(0.9 - 0.0) ** 2)) < 1e-9


def test_conf_paren_format_now_parses():
    from src.training.rewards import _parse_confidence
    assert _parse_confidence("confidence (0.0-1.0): 0.6") == 0.6   # the prompt-echo format
    assert _parse_confidence("Confidence: 0.7") == 0.7


# ── sandbagging circuit-breaker ─────────────────────────────────────────────
def test_sandbag_clamp_zeros_meta_on_prelim_collapse():
    # all two-pass rollouts have WRONG prelims -> canary=0 < floor, after warmup -> clamp 0.
    comps = [_mk(7, 4), _mk(7, 4)]   # both prelim 7 wrong; finals 4 right -> p_hat=1.0 anyway
    r = run(comps, [GT, GT], step=300, warmup_steps=200, sandbag_clamp=True, sandbag_floor=0.05)
    assert r["sandbag_clamp"][0] == 0.0
    assert r["R_meta"][0] == 0.0


def test_clamp_inactive_before_warmup():
    comps = [_mk(7, 4), _mk(7, 4)]
    r = run(comps, [GT, GT], step=10, warmup_steps=200, sandbag_clamp=True)
    assert r["sandbag_clamp"][0] == 1.0
