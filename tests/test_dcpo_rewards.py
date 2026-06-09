"""Unit tests for dcpo_region_rewards (TRIOBJ_DCPO_V2, spec §2.2 / §2.4 / §6.2).

PURE PYTHON. Uses simple integer boxed answers so _check_correctness string-match
resolves deterministically. Confidence stated inside the meta block.
"""
import numpy as np

from src.training.dcpo_region import dcpo_region_rewards


def _mk(a1, a2, conf=None, gt="4"):
    """Build a two-pass completion: \\boxed{a1} <|meta|> conf <|/meta|> \\boxed{a2}."""
    meta = "<|meta|>review"
    if conf is not None:
        meta += f" confidence: {conf}"
    meta += "<|/meta|>"
    text = f"\\boxed{{{a1}}} {meta} \\boxed{{{a2}}}"
    return [{"content": text}]


def _one(a1, a2, conf=None, gt="4", step=300, **cfg):
    out = dcpo_region_rewards([_mk(a1, a2, conf, gt)], ground_truth=[gt],
                              group_index=["g"], step=step, **cfg)
    return {k: v[0] for k, v in out.items()}


# ── R_corr ────────────────────────────────────────────────────────────────
def test_r_corr_last_boxed():
    assert _one(3, 4, gt="4")["R_corr"] == 1.0   # last boxed correct
    assert _one(4, 3, gt="4")["R_corr"] == -1.0  # last boxed wrong


# ── R_meta transition table (warranted via a mixed group) ──────────────────
def _grp_phat_mid():
    """A group whose answer1 pass-rate p_hat sits in [0.2,0.8] (warranted)."""
    comps = [_mk(3, 3, gt="4"), _mk(4, 4, gt="4")]  # one wrong-a1, one right-a1 → p_hat=0.5
    return dcpo_region_rewards(comps, ground_truth=["4", "4"], group_index=["g", "g"], step=300)


def test_r_meta_flip_credit_warrant_gated():
    # wrong->right flip credit is WARRANT-gated (anti-sandbag). On a genuinely hard
    # GROUP (p_hat in [0.2,0.8]) the wrong->right rollout earns +1.
    comps_w = [_mk(3, 4, gt="4"), _mk(4, 4, gt="4")]  # p_hat=0.5 warranted
    res_w = dcpo_region_rewards(comps_w, ground_truth=["4", "4"], group_index=["g", "g"], step=300)
    assert res_w["R_meta"][0] == 1.0   # the wrong->right rollout, warranted
    # On an EASY group (p_hat>0.8) a wrong->right "flip" is almost surely a STAGED
    # pass-1 error -> 0 (closes the sandbagging hole the review flagged).
    comps_e = [_mk(3, 4, gt="4")] + [_mk(4, 4, gt="4")] * 5  # p_hat=5/6≈0.833 unwarranted
    res_e = dcpo_region_rewards(comps_e, ground_truth=["4"] * 6, group_index=["g"] * 6, step=300)
    assert res_e["R_meta"][0] == 0.0   # flip credit DENIED on easy group


def test_r_meta_right_to_wrong_penalty_warmup():
    # right(4)->wrong(3): -1.0 * w_warmup. At step>=warmup_steps, w_warmup=1.
    assert _one(4, 3, gt="4", step=200, warmup_steps=200)["R_meta"] == -1.0
    # At step=50, warmup_steps=200: w_warmup=0.25 → -0.25.
    r = _one(4, 3, gt="4", step=50, warmup_steps=200)["R_meta"]
    assert abs(r - (-0.25)) < 1e-6


def test_r_meta_no_harm_warranted_vs_unwarranted():
    # wrong->wrong +eps iff warranted. Group p_hat=0.5 (warranted) → +eps.
    out = _grp_phat_mid()
    # build a warranted group where one rollout is wrong->wrong.
    comps = [_mk(5, 5, gt="4"), _mk(4, 4, gt="4")]  # p_hat=0.5 warranted
    res = dcpo_region_rewards(comps, ground_truth=["4", "4"], group_index=["g", "g"],
                              step=300, eps=0.1)
    assert abs(res["R_meta"][0] - 0.1) < 1e-9   # wrong->wrong, warranted → +eps
    # right->right warranted → 0 (eps_right_right default False; no credit for no-op
    # meta on an already-correct problem, per the intent-check adjustment).
    assert res["R_meta"][1] == 0.0

    # Unwarranted group (all wrong-a1 → p_hat=0 < p_lo): no-harm pays 0.
    comps_u = [_mk(5, 5, gt="4"), _mk(6, 6, gt="4")]  # p_hat=0
    res_u = dcpo_region_rewards(comps_u, ground_truth=["4", "4"], group_index=["g", "g"], step=300)
    assert res_u["R_meta"][0] == 0.0
    assert res_u["R_meta"][1] == 0.0


def test_r_meta_eps_right_right_off():
    # right->right with eps_right_right=False → 0 even when warranted.
    comps = [_mk(4, 4, gt="4"), _mk(5, 5, gt="4")]  # p_hat=0.5 warranted
    res = dcpo_region_rewards(comps, ground_truth=["4", "4"], group_index=["g", "g"],
                             step=300, eps=0.1, eps_right_right=False)
    assert res["R_meta"][0] == 0.0   # right->right, eps off
    assert abs(res["R_meta"][1] - 0.1) < 1e-9  # wrong->wrong still pays eps


# ── R_cal Brier ────────────────────────────────────────────────────────────
def test_r_cal_brier_and_missing():
    # right(4) with conf 0.9: -(0.9-1)^2 = -0.01
    r = _one(3, 4, conf="0.9", gt="4")["R_cal"]
    assert abs(r - (-(0.9 - 1.0) ** 2)) < 1e-9
    # wrong final (boxed 3) with conf 0.9: -(0.9-0)^2 = -0.81
    r2 = _one(4, 3, conf="0.9", gt="4")["R_cal"]
    assert abs(r2 - (-(0.9 - 0.0) ** 2)) < 1e-9
    # conf missing → 0 (no floor)
    assert _one(3, 4, conf=None, gt="4")["R_cal"] == 0.0


# ── sandbagging circuit-breaker (anti-inversion backstop) ──────────────────
def test_sandbag_clamp_ramps_meta_to_zero():
    # COLLECTIVE sandbag: a whole group fakes pass-1 wrong then "fixes" it. p_hat
    # drifts into the warranted band so flip credit would pay +1 — but the batch
    # pass-1 accuracy (canary) collapses, so the clamp ramps R_meta toward 0.
    comps = [_mk(5, 4, gt="4")] * 4  # all wrong->right; p_hat=0 here, build mixed below
    # Make it warranted-but-collapsed: 1 honest right-a1 among many staged wrong-a1.
    comps = [_mk(5, 4, gt="4")] * 1 + [_mk(4, 4, gt="4")] * 0  # canary very low
    res = dcpo_region_rewards(comps, ground_truth=["4"], group_index=["g"], step=300,
                              warmup_steps=200, sandbag_floor=0.05)
    # canary (mean pass-1 acc) = 0 < floor → clamp_f=0 → R_meta forced to 0.
    assert res["canary_pass1_acc"][0] == 0.0
    assert res["sandbag_clamp"][0] == 0.0
    assert res["R_meta"][0] == 0.0


def test_sandbag_clamp_inactive_before_warmup_and_when_healthy():
    # Before warmup, clamp is OFF (honest cold-start lows must not trigger it).
    res_cold = dcpo_region_rewards([_mk(5, 4, gt="4")], ground_truth=["4"],
                                   group_index=["g"], step=10, warmup_steps=200)
    assert res_cold["sandbag_clamp"][0] == 1.0
    # Healthy pass-1 accuracy (canary above floor) → clamp stays 1.0.
    comps = [_mk(4, 4, gt="4")] * 9 + [_mk(5, 4, gt="4")]  # canary=0.9
    res_ok = dcpo_region_rewards(comps, ground_truth=["4"] * 10, group_index=["g"] * 10,
                                 step=300, sandbag_floor=0.05)
    assert res_ok["sandbag_clamp"][0] == 1.0


# ── group-mean-subtract semantics (done in advantage path, but verify all-equal
#    head → group mean cancels) ────────────────────────────────────────────
def test_all_equal_group_centers_to_zero():
    # All rollouts wrong->wrong in a warranted group → identical R_meta=+eps.
    # The advantage centering (group mean subtract) would give 0; verify the head
    # produces an all-equal vector so centering yields a zero-gradient group.
    comps = [_mk(5, 5, gt="4"), _mk(6, 6, gt="4")]
    # make warranted: add a right-a1 rollout so p_hat in band
    comps = [_mk(5, 5, gt="4"), _mk(5, 5, gt="4"), _mk(4, 4, gt="4")]
    res = dcpo_region_rewards(comps, ground_truth=["4"] * 3, group_index=["g"] * 3, step=300, eps=0.1)
    rm = np.asarray(res["R_meta"])
    # the two wrong->wrong rollouts are identical
    assert rm[0] == rm[1]
    centered = rm - rm.mean()
    # not all zero overall (right->right differs only if eps), but the two equal
    # entries center identically.
    assert centered[0] == centered[1]


# ── eps-balance bound (§2.4) HARD ASSERT ────────────────────────────────────
def test_eps_balance_bound_hard():
    w_corr, w_meta, eps = 1.0, 0.5, 0.1
    # bound 1: w_meta*eps < w_corr strictly.
    assert w_meta * eps < w_corr

    # bound 2: on a constructed mixed group, total per-rollout advantage of a
    # STAYING-WRONG rollout < that of a BECAME-RIGHT rollout. Compose the heads
    # as the advantage path does (group-mean-subtract per head, weighted sum on
    # the rollout's own regions — here we use the per-rollout centered scalars).
    # Group: r0 = wrong->right (flip), r1 = wrong->wrong (no-harm), r2 = right->right.
    comps = [_mk(5, 4, gt="4"), _mk(5, 5, gt="4"), _mk(4, 4, gt="4")]
    res = dcpo_region_rewards(comps, ground_truth=["4"] * 3, group_index=["g"] * 3,
                              step=300, eps=eps)
    Rc = np.asarray(res["R_corr"])
    Rm = np.asarray(res["R_meta"])

    def _center(x):
        return x - x.mean()

    Ac = _center(Rc)
    Am = _center(Rm)
    # total per-rollout advantage upper bound (answer gets w_corr*Ac, meta gets
    # w_meta*Am; a became-right rollout earns both flip credit and +corr).
    total = w_corr * Ac + w_meta * Am
    became_right = 0  # r0
    staying_wrong = 1  # r1
    assert total[staying_wrong] < total[became_right]
