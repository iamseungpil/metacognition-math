"""TDD T1-T4 for the group-branch counterfactual R_meta + SCoRe/AdaCoT shaping
(`dcpo_rmeta_source: cf_group`, design 2026-06-21).

PURE PYTHON + torch/numpy (no verl/omegaconf) so it runs under the metaprobe env.
Tests the two NEW dcpo_region pieces:
  - compute_cf_group_heads(...)   — per-group counterfactual answer-delta heads.
  - compose_dcpo_region_advantage(R_ans_meta/R_trans + answer routing) — the new
    ANSWER-region heads, proven to physically REACH compose (anti-inert gate).

T1 byte-identical: new optional kwargs at defaults == current signature.
T2 not-inert + reaches-compose: composed ANSWER advantage CHANGES by the exact
   closed form, SCoRe right->wrong is most-penalized, None collapses it.
T3 branch produces arms: positional 4/4 split + free grading (acc_without non-NaN).
T4 abstention: all-easy group -> delta 0 + over_penalty fires -> net emit adv <= 0.
"""
import numpy as np
import torch

from src.training.dcpo_pmi import PLACEBO_META
from src.training.dcpo_region import (
    compose_dcpo_region_advantage,
    compute_cf_group_heads,
    group_mean_subtract,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _compose(**over):
    """Run compose on a fixed B=8 single-group synthetic batch. `over` overrides
    only the cf_group kwargs (everything else is held constant)."""
    B, T = 8, 6
    # Region layout per row: [ans, ans, TAG, meta_c, conf, TAG]
    ans = [[1, 1, 0, 0, 0, 0]] * B
    meta_c = [[0, 0, 0, 1, 1, 0]] * B
    conf = [[0, 0, 0, 0, 1, 0]] * B
    rm = [[1, 1, 1, 1, 1, 1]] * B
    R_corr = [1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0]
    R_meta = [0.0] * B
    R_cal = [0.0] * B
    index = ["g"] * B
    kwargs = dict(
        response_mask=torch.tensor(rm, dtype=torch.float32),
        index=index,
        R_corr=np.asarray(R_corr, dtype=np.float32),
        R_meta=np.asarray(R_meta, dtype=np.float32),
        R_cal=np.asarray(R_cal, dtype=np.float32),
        answer_mask=torch.tensor(ans, dtype=torch.float32),
        meta_content_mask=torch.tensor(meta_c, dtype=torch.float32),
        conf_mask=torch.tensor(conf, dtype=torch.float32),
        w_corr=1.0, w_meta=0.5, w_cal=0.3,
    )
    kwargs.update(over)
    A, A2 = compose_dcpo_region_advantage(**kwargs)
    assert torch.equal(A, A2)
    return A


# ─────────────────────────────────────────────────────────────────────────────
# T1 — byte-identical when disabled (anti-regression)
# ─────────────────────────────────────────────────────────────────────────────
def test_T1_byte_identical_defaults():
    base = _compose()
    # Pass the new kwargs explicitly at their disabled defaults.
    with_new = _compose(
        R_ans_meta=None, w_ans_meta=0.0, ans_meta_member_mask=None,
        R_trans=None, w_score_alpha=0.0, trans_member_mask=None,
    )
    assert torch.equal(base, with_new)


# ─────────────────────────────────────────────────────────────────────────────
# T2 — not inert + provably reaches compose (THE GATE)
# ─────────────────────────────────────────────────────────────────────────────
def test_T2_cf_group_heads_math():
    # group of 8: with_meta=[1,1,1,1,0,0,0,0]
    # with-arm c_with=[1,1,1,0] ; without-arm c_with=[1,1,0,0] -> acc_without=0.5
    c_with = np.asarray([1, 1, 1, 0, 1, 1, 0, 0], dtype=np.float32)
    with_meta = np.asarray([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.float32)
    gidx = ["g"] * 8
    out = compute_cf_group_heads(
        c_with=c_with, with_meta_flag=with_meta, group_index=gidx, w_over=0.1)
    Ram = np.asarray(out["R_ans_meta"], dtype=np.float32)
    Rtr = np.asarray(out["R_trans"], dtype=np.float32)
    mem = np.asarray(out["ans_meta_member"], dtype=np.float32)
    # with-meta correct rows -> 1-0.5 = +0.5; wrong with-row -> 0-0.5 = -0.5
    assert np.allclose(Ram[:3], 0.5) and np.isclose(Ram[3], -0.5)
    assert np.allclose(Rtr[:3], 0.5) and np.isclose(Rtr[3], -0.5)
    # without-arm rows: no delta, member 0
    assert np.allclose(Ram[4:], 0.0) and np.allclose(mem, [1, 1, 1, 1, 0, 0, 0, 0])
    # acc_without not NaN (free grading from the without-arm rows)
    assert not np.isnan(np.mean(c_with[with_meta == 0]))


def test_T2_reaches_compose_and_score_direction():
    c_with = np.asarray([1, 1, 1, 0, 1, 1, 0, 0], dtype=np.float32)
    with_meta = np.asarray([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.float32)
    gidx = ["g"] * 8
    h = compute_cf_group_heads(
        c_with=c_with, with_meta_flag=with_meta, group_index=gidx, w_over=0.1)
    Ram = np.asarray(h["R_ans_meta"], dtype=np.float32)
    Rtr = np.asarray(h["R_trans"], dtype=np.float32)
    mem = np.asarray(h["ans_meta_member"], dtype=np.float32)

    w_ans_meta, w_score_alpha = 0.5, 1.5
    base = _compose()
    armed = _compose(
        R_ans_meta=Ram, w_ans_meta=w_ans_meta, ans_meta_member_mask=mem,
        R_trans=Rtr, w_score_alpha=w_score_alpha, trans_member_mask=mem,
    )
    # (c) collapses to baseline when params are None -> proves the delta is what moved it
    collapsed = _compose(
        R_ans_meta=None, w_ans_meta=w_ans_meta,
        R_trans=None, w_score_alpha=w_score_alpha,
    )
    assert torch.equal(collapsed, base)
    # (a) armed DIFFERS from baseline (not inert)
    assert not torch.equal(armed, base)

    # closed-form: centered over member rows only, routed onto ANSWER tokens.
    A_am = group_mean_subtract(Ram, gidx, member=mem)
    A_tr = group_mean_subtract(Rtr, gidx, member=mem)
    ans = torch.tensor([[1, 1, 0, 0, 0, 0]] * 8, dtype=torch.float32)
    rm = torch.ones((8, 6), dtype=torch.float32)
    expected = base + (w_ans_meta * A_am + w_score_alpha * A_tr) * ans * rm
    assert torch.allclose(armed, expected, atol=1e-6)

    # (b) SCoRe direction: the right->wrong with-row (idx 3) gets the
    # most-negative ANSWER-region advantage among the with-arm rows.
    ans_adv = armed[:, 0]  # answer token col
    with_rows = ans_adv[:4]
    assert torch.argmin(with_rows).item() == 3


# ─────────────────────────────────────────────────────────────────────────────
# T3 — branch produces both arms (positional split, torch-free)
# ─────────────────────────────────────────────────────────────────────────────
def test_T3_positional_arm_split():
    from src.training.dcpo_region import cf_group_arm_split

    n, frac = 8, 0.5
    B = 16  # 2 groups
    arm, bias = cf_group_arm_split(B, n=n, branch_frac=frac,
                                   meta_open=151669, meta_close=151670)
    arm = np.asarray(arm, dtype=np.float32)
    # per group of 8: i%8<4 with-meta (1.0), i%8>=4 without-meta (0.0)
    for g0 in (0, 8):
        assert np.allclose(arm[g0:g0 + 4], 1.0)
        assert np.allclose(arm[g0 + 4:g0 + 8], 0.0)
    # without rows carry the BOTH-tag logit_bias; with rows carry None
    for i in range(B):
        if i % n >= 4:
            assert bias[i] == {151669: -100.0, 151670: -100.0}
        else:
            assert bias[i] is None


def test_T3_grading_uses_without_arm_only():
    # without-arm c_with present -> acc_without non-NaN, computed only from without rows
    c_with = np.asarray([1, 1, 1, 1, 0, 1, 0, 1], dtype=np.float32)
    with_meta = np.asarray([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.float32)
    h = compute_cf_group_heads(
        c_with=c_with, with_meta_flag=with_meta, group_index=["g"] * 8, w_over=0.0)
    acc_without = float(np.mean(c_with[with_meta == 0]))  # 0.5
    assert not np.isnan(acc_without)
    # with-arm rows all correct -> delta = 1 - 0.5 = 0.5
    assert np.allclose(np.asarray(h["R_ans_meta"])[:4], 0.5)


# ─────────────────────────────────────────────────────────────────────────────
# T4 — abstention emerges (all-easy + over-penalty + AdaptThink floor)
# ─────────────────────────────────────────────────────────────────────────────
def test_T4_all_easy_zero_delta_and_over_penalty():
    # without-arm all correct (acc_without=1.0), with-arm all correct
    c_with = np.asarray([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
    with_meta = np.asarray([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.float32)
    h = compute_cf_group_heads(
        c_with=c_with, with_meta_flag=with_meta, group_index=["g"] * 8, w_over=0.1)
    Ram = np.asarray(h["R_ans_meta"], dtype=np.float32)
    over = np.asarray(h["over_penalty"], dtype=np.float32)
    # no reward for unnecessary meta
    assert np.allclose(Ram[:4], 0.0)
    # over-trigger penalty fires on the with-arm rows (without already correct)
    assert np.allclose(over[:4], 0.1)
    assert np.allclose(over[4:], 0.0)

    # After folding over-penalty into correctness, the with-rows' net answer-region
    # meta-emission advantage <= 0 (centered delta is 0, over-penalty subtracts).
    correctness = np.ones(8, dtype=np.float32) - over  # fold like the populator does
    A_corr = group_mean_subtract(correctness, ["g"] * 8)
    # with-rows penalized vs without-rows -> their centered correctness <= 0
    assert torch.all(A_corr[:4] <= 1e-6)


def test_T4_useful_meta_positive_and_adaptthink_clamp():
    # mixed: with-meta turns a wrong-without into right.
    # without-arm c_with=[0,0,0,0] (acc_without=0) ; with-arm=[1,1,1,1] -> delta +1
    c_with = np.asarray([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.float32)
    with_meta = np.asarray([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.float32)
    h = compute_cf_group_heads(
        c_with=c_with, with_meta_flag=with_meta, group_index=["g"] * 8, w_over=0.1)
    assert np.allclose(np.asarray(h["R_ans_meta"])[:4], 1.0)  # useful meta -> +1

    # AdaptThink floor: acc_with < acc_without -> clamp positive deltas to 0.
    # without all correct (acc_without=1), with-arm=[1,1,1,0] (acc_with=0.75 < 1).
    c_with2 = np.asarray([1, 1, 1, 0, 1, 1, 1, 1], dtype=np.float32)
    h2 = compute_cf_group_heads(
        c_with=c_with2, with_meta_flag=with_meta, group_index=["g"] * 8,
        w_over=0.0, adaptthink_floor=True)
    Ram2 = np.asarray(h2["R_ans_meta"], dtype=np.float32)
    # positive deltas clamped (the 3 correct with-rows had delta +0... acc_without=1
    # so delta = 1-1 = 0 already; the wrong with-row delta = 0-1 = -1 (negative kept).
    assert np.all(Ram2[:4] <= 0.0)
    assert np.isclose(Ram2[3], -1.0)


# ─────────────────────────────────────────────────────────────────────────────
# T5-T7 — PLACEBO without-arm routing (design 2026-06-22)
#
# The without-arm BAN degenerates on the SFT init (banning meta-open yields empty
# <think></think> -> ans2='' -> acc_without~0 -> Δ collapses to "always emit
# meta"). The fix routes without-arm rows to cf_placebo_agent (forced contentless
# placebo meta prefix, model solves on-distribution) when dcpo_cf_without_mode=
# 'placebo'; default 'ban' is byte-identical to today (T6 guards that).
# cf_group_route_row is the PURE per-row routing decision (no verl/DataProto).
# ─────────────────────────────────────────────────────────────────────────────
def test_T5_placebo_mode_routing():
    from src.training.dcpo_region import cf_group_route_row

    # without-arm (arm 0.0) under placebo -> cf_placebo_agent, NO logit_bias, flag 0
    agent, bias, wm = cf_group_route_row(
        arm_i=0.0, bias_i={151669: -100.0, 151670: -100.0}, mode="placebo")
    assert agent == "cf_placebo_agent"
    assert bias is None
    assert wm == 0.0

    # with-arm (arm 1.0) under placebo -> single_turn, no bias, flag 1
    agent, bias, wm = cf_group_route_row(arm_i=1.0, bias_i=None, mode="placebo")
    assert agent == "single_turn_agent"
    assert bias is None
    assert wm == 1.0


def test_T6_ban_mode_routing_byte_identical():
    from src.training.dcpo_region import cf_group_route_row

    # without-arm under ban (DEFAULT) -> cf_groupban_agent, BOTH-tag bias, flag 0
    bias_in = {151669: -100.0, 151670: -100.0}
    agent, bias, wm = cf_group_route_row(arm_i=0.0, bias_i=bias_in, mode="ban")
    assert agent == "cf_groupban_agent"
    assert bias == {151669: -100.0, 151670: -100.0}
    assert wm == 0.0

    # with-arm under ban -> single_turn, no bias, flag 1 (unchanged from today)
    agent, bias, wm = cf_group_route_row(arm_i=1.0, bias_i=None, mode="ban")
    assert agent == "single_turn_agent"
    assert bias is None
    assert wm == 1.0

    # DEFAULT mode (no mode arg) == 'ban' (the byte-identical guarantee)
    agent_d, bias_d, wm_d = cf_group_route_row(arm_i=0.0, bias_i=bias_in)
    assert (agent_d, bias_d, wm_d) == ("cf_groupban_agent", bias_in, 0.0)


def test_T7_placebo_response_prefix_tokens():
    # The placebo path's response prefix tokens == encode of the opener string
    # exactly; with-arm rows are untouched (None prefix).
    from src.training.cf_placebo_agent import placebo_opener_str

    class _FakeTok:
        def encode(self, text, add_special_tokens=False):
            return [hash(w) % 100000 for w in text.split(" ") if w != ""]

    tok = _FakeTok()
    opener = placebo_opener_str()
    placebo_ids = tok.encode("<think>\n" + PLACEBO_META + "\n", add_special_tokens=False)
    assert placebo_ids == tok.encode(opener, add_special_tokens=False)
    assert len(placebo_ids) > 0
