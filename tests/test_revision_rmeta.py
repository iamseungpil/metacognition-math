r"""`verl_sdc._compute_revision_rmeta` — 행 선택·앵커·구간 마스크·무중심화·fail-closed(CPU).

토크나이저는 **문자 단위 가짜**다(1 문자 = 1 토큰) — char 오프셋 ↔ 토큰 인덱스가 1:1 이라
구간 마스크가 정확히 어디에 떨어지는지 눈으로 검증할 수 있다. ref forward 는 monkeypatch 로
끊는다(GPU 없음).
"""
import numpy as np
import pytest
import torch

from src.training import verl_sdc as V


class CharTok:
    """1 문자 = 1 토큰인 가짜 토크나이저."""

    def encode(self, s, add_special_tokens=False):
        return [ord(c) for c in s]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(int(i)) for i in ids)


TOK = CharTok()
LAST_TEL: dict = {}      # 마지막 호출의 계기(스킵 회계 검증용)


def knobs(**over):
    def read(name, default=None):
        return over.get(name, default)
    return read


@pytest.fixture(autouse=True)
def _clear_history():
    V._REV_FIRST_CORRECT_HISTORY.clear()
    yield
    V._REV_FIRST_CORRECT_HISTORY.clear()


def _ids(text):
    return [ord(c) for c in text]


def run(source, texts, golds, uids, read, *, trainer=None):
    T = max(len(t) for t in texts) + 4
    r, m, z, tel = V._compute_revision_rmeta(
        source=source, tokenizer=TOK, trainer=trainer, data=None,
        prompt_texts=["P: "] * len(texts), response_texts=texts,
        response_ids_list=[_ids(t) for t in texts], ground_truths=golds,
        uids=uids, response_length=T, read_knob=read, step=0)
    LAST_TEL.clear()
    LAST_TEL.update(tel)
    return r, m, z


REV_WR = r"first \boxed{3}. Wait, recheck. So \boxed{7}"     # 오답→정답 (gold 7)
REV_RW = r"first \boxed{7}. Hmm, actually. So \boxed{3}"     # 정답→오답
NOREV = r"the answer is \boxed{7} and again \boxed{7}"
MULTI = r"\boxed{3} a \boxed{5} b \boxed{7}"                  # 변화점 2개


# ── revision_cf ─────────────────────────────────────────────────────────────
def test_cf_credit_and_zone_mask():
    texts = [REV_WR, REV_RW, NOREV, NOREV, NOREV, NOREV, NOREV, NOREV]
    r, m, zm = run("revision_cf", texts, ["7"] * 8, ["g"] * 8, knobs())
    assert r[0] == pytest.approx(1.0) and m[0] == 1.0
    assert r[1] == pytest.approx(-2.0) and m[1] == 1.0
    # 수정하지 않은 행은 크레딧도 멤버십도 마스크도 전부 0
    assert r[2:].tolist() == [0.0] * 6
    assert m[2:].tolist() == [0.0] * 6
    assert zm[2:].sum() == 0.0
    # 구간 마스크는 «첫 박스 끝 → 마지막 \boxed 시작» 토큰만 덮는다
    from src.training.revision import revision_zone
    z = revision_zone(REV_WR)
    on = np.nonzero(zm[0])[0]
    assert on.min() == z["zone_start"] and on.max() == z["zone_end"] - 1
    assert TOK.decode(_ids(REV_WR)[on.min():on.max() + 1]) == \
        REV_WR[z["zone_start"]:z["zone_end"]]


def test_cf_knob_defaults_are_1_and_2():
    texts = [REV_WR, REV_RW] + [NOREV] * 6
    r, _m, _z = run("revision_cf", texts, ["7"] * 8, ["g"] * 8, knobs())
    assert r[0] == pytest.approx(1.0)
    assert r[1] == pytest.approx(-2.0)


def test_multi_change_point_row_is_skipped_and_counted(capsys):
    r, m, _z = run("revision_cf", [MULTI, REV_WR], ["7", "7"], ["g", "g"], knobs())
    assert m[0] == 0.0 and r[0] == 0.0
    assert LAST_TEL["rev_skipped_multi"] == 1.0
    assert "skip breakdown" in capsys.readouterr().out


def test_skip_counters_account_for_every_row():
    """★credit_rows=0 이 언제나 설명 가능해야 한다 — B = 크레딧 + 모든 skip_* 의 합."""
    texts = [REV_WR, REV_RW, NOREV, MULTI, "no box here at all"] + [NOREV] * 3
    golds = ["7"] * 8
    _r, m, _z = run("revision_cf", texts, golds, ["g"] * 8, knobs())
    keys = ("rev_skipped_state", "rev_skipped_multi", "rev_skip_nobox",
            "rev_skip_unrevised", "rev_skip_nogold", "rev_skip_anchor",
            "rev_skip_zone", "rev_skip_tok", "rev_skip_pmi_nan")
    assert set(keys) <= set(LAST_TEL)
    assert sum(LAST_TEL[k] for k in keys) + LAST_TEL["rev_credit_rows"] == 8
    assert LAST_TEL["rev_skip_nobox"] == 1.0          # 박스 없는 행 하나
    assert LAST_TEL["rev_skip_unrevised"] == 4.0      # NOREV × 4
    assert LAST_TEL["rev_skipped_multi"] == 1.0       # MULTI
    assert LAST_TEL["rev_credit_rows"] == 2.0         # REV_WR, REV_RW


def test_nogold_rows_are_counted_not_silent():
    _r, m, _z = run("revision_cf", [REV_WR] + [NOREV] * 7, [""] * 8, ["g"] * 8, knobs())
    assert m.sum() == 0.0
    assert LAST_TEL["rev_skip_nogold"] == 1.0


def test_state_gate_skips_split_groups(capsys):
    # 여덟 행이 전부 다른 최종 답 → SCATTER → 기본 states(ALL_SAME,DOMINANT) 밖
    texts = [REV_WR.replace("{7}", "{%d}" % (10 + i)) for i in range(8)]
    r, m, _z = run("revision_cf", texts, ["11"] * 8, ["g"] * 8, knobs())
    assert m.sum() == 0.0 and r.sum() == 0.0
    assert LAST_TEL["rev_skipped_state"] == 8.0
    assert "skip breakdown" in capsys.readouterr().out


def test_states_knob_can_admit_scatter():
    texts = [REV_WR.replace("{7}", "{%d}" % (10 + i)) for i in range(8)]
    _r, m, _z = run("revision_cf", texts, ["11"] * 8, ["g"] * 8,
                    knobs(dcpo_revpmi_states="ALL_SAME,DOMINANT,SPLIT,SCATTER"))
    assert m.sum() > 0.0


# ── revision_pmi ────────────────────────────────────────────────────────────
class _Trainer:
    class config:
        class trainer:
            nnodes = 1
            n_gpus_per_node = 1
        class actor_rollout_ref:
            class ref:
                log_prob_micro_batch_size_per_gpu = 1


def _patch_ref(monkeypatch, per_arm):
    """per_arm(k, ctx_text, ans_text) -> 팔 하나의 토큰별 logp 리스트."""
    seen = {}

    def fake_build(prompts, resps, pad_unit):
        seen["prompts"] = [TOK.decode(p) for p in prompts]
        seen["resps"] = [TOK.decode(r) for r in resps]
        return object(), len(prompts)

    def fake_ref(trainer, tensors):
        rows = [per_arm(k, seen["prompts"][k], seen["resps"][k])
                for k in range(len(seen["prompts"]))]
        L = max(len(x) for x in rows)
        out = torch.zeros(len(rows), L)
        for k, x in enumerate(rows):
            out[k, :len(x)] = torch.tensor(x, dtype=torch.float32)
        return out

    monkeypatch.setattr(V, "_build_pmi_score_batches", fake_build)
    monkeypatch.setattr(V, "_dcpo_v4_ref_logprobs", fake_ref)
    return seen


def _save_arms(k, ctx, ans):
    """A+ 를 CLOSE 에서만 선호하게 만든다(= decoy→gold 반전 = save)."""
    n = len(ans)
    plus = (k % 2 == 0)
    close = (k % 4) >= 2
    v = (1.0 if (plus == close) else -1.0)
    return [v] * n


def test_pmi_gold_x_anchor_scores_a_save(monkeypatch):
    seen = _patch_ref(monkeypatch, _save_arms)
    r, m, zm = run("revision_pmi", [REV_WR], ["7"], ["g"], knobs(), trainer=_Trainer)
    assert m[0] == 1.0 and r[0] > 0.0
    assert zm[0].sum() > 0
    # A+ = gold(7), A- = 첫 답(3)
    assert seen["resps"][:2] == [r"\boxed{7}", r"\boxed{3}"]


def test_close_context_excludes_the_last_boxed(monkeypatch):
    seen = _patch_ref(monkeypatch, _save_arms)
    run("revision_pmi", [REV_WR], ["7"], ["g"], knobs(), trainer=_Trainer)
    open_ctx, close_ctx = seen["prompts"][0], seen["prompts"][2]
    assert open_ctx.endswith(r"\boxed{3}")          # OPEN = 첫 박스까지
    assert close_ctx.startswith(open_ctx)
    assert "recheck" in close_ctx
    assert not close_ctx.endswith(r"\boxed{7}")      # 최종 답을 쓰기 **전**
    assert close_ctx.count(r"\boxed") == 1


def test_gold_x_skips_row_whose_first_answer_is_already_gold(monkeypatch):
    _patch_ref(monkeypatch, _save_arms)
    # gold = 3 = 첫 답 -> gold_x 는 신호가 없다(스킵). ★조용히 사라지면 안 된다 — 세어야 한다.
    _r, m, _z = run("revision_pmi", [REV_WR], ["3"], ["g"], knobs(), trainer=_Trainer)
    assert m.sum() == 0.0
    assert LAST_TEL["rev_skip_anchor"] == 1.0


def test_self_mx_uses_group_majority_and_skips_when_majority_equals_x(monkeypatch):
    seen = _patch_ref(monkeypatch, _save_arms)
    # 그룹 다수 최종답 = 7, 첫 답 = 3 -> A+ = 7
    texts = [REV_WR] + [NOREV] * 7
    _r, m, _z = run("revision_pmi", texts, ["99"] * 8, ["g"] * 8,
                    knobs(dcpo_revpmi_anchor="self_mx"), trainer=_Trainer)
    assert m[0] == 1.0
    assert seen["resps"][:2] == [r"\boxed{7}", r"\boxed{3}"]
    # 다수답이 첫 답과 같으면(모두 3) 스킵
    texts2 = [r"x \boxed{3} wait \boxed{5}"] + [r"a \boxed{3} b \boxed{3}"] * 7
    _r2, m2, _z2 = run("revision_pmi", texts2, ["99"] * 8, ["h"] * 8,
                       knobs(dcpo_revpmi_anchor="self_mx"), trainer=_Trainer)
    assert m2.sum() == 0.0
    assert LAST_TEL["rev_skip_anchor"] == 1.0


def test_combo_doubles_save_when_majority_is_wrong(monkeypatch):
    _patch_ref(monkeypatch, _save_arms)
    texts = [REV_WR] + [NOREV] * 7          # 다수 최종답 7 = gold -> 다수가 맞다
    r_ok, _m, _z = run("revision_pmi", texts, ["7"] * 8, ["g"] * 8,
                       knobs(dcpo_revpmi_anchor="combo"), trainer=_Trainer)
    texts_bad = [REV_WR.replace(r"\boxed{7}", r"\boxed{9}")] + \
        [r"a \boxed{9} b \boxed{9}"] * 7    # 다수 최종답 9 != gold 9? -> gold 7
    r_bad, _m2, _z2 = run("revision_pmi", texts_bad, ["7"] * 8, ["h"] * 8,
                          knobs(dcpo_revpmi_anchor="combo"), trainer=_Trainer)
    # combo 는 다수가 틀린 그룹에서 save 보너스를 두 배로 준다
    assert r_bad[0] > r_ok[0]


def test_bad_anchor_raises(monkeypatch):
    _patch_ref(monkeypatch, _save_arms)
    with pytest.raises(ValueError):
        run("revision_pmi", [REV_WR], ["7"], ["g"],
            knobs(dcpo_revpmi_anchor="nope"), trainer=_Trainer)


def test_ref_failure_fails_closed_to_zeros(monkeypatch, capsys):
    _patch_ref(monkeypatch, _save_arms)

    def boom(trainer, tensors):
        raise RuntimeError("ref exploded")
    monkeypatch.setattr(V, "_dcpo_v4_ref_logprobs", boom)
    r, m, zm = run("revision_pmi", [REV_WR], ["7"], ["g"], knobs(), trainer=_Trainer)
    assert r.sum() == 0.0 and m.sum() == 0.0 and zm.sum() == 0.0
    assert "revision_pmi ref scoring FAILED" in capsys.readouterr().out


# ── 무중심화 + 사행 가드 ────────────────────────────────────────────────────
def test_rmeta_is_not_group_centered_when_zone_mask_present():
    from src.training.dcpo_region import compose_dcpo_region_advantage
    B, T = 4, 6
    zone = torch.zeros(B, T)
    zone[0, 2:4] = 1.0
    R_meta = torch.tensor([1.0, 0.0, 0.0, 0.0])
    mem = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    kw = dict(response_mask=torch.ones(B, T), index=["g"] * B,
              R_corr=torch.zeros(B), R_cal=torch.zeros(B),
              answer_mask=torch.zeros(B, T), conf_mask=torch.zeros(B, T),
              w_corr=0.0, w_meta=1.0, w_cal=0.0, rmeta_member_mask=mem)
    A_off, _ = compose_dcpo_region_advantage(
        R_meta=R_meta, meta_content_mask=zone, rmeta_center=False, **kw)
    A_on, _ = compose_dcpo_region_advantage(
        R_meta=R_meta, meta_content_mask=zone, rmeta_center=True, **kw)
    # 무중심화: 멤버 행은 raw 1.0 을 구간 토큰에 그대로 받는다
    assert A_off[0, 2].item() == pytest.approx(1.0)
    # 중심화하면 형제 평균이 끼어들어 값이 달라진다(= 이 경로가 실제로 꺼진다)
    assert A_on[0, 2].item() != pytest.approx(1.0)
    # 어느 쪽이든 구간 밖 토큰은 0
    assert A_off[0, 0].item() == 0.0 and A_off[1].abs().sum().item() == 0.0


def test_sandbagging_guard_aborts_on_first_correct_drop():
    assert V._rev_guard_reason([0.5] * 4) is None          # 이력이 짧으면 판정 없음
    assert V._rev_guard_reason([0.5] * 20) is None
    why = V._rev_guard_reason([0.5] * 10 + [0.35] * 10)
    assert why and "first_correct_mean" in why
    assert V._rev_guard_reason([0.5] * 10 + [0.45] * 10) is None   # 5pp 하락은 통과(잡음 폭)


# ── 길이가 다른 앵커 쌍(0919) ───────────────────────────────────────────────
def test_unequal_length_anchor_pair_is_credited_not_skipped(monkeypatch):
    r"""★0919: `len(plus_ids) != len(minus_ids)` 스킵을 뺐다.

    SHIFT = PMI_close − PMI_open 은 **같은 두 문자열**을 두 문맥에서 재므로 각 후보의
    길이 편향이 차분에서 상쇄된다. gold=`\frac{1}{2}`(여러 토큰) vs 첫 답 `3`(한 토큰)
    같은 쌍이 통째로 버려지던 것이 0919 중간 분석의 최대 손실(≈8.2행/스텝)이었다.
    """
    seen = _patch_ref(monkeypatch, _save_arms)
    texts = [r"first \boxed{3}. Wait, recheck. So \boxed{\frac{1}{2}}"] + [NOREV] * 7
    r, m, zm = run("revision_pmi", texts, [r"\frac{1}{2}"] * 8, ["g"] * 8,
                   knobs(), trainer=_Trainer)
    assert m[0] == 1.0 and r[0] != 0.0 and zm[0].sum() > 0
    assert LAST_TEL["rev_skip_tok"] == 0.0
    # 두 후보의 토큰 길이가 실제로 다르다(테스트가 의도한 경로를 탔다)
    assert len(seen["resps"][0]) != len(seen["resps"][1])


def test_empty_or_identical_anchor_pair_still_skipped(monkeypatch):
    """빈 토큰화·A+ ≡ A− 는 여전히 신호가 0 이라 거른다(그 두 검사는 남겼다)."""
    _patch_ref(monkeypatch, _save_arms)
    # A+ = gold = 첫 답이면 gold_x 는 anchor 스킵이 먼저 잡는다 — 여기선 self_mx 로
    # 다수답을 첫 답과 다르게 두되 문자열이 같은 경우가 없으므로, 빈 답 쪽만 본다.
    texts = [r"x \boxed{3} wait \boxed{5}"] + [r"a \boxed{} b \boxed{}"] * 7
    _r, m, _z = run("revision_pmi", texts, [""] * 8, ["g"] * 8,
                    knobs(dcpo_revpmi_anchor="self_mx"), trainer=_Trainer)
    assert m.sum() == 0.0


def test_revision_cf_never_tokenizes_so_never_skips_on_tok():
    """★cf 팔은 ref forward 도 토큰화도 하지 않는다 — tok 스킵은 구조적으로 0 이다."""
    texts = [r"first \boxed{3}. recheck. So \boxed{\frac{1}{2}}"] + [NOREV] * 7
    _r, m, _z = run("revision_cf", texts, [r"\frac{1}{2}"] * 8, ["g"] * 8, knobs())
    assert m[0] == 1.0
    assert LAST_TEL["rev_skip_tok"] == 0.0


def test_rev_pmi_scalar_falls_back_to_own_span_sums_on_unequal_lengths():
    import numpy as _np
    g = _np.array([-1.0, -2.0, -3.0])
    d = _np.array([-0.5])
    assert V._rev_pmi_scalar(g, d, None) == pytest.approx(-6.0 + 0.5)
    # 길이가 같으면 기존 발산-마스크 경로 그대로(현행 팔 값 불변)
    dm = _np.array([True, False, True])
    same = V._rev_pmi_scalar(g, _np.array([-0.5, -2.0, -0.25]), dm)
    assert same == pytest.approx(V._pmi_position_scalar(g, _np.array([-0.5, -2.0, -0.25]), dm))


# ── 혼합 팔 revision_pmi_cf (0919) ──────────────────────────────────────────
def test_hybrid_sums_both_terms_on_a_row_credited_by_both(monkeypatch):
    _patch_ref(monkeypatch, _save_arms)
    monkeypatch.setenv("MATH_REV_W_PMI", "1.0")
    monkeypatch.setenv("MATH_REV_W_CF", "1.0")
    texts = [REV_WR] + [NOREV] * 7
    r_h, m_h, _z = run("revision_pmi_cf", texts, ["7"] * 8, ["g"] * 8, knobs(),
                       trainer=_Trainer)
    r_p, m_p, _z2 = run("revision_pmi", texts, ["7"] * 8, ["g"] * 8, knobs(),
                        trainer=_Trainer)
    r_c, m_c, _z3 = run("revision_cf", texts, ["7"] * 8, ["g"] * 8, knobs())
    assert m_h[0] == 1.0 and m_p[0] == 1.0 and m_c[0] == 1.0
    assert float(r_h[0]) == pytest.approx(float(r_p[0]) + float(r_c[0]), rel=1e-5)
    assert LAST_TEL["rev_credit_rows_any"] == 1.0


def test_hybrid_restores_derail_on_a_row_the_pmi_anchor_skips(monkeypatch):
    r"""★의도된 동작: gold ≡ 첫 답이면 PMI 앵커는 그 행을 건너뛴다(R_pmi=0). 그런데 그 행은
    «맞았다가 틀리게 고친» derail 행이다 — 결과 항이 그 벌점을 되살린다."""
    _patch_ref(monkeypatch, _save_arms)
    texts = [REV_RW] + [NOREV] * 7          # 첫 답 7(=gold) → 마지막 3
    r_p, m_p, _z = run("revision_pmi", texts, ["7"] * 8, ["g"] * 8, knobs(),
                       trainer=_Trainer)
    assert m_p.sum() == 0.0 and LAST_TEL["rev_skip_anchor"] == 1.0   # PMI 는 못 본다
    r_h, m_h, zm = run("revision_pmi_cf", texts, ["7"] * 8, ["g"] * 8, knobs(),
                       trainer=_Trainer)
    assert m_h[0] == 1.0 and float(r_h[0]) == pytest.approx(-2.0)    # −derail
    assert zm[0].sum() > 0
    assert LAST_TEL["rev_credit_rows_pmi"] == 0.0
    assert LAST_TEL["rev_credit_rows_cf"] == 1.0


def test_hybrid_with_w_cf_zero_reproduces_pure_pmi(monkeypatch):
    _patch_ref(monkeypatch, _save_arms)
    monkeypatch.setenv("MATH_REV_W_PMI", "1.0")
    monkeypatch.setenv("MATH_REV_W_CF", "0")
    texts = [REV_WR] + [NOREV] * 7
    r_h, _m, _z = run("revision_pmi_cf", texts, ["7"] * 8, ["g"] * 8, knobs(),
                      trainer=_Trainer)
    r_p, _m2, _z2 = run("revision_pmi", texts, ["7"] * 8, ["g"] * 8, knobs(),
                        trainer=_Trainer)
    assert r_h.tolist() == pytest.approx(r_p.tolist())


def test_unknown_source_still_raises():
    with pytest.raises(ValueError):
        run("revision_bogus", [REV_WR], ["7"], ["g"], knobs())


# ── 확인(confirm) 크레딧 (0920, WHY_NO_GAIN_0920 §4) ────────────────────────
CONF = r"so \boxed{3}. Wait, let me double-check. Yes, \boxed{3}"
SOLO = r"the answer is \boxed{7}"          # 박스 하나 — 확인 행이 아니다(skip_nobox)


def _stub(vals):
    """팔 순서(plus@open, minus@open, plus@close, minus@close)에 상수를 꽂는 ref 스텁."""
    def _f(k, _ctx, ans):
        return [float(vals[k % 4])] * len(ans)
    return _f


# 믿음이 gold 쪽으로 이동: PMI_open<0 → PMI_close>0 (반전)
_TO_GOLD = _stub([-1.0, +1.0, +1.0, -1.0])
# 믿음이 오답 X 에 그대로(더 굳음): PMI_open<0, PMI_close 가 더 음수 → shift<0
_STUCK = _stub([-1.0, +1.0, -1.5, +1.5])


def test_confirm_row_is_skip_unrevised_when_knob_off(monkeypatch):
    _patch_ref(monkeypatch, _TO_GOLD)
    monkeypatch.delenv("MATH_REV_CONFIRM", raising=False)
    _r, m, _z = run("revision_pmi", [CONF] + [SOLO] * 7, ["7"] * 8, ["g"] * 8,
                    knobs(), trainer=_Trainer)
    assert m.sum() == 0.0
    assert LAST_TEL["rev_skip_unrevised"] == 1.0       # 확인 행도 오늘과 똑같이 센다
    assert LAST_TEL.get("rev_confirm_rows", 0.0) == 0.0


def test_confirm_x_wrong_belief_moving_to_gold_is_rewarded(monkeypatch):
    _patch_ref(monkeypatch, _TO_GOLD)
    r, m, zm = run("revision_pmi", [CONF] + [SOLO] * 7, ["7"] * 8, ["g"] * 8,
                   knobs(dcpo_rev_confirm=1), trainer=_Trainer)
    assert m[0] == 1.0 and float(r[0]) > 0.0 and zm[0].sum() > 0
    assert LAST_TEL["rev_confirm_x_wrong"] == 1.0
    assert LAST_TEL["rev_credit_rows_confirm"] == 1.0
    assert LAST_TEL["rev_confirm_shift_mean"] > 0.0


def test_false_confirmation_belief_stuck_on_x_earns_nothing(monkeypatch):
    """★가짜 확인: 믿음이 오답 X 에 그대로인데 다시 박스에 쓴다 → r ≤ 0."""
    _patch_ref(monkeypatch, _STUCK)
    r, m, _z = run("revision_pmi", [CONF] + [SOLO] * 7, ["7"] * 8, ["g"] * 8,
                   knobs(dcpo_rev_confirm=1), trainer=_Trainer)
    assert m[0] == 1.0 and float(r[0]) <= 0.0
    assert LAST_TEL["rev_confirm_x_wrong"] == 1.0
    assert LAST_TEL["rev_confirm_shift_mean"] < 0.0


def test_mode1_skips_confirm_rows_whose_first_answer_is_right(monkeypatch):
    """★모드 1 = 가짜 확인만. 첫 답이 맞는 확인 행(참조 롤아웃의 ~90%)은 오늘과 똑같이
    skip_unrevised 로 돌아간다 — 그러지 않으면 «맞은 답 재박스»가 수정 신호를 덮는다."""
    _patch_ref(monkeypatch, _TO_GOLD)
    _r, m, _z = run("revision_pmi", [CONF] + [SOLO] * 7, ["3"] * 8, ["g"] * 8,
                    knobs(dcpo_rev_confirm=1), trainer=_Trainer)
    assert m.sum() == 0.0
    assert LAST_TEL["rev_confirm_x_right_skipped"] == 1.0
    assert LAST_TEL["rev_skip_unrevised"] == 1.0
    assert LAST_TEL["rev_confirm_rows"] == 0.0
    assert LAST_TEL["rev_confirm_mode"] == 1.0


def test_mode1_still_credits_the_false_confirmation_row(monkeypatch):
    _patch_ref(monkeypatch, _TO_GOLD)
    r, m, _z = run("revision_pmi", [CONF] + [SOLO] * 7, ["7"] * 8, ["g"] * 8,
                   knobs(dcpo_rev_confirm=1), trainer=_Trainer)
    assert m[0] == 1.0 and float(r[0]) > 0.0
    assert LAST_TEL["rev_confirm_x_right_skipped"] == 0.0


def test_mode2_confirm_x_right_gets_continuous_term_without_bonus(monkeypatch):
    """모드 2(대조): X ≡ gold 인 확인 행도 받는다 — A−는 decoy, **보너스 없이** 연속항만."""
    seen = _patch_ref(monkeypatch, _TO_GOLD)
    r, m, _z = run("revision_pmi", [CONF] + [SOLO] * 7, ["3"] * 8, ["g"] * 8,
                   knobs(dcpo_rev_confirm=2), trainer=_Trainer)
    assert m[0] == 1.0
    assert LAST_TEL["rev_confirm_x_right"] == 1.0 and LAST_TEL["rev_confirm_x_wrong"] == 0.0
    # A+ = gold(3), A− 는 X 가 아니라 decoy (그렇지 않으면 신호가 구조적으로 0)
    assert seen["resps"][0] == r"\boxed{3}" and seen["resps"][1] != r"\boxed{3}"
    # 반전 보너스(±1/±2)가 붙지 않았다 — 연속항은 clip(2.0) 을 못 넘는다
    assert abs(float(r[0])) <= 2.0 + 1e-6


def test_cf_path_gives_confirm_rows_zero(monkeypatch):
    _r, m, _z = run("revision_cf", [CONF] + [SOLO] * 7, ["7"] * 8, ["g"] * 8,
                    knobs(dcpo_rev_confirm=1))
    assert m.sum() == 0.0
    assert LAST_TEL["rev_skip_confirm_cf"] == 1.0
    assert LAST_TEL["rev_confirm_rows"] == 1.0
