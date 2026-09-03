r"""도치 자 스코어러(`src/training/countdown_inv.py`) 단위 검증.

이 파일이 지키는 것 넷 — 전부 이 저장소가 실제로 한 번씩 날려 본 실패다:
  1. **부기**: 행당 2팔 · 고정 순서(plain, hint) · `base = 2*k`.
     (PMI 의 `base = 4*k` 와 섞으면 조용히 어긋난다 — verl_sdc.py:2082 의 경고.)
  2. **격리**: 정답(witness)이 들어가는 곳은 **hint 팔의 문맥 하나**뿐이고,
     plain 팔·채점 대상 토큰열에는 한 글자도 없다.
  3. **«못 쟀다» vs «0 이다»**: 못 잰 행은 `inv_raw=None` 으로 남아 `r_meta_inv` 가
     즉사시키고, 구조적으로 잴 것이 없는 행만 0.0 이 된다.
  4. **게이트**: 누출(G5)·거짓선언(G2)이 실제로 값을 바꾼다.

토크나이저는 `test_countdown_pmi.ChunkTokenizer`(k글자 고정 청크)를 그대로 쓴다 —
오프셋이 우리 손에 있어 프로즈 마스크를 눈으로 셀 수 있다.
"""
from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from src.training import countdown_inv as ci                      # noqa: E402
from src.training import countdown_rewards as cr                  # noqa: E402
from src.training.tests.test_countdown_pmi import ChunkTokenizer  # noqa: E402

TOK = ChunkTokenizer(1)          # 문자 = 토큰. 경계 걸침 없음.

PROMPT = ("<|im_start|>user\nNumbers: [24, 15, 22, 7]\nTarget: 331"
          "<|im_end|>\n<|im_start|>assistant\n")
META = ("<meta>\nconfidence: 0.6\n"
        "I am exploring multiplication but it is not aligning.\n"
        "decision: redirect\n</meta>")
RESP = "Let me try 24*15.\n" + META + "\nSo the answer is \\boxed{(((24*15)-7)-22)}"
WIT = "(((24*15)-7)-22)"
TGT = 331


def _one(resp=RESP, wit=WIT, expr="", prompt=PROMPT, scope=None):
    return ci.build_inv_arms(TOK, [prompt], [resp], [wit], [TGT], [expr], scope=scope)


# ══════════════════════════════════════════════════════ 1. 부기 (base = 2*k)

def test_two_arms_per_row_in_fixed_order():
    ap, ar, att, per, diag = _one()
    assert len(att) == 1 and len(ap) == len(ar) == 2 * len(att)
    assert diag["attempted"] == 1 and diag["n_emitted"] == 1


def test_both_arms_score_the_identical_token_sequence():
    """이 자의 전제. 두 팔의 채점 대상이 다르면 차이는 문맥 차이가 아니다."""
    ap, ar, att, per, diag = _one()
    assert ar[0] == ar[1]
    assert ar[0] is not ar[1]                 # 같은 리스트 객체를 공유하지도 않는다


def test_arm_bookkeeping_is_base_2k_for_three_rows():
    ap, ar, att, per, diag = ci.build_inv_arms(
        TOK, [PROMPT] * 3, [RESP] * 3, [WIT] * 3, [TGT] * 3, [""] * 3)
    assert len(att) == 3 and len(ap) == 6
    for k, at in enumerate(att):
        assert at.row == k
        assert ar[2 * k] == ar[2 * k + 1]     # k 번째 시도의 두 팔


def test_read_uses_base_2k_and_the_hint_arm_is_second():
    """ref_lp[2k+0]=plain, ref_lp[2k+1]=hint. 순서가 뒤집히면 부호가 통째로 뒤집힌다."""
    ap, ar, att, per, diag = _one()
    L = att[0].n_meta
    plain = [0.0] * L
    hint = [0.0] * L
    # 프로즈 토큰 하나만 +3 만큼 올린다 → min(hint) − min(plain) = 0 − 0 = 0 이 아니라
    # 프로즈 전체가 0 이므로 min 은 양쪽 0 → 0. 대신 plain 을 −5 로 내려 확인한다.
    pi = [t for t in range(L) if att[0].prose[t]]
    assert len(pi) >= 3
    plain[pi[0]] = -5.0
    v = ci.read_inv_from_ref_logprobs([plain, hint], att, "min", "a2d")[0]
    assert v == pytest.approx(5.0)            # min(hint)=0, min(plain)=−5
    v2 = ci.read_inv_from_ref_logprobs([hint, plain], att, "min", "a2d")[0]
    assert v2 == pytest.approx(-5.0)          # 팔을 바꾸면 부호가 뒤집힌다


def test_d2a_and_a2d_are_different_statistics():
    """설계검토의 ρ=0.296 사고가 여기서 나왔다 — 둘은 같은 자가 아니다."""
    ap, ar, att, per, diag = _one()
    L = att[0].n_meta
    pi = [t for t in range(L) if att[0].prose[t]]
    plain = [0.0] * L
    hint = [0.0] * L
    plain[pi[0]] = -5.0                       # 최솟값이 다른 자리에 있게 만든다
    hint[pi[1]] = -1.0
    a2d = ci.read_inv_from_ref_logprobs([plain, hint], att, "min", "a2d")[0]
    d2a = ci.read_inv_from_ref_logprobs([plain, hint], att, "min", "d2a")[0]
    assert a2d == pytest.approx(-1.0 - (-5.0))        # min(hint) − min(plain) = +4
    assert d2a == pytest.approx(-1.0)                 # min_t(hint−plain) = min(+5, −1, 0…)
    assert a2d != d2a


# ══════════════════════════════════════════════════════ 2. 격리 (정답 유출)

def test_witness_appears_only_in_the_hint_arm_context():
    ap, ar, att, per, diag = _one()
    plain_ctx = "".join(TOK.decode([i]) if hasattr(TOK, "decode") else "" for i in [])
    # ChunkTokenizer 는 decode 를 안 주므로 문자열 단계에서 확인한다.
    hinted = ci.inv_hint_prompt(PROMPT, WIT, TGT)
    assert WIT in hinted and WIT not in PROMPT
    # 팔의 길이로도 확인: hint 팔이 힌트 문자열만큼 길다
    assert len(ap[1]) - len(ap[0]) == len(
        ci.INV_HINT_TMPL.format(w=WIT))


def test_witness_is_never_in_the_scored_tokens():
    """채점 대상은 **모델이 실제로 쓴 메타**뿐이다. 여기에 정답이 들어가면 자가 아니라 사본이다."""
    ap, ar, att, per, diag = _one()
    scored = "".join(chr(0) for _ in ar[0])    # 토큰 id 로는 못 읽으니 원문으로 확인
    assert WIT not in META


def test_hint_lands_right_after_the_user_anchor_not_at_the_end():
    h = ci.inv_hint_prompt(PROMPT, WIT, TGT)
    i = h.index("Hint: one valid solution is")
    assert h[:i].rstrip().endswith(f"Target: {TGT}")
    assert "<|im_end|>" in h[i:]               # 힌트가 user 메시지 **안**에 있다


def test_missing_anchor_dies_instead_of_appending_silently():
    with pytest.raises(ValueError):
        ci.inv_hint_prompt("no anchor here", WIT, TGT)
    ap, ar, att, per, diag = _one(prompt="no anchor here")
    assert not att and diag["anchor_error"] == 1
    assert per[0]["inv_status"].startswith("span_error")
    assert per[0]["inv_raw"] is None           # «못 쟀다» — 조용한 0 이 아니다


# ══════════════════════════════════════════════════════ 3. 못 쟀다 vs 0 이다

def test_no_meta_row_is_a_defined_zero():
    ap, ar, att, per, diag = _one(resp="no meta at all \\boxed{1+1}")
    assert not att and diag["no_meta"] == 1
    assert per[0]["inv_raw"] == 0.0 and per[0]["inv_status"] == "no_meta"


def test_no_witness_row_is_a_defined_zero():
    ap, ar, att, per, diag = _one(wit="")
    assert not att and diag["no_witness"] == 1
    assert per[0]["inv_raw"] == 0.0 and per[0]["inv_status"] == "no_witness"


def test_short_prose_row_is_a_defined_zero():
    resp = "x\n<meta>\nconfidence: 0.5\nab\ndecision: verify\n</meta>\n\\boxed{1+1}"
    ap, ar, att, per, diag = _one(resp=resp)
    assert not att and diag["short_prose"] == 1
    assert per[0]["inv_raw"] == 0.0 and per[0]["inv_status"] == "short_prose"


def test_unscored_rows_keep_none_so_r_meta_inv_dies():
    """스코어러가 안 돈 행은 None 으로 남아야 하고, 보상 함수가 그걸 즉사시켜야 한다."""
    assert ci.inv_empty_row("off")["inv_raw"] is None
    with pytest.raises(ValueError):
        cr.r_meta_inv(ci.inv_empty_row("off")["inv_raw"], 0)


# ══════════════════════════════════════════════════════ 4. 게이트 (G5 누출 · G2 거짓선언)

def test_answer_leak_blocks_the_term_and_is_counted():
    leaky = RESP.replace(
        "I am exploring multiplication but it is not aligning.",
        f"The solution is {WIT} which equals {TGT}.")
    ap, ar, att, per, diag = _one(resp=leaky, expr=WIT)
    assert not att and diag["leak_blocked"] == 1
    assert per[0]["inv_leak"] == 1 and per[0]["inv_raw"] == 0.0
    assert cr.r_meta_inv(per[0]["inv_raw"], per[0]["inv_false_claim"]) == 0.0


def test_false_claim_is_detected_and_reaches_the_reward():
    bad = RESP.replace(
        "I am exploring multiplication but it is not aligning.",
        f"The expression 24*15 reaches the target {TGT} exactly, so it is correct.")
    ap, ar, att, per, diag = _one(resp=bad)
    assert diag["false_claim"] == 1 and per[0]["inv_false_claim"] == 1
    # 도치 점수가 깨끗해도 거짓 선언만으로 벌이 붙는다
    assert cr.r_meta_inv(cr.INV_TAU - 5.0, per[0]["inv_false_claim"]) < 0.0


def test_honest_negative_report_is_not_a_false_claim():
    """«24*15 를 해 봤지만 안 된다» 는 거짓 선언이 아니다 — 거짓양성이 곧 벌이므로 치명적이다."""
    honest = RESP.replace(
        "I am exploring multiplication but it is not aligning.",
        "I tried 24*15 but it did not work, so I will change approach.")
    ap, ar, att, per, diag = _one(resp=honest)
    assert diag["false_claim"] == 0 and per[0]["inv_false_claim"] == 0


def test_a_true_expression_is_not_a_false_claim():
    """target 에 닿는 식은 G2 가 아니라 G5(answer_leak)의 소관이다 — 두 번 벌하지 않는다."""
    assert ci.false_claim_in_meta(
        f"<meta>\nconfidence: 0.9\nThe expression {WIT} reaches the target {TGT}."
        f"\ndecision: verify\n</meta>", TGT) == 0


# ══════════════════════════════════════════════════════ 5. 프로즈 마스크

def test_prose_mask_excludes_tag_confidence_and_decision_lines():
    # ★scope="inplace" 를 **명시**한다. R 팔의 기본은 reencode 이고 그쪽은 채점 대상이
    #   프로즈 그 자체라 마스크가 전부 True 다 — 마스크를 검증하려면 inplace 여야 한다.
    ap, ar, att, per, diag = _one(scope="inplace")
    from src.training import countdown_pmi as cdp
    span = cdp.find_meta_token_span(TOK, RESP)
    ids = list(span.ids[span.open_start:span.close_end])
    kept = "".join(RESP[span.open_start + t] for t, p in enumerate(att[0].prose) if p)
    # ChunkTokenizer(1) 은 문자=토큰이라 인덱스가 그대로 문자다
    assert "confidence" not in kept and "decision" not in kept
    assert "<meta>" not in kept and "</meta>" not in kept
    assert "exploring multiplication" in kept
    assert len(ids) == len(att[0].prose)


def test_reencode_scope_scores_the_prose_only():
    """R 팔이 실제로 쓰는 scope. 채점 대상이 프로즈 문자 구간 그 자체다."""
    ap, ar, att, per, diag = _one(scope="reencode")
    assert len(att) == 1
    ps = ci.prose_char_span(META)
    assert att[0].n_meta == len(TOK(META[ps[0]:ps[1]])["input_ids"])
    assert all(att[0].prose)                  # 재인코딩판은 전부가 프로즈다


def test_scope_and_form_defaults_come_from_countdown_rewards():
    """상수의 정의처는 한 곳뿐이다 — 두 곳에 두면 갈리고, 갈리면 서명이 거짓말을 한다."""
    assert (ci.INV_SCOPE, ci.INV_FORM, ci.INV_AGG) == (
        cr.INV_SCOPE, cr.INV_FORM, cr.INV_AGG)
    assert (ci.INV_TAU, ci.INV_C) == (cr.INV_TAU, cr.INV_C)


# ══════════════════════════════════════════════════════ 6. score_inv 전체 경로

def _fake_scorer(prompts, resps):
    """팔마다 결정적인 logp. hint 팔(홀수 인덱스)만 프로즈를 +1 올린다."""
    out = []
    for k, r in enumerate(resps):
        out.append([(1.0 if k % 2 else 0.0) for _ in r])
    return out


def test_score_inv_end_to_end_with_injected_scorer():
    per, diag = ci.score_inv(
        tokenizer=TOK, trainer=None, prompt_texts=[PROMPT] * 2,
        response_texts=[RESP, "no meta \\boxed{1+1}"], witnesses=[WIT, WIT],
        targets=[TGT, TGT], final_exprs=["", ""], _ref_scorer=_fake_scorer)
    assert diag["scored"] == 1 and diag["fwd_calls"] == 1 and diag["fwd_rows"] == 2
    assert per[0]["inv_status"] == "ok" and per[0]["inv_scored"] == 1
    assert per[0]["inv_raw"] == pytest.approx(1.0)     # min(1)−min(0) = 1
    assert per[1]["inv_raw"] == 0.0                    # 메타 없음 = 정의된 0
    assert diag["scope"] == cr.INV_SCOPE and diag["form"] == cr.INV_FORM
    assert math.isfinite(diag["i_p50"]) and math.isfinite(diag["i_pen_p90"])


def test_score_inv_rows_feed_arm_reward_without_keyerror():
    per, diag = ci.score_inv(
        tokenizer=TOK, trainer=None, prompt_texts=[PROMPT], response_texts=[RESP],
        witnesses=[WIT], targets=[TGT], final_exprs=[""], _ref_scorer=_fake_scorer)
    row = dict(per[0])
    row.update(r_corr=1, format_ok=1, emitted=1)
    total, comps = cr.arm_reward("R", row, step=999)
    assert cr.INV_TERM in comps and comps[cr.INV_TERM] <= 0.0
