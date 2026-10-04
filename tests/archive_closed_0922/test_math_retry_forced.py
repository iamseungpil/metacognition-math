"""cd9 사전등록 수정 3 후속 — forced-redirect 탐색(0914) 회귀 시험.

M_RETRY 의 redirect_rate 붕괴(judgment 항이 gradient 를 못 받는 문제) 대응: TRAIN 행의 고정
비율을 강제로 redirect+두 번째 시도 시키되, 그 결정은 judgment 신호가 아니므로
① judgment 항(meta_val/meta_defined) undefined, ② redirect_rate(중단 규칙 지표) 계산에서
완전히 배제 — 하지만 answer_total(최종 정오)은 정상 학습돼야 한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "local"))

from src.training import math_meta as M  # noqa: E402
from src.metacot import math_meta_prompt as P  # noqa: E402


def _meta(decision=None, conf="0.4", body="The substitution step might be off."):
    d = f"decision: {decision}\n" if decision else ""
    return f"<meta>\nconfidence: {conf}\n{body}\n{d}</meta>"


def _row(first, decision, second=None, *, marker=True):
    t = f"work \\boxed{{{first}}}\n"
    if decision is not False:
        t += _meta(decision) + "\n"
    if second is not None:
        t += ("Second attempt: " if marker else "") + f"other method \\boxed{{{second}}}"
    return t


# ── math_retry_forced 프롬프트 ──────────────────────────────────────────────────
def test_math_retry_forced_variant_registered_and_importable():
    assert "math_retry_forced" in P.MATH_PROMPT_VARIANTS
    msgs = P.build_math_prompt("2+2=?", "math_retry_forced")
    assert msgs[0]["role"] == "system" and "redirect" in msgs[0]["content"]
    assert "<meta>" in msgs[0]["content"] and "Second attempt:" in msgs[0]["content"]


# ── compute_rows: forced 행의 judgment/answer 분리 ─────────────────────────────
def test_forced_row_judgment_undefined_but_answer_total_normal():
    # 틀린 첫 답 -> redirect -> 맞은 최종 답. 강제가 아니면 이건 +1 판단 크레딧을 받는다.
    text = _row("7", "redirect", "42")
    rows_free = M.compute_rows([text], ["42"], ["P"], "M_RETRY", uids=["u"])
    assert rows_free[0]["meta_defined"] == 1
    assert rows_free[0]["meta_val"] != 0.0
    assert rows_free[0]["answer_total"] == 1.0

    rows_forced = M.compute_rows([text], ["42"], ["P"], "M_RETRY", uids=["u"],
                                 forced_redirect=[1])
    assert rows_forced[0]["meta_defined"] == 0
    assert rows_forced[0]["meta_val"] == 0.0
    assert rows_forced[0]["judge"] == 0.0
    # answer_total(최종 정오)은 강제 여부와 무관하게 그대로 학습된다
    assert rows_forced[0]["answer_total"] == 1.0 == rows_free[0]["answer_total"]


def test_forced_row_answer_total_reflects_final_correctness_when_wrong():
    text = _row("7", "redirect", "9")  # 최종도 틀림
    rows = M.compute_rows([text], ["42"], ["P"], "M_RETRY", uids=["u"], forced_redirect=[1])
    assert rows[0]["meta_defined"] == 0 and rows[0]["meta_val"] == 0.0
    assert rows[0]["answer_total"] == 0.0


def test_forced_rows_also_undefined_in_rand_arm():
    text = _row("7", "redirect", "42")
    rows = M.compute_rows([text], ["42"], ["P"], "M_RETRY_RAND", uids=["u"], forced_redirect=[1])
    assert rows[0]["meta_defined"] == 0 and rows[0]["meta_val"] == 0.0
    assert rows[0]["answer_total"] == 1.0


# ── redirect_rate 는 forced 행을 완전히 배제 ────────────────────────────────────
def test_redirect_rate_excludes_forced_rows():
    # 비강제 3행: verify만(redirect_rate=0). 강제 1행: redirect(포함되면 rate가 뛴다).
    texts = [_row("42", "verify"), _row("42", "verify"), _row("42", "verify"),
             _row("7", "redirect", "42")]
    forced = [0, 0, 0, 1]
    rows = M.compute_rows(texts, ["42"] * 4, ["P"] * 4, "M_RETRY", uids=["u"] * 4,
                          forced_redirect=forced)
    rep = M.telemetry(rows, arm="M_RETRY", step=10)
    assert rep["redirect_rate"] == 0.0            # 강제 행이 섞였으면 0.25 가 됐을 것
    assert rep["n_decided"] == 3                  # 강제 행은 분모에서도 빠진다
    assert rep["forced_frac"] == 0.25
    assert rep["forced_first_acc"] == 0.0          # 강제 행의 첫 답(7)은 틀림
    assert rep["forced_rescue_rate"] == 1.0        # 강제 행: 틀렸다가 최종엔 맞음


def test_telemetry_new_keys_present_alongside_old():
    texts = [_row("7", "redirect", "42"), _row("42", "verify")]
    rows = M.compute_rows(texts, ["42"] * 2, ["P"] * 2, "M_RETRY", uids=["u"] * 2,
                          forced_redirect=[0, 0])
    rep = M.telemetry(rows, arm="M_RETRY", step=1)
    # ★B4(0914): judgment_acc_unforced 는 judgment_acc 와 바이트 동일 중복이라 제거됨(dec 가
    #   이미 비강제 행만 담는다) — judgment_acc 하나로 확인한다.
    for k in ("forced_frac", "forced_rescue_rate", "forced_first_acc", "judgment_acc"):
        assert k in rep, k
    # 기존 키도 전부 살아있다
    for k in ("first_acc", "final_acc", "redirect_rate", "redirect_rate_all_rows",
              "redirect_rate_given_wrong", "redirect_rate_given_right", "judgment_acc",
              "second_attempt_rate", "trunc_rate", "n_decided"):
        assert k in rep, k


def test_trunc_rate_excludes_forced_rows_but_all_rows_variant_includes_them():
    """★B6(0914): trunc_rate(중단 규칙용) 는 비강제 행만 분모/분자로 쓴다. 강제 행이 잘려도
    «판단 오염»이 아니므로 죽이면 안 된다 — trunc_rate_all_rows 는 참고용으로 전체를 본다."""
    texts = [_row("7", "redirect", "42"), _row("42", "verify"), _row("7", None), _row("42", "redirect", "7")]
    rows = M.compute_rows(texts, ["42"] * 4, ["P"] * 4, "M_RETRY", uids=["u"] * 4,
                          truncated=[0, 0, 1, 1], forced_redirect=[0, 0, 0, 1])
    rep = M.retry_telemetry(rows)
    assert rep["trunc_rate"] == 1 / 3          # 비강제 3행 중 1행(third row) 만 잘림
    assert rep["trunc_rate_all_rows"] == 0.5   # 전체 4행 중 2행 잘림


def test_format_tel_prints_every_retry_telemetry_key():
    """★B5(0914): retry_telemetry 가 낸 키(mixed 계열 제외 — 별도 게이트로 이미 검증됨)는
    전부 TEL 줄에 나타나야 사람이 로그만 보고 판단할 수 있다."""
    texts = [_row("7", "redirect", "42"), _row("42", "verify"), _row("7", "verify"), _row("42", "redirect", "7")]
    rows = M.compute_rows(texts, ["42"] * 4, ["P"] * 4, "M_RETRY", uids=["u"] * 4,
                          forced_redirect=[0, 0, 0, 1])
    rep = M.telemetry(rows, arm="M_RETRY", step=7)
    line = M.format_tel(rep)
    assert "meta_val_abs_mean=" in line
    for k in ("forced_frac", "forced_first_acc", "forced_rescue_rate",
              "second_attempt_rate_given_redirect", "redirect_rate_all_rows", "n_decided"):
        assert k in rep, k  # sanity: retry_telemetry actually emits it
    assert "forced_frac=" in line
    assert "forced_first_acc=" in line
    assert "forced_rescue_rate=" in line
    assert "second_attempt|redirect=" in line
    assert "redirect_all_rows=" in line
    assert "n_decided=" in line


def test_forced_redirect_default_all_zero_when_absent():
    texts = [_row("7", "redirect", "42"), _row("42", "verify")]
    rows_a = M.compute_rows(texts, ["42"] * 2, ["P"] * 2, "M_RETRY", uids=["u"] * 2)
    rows_b = M.compute_rows(texts, ["42"] * 2, ["P"] * 2, "M_RETRY", uids=["u"] * 2,
                            forced_redirect=[0, 0])
    assert [r["meta_defined"] for r in rows_a] == [r["meta_defined"] for r in rows_b]
    assert [r["meta_val"] for r in rows_a] == [r["meta_val"] for r in rows_b]


# ── nt_col default ───────────────────────────────────────────────────────────
def test_nt_col_default_when_column_absent():
    nt = {"uid": ["a", "b", "c"]}
    assert M.nt_col(nt, "forced_redirect", default=0) == [0, 0, 0]


def test_nt_col_still_fails_loud_without_default():
    import pytest
    nt = {"uid": ["a", "b"]}
    with pytest.raises(RuntimeError):
        M.nt_col(nt, "forced_redirect")


# ── build_math_parquet.py --forced_frac ─────────────────────────────────────
def _load_build_module():
    import build_math_parquet as B  # scripts/local (path inserted above)
    return B


def test_build_math_parquet_forced_frac_marks_rows():
    B = _load_build_module()
    rows = [{"problem": f"p{i}", "solution": f"\\boxed{{{i}}}", "level": "Level 5", "type": "algebra"}
            for i in range(20)]
    train, val, stats = B.split_records(rows, set(), val_n=4, seed=11, variant="math_retry")
    assert len(train) == 16 and len(val) == 4
    assert all(r["extra_info"]["forced_redirect"] == 0 for r in val)

    train_f, n_forced = B.apply_forced_redirect(train, forced_frac=0.25, forced_variant="math_retry_forced",
                                                 seed=11)
    assert n_forced == round(0.25 * len(train))
    n_marked = sum(1 for r in train_f if r["extra_info"]["forced_redirect"] == 1)
    assert n_marked == n_forced
    for r in train_f:
        if r["extra_info"]["forced_redirect"] == 1:
            assert r["extra_info"]["prompt_variant"] == "math_retry_forced"
        else:
            assert r["extra_info"]["prompt_variant"] == "math_retry"
    # val 은 절대 강제되지 않는다(함수를 호출조차 하지 않는 경로가 기본)
    assert all(r["extra_info"]["forced_redirect"] == 0 for r in val)


def test_build_math_parquet_level_filter():
    B = _load_build_module()
    rows = [{"problem": f"p{i}", "solution": f"\\boxed{{{i}}}",
             "level": "Level 5" if i % 2 == 0 else "Level 1", "type": "algebra"}
            for i in range(10)]
    train, val, stats = B.split_records(rows, set(), val_n=0, seed=11, variant="math_retry", level="Level 5")
    assert len(train) == 5
    assert stats["n_level_excluded"] == 5
