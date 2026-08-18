"""Countdown 과제 정의의 단위 검사 (GPU·토크나이저·네트워크 없음).

사양이 요구한 셋을 그대로 검사한다:
  ① 증인 200개가 전부 채점 통과
  ② 오답 4종이 전부 0점
  ③ 수리된 decoy 가 유효 등식 (Countdown 규칙 하 합법 + 같은 수 + 값이 목표와 다름)
그리고 원본에서 실제로 깨져 있던 자리(OF 등식의 산술적 참·부동소수 오차)를
회귀검사로 못 박는다.
"""
import random
import re
import sys
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))  # repo root

from src.training.countdown_task import (  # noqa: E402
    DEFAULT_N_NUMS, SEARCH_BUDGET, SOLVE_SYS_NEW, SOLVE_SYS_OLD,
    build_parquet, build_prompt, eval_countdown, eval_exact, expr_numbers,
    gen_instances, grade, grade_from_gt, make_ground_truth, n_attempts,
    oracle_metas, parse_ground_truth, parse_ok, swap_op_decoy,
)

N_ROWS = 200
SEED = 20260818


@pytest.fixture(scope="module")
def rows():
    return gen_instances(N_ROWS, SEED, n_nums=DEFAULT_N_NUMS)


def boxed(expr: str) -> str:
    return f"Some search text here.\nFinal answer: \\boxed{{{expr}}}"


# ───────────────────────────────────────────── ① 증인 200개가 전부 채점 통과

def test_two_hundred_rows_generated(rows):
    assert len(rows) == N_ROWS
    assert all(len(r["nums"]) == DEFAULT_N_NUMS for r in rows)


def test_every_witness_grades_one(rows):
    bad = [r for r in rows if grade(boxed(r["witness"]), r["nums"], r["target"]) != 1]
    assert not bad, f"{len(bad)} witnesses failed grading, e.g. {bad[:3]}"


def test_every_witness_is_countdown_legal(rows):
    """생성이 합법 결합만 쓰므로 증인은 중간값 규칙도 지켜야 한다 (생성기 배선검사)."""
    for r in rows:
        v = eval_countdown(r["witness"])
        assert v == r["target"], (r["witness"], v, r["target"])


def test_every_witness_uses_each_number_once(rows):
    for r in rows:
        assert expr_numbers(r["witness"]) == sorted(r["nums"]), r


def test_ground_truth_roundtrip_grades(rows):
    for r in rows:
        gt = make_ground_truth(r["nums"], r["target"])
        nums, target = parse_ground_truth(gt)
        assert nums == r["nums"] and target == r["target"]
        assert grade_from_gt(boxed(r["witness"]), gt) == 1


# ───────────────────────────────────────────────────── ② 오답 4종 전부 0점

def test_wrong_kind_1_no_boxed(rows):
    """(1) \\boxed 없이 끝난 응답 — 0점이고 형식 준수도 0."""
    for r in rows[:50]:
        text = f"I tried {r['witness']} but ran out of attempts."
        assert grade(text, r["nums"], r["target"]) == 0
        assert parse_ok(text) == 0


def test_wrong_kind_2_right_numbers_wrong_value(rows):
    """(2) 수는 맞는데 값이 틀린 식 — 연산자 교체 오답을 그대로 쓴다."""
    checked = 0
    for r in rows:
        if not r["decoy"]:
            continue
        assert grade(boxed(r["decoy"]), r["nums"], r["target"]) == 0
        assert parse_ok(boxed(r["decoy"])) == 1     # 형식은 맞다 — 값만 틀리다
        checked += 1
    assert checked >= 150, f"only {checked} rows carried a decoy"


def test_wrong_kind_3_right_value_wrong_multiset(rows):
    """(3) 값은 맞는데 주어진 수를 안 쓴 식 — 목표수를 그대로 박은 응답."""
    for r in rows[:50]:
        assert grade(boxed(str(r["target"])), r["nums"], r["target"]) == 0
    # 수 하나를 빠뜨리고 다른 수를 두 번 쓴 경우도 0
    for r in rows[:50]:
        dup = r["witness"].replace(str(r["nums"][0]), str(r["nums"][1]), 1)
        if expr_numbers(dup) == sorted(r["nums"]):
            continue                                 # 우연히 다중집합이 같아진 경우는 건너뛴다
        assert grade(boxed(dup), r["nums"], r["target"]) == 0


def test_wrong_kind_4_unparseable(rows):
    """(4) 파싱 불가 — 문자열/깨진 식 모두 0점, 형식 준수도 0."""
    for r in rows[:50]:
        for junk in ("(3+*)", "no solution found", "\\text{cannot}", "((2+3)"):
            assert grade(boxed(junk), r["nums"], r["target"]) == 0
            assert parse_ok(boxed(junk)) == 0


def test_grade_is_exact_not_float():
    """★회귀: 원본 `grade` 는 `eval` 의 **float** 나눗셈으로 값을 낸 뒤에야
    `Fraction` 으로 쌌다(docstring 은 "부동소수 오차 0"이라고 적어 놓고서).

    아래 식은 정확 유리수로 딱 25 다. 원본은 0점(거짓 음성)을 줬고 수리판은 1점을 준다
    — 2026-08-18 실측으로 확인한 실제 발산 사례다. R_corr 이 이 함수 위에 서므로
    거짓 음성은 곧 정답을 낸 롤아웃에서 correctness 를 빼앗는 자리였다.
    """
    assert eval_exact("(((1/3)*5)*15)") == Fraction(25)
    assert grade(boxed("(((1/3)*5)*15)"), [1, 3, 5, 15], 25) == 1
    assert grade(boxed("(((1/3)*7)*9)"), [1, 3, 7, 9], 21) == 1


# ─────────────────────────────────────────────── ③ 수리된 decoy 가 유효 등식

def test_decoy_is_countdown_legal_and_off_target(rows):
    """★원본 결함의 회귀검사.

    원본 `swap_op_decoy` 는 최종값만 봐서 **중간값 음수**·**나누어떨어지지 않는
    나눗셈**을 통과시켰다(= Countdown 규칙 위에서 산술적으로 거짓).
    수리판은 전 중간값이 양의 정수여야 한다.
    """
    n_decoy = 0
    for r in rows:
        d = r["decoy"]
        if not d:
            continue
        n_decoy += 1
        v = eval_countdown(d)                        # 규칙 위반이면 None
        assert v is not None, f"decoy illegal under Countdown rules: {d}"
        assert v > 0 and v != r["target"], (d, v, r["target"])
        assert expr_numbers(d) == sorted(r["nums"]), (d, r["nums"])
        assert eval_exact(d) == Fraction(v)          # 정확 평가와도 일치
    assert n_decoy >= 150, f"only {n_decoy}/{len(rows)} rows produced a decoy"


def test_decoy_differs_from_witness_by_one_operator(rows):
    """구조는 그대로고 연산자 하나만 다르다 — PMI 대조식의 전제."""
    for r in rows:
        d = r["decoy"]
        if not d:
            continue
        w = r["witness"]
        assert len(w) == len(d), (w, d)
        diff = [k for k in range(len(w)) if w[k] != d[k]]
        assert len(diff) == 1, (w, d, diff)
        assert w[diff[0]] in "+-*/" and d[diff[0]] in "+-*/"


def test_swap_op_decoy_picks_only_the_legal_swap():
    """직접 호출: 합법 대안이 하나뿐이면 반드시 그것을 고른다.

    `(2+3)` 의 대안 셋 중 `(2-3)` 은 음수, `(2/3)` 은 나누어떨어지지 않는다.
    ★원본은 이 둘도 통과시켰다(최종값만 봤으므로 `-1`, `0.666` 은 걸렀지만
      중첩식 안쪽에서는 그대로 새어 나갔다). 수리판은 `(2*3)` 만 남긴다.
    """
    rng = random.Random(0)
    assert swap_op_decoy("(2+3)", [2, 3], 5, rng) == "(2*3)"
    assert eval_countdown("(2-3)") is None and eval_countdown("(2/3)") is None
    # 중첩식 안쪽의 위반도 걸러야 한다: (9-(2+3))=4 는 합법, (9-(2*3))=3 도 합법이지만
    # ((2-3)+9) 처럼 안쪽이 음수인 후보는 절대 나오면 안 된다.
    for _ in range(20):
        d = swap_op_decoy("((2+3)+9)", [2, 3, 9], 14, rng)
        assert d is None or eval_countdown(d) is not None


def test_swap_op_decoy_none_on_unparseable():
    rng = random.Random(0)
    assert swap_op_decoy("(2+*)", [2, 3], 5, rng) is None
    assert swap_op_decoy("", [2, 3], 5, rng) is None


# ───────────────────────────────────────────── 오라클 메타 (원본 결함 회귀)

_OF_RE = re.compile(r"The pairing to pursue is (\d+) ([+\-*/]) (\d+) = (\d+);")


def test_oracle_of_equation_is_arithmetically_true(rows):
    """★원본 결함의 회귀검사 — OF 37.5% 가 산술적으로 거짓이었다.

    수리판의 OF 는 (a) 등식이 참이고 (b) Countdown 합법 결합이며
    (c) 두 수가 모두 nums 안에 (서로 다른 자리로) 있어야 한다.
    """
    rng = random.Random(7)
    n_of = 0
    for r in rows:
        m = oracle_metas(r["witness"], r["nums"], rng)
        if m["OF"] is None:
            continue
        n_of += 1
        g = _OF_RE.search(m["OF"])
        assert g, m["OF"]
        c, op, d, v = int(g.group(1)), g.group(2), int(g.group(3)), int(g.group(4))
        assert eval_countdown(f"({c}{op}{d})") == v, m["OF"]   # 참 + 합법
        pool = list(r["nums"])
        assert c in pool, (c, r["nums"])
        pool.remove(c)
        assert d in pool, (d, r["nums"])
    assert n_of >= 150, f"only {n_of} rows produced an OF meta"


def test_oracle_o1_equation_is_true_and_o3_grades(rows):
    rng = random.Random(11)
    for r in rows[:60]:
        m = oracle_metas(r["witness"], r["nums"], rng)
        assert m["O1"] and m["O3"]
        g = _OF_RE.search(m["O1"])
        assert g, m["O1"]
        a, op, b, v = int(g.group(1)), g.group(2), int(g.group(3)), int(g.group(4))
        assert eval_countdown(f"({a}{op}{b})") == v, m["O1"]
        # O3 는 정답 전체 식을 담는다 → 그 문자열로 채점하면 1점이어야 한다(배선검사)
        assert grade(boxed(r["witness"]), r["nums"], r["target"]) == 1
        assert r["witness"] in m["O3"]


def test_oracle_meta_formats(rows):
    rng = random.Random(3)
    r = rows[0]
    new = oracle_metas(r["witness"], r["nums"], rng, fmt="new")
    old = oracle_metas(r["witness"], r["nums"], rng, fmt="old")
    assert new["O1"].startswith("<meta>") and new["O1"].rstrip().endswith("</meta>")
    assert "\n" in new["O1"]
    assert "\n" not in old["O1"] and old["O1"].startswith("confidence:")
    assert "|" in old["O1"] and old["O1"].endswith("decision: redirect")


# ──────────────────────────────────────────────────────────── 프롬프트 두 벌

@pytest.mark.parametrize("sys_prompt", [SOLVE_SYS_NEW, SOLVE_SYS_OLD])
def test_prompts_state_budget_and_boxed(sys_prompt):
    assert str(SEARCH_BUDGET) in sys_prompt
    assert "\\boxed{...}" in sys_prompt
    assert "EXACTLY ONCE" in sys_prompt
    assert "positive integer" in sys_prompt
    assert "confidence:" in sys_prompt and "decision:" in sys_prompt
    assert "verify" in sys_prompt and "redirect" in sys_prompt


def test_new_prompt_is_block_format_and_bans_arithmetic():
    assert "<meta>" in SOLVE_SYS_NEW and "</meta>" in SOLVE_SYS_NEW
    assert "Do NOT do arithmetic" in SOLVE_SYS_NEW
    assert "do not solve the" in SOLVE_SYS_NEW


def test_old_prompt_is_one_line_format():
    assert "<meta>" not in SOLVE_SYS_OLD
    assert "confidence: 0.6 | " in SOLVE_SYS_OLD


def test_build_prompt_shape(rows):
    msgs = build_prompt(rows[0], "new")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == SOLVE_SYS_NEW
    assert str(rows[0]["target"]) in msgs[1]["content"]
    assert build_prompt(rows[0], "old")[0]["content"] == SOLVE_SYS_OLD
    with pytest.raises(ValueError):
        build_prompt(rows[0], "H")


# ────────────────────────────────────────────────────── 기타 이식 함수·결정성

def test_n_attempts_counts_distinct_tries():
    text = "3+7=10\n10*8=80\n3+7=10\n80-25=55"
    assert n_attempts(text) == 3
    assert n_attempts("no arithmetic here") == 0


def test_generation_is_deterministic():
    a = gen_instances(20, 123)
    b = gen_instances(20, 123)
    assert a == b
    assert gen_instances(20, 124) != a


# ────────────────────────────────────────────────────────── parquet 빌더

def test_build_parquet_roundtrip(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    out = tmp_path / "cd_train.parquet"
    info = build_parquet(32, 5, out, variant="new")
    assert info["rows"] == 32 and info["with_decoy"] == 32

    df = pd.read_parquet(out)
    for col in ("data_source", "prompt", "ability", "reward_model", "extra_info",
                "nums", "target", "witness", "decoy"):
        assert col in df.columns, (col, list(df.columns))

    row = df.iloc[0]
    assert row["prompt"][0]["role"] == "system"
    assert row["prompt"][1]["role"] == "user"
    assert row["reward_model"]["style"] == "rule"
    nums, target = parse_ground_truth(row["reward_model"]["ground_truth"])
    assert list(row["nums"]) == nums and int(row["target"]) == target
    assert grade(boxed(row["witness"]), nums, target) == 1
    assert grade(boxed(row["decoy"]), nums, target) == 0
    assert row["extra_info"]["prompt_variant"] == "new"
    assert row["extra_info"]["search_budget"] == SEARCH_BUDGET


def test_build_parquet_old_variant_and_no_flat_cols(tmp_path):
    pytest.importorskip("pyarrow")
    pd = pytest.importorskip("pandas")
    out = tmp_path / "cd_h.parquet"
    build_parquet(8, 5, out, variant="old", include_flat_cols=False)
    df = pd.read_parquet(out)
    assert "nums" not in df.columns and "witness" not in df.columns
    assert df.iloc[0]["prompt"][0]["content"] == SOLVE_SYS_OLD
    assert df.iloc[0]["extra_info"]["nums"]  is not None


def test_parquet_train_val_disjoint(tmp_path):
    """서로 다른 seed 는 서로 다른 문제를 준다 (train/val 겹침 방지 배선검사)."""
    tr = {(tuple(r["nums"]), r["target"]) for r in gen_instances(200, 1)}
    va = {(tuple(r["nums"]), r["target"]) for r in gen_instances(200, 2)}
    assert len(tr & va) <= 2, f"unexpected overlap: {len(tr & va)}"
