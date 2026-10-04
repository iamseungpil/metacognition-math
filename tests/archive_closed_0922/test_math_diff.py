"""cd9 0914e — M_DIFF(난이도 판단) 회귀 시험 (CPU, 모의 생성기).

버킷 파싱 / meta_first / LOO 동의도(자기 제외) / 방향 크레딧 진리표(gold·r_corr 를 지운 행
포함) / 버킷 엔트로피 / 프롬프트 변형(시스템은 math_opt 와 동일, 사용자 턴 접미) / parquet
빌더(레벨이 섞인다) / 팔 명세와 두 대조군 / verl_sdc 배선(**비중심화** 주입과 클립·Ray env) /
텔레메트리·중단 규칙 / 런처 dry-run(RESP_LEN 4096·math_diff_eval.py) / held-out 평가(모의
생성기, 배분 표) / gate_judgment 가 acc_first 를 읽는가.
"""
from __future__ import annotations

import math
import random
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "local"))

import build_math_parquet as BP  # noqa: E402
import math_diff_eval as DE  # noqa: E402
from src.metacot.math_meta_prompt import MATH_PROMPT_VARIANTS, build_math_prompt  # noqa: E402
from src.training import math_diff as MDF  # noqa: E402
from src.training import math_meta as M  # noqa: E402

WHY = "why: the substitution step is easy to get wrong."


def _block(bucket: str | None = "hard", why: str | None = WHY) -> str:
    b = f"difficulty: {bucket}\n" if bucket is not None else ""
    w = f"{why}\n" if why else ""
    return f"<meta>\n{b}{w}</meta>"


def _text(answer: str, bucket: str | None = "hard", why: str | None = WHY, pre: str = "") -> str:
    return pre + _block(bucket, why) + f"\nwork work \\boxed{{{answer}}}\n"


def _row(answer: str, bucket: str | None = "hard", why: str | None = WHY, pre: str = "",
         gold: str = "42") -> dict:
    return MDF.parse_diff_row(_text(answer, bucket, why, pre), gold, "p0")


# ── 1. 파싱 ────────────────────────────────────────────────────────────────────
def test_parse_difficulty_three_buckets_missing_and_garbage():
    for b in MDF.BUCKETS:
        assert MDF.parse_difficulty(_block(b)) == b
    assert MDF.parse_difficulty(_block("HARD")) == "hard"        # 대소문자 무관
    assert MDF.parse_difficulty(_block(None)) is None            # 줄 자체가 없다
    assert MDF.parse_difficulty(_block("impossible")) is None    # 세 낱말이 아니다
    assert MDF.parse_difficulty("") is None
    # 둘 이상 쓰면 **첫** 것이 구속력을 갖는다(말 바꾸기 금지 — math_dis.parse_commit 과 같은 규약)
    assert MDF.parse_difficulty("difficulty: easy\n...\ndifficulty: hard") == "easy"


def test_meta_first_requires_the_block_at_the_very_start():
    assert _row("42")["meta_first"] == 1
    assert _row("42", pre="  \n")["meta_first"] == 1              # 앞 공백은 허용
    assert _row("42", pre="Let me think first. ")["meta_first"] == 0
    assert _row("42", pre="")["has_meta"] == 1


def test_parse_row_answer_is_outside_meta_and_flags():
    r = MDF.parse_diff_row("<meta>\ndifficulty: hard\nwhy: x\n\\boxed{9}\n</meta>\nans \\boxed{42}",
                           "42", "p0")
    assert r["final_answer"] == "42" and r["r_corr"] == 1        # 메타 안 박스는 답이 아니다
    assert r["boxed_in_meta"] == 1
    f = MDF.diff_row_flags(r)
    assert f["boxed_in_meta"] == 1 and f["bucket_parsed"] == 1 and f["has_why"] == 1
    assert MDF.diff_row_flags(_row("42", why=None))["has_why"] == 0
    two = MDF.parse_diff_row(_block("hard") + _block("easy") + "\\boxed{1}", "1", "p0")
    assert two["multi_block"] == 1 and MDF.diff_row_flags(two)["multi_block"] == 1


# ── 2. LOO 동의도 · 버킷 ────────────────────────────────────────────────────────
def _ans_rows(answers):
    return [{"final_answer": a} for a in answers]


def test_loo_agreement_excludes_self_and_changes_the_bucket():
    """자기를 포함하면 «다수와 일치»가 되는 행 — LOO 는 그 표를 빼야 한다."""
    group = _ans_rows(["9", "42", "42", "42", "9", "9", "9"])
    # 행 0("9")을 **포함한** 동의도는 9 가 4:3 다수 → easy 쪽으로 밀린다.
    # LOO(자기 제외)에서는 나머지 6 개가 42/9 = 3:3 동률 → 라벨 미정의.
    assert MDF.loo_agreement(0, group) is None
    # 행 1("42")을 빼면 나머지는 9 가 4, 42 가 2 → 동의도 4/6
    assert math.isclose(MDF.loo_agreement(1, group), 4 / 6)
    assert MDF.agree_bucket(4 / 6) == "medium"
    # 같은 그룹의 «자기 포함» 동의도(참고): 9 가 4/7 로 다수 — 버킷이 달라진다
    assert MDF.agree_bucket(MDF.loo_agreement(1, group)) != MDF.agree_bucket(4 / 7 + 0.3)


def test_loo_agreement_undefined_on_tie_or_too_few_siblings():
    assert MDF.loo_agreement(0, _ans_rows(["1", "2", "3"])) is None      # 1:1:1 동률
    assert MDF.loo_agreement(0, _ans_rows(["1", "2"])) is None           # 답 있는 형제 1 개
    assert MDF.loo_agreement(0, _ans_rows(["1", "", ""])) is None        # 빈 답은 세지 않는다
    assert math.isclose(MDF.loo_agreement(0, _ans_rows(["9", "5", "5"])), 1.0)


def test_agree_bucket_thresholds():
    assert MDF.agree_bucket(0.0) == "hard" and MDF.agree_bucket(MDF.AGREE_HARD_MAX) == "hard"
    assert MDF.agree_bucket(0.5) == "medium"
    assert MDF.agree_bucket(MDF.AGREE_EASY_MIN) == "easy" and MDF.agree_bucket(1.0) == "easy"
    assert MDF.agree_bucket(None) is None


# ── 3. 방향 크레딧 진리표 ───────────────────────────────────────────────────────
def _group_for(label_bucket: str):
    """라벨 버킷이 정확히 `label_bucket` 이 되는 7-형제 그룹(자기 행 idx 0 은 답이 없다 —
    LOO 대상에서 자동으로 빠지므로 라벨은 나머지가 정한다)."""
    if label_bucket == "easy":       # 동의도 1.0
        rest = ["5"] * 6
    elif label_bucket == "medium":   # 4/6
        rest = ["5", "5", "5", "5", "9", "7"]
    else:                            # hard: 3/7 이하 → 3/8
        rest = ["5", "5", "5", "9", "7", "1", "2", "3"]
    return rest


def _credit(stated: str, label: str, **kw) -> tuple[float, bool]:
    self_row = _row("5", stated, **kw)
    group = [self_row] + [MDF.parse_diff_row(f"x \\boxed{{{a}}}", "5", "p0")
                          for a in _group_for(label)]
    return MDF.diff_row_credit(self_row, group, self_idx=0)


def test_credit_truth_table_match_adjacent_opposite():
    """방향-only: 일치는 +1, 그 외(이웃이든 반대든)는 전부 −1 — «이웃=0» 세이프 하버 제거."""
    for b in MDF.BUCKETS:
        assert _credit(b, b) == (1.0, True), f"{b}: 일치는 +1"
    for a, b in (("easy", "medium"), ("medium", "easy"), ("medium", "hard"), ("hard", "medium")):
        assert _credit(a, b) == (-1.0, True), f"{a}/{b}: 이웃도 −1(세이프 하버 없음)"
    assert _credit("easy", "hard") == (-1.0, True)
    assert _credit("hard", "easy") == (-1.0, True)


def test_credit_undefined_cases_are_all_member_false():
    # 버킷 미파싱 / why 없음 / 메타 없음 / 블록이 맨 앞이 아님 / 블록 다수 / 메타 안 \boxed
    assert _credit("impossible", "easy") == (0.0, False)
    assert _credit(None, "easy") == (0.0, False)
    assert _credit("easy", "easy", why=None) == (0.0, False)
    assert _credit("easy", "easy", pre="thinking out loud first ") == (0.0, False)
    no_meta = MDF.parse_diff_row("no block at all \\boxed{5}", "5", "p0")
    group = [no_meta] + [MDF.parse_diff_row(f"x \\boxed{{{a}}}", "5", "p0") for a in ["5"] * 6]
    assert MDF.diff_row_credit(no_meta, group, self_idx=0) == (0.0, False)
    multi = MDF.parse_diff_row(_block("easy") + _block("easy") + " \\boxed{5}", "5", "p0")
    assert MDF.diff_row_credit(multi, [multi] + group[1:], self_idx=0) == (0.0, False)
    leak = MDF.parse_diff_row("<meta>\ndifficulty: easy\nwhy: x\n\\boxed{5}\n</meta>\n\\boxed{5}",
                              "5", "p0")
    assert MDF.diff_row_credit(leak, [leak] + group[1:], self_idx=0) == (0.0, False)


def test_credit_undefined_when_label_is_undefined():
    self_row = _row("5", "easy")
    tie = [self_row, MDF.parse_diff_row("x \\boxed{1}", "5", "p0"),
           MDF.parse_diff_row("x \\boxed{2}", "5", "p0")]
    assert MDF.diff_row_credit(self_row, tie, self_idx=0) == (0.0, False)   # 동률
    one = [self_row, MDF.parse_diff_row("x \\boxed{1}", "5", "p0")]
    assert MDF.diff_row_credit(self_row, one, self_idx=0) == (0.0, False)   # 형제 1 개


def test_credit_never_consults_gold_or_r_corr():
    """gold·r_corr 를 **지운** 행으로 불러도 같은 값이 나와야 한다 — 메타 크레딧이 gold 를
    읽으면 «판단»이 아니라 «정답 복사»를 보상한다."""
    self_row = _row("5", "easy")
    group = [self_row] + [MDF.parse_diff_row(f"x \\boxed{{{a}}}", "5", "p0")
                          for a in _group_for("easy")]
    ref = MDF.diff_row_credit(self_row, group, self_idx=0)
    for r in group:
        r.pop("gold", None)
        r.pop("r_corr", None)
        r.pop("cand_correct", None)
    assert MDF.diff_row_credit(group[0], group, self_idx=0) == ref == (1.0, True)


def test_credit_identity_when_self_idx_missing():
    self_row = _row("5", "easy")
    group = [self_row] + [MDF.parse_diff_row(f"x \\boxed{{{a}}}", "5", "p0")
                          for a in _group_for("easy")]
    assert MDF.diff_row_credit(self_row, group) == (1.0, True)


def test_compute_diff_rows_always_passes_explicit_self_idx(monkeypatch):
    """F3: `_compute_diff_rows` 는 학습에서 `diff_row_credit` 을 **항상** `self_idx=` 를 준
    채로 부른다 — identity(`is`) fallback 이 훈련 경로에서 쓰이지 않는지 확인한다. 값이
    같은(딕셔너리로 `==` 비교되는) 두 그룹을 만들어, self_idx 가 명시되지 않으면 첫 매치(자기
    자신이 아닌 값-동일 행)를 잘못 집는 상황을 만들고, 실제 호출은 매번 `self_idx` 인자를
    받는지 가로채 확인한다."""
    from src.training import math_diff as _mdf

    calls = []
    orig = _mdf.diff_row_credit

    def _spy(row, group_rows, self_idx=None):
        calls.append(self_idx)
        return orig(row, group_rows, self_idx=self_idx)

    monkeypatch.setattr(_mdf, "diff_row_credit", _spy)

    # 값이 완전히 같은 두 그룹(같은 텍스트) — self_idx 없이 object identity 로 자기 자신을
    # 찾으면 두 그룹의 행이 서로 «값 동일 형제»로 오인될 수 있는 배치.
    texts = [_text("42", "hard")] * 4 + [_text("42", "hard")] * 4
    golds = ["42"] * 8
    probs = ["p0"] * 8
    uids = ["g0"] * 4 + ["g1"] * 4
    rows = M.compute_rows(texts, golds, probs, "M_DIFF", uids=uids, rng=random.Random(0))
    assert len(rows) == 8
    # 각 uid 그룹 안에서 self_idx 는 그 행의 **그룹 내 위치**(0..3)여야 한다 — None 이면 안 된다.
    assert all(idx is not None for idx in calls), "self_idx 가 None 으로 호출된 행이 있다(identity fallback 의존)"
    assert calls == [0, 1, 2, 3, 0, 1, 2, 3]


def test_loo_agreement_label_excludes_exactly_self_within_group():
    """F3: 두 **값-동일** 딕셔너리 그룹(같은 답들의 나열)을 만들어, 각 행의 라벨(LOO 동의도)이
    자기 자신의 위치(self_idx)만 정확히 빼고 계산되는지 확인한다 — 값 비교(`==`)로 자기 자신을
    찾으면(identity 대신) 값이 같은 다른 그룹의 행이나 같은 그룹의 값-동일 형제를 대신 뺄 수
    있다. 한 자리(홀수답 "2")만 다수와 달라서, 그 자리를 뺐는지 아닌지가 동의도 값으로 드러난다."""
    def _mk(answer):
        return MDF.parse_diff_row(f"x \\boxed{{{answer}}}", "5", "p0")

    group_a = [_mk("1"), _mk("1"), _mk("1"), _mk("2")]     # 값이 그룹 b 와 완전히 같다
    group_b = [_mk("1"), _mk("1"), _mk("1"), _mk("2")]
    for group in (group_a, group_b):
        # majority 행(자기 답 "1") 을 빼면 나머지 3명 중 2명이 "1" 이므로 동의도는 2/3.
        for i in (0, 1, 2):
            assert math.isclose(MDF.loo_agreement(i, group), 2.0 / 3.0), \
                f"idx {i}(값 '1')의 라벨이 자기 자신 위치를 정확히 빼지 않았다"
        # minority 행(자기 답 "2")을 빼면 나머지 3명 전원이 "1" 이므로 동의도는 3/3 = 1.0.
        assert math.isclose(MDF.loo_agreement(3, group), 1.0), \
            "idx 3(값 '2')의 라벨이 자기 자신 위치를 정확히 빼지 않았다"


# ── 4. 텔레메트리 헬퍼 ─────────────────────────────────────────────────────────
def test_bucket_entropy_and_histogram():
    rows = [_row("1", b) for b in ("easy", "medium", "hard")]
    assert MDF.bucket_histogram(rows) == {"easy": 1, "medium": 1, "hard": 1}
    assert math.isclose(MDF.bucket_entropy(rows), math.log(3))
    assert math.isclose(MDF.bucket_entropy([_row("1", "hard")] * 5), 0.0)
    assert math.isnan(MDF.bucket_entropy([_row("1", None)]))


def test_label_bucket_histogram_and_entropy():
    """F2: LABEL(동의도) 쪽 히스토그램·엔트로피는 STATED 쪽과 별도로 계산된다 — `bucket`
    은 골고루 있어도 `label_bucket` 이 한쪽으로 쏠릴 수 있다(그 반대도)."""
    rows = [dict(_row("1", "hard"), label_bucket="easy"),
            dict(_row("1", "medium"), label_bucket="easy"),
            dict(_row("1", "easy"), label_bucket="hard")]
    assert MDF.label_bucket_histogram(rows) == {"easy": 2, "medium": 0, "hard": 1}
    assert math.isclose(MDF.label_bucket_entropy(rows),
                        -(2 / 3 * math.log(2 / 3) + 1 / 3 * math.log(1 / 3)))
    all_same = [dict(_row("1", "hard"), label_bucket="hard")] * 5
    assert math.isclose(MDF.label_bucket_entropy(all_same), 0.0)
    undefined = [dict(_row("1", "hard"), label_bucket=None)]
    assert math.isnan(MDF.label_bucket_entropy(undefined))
    assert MDF.label_bucket_histogram(undefined) == {"easy": 0, "medium": 0, "hard": 0}


def test_diff_telemetry_has_label_bucket_fields_and_abort_key():
    """F2: diff_telemetry 는 label_bucket_* 키를 STATED bucket_* 키 옆에 낸다, 그리고
    ABORT_RULES 는 label_bucket_max_share(라벨 붕괴)를 따로 감시한다."""
    rows = _rows_for("M_DIFF", buckets=("hard", "hard", "easy", "medium"))
    rep = M.telemetry(rows, arm="M_DIFF", step=9)
    for k in ("label_bucket_easy", "label_bucket_medium", "label_bucket_hard",
              "label_bucket_max_share", "label_bucket_entropy"):
        assert k in rep, k
    assert "lbl=e/m/h=" in M.format_tel(rep)
    assert "label_bucket_max_share" in M.ABORT_RULES
    rule = M.ABORT_RULES["label_bucket_max_share"]
    assert rule["arms"] == M._DIFF_ARMS
    assert _names(M.check_abort(_rep(label_bucket_max_share=0.99), arm="M_DIFF")) == \
           {"label_bucket_max_share"}
    assert _names(M.check_abort(_rep(label_bucket_max_share=0.5), arm="M_DIFF")) == set()


def test_stated_vs_agree_spearman_sign():
    rows = []
    for b, a in (("easy", 1.0), ("easy", 0.9), ("medium", 0.6), ("hard", 0.2), ("hard", 0.1)):
        r = _row("1", b)
        r["loo_agreement"] = a
        rows.append(r)
    assert MDF.stated_vs_agree_spearman(rows) > 0.9


# ── 5. 프롬프트 변형 ───────────────────────────────────────────────────────────
def test_math_diff_variant_shares_the_math_opt_system_prompt():
    assert MATH_PROMPT_VARIANTS["math_diff"] == MATH_PROMPT_VARIANTS["math_opt"]


def test_build_math_prompt_appends_the_ask_and_matches_the_module_constant():
    from src.metacot.math_meta_prompt import MATH_DIFF_ASK
    assert MATH_DIFF_ASK == MDF.DIFF_ASK, "프롬프트 모듈과 math_diff 의 지시문이 갈렸다"
    msgs = build_math_prompt("  What is 1+1?  ", "math_diff")
    assert msgs[0]["content"] == MATH_PROMPT_VARIANTS["math_opt"]
    assert msgs[1]["content"] == MDF.build_diff_user_turn("What is 1+1?")
    assert MDF.DIFF_MARKER in msgs[1]["content"] and "why:" in msgs[1]["content"]


# ── 6. parquet 빌더(레벨이 섞여야 한다) ────────────────────────────────────────
def test_parquet_builder_keeps_mixed_levels_for_math_diff():
    rows = [{"problem": f"p{i}", "solution": f"... \\boxed{{{i}}}",
             "level": f"Level {1 + i % 5}", "type": "algebra"} for i in range(60)]
    train, val, stats = BP.split_records(rows, set(), val_n=10, seed=0, variant="math_diff")
    assert stats["n_val"] == 10 and stats["n_train"] == 50
    levels = {r["extra_info"]["level"] for r in train}
    assert len(levels) >= 3, f"배분은 쉬운·어려운 문제가 섞여야 값을 낸다(levels={levels})"
    # ★프롬프트가 난이도 지시를 담고 있어야 한다(담지 않으면 M_G1 과 바이트 동일한 무효 레버).
    assert MDF.DIFF_MARKER in train[0]["prompt"][1]["content"]
    assert train[0]["prompt"][0]["content"] == MATH_PROMPT_VARIANTS["math_opt"]


def test_level_filter_would_break_mixture_if_used():
    """L5 전용 parquet 를 쓰면 레벨이 하나뿐 — 런처 주석이 그것을 금지하는 이유를 고정한다."""
    rows = [{"problem": f"p{i}", "solution": f"\\boxed{{{i}}}", "level": f"Level {1 + i % 5}"}
            for i in range(60)]
    train, _, _ = BP.split_records(rows, set(), val_n=5, seed=0, variant="math_diff",
                                   level="Level 5")
    assert {r["extra_info"]["level"] for r in train} == {"Level 5"}


# ── 7. 팔 명세와 두 대조군 ──────────────────────────────────────────────────────
def test_arm_specs_present_and_controls_differ_only_as_specified():
    base = M.MATH_ARM_SPECS["M_DIFF"]
    assert base["variant"] == "math_diff" and base["meta_term"] == "diff" and base["require_meta"]
    for arm, term in (("M_DIFF_RAND", "diff_shuffled"), ("M_DIFF0", "diff_zero")):
        s = M.MATH_ARM_SPECS[arm]
        assert s["variant"] == base["variant"], "대조군은 프롬프트가 같아야 한다"
        assert s["meta_term"] == term
    assert M.MATH_ARM_SPECS["M_DIFF_RAND"]["require_meta"] is True
    assert M.MATH_ARM_SPECS["M_DIFF0"]["require_meta"] is False
    assert M._DIFF_ARMS == {"M_DIFF", "M_DIFF_RAND", "M_DIFF0"}


def _rows_for(arm, buckets=("hard", "hard", "hard", "hard"), answers=("42", "42", "42", "9"),
              uids=None, rng=None):
    texts = [_text(a, b) for a, b in zip(answers, buckets)]
    return M.compute_rows(texts, ["42"] * len(texts), ["p0"] * len(texts), arm,
                          uids=uids or ["g0"] * len(texts), rng=rng)


def test_m_diff_rows_credit_and_weight(monkeypatch):
    monkeypatch.delenv("MATH_DIFF_W", raising=False)
    # 4행 중 3개가 "42" — LOO 로 자기를 빼면 동의도는 2/3 또는 3/3 이다.
    rows = _rows_for("M_DIFF")
    assert all(r["diff_defined"] == 1 for r in rows)
    assert all(r["meta_defined"] == r["diff_defined"] for r in rows)
    assert all(math.isclose(r["meta_val"], 0.5 * r["diff_credit"]) for r in rows)
    assert all(math.isclose(r["answer_total"], float(r["r_corr"])) for r in rows)
    monkeypatch.setenv("MATH_DIFF_W", "0.25")
    r2 = _rows_for("M_DIFF")[0]
    assert math.isclose(r2["meta_val"], 0.25 * r2["diff_credit"])


def test_m_diff0_has_zero_meta_value_but_identical_rows(monkeypatch):
    monkeypatch.setenv("MATH_DIFF_W", "0.5")
    base, zero = _rows_for("M_DIFF"), _rows_for("M_DIFF0")
    assert all(r["meta_val"] == 0.0 for r in zero)
    for a, b in zip(base, zero):
        assert a["diff_credit"] == b["diff_credit"] and a["diff_defined"] == b["diff_defined"]
        assert a["answer_total"] == b["answer_total"]


def test_m_diff_rand_permutes_labels_per_uid_group_not_per_row():
    r"""라벨은 **그룹 단위**로 섞인다(M_RAND 규약): 같은 uid 그룹의 행들은 같은 도너 그룹의
    동의도로 채점되고, 라벨 분포 자체는 보존된다."""
    n_groups, gs = 6, 4
    texts, golds, probs, uids = [], [], [], []
    for gi in range(n_groups):
        # 그룹마다 답 동의도를 다르게 — 그룹 g0..g2 는 만장일치(easy), g3..g5 는 갈린다.
        answers = (["42"] * gs) if gi < 3 else ["42", "9", "7", "1"]
        for a in answers:
            texts.append(_text(a, "easy"))
            golds.append("42")
            probs.append(f"p{gi}")
            uids.append(f"g{gi}")
    kw = dict(uids=uids)
    base = M.compute_rows(texts, golds, probs, "M_DIFF", **kw)
    rand = M.compute_rows(texts, golds, probs, "M_DIFF_RAND", rng=random.Random(0), **kw)
    for gi in range(n_groups):
        blk = rand[gi * gs:(gi + 1) * gs]
        assert len({r["diff_credit"] for r in blk}) == 1, "같은 uid 그룹은 같은 라벨을 받아야 한다"
    assert [r["diff_credit"] for r in rand] != [r["diff_credit"] for r in base], \
        "도너 순열이 아무것도 바꾸지 않으면 무효 대조군이다"
    # 라벨 분포(= 크레딧 값들의 다중집합)는 그룹 단위로 보존된다
    def _by_group(rows):
        return sorted(rows[gi * gs]["diff_credit"] for gi in range(n_groups))
    assert _by_group(rand) == _by_group(base)


def test_metrics_only_fields_never_enter_the_credit():
    """`loo_pass_rate`(gold 파생)는 행에 실리지만 크레딧은 그것을 읽지 않는다."""
    rows = _rows_for("M_DIFF")
    assert all("loo_pass_rate" in r for r in rows)
    src = (REPO / "src/training/math_diff.py").read_text()
    credit_fn = src.split("def diff_row_credit")[1].split("\ndef ")[0]
    body = credit_fn.split('"""')[2]        # docstring(★설명에는 두 이름이 나온다) 뒤의 코드만
    assert "r_corr" not in body and "gold" not in body


# ── 8. 텔레메트리 · 중단 규칙 ──────────────────────────────────────────────────
def test_diff_telemetry_keys_and_values():
    rows = _rows_for("M_DIFF", buckets=("hard", "hard", "easy", "medium"))
    rep = M.telemetry(rows, arm="M_DIFF", step=9)
    for k in ("diff_rows", "diff_emit_rate", "meta_first_rate", "bucket_parsed", "has_why_rate",
              "bucket_max_share", "bucket_entropy", "diff_defined_rate", "diff_credit_mean",
              "stated_vs_agree_spearman", "stated_vs_truepass_spearman", "final_acc",
              "bucket_easy", "bucket_medium", "bucket_hard"):
        assert k in rep, k
    assert rep["diff_emit_rate"] == 1.0 and rep["bucket_parsed"] == 1.0
    assert rep["meta_first_rate"] == 1.0 and rep["has_why_rate"] == 1.0
    assert math.isclose(rep["bucket_hard"], 0.5) and math.isclose(rep["bucket_max_share"], 0.5)
    assert "first_acc" not in rep, "first_acc 키는 format_tel 을 재시도 팔 경로로 보낸다"
    assert "diff_rows=" in M.format_tel(rep)


def test_diff_telemetry_nan_when_nothing_parsed():
    rows = M.compute_rows(["plain answer \\boxed{42}"] * 3, ["42"] * 3, ["p0"] * 3, "M_DIFF",
                          uids=["g0"] * 3)
    rep = M.telemetry(rows, arm="M_DIFF", step=9)
    assert rep["diff_emit_rate"] == 0.0 and rep["bucket_parsed"] == 0.0
    assert math.isnan(rep["bucket_entropy"]) and math.isnan(rep["bucket_max_share"])
    assert math.isnan(rep["meta_first_rate"])       # 분모 0 → «못 쟀다»
    assert rep["diff_defined_rate"] == 0.0


def _rep(**kw):
    base = {"step": 20, "arm": "M_DIFF", "emit_rate": 1.0, "boxed_in_meta": 0.0,
            "multi_block_rate": 0.0, "boilerplate_rate": 0.1, "n_emitted": 100, "acc": 0.5,
            "diff_emit_rate": 1.0, "bucket_parsed": 1.0, "bucket_max_share": 0.5,
            "diff_defined_rate": 0.9}
    base.update(kw)
    return base


def _names(hits):
    return {h["metric"] for h in hits if h["status"] == "abort"}


def test_abort_rules_for_diff_arm():
    assert _names(M.check_abort(_rep(), arm="M_DIFF")) == set()
    assert "diff_emit_rate" in _names(M.check_abort(_rep(diff_emit_rate=0.5), arm="M_DIFF"))
    assert "bucket_parsed" in _names(M.check_abort(_rep(bucket_parsed=0.5), arm="M_DIFF"))
    assert "bucket_max_share" in _names(M.check_abort(_rep(bucket_max_share=0.95), arm="M_DIFF"))
    assert "diff_defined_rate" in _names(M.check_abort(_rep(diff_defined_rate=0.2), arm="M_DIFF"))
    assert "boxed_in_meta" in _names(M.check_abort(_rep(boxed_in_meta=0.05), arm="M_DIFF"))
    assert "multi_block_rate" in _names(M.check_abort(_rep(multi_block_rate=0.2), arm="M_DIFF"))


def test_abort_warmups_for_diff_arm():
    # 형식 두 규칙은 step≤3, 붕괴·정의율은 step≤5 까지 봐준다
    assert _names(M.check_abort(_rep(step=3, diff_emit_rate=0.1, bucket_parsed=0.1,
                                     bucket_max_share=1.0, diff_defined_rate=0.0),
                                arm="M_DIFF")) == set()
    assert _names(M.check_abort(_rep(step=5, bucket_max_share=1.0, diff_defined_rate=0.0),
                                arm="M_DIFF")) == set()
    assert "bucket_max_share" in _names(M.check_abort(_rep(step=6, bucket_max_share=1.0),
                                                      arm="M_DIFF"))


def test_diff_rules_do_not_touch_other_arms_and_vice_versa():
    other = {"step": 10, "arm": "M_G1", "emit_rate": 0.9, "boxed_in_meta": 0.0,
             "multi_block_rate": 0.0, "boilerplate_rate": 0.1, "n_emitted": 100, "acc": 0.7}
    assert not any(h["metric"] in ("diff_emit_rate", "bucket_parsed", "bucket_max_share",
                                   "diff_defined_rate") for h in M.check_abort(other, arm="M_G1"))
    hits = M.check_abort(_rep(), arm="M_DIFF")
    assert not any(h["metric"] in ("redirect_rate", "leak_rate", "dis_emit_rate",
                                   "commit_parsed", "agree_line_rate") for h in hits)
    # ★M_DIFF0 은 require_meta=False — 발화 0 이어도 메타 전용 규칙이 죽이지 않는다
    assert _names(M.check_abort(_rep(arm="M_DIFF0", diff_emit_rate=0.0, bucket_parsed=0.0,
                                     diff_defined_rate=0.0, emit_rate=0.0),
                                arm="M_DIFF0")) == set()


# ── 9. verl_sdc 배선(비중심화·클립) ─────────────────────────────────────────────
def test_stash_splits_diff_out_of_the_centered_path():
    src = (REPO / "src/training/verl_sdc.py").read_text()
    assert "_mm._DIFF_TERMS" in src
    assert '"diff": diff_vals' in src and '"diff_spans": diff_spans' in src
    assert "meta_vals = [0.0] * bs" in src, "M_DIFF 의 항이 중심화 경로에도 실리면 두 번 얹힌다"


def test_diff_advantage_is_non_centered_and_clipped():
    src = (REPO / "src/training/verl_sdc.py").read_text()
    fn = src.split("def _math_add_diff_meta_advantage")[1].split("\ndef ")[0]
    assert "group_mean_subtract" not in fn, "이 경로는 **중심화하지 않는다**(그러면 신호가 0 이 된다)"
    assert "cap" in fn and "abs(c) > cap" in fn, "보너스 절댓값을 답 어드밴티지 평균으로 잘라야 한다"
    assert "data = _math_add_diff_meta_advantage(data)" in src, "어드밴티지 훅에 안 걸렸다"


def test_ray_env_forwards_math_diff_w():
    src = (REPO / "src/training/verl_sdc.py").read_text()
    assert '"MATH_DIFF_W"' in src, "MATH_DIFF_W 가 Ray runtime_env 목록에 없다(워커가 못 읽는다)"


def test_non_centered_injection_keeps_a_shared_label_signal():
    """★핵심 계약: 한 그룹의 형제가 **전부 같은 크레딧**일 때 — 중심화하면 0, 이 경로는 credit×W.
    (여기서는 어드밴티지 텐서 없이 그 대수만 고정한다 — 위 두 소스 테스트가 배선을 지킨다.)"""
    from src.training.dcpo_region import group_mean_subtract
    rows = _rows_for("M_DIFF", buckets=("easy",) * 4, answers=("42",) * 4)
    vals = [r["meta_val"] for r in rows]
    assert all(math.isclose(v, 0.5) for v in vals), "만장일치 그룹 = easy 라벨 일치 → +1·W"
    centered = group_mean_subtract(vals, ["g0"] * 4,
                                   member=[r["meta_defined"] for r in rows]).reshape(-1)
    assert all(abs(float(c)) < 1e-9 for c in centered), "중심화하면 신호가 0 — 그래서 안 쓴다"


# ── 10. 런처 dry-run ────────────────────────────────────────────────────────────
def test_run_math_arm_diff_defaults():
    r = subprocess.run(["bash", "scripts/local/run_math_arm.sh", "M_DIFF", "2", "30", "--dry-run"],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "ARM=M_DIFF" in out and "VARIANT=math_diff" in out
    assert "RESP_LEN=4096" in out
    assert "EVAL_SCRIPT=math_diff_eval.py" in out and "math500_diff_8k" in out
    assert "MATH_DIFF_W=0.5" in out
    assert "math_train_math_diff.parquet" in out and "math_val_math_diff.parquet" in out


def test_run_math_arm_diff_controls_share_the_launch_contract():
    for arm in ("M_DIFF_RAND", "M_DIFF0"):
        r = subprocess.run(["bash", "scripts/local/run_math_arm.sh", arm, "2", "30", "--dry-run"],
                           cwd=REPO, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert "RESP_LEN=4096" in r.stdout and "EVAL_SCRIPT=math_diff_eval.py" in r.stdout
        assert "VARIANT=math_diff" in r.stdout


# ── 11. held-out 평가(모의 생성기) ──────────────────────────────────────────────
def _mock_gen(per_problem):
    def gen(prompts):
        assert len(prompts) == len(per_problem)
        return [[(t, 0, max(1, len(t) // 4)) for t in texts] for texts in per_problem]
    return gen


def _samples(bucket, answers):
    return [_text(a, bucket) for a in answers]


def test_eval_summary_and_allocation_table_on_mocked_generations():
    problems = [{"problem": f"p{i}", "gold": "42"} for i in range(4)]
    # p0,p1: easy 라고 말하고 실제로 첫 표본부터 맞힌다.
    # p2,p3: hard 라고 말한다 — 첫 표본은 틀리지만 4 표를 모으면 다수결이 맞는다.
    per = [_samples("easy", ["42"] * 8), _samples("easy", ["42"] * 8),
           _samples("hard", ["9", "42", "42", "42", "42", "42", "42", "42"]),
           _samples("hard", ["7", "42", "42", "42", "42", "42", "42", "42"])]
    groups = DE.sample_rows(problems, _mock_gen(per))
    assert [len(g) for g in groups] == [8] * 4
    rows, tel = DE.evaluate(problems, groups, baseline_acc=0.5)
    assert len(rows) == 32 and tel["n_groups"] == 4
    assert math.isclose(tel["acc_first"], 0.5)          # 첫 표본: p0,p1 만 맞다
    assert math.isclose(tel["stated_hard_frac"], 0.5)
    assert math.isclose(tel["pass_easy"], 1.0) and tel["pass_hard"] < 1.0
    assert math.isclose(tel["acc_tax_vs_baseline"], 0.0)
    assert tel["diff_emit_rate"] == 1.0 and tel["bucket_parsed"] == 1.0
    assert tel["meta_first_rate"] == 1.0 and tel["has_why_rate"] == 1.0
    # ★배분: hard 에 k_max=3 을 주면(평균 2) 다수결이 두 문제를 구제한다 — 균일 예산 2 는
    #   p2/p3 에서 "9"/"7" 1표 vs "42" 1표 동률이라 첫 표본으로 떨어져 여전히 틀린다.
    assert math.isclose(tel["acc_alloc_b2"], 1.0)
    assert tel["acc_alloc_b2"] > tel["acc_uniform_b2"]
    assert math.isclose(tel["alloc_gain_b2"], tel["acc_alloc_b2"] - tel["acc_uniform_b2"])
    assert tel["alloc_gain_b2_ci_lo"] <= tel["alloc_gain_b2"] <= tel["alloc_gain_b2_ci_hi"]
    assert set(DE._SUMMARY_KEYS) <= set(tel), "요약 키가 텔레메트리에 전부 있어야 한다"
    rowsB = {r["budget"]: r for r in tel["allocation"]["rows"]}
    assert set(rowsB) == set(DE.BUDGETS)
    # B=1 은 «배분 없음» — 세 곡선이 같아야 한다(sanity).
    assert math.isclose(rowsB[1.0]["acc_alloc"], rowsB[1.0]["acc_uniform"])
    assert math.isclose(rowsB[2.0]["realized_budget_stated"], 2.0)
    assert math.isclose(rowsB[2.0]["realized_budget_uniform"], 2.0)
    assert "B=2.0" in DE.format_summary(tel)


def test_eval_prompt_is_the_training_prompt():
    problems = [{"problem": "p0", "gold": "42"}]
    seen = {}

    def gen(prompts):
        seen["msgs"] = prompts[0]
        return [[(_text("42"), 0, 10)] * 3]
    DE.sample_rows(problems, gen)
    assert seen["msgs"] == build_math_prompt("p0", "math_diff")


def test_uniform_and_alloc_budgets_match():
    ks = DE.uniform_ks(10, 1.5, 8)
    assert sum(ks) / 10 == 1.5 and set(ks) == {1, 2}
    ks2, kmax = DE.alloc_ks([True, False, False, False], 1.75, 8)
    assert kmax == 4 and ks2 == [4, 1, 1, 1] and sum(ks2) / 4 == 1.75
    # hard 가 하나도 없으면 배분이 불가능 — 전부 1(정직하게 이득 0 으로 보고된다)
    assert DE.alloc_ks([False, False], 3.0, 8) == ([1, 1], 1)
    # 표본이 모자라면 k_max 가 잘린다 — realized_budget 이 B 보다 작아진다(호출자가 보고한다)
    ks3, kmax3 = DE.alloc_ks([True] + [False] * 9, 3.0, 8)
    assert kmax3 == 8


def test_majority_vote_falls_back_to_first_on_tie():
    rows = [{"final_answer": "1"}, {"final_answer": "2"}]
    assert DE.majority_vote(rows, 2) == "1"
    assert DE.majority_vote([{"final_answer": ""}], 1) is None
    assert DE.majority_vote([{"final_answer": "0.5"}, {"final_answer": "\\frac{1}{2}"},
                             {"final_answer": "3"}], 3) in ("0.5", "\\frac{1}{2}")


# ── 12. 게이트가 이 평가의 키를 읽는가 ──────────────────────────────────────────
def test_gate_judgment_reads_acc_first(tmp_path, monkeypatch):
    import json as _json

    import gate_judgment as G
    monkeypatch.setattr(G, "WORK", tmp_path)
    d = tmp_path / "eval" / "cd9_M_DIFF_s2" / "step_30" / "math500_diff_8k"
    d.mkdir(parents=True)
    (d / "telemetry.json").write_text(_json.dumps({"acc_first": 0.42, "acc_alloc_b2": 0.55}))
    got = G.read_acc("cd9_M_DIFF_s2", 30, eval_subdir="math500_diff_8k", acc_key="acc_first")
    assert math.isclose(got, 0.42)
