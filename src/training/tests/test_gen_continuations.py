"""`scripts/local/gen_continuations.py` 의 순수 함수 회귀 테스트 — CPU 전용.

무엇을 지키는가:
  · `build_fed_prefix_text` — donor 모드만 prefix 뒤에 기증 메타를 붙이고, meta/nometa
    는 prefix 를 그대로 돌려준다. donor 인데 donor_meta_raw 가 없으면 즉사.
  · `build_messages_for_mode` — nometa 는 시스템 메시지만, donor 는 마지막(assistant
    프리픽스) 메시지만 바뀌고 나머지는 원본과 바이트가 같다. 원본 리스트를 변형하지
    않는다(세 모드가 같은 `prompt_msgs` 를 공유해도 서로 오염되지 않는다).
  · `parse_donor_pool` — 완결된 메타가 있는 행만 후보로 남는다.
  · `sample_donor` — 같은 문제를 제외하고 뽑는다. 필터링 후 풀이 비면 fallback 으로
    전체 풀에서 뽑고 그 사실을 알려준다.
  · `build_record` — 정답 식을 이어쓰면 r_corr=1, continuation 안의 메타만 파싱한다
    (fed prefix 안의 메타는 emitted 에 안 잡힌다).
  · `summarize` — 성공률·발화율·site 별 분산(전부 같은 결과면 분산 0)·family_dead
    별 평균이 손으로 계산한 값과 맞는가.

실행:  python -m pytest src/training/tests/test_gen_continuations.py -q
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))  # repo root

import scripts.local.gen_continuations as gc  # noqa: E402


NUMS = [5, 19, 25, 3]
TARGET = 21


def _prompt_msgs(prefix: str) -> list[dict]:
    return [
        {"role": "system", "content": "SYSTEM_META"},
        {"role": "user", "content": "Numbers: [5, 19, 25, 3]\nTarget: 21"},
        {"role": "assistant", "content": prefix},
    ]


# ══════════════════════════════════════════════════════════════════════════════
# build_fed_prefix_text
# ══════════════════════════════════════════════════════════════════════════════

def test_fed_prefix_meta_nometa_is_prefix_unchanged():
    assert gc.build_fed_prefix_text("meta", "let's try 5+19.", None) == "let's try 5+19."
    assert gc.build_fed_prefix_text("nometa", "let's try 5+19.", None) == "let's try 5+19."


def test_fed_prefix_donor_appends_after_prefix():
    out = gc.build_fed_prefix_text("donor", "let's try 5+19.", "<meta>\nX\n</meta>")
    assert out == "let's try 5+19.\n<meta>\nX\n</meta>\n"


def test_fed_prefix_donor_without_donor_raw_raises():
    try:
        gc.build_fed_prefix_text("donor", "prefix", None)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_fed_prefix_unknown_mode_raises():
    try:
        gc.build_fed_prefix_text("bogus", "prefix", None)
        assert False, "expected ValueError"
    except ValueError:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# build_messages_for_mode
# ══════════════════════════════════════════════════════════════════════════════

def test_messages_meta_mode_is_untouched_prompt():
    msgs = _prompt_msgs("let's try 5+19.")
    out = gc.build_messages_for_mode(msgs, "meta", prefix="let's try 5+19.",
                                      plain_system="PLAIN")
    assert out[0]["content"] == "SYSTEM_META"
    assert out[-1]["content"] == "let's try 5+19."
    assert out[1] == msgs[1]                       # user 메시지 그대로


def test_messages_nometa_replaces_only_system():
    msgs = _prompt_msgs("let's try 5+19.")
    out = gc.build_messages_for_mode(msgs, "nometa", prefix="let's try 5+19.",
                                      plain_system="PLAIN")
    assert out[0]["content"] == "PLAIN"
    assert out[-1]["content"] == "let's try 5+19."
    assert out[1]["content"] == msgs[1]["content"]


def test_messages_donor_replaces_only_last_message():
    msgs = _prompt_msgs("let's try 5+19.")
    out = gc.build_messages_for_mode(msgs, "donor", prefix="let's try 5+19.",
                                      donor_meta_raw="<meta>\nY\n</meta>",
                                      plain_system="PLAIN")
    assert out[0]["content"] == "SYSTEM_META"      # 시스템은 원본 그대로(meta 지시 유지)
    assert out[-1]["content"] == "let's try 5+19.\n<meta>\nY\n</meta>\n"


def test_messages_does_not_mutate_input():
    msgs = _prompt_msgs("let's try 5+19.")
    original = [dict(m) for m in msgs]
    gc.build_messages_for_mode(msgs, "nometa", prefix="let's try 5+19.", plain_system="PLAIN")
    gc.build_messages_for_mode(msgs, "donor", prefix="let's try 5+19.",
                                donor_meta_raw="<meta>\nY\n</meta>", plain_system="PLAIN")
    assert msgs == original


def test_messages_rejects_bad_shape():
    bad_no_system = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    try:
        gc.build_messages_for_mode(bad_no_system, "meta", prefix="y")
        assert False, "expected ValueError"
    except ValueError:
        pass
    bad_no_assistant_tail = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    try:
        gc.build_messages_for_mode(bad_no_assistant_tail, "meta", prefix="u")
        assert False, "expected ValueError"
    except ValueError:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# parse_donor_pool / sample_donor
# ══════════════════════════════════════════════════════════════════════════════

def _meta_text(conf: str, decision: str) -> str:
    return (f"blah blah\n<meta>\nconfidence: {conf}\n"
            f"some sentence about the approach so far here.\n"
            f"decision: {decision}\n</meta>\nmore text")


def test_parse_donor_pool_keeps_only_completed_metas():
    rows = [
        {"text": _meta_text("0.8", "verify"), "nums": [1, 2, 3, 4], "target": 10},
        {"text": "no meta block here at all", "nums": [5, 6, 7, 8], "target": 20},
        {"text": _meta_text("0.2", "redirect"), "nums": [9, 8, 7, 6], "target": 30},
    ]
    pool = gc.parse_donor_pool(rows)
    assert len(pool) == 2
    assert pool[0]["decision"] == "verify"
    assert pool[0]["confidence"] == 0.8
    assert pool[0]["problem"] == ((1, 2, 3, 4), 10)
    assert pool[1]["decision"] == "redirect"


def test_sample_donor_excludes_same_problem():
    pool = [
        {"raw": "A", "decision": "verify", "confidence": 0.8, "problem": ((1, 2, 3, 4), 10)},
        {"raw": "B", "decision": "redirect", "confidence": 0.2, "problem": ((5, 6, 7, 8), 20)},
    ]
    rng = random.Random(0)
    for _ in range(20):
        entry, fallback = gc.sample_donor(pool, ((1, 2, 3, 4), 10), rng)
        assert entry["raw"] == "B"
        assert fallback is False


def test_sample_donor_falls_back_when_pool_is_all_same_problem():
    pool = [{"raw": "A", "decision": "verify", "confidence": 0.9, "problem": ((1, 2, 3, 4), 10)}]
    rng = random.Random(0)
    entry, fallback = gc.sample_donor(pool, ((1, 2, 3, 4), 10), rng)
    assert entry["raw"] == "A"
    assert fallback is True


def test_sample_donor_empty_pool_returns_none():
    entry, fallback = gc.sample_donor([], ((1, 2, 3, 4), 10), random.Random(0))
    assert entry is None
    assert fallback is False


# ══════════════════════════════════════════════════════════════════════════════
# build_record
# ══════════════════════════════════════════════════════════════════════════════

def test_build_record_grades_full_text_meta_mode():
    prefix = "Trying a few groupings.\n"
    continuation = "Final answer: \\boxed{(25/5)+(19-3)}"
    rec = gc.build_record(site_id="s1", mode="meta", policy_tag="p0", k_index=0,
                          prefix=prefix, donor_meta_raw=None, nums=NUMS, target=TARGET,
                          continuation=continuation, n_tokens=12, truncated=False)
    assert rec["r_corr"] == 1
    assert rec["full_text"] == prefix + continuation
    assert rec["continuation"] == continuation
    assert rec["donor_meta_raw"] is None
    assert rec["site_id"] == "s1" and rec["mode"] == "meta" and rec["k_index"] == 0


def test_build_record_wrong_answer_is_zero():
    prefix = "Trying.\n"
    continuation = "\\boxed{(25/5)+(19+3)}"      # 5+22=27, target 21 이 아님
    rec = gc.build_record(site_id="s1", mode="meta", policy_tag="p0", k_index=1,
                          prefix=prefix, donor_meta_raw=None, nums=NUMS, target=TARGET,
                          continuation=continuation, n_tokens=5, truncated=False)
    assert rec["r_corr"] == 0


def test_build_record_donor_mode_includes_donor_block_in_full_text():
    prefix = "Trying.\n"
    donor_raw = "<meta>\nconfidence: 0.5\nsome judgement here.\ndecision: verify\n</meta>"
    continuation = "\\boxed{(25/5)+(19-3)}"
    rec = gc.build_record(site_id="s2", mode="donor", policy_tag="p0", k_index=0,
                          prefix=prefix, donor_meta_raw=donor_raw, nums=NUMS, target=TARGET,
                          continuation=continuation, n_tokens=8, truncated=False)
    assert donor_raw in rec["full_text"]
    assert rec["full_text"] == prefix + "\n" + donor_raw + "\n" + continuation
    assert rec["r_corr"] == 1
    # continuation 자체에는 메타가 없다 — donor 블록은 fed prefix 안에 있으므로
    # `parse_meta(continuation, ...)` 에는 안 잡혀야 한다.
    assert rec["emitted"] == 0
    assert rec["meta_raw"] == ""


def test_build_record_parses_meta_from_continuation_only():
    prefix = "Trying.\n"
    continuation = ("<meta>\nconfidence: 0.3\nthis family looks weak so far.\n"
                    "decision: redirect\n</meta>\n\\boxed{(5+19)-3}")
    rec = gc.build_record(site_id="s3", mode="meta", policy_tag="p0", k_index=0,
                          prefix=prefix, donor_meta_raw=None, nums=NUMS, target=TARGET,
                          continuation=continuation, n_tokens=20, truncated=False)
    assert rec["emitted"] == 1
    assert rec["decision"] == "redirect"
    assert rec["confidence"] == 0.3
    assert rec["meta_start_in_cont"] == 0


def test_build_record_truncated_and_n_tokens_pass_through():
    rec = gc.build_record(site_id="s4", mode="nometa", policy_tag="p0", k_index=3,
                          prefix="x", donor_meta_raw=None, nums=NUMS, target=TARGET,
                          continuation="no boxed here", n_tokens=999, truncated=True)
    assert rec["n_tokens"] == 999
    assert rec["truncated"] == 1
    assert rec["r_corr"] == 0


# ══════════════════════════════════════════════════════════════════════════════
# summarize
# ══════════════════════════════════════════════════════════════════════════════

def _rec(site_id, mode, r_corr, emitted=0):
    return {"site_id": site_id, "mode": mode, "r_corr": r_corr, "emitted": emitted}


def test_summarize_success_and_emit_rate():
    records = [
        _rec("s1", "meta", 1, emitted=1),
        _rec("s1", "meta", 0, emitted=0),
        _rec("s2", "meta", 1, emitted=1),
        _rec("s2", "meta", 1, emitted=1),
    ]
    fam = {"s1": 0, "s2": 1}
    summ = gc.summarize(records, fam)
    m = summ["per_mode"]["meta"]
    assert m["n_rows"] == 4
    assert m["n_sites"] == 2
    assert abs(m["success_rate"] - 0.75) < 1e-9
    assert abs(m["emit_rate"] - 0.75) < 1e-9


def test_summarize_variance_frac_flags_sites_with_mixed_outcomes():
    records = [
        _rec("s1", "meta", 1), _rec("s1", "meta", 1),      # all-same → no variance
        _rec("s2", "meta", 1), _rec("s2", "meta", 0),      # mixed → variance
    ]
    summ = gc.summarize(records, {"s1": None, "s2": None})
    assert summ["per_mode"]["meta"]["variance_frac"] == 0.5


def test_summarize_success_by_family_dead():
    records = [
        _rec("s1", "meta", 1), _rec("s2", "meta", 0), _rec("s3", "meta", 1),
    ]
    fam = {"s1": 0, "s2": 1, "s3": None}
    summ = gc.summarize(records, fam)
    by_fam = summ["per_mode"]["meta"]["success_by_family_dead"]
    assert by_fam["0"] == 1.0
    assert by_fam["1"] == 0.0
    assert by_fam["none"] == 1.0


def test_summarize_separates_modes():
    records = [
        _rec("s1", "meta", 1), _rec("s1", "nometa", 0),
    ]
    summ = gc.summarize(records, {"s1": None})
    assert set(summ["per_mode"].keys()) == {"meta", "nometa"}
    assert summ["per_mode"]["meta"]["success_rate"] == 1.0
    assert summ["per_mode"]["nometa"]["success_rate"] == 0.0


def test_fam_key_handles_none_and_nan():
    import math
    assert gc._fam_key(None) == "none"
    assert gc._fam_key(float("nan")) == "none"
    assert gc._fam_key(0.0) == "0"
    assert gc._fam_key(1) == "1"
