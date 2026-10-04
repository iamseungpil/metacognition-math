r"""`scripts/local/build_restraint_sft.py` 순수 함수 회귀 테스트 — CPU 전용,
모델·parquet I/O 없이 딕셔너리로만 검증한다.

무엇을 지키는가:
  · `row_passes_restraint` — 건강 자리(family_dead==0) ∧ 메타 미발현 ∧ 정답 ∧
    비절단 만 통과. family_dead 가 1/None/NaN 이면 탈락(근거 없는 행 금지).
  · `row_passes_decorative` — 건강 자리 ∧ 메타 발현 ∧ 프리픽스 **뒤**의 메타.
  · `build_restraint_row` — restraint 는 wrong_prefix=prefix, decorative 는
    `</meta>` 까지 확장(= sft.py 가 장식 블록을 통째로 loss-mask 한다),
    둘 다 scenario=="redirect"(마스크 스위치)이고 스키마는 coupling 과 동일.
  · `summarize_kinds` — kind 별 통과/탈락 집계.

실행:  python -m pytest tests/test_build_restraint_sft.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.local.build_coupling_sft import _is_dead  # noqa: E402  (재사용 확인)
from scripts.local.build_restraint_sft import (  # noqa: E402
    KINDS, build_restraint_row, meta_end_after_prefix, row_passes_decorative,
    row_passes_restraint, summarize_kinds,
)

PREFIX = "PREFIX. 5+3=8.\n"
# 건강 자리에서 메타 없이 곧장 정답으로 간 이어쓰기.
_NOMETA_TEXT = PREFIX + "8*2=16, 16+4=20.\n\\boxed{8*2+4}"
# 같은 자리에서 굳이 검산 블록을 낸 이어쓰기(장식).
_META_TEXT = (PREFIX + "<meta>\nconfidence: 0.9\nThe additive family looks fine.\n"
              "decision: verify\n</meta>\n8*2=16, 16+4=20.\n\\boxed{8*2+4}")


def _cont_row(site_id="s1", *, mode="meta", family_dead=0, emitted=0, r_corr=1,
              truncated=0, k_index=0, full_text=_NOMETA_TEXT, prefix=PREFIX):
    return {
        "site_id": site_id, "mode": mode, "family_dead": family_dead, "emitted": emitted,
        "r_corr": r_corr, "truncated": truncated, "k_index": k_index,
        "full_text": full_text, "prefix": prefix, "decision": "none",
        "novel": 0, "followed": 0, "n_tokens": 12,
    }


def _site_row(prefix=PREFIX, family_dead=0, nums=(1, 2, 3, 4), target=20):
    return {
        "prompt": [{"role": "system", "content": "SYS"},
                   {"role": "user", "content": "Numbers: [1, 2, 3, 4]\nTarget: 20"},
                   {"role": "assistant", "content": prefix}],
        "prefix": prefix, "family_dead": family_dead, "nums": list(nums),
        "target": target, "cut_type": "attempt-boundary",
    }


# ══════════════════════════════════════════════════════════════════════════════
# row_passes_restraint
# ══════════════════════════════════════════════════════════════════════════════

def test_clean_restraint_row_passes():
    ok, stage = row_passes_restraint(_cont_row())
    assert ok and stage == ""


@pytest.mark.parametrize("field,bad,expected", [
    ("mode", "nometa", "mode_meta"),        # 다른 system 프롬프트로 생성 → 절제의 증거 아님
    ("family_dead", 1, "alive_site"),       # 막힌 자리는 절제 대상이 아니다
    ("emitted", 1, "no_meta_emitted"),
    ("r_corr", 0, "r_corr"),
    ("truncated", 1, "not_truncated"),
])
def test_restraint_rejects_each_field(field, bad, expected):
    ok, stage = row_passes_restraint({**_cont_row(), field: bad})
    assert not ok and stage == expected


@pytest.mark.parametrize("fd", [None, float("nan")])
def test_restraint_rejects_unknown_family_dead(fd):
    """시도 0회(family_dead 없음) site 는 건강/막힘 근거가 없으므로 조용히 통과시키지
    않는다 — `_is_dead` 가 False 를 주더라도 절제 양성으로는 쓰지 않는다."""
    assert not _is_dead(fd)                                # 막힘은 아니지만
    ok, stage = row_passes_restraint({**_cont_row(), "family_dead": fd})
    assert not ok and stage == "alive_site"                # 그래도 탈락


# ══════════════════════════════════════════════════════════════════════════════
# row_passes_decorative / meta_end_after_prefix
# ══════════════════════════════════════════════════════════════════════════════

def test_clean_decorative_row_passes():
    row = _cont_row(emitted=1, full_text=_META_TEXT)
    ok, stage = row_passes_decorative(row)
    assert ok and stage == ""


def test_decorative_requires_emitted_meta():
    ok, stage = row_passes_decorative(_cont_row())
    assert not ok and stage == "meta_emitted"


def test_decorative_rejects_meta_inside_prefix():
    """메타가 site 프리픽스 **안**에 있으면(정책이 새로 낸 장식이 아니라 자리 자체의
    일부) decorative 로 쓸 수 없다."""
    prefix = "PREFIX <meta>\nold\n</meta>\n"
    row = _cont_row(emitted=1, prefix=prefix, full_text=prefix + "1+1=2\n\\boxed{1+1}")
    assert meta_end_after_prefix(row["full_text"], prefix) is None
    ok, stage = row_passes_decorative(row)
    assert not ok and stage == "meta_after_prefix"


def test_decorative_r_corr_gate_is_optional():
    row = _cont_row(emitted=1, r_corr=0, full_text=_META_TEXT)
    assert row_passes_decorative(row, require_correct=True) == (False, "r_corr")
    assert row_passes_decorative(row, require_correct=False)[0]


# ══════════════════════════════════════════════════════════════════════════════
# build_restraint_row
# ══════════════════════════════════════════════════════════════════════════════

_COUPLING_COLS = {"site_id", "messages", "wrong_prefix", "scenario", "cut_type",
                  "family_dead", "decision", "r_corr", "novel", "followed",
                  "n_tokens", "hint_text", "nums", "target"}


def test_restraint_row_schema_matches_coupling_plus_kind():
    out = build_restraint_row(_cont_row(), _site_row(), kind="restraint")
    assert set(out) == _COUPLING_COLS | {"kind"}


def test_restraint_row_masks_only_the_site_prefix():
    out = build_restraint_row(_cont_row(), _site_row(), kind="restraint")
    assert out["wrong_prefix"] == PREFIX
    assert out["scenario"] == "redirect"          # sft.py::_should_mask_prefix 스위치
    assert out["messages"][-1] == {"role": "assistant", "content": _NOMETA_TEXT}
    assert [m["role"] for m in out["messages"]] == ["system", "user", "assistant"]
    trained = out["messages"][-1]["content"][len(out["wrong_prefix"]):]
    assert "<meta>" not in trained                # 절제를 가르치는 행 — 블록이 없다


def test_decorative_row_masks_through_the_meta_block():
    row = _cont_row(emitted=1, full_text=_META_TEXT)
    out = build_restraint_row(row, _site_row(), kind="decorative")
    assert out["wrong_prefix"].endswith("</meta>")
    assert out["wrong_prefix"] == _META_TEXT[:_META_TEXT.index("</meta>") + len("</meta>")]
    trained = out["messages"][-1]["content"][len(out["wrong_prefix"]):]
    assert "<meta>" not in trained                # 장식 블록은 학습되지 않는다
    assert "\\boxed{" in trained                  # 뒤 풀이만 학습된다


def test_decorative_row_without_meta_dies_loudly():
    with pytest.raises(ValueError):
        build_restraint_row(_cont_row(), _site_row(), kind="decorative")


def test_prefix_mismatch_dies_loudly():
    row = _cont_row(full_text="OTHER PREFIX. \\boxed{1+1}")
    with pytest.raises(ValueError):
        build_restraint_row(row, _site_row(), kind="restraint")


def test_unknown_kind_dies_loudly():
    with pytest.raises(ValueError):
        build_restraint_row(_cont_row(), _site_row(), kind="verify")
    assert KINDS == ("restraint", "decorative")


# ══════════════════════════════════════════════════════════════════════════════
# summarize_kinds
# ══════════════════════════════════════════════════════════════════════════════

def test_summarize_counts_both_kinds():
    rows = [
        _cont_row("a"),                                            # restraint 통과
        _cont_row("b"),                                            # restraint 통과
        _cont_row("c", emitted=1, full_text=_META_TEXT),           # decorative 통과
        _cont_row("d", family_dead=1),                             # 둘 다 탈락(막힘)
    ]
    s = summarize_kinds(rows, require_correct_decorative=True)
    assert s["total_rows"] == 4
    assert s["restraint"]["pass"] == 2
    assert s["decorative"]["pass"] == 1
    assert s["restraint"]["rejected_at"]["alive_site"] == 1
    assert s["restraint"]["rejected_at"]["no_meta_emitted"] == 1
