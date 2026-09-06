r"""`coupling_sft_v1.parquet` (Task A 실제 산출물) 5행 표본 read-only 감사.

무엇을 지키는가 (0907 Task A):
  1. `messages` 어디에도 힌트 텍스트(`hint_text` 컬럼 값, "Hint: ...")가 새어
     들어가지 않았는가 — golden demo 는 **힌트 없이** 학습시켜야 한다
     (`scripts/local/build_coupling_sft.py` 모듈 docstring §5).
  2. `wrong_prefix`/`scenario` 조합이 `src/training/sft.py::_should_mask_prefix`
     규약과 일치하는가 — 이 corpus 는 전부 scenario=="redirect" ∧ wrong_prefix
     nonempty 여야 하고, 그 경우 `_should_mask_prefix` 는 True(=프리픽스
     loss-mask)를 반환해야 한다.
  3. `messages[-1]["content"]` (assistant 응답)가 `wrong_prefix` 로 시작하는가
     (프리픽스가 실제로 응답의 머리에 있어야 마스킹이 의미가 있다).

산출물이 없는 환경(이 parquet 은 `$WORK` 에만 있고 repo 에는 없다)에서는 skip한다
— 이 테스트는 "만들어진 실제 파일이 규약을 지키는가"의 read-only 감사이지,
빌드 자체를 재현하지 않는다.

실행:  python -m pytest tests/test_coupling_sft_v1_output.py -q
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WORK = os.environ.get("WORK", "/hdd_data/seungpil/scratch")
OUT_PATH = Path(WORK) / "data" / "sites_v1" / "coupling_sft_v1.parquet"

pd = pytest.importorskip("pandas")

pytestmark = pytest.mark.skipif(
    not OUT_PATH.exists(),
    reason=f"{OUT_PATH} not present in this environment (generated artifact under $WORK)")


def _load_sample(n=5):
    df = pd.read_parquet(OUT_PATH)
    assert len(df) > 0, f"{OUT_PATH} is empty"
    return df.head(n)


def test_no_hint_text_in_messages():
    from scripts.local.build_coupling_sft import build_sft_row  # noqa: F401  (import-side sanity)

    sample = _load_sample()
    for _, row in sample.iterrows():
        hint_text = str(row.get("hint_text") or "")
        for msg in row["messages"]:
            content = str(msg["content"])
            assert "Hint:" not in content, (
                f"site {row['site_id']!r}: literal 'Hint:' leaked into messages")
            if hint_text:
                assert hint_text not in content, (
                    f"site {row['site_id']!r}: hint_text leaked verbatim into messages")


def test_scenario_and_wrong_prefix_match_should_mask_prefix_convention():
    from src.training.sft import _should_mask_prefix

    sample = _load_sample()
    for _, row in sample.iterrows():
        assert row["scenario"] == "redirect"
        assert row["wrong_prefix"], f"site {row['site_id']!r}: wrong_prefix is empty"
        assert _should_mask_prefix(row["wrong_prefix"], row["scenario"]) is True


def test_assistant_response_starts_with_wrong_prefix():
    sample = _load_sample()
    for _, row in sample.iterrows():
        assistant_content = str(row["messages"][-1]["content"])
        assert assistant_content.startswith(row["wrong_prefix"]), (
            f"site {row['site_id']!r}: assistant response does not start with wrong_prefix")


def test_messages_have_no_assistant_role_before_last():
    """system/user 만 프리픽스 앞에 오고, assistant 는 마지막 한 번뿐이어야 한다 —
    힌트 없는 원본 프롬프트 뒤에 정확히 하나의 (프리픽스+이어쓰기) 응답만 붙는다."""
    sample = _load_sample()
    for _, row in sample.iterrows():
        roles = [m["role"] for m in row["messages"]]
        assert roles[-1] == "assistant"
        assert roles[:-1].count("assistant") == 0
