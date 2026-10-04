"""관문 G1 — `mc.grade.grade_math` 가 구 `src.training.math_meta.grade_math` 와 **바이트
동일한 평결**을 내는가. 실측 산출물 2,000행에서 100% 일치여야 한다. 갈리면 mc/ 의 모든
숫자가 옛 원장과 비교 불가가 된다."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

REF = Path("/hdd_data/seungpil/scratch/eval/mathL5_base_b12k/texts.jsonl")
N = int(os.environ.get("MC_PARITY_N", "2000"))


@pytest.mark.skipif(not REF.exists(), reason=f"{REF} 없음")
def test_grade_parity_2000_rows():
    from src.training.math_meta import grade_math as old

    from mc.grade import grade_math as new

    rows = []
    with REF.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= N:
                break
    assert len(rows) == N, f"{REF} 에 {N}행이 없다({len(rows)})"
    bad = []
    for i, r in enumerate(rows):
        t, g = str(r.get("text") or ""), str(r.get("gold") or "")
        o, n = bool(old(t, g)), new(t, g)
        if o != n:
            bad.append((i, g, o, n))
    assert not bad, f"{len(bad)}/{N} 행에서 평결이 갈렸다: {bad[:5]}"
