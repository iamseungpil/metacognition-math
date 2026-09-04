"""메타 블록 품질을 재는 자(ruler) 패키지 — cd6 실험(`docs/POSTMORTEM_cd6_rulers_
2026-09-03.md`, `docs/VERDICT_cd6_pair_rulers.md`)의 흩어진 스코어러를 한
`Ruler` 프로토콜(`src.rulers.base.Ruler`) 아래로 모으고, `src.rulers.table`이
표준 채점 방법론(within-site AUC/Spearman, outcome-fixed, 공격 배터리, partial
correlation, advantage 시뮬레이션)을 임의의 자에 적용한다.

CLI: `scripts/local/ruler_table.py`.
"""
from src.rulers.base import MetaSample, Ruler, Site  # noqa: F401
