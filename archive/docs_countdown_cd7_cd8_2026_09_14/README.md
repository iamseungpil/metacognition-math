# 보관 — Countdown cd6~cd8 세대 사전등록·설계 문서 (2026-09-14 이관)

이 폴더의 문서는 **종료된 라운드**의 사전등록·설계다. 사후 수정하지 않는다(사전등록 규약).
판정은 `docs/RESULTS_cd7.md`(원장), `docs/POSTMORTEM_cd6_rulers_2026-09-03.md`,
`docs/VERDICT_cd6_pair_rulers.md` 에 남아 있다.

| 문서 | 종료 판정 | 후속 |
|---|---|---|
| PREREGISTRATION_countdown_6arm / sc_round / osd_round2 / anchored_2x2 | 내부 자 25종 전부 결과 고정 AUC ≤ 0 → 자 계열 종료. 학습 팔은 N0(순수 GRPO)를 넘지 못함. 4수 Countdown 은 Qwen3.5 학습 전 .876 으로 포화 | `docs/PREREGISTRATION_cd9_math_judgment.md` |
| PREREGISTRATION_rq3v2_base_replication | amlt VC 차단(0726)으로 중단. 클러스터 복구 시에만 유효 | CLAUDE.md «amlt 복구 시» 절 |
| DESIGN_opd_hint_teacher | OPD 계열은 SAVE 밀도 3~15% 에서 학습 신호 굶음 → 폐기 | cd9 밀도 게이트(SAVE ≥ 5%) |
| CODE_MAP, mainline_registry_2026_04_13 | RQ3v2(Qwen3-8B, math-DCPO) 세대 인벤토리. 현행 경로 아님 | `ARCHITECTURE.md` |
