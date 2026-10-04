# REVISION 팔 — 발사 절차 (0918 사전등록 수정 6)

팔은 `src.training.math_meta.MATH_ARM_SPECS` 에 **들어가 있다**. 배선은 MATH_META 경로다
(`_compute_math_arm_stash` → `_math_revision_stash` → `_MATH_REGION_STASH["rev"/"rev_spans"]`
→ GRPO 어드밴티지 뒤 `_math_add_revision_advantage`). 중심화 없음, 구간은 첫 `\boxed` 끝 →
마지막 `\boxed` 시작. 시퀀스 보상(gold 정오)은 불변.

| 팔 | 처치 |
|---|---|
| `M_G1` | 결과만 — **이 네 팔 계획의 대조군이다**(새 팔을 만들지 않았다: math_opt·meta_term None·require_meta False 로 세 팔과 프롬프트가 글자 그대로 같다) |
| `M_REV_CF` | 결과 + 행 내부 정오개선 크레딧(추가 forward 없음) |
| `M_REV_PMI_GOLD` | 결과 + 교사강제 믿음 이동, 앵커 `gold_x` |
| `M_REV_PMI_COMBO` | 결과 + 교사강제 믿음 이동, 앵커 `combo`(다수답이 틀린 그룹에서 구제 2배) |

## 발사

```bash
scripts/local/run_math_arm.sh M_G1            <SEED> 50
MATH_REV_W=1.0 scripts/local/run_math_arm.sh M_REV_CF        <SEED> 50
MATH_REV_W=1.0 scripts/local/run_math_arm.sh M_REV_PMI_GOLD  <SEED> 50
MATH_REV_W=1.0 scripts/local/run_math_arm.sh M_REV_PMI_COMBO <SEED> 50
```

- `MATH_REV_W` 기본 1.0. **0 이면 런처가 즉사시킨다**(M_G1 과 바이트 동일한 무효 레버).
- `MATH_REV_ANCHOR` 는 **비워 둔다** — 팔 이름이 앵커를 정한다(GOLD→`gold_x`, COMBO→`combo`).
  덮으면 두 PMI 팔이 같은 처치가 될 수 있다. `self_mx` 탐색을 할 때만 명시적으로 준다.
- `RESP_LEN` 기본은 이 세 팔에서 **8192** 다(다른 팔 4096). 수정 행은 길다 — 참조 롤아웃에서
  평균 4,446 토큰(전체 평균 2,885). 4096 이면 처치가 걸린 행만 골라 잘라낸다.
- 나머지 손잡이(`dcpo_revpmi_*` / `dcpo_revcf_*`)는 `algorithm` 설정에서 읽는다. 기본값으로
  충분하며, 바꾸려면 `++algorithm.dcpo_revcf_save=...` 처럼 넘긴다. 목록은 `core/KNOBS.yaml`.

## 계기 (`[MATH][TEL]` 줄 꼬리)

`rev_rate` · `rev_member` · `rev_save` · `rev_derail` · `rev_shift` · `rev_skip_state` ·
`rev_skip_multi` · `first_correct`. 사행 감시는 `first_correct` 이력으로 돌고 위반 시
`_CountdownAbort`(rc 75 + ABORTED.txt)를 던진다.

## 평가

```bash
python scripts/local/math_revision_eval.py --rollouts <texts.jsonl> --out <DIR> [--compare <base.jsonl>]
```

## DCPO(TRIOBJ_DCPO_V4) 쪽 경로

같은 크레딧을 `dcpo_rmeta_source=revision_cf|revision_pmi` 로도 쓸 수 있다(구간 토큰 마스크
`dcpo_rev_zone_mask` + `rmeta_center=False`). 수학 팔은 그 경로를 타지 않는다 — 두 경로가
`_compute_revision_rmeta` 라는 **같은 함수**를 공유한다는 점만 기억하면 된다.
