# 아카이브 계획 — 옛 트리 → `archive/` (제안, **실행하지 않았다**)

`mc/` 는 `src/` 를 한 줄도 import 하지 않는다(`grep -rn "from src\|import src" mc/` → 0건).
그래서 GPU 스모크가 통과하면 아래를 옮겨도 활성 경로는 움직이지 않는다. **이동은 스모크
통과 뒤 조정자가 한다.**

## 남길 것 (활성)
| 경로 | 이유 |
|---|---|
| `mc/` · `tests/mc/` | 활성 코드 전부. **주입 2턴은 측정만** 남았다(학습 경로는 0921 삭제 — mc/ 는 커밋이 없었으므로 git 이력에 없고, 사본은 `/hdd_data/seungpil/tmp/mc_deleted_2turn_0921/`) |
| `docs/` · `paper/` | 원장·사전등록·계획서·논문. 결과는 코드보다 오래 산다 |
| `scripts/local/gpu_queue.py` | **지금 돌고 있는 워커** — 절대 건드리지 않는다 |
| `scripts/local/env.sh` | `mc/run.sh` 가 source 한다(REPO_ROOT·WORK·PATH) |
| `scripts/local/retry_cmd.sh` | 큐 잡 cmd 가 쓴다(rc 75 는 재시도 안 함) |
| `configs/countdown_6arm.yaml` | `mc/run.sh` 의 `--config-name` 기본값(예산·배관 절반) |
| `CLAUDE.md` · `ARCHITECTURE.md` · `core/KNOBS.yaml` | 프로젝트 규약 |
| `.env` · `requirements*` · `pyproject`/`pytest.ini` 류 | 환경 |

⚠`configs/countdown_6arm.yaml` 이 `mc/run.sh` 의 유일한 옛-트리 의존이다. 다음 라운드에
`mc/train.yaml` 로 필요한 키만 뽑아 두면 `configs/` 도 통째로 아카이브할 수 있다.

## 옮길 것 (archive/)
| 경로 | 무엇인가 |
|---|---|
| `src/` 전부 | verl_sdc(8,447줄)·trial2·math_meta·dcpo_*·countdown_*·metacot·eval·curriculum·training 전부. 필요한 조각은 mc/ 로 이미 옮겼다 |
| `scripts/local/*` (위 3종 제외) | math_rollout·math_sites·math_*_eval·math_*_gate·build_*·regrade_jsonl·math_pmi_shift_probe 등 ~60개 일회성 도구 |
| `scripts/*` (local 제외) | patch_math_verify·push_ckpts_to_hf·pull_parquets·ruler_battery 등 |
| `tests/*` (tests/mc 제외) | 옛 트리의 테스트 — 옮긴 코드와 함께 간다 |
| `configs/*` (countdown_6arm 제외) | verl_sdc_*·meta_*·cf_prefix_agent·sft_* yaml |
| `sitecustomize.py` | verl agent-loop 의 이질 프롬프트 폭 concat 패치. **mc 는 필요 없다** (mc/README «아카이브 시» 항목의 근거 셋) — 오히려 PYTHONPATH 로 모든 Ray 워커에 들어가 워커 기동을 느리게 해 2026-09-21 의 «영원히 대기» 사고를 만든 요인이다 |
| `amlt/`·루트 `*.yaml` 런처 | amlt 세대(클러스터 차단 상태) |
| `code_snapshots/`·`archive/incidents_pre_rq3/` | 이미 죽은 세대 |

⚠`scripts/patch_math_verify.py` 는 **아카이브 뒤에도 적용된 상태여야 한다**(math_verify 의
timeout 래퍼가 워커 스레드에서 정답을 조용히 오답으로 만든다). `mc.grade.selftest()` 가
깨져 있으면 즉사하므로 탐지는 되지만, 패치 파일 자체는 지우지 말고 archive 에서 꺼내 쓴다.

## 이동 뒤 예상 줄 수
| 묶음 | 줄 |
|---|---|
| `mc/*.py` + `mc/*.sh` | 2,637 |
| `tests/mc/*.py` | 1,136 |
| 남기는 워커 3종(`gpu_queue.py`·`env.sh`·`retry_cmd.sh`) | 586 |
| **활성 파이썬/셸 합계** | **4,359** |
| (지금 `src/` + `scripts/` + `tests/`) | 약 60,000 |

목표선(전체 5,000 이하)을 여유 있게 지킨다. `mc/README.md`·`mc/ARCHIVE_PLAN.md` 는
위 합계에 들어가지 않는다(문서).

## 되돌리기
`archive/` 는 같은 저장소 안의 이동이므로 되돌리는 것은 경로 복원 하나다. 이동 전에
`wc -l` 전수와 `pytest tests/mc` 결과를 원장에 적어 두면 «무엇이 활성이었나»가 남는다.
