# ARCHITECTURE — metacognition-math

> **START HERE.** 10분 안에 "지금 뭐가 도는가"를 알기 위한 지도. 세부 근거는 각 절이
> 가리키는 문서에 있다.

## (a) 한 줄 목표

**메타인지 행동을 강화해서 정확도를 올린다.** calibration(자기 confidence를 정답률에
맞추기)은 부분 목표일 뿐이고, 최종 목표는 *좋은 메타인지 행동 → 정답률 향상*이다.
그러려면 먼저 metacot(언제·무엇이 좋은 메타인지인가)를 정의하고, 그 행동이 실제로
정답률을 올릴 때만 보상하는 RL을 설계해야 한다. 전문은 `CLAUDE.md`의 "Intent" 절.

## (b) 세 세대 요약 (prior round — Countdown, cd9 이전)

| 세대 | 무엇을 했나 | 배운 것 | 증거 |
|---|---|---|---|
| 1. instruct PMI-shift | Qwen3-8B(**instruct**) 기질, meta-SFT + PMI-shift(자기 gold-vs-decoy 증거 이동) 보상 RL | 방법이 **작동한다** — MATH500 +14.00pp, OOD(L4–5)에서 기울기 더 큼. 단 기울기의 출처는 프라이밍이지 RL 보상이 아님 | `docs/archive/CLAIMS.md` C-001 계열, `README.md` "검증된 것" |
| 2. Qwen3-8B-Base 복제 | 같은 방법을 **pretrained-only base** 기질로 이식(RQ3v2 사다리, b0p/b2p/b3p) | **복제 실패.** 프라이밍은 base에서 널(+0.18pp, C-026), 우리 보상 패키지는 통제군보다 **음수**(−2.48pp, C-029) — instruct 이득은 그 기질 고유의 성질이었다 | `docs/archive/CLAIMS.md` C-026·C-029 |
| 3. Countdown 자(ruler) 탐색 (cd6) | base 복제가 막히자 과제를 Countdown(다중해 산술 탐색)으로 바꾸고, "모델 속 신호로 좋은 메타를 가려내는 자" 25개를 적대적으로 검증 | **내부 자 전부 탈락.** Countdown은 정답이 여럿이라 "정답 닮음 = 좋음"이 성립하지 않는다. 살아남은 것은 모델 속을 안 보는 근거-진리(완전열거) 계획 항 하나뿐이고, 그것으로 학습해 정확도가 오르는지는 **아직 확인 안 됨(결과 0건)** | `docs/archive/FINDINGS_cd6.md`, `docs/archive/POSTMORTEM_cd6_rulers_2026-09-03.md` |
| 4. **현재 — cd7 SC 라운드** | 근거-진리 없이, 모델 **자신의** 신호(프리픽스 시도 수·자기보고 confidence/decision·메타 뒤 행동)만으로 "막히면 새 계열 탐색, 과신하면 실제 재계산"을 보상(SC 팔). 로컬 H100에서 100스텝, 판정 30/50/100 | **결과 0건 시점** — 학습 전 gs0 기준선만 관측. new 프롬프트에서 explore 발동률 0.0145로 낮고, RL이 이걸 못 올리면 "침묵 항" 판정 | `docs/PREREGISTRATION_countdown_sc_round.md` |

## (c) Countdown 라이브 경로 (prior round — spine, cd7 당시)

```
scripts/local/gpu_queue.py start-workers 0 1 2 3      GPU당 워커 1개, 큐 폴링
   │  submit --cmd "bash scripts/local/run_arm.sh <ARM> <SEED> 100 <VARIANT>"
   ▼
scripts/local/run_arm.sh ARM SEED [STEPS=100] [VARIANT]
   │  python -m src.training.verl_sdc --config-name=countdown_6arm \
   │      ++algorithm.countdown_arm=<ARM>
   ▼
src.training.verl_sdc                                  hydra 진입점 / RayPPOTrainer
   │
   ├─ src.training.countdown_rewards.arm_reward         팔 정체(ARM_SPECS)와 항(TERMS) 조립
   │    └─ src.training.countdown_selfcontrol.sc_row    SC/SCg 팔의 원재료(stuck·hi·novel·
   │                                                      followed·checked) — 순수 함수, torch 무의존
   │    └─ src.training.countdown_task.grade             정답 채점(완전 열거 기반)
   ▼
verl.model_merger merge --backend fsdp                  판정 스텝(30/50/100)마다 FSDP 샤드 → bf16 병합
   ▼
scripts/countdown_gs0_eval.py                           held-out 500×8 평가, 씨앗 11
   ▼
scripts/local/hf_upload.py                              병합 체크포인트만 HF 업로드
                                                          (iamseungpil/metacot-countdown-local)
```

팔 하나 = GPU 한 장. 현재 라운드(cd7)의 팔: N0(맨 GRPO 기준선) / A(메타 요구, 무채점) /
SC(자기제어) / G(길이 위약) + SC_GH(정답 항 뺀 굿하트 압력시험, 20스텝, 학습 주장 미사용).
전체 사용법은 `scripts/local/README.md`, 설계·판정 기준은
`docs/PREREGISTRATION_countdown_sc_round.md`.

## (d) cd9 math (현행, 2026-09-14)

Countdown(b/c) 뒤 무대를 수학으로 옮긴 현행 라운드. 라이브 모듈:

| 파일 | 역할 |
|---|---|
| `src/training/math_meta.py` | MATH_ARM_SPECS(팔 정의) · ABORT_RULES(중단 규칙) · 재시도 판단 진리표(`retry_judgment_parts`) · 텔레메트리 |
| `src/metacot/math_meta_prompt.py` | 팔별 프롬프트 변형(`math_plain`/`math_opt`/`math_retry`/`math_retry_forced`) |
| `scripts/local/build_math_parquet.py` | 학습/평가 parquet 빌드, `--level`·`--forced_frac`·`--forced_from_rollouts`(mixed-only 강제 배정) |
| `scripts/local/run_math_arm.sh` | 팔 하나를 GPU 하나에서 학습→머지→평가까지 도는 러너(`run_arm.sh` Countdown 관례 계승) |
| `scripts/local/math_retry_eval.py` | 재시도 팔 사후 평가(첫 답/판단/재시도 분해 — `math_rollout.py` 는 이 분해가 없어 M_RETRY 판정 불가) |
| `scripts/local/gpu_queue.py` | GPU 큐(2·3만 사용) — Countdown 때와 같은 큐를 그대로 공유 |

사전등록 `docs/PREREGISTRATION_cd9_math_judgment.md`, 원장 `docs/RESULTS_cd9.md`. Countdown 사전등록
(`archive/docs_countdown_cd7_cd8_2026_09_14/PREREGISTRATION_countdown_sc_round.md`)은 cd7 아카이브로
옮겨졌다 — cd9 은 새 사전등록을 쓴다.

## (e) 모듈 지도

| 상태 | 파일 |
|---|---|
| **LIVE (Countdown, cd7)** | `src/training/countdown_rewards.py`(ARM_SPECS/TERMS 단일 정의처), `src/training/countdown_selfcontrol.py`(SC 행 특징), `src/training/countdown_task.py`(채점·프롬프트), `src/training/verl_sdc.py`(트레이너), `src/training/verl_sdc_utils.py`, `src/training/sft.py`, `src/training/tokenizer_utils.py`, `scripts/countdown_gs0_eval.py`, `scripts/local/*.py`·`*.sh`(큐·러너·데이터·업로드·디스크가드) |
| **math-DCPO 경로 — 보존되나 현재 안 돎** | `src/training/dcpo_pmi_shift.py`, `src/training/dcpo_region.py`, `src/training/rewards.py`, `src/training/meta_revision_rewards.py`, `src/training/_decoy_utils.py` — 세대 1·2(instruct PMI-shift, base 복제)의 방법. 재현 가능하나 cd7은 이 경로를 안 쓴다 |
| **verl_sdc.py 안의 config-inert 계열** | `cf_prefix_agent`/`meta_inject`/GFN/teacher 관련 블록과 구세대 `REWARD_CONFIGS` 12개 — import는 되지만 `countdown_6arm` config에서는 선택되지 않는 죽은 분기. `core/KNOBS.yaml`에 노브별로 등록돼 있고, **이번 라운드가 끝난 뒤** 테스트로 지켜가며 제거 예정(지금은 건드리지 않는다 — 도는 arm이 이 파일을 공유한다) |

## (f) 아카이브 배치

| 디렉터리 | 무엇 | 시점 |
|---|---|---|
| `archive/dead_code_2026_09_04/` | 임포터 0건인 src 모듈 10개 + 고아 스크립트 24개 + `scripts/retired/` 전체 | 2026-09-04 |
| `archive/launchers_retired_0904/` | 현행 6개 arm(SFT2 쌍 3 + RL 3)을 제외한 루트 `h100std_*.yaml` 27개 | 2026-09-04 |
| `archive/launchers_retired_0818/`, `_0803/`, `_0727/`, `launchers_pre_rq3/` | 각 세대 전환 시 은퇴한 amlt 런처 (날짜별) | 각 날짜 |
| `archive/reward_lineages_retired_0803/`, `reports_pre_regrade_0803/` | 재채점 이전 보상 계보·리포트 | 2026-08-03 |
| `archive/incidents_pre_rq3/`, `docs_pre_rq3/`, `data_pre_rq3/`, `results_archive/`, `runs_archive/`, `2026_04_16_cleanup/`, `dead_code_2026_07_12/` | 세대 1(instruct) 이전 문서·데이터·런·정리 이력 | 각 날짜 |

각 디렉터리에 왜 옮겨졌는지 설명하는 `README.md`가 있다(예:
`archive/dead_code_2026_09_04/README.md`, `archive/launchers_retired_0904/README.md`).
아무것도 삭제되지 않았다 — `git log --follow <경로>`로 이력을 볼 수 있다.

## 더 보기

- `docs/PREREGISTRATION_cd9_math_judgment.md`, `docs/RESULTS_cd9.md` — 현행(cd9) 사전등록·원장
- `archive/docs_countdown_cd7_cd8_2026_09_14/PREREGISTRATION_countdown_sc_round.md` — Countdown
  라운드(cd7) 설계·판정 기준 (아카이브)
- `docs/archive/POSTMORTEM_cd6_rulers_2026-09-03.md`, `docs/archive/FINDINGS_cd6.md` — 왜 자 탐색에서
  Countdown 자기제어 라운드로 왔는가
- `docs/CODE_MAP.md` — math-DCPO 경로(세대 1·2)의 상세 호출 사슬과 config 함정
- `scripts/local/README.md` — 실행 방법
- `NODE_POLICY.md`, `docs/mainline_registry_2026_04_13.md` — DEPRECATED(pre-rq3 세대)
