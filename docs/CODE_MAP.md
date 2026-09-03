# CODE_MAP — 신규 인력용 코드 인벤토리 (read-only, 2026-07-17 · 0904 갱신)

> **세대 주석(0904).** 이 문서 본문(§1~§8)이 서술하는 것은 **math-DCPO 경로**
> (instruct PMI-shift → Qwen3-8B-Base 복제, RQ3v2 세대)다. 이 경로는 **현재 도는 arm이
> 아니다** — 현행 실험은 Countdown cd7 SC 라운드(`ARCHITECTURE.md` (c) 참조,
> `src/training/verl_sdc.py --config-name=countdown_6arm`). 아래 메커니즘 서술 — 모드
> 분기, config 상속 순서, §2의 rmeta 함정, §3의 RGS 규칙 — 은 math-DCPO 경로를 다시 돌릴
> 때 여전히 유효하지만, Countdown 경로의 라이브 파일은 §4·§9를 볼 것. 구세대 런처는
> `archive/launchers_retired_0727/`·`_0803/`·`_0818/`·`_0904/`에 있다.

이 문서는 "지금 살아있는 코드가 무엇이고 어디서 불리는가"의 지도다. 수정
지침이 아니다 — src/·configs/·h100std_*.yaml은 tarball(CODE_TAR_REVISION)과
byte-동일 유지가 원칙이다.

## 1. 라이브 트레이너 모드 2개와 호출 사슬

**Mode 1 — VANILLA_GRPO** (컨트롤 b0p = `h100std_rq3v2f_b0p.yaml`, 메타 b2p = `h100std_rq3v2f_b2p.yaml`;
H100 판은 `h100std_rq3v2f_*`)

```
런처 yaml → python -m src.training.verl_sdc --config-name=base_matched_grpo_h100_4x4k
  → configs/base_matched_grpo_h100_4x4k.yaml (mode: VANILLA_GRPO)
  → reward manager: correctness_reward만 (rewards.py, REWARD_CONFIGS['VANILLA_GRPO'],
    verl_sdc.py:1232 — 단일 [correctness] GDPO head)
  → advantage: _VANILLA_MODES early-return (verl_sdc.py:1451, 분기 :2580)
    — teacher/PMI/cf_group forward는 절대 안 돈다
```

b0p vs b2p의 차이는 **init 하나뿐**이다. 현행 런처는 그 경로를 실행 시점에 해소해
`/scratch/models/sft2_init`으로 고정 스테이징하며, **두 arm의 SFT2 산출물이 같은 접미사로
모두 존재할 때만** 스테이징한다(한쪽만 있으면 world-size가 어긋난 짝이 되므로 abort).

**Mode 2 — TRIOBJ_DCPO_V4** (풀패키지 b3p = `h100std_rq3v2f_b3p.yaml`)

```
런처 yaml → --config-name=triobj_dcpo_v4_stage3b_h100_4x4k
  → verl_sdc.py:_populate_dcpo_region_keys (:219)
  → dcpo_region.build_dcpo_region_masks + dcpo_region_rewards
    (correctness/format/cal/emit head; 채점 헬퍼는 rewards.py에서 import — byte-동일 채점)
  → rmeta 분기 _rmeta_src == "pmi_shift" (:572)
  → _compute_dcpo_v4_pmi_shift_rmeta (:2341)
  → dcpo_pmi_shift.pmi_shift_reward (frozen ref worker에서 gold-vs-decoy
    log-odds; decoy는 _decoy_utils._rule_based_decoy)
  → verl_sdc_utils._compute_dcpo_region_advantage (:260)
  → dcpo_region.compose_dcpo_region_advantage (region-routed GDPO,
    w_meta 80-step warmup, anchor-EMA 상태 = verl_sdc_utils._ANCHOR_EMA_STATE)
```

## 2. config 상속 사슬 (우선순위 높은 것부터)

1. **런처 CLI 오버라이드** — `h100std_rq3v2f_*.yaml`의 `++`/`key=`
   (v2 레시피: temp 1.0, top_k −1, resp 8192, norm_adv_by_std=false,
   logprob micro_bs 2, save_freq, resume_mode=auto)
2. **네임드 config**: `base_matched_grpo_h100_4x4k.yaml` 또는
   `triobj_dcpo_v4_stage3b_h100_4x4k.yaml`
3. **부모**: 둘 다 `verl_e4_selfdistill_h200_4x4k.yaml`
   ← `verl_sdc_e21r_shared.yaml` 상속
4. **verl 패키지 기본값**: `++hydra.searchpath=[pkg://verl/trainer/config]`

### ⚠️ 신규 인력 함정 (확인됨): rmeta 소스는 yaml이 아니라 런처가 결정

`configs/triobj_dcpo_v4_stage3b_h100_4x4k.yaml:175`는 yaml 기본값으로
`dcpo_rmeta_source: cf_group`을 두지만, 라이브 B3 런처가 이를 **뒤집는다**:
b3p 런처가 `++algorithm.dcpo_rmeta_source=pmi_shift`를 넘긴다(줄번호는 인용하지 않는다 —
하루 만에 썩는다). cf_group
with/without-arm 장치(cf_placebo_agent 등)는 rq3에서 **전부 휴면** —
rollout은 single_turn, 전 arm 매치드. **yaml만 읽으면 보상 소스를 틀리게
안다. 진실은 런처.** (구세대에는 `++algorithm.dcpo_w_meta=0.0`인 b3nopmi arm이 있었다 — 같은 장치, 가중치 0.
현행 3-arm에는 없다.)

## 3. 체크포인트 / 재개 장치

| 부품 | 위치 | 역할 |
|---|---|---|
| RGS 완전성 규칙 | 런처 yaml 인라인 (예: `h100std_rq3v2f_b3p.yaml`) | HF repo `iamseungpil/metacot-h200-triobj-dcpo-v3`에서 model+extra+optim 샤드 ≥4인 최고 `global_step_N` 탐색; fail-closed — RGS 빈/깨짐 → abort, RGS=−1(HF 3회 실패) → abort, 계보 존재하나 pull 결과 없음 → abort(gs0 콜드스타트 거부) |
| `scripts/pull_resume_ckpt.py` | yaml 인라인 호출, 3회 재시도 | 최신 완전 ckpt를 `/scratch/checkpoints/<arm>`으로 pull → `trainer.resume_mode=auto`가 재개 |
| `scripts/push_ckpts_to_hf.py` | nohup 데몬 (`--interval 90 --keep 2`) | 학습 중 신규 global_step 디렉토리를 per-file 내구 push |
| 최종 sync push | verl 종료 후 yaml 인라인 | `pkill push_ckpts_to_hf` 후 동기 `upload_folder` 최대 10회 재시도 + 샤드 수 검증 |

## 4. src/training/*.py

> **LIVE 표시의 의미(0904).** 아래 표의 "LIVE"는 "math-DCPO 경로가 다시 돌 때 필요한
> 파일"이다 — **지금 도는 arm은 Countdown cd7이고, verl_sdc.py를 공유하는 것을 빼면
> 이 표의 나머지는 현재 어느 GPU에서도 실행되지 않는다.** 지금 실제로 도는 파일은
> §9(Countdown cd7 경로)를 볼 것.

| 파일 | 역할 | 상태 |
|---|---|---|
| verl_sdc.py | 메인 hydra 진입점(`-m src.training.verl_sdc`); RayPPOTrainer 래퍼, REWARD_CONFIGS 모드 디스패치, rmeta 라우팅 | **LIVE — 양쪽 경로 공유** (math-DCPO rq3 런처 5개 + Countdown `countdown_6arm`) |
| countdown_rewards.py | Countdown 팔 정체(ARM_SPECS)·항(TERMS) 단일 정의처, `arm_reward` 조립 | **LIVE (Countdown cd7, 현재 도는 경로)** — §9 참조 |
| countdown_selfcontrol.py | SC/SCg 팔의 행 특징(stuck·hi·novel·followed·checked) 계산 — `countdown_rewards`가 조립만, 원재료는 여기 | **LIVE (Countdown cd7)** |
| countdown_task.py | Countdown 채점(`grade`, 완전열거 `_solvable`)·프롬프트 변형(`PROMPT_VARIANTS`: plain/new/p3) | **LIVE (Countdown cd7)** |
| dcpo_region.py | region 마스크(META_REGION/META_CONTENT/CONF/ANSWER), region 보상, advantage 조성 | math-DCPO, 현재 미가동 (TRIOBJ arm 전용) |
| dcpo_pmi_shift.py | pmi_shift R_meta numpy 코어(save/derail 비대칭 보상) | math-DCPO, 현재 미가동 (b3pkg; b3nopmi는 가중치 0) |
| rewards.py | 정준 correctness 채점(math_verify + thread-safe SIGALRM 가드), format/cal 헬퍼 | math-DCPO, 현재 미가동 (수학 arm 전부) — Countdown은 `countdown_task.grade`를 씀 |
| verl_sdc_utils.py | region advantage 계산, 마스크 빌더, anchor-EMA 상태 | math-DCPO, 현재 미가동 (TRIOBJ; import는 항상) |
| sft.py | TRL SFT 트레이너(B0-gold·B23 meta-SFT init 생성) | math-DCPO, 현재 미가동 (SFT 런처 2개 — cd7은 SFT 없이 RL만) |
| _decoy_utils.py | rule-based decoy 생성(pmi_shift용 gold-vs-decoy) | math-DCPO, 현재 미가동 (전이적) |
| meta_close_processor.py | vLLM logits proc — `<\|/meta\|>` 강제 닫기 (b3 런처 env `DCPO_META_CLOSE_FORCE=1`) | 휴면 (env는 b3 런처가 설정하나 유일 소비처 cf_prefix_agent가 rq3 single_turn/pmi_shift 경로에서 미호출 — sdc_counterfactual=false·cf_group 아님) |
| meta_quality.py | meta 품질 점수 헬퍼 (rewards.py가 import) | LIVE-전이적 |
| tokenizer_utils.py / meta_token_init.py / meta_template.py | tokenizer 호환 / think→meta embedding 이식 / SFT용 meta 템플릿 | SFT 계보 LIVE |
| dcpo_pmi.py / dcpo_directional.py / dcpo_asymcf.py | 구세대 R_meta(pmi dense / gm-contrast / asym_cf) | LEGACY (import되나 미선택) |
| cf_placebo_agent.py / cf_groupban_agent.py / cf_prefix_agent.py | cf_group rollout agent | rq3에서 LEGACY 휴면 |
| grpo_v2.py / verl_gdpo*.py / verl_reward.py | 구세대 TRL/verl-GDPO 파이프라인 | LEGACY |
| verl_gdpo_data.py | parquet 빌더(`--mode meta_mix`) — 라이브 train/val parquet의 생산자 | SEMI-LIVE (오프라인 데이터 생산) |
| meta_inject.py 외 (revision_rewards/rlsd_data/redirect_cf/segment_loss_mask/switch_ban) | pre-rq3 단계 산물 | LEGACY |

## 5. src/eval/ — 전부 LEGACY/오프라인 분석 (rq3 런처가 참조 안 함)

eval_hf.py(pre-vLLM), pmi_shift_signal.py(pmi 오프라인 프로브),
decoy_did_pregate.py, eval_counterfactual_difficulty*, eval_passk_headroom.py,
cf_stats.py, redirect_* 등. **rq3 ckpt의 held-out eval은
`scripts/eval_vllm_1030.py`**(SFT 런처 `*_sft_b?p2_rvfull.yaml`에서 참조)이며 src/eval이
아니다.

## 6. scripts/ (그룹만)

| 그룹 | 파일 | 상태 |
|---|---|---|
| **Countdown cd7 로컬 러너 (현재 도는 경로)** | `scripts/local/gpu_queue.py`·`run_arm.sh`·`make_data.sh`·`hf_upload.py`·`disk_guard.py`·`env.sh` | **LIVE** — `scripts/local/README.md` 참조 |
| **Countdown 평가** | `scripts/countdown_gs0_eval.py` — held-out 500×8, `run_arm.sh`가 판정 스텝마다 호출 | **LIVE** |
| math-DCPO rq3 노드 라이프사이클 | bootstrap_sdc_node.sh, gpu_keeper.py, pull_parquets.py, pull_resume_ckpt.py, push_ckpts_to_hf.py | math-DCPO, 현재 미가동 (amlt 클러스터 복구 시) |
| math-DCPO SFT arm | push_models_hf.py, verify_eos_invariant.py, eval_vllm_1030.py | math-DCPO, 현재 미가동 |
| rq3 사이드 eval/smoke | run_rq3_side_eval.py 외 | SEMI-LIVE (로컬용, 런처 미참조) |
| 구세대 launch/데이터/분석 | launch_*, build_*, analyze_*, s3b_retry_daemon.sh 등 | LEGACY |
| env/지원 | check_runtime_env.py, patch_math_verify.py, install_verl.sh, setup_node.sh 등 | 지원 (일부 bootstrap이 호출) |
| smoke/테스트 | smoke_*.py, test_*.py, format_parser_harness.py 등 | 개발용 |
| 아카이브(2026-09-04) | `archive/dead_code_2026_09_04/scripts/` — 임포터 0건 확인된 24개 + retired/ | 이동됨, 코드 무수정 |

## 7. configs/ 와 루트 런처

| 파일 | 역할 | 상태 |
|---|---|---|
| base_matched_grpo_h100_4x4k.yaml | VANILLA_GRPO meta-제거 twin (B0/B2) | **LIVE** |
| triobj_dcpo_v4_stage3b_h100_4x4k.yaml | TRIOBJ_DCPO_V4 풀패키지 (B3) — §2 rmeta 함정 주의 | **LIVE** |
| verl_e4_selfdistill_h200_4x4k.yaml / verl_sdc_e21r_shared.yaml | 부모/조부모 base config | LIVE-as-parent |
| sft_b0_gold.yaml / sft_b23_unmasked.yaml (+accelerate_sft, ds_zero3*) | rq3 SFT init config | **LIVE** |
| sft_v8_*, accelerate_grpo.yaml, archive/(30+) | 구세대 | LEGACY |
| mainline_contract.yaml, CTSD_NODE_INDEX.md | 계약/색인 문서 | Meta |

루트 런처(실제 진입 표면, 0727 현재). **순서가 있다** — SFT2 쌍이 HF에 착지해야 RL이
init을 스테이징한다:

| 단계 | A100 판 | H100 판 |
|---|---|---|
| SFT2 컨트롤 | (a100 판 은퇴 → `archive/launchers_retired_0803/`) | `h100std_sft_b0p2_rvfull.yaml` |
| SFT2 메타 | (a100 판 은퇴 → `archive/launchers_retired_0803/`) | `h100std_sft_b2p2_rvfull.yaml` |
| RL 3-arm | (a100 판 은퇴 → `archive/launchers_retired_0803/`, 제출 금지) | `h100std_rq3v2f_{b0p,b2p,b3p}.yaml` |

A100/H100 판은 target·sku·tier만 다르고 내용은 같다. 어느 쪽이 실제로 제출 가능한지는
CLAUDE.md의 Compute 절을 볼 것 — 두 VC 모두 현재 제약이 있다.

그 밖에: `archive/launchers_retired_0803/a100_rq3v2_b3p.yaml`은 **구 init**(`b2p2_rvseg_sft`)을
쓰는 RQ2 부록 전용이며 복제 결과로 보고하면 안 된다.
`archive/launchers_retired_0803/h100std_env_builder.yaml`은 conda env 빌더(실험 아님) —
0803에 아카이브했고, `scripts/bootstrap_sdc_node.sh`가 매 노드 부트스트랩마다 받아가는
env 팩(`env_snapshots/simplerl_v4.tar.gz`)의 **유일한 재생성 레시피**다.

⛔ **0803 은퇴 런처 경고**: `archive/launchers_retired_0803/a100_rq3v2f_{b0p,b2p,b3p}.yaml`은
살아있는 arm과 durable 출력 경로가 바이트 동일이라 제출하면 그 arm의 체크포인트를 지운다.
구세대 런처 전량은 `archive/launchers_retired_0727/`와 `archive/launchers_pre_rq3/`.

## 8. experiments/ — §4 인과 프로브 워크스트림 (rq3 아님)

probes/(a1…e5 대조 스티어링·inject-causal 등), analysis/(paired-eval 집계),
common/, launch/run.sh + configs/{infra,science} — 전부 sec4 논문용 LEGACY.
단 `experiments/configs/science/eval_1030.yaml` + `launch/run.sh eval`은
held-out eval 스테이징에 재사용되며, models 블록이 pre-rq3 arm을 가리키고
있어 rq3 ckpt eval 시 그 블록 수정이 필요하다(LOCAL_RUN.md 참조).

## 9. Countdown cd7 경로 (0904, 현재 실제로 도는 것)

§1~§8은 math-DCPO(instruct PMI-shift → base 복제) 경로다. 그 경로는 지금 어느 GPU에서도
돌지 않는다. **지금 도는 것**은 로컬 H100×4(GPUs 0-3)에서 실행되는 Countdown 자기제어
RL이며, 호출 사슬은 `ARCHITECTURE.md` (c)에 있다. 요약:

```
scripts/local/gpu_queue.py (큐 워커) → scripts/local/run_arm.sh ARM SEED STEPS VARIANT
  → python -m src.training.verl_sdc --config-name=countdown_6arm ++algorithm.countdown_arm=<ARM>
  → src/training/countdown_rewards.py (ARM_SPECS/TERMS 정의처, arm_reward 조립)
      ← src/training/countdown_selfcontrol.py (SC/SCg 행 특징)
      ← src/training/countdown_task.py (채점 grade, 프롬프트 PROMPT_VARIANTS)
  → verl.model_merger (판정 스텝 병합) → scripts/countdown_gs0_eval.py → scripts/local/hf_upload.py
```

이 경로의 config는 `configs/countdown_6arm.yaml` 하나이고, 런처 yaml이 아니라
`run_arm.sh`가 CLI로 `++algorithm.countdown_arm`을 넘겨 팔을 고른다(math-DCPO의 "런처가
config를 뒤집는다"는 §2 함정과 같은 종류의 함정 — **진실은 run_arm.sh**). 현재 팔 정의와
판정 기준은 `docs/PREREGISTRATION_countdown_sc_round.md`.
