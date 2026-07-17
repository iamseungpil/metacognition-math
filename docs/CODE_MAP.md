# CODE_MAP — 신규 인력용 코드 인벤토리 (read-only, 2026-07-17)

이 문서는 "지금 살아있는 코드가 무엇이고 어디서 불리는가"의 지도다. 수정
지침이 아니다 — src/·configs/·h100std_*.yaml은 tarball(CODE_TAR_REVISION)과
byte-동일 유지가 원칙이다.

## 1. 라이브 트레이너 모드 2개와 호출 사슬

**Mode 1 — VANILLA_GRPO** (B0 = `h100std_rq3_b0.yaml`, B2 = `h100std_rq3_b2.yaml`)

```
런처 yaml → python -m src.training.verl_sdc --config-name=base_matched_grpo_h100_4x4k
  → configs/base_matched_grpo_h100_4x4k.yaml (mode: VANILLA_GRPO)
  → reward manager: correctness_reward만 (rewards.py, REWARD_CONFIGS['VANILLA_GRPO'],
    verl_sdc.py:1232 — 단일 [correctness] GDPO head)
  → advantage: _VANILLA_MODES early-return (verl_sdc.py:1451, 분기 :2580)
    — teacher/PMI/cf_group forward는 절대 안 돈다
```

B0 vs B2의 차이는 **`actor_rollout_ref.model.path` 하나뿐**
(b0_gold_sft vs b23_rv_unmasked_sft).

**Mode 2 — TRIOBJ_DCPO_V4** (B3pkg = `h100std_rq3_b3.yaml`, B3-noPMI = `h100std_rq3_b3nopmi.yaml`)

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

1. **런처 CLI 오버라이드** — `h100std_rq3_*.yaml`의 `++`/`key=`
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
`h100std_rq3_b3.yaml:191`·`h100std_rq3_b3nopmi.yaml:169`이
`++algorithm.dcpo_rmeta_source=pmi_shift`를 넘긴다. cf_group
with/without-arm 장치(cf_placebo_agent 등)는 rq3에서 **전부 휴면** —
rollout은 single_turn, 전 arm 매치드. **yaml만 읽으면 보상 소스를 틀리게
안다. 진실은 런처.** (b3nopmi는 추가로 `++algorithm.dcpo_w_meta=0.0` —
같은 장치, 가중치 0.)

## 3. 체크포인트 / 재개 장치

| 부품 | 위치 | 역할 |
|---|---|---|
| RGS 완전성 규칙 | 런처 yaml 인라인 (예: `h100std_rq3_b3.yaml:120-166`) | HF repo `iamseungpil/metacot-h200-triobj-dcpo-v3`에서 model+extra+optim 샤드 ≥4인 최고 `global_step_N` 탐색; fail-closed — RGS 빈/깨짐 → abort, RGS=−1(HF 3회 실패) → abort, 계보 존재하나 pull 결과 없음 → abort(gs0 콜드스타트 거부) |
| `scripts/pull_resume_ckpt.py` | yaml 인라인 호출, 3회 재시도 | 최신 완전 ckpt를 `/scratch/checkpoints/<arm>`으로 pull → `trainer.resume_mode=auto`가 재개 |
| `scripts/push_ckpts_to_hf.py` | nohup 데몬 (`--interval 90 --keep 2`) | 학습 중 신규 global_step 디렉토리를 per-file 내구 push |
| 최종 sync push | verl 종료 후 yaml 인라인 | `pkill push_ckpts_to_hf` 후 동기 `upload_folder` 최대 10회 재시도 + 샤드 수 검증 |

## 4. src/training/*.py

| 파일 | 역할 | 상태 |
|---|---|---|
| verl_sdc.py | 메인 hydra 진입점(`-m src.training.verl_sdc`); RayPPOTrainer 래퍼, REWARD_CONFIGS 모드 디스패치, rmeta 라우팅 | **LIVE** (rq3 런처 5개 전부) |
| dcpo_region.py | region 마스크(META_REGION/META_CONTENT/CONF/ANSWER), region 보상, advantage 조성 | **LIVE** (TRIOBJ arm) |
| dcpo_pmi_shift.py | pmi_shift R_meta numpy 코어(save/derail 비대칭 보상) | **LIVE** (b3pkg; b3nopmi는 가중치 0) |
| rewards.py | 정준 correctness 채점(math_verify + thread-safe SIGALRM 가드), format/cal 헬퍼 | **LIVE** (전 arm) |
| verl_sdc_utils.py | region advantage 계산, 마스크 빌더, anchor-EMA 상태 | **LIVE** (TRIOBJ; import는 항상) |
| sft.py | TRL SFT 트레이너(B0-gold·B23 meta-SFT init 생성) | **LIVE** (SFT 런처 2개) |
| _decoy_utils.py | rule-based decoy 생성(pmi_shift용 gold-vs-decoy) | **LIVE** (전이적) |
| meta_close_processor.py | vLLM logits proc — `<\|/meta\|>` 강제 닫기 (b3 런처 env `DCPO_META_CLOSE_FORCE=1`) | **LIVE** (b3) |
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
`scripts/eval_vllm_1030.py`**(h100std_sft_*.yaml에서 참조)이며 src/eval이
아니다.

## 6. scripts/ (75개 — 그룹만)

| 그룹 | 파일 | 상태 |
|---|---|---|
| rq3 노드 라이프사이클 | bootstrap_sdc_node.sh, gpu_keeper.py, pull_parquets.py, pull_resume_ckpt.py, push_ckpts_to_hf.py | **LIVE** (전 rq3 yaml) |
| SFT arm | push_models_hf.py, verify_eos_invariant.py, eval_vllm_1030.py | **LIVE** (SFT 런처) |
| rq3 사이드 eval/smoke | run_rq3_side_eval.py 외 | SEMI-LIVE (로컬용, 런처 미참조) |
| 구세대 launch/데이터/분석 | launch_*, build_*, analyze_*, s3b_retry_daemon.sh 등 | LEGACY |
| env/지원 | check_runtime_env.py, patch_math_verify.py, install_verl.sh, setup_node.sh 등 | 지원 (일부 bootstrap이 호출) |
| smoke/테스트 | smoke_*.py, test_*.py, format_parser_harness.py 등 | 개발용 |

## 7. configs/ 와 루트 런처

| 파일 | 역할 | 상태 |
|---|---|---|
| base_matched_grpo_h100_4x4k.yaml | VANILLA_GRPO meta-제거 twin (B0/B2) | **LIVE** |
| triobj_dcpo_v4_stage3b_h100_4x4k.yaml | TRIOBJ_DCPO_V4 풀패키지 (B3) — §2 rmeta 함정 주의 | **LIVE** |
| verl_e4_selfdistill_h200_4x4k.yaml / verl_sdc_e21r_shared.yaml | 부모/조부모 base config | LIVE-as-parent |
| sft_b0_gold.yaml / sft_b23_unmasked.yaml (+accelerate_sft, ds_zero3*) | rq3 SFT init config | **LIVE** |
| sft_v8_*, accelerate_grpo.yaml, archive/(30+) | 구세대 | LEGACY |
| mainline_contract.yaml, CTSD_NODE_INDEX.md | 계약/색인 문서 | Meta |

루트 런처(실제 rq3 진입 표면): `h100std_rq3_b0.yaml`, `h100std_rq3_b2.yaml`,
`h100std_rq3_b3.yaml`(b3pkg), `h100std_rq3_b3nopmi.yaml`,
`h100std_rq3_b3_dbg.yaml`(디버그), `h100std_sft_b0_gold.yaml`,
`h100std_sft_b23_unmasked.yaml`, `h100std_env_builder.yaml`(conda env 빌더).
`h100std_rq3_b1.yaml`은 아직 없다(B1 arm은 EXPERIMENT_PLAN E1의 신설 예정).

## 8. experiments/ — §4 인과 프로브 워크스트림 (rq3 아님)

probes/(a1…e5 대조 스티어링·inject-causal 등), analysis/(paired-eval 집계),
common/, launch/run.sh + configs/{infra,science} — 전부 sec4 논문용 LEGACY.
단 `experiments/configs/science/eval_1030.yaml` + `launch/run.sh eval`은
held-out eval 스테이징에 재사용되며, models 블록이 pre-rq3 arm을 가리키고
있어 rq3 ckpt eval 시 그 블록 수정이 필요하다(LOCAL_RUN.md 참조).
