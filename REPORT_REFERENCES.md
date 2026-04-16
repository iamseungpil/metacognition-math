# Report References — 어디에 뭐가 있는지

Last updated: 2026-04-16

보고서 `results/study_2026_04_16_metacot_v8_final_report.md` 에서 인용하는 모든 데이터/코드가 어디에 있는지 정리한 크로스 레퍼런스.

## Quick lookup: 보고서 → 파일 → 경로

### 1. Accuracy 수치 (Section 3.1, Executive Summary)

| 보고서 수치 | 소스 JSON | git 경로 | HF 경로 |
|---|---|---|---|
| Base 4k: 75.92% | step300 eval | `results/step300_deep_analysis/base_step300.json` | `iamseungpil/metacot:results/step300_deep_analysis/base_step300.json` |
| Base 16k: 77.0% | 오늘 재평가 | `results/eval_1030_base_grpo_step300_16k/base_grpo_step300_16k.json` | `...:results/eval_1030_base_grpo_step300_16k/base_grpo_step300_16k.json` |
| Meta 4k: 79.81% | step300 eval | `results/step300_deep_analysis/e21r_v2_step300.json` | `...:results/step300_deep_analysis/e21r_v2_step300.json` |
| Meta 16k: 81.65% | 오늘 재평가 | `results/eval_1030_meta_grpo_e21r_v2_step300_16k/meta_grpo_e21r_v2_step300_16k.json` | `...:results/eval_1030_meta_grpo_e21r_v2_step300_16k/meta_grpo_e21r_v2_step300_16k.json` |

### 2. Entropy 4-way 비교 (Section 3.2)

| 보고서 항목 | n | Δ (nats) | git 경로 | HF 경로 |
|---|---|---|---|---|
| SFT meta (strict) | 120 | +0.300 | `results/entropy_strict_meta/entropy_stats.json` | 동일 경로 ✓ |
| Step300 SFT meta | 208 | −0.113 | `results/entropy_analysis_step300/sft_meta/entropy_stats.json` | 동일 경로 ✓ |
| Step300 RL conf @ 4k | 200 | −0.052 | `results/entropy_analysis_step300/rl_meta_confidence/conf_entropy_stats.json` | 동일 경로 ✓ |
| 16k Meta RL conf | 200 | −0.031 | `results/entropy_meta_grpo_e21r_v2_step300_16k_conf/entropy_stats.json` | 동일 경로 ✓ |

토큰 단위 CSV도 각 디렉토리에 `entropy_per_block.csv`, `entropy_per_sample.csv` 로 같이 있음.

### 3. AIME failure analysis (Section 3.3)

- 보고서 수치: Meta 13 decohered + 12 no-boxed + 1 coherent (of 26 wrong), Base 1+0+18 (of 19 wrong)
- git 경로: `results/aime_failure_analysis_16k/aime_failure_modes.json`
- HF 경로: 동일 ✓

### 4. Marker prevalence (Section 3.4)

- 보고서 수치: has_verify 95.3% / +77.2pp, has_redirect 13.1% / −57.7pp, has_diagnosis 8.1% / −59.3pp, has_epistemic 20.4% / −46.4pp
- git 경로: `results/step300_deep_analysis/meta_behavior.json` (e21r_meta_step300 키 아래)
- HF 경로: 동일 ✓

## Code referenced in report

### Eval 실행 코드 (16k 재평가에 사용됨)

| 보고서 언급 | git 경로 | HF 경로 | 용도 |
|---|---|---|---|
| vLLM 1030 eval | `scripts/eval_vllm_1030.py` | `...:scripts/eval_vllm_1030.py` | max_tokens 가변 1030 문제 eval (base/meta 16k 재평가) |
| Entropy 분석 | `scripts/analyze_entropy_meta.py` | `...:scripts/analyze_entropy_meta.py` | `--marker_mode {meta,confidence}` 토큰 전후 entropy 측정 |
| 파싱 helper | `src/metacot/prompt.py` (`parse_meta_blocks`) | `...:src/metacot/prompt.py` | `<\|meta\|>` + `confidence:` fallback 파싱 |
| Correctness 체크 | `src/training/rewards.py` (`_check_correctness`, `_extract_answer_fallback`) | `...:src/training/rewards.py` | boxed answer 추출 + 정답 비교 |

### Training 코드 (실험 재현에 필요)

| 보고서 언급 | git 경로 | HF 경로 |
|---|---|---|
| SFT 학습 | `src/training/sft.py` | `...:src/training/sft.py` |
| GRPO (E1-E7 reward modes) | `src/training/grpo_v2.py` | `...:src/training/grpo_v2.py` |
| VERL 2-head GDPO (E21R-v2) | `src/training/verl_gdpo.py`, `src/training/verl_reward.py` | 동일 ✓ |
| 7개 reward 함수 | `src/training/rewards.py` | 동일 ✓ |
| **NEW** no-boxed penalty (Phase 6) | `src/training/rewards.py` (`compute_no_boxed_penalty`, line ~2174) | 동일 ✓ |
| Reward 테스트 (TC25a-j) | `tests/test_rewards.py` | 동일 ✓ |

### 실험 launcher (checkpoint 재현)

| 실험 | launcher | git 경로 |
|---|---|---|
| V8 strict paired SFT | `scripts/launch_v8_meta_inside_strict_remote.sh`, `launch_v8_base_matched_strict_remote.sh` | `scripts/launch_v8_*_strict_remote.sh` |
| E21 RL | `scripts/launch_e21_vs_base_matched_0410.sh` | `scripts/launch_e21_vs_base_matched_0410.sh` |
| Base GRPO 16k 재평가 bootstrap (2026-04-16) | `/tmp/run_grpo_eval_base_16k.sh` (ephemeral) | — (ephemeral, bootstrapped on node) |

## Data referenced

### SFT 학습 데이터 (V8 strict)

| 데이터 | git | HF |
|---|---|---|
| `data/v8_meta_inside_strict.parquet` | gitignored (local) | `...:data/v8_meta_inside_strict.parquet` ✓ |
| `data/v8_base_matched_strict.parquet` | gitignored (local) | `...:data/v8_base_matched_strict.parquet` ✓ |

### RL 학습 데이터

| 데이터 | HF 경로 |
|---|---|
| verl_train_redirect / verl_val_redirect (meta) | `iamseungpil/metacot:data/verl_train_redirect.parquet`, `...:data/verl_val_redirect.parquet` |
| verl_train_redirect_base / verl_val_redirect_base | 동일 `data/` |

### Control/RAG

| 데이터 | git (canonical path 2026-04-16 이후) | HF |
|---|---|---|
| RAG seed library | `data/control_rag_seed_library.json` | `...:data/control_rag_seed_library.json` ✓ |

### Benchmarks (외부 HF)

- GSM8K: HF `openai/gsm8k` main/test (500 문제 추출)
- MATH500: HF `HuggingFaceH4/MATH-500` (500 전부)
- AIME-24+AIME-25: HF `HuggingFaceH4/aime_2024` + `HuggingFaceH4/aime_2025` (총 30 문제)

## Checkpoints referenced

### SFT checkpoints (HF 상)

| Checkpoint | HF 경로 |
|---|---|
| v8 meta strict SFT | `iamseungpil/metacot:models/v8_meta_inside_strict_sft/` (merged safetensors, 16.4 GB) |
| v8 base matched strict SFT | `...:models/v8_base_matched_strict_sft/` (sharded) |

### RL checkpoints (FSDP shards on HF)

| Checkpoint | HF 경로 | steps |
|---|---|---|
| E21R-v2 meta GRPO | `...:checkpoints/verl_e21r_v2_0413/` | 190, 220, 250, latest=300 |
| Base matched GRPO | `...:checkpoints/verl_base_matched_0410/` | 50, 70, 150, 200, 240, latest=300 |

FSDP 복구는 `python3 -m verl.model_merger merge --backend fsdp --local_dir <ckpt_path>/latest/actor --target_dir <output>` 로.

## 보고서 자체

| 보고서 | git 경로 | HF 경로 |
|---|---|---|
| **Final V8 report (이 세션)** | `results/study_2026_04_16_metacot_v8_final_report.md` | 동일 ✓ |
| V8 experiment report | `results/metacot_v8_experiment_report.md` | 동일 ✓ |
| V8 active plan (Phase 6 추가됨) | `results/plan_metacot_v8_active_2026_04_09.md` | 동일 ✓ |
| 2026-04-16 status | `results/study_2026_04_16_metacot_v8_status_report.md` | 동일 ✓ |
| Cleanup audit | `results/cleanup_audit_2026_04_16.md` | 동일 ✓ |
| **이 파일 (REPORT_REFERENCES)** | `REPORT_REFERENCES.md` | 동일 ✓ |

## 네비게이션 계층

```
ANALYSIS_MAP.md            ← 프로젝트 전체 맵
  └→ REPORT_REFERENCES.md  ← 이 파일: 보고서 수치→파일 대응
  └→ scripts/ANALYSIS_INDEX.md  ← 코드 인벤토리
  └→ results/README.md     ← 결과 디렉토리 인벤토리
  └→ docs/mainline_registry_2026_04_13.md  ← canonical registry
```
