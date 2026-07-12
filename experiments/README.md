# experiments/ — 메타인지적 자기증류 실험 가이드

처음 온 협업자가 30분 안에 이 프로젝트의 질문·지표·실행법을 파악하고 실험을
재현할 수 있게 쓴 문서다. 코드 진입점과 경로는 전부 실제 리포 기준이다.

---

## 1. 연구 프레이밍 — 메타인지적 자기증류로 OOD에 강건한 수학 추론을 학습한다

이 프로젝트는 Qwen3-8B가 풀이 도중 `<|meta|>...<|/meta|>` 블록으로 메타인지를
외재화하도록 학습한다. 핵심 학습 신호인 **PMI-shift 보상**은 외부 교사가 아니라
모델 자신의 신호를 쓴다. 메타 블록 앞뒤에서 gold 답과 decoy 답의 로그확률 차이가
어떻게 움직이는지를 측정해서, decoy 쪽으로 기울던 확률이 메타 블록을 지나며 gold
쪽으로 돌아오면(SAVE) 보상하고 반대로 gold에서 decoy로 이탈하면(DERAIL) 벌한다.
즉 모델이 이미 갖고 있는 gold/decoy 판별 신호를 메타 구간에 증류하는
**자기증류(self-distillation)** 다. 구현은 `src/training/dcpo_pmi_shift.py`.

가설은 이렇다: in-distribution 정답을 암기하는 것보다 메타인지 행동(막힘 감지,
가정 점검, 접근 전환, 검산)을 학습하는 쪽이 **분포 밖(OOD) 어려운 문제 —
특히 AIME — 에서 더 강건하게 일반화**한다. 그래서 최종 판정 지표는 정확도이고,
calibration(ECE/Brier/과신율)은 항상 함께 측정하되 보조 지표다.

두 arm은 메타 메커니즘만 빼고 byte-identical 파이프라인이다:

```
meta arm:  Qwen3-8B → SFT-1 (v8_meta_inside_strict) → SFT-2 (rv_functional)
                    → RL (TRIOBJ_DCPO_V4 + pmi_shift)        h100std_pmishift.yaml
base arm:  Qwen3-8B → 같은 데이터에서 meta만 제거한 SFT
                    → RL (VANILLA_GRPO, correctness-only)    h100std_base_matched_rl.yaml
```

RL 하이퍼파라미터(lr 1e-6, train_batch 64, n 8, clip, max_response 4096, 300
step)는 두 arm이 동일하고, 다른 것은 메타 보상 헤드의 유무뿐이다. yaml의
description 필드에 이 매칭 조건이 명문화되어 있다.

## 2. 연구 질문 → 테이블 → 실험 매핑

**(주의) 아래 RQ1–4는 pre-rq3 세대(instruct 기반) 넘버링이다.** 현행 실험은
**rq3 매치드 래더**(진짜 Qwen3-8B-Base, 런처 `h100std_rq3_b0/b2/b3.yaml`)이고,
그 안의 과학 질문은 **RQ1 = B2−B0(메타 SFT 효과), RQ2 = B3−B2(pmi_shift 메타
보상 효과)** 다 — `docs/redesign/base_rl_recipe.md`·`docs/CONSTITUTION.md` 참조.
"RQ3"라는 표기가 (a) 아래 표의 난이도 층화 질문, (b) 현행 실험명(래더),
(c) 구세대 docs의 "RQ3 mainline" 세 가지로 충돌해 왔으니 문서를 읽을 때 세대를
먼저 확인할 것.

연구 질문은 4개로 압축되어 있다 (구 6-RQ 구조에서: 구 RQ2 SFT 능력비용과
REF-0은 별도 실험이 아니라 T1의 참조행으로 흡수, 구 RQ3+RQ4는 새 RQ2로 통합,
구 RQ5 시드·토큰통제는 T1의 프로토콜에 내장, 구 RQ6은 순수 calibration만 남겨
새 RQ4가 되고 OOD 부분은 새 RQ3으로 이동).

| RQ | 질문 | 테이블 | 실험 (진입점) | 상태 |
|---|---|---|---|---|
| RQ1 | PMI-shift가 실제로 정확도를 올리는가? | T1 (메인): matched-base gs300 vs pmishift gs300, held-out 1030 + 9도메인, 16k tokens, avg@8(AIME avg@16). 시드 ×3(mean±std)과 응답 토큰 통제열은 별도 RQ가 아니라 **T1 프로토콜에 내장**. 참조행: raw Qwen3-8B, SFT-only — SFT 능력비용이 표 안에서 읽힌다 | RL(pre-rq3): `h100std_pmishift.yaml`, `h100std_base_matched_rl.yaml` / RL(현행 rq3 래더): `h100std_rq3_b0.yaml`, `h100std_rq3_b2.yaml`, `h100std_rq3_b3.yaml` / eval: `h100std_pmishift_1030_eval.yaml`, `h100std_base_matched_1030_eval.yaml`, `h100std_pmishift_heldout_eval.yaml` / 참조행: `experiments/configs/science/rl_ref0_nosft.yaml` 계열 | **진행 중** — base arm gs300 확보 단계 (Stage 0) |
| RQ2 | 효과는 무엇이며 어디서 오는가? | T2 (분해·메커니즘): Gandhi-arm(meta-SFT + VANILLA_GRPO)으로 SFT-프라이밍 vs RL-보상 기여 분해 + flip 분석(SAVE/DERAIL) + placebo(셔플 메타) | Gandhi: `experiments/configs/science/rl_gandhi_arm.yaml` (Stage 2 발사) / 분석: `experiments/analysis/flip.py`, `experiments/analysis/placebo.py`, rollout dump: `h100std_rollout_dump.yaml` | 분석 트랙 (GPU 불필요 부분 있음) / Gandhi arm 대기 |
| RQ3 | 난이도·문제 유형·OOD에 따라 메타 효과가 어떻게 달라지는가? | T3 (층화): 난이도 사분위 층화 정확도+방출률(Simpson 함정: Q1 easy 방출 10%, Q3 mid-hard 메타 0.83 vs base 0.67 — PRELIMINARY), 도메인별, AIME 등 어려운 도메인의 OOD 강건성 | `experiments/analysis/stratify.py` (T1 eval 산출물 입력) | T1 산출물 대기 |
| RQ4 | Calibration은 어떻게 되는가? | T4: arm별 ECE(15-bin)/Brier/과신율 — 순수 calibration만, OOD 강건성은 RQ3 소관. 주-지표는 정확도, calibration은 보조라는 위계 유지 | `experiments/analysis/calibration.py` (T1 eval 산출물 입력) | T1 산출물 대기 |

표 생성 매핑: `flip.py`+`placebo.py`→T2, `stratify.py`→T3, `calibration.py`→T4,
그리고 `experiments/analysis/aggregate_tables.py`가 T1–T4 네 표를 하나의
markdown으로 전부 생성한다.

단계 게이트: **Stage 0**(지금) = base gs300 확보 + 통합 eval 준비 → **Stage 1** =
T1 held-out 판정 → **Stage 2** = WIN이면 T1 참조행(REF-0/raw/SFT-only 평가)·
Gandhi-arm(RQ2)·T1 내장 프로토콜(시드 ×3, 토큰 통제) 병렬, LOSE면 T2 메커니즘
분석으로 피벗. **Stage 3**(SFT v2 파일럿, 아래 트랙 C)은 게이트와 독립적으로
시작할 수 있다.

## 3. 지표 규약 (전부 실제로 밟았던 함정에서 나온 규칙이다)

1. **학습 중 정확도는 wandb의 `val-aux/<ds>/correctness/mean@1`만 본다.**
   `val-core/*`와 `reward`는 메타 shaping이 섞인 합성 지표라서 base-vs-meta를
   비교하면 가짜 격차가 생긴다. 실제로 한 번 속았던 함정이다. wandb 프로젝트는
   `gistdslab/metacot-dcpo-v4`.
2. **최종 판정은 held-out 1030문제**(GSM8K 500 + MATH-500 500 + AIME 30)이고,
   채점은 **math_verify**로 한다. `src/eval/eval_hf.py`의 `check_correctness`는
   MATH에서 정답을 약 26%p 잘못 깎는 버그가 문서화되어 있으므로 최종 숫자에
   쓰면 안 된다. math_verify는 워커 스레드에서 hang하는 문제가 있어서
   `scripts/patch_math_verify.py`를 먼저 적용한다.
3. **논문 eval 프로토콜**: max_tokens 16k, avg@8 (AIME는 avg@16), temperature
   0.7, 그리고 **두 arm을 같은 eval job에서 같은 seed로** 돌린다. 하니스는
   `scripts/eval_vllm_1030.py` (`--max_tokens 16384 --temperature 0.7
   --num_samples 8 --seed 42`).
4. **난이도 층화 정확도를 반드시 같이 보고한다.** 과거에 meta-vs-nometa 집계
   비교가 문제 선택 편향으로 뒤집힌 Simpson 함정이 있었다 (mid-hard Q3에서는
   meta 0.83 vs 0.67로 이기는데 집계는 반대). 집계 숫자 하나만 보고 판정하지
   않는다.
5. **메타 방출(emission)은 닫힌 블록만 센다** — `<|meta|>...<|/meta|>`가 모두
   있는 블록. 열리고 안 닫힌 블록을 세면 방출률이 부풀려진다. 파서는
   `src/metacot/prompt.py`의 `parse_meta_blocks`.

## 4. 폴더 구조 — science와 infra의 분리

과학적 질문에 답하는 코드(재현 가능해야 하고, 가능하면 GPU 없이 돌아야 함)와
클러스터를 굴리는 코드(preemption 복구, ckpt 릴레이)를 분리한다.

```
experiments/            science: 프로브·분석 (로컬 실행, 판정을 JSON으로 남김)
├── common/             공용 유틸 — load_secrets.sh(.env 로드), env.py, grading.py, vllm_gen.py
└── probes/             단발 프로브 (a1~e5: contrastive control, entropy, inject-causal, ...)

src/                    학습·평가 소스
├── training/           sft.py, verl_sdc.py(TRIOBJ_DCPO 트레이너), dcpo_pmi_shift.py(PMI-shift 보상)
└── eval/               eval_hf.py(로더), pmi_shift_signal.py, eval_counterfactual_difficulty.py

configs/                학습 설정 (sft_*.yaml, triobj_dcpo_v4_*.yaml, base_matched_grpo_*.yaml)
scripts/                infra: 노드 오케스트레이션·ckpt 릴레이·분석 스크립트 (scripts/README.md 참조)
h100std_*.yaml (루트)   amlt 제출 yaml — 실험 하나 = yaml 하나, description에 의도 명시
results/                eval 산출물 (HF dataset iamseungpil/metacot 의 results/에 미러)
docs/                   계획·리포트·사이트 소스 (docs/site/)
```

**ADDITIVE ONLY 원칙**: 진행 중인 런은 이 코드의 tarball에서 resume하므로, 기존
학습 코드·설정은 수정/삭제하지 않고 새 파일을 추가한다.

## 5. 실행법

### 준비: .env

```bash
cp .env.example .env          # HF_TOKEN, GH_TOKEN, WANDB_API_KEY 채우기
source experiments/common/load_secrets.sh   # 로드 + placeholder 검사
```

### 예시 1 — SFT (로컬 4-GPU)

```bash
accelerate launch --config_file configs/accelerate_grpo.yaml \
    src/training/sft.py --config configs/sft_rv_functional.yaml
```

SFT-2(rv_functional) 예시다. SFT-1은 `configs/phase1_*` 계열과
`scripts/launch_v8_meta_inside_strict_remote.sh`, base arm 쌍은
`configs/sft_base_rv.yaml`을 쓴다. 데이터 parquet은 파일별 HF repo에서
`hf_hub_download`로 `<repo>/data/`에 스테이징한다:

- `iamseungpil/metacot` — `data/v8_meta_inside_strict.parquet`,
  `data/v8_base_matched_strict.parquet`, `data/v8_base_rv_sft.parquet` (SFT),
  `results/*` (eval 산출물)
- `iamseungpil/metacot-rv` — `data/rv_redirect_verify_functional.parquet`
- `iamseungpil/metacot-sdc-data` — `verl_*_meta_mix.parquet` 등 verl RL
  parquet 6종 (**repo 루트**, `data/` prefix 없음)

`scripts/pull_parquets.py`는 이 중 verl RL parquet 6종만, MSR 노드의
`/scratch/metacognition/data`로 받는 스크립트다 — SFT parquet 스테이징에는
쓰지 않는다.

### 예시 2 — held-out 1030 eval (로컬, vLLM)

```bash
python scripts/eval_vllm_1030.py \
    --model_path <merged_ckpt_dir> \
    --model_name pmishift_gs300_16k \
    --output_dir results/eval_1030_pmishift_gs300/ \
    --max_tokens 16384 --temperature 0.7 --num_samples 8 --seed 42
```

논문 숫자를 만들 때는 3절의 규약대로: 두 arm을 같은 job에서, AIME는
`--num_samples 16`, 채점은 math_verify로 재채점.

### 예시 3 — RL (MSR 클러스터, amlt)

```bash
set -a; source .env; set +a          # yaml의 ${HF_TOKEN} 등이 로컬 env에서 치환됨
amlt run h100std_pmishift.yaml pmishift-<날짜> -d "meta arm RL"
amlt run h100std_base_matched_rl.yaml base-matched-<날짜> -d "base arm RL"
```

클러스터 사실관계: H100×4 Standard, preemption 윈도우 약 6시간. 그래서 ckpt는
HF model repo `iamseungpil/metacot-h200-triobj-dcpo-v3`를 경유해 릴레이한다 —
job 안에서 `scripts/push_ckpts_to_hf.py`(파일 단위 내구 푸셔, dedup)가 데몬으로
돌고, 재시작 시 `scripts/pull_resume_ckpt.py`가 최신 `global_step_N`을 받아
verl `resume_mode=auto`가 이어 달리게 한다. 이 배선은 전부 제출 yaml 안에
들어 있으므로 협업자는 yaml만 제출하면 된다.

## 6. 협업자 트랙 온보딩

**트랙 A — 클러스터 학습.** amlt 접근 권한 필요. 5절 예시 3으로 RL/eval yaml을
제출하고, wandb `gistdslab/metacot-dcpo-v4`에서 `val-aux/*/correctness/mean@1`만
본다(규약 1). 새 실험은 기존 yaml을 복제해 description·WANDB_NAME을 바꾼 새
파일로 만든다(ADDITIVE ONLY). 실험 구조(설계·arm·지표)는 발사 전에 공유하고
승인받는 것이 팀 규칙이다.

**트랙 B — 분석 (GPU 불필요).** 입력은 HF dataset `iamseungpil/metacot`의
rollout/eval parquet(`results/*`)이다. `hf_hub_download(repo_id="iamseungpil/metacot",
repo_type="dataset", filename="results/...")`로 받는다
(`scripts/pull_parquets.py`는 verl RL parquet 6종 전용이라 eval parquet을
받지 못한다). rollout dump가 더
필요하면 트랙 A에 `h100std_rollout_dump.yaml` 제출을 요청한다. **T2/T3/T4
분석이 이 트랙 소관이다**: T2(flip/placebo)는 `experiments/analysis/flip.py`·
`placebo.py`, T3(난이도 층화, 규약 4)는 `experiments/analysis/stratify.py`,
T4(calibration)는 `experiments/analysis/calibration.py`, 표 통합은
`experiments/analysis/aggregate_tables.py`(T1–T4 전부 생성). 배경 도구는
`src/eval/pmi_shift_signal.py`, `scripts/ANALYSIS_INDEX.md` 참조.

**트랙 C — SFT v2 데이터.** R1류 long-CoT에 메타 주석을 다는 차세대 SFT 코퍼스
파일럿. Stage 게이트와 독립적으로 시작 가능하다. 기존 빌더
(`scripts/build_v8_strict_paired_data.py`, `scripts/build_rv_full.py`)와 검증기
(`scripts/validate_v8_strict_data.py`)를 참고해 같은 스키마(parquet, segment
loss mask 호환)로 만든다.

**트랙 D — 사이트·집필.** 설명 사이트는 https://metacog-explainer.pages.dev,
소스는 `docs/site/`. 논문 숫자는 반드시 3절 규약을 통과한 것만 싣고, PRELIMINARY
결과는 그렇게 표기한다.

## 7. 보안

토큰(HF/GitHub/WandB)은 **.env에만** 둔다. .env는 gitignore되어 있고, 코드·yaml·
문서 어디에도 실제 토큰을 하드코딩하지 않는다 — yaml은 `${HF_TOKEN}` 치환,
스크립트는 env 변수만 읽는다. 커밋 전 diff에서 `hf_`, `ghp_` 패턴을 확인하는
습관을 들일 것.
