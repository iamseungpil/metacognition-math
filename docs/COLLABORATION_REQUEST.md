# 협업 요청 — 채점 격자 + instruct 사다리 독립 재현

작성 2026-08-03. 이 문서 하나로 착수할 수 있게 썼다. 막히는 지점이 있으면 그건 우리 문서의 결함이니 알려달라.

> **먼저 읽을 것 셋**: [`README.md`](../README.md) → [`docs/CLAIMS.md`](CLAIMS.md) → [`docs/CONSTITUTION.md`](CONSTITUTION.md)

---

## 0. 무엇을 재현해달라는 것인가

**재현 대상은 딱 하나다 — base 대비 meta 의 RL 차이.**

> 같은 문제 · 같은 GRPO · 같은 하이퍼파라미터에서,
> **메타 블록을 쓰도록 SFT 된 모델**과 **메타 블록만 제거한 쌍둥이 SFT 모델**이
> RL 을 거친 뒤 held-out 정확도에서 갈리는가.

우리 instruct 기질(`Qwen/Qwen3-8B`) 실측은 **MATH500 +14.00pp** (`pmishift` − `base_matched`,
p<.001), AIME +8.75pp 이고 **모든 난이도 레벨에서 유의**했다. 이 수가 당신 손에서 다시 나오는가가
이 요청의 전부다.

**왜 남에게 부탁하는가** — 전 팔이 **단일 학습 시드**다. 그리고 우리가 못 본 설정 의존성이
있다면 다른 손에서만 드러난다.

**왜 base 기질이 아니라 instruct 인가.** 목표 기질은 `Qwen/Qwen3-8B-Base` 이고 우리는 거기서
같은 사다리를 돌리는 중인데, 학습 도중 **메타 발화가 1.00 → 0.018 로 무너져** 아직 판정을 못 했다.
원인을 노브 단위로 좁히는 중이다. 당신이 base 를 돌려 실패하면 그것이 **우리 실패의 충실한
재현인지 당신 쪽 설정 오류인지 구분할 수 없다.** instruct 는 기대값이 확정돼 있어 결과가 즉시
해석된다 — 맞으면 파이프라인이 검증된 것이고, 어긋나면 그 자체가 발견이다.
base 사다리는 우리가 원인을 규명한 뒤 **과제 C** 로 부탁드릴 예정이다.

---

## 1. 과제 A — 채점 격자 (GPU 0, 먼저)

**왜 먼저인가**: 같은 바이트에서 채점 규칙 하나가 결과를 **10.2pp 움직이고 부호를 뒤집는다.**

| MATH500 16k | shiftonly | gandhi | Δ |
|---|---|---|---|
| parquet에 저장된 `is_correct` | 62.28% | 68.08% | **−5.80pp** |
| `math_verify` 재채점 | 64.40% | 60.02% | **+4.38pp** |

우리 재채점은 **동치 판정(2단계)만** 고쳤고 **답 추출(1단계)은 저장값을 그대로** 썼다. 그리고
`shiftonly`는 메타 블록을 2배 더 쓰므로 **블록 안의 `\boxed`를 최종답으로 오인할 위험이 구조적으로
크다.** 여기서 무너지면 아래 GPU 작업은 전부 의미가 없다.

**자료** (HF, 토큰 없이 공개):
- `iamseungpil/metacot-h200-triobj-dcpo-v3` → `eval/{base_matched,pmishift,shiftonly,gandhi}_1030_v2/`
- `iamseungpil/metacot-rv` → `eval/pmishift_1030_gs300/`, `eval/pmishift_heldout_gs180/`, `eval/decode_sweep_v1/`

각 parquet에 `question` · `gold_answer` · `completion`(응답 전문) · `answer_extracted` ·
`is_correct`(옛 채점기, **믿지 말 것**) · `num_meta_blocks` · `finish_reason`이 들어 있다.

**해야 할 것**: **우리 스크립트를 쓰지 말고** `math_verify`로 독립 재채점한 뒤,
**결과를 보기 전에 선언한** 36셀을 전수로 돌린다.

- 추출 6종: ①마지막 `\boxed` ②**메타 span 제거 후** 마지막 `\boxed` ③마지막 `<|/meta|>` 이후
  ④strict boxed(fallback 없음) ⑤저장된 `answer_extracted` ⑥끝 200자 내 첫 `\boxed`
- 절단 3종: ①그대로 ②`finish_reason != "stop"`을 오답 ③해당 행 제거
- 통계: 문항 단위 avg@8 → **문항 페어드 부트스트랩 10,000회**

**판정**: 36셀 **전부**에서 `shiftonly − gandhi ≥ +2pp`이고 CI가 0을 배제해야 "채점 강건"이다.
한 셀이라도 걸치면 헤드라인은 최대값이 아니라 **방어 가능한 최솟값**으로 내린다.
②(메타 span 제거)는 저장소 어디에도 구현이 없다 — `grep -rn "strip_meta" experiments/` → 0건.

---

## 2. 과제 B — **instruct 사다리 전체 재현** (SFT → meta-GRPO → 평가)

과제 A 가 통과한 뒤. **SFT 부터 평가까지 전 구간이 맞다.** 중간만 재현하면 설정 의존성이
어디서 들어왔는지 알 수 없다.

### ⚠ 우리 SFT 체크포인트를 가져다 쓰지 말 것

메타 쪽 산출물은 HF 에 살아 있다(`v8_meta_inside_strict_sft` 16.4GB · `v8_rv_functional_sft`
16.4GB). **무메타 쪽 `v8_base_matched_strict_sft` 는 가중치가 없다** — 실측으로 61개 파일 0.1GB,
safetensors 0개, tokenizer·config·rng 상태뿐이다. 즉 무메타 팔은 어차피 SFT1 부터 돌려야 한다.

그런데 **한쪽만 우리 체크포인트를 쓰면 "메타 vs 무메타" 가 "우리 SFT run vs 당신 SFT run" 과
교란된다.** 그 대비가 바로 재현 대상이므로 이 교란은 실험을 통째로 무의미하게 만든다.
⇒ **양쪽 SFT 를 모두 당신 쪽에서 돌린다.** 우리 체크포인트는 중간 점검용 대조로만 쓴다.

### B-1. SFT1 (2팔, 각 ~3h) — init 은 둘 다 `Qwen/Qwen3-8B`

| 팔 | config | 데이터 (HF `iamseungpil/metacot`) | 행 |
|---|---|---|---|
| 메타 | `configs/sft_v8_meta_inside_strict.yaml` | `data/v8_meta_inside_strict.parquet` | 4,264 |
| 무메타 쌍둥이 | `configs/sft_v8_base_matched_strict.yaml` | `data/v8_base_matched_strict.parquet` | 4,264 |

3 epoch · **lr 2.0e-6** · max_length 4096. (base 계보는 lr 1e-5 를 쓴다 — 헷갈리기 쉬우니 주의.)

**중간 확인**: 무메타 산출물에서 `<|meta|>` 가 나오면 안 된다. 나오면 쌍둥이가 오염돼 대조가 깨진다.
`v8_meta_inside_strict_sft` 로 같은 프롬프트를 돌려 발화율 1.00 / wellformed 1.0 이 나오는지,
그리고 MATH500-100 에서 메타 팔 **58.0%** / 쌍둥이 **55.5%** 근처인지로 SFT1 을 검산한다.

### B-2. SFT2 (2팔, 각 ~4h)

| 팔 | config | init | 데이터 | 행 |
|---|---|---|---|---|
| 메타 | `configs/archive/sft_rv_functional.yaml` | B-1 메타 산출물 | **`metacot-rv`** `data/rv_redirect_verify_functional.parquet` | 1,763 |
| 무메타 | `configs/archive/sft_base_rv.yaml` | B-1 무메타 산출물 | **`metacot`** `data/v8_base_rv_sft.parquet` | 1,763 |

⚠ **두 parquet 이 서로 다른 HF repo 에 있다.** 메타 쪽만 `metacot-rv` 다.
3 epoch · lr 1.0e-5 · bs1 × ga4.

**두 코퍼스는 행 단위로 대응하고 메타 블록 유무만 다르다** — 둘 다 verify 1,209 / redirect 554,
easy 870 / medium 893, **hard 0건**.

⚠ **이 hard 0건이 이 프로젝트의 "분포 밖" 축을 정의한다.** 습관을 심는 데이터에 어려운 문제가
없으므로 **MATH500 level 4–5 (262문항)가 곧 분포 밖 표본**이다. 데이터셋 이름이 아니라 난이도다.

### B-3. RL (각 ~30h) — **헤드라인 2팔이 먼저**

| 우선 | 팔 | 런처 | init | 보상 |
|---|---|---|---|---|
| **1** | `base_matched` | `archive/launchers_pre_rq3/h100std_base_matched_rl.yaml` | 무메타 SFT2 | vanilla GRPO |
| **1** | `pmishift` | `archive/launchers_pre_rq3/h100std_pmishift.yaml` | 메타 SFT2 | PMI-shift 7헤드 패키지 |
| 2 | `gandhi` | `archive/launchers_pre_rq3/h100std_gandhi.yaml` | 메타 SFT2 | vanilla GRPO |
| 2 | `shiftonly` | `archive/launchers_pre_rq3/h100std_shiftonly.yaml` | 메타 SFT2 | PMI-shift **단독**(타 헤드 0) |

**우선순위 1 의 두 팔이 곧 재현 대상**이다 — `pmishift − base_matched` = **MATH500 +14.00pp**.
우선순위 2 는 그 +14.00pp 를 성분으로 쪼개는 선택 과제다(`shiftonly − gandhi` = **+4.38pp** 가
보상 단독 기여). 시간이 없으면 1 만 해도 요청은 충족된다.

공통 하이퍼파라미터: `train_batch_size 64` × `rollout.n 8` = 512 rollout/step ·
`ppo_mini_batch_size 8` · lr 1e-6 · **300 step** · clip 0.2/0.28 · temp 1.0 · max_response 8192 ·
Dr.GRPO(`norm_adv_by_std=false`). 런처에 이미 들어 있으니 바꾸지 말 것.

### ⛔ 발사 전 반드시 고칠 두 줄 — 그대로 돌리면 실패한다

런처는 우리 SFT2 산출물을 HF 에서 스테이징하도록 **경로가 박혀 있다.** 당신 SFT2 로 바꾸지 않으면
(a) 무메타 팔은 **다운로드 자체가 실패하고**(`metacot-rv` 에 `models/v8_base_rv_sft` 가 없다),
(b) 메타 팔은 조용히 **우리 체크포인트로 학습해** 재현이 아니라 우리 런의 반복이 된다.

| 파일 | 줄 | 지금 | 바꿀 것 |
|---|---|---|---|
| `h100std_pmishift.yaml` | 112 | `actor_rollout_ref.model.path=/scratch/models/v8_rv_functional_sft` | 당신 **메타 SFT2** |
| `h100std_base_matched_rl.yaml` | 108 | `actor_rollout_ref.model.path=/scratch/models/v8_base_rv_sft` | 당신 **무메타 SFT2** |

위쪽 `snapshot_download(... allow_patterns=["models/<이름>/**"] ...)` 블록의 이름도 같이 바꾼다
(`h100std_pmishift.yaml:80`, `h100std_base_matched_rl.yaml:78`). 가장 간단한 길은 당신 SFT2 둘을
같은 이름으로 당신 HF repo 에 올리고 `repo_id` 만 바꾸는 것이다.

학습·검증 데이터(`verl_{train,val}_meta_mix.parquet`)는 `scripts/pull_parquets.py` 가
`iamseungpil/metacot-sdc-data` 에서 자동으로 당겨오므로 손댈 필요 없다.

**학습 중 게이트** — 메타 팔에만 해당(무메타 팔은 메타를 안 쓰므로 발화 지표가 없다):

| 시점 | 조건 |
|---|---|
| gs25 | `dcpo/meta_emit_rate` ≥ 0.80 · `dcpo/pmishift_attempted_rate` ≥ 0.30 · `actor/entropy` > 0.1 |
| **gs50** | **`meta_emit_rate` ≥ 0.80. 미달이면 중단하고 알려달라** — 우리 base 팔이 여기를 못 넘고 30시간을 버렸다 |
| gs150 | `meta_emit_rate` ≥ 0.80 **유지** |

**추가로 켤 것 3종**(학습 궤적에 영향 없음, 우리가 안 켜서 손해 본 것들):
- `++trainer.val_before_train=True` — gs0 기준선 행. 없으면 SFT 단계 출발선 차이를 뺄 수 없다
- `++trainer.log_val_generations=8` — **응답 로깅.** 안 켜면 응답을 하나도 못 본다
- `--keep 3` — `--keep 1` 은 판정 지점 체크포인트를 프루닝한다. 우리는 그렇게 gs100–150 을 잃었다

### B-4. 평가 (~2h)

`archive/launchers_pre_rq3/h100std_pmishift_1030_eval.yaml` 을 클론해 체크포인트 경로만 바꾼다.
FSDP 병합 → `scripts/eval_vllm_1030.py` → HF 업로드가 자동이다. 로컬에서 직접 돌리려면:

```bash
python scripts/eval_vllm_1030.py \
    --model_path <merged_ckpt_dir> --model_name <arm>_1030 \
    --output_dir results/eval_1030_<arm>/ \
    --max_tokens 16384 --temperature 0.7 --num_samples 8 --seed 42
```

held-out 1030 = GSM8K 500 + MATH500 500 + AIME 30. **두 팔을 같은 job · 같은 seed 로** 돌린다.
avg@8 (AIME 는 avg@16). 산출 parquet 에 `completion` 전문이 남아야 재채점이 가능하다.


## 3. 무엇을 보고 판정하는가

**두 개만 본다.**

| | 지표 | 우리 실측 | 판정 |
|---|---|---|---|
| **주** | MATH500 전체 `acc(meta) − acc(base_matched)` | **+14.00pp** (p<.001) | CI 하한 > 0 이면 재현 성공 |
| 보조 | 난이도 기울기 `Δacc(L4–5) − Δacc(L1–2)` | **+7.17pp** (p=.009) | north-star: 분포 밖에서 더 커야 한다 |

난이도 기울기의 해상도: 바닥 0.00 · 천장 +29.97 · **잡음바닥 ±3.08pp**
(같은 모델의 8샘플을 4/4 로 쪼갠 A-vs-A 실측). L1–2 +10.53 → L4–5 **+17.70**.

**층화는 사전 처리(pre-treatment) `level` 필드로 한다.** 통제군 자신의 정답률로 쪼갠 뒤 그 통제군과
차분하면 평균회귀 때문에 부호가 뒤집힌다 — 우리가 실제로 당했다.

**채점**: `math_verify`. ⛔`check_correctness` 는 버그가 문서화돼 있으니 쓰지 말 것.
parquet 에 저장된 `is_correct` 도 옛 채점기 산물이라 **믿지 말 것**(과제 A 참조).

⛔ **in-training val594 는 판정에 쓰지 않는다** — 벤치별 셀이 21~38문항이라 한 문제가 2.6~4.8pp 다.
추세용으로만 본다.

**보고 형식**: 실험 하나 = `VERDICT.md` 하나 + `docs/CLAIMS.md` 에 한 줄.
주장마다 **닫는 것 / 여는 것 / 재확인 계수기** 세 필드가 필수다(형식은 `CLAIMS.md` 참조).

---

## 4. 자료 위치 (전부 PUBLIC, 토큰 불필요)

**데이터 — 두 repo 로 갈려 있으니 주의**

| parquet | HF repo | 경로 |
|---|---|---|
| `v8_meta_inside_strict` (SFT1 메타, 4264행) | `datasets/iamseungpil/metacot` | `data/` |
| `v8_base_matched_strict` (SFT1 무메타, 4264행) | `datasets/iamseungpil/metacot` | `data/` |
| `rv_redirect_verify_functional` (SFT2 메타, 1763행) | **`datasets/iamseungpil/metacot-rv`** | `data/` |
| `v8_base_rv_sft` (SFT2 무메타, 1763행) | `datasets/iamseungpil/metacot` | `data/` |
| **RL 학습셋** `verl_train_meta_mix` | `datasets/iamseungpil/metacot-sdc-data` | 최상위 |
| **RL val** `verl_val_meta_mix` (594문항) | `datasets/iamseungpil/metacot-sdc-data` | 최상위 |

여덟 개 전부 이 레포 `share/data_parquets/` 에도 커밋돼 있다(18MB). `scripts/pull_parquets.py` 가
job 시작 시 HF 에서 당겨온다.

**모델**

| 모델 | 위치 | 상태 |
|---|---|---|
| `v8_meta_inside_strict_sft` (SFT1 메타) | `metacot` `models/` | ✅ 16.4GB · 참고용 |
| `v8_base_matched_strict_sft` (SFT1 무메타) | `metacot` `models/` | ❌ **가중치 없음** (0.1GB, config·tokenizer 뿐) |
| `v8_rv_functional_sft` (SFT2 메타) | `metacot-rv` `models/` | ✅ 16.4GB · 참고용 |
| SFT2 무메타 | — | ❌ 존재하지 않음 |
| instruct RL 체크포인트 4팔 | — | ❌ **삭제됨** — 생성 산출물만 보존(재채점 가능, 재실행 불가) |

⇒ B-1·B-2 를 양쪽 다 돌려야 하는 이유가 이 표다.

**코드**

| | 어디 |
|---|---|
| SFT config | `configs/sft_v8_*.yaml`, `configs/archive/sft_{rv_functional,base_rv}.yaml` |
| RL 런처 | `archive/launchers_pre_rq3/h100std_{pmishift,shiftonly,gandhi,base_matched_rl}.yaml` |
| 평가 | `scripts/eval_vllm_1030.py` + `archive/launchers_pre_rq3/h100std_*_1030_eval.yaml` |
| 트레이너 | `src/training/verl_sdc.py` (entry `python -m src.training.verl_sdc`) |
| PMI-shift 보상 | `src/training/dcpo_pmi_shift.py` |
| 노브 등록부 | `core/KNOBS.yaml` — 49개 전수, 끄면 나는 실패까지 |
| 우리 판정 기록 | `docs/CLAIMS.md`, `docs/reports/2026-08-03-*.md` |

**환경**: `cp .env.example .env` 후 `HF_TOKEN` 만 채우면 된다(업로드용). 읽기는 토큰 없이 된다.
이미지 `mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch28x`,
conda env 는 `scripts/` 의 conda-pack 참조. 하드웨어는 **H100 × 4** 기준이다.

## 5. 우리가 지금 하고 있는 것 (중복 방지)

`b0p`(gs191) · `b2p`(gs175에서 재개) · **`b3p + meta_floor=0.05`**(gs16) 세 팔을 base에서 돌리는 중이다.
당신 재현과 겹치지 않는다 — 우리는 **노브 원인 규명**, 당신은 **독립 재현**이다.
