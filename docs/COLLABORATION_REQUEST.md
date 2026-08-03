# 협업 요청 — base 기질에서 SFT → meta-GRPO → 평가 전체 재현

작성 2026-08-03. 이 문서 하나로 착수할 수 있게 썼다. 막히는 지점이 있으면 그건 우리 문서의 결함이니 알려달라.

> **먼저 읽을 것 셋**: [`README.md`](../README.md) → [`docs/CLAIMS.md`](CLAIMS.md) → [`docs/CONSTITUTION.md`](CONSTITUTION.md)

---

## 0. 왜 이걸 부탁하는가

우리는 instruct 기질(`Qwen/Qwen3-8B`)에서 메타인지 보상(PMI-shift)이 **MATH500 +14.00pp**(vs 메타
제거 쌍둥이)를 낸다는 걸 보존 산출물 재채점으로 확인했다. 그런데 **목표 기질은 base
(`Qwen/Qwen3-8B-Base`)**이고, 거기서는 학습 도중 **메타 발화가 1.00 → 0.018로 무너져** 아직 판정을
못 했다. 우리가 지금 그 원인을 노브 단위로 좁히고 있다.

**당신에게 부탁하는 것은 우리 결과의 독립 재현이다.** 두 가지를 얻는다 — 우리가 못 본 설정
의존성이 있으면 드러나고, 전 팔이 **단일 학습 시드**인 현 상태에 **두 번째 시드**가 생긴다.

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

## 2. 과제 B — base 사다리 전체 재현 (SFT → meta-GRPO → 평가)

과제 A가 통과한 뒤. **네, SFT부터 성능 확인까지 전 구간이 맞다.** 중간만 재현하면 우리가 못 본
설정 의존성이 어디서 들어왔는지 알 수 없다.

### B-1. SFT1 (2팔, ~6h)

| 팔 | config | init | 데이터 |
|---|---|---|---|
| 메타 | `configs/sft_b2p_v8meta.yaml` | `Qwen/Qwen3-8B-Base` | `b2on_v8meta_strict_sft.parquet` (4,245행) |
| 무메타 쌍둥이 | `configs/sft_b0p_v8base.yaml` | 동일 | `b0on_v8base_strict_sft.parquet` (4,245행) |

3ep · lr 1e-5 · max_length 4096.

**중간 확인 앵커** — 이게 안 맞으면 그 아래는 볼 필요 없다:

| | MATH500-100 | AIME | 메타 발화 | wellformed |
|---|---|---|---|---|
| 메타 SFT1 | **58.0%** | 10.0% | **100%** | **1.0** |
| 무메타 쌍둥이 | 55.5% | 8.3% | **0.5% ≈ 0** | — |

무메타 쪽 `meta_rate`가 오르면 **쌍둥이가 메타를 흘린 것**이고 대조가 깨진다.

### B-2. SFT2 (2팔, ~4h)

| 팔 | config | init | 데이터 |
|---|---|---|---|
| 메타 | `configs/sft_b2p2_rvfull.yaml` | B-1 메타 산출물 | `rv_redirect_verify_functional.parquet` (1,763행) |
| 무메타 | `configs/sft_b0p2_rvfull.yaml` | B-1 무메타 산출물 | `v8_base_rv_sft.parquet` (1,763행) |

3ep · lr 1e-5. **두 코퍼스는 행 단위로 대응하고 메타 블록 유무만 다르다**
(둘 다 verify 1,209 / redirect 554, easy 870 / medium 893, **hard 0건**).

⚠ **이 hard 0건이 이 프로젝트의 "분포 밖" 축을 정의한다.** 습관을 심는 데이터에 어려운 문제가
없으므로 **MATH500 level 4–5 (262문항)가 곧 분포 밖 표본**이다. 데이터셋 이름이 아니라 난이도다.

### B-3. RL (3팔, 각 ~30h)

| 팔 | 런처 | init | 보상 |
|---|---|---|---|
| b0p (통제군) | `h100std_rq3v2f_b0p.yaml` | 무메타 SFT2 | vanilla GRPO |
| b2p (프라이밍) | `h100std_rq3v2f_b2p.yaml` | 메타 SFT2 | vanilla GRPO |
| b3p (패키지) | `h100std_rq3v2f_b3p.yaml` | 메타 SFT2 | triobj + PMI-shift |

**⚠ 런처의 `models/...` 경로를 당신 SFT 산출물로 바꿔야 한다.** 안 바꾸면 우리 init을 스테이징한다.

**발사 전 게이트**(`docs/CONSTITUTION.md` Part VI + 우리가 추가한 것):

| 시점 | 조건 |
|---|---|
| gs25 | `dcpo/meta_emit_rate` ≥ 0.80 · `pmishift_attempted_rate` ≥ 0.30 · `n_save` > 0 · `actor/entropy` > 0.1 |
| **gs50** | **`meta_emit_rate` ≥ 0.80. 미달이면 중단** — 우리 b3p가 여기를 못 넘고 30시간을 버렸다 |
| gs150 | `meta_emit_rate` ≥ 0.80 **유지** |

**추가로 켤 것 3종**(학습 궤적 무영향, 우리가 b3p에서 놓쳐 손해 본 것들):
- `++trainer.val_before_train=True` — gs0 기준선 행. 없으면 SFT 단계 출발선 차이를 뺄 수 없다
- `++trainer.log_val_generations=8` — **응답 로깅.** 지금 base 학습은 응답을 하나도 안 남긴다
- `--keep 3` — `--keep 1`은 판정 지점 체크포인트를 프루닝한다. 우리는 그렇게 gs100–150을 잃었다

### B-4. 평가

`archive/launchers_pre_rq3/h100std_pmishift_1030_eval.yaml`을 클론해 경로만 바꾼다.
FSDP 병합 → `eval_vllm_1030.py` → HF 업로드가 자동이다.
held-out 1030 = GSM8K 500 + MATH500 500 + AIME 30, 16k tokens, avg@8(AIME avg@16), temp 0.7.

---

## 3. 무엇을 보고하면 되는가

**주 지표**: MATH500 **level 4–5 (n=262)** 에서 `Δacc(L4–5) − Δacc(L1–2)`.
바닥 0.00 · 천장 +29.97 · **잡음바닥 ±3.08pp**(같은 모델 8샘플을 4/4로 쪼갠 A-vs-A 실측).

우리 instruct 실측: 전체 **+7.17pp** (p=.009). 분포 내 +10.53 → 분포 밖 **+17.70**.

⛔ **in-training val594는 판정에 쓰지 않는다** — 벤치별 셀이 21~38문항이라 한 문제가 2.6~4.8pp다.
추세용으로만 본다.

**보고 형식**: 실험 하나 = `VERDICT.md` 하나 + `docs/CLAIMS.md`에 한 줄.
주장마다 **닫는 것 / 여는 것 / 재확인 계수기** 세 필드가 필수다(형식은 `CLAIMS.md` 참조).

---

## 4. 자료 위치

| | 어디 |
|---|---|
| 코퍼스 8종 | `share/data_parquets/` (레포에 커밋됨) · 정본은 HF `metacot` / `metacot-rv` |
| SFT init 모델 | HF `metacot`(dataset) `models/`, `metacot-rv`(dataset) `models/v8_rv_functional_sft` |
| 런처·config | 레포 루트 + `configs/` + `archive/launchers_pre_rq3/` |
| 노브 등록부 | `core/KNOBS.yaml` — 49개 전수, 끄면 나는 실패까지 |
| 우리 판정 기록 | `docs/CLAIMS.md`, `docs/reports/2026-08-03-*.md` |

⚠ **instruct 세대 RL 체크포인트는 삭제됐다.** 생성 산출물만 남아 **재채점은 되고 재실행은 안 된다.**
복원하려면 `models/v8_rv_functional_sft`에서 RL을 다시 돌려야 한다(과제 A 이후 선택).

## 5. 우리가 지금 하고 있는 것 (중복 방지)

`b0p`(gs191) · `b2p`(gs175에서 재개) · **`b3p + meta_floor=0.05`**(gs16) 세 팔을 base에서 돌리는 중이다.
당신 재현과 겹치지 않는다 — 우리는 **노브 원인 규명**, 당신은 **독립 재현**이다.
