# Meta-CoT × RLSD/OPD/ROD-PT 실험 리포트 (Plan v5.7)

**작성일**: 2026-05-08
**대상 독자**: ML 비전공자도 따라올 수 있도록 작성. 핵심 용어는 처음 등장할 때 정의.
**문서 목적**: 지금까지의 가설·실험·결과를 한 곳에 모으고, 진행 중인 실험(ROD-PT R15, OPD R8)이 검증할 가설을 명확히 한다.

---

## 1. Executive Summary

수학 추론 모델이 자기 자신의 사고 과정을 점검(metacognition)하면 더 정확해질까? 우리는 Qwen3-8B 모델에 `<|meta|>` 토큰을 학습시켜 모델이 풀이 중간에 "여기서 잠깐, 이 단계가 맞나?"를 명시적으로 표현하도록 했다. 첫 결과는 도메인마다 엇갈렸다. 메타토큰을 쓰는 모델은 GSM8K(쉬움)와 MATH-500(중간)에서 점수가 올랐지만, AIME 2024(어려움)에서는 **36.7% → 13.3%로 23.4%포인트 떨어졌다**. 메타토큰이 긴 사고 흐름(long chain-of-thought)을 중간에 끊는 부작용이 있다는 가설(H-A)이 등장했고, 이를 검증·해결하기 위해 Plan v5.7에서 세 가지 방법(ROD-PT, OPD-Decoy, OPD-PT)을 설계했다. 본 리포트는 이 방법들이 무엇을 가설로 삼는지, 어떻게 검증하는지, 그리고 현재까지의 진행 상황(인프라 OOM 디버깅 9사이클 끝에 R5와 동일한 H200 141GB 환경으로 안정화)을 정리한다.

---

## 2. 배경: 무엇을 풀고 있는가

### 2.1 Meta-CoT란

Meta-CoT는 풀이 과정 중간에 자기점검을 끼워 넣은 CoT다. 배경부터 짚으면, CoT(Chain-of-Thought)는 모델이 "정답"이 아니라 "정답에 이르는 풀이 과정"을 텍스트로 출력하는 방식이다. Meta-CoT는 여기에 한 층을 더 얹어, 풀이 중간에 모델이 자기 풀이의 신뢰도와 가능한 오류를 명시적으로 점검하는 메타 블록을 넣는다. 코드 상으로는 `<|meta|>...<|/meta|>` 토큰 쌍 안에 점검 텍스트가 들어간다.

> 예시:
> "First, multiply 12 × 7. <|meta|> Confidence 0.8. Could fail if I miscount carries. <|/meta|> Result: 84."

### 2.2 학습 파이프라인

학습은 SFT + RL 두 단계 구성이고, 우리 RL은 모두 GRPO에 메타 신호를 얹는 변형이다. 비전공자를 위해 용어 먼저 정리한다.

- **SFT(Supervised Fine-Tuning)**: 정답 풀이를 직접 보여주고 따라 쓰게 하는 지도학습.
- **RL(Reinforcement Learning)**: 모델이 답을 여러 번 시도(rollout)하게 하고, 정답이면 +, 오답이면 − 보상을 줘서 정책을 점차 정답 쪽으로 옮긴다. **advantage**는 한 rollout이 평균보다 얼마나 좋았는지를 나타내는 점수다.
- **GRPO**: PPO(표준 RL 알고리즘)의 그룹 변형. 한 문제에 여러 rollout을 띄워 그룹 평균을 baseline으로 쓴다.

학습 단계는 다음과 같다.

1. **SFT 단계**: 4,996개의 Meta-CoT 풀이 예시(GPT-5.4-mini가 생성, 평균 신뢰도 0.745)로 모델을 지도학습. 결과 = `v8_meta_inside_strict_sft` (이후 모든 RL의 cold start).
2. **RL 단계**: 다음 세 baseline이 있다.
   - **Base GRPO**: 메타토큰 없는 일반 GRPO. 정답이면 +, 오답이면 −.
   - **RLVR(RL with Verifiable Rewards)**: 수학·코드 분야의 사실상 표준. 자동 채점 가능한 정답/오답을 reward로 받아 GRPO/PPO로 학습한다. 우리의 Base GRPO와 같다고 봐도 된다 — 즉 §3.1의 "Base GRPO step 300" 점수가 곧 RLVR ceiling 역할을 한다. 우리 방법은 이 ceiling을 메타 신호로 더 끌어올릴 수 있는지 검증하는 것이다.
   - **RLSD(RL with Self-Distillation)**: teacher 모델의 분포를 추가 신호로 삼는 변형. R5가 이 계열이다.

### 2.3 두 가지 SFT 비교

본 연구 전체에서 가장 중요한 비교는 다음 두 모델이다.

| 모델 | 메타토큰 | 핵심 차이 |
|---|---|---|
| **Base SFT** | ❌ | 표준 CoT만 학습 |
| **Meta SFT** (`v8_meta_inside_strict_sft`) | ✅ | 메타블록까지 학습. 모든 ROD/OPD 실험의 cold start |

---

## 3. 기존 결과: 무엇이 알려져 있는가

### 3.1 1,030 문제 벤치마크 (GSM8K 500 + MATH-500 500 + AIME 30, 16k context)

| 모델 | Overall | GSM8K | MATH-500 | AIME 2024 | Meta emission |
|---|---:|---:|---:|---:|---:|
| **Base GRPO step 300** | 76.99% | 93.4% | 63.0% | **36.7%** | 0% |
| **Meta GRPO E21R step 300** | **81.65%** | 92.6% | **74.8%** | **13.3%** | 88.7% |
| (변화) | +4.66pt | −0.8pt | +11.8pt | **−23.4pt** | +88.7pt |

> 출처: `results/eval_1030_base_grpo_step300_16k/base_grpo_step300_16k.json`, `results/eval_1030_meta_grpo_e21r_v2_step300_16k/meta_grpo_e21r_v2_step300_16k.json`

### 3.2 결과의 의미: Hypothesis A의 등장

메타토큰을 도입하면 GSM8K(쉬운 산수)는 거의 변화가 없고, MATH-500(중간 난이도)은 +11.8pt 오른다. 반면 AIME(올림피아드급)에서는 23.4pt 떨어진다. AIME에서만 이런 일이 벌어진 이유로 두 가지 후보 가설이 있다.

- **H-A (working memory disruption)**: 메타블록이 긴 풀이 흐름을 자르고 들어가면, 어려운 문제일수록 그 끊김이 풀이 전체를 흔든다. AIME 평균 12,830 토큰 중 메타블록은 평균 0.2개 — 적은 횟수지만 결정적 위치에서 끊을 가능성.
- **H-B (over-confidence on easy steps)**: 메타 confidence가 GSM 0.95 / MATH 0.92 / AIME 0.83으로 어디서나 높다 — 거의 항상 "맞다"고 자기보고. AIME처럼 자주 틀리는 도메인에서는 잘못된 자기점검이 잘못된 자신감을 강화한다.

E21R 결과만으로는 H-A와 H-B를 구분할 수 없다. 그래서 Plan v5.7의 새 방법들이 등장한다.

### 3.3 R5 (RLSD with SDC factor) 결과

R5는 H-A를 해결하지 못한다는 것이 plan v5.7 anchor의 핵심 음의 결과다. R5는 메타토큰의 자기점검을 teacher 모델의 logprob과 정합화하는 방향으로 학습한다(메커니즘은 §4.2). Step 300까지 진행하고 16k context에서 평가했을 때 **AIME 10.0% — Base SFT와 동일** (출처: plan v5.7 §1 anchor table, line 35). 즉, content-only self-distillation만으로는 AIME 회복이 일어나지 않으며, 이 음의 결과가 Plan v5.7에서 위치 신호(ROD-PT) · 분포 KL(OPD)로 분기한 출발점이다.

---

## 4. Plan v5.7의 세 가지 방법

### 4.1 공통 설정

| 항목 | 값 |
|---|---|
| 모델 | Qwen3-8B (8B 파라미터, bf16) |
| Cold start | Meta SFT v8 (`v8_meta_inside_strict_sft/checkpoint-254`) |
| Teacher | 동일한 Meta SFT v8 (Plan v5.7 §10.5에서 확정) |
| Optimizer | AdamW (FSDP shard) |
| Rollout | vLLM colocate, num_rollouts=2, max_response_length=4096 |
| 하드웨어 | **H200 141GB × 4** (R5 성공 환경과 동일) |

### 4.2 R5 baseline (참고): SDC factor

R5는 PPO 손실의 advantage에 메타 영역에서 Self-Distillation Coefficient(SDC) factor를 곱해, teacher가 더 선호하는 메타 토큰의 학습 신호를 증폭한다.

- factor 정의: `w_t = exp(sign(A) × (logp_T − logp_S))`, 클립 `[1−0.2, 1+0.2] = [0.8, 1.2]` (`meta_rlsd_trainer.py:894-896`, `clip_eps_w=0.2`)
- A>0(좋은 rollout)이면 teacher가 더 좋아하는 토큰의 신호를 증폭, A<0이면 teacher가 덜 좋아하는 토큰의 신호를 증폭
- 메타 영역(meta_mask=1)에만 적용, 그 외 영역은 factor=1
- (참고: ROD-PT는 같은 SDC 식을 쓰되 클립 범위를 `[0.2, 5.0]`으로 더 넓혀 더 강한 증폭을 허용한다 — `meta_rod_pt_trainer.py:39-40`)

R5 결과 = AIME 10% (개선 없음). H-A를 해결하지 못함.

### 4.3 ROD-PT (R5 + Position Teacher, decoy off)

**의도**: 메타토큰을 **언제** 발화할지를 추가로 제약. R5는 메타 내용은 가르치지만 위치는 학생 자유. ROD-PT는 "teacher가 보기에도 메타가 필요한 위치였는가"를 점수로 환산해 advantage에 페널티로 더한다.

**구조**:
- T_content (= T+, R5와 동일): 학생 풀이 + gold 정답을 conditioning. 메타 내용에 SDC factor.
- T_position (신규): 학생이 메타를 시작한 위치 직전까지만 잘라서 teacher에게 입력. teacher의 다음 토큰 top-K(K=16)에 `<|meta_start|>`가 들어 있는가? 아니면 페널티 −1.0 (rollout-level).
- decoy(잘못된 정답을 conditioning한 T−)는 끄고 = decoy off.

**가설(H-ROD-PT)**:
- H1: position penalty가 활성화되면 메타 발화 위치가 teacher 분포와 정렬된다.
- H2: 메타 위치 정렬이 long-CoT 흐름 단절을 줄여 AIME 정확도가 base GRPO(36.7%)에 근접하거나 능가한다.
- H3: GSM8K/MATH-500은 큰 변화 없음(이미 메타 위치 정렬이 자연스러운 도메인).

**검증**:
- Step 100/200/300 ckpt에서 1,030-문제 16k eval. AIME, GSM8K, MATH-500 정확도 + 메타 emission rate + AIME 평균 길이 추적.
- W&B 메트릭: `rod_pt/sdc_factor_mean`, `rod_pt/n_rollouts_with_meta`, `rod_pt/penalty_rate`, `rod_pt/meta_coverage`.
- 합격선: AIME ≥ 22% (Plan v5.7 forecast 22-32%)이면 H2 지지. AIME ≤ 13.3%면 H2 기각.

**현재 상태**: R15 (`metacot-rod-pt-R15-0508-h200`) — H200 141GB BSC tier, 19:55Z 기준 **queued 2시간 55분**. 노드 할당 대기 중. 노드 받는 즉시 yaml main이 자동으로 학습 시작.

### 4.4 OPD-Decoy (M5.2)

**의도**: R5의 SDC factor는 스칼라 Δ(per-token logprob 차이)에 의존해 신호 분산이 큼. OPD는 메타 영역에서 teacher 분포와 학생 분포의 top-K full-logit KL을 직접 PPO 손실에 더해 더 조밀한 신호를 준다. decoy는 잘못된 정답을 conditioning한 T−의 KL을 빼는 negative 항.

**손실 식**:
```
Loss = PPO_loss + α × (λ_pos × KL(T+ ∥ S) − λ_neg × KL(T− ∥ S))
```
- α, λ_pos, λ_neg = scheduling factors
- top-K=64 default (OOM시 32 fallback)
- meta region only

**가설(H-OPD)**:
- H5.2.1 (signal density): OPD per-step gradient variance < R5 per-step gradient variance, train step 50 시점 W&B `opd/grad_norm` std 비교에서 OPD가 **낮으면 지지, 같거나 크면 기각**.
- H5.2.2 (no regression): OPD-Decoy AIME @ step 200 16k ≥ **10%** AND GSM ≥ **84.2%** (R5 anchor). 둘 중 하나라도 미달이면 기각.
- H5.2.3 (H-A fallback): 단일 teacher의 분포 정렬만으로는 H-A를 못 풀 가능성. AIME ≤ Base GRPO − 10pt = **≤ 26.7%**이면 H-A는 분포 정렬과 직교한다는 신호. ROD-PT가 이를 푸는지 비교.

**검증**: ROD-PT와 동일한 1,030 eval + W&B `opd/kl_pos`, `opd/kl_neg` 추적.

**현재 상태**: R8 (`metacot-opd-R8-0508-h200`) — H200 141GB BSC, 19:55Z 기준 **queued 2시간 55분**. R15와 같은 클러스터 큐에 있음.

### 4.5 OPD-PT (계획)

ROD-PT의 R5 SDC factor 대신 OPD top-K KL을 쓰고 decoy 자리에 position teacher를 넣는 변형. ROD-PT와 OPD-Decoy 결과가 나온 뒤 다음 ablation으로 진행.

---

## 5. 비교 표: 우리는 무엇 위에 무엇을 더하는가

이 절의 핵심은 **우리 방법이 RLVR(=Base GRPO) 점수를 넘어야만 의미가 있다**는 것이다. RLVR은 수학·코드 도메인 RL의 사실상 표준이고, 우리 코드 상 Base GRPO는 RLVR과 같은 방식 — 자동 채점 가능한 정답/오답을 보상으로 GRPO 학습 — 으로 돌아간다. 따라서 §3.1의 Base GRPO step 300 결과(AIME 36.7%)가 **우리가 넘어서야 할 ceiling**이다. 메타토큰을 더한 E21R은 AIME에서 이 ceiling을 24pt 깨뜨렸고, 그래서 §6의 ROD-PT/OPD 합격선은 "AIME에서 적어도 13.3%(E21R)를 회복하고, 가능하면 36.7%(RLVR)에 근접"이 된다.

| 방법 | 외부 기준 | 메타 위치 신호 | 메타 내용 신호 | Decoy(T−) |
|---|---|---|---|---|
| RLVR / Base GRPO | 표준 RL with Verifiable Rewards | — (메타 없음) | — | — |
| Meta GRPO E21R | RLVR + 메타토큰 | 없음 | 없음 (정답만 보상) | — |
| **R5 (RLSD)** | + Self-Distillation | 없음 | SDC factor (스칼라 Δ) | 없음 |
| **OPD-Decoy** | + 분포 정렬 | 없음 | top-K KL (T+) | top-K KL (T−) |
| **ROD-PT** ← 진행 중 | + 위치 정렬 | top-K position teacher | SDC factor (R5와 동일) | off |
| **OPD-PT** ← 후속 | 위치 + 분포 | top-K position teacher | top-K KL (T+) | off |

표를 한 줄로 요약하면, R5는 메타 "내용만" 가르치고, OPD-Decoy는 "내용을 더 정밀하게" 가르치며, ROD-PT는 "내용 + 위치"를, OPD-PT는 "내용(분포 KL) + 위치"를 함께 학습한다. 어느 신호가 H-A를 푸는지는 §6에서 정의한 합격선이 가른다.

---

## 6. 진행 중인 실험: 가설→검증 매핑

| ID | 방법 | 검증할 가설 | Pass 기준 | Fail 기준 |
|---|---|---|---|---|
| R15 | ROD-PT | H-ROD-PT-2 (위치 정렬→AIME 회복) | AIME ≥ 22% @ step 200 | AIME ≤ 13.3% |
| R8 | OPD-Decoy | H-OPD-5.2.2 (분포 정렬 OK) | AIME ≥ 10% AND GSM ≥ Base GRPO | AIME < R5 (= 10%) |

두 실험 모두 결과가 나오면 **H-A vs H-B 구분**까지 가능:
- ROD-PT만 AIME 회복하면 H-A (위치 단절 문제)가 옳고 위치 신호가 답.
- OPD-Decoy도 회복하면 H-A는 부분적으로 옳지만 분포 정렬만으로도 충분.
- 둘 다 회복 못 하면 H-B (over-confidence)가 옳고 다른 처방 필요.

---

## 7. 인프라 디버깅 후기 (Plan v5.7 구현)

R10 ~ R14는 모두 H100 80GB STD에서 학습 시작도 못한 사이클이었다. 사이클별 root cause:

| Round | 근본 원인 | Fix | 결과 |
|---|---|---|---|
| R10/R11 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` × vLLM `custom_all_reduce` IPC 충돌 | env 제거 | OOM 78GB |
| R12 | OOM 첫 step (vLLM 0.25 + teacher 16GB + adam 16GB + activations 22GB ≈ 78GB) | teacher CPU offload + vllm 0.18 | OOM 동일 위치 |
| R13 | offload 코드가 `_teacher_logprobs`에만 적용 — ROD-PT는 `_completion_logits` 경로 사용 | compute_loss Phase 재배치 (teacher → student → SDC) | HF 500 transient (검증 못 함) |
| R14 | HF 500 (인프라) — 코드 fix 검증 실패 | (재시도 방향 변경) | killed |
| **R15/R8** | **하드웨어 자체가 부족 (H100 80GB)**. R5는 H200 141GB에서 성공한 환경. | yaml `141G4-H200 / Basic`로 변경 = R5와 동일 | **queued 대기 중** |

**핵심 교훈**: R5와 같은 의도(teacher in compute_loss) + 같은 아키텍처면 같은 메모리 환경(H200 141GB)이 필요. ROD-PT/OPD는 R5 위에 teacher forward를 한 번 더 추가하므로 메모리 여유는 더 필요. H100 80GB에서 OOM은 환경 차이의 직접 결과였다.

---

## 8. 다음 단계

1. **R15 + R8 노드 할당 대기** — H200 BSC 큐, 1-3시간 wait는 R5 0506b와 동일 패턴.
2. **첫 train step 검증** — `'loss':` 로그 + `rod_pt/*` / `opd/*` W&B 메트릭 양수 확인.
3. **Step 25/50 ckpt → HF auto-push** — yaml main이 자동 처리.
4. **Step 100/200/300 16k eval** — yaml step 10 이후 자동 트리거.
5. **결과를 본 리포트 §6에 채워 넣고**, ROD-PT vs OPD-Decoy vs Base GRPO 직접 비교 표 추가.
6. **OPD-PT (R9 예정) 실행** — ROD-PT/OPD 결과 본 뒤 ablation.

---

## 9. 데이터/모델 위치

- HF code snapshot: `iamseungpil/metacot:code_snapshots/metacognition.tar.gz` (Phase reorder 포함, 2026-05-08)
- HF eval push: `iamseungpil/metacot:eval/rod_pt_R10_2026_05_07` (R15 완료 시), `iamseungpil/metacot:eval/opd_R7_2026_05_06` (R8 완료 시)
- HF ckpt push: `iamseungpil/metacot-h100-rod-pt-R10-0507`, `iamseungpil/metacot-h100-meta-opd-R7-0506`
- W&B: project `skilldiscovery2`, runs `rod_pt_R10_h100_4x4k`, `meta_opd_decoy_R7_h200_0506`

---

## 10. 한 줄 요약

> Meta-CoT는 GSM8K/MATH-500 점수는 올리지만 AIME에서는 36.7% → 13.3%로 무너진다. ROD-PT는 "메타가 발화되는 위치"를 teacher와 정렬해 이 단절을 메우려 하고, OPD-Decoy는 "메타 내용의 분포"를 teacher와 정렬해 같은 문제에 다른 각도로 접근한다. R5 성공 환경(H200 141GB)으로 옮긴 R15/R8 두 실험이 첫 step 결과를 내면 H-A(위치 단절) vs H-B(과신) 가설을 구분할 수 있다.
