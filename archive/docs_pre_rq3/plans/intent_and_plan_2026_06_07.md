# Meta-CoT Intent & Plan (2026-06-07)

## 0. Intent (north-star)

> **메타인지 행동을 강화해서 성능(정확도)을 올린다.**
> 이를 위해 (1) *언제* 어떤 메타인지를 하고 *무엇이 좋은 습관*인지 규정한 **metacot를
> 정의**하고, (2) 그 metacot를 **강화하는 RL 방법을 개발**한다.

핵심 명제:
- 메타인지(`<|meta|>` 블록: 가정 점검, 검산, 불확실성 표명, 접근 전환)는 **목적이 아니라
  정확도를 끌어올리는 수단**이다.
- 따라서 RL은 "메타인지를 많이 해라"가 아니라 **"유용할 때 하는 메타인지를 보상"**해야 한다.
- **calibration(자기 confidence를 실제 정답률에 맞추기)은 부분 목표**다. confidence 정렬은
  모델이 "언제 검산/전환할지" 판단하는 신호로서 가치가 있을 때만 의미가 있다.

성공 기준(2축):
- **A축(주):** Meta-CoT 정확도 > Base SFT (✅ 이미 달성: RLVR 0.786 ≫ Base 0.548).
- **B축(부):** 같은 모델이 calibration도 개선 (❌ 미달: e4_baseline ECE 0.557, conf 0.29 과소확신).

---

## 1. 지금 단계 (where we are)

**A축은 풀렸고 B축이 미해결.** 정확도는 correctness 중심 RLVR로 해결됐다. 남은 문제는
"정답인데 자신 없다"(과소확신) — 모델이 메타인지로 *유용한* 자기평가를 못 하고 confidence를
무조건 낮게 부른다. 현재 단계 = **B축(유용한 메타인지/calibration)을 직접 보상하는 RL 탐색.**

지금까지의 학습 결과(요약, 1030@16k k=8, ECE 포함):

| 실험 | 방식 | acc | ECE | conf | meta emit | 판정 |
|---|---|---|---|---|---|---|
| base_sft_v8 | SFT only (RL 없음) | 0.548 | — | — | 0.91 | 정확도 기준선 |
| **e4_baseline** | VANILLA_GRPO, correctness only | **0.786** | 0.557 | 0.29 | 0.92 | 최고 메타-보존 RLVR, 과소확신 |
| e4_gold_conf_down | RLSD + gold conf teacher | 0.714 | 0.435 | 0.28 | 1.00 | teacher가 정확도 **손해** |
| e4_gold_decoy | RLSD + gold/decoy contrast | 0.588 | 0.295 | 0.30 | 0.98 | 더 큰 손해 |

핵심 교훈: **teacher 기반 contrastive 조형은 정확도를 깎는다**(E.4에서 결정적). 추론 시
steering으로 통하던 효과(E.6b +5.5pp)도 RL 학습으로는 전이되지 않았다. → 학습은 teacher가
아니라 **결과(정답률)에 직접 묶인 보상**이어야 한다.

---

## 2. 진행 중 + 계획된 실험 (의도별 정리)

각 실험을 **의도 / 가설 / 검증 방법**으로 기술한다.

### E.8 — gold-free RLSD (진행 중, gs15 정체)
- **의도:** E.4의 teacher 손해가 "gold(정답) 조건화" 때문인지, contrastive 자체 때문인지
  분리. gold 조건화만 제거한 단일 변수 ablation.
- **가설:** gold 누수를 빼면 teacher가 정확도를 덜 깎고 메타 영역만 건강하게 조형한다.
- **검증:** gs300 도달 → 1030@16k eval로 acc를 e4_baseline 0.786 / conf_down 0.714와 비교.
  baseline 이상이면 "gold 조건화가 주범" 확정, 미달이면 "teacher 방향 자체가 손해" 확정.

### E.9 inject — binned-confidence-injection RLVR (학습 ✅ gs300, eval 대기)
- **의도:** calibration을 *직접* 타깃. 한 문제의 4개 rollout에 confidence를
  0.2/0.4/0.6/0.8로 **강제 주입**하고 proper-score(Brier)로 잘 보정된 bin을 선택하게 해서,
  모델이 "정답률에 맞는 confidence"를 배우게 한다(과소확신 교정).
- **가설:** correctness가 GDPO dominant head로 정확도를 보존하는 동안,
  outcome_calibration(proper-scoring)이 confidence를 정답률에 정렬 → **acc 유지 + ECE 하락.**
- **검증:** gs300 1030@16k eval → e4_baseline(acc 0.786 / ECE 0.557 / conf 0.29) 대비
  **acc ≥ ~0.77 AND ECE ≪ 0.557**이면 calibration 개선 성공. ← **지금 막혀 있는 eval이 이것.**

### E.9 v2 — BCI_RLVR self-emit (진행 중, gs10 정체)
- **의도:** 주입 없이 모델 *자신의* confidence에 proper-score 보상. inject(주입)와의 대조군 —
  강제 주입 없이도 self-emit만으로 calibration이 잡히는지.
- **가설:** self-emit proper-score만으로도 과소확신이 줄지만, 주입(sweep)보다 약할 것.
- **검증:** gs300 eval → inject 결과와 ECE/acc 비교. inject > self-emit이면 "sweep(주입)이
  실제 레버" 확정.

### (계획) 다음 분기 — 의도("유용한 메타인지")로의 정렬
- **calibration이 풀리면:** confidence를 *행동 트리거*로 승격. 즉 낮은 confidence → 검산/전환을
  하고, 그 전환이 **정답을 바꿀 때만** 보상(utility-gated metacognition). 이게 intent의 핵심:
  "유용할 때만 하는 메타인지".
  - **가설:** confidence를 단순 정렬하는 것보다, confidence→행동→정답개선 사슬을 보상하면
    정확도가 추가로 오른다.
  - **검증:** self-correction success rate(=불확실 표명 후 접근 전환했을 때 정답률↑)와
    acc를 동시에 측정, baseline 대비.
- **calibration이 정확도를 깎으면:** calibration weight를 낮추거나 group-Brier로 전환,
  또는 teacher 방향을 접고 outcome 기반 보상에만 집중.

---

## 3. 즉시 블로커 (2026-06-07)

- **e9 inject eval이 8h+ queued** — Basic(preemptible) 풀 만석 + 우리 학습 잡 2개(e8,
  rlvr-v2)가 노드 점유. inject 학습은 끝났으므로(gs300 HF 저장) eval만 돌면 B축 첫 직접 결과.
  레버: 정체된 학습 잡(e8 또는 둘 다) 취소 → 노드 확보. (사용자 승인 대기.)
- **durability gap:** 학습 잡은 cross-node preempt 시 step 0부터 재시작(resume-from-HF 없음).
  → 장기적으로 bootstrap에 HF-resume 추가 필요.

---

## [2026-06-09 추가] TRIOBJ_META_V1 결과 + 다음 분기 = TRIOBJ_DCPO_V2

### TRIOBJ_META_V1 결과 (tri-objective GDPO, env-reward-only)
- **학습 300 step COMPLETED (rc=0)**, gs300 HF 저장 `iamseungpil/metacot-h200-triobj-meta-v1`.
  INLINE auto-eval은 OOM/preempt로 KILL → eval 수치 미확보, 별도 클린 eval 잡 `triobj-eval-gs300` 큐 대기.
- **결정적 실패: `gdpo/meta_revision_utility/mean`이 전 구간 0.0(std 0)** — 두-pass meta-revision 보상이 **한 번도 안 켜졌다.** response_length 924→308, entropy 0.12→0.014 → policy가 meta를 버리고 **terse single-pass**로 수렴. train-val은 easy 보존(algebra ~0.78) / hard 붕괴(geometry ~0.23, omni-math ~0.16).
- **진단 4원인:** (a) 모든 head를 한 advantage로 SUM→전 토큰 균일 broadcast(correctness가 meta 압살), (b) warranted meta ATTEMPT 무보상→"never revise"가 안전 최적, (c) 상속된 `meta_penalty(-0.2)`/`meta_floor(-0.5)` 잔존→net meta 억압, (d) correctness가 terse low-risk 답 선호.
- **의도 관점 판정:** §2의 "다음 분기"가 노린 *utility-gated metacognition*(confidence→행동→정답개선)을 구현하려 했으나, **보상 신호가 토큰 라우팅 없이 합산·broadcast되는 구조 자체가 meta 신호를 0으로 죽였다.** B축도 A축도 못 건드림.
- (참고) **E.8 gold-free RLSD v2:** RUNNING gs290/300(~97%), correctness 양수(~+0.04) 도달.

### 다음 분기 → TRIOBJ_DCPO_V2 (DCPO 3-region token-masked advantage routing)
spec: `docs/superpowers/specs/2026-06-09-dcpo-3region-design.md`
- **핵심 전환:** 세 objective(correctness / meta-utility / calibration)에 **각자의 region-별 group-normalized advantage**를 주고 **자기 토큰 span에만 마스킹** → 한 objective의 gradient가 다른 objective 토큰으로 broadcast되지 않음(DCPO block-wise decoupling을 두-pass `<|meta|>` policy에 적용).
- **3 head:** R_corr(non-meta reasoning+answer), R_meta(meta-content, **warrant-gated flip-credit, keyword gate 없음**), R_cal(conf 토큰, per-instance Brier). KL off(`use_kl_loss=false`, Dr.GRPO 암묵 정규화). guard는 analysis-only.
- **4원인 직접 수리:** (a)→region 라우팅, (b)→`+eps` warrant-gated no-harm bonus + flat +1.0 flip credit, (c)→`meta_penalty`/`meta_floor` DISABLED, (d)→KL/entropy global-mask 재결합 채널 차단.
- **north-star 정합:** correctness가 answer span에서 dominant 유지, meta-utility는 **보장된 난이도에서 정답을 옮기거나 지킬 때만** 보상, calibration은 conf 토큰에 국한된 sub-signal. 어떤 objective도 공유 broadcast advantage로 다른 것을 압살 못 함 = §0 intent("유용할 때만 하는 메타인지로 정확도↑")의 구조적 구현.
