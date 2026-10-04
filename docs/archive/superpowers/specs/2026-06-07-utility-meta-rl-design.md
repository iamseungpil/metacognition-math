# Spec: Tri-Objective Metacognition RL (correctness + calibration + meta-quality) — 2026-06-07

## 한 줄
**하나의 RL에서 세 목표를 token-masked로 분리해 같이 학습한다: (1) correctness, (2) calibration,
(3) meta-quality.** 셋은 병렬이 아니라 *인과 사슬*로 묶인다 — calibration이 confidence를 믿을
trigger로 만들고 → 낮은 confidence가 meta 행동(검산/전환)을 부르고 → meta가 답을 개선(correctness).

## 왜 (진단)
inject 실패 = correctness에 Brier를 *결합*해 공유 토큰에서 gradient 충돌(2603.09117) + 길이/붕괴
편향(Dr.GRPO) + meta 채널 붕괴. calibration을 *목적*으로 두면 "낮은 conf+안정적 오답"으로 게이밍됨.

## 선행연구 위치 (novelty)
- **2603.09117 (Decoupling Reasoning & Confidence)**: correctness+calibration **2개**를 block-wise
  token-masked advantage로 분리(A_r=정확도, A_c=−|conf−RIG|, RIG=λ·group_acc+(1−λ)·instance).
  우리는 이걸 **meta-quality 3번째 축으로 일반화**.
- **MaR (2605.23384)**: correctness + meta 3차원, 그러나 calibration 없음·decoupling 없음.
- **2602.22751 (Metacognitive Entropy Calibration)**: 개념상 3-way 최근접(entropy 기반) → **novelty
  주장 전 정독 필수.**
- 결합 안티해킹: **PROF (2509.03403)** = process 보상은 gradient에 *섞지 말고* outcome에 *곱해
  rank/filter*; **PRM은 hackable fluency detector (2603.06621)** — presence 보상 금지;
  **SCoRe (2409.12917)** = Δ보상 α·(r(y₂)−r(y₁))+KL warmup; **Reflect-Retry-Reward (2505.24726)** =
  reflection 토큰만, retry 성공 시에만 보상(답은 보상 X → 일반화); **min-form/HRM** = 정정이
  앞 오류 크레딧 받게(backtrack을 오류로 벌하지 않음); **adaptive Lagrangian length penalty**;
  **steered/CAA expert iteration** = steer로 좋은 trajectory 생성→outcome 필터→가중치 distill.
→ 결론: **full 3-way + token-masked decoupling은 미발표(novel).**

---

## 통합 설계 — 세 축을 한 rollout/한 손실로

출력 = `[reasoning … 답₁(boxed)] <meta> [검토/전환 + conf] [최종 답₂(boxed)] <conf> [conf 스칼라]`.
세 token span에 **각자의 advantage만** 마스킹(2603.09117을 3-way로 확장):

1. **correctness (gradient, dominant)** — 답 span. GRPO 그룹 정규화, 1/0. 정확도 보호.
2. **calibration (gradient, decoupled)** — `<conf>` 스칼라 토큰에만. `R_c=−|conf−RIG|`,
   RIG=λ·group_acc+(1−λ)·instance_correct (closed-form, 추가 rollout 0). DCPO식.
3. **meta-quality (Δ-보상, 안티해킹)** — 검토 span. **presence가 아니라 Δ로**:
   `R_m = clip[ α·(correct(답₂) − correct(답₁)) ]` (two-sided: wrong→right +큰, right→wrong −큰).
   추가로 **outcome-곱**(PROF): 효과 없으면 ~0, 확신 높은데 검토하면 −ε(over-check). conf₁이 낮을
   때 검토하면 trigger 가점(uncertainty_meta). **답 토큰엔 절대 meta 보상 안 감(마스킹).**
   답₁ vs 답₂는 **한 생성 안에서** 파싱 → 추가 rollout 없음.

## 학습 위생 (붕괴 방지)
Dr.GRPO loss(길이/오답-길이 편향 제거) · clip-higher + KL-to-ref anchor · **warmup 후 meta 보상**
(SCoRe; cold에 바로 얹으면 no-op 붕괴) · dynamic sampling(전부정답/전부오답 그룹 드롭) ·
anti-decoherence(box/commit + adaptive length penalty + truncated 마스킹).

## hard 문제 무신호 격파 (선택) — Steered Expert-Iteration
AIME처럼 그룹 전부 오답이면 advantage=0. **오프라인(HF)** 에서 steering(검증된 +5.5pp, conf-down
contrastive)으로 rollout 생성 → 성공·Δ-positive trajectory만 필터 → RFT/RL 데이터로 **메타인지를
가중치에 distill**(logit 매칭 아님 → E.4 teacher 실패 회피). steering은 HF/이중컨텍스트/eval-only라
rollout 루프 밖에서만.

## 단계 (staging)
- **S1**: correctness + calibration(2축, token-masked) + 위생. inject 출혈 정석 수리, decoupling 머신 검증.
- **S2**: meta-quality Δ-보상(3번째 span) 추가 — 의도(meta로 정확도↑) 정면.
- **S3(선택)**: steered-EI로 hard 문제 탐색 보강.

## 진단지표 (착시 차단; ECE 단독 금지)
AUROC(conf,correct)=discrimination · accuracy-stratified ECE · Pass@k · 난이도층화 acc ·
truncated-no-box율 · clip_ratio/entropy 추세 · self-correction success(wrong→right flip)율 ·
counterfactual meta-utility(eval: meta 제거 후 재디코딩 Δ).

## 구현 (Karpathy minimal-change, additive)
새 모드 `TRIOBJ_META`를 REWARD_CONFIGS에 additive(off시 기존 byte-identical) · 새 로직은
`src/training/meta_utility.py` · 탐지 헬퍼는 rewards.py에서 import만 · token-masked multi-advantage는
verl advantage부에 span-mask 곱(gated) · step-1 smoke green 후 train→eval→push 파이프라인 재사용.

## 통제군
e4_baseline(0.786/ECE0.557) · inject(0.609/0.083/meta0.41) · 진행중 e8/e9-v2.

## 열린 결정
S1→S2→S3 순차 vs S1+S2 동시 / meta Δ를 gradient(SCoRe) vs rank-filter(PROF) / λ·KL·warmup·α 튜닝.
