# Brainstorm: 왜 의도대로 안 됐나 + 다음 설계 (2026-06-07)

## 의도
**유용한 메타인지를 강화해 정확도를 올린다.** calibration은 부분 목표(유용한 메타인지의 신호).

## 무엇이 문제였나 — 3개 메커니즘이 겹침 (전부 선행연구로 확인)

1. **Gradient 충돌 (DCPO, arXiv:2603.09117).** correctness에 Brier를 *결합*해 공유 파라미터에서 학습하면
   정확도 gradient와 calibration gradient가 반대로 당김. hard 문제에선 푸는 것보다 **conf를 낮추고
   커밋을 멈추는 게** joint reward를 올리는 더 싼 길 → 정확도 출혈. RLCR도 동일하게 AIME 40%→32.8%.
2. **길이/붕괴 편향 (Dr.GRPO 2503.20783, DAPO 2503.14476).** GRPO per-response 길이 정규화가
   **오답일수록 길이를 덜 벌함** + commit/format 보상 없음 + truncated=0보상 → hard에서 한도까지 헛돎.
   우리 추세(len·clip_ratio 2배)가 교과서적 시그니처.
3. **채널 붕괴 + utility 신호 부재.** calibration 크레딧은 meta 블록 안에서만 생기는데 meta 발화를
   *보호*하는 보상도, meta *유용성*을 보는 보상도 없음 → 모델이 Brier 벌 피하려 meta를 버리고(0.92→0.41)
   남은 meta는 cosmetic. **의도(유용한 메타인지)는 애초에 보상에 없었고, "보정된 숫자 emit"만 있었다.**

## 더 깊은 개념적 오류
**calibration을 목적으로 삼은 것 자체가 함정.** 낮은 conf + 안정적 오답 = 완벽히 calibrated → 나쁜 모델로도
trivially 달성 가능. 의도의 non-gameable 버전은 **메타인지를 그 *인과적 정확도 효과*로 보상**하는 것뿐
(utility-gating). 선행연구가 이 형태를 검증: SCoRe(Δ-correctness 보상), Reflect-Retry-Reward(retry가
fail→success 뒤집을 때만 reflection 보상), Self-Verification Dilemma(검증 과보상은 정확도↓).

## 진단 방법 (조기 탐지 + 착시 차단)
- **discrimination > calibration**: AUROC(conf, correct). ECE↓인데 AUROC 평평 = gaming.
- **truncated-no-box 비율 / clip_ratio / policy entropy**를 1급 학습 대시보드로 (step~20에 붕괴 포착, 300 아님).
- **Pass@k + 난이도층화 acc** (easy↑/hard↓ = diversity collapse, Yue 2504.13837).
- **counterfactual meta-utility probe**: meta 블록을 제거하고 재디코딩 → 정답이 바뀌나? meta가 인과적인지
  cosmetic인지 측정 (보상으로도 사용 가능).
- **gradient-conflict probe**: correctness-loss grad와 calibration-loss grad 코사인(음수면 분리, DCPO).

## 다음 실험 후보 (scaled)
- **D1 — 붕괴 수리(최소)**: plain GRPO(correctness, 검증된 0.786) + anti-decoherence(DAPO soft-overlong +
  box-format 보상 + truncated 마스킹) + Dr.GRPO loss. 목표: 정확도 회복 + truncation 제거. calibration/meta 없음.
  decoherence 진단을 직접 검증하는 싼 baseline-repair.
- **D2 — utility-gated metacognition(의도 정면)**: correctness(dominant) + **two-sided utility 보상**(meta-action이
  wrong→right면 +, right→wrong면 −, counterfactual/2-attempt로 측정) + anti-decoherence. emission 보너스 X,
  Brier X. "유용한 메타인지→정확도"를 직접 인코딩. (Reflect-Retry-Reward/SCoRe 형태.)
- **D3 — decoupled calibration(숫자도 원하면)**: DCPO식 — correctness는 답 토큰, Brier는 conf 토큰에만,
  advantage 마스킹 + emission floor. 정확도 출혈 없이 calibration. D2와 결합 가능.

## 권장
**D1의 anti-decoherence를 D2에 흡수한 형태**가 의도에 가장 부합 — 또는 D1을 먼저 빠르게 돌려 decoherence
진단을 확정한 뒤 D2로. calibration 숫자가 꼭 필요하면 D3를 D2에 얹음. 진행 중인 e8(teacher)/e9-v2(self-emit)
결과가 "붕괴가 주입 탓인지 Brier 일반 탓인지"를 추가로 가려줌 — 그 결과도 설계 입력으로 사용.
