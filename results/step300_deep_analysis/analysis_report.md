# E21R-v2 Step 300 분석 보고서 (수정본)

Date: 2026-04-15 (verified)

## 1. 실험 구성

- **E21R-v2**: Qwen3-8B → v8_meta_inside_strict SFT → 2-head GDPO (correctness + outcome_calibration), 300 steps
- **Base**: Qwen3-8B → v8_base_matched_strict SFT → GRPO (correctness only), 300 steps
- **평가**: 1030 문제 (GSM8K 500 + MATH500 500 + AIME 30), vLLM TP=4, temp=0.7

## 2. 전체 성능

| Metric | E21R-v2 | Base | Delta |
|--------|---------|------|-------|
| Overall accuracy | 79.81% (822/1030) | 75.92% (782/1030) | **+3.88pp** |
| GSM8K | 92.0% (460/500) | 92.6% (463/500) | -0.6pp |
| MATH500 | 71.6% (358/500) | 61.8% (309/500) | **+9.8pp** |
| AIME | 13.3% (4/30) | 33.3% (10/30) | **-20.0pp** |

## 3. Paired Analysis (1030 matched questions)

| Category | Count | Rate |
|----------|-------|------|
| Meta-only wins | 117 | 11.4% |
| Base-only wins | 77 | 7.5% |
| Both correct | 705 | 68.4% |
| Both wrong | 131 | 12.7% |
| **Net meta advantage** | **+40** | **+3.88pp** |

Per-benchmark:
- GSM8K: meta=14 base=17 (net -3)
- **MATH500: meta=102 base=53 (net +49)** — 핵심 이득 원천
- **AIME: meta=1 base=7 (net -6)** — 핵심 손실 원천

## 4. Meta Content 형태 분석

### `<|meta|>` wrapping 소실 원인 (verified)

`<|meta|>` (token ID 151669), `<|/meta|>` (token ID 151670)은 tokenizer에 존재하고, `special: false`라 `skip_special_tokens=True`에서도 제거되지 않음.

**근본 원인**: `rewards.py`의 `_parse_meta_blocks_with_spans`에 3단계 fallback 존재:
1. `<|meta|>...<|/meta|>` regex
2. `[META]...[/META]` text markers
3. **free-text `confidence: X.XX` 패턴** ← 이것 때문

Reward 시뮬레이션 (verified):

| 패턴 | corr | cal | floor | combined |
|---|---|---|---|---|
| `<\|meta\|>confidence: 0.9<\|/meta\|>` + 정답 | 1.00 | 0.27 | 0.00 | **1.27** |
| `confidence: 0.9` (plain text) + 정답 | 1.00 | 0.27 | 0.00 | **1.27** |
| confidence 없음 + 정답 | 1.00 | 0.00 | -0.50 | **0.85** |

wrapping 유무가 **동일 reward** → RL이 `<|meta|>` 토큰을 2-token overhead로 인식하고 drop. 버그가 아닌 RL 최적화 결과.

### 실제 출력 구조

```
<think>
... 풀이 ...
</think>

confidence: 0.96
assessment: the algebra is clean; a numerical substitution will confirm
I should double-check the arithmetic before finalizing.

</think>

Numerical spot-check: picking a specific case and tracing through gives X.
</think>

The answer is ...
```

- `</think>` 3회 패턴이 810/1030 (78.6%)에서 출현 → accuracy 89.0%
- `</think>` 0회 = token limit 도달 → accuracy 9.4%

### Meta content 통계

| Metric | E21R-v2 | Base |
|--------|---------|------|
| Free-text confidence 포함 | 920/1030 (89.3%) | 2/1030 (0.2%) |
| Confidence value = 0.96 (exact) | 910/920 (98.9%) | 0 |
| Assessment text 포함 | 918/1030 (89.1%) | 0 |
| 동일 assessment template | 908/918 (98.9%) | N/A |
| study_need | 0% | 0% |
| `<\|meta\|>` wrapping | 0% | 0% |

**Template collapse**: 908/1030 completions에 동일 문구 `"the algebra is clean; a numerical substitution will confirm"` 포함.

## 5. Calibration 분석

### Confidence 분포 (reparsed, verified)

| Bucket | N | Accuracy | Avg Conf |
|--------|---|----------|----------|
| <0.3 | 8 | 37.5% | 0.154 |
| 0.3-0.5 | 4 | 75.0% | 0.385 |
| 0.5-0.7 | 7 | 57.1% | 0.583 |
| 0.7-0.9 | 1 | 100% | 0.798 |
| >0.9 | 900 | 88.6% | 0.960 |

**Overall ECE**: 0.077 | **AUROC**: 0.522

비-0.96 값 20개에서는 **confidence와 accuracy 사이 방향성 존재** (low conf → low acc). 그러나 99% 이상이 0.96으로 붕괴되어 실용적 구분 불가.

### Confidence 0.96 수렴 이유

Calibration reward는 **active**했음 (free-text fallback으로 작동). 수렴 원인:

E[cal] = accuracy × (+0.3 × c) + (1-accuracy) × (-0.3 × c) = 0.3c × (2 × accuracy - 1)

accuracy = 80%일 때: E[cal] = 0.18c → **c가 클수록 reward 높음** → c → 0.99 수렴.

이는 calibration reward 설계의 구조적 결함: accuracy > 50%이면 높은 constant confidence가 항상 최적.

## 6. Behavioral Marker 분석

### Fisher Exact Test (verified, p-values)

**E21R-v2:**

| Benchmark | Marker | N(+) | Acc(+) | Acc(-) | p-value | Sig |
|---|---|---|---|---|---|---|
| overall | verify | 982 | 83.4% | 6.2% | <0.001 | * |
| overall | redirect | 135 | 29.6% | 87.4% | <0.001 | * |
| overall | diagnosis | 83 | 25.3% | 84.6% | <0.001 | * |
| overall | epistemic | 210 | 42.9% | 89.3% | <0.001 | * |
| MATH500 | redirect | 107 | 32.7% | 82.2% | <0.001 | * |
| AIME | redirect | 23 | 4.3% | 42.9% | 0.031 | * |

**해석 주의사항**:
- **verify 95.3% rate는 boilerplate template** (908/1030에 동일 문구). verify 없는 48개 중 85%가 token-truncated completions. verify/accuracy 상관은 "검증 행동 = 정확" 이 아니라 **"완성된 응답 = 정확"**.
- **redirect/diagnosis가 accuracy 하락과 연관되는 건 Base에서도 동일** (Base redirect: 32.9% vs 79.5%, p<0.001). 즉 meta-CoT 특유가 아닌, 어려운 문제에서 발동하는 보편적 신호.

### Trigger-Conditioned Correction Rate (Plan RQ1 metric 4)

Meta redirect가 발동한 135개 문제:
- **Meta accuracy: 29.6%**
- **Base accuracy (같은 문제): 51.9%**
- **Delta: -22.3pp** → redirect가 같은 난이도에서 base보다 더 못함

## 7. has_meta Impact (reparsed, verified)

| Group | N | Accuracy | Avg Tokens | Tok/Correct |
|---|---|---|---|---|
| has_meta (confidence 포함) | 920 | **87.8%** | 359 | 409 |
| no_meta (confidence 없음) | 110 | 12.7% | 3768 | 29606 |

Per-benchmark:

| Bench | has_meta Acc | no_meta Acc | has_meta N | no_meta N |
|---|---|---|---|---|
| GSM8K | 92.9% | 0.0% (n=5) | 495 | 5 |
| MATH500 | 82.5% | 16.9% (n=83) | 417 | 83 |
| AIME | 50.0% | 0.0% (n=22) | 8 | 22 |

no_meta = token limit (4096)에 도달하여 meta block까지 생성 불가 → 대부분 오답.

## 8. AIME -20pp 원인 (verified)

| Metric | Meta | Base |
|---|---|---|
| Token-maxed (≥4090) | 22/30 (73%) | 13/30 (43%) |
| Wrong + token-maxed | 22/26 (85%) | - |
| Wrong + has_redirect | 22/26 (85%) | - |
| Correct avg tokens | 1516 | 1096 |
| Wrong avg tokens | 3798 | 3356 |

5/7 base-only AIME wins에서 meta가 4096 token limit에 도달. 메커니즘:
1. 어려운 문제 → redirect 시도
2. 새 접근법 시도 → 토큰 소모
3. 4096 limit → `\boxed{}` 없이 절단

## 9. MATH500 +9.8pp 원인

| Pattern | Meta tokens | Base tokens | N |
|---|---|---|---|
| Meta-only wins | avg 648 | avg 533 | 102 |
| Base-only wins (meta side) | avg 2760 | avg 1132 | 53 |

Meta-only win: 82%에서 meta가 base보다 더 많은 토큰 사용 (avg +115 tokens). 추가 reasoning이 정답 도달에 기여. Meta-only win 중 65%는 `</think>` x3 (정상 구조), 29%는 x4 (extra step).

## 10. RQ Verdicts

### RQ1: Meta-CoT Controller Learning
- **Controller acquisition**: Weakly supported. 89% meta content emission, but 99% 단일 template collapse
- **Controller depth**: NOT supported. Confidence → 0.96 constant, study_need 0%, assessment 단일 template
- **Controller utility**: Mixed. MATH500 +9.8pp (longer reasoning), AIME -20pp (token budget failure)

### RQ2: Meta-RL Strengthening
- **Verify**: NOT clearly supported as "verification behavior". 95.3% rate = boilerplate template, not adaptive
- **Redirect execution**: NOT supported. Redirect → 29.6% acc (same problems base = 51.9%)
- **Calibration**: Active but structurally flawed. accuracy > 50% → high confidence always optimal → collapse
- **Overall**: RL produced meta content emission but in collapsed single-template form. Net accuracy gain (+3.88pp) comes from longer reasoning on MATH500, not from metacognitive control

### Core Finding
> RL은 meta-like content를 89%까지 학습시켰지만, 이는 **단일 boilerplate template** (confidence: 0.96 + 동일 assessment)으로 수렴했다. `<|meta|>` 구조 토큰은 reward-neutral이라 RL이 drop. Confidence는 calibration reward의 구조적 결함으로 0.96 constant로 수렴. +3.88pp accuracy gain은 MATH500에서의 longer reasoning에서 오지, metacognitive control에서 오지 않는다.

## 11. 설계 교훈

1. **Reward fallback이 구조 토큰을 무력화**: `_parse_meta_blocks`의 free-text fallback이 `<|meta|>` wrapping의 reward value를 0으로 만듦
2. **Calibration reward의 수학적 결함**: E[cal] = 0.3c×(2×acc-1), acc>50%이면 c→1이 최적. Proper scoring rule (Brier, log-score) 필요
3. **Template collapse 방지 필요**: RL이 단일 최적 template으로 수렴. Diversity reward 또는 per-problem adaptive reward 필요
4. **Token budget 관리**: AIME 손실은 redirect의 토큰 소모 때문. Max redirect attempts 또는 early commitment 메커니즘 필요

## Appendix: 이전 보고서의 오류 정정

| 이전 주장 | 실제 | 유형 |
|---|---|---|
| `<\|meta\|>` 소실 = vLLM 디코딩 문제 | RL 최적화로 자연 drop (reward 동일) | 원인 오해 |
| Calibration reward 비활성 | Free-text fallback으로 active | 원인 오해 |
| Verify 95.3% = "검증 행동 강화" | Boilerplate template 채택 (908/1030 동일 문구) | Confound |
| AIME meta correct avg 1266 tokens | 1516 tokens | 수치 오류 |
| Base correct avg ~1900 tokens | 1096 tokens | 수치 오류 |
| Confidence histogram [0.0-0.3]: 7 | 6 | 수치 오류 |
| Confidence histogram [0.9-1.0]: 912 | 911 | 수치 오류 |
| Delta +3.89pp | +3.88pp (exact: 3.8835pp) | 반올림 오류 |
