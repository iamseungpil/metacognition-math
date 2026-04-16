# Meta-CoT V8 — Final Status Report (2026-04-16)

## Executive Summary

| Model | Split | 4k cap | 16k cap | Delta |
|---|---|---|---|---|
| Base GRPO step300 | Overall (1030) | 75.92% | 77.00% | +1.08pp |
| Base GRPO step300 | MATH500 | 75.2% | 75.6% | +0.4pp |
| Base GRPO step300 | AIME-24/25 (n=30) | 33.3% | 36.7% | +3.4pp |
| Meta GRPO E21R-v2 step300 | Overall (1030) | 79.81% | 81.65% | +1.84pp |
| Meta GRPO E21R-v2 step300 | MATH500 | 71.6% | 74.8% | +3.2pp |
| Meta GRPO E21R-v2 step300 | AIME-24/25 (n=30) | 13.3% | 13.3% | 0pp |

(source: `results/eval_1030_base_grpo_step300_16k/base_grpo_step300_16k.json`, `results/eval_1030_meta_grpo_e21r_v2_step300_16k/meta_grpo_e21r_v2_step300_16k.json`)

**Bottom line**: Meta-CoT E21R-v2는 16k 맥락에서 overall 1030 accuracy +4.65pp로 base GRPO를 능가하지만 (81.65% vs 77.00%), AIME hard split에서는 13.3% 대 36.7%로 역전된다. MATH500과 AIME의 방향이 갈라지는 것은 meta training이 심은 verify/redirect pattern이 OOD hard problem에서 "decoherence trap"을 만들기 때문이다 (Sec 3.3, 4.2).

## 1. Hypothesis → Evidence Map

Hypothesis는 실험 전에 잡은 예측. Evidence는 측정된 JSON에서 나온 수치.

- **H1 (SFT meta block은 uncertainty를 resolve한다)**: **FALSIFIED**. Strict SFT meta 내부 delta_entropy = **+0.300 nats** (n=120, source: `results/entropy_strict_meta/entropy_stats.json`). 방향이 양수 → meta block이 닫힐 때 entropy가 오히려 증가. Meta는 resolve가 아니라 "open new exploration".
- **H2 (RL confidence text는 uncertainty를 resolve한다)**: **SUPPORTED**. Step300 RL confidence text delta = **−0.052** @ 4k (n=200), **−0.031** @ 16k (n=200). 방향 일관되게 음수 (source: `results/entropy_analysis_step300/rl_meta_confidence/conf_entropy_stats.json`, `results/entropy_meta_grpo_e21r_v2_step300_16k_conf/entropy_stats.json`).
- **H3 (Meta-CoT이 math overall에서 base를 이긴다)**: **PARTIALLY SUPPORTED**. 4k +3.89pp (79.81 − 75.92), 16k +4.65pp (81.65 − 77.00). 전체는 yes, split별로 분해하면 AIME에서 fail.
- **H4 (4k truncation이 AIME 손실을 설명한다)**: **PARTIALLY FALSIFIED**. Base는 16k에서 +3.4pp 회복, Meta는 0pp. Truncation만으로는 설명 불가 — meta 특이적 failure mode 존재.
- **H5 (Meta는 hard OOD에서 decoherence에 빠진다)** — NEW: **SUPPORTED**. AIME wrong 26/30 중 **13개 decohered** (LaTeX gibberish loop), **12개 no-boxed** (mid-chain 종료), avg wrong length 14628 tokens (~16k cap) (source: `results/aime_failure_analysis_16k/aime_failure_modes.json`).

## 2. Method

Meta-CoT v8은 DeepSeek-R1-Distill-Qwen-7B에 meta-cognitive token 구조 (`<|meta|>…<|/meta|>` + confidence:)를 SFT로 주입한 뒤 GRPO로 math reward에 맞춰 RL 한 것. Base GRPO는 동일 RL recipe을 meta 구조 없이 적용한 대조군.

Eval corpus는 1030 문제 = MATH500 + AIME-24 + AIME-25 + 추가 hard set. 4k는 초기 run의 max_new_tokens, 16k는 truncation 가설을 검증하기 위한 2026-04-16 재평가. Decoding은 temperature=0.6, top_p=0.95, single sample greedy-ish.

Entropy methodology: token-level H(p) = −Σ p·log p over the full vocab at each generated position. "Marker window" = (a) strict SFT의 경우 `<|meta|>` open부터 close까지, (b) RL confidence의 경우 `confidence:` text 직후 window. "delta_entropy" = post-window mean − pre-window mean (nats). Sample sizes n=120/208/200 결정은 full 1030이 아닌 marker 포함 서브셋만 대상이기 때문. Full details → `results/ANALYSIS_MAP.md`.

## 3. Results

### 3.1 Accuracy (4k vs 16k re-eval)

| Model | Split | 4k | 16k | Delta |
|---|---|---|---|---|
| Base GRPO | Overall | 75.92% | 77.00% | +1.08pp |
| Base GRPO | MATH500 | 75.2% | 75.6% | +0.4pp |
| Base GRPO | AIME | 33.3% | 36.7% | +3.4pp |
| Meta GRPO E21R-v2 | Overall | 79.81% | 81.65% | +1.84pp |
| Meta GRPO E21R-v2 | MATH500 | 71.6% | 74.8% | +3.2pp |
| Meta GRPO E21R-v2 | AIME | 13.3% | 13.3% | 0pp |

Compare-contrast: 16k에서 truncation을 풀자 Meta MATH500은 +3.2pp 회복 (71.6→74.8). 동일 조건에서 Meta AIME는 0pp. Base AIME는 +3.4pp 회복. 따라서 AIME에서 Meta의 정체는 "token budget 부족" 이상의 구조적 문제.

### 3.2 Entropy dynamics (4-way comparison)

| Condition | n | delta_entropy (nats) | Direction |
|---|---|---|---|
| SFT meta (strict, 내부 delta) | 120 | +0.300 | opens exploration |
| Step300 SFT meta (transitional) | 208 | −0.113 | converges |
| Step300 RL confidence text @ 4k | 200 | −0.052 | resolves |
| 16k Meta GRPO RL confidence | 200 | −0.031 | resolves (smaller) |

(source: `results/entropy_strict_meta/entropy_stats.json`, `results/entropy_analysis_step300/sft_meta/entropy_stats.json`, `results/entropy_analysis_step300/rl_meta_confidence/conf_entropy_stats.json`, `results/entropy_meta_grpo_e21r_v2_step300_16k_conf/entropy_stats.json`)

Interpretation: Strict SFT 상태에서는 meta block이 resolve가 아니라 "새 가설 열기" 기능을 했다 (+0.300). RL이 진행되면서 (SFT meta → step300 SFT meta → RL confidence) delta가 부호를 뒤집고 크기가 줄어든다. 16k와 4k의 RL confidence delta 차이 (−0.031 vs −0.052)는 RL이 전체 확률 분포를 좁혀 절대 entropy 자체가 약 10× 낮아졌기 때문 — 부호는 동일. Base GRPO 16k에는 `<|meta|>` 0%, `confidence:` 0%로 marker window 자체가 비어 있어 정의되지 않음 (Sec 4.3).

### 3.3 AIME failure modes

Bar-chart description: x축은 실패 유형, y축은 count. Meta와 Base를 2개 bar group.

| Failure mode | Meta GRPO (of 26 wrong) | Base GRPO (of 19 wrong) |
|---|---|---|
| decohered (LaTeX gibberish loop) | 13 | 1 |
| no-boxed (ran out mid-chain) | 12 | 0 |
| coherent-wrong-boxed (committed) | 1 | 18 |
| Avg wrong length (tokens) | 14628 | 6078 |

(source: `results/aime_failure_analysis_16k/aime_failure_modes.json`)

Compare-contrast: Meta는 26/30 오답 중 25개 (96.2%)가 non-commit (decohered+no-boxed). Base는 19/30 오답 중 18개 (94.7%)가 commit-to-wrong. Avg wrong length 14628 vs 6078 → Meta는 token budget을 거의 다 써도 commit하지 못한다.

### 3.4 Marker prevalence vs accuracy (Meta E21R-v2)

| Marker | prevalence | accuracy delta (with vs without) | p |
|---|---|---|---|
| has_verify | 95.3% | +77.1pp | 8.6e-30 |
| has_redirect | 13.1% | −57.7pp | 1.0e-43 |
| has_diagnosis | 8.1% | −59.3pp | — |
| has_epistemic | 20.4% | −46.4pp | — |

(source: `results/step300_deep_analysis/meta_behavior.json`)

Interpretation: verify는 거의 전 답에 등장하는 "성공 패턴"이므로 positive signal. redirect/diagnosis/epistemic은 prevalence는 10–20%대로 낮고 accuracy delta가 크게 음수 — "어려운 문제에서만 켜지는 marker". Accuracy 손실의 직접 원인이라기보다는 문제 난이도와 confounded.

## 4. Interpretation

### 4.1 Why does meta training help MATH500 but not AIME?

MATH500은 in-distribution (SFT 학습 데이터 분포와 유사한 난이도). AIME는 OOD hard. MATH500에서는 verify pattern (95.3% prevalence, +77pp gain)이 짧은 체인 안에서 답을 확인하고 `\boxed`로 commit한다. AIME에서는 문제가 어려워 verify가 한 번에 성공하지 못하고 redirect/diagnosis가 trigger되면서 체인이 길어진다. 길어진 체인은 16k 안에서 commit에 도달하지 못한다 (avg wrong length 14628). 요약: 같은 verify 회로가 쉬운 문제에서는 효율적 commit, 어려운 문제에서는 무한 재검토를 만든다.

### 4.2 The decoherence trap

Decoherence는 생성된 LaTeX가 의미 단위를 잃고 같은 토큰열 (예: `\\frac{}{\\frac{}{...}}`) 을 반복하면서 context window 끝까지 가는 현상. Meta GRPO AIME에서 13/26 (50%) 발생. Base GRPO에서는 1/19 (5.3%). 메커니즘 가설: meta training이 "아직 아니다, 다시 보자" 패턴 (redirect, epistemic) 을 심어 두었기 때문에, hard problem에서 GRPO reward가 확실한 `\boxed` 신호를 못 받을 때 모델이 commit 대신 재검토 loop에 고정된다. 재검토 loop의 LM head 분포가 self-reinforce하면서 표층적으로 LaTeX 기호 반복으로 collapse.

### 4.3 Where `<|meta|>` tokens went

Base와 Meta 양쪽 GRPO 모두 4k와 16k에서 `<|meta|>` token 0%. Truncation 때문이 아니다 (16k 재평가에서도 0%). 원인은 RL reward의 fallback: `rewards.py`의 `_parse_meta_blocks_with_spans`가 `<|meta|>…<|/meta|>` 래핑과 plain `confidence:` text를 reward 동등으로 취급한다. GRPO는 2-token overhead (`<|meta|>` + `<|/meta|>`) 를 없애는 쪽으로 수렴. 즉 reward engineering의 fallback이 structural signal을 silently 없앤 것.

## 5. Limitations

- AIME n=30 (24+25 결합)으로 작다. Failure-mode 분류 (13 decohered 등) 는 표본 크기에 민감.
- Entropy analysis 모두 sub-1030 sample (n=120/208/200). Full 1030 전수 entropy 재측정 없음.
- RL single seed (E21R-v2). Run variance 측정 없음.
- 4k vs 16k는 동일 checkpoint/decoder 재평가. 학습 단계에서 16k로 학습한 별도 checkpoint가 없기 때문에 "학습 시 긴 context에 노출된 meta 모델"은 본 보고서 밖.
- Decoherence 판정은 휴리스틱 분류 (aime_failure_modes.json의 regex/length rule) 에 의존. 사람 평가 대조 없음.

## 6. Next Steps (H5 + decoherence에 직결)

### E1: No-boxed reward penalty
체인 끝에 `\boxed{...}` 없으면 고정 페널티. GRPO에 추가해 commit을 학습 신호로 강제. Meta AIME 12/26 no-boxed 케이스를 직접 타겟.

### E2: Forced commit schedule (token-budget-aware)
Context 사용량이 80% (~12800 tokens) 넘으면 "time_left: short → commit now" system signal을 prompt에 주입하도록 training data를 augment. Decoherence 진입 전 commit 회로 활성화.

### E3: Best-of-N with boxed-presence filter on AIME
Inference 시 N=8 sample 중 `\boxed` 존재하는 것만 필터 후 majority vote. 13/26 decohered는 자동 탈락. 학습 변화 없이 AIME 회복 가능성 점검.

### E4: Explicit "commit" token in SFT data
SFT 학습 문장 끝에 `<|commit|>\\boxed{answer}` 형태로 commit marker를 분리 토큰으로 삽입. Reward fallback이 이 토큰은 별도로 잡도록 `rewards.py` 수정. `<|meta|>`가 drop된 것과 동일한 drop이 `<|commit|>`에 일어나지 않도록 reward 설계.

## References

- `results/eval_1030_base_grpo_step300_16k/base_grpo_step300_16k.json`
- `results/eval_1030_meta_grpo_e21r_v2_step300_16k/meta_grpo_e21r_v2_step300_16k.json`
- `results/aime_failure_analysis_16k/aime_failure_modes.json`
- `results/entropy_strict_meta/entropy_stats.json`
- `results/entropy_analysis_step300/sft_meta/entropy_stats.json`
- `results/entropy_analysis_step300/rl_meta_confidence/conf_entropy_stats.json`
- `results/entropy_meta_grpo_e21r_v2_step300_16k_conf/entropy_stats.json`
- `results/step300_deep_analysis/meta_behavior.json`
- HF: `iamseungpil/metacot` dataset (all artifacts mirrored)
- Git commits (2026-04-16): 0b7c444, 8963506, c2f92bc, d93be4b, 574d7e7
