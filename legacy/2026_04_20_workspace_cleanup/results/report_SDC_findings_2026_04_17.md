# Meta-CoT / EAD / SDC 연구 라인 통합 상태 보고서

**날짜**: 2026-04-17
**대상 독자**: 이 프로젝트를 처음 보는 동료 연구자
**범위**: Meta-CoT V8 관찰에서 출발해 EAD(Epistemic Alignment Distillation) 프레임을 거쳐 SDC(Split-Directional Contrastive) 에 이르는 경위·현상·이론·방법·검증·위험·다음 행동
**약어**: EV = epistemic verbalization, EAD = Epistemic Alignment Distillation, SDC = Split-Directional Contrastive, BU = Four Habits Behavior-Uncertainty working note, BoN = best-of-N, PPO = proximal policy optimization, GRPO = group-relative PPO, RLSD = reinforcement learning with self-distillation, RLVR = reinforcement learning with verifiable reward, SFT = supervised fine-tuning

---

## 1. 경위

이 연구 라인은 Qwen3-8B + MATH/AIME/GSM8K 위에 `<|meta|>` 토큰 기반 metacognitive controller 를 SFT 로 심고 그 위에 RL 로 정확도를 끌어올리는 표준 두 단계 recipe 에서 시작했다. Meta SFT 는 base SFT 대비 1,030 문제 평균 정확도를 75.92% 에서 79.81% 로 +3.88 pp 올렸고, meta 마커 전후 5-토큰 창의 평균 엔트로피 차이 ΔH 는 +0.300 nats, wrap rate 은 100% 를 보였다 (§3, paper 03_problem_setup.tex). 이 지점까지는 예측 가능한 성공이었다.

첫 전환점은 RL 300 step 이후 나타난 controller 붕괴였다. 같은 checkpoint 위에 GDPO + correctness + calibration + overconfidence 계열 reward 를 300 step 돌리자 정확도는 유지됐지만 내부 신호 네 개가 같은 방향으로 반전됐다. Wrap rate 이 88.2% 로 떨어지고, stated confidence 가 0.96 한 값에 98.9% 몰리고, AIME 응답의 13/30 (43.3%) 이 16k 토큰 budget 안에서 `\boxed{}` 로 닫히지 못했으며, ΔH 부호가 −0.052 nats 로 뒤집혔다 (HF results/entropy_analysis_step300/rl_meta_confidence/conf_entropy_stats.json 의 delta_entropy = −0.05229). 이 네 지표가 독립이 아니라 한 방향으로 동시에 움직인다는 사실이 "단일한 구조적 현상" 가설의 관찰적 근거가 됐다.

두 번째 전환점은 naive 자기증류 경로 (D1, D2, D2 rebuilt) 가 controller 구조만 되살리고 정확도를 회귀시킨다는 발견이었다. D2 rebuilt 는 ΔH 를 +0.231 nats 로 복원했지만 AIME 정확도는 6.7% 까지 떨어졌고 1,030 문제 전체 정확도는 59.8% 였다 (HF eval_d2_rebuilt_16k/d2_rebuilt_h200.json). 이 trade-off 가 "붕괴의 원인은 answer trace imitation" 이라는 더 좁은 가설로 이어져 M1 (meta-span RLSD) 와 N3 (contrastive meta-span RLSD) 설계를 낳았다.

세 번째 전환점은 plan critic 이 M1/N3 가 §3 의 네 붕괴 증상 중 post-meta 에 직접 걸린 두 개 (confidence mode, truncation) 에는 advantage signal 이 들어가지 않는다는 점을 지적한 것이었다. 이 gap 이 SDC 를 낳았다. SDC 는 meta span 에는 정답 teacher 쪽으로 끌어당기는 attract, post-meta 구간에는 decoy teacher 분포에서 멀어지는 repel 을 각각 부여해 한 loss 안에 두 방향성을 공존시킨다.

네 번째 전환점은 EAD v3 를 전체 논문 프레임으로 묶으며 네 축 (증류 범위, teacher 필터, 토큰 가중치, 금지 패턴) 과 여섯 variant (A/B/C/D/E/F) 을 정리한 것이었다. 이 정리는 M1/N3/SDC 를 같은 설계 공간 위의 점들로 묶고 NeurIPS 2027 제출 가능한 단일 논문 윤곽을 만들었다.

## 2. 관찰 결과

관찰 layer 는 §3 의 네 단면을 HF 에 기록된 실제 측정치와 대조한다. 아래 표는 각 셀에 실제 측정치가 있으면 값과 출처를, 없으면 "not measured" 를 적는다.

| Stage | 1030 정확도 | Wrap rate | Confidence 분포 | AIME truncation | ΔH_meta±5 | 출처 |
|---|---|---|---|---|---|---|
| Base SFT | 72.04% | n/a (meta 없음) | n/a | not measured | n/a | paper §1 |
| Meta SFT (v8 meta inside strict) | 79.81% | 100% | 다점 분포 | 0% reported | +0.300 nats (보고값) | paper §3 |
| D1 (question-only BoN self-distill) | 68.35% | 0.29% | n/a (meta 거의 없음) | not measured | not measured | HF eval_d1_16k/d1_naive_h200.json summary |
| D1 rebuilt | 68.16% | 0.097% | n/a | not measured | not measured | HF eval_d1_rebuilt_16k/d1_rebuilt_h200.json summary |
| D2 (correct_then_meta self-distill) | 66.89% | 100% | 다점 분포 | not measured | +0.2356 nats (correct blocks +0.2573) | HF eval_d2_16k + entropy_d2_16k/entropy_stats.json |
| D2 rebuilt | 59.81% | 98.9% | 다점 분포 | AIME acc 6.7%, avg 8579 토큰 | +0.2309 nats (correct +0.2525) | HF eval_d2_rebuilt_16k + entropy_d2_rebuilt_16k |
| D3b Meta-KL (control spans) | 58.06% | 98.9% | 다점 분포 | AIME acc 6.7% | +0.2230 nats | HF eval_d3b_metakl_16k + entropy_d3b_metakl_16k |
| RL step 300 (E21R-v2) | ~72–73% | 88.2% | 0.96 에 98.9% | 13/30 = 43.3% | −0.0523 nats (correct −0.0417) | HF entropy_analysis_step300/rl_meta_confidence/conf_entropy_stats.json; paper §3 |

D3a 는 HF 상에 eval 결과 파일이 없어 "not measured" 이다 (results/d3a_metakl/ 디렉터리만 있고 eval JSON 비어 있음). M1, N3 300 step 결과는 plan v2 에 수치 target 은 있으나 실제 1030 eval 은 아직 HF 에 commit 되지 않아 본 보고서는 plan 단의 예상치를 숫자로 기재하지 않는다. 이 공백이 §5 에 열거한 hypothesis 의 "pre-registered" 상태를 결정한다.

이 표에서 읽히는 관찰은 세 가지다. 첫째, question-only BoN 자기증류 (D1 계열) 는 meta 구조 자체를 복원하지 못한다 (wrap rate 0.1–0.3% 수준). 이는 D1 이 "meta 가 선호되는" teacher rollout 신호를 충분히 주지 않음을 의미한다. 둘째, correct_then_meta (D2 계열) 는 meta wrap 을 되살리지만 정확도와 AIME 은 크게 떨어진다. D2 rebuilt 의 AIME 정확도 6.7% 는 D2 원본의 AIME 6.7% 와 동일하며, meta 구조 복원이 정답 수렴을 보상해 주지 않는다는 점을 재확인한다. 셋째, D2 rebuilt 와 D3b Meta-KL 은 entropy 패턴 (correct blocks ΔH +0.25 근방) 이 유사하되 정확도는 각각 59.81% 와 58.06% 로 D3b 가 더 낮다. D3b 에 control spans Meta-KL 을 얹은 조치가 정확도 회복에는 도움이 되지 않았음이 직접 관측된다.

## 3. 이론 다리

이 이론 layer 는 BU working note (reports/behavior_uncertainty_working_note_ko.pdf) 와 paper §4 (04_theory.tex) 의 Prop 1 부호 정리를 Meta-CoT 관찰에 이식하는 구조를 설명한다. BU 는 Llama-3.2-3B Countdown 환경에서 네 PPO 조건 — only_backtracking, all_strategies, backtracking_verification, backtracking_subgoal — 이 shared n=400 gate 를 통과하고 backtracking_backward 가 0.225 → 0.025 로 붕괴함을 관찰했으며, 이 네 조건이 서로 다른 entropy signature 를 남긴다는 점에서 Prop 1 의 네 pathway 분류 — Opener, Compression, Scaffold, Alignment-failure — 를 세웠다.

Prop 1 의 부호 조건은 meta event t_e 에서 ΔH_{t_e} > 0 (marker 가 엔트로피를 연다) AND γ_{t_e} > 0 (marker 가 정답 방향으로 hidden 을 민다) 동시 성립 시 기대 utility 상승 E[ΔU_T] > 0 을 보장한다는 기술적 주장이다. 이 두 조건이 동시에 필요하다는 점이 이 이론의 첫 번째 conditionality 다. BU 는 이 조건을 가정 체계 (A1)–(A7) 위의 Conditional Sign Theorem 으로 격상하면서, Opener/Scaffold/Alignment-failure 세 분기는 그 가정들 아래 직접 유도되고 Compression 분기는 별도의 bridge proposition (Conjecture 7 의 attention-entropy 감소가 Fano 수렴을 가속한다는 주장, 그리고 Conjecture 8 의 그 Fano 가속이 per-token drift 의 effective alignment γ^eff 를 끌어올린다는 주장) 양쪽에 동시에 conditional 함을 명시한다. 이것이 두 번째 conditionality 다.

Meta-CoT 관찰 layer 와 이 이론이 만나는 지점은 RL step 300 의 ΔH = −0.052 nats 이다. BU 분류 체계에서 ΔH < 0 만으로는 Compression (유용) 또는 Scaffold 약형 (중립) 으로 해석 가능하지만 Compression 에는 γ > 0 조건이 더 붙는다. Meta-CoT 실험에서는 γ 를 직접 측정하지 않았으므로 RL step 300 이 Compression 인지 Alignment-failure 근방인지 단정할 수 없고, 이 미확정성이 H1 (theory-observation bridge) 의 검증 의의이자 §5 의 검증 공백으로 남는다.

SDC 의 two-region 확장은 이 이론을 post-meta 구간까지 넓히는 시도다. Meta span 내부는 M1/N3 의 attract 로 γ > 0 방향 정렬을 장려하고, post-meta 구간 (meta 종료 직후부터 첫 `\boxed{}` 까지) 은 decoy teacher 분포에서 repel 해 Alignment-failure 경로의 역방향 drift 를 억제한다. 이 설계의 이론적 근거는 두 region 의 역할이 다르다는 것 — meta span 은 epistemic state 를 여는 개구부, post-meta 는 그 상태를 answer 에 실어 committing 하는 닫음부 — 이고, 붕괴 증상 (confidence mode, truncation) 이 post-meta 에서 발생한다는 §3 의 empirical 분해가 이 설계 의도를 뒷받침한다. 다만 Prop 1 이 post-meta 구간에 직접 적용된다는 주장은 본 연구 라인에서 pre-registered hypothesis (H-SDC-5, plan_SDC_v2 §2.5) 로만 존재하며, 아직 검증되지 않았다.

## 4. 방법 스펙트럼

방법 layer 는 세 종류의 질문을 분리해 매핑한다. RQ1 (diagnostic) 은 §3 의 네 단면이 무엇을 진단했는지를 묻는다. 답은 "controller 가 answer trace imitation 경로로 붕괴할 때 wrap, confidence, truncation, ΔH 네 지표가 같은 방향으로 움직인다" 는 단일 현상의 관찰이고, EV alignment 개념은 이 네 지표를 meta marker 주변 drift 하나의 단면들로 해석한다. 이 진단 자체는 plan 이 아니라 이미 paper §3 에 통합되어 있다.

RQ2 (method) 는 이 진단을 loss 로 옮기는 점진적 방법을 묻는다. M1 은 meta span 에만 privileged teacher (정답 conditioning) 의 log-ratio 를 advantage 에 곱해 attract 한다. N3 는 같은 meta span 에서 정답 teacher 와 결정적 decoy teacher 의 log-ratio 격차를 쓴다. 이 격차는 paper §4 의 derivation 에 의해 "정답 가설 vs 오답 가설의 Bayes factor" 로 해석되며, 공통 student marginal P_S 가 상쇄되어 초기 시점에도 non-trivial signal 을 준다는 점에서 M1 대비 signal polarization 이 높다. SDC 는 여기에 post-meta region 을 추가해 attract 는 meta span, repel 은 post-meta 에 배치한다. 이 점진성은 "증류 범위 와 방향 축" 두 개의 설계 차원을 따라 M1 → N3 → SDC 로 펼쳐진다.

RQ3 (meta-cognition effect) 은 이 네 지표가 실제로 meta 에 특수한 효과인지 일반 RL 의 부산물인지를 묻는다. 이 질문의 검증은 BU 의 cross-model 일반화 (H6) 와 non-meta mask ablation (EAD A2a/A2b) 에 기대고 있고, 현재는 pre-registered 상태다. SDC 의 H-SDC-3 "no imitation" 검증도 이 층에 속한다.

이 스펙트럼에서 SDC 를 기존 방법들과 나란히 놓으면 다음 비교가 성립한다. 순수 SFT (Naive-D2) 는 full trace 모방이므로 증류 범위 축이 가장 넓고 방향 축은 없다. Standard RL (E21R-v2) 은 환경 reward 만 쓰고 teacher signal 이 없으므로 본 축 위에서 원점이다. M1 은 범위 축을 meta span 으로 좁히고 방향 축에 attract 를 켠다. N3 는 같은 범위에서 방향 축의 해상도를 Bayes factor 로 높인다. SDC 는 범위 축을 meta + post-meta 두 region 으로 분할하고 각 region 에 다른 방향 (attract / repel) 을 배치한다. 이 마지막 조합 — region-asymmetric directional masks — 이 EAD survey 가 찾아낸 선행 literature 어디에도 명시되지 않은 좁은 novelty 이며 (plan_SDC_v2 §4, survey_rlsd_family_2026_04_17 §3–4), 이 novelty 에 기대어 SDC 를 "collapse 없이 자기증류를 유지" 스펙트럼의 새 점으로 제안한다.

## 5. 검증 현황

검증 layer 는 네 개 plan 문서에 분산된 가설들을 한 표로 묶는다. EAD 통합 플랜의 일곱 가설 (H1, H2, H2b, H3, H4, H5, H6) 과 SDC 의 여섯 가설 (H-SDC-1..6) 총 13 개가 대상이다.

H1 (theory-observation bridge) 은 Prop 1 의 pathway 부호 예측과 Meta-CoT 관찰 부호의 일치율을 per-meta-event 단위로 측정하는 것으로 정의된다 (plan_EAD_v3 §4). 현재 상태는 pre-registered 이다. γ 측정에 필요한 hidden probe 가 Meta-CoT V8 trace 에 대해 아직 돌지 않았고, 일치율 binomial test 의 관찰치도 비어 있다. H2 (EAD-Main > Naive-D2) 는 1030 eval 에서 +3 pp Overall, +5 pp AIME, wrap ≥ 95%, truncation ≤ 20% 를 McNemar paired test 로 검증한다. 현재 pre-registered 이며 EAD-Main 훈련이 아직 수행되지 않았다. H2b (B×F interaction) 는 EAD-Main 이 EAD-B, EAD-F-prior 각 단독 대비 +1.5 pp 이상 우위임을 검증한다. 역시 pre-registered 이다.

H3 (B filter effect isolation) 은 commit-quality τ 필터 on/off 시 truncation, ΔH 가 방향성 있게 움직이는지를 본다. 간접적 관련 증거로 D2 vs D2 rebuilt 간 AIME 정확도 차이 (6.7% 대 6.7% 로 동일, meta block 수 0.93 대 0.93 유사) 가 HF 에 기록되어 있으나 B 축 ablation 으로 재실행되지는 않아 partial 상태다. H4 (C contrastive signal additive), H5 (D entropy-shape amplification), H6 (cross-model generalization) 는 모두 pre-registered 이다.

SDC 계열 여섯 가설은 모두 pre-registered 상태다. H-SDC-1 (SDC-split 이 §3 의 post-meta 직접 두 증상을 N3 대비 개선) 의 gate 는 confidence entropy ≥ 0.3 bits, AIME truncation ≤ 25% 이며 Wilcoxon paired test 를 요구한다. 현재 e8 노드에서 SDC-split 300 step 실행이 진행 중이어서 수일 내에 partial 또는 confirmed/falsified 로 전이할 예정이다. H-SDC-2 (λ_post monotonicity), H-SDC-3 (SDC 가 gold imitate 하지 않음, KL noise floor 대비 |Δ| ≤ 2σ), H-SDC-4 (split 이 matched-L1 uniform 대비 ≥ 2 pp 우위), H-SDC-5 (Prop 1 two-region alignment 가 unseen prompt 의 ΔH 부호를 60% 이상 예측), H-SDC-6 (null-signal noise control 대비 SDC 가 ≥ 1 pp 우위) 모두 실행 대기 중이다.

종합하면 13 개 가설 전부 pre-registered 이며 confirmed/falsified 된 항목은 아직 없다. 부분 증거로 읽을 수 있는 셀은 (a) D2 계열 HF 실측치가 "naive self-distill 이 post-meta 붕괴를 푼다" 가설을 기각한다는 점, 그리고 (b) D3b 의 AIME 회복 실패 (6.7% 유지) 가 control spans Meta-KL 만으로는 post-meta 문제가 해결되지 않는다는 negative evidence 로 사용 가능하다는 점이다. 이 두 negative partial 결과가 EAD/SDC 의 설계 필요성을 empirical 하게 지탱한다.

## 6. 남은 위험

위험 layer 는 plan 문서 네 개의 risk register 를 교차 정렬해 영향도와 발생 가능성의 곱이 높은 다섯 개를 추린다. 첫째 위험은 compute over-budget 이다. EAD 통합 플랜은 505 GPU-hour, 5–7 wall-day 를 요구하고 여기에 SDC 가 +95.6 GPU-hour 를 추가한다. plan_SDC_v2 §3.4 는 이를 A4 (Opener probing 14 GPU-hr) 와 half-A1 (B-τ sweep 21 GPU-hr) 를 drop 해 흡수하도록 했으나 parent plan v3 owner sign-off 가 아직 없다. 현재 mitigation 상태는 "승인 대기" 이다.

둘째 위험은 H-SDC-3 실패다. 이 실패는 SDC 의 structural no-imitation 주장을 철회시키고 headline 으로 쓸 수 없게 만든다. Mitigation 은 pre-registered stop 이며 branch A 가 artifact 로 정의되어 있다. 현재 상태는 noise floor 측정 (M1 2 seed) 이 pre-registered 되어 있으나 실제 수치는 비어 있음이다.

셋째 위험은 H6 (cross-model generalization) 실패다. BU Llama Countdown 과 Qwen Meta-CoT 의 pathway 분포가 양의 상관을 보이지 않으면 Prop 1 의 Meta-CoT 이식이 약화된다. Mitigation 은 paper framing 을 Qwen primary + Llama weak evidence 로 이미 downscope 한 것 (plan_EAD_v3 §0, C2) 이며, H6 실패 시에도 H2/H2b/H3/H4/H5 는 독립적으로 유지된다.

넷째 위험은 decoy leakage 이다. N3 의 결정적 decoy 가 gold 와 숫자적으로 일치하거나 학습 중 답을 간접 노출할 수 있다. plan_contrastive_rlsd_v1 §2.1 의 make_decoy_answer 가 A–D 네 조건 (문자열 불일치, 숫자적 불일치, 결정성, syntactic validity) 을 명시하고 abort 조건 decoy_is_correct_rate > 5% 를 둔다. 현재 smoke 단계에서 이 rate 은 < 1% 수준이나 long-run 에서의 안정성은 미검증이다.

다섯째 위험은 post-meta mask fallback 빈도다. SDC 의 post-meta mask 는 `\boxed{}` 존재를 전제로 하고 없으면 end-of-completion 까지 fallback 한다. plan_SDC_v2 §3.2 는 fallback rate > 20% 를 halt 조건으로 두지만 AIME subset 에서 §3 의 truncation 43.3% 관찰을 고려하면 SDC 훈련 초기에는 fallback 이 자주 일어날 수 있다. Mitigation 은 smoke rule §3.6 의 halt 와 fallback rate 로깅이다. 현재 상태는 code-level 구현만 완료되고 production smoke 에서의 rate 은 아직 수집되지 않았다.

## 7. 다음 행동

가장 즉각적 행동은 e8 노드에서 진행 중인 SDC-split 300 step 실행의 완료를 지켜보는 것이다. 이 run 은 2 seed ({42, 43}) 로 주 variant 를 만들고, 완료 시 H-SDC-1 의 gate — confidence entropy ≥ 0.3 bits AND AIME truncation ≤ 25% — 를 판정한다. 이 gate 의 통과 여부가 논문의 headline 방향을 결정한다. 통과 시 SDC 가 §5 primary method 로 승격되고 H-SDC-3 의 KL noise floor 검증이 다음 step 이다. 미통과 시 branch B/C (decision tree §7 of plan_SDC_v2) 에 따라 SDC 는 exploration variant 로 남고 EAD-Main 이 headline 을 담당한다.

단기 행동은 H-SDC-3 검증에 필요한 M1 noise floor 측정이다. M1 seed 42, 43 두 개의 200 held-out problem inference 를 돌려 KL 의 seed-간 분산 σ_noise 를 측정하고, 그 값 대비 SDC 의 KL_SDC 가 2σ 이내임을 보이는 것이 "SDC 가 gold 로 수렴하지 않음" 주장의 structural 근거다. 이 측정은 SDC-split 훈련과 병렬 가능하다.

중기 행동은 EAD-Main 훈련 (A ∧ B ∧ F-prior, τ = 0.5) 과 Naive-D2 baseline 재실행이다. 현재 HF 에 올라간 D2, D2 rebuilt 는 raw correct filter 상태이며 commit-quality τ filter 를 적용한 B axis 실험이 H2b 의 primary 비교 대상이다. EAD-Main 이 EAD-B 와 EAD-F-prior 각 단독 대비 +1.5 pp 이상 우위를 보여야 B×F interaction 의 novelty 가 empirically 지지된다.

장기 행동은 H6 cross-model 검증이다. BU Llama-3.2-3B Countdown 의 pathway 빈도 벡터 p_Llama 는 이미 bu_analysis_0416/ 아래에 존재하고, Qwen3-8B MATH 의 Meta-CoT trace 에서 동일 네 pathway 분류를 재계산해 p_Qwen 을 얻으면 Pearson r(p_Llama, p_Qwen) 의 bootstrap point estimate 와 95% CI 를 계산할 수 있다. 이 측정은 추가 훈련을 요구하지 않고 기존 SFT/RL/D2 trace 를 재분석하는 작업이므로 compute 압력이 가장 낮다. H6 가 통과하면 이론 다리가 cross-model 로 단단해지고, 실패해도 method layer 는 영향받지 않는다.

---

Sources: HF iamseungpil/metacot (results/entropy_*/entropy_stats.json, results/eval_*/*.json, results/entropy_analysis_step300/); paper sections/{01,03,04,05}.tex; results/plan_{meta_rlsd_v2,contrastive_rlsd_v1,EAD_unified_v3,SDC_v2}_2026_04_17.md; results/survey_rlsd_family_2026_04_17.md; BU reports/behavior_uncertainty_working_note_ko.md.
