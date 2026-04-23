# Epistemic Alignment Distillation (EAD) — Unified Paper Plan v2

**Date**: 2026-04-17
**Status**: Draft v3 — revised after critic v2 REQUEST REVISIONS (v2 critical fixes applied: changelog consistency, ablation arithmetic, H1 null baseline)
**Scope**: Single paper integrating (i) Four Habits working note EV alignment theory, (ii) Meta-CoT V8 empirical observations, (iii) Meta-RLSD family (M1/N3/B/D/F) methods and ablations

**Changelog v1 → v2 → v3**
- **C1** (v2): H1 falsification binomial test vs chance baseline — v3 further refined with empirical null option (§4.H1)
- **C2** (v2): §2.C1 downgraded to hypothesis; v3 scoped abstract claim to Qwen + weak Llama (§0, §2, §4.H6)
- **C3** (v2): Compute re-estimated; v3 corrected internal contradiction (changelog vs body) and fixed ablation arithmetic → **Total ≈ 380 GPU-hours (primary 300 + ablations 80), 5-7 wall-days** (§5.2, §8)
- **W1** (v2+v3): "+0.3 nats" tied to empirical ceiling; v3 strengthened falsification to require *both* mean convergence AND variance reduction vs naive-D2 baseline (§3.3, §4.H5)
- **W2** (v2): F axis weight ablation — F-prior / F-uniform / F-invfreq (§3.2, §5.2 A2)
- **W3** (v2+v3): B filter τ with commit-quality formula; v3 added justification for coefficients 0.5/0.3/0.2 (§3.1)
- **W6** (v2): Self-Rewarding, Meta-Rewarding, SPPO, Iterative-DPO added (§6)
- **W7** (v2+v3): Main/appendix split; v3 redistributed page budget (§3 → 1 p, §6 → 2.5 p, §7 → 1 p)
- **Sug-7** (v3): H2 test → McNemar paired bootstrap, not z-test (§4.H2)
- **Sug-8** (v3): Run count clarified — **6 trainable configurations × 2 seeds = 12 primary runs** (§5.2)

---

## 0. Intent (의도)

Self-distillation이 meta-reasoning (epistemic verbalization, 이하 EV) controller를 **answer imitation으로 붕괴시키는** 현상을 단일 원리로 설명하고, 그 원리로부터 도출된 **control-preserving distillation family (EAD)** 가 collapse를 방지함을 이론 가설·관찰·방법 세 층위에서 보인다.

**한 줄 요약** (scope: Qwen3-8B MATH/AIME primary, Llama-3.2-3B Countdown as weak cross-model evidence)
> Qwen3-8B MATH에서 naive self-distillation은 EV token alignment를 보존하지 못해 collapse를 일으킨다 (가설, H1-H2). EAD는 alignment 조건을 loss로 직접 translate하여 collapse-free distillation을 만든다 (H3-H5 검증). 이 원리는 Llama-3.2-3B Countdown에서도 약하게 관측된다 (H6, weak supplementary).

## 1. Why — 3-단계 motivation

### 1.1 Theory layer (working note: `네 가지 추론 습관의 PPO 단계 정보이론 메커니즘`)
- Llama-3.2-3B Countdown-3to4: 5 PPO 조건, 정확도 27배 분산
- 통합 정리 (Prop 1 부호 정리): EV token이 ΔH_{t_e} > 0 **AND** γ_{t_e} > 0 (정렬) 성립 시 ΔU_T > 0 (success); γ ≤ 0 (misalignment) 시 실패
- 4경로: **Opener / Compression / Scaffold / Alignment-failure**
- **중요**: 이 이론은 *Llama-3.2-3B Countdown* 에서 derivative 관찰되었으며, Qwen3-8B MATH로의 generalization 자체가 이 논문의 검증 대상 (non-trivial)

### 1.2 Observation layer (Meta-CoT V8)
- Qwen3-8B + MATH/AIME/GSM8K, `<|meta|>` token 기반 controller
- **Meta SFT 성립**: +3.88 pp (75.92 → 79.81 %), Δentropy = +0.300 nats (Opener signature와 일치)
- **RL step 300 붕괴**: wrap rate 100→88.2 %, confidence 0.96 mode collapse 98.9 %, AIME truncation 13/30 (43.3 %), Δentropy = **−0.052** nats (alignment 반전)
- **Naive self-distill (D2 rebuilt)**: controller 복원 (Δentropy +0.231) but AIME truncation 47 %, 정확도 60 % (baseline 대비 -16 pp)

→ Pattern: RL과 naive distill 모두 EV alignment 보존 실패. 핵심은 **meta token 빈도가 아니라 정렬된 drift 방향**.

### 1.3 Method gap
기존 self-distill은 answer trace imitation이 중심. EV alignment가 loss에서 explicit하지 않음 → 학습이 (i) answer-wrong-but-well-formed teacher 복제, 또는 (ii) structural collapse에 indifferent.

## 2. Core claim + 3 sub-claims

**Core**: Self-distillation은 control-preserving하려면 reasoning 본문이 아니라 **EV alignment geometry**를 distill 대상으로 삼아야 한다.

- **C1** (Theory↔Observation bridge — **testable hypothesis, not assertion**): Meta-CoT의 RL step 300 failure mode가 working note의 alignment-failure 경로와 동일한 signature 서명을 갖는지 Qwen3-8B에서 직접 검증한다. (ref §4.H1, §4.H6)
- **C2** (EAD family 설계 원리): 4 제약 축의 교집합이 collapse-free 설계를 만든다.
- **C3** (Empirical 검증): M1/N3/B/D/F 중 하나 이상이 naive baseline을 초과하며 (AIME ≥ baseline + 3 pp) controller 보존 (wrap ≥ 95 %, Δentropy 부호 양).

> 참고: v1 §2.C1은 **assertion** ("일치한다")로 적혀 있었으나, 이는 이 논문의 검증 대상(H1, H6)과 중복되어 원래 주장이 이미 참이라는 circular 가정이 되므로, v2에서 *hypothesis to be verified* 로 downgrade.

## 3. EAD framework (4 × 6 matrix)

**4 제약 축**:
1. **What to distill** (scope: meta span / meta+post-meta / full trace)
2. **Which teacher** (filter: raw correct / alignment-quality ≥ τ / contrastive pair)
3. **Which token** (weighting: binary / control-critical gradation / entropy-shaped)
4. **What to forbid** (penalty: meta-loop / no-boxed / boxed-after-drift / entropy reversal)

**6 instantiations**:

| ID | Name | Axis 1: scope | Axis 2: teacher | Axis 3: token | Axis 4: forbid |
|---|---|---|---|---|---|
| **A** | Meta-only KL (M1) | meta span | raw correct | binary 0/1 | — |
| **B** | Alignment-filtered | meta + post-meta 5 | `commit_quality(tr) ≥ τ ∧ no_boxed=0 ∧ decoherence=0` | same as A | — |
| **C** | Contrastive T+/T- (N3) | meta only | T+ = correct, T- = deterministic decoy | binary | — |
| **D** | Entropy-shape regularizer | meta + post-meta 5 | correct | 0/1 | ‖Δentropy_5tok − τ_e‖² penalty |
| **E** | Counterfactual | mixed (epistemic vs overconfident) | contrastive pair | binary | — |
| **F** | Commit-aware | meta + post-meta | correct | **control-critical weighting** | meta-loop, no-boxed, boxed-after-drift |

### 3.1 B axis — commit-quality τ (neu W3 + Sug-9)
- `commit_quality(tr) := 0.5·I[wrap_rate(tr)=1.0] + 0.3·I[has_boxed] + 0.2·(1 − conf_mode_ratio)`; ∈ [0, 1]
- **Coefficient justification**: weights reflect observed failure-mode severity in D2 traces — wrap break (most predictive of structural collapse, 0.5), missing boxed (AIME truncation driver, 0.3), confidence mode collapse (calibration failure signature, 0.2). Full sensitivity analysis in Appendix E.
- Primary: **τ = 0.5** (retain ~60 % of D2 trace)
- Ablations: τ ∈ {0.3, 0.5, 0.7} → retention {80 %, 60 %, 30 %}
- Definition 의도: high-wrap + boxed 완결 + confidence 다양성 보존 trace를 prioritize (Δentropy 양수 signature와 상관)

### 3.2 F axis — control-critical weighting (neu W2)
- Weighting 후보:
  - **F-prior** (default): confidence=1.5, diagnosis=1.25, verify=1.10, other meta=1.0
  - **F-uniform**: all meta=1.0 (baseline ablation)
  - **F-invfreq**: weight ∝ 1 / token frequency in teacher corpus (1.5/1.25/1.10과 비교할 data-driven variant)
- Primary run: F-prior
- Ablation: F-uniform vs F-prior vs F-invfreq

### 3.3 D axis — entropy-shape target τ_e (neu W1)
- τ_e = 0.30 nats **empirical target** derived from v8 meta SFT Opener signature (observed Δentropy = +0.300 nats at successful step)
- 이는 **theoretical bound가 아니라 observed ceiling** — working note는 "temporary uncertainty expansion → concentration" 경로를 제시하지만 그 amplitude는 learned property이므로 lower bound만 제공
- Paper 표기: `τ_e` as a *hyperparameter tied to empirical ceiling*, not a closed-form constant. Ablation: τ_e ∈ {0.15, 0.30, 0.45}

**Main run (EAD-Main)**: **A ∧ B(τ=0.5) ∧ F-prior**
**Ablations**: ±C, ±D, ±E, ±F weight choice, B τ sweep
**Baseline**: Naive D2 (no EAD), RL E21R-v2 step 300

## 4. Hypotheses (falsifiable, revised per C1)

### H1 (Theory-Observation bridge) — null fixed per v2-C3
- **Claim**: Meta-CoT failure mode는 working note의 alignment-failure 4 pathway 중 하나와 일치하는 signature 를 보인다.
- **Operationalization (binomial test)**: Prop 1 부호 예측과 관찰 부호의 일치율을 per-meta-event 단위로 측정 (N=100 meta events stratified across {RL step 300, v8 SFT, naive D2}).
- **Two null baselines reported** (robustness to pathway-frequency assumption):
  - **Uniform null** (p_null = 0.125): chance assumes uniform pathway prior (4 pathway × 2 sign outcomes). Threshold for p<0.05 one-sided: agreement ≥ 19/100.
  - **Empirical null** (p_null = empirical marginal pathway frequency in v8 SFT traces): computed per seed from held-out SFT sample; anti-conservative bound. Threshold varies (estimated 25-35 / 100).
- **Falsification**: BOTH nulls fail to reject (uniform AND empirical), i.e., agreement < max(19, empirical threshold). Paper reports both.
- **Strong prediction (not required)**: agreement ≥ 70 % (highly aligned bridge)
- **Power**: at N=100, p_alt=0.5 vs p_null=0.125 → power > 0.999; even p_alt=0.35 vs 0.125 → power ≈ 0.95. Sample size adequate.

### H2 (EAD-Main > naive)
- 1030-problem 16k eval, seed=×2
- **Claim**: EAD-Main vs naive D2 — Overall ≥ +3 pp (paired per-problem comparison), AIME ≥ +5 pp, meta wrap rate ≥ 95 %, AIME truncation ≤ 20 %
- **Statistical test** (Sug-7): **McNemar paired test** on per-problem correctness (handles intra-problem dependence across seeds); **paired bootstrap 95 % CI** on accuracy difference as supplementary.
- **Falsification**: EAD-Main Overall ≤ naive D2 + 1 pp (McNemar p > 0.1 OR bootstrap 95 % CI includes 0)

### H3 (B filter effect isolation — W3 revised)
- **Claim**: EAD-Main vs (EAD-Main \ B axis) — B filter 제거 시 truncation 증가, Δentropy 감소
- **τ sweep prediction**: truncation rate monotonically 증가 τ=0.7→0.3
- **Falsification**: |ΔΔentropy(τ=0.5 vs τ=0)| < 0.05 nats AND truncation rate difference < 3 pp

### H4 (C contrastive signal additive)
- **Claim**: EAD-Main vs EAD-Main+C (N3 integration) — AIME +2 pp 이상 또는 calibration ECE ≥ -0.02 개선
- **Falsification**: 두 metric 모두에서 개선 < 1 pp / < 0.01

### H5 (D entropy-shape amplification — v2-W1 + v3 strengthened)
- **Background**: Naive D2 already recovers Δentropy ≈ +0.231 nats (§1.2). For D to show additional effect beyond naive teacher, both mean convergence AND distribution tightening are required.
- **Claim**: D 활성화 시 meta-token 후 5-token window Δentropy 분포 (i) 평균이 τ_e = +0.30 nats 근접 (|mean − τ_e| < 0.05 nats) AND (ii) 분산이 naive-D2 baseline의 70 % 이하.
- **Falsification**: |mean − τ_e| ≥ 0.05 nats **OR** variance ratio (D / naive-D2) ≥ 0.9 (i.e., no distribution tightening).

### H6 (Cross-model generalization — C2 revised)
- **Claim (weaker than v1 §2 assertion)**: Working note (Llama-3.2-3B Countdown) 와 Meta-CoT (Qwen3-8B MATH)는 **alignment signature의 *존재*를 공통으로** 보이며 (공통 4 pathway 중 적어도 2개 공유), 이는 EV alignment가 단일 모델/도메인 현상이 아니라는 weak evidence
- **Falsification**: Llama와 Qwen 중 한쪽에서 4 pathway 중 0개만 관측됨 (binomial null p > 0.2)
- **중요**: H6이 실패해도 H2/H3/H4/H5 (EAD method 효과)는 독립적으로 유지됨. 즉 cross-model은 **additional contribution** 로 positioning.

## 5. Verification methodology

### 5.1 Metric suite (BU analysis 재사용 + 확장)

(working note §2 4 signatures 재구현 예정 — code port plan `plan_BU_analysis_port_v1.md` 별도 문서)

1. **ΔH_{t_e±5}**: EV marker 전후 5-token 윈도우 평균 entropy 차이
2. **d_M (Mahalanobis distribution rearrangement)**: (H_t, top1, top1-top2) 3-axis에서 EV pair vs neutral pair 거리
3. **I(M_c; Y | D)**: trace의 meta count (cap=5) × correctness, difficulty tercile 조건부 mutual information
4. **C_t = Σ_s (1 − H_s / log₂ V)**: post-marker 5-token 누적 confidence gain, SFT vs PPO Cohen's d

+ **Meta-CoT 확장 metrics** (이전 §5.1와 동일):
- AIME truncation rate (no_boxed in 16k budget)
- Boilerplate share (top-1 assessment 비중)
- Confidence distribution mode + entropy
- Wrap rate (`<|meta|>`/`<|/meta|>` balanced pair 비율)
- **Commit-quality** score (§3.1)

### 5.2 Experimental matrix (v3: Sug-8 run count clarified, C2 ablation arithmetic fixed)

**Primary runs** (student init = v8 meta SFT Qwen3-8B, 2 seeds each on trainable runs):

| Run | Method | Teacher | Scale | Trainable? | Wall-hours (4×H200) |
|---|---|---|---|---|---|
| Naive-D2 (baseline) | SFT only | D2 rebuilt teacher data | 1 epoch 10k | ✔ | 3 |
| RL-step300 (baseline) | verl-GDPO | — | 완료 (eval only) | ✘ (reuse) | 0 |
| M1 (A) | Meta-RLSD | single priv | 300 steps | ✔ | 4 |
| N3 (A ∧ C) | Contrastive | T+ / T- | 300 steps | ✔ | 5 |
| EAD-B (A ∧ B, τ=0.5) | Filter + meta-only | filtered priv | 300 steps | ✔ | 4 |
| EAD-F-prior (A ∧ F-prior) | Commit weight | correct | 300 steps | ✔ | 4 |
| **EAD-Main (A ∧ B ∧ F-prior, τ=0.5)** | combined | filtered | 300 steps | ✔ | 5 |
| EAD-Full (A ∧ B ∧ C ∧ D ∧ F) | all axes | filtered | 300 steps | ✔ | 6 |

→ **7 trainable configs × 2 seeds = 14 primary runs**. (RL-step300 reuses existing checkpoint, Naive-D2 SFT is 1 seed baseline.)
→ Primary trainable wall-hours: sum of 7 configs = 3 + 4 + 5 + 4 + 4 + 5 + 6 = 31 wall-hours × 2 seeds = **62 run-wall-hours**

**Ablations** (each 1 seed, 150 steps = 50 % of primary → ~1.75 h/run):
- A1. B-τ sweep: τ ∈ {0.3, 0.5, 0.7} — 3 × 1.75 h = **5.25 run-hours** (EAD-Main τ=0.5 reused from primary → net 2 × 1.75 = 3.5 new run-hours)
- A2. F-weight sweep: F-uniform, F-prior, F-invfreq — 3 × 1.75 h = 5.25 (F-prior reused → 3.5 new run-hours)
- A3. D τ_e sweep: τ_e ∈ {0.15, 0.30, 0.45} — 3 × 1.75 h = 5.25 (τ_e=0.30 reused from EAD-Full → 3.5 new run-hours)
- A4. Cross-model: Qwen3-8B Countdown Opener probing — 1 × 6 h = 6 run-hours

→ **Ablation new run-hours**: 3.5 + 3.5 + 3.5 + 6 = **16.5 run-hours**

**Compute summary** (corrected arithmetic):
- Primary trainable: 62 run-hours × 4 GPUs = **248 GPU-hours** compute
- Ablations new: 16.5 × 4 = **66 GPU-hours**
- Sub-total clean: **314 GPU-hours**
- Preempt/restart overhead on BSC cluster (+35 %): **314 × 1.35 ≈ 424 GPU-hours**
- Rounded planning budget: **≈ 420 GPU-hours (= 105 node-hours on 1 node of 4×H200)**
- **Wall-time**: 1 node 105 hrs ≈ 4.4 days; 2-4 nodes parallel ≈ 1.1-2.2 days best case; **5-7 wall-days realistic with preempt cycles**
- Preempt budget: HF checkpoint every 20 steps (~5 min), resume-on-restart driver, up to 5 restart cycles 허용

**Cross-validation**: working note Llama-3.2-3B Countdown metric 재실행 (BU codebase) + Qwen3-8B Countdown Opener probing (A4).

### 5.3 Success / failure criteria

| 레벨 | 성공 | 실패 |
|---|---|---|
| Theory (H1) | Binomial p < 0.05 agreement vs chance | agreement < 19 % (p ≥ 0.05) |
| Method (H2) | EAD-Main > Naive-D2 (+3 pp Overall p < 0.05, +5 pp AIME) | ≤ +1 pp Overall or p > 0.1 |
| Ablation (H3-H5) | B/D/F 각각 제거 시 유의 감소 (> 3 pp on primary metric) | 모든 제거에서 < 1 pp |
| Cross-model (H6) | ≥ 2 common pathways observed in both Llama & Qwen | 한쪽 모델 0 pathway |

## 6. Paper structure (NeurIPS 9 main + ∞ appendix — W7)

### Main (9 pages, target NeurIPS 2027 submission) — v3 redistribution per W5
```
§1 Introduction (1 p)
  - Controller paradox, alignment hypothesis, contributions

§2 Related Work (0.75 p)
  - RLCD, REDI, DistiLLM-2, RLSD, OPSD, GATES, HDPO,
    Self-Rewarding LM, Meta-Rewarding, SPPO, Iterative DPO

§3 Epistemic Verbalization Alignment (Theory summary, 1 p)
  - Prop 1 statement, 4 pathway intuition, alignment assumption A_EA
  - Full proof in Appendix A

§4 Empirical Collapse Observation (0.75 p)
  - Meta SFT (Δentropy +0.300), RL step 300 (wrap/confidence collapse, Δentropy −0.052),
    Naive D2 trade-off table

§5 EAD Framework (1.5 p)
  - 4 axis × 6 instantiation matrix
  - Main A ∧ B ∧ F definition
  - Alignment-as-loss derivation sketch

§6 Experiments (2.5 p) — expanded to fit 7 primary + 3 ablation sweeps + cross-model
  - Matrix (compact table), metrics, H1-H6 results (main table)
  - Ablations B τ / F weight / D τ_e (one sub-table each)
  - Cross-model Llama + Qwen (compact row)

§7 Discussion + Limitations (1 p)
  - Alignment-first perspective
  - Limitations: A_EA heuristic, τ_e empirical (observed ceiling, not bound), single-domain

§8 Conclusion (0.5 p)
```
(Total: 9.0 p main excluding abstract + references; tight but achievable.)

### Appendix (unlimited — submission + supp package)
- A. Full Prop 1 proof, 4 pathway formal statements
- B. All reward/KL/entropy formulas (M1 §2, N3 §2, D entropy penalty)
- C. Full experimental hyperparameters, preempt checkpointing strategy
- D. Cross-model raw metrics (Llama + Qwen per-condition)
- E. B τ / F-weight / D τ_e sweep tables
- F. Reproducibility: commit SHAs, HF checkpoint URLs, conda env spec

## 7. Risks + mitigation

| Risk | Mitigation |
|---|---|
| H6 (cross-model) signature 불일치 | Paper positioning은 H2/H3/H4/H5 (EAD method) 중심; H6 실패 시 "scope limitation" 으로 discuss |
| Compute over-budget (400 GPU-hrs) | 4-node parallel priority; preempt-defense driver v4 (5 min HF push); ablations sequentially after primary |
| Naive D2 baseline이 이미 약함 | SFT baseline (v8 meta SFT) 별도 표로 anchor |
| H5 (entropy-shape D) τ_e = +0.30 nats가 empirical이므로 theoretical bound 아님 | Paper text는 "observed ceiling" 명시, ablation τ_e ∈ {0.15, 0.30, 0.45} 로 sensitivity 제시 |
| Decoy quality (C N3) | Random vs rule-based ablation; deterministic md5 hash guarantees reproducibility |
| B filter τ가 retention curve 연속성에 의존 | τ sweep {0.3, 0.5, 0.7} 포함하여 plateau 여부 검증 |

## 8. Timeline (revised per C3)

| Phase | Duration | Deliverable |
|---|---|---|
| Plan iteration (this doc) | 0.5 day | Plan approval (v2 critic pass) |
| B, D, F code addition (iterative-code-review) | 2-3 days | Extended MetaRLSDTrainer ready |
| BU analysis port (iterative-code-review) | 1-2 days | 4 EV metrics on Meta-CoT traces |
| Smoke (8 runs × 10 prompts) | 1 day | Bug-free stack |
| Full training runs (8 × 2 seeds + 4 ablations) | **5-7 wall-days** (400 GPU-hrs with preempt cycles) | Training done |
| Eval + analysis (EV metrics + H1-H6) | 1-2 days | Metric tables |
| Cross-model (Llama Countdown replay + Qwen Countdown probe) | 1 day | Table 3 |
| Paper draft + self-critic + codex-critic iteration | 2-3 days | Submission-ready draft |

**Total**: **14-19 days** for full paper readiness (v1에서 11-12일로 과소추정했던 항목을 현실화; v3 compute ≈ 420 GPU-hrs).

## 9. Acceptance checklist for v2

- [x] Intent explicit (§0)
- [x] Theory/Observation/Method gap structured (§1)
- [x] Core claims downgraded to hypotheses (C1 revised, W7)
- [x] Falsifiable hypotheses with binomial tests (H1 revised, §4)
- [x] Operational metric suite (§5.1)
- [x] 8-run × 2-seed primary + 4 ablation experimental matrix (§5.2)
- [x] B filter τ specification + sweep (§3.1, A1)
- [x] F weighting ablation (§3.2, A2)
- [x] D τ_e empirical tie + sweep (§3.3, A3)
- [x] Success/failure criteria with statistical tests (§5.3)
- [x] NeurIPS page-budget paper structure main/appendix (§6, W7)
- [x] Risk + mitigation (§7)
- [x] Realistic compute 400 GPU-hrs, 5-7 wall-days timeline (§5.2, §8, C3)
- [x] Related work adds Self-Rewarding/Meta-Rewarding/SPPO/Iterative-DPO (§6, W6)

→ v2 → critic re-review → 통과 시 B, D, F 코드 구현 + BU analysis port phase 진입.

---

## A. Diff summary v1 → v2 → v3

| Section | v1 | v2 | v3 |
|---|---|---|---|
| §0 one-line summary | Universal claim | Universal claim | Scoped to Qwen primary + Llama weak |
| §2 C1 | "일치한다" (assertion) | Downgraded to hypothesis | (same as v2) |
| §4 H1 falsification | "50 % 이상 어긋남" | Binomial p<0.05, p_null=0.125, ≥19/100 | **Dual null** (uniform + empirical), power analysis |
| §4 H2 test | "paired z-test on 1030" | (same) | **McNemar + bootstrap 95% CI** |
| §4 H5 falsification | "수렴 ≤ +0.2 nats" | "|mean - τ_e| > 0.15 nats" | **Both mean (<0.05) AND variance ratio <0.9** (stricter) |
| §4 H6 | Strong generalization | Weak "≥2 common pathways" + independence | (same as v2) |
| §3.1 B axis | Undefined τ | Formula + τ sweep {0.3,0.5,0.7} | **+Coefficient justification** |
| §3.2 F axis | Fixed 1.5/1.25/1.10 | F-prior/uniform/invfreq ablation | (same as v2) |
| §3.3 D τ_e | "+0.3 nats" undefined | Empirical ceiling + sweep | (same as v2) |
| §5.2 run count | 8 runs × 2 seeds | "8 runs × 2 seeds" ambiguous | **7 trainable × 2 seeds = 14 runs** clarified |
| §5.2 compute | 56 GPU-hrs, 14 wall-hrs | "400 GPU-hrs" (changelog/body mismatch) | **420 GPU-hrs**, ablation arithmetic fixed (1.75h/150-step) |
| §6 paper | Flat section list | 9-p main split | **Budget redistributed** (§3→1, §6→2.5, §7→1) |
| §7 related work | 7 papers | +Self-Rewarding/Meta-Rewarding/SPPO/Iterative-DPO | (same as v2) |
| §8 timeline | 11-12 days | 14-19 days | (same as v2) |
