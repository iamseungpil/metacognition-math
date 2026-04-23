# Epistemic Alignment Distillation (EAD) — Unified Paper Plan v3

**Date**: 2026-04-17
**Status**: Draft v3 — revised after critic v2 REQUEST REVISIONS (v2 critical fixes applied: changelog consistency, ablation arithmetic, H1 null baseline)
**Scope**: Single paper integrating (i) Four Habits working note EV alignment theory, (ii) Meta-CoT V8 empirical observations, (iii) Meta-RLSD family (M1/N3/B/D/F) methods and ablations

**Changelog v1 → v2 → v3**
- **C1** (v2 → v3.2): H1 falsification binomial test vs chance baseline — v3 added empirical null option; **v3.2** replaced empirical null with permutation test on the same N=100 test set (10,000 shuffles) and added pathway-exclusivity caveat (§4.H1)
- **C2** (v2): §2.C1 downgraded to hypothesis; v3 scoped abstract claim to Qwen + weak Llama (§0, §2, §4.H6)
- **C3** (v2 → v3 → v3.1 → v3.2): Compute re-estimated; v3 corrected internal contradiction (changelog vs body) and fixed ablation arithmetic; v3.1 re-did arithmetic with ablations trained at the same 300-step budget as primary runs; **v3.2** expanded per-variant ablation arithmetic (A1 42 + A2 28 + A3 42 + A4 14 = 126 GPU-hrs; A2 reuses F-prior from primary EAD-F-prior so only 2 new F-weight variants) → **Total ≈ 505 GPU-hours (primary 248 + ablation 126 + 35 % preempt reserve), 5-7 wall-days** (§5.2, §8)
- **W1** (v2+v3): "+0.3 nats" tied to empirical ceiling; v3 strengthened falsification to require *both* mean convergence AND variance reduction vs naive-D2 baseline (§3.3, §4.H5)
- **W2** (v2): F axis weight ablation — F-prior / F-uniform / F-invfreq (§3.2, §5.2 A2)
- **W3** (v2+v3): B filter τ with commit-quality formula; v3 added justification for coefficients 0.5/0.3/0.2 (§3.1)
- **W6** (v2): Self-Rewarding, Meta-Rewarding, SPPO, Iterative-DPO added (§6)
- **W7** (v2+v3): Main/appendix split; v3 redistributed page budget (§3 → 1 p, §6 → 2.5 p, §7 → 1 p)
- **Sug-7** (v3): H2 test → McNemar paired bootstrap, not z-test (§4.H2)
- **Nov-1** (v3.2): **H2b B×F interaction test** added — EAD-Main − max(EAD-B, EAD-F-prior) ≥ +1.5 pp paired McNemar; directly validates the "commit-quality × control-critical weighting" novelty claim beyond H2 (§4.H2b, §5.3)
- **Sug-8** (v3): Run count clarified — **7 trainable configurations × 2 seeds = 14 primary runs** (Naive-D2 trainable with 2 seeds; RL-step300 reuses existing ckpt) (§5.2)

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

### H1 (Theory-Observation bridge) — null fixed per v3.2 (permutation test replaces circular empirical null)
- **Claim**: Meta-CoT failure mode는 working note의 alignment-failure 4 pathway 중 하나와 일치하는 signature 를 보인다.
- **Operationalization (binomial test)**: Prop 1 부호 예측과 관찰 부호의 일치율을 per-meta-event 단위로 측정 (N=100 meta events stratified across {RL step 300, v8 SFT, naive D2}).
- **Two null baselines reported** (robustness to pathway-assignment structure):
  - **Uniform null** (p_null = 0.125): chance assumes uniform pathway prior (4 pathway × 2 sign outcomes). Threshold for one-sided binomial test at α=0.05, N=100: agreement ≥ 19/100.
  - **Empirical null = permutation test** (v3.2 replacement; previous "held-out SFT base-rate" formulation was circular because it used the same pathway labels being tested):
    1. On the same N=100 event test set, shuffle the (pathway, sign) pair assignment 10,000 times while holding the observation labels fixed.
    2. For each permutation, recompute the agreement-rate between the Prop-1 predicted sign and the permuted observed sign.
    3. Report the permutation distribution of agreement-rate (mean, 95th percentile).
    4. α=0.05 threshold: observed agreement must exceed the 95th percentile of the permutation distribution.
- **Pass criterion**: H1 passes only if observed agreement rejects BOTH (i) uniform null p=0.125 at α=0.05 AND (ii) permutation null at α=0.05.
- **Falsification**: Either null fails to reject (uniform OR permutation). Paper reports both thresholds and observed agreement.
- **Exclusivity caveat (new in v3.2)**: If pathways co-occur in >5 % of held-out meta-events (i.e., the 4-pathway partition is not exclusive in practice), downgrade H1 to a "dominant pathway classification" hypothesis and record pathway-exclusivity violation as a formal limitation in §7.
- **Strong prediction (not required)**: agreement ≥ 70 % (highly aligned bridge)
- **Power**: at N=100, p_alt=0.5 vs p_null=0.125 → power > 0.999; even p_alt=0.35 vs 0.125 → power ≈ 0.95. Sample size adequate. Permutation null has equal or higher power because its null distribution is tightly concentrated around the marginal agreement-rate under random assignment.

### H2 (EAD-Main > naive)
- 1030-problem 16k eval, seed=×2
- **Claim**: EAD-Main vs naive D2 — Overall ≥ +3 pp (paired per-problem comparison), AIME ≥ +5 pp, meta wrap rate ≥ 95 %, AIME truncation ≤ 20 %
- **Statistical test** (Sug-7): **McNemar paired test** on per-problem correctness (handles intra-problem dependence across seeds); **paired bootstrap 95 % CI** on accuracy difference as supplementary.
- **Falsification**: EAD-Main Overall ≤ naive D2 + 1 pp (McNemar p > 0.1 OR bootstrap 95 % CI includes 0)

### H2b (B×F interaction — tests the claimed novelty; new in v3.2)
- **Motivation**: EAD-Main 의 핵심 novelty 주장은 "commit-quality B 필터와 control-critical F 가중의 **결합**이 각 축 단독보다 우위" 라는 점이다. H2 는 EAD-Main > Naive-D2 만 보므로, B 단독 또는 F 단독으로도 이 이득을 낼 수 있다면 novelty 가 사라진다. H2b 는 이 interaction 을 직접 검증한다.
- **Claim**: EAD-Main accuracy − max(EAD-B, EAD-F-prior) ≥ +1.5 pp Overall, paired McNemar test across 1030 problems at α=0.05 for both seeds.
- **Pass criterion**: EAD-Main > max(EAD-B, EAD-F-prior) + 1.5 pp Overall AND McNemar p < 0.05 for each seed independently.
- **Falsification**: EAD-Main ≤ max(EAD-B, EAD-F-prior) + 0.5 pp Overall OR McNemar p > 0.1 in either seed. Falsification implies the "B × F combination" framing is not empirically distinguishable from either single axis; novelty claim weakens to "B OR F alone suffices".
- **Interpretation**: H2b 가 falsify 되면 H2 (EAD-Main > Naive-D2) 는 그대로 유지되지만, paper framing 은 "단일 축 접근이 충분" 으로 재조정되어야 한다.

### H3 (B filter effect isolation — W3 revised)
- **Claim**: EAD-Main vs (EAD-Main \ B axis) — B filter 제거 시 truncation 증가, Δentropy 감소
- **τ sweep prediction**: truncation rate monotonically 증가 τ=0.7→0.3
- **Falsification**: |ΔΔentropy(τ=0.5 vs τ=0)| < 0.05 nats AND truncation rate difference < 3 pp

### H4 (C contrastive signal additive)
- **Claim**: EAD-Main vs EAD-Main+C (N3 integration) — AIME +2 pp 이상 또는 calibration ECE ≥ -0.02 개선
- **Falsification**: 두 metric 모두에서 개선 < 1 pp / < 0.01

### H5 (D entropy-shape amplification — v2-W1 + v3 strengthened, v3.1 variance-test corrected)
- **Background**: Naive D2 already recovers Δentropy ≈ +0.231 nats (§1.2). For D to show additional effect beyond naive teacher, both mean convergence AND distribution tightening are required.
- **Variance-test domain (corrected)**: The H5 variance test is computed on the **per-problem post-meta 5-token ΔH distribution within each seed** (n≈1030 problems per seed). **Do NOT** compute variance across only 2 seeds per variant — with n=2, the Fisher F variance ratio has 95 % CI ≈ [0.025, 39], so a 0.9 threshold is indistinguishable from noise.
- **Claim**: D 활성화 시 meta-token 후 5-token window ΔH 분포 (i) 평균이 τ_e = +0.30 nats 근접 (|mean − τ_e| < 0.05 nats) AND (ii) per-problem variance 가 naive-D2 baseline 대비 축소.
- **Variance-ratio statistic**: For each seed s ∈ {s1, s2}, compute Var_D(s) and Var_naiveD2(s) over the n≈1030 per-problem ΔH values in that seed. Across seeds, report the **median variance-ratio D / Naive-D2 with a 1000-resample bootstrap 95 % CI** (bootstrap over problems within each seed, aggregated across seeds).
- **Pass criterion**: median ratio < 0.9 **AND** bootstrap upper CI < 1.0.
- **Falsification**: |mean − τ_e| ≥ 0.05 nats in either seed, **OR** bootstrap upper CI > 1.0 in either seed (i.e., variance reduction is not reliably present).

### H6 (Cross-model generalization — C2 revised, v3.2 Pearson-r correlation replaces set-overlap)
- **Claim (weaker than v1 §2 assertion)**: Working note (Llama-3.2-3B Countdown) 와 Meta-CoT (Qwen3-8B MATH) 의 per-pathway observed-frequency 분포가 양의 상관을 보이며, 이는 EV alignment가 단일 모델/도메인 현상이 아니라는 weak evidence.
- **Operationalization (v3.2)**: Per-pathway observed frequency vectors p_Llama, p_Qwen ∈ R^4 (4-pathway 축에서 normalized 되어 Σp_i = 1). Pearson correlation r(p_Llama, p_Qwen) 을 per-meta-event 단위로 1000-resample bootstrap 하여 point estimate 와 95 % CI 를 보고한다.
- **Pass criterion**: r(p_Llama, p_Qwen) ≥ 0.5 AND bootstrap 95 % CI excludes 0.
- **Falsification**: r < 0.3 OR bootstrap 95 % CI includes 0. (즉 "한쪽 모델에서 pathway 0개 관측" 라는 이전 기준의 노이즈 sensitivity 를 Pearson 기반 연속 측정으로 대체.)
- **중요**: H6이 실패해도 H2/H2b/H3/H4/H5 (EAD method 효과)는 독립적으로 유지됨. 즉 cross-model은 **additional contribution** 로 positioning.

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

→ **7 trainable configs × 2 seeds = 14 primary runs**. RL-step300 reuses existing checkpoint (no new compute). Naive-D2 SFT runs 2 seeds for variance estimate.
→ Primary trainable wall-hours: sum of 7 configs = 3 + 4 + 5 + 4 + 4 + 5 + 6 = 31 wall-hours × 2 seeds = **62 run-wall-hours**

**Ablations** (v3.2 correction: per-variant arithmetic explicit; each ablation variant trained at same 300-step budget as primary, ≈ 1.75 h/run/GPU on 4×H200. F-prior is reused from primary EAD-F-prior, so A2 adds only 2 new variants, not 3):

- **A1 B-τ**: 3 τ values (τ ∈ {0.3, 0.5, 0.7}) × 2 seeds × 1.75 h × 4 GPUs = **42 GPU-hours**
- **A2 F-weight**: 2 **new** variants (F-uniform, F-invfreq; F-prior reused from primary EAD-F-prior) × 2 seeds × 1.75 h × 4 GPUs = **28 GPU-hours**
- **A3 D τ_e**: 3 τ_e values (τ_e ∈ {0.15, 0.30, 0.45}) × 2 seeds × 1.75 h × 4 GPUs = **42 GPU-hours**
- **A4 D-only-mask**: 2 seeds × 1.75 h × 4 GPUs = **14 GPU-hours**
- Cross-model Llama/Qwen Countdown probing (non-training analysis): counted under eval, not training GPU-hour budget

→ **Ablation training GPU-hours**: 42 + 28 + 42 + 14 = **126 GPU-hours**

**Compute summary** (v3.2 corrected per-variant arithmetic):
- Primary trainable: 62 run-hours × 4 GPUs = **248 GPU-hours** (14 runs × 4.4 h average × 4 GPU)
- Ablations: **126 GPU-hours** (A1: 42 + A2: 28 + A3: 42 + A4: 14)
- Sub-total clean: **374 GPU-hours**
- Preempt/restart overhead on BSC cluster (+35 %): 374 × 1.35 ≈ 505 GPU-hours
- **Planning budget: ≈ 505 GPU-hours** (= 248 primary + 126 ablation + 35 % preempt cycle reserve; ≈ 126 node-hours on 1 node of 4×H200)
- Explicit reconciliation: 505 − 248 primary − 126 ablation = 131 GPU-hours for 35 % preempt cycle reserve on the combined 374-hour clean budget.
- **Wall-time**: 1 node 126 hrs ≈ 5.3 days; 2-4 nodes parallel ≈ 1.3-2.6 days best case; **5-7 wall-days realistic with preempt cycles**
- Preempt budget: HF checkpoint every 20 steps (~5 min), resume-on-restart driver, up to 5 restart cycles 허용

**Cross-validation**: working note Llama-3.2-3B Countdown metric 재실행 (BU codebase) + Qwen3-8B Countdown Opener probing (A4).

### 5.3 Success / failure criteria

| 레벨 | 성공 | 실패 |
|---|---|---|
| Theory (H1) | Binomial p < 0.05 agreement vs **uniform AND permutation null** (10,000 shuffles on same N=100) | agreement < 19 % OR does not exceed permutation 95th %ile |
| Method (H2) | EAD-Main > Naive-D2 (+3 pp Overall p < 0.05, +5 pp AIME) | ≤ +1 pp Overall or p > 0.1 |
| **Novelty (H2b)** | EAD-Main − max(EAD-B, EAD-F-prior) ≥ +1.5 pp Overall AND McNemar p < 0.05 for both seeds | ≤ +0.5 pp OR McNemar p > 0.1 in either seed |
| Ablation (H3-H5) | B/D/F 각각 제거 시 유의 감소 (> 3 pp on primary metric); H5 variance test는 per-problem 분포 (n≈1030/seed) 위에서 median variance-ratio < 0.9 AND bootstrap upper CI < 1.0 | 모든 제거에서 < 1 pp; H5 의 경우 bootstrap upper CI > 1.0 (either seed) |
| Cross-model (H6) | Pearson r(p_Llama, p_Qwen) ≥ 0.5 AND bootstrap 95 % CI excludes 0 (1000 resamples over meta-events) | r < 0.3 OR bootstrap 95 % CI includes 0 |

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
| Compute over-budget (505 GPU-hrs) | 4-node parallel priority; preempt-defense driver v4 (5 min HF push); ablations sequentially after primary |
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
| Full training runs (7 × 2 seeds = 14 primary + 4 ablation sweeps @ 300 steps each) | **5-7 wall-days** (≈ 505 GPU-hrs with preempt cycles) | Training done |
| Eval + analysis (EV metrics + H1-H6) | 1-2 days | Metric tables |
| Cross-model (Llama Countdown replay + Qwen Countdown probe) | 1 day | Table 3 |
| Paper draft + self-critic + codex-critic iteration | 2-3 days | Submission-ready draft |

**Total**: **14-19 days** for full paper readiness (v1에서 11-12일로 과소추정했던 항목을 현실화; v3.2 compute ≈ 505 GPU-hrs with per-variant ablation arithmetic, ablations trained at same 300-step budget as primary).

## 9. Acceptance checklist for v2

- [x] Intent explicit (§0)
- [x] Theory/Observation/Method gap structured (§1)
- [x] Core claims downgraded to hypotheses (C1 revised, W7)
- [x] Falsifiable hypotheses with binomial tests (H1 revised with permutation null, §4)
- [x] H2b B×F interaction test validating novelty (§4.H2b, §5.3)
- [x] Operational metric suite (§5.1)
- [x] 7-run × 2-seed = 14 primary + 4 ablation sweeps matrix (§5.2)
- [x] B filter τ specification + sweep (§3.1, A1)
- [x] F weighting ablation (§3.2, A2)
- [x] D τ_e empirical tie + sweep (§3.3, A3)
- [x] Success/failure criteria with statistical tests (§5.3)
- [x] NeurIPS page-budget paper structure main/appendix (§6, W7)
- [x] Risk + mitigation (§7)
- [x] Realistic compute ≈ 505 GPU-hrs, 5-7 wall-days timeline (§5.2, §8, C3; v3.2 correction: per-variant ablation arithmetic — A1 42 + A2 28 + A3 42 + A4 14 = 126 ablation GPU-hrs)
- [x] Related work adds Self-Rewarding/Meta-Rewarding/SPPO/Iterative-DPO (§6, W6)

→ v2 → critic re-review → 통과 시 B, D, F 코드 구현 + BU analysis port phase 진입.

---

## A. Diff summary v1 → v2 → v3

| Section | v1 | v2 | v3 |
|---|---|---|---|
| §0 one-line summary | Universal claim | Universal claim | Scoped to Qwen primary + Llama weak |
| §2 C1 | "일치한다" (assertion) | Downgraded to hypothesis | (same as v2) |
| §4 H1 falsification | "50 % 이상 어긋남" | Binomial p<0.05, p_null=0.125, ≥19/100 | **Dual null** (uniform + **permutation test on same N=100**, 10,000 shuffles; replaces circular held-out base-rate); exclusivity caveat if pathway co-occurrence >5 % |
| §4 H2 test | "paired z-test on 1030" | (same) | **McNemar + bootstrap 95% CI** |
| §4 H5 falsification | "수렴 ≤ +0.2 nats" | "|mean - τ_e| > 0.15 nats" | **Both mean (<0.05) AND variance ratio <0.9** (stricter); v3.1: variance test now on **per-problem ΔH within each seed (n≈1030/seed)** with bootstrap 95 % CI upper < 1.0, replacing naive n=2 seed-variance comparison |
| §4 H6 | Strong generalization | Weak "≥2 common pathways" + independence | **Pearson r(p_Llama, p_Qwen) ≥ 0.5 with bootstrap 95 % CI excluding 0**, replaces set-overlap count |
| §3.1 B axis | Undefined τ | Formula + τ sweep {0.3,0.5,0.7} | **+Coefficient justification** |
| §3.2 F axis | Fixed 1.5/1.25/1.10 | F-prior/uniform/invfreq ablation | (same as v2) |
| §3.3 D τ_e | "+0.3 nats" undefined | Empirical ceiling + sweep | (same as v2) |
| §5.2 run count | 8 runs × 2 seeds | "8 runs × 2 seeds" ambiguous | **7 trainable × 2 seeds = 14 runs** clarified |
| §5.2 compute | 56 GPU-hrs, 14 wall-hrs | "400 GPU-hrs" (changelog/body mismatch) | **420 → 470 → 505 GPU-hrs** (v3.2: per-variant ablation arithmetic — A1 42 + A2 28 + A3 42 + A4 14 = 126 GPU-hrs ablation, A2 reuses F-prior from primary so only 2 new variants) |
| §6 paper | Flat section list | 9-p main split | **Budget redistributed** (§3→1, §6→2.5, §7→1) |
| §7 related work | 7 papers | +Self-Rewarding/Meta-Rewarding/SPPO/Iterative-DPO | (same as v2) |
| §8 timeline | 11-12 days | 14-19 days | (same as v2) |
