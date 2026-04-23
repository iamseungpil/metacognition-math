# Contrastive Meta-RLSD — Experiment Plan v1

**Date**: 2026-04-17
**Variant**: N3 — Contrastive Privileged Teacher
**Builds on**: `plan_meta_rlsd_v2_2026_04_17.md` (M1 baseline)

---

## 0. Intent

Meta-RLSD M1 baseline은 **단일 privileged teacher** ($T = $ student + correct answer)의 per-token ratio를 meta 토큰 advantage에 곱. Contrastive variant **N3**는 privileged teacher 쌍 — $T^+$ (correct answer 주입) vs $T^-$ (오답 decoy 주입) — 의 log-likelihood ratio를 사용:

$$\Delta_t = \text{sg}\big[\log P_{T^+}(y_t|x, a^\star, y_{<t}) - \log P_{T^-}(y_t|x, a^-, y_{<t})\big]$$

여기서 $a^-$ = 자동 생성한 **plausible incorrect answer** (decoy).

**가설**: $P_{T^+}/P_{T^-}$ = **정답 vs 오답 hypotheses의 Bayes factor** — per-token "정답 방향 기여도" 더 선명. Single-teacher 대비 더 구별 가능한 signal.

## 1. Why — 문제 동기

### M1의 한계
- $P_T(y_t|x, a^\star)$ vs $P_S(y_t|x)$ ratio는 **teacher vs student 차이** 측정
- 근데 teacher = same weights (student init) → $P_T \approx P_S$ at init → 초기 ratio $\approx 1$ → signal 약함
- λ decay 후 순수 GRPO로 degenerate

### N3의 차별화
- $P_{T^+}/P_{T^-}$ = **두 privileged teacher 분포 차**
- Init 시점에도 non-trivial ratio (서로 다른 answer 조건)
- $y_t$가 정답 방향이면 $P_{T^+} > P_{T^-}$ → ratio > 1 → advantage 증폭
- $y_t$가 오답 쪽 or 중립이면 ratio ≈ 1 → 효과 없음
- **Bayes factor of hypotheses** — 정보량 더 큼

### 선행 연구와의 차별화

N3의 novel 기여는 두 가지다. (i) **deterministic answer-conditioned decoy** $a^- = f(a^\star, \text{seed})$ 를 통해 contrastive signal 을 per-token Bayes-factor of answer-hypotheses 로 해석할 수 있게 하고, leakage-isolation 증명 (§2.5) 으로 정보 누설을 제거한다. (ii) meta-span token 에만 restricted 되어 non-meta GRPO exploration 을 보존한다.

이 두 축은 선행 literature 와 다음과 같이 구별된다.
- RLCD (arXiv:2307.12950): prompt pair (pos/neg principle) → preference → DPO. **Trajectory-level, no answer conditioning** — per-token Bayes factor 해석이 불가하고 answer-conditioned leakage proof 도 필요하지 않다.
- REDI (arXiv:2505.24850): 정답/오답 trace pair → REINFORCE. **SFT-style pair loss**, per-token advantage scaling 없음.
- DistiLLM-2: teacher↑ / student↓. **Non-RL**, contrastive 해석 다름.

Axes B (commit-quality 필터) 와 F (control-critical 가중) 와의 통합 이득 (즉 A$\wedge$B$\wedge$C$\wedge$F 가 각 축 단독을 초과하는지) 은 EAD-paper 차원에서 **H2b (B×F interaction test)** 로 따로 검증되며, 본 N3 sub-plan 의 novelty 주장은 위 (i)-(ii) 의 per-token Bayes-factor + meta-span restriction 에 한정한다.

## 2. Method — Contrastive Privileged RLSD

### 2.1 Decoy (incorrect answer) 생성

`make_decoy_answer(gold, seed)` — **deterministic function** of gold + seed:

```python
def make_decoy_answer(gold: str, seed: int = 42) -> str:
    """Deterministic decoy: f(gold, seed) → decoy.
    
    Guarantees:
    (A) decoy != gold (string)
    (B) decoy not numerically equivalent to gold
    (C) Deterministic: same (gold, seed) → same decoy
    """
    rng = random.Random(hash((gold, seed)) & 0xFFFFFFFF)  # per-gold deterministic seed
    s = gold.strip()
    candidates: list[str] = []
    
    # Strategy 1: integer perturbation
    if re.fullmatch(r"-?\d+", s):
        n = int(s)
        for delta in [1, -1, 2, -2, 10, -10, 5, -5]:
            c = str(n + delta)
            if c != s: candidates.append(c)
    
    # Strategy 2: float perturbation
    elif re.fullmatch(r"-?\d+\.?\d*", s):
        v = float(s)
        for delta in [0.1, -0.1, 1.0, -1.0, 0.5, -0.5]:
            c = str(round(v + delta, 2))
            if c != s: candidates.append(c)
    
    # Strategy 3: LaTeX constants
    if "\\pi" in s:
        candidates.append(s.replace("\\pi", ""))
        candidates.append(s.replace("\\pi", "\\pi/2"))
    if "\\sqrt" in s:
        candidates.append(re.sub(r"\\sqrt\{(\d+)\}", r"\1", s))
    
    # Strategy 4: fraction manipulation (skip palindromes a/a)
    m = re.match(r"\\?frac\{(-?\d+)\}\{(-?\d+)\}", s)
    if m and m.group(1) != m.group(2):  # skip \frac{a}{a}
        # swap numerator/denominator
        swapped = f"\\frac{{{m.group(2)}}}{{{m.group(1)}}}"
        if swapped != s: candidates.append(swapped)
        # perturb numerator by +1
        candidates.append(f"\\frac{{{int(m.group(1))+1}}}{{{m.group(2)}}}")
    
    # Strategy 5: sign flip (excluding 0)
    if s.startswith("-") and s != "-0":
        candidates.append(s[1:])
    elif s not in {"0", "0.0"}:
        candidates.append("-" + s)
    
    # Strategy 6: fallback — append "+1"
    if not candidates:
        candidates.append(f"({s})+1")
    
    # Filter: decoy != gold (both string + numerical equivalence)
    def numerically_equal(a: str, b: str) -> bool:
        try: return abs(float(a) - float(b)) < 1e-9
        except ValueError: return False
    
    valid = [c for c in candidates 
             if c != s and not numerically_equal(c, s)]
    if not valid:
        # Absolute fallback — guaranteed different string
        return s + " + 1"  # symbolic, cannot equal gold
    
    # Deterministic pick
    return rng.choice(valid)
```

**Guarantees** (testable):
- (A) `decoy != gold` strictly (string compare)
- (B) Not numerically equivalent (e.g., `2/1 != 2` ruled out by Strategy 4 palindrome check + numerical check)
- (C) `f(gold, seed)` deterministic — required for §2.5 leakage proof
- (D) `decoy` is syntactically parse-valid as an answer string

Edge cases covered:
- `\frac{2}{2}` → palindrome skip → fallback sign flip or `+1` suffix
- `0`, `-0`: explicit handling
- Integer `3` + possible `3` from another strategy: numerical check filters
- `\frac{2}{1}` vs `2`: caught by `numerically_equal`

**Invariant**: Strategies 1–6 produce mutually disjoint output classes by construction:
- S1 (integer) + S2 (float) + S5 (sign) → numeric strings; `numerically_equal` float path covers
- S3 (`\pi`/`\sqrt`) + S4 (fraction) → LaTeX strings with specific tokens; distinct structural class from S1/S2/S5
- S6 (+1 suffix) → explicit non-numeric expression
- No cross-strategy symbolic collision (e.g., `\sqrt{4}` from S3 never meets gold `2` from S1 path since S3 requires `\sqrt` in input)
- ∴ `numerically_equal(a, b)` float-based check suffices; no sympy parse needed.

### 2.2 Per-token advantage

M1 수식 확장:
$$\Delta_t = \text{sg}\big[\log P_{T^+}(y_t|x, a^\star, y_{<t}) - \log P_{T^-}(y_t|x, a^-, y_{<t})\big] \in [-10, 10]$$
$$w_t = \exp(\text{sign}(A_i) \cdot \Delta_t)$$
$$\hat{A}_t = A_i \cdot \Big[(1-\lambda) + \lambda \cdot \big(m_t \cdot \text{clip}(w_t, 1-\varepsilon_w, 1+\varepsilon_w) + (1-m_t)\big)\Big]$$

$\varepsilon_w = 0.2$ 유지. Bayes factor가 더 변동 크므로 clip은 더 중요.

### 2.3 Teacher 구성

Teacher = 단일 frozen model ($T^+ = T^- = $ same weights), 다만 input context 다름:

```
T+ input: <|im_start|>user {problem} Answer: {gold}  <|im_end|>
          <|im_start|>assistant {rollout}

T- input: <|im_start|>user {problem} Answer: {decoy} <|im_end|>
          <|im_start|>assistant {rollout}
```

- **Forward compute**: 2× teacher (T+, T-) + 1× student = **3× forward** per step
- $T^\pm$ 같은 weights 공유 → 모델 하나만 저장 (context만 교체)
- Periodic sync 동일 (every 10 steps)
- **Sequential forward** (메모리 절약): T+ forward → capture log_T+ → GPU memory 회수 → T- forward → capture log_T- → 다시 회수

**VRAM budget (H200 143GB, DDP × 4, N3 specific)**:

| Component | M1 (single teacher) | **N3 (contrastive)** | Δ |
|---|---|---|---|
| Student weights (bf16) | 16 GB | 16 GB | — |
| Student Adam (fp32) | 32 GB | 32 GB | — |
| Student gradient (bf16) | 16 GB | 16 GB | — |
| Teacher weights (bf16) | 16 GB | 16 GB (shared) | — |
| Student activation (G=4 × seq 6144 × bs 4 local) | ~15 GB | ~15 GB | — |
| Teacher activation (peak, sequential fwd) | ~8 GB | **~8 GB × 2 sequential = ~8 GB peak** | — |
| **Total per GPU** | **~103 GB** | **~103 GB** (sequential!) | **0 GB** |

핵심: **Sequential forward + activation free between T+/T-** 하면 peak memory 동일. `torch.no_grad()` + 중간 free 보장. H200 143GB 여유 40GB+.

Batch/G 값은 M1 smoke와 동일 (G=4, batch=4 local). Full run (G=8)에서는 sequential 유지 필수.

### 2.4 수식 - Bayesian 해석 (v2 corrected)

Teacher = student weights 가정. Paper A.5에 따르면:
$$\frac{P_T(y_t|x, r)}{P_S(y_t|x)} = \frac{P(r|x, y_{\le t})}{P(r|x, y_{<t})}$$

이걸 양변에 로그 취하면:
$$\log\frac{P_T(y_t|x, r)}{P_S(y_t|x)} = \underbrace{\log P(r|x, y_{\le t}) - \log P(r|x, y_{<t})}_{\Delta_r(y_t): \text{token } y_t \text{의 "} r \text{" posterior 기여}}$$

이것을 $r = a^\star$ (정답) 및 $r = a^-$ (오답 decoy)에 각각 적용:
$$\log\frac{P_{T^+}}{P_S} = \Delta_{a^\star}(y_t), \quad \log\frac{P_{T^-}}{P_S} = \Delta_{a^-}(y_t)$$

Student marginal $P_S$가 분모에서 **상쇄**:
$$\log\frac{P_{T^+}}{P_{T^-}} = \Delta_{a^\star}(y_t) - \Delta_{a^-}(y_t)$$

**의미**: token $y_t$가 "정답 $a^\star$의 posterior를 증가시키는 정도" **빼기** "오답 $a^-$의 posterior를 증가시키는 정도" = **순수 정답-방향 기여도** (correct likelihood gain - incorrect likelihood gain).

- $y_t$가 정답 쪽 reasoning → $\Delta_{a^\star} > 0$, $\Delta_{a^-} < 0$ → 큰 양수
- $y_t$가 중립 → 두 항 모두 0 → Δ_t ≈ 0
- $y_t$가 오답 쪽 → $\Delta_{a^\star} < 0$, $\Delta_{a^-} > 0$ → 큰 음수

M1은 $\Delta_{a^\star}$만 사용 (single teacher). N3은 $\Delta_{a^\star} - \Delta_{a^-}$ → **신호 polarization 증대**, Bayes factor of hypotheses.

### 2.5 Leakage isolation — Theorem 확장

**Proposition (Contrastive RLSD leakage-free)**:
M1의 Thm 5와 동일하게 4개 속성 유지, 단 **decoy deterministic 조건** 추가 필요.

**Precondition**: $a^- = f(a^\star, \text{seed})$ 는 **decoy function이 deterministic** (same seed + same gold → same decoy). §2.1 `make_decoy_answer`에서 `rng = random.Random(hash((gold, seed)) & 0xFFFFFFFF)` 로 보장.

Proof sketch:
- (i) **Direction isolation**: $\text{sign}(\hat{A}_t) = \text{sign}(A_i) = \text{sign}(R_i - \mu_G)$. $R$ = env reward (correctness $\pm 1$ + meta_floor ∈ {-0.30, 0, +0.20}). 모두 $a^\star$만 사용 (answer-conditioned for direction OK since it's env reward). Clip factor $> 0$이므로 sign 보존. ∎
- (ii) **Magnitude bounded**: $|\hat{A}_t/A_i| \in [1-\varepsilon, 1+\varepsilon]$ by clip. ∎
- (iii) **Support isolation**: $y \sim \pi_S(\cdot|x)$. $a^\star$와 $a^-$ 둘 다 teacher context에만 주입 (student 입력에는 없음). Sampling support에 등장 불가. 
  - **Additional requirement satisfied**: decoy가 stochastic이면 $\hat{A}_t$에 새 random channel 발생 가능 → leakage 경로. Precondition ("$a^- = f(a^\star, \text{seed})$ deterministic")으로 이 경로 차단. 즉 ${a^\star} \to a^-$ 는 bijective deterministic map → $\hat{A}_t$는 여전히 $(y_t, x, y_{<t}, a^\star)$의 함수, 새 random source 없음. ∎
- (iv) **Graceful degeneration**: $\lambda \to 0$ 시 $\hat{A}_t = A_i$ → pure GRPO, contrastive signal 소멸. ∎

**Additional property (N3 unique)**:
- **Contrastive signal bounded by KL divergence**: $|\Delta_t|$ bounded by $\max_t KL(P_{T^+} \| P_{T^-})$ — decoy가 정답과 너무 다르면 ratio 과대 → clip이 방지.
- **Decoy robustness**: decoy가 "plausible"할수록 $|\Delta_t|$ 더 의미 있게 분포. Random noise decoy면 signal 약화.

### 2.6 Hyperparameters (M1 기반 + N3 추가)

| Name | M1 | N3 (초기값) | 정당화 |
|---|---|---|---|
| clip_eps_w | 0.2 | 0.2 (동일) | Paper 기본 |
| lambda_init | 0.5 | **0.5 (M1과 동일)** | Clean A/B: N3 effect만 isolate. Lambda tuning은 별도 sweep에서. |
| lambda_decay_steps | 75 | 75 | 동일 |
| teacher_sync_freq | 10 | 10 | 동일 |
| decoy_seed | — | **42** (고정, deterministic) | Leakage isolation precondition |
| decoy_strategy | — | **rule_based** (§2.1) | Default; H3 ablation으로 `random` 비교 |
| teacher_forward_mode | single | **contrastive** (2× sequential) | Peak VRAM 유지 |

**Note**: `lambda_init`은 M1과 같은 0.5로 유지하여 "contrastive signal effect"만 isolate. λ tuning은 post-hoc sweep (원하면 Phase 2).

## 3. Hypotheses (falsifiable)

### H1 (primary): Contrastive signal > Single signal
- **Prediction**: N3 accuracy (Overall, AIME) ≥ M1 + 2 pp on same smoke/full setup
- **Falsification**: N3 ≤ M1 − 1 pp (contrastive signal hurts more than helps)
- **Metric**: 1030 eval Overall + AIME

### H2: Controller preservation equivalent
- **Prediction**: meta wrap rate, entropy Δ within ±5% of M1
- **Falsification**: significant controller degradation (wrap < 90%, Δ < +0.15 nats)

### H3: Decoy quality matters
- **Ablation**: N3 with random-noise decoy vs rule-based decoy
- **Random baseline definition** (operational):
  - Integer gold: uniform random int in `[-100, 100]` with $\neq$ gold enforced
  - Float gold: uniform random float in `[gold−10, gold+10]`, excluding near-gold (Δ < 0.1)
  - LaTeX gold: random token sequence of matching character length from `[0-9a-zA-Z\\pi\\sqrt\\frac]`
  - Fallback: `random.randint(1, 1000)`
  - All deterministic via same `random.Random(hash((gold, "random", seed)))`
- **Prediction**: rule-based > random by ≥ 2pp (plausibility hypothesis)
- **Falsification**: |rule-random| < 1 pp → contrastive signal itself not informative

### H4: Bayes factor variance
- **Prediction**: $\text{std}(\Delta_t)$ on meta tokens > M1 by 2× (더 큰 signal)
- **Falsification**: variance 유사 → contrastive 효과 없음

### H5: Exploration preservation
- **Prediction**: Non-meta token entropy during rollout ≥ M1 (meta-only mask 보존)
- **Falsification**: 전체 reasoning entropy 감소 (mask 실패)

## 4. Verification

### 4.1 Metrics

1. **Standard** (M1과 동일): 1030 eval Overall/AIME/GSM8K/MATH500, meta wrap rate, entropy Δ, AIME truncation, no_boxed rate
2. **N3-specific**:
   - Decoy uniqueness rate (decoy == gold 비율; 목표 < 1%)
   - $\Delta_t$ 분포 통계 (mean, std, kurtosis)
   - $\log(P_{T^+}/P_{T^-})$ distribution on meta tokens
   - Clip fraction (meta token level)

### 4.2 Smoke acceptance (plan §4.3 + N3 추가)

기존 M1 acceptance criteria +:
- ✓ Decoy generation success rate ≥ 98% (실패 = fallback trigger)
- ✓ `decoy_is_correct_rate` < 5% (decoy가 우연히 정답되는 비율 — precondition 검증)
- ✓ `decoy != gold` 100% (절대 조건)
- ✓ No NaN/Inf in contrastive teacher forwards (T+ 및 T- 각각)
- ✓ $\Delta_t$ variance ∈ [0.01, 10] (너무 작으면 signal 없음, 너무 크면 clip 과다)
- ✓ $\text{std}(\Delta_t) > \text{std}(\Delta_t^{M1})$ (H4 direct check — N3 signal이 M1보다 더 큼)
- ✓ Teacher forward count = 2× per training step (log `contrastive_teacher_fwd_count`)
- ✓ `clip_fraction` on meta tokens ∈ [0.05, 0.40] (contrastive 더 큰 signal → upper bound 약간 완화)
- ✓ $\log \text{KL}(P_{T^+} \| P_{T^-})$ finite and > 0 on meta tokens

### 4.3 Ablation matrix

| Run | Teacher | Mask | Decoy | 비교 목적 |
|---|---|---|---|---|
| **M1** | single T (answer) | meta_only | — | baseline |
| **N3** | T+ / T- (contrastive) | meta_only | rule-based | main novelty |
| **N3-random** | T+ / T- | meta_only | random-noise (H3 defined) | H3 decoy quality |
| **N3-fullmask** | T+ / T- | **all tokens** | rule-based | mask × contrastive interaction (**not** paper-faithful RLSD; that's `A2b`) |
| A4 (기존) | — | — | — | non-ablation ref (v4 shape RL) |

**중요 distinction**: `N3-fullmask`는 4번째 변형 (contrastive × all tokens). Paper RLSD (arXiv:2604.03128) 과 다름 — paper는 single-teacher + all tokens. 직접 비교 대상은 M1 ablation의 `A2b` (single + non-priv + all tokens).

5 runs × 2 seeds = 10 jobs.

**Parallelization assumption**:
- 3 H200 nodes available (eval + train_b + 1 more) → ~3 jobs concurrent
- 10 jobs × 3 hr / 3 nodes ≈ **10 wall-hours** (serial within node, parallel across nodes)
- Single node: **30 wall-hours** (serial)
- 자동 job queue + HF push 활용하여 세션 간 복구 가능

## 5. Implementation

### 5.1 코드 변경 — MetaRLSDTrainer 확장

**Overrides 필요한 method (전체 list)**:
1. `__init__` — decoy config 추가
2. `_build_teacher_inputs` → `_build_contrastive_teacher_inputs` (T+ + T- 둘 다 생성)
3. `_compute_teacher_logprobs` → sequential 2× forward
4. `_compute_per_token_advantage` — $\Delta_t$ 공식 변경
5. `_log_metrics` — contrastive 특화 로깅 (`delta_t_mean`, `delta_t_std`, `kl_T+_T-`, `decoy_is_correct_rate`, `contrastive_teacher_fwd_count`)

```python
class ContrastiveMetaRLSDTrainer(MetaRLSDTrainer):
    """N3: contrastive privileged teacher (T+/T- pair) with meta-only mask."""
    
    def __init__(self, *args, decoy_strategy="rule_based", decoy_seed=42, **kwargs):
        super().__init__(*args, **kwargs)
        self.decoy_strategy = decoy_strategy
        self.decoy_seed = decoy_seed
    
    def _make_decoy(self, gold: str) -> str:
        """§2.1 deterministic decoy. Raises ValueError if unable to generate valid decoy."""
        ...
    
    def _build_contrastive_teacher_inputs(self, prompts, gold_list, rollout_ids):
        pos_text = [f"{p} Answer: {g}" for p, g in zip(prompts, gold_list)]
        decoys = [self._make_decoy(g) for g in gold_list]
        neg_text = [f"{p} Answer: {d}" for p, d in zip(prompts, decoys)]
        # Tokenize both, concatenate with rollout
        pos_input = self._tokenize_concat(pos_text, rollout_ids)
        neg_input = self._tokenize_concat(neg_text, rollout_ids)
        return pos_input, neg_input, decoys
    
    @torch.no_grad()
    def _teacher_contrastive_logprobs(self, pos_input, neg_input, completion_ids, prompt_lens):
        """Sequential forward T+ then T-, release activation in between."""
        # T+ forward
        logp_T_pos = self._teacher_logprobs(
            pos_input["input_ids"], pos_input["attention_mask"],
            completion_ids, prompt_lens[0]
        )
        torch.cuda.empty_cache()  # release T+ activation
        
        # T- forward  
        logp_T_neg = self._teacher_logprobs(
            neg_input["input_ids"], neg_input["attention_mask"],
            completion_ids, prompt_lens[1]
        )
        return logp_T_pos, logp_T_neg
    
    def _compute_per_token_advantage(self, rewards, log_T_pos, log_T_neg, log_S, meta_mask, step):
        """Override M1: Δ_t = log(P_T+/P_T-) instead of log(P_T/P_S)."""
        A_scalar = self._group_relative(rewards)  # from M1
        delta = torch.clamp((log_T_pos - log_T_neg).detach(), -10.0, 10.0)  # **new Δ**
        sign_A = torch.sign(A_scalar).unsqueeze(-1)
        w = torch.exp(delta * sign_A)
        w_clip = torch.clamp(w, 1 - self.clip_eps_w, 1 + self.clip_eps_w)
        lam = self._compute_lambda(step)
        per_token_factor = meta_mask * w_clip + (1 - meta_mask)
        A_token = A_scalar.unsqueeze(-1) * ((1 - lam) + lam * per_token_factor)
        
        # Logging (new metrics)
        self._log_contrastive_metrics(delta, w_clip, meta_mask, log_T_pos, log_T_neg)
        return A_token
    
    def _log_contrastive_metrics(self, delta, w_clip, meta_mask, log_T_pos, log_T_neg):
        meta_delta = delta[meta_mask.bool()]
        self.log({
            "delta_t_mean": meta_delta.mean().item(),
            "delta_t_std": meta_delta.std().item(),
            "kl_T+_T-": (torch.exp(log_T_pos) * (log_T_pos - log_T_neg))[meta_mask.bool()].sum().item(),
            "contrastive_teacher_fwd_count": 2,  # per step
        })
```

### 5.2 Config — `configs/contrastive_meta_rlsd.yaml`

M1 config 복사 + 추가:
```yaml
trainer_class: ContrastiveMetaRLSDTrainer
decoy_strategy: rule_based  # or "random"
decoy_seed: 42
lambda_init: 0.5  # match M1 for clean A/B comparison
# rest: same as M1
```

### 5.3 CLI variant

```python
# main()에서 variant에 추가
elif variant == "n3":
    cfg.trainer_class = "ContrastiveMetaRLSDTrainer"
    cfg.decoy_strategy = "rule_based"
elif variant == "n3-random":
    cfg.trainer_class = "ContrastiveMetaRLSDTrainer"
    cfg.decoy_strategy = "random"
elif variant == "n3-fullmask":
    cfg.trainer_class = "ContrastiveMetaRLSDTrainer"
    cfg.decoy_strategy = "rule_based"
    cfg.mask_mode = "all_tokens"
```

### 5.4 Cost-Benefit analysis

- **Cost**: +1 teacher forward per step = +~33% step time (student 3× teacher forward로 ~10s/step → ~13s/step)
- **Benefit target (H1)**: ≥ +2 pp Overall accuracy vs M1
- **Go/no-go**: N3 ≥ M1 + 2 pp → 유지. N3 < M1 + 1 pp → drop, M1 baseline으로 복귀.

## 6. Risk & Mitigation

| Risk | Mitigation |
|---|---|
| 2× teacher forward VRAM 압박 | Single teacher model + context 교체 (가중치 공유); gradient checkpointing |
| Decoy가 gold와 동일 | `make_decoy` 엄격 check + fallback |
| Decoy signal이 noise에 불과 | H3 ablation (random vs rule-based); abort if `decoy_is_correct_rate > 5%` (decoy가 실제 답이 되는 경우) |
| $\Delta_t$ 분산 과대 | log_ratio clamp [-10, 10] 그대로 사용 |
| Leakage (Thm 확장) | §2.5 증명 — Thm 5 성질 4개 유지 |
| M1 코드 regression | ContrastiveMetaRLSDTrainer는 subclass — M1 코드 불변 |

## 7. Open Questions

1. **Decoy 품질**: rule-based로 충분한가? 또는 "previous incorrect answer seen during rollout" 사용?
2. **λ tuning**: M1과 동일 0.5로 시작 (clean A/B). Phase 2에서 sweep 고려.
3. **Ablation scale**: 5 variants × 2 seeds = 10 runs × 3시간 = 30 GPU-hours. 현재 인프라에서 완주 가능?

## 8. Acceptance for coding phase

plan v2 승인 조건:
- ✅ Intent explicit (§0)
- ✅ 모든 hypothesis에 falsification threshold (§3)
- ✅ Theorem extension 증명 스케치 (§2.5)
- ✅ Decoy 알고리즘 구체화 (§2.1)
- ✅ Ablation matrix (§4.3)
- ✅ Risk + mitigation (§6)

→ Critic 통과 시 코딩 phase 진입.

---

## 9. Addendum — 2026-04-17 critique loop (v1 → v1.1)

v1 대비 추가된 항목. 코딩/실행 phase에서 반드시 반영.

### 9.1 Intent 정합: Working-note Theorem 8과의 연결

**Claim (γ restoration corollary)**. Working note Theorem 8(4) (Alignment 실패 분기)은 $\gamma_i \le -\gamma_\text{min} < 0$일 때 $\mathbb{E}[\Delta U_T] < 0$을 보인다. N3의 contrastive advantage $\hat{A}_t = A_i \cdot [(1-\lambda) + \lambda(m_t w_t' + (1-m_t))]$은 meta 토큰 $y_t$에 대해 $w_t' = \text{clip}(\exp(\text{sign}(A_i) \cdot \Delta_t), 1\pm\varepsilon_w)$ 가중을 건다. $\Delta_t = \log P_{T^+} - \log P_{T^-}$는 §2.4에서 "token $y_t$의 정답 posterior 기여 − 오답 posterior 기여"로 해석되므로, 기댓값 수준에서 $\mathbb{E}_{y_t \sim \pi_S}[\Delta_t \mid \text{correct trace}] > 0$이 성립하는 meta 토큰들에 대해 PPO update의 policy 그래디언트가 gold-aligned 방향으로 편향된다. 이 편향이 누적되면 해당 경로의 $v_{t_e}$(학습된 marker value)가 $\nabla U$ 방향으로 이동하고 $\gamma := \mathbb{E}[\langle v_{t_e} - x^\text{old}, \nabla U\rangle]$이 음에서 양으로 flip할 수 있다. 즉 N3는 Theorem 8의 실패 분기에서 정렬 분기로 넘어가는 경로를 제공한다.

**이 연결은 H1 (accuracy)의 메커니즘적 이유를 준다**. accuracy 상승의 원인이 단순히 teacher signal이 아니라 ``γ 부호 복원''이라는 것이 가설. 이 가설을 직접 검증하는 추가 실험이 §9.3 H6.

### 9.2 Preflight (PF6) — 자원 존재 확인

기존 PF1–PF5에 다음 PF6 추가. smoke 시작 전 실패 시 aboriginal stop:

- PF6a. `student_init` path 접근 가능 (본 설정: `/scratch/metacognition/checkpoints/self_distill_rebuilt_d2_epistemic_h200/`). 없으면 §9.5 staging 절차 선행.
- PF6b. `train_data` parquet 존재 + schema 검증 (`problem`, `solution`, `ground_truth` 열).
- PF6c. Meta tokens `<|meta|>` / `<|/meta|>` 이 tokenizer에 있고 non-special (§2.10 기존 PF 재사용).
- PF6d. Device budget 체크: 4×H200 available + 총 free VRAM ≥ 500 GB (기존 PF3 확장).
- PF6e. `make_decoy_answer` self-test: 20개 샘플 gold에 대해 A–D guarantee 만족. 실패 시 즉시 abort.

### 9.3 H6 (stretch): γ probe의 direct flip 검증

기존 H1–H5가 behavioral/signal 수준이라면 H6은 **representation 수준** 검증.

- **Setup**: smoke 완료 후 student checkpoint에서 committed\_path target 3072-dim logistic probe를 20 샘플 meta 토큰 위치에 학습, $\gamma_\text{N3}$ 측정. 동일 절차를 M1 baseline checkpoint에도 적용.
- **Prediction**: $\gamma_\text{N3} > \gamma_\text{M1}$ (즉 N3가 정렬 방향으로 더 많이 이동)
- **Falsification**: $\gamma_\text{N3} \le \gamma_\text{M1}$ — contrastive signal이 representation 수준 정렬을 만들지 못함. H1 positive여도 H6 negative면 "어떤 다른 메커니즘"이 accuracy 상승의 원인임을 암시.
- **Smoke 단계에서는 optional** (full run 이후 수행). smoke 내 log는 γ direct 측정 없이도 Δ_t 분포로 간접 추정.

### 9.4 Smoke 자원 + 시간 budget

기존 §4.2 acceptance criteria를 compute budget으로 보완:

- **Target**: 50 steps (LR warmup 완료 시점) × ~13s/step ≈ **11분/1GPU** (M1 10s 대비 +3s per step).
- **4×H200 DDP**: global batch 4, G=4 rollouts → **~25분 wall-time**.
- **Hard cutoff**: 45분 안에 50 steps 완료 못 하면 abort (teacher forward deadlock 가능성).
- **Storage**: smoke 체크포인트 ≤ 20GB (not to push to HF). Run log + metrics 만 HF에 push.

### 9.5 Node staging 절차 (E8 target)

E8 (`/scratch` empty) → 실행 전 다음 절차 수행:

```bash
# 1. HF login (token)
export HF_TOKEN=<token>

# 2. Pull code snapshot (tarball name on HF: metacot_code.tar.gz)
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='iamseungpil/metacot', repo_type='dataset',
                  local_dir='/scratch/meta/hf',
                  allow_patterns=['code_snapshot/metacot_code.tar.gz'])
"
mkdir -p /scratch/meta/code && tar xzf /scratch/meta/hf/code_snapshot/metacot_code.tar.gz -C /scratch/meta/code

# 3. Pull D2 checkpoint
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='iamseungpil/metacot', repo_type='dataset',
                  local_dir='/scratch/meta/hf',
                  allow_patterns=['models/self_distill_rebuilt_d2_epistemic_h200/**'])
"
ln -s /scratch/meta/hf/models/self_distill_rebuilt_d2_epistemic_h200 \
      /scratch/meta/code/checkpoints/self_distill_rebuilt_d2_epistemic_h200

# 4. Pull train data
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='iamseungpil/metacot', repo_type='dataset',
                  local_dir='/scratch/meta/hf',
                  allow_patterns=['data/**/verl_train_redirect.parquet'])
"
# (link data path accordingly)
```

기존 `metacognition.tar.gz` snapshot이 HF에 없으면, node 진입 후 `git clone` 또는 `amlt code download` 사용.

### 9.6 Tokenization boundary invariant

§2.3 `_build_contrastive_teacher_inputs`의 `f"{decoded} Answer: {gold}"` 연결에는 한 가지 위험이 있다: tokenizer가 leading whitespace를 별개 토큰으로 처리하면 `gold`의 tokenization이 standalone tokenize와 다를 수 있고, 이는 `logprob` 계산에서 정답/오답을 비대칭하게 만든다.

**Smoke assertion (must-pass)**: pos/neg prompt의 마지막 $|answer|$ 토큰이 standalone `tokenizer(gold, add_special_tokens=False)`의 토큰과 완전 일치. 불일치 비율 >5%이면 staging으로 abort.

구현: `tests/test_contrastive_meta_rlsd.py::test_answer_token_boundary` 로 CPU 수준에서 검증.

### 9.7 Ablation 경계 명확화

`N3 vs T+-only`는 **현 M1과 다르다**. 기존 M1 = $\log(P_{T^+} / P_S)$ (teacher-vs-student ratio). `T+-only` (decoy 없이 $\log P_{T^+}$ 절대값) 는 또 다른 변형으로, scope 밖. 본 실험에서는 M1 (ratio) vs N3 (contrastive pair) 비교만 primary. `T+-only` 는 future work.

### 9.8 Periodic monitoring + incremental HF push (resume-safe 실행)

BSC 노드는 idle 감지 시 30분마다 재시작될 수 있고, E8 할당도 preempt될 수 있음. 따라서 실행 스크립트는 **resume-safe** 구조여야 한다.

**디자인 원칙**
- 모든 artifact (체크포인트, 로그, metrics.json, stats snapshot) 는 생성 즉시 HF에 push.
- Daemon 실행 스크립트가 주기적으로 (**20분 간격**) 다음 작업을 반복:
  1. 최근 수정된 artifact 감지 (mtime 비교)
  2. HF upload (commit msg에 timestamp)
  3. 노드 health check: `nvidia-smi`, disk usage, process liveness
  4. 실패 감지 시 summary 로그를 HF에 push하고 escalate

**Push 대상 경로 (HF repo: `iamseungpil/metacot`, subpath `n3_runs/<run_id>/`)**
- `config.yaml` (immutable, push once)
- `stdout.log` (tail every 20 min)
- `stderr.log`
- `metrics.jsonl` (per-step scalar log)
- `smoke_acceptance.json` (§4.2 checklist 결과)
- `checkpoint-*/` (`save_interval` 시점, compressed)
- `final_metrics.json` (완료 시)

**Resume 절차**
- 노드 재시작 시 HF에서 최신 checkpoint + metrics.jsonl 다운로드 → `--resume_from_hf` 플래그로 이어서 실행.
- 체크포인트가 없으면 처음부터 재시작 (smoke 짧으므로 OK).

**구현**: `scripts/n3_monitor_push.sh` — `run_in_background=true`로 daemon 실행. `scripts/run_smoke_n3.sh`와 병렬 실행.

### 9.9 Rollback / kill policy

- Smoke 실패 시 (PF 또는 acceptance): 실패 마커 파일 `/scratch/meta/run/FAILED` 생성, summary를 HF에 push, 프로세스 종료. 노드는 해제 안 함 (재실행 가능 유지).
- 5회 연속 실패 시: 원인 분석 후 사람 개입 요청 (slack/message 로그 기록 = HF push).
- Leakage 의심 증거 (decoy_is_correct_rate > 5%, answer token boundary mismatch > 5%): 즉시 abort + 재설계.

### 9.10 v1.1 Acceptance

v1 criteria 유지 + 추가:
- ✅ Theorem 8 연결 (§9.1)
- ✅ PF6 preflight (§9.2)
- ✅ H6 representation-level hypothesis (§9.3)
- ✅ Compute budget (§9.4)
- ✅ Node staging procedure (§9.5)
- ✅ Tokenization invariant + test (§9.6)
- ✅ Ablation scope 명확화 (§9.7)
- ✅ Resume-safe monitoring + incremental HF push (§9.8)
- ✅ Rollback / kill policy (§9.9)

→ v1.1 승인; 코딩 phase로 진입.

### 9.11 Loop #2 critic — 남은 잠재 이슈와 해결
1. M1 checkpoint이 HF에 없음 → H6 내 M1 비교는 full run 단계로 deferred. Smoke 단계에서는 N3 vs **base student** 비교로 충분 (Δ_t 분포만으로 signal 유무 판정).
2. `metacot_code.tar.gz` 실존 확인됨 (§9.5 fix 반영).
3. Tarball이 오래된 경우: 실행 전 `git rev-parse HEAD` 출력 HF push — 코드 버전 추적.
4. Smoke 자체에 M1 dual-run 포함하지 않음 (compute 2× 절약). M1 parity 확인은 기존 M1 smoke log (HF `results/meta_rlsd_smoke/`) 참조.

v1.1에서 남은 이슈 없음. 코딩 phase 시작.

