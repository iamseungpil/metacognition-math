# Meta-RLSD v2 — Experiment Plan (post-critic iteration)

**Date**: 2026-04-17
**Iteration**: v2 (addresses B1-B5 paper fidelity + internal consistency critics)
**Reference**: arXiv:2604.03128 (RLSD / Self-Distilled RLVR)

---

## 0. Intent (의도)

정통 RLSD의 per-token teacher-ratio advantage scaling을 **`<|meta|>...<|/meta|>` block 내부 토큰에 한정**하여 적용. Meta 외 reasoning 토큰은 순수 GRPO reward-only exploration. Teacher는 student와 동일 weights에 **ground truth answer만 privileged context로 주입** (paper §5.1 L606 준수).

## 1. Why — 문제 재진술

**D2 rebuilt**: controller alive (meta 99%, Δ entropy +0.231) but accuracy ↓ (59.8%, AIME 6.7%, 47% truncated).
**RL E21R-v2 step300**: accuracy 유지 (79.8%, AIME 46.7%) but controller broken (wrap 88%, confidence 0.96 고착).
**D3b (control_spans Meta-KL)**: AIME +6.6pp 회복 (6.7→13.3%) but MATH500 −4.6pp, exploration 없음.

**가설**: Meta 토큰에만 teacher guidance + reasoning RL exploration 결합 → **두 체제의 장점 동시** 달성. 정통 RLSD의 leakage theorem (App A.6 Thm 5)에 따르면 sign(Â_t) = sign(A) 로 directional isolation이 meta-only mask에서도 보존됨.

## 2. Method — Canonical RLSD + meta-mask (paper-faithful)

### 2.1 Per-token advantage (Paper Alg. 1 L17 + mask)

$$\Delta_t = \text{sg}\big[\log P_T(y_t | x, a^\star, y_{<t}) - \log P_S(y_t | x, y_{<t})\big]$$
$$w_t = \exp\big(\text{sign}(A_i) \cdot \Delta_t\big)$$
$$\hat{A}_t = A_i \cdot \Big[(1-\lambda) + \lambda \cdot \big(m_t \cdot \text{clip}(w_t, 1-\varepsilon_w, 1+\varepsilon_w) + (1-m_t)\big)\Big]$$

- $A_i = (R_i - \mu_G) / \sigma_G$ — group-relative scalar advantage
- $m_t = 1$ if 토큰 $t$ ∈ `<|meta|>...<|/meta|>` block, else 0
- $\lambda$: **mixing schedule** — initial 0.5 → linear decay to 0 over first 75 steps (25 % of 300) — paper: 50/200 비율 유지
- $\varepsilon_w = 0.2$
- sg = stop-gradient

**Degeneration guarantee** (paper Thm 5 App A.6.3b): λ → 0 이후 vanilla GRPO와 동치. mask는 이 성질 보존 (non-meta 토큰: $(1-\lambda) + \lambda \cdot 1 = 1$, meta 토큰: $(1-\lambda) + \lambda \cdot 1 = 1$ when $w_t → 1$).

### 2.2 PPO outer clip (Paper Eq. 16 — $w_t$를 policy ratio로)

$$\mathcal{L}_{\text{PPO}} = -\mathbb{E}_t\left[\min\big(w_t^{(\pi)} \cdot \hat{A}_t, \text{clip}(w_t^{(\pi)}, 1-\varepsilon_{\text{low}}, 1+\varepsilon_{\text{high}}) \cdot \hat{A}_t\big)\right]$$

where $w_t^{(\pi)} = \pi_{\theta}(y_t|x,y_{<t}) / \pi_{\theta_{\text{old}}}(y_t|x,y_{<t})$ (student current vs. sampling policy).

- $\varepsilon_{\text{low}} = 0.2$, $\varepsilon_{\text{high}} = 0.28$ (paper asymmetric clip from DAPO, §5.1 L601)
- **KL to reference policy = 0** (paper §5.1 L599, omitted)
- Entropy regularization = 0

### 2.3 Teacher — privileged context (paper §5.1 L606, minimal form)

Paper: teacher = same model conditioned on final answer $r$. No instructional framing.

```
Teacher input:
<|im_start|>user
{problem}
Answer: {gold_answer}
<|im_end|>
<|im_start|>assistant
{rollout}
```

- **Answer-only privileged context** (no tutor / no instructions)
- Teacher는 scoring only — rollout 토큰 시퀀스에 대한 per-token log prob 계산, 새 generation 없음
- Student input: plain (no answer)
- **답 복사 방지**: rollout은 student가 이미 생성한 것. Teacher는 토큰 시퀀스 주어진 상태에서 log prob만 계산.

### 2.4 Teacher periodic sync (Paper §5.1 L601-602)

- Teacher `θ_T` = frozen snapshot of student
- **Resync every 10 training steps**: `θ_T ← θ_S` (copy current student weights)
- 사이 10 step은 teacher 완전 freeze

구현: `self.sync_teacher()` callback on `global_step % 10 == 0`.

### 2.5 Reward (minimal — meta signal은 teacher ratio에서)

$$R_i = 1.0 \cdot r_{\text{correct}}(y^{(i)}, a^\star) + 0.2 \cdot r_{\text{meta\_floor}}(y^{(i)})$$

- `correctness_reward`: \boxed{} 답 일치 → 1, else 0
- `meta_floor`: 3-level discrete
  - meta block 없음 → **−0.30** (`meta_floor_no_meta_penalty = -0.30`)
  - meta block ≥ 1 & len < 20 tokens → **0.0** (no bonus, no penalty; reward hacking 방지)
  - meta block ≥ 1 & len ≥ 20 tokens → **+0.20** (full bonus; 실질적 controller 학습 보상)

**Note (v2 → v2.1)**: no-meta penalty 를 plan v1 의 −0.15 에서 **−0.30 으로 강화**. E21R-v2 post-mortem 에서 −0.15 는 wrap ≥ 95 % 유지에 불충분함이 관찰되었다. −0.30 은 M1 과 N3 (contrastive) 가 공유하는 default 이며 production config (`configs/contrastive_meta_rlsd.yaml` 의 `reward_meta_no_penalty: -0.30`) 와 정렬된다.

**기존 v4-shape 8-component 사용 안 함** (cleanse). Meta 관련 신호는 teacher ratio가 담당.

### 2.6 학습 loop

```
per batch:
  1. Sample G=8 rollouts per prompt from student π_θ_old
  2. Compute rewards R_i (correctness + meta_floor)
  3. Compute A_i = (R_i − μ_G) / (σ_G + 1e-8) per prompt group
  4. For each rollout:
     a. Build teacher input (problem + "Answer: {gold}" + rollout)
     b. Teacher forward (frozen snapshot) → per-token log P_T
     c. Student forward (current policy) → per-token log P_S
     d. Δ_t = log P_T − log P_S (stop-grad)
     e. w_t = exp(sign(A_i) · Δ_t)
     f. Clip w_t to [0.8, 1.2]
     g. meta_mask m_t (from <|meta|> regex span on tokenized rollout)
     h. Â_t = A_i · [(1-λ) + λ·(m_t·clip(w_t) + (1-m_t))]
     i. w^π_t = π_θ(y_t) / π_θ_old(y_t) for PPO ratio
     j. L_t = -min(w^π_t · Â_t, clip(w^π_t, 0.8, 1.28) · Â_t)
  5. Loss = mean over active (assistant) tokens of L_t
  6. λ update: λ = max(0, 0.5 · (1 − global_step/75))
  7. If global_step % 10 == 0: sync teacher weights
  8. Gradient step on student
```

### 2.7 Hyperparameters (전체)

| Name | Value | Source |
|---|---|---|
| `num_rollouts_G` | **8** | paper §5.1 |
| `temperature` | **1.0** | paper §5.1 |
| `clip_eps_w` (teacher ratio) | 0.2 | paper |
| `clip_eps_low` (PPO) | 0.2 | paper |
| `clip_eps_high` (PPO) | **0.28** | paper asymmetric |
| `lambda_init` | 0.5 | paper |
| `lambda_final` | 0.0 | paper |
| `lambda_decay_steps` | 75 (25% × 300) | paper 50/200 비율 |
| `lr` | 1e-6 | paper §5.1 |
| `kl_coef` (to ref) | **0.0** | paper §5.1 |
| `entropy_coef` | **0.0** | paper §5.1 |
| `teacher_sync_freq` | **10 steps** | paper §5.1 |
| `warmup_steps` | 10 | ε_high asymmetric 완전 적용 전 |
| `batch_size` | 64 (× G=8 → 512 rollouts) | budget |
| `gradient_accumulation` | 1 | |
| `total_steps` | 300 | 이전 E21R-v2와 동일 |
| `max_response_length` | 4096 | |
| `prompt_length` | 2048 | |
| `teacher_forward_dtype` | bfloat16 | VRAM |
| `seed_list` | **[42, 43]** | statistical significance |
| `eval_interval` | 50 steps | |
| `save_interval` | 100 steps | |

### 2.8 Numeric stability (safety rails)

1. **Log-ratio clip pre-exp**: `Δ_t = clamp(Δ_t, -10, 10)` → prevents overflow when P_S tiny (exp(10)≈22026, bf16 safe)
2. **Gradient clip**: `max_grad_norm = 1.0`
3. **Reward sanity**: skip prompt if std_G < 1e-6 (degenerate group)
4. **Meta length enforcement**: 3-level meta_floor (see §2.5)

### 2.9 VRAM budget (H200 143GB per GPU, DDP × 4)

| Component | Size | Note |
|---|---|---|
| Student weights (bf16) | 16 GB | Qwen3-8B |
| Student Adam state (fp32) | 32 GB | m + v |
| Student gradients (bf16) | 16 GB | |
| Teacher frozen (bf16) | 16 GB | periodic snapshot, no grad |
| Activation (G=8 × seq 6144 × bs 16 local) | ≈30 GB | recompute + bf16 |
| **Total per GPU** | **≈110 GB** | H200 143GB — **fits with ~30GB headroom** |

Global batch: 64 prompts × G=8 rollouts = 512 rollouts / step.
Per-GPU batch: 16 prompts × G=8 = 128 rollouts (DDP × 4).

### 2.10 Data pre-flight checks (abort if fail)

Before first training step, validate `data/verl_train_redirect.parquet`:
- **PF1**: 100 % rows have non-empty `gold_answer` string
- **PF2**: `\boxed{...}` extraction regex matches gold for ≥ 99 % of rows (canonicalization sanity)
- **PF3**: tokenized `(problem + "Answer: " + gold)` ≤ 2048 for 100 % rows
- **PF4**: tokenizer의 `<|meta|>` / `<|/meta|>` special token이 vocab에 존재
- **PF5**: `meta_rate` on train data > 0 (if using for sanity — not required for fresh rollouts)

실패 시 **training abort**, error log.

## 3. Hypotheses (v2 — falsification threshold 명시)

**H1: Meta-only RLSD preserves controller AND eliminates token exhaustion**
- Target: meta_rate ≥ 95 %, AIME truncation ≤ 10 %, Δ entropy ≥ +0.15
- **Falsification**: meta_rate < 90 % OR truncation > 20 % OR Δ entropy < 0

**H2: Privileged teacher context amplifies over non-privileged**
- Compare M1 (privileged) vs A1 (same model, no answer injection)
- Prediction: Overall Δ ≥ +2 pp, AIME Δ ≥ +3 pp in favor of M1
- **Falsification**: |Δ Overall| < 1 pp AND |Δ AIME| < 2 pp (no significant difference)

**H3: Meta-only mask + privileged > Full-token + privileged (exploration argument)**
- Compare M1 vs A2a (privileged everywhere mask)
- Prediction: M1 AIME ≥ A2a AIME + 2 pp (reasoning exploration 자유)
- **Falsification**: M1 AIME ≤ A2a AIME − 1 pp (strictly worse by ≥ 1 pp — mask actively hurts)

**H4: Meta-only RLSD achieves combined RL + SFT best**
- Target: Overall ≥ 75 %, AIME ≥ 40 %, meta_rate ≥ 95 %, truncation ≤ 10 %, boilerplate share < 30 %
- **Falsification (composite)**: fails ≥ 2 of [Overall ≥ 70, AIME ≥ 30, meta_rate ≥ 90, truncation ≤ 20]

## 4. Verification

### 4.1 Metrics per run

1. **1030-problem 16k eval** (동일 파이프라인):
   - Overall, AIME, GSM8K, MATH500 accuracy
   - Meta wrap rate (per-bench)
   - Avg completion length tokens (p50, p90, max)
   - AIME truncation 비율 (tokens ≥ 16380)
   - no_boxed_rate

2. **Entropy signature** (`scripts/analyze_entropy_meta.py`, marker=meta, window=8):
   - Δ entropy (all / correct / incorrect split)
   - Target: Correct Δ ≥ +0.15 nats, Incorrect Δ ≥ 0 (no forced certainty)

3. **Controller quality**:
   - Confidence 분포 (mode, entropy over emitted values)
   - Boilerplate share (가장 흔한 assessment top-1 빈도)
   - Redirect correction rate (redirect subset eval)

4. **Training dynamics (per 10 steps logging)**:
   - `teacher_ratio_mean`, `teacher_ratio_std` on meta tokens
   - `clip_fraction` (fraction of tokens where w_t hit clip boundary)
   - `meta_token_fraction` (fraction of rollout tokens inside meta)
   - `lambda_current`, `kl_to_old_policy`
   - `A_scalar_mean/std`, `loss_mean`
   - **Abort criteria**: `clip_fraction > 50 %` for 20 consecutive steps → auto-stop

### 4.2 Ablation (5 runs × 2 seeds = 10 jobs)

| Run | Teacher | Mask | Reward | 목적 |
|---|---|---|---|---|
| **M1** (main) | Privileged (answer only) | meta only | correct + meta_floor | 제안된 method |
| **A1** | Same-weights, no answer | meta only | correct + meta_floor | privileged 기여 |
| **A2a** | Privileged (answer only) | **all tokens** | correct + meta_floor | mask 기여 (with privileged) |
| **A2b** | Same-weights, no answer | **all tokens** | correct + meta_floor | classical RLSD variant without priv (paper replica 사촌) |
| **A4** | — (no teacher) | — | v4 8-component (기존 RL) | **non-ablation 참고 baseline** — 2축 차이 (reward + teacher), causal 귀속 금지, 단순 컨텍스트용 |

Confound 분리:
- M1 vs A1 → privileged 기여도 (mask 고정, teacher 유무)
- M1 vs A2a → mask 기여도 (privileged 고정, mask 범위)
- A1 vs A2b → privileged 기여도 in non-privileged setting (교차 체크)
- A2a vs A2b → 완전 RLSD에서 privileged 기여도

### 4.3 Smoke test acceptance (before full run)

10 prompts × 100 steps:
- ✓ Loss finite for all steps
- ✓ meta-token 비율 5-50 % of rollout tokens (sanity)
- ✓ clip_fraction: **5-30 %** (아니면 hyperparameter 재검토)
- ✓ teacher_ratio_mean ∈ [0.95, 1.05] after step 20
- ✓ A_scalar_mean 절대값 ≥ 0.3 (non-degenerate reward)
- ✓ Gradient norm finite for all steps
- ✓ Meta wrap rate ≥ **90 %** at step 100 (D2 init is 99 %; collapse floor to detect catastrophic early drift)

## 5. Implementation plan

### 5.1 새 모듈 — `src/training/meta_rlsd_trainer.py`

```python
class MetaRLSDTrainer(GRPOTrainer):
    """Meta-only token-level RLSD with privileged teacher (paper-faithful)."""
    
    def __init__(self, teacher_init_path, privileged_answer=True,
                 clip_eps_w=0.2, clip_eps_low=0.2, clip_eps_high=0.28,
                 lambda_init=0.5, lambda_decay_steps=75,
                 teacher_sync_freq=10, **kwargs):
        super().__init__(**kwargs)
        self._init_teacher(teacher_init_path)
        self.privileged_answer = privileged_answer
        self.clip_eps_w = clip_eps_w
        self.clip_eps_low = clip_eps_low
        self.clip_eps_high = clip_eps_high
        self.lambda_init = lambda_init
        self.lambda_decay_steps = lambda_decay_steps
        self.teacher_sync_freq = teacher_sync_freq
    
    def _init_teacher(self, path): ...  # bfloat16, eval(), no_grad
    def _sync_teacher_from_student(self): ...  # copy student weights → teacher snapshot
    def _compute_lambda(self, step):
        return max(0.0, self.lambda_init * (1 - step / self.lambda_decay_steps))
    
    def _build_teacher_inputs(self, problems, gold_answers, rollouts):
        if self.privileged_answer:
            return [f"{p}\nAnswer: {g}" for p, g in zip(problems, gold_answers)]
        else:
            return problems
    
    def _compute_teacher_logprobs(self, inputs, rollouts): ...  # extra forward
    def _build_meta_mask(self, rollout_tokens): ...  # reuse kl.py regex
    def _compute_per_token_advantage(self, R, log_T, log_S, meta_mask, step):
        A_scalar = self._group_relative(R)
        delta = torch.clamp(log_T - log_S, -10.0, 10.0).detach()
        sign_A = torch.sign(A_scalar).unsqueeze(-1)
        w = torch.exp(delta * sign_A)
        w_clip = torch.clamp(w, 1 - self.clip_eps_w, 1 + self.clip_eps_w)
        lam = self._compute_lambda(step)
        per_token_factor = meta_mask * w_clip + (1 - meta_mask)
        A_token = A_scalar.unsqueeze(-1) * ((1 - lam) + lam * per_token_factor)
        return A_token
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # 1. rollouts from sampling
        # 2. compute rewards
        # 3. compute log_T (teacher), log_S (student policy)
        # 4. compute per-token A_t
        # 5. PPO-clip on w^π = exp(logp_new − logp_old.detach())
        # 6. periodic teacher sync
        ...
```

### 5.2 Data pipeline

- Input: `data/verl_train_redirect.parquet` (2935 rows, gold_answer field)
- Pre-check: gold_answer format canonicalization (`str(x).strip()` + regex 유효성)
- Batch collator: tokenize (question + rollout), mark assistant span, mark meta span

### 5.3 Config — `configs/meta_rlsd_m1.yaml`

```yaml
# M1 (main run — paper-faithful meta-RLSD + privileged)
student_init: checkpoints/self_distill_rebuilt_d2_epistemic_h200
teacher_init: checkpoints/self_distill_rebuilt_d2_epistemic_h200  # same weights
privileged_answer: true
lambda_init: 0.5
lambda_decay_steps: 75
clip_eps_w: 0.2
clip_eps_low: 0.2
clip_eps_high: 0.28
teacher_sync_freq: 10
kl_coef: 0.0
entropy_coef: 0.0
lr: 1.0e-6
num_rollouts: 8
temperature: 1.0
max_response_length: 4096
prompt_length: 2048
batch_size: 64
total_steps: 300
seed: 42
eval_interval: 50
save_interval: 100
train_data: data/verl_train_redirect.parquet
val_data: data/verl_val_redirect.parquet
output_dir: checkpoints/meta_rlsd_m1_seed42
reward:
  correctness: 1.0
  meta_floor_no_meta_penalty: -0.30   # meta block 없을 때 (v2.1: strengthened from -0.15 per E21R-v2 post-mortem; aligned with N3 default)
  meta_floor_short_meta_bonus: 0.0    # meta 있지만 <20 tokens (no bonus no penalty)
  meta_floor_full_bonus: 0.20         # meta ≥ 20 tokens
  meta_min_length: 20                  # threshold between short and full
```

5개 config × 2 seed = 10 config 파일.

### 5.4 Launcher — `scripts/launch_meta_rlsd.sh`

```bash
#!/bin/bash
# Run one of {m1, a1, a2a, a2b, a4} with seed {42, 43}
set -u
: "${VARIANT:=m1}"
: "${SEED:=42}"
CONFIG=configs/meta_rlsd_${VARIANT}_seed${SEED}.yaml

python -m torch.distributed.run --nproc_per_node=4 --standalone \
  src/training/meta_rlsd_trainer.py --config $CONFIG
```

## 6. Risk & Mitigation (v2)

| Risk | Mitigation | Reference |
|---|---|---|
| Teacher forward 2× 계산 (compute) | bf16 teacher, same tokenizer, shared embedding where possible | §2.3 |
| Privileged prompt의 정답 leak (teacher가 답 복사) | Teacher는 scoring only, new generation 없음; 정답은 prompt에만 (answer-only) | §2.3 |
| Meta mask alignment drift (tokenizer 불일치) | Teacher와 student 같은 tokenizer (guaranteed: same model) | §5.1 |
| Ratio 폭주 ($(P_T/P_S)^{\pm 1}$ overflow when P_S tiny) | **Log-ratio clip [-10, 10] pre-exp** | §2.8 |
| Gradient norm blow-up | `max_grad_norm = 1.0` | §2.8 |
| Degenerate group (all rollouts same R) | Skip batch if σ_G < 1e-6 | §2.8 |
| Meta reward hacking (1-token meta 블록) | `meta_min_length = 20 tokens` threshold | §2.5 |
| Teacher drift with student (θ_T stale) | Periodic sync every 10 steps | §2.4 |
| Gold answer format mismatch (teacher 접근 불가) | Pre-flight canonicalization check | §5.2 |
| KL to ref (plan v1) 적용 시 paper와 불일치 | `kl_coef = 0` 고정 | §2.2 |
| Asymmetric clip 누락 | ε_low=0.2, ε_high=0.28 명시 | §2.7 |
| Single-seed noise | **2 seeds (42, 43)** for main comparisons | §2.7 |
| λ schedule 없으면 degeneration 보장 없음 | `lambda_init=0.5, decay_steps=75` | §2.1 |
| clip_fraction 과도 (training unstable) | Abort criteria: >50 % for 20 consecutive steps | §4.1 |
| Meta-only mask 효과 불확실 (empirical untested) | **A2a/A2b ablation으로 직접 측정** | §4.2 |

## 7. Timeline (revised)

| Phase | Duration | Milestone |
|---|---|---|
| Plan critic iteration (done) | 0.5 day | v2 passes critic agents |
| Code scaffold | 1 day | `MetaRLSDTrainer` + data pipeline |
| Code critic (iterative-code-review) | 0.5 day | Reviewer agent approves |
| Smoke test (10 prompts × 100 steps) | 1 hr | Acceptance criteria §4.3 met |
| Full M1 run (300 steps) | 3 hr | seed 42 |
| Parallel ablations (A1, A2a, A2b, A4) | 4 hr × 4 nodes | seed 42 |
| Re-run top 2 variants seed 43 | 3 hr | statistical check |
| Eval (all 10 runs × 16k eval) | 1 hr each | 10 hrs |
| Entropy + report analysis | 1 day | §6 update with findings |

Total: ~5 days.

## 8. Open Questions (해결 완료)

| Q | v1 상태 | v2 해결 |
|---|---|---|
| Teacher answer prompt wording | 미정 | Answer-only, no framing (§2.3) |
| Meta mask robustness | 미정 | `kl.py:_META_BLOCK_RE` 재사용, tokenizer 동일 (§5.1) |
| Per-token advantage 분산 | 미정 | Group normalization + clip, plus 2-seed (§2.8, §2.7) |
| Ratio ≈ 1 failure monitoring | 미정 | Training dynamics logging + abort (§4.1) |

## 9. Acceptance for next phase (coding)

Plan v2가 다음 조건 모두 만족하면 coding phase 진입:
- ✅ 모든 hypothesis에 falsification threshold
- ✅ Ablation confound 해소 (A2 split → A2a/A2b)
- ✅ Paper 5 blocking issues (B1-B5) 해결: λ, sync, PPO clip, KL, privileged context
- ✅ Numeric stability + reward hacking 방어
- ✅ Smoke test acceptance criteria
- ✅ 2-seed minimum

→ **모든 항목 해결됨. 코딩 단계 진입 가능.**
