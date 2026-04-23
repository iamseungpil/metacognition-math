# Meta-RLSD (Meta-only Token-level RLSD) — Experiment Plan

**Date**: 2026-04-17
**Status**: Draft v1 → iterate with agent critics until passing

---

## 0. Intent (의도)

`<|meta|>...<|/meta|>` block **내부 토큰에 대해서만** 정통 RLSD (arXiv:2604.03128)의 per-token teacher-ratio advantage scaling을 적용. Meta 외 reasoning 토큰은 순수 GRPO reward-only exploration.

**Teacher = Privileged Context**: Student와 동일 가중치 모델에 **ground truth answer를 system prompt로 주입**. Teacher는 정답을 알고 있는 상태에서 forward pass.

## 1. Why — 현 문제 정리

**D2 rebuilt (§4.5 보고서)의 트레이드오프**:
- Meta wrap rate 98.9%, entropy Δ +0.231 (controller alive) ✓
- BUT Overall 59.8%, AIME 6.7% (accuracy −7pp vs old), AIME 47% truncated

**RL E21R-v2 step300의 트레이드오프**:
- Overall 79.8% (accuracy 유지) ✓
- BUT meta wrap 88.2%, confidence 0.96 고착 98.9%, boilerplate 88.2%, AIME 13/30 truncated

**이전 방법들의 한계**:
| 방법 | Meta signal | Reasoning signal | 한계 |
|---|---|---|---|
| D3b (control_spans KL) | teacher 분포 | teacher sequence (SFT) | exploration 없음, data scale 작음 |
| RL v4-shape | reward only | reward only | controller 깨짐 |
| 정통 RLSD (원본) | teacher everywhere | teacher everywhere | meta 아닌 토큰도 teacher 따라 — exploration 제한 |

**핵심 가설**: Meta 토큰에만 teacher guidance + reasoning은 exploration을 결합하면 **정확도 (RL 수준) + controller 보존 (SFT 수준) 동시** 달성 가능.

## 2. Method — Meta-only Token-level RLSD + Privileged Teacher

### 2.1 수식

Per-token advantage:
$$\hat{A}_t = A_i \cdot \Big[ m_t \cdot \text{clip}\Big(\big(\tfrac{P_T(y_t|x_{<t},a^\star)}{P_S(y_t|x_{<t})}\big)^{\text{sign}(A_i)}, 1-\varepsilon_w, 1+\varepsilon_w\Big) + (1-m_t) \Big]$$

여기서:
- $A_i$ = rollout $i$의 group-relative advantage (기존 GRPO)
- $m_t = 1$ if 토큰 $t$ ∈ `<|meta|>...<|/meta|>` block, else 0
- $P_T(y_t | x_{<t}, a^\star)$ = **teacher conditioned on ground truth answer** $a^\star$
- $P_S(y_t | x_{<t})$ = student (no answer access)
- $\varepsilon_w = 0.2$

**의미**:
- Meta 토큰: teacher가 정답 알고 있을 때의 분포로 ratio 계산 → 정답 방향으로 advantage 크기 조정
- Non-meta 토큰: $m_t = 0$ → 기본 advantage 그대로 (exploration 유지)

### 2.2 Teacher 구성 (Privileged Context, 방식 1)

```
Teacher input format:
<|im_start|>system
You are a math tutor who uses <|meta|>confidence/diagnosis/action<|/meta|>
blocks to guide reasoning. The correct final answer is: {gold_answer}.
Produce meta-wrapped reasoning that naturally arrives at this answer.
<|im_end|>
<|im_start|>user
{problem}
<|im_end|>
<|im_start|>assistant
{rollout}  # student's sampled completion

Student input format (no privileged context):
<|im_start|>user
{problem}
<|im_end|>
<|im_start|>assistant
{rollout}
```

Teacher forward pass는 rollout의 actual tokens를 target으로 per-token $\log P_T$ 계산.

**답 복사 방지 장치**:
1. Teacher는 이미 생성된 student rollout을 scoring만 함 (new generation 없음) — system prompt의 정답 정보가 rollout에 "주입"될 수 없음
2. Teacher는 학생의 토큰 시퀀스를 받아 $P_T(y_t|x_{<t}, a^\star)$만 계산
3. 즉 teacher는 정답을 알면서 "이 토큰 시퀀스가 그 정답에 얼마나 잘 접근하는가"를 logprob으로 표현

### 2.3 Reward (순수화)

기존 `compute_score_e21r_v4_smoke`의 복잡한 8-component shape를 **최소화**:
- `correctness_reward` × 1.0 (유일한 핵심)
- 선택적: `meta_floor` × 0.2 (meta block 최소 존재 강제)

**Meta guidance는 reward에서 오지 않고 teacher ratio에서 옴** — cleaner separation.

### 2.4 학습 루프

```
1. Student policy에서 N=4 rollouts per prompt (temperature=0.7)
2. Reward 계산: r_i = correctness_reward(rollout_i) [+ meta_floor]
3. Teacher forward pass:
   a. Build privileged prompt (ground truth 주입)
   b. Concatenate (privileged_prompt + rollout_i)
   c. Forward → per-token log P_T
4. Student forward pass (already have from step 1):
   a. Build plain prompt (no answer)
   b. Concatenate (plain_prompt + rollout_i)
   c. Forward → per-token log P_S
5. Compute per-token ratio log(P_T/P_S) = log_T - log_S
6. Build meta_mask for each rollout
7. Compute per-token advantage (Eq. 2.1)
8. PPO-clip loss with per-token advantage
9. Update student (gradient step)
10. Teacher freeze — no update
```

### 2.5 Hyperparameters (초안)

| Name | Value | Source |
|---|---|---|
| `teacher_ratio_clip_eps` ($\varepsilon_w$) | 0.2 | arXiv:2604.03128 |
| `ppo_clip_eps` | 0.2 | GRPO standard |
| `lr` | 1e-6 | 이전 E21R-v2 사용 |
| `kl_coef` (ref policy) | 0.001 | 이전 E21R-v2 |
| `N_rollouts` | 4 | |
| `temperature` | 0.7 | |
| `max_response_length` | 4096 | |
| `prompt_length` | 2048 | |
| `batch_size` | 64 | |
| `total_steps` | 300 | |
| `teacher_forward_dtype` | bfloat16 | |

## 3. Hypotheses (가설)

**H1**: Meta-only RLSD은 D2 rebuilt의 controller를 유지하며 (meta_rate ≥ 95%) AIME truncation 문제 해소한다 (truncation ≤ 10%).
- *Falsification*: meta_rate < 90% 또는 AIME truncation > 20%

**H2**: Privileged teacher context는 meta-only guidance의 효과를 증폭한다.
- A/B: Variant A (teacher = 같은 모델, no answer) vs Variant B (teacher + answer)
- *Prediction*: Variant B가 Variant A보다 AIME 정확도 ≥ +3pp, controller 유사 유지

**H3**: Meta-only (teacher guidance 토큰에 한정) 이 Full-RLSD (모든 토큰 teacher guidance) 보다 exploration 이득이 크다.
- A/B: Meta-only vs Full-RLSD ablation
- *Prediction*: Meta-only의 AIME 정확도 ≥ Full-RLSD +3pp (reasoning이 teacher 분포에 고정되지 않음)

**H4**: 최종 모델은 RL step300과 self-distill의 장점 결합 달성.
- *Target*: Overall ≥ 75%, AIME ≥ 40%, meta_rate ≥ 95%, truncation ≤ 10%, boilerplate < 30%
- *Falsification*: Overall < 65%

## 4. Verification (검증 방법)

### 4.1 Metric 및 측정

1. **1030-problem 16k eval** (동일 파이프라인):
   - Overall / per-benchmark accuracy
   - Meta wrap rate
   - Avg completion length (tokens) + AIME truncation 비율
   - no_boxed_rate

2. **Entropy signature** (`analyze_entropy_meta.py`, marker=meta):
   - Before / meta / after window 8 tokens
   - Δ entropy (correct vs incorrect split)

3. **Controller quality**:
   - Confidence distribution (mode, entropy)
   - Boilerplate share (가장 흔한 assessment의 비중)
   - Redirect correction rate (redirect 서브셋에서)

4. **Teacher ratio 분포** (training dynamics):
   - Log every 10 step: meta 토큰의 $P_T/P_S$ 평균, 분산, clip 비율
   - 과도 drift 방지 (학생이 teacher를 완전히 무시하거나 너무 따라가면 경고)

### 4.2 Ablation study

| Run | Teacher | Mask | Reward | 목적 |
|---|---|---|---|---|
| **M1** (main) | Privileged (answer) | meta only | correctness + meta_floor | 우리 method |
| A1 | Same (no answer) | meta only | correctness + meta_floor | privileged 기여도 |
| A2 | Privileged (answer) | everywhere | correctness + meta_floor | meta-only 기여도 |
| A3 | No teacher | meta only | correctness + meta_floor | teacher 기여도 전체 |
| A4 (baseline) | - | - | v4 8-component (기존 RL) | 기존 RL 비교 |

총 5개 run. M1 + 4개 ablation.

## 5. Implementation plan

### 5.1 새 모듈 — `src/training/meta_rlsd.py`

```python
class MetaRLSDTrainer(GRPOTrainer):
    """Meta-only token-level RLSD with privileged teacher."""
    
    def __init__(self, teacher_model_path, privileged_answer=True, 
                 teacher_ratio_clip_eps=0.2, ...):
        ...
        self.teacher = AutoModelForCausalLM.from_pretrained(teacher_model_path).eval()
        self.teacher.requires_grad_(False)
        self.privileged_answer = privileged_answer
    
    def _build_teacher_input(self, problem, gold_answer, rollout):
        if self.privileged_answer:
            system = f"...correct answer is: {gold_answer}..."
        else:
            system = "..." 
        return [{"role": "system", "content": system},
                {"role": "user", "content": problem},
                {"role": "assistant", "content": rollout}]
    
    def _compute_teacher_logprobs(self, teacher_inputs):
        """Forward teacher, return per-token log probs for assistant tokens."""
        ...
    
    def _compute_meta_mask(self, rollouts):
        """Binary mask: 1 if token in <|meta|>...<|/meta|>"""
        ...
    
    def _compute_per_token_advantages(self, rewards, student_logp, teacher_logp, meta_mask):
        """Eq. 2.1"""
        A_scalar = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        log_ratio = teacher_logp - student_logp
        sign_A = torch.sign(A_scalar).unsqueeze(-1)
        ratio = torch.exp(log_ratio * sign_A)
        clipped = torch.clamp(ratio, 1 - self.eps_w, 1 + self.eps_w)
        per_token_scale = meta_mask * clipped + (1 - meta_mask)
        A_token = A_scalar.unsqueeze(-1) * per_token_scale
        return A_token
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Override to use per-token advantage in PPO-clip."""
        # 1. rollouts already sampled via TRL
        # 2. compute rewards (scalar per rollout)
        # 3. compute teacher + student logprobs (per token)
        # 4. compute per-token advantage
        # 5. PPO-clip with per-token advantage
        ...
```

### 5.2 기존 코드 재사용

- `src/training/self_distill/kl.py`의 `_META_BLOCK_RE`, `find_meta_token_spans` → meta mask
- `src/training/rewards.py::correctness_reward` → reward
- `src/training/verl_reward.py::compute_score_e21r_v4_smoke`의 meta_floor component
- `src/training/self_distill/teacher_query.py` 패턴 (top-k 로직 대신 full logprob)

### 5.3 설정 파일 — `configs/meta_rlsd_main.yaml`

```yaml
# Meta-RLSD main run (M1)
student_init: checkpoints/self_distill_rebuilt_d2_epistemic_h200
teacher_path: checkpoints/self_distill_rebuilt_d2_epistemic_h200  # same weights
privileged_answer: true
teacher_ratio_clip_eps: 0.2
ppo_clip_eps: 0.2
learning_rate: 1.0e-6
kl_coef: 0.001
num_rollouts: 4
temperature: 0.7
max_response_length: 4096
prompt_length: 2048
batch_size: 64
total_steps: 300
train_data: data/verl_train_redirect.parquet
val_data: data/verl_val_redirect.parquet
output_dir: checkpoints/meta_rlsd_m1
wandb_project: metacot-math
run_name: meta_rlsd_m1_privileged
reward_components:
  correctness: 1.0
  meta_floor: 0.2
```

### 5.4 런처 — `scripts/launch_meta_rlsd.sh`

기존 `launch_e21r_v4_commit_shape_0416.sh` 패턴 기반. verl 대신 자체 trainer 사용.

## 6. Risk & Mitigation

| Risk | Mitigation |
|---|---|
| Teacher forward pass 매 step 큼 (double compute) | bfloat16 teacher, teacher model sharing (same device) |
| Privileged prompt의 정답 leak (teacher가 정답 복사) | Teacher는 scoring만 (새 생성 없음), student rollout 고정 |
| Meta mask alignment drift (tokenizer 불일치) | teacher와 student 같은 tokenizer 사용 (guaranteed since same model) |
| Ratio 폭주 (학생 drift) | clip $\varepsilon_w = 0.2$, gradient clip, warmup steps 10 |
| Meta block이 rollout에 없으면 mask 전부 0 → advantage scaling 없음 | correctness_reward + meta_floor로 wrap rate 유지 |
| verl 환경 의존성 | verl 없이 직접 TRL/Transformers로 구현 (train_b 노드에 이미 있음) |

## 7. Timeline

| Phase | Duration | Milestone |
|---|---|---|
| Plan iteration | 0.5 day | Plan passes all critic agents |
| Code scaffold | 1 day | MetaRLSDTrainer class + data pipeline |
| Smoke test (10 prompts) | 0.5 day | 100 step smoke, verify advantage shape |
| Code critics iteration | 1 day | iterative-code-review agent approves |
| Full M1 run | 3 hr | 300 steps, 2935 prompts |
| Ablation A1-A3 | 6 hr | 3 parallel runs |
| Analysis + eval | 3 hr | 16k eval, entropy, compared to table |
| Report | 0.5 day | §6 update with findings |

Total ~4 일.

## 8. Open Questions

1. **Teacher answer prompt의 optimal wording?** 답 복사 방지 + 정보 전달의 balance.
2. **Meta mask가 assistant token sequence에서 어떻게 robust하게 추출?** Tokenizer 경계 처리.
3. **Per-token advantage의 분산 큼 → policy gradient 노이즈?** Group normalization 이후에 clip 이면 완화될지.
4. **Failure mode**: ratio가 모든 meta 토큰에서 1 근처로 수렴하면 학습 신호 없음 → 어떻게 monitor?

---

_[다음 단계: task-planner-analyzer agent로 이 plan의 의도/가설/검증 일관성 review → 수정 → 통과할 때까지 iteration.]_
