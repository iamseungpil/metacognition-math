# Meta-CoT V8 Active Plan (2026-04-09)

This is the only active execution plan. Previous plans are superseded.

## 1. Project Goal

Train a Meta-CoT model that exceeds base_sft (67.1%) on math benchmarks through metacognitive control: route switching, overconfidence correction, and stepwise verification.

## 2. Key Discoveries (Chronological)

### 2.1 V6 Era: Theatrical Redirect + Behavioral Mimicry
- gpt-5.4-mini produced 86% theatrical redirect (pre-meta=0) → fixed with gpt-5.4 full
- E19 (V6 SFT): +7.9pp accuracy, but meta only 15.9%, behavioral mimicry (78% say switch, 2% actually switch)
- Root cause: Qwen3 `<think>` habit + 55% straight data suppressing meta

### 2.2 V7 Failure: Think Structure Conflict
- V7 format `<think>A</think> <|meta|>B<|/meta|> <think>C</think>` → **meta 0%**
- Qwen3 refuses to reopen `<think>` after closing it

### 2.3 V8 Success: Meta Inside Think
- V8 format `<think>A <|meta|>B<|/meta|> C</think> \boxed{answer}` → **meta 99.7%**
- E20a (V8 SFT): **76.0% accuracy (+8.9pp), meta 99.7%, verify 72%**
- Structural switch still low (0.4%) — SFT alone doesn't teach real switching

### 2.4 RL History
- E9 (verify reward): **best RL** — 62.1% acc, verify 847, meta 93.2%
- E5 (confidence_revision): best calibration but killed meta to 1.5%
- All E1-E10: 200 steps insufficient, probe-based rewards unstable
- E21 reward design: switch_v2(soft) + verify_v2(template penalty) + meta_floor

### 2.5 Entropy Analysis
- Meta blocks reduce entropy by 0.41 nats (E19)
- Correct samples: -0.50 nats (stronger resolution)
- Incorrect: -0.34 nats (weaker)
- Validates entropy delta as potential future reward signal

## 3. Current State (2026-04-09)

### 3.1 V8 SFT Results

| Model | Overall | GSM8K | MATH500 | Meta% | Verify | Switch |
|---|---|---|---|---|---|---|
| base_sft | 67.1% | — | — | 0% | — | — |
| **E20a (V8, 3ep)** | **76.0%** | **89.0%** | **49.2%** | **99.7%** | 72% | 0.4% |
| E20b (V8, 5ep) | 74.3% | 87.6% | 47.0% | 99.7% | 70% | — |

### 3.2 Infrastructure

| Node | Role | Status |
|---|---|---|
| EVAL | Meta-CoT experiments | veRL installed ✅, E21 TRL screen running |
| TRAIN_B | Baseline experiments | veRL installed ✅, base_matched SFT running |
| E8 | Boltzmann-attention (reserved) | Do not use for Meta-CoT |
| RUN_C | behavior-uncertainty (reserved) | Do not use for Meta-CoT |

### 3.3 veRL Setup
- vLLM 0.19.0 + Ray 2.10.0 installed on EVAL + TRAIN_B
- veRL source from simpleRL-reason (proven GRPO recipe)
- GDPO advantage computation: per-reward normalization
- All imports verified: DataProto, RayPPOTrainer, MetaCotRewardManager, GDPO algos

### 3.4 Data

| Dataset | Samples | Meta% | Think% | Purpose | HF |
|---|---|---|---|---|---|
| v8_meta_inside_think.parquet | 6329 | 100% | 100% | Meta SFT | ✅ |
| v8_base_matched_clean.parquet | 6329 | 0% | 100% | Baseline SFT | ✅ |
| verl_train.parquet | TBD | — | — | RL training (GSM8K+MATH) | TBD |

## 4. Experiment Plan

### Phase 1: E21 TRL Screen (현재 진행 중)

**의도**: E21 reward가 의도대로 작동하는지 빠르게 확인

**가설**: V8 SFT base (meta 99.7%) 위에서 E21 GDPO를 돌리면, meta 유지 + verify/switch 행동 강화

**검증**:
- meta_emission >= 95%
- switch_v2 > 0 (non-zero)
- accuracy 유지
- reward 분포 정상

**Status**: EVAL에서 118/200 steps, ~1.3h 남음

### Phase 2: veRL RL 본실험 (Phase 1 완료 후)

**의도**: veRL로 충분한 steps (2000)의 RL 학습

**가설**: E21 reward가 200-step screen에서 작동 확인되면, 2000 steps에서 verify/switch가 유의미하게 강화됨

**검증**:
| Metric | Target |
|---|---|
| accuracy | > E20a (76.0%) |
| meta_emission | >= 95% |
| verify (non-template) | >= 30% |
| soft switch rate | >= 15% |
| hard structural switch | >= 5% |

**Execution**:

| Node | Experiment | Framework | Base | Reward | Steps |
|---|---|---|---|---|---|
| EVAL | E21 meta-RL | **veRL** GDPO | E20a (V8 SFT) | correctness(1.0) + switch_v2(0.15) + verify_v2(0.3) + conf_traj(0.15) + meta_floor(0.5) | 2000 |
| TRAIN_B | Baseline GRPO | **veRL** GRPO | base_matched (no meta) | correctness(1.0) only | 2000 |

### Phase 3: Analysis + Comparison

**2×2 비교**:

| | SFT only | SFT + RL (2000 steps) |
|---|---|---|
| **No meta** | base_matched (eval pending) | Baseline GRPO |
| **Meta** | E20a (76.0%) | E21 GDPO |

**분석**:
1. Accuracy comparison (GSM8K + MATH500)
2. Meta behavioral analysis (verify/switch/subgoal)
3. Token-level entropy analysis (before/after meta)
4. Confidence calibration (ECE)
5. Per-subset analysis (redirect vs verify origin)
6. 50-sample manual meta quality review

### Phase 4: OOD Evaluation + Test-Time

**의도**: Meta-CoT의 일반화 능력 확인

**Execution**:
1. AIME 2024 eval (custom JSONL, 30 problems)
2. Best-of-N with meta confidence (N=8, confidence-weighted selection)
3. Curriculum difficulty analysis (easy → medium → hard)

### Phase 5: Future Rewards (E21 결과 기반)

| Reward | 조건 | 설명 |
|---|---|---|
| stepwise (answer flip) | E21 switch가 여전히 낮으면 | meta 전후 답 변화에 보상 |
| confidence delta | E21 calibration 부족하면 | 정답조건부 confidence 변화 |
| entropy delta | E21 entropy 분석에서 유효하면 | meta 후 entropy 감소에 보상 |
| template penalty 강화 | verify가 여전히 formulaic이면 | 더 강한 template 감지 |

## 5. E21 Reward Design (Codex 합의)

```
E21: correctness(1.0) + structural_switch_reward_v2(0.15) + verify_outcome_v2(0.3)
     + confidence_trajectory_reward(0.15) + confidence_omission_floor(0.5)
```

### 각 reward의 의도

| Reward | 의도 | 근거 |
|---|---|---|
| correctness | 정답 여부 (anchor) | 모든 RL 기본 |
| switch_v2 | 실제 방법 전환 (soft score, gated) | SFT만으로는 0.4%밖에 안 됨 |
| verify_v2 | 실제 검산 (template penalty) | E9(verify) 최고 RL, template 탈출 필요 |
| conf_traj | calibration | E5 최고 보정, 하지만 meta 억제 위험이라 낮은 weight |
| meta_floor | meta 억제 방지 | E5/E6/E8 모두 meta를 억제한 교훈 |

### 왜 다른 reward는 나중에?

| Reward | 이유 |
|---|---|
| stepwise (answer flip) | 구현 복잡, E21 결과 먼저 |
| entropy delta | forward pass 비용 높음, entropy 분석 데이터 수집 먼저 |
| confidence delta | E5처럼 meta 억제 위험, E21에서 conf_traj로 충분한지 확인 |

## 6. Code Architecture

### RL Frameworks

| Framework | 용도 | 장점 |
|---|---|---|
| **veRL** (primary) | E21 본실험, base GRPO | vLLM 가속, Ray 분산, simpleRL 검증 |
| **TRL** (fallback) | E21 screen, 기존 실험 호환 | 안정적, 검증됨 |

### Key Files

| File | Role |
|---|---|
| `src/training/verl_gdpo.py` | veRL entry: MetaCotRewardManager, GDPO patching |
| `src/training/verl_gdpo_algos.py` | GDPO advantage computation |
| `src/training/verl_gdpo_data.py` | Data preparation for veRL |
| `src/training/grpo_v2.py` | TRL GDPO (fallback) |
| `src/training/rewards.py` | All reward functions (v1 + v2) |
| `configs/verl_gdpo_e21.yaml` | veRL E21 config |
| `scripts/run_verl_gdpo.sh` | veRL launch script |
| `scripts/analyze_entropy_meta.py` | Token-level entropy analysis |
| `scripts/analyze_e11_pilot.py` | Behavioral analysis (V8 fixed) |

## 7. HuggingFace Sync (iamseungpil/metacot)

All checkpoints and data must be pushed to HF after completion.

| Artifact | HF Status |
|---|---|
| v8_meta_inside_think.parquet | ✅ |
| v8_base_matched_clean.parquet | ✅ |
| E20a model (v8_meta_inside_E20a) | pending |
| base_matched SFT model | pending (training) |
| E21 RL checkpoint | pending (after training) |

## 8. Eval Policy

- **항상 4 GPU 전부 사용** — 단일 GPU eval은 느림. `device_map="auto"` 또는 multi-GPU eval 사용.
- vLLM eval 가능 시: `tensor_parallel_size=4`로 4장 활용.
- HF generate eval 시: `accelerate`로 data parallel 또는 `device_map="auto"`.

## 9. Stop Rules

1. Do not launch RL without verifying reward functions on 50 samples first
2. Do not use E8 or RUN_C for Meta-CoT experiments
3. All checkpoints must be HF-pushed before next experiment
4. If meta_emission drops below 50% during RL, stop and increase meta_floor weight
5. If accuracy drops below 65% (base_sft - 2pp), stop and reduce auxiliary rewards
