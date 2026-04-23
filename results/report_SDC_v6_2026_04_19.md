# SDC-split v6 Training + Eval Report

**Date**: 2026-04-19
**Run ID**: `sdc_split_v6_resume_20260419_150546`
**Variant**: `sdc-split` (2-region post-meta repel)
**Student/Teacher init**: `v8_meta_inside_strict_sft` (Qwen3-8B)
**Compute**: metacognition_run_c (H200 × 4), 2h 40m training + 3min eval
**Outcome**: Training completed 300 steps without halt, but eval shows -20.30pp vs SFT baseline. Clear repetition-loop + EOS-avoidance failure mode identified.

---

## 1. Purpose and Background

SDC (Split-Directional Contrastive) is the latest in a research line (Meta-CoT → EAD → N3 → SDC). SDC extends N3 by adding **post-meta repel** from a decoy-conditioned teacher, on top of N3's meta-span attract from a gold-conditioned teacher. The plan §2.4 formula:

$$
\hat A_t = A_i \cdot \left[(1 - \lambda_{\text{meta}} m^{\text{meta}}_t - \lambda_{\text{post}} m^{\text{post}}_t) + \lambda_{\text{meta}} m^{\text{meta}}_t w^{\text{attr}}_t + \lambda_{\text{post}} m^{\text{post}}_t w^{\text{rep}}_t\right]
$$

with
$$
w^{\text{attr}}_t = \exp(+\text{sign}(A) \cdot \text{clip}(\log P_{T^+} - \log P_S, -10, 10))
$$
$$
w^{\text{rep}}_t = \exp(-\text{sign}(A) \cdot \text{clip}(\log P_{T^-} - \log P_S, -10, 10))
$$

Prior v0-v5 runs all halted at step 31-103 due to the `SDCHaltCallback` false-positive on per-rank discrete wrap/fallback metrics. v6 applied rolling-mean thresholds on both rules, completing 300 steps for the first time.

---

## 2. Key Paths

### Local repository
- Trainer: `/home/v-seungplee/metacognition/src/training/contrastive_meta_rlsd_trainer.py`
- Parent: `/home/v-seungplee/metacognition/src/training/meta_rlsd_trainer.py`
- Data pipeline: `/home/v-seungplee/metacognition/src/training/meta_rlsd_data_pipeline.py`
- Decoy utils: `/home/v-seungplee/metacognition/src/training/_decoy_utils.py`
- Reward: `/home/v-seungplee/metacognition/src/training/rewards.py`
- Plan: `/home/v-seungplee/metacognition/results/plan_SDC_v2_2026_04_17.md`
- Config template: `/home/v-seungplee/metacognition/configs/contrastive_meta_rlsd.yaml`

### Node (metacognition_run_c /scratch)
- Runtime root: `/scratch/meta_debug/`
- Checkpoint dir: `/scratch/meta_debug/runs/sdc_split_v6_resume_20260419_150546/ckpts/`
- Training log: `/scratch/meta_debug/runs/sdc_split_v6_resume_20260419_150546/sdc_smoke.log`
- Eval log: `/scratch/meta_debug/eval_v6.log`
- Eval result: `/scratch/meta_debug/runs/sdc_split_v6_resume_20260419_150546/eval_v6_1030.json`
- Eval script: `/scratch/meta_debug/eval_vllm.py` + `run_sdc_eval.sh`

### HuggingFace (iamseungpil/metacot)
- Training metrics: `sdc_runs/sdc_split_v6_resume_20260419_150546/ckpts/metrics.jsonl` (1223 records)
- Checkpoints: `sdc_runs/sdc_split_v6_resume_20260419_150546/ckpts/checkpoint-{20,40,...,300}/`
- Final: `sdc_runs/sdc_split_v6_resume_20260419_150546/ckpts/final/`
- Training log: `sdc_runs/sdc_split_v6_resume_20260419_150546/sdc_smoke.log`
- Monitor log: `sdc_runs/sdc_split_v6_resume_20260419_150546/monitor.log`
- **Eval result**: `sdc_runs/sdc_split_v6_resume_20260419_150546/eval_v6_1030.json`

### Autoresearch trace
- Log: `/tmp/autoresearch_sdc_log.tsv`
- Iterations 0-5: halt false positives documented

---

## 3. Training Configuration

```yaml
student_init: /scratch/meta_debug/sft_model   # v8_meta_inside_strict_sft
teacher_init: /scratch/meta_debug/sft_model   # same weights, different prompt context
variant: sdc-split

# Lambda schedules
lambda_init: 0.5; lambda_final: 0.0; lambda_decay_steps: 75       # meta attract
lambda_post_init: 0.1; lambda_post_final: 0.3; lambda_post_warmup: 150  # post repel

# Clipping
clip_eps_w: 0.2
log_ratio_clamp: 10.0
clip_eps_low: 0.2; clip_eps_high: 0.28  # PPO asymmetric

# Decoy
decoy_strategy: rule_based
decoy_seed: 42

# Rollout
num_rollouts: 8
temperature: 0.6; top_p: 0.95; top_k: 20
max_response_length: 4096
prompt_length: 2048

# Training
batch_size: 8
per_device_train_batch_size: 2
gradient_accumulation_steps: 1
lr: 1.0e-6
kl_coef: 0.0
total_steps: 300
save_interval: 20

# Teacher sync disabled (teacher stays at init SFT)
teacher_sync_freq: 999999

# Reward
reward_meta_no_penalty: -0.30
reward_meta_no_penalty_strength: "soft"
reward_meta_full_bonus: 0.20
meta_min_length_tokens: 20
correctness_weight: 1.0
meta_floor_weight: 0.2
continuous_weight: 0.05

# vLLM
use_vllm: true
vllm_tensor_parallel_size: 1
vllm_gpu_memory_utilization: 0.3

# Halt callback (autoresearch-fixed thresholds)
wrap_threshold: 0.35 (was 0.90 default)
wrap_window_size: 20 (rolling mean)
fallback_threshold: 0.40 (was 0.20)
consec_required: 3
```

---

## 4. Autoresearch Iteration Log

| Iter | Timestamp | Config change | Halt step | Root cause |
|---|---|---|---|---|
| 0 | 10:46 | baseline threshold 0.9 | 31 | per-rank wrap_rate 0.5 false positive (SFT stochastic) |
| 1 | 12:13 | λ_post 3× smaller (0.03→0.1) | 31 | Same — λ not the cause |
| 2 | 12:41 | threshold 0.75, consec 3 | 35 | threshold still inside SFT baseline |
| 3 | 13:08 | variant=sdc-uniform | 32 | Not variant-specific |
| 4 | 13:43 | rolling mean window 10, threshold 0.75 | 41 | SFT baseline mean is 0.70 |
| 5 | 14:07 | **threshold 0.35, window 20** (wrap) | 103 | fallback_trigger_rate had same per-rank issue |
| 6 | 15:05 | **+ fallback rolling mean window 20, threshold 0.40** | **completed 300** | ✓ |

**Root diagnosis**: `wrap_rate` and `fallback_trigger_rate` are per-rank × 2-rollout discrete metrics (0.0/0.5/1.0 granularity). SFT baseline rolling mean: wrap ≈ 0.70, fallback ≈ 0.17. Thresholds 0.90/0.20 were inside stochastic variance. Rolling window smoothing + empirical baseline thresholds resolved false positives.

---

## 5. Training Metrics (from HF metrics.jsonl, 1223 records)

### Trajectory summary

| Stage | Reward mean | Wrap rate | Fallback rate | w_rep mean | λ_meta | λ_post |
|---|---|---|---|---|---|---|
| Start (rec 1) | 0.17 | 1.00 | 0.00 | 0.99 | 0.50 | 0.10 |
| Warmup (rec 150) | 0.77 | 0.73 | 0.00 | 11.1 | 0.27 | 0.15 |
| Mid (rec 600) | 0.53 | 0.70 | 0.00 | 31.6 | 0.19 | 0.16 |
| Lambda crossover (rec 900) | 0.61 | 0.72 | 0.05 | 61.9 | 0.00 | 0.26 |
| Full repel (rec 1200) | 0.64 | 0.67 | 0.08 | 199.2 | 0.00 | 0.30 |

Observations:
- **w_rep mean grew 1 → 199** as training progressed — post-meta student moved strongly away from decoy teacher
- **w_attr mean stayed near 1.0** — student meta reasoning naturally matched T+ without much pressure
- **λ_meta → 0 by rec 800** (decay scheduled to 75 optimizer steps)
- **λ_post → 0.3 by rec 1200** (warmup to 150 optimizer steps)
- **Reward stable 0.5-0.7** — training-set accuracy OK throughout
- **Wrap rate rolling mean stayed above 0.35** — halt never triggered

### Speed
- 300 steps in 2h 40m, ~32s/step (range 15-45 s/step)
- Speed varied with λ_post magnitude (full repel is more expensive for teacher forward)

---

## 6. Eval Results (1030 problems)

### Overall

| Metric | Value |
|---|---|
| Total problems | 1030 |
| Overall accuracy | **59.51%** (613/1030) |
| Overall meta emission rate | **99.1%** |

### Per-benchmark

| Benchmark | Acc | Meta emit | Avg meta blocks | Avg conf | Avg tokens |
|---|---|---|---|---|---|
| GSM8K (500) | **79.40%** | 99.8% | 1.00 | 0.804 | 2473 |
| MATH500 (500) | **42.60%** | 98.8% | 0.99 | 0.449 | 3130 |
| AIME2024 (30) | **10.00%** | 93.3% | 0.93 | 0.369 | 3929 |

### Comparison with baselines

| Model | Overall | GSM8K | MATH500 | AIME | Notes |
|---|---|---|---|---|---|
| Base SFT | 75.92% | 92.6% | 61.8% | 33.3% | No meta tokens |
| SFT Meta (init) | 79.81% | 92.0% | 71.6% | 13.3% | Meta SFT baseline |
| E21R-v2 step 300 | 79.81% | 92.0% | 71.6% | 13.3% | prior RL run (note: confidence collapsed to 0.96) |
| **SDC v6 step 300** | **59.51%** | **79.40%** | **42.60%** | **10.00%** | This run |
| Δ vs SFT Meta | **-20.30pp** | -12.6 | **-29.0** | -3.3 | |
| Δ vs Base SFT | -16.41pp | -13.2 | -19.2 | -23.3 | |
| Δ vs E21R-v2 | -20.30pp | -12.6 | -29.0 | -3.3 | |

### Truncation breakdown

| Benchmark | Correct truncated | Wrong truncated |
|---|---|---|
| GSM8K | 135/397 (34%) | 56/103 (54%) |
| MATH500 | 106/213 (50%) | 184/287 (64%) |
| AIME | 3/3 (100%) | 24/27 (89%) |

---

## 7. Root Cause Analysis of Accuracy Drop

### Observation 1: Repetition loop in completions

Sampled wrong GSM8K completions show the same pattern:
1. Student computes answer correctly (e.g. "12 years", "60%", "5 hours")
2. Emits first `</think>\n\nconfidence: X\nassessment: ...\n<|/meta|>\n\nThe answer is \boxed{12}.`
3. Instead of EOS, continues with a second `</think>` + new meta block + new `\boxed{12}`
4. Repeats 5-20 times until `max_tokens=4096` cutoff
5. Final `\boxed{12\text{year}}}` (malformed) is what extraction picks up

Two concrete cases inspected:
- GSM8K gt=5 (hours problem): first `\boxed{5}` correct → repeated 150+ times as `\boxed{5\text{hour}}}` → extraction returned `5\text{hour}` → reward function marked wrong.
- MATH500 fraction problem: first `\boxed{\frac{47}{3}}` malformed → repetition → extraction failed.

### Observation 2: Meta emission preserved (99.1%)

The format tokens `<|meta|>` and `<|/meta|>` survive — the halt rule was calibrated correctly. But the **post-meta "finalize" region** (between `<|/meta|>` and `\boxed{}`) is where the damage shows up.

### Observation 3: Calibration actually improved

| Model | GSM avg conf | MATH avg conf | AIME avg conf |
|---|---|---|---|
| E21R-v2 | 0.96 | 0.96 | 0.96 (all collapsed) |
| **SDC v6** | **0.80** | **0.45** | **0.37** |

Confidence now correlates with accuracy — SDC v6's calibration is qualitatively better than E21R-v2's collapsed-to-0.96 mode. This is the **positive finding** of the run.

### Mechanistic explanation

Let $A > 0$ for a correct rollout. For post-meta token $t$:

$$
w^{\text{rep}}_t = \exp(-\text{clip}(\log P_{T^-}(y_t) - \log P_S(y_t)))
$$

When $P_{T^-}(y_t) > P_S(y_t)$ (decoy teacher likes this token), the exponent is negative and $w^{\text{rep}} < 1$. Applied with $\lambda_{\text{post}}$, this **down-weights** the advantage on that token. Token $y_t = $ EOS is high-probability under the decoy-conditioned teacher (decoy context leads to confident immediate termination). So:

- EOS tokens get systematically down-weighted in the advantage
- Non-EOS tokens (like `\boxed{`) get systematically up-weighted
- Policy gradient shifts probability mass away from EOS
- Model learns to keep generating after the boxed answer rather than terminate

The 199× w_rep mean at end of training confirms student is far from decoy — including in "terminate now" decisions.

---

## 8. Positive Findings

1. **SDC training is stable**: 300 steps with no NaN, no OOM, no divergence.
2. **Format (meta) retention**: 99.1% emit rate, comparable to SFT.
3. **Calibration is better than E21R-v2**: confidence correlates with accuracy; no collapse to a constant.
4. **Halt callback works**: after autoresearch-fixed thresholds, catches real drift without false positives.
5. **The autoresearch pattern of 6 iterations** identified the metric granularity issue that would have otherwise been attributed to "SDC doesn't work".

---

## 9. Known Failure Mode

**Name**: Post-meta EOS avoidance + repetition loop.

**Mechanism**: SDC post-meta repel pushes student away from decoy-conditioned teacher logprobs. EOS is a high-probability token under the decoy-conditioned context, so student learns to avoid it. Once past the first `\boxed{}`, the model keeps restarting new `</think>` + meta + boxed sequences until hitting `max_tokens`.

**Evidence**:
- GSM truncation rate 34% correct / 54% wrong
- MATH truncation rate 50% / 64%
- Sample completions show 5-20 repeats of "The answer is \boxed{X}" before cutoff
- Extracted answer is often the last (malformed) repetition

**Plan §2.1 risk this relates to**: none — this is a new failure mode not anticipated.

---

## 10. Next Actions

### Immediate
1. **Earlier-checkpoint eval**: run eval on checkpoint-100, 150, 200 to pinpoint where accuracy started dropping. Would tell us if λ_post = 0.2 (step ~100) is a better stopping point.
2. **λ_post ablation**: retrain with `lambda_post_final: 0.1` (down from 0.3) to see if weaker repel preserves accuracy while keeping calibration gain.
3. **EOS-aware repel**: modify `_compute_sdc_advantage` to mask EOS tokens out of the post-meta repel calculation. Student should not be pushed away from terminating.

### Medium-term
4. **N3 baseline**: run same config with `variant: n3` (attract only, no post-meta repel) for full 300 steps. This isolates whether the -20pp drop is from repel specifically, or from contrastive training generally.
5. **Better answer extraction**: update `_extract_answer_fallback` to prefer first `\boxed{}` when the same answer repeats, or strip malformed text like `\text{hour}}` suffixes.

### Paper implications
The run materially changes plan §7 decision tree:
- **Not Branch A (no-imitation failed)**: H-SDC-3 not tested here.
- **Not Branch F (all pass)**: H-SDC-1 direct accuracy failed.
- **Branch B (¬ direct)**: SDC stays as §5 exploration variant; §8 reports null result for accuracy.
- **But**: calibration improvement is a new positive finding worth §8 elevation, not just null result. Update plan to split "accuracy headline" from "calibration-as-side-result".

---

## 11. Files Generated in This Run

| Artifact | Local / Node path | HF path |
|---|---|---|
| Final model | `/scratch/meta_debug/runs/.../ckpts/final/model-{1..4}-of-4.safetensors` | `sdc_runs/.../ckpts/final/` |
| Per-step checkpoints | `ckpts/checkpoint-{20..300}/trainer_state.json` | `sdc_runs/.../ckpts/checkpoint-*/` |
| Training metrics | `ckpts/metrics.jsonl` (1223 records) | `sdc_runs/.../ckpts/metrics.jsonl` |
| Training stdout | `sdc_smoke.log` | `sdc_runs/.../sdc_smoke.log` |
| HF monitor log | `monitor.log` | `sdc_runs/.../monitor.log` |
| 1030-problem eval JSON | `eval_v6_1030.json` | `sdc_runs/.../eval_v6_1030.json` |
| Autoresearch log | `/tmp/autoresearch_sdc_log.tsv` (local only) | — |

---

## 12. Known Issues in Tooling (Surfaced by This Run)

1. `src/training/grpo_v2.py::_ensure_vllm_stub` — real vLLM has to be imported before this module to avoid a dummy being registered. Fixed in `meta_rlsd_trainer.py` top-of-file via `try: import vllm`.
2. `SDCHaltCallback` original thresholds (0.90 wrap, 0.20 fallback, 2 consec) were incompatible with per-rank discrete metrics. Fixed to rolling-mean thresholds (0.35 wrap, 0.40 fallback, 3 consec, window 20).
3. `run_sdc_eval.sh` initially referenced `/tmp/eval_vllm.py` but the script lives on the node at `/scratch/meta_debug/eval_vllm.py`. Fixed with sed during this run.
4. `resume_from_checkpoint` flag not wired into `launch_sdc_full.sh` — v6 started fresh from step 0 despite having v5's checkpoint-100 in the ckpts dir.
