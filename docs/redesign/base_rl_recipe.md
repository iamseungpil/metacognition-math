# Base RL Recipe — Reference-Matched Qwen3-8B-Base (B0 / B3)

This maps every knob of the reference **RLRT** and **SRPO** *base* runs
(`self-distillation-analysis-0330-clean-code`) onto our two RL arms:

- **B0** = clean GRPO base (reference `baseline_grpo` / RLRT direction, lr 1e-6).
- **B3** = self-distillation base with PMI-shift routed onto meta tokens
  (reference SRPO recipe, lr 5e-6), our meta variant.

Reference scripts studied:
`experiments/math/run_math_rlrt_qwen3_4b_base_1.0.sh`,
`experiments/math/run_math_srpo.sh`,
`verl/trainer/config/{rlrt,srpo,rlsd,baseline_grpo}.yaml`.

**THINK-ON PRESERVED.** We keep `<think>...</think>` and `<|meta|>` spans in the
SFT and at rollout. We do **not** set `enable_thinking=false`; think stays at the
Qwen3 chat-template default, exactly as the reference RLRT/SRPO base runs leave it.

## Knob table

| Knob (verl path) | Reference (RLRT/SRPO base) | Our B0 | Our B3 |
|---|---|---|---|
| `actor_rollout_ref.model.path` | `Qwen/Qwen3-8B-Base` (SRPO) / 4B (RLRT script) | `Qwen/Qwen3-8B-Base` | `Qwen/Qwen3-8B-Base` |
| `data.train_batch_size` | 256 | 256 | 256 |
| `actor_rollout_ref.rollout.n` | 8 | 8 | 8 |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | 128 | 128 | 128 |
| `actor_rollout_ref.actor.optim.lr` | 1e-6 (GRPO/RLRT), 5e-6 (SRPO) | **1e-6** | **5e-6** |
| `actor_rollout_ref.actor.optim.lr_warmup_steps` | 10 | 10 | 10 |
| KL penalty | OFF (GRPO, no critic; `adv_estimator=grpo`) | **OFF** | **OFF** |
| PPO clip (low) | 0.2 | 0.2 | 0.2 |
| PPO clip (high, Clip-Higher) | 0.28 | 0.28 | 0.28 |
| `algorithm.rollout_correction.rollout_is` | `token` | `token` | `token` |
| `algorithm.rollout_correction.rollout_is_threshold` | 2.0 | 2.0 | 2.0 |
| `algorithm.adv_estimator` | `grpo` | `grpo` | `grpo` |
| `norm_adv_by_std_in_grpo` | True (RLRT) / False (SRPO) | True | False |
| think / `enable_thinking` | template default (ON) | **ON** | **ON** |

## Self-distillation knobs (B3 = SRPO-matched, PMI-shift routed onto meta)

| Knob | Reference SRPO | Our B3 |
|---|---|---|
| `actor.policy_loss.loss_mode` | `srpo` | `srpo` |
| `self_distillation.distillation_topk` | 100 | 100 |
| `self_distillation.alpha` | 0.5 (Jensen-Shannon) | 0.5 |
| `self_distillation.dynamic_weight_beta` | 1.0 | 1.0 |
| `self_distillation.dont_reprompt_on_self_success` | True | True |
| `self_distillation.is_clip` | 2.0 | 2.0 |
| `self_distillation.include_environment_feedback` | False | False |
| `self_distillation.max_reprompt_len` | 22528 | 22528 |
| **routing** `dcpo_rmeta_source` | n/a (reference has no meta) | **`pmi_shift`** |

`dcpo_rmeta_source=pmi_shift` (src/training/dcpo_pmi_shift.py,
compute_pmi_shift_reward) routes the PMI-shift signal specifically onto the
`<|meta|>` token spans — this is our meta-specific addition on top of the
reference-matched SRPO base. B0 does not use any self_distillation branch.

## Length budget

Reference SRPO: `MAX_PROMPT_LENGTH=2048`, `MAX_RESPONSE_LENGTH=8192`,
`MAX_FEEDBACK_LENGTH=8192`, `max_model_len=18944`. Our SFT trains at
`max_length: 8192`; the RL `max_model_len` follows the SRPO budget so meta+think
rollouts terminate cleanly (EOS invariant `<|im_end|>`=151645, verified by
`scripts/verify_eos_invariant.py` before any RL launch).
