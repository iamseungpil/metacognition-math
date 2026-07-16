# Base RL Recipe — OUR-STACK KNOBS ONLY (B0 / B2 / B3)

> **v2 (2026-07-11, 현행).** This is the live recipe for the **RQ3 MATCHED
> LADDER** on the real `Qwen/Qwen3-8B-Base`. The v1 recipe (instruct-copied
> hyperparams) collapsed all base arms at gs50–100 and is documented in the
> "DO NOT" section below and in `docs/redesign/EXPERIMENT_LOG.md`. A still-older
> version of this file mapped the reference `self-distillation-analysis` verl
> fork keys — none of those exist in our stack; only knobs our real configs and
> launchers actually set are documented here.

Our RL entrypoint is `python -m src.training.verl_sdc --config-name=<cfg>`.
Advantage composition (region-split) lives in `src/training/dcpo_region.py`;
SFT is `src/training/sft.py` (wrong_prefix segment-mask). The three live
launchers are the ground truth:

- **B0 / no-meta baseline** — `h100std_rq3_b0.yaml` → config
  `base_matched_grpo_h100_4x4k` (mode `VANILLA_GRPO`, init `models/b0_gold_sft`
  = no-meta gold SFT from `data/b0_gold_sft.parquet`, public-HF gold, 1290 rows).
- **B2 / meta-SFT + vanilla GRPO** — `h100std_rq3_b2.yaml` → config
  `base_matched_grpo_h100_4x4k` (SAME `VANILLA_GRPO` reward path as B0; differs
  ONLY in init = `models/b23_rv_unmasked_sft`, the meta RV unmasked SFT from
  `data/b23_rv_unmasked_sft.parquet`, 1763 rows).
- **B3 / region-split pmi_shift** — `h100std_rq3_b3.yaml` → config
  `triobj_dcpo_v4_stage3b_h100_4x4k` (same init as B2; region-split advantage,
  `dcpo_rmeta_source=pmi_shift`, every other meta head zeroed via `++`
  overrides — correctness→ANSWER span, pmi_shift→META_CONTENT span).

Science questions: **RQ1 = B2 − B0** (meta-SFT effect), **RQ2 = B3 − B2**
(replacing correctness-on-meta with pmi_shift). A *pure* RQ2 isolation needs the
planned **B2-R** arm (region-split with meta advantage = 0) — not launched yet.

Corresponding SFT launchers: `h100std_sft_b0_gold.yaml`
(`configs/sft_b0_gold.yaml`) and `h100std_sft_b23_unmasked.yaml`
(`configs/sft_b23_unmasked.yaml`). The b23 parquet ships with the `wrong_prefix`
field EMPTY → whole-response training; this unmask fix raised base meta emission
38% → 92%.

**THINK-ON PRESERVED.** `<think>...</think>` and `<|meta|>` (151669) /
`<|/meta|>` (151670) stay in vocab and at rollout; we never set
`enable_thinking=false`.

## v2 knob table (Hydra path → value; identical across arms unless noted)

Values resolve through the `verl_e4_selfdistill_h200_4x4k` base → per-arm config
→ launcher `++` overrides.

| Knob | OUR key (Hydra path) | Value (all arms) |
|---|---|---|
| model init | `actor_rollout_ref.model.path` | B0 `models/b0_gold_sft` / B2·B3 `models/b23_rv_unmasked_sft` |
| rollout temperature | `actor_rollout_ref.rollout.temperature` | **1.0** (v1 was 0.6 — collapse) |
| rollout top_k | `actor_rollout_ref.rollout.top_k` | **-1** (off; v1 was 20 — collapse) |
| rollout top_p | `actor_rollout_ref.rollout.top_p` | **1.0** |
| max response length | `data.max_response_length` | **8192** (v1 was 4096 — collapse) |
| max model len / batched tokens | `actor_rollout_ref.rollout.max_model_len` / `max_num_batched_tokens` | **10240** |
| advantage std-normalization | `algorithm.norm_adv_by_std` | **false** (Dr.GRPO; v1 GDPO std-norm — collapse) |
| logprob micro-batch | `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` (logprob) | 2 |
| learning rate | `actor_rollout_ref.actor.optim.lr` | 1e-6 |
| train batch | `data.train_batch_size` | 64 |
| ppo mini-batch | `actor_rollout_ref.actor.ppo_mini_batch_size` | 8 |
| rollout group size | `actor_rollout_ref.rollout.n` | 8 |
| total steps | `trainer.total_training_steps` | 300 |
| PPO clip low / high | `actor_rollout_ref.actor.clip_ratio_low/high` | 0.2 / 0.28 (Clip-Higher/DAPO) |
| KL loss coef | `actor_rollout_ref.actor.kl_loss_coef` | 0.0 (`use_kl_loss=true` kept for the frozen ref worker — see note) |
| save frequency | `trainer.save_freq` | **B0/B2 = 10, B3 = 5** (b3 콜드스타트 발판 단축) |
| test frequency | `trainer.test_freq` | 50 (all arms) |
| RL mode | `mode` / `algorithm.sdc_mode` | B0/B2 `VANILLA_GRPO` / B3 `TRIOBJ_DCPO_V4` (region-split) |

### Notes on the matched knobs

- **`use_kl_loss=true` with coef `0.0` is intentional in ALL arms.** It exists
  NOT to add a KL gradient but to keep verl's FROZEN ref worker alive: with
  `use_kl_loss=false`, `need_reference_policy` skips ref-worker init and the
  first PMI batch crashes `trainer._compute_ref_log_prob`. Keeping it `true/0.0`
  everywhere preserves the byte-identical-actor property. Do not "simplify".
- **`clip_ratio_low/high` live on the ACTOR**, not on `algorithm`.
- The GDPO code path with a single `[correctness]` head IS our GRPO (group-mean
  centering over one head) once `norm_adv_by_std=false`. B0/B2 use
  `gdpo_reward_keys: [correctness]`; B3 uses the stage3b five-way key list with
  non-pmi weights zeroed (avoids the five-way-sync crash).

## DO NOT — the three v1 collapse causes (2026-07-08)

v1 launched the base arms with instruct-generation hyperparams copied over; all
base arms entropy-collapsed at gs50–100 while instruct twins on the identical
recipe completed 300 steps (substrate causation confirmed). Never reintroduce:

1. **Low-temperature clipped sampling: `temperature 0.6` + `top_k 20`.** On the
   base substrate this drives entropy collapse. v2 uses temp 1.0, top_k -1,
   top_p 1.0.
2. **`max_response_length 4096` cap.** Truncation pressure compounds the
   collapse. v2 uses 8192 (model_len/batched-tokens 10240).
3. **GDPO advantage std-normalization.** Amplifies near-zero-variance groups
   into noise. v2 sets `norm_adv_by_std=false` (Dr.GRPO), matching the
   SimpleRL-Zoo / Dr.GRPO / DAPO literature for base-model RL.

Collapse gates for monitoring: `actor/entropy < 0.05` or
`response_length/clip_ratio > 0.5` ⇒ kill and diagnose, do not ride to gs300.

## Validity gates at gs25 (see docs/CONSTITUTION.md Part VI)

A meta-RL arm is only interpretable if, by ~gs25: `dcpo/meta_emit_rate ≥ 0.8`,
`dcpo/pmishift_attempted_rate ≥ 0.3`, `dcpo/pmishift_n_save > 0`, and
`acc_with ≫ acc_without`.

**B3 PASSED (confirmed):** emit 0.89 · attempted 0.40 · n_save 7 ·
acc_with 0.70 vs acc_without 0.28. (Watch item: B3 meta emission is eroding
during RL 0.89 → 0.54 — structural pressure from correctness reaching only the
answer span; acc_with ≫ acc_without still holds, behavior intact. Under
observation.)

## B3 PMI-shift routing (the ONLY arm with a meta reward)

> **⚠️ 0712 정정 (2026-07-12).** 아래의 "다른 head 전부 0" 스트립 설계는
> **실패로 판정돼 폐기됐다** — 형식 비계(w_format·w_emit·trunc_open 등)를
> 제거하자 RL 중 wellformed가 붕괴하고 pmi_shift가 불발(n_save→0), RQ2가
> +0.042→−0.120으로 반전했다(`EXPERIMENT_LOG.md` §9). 현행 B3 =
> **b3pkg 풀 패키지**: w_meta 0.8(rmeta_source=pmi_shift) + w_format 0.35 +
> w_emit 0.1 + w_cal 0.3 + len_cost 0.08 + trunc_open 0.3, **w_over만 0** —
> 검증된 선행(pre-rq3 pmishift 승리 런)과 동일 구성. 순수 pmi 격리는
> B3-noPMI arm(패키지에서 w_meta=0 하나만 제거, §10)이 담당하며 보류 중.
> 아래 원문은 실패 기록의 근거로 보존한다.

`h100std_rq3_b3.yaml` routes the PMI-shift signal onto `<|meta|>` spans and
turns every other auxiliary head OFF via `++` overrides on
`triobj_dcpo_v4_stage3b`:

| OUR key | Value | Role |
|---|---|---|
| `algorithm.dcpo_rmeta_source` | `pmi_shift` | R_meta = PMI-shift on META_CONTENT (`src/training/dcpo_pmi_shift.py`), two-position teacher-forcing on the frozen ref worker |
| `algorithm.dcpo_w_meta` | 0.8 | meta quality weight (warmup inherited from stage3b) |
| `algorithm.dcpo_w_cal` / `dcpo_w_format` / `dcpo_w_emit` / `dcpo_len_cost` / `dcpo_w_over` | 0.0 | all other heads OFF |
| `actor_rollout_ref.rollout.agent.agent_loop_config_path` | `null` | single_turn_agent, matches B0/B2 rollout structure |

Result for B3 = `correctness (answer span) + pmi_shift (meta span)` only.

## Ops facts

- Cluster msrresrchbasicvc H100×4 **Standard** (frequent preemption); image
  `mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch28x`; env
  `/scratch/conda_envs/simplerl` (conda-pack); amlt project `skilldiscovery2`.
- wandb `gistdslab/metacot-dcpo-v4`, runs `rq3-b0/b2/b3` with fixed
  `WANDB_RUN_ID` + `WANDB_RESUME=allow` (fixes v1 run fragmentation, 18 shards).
- In-training val = 594 problems (`verl_val_meta_mix`), reward is **+1/−1** so
  accuracy = (reward+1)/2. Final judge stays the 1030 held-out eval.
- Checkpoints relay via HF model repo `iamseungpil/metacot-h200-triobj-dcpo-v3`.

## Pre-rq3 history (kept for provenance)

The pre-rq3 (instruct-generation) arms B0=`h100std_base_matched_rl.yaml`
(init `v8_base_rv_sft`), B2=`h100std_gandhi.yaml` (init `v8_rv_functional_sft`),
B3=`h100std_shiftonly.yaml` used the v1 knobs (temp 0.6, top_k 20, resp 4096,
std-normalized GDPO, `use_kl_loss` handling as above). Those launchers and
numbers remain valid *for the instruct generation only* — do not mix them with
the rq3 ladder.
