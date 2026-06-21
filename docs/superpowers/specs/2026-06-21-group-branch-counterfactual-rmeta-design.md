# Group-Branch Counterfactual R_meta + SCoRe/AdaCoT Shaping — Design

**Date**: 2026-06-21  **Status**: design (pre-implementation)  **Branch**: ctsd-phase-c

## Problem (diagnosed, data-grounded)

confidence-rv Stage-2 DCPO RL (`triobj_dcpo_v4_rv`) makes meta **net-harmful**:
gs100 Δ(acc_with−acc_without) = −0.013 → gs190 Δ = −0.008 (90 more steps, ~flat,
still negative). emission rate **1.000**, ~94% generic `decision: verify`, saved 11 /
broke 16. Reading the actual meta text: the verify is a **generic "let me verify by
re-deriving the answer"** ritual, not a targeted check — on already-correct (easy)
problems the re-derivation is a pure downside dice-roll that sometimes overrides a
correct answer.

**Root cause (mechanism, confirmed in code):** the current meta reward
`R_meta = PMI` (`dcpo_rmeta_source: pmi`) is a **likelihood-delta**: how much the meta
CONTENT raises the per-token logprob of its own continuation vs a placebo
("Let me continue."), placebo-corrected, sign-gated by correctness. PMI rewards
**coherence-correlated-with-correctness, NOT causal usefulness**: a fluent generic
verify on an easy problem is coherent (high PMI) and lands correct (sign-gate passes)
→ rewarded, even though it changed nothing. **No term anywhere asks "would the answer
have been correct WITHOUT this meta?"** → no penalty for unnecessary meta → the reward
optimum IS generic-verify-everywhere. More training only sharpens it (Δ flat
gs100→gs190).

The principled signal we lack is the **counterfactual answer-delta**
`correct_with_meta − correct_without_meta` (= the north-star "meta→accuracy, only when
useful"). It is supported as `dcpo_rmeta_source: cf` but needs `sdc_counterfactual=true`
= a **second decode per rollout** without meta (~2× generation), currently OFF; PMI was
chosen as the cheap dense proxy that avoids the second decode.

## Prior-work basis (arXiv IDs verified)

- **GRPO-as-PRM / λ-GRPO** (2509.21154 / 2510.00194) on **RLOO** (2402.14740): within a
  GRPO group sharing a prefix, group-relative advantage on the post-branch span *equals*
  the MC counterfactual credit for the diverging step — the answer-delta for ~free.
- **SCoRe** (2409.12917, DeepMind): transition bonus `α·(correct_after − correct_before)`,
  **α>1 so right→wrong is the most-penalized event** — direct cure for breaks>saves.
- **AdaCoT** (2505.11896): `P_over` penalty for triggering reasoning on easy queries +
  **Selective Loss Masking (SLM)** on the decision token to stop always-trigger collapse.
- **AdaptThink** (2505.13417): constrain so emitting meta never drops the problem's
  pre-measured baseline accuracy (net-harmful forbidden by construction).
- **Spurious Rewards** (2506.10947): GRPO amplifies pretraining priors (even random
  rewards lift Qwen-Math) → PMI may be spurious; **cross-model causal check** as a gate.
- Cheaper-PMI alternative (not chosen now): **Implicit-PRM / PRIME** (2412.01981 /
  2502.01456) — swap meta-vs-placebo for policy-vs-reference log-ratio (Q of correctness),
  same 1-decode cost.

## Chosen approach

**Group-branch counterfactual R_meta + SCoRe/AdaCoT shaping**, computed from the
existing rollout group (no extra decode).

### Feasibility (confirmed in code)
- `actor_rollout_ref.rollout.n: 8` — GRPO group of 8 per prompt.
- without-meta arm mechanism already exists: eval bans the meta-open token via
  `logit_bias={meta_open_id: -100}` (`src/eval/eval_counterfactual_difficulty.py:93-95`),
  `dcpo_meta_open: 151669`.
- counterfactual bookkeeping infra (`c_with`/`c_without`, `dcpo/acc_without`,
  `cw_graded_rate`) partially exists, gated by `sdc_counterfactual` (currently false).
- Problem-specific wrinkle: emission rate = 1.0, so a group is all-with-meta and has
  NO natural without-meta sibling → the "free" counterfactual requires **forced
  branching**: split each group into a with-meta sub-arm and a without-meta sub-arm
  (meta-open banned) using the eval's logit-bias mechanism inside the TRAINING rollout.

### Design
1. **Group split (the backbone):** for each prompt's group of `n=8`, generate
   `n_with` with-meta (normal) + `n_without` without-meta (meta-open token logit-biased
   to −inf, same mechanism as eval arm B). Default split 4/4 (knob `dcpo_cf_branch_frac`).
   Both arms share the prompt prefix; both graded for correctness against gold.
2. **Counterfactual R_meta from group-relative advantage:** the group baseline now
   includes both arms, so a with-meta rollout's advantage on correctness is
   `correct_with − mean(group incl. without-meta)`. A verify that does no better than the
   without-meta siblings on the same prompt earns ~0 advantage → generic-verify-everywhere
   loses its gradient; Δ=0 cases reward `none` equally → **auto-abstention**. This
   REPLACES the PMI R_meta head (`dcpo_rmeta_source: cf_group`, new source value).
3. **SCoRe transition shaping (anti-break-correct):** add
   `R_trans = α·(correct_with − correct_without_groupmean)` with **α>1** (knob
   `dcpo_w_score_alpha`, default 1.5) so right→wrong is the most-penalized transition.
   `correct_without` comes free from the without-meta sub-arm group mean — no extra decode.
4. **AdaCoT over-trigger + decision-token SLM (anti-collapse-to-always-trigger):**
   `P_over` penalty when meta fires on a problem whose without-meta arm was already correct
   (knob `dcpo_w_over`, default 0.1); Selective Loss Masking on the `decision:` token so the
   emit/abstain decision can't collapse.
5. **AdaptThink accuracy-floor (safety):** clip so a group where with-meta acc < without-meta
   acc cannot push meta-emission up (guard, not a new head).
6. **Abstention knobs** (secondary): `dcpo_meta_floor` 0.05→0, `dcpo_w_emit` 0.15→0.1
   (the counterfactual now provides the emit-when-useful gradient; reduce blunt floors).

### Files to change (verified surface)
- `src/training/verl_sdc.py`: rollout generation — inject per-rollout meta-open ban for
  the without-meta sub-arm (arm tag in non_tensor_batch); REWARD_CONFIGS / new
  `dcpo_rmeta_source: cf_group` populator path computing R_meta from the group split.
- `src/training/verl_sdc_utils.py` + `src/training/dcpo_region.py`: route the
  counterfactual R_meta + SCoRe `R_trans` heads into `compose_dcpo_region_advantage`
  (new R_trans param + `w_score_alpha`, onto ANSWER region) — same pattern as R_corr/R_format.
- `configs/triobj_dcpo_v4_stage3b_h100_4x4k.yaml`: `dcpo_rmeta_source: pmi→cf_group`,
  `dcpo_cf_branch_frac: 0.5`, `dcpo_w_score_alpha: 1.5`, `dcpo_w_over: 0.1`,
  `dcpo_meta_floor: 0.05→0.0`, `dcpo_w_emit: 0.15→0.1`.

### Anti-inert / anti-regression TDD (the critical gate — this project's recurring trap)
1. **Byte-identical when disabled:** `dcpo_rmeta_source != cf_group` AND new knobs at
   defaults → advantage tensor byte-identical to current (no regression).
2. **Not inert when enabled:** with a synthetic group (known with/without correctness),
   the composed advantage CHANGES in the predicted direction and the new heads provably
   reach `compose_dcpo_region_advantage` (guards against the "key computed but never read"
   trap that made the naive `gdpo_reward_keys` add inert).
3. **Branching produces both arms:** the rollout split yields the configured fraction of
   without-meta rollouts (meta-open absent) and they are graded (`c_without` non-NaN).
4. **Abstention emerges:** on an all-easy synthetic group (without-meta already correct),
   meta-emission advantage ≤ 0 (no reward for unnecessary meta).

### Production-parity smoke (memory: isolation harness must match production knobs)
Before launch, a short isolated run with the EXACT production config knobs (all 7
relevant knobs setdefault-checked against the yaml) to confirm the cf_group path fires,
acc_without is graded, and no NaN/guard explosions — NOT a stripped harness.

### Validation
- D-gate: wellformed ≥ 0.40 (no collapse) + utility-conditioned Δ > 0 at gs-checkpoints.
- Offline validators (Math-Shepherd MC N≈8 / VinePPO K=9) optional: confirm the cheap
  group-split surrogate tracks true ΔP(correct).
- Cross-model causal check (Spurious-Rewards safeguard) before trusting Δ>0.

## Resolved decisions (user, 2026-06-21)
- **Split fraction = 4/4** (`dcpo_cf_branch_frac: 0.5`): tighter counterfactual baseline
  over more with-meta samples.
- **R_trans (and cf_group R_meta) route onto ANSWER region** (outcome locus) — the
  counterfactual answer-delta is an outcome signal; credit/blame the answer tokens.
- **Fully replace PMI** with `cf_group` (no combined dense PMI term). `dcpo_rmeta_source:
  pmi → cf_group`; the PMI populator path is bypassed for this run.
