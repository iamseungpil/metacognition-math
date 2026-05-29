# CTSD node-launch index (PLAN.md v5)

The repo has 59 root AMLT job yamls + 80+ `configs/` training yamls (mostly legacy
R0–R21 / SFT experiments). To avoid hunting, these are the **only files the CTSD
Phase C / E force-inject arm uses**:

| Role | File |
|---|---|
| **Training config** (hydra) | `configs/verl_ctsd_inject_C_h200_4x4k.yaml` |
| **AMLT job yaml** (node submit) | `h200_ctsd_inject_C_smoke.yaml` (repo root) |
| Inject core (pure, unit-tested) | `src/training/meta_inject.py` |
| Inject core tests (9, passing) | `src/training/tests/test_meta_inject.py` |
| Reward mode | `verl_sdc.py` → `REWARD_CONFIGS["ROD_MQ_CONTRAST_INJECT"]` |
| Rollout hook (node-smoke-req) | `verl_sdc.py` → `SDCRayPPOTrainer._force_inject_rollout` |
| Shared base config | `configs/verl_sdc_e21r_shared.yaml` (via `defaults:`) |

## What it is
`ROD_MQ_CONTRAST_INJECT` = R18b (contrastive RL, `ROD_MQ_CONTRAST`, which FAILED at
70.9%) + the one new axis: **force-inject `<|meta|>` at the max-entropy pre-answer
position** so the contrastive reward has a meta region to shape. Clean one-axis
ablation (inject on/off) vs `configs/verl_rod_mq_contrast_R18b_h200_4x4k.yaml`.

## Launch order (do NOT skip)
1. **A.3 PASS** required first (force-inject shown causally helpful offline).
2. **Re-package code** to a GitHub release; set `CODE_TAR_REVISION` in the job yaml.
3. **Wire + 1-step smoke** `_force_inject_rollout` on the node (DataProto repack),
   then remove the `SDCRayPPOTrainer.__init__` fail-fast guard. Until then the job
   intentionally crashes at startup (sdc_force_inject=true) — by design.
4. Phase C smoke = `total_training_steps=50`, gate AIME +3pp → Phase E raises to 300.

## Legacy reference (NOT the CTSD arm)
`verl_rod_mq_contrast_R18b_*` (contrastive, no inject — the failed baseline),
`verl_rod_mq_R18a_*`, `verl_gfn_opsd_R18c_*`, `verl_stable_gfn_R21_*`, and the
root `h200_*` / `h100_*` job yamls are prior experiments — see git history.
