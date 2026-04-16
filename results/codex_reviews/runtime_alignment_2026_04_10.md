# Runtime Alignment Review (2026-04-10)

## Scope

Reviewed alignment between:

1. `results/plan_metacot_v8_active_2026_04_09.md`
2. `NODE_POLICY.md`
3. `src/training/verl_gdpo.py`
4. `src/training/verl_reward.py`
5. `scripts/relaunch_verl_e21_0410.sh`
6. `scripts/relaunch_verl_e21r_0410.sh`
7. live AMLT runtime on `metacognition_eval` and `metacognition_train_b`

## Verified Findings

1. `TRAIN_B` is the clean mainline comparison lane.
   - base GRPO is actively training
   - checkpoints are being written
   - base-matched SFT artifact is already on HF

2. Historical `E21` relaunch is not currently reproducible.
   - `scripts/relaunch_verl_e21_0410.sh` is repo-tracked
   - the corresponding runtime on `EVAL` still fails with the vLLM/Triton `Python.h` compile error
   - this run must not be treated as claim-bearing evidence

3. The live `EVAL` job is an `E21`-family pilot, not `E21R`.
   - live process uses historical reward keys:
     `correctness, switch_v2, verify_v2, conf_traj, meta_floor`
   - it is therefore not evidence for the confidence-centered controller
   - because it is launched from ad hoc `/scratch/run_e21.sh`, it should be treated as runtime side evidence until reproduced from a repo-tracked launcher

4. Reward/config separation between `E21` and `E21R` is clean in repo code.
   - `src/training/verl_reward.py` exposes separate entrypoints:
     `compute_score` for `E21`
     `compute_score_confidence_centered` for `E21R`
   - `scripts/relaunch_verl_e21_0410.sh` and `scripts/relaunch_verl_e21r_0410.sh` also keep distinct reward key families
   - local smoke tests for reward/config wiring pass

## Required Interpretation

1. `E20a` remains the mainline representation anchor.
2. `TRAIN_B` remains the mainline comparison baseline.
3. live `verl_e21_gdpo_v2` on `EVAL` is `side_evidence`.
4. failed `verl_e21_historical_0410` relaunch is `invalid_for_claim`.
5. `E21R` remains prepared but not yet the active mainline runtime.

## Action Taken

Updated the active plan to:

1. remove the stale claim that current `E21R` on `EVAL` is mainline evidence
2. mark the live `E21` pilot as `side_evidence`
3. mark the failed historical relaunch as `invalid_for_claim`
4. keep `E21R` as a separated future controller experiment rather than silently merging it into current evidence
