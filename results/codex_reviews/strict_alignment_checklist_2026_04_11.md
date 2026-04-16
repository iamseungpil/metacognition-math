# Strict Alignment Checklist

Date: 2026-04-11

This checklist is the minimum precondition for trusting any `mainline` claim.

## A. Initializer Integrity

- [x] Strict meta SFT starts from raw `Qwen/Qwen3-8B`
- [x] Strict base SFT starts from raw `Qwen/Qwen3-8B`
- [x] Mainline strict SFT does not start from `checkpoints/qwen3_base_sft`
- [ ] Claim-bearing paired RL launcher is frozen and preflight-checked

## B. Paired Data Integrity

- [x] `v8_meta_inside_strict.parquet` exists
- [x] `v8_base_matched_strict.parquet` exists
- [x] Row counts match: `4264 == 4264`
- [x] Scenario counts recorded: `redirect=2199`, `verify=2065`
- [x] Prompt parity passes row-by-row
- [x] Boxed answer parity passes row-by-row
- [x] Strict validator passes with zero failures

## C. Runtime Integrity

- [x] `metacognition_eval` uses 4 GPUs for strict meta SFT
- [x] `metacognition_train_b` uses 4 GPUs for strict base SFT
- [x] Duplicate strict meta SFT launch on `eval` was detected and cleaned on 2026-04-11
- [x] W&B runs exist for active strict SFT lanes
- [x] Strict SFT checkpoints fully written on the remote nodes
- [ ] Strict SFT checkpoints synced locally

## D. Reward / Code Integrity

- [x] reward unit tests pass: `pytest -q tests/test_rewards.py`
- [x] `launch_sft_remote.sh` now uses `python -m accelerate.commands.launch`
- [x] duplicate-launch guard added for SFT remote launcher
- [ ] claim-bearing paired RL config pair finalized under frozen shared keys

## E. Analysis Integrity

- [x] eval pipeline saves JSON + metadata + parquet
- [x] full completion text is stored for qualitative analysis
- [x] behavior analysis docs updated from historical V5 framing to V8 strict framing
- [x] post-SFT bundle pins deterministic eval (`do_sample=False`, fixed seed)
- [ ] entropy / confidence / AIME qualitative bundle generated from strict SFT outputs
- [ ] strict SFT checkpoints synced locally before bundle execution

## F. Curriculum Integrity

- [x] curriculum code exists and imports
- [x] curriculum is treated as downstream, not current claim evidence
- [ ] curriculum trigger evaluation waits for diagnosis-ready checkpoints

## G. Claim Interpretation Rules

- [x] Active plan rewritten around `의도 -> 가설 -> 검증 방법 -> 해석`
- [x] Node policy freezes paired shared RL keys
- [x] `mainline`, `side_evidence`, `historical`, `invalid_for_claim` evidence classes defined
- [x] current strict SFT lanes are the active `mainline`
- [ ] all active launchers classified explicitly by evidence class

## Current Verdict

1. Strict paired SFT is claim-eligible in principle and has completed under the correct initializer contract.
2. The repository is not yet ready to make claim-bearing paired RL comparisons.
3. Historical RL launchers and stale analysis docs still exist, so the worktree needed a stronger contract layer.
4. The immediate safe next step is: sync checkpoints, run paired eval, then freeze paired RL launchers under the shared-key contract.
