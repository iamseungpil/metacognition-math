# Pipeline Audit

Date: 2026-04-11

## Scope

Audited:

1. `results/plan_metacot_v8_active_2026_04_09.md`
2. `NODE_POLICY.md`
3. strict paired SFT configs and launchers
4. reward tests
5. strict paired data validator
6. analysis and curriculum entrypoints

## Verified Findings

1. The current strict paired SFT lanes are now aligned with the intended initializer contract.
   - Both strict configs use raw `Qwen/Qwen3-8B`.
   - Both nodes are training on 4 GPUs.

2. The repo still contains many historical launchers and configs that are runnable but not claim-bearing.
   - This is the main remaining operational risk.
   - The risk is not only stale code; it is stale code that still looks launchable.

3. The data contract is much stronger than before.
   - strict paired data passes validation
   - prompt and boxed answer parity are preserved
   - scenario counts are explicit

4. The analysis path is partially ready.
   - `eval_hf.py` already saves machine-readable outputs and full completions
   - behavior analysis docs in `analysis/behavior_uncertainty_lab` were stale and needed reframing

5. Curriculum code is present but should remain downstream.
   - current repository can smoke-test curriculum helpers
   - it should not yet be used as proof of RQ1/RQ2 success

## Corrections Applied

1. Rewrote the active plan around a strict `의도 -> 가설 -> 검증 방법 -> 해석` structure.
2. Added a strict alignment checklist.
3. Added a pipeline stage contract and artifact policy.
4. Added a verification script for mainline alignment.
5. Added a duplicate-launch guard to the remote SFT launcher.
6. Cleaned the accidental duplicate strict meta SFT process on `metacognition_eval`.

## Remaining Risks

1. RL mainline pair is still not frozen in a single canonical launcher/config pair.
2. Historical exploratory launchers remain in the tree and can still confuse future runs.
3. HF sync policy still needs to be followed after strict SFT checkpoints complete.

## Recommended Immediate Order

1. Let strict paired SFT finish.
2. Run the alignment verifier again.
3. Run paired eval and save the analysis bundle.
4. Freeze the paired RL launcher/config pair before any new RL run.
