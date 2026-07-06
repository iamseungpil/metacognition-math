# Matched-Base Implementation Readiness Verdict

**Date:** 2026-06-29
**Spec:** docs/superpowers/specs/2026-06-29-matched-base-clean-meta-comparison-design.md
**Goal:** Build the meta-removed twin of the meta pipeline so base vs meta differ ONLY in the meta mechanism.
**Constraint honored:** All work CPU-only. No amlt/GPU job launched. This report gates the human's GPU spend.

## VERDICT: GO — implementation is correct and confound-free, BUT launch is BLOCKED on one missing weight (base SFT-1) that must be re-SFT'd first. The data/config/strip side is fully built and verified; the launch plan below is unblocked once SFT-1 runs.

---

## (1) Artifacts built (absolute paths)

| Artifact | Path | Status |
|---|---|---|
| Strip builder | /home/v-seungplee/metacognition-math/scripts/build_base_rv_sft_data.py | built |
| Base SFT-2 data (meta-stripped rv) | /home/v-seungplee/metacognition-math/data/v8_base_rv_sft.parquet | built (1763 rows, 1.8MB) |
| Base RL config | /home/v-seungplee/metacognition-math/configs/base_matched_grpo_h100_4x4k.yaml | built |
| Base RL launch yaml | /home/v-seungplee/metacognition-math/h100std_base_matched_rl.yaml | built |
| Base SFT-2 config | /home/v-seungplee/metacognition-math/configs/sft_base_rv.yaml | built |
| Base SFT-2 launch yaml | /home/v-seungplee/metacognition-math/h100std_base_rv_sft.yaml | built |
| Base SFT-1 config (pre-existing) | /home/v-seungplee/metacognition-math/configs/sft_v8_base_matched_strict.yaml | exists (Qwen3-8B init) |

---

## (2) Verification verdicts

### V1 — strip answer-preservation + coherence: PASS
Re-confirmed independently in this session over the full file (not a sample):
- **Meta-token leakage = 0** across every object column (independent regex scan): 0/1763 rows contain `<|meta|>` or `<|/meta|>`.
- **Final-answer preservation = 100%**: 1763/1763 rows retain `\boxed{}` in the assistant turn; V1's full-set check confirmed the original last `\boxed{}` survives in the stripped recovery in every row.
- **Coherence = 5/5** standalone solutions read as complete, self-contained derivations reaching the correct boxed answer; no dangling "as verified above" references (the meta block sat between attempt and recovery, so removal leaves recovery logic intact).
- **Differ-only-in-meta on the trained region**: stripped recovery == (original recovery minus meta blocks), byte-exact, 1763/1763.
- Schema preserved as rv-native 6 cols (messages, scenario, confidence_label, wrong_prefix, prefix_split_char, split_tags); scenarios unchanged (verify 1209 / redirect 554).
- Non-blocking caveat: some `wrong_prefix` values are degenerate student attempts (repetition); they sit in the loss-masked region (sft.py masks wrong_prefix) and are identical to what the meta rv data carries, so they do not affect trained loss or the invariant.

### V2 — no-new-confound gate (RL config): PASS (CLEAN)
Independently re-derived by composing the full Hydra defaults chain and diffing every non-meta hyperparam. All non-meta hyperparams are **byte-identical** between meta (triobj_dcpo_v4_stage3b + pmishift overrides) and base: train_files, max_prompt_length=2048, max_response_length=4096, train_batch_size=64, lr=1e-6 AdamW, ppo_mini_batch_size=8, use_kl_loss=true/kl_loss_coef=0.0, clip 0.2/0.28, rollout.n=8, temp 0.6/top_k 20/top_p 0.95, adv_estimator=gdpo, use_kl_in_reward=false, total_training_steps=300, save_freq=10.

The ONLY differences are the meta mechanism itself or the init lineage:
1. gdpo_reward_keys: 5 heads → **[correctness]** (the 4 extra heads ARE the meta reward — removed, not zero-weighted)
2. gdpo_reward_weights: [1.0,0.5,0.3,0.0,0.1] → **[1.0]**
3. mode/sdc_mode: TRIOBJ_DCPO_V4 → **VANILLA_GRPO**
4. agent_loop_config_path: cf_agents_combined.yaml → **null** (the cf_group with/without-meta rollout split is the removed mechanism)
5. init: v8_rv_functional_sft → **v8_base_rv_sft**
6. env: meta sets DCPO_META_CLOSE_FORCE/N (pmi_shift teacher-forcing) — base omits.

Source-traced that VANILLA_GRPO is genuinely correctness-only: it is in `_VANILLA_MODES`, disjoint from `_REGION_ROUTED_MODES`; `_attach_teacher_signals` early-returns; the advantage path uses the **same GDPO group-mean-centering code** as meta's correctness head with no region populate / len_cost / meta-mask read; REWARD_CONFIGS maps it to the byte-identical `correctness_reward`/math_verify function meta uses. Zero residual `dcpo_*`/`cf_group`/`len_cost`/`inject` knobs in the base config.

**No mismatches found.** The optimization is identical; only the meta reward and meta rollout split are absent.

---

## (3) Blockers

| # | Blocker | Severity | Resolution |
|---|---|---|---|
| B-1 | **Base SFT-1 weights `v8_base_matched_strict_sft` do NOT exist on HF** (confirmed live: absent from iamseungpil/metacot model+dataset and metacot-rv). | HARD — gates everything | Re-SFT from pretrained Qwen3-8B on `v8_base_matched_strict.parquet` using existing config `configs/sft_v8_base_matched_strict.yaml`, push to `iamseungpil/metacot models/v8_base_matched_strict_sft/`. Data EXISTS on HF (confirmed: `v8_base_matched_strict.parquet` present in iamseungpil/metacot dataset). |
| B-2 | **Base SFT-2 weights `v8_base_rv_sft` do NOT exist on HF.** | Expected — this is the planned SFT-2 output | Produced by the base SFT-2 step. Its INPUT data parquet is built locally and must be uploaded to `iamseungpil/metacot data/v8_base_rv_sft.parquet` (the base SFT-2 yaml stages it from there). |
| B-3 | **Code tar revision is stale** (CODE_TAR_REVISION pins a release predating the new configs). | MEDIUM — both new launch yamls | Cut a new GitHub release containing configs/sft_base_rv.yaml + configs/base_matched_grpo_h100_4x4k.yaml and bump CODE_TAR_REVISION in both yamls before launch. |

Init-lineage parity note (not a blocker, an invariant to honor): base SFT-1 must use the same pretrained Qwen3-8B and same SFT hyperparams as the meta twin. The existing SFT-1 config already pins Qwen/Qwen3-8B; both SFT-1 corpora are 4264 rows with identical batch/epoch, so base SFT-1 lands at checkpoint-254 (same as meta) — the launch yaml stages that checkpoint; verify after SFT-1 completes.

GRPO gold answers: AVAILABLE — verl_train_meta_mix.parquet has ground_truth populated 5344/5344.

---

## (4) EXACT ordered launch plan (human runs on GPU)

Prereq (one-time): cut a GitHub release with the new configs and bump `CODE_TAR_REVISION` in `h100std_base_rv_sft.yaml` and `h100std_base_matched_rl.yaml`.

**Step 0 — base SFT-1 (REQUIRED, B-1). Authoring an amlt yaml for this is NOT yet done; reuse the existing meta SFT-1 launch yaml with the base config swapped in, or run the existing pipeline that consumes `configs/sft_v8_base_matched_strict.yaml`.** Output: push `v8_base_matched_strict_sft` (expect checkpoint-254) to `iamseungpil/metacot models/v8_base_matched_strict_sft/`. Gate: confirm the pushed checkpoint number before Step 1.

**Step 0.5 — upload base SFT-2 data**: push `data/v8_base_rv_sft.parquet` to `iamseungpil/metacot data/v8_base_rv_sft.parquet`.

**Step 1 — base SFT-2**:
```
amlt run h100std_base_rv_sft.yaml -t <target>
```
Init=v8_base_matched_strict_sft, data=v8_base_rv_sft.parquet → pushes `v8_base_rv_sft`. (Hyperparams identical to meta rv SFT: lr 1e-5, 3 epochs, bs1×ga4, max_length 4096, teacher_kl OFF.)

**Step 2 — base RL (correctness-only GRPO)**:
```
amlt run h100std_base_matched_rl.yaml -t <target>
```
Init=v8_base_rv_sft, data=verl_train_meta_mix.parquet, VANILLA_GRPO, 300 steps → `base_rv_grpo`. Reuses the same verl_sdc harness as meta with the meta reward OFF.

**Step 3 — eval (existing pipeline)**: run the 1030-problem eval (gsm8k/math500/aime) at 4k AND 16k with math_verify, comparing `base_rv_grpo@300` vs `pmishift gs300` (+ the other 3 meta runs). Use the existing eval harness (src/eval/eval_hf.py / the project's standard 1030 eval yaml); no new eval yaml was authored in this task.

YAMLs to use: Step 1 = `h100std_base_rv_sft.yaml`; Step 2 = `h100std_base_matched_rl.yaml`; Step 0 = existing meta SFT-1 launch yaml with `configs/sft_v8_base_matched_strict.yaml`; Step 3 = existing eval yaml.

---

## (5) Go / No-Go

**GO on correctness and confound-freedom.** The strip is verified meta-clean and answer-preserving (V1 PASS, 100% on the full set), and the RL config differs from meta in ONLY the meta mechanism + init (V2 PASS, byte-identical non-meta hyperparams, source-traced VANILLA_GRPO to a pure correctness path). The "differ only in meta" invariant holds at both the SFT trained-region level and the RL hyperparam level.

**Must-fix before pressing launch (ordered):**
1. Re-SFT base SFT-1 `v8_base_matched_strict_sft` (B-1) — the only HARD blocker; data is on HF, config exists.
2. Upload `data/v8_base_rv_sft.parquet` to HF (B-2 input).
3. Cut release + bump CODE_TAR_REVISION in both launch yamls (B-3).
4. After SFT-1, verify its checkpoint number matches the path staged in the SFT-2 yaml.

No GPU should be spent on Step 1/2 until Step 0 (SFT-1) exists, since the entire init lineage hangs off it. Once SFT-1 is pushed and B-2/B-3 are resolved, the pipeline is launch-ready with no remaining confounds.
