# Meta-CoT V6.4 Active Plan (2026-04-06)

This is the only active execution plan.

Historical documents remain useful, but they are not allowed to define runtime behavior:

1. `experiment_analysis_plan_2026_04_01.md`
   - long-lived RQ contract
2. `experiment_plan_v5.md`
   - v5 failure analysis
3. `plan_metacot_v6.3_final_2026_04_05.md`
   - pre-pivot plan before the E11 gate result
4. `plan_12gpu_experiments_2026_04_05.md`
   - branch table written before scenario-C became the active branch

## 1. Decision

### 1.1 Active Question

`Can metacognitive control improve OOD math solving by causing real strategy change, not just more meta text?`

### 1.2 Why This Branch Is Active

The latest completed pilot changed the active direction.

1. `E11` improved accuracy from `E9 62.1%` to `64.1%`.
2. But `approach_change` stayed near zero.
3. Multi-meta and information-checkpoint behavior increased sharply.
4. Therefore, the `E9 -> seed -> RL` line can create more meta activity, but it has not yet created structural rerouting.

The active bottleneck is now:

`how to create real solver rerouting`

not:

`how to make the model emit more meta blocks`

### 1.3 Current Branch Classification

1. `Scenario C` is the active mainline branch.
2. `SlotC` is still worth finishing because it is the strongest remaining test of whether stronger seeding can unexpectedly break the switch bottleneck.
3. `SlotB` is exploratory side evidence only because it runs `E13` on top of `control_v5_E9c/final`.
4. A clean-data restart from `base_sft` becomes the mainline continuation if `SlotC` fails the gate.

## 2. Long-Lived RQ Alignment

This active plan must still satisfy the project-level contract in `experiment_analysis_plan_2026_04_01.md`.

### 2.1 RQ1: Meta-CoT

`Intent`

Teach parseable meta control that is cleanly separated from ordinary derivation and can cause test-time adaptation.

`What counts as success now`

1. meta remains parseable and separated from ordinary CoT
2. meta triggers verify / redirect / diagnosis behavior
3. meta leads to real route change on hard problems

### 2.2 RQ2: Meta-RL

`Intent`

Learn calibration and intervention behavior causally, not by collapsing all rewards together.

`What counts as success now`

1. calibration-only and revision-only effects stay interpretable
2. isolated behavior rewards remain attributable
3. combined controller is only treated as mainline once the base checkpoint already shows non-trivial switching

### 2.3 RQ3: Curriculum

`Intent`

Use diagnosis and `study_need` to trigger retrieval or retry-time adaptation only when the model knows why the current route is insufficient.

`What counts as success now`

1. low confidence alone does not trigger retrieval
2. diagnosis and `study_need` become parseable enough to justify retrieval
3. retry gain is measured only after route-change behavior exists

## 3. Mainline Plan

### 3.1 Mainline RQ1

`의도`

Create structural switching behavior that can later support verify, redirect, and curriculum use.

`가설`

Stronger SFT seeding with clean redirect/verify exemplars can increase structural switch rate beyond the failed `E11` pilot.

`검증 방법`

1. `approach_change / structural_switch`
   - primary
   - target: materially above `E11 ~0.1%`
2. `accuracy`
   - must remain `>= 60%`
3. `verify_effectiveness`
   - check whether verify ever catches and changes wrong answers
4. qualitative hard-slice review
   - especially AIME and MATH failures

`현재 mainline 실험`

`SlotC`

1. base: `control_v5_E9/final`
2. method: stronger seed SFT (`164 seed x 5 epochs`)
3. role: last E9-line gate before a clean-data restart

### 3.2 Mainline RQ2

`의도`

Only run RL as main evidence when the base checkpoint already shows non-trivial switching behavior.

`가설`

If the base checkpoint already contains switch behavior, RL can selectively reinforce useful switching and calibration.

`검증 방법`

1. `switch_success_rate`
2. `accuracy`
3. `ECE / wrong-high-confidence`
4. `verify_effectiveness`

`Mainline gate`

RL is mainline only if the chosen base checkpoint satisfies both:

1. `approach_change >= 5%`
2. `accuracy >= 60%`

If the gate fails, RL on that base is side evidence only.

### 3.3 Mainline RQ3

`의도`

Curriculum / RAG is deferred until diagnosis is shown to trigger real rerouting, not just more text.

`가설`

Clean diagnosis plus `study_need` is only useful once route change exists.

`검증 방법`

1. diagnosis quality
2. route change after diagnosis
3. retry gain with retrieved examples

## 4. Sidecars And Evidence Policy

### 4.1 Mainline Evidence

1. `base_sft`
   - performance anchor
2. `E9`
   - best rigid meta baseline
3. `E11`
   - gate-setting pilot that selected scenario C
4. `SlotC`
   - remaining E9-line gate test
5. clean-data generation for the restart path
   - supporting mainline artifact, but not itself causal evidence

### 4.2 Exploratory Side Evidence

1. `SlotB`
   - currently runs `E13` on top of `control_v5_E9c/final`
   - may provide characterization, but must not be cited as mainline causal evidence
2. `E6 / E7`
   - probe feasibility and smoke are useful historical evidence
   - not active mainline
3. `E9v2 / E9bv2 / E10v2`
   - useful repair and ablation evidence
   - not mainline proof of rerouting

### 4.3 Evidence Ledger Rule

Every finished run must be labeled as one of:

1. `mainline`
2. `side_evidence`
3. `historical`
4. `invalid_for_claim`

No table, report, or launcher may silently mix these classes.

## 5. Implementation Contract

### 5.0 SlotC Gate Result (2026-04-06)

`SlotC` eval completed. **Case B confirmed.**

| Metric | Value | Gate | Pass? |
|---|---|---|---|
| accuracy | 67.3% | >= 60% | yes |
| approach_change | 0.3% | >= 5% | **NO** |
| ECE | 0.109 | — | improved |
| multi-meta | 49.9% | — | strong |

Key finding: E9-line SFT can improve calibration and multi-meta, but cannot create structural switching. Clean-data restart is now the mainline.

### 5.1 Current Node Roles (Case B Execution — 3 nodes)

1. `metacognition_eval`
   - mainline node
   - runtime: `E19` SFT (base_sft + V6 clean 10K, 3ep, lr=2e-6)
2. `metacognition_train_b`
   - mainline ablation node
   - runtime: `E19b` SFT (base_sft + V6 clean 10K, 5ep, lr=1e-6)
3. `metacognition_e8`
   - mainline ablation node
   - runtime: `E19c` SFT (base_sft + V6 clean 10K, 3ep, lr=5e-6)

### 5.2 Active Runtime Contract

| Node | Role | Launcher / Runtime | Base | Config | Evidence Class |
|---|---|---|---|---|---|
| `metacognition_eval` | mainline SFT | `sft.py --config sft_v6_clean_10k.yaml` | `qwen3_base_sft` | 3ep, lr=2e-6 | `mainline` |
| `metacognition_train_b` | epoch ablation | `sft.py --config sft_v6_clean_10k_5ep.yaml` | `qwen3_base_sft` | 5ep, lr=1e-6 | `mainline_ablation` |
| `metacognition_e8` | LR ablation | `sft.py --config sft_v6_clean_10k_highlr.yaml` | `qwen3_base_sft` | 3ep, lr=5e-6 | `mainline_ablation` |

### 5.2.1 Data Pipeline

1. `v6_10k_redirect_full.parquet` — gpt-5.4 full redirect (in progress, ~40%)
2. `v6_10k_verify_straight.parquet` — gpt-5.4-mini verify+straight (done, 3829 rows)
3. `v6_clean_10k_merged.parquet` — merge of 1+2 after quality audit
4. Auto-launch monitor running (PID tracked in `logs/monitor_redirect.log`)

### 5.3 Launcher Policy

Before launching any new RL job:

1. the base checkpoint must be written explicitly
2. the run must be labeled as either `mainline` or `exploratory`
3. the launcher must print:
   - `plan_id`
   - `run_label`
   - `mode`
   - `base_checkpoint`
   - `output_dir`
4. no launcher may silently hard-code a base that changes the hypothesis being tested

## 6. Decision Tree (Resolved)

### ~~Case A: SlotC succeeds~~ — NOT triggered

### Case B: SlotC fails — **ACTIVE**

SlotC `approach_change=0.3%` < 5% gate.

Execution:

1. ~~stop treating E9-based seed SFT as mainline~~ (done)
2. clean-data restart via V6 10K (gpt-5.4 full redirect + verify + straight)
3. 3 ablations: E19 (mainline), E19b (5ep), E19c (high-LR)
4. RL only after E19 SFT shows `approach_change >= 5%`

## 7. E19 Evaluation Gate

After E19 SFT completes, run `analyze_e11_pilot.py` 5-dimension analysis:

| Metric | Target | Action if fail |
|---|---|---|
| accuracy | >= 67.1% (base_sft) | investigate loss curves, check data quality |
| approach_change | >= 5% | data still insufficient — need more stitching examples |
| verify_effectiveness | > 0% | check verify scenario quality |
| ECE | < 0.15 | check confidence distribution |

If E19 passes the gate:
1. Promote best E19 variant to RL base
2. Launch E13 RL (V6.2 reward) with `max_steps=1000` (~2 epochs)
3. Use all 3 nodes for RL ablations (see 7.1)

### 7.1 SFT → RL Sequential Execution Plan

Informed by behavior-uncertainty findings (PPO 15 epochs / 5859 steps on 100K data).
Our data (~8K math) at effective batch 16 → **500 steps/epoch**.
Previous RL attempts used only 200 steps (0.4 epochs) — insufficient to learn behavioral rewards.

**Phase 1: SFT (current)**

| Node | Experiment | Config |
|---|---|---|
| EVAL | E19 mainline | 3ep, lr=2e-6 |
| TRAIN_B | E19b ablation | 5ep, lr=1e-6 |
| E8 | E19c ablation | 3ep, lr=5e-6 |

**Phase 2: Eval gate** — best E19 variant must pass `approach_change >= 5%` + `accuracy >= 67.1%`

**Phase 3: RL (after gate pass)**

| Node | Experiment | RL mode | Steps | Description |
|---|---|---|---|---|
| EVAL | E20 mainline | E13 (full V6.2) | 1000 | correctness + switch + conf_trajectory + verify |
| TRAIN_B | E20b ablation | E12 (switch-only) | 1000 | correctness + switch only |
| E8 | E20c ablation | E13 long | 1500 | full V6.2, 3 epochs for saturation check |

All RL runs use the **same best E19 SFT checkpoint** as base.

## 8. Stop Rule

Do not launch RL experiments until:

1. E19 SFT eval metrics confirm `approach_change >= 5%`
2. E19 accuracy >= base_sft (67.1%)
3. The launcher explicitly names the E19 checkpoint as base
4. RL `max_steps >= 700` (minimum ~1.5 epochs)
