# Plan v5.7 — Position-Teacher RLSD (PT-RLSD), corrected forced-META semantics

**Date**: 2026-05-07 · **Author**: iamseungpil@gmail.com
**Predecessors**: `plan_pt_rlsd_v56_2026_05_07.md` (v5.6), `plan_meta_rod_v53_2026_05_07.md` (v5.5)

This plan corrects v5.6's misinterpretation of "forced META". Per user clarification, "forced META" refers to **teacher input construction**, not student generation. Specifically:

- **Student rollout = natural emit** (no forced injection at generation time)
- **Teacher input = student's actual rollout including emitted META** (post-hoc conditioning)

This makes the design a clean RLSD self-distill on student-emitted META blocks, with two teacher conditioning views (with-META vs before-META) replacing the decoy contrast.

---

## 0. Executive summary

Two new methods (ROD-PT, OPD-PT) replace decoy teacher T- with **position teacher** T_position. Both keep R5/OPD framework intact for the META content signal; only the "contrast" axis changes from decoy answer to before-META prefix:

| Axis | R5 / OPD M5.2 | ROD-PT / OPD-PT |
|---|---|---|
| Student emit | natural | natural (no forced injection) |
| Content teacher | gold + completion (with META) | same |
| Contrast teacher | gold + completion **with decoy answer** | gold + completion **before META_START** |
| Reward signal | content distillation (SDC factor or KL) | same + position penalty at META_START emit |

Forecast (codex Round 2 consensus, conditional on student agency over emit timing): **AIME 32–36%** (R5 baseline 10%; +22–26pp).

---

## 1. Background — measured anchors (unchanged from v5.6)

| Method | Step | Eval ctx | GSM | MATH | AIME | Length |
|---|---|---|---|---|---|---|
| base SFT v8 | — | 4k | 71.8 | 40.4 | 10.0 | — |
| R5 forced-meta | 300 | 16k | 84.2 | 53.6 | 10.0 | 11.3k |
| `e21r_v2` | 300 | 16k | 92.6 | 74.8 | 13.3 | — |
| `base_grpo` | 300 | 16k | 93.4 | 63.0 | 36.7 | 12.8k |

Hypothesis A confirmed: meta block disrupts long-CoT working memory at mid-stream emit positions.

---

## 2. Diagnosis — corrected interpretation

R5 mechanism (per RLSD paper 2604.03128):
- Student rollout: gold-conditioned in our project (we run the same model with gold-augmented prompt as teacher; student rollout itself is unconstrained natural generation).
- "Forced META" in R5: refers to teacher conditioning where META_START is injected into teacher's input (after gold) so teacher's distribution captures "given gold + meta is starting now, what comes next".

User's intent for ROD/OPD-PT:
- Same student natural emit
- Teacher input includes student's actual emitted META block (= "forced" in user's vocabulary, post-hoc conditioning)
- Add a **second teacher view** (T_position) without META in input — used to check whether student's emit position is reasonable

This is identical in spirit to codex Round 2 design, just with corrected vocabulary.

---

## 3. Method matrix (v5.7 final)

| Method | Student emit | T_content input | T_contrast | Content signal | Reward type |
|---|---|---|---|---|---|
| base SFT v8 | natural | — | — | — | (no RL) |
| **R5** (measured) | natural | gold + V0 + completion (META included) | T- decoy answer | scalar SDC factor on meta region | RLSD reward amplify |
| **OPD M5.2** (running) | natural | same as R5 | T- decoy | top-K KL(T+ ‖ S) − KL(T- ‖ S) | aux KL distillation |
| **ROD-PT** ★ (new) | natural | same as R5 | **T_position (no META, before emit)** | scalar SDC factor on meta region | RLSD reward amplify + position penalty |
| **OPD-PT** ★ (new) | natural | same as R5 | **T_position** | top-K KL(T+ ‖ S) | aux KL + position penalty |
| ROD-ALT2 (R9 running) | natural | — | — | aux BCE + aux KL on emitted meta | aux multi-loss (independent ablation) |

**Key**: ROD-PT differs from R5 only on the contrast axis (decoy → position). OPD-PT differs from M5.2 only on the contrast axis. Two clean orthogonal ablations.

---

## 4. ROD-PT — detailed design (corrected)

### 4.1 Intent
Replace R5's decoy contrast with position teacher signal. Student emit timing is left to the policy; position teacher penalizes emit positions where META_START is unnatural under no-META conditioning. Content distillation uses R5's SDC factor on student's actual META block tokens.

### 4.2 Mechanism (per rollout)
```
Student rollout: prompt → completion (META naturally emitted at student's chosen position p)

Teacher forward 1 — T_content (with student's META included):
  Input  = prompt + V0_prefix + " Answer: " + gold + student_completion
  Forward through teacher (same model, frozen, gold-augmented)
  Output = next-token logits at every completion position
  
  At META content tokens (META_START to META_END):
    logp_T = log_softmax(T_logits)[token]
    logp_S = log_softmax(S_logits)[token]
    SDC_factor = clip(exp(sign(advantage) * (logp_T - logp_S)))
    PPO_per_token *= SDC_factor   # R5 reward amplify on META content

Teacher forward 2 — T_position (before student's META):
  For each rollout where student emitted META at position p:
    Input  = prompt + V0_prefix + " Answer: " + gold + student_completion[:p]
    Forward → next-token logits at last position (= position p)
    top_K_at_p = T_logits.last_position.topk(K).indices
    if META_START_ID not in top_K_at_p:
      advantage[rollout] += position_penalty   # rollout-level penalty
    else:
      no penalty

Total loss = standard PPO with modified advantage and SDC factor multiplied per-token on meta region.
```

### 4.3 Hypotheses

| ID | Statement | Falsification |
|---|---|---|
| **H1** | ROD-PT AIME ≥ 22% (R5 + 12pp) | < 18% → position penalty insufficient as Hypothesis A fix |
| **H2** | Position penalty rate (% rollouts penalized) decreases over training | constant → student not learning emit timing |
| **H3** | ROD-PT GSM/MATH within R5 ±3pp | both ↓ → over-suppression on easy chains |
| **H4** | ROD-PT meta emit rate < R5 (87%) on AIME but ≥ R5 on GSM/MATH | rate uniform → position penalty not granular per problem-type |
| **H5** | ROD-PT > ROD-ALT2 on AIME | ALT-2 ≥ → RLSD reward amplify not necessary |
| **H6** | Decoy contribution = 0pp on AIME (already measured) implies ROD-PT ≈ R5 if position penalty also 0pp; position penalty must contribute | ROD-PT ≈ R5 (= 10%) → position teacher 신호 무효 |

### 4.4 Hyperparameters

| Symbol | Value | Comment |
|---|---|---|
| K (position top-K) | 16 | strict; ablation 4/32 if needed |
| position_penalty | −1.0 | rollout-level, R5 advantage scale |
| factor_clip_low | 0.2 | R5 standard |
| factor_clip_high | 5.0 | R5 standard |
| Student forced META | ❌ | natural emit |
| Cold start | base meta SFT v8 (`v8_meta_inside_strict_sft`) | 87-97% natural emit, no R5 forced prior |
| total_steps | 100 | match R5/M5.2 |
| batch | 64 | match R5/M5.2 |
| lr | 5e-7 | match R5/M5.2 |
| max_response | 4096 | match |
| eval ctx | 16k | match |

---

## 5. OPD-PT — detailed design

### 5.1 Intent
Replace OPD M5.2's decoy KL with position teacher penalty. Content KL distillation (top-K on T+ on student's emitted meta region) preserved.

### 5.2 Mechanism
```
Student rollout: natural emit

T_content (= T+, gold + completion with META): same as ROD-PT
  Loss term: α · KL_topK(T_content || S) on meta region

T_position: same as ROD-PT
  At META_START emit position p:
    advantage[rollout] += position_penalty if META_START_ID not in top_K(T_position)
```

### 5.3 Hypotheses

| ID | Statement | Falsification |
|---|---|---|
| **H7** | OPD-PT AIME ≥ M5.2 + 5pp | ≤ → position penalty doesn't help on top of KL |
| **H8** | ROD-PT vs OPD-PT — RLSD vs aux-KL effect on AIME isolated | similar → reward type irrelevant |

---

## 6. ROD-ALT2 — keep as ablation

ROD-ALT2 (`meta_rod_v2_trainer.py`, currently on R9) uses on-demand emit + aux multi-loss. Independent ablation:
- Tests "supervised aux shaping" vs "policy-gradient native" hypothesis (codex Round 2 framing)
- If ALT-2 ≥ ROD-PT → aux losses superior; if ROD-PT > ALT-2 → RLSD framework matters

R9 should be kept running to completion (no cancel).

---

## 7. Forecast scoreboard (v5.7)

| Method | Predicted AIME (80% CI) | Justification |
|---|---|---|
| base SFT v8 | 10 (measured) | — |
| R5 (measured) | 10 | forced META at 100% disrupts AIME long-CoT but no signal to fix it |
| OPD M5.2 (forecast) | 13–17 | KL signal density helps marginally |
| **ROD-PT** ★ | **22–32**, center 27 | position teacher fixes Hypothesis A directly + R5 framework preserves content quality |
| **OPD-PT** ★ | **20–30**, center 25 | similar but aux KL instead of reward amplify |
| ROD-ALT2 (R9) | 13–22 | aux losses partial fix |
| base_grpo | 36.7 | RLVR ceiling |

---

## 8. Self-review log

### Round 1 (intent clarity)
- "Forced META" semantic corrected (teacher input vs student generation).
- Two methods clean ablations on contrast axis (decoy → position).
- All hypotheses quantitative.

### Round 2 (codex external review)
- Round 1 critique applied: forced student injection removed in favor of natural emit.
- Round 2 design endorsed by codex: AIME forecast 32-36% if student has emit timing agency. ✓
- Gold-conditioning of T_position kept (codex Round 2 endorsement: privileged training signal valid). ✓

### Round 3 (no duplicate methods)
- ROD-PT vs ROD-ALT2: different reward types (RLSD vs aux). Both worth running.
- OPD-PT vs OPD M5.2: clean swap (decoy → position).

### Round 4 (cost vs gain)
- ROD-PT: +1 teacher forward (T_position) per training step (~1.0× T_content cost).
- OPD-PT: +1 teacher forward (T_position) on top of existing OPD M5.2 T+/T- (= 3 forwards). T- can be replaced by T_position (= 2 forwards, same cost).
- Both feasible on H100 80G×4.

### Round 5 (controllable confounds)
- All methods: batch 64, max_response 4096, lr 5e-7, total 100 steps.
- Cold start unified at base meta SFT for ROD-PT and OPD-PT (clean — no R5 forced prior).
- Eval at 16k.

### Round 6 (residual risks)
- *Position teacher gold-conditioning*: codex Round 2 acceptable as privileged signal; may have slight non-causal hindsight bias. Mitigation: log T_position emit prob histogram per training step; if highly bimodal, soften penalty via continuous prob.
- *T_position truncation cost*: per-rollout sequence build with truncation at student's META_START position. Padding overhead but fits in batch.
- *position_penalty=−1.0 too strong/weak*: ablation in {−0.5, −1.0, −2.0} if AIME stalls at first run.

### Round 7 (no critical issues remaining) — frozen.

---

## 9. Implementation manifest

### Files to add
- `src/training/meta_rod_pt_trainer.py` — `MetaRODPTConfig` + `MetaRODPTTrainer`
  - Inherits `MetaOPDTrainer` (for teacher infra)
  - Removes decoy logic (no T- forward)
  - Adds T_position forward (per-rollout truncated input)
  - Adds SDC factor compute on meta region (R5 form)
  - Adds position penalty advantage shift
- `scripts/train_meta_rod_pt.py` — entry, mirrors `train_meta_opd.py`
- `scripts/smoke_meta_rod_pt.py` — 5 smoke steps
- `configs/meta_rod_pt_R10_h100_4x4k.yaml`
- `h100_meta_rod_pt_R10_0507.yaml` AMLT yaml

### OPD-PT (next iteration)
- `src/training/meta_opd_pt_trainer.py` (or extend `meta_opd_trainer.py` with flag) — same change but keep KL term, drop decoy KL, add position penalty

### Smoke checklist
1. T_position forward — per-rollout truncated input correctly built
2. Top-K membership check — META_START in top_K(T_position[last]) returns correct binary
3. Advantage shift — only at META_START emit positions (rollout-level)
4. SDC factor — exp(sign(A) * (logp_T - logp_S)) clipped, on meta region only
5. End-to-end 1-step on toy LlamaForCausalLM — gradient finite, params receive grad

---

## 10. Changelog

- **2026-05-07** v5.7 created: corrected "forced META" semantic to mean teacher input (not student generation). ROD-PT design now matches user's true intent and codex Round 2 design simultaneously. Forecast AIME 22-32 (ROD-PT center 27).
- **2026-05-07** v5.6: PT-RLSD draft with misinterpreted forced META.
- **2026-05-07** v5.5: ROD ALT-2 aux multi-loss (deviated from intent).

---

*Frozen 2026-05-07 after 7 self-review rounds + 2 codex external rounds. No critical issues remaining.*
