# Plan v5.6 — Position-Teacher RLSD (PT-RLSD)

**Date**: 2026-05-07 · **Author**: iamseungpil@gmail.com
**Predecessors**: `plan_meta_rod_v53_2026_05_07.md` (v5.5)
**Format**: Intent → Hypothesis → Verification (with falsification criterion).

This plan replaces v5.5's M5.6.5 v2 ROD ALT-2 design (which deviated from user intent into auxiliary multi-loss distillation) with a faithful **R5-framework-preserving** design that swaps decoy teacher for **position teacher**. The two methods are:

- **ROD-PT** = R5 (RLSD reward amplify) + position teacher (decoy 자리)
- **OPD-PT** = OPD M5.2 (top-K KL T+ distillation) + position teacher (decoy 자리)

Both retain R5's forced META injection, V0 prefix, and gold-conditioned content teacher.

---

## 0. Executive Summary

R5 step 300 (16k eval) measured: AIME 10.0%, MATH 53.6%, GSM 84.2%. Forced META preserved emit rate 87% on AIME but failed to lift accuracy — strong evidence for **Hypothesis A** (meta block disrupts long-CoT working memory at mid-stream emit positions).

PT-RLSD addresses Hypothesis A by adding a **position teacher** that checks whether the student's META_START emit position is reasonable under the no-forced-META natural distribution. Wrong-position emits get a negative reward; right-position emits proceed under R5's SDC factor (or OPD's KL distillation) for content learning.

**Forecast (mechanistic)**: AIME 18–28% (median 23%), preserving GSM/MATH at R5 levels.

---

## 1. Background — measured anchors

| Method | Step | Eval ctx | GSM | MATH | AIME | Avg comp tokens |
|---|---|---|---|---|---|---|
| base SFT v8 | — | 4k | 71.8 | 40.4 | 10.0 | — |
| R5 forced-meta | 200 | 4k | 85.0 | 52.4 | 10.0 | 6.2k |
| **R5 forced-meta** | **300** | **16k** | **84.2** | **53.6** | **10.0** | **11.3k** |
| `e21r_v2` (legacy) | 300 | 16k | 92.6 | 74.8 | 13.3 | — |
| `base_grpo` (no meta) | 300 | 16k | 93.4 | 63.0 | **36.7** | 12.8k |

Critical observations:
- R5 step 200→300 + 4k→16k both fail to move AIME (Hypothesis C and D falsified).
- Avg AIME completion length R5 (11.3k) ≈ base_grpo (12.8k) — length itself not the issue.
- AIME meta emit rate 87% (R5) vs 0% (base_grpo) → 26.7pp accuracy gap.
- Conclusion: **forced META at every position 0 disrupts AIME long-CoT**; emit position discipline is the missing ingredient.

---

## 2. Diagnosis — what fails in R5

R5 forces every rollout to start with `<|meta|>` at completion position 0. This creates a ~6k–10k token meta-then-body-then-meta pattern even on AIME problems where the natural answer is one continuous algebraic argument. The teacher (`v8_meta_inside_strict_sft`) was SFT-trained without forced injection and emits META at ~5–20% of natural positions on long chains (estimated from AIME data). Forcing 100% emit on student rollouts pushes student off the teacher's natural distribution at mid-stream positions, where the meta block bytes interrupt working-memory representations the model needs for the next 5k+ reasoning tokens.

**Position teacher** is the diagnostic intervention: at each META_START emit position p, run a no-force teacher forward and check whether META_START is in the teacher's top-K candidates at that position. Wrong positions are penalized at the policy-gradient level.

---

## 3. Related work survey

| Paper | Mechanism | Relation to PT-RLSD |
|---|---|---|
| RLSD (2604.03128) | scalar SDC factor `(P_T/P_S)^sign(A)`, gold-conditioned same-model self-distill | **ROD-PT base** — keep SDC factor on META content; replace decoy contrast with position |
| Why-Degrade (2603.24472) | epistemic suppression analysis; fixed teacher > EMA | Justifies fixed teacher in our setup |
| Rethinking OPD (2604.13016) | top-K mass concentration, off-policy cold start | **OPD-PT base** for content distillation |
| Revisiting OPD (2603.25562) §F3 | top-K teacher support matching, +19.8% | Used in our `_topk_kl` |
| CoMT (2601.21909) | two-stage cognitive separation; CCRL confidence rewards | Closest to position teacher idea but at process-step granularity |
| OPSD (2601.18734) | privileged-info teacher with verified traces | Two-teacher pattern formalized; we extend to (gold + position) |
| SD-Zero (2604.12002) | reviser-conditioned binary→KL | Localized signal; we localize to META_START emit positions |
| Lightning OPD (2604.13010) | offline cached teacher, 4× speedup | Future cost optimization |
| DistiLLM (2402.03898) | skew JSD | Alternative content signal (not used here) |
| GKD (2306.13649) | on-policy rollouts + reverse KL | Our content KL semantics |
| PRM (Lightman et al. 2023) | per-step process reward | Closest reward-shaping prior; we differ in (a) META_START token only, (b) binary top-K instead of trained PRM |

**Novelty positioning of PT-RLSD**:
1. **Position teacher with binary top-K** for emit decision (no prior method does this exactly — CoMT/CCRL use per-step confidence rewards but not top-K membership of a special token).
2. **Decoy → position teacher swap** within R5/OPD framework — reuses validated content distillation while replacing weakest contrast signal (decoy showed 0pp on AIME).
3. **Reward-shaping at META_START token only** — minimum-intrusion process reward.

---

## 4. Method matrix (final)

| Method | Forced META | Decoy T- | Position teacher | Content signal | Reward type |
|---|---|---|---|---|---|
| base SFT v8 | natural (87%) | — | — | — | (no RL) |
| **R5** (measured) | ✅ 100% | ✅ | ❌ | scalar SDC factor on meta region | RLSD reward amplify |
| **OPD M5.2** (running) | ✅ 100% | ✅ | ❌ | top-K KL(T+ ‖ S) − KL(T- ‖ S) | aux KL distillation |
| **ROD-PT** ★ (new) | ✅ 100% | ❌ removed | ✅ T_pos top-K | scalar SDC factor on meta region | RLSD reward amplify + position penalty |
| **OPD-PT** ★ (new) | ✅ 100% | ❌ removed | ✅ T_pos top-K | top-K KL(T+ ‖ S) | aux KL + position penalty |
| ROD-ALT2 (R9 running) | ❌ | ❌ | BCE soft (aux) | top-K KL on emitted meta | aux multi-loss (deviated from intent) |
| `base_grpo` (measured) | none | — | — | — | RLVR no meta |

The two new methods (ROD-PT, OPD-PT) isolate **decoy → position teacher swap**, holding everything else constant vs R5/OPD M5.2 respectively.

---

## 5. ROD-PT — detailed design

### 5.1 Intent
Preserve R5's RLSD reward-amplify framework on META content (paper-faithful self-distill on the meta region) and replace R5's decoy contrast with a **position teacher** that penalizes emit positions where META_START is not in the gold-conditioned no-force teacher's top-K.

### 5.2 Mechanism (per rollout)
```
Forward pass:
  T_content forward: prompt + V0 + gold + completion + forced META  (R5's T+)
  T_position forward: prompt + V0 + gold + completion              (no forced META)

Per-token loss assembly:
  At meta region tokens t ∈ student's actual META block:
    SDC_factor[t] = clip((P_T_content[t] / P_S[t])^sign(A))
    PPO_per_token[t] *= SDC_factor[t]                              (R5 reward amplify)

  At META_START emit position p (one per emitted block):
    top_K_at_p = T_position(prompt + V0 + gold + completion[:p]).next.topk(K).indices
    if META_START_ID not in top_K_at_p:
      advantage[p] += position_penalty                              (e.g., -1.0)

Total loss = standard PPO with modified per-token advantage and SDC factor.
```

### 5.3 Hypotheses

| ID | Statement | Falsification criterion |
|---|---|---|
| **H1** | ROD-PT AIME ≥ R5 + 5pp (= 15%) | < 13% → position teacher signal too weak |
| **H2** | ROD-PT AIME meta emit rate < R5 (87%) | rate ≥ R5 → position penalty not biting |
| **H3** | ROD-PT GSM/MATH within R5 ±3pp | both ↓ → position penalty over-suppresses on easy chains |
| **H4** | Position penalty rate negatively correlated with AIME accuracy across training (Pearson < −0.3) | r ≥ 0 → penalty signal not driving accuracy |
| **H5** | ROD-PT > ROD-ALT2 in AIME | ALT-2 ≥ → RLSD framework not necessary, aux losses sufficient |
| **H6** | Decoy ablation (ROD-PT vs R5): no meaningful AIME diff if decoy contributes 0pp on AIME (already measured) | ROD-PT ≈ R5 → position teacher contributes nothing either |

### 5.4 Hyperparameters

| Symbol | Value | Comment |
|---|---|---|
| K (position top-K) | 16 | strict; ablation 4/32 if needed |
| position_penalty | −1.0 | matches R5 advantage scale |
| forced META | ✅ 100% | R5 framework |
| decoy | ❌ | removed |
| Cold start | R5 step 300 | preserves R5 GSM/MATH gain |
| total_steps | 100 | R5 step 300 → 400 |
| batch | 64 | R5/OPD/M5.2 controlled |
| lr | 5e-7 | conservative fine-tune |
| max_response | 4096 | R5 framework |
| eval ctx | 16k | matches all comparators |

---

## 6. OPD-PT — detailed design

### 6.1 Intent
Preserve OPD M5.2's auxiliary KL distillation on META content (top-K KL with T+) and replace decoy KL contrast with **position teacher penalty** (same mechanism as ROD-PT).

### 6.2 Mechanism (per rollout)
```
Forward:
  T_content (= T+, gold + forced META): same as OPD M5.2 T+
  T_position (gold, no forced META): same as ROD-PT

Loss:
  total = PPO_standard
        + α · KL_topK(T_content || S) on meta region                (OPD M5.2 K=64 KL)
  
  advantage[META_START_emit_pos_p] += position_penalty
    if META_START_ID not in top_K(T_position[p])
```

### 6.3 Hypotheses

| ID | Statement | Falsification |
|---|---|---|
| **H7** | OPD-PT AIME ≥ OPD M5.2 + 3pp | ≤ → position penalty doesn't help on top of KL distillation |
| **H8** | OPD-PT meta emit rate < OPD M5.2 | rate ≥ → penalty not biting |
| **H9** | ROD-PT vs OPD-PT difference reveals RLSD vs aux-KL effect | similar → reward type axis irrelevant |

### 6.4 Hyperparameters

Same as ROD-PT except:
- Loss = OPD M5.2 KL T+ (α = 1.0, K = 64, T = 1.0)
- Cold start: M5.2 step 50 ckpt (when available) or R5 step 300

---

## 7. ROD-ALT2 (current R9) — ablation only

ROD-ALT2 (`meta_rod_v2_trainer.py`) deviates from intent: PPO standard + auxiliary BCE/KL/rate-penalty losses; no R5 SDC factor; on-demand emit instead of forced META; cold start = base meta SFT v8.

**Decision**: keep ROD-ALT2 running as-is (already on H100 STD). Its result becomes a useful ablation: "RLSD reward amplify is/is-not necessary." If ROD-ALT2 ≥ ROD-PT, RLSD is unnecessary; if ROD-PT > ROD-ALT2, RLSD framework matters.

---

## 8. Forecast scoreboard

| Method | Predicted AIME (80% CI) | Mechanism justification |
|---|---|---|
| base SFT v8 | 10.0 (measured) | — |
| R5 (measured) | 10.0 | forced META disrupts long CoT |
| OPD M5.2 (forecast) | 13–17 | KL signal density helps a little |
| **ROD-PT** ★ | **18–28**, center 23 | position teacher fixes Hypothesis A directly while preserving R5 framework |
| **OPD-PT** ★ | **18–26**, center 22 | similar to ROD-PT but aux KL instead of reward amplify |
| ROD-ALT2 (current R9) | 13–22 | aux losses partial Hypothesis A fix |
| base_grpo (measured) | 36.7 | RLVR ceiling without meta |

**Decision tree on results**:
- ROD-PT AIME ≥ 22 → publishable: "first metacognitive RL preserving long-CoT"
- 15–22 → partial; iterate position penalty form (continuous, K=4/32)
- < 15 → Hypothesis A fix insufficient; pivot to M5.7 ADOPD (base vs meta SFT differential)

---

## 9. Self-review log

### Round 1 (intent clarity)
- ROD-PT differentiates from R5 on exactly one axis (decoy → position) — clean ablation.
- OPD-PT differentiates from OPD M5.2 on exactly one axis — clean ablation.
- ROD-PT vs OPD-PT differentiates on reward type (RLSD vs aux KL) — clean orthogonal axis.

### Round 2 (hypothesis falsifiability)
- H1, H7 quantitative (AIME thresholds) ✓
- H4 measurable (Pearson correlation across training steps) ✓
- H6 explicitly addresses the "what if position teacher contributes nothing" risk ✓

### Round 3 (no duplicate methods)
- ROD-PT vs ROD-ALT2: different mechanism axis (RLSD vs aux loss). Both worth running.
- OPD-PT vs OPD M5.2: clean swap, sibling.
- M5.7 ADOPD (Plan v5.5) deferred until ROD-PT/OPD-PT results.

### Round 4 (cost vs gain)
- ROD-PT: +1 teacher forward (T_position) per step. Cost ~50% of T+ forward.
- OPD-PT: same overhead, but KL forward already exists in OPD M5.2.
- Both feasible on H100 80G×4 with batch 64.

### Round 5 (controllable confounds)
- All methods: batch 64, max_response 4096, lr 5e-7, total 100 steps from R5 step 300.
- Eval at 16k.
- Cold start unified (R5 step 300) for ROD-PT; OPD-PT can branch.

### Round 6 (residual risks)
- *Position teacher itself uses gold conditioning*: wrong positions are wrong "given the answer is known"; student's no-gold rollout might emit at positions teacher wouldn't (gold's hint shifts emit prior). Mitigation: log per-step "teacher emit rate at student's chosen position" and adjust K if rate < 0.05.
- *Decoy removal*: R5 measured 0pp AIME effect from decoy, so removal cost ≈ 0. But on GSM8K decoy might have contributed. Watch for GSM regression.
- *position_penalty=−1.0 too strong*: ablation with −0.5 or −2.0 if AIME stalls.

### Round 7 (no critical issues remaining) — frozen.

---

## 10. Implementation manifest

### Files to add
- `src/training/verl_sdc.py` — extend with new mode `RLSD_ROD_PT`:
  - fork `RLSD_FORCED_META` (R5)
  - new flags: `disable_decoy=True`, `enable_position_teacher=True`, `position_top_k=16`, `position_penalty=-1.0`
  - new function `_compute_position_penalty()` — T_pos forward, top-K check, advantage shift at META_START emit positions
- `configs/verl_rlsd_rod_pt_h100_4x4k.yaml` — config for ROD-PT
- `src/training/meta_opd_trainer.py` — add `disable_decoy` + `enable_position_teacher` flags (mirror ROD-PT logic)
- `configs/meta_opd_pt_h100_4x4k.yaml` — config for OPD-PT
- `h100_rod_pt_R10_0507.yaml` — AMLT yaml for ROD-PT
- `h100_opd_pt_R11_0507.yaml` — AMLT yaml for OPD-PT
- `scripts/smoke_pt_rlsd.py` — smoke for both methods

### Files to reuse
- R5 verl_sdc framework (T+ gold-conditioned, V0 prefix, forced META, SDC factor)
- OPD M5.2 KL infrastructure (`_topk_kl`, meta_mask)
- Cold start ckpt staging (R5 step 300 from HF)
- Eval pipeline (`eval_vllm_1030.py` 16k tp=4)

### Smoke checklist
1. Position teacher forward — output shape correct, no_grad context
2. Top-K membership check — META_START_ID in top_K(T_pos[p]) returns correct binary
3. Advantage shift — advantage[p] modified only at META_START emit positions, not elsewhere
4. SDC factor (ROD-PT) on META content — same as R5
5. KL T+ (OPD-PT) on meta region — same as OPD M5.2
6. Decoy disabled — no T- forward, no decoy logits in loss
7. End-to-end 1-step on toy LlamaForCausalLM — gradient finite, model params receive grad

---

## 11. Changelog

- **2026-05-07** v5.6 created. ROD-original was renamed ROD-PT (Position Teacher). OPD-PT added as sibling. Decoy → position teacher swap formalized as the unified novelty axis. ROD-ALT2 (current R9) preserved as ablation. Forecast: AIME 18–28 (ROD-PT), 18–26 (OPD-PT).
- **2026-05-07** v5.5 (predecessor): ROD ALT-2 (aux multi-loss) — deviated from intent.
- **2026-05-07** v5.4 (early): codex review identified injection collapse + tug-of-war.
- **2026-05-06** v5.3: M5.6.5 ROD on-demand emit primary.
- **2026-05-06** v5.2: M5.1 R5 step 300 + 16k eval result integrated.

---

*Frozen 2026-05-07 after 7 self-review rounds. No critical issues remaining.*
