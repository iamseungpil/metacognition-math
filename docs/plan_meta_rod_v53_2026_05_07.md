# Plan v5.3 — Meta-CoT × On-Demand RLSD (ROD-RLSD)

**Date**: 2026-05-07 · **Author**: iamseungpil@gmail.com · **Predecessors**: `plan_meta_opd_2026_05_03.md` (v5.2)

**Format**: each method block carries **Intent → Hypothesis → Verification (with falsification criterion)**.

This plan is the response to two findings on 2026-05-07:
1. R5 step 300 + 16k eval gave AIME **10.0%** (=base SFT, =step 200) — both 4k truncation (Hyp C) and HP under-train (Hyp D) **falsified**.
2. MATH-500 saw **+13.2pp** while AIME saw **0pp**. The gap correlates with chain length (943 / 2999 / 11322 tokens) and step count (5–10 / 10–20 / 30–50+).

The remaining hypotheses (A: meta interrupts long CoT; B: forced-meta = SFT prior anchor) cannot be addressed by tuning R5. They require redesigning **where and how** meta is emitted.

---

## 0. Problem statement (one paragraph)

Forcing a `<|meta|>` block at the start of every student rollout (R5) does keep the meta region alive (87% emission on AIME, 97% on GSM8K), but the **placement itself is wrong** for hard problems: AIME requires 30+ uninterrupted reasoning steps and a meta block in mid-stream disrupts the implicit working memory the transformer is building. The empirical signature is monotone — `base_grpo` (0% meta) AIME 36.7% → `e21r_v2` (20% meta) 13.3% → R5 (87% meta) 10.0%. We need a method that emits meta **only where T_meta itself would naturally emit it under gold conditioning**, while keeping calibration content (the metadata that distinguishes meta from regular tokens) under teacher control.

---

## 1. Survey — what is each prior method actually doing?

| Paper | Mechanism | Forced emit? | Decoy? | Region-specific teacher? | Token-level adaptive? |
|---|---|---|---|---|---|
| **paper RLSD** (2604.03128) | scalar SDC factor with gold teacher | no (relies on natural emit) | no | no | no |
| **fresh-0428** (our v1) | paper RLSD with meta in scope | no | no | no | no |
| **R5 forced-meta** (our v4) | forced first-token + scalar SDC + decoy | **yes (100%)** | yes (T-) | no | no |
| **M5.2 OPD-Decoy** (in progress) | full-logit KL on top-K + decoy + forced | yes | yes | no (meta-only mask) | no |
| **CoMT** (2601.21909) | two-stage: meta-thought then execution | structural | no | yes (region-aware) | no |
| **OPSD** (2601.18734) | privileged-info teacher + verified traces | no | no | yes (privileged region) | no |
| **SD-Zero** (2604.12002) | reviser-conditioned binary→KL | no | no | no | mistake-token concentration |
| **Rethinking-OPD** (2604.13016) | top-K mass concentration cold start | no | no | no | no |
| **Revisiting-OPD** (2603.25562) | top-K teacher support matching | no | no | no | yes (per-position) |
| **Why-degrade** (2603.24472) | epistemic markers analysis (diagnostic) | — | — | — | — |
| **CCRL** (2601.21909) | confidence-aware rewards on intermediate | no | no | no | yes (per-step) |

**Gap identified**: no prior method **lets the teacher decide whether to emit meta at each token position**, while also controlling **the meta content distribution** when emit happens. This is the M5.6.5 ROD-RLSD novelty.

Two related lines:
- **CoMT**'s region separation is closest in spirit but pre-fixes the meta region structurally (meta-thought in the prompt scaffold). M5.6.5 is *adaptive at the rollout-token level*.
- **Why-degrade**'s 20× epistemic-marker reduction tells us *forced* emission is unnatural; **R5's 87%** is overshooting the 5–20% natural range.

---

## 2. New diagnostic — why MATH-500 +13.2pp but AIME 0pp?

Three additive mechanisms that all hit AIME harder:

### 2.1 Length-induced disruption
Same single meta block emit in a 943-token chain lands at the **end** (post-answer-extraction self-check), but in an 11,322-token AIME chain the same block lands at **token 5–10k = mid-stream**. Mid-stream meta block disrupts attention pattern over the rest of the unfinished proof.

### 2.2 Confidence-correctness decoupling on AIME
- MATH conf 0.36 paired with acc 53.6% — model knows it's uncertain *and is right*; calibration-helpful.
- AIME conf 0.30 paired with acc 10.0% — model knows it's uncertain but *cannot find the answer either way*; calibration informs nothing for accuracy.
- Meta block carries calibration signal but no *redirect* signal.

### 2.3 Strategy-switch absent in current meta
AIME problems often need "let's try a combinatorial argument" mid-chain. Current meta carries `confidence: 0.7, action: revise`. Action is metadata. **Reasoning-redirect content is absent**, so when the model needs to switch strategies, meta doesn't help.

These three compound: long chain ⊃ mid-stream emit ⊃ disruption without redirect benefit.

---

## 3. Design space — five orthogonal axes

| Axis | Options | What we're varying |
|---|---|---|
| **A1 Emit decision** | forced 100% / forced rate / **on-demand from T_emit** / supervised location | when to emit |
| **A2 Distill signal** | scalar Δt / **full-logit top-K KL** / DistiLLM JSD / SD-Zero binary | how strong the signal is per token |
| **A3 Teacher count** | single (gold) / two (gold + decoy) / **two (gold + base SFT differential)** / region-specific (meta-T + body-T) | teacher diversity |
| **A4 Region masking** | full / **meta-only** / body-only / per-token weighted | where to apply distill |
| **A5 Cold start** | base SFT / R5 step 200 / **R5 step 300** / GRPO-warmed | starting policy |

R5 = (forced 100%, scalar Δt, two-decoy, meta-only, base SFT)
M5.2 = (forced 100%, full-logit, two-decoy, meta-only, R5 step 200)
**M5.6.5 ★** = (**on-demand T_emit**, content KL, **two-query single teacher**, per-token, R5 step 300)
M5.7 = (on-demand, content KL, **two real teachers + differential**, **token-adaptive weight**, R5 step 300 or M5.6.5 result)

Axes A1+A3 are jointly novel in M5.6.5. A2+A3 differential is M5.7's novelty.

---

## 4. **M5.6.5 — ROD-RLSD (R5 + On-Demand emit, single teacher / two queries)**

### 4.1 Intent

Replace R5's forced 100% emit with a per-token on-demand decision. The teacher (gold-conditioned meta SFT) is queried twice per token: (a) without forced meta, to ask "would you emit meta here?"; (b) with forced meta, to provide the content distribution if emit happens. The student learns both: emit pattern matching from (a), content distillation from (b). Decoy is removed (R5 step 300 measurement showed δ contrast = 0pp on AIME).

### 4.2 Mechanism (per student rollout token t)

```
Two teacher queries on the same model:

T_emit_query  = prompt + V0_prefix + gold + reasoning[:t]                # NO forced meta
T_content_q   = prompt + V0_prefix + gold + reasoning[:t] + <|meta|>     # WITH forced meta

p_emit_T  = softmax(T(T_emit_query).logit)[META_START_ID]
emit_target_t = (p_emit_T > τ).float()

# Student emit decision via its own logit
p_emit_S  = softmax(student.next_logit)[META_START_ID]

# Loss components
L_emit    = BCE(p_emit_S, emit_target_t)                                    # always
L_content = if (emit_target_t==1 and student emitted at t):                # gated
              KL_topK(T(T_content_q).next ~50 tokens || student same range)
L_ppo     = standard PPO/GRPO with correctness reward

total = L_ppo + α * L_emit + β * L_content
```

KV cache reuse: T_emit_query and T_content_q share 95% prefix → ~5% additional forward cost for the second query.

### 4.3 Hyperparameters (initial)

| Symbol | Value | Comment |
|---|---|---|
| τ (emit threshold) | **0.10** (post-review) | 152k-vocab teacher rarely peaks > 0.30 on natural emit; 0.10 is realistic |
| W (emit window) | **64** (post-review) | first 8 too narrow (cold start has forced META at pos 0); 64 covers natural mid-context emit |
| α (emit BCE weight) | 0.50 | moderate, matched to PPO scale via grad-norm |
| β (content KL weight) | 0.30 | light, content is secondary to emit |
| top-K | 64 | matches M5.2 |
| total_steps | 100 | R5 step 300 → 400 |
| batch | 64 | matches R5/M5.2 (controlled) |
| max_response | 4096 | matches R5/M5.2 |
| lr | 5e-7 | conservative — fine-tune of trained policy |
| Cold start | R5 step 300 (HF) | already merged |
| Eval ctx | 16k | matches all comparators |
| Single-forward semantics | (review C1) | Plan §4.2 "two queries" reduces to ONE teacher forward — same logits cover emit-decision (position 0..W) and content distillation (meta region positions). Net cost ~5%. |

### 4.4 Hypotheses

| ID | Statement | Why this would be true | Falsification |
|---|---|---|---|
| **H6.5.1** | AIME emit rate < 50% post-train | T_emit's natural emit on long chains is sparse (Why-degrade: 20× lower than forced) | rate ≥ 70% → τ too low or T over-emits |
| **H6.5.2** | AIME accuracy 17–25% (vs R5 10%) | mid-chain interruption reduced → working memory preserved | < 13% → on-demand alone insufficient, M5.7 differential needed |
| **H6.5.3** | GSM8K 80–87%, MATH 48–58% (within R5 ±5pp) | short chains: emit pattern barely changes | both ↓ → over-suppression on easy chains, τ too high |
| **H6.5.4** | meta emit rate **negatively correlated** with chain length (Pearson < −0.3) | T_emit naturally less likely to emit when chain is far in | r ≥ 0 → T_emit pattern not length-sensitive, design assumption wrong |
| **H6.5.5** | ECE ≤ 0.671 (1.10× base SFT) on AIME | content KL preserves calibration vocabulary | ECE > 0.671 → content KL too weak, β increase or decoy add-back |
| **H6.5.6** | Strategy-switch indicator (token "Wait" / "Hmm" / "Actually" rate) ≥ 1.5× R5 | sparse meta ↔ epistemic markers can re-emerge | rate ≤ R5 → meta block 자체가 markers를 흡수했음, M5.6 (meta-as-content) trigger |

### 4.5 Verification protocol

Pre-launch:
- smoke step 1: emit prob computation on synthetic logits; verify τ thresholding produces sensible binary
- smoke step 2: BCE + KL backward, gradient finite, sign correct
- smoke step 3: real toy model 1-step end-to-end; emit_target_t shape matches student logit
- smoke step 4: KV-cache reuse correctness (T_emit and T_content prefixes match)
- smoke step 5: HF cold-start integration (R5 step 300 ckpt loads, tokenizer compatible)

In-training:
- wandb logs: `train/emit_rate_target`, `train/emit_rate_student`, `train/content_kl`, `train/grad_norm`
- step 50 ckpt + 16k eval intermediate (early signal)

Post-train (16k eval at step 100):
- AIME / MATH-500 / GSM8K accuracy
- emit rate per benchmark
- avg confidence at meta blocks
- chain-length distribution
- ECE on AIME
- "Wait/Hmm" markers per 1000 tokens

### 4.6 Decision matrix from M5.6.5 results

| AIME | Action |
|---|---|
| ≥ 22% | on-demand alone strong; proceed M5.7 (differential) for further |
| 17–22% | success in predicted band; M5.7 + length-conditional add-on |
| 13–17% | partial; investigate emit pattern, possibly increase τ tuning |
| < 13% | on-demand insufficient; M5.7 differential becomes priority, length-conditional is add-on |
| ≥ 36.7% (=base_grpo) | hypothesis A *substantially* refuted; rare but possible; immediate paper-grade write-up |

---

## 5. **M5.7 — ADOPD (Adaptive Differential OPD)** — successor

### 5.1 Intent

Augment M5.6.5 with a *second* teacher (base SFT, no meta, gold-conditioned) and use the **token-level differential** `log T_meta − log T_base` as a gating signal: meta information is "informative" only where the meta-conditioned distribution actually diverges from the base distribution. Apply distillation only at high-informativeness positions.

### 5.2 Mechanism (changes vs M5.6.5)

```
T_meta = gold-conditioned meta SFT (same as M5.6.5)
T_base = gold-conditioned BASE SFT (no meta in vocabulary or tokens stripped)  ← new

Per token t:
  delta_t = log T_meta(x_t | prefix_t) - log T_base(x_t | prefix_t)
  weight_t = sigmoid(delta_t - margin)        # margin = 0.5 default

  L_content = weight_t * KL_topK(T_meta || S)  # gated by informativeness
  
  Other components same as M5.6.5
```

This is **A3 (two real teachers) + token-adaptive weighting** added on top of M5.6.5.

### 5.3 Hypotheses (deferred until M5.6.5 result)

| ID | Statement | Falsification |
|---|---|---|
| H7.1 | AIME accuracy ≥ M5.6.5 + 3pp (additional gain from differential) | < M5.6.5 → differential has no marginal value, single-teacher KL sufficient |
| H7.2 | weight_t distribution is bimodal (high near meta-important tokens, low elsewhere) | uniform → T_base too similar to T_meta, our base SFT was contaminated by meta data |
| H7.3 | Calibration ECE ≤ M5.6.5 (no regression from differential) | ↑ → gating dropped some calibration-essential positions |

### 5.4 Cold start / cost
Cold start = M5.6.5 result. Total steps = 50–100 (incremental fine-tune). T_base is `v8_base_matched_strict_sft` (already trained, on HF).

---

## 6. Method comparison matrix (final)

| Method | A1 emit | A2 signal | A3 teacher | A4 mask | A5 cold | Steps | Predicted AIME |
|---|---|---|---|---|---|---|---|
| base SFT | natural | — | — | — | — | — | 10.0% (measured) |
| `e21r_v2` (legacy) | natural | scalar | single | none | base SFT | 300 | 13.3% (measured) |
| `base_grpo` | none | none | none | none | base SFT | 300 | 36.7% (measured) |
| R5 step 200/300 | forced 100% | scalar+decoy | two (gold+decoy) | meta-only | base SFT/R5_200 | 200/300 | 10.0% (measured) |
| **M5.2 OPD-Decoy** ⏳ | forced 100% | full-logit+decoy | two (gold+decoy) | meta-only | R5 step 200 | 200 | 13–17% (forecast) |
| **M5.6.5 ROD-RLSD** ★ | **on-demand T_emit** | content KL | **single, two queries** | per-token | R5 step 300 | **100** | **17–25% (forecast)** |
| M5.7 ADOPD | on-demand | content KL + differential | **two real (meta+base)** | informativeness-gated | M5.6.5 result | 50–100 | 22–30% (forecast) |
| M5.4 (deprecated) | forced 100% | full-logit | two real (meta+RLVR) | region-fixed | R5 step 300 | 200 | 22–30% — duplicates M5.7 capability with less flexibility |

---

## 7. Self-review log

**Round 1 (intent clarity)**:
- Clear: M5.6.5 is differentiated on emit decision (axis A1) AND teacher count interpretation (axis A3 single-vs-two-query).
- Verified: each axis has a measurable hypothesis that maps to a wandb / eval metric.

**Round 2 (hypothesis falsifiability)**:
- H6.5.1 emit rate threshold rationale: Why-degrade reports 20× reduction in epistemic markers when forced→natural; if our emit goes from 87% to 4–8%, that's the realistic range. Setting < 50% is generous.
- H6.5.4 chain-length correlation: this is the *direct* falsification of design assumption "T_emit naturally emits less on long chains." If r ≥ 0, the entire design is wrong.

**Round 3 (no duplicate methods)**:
- M5.4 (region-fixed two-teacher) is **strictly weaker** than M5.7 (token-adaptive two-teacher with differential gate). Removing M5.4 from ladder.
- M5.3 (DualTeacher scalar) is now also subsumed by M5.7 once M5.6.5 establishes on-demand baseline.
- Final ladder: M5.2 (in flight) → **M5.6.5** (next) → M5.7 (after).

**Round 4 (cost vs gain)**:
- M5.6.5 is cheap: single teacher with two queries, KV cache reuse, no new model. ~5% per-step overhead.
- M5.7 adds T_base forward pass = +~25–30% per-step (separate model). Justified only if M5.6.5 hits 17–22% (decision matrix §4.6).

**Round 5 (controllable confounds)**:
- Batch unified at 64 (matches R5, M5.2) — no batch confound vs M5.2.
- Total steps 100 (additive on R5 step 300) — total exposure 400 step ≈ R5 step 300 + 1/3 — controlled.
- Cold start unified to R5 step 300 — controls for "did R5 itself help?"
- Eval at 16k — matches base_grpo, e21r_v2.

**Round 6 (residual risks)**:
- *T_emit may itself be over-emitting if gold-conditioned*: mitigated by V0 prefix in T_emit_query (V0 = pre-meta natural reasoning), so T sees student's actual prefix not gold-anchored prefix. Verify in smoke step 4.
- *KL on top-K when meta block is short (~10 tokens)*: gradient may be noisy. β=0.30 conservative.
- *PPO clip fraction abort callback* same as M5.2.

**Round 7 (no critical issues remaining)** — frozen.

---

## 8. Implementation manifest

### Files to add
- `src/training/meta_rod_trainer.py` — `MetaRODConfig`, `MetaRODTrainer` (extends `MetaRLSDTrainer`)
- `scripts/train_meta_rod.py` — entry point, mirrors `train_meta_opd.py`
- `scripts/smoke_meta_rod.py` — 5 smoke steps
- `configs/meta_rod_R8_h100_4x4k.yaml` — config (see §4.3)
- `h100_meta_rod_R8_0507.yaml` — AMLT yaml (H100 80G×4 Standard tier)

### Files to reuse
- `src/training/meta_rlsd_trainer.py` (parent class, `correctness_plus_meta_floor_reward`, `_build_grpo_config`, `ClipFractionAbortCallback`, `_partial`)
- `src/training/meta_opd_trainer.py` (KL top-K helper, T_content query pattern)
- `src/metacot/prompt.py` (META_START, META_END)
- `src/training/meta_rlsd_data_pipeline.py` (load_meta_rlsd_dataset, preflight_checks)
- `scripts/eval_vllm_1030.py` (16k eval)
- `scripts/build_sdc_code_snapshot.sh` (tarball + push to HF before launch)

### HF artifacts (pre-existing)
- Cold start: `iamseungpil/metacot-h100-rlsd-forced-meta-R5-0504` step 300 (merged form: noproduce on cold-start node)
- Teacher meta SFT: `iamseungpil/metacot/models/v8_meta_inside_strict_sft/checkpoint-254` (already staged via yaml step 6.5)
- Eval push target: `iamseungpil/metacot/eval/meta_rod_R8_step_final_2026_05_07`

---

## 9. Changelog

- **2026-05-07 (v5.4 update)** — Post-codex review (3 rounds). M5.6.5 v1 (current `meta_rod_trainer.py`) found to have 4 critical issues: (C1) collapsed two teacher queries into one forward, (C2) emit_window=64 starves AIME mid-chain signal, (C3) R5 step 300 cold start saturates emit prior, (C4) meta floor reward fights on-demand objective. **M5.6.5 v2 design (consensus)**: cold start = base SFT (not R5 step 300), soft Bernoulli BCE (target = `p_emit_T` continuous, not hard threshold), 16 sampled non-meta body positions per rollout (not first-64), two real teacher forwards (T_emit no-force + T_content with META-injected at sampled), meta floor reward disabled, replaced by `emit_rate_penalty = gamma * (EMA(p_T) - actual_rate)^2`. Forecast AIME (80% CI): **24–31**, center 27–28. Implementation tracked in `meta_rod_v2_trainer.py` (M5.6.5 v2 / "R9").
- **2026-05-07** v5.3 created: M5.6.5 ROD-RLSD primary, M5.7 ADOPD successor, M5.3/M5.4 deprecated, R5 16k results integrated.
- **2026-05-06** v5.2 (predecessor): M5.1 R5 step 300 + M5.2 OPD-Decoy parallel, BSC tier issue.
- **2026-05-05** v5.1: forced injection helper, exit 255 fix.
- **2026-05-04** v5.0: Multi-Teacher OPD pivot.
- **2026-05-03** v4.0: R5 forced-meta design.

---

*Frozen 2026-05-07 after 7 self-review rounds. No critical issues remaining.*
