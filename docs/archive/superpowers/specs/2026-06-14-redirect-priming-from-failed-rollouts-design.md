# Redirect-Priming from Failed Rollouts — design & experiment plan

**Date**: 2026-06-14  **Author**: ctsd-phase-c autonomous loop  **Branch**: ctsd-phase-c
**Status**: APPROVED direction (user 2026-06-14: guiding = **G2′** answer-blind generation +
gold-as-acceptance-filter; order = **spec first → user review → implement**). Pending user review
of this spec before writing-plans.

---

## 1. North-star (unchanged)

Reinforce **useful** metacognition to raise **accuracy**. Metacognition is a *means*; calibration
is a sub-goal. This spec targets one specific metacognitive habit — **redirect (backtracking)**:
recognizing the current solution path is failing, lowering confidence, switching to a genuinely
different method, and then solving correctly. Redirect is the habit most associated with success on
*hard* problems (Gandhi et al. 2025, "Four Habits of Highly Effective STaRs", COLM) and the one our
rollouts almost never produce.

---

## 2. Problem — redirect is sampling-starved, not reward-starved

In DCPO v4 the PMI head can only reward metacognition that **appears in rollouts**. Redirect almost
never appears (the SFT init rarely backtracks, and on-policy sampling reproduces that base rate), so
PMI has no redirect tokens to grade — there is no gradient to strengthen. Tuning reward weights
cannot manufacture a behavior the policy does not emit.

**Independent confirmation from the s2b collapse (2026-06-14 postmortem).** The s2b run
(`triobj_dcpo_v4_stage2b_h100_4x4k`) showed that even *basic well-formed meta* is fragile under pure
RL. It built format cleanly through the warmup window (steps ~20–50: `wellformed_rate` peaked 0.40 @
s41, `meta_emit_rate` ~0.6, `correctness` climbing to 0.23, length controlled ~1000–1200), then
collapsed irreversibly:

- The trigger was **length inflation** beginning ~s55–60 (length 1162→1329→1552→1681→**1963** @ s81;
  `clip_ratio` 0.17→0.26→**0.355**), coinciding with the w_meta warmup (80 steps) bringing the dense
  PMI reward to full weight. PMI rewards longer high-likelihood continuations; `len_cost 0.03` was
  too weak to contain it.
- As responses inflated and truncated at `max_response_length=4096`, the model **stopped emitting
  meta** (emit 0.61→**0.09** @ s80) rather than emitting malformed meta (`discard_rate` fell too).
  A long solution leaves no budget to open+close `<|meta|>…<|/meta|>`, so abstention beats a
  truncated block that pays `format_neg −0.2`. This is the chain-4 abstention escape, re-triggered
  by length pressure.
- The collapse **ratcheted**: with emission gone, PMI coverage (`pmi_member_rate`) collapsed to 0.08,
  removing the quality gradient that could pull meta back; `floor 0.05 + w_emit 0.15` could not hold
  the channel open. Even after length receded post-s90 (back to ~1150), emission/format never
  recovered.

Two lessons feed this design: **(L1)** behavior (even basic meta) must be *primed*, not coaxed out
of RL alone; **(L2)** any RL stage that uses dense PMI **must contain length from step 0** (see §8).

---

## 3. Approach — STaR ∘ Gandhi ∘ SCoRe, three stages

Reuse failed rollouts as raw material: take a rollout that got the answer wrong, *splice in* a
redirect at the point its path went bad, regenerate the continuation, and keep only regenerations
that genuinely switch method **and** reach the correct answer. Those harvested traces prime the
behavior by SFT; then RL (PMI) grades it for usefulness.

### Stage A — Harvest (failed-rollout → redirect trace)

1. **Source.** From a base-model (SFT-lineage) rollout pass over the training problems, collect
   rollouts that are **incorrect** (math_verify against gold).
2. **Splice point.** Locate where the path went wrong (first error / a fixed fraction of the wrong
   trace; start simple: truncate the wrong reasoning at a sampled cut point) and append a redirect
   `<|meta|>` block prompt: *"this path is not working — lower confidence, switch to a different
   method"* (reuse the redirect scenario definitions in `prompt_behavior.py` /
   `prompt_control_v5.py`).
3. **Regenerate (G2′ — answer-blind + gold filter).** Continue generation from the spliced prefix
   **without revealing gold**, high-temperature, **k samples**. The gold answer is used **only** to
   *accept/reject* completed regenerations, never injected into the context — so the harvested text
   carries **zero answer-leak by construction**.
4. **Accept.** Keep a regeneration iff **(a)** it is a genuine switch — passes the redirect detector
   (reuse the redirect-detection regex in `build_v8_strict_paired_data.py`) — **and (b)** its final
   answer is correct (math_verify). Discard everything else.
5. **Leak guards (defense in depth on top of G2′).** Run the existing PMI guards on the kept trace:
   placebo correction (`pmi_placebo_fail_rate` ≈ 0) and the n-gram overlap guard
   (`pmi_ngram_threshold 0.25`) to flag any trace whose redirect text just echoes the gold/answer.
   Log acc-with vs acc-without on a held-out slice of harvested-vs-not as a leak tripwire.

Yield is the diagnostic: low accept-rate = the policy genuinely cannot solve those problems by
switching (a capability ceiling), which is itself a finding. Concentrate k on wrong problems only.

### Stage B — Prime (SFT on harvested traces)

Light SFT on the harvested redirect traces **mixed with general SFT data** (to avoid catastrophic
forgetting), low LR, few steps. Goal: a checkpoint where redirect appears in rollouts at a rate PMI
can grade — Gandhi's "presence > correctness" priming. Verify by sampling: `redirect_rate` in
rollouts rises materially above the SFT base rate; general accuracy does not regress.

### Stage C — RL (PMI, length-contained)

From the primed checkpoint, run the DCPO v4 PMI recipe — **with the §8 length containment baked in**.
PMI now has redirect samples to grade and rewards only *useful* redirect (sign-gated by correctness,
placebo-corrected). Optional SCoRe-style bonus: extra credit when a rollout transitions
wrong→right within a single response (genuine self-correction), guarding against the SCoRe "behavior
collapse" where the model ignores the correction signal.

---

## 4. Interfaces & reusable parts (minimize new code — Karpathy)

| Need | Reuse (verify at implementation) | New |
|---|---|---|
| Prefix-injection regeneration engine | `configs/cf_prefix_agent.yaml` + `dcpo_meta_open` (151669) prefix-injection machinery (currently DORMANT under `sdc_counterfactual=false`) | Harvest driver that sets cut point + redirect prompt, runs k-sample answer-blind gen |
| Redirect detection | redirect-detection regex in `build_v8_strict_paired_data.py` | thin wrapper as accept-filter |
| Redirect scenario text | `prompt_behavior.py` / `prompt_control_v5.py` | none |
| Leak guards | PMI placebo + n-gram overlap guards (`src/training/`) | acc-with/without tripwire on harvest slice |
| SFT priming | existing v8_strict SFT pipeline | mix-in of harvested parquet + LR/steps config |
| Stage-C RL | `configs/triobj_dcpo_v4_stage2b_h100_4x4k.yaml` (PMI recipe) | new config with §8 length knobs + optional SCoRe bonus |

Each unit has one purpose and a defined I/O: Harvest emits a parquet of accepted redirect traces;
Prime consumes it and emits a checkpoint; Stage-C consumes the checkpoint and emits a trained model
+ eval. They can be built and smoke-tested independently.

---

## 5. Hypotheses (falsifiable, with thresholds)

- **H-A (harvest yields redirect).** With G2′ k-sampling on wrong problems, accept-rate (genuine
  switch ∧ correct) is **> 5%** of attempted wrong problems, yielding a non-trivial trace set.
  *Falsified if* accept-rate < 1% (capability ceiling — switching doesn't recover these problems;
  report and pivot to easier difficulty band or stronger guidance G3).
- **H-A-leak (G2′ is clean).** On harvested traces, `pmi_placebo_fail_rate` ≈ 0, n-gram guard hit
  rate low, and held-out acc-with ≈ acc-without (no answer leak). *Falsified if* harvested traces
  show acc-with ≫ acc-without (leak) — then tighten the cut/guard.
- **H-B (priming raises redirect rate).** Post-Prime rollout `redirect_rate` rises **≥ 3×** the SFT
  base rate, with general accuracy within −1pp of SFT. *Falsified if* redirect_rate does not rise or
  accuracy regresses > 1pp.
- **H-C (useful redirect selected).** In Stage-C, `acc_with ≥ acc_without` holds with redirect
  present, and held-out per-benchmark self_consistency on **hard** sets (math500/aime) ≥ the
  boilerplate baseline (gsm8k 0.906 / math 0.542 / aime 0.133) and ideally ≥ Meta SFT
  (0.885 / 0.518 / 0.167). *Falsified if* redirect is accuracy-neutral/harmful held-out.
- **H-len (length stays contained).** With §8 knobs, `response_length/mean` stays < 1500 and
  `clip_ratio` < 0.20 through warmup and beyond — no s2b-style inflation. *Falsified if* length
  inflates past 1500 sustained (re-tighten len_cost / cut max_len).

---

## 6. Metrics — log & watch

- **Harvest (H-A, H-A-leak):** attempted/accepted counts, accept-rate, redirect-detector pass-rate,
  correct-rate, `pmi_placebo_fail_rate`, n-gram guard hit-rate, acc-with/without on harvest slice.
- **Prime (H-B):** rollout `redirect_rate` (pre/post), held-out accuracy vs SFT, general-data loss.
- **Stage-C (H-C, H-len):** `dcpo/redirect_rate`, **`dcpo/acc_with` vs `dcpo/acc_without`** (primary),
  `dcpo/wellformed_rate`, `dcpo/pmi_member_rate`, `dcpo/meta_emit_rate`, `response_length/mean` (< 1500),
  `response_length/clip_ratio` (< 0.20), `gdpo/correctness/mean`, and the SCoRe wrong→right rate if on.
- **Final verdict:** held-out per-benchmark self_consistency at the **same 16k / k8 / 1030 protocol**
  vs Meta SFT and the boilerplate baseline — with a matched SFT eval run so the comparison is clean
  (the existing SFT 75.4% is a different protocol).

Primary watch: **redirect_rate (does priming take?)** then **acc_with vs acc_without (is it useful?)**.

## 7. Decision tree

- H-A fails (accept-rate < 1%) → capability ceiling; narrow to a difficulty band where switching
  recovers, or escalate guidance to G3 (weak strategy hint, still answer-blind).
- H-A-leak fails → answer leak in harvest; move the cut point earlier, tighten n-gram guard, re-harvest.
- H-B fails (priming doesn't raise redirect_rate) → SFT mix too dilute or LR too low; rebalance mix.
- H-C fails (redirect held-out neutral/harmful) → redirect *presence* learned but not *usefulness*;
  revisit PMI sign-gate / add/strengthen the SCoRe wrong→right bonus.
- H-len fails → re-tighten §8 before continuing.

## 8. Length containment requirement (from the s2b postmortem — MANDATORY for Stage-C)

The s2b collapse was an uncontained length–PMI interaction. Stage-C uses the **conflict-free GDPO
reward composition** in
[2026-06-15-gdpo-conflict-free-reward-composition-design.md](2026-06-15-gdpo-conflict-free-reward-composition-design.md),
which supersedes this sketch. Key points (full detail in that spec):

- `max_response_length` stays **4096** — shrinking it (the earlier 2048 idea, now WITHDRAWN) would
  kill redirect, which is inherently a longer "wrong path + switch + new solution".
- `dcpo_len_cost` **0.06–0.10** (2–3× s2b's 0.03), warmed up with / ahead of w_meta.
- Meta-block token-length **cap** so floor/PMI do not pay for sheer meta length.
- **Medium** "open-meta-then-truncation" penalty to shut the abstention escape, applied only to rows
  that opened a meta and truncated before closing (not to legitimately-long meta-less answers).
- Composition fixes that stop heads fighting: **anchor-on-R_corr scale normalization** (PMI no
  longer buried) + **R_emit first-token routing** (emission no longer mixed into answer tokens).
- Watchdog alarm: `response_length/mean > 1500` or `clip_ratio > 0.20` sustained ⇒ stop, don't ride.

## 9. Operational (ultracode execution after this spec is approved)

- **Order (user-approved):** spec → user review → writing-plans → implement → smoke (green) → run.
- **Execution:** ultracode (Workflow) orchestration, combining **karpathy-guidelines** (surgical
  minimal change; every change traces to the request; verifiable success criteria with Keep/Discard)
  and **autoresearch** (goal-directed Modify→Verify→Keep/Discard loops) for the implement→smoke and
  the harvest-yield tuning.
- **Nodes:** Harvest = inference (1 GPU ok, k-sample gen). Prime = short SFT. Stage-C = H100 4-GPU
  (same class as s2b). Each pushes to HF with pull_resume durability; **save the best checkpoint
  separately** (s2b lost its good s40–50 ckpt to the latest-only push daemon).
- **wandb:** dedicated project `metacot-dcpo-v4` set via `config.trainer.project_name` (the env var
  is ignored by verl); one run id per stage (`redirect_harvest`, `redirect_prime`, `redirect_rl_s3`).
- **s2b:** left running per user; collapsed and not a usable-checkpoint source — observe only.
