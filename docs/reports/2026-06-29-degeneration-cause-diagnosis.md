# Why Meta-RL Models Degenerate / Fail to Terminate on Long Reasoning

Date: 2026-06-29
Analyst synthesis over forensics D1–D6 + adversarial verification V1–V2.

Models compared: BASE (correctness-only GRPO) vs 5 meta checkpoints across the reward ladder
cf130 → gmadd170 → pmi200 → gmmult301 → pmishift300 (step / reward-variant entangled).
Benchmarks: gsm8k (easy/short), math500, aime2024. Eval cap = 16384 tokens. Train rollout cap = 4096.

---

## 0. The headline tension and how it resolves

The task's interpretation rule: if collapsed traces OFTEN had the correct answer before collapsing
→ capability INTACT → stability/termination problem (C2/C4/C5), cheaply decode-fixable.
If RARELY → capability erosion (C1), needs hard-data retraining.

The forensics force a **two-part** answer that the single rule alone would mislabel:

1. **Recoverable-by-decode fraction is ~0%** (D1/V1: genuine correct-before-collapse = **1/71 ≈ 1.4%**
   across all ckpts; **1/62 ≈ 1.6%** on the two recomputed). A stop/anti-repetition token recovers
   almost no ACCURACY because there is no correct answer upstream to stop on.

2. **BUT the degeneration is still a STABILITY failure, not capability-loss-relative-to-base.**
   D6 (capability-controlled: items BOTH base and meta get wrong, tok>3000) shows base fails the SAME
   hard items **coherently** — 0.0% degenerate, 84–92% emit a clean (wrong) `\boxed`, ~8–16% truncate —
   while meta fails the identical items **by degenerating** (14–69% degenerate, clean-boxed collapses to
   22–47%, trunc up to 83%). Meta did not "forget math base could do"; on the matched-wrong set both
   miss the problem. Meta specifically lost the ability to TERMINATE/commit and instead decays into a
   repetition sink.

Resolution: **the degeneration/non-termination phenomenon itself is generation-instability (C5+C4),
meta-induced and length-gated.** The 1/71 number does NOT implicate C1 forgetting as the driver of
degeneration (D6 refutes that by matched-wrong control); it only means the decode fix cures the
*pathology* (clean termination, token savings, no 16k runaways) **without** moving accuracy. Accuracy is
a separate, orthogonal capability ceiling (these are genuinely hard items both models miss).

---

## 1. Ranked causes (most → least supported)

### #1 — C5: repetition self-reinforcement into a structural-token sink + non-termination  [STRONGLY SUPPORTED — primary mechanism]

- D4: repetition rises as a **graded ramp, not a step function** — mean repeated-4gram fraction
  0.335 (0–1k chars) → 0.381 (4–5k) → 0.417 (8–9k) → **knee 0.494 (10–11k) → 0.678 (11–12k) →
  0.822 (12–13k) → plateau 0.88–0.93 (13–15k)**. Normalized-decile shape identical
  (pmishift d0 0.364 → d3 0.487 → d4 0.786 → d5 0.935). Self-feeding over ~2k chars = textbook
  self-reinforcement (Holtzman/Xu/Welleck).
- D4 sink identity: the attractor is **the bare newline**, not brackets. pmishift last-3000-char counts:
  `'\n'` 90183, `'\n\n'` 73219 (blank-line spam), lone `'\\'` 3151. gmmult is even more
  backslash-dominated: `'\\'` 10212, `'\n\n'` 44140. None ever resolve into a final answer — decay into a
  whitespace/backslash void.
- D6: base 0.0% degenerate on BOTH benchmarks (0/12 aime, 0/19 math500); every meta ckpt 14–69%.
- Degeneracy ≈ truncation: pmishift 39/46 truncated traces degenerate; gmmult 28/32.

### #2 — C4: length-bound EOS / termination de-calibration beyond the train cap  [STRONGLY SUPPORTED]

- D5 decisive table — termination by length bucket:
  `<2k`: every model 1.00. `8–16k`: base **0.75 (12/16)** vs **every meta = 0.00**.
- Ran-to-cap-no-final among traces >4k tokens: base 0.114 → cf 0.267 → gmadd 0.438 → pmi 0.341 →
  gmmult 0.667 → **pmishift 0.830** (~7× base).
- Difficulty-gated: gsm8k 1.000 for all; math500 base 0.994→pmishift 0.936; **aime base 0.967→pmishift 0.500**.
- Collapse onset median ~7–10k chars/tokens — i.e. **past the 4096 train rollout cap**, exactly where EOS
  behavior is out-of-distribution. Termination is fully intact for short/easy traces; the failure only
  appears once a trace exceeds ~4k tokens. This is length-OOD termination calibration (Newman), not a
  global EOS loss.

### #3 — C3: reward over-optimization vs step  [DESCRIPTIVELY SUPPORTED; CAUSALLY WEAK — confounded]

- D3: monotonic degradation with ladder rank — degen rho +0.90, trunc +1.00, onset +1.00, len_wrong +0.90
  (AIME+MATH500). Every meta ckpt worse than base (base degen 0.000, trunc 0.020). Gap widens:
  pmishift truncates 22.1% combined / 65.2% aime wrong vs base 2.0%/5.3%. Length inflates without
  accuracy gain among-wrong = Gao's over-optimization signature.
- BUT (D3 + V2): step is **fully confounded with reward variant** — 5 different reward functions, no
  same-method/different-step pair. Raw-step Spearman drops to **0.7** (vs 0.9 by rank); **pmishift (s300)
  is worse than gmmult (s301)** on every degradation metric → method, not step, drives the gradient.
  Supports "heavier-shaping reward over-optimizes the length proxy"; does NOT support a causal
  "more steps → more over-opt."

### #4 — C2: RLHF mode/diversity collapse  [MIXED — refuted at vocab level, real at scaffold level, not the driver]

- D2 original: NOT supported — meta is **longer not narrower** (gsm8k base 119 tok → meta 220–405;
  CV base 0.441 → meta 0.492–0.671), distinct-2 ≥ base on 4/5 (math500 base 0.9146 vs meta up to 0.9614),
  and math500 CV rises monotonically 0.586→2.070 (tail-blowup, the opposite of narrowing).
- V2 audit: length-expansion claim **survives** difficulty-matched/paired. The "no narrowing" claim is
  **partly reversed once length is controlled**: first-80-token-window distinct-2 is consistently LOWER for
  meta (gsm8k base 0.60 vs meta 0.42–0.45), and the **12-token opening prefix** is severely templated —
  base distinct-prefix = **1.000** (every problem unique opening) vs **gmadd 86.5% identical opening
  (distinct 0.038)**, gmmult 81.9%, pmishift ~68%.
- Net: meta narrows the SCAFFOLD/opening distribution while inflating body length. This is a real texture
  but it is the Kirk-style narrowing of *structure*, not vocabulary collapse, and it is not the proximate
  cause of the newline-sink degeneration. Contributing, not root.

### #5 — C1: catastrophic forgetting / capability erosion  [NOT SUPPORTED as the degeneration driver — refuted by capability control]

- The literal D1 number (genuine correct-before-collapse 1/71 ≈ 1.4%, V1 1/62 ≈ 1.6%) means the
  destabilized traces have no correct answer upstream, so decode fixes recover ~0% accuracy.
- But D6 controls for capability: on items BOTH base and meta get wrong (tok>3000), base fails
  **coherently** (0% degen, 84–92% clean boxed) and meta fails **by degenerating**. Same problems, same
  (lack of) capability — only the failure MODE differs. Therefore meta did not lose math ability that base
  retained; "forgetting relative to base" is refuted. The 1/71 caps *accuracy-recoverability*, not the
  *cause of degeneration*.
- Caveat for honesty: D1's loose-extraction 20–36% "capability intact" reading is a confirmed artifact
  (V1: every loose hit = intermediate arithmetic / constraint line / given condition, e.g. gold 480 ←
  `24u^2-20v^2=480`; gold 16 ← `=16z^3`; gold 1736 ← `bd=1736`). math_verify is reliable (no false
  negatives; only 10/62 traces box anything at all). The single genuine intact-but-destabilized trace
  (pmishift math500 gold 1/16: boxed `\frac{1}{16}` at 41% through, then re-boxed → drifted to
  `1/(16 sin π/15)` → `1/15` → `\`+newline sink) is the lone decode-recoverable case.

---

## 2. Verdict (single sentence)

**Root cause = C5 repetition self-reinforcement into a structural-token (newline/backslash) sink plus
C4 termination de-calibration on lengths beyond the 4096 train cap, both meta-induced and amplified
monotonically by length-inflating reward shaping (C3, method-driven); this is a GENERATION-STABILITY
failure, not capability loss relative to base (D6: base misses the same hard items but terminates
coherently at 0% degeneration), and although the pathology is decode-fixable, the fix recovers ~0%
accuracy because the collapsed traces had no correct answer upstream (D1: 1/71).**

---

## 3. The two most decisive next experiments (cheap / decode-time first)

### EXP-1 (cheapest, decode-time) — anti-degeneration decode sweep × length-cap, on the worst ckpts

- Setup: re-decode **pmishift_s300** and **gmmult_s301** on math500 + aime2024 (the long-trace regime;
  skip gsm8k, never triggers). Conditions: baseline greedy/sampling vs
  (a) `no_repeat_ngram_size=3`, (b) `repetition_penalty≈1.15`, (c) `min_p=0.05`,
  each at length caps {2k, 4k, 8k, 16k}.
- Metrics: termination rate (boxed/answer & tok<cap), degenerate-tail rate (D4 def: last-1500-char
  >60% punct/whitespace), ran-to-cap-no-final rate, mean tokens, and **accuracy**.
- CONFIRM (expected): termination rate → ~1.0 and degen rate → ~0 under (a)/(b)/(c) at every cap,
  while **accuracy stays ≈ base-among-wrong (recovers ~0–2 pp)**. This confirms C5/C4 stability-loss is
  the mechanism and is decode-fixable, AND confirms D1 that capability is not buried.
- REFUTE: if accuracy jumps materially (≥5 pp) under min-p / anti-repetition, then a correct answer WAS
  being buried by the sink → resurrects the capability-intact reading and implicates sampling, not
  termination per se. Would redirect the fix toward decoding, not RL objective.

### EXP-2 (cheap, data-free) — weight-averaging interpolation base ⊕ pmishift (Lin 2309.06256)

- Setup: linearly interpolate weights θ(α) = (1−α)·θ_base + α·θ_pmishift for
  α ∈ {0, 0.25, 0.5, 0.75, 1.0} (no retraining), eval on math500 + aime2024.
- Metrics: degen rate, trunc rate, termination rate, accuracy, AND meta-emission / epistemic-marker rate
  (to check meta behavior is retained, not just diluted away).
- CONFIRM: a midpoint α (~0.5) restores base-like termination (degen → ~0, trunc → ~base 2%) while
  keeping accuracy ≥ base and retaining meta markers → degeneration is a recoverable weight-space
  perturbation (stability), and yields a free deployable fix.
- REFUTE: if no α simultaneously terminates cleanly AND ≥ base accuracy AND keeps meta markers, the meta
  reward objective is intrinsically incompatible → must change the objective (dense n-gram penalty in RL,
  Yeo 2502.03373) or accept the soup as a behavior-trade fallback.

(Deferred / more expensive third option, only if EXP-1/2 implicate the objective: continue-training the
worst variant with **rollouts at eval length (>4096, ideally 16k)** + Yeo dense n-gram repetition penalty
in the reward, to calibrate EOS on the lengths actually seen at eval. Not first because it is the costly
RL path and the two cheap experiments above are expected to settle the mechanism.)

---

## 4. Recommended fix (given the verdict)

Two tiers, because the *pathology* and the *accuracy ceiling* are separate problems:

1. **Immediate (cures the degeneration pathology — it is stability-loss, decode-fixable):** deploy
   decode-time anti-degeneration at inference — `no_repeat_ngram_size=3` plus `repetition_penalty≈1.1–1.15`
   (or `min_p≈0.05`). This eliminates the 16k newline/backslash runaways, restores clean termination
   (D5 0.00 → ~1.0 in the 8–16k bucket), and recovers the wasted token budget. Pair with
   base⊕meta weight-averaging (Lin, data-free) as a fallback that restores base termination while
   retaining meta behavior.

2. **Root fix for future RL training (fixes C4+C5 at source):** (i) train rollouts AT or ABOVE eval
   length (≥4096, target 16k) so EOS is calibrated on in-distribution lengths rather than extrapolating
   past the 4096 cap; (ii) add Yeo 2502.03373 **dense n-gram repetition penalty** into the reward to
   kill the length-hacking self-reinforcement; (iii) prefer additive / lighter reward shapes — the
   degradation gradient is method-driven (pmishift/gmmult are the worst), so the multiplicative /
   shifted variants over-optimize the length proxy hardest.

3. **Do NOT prioritize hard-data retraining for the DEGENERATION problem** — D6 refutes capability-loss
   vs base. However, note explicitly: **none of the above will move accuracy** (D1: 1/71 had a correct
   answer upstream). Raising accuracy on these hard items is an orthogonal capability problem that does
   require harder/longer training data — consistent with prior project findings that meta SFT trained on
   easy metadata erodes hard-math capability while the failure shown here is a separate
   generation-stability artifact layered on top.
