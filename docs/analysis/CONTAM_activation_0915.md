# Why showing the model its own failed solution hurts

**Data.** `/hdd_data/seungpil/scratch/eval/activation_gate_s1/gens.jsonl` (8,400 generations;
1,200 per condition on the wrong population = 150 rollouts x K=8), joined to the originating wrong
rollouts in `/hdd_data/seungpil/scratch/eval/mathL5_q3i2507_opt_b8k/texts.jsonl`.
The join key is `roll_id = "{group_id}#{line_index}"`, exactly as
`math_cited_site_gate.select_wrong_rollouts` builds it, so every generation is matched to the
specific wrong rollout it was derived from. Grading and answer comparison use
`src/training/math_meta.py` (`grade_math`, `last_boxed`, `answers_equivalent`); plurality uses
`src/training/math_dis.plurality_answer`. All CIs are 95% percentile bootstrap over **rollouts**
(2,000 resamples, seed 11), reusing `math_cited_site_gate.bootstrap_ci`. Paired contrasts are
per-rollout differences. Scripts: `a1_reproduction.py`, `a1b_sensitivity.py`, `a2_verbatim.py`,
`a2b_wait_mode.py`, `a2c_matched.py`, `a3_where_fails.py`, `a3b_defend_whole.py`, `a4a5_dose.py`,
`a4b_twoway.py`, `excerpts.py`, shared helpers in `common.py`.

The correct null for each contaminated arm is the fresh arm with the same *instruction*:
`wait` (own solution continued) vs `blind` (problem only); `external` (solution shown + error cue)
vs `blind_external` (told a previous attempt was wrong, solution not shown).

---

## A1. Reproduction of the original wrong answer

n = 150 rollouts, K = 8, 1,200 generations per condition.

| condition | reproduces orig wrong answer | rescue | no boxed answer |
|---|---|---|---|
| blind (fresh) | 0.206 [0.161, 0.256] | 0.583 [0.530, 0.637] | 0.098 |
| wait (contaminated) | **0.796** [0.746, 0.843] | 0.233 [0.179, 0.289] | 0.010 |
| external (contaminated) | **0.447** [0.381, 0.510] | 0.512 [0.452, 0.575] | 0.004 |
| blind_external (fresh) | 0.185 [0.142, 0.234] | 0.617 [0.567, 0.670] | 0.119 |

**Excess reproduction (contaminated − fresh), paired per rollout:**

| contrast | excess reproduction | sign-test p | delta rescue |
|---|---|---|---|
| wait − blind | **+0.590** [+0.532, +0.647] | 9.9e-32 | −0.349 [−0.411, −0.285] |
| external − blind_external | **+0.262** [+0.203, +0.322] | 1.4e-11 | −0.104 [−0.162, −0.045] |
| external − blind | +0.241 [+0.179, +0.302] | 1.6e-10 | −0.070 [−0.125, −0.012] |
| wait − external | +0.349 [+0.296, +0.403] | 1.4e-24 | — |

The rescue loss is almost exactly the reproduction gain: `wait` loses 0.349 rescue and gains 0.590
reproduction; `external` loses 0.104 and gains 0.262. Nearly every generation that fails in a
contaminated arm fails **by returning the same wrong answer**, not by finding a new wrong answer.
`P(neither reproduce nor rescue)` = 0.212 (blind), 0.198 (blind_external), 0.041 (external),
−0.029 (wait). The small negative is a grading artefact: 8/150 rollouts have a `last_boxed` that is
math-equal to gold but were graded 0 by `grade_math` on the full text (`\text{(D) } 325` vs `325`).
Dropping those 8 (`a1b.md`) does not change anything: excess reproduction wait−blind +0.621
[+0.563, +0.679], external−blind_external +0.270 [+0.213, +0.331], and `P(neither)` becomes
0.019 (wait) / 0.089 (external) vs 0.251 / 0.246 fresh.

---
## A2. Verbatim continuation — does `wait` copy the prefix?

Every generation is compared against the original wrong rollout. `wait` and `external` had that
text in context; `blind` and `blind_external` did not — for them it is the prefix they *would* have
had, which is the right null for "how much overlap arises just from solving the same problem again".

**Raw (unmatched) statistics, n = 150 rollouts:**

| condition | prefix in ctx | LCS chars | content-word overlap | numeric-literal LCSubseq / n | mean gen chars |
|---|---|---|---|---|---|
| blind | no | 214.3 [200.1, 228.8] | 0.814 [0.802, 0.825] | 0.411 [0.383, 0.436] | 11,384 |
| wait | YES (same turn) | 37.1 [34.5, 40.2] | 0.760 [0.744, 0.776] | **0.697** [0.662, 0.729] | 1,934 |
| external | YES (prior turn) | 79.5 [75.1, 84.2] | 0.737 [0.720, 0.753] | 0.393 [0.368, 0.418] | 9,571 |
| blind_external | no | 128.1 [121.6, 135.2] | 0.806 [0.793, 0.817] | 0.382 [0.358, 0.405] | 11,892 |

| paired contrast | d LCS chars | d word overlap | d numeric LCSubseq |
|---|---|---|---|
| wait − blind | −177.2 [−191.2, −163.9] | −0.054 [−0.068, −0.041] | **+0.286** [+0.254, +0.320] |
| external − blind_external | −48.6 [−55.0, −42.2] | −0.069 [−0.081, −0.057] | +0.011 [−0.018, +0.040] |

Longest-common-substring is **lower** in the contaminated arms, and 0.1% of `wait` generations share
a ≥200-char span with the prefix (vs 23.3% of `blind`). But this is length-confounded: fresh arms
emit ~6× more text, and every overlap statistic grows with generation length.

**Length-matched (`a2c_matched.py`), every generation truncated to its first 1,200 characters and
first 60 numeric literals before the same statistics are computed (the prefix is kept whole;
`wait` continuations shorter than 1,200 chars are kept as-is, which if anything favours the fresh
arms):**

| condition | mean chars compared | LCS chars | content-word overlap | numeric-literal LCSubseq / n |
|---|---|---|---|---|
| blind | 1,198 | 194.4 [183.9, 205.5] | 0.876 [0.867, 0.885] | 0.856 [0.839, 0.873] |
| wait | 774 | **34.5** [32.1, 37.2] | 0.759 [0.742, 0.776] | **0.503** [0.472, 0.535] |
| external | 1,200 | 65.6 [61.7, 70.0] | 0.733 [0.721, 0.744] | 0.659 [0.631, 0.688] |
| blind_external | 1,200 | 121.3 [114.9, 128.5] | 0.862 [0.851, 0.873] | 0.842 [0.823, 0.861] |

| paired contrast | d LCS chars | d word overlap | d numeric LCSubseq |
|---|---|---|---|
| wait − blind | **−160.0** [−169.5, −149.7] | **−0.117** [−0.133, −0.102] | **−0.353** [−0.384, −0.324] |
| external − blind_external | −55.7 [−62.4, −49.3] | −0.130 [−0.138, −0.120] | −0.183 [−0.211, −0.156] |

LCS ≥150 chars: 0.2% of `wait`, 3.3% of `external`, 36.2% of `blind`, 27.3% of `blind_external`.

**The answer to A2 is a clean no.** Length-matched, the contaminated arms overlap the wrong prefix
*less* than the fresh arms do — in literal substring, in content-word reuse, and in the ordered
numeric-literal subsequence. `wait` does **not** redo the same steps in the same order (0.503 vs
0.856 for a fresh solve in the same window). It writes short, new verification prose and then
re-emits the same **answer**. The reproduction measured in A1 is at the level of the answer, not
the text.

---
### A2b. The shape of the continuation (`a2b_wait_mode.py`)

| condition | mean chars | median chars | mean numeric literals | confirmation verdict in last 300 chars | truncated |
|---|---|---|---|---|---|
| blind | 11,384 | 11,058 | 560.4 | 0.372 [0.333, 0.407] | 0.132 |
| wait | **1,934** | **733** | **105.9** | 0.325 [0.287, 0.362] | **0.011** |
| external | 9,571 | 9,038 | 456.2 | 0.276 [0.237, 0.318] | 0.026 |
| blind_external | 11,892 | 11,722 | 591.1 | 0.376 [0.338, 0.415] | 0.157 |

The original rollouts they continue from average 10,728 chars. **`wait` does not spend the budget —
it declines to.** The median continuation is 733 characters (6.6% of a fresh solve) with 106 numeric
literals against 560, and truncates at 1.1% against 13.2%. The contaminated arms both finish early
and well inside the 8,192-token budget.

---
### A2d. Rescue / reproduction bucketed by the generation's own length (`a2d_effort_match.py`)

| condition | <1k | 1-3k | 3-6k | 6-10k | >10k |
|---|---|---|---|---|---|
| blind | n=4 | n=104<br>resc 0.75<br>repro 0.33 | n=194<br>resc 0.68<br>repro 0.24 | n=224<br>resc 0.70<br>repro 0.25 | n=674<br>resc 0.49<br>repro 0.16 |
| wait | n=754<br>resc 0.13<br>repro 0.92 | n=257<br>resc 0.19<br>repro 0.82 | n=78<br>resc 0.65<br>repro 0.29 | n=66<br>resc 0.85<br>repro 0.12 | n=45<br>resc 0.49<br>repro 0.40 |
| external | - | n=33<br>resc 0.94<br>repro 0.06 | n=210<br>resc 0.71<br>repro 0.31 | n=468<br>resc 0.49<br>repro 0.50 | n=489<br>resc 0.42<br>repro 0.48 |
| blind_external | - | n=85<br>resc 0.75<br>repro 0.40 | n=198<br>resc 0.76<br>repro 0.21 | n=226<br>resc 0.78<br>repro 0.21 | n=691<br>resc 0.50<br>repro 0.14 |

The `wait` deficit is confined to short continuations: at <1k chars (754/1200 generations) rescue
is 0.13 with 0.92 reproduction, but at 3–6k and 6–10k chars it is 0.65 / 0.85 with 0.29 / 0.12
reproduction — matching or beating `blind` at the same length. `external` behaves oppositely: at
6–10k chars it rescues 0.49 with 0.50 reproduction while `blind_external` at the same length
rescues 0.78 with 0.21 reproduction. Effort buckets are outcomes of the same generation, not a
randomised treatment.

---

## A3. Where `external` fails

Lexicon built by reading 30 random `external` generations (seed 11, saved in
`samples_external.txt`), regexes applied to the **first 400 characters**, documented in
`a3_where_fails.py`. All 30 opened by **agreeing with the user** — none defended the solution in
its opening — so the spec's class (ii) needed a companion class (ia): *agrees an error exists,
announces a re-examination, names no location*.

| class | share | n_gen | rescue **within** class | reproduces orig wrong answer |
|---|---|---|---|---|
| (i) locates a specific step/assumption as wrong | 0.388 | 465 | **0.376** | 0.529 |
| (ia) accepts the error claim, no location named | 0.606 | 727 | **0.598** | 0.398 |
| (ii) asserts the solution is correct after all | 0.003 | 3 | 0.333 | 0.333 |
| (iii) re-derives without commenting | 0.000 | 0 | n/a | n/a |
| (iv) changes the answer without saying why | 0.004 | 5 | 0.800 | 0.000 |

Agreement / re-examination marker in the first 400 chars: **0.995**. The same classifier on `wait`
(for contrast): (iii) re-derives-without-commenting **0.718**, rescue **0.065**, reproduction
**0.986**; (iv) changes-answer-mutely 0.149, rescue **0.905**; marker rate 0.093.

Class (ii) is rare *in the opening* because the model never opens by defending — it defends at the
**end**. Scanning the whole text (`a3b_defend_whole.py`) for an explicit reaffirmation ("the
original solution is correct", "the error is not present", "since you insist"):

| condition | asserts-correct anywhere | of those, reproduces | of those, rescued |
|---|---|---|---|
| external | **0.200** (240/1200) | 0.808 | 0.321 |
| blind / wait / blind_external | 0.000 | — | — |

So the loss is **not** the model defending the wrong answer up front — it capitulates instantly
(99.5%). The loss is that the capitulation is performative: the arm that asserts a specific error
location rescues at 0.376 while the arm that merely agrees to re-check rescues at 0.598, and the
"locates" arm reproduces the original wrong answer *more* (0.529 vs 0.398). Confabulated
localisation is worse than no localisation. (Caveat: class is an *outcome* of the same generation,
not a randomised treatment, so this is a decomposition of the failure, not a causal claim about
prompting for localisation.)

---

## A4. Dose-response on contaminant length

Terciles of the original wrong rollout's `n_tok` (from `texts.jsonl`), 50 rollouts each.

| tercile | n_tok | blind | wait | external | blind_external |
|---|---|---|---|---|---|
| T1 short | 333–2762 | 0.685 [0.595, 0.767] | 0.495 [0.388, 0.613] | 0.765 [0.665, 0.860] | 0.735 [0.652, 0.812] |
| T2 mid | 2969–5204 | 0.635 [0.555, 0.713] | 0.170 [0.100, 0.250] | 0.555 [0.460, 0.647] | 0.677 [0.595, 0.755] |
| T3 long | 5218–8133 | 0.427 [0.335, 0.522] | 0.035 [0.007, 0.075] | 0.217 [0.145, 0.302] | 0.438 [0.338, 0.540] |

| tercile | external − blind_external | wait − blind | reproduction: wait | reproduction: external |
|---|---|---|---|---|
| T1 short | +0.030 [−0.062, +0.117] | −0.190 [−0.312, −0.075] | 0.588 [0.480, 0.693] | 0.285 [0.175, 0.398] |
| T2 mid | −0.122 [−0.220, −0.020] | −0.465 [−0.575, −0.350] | 0.850 [0.790, 0.905] | 0.415 [0.315, 0.522] |
| T3 long | **−0.220** [−0.318, −0.117] | −0.393 [−0.485, −0.302] | **0.950** [0.912, 0.983] | **0.640** [0.547, 0.730] |

r(n_tok, external−blind_external) = −0.240; r(n_tok, reproduction wait) = +0.494;
r(n_tok, reproduction external) = +0.380.

Longer contaminants correlate with harder problems (blind falls 0.685→0.427 too), so a two-way
split (`a4b_twoway.py`, difficulty by median blind rescue × length by median n_tok within stratum)
checks that length does work of its own:

| difficulty | length | n | median n_tok | blind | external − blind_external | wait − blind | repro external |
|---|---|---|---|---|---|---|---|
| easier | short | 42 | 1634 | 0.857 | −0.051 [−0.158, +0.048] | −0.345 [−0.467, −0.226] | 0.211 [0.119, 0.310] |
| easier | long | 42 | 5056 | 0.830 | **−0.333** [−0.440, −0.226] | **−0.714** [−0.786, −0.628] | 0.443 [0.333, 0.551] |
| harder | short | 33 | 3140 | 0.292 | +0.053 [−0.061, +0.152] | −0.068 [−0.182, +0.057] | 0.530 [0.394, 0.678] |
| harder | long | 33 | 7111 | 0.208 | −0.038 [−0.140, +0.072] | −0.170 [−0.246, −0.095] | 0.667 [0.553, 0.784] |

Within the *easier* stratum, where blind rescue is flat at 0.86 vs 0.83, the external penalty goes
from −0.05 to −0.33 and the wait penalty from −0.35 to −0.71 purely with prefix length. The harder
stratum is floor-limited (blind 0.29/0.21) and shows no separable effect. The damage does grow with
contaminant length — but see the closing section for why this does not settle the budget account.

---

## A5. Answer-difficulty control (was the original wrong answer the group's plurality?)

`plurality_answer` over all K=8 siblings of the group; 29 of 150 rollouts had the wrong answer as
the plurality.

| subset | n | blind | wait | external | blind_external |
|---|---|---|---|---|---|
| orig IS plurality | 29 | 0.276 [0.172, 0.384] | 0.211 [0.091, 0.358] | 0.297 [0.168, 0.440] | 0.319 [0.207, 0.440] |
| orig is NOT plurality | 121 | 0.656 [0.602, 0.706] | 0.239 [0.180, 0.301] | 0.564 [0.497, 0.636] | 0.688 [0.634, 0.742] |

| subset | external − blind_external | wait − blind | repro wait | repro external | excess repro (wait − blind) |
|---|---|---|---|---|---|
| orig IS plurality | −0.022 [−0.112, +0.073] | −0.065 [−0.147, +0.022] | 0.875 [0.776, 0.957] | 0.746 [0.608, 0.871] | +0.293 [+0.181, +0.409] |
| orig is NOT plurality | −0.124 [−0.190, −0.055] | −0.417 [−0.488, −0.345] | 0.777 [0.719, 0.832] | 0.375 [0.310, 0.440] | **+0.661** [+0.599, +0.720] |

The contamination penalty is **concentrated where the wrong answer was NOT a strong attractor**,
the opposite of the attractive-wrong-answer prediction — but the plurality subset is floor-limited
(blind is already 0.276 there), so this contrast cannot carry much weight. The informative number
is the *excess* reproduction: showing the text adds +0.661 of pull when the answer had no
sibling support, and only +0.293 when it did. The pull comes from the text being in context, not
from the answer's intrinsic attractiveness.

---
## Verbatim excerpts (truncated to ~40 words each)

**`wait` — the dominant mode: a short consistency re-check inside the wrong frame, then a
rubber stamp.** (71.8% of `wait` generations never comment on the answer; 98.6% of those re-emit it.)

1. `g233#1865` (orig 9, gold 10, 252 chars): *"Is there any possibility of overlapping or missing?
   Another way: number of rectangles in a 3×3 grid of points is always: C(3,2)×C(3,2) = 3×3 = 9.
   Yes — this is a standard result. So no mistake."*
2. `g153#1226` (orig 64, gold 484, 447 chars): *"…ratio ≈ 6.5 / 1.99 ≈ 3.27, and 49/15 ≈ 3.266 —
   matches. Yes, correct. Thus the ratio is 49/15, m+n = 49 + 15 = \boxed{64}"* — it re-checks the
   arithmetic *of its own wrong setup* and confirms it.
3. `g642#5138` (orig 4, gold 124/13, 305 chars): *"…and that seems correct. We have A+B=1, AB=1/6,
   A²+B²=1−2(1/6)=2/3 — so (A²+B²)/(AB) = (2/3)/(1/6) = 4. Yes. Correct. Thus, the answer is
   indeed: \boxed{4}"*
4. `g292#2336` (orig 63, gold 83°, 200 chars): *"I found a similar problem online in memory: for
   triangle with angles 106°, 37°, 37°, with MAC=7°, MCA=23°, the answer is indeed 63°. Yes. Thus
   … \boxed{63}"* — appeal to recall in place of a check.
5. `g124#994` (orig 155, gold 156, 278 chars): *"Is the pair (n, n+1) being considered, and are we
   counting the number of such pairs? Yes — and for each valid n from 1000 to 1999 we are checking
   whether adding n and n+1 causes any carry. We found 125 + …"*

**`external` — the dominant mode: instant capitulation, a confabulated "critical error", then a
long re-derivation that lands back on the same answer.** (99.5% open by agreeing; 44.7% end on the
original wrong answer; 20.0% explicitly reaffirm it somewhere.)

6. `g578#4625` (orig 395, gold 396): *"You are absolutely right — I made a critical error in the
   reasoning. ### 🚨 Error Identified: I assumed that the total number of non-factorial tails up to
   z(M) is z(M) − ⌊M/5⌋, and…"* — 11,075 chars later, still wrong.
7. `g723#5785` (orig 5/2, gold 15/2): *"After extensive re-evaluation, the original solution is
   correct. The identity, algebraic simplification, and numerical consistency all support it.
   Therefore, the error is not present in the reasoning. But since you insist there is an error,
   perhaps it is…"*
8. `g326#2608` (orig 4√5/5, gold 3√5+2√10): *"…the verified answer is: \boxed{4√5/5}. But to be
   fully honest: ⚠️ I made an error in trusting a recalled result without verification. The code
   suggests √2, which contradicts the side lengths. However, after checking a reliable source…"*

---
## Which account does the data favour?

**(b) — the failed text consumes attention/budget — is rejected for the direct form, and survives
only in a weakened form.** The contaminated arms do not run out of anything: `wait` truncates at
1.1% and `external` at 2.6%, against 13.2% and 15.7% in the fresh arms, and the median `wait`
continuation is 733 characters against a fresh solve's 11,058 — the model stops six times earlier
than it needs to, well inside an 8,192-token budget. A budget account predicts the opposite
truncation ordering. The one piece of evidence that still points that way is the A4 dose-response
(the external penalty grows from −0.05 to −0.33 with prefix length inside a matched-difficulty
stratum), but prefix length is equally the amount of wrong reasoning on display, so it measures
anchor strength as well as context load, and this analysis cannot separate the two.

**The remaining two accounts split cleanly between the two arms** — the `a2d.md` table, which
buckets generations by the effort they actually spent, is what separates them:

- **`wait` is (c): a verification mode the model performs badly.** 71.8% of `wait` continuations
  never comment on the answer at all, reproduce it 98.6% of the time and rescue at 0.065; the 14.9%
  that *do* change the answer rescue at 0.905. What they do in those 733 characters is re-check the
  arithmetic *inside the wrong frame* — recomputing the same shoelace sum, re-deriving the same
  identity, appealing to recall — and then stamp it ("So no mistake", "Yes. Correct.",
  "decision: verify"). It is not copying: at matched length the contaminated arms share *less*
  literal text with the prefix than the fresh arms do, and only 0.1% of `wait` generations reuse a
  ≥200-character span. And when a `wait` continuation does spend fresh-solve effort (≥3k chars,
  16% of them) its rescue rate is 0.65–0.85 and its reproduction falls to 0.12–0.29 — at or above
  `blind` at the same length. The deficit lives entirely in the stopping decision, not in the
  ability.
- **`external` is (a): the failed text is an attractor.** Here the model *does* spend the effort —
  median 9,038 characters, 456 numeric literals, near-fresh — and still lands back on the same wrong
  answer 44.7% of the time against 18.5% fresh (+0.262 paired). At matched effort the anchor
  survives: in the 6–10k bucket `external` rescues 0.49 with 0.50 reproduction while
  `blind_external` rescues 0.78 with 0.21 reproduction. The excess reproduction is largest exactly
  where the wrong answer had *no* sibling support (+0.661 vs +0.293, A5), so the pull comes from the
  text being in context, not from the answer being intrinsically attractive. The behavioural
  signature is a sycophancy ritual: 99.5% open by agreeing they erred, 38.8% then name a specific
  step as the culprit — and those rescue at 0.376 against 0.598 for the ones that name nothing —
  while 20.0% end up reaffirming the original solution outright.

**What the data cannot separate.** (a) and (c) are not mutually exclusive and we have not isolated
them within a single arm: `wait` also reproduces at 79.6%, and `external` also performs its
verification badly (confabulated localisation). The effort-bucket and length-tercile splits are
conditioning on outcomes of the same generation, not randomised, so they decompose the failure
rather than prove a mechanism. Settling it would take an intervention that holds effort fixed — e.g.
forcing a minimum continuation length in `wait`, or showing a *different* rollout's wrong solution
in `external` to separate "any wrong solution in context" from "my wrong solution in context".

**The one-line summary.** Telling the model it failed helps because it re-solves from scratch at
full effort. Showing it *how* it failed hurts through two different mechanisms: in the same-turn
form it substitutes a cheap self-confirmation for the re-solve (it stops in 733 characters and
rubber-stamps), and in the new-turn form it re-solves at full length but stays anchored to the
displayed answer while performing an apology it does not act on. In both cases the loss is almost
entirely *reproduction of the same wrong answer*, not a new error.
