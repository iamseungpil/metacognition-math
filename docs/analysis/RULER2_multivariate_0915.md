# ruler2 — can a multivariate / positional / hidden-state readout clear .65 within-problem?

Scripts: `common.py` (harness), `sanity.py`, `r1.py`+`r1b.py`, `r2_prep.py`+`r2.py`, `r3.py`+`r3b.py`, `r4.py`.
All in this directory; env `/hdd_data/seungpil/envs/simplerl/bin/python`, CPU only, nothing written to the repo.

**Verdict: no. Every variant stays below .60 on the truncation-excluded headline. The readout ceiling
at 4B is confirmed; multivariate, positional and hidden-state combinations do not rescue it.**

## 0. Harness validation
`sanity.py` reproduces the published univariate ALL table **exactly** (mean_logp .559, max_entropy .566,
lowest_window_logp .556, …) from `scores.jsonl`, and confirms `scores.jsonl` row *i* ↔ `texts.jsonl`
row *i* for all 10,400 rows (group_id, r_corr, n_tok all match), and `entropy_series.npz` row order
identical with `len == n_resp_tok` for every row.

Protocol everywhere below: **GroupKFold by problem (group_id), 5 folds, 5 seeds, out-of-fold predictions
only**; within-problem AUC = mean of per-problem AUCs over problems that are *mixed among the rows
actually used*; 95% CI = 2,000-sample bootstrap over problems (seed-0 model). Missing features
(mean_logp_at_forks 200, answer_logp 484) imputed with the training-fold median. No sklearn exists in any
env on this box, so logistic regression is `src.rulers.hidden_probe.fit_logreg` (IRLS, L2=1.0) and the
gradient booster is a purpose-written histogram GBM (depth 3, 200 trees, lr .05, 32 bins) in `common.py`.

## R1 — multivariate readout (12 metrics + n_resp_tok + truncated)

**HEADLINE — non-truncated rows: 9,808 rows · 1,292 problems · 140 mixed**

| model | features | within-problem AUC (mean of 5 seeds) | 95% CI (seed 0) |
|---|---|---|---|
| logreg | raw | 0.486 (sd .006) | [0.444, 0.548] |
| logreg | within-problem z | 0.515 (sd .016) | [0.463, 0.568] |
| gb | raw | 0.501 (sd .005) | [0.456, 0.557] |
| gb | within-problem z | 0.505 (sd .014) | [0.455, 0.558] |
| *best single metric on the same rows* (mean_logp) | — | *0.557* | *[0.503, 0.610]* |
| *sibling_share reference on the same rows* | — | *0.845* | *[0.782, 0.904]* |

Combining the twelve metrics is **at chance** and **worse than the best single metric**. It does not
approach .65.

All rows (10,400 · 1,300 problems · 208 mixed): logreg raw .661, logreg z .675, gb raw .684, gb z .659.
This is the truncation confound and nothing else — `not_truncated` alone scores .807 and `n_resp_tok`
.663 on that population. Per source, non-truncated: L5 (5,949 rows, 110 mixed) .500–.521;
MATH-500 (3,859 rows, 30 mixed) .516–.591 (n=30, CI [.497,.688] — noise).

## R2 — positional readout at the sibling divergence point

`r2_prep.py` tokenized every rollout with the policy tokenizer (Qwen3-4B-Instruct-2507, variant
`math_opt`, prompt rebuilt with `render_generation_prompt`); **0 roundtrip failures, all 10,400 rows
usable**, `len(ids)-n_prompt == n_resp_tok` for every row, so d indexes the entropy series directly.

**The divergence point is at the very start of the response.** d_group (LCP of all 8) median **5**
tokens (p10 2, p90 37); d_self (max LCP with any sibling) median **47** (p10 9, p90 104) — against a
median response length of 1,393 tokens. Two group-level features are undefined by construction:
entropy *at* d_group and mean entropy *before* d_group are identical across siblings (same prefix,
same context), so their within-problem AUC does not exist.

**HEADLINE — non-truncated rows (9,808 · 140 mixed), univariate:**

| feature | AUC | CI | | feature | AUC | CI |
|---|---|---|---|---|---|---|
| ds_before (mean H over [0,d_self)) | **0.579** | [0.532, 0.630] | | ds_at_d | 0.477 | [0.428, 0.525] |
| ds_after_all | 0.539 | [0.483, 0.593] | | ds_d32 | 0.497 | [0.450, 0.545] |
| dg_after_all / dg_diff_all | 0.538 | [0.482, 0.593] | | ds_d128 | 0.501 | [0.449, 0.550] |
| d_self_frac | 0.525 | [0.470, 0.580] | | ds_diff128 (after−before) | 0.461 | [0.407, 0.513] |
| dg_d32 / dg_d128 / dg_diff128 | 0.497–0.501 | ~[0.45, 0.55] | | ds_diff_all | 0.490 | [0.433, 0.545] |

Multivariate on the same rows: positional only .490–.531; 12 metrics + positional .481–.533.
On all rows the same models reach .61–.67 — again the truncation confound (d_self_frac alone jumps to
.597 there because it is a length proxy).

Entropy **at** the divergence point, and the entropy **rise** across it, are at chance (.46–.50). The one
feature whose CI excludes .5 is `ds_before` = mean entropy over the first ~47 tokens, i.e. a *prefix*
statistic, not a "moment it goes wrong" statistic, and it is .579 — below .60.

## R3 — hidden state + token statistics (L5 rows that emitted a meta block)

Alignment reproduced exactly as `c1.py` (1,941 emitted rows ↔ npz; targets and group_ids identical).
Probe = `math_ruler_pivot.grouped_oof_probe(hidden_metaend_L36, 5 folds, l2=1.0)`, verbatim.

**HEADLINE — L5 emitted, non-truncated: 1,939 rows · 652 problems · 37 mixed**

| feature set | best model / repr | AUC (5 seeds) | CI (seed 0) |
|---|---|---|---|
| probe alone | — | 0.565 | [0.421, 0.715] |
| 12 metrics | logreg / within-z | 0.545 | [0.443, 0.706] |
| 12 metrics + probe | logreg / within-z | 0.590 | [0.504, 0.748] |
| 12 metrics + probe, **nested** probe folds | logreg / within-z | **0.607** | [0.509, 0.773] |

On all emitted L5 rows (1,941 · 38 mixed) the probe alone gives **.577** — the G1 number reproduced to
three decimals. The combination is nominally the best figure anywhere in this study but rests on
**37 mixed problems**, its CI runs from .51 to .77, and its seed-to-seed sd is .03; it is not evidence of
a .65 readout.

The emitted subset does **not** give a larger mixed-problem set (hidden states exist only for emitted
rows → 38 mixed, the same n as G1). Pooling L5 + MATH-500 emitted rows (`r3b.py`, 3,179 rows ·
1,049 problems · **44 mixed**, non-truncated) is the largest available check and the probe **does not
replicate**: probe alone **0.528** [0.405, 0.658]; best combination 0.564 (gb / raw). The .577 is
within noise of .5 once one more population is added.

## R4 — ceiling: does any of this buy accuracy?

Selector = the R1 all-rows GB (5-seed-averaged OOF, within-problem AUC .686 — the strongest legitimate
readout available, truncation included). Vote = plurality over `final_answer`, ties broken at random
(expected accuracy); matched baseline = exact expectation over **all** C(8,8−k) random subsets.

**All 1,300 problems** — pass@1 .7051, uniform vote over 8 = **.7274**

| selector | drop-1, vote 7 | uniform 7 | drop-2, vote 6 | uniform 6 |
|---|---|---|---|---|
| R1 GB readout | .7300 | .7270 | .7322 | .7257 |
| not_truncated | .7279 | .7270 | .7305 | .7257 |
| sibling_share (reference) | .7259 | .7270 | .7267 | .7257 |
| **oracle r_corr (upper bound)** | **.7340** | .7270 | **.7381** | .7257 |

**208 mixed problems** — pass@1 .6232, uniform vote over 8 = .7628

| selector | drop-1, vote 7 | uniform 7 | drop-2, vote 6 | uniform 6 |
|---|---|---|---|---|
| R1 GB readout | .7788 | .7603 | .7925 | .7518 |
| not_truncated | .7659 | .7603 | .7821 | .7518 |
| sibling_share | .7532 | .7603 | .7580 | .7518 |
| **oracle** | **.8037** | .7603 | **.8293** | .7518 |

The decisive number is the **oracle**: a perfect per-rollout discriminator buys **+0.007** accuracy
(+0.043 on mixed problems) over a matched-count uniform vote. Majority voting over 8 already captures
almost all of the available headroom; drop-k selection has essentially nothing left to win, whatever the
readout. `sibling_share`, the .77-AUC "upper bound" signal, actually *loses* to the uniform vote as a
selector (−.007 on mixed) because it is collinear with the vote it is meant to improve.

## Caveats stated plainly
- The headline population has **140 mixed problems**; bootstrap CIs are ±.05, so anything in .45–.55 is
  indistinguishable from chance, and .58 vs .55 is not a real difference.
- Every number above is out-of-fold with problems held out whole; no figure is trained and evaluated on
  the same problems. The R3 "nested" row additionally refits the hidden probe inside each outer fold.
- The all-rows numbers (.66–.69) are reported only to show they are the truncation/length confound;
  they are not a within-problem self-knowledge readout.
