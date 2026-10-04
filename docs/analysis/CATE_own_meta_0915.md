# Conditional metacognition (CATE) analysis — does the model's own critique help a *detectable subset* of rows?

*CPU-only re-analysis of data already on disk (cd9 gates G4/G5 + the mathL5 rollout corpus).
No new generation, no GPU. Scripts: `featlib.py`, `build_rows.py`, `a1_null.py`, `a1b_trunc.py`,
`a3_cate.py`, `a3b_dtrunc.py`, `a5_spont.py`, `a5b_presence.py`, `examples.py` in this directory.*

## Sources and sample sizes

| source | path | rows used |
|---|---|---|
| G4 critique-resolve gate (per-row, K=8 per condition) | `/hdd_data/seungpil/scratch/eval/critique_resolve_s1/per_rollout.jsonl` | **133** wrong Level-5 rollouts over **92** problems |
| G4 raw resolves (for truncation/length per condition) | `.../critique_resolve_s1/resolves.jsonl` | 4,256 = 133 x 4 conditions x 8 |
| G4 critique texts + leak flags | `.../critique_resolve_s1/critiques.jsonl` | 169 generated, 36 leaked and excluded upstream -> the 133 analysed rows are all leak-free |
| G5 information-gain ruler | `/hdd_data/seungpil/scratch/eval/critique_ig_s1/per_rollout_ig.jsonl` | 118 of the 133 rows join |
| cited-site gate | `/hdd_data/seungpil/scratch/eval/cited_site_gate_s1/per_rollout.jsonl` | 88 of the 133 rows join |
| raw Level-5 rollout corpus | `/hdd_data/seungpil/scratch/eval/mathL5_q3i2507_opt_b8k/texts.jsonl` | 6,400 rollouts / 800 problems / K=8 |

`roll_id` "gNNN#M" indexes `texts.jsonl` line M directly (verified: group id, problem text and
`final_answer == wrong_answer` match on all 133 rows). Sibling plurality is computed by clustering
the 8 group answers with `math_meta.answers_equivalent`; **114 of 133** wrong rollouts are minority
answers, 19 are the group plurality.

---

# THE GO / NO-GO (A1 + A1b)

**A1 alone says yes, there is heterogeneity. A1b says the heterogeneity is the 8k token budget, not metacognition.**

1. Raw per-row treatment effect `delta_i = p_crit_i - p_blind_i` has variance **0.0611** against a
   binomial-null variance of **0.0281** — ratio **2.17**, p(null >= observed) **< 0.001**. Both tails
   are inflated: 12.0% of rows gain >= 2/8, 20.3% lose >= 2/8, against 8.2% [3.8, 12.8] expected for
   |delta| > 0.25 under the null. On the face of it, the zero average IS a mixture.
2. But the re-solves run against an 8k budget and **54% of blind re-solves truncate**. Splitting on
   whether the critique changed the row's truncation rate destroys the excess entirely:

| subset | n rows | mean delta | var(delta) | binomial-null var | ratio | p(null>=obs) |
|---|---|---|---|---|---|---|
| all rows | 133 | -0.0291 | 0.06105 | 0.02823 | 2.16 | 0.000 |
| rows with dtrunc == 0 (critique did not change truncation rate) | 62 | +0.0060 | 0.01610 | 0.02191 | 0.73 | 0.856 |
| rows with trunc_blind == 0 and trunc_crit == 0 (nothing truncated either way) | 19 | +0.0066 | 0.02339 | 0.03088 | 0.76 | 0.692 |
| rows with trunc_blind + trunc_crit <= 0.25 | 30 | -0.0083 | 0.03226 | 0.03239 | 1.00 | 0.450 |
| completion-conditional (>=4 finished resolves per condition; rate over finished only) | 51 | -0.0361 | 0.04750 | 0.05610 | 0.85 | 0.710 |

(The last row uses a K=4 null as a conservative stand-in for the variable number of finished resolves, so its ratio is a lower bound on the excess.)

- Completion-conditional mean delta: -0.0361 [-0.1024, +0.0268] (n=51 rows).
- Truncation rate of the re-solves: blind 0.540, own-critique 0.570, donor 0.515, in-context retry 0.066.


   With truncation held fixed the variance ratio is **0.73 (p=0.86)** — at or *below* the binomial
   null, i.e. no detectable heterogeneity left — and the mean delta is **+0.006**, dead zero.
   Conditioning on `dtrunc` is conditioning on a post-treatment variable, so this is a decomposition
   rather than a clean causal claim; but it localises 100% of the excess spread in the one channel
   that has nothing to do with the content of the critique. `delta` vs `dtrunc` spearman **-0.421**
   (p=5e-7), vs the change in re-solve length **-0.359** (p=2e-5).

**Verdict on the gate: the exploitable-mixture hypothesis fails.** The part of the spread that is
real is "the critique changed how many tokens the re-solve spends"; the part that would have to
carry a metacognitive signal is indistinguishable from K=8 coin noise. Everything below is reported
for completeness and is consistent with this.

---


Rows: 133 wrong Level-5 rollouts over 92 distinct problems, K=8 resolves per condition.

### own-critique minus blind  (delta_i)  (n=133)

| stat | value |
|---|---|
| mean delta | -0.0291 |
| mean delta, cluster-bootstrap 95% CI | [-0.0722, +0.0116] |
| sd(delta) observed | 0.2471 |
| **var(delta) observed** | **0.06105** |
| var(delta) binomial null, mean [2.5%,97.5%] | 0.02810 [0.01993, 0.03742] |
| variance ratio obs/null | 2.173 |
| p(null var >= observed var) | 0.000 |
| frac delta > 0 | 0.278 (n=37) |
| frac delta >= +0.125 (1 of 8) | 0.278 (n=37) |
| frac delta >= +0.25 (2 of 8) | 0.120 (n=16) |
| frac delta >= +0.5 | 0.038 (n=5) |
| frac delta < 0 | 0.308 (n=41) |
| frac delta <= -0.125 | 0.308 (n=41) |
| frac delta <= -0.25 | 0.203 (n=27) |
| frac |delta| > 0.25 observed vs null | 0.173 vs 0.082 [0.038, 0.128] |
| quantiles (0,5,10,25,50,75,90,95,100%) | -1.000, -0.425, -0.375, -0.125, +0.000, +0.125, +0.250, +0.300, +0.750 |

```
-1.000 | obs    1 #
       | nul    0.0 
-0.750 | obs    2 ##
       | nul    0.1 
-0.625 | obs    0 
       | nul    0.4 
-0.500 | obs    4 ####
       | nul    1.3 .
-0.375 | obs    9 #########
       | nul    3.8 ....
-0.250 | obs   11 ###########
       | nul    9.3 .........
-0.125 | obs   14 ##############
       | nul   18.2 ..................
+0.000 | obs   55 #######################################################
       | nul   67.2 ...................................................................
+0.125 | obs   21 #####################
       | nul   17.9 ..................
+0.250 | obs    9 #########
       | nul    9.4 .........
+0.375 | obs    2 ##
       | nul    3.8 ....
+0.500 | obs    3 ###
       | nul    1.1 .
+0.625 | obs    1 #
       | nul    0.4 
+0.750 | obs    1 #
       | nul    0.1 
```

### own-critique minus donor-critique  (delta_i^donor)  (n=133)

| stat | value |
|---|---|
| mean delta | -0.0498 |
| mean delta, cluster-bootstrap 95% CI | [-0.0912, -0.0110] |
| sd(delta) observed | 0.2408 |
| **var(delta) observed** | **0.05799** |
| var(delta) binomial null, mean [2.5%,97.5%] | 0.02858 [0.02030, 0.03823] |
| variance ratio obs/null | 2.029 |
| p(null var >= observed var) | 0.000 |
| frac delta > 0 | 0.256 (n=34) |
| frac delta >= +0.125 (1 of 8) | 0.256 (n=34) |
| frac delta >= +0.25 (2 of 8) | 0.113 (n=15) |
| frac delta >= +0.5 | 0.030 (n=4) |
| frac delta < 0 | 0.353 (n=47) |
| frac delta <= -0.125 | 0.353 (n=47) |
| frac delta <= -0.25 | 0.188 (n=25) |
| frac |delta| > 0.25 observed vs null | 0.173 vs 0.082 [0.038, 0.128] |
| quantiles (0,5,10,25,50,75,90,95,100%) | -0.750, -0.500, -0.375, -0.125, +0.000, +0.125, +0.250, +0.250, +0.500 |

```
-0.750 | obs    2 ##
       | nul    0.0 
-0.625 | obs    4 ####
       | nul    0.4 
-0.500 | obs    7 #######
       | nul    1.3 .
-0.375 | obs    5 #####
       | nul    3.7 ....
-0.250 | obs    7 #######
       | nul    9.5 .........
-0.125 | obs   22 ######################
       | nul   19.1 ...................
+0.000 | obs   52 ####################################################
       | nul   65.4 .................................................................
+0.125 | obs   19 ###################
       | nul   18.8 ...................
+0.250 | obs   10 ##########
       | nul    9.2 .........
+0.375 | obs    1 #
       | nul    3.7 ....
+0.500 | obs    4 ####
       | nul    1.4 .
+0.625 | obs    0 
       | nul    0.4 
+0.750 | obs    0 
       | nul    0.1 
```

### in-context retry minus blind (reference arm)  (n=133)

| stat | value |
|---|---|
| mean delta | -0.1128 |
| mean delta, cluster-bootstrap 95% CI | [-0.1809, -0.0456] |
| sd(delta) observed | 0.3488 |
| **var(delta) observed** | **0.12165** |
| var(delta) binomial null, mean [2.5%,97.5%] | 0.02898 [0.02084, 0.03869] |
| variance ratio obs/null | 4.197 |
| p(null var >= observed var) | 0.000 |
| frac delta > 0 | 0.218 (n=29) |
| frac delta >= +0.125 (1 of 8) | 0.218 (n=29) |
| frac delta >= +0.25 (2 of 8) | 0.150 (n=20) |
| frac delta >= +0.5 | 0.053 (n=7) |
| frac delta < 0 | 0.459 (n=61) |
| frac delta <= -0.125 | 0.459 (n=61) |
| frac delta <= -0.25 | 0.353 (n=47) |
| frac |delta| > 0.25 observed vs null | 0.346 vs 0.085 [0.045, 0.135] |
| quantiles (0,5,10,25,50,75,90,95,100%) | -1.000, -0.750, -0.600, -0.375, +0.000, +0.000, +0.250, +0.425, +1.000 |

```
-1.000 | obs    3 ###
       | nul    0.0 
-0.875 | obs    2 ##
       | nul    0.0 
-0.750 | obs    4 ####
       | nul    0.0 
-0.625 | obs    5 #####
       | nul    0.4 
-0.500 | obs    8 ########
       | nul    1.3 .
-0.375 | obs   12 ############
       | nul    4.1 ....
-0.250 | obs   13 #############
       | nul    9.5 ..........
-0.125 | obs   14 ##############
       | nul   18.7 ...................
+0.000 | obs   43 ###########################################
       | nul   65.5 ..................................................................
+0.125 | obs    9 #########
       | nul   18.5 ..................
+0.250 | obs    8 ########
       | nul    9.3 .........
+0.375 | obs    5 #####
       | nul    3.9 ....
+0.500 | obs    4 ####
       | nul    1.4 .
+0.625 | obs    1 #
       | nul    0.3 
+0.750 | obs    1 #
       | nul    0.0 
+1.000 | obs    1 #
       | nul    0.0 
```


---


n=133 rows / 92 problems. Target `delta` = p_crit - p_blind (K=8 each).

**A2/A3 univariate screen, target = delta (own - blind)** — univariate (one feature at a time), n=133

| feature | mean | sd | beta/sd (HC3) | 95% CI | p | spearman rho |
|---|---|---|---|---|---|---|
| sol_has_meta | 0.353 | 0.478 | +0.0471 | [+0.0050, +0.0892] | 0.030 | +0.223 |
| n_numbers | 7.466 | 6.716 | -0.0460 | [-0.0882, -0.0038] | 0.035 | -0.102 |
| jac_problem | 0.070 | 0.049 | +0.0330 | [-0.0086, +0.0745] | 0.122 | +0.134 |
| n_words | 123.000 | 22.110 | +0.0205 | [-0.0123, +0.0534] | 0.223 | +0.032 |
| n_shared_quantities | 4.639 | 5.157 | -0.0215 | [-0.0577, +0.0147] | 0.246 | -0.046 |
| n_sol_tok_log | 8.155 | 0.681 | +0.0179 | [-0.0187, +0.0545] | 0.341 | +0.112 |
| recheck_same_n | 1.263 | 1.213 | +0.0195 | [-0.0212, +0.0602] | 0.350 | +0.089 |
| is_minority | 0.857 | 0.350 | +0.0123 | [-0.0190, +0.0435] | 0.442 | +0.069 |
| generic_rate | 2.413 | 1.184 | +0.0150 | [-0.0309, +0.0609] | 0.522 | +0.049 |
| specific_rate | 0.361 | 0.497 | -0.0080 | [-0.0442, +0.0282] | 0.667 | -0.139 |
| n_step_refs | 0.797 | 1.353 | +0.0054 | [-0.0233, +0.0340] | 0.715 | +0.023 |
| alt_approach | 0.226 | 0.418 | -0.0068 | [-0.0454, +0.0318] | 0.732 | -0.065 |
| words_per_sent | 43.726 | 20.318 | -0.0068 | [-0.0482, +0.0346] | 0.747 | -0.063 |
| plur_share | 0.659 | 0.187 | -0.0054 | [-0.0434, +0.0326] | 0.780 | +0.004 |
| jac_solution | 0.134 | 0.036 | +0.0055 | [-0.0345, +0.0455] | 0.789 | -0.020 |
| hedge_rate | 1.236 | 0.874 | -0.0056 | [-0.0487, +0.0374] | 0.798 | -0.019 |
| n_math_spans | 5.256 | 4.706 | -0.0048 | [-0.0472, +0.0375] | 0.824 | -0.005 |
| certain_rate | 3.164 | 1.237 | +0.0038 | [-0.0343, +0.0419] | 0.846 | +0.016 |
| self_affirm | 0.218 | 0.413 | -0.0005 | [-0.0420, +0.0410] | 0.979 | -0.006 |

**A3 multivariate, target = delta (own - blind)** — OLS on standardized features, HC3 SEs, n=133, k=19 features, R^2=0.167

| feature | beta (per 1 sd) | HC3 se | 95% CI | t | p |
|---|---|---|---|---|---|
| sol_has_meta | +0.0660 | 0.0273 | [+0.0126, +0.1194] | +2.42 | 0.017 |
| n_numbers | -0.1048 | 0.0499 | [-0.2026, -0.0071] | -2.10 | 0.038 |
| n_sol_tok_log | +0.0603 | 0.0369 | [-0.0120, +0.1327] | +1.63 | 0.105 |
| n_shared_quantities | +0.0712 | 0.0454 | [-0.0178, +0.1603] | +1.57 | 0.120 |
| jac_problem | +0.0337 | 0.0262 | [-0.0176, +0.0850] | +1.29 | 0.200 |
| (intercept) | -0.0291 | 0.0229 | [-0.0740, +0.0157] | -1.27 | 0.206 |
| recheck_same_n | +0.0335 | 0.0406 | [-0.0460, +0.1130] | +0.83 | 0.411 |
| is_minority | +0.0157 | 0.0210 | [-0.0256, +0.0569] | +0.74 | 0.458 |
| generic_rate | +0.0263 | 0.0422 | [-0.0565, +0.1090] | +0.62 | 0.535 |
| n_words | +0.0150 | 0.0264 | [-0.0368, +0.0668] | +0.57 | 0.572 |
| hedge_rate | -0.0150 | 0.0289 | [-0.0717, +0.0417] | -0.52 | 0.604 |
| n_math_spans | +0.0194 | 0.0386 | [-0.0562, +0.0951] | +0.50 | 0.615 |
| words_per_sent | +0.0124 | 0.0300 | [-0.0465, +0.0713] | +0.41 | 0.680 |
| plur_share | +0.0095 | 0.0308 | [-0.0508, +0.0698] | +0.31 | 0.758 |
| certain_rate | +0.0094 | 0.0344 | [-0.0581, +0.0769] | +0.27 | 0.786 |
| alt_approach | -0.0057 | 0.0269 | [-0.0584, +0.0470] | -0.21 | 0.833 |
| jac_solution | +0.0061 | 0.0298 | [-0.0523, +0.0646] | +0.21 | 0.837 |
| self_affirm | +0.0018 | 0.0266 | [-0.0502, +0.0539] | +0.07 | 0.945 |
| specific_rate | -0.0015 | 0.0230 | [-0.0466, +0.0435] | -0.07 | 0.947 |
| n_step_refs | +0.0006 | 0.0181 | [-0.0348, +0.0360] | +0.03 | 0.973 |

**A3 multivariate, target = delta_donor (own - donor)** — OLS on standardized features, HC3 SEs, n=133, k=19 features, R^2=0.117

| feature | beta (per 1 sd) | HC3 se | 95% CI | t | p |
|---|---|---|---|---|---|
| (intercept) | -0.0498 | 0.0231 | [-0.0950, -0.0046] | -2.16 | 0.033 |
| sol_has_meta | +0.0423 | 0.0253 | [-0.0073, +0.0920] | +1.67 | 0.097 |
| n_numbers | -0.0733 | 0.0485 | [-0.1683, +0.0217] | -1.51 | 0.133 |
| n_math_spans | -0.0502 | 0.0414 | [-0.1313, +0.0309] | -1.21 | 0.228 |
| self_affirm | +0.0320 | 0.0266 | [-0.0201, +0.0841] | +1.21 | 0.231 |
| n_sol_tok_log | +0.0399 | 0.0355 | [-0.0298, +0.1096] | +1.12 | 0.264 |
| n_shared_quantities | +0.0514 | 0.0465 | [-0.0398, +0.1426] | +1.11 | 0.271 |
| words_per_sent | +0.0299 | 0.0315 | [-0.0318, +0.0917] | +0.95 | 0.344 |
| n_words | -0.0282 | 0.0342 | [-0.0952, +0.0389] | -0.82 | 0.412 |
| hedge_rate | -0.0172 | 0.0244 | [-0.0651, +0.0306] | -0.71 | 0.482 |
| recheck_same_n | +0.0217 | 0.0336 | [-0.0441, +0.0875] | +0.65 | 0.519 |
| specific_rate | +0.0149 | 0.0278 | [-0.0395, +0.0694] | +0.54 | 0.592 |
| certain_rate | +0.0154 | 0.0343 | [-0.0518, +0.0826] | +0.45 | 0.654 |
| n_step_refs | +0.0087 | 0.0200 | [-0.0306, +0.0479] | +0.43 | 0.666 |
| alt_approach | -0.0074 | 0.0276 | [-0.0615, +0.0467] | -0.27 | 0.789 |
| jac_solution | +0.0093 | 0.0371 | [-0.0635, +0.0821] | +0.25 | 0.803 |
| generic_rate | -0.0054 | 0.0317 | [-0.0676, +0.0568] | -0.17 | 0.866 |
| is_minority | -0.0035 | 0.0250 | [-0.0526, +0.0456] | -0.14 | 0.889 |
| plur_share | +0.0039 | 0.0278 | [-0.0507, +0.0584] | +0.14 | 0.890 |
| jac_problem | -0.0027 | 0.0300 | [-0.0615, +0.0561] | -0.09 | 0.928 |

**A3 G5 information-gain features (IG subset)** — univariate (one feature at a time), n=118

| feature | mean | sd | beta/sd (HC3) | 95% CI | p | spearman rho |
|---|---|---|---|---|---|---|
| ig_advantage | -0.026 | 0.030 | +0.0487 | [+0.0034, +0.0941] | 0.037 | +0.151 |
| ig | -0.039 | 0.041 | +0.0455 | [-0.0014, +0.0925] | 0.060 | +0.101 |

**A3 multivariate + IG features (IG subset)** — OLS on standardized features, HC3 SEs, n=118, k=21 features, R^2=0.210

| feature | beta (per 1 sd) | HC3 se | 95% CI | t | p |
|---|---|---|---|---|---|
| sol_has_meta | +0.0748 | 0.0301 | [+0.0158, +0.1338] | +2.48 | 0.015 |
| n_numbers | -0.0928 | 0.0498 | [-0.1905, +0.0048] | -1.86 | 0.065 |
| plur_share | +0.0398 | 0.0302 | [-0.0194, +0.0989] | +1.32 | 0.191 |
| is_minority | +0.0300 | 0.0231 | [-0.0152, +0.0752] | +1.30 | 0.197 |
| certain_rate | +0.0390 | 0.0309 | [-0.0215, +0.0995] | +1.26 | 0.210 |
| n_shared_quantities | +0.0580 | 0.0480 | [-0.0360, +0.1519] | +1.21 | 0.230 |
| (intercept) | -0.0286 | 0.0243 | [-0.0761, +0.0189] | -1.18 | 0.241 |
| jac_problem | +0.0277 | 0.0294 | [-0.0300, +0.0853] | +0.94 | 0.349 |
| ig | +0.1004 | 0.1389 | [-0.1720, +0.3727] | +0.72 | 0.472 |
| n_sol_tok_log | +0.0335 | 0.0492 | [-0.0630, +0.1299] | +0.68 | 0.498 |
| n_math_spans | -0.0192 | 0.0326 | [-0.0831, +0.0446] | -0.59 | 0.556 |
| alt_approach | -0.0150 | 0.0284 | [-0.0707, +0.0407] | -0.53 | 0.599 |
| words_per_sent | +0.0156 | 0.0316 | [-0.0464, +0.0776] | +0.49 | 0.623 |
| n_words | +0.0121 | 0.0274 | [-0.0415, +0.0657] | +0.44 | 0.659 |
| n_step_refs | -0.0090 | 0.0215 | [-0.0511, +0.0331] | -0.42 | 0.675 |
| recheck_same_n | +0.0143 | 0.0382 | [-0.0606, +0.0893] | +0.37 | 0.709 |
| hedge_rate | -0.0121 | 0.0336 | [-0.0780, +0.0538] | -0.36 | 0.720 |
| ig_advantage | -0.0382 | 0.1212 | [-0.2756, +0.1993] | -0.32 | 0.753 |
| self_affirm | +0.0067 | 0.0291 | [-0.0504, +0.0637] | +0.23 | 0.819 |
| specific_rate | +0.0046 | 0.0255 | [-0.0454, +0.0546] | +0.18 | 0.857 |
| generic_rate | -0.0050 | 0.0445 | [-0.0922, +0.0822] | -0.11 | 0.911 |
| jac_solution | +0.0003 | 0.0342 | [-0.0669, +0.0674] | +0.01 | 0.994 |

### Depth-3 regression tree (whole sample, in-sample — descriptive only)

```
[n_words <= 127] n=133 mean=-0.0291
  [generic_rate <= 1.709] n=81 mean=-0.0725
    leaf n=17 mean_delta=-0.2426
    [words_per_sent <= 31.68] n=64 mean=-0.0273
      leaf n=26 mean_delta=+0.0625
      leaf n=38 mean_delta=-0.0888
  [sol_has_meta <= 0] n=52 mean=+0.0385
    [n_sol_tok_log <= 8.263] n=36 mean=-0.0139
      leaf n=18 mean_delta=-0.0903
      leaf n=18 mean_delta=+0.0625
    leaf n=16 mean_delta=+0.1562
```

### A3 held-out selector (GroupKFold by problem, 5 folds; train and eval never share a problem)

| model | target | n rows | held-out top-quartile mean delta [95% CI] | bottom-quartile [95% CI] | top-bottom gap [95% CI] | seeds top-q means | held-out spearman (5 seeds) |
|---|---|---|---|---|---|---|---|
| OLS | delta | 133 | -0.0441 [-0.1179, +0.0282] (n=34) | -0.0404 [-0.1176, +0.0341] (n=34) | -0.0037 [-0.1127, +0.1051] | -0.044, -0.029, +0.048, -0.007, +0.040 | +0.08, -0.01, +0.14, +0.07, +0.11 |
| ridge(a=10) | delta | 133 | -0.0331 [-0.1098, +0.0417] (n=34) | -0.0257 [-0.1000, +0.0444] (n=34) | -0.0074 [-0.1104, +0.0982] | -0.033, -0.037, +0.059, +0.000, +0.044 | +0.05, -0.00, +0.13, +0.06, +0.11 |
| tree d3 | delta | 133 | -0.0221 [-0.1014, +0.0530] (n=34) | -0.0461 [-0.1086, +0.0208] (n=38) | +0.0240 [-0.0832, +0.1210] | -0.022, -0.028, -0.084, -0.088, +0.015 | +0.03, -0.07, -0.11, -0.07, -0.03 |
| ridge(a=10) +IG | delta | 118 | -0.0333 [-0.1116, +0.0417] (n=30) | -0.0583 [-0.1509, +0.0312] (n=30) | +0.0250 [-0.0983, +0.1427] | -0.033, -0.025, +0.017, +0.008, +0.004 | +0.08, -0.01, +0.06, +0.13, +0.10 |
| ridge(a=10) | delta_donor | 133 | -0.0662 [-0.1591, +0.0152] (n=34) | -0.0441 [-0.1214, +0.0312] (n=34) | -0.0221 [-0.1440, +0.0947] | -0.066, -0.088, -0.092, -0.051, -0.077 | -0.02, -0.10, -0.12, +0.04, -0.19 |

### A3 multiplicity check — max |t| over the univariate screen vs a permutation null

- target `delta`: observed max|t| over 19 features = **2.19**; permutation null (problem labels shuffled, 2000 draws) mean 2.09, 95th pct 2.93; family-wise p = **0.388**
- target `delta_donor`: observed max|t| over 19 features = **1.76**; permutation null (problem labels shuffled, 2000 draws) mean 2.16, 95th pct 3.07; family-wise p = **0.773**

### A3 mechanism check (post-treatment variables — diagnostic only, never selector inputs)


**delta vs re-solve length/truncation shift caused by the critique** — univariate (one feature at a time), n=133

| feature | mean | sd | beta/sd (HC3) | 95% CI | p | spearman rho |
|---|---|---|---|---|---|---|
| dchars | 144.195 | 1983.379 | -0.1141 | [-0.1640, -0.0643] | 0.000 | -0.359 |
| dtrunc | 0.029 | 0.261 | -0.1446 | [-0.2102, -0.0790] | 0.000 | -0.421 |

## A4 — sibling-plurality subset


**minority (own answer != sibling plurality)** — n=114 rows / 80 problems

| quantity | mean [cluster-bootstrap 95% CI] |
|---|---|
| p_blind | +0.3739 [+0.2956, +0.4576] |
| p_crit | +0.3498 [+0.2766, +0.4264] |
| p_donor | +0.4002 [+0.3238, +0.4830] |
| p_incontext | +0.2577 [+0.1875, +0.3365] |
| delta | -0.0241 [-0.0717, +0.0237] |
| delta_donor | -0.0504 [-0.0967, -0.0046] |
| delta_ic | -0.1162 [-0.1892, -0.0424] |
| frac delta > 0 | 0.298 |
| var(delta) | 0.06689 |

**majority (own wrong answer IS the plurality)** — n=19 rows / 13 problems

| quantity | mean [cluster-bootstrap 95% CI] |
|---|---|
| p_blind | +0.2829 [+0.1310, +0.4633] |
| p_crit | +0.2237 [+0.0789, +0.4044] |
| p_donor | +0.2697 [+0.1323, +0.4306] |
| p_incontext | +0.1908 [+0.0139, +0.4062] |
| delta | -0.0592 [-0.1187, +0.0000] |
| delta_donor | -0.0461 [-0.1184, +0.0209] |
| delta_ic | -0.0921 [-0.2322, +0.0375] |
| frac delta > 0 | 0.158 |
| var(delta) | 0.02668 |

**A4 univariate screen inside the minority subset** — univariate (one feature at a time), n=114

| feature | mean | sd | beta/sd (HC3) | 95% CI | p | spearman rho |
|---|---|---|---|---|---|---|
| sol_has_meta | 0.342 | 0.474 | +0.0498 | [+0.0017, +0.0978] | 0.045 | +0.218 |
| n_numbers | 7.430 | 6.702 | -0.0495 | [-0.0977, -0.0013] | 0.047 | -0.102 |
| generic_rate | 2.420 | 1.173 | +0.0305 | [-0.0143, +0.0753] | 0.185 | +0.082 |
| n_sol_tok_log | 8.176 | 0.645 | +0.0273 | [-0.0164, +0.0710] | 0.224 | +0.141 |
| jac_problem | 0.066 | 0.046 | +0.0314 | [-0.0196, +0.0823] | 0.230 | +0.127 |
| n_shared_quantities | 4.658 | 5.273 | -0.0224 | [-0.0615, +0.0167] | 0.265 | -0.056 |
| n_words | 122.272 | 21.570 | +0.0179 | [-0.0192, +0.0550] | 0.347 | -0.005 |
| recheck_same_n | 1.281 | 1.232 | +0.0224 | [-0.0243, +0.0691] | 0.349 | +0.106 |
| alt_approach | 0.228 | 0.420 | -0.0130 | [-0.0572, +0.0311] | 0.564 | -0.094 |
| self_affirm | 0.202 | 0.401 | +0.0094 | [-0.0356, +0.0544] | 0.683 | +0.021 |
| plur_share | 0.650 | 0.182 | -0.0094 | [-0.0547, +0.0358] | 0.683 | +0.011 |
| n_step_refs | 0.833 | 1.376 | +0.0058 | [-0.0272, +0.0389] | 0.729 | +0.027 |
| hedge_rate | 1.184 | 0.871 | -0.0072 | [-0.0572, +0.0428] | 0.778 | -0.030 |
| words_per_sent | 43.281 | 20.292 | -0.0057 | [-0.0544, +0.0430] | 0.819 | -0.066 |
| certain_rate | 3.183 | 1.254 | +0.0040 | [-0.0395, +0.0474] | 0.859 | +0.025 |
| n_math_spans | 5.254 | 4.725 | -0.0029 | [-0.0520, +0.0462] | 0.909 | +0.015 |
| jac_solution | 0.133 | 0.035 | +0.0023 | [-0.0433, +0.0480] | 0.920 | -0.033 |
| specific_rate | 0.341 | 0.505 | +0.0001 | [-0.0395, +0.0396] | 0.997 | -0.085 |

Held-out selector inside the minority subset (ridge a=10, GroupKFold 5): top-quartile delta +0.0000 [-0.1055, +0.1072] (n=29), bottom -0.0259 [-0.1169, +0.0670] (n=29), gap +0.0259 [-0.1157, +0.1691].


---

### A3b — can pre-treatment features predict the truncation shift (the one real channel)?

| feature | beta/sd on dtrunc (HC3) | 95% CI | p | spearman |
|---|---|---|---|---|
| hedge_rate | +0.0590 | [+0.0142, +0.1038] | 0.011 | +0.266 |
| n_words | -0.0401 | [-0.0732, -0.0070] | 0.019 | -0.145 |
| n_shared_quantities | +0.0394 | [+0.0004, +0.0785] | 0.050 | +0.097 |
| n_numbers | +0.0430 | [-0.0037, +0.0898] | 0.073 | +0.124 |
| n_sol_tok_log | -0.0223 | [-0.0589, +0.0142] | 0.233 | -0.120 |
| specific_rate | -0.0216 | [-0.0598, +0.0166] | 0.270 | -0.036 |
| is_minority | -0.0150 | [-0.0482, +0.0182] | 0.379 | -0.079 |
| alt_approach | +0.0203 | [-0.0250, +0.0655] | 0.381 | +0.035 |

Full-model in-sample R^2 for dtrunc = 0.144 (n=133, k=19).
Baseline row-level correlates instead: dtrunc vs solution length rho=-0.120, vs blind truncation rate rho=-0.414.


The single most interesting lead in the whole analysis: **hedging language in the critique predicts
that the re-solve gets longer and truncates** (beta +0.059 per sd on `dtrunc`, rho +0.266, nominal
p=0.011). It does not survive multiplicity (max |t| 2.57 against a 95th-percentile permutation null
of ~2.9), and it points at a budget effect rather than a reasoning effect, but it is the only
feature-to-mechanism link with any signal in it.

---


- rollouts scanned: 6400 (800 problems, K=8)
- rollouts containing a `<meta>…</meta>` block: 1940 (30.3%) over 653 problems
- accuracy with meta 0.736 vs without 0.657 (observational, confounded by problem difficulty)
- **mixed problems** used for the within-problem AUC (>=2 meta rollouts, both outcomes present): **38** problems, 145 rows

| meta-text feature | problems with AUC defined | mean within-problem AUC [bootstrap 95% CI over problems] |
|---|---|---|
| certain_rate ** | 20 | 0.664 [0.549, 0.780] ** |
| generic_rate | 21 | 0.370 [0.225, 0.523] |
| meta_rel_pos | 38 | 0.625 [0.492, 0.756] |
| confidence | 24 | 0.607 [0.466, 0.747] |
| self_affirm | 20 | 0.604 [0.467, 0.730] |
| log_n_tok | 38 | 0.604 [0.461, 0.740] |
| words_per_sent | 37 | 0.401 [0.282, 0.528] |
| n_words | 37 | 0.425 [0.300, 0.551] |
| specific_rate | 16 | 0.553 [0.403, 0.694] |
| jac_solution | 38 | 0.450 [0.321, 0.575] |
| n_numbers | 34 | 0.452 [0.328, 0.578] |
| jac_problem | 36 | 0.547 [0.413, 0.676] |
| n_step_refs | 24 | 0.463 [0.331, 0.600] |
| n_math_spans | 23 | 0.469 [0.317, 0.621] |
| n_sentences | 36 | 0.521 [0.391, 0.650] |
| recheck_same_n | 30 | 0.482 [0.351, 0.609] |
| meta_chars | 38 | 0.486 [0.357, 0.612] |
| hedge_rate | 15 | 0.490 [0.300, 0.667] |
| n_shared_quantities | 26 | 0.502 [0.353, 0.652] |

Bonferroni threshold for 19 features at family alpha 0.05: a single feature needs its 99.8% CI to exclude 0.5.

**Multiplicity check**: observed max |AUC-0.5| over 19 features = 0.164; within-problem label-permutation null (1000 draws) mean 0.152, 95th pct 0.219; family-wise p = **0.331**


### A5b — within-problem value of *emitting* a spontaneous meta block

- problems with both outcomes and both meta/no-meta rollouts present: **121** (of 800); rows in them: 968
- mean within-problem AUC of `has_meta` for correctness: **0.523** [0.485, 0.559] (n=121 problems)
- mean within-problem accuracy difference (meta rollouts minus non-meta rollouts): **+0.0684** [+0.0042, +0.1347]


A5 is badly underpowered by construction: only 30.3% of rollouts emit a spontaneous `<meta>` block
and the model solves 73.6% of the meta-carrying rollouts, so only **38 problems** have two or more
meta rollouts with mixed outcomes (145 rows). `certain_rate` reaches AUC 0.664 [0.549, 0.780] but
the family-wise permutation p is 0.331 — exactly what one expects from screening 19 features over
38 problems. A5b, which has 121 problems of support, finds emitting meta at all worth
**+0.068 [+0.004, +0.135]** accuracy within problem (AUC 0.523 [0.485, 0.559]) — a barely-positive
selection effect, not a feature of the meta text.

---

### Verbatim examples (critique truncated to ~40 words)

**4 highest-delta rows (own critique helped most)**

- `g43#351` delta=+0.750 (p_blind 0.250 -> p_crit 1.000, donor 0.875); critique 134 words, dtrunc=-0.625
  > The most likely mistake is in assuming that having at least one positive real root in $ y $ guarantees at least two distinct real $ x $-solutions, without properly accounting for multiplicity or the case when the positive root is zero. …
- `g223#1785` delta=+0.625 (p_blind 0.375 -> p_crit 1.000, donor 0.750); critique 141 words, dtrunc=-0.500
  > The critical mistake lies in assuming that the offset along the curb (15 feet) directly projects onto the normal direction via $ \cos\theta $, without verifying whether the angle $ \theta $ is correctly defined relative to the direction of the offset. …
- `g122#983` delta=+0.500 (p_blind 0.250 -> p_crit 0.750, donor 0.250); critique 163 words, dtrunc=-0.750
  > The critical mistake lies in assuming that the folding brings the three original vertices to a single point directly above the midpoint triangle with height equal to the $ z $-coordinate of that point. This assumes the base lies in a plane …
- `g5#46` delta=+0.500 (p_blind 0.000 -> p_crit 0.500, donor 0.500); critique 134 words, dtrunc=-0.125
  > The most likely mistake is in assuming that the smallest $ x $ occurs when $ k $ is maximized in the scaling process, without properly verifying that the corresponding $ x $ lies within the domain where $ f $ is …

**4 lowest-delta rows (own critique hurt most)**

- `g760#6082` delta=-0.500 (p_blind 0.500 -> p_crit 0.000, donor 0.250); critique 114 words, dtrunc=+0.125
  > The most likely mistake is in the assumption that the transformation $ (a+1)(b+1) \equiv 2 \pmod{5} $ correctly captures the condition $ ab + a + b \equiv 1 \pmod{5} $. While algebraically valid, this step must be verified carefully: the original …
- `g549#4394` delta=-0.750 (p_blind 0.750 -> p_crit 0.000, donor 0.625); critique 112 words, dtrunc=+0.250
  > The critical mistake lies in assuming that $ n $ must be of the form $ 3^a \cdot 5^b \cdot m^2 $ with $ a $ and $ b $ odd, and only considering the primes 3 and 5. While this captures …
- `g637#5096` delta=-0.750 (p_blind 1.000 -> p_crit 0.250, donor 0.875); critique 123 words, dtrunc=+1.000
  > The most likely mistaken assumption is that the minor axis of the ellipse is equal to the diameter of the cylinder (2). This is incorrect when the plane is tilted — the minor axis is actually the projection of the cylinder's diameter …
- `g337#2703` delta=-1.000 (p_blind 1.000 -> p_crit 0.000, donor 0.750); critique 117 words, dtrunc=+1.000
  > The most likely mistake is in the numerical verification step: assuming that the identity $3 \sqrt{\sqrt[3]{5} - \sqrt[3]{4}} = \sqrt[3]{25} + \sqrt[3]{20} - \sqrt[3]{10}$ holds without proper validation. This fails numerically, as the right-hand side is approximately 3.48 while the left-hand side …


Note the pattern in the eight examples: all four winners have `dtrunc <= 0` (the critique made the
re-solve *finish*), all four losers have `dtrunc >= +0.125` (it made the re-solve run out of budget).
Reading the texts, the winners and losers are stylistically the same object — "The most likely
mistake is in assuming X ... A correct approach must verify Y". Two of the four losers
(`g760#6082`, `g337#2703`) even contain correct, specific diagnoses; they still cost the row
0.5-1.0 of accuracy. `g337#2703` is the clearest: the critique correctly computes that 3.48 != 1.05,
and the row goes 1.000 -> 0.000.

---

# Closing statement

**No. On this data the zero average is genuinely uniform.**

There is excess spread in the per-row effect (2.17x binomial), so the average of ~0 is not by itself
proof of homogeneity — but every bit of that excess lives in the truncation channel, and once
truncation is held fixed the spread is at the K=8 coin-flip floor (ratio 0.73, p=0.86) around a mean
of +0.006. Of 19 interpretable features of the critique text (length, specificity, step/quantity
citation, hedging, certainty, generic-advice lexicon, alternative-approach proposal, problem/solution
overlap) plus the two G5 information-gain numbers, none survives multiplicity correction
(family-wise p = 0.39 for own-minus-blind, 0.77 for own-minus-donor), and **no selector generalises
across problems**: five model/target combinations, GroupKFold by problem, five seeds each — every
held-out top-quartile mean delta is negative or zero with a CI containing 0, and every
top-minus-bottom gap CI contains 0 (best gap +0.025 [-0.098, +0.143]). The same holds inside the
sibling-minority subset (A4: delta -0.024 [-0.072, +0.024], n=114; held-out selector gap
-0.043 [-0.165, +0.092]) and inside the spontaneous-meta corpus (A5, family-wise p = 0.33).

The practical consequence for the north star: **"reward the metacognitive act when it helps" has no
per-row target to learn from here.** A reward that amplifies any of these text features would be
fitting K=8 sampling noise plus a token-budget artifact. Two things would have to change before the
question is worth asking again: (a) a re-solve budget that does not truncate — 54% truncation makes
the outcome a length race, and the in-context retry arm at 6.6% truncation shows the budget is the
binding constraint; (b) far more rows and/or a larger K — with n=133 and K=8 the per-row effect is
measured to +/-0.17 (the binomial-null sd of delta), so only a selector worth >= 0.15 accuracy would have been visible at all.
This analysis can rule out a large exploitable subset; it cannot rule out a small one.

## Caveats

- All 133 rows are already-wrong Level-5 rollouts from a single policy (Qwen3-4B-Instruct-2507,
  `opt` prompt), seed s1, single run. Nothing here speaks to critiques of *correct* solutions.
- `dtrunc`, `dchars` and the completion-conditional rates are post-treatment; they are reported as a
  decomposition of where the variance sits, never as selector inputs, and no selector in the tables
  above uses them.
- The leak filter is upstream: 36 of 169 critiques leaked the answer and were dropped before
  `per_rollout.jsonl`, so the analysed set is leak-free but also selected.
- The 19-feature lexicon was built by reading 30 sampled critiques before any outcome was inspected;
  the multiplicity tests treat the whole family as one screen.
