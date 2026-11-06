# en/: Two Teachers in the Middle: Self-Distilling Metacognitive Language Habits Mid-Solution

English LaTeX draft in the **ICML 2026 submission format** (anonymous, two columns, `icml2026.sty`; style files copied from
`/hdd_data/seungpil/scratch/icml2026_style/`). Build (no errors, no undefined references or citations; output kept off the repo):
`cd paper/en && /hdd_data/seungpil/texenv/bin/tectonic -o /hdd_data/seungpil/tmp/paper_build/out main.tex && cp /hdd_data/seungpil/tmp/paper_build/out/main.pdf build_main.pdf`.
Remaining warnings: underfull boxes only (no overfull boxes, 2026-10-12 final polish).

Layout (final integration pass, 2026-10-12 evening):
- Main text = 8 pages (pages 1-8; Impact Statement, references from page 9, appendix after). Page allocation:
  intro ~1.4 p (incl. Fig. 1 overview and Fig. 2 oracle), related ~0.6 p, method + theory ~2.3 p (incl. Fig. 3 scoring),
  experiments ~2.9 p (Tab. 1 main, Tab. 2 transitions, Fig. 4 Four Habits), limitations + conclusion ~0.4 p.
  Before this pass (11pt article, 1-inch margins): intro 5 p, related 3 p, method 6 p, experiments 17 p, discussion 2 p, conclusion < 1 p (34 body pages).
- `main.tex`: ICML preamble, `\todo{}` = red inline text (`\todo{}` with an empty argument = a small red "TODO" table cell),
  `\figplaceholder{}`, `\Tblind`/`\Tkeep` = T1/T2, column type `P{w}`, theorem environments `proposition`, `assumption` (A1-A4).
- Main sections: `sections/{abstract,intro,related,method,experiments,discussion,conclusion}.tex`
  (method holds Sec. 3.5 "What the two teachers optimize": Propositions 1-5, assumptions A1-A4; `discussion.tex` is "Limitations").
- Appendix (`sections/appendix.tex`, one column): A extended related work (`related_full.tex`), B derivations (`app_theory.tex`),
  C experimental details incl. the one-step pipeline figure, arms and compute tables (`app_setup.tex`),
  D/E/F additional results for H1/H3/H2 (`app_results.tex`).
- Main-text floats: `fig:overview` (figure*, concept), `fig:oracle`, `fig:scoring` (figure*, method), `tab:main` (table*, incl. MMLU-Pro column),
  `tab:quad`, `fig:habits` (figure*, full \textwidth). Everything else (`fig:forest`, `fig:behavior`, `fig:signal`, `fig:pipeline`, `fig:teaser`, `fig:main`,
  `fig:testtime`, `fig:budget`, `fig:passk`, `fig:markers`, `fig:tokens`, `fig:shift`, `fig:inject`, `fig:behavior_main`, `fig:oneteacher`,
  `fig:dynamics`, `fig:dynamics_main`, `fig:teachers`, `fig:ablation`, `tab:note`, `tab:arms`, `tab:compute`, `tab:prelim`, `tab:ood`,
  `tab:testtime`, `tab:behavior`, `tab:metaood`, `tab:oneteacher`, `tab:ablation`, `tab:teachers`, `tab:trunc`) is in the appendix.
- Figure files are used by name; do not edit `figures/*.py` from the paper side. `fig_overview.pdf` and `fig_scoring.pdf` are the split
  concept figure (2026-10-12 19:34).

Theory check (2026-10-12): every derivation in `app_theory.tex` was re-derived. Fixed: Proposition 5(i) (outcome routing) was wrong;
the outcome-gated update is not the gradient of the gated KL, so the "hidden outcome reward of weight beta" claim was replaced by the
correct stationary point (token-dependent teacher weight beta g(y)/(lambda + beta g(y)), Eq. app-outfix); the "a re-solve that fails still
receives T1's credit" contrast (true for outcome routing as well) was replaced by "every continuation of a wrong first answer receives
T1's pull, whether it ends right or wrong"; the equality case of D_delta >= 0 was made rigorous; J_out now states that the mask, break
weight and repetition penalty are folded in; every numbered equation is referenced.

Writing rules applied (paper-section-rewrite skill; full 4-level loop log in `REWRITE_LOG.md`): insight-first paragraph leads, one sentence per line, main-text sections
>= 70% sentences of <= 20 words (method .79, experiments .70, intro .70, related .80), em-dash 0, accuracies in % and differences in points.

## Figures (one per finding)

Regenerate every figure (PDF, into `figures/`):

```
cd paper/en/figures
/home/ubuntu/miniforge3/bin/python3 make_figs.py                 # draw all (matplotlib); add figure names to draw only those
/hdd_data/seungpil/envs/verl09/bin/python make_figs.py --refresh  # recompute data_hdev.json and results_a82 (math_verify), ~10 min CPU
```

Code: `make_figs.py` (entry, figure list) -> `figs_data.py` (every data source; refresh; training-log parser; pass@k),
`figs_plot.py` (style, arm palette, chart helpers), `figs_method.py` (overview, token credit, oracle, teacher signal,
failed teachers), `figs_results.py` (teaser, accuracy, pass@k, behaviour, markers, training dynamics, ablations).
Style: 9 pt text at final size (>= 7.5 pt for dense labels), bold "(a) Question?" panel titles, light y grid, no top/right
spines, one colour per arm in every figure, direct labels. `save()` reads the real PDF width back and flags anything
wider than 6.5 in. Every figure was rendered to PNG and checked for overlaps and clipping (2026-10-12).

Run `--refresh` and then the draw command once main-protocol evaluations land in
`/hdd_data/seungpil/scratch/eval/rl_<arm>_s<seed>_<bench>/` (arms base, grpo, grpokl, reset, ours, rlsd, rlrt;
benches aime24, aime25, hmmt25, math500, hdev, mmlupro). Missing cells are drawn as hatched "pending" slots and
fill in automatically. Main-protocol training curves (`fig_dynamics_main`) appear as soon as
`logs/mc_SPONT_PFX_*_dapo*.log` exist (lineage names are parsed by `figs_data.lineage`).
Data: `data_hdev.json` (a64 logic: per-problem accuracy, wrong-first transitions, right-first keep/repeat, budget curves),
`data_ledger.json` (numbers that exist only in `docs/RESULTS_cd9.md`, each with `src`), and in `u_checks_0925/`:
`results_a39_habits.json`, `results_a68_entropy.json`, `results_a74_ttreset.json`,
`results_a81_token_credit.json` (token credit, GPU; `_cputest.json` until then), `results_a82_text_stats.json`
(per-problem correct counts for pass@k, discourse markers after the first box); plus `eval/rl_*/summary.json` and training logs.
Palette (fixed per arm): base grey, GRPO blue, ours vermillion, reset-only orange, tether-only light orange,
RLSD reddish purple, RLRT bluish green, uniform olive, test-time reset sky blue (Okabe-Ito based).
Marker categories: re-solve purple triangle, verify green square, commit brown circle, hesitation magenta diamond.

Token-credit run (done on one GPU, 2 pairs; the 78-pair paired test `a81_all_pairs` is queued):
`cd /hdd_data/seungpil/scratch/analysis/u_checks_0925 && CUDA_VISIBLE_DEVICES=3 /hdd_data/seungpil/envs/verl09/bin/python a81_token_credit.py --device cuda`
(2 pairs = 4 rollouts, ~5-10 min with 2-3 min of CPU grading first, < 40 GB; writes `results_a81_token_credit.json`), then redraw `fig_tokens`.

| Fig. | Label | Finding | Data source | Status |
|---|---|---|---|---|
| 1 | `fig:teaser` | (a) hdev lenient/strict/first box, bars = mean of seeds 2/3/5, ours - GRPO +2.9/+8.8/+0.9 (three seeds); (b) after a wrong first answer: no second answer 25/22/11%, re-solve right kept 7.7/5.7/12.0%; (c) main-protocol math avg@16 | `data_hdev.json` + ledger `ours_minus_grpo`; `eval/rl_*` | (a)(b) final, (c) pending |
| 1 | `fig:overview` | Concept: the habit we want (re-check vs anchor on first answer 56) and who reads what (student, blindfolded T1, yesterday's-self T2); test-time strip | schematic | final (redrawn 10-12) |
| 3 | `fig:scoring` (Sec. 3) | How each word is scored: illustrative probability bars student vs T1, word credit = log teacher - log student, wrong-first only; T2 KL on every word | schematic | final (10-12) |
| app. C | `fig:pipeline` | One training step (verifier checks X, 8 continuations, GRPO, T1 credit, T2 KL) | schematic | final |
| 3 | `fig:oracle` | An oracle "likely wrong / likely correct" note raises fix 2.6% -> 21.6%, break 0.3% -> 0.1%: noticing is the bottleneck | `results_a39_habits.json` (base vs base_hint, probe 350x16) | final |
| app. E | `fig:tokens` | Token-level T1 credit d_t on two real rollouts (pid 232, X = 56, gold 32): base re-confirms, ours s5 re-solves; mean d_t -.49 vs -.45 and (pid 241) -.50 vs -.03 | `results_a81_token_credit.json` (a81 GPU run, ledger 10-12 11:xx) | final (24 scorable pairs: -.771 vs -.253, +.518 [+.338, +.704]) |
| 5 | `fig:signal` | A teacher that sees the student's text gives ~0 signal (-0.007) vs blind T1 (-0.25) | ledger 10-08 19:xx | final |
| 6 | `fig:main` | Main-protocol accuracy in domain + MMLU-Pro OOD, 7 arms | `eval/rl_*/summary.json` | pending |
| 7 | `fig:forest` | Ours - GRPO, three seeds: hdev +2.9 lenient, +8.8 strict, first box +0.9 (CI includes 0); MATH L5 +2.0; MATH-500 +0.7; AIME/HMMT +1.8 (CI includes 0); two-seed rows: fix, non-systematic fix, AIME/HMMT strict; main-protocol + OOD rows | ledger 2026-10-12 11:xx (three seeds) and 10-09 (two seeds) + `rl_*` | preliminary final, main pending |
| 8 | `fig:testtime` | Ours ~ test-time reset (+2.1, CI includes 0) with fewer tokens; maj@3/4 costlier | `results_a74_ttreset.json` | final |
| 9 | `fig:budget` | Gain needs >= 16k tokens; base best at 8k; thinking-off model flat | `data_hdev.json` budget (3 seeds) + ledger `budget_extra` | final (main-protocol curves pending) |
| 10 | `fig:passk` | pass@k on hdev: ours leads GRPO at k=1 (65.4 vs 62.5 lenient), level at k=8 (92.6 vs 93.1); strict ahead at all k (90.3 vs 85.6 at k=8); base highest at k=8 (94.8/93.8, selection-biased) | `results_a82_text_stats.json` (per-problem correct counts, seeds 2/3/5); main from `rl_*/summary.json` | (a)(b) final, (c)(d) pending |
| 11 | `fig:behavior` | After wrong first answer: no-reopen 25/22/11%, re-solve-keep 7.7/5.7/12.0%; keep-right 93/88/93%; repeat 21/25/33% (cost); entropy AUC .586/.595/.649 | `data_hdev.json` (base, GRPO s3, ours s3) + `results_a68_entropy.json` | final |
| 12 | `fig:markers` | After a wrong first answer GRPO commits ("is correct" 6.6 vs 1.6 per response, "so the answer is" .91 vs .27) and ours re-opens ("but wait" .39 vs .20, "alternatively" .63 vs .40, "go back" .15 vs .08); same direction after right first answers | `results_a82_text_stats.json` (seeds 2/3/5 pooled, bootstrap over problems) | final |
| 13 | `fig:behavior_main` | Same metacognition metrics, main protocol hdev + MMLU-Pro | `data_hdev.json` rl_* entries after `--refresh` | pending |
| 14 | `fig:oneteacher` | (a) ours - one-teacher arms (pending); (b) tethered runs finish, reset-only runs stop early (guards differ) | `rl_*` + ledger training runs | (a) pending, (b) final |
| 15 | `fig:dynamics` | Training curves (GRPO, uniform, reset only, ours; steps 1-20): fix rate on wrong-first prefixes .2 -> .40-.56 by step 5 in all prefix arms; KL at step 5 .092 reset only vs .073 ours; per-step break rate does not separate them; T1 share 7-14% of advantage mass; GRPO entropy .21 vs ours .31 at step 20 | `logs/mc_SPONT_PFX_gold_*_hc2*.log` (verl + [MC] step lines) | final |
| 16 | `fig:ablation` | Placebo / uniform / on-policy reset differences | ledger 10-06, 10-07, 10-10 | final |
| 17 | `fig:teachers` | Nine failed teachers by failure mode: sees text -> no signal; blind -> wrong direction or breaks right answers | ledger 10-07, 10-08 | final |
| 18 (app.) | `fig:dynamics_main` | Main-protocol training curves, same panels as Fig. 15 | `logs/*_dapo*.log` | pending |
| 4 | `fig:habits` (Sec. 4.2) | Four Habits after the first answer rise on wrong- and right-first rows alike (ours - GRPO, >= 1 habit: W verification +9.4, backtracking +8.3; R +17.1, +15.5); H3 partly supported; habit -> fix .236 vs .063 (backtracking) | `results_a83_four_habits.json` (judge Haiku 5.5, 69,188 calls; ICC vs Opus 5.5 on 358) | final |
| new | `fig:inject` (Sec. 4.2) | Reflection injection ("Wait, let me reconsider.") after the first box vs a random point; flip->R / flip->W for base, GRPO, ours; hard/easy | `results_a85_inject.json` (`a85_inject.py`) | pending (`\figplaceholder`) |
| app. E | `fig:shift` | Where training changed the policy: share after the first box +0.6 [-0.7, +1.8] (criterion not met); own texts +7.3 nats/1k after | `results_a86_shift.json` | final |
| table | `tab:quad` (Sec. 4.1) | SCoRe-style transitions on hdev, seeds 2/3/5 pooled: first box 58.8/59.8/60.7, last 62.1/62.5/65.4, W->R 5.6/5.1/6.9, R->W 2.4/2.3/2.3 (base/GRPO/ours) | `figures/data_quadrant.json` (`figures/data_quadrant.py`, a64 grader on the same eval dirs as `data_hdev.json`; reproduces the ledger's 62.5/65.4, 54.5/63.3, 59.8/60.7) | preliminary final, main pending |
| table (app. F) | `tab:note` | Masked note: same-X return -14.5 [-19.8, -9.3], fix non-systematic +4.3 [-3.1, +11.6], keep R -14.7 [-19.8, -10.1] -> failed; "another student" +1.7 [-2.9, +6.4] -> failed | `eval/fresh_stage0_mask`, `eval/fresh_stage0_other` | final |


## The claim (fixed by the user, 2026-10-12)

Intent: «메타인지 언어 습관을 자동으로 강화하는 멀티 셀프 디스틸레이션» = automatically strengthening a metacognitive
language habit (after the first \boxed{X}: say "Wait", re-check, fix wrong answers, keep right ones) with multi-teacher
self-distillation. "Automatically" = no human-written metacognition data, no reward on specific phrases, no external teacher;
the only supervision is the outcome reward. Test time is one shot, with no prompt asking to verify.

- T1 = blindfolded self (problem + "a previous attempt answered X", reasoning hidden): re-solve instead of anchoring, wrong-first rows only.
- T2 = yesterday's self (frozen reference, full context, KL tether .05): keep what is right, all rows.
- H1 accuracy in and out of domain (MMLU-Pro OOD); H2 both teachers needed; H3 the change is a real language habit
  (Four Habits: verification/backtracking rise after the first answer, selectively on wrong-first rows).
- Section map: 1 intro (teaser, overview, oracle); 2 related work (paragraphs open with our position, no bold leaders);
  3 method (training state + "automatic" paragraph, T1 with the context-effect/drift split of d_t, T2, combined update,
  teacher-context principle); 4 experiments: Setup (protocols, metrics, `tab:arms`), 4.1 H1 (`tab:main`, `tab:ood`,
  `fig:forest`, `tab:quad`, `fig:testtime`, `tab:compute`, `fig:budget`, `fig:passk`), 4.2 H3 (`fig:habits`,
  `fig:behavior`, `fig:markers`, entropy, `fig:tokens`, `fig:inject`, `fig:shift`, `fig:behavior_main`), 4.3 H2 and
  ablations (`fig:oneteacher`, `fig:dynamics`, `tab:note`, `fig:teachers`, `fig:ablation`, `tab:trunc`, beta, training length);
  5 discussion (thesis-led paragraphs); 6 conclusion; appendix tables.

## Where the claim is not yet supported (keep in sync with the text)

1. **H1 main protocol and OOD**: all pending (60 steps x 64 problems, GRPO and ours continued to 100; amendments 70-72b).
   MMLU-Pro criterion (70c(3)): lower bound of ours - GRPO > -1 point = "works OOD without loss"; > 0 = "OOD gain".
   Preliminary three-seed AIME/HMMT gain +1.8 [-0.4, +4.3] includes 0 (two seeds: +2.4 [+0.1, +4.7]).
2. **H2 both teachers needed**: preliminary support is training stability only (guards differ; ours s4 also stopped).
   Decided by main-protocol `reset` and `grpokl` arms against the 70c(1) criterion. The reset-only arm keeps the drift term
   of the blind credit (amendment 72b "hidden tether"), so it is not fully tether-free.
3. **H3 language habit**: partly supported (Four Habits, amendment 73): verification and backtracking after the first answer rise
   (lower bounds > 0), but not selectively (they rise more on right-first rows) and the control habits rise too. Selectivity shows only in
   outcomes (W->R up, R->W unchanged). Reflection injection (a85) and the main-protocol / MMLU-Pro habit metrics (`tab:metaood`) are pending.
4. **Note form**: masked note and "another student" note both failed their pre-registered gates (amendment 71); the current note is kept.

Numbers in the text are final preliminary-protocol numbers, numbers copied from `docs/RESULTS_cd9.md`,
`docs/PREREGISTRATION_cd9_math_judgment.md` and `eval/*/summary.json` (listed below), or recomputed from eval files by
the figure package (`data_hdev.json`, `data_quadrant.json`). Everything else is a `\todo{}`.

Numbers added in the 2026-10-12 audit:
- Three seeds (ledger 2026-10-12 11:xx): hdev lenient +2.9 [+0.7, +4.8] (62.5 -> 65.4), strict +8.8 [+6.6, +10.9]
  (54.5 -> 63.3), first box +0.9 [-1.0, +2.9] (59.8 -> 60.7); MATH L5 (800 x 4) +2.0 [+1.4, +2.7] (GRPO 87.7-89.1,
  ours 89.8-90.6, base 88.3); MATH-500 +0.7 [+0.1, +1.3]; AIME24/25 + HMMT25 +1.8 [-0.4, +4.3].
- Transition table `tab:quad` (new file `figures/data_quadrant.json`).
- Masked-note gate (`eval/fresh_stage0_mask/summary.json`, `vs_fact`): same-X return non-systematic -14.5 [-19.8, -9.3],
  fix non-systematic +4.3 [-3.1, +11.6], keep R -14.7 [-19.8, -10.1]; systematic wrong first answers 336 of 948 (`sys_rows`).
- Teacher anchors on the note: Y == X | W .496 vs .31 for independent samples (amendment 71).
- Multiplicative contrast teacher (chac4hc2): repetition .522 (steps 11-15) and .554 (16-20) > loop check .5 (ledger 10-04 11:4x).
- Truncation rules (earlier rounds): keep 5.6 -> 24.7% right-first truncation, 78-92% of truncated right-first rows rewarded,
  guard stop at step 19 (amendment 32); zero: meta after right/wrong -.044/-.040, probe fix -.028 [-.047, -.009], 12k accuracy
  -.0061 [-.0134, +.0013] (ledger 0927 04:3x); mask: 30.6% of hdev responses at 32k vs base 4.2% (0930 05:3x); loop: -4.37
  [-7.44, -1.34] vs base, probe fix .205 vs .340 (0930 13:4x).
- Main-protocol arms: GRPO+tether trains on the fixed prefixes (`a78_rlrt_train.sh`: grpokl = prefixes + KL .05);
  SDPO-style arm (amendment 72); STaR (D_STaR+) and Pair-SFT offline baselines (`a84_sft_data.py` docstring: 120 updates x 64
  sequences, LR 1e-5, STaR 3 iterations); 60 steps x 64 problems = 3,840 (about 15% of RLRT), GRPO and ours to 100 (72b).
- Earlier revision numbers (reset-only, beta .25, uniform, placebo, all-rows reset, allocation, GRPO KL .002, hdev
  selection) are unchanged; see the ledger sections cited in `figures/data_ledger.json`.

## References (updated 2026-10-12)

- `refs.bib` has 123 entries (95 + 6 added on 2026-10-12: `welleck2022selfcorrect`, `zelikman2022star`,
  `singh2023restem`, `chen2021codex`, `hinton2015distilling`, `liu2021amtml`, all from the arXiv API record), grouped as in Section 2: RLVR and credit assignment,
  self-distillation with privileged context, self-correction and anchoring, metacognitive signals and
  calibration, overthinking and stopping, exploration and entropy, plus models and benchmarks.
- Novelty-audit pass (2026-10-12, `docs/survey_novelty_1012.md`): 22 entries appended at the end of `refs.bib`
  (`wang2026re2`, `maniparambil2026critique`, `cai2026h2sd`, `peng2026adopsd`, `li2026cast`, `yue2026sipo`, `pan2026rlcsd`,
  `yang2026oglssd`, `heakl2026cepo`, `tian2026whatsee`, `kara2026feedbackalign`, `li2026hsd`, `tian2025thinktwice`,
  `kim2026failureprefix`, `amjith2025flawed`, `xu2026denoiserl`, `zhao2026twostage`, `wei2026dre`, `li2016lwf`,
  `tang2026reteach`, `wang2026refguidance`, `jeon2026upturn`), each from the arXiv API record. Claims changed with it:
  no "first multi-teacher" claim; T1 is a teacher whose context is a strict subset of the student's; credit is confined to
  the state after the first answer; routing uses the fixed first answer; T2 is the standard reference KL read as a teacher
  (LwF / SCoRe / AD-OPSD family); the teacher-context principle is a necessary condition only (cf. `tian2026whatsee`).
- Sources: `docs/SURVEY_when_meta_helps_0914.md`, `SURVEY_own_sample_diagnosis_0914.md`,
  `SURVEY_reward_mechanics_0915.md`, `SURVEY_revision_credit_0919.md`, `SURVEY_meta_habit_selfdistill_0920.md`,
  `SURVEY_credit_density_0923.md`, `survey_selfdistill_teacher_context_1008.md`, the old `paper/refs.bib`,
  and the RLRT reference list.
- Verification: every entry's arXiv id, title, author list and year were taken from the arXiv API record
  (`export.arxiv.org`, 2026-10-12). No author was typed by hand. Nothing is flagged `\todo{verify}`.
- "and others": author lists longer than 15 names are cut to the first 10 names plus `others`
  (DAPO, PRIME, Askell 2021, Self-Refine, SCoRe, Kadavath 2022, TTRL, entropy mechanism, 80/20, Qwen3,
  MMLU-Pro). DeepSeek-R1 uses the corporate author `{DeepSeek-AI}` as on arXiv. Expand before submission
  if the venue requires full lists.
- Corrected against the API: Self-Correction Bench title ("Uncovering", "in Large Language Models"),
  Self-Correction Illusion title (adds "in Large Language Models"), first names for Kaur (Simran) and the
  previously "and others" entries (SDPO, RLSD, SRPO, Kim 2026, Baumann, SD-Zero, LEMMA, etc.).
- Acronym collision: three different papers are called SRPO. `li2026srpo` (2604.02288, sample routing) is
  the one the text calls SRPO; `samanta2026resets` (2605.25507, Self-Reset PO) and `liu2026srporeflect`
  (2608.23493, Self-Reflective PO) are named only by description. Keep it that way.
- Content notes from the surveys that the text relies on: 2602.08503 (the "correct after answering wrong" hack)
  is a vision-language paper; 2603.22293 (TIPS) is about search agents; Kaur 2026 reports a gain, not a loss,
  for Qwen3-4B-Instruct-2507 under privileged context; RLSD is below plain GRPO in RLRT's and AntiSD's tables.
- `related.tex` cites our earlier probe numbers (within-problem verbalized-confidence AUC .50, sibling-majority
  AUC .78, outcome-only GRPO revision rate 2.91% to 1.70%). They come from earlier cd9 protocols
  (09-14 to 09-19) and need an appendix with those protocols (`\todo` in the text).

## TODO list, grouped by the experiment that fills it (186 `\todo` in total, 2026-10-12 final pass)

| Pending experiment | Count | Where |
|---|---|---|
| Main protocol, all arms (a78 train + a79 eval; amendments 70-72b) | 97 | `tab:main` 70 cells; experiments: runs + seed `n`, results paragraph, `tab:quad` main row; abstract, intro x2, conclusion; limitations (seeds/compute); appendix: `fig:main` caption, `tab:compute` 11 cells + GPU-h text, budget curves, pass@k (c, d), matched-token TT-reset / maj@k, markers of SDPO-style/RLSD/RLRT, `fig:dynamics_main` |
| MMLU-Pro OOD (a80 set, a79 k16) | 66 | experiments: OOD criterion paragraph; appendix: `tab:ood` 23 cells, `tab:metaood` 40 cells, results paragraph, `fig:behavior_main` caption |
| One-teacher arms `grpokl`, `reset` (70c(1)) | 15 | experiments H2 criterion; limitations verdict; appendix `tab:oneteacher` 12 cells, `tab:ablation` tether-only row |
| 100-step continuation of GRPO and ours (72b) | 2 | experiments training-length sentence; appendix `tab:ablation` last row |
| Reflection injection (a85) | 2 | experiments H3 TODO; appendix results paragraph (+ `\figplaceholder` for `fig:inject`) |
| TT-reset definition check (a74 output) | 1 | appendix D: denominator of fix .145 / break .100 |
| Prefix pipeline for the main protocol (a77_dapo.sh) | 1 | method 3.1 |
| Earlier probe protocols (cd9 09-14 to 09-19) | 1 | appendix A (internal signals paragraph) |
| Exploratory second model (69(d), on hold) | 1 | limitations |
| Authors | 0 (comment) | `main.tex`: author list (hidden in anonymous mode) |

Filled in this pass: Four Habits judge (a83), token-credit pairs (a81, 24 scorable pairs), distribution-shift localisation (a86),
"another student" note gate, 40-step comparison.
