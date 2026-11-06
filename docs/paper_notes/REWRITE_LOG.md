# REWRITE_LOG: paper-section-rewrite 4-level loop (2026-10-12, final polish)

Scope: main text of `en/` (source of truth), mirrored into `ko/`. Appendix was not rewritten; it was
covered only by the Phase 6 notation audit. Every number, `\todo{}` (texts identical, per-file counts unchanged), label and
ref was kept. Fix rates written as fractions (.236 etc.) were converted to percent (23.6%) for unit consistency.

Measures: "short" = sentence of at most 20 words (EN) or 50 characters without spaces (KO); gate is at least 60% per paragraph.
Scripts (scratch, not in the repo): `metrics.py` (per-paragraph short ratio + banned-phrase grep), `parity.py` (EN/KO numbers,
labels, refs, cites, todo per file), `fontcheck.py` (smallest figure text at printed size).

## Per section

| Section | L1 content | L2 structure | L3 paragraph flow | L4 sentences (short ratio before -> after; EN / KO) |
|---|---|---|---|---|
| Abstract | 1 round, CONVERGED | 1 round | 1 round | 2 rounds; 77% -> 94% / 85% |
| Introduction | 2 rounds | 2 rounds | 1 round | 2 rounds; 74% -> 76% (every paragraph >= 60%) / 87% |
| Related work | 1 round | 2 rounds | 1 round | 1 round; 84% -> 96% / 100% |
| Method (incl. 3.5 theory) | 1 round | 1 round | 2 rounds | 2 rounds; 80% -> 93% / 99% |
| Experiments 4.0-4.3 | 1 round | 2 rounds | 2 rounds | 2 rounds; 71% -> 86% / 93% |
| Limitations | 1 round | 2 rounds | 1 round | 1 round; 60% -> 90% / 91% |
| Conclusion | 1 round | 1 round | 1 round | 2 rounds; 0% -> 75% / 100% |

Gates passed in every main-text section: insight-first leads, R1-R5, results flow intent -> method -> result -> interpretation
(Section 4), one sentence one role, every paragraph >= 60% short, em-dash 0 (EN and KO), no bold run-in leaders, no
"moreover/furthermore/clearly/various", KO: 계사 대용 0, "~하고자 한다" 0, no 명사형 종결.

### Abstract
- L1 intent: strengthen the post-answer re-check habit by self-distillation from two frozen copies that differ only in context.
  Insight-first fix: added the cause ("the check reads the model's own reasoning, so it tends to follow it back to the same answer"),
  which is why a blindfolded teacher helps.
- L4: KL expanded on first use; result sentence split; closing line now states the core finding once (broad habit, selective outcome).

### Introduction
- L1: core insight made explicit as the lead of the proposal paragraph ("anchoring needs the visible reasoning, so a copy that
  cannot see it can teach re-solving"); it was previously implicit ("confine teacher credit...").
- L2 before: background / cause / "automatic" / insight+teachers / analysis / differentiation / results+contributions.
  R3/R4: "automatic" came before the method it qualifies; the analysis paragraph repeated contribution (2).
  After: background / cause (anchoring, oracle -> noticing) / insight + two teachers / automatic supervision / differentiation
  (lead: "our teacher knows strictly less") / results + contributions. The analysis paragraph was folded into contribution (2).
- L4: GRPO and RL expanded at first use; a 4-jargon sentence (on-policy, reverse-KL, distillation, PMI) split; "at test time
  the model answers once" removed from the body because the Figure 1 caption states it.
- Phase 5.4 caption <-> body: Figure 1 caption (8 sentences) repeated the teacher descriptions of the body; cut to 4 sentences.
  Oracle numbers (21.6/2.6/0.1) are kept in both figure and body as the design rationale (allowed exception).

### Related work
- L2: three bold run-in leaders removed; each paragraph now opens with our position (teacher sees less / two opposite roles /
  credit within one response). Pointer to the full comparison moved from the first line to the end.
- Phase 5.3 cross-section duplication: the on-policy-distillation estimator sentence (also in Sec. 3.5) and "principle is a
  necessary condition only" (also in Sec. 3.5) removed here; the LwF/SCoRe tether sentence shortened (Sec. 3.3 holds the detail).

### Method
- L1/L2 unchanged (setup -> T1 -> T2 -> combined update -> theory); leads made insight-first (T2: "keeps what is right").
- L4/ML-beginner: natural-language sentence added before Eq. 1, Eq. 2 and the PMI definition; "row" defined; the reset-only arm
  defined where first used; "stop teacher" unpacked; the 39-word shared-terms sentence split in three; theory consequences split.
- Phase 5.4: Figure 3 caption lost its last two sentences, which the figure itself prints.

### Experiments (4.0 setup, 4.1 H1, 4.2 H3, 4.3 H2)
- L2 setup: bold leaders "Protocols." / "Metrics and arms." removed; claims + protocols + arms in one paragraph, metrics in one.
  avg@16 / pass@16, R/W and AUC defined (ML-beginner).
- L2 4.2 H3 (main restructure): before = method -> results -> interpretation -> more results (R3 inversion). After = intent
  (H3 question) + method + pre-registered predictions -> habit results -> outcome results (transitions, habit-fix link) ->
  interpretation (Proposition 5, T1 broadens, T2 makes safe).
- L2 4.3: the last paragraph's lead covered only the placebo; new lead covers all four ablations (content of the credit and the
  fixed prefix, not signal size or longer training).
- Duplicates removed: "so more wrong answers are fixed..." (restated the paragraph lead), "R->W stays at the base level" in the
  interpretation (already stated next to Table 2), "Noticing remains open" (stated in Limitations and Conclusion).
- Each finding now sits next to its float: transitions next to Table 2, habits next to Figure 4.

### Limitations
- L2: noticing (the main limit) first, then a pivot sentence "The evidence is also narrower than the claim" for the scope limits.
  A two-paragraph split was tried and reverted: it pushed the conclusion to page 9.

### Conclusion
- Two long sentences (0% short) rewritten as four; claim, result and open problem each stated once.

## Phase 5 (integration)
- Self-containment: appendix references only for proofs and extra evidence; no forward references in prose.
- Caption <-> body: Figures 1 and 3 trimmed (see above); Table 2 values are not repeated in the body (body states only the
  pattern and the paired differences with intervals, which the table does not hold).

## Phase 6 (notation audit, body + appendix)
- Label/ref integrity: 0 undefined refs, every numbered equation referenced. Unreferenced floats (`fig:main`, `fig:pipeline`,
  `fig:teaser`, `tab:behavior`, `tab:ood`) are appendix floats and were already unreferenced before this pass.
- Symbol clashes fixed (EN and KO): `\xi` meant both "extra content determined by c_S" (Prop. 3) and "the note's instruction"
  (App. B, Eq. app-anchor) -> the instruction is now `\iota`; `r` meant both reward and the ratio argument of `f_delta`
  (App. B) -> the ratio is now `q`.
- Units: fix rates in % throughout (main text and App. E); ICC, AUC and KL stay as decimals.
- Body vs appendix: alpha, lambda = 0.002 for the reset-only arm, beta/(lambda+beta) = 10/11 and the stop-teacher signal
  (-0.007 vs -0.25) agree between Sec. 3.5 and App. B/F.

## Figures
- `fig_habits`: now `figure*` at `\textwidth` (was 0.8\textwidth); figure redrawn at 6.75 x 3.3 in, tick labels 8.5-9 pt and
  in-panel legends 8.5 pt. Smallest printed text: EN 9.2 pt (was 6.5 pt), KO 10 pt. `pairbars()` got an `fs` argument.
- `fig_overview`, `fig_scoring`: "Wait — re-solve" -> "Wait, re-solve" (`figs_method.py`; model text, so identical in KO).
- All main-text figures at printed size: overview 7.7 pt, scoring 7.5 pt, oracle 8.1 pt, habits 9.2 pt minimum (EN; KO equal or larger).

## Phase 7 (build)
- EN: 39 pages, main text pages 1-8 (Impact Statement and references from page 9). KO: 37 pages, main text pages 1-8.
- No errors, no undefined references or citations, no overfull boxes in either build.

# 2026-10-12 — Amendment 74 (baseline teachers, online-T1 ablation, baseline choice)

No new results; all new numbers are `\todo`. EN and KO mirrored (numbers, refs, todos: per-file counts equal).
- 4 setup (`experiments.tex`): arms sentence now states GRPO = DAPO recipe, SDPO-style = frozen teacher, RLSD/RLRT = current
  policy as teacher (their equations); one sentence on shared fixed prefixes (ours, one-teacher, SFT arms) vs from-scratch
  baselines; one sentence pointing to App. C.4 for why full SCoRe, OPSD and Four Habits priming are not run.
- 4.3 (`experiments.tex`): new 3-sentence paragraph on the online-T1 arm (`ourson`) with the pre-registered switch rule.
- App. B.3 (`app_theory.tex`, new `app:online`): online T1 => drift term of Eq. split is zero, d_t = current-model -PMI,
  T1 only changes, frozen T2 alone keeps; Prop. 3 holds at every step; cost = moving target in Prop. 4.
- App. C.3 (`app_setup.tex`): prefix sharing; online-T1 ablation arm + rule; new "Self-distillation baselines" paragraph
  (RLSD Alg. 1 / RLRT Eq. 1, Table 4; RLRT Fig. 4a and SDPO-fork base code use a frozen teacher, we follow the equations;
  w_t formula, lambda_init .5, eps_w .2/.5, decay 50/30, RLRT correct rows only; SDPO-style frozen, sampled-token additive,
  not top-100 JSD). Table 3: teacher identity per baseline, new "Ours, online T1" row. New App. C.4 "Baseline choice"
  (`app:baselines`); compute moved to C.5.
- App. F (`app_results.tex`): online-T1 row in Table 13 (caption now "preliminary protocol unless marked main"); one
  sentence separating it from the on-policy reset. Per-1k-token habit densities moved here from 4.2.
- Page budget (EN was full): 4.2 ICC sentence merged, per-1k numbers and the AUC .46 sentence moved to the appendix,
  intro contribution (1) merged, shorter todo texts in 4.3, Limitations and Conclusion. KO only: Table 1 and Figure 4
  float definitions moved to the start of Sec. 4 so they print before the references.
- Build: EN 41 pages, main text pages 1-8; KO 39 pages, main text pages 1-8. No errors, no undefined refs, no em-dashes.

# 2026-10-12 — One-method cleanup (abbreviations, working notes, todos)

No number, citation, equation or figure file changed. EN and KO mirrored; per-file counts of `\todo`, `\ref`, citation keys
and decimal numbers are equal in all 11 section files.
- Naming: T1/T2 macros removed from all prose, captions and tables (0 uses left; symbols stay only in equations as
  `\pi_{T1}`). Prose uses "the blindfolded teacher" and "the tether" (yesterday's self); the overview caption and the first
  appendix paragraph say that figures label them T1 and T2. `hdev` -> "our hard development set" / "hard dev set" ("Hard dev"
  in table headers); the label is decoded once in App. C.2 and App. D. W->R / R->W -> "wrong-to-right" / "right-to-wrong" in
  prose; arrows kept only in tables with a caption definition. fix|W / keep-correct|R -> "fix rate" / "keep rate" in prose.
  OOD, CI, MATH L5, EMA removed; AUC, maj@k, SFT, GRPO (abstract) spelled out at first use; PMI spelled out in the method,
  App. B.2/B.3 titles and text; baselines in Sec. 4 each get one phrase (SDPO-style, RLSD/RLRT, STaR/Pair-SFT), split
  into separate sentences.
- Working notes removed: "preliminary protocol" -> "small-scale protocol" (abstract/intro/conclusion: "a small-scale study");
  "pre-registered" kept nowhere except one setup sentence ("We fixed all decision criteria before seeing results");
  seed lists "2, 3 and 5" -> "three seeds" (one decoding sentence left in App. C.2 because figures print s2/s3/s5);
  "pending", "All cells are pending", "so far", "(if reported)", "earlier rounds", code names (grpokl, cd9 rounds,
  a85_inject.py, teacher_update_rate, eval paths) removed. GPQA-Diamond column and "added only if obtained" deleted.
- One method: the online-teacher arm is now an ablation question in 4.3 ("Does the blindfolded teacher need to be frozen?")
  with `\todo{result}`; the switch rule is deleted from 4.3, App. C.3 and Table 13. App. B.3 kept as a neutral remark.
- Todos: every planning todo replaced by a short placeholder (`\todo{result}`, `\todo{main-protocol result}`,
  `\todo{denominator}`, `\todo{GPU-hours}`, `\todo{probe details}`); deleted: second-model, final-seeds/compute,
  contribution-(3) and limitations todos.
- Skill gates (main text): short-sentence ratio EN 75-96% per section; em-dashes 0; abstract long sentence split.
  Acronym-like tokens in the main text 159 -> 96 (T1/T2 30 -> 0, hdev 11 -> 0, arrows 11 -> 6, table only).
- Page budget: setup sentence on prefix sharing moved out (already in App. C.3), fix/keep definitions merged, Table 1
  `\tabcolsep` 5pt -> 3.4pt for the spelled-out headers. EN main text ends on page 8 (Impact Statement on page 9);
  KO ends on page 8. EN 42 pages, KO 39. No errors, undefined references or overfull boxes.
- Open (figure files untouched by instruction): figure PDFs still print `hdev`, "Preliminary protocol", "pending" and, in
  `fig_passk` panels c/d, the path "eval/rl_*"; regenerate with make_figs.py to finish the cleanup. `fig_ablation` prints the
  uniform first-answer interval as [-5.0, +0.9], Table 13 as [-5.0, +1.0].

# 2026-10-12 (2): Figure labels cleaned, text and figures made consistent

No plotted number or data file changed; `--refresh` not run. Labels edited at their source (figs_plot.C, BENCH, label
literals in figs_results/figs_method/figs_causal) and the KO dictionary in figs_i18n.py; ledger run/teacher names are mapped
for display only (figs_results._run_label, figs_method.TEACHER).
- Arms: "Reset only (T1)" / "Tether only (T2)" / "Ours (T1+T2)" -> "Reset only" / "GRPO + tether" / "Ours", the names the
  tables use (KO 리셋만 / GRPO + 끈 / 제안). Overview "(T1, frozen)" -> "(frozen)"; pipeline, teacher-signal, dynamics (d) and
  token-credit legend drop T1/T2 (only the math symbol pi_T1 of Eq. d remains). KO figures now say 눈 가린 교사 (as the text),
  not 눈 가린 나.
- hdev -> "Hard dev set accuracy (%)", "Hard dev, lenient" etc., tick "Hard dev" (KO 어려운 개발 (세트), as the text);
  "MATH L5" -> "MATH Level 5", "first box" -> "first answer", "MATH500" tick -> "MATH-500".
- "Preliminary protocol, 3 seeds (final)" -> "Small-scale protocol, 3 seeds" (KO 소규모 프로토콜).
- Placeholder slots: hatched slot with the single word "result" (KO 결과) in every panel, forest row and legend; file names,
  scripts, "pending", "vLLM", "one GPU", "eval/rl_*" removed.
- Seeds renumbered everywhere they are printed: training seeds 2, 3, 5 -> seeds 1, 2, 3; seed 4 (reset-only and ours
  stability runs) stays seed 4 (figs_plot.SEED). App. C.2 now says "three seeds (seeds 1, 2 and 3)" plus one sentence on
  seed 4; tab:prelim, tab:ablation, App. D/E prose and captions renumbered in EN and KO (KO particles fixed).
- Text sentences that only decoded figure labels deleted (intro overview caption T1/T2, App. C.2 hdev/seed labels, pipeline
  caption T1/T2, App. D first paragraph, forest caption "labelled preliminary"), EN and KO.
- Rounding: uniform first-answer CI stored as [-0.0499, +0.0095] (ledger source: [-4.99, +0.95] points). forest() now rounds
  half away from zero on the stored digits, so the figure prints [-5.0, +1.0] like Table 13 (Python's float format gave
  +0.9). No other forest label changes.
- Layout: forest left margin 0.31 -> 0.36 and OOD note shortened; oneteacher 6.2 -> 6.0 in; teachers right 0.72 -> 0.70;
  dynamics (d) title "Blind credit?"; pass@k (a)/(b) titles "Lenient"/"Strict". All figures <= 6.5 in except fig_tokens
  (6.59 in, included at 0.92 textwidth). Main text ends on page 8 in EN and KO; no undefined references.

# 2026-10-13 — paper-section-rewrite pass over every section (EN, mirrored to KO)

Skill run per file: Phase 0/0.5, Level 1-4 critic loops (Level 1-3 converged), Phase 5 (self-containedness,
caption<->body), Phase 6 notation audit (body + appendix), Phase 7 build. No reported number, citation key, label, data
file or figure file changed; no figure regenerated. Parity script (todo, refs, labels, cite keys, decimals): all 11 files
EN = KO. Short-sentence share: EN <= 20 words, KO <= 50 characters including spaces. Backups of the pre-pass section
files: /hdd_data/seungpil/tmp/paper_rewrite_1013/{en,ko}_sections_before.

Per section (rounds L1/L2/L3/L4; short share before -> after, EN | KO; min paragraph after):
- abstract (1/1/1/2): 88->89% | 71->83%. "(anchoring)" defined where the cause is stated; tether definition and role split;
  "blind credit" (undefined in the abstract) -> "the blindfolded teacher's credit"; KO long result sentences split.
- intro (1/2/1/2): 70->86% (min 75) | 77->86% (min 67). Oracle caption split and probe set defined in the caption;
  "Yesterday's self ... is the KL penalty" (category error) -> "its signal is the KL penalty"; "routed" -> "which teacher
  acts is set by the fixed first answer"; results sentence split; contribution (2) split.
- related (1/2/1/1): 96->96% | 90->86%. P1 reordered (critique-conditioned teachers next to the other "see more"
  teachers, before "ours sees a strict subset"); leap fixed: "these Four Habits" after naming two habits -> "these two
  and two control habits, the Four Habits of that study".
- method (1/1/1/2): 90->95% (min 83) | 91->96% (min 67). Scoring caption split (tether named); credit definition split;
  leap fixed: why the credit gives signal when all K continuations fail (outcome advantage is zero); 3.4 lead now
  insight-first (no new loss: credit joins the advantage, tether is the usual KL term); 3.5 lead split; "the condition is
  necessary only" -> "the information gap is necessary, not sufficient"; Phase 6 additions: "(for us, c_S = s)" in the
  teacher-context proposition and one sentence that the implemented KL estimator matches the tether to second order.
- experiments (1/2/2/2): 78->87% (min 67; 4 failing paragraphs/captions -> 0) | 74->92%. Setup paragraph split into
  protocols and baselines; claims H1/H3/H2 one per sentence; RLRT glossed at first use; H1 result paragraph reordered
  (strict-gain cause right after the strict number, transitions before the other benchmarks); judge paragraph split;
  habits caption no longer repeats the judge setup of the body; vague "point the same way" -> named analyses agree;
  "costs little" -> "costs little accuracy"; "guards differed" -> "between the two arms"; appendix refs per subsection
  <= 2 (12 -> 6 in the file: setup 2, H1 1, H3 1, H2 2). KO: ICC abbreviation dropped (EN had none).
- limitations (1/2/1/1): 92->92% | 92->93%. Split into two paragraphs (noticing; costs and narrow evidence); leap
  fixed: a blind copy shares the blind spot, "so its credit cannot point away from it".
- conclusion (1/1/1/1): 75->100% | 50->100%. Two-role sentences split.
- App. A related_full (1/2/1/2): 58->~100% (7 paragraphs < 60% -> 0) | 90->90% (min 73). Insight-first leads, citation
  lists one sentence per method, acronym glosses from bib titles, note-like phrases removed ("appeared in October 2026",
  "earlier probes"), 7-gram overlap with related.tex 8 -> 0.
- App. B app_theory (1/2/2/2): 85->98% (min 86) | 96->100%. Claim-first leads, run-in labels -> claim sentences, proof
  steps made explicit; g (geometric mean) -> pi_geo, w_h/W -> omega_h/Omega (clashed with g, w, \mathcal{W}).
- App. C app_setup (1/2/2/2): 84->97% (min 80) | 87->96%. Arms and baseline-choice paragraphs split; 4 caption-body
  duplications removed; baseline symbols lambda -> mu, alpha -> gamma, then (main pass) w_t -> z_t and Delta_t -> tilde d_t
  (clashed with w in Eq. adv and Delta in Table 2), tilde d_t tied to Eq. d.
- App. D-F app_results (1/2/1/2): 67->98% (15 paragraphs < 60% -> 0) | 81->90% (min 62). Leads added (H3 had none);
  13 caption/table-body duplications removed; row-by-row ablation prose -> two interpretive paragraphs. Factual
  correction: the beta = 0.25 reset-only arm has lower lenient but higher strict accuracy than ours (63.0 vs 60.9),
  not "did not beat ours".
- Gates (all files, EN and KO): em-dashes 0; AI tells <= 2 per paragraph; KO goja ratio 0.00; KO copula substitutes 0;
  KO nominal endings left only in run-in headings and two questions.
- Build: EN 41 pages, main text pages 1-8 (Impact Statement p. 9, ~2 lines spare in the right column of p. 8);
  KO 40 pages, main text ends on p. 8. No errors, no undefined references or citations, no overfull boxes.
- Open: hypothesis labels are numbered H1, H3, H2 in section order (kept to match the appendix); c_T/c_S (contexts) vs
  c_t (credit) and D_delta vs D_STaR share letters with different subscripts; k means both pass@k and top-k.
- 2026-10-13 reflection injection (EN+KO; app_results "Reflection injection", experiments H3 end): placeholder -> fig_inject.pdf
  with a panel caption (RLRT fix = wrong->right on hard, break = right->wrong on easy, phrase effect; setup only, no result
  numbers); result paragraph from results_a85_inject.json (all-problem phrase effect after the first answer, random point
  intervals contain 0, ours-GRPO phrase effect +1.7 [-4.6, +8.1], unprompted fix 18.9/12.9/10.1 with +6.0 [+1.5, +11.1],
  break cost 5.1 vs 2.6). Main text: two sentences replace the inject todo; Table metaood todo kept. Short sentences
  EN 89%, KO 90%; em-dashes 0. EN space: four line-saving trims in experiments.tex (baselines "teacher alone", "correct
  sibling", "the controls stay unchanged", frozen-teacher sentence "ties it to piref"). Build: EN 42 pages, KO 40; main
  text ends on p. 8 in both (EN ~2 lines spare); no errors, no undefined references, no new warning types.
- 2026-10-13 Re^2 / Critique-OPSD comparison (EN+KO): related.tex P1 rewritten as a three-way contrast (Re^2 = no teacher,
  outcome reward, redo template, test-time resampling; critique-conditioned and reflective SD = teacher sees critique + full
  rollout; ours = problem + previous answer, no critic or template); rest of the old P1 kept as supporting evidence; CAST
  sentence dropped (still in intro and App. A), Re^2 removed from the self-correction paragraph, multi-round sentences merged,
  "however" cut. related_full: Critique-OPSD details (critique source, wrong-final-answer gating, error sentence + ~50 tokens,
  Wait/But/Actually) plus our three differing choices (strict-subset context, state routing, dense credit after the first
  answer); Re^2 details (8x8 two-stage generation, truncation uniform in [0, 0.8], redo template, out-of-group redo reward,
  0.5% -> >30% redo, multi-attempt test). app_setup: main-protocol MATH Level 5 = fixed random 200-problem subset, disjoint
  from hard dev and training, 16 samples; baseline choice gains two paragraphs (Re^2: multi-attempt test, 64 vs 8 rollouts,
  GRPO + tether and uniform as the controlled prefix baselines; Critique-OPSD: per-rollout critique, concurrent work,
  -0.007 vs -0.25 teacher-context measurement reused unchanged). Short sentences related.tex EN 96%; em-dashes 0. Build: EN
  43 pages, KO 40; main text ends on p. 8 in both (EN ~2 lines spare, as before); no errors, no undefined references or
  citations, no new warning types.
