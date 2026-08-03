# Reports invalidated by the 2026-08-03 independent regrade

**Every benchmark number in this directory is retracted.** These four reports are kept as the
record of what was believed before the regrade, and for no other purpose. Each carries a
line-1 RETRACTED banner added when it was archived.

## What invalidated them

`docs/reports/2026-08-03-independent-regrade-instruct-arms.md` re-graded the instruct arms with
a corrected grader and declared the stored values "전부 무효". The supersession is numerical and
exact, not a matter of interpretation:

| quantity | as reported here | after regrade |
|---|---|---|
| MATH500 16k base_matched | 54.40% | **55.17%** |
| MATH500 16k pmishift | 66.00% | **69.17%** |
| MATH500 16k gandhi | 68.08% | **60.02%** |
| headline `pmishift − base` | +18.8pp | **+14.00pp** [+11.62,+16.50] |
| net reward `shiftonly − gandhi` | +5.6~5.9pp | **+4.38pp** [+2.12,+6.75] |
| "gains concentrate in hardest quartile" | +34.8pp | judged "회귀-평균 인공물일 가능성이 높다" |

The 54.40 / 66.00 pair is literally the "저장된 값" column the regrade voids, which is how these
four were identified rather than guessed at.

## Why each thing is here

`2026-07-05-pmishift-gs300-provisional-T1.md` — provisional T1 at gs300.
`2026-07-06-T1-paired-full-pmishift-vs-base.md` — the full paired T1; the direct source of the
retracted 66.0 / 54.4 headline.
`2026-07-06-T1-paired-full.json` — the machine-readable sidecar for the above. It has **zero**
inbound references anywhere in the repo; it is archived with its report rather than left behind
so the report's underlying numbers stay next to the report.
`2026-07-07-RQ2-priming-gandhi-vs-base.md` and
`2026-07-08-RQ2-isolated-pmishift-net-shiftonly-vs-gandhi.md` — the RQ2 priming decomposition.
Both report gandhi MATH500 at 71.5–72.1% against the regrade's 60.02%.

## Still-open content debt this archive does not discharge

Archiving the sources does not remove the retracted numbers from where they are still asserted
as live. Two carriers remain and need a numbers edit, not a path edit:

1. `docs/EXPERIMENT_PLAN.md:133` §2.4 item 2 still states "+5.6~+5.9pp" and cites the 07-08
   report. The path citation was repaired to point here; **the number was not**, because the
   regrade puts `shiftonly − gandhi` at +4.38pp. A live front-door doc (linked from `README.md:5`)
   therefore still carries a retracted figure.
2. `paper/` still publishes the retracted set: `paper/sections/abstract.tex:28`
   ("GSM8K by 4.0, MATH500 by 18.8, AIME by 14.2"), `paper/sections/experiments.tex:94,117,119`,
   `paper/RESULTS.md:19`, and the +34.8pp hardest-quartile claim at `paper/sections/intro.tex:80`,
   `experiments.tex:232,251`, `paper/RESULTS.md:32`.

Both are out of scope for a reorganisation series and are flagged here so the debt is not lost.
