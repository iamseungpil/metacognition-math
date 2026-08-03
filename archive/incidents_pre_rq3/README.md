# Resolved cluster-access incidents (archived 2026-08-03)

Two documents from the 0717–0726 submission-block incident. The incident is over.

## Why each thing is here

`CLUSTER_SUPPORT_REQUEST.md` (2026-07-17) — the support-request text and evidence log for the
cluster access failure.

`2026-07-26-basicvc-submission-block-escalation.md` — the escalation for the msrresrchbasicvc
submission block, including the correlation ID.

## Why they are resolved rather than merely old

Two independent attestations. `docs/reports/2026-08-03-intent-vs-evidence-ood-audit.md:78`
already called `CLUSTER_SUPPORT_REQUEST.md` "해결된 사건 문서(아카이브 대상)". And the block
itself is demonstrably lifted: the three RQ3v2-F arms (solid-gibbon, hip-hound, pure-stag) are
submitted and running on msrresrchbasicvc / 80G4-H100 as of this commit.

Neither file carried a banner; both were given one when archived.

## References repaired

`README.md:8` (a real markdown link — the only genuine link break in the whole 0803 docs move),
`CLAUDE.md:43`, and `docs/reports/2026-08-03-intent-vs-evidence-ood-audit.md:78` were all
repointed here.

`docs/reports/2026-07-17-rq3-run-and-iteration-log.md` and its `-part1` also cite these files
(part1:2330, part1:2446). They were **deliberately left unedited**: they are append-only
historical run logs, and rewriting a past entry to match a later filesystem layout falsifies
the record of what the paths were at the time. Their citations are backticked prose, not links.
The same rule was applied to the 0803 launcher archive.
