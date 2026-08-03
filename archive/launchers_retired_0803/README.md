# Retired launchers (moved 2026-08-03)

Nine root launchers, archived — not deleted, and not for resubmission. The live
set at root is now exactly five: `h100std_rq3v2f_{b0p,b2p,b3p}.yaml` (the three
running RL arms: solid-gibbon / hip-hound / pure-stag) plus
`h100std_sft_{b0p2,b2p2}_rvfull.yaml`.

## Why each is here

### `a100_rq3v2f_b0p.yaml` · `a100_rq3v2f_b2p.yaml` · `a100_rq3v2f_b3p.yaml`

**These three could destroy a running arm.** Each is an A100 twin of a live
H100 launcher, and the twinning is not merely cosmetic — the full diff against
the live file is 24–26 lines, all of them cluster (`msrresrchvc` vs
`msrresrchbasicvc`), SKU (`80G4-A100` vs `80G4-H100`), `sla_tier`, job name,
description, `WANDB_RUN_ID`, and the pusher-kill block. Every durable output
path is **byte-identical** to the live arm's:

| shared with the live arm | value |
|---|---|
| `--ckpt_dir` | `/scratch/checkpoints/rq3v2f_{b0p,b2p,b3p}` |
| `--repo_id` | `iamseungpil/metacot-h200-triobj-dcpo-v3` |
| `--config_name` | `rq3v2f_{b0p,b2p,b3p}` |
| final-sync `path_in_repo` | `checkpoints/rq3v2f_{b0p,b2p,b3p}/$${FN}` |
| resume puller | `pull_resume_ckpt.py --repo … --config_name rq3v2f_b0p --local_dir /scratch/checkpoints/rq3v2f_b0p` |

That is an overlap in the *write* direction, not just the read direction.
`scripts/push_ckpts_to_hf.py:35-95` (`_prune_old_verl_ckpts`) calls
`api.delete_folder(...)` on older complete checkpoints inside the same
`config_name` namespace, and all three launchers invoke it with `--keep 1`. So
a stray `amlt run` on any of these would (a) resume-pull the live arm's HF
state as if it were its own, (b) push its own steps into the live arm's path,
and (c) **actively delete the live arm's saved `global_step`s** — the only
resumable state a preempted arm has. Archiving them is a safety measure, not
tidying.

Second, independent reason to retire them: they still carry the pre-0802
self-kill bug. `a100_rq3v2f_b0p.yaml:311` (b2p:305, b3p:295) is
`pkill -f push_ckpts_to_hf`, which matches the launcher's own `bash -c` command
line and SIGTERMs its own shell, so FINAL SYNC PUSH never runs. That bug already
cost b2p its gs300 weights. The live trio replaced it with a PID-only kill
(`h100std_rq3v2f_b0p.yaml:313-320`). Resubmitting an A100 twin would reintroduce
a known data-loss defect.

### `h100std_rq3v2_b2p.yaml` · `h100std_rq3v2_b3p.yaml` · `a100_rq3v2_b3p.yaml`

Old lineage (`rq3v2_*`, no `f`). They write to the **same HF repo** as the live
trio, `iamseungpil/metacot-h200-triobj-dcpo-v3`, but under
`/scratch/checkpoints/rq3v2_{b2p,b3p}` and `checkpoints/rq3v2_{b2p,b3p}/`, so
there is no path collision with the live `rq3v2f_*` arms. They are retired
because the lineage is superseded, not because they threaten a live arm.

`a100_rq3v2_b3p.yaml` and `h100std_rq3v2_b3p.yaml` do, however, collide
**exactly with each other** — both target `/scratch/checkpoints/rq3v2_b3p` and
`checkpoints/rq3v2_b3p/`. That hazard was recorded at
`docs/reports/2026-07-17-rq3-run-and-iteration-log-part1.md:1784`. Both members
of the pair are archived here, so the pair is retired together and the hazard
is closed rather than half-closed.

### `a100_sft_b0p2_rvfull.yaml` · `a100_sft_b2p2_rvfull.yaml`

A100 twins of the SFT2 pair that stays at root. They use
`--repo_id iamseungpil/metacot-sft2-4g` with
`--ckpt_dir /scratch/checkpoints/{b0p2,b2p2}_rvfull_sft` — zero overlap with any
live arm's repo or ckpt_dir, so no hazard. They are exact duplicates of
`h100std_sft_{b0p2,b2p2}_rvfull.yaml:126-131`, which remain live. Archiving them
*reduces* the duplicate-launcher surface; it does not create one.

### `h100std_env_builder.yaml` — **this reverses a recorded decision**

This file is not an experiment and has no `--ckpt_dir` and no training. Its only
write is `h100std_env_builder.yaml:79`, uploading
`env_snapshots/simplerl_v4.tar.gz` to `iamseungpil/metacot` (repo_type
`dataset`). There is no checkpoint hazard of any kind.

Two documents previously and explicitly decided to **keep it at root**:

- `docs/reports/2026-07-17-rq3-run-and-iteration-log.md:742` — "h100std_env_builder.yaml은 실험이 아니라 conda env 빌더이므로 루트에 남겼다"
- `docs/CODE_MAP.md:142` — "h100std_env_builder.yaml은 conda env 빌더(실험 아님)"

That decision is being reversed here, deliberately and on the record: root is
reserved for launchers you might actually submit, and this one is a one-off
producer that has already produced. Nothing breaks at runtime. The **consumer**
of its output is `scripts/bootstrap_sdc_node.sh:68` (`PACK_URL`) and `:92`
(`filename=`), which fetches the tarball from HF **by URL** on every node
bootstrap — including live-arm preemption resubmits — and never reads this yaml.

> **Do not delete this file.** It is the *only* recipe for regenerating
> `env_snapshots/simplerl_v4.tar.gz`. If the HF copy is ever lost, every node
> bootstrap — and therefore every preemption resubmit of a live arm — fails at
> `scripts/bootstrap_sdc_node.sh:68`, and this file is how you rebuild the pack.

## References

No executable reference to any of the nine exists anywhere in the repo. A
repo-wide grep excluding `.git` and `archive/`, run once restricted to
non-Markdown files and again restricted to `*.yaml` (to catch one launcher
invoking another), returned **zero** hits. Nothing in `scripts/`, `src/`,
`configs/`, `experiments/`, or any protected path invokes these filenames.
Neither protected doc (`docs/PREREGISTRATION_rq3v2_base_replication.md`,
`docs/CONSTITUTION.md`) mentions them.

Every inbound reference is Markdown prose. The stale ones —
`CLAUDE.md:61-62` and `docs/CODE_MAP.md:16,32,74,134,135,141`, which presented
five of these nine as the *current* launcher set and would have pointed an
operator straight at the three destructive files — are repaired in the edit
commit that follows this one, along with the "What replaced them" pointer in
`archive/launchers_retired_0727/README.md`.

`docs/reports/2026-07-17-rq3-run-and-iteration-log.md` and its `-part1` are
**deliberately left untouched**. Their ~25 references are a historical log of
what was launched and when; editing a run log to match a later filesystem
layout falsifies the record.
