# Retired launchers (moved 2026-07-27)

These were live for part of 0727 and are kept for the record, not for use.

## Why each is here

`a100g1_sft_*` / `a100g1_sft_*_eb16` — the 1-GPU SFT2 path. Created to escape an
11-12h queue for 4-GPU slots, then abandoned when the quota table showed the
A100-80GB **user max is 1 GPU** and, separately, that a 1-GPU run needs ~7.7h
against an observed ~3.5h preemption window. It cannot finish. The `_eb16`
pair additionally exists because the first 1-GPU conversion silently kept
`gradient_accumulation_steps: 4`, which on one process is effective batch 4
rather than T1's 16 — the fix that produced them is the reason they are worth
reading.

`a100g2_sft_*` — the 2-GPU variant. Never submitted; superseded by the same
user-max finding. Note its output paths collide with the live 4-GPU pair, so
running it would overwrite the real artifacts.

`h100std_probe_*` — one-shot gate probes that completed.

## What replaced them

The live path is `a100_sft_{b0p2,b2p2}_rvfull.yaml` (SFT2 pair) and
`a100_rq3v2f_{b0p,b2p,b3p}.yaml` (RL), listed in CLAUDE.md. Nothing here should
be resubmitted; the compute assumptions they were built around no longer hold.
