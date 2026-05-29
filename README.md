# Meta-CoT — SDC, RLSD, Baseline GRPO

Train LLaMA-3.1-8B with metacognitive reasoning via `<|meta|>` tokens.
Three RL algorithms are compared: **SDC** (Shared-preserve Directional Credit), **RLSD** (Reinforced Self-Distillation contrastive meta), and **Baseline GRPO** (vanilla, no meta).

```
SFT init  ──▶  RL training  ──▶  1,030-problem eval  ──▶  ckpt push to HF
                  │
                  ├── SDC          (meta SFT init + SDC algo)
                  ├── SDC-base     (base SFT init + SDC algo, ablation)
                  ├── RLSD         (meta SFT init + contrastive RLSD)
                  └── Baseline     (base SFT init + vanilla GRPO)
```

> Model is **Qwen3-8B** (the line above is historical). The sections below from
> "TL;DR" down are the older SDC/RLSD/Baseline infra reference. The **current**
> work is CTSD — read the next section.

---

# CURRENT WORK — CTSD (Contrastive Triggered Self-Distill)

Full plan with intent / hypothesis / verification per step: **[PLAN.md](PLAN.md)**.
Node-launch index for the training arm: **[configs/CTSD_NODE_INDEX.md](configs/CTSD_NODE_INDEX.md)**.

**The question.** Can self-monitoring metacognition (`<|meta|>` blocks emitted
mid-solution) make Qwen3-8B better at math? Prior approaches failed: single-teacher
self-distill is null on our distribution, the model's natural meta is *decorative*
(carries no good/bad signal), and a contrastive RL run (R18b) scored 70.9% < 72.3%
baseline. CTSD adds **force-inject**: insert `<|meta|>` at the model's most-uncertain
point so a contrastive reward has a meta region to shape.

## Pipeline & entry points (run this code → get this experiment)

```
Phase A (local A100, inference only, NO training)
   ├─ A.1 contrastive control      experiments/probes/a1_contrastive_with_controls.py   → reports/a1_*.json   [done: FAIL — meta decorative]
   ├─ A.2 entropy threshold        experiments/probes/a2_entropy_distribution.py        → reports/a2_*.json   [done: PASS, AUC 0.749]
   ├─ A.6 teacher discrimination   experiments/probes/a6_six_cell_teacher_swap.py       → reports/a6_*.json   [done: PASS, AUC 0.81–0.95]
   └─ A.3 force-inject CAUSAL ★    experiments/probes/a3_inject_causal.py               → reports/a3_*.json   [THE GATE for training]
                                          │  PASS → Phase C/E   |  INCONCLUSIVE → raise --max_new   |  FAIL → stop + rethink
Phase C/E (AMLT H200 node, RL training — gated by A.3 PASS)
   └─ ROD_MQ_CONTRAST_INJECT       configs/verl_ctsd_inject_C_h200_4x4k.yaml + h200_ctsd_inject_C_smoke.yaml
```

### Phase A probes — local, no training
```bash
source experiments/common/load_secrets.sh          # loads .env (HF_TOKEN etc.)

# A.3 — the gate: does force-injecting <|meta|> at the max-entropy point causally help?
python -u experiments/probes/a3_inject_causal.py --n 30 --k 4 --max_new 2048
python -u experiments/probes/a3_inject_causal.py --smoke 2          # 2-problem smoke first

# other Phase A probes (already passed)
python -u experiments/probes/a2_entropy_distribution.py
python -u experiments/probes/a6_six_cell_teacher_swap.py --n_per_bench 7
```
Each writes a JSON verdict to `reports/`. A.3 gates: **helps** (good-inject beats
no-inject, +3pp, p<0.05) **AND direction** (good beats bad-inject, +5pp, p<0.05),
with a `boxed_rate<0.5` power guard → INCONCLUSIVE rather than a false null.

### Phase C — RL training (only after A.3 PASS; runs on a node, not locally)
`ROD_MQ_CONTRAST_INJECT` = R18b's contrastive reward **+ force-inject** (one-axis
ablation). Inject core is pure + unit-tested:
```bash
python src/training/tests/test_meta_inject.py      # 9 tests, core logic
amlt run h200_ctsd_inject_C_smoke.yaml ctsd-inject-c-smoke -d "CTSD Phase C smoke"
```
**Node-first step:** wire `SDCRayPPOTrainer._force_inject_rollout` (the two-phase
DataProto repack) and 1-step smoke it, then remove the `__init__` fail-fast guard.
Until then the job intentionally refuses to launch (`sdc_force_inject=true`). See
[configs/CTSD_NODE_INDEX.md](configs/CTSD_NODE_INDEX.md) for the full launch order.

---

## TL;DR — pick the experiment, submit the yaml

| Experiment | yaml | Cluster | SKU |
|---|---|---|---|
| SDC single-node H100 STD ×4 | `h100_1node_a_0424.yaml` (also b/c/d) | msrresrchbasicvc | 80G4-H100 |
| SDC + Baseline (paired, 2 nodes) | `h200_2nodes_sdc_baseline_0424.yaml` | msrresrchbasicvc | 141G4-H200 |
| SDC + RLSD (paired, 2 nodes) | `h200_2nodes_sdc_rlsd_0423.yaml` | msrresrchbasicvc | 141G4-H200 |

```bash
# 1. Refresh the HF code tarball if scripts/ changed
bash scripts/build_sdc_code_snapshot.sh

# 2. Submit
amlt run h100_1node_a_0424.yaml metacot-h100-1n-a-0425 -d "SDC node a"
```

The yaml command pulls the tarball from HF and runs `scripts/run_sdc_on_h200_node.sh`, which orchestrates everything (see [Runtime](#runtime) below).

## Repository layout

```
metacognition/
├── README.md                 ← this file
├── CLAUDE.md                 ← Claude/agent guide (tokens, goal, status)
├── NODE_POLICY.md            ← AMLT node ownership contract
├── REPORT_REFERENCES.md      ← report → file → HF path map
├── ANALYSIS_MAP.md           ← analysis output index
│
├── scripts/                  ← all shell + python entry points
│   └── CATALOG.md            ← per-script role index (start here)
├── src/                      ← training/eval source code
├── configs/                  ← verl + SFT configs
├── data/                     ← local data caches (gitignored)
├── results/                  ← evaluation outputs (HF-mirrored)
│
├── h100_*.yaml               ← H100 STD single/multi-node submissions
├── h200_*.yaml               ← H200 BSC submissions
├── metacognition_*.yaml      ← long-running A100/H200 reservation holders
│
├── archive/                  ← dated cleanups
└── legacy/                   ← retired code
```

## Experiment menu

### 1. SDC (main)

**Goal**: validate that SDC RL on a meta-SFT init outperforms base-SFT + vanilla GRPO on math reasoning.

| Component | Path |
|---|---|
| Outer orchestrator | `scripts/run_sdc_on_h200_node.sh` |
| Bootstrap (env install) | `scripts/bootstrap_sdc_node.sh` |
| Inner launcher | `scripts/launch_sdc_verl.sh` |
| Default config | `configs/verl_sdc_e21r_shared_h100_4x4k.yaml` (H100) / `verl_sdc_e21r_shared_h200_4x16k.yaml` (H200) |
| WandB project | `skilldiscovery2` |
| HF ckpt repo | `iamseungpil/metacot-sdc-verl-shared` |

Submission yamls: `h100_1node_a/b/c/d_0424.yaml`, `h200_2nodes_sdc_baseline_0424.yaml` (sdc job), `h200_2nodes_sdc_rlsd_0423.yaml` (sdc job).

### 2. Baseline GRPO

**Goal**: paired control. Same problems, base SFT init, vanilla GRPO without `<|meta|>` rewards.

| Component | Path |
|---|---|
| Outer orchestrator | `scripts/run_baseline_verl_on_node.sh` |
| Inner launcher | `scripts/launch_baseline_verl.sh` |
| Default config | `verl07_base_redirect` |
| HF ckpt repo | `iamseungpil/metacot-baseline-v100` |

### 3. RLSD

**Goal**: contrastive self-distill alternative to SDC.

| Component | Path |
|---|---|
| Outer orchestrator | `scripts/run_rlsd_on_h200_node.sh` |
| Inner launcher | `scripts/launch_rlsd_h200.sh` |
| Default config | `contrastive_meta_rlsd` |
| HF ckpt repo | `iamseungpil/metacot-rlsd-v100` |

### 4. SDC-base (ablation)

Same SDC algorithm but on a **base** SFT init (`v8_base_matched_clean_sft`). Isolates whether the gain comes from meta-SFT priming or from SDC itself.

## Runtime

`run_sdc_on_h200_node.sh` (and the parallel baseline/rlsd orchestrators) provide three resilience layers:

1. **gpu_keeper tmux** suppresses BSC idle-suspend during slow installs and idle phases.
2. **Bootstrap retry**: if the simplerl env install hangs >60 min, drop into keep-alive (does **not** terminate the AMLT job; SSH in, fix, `touch /scratch/retry.trigger`).
3. **Training retry loop**: 10 attempts with 60 s backoff. On exhaustion: keep-alive again. AMLT job stays alive until `max_run_duration_seconds`.

A heartbeat daemon (`nohup bash`) writes to `/scratch/logs/keepalive.log` every 3 min, plus nvidia-smi.

## Eval

```bash
bash scripts/run_eval_1030.sh <ckpt_path>
# or remotely
bash scripts/run_eval_1030_eval_node.sh <hf_ckpt_id>
```

The 1,030-problem benchmark = 500 GSM8K + 500 MATH-500 + 30 AIME2024. Output JSON goes to `results/<run>/<split>.json` and is mirrored to `iamseungpil/metacot:results/`.

## HF data layout

Code, models, results all live under `iamseungpil/metacot` (dataset repo).

```
iamseungpil/metacot
├── code_snapshots/metacognition.tar.gz     # current tarball (refresh via build_sdc_code_snapshot.sh)
├── models/<run_name>/                       # SFT/RL ckpts
├── results/<run_name>/<split>.json          # eval outputs
└── datasets/                                # SFT data parquets
```

Separate model-only repos for active RL runs:
- `iamseungpil/metacot-sdc-verl-shared`
- `iamseungpil/metacot-baseline-v100`
- `iamseungpil/metacot-rlsd-v100`
- `iamseungpil/metacot-sdc-base-v100`

## Reports

- `../metacognition-paper/main.pdf` — main NeurIPS submission (sections under `metacognition-paper/sections/`).
- `../metacognition-behavior-uncertainty/reports/behavior_uncertainty_working_note_ko.pdf` — Four Habits derivative working note.
- `REPORT_REFERENCES.md` — per-figure / per-number cross reference for the 2026-04-16 V8 final report.

## Status snapshot (2026-04-25)

Active SDC training run: `iamseungpil/metacot-sdc-verl-shared` (last commit timestamp = current training step).

Queued AMLT experiments:
- `metacot-h100-1n-{a,b,c,d}-0424` — STD H100 single-node SDC (4 lanes)
- `metacot-sdc-baseline-0424` — H200 BSC SDC vs Baseline
- `metacot-sdc-rlsd-0423` — H200 BSC SDC vs RLSD

For per-node detail and history, see `NODE_POLICY.md`.

## See also

- `QUICK_START.md` — clone → first run, three paths (AMLT / standalone / eval-only).
- `docs/experiments_intent_hypothesis.md` — Intent / Hypothesis / Validation Method per experiment.
- `scripts/CATALOG.md` — every script with a one-line role.
- `CLAUDE.md` — agent / token / data registry.
- `NODE_POLICY.md` — node ownership contract.
- `.env.example` — required environment variables for a fresh-server run.
