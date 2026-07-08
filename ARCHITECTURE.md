# ARCHITECTURE — metacognition-math

> **START HERE.** This file maps the one live method, its exact file spine, and
> where the superseded variants and archived reward modes now live. Everything
> that is not on the spine below is history, kept for reproducibility under
> `runs/archive/` and `configs/archive/`.

## The one path that matters

The live method is **PMI-shift metacognitive self-distillation**: a teacher-free
self-distillation reward applied to the meta tokens. It is one command with one
config and one reward branch. Everything else in the tree is history — superseded
variants and probes, kept for reproducibility under `runs/archive/` and
`configs/archive/`.

```
LAUNCH   h100std_pmishift.yaml          (meta arm)
         h100std_base_matched_rl.yaml   (control arm — "base" = no-meta CONTROL)
   │  amlt → python -m src.training.verl_sdc
   ▼
CONFIG   configs/triobj_dcpo_v4_stage3b_h100_4x4k.yaml  (+ ++dcpo_rmeta_source=pmi_shift)
         configs/base_matched_grpo_h100_4x4k.yaml       (control; correctness-only)
   ▼
TRAINER  src/training/verl_sdc.py           entry + GDPO trainer (monolith)
         src/training/verl_sdc_utils.py     region masks / advantage / length cost
   ▼
REWARD   src/training/dcpo_pmi_shift.py     ★ the paper's reward
         src/training/dcpo_region.py        meta-region routing (where reward lands)
         src/training/rewards.py            correctness + meta shape/penalty heads
```

The RQ2 decomposition arms `h100std_shiftonly.yaml` and `h100std_gandhi.yaml`
launch the same spine with the PMI-shift head decomposed into its parts.

`dcpo_rmeta_source=pmi_shift` selects the `dcpo_pmi_shift.py` branch inside
`verl_sdc.py`. The sibling reward modules (`dcpo_pmi.py`, `dcpo_directional.py`,
`dcpo_asymcf.py`) are imported unconditionally and are **load-bearing at import
time**, but their code paths run only under *other* `dcpo_rmeta_source` values —
they are inert for the live method and must stay in place.

## What PMI-shift does (one paragraph)

For each rollout the frozen SFT reference model scores the log-prob of the gold
answer and a decoy answer at two teacher-forced positions — just before the meta
block opens and just after it closes. If probability that had drifted toward the
decoy swings back toward gold across the meta block (SAVE) the meta span is
rewarded; if gold drifts to decoy (DERAIL) it is penalized (asymmetric,
sign-reversal). The signal is the model's own gold/decoy discrimination distilled
into the meta region — **no external teacher**. Reward is routed by
`dcpo_region.py` onto META_CONTENT tokens only, sign-gated by correctness,
combined with a correctness head (`rewards.py`) and a length cost
(`verl_sdc_utils.py`) under a GDPO advantage.

## The "base" arm is a CONTROL, not a weaker model (naming note)

"base" / "base_matched" / "qwen3_base_sft" / "basearm" everywhere in this repo
means the **no-meta CONTROL arm**. It SFTs from `Qwen/Qwen3-8B` (the **INSTRUCT**
release) — the *same* starting model as the meta arm — and runs the *same* RL
pipeline minus the `<|meta|>` tokens and the PMI-shift head. It is **NOT** the
pretrained-only `Qwen/Qwen3-8B-Base`. The two arms are byte-identical except the
meta mechanism.

This is the single most misleading naming in the repo. Do not read "base" as a
weaker pretrained-only model. (A separate, genuinely-new **Qwen3-8B-Base
redesign** is a distinct effort; when that lands it will use `Qwen3-8B-Base` for
real — do not conflate it with the historical "base" control here.)

## Module map

| Role | Files |
|---|---|
| CORE entry/trainer | `verl_sdc.py`, `verl_sdc_utils.py` |
| CORE reward (live) | `dcpo_pmi_shift.py`, `dcpo_region.py`, `rewards.py` |
| CORE reward (imported, other rmeta modes) | `dcpo_pmi.py`, `dcpo_directional.py`, `dcpo_asymcf.py`, `meta_revision_rewards.py`, `_decoy_utils.py` |
| CORE SFT/tokens | `sft.py`, `tokenizer_utils.py`, `meta_template.py`, `meta_token_init.py` |
| VARIANT trainers (not the paper method) | `grpo_v2.py`, `grpo_clean.py`, `verl_gdpo*.py`, `verl_reward.py`, `meta_rod*_trainer.py`, `meta_opd_trainer.py`, `meta_rlsd_trainer.py`, `contrastive_meta_rlsd_trainer.py`, `meta_rlsd_data_pipeline.py` |
| DEAD / probe-only | `bci_agent_loop.py`, `cf_*_agent.py`, `meta_inject.py`, `meta_quality.py`, `*_processor.py`, `redirect_*.py`, `segment_loss_mask.py`, `self_distill_data.py` |

## Where to look for canonical descriptions

`experiments/configs/science/rl_pmishift.yaml` and `experiments/README.md`
describe this same run in clean, already-de-cluttered form. When the root yamls
and the science configs disagree, the science configs are the intended spec; the
root yamls are the actual historical launch scripts.

## Archived, not deleted

`runs/archive/` = old amlt launchers (ROD/OPD/RLSD/GDPO/e4-e9 lines, metacognition
A100 launchers, triobj v2-v3 and intermediate v4 stages, decoy/asymcf/weight-soup
probes, superseded eval one-offs). `configs/archive/` = their hydra configs.
Nothing was removed — `git log --follow <archived-file>` for its history, and any
archived run is reproducible by its original path.
