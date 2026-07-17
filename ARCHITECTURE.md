# ARCHITECTURE — metacognition-math

> **START HERE.** This file maps the one live method, its exact file spine, and
> where the superseded variants and archived reward modes now live. Everything
> that is not on the spine below is history, kept for reproducibility under
> `runs/archive/` and `configs/archive/`.

## The one path that matters

The live method is **PMI-shift metacognitive self-distillation**: a teacher-free
self-distillation reward applied to the meta tokens. The current live experiment
is the **RQ3 MATCHED LADDER** (2026-07-11): a 4-arm matched RL comparison on the
real `Qwen/Qwen3-8B-Base` substrate — B0 (no-meta gold SFT init + VANILLA_GRPO),
B2 (meta SFT init + VANILLA_GRPO), B3pkg (same meta SFT init + TRIOBJ_DCPO_V4
full package, region-split, rmeta=pmi_shift — corrected 2026-07-12 from the
failed pmi-only strip), and B3-noPMI (identical to B3pkg except
`dcpo_w_meta=0.0` — the minus-one pmi-isolation control). RQ1 = B2−B0 (meta-SFT
effect), RQ2 = B3pkg−B2 (meta reward package effect; the pure pmi isolation is
B3pkg−B3-noPMI). Everything else in the
tree is history — superseded variants and probes, kept for reproducibility under
`runs/archive/` and `configs/archive/`.

```
SFT      h100std_sft_b0_gold.yaml        → configs/sft_b0_gold.yaml        → models/b0_gold_sft        (B0 init, no-meta gold)
         h100std_sft_b23_unmasked.yaml   → configs/sft_b23_unmasked.yaml   → models/b23_rv_unmasked_sft (B2/B3 init, meta RV unmasked)
   │  src/training/sft.py  (wrong_prefix segment-mask)
   ▼
LAUNCH   h100std_rq3_b0.yaml   (B0: no-meta init + VANILLA_GRPO)
         h100std_rq3_b2.yaml   (B2: meta init  + VANILLA_GRPO)
         h100std_rq3_b3.yaml   (B3pkg: meta init + TRIOBJ_DCPO_V4 FULL package — w_meta 0.8/w_format 0.35/w_emit 0.1/w_cal 0.3/len 0.08, w_over=0, rmeta=pmi_shift; ⚠️ 2026-07-12 corrected — the pmi-only strip failed)
         h100std_rq3_b3nopmi.yaml (B3-noPMI: same as B3pkg with only ++algorithm.dcpo_w_meta=0.0 — pmi-isolation control)
   │  amlt → python -m src.training.verl_sdc
   ▼
CONFIG   configs/base_matched_grpo_h100_4x4k.yaml       (B0/B2)
         configs/triobj_dcpo_v4_stage3b_h100_4x4k.yaml  (B3pkg/B3-noPMI)
         parent: configs/verl_e4_selfdistill_h200_4x4k.yaml
   ▼
TRAINER  src/training/verl_sdc.py           entry + GDPO trainer (monolith)
         src/training/verl_sdc_utils.py     region masks / advantage / length cost
   ▼
REWARD   src/training/dcpo_pmi_shift.py     ★ the paper's reward
         src/training/dcpo_region.py        meta-region routing (where reward lands)
         src/training/rewards.py            correctness + meta shape/penalty heads
```

The RQ2 decomposition arms `h100std_shiftonly.yaml` and `h100std_gandhi.yaml`
launch the same spine with the PMI-shift head decomposed into its parts
(pre-rq3 generation; the current live experiment is the rq3 ladder above).

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

## "base" naming — pre-rq3 vs current rq3 ladder (naming note)

**2026-07-11 — the Qwen3-8B-Base redesign has LANDED.** The current rq3 ladder
uses the real pretrained-only `Qwen/Qwen3-8B-Base` as the substrate for all
three arms (see `configs/sft_b0_gold.yaml` / `configs/sft_b23_unmasked.yaml`).
The earlier instruct-substrate generation (pre-rq3) is now an archived
generation.

Historical caveat for old docs/runs: in the **pre-rq3** generation, "base" /
"base_matched" / "qwen3_base_sft" / "basearm" meant the **no-meta CONTROL arm**
SFT'd from `Qwen/Qwen3-8B` (the **INSTRUCT** release) — the same starting model
as the meta arm, minus the `<|meta|>` tokens and the PMI-shift head. When
reading pre-rq3 material, do not read that "base" as the pretrained-only model.
In the current rq3 ladder, by contrast, every arm really does start from
`Qwen3-8B-Base`, and the no-meta control is the **B0** arm.

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

For the current rq3 ladder, `docs/redesign/base_rl_recipe.md` (v2 recipe) and
`docs/redesign/EXPERIMENT_LOG.md` are canonical. For the pre-rq3 generation,
`experiments/configs/science/rl_pmishift.yaml` and `experiments/README.md`
describe that run in clean, already-de-cluttered form. When the root yamls
and the science configs disagree, the science configs are the intended spec; the
root yamls are the actual historical launch scripts.

## Archived, not deleted

`runs/archive/` = old amlt launchers (ROD/OPD/RLSD/GDPO/e4-e9 lines, metacognition
A100 launchers, triobj v2-v3 and intermediate v4 stages, decoy/asymcf/weight-soup
probes, superseded eval one-offs). `configs/archive/` = their hydra configs.
Nothing was removed — `git log --follow <archived-file>` for its history, and any
archived run is reproducible by its original path.
