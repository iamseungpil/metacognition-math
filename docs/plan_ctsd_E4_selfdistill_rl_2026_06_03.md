# PLAN — E.4 Self-Distill RL with Contrastive Teacher (4-arm amlt H200, LOCKED 2026-06-03)

> **Origin.** E.3 ([[e2-steering-probe]]) inference A/B ranked contrastive-teacher DIRECTIONS:
> gold_stance ≈ gold_decoy (+0.046, underpowered) > cautious > gold_conf_down (null) > conf_down(−).
> gold_stance = grounded verification-process, answer CANCELS (no leak), best self-consistency,
> qualitatively executes concrete alternative-method verification. Inference-steering is a WEAK
> proxy; the real test is whether SELF-DISTILL RL with these teacher directions improves Meta-CoT.
> E.4 runs the RL A/B on the cluster. Includes gold_conf_down (user call): its inference-null was
> on ACCURACY; its real target is CALIBRATION (gold-grounded skepticism = anti-overconfidence =
> the project's ECE north-star), so it is measured on ECE, not only accuracy.

## Intent
Find whether a contrastive-teacher direction, used as verl_sdc advantage-shaping on the meta region
(sign from RLVR correctness), makes Meta-CoT > Base SFT (CLAUDE.md north-star) — and which direction.

## Arms (4 amlt H200 jobs, one YAML / 4 jobs, identical data/steps/eval, differ ONLY in teacher)
| job | sdc teacher | β_contrast | measures |
|---|---|---|---|
| `e4_baseline` | none (plain RLVR / GRPO) | 0 | CONTROL (non-negotiable) |
| `e4_gold_decoy` | T+=prompt+gold, T−=prompt+decoy | >0 | accuracy (answer axis) |
| `e4_gold_stance` | T+=prompt+gold+CAUTIOUS, T−=prompt+gold+CONFIDENT | >0 | accuracy (verify-process, no leak) — the LEAD |
| `e4_gold_conf_down` | T+=prompt+gold+"confidence: 0.15", T−=prompt+gold+"confidence: 0.95" | >0 | CALIBRATION/ECE (anti-overconfidence) |

- mode = `ROD_MQ_CONTRAST` (NOT _INJECT — force-inject is NotImplemented; v8_strict self-emits meta ~32%, train on self-emitted meta).
- sign from RLVR correctness (RLVR-invariant); teacher modulates meta-region advantage MAGNITUDE only.
- CAUTIOUS/CONFIDENT suffixes per E.3. The gold_stance/gold_conf_down "gold both sides" → answer cancels in T+−T− → low leak.

## Implementation (verl_sdc — modifiable; rewards.py is protected, import only)
- Add `sdc_contrast_variant ∈ {decoy, stance, conf}` controlling how `_build_teacher_logprob_batch`
  builds the T+ and T− contexts (suffix pair). `decoy` = existing path. `stance`/`conf` = gold on
  BOTH sides + a stance/confidence suffix that differs. Keep `decoy` byte-identical.
- Turn on β_contrast (config), keep α_attr as configured. baseline = β=0 + teacher off (or plain GRPO config).
- Do NOT touch the force-inject path (stays NotImplemented; we use self-emitted meta).

## amlt config (adapt h200_*verl*.yaml templates)
- target sing / msrresrchbasicvc / sku 141G4-H200 / sla_tier Basic / max_run 7d.
- code via GH release tar (CODE_TAR_REVISION) — a packaging helper script git-archives + `gh release`.
- wandb: WANDB_PROJECT (skilldiscovery2 or metacot-math — match existing), WANDB_RUN_GROUP=e4_self_distill,
  distinct WANDB_NAME per job, WANDB_TAGS=CTSD,E4,self_distill,<variant>. So all 4 overlay in one dashboard.
- init from v8_strict cold-start (SFT_V8_STRICT). Eval GSM8K/MATH/AIME + ECE every N steps to wandb.

## Process loop (implement → module-test → smoke → launch → monitor+intent → fix)
1. Implement verl_sdc 4-variant teacher + verl config + amlt 4-job YAML + packaging helper.
2. MODULE TESTS (local, no cluster): each variant builds the correct T+/T− contexts; β active; advantage
   shaping applies to meta region only; sign from correctness; decoy path unchanged.
3. SMOKE (1 H200 node, short — few steps): trains without crash; teacher signal non-trivial; meta shapes
   toward verification (intent check). GATE before the 4-job launch.
4. LAUNCH 4 jobs (amlt run, Basic). 5. MONITOR wandb: reward, meta-quality (verify rate), eval acc + ECE
   vs Base SFT; INTENT-check (is the teacher shaping meta? is Meta-CoT approaching/beating Base SFT?).
6. FEEDBACK (intent first, not just errors) → fix → re-implement loop.

## Verifiable criteria
- SMOKE PASS: job runs ≥ a few steps, loss/reward finite, β-contrast term non-zero, meta-region advantage
  modulation active, ≥1 logged verification-meta shaped. Else fix before scaling.
- H-RL: e4_gold_stance eval accuracy > e4_baseline AND ≥ Base SFT (north-star); e4_gold_conf_down ECE <
  e4_baseline ECE (calibration) at ≥ baseline accuracy.
- KILL: if after the smoke + early monitoring NO teacher arm beats baseline on its target metric → the
  meta-teacher lever is dead → stop (save compute), report definitively.

## Submission GATES (need user / setup before cluster launch)
- amlt project init + Azure auth in this dir (`amlt project ...`); CLI = envs/amlt/bin/amlt.
- GH release of the code tar; set CODE_TAR_REVISION in the YAML.
- Confirm local-vs-1-node for the smoke (local verl/ray env availability TBD).
Tokens (HF/GH/WANDB) live ONLY in .env; never commit; scan diffs for hf_/ghp_/2f4e6278.
