# Pre-registration — RQ3v2 base replication of the T1 pmi_shift result

**Frozen 2026-07-27, before any base-arm result exists.** Written because this
project has twice chosen a judgement rule after seeing numbers, and because the
same grader flips one axis from **+8.2pp to −0.97pp** depending on whether a
`\boxed{}` wrapper is required. The rule has to precede the data.

Companion artifacts: `docs/manifests/sft2_pair_20260727.json` (what was frozen),
`scripts/freeze_run_manifest.py` (how to regenerate and diff it).

---

## 1. What is being replicated

T1, on an instruct substrate, showed a metacognition reward package beating a
matched control on held-out math: **MATH500 +18.8pp, 6/6 difficulty cells, one
seed**. The package is `pmi_shift` scored on the meta span plus correctness on
the answer span, with auxiliary heads (`w_meta` 0.8, `w_format` 0.35,
`trunc_open_penalty` 0.3, `w_emit` 0.1, `w_cal` 0.3, `len_cost` 0.08).

The base replication runs the same ladder on Qwen3-8B-Base:

| arm | init | SFT2 corpus | RL |
|---|---|---|---|
| `b0p` control | `b0p_v8base_strict_sft` | `v8_base_rv_sft.parquet` (meta-removed twin) | VANILLA_GRPO |
| `b2p` priming | `b2p_v8meta_strict_sft` | `rv_redirect_verify_functional.parquet` | VANILLA_GRPO |
| `b3p` package | same as b2p | same as b2p | triobj with pmi_shift |

Verified identical to T1 at the SFT2 stage: **T1 used this same corpus pair**
(`archive/launchers_pre_rq3/h100std_base_matched_pipeline.yaml` ran its control
SFT2 on `v8_base_rv_sft.parquet`), and all four configs carry the same dose —
3 epochs, lr 1e-5, bs1 × ga4, max_length 4096, save_strategy epoch.

### Known, accepted asymmetry

The trained region — after `sft.py` masks prompt and wrong_prefix — is **29.5%
larger on the meta arm** (378,614 vs 292,353 tokens; mean +48.9, which is the
meta block). transformers 4.57.6 normalises by `num_items_in_batch`, i.e. per
token across the accumulation window, so the consequence is that the meta arm
spends only ~77% of each step's gradient on the shared recovery text where the
control spends 100%. This is inherent to an additive treatment and **T1 carried
it too**, so it is replicated rather than introduced. It is recorded here so it
cannot later be presented as a discovery that explains an inconvenient result.

---

## 2. Three outcomes, declared in advance

Two outcomes would be a trap: a starved run looks like a substrate failure. The
third exists so that starvation cannot be reported as non-replication.

### Outcome A — the mechanism transfers

All of the following, on the frozen grader and the frozen item list:

| # | condition | metric |
|---|---|---|
| A1 | convention installed | `dcpo/meta_emit_rate` ≥ 0.80 sustained, not eroding below 0.80 by gs80 |
| A2 | pmi signal live | `dcpo/pmishift_attempted_rate` ≥ 0.30 and `dcpo/pmishift_n_save` > 0 |
| A3 | package beats control | held-out MATH500 `b3p − b0p` ≥ **+10pp** |
| A4 | pmi adds over priming | `b3p − b2p` ≥ **+3pp**, bootstrap CI lower bound > 0 |
| A5 | not a formatting effect | A3 holds under **both** `format_fair` and `strict_boxed` |
| A6 | not a length effect | after length correction, ≥50% of the raw effect survives |

### Outcome B — the effect was substrate-specific

Diagnostics **healthy** (A1 and A2 both hold for the full run) **and**
`b3p − b2p` bootstrap CI upper bound < +3pp **and** `b3p − b0p` clearly below
+10pp. Healthy diagnostics are required: without them this is Outcome C.

### Outcome C — invalid run, no claim about the substrate

Any of, at gs25: `dcpo/meta_emit_rate` < 0.80, or
`dcpo/pmishift_attempted_rate` < 0.10, or `dcpo/pmishift_n_save` = 0, or
`actor/entropy` < 0.1. The reward was starved or inert. **Abort, fix upstream,
relaunch.** Reporting this as "replication failed" is the pre-registered error —
the previous b3p attempt was in exactly this state.

### Inconclusive

If the effect estimates fall below threshold but the bootstrap intervals are
wide enough to contain both A and B, the result is **inconclusive**, not
negative.

---

## 3. Power, and what may not be claimed

One seed, 500 MATH500 items. Adequate for a 15–20pp effect, **marginal for 5pp**.
Six difficulty cells give ~83 items each.

- **No per-cell claims.** T1's "6/6 cells" cannot be replicated at this n.
- **No claim about Qwen3-8B-Base in general.** One seed measures one
  optimisation trajectory. The conclusion is phrased "in this run, under the
  pre-registered conditions".
- Estimating a 5pp effect credibly would need ~1,500–2,000 paired items; a 3pp
  exclusion threshold ~3,000.

---

## 4. Frozen grader

`format_fair` — `experiments/analysis/analysis_common.py::Grader.grade`: use the
last `\boxed{}` when present, otherwise grade the runtime `answer_extracted`
through the same `robust_grade` path. It exists because arms box at different
rates (one T1 arm boxed ~85% of GSM8K against ~100% for others), so requiring
`\boxed{}` is a per-arm format bias rather than a neutral stricture.

`strict_boxed` (no fallback) is reported **alongside**, never instead. A5 exists
because the axis-A sign inversion between the two graders is the reason this
section is here at all.

Module hash at freeze time is recorded in the manifest. If it changes before
judgement, say so.

---

## 5. Live-run readouts

Readable within the first ~50 RL steps, so a bad run is caught before it is
expensive. All keys verified present in `src/training/verl_sdc.py`.

| metric | intent realised | intent NOT realised |
|---|---|---|
| `dcpo/meta_emit_rate` | ≥0.80, flat | drifting down → RL is eroding emission before the reward pays |
| `dcpo/pmishift_attempted_rate` | 0.30–0.50 | <0.10 → pmi head sees almost nothing; starved |
| `dcpo/pmishift_n_save` | >0 and rising | 0 → the reward never fires |
| `dcpo/pmishift_n_derail` | ≪ n_save | ≥ n_save → the reward is moving evidence the wrong way |
| `dcpo/wellformed_rate` | stable | collapsing → form breaking under RL (the 0712 failure) |
| `dcpo/acc_with − acc_without` | > 0 and holding | ≤0 → meta blocks are not helping the answer |
| `actor/entropy` | 0.1–1.8 | <0.1 → collapse; >1.8 → no learning |
| `actor/pg_clipfrac` | <0.2 | ≥0.2 → step size fighting the ratio clip |
| response length | flat | inflating → length is buying the reward, cf. A6 |

Two of these encode failures this project has already had: emission eroding
under RL while the behaviour looks fine, and the form being emitted without the
behaviour.

---

## 6. What would most likely fool us

A change that reaches **one arm only**. It does not announce itself: both arms
boot, both reward curves look plausible, and the result — positive or negative —
is about the drift. Within a single day this project produced an asymmetric
ladder, an OOM fix ported to one arm and not its twin, a durability fix that
landed only in launchers retired hours later, and a save cadence that differed
between arms with nothing declaring it.

The countermeasure is `scripts/freeze_run_manifest.py`, run before results are
looked at and `--compare`d afterwards. Its output for the SFT2 pair shows
exactly four config differences — dataset, init, output dir, run name — and any
fifth is the failure this document exists to catch.
