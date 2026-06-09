# TRIOBJ_DCPO_V2 — DCPO-style 3-Region Token-Masked Advantage Routing

Status: DESIGN v2 (CONVERGED; surgical, additive). Date: 2026-06-09.
Owner: metacognition-math / verl_sdc training path.
Predecessor: `TRIOBJ_META_V1` (env-reward-only tri-objective GDPO, `verl_sdc.py:406-410`).

This spec defines a NEW additive verl mode `TRIOBJ_DCPO_V2`. It MUST NOT alter the
behavior of any existing mode. Every change is gated behind `sdc_mode == "TRIOBJ_DCPO_V2"`
or a new `REWARD_CONFIGS` key, exactly like the existing additive-mode pattern
(`verl_sdc.py:56-57`, `:414-425`).

**v2 converged decisions (vs v1):** EXACTLY 3 region-routed reward heads (no keyword/localize
gate in the reward); R_meta is a flat-`+1.0` transition table with a `+eps` warrant-gated
no-harm bonus and a warmup-only `right→wrong` penalty; R_cal is a per-instance Brier on the
confidence number (RIG demoted to analysis-only); KL/entropy is disabled (Dr.GRPO implicit
reg) to close the global-mask coupling channel; meta tokens route ONLY to META_CONTENT (tag
tokens get advantage 0); CONF token span is computed via an exact char-span→token-span offset
table; all keyword/structure/floor/count guards are ANALYSIS-ONLY wandb metrics, except two
canaries promoted to early-stop/weight-clamp triggers.

**Scope statement (v2):** this design measures **metacognition that produces a visible answer
revision** — the reward signal is defined over the `answer1→answer2` transition. In-line
verification that *prevents* an error before any boxed answer is written (no `wrong→right`
trace because the first draft was already steered correct) is NOT captured by the transition
reward. To keep the design's claimed scope ("reward meta when it is useful") honest rather
than silently narrowed to the self-correction sub-case, §5/§8 add an ANALYSIS metric for
in-line verification (meta that fixed an error before `\boxed{a1}`), and §2.5 specifies a
counterfactual meta-ablation cross-check that audits whether the transition proxy actually
tracks causal meta-utility.

---

## 1. Goal + how it fixes v1's 4 failure causes

### Goal
Give each of the three objectives — **correctness**, **meta-utility**, **calibration** —
its own per-region group-normalized advantage, masked onto its own token span, so that
the gradient for one objective never broadcasts onto the tokens of another. This is the
DCPO (block-wise decoupled advantage) idea applied to the two-pass `<|meta|>` policy.

North-star alignment (CLAUDE.md): metacognition is a *means* to accuracy. Correctness stays
dominant on the answer span; meta-utility is rewarded only when it **demonstrably moved or
held a correct answer under genuine (warranted) difficulty**; calibration is a sub-signal
confined to the confidence token. No objective is allowed to crush another via a shared
broadcast advantage.

### v1's 4 failure causes and the v2 fix

| # | v1 failure (TRIOBJ_META_V1) | Root cause | TRIOBJ_DCPO_V2 fix |
|---|---|---|---|
| (a) | Meta tokens collapse (count 924→308, entropy 0.12→0.014) | All heads summed into ONE group-normalized advantage, then broadcast row-uniform across every response token (`core_algos.py:329`, `:451-466`). The large correctness gradient dominates the few meta tokens. | **Per-region routing.** R_corr masked to the ANSWER span (all non-meta tokens). R_meta masked to META_CONTENT **only** (open/close TAG tokens get advantage 0). The correctness gradient never reaches a meta-content token. (§2, §2.3.) |
| (b) | "Never revise" is the safe optimum | v1's meta head paid flip-credit but nothing for *holding* a correct answer / *attempting* under difficulty, so not revising dominated. | **`+eps` warrant-gated no-harm bonus** on `wrong→wrong` and (documented) `right→right`, paid ONLY when the GROUP is in the warranted difficulty band; flat `+1.0` flip credit. The §2.4 eps-balance bound proves `+eps` can never out-earn solving. (§2.2.) |
| (c) | Blanket meta penalty suppresses meta on hard problems | Inherited `meta_penalty` / `meta_floor` heads paid −0.20/−0.50 for "no meta" regardless of warrant. | These heads are **DISABLED** in this mode (absent from `funcs`/`keys`). The ONLY penalty in R_meta is `right→wrong`, scaled by `w_warmup`. No "no-meta" penalty exists. (§2.2, §3.6.) |
| (d) | Single summed broadcast advantage; KL anchors meta tokens to reference regardless of routing | The whole GDPO design sums per-head normalized advantages into one tensor (`core_algos.py:461-466`) then broadcasts; KL/entropy applied over the GLOBAL response mask re-couples regions. | **Advantage is the masked per-region tensor**, NOT the summed scalar. **AND** `use_kl_loss=false` + `use_kl_in_reward=false` (Dr.GRPO implicit regularization) — this closes the KL/entropy-over-global-mask coupling channel the gradient review flagged. (§2.3, §2.6.) |

---

## 2. Architecture

### 2.1 Region partition (one rollout, two-pass format)

```
[ think + answer1(\boxed) ]  <|meta|> [ review/redirect + conf:0.NN ]  <|/meta|>  [ answer2(\boxed) ]
└──────── ANSWER span (A) ──┘  TAG    └────────── META_CONTENT (M) ───────────┘   TAG   └ ANSWER (A) ┘
                                                          └ CONF tokens (K) ┘
```

- **A = ANSWER** = `response_mask ∧ ¬META_BLOCK`, where `META_BLOCK` is tag-inclusive (open
  token .. close token). The pre-meta reasoning, `\boxed{a1}`, post-meta reasoning, and
  `\boxed{a2}` ALL belong to A. R_corr routes to A and **never** touches a meta-content token.
- **M = META_CONTENT** = the tokens strictly INSIDE the block, i.e. `META_BLOCK` MINUS the
  open(`151669`) and close(`151670`) TAG tokens. R_meta routes to M ONLY. **The open/close
  tag tokens get advantage 0** — they are pure delimiters and belong to neither A nor M.
- **K = CONF** = the contiguous token run of the numeric literal after `confidence:`/
  `probability:` **inside M only** (multi-token, e.g. `0`,`.`,`88`). `K ⊆ M`. R_cal routes to K.

Token ids (verified, `/home/v-seungplee/sft_e20a_local/added_tokens.json`,
`tests/test_meta_inject.py:11`): `META_OPEN=151669`, `META_CLOSE=151670`. Confidence is
NOT a special token — it is plain text parsed by regex (`rewards.py:701-712`).

### 2.2 The three region rewards (raw per-rollout scalars)

Reuse helpers from `rewards.py` / `meta_revision_rewards.py`:
`_check_correctness` (`rewards.py:27`), `_boxed_matches`/`_BOXED_RE` (`meta_revision_rewards.py:61-68`),
`_parse_confidence` (`rewards.py:701`). **NOTE (v2):** `_has_genuine_meta` and
`_meta_localizes_error` are NO LONGER reward inputs — the keyword/localize GATE is DROPPED
from the reward entirely (it was the hack surface; group-relative advantage + warrant + small
eps do the work). They survive only as ANALYSIS metrics (§5, §8).

`answer1` = FIRST `\boxed` (pre-meta draft). `answer2` = LAST `\boxed` (graded answer).
GRPO group `G` = the `rollout.n` rollouts sharing one prompt. `i` = rollout index in `G`.

**R_corr (ANSWER span A):**
```
R_corr(i) = +1 if _check_correctness(answer2_i, gt) else -1     # answer2 = last \boxed
```

**R_meta (META_CONTENT span M):** flat-+1 transition table, group-warrant gated.
```
p̂  = mean_{j∈G} _check_correctness(answer1_j, gt)     # group preliminary pass-rate (difficulty)
warranted = (p_lo ≤ p̂ ≤ p_hi)                         # p_lo=0.2, p_hi=0.8
c1, c2    = _check_correctness(answer1_i, gt), _check_correctness(answer2_i, gt)

#   transition         payoff
#   wrong → right      +1.0                            # flat flip credit (NO genuine/localize split)
#   wrong → wrong      +eps   if warranted else 0.0    # no-harm under genuine difficulty
#   right → right      +eps   if warranted else 0.0    # held a correct answer under difficulty (see note)
#   right → wrong      -1.0 * w_warmup                 # the ONLY penalty

if   (not c1) and c2:           R_meta = +1.0
elif (not c1) and (not c2):     R_meta = eps if warranted else 0.0
elif c1 and c2:                 R_meta = eps if warranted else 0.0
else:  # c1 and not c2          R_meta = -1.0 * w_warmup
```
`eps = 0.1`. `w_warmup = min(1, step / warmup_steps)`, `warmup_steps = 200`.

The `+eps` is paid ONLY when `warranted` (group `p̂ ∈ [0.2,0.8]`): NOT on easy groups
(`p̂ > 0.8`) and NOT on hopeless groups (`p̂ < 0.2`). On unwarranted groups every no-harm
transition pays 0.

> **`right→right +eps` decision (REQUIRED documentation / open knob `dcpo_eps_right_right`).**
> On its face, paying `+eps` for a no-op meta block on an already-correct problem rewards
> non-useful meta. v2 pays it ONLY under the warrant band (medium-difficulty group), reading it
> as **"held a correct answer under genuine difficulty"** — i.e. the meta block reviewed and
> *did not corrupt* a correct draft on a problem where the group as a whole was at risk. This is
> a deliberate, narrow credit, NOT a blanket "meta is good" payment. **If the canaries in §5
> (boilerplate-repetition rate, sandbagging) trend up, or the in-line-verification analysis shows
> `right→right +eps` is just paying for decorative no-op meta, set `dcpo_eps_right_right=0`**
> (eps=0 for `right→right`, reward meta only when it demonstrably *moved* an answer
> `wrong→right` or *held* it under difficulty `wrong→wrong`). The knob defaults to ON (eps paid)
> but is the first lever to flip if §5 triggers fire.

**R_cal (CONF token K):** per-instance BRIER on the parsed confidence.
```
conf_i = _parse_confidence(meta_text_i)                # in [0.01,0.99], or None
R_cal(i) = -(conf_i - c2)^2     if conf_i is not None  # c2 = correct(answer2) ∈ {0,1}
           0.0                  if conf_i is None       # NO floor penalty
```
**RIG / group-difficulty anchor is REMOVED from the reward** (v1 used a group-RIG blend).
`group_acc` and difficulty buckets are computed for ANALYSIS ONLY (§5), never as a reward term.

### 2.3 Per-token advantage composition formula

Each raw head is group-normalized **independently and over its own group** (Dr.GRPO block-wise:
subtract group mean, NO `/std`), NOT summed-then-normalized:
```
Â_corr(i) = R_corr(i) - mean_G R_corr            # Dr.GRPO (no /std), dr_grpo=True
Â_meta(i) = R_meta(i) - mean_G R_meta            # own baseline; all-zero group → 0 gradient
Â_cal(i)  = R_cal(i)  - mean_G R_cal
```
Each `Â_*(i)` is a per-rollout scalar `[B,1]`. Routed per token `t`:
```
A_token[i,t] = (
      w_corr * Â_corr(i) * 1[t ∈ A]          # correctness on all non-meta tokens (a1, a2, all reasoning)
    + w_meta * Â_meta(i) * 1[t ∈ M]          # meta-utility INSIDE <|meta|>…<|/meta|> content ONLY
    + w_cal  * Â_cal(i)  * 1[t ∈ K]          # calibration on the confidence number tokens
    ) * response_mask[i,t]
```
Defaults `w_corr=1.0, w_meta=0.5, w_cal=0.3`. **Open/close TAG tokens and any token outside
`A∪M` get advantage 0** (tags are pure delimiters). The summed scalar (`combined`/`rm_scores`,
`verl_sdc.py:1413-1416`) is left untouched and feeds only logging; it is NOT the advantage
source, and (per §2.6) it is NOT fed into KL either.

### 2.4 eps-balance bound (ASSERTION, not a defaults assumption)

The `+eps` on no-harm transitions must NEVER, after group-normalization and `w_meta`, dominate
the `R_corr` (w_corr=1.0) incentive to actually solve. **Bound (verified by unit assertion):**

- Max achievable post-norm META advantage from a `wrong→wrong` (or `right→right`) rollout is
  bounded by the head's group-mean-subtracted range. With payoffs in `{-1·w_warmup, 0, +eps, +1}`,
  the centered META advantage magnitude is `≤ (1 + eps)` (worst case a single `+1` flip vs a
  group of `0`/`+eps`), and a *no-harm* rollout's own centered value is `≤ eps`. After `w_meta`:
  `w_meta · eps = 0.5 · 0.1 = 0.05`.
- A correct vs incorrect answer earns centered R_corr magnitude up to `w_corr · 1.0 = 1.0`
  (R_corr ∈ {−1,+1}, post-center range up to 2, conservatively ≥ 1.0 when the group is mixed).
- **Bound asserted:** `w_meta · eps  <  w_corr · 1.0` strictly, with margin
  `1.0 − 0.05 = 0.95`. Equivalently `eps < w_corr / w_meta = 2.0`; the chosen `eps=0.1` sits
  20× under the cap. The unit check (§6.2) asserts `w_meta * eps < w_corr` AND, on a constructed
  group, that the total per-rollout advantage of a *staying-wrong* rollout never exceeds that of
  a *became-right* rollout. This is a hard `assert`, NOT a comment.

### 2.5 Counterfactual meta-ablation cross-check (audited, OUT of reward)

The transition table is a **proxy** for causal meta-utility: it assumes a `wrong→right` flip was
*caused* by the meta block. To confirm the proxy is tracking causal meta-utility and not
second-draw noise, periodically (every `dcpo_cf_audit_every` steps, default 50; on a fixed
held-out audit minibatch) re-decode each rollout with the **meta block masked** (decode the
post-meta continuation conditioned as if `<|meta|>…<|/meta|>` were absent / replaced by a
neutral no-op) and recompute correctness → `c2_ablated`.

- Log `Δ_cf = mean(c2 − c2_ablated)` (counterfactual meta-utility) alongside the
  transition-credit signal `Δ_2att = mean(R_meta over the audit batch)` (2-attempt transition
  credit). Log both to wandb as `audit/meta_cf_delta` and `audit/meta_2att_credit`, plus their
  per-difficulty-bucket breakdown and the correlation `corr(Δ_cf, Δ_2att)`.
- **This is an audited cross-check, NOT a reward term** (the policy is not trained on `Δ_cf`).
- **Escalation rule:** if `corr(Δ_cf, Δ_2att)` drops below `dcpo_cf_corr_floor` (default 0.3)
  for `dcpo_cf_audit_patience` consecutive audits (default 3) — i.e. the transition proxy and
  the counterfactual diverge — escalate to using the **counterfactual delta as the meta reward**
  (set `R_meta(i) = c2_i − c2_ablated_i`, gated by the same warrant). This escalation is a
  documented, pre-registered switch (`dcpo_meta_reward=counterfactual`), not an ad-hoc change.

### 2.6 KL / entropy decoupling (the v1-(d) fix)

Set, for this mode only:
```
algorithm.use_kl_loss      = false
algorithm.use_kl_in_reward = false
```
Rationale: with KL active over the GLOBAL response mask, meta tokens are anchored to the
reference policy **regardless of region routing** (the gradient review's coupling channel) —
the per-region advantage decoupling is undone by a per-token KL that ignores regions. Dr.GRPO's
implicit regularization (no `/std`, group-mean-subtract) replaces explicit KL. With both flags
false, the only per-token signal on a meta-content token is `w_meta · Â_meta`, as intended.

### 2.7 Token-region mask util (new)

`build_dcpo_region_masks(resp_ids, response_mask, decode_fn, meta_open=151669, meta_close=151670)`
→ returns `{META_BLOCK, META_CONTENT, CONF, ANSWER_REGION}`, all `[T]` bool over response
positions, where `META_CONTENT = META_BLOCK ∧ ¬(open|close tag)`. Algorithm in §4. Reuses the
open/close scan from `meta_inject.meta_mask` (`meta_inject.py:54-71`, including the
unclosed-to-end branch) and `_parse_confidence` for the CONF char-span (mapped to a token span
via §4 Pass B's exact offset table).

---

## 3. EXACT files to touch + change points

All edits are additive and mode-gated. Cited lines are from the verified traces.

### 3.1 NEW module: `src/training/dcpo_region.py`
- `build_dcpo_region_masks(...)` — token-region mask util (§4). Pure python + numpy/torch,
  importable under system python3 (no transformers needed if `decode_fn` is injected). Emits
  `META_CONTENT` with tag tokens EXCLUDED (advantage 0).
- `dcpo_region_rewards(completions, ground_truth, group_index, step, **cfg)` — computes the
  three raw scalars `R_corr/R_meta/R_cal` per rollout (§2.2). Imports `_check_correctness`,
  `_BOXED_RE`, `_parse_confidence`; does NOT duplicate them. Does NOT call `_has_genuine_meta`/
  `_meta_localizes_error` in the reward path (analysis-only, §5).
- `dcpo_meta_counterfactual(...)` — §2.5 audit helper (re-decode with meta masked, returns
  `c2_ablated`). Called only on audit steps, behind `dcpo_cf_audit_every`.

### 3.2 `src/training/verl_sdc.py`
- **REWARD_CONFIGS** (`:68-411`): ADD after the `TRIOBJ_META_V1` entry (`:406-410`):
  ```python
  "TRIOBJ_DCPO_V2": {
      "funcs":   [correctness_reward, meta_region_utility_reward, cal_region_reward],
      "weights": [1.0, 0.5, 0.3],
      "keys":    ["correctness", "meta_region_utility", "cal_region_reward"],
  },
  ```
  EXACTLY 3 heads. `meta_region_utility_reward` and `cal_region_reward` are thin wrappers
  around `dcpo_region_rewards` exposing the per-head scalar each (GDPO asserts every key in
  `gdpo_reward_keys` exists in `non_tensor_batch`, `:609`). The wrappers also write group-level
  `p̂` / `group_acc` per-rollout so the advantage path can read warrant (+ analysis difficulty).
- **`_VANILLA_MODES`** (`:419`): do NOT add `TRIOBJ_DCPO_V2`. Add it to a NEW set
  `_REGION_ROUTED_MODES = {"TRIOBJ_DCPO_V2"}` and in `_attach_teacher_signals` short-circuit the
  teacher-forward (return after building masks, before T+/T−/position forward). Keep
  `_VANILLA_MODES` byte-identical.
- **KL flags** (algorithm config / trainer wiring): for `TRIOBJ_DCPO_V2`, force
  `algorithm.use_kl_loss=false` and `algorithm.use_kl_in_reward=false` (§2.6). Set in the yaml
  (§3.4); assert at boot that both are false for this mode.
- **Mask stacking** (`:1355-1376`): ADD a parallel block (gated `if mode in _REGION_ROUTED_MODES:`)
  that calls `build_dcpo_region_masks` and stacks `dcpo_answer_mask / dcpo_meta_content_mask /
  dcpo_conf_mask` into `data.batch`. Reuse the `_pad` helper (`:1361-1365`). Also stack the
  per-rollout group scalars `dcpo_phat` and `dcpo_group_acc` into `non_tensor_batch`.
- **EOS reward placement** (`:1409-1412`): UNCHANGED. `combined`/`rm_scores` is logging-only;
  with KL-in-reward off it does NOT feed the reward either.

### 3.3 `src/training/verl_sdc_utils.py` — advantage path
- **`compute_sdc_gdpo_advantage`** (`:232`): ADD a NEW branch BEFORE the existing early-return
  OR-clause (`:314-317`):
  ```python
  if sdc_mode == "TRIOBJ_DCPO_V2":
      return _compute_dcpo_region_advantage(base_advantages, response_mask, batch,
                                            non_tensor_batch, index, config)
  ```
  Purely additive — the existing OR-clause (`:314`) and teacher-mode factor path (`:575-582`)
  are untouched.
- **NEW function `_compute_dcpo_region_advantage(...)`**:
  1. Read three heads from `non_tensor_batch["correctness"/"meta_region_utility"/"cal_region_reward"]`.
  2. Group-normalize each independently with `index=uid` (Dr.GRPO: subtract group mean, no /std;
     `clamp_min` degenerate groups like `core_algos.py:457`). → `Â_corr,Â_meta,Â_cal` `[B,1]`.
  3. Read region masks `dcpo_answer_mask / dcpo_meta_content_mask / dcpo_conf_mask` `[B,T]`.
  4. Compose: `A = (w_corr*Â_corr*ans + w_meta*Â_meta*meta_content + w_cal*Â_cal*conf) * response_mask`
     (§2.3). TAG tokens are in NEITHER `ans` nor `meta_content` → advantage 0. Do NOT re-whiten
     globally (codex-r13 LOCK, `:583-586`).
  5. Return `(A, A)`.

### 3.4 NEW config yaml: `configs/triobj_dcpo_v2_h100_4x4k.yaml`
- Copy the `TRIOBJ_META_V1` yaml; set `sdc_mode: TRIOBJ_DCPO_V2`, `sdc_enabled: false`
  (no teacher), `algorithm.adv_estimator: gdpo`, **`algorithm.use_kl_loss: false`**,
  **`algorithm.use_kl_in_reward: false`** (§2.6). Add knobs:
  `dcpo_w_corr=1.0, dcpo_w_meta=0.5, dcpo_w_cal=0.3, dcpo_eps=0.1, dcpo_eps_right_right=true,
  dcpo_p_lo=0.2, dcpo_p_hi=0.8, dcpo_warmup_steps=200, dcpo_dr_grpo=true,
  dcpo_residual_to_answer=true, dcpo_cf_audit_every=50, dcpo_cf_corr_floor=0.3,
  dcpo_cf_audit_patience=3, dcpo_meta_reward=transition`. Data = `data/verl_train_redirect.parquet`.

### 3.5 NEW amlt yaml: `amlt/triobj_dcpo_v2.yaml`
- Clone the existing tri-objective amlt job; point at `configs/triobj_dcpo_v2_h100_4x4k.yaml`;
  msrresrchvc A100×4; tar via `code_snapshots/metacognition.tar.gz` (CLAUDE.md). wandb group
  `TRIOBJ_DCPO_V2`.

### 3.6 EXACTLY-3-heads / disabled inherited heads
- No edit required IF the disabled heads are simply absent from this mode's `REWARD_CONFIGS`
  `funcs`/`keys` (they are). Confirm no global `combined` term in `verl_reward.py:99,147`
  injects `meta_floor`/`meta_count` for this mode; if it does, gate it off for `TRIOBJ_DCPO_V2`.

**DISABLED heads for this mode** (absent from `funcs`/`keys`, no `combined` contribution):
`meta_penalty_reward` (`rewards.py:2313`), `meta_penalty_adaptive_reward` (`:2527`),
`confidence_omission_floor`/`meta_floor` (`rewards.py:1700`; `verl_reward.py:99,147`),
`meta_count_bonus` (`rewards.py:715`), `meta_structure_reward` (`:786`),
`meta_quality_reward` (`:867`), standalone `outcome_calibration_reward`/`calibration_reward`/
`confidence_trajectory_reward`. Keep ONLY `correctness_reward → R_corr`,
`meta_region_utility_reward → R_meta`, `cal_region_reward → R_cal`. The degeneration/no-boxed
filter may remain as a MULTIPLICATIVE filter on R_corr (not an additive head).

---

## 4. Token-region mask algorithm (pseudocode) + edge cases

```
INPUTS: resp_ids[T] (response token ids), response_mask[T] bool,
        META_OPEN=151669, META_CLOSE=151670, decode_fn(ids)->str
OUTPUT: META_BLOCK[T], META_CONTENT[T], CONF[T], ANSWER_REGION[T]  (bool over response)

# Pass A — meta spans (reuse meta_inject.meta_mask scan, meta_inject.py:54-71)
META_BLOCK = META_CONTENT = zeros(T)
spans = []; in_meta=False; open_idx=content_start=None
for i in 0..T-1:
    if not response_mask[i]:
        if in_meta: close_span(i); in_meta=False     # pad while open = truncation
        continue
    t = resp_ids[i]
    if t == META_OPEN:
        if in_meta: close_span(i)                     # EDGE: nested/dup open → force-close prev
        in_meta=True; open_idx=i; content_start=i+1; META_BLOCK[i]=True   # TAG in BLOCK, NOT CONTENT
    elif t == META_CLOSE and in_meta:
        META_BLOCK[open_idx..i]=True
        META_CONTENT[content_start..i-1]=True         # CONTENT excludes BOTH tag tokens
        spans.append((content_start, i)); in_meta=False   # (lo, hi) hi exclusive = close idx
    elif t == META_CLOSE and not in_meta:
        pass                                          # EDGE: stray close → ignore (default)
if in_meta:                                           # EDGE: missing close (truncation)
    last = last index where response_mask==1
    META_BLOCK[open_idx..last]=True; META_CONTENT[content_start..last]=True
    spans.append((content_start, last+1))

# Pass B — confidence run via EXACT char-span -> token-span map (fixes boundary review C1/C2)
#   Do NOT re-scan token-by-token. Use _parse_confidence to get the char span of the matched
#   number, then map char-span -> token-span via ONE cumulative-decode offset table.
CONF = zeros(T)
for (lo, hi) in spans:                                # content token range [lo, hi)
    toks = resp_ids[lo:hi]
    # 4.B.1 build cumulative char-offset table over the content tokens (single pass):
    #   offsets[j] = len(decode_fn(toks[:j]));  char position of token j's surface start
    offsets = cumulative_decode_offsets(toks, decode_fn)   # len = (hi-lo)+1
    text = decode_fn(toks)
    span = _parse_confidence_charspan(text)           # (char_start, char_end) of the number literal; None if no conf
    if span is None: continue
    cs, ce = span
    # 4.B.2 char-span -> token-span: tokens whose [offsets[j], offsets[j+1]) overlaps [cs, ce)
    k0 = first j in [0,hi-lo) with offsets[j+1] > cs
    k1 = last  j in [0,hi-lo) with offsets[j]   < ce
    CONF[lo+k0 .. lo+k1] = True
    # 4.B.3 ASSERT round-trip: decode_fn(resp_ids[lo+k0 : lo+k1+1]) parses back to the same conf
    assert _parse_confidence(decode_fn(resp_ids[lo+k0:lo+k1+1])) == _parse_confidence(text)
    break   # FIRST conf per block only

# Pass C — answer region = response minus the FULL meta block (tag-inclusive complement)
ANSWER_REGION = response_mask & ~META_BLOCK            # tags are NOT in ANSWER

# Routing buckets: A = ANSWER_REGION, M = META_CONTENT (NO tags), K = CONF.
# TAG tokens are in META_BLOCK but NOT in META_CONTENT and NOT in ANSWER_REGION → advantage 0.

# INVARIANTS (assert): CONF ⊆ META_CONTENT ⊆ META_BLOCK ⊆ response_mask;
#   META_CONTENT and ANSWER_REGION are DISJOINT; ANSWER_REGION ∪ META_BLOCK == response_mask;
#   tag tokens ∈ (META_BLOCK \ META_CONTENT \ ANSWER_REGION).
```

`_parse_confidence_charspan` is a thin variant of `_parse_confidence` (`rewards.py:701-712`)
that returns the regex match's `.span()` of the numeric group instead of the float — same
regex, so the same conf parses; the only addition is the char offset.

| Case | Behavior |
|---|---|
| No meta at all | META_BLOCK/CONTENT/CONF all 0; ANSWER_REGION == response_mask (R_corr on everything). |
| Missing close (truncation, 31%) | META_CONTENT runs content_start..last; answer after open empty (no committed a2). No NaN. |
| Multiple meta blocks | each pair its own span; META_CONTENT = union; CONF = first number in FIRST block with a conf. |
| Nested/dup open before close | second open force-closes first span, starts fresh (no overlap). |
| Stray close, no open | ignored (default). |
| Conf multi-token | char-span→token-span map captures whole literal; round-trip assert guards boundaries. |
| Free-text conf outside meta | NOT marked (CONF ⊆ META_CONTENT). |
| Empty meta content (95% pathology) | Â_meta group-mean-subtract → all-zero group contributes 0 gradient; no /0. |
| Pad interleaved | pad never enters/extends a span; 0 in all masks. |
| Tag tokens | in META_BLOCK only → advantage 0 (delimiters). |

---

## 5. Guards — ANALYSIS-ONLY metrics + the two promoted triggers

All guards below are **wandb metrics, NOT reward terms** (the keyword/localize/floor/count gates
are OUT of the reward). Two are PROMOTED from analysis-only to early-stop / weight-clamp triggers
so the loop can react to inversion-style failure modes at ~step 20 (project early-collapse note)
rather than discovering them post-hoc.

**Analysis-only metrics (log, do not act):**
1. **Difficulty buckets** — `p̂` histogram (group preliminary pass-rate); per-bucket R_meta,
   flip-rate, and counterfactual `Δ_cf` (§2.5). Difficulty/RIG anchor is ANALYSIS-ONLY.
2. **Per-region entropy / KL** — entropy of A, M, K spans separately; (KL is off in training but
   still logged vs reference for diagnostics).
3. **`group_acc`** — mean correctness(answer2) per group (for calibration analysis, NOT R_cal).
4. **Genuine/localize rate** — `_has_genuine_meta ∧ _meta_localizes_error` rate (the dropped
   gate, kept as a descriptive metric to confirm warrant+eps reproduces good meta WITHOUT it).
5. **In-line-verification metric (scope guard, REQUIRED).** Fraction of rollouts where the meta
   block prevented an error *before any revision* — operationalized as: `c1==correct` AND the
   meta-ablated decode (§2.5) flips the FIRST committed answer to wrong (`c1_ablated==wrong`),
   i.e. meta fixed/held the draft pre-`\boxed{a1}` with NO `wrong→right` transition trace. Logged
   as `analysis/inline_verify_rate`. This makes explicit that the transition reward measures only
   the *visible-revision* sub-case, and surfaces the in-line-verification sub-case the reward does
   not capture.

**PROMOTED triggers (early-stop / weight-clamp):**
6. **Sandbagging canary → TRIGGER.** `mean_G correctness(answer1)` (mean group correctness of the
   FIRST draft). If it falls below `dcpo_sandbag_floor` (default: `0.6 ×` its first-10-step EMA)
   for `dcpo_sandbag_patience` steps (default 5), the policy is faking early errors to farm flip
   credit → **clamp `w_meta → 0` (weight-clamp) and raise an early-stop alert**. Active from step 1.
7. **Boilerplate-repetition canary → TRIGGER.** Meta-content n-gram repetition rate
   (`dcpo_meta_ngram_rep`, e.g. 4-gram self-repetition within and across meta blocks). If it
   exceeds `dcpo_boiler_ceil` (default 0.5) for `dcpo_boiler_patience` steps (default 5) →
   **clamp `w_meta → 0.5×` (weight-clamp) and alert**; if it persists, early-stop. This catches
   the meta-collapse-into-template inversion mode at ~step 20.

Both triggers write `trigger/sandbagging` and `trigger/boilerplate` events to wandb with the
step, the offending value, and the clamp action taken.

---

## 6. Smoke + unit-test plan

All token-mask and reward unit tests are PURE PYTHON (inject a fake `decode_fn`), runnable under
system `python3`. The verl smoke needs GPU.

### 6.1 Unit — token-region masks (`tests/test_dcpo_region.py`)
- Synthetic id sequences + fake `decode_fn`. Cases: no-meta; single block; missing close; double
  block; nested open; stray close; pad interleaved; conf multi-token (`0`,`.`,`88` and `0.`,`8`,`8`).
- Assert invariants: `CONF ⊆ META_CONTENT ⊆ META_BLOCK ⊆ response_mask`; META_CONTENT/ANSWER
  disjoint and `∪ META_BLOCK == response_mask`; **tag tokens ∈ META_BLOCK but ∉ META_CONTENT and
  ∉ ANSWER_REGION (advantage 0)**; CONF char-span→token-span round-trip assert (§4 Pass B.3) holds.

### 6.2 Unit — region rewards (`tests/test_dcpo_rewards.py`)
- R_corr ±1 on last `\boxed`. R_meta FULL transition table: `wrong→right=+1`, `wrong→wrong=+eps`
  iff warranted, `right→right=+eps` iff warranted (and `=0` when `dcpo_eps_right_right=false`),
  `right→wrong=-1·w_warmup` (warmup ramp), all no-harm `=0` when unwarranted. R_cal `=-(conf-c2)^2`,
  conf-missing→0, NO RIG term.
- Group-normalization: all-equal group → Â=0 (no spurious gradient); single flip stands out.
- **eps-balance bound (§2.4) HARD ASSERT:** `w_meta*eps < w_corr`; AND on a constructed mixed
  group, total per-rollout advantage of a *staying-wrong* rollout `<` that of a *became-right*
  rollout. Test FAILS the build if violated.

### 6.3 Unit — advantage composition (`tests/test_dcpo_advantage.py`)
- Build `[B,T]` masks + `[B,1]` head scalars; assert `A_token` carries Â_corr only on A,
  Â_meta only on META_CONTENT, Â_cal only on K; **TAG tokens = 0**; partition-coverage of
  `(A ∪ META_BLOCK)` over response_mask == 1.0; no NaN with empty meta / empty conf rows.
- Regression: existing modes' path (`verl_sdc_utils.py:314-317`, teacher factor `:575-582`)
  byte-identical (new branch BEFORE the OR-clause, fires only on `TRIOBJ_DCPO_V2`).

### 6.4 Unit — counterfactual audit + canaries (`tests/test_dcpo_audit.py`)
- `dcpo_meta_counterfactual` returns `c2_ablated`; `corr(Δ_cf, Δ_2att)` computed; escalation
  switch flips `dcpo_meta_reward` after `patience` sub-floor audits (mock).
- Sandbagging + boilerplate triggers fire and clamp `w_meta` on synthetic collapse traces.

### 6.5 Step-1 verl smoke (GPU)
- `configs/triobj_dcpo_v2_h100_4x4k.yaml`, `max_steps=1`, tiny batch. Assert: job boots;
  `data.batch` has `dcpo_answer_mask/dcpo_meta_content_mask/dcpo_conf_mask`; advantage tensor
  finite, non-uniform across regions; tag tokens carry 0; `use_kl_loss`/`use_kl_in_reward` both
  false at runtime; `critic/rewards/mean` still logs; existing-mode smokes unaffected.

---

## 7. Risks + mitigations

| Risk | Mitigation |
|---|---|
| **Tokens with zero advantage → zero gradient** (strict routing leaves glue uncredited). | `dcpo_residual_to_answer=true`: ANSWER = `response_mask ∧ ¬META_BLOCK` is a true complement, so every non-meta token is credited by R_corr. ONLY the tag tokens are intentionally 0. Assert partition-coverage (`verl_sdc_utils.py:660`). |
| **Tag tokens at advantage 0** | Intentional (pure delimiters). Asserted in §6.3. They are 2 tokens/block, negligible. |
| **`+eps` out-earns solving** | §2.4 eps-balance bound, asserted in §6.2 (`w_meta*eps=0.05 ≪ w_corr=1.0`). |
| **`right→right +eps` rewards decorative meta** | Paid only under warrant; §5 boilerplate + in-line-verify metrics watch it; `dcpo_eps_right_right=0` is the documented kill-switch (§2.2 note). |
| **Transition proxy ≠ causal meta-utility (second-draw noise)** | §2.5 counterfactual audit logs `Δ_cf` vs `Δ_2att`; escalation to counterfactual-delta reward if `corr` < floor for patience audits. |
| **Independent per-region signs trip the sign-flip watchdog** (`:613-617`). | Redefine the watchdog PER REGION for this mode (compare A_token sign to its OWN region head sign). |
| **KL re-couples regions** (v1-(d)). | `use_kl_loss=false` + `use_kl_in_reward=false` (§2.6); asserted at boot. |
| **Empty meta (95%) / absent conf (31%) → NaN**. | Group-mean-subtract (no /std); all-zero group → 0 contribution, not NaN. |
| **p̂ needs the whole group** | Computed once per group in the reward manager (already groups by uid for GDPO); stacked as `dcpo_phat/dcpo_group_acc` (§3.2). |
| **CONF boundary mis-mapping (review C1/C2)** | char-span→token-span via single cumulative-decode offset table + round-trip parse assert (§4 Pass B). NO token-by-token re-scan. |
| **Breaking an existing mode** | New branch BEFORE the OR-clause, fires only on `TRIOBJ_DCPO_V2`; masks/keys mode-gated; `_VANILLA_MODES` untouched; EOS reward + `combined`/`rm_scores` untouched; KL flags scoped to this yaml. Regression test 6.3 asserts byte-identity. |
| **Scope silently narrowed to self-correction** | §5.5 in-line-verification metric makes the visible-revision scope explicit and surfaces the un-rewarded sub-case. |

---

## 8. What MUST be VERIFIED before training

1. **Token ids** — confirm `convert_tokens_to_ids("<|meta|>")==151669` / `"<|/meta|>")==151670`
   at runtime (lookup like `verl_sdc.py:1624-1625`; hardcode only in unit tests).
2. **Region partition on REAL rollouts** — dump 50 decoded rollouts; assert
   `ANSWER ∪ META_BLOCK == response_mask`, `CONF ⊆ META_CONTENT`, and **tag tokens carry 0**.
3. **Advantage NOT row-uniform** — step-1 smoke: per-region variance of `A_token` > 0 (routing
   happened); tag-token advantage exactly 0.
4. **eps-balance bound holds** (§2.4) — unit assert `w_meta*eps < w_corr` AND staying-wrong <
   became-right per-rollout advantage on a constructed group.
5. **KL fully off** — runtime assert `use_kl_loss is False and use_kl_in_reward is False`.
6. **p̂ / group_acc correctness** — `dcpo_phat` == mean answer1 correctness of the group
   (spot-check 3 groups).
7. **EXACTLY 3 heads / disabled heads absent** — assert `gdpo_reward_keys ==
   {"correctness","meta_region_utility","cal_region_reward"}` and that `meta_penalty`,
   `meta_floor`, `meta_count`, `meta_structure`, `meta_quality` contribute nothing to `combined`.
8. **Existing-mode regression** — `TRIOBJ_META_V1` + `VANILLA_GRPO` smokes: advantage tensors
   byte-identical to pre-change.
9. **Both canary triggers live from step 1** — `mean_G correctness(answer1)` (sandbagging) and
   `dcpo_meta_ngram_rep` (boilerplate) logged AND wired to weight-clamp/early-stop (§5.6, §5.7).
10. **Counterfactual audit wired** — `audit/meta_cf_delta`, `audit/meta_2att_credit`,
    `corr(Δ_cf,Δ_2att)` logged every `dcpo_cf_audit_every` steps; escalation switch reachable.
11. **In-line-verification metric live** — `analysis/inline_verify_rate` logged (scope honesty).
12. **No NaN/Inf** — first 5 steps finite, esp. on truncated/empty-meta batches.

---

### Files referenced (absolute)
- `/home/v-seungplee/metacognition-math/src/training/verl_sdc.py` — REWARD_CONFIGS `:68-411` (TRIOBJ_META_V1 `:406-410`), `_VANILLA_MODES :419`, always-emit `:609-669`, mask stacking `:1355-1376`, EOS reward `:1409-1412`, combined/rm_scores `:1413-1416`, token-id lookup `:1624-1625`, patched_compute_advantage `:2044-2079`.
- `/home/v-seungplee/metacognition-math/src/training/verl_sdc_utils.py` — `compute_sdc_gdpo_advantage :232`, base GDPO call `:281`, early-return OR-clause `:314-317`, seq_adv/sign `:328-330`, teacher factor `:575-582`, codex-r13 LOCK `:583-586`, sign-flip watchdog `:613-617`, coverage metric `:660`.
- `/home/v-seungplee/metacognition-math/src/training/meta_revision_rewards.py` — `_BOXED_RE :61`, `_has_genuine_meta :71` (ANALYSIS-ONLY now), `_meta_localizes_error :86` (ANALYSIS-ONLY now), `meta_revision_utility_reward :97`, piecewise `:125-142`.
- `/home/v-seungplee/metacognition-math/src/training/rewards.py` — `_check_correctness :27`, `_extract_answer_fallback :71`, `_last_confidence :89`, `_parse_confidence :701-712` (+ new `_parse_confidence_charspan` variant), disabled heads `meta_count_bonus :715`, `meta_structure_reward :786`, `meta_quality_reward :867`.
- `/home/v-seungplee/metacognition-math/src/training/meta_inject.py` — `meta_mask :54-71` (open/close scan + unclosed-to-end, the reuse target).
- `/tmp/verl_src/verl/trainer/ppo/core_algos.py` — `compute_grpo_outcome_advantage` broadcast `:304,329`, `compute_gdpo_outcome_advantage :362`, per-head read `:423-433`, group-norm clamp `:457`, weighted sum `:461-464`, final whiten `:466`.
- `/home/v-seungplee/sft_e20a_local/added_tokens.json` — authoritative ids 151669/151670.
