# RL Data Regime Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a signal-alive RL dataset (multi-scenario, multi-source, easy restored, prose-gold filtered) and an s3 training config (n=8, clip-higher, composition knobs) so the meta channel stops collapsing.

**Architecture:** Add a generalized `build_v8_meta_subset` next to the existing `build_v8_redirect_subset` (don't break the old one), driven from the already-mixed corpus `data/v8_meta_inside_think.parquet`; build a new `verl_train_meta_mix` parquet + HF upload; create an s3 config inheriting the s2c composition knobs with rollout n=8 and clip-higher; check verl for built-in dynamic sampling and enable it if present (else defer to a follow-up — do NOT block s3 on a large verl change).

**Tech Stack:** Python, pandas, pyarrow; pytest (`/home/v-seungplee/miniconda3/envs/metaprobe/bin/python`); HF datasets (`iamseungpil/metacot-sdc-data`); verl/Dr.GRPO; amlt H100.

**Byte-identical lock:** the existing `build_v8_redirect_subset` and `verl_train_redirect.parquet` path stay untouched; we ADD a parallel builder.

---

### Task 1: Prose-gold extractability helper

**Files:**
- Modify: `src/training/verl_gdpo_data.py` (add helper near `_extract_math_answer`)
- Test: `tests/test_verl_meta_subset.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_verl_meta_subset.py
from src.training.verl_gdpo_data import _gold_is_rule_gradable

def test_numeric_and_boxed_gold_pass():
    assert _gold_is_rule_gradable("42")
    assert _gold_is_rule_gradable("\\frac{3}{4}")
    assert _gold_is_rule_gradable("7\\sqrt{5}")

def test_prose_gold_rejected():
    assert not _gold_is_rule_gradable("\\text{Yes, it must be a cube.}")
    assert not _gold_is_rule_gradable("Player 0 wins")
    assert not _gold_is_rule_gradable("")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/v-seungplee/miniconda3/envs/metaprobe/bin/python -m pytest tests/test_verl_meta_subset.py -v`
Expected: FAIL — `cannot import name '_gold_is_rule_gradable'`.

- [ ] **Step 3: Implement the helper**

```python
def _gold_is_rule_gradable(gt: str) -> bool:
    """True iff gold answer is rule-gradable (numeric / boxed-able), NOT prose.
    Drops omni-math prose golds (~26%) that rule-based grading scores 0 even when
    the model is right (spec 2026-06-15 §3.6)."""
    if gt is None:
        return False
    s = str(gt).strip()
    if not s:
        return False
    # Reject obvious prose: \text{...} wrappers or >2 alphabetic words.
    if "\\text{" in s:
        return False
    import re as _re
    words = _re.findall(r"[A-Za-z]{2,}", s)
    if len(words) > 2:  # e.g. "Player 0 wins ..." — prose
        return False
    # Accept if it contains a digit, a fraction/sqrt/expression token, or is short symbolic.
    if _re.search(r"[0-9]", s) or any(t in s for t in ("\\frac", "\\sqrt", "\\pi", "(", "=")):
        return True
    return len(s) <= 8  # short symbolic like 'x', 'a+b'
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/v-seungplee/miniconda3/envs/metaprobe/bin/python -m pytest tests/test_verl_meta_subset.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/training/verl_gdpo_data.py tests/test_verl_meta_subset.py
git commit -m "feat(data): prose-gold rule-gradability filter helper

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Generalized `build_v8_meta_subset`

**Files:**
- Modify: `src/training/verl_gdpo_data.py` (add new function after `build_v8_redirect_subset`)
- Test: `tests/test_verl_meta_subset.py`

- [ ] **Step 1: Write the failing test**

```python
import pandas as pd
from src.training.verl_gdpo_data import build_v8_meta_subset

def _toy_corpus(tmp_path):
    rows = []
    msgs = [{"role":"user","content":"Q?"},{"role":"assistant","content":"... \\boxed{5}"}]
    specs = [("redirect","easy","gsm8k","5"), ("verify","medium","hendrycks_math/algebra","7"),
             ("redirect","hard","omni-math","\\text{Yes}"),  # prose -> dropped
             ("verify","easy","gsm8k","3")]
    for sc,df_,src,gt in specs:
        rows.append({"scenario":sc,"difficulty":df_,"source":src,"trigger":"anomaly",
                     "messages":[{"role":"user","content":"Q?"},
                                 {"role":"assistant","content":f"x \\boxed{{{gt}}}"}]})
    meta=pd.DataFrame(rows); base=meta.copy()
    mp=tmp_path/"meta.parquet"; bp=tmp_path/"base.parquet"
    meta.to_parquet(mp); base.to_parquet(bp)
    return str(mp), str(bp)

def test_meta_subset_widens_scenarios_difficulties_and_drops_prose(tmp_path):
    mp, bp = _toy_corpus(tmp_path)
    out = build_v8_meta_subset(mp, bp, scenarios=("redirect","verify"),
                               allowed_difficulties=("easy","medium","hard"),
                               require_gradable_gold=True, val_ratio=0.25, seed=0)
    rows = out["meta_train"] + out["meta_val"]
    scns = {r["split_tags"]["scenario"] for r in rows}
    diffs = {r["split_tags"]["difficulty"] for r in rows}
    assert scns == {"redirect","verify"}        # both scenarios kept
    assert "easy" in diffs                        # easy restored
    assert len(rows) == 3                          # prose-gold row dropped (4 -> 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/v-seungplee/miniconda3/envs/metaprobe/bin/python -m pytest tests/test_verl_meta_subset.py::test_meta_subset_widens_scenarios_difficulties_and_drops_prose -v`
Expected: FAIL — `cannot import name 'build_v8_meta_subset'`.

- [ ] **Step 3: Implement (copy build_v8_redirect_subset, widen selector + gold filter)**

Add a new function (keep the old one intact). Reuse `_extract_prompt_and_gt_from_messages`,
`records_to_parquet`, `_gold_is_rule_gradable`:

```python
def build_v8_meta_subset(
    meta_path: str,
    base_path: str,
    *,
    scenarios: tuple[str, ...] = ("redirect", "verify"),
    allowed_difficulties: tuple[str, ...] = ("easy", "medium", "hard"),
    require_gradable_gold: bool = True,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Generalized v8 subset: multi-scenario + multi-difficulty (easy restored) +
    prose-gold filter. Source mix is whatever the corpus carries (gsm8k/MATH/omni).
    Parallel to build_v8_redirect_subset (which stays as-is). Spec 2026-06-15."""
    import random
    meta_df = pd.read_parquet(meta_path)
    base_df = pd.read_parquet(base_path)
    if len(meta_df) != len(base_df):
        raise ValueError(f"Meta/base length mismatch: {len(meta_df)} vs {len(base_df)}")
    selector = (
        meta_df["scenario"].isin(list(scenarios))
        & meta_df["difficulty"].isin(list(allowed_difficulties))
    )
    selected_idx = meta_df.index[selector].tolist()
    if require_gradable_gold:
        keep = []
        for idx in selected_idx:
            _, gt = _extract_prompt_and_gt_from_messages(meta_df.loc[idx]["messages"])
            if _gold_is_rule_gradable(gt):
                keep.append(idx)
        selected_idx = keep
    if not selected_idx:
        raise ValueError("Meta subset selection produced zero rows")
    rng = random.Random(seed)
    rng.shuffle(selected_idx)
    n_val = max(1, int(round(len(selected_idx) * val_ratio)))
    val_idx = set(selected_idx[:n_val])
    outputs = {"meta_train": [], "meta_val": [], "base_train": [], "base_val": []}
    for idx in selected_idx:
        meta_row = meta_df.loc[idx]; base_row = base_df.loc[idx]
        prompt_text, gt = _extract_prompt_and_gt_from_messages(meta_row["messages"])
        data_source = str(meta_row.get("source", "v8_meta"))
        record = {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": prompt_text}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": gt},
            "split_tags": {
                "scenario": str(meta_row.get("scenario", "")),
                "difficulty": str(meta_row.get("difficulty", "")),
                "trigger": str(meta_row.get("trigger", "")),
                "row_index": int(idx),
            },
        }
        outputs["meta_val" if idx in val_idx else "meta_train"].append(record)
        base_prompt, base_gt = _extract_prompt_and_gt_from_messages(base_row["messages"])
        if base_prompt != prompt_text or base_gt != gt:
            raise ValueError(f"Base-matched row mismatch at index {idx}")
        outputs["base_val" if idx in val_idx else "base_train"].append(
            {**record, "data_source": f"{data_source}::base_matched"})
    return outputs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/v-seungplee/miniconda3/envs/metaprobe/bin/python -m pytest tests/test_verl_meta_subset.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/training/verl_gdpo_data.py tests/test_verl_meta_subset.py
git commit -m "feat(data): build_v8_meta_subset (multi-scenario/difficulty + prose-gold filter)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: CLI entrypoint to emit the mix parquets

**Files:**
- Modify: `src/training/verl_gdpo_data.py` (`__main__`, ~line 280-321)
- Test: manual run (Task 4)

- [ ] **Step 1: Add a `meta_mix` subcommand to `__main__`**

In the argparse block add a mode that calls `build_v8_meta_subset` and writes four parquets via
`records_to_parquet`:

```python
    parser.add_argument("--mode", default="redirect", choices=["redirect", "meta_mix"])
    parser.add_argument("--out_train_meta_mix", default="data/verl_train_meta_mix.parquet")
    parser.add_argument("--out_val_meta_mix", default="data/verl_val_meta_mix.parquet")
    parser.add_argument("--out_train_meta_mix_base", default="data/verl_train_meta_mix_base.parquet")
    parser.add_argument("--out_val_meta_mix_base", default="data/verl_val_meta_mix_base.parquet")
```
and after parsing:
```python
    if args.mode == "meta_mix":
        outs = build_v8_meta_subset(args.meta_path, args.base_path)
        records_to_parquet(outs["meta_train"], args.out_train_meta_mix)
        records_to_parquet(outs["meta_val"], args.out_val_meta_mix)
        records_to_parquet(outs["base_train"], args.out_train_meta_mix_base)
        records_to_parquet(outs["base_val"], args.out_val_meta_mix_base)
        raise SystemExit(0)
```
(Use the existing `--meta_path` / `--base_path` args; if absent, add them defaulting to
`data/v8_meta_inside_think.parquet` and `data/v8_base_matched.parquet`.)

- [ ] **Step 2: Commit**

```bash
git add src/training/verl_gdpo_data.py
git commit -m "feat(data): meta_mix CLI mode emits verl_train/val_meta_mix parquets

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Build the parquet + verify distribution

**Files:**
- Run only (produces `data/verl_train_meta_mix.parquet` etc.)

- [ ] **Step 1: Build**

Run: `/home/v-seungplee/miniconda3/envs/metaprobe/bin/python -m src.training.verl_gdpo_data --mode meta_mix`
Expected: "Saved N rows to data/verl_train_meta_mix.parquet" (+ val/base).

- [ ] **Step 2: Verify distribution (assert signal-alive composition)**

```bash
/home/v-seungplee/miniconda3/envs/metaprobe/bin/python - <<'PY'
import pandas as pd
d=pd.read_parquet("data/verl_train_meta_mix.parquet")
st=pd.json_normalize(d["split_tags"])
print("rows", len(d))
print("scenario", st["scenario"].value_counts().to_dict())
print("difficulty", st["difficulty"].value_counts().to_dict())
print("source", d["data_source"].apply(lambda s: s.split('/')[0]).value_counts().to_dict())
assert {"redirect","verify"} <= set(st["scenario"].unique())
assert "easy" in set(st["difficulty"].unique())
assert len(d) > 3000   # widened from 2935 redirect-only
PY
```
Expected: easy present, both scenarios present, >3000 rows.

- [ ] **Step 3: Commit the built parquet pointer (data lives on HF, not git)**

No git commit of the parquet (large/binary). Proceed to upload (Task 5).

---

### Task 5: Upload the mix parquets to HF

**Files:**
- Run: `scripts/upload_dataset_artifacts.py`

- [ ] **Step 1: Upload**

```bash
cd /home/v-seungplee/metacognition-math && set -a; source .env; set +a
/home/v-seungplee/miniconda3/envs/metaprobe/bin/python scripts/upload_dataset_artifacts.py \
  --repo_id iamseungpil/metacot-sdc-data \
  --files data/verl_train_meta_mix.parquet data/verl_val_meta_mix.parquet
```
(Check upload_dataset_artifacts.py's exact arg names first; adapt the flags to match. If it takes
positional paths, pass them positionally.)
Expected: "uploaded data/verl_train_meta_mix.parquet -> iamseungpil/metacot-sdc-data:...".

- [ ] **Step 2: Add the new files to `scripts/pull_parquets.py` FILES list**

Append `"verl_train_meta_mix.parquet"`, `"verl_val_meta_mix.parquet"` so the node pulls them.

- [ ] **Step 3: Commit the pull_parquets change**

```bash
git add scripts/pull_parquets.py
git commit -m "feat(data): pull verl_*_meta_mix parquets on node

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Check verl for built-in dynamic sampling

**Files:**
- Investigate only; decision recorded here.

- [ ] **Step 1: Grep the node's verl for dynamic-sampling / filter-groups support**

On a node (or the simplerl env) run:
```bash
python -c "import verl, os; print(verl.__file__)"
grep -rn "filter_groups\|dynamic.*sampl\|gen_batch_size\|over_sampl" $(python -c "import verl,os;print(os.path.dirname(verl.__file__))")/trainer 2>/dev/null | head
```
- [ ] **Step 2: Decide (record in the s3 config comments)**
  - If verl exposes it (e.g. `algorithm.filter_groups.enable`): enable in the s3 config (Task 7).
  - If NOT: s3 ships WITHOUT dynamic sampling (data mix + n=8 + clip-higher alone already restore most
    signal); dynamic sampling becomes a separate follow-up plan. **Do not block s3 on a large verl
    change** (spec §3.4 / §8 — data-change-first).

---

### Task 7: s3 training config

**Files:**
- Create: `configs/triobj_dcpo_v4_stage3_h100_4x4k.yaml`
- Test: yaml load

- [ ] **Step 1: Copy s2c config and change data + sampling knobs**

Copy `configs/triobj_dcpo_v4_stage2c_h100_4x4k.yaml` → stage3. Change:
```yaml
data:
  train_files: /scratch/metacognition/data/verl_train_meta_mix.parquet
  val_files: /scratch/metacognition/data/verl_val_meta_mix.parquet
  max_response_length: 4096
actor_rollout_ref:
  rollout:
    n: 8                       # was 4 — larger group => mixed groups frequent
algorithm:
  # clip-higher (DAPO): preserve entropy / meta-emission diversity
  clip_ratio_low: 0.2
  clip_ratio_high: 0.28
  # dynamic sampling: ENABLE here ONLY if Task 6 found verl support, e.g.
  # filter_groups: {enable: true, metric: acc, max_num_gen_batches: 4}
trainer:
  project_name: metacot-dcpo-v4
  experiment_name: triobj_dcpo_v4_stage3_h100_4x4k
  default_local_dir: /scratch/checkpoints/triobj_dcpo_v4_stage3_h100_4x4k
```
Keep all s2c composition knobs (`dcpo_anchor_norm`, `dcpo_emit_route`, `dcpo_meta_len_cap`,
`dcpo_trunc_open_penalty`, `dcpo_len_cost: 0.08`). Verify `clip_ratio_low/high` are the correct verl
keys (else use `actor_rollout_ref.actor.clip_ratio_low/high`) when checking Task 6's verl.

- [ ] **Step 2: Validate load**

Run: `/home/v-seungplee/miniconda3/envs/metaprobe/bin/python -c "import yaml; c=yaml.safe_load(open('configs/triobj_dcpo_v4_stage3_h100_4x4k.yaml')); print(c['actor_rollout_ref']['rollout']['n'], c['data']['train_files'], c['trainer']['project_name'])"`
Expected: `8 /scratch/.../verl_train_meta_mix.parquet metacot-dcpo-v4`.

- [ ] **Step 3: Commit**

```bash
git add configs/triobj_dcpo_v4_stage3_h100_4x4k.yaml
git commit -m "feat(dcpo): stage-3 config (meta-mix data, n=8, clip-higher, composition on)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Node yaml + release + smoke-validate

**Files:**
- Create: `h100std_triobj_dcpo_v4_s3.yaml`

- [ ] **Step 1: Clone s2c node yaml → s3**

Copy `h100std_triobj_dcpo_v4_s2c.yaml` → `h100std_triobj_dcpo_v4_s3.yaml`. Change everywhere:
`stage2c`→`stage3`, `s2c`→`s3`, `WANDB_NAME: dcpo_v4_s3`, job `name: triobj_dcpo_v4_s3`,
`--config-name=triobj_dcpo_v4_stage3_h100_4x4k`, push/pull `--config_name triobj_dcpo_v4_stage3_h100_4x4k`,
checkpoint dirs `triobj_dcpo_v4_stage3_h100_4x4k`. Update `description`. `CODE_TAR_REVISION` will be set
after the release (Step 3).

- [ ] **Step 2: yaml.safe_load validate**

Run: `/home/v-seungplee/miniconda3/envs/metaprobe/bin/python -c "import yaml; d=yaml.safe_load(open('h100std_triobj_dcpo_v4_s3.yaml')); print(d['jobs'][0]['name'], d['jobs'][0]['submit_args']['env']['WANDB_NAME'])"`
Expected: `triobj_dcpo_v4_s3 dcpo_v4_s3`.

- [ ] **Step 3: Build release asset + set CODE_TAR_REVISION**

```bash
cd /home/v-seungplee/metacognition-math && set -a; source .env; set +a
# build tarball (gh CLI absent -> REST API, as done for s2c asset 447900110):
git archive --worktree-attributes --format=tar.gz --prefix=metacognition/ -o /tmp/metacognition.tar.gz HEAD
# token-leak scan (reuse package_e4_release.sh logic) then create release + upload via curl REST,
# capture the numeric asset id, and sed it into h100std_triobj_dcpo_v4_s3.yaml CODE_TAR_REVISION.
```
(Mirror the exact REST sequence already used for s2c: POST /releases, POST uploads.github.com asset,
parse `.id`.)

- [ ] **Step 4: Commit the node yaml**

```bash
git add h100std_triobj_dcpo_v4_s3.yaml
git commit -m "feat(dcpo): stage-3 node yaml (meta-mix + composition)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 5: Full dcpo regression still green (no reward-path change, but confirm)**

Run: `/home/v-seungplee/miniconda3/envs/metaprobe/bin/python -m pytest tests/test_dcpo_region.py tests/test_dcpo_anchor_emit.py tests/test_verl_meta_subset.py -q`
Expected: all pass.

---

## Self-Review notes

- **Spec coverage:** §3.1 source mix → Tasks 2,4 (corpus already mixed, filter widened); §3.2 scenario mix → Task 2 (redirect+verify); §3.3 difficulty coarse+dynamic → Task 2 (easy restored) + Task 6 (dynamic); §3.4 dynamic sampling → Task 6 (check/enable, else follow-up); §3.5 n 4→8 → Task 7; §3.6 grading robustness → Tasks 1,2; §3.7 clip-higher/len_cost/composition → Task 7.
- **Launch + autoresearch:** after Task 8, submit s3 (`amlt run h100std_triobj_dcpo_v4_s3.yaml triobj-dcpo-v4-s3 -y`) and start autoresearch monitoring (HD1-HD4 + composition dashboard).
- **Deferred:** redirect Harvest/Prime (2026-06-14 spec); dynamic sampling implementation if verl lacks it.
