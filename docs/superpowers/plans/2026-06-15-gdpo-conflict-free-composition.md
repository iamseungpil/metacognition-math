# Conflict-Free GDPO Reward Composition — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the GDPO region-routed advantage compose multiple reward heads without fighting — anchor-on-R_corr scale normalization, R_emit first-token routing, and meta-length cap — all default-off so existing runs stay byte-identical.

**Architecture:** Extend `compose_dcpo_region_advantage` (pure-torch, unit-testable) with three optional, default-off mechanisms; thread new knobs through `verl_sdc_utils._compute_dcpo_region_advantage` (which owns a module-level EMA state dict across steps); apply the meta-length cap and the open-meta-then-truncation penalty in the `verl_sdc.py` populator. The length-cost knob already exists (raise its value in the stage-C config only).

**Tech Stack:** Python, PyTorch, numpy; pytest (env `/home/v-seungplee/miniconda3/envs/metaprobe/bin/python`); verl/Dr.GRPO; configs under `configs/`.

**Scope note:** This plan covers ONLY the composition (spec 2026-06-15). The redirect Harvest/Prime/Stage-C pipeline (spec 2026-06-14) is a SEPARATE follow-up plan built on top of this once it is green.

**Byte-identical lock:** every new knob defaults to the current behavior. Task 7 proves the full `tests/test_dcpo_*.py` suite still passes unchanged.

---

### Task 1: Anchor-on-R_corr scale normalization in compose

**Files:**
- Modify: `src/training/dcpo_region.py` (`compose_dcpo_region_advantage`, ~1025-1173; add helper near `group_mean_subtract` ~978)
- Test: `tests/test_dcpo_anchor_emit.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dcpo_anchor_emit.py
import torch
from src.training.dcpo_region import compose_dcpo_region_advantage

def _masks(B, T):
    ans = torch.zeros(B, T); ans[:, :2] = 1.0          # answer tokens 0,1
    meta = torch.zeros(B, T); meta[:, 2:4] = 1.0        # meta tokens 2,3
    conf = torch.zeros(B, T)
    rm = torch.ones(B, T)
    return ans, meta, conf, rm

def test_anchor_rescales_aux_to_corr_scale():
    # R_corr scale ~1 (±1), R_meta scale ~0.05 — anchor should lift meta to corr scale.
    B, T = 4, 6
    idx = [0, 0, 1, 1]
    R_corr = [1.0, -1.0, 1.0, -1.0]
    R_meta = [0.05, -0.05, 0.05, -0.05]
    ans, meta, conf, rm = _masks(B, T)
    state = {}
    # warmup=0 -> anchor active immediately; ema=0.0 -> EMA == current batch stats
    A, _ = compose_dcpo_region_advantage(
        response_mask=rm, index=idx, R_corr=R_corr, R_meta=R_meta, R_cal=[0.0]*B,
        answer_mask=ans, meta_content_mask=meta, conf_mask=conf,
        w_corr=1.0, w_meta=1.0, w_cal=0.0,
        anchor_norm=True, anchor_ema_state=state, anchor_ema_decay=0.0,
        anchor_warmup_steps=0,
    )
    # meta-token advantage magnitude should now be ~ corr-token magnitude (within 5%),
    # NOT ~0.05x of it.
    corr_mag = A[:, 0].abs().mean().item()
    meta_mag = A[:, 2].abs().mean().item()
    assert abs(meta_mag - corr_mag) / corr_mag < 0.05, (meta_mag, corr_mag)

def test_anchor_off_is_byte_identical():
    B, T = 4, 6
    idx = [0, 0, 1, 1]
    args = dict(
        response_mask=torch.ones(B, T), index=idx,
        R_corr=[1.0,-1.0,1.0,-1.0], R_meta=[0.05,-0.05,0.05,-0.05], R_cal=[0.0]*B,
        answer_mask=_masks(B,T)[0], meta_content_mask=_masks(B,T)[1],
        conf_mask=_masks(B,T)[2], w_corr=1.0, w_meta=1.0, w_cal=0.0,
    )
    base, _ = compose_dcpo_region_advantage(**args)
    off, _ = compose_dcpo_region_advantage(**args, anchor_norm=False)
    assert torch.allclose(base, off)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/v-seungplee/miniconda3/envs/metaprobe/bin/python -m pytest tests/test_dcpo_anchor_emit.py -v`
Expected: FAIL — `compose_dcpo_region_advantage() got an unexpected keyword argument 'anchor_norm'`.

- [ ] **Step 3: Add the anchor helper + params (minimal implementation)**

In `src/training/dcpo_region.py`, add near `group_mean_subtract`:

```python
def _ema_mean_abs(centered, state, key, decay):
    """EMA of mean(|centered advantage|) for one head. Updates `state` in place,
    returns the running value. `centered` is [B,1] or [B]. decay in [0,1):
    new = decay*old + (1-decay)*cur (decay 0 -> just current batch)."""
    cur = torch.as_tensor(centered, dtype=torch.float32).abs().reshape(-1)
    cur = float(cur[cur > 0].mean()) if (cur > 0).any() else 0.0
    old = state.get(key)
    val = cur if old is None else (decay * old + (1.0 - decay) * cur)
    state[key] = val
    return val
```

Add params to `compose_dcpo_region_advantage` signature (after `w_emit`):
```python
    anchor_norm: bool = False,
    anchor_ema_state: dict | None = None,
    anchor_ema_decay: float = 0.9,
    anchor_warmup_steps: int = 0,
```

After `A_corr`, `A_meta`, `A_cal` are computed (and before they are routed), insert:
```python
    # Anchor-on-R_corr scale normalization (spec 2026-06-15 §3.1): keep R_corr as
    # the Dr.GRPO anchor; rescale auxiliary heads to its mean|advantage| so w_* is
    # "strength relative to correctness" and weak heads (PMI) are not buried.
    # Default-off (anchor_norm False) -> byte-identical.
    if anchor_norm and anchor_ema_state is not None:
        st = anchor_ema_state
        n = st.get("_n", 0) + 1
        st["_n"] = n
        corr_s = _ema_mean_abs(A_corr, st, "corr", anchor_ema_decay)
        meta_s = _ema_mean_abs(A_meta, st, "meta", anchor_ema_decay)
        cal_s = _ema_mean_abs(A_cal, st, "cal", anchor_ema_decay)
        if n > anchor_warmup_steps and corr_s > 0:
            _floor = 1e-6
            A_meta = A_meta * (corr_s / max(meta_s, _floor))
            A_cal = A_cal * (corr_s / max(cal_s, _floor))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/v-seungplee/miniconda3/envs/metaprobe/bin/python -m pytest tests/test_dcpo_anchor_emit.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add src/training/dcpo_region.py tests/test_dcpo_anchor_emit.py
git commit -m "feat(dcpo): anchor-on-R_corr scale normalization (default-off)"
```

---

### Task 2: Anchor the format + emit heads too

**Files:**
- Modify: `src/training/dcpo_region.py` (format/emit blocks ~1136-1158)
- Test: `tests/test_dcpo_anchor_emit.py`

- [ ] **Step 1: Write the failing test**

```python
def test_anchor_rescales_format_and_emit():
    B, T = 4, 6
    idx = [0, 0, 1, 1]
    ans = torch.zeros(B, T); ans[:, :2] = 1.0
    meta = torch.zeros(B, T); meta[:, 2:4] = 1.0
    fv = torch.zeros(B, T); fv[:, 4] = 1.0
    state = {}
    A, _ = compose_dcpo_region_advantage(
        response_mask=torch.ones(B, T), index=idx,
        R_corr=[1.0,-1.0,1.0,-1.0], R_meta=[0.0]*B, R_cal=[0.0]*B,
        answer_mask=ans, meta_content_mask=meta, conf_mask=torch.zeros(B,T),
        R_format=[0.02,-0.02,0.02,-0.02], format_violation_mask=fv, w_format=1.0,
        w_corr=1.0, w_meta=0.0, w_cal=0.0,
        anchor_norm=True, anchor_ema_state=state, anchor_ema_decay=0.0, anchor_warmup_steps=0,
    )
    corr_mag = A[:, 0].abs().mean().item()
    fmt_mag = A[:, 4].abs().mean().item()
    assert abs(fmt_mag - corr_mag) / corr_mag < 0.05, (fmt_mag, corr_mag)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/v-seungplee/miniconda3/envs/metaprobe/bin/python -m pytest tests/test_dcpo_anchor_emit.py::test_anchor_rescales_format_and_emit -v`
Expected: FAIL (format advantage still ~0.02 scale, not lifted).

- [ ] **Step 3: Apply anchor to A_format and A_emit**

In the format block, after `A_format = group_mean_subtract(R_format, index)...`:
```python
        if anchor_norm and anchor_ema_state is not None and \
           anchor_ema_state.get("_n", 0) > anchor_warmup_steps:
            cs = anchor_ema_state.get("corr", 0.0)
            fs = _ema_mean_abs(A_format, anchor_ema_state, "format", anchor_ema_decay)
            if cs > 0:
                A_format = A_format * (cs / max(fs, 1e-6))
```
In the emit block, after `A_emit = group_mean_subtract(R_emit, index)...`:
```python
        if anchor_norm and anchor_ema_state is not None and \
           anchor_ema_state.get("_n", 0) > anchor_warmup_steps:
            cs = anchor_ema_state.get("corr", 0.0)
            es = _ema_mean_abs(A_emit, anchor_ema_state, "emit", anchor_ema_decay)
            if cs > 0:
                A_emit = A_emit * (cs / max(es, 1e-6))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/v-seungplee/miniconda3/envs/metaprobe/bin/python -m pytest tests/test_dcpo_anchor_emit.py -v`
Expected: PASS (all three).

- [ ] **Step 5: Commit**

```bash
git add src/training/dcpo_region.py tests/test_dcpo_anchor_emit.py
git commit -m "feat(dcpo): anchor format+emit heads to R_corr scale"
```

---

### Task 3: R_emit first-token routing

**Files:**
- Modify: `src/training/dcpo_region.py` (emit block ~1156-1158)
- Test: `tests/test_dcpo_anchor_emit.py`

- [ ] **Step 1: Write the failing test**

```python
def test_emit_first_token_routing_clean():
    # Silent row (R_emit 0) must get NEGATIVE emit advantage on token 0 only;
    # answer/meta tokens must be unchanged vs no-emit baseline.
    B, T = 2, 6
    idx = [0, 0]
    ans = torch.zeros(B, T); ans[:, 1:3] = 1.0   # answer tokens 1,2 (NOT token 0)
    meta = torch.zeros(B, T); meta[:, 3:5] = 1.0
    common = dict(
        response_mask=torch.ones(B, T), index=idx,
        R_corr=[1.0, -1.0], R_meta=[0.0]*B, R_cal=[0.0]*B,
        answer_mask=ans, meta_content_mask=meta, conf_mask=torch.zeros(B,T),
        w_corr=1.0, w_meta=0.0, w_cal=0.0,
    )
    base, _ = compose_dcpo_region_advantage(**common)
    routed, _ = compose_dcpo_region_advantage(
        **common, R_emit=[1.0, 0.0], w_emit=0.5,
        emit_route="first_token", emit_first_n=1,
    )
    # token 0 differs (emit landed there); answer tokens 1,2 identical to baseline.
    assert not torch.allclose(routed[:, 0], base[:, 0])
    assert torch.allclose(routed[:, 1:3], base[:, 1:3])
    # silent row (row 1) gets negative emit on token 0
    assert routed[1, 0] < 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/v-seungplee/miniconda3/envs/metaprobe/bin/python -m pytest tests/test_dcpo_anchor_emit.py::test_emit_first_token_routing_clean -v`
Expected: FAIL — unexpected kwarg `emit_route`.

- [ ] **Step 3: Add emit routing params + first-token mask**

Add params after `w_emit`:
```python
    emit_route: str = "global",
    emit_first_n: int = 1,
```
Replace the emit broadcast (`advantages = advantages + float(w_emit) * A_emit * rm.to(device)`) with:
```python
        if emit_route == "first_token":
            # Route emit onto the first emit_first_n REAL tokens (the emit/abstain
            # decision point) — disjoint from ANSWER/META, so it no longer mixes
            # into answer learning while still penalizing silent rows. (spec §3.2)
            first_mask = torch.zeros_like(rm)
            rmb = rm > 0.5
            for b in range(rm.shape[0]):
                pos = torch.nonzero(rmb[b], as_tuple=False).flatten()
                if pos.numel():
                    first_mask[b, pos[:emit_first_n]] = 1.0
            advantages = advantages + float(w_emit) * A_emit * first_mask.to(device)
        else:
            advantages = advantages + float(w_emit) * A_emit * rm.to(device)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/v-seungplee/miniconda3/envs/metaprobe/bin/python -m pytest tests/test_dcpo_anchor_emit.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/training/dcpo_region.py tests/test_dcpo_anchor_emit.py
git commit -m "feat(dcpo): R_emit first-token routing (default global)"
```

---

### Task 4: Meta-length cap on the floor

**Files:**
- Modify: `src/training/dcpo_region.py` (floor block ~1167-1171)
- Test: `tests/test_dcpo_anchor_emit.py`

- [ ] **Step 1: Write the failing test**

```python
def test_meta_len_cap_limits_floor():
    # A row with many meta tokens: with a cap, only the first `cap` meta tokens
    # share the +floor (row total <= floor*cap/row_n stays bounded); without cap
    # the whole meta span shares it. Assert capped floor total < uncapped.
    B, T = 1, 10
    meta = torch.zeros(B, T); meta[:, 2:9] = 1.0   # 7 meta tokens
    args = dict(
        response_mask=torch.ones(B, T), index=[0],
        R_corr=[1.0], R_meta=[0.0], R_cal=[0.0],
        answer_mask=torch.zeros(B,T), meta_content_mask=meta, conf_mask=torch.zeros(B,T),
        w_corr=0.0, w_meta=0.0, w_cal=0.0,
        meta_floor=0.1, floor_mask=[1.0],
    )
    uncapped, _ = compose_dcpo_region_advantage(**args)
    capped, _ = compose_dcpo_region_advantage(**args, meta_len_cap=3)
    assert capped[:, 2:9].sum().item() < uncapped[:, 2:9].sum().item()
    # capped applies floor to only the first 3 meta tokens
    assert capped[0, 5:9].abs().sum().item() == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/v-seungplee/miniconda3/envs/metaprobe/bin/python -m pytest tests/test_dcpo_anchor_emit.py::test_meta_len_cap_limits_floor -v`
Expected: FAIL — unexpected kwarg `meta_len_cap`.

- [ ] **Step 3: Add meta_len_cap param + cap the floor mask**

Add param after `floor_mask`:
```python
    meta_len_cap: int = 0,
```
In the floor block, before computing `row_n`, cap `meta_in_resp` to the first `meta_len_cap` meta tokens per row:
```python
    if meta_floor and floor_mask is not None:
        fl = torch.as_tensor(floor_mask, dtype=torch.float32).to(device).view(-1, 1)
        meta_in_resp = meta_c * rm
        if meta_len_cap and meta_len_cap > 0:
            # keep only the first meta_len_cap meta tokens per row (spec §3.3:
            # floor must not pay for sheer meta length).
            capped = torch.zeros_like(meta_in_resp)
            mb = meta_in_resp > 0.5
            for b in range(meta_in_resp.shape[0]):
                pos = torch.nonzero(mb[b], as_tuple=False).flatten()
                if pos.numel():
                    capped[b, pos[:meta_len_cap]] = 1.0
            meta_in_resp = capped
        row_n = meta_in_resp.sum(dim=1, keepdim=True).clamp(min=1.0)
        advantages = advantages + float(meta_floor) * fl * (meta_in_resp / row_n)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/v-seungplee/miniconda3/envs/metaprobe/bin/python -m pytest tests/test_dcpo_anchor_emit.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/training/dcpo_region.py tests/test_dcpo_anchor_emit.py
git commit -m "feat(dcpo): meta-length cap on the anti-collapse floor"
```

---

### Task 5: Open-meta-then-truncation penalty (reward layer)

**Files:**
- Modify: `src/training/dcpo_region.py` (`dcpo_region_rewards`, format_penalty list ~949-957)
- Test: `tests/test_dcpo_anchor_emit.py`

- [ ] **Step 1: Write the failing test**

```python
from src.training.dcpo_region import dcpo_region_rewards

def test_trunc_open_penalty_only_for_opened_then_truncated():
    # Row A: opened a meta then truncated (fmt_class 'truncation', has_meta True).
    # Row B: no meta, long answer truncated (fmt_class 'truncation', no <|meta|>).
    # Penalty must hit A, not B.
    comps = [
        {"content": "reason <|meta|> confidence: 0.5 ... (cut)"},   # opened+cut
        {"content": "just a long answer with no meta block ... (cut)"},
    ]
    out = dcpo_region_rewards(
        comps, ground_truth=["1", "1"], group_index=[0, 0],
        fmt_class=["truncation", "truncation"], trunc_open_penalty=0.3,
    )
    fp = out["format_penalty"]
    assert fp[0] == -0.3      # opened-then-truncated penalized
    assert fp[1] == 0.0       # meta-less truncation untouched
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/v-seungplee/miniconda3/envs/metaprobe/bin/python -m pytest tests/test_dcpo_anchor_emit.py::test_trunc_open_penalty_only_for_opened_then_truncated -v`
Expected: FAIL — unexpected kwarg `trunc_open_penalty` (or wrong values).

- [ ] **Step 3: Add trunc_open_penalty param + apply in format_penalty**

Add param to `dcpo_region_rewards` signature (near `format_neg`):
```python
    trunc_open_penalty: float = 0.0,
```
In the `format_penalty` construction (fmt_class branch), add a truncation case that
checks `has_meta[i]`:
```python
        "format_penalty": (
            [
                1.0 if c == "wellformed"
                else (-float(format_neg) if c in ("drift", "discard")
                      else (-float(trunc_open_penalty)
                            if (c == "truncation" and trunc_open_penalty and has_meta[i])
                            else 0.0))
                for i, c in enumerate(fmt_class)
            ]
            if fmt_class is not None
            else [-float(format_neg) if meta_drift[i] else 0.0 for i in range(B)]
        ),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/v-seungplee/miniconda3/envs/metaprobe/bin/python -m pytest tests/test_dcpo_anchor_emit.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/training/dcpo_region.py tests/test_dcpo_anchor_emit.py
git commit -m "feat(dcpo): medium open-meta-then-truncation penalty (default 0)"
```

---

### Task 6: Wire the knobs through verl_sdc_utils + verl_sdc

**Files:**
- Modify: `src/training/verl_sdc_utils.py` (`_compute_dcpo_region_advantage` ~290-395; add module-level EMA dict)
- Modify: `src/training/verl_sdc.py` (populator: pass `trunc_open_penalty` into `dcpo_region_rewards`; read `dcpo_meta_len_cap`)
- Test: `tests/test_dcpo_v4_integration.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
# in tests/test_dcpo_v4_integration.py — a smoke that the wiring forwards knobs.
def test_anchor_knobs_forwarded(monkeypatch):
    import src.training.verl_sdc_utils as U
    captured = {}
    import src.training.dcpo_region as R
    real = R.compose_dcpo_region_advantage
    def spy(**kw):
        captured.update(kw); return real(**kw)
    monkeypatch.setattr(R, "compose_dcpo_region_advantage", spy)
    # minimal batch/config with anchor on (reuse the integration fixture builder
    # already in this file; set config dcpo_anchor_norm=True, dcpo_emit_route='first_token').
    _run_min_compute(config_overrides={"dcpo_anchor_norm": True,
                                       "dcpo_emit_route": "first_token",
                                       "dcpo_meta_len_cap": 3})
    assert captured.get("anchor_norm") is True
    assert captured.get("emit_route") == "first_token"
    assert captured.get("meta_len_cap") == 3
    assert captured.get("anchor_ema_state") is not None   # module-level dict passed
```
(If `_run_min_compute` does not exist, add a tiny helper in the test mirroring the
existing integration fixture; reuse the batch/non_tensor_batch builder already used by
the other tests in this file.)

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/v-seungplee/miniconda3/envs/metaprobe/bin/python -m pytest tests/test_dcpo_v4_integration.py::test_anchor_knobs_forwarded -v`
Expected: FAIL — knobs not forwarded.

- [ ] **Step 3: Forward knobs + add module-level EMA state**

At module top of `verl_sdc_utils.py`:
```python
# Anchor EMA state persists ACROSS steps (spec 2026-06-15 §3.5). Module-level so a
# single training process keeps one running estimate; re-warms on resume (acceptable).
_ANCHOR_EMA_STATE: dict = {}
```
In `_compute_dcpo_region_advantage`, before the `return compose_dcpo_region_advantage(`,
build the new kwargs (all default-off):
```python
    _anchor_kwargs = {}
    if bool(config.get("dcpo_anchor_norm", False)):
        _anchor_kwargs = dict(
            anchor_norm=True,
            anchor_ema_state=_ANCHOR_EMA_STATE,
            anchor_ema_decay=float(config.get("dcpo_anchor_ema", 0.9)),
            anchor_warmup_steps=int(config.get("dcpo_anchor_warmup_steps", 0)),
        )
    _emit_route_kwargs = {}
    if str(config.get("dcpo_emit_route", "global")) == "first_token":
        _emit_route_kwargs = dict(
            emit_route="first_token",
            emit_first_n=int(config.get("dcpo_emit_first_n", 1)),
        )
    _len_cap = int(config.get("dcpo_meta_len_cap", 0))
    if _len_cap:
        _emit_route_kwargs["meta_len_cap"] = _len_cap
```
and add `**_anchor_kwargs, **_emit_route_kwargs,` to the compose call argument list
(alongside `**_fmt_kwargs, **_emit_kwargs,`).

In `verl_sdc.py`, where `dcpo_region_rewards(...)` is called in the populator, pass:
```python
            trunc_open_penalty=float(_v4_read("dcpo_trunc_open_penalty", 0.0) or 0.0),
```
(`_v4_read` is the existing v4 config reader used for `dcpo_len_cost` at line 433.)

- [ ] **Step 4: Run test to verify it passes + the v4 suites**

Run: `/home/v-seungplee/miniconda3/envs/metaprobe/bin/python -m pytest tests/test_dcpo_v4_integration.py tests/test_dcpo_anchor_emit.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/training/verl_sdc_utils.py src/training/verl_sdc.py tests/test_dcpo_v4_integration.py
git commit -m "feat(dcpo): wire anchor/emit-route/len-cap/trunc-penalty knobs (default-off)"
```

---

### Task 7: Regression lock + stage-C config

**Files:**
- Create: `configs/triobj_dcpo_v4_stage2c_h100_4x4k.yaml`
- Test: full `tests/test_dcpo_*.py` suite

- [ ] **Step 1: Run the full dcpo suite to prove byte-identical**

Run: `/home/v-seungplee/miniconda3/envs/metaprobe/bin/python -m pytest tests/test_dcpo_region.py tests/test_dcpo_advantage.py tests/test_dcpo_rewards.py tests/test_dcpo_v3.py tests/test_dcpo_v3_cf.py tests/test_dcpo_v3m.py tests/test_dcpo_v4_integration.py tests/test_dcpo_v4_pmi.py -v`
Expected: ALL PASS (new knobs default-off ⇒ no existing test changes).

- [ ] **Step 2: Create the stage-C config (copy s2b, flip the new knobs on)**

Copy `configs/triobj_dcpo_v4_stage2b_h100_4x4k.yaml` → `configs/triobj_dcpo_v4_stage2c_h100_4x4k.yaml` and set, under `algorithm:`:
```yaml
  dcpo_anchor_norm: true
  dcpo_anchor_ema: 0.9
  dcpo_anchor_warmup_steps: 20
  dcpo_emit_route: first_token
  dcpo_emit_first_n: 1
  dcpo_meta_len_cap: 96
  dcpo_trunc_open_penalty: 0.3
  dcpo_len_cost: 0.08          # s2b 0.03 -> 0.08 (length containment)
```
and under `trainer:` set `experiment_name`/`default_local_dir` to `triobj_dcpo_v4_stage2c_h100_4x4k`. Keep `max_response_length: 4096`.

- [ ] **Step 3: Validate the config loads**

Run: `/home/v-seungplee/miniconda3/envs/metaprobe/bin/python -c "import yaml; yaml.safe_load(open('configs/triobj_dcpo_v4_stage2c_h100_4x4k.yaml'))" && echo OK`
Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add configs/triobj_dcpo_v4_stage2c_h100_4x4k.yaml
git commit -m "feat(dcpo): stage-C config with conflict-free composition knobs on"
```

---

## Self-Review notes

- **Spec coverage:** §3.1 anchor → Tasks 1-2; §3.2 emit routing → Task 3; §3.3 meta cap + trunc penalty + len_cost → Tasks 4,5,7; §3.5 EMA state → Task 6; §4 default-off knobs → every task's "off" test + Task 7 regression; §7 tests → Tasks 1-6 unit tests + Task 7 suite.
- **Intent link:** stronger confidence-introspection/metacognition without the difficulty/step collapse — Tasks 3-5+7 are exactly the length/structure containment that stops the s2b ratchet; Tasks 1-2 keep the PMI quality signal (the "useful metacognition" grader) from being buried.
- **Deferred to follow-up plan:** redirect Harvest/Prime/Stage-C (spec 2026-06-14) — built on this composition once green.
