"""Integration tests for TRIOBJ_DCPO_V4 — dense likelihood (PMI) R_meta wiring.

PURE PYTHON (runs under /home/v-seungplee/miniconda3/envs/metaprobe/bin/python).
verl / ray / omegaconf / tensordict are NOT installed in this env; importing
tests.test_dcpo_v3_cf installs the auto-stub finder so src.training.verl_sdc /
verl_sdc_utils import cleanly (same pattern as the v3 test files).

Covers (task spec, all review-traced):
  - FIVE-WAY SYNC: REWARD_CONFIGS['TRIOBJ_DCPO_V4'] <-> BOTH stage yamls
    (keys/weights/knob mirrors), source-level populator-writes-every-key.
  - verl-STANDARD batch layout of _build_pmi_score_batches (the API-scout
    no_padding_2_padding misalignment fix) + T=1.0 hardcode (M1, source-level).
  - conf carve-out routing through compose via the call site (I4).
  - dcpo_rmeta_member centering (I2) at the compose level.
  - w_meta warmup schedule (M4) + its non_tensor threading.
  - _compute_dcpo_v4_pmi_rmeta end-to-end with a fake tokenizer + monkeypatched
    ref scorer (selection / alignment-failure / answer-leak guard / membership).
"""
import os
import types

import numpy as np
import pytest
import torch

import tests.test_dcpo_v3_cf  # noqa: F401  (installs the verl/omegaconf auto-stub)
import src.training.verl_sdc as V
from src.training.verl_sdc import (
    REWARD_CONFIGS,
    _REGION_ROUTED_MODES,
    _DCPO_V3_FMT_MODES,
    _build_pmi_score_batches,
    _compute_dcpo_v4_pmi_rmeta,
    _compute_dcpo_v4_pmi_shift_rmeta,
    _dcpo_v4_ref_logprobs,
    _populate_dcpo_region_keys,
    _v4_rmeta_source_strict,
)
from src.training.verl_sdc_utils import (
    _compute_dcpo_region_advantage,
    compute_sdc_gdpo_advantage,
    dcpo_length_cost,
    dcpo_w_meta_warmup_scale,
)
from src.training.dcpo_region import compose_dcpo_region_advantage
from tests.test_dcpo_v4_pmi import FakeMergeTokenizer

_CFG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")


def _load_yaml(name):
    import yaml as _yaml
    with open(os.path.join(_CFG_DIR, name)) as f:
        return _yaml.safe_load(f)


# ═══════════════════════════════════════════════════════════════════════════
# mode registration + five-way sync (REWARD_CONFIGS <-> both yamls)
# ═══════════════════════════════════════════════════════════════════════════
def test_v4_mode_registered_in_routing_sets():
    assert "TRIOBJ_DCPO_V4" in REWARD_CONFIGS
    assert "TRIOBJ_DCPO_V4" in _REGION_ROUTED_MODES
    assert "TRIOBJ_DCPO_V4" in _DCPO_V3_FMT_MODES
    # v2 stays OUT of the v3 format-machinery set (KARPATHY byte-identity lock).
    assert "TRIOBJ_DCPO_V2" not in _DCPO_V3_FMT_MODES


def test_v4_reward_configs_same_head_shape_as_v3():
    rc4, rc3 = REWARD_CONFIGS["TRIOBJ_DCPO_V4"], REWARD_CONFIGS["TRIOBJ_DCPO_V3"]
    assert rc4["keys"] == rc3["keys"]          # same 5 keys (R_meta SOURCE differs)
    assert len(rc4["funcs"]) == len(rc4["keys"]) == len(rc4["weights"]) == 5
    assert [f.__name__ for f in rc4["funcs"]] == [f.__name__ for f in rc3["funcs"]]


def test_v4_stage1_yaml_five_way_sync():
    ycfg = _load_yaml(os.path.join("archive", "triobj_dcpo_v4_stage1_h100_4x4k.yaml"))
    alg = ycfg["algorithm"]
    rc = REWARD_CONFIGS["TRIOBJ_DCPO_V4"]
    assert ycfg["mode"] == "TRIOBJ_DCPO_V4" and alg["sdc_mode"] == "TRIOBJ_DCPO_V4"
    assert alg["gdpo_reward_keys"] == rc["keys"]
    assert len(alg["gdpo_reward_weights"]) == len(rc["funcs"])
    # stage 1 = format only: corr 1.0 / meta 0.0 / cal 0.0 / emission 0.0 / format 0.1
    assert [float(w) for w in alg["gdpo_reward_weights"]] == [1.0, 0.0, 0.0, 0.0, 0.1]
    # yaml weights mirror the dcpo_w_* routing knobs (audit-clarity invariant).
    assert float(alg["dcpo_w_corr"]) == 1.0
    assert float(alg["dcpo_w_meta"]) == 0.0
    assert float(alg["dcpo_w_cal"]) == 0.0
    assert float(alg["dcpo_w_format"]) == 0.1
    assert float(alg["dcpo_meta_floor"]) == 0.1     # M2: floor IS the stage-1 bonus
    assert alg["dcpo_rmeta_source"] == "none"       # no likelihood scoring in stage 1
    assert alg["sdc_counterfactual"] is False       # CF machinery dormant
    # length cost is a STAGE-2 term (introduced with the w_meta warmup, spec §2):
    # stage 1 must leave the knob at its 0.0 default.
    assert float(alg.get("dcpo_len_cost", 0.0)) == 0.0


def test_v4_stage2_yaml_five_way_sync():
    ycfg = _load_yaml(os.path.join("archive", "triobj_dcpo_v4_stage2_h100_4x4k.yaml"))
    alg = ycfg["algorithm"]
    rc = REWARD_CONFIGS["TRIOBJ_DCPO_V4"]
    assert ycfg["mode"] == "TRIOBJ_DCPO_V4" and alg["sdc_mode"] == "TRIOBJ_DCPO_V4"
    assert alg["gdpo_reward_keys"] == rc["keys"]
    assert len(alg["gdpo_reward_weights"]) == len(rc["funcs"])
    assert [float(w) for w in alg["gdpo_reward_weights"]] == [1.0, 0.5, 0.3, 0.0, 0.1]
    assert float(alg["dcpo_w_meta"]) == 0.5
    assert float(alg["dcpo_w_cal"]) == 0.3
    assert int(alg["dcpo_w_meta_warmup_steps"]) == 50   # M4 warmup
    assert float(alg["dcpo_meta_floor"]) == 0.05        # I1: floor STAYS
    # spec §2 mild length cost (review round 1: in-scope, NOT deferred) — small
    # per the RLT weakness warning, warmed up alongside w_meta.
    assert 0.0 < float(alg["dcpo_len_cost"]) <= 0.05
    assert alg["dcpo_rmeta_source"] == "pmi"
    assert alg["dcpo_meta_excl_conf"] is True           # I4 conf carve-out
    assert alg["sdc_counterfactual"] is False           # CF machinery dormant
    # PMI knobs present (placeholders until the probe freezes them).
    for k in ("dcpo_pmi_agg", "dcpo_pmi_topk_frac", "dcpo_pmi_clip_token",
              "dcpo_pmi_clip_gate", "dcpo_pmi_ngram_n", "dcpo_pmi_ngram_threshold"):
        assert k in alg, f"stage2 yaml missing PMI knob {k!r}"
    # ref-worker keep-alive footgun (API scout): actor.use_kl_loss must stay
    # TRUE with coef 0.0, else verl skips ref-worker init and the PMI scorer
    # crashes on a missing ref_policy_wg.
    actor = ycfg["actor_rollout_ref"]["actor"]
    assert actor["use_kl_loss"] is True
    assert float(actor["kl_loss_coef"]) == 0.0


def test_populate_writes_every_gdpo_reward_key_v4():
    # Same regression class as the v3g step-1 crash: each configured key must
    # appear as a non_tensor_batch["<key>"] write in the populator, PLUS the
    # v4-only diagnostic-style keys (rmeta membership + warmup scale) and the
    # source-select knob.
    import inspect
    src = inspect.getsource(_populate_dcpo_region_keys)
    for key in REWARD_CONFIGS["TRIOBJ_DCPO_V4"]["keys"]:
        assert f'non_tensor_batch["{key}"]' in src, f"populator does not write {key!r}"
    assert '"dcpo_rmeta_member"' in src        # I2 centering membership
    assert '"dcpo_w_meta_scale"' in src        # M4 warmup transport
    assert "dcpo_rmeta_source" in src          # the v4 source gate
    # spec §2 length cost: subtracted from the SAME 'correctness' key inside
    # the v4 block (no 6th GDPO key) via the pure helper, warmup-coupled.
    assert "dcpo_len_cost" in src
    assert "dcpo_length_cost(" in src


def test_v4_advantage_dispatch_routes_region_path():
    # source-level: compute_sdc_gdpo_advantage must route V4 through the same
    # per-region branch as V2/V3 (a miss silently trains summed-GDPO — the v1
    # failure mode that crushed meta under the correctness broadcast).
    import inspect
    src = inspect.getsource(compute_sdc_gdpo_advantage)
    assert 'sdc_mode == "TRIOBJ_DCPO_V4"' in src


def test_pmi_scorer_temperature_hardcoded_to_one():
    # M1: the ref scorer must NOT inherit rollout.temperature (0.6 compresses
    # the delta by 1/T). Source-level because the DataProto wrap is stubbed.
    import inspect
    src = inspect.getsource(_dcpo_v4_ref_logprobs)
    assert 'meta_info["temperature"] = 1.0' in src
    # no config READ of the rollout temperature (the precedent's inheritance
    # line) — the docstring may MENTION it, so match the actual access chain.
    assert "config.actor_rollout_ref.rollout.temperature" not in src


# ═══════════════════════════════════════════════════════════════════════════
# _build_pmi_score_batches — verl-STANDARD layout (API-scout C3-class fix)
# ═══════════════════════════════════════════════════════════════════════════
def test_pmi_batch_layout_verl_standard():
    prompts = [[11, 12, 13], [21, 22, 23, 24, 25]]
    resps = [[31, 32], [41, 42, 43, 44]]
    tensors, real_n = _build_pmi_score_batches(prompts, resps, pad_to_multiple=4)
    assert real_n == 2
    p_max, r_max = 5, 4
    assert tensors["input_ids"].shape == (4, p_max + r_max)  # padded to multiple
    # prompt LEFT-PADDED INTO the full tensor; response starts exactly at P_max
    # (the precedent's left-aligned packing shifts short-prompt rows' logprobs).
    assert tensors["input_ids"][0, p_max - 3 : p_max].tolist() == [11, 12, 13]
    assert tensors["input_ids"][0, :p_max - 3].tolist() == [0, 0]
    assert tensors["input_ids"][0, p_max : p_max + 2].tolist() == [31, 32]
    assert tensors["input_ids"][1, :p_max].tolist() == [21, 22, 23, 24, 25]
    assert tensors["input_ids"][1, p_max:].tolist() == [41, 42, 43, 44]
    # attention contiguous across the P_max boundary; pads zero.
    assert tensors["attention_mask"][0].tolist() == [0, 0, 1, 1, 1, 1, 1, 0, 0]
    assert tensors["attention_mask"][1].tolist() == [1] * 9
    # response_mask full-width, marking ONLY valid response positions.
    assert tensors["response_mask"][0].tolist() == [0, 0, 0, 0, 0, 1, 1, 0, 0]
    assert tensors["response_mask"][1].tolist() == [0] * 5 + [1] * 4
    # verl position convention: clip(cumsum(attn)-1, min=0).
    assert tensors["position_ids"][0].tolist() == [0, 0, 0, 1, 2, 3, 4, 4, 4]
    assert tensors["position_ids"][1].tolist() == list(range(9))
    # prompts/responses are the column-split views the ref path asserts on.
    assert torch.equal(tensors["prompts"], tensors["input_ids"][:, :p_max])
    assert torch.equal(tensors["responses"], tensors["input_ids"][:, p_max:])
    # padding rows duplicate row 0 (skipped by the caller via real_n).
    assert torch.equal(tensors["input_ids"][2], tensors["input_ids"][0])
    assert torch.equal(tensors["input_ids"][3], tensors["input_ids"][0])


# ═══════════════════════════════════════════════════════════════════════════
# I4 conf carve-out + M4 warmup threading through _compute_dcpo_region_advantage
# ═══════════════════════════════════════════════════════════════════════════
def _carveout_setup():
    # 2 rollouts, one group. row0: ANSWER@0, META_CONTENT@{1,2}, CONF@2.
    rm = torch.ones(2, 4)
    batch = {
        "dcpo_answer_mask": torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=torch.float32),
        "dcpo_meta_content_mask": torch.tensor([[0, 1, 1, 0], [0, 1, 1, 0]], dtype=torch.float32),
        "dcpo_conf_mask": torch.tensor([[0, 0, 1, 0], [0, 0, 1, 0]], dtype=torch.float32),
    }
    non_tensor = {
        "correctness": np.zeros(2, dtype=np.float32),
        "meta_region_utility": np.asarray([1.0, -1.0], dtype=np.float32),
        "cal_region_reward": np.zeros(2, dtype=np.float32),
    }
    return rm, batch, non_tensor


def test_conf_carveout_routes_meta_off_conf_token():
    rm, batch, non_tensor = _carveout_setup()
    # knob OFF -> meta advantage lands on BOTH meta tokens incl. the conf token.
    a_off, _ = _compute_dcpo_region_advantage(
        response_mask=rm, index=["g", "g"], batch=batch, non_tensor_batch=non_tensor,
        config={"dcpo_w_meta": 0.5},
    )
    assert a_off[0, 1].item() == a_off[0, 2].item() != 0.0
    # knob ON (I4) -> the conf token carries ONLY the (zero) cal head; the rest
    # of META_CONTENT is untouched.
    a_on, _ = _compute_dcpo_region_advantage(
        response_mask=rm, index=["g", "g"], batch=batch, non_tensor_batch=non_tensor,
        config={"dcpo_w_meta": 0.5, "dcpo_meta_excl_conf": True},
    )
    assert a_on[0, 2].item() == 0.0
    assert a_on[0, 1].item() == a_off[0, 1].item() != 0.0


def test_w_meta_scale_threading_halves_meta_advantage():
    rm, batch, non_tensor = _carveout_setup()
    cfg = {"dcpo_w_meta": 0.5}
    a_full, _ = _compute_dcpo_region_advantage(
        response_mask=rm, index=["g", "g"], batch=batch, non_tensor_batch=non_tensor,
        config=cfg,
    )
    non_tensor_scaled = dict(non_tensor)
    non_tensor_scaled["dcpo_w_meta_scale"] = np.full(2, 0.5, dtype=np.float32)
    a_half, _ = _compute_dcpo_region_advantage(
        response_mask=rm, index=["g", "g"], batch=batch, non_tensor_batch=non_tensor_scaled,
        config=cfg,
    )
    assert abs(a_half[0, 1].item() - 0.5 * a_full[0, 1].item()) < 1e-6
    # answer head untouched by the meta-only scale.
    assert a_half[0, 0].item() == a_full[0, 0].item()


def test_w_meta_warmup_schedule():
    assert dcpo_w_meta_warmup_scale(0, 50) == 0.0
    assert dcpo_w_meta_warmup_scale(25, 50) == 0.5
    assert dcpo_w_meta_warmup_scale(50, 50) == 1.0
    assert dcpo_w_meta_warmup_scale(300, 50) == 1.0
    # warmup off (v2/v3 byte-identity): no knob -> scale 1 at every step.
    assert dcpo_w_meta_warmup_scale(0, 0) == 1.0
    assert dcpo_w_meta_warmup_scale(7, None) == 1.0


def test_dcpo_length_cost_schedule():
    # spec §2 mild length cost: coef * (valid_len / max_len) * warmup_scale,
    # per row, float32 — the populator subtracts this from 'correctness'.
    out = dcpo_length_cost([0, 2048, 4096], 4096, 0.02, [1.0, 1.0, 0.5])
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [0.0, 0.01, 0.01], rtol=1e-6)
    # warmup at step 0 (scale 0.0) silences the cost entirely (M4 coupling).
    assert not dcpo_length_cost([4096], 4096, 0.02, [0.0]).any()
    # coef 0.0 (the knob default) -> exact zeros: v4-off paths byte-identical.
    assert not dcpo_length_cost([4096, 17], 4096, 0.0, [1.0, 1.0]).any()


# ═══════════════════════════════════════════════════════════════════════════
# I2 rmeta_member_mask centering at the compose level
# ═══════════════════════════════════════════════════════════════════════════
def _compose_kwargs(B=4, T=3):
    return dict(
        response_mask=torch.ones(B, T),
        index=["g"] * B,
        R_corr=[1.0, -1.0, 1.0, -1.0],
        R_meta=[1.0, 0.0, 0.0, 0.0],
        R_cal=[0.0] * B,
        answer_mask=torch.tensor([[1, 0, 0]] * B, dtype=torch.float32),
        meta_content_mask=torch.tensor([[0, 1, 0]] * B, dtype=torch.float32),
        conf_mask=torch.zeros(B, T),
        w_corr=1.0, w_meta=1.0, w_cal=0.3,
    )


def test_rmeta_member_mask_centers_over_members_only():
    # rows 2,3 never had a PMI computed: structural zeros, member 0.
    A, _ = compose_dcpo_region_advantage(
        **_compose_kwargs(), rmeta_member_mask=[1.0, 1.0, 0.0, 0.0],
    )
    # centered over members {1.0, 0.0} -> mean 0.5 -> +0.5 / -0.5 on the meta token.
    assert abs(A[0, 1].item() - 0.5) < 1e-6
    assert abs(A[1, 1].item() + 0.5) < 1e-6
    # non-members receive EXACTLY 0 on their meta span (no spurious centered pull).
    assert A[2, 1].item() == 0.0 and A[3, 1].item() == 0.0
    # R_corr centering is NOT affected by the rmeta-only membership.
    A_default, _ = compose_dcpo_region_advantage(**_compose_kwargs())
    assert torch.equal(A[:, 0], A_default[:, 0])


def test_rmeta_member_mask_none_is_byte_identical():
    A_old, _ = compose_dcpo_region_advantage(**_compose_kwargs())
    A_new, _ = compose_dcpo_region_advantage(**_compose_kwargs(), rmeta_member_mask=None)
    assert torch.equal(A_old, A_new)


# ═══════════════════════════════════════════════════════════════════════════
# _compute_dcpo_v4_pmi_rmeta end-to-end (fake tokenizer + patched ref scorer)
# ═══════════════════════════════════════════════════════════════════════════
def test_compute_pmi_rmeta_selection_guard_membership(monkeypatch):
    tok = FakeMergeTokenizer(merges=())  # char-level, no boundary merges
    # row0: trusted, closed meta, correct -> scored, positive (delta +1/token).
    # row1: no meta emitted -> not attempted.
    # row2: trusted but the meta states the boxed answer "7" -> C2 guard, 0.
    # row3: trusted, closed meta, EMPTY continuation -> SpliceAlignmentError.
    response_texts = [
        "work<|meta|>check the sum<|/meta|>so it is 42.",
        "plain answer 9.",
        "x<|meta|>answer 7 surely<|/meta|>boxed 7.",
        "y<|meta|>hm<|/meta|>",
    ]
    fmt_classes = ["wellformed", "no_meta", "wellformed", "wellformed"]
    heads = {
        "c_with": [1.0, 1.0, 1.0, 1.0],
        "answer": ["42", "9", "7", ""],
    }

    # with-arms are rows [0, n_al), without-arms rows [n_al, 2*n_al); padding
    # duplicates row 0. Give with-arms -1.0 and everything else -2.0 so
    # delta = +1.0 per C token for every aligned row.
    def _fake_ref2(trainer, tensors):
        n, r_max = tensors["responses"].shape
        out = torch.full((n, r_max), -2.0)
        out[:2] = -1.0  # n_al == 2 (rows 0 and 2 align; row 3 fails alignment)
        return out

    monkeypatch.setattr(V, "_dcpo_v4_ref_logprobs", _fake_ref2)
    r_meta, member = _compute_dcpo_v4_pmi_rmeta(
        tokenizer=tok,
        trainer=object(),  # config reads fall back to defaults (try/except)
        prompt_texts=["P0 ", "P1 ", "P2 ", "P3 "],
        response_texts=response_texts,
        fmt_classes=fmt_classes,
        heads=heads,
        read_knob=lambda name, default: default,
        step=3,
    )
    assert r_meta.shape == (4,) and member.shape == (4,)
    assert r_meta[0] > 0.0 and member[0] == 1.0      # scored, correct, +delta
    assert r_meta[1] == 0.0 and member[1] == 0.0     # no meta -> not attempted
    assert r_meta[2] == 0.0 and member[2] == 0.0     # answer-leak guard (C2/M3 gate)
    assert r_meta[3] == 0.0 and member[3] == 0.0     # alignment failure (C3)


def test_compute_pmi_rmeta_wrong_row_gets_nonpositive(monkeypatch):
    tok = FakeMergeTokenizer(merges=())
    # one WRONG row whose meta still RAISES the continuation likelihood:
    # sign gate (M3) must flip the clipped-positive delta to <= 0.
    def _fake_ref(trainer, tensors):
        n, r_max = tensors["responses"].shape
        out = torch.full((n, r_max), -2.0)
        out[:1] = -1.0
        return out

    monkeypatch.setattr(V, "_dcpo_v4_ref_logprobs", _fake_ref)
    r_meta, member = _compute_dcpo_v4_pmi_rmeta(
        tokenizer=tok,
        trainer=object(),
        prompt_texts=["P "],
        response_texts=["w<|meta|>try again<|/meta|>it equals 5."],
        fmt_classes=["wellformed"],
        heads={"c_with": [0.0], "answer": ["5"]},
        read_knob=lambda name, default: default,
        step=1,
    )
    assert r_meta[0] < 0.0          # (+|delta| under a wrong outcome) -> negative
    assert member[0] == 1.0         # still a scored row (centering population)


def test_v4_rmeta_source_must_be_explicit():
    # Round 2 M-A: missing/unreadable knob RAISES — the old silent 'cf' default
    # fell open onto the deprecated CF path with plausible nonzero values.
    with pytest.raises(ValueError, match="dcpo_rmeta_source"):
        _v4_rmeta_source_strict(lambda name, default: default)   # knob absent
    with pytest.raises(ValueError, match="dcpo_rmeta_source"):
        _v4_rmeta_source_strict(lambda name, default: None)      # yaml null
    with pytest.raises(ValueError, match="not in"):
        _v4_rmeta_source_strict(lambda name, default: "bogus")   # invalid value
    # the three explicit values pass through ('cf' allowed as OPT-IN only)
    for src in ("cf", "pmi", "none"):
        assert _v4_rmeta_source_strict(lambda n, d, s=src: s) == src
    # source-level: the populator uses the strict reader, not a 'cf'-defaulted
    # read (the fail-open line this fix removes).
    import inspect
    pop = inspect.getsource(_populate_dcpo_region_keys)
    assert "_v4_rmeta_source_strict" in pop
    assert '_v4_read("dcpo_rmeta_source", "cf")' not in pop


def test_compute_pmi_rmeta_nonfinite_logprob_zeroes_row_and_member(monkeypatch):
    # Round 2 IMPORTANT-3 at the verl_sdc layer: NaN in one arm's ref logprobs
    # -> that row R_meta 0 AND member 0 (out of the centering population);
    # healthy sibling rows in the same batch keep scoring.
    tok = FakeMergeTokenizer(merges=())

    def _fake_ref(trainer, tensors):
        n, r_max = tensors["responses"].shape
        out = torch.full((n, r_max), -2.0)
        out[:2] = -1.0                     # n_al == 2 with-arms
        out[0, 1] = float("nan")           # poison row 0's with-arm
        return out

    monkeypatch.setattr(V, "_dcpo_v4_ref_logprobs", _fake_ref)
    r_meta, member = _compute_dcpo_v4_pmi_rmeta(
        tokenizer=tok,
        trainer=object(),
        prompt_texts=["P0 ", "P1 "],
        response_texts=[
            "a<|meta|>check once<|/meta|>so it is 42.",
            "b<|meta|>check twice<|/meta|>so it is 43.",
        ],
        fmt_classes=["wellformed", "wellformed"],
        heads={"c_with": [1.0, 1.0], "answer": ["", ""]},
        read_knob=lambda name, default: default,
        step=2,
    )
    assert r_meta[0] == 0.0 and member[0] == 0.0      # poisoned: fail-closed
    assert np.isfinite(r_meta).all()                  # never NaN out of here
    assert r_meta[1] > 0.0 and member[1] == 1.0       # sibling unaffected


def test_compute_pmi_rmeta_whitespace_continuation_not_attempted():
    # Round 2 M-D: split_first_meta's STRICTER probe semantics — a whitespace-
    # only continuation is not attempted, so the ref scorer is never called
    # (a call would raise here) and the row scores 0 with member 0.
    r_meta, member = _compute_dcpo_v4_pmi_rmeta(
        tokenizer=FakeMergeTokenizer(merges=()),
        trainer=object(),
        prompt_texts=["P "],
        response_texts=["w<|meta|>hm<|/meta|>   \n"],
        fmt_classes=["wellformed"],
        heads={"c_with": [1.0], "answer": ["5"]},
        read_knob=lambda name, default: default,
        step=1,
    )
    assert r_meta[0] == 0.0 and member[0] == 0.0
    # source-level lock: BOTH consumers route through the shared splitter.
    import inspect
    assert "split_first_meta" in inspect.getsource(_compute_dcpo_v4_pmi_rmeta)
    import scripts.probe_pmi_offline as probe
    assert "split_first_meta" in inspect.getsource(probe.parse_rollout)


class _FakeWandb(types.ModuleType):
    def __init__(self):
        super().__init__("wandb")
        self.run = object()      # truthy: logging path active
        self.logged = []

    def log(self, payload, step=None):
        self.logged.append((dict(payload), step))


def test_compute_pmi_rmeta_ref_failure_logs_zero_scalars(monkeypatch):
    # Round 2 M-C: the early returns must chart ZEROS, not wandb gaps.
    import sys
    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    def _boom(trainer, tensors):
        raise RuntimeError("ref worker down")

    monkeypatch.setattr(V, "_dcpo_v4_ref_logprobs", _boom)
    _compute_dcpo_v4_pmi_rmeta(
        tokenizer=FakeMergeTokenizer(merges=()),
        trainer=object(),
        prompt_texts=["P "],
        response_texts=["w<|meta|>try<|/meta|>it equals 5."],
        fmt_classes=["wellformed"],
        heads={"c_with": [1.0], "answer": [""]},
        read_knob=lambda name, default: default,
        step=7,
    )
    assert len(fake_wandb.logged) == 1
    payload, step = fake_wandb.logged[0]
    assert step == 7
    assert payload["dcpo/pmi_member_rate"] == 0.0
    assert payload["dcpo/pmi_guard_hit_rate"] == 0.0
    assert payload["dcpo/pmi_nonfinite_rate"] == 0.0
    assert payload["dcpo/pmi_attempted_rate"] == 1.0   # the row WAS attempted
    assert payload["dcpo/pmi_aligned_rate"] == 1.0     # and aligned pre-failure


def test_compute_pmi_rmeta_no_aligned_logs_zero_scalars(monkeypatch):
    # the other early return: attempted > 0 but every splice fails to align
    # (whole continuation merges across the boundary on the merge tokenizer).
    import sys
    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    r_meta, member = _compute_dcpo_v4_pmi_rmeta(
        tokenizer=FakeMergeTokenizer(merges=("ab",)),
        trainer=object(),
        prompt_texts=["a"],
        response_texts=["<|meta|>m<|/meta|>b"],   # prefix 'a' + C 'b' -> 'ab' merge
        fmt_classes=["wellformed"],
        heads={"c_with": [1.0], "answer": [""]},
        read_knob=lambda name, default: default,
        step=9,
    )
    assert float(member.sum()) == 0.0
    assert len(fake_wandb.logged) == 1
    payload, step = fake_wandb.logged[0]
    assert step == 9
    assert payload["dcpo/pmi_attempted_rate"] == 1.0
    assert payload["dcpo/pmi_aligned_rate"] == 0.0
    assert payload["dcpo/pmi_member_rate"] == 0.0


def test_compute_pmi_rmeta_ref_failure_is_crash_safe(monkeypatch):
    tok = FakeMergeTokenizer(merges=())

    def _boom(trainer, tensors):
        raise RuntimeError("ref worker down")

    monkeypatch.setattr(V, "_dcpo_v4_ref_logprobs", _boom)
    r_meta, member = _compute_dcpo_v4_pmi_rmeta(
        tokenizer=tok,
        trainer=object(),
        prompt_texts=["P "],
        response_texts=["w<|meta|>try<|/meta|>it equals 5."],
        fmt_classes=["wellformed"],
        heads={"c_with": [1.0], "answer": ["5"]},
        read_knob=lambda name, default: default,
        step=1,
    )
    assert float(np.abs(r_meta).sum()) == 0.0 and float(member.sum()) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# M1 runtime guard (review round 1): the T=1.0 hardcode only survives on the
# ENGINE worker path — the legacy fsdp worker clobbers meta_info["temperature"]
# with rollout.temperature AFTER the caller sets it (silent 1/T compression).
# ═══════════════════════════════════════════════════════════════════════════
def _guard_trainer(legacy):
    return types.SimpleNamespace(
        config=types.SimpleNamespace(
            trainer=types.SimpleNamespace(
                use_legacy_worker_impl=legacy, nnodes=1, n_gpus_per_node=1)))


def test_pmi_ref_guard_rejects_legacy_worker_path():
    # legacy path enabled -> the scorer must CRASH (AssertionError), and the
    # crash-safe wrapper must RE-RAISE it (deterministic misconfig), never
    # swallow it into an all-zero flatline.
    with pytest.raises(AssertionError, match="legacy fsdp worker"):
        _compute_dcpo_v4_pmi_rmeta(
            tokenizer=FakeMergeTokenizer(merges=()),
            trainer=_guard_trainer("enable"),
            prompt_texts=["P "],
            response_texts=["w<|meta|>try<|/meta|>it equals 5."],
            fmt_classes=["wellformed"],
            heads={"c_with": [1.0], "answer": ["5"]},
            read_knob=lambda name, default: default,
            step=1,
        )


def test_pmi_ref_guard_unreadable_config_fails_closed():
    # config without the knob (standalone yaml copy / future verl rename):
    # fail CLOSED rather than risk the silent 1/T compression.
    with pytest.raises(AssertionError, match="unreadable"):
        _dcpo_v4_ref_logprobs(object(), {})


def test_pmi_ref_guard_engine_path_passes_and_sets_temperature_one(monkeypatch):
    # disable (the inherited shared-yaml value) passes the guard, and the
    # batch handed to the ref worker carries meta_info["temperature"] == 1.0
    # (functional M1 lock, beyond the source-level assert above).
    captured = {}

    class _FakeProto:
        def __init__(self, tensors):
            self.tensors = tensors
            self.meta_info = {}

        @classmethod
        def from_dict(cls, tensors):
            return cls(tensors)

    def _ref(batch):
        captured["temperature"] = batch.meta_info.get("temperature")
        n, r_max = batch.tensors["responses"].shape
        return types.SimpleNamespace(batch={"ref_log_prob": torch.zeros(n, r_max)})

    trainer = _guard_trainer("disable")
    trainer._compute_ref_log_prob = _ref
    monkeypatch.setattr(V, "DataProto", _FakeProto)
    tensors, real_n = _build_pmi_score_batches([[1, 2]], [[3, 4]], 1)
    out = _dcpo_v4_ref_logprobs(trainer, tensors)
    assert captured["temperature"] == 1.0
    assert real_n == 1 and out.shape == (1, 2)


def test_compute_pmi_rmeta_placebo_corrected_three_arm_wiring(monkeypatch):
    # verl-layer lock for the placebo-corrected path (review 2b83bf3 minor-4):
    # arm layout [0,n)=with, [n,2n)=without, [2n,2n+p)=placebo; the gated value
    # must be agg(d_real) - agg(d_placebo), and a row is member only when its
    # placebo arm scored.
    tok = FakeMergeTokenizer(merges=())
    response_texts = [
        "work<|meta|>check the sum<|/meta|>so it is 42.",   # scored + corrected
        "plain answer 9.",                                   # not attempted
    ]
    heads = {"c_with": [1.0, 1.0], "answer": ["42", "9"]}

    def _fake_ref3(trainer, tensors):
        n, r_max = tensors["responses"].shape
        out = torch.full((n, r_max), -2.0)   # without-arm + padding
        out[0] = -1.0                        # with-arm (n_al == 1)
        out[2] = -1.5                        # placebo arm [2n, 2n+1)
        return out

    knobs = {"dcpo_pmi_placebo_correct": True, "dcpo_pmi_agg": "mean"}
    monkeypatch.setattr(V, "_dcpo_v4_ref_logprobs", _fake_ref3)
    r_meta, member = _compute_dcpo_v4_pmi_rmeta(
        tokenizer=tok,
        trainer=object(),
        prompt_texts=["P0 ", "P1 "],
        response_texts=response_texts,
        fmt_classes=["wellformed", "no_meta"],
        heads=heads,
        read_knob=lambda name, default: knobs.get(name, default),
        step=7,
    )
    # real delta +1.0/tok, placebo delta +0.5/tok -> corrected +0.5
    assert r_meta[0] == pytest.approx(0.5)
    assert member[0] == 1.0
    assert r_meta[1] == 0.0 and member[1] == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# spec 2026-06-15: anchor / emit-route / meta-len-cap knob forwarding
# ═══════════════════════════════════════════════════════════════════════════
def _run_min_compute(config_overrides=None):
    # Minimal compute call reusing the carve-out fixture (no _run_min_compute
    # helper existed in this file; mirror the integration fixture builder).
    rm, batch, non_tensor = _carveout_setup()
    cfg = {"dcpo_w_meta": 0.5}
    if config_overrides:
        cfg.update(config_overrides)
    return _compute_dcpo_region_advantage(
        response_mask=rm, index=["g", "g"], batch=batch,
        non_tensor_batch=non_tensor, config=cfg,
    )


def test_anchor_knobs_forwarded(monkeypatch):
    captured = {}
    import src.training.dcpo_region as R
    real = R.compose_dcpo_region_advantage
    def spy(**kw):
        captured.update(kw); return real(**kw)
    monkeypatch.setattr(R, "compose_dcpo_region_advantage", spy)
    _run_min_compute(config_overrides={"dcpo_anchor_norm": True,
                                       "dcpo_emit_route": "first_token",
                                       "dcpo_meta_len_cap": 3})
    assert captured.get("anchor_norm") is True
    assert captured.get("emit_route") == "first_token"
    assert captured.get("meta_len_cap") == 3
    assert captured.get("anchor_ema_state") is not None   # module-level dict passed


# ═══════════════════════════════════════════════════════════════════════════
# PMI-SHIFT-ACROSS-META (design 2026-06-25): source registration + two-position
# teacher-forcing end-to-end + production-parity (non-inert advantage routing).
# ═══════════════════════════════════════════════════════════════════════════
def test_pmi_shift_registered_as_v4_source():
    from src.training.verl_sdc import _V4_RMETA_SOURCES
    assert "pmi_shift" in _V4_RMETA_SOURCES
    assert _v4_rmeta_source_strict(lambda n, d: "pmi_shift") == "pmi_shift"


def _shift_fake_ref(per_arm_const):
    """Build a fake _dcpo_v4_ref_logprobs that returns a CONSTANT logp per arm.

    per_arm_const: list of 4 constants in arm order [gold@open, decoy@open,
    gold@close, decoy@close]; tiled over the n rows the batch builder padded to.
    Each arm gets a flat per-token logp, so the summed divergent-token PMI is
    const * n_divergent_tokens; gold-minus-decoy isolates the open/close shift.
    """
    def _ref(trainer, tensors):
        n, r_max = tensors["responses"].shape
        out = torch.zeros((n, r_max))
        for j in range(n):
            out[j, :] = per_arm_const[j % 4]
        return out
    return _ref


def test_pmi_shift_save_reversal_positive(monkeypatch):
    # decoy->gold SAVE: gold worse than decoy at OPEN (pmi_open<0), gold better at
    # CLOSE (pmi_close>0). gold@open=-2, decoy@open=-1 -> open diff = -1*Ntok < 0;
    # gold@close=-1, decoy@close=-2 -> close diff = +1*Ntok > 0 -> reversal +save.
    tok = FakeMergeTokenizer(merges=())
    monkeypatch.setattr(V, "_dcpo_v4_ref_logprobs",
                        _shift_fake_ref([-2.0, -1.0, -1.0, -2.0]))
    r_meta, member, shift_raw = _compute_dcpo_v4_pmi_shift_rmeta(
        tokenizer=tok, trainer=object(),
        prompt_texts=["P "],
        response_texts=["w<|meta|>recheck<|/meta|>so it is 42."],
        ground_truths=["42"],
        fmt_classes=["wellformed"],
        heads={"c_with": [1.0]},
        read_knob=lambda name, default: default,
        step=1,
    )
    assert member[0] == 1.0
    assert r_meta[0] > 0.0           # SAVE bonus present
    assert shift_raw[0] > 0.0        # pmi_close - pmi_open > 0


def test_pmi_shift_derail_reversal_strongly_negative(monkeypatch):
    # gold->decoy DERAIL: gold better at OPEN (pmi_open>0), worse at CLOSE
    # (pmi_close<0). The asymmetric derail penalty (default 2.0 > save 1.0) makes
    # |R_derail| > the SAVE magnitude of the mirror case.
    tok = FakeMergeTokenizer(merges=())
    monkeypatch.setattr(V, "_dcpo_v4_ref_logprobs",
                        _shift_fake_ref([-1.0, -2.0, -2.0, -1.0]))
    r_meta, member, shift_raw = _compute_dcpo_v4_pmi_shift_rmeta(
        tokenizer=tok, trainer=object(),
        prompt_texts=["P "],
        response_texts=["w<|meta|>recheck<|/meta|>so it is 42."],
        ground_truths=["42"],
        fmt_classes=["wellformed"],
        heads={"c_with": [1.0]},
        read_knob=lambda name, default: default,
        step=1,
    )
    assert member[0] == 1.0
    assert r_meta[0] < 0.0
    assert shift_raw[0] < 0.0

    # mirror SAVE case for asymmetry comparison
    monkeypatch.setattr(V, "_dcpo_v4_ref_logprobs",
                        _shift_fake_ref([-2.0, -1.0, -1.0, -2.0]))
    r_save, _, _ = _compute_dcpo_v4_pmi_shift_rmeta(
        tokenizer=tok, trainer=object(),
        prompt_texts=["P "],
        response_texts=["w<|meta|>recheck<|/meta|>so it is 42."],
        ground_truths=["42"],
        fmt_classes=["wellformed"],
        heads={"c_with": [1.0]},
        read_knob=lambda name, default: default,
        step=1,
    )
    assert abs(r_meta[0]) >= abs(r_save[0])   # derail >= save (asymmetric)


def test_pmi_shift_no_meta_not_attempted(monkeypatch):
    tok = FakeMergeTokenizer(merges=())
    monkeypatch.setattr(V, "_dcpo_v4_ref_logprobs",
                        _shift_fake_ref([-2.0, -1.0, -1.0, -2.0]))
    r_meta, member, _ = _compute_dcpo_v4_pmi_shift_rmeta(
        tokenizer=tok, trainer=object(),
        prompt_texts=["P "],
        response_texts=["plain answer 9 no meta"],
        ground_truths=["9"],
        fmt_classes=["no_meta"],
        heads={"c_with": [1.0]},
        read_knob=lambda name, default: default,
        step=1,
    )
    assert r_meta[0] == 0.0 and member[0] == 0.0


def test_pmi_shift_ref_failure_crash_safe(monkeypatch):
    tok = FakeMergeTokenizer(merges=())

    def _boom(trainer, tensors):
        raise RuntimeError("ref worker exploded")

    monkeypatch.setattr(V, "_dcpo_v4_ref_logprobs", _boom)
    r_meta, member, shift_raw = _compute_dcpo_v4_pmi_shift_rmeta(
        tokenizer=tok, trainer=object(),
        prompt_texts=["P "],
        response_texts=["w<|meta|>recheck<|/meta|>so it is 42."],
        ground_truths=["42"],
        fmt_classes=["wellformed"],
        heads={"c_with": [1.0]},
        read_knob=lambda name, default: default,
        step=1,
    )
    assert float(r_meta[0]) == 0.0 and float(member[0]) == 0.0
    assert np.isnan(shift_raw[0])


def test_pmi_shift_entry_guard_rejects_readable_legacy_config(monkeypatch):
    # A READABLE-but-WRONG use_legacy_worker_impl must crash at the pmi_shift entry
    # (before any work), with the T=1.0/PMI-compression message.
    tok = FakeMergeTokenizer(merges=())
    monkeypatch.setattr(V, "_dcpo_v4_ref_logprobs",
                        _shift_fake_ref([-2.0, -1.0, -1.0, -2.0]))
    with pytest.raises(AssertionError, match="use_legacy_worker_impl=disable"):
        _compute_dcpo_v4_pmi_shift_rmeta(
            tokenizer=tok, trainer=_guard_trainer("enable"),
            prompt_texts=["P "],
            response_texts=["w<|meta|>recheck<|/meta|>so it is 42."],
            ground_truths=["42"],
            fmt_classes=["wellformed"],
            heads={"c_with": [1.0]},
            read_knob=lambda name, default: default,
            step=1,
        )


def test_pmi_shift_entry_guard_unreadable_config_defers(monkeypatch):
    # trainer=object() (unreadable config) must NOT crash at the entry — the deep
    # ref-scorer assert handles the real path; with the ref monkeypatched it scores.
    tok = FakeMergeTokenizer(merges=())
    monkeypatch.setattr(V, "_dcpo_v4_ref_logprobs",
                        _shift_fake_ref([-2.0, -1.0, -1.0, -2.0]))
    r_meta, member, _ = _compute_dcpo_v4_pmi_shift_rmeta(
        tokenizer=tok, trainer=object(),
        prompt_texts=["P "],
        response_texts=["w<|meta|>recheck<|/meta|>so it is 42."],
        ground_truths=["42"],
        fmt_classes=["wellformed"],
        heads={"c_with": [1.0]},
        read_knob=lambda name, default: default,
        step=1,
    )
    assert member[0] == 1.0


def test_pmi_shift_empty_meta_content_skipped(monkeypatch):
    # A meta block with NO content between the tags makes CLOSE==OPEN by
    # construction (spurious null) and lets PRESENCE earn credit -> must be skipped.
    tok = FakeMergeTokenizer(merges=())
    monkeypatch.setattr(V, "_dcpo_v4_ref_logprobs",
                        _shift_fake_ref([-2.0, -1.0, -1.0, -2.0]))
    r_meta, member, shift_raw = _compute_dcpo_v4_pmi_shift_rmeta(
        tokenizer=tok, trainer=object(),
        prompt_texts=["P "],
        response_texts=["w<|meta|>   <|/meta|>so it is 42."],  # whitespace-only meta
        ground_truths=["42"],
        fmt_classes=["wellformed"],
        heads={"c_with": [1.0]},
        read_knob=lambda name, default: default,
        step=1,
    )
    assert r_meta[0] == 0.0 and member[0] == 0.0
    assert np.isnan(shift_raw[0])


def test_pmi_shift_dup_meta_skipped_when_thresh_enabled(monkeypatch):
    # When the meta-inner text is a near-duplicate of the body prefix and the
    # dup-threshold knob is enabled (<1.0), the row is skipped (presence-confound
    # guard). With the default thresh (1.0, disabled) the same row scores.
    tok = FakeMergeTokenizer(merges=())
    monkeypatch.setattr(V, "_dcpo_v4_ref_logprobs",
                        _shift_fake_ref([-2.0, -1.0, -1.0, -2.0]))
    body_and_meta = "check the sum<|meta|>check the sum<|/meta|>so it is 42."

    def _knob_dup(name, default):
        return 0.5 if name == "dcpo_pmishift_meta_body_dup_thresh" else default

    r_meta, member, _ = _compute_dcpo_v4_pmi_shift_rmeta(
        tokenizer=tok, trainer=object(),
        prompt_texts=["P "], response_texts=[body_and_meta],
        ground_truths=["42"], fmt_classes=["wellformed"],
        heads={"c_with": [1.0]}, read_knob=_knob_dup, step=1,
    )
    assert r_meta[0] == 0.0 and member[0] == 0.0   # duplicate meta skipped

    # default thresh disabled -> the SAME duplicate-meta row scores (non-inert).
    r_meta2, member2, _ = _compute_dcpo_v4_pmi_shift_rmeta(
        tokenizer=tok, trainer=object(),
        prompt_texts=["P "], response_texts=[body_and_meta],
        ground_truths=["42"], fmt_classes=["wellformed"],
        heads={"c_with": [1.0]},
        read_knob=lambda name, default: default, step=1,
    )
    assert member2[0] == 1.0


def test_pmi_shift_production_parity_non_inert():
    # PRODUCTION-PARITY (gs190 anti-inert): R_shift routed to meta_region_utility
    # + dcpo_rmeta_member must actually CHANGE the composed advantage on the META
    # token vs an all-zero R_shift. (Same compose path the populator writes into.)
    base = _compose_kwargs()
    A_zero, _ = compose_dcpo_region_advantage(
        **{**base, "R_meta": [0.0, 0.0, 0.0, 0.0]},
        rmeta_member_mask=[1.0, 1.0, 0.0, 0.0],
    )
    A_shift, _ = compose_dcpo_region_advantage(
        **{**base, "R_meta": [3.0, -4.0, 0.0, 0.0]},
        rmeta_member_mask=[1.0, 1.0, 0.0, 0.0],
    )
    # the meta-token advantage column (index 1) must differ -> head is NON-INERT.
    assert not torch.equal(A_zero[:, 1], A_shift[:, 1])
    # the correctness column is untouched by the R_meta head.
    assert torch.equal(A_zero[:, 0], A_shift[:, 0])


def test_pmi_shift_existing_arms_byte_identical():
    # Adding pmi_shift to the source set must not alter the other sources'
    # validation (cf/pmi/none still pass; gm/asym_cf unchanged).
    for src in ("cf", "pmi", "none", "cf_group", "decoy_did_gm",
                "decoy_did_rlsd", "asym_cf"):
        assert _v4_rmeta_source_strict(lambda n, d, s=src: s) == src
