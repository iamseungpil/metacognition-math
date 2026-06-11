"""v3m anti-collapse FLOOR + CF signature suppression (2026-06-11).

The collapse root cause (v3l): meta_emit 0.5→0 by step 60. The FORMAT penalty
punished malformed meta delimiters, and "stop emitting meta" avoided the penalty
entirely while the only counter-reward (R_meta) was silent. v3m adds:
  (1) a small UN-CENTERED +meta_floor on TRUSTED-meta rows' META_CONTENT tokens
      (post group-mean-subtract so it survives — a pre-centering constant cancels),
  (2) CF signature suppression so c_without grades more often (R_meta less silent).
"""

import numpy as np
import torch

from src.training.dcpo_region import (
    compose_dcpo_region_advantage,
    signature_suppression_ids,
    TRUSTED_META_CLASSES,
)


# ── layout T=4: [ans, TAG, meta_c, conf]; B=2 group ─────────────────────────
def _base_kwargs():
    return dict(
        response_mask=torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.float32),
        index=["g", "g"],
        R_corr=np.asarray([0.0, 0.0], dtype=np.float32),
        R_meta=np.asarray([0.0, 0.0], dtype=np.float32),
        R_cal=np.asarray([0.0, 0.0], dtype=np.float32),
        answer_mask=torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=torch.float32),
        meta_content_mask=torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=torch.float32),
        conf_mask=torch.tensor([[0, 0, 0, 1], [0, 0, 0, 1]], dtype=torch.float32),
    )


# ═══════════════════════════════════════════════════════════════════════════
# FLOOR — the cancellation-avoidance crux
# ═══════════════════════════════════════════════════════════════════════════
def test_floor_survives_when_centered_meta_cancels():
    # ALL rows identical R_meta → centered Â_meta = 0 for every meta token. A
    # constant folded into R_meta BEFORE centering would vanish here. Routed
    # post-centering, the floor must still land on the trusted rows' META_CONTENT
    # tokens — this is the whole point of v3m. PER-ROW total = meta_floor (here 2
    # meta tokens idx 2,3 → 0.05 each), length-neutral.
    A, _ = compose_dcpo_region_advantage(
        **_base_kwargs(), meta_floor=0.1, floor_mask=np.asarray([1.0, 1.0], np.float32))
    assert torch.allclose(A[0, 2], torch.tensor(0.05))
    assert torch.allclose(A[0, 3], torch.tensor(0.05))
    assert torch.allclose(A[0, 2:4].sum(), torch.tensor(0.1))   # row TOTAL = meta_floor
    assert torch.allclose(A[1, 2:4].sum(), torch.tensor(0.1))


def test_floor_is_length_neutral_per_row():
    # row with 1 meta token vs row with 3 meta tokens → BOTH rows total +meta_floor
    # (no length-farm: verbose meta does not harvest more floor).
    kw = _base_kwargs()
    kw["meta_content_mask"] = torch.tensor(
        [[0, 0, 0, 1], [0, 1, 1, 1]], dtype=torch.float32)  # 1 vs 3 meta tokens
    A, _ = compose_dcpo_region_advantage(
        **kw, meta_floor=0.1, floor_mask=np.asarray([1.0, 1.0], np.float32))
    assert torch.allclose(A[0].sum(), torch.tensor(0.1))   # 1-token row: total 0.1
    assert torch.allclose(A[1, 1:4].sum(), torch.tensor(0.1))  # 3-token row: ALSO 0.1
    assert torch.allclose(A[1, 1], torch.tensor(0.1 / 3))  # each of 3 → meta_floor/3


def test_floor_only_on_meta_content_not_answer_or_tag():
    A, _ = compose_dcpo_region_advantage(
        **_base_kwargs(), meta_floor=0.1, floor_mask=np.asarray([1.0, 1.0], np.float32))
    # answer idx 0 and TAG idx 1 must NOT receive the floor (R_corr=0 → 0).
    assert A[0, 0] == 0.0
    assert A[0, 1] == 0.0


def test_floor_skips_untrusted_rows():
    # row 0 trusted (1.0), row 1 untrusted (0.0, e.g. discard/no_meta).
    A, _ = compose_dcpo_region_advantage(
        **_base_kwargs(), meta_floor=0.1, floor_mask=np.asarray([1.0, 0.0], np.float32))
    assert torch.allclose(A[0, 2], torch.tensor(0.05))  # trusted gets floor (2 tokens)
    assert A[1, 2] == 0.0                                 # untrusted gets none


def test_floor_rides_on_top_of_centered_rmeta():
    # R_meta {+1,-1}: centered Â_meta = {+1,-1}; meta token gets w_meta*Â + floor/n.
    kw = _base_kwargs()
    kw["R_meta"] = np.asarray([1.0, -1.0], dtype=np.float32)
    A, _ = compose_dcpo_region_advantage(
        **kw, meta_floor=0.1, floor_mask=np.asarray([1.0, 1.0], np.float32))
    # 2 meta tokens → floor 0.05 each. row0: 0.5*(+1)+0.05=0.55 ; row1: 0.5*(-1)+0.05=-0.45
    assert torch.allclose(A[0, 2], torch.tensor(0.55))
    assert torch.allclose(A[1, 2], torch.tensor(-0.45))


def test_floor_default_is_byte_identical():
    kw = _base_kwargs()
    kw["R_meta"] = np.asarray([1.0, -1.0], dtype=np.float32)
    A_old, _ = compose_dcpo_region_advantage(**kw)
    A_new, _ = compose_dcpo_region_advantage(**kw, meta_floor=0.0, floor_mask=None)
    assert torch.equal(A_old, A_new)
    # floor_mask given but meta_floor 0 → still no-op (guard is `if meta_floor`).
    A_zero, _ = compose_dcpo_region_advantage(
        **kw, meta_floor=0.0, floor_mask=np.asarray([1.0, 1.0], np.float32))
    assert torch.equal(A_old, A_zero)


def test_floor_respects_response_mask():
    # padding token (rm=0) at the meta position must stay 0 even with floor.
    kw = _base_kwargs()
    kw["response_mask"] = torch.tensor([[1, 1, 0, 1], [1, 1, 1, 1]], dtype=torch.float32)
    A, _ = compose_dcpo_region_advantage(
        **kw, meta_floor=0.1, floor_mask=np.asarray([1.0, 1.0], np.float32))
    assert A[0, 2] == 0.0   # masked-out position: no floor leaks through


# ═══════════════════════════════════════════════════════════════════════════
# TRUSTED set — which classes are floor-eligible
# ═══════════════════════════════════════════════════════════════════════════
def test_trusted_set_includes_recovered_classes_excludes_discard():
    for c in ("wellformed", "swapped", "dup_open", "reversed", "drift"):
        assert c in TRUSTED_META_CLASSES
    for c in ("discard", "truncation", "no_meta"):
        assert c not in TRUSTED_META_CLASSES


# ═══════════════════════════════════════════════════════════════════════════
# CF signature suppression ids
# ═══════════════════════════════════════════════════════════════════════════
def test_signature_ids_takes_first_token_unique_sorted():
    # fake tokenizer: first token id = len of the (stripped) variant, deterministic.
    fake = {
        "confidence": [11, 9], " confidence": [11, 9], "Confidence": [12],
        " Confidence": [12], "assessment": [20], " assessment": [20],
        "Assessment": [21], " Assessment": [21], "action": [30], " action": [30],
        "Action": [31], " Action": [31],
    }
    ids = signature_suppression_ids(lambda s: fake.get(s, []))
    assert ids == sorted(set([11, 12, 20, 21, 30, 31]))   # unique + sorted


def test_signature_ids_tolerates_encode_failure():
    def boom(s):
        raise RuntimeError("tokenizer down")
    assert signature_suppression_ids(boom) == []   # never raises, empty on failure


def test_signature_ids_skips_empty_encodings():
    assert signature_suppression_ids(lambda s: []) == []
