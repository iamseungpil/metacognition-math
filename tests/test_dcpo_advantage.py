"""Unit tests for the TRIOBJ_DCPO_V2 region-advantage composition (spec §2.3 / §6.3).

PURE PYTHON + torch. Build [B,T] region masks + [B] head scalars; assert per-region
routing, tag-token=0, and no-NaN on empty-meta / empty-conf rows.

The composition core lives in dcpo_region (torch-only, no verl/omegaconf) so it is
importable under a minimal env; verl_sdc_utils._compute_dcpo_region_advantage is a
thin delegator to compose_dcpo_region_advantage.
"""
import numpy as np
import torch

from src.training.dcpo_region import (
    compose_dcpo_region_advantage,
    group_mean_subtract as _group_mean_subtract,
)


def _run(R_corr, R_meta, R_cal, ans, meta_c, conf, response_mask, index,
         w_corr=1.0, w_meta=0.5, w_cal=0.3):
    A, A2 = compose_dcpo_region_advantage(
        response_mask=torch.tensor(response_mask, dtype=torch.float32),
        index=index,
        R_corr=np.asarray(R_corr, dtype=np.float32),
        R_meta=np.asarray(R_meta, dtype=np.float32),
        R_cal=np.asarray(R_cal, dtype=np.float32),
        answer_mask=torch.tensor(ans, dtype=torch.float32),
        meta_content_mask=torch.tensor(meta_c, dtype=torch.float32),
        conf_mask=torch.tensor(conf, dtype=torch.float32),
        w_corr=w_corr, w_meta=w_meta, w_cal=w_cal,
    )
    assert torch.equal(A, A2)
    return A


def test_group_mean_subtract_all_equal_zero():
    v = torch.tensor([0.5, 0.5, 0.5])
    out = _group_mean_subtract(v, ["g", "g", "g"])
    assert torch.allclose(out, torch.zeros(3, 1))


def test_group_mean_subtract_flip_stands_out():
    v = torch.tensor([1.0, -1.0, -1.0, -1.0])
    out = _group_mean_subtract(v, ["g"] * 4).squeeze(1)
    assert out[0] > 0 and torch.all(out[1:] < 0)


def test_routing_corr_only_on_answer():
    # B=1, T=6. Region layout: [ans, ans, TAG, meta_c, conf, TAG]
    ans = [[1, 1, 0, 0, 0, 0]]
    meta_c = [[0, 0, 0, 1, 1, 0]]
    conf = [[0, 0, 0, 0, 1, 0]]
    rm = [[1, 1, 1, 1, 1, 1]]
    # group of 2 so centering is non-trivial; second rollout zeros everything.
    R_corr = [1.0, -1.0]
    R_meta = [1.0, -1.0]
    R_cal = [-0.1, 0.1]
    ans2 = ans + [[0, 0, 0, 0, 0, 0]]
    meta2 = meta_c + [[0, 0, 0, 0, 0, 0]]
    conf2 = conf + [[0, 0, 0, 0, 0, 0]]
    rm2 = rm + [[1, 1, 1, 1, 1, 1]]
    A = _run(R_corr, R_meta, R_cal, ans2, meta2, conf2, rm2, ["g", "g"])
    row = A[0]
    # Â_corr = 1 - mean(1,-1) = 1; Â_meta = 1; Â_cal = -0.1 - mean(-0.1,0.1) = -0.1
    # answer tokens (idx 0,1) carry only w_corr*Â_corr = 1.0
    assert torch.allclose(row[[0, 1]], torch.tensor([1.0, 1.0]))
    # TAG tokens (idx 2,5) carry 0 (in neither answer nor content nor conf)
    assert row[2] == 0.0 and row[5] == 0.0
    # meta_content idx 3 carries only w_meta*Â_meta = 0.5*1 = 0.5
    assert torch.allclose(row[3], torch.tensor(0.5))
    # conf idx 4 is in BOTH meta_content and conf → w_meta*Â_meta + w_cal*Â_cal
    expected = 0.5 * 1.0 + 0.3 * (-0.1)
    assert torch.allclose(row[4], torch.tensor(expected))


def test_meta_only_on_content_not_answer():
    ans = [[1, 1, 0, 0]]
    meta_c = [[0, 0, 1, 0]]
    conf = [[0, 0, 0, 0]]
    rm = [[1, 1, 1, 1]]
    A = _run([1.0, -1.0], [2.0, 0.0], [0.0, 0.0],
             ans + [[0, 0, 0, 0]], meta_c + [[0, 0, 0, 0]], conf + [[0, 0, 0, 0]],
             rm + [[1, 1, 1, 1]], ["g", "g"])
    row = A[0]
    # Â_meta = 2 - 1 = 1; meta-content idx 2 gets 0.5*1; answer idx 0,1 get NO meta
    assert torch.allclose(row[2], torch.tensor(0.5))
    # answer carries only corr, never meta
    Ac = 1.0 - 0.0  # mean(1,-1)=0 → Â_corr=1
    assert torch.allclose(row[[0, 1]], torch.tensor([Ac, Ac]))


def test_cal_only_on_conf():
    ans = [[1, 0, 0]]
    meta_c = [[0, 1, 1]]
    conf = [[0, 0, 1]]
    rm = [[1, 1, 1]]
    A = _run([0.0, 0.0], [0.0, 0.0], [-0.5, 0.5],
             ans + [[0, 0, 0]], meta_c + [[0, 0, 0]], conf + [[0, 0, 0]],
             rm + [[1, 1, 1]], ["g", "g"])
    row = A[0]
    # Â_cal = -0.5 - 0 = -0.5; conf idx 2 = w_meta*0 + w_cal*(-0.5) = -0.15
    assert torch.allclose(row[2], torch.tensor(0.3 * -0.5))
    # idx 1 is meta_content but Â_meta=0 → 0
    assert row[1] == 0.0
    assert row[0] == 0.0


def test_no_nan_empty_meta_empty_conf():
    # rollout with NO meta and NO conf (all answer), single group.
    ans = [[1, 1, 1], [1, 1, 0]]
    meta_c = [[0, 0, 0], [0, 0, 1]]
    conf = [[0, 0, 0], [0, 0, 0]]   # empty conf everywhere
    rm = [[1, 1, 1], [1, 1, 1]]
    A = _run([1.0, -1.0], [0.0, 0.5], [0.0, 0.0], ans, meta_c, conf, rm, ["g", "g"])
    assert torch.isfinite(A).all()
    # empty-meta row 0: no content/conf tokens → only answer carries advantage
    assert A[0, 2] != 0.0 or A[0, 0] != 0.0


def test_tag_tokens_zero_partition_coverage():
    # response = ans ∪ meta_content ∪ conf ∪ TAGs; assert TAGs are 0 and coverage.
    ans = [[1, 0, 0, 1]]
    meta_c = [[0, 0, 1, 0]]   # idx 1 is a TAG (in neither ans nor content)
    conf = [[0, 0, 0, 0]]
    rm = [[1, 1, 1, 1]]
    A = _run([1.0, -1.0], [1.0, -1.0], [0.0, 0.0],
             ans + [[0, 0, 0, 0]], meta_c + [[0, 0, 0, 0]], conf + [[0, 0, 0, 0]],
             rm + [[1, 1, 1, 1]], ["g", "g"])
    row = A[0]
    # idx 1 is a tag (ans=0, content=0) → advantage 0
    assert row[1] == 0.0
