"""DPO 손실(src/training/dpo_check.py) + 쌍 구성(scripts/local/build_check_pairs.py) 단위 시험. CPU 전용."""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.training.dpo_check import (  # noqa: E402
    dpo_loss, dpo_loss_from_tokens, dpo_train_step, masked_sum, token_logprobs,
)


def _load_builder():
    path = REPO_ROOT / "scripts" / "local" / "build_check_pairs.py"
    spec = importlib.util.spec_from_file_location("build_check_pairs", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── DPO 손실 ────────────────────────────────────────────────────────────────

def test_loss_near_zero_when_chosen_strongly_preferred():
    loss, st = dpo_loss(torch.tensor([0.0]), torch.tensor([-200.0]),
                        torch.tensor([0.0]), torch.tensor([0.0]), beta=0.1)
    assert float(loss) < 1e-6
    assert st["dpo/acc"] == 1.0 and st["dpo/margin"] > 0


def test_loss_large_when_rejected_preferred():
    loss, st = dpo_loss(torch.tensor([-200.0]), torch.tensor([0.0]),
                        torch.tensor([0.0]), torch.tensor([0.0]), beta=0.1)
    assert float(loss) > 10.0
    assert st["dpo/acc"] == 0.0


def test_loss_is_log2_at_tie():
    loss, _ = dpo_loss(torch.tensor([1.0]), torch.tensor([1.0]),
                       torch.tensor([0.5]), torch.tensor([0.5]), beta=0.3)
    assert float(loss) == pytest.approx(math.log(2), abs=1e-6)


def test_reference_cancels_out():
    """참조가 chosen/rejected 에 같은 만큼 더해지면 손실은 안 변한다(마진만이 신호)."""
    a, _ = dpo_loss(torch.tensor([2.0]), torch.tensor([1.0]),
                    torch.tensor([0.0]), torch.tensor([0.0]), beta=0.1)
    b, _ = dpo_loss(torch.tensor([5.0]), torch.tensor([4.0]),
                    torch.tensor([3.0]), torch.tensor([3.0]), beta=0.1)
    assert float(a) == pytest.approx(float(b), abs=1e-6)


def test_gradient_sign():
    """chosen logp 를 올리거나 rejected logp 를 내리면 손실이 준다."""
    c = torch.tensor([0.5], requires_grad=True)
    r = torch.tensor([0.4], requires_grad=True)
    loss, _ = dpo_loss(c, r, torch.zeros(1), torch.zeros(1), beta=0.2)
    loss.backward()
    assert float(c.grad) < 0    # 경사하강 → chosen logp 증가
    assert float(r.grad) > 0    # 경사하강 → rejected logp 감소


def test_beta_scales_margin():
    l_small, _ = dpo_loss(torch.tensor([1.0]), torch.tensor([0.0]),
                          torch.zeros(1), torch.zeros(1), beta=0.01)
    l_big, _ = dpo_loss(torch.tensor([1.0]), torch.tensor([0.0]),
                        torch.zeros(1), torch.zeros(1), beta=1.0)
    assert float(l_big) < float(l_small) < math.log(2) + 1e-6


def test_masked_sum_ignores_padding():
    lp = torch.tensor([[1.0, 2.0, 99.0]])
    assert float(masked_sum(lp, torch.tensor([[1, 1, 0]]))) == pytest.approx(3.0)


def test_token_logprobs_and_from_tokens():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 7)
    ids = torch.randint(0, 7, (2, 3))
    tlp = token_logprobs(logits, ids)
    assert tlp.shape == (2, 3) and bool((tlp <= 0).all())
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    loss, st = dpo_loss_from_tokens(tlp, mask, tlp, mask, torch.zeros_like(tlp), torch.zeros_like(tlp))
    # chosen 과 rejected 가 동일하면 마진 0 → log 2.
    assert float(loss) == pytest.approx(math.log(2), abs=1e-6)
    assert st["dpo/margin"] == pytest.approx(0.0, abs=1e-6)


def test_train_step_decreases_loss():
    torch.manual_seed(0)
    V, T = 11, 4
    model = torch.nn.Sequential(torch.nn.Embedding(V, 16), torch.nn.Linear(16, V))
    batch = {
        "chosen_ids": torch.randint(0, V, (2, T)),
        "chosen_mask": torch.ones(2, T),
        "rejected_ids": torch.randint(0, V, (2, T)),
        "rejected_mask": torch.ones(2, T),
        "chosen_ref_logp": torch.zeros(2),
        "rejected_ref_logp": torch.zeros(2),
    }
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    first, _ = dpo_train_step(model, batch, beta=0.5, optimizer=opt)
    for _ in range(10):
        last, _ = dpo_train_step(model, batch, beta=0.5, optimizer=opt)
    assert float(last) < float(first)


def test_bad_shapes_raise():
    with pytest.raises(ValueError):
        token_logprobs(torch.randn(2, 3), torch.zeros(2, 3, dtype=torch.long))
    with pytest.raises(ValueError):
        masked_sum(torch.zeros(2, 3), torch.zeros(2, 4))
    with pytest.raises(ValueError):
        dpo_loss(torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1), beta=0.0)


# ── 쌍 구성 로직 (합성 픽스처, GPU 데이터 불요) ─────────────────────────────

NUMS, TARGET = [2, 3, 4], 24  # 2*3*4 = 24

SOLVED = ("<check>2*3*5=30 ✗</check> that is wrong. 2+3=5, 5*4=20, 2*3=6, 6*4=24. "
          "\\boxed{2*3*4}")
OVERCLAIM = "I am confident. \\boxed{2+3+4}"
# 박스도 검산도 없이 끊긴 이어쓰기 — over_claim 이 아니라 중립(미검산·미해결).
NOBOX_UNSOLVED = "let me try 2+3 = 5 and then"


def _site(sid="s0"):
    return {"site_id": sid, "nums": NUMS, "target": TARGET, "prefix": "prefix text", "prompt": None}


def _cont(sid, k, text, r_corr):
    return {"site_id": sid, "k_index": k, "continuation": text, "full_text": text, "r_corr": r_corr}


def test_pairing_emits_solved_over_overclaim():
    mod = _load_builder()
    pairs, stats = mod.build_pairs(
        [_site()],
        [_cont("s0", 0, SOLVED, 1), _cont("s0", 1, OVERCLAIM, 0)],
    )
    assert stats["n_pairs"] == 1
    p = pairs[0]
    assert p["chosen_text"] == SOLVED and p["rejected_text"] == OVERCLAIM
    assert p["rejected_kind"] == "over_claim"
    assert p["nums"] == NUMS and p["target"] == TARGET


def test_no_pair_when_one_side_missing():
    """긍정만 있는 자리는 «무(無)보다 선호» 쌍을 조작하지 않는다."""
    mod = _load_builder()
    _, stats = mod.build_pairs([_site()], [_cont("s0", 0, SOLVED, 1)])
    assert stats["n_pairs"] == 0


def test_fallback_rejected_is_nocheck_wrong():
    mod = _load_builder()
    pairs, stats = mod.build_pairs(
        [_site()], [_cont("s0", 0, SOLVED, 1), _cont("s0", 1, NOBOX_UNSOLVED, 0)])
    assert stats["n_pairs"] == 1
    assert pairs[0]["rejected_kind"] == "nocheck_wrong"


def test_orphan_continuations_counted_not_paired():
    mod = _load_builder()
    pairs, stats = mod.build_pairs(
        [_site("s0")],
        [_cont("s0", 0, SOLVED, 1), _cont("s0", 1, OVERCLAIM, 0), _cont("ghost", 0, SOLVED, 1)])
    assert stats["n_orphan_conts"] == 1 and len(pairs) == 1


def test_pair_rate_over_multiple_sites():
    mod = _load_builder()
    sites = [_site("a"), _site("b")]
    conts = [_cont("a", 0, SOLVED, 1), _cont("a", 1, OVERCLAIM, 0), _cont("b", 0, SOLVED, 1)]
    _, stats = mod.build_pairs(sites, conts)
    assert stats["n_sites_with_conts"] == 2 and stats["n_pairs"] == 1
    assert stats["pair_rate"] == pytest.approx(0.5)
