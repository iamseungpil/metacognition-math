"""어드밴티지 자리 — 구간 크레딧이 **시퀀스 합으로 뭉개지지 않고** 그 토큰에만 얹히는가.

★왜 이 테스트가 있는가(0921 판정): verl 기본 GRPO 는 `scores = token_level_rewards.sum(-1)`
→ uid 중심화 → **스칼라를 응답 전 토큰에 브로드캐스트** 한다. 그래서 토큰별 크레딧을
`token_level_rewards` 에 넣으면 합만 남고 구간 국소화·DECISION_TOKENS ×2 가 전부 사라진다.
크레딧은 **중심화 뒤에** 어드밴티지 텐서의 그 자리에 직접 더해야 한다.
"""
from __future__ import annotations

import pytest
import torch

from mc import train_hook as H
from mc import trainer as T

B, T_LEN, PLEN = 4, 40, 3


class _Data:
    """verl DataProto 대역 — 훅이 만지는 키만 갖는다."""

    def __init__(self, uids):
        self.batch = {
            "prompts": torch.zeros(B, PLEN, dtype=torch.long),
            "responses": torch.arange(1, B * T_LEN + 1, dtype=torch.long).reshape(B, T_LEN),
            "attention_mask": torch.ones(B, PLEN + T_LEN, dtype=torch.long),
        }
        # 행마다 유효 응답 길이를 다르게 둔다(오른쪽 패딩) — 마스크 밖 가산을 잡기 위해.
        self.lens = [T_LEN, T_LEN, 30, 20]
        for i, n in enumerate(self.lens):
            self.batch["attention_mask"][i, PLEN + n:] = 0
        self.non_tensor_batch = {"uid": list(uids)}

    def __len__(self):
        return B


def _mock_original(data, adv_estimator=None, gamma=1.0, lam=1.0, num_repeat=1,
                   norm_adv_by_std_in_grpo=True, config=None):
    """verl GRPO 의 본질만 재현: 시퀀스 합 → uid 중심화 → 응답 전 토큰 브로드캐스트."""
    tlr = data.batch["token_level_rewards"]
    am = data.batch["attention_mask"]
    mask = am[:, PLEN:].float()
    data.batch["response_mask"] = mask
    scores = (tlr * mask).sum(-1)
    uids = list(data.non_tensor_batch["uid"])
    out = torch.zeros_like(tlr)
    for u in set(uids):
        idx = [i for i, x in enumerate(uids) if x == u]
        mean = scores[idx].mean()
        for i in idx:
            out[i] = (scores[i] - mean)
    data.batch["advantages"] = out * mask
    data.batch["returns"] = data.batch["advantages"]        # verl GRPO 규약: 같은 텐서(alias) — clone 아님
    _MOCK_SCORES.clear()
    _MOCK_SCORES.extend([float(x) for x in scores])
    return data


_MOCK_SCORES: list[float] = []


@pytest.fixture
def patched(monkeypatch):
    """`patch_compute_advantage` 를 모의 original 위에 설치하고 그 결과 함수를 돌려준다."""
    import verl.trainer.ppo.ray_trainer as rt
    monkeypatch.setattr(rt, "compute_advantage", _mock_original, raising=False)
    monkeypatch.setattr(rt, "_mc_patched", False, raising=False)
    T.patch_compute_advantage()
    fn = rt.compute_advantage
    assert fn is not _mock_original
    return fn


def _install_rewards(monkeypatch, *, r, credit, tel=None):
    """`token_rewards` 를 합성 결과로 바꾼다 — 결과 보상 r(마지막 토큰) + 구간 크레딧.
    `tel` 은 **같은 객체**를 돌려준다(호출자가 mass_share 를 되읽는다)."""
    tel = tel if tel is not None else {}

    def fake(data, tok, trainer, step):
        am = data.batch["attention_mask"]
        tlr = [[0.0] * T_LEN for _ in range(B)]
        for i in range(B):
            last = int(am[i, PLEN:].sum()) - 1
            tlr[i][last] = float(r[i])
        return tlr, credit, tel

    monkeypatch.setattr(H, "token_rewards", fake)
    T.CTX["tokenizer"], T.CTX["trainer"] = object(), None


def test_credit_lands_only_on_its_tokens(patched, monkeypatch):
    """결과가 같은 두 행은 구간 토큰에서만 a×w_t 만큼 다르고, 구간 밖은 **정확히 같다**."""
    j0, n = 5, 6
    credit = {1: (j0, [0.5] * n)}
    _install_rewards(monkeypatch, r=[1.0, 1.0, 1.0, 1.0], credit=credit)
    data = _mock_data_with(patched, monkeypatch, credit, r=[1.0, 1.0, 1.0, 1.0])
    a = data.batch["advantages"]
    diff = a[1] - a[0]
    assert torch.allclose(diff[j0:j0 + n], torch.full((n,), 0.5)), diff[j0:j0 + n]
    off = torch.cat([diff[:j0], diff[j0 + n:]])
    assert torch.allclose(off, torch.zeros_like(off)), off


def _mock_data_with(fn, monkeypatch, credit, r):
    _install_rewards(monkeypatch, r=r, credit=credit)
    data = _Data(["u", "u", "u", "u"])
    return fn(data, adv_estimator="grpo", config=None)


def test_credit_not_included_in_centering(patched, monkeypatch):
    """중심화에 들어가는 scores 는 **결과 보상만**이어야 한다(크레딧이 섞이면 형제 평균이
    크레딧으로 움직여 «구간이 있는 행이 이득»이 되는 가짜 신호가 생긴다)."""
    credit = {0: (2, [10.0] * 4)}
    _mock_data_with(patched, monkeypatch, credit, r=[1.0, 0.0, 0.0, 0.0])
    assert _MOCK_SCORES == [1.0, 0.0, 0.0, 0.0]


def test_decision_tokens_double_survives_to_advantages(patched, monkeypatch):
    """구간 안 상대 비중(마지막 DECISION_TOKENS ×2)이 어드밴티지에 **그대로** 남는다."""
    j0 = 4
    vals = [1.0, 1.0, 2.0, 2.0]
    data = _mock_data_with(patched, monkeypatch, {2: (j0, vals)}, r=[0.0] * 4)
    a = data.batch["advantages"][2]
    base = data.batch["advantages"][0]
    d = (a - base)[j0:j0 + 4]
    assert torch.allclose(d, torch.tensor(vals))
    assert d[-1] == pytest.approx(2 * d[0])


def test_credit_outside_response_mask_is_dropped(patched, monkeypatch):
    """행 3 의 유효 길이는 20 — 그 뒤 자리에 실린 크레딧은 버려야 한다(패딩에 보상 금지)."""
    data = _mock_data_with(patched, monkeypatch, {3: (18, [1.0] * 6)}, r=[0.0] * 4)
    a = data.batch["advantages"][3]
    assert torch.allclose(a[18:20], torch.tensor([1.0, 1.0]))
    assert torch.allclose(a[20:], torch.zeros(T_LEN - 20))


def test_returns_track_advantages(patched, monkeypatch):
    """GRPO 규약: returns 도 같은 자리에서 같이 움직인다."""
    data = _mock_data_with(patched, monkeypatch, {1: (3, [0.7] * 3)}, r=[0.0] * 4)
    assert torch.allclose(data.batch["returns"], data.batch["advantages"])
    # alias 인데 두 번 더해지면 크레딧이 2배가 된다(0921 step-1 덤프에서 실측된 결함).
    assert data.batch["returns"].data_ptr() == data.batch["advantages"].data_ptr()


def test_mass_share_matches_hand_calc(patched, monkeypatch):
    """mass_share = Σ|크레딧| / (Σ|크레딧| + Σ_{response_mask}|결과 어드밴티지|)."""
    tel: dict = {}
    credit = {0: (2, [1.0, 1.0])}          # Σ|크레딧| = 2.0
    _install_rewards(monkeypatch, r=[1.0, 0.0, 0.0, 0.0], credit=credit, tel=tel)
    data = _Data(["u"] * 4)
    patched(data, adv_estimator="grpo", config=None)
    # 결과 어드밴티지: scores [1,0,0,0], 평균 .25 → [.75,-.25,-.25,-.25], 각 행 유효 토큰 전체
    outcome = 0.75 * 40 + 0.25 * 40 + 0.25 * 30 + 0.25 * 20
    assert tel["mass_share"] == pytest.approx(2.0 / (2.0 + outcome))


def test_mass_share_is_loss_weighted_under_token_mean():
    """`loss_agg_mode=token-mean` 에서는 모든 유효 토큰이 1/batch_num_tokens 로 똑같이 손실에
    실린다(verl09 `agg_loss`) — 그래서 Σ|크레딧| / (Σ|크레딧|+Σ|결과 A|) 가 곧 손실 기준 몫이다.
    이 계약이 코드 주석에 남아 있는지 고정한다(모드가 바뀌면 경고를 찍는 분기도 함께)."""
    import inspect
    src = inspect.getsource(T.add_span_credit)
    assert "token-mean" in src and "batch_num_tokens" in src
    assert "seq-mean-token-mean" in src            # 갈리는 모드를 명시
    assert "advantages" in src and "critic" in src  # verl 이 소비하는 키 확인 기록


def test_mc_dump_adv_writes_before_and_after(patched, monkeypatch, tmp_path):
    """`MC_DUMP_ADV=N` — 사행 유인 실측(«틀리고 고친 행» 대 «한 번에 정답 행»의 토큰당 평균)과
    구간 밖 불변 확인을 위해 가산 **전/후** 어드밴티지를 그대로 떨군다."""
    import json

    import numpy as np
    monkeypatch.setenv("MC_DUMP_ADV", "1")
    monkeypatch.setenv("MC_CKPT_DIR", str(tmp_path))
    tel: dict = {"rows": {1: {"first_correct": 0.0, "last_correct": 1.0, "zone": [5, 11],
                             "a": 1.0}}}
    credit = {1: (5, [1.0] * 6)}
    _install_rewards(monkeypatch, r=[1.0, 1.0, 0.0, 0.0], credit=credit, tel=tel)
    patched(_Data(["u"] * 4), adv_estimator="grpo", config=None)
    npz, meta = tmp_path / "adv_dump_step0.npz", tmp_path / "adv_dump_step0.json"
    assert npz.is_file() and meta.is_file()
    z = np.load(npz)
    assert {"adv_before", "adv_after", "response_mask"} <= set(z.files)
    assert z["adv_before"].shape == z["adv_after"].shape == (B, T_LEN)
    d = z["adv_after"] - z["adv_before"]
    assert np.allclose(d[1, 5:11], 1.0)                 # 구간에만 가산
    assert np.allclose(np.delete(d, np.s_[5:11], axis=1), 0.0)   # 구간 밖 불변
    m = json.loads(meta.read_text())
    assert m["step"] == 0 and m["credit"]["1"][0] == 5
    assert m["rows"]["1"]["first_correct"] == 0.0 and m["rows"]["1"]["zone"] == [5, 11]
    assert "mass_share" in m


def test_mc_dump_adv_off_by_default(patched, monkeypatch, tmp_path):
    monkeypatch.delenv("MC_DUMP_ADV", raising=False)
    monkeypatch.setenv("MC_CKPT_DIR", str(tmp_path))
    _mock_data_with(patched, monkeypatch, {0: (2, [1.0])}, r=[0.0] * 4)
    assert not list(tmp_path.glob("adv_dump_*"))


def test_mc_dump_adv_carries_text_and_token_credit(patched, monkeypatch, tmp_path):
    """자리 보기용 확장 — 크레딧 받은 행의 본문·구간·라벨과 토큰별 크레딧 배열이 실린다."""
    import json

    import numpy as np
    monkeypatch.setenv("MC_DUMP_ADV", "1")
    monkeypatch.setenv("MC_CKPT_DIR", str(tmp_path))

    class _Tok:
        def decode(self, ids, skip_special_tokens=True):
            return "TOK" * 4000

    tel: dict = {"rows": {1: {"first_correct": 0.0, "last_correct": 1.0, "zone": [5, 11],
                             "a": 1.0, "turn_end": 7, "label": "gold",
                             "label_value": "42"}}}
    credit = {1: (5, [1.0] * 6)}
    _install_rewards(monkeypatch, r=[1.0, 1.0, 0.0, 0.0], credit=credit, tel=tel)
    monkeypatch.setitem(T.CTX, "tokenizer", _Tok())
    patched(_Data(["u"] * 4), adv_estimator="grpo", config=None)
    z = np.load(tmp_path / "adv_dump_step0.npz")
    m = json.loads((tmp_path / "adv_dump_step0.json").read_text())
    r = m["rows"]["1"]
    assert r["credited"] and r["turn_end"] == 7 and r["label"] == "gold"
    assert r["label_value"] == "42" and "pmi" not in r        # 옛 SHIFT 필드 없음(0924)
    assert len(r["text"]) == 6000                      # 6000자 상한
    assert 1 in m["detail_rows"] and len(m["detail_rows"]) <= 64 + 8
    assert "credit_1" in z.files and z["credit_1"].dtype == np.float16
    j0, j1 = r["zone"]
    assert len(z["credit_1"]) == j1 - j0               # 구간 토큰 수와 같다
    assert any(not m["rows"][str(i)]["credited"]       # 대조용 무크레딧 행
               for i in m["detail_rows"] if i != 1)
