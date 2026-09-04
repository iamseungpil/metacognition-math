r"""자(ruler)들이 공유하는 HF 모델 래퍼 — 지연 로드, 배치 teacher-forced logprob,
문자 오프셋 → 은닉상태.

왜 새로 짜는가 (기존 코드를 재사용하지 않는 이유).
`scripts/pair_rulers.py:gather_logp`(:186-195)와 `src/training/verl_sdc.py`의
`_build_pmi_score_batches`/`_build_osd_arms`는 각각 "배치 1, forward 한 번"과
"verl 배치 조립+ray dispatch"라는 서로 다른 실행 모델을 전제한다. 이 패키지는
두 자리 모두에서 쓰이는 자를 하나의 인터페이스로 통일해야 하므로, 여기서는
①GPU 유무를 몰라도 되고(CPU/단일 GPU 로컬 실행도 지원) ②주입 가능한 소형
스텁으로 CPU 단위테스트가 가능해야 한다. `pair_rulers.gather_logp`의 "spans의
[s,e) 토큰을 예측하는 위치의 logp"라는 정의는 그대로 가져온다(수학은 같다) —
아래 `token_logprobs`의 주석에 그 출처를 인용한다.

★지연 로드. `HfCtx(model_path=...)`는 생성 시점에 아무것도 로드하지 않는다.
  실제 forward가 필요한 첫 호출에서만 `transformers.AutoModelForCausalLM`/
  `AutoTokenizer`를 불러온다 — CPU 전용 환경에서 이 모듈을 import 하는 것만으로
  무거운 체크포인트를 당기지 않기 위함이다(학습이 GPU를 쓰고 있는 이 저장소의
  관례, `countdown_pmi.py`가 verl/ray를 지연 import 하는 것과 같은 규약).

★테스트 주입구. `HfCtx(_model=..., _tokenizer=...)`로 진짜 HF 객체 대신 최소
  인터페이스만 흉내 낸 스텁(테스트 파일 안의 `FakeTokenizer`/`FakeModel`)을
  넣을 수 있다 — CPU 에서 이 파일의 배치·오프셋 로직을 실측한다(모델 자체의
  품질은 검증 대상이 아니다).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

__all__ = ["HfCtx", "TokenSpan"]


@dataclass(frozen=True)
class TokenSpan:
    """토큰 인덱스 반열린 구간 [start, end)."""
    start: int
    end: int

    def __post_init__(self):
        if self.end < self.start:
            raise ValueError(f"TokenSpan: end<start ({self.start},{self.end})")


class HfCtx:
    r"""배치 teacher-forced logprob + 은닉상태 추출. 지연 로드.

    사용:
        ctx = HfCtx(model_path="models/Qwen3-4B", device="cpu")
        ids = ctx.encode(text)                           # -> list[int]
        lp  = ctx.token_logprobs(ctx_ids, target_ids)     # -> np.ndarray, len(target_ids)
        h   = ctx.hidden_at(ids, char_offset=..., text=text)  # -> np.ndarray

    `needs_model=True` 인 자(ruler)는 이 클래스의 인스턴스(또는 같은 인터페이스를
    구현한 무엇)를 `ctx`로 받는다.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu",
                 dtype: Optional[str] = None, *, _model: Any = None,
                 _tokenizer: Any = None):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self._model = _model
        self._tokenizer = _tokenizer
        self._loaded = _model is not None and _tokenizer is not None

    # ── 지연 로드 ──────────────────────────────────────────────────────────
    def _ensure_loaded(self):
        if self._loaded:
            return
        if self.model_path is None:
            raise RuntimeError(
                "HfCtx: model_path 도 _model/_tokenizer 주입도 없다 — 이 자는 "
                "forward 가 필요한데(needs_model=True) 모델이 없다. "
                "ruler_table.py --no-model 경로에서는 이 자를 부르면 안 된다.")
        import torch                                  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if not getattr(self._tokenizer, "is_fast", False):
            raise TypeError(
                f"HfCtx: {self.model_path} 의 토크나이저가 fast 가 아니다 — "
                "offset_mapping 이 필요한 자(pmi_shift/osd/inv/move_kl)가 못 돈다.")
        kw = {}
        if self.dtype:
            kw["torch_dtype"] = getattr(torch, self.dtype, self.dtype)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_path, **kw)
        self._model.to(self.device)
        self._model.eval()
        self._loaded = True

    @property
    def tokenizer(self):
        self._ensure_loaded()
        return self._tokenizer

    @property
    def model(self):
        self._ensure_loaded()
        return self._model

    # ── 인코딩 ────────────────────────────────────────────────────────────
    def encode(self, text: str) -> list[int]:
        self._ensure_loaded()
        return list(self._tokenizer(text or "", add_special_tokens=False)["input_ids"])

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """fast 토크나이저의 offset_mapping 포함 인코딩. `countdown_pmi.find_meta_token_span`
        과 같은 방식(`response_text`에 대해서만 호출하고, 프롬프트를 붙이지 않는다)."""
        self._ensure_loaded()
        enc = self._tokenizer(text or "", add_special_tokens=False,
                              return_offsets_mapping=True)
        ids = list(enc["input_ids"])
        offsets = [tuple(o) for o in enc["offset_mapping"]]
        return ids, offsets

    # ── teacher-forced logprob ──────────────────────────────────────────
    def token_logprobs(self, ctx_ids: Sequence[int], target_ids: Sequence[int]) -> "Any":
        r"""단일 시퀀스 forward. ctx_ids 뒤에 target_ids 를 이어붙여 target 구간의
        토큰별 log P(target_t | ctx + target[:t]) 를 돌려준다.

        `scripts/pair_rulers.py:gather_logp`(:186-195)와 같은 정의 — "logits[s-1:e-1]을
        log_softmax 한 뒤 실제 다음 토큰을 gather"하는 그 수식을 그대로 옮겼다. 그
        함수는 한 forward에서 여러 span을 뽑는 형태였는데, 여기서는 한 번에 한 target
        구간만 필요한 자가 대부분이라 인터페이스를 단순화했다(`batched_logprobs`가
        다중 시퀀스 배치 버전).
        """
        import torch                                  # noqa: PLC0415
        self._ensure_loaded()
        ctx_ids, target_ids = list(ctx_ids), list(target_ids)
        if not target_ids:
            return _np_zeros(0)
        full = ctx_ids + target_ids
        with torch.no_grad():
            x = torch.tensor([full], device=self.model.device)
            logits = self.model(input_ids=x).logits[0]
        s = len(ctx_ids)
        part = torch.log_softmax(logits[s - 1:s - 1 + len(target_ids)].float(), dim=-1)
        tgt = torch.tensor(target_ids, device=logits.device)
        lp = part.gather(1, tgt[:, None])[:, 0]
        return lp.double().cpu().numpy()

    def batched_logprobs(self, contexts: Sequence[Sequence[int]],
                         targets: Sequence[Sequence[int]], batch: int = 8) -> list:
        r"""여러 (ctx, target) 쌍을 좌측패딩 배치로 묶어 `token_logprobs`와 같은 값을
        낸다. 순수 편의 함수 — 정확도가 중요한 소량 호출은 `token_logprobs`를 반복
        호출해도 된다. 여기서는 pad_token_id 가 없으면 eos 를 쓴다(HF 통상 관례)."""
        import torch                                  # noqa: PLC0415
        self._ensure_loaded()
        if not (len(contexts) == len(targets)):
            raise ValueError("batched_logprobs: contexts/targets 길이 불일치")
        pad_id = self._tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self._tokenizer.eos_token_id
        out: list = []
        for lo in range(0, len(contexts), batch):
            cs = [list(c) for c in contexts[lo:lo + batch]]
            ts = [list(t) for t in targets[lo:lo + batch]]
            seqs = [c + t for c, t in zip(cs, ts)]
            L = max(len(s) for s in seqs) if seqs else 0
            ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
            att = torch.zeros((len(seqs), L), dtype=torch.long)
            for j, s in enumerate(seqs):
                ids[j, :len(s)] = torch.tensor(s)
                att[j, :len(s)] = 1
            with torch.no_grad():
                logits = self.model(input_ids=ids.to(self.model.device),
                                    attention_mask=att.to(self.model.device)).logits
            for j, (c, t) in enumerate(zip(cs, ts)):
                if not t:
                    out.append(_np_zeros(0))
                    continue
                s = len(c)
                part = torch.log_softmax(logits[j, s - 1:s - 1 + len(t)].float(), dim=-1)
                tgt = torch.tensor(t, device=logits.device)
                lp = part.gather(1, tgt[:, None])[:, 0]
                out.append(lp.double().cpu().numpy())
        return out

    def sequence_logprob(self, ids: Sequence[int], upto: Optional[int] = None) -> float:
        """마지막 토큰(또는 인덱스 `upto`)의 로그확률 분포 전체(logsoftmax) — EOS 확률
        등을 읽을 때 쓴다(`dcont.py`가 `move_space_probe.p_continue`를 이 위에 재구현)."""
        import torch                                  # noqa: PLC0415
        self._ensure_loaded()
        ids = list(ids)
        with torch.no_grad():
            x = torch.tensor([ids], device=self.model.device)
            logits = self.model(input_ids=x).logits[0]
        idx = -1 if upto is None else upto
        return torch.log_softmax(logits[idx].float(), dim=-1)

    # ── 은닉상태 ──────────────────────────────────────────────────────────
    def hidden_at(self, ids: Sequence[int], token_index: int, *, layer: int = -1,
                  mean_last_n: Optional[int] = None):
        r"""`ids`를 forward 한 뒤 `token_index` 위치의 은닉상태(마지막 레이어, 또는
        `mean_last_n`이 주어지면 마지막 n개 레이어 평균)를 numpy 벡터로 돌려준다.

        `hidden_probe.py`가 meta_end 문자 오프셋을 토큰 인덱스로 옮긴 뒤 이걸 부른다.
        """
        import torch                                  # noqa: PLC0415
        self._ensure_loaded()
        ids = list(ids)
        with torch.no_grad():
            x = torch.tensor([ids], device=self.model.device)
            out = self.model(input_ids=x, output_hidden_states=True)
        hs = out.hidden_states                          # tuple(len=n_layer+1) of [1,T,H]
        if mean_last_n is not None:
            stacked = torch.stack([h[0, token_index] for h in hs[-mean_last_n:]], dim=0)
            vec = stacked.mean(dim=0)
        else:
            vec = hs[layer][0, token_index]
        return vec.float().cpu().numpy()

    def char_to_token_index(self, text: str, char_offset: int) -> int:
        """fast 토크나이저 offset_mapping으로 문자 오프셋 → 토큰 인덱스(그 문자를 담은,
        또는 그 문자 바로 앞에서 끝나는 토큰). `countdown_pmi.find_meta_token_span`의
        `_count_end_le`와 같은 규약(경계는 보수적으로 안쪽으로)."""
        _, offsets = self.encode_with_offsets(text)
        idx = 0
        for k, (a, b) in enumerate(offsets):
            if b <= a:
                continue
            if a >= char_offset:
                return k
            idx = k
        return idx


def _np_zeros(n: int):
    import numpy as np
    return np.zeros(n, dtype=float)
