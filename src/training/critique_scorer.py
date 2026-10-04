r"""CRITIQUE_SCORER — M_CRIT 의 **얼어붙은 채점기**(frozen scorer).

무엇을 재는가. 주어진 문맥에서 목표 텍스트의 **토큰당 평균 로그확률**
    meanlogp(target | prompt) = (1/|target|) · Σ_t log p(target_t | prompt, target_<t>)
을 teacher-forcing 으로 잰다. `math_meta.annotate_crit_ig` 가 이 값 셋(plain / note / donor)
으로 비평의 정보 이득(IG)과 그 내용 대조(IG − IG_donor)를 만든다.

★왜 «얼어붙은» 채점기인가. 학습 중인 정책(actor)으로 재면 최적해가 «좋은 비평을 쓰기»가 아니라
  «채점기가 좋아하도록 정책을 옮기기»가 된다 — 보상이 자기 자신을 참조하면 IG 는 스텝마다 뜻이
  바뀌는 자다. 그래서 **초기 정책**(actor init = `MODEL_PATH`, env `MATH_CRIT_SCORER_PATH` 로
  덮어쓸 수 있다)을 한 번만 로드해 학습 내내 고정한다. gradient 는 흐르지 않는다(no_grad).

★메모리. 4B bf16 ≈ **8 GB**(가중치) + forward 활성값·로짓 ~2~4 GB(batch 2 × 3k 토큰의
  [B,T,vocab] bf16 로짓이 지배적이다; fp32 log_softmax 는 512 위치씩 끊어 ~0.3 GB 로 묶는다)
  → 합 ~10~12 GB. vLLM 롤아웃 엔진과 **같은 CUDA 장치**에 올라가므로 `VLLM_UTIL` 을
  **0.3 이하**로 둬야 한다(run_math_arm.sh 가 M_CRIT 에 한해 기본을 0.35 → 0.3 으로 낮춘다).
  다른 것을 해제하거나 옮기지 않는다 — 이 모듈은 자기 모델만 들고 있다.
★시간. 한 스텝(train_batch_size 64 프롬프트 × rollout.n 8 = 512 행)에서 요청 수는
  «그룹 수 + 2 × 정의된 행» ≈ 64 + 2×250 ≈ 560 forward, 시퀀스당 ~1.5~3k 토큰 → 프리필
  ~1.2 M 토큰 ≈ 1.0e16 FLOP → H100 bf16 실효 200 TFLOP/s 기준 **약 1~2 분/스텝**.
  RL 스텝 자체가 보통 3~5 분이므로 20~40% 의 추가 비용이다 — 줄이려면 MATH_CRIT_MAX_TOK
  (표적 상한)을 낮추거나 정의된 행을 표집하라(현재는 전수).

사용:
    sc = get_scorer(default_path="/path/to/init")          # 지연 로드(첫 호출 때 GPU 로)
    sc.score_meanlogp([prompt1, prompt2], [target1, target2])   # -> [float, float]

테스트(모의):
    sc = CritiqueScorer(mock_fn=lambda p, t: -1.0 + 0.1 * ("critique" in p))
"""
from __future__ import annotations

import os
from collections.abc import Callable, Sequence

_NAN = float("nan")
# ★fp32 log_softmax 를 끊어 거는 위치 단위(메모리 상한). 값이 크면 빠르고 메모리를 더 쓴다.
_CHUNK = 512


def scorer_path(default_path: str | None = None) -> str:
    """MATH_CRIT_SCORER_PATH 가 있으면 그것, 없으면 actor init 경로(`default_path`).
    ★조용한 기본값 금지 — 둘 다 없으면 즉사한다(무엇으로 채점했는지 사후 확정 불가)."""
    p = os.environ.get("MATH_CRIT_SCORER_PATH") or (default_path or "")
    if not p:
        raise RuntimeError(
            "[CRIT-SCORER] 채점기 경로가 없다 — MATH_CRIT_SCORER_PATH 를 주거나 "
            "actor init 경로(actor_rollout_ref.model.path)를 default_path 로 넘겨라.")
    return str(p)


def max_target_tokens() -> int:
    """MATH_CRIT_MAX_TOK (기본 2048) — 표적 S+ 의 토큰 상한.
    ★평균 로그확률은 길이에 민감하고 긴 표적은 forward 비용을 그대로 늘린다. 상한을 넘으면
    **앞에서부터** 자른다(뒤를 자르면 \\boxed 가 사라져 «푸는 과정»이 아니라 서두만 채점된다)."""
    v = os.environ.get("MATH_CRIT_MAX_TOK")
    return 2048 if v is None else int(v)


class CritiqueScorer:
    """teacher-forcing meanlogp 채점기. bf16 · no_grad · 오른쪽 패딩 배치.

    `mock_fn` 을 주면 모델을 로드하지 않는다(CPU 테스트용) — `score_meanlogp` 가
    `mock_fn(prompt, target)` 를 그대로 돌려준다.
    """

    def __init__(self, model_path: str | None = None, *, device: str = "cuda",
                 batch_size: int = 2, max_len: int = 8192,
                 max_target_tok: int | None = None,
                 mock_fn: Callable[[str, str], float] | None = None,
                 tokenizer=None):
        self.model_path = model_path
        self.device = device
        self.batch_size = int(batch_size)
        self.max_len = int(max_len)
        self.max_target_tok = int(max_target_tok if max_target_tok is not None
                                  else max_target_tokens())
        self.mock_fn = mock_fn
        self._tok = tokenizer
        self._model = None

    # ── 지연 로드 ────────────────────────────────────────────────────────────
    def _load(self):
        """첫 채점 때 한 번만 로드한다. ★프리패스가 도는 트레이너 프로세스에서 부르므로
        vLLM 엔진이 이미 메모리를 잡은 뒤다 — bf16 8 GB 가 남아 있어야 한다(VLLM_UTIL ≤ 0.3)."""
        if self._model is not None:
            return
        if self.mock_fn is not None:
            return
        import torch  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
        path = scorer_path(self.model_path)
        if self._tok is None:
            self._tok = AutoTokenizer.from_pretrained(path)
        try:
            model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
        except TypeError:                 # transformers 4.x 는 torch_dtype
            model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
        self._model = model.to(self.device).eval()
        for p in self._model.parameters():
            p.requires_grad_(False)       # ★얼어붙음 — 이 모델로는 절대 학습하지 않는다
        print(f"[CRIT-SCORER] loaded frozen scorer from {path} (bf16, device={self.device}, "
              f"max_target_tok={self.max_target_tok})", flush=True)

    @property
    def tokenizer(self):
        """프롬프트 조립(math_meta.crit_prompt_pair)이 쓰는 토크나이저."""
        if self._tok is None:
            if self.mock_fn is not None:
                raise RuntimeError("[CRIT-SCORER] mock 모드인데 tokenizer 가 주입되지 않았다")
            self._load()
        return self._tok

    # ── 채점 ────────────────────────────────────────────────────────────────
    def score_meanlogp(self, prompts: Sequence[str], targets: Sequence[str]) -> list[float]:
        """[meanlogp(target_i | prompt_i)]. 길이가 다르면 즉사(짝이 어긋나면 IG 가 뜻을 잃는다)."""
        if len(prompts) != len(targets):
            raise RuntimeError(f"[CRIT-SCORER] prompts {len(prompts)} != targets {len(targets)}")
        if self.mock_fn is not None:
            return [float(self.mock_fn(p, t)) for p, t in zip(prompts, targets)]
        self._load()
        import torch  # noqa: PLC0415
        tok = self._tok
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
        enc = []
        for p, t in zip(prompts, targets):
            pi = list(tok.encode(p, add_special_tokens=False))
            ti = list(tok.encode(t, add_special_tokens=False))[:self.max_target_tok]
            # ★넘치면 **프롬프트의 왼쪽**을 자른다. 표적을 자르면 조건마다 분모(표적 토큰 수)가
            #   달라져 IG 가 «비평 효과»가 아니라 «잘림 차이»를 잰다.
            room = self.max_len - len(ti)
            if room <= 0:
                enc.append(None)
                continue
            pi = pi[-room:]
            enc.append((pi + ti, len(pi), len(ti)))
        out: list[float] = [_NAN] * len(enc)
        idx = [i for i, e in enumerate(enc) if e is not None and e[2] > 0]
        idx.sort(key=lambda i: len(enc[i][0]))      # 길이순 정렬 — 패딩 낭비를 줄인다
        with torch.no_grad():
            for b0 in range(0, len(idx), self.batch_size):
                sel = idx[b0:b0 + self.batch_size]
                seqs = [enc[i][0] for i in sel]
                T = max(len(x) for x in seqs)
                inp = torch.full((len(seqs), T), pad_id, dtype=torch.long)
                att = torch.zeros((len(seqs), T), dtype=torch.long)
                for r, x in enumerate(seqs):
                    inp[r, :len(x)] = torch.tensor(x, dtype=torch.long)
                    att[r, :len(x)] = 1    # ★오른쪽 패딩 — 위치 인덱스가 왼쪽 기준으로 유지된다
                res = self._model(input_ids=inp.to(self.device), attention_mask=att.to(self.device))
                for r, i in enumerate(sel):
                    _ids, n_head, n_tgt = enc[i]
                    # 위치 t 의 로짓이 토큰 t+1 을 예측한다 → 표적 토큰 p 의 로짓은 위치 p-1.
                    # ★fp32 log_softmax 를 표적 전체에 한 번에 걸면 [n_tgt, vocab] (2048×152k×4B
                    #   ≈ 1.2 GB)가 순간적으로 잡힌다 — 512 위치씩 끊어 상한을 ~300 MB 로 둔다.
                    total = 0.0
                    for c0 in range(0, n_tgt, _CHUNK):
                        c1 = min(c0 + _CHUNK, n_tgt)
                        pos = list(range(n_head - 1 + c0, n_head - 1 + c1))
                        lg = torch.log_softmax(res.logits[r, pos].float(), dim=-1)
                        tgt = torch.tensor(_ids[n_head + c0:n_head + c1], device=lg.device)
                        total += float(lg.gather(1, tgt[:, None]).sum())
                        del lg
                    out[i] = total / n_tgt
                del res
        return out


# ── 프로세스당 하나(얼어붙은 채점기는 한 번만 로드한다) ──────────────────────────
_SINGLETON: dict = {"scorer": None, "path": None}


def get_scorer(default_path: str | None = None, **kw) -> CritiqueScorer:
    """프로세스 안에서 **같은 경로면 같은 인스턴스**를 돌려준다(모델 두 벌 = OOM).
    경로가 바뀌면 즉사 — 한 런 안에서 채점기가 갈리면 IG 가 스텝 사이에 비교 불가능해진다."""
    path = scorer_path(default_path)
    if _SINGLETON["scorer"] is None:
        _SINGLETON.update({"scorer": CritiqueScorer(path, **kw), "path": path})
    elif _SINGLETON["path"] != path:
        raise RuntimeError(
            f"[CRIT-SCORER] 채점기 경로가 런 중에 바뀌었다: {_SINGLETON['path']!r} -> {path!r} — "
            "IG 는 고정된 자여야 비교 가능하다.")
    return _SINGLETON["scorer"]


def set_scorer(scorer: CritiqueScorer | None, path: str = "<injected>") -> None:
    """테스트·주입용. None 이면 초기화한다."""
    _SINGLETON.update({"scorer": scorer, "path": None if scorer is None else path})
