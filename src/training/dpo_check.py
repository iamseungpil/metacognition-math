r"""DPO(쌍 선호) 손실 — 검산(`<check>`) 행동 전용, **순수 함수 + 단독 스텝**.

왜 새 기제인가. 지금까지 검산 사다리(§13/§14 — EVC·EVCM·EVCA·TAG0)는 전부 «스칼라
보상 항 또는 기존 크레딧의 마스크/배율»이었다. 보너스를 준 판(EVC)은 정확도를 깎았고,
마스크 판(EVCM)만 방향이 맞았다. **두 궤적을 같은 손실 안에서 직접 맞대는 목적함수**는
이 프로젝트에서 한 번도 실제로 돌아간 적이 없다 — 수학 단계의 R18b(`archive/docs_pre_rq3/
PLAN.md` "A.1 contrastive-on-natural-meta — FAIL")가 **자연 발생 그룹**에서 쌍을 찾다가
굶었을 뿐이다. 자연 그룹의 쌍 형성률은 재측정에서도 0.4~5.0%였다(`scripts/local/
chk_pair_probe.py`). 그래서 쌍은 **자리(site) 이어쓰기에서 구성**한다
(`scripts/local/build_check_pairs.py`).

참조 로그확률. 이 모듈은 **계산하지 않고 받는다**. 트레이너 쪽에서 기존 OPD/PMI 경로와
같은 방식(`verl_sdc._dcpo_v4_ref_logprobs` → `trainer._compute_ref_log_prob(batch).
batch["ref_log_prob"]`, `ref_log_prob[i, t] = logP_ref(responses[i, t] | ...)`)으로 뽑아
토큰별 텐서를 그대로 넘기면 된다 — 새 동결 모델 로딩 경로를 만들지 않는다.

⚠️ **다음 단계: verl 트레이너 루프에 배선.** 이 파일은 손실 수학과 단독 스텝만 담는다.
verl PPO/GRPO 스텝 함수에 배선된 것이 **아니며**, 따라서 팔(ARM)로 발사할 수 없다.

CPU 전용 · 순수 torch.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

DEFAULT_BETA = 0.1


def token_logprobs(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """`logits`[B, T, V] 에서 `ids`[B, T] 토큰의 로그확률 [B, T] 을 뽑는다.

    호출자가 이미 «다음 토큰» 정렬(logits[:, t] 가 ids[:, t] 를 예측)을 맞춰서
    넘겨야 한다 — verl 의 `ref_log_prob` 이 쓰는 정렬과 같은 규약이다.
    """
    if logits.dim() != 3 or ids.dim() != 2:
        raise ValueError(f"logits 는 [B,T,V], ids 는 [B,T] 여야 한다: {tuple(logits.shape)}, {tuple(ids.shape)}")
    logp = F.log_softmax(logits.float(), dim=-1)
    return logp.gather(-1, ids.unsqueeze(-1)).squeeze(-1)


def masked_sum(token_logp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """응답 토큰만 골라 시퀀스 로그확률 [B] 로 합친다."""
    if token_logp.shape != mask.shape:
        raise ValueError(f"token_logp/mask 모양 불일치: {tuple(token_logp.shape)} vs {tuple(mask.shape)}")
    return (token_logp * mask.to(token_logp.dtype)).sum(dim=-1)


def dpo_loss(
    chosen_logp: torch.Tensor,
    rejected_logp: torch.Tensor,
    chosen_ref_logp: torch.Tensor,
    rejected_ref_logp: torch.Tensor,
    beta: float = DEFAULT_BETA,
) -> tuple[torch.Tensor, dict]:
    """표준 DPO(Rafailov et al. 2023) 손실. 입력은 전부 **시퀀스 합** [B].

        loss = -logsigmoid(beta * ((logp_c - logp_c_ref) - (logp_r - logp_r_ref)))

    반환: (스칼라 평균 손실, 진단 dict — chosen/rejected 보상, 마진, 정확도).
    """
    if beta <= 0:
        raise ValueError(f"beta 는 양수여야 한다: {beta}")
    r_chosen = beta * (chosen_logp - chosen_ref_logp)
    r_rejected = beta * (rejected_logp - rejected_ref_logp)
    margin = r_chosen - r_rejected
    loss = -F.logsigmoid(margin).mean()
    stats = {
        "dpo/loss": float(loss.detach()),
        "dpo/reward_chosen": float(r_chosen.detach().mean()),
        "dpo/reward_rejected": float(r_rejected.detach().mean()),
        "dpo/margin": float(margin.detach().mean()),
        "dpo/acc": float((margin.detach() > 0).float().mean()),
    }
    return loss, stats


def dpo_loss_from_tokens(
    chosen_token_logp: torch.Tensor,
    chosen_mask: torch.Tensor,
    rejected_token_logp: torch.Tensor,
    rejected_mask: torch.Tensor,
    chosen_ref_token_logp: torch.Tensor,
    rejected_ref_token_logp: torch.Tensor,
    beta: float = DEFAULT_BETA,
) -> tuple[torch.Tensor, dict]:
    """토큰별 로그확률 [B, T] 판. 참조 쪽도 **같은 마스크**로 합친다."""
    return dpo_loss(
        masked_sum(chosen_token_logp, chosen_mask),
        masked_sum(rejected_token_logp, rejected_mask),
        masked_sum(chosen_ref_token_logp, chosen_mask),
        masked_sum(rejected_ref_token_logp, rejected_mask),
        beta=beta,
    )


def dpo_train_step(
    model,
    batch: dict,
    beta: float = DEFAULT_BETA,
    optimizer: Optional["torch.optim.Optimizer"] = None,
) -> tuple[torch.Tensor, dict]:
    """단독 학습 스텝 — 나중에 트레이너가 호출할 수 있게 최소 형태로만 둔다.

    `batch` 키: chosen_ids/chosen_mask/rejected_ids/rejected_mask [B, T],
    chosen_ref_logp/rejected_ref_logp — 토큰별 [B, T] 또는 시퀀스 합 [B] 둘 다 받는다.
    `model(ids)` 는 [B, T, V] logits 를 돌려주면 된다(HF 모델은 `.logits` 도 허용).

    ⚠️ **다음 단계: verl 트레이너 루프에 배선.** 아직 어떤 팔에도 연결돼 있지 않다.
    """
    def _logits(ids):
        out = model(ids)
        return getattr(out, "logits", out)

    c_logp = masked_sum(token_logprobs(_logits(batch["chosen_ids"]), batch["chosen_ids"]), batch["chosen_mask"])
    r_logp = masked_sum(token_logprobs(_logits(batch["rejected_ids"]), batch["rejected_ids"]), batch["rejected_mask"])

    def _ref(key, mask):
        ref = batch[key]
        return masked_sum(ref, mask) if ref.dim() == 2 else ref

    loss, stats = dpo_loss(
        c_logp, r_logp,
        _ref("chosen_ref_logp", batch["chosen_mask"]),
        _ref("rejected_ref_logp", batch["rejected_mask"]),
        beta=beta,
    )
    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss, stats
