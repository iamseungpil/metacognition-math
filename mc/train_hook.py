#!/usr/bin/env python
r"""보상 자리 — mc 의 **유일한** 보상 계산(mc/credit.py 가 순수 함수를 준다). 남은 팔은 `SPONT_PFX` 하나뿐이다:
고정 첫 박스 앞부분 뒤 K 이어 쓰기 · r = 이어 쓰기의 최종 답 ≡ LOO 라벨(또는 gold) · verl GRPO. 라벨
`LABEL=gold|majority`."""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mc.credit import dir_weights, fork_weights
from mc.grade import boxed_answer, boxed_spans, grade_answer
from mc.shift_check import _enc, leak_point

# ── 팔 명세 ──────────────────────────────────────────────────────────────────
MC_ARM_SPECS: dict[str, dict] = {
    "SPONT_PFX": {"term": "pfx", "pfx": True,
                  "note": "고정 첫 박스 앞부분(프롬프트) 뒤 K 이어 쓰기 · r=최종 답≡LOO 라벨 · verl GRPO."},
}


def _f(env: str, default: float) -> float:
    v = os.environ.get(env)
    return default if v is None or not str(v).strip() else float(v)


def arm_spec(arm: str) -> dict:
    a = str(arm or "").upper()
    if a not in MC_ARM_SPECS:
        raise ValueError(f"[MC] arm={arm!r} 이 {sorted(MC_ARM_SPECS)} 에 없다.")
    return MC_ARM_SPECS[a]


def label_mode() -> str:
    m = (os.environ.get("LABEL") or "gold").strip().lower()
    if m not in ("gold", "majority"):
        raise ValueError(f"[MC] LABEL={m!r} 은 gold|majority 중 하나여야 한다.")
    return m


def outcome_mode() -> str:
    """`OUTCOME_MODE`(기본 group) — SPONT_PFX 는 group 전용(pfx_rewards 가 즉사시킨다)."""
    m = (os.environ.get("OUTCOME_MODE") or "group").strip().lower()
    if m != "group":
        raise ValueError(f"[MC] OUTCOME_MODE={m!r} — SPONT_PFX 는 group 전용")
    stale = sorted(k for k in os.environ if k in ("W_PMI2", "PFX_CHECK_COST", "PFX_CHECK_W", "PFX_RIGHT_SHARE", "PFX_STOP",
                                                          "PFX_STOP_W", "PFX_TAIL0") or k.startswith("PMI2_"))
    if stale:                                  # 지운 손잡이 — 옛 큐 명령이 조용히 다른 팔로 도는 것을 막는다
        raise ValueError(f"[MC] 지운 손잡이 {stale} — PMI2/U/Dir/R4(0925)·R2 검산 비용(수정 33)·맞은 첫 답 표집 몫(28f ii, 수정 38)·멈춤 STOP·TAIL0(수정 43) — 실패로 삭제됨")
    return m


def describe() -> str:
    from mc.context import default_variant  # noqa: PLC0415
    return f"label={label_mode()} outcome_mode={outcome_mode()} prompt_variant={default_variant()}"


# ── 중단 규칙 — PFX 는 파괴 가드 하나뿐이다 ─────────────────────────────────────
PFX_BREAK_LIMIT, PFX_BREAK_SPAN = 0.05, 3   #: PFX 유일한 가드(`pfx_guard`) — 하한 .05 · 기준선·최근 창 3스텝

HISTORY: list[dict] = []


def first_ref_path(name: str = "FIRST_REF.txt") -> Path | None:
    d = os.environ.get("MC_CKPT_DIR")
    return (Path(d) / name) if d else None


def pfx_guard(steps) -> str | None:
    """PFX 파괴 가드 — break|right 최근 3스텝 평균 > max(.05, 2 × 처음 3스텝 평균)(라벨·gold 각각). 학습 풀은 답이
    잘 바뀌는 문제라 base 부터 gold 파괴가 높다(스모크 step1 .125 대 탐침 .004) → 절대선 대신 자기 기준선의 2배.
    기준선은 `FIRST_REF_PFX.json` 에 남겨 끊어 학습(resume)을 견딘다. 수정 36: `*_live`(PFX_TRUNC=mask 에서 학습에 들어가는
    잘리지 않은 행만)가 있으면 그것으로 본다 — 잘린 행의 도중 박스 «망침» 은 길이 문제라 가드를 헛울렸다(0929 H4c), 기준선은 옛 키를 잇는다."""
    p = first_ref_path("FIRST_REF_PFX.json")
    ref = json.loads(p.read_text()) if p is not None and p.is_file() else {}
    live = "_live" if any("pfx_break_right_live" in s for s in steps) else ""
    for key in (f"pfx_break_right{live}", f"gold_pfx_break_right{live}"):   # 라벨 판은 보상이 스스로 낮춘다 → gold 로도 본다
        if key not in ref and live and key[:-5] in ref:
            ref[key] = ref[key[:-5]]
        xs = [v for v in (float(s.get(key, "nan")) for s in steps) if v == v]
        if key not in ref and len(xs) >= PFX_BREAK_SPAN:
            ref[key] = sum(xs[:PFX_BREAK_SPAN]) / PFX_BREAK_SPAN
            if p is not None:
                p.write_text(json.dumps(ref))
        m = sum(xs[-PFX_BREAK_SPAN:]) / PFX_BREAK_SPAN if len(xs) >= PFX_BREAK_SPAN else 0.0
        lim = max(PFX_BREAK_LIMIT, 2 * ref.get(key, PFX_BREAK_LIMIT))
        if key in ref and m > lim:
            return (f"PFX 파괴 가드 — {key} 최근 {PFX_BREAK_SPAN}스텝 평균 {m:.4f} > {lim:.4f}"
                    f"(= max(.05, 2 × 기준선 {ref[key]:.4f})): 맞은 첫 답을 깨고 있다.")
    return None


def stop_reason(steps, mode: str | None = None) -> str | None:
    """PFX(`mode="pfx"`)는 첫 답이 프롬프트에 고정이라 사행·파괴율 규칙이 면제고, `pfx_guard` 하나만 본다."""
    if mode == "pfx":
        return pfx_guard(steps)
    return None


def write_aborted(reason: str) -> None:
    d = os.environ.get("MC_CKPT_DIR")
    if d:
        try:
            Path(d).mkdir(parents=True, exist_ok=True)
            (Path(d) / "ABORTED.txt").write_text(reason + "\n")
        except Exception as e:
            print(f"[MC] ABORTED.txt 기록 실패: {type(e).__name__}: {e}", flush=True)


# ── 토큰 자리 ────────────────────────────────────────────────────────────────
def char_to_tok(tok, ids, c: int) -> int:
    """문자 오프셋 c → 그 앞의 토큰 개수(이분 탐색). ★skip_special_tokens 는 `decode_active` 와 같다."""
    lo, hi = 0, len(ids)
    while lo < hi:
        mid = (lo + hi) // 2
        if len(tok.decode(ids[:mid], skip_special_tokens=True)) < c:
            lo = mid + 1
        else:
            hi = mid
    return lo


# ── 보상 훅 본체 ─────────────────────────────────────────────────────────────
def token_rewards(data, tok, trainer, step) -> tuple[list, dict, dict]:
    """(토큰 보상 [B, T], 구간 크레딧 {행: (시작토큰, [토큰별])}, 계기) — 유일한 팔 SPONT_PFX 로 곧장 위임."""
    arm = str(getattr(getattr(trainer.config, "algorithm", None), "math_arm", "") or "").upper()
    spec = arm_spec(arm)
    return pfx_rewards(data, tok, trainer, step, spec)


def pfx_rewards(data, tok, trainer, step, spec) -> tuple[list, dict, dict]:
    r"""SPONT_PFX — 응답 = 고정 앞부분(프롬프트, `mc.trainer.prefix_agent_loop`) 뒤 이어 쓰기. r = 1[앞부분+이어 쓰기의
    최종 답 ≡ L](`mc.probe.score` — 새 박스가 없으면 앞부분의 첫 답; L = gold 또는 parquet 다수결 `label`)을 마지막
    토큰에 → verl GRPO(같은 앞부분 K 개 = 같은 uid). 앞부분 무게는 **표집**에서만(`trainer.pfx_sampler`, 수정 25).
    PFX_TRUNC: keep(도중 박스로 채점) · mask(32/32c, 학습에서 제외 — 묶음 통계·손실 분모·KL 밖).
    PFX_FORK(ch|hsd|chshuf|chdir, 수정 28/28e/45): 섞인 묶음의 결과 adv 를 내부 대조 PMI 로 재배분(곱).
    PFX_BREAK_W=w(수정 43): 맞은 첫 답을 뒤집어 실패한 행의 결과 adv × w(수정 18 «맞은 첫 답: 망침만 벌» — 어느 말에 몰릴지는 CH).
    PFX_DISTILL=β(수정 51, `pfx_distill`): 틀린 첫 답 행에 «자기 풀이를 못 본 자기 자신» 증류 가산 크레딧.
    PFX_REP=c(수정 43, `pfx_rep`): 같은 답 LOOP_RUN 번째 확인 뒤 말에만 −c 가산(끝낸·잘린 행 모두; PFX_REP_HARD=1 = 그 구간 결과
    adv ≤ 0 · 총량 상한 없음, 수정 46). 계기 = 탐침과 같은 자 + 정보 묶음 수·고친 행 수."""
    from mc.probe import per_problem, score  # noqa: PLC0415
    from mc.trainer import col, decode_active  # noqa: PLC0415
    if outcome_mode() != "group":
        raise ValueError("[MC] SPONT_PFX 는 OUTCOME_MODE=group 전용이다.")
    nt, lbl = data.non_tensor_batch, label_mode()
    B, T = data.batch["responses"].shape
    plen = data.batch["prompts"].shape[-1]
    n_tok = [int(data.batch["attention_mask"][i, plen:].sum()) for i in range(B)]
    texts = decode_active(tok, data)
    pre, lab, pu, gold = (col(nt, k) for k in ("prefix", "label", "problem_uid", "gold"))
    L = [str(g) if lbl == "gold" else str(x) for g, x in zip(gold, lab)]
    a0 = [boxed_answer(p) or "" for p in pre]   # 약속한 첫 답 = 앞부분의 **마지막** 박스(수정 22/23: 앞에 중간 박스 가능, 0925 수리)
    rows: dict[str, list[dict]] = {"": [], "gold_": []}
    for i in range(B):
        for key, ref in (("", L[i]), ("gold_", str(gold[i]))):          # gold_ = 감시 전용
            cls = "none" if not pre[i] else "right" if grade_answer(a0[i], ref) else "wrong"   # none = 빈 앞부분(GRPO 행, 28h)
            rows[key].append({"uid": str(pu[i]), "cls": cls,
                              "n_tok": n_tok[i], "trunc": n_tok[i] >= T, **score(pre[i], texts[i], ref)})
    r = [float(x["final_correct"]) for x in rows[""]]
    trunc = (os.environ.get("PFX_TRUNC") or "keep").strip().lower()
    if trunc not in ("keep", "mask"):
        raise ValueError(f"[MC] PFX_TRUNC={trunc!r} — keep|mask (zero 28f · loop 42 는 실패로 삭제)")
    cut = [i for i in range(B) if rows[""][i]["trunc"]]   # 상한에서 잘림 = 응답을 끝내지 못함
    drop = cut if trunc == "mask" else []
    if drop:                # 수정 32/32c(DAPO overlong filtering): 잘린 행 = 고유 uid(GRPO 묶음 통계 밖) · adv 0 · 손실 마스크 0
        nt["uid"] = np.array([f"{u}#cut{i}" if i in drop else str(u) for i, u in enumerate(nt["uid"])], dtype=object)
    tlr = [[0.0] * int(T) for _ in range(B)]
    for i in range(B):
        if n_tok[i] > 0:
            tlr[i][n_tok[i] - 1] = r[i]
    groups = {u: {r[i] for i in ix} for u, ix in _groups(nt).items() if ix[0] not in drop}
    nw = float(sum(x["cls"] == "wrong" for x in rows[""]))
    tel = {"pfx_wrong_rows": nw, "pfx_right_rows": float(sum(x["cls"] == "right" for x in rows[""])), "pfx_r": sum(r) / max(1, B),
           "pfx_cut_rows": float(len(cut)), "pfx_cut_correct": float(sum(rows[""][i]["final_correct"] for i in cut)),
           "gold_pfx_acc": sum(x["final_correct"] for x in rows["gold_"]) / max(1, B),
           "informative_groups": sum(len(v) > 1 for v in groups.values()) / max(1, len(groups)),
           "pfx_informative_n": float(sum(len(v) > 1 for v in groups.values())), "pfx_groups": float(len(groups)),
           "pfx_fix_rows": float(sum(x["cls"] == "wrong" and x["final_correct"] for x in rows[""])),
           "r_last_all": r, "v5_mode": "pfx", "outcome_mode": "group"}
    for key, rs in rows.items():
        for name, per in per_problem(rs).items():
            if not (key and name in ("tokens", "trunc")):
                tel[f"{key}pfx_{name}"] = (sum(per.values()) / len(per)) if per else float("nan")
    credit: dict = {}
    if (c := _f("PFX_REP", 0.0)) > 0:        # 수정 43 쓸데없는 반복 확인에만 작은 벌(가산 — CH-Fork 곱 뒤)
        if (hd := os.environ.get("PFX_REP_HARD", "")) not in ("", "1"):
            raise ValueError(f"[MC] PFX_REP_HARD={hd!r} — 미설정 또는 1(계보 `_rp<c>h` 와 동작이 어긋나지 않게)")
        tel.update(pfx_rep(tok, data, texts, n_tok, c, credit, drop, hard=hd == "1"))
    fork = (os.environ.get("PFX_FORK") or "").strip().lower()
    if fork:                                  # 수정 28 CH-Fork · HSD 절제 · chshuf 위약
        tel.update(pfx_fork(trainer, tok, data, nt, pre, texts, r, n_tok, L, step, fork))
        print(f"[MC][FORK] step={step} mode={fork} rows={int(tel['fork_rows'])} succ={int(tel['fork_rows_succ'])} "
              f"oom={int(tel['fork_oom'])} bank={int(tel['fork_bank'])} moved={tel['fork_moved']:.3f} top5={tel['fork_top5_share']:.3f} w_front16={tel['fork_w_front16']:.3f} "
              f"w_max={tel['fork_w_max']:.3f} score_succ={tel['fork_score_succ']:.3f} "
              f"score_fail={tel['fork_score_fail']:.3f}", flush=True)
    if (beta := _f("PFX_DISTILL", 0.0)) > 0:   # 수정 51: 자기 풀이를 못 본 자기 자신으로부터의 문맥 증류(가산)
        tel.update(pfx_distill(trainer, tok, data, nt, pre, texts, rows[""], n_tok, drop, beta))
        print(f"[MC][DS] step={step} rows={int(tel['ds_rows'])} oom={int(tel['ds_oom'])} "
              f"neg={tel['ds_neg_frac']:.3f} d_mean={tel['ds_d_mean']:.3f} tok_absmax={tel['ds_tok_absmax']:.2f}", flush=True)
    if (bw := _f("PFX_BREAK_W", 1.0)) != 1.0:   # 수정 43: 맞은 첫 답을 뒤집어 실패 = 결과 벌 × bw(CH 가중 뒤 곱)
        pw = tel.setdefault("pfx_weights", {})
        brk = [i for i in range(B) if rows[""][i]["cls"] == "right" and r[i] <= 0 and n_tok[i] > 0 and i not in drop]
        for i in brk:
            pw[i] = [bw * v for v in pw.get(i, [1.0] * n_tok[i])]
        tel["pfx_break_rows"] = float(len(brk))
    if trunc != "keep":             # 수정 36: 가드는 학습에 들어가는 행만(`pfx_guard`, 옛 키는 비교용으로 그대로)
        for key, rs in rows.items():
            per = per_problem([x for i, x in enumerate(rs) if i not in drop])["break_right"]
            tel[f"{key}pfx_break_right_live"] = sum(per.values()) / len(per) if per else float("nan")
    if drop:                        # 빠진 행 adv 0(외톨이 묶음이라 CH·멈춤 크레딧 대상도 아니다)
        tel["pfx_weights"] = {**(tel.get("pfx_weights") or {}), **{i: [0.0] * n_tok[i] for i in drop}}
        tel["pfx_drop"] = drop
    print(f"[MC][PFX] step={step} informative={int(tel['pfx_informative_n'])}/{int(tel['pfx_groups'])} "
          f"fix_rows={int(tel['pfx_fix_rows'])} wrong_rows={int(nw)}/{B} cut={len(cut)} rep={int(tel.get('rep_rows', 0))} "
          f"break={int(tel.get('pfx_break_rows', 0))}", flush=True)
    dump_pfx(step, nt, texts, r, [x["final_correct"] for x in rows["gold_"]], [x["trunc"] for x in rows[""]])
    _log(step, spec, lbl, tel)
    return tlr, credit, tel


def dump_pfx(step, nt, texts, r, gold_ok, trunc) -> None:
    """스텝마다 `${MC_CKPT_DIR}/pfx_rollouts.jsonl` 에 이어 쓰기 행을 덧붙인다(추가 forward 없음) — 같은 앞부분의
    고친/못 고친 이어 쓰기를 뒤의 내부 신호(PMI) 진단이 읽는다. resume 으로 다시 돈 스텝은 중복될 수 있다."""
    from mc.trainer import col  # noqa: PLC0415
    d = os.environ.get("MC_CKPT_DIR")
    if not d:
        return
    try:
        pid, pix, lab, fa = (col(nt, k) for k in ("pfx_id", "problem_idx", "label", "first_answer"))
        with open(Path(d) / "pfx_rollouts.jsonl", "a") as fh:
            fh.writelines(json.dumps({"step": int(step), "uid": str(pid[i]), "problem_idx": int(pix[i]),
                                      "label": str(lab[i]), "first_answer": str(fa[i]),
                                      "text": texts[i], "r": r[i], "gold_correct": bool(gold_ok[i]),
                                      "trunc": bool(trunc[i])},
                                     ensure_ascii=False) + "\n" for i in range(len(texts)))
    except Exception as e:
        print(f"[MC] pfx_rollouts 기록 실패: {type(e).__name__}: {e}", flush=True)


_LOG_KEYS = ("pfx_wrong_rows", "pfx_right_rows", "informative_groups", "pfx_informative_n", "pfx_fix_rows", "pfx_r",
             "gold_pfx_acc", *(f"{g}pfx_{n}" for g in ("", "gold_") for n in ("reopen_wrong", "revise_wrong", "fix_wrong",
                                                                             "reopen_right", "break_right")),
             "pfx_tokens", "pfx_trunc", "pfx_break_right_live")


def _log(step, spec: dict, lbl, tel: dict) -> None:
    """스텝당 한 줄(PFX 계기 — 옛 로그와 같은 키 이름) + PFX 파괴 가드(사유는 `tel["abort"]`)."""
    body = " ".join(f"{k}={tel[k]:.4f}" for k in _LOG_KEYS if k in tel)
    print(f"[MC] step={step} term={spec['term']} label={lbl} om=group {body}", flush=True)
    HISTORY.append({"step": int(step), **{k: v for k, v in tel.items() if isinstance(v, (int, float))}})
    why = stop_reason(HISTORY, "pfx")
    if why:
        write_aborted(f"step={step} {why}")
        tel["abort"] = why


FORK_CAP = 6000   #: 이웃 이어쓰기 글자 상한(선생님 트리 ≤ ~15k 토큰 — ref 1 forward 메모리; 갈림길은 이어쓰기 앞머리)
FORK_NOTE = ("\n\n(For reference only: below is another continuation that was written from the same point of this "
             "solution, right after the first final answer. It may or may not be right.)\n<<<\n{s}\n>>>")


def fork_neighbor(text: str) -> str:
    r"""선생님 문맥의 이웃 이어쓰기 — FORK_CAP 자 + `\boxed{…}` 안 답을 가림(수정 28b: s⁺ 의 정답 박스를 베끼는 공로 차단)."""
    t = text[:FORK_CAP]
    for _, a, b in reversed(boxed_spans(t)):
        t = t[:a] + "\\boxed{...}" + t[b:]
    return t


def fork_head(tok, problem: str, prefix: str, neighbor: str | None) -> list[int]:
    """선생님 문맥 ids = 문제(+ 중립 문구로 붙인 이웃 이어쓰기) 프롬프트 + 약속 앞부분 — CH-Fork·멈춤·관문 a48 공용."""
    from mc import context as ctx  # noqa: PLC0415
    note = FORK_NOTE.format(s=fork_neighbor(neighbor)) if neighbor is not None else ""
    return _enc(tok, ctx.turn1_prompt(tok, problem + note, "plain")) + tok.encode(prefix, add_special_tokens=False)


def tree(head: list[int], resp: list[int]) -> tuple:
    """(접두, [(t, 토큰열, m)]) — head 뒤 resp 의 토큰별 log p 한 블록(`ref_tree_score`/`score_trees` per_token)."""
    return head, [(len(head) - 1, [head[-1]] + resp, len(resp))]


def _groups(nt) -> dict:
    """uid → 행 번호들(수정 32c mask: 잘린 행은 고유 uid → 외톨이 묶음이라 CH 대조·이웃·멈춤 순위에서 빠진다)."""
    g: dict = {}
    for i, u in enumerate(nt["uid"]):
        g.setdefault(str(u), []).append(i)
    return g


def pfx_fork(trainer, tok, data, nt, pre, texts, r, n_tok, L, step, mode: str) -> dict:
    r"""수정 28/28b CH-Fork — 섞인 묶음(같은 약속 앞부분 이어쓰기 중 성공·실패 모두)의 행마다 교차 적합 이웃(자기 제외)
    s⁺·s⁻ 를 문제 뒤 **중립 문구**로 붙인 선생님 문맥(`fork_head`: 정답·판정 문구 없음, 이웃의 박스 답 가림)에서 동결 ref 로
    응답 토큰별 log p. ch: CH_t = log p(·|s⁺) − log p(·|s⁻)(외톨이 성공·실패는 없는 쪽 = 문맥 없음) · hsd(절제):
    log p(·|s⁺) − log p(·|없음) · chshuf = 같은 가중값을 섞은 위약 · chdir(수정 45) = ch 대조 + `credit.dir_weights`(방향 비례).
    가중은 **누설 전 구간**[0,tL)(첫 새 답 진술 전 = 되짚기 말)에만(결과 방향, 구간 평균 1), 누설 뒤 1 → `pfx_weights`
    (trainer.add_span_credit 가 결과 adv 에 곱한다). 외톨이(같은 쪽 형제 없음) 행은 데이터의 `succ_bank`·`fail_bank`(같은
    앞부분의 base 거르기 이어쓰기, 수정 45)에서 그쪽 이웃을 빌린다 — 별도 rng 라 형제 뽑기 순서(HSD 짝 맞춤)는 그대로."""
    from mc.trainer import col, ref_tree_score  # noqa: PLC0415
    if mode not in ("ch", "hsd", "chshuf", "chdir"):
        raise ValueError(f"[MC] PFX_FORK={mode!r} — ch|hsd|chshuf|chdir (anchor 는 28g 관문 실패로 삭제)")
    probs, rng = col(nt, "problem"), random.Random(int(step))
    ei = list(nt.get("extra_info", []))
    bank = {k: [[str(x) for x in ((e or {}).get(k) if (e or {}).get(k) is not None else ())] for e in ei]
            if ei and all(k in (e or {}) for e in ei) else None for k in ("succ_bank", "fail_bank")}   # parquet → numpy 배열
    trees, owner, n_bank = [], [], 0
    for ix in _groups(nt).values():
        S, F = [i for i in ix if r[i] > 0], [i for i in ix if r[i] <= 0]
        for i in (ix if S and F else []):
            sp, sf = [j for j in S if j != i], [j for j in F if j != i]
            jp, jf = (rng.choice(sp) if sp else None), (rng.choice(sf) if sf else None)   # 두 팔 같은 뽑기
            br = random.Random(int(step) * 7919 + i)
            tp, tf = (texts[j] if j is not None else (br.choice(bank[k][i]) if bank[k] and bank[k][i] else None)
                      for j, k in ((jp, "succ_bank"), (jf, "fail_bank")))
            ctxs = (tp, None) if mode == "hsd" else (tp, tf)
            if n_tok[i] < 1 or ctxs[0] == ctxs[1] or (mode == "hsd" and tp is None):
                continue
            n_bank += (jp is None and tp is not None) + (mode != "hsd" and jf is None and tf is not None)
            resp = [int(t) for t in data.batch["responses"][i, :n_tok[i]]]
            trees += [tree(fork_head(tok, probs[i], pre[i], c), resp) for c in ctxs]
            fin = boxed_answer(pre[i] + texts[i]) or ""
            owner.append((i, char_to_tok(tok, resp, leak_point(texts[i], (L[i], fin)))))
    lps = ref_tree_score(trainer, trees, per_token=True) if trees else []
    w, oom, moved, top, n_s, zs = {}, 0, [], [], 0, {True: [], False: []}
    for k, (i, tl) in enumerate(owner):
        a, b = lps[2 * k], lps[2 * k + 1]
        if a is None or b is None:
            oom += 1
            continue
        tl = min(tl, len(a))
        sc = [a[t] - b[t] for t in range(tl)]
        if sc:
            zs[r[i] > 0].append(sum(sc) / len(sc))
        zf = (dir_weights if mode == "chdir" else fork_weights)(sc, 1.0 if r[i] > 0 else -1.0)
        if mode.endswith("shuf"):            # 위약(수정 28e): 같은 가중값을 구간 안에서 섞음 — 크기 분포 유지, 토큰 정렬 파괴
            random.Random(int(step) * 100003 + i).shuffle(zf)
        w[i] = zf + [1.0] * (len(a) - tl)
        moved.append(sum(abs(v - 1.0) for v in w[i]) / max(1, len(w[i])))
        if zf:                                # 구간 안 상위 5% 말이 가진 몫(균등 = .05) — 신호가 실제로 몰리는가
            top.append(sum(sorted(zf)[-max(1, len(zf) // 20):]) / sum(zf))
        n_s += r[i] > 0
    front = [sum(v[:16]) / len(v[:16]) for v in w.values()]
    return {"pfx_weights": w, "fork_mode": mode, "fork_rows": float(len(w)), "fork_rows_succ": float(n_s),
            "fork_oom": float(oom), "fork_bank": float(n_bank), "fork_moved": sum(moved) / len(moved) if moved else float("nan"),
            "fork_top5_share": sum(top) / len(top) if top else float("nan"),
            "fork_w_front16": sum(front) / len(front) if front else float("nan"),
            "fork_w_max": sum(max(v) for v in w.values()) / len(w) if w else float("nan"),
            **{f"fork_score_{n}": sum(v) / len(v) if v else float("nan") for n, v in (("succ", zs[True]), ("fail", zs[False]))}}


DS_FRONT, DS_CLIP = 16, 2.0   #: 수정 51 — 앞 말 보호(«Wait, let me…» 구조 인공물) · d_t 자르기


def pfx_distill(trainer, tok, data, nt, pre, texts, rows, n_tok, drop, beta: float) -> dict:
    r"""수정 51 문맥 증류(단순판) — **틀린 첫 답** 행의 이어쓰기 말 [DS_FRONT, 끝)(끝 = 마지막 박스 끝·반복 시작 중 앞)에
    말당 가산 크레딧 β·clip(d_t, ±DS_CLIP), d_t = log π_ref(y_t | 문제 + FACT_TMPL(X)) − log π_old(y_t | 문제 + 앞부분)
    (말 단위 역-KL 온폴리시 증류 — 수정 52: 51b 의 행 총량 고정(128/n)은 긴 행에서 몫이 결과 질량의 0.3–1% 로 꺼져 뺐다).
    선생님 = 자기 풀이를 못 본 얼린 자기 자신 → 풀이에 끌려간 말은 벌, 새로 푸는 말은 칭찬(정박 해소의 단일 패스 내재화).
    말당 ≤ β·DS_CLIP. E[d_t] = −KL ≤ 0 이라 평균 음수는 구조적. 결과 부호와 무관해 모든 묶음(만장일치 포함)에 닿는다.
    `compute_advantage` 훅 안에서 불리므로 `old_log_probs` 가 이미 있다(verl: reward → old_log_prob → ref → advantage)."""
    from mc import context as ctx  # noqa: PLC0415
    from mc.trainer import col, ref_tree_score  # noqa: PLC0415
    probs, olp = col(nt, "problem"), data.batch["old_log_probs"]
    spans, trees = [], []
    for i, t in enumerate(texts):
        if rows[i]["cls"] != "wrong" or i in drop or n_tok[i] <= DS_FRONT:
            continue
        resp = [int(x) for x in data.batch["responses"][i, :n_tok[i]]]
        sp, e = boxed_spans(t), loop_char(t)
        j1 = min(n_tok[i], char_to_tok(tok, resp, min(sp[-1][2] if sp else len(t), e if e is not None else len(t))))
        if j1 - DS_FRONT >= 1:
            spans.append((i, j1))
            trees.append(tree(fork_head(tok, probs[i] + ctx.FACT_TMPL.format(answer=boxed_answer(pre[i]) or ""), "", None), resp))
    lps = ref_tree_score(trainer, trees, per_token=True) if trees else []
    out, ds = {}, []
    for (i, j1), lp in zip(spans, lps):
        if lp is None:
            continue
        d = [max(-DS_CLIP, min(DS_CLIP, lp[t] - float(olp[i, t]))) for t in range(DS_FRONT, j1)]
        out[i] = (DS_FRONT, [beta * v for v in d])
        ds += d
    return {"ds_credit": out, "ds_rows": float(len(out)), "ds_oom": float(len(spans) - len(out)),
            "ds_neg_frac": sum(v < 0 for v in ds) / len(ds) if ds else float("nan"),
            "ds_d_mean": sum(ds) / len(ds) if ds else float("nan"),
            "ds_tok_absmax": max((abs(v) for _, vals in out.values() for v in vals), default=0.0)}


LOOP_RUN = 3   #: 수정 42/43 — 같은 답(동치) 박스 연속 3번 = 쓸데없는 확인 시작(평가 «같은 답 3번 멈춤» 모의: 손실 ≈0, 0929)
REP_CAP = 0.1  #: 수정 43 — 반복 벌 총량 ≤ 결과 adv 대비 몫(`trainer.add_span_credit` — 고치기 신호가 늘 주인)


def loop_char(text: str) -> int | None:
    """같은 답(동치) 박스가 연속 LOOP_RUN 번째로 닫힌 글자 뒤(없으면 None) — 그 뒤 말 = 쓸데없는 반복 확인."""
    run, prev = 0, None
    for a, _, e in boxed_spans(text):
        run = run + 1 if prev is not None and (a.strip() == prev.strip() or grade_answer(a, prev)) else 1
        if run >= LOOP_RUN:
            return e
        prev = a
    return None


def pfx_rep(tok, data, texts, n_tok, c: float, credit: dict, drop: list, hard: bool = False) -> dict:
    r"""수정 43 — 이어쓰기마다 `loop_char` 뒤 토큰에만 −c 가산(끝낸 행·잘린 행 모두: 잘린 행이 벌을 피하는 구멍(H4c 실패 원인)을
    막고, 행 전체 벌(L 실패 원인) 대신 반복 구간만). 잘린(drop) 행은 결과 adv 0 이지만 그 구간은 손실에 남긴다(`pfx_keep_from`).
    hard(수정 46, `PFX_REP_HARD=1`): 반복 구간의 결과 adv 를 ≤ 0 으로 막고(`rep_clamp` — 정답으로 끝난 반복이 꼬리까지 칭찬받던
    구멍) 총량 상한 없이 말당 −c(긴 반복일수록 말당 벌이 작아지던 구멍). 행 전체가 아니라 반복 구간만이라 수정 42 L 과 다르다."""
    keep_from, share = {}, []
    for i, t in enumerate(texts):
        if n_tok[i] < 1 or (e := loop_char(t)) is None:
            continue
        j0 = char_to_tok(tok, [int(x) for x in data.batch["responses"][i, :n_tok[i]]], e)
        if j0 >= n_tok[i]:
            continue
        credit[i] = (j0, [-c] * (n_tok[i] - j0))
        share.append((n_tok[i] - j0) / n_tok[i])
        if i in drop:
            keep_from[i] = j0
    return {"credit_cap": 0.0 if hard else REP_CAP, "pfx_keep_from": keep_from, "rep_rows": float(len(share)),
            **({"rep_clamp": {i: j0 for i, (j0, _) in credit.items()}} if hard else {}),
            "rep_tail_share": sum(share) / len(share) if share else float("nan")}
