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

from mc.grade import boxed_answer, boxed_spans, grade_answer
from mc.shift_check import _enc

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
                                                          "PFX_STOP_W", "PFX_TAIL0", "PFX_REP_HARD", "PFX_DISTILL_ROWS",
                                                          "PFX_ALLOC", "PFX_ALLOC_SHUF", "PFX_CONFIRM", "PFX_BRAKE", "PFX_KEEP", "PFX_CELL_KEEP", "PFX_CELL_BREAK", "PFX_ONPOL",
                                                          "PFX_FIN", "PFX_OUTCOME")
                   or k.startswith("PMI2_"))
    if stale:                                  # 지운 손잡이 — 옛 큐 명령이 조용히 다른 팔로 도는 것을 막는다
        raise ValueError(f"[MC] 지운 손잡이 {stale} — PMI2/U/Dir/R4(0925)·R2 검산 비용(수정 33)·맞은 첫 답 표집 몫(28f ii, 수정 38)·멈춤 STOP·TAIL0(수정 43)"
                         "·REP_HARD(수정 46)·리셋 모든 행 DISTILL_ROWS(59b 가드 중단)·말 단위 배분 ALLOC(κ .5·.25 가드 중단)·확인 멈춤 CONFIRM(63c)·멈추는 나 BRAKE(65·65b 관문)·맞은 줄 유지 KEEP(53 고침 미달)·맞은 첫 답 끝내기 FIN(82 10-18 실패→83 판정 교사로 흡수)·결과 보상 끔 OUTCOME(82 10-17 중단)"
                         " — 실패로 삭제됨")
    return m


def describe() -> str:
    from mc.context import default_variant  # noqa: PLC0415
    return f"label={label_mode()} outcome_mode={outcome_mode()} prompt_variant={default_variant()}"


# ── 중단 규칙 — PFX 는 파괴 가드 하나뿐이다 ─────────────────────────────────────
PFX_BREAK_LIMIT, PFX_BREAK_SPAN = 0.05, 3   #: PFX 유일한 가드(`pfx_guard`) — 하한 .05 · 기준선·최근 창 3스텝
FORK_OOM_MAX = 0.1   #: 수정 70 — 선생님 채점 OOM 몫이 이를 넘으면 중단(곱 재가중이 조용히 꺼지는 것을 막음)

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
        xs, fresh = [v for v in (float(s.get(key, "nan")) for s in steps) if v == v], key not in ref
        if key not in ref and len(xs) >= PFX_BREAK_SPAN:
            ref[key] = sum(xs[:PFX_BREAK_SPAN]) / PFX_BREAK_SPAN
            if p is not None:
                p.write_text(json.dumps(ref))
        m = sum(xs[-PFX_BREAK_SPAN:]) / PFX_BREAK_SPAN if len(xs) >= PFX_BREAK_SPAN + fresh else 0.0   # 수정 85b: 새 학습은 스텝 1(학습 전 정책) 을 창에서 뺀다 · resume 은 그대로
        lim = _f("PFX_GUARD_ABS", 0.0) or max(PFX_BREAK_LIMIT, 2 * ref.get(key, PFX_BREAK_LIMIT))   # 수정 53: 씨앗 무관 절대 한계
        if key in ref and m > lim:
            return (f"PFX 파괴 가드 — {key} 최근 {PFX_BREAK_SPAN}스텝 평균 {m:.4f} > {lim:.4f}"
                    f"({'절대 한계 PFX_GUARD_ABS' if _f('PFX_GUARD_ABS', 0.0) else 'max(.05, 2 × 기준선)'}, 기준선 {ref[key]:.4f})"
                    ": 맞은 첫 답을 깨고 있다.")
    last = steps[-1] if steps else {}
    if "fork_lam" in last and last["fork_oom"] > FORK_OOM_MAX * max(1.0, last["fork_rows"] + last["fork_oom"]):   # 수정 70 rlsd·rlrt
        return f"선생님 채점 OOM {last['fork_oom']:.0f}/{last['fork_rows'] + last['fork_oom']:.0f} > {FORK_OOM_MAX:.0%} — 비교군이 조용히 GRPO 가 된다."
    for k, rk, who in (("rt_oom", "rt_rows", "판정 교사"), ("ds_oom", "ds_rows", "눈 가린 나")):   # 리뷰 10-12·10-17: 선생님 채점이 조용히 꺼지지 않게
        if k in last and last[k] > FORK_OOM_MAX * max(1.0, last[rk] + last[k]):
            return f"{who} 채점 OOM {last[k]:.0f}/{last[rk] + last[k]:.0f} > {FORK_OOM_MAX:.0%} — 선생님이 조용히 꺼진다."
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
    곱(결과 adv 재배분): PFX_BREAK_W=w(43, 맞은 첫 답 뒤집은 실패 × w) · 비교군 PFX_FORK=rlsd|rlrt|blind(70·82, `pfx_ratio`).
    가산(자기 선생님): PFX_DISTILL=β(51, `pfx_distill`, 틀린 첫 답 행) · PFX_REP=c(43, `pfx_rep`, 같은 답 LOOP_RUN 번째 뒤 −c).
    PFX_GUARD_ABS = 파괴 가드 절대 한계."""
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
    if (c := _f("PFX_REP", 0.0)) > 0:        # 수정 43 쓸데없는 반복 확인에만 작은 벌(가산)
        tel.update(pfx_rep(loop_points(tok, data, texts, n_tok), n_tok, c, credit, drop))
    fork, beta = (os.environ.get("PFX_FORK") or "").strip().lower(), _f("PFX_DISTILL", 0.0)
    vb, route = _f("PFX_VERDICT", 0.0), (os.environ.get("PFX_ROUTE") or "").strip().lower()
    if route not in ("", "blind", "gold", "tie") or (route and beta <= 0) or (os.environ.get("PFX_VERDICT", "").strip() and vb <= 0) \
            or (_f("PFX_VERDICT_MIN", 0) and not (vb > 0 or route)):
        raise ValueError(f"[MC] PFX_ROUTE={route!r} — blind|gold|tie 이고 PFX_DISTILL>0 필요 · PFX_VERDICT 는 양수(수정 83) · PFX_VERDICT_MIN 은 판정을 쓰는 팔만(수정 84)")
    if fork in ("rlsd", "rlrt", "sdpo", "blind"):   # 수정 70·72 비교군(RLRT 논문 세팅): 선생님 비 곱 재가중 · sdpo 가산 증류 · 82 blind = 눈 가린 선생님 곱
        if _f("PFX_DISTILL", 0.0) > 0:
            raise ValueError("[MC] PFX_FORK=rlsd|rlrt|sdpo|blind 는 비교군 — PFX_DISTILL 과 섞지 않는다")
        tel.update(pfx_ratio(trainer, tok, data, nt, pre, texts, r, n_tok, L, fork, set(drop), step))
    elif fork:
        raise ValueError(f"[MC] PFX_FORK={fork!r} — 닫힌 축(CH-Fork ch·hsd 등은 실패로 삭제, 수정 28/43/45/48/65b), rlsd|rlrt|sdpo|blind 만")
    if fork:
        print(f"[MC][FORK] step={step} mode={fork} " + " ".join(f"{k[5:]}={v:.3f}" for k, v in tel.items()
                                                               if k.startswith("fork_") and isinstance(v, float)), flush=True)
    src = (os.environ.get("PFX_VERDICT_SRC") or "").strip().lower()
    if src not in ("", "probe") or (src and vb <= 0) or (not src and (os.environ.get("PFX_PROBE_TAU") or "").strip()) \
            or not 0 < _f("PFX_PROBE_TAU", 0.5) < 1:
        raise ValueError(f"[MC] PFX_VERDICT_SRC={src!r} — ''|probe 이고 PFX_VERDICT>0 필요 · PFX_PROBE_TAU 는 (0,1) 이고 probe 에서만(수정 90)")
    vd = verdicts(tok, data, rows[""], pre, texts, n_tok, drop) if vb > 0 or route else {}   # 수정 83: 둘째 답이 드러낸 판정
    di = ds_rows(rows[""], n_tok, drop, ("wrong",)) if beta > 0 else []
    ends = {i: v[3] for i, v in vd.items() if route and v[:2] == ("wrong", "diff")}   # 고치기 교사가 맡는 말은 눈 가린 나 창에서 뺀다
    bl, tried = blind_scores(trainer, tok, data, nt, pre, texts, di, n_tok, ends) if di else ({}, set())
    if beta > 0:                               # 수정 51: 자기 풀이를 못 본 자기 자신으로부터의 문맥 증류(가산)
        tel.update(pfx_distill(data, {i: bl[i] for i in di if i in bl}, sum(i in tried and i not in bl for i in di), rows[""], beta))
        print(f"[MC][DS] step={step} rows={int(tel['ds_rows'])} oom={int(tel['ds_oom'])} "
              f"neg={tel['ds_neg_frac']:.3f} d_mean={tel['ds_d_mean']:.3f} tok_absmax={tel['ds_tok_absmax']:.2f}", flush=True)
    if vb > 0:
        pp = None
        if src == "probe":                     # 수정 90: 부호를 정답이 아니라 탐침 p_wrong 이 가른다(앞부분마다 미리 계산한 열)
            try:
                pp = [float(x) for x in col(nt, "p_wrong")]
            except RuntimeError as e:
                raise ValueError("[MC] PFX_VERDICT_SRC=probe 는 extra_info.p_wrong 이 있는 데이터가 필요(a90_prefix_probe.py)") from e
        tel.update(pfx_verdict(vd, vb, pp, _f("PFX_PROBE_TAU", 0.5)))
    if route:
        tel.update(route_credit(trainer, tok, data, nt, pre, texts, L, vd, n_tok, beta, route))
    if (bw := _f("PFX_BREAK_W", 1.0)) != 1.0:   # 수정 43: 맞은 첫 답을 뒤집어 실패 = 결과 벌 × bw(곱)
        pw = tel.setdefault("pfx_weights", {})
        brk = [i for i in range(B) if rows[""][i]["cls"] == "right" and r[i] <= 0 and n_tok[i] > 0 and i not in drop]
        for i in brk:
            pw[i] = [bw * v for v in pw.get(i, [1.0] * n_tok[i])]
        tel["pfx_break_rows"] = float(len(brk))
    if trunc != "keep":             # 수정 36: 가드는 학습에 들어가는 행만(`pfx_guard`, 옛 키는 비교용으로 그대로)
        for key, rs in rows.items():
            per = per_problem([x for i, x in enumerate(rs) if i not in drop])["break_right"]
            tel[f"{key}pfx_break_right_live"] = sum(per.values()) / len(per) if per else float("nan")
    if drop:                        # 빠진 행 adv 0(외톨이 묶음이라 곱 재배분 대상도 아니다)
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
             "pfx_tokens", "pfx_trunc", "pfx_break_right_live", "ver_rows", "ver_same_right", "ver_diff_wrong", "ver_diff_right",
             "ver_same_wrong", "ver_none_right", "ver_none_wrong", "rt_rows", "rt_oom", "rt_fin_rows", "rt_fix_rows", "rt_tie_rows")


def _log(step, spec: dict, lbl, tel: dict) -> None:
    """스텝당 한 줄(PFX 계기 — 옛 로그와 같은 키 이름) + PFX 파괴 가드(사유는 `tel["abort"]`)."""
    body = " ".join(f"{k}={tel[k]:.4f}" for k in _LOG_KEYS if k in tel)
    print(f"[MC] step={step} term={spec['term']} label={lbl} om=group {body}", flush=True)
    HISTORY.append({"step": int(step), **{k: v for k, v in tel.items() if isinstance(v, (int, float))}})
    why = stop_reason(HISTORY, "pfx")
    if why:
        write_aborted(f"step={step} {why}")
        tel["abort"] = why


def fork_head(tok, problem: str, prefix: str) -> list[int]:
    """선생님 문맥 ids = 문제 프롬프트 + 약속 앞부분 — 눈 가린 나·판정 교사·비교군·관문 공용."""
    from mc import context as ctx  # noqa: PLC0415
    return _enc(tok, ctx.turn1_prompt(tok, problem, "plain")) + tok.encode(prefix, add_special_tokens=False)


def tree(head: list[int], resp: list[int]) -> tuple:
    """(접두, [(t, 토큰열, m)]) — head 뒤 resp 의 토큰별 log p 한 블록(`ref_tree_score`/`score_trees` per_token)."""
    return head, [(len(head) - 1, [head[-1]] + resp, len(resp))]


def _groups(nt) -> dict:
    """uid → 행 번호들(수정 32c mask: 잘린 행은 고유 uid → 외톨이 묶음이라 비교군 형제 선택에서 빠진다)."""
    g: dict = {}
    for i, u in enumerate(nt["uid"]):
        g.setdefault(str(u), []).append(i)
    return g


FORK_CAP = 6000   #: 형제 풀이 글자 상한(선생님 트리 ≤ ~15k 토큰 — ref 1 forward 메모리)
GOLD_NOTE = "\n\n(For reference: the correct final answer to this problem is {g}.)"   #: 수정 70 RLSD 선생님 문맥
SOL_NOTE = "\n\n(For reference, here is a correct solution to this problem:)\n<<<\n{s}\n>>>"   #: 수정 70 RLRT 선생님 문맥
RATIO_LAM, RATIO_EPS, RATIO_DECAY = 0.5, {"rlsd": 0.2, "rlrt": 0.5, "blind": 0.2}, {"rlsd": 50, "rlrt": 30, "blind": 10**6}   #: RLRT 논문 표 5(instruct): λ_init·ε_w·λ 감쇠 스텝
SDPO_BETA = 0.5   #: 수정 72 SDPO 식 비교군 — 가산 증류 세기(리셋 PFX_DISTILL 과 같은 β: 선생님 문맥·행·구간만 다름)


def _sol(t: str, half: int = FORK_CAP // 2) -> str:
    """맞힌 형제 풀이 — 길면 앞·뒤 반씩(끝의 답 박스까지 보이게; 리뷰 10-12: 앞 6,000자만이면 73% 가 답 없음)."""
    return t if len(t) <= 2 * half else t[:half] + "\n[...]\n" + t[-half:]


def pfx_ratio(trainer, tok, data, nt, pre, texts, r, n_tok, L, mode: str, drop: set, step) -> dict:
    r"""수정 70 비교군 — RLSD · RLRT(Kim 외 2026, arXiv 2605.10781)를 같은 GRPO 위에서. Δ_t = log P_T(y_t) − log P_S(y_t): 기본은 두 쪽 모두 얼린
    ref(P_T = ref + 특권 문맥, P_S = `ref_log_prob`), `PFX_TEACHER=policy`(수정 74, 논문 식)면 둘 다 학습 중 정책(P_S = `old_log_probs`).
    무게 w_t = (1−λ_t) + λ_t·clip(e^{±Δ_t}, 1−ε, 1+ε), λ_t = .5·max(0, 1 − step/D)(논문 감쇠) 를 결과 adv 에 곱(`pfx_weights`).
    rlsd: 선생님 = 문제 + 정답(GOLD_NOTE), 지수 sign(A)·Δ. rlrt: 선생님 = 문제 + 맞힌 형제 풀이(SOL_NOTE, 자기 제외, 답 보임), **맞은 행만**,
    지수 −Δ(선생님이 예측 못 한 «스스로 고른» 말 칭찬). 섞인 묶음만(만장일치는 adv 0 — 계산 낭비); 이진 보상이라 sign(A) = 맞음 +1·틀림 −1.
    sdpo(수정 72, Hübotter 외 SDPO 의 말 표본 판): 선생님 = RLRT 와 같은 맞힌 형제 풀이 문맥, **모든 행**(맞은 행은 다른 맞은 형제가 있을 때)·응답 전체에
    가산 SDPO_BETA·clip(log P_T − log π_old, ±DS_CLIP)(`ds_credit` — 리셋과 같은 역-KL 식; 논문의 상위 100 logit JS 대신 표본 말). 감쇠 없음."""
    import math  # noqa: PLC0415
    from mc import context as ctx  # noqa: PLC0415
    from mc.trainer import col, ref_tree_score, teacher_is_policy  # noqa: PLC0415
    probs, eps = col(nt, "problem"), RATIO_EPS.get(mode, 0.0)
    ref = data.batch["old_log_probs" if teacher_is_policy() or mode == "blind" else "ref_log_prob"]   # 수정 74: 선생님이 정책이면 학생도 π_old(논문 식)
    lam = RATIO_LAM * max(0.0, 1.0 - int(step) / RATIO_DECAY.get(mode, 1))
    trees, own = [], []
    for ix in _groups(nt).values():
        S = [j for j in ix if r[j] > 0]
        if (mode != "sdpo" and (lam <= 0 or not 0 < len(S) < len(ix))) or not S:
            continue
        for i in ix:
            if mode == "blind":                # 수정 82: 눈 가린 선생님 × 결과 adv(RLSD 식 곱) — 틀린 첫 답 행의 [DS_FRONT, j1)만
                x, (j0, j1) = boxed_answer(pre[i]) or "", ds_span(tok, texts, data, n_tok, i)
                if i not in drop and pre[i] and not grade_answer(x, L[i]) and j1 > j0:
                    trees.append(tree(fork_head(tok, probs[i] + ctx.FACT_TMPL.format(answer=x), ""),
                                      [int(t) for t in data.batch["responses"][i, :j1]]))
                    own.append(i)
                continue
            if i in drop or n_tok[i] < 1 or (mode == "rlrt" and r[i] <= 0) or (mode != "rlsd" and r[i] > 0 and len(S) < 2):
                continue
            ctxq = probs[i] + (GOLD_NOTE.format(g=L[i]) if mode == "rlsd" else SOL_NOTE.format(s=_sol(texts[[j for j in S if j != i][0]])))
            trees.append(tree(fork_head(tok, ctxq, pre[i]), [int(t) for t in data.batch["responses"][i, :n_tok[i]]]))
            own.append(i)
    w, lps = {}, list(zip(own, ref_tree_score(trainer, trees, per_token=True) if trees else []))
    if mode == "sdpo":                         # 학생 = π_old(리셋 `pfx_distill` 과 같은 식) — 말 단위 파이썬 색인 대신 행 단위 텐서
        olp = data.batch["old_log_probs"]
        d = {i: (olp.new_tensor(lp).double() - olp[i, :len(lp)].double()).clamp(-DS_CLIP, DS_CLIP) for i, lp in lps if lp is not None}
        return {"ds_credit": {i: (0, (SDPO_BETA * v).tolist()) for i, v in d.items()}, "fork_mode": mode, "fork_rows": float(len(d)),
                "fork_oom": float(len(own) - len(d)), "fork_lam": SDPO_BETA,
                "fork_d_mean": float(sum(v.sum() for v in d.values()) / max(1, sum(len(v) for v in d.values()))) if d else float("nan")}
    for i, lp in lps:
        if lp is not None:
            sg = (1.0 if r[i] > 0 else -1.0) * (1.0 if mode in ("rlsd", "blind") else -1.0)
            f0 = DS_FRONT if mode == "blind" else 0      # blind: 앞 DS_FRONT 말은 무게 1(보호)
            w[i] = [1.0 if t < f0 else (1 - lam) + lam * min(1 + eps, max(1 - eps, math.exp(sg * max(-DS_CLIP, min(DS_CLIP, v - float(ref[i, t]))))))
                    for t, v in enumerate(lp)]
    mv = [sum(abs(x - 1) for x in v) / len(v) for v in w.values()]
    return {"pfx_weights": w, "fork_mode": mode, "fork_rows": float(len(w)), "fork_oom": float(len(own) - len(w)), "fork_lam": lam,
            "fork_moved": sum(mv) / len(mv) if mv else float("nan")}


DS_FRONT, DS_CLIP = 16, 2.0   #: 수정 51 — 앞 말 보호(«Wait, let me…» 구조 인공물) · d_t 자르기


def ds_rows(rows, n_tok, drop, cls) -> list[int]:
    """눈 가린 나가 채점할 수 있는 행 — 첫 답 부류 cls · 학습에 남음 · 앞 DS_FRONT 말보다 긺."""
    return [i for i, x in enumerate(rows) if x["cls"] in cls and i not in drop and n_tok[i] > DS_FRONT]


def ds_span(tok, texts, data, n_tok, i) -> tuple[int, int]:
    """눈 가린 나·곱 재가중이 채점하는 이어쓰기 구간 [DS_FRONT, j1) — j1 = 마지막 박스 끝·반복 시작(`loop_char`) 중 앞(응답 말 번호)."""
    t, resp = texts[i], [int(x) for x in data.batch["responses"][i, :n_tok[i]]]
    sp, e = boxed_spans(t), loop_char(t)
    return DS_FRONT, min(n_tok[i], char_to_tok(tok, resp, min(sp[-1][2] if sp else len(t), e if e is not None else len(t))))


def blind_scores(trainer, tok, data, nt, pre, texts, idx, n_tok, ends: dict | None = None) -> tuple[dict, set]:
    r"""눈 가린 나(수정 51) 채점 — 행 i → (j0, j1, log π_ref(y_t | 문제 + FACT_TMPL(X)) 말 목록; `PFX_TEACHER=policy` 면 π_old), X = 앞부분의 답. `ends`(수정 83) = 행 → 창 끝 상한. 둘째 값 = 요청한 행(oom 계기)."""
    from mc import context as ctx  # noqa: PLC0415
    from mc.trainer import col, ref_tree_score  # noqa: PLC0415
    probs, spans, trees = col(nt, "problem"), [], []
    for i in idx:
        j0, j1 = ds_span(tok, texts, data, n_tok, i)
        j1 = min(j1, (ends or {}).get(i, j1))
        if j1 - j0 >= 1:
            spans.append((i, j0, j1))
            trees.append(tree(fork_head(tok, probs[i] + ctx.FACT_TMPL.format(answer=boxed_answer(pre[i]) or ""), ""),
                              [int(x) for x in data.batch["responses"][i, :n_tok[i]]]))
    lps = ref_tree_score(trainer, trees, per_token=True) if trees else []
    return {i: (j0, j1, lp) for (i, j0, j1), lp in zip(spans, lps) if lp is not None}, {i for i, *_ in spans}


STOP_K = 8   #: 수정 82 — 끝내기 선생님 창(둘째 박스 줄 끝 뒤 말 수; a85: 끝낼지 다시 열지는 첫 몇 말에서 갈린다)


def verdict(pre_i: str, text: str, c0: int = 0) -> tuple[str, int] | None:
    r"""수정 83 — 이어쓰기의 첫 `\boxed`(둘째 답 Y)가 앞부분의 답 X 와 같은가: ("same"|"diff", 그 박스 끝 글자) · 박스 없으면 None.
    수정 84: c0 글자 앞에서 시작한 박스(«Final Answer» 처럼 계산 없이 답만 다시 적기)는 판정이 아니므로 건너뛴다."""
    sp = [s for s in boxed_spans(text) if s[1] >= c0]
    return None if not sp else ("same" if same_ans(sp[0][0], boxed_answer(pre_i) or "") else "diff", sp[0][2])


def verdicts(tok, data, rows, pre, texts, n_tok, drop) -> dict:
    """수정 83 — 대상 행(맞/틀 첫 답 · 남은 행 · 말 있음) → (부류, 판정 same|diff|none, 둘째 박스 끝 글자, jY). jY = 둘째 박스까지의 말 수(없으면 행 전체)."""
    out, k = {}, int(_f("PFX_VERDICT_MIN", 0))   # 수정 84: 이어쓰기 앞 k 말 안에서 시작한 박스는 답 다시 적기로 보고 건너뜀
    for i, x in enumerate(rows):
        if x["cls"] in ("right", "wrong") and i not in drop and n_tok[i] > 0:
            resp = [int(t) for t in data.batch["responses"][i, :n_tok[i]]]
            v = verdict(pre[i], texts[i], len(tok.decode(resp[:k], skip_special_tokens=True)) if k > 0 else 0)
            out[i] = (x["cls"], v[0], v[1], char_to_tok(tok, resp, v[1])) if v else (x["cls"], "none", -1, n_tok[i])
    return out


def pfx_verdict(vd: dict, v: float, p: list | None = None, tau: float = 0.5) -> dict:
    """수정 83 판정 보상(PFX_VERDICT=v) — 맞→같음·틀→다름 +v, 맞→다름·틀→같음 −v, 틀린 첫 답인데 다시 안 엶(놓침) −v 를 [0, jY) 모든 말에 가산.
    맞은 첫 답을 안 연 줄은 0(다시 안 열기가 다시 열고 같다고 하기보다 유리해지지 않게, 놓침은 둘 다 −).
    수정 90 PFX_VERDICT_SRC=probe(p = 행별 탐침 p_wrong): 정답 부류를 안 쓰고 v·2(p−τ)·s — s=+1 다시 열고 바꿈 · −1 같음/안 엶(안 엶은 jY = 행 전체). p 높으면 열기·낮으면 지키기가 +."""
    sg = lambda c, k: (-1 if c == "wrong" else 0) if k == "none" else (1 if (c == "right") == (k == "same") else -1)  # noqa: E731
    cells = {f"ver_{k}_{c}": 0.0 for k in ("same", "diff", "none") for c in ("right", "wrong")}
    for c, k, _, _ in vd.values():
        cells[f"ver_{k}_{c}"] += 1
    cr = {i: (0, [sg(c, k) * v] * jy) for i, (c, k, _, jy) in vd.items() if sg(c, k)}
    if p is not None:
        cr = {i: (0, [x] * jy) for i, (_, k, _, jy) in vd.items() if (x := v * 2 * (p[i] - tau) * (1 if k == "diff" else -1))}
        cells["ver_probe_mean"] = sum(p[i] for i in vd) / max(1, len(vd))
    return {"ver_credit": cr, "ver_rows": float(sum(k != "none" for _, k, _, _ in vd.values())), **cells}


def route_credit(trainer, tok, data, nt, pre, texts, L, vd: dict, n_tok, beta: float, mode: str) -> dict:
    r"""수정 83 판정 교사(PFX_ROUTE=blind|gold|tie, 가산 β·clip(log P_T − log π_old, ±DS_CLIP)) — 학생 스스로 쓴 판정으로 교사를 가른다(새는 신호 방지).
    같은 답·맞은 첫 답 = 끝내기 교사(수정 82 에서 이어받음): 둘째 박스 줄 끝에 STOP_NOTE 를 넣은 문맥으로 그 뒤 STOP_K 말.
    다른 답(맞/틀) = 고치기 교사: 문제 + FACT(X)(gold 면 + 정답 쪽지)로 jY 이후를 새로 푼다 → [jY + DS_FRONT, j1). 같은 답·틀린 첫 답 = 교사 없음. 트리 한 번에 채점.
    수정 85 tie = 끝내기 없이 다른 답(맞/틀)마다 «두 답 X·Y 가 엇갈림» 쪽지(TIE_TMPL)로 [Y 줄 끝, j1) — DS_FRONT 건너뜀 없음(Y 뒤 첫 문장 = 확정 vs 다시 풀기)."""
    from mc import context as ctx  # noqa: PLC0415
    from mc.trainer import col, ref_tree_score  # noqa: PLC0415
    probs, trees, at = col(nt, "problem"), [], []
    for i, (cls, kind, ce, jy) in vd.items():
        resp, t = [int(x) for x in data.batch["responses"][i, :n_tok[i]]], texts[i]
        if kind == "same" and cls == "right" and mode != "tie":
            cut = (t + "\n").find("\n", ce)          # 둘째 박스가 있는 줄의 끝(줄바꿈 앞)
            jc = char_to_tok(tok, resp, cut)
            if jc < n_tok[i]:
                trees.append(tree(fork_head(tok, probs[i], pre[i] + t[:cut] + ctx.STOP_NOTE), resp[jc:jc + STOP_K]))
                at.append((i, jc, 0, "fin"))
        elif kind == "diff":
            j1 = ds_span(tok, texts, data, n_tok, i)[1]
            if mode == "tie":
                j0 = char_to_tok(tok, resp, (t + "\n").find("\n", ce))   # Y 줄 끝부터 — 쪽지 바로 뒤에 오는 닫는 서식($$ 등) 인공물 제외
                if j1 > j0:
                    y = next(b[0] for b in boxed_spans(t) if b[2] == ce)
                    trees.append(tree(fork_head(tok, probs[i] + ctx.TIE_TMPL.format(x=boxed_answer(pre[i]) or "", y=y), ""), resp[j0:j1]))
                    at.append((i, j0, 0, "tie"))
            elif j1 > jy + DS_FRONT:
                trees.append(tree(fork_head(tok, probs[i] + ctx.FACT_TMPL.format(answer=boxed_answer(pre[i]) or "")
                                            + (GOLD_NOTE.format(g=L[i]) if mode == "gold" else ""), ""), resp[jy:j1]))
                at.append((i, jy + DS_FRONT, DS_FRONT, "fix"))
    lps = ref_tree_score(trainer, trees, per_token=True) if trees else []
    olp, out, n = data.batch["old_log_probs"], {}, {"fin": 0, "fix": 0, "tie": 0}
    for (i, j0, lo, kind), lp in zip(at, lps):
        if lp is not None:
            out[i] = (j0, [beta * max(-DS_CLIP, min(DS_CLIP, v - float(olp[i, j0 + k]))) for k, v in enumerate(lp[lo:])])
            n[kind] += 1
    return {"rt_credit": out, "rt_rows": float(len(out)), "rt_oom": float(len(at) - len(out)),
            "rt_fin_rows": float(n["fin"]), "rt_fix_rows": float(n["fix"]), "rt_tie_rows": float(n["tie"])}


def pfx_distill(data, bl: dict, n_oom: int, rows, beta: float) -> dict:
    r"""수정 51 문맥 증류(단순판) — `ds_rows`(기본 **틀린 첫 답** 행)의 이어쓰기 말 [j0, j1)(앞부분 행 j0 = DS_FRONT, 온폴리시 68)에
    말당 가산 β·clip(d_t, ±DS_CLIP), d_t = log π_ref|π_old(수정 74)(y_t | 문제 + FACT_TMPL(X)) − log π_old(y_t | 문제 + 앞부분)(역-KL 온폴리시 증류).
    선생님 = 자기 풀이를 못 본 얼린 나(`blind_scores`) → 끌려간 말은 벌, 새로 푸는 말은 칭찬. E[d]=−KL ≤ 0 · 만장일치 묶음에도 닿음.
    `PFX_DISTILL_SHUF=1`(수정 55): 위약 — 행 안에서 d_t 의 자리를 섞는다(크기·부호 분포·행 총량 보존; 10-07 probe: 고침 −.122).
    `PFX_DISTILL_SIGN=1`(수정 87): 결과와 방향이 같은 칸만 — 맞힌 행은 d_t>0(새로 풀기)만, 틀린 행은 d_t<0(끌려감)만 남긴다
    («새로 풀었지만 틀림» 칭찬·«참고해서 맞힘» 벌 = 결과 보상과 싸우는 몫 제거; 방향 = 결과, 자리 = 눈가림 선생님).
    `compute_advantage` 훅 안에서 불리므로 `old_log_probs` 가 이미 있다(verl: reward → old_log_prob → ref → advantage)."""
    olp, shuf, sign = data.batch["old_log_probs"], os.environ.get("PFX_DISTILL_SHUF") == "1", os.environ.get("PFX_DISTILL_SIGN") == "1"
    if shuf and sign:
        raise ValueError("[MC] PFX_DISTILL_SHUF 와 PFX_DISTILL_SIGN 은 함께 쓰지 않는다(수정 87)")
    out, ds, cut = {}, [], 0.0
    for i, (j0, j1, lp) in bl.items():
        d = [max(-DS_CLIP, min(DS_CLIP, lp[t] - float(olp[i, t]))) for t in range(j0, j1)]
        if shuf:                       # 수정 55 위약: 같은 값·같은 행 총량, 말 자리만 섞음(선생님 «내용» 제거)
            random.Random(len(d) * 1009 + i).shuffle(d)
        if sign:                       # 수정 87: 결과 부호와 어긋난 칸은 0
            ok = bool(rows[i]["final_correct"]); keep = [v if (v > 0) == ok and v != 0 else 0.0 for v in d]
            cut += sum(abs(v) for v in d) - sum(abs(v) for v in keep); d = keep
        out[i] = (j0, [beta * v for v in d])
        ds += d
    return {"ds_credit": out, "ds_rows": float(len(out)), "ds_oom": float(n_oom), "ds_shuf": float(shuf), "ds_sign": float(sign),
            "ds_sign_cut": cut / max(1e-9, cut + sum(abs(v) for v in ds)),   # 수정 87: 부호가 어긋나 지운 몫(|d| 질량 비)
            "ds_neg_frac": sum(v < 0 for v in ds) / len(ds) if ds else float("nan"),
            "ds_d_mean": sum(ds) / len(ds) if ds else float("nan"),
            "ds_tok_absmax": max((abs(v) for _, vals in out.values() for v in vals), default=0.0)}


def same_ans(a: str, b: str) -> bool:
    """답 동치(글자 같음 또는 수학 동치) — 반복(`loop_char`)·분석 스크립트 공용."""
    return a.strip() == b.strip() or bool(grade_answer(a, b))


LOOP_RUN = 3   #: 수정 42/43 — 같은 답(동치) 박스 연속 3번 = 쓸데없는 확인 시작(평가 «같은 답 3번 멈춤» 모의: 손실 ≈0, 0929)
REP_CAP = 0.1  #: 수정 43 — 반복 벌 총량 ≤ 결과 adv 대비 몫(`trainer.add_span_credit` — 고치기 신호가 늘 주인)


def loop_char(text: str) -> int | None:
    """같은 답(동치) 박스가 연속 LOOP_RUN 번째로 닫힌 글자 뒤(없으면 None) — 그 뒤 말 = 쓸데없는 반복 확인."""
    run, prev = 0, None
    for a, _, e in boxed_spans(text):
        run = run + 1 if prev is not None and same_ans(a, prev) else 1
        if run >= LOOP_RUN:
            return e
        prev = a
    return None


def loop_points(tok, data, texts, n_tok) -> dict:
    """행 → `loop_char` 자리의 말 번호(행 끝 전만) — 반복 벌(`pfx_rep`)의 자리."""
    out = {}
    for i, t in enumerate(texts):
        if n_tok[i] > 0 and (e := loop_char(t)) is not None:
            j = char_to_tok(tok, [int(x) for x in data.batch["responses"][i, :n_tok[i]]], e)
            if j < n_tok[i]:
                out[i] = j
    return out


def pfx_rep(lp: dict, n_tok, c: float, credit: dict, drop: list) -> dict:
    r"""수정 43 — 이어쓰기마다 반복 자리(`loop_points`) 뒤 토큰에만 −c 가산(끝낸 행·잘린 행 모두: 잘린 행이 벌을 피하는 구멍(H4c 실패 원인)을
    막고, 행 전체 벌(L 실패 원인) 대신 반복 구간만). 잘린(drop) 행은 결과 adv 0 이지만 그 구간은 손실에 남긴다(`pfx_keep_from`).
    (수정 46 hard 판 = 구간 칭찬 차단·상한 없음은 10-02 망침 가드 중단으로 10-07 삭제.)"""
    for i, j0 in lp.items():
        credit[i] = (j0, [-c] * (n_tok[i] - j0))
    share = [(n_tok[i] - j0) / n_tok[i] for i, j0 in lp.items()]
    return {"credit_cap": REP_CAP, "pfx_keep_from": {i: j0 for i, j0 in lp.items() if i in drop}, "rep_rows": float(len(lp)),
            "rep_tail_share": sum(share) / len(share) if share else float("nan")}
