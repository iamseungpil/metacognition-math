#!/usr/bin/env python
r"""math_meta_probe — «메타인지 **행위 그 자리**의 내부상태가 읽히는가» 를 재는 자(ruler).

왜: 우리 «문제 안» 자는 지금까지 **층 하나(L36) · 자리 하나(자발적 메타 블록 끝) · 선형 머리**
  로만 읽혔고 천장이 문제 안 AUC **.577**(§C1, `docs/HYPOTHESIS_LEDGER_cd9.md`) — 토큰 스칼라
  12종의 천장(≤.572)과 같다. 그런데 HSRM(arXiv 2608.30841; **얼린** 생성기의 은닉 위에 ~2M 파라미터
  인코더를 자기 생성 궤적만으로, 사람 라벨 없이 학습)은 같은 «문제 안» 양이 입력을 풍부하게 하는
  것만으로 .519(텍스트 검증기) → .691(마지막 은닉) → .714(단계 경계) → **.724(상위 4개 층)** 로
  오른다고 보고한다. 즉 우리 천장은 **프로브 설계의 산물**일 수 있다. 이 도구는 그 가능성을 **행위
  하나하나의 자리에서** 잰다 — 뭉뚱그린 궤적 통계가 아니라 메타인지가 일어난 바로 그 토큰에서.

설계: 입력 = `math_meta_content_gate.py` 의 `gens.jsonl`. 조건 정의·프롬프트 조립은 **그 모듈에서
  그대로 import**(META_ACTS / act_prompt / act_continuation) — 다시 유도하면 생성 때와 다른 문맥에서
  은닉을 읽는다. 문제 본문은 `gate_summary.json` 이 적어 둔 원 롤아웃에서 되찾고, 프롬프트를 바이트
  동일하게 못 되살리는 행(사라진 조건·복원 불가 후보값)은 **버리고 센다**. 자리 넷(`POSITIONS` —
  정의는 `locate_act_span` / `char_positions`)을 **한 forward 로** 읽고, 자리가 없는 행은 **«없음»
  으로 세고 대체하지 않는다**(전체 응답으로 조용히 물러서지 않는다). 프로브는 `grouped_oof_probe`
  (문제-그룹 5-fold OOF, l2=1.0) — 같은 문제가 학습·평가에 갈리지 않는다. 셀마다 §C1 의 세 숫자
  (풀링 AUC · **문제 안 AUC**: 혼합문제 per-problem AUC 평균 + 문제 단위 부트스트랩 CI 2,000/seed 11
  + 혼합문제 수 · 난이도 ρ) + 비-잘림 한정 판 + 버린 행 수. 단일 층 대 **층 이어붙이기**(top4/all)를
  같은 표에 올려 «여러 층» 주장을 시험 가능하게 둔다.
  ★고차원 대책: 2,560차원 × 여러 층을 IRLS 로 바로 못 푼다(헤시안 d³). **씨앗 고정 가우시안 무작위
    사영**(`--proj_dim`, 기본 256)을 모든 셀에 **똑같이** 건다 — 단일 층과 이어붙인 층이 같은 용량을
    받아야 «층을 늘려서 올랐다» 가 성립한다.

판정: **없다 — 이것은 자다.** `BEST-WITHIN: <cond> <position> <layers> <auc> [ci]` 한 줄씩일 뿐.
  ★함정 하나를 같이 읽는다: cd9 G4 는 조건별 정확도 순서가 **절단률 순서의 정확한 역순**이라 판정이
    통째로 절단 산물이었다. 그래서 모든 AUC 옆에 절단률을 찍고, «문제 안 AUC 순위» 와 «절단률 순위»
    의 Spearman 을 내 |ρ| 가 크면 **[CONFOUND?]** 를 요약에 크게 남긴다.

사용(GPU — transformers forward 한 번, vllm 불필요. scipy 가 필요하므로 **simplerl** env):
  CUDA_VISIBLE_DEVICES=2 python scripts/local/math_meta_probe.py --gens <gate_dir>/gens.jsonl \
      --model_path /hdd_data/seungpil/scratch/models/Qwen3-4B-Instruct-2507 \
      --out_dir /hdd_data/seungpil/scratch/eval/meta_probe_prompt_s1 --max_rows_per_cond 400
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import math_ruler_pivot as P  # noqa: E402  (Job/hf_forward_factory/grouped_oof_probe 재사용)
import math_uncertainty_ruler as U  # noqa: E402  (§C1 세 숫자의 관례를 그대로 쓴다)
from math_meta_content_gate import (  # noqa: E402  (조건·프롬프트의 단일 진실 원천)
    META_ACTS, PLAN_RX_HEAD_CHARS, act_continuation, act_prompt, pick_non_plurality_wrong,
)
from src.training.math_meta import boxed_spans, last_boxed  # noqa: E402

_NAN, POSITIONS = float("nan"), ("meta_end", "answer_start", "last", "post_answer")
N_BOOT, BOOT_SEED = 2000, 11        # §C1 관례(문제 단위 재표본)
L2, PROJ_DIM = 1.0, 256             # §C1 은닉 프로브와 같은 정칙화 / 사영 차원(0 이면 끈다)
MAX_ACT_CHARS, CONFOUND_RHO = 600, 0.60   # 스팬 상한 / |ρ(AUC 순위, 절단률 순위)| 경보선

# ★행위마다 «메타인지가 언제 일어나는가» 가 다르므로 자리 정의도 다르다.
#   post : 답을 낸 **뒤** 하는 검사(대입·특수경우·재계산) → 그 검사의 마지막 토큰.
#   pre  : 풀기 **전** 하는 행위(제약 재진술·계획·크기 기대·후보 검증) → 풀이 본론 직전 토큰.
#   mid  : 풀이 **중** 반복되는 검사(단계 검산) → 마지막 답 앞의 마지막 검사 문단 끝.
#   ★META_ACTS 에 행위를 더하면 여기도 채운다(빠지면 테스트가 그 이름을 찍고 떨어진다).
ACT_PHASE = {"substitute": "post", "special_case": "post", "recompute": "post",
             "backward_mask": "post", "constraint": "pre", "plan": "pre", "magnitude": "pre",
             "verification_first": "pre", "verification_first_random": "pre",
             "pad_length": "pre", "stepcheck": "mid"}


def locate_act_span(text: str, cond: str, *, mode: str = "prompt") -> Optional[tuple[int, int]]:
    r"""조건 `cond` 의 메타인지 **행위**가 차지한 문자 구간 (start, end). 없으면 **None**.

    규약(★전체 응답으로 조용히 물러서지 않는다 — 없으면 None 이고 호출자가 «없음»으로 센다):
      · `rx` 가 None 인 기준·대조군(plain/filler/blind/filler_continue)은 **정의상 행위가 없다**.
      · mode="continue": 씨앗(cue)이 프롬프트 끝에 박혀 있으므로 이어쓰기의 **첫 문단**이 그 행위.
      · mode="prompt": `rx` 매치를 단계(ACT_PHASE)에 맞는 구역에서 고른다 — post = 첫 \boxed
        **이후**의 마지막 매치(\boxed 가 없으면 «답 뒤»가 정의되지 않아 None), pre = 첫 \boxed 앞의
        첫 매치(`plan` 은 준수 판정과 같게 서두 600자 안에서만), mid = 마지막 \boxed 앞의 마지막 매치.
      · 끝은 그 매치가 속한 **문단의 끝**. 단 빈 줄 없는 응답이 통째로 삼켜지지 않게 매치 끝 +
        MAX_ACT_CHARS 로 자르고, pre/mid 는 \boxed 시작을 넘지 않는다.
    """
    act = META_ACTS.get(cond)
    if act is None or act.rx is None or not text:
        return None
    if mode == "continue":   # 씨앗(cue)이 프롬프트 끝에 있으므로 이어쓰기 첫 문단이 그 행위다
        j = text.find("\n\n")
        return None if act.cue is None else (0, min(j if j >= 0 else len(text), MAX_ACT_CHARS))
    rx = re.compile(act.rx, re.I)   # ★"{CAND}" 갈래(verification_first*)는 리터럴이라 안 맞는다
                                    #   — 나머지 갈래로만 자리를 찾는다(자리 정의에 후보는 불필요)
    phase = ACT_PHASE.get(cond, "pre")
    boxes = boxed_spans(text)
    first_b, last_b = (boxes[0][1], boxes[-1][1]) if boxes else (None, None)
    head = text[:PLAN_RX_HEAD_CHARS] if cond == "plan" else text
    if phase == "post" and first_b is None:
        return None                          # 답이 없으면 «답 뒤 검사» 자리가 없다
    if phase == "post":
        ms = [m for m in rx.finditer(text) if m.start() >= first_b]
        m, cap = (ms[-1] if ms else None), len(text)
    elif phase == "mid":
        ms = [m for m in rx.finditer(head) if last_b is None or m.start() < last_b]
        m, cap = (ms[-1] if ms else None), (last_b if last_b is not None else len(text))
    else:
        ms = [m for m in rx.finditer(head) if first_b is None or m.start() < first_b]
        m, cap = (ms[0] if ms else None), (first_b if first_b is not None else len(text))
    if m is None:
        return None
    para = text.find("\n\n", m.end())          # 그 매치가 속한 문단의 끝(없으면 문서 끝)
    end = min(para if para >= 0 else len(text), m.end() + MAX_ACT_CHARS, max(cap, m.end()))
    return (m.start(), max(end, m.end()))


def char_positions(text: str, cond: str, *, mode: str = "prompt") -> dict:
    r"""자리 넷의 **문자 오프셋**(읽을 문자의 인덱스), 없으면 None. meta_end = 행위 스팬의
    마지막 문자 · answer_start = 마지막 \boxed 바로 **앞** 문자 · last = 마지막 문자 ·
    post_answer = 마지막 \boxed{...} 가 닫힌 직후 문자(뒤에 아무것도 없으면 None).
    ★BPE 경계 주의: 그 «앞 문자»를 덮는 토큰이 실제로는 " \"(공백+역슬래시)처럼 \boxed 의 첫
      글자까지 품을 수 있다 — answer_start 는 «답을 쓰기 직전»의 근사다(실측 확인)."""
    out: dict = {p: None for p in POSITIONS}
    sp = locate_act_span(text, cond, mode=mode)
    out["meta_end"] = (sp[1] - 1) if sp is not None and sp[1] > 0 else None
    out["last"] = len(text) - 1 if text else None
    boxes = boxed_spans(text)
    if boxes:
        _c, s, e = boxes[-1]
        out["answer_start"] = s - 1 if s > 0 else None
        out["post_answer"] = e if e < len(text) else None
    return out


def token_index_at(offsets: Sequence[tuple[int, int]], ch: Optional[int]) -> Optional[int]:
    """문자 인덱스 ch 를 덮는 토큰 인덱스. 어떤 토큰도 안 덮으면 None(«없음»으로 센다).
    ★빈 구간 (a,a)(특수토큰 등)은 어떤 문자도 안 덮으므로 자동으로 배제된다."""
    return (None if ch is None or ch < 0 else
            next((i for i, (a, b) in enumerate(offsets) if b > a and a <= ch < b), None))


def projector(dim: int, out_dim: int, seed: int = 0) -> Optional[np.ndarray]:
    """씨앗 고정 가우시안 무작위 사영(dim → out_dim), out_dim 이 0 이거나 dim 이하면 None.
    ★성능이 아니라 **모든 셀에 같은 용량**을 주려는 것 — 단일 층과 이어붙인 층이 다른 차원으로
      프로브에 들어가면 «층을 늘려서 올랐다» 를 못 가른다."""
    return (None if not out_dim or dim <= out_dim else
            np.random.RandomState(seed).normal(0.0, out_dim ** -0.5, (dim, out_dim)))


def cell_stats(scores: np.ndarray, y: np.ndarray, groups: Sequence) -> dict:
    """§C1 의 세 숫자: 풀링 AUC · 문제 안 AUC(혼합문제, 문제 단위 부트스트랩 CI) · 난이도 ρ."""
    from scipy.stats import spearmanr  # noqa: PLC0415
    s, yy = np.asarray(scores, dtype=float), np.asarray(y, dtype=float)  # 점수는 전부 OOF
    gidx: dict = defaultdict(list)
    for i, g in enumerate(groups):
        gidx[g].append(i)
    prate = {g: float(np.mean(yy[idx])) for g, idx in gidx.items()}
    per = U.per_problem_aucs(s, yy, gidx, [g for g, p in prate.items() if 0.0 < p < 1.0])
    mean_w, lo, hi = U.bootstrap_mean_ci(per, n_boot=N_BOOT, seed=BOOT_SEED)
    gm = {g: float(np.mean([s[i] for i in idx if np.isfinite(s[i])]))
          for g, idx in gidx.items() if any(np.isfinite(s[i]) for i in idx)}
    rho = (float(spearmanr(list(gm.values()), [prate[g] for g in gm]).statistic)
           if len(gm) > 5 and len(set(gm.values())) > 1 else _NAN)
    return {"auc_pooled": U.auc(s, yy), "auc_within": mean_w, "ci_lo": lo, "ci_hi": hi,
            "n_mixed_used": len(per), "difficulty_rho": rho, "n_problems": len(gidx)}  # §C1 열


def probe_cell(X: np.ndarray, y: np.ndarray, groups: Sequence, *, seed: int = 11,
               proj: Optional[np.ndarray] = None, head: str = "linear") -> dict:
    """문제-그룹 5-fold OOF 프로브 → §C1 세 숫자. ★학습과 평가가 같은 문제를 보지 않는다.
    `head="mlp"` 는 같은 grouped-OOF 규율 위에 1-은닉층 MLP 를 얹는다(HSRM 대조 —
    천장이 선형 머리의 약점인지 신호 자체가 없는지 가르는 시험, `math_ruler_pivot.
    grouped_oof_probe_mlp` §Why 참조)."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    X = X @ proj if (proj is not None and X.size and X.shape[1] == proj.shape[0]) else X
    if X.shape[0] < 10 or len(set(y.tolist())) < 2:
        pr = {"oof": np.full(len(y), _NAN), "n_folds_used": 0}
    elif head == "mlp":
        pr = P.grouped_oof_probe_mlp(X, y, groups, n_folds=5, seed=seed)
    else:
        pr = P.grouped_oof_probe(X, y, groups, n_folds=5, l2=L2, seed=seed)
    return {**cell_stats(pr["oof"], y, groups), "n_rows": int(X.shape[0]),
            "n_folds_used": int(pr["n_folds_used"])}


def layer_sets(layers: Sequence[int]) -> list[tuple[str, tuple[int, ...]]]:
    """단일 층 각각 + 상위 4개 층 이어붙이기 + 전체 이어붙이기(중복 제거)."""
    ls = sorted(layers)
    out: list[tuple[str, tuple[int, ...]]] = [(f"L{L}", (L,)) for L in ls]
    top4, allL = tuple(ls[-4:]), tuple(ls)
    out += [(n, sub) for n, sub in (("top4", top4), ("all", allL))
            if len(sub) > 1 and (n == "top4" or sub != top4)]
    return out


def confound_check(table: Sequence[dict], trunc_rate: dict) -> dict:
    """★cd9 G4 함정 방어: 조건의 «문제 안 AUC» 순위가 «절단률» 순위와 같이 움직이는가.
    조건마다 (자리·층 전부에서) 최고 문제 안 AUC 를 잡아 절단률과 Spearman 을 낸다."""
    from scipy.stats import spearmanr  # noqa: PLC0415
    best: dict = {}
    for r in table:
        a = r.get("auc_within") if r.get("auc_within") is not None else _NAN
        if np.isfinite(a) and a > best.get(r["cond"], -np.inf):
            best[r["cond"]] = a
    conds = [c for c in best if np.isfinite(trunc_rate.get(c, _NAN))]
    rho = (float(spearmanr([best[c] for c in conds], [trunc_rate[c] for c in conds]).statistic)
           if len(conds) >= 4 else _NAN)
    return {"rho": rho, "n_conds": len(conds), "conds": conds,
            "flag": bool(np.isfinite(rho) and abs(rho) >= CONFOUND_RHO)}  # ★크게 찍을 신호


def group_rows(rows: Sequence[dict], key: str) -> dict:
    """행을 key 값으로 묶는다(조건별·문제별 — 여러 곳이 같은 패턴을 쓴다)."""
    out: dict = defaultdict(list)
    for r in rows:
        out[r[key]].append(r)
    return out


def subsample_rows(rows: list[dict], max_rows: int) -> list[dict]:
    """문제별 라운드로빈 결정적 부분표집 — 조건마다 같은 문제가 남아야 조건 간 비교가 짝지어진다."""
    if not max_rows or len(rows) <= max_rows:
        return rows
    by_g = group_rows(rows, "group_id")
    out: list[dict] = []
    for i in range(max(len(v) for v in by_g.values())):
        out += [by_g[g][i] for g in sorted(by_g) if i < len(by_g[g])]
    return out[:max_rows]


def load_units(gens_path: str, gate_summary: Optional[str] = None) -> tuple[dict, dict]:
    """gens.jsonl 옆 `gate_summary.json` 이 적어 둔 원 롤아웃에서 문제 본문을 되찾는다
    (gens.jsonl 은 문제 본문을 안 담는다). Returns (unit_id → {problem,gold,text}, meta)."""
    sp = Path(gate_summary or Path(gens_path).parent / "gate_summary.json")
    meta, units, box = json.loads(sp.read_text())["meta"], {}, defaultdict(list)
    for path in meta["rollouts"]:
        tag = Path(path).resolve().parent.name
        for i, line in enumerate(Path(path).open()):
            if line.strip():
                r = json.loads(line)
                rec = {"problem": r["problem"], "gold": r.get("gold"), "text": r.get("text") or ""}
                units.setdefault(f"{tag}::{r['group_id']}", {**rec, "text": ""})  # prompt 단위=문제
                units[f"{tag}::{r['group_id']}#{i}"] = rec                        # continue=롤아웃
                box[f"{tag}::{r['group_id']}"].append(last_boxed(rec["text"]))
    for gk, bx in box.items():   # ★verification_first 후보는 형제 답에서 결정적으로 복원된다
        units[gk]["cand_vf"] = pick_non_plurality_wrong(bx, units[gk]["gold"])
    return units, meta


def load_gens(gens_path: str) -> list[dict]:
    """gens.jsonl 한 줄 → 프로브 한 행. 정오는 `gen_r_corr`, 문제는 unit_id("<tag>::<gid>[#줄]")
    의 '#' 앞부분 — 그래야 continue 모드에서도 같은 문제의 행이 한 그룹으로 묶인다."""
    return [{"unit_id": r["unit_id"], "cond": r["cond"], "text": r.get("text") or "",
             "r_corr": float(r["gen_r_corr"]), "trunc": float(r.get("truncated", 0)),
             "n_tok": float(r.get("n_tok", _NAN)), "cand": r.get("cand"),
             "group_id": str(r["unit_id"]).split("#", 1)[0]}
            for r in (json.loads(ln) for ln in Path(gens_path).open() if ln.strip())]


def prepare_row(tok, prompt: str, text: str, cond: str, mode: str) -> Optional[dict]:
    r"""토큰열 + 자리 넷의 토큰 인덱스. ★프롬프트 토큰이 prompt+text 토큰의 **접두가 아니면**
    (경계가 재토큰화로 틀어지면) «응답 위치» 정의가 깨지므로 None 을 돌려 그 행을 버린다."""
    ids_p, full = tok(prompt, add_special_tokens=False)["input_ids"], prompt + text
    try:                      # offset 없는 느린 토크나이저는 접두 길이로 센다(±1 토큰 근사)
        enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
        offs = list(enc["offset_mapping"])
    except Exception:
        enc, offs = tok(full, add_special_tokens=False), None
    ids = list(enc["input_ids"])
    if len(ids) <= len(ids_p) or ids[:len(ids_p)] != ids_p:
        return None
    pos = {n: (None if c is None else
               token_index_at(offs, len(prompt) + c) if offs is not None else
               len(tok.encode(full[:len(prompt) + c + 1], add_special_tokens=False)) - 1)
           for n, c in char_positions(text, cond, mode=mode).items()}
    return {"ids": ids, "n_prompt": len(ids_p), "pos": pos}


def extract(rows: list[dict], units: dict, tok, forward, layers: list[int], *,
            variant: str, mode: str) -> tuple[list[dict], dict]:
    """행마다 forward **한 번**(자리 넷 동시). Returns (특징 붙은 행들, 진단 카운터)."""
    jobs, keep = [], []          # P.Job 들 / 특징을 붙일 행들
    n_rt = n_miss = n_gone = 0   # 라운드트립 실패 / unit_id 못 찾음 / 프롬프트 복원 불가
    for r in rows:
        u = units.get(r["unit_id"])
        if u is None:
            n_miss += 1
            continue
        # 후보값은 기록된 것 우선, 없으면 verification_first 에 한해 결정적 재계산.
        cnd = r.get("cand") or (u.get("cand_vf") if r["cond"] == "verification_first" else None)
        try:
            prompt = (act_prompt(tok, variant, u["problem"], r["cond"], cand=cnd)
                      if mode == "prompt" else
                      act_continuation(tok, variant, u["problem"], u["text"], r["cond"]))
        except KeyError:   # ★조건이 META_ACTS 에서 사라졌거나 후보값을 못 되살렸다 — 프롬프트를
            n_gone += 1    #   생성 때와 바이트 동일하게 못 만드니 그 행을 버리고 센다
            continue
        got = prepare_row(tok, prompt, r["text"], r["cond"], mode)
        if got is None:
            n_rt += 1
            continue
        keep.append({**r, "pos": got["pos"], "job": len(jobs)})
        jobs.append(P.Job(got["ids"], hidden_ats=[p for p in got["pos"].values() if p is not None]))
    print(f"[meta_probe] jobs={len(jobs)} roundtrip_fail={n_rt} unit_missing={n_miss} "
          f"prompt_unrecoverable={n_gone}", flush=True)
    res = forward(jobs, layers) if jobs else []
    for r in keep:   # 자리가 없거나 왼쪽 잘림으로 사라진 자리는 키가 안 생긴다(«없음»)
        hm = (res[r["job"]] or {}).get("hidden_multi", {})
        r["hidden"] = {n: hm[p] for n, p in r["pos"].items() if p in hm}
    return keep, {"n_roundtrip_fail": n_rt, "n_unit_missing": n_miss,
                  "n_prompt_unrecoverable": n_gone, "n_rows_forwarded": len(keep)}


def build_table(rows: list[dict], layers: list[int], positions: Sequence[str], *,
                seed: int, proj_dim: int, head: str = "linear") -> tuple[list[dict], dict]:
    """조건 × 자리 × 층집합 마다 프로브를 새로 적합해 §C1 세 숫자를 낸다.
    `head="mlp"` 면 선형·MLP 둘 다 같은 셀에서 재고(행마다 "head" 필드를 붙여) 선형 숫자가
    비교로 남게 한다 — `head="linear"`(기본)는 옛 표와 바이트 단위로 같다(기존 호출자 안 깨짐)."""
    table: list[dict] = []
    heads = ["linear"] if head != "mlp" else ["linear", "mlp"]
    trunc_rate, projs, by_cond = {}, {}, group_rows(rows, "cond")
    for cond, crs in sorted(by_cond.items()):
        trunc_rate[cond] = float(np.mean([r["trunc"] for r in crs])) if crs else _NAN
        for pos in positions:
            have = [r for r in crs if r["hidden"].get(pos) is not None]
            base = {"cond": cond, "position": pos, "trunc_rate": trunc_rate[cond],
                    "n_missing": len(crs) - len(have)}   # ★없는 자리는 세고 대체하지 않는다
            if not have:
                miss_row = {**base, "layers": "-", "n_rows": 0, "n_dropped_trunc": 0,
                            "auc_within": _NAN, "auc_within_nontrunc": _NAN}
                if head == "mlp":
                    miss_row["head"] = "-"
                table.append(miss_row)
                continue
            y = np.array([r["r_corr"] for r in have], dtype=float)
            g = [r["group_id"] for r in have]
            nt_ix = [i for i, r in enumerate(have) if r["trunc"] < 0.5]
            for name, sub in layer_sets(layers):
                X = np.concatenate([np.stack([r["hidden"][pos][L] for r in have]) for L in sub], 1)
                pj = projs.setdefault(X.shape[1], projector(X.shape[1], proj_dim, seed=seed))
                for h in heads:
                    nt = ({"auc_within": _NAN} if len(nt_ix) < 10 else
                          probe_cell(X[nt_ix], y[nt_ix], [g[i] for i in nt_ix], seed=seed, proj=pj,
                                     head=h))
                    row = {**base, "layers": name, "n_dropped_trunc": len(have) - len(nt_ix),
                           "auc_within_nontrunc": nt["auc_within"],
                           **probe_cell(X, y, g, seed=seed, proj=pj, head=h)}
                    if head == "mlp":
                        row["head"] = h
                    table.append(row)
    return table, trunc_rate


def best_line(table: Sequence[dict], tag: str) -> str:
    """★판정이 아니라 자 한 줄 — 조건·자리·층(·머리, `--head mlp` 일 때)을 통틀어 문제 안 AUC
    가 가장 높은 셀. head 열이 있는 표에서는 선형·MLP 둘 다 후보라 «최고 머리»가 그대로 드러난다."""
    c = [r for r in table if np.isfinite(r.get("auc_within") if r.get("auc_within") is not None
                                         else _NAN) and r.get("n_rows", 0) >= 10]
    b = max(c, key=lambda r: r["auc_within"], default=None)
    if b is None:
        return f"BEST-WITHIN: none - - nan [nan, nan]  ({tag})"
    head_tag = f"/{b['head']}" if b.get("head") else ""
    return (f"BEST-WITHIN: {b['cond']} {b['position']} {b['layers']}{head_tag} "
            f"{U._fmt(b['auc_within'])} [{U._fmt(b['ci_lo'])}, {U._fmt(b['ci_hi'])}]  ({tag})")


def to_markdown(table: Sequence[dict], trunc_rate: dict, diag: dict, tag: str) -> str:
    # ★head 열은 `--head mlp` 로 낸 표에만 붙는다(행에 "head" 키가 있을 때만) — 기본(선형만)
    #   경로는 옛 표와 바이트 단위로 같다(기존 호출자 안 깨짐).
    show_head = any("head" in r for r in table)
    header = ["cond", "position", "layers"] + (["head"] if show_head else []) + [
        "n", "없음", "절단률", "AUC 풀링", "**AUC 문제 안**", "95% CI", "혼합문제", "난이도 ρ",
        "문제 안(비-잘림)", "잘림 제외 행"]
    md = [f"## {tag}", "", f"행 {diag.get('n_rows_forwarded', 0)} · 라운드트립 실패 "
          f"{diag.get('n_roundtrip_fail', 0)} · 단위 못 찾음 {diag.get('n_unit_missing', 0)} · "
          f"프롬프트 복원 불가 {diag.get('n_prompt_unrecoverable', 0)}", "",
          "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in table:
        cells = [r["cond"], r["position"], r["layers"]]
        if show_head:
            cells.append(r.get("head", "-"))
        cells += [str(r.get("n_rows", 0)), str(r["n_missing"]),
                  U._fmt(r["trunc_rate"]), U._fmt(r.get("auc_pooled")),
                  f"**{U._fmt(r.get('auc_within'))}**",
                  f"[{U._fmt(r.get('ci_lo'))}, {U._fmt(r.get('ci_hi'))}]", str(r.get("n_mixed_used", 0)),
                  U._fmt_signed(r.get("difficulty_rho")), U._fmt(r.get("auc_within_nontrunc")),
                  str(r.get("n_dropped_trunc", 0))]
        md.append("| " + " | ".join(cells) + " |")
    cf = confound_check(table, trunc_rate)
    md += ["", f"**{best_line(table, tag)}**", "", f"ρ(조건별 최고 문제 안 AUC 순위, 조건 "
           f"절단률 순위) = {U._fmt_signed(cf['rho'])} (조건 {cf['n_conds']}개)"]
    if cf["flag"]:
        md += ["", f"**[CONFOUND?]** |ρ| ≥ {CONFOUND_RHO} — 조건 사이의 «문제 안 AUC» 순서가 "
               "**절단률 순서와 같이 움직인다**(cd9 G4 에서 정확도 순서가 절단률 순서의 정확한 "
               "역순이었던 것과 같은 종류). 조건 간 비교를 그대로 읽지 말고 비-잘림 열부터 보라."]
    return "\n".join(md) + "\n"


def save_features(out_dir: Path, rows: list[dict], layers: list[int],
                  positions: Sequence[str]) -> None:
    """조건마다 features_<cond>.npz — 은닉은 fp16, 없는 자리는 NaN + mask_<pos> 로 남긴다."""
    for cond, crs in group_rows(rows, "cond").items():   # ★없는 자리는 0 이 아니라 NaN + mask 로 남긴다
        def col(k, t=np.float32, rs=crs): return np.array([r[k] for r in rs], dtype=t)  # noqa: E704
        kw: dict = {"layers": np.array(layers), "target": col("r_corr"),
                    "truncated": col("trunc"), "n_tok": col("n_tok"),
                    "group_id": col("group_id", object), "unit_id": col("unit_id", object)}
        for pos in positions:
            hs = [r["hidden"].get(pos) for r in crs]
            got = [h for h in hs if h is not None]
            kw[f"mask_{pos}"] = np.array([h is not None for h in hs])
            for L in (layers if got else ()):
                nan = np.full(len(got[0][L]), np.nan)
                kw[f"hidden_{pos}_L{L}"] = np.array([nan if h is None else h[L] for h in hs],
                                                    dtype=np.float16)
        np.savez_compressed(out_dir / f"features_{cond}.npz", **kw)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gens", action="append", required=True, help="content gate gens.jsonl")
    ap.add_argument("--gate_summary", default=None, help="기본: gens.jsonl 옆 gate_summary.json")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--layers", default=None, help="쉼표 구분(-1=마지막, mid=중간). 기본은 "
                    "**마지막 4개 + 중간 1개**를 config 에서 뽑는다(하드코딩 없음)")
    ap.add_argument("--positions", default=",".join(POSITIONS))
    ap.add_argument("--max_rows_per_cond", type=int, default=400)
    ap.add_argument("--proj_dim", type=int, default=PROJ_DIM, help="0 이면 무작위 사영 끔")
    ap.add_argument("--head", choices=("linear", "mlp"), default="linear",
                    help="기본은 선형(IRLS) 머리. mlp 는 같은 grouped-OOF 위에 1-은닉층 MLP 도 "
                    "재고 표에 같이 올린다(선형 숫자는 비교로 남는다) — 천장이 «선형 머리가 "
                    "약하다» 때문인지 «신호 자체가 없다» 때문인지 가르는 시험(HSRM 대조, §Why)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=2, help="★logits 버퍼가 B×T×vocab 이라 긴 응답에선 이 값이 VRAM 을 지배한다")
    ap.add_argument("--max_len", type=int, default=12288)
    ap.add_argument("--seed", type=int, default=11, help="fold 배정 씨앗(부트스트랩은 11 고정)")
    a = ap.parse_args()
    positions = [p.strip() for p in a.positions.split(",") if p.strip()]
    if [p for p in positions if p not in POSITIONS]:
        raise SystemExit(f"[meta_probe] 모르는 자리 {positions} — 가능한 값 {POSITIONS}")
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    forward, tok, n_layers = P.hf_forward_factory(a.model_path, a.device, a.batch_size, a.max_len)
    layers = (P.parse_layers(a.layers, n_layers) if a.layers
              else sorted({n_layers // 2, n_layers - 3, n_layers - 2, n_layers - 1, n_layers}))
    print(f"[meta_probe] n_layers={n_layers} layers={layers} positions={positions}", flush=True)

    md = ["# math_meta_probe — 메타인지 행위 자리의 내부상태 자", "",
          "★판정 없음(자다). §C1 비교선: 은닉 L36 metaend 문제 안 **.577** · 적힌 확신도 .568 ·"
          " 토큰 스칼라 천장 ≤.572 · 형제 답 점유율 .754. HSRM(2608.30841): .691 → .714 → .724.",
          ""]
    summary: dict = {"files": {}, "layers": layers, "positions": positions, "l2": L2,
                     "proj_dim": a.proj_dim, "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
                     "head": a.head}
    pops: list[tuple[str, list[dict], dict]] = []
    for gp in a.gens:
        units, meta = load_units(gp, a.gate_summary)
        mode, variant = meta.get("mode", "prompt"), meta.get("variant", "math_opt")
        by_cond = group_rows(load_gens(gp), "cond")   # 조건마다 따로 상한(문제 집합을 맞춘다)
        rows = [r for c in sorted(by_cond) for r in subsample_rows(by_cond[c], a.max_rows_per_cond)]
        print(f"[meta_probe] {gp}: mode={mode} n={len(rows)} conds={sorted(by_cond)}", flush=True)
        got, diag = extract(rows, units, tok, forward, layers, variant=variant, mode=mode)
        save_features(od, got, layers, positions)
        pops.append((Path(gp).resolve().parent.name, got, {**diag, "mode": mode, "gens": gp}))
    if len(pops) > 1:   # 두 코퍼스를 합친 판도 같은 관례로 한 번 더 찍는다
        mg = [r for _t, g, _d in pops for r in g]
        pops.append(("ALL(합침)", mg, {"n_rows_forwarded": len(mg)}))
    for tag, got, diag in pops:
        table, trunc = build_table(got, layers, positions, seed=a.seed, proj_dim=a.proj_dim,
                                   head=a.head)
        md.append(to_markdown(table, trunc, diag, tag))
        summary["files"][tag] = {"diag": diag, "trunc_rate": trunc, "table": table,
                                 "confound": confound_check(table, trunc)}
        print(best_line(table, tag), flush=True)

    (od / "meta_probe_summary.md").write_text("\n".join(md), encoding="utf-8")
    (od / "meta_probe_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2,
        default=lambda o: None if isinstance(o, float) and not np.isfinite(o) else float(o)))
    print(f"[out] {od}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
