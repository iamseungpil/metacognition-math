#!/usr/bin/env python
r"""박스별 내부 신호 채점(수정 22) — 롤아웃의 각 `\boxed` 후보에 대해 동결 ref 로 «약속 무게» w_ans 를 매긴다
(`mc.pool.pfx_commit_records` 가 이 점수로 PFX 앞부분을 자른다). 채점기(`score_trees`/`mc.trainer.tree_score`)와
후보 집합(`v5_candidates`)은 학습(`mc/train_hook.py`)이 쓰던 것과 같은 자다.

    python -m mc.shift_check --boxes [--prior] --pool_texts <pool>.texts.jsonl --model_path <ckpt> \
        --out_dir /hdd_data/seungpil/scratch/eval/box_base [--dry_run | --from_scores <scores.json>]
"""
from __future__ import annotations

import argparse
import bisect
import functools
import json
import math
import os
import re
import sys
import traceback
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mc import context as ctx
from mc.credit import near_miss_decoy
from mc.eval import load_jsonl
from mc.grade import answers_equivalent, boxed_spans, selftest
from mc.rollout import merged_model

ANSWER_HEAD = "\n\nFinal answer: \\boxed{"
V5_HEADS = (ANSWER_HEAD, "\n\nSo the final answer is \\boxed{", "\n\nTherefore, the answer is \\boxed{")  #: 수정 19 H=3
_NAN = float("nan")
W = "/hdd_data/seungpil/scratch/"
BOX_CAP = 8          #: 수정 22 박스 채점 — 롤아웃마다 앞 8개 박스까지
TREE_MAX = 16384     #: 트리 한 번의 토큰 상한
TREE_TFLOPS = 26.0   #: 실측 유효 처리량 — v1p_dense 22.6M 토큰 트리 189분(0924, H100·sdpa 임의 가림)
POOL_TEXTS = W + "data/mc_pool_learn_200_plain.texts.jsonl"


def _enc(tok, text: str) -> list[int]:
    return list(tok.encode(text or "", add_special_tokens=False))


def _common(a, b) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _pair_ids(tok, ctx: str, head: str, *cands: str):
    """(문맥, 대상) 쌍들 — 후보들을 **같은 토큰 자리**에서 채점하기 위해 자르는 자리 = 후보들의 공통 접두 길이 중
    가장 작은 것(문맥 토큰열이 바이트 동일). ★Qwen3: `-3`·`\\frac…` 은 `{` 가 답 첫 토큰에 합쳐져 따로 자르면
    자리가 어긋난다. `\\boxed` 와 `{` 사이는 pretoken 경계라 문맥은 한 번만 토큰화한다."""
    head, brace = _enc(tok, ctx + head[:-1]), _enc(tok, "{")
    tails = [_enc(tok, "{" + str(a or "")) for a in cands]
    k = min(_common(brace, t) for t in tails)
    c = head + tails[0][:k]
    return tuple((c, t[k:]) for t in tails)


def _fin(x) -> bool:
    return isinstance(x, (int, float)) and x == x and abs(float(x)) != float("inf")


def leak_point(cont: str, answers=()) -> int:
    r"""누설 = 첫 새 `\boxed` 또는 answers(L · 행 자신의 최종 답) 중 하나의 첫 **진술** 시작. 진술 = `= a`·`is a`·
    `equals a`·`answer … a`(같은 줄)·a 로 끝나는 줄(# 제목 제외). 앞 [\w.\-/^_{] · 뒤 [\w/^]·`.숫자` 가 붙으면 아니다
    («-1/3» 안의 1·3, «6.5» 안의 6). \frac{a}{b} ≡ a/b ≡ \dfrac{a}{b}. 없으면 끝.
    쓰는 곳: `train_hook.pfx_fork`(CH-Fork 가중 구간 = 누설 전) · 관문 G1 채점(`analysis/u_checks_0925/a38_g1_score.py`)."""
    w, hits = r"[ \t]*(?:\$\$?|\*\*|\\[()\[\]])?[ \t]*", [cont.find("\\boxed")]
    for a in (str(a) for a in answers if a):
        for v in filter(None, {a, re.sub(r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", a), a.replace("\\frac", "\\dfrac")}):
            x = rf"(?<![\w.\-/^_{{]){re.escape(v)}(?![\w/^]|\.\d)"
            pats = (rf"(?:=|\bequals|\bis){w}{x}", rf"(?i:answer)[^\n]{{0,60}}?{x}", rf"^(?![ \t]*#)[^\n]*?{x}{w}\.?[ \t]*$")
            hits += [m.start() for p in pats if (m := re.search(p, cont, re.M))]
    return min((h for h in hits if h >= 0), default=len(cont))


def _lse(xs) -> float:
    xs = [x for x in xs if x is not None]
    return (m := max(xs)) + math.log(sum(math.exp(x - m) for x in xs)) if xs else _NAN


_eq = functools.lru_cache(maxsize=None)(answers_equivalent)
_decoy = functools.lru_cache(maxsize=None)(lambda L, sd: near_miss_decoy(L, sd, checker=_eq))


def v1p_candidates(a0: str, labs: dict, sibs: list[str], surf: list[str]) -> tuple[dict, dict]:
    """후보 → 표기 형(자신 + surf 중 가장 흔한 동치 다른 표기) · 단계 → (L, 대안). 대안 = 가짜 답 2(`near_miss_decoy`
    중 A0·L 과 다른 것) + 형제 최종 답 중 L·A0 아닌 흔한 것 ≤2. 동치 후보는 한 이름(맞은 첫 답: L = A0)."""
    cand, st = {}, {}
    rep = functools.partial(_rep, cand, surf)
    rep(a0)
    for s, L in ((s, L) for s, L in labs.items() if L):
        ok, ds, ss = (lambda x, have: not any(_eq(x, y) for y in (L, a0, *have))), [], []  # noqa: E731
        for x in (_decoy(L, sd) for sd in range(17, 41)):
            ds += [x] if len(ds) < 2 and ok(x, ds) else []
        for x, _ in Counter(filter(None, sibs)).most_common():
            ss += [x] if len(ss) < 2 and ok(x, ss) else []
        st[s] = (rep(L), list(dict.fromkeys(rep(x) for x in ds + ss)))
    return cand, st


def _rep(cand: dict, surf: list[str], x: str) -> str:
    """x 의 후보 이름(동치 후보가 있으면 그것, 없으면 새로 — 표기 형 = 자신 + surf 중 가장 흔한 동치 다른 표기)."""
    if (k := next((k for k in cand if _eq(k, x)), None)) is None:
        k, cand[x] = x, [x] + [y for y, _ in Counter(y for y in surf if y != x and _eq(x, y)).most_common(1)]
    return k


def v5_candidates(a0: str, labs: dict, fins: list[str], surf: list[str]) -> tuple[dict, dict]:
    """수정 19 — 앞부분 묶음마다 **고정** 후보 집합 C = A0 ∪ 그 앞부분 모든 이어쓰기의 최종 답 ∪ L ∪ 가짜 답 2(단계마다,
    `v1p_candidates` 와 같은 가짜 답). LOO 형제 없음(형제 집합이 행 결과를 흘렸다: changed 87% 대 fix 15%).
    → (후보 → 표기 형, 단계 → (L, C 이름들))."""
    cand, st = v1p_candidates(a0, labs, [], surf)
    names = [_rep(cand, surf, x) for x in fins if x]
    return cand, {s: (L, list(dict.fromkeys([L, _rep(cand, surf, a0), *ds, *names])))
                  for s, (L, ds) in st.items()}


def score_trees(model, trees, max_tok: int = TREE_MAX, log_every: int = 200, dev=None, per_token: bool = False) -> list:
    """트리들 → 이어 붙인 블록 점수(`mc.trainer.tree_score`, 한 forward 의 토큰이 max_tok 을 넘거나 OOM 이면 블록을 나눈다).
    `dev` = 계산 장치(학습 ref 워커는 매개변수 오프로드라 cuda 를 준다 — None 이면 모델 매개변수 장치)."""
    import time  # noqa: PLC0415

    import torch  # noqa: PLC0415

    from mc.trainer import tree_score  # noqa: PLC0415
    out, t0 = [], time.time()

    def run(pre, ch):                          # OOM 이면 블록을 반으로 나눠 다시(블록 하나로도 넘치면 그대로 올린다)
        try:
            return tree_score(model, pre, ch, dev, per_token=True) if per_token else tree_score(model, pre, ch, dev)
        except torch.cuda.OutOfMemoryError:
            if len(ch) < 2 and per_token:    # 수정 28b: 토큰별(CH-Fork) 한 블록 OOM → None(호출자가 균등 가중·계수)
                torch.cuda.empty_cache()
                return [None]
            if len(ch) < 2:
                raise
            torch.cuda.empty_cache()
            return run(pre, ch[:len(ch) // 2]) + run(pre, ch[len(ch) // 2:])
    with torch.no_grad():
        for j, (pre, blocks) in enumerate(trees):
            ch, n = [], len(pre)
            for b in blocks + [None]:
                if ch and (b is None or n + len(b[1]) > max_tok):
                    out += run(pre, ch)
                    ch, n = [], len(pre)
                if b is not None:
                    ch, n = ch + [b], n + len(b[1])
            if (j + 1) % log_every == 0:
                el = time.time() - t0
                print(f"[MC][V1P] 트리 {j + 1}/{len(trees)} · {el / 60:.1f}분 · 남은 ≈{el / (j + 1) * (len(trees) - j - 1) / 60:.1f}분",
                      flush=True)
    return out


def tree_minutes(n_tok) -> float:
    """트리 토큰 길이들 → 추정 GPU 분(FLOP = 8e9·S + 5.9e5·S², 실측 `TREE_TFLOPS`)."""
    return sum(8e9 * x + 5.9e5 * x * x for x in n_tok) / (TREE_TFLOPS * 1e12) / 60


def box_requests(tok, recs: list[dict], heads=V5_HEADS, cap: int = BOX_CAP,
                 prior: bool = False) -> tuple[list, dict, list]:
    r"""수정 22 (A) — 롤아웃마다 트리 하나: 박스 k(앞 cap 개)의 끝 토큰 자리(`mc.pool.first_box_cut` 처럼 닫는 괄호를 담은
    토큰 끝) × 머리 × 표기. 박스 k 의 후보 = `v5_candidates(박스 k 답, {"gold": gold}, 같은 문제 모든 롤아웃 최종 답)`.
    prior(수정 27a T1) = 같은 후보·머리를 **응답 시작 자리**(문제만, 자기 풀이 없음)에서 → w_ans = 사전 믿음 π₀.
    → (트리, meta[(i, k, 표기, 머리)] = 점수 번호, 항목[i] = {rec, boxes: [(k, 자름 글자, 답, cand, C)]})."""
    fins: dict = {}
    for r in recs:
        fins.setdefault(str(r["uid"]), []).append(str(r.get("final_answer") or ""))
    trees, meta, items = [], {}, []
    for r in recs:
        sp = boxed_spans(r["text"])[:cap]
        if not sp:
            continue
        enc = tok(r["text"], add_special_tokens=False, return_offsets_mapping=True)
        ends = [e for _, e in enc["offset_mapping"]]
        head, fs, gold = _enc(tok, ctx.turn1_prompt(tok, r["problem"], "plain")), fins[str(r["uid"])], str(r["gold"])
        boxes = []
        for k, (a0, _, e) in enumerate(sp):
            t = next((j + 1 for j, (b, x) in enumerate(enc["offset_mapping"]) if b < e <= x), bisect.bisect_left(ends, e) + 1)
            cand, st = v5_candidates(a0, {"gold": gold}, fs, [a0, *fs, gold])
            boxes.append((k, ends[t - 1], 0 if prior else t, a0, cand, st["gold"][1]))
        forms = list(dict.fromkeys(f for *_, cand, _ in boxes for fs_ in cand.values() for f in fs_))
        pairs = [_pair_ids(tok, "", h, *(f + "}" for f in forms)) for h in heads]
        i, blocks = len(items), []
        for k, _, t, *_ in boxes:
            for h, pr in enumerate(pairs):
                for f, (c, x) in zip(forms, pr):
                    meta[(i, k, f, h)] = len(meta)
                    blocks.append((len(head) + t, list(c) + list(x), len(x)))
        trees.append((head + list(enc["input_ids"][:max(b[2] for b in boxes)]), blocks))
        items.append({"rec": r, "boxes": [(k, cut, a0, cand, C) for k, cut, _, a0, cand, C in boxes]})
    return trees, meta, items


def box_rows(items: list[dict], scores: list, meta: dict, heads: int = len(V5_HEADS)) -> list[dict]:
    """점수 → 롤아웃마다 박스별 {k, cut(글자), answer, w_ans, commit_weight} (`w_ans` 는 박스 k 자리, 후보 = 그 박스의 C)."""
    out = []
    for i, it in enumerate(items):
        r, bx = it["rec"], []
        for k, cut, a0, cand, C in it["boxes"]:
            ell = [{0: {c: _lse([scores[meta[(i, k, f, h)]] for f in fs]) for c, fs in cand.items()}} for h in range(heads)]
            w = w_ans(ell, a0, C)
            bx.append({"k": k, "cut": cut, "answer": a0, "w_ans": w, "commit_weight": commit_weight(w)})
        out.append({**{x: r.get(x) for x in ("uid", "problem_idx", "gold", "selected")}, "row": i, "boxes": bx})
    return out


def w_ans(ell: list[dict], a0: str, C: list[str], o=0) -> float:
    r"""수정 22 (i) 내부 신호 — 자리 o(기본 t0 = 첫 박스 직후)에서 A0 의 후보 집합 몫 mean_h exp(ℓ_o(a0) − lse_C ℓ_o) ∈ [0,1]
    (유한한 머리만 평균, 없으면 NaN)."""
    xs = [x for e in ell if _fin(x := e[o][a0] - _lse([e[o][c] for c in C]))]
    return sum(math.exp(x) for x in xs) / len(xs) if xs else _NAN


#: 수정 22 가중치 보정 — 손 라벨 126(약속 56·중간 70)에 logit(w_ans) 로지스틱 적합(0925, AUC .872[.792,.946]).
#: w_ans 는 거의 1 에 붙어(중앙값 .99998 대 .9999999) 그대로는 가중치가 못 된다 → P(약속 | w_ans).
COMMIT_A, COMMIT_B = -3.2009, 0.17339


def commit_weight(w: float, a: float = COMMIT_A, b: float = COMMIT_B) -> float:
    """약속한 첫 답 가중치 = σ(a + b·logit(w_ans))(w 는 [1e-15, 1−1e-15] 로 자른다; NaN 이면 NaN)."""
    if not _fin(w):
        return _NAN
    w = min(max(float(w), 1e-15), 1 - 1e-15)
    return 1 / (1 + math.exp(-(a + b * math.log(w / (1 - w)))))


def _dump(x) -> str:
    return json.dumps(x, ensure_ascii=False, indent=2, default=float)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    for k in ("--boxes", "--prior", "--only_selected", "--only_unselected", "--dry_run"):
        ap.add_argument(k, action="store_true")
    for k, t, d in (("--out_dir", str, W + "eval/box_base"), ("--model_path", str, W + "models/Qwen3-4B-Instruct-2507"),
                    ("--pool_texts", str, POOL_TEXTS), ("--from_scores", str, "")):
        ap.add_argument(k, type=t, default=d)                # --from_scores = 저장된 scores.json 으로 다시(CPU)
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    from transformers import AutoTokenizer  # noqa: PLC0415
    selftest()
    recs = [r for r in load_jsonl(a.pool_texts)                       # --only_(un)selected = 학습 풀 / 탐침 쪽만
            if not (a.only_selected and not r.get("selected")) and not (a.only_unselected and r.get("selected"))]
    trees, meta, items = box_requests(AutoTokenizer.from_pretrained(a.model_path), recs, prior=a.prior)
    n_tok = [len(pre) + sum(len(b[1]) for b in bl) for pre, bl in trees]
    plan = {"mode": "boxes_prior" if a.prior else "boxes", "pool_texts": a.pool_texts, "n_rollouts": len(recs), "n_trees": len(trees),
            "n_boxes": sum(len(it["boxes"]) for it in items), "n_scores": len(meta), "tree_tokens": sum(n_tok),
            "tree_tokens_max": max(n_tok, default=0), "est_gpu_min": tree_minutes(n_tok)}
    return _run(a, out, plan, trees, lambda sc: _write_rows(out / "box_scores.jsonl", box_rows(items, sc, meta)))


def _run(a, out: Path, plan: dict, trees: list, finish) -> int:
    """plan.json → (dry_run 이면 끝) → 점수(`--from_scores` 또는 GPU `score_trees`) → scores.json 을 **먼저** 쓰고 finish.
    모델 적재 뒤 예외 = 즉시 비0 종료(잡이 GPU 를 붙잡지 않게)."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(_dump(plan))
    print(_dump(plan), flush=True)
    if a.dry_run:
        return 0
    if a.from_scores:
        finish(json.loads(Path(a.from_scores).read_text()))
        return 0
    try:
        import torch  # noqa: PLC0415
        from transformers import AutoModelForCausalLM  # noqa: PLC0415
        model = AutoModelForCausalLM.from_pretrained(merged_model(a.model_path), dtype=torch.bfloat16,
                                                     attn_implementation="sdpa").cuda().eval()
        scores = score_trees(model, trees)
        (out / "scores.json").write_text(json.dumps(scores))   # 요약보다 먼저 — 요약 버그는 --from_scores 로
        finish(scores)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush(), sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush(), sys.stderr.flush()
    os._exit(0)


def _write_rows(path: Path, rows: list[dict]) -> list[dict]:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return rows


if __name__ == "__main__":
    sys.exit(main())
