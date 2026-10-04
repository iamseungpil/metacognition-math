#!/usr/bin/env python
r"""학습 풀 선별 — 자발 수정이 **일어나는 자리**만 남긴다. gold 는 선별에 쓰지 않는다.

`--select state`(기본) = 합의 상태로 ALL_SAME·NOANS 제외 · `learnability`(0923) = p(1−p) 상위 keep_n,
p = 사건(답이 바뀜 ∧ 첫 답이 **K 롤아웃 LOO 다수결**로 오답) 비율. gold 정오는 저장만(보고용).
산출: parquet(`prompt` 는 PROMPT_VARIANT 로 굽는다) · `.texts.jsonl`(전 롤아웃) · `.prefixes.jsonl`
(**선별 안 된** 문제의 첫 박스 닫힘까지 앞부분 → mc/probe.py) · `.summary.json`.

    PROMPT_VARIANT=plain python -m mc.pool --dataset parquet:<train> --model_path <ckpt> \
        --k 8 --max_tokens 8192 --select learnability --keep_n 200 --out $WORK/data/pool.parquet

`--pfx_from <pool>.texts.jsonl`(SPONT_PFX, GPU 없음) = **선별된** 문제의 앞부분 + LOO 라벨 parquet
(`pfx_records`) — `--probe_selection`·`--val` 과 문제가 겹치면 즉사. `--model_path` = 토큰 경계·길이용 토크나이저.
`--commit_scores <box_scores.jsonl>`(수정 22, `mc.shift_check --boxes` 산출) = 약속 무게 ≥ .5 인 첫 박스에서 자르고 gold 라벨·
유효 반반 `weight`(`pfx_commit_records`). `--pfx_screen <pfx.parquet>` = 학습 가능 앞부분 거르기(`pfx_screen`: 선택 json → `mc.probe` K 이어쓰기 → 0<p<1 만).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mc import context as ctx
from mc.credit import label_correct, majority_label
from mc.grade import boxed_spans, grade_answer, grade_math, parse_options, revision_zone, strip_choice_marker

_BARE_CHOICE = re.compile(r"(?:\\text(?:bf)?\s*\{\s*)?\(?\s*[A-E]\s*\)?\s*\}?")
#: state 선별이 남기는 상태 — ALL_SAME(수정률 2.05%)과 NOANS(예산 절단 인공물, 구간 불성립)를 뺀다.
KEEP_STATES = ("DOMINANT", "SPLIT", "SCATTER")


def select(rows_in, texts, k: int) -> tuple[list[int], dict]:
    """문제별 상태 + 남길 인덱스 + 상태별 base 자발 수정률(첫 박스 ≠ 마지막 박스 행 비율)."""
    keep, states, tab = [], [], {}
    for p, _src in enumerate(rows_in):
        cell = texts[p * k:(p + 1) * k]
        st = str(ctx.agreement_state([t["ans"] for t in cell], k=k)["state"])
        states.append(st)
        c = tab.setdefault(st, {"problems": 0, "rows": 0, "revised": 0, "correct": 0})
        c["problems"] += 1
        for t in cell:
            c["rows"] += 1
            c["revised"] += int(t["revised"])
            c["correct"] += int(t["correct"])
        if st in KEEP_STATES:
            keep.append(p)
    for c in tab.values():
        c["revise_rate"] = c["revised"] / max(1, c["rows"])
        c["acc"] = c["correct"] / max(1, c["rows"])
    return keep, {"by_state": tab, "n_problems": len(rows_in), "n_kept": len(keep),
                  "n_excluded_noans": sum(1 for x in states if x == "NOANS"),
                  "n_excluded_all_same": sum(1 for x in states if x == "ALL_SAME"),
                  "kept_frac": len(keep) / max(1, len(rows_in)),
                  "keep_states": list(KEEP_STATES), "states": states}


def rollout_records(src: dict, p: int, texts: list[str], variant: str, tok=None) -> list[dict]:
    """한 문제의 K 롤아웃 → 행 기록. `event` = 답이 바뀜 ∧ 첫 답이 **LOO 형제 다수결**로 오답
    (학습의 LABEL=majority 와 같은 자) · gold 정오는 `first_correct_gold`/`correct` 로 저장만."""
    gold, finals = str(src["gold"]), [ctx.turn1_answer(t) for t in texts]
    uid = hashlib.md5(str(src["problem"]).encode()).hexdigest()[:12]
    out = []
    for j, t in enumerate(texts):
        z, sp = revision_zone(t), boxed_spans(t)
        first = sp[0][0] if sp else ""
        fm = label_correct(first, label="majority", sib_answers=finals, self_idx=j)
        rev = bool(z and z["revised"])
        out.append({"problem_idx": p, "uid": uid, "problem": str(src["problem"]), "gold": gold,
                    "prompt_variant": variant, "text": t, "first_answer": first,
                    "final_answer": finals[j], "ans": finals[j], "revised": rev,
                    "first_correct_gold": bool(first) and bool(grade_answer(first, gold)),
                    "first_correct_major": fm, "event": rev and fm == 0.0,
                    "correct": bool(grade_math(t, gold)), "prefix_end": first_box_cut(t, tok)})
    return out


def first_box_cut(text: str, tok=None) -> int | None:
    r"""첫 `\boxed{…}` 닫는 괄호 바로 뒤 위치(없으면 None) — 탐침·PFX 앞부분을 자르는 유일한 자리. `tok` 을 주면
    그 괄호를 담은 **토큰의 끝**으로 민다: 정책은 `}\n`·`}}\n`·`}$` 를 한 토큰으로 내므로(0923 검토 62%) 맨 `}` 에서
    끊으면 정책이 스스로는 서지 않는 경계에서 «다시 볼까» 를 배우게 된다."""
    sp = boxed_spans(text)
    if not sp or tok is None:
        return sp[0][2] if sp else None
    offs = tok(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
    return next((e for b, e in offs if b < sp[0][2] <= e), sp[0][2])


def prefix_records(recs: list[dict], selected: set) -> list[dict]:
    """선별 **안 된** 문제의 롤아웃 → «첫 `\\boxed{…}` 닫는 괄호까지» 앞부분(탐침 입력)."""
    return [{"uid": r["uid"], "problem_idx": r["problem_idx"], "problem": r["problem"],
             "gold": r["gold"], "prefix": r["text"][:r["prefix_end"]],
             "first_correct": r["first_correct_gold"]}             # gold = 측정 전용
            for r in recs if r["problem_idx"] not in selected and r["prefix_end"] is not None]


def is_choice(a) -> bool:
    r"""객관식 답(`\text{B}`·`(B)`·`\text{(B)}\ …`) — 맨 글자도 포함(`strip_choice_marker` 는 일부러 맨 글자를 뺀다)."""
    return strip_choice_marker(a) is not None or bool(_BARE_CHOICE.fullmatch(str(a or "").strip()))


def pfx_records(texts: list[dict], seed: int = 11, tok=None) -> tuple[list[dict], dict]:
    r"""SPONT_PFX 시작 상태 — **선별된** 문제의 롤아웃을 `first_box_cut` 에서 자른 앞부분 + 라벨 = 나머지
    K−1 롤아웃 최종 답의 LOO 다수결(`majority_label`, None 이면 버림). 첫 답 오답(라벨 기준)은 전부,
    정답은 같은 수만큼 시드 표집(모자라면 전부 — «맞은 답을 깨지 않기»도 배운다). gold 는 쓰지 않는다."""
    cells: dict[int, list[dict]] = {}
    for r in texts:
        if r["selected"]:
            cells.setdefault(int(r["problem_idx"]), []).append(r)
    wrong, right, nolabel, tied, choice = [], [], 0, 0, 0
    for p, cell in sorted(cells.items()):
        finals = [str(r["final_answer"] or "") for r in cell]
        if majority_label(finals) is None:        # K 전체 동률 → LOO 가 늘 «반대편» 답을 준다(0923 검토: 57행 모순 라벨)
            tied += 1
            continue
        if any(is_choice(r["first_answer"]) or is_choice(r["final_answer"]) for r in cell):
            choice += 1                            # 값 대 선택지 글자 표기 차가 «고침» 보상이 된다(0923 스모크: 67행)
            continue
        for j, r in enumerate(cell):
            cut = first_box_cut(r["text"], tok)
            label = majority_label(finals, self_idx=j) if cut is not None else None
            nolabel += int(cut is not None and label is None)
            if label is None:
                continue
            rec = {"problem_uid": r["uid"], "problem_idx": p, "rollout": j, "problem": r["problem"],
                   "pfx_id": f"{r['uid']}:{j}",
                   "prefix": r["text"][:cut], "first_answer": r["first_answer"], "label": label,
                   "first_wrong": not grade_answer(r["first_answer"], label)}
            (wrong if rec["first_wrong"] else right).append(rec)
    kept = right if len(right) <= len(wrong) else random.Random(seed).sample(right, len(wrong))
    out = sorted(wrong + kept, key=lambda r: (r["problem_idx"], r["rollout"]))
    return out, {"n_problems": len(cells), "n_rollouts": sum(map(len, cells.values())),
                 "n_tied_problems": tied, "n_choice_problems": choice, "n_nolabel": nolabel, "n_wrong": len(wrong), "n_right_pool": len(right),
                 "n_right": len(kept), "n_rows": len(out), "seed": seed}


def pfx_commit_records(texts: list[dict], box_rows: list[dict], thr: float = 0.5, veto: bool = True,
                       selected: bool = True) -> tuple[list[dict], dict]:
    r"""수정 22·23a PFX — 롤아웃(`selected` 쪽)을 약속 무게(`shift_check.box_rows` 의 commit_weight) ≥ thr ∧ ¬글자 중간 판정
    (`mc.probe.intermediate_box`, `veto`) 인 **첫** 박스의 토큰 끝에서 자른다(없으면 앞부분 없음). 라벨 = gold · 첫 답 정오 =
    선택지 대응 채점(`grade_answer`+`parse_options`) · 가지치기 없이 둘 다 남기고 `weight` = N/(2·부류 수)(유효 반반).
    box_rows 는 롤아웃 순서(박스 있는 것만)로 맞춘다. 요약에 거부 없는 판의 수도 싣는다."""
    from mc.probe import intermediate_box  # noqa: PLC0415
    boxes: dict[str, list[dict]] = {}
    for b in box_rows:
        if bool(b.get("selected")) == selected:
            boxes.setdefault(str(b["uid"]), []).append(b)
    recs, seen, none_, n_noveto = [], {}, 0, 0
    for r in (t for t in texts if bool(t["selected"]) == selected):
        j = seen[str(r["uid"])] = seen.get(str(r["uid"]), -1) + 1
        if not boxed_spans(r["text"]):
            continue
        b = boxes[str(r["uid"])].pop(0)
        if b["boxes"][0]["answer"] != boxed_spans(r["text"])[0][0]:
            raise SystemExit(f"[MC][PFX] 박스 점수 순서가 롤아웃과 어긋난다: {r['uid']}:{j}")
        sp = boxed_spans(r["text"])
        ok = [x for x in b["boxes"] if (x["commit_weight"] or 0) >= thr]
        n_noveto += bool(ok)
        k = next((x for x in ok if not (veto and intermediate_box(r["text"], sp[x["k"]][1], sp[x["k"]][2]))), None)
        if k is None:
            none_ += 1
            continue
        opt, gold = parse_options(r["problem"]), str(r["gold"])
        recs.append({"problem_uid": r["uid"], "problem_idx": int(r["problem_idx"]), "rollout": j, "problem": r["problem"],
                     "pfx_id": f"{r['uid']}:{j}:{k['k']}", "prefix": r["text"][:k["cut"]], "first_answer": k["answer"],
                     "label": gold, "gold": gold, "box_k": k["k"], "commit_weight": k["commit_weight"],
                     "first_wrong": not grade_answer(k["answer"], gold, opt), "choice": opt is not None})
    n_w = sum(r["first_wrong"] for r in recs)
    for r in recs:
        r["weight"] = len(recs) / (2 * (n_w if r["first_wrong"] else len(recs) - n_w))
    ws = [r["weight"] for r in recs]
    return recs, {"n_rollouts": sum(bool(t["selected"]) == selected for t in texts), "n_no_box": sum(
        bool(t["selected"]) == selected and not boxed_spans(t["text"]) for t in texts), "n_no_committed_box": none_,
        "veto": veto, "n_rows_without_veto": n_noveto, "n_rows": len(recs),
        "n_wrong": n_w, "n_right": len(recs) - n_w, "n_choice": sum(r["choice"] for r in recs),
        "box_k_hist": {k: sum(r["box_k"] == k for r in recs) for k in sorted({r["box_k"] for r in recs})},
        "weight": {"wrong": next((r["weight"] for r in recs if r["first_wrong"]), None),
                   "right": next((r["weight"] for r in recs if not r["first_wrong"]), None), "sum": sum(ws)},
        "thr": thr, "label": "gold"}


def commit_probe_selection(a) -> int:
    """`--probe_out` — 선별 **안 된** 문제 롤아웃을 수정 23a 규칙으로 자른 앞부분에서 탐침 선택(`mc.probe.select`: 틀림
    `--n_wrong`·맞음 `--n_right`, 문제 돌림 층화, 시드) → `mc.probe --selection` 형식 json(`committed` = 1)."""
    from mc.probe import select as probe_select  # noqa: PLC0415
    recs, summ = pfx_commit_records([json.loads(x) for x in open(a.pfx_from) if x.strip()],
                                    [json.loads(x) for x in open(a.commit_scores) if x.strip()], veto=not a.no_veto,
                                    selected=False)
    sel = probe_select([{"uid": r["problem_uid"], "problem_idx": r["problem_idx"], "problem": r["problem"], "gold": r["gold"],
                         "prefix": r["prefix"], "first_correct": not r["first_wrong"], "committed": 1.0,
                         "pfx_id": r["pfx_id"]} for r in recs], a.n_wrong, a.n_right, a.seed)
    Path(a.probe_out).write_text(json.dumps(sel, ensure_ascii=False))
    summ.update(n_selected_wrong=sum(not e["first_correct"] for e in sel), n_selected_right=sum(bool(e["first_correct"]) for e in sel),
                n_problems=len({e["uid"] for e in sel}), out=a.probe_out)
    Path(a.probe_out).with_suffix(".summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2))
    return print(f"[MC][PROBE-SEL] {summ}", flush=True) or 0


def build_pfx(a) -> int:
    """`--pfx_from` 본체 — 원 풀 parquet(`<base>.parquet`) 행에 extra_info(앞부분·라벨)를 얹어 쓴다."""
    import pandas as pd
    from transformers import AutoTokenizer

    from mc.trainer import check_prompt_lengths
    tok = AutoTokenizer.from_pretrained(a.model_path)
    texts = [json.loads(x) for x in open(a.pfx_from) if x.strip()]
    recs, summ = (pfx_commit_records(texts, [json.loads(x) for x in open(a.commit_scores) if x.strip()], veto=not a.no_veto)
                  if a.commit_scores else pfx_records(texts, a.seed, tok))
    probe = {str(r["uid"]) for r in json.loads(Path(a.probe_selection).read_text())}
    val = set(pd.read_parquet(a.val)["problem"].astype(str))
    for name, bad in (("탐침 선택", {r["problem_uid"] for r in recs} & probe),
                      ("val", {r["problem"] for r in recs} & val)):
        if bad:
            raise SystemExit(f"[MC][PFX] 학습 앞부분이 {name} 과 {len(bad)} 문제 겹친다 — 분리 위반.")
    pool = pd.read_parquet(a.pfx_from.removesuffix(".texts.jsonl") + ".parquet")
    src = {str(r["problem"]): r for r in pool.to_dict("records")}
    keys = ("pfx_id", "problem_uid", "problem_idx", "prefix", "first_answer", "label", "first_wrong") + (
        ("gold", "weight", "box_k", "commit_weight") if a.commit_scores else ())
    pd.DataFrame([{**src[r["problem"]], "extra_info": {**src[r["problem"]]["extra_info"],
                                                       **{k: r[k] for k in keys}}}
                  for r in recs]).to_parquet(a.out, index=False)
    summ.update(pfx_from=a.pfx_from, out=a.out, disjoint_from=[a.probe_selection, a.val],
                label="gold (commit-cut, 수정 22)" if a.commit_scores else
                "LOO majority of the other K-1 rollouts' final answers (gold unused)",
                n_first_wrong_gold=sum(not grade_answer(r["first_answer"], src[r["problem"]]["gold"])
                                       for r in recs))                       # 감시 전용
    summ["prompt_tokens"] = check_prompt_lengths(tok, a.out, 0)          # 앞부분 포함 → MAX_PROMPT
    enc = lambda x: tok.encode(x, add_special_tokens=False)  # noqa: E731
    summ["n_prefix_token_split_mismatch"] = sum(       # 에이전트 루프(템플릿 ids + 앞부분 ids) ≡ 이어 붙인 문자열
        enc(ctx.turn1_prompt(tok, r["problem"], "plain")) + enc(r["prefix"])
        != enc(ctx.turn1_prompt(tok, r["problem"], "plain") + r["prefix"]) for r in recs)
    Path(a.out).with_suffix(".summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2))
    print(f"[MC][PFX] {summ}", flush=True)
    return 0


def pfx_screen(a) -> int:
    r"""`--pfx_screen <pfx.parquet>` — 학습 가능 앞부분 거르기(0923: PFX 10스텝 그룹의 ~85% 가 r 상수라 기울기 0).
    `--screen_texts` 없으면 탐침 형식 선택 json 을 `--out` 에 쓴다(gold 칸 = **라벨** → `mc.probe` 가 K 이어쓰기를 라벨로
    채점, gold 미사용). 있으면 그 texts.jsonl 에서 앞부분별 p = 라벨 일치율을 세어 0<p<1 행만 `--out` parquet 에 남긴다."""
    import pandas as pd
    df = pd.read_parquet(a.pfx_screen)
    if not a.screen_texts:
        sel = [{"pid": i, "uid": e["problem_uid"], "problem": q, "prefix": e["prefix"], "gold": e["label"],
                "first_correct": not e["first_wrong"]} for i, (q, e) in enumerate(zip(df["problem"], df["extra_info"]))]
        Path(a.out).write_text(json.dumps(sel, ensure_ascii=False))
        return print(f"[MC][PFX] 거르기 선택 {len(sel)} → {a.out}") or 0
    by: dict[int, list[float]] = {}
    for x in open(a.screen_texts):
        r = json.loads(x)
        by.setdefault(int(r["pid"]), []).append(float(r["final_correct"]))
    keep = [i for i in range(len(df)) if 0 < sum(by.get(i, [0])) < len(by.get(i, [0]))]
    out = df.iloc[keep].reset_index(drop=True)
    out.to_parquet(a.out, index=False)
    fw = out["extra_info"].map(lambda e: bool(e["first_wrong"]))
    summ = {"from": a.pfx_screen, "screen_texts": a.screen_texts, "n_in": len(df), "n_screened": len(by),
            "n_keep": len(out), "n_keep_wrong": int(fw.sum()), "n_keep_right": int((~fw).sum())}
    Path(a.out).with_suffix(".summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2))
    return print(f"[MC][PFX] {summ}", flush=True) or 0


def learnability(rows_in, texts, k: int) -> list[float]:
    """문제별 **학습 가능성** p(1−p) — p = K 롤아웃 중 `event`(첫 답이 LOO 다수결로 오답 ∧ 답이
    바뀜, `rollout_records`) 비율.

    왜 ALL_SAME 제외가 아니라 이것인가(0923): 사건이 전부/전혀면 그룹 안 대비가 없어 어떤
    크레딧도 기울기를 못 만든다(합의 상태는 사건이 아니라 답의 흩어짐을 봤다)."""
    cells = [texts[p * k:(p + 1) * k] for p in range(len(rows_in))]
    ps = [(sum(1 for t in c if t["event"]) / len(c)) if c else 0.0 for c in cells]
    return [q * (1.0 - q) for q in ps]


def filter_pass_rate(df, lo, hi):
    """group_pass_rate 포함 구간 필터(중간 난이도 풀용) — lo/hi 둘 다 None 이면 무변화."""
    if lo is None and hi is None:
        return df
    def ok(e):
        pr = (e or {}).get("group_pass_rate")
        return (lo is None or pr >= lo) and (hi is None or pr <= hi)
    return df[[ok(e) for e in df["extra_info"]]].reset_index(drop=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="parquet:<path>[:<level>]")
    ap.add_argument("--model_path")
    for k in ("--pfx_from", "--probe_selection", "--val", "--pfx_screen", "--screen_texts", "--commit_scores"):  # SPONT_PFX
        ap.add_argument(k)
    ap.add_argument("--no_veto", action="store_true")         # --commit_scores: 글자 중간 거부 끄기(보고용)
    ap.add_argument("--probe_out")                             # --commit_scores: 탐침 선택 json(선별 안 된 문제)
    for k, d in (("--n_wrong", 150), ("--n_right", 200)):
        ap.add_argument(k, type=int, default=d)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--gpu_util", type=float, default=0.80)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--keep_n", type=int, default=0,
                    help="선별 뒤 상태별 층화 표집으로 문제 수를 이만큼으로 자른다(파일럿 200)")
    for k in ("--pass_lo", "--pass_hi"):      # group_pass_rate 포함 구간(keep_n 전), 기본 필터 없음
        ap.add_argument(k, type=float, default=None)
    ap.add_argument("--select", choices=("state", "learnability", "all"), default="state",
                    help="state=합의 상태(기본) · learnability=사건 분산 p(1−p) 상위 keep_n · all=전부(수정 29 새 앞부분: 같은 문제)")
    ap.add_argument("--out", required=True, help="선별된 parquet 경로")
    a = ap.parse_args(argv)
    if a.pfx_screen:
        return pfx_screen(a)
    if a.pfx_from and a.probe_out:
        return commit_probe_selection(a)
    if a.pfx_from:
        if not (a.probe_selection and a.val and a.model_path):
            raise SystemExit("[MC][PFX] --probe_selection·--val(분리 확인)·--model_path(토큰 경계)가 필요하다.")
        return build_pfx(a)
    if not (a.model_path and (a.dataset or "").startswith("parquet:")):
        raise SystemExit("[MC] --model_path 와 --dataset parquet:<path> 가 필요하다(원 컬럼을 그대로 싣는다).")

    import pandas as pd
    from vllm import SamplingParams

    from mc.grade import selftest
    from mc.rollout import build_engine
    selftest()
    parts = a.dataset.split(":", 2)
    df = pd.read_parquet(parts[1])
    if len(parts) > 2:
        df = df[[(e or {}).get("level") == parts[2] for e in df["extra_info"]]].reset_index(drop=True)
    if a.limit:
        df = df.head(a.limit)
    rows_in = df.to_dict("records")
    print(f"[MC][POOL] {parts[1]}: {len(rows_in)} 문제 × K={a.k}", flush=True)

    llm, tok = build_engine(a.model_path, max_tokens=a.max_tokens, gpu_util=a.gpu_util,
                            seed=a.seed)
    prompts = [ctx.turn1_prompt(tok, str(r["problem"])) for r in rows_in]
    # ★프롬프트 길이 기록 — MAX_PROMPT 를 이 값으로 정해야 verl 이 긴 문제를 조용히 안 버린다(0921).
    plens = sorted(len(tok.encode(x, add_special_tokens=False)) for x in prompts)
    prompt_stat = {"max": plens[-1], "p99": plens[min(len(plens) - 1, int(0.99 * len(plens)))],
                   "p50": plens[len(plens) // 2], "n": len(plens)}
    print(f"[MC][POOL] 프롬프트 토큰 {prompt_stat}", flush=True)
    outs = llm.generate(prompts, SamplingParams(n=a.k, temperature=1.0, top_p=1.0,
                                                max_tokens=a.max_tokens, seed=a.seed))
    variant = ctx.default_variant()
    texts = [r for p, (src, o) in enumerate(zip(rows_in, outs))
             for r in rollout_records(src, p, [c.text for c in o.outputs], variant, tok)]

    keep, summ = select(rows_in, texts, a.k)
    if a.select == "all":                                # 수정 29: 이미 고른 학습 풀을 다른 정책으로 다시 풂 — 문제 그대로
        keep = list(range(len(rows_in)))
        summ.update(select_mode="all", n_kept=len(keep), kept_frac=1.0)
    if a.select == "learnability":                       # 상위 keep_n 만 남긴다(상태 무시)
        sc = learnability(rows_in, texts, a.k)
        keep = sorted(sorted(range(len(rows_in)), key=lambda p: (-sc[p], p))[
            :(a.keep_n or len(rows_in))])
        summ.update(select_mode="learnability", n_kept=len(keep),
                    kept_frac=len(keep) / max(1, len(rows_in)),
                    learn_score_mean=sum(sc) / max(1, len(sc)),
                    learn_score_min_kept=min((sc[p] for p in keep), default=0.0))
    out_df = df.iloc[keep].copy()
    out_df["state"] = [summ["states"][p] for p in keep]      # 원 컬럼은 그대로 + state 하나
    out_df["_p"] = keep                                      # 최종 선별 추적용(쓰기 전에 뺀다)
    out_df = filter_pass_rate(out_df, a.pass_lo, a.pass_hi)   # keep_n 이전에 적용(중간 난이도)
    if a.pass_lo is not None or a.pass_hi is not None:
        prs = pd.Series([(e or {}).get("group_pass_rate") for e in out_df["extra_info"]])
        summ["pass_filter"] = {
            "lo": a.pass_lo, "hi": a.pass_hi, "n": len(out_df),
            "pass_rate_mean": float(prs.mean()), "pass_rate_sd": float(prs.std()),
            "by_agree_state": out_df["state"].value_counts().to_dict(),
            "by_level": pd.Series([(e or {}).get("level") for e in out_df["extra_info"]])
                        .value_counts().to_dict()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if a.select == "state" and a.keep_n and len(out_df) > a.keep_n:   # 파일럿 규모(0921 승인: 200문제)
        frac = a.keep_n / len(out_df)
        out_df = (out_df.groupby("state", group_keys=False)
                  .apply(lambda g: g.sample(frac=frac, random_state=0))
                  .head(a.keep_n).reset_index(drop=True))
        summ["n_kept"] = len(out_df)
    chosen = {int(x) for x in out_df.pop("_p")}
    out_df["prompt"] = [ctx.build_math_prompt(str(q), variant) for q in out_df["problem"]]
    out_df.to_parquet(a.out, index=False)
    base = str(Path(a.out).with_suffix(""))
    with open(base + ".texts.jsonl", "w") as fh:
        fh.writelines(json.dumps({**{k: v for k, v in r.items() if k not in ("ans", "prefix_end")},
                                  "selected": r["problem_idx"] in chosen}, ensure_ascii=False) + "\n"
                      for r in texts)
    pre = prefix_records(texts, chosen)
    with open(base + ".prefixes.jsonl", "w") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in pre)
    summ["prefixes"] = {"n": len(pre), "wrong_first": sum(not r["first_correct"] for r in pre),
                        "right_first": sum(bool(r["first_correct"]) for r in pre)}
    summ.update({"dataset": a.dataset, "model_path": a.model_path, "k": a.k, "out": a.out,
                 "select": a.select, "prompt_variant": variant,
                 "event": "revised ∧ first answer wrong by LOO sibling majority (gold unused)",
                 "prompt_tokens": prompt_stat})
    summ.pop("states")
    Path(a.out).with_suffix(".summary.json").write_text(
        json.dumps(summ, ensure_ascii=False, indent=2, default=float))
    for st, c in sorted(summ["by_state"].items()):
        print(f"  {st:9s} problems={c['problems']:5d} revise_rate={c['revise_rate']:.4f} "
              f"acc={c['acc']:.4f}")
    print(f"[MC][POOL] kept {summ['n_kept']}/{summ['n_problems']} "
          f"({summ['kept_frac']:.3f}) -> {a.out} · prefixes {summ['prefixes']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
