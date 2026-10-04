#!/usr/bin/env python
r"""습관 탐침 — 고정된 «첫 박스까지» 앞부분(mc/pool.py `.prefixes.jsonl`)을 plain 1턴 템플릿 뒤에 그대로
붙여 이어 쓰게 하고 reopen(박스 ≥2)·revised·최종 정답(gold)을 잰다. 선택(첫 답 정오별 문제 층화·시드
고정)은 `selection.json`(`--selection` 재사용 = 모든 체크포인트가 같은 앞부분). CI = 문제 군집
부트스트랩, `--ref` = 같은 선택의 문제 짝 차이. `--merge_only ACTOR` = 병합 경로만 출력. `--resummarize DIR` = GPU 없이 DIR 의
selection·texts 를 다시 채점해 summary.json 을 새 지표(메타 말투 등)로 다시 쓴다(`--ref` 가능)."""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mc import context as ctx
from mc.eval import paired_bootstrap
from mc.grade import answers_equivalent_loose, boxed_answer, boxed_spans, grade_answer, selftest
from mc.rollout import build_engine, merged_model

PREFIX_BUDGET = 10240   #: 앞부분(≤8,192 토큰 롤아웃에서 자름)+프롬프트 여유 — max_model_len 에 더한다
METRICS = (("reopen_wrong", "wrong", "reopen"), ("revise_wrong", "wrong", "revised"),   # (이름, 계급[:조건 키], 키)
           ("fix_wrong", "wrong", "final_correct"), ("reopen_right", "right", "reopen"),
           ("break_right", "right", "broke"), ("tokens", None, "n_tok"), ("trunc", None, "trunc"),
           ("meta_wrong", "wrong", "meta"), ("meta_right", "right", "meta"), ("fix_after_meta", "wrong:meta",
                                                                            "final_correct"),
           ("format_wrong", "wrong", "format"), ("format_right", "right", "format"))
#: 수정 20′ 메타 말투 — 이어쓰기(첫 박스 바로 뒤) 앞 200자 안의 되짚기 말. «형식» = 되짚었는데(박스 ≥2) 메타 말이 없음.
#: 0925 손 점검(probe3_base_k16 되짚은 행, 메타 25·형식 25)으로 ’ 아포스트로피·«Check:»·«is X really/correct» 를 더했다.
META_RE = re.compile(r"\b(wait|hold on|hmm+|let(?: me|['’]s| us)(?: briefly| quickly)? (?:re)?(?:check|verify|double|"
                     r"recompute|re-?examine|reconsider|confirm)|but let['’]?s|actually|however|re-?check|double[- ]check|sanity check|"
                     r"mistake|error|not correct|incorrect|check(?=\s*:)|is (?:this|that|it|the \w+) (?:really|indeed|"
                     r"(?:the )?correct|right|valid))\b", re.I)
META_CHARS = 200
#: 측정 전용(0925) — 첫 박스 뒤 600자 안 메타 말 세 갈래. 갈래마다 비율(틀림·맞음)·고침|틀림·망침|맞음.
META_CATS = {"verify": r"wait|hold on|hmm+|let me (?:re)?(?:check|verify|double|recompute|re-?examine)|re-?check|double[- ]check|verify",
             "error": r"mistake|error|incorrect|not correct|wrong|miscalculat\w*",
             "switch": r"alternatively|another (?:approach|way|method)|different (?:approach|method|way)|let['’]s try (?:a|another|"
                       r"again)|instead, let|start over|from scratch|re-?approach|let me try"}
META_CATS = {k: re.compile(rf"\b(?:{v})\b", re.I) for k, v in META_CATS.items()}
CAT_CHARS = 600
METRICS += tuple(m for k in META_CATS for m in ((f"{k}_wrong", "wrong", f"m_{k}"), (f"{k}_right", "right", f"m_{k}"),
                                                (f"fix_after_{k}", f"wrong:m_{k}", "final_correct"),
                                                (f"break_after_{k}", f"right:m_{k}", "broke")))


#: 수정 22 (ii) 글자 규칙 — 첫 박스 앞 200자 종결 표지 ∧ ¬(뒤 300자 안 Step/Case 제목 ∨ 박스 뒤 같은 줄 본문 이어짐).
END_RE = re.compile(r"final answer|answer is|answer:|therefore|thus|hence|in conclusion|conclude|go with|closest to", re.I)
STEP_RE = re.compile(r"#+\s*(?:step|case|subcase|part)|\*\*case|^#+ ", re.I | re.M)


def committed_first_box(text: str) -> bool | None:
    r"""첫 `\boxed` 가 «약속한 첫 답»인가(수정 22 (ii)) — text = 응답 전체(첫 박스 뒤 본문 포함). 박스가 없으면 None.
    이어짐 = 박스 뒤 같은 줄에 `$`·구두점·`\quad`·`\text{…}`·`(1)` 말고 본문이 있다(«…= \boxed{3} at x = 0»)."""
    sp = boxed_spans(text)
    if not sp:
        return None
    _, a, b = sp[0]
    line = re.sub(r"[\s$.,;:!?)\]}]|\\\]|\\\)|\\quad|\\text\{[^}]*\}|\(\d+\)", "", text[b:].split("\n", 1)[0])
    return bool(END_RE.search(text[max(0, a - 200):a])) and not (STEP_RE.search(text[b:b + 300]) or line)


def intermediate_box(text: str, a: int, b: int) -> bool:
    r"""수정 23a 글자 거부 — [a, b) 박스가 «중간값» 이다: (뒤 600자 메타 말(`META_CATS`) 없음 ∨ 뒤 300자 Step/Case 제목) ∧ 앞 200자
    종결 표지(`END_RE`) 없음. 손 라벨 166: 약속 무게 ≥ .5 와 결합하면 정밀도 .94 · 재현율 .76(독립 40: .89 · .73)."""
    no_meta = not any(p.search(text[b:b + CAT_CHARS]) for p in META_CATS.values())
    return bool((no_meta or STEP_RE.search(text[b:b + 300])) and not END_RE.search(text[max(0, a - 200):a]))


def select(prefixes: list[dict], n_wrong: int, n_right: int, seed: int) -> list[dict]:
    """첫 답 오답/정답 각 n 개 — 문제(uid)를 돌아가며 하나씩 뽑는다(층화, 시드 고정)."""
    rng, out = random.Random(seed), []
    for right, n in ((False, n_wrong), (True, n_right)):
        by: dict[str, list[dict]] = {}
        for r in (x for x in prefixes if bool(x["first_correct"]) == right):
            by.setdefault(str(r["uid"]), []).append(r)
        pools = [rng.sample(by[k], len(by[k])) for k in rng.sample(sorted(by), len(by))]
        picked: list[dict] = []
        while len(picked) < n and any(pools):
            picked += [q.pop() for q in pools if q][:n - len(picked)]
        out += picked
    return [{**r, "pid": i} for i, r in enumerate(out)]


def score(prefix: str, cont: str, gold: str) -> dict:
    """앞부분의 **마지막** 박스(= 약속한 첫 답 A0; 수정 22 앞부분은 그 앞에 중간 박스가 있을 수 있다) 뒤 이어쓰기에 박스가
    있으면 다시 열었다, 그 마지막 답이 A0 와 loose 동치가 아니면 바꿨다(앞부분 박스 1개면 `revision_zone(full)` 과 같다).
    meta = 앞 200자 메타 말투, format = 되짚음 ∧ 메타 말 없음, committed = 첫 박스가 약속한 답(글자 규칙 — 측정 전용)."""
    full, cb = prefix + cont, boxed_spans(cont)
    ok, meta = bool(grade_answer(boxed_answer(full) or "", gold)), bool(META_RE.search(cont[:META_CHARS]))
    z = bool(cb)
    return {"reopen": z, "revised": z and not answers_equivalent_loose(boxed_answer(prefix) or "", cb[-1][0]),
            "final_correct": ok, "broke": not ok, "meta": meta, "format": z and not meta,
            **{f"m_{k}": bool(r.search(cont[:CAT_CHARS])) for k, r in META_CATS.items()},
            "committed": bool(committed_first_box(full))}


def per_problem(rows: list[dict]) -> dict:
    """METRICS 이름 → {uid: 문제 안 (무게) 평균} — 탐침 요약과 SPONT_PFX 스텝 계기가 같은 자를 쓴다(키 없는 행은 그 지표에서
    뺀다, 무게 `w` 기본 1 — 무게 합 0 인 문제는 빠진다)."""
    per = {}
    for name, cls, key in METRICS:
        by: dict[str, list[tuple]] = {}
        c, *cond = (cls or "").split(":")
        for r in rows:
            if key in r and (cls is None or (r["cls"] == c and all(r.get(k) for k in cond))):
                by.setdefault(str(r["uid"]), []).append((float(r[key]), float(r.get("w", 1.0))))
        per[name] = {u: sum(x * w for x, w in v) / sw for u, v in by.items() if (sw := sum(w for _, w in v)) > 0}
    return per


def summarize(rows: list[dict]) -> dict:
    """문제별 평균 → `paired_bootstrap([(0, v)])` (`delta` = 문제 평균, lo/hi = 95% CI)."""
    per = per_problem(rows)
    return {"per_problem": per,
            **{n: paired_bootstrap([(0.0, v) for v in per[n].values()]) for n, *_ in METRICS}}


def vs_ref(summ: dict, ref_path: str) -> dict:
    ref = json.loads(Path(ref_path).read_text())["per_problem"]
    return {n: paired_bootstrap([(ref[n][k], summ["per_problem"][n][k]) for k in
                                 sorted(set(ref.get(n, {})) & set(summ["per_problem"][n]))])
            for n, *_ in METRICS}


def _report(summ: dict, path: Path) -> None:
    path.write_text(json.dumps(summ, ensure_ascii=False, indent=2, default=float))
    print({n: [round(d[k], 4) for d in (summ[n], summ.get("vs_ref", {}).get(n)) if d
               for k in ("delta", "lo", "hi")] for n, *_ in METRICS})    # 추정·CI (+ Δref·CI)
    print(f"[out] {path}", flush=True)


def resummarize(d: Path, ref: str | None = None, weight_key: str | None = None) -> int:
    """CPU 만 — d/selection.json·texts.jsonl 을 `score` 로 다시 채점해 d/summary.json 을 다시 쓴다(생성 메타 필드는 둔다).
    `weight_key` = 선택 항목의 앞부분 무게 열(수정 22 약속한 첫 박스: `w_commit` · `commit_rule`, 없으면 0) → 무게 요약
    d/summary_<weight_key>.json(`ref` 도 같은 판을 준다)."""
    selftest()
    sel = {int(r["pid"]): r for r in json.loads((d / "selection.json").read_text())}
    rows = [{**t, **score(sel[int(t["pid"])]["prefix"], t["text"], str(sel[int(t["pid"])]["gold"])),
             "w": float(sel[int(t["pid"])].get(weight_key, 0.0)) if weight_key else 1.0}
            for t in (json.loads(x) for x in open(d / "texts.jsonl") if x.strip())]
    old = json.loads((d / "summary.json").read_text())
    summ = {**{k: v for k, v in old.items() if k not in ("vs_ref", "per_problem")}, **summarize(rows)}
    if ref:
        summ["vs_ref"] = vs_ref(summ, ref)
    _report({**summ, "n_rows_weighted": sum(r["w"] for r in rows)},
            d / (f"summary_{weight_key}.json" if weight_key else "summary.json"))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    for k in ("--merge_only", "--model_path", "--prefixes", "--selection", "--out_dir", "--ref", "--resummarize"):
        ap.add_argument(k)
    ap.add_argument("--weight_key")                          # --resummarize 전용: 앞부분 무게 열(w_commit · commit_rule)
    for k, t, d in (("--n_wrong", int, 200), ("--n_right", int, 200), ("--k", int, 4),
                    ("--max_tokens", int, 4096), ("--seed", int, 11), ("--gpu_util", float, 0.8)):
        ap.add_argument(k, type=t, default=d)
    a = ap.parse_args(argv)
    if a.merge_only:
        return print(merged_model(a.merge_only)) or 0
    if a.resummarize:
        return resummarize(Path(a.resummarize), a.ref, a.weight_key)
    if not (a.model_path and a.out_dir and (a.prefixes or a.selection)):
        raise SystemExit("[MC] --model_path·--out_dir 와 --prefixes 또는 --selection 이 필요하다.")
    from vllm import SamplingParams
    selftest()
    (out := Path(a.out_dir)).mkdir(parents=True, exist_ok=True)
    sel = (json.loads(Path(a.selection).read_text()) if a.selection else select(
        [json.loads(x) for x in open(a.prefixes) if x.strip()], a.n_wrong, a.n_right, a.seed))
    (out / "selection.json").write_text(json.dumps(sel, ensure_ascii=False))
    model = merged_model(a.model_path)
    llm, tok = build_engine(model, max_tokens=a.max_tokens + PREFIX_BUDGET, gpu_util=a.gpu_util,
                            seed=a.seed)
    outs = llm.generate([ctx.turn1_prompt(tok, r["problem"], "plain") + r["prefix"] for r in sel],
                        SamplingParams(n=a.k, temperature=1.0, top_p=1.0,
                                       max_tokens=a.max_tokens, seed=a.seed))
    rows = [{"pid": r["pid"], "uid": r["uid"], "cls": "right" if r["first_correct"] else "wrong",
             "n_tok": len(c.token_ids), "trunc": c.finish_reason == "length", "text": c.text,
             **score(r["prefix"], c.text, str(r["gold"]))} for r, o in zip(sel, outs) for c in o.outputs]
    summ = {**summarize(rows), "model_path": model, "n_prefixes": len(sel), "k": a.k,
            "max_tokens": a.max_tokens, "seed": a.seed}
    if a.ref:
        summ["vs_ref"] = vs_ref(summ, a.ref)
    (out / "texts.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    _report(summ, out / "summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
