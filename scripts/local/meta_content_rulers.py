#!/usr/bin/env python
"""meta_content_rulers — 메타/검산 **내용**을 오라클과 대조하는 자(ruler) 묶음.

왜 새로 만드나 (2026-09-11, 사용자 지시):
  cd7 자 표(09-05)에서 시험한 7종(PMI-shift·OSD×2·도치·dCont·move_kl×2·행동변화)은
  전부 «메타가 무언가에 끼친 영향» 이거나 «모델 내부 확신» 이었다. **메타가 말하는
  내용이 사실인가**를 잰 자는 하나도 없었다. Countdown 은 완전열거 오라클이 있어
  그걸 잴 수 있는데도 비어 있던 축이다.

재는 것 (블록 하나당):
  names_move     — 주어진 수 중 둘로 만든 구체적 첫수를 **지목**했는가(구체성)
  move_novel     — 지목한 수가 프리픽스에서 **아직 안 써본** 것인가(새로움)
  move_live      — 지목한 수가 오라클 기준 **아직 해로 가는 길**인가(진리)
  followed       — 지목한 수를 블록 **뒤에서 실제로 시도**했는가(약속 이행)
  arith_true     — 블록 안 "a op b = c" 주장이 실제로 맞는가(진술 검증)
  localizes      — 프리픽스에 나온 식을 **짚어서** 언급하는가(오류 위치)

그리고 뭉치 단위로 정형문(boilerplate) 비율을 잰다 — `countdown_rewards.boilerplate_rate`
는 `<meta>` 만 보므로 chk 계열에서 n_emitted=0/NaN 으로 눈이 먼다(0911 발견). 여기서는
`--form` 으로 meta/check 둘 다 같은 자를 적용한다.

사용법:
  meta_content_rulers.py --texts $WORK/eval/<lineage>/step_50/texts.jsonl --form meta
  (여러 개면 --texts 를 반복. --out 에 JSON 요약, --dump 에 행별 jsonl)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training import countdown_sites as CS  # noqa: E402
from src.training import countdown_task as CT  # noqa: E402

_BLOCK = {
    "meta": re.compile(r"<meta>(.*?)</meta>", re.IGNORECASE | re.DOTALL),
    "check": re.compile(r"<check>(.*?)</check>", re.IGNORECASE | re.DOTALL),
}
# "24 * 17", "24*17", "24 x 17" — 두 정수와 연산자 하나.
_PAIR = re.compile(r"(?<![\d.])(\d{1,4})\s*([+\-*/x×÷])\s*(\d{1,4})(?![\d.])")
# "24 * 17 = 408" — 주장된 등식.
_CLAIM = re.compile(r"(?<![\d.])(\d{1,4})\s*([+\-*/x×÷])\s*(\d{1,4})\s*=\s*(\d{1,6})(?![\d.])")
_OPMAP = {"x": "*", "×": "*", "÷": "/"}


def _norm_op(op: str) -> str:
    return _OPMAP.get(op, op)


def _norm_body(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _named_moves(body: str, nums) -> list[tuple]:
    """블록 안에서 지목한 «첫수» 후보 — 양쪽 피연산자가 모두 주어진 수일 때만."""
    pool = Counter(int(n) for n in nums)
    out = []
    for a, op, b in _PAIR.findall(body):
        a, b, op = int(a), int(b), _norm_op(op)
        if a == b:
            ok = pool.get(a, 0) >= 2
        else:
            ok = pool.get(a, 0) >= 1 and pool.get(b, 0) >= 1
        if ok:
            out.append(CS.canon_move(a, op, b))
    return out


_EXPR_CHARS = set("0123456789+-*/(). ")


def _lhs_expr(seg: str) -> str | None:
    """`=` 앞 조각에서 **완결된** 산술식만 떼어낸다.

    ★0911 수리: 앞선 판은 `(\\d)op(\\d)=(\\d)` 정규식으로 부분식을 떼어내 채점했다 —
    `(22 + 11) * 20 * 3 = 1980` 에서 `20 * 3 = 1980` 만 보고 «거짓» 으로 셌다(거짓
    음성). 괄호를 포함한 꼬리 전체를 잡아 괄호 균형을 맞춘 뒤 통째로 평가한다.
    """
    tail = []
    for ch in reversed(seg):
        if ch in _EXPR_CHARS:
            tail.append(ch)
        else:
            break
    e = "".join(reversed(tail)).strip()
    if not e or not any(c.isdigit() for c in e):
        return None
    while e and e.count("(") < e.count(")"):     # 왼쪽이 잘렸으면 닫는 괄호를 버린다
        e = e[1:].strip() if e[0] == ")" else e[:-1].strip()
    if e.count("(") > e.count(")"):
        return None
    return e if re.search(r"[+\-*/]", e) else None


def _arith_claims(body: str) -> tuple[int, int]:
    """(주장 수, 그중 참인 수). `A = B` 의 A 를 **식 전체로** 평가해 B 와 비교한다.

    `a = b = c` 연쇄는 인접 쌍마다 하나의 주장으로 센다.
    """
    parts = re.split(r"=", body)
    n = ok = 0
    for i in range(len(parts) - 1):
        lhs = _lhs_expr(parts[i])
        m = re.match(r"\s*(\d{1,7})(?![\d.])", parts[i + 1])
        if lhs is None or m is None:
            continue
        n += 1
        try:
            v = CT.eval_exact(lhs)
        except Exception:
            continue
        if v is not None and abs(float(v) - int(m.group(1))) < 1e-9:
            ok += 1
    return n, ok


def analyse_row(row: dict, form: str) -> list[dict]:
    text = row.get("text") or ""
    nums = list(row.get("nums") or [])
    target = row.get("target")
    pat = _BLOCK[form]
    recs = []
    for m in pat.finditer(text):
        body = m.group(1)
        prefix, suffix = text[: m.start()], text[m.end():]
        try:
            orc = CS.oracle_for_site(prefix, nums, target)
        except Exception:
            orc = {}
        live = set(orc.get("live_new_moves") or ())
        pairs_pre = {tuple(sorted(p)) for p in (orc.get("pairs_pre") or ())}

        moves = _named_moves(body, nums)
        keys = [CS.move_key_str(mv) for mv in moves]
        novel = [k for k, mv in zip(keys, moves)
                 if tuple(sorted((mv[0], mv[2]))) not in pairs_pre]
        live_hit = [k for k in keys if k in live]
        followed = [k for k, mv in zip(keys, moves)
                    if _PAIR.search(suffix) and any(
                        CS.canon_move(int(a), _norm_op(o), int(b)) == mv
                        for a, o, b in _PAIR.findall(suffix))]
        n_claim, n_true = _arith_claims(body)
        # 오류 위치 지목: 블록이 언급한 식 중 프리픽스에 이미 나온 것
        localizes = any(tuple(sorted((mv[0], mv[2]))) in pairs_pre for mv in moves)

        recs.append({
            "group_id": row.get("group_id"),
            "r_corr": int(row.get("r_corr") or 0),
            "family_dead": orc.get("family_dead"),
            "pos_frac": round(m.start() / max(1, len(text)), 4),
            "body": body.strip(),
            "names_move": int(bool(moves)),
            "n_named": len(moves),
            "move_novel": int(bool(novel)),
            "move_live": int(bool(live_hit)),
            "followed": int(bool(followed)),
            "n_claim": n_claim,
            "n_claim_true": n_true,
            "localizes": int(localizes),
        })
    return recs


def summarise(recs: list[dict], bodies: list[str]) -> dict:
    n = len(recs)
    if n == 0:
        return {"n_blocks": 0}

    def rate(k):
        return round(sum(r[k] for r in recs) / n, 4)

    norm = [_norm_body(b) for b in bodies if _norm_body(b)]
    c = Counter(norm)
    top = c.most_common(5)
    claims = sum(r["n_claim"] for r in recs)
    return {
        "n_blocks": n,
        "names_move": rate("names_move"),
        "move_novel": rate("move_novel"),
        "move_live": rate("move_live"),
        "followed": rate("followed"),
        "localizes": rate("localizes"),
        "n_claim_total": claims,
        "claim_true_rate": round(sum(r["n_claim_true"] for r in recs) / claims, 4) if claims else None,
        "pos_frac_median": round(sorted(r["pos_frac"] for r in recs)[n // 2], 4),
        "boilerplate_top1_share": round(top[0][1] / len(norm), 4) if norm else None,
        "unique_ratio": round(len(c) / len(norm), 4) if norm else None,
        "top_bodies": [(b[:100], round(k / len(norm), 4)) for b, k in top],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", action="append", required=True)
    ap.add_argument("--form", choices=("meta", "check"), default="meta")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump", default=None)
    a = ap.parse_args()

    report = {}
    for path in a.texts:
        recs, bodies = [], []
        with open(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                rs = analyse_row(json.loads(line), a.form)
                recs.extend(rs)
                bodies.extend(r["body"] for r in rs)
        report[path] = summarise(recs, bodies)
        print(f"\n=== {path} (form={a.form}) ===")
        for k, v in report[path].items():
            if k != "top_bodies":
                print(f"  {k:24s} {v}")
        for b, share in (report[path].get("top_bodies") or []):
            print(f"    top {share:6.3f}  {b}")
        if a.dump:
            with open(a.dump, "a") as fh:
                for r in recs:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    if a.out:
        Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"\n[out] {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
