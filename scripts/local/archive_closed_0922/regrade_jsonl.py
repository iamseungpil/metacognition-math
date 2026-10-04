#!/usr/bin/env python
r"""jsonl 롤아웃 산출물의 `r_corr` 를 수리된 `grade_math` 로 다시 채점한다.

    python scripts/local/regrade_jsonl.py PATH [PATH...] [--dry_run] [--workers 8]
                                          [--gold_from texts.jsonl] [--schema auto]

- texts.jsonl / gens.jsonl 둘 다 받는다. gens 행에 `gold` 가 없으면 `--gold_from` 의
  texts.jsonl 에서 problem_id → group_id 순으로 찾는다.
- `--schema` (기본 auto, 행의 필드로 판별):
  * `texts` — 채점 필드 `r_corr`(행 자신의 gold).
  * `gate`  — 게이트 `gens.jsonl`(math_activation_gate). 채점 필드는 **`gen_r_corr`**
    (재생성의 정오)다. 동시에 `r_corr`(원 1차 시도의 정오)를 **재채점된 texts.jsonl**
    에서 roll_id 로 찾아 **새로 고친다**(0→1 은 정상이라 `n_a1_refreshed` 로 따로 센다;
    roll_id 를 못 찾으면 그대로 두고 `n_a1_unmatched` 로 센다).
  * `effort` — math_effort_gate `gens.jsonl`. 채점 필드는 `r_corr`(**생성**의 정오)다.
    단 이어쓰기 계열(wait_free/wait_forced_N/pad_forced_N) 이면서 생성에 새 `\boxed`
    가 없는 행은 값이 «접두의 원 정오를 물려받은 것»(effective_r_corr)이라 **건드리지
    않는다**(`n_skipped_cont`).
- `mode == "label"` 이거나 채점 필드가 -1 인 행은 **건드리지 않는다**.
- 쓰기 전에 원본을 PATH.orig 로 복사한다(이미 있으면 **절대 덮지 않는다**).
- 쓰기는 PATH.tmp → os.replace 원자 교체.
- 1→0 강등이 하나라도 있으면 출력만 하고 **쓰지 않고 exit 2**.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.training.math_meta import grade_math, last_boxed  # noqa: E402

#: 채점 필드(스키마별) — `gate` 만 «생성의 정오» 가 r_corr 가 아니다.
CORR_FIELD = {"texts": "r_corr", "gate": "gen_r_corr", "effort": "r_corr"}
#: math_effort_gate 의 이어쓰기 계열 조건(새 박스 없음 규칙이 걸리는 팔).
CONT_PREFIX = ("wait_free", "wait_forced_", "pad_forced_")


def detect_schema(row: dict) -> str:
    if "gen_r_corr" in row:
        return "gate"
    if "cond" in row and "n_gen_tokens" in row:
        return "effort"
    return "texts"


def _skip(row: dict, field: str = "r_corr") -> bool:
    return row.get("mode") == "label" or row.get(field) == -1


def _grade(arg: tuple[str, str]) -> int:
    return grade_math(arg[0], arg[1])


def _load_gold_map(path: str) -> dict:
    """texts.jsonl → {"by_pid", "by_gid", "a1"} — a1 은 roll_id(`{group_id}#{줄번호}`,
    두 게이트의 선별 규약과 같은 키) → 그 1차 시도 행의 (재채점된) r_corr."""
    out: dict = {"by_pid": {}, "by_gid": {}, "a1": {}}
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("problem_id") is not None and d.get("gold") is not None:
                out["by_pid"].setdefault(d["problem_id"], d["gold"])
            if d.get("group_id") is not None:
                if d.get("gold") is not None:
                    out["by_gid"].setdefault(d["group_id"], d["gold"])
                out["a1"][f"{d['group_id']}#{i}"] = int(d.get("r_corr") or 0)
    return out


def _gold_of(row: dict, gold_map: dict, schema: str):
    if schema == "texts" and row.get("gold") is not None:
        return row.get("gold")
    if row.get("gold") is not None:
        return row["gold"]
    g = gold_map.get("by_pid", {}).get(row.get("problem_id"))
    if g is None:
        g = gold_map.get("by_gid", {}).get(row.get("group_id"))
    return g


def regrade_file(path: str, *, dry_run: bool, workers: int, gold_map: dict,
                 schema: str = "auto", allow_downgrade: bool = False) -> int:
    rows, raw = [], []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            raw.append(line)
            rows.append(json.loads(line))

    if schema == "auto":
        schema = detect_schema(rows[0]) if rows else "texts"
    field = CORR_FIELD[schema]

    idx, jobs, n_skip_cont = [], [], 0
    for i, r in enumerate(rows):
        if _skip(r, field):
            continue
        if schema == "effort" and str(r.get("cond", "")).startswith(CONT_PREFIX) \
                and not last_boxed(r.get("text") or ""):
            n_skip_cont += 1      # 새 박스 없는 이어쓰기 = 접두의 원 정오를 물려받은 값
            continue
        gold = _gold_of(r, gold_map, schema)
        if gold is None:
            continue
        idx.append(i)
        jobs.append((r.get("text") or r.get("response") or "", str(gold)))

    # ★한 행도 채점되지 않았다 = gold 를 못 찾았거나(problem_id 불일치·--gold_from 누락)
    #   스키마 오판이다. 조용히 «0 행 변경»으로 끝나면 «재채점했다»는 거짓 신호가 남는다.
    if rows and not idx:
        print(f"[REGRADE][FATAL] {path}: {len(rows)}행 중 **0행**만 채점 대상이다 — "
              f"gold 를 찾지 못했거나(problem_id 불일치 / --gold_from 누락) 스키마"
              f"(schema={schema}, field={field}) 가 틀렸다. 쓰지 않고 멈춘다.", flush=True)
        return 3

    if workers > 1 and jobs:
        with ProcessPoolExecutor(workers) as ex:
            new = list(ex.map(_grade, jobs, chunksize=32))
    else:
        new = [_grade(j) for j in jobs]

    up, down = [], []
    pos = {i: k for k, i in enumerate(idx)}
    for i, n in zip(idx, new):
        o = int(rows[i].get(field) or 0)
        if o == 0 and n == 1:
            up.append(i)
        elif o == 1 and n == 0:
            # ★워커 프로세스의 math_verify 타임아웃(CPU 경합)이 정답을 0 으로 만드는 함정 —
            #   강등 후보는 본 프로세스에서 한 번 더 직렬 채점해 확인한다.
            if _grade(jobs[pos[i]]) == 1:
                new[pos[i]] = 1
                continue
            down.append(i)

    # ★gate: 1차 시도의 정오(`r_corr`)를 재채점된 texts.jsonl 에서 새로 고친다(0→1 은 정상).
    a1_fix: dict = {}
    n_a1_unmatched = 0
    if schema == "gate":
        a1 = gold_map.get("a1", {})
        for i, r in enumerate(rows):
            v = a1.get(r.get("roll_id"))
            if v is None:
                n_a1_unmatched += 1
                continue
            if int(r.get("r_corr") or 0) != v:
                a1_fix[i] = v

    graded = [int(rows[i].get(field) or 0) for i in idx]
    old_acc = sum(graded) / len(graded) if graded else float("nan")
    new_acc = sum(new) / len(new) if new else float("nan")
    print(f"{path}: schema={schema} field={field} n_rows={len(rows)} n_graded={len(idx)} "
          f"n_changed_0to1={len(up)} n_changed_1to0={len(down)} "
          f"n_skipped_cont={n_skip_cont} n_a1_refreshed={len(a1_fix)} "
          f"n_a1_unmatched={n_a1_unmatched} old_acc={old_acc:.4f} new_acc={new_acc:.4f}")

    if down and allow_downgrade:
        # ★검토 후 허용(예: 마지막 박스가 선택지 글자인 행) — 목록은 남기고 계속 쓴다.
        print(f"  !! 1→0 강등 {len(down)}건 — --allow_downgrade 로 허용:")
        for i in down[:50]:
            print(f"    roll_id={rows[i].get('roll_id')} cond={rows[i].get('cond')}")
    elif down:
        print("  !! 1→0 강등 발생 — 쓰지 않는다:")
        for i in down[:50]:
            print(f"    problem_id={rows[i].get('problem_id')} roll_id={rows[i].get('roll_id')} "
                  f"cond={rows[i].get('cond')} gold={rows[i].get('gold')!r}")
        return 2

    if dry_run:
        return 0

    orig = path + ".orig"
    if not os.path.exists(orig):
        shutil.copy2(path, orig)
        print(f"  원본 보존: {orig}")
    else:
        print(f"  원본 이미 존재(유지): {orig}")

    for i, n in zip(idx, new):
        rows[i][field] = n
    for i, v in a1_fix.items():
        rows[i]["r_corr"] = v
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    print(f"  기록 완료: {path}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--allow_downgrade", action="store_true",
                    help="검토를 마친 1→0 강등을 허용하고 기록한다(기본은 거부)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gold_from", default=None,
                    help="gold 없는 gens.jsonl 용 (재채점된) texts.jsonl 경로")
    ap.add_argument("--schema", default="auto", choices=("auto", "texts", "gate", "effort"))
    a = ap.parse_args(argv)
    gold_map = _load_gold_map(a.gold_from) if a.gold_from else {"by_pid": {}, "by_gid": {}, "a1": {}}
    rc = 0
    for p in a.paths:
        r = regrade_file(p, dry_run=a.dry_run, workers=a.workers, gold_map=gold_map,
                         schema=a.schema, allow_downgrade=a.allow_downgrade)
        rc = r if r else rc
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
