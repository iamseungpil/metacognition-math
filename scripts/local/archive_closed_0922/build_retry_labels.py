#!/usr/bin/env python
"""build_retry_labels — 사전등록 수정 3 의 **Stage-1 숫자**: «재시도가 옳은 결정인 행»이 얼마나 되나.

★왜 이 숫자가 먼저인가. M_RETRY 는 «첫 답이 틀렸을 때 redirect, 맞았을 때 verify» 를 보상한다.
그 보상이 학습 신호가 되려면 (1) 첫 답이 틀린 행이 충분히 있어야 하고 (2) 그 행에서 다시 풀면
실제로 맞힐 확률(p_retry)이 0 보다 커야 한다. 둘 중 하나가 없으면 판단 항은 «항상 verify» 로
붕괴한다(RESULTS_cd9 s5b: MATH-500 은 52% 문제가 8/8 정답 — 재시도 자리가 거의 없다).

입력: math_rollout.py 의 texts.jsonl(행 = 롤아웃, group_id / r_corr / truncated / n_tok).
행마다:
    p_retry        = 같은 그룹의 **다른** 롤아웃들의 정답률(자기 자신 제외 — 자기 결과로 자기
                     재시도 확률을 재면 «틀린 행의 p_retry» 가 체계적으로 낮아진다)
    redirect_right = r_corr==0 ∧ p_retry>0   (틀렸고, 다시 풀면 맞을 여지가 있다)
    verify_right   = r_corr==1               (맞았으니 멈추는 게 옳다)
    neither        = r_corr==0 ∧ p_retry==0  (틀렸고 다시 풀어도 못 맞힌다 — 재시도 무익)
요약(데이터셋별): 행 수, first_acc(=평균 r_corr), redirect_right 비율, verify_right 비율, neither 비율,
그리고 «재시도 결정이 옳은 행에서 재시도가 맞힐 기대 확률»(mean p_retry | redirect_right).

⚠AIME25/HMMT25 파일은 **Qwen3.5** 롤아웃(정책 후보는 Instruct-2507)이다 — 출력에 모델 태그를 찍는다.

사용법:
  build_retry_labels.py --inputs math500=/…/math500_q3i2507_plain_b8k/texts.jsonl \\
      aime25=/…/aime25_qwen35_plain_b32k/texts.jsonl [--out labels.jsonl]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def label_rows(rows: Sequence[Mapping]) -> list[dict]:
    """행마다 p_retry(다른 행 기준) / redirect_right / verify_right / neither 를 붙인다(원행 불변)."""
    groups: dict = defaultdict(list)
    for r in rows:
        groups[str(r["group_id"])].append(int(r["r_corr"]))
    out = []
    for r in rows:
        g = groups[str(r["group_id"])]
        corr = int(r["r_corr"])
        others_n = len(g) - 1
        # ★자기 자신 제외: 그룹 정답 합에서 자기 결과를 뺀다(K=1 그룹이면 정의 불가 → NaN).
        p_retry = ((sum(g) - corr) / others_n) if others_n > 0 else float("nan")
        redirect_right = int(corr == 0 and p_retry == p_retry and p_retry > 0)
        verify_right = int(corr == 1)
        neither = int(corr == 0 and p_retry == p_retry and p_retry == 0)
        out.append({**dict(r), "p_retry": p_retry, "redirect_right": redirect_right,
                    "verify_right": verify_right, "neither": neither})
    return out


def summarize(labeled: Sequence[Mapping]) -> dict:
    n = max(1, len(labeled))
    rr = [r for r in labeled if r["redirect_right"]]
    return {
        "n_rows": len(labeled),
        "n_groups": len({str(r["group_id"]) for r in labeled}),
        "first_acc": sum(int(r["r_corr"]) for r in labeled) / n,
        "redirect_right_rate": len(rr) / n,
        "verify_right_rate": sum(int(r["verify_right"]) for r in labeled) / n,
        "neither_rate": sum(int(r["neither"]) for r in labeled) / n,
        "p_retry_given_redirect_right": (sum(float(r["p_retry"]) for r in rr) / len(rr)) if rr else float("nan"),
        "trunc_rate": sum(int(r.get("truncated", 0)) for r in labeled) / n,
        "n_tok_mean": sum(int(r.get("n_tok", 0)) for r in labeled) / n,
    }


def _model_tag(path: str) -> str:
    p = path.lower()
    if "qwen35" in p or "qwen3.5" in p:
        return "Qwen3.5 (NOT the Instruct-2507 policy candidate)"
    if "q3i2507" in p or "instruct-2507" in p or "2507" in p:
        return "Qwen3-4B-Instruct-2507"
    return "unknown-model (see path)"


def run(inputs: Iterable[tuple[str, str]], out_path: str | None = None) -> dict:
    summaries = {}
    all_labeled = []
    for name, path in inputs:
        labeled = label_rows(load_rows(path))
        for r in labeled:
            r["dataset"] = name
        all_labeled.extend(labeled)
        st = summarize(labeled)
        st["model"] = _model_tag(path)
        st["path"] = path
        summaries[name] = st
    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            for r in all_labeled:
                slim = {k: r[k] for k in ("dataset", "group_id", "problem_id", "gold", "r_corr",
                                          "p_retry", "redirect_right", "verify_right", "neither")}
                fh.write(json.dumps(slim, ensure_ascii=False) + "\n")
    return summaries


def format_summary(summaries: Mapping[str, Mapping]) -> str:
    lines = ["[retry-labels] Stage-1: fraction of rows where the retry decision is right",
             f"{'dataset':10s} {'n':>5s} {'first_acc':>9s} {'redir_right':>11s} {'verify_right':>12s} "
             f"{'neither':>8s} {'p_retry|rr':>10s} {'trunc':>6s} {'tok':>6s}  model"]
    for name, st in summaries.items():
        lines.append(f"{name:10s} {st['n_rows']:5d} {st['first_acc']:9.3f} {st['redirect_right_rate']:11.3f} "
                     f"{st['verify_right_rate']:12.3f} {st['neither_rate']:8.3f} "
                     f"{st['p_retry_given_redirect_right']:10.3f} {st['trunc_rate']:6.3f} "
                     f"{st['n_tok_mean']:6.0f}  {st['model']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="name=path/to/texts.jsonl ...")
    ap.add_argument("--out", default=None, help="행 라벨 jsonl(선택)")
    a = ap.parse_args(argv)
    pairs = []
    for it in a.inputs:
        if "=" not in it:
            raise SystemExit(f"--inputs 항목은 name=path 형식이어야 한다: {it!r}")
        pairs.append(tuple(it.split("=", 1)))
    summ = run(pairs, a.out)
    print(format_summary(summ))
    print(json.dumps(summ, ensure_ascii=False, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
