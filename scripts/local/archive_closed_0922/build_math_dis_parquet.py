#!/usr/bin/env python
r"""build_math_dis_parquet — M_DIS(불일치 진단) 학습/검증 parquet 빌더.

★왜 별도 빌더인가. 다른 팔의 프롬프트는 «문제 → 변형 하나»로 조립되므로
`build_math_parquet.py --variant <v>` 하나면 된다. M_DIS 의 사용자 턴은 **행마다 다르다** —
이 정책 자신의 후보 풀이 마무리 4개가 프롬프트 안에 들어가기 때문이다. 그래서 후보를
길어 올릴 롤아웃(`math_rollout.py` 의 texts.jsonl)이 입력으로 하나 더 필요하고, 완성된
사용자 턴을 parquet 의 `prompt` 컬럼에 통째로 싣는다(verl 이 거기서 채팅 메시지를 읽는다).

입력 둘:
  --rollouts      math_rollout.py 가 **TRAIN 문제**에 대해 낸 texts.jsonl
                  (문제당 ≥5 샘플 권장 — 앞 4개를 후보로 쓴다)
  --train_parquet 그 문제들이 들어 있는 원본 학습 parquet(기본: L5 판 math_train_math_crit.
                  parquet — M_CRIT/M_RETRY 와 **같은 문제 풀**이어야 팔 간 비교가 성립한다)

출력: <out_dir>/math_train_math_dis.parquet · <out_dir>/math_val_math_dis.parquet(200행)
행은 원본 parquet 의 컬럼을 전부 물려받고(gold·reward_model·extra_info·level…) 아래가 는다:
  cand_answers    list[str]  후보 4개의 최종 답
  cand_correct    list[int]  그 gold 정오 — ★**지표 전용**. 보상 경로는 읽지 않는다
                             (src/training/math_dis.py 모듈 docstring).
  cand_agree_all  int        후보 4개가 전부 수학적으로 같은 답인가

★후보가 전부 일치하는 문제도 **버리지 않는다** — 불일치가 없을 때 «그냥 커밋한다»도
  배워야 하는 행동이다(불일치 있는 문제만 남기면 «항상 의심하라»를 가르치게 된다).
  그 비율은 빌드 때 보고한다(all_agree_frac).

★0914 리뷰 D3: **잘렸거나(finish_reason=length) \boxed 가 없는 롤아웃은 후보가 아니다** —
  `math_dis.make_sketch` 가 답 없는 후보에 빈 `\boxed{}` 를 붙여 넘어가면, 그 빈 답이 (a) 자기
  군집을 이뤄 다수결 동점을 만들거나 이기고 (b) 커밋되면 미정의가 아니라 −1(오답)로 채점된다.
  `scripts/local/math_disagree_gate.py` 는 이미 이렇게 거른다(select_candidate_sets 의 usable
  필터) — 학습 데이터가 게이트와 다른 후보 풀을 보면 둘이 같은 것을 재지 못한다. 그래서 문제당
  후보를 고를 때 truncated 이거나 최종 답이 빈 롤아웃은 건너뛰고, 그러고도 N_CAND 에 못 미치면
  그 문제 전체를 드롭한다(개수는 n_dropped_truncated 로 보고).

사용법:
  build_math_dis_parquet.py --rollouts $WORK/eval/dis_cand_train/texts.jsonl \
      --train_parquet /hdd_data/seungpil/scratch/data/math_train_math_crit.parquet
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.metacot.math_meta_prompt import build_math_prompt  # noqa: E402
from src.training import math_dis as MD  # noqa: E402
from src.training.math_meta import grade_math, last_boxed, norm_problem  # noqa: E402

VARIANT = "math_dis"
OUT_DIR = Path("/hdd_data/seungpil/scratch/data")
# ★기본 원본: M_CRIT 이 쓰는 Level 5 학습 parquet(2,101행). M_RETRY 계열의
#   math_train_math_retry_fmix0.25.parquet 과 같은 split/seed 에서 나온 같은 문제 풀이다.
DEFAULT_TRAIN_PARQUET = OUT_DIR / "math_train_math_crit.parquet"


def load_candidates(rollouts_path: str, *, n_cand: int = MD.N_CAND) -> dict[str, list[dict]]:
    r"""math_rollout texts.jsonl → {norm_problem: [롤아웃, ...]} (파일 순서 유지).

    ★행에 `final_answer` 가 있으면 그대로 쓴다(math_rollout 이 학습과 **같은** last_boxed 로
    뽑은 값) — 없으면 여기서 다시 뽑는다. `r_corr` 도 같은 규약(없으면 grade_math). `truncated`
    도 그대로 싣는다(없으면 0) — ★0914 리뷰 D3: **여기서는 걸러내지 않는다**. «몇 개가
    쓸 만한가»(잘림·빈 답 제외)는 `build_records` 가 문제당 후보를 고르는 자리에서 판정하고
    드롭 개수를 센다 — 원 목록은 그 판정의 재료로만 쓴다(n_cand 는 더 이상 이 함수의 필터가
    아니다, 하위 호환을 위해 시그니처만 남긴다).
    """
    by: dict[str, list[dict]] = {}
    with open(rollouts_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            text = r.get("text") or ""
            ans = r.get("final_answer")
            if ans is None:
                ans = last_boxed(text)
            corr = r.get("r_corr")
            if corr is None:
                corr = grade_math(text, str(r.get("gold", "")))
            by.setdefault(norm_problem(r["problem"]), []).append(
                {"text": text, "final_answer": str(ans or ""), "r_corr": int(bool(corr)),
                 "truncated": int(bool(r.get("truncated", 0)))})
    return by


def make_dis_record(src: Mapping, cands: Sequence[Mapping], *, n_cand: int = MD.N_CAND) -> dict:
    """원본 parquet 한 행 + 후보 롤아웃 4개 → M_DIS 학습 행.

    ★`prompt` 는 완성된 채팅 메시지다 — 시스템은 math_opt 와 바이트 동일, 사용자 턴은
    `math_dis.build_dis_user_turn`(문제 + 후보 스케치 4개 + 진단 지시). build_math_prompt 가
    «사용자 턴에 후보가 있는가»를 어서션으로 확인한다(없으면 즉사)."""
    c = list(cands)[:n_cand]
    if len(c) != n_cand:
        raise ValueError(f"[MATH][DIS] 후보가 {len(c)} 개다 — {n_cand} 개가 필요하다")
    problem = str(src["problem"])
    sketches = [MD.make_sketch(x["text"]) for x in c]
    answers = [str(x.get("final_answer") or "") for x in c]
    correct = [int(bool(x.get("r_corr", 0))) for x in c]
    # ★«전부 같은 답» = 네 답이 모두 비어 있지 않고 서로 수학적으로 동치(7 ≡ 7.0).
    #   정의는 math_dis.all_agree 한 곳에만 둔다(평가의 all_agree_frac 이 같은 함수를 쓴다).
    agree_all = MD.all_agree(answers)
    user_turn = MD.build_dis_user_turn(problem, sketches)
    rec = dict(src)
    rec["prompt"] = build_math_prompt(user_turn, VARIANT)
    rec["cand_answers"] = answers
    rec["cand_correct"] = correct
    rec["cand_agree_all"] = agree_all
    ei = dict(src.get("extra_info") or {})
    ei.update({"problem": problem, "prompt_variant": VARIANT, "cand_answers": answers,
               "cand_correct": correct, "cand_agree_all": agree_all})
    rec["extra_info"] = ei
    return rec


def build_records(src_rows: Sequence[Mapping], cands_by_problem: Mapping[str, list[dict]], *,
                  val_n: int, seed: int, n_cand: int = MD.N_CAND) -> tuple[list[dict], list[dict], dict]:
    """순수 함수(테스트 가능): 원본 행들 + 후보표 → (train, val, stats).
    후보가 아예 없는 문제는 조용히 건너뛰고 개수를 보고한다(n_no_cand).

    ★0914 리뷰 D3: 문제당 후보를 고를 때 **잘렸거나(truncated) 최종 답이 빈** 롤아웃은 건너뛴다
    (`math_disagree_gate.select_candidate_sets` 의 usable 필터와 같은 규칙 — 게이트와 학습
    데이터가 같은 후보 풀을 재야 한다). 그러고도 `n_cand` 에 못 미치면 그 문제 전체를 드롭하고
    별도로 센다(n_dropped_truncated) — 빈 답 후보가 몰래 들어가면 자기 군집을 이뤄 다수결
    동점을 만들거나 이기고, 커밋되면 미정의가 아니라 −1(오답)로 채점된다."""
    kept, n_no_cand, n_dropped_truncated = [], 0, 0
    for src in src_rows:
        c = cands_by_problem.get(norm_problem(src["problem"]))
        if not c:
            n_no_cand += 1
            continue
        usable = [x for x in c if not int(x.get("truncated", 0) or 0)
                 and str(x.get("final_answer") or "").strip()]
        if len(usable) < n_cand:
            n_dropped_truncated += 1
            continue
        kept.append(make_dis_record(src, usable, n_cand=n_cand))
    random.Random(seed).shuffle(kept)
    val, train = kept[:val_n], kept[val_n:]
    n = max(1, len(kept))
    stats = {
        "n_src": len(src_rows), "n_no_cand": n_no_cand,
        "n_dropped_truncated": n_dropped_truncated, "n_kept": len(kept),
        "n_train": len(train), "n_val": len(val),
        # ★«후보 4개가 전부 같은 답» 비율 — 이 팔이 마주치는 «불일치 없음» 문제의 밀도다.
        "all_agree_frac": sum(int(r["cand_agree_all"]) for r in kept) / n,
        "cand_acc_mean": sum(sum(r["cand_correct"]) / n_cand for r in kept) / n,
    }
    return train, val, stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True,
                    help="math_rollout.py 가 TRAIN 문제에 대해 낸 texts.jsonl(문제당 ≥5 샘플)")
    ap.add_argument("--train_parquet", default=str(DEFAULT_TRAIN_PARQUET),
                    help=f"원본 학습 parquet (기본 {DEFAULT_TRAIN_PARQUET})")
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    ap.add_argument("--val_n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args(argv)

    import pandas as pd
    src = pd.read_parquet(a.train_parquet)
    src_rows = [{k: v for k, v in r.items()} for r in src.to_dict(orient="records")]
    cands = load_candidates(a.rollouts)
    print(f"[build_math_dis_parquet] rollouts={a.rollouts}: {len(cands)} 문제(≥{MD.N_CAND} 샘플) "
          f"· src={a.train_parquet}: {len(src_rows)} 행", flush=True)
    train, val, stats = build_records(src_rows, cands, val_n=a.val_n, seed=a.seed)
    if not train:
        raise SystemExit("[build_math_dis_parquet] FATAL: 후보가 붙은 행이 0 — 롤아웃의 문제 텍스트와 "
                         "parquet 의 문제 텍스트가 같은 원문인지 확인하라(norm_problem 키).")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tp = out / f"math_train_{VARIANT}.parquet"
    vp = out / f"math_val_{VARIANT}.parquet"
    pd.DataFrame(train).to_parquet(tp, index=False)
    pd.DataFrame(val).to_parquet(vp, index=False)
    print(f"[build_math_dis_parquet] {stats}")
    print(f"[build_math_dis_parquet] {tp} ({len(train)} rows)\n"
          f"[build_math_dis_parquet] {vp} ({len(val)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
