#!/usr/bin/env python
"""build_math_parquet — MATH_META 학습/검증 parquet 빌더.

★데이터 선택 이유(0914). MATH-500(HuggingFaceH4/MATH-500)은 **시험 세트**다 — 거기서
학습하면 held-out 판정이 오염된다(Countdown E-133 과 같은 사고). 그래서 학습은
`EleutherAI/hendrycks_math` **train** split(7 config 전부, 7,500문제)에서 뽑고, MATH-500
문제는 **문제 텍스트 완전 일치**로 제외한다(MATH-500 은 hendrycks test 의 부분집합이라
원리상 train 과 겹치지 않지만, 겹침을 «가정»하지 않고 확인해 제거한다). 검증은 train 에서
떼어 낸 200문제(학습에서 제외).

정답(gold)은 `solution` 의 **마지막** \\boxed{...} — `math_meta.last_boxed`(트레이너·롤아웃과
같은 함수). 빈 gold 행은 버린다(math_verify 가 채점할 수 없으므로 조용한 0 이 된다).

행 스키마(verl `data.prompt_key=prompt`):
    prompt        chat messages (build_math_prompt(problem, variant))
    problem, gold flat 컬럼
    extra_info    {problem, gold}   ★async 경로는 flat 을 pop 하므로 여기서도 읽는다
    reward_model  {style: rule, ground_truth: gold}
    data_source   "math_meta"

출력:  /hdd_data/seungpil/scratch/data/math_train_<variant>.parquet
       /hdd_data/seungpil/scratch/data/math_val_<variant>.parquet
HF 캐시: /hdd_data/seungpil/scratch/hf_home (루트 디스크 금지 규약).
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/hdd_data/seungpil/scratch/hf_home")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.metacot.math_meta_prompt import MATH_PROMPT_VARIANTS, build_math_prompt  # noqa: E402
from src.training.math_meta import last_boxed, norm_problem  # noqa: E402

HENDRYCKS = "EleutherAI/hendrycks_math"
HENDRYCKS_CONFIGS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                     "number_theory", "prealgebra", "precalculus")
MATH500 = "HuggingFaceH4/MATH-500"
OUT_DIR = Path("/hdd_data/seungpil/scratch/data")

# ★0914 AIME-급 확장. deepscaler = agentica-org/DeepScaleR-Preview-Dataset(problem/answer/solution,
#   gold 은 answer 컬럼을 직접 쓴다 — hendrycks 처럼 solution 에서 last_boxed 로 뽑지 않는다;
#   answer 가 이미 순수 답 문자열이라 last_boxed 를 씌우면 오히려 실패한다). dapo17k =
#   BytedTsinghua-SIA/DAPO-Math-17k(prompt 가 채팅 리스트로 감싸인 지시문 템플릿 — 아래
#   _DAPO_PREFIX_RE/_DAPO_SUFFIX_RE 로 벗겨 순문제만 남긴다; gold 는 reward_model.ground_truth).
DEEPSCALER = "agentica-org/DeepScaleR-Preview-Dataset"
DAPO17K = "BytedTsinghua-SIA/DAPO-Math-17k"
SOURCES = ("hendrycks", "deepscaler", "dapo17k")

# ★0914b: --forced_variant 의 기본값은 --variant 에 종속된다(math_retry→math_retry_forced,
#   math_agree→math_agree_forced) — 고정 문자열 하나였을 때는 M_AGREE 빌드에서 매번
#   --forced_variant math_agree_forced 를 명시해야 했다(잊으면 조용히 math_retry_forced 로
#   다시 써 버린다 — decision 이 항상 redirect 인 것은 같아도 agreement: 줄이 없는 블록으로
#   강제되므로 M_AGREE 의 판단 항 정의가 깨진다). 매핑에 없는 --variant 로 --forced_frac>0 을
#   쓰면 즉사(조용한 기본값 금지).
_FORCED_VARIANT_OF = {"math_retry": "math_retry_forced", "math_agree": "math_agree_forced"}


def resolve_forced_variant(forced_variant: str | None, variant: str) -> str | None:
    """--forced_variant 명시값이 있으면 그대로, 없으면 --variant 에서 유도(_FORCED_VARIANT_OF).
    매핑에 없는 variant 면 None(호출자가 --forced_frac>0 이면 즉사시킨다)."""
    return forced_variant or _FORCED_VARIANT_OF.get(variant)

import re as _re  # noqa: E402

_DAPO_PREFIX_RE = _re.compile(
    r"^Solve the following math problem step by step\. The last line of your response "
    r"should be of the form Answer: \$Answer \(without quotes\) where \$Answer is the "
    r"answer to the problem\.\n\n")
_DAPO_SUFFIX_RE = _re.compile(
    r"\n\nRemember to put your answer on its own line after \"Answer:\"\.\s*$")


def strip_dapo_template(text: str) -> str:
    """★DAPO-Math-17k 의 prompt 는 지시문 템플릿으로 문제를 감싼다. 알려진 접두/접미를
    벗겨 «순문제»만 남긴다 — 못 벗기면(템플릿이 바뀌었으면) 원문 그대로 반환한다(조용히
    잘못 자르는 것보다 안전; 호출부가 n_unstripped 로 몇 건인지 보고한다)."""
    out = _DAPO_PREFIX_RE.sub("", text)
    out = _DAPO_SUFFIX_RE.sub("", out)
    return out.strip()


def load_deepscaler_rows() -> tuple[list[dict], int]:
    from datasets import load_dataset
    ds = load_dataset(DEEPSCALER, split="train")
    rows, n_nogold = [], 0
    for r in ds:
        problem = str(r.get("problem", "")).strip()
        gold = str(r.get("answer", "")).strip()
        if not problem or not gold:
            n_nogold += 1
            continue
        rows.append({"problem": problem, "gold": gold, "level": "", "type": "deepscaler"})
    return rows, n_nogold


def load_dapo17k_rows() -> tuple[list[dict], int, int]:
    from datasets import load_dataset
    ds = load_dataset(DAPO17K, split="train")
    rows, n_nogold, n_unstripped = [], 0, 0
    for r in ds:
        prompt = r.get("prompt")
        raw = str(prompt[0]["content"]) if prompt else ""
        stripped = strip_dapo_template(raw)
        if stripped == raw.strip():
            n_unstripped += 1
        gold = str((r.get("reward_model") or {}).get("ground_truth", "")).strip()
        if not stripped or not gold:
            n_nogold += 1
            continue
        rows.append({"problem": stripped, "gold": gold, "level": "", "type": "dapo17k"})
    return rows, n_nogold, n_unstripped


def load_heldout_problems() -> set[str]:
    """★MATH-500 + AIME 2024/2025 + HMMT 2025-02(math_rollout.DATASETS 와 같은 id)를 전부
    합쳐 제거 대상으로 쓴다 — «AIME 급으로 옮긴다»는 목적상 AIME/HMMT 를 학습에 흘리면
    바로 그 held-out 판정이 오염된다."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from math_rollout import DATASETS  # noqa: E402

    from datasets import load_dataset as _ld
    heldout: set[str] = set()
    for name, (ds_id, cfg, split, qcol, _acol) in DATASETS.items():
        ds = _ld(ds_id, cfg, split=split) if cfg else _ld(ds_id, split=split)
        heldout.update(norm_problem(r[qcol]) for r in ds)
        print(f"[build_math_parquet] heldout[{name}] += {len(ds)} problems", flush=True)
    return heldout


def make_record(problem: str, gold: str, variant: str, *, level: str = "", subject: str = "",
                forced_redirect: int = 0) -> dict:
    return {
        "data_source": "math_meta",
        "prompt": build_math_prompt(problem, variant),
        "problem": problem,
        "gold": gold,
        "reward_model": {"style": "rule", "ground_truth": gold},
        "extra_info": {"problem": problem, "gold": gold, "level": level, "subject": subject,
                       "prompt_variant": variant, "forced_redirect": forced_redirect},
    }


def split_records(train_rows: list[dict], heldout_problems: set[str], *, val_n: int, seed: int,
                  variant: str, level: str | None = None) -> tuple[list[dict], list[dict], dict]:
    """순수 함수(테스트 가능). train_rows: [{problem, solution, level?, type?}].

    ★level: 지정 시(예: "Level 5") row["level"] 이 정확히 일치하는 행만 남긴다 — MATH
    난이도별 학습 세트를 좁히기 위함(build_math_parquet.py 유일한 필터 지점; 다른 곳에
    ad-hoc pandas 필터가 없음을 확인했다).
    """
    seen: set[str] = set()
    kept, n_dup, n_heldout, n_nogold, n_level = [], 0, 0, 0, 0
    for r in train_rows:
        p = str(r["problem"])
        k = norm_problem(p)
        if k in heldout_problems:
            n_heldout += 1
            continue
        if k in seen:
            n_dup += 1
            continue
        if level is not None and str(r.get("level", "")) != level:
            n_level += 1
            continue
        # ★deepscaler/dapo17k 는 이미 순수 gold 문자열을 들고 온다(row["gold"]) —
        #   hendrycks 만 solution 텍스트에서 last_boxed 로 뽑는다.
        g = str(r["gold"]) if "gold" in r else last_boxed(str(r.get("solution", "")))
        if not g:
            n_nogold += 1
            continue
        seen.add(k)
        kept.append(make_record(p, g, variant, level=str(r.get("level", "")),
                                subject=str(r.get("type", ""))))
    rng = random.Random(seed)
    rng.shuffle(kept)
    val, train = kept[:val_n], kept[val_n:]
    stats = {"n_in": len(train_rows), "n_heldout_removed": n_heldout, "n_dup": n_dup,
             "n_nogold": n_nogold, "n_level_excluded": n_level, "n_train": len(train),
             "n_val": len(val)}
    return train, val, stats


def apply_forced_redirect(train: list[dict], *, forced_frac: float, forced_variant: str,
                          seed: int) -> tuple[list[dict], int]:
    """★DEPRECATED(0914): **완전 무작위** 행 선택 — 전체가 다 맞거나 다 틀리는 문제까지
    강제해 신호가 없는 자리도 섞인다. 후속 `apply_forced_redirect_from_rollouts`(mixed-only,
    0<pass_rate<1 인 문제에만 강제)가 정본이다 — 새 발사는 그쪽을 쓴다. 이 함수는
    `tests/test_math_retry_forced.py::test_apply_forced_redirect_...` 회귀만을 위해 남긴다
    (테스트가 이 함수를 직접 호출한다 — 제거하려면 그 테스트를 먼저 mixed-only 로 옮겨야 한다).
    ★forced-redirect 탐색(0914 사전등록 수정): redirect_rate 붕괴로 judgment 항이
    gradient 를 못 받는 문제 대응 — TRAIN 행의 고정 비율 F 를 강제로 math_retry_forced
    변형으로 다시 쓴다(prompt 재빌드 + extra_info.forced_redirect=1). math_meta.py 쪽에서
    이 플래그를 읽어 judgment 항을 undefined(0) 처리하므로 redirect_rate 지표를 오염시키지
    않는다 — 여기서는 단지 데이터에 표시만 한다. val 은 이 함수를 호출하지 않는다(항상 0).
    seed 는 --seed 와 별도 스트림(offset)을 써서 split_records 의 셔플과 상관없이 재현
    가능하게 한다.
    """
    if forced_frac <= 0:
        return train, 0
    rng = random.Random(seed + 97)  # ★split 셔플과 다른 스트림(우연한 상관 방지)
    idx = list(range(len(train)))
    rng.shuffle(idx)
    n_forced = round(forced_frac * len(train))
    forced_idx = set(idx[:n_forced])
    out = []
    for i, r in enumerate(train):
        if i in forced_idx:
            r = dict(r)
            r["prompt"] = build_math_prompt(r["problem"], forced_variant)
            ei = dict(r["extra_info"])
            ei["forced_redirect"] = 1
            ei["prompt_variant"] = forced_variant
            r["extra_info"] = ei
        out.append(r)
    return out, n_forced


def load_group_pass_rates(rollouts_path: str) -> dict[str, float]:
    """math_rollout texts.jsonl(행 = 롤아웃, group_id/problem/r_corr) → {norm_problem: pass_rate}.

    ★같은 group_id 라도 problem 텍스트로 재키잉한다(우리가 매칭할 단위는 problem, build_math_parquet
    의 train pool 은 group_id 를 모른다). 한 problem 에 여러 group_id 가 섞이는 일은 없다고 가정하되,
    섞이면 즉사한다(조용한 평균 오염 방지)."""
    import json as _json
    by_problem: dict[str, list[int]] = {}
    seen_group: dict[str, set] = {}
    with open(rollouts_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = _json.loads(line)
            k = norm_problem(r["problem"])
            by_problem.setdefault(k, []).append(int(r["r_corr"]))
            seen_group.setdefault(k, set()).add(str(r.get("group_id", "")))
    bad = {k: g for k, g in seen_group.items() if len(g) > 1}
    if bad:
        raise RuntimeError(f"[MATH][ROLLOUTS] 같은 문제에 group_id 가 둘 이상: {list(bad)[:5]}")
    return {k: sum(v) / len(v) for k, v in by_problem.items()}


def apply_forced_redirect_from_rollouts(train: list[dict], *, forced_frac: float, forced_variant: str,
                                        seed: int, pass_rates: dict[str, float]) -> tuple[list[dict], dict]:
    """★0914 mixed-only forced-redirect. `pass_rates`(norm_problem→group pass rate)에서 **0<rate<1**
    (mixed — 결과가 흔들리는 문제)인 TRAIN 행에만 forced_redirect=1 을 배정한다(전체가 다 맞거나 다
    틀리는 문제는 redirect 가 옳은지 verify 가 옳은지 신호가 없다). 목표 개수는 forced_frac × TRAIN
    전체 행 수 — mixed 문제가 모자라면 있는 만큼만 강제하고 shortfall 을 보고한다(조용히 덜 강제하지
    않는다). extra_info.group_pass_rate 는 **모든** train 행에 붙는다(mixed 아니어도 트레이너가 로그
    남기도록) — 매칭 안 되면 None."""
    n_total = len(train)
    n_target = round(forced_frac * n_total) if forced_frac > 0 else 0
    keyed = [(i, norm_problem(r["problem"])) for i, r in enumerate(train)]
    mixed_idx = [i for i, k in keyed if pass_rates.get(k) is not None and 0.0 < pass_rates[k] < 1.0]
    rng = random.Random(seed + 97)  # ★split 셔플과 다른 스트림(apply_forced_redirect 와 같은 관례)
    rng.shuffle(mixed_idx)
    forced_idx = set(mixed_idx[:n_target])
    n_forced = len(forced_idx)
    shortfall = max(0, n_target - len(mixed_idx))
    out = []
    for i, r in enumerate(train):
        r = dict(r)
        ei = dict(r["extra_info"])
        gpr = pass_rates.get(norm_problem(r["problem"]))
        ei["group_pass_rate"] = float(gpr) if gpr is not None else None
        if i in forced_idx:
            r["prompt"] = build_math_prompt(r["problem"], forced_variant)
            ei["forced_redirect"] = 1
            ei["prompt_variant"] = forced_variant
        r["extra_info"] = ei
        out.append(r)
    stats = {"n_train": n_total, "n_mixed_found": len(mixed_idx), "n_forced_target": n_target,
             "n_forced": n_forced, "forced_shortfall": shortfall}
    return out, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="math_opt", choices=sorted(MATH_PROMPT_VARIANTS))
    ap.add_argument("--source", default="hendrycks", choices=sorted(SOURCES),
                    help="hendrycks(MATH train, default) | deepscaler | dapo17k(AIME-급 확장, 0914)")
    ap.add_argument("--val_n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    ap.add_argument("--level", default=None,
                    help='e.g. "Level 5" — hendrycks 에만 적용(다른 source 는 level 필드가 없다)')
    ap.add_argument("--forced_frac", type=float, default=0.0,
                    help="fraction of TRAIN rows forced to redirect (math_retry_forced/math_agree_forced), "
                         "excluded from judgment reward/metrics")
    ap.add_argument("--forced_variant", default=None,
                    help="기본: --variant 에서 유도(_FORCED_VARIANT_OF) — math_retry→math_retry_forced, "
                         "math_agree→math_agree_forced. 명시하면 그 값을 그대로 쓴다.")
    ap.add_argument("--forced_from_rollouts", default=None,
                    help="math_rollout texts.jsonl(group_id/problem/r_corr) 경로 — 주어지면 "
                         "forced_redirect 를 mixed(0<pass_rate<1) 문제에만 배정하고 모든 train 행에 "
                         "extra_info.group_pass_rate 를 붙인다(--forced_frac 은 그대로 목표 비율)")
    a = ap.parse_args()

    import pandas as pd
    from datasets import load_dataset

    heldout = load_heldout_problems()
    level = a.level
    if a.source == "hendrycks":
        rows = []
        for cfg in HENDRYCKS_CONFIGS:
            ds = load_dataset(HENDRYCKS, cfg, split="train")
            rows.extend({"problem": r["problem"], "solution": r["solution"],
                         "level": r.get("level", ""), "type": r.get("type", cfg)} for r in ds)
    elif a.source == "deepscaler":
        rows, n_nogold_src = load_deepscaler_rows()
        print(f"[build_math_parquet] source=deepscaler raw={len(rows)} n_nogold={n_nogold_src}")
        if a.level:
            print(f"[build_math_parquet] --level={a.level!r} 무시됨 — deepscaler 는 level 필드가 없다")
        level = None
    elif a.source == "dapo17k":
        rows, n_nogold_src, n_unstripped = load_dapo17k_rows()
        print(f"[build_math_parquet] source=dapo17k raw={len(rows)} n_nogold={n_nogold_src} "
              f"n_unstripped_template={n_unstripped}")
        if a.level:
            print(f"[build_math_parquet] --level={a.level!r} 무시됨 — dapo17k 는 level 필드가 없다")
        level = None
    else:
        raise SystemExit(f"unknown --source {a.source!r}")
    train, val, stats = split_records(rows, heldout, val_n=a.val_n, seed=a.seed, variant=a.variant,
                                      level=level)
    # ★0914b: --forced_variant 미지정이면 --variant 에서 유도한다(_FORCED_VARIANT_OF). --forced_frac
    #   >0 인데 유도할 곳이 없으면(모르는 variant) 조용히 넘어가지 않고 즉사한다.
    forced_variant = resolve_forced_variant(a.forced_variant, a.variant)
    if a.forced_frac > 0 and not forced_variant:
        raise SystemExit(
            f"--forced_frac={a.forced_frac}>0 인데 --variant={a.variant!r} 에 대한 기본 "
            f"forced_variant 가 없다({sorted(_FORCED_VARIANT_OF)}) — --forced_variant 를 직접 넘겨라.")
    if a.forced_from_rollouts:
        pass_rates = load_group_pass_rates(a.forced_from_rollouts)
        train, rstats = apply_forced_redirect_from_rollouts(
            train, forced_frac=a.forced_frac, forced_variant=forced_variant, seed=a.seed,
            pass_rates=pass_rates)
        stats.update(rstats)
        # ★F==0 도 mixed 경로는 suffix 를 붙인다 — 무슨 소스로 만들었는지가 파일명에 남아야 한다.
        suffix = f"_fmix{a.forced_frac:g}"
    else:
        train, n_forced = apply_forced_redirect(train, forced_frac=a.forced_frac,
                                                forced_variant=forced_variant, seed=a.seed)
        stats["n_forced"] = n_forced
        # ★F==0 은 접미사 없음(기존 동작/테스트 보존), F>0 은 파일명에 _f{F} 를 붙여 판별 가능하게 한다.
        suffix = f"_f{a.forced_frac:g}" if a.forced_frac > 0 else ""
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # ★source suffix: hendrycks(기존 소스)는 이름을 바꾸지 않는다 — run_math_arm.sh 의
    #   DATA_TRAIN/DATA_VAL 기본값(math_train_${VARIANT}.parquet, 접미사 없음)과
    #   기존 테스트가 이 이름을 가정한다. deepscaler/dapo17k 는 파일명에 소스를 남긴다.
    src_suffix = "" if a.source == "hendrycks" else f"_{a.source}"
    tp = out / f"math_train_{a.variant}{src_suffix}{suffix}.parquet"
    # ★B8(0914): val 은 forced/mixed 접미사를 달지 않는다 — val 은 절대 강제되지 않으므로
    #   (apply_forced_redirect* 는 train 에만 적용) 접미사가 붙으면 «강제 val 이 따로 있다»는
    #   거짓 신호가 된다. F 가 달라도 같은 variant 의 val 은 한 파일을 공유(run_math_arm.sh 의
    #   DATA_VAL 기본값이 접미사 없는 이 이름을 찾는다).
    vp = out / f"math_val_{a.variant}{src_suffix}.parquet"
    pd.DataFrame(train).to_parquet(tp, index=False)
    pd.DataFrame(val).to_parquet(vp, index=False)
    print(f"[build_math_parquet] variant={a.variant} {stats}")
    print(f"[build_math_parquet] {tp} ({len(train)} rows)\n[build_math_parquet] {vp} ({len(val)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
