#!/usr/bin/env python
r"""math_critique_ig_ruler — 비평의 **정보 이득**(생성 없이 forward 만).

math_critique_resolve_gate 가 «비평을 들고 다시 풀면 오르는가»를 표본으로 재는 동안, 이 자는
같은 비평이 **정답 풀이를 얼마나 더 그럴듯하게 만드는지**를 teacher-forcing 으로 직접 잰다.
생성 표본이 없으니 잡음이 작고(자리당 forward 몇 번), Δ가 0 근처일 때 «효과가 없다»와
«표본이 모자라다»를 가른다.

같은 문제의 **정답 형제** S+ (같은 group_id 의 정답·미잘림 롤아웃 중 **가장 짧은 것**) 에 대해

    IG        = mean_tok log p(S+ | prompt + note(비평))  −  mean_tok log p(S+ | prompt)
    IG_donor  = 같은 것, 단 **다른 문제**의 비평
    IG_minus  = 같은 것, 단 대상이 **그 오답 풀이 S−** 자신

★IG_donor 가 대조다. IG 가 양수여도 IG − IG_donor 가 0 이면 «무슨 말이든 덧붙이면 다음 토큰이
  쉬워진다»(문맥 길이·주의 효과)일 뿐 비평 **내용**의 값이 아니다.
★IG_minus 가 함께 양수면 비평이 정답 쪽이 아니라 **그 문제 전반**(오답 포함) 쪽으로 민다는
  뜻이다 — 즉 원래 오류를 밀어내지 못한다. IG − IG_minus 를 같이 읽어야 «방향»이 읽힌다.
★문맥 조립은 math_critique_resolve_gate.resolve_prompt 를 **그대로 import** 한다 — 그 게이트가
  생성에 쓴 프롬프트와 한 글자라도 달라지면 두 산출물을 나란히 읽을 수 없다.

한 행당 forward 8 번(S+ plain/note/donor/generic/shuffled/masked, S− plain/note; plain 은
캐시 하나씩만 쓴다). 200 행이면 ~1,600 forward.

★네 가지 대조 통제(양성 판정이 반드시 이겨야 하는 것들):
  1. IG_generic  note 자리에 **모든 문제에 똑같은** 일반 힌트("각 단계를 다시 확인하고…")를
     넣는다 — «구체적 비평 내용»이 아니라 «뭔가 조언이 붙었다는 사실» 자체의 효과를 잰다.
  2. IG_shuffled 비평 자신의 단어를 시드 고정으로 무작위 섞는다 — 길이·어휘는 같고 **내용만
     파괴**된다. IG_shuffled 가 IG 와 다르지 않으면 «단어 뭉치가 있다»만으로 오른 것이다.
  3. IG_masked   비평 안의 모든 숫자와 \boxed{...} 를 "[…]" 로 가린다 — 비평이 (의도치 않게)
     구체적 수치를 흘려서 오른 것인지 감사한다. leak_flag(가려질 내용이 있었는가)와
     (IG−IG_masked) 의 Spearman, 그리고 가림으로 IG 가 0.02 나트/토큰 넘게 바뀌는 행의 비율을
     같이 본다.
  4. held-out gold  S+(라벨 1)와 S−(라벨 0) 에 대한 IG 를 한 덩어리로 모은 Mann-Whitney AUC —
     비평이 오답보다 정답을 더 올리는 방향인지, 짝 비교(ig_direction) 말고 순위로도 본다.
     IG_delta_var·저분산 행 비율(|IG−IG_donor|<0.01)은 RLPR 식 «질문이 밋밋해서 뭘 줘도 IG 가
     그대로인» 행을 골라내는 진단이다.

사용(예):
  python scripts/local/math_critique_ig_ruler.py \
      --rollouts /hdd_data/seungpil/scratch/eval/mathL5_q3i2507_opt_b8k/texts.jsonl \
      --critiques /hdd_data/seungpil/scratch/eval/critique_resolve_s1/critiques.jsonl \
      --model_path <hf> --variant math_opt --out_dir /hdd_data/seungpil/scratch/eval/critique_ig_s1 \
      [--resolve_out /hdd_data/seungpil/scratch/eval/critique_resolve_s1/per_rollout.jsonl]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Optional, Sequence


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import math_geometry_probe as G  # noqa: E402  (spearman — 정의를 한 곳에)
import math_ruler_pivot as P  # noqa: E402  (Job/_enc/hf_forward_factory/auc 재사용)
from math_cited_site_gate import bootstrap_ci  # noqa: E402
from math_critique_resolve_gate import assign_donors, resolve_prompt  # noqa: E402
from src.training.math_meta import boxed_spans  # noqa: E402  (마스킹용 균형 \boxed 스캐너)

_NAN = float("nan")
# ★한 행의 여덟 forward. (대상, note 종류) — 이름이 곧 결과 dict 의 키다. plain 은 S+/S− 마다
#   하나씩만 쓰고 note/donor/generic/shuffled/masked 가 그 위에 얹힌다(«plain 캐시 유지»).
JOB_KINDS = ("plus_plain", "plus_note", "plus_donor", "plus_generic", "plus_shuffled",
            "plus_masked", "minus_plain", "minus_note")

# ── 통제 1: 고정 일반 힌트(모든 행에 바이트 동일) ──────────────────────────────────
GENERIC_CRITIQUE = ("Re-check each step carefully, verify the algebra and the case analysis, "
                    "and re-derive the result from the problem statement.")
MASK_TOKEN = "[…]"          # "[…]"
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

# ── 통과 규칙 상수 ───────────────────────────────────────────────────────────────
PASS_MASK_DELTA = 0.02            # |IG − IG_masked| 이 이보다 작아야 "안정"
PASS_MASK_FRAC = 0.80             # 안정 행 비율의 하한
PASS_LOW_VAR_DONOR = 0.01         # |IG − IG_donor| 이 이보다 작으면 RLPR 식 저분산 프롬프트


# ── 형제 고르기 ─────────────────────────────────────────────────────────────────
def roll_ids(rolls: Sequence[dict]) -> list[str]:
    """math_cited_site_gate.select_wrong_rollouts 와 **같은 규약**의 roll_id 를 전 행에 매긴다
    ("<group_id>#<파일 내 인덱스>") — 비평 파일의 roll_id 로 원 롤아웃을 되찾기 위한 열쇠."""
    return [f"{r['group_id']}#{i}" for i, r in enumerate(rolls)]


def shortest_correct_sibling(rolls: Sequence[dict], group_id: str,
                             exclude_id: Optional[str] = None) -> Optional[dict]:
    """같은 문제의 정답·미잘림 롤아웃 중 **가장 짧은 것**(동률이면 파일 순서 먼저).
    ★왜 가장 짧은가: teacher-forcing 평균 로그확률은 길이에 민감하고(긴 풀이일수록 쉬운 토큰이
      섞여 평균이 올라간다), 짧은 정답이 «그 문제를 푸는 최소 서술»에 가장 가깝다. 무엇보다
      행마다 같은 규칙으로 골라야 IG 가 문제 사이에서 비교 가능하다."""
    ids = roll_ids(rolls)
    best = None
    for rid, r in zip(ids, rolls):
        if r["group_id"] != group_id or not int(r.get("r_corr", 0)) or r.get("truncated"):
            continue
        if exclude_id is not None and rid == exclude_id:
            continue
        if best is None or len(r["text"]) < len(best["text"]):
            best = {**r, "roll_id": rid}
    return best


def shuffle_critique(critique: str, rng: random.Random) -> str:
    """비평의 단어를 시드 고정으로 무작위 섞는다(공백 분할·재결합) — 어휘·길이는 그대로,
    **순서(=내용)만** 파괴한 통제. 한 단어짜리 비평은 그대로 돌려준다."""
    words = (critique or "").split()
    out = words[:]
    rng.shuffle(out)
    return " ".join(out)


def critique_leak_flag(critique: str) -> int:
    """이 비평에 «가려질 만한 것»(숫자 또는 \\boxed{...})이 있는가 — mask_critique 가 실제로
    뭔가를 바꾸는지의 사전 신호. math_meta.critique_leaks 와 달리 **답과 무관하게** 숫자
    존재 자체를 본다(마스킹 대상 유무 플래그이지 정답 누출 판정이 아니다)."""
    c = critique or ""
    return int(bool(_NUM_RE.search(c)) or bool(boxed_spans(c)))


def mask_critique(critique: str) -> str:
    r"""비평 안의 모든 숫자와 균형 \boxed{...} 를 MASK_TOKEN 으로 가린다. \boxed{...} 안의
    숫자는 그 박스 전체가 이미 가려지므로 중복 치환하지 않는다."""
    text = critique or ""
    boxed = [(s, e) for _, s, e in boxed_spans(text)]

    def _inside_boxed(pos: int) -> bool:
        return any(a <= pos < b for a, b in boxed)

    nums = [(m.start(), m.end()) for m in _NUM_RE.finditer(text) if not _inside_boxed(m.start())]
    spans = sorted(boxed + nums)
    out, i = [], 0
    for a, b in spans:
        if a < i:
            continue                        # 겹치는 구간은 앞선 치환으로 이미 처리됨
        out.append(text[i:a])
        out.append(MASK_TOKEN)
        i = b
    out.append(text[i:])
    return "".join(out)


def build_rows(rolls: Sequence[dict], crits: Sequence[dict], rng: random.Random,
               *, limit: int = 0) -> list[dict]:
    """비평 행 + 원 롤아웃 + 정답 형제 + donor 비평 배정. 형제가 없는 행은 버린다."""
    by_id = {rid: r for rid, r in zip(roll_ids(rolls), rolls)}
    kept = []
    for c in crits:
        if int(c.get("leaked", 0)) or not (c.get("critique") or "").strip():
            continue
        src = by_id.get(c["roll_id"])
        if src is None:
            continue
        sib = shortest_correct_sibling(rolls, src["group_id"], exclude_id=c["roll_id"])
        if sib is None:
            continue
        crit = c["critique"].strip()
        kept.append({"roll_id": c["roll_id"], "group_id": src["group_id"],
                     "problem": src["problem"], "gold": src["gold"],
                     "critique": crit,
                     "generic_critique": GENERIC_CRITIQUE,
                     "shuffled_critique": shuffle_critique(crit, rng),
                     "masked_critique": mask_critique(crit),
                     "leak_flag": critique_leak_flag(crit),
                     "s_plus": sib["text"], "s_plus_id": sib["roll_id"],
                     "s_minus": src["text"]})
    if limit:
        kept = kept[:limit]
    for r, d in zip(kept, assign_donors(kept, rng)):
        r["donor_roll_id"] = kept[d]["roll_id"] if d is not None else None
        r["donor_critique"] = kept[d]["critique"] if d is not None else None
    return [r for r in kept if r["donor_critique"]]


# ── forward 요청 조립 ───────────────────────────────────────────────────────────
def build_jobs(tok, rows: Sequence[dict], variant: str, *, max_len: int = 8192) -> dict:
    """행마다 여덟 Job(JOB_KINDS). 각 Job 은 «prompt(+note) + 대상 풀이» 의 토큰열이고
    lp_spans 는 대상 풀이 구간 하나다 — 평균은 그 구간 토큰 수로 나눈다.

    ★어느 한 Job 이라도 max_len 을 넘으면 그 **행 전체**를 버린다. hf_forward_factory 는
      왼쪽을 자르므로, 한 조건만 잘리면 평균의 분모·문맥이 조건마다 달라져 IG 가 «비평 효과»가
      아니라 «잘림 차이»를 잰다."""
    jobs: list[P.Job] = []
    index: list[dict] = []
    n_long = 0
    for r in rows:
        heads = {
            "plus_plain": (resolve_prompt(tok, variant, r["problem"], None), r["s_plus"]),
            "plus_note": (resolve_prompt(tok, variant, r["problem"], r["critique"]), r["s_plus"]),
            "plus_donor": (resolve_prompt(tok, variant, r["problem"], r["donor_critique"]),
                           r["s_plus"]),
            "plus_generic": (resolve_prompt(tok, variant, r["problem"], r["generic_critique"]),
                             r["s_plus"]),
            "plus_shuffled": (resolve_prompt(tok, variant, r["problem"], r["shuffled_critique"]),
                              r["s_plus"]),
            "plus_masked": (resolve_prompt(tok, variant, r["problem"], r["masked_critique"]),
                            r["s_plus"]),
            "minus_plain": (resolve_prompt(tok, variant, r["problem"], None), r["s_minus"]),
            "minus_note": (resolve_prompt(tok, variant, r["problem"], r["critique"]),
                           r["s_minus"]),
        }
        built = {}
        for kind in JOB_KINDS:
            head, target = heads[kind]
            hi, ti = P._enc(tok, head), P._enc(tok, target)
            if not ti or len(hi) + len(ti) > max_len:
                built = {}
                break
            built[kind] = (hi + ti, len(hi), len(ti))
        if not built:
            n_long += 1
            continue
        rec = {k: v for k, v in r.items() if k not in ("s_plus", "s_minus")}
        rec["jobs"] = {}
        rec["n_tok"] = {}
        for kind in JOB_KINDS:
            ids, n_head, n_tgt = built[kind]
            rec["jobs"][kind] = len(jobs)
            rec["n_tok"][kind] = n_tgt
            jobs.append(P.Job(ids, lp_spans=[(n_head, len(ids))]))
        index.append(rec)
    return {"jobs": jobs, "index": index, "n_dropped_long": n_long}


def ig_records(res: Sequence[dict], built: dict) -> list[dict]:
    """forward 결과 → 행별 IG 레코드. mean_tok log p = lp 합 / 대상 토큰 수."""
    recs = []
    for it in built["index"]:
        mean = {}
        for kind in JOB_KINDS:
            lp = res[it["jobs"][kind]]["lp"][0]
            n = it["n_tok"][kind]
            mean[kind] = (float(lp) / n) if (n and lp is not None
                                             and math.isfinite(float(lp))) else _NAN
        rec = {k: v for k, v in it.items() if k not in ("jobs", "n_tok")}
        rec.update({f"mlp_{k}": mean[k] for k in JOB_KINDS})
        rec["n_tok_plus"] = it["n_tok"]["plus_plain"]
        rec["n_tok_minus"] = it["n_tok"]["minus_plain"]
        rec["ig"] = mean["plus_note"] - mean["plus_plain"]
        rec["ig_donor"] = mean["plus_donor"] - mean["plus_plain"]
        rec["ig_generic"] = mean["plus_generic"] - mean["plus_plain"]
        rec["ig_shuffled"] = mean["plus_shuffled"] - mean["plus_plain"]
        rec["ig_masked"] = mean["plus_masked"] - mean["plus_plain"]
        rec["ig_minus"] = mean["minus_note"] - mean["minus_plain"]
        rec["ig_advantage"] = rec["ig"] - rec["ig_donor"]        # 내용 대조 1: donor
        rec["ig_vs_generic"] = rec["ig"] - rec["ig_generic"]     # 내용 대조 2: generic hint
        rec["ig_vs_shuffled"] = rec["ig"] - rec["ig_shuffled"]   # 내용 대조 3: shuffled
        rec["ig_vs_masked"] = rec["ig"] - rec["ig_masked"]       # 누출 감사용 Δ
        rec["ig_direction"] = rec["ig"] - rec["ig_minus"]        # 정답 쪽 − 오답 쪽
        recs.append(rec)
    return recs


def held_out_auc(recs: Sequence[dict]) -> float:
    """S+(라벨 1) 의 IG 와 S−(라벨 0) 의 IG_minus 를 한 덩어리로 모은 Mann-Whitney AUC —
    비평이 오답 쪽보다 정답 쪽을 더 올리는 **순위**인지(짝 비교인 ig_direction 과 달리 풀링).
    한 쪽 클래스가 비면(recs 가 비었으면) NaN."""
    y = [1] * len(recs) + [0] * len(recs)
    s = [r["ig"] for r in recs] + [r["ig_minus"] for r in recs]
    return P.auc(y, s)


def load_resolve(path: Optional[str]) -> dict:
    """math_critique_resolve_gate 의 per_rollout.jsonl → {roll_id: {p_crit, p_blind, ...}}.
    디렉터리를 줘도 그 안의 per_rollout.jsonl 을 읽는다."""
    if not path:
        return {}
    p = Path(path)
    if p.is_dir():
        p = p / "per_rollout.jsonl"
    if not p.exists():
        raise SystemExit(f"[ig] --resolve_out 을 찾을 수 없다: {p}")
    out = {}
    for line in p.open():
        r = json.loads(line)
        out[r["roll_id"]] = r
    return out


def _variance(vals: Sequence[float]) -> float:
    """모집단 분산(유한값만) — 요약 통계용, 표본 n 보정 없이 그대로 쓴다."""
    v = [float(x) for x in vals if isinstance(x, (int, float)) and math.isfinite(float(x))]
    if len(v) < 2:
        return _NAN
    m = sum(v) / len(v)
    return sum((x - m) ** 2 for x in v) / len(v)


def _ci_excludes_zero_positive(ci: Optional[dict]) -> bool:
    """부트스트랩 CI 가 0 을 제외하고(부호 일관) 평균이 양수인가."""
    ci = ci or {}
    lo, hi, mean = ci.get("lo"), ci.get("hi"), ci.get("mean")
    if not all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in (lo, hi, mean)):
        return False
    return (lo > 0 or hi < 0) and mean > 0


def gate_pass(summ: dict) -> bool:
    """PASS ⟺ 세 짝지은 Δ(IG−IG_donor, IG−IG_generic, IG−IG_shuffled) 의 CI 가 모두 0 을
    제외하고 평균 > 0 ∧ |IG−IG_masked| < PASS_MASK_DELTA 인 행이 전체의 ≥ PASS_MASK_FRAC."""
    deltas_ok = all(_ci_excludes_zero_positive(summ.get(k))
                    for k in ("ig_advantage", "ig_vs_generic", "ig_vs_shuffled"))
    frac = summ.get("frac_masked_delta_small", _NAN)
    mask_ok = isinstance(frac, (int, float)) and math.isfinite(frac) and frac >= PASS_MASK_FRAC
    return bool(deltas_ok and mask_ok)


def summarize(recs: Sequence[dict], resolve: Optional[dict] = None, *, seed: int = 0,
              n_boot: int = 2000) -> dict:
    """행별 IG → 요약(부트스트랩 CI + 네 통제 + resolve 결과와의 Spearman)."""
    resolve = resolve or {}
    n = len(recs)
    d_donor = [r["ig_advantage"] for r in recs]
    d_masked = [r["ig_vs_masked"] for r in recs]
    leak_flags = [r["leak_flag"] for r in recs]
    summ = {
        "n_rows": n,
        "ig": bootstrap_ci([r["ig"] for r in recs], seed=seed, n_boot=n_boot),
        "ig_donor": bootstrap_ci([r["ig_donor"] for r in recs], seed=seed + 1, n_boot=n_boot),
        "ig_generic": bootstrap_ci([r["ig_generic"] for r in recs], seed=seed + 10,
                                    n_boot=n_boot),
        "ig_shuffled": bootstrap_ci([r["ig_shuffled"] for r in recs], seed=seed + 11,
                                     n_boot=n_boot),
        "ig_masked": bootstrap_ci([r["ig_masked"] for r in recs], seed=seed + 12,
                                   n_boot=n_boot),
        "ig_minus": bootstrap_ci([r["ig_minus"] for r in recs], seed=seed + 2, n_boot=n_boot),
        # 짝지은 Δ(내용 대조 3종) — 양성 판정이 반드시 이겨야 하는 통제.
        "ig_advantage": bootstrap_ci(d_donor, seed=seed + 3, n_boot=n_boot),
        "ig_vs_generic": bootstrap_ci([r["ig_vs_generic"] for r in recs], seed=seed + 13,
                                       n_boot=n_boot),
        "ig_vs_shuffled": bootstrap_ci([r["ig_vs_shuffled"] for r in recs], seed=seed + 14,
                                        n_boot=n_boot),
        "ig_direction": bootstrap_ci([r["ig_direction"] for r in recs], seed=seed + 4,
                                      n_boot=n_boot),
        "frac_ig_gt_donor": (sum(1 for r in recs if r["ig"] > r["ig_donor"]) / n
                             if n else _NAN),
        # 누출 감사: 마스킹이 IG 를 얼마나 흔드는가 + 그게 «가려질 게 있었나»와 관계있는가.
        "spearman_ig_vs_masked_delta_vs_leak_flag": G.spearman(d_masked, leak_flags),
        "frac_masked_delta_small": (sum(1 for d in d_masked if math.isfinite(d)
                                        and abs(d) < PASS_MASK_DELTA) / n if n else _NAN),
        # held-out gold: S+ 대 S− 풀링 AUC + IG>IG_minus 비율.
        "auc_ig_vs_minus": held_out_auc(recs),
        "frac_ig_gt_minus": (sum(1 for r in recs if r["ig"] > r["ig_minus"]) / n
                             if n else _NAN),
        # RLPR 식 저분산 프롬프트 진단(도너 대비 IG 가 거의 안 움직이는 행).
        "ig_delta_var": _variance(d_donor),
        "frac_low_var_donor": (sum(1 for d in d_donor if math.isfinite(d)
                                   and abs(d) < PASS_LOW_VAR_DONOR) / n if n else _NAN),
    }
    if resolve:
        pairs = [(r["ig"], resolve[r["roll_id"]].get("p_crit"),
                  resolve[r["roll_id"]].get("p_crit", _NAN)
                  - resolve[r["roll_id"]].get("p_blind", _NAN))
                 for r in recs if r["roll_id"] in resolve]
        summ["n_joined_resolve"] = len(pairs)
        summ["spearman_ig_vs_p_crit"] = G.spearman([a for a, _, _ in pairs],
                                                   [b for _, b, _ in pairs])
        summ["spearman_ig_vs_crit_minus_blind"] = G.spearman([a for a, _, _ in pairs],
                                                             [c for _, _, c in pairs])
        summ["spearman_igadv_vs_crit_minus_blind"] = G.spearman(
            [r["ig_advantage"] for r in recs if r["roll_id"] in resolve],
            [c for _, _, c in pairs])
    summ["pass"] = int(gate_pass(summ))
    return summ


def _f(v) -> str:
    if isinstance(v, dict):
        return f"{_f(v.get('mean'))} [{_f(v.get('lo'))}, {_f(v.get('hi'))}] (n={v.get('n')})"
    if isinstance(v, float):
        return "nan" if not math.isfinite(v) else f"{v:.4f}"
    return str(v)


def to_markdown(summ: dict, recs: Sequence[dict] = (), *, n_show: int = 20) -> str:
    keys = ["n_rows", "ig", "ig_donor", "ig_generic", "ig_shuffled", "ig_masked", "ig_minus",
            "ig_advantage", "ig_vs_generic", "ig_vs_shuffled", "ig_direction",
            "frac_ig_gt_donor", "frac_ig_gt_minus", "auc_ig_vs_minus",
            "spearman_ig_vs_masked_delta_vs_leak_flag", "frac_masked_delta_small",
            "ig_delta_var", "frac_low_var_donor",
            "n_joined_resolve", "spearman_ig_vs_p_crit",
            "spearman_ig_vs_crit_minus_blind", "spearman_igadv_vs_crit_minus_blind", "pass"]
    lines = ["## math_critique_ig_ruler — 비평의 정보 이득(teacher-forcing)", "",
             "| metric | value |", "|---|---|"]
    lines += [f"| {k} | {_f(summ.get(k))} |" for k in keys if k in summ]
    lines += ["", "IG = mean_tok log p(S+ | prompt+note) − mean_tok log p(S+ | prompt). "
                  "`ig_advantage` = IG − IG_donor(내용 대조: donor 비평), "
                  "`ig_vs_generic` = IG − IG_generic(내용 대조: 고정 일반 힌트), "
                  "`ig_vs_shuffled` = IG − IG_shuffled(내용 대조: 단어 섞기), "
                  "`ig_direction` = IG − IG_minus(방향). "
                  f"**PASS** 규칙: 세 짝지은 Δ(donor/generic/shuffled) CI 가 0 제외·평균>0 "
                  f"∧ |IG−IG_masked|<{PASS_MASK_DELTA} 인 행 ≥ {PASS_MASK_FRAC:.0%}.", ""]
    if recs:
        lines += [f"### 행별 (앞 {n_show})", "",
                  "| roll_id | n_tok(S+) | IG | IG_donor | IG_generic | IG_shuffled | "
                  "IG_masked | IG_minus | leak_flag |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in list(recs)[:n_show]:
            lines.append(f"| {r['roll_id']} | {r['n_tok_plus']} | {_f(r['ig'])} | "
                         f"{_f(r['ig_donor'])} | {_f(r['ig_generic'])} | "
                         f"{_f(r['ig_shuffled'])} | {_f(r['ig_masked'])} | "
                         f"{_f(r['ig_minus'])} | {r.get('leak_flag')} |")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True, help="math_rollout 산출물 texts.jsonl")
    ap.add_argument("--critiques", required=True,
                    help="math_critique_resolve_gate 산출물 critiques.jsonl")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--variant", default="math_opt")
    ap.add_argument("--resolve_out", default=None,
                    help="같은 게이트의 per_rollout.jsonl(또는 그 디렉터리) — Spearman 용")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--max_len", type=int, default=8192)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="행 수 상한(0=전부)")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)

    rolls = [json.loads(l) for l in open(a.rollouts)]
    crits = [json.loads(l) for l in open(a.critiques)]
    rows = build_rows(rolls, crits, rng, limit=a.limit)
    print(f"[ig] 비평 {len(crits)}개 → 정답 형제·donor 가 있는 행 {len(rows)}개", flush=True)
    if not rows:
        raise SystemExit("[ig] 행이 없다 — 비평 파일에 누출 아닌 비평이, 롤아웃에 정답 형제가 있는지 보라.")

    forward, tok, _ = P.hf_forward_factory(a.model_path, batch_size=a.batch_size,
                                           max_len=a.max_len)
    built = build_jobs(tok, rows, a.variant, max_len=a.max_len)
    print(f"[ig] forward {len(built['jobs'])}개 (행 {len(built['index'])}, "
          f"너무 긴 행 {built['n_dropped_long']}개 제외)", flush=True)
    res = forward(built["jobs"], [])
    recs = ig_records(res, built)

    resolve = load_resolve(a.resolve_out)
    summ = summarize(recs, resolve, seed=a.seed, n_boot=a.n_boot)
    summ.update({"model_path": a.model_path, "variant": a.variant, "rollouts": a.rollouts,
                 "critiques": a.critiques, "seed": a.seed,
                 "n_dropped_long": built["n_dropped_long"], "n_forwards": len(built["jobs"])})

    with (out / "per_rollout_ig.jsonl").open("w") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "ig_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2))
    (out / "ig_summary.md").write_text(to_markdown(summ, recs))
    print(to_markdown(summ, recs))
    print(f"[out] {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
