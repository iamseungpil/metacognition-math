r"""MATH_META — 수학 무대의 «분리 가능한 메타 스팬 보상» 팔 명세 + 행 계산 + 텔레메트리.

★왜 별도 모듈인가 (2026-09-14, cd9). Countdown 은 오라클(nums/target/witness/decoy)이
있어 메타의 «좋음»을 규칙으로 잴 수 있었다. 수학엔 그 오라클이 없다 — 있는 것은
(1) math_verify 정답 여부, (2) `scripts/local/math_sites.py` 가 **같은 자리 개입**으로
오프라인에서 캐낸 판단 라벨(best_decision ∈ verify/redirect/tie), (3) 외부 프로브
점수(아직 없음, `set_meta_scorer` 로 꽂는다) 뿐이다. 그래서 `countdown_rewards.py`
의 여덟 팔·오라클 항을 건드리지 않고, 수학은 이 파일 한 곳에서 정의한다.

구조는 COUNTDOWN_6ARM 과 같다 — 배치당 한 번 도는 프리패스(`verl_sdc.
_compute_math_arm_stash`)가 여기 함수들로 행을 계산해 스태시를 채우고, 얇은 보상
헤드(`verl_sdc.math_arm_reward`)는 그것을 읽기만 한다. 메타 스팬 항은 시퀀스 보상에
**넣지 않고** 스태시에 따로 두었다가 `verl_sdc._math_add_meta_region_advantage` 가
GRPO 어드밴티지 계산 뒤 메타 토큰 구간에만 얹는다(Countdown §13-b CHK_REGION 과
같은 경로). 답 스팬은 정답 보상만 받는다 — «메타 부분과 정답을 다르게 채점».

팔 (MATH_ARM_SPECS):
  M_G0    math_plain 프롬프트, 정답만.                      (세금 0 기준선)
  M_G1    math_opt 프롬프트, 정답만. 메타 허용, 메타 보상 없음. (허가의 세금)
  M_JUDGE math_opt; 메타 스팬 = 판단 일치 항(아래).
  M_PROBE math_opt; 메타 스팬 = 외부 프로브 점수(`set_meta_scorer`; 기본 0, 한 번 경고).
  M_RAND  M_JUDGE 와 같되 라벨을 **배치 안에서 섞는다** — 대조군. 이 팔이 M_JUDGE 와
          같은 성적이면 판단 항은 내용이 아니라 «메타 스팬에 잡음을 얹은 효과»다.
          ★감사 4(0914): 섞는 단위는 행이 아니라 **프롬프트 그룹(uid)** 이다 — uid→라벨
          표를 순열해 같은 그룹의 8 롤아웃은 같은(섞인) 라벨을 받는다. tie/무라벨 그룹은
          풀 밖. 행 단위로 섞으면 그룹 안 라벨이 갈려 «그룹 중심화 뒤 잡음»의 분산이
          M_JUDGE 와 달라져 대조군이 아니게 된다.

판단 일치 항 (M_JUDGE):
  +1  메타를 냈고 decision == best_decision
  −1  메타를 냈고 decision 이 best_decision 의 반대(verify↔redirect)
   0  tie / decision 없음 / 미발화 / 문제가 라벨표에 없음 / n_blocks>1(형식 위반)
  가중치 MATH_JUDGE_W(기본 0.5). 라벨표는 MATH_JUDGE_LABELS(json, problem→best_decision).
  ★감사 3: 위 «0» 은 보상 0 이 아니라 **항이 정의되지 않음**이다. 행마다 `meta_defined`
  (0/1)를 같이 내고, 그룹 중심화(`verl_sdc._math_add_meta_region_advantage`)는 member=
  meta_defined 로 미정의 행을 평균에서도 배제하고 중심화 값도 0 으로 둔다.
"""
from __future__ import annotations

import json
import os
import random
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence

from src.training import countdown_rewards as _cdr

SPEC_VERSION = "math-meta-0914"

# ── 팔 명세 ───────────────────────────────────────────────────────────────────
# `meta_term`: None(메타 항 없음) / "judge" / "probe" / "judge_shuffled".
# `require_meta`: 발화율 중단 규칙을 적용하는가 — 메타 항이 있는 팔만 True 다
#   (M_G1 은 허용만 하므로 발화 0 이 정상, Countdown OPT 계열과 같은 규약).
MATH_ARM_SPECS: dict[str, dict] = {
    "M_G0": {"variant": "math_plain", "meta_term": None, "require_meta": False,
             "note": "메타 지시문 없음. 정답만. 세금 0 기준선."},
    "M_G1": {"variant": "math_opt", "meta_term": None, "require_meta": False,
             "note": "메타 허용·비요구. 정답만. 허가 문장의 세금을 잰다."},
    "M_JUDGE": {"variant": "math_opt", "meta_term": "judge", "require_meta": True,
                "note": "메타 스팬에 판단 일치 항(±1·MATH_JUDGE_W), 답 스팬은 정답만."},
    "M_PROBE": {"variant": "math_opt", "meta_term": "probe", "require_meta": True,
                "note": "메타 스팬에 외부 프로브 점수. set_meta_scorer 로 꽂는다."},
    "M_RAND": {"variant": "math_opt", "meta_term": "judge_shuffled", "require_meta": True,
               "note": "M_JUDGE 의 라벨을 배치 안에서 섞은 대조군."},
}

_META_TERM_ARMS = frozenset(a for a, s in MATH_ARM_SPECS.items() if s["meta_term"])
_OPPOSITE = {"verify": "redirect", "redirect": "verify"}


def require_arm(arm: str) -> dict:
    """fail-closed. 조용한 기본값 금지 — 팔이 틀리면 다섯 잡이 같은 보상으로 돈다."""
    if arm not in MATH_ARM_SPECS:
        raise ValueError(
            f"[MATH] arm={arm!r} 가 MATH_ARM_SPECS {sorted(MATH_ARM_SPECS)} 에 없다 — "
            "런처가 ++algorithm.math_arm 을 넘겼는지 확인하라.")
    return MATH_ARM_SPECS[arm]


def judge_weight() -> float:
    """MATH_JUDGE_W (기본 0.5). ★워커에서 읽으므로 verl_sdc.main 의 Ray env 목록에 실려야 한다."""
    v = os.environ.get("MATH_JUDGE_W")
    return 0.5 if v is None else float(v)


# ── 채점: math_verify (scripts/local/math_rollout.py 와 **같은 함수**를 쓴다) ──────
def grade_math(pred_text: str, gold: str) -> int:
    """math_verify parse+verify. 예외는 0 — 단 아래 자가검사가 «조용한 오채점»을 막는다."""
    from math_verify import parse, verify  # noqa: PLC0415
    try:
        return int(verify(parse(str(gold)), parse(pred_text)))
    except Exception:
        return 0


def selftest_math_verify() -> None:
    """math_verify timeout 래퍼가 워커 스레드에서 정답을 조용히 오답으로 만드는 함정
    (`scripts/patch_math_verify.py`)이 있다 — 깨져 있으면 즉사한다."""
    cases = [("\\boxed{42}", "42", 1), ("\\boxed{\\frac{1}{2}}", "0.5", 1),
             ("\\boxed{7}", "42", 0)]
    got = [grade_math(p, g) for p, g, _ in cases]
    want = [e for *_, e in cases]
    if got != want:
        raise RuntimeError(
            f"math_verify 자가검사 실패: got={got} want={want} — "
            "scripts/patch_math_verify.py 를 먼저 적용하라(조용한 오채점 방지).")


def last_boxed(text: str) -> str:
    r"""마지막 \boxed{...} 안의 문자열(중괄호 균형; 없거나 안 닫히면 "")."""
    t = text or ""
    i = t.rfind("\\boxed{")
    if i < 0:
        return ""
    j, depth = i + len("\\boxed{"), 1
    while j < len(t) and depth:
        depth += (t[j] == "{") - (t[j] == "}")
        j += 1
    return t[i + len("\\boxed{"): j - 1].strip() if depth == 0 else ""


# ── 판단 라벨표 ──────────────────────────────────────────────────────────────
def norm_problem(p: str) -> str:
    """라벨표 키. 공백 정규화만 — 문제 텍스트는 parquet 과 math_sites 가 같은 원문을 쓴다."""
    return re.sub(r"\s+", " ", str(p or "")).strip()


def load_judge_labels(path: str | None = None) -> dict[str, str]:
    """MATH_JUDGE_LABELS(json) → {norm_problem: best_decision}. 없으면 빈 표(항 전부 0).

    형식 두 가지를 받는다: {problem: decision} 사전, 또는 math_sites 의 sites.jsonl 을
    모은 [{problem, best_decision}, ...] 리스트. 값이 verify/redirect/tie 가 아니면 버린다.
    """
    p = path if path is not None else os.environ.get("MATH_JUDGE_LABELS", "")
    if not p:
        return {}
    with open(p, encoding="utf-8") as fh:
        raw = json.load(fh)
    items = raw.items() if isinstance(raw, dict) else \
        ((r.get("problem"), r.get("best_decision")) for r in raw)
    out = {}
    for k, v in items:
        if v in ("verify", "redirect", "tie") and k:
            out[norm_problem(k)] = v
    return out


def check_labels_cover(parquet_path: str, labels_path: str) -> dict:
    """★감사 2: 라벨표 키가 학습 parquet 의 문제와 교집합이 0 이면 판단 항은 전 행 0 —
    M_JUDGE/M_RAND 가 M_G1 과 바이트 동일해진다(무효 레버). 발사 전에 즉사시킨다.
    반환 {n_train, n_labels, n_cover}. `problem` 컬럼이 없으면 extra_info.problem 을 쓴다."""
    import pandas as pd  # noqa: PLC0415
    df = pd.read_parquet(parquet_path)
    if "problem" in df.columns:
        probs = [norm_problem(x) for x in df["problem"].tolist()]
    elif "extra_info" in df.columns:
        probs = [norm_problem((e or {}).get("problem")) for e in df["extra_info"].tolist()]
    else:
        raise RuntimeError(f"[MATH][LABELS] {parquet_path}: problem/extra_info 컬럼이 없다: {list(df.columns)}")
    probs_set = {x for x in probs if x}
    labels = load_judge_labels(labels_path)
    n_cover = len(probs_set & set(labels))
    st = {"n_train": len(probs_set), "n_labels": len(labels), "n_cover": n_cover}
    if n_cover == 0:
        raise RuntimeError(
            f"[MATH][LABELS] 라벨 키가 학습 문제와 교집합이 없다: n_train={st['n_train']} "
            f"n_labels={st['n_labels']} n_cover=0 (parquet={parquet_path}, labels={labels_path}). "
            "math_sites 의 문제 텍스트와 build_math_parquet 의 문제 텍스트가 같은 원문인지 확인하라.")
    return st


def judgment_term(emitted, decision, best_decision) -> float:
    """진리표 — 모듈 docstring 참조."""
    if not _cdr._bool01(emitted) or decision not in _OPPOSITE or best_decision not in _OPPOSITE:
        return 0.0
    return 1.0 if decision == best_decision else -1.0


# ── 외부 프로브 스코어러(M_PROBE) ────────────────────────────────────────────
_SCORER: dict = {"fn": None, "warned": False}


def set_meta_scorer(fn: Callable[[Sequence[Mapping]], Sequence[float]] | None) -> None:
    """fn(rows) -> [float]*len(rows). None 이면 기본(0, 한 번 경고)으로 되돌린다."""
    _SCORER["fn"] = fn
    _SCORER["warned"] = False


def score_meta_probe(rows: Sequence[Mapping]) -> list[float]:
    fn = _SCORER["fn"]
    if fn is None:
        if not _SCORER["warned"]:
            _SCORER["warned"] = True
            print("[MATH][PROBE][WARN] meta scorer 미등록 — M_PROBE 의 메타 항은 전부 0 이다. "
                  "set_meta_scorer(fn) 으로 꽂아라(이 경고는 한 번만 찍힌다).", flush=True)
        return [0.0] * len(rows)
    out = [float(x) for x in fn(rows)]
    if len(out) != len(rows):
        raise RuntimeError(f"[MATH][PROBE] scorer 가 {len(out)} 개를 돌려줬다(행 {len(rows)}).")
    return out


# ── 행 계산 ─────────────────────────────────────────────────────────────────
_BOXED_RE = re.compile(r"\\boxed\s*\{")


def parse_row(text: str, gold: str, problem: str) -> dict:
    """한 롤아웃의 원재료. 메타는 form="math"(decision 선택)."""
    m = _cdr.parse_meta(text or "", "math")
    raw = m.get("raw") or ""
    return {
        "text": text or "", "gold": str(gold), "problem": problem,
        "r_corr": grade_math(text or "", gold),
        "emitted": int(m["emitted"]), "n_blocks": int(m["n_blocks"]),
        "meta_start": m["start"], "meta_end": m["end"],
        "confidence": m["confidence"], "decision": m["decision"], "body": m["body"],
        # ★누출 가드: 메타 안에 \boxed 가 있으면 «접근 평가»가 아니라 답이다.
        "boxed_in_meta": int(bool(m["n_blocks"]) and bool(_BOXED_RE.search(raw))),
        "n_chars": len(text or ""),
    }


def _permute_group_labels(best: list, keys: Sequence, rng: random.Random) -> list:
    """★감사 4: 그룹(uid)→라벨 표를 순열한다. 풀 = verify/redirect 라벨이 있는 그룹만
    (tie/None 은 그대로). 같은 그룹 안에 다른 문제·라벨이 섞여 있으면 즉사(규약 위반)."""
    by_key: dict = {}
    for k, b in zip(keys, best):
        by_key.setdefault(k, set()).add(b)
    mixed = {k for k, v in by_key.items() if len(v) > 1}
    if mixed:
        raise RuntimeError(f"[MATH][RAND] 같은 uid 그룹에 라벨(문제)이 둘 이상 섞였다: {sorted(map(str, mixed))[:5]}")
    pool = [k for k, v in by_key.items() if next(iter(v)) in _OPPOSITE]
    vals = [next(iter(by_key[k])) for k in pool]
    rng.shuffle(vals)
    table = dict(zip(pool, vals))
    return [table.get(k, b) for k, b in zip(keys, best)]


def compute_rows(texts: Sequence[str], golds: Sequence[str], problems: Sequence[str],
                 arm: str, *, labels: Mapping[str, str] | None = None,
                 rng: random.Random | None = None, uids: Sequence | None = None) -> list[dict]:
    """배치 전체의 행 + 메타 항. 반환 행마다 `answer_total`(시퀀스 보상)과
    `meta_val`(메타 스팬 전용 항), `meta_defined`(항이 정의된 행 = 그룹 중심화 member)이
    분리돼 있다.

    M_RAND: uid→best_decision **표**를 순열한다(`uids` 없으면 정규화 문제 텍스트가 그룹 키).
    라벨 분포는 그대로고 그룹↔라벨 대응만 깨진다 — 판단 «내용»만 제거한 대조군.
    ★감사 6: n_blocks>1 은 형식 위반 — 그 행의 메타 항은 0 이고 정의되지 않은 것으로 둔다.
    """
    spec = require_arm(arm)
    labels = labels or {}
    rows = [parse_row(t, g, p) for t, g, p in zip(texts, golds, problems)]
    best = [labels.get(norm_problem(r["problem"])) for r in rows]
    if spec["meta_term"] == "judge_shuffled":
        keys = [str(u) for u in uids] if uids is not None else [norm_problem(r["problem"]) for r in rows]
        if len(keys) != len(rows):
            raise RuntimeError(f"[MATH][RAND] uids 길이 {len(keys)} != 행 {len(rows)}")
        before = sum(1 for b in best if b in _OPPOSITE)
        best = _permute_group_labels(best, keys, rng or random.Random(0))
        n_lab = sum(1 for b in best if b in _OPPOSITE)
        print(f"[MATH][RAND] uid→label 표 순열: n_groups={len(set(keys))} "
              f"n_groups_in_pool={len({k for k, b in zip(keys, best) if b in _OPPOSITE})} "
              f"n_lab={n_lab} (before={before})", flush=True)
    for r, b in zip(rows, best):
        r["best_decision"] = b
        r["multi_block"] = int(int(r.get("n_blocks", 0)) > 1)
        r["judge"] = 0.0 if r["multi_block"] else judgment_term(r["emitted"], r["decision"], b)
        r["answer_total"] = float(r["r_corr"])
    if spec["meta_term"] in ("judge", "judge_shuffled"):
        w = judge_weight()
        for r in rows:
            r["meta_val"] = w * r["judge"]
            r["meta_defined"] = int(bool(_cdr._bool01(r["emitted"]) and not r["multi_block"]
                                        and r["decision"] in _OPPOSITE and r["best_decision"] in _OPPOSITE))
    elif spec["meta_term"] == "probe":
        for r, s in zip(rows, score_meta_probe(rows)):
            ok = bool(_cdr._bool01(r["emitted"]) and not r["multi_block"])
            r["meta_val"] = float(s) if ok else 0.0
            r["meta_defined"] = int(ok)
    else:
        for r in rows:
            r["meta_val"] = 0.0
            r["meta_defined"] = 0
    return rows


def meta_char_spans(row: Mapping) -> list[tuple[int, int]]:
    """메타 항이 얹힐 문자 구간(첫 블록만 — parse_meta 가 첫 블록을 채점하므로 같은 구간)."""
    s, e = row.get("meta_start"), row.get("meta_end")
    if not _cdr._bool01(row.get("emitted", 0)) or s is None or e is None or e <= s:
        return []
    return [(int(s), int(e))]


# ── 텔레메트리·중단 ────────────────────────────────────────────────────────────
ABORT_RULES = {
    "emit_rate":        {"op": "<", "thr": 0.2, "meta_arms_only": True,
                         "why": "발화가 무너지면 메타 스팬 항이 얹힐 토큰이 없다"},
    "boxed_in_meta":    {"op": ">", "thr": 0.02, "meta_arms_only": False,
                         "why": "메타가 답을 담으면 메타 보상이 정답 보상의 사본이 된다"},
    # ★감사 1: 이 통계는 «최빈 메타 문장 하나의 점유율»(바닥 1/n_emitted)이라 0.05 는 n_emitted≤20
    #   이면 무조건 발화하는 규칙이었다. Countdown 과 같은 0.5, 표본 30 미만이면 «못 쟀다»(missing).
    "boilerplate_rate": {"op": ">", "thr": 0.5, "meta_arms_only": False, "min_n": ("n_emitted", 30),
                         "why": "최빈 메타 문장이 발화 행의 절반을 넘으면 판단이 아니라 상투구다"},
    # ★감사 6: 블록 둘 이상은 형식 위반(첫 블록만 채점되므로 나머지는 공짜 토큰).
    "multi_block_rate":  {"op": ">", "thr": 0.10, "meta_arms_only": False,
                         "why": "메타 블록이 둘 이상인 행이 10% 를 넘으면 형식이 무너진 것이다"},
    # ★감사 7: 사전등록 «acc < M_G0 − 1pp». 문턱은 런처가 MATH_ACC_FLOOR 로 준다 — 미설정이면
    #   이 규칙은 참여하지 않는다(check_abort 가 건너뛴다). 설정되면 3-스텝 연속 규칙에 참여.
    "acc":              {"op": "<", "thr": None, "env_thr": "MATH_ACC_FLOOR", "meta_arms_only": False,
                         "why": "정확도가 M_G0 − 1pp 아래로 3 스텝 연속이면 메타 항이 해롭다"},
}


def telemetry(rows: Sequence[Mapping], *, arm: str, step) -> dict:
    n = max(1, len(rows))
    emitted = [r for r in rows if _cdr._bool01(r.get("emitted", 0))]
    n_dec = sum(1 for r in emitted if r.get("decision") in _OPPOSITE)
    n_lab = sum(1 for r in emitted if r.get("best_decision") in _OPPOSITE and r.get("decision") in _OPPOSITE)
    bp = _cdr.boilerplate_rate(rows, form="math")
    rep = {
        "step": int(step), "arm": arm, "n_rows": len(rows),
        "acc": sum(int(r["r_corr"]) for r in rows) / n,
        "emit_rate": len(emitted) / n,
        "n_blocks_mean": sum(int(r.get("n_blocks", 0)) for r in rows) / n,
        "multi_block_rate": sum(1 for r in rows if int(r.get("n_blocks", 0)) > 1) / n,
        "decision_rate": n_dec / max(1, len(emitted)),
        # 라벨·결정이 둘 다 있는 발화 행 중 일치 비율(없으면 NaN — 0 으로 읽히면 안 된다)
        "judge_match": (sum(1 for r in emitted if r.get("judge", 0) > 0) / n_lab
                        if n_lab else float("nan")),
        "boxed_in_meta": sum(int(r.get("boxed_in_meta", 0)) for r in rows) / n,
        "boilerplate_rate": bp.get("boilerplate_rate"),
        "n_emitted": bp.get("n_emitted", 0),
        "len_mean": sum(int(r.get("n_chars", 0)) for r in rows) / n,
        "meta_val_abs_mean": sum(abs(float(r.get("meta_val", 0.0))) for r in rows) / n,
    }
    return rep


def format_tel(rep: Mapping) -> str:
    def _f(x):
        try:
            return f"{float(x):.3f}"
        except (TypeError, ValueError):
            return "nan"
    return (f"[MATH][TEL] step={rep['step']} arm={rep['arm']} acc={_f(rep['acc'])} "
            f"emit={_f(rep['emit_rate'])} n_blocks_mean={_f(rep['n_blocks_mean'])} "
            f"multi_block={_f(rep.get('multi_block_rate'))} "
            f"decision_rate={_f(rep['decision_rate'])} judge_match={_f(rep['judge_match'])} "
            f"boxed_in_meta={_f(rep['boxed_in_meta'])} len_mean={_f(rep['len_mean'])} "
            f"boilerplate={_f(rep['boilerplate_rate'])} n_emitted={rep.get('n_emitted', 0)}")


def check_abort(rep: Mapping, *, arm: str) -> list[dict]:
    """위반 목록(빈 리스트 = 통과). 지표가 NaN 이면 «못 쟀다» — 죽이지 않고 missing 으로."""
    spec = require_arm(arm)
    out = []
    for name, rule in ABORT_RULES.items():
        if rule["meta_arms_only"] and not spec.get("require_meta", False):
            continue
        thr = rule["thr"]
        if rule.get("env_thr"):
            ev = os.environ.get(rule["env_thr"])
            if ev is None or ev == "":
                continue                      # 문턱 미설정 → 규칙 불참(감사 7)
            thr = float(ev)
        v = rep.get(name)
        if v is None or not _cdr._finite(v):
            out.append({"metric": name, "status": "missing", "value": v})
            continue
        if rule.get("min_n"):
            nkey, nmin = rule["min_n"]
            if int(rep.get(nkey, 0) or 0) < nmin:
                out.append({"metric": name, "status": "missing", "value": float(v),
                            "note": f"{nkey}={rep.get(nkey)}<{nmin}"})
                continue
        bad = (float(v) < thr) if rule["op"] == "<" else (float(v) > thr)
        if bad:
            out.append({"metric": name, "status": "abort", "value": float(v),
                        "thr": thr, "why": rule["why"]})
    return out


def update_streak(streak: dict, key: str, hits: Sequence[Mapping], *, patience: int | None = None) -> bool:
    """연속 위반 카운터 — patience(기본 countdown_rewards.get_abort_patience(), 3)에 닿으면 True.
    호출자(verl_sdc)가 True 를 받으면 `_CountdownAbort` 를 던진다(rc 75 경로 재사용)."""
    if patience is None:
        patience = _cdr.get_abort_patience()
    if hits:
        streak[key] = streak.get(key, 0) + 1
    else:
        streak[key] = 0
    return streak.get(key, 0) >= patience


# ── non_tensor_batch 컬럼 읽기(flat → extra_info 폴백) ────────────────────────
def nt_col(nt: Mapping, name: str) -> list:
    """★verl 0.7.1 async 경로는 flat 컬럼을 gen_batch 로 pop 해 버린다 — `extra_info`
    안의 같은 값으로 폴백한다(COUNTDOWN `_col` 과 같은 이유). 둘 다 있고 어긋나면 즉사."""
    v = nt.get(name, None)
    ei = nt.get("extra_info", None)
    alt = None
    if ei is not None:
        try:
            cand = [(e or {}).get(name, None) for e in list(ei)]
            if not all(x is None for x in cand):
                alt = cand
        except Exception:
            alt = None
    if v is None:
        if alt is None:
            raise RuntimeError(
                f"[MATH] non_tensor_batch 에 '{name}' 컬럼이 없다(extra_info 폴백도 실패). "
                f"사용 가능한 키: {sorted(nt.keys())}. scripts/local/build_math_parquet.py 로 "
                "빌드한 parquet 인지 확인하라.")
        return list(alt)
    v = list(v)
    if alt is not None:
        bad = sum(1 for a, b in zip(alt, v) if str(a) != str(b))
        if bad:
            raise RuntimeError(
                f"[MATH] '{name}' 이 flat 컬럼과 extra_info 에서 다르다({bad}/{len(v)} 행) — "
                "어느 쪽으로 학습했는지 사후 확정이 불가능해지므로 즉사한다.")
    return v
