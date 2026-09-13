#!/usr/bin/env python
"""math_sites — 수학판 «같은 자리» 인과 측정: 메타 하나의 기여 Δ̂ 를 재고 라벨한다.

배경(0912). Countdown 에서 «좋은 메타»의 정답 라벨은 오라클(지목한 수가 해를
살리는가)이었다. 수학엔 그 오라클이 없으므로 **행동 결과 라벨**을 쓴다 — 이 메타를
끼워 넣으면 정답률이 실제로 오르는가.

    Δ̂ = p̂(정답 | 앞부분 + 메타) − p̂(정답 | 앞부분)

세 조건을 같은 자리에서 각각 K 개 이어 쓴다:
    nometa — 앞부분 그대로 이어 씀(기준선)
    meta   — 앞부분 + "\\n<meta>\\n" 를 심어 **그 자리에서** 메타를 쓰게 만든다
    donor  — 앞부분 + **다른 문제**에서 뽑은 완성된 메타 블록
★donor 가 핵심 대조다. 이게 없으면 «메타 텍스트가 끼어든 효과»를 «이 메타의 내용
효과»로 오독한다(Countdown gen_continuations 의 donor 모드와 같은 역할).

자리 선별: `p̂(nometa)` 가 0 이나 1 인 자리는 Δ̂ 가 구조적으로 0 이라 버린다
(math500 실측: 문제의 52% 가 8/8 정답, 24.8% 가 8/8 오답 — 위치 단위로 잘라야
움직일 여지가 있는 자리가 나온다).

라벨:
    SAVE   p̂(nometa) <= 0.25  ∧  Δ̂ >= +0.25
    DERAIL p̂(nometa) >= 0.75  ∧  Δ̂ <= -0.25
    NEUTRAL 나머지

★0914 `--site_source own_meta` — 자리를 우리가 자르지 않고 **모델이 스스로 멈춘 지점**
(허용판 롤아웃의 첫 `<meta>`)에서 잡는다. 앞부분 = `<meta>` 태그 직전까지, `own` 모드 =
모델이 실제로 쓴 블록을 **그대로** 이어 붙인 것(원 생성 문맥과 바이트 동일). 같은 자리에
verify/redirect 를 심어 정답 판단을 얻고, 모델 자신의 결정이 그것과 맞는지(`own_judgment_
correct`)를 잰다 — 사전등록 §1 «자기 멈춤 자리»의 «언제 판단» 정확도.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.metacot.math_meta_prompt import MATH_PROMPT_VARIANTS  # noqa: E402
from src.training.countdown_rewards import parse_meta  # noqa: E402

# ★parse_meta(countdown_rewards._META_BLOCK) 와 같은 대소문자 무시 — 파서가 «메타»로 세는
#   블록과 자리를 자르는 블록이 어긋나면 own 자리와 own_decision 이 서로 다른 블록을 본다.
_META_BLOCK = re.compile(r"<meta>.*?</meta>", re.S | re.I)
SITE_SOURCES = ("cut", "own_meta")
# 자리 원천별 «롤아웃이 생성된 프롬프트» — 이어 쓰기 문맥은 이것과 바이트 동일해야 한다.
# cut 은 math_plain 롤아웃, own_meta 는 허용판(math_opt) 롤아웃에서 나온다.
SOURCE_VARIANT = {"cut": "math_plain", "own_meta": "math_opt"}


def _grade(text: str, gold: str) -> int:
    from math_verify import parse, verify
    try:
        return int(verify(parse(str(gold)), parse(text)))
    except Exception:
        return 0


def _selftest() -> None:
    if [_grade("\\boxed{42}", "42"), _grade("\\boxed{7}", "42")] != [1, 0]:
        raise RuntimeError("math_verify 자가검사 실패 — 조용한 오채점 방지를 위해 즉사한다.")


def cut_points(text: str, n_cuts: int, rng: random.Random) -> list[int]:
    """줄 경계에서 자리 후보를 뽑는다(수학엔 Countdown 의 «시도 경계»가 없다).

    응답 앞 10%·뒤 20% 는 버린다 — 너무 이르면 아무 진전이 없고, 너무 늦으면
    결과가 이미 정해져 Δ̂ 가 구조적으로 0 이 된다.
    """
    bounds = [m.end() for m in re.finditer(r"\n", text)]
    lo, hi = int(len(text) * 0.10), int(len(text) * 0.80)
    cand = [b for b in bounds if lo <= b <= hi]
    if not cand:
        return []
    rng.shuffle(cand)
    return sorted(cand[:n_cuts])


# ★0913 «판단 조건부 반사실»: 같은 자리에서 두 판단을 **심어** 이어 쓴다. 어느 쪽이 더 잘
#   되는지가 그 자리의 정답 판단이고, 모델이 스스로 쓴 메타의 결정이 그것과 맞으면 «옳은
#   메타인지»다. 정답 여부와 분리된 메타 내용 채점 — 결과 고정 검사를 정의상 통과한다.
SEED_VERIFY = ("<meta>\nconfidence: 0.8\nThe current approach is sound. I will push it through "
               "carefully and check the result.\ndecision: verify\n</meta>\n")
SEED_REDIRECT = ("<meta>\nconfidence: 0.2\nThe current approach is not working. I will abandon it "
                 "and solve the problem with a completely different method.\ndecision: redirect\n</meta>\n")
ALL_MODES = ("nometa", "meta", "donor", "verify", "redirect", "own")
DEFAULT_MODES = {"cut": "nometa,meta,donor", "own_meta": "nometa,own,verify,redirect"}


def build_fed(mode: str, prefix: str, donor_meta: str | None, own_meta: str | None = None) -> str:
    """모델에게 «이미 이렇게 썼다»고 먹일 텍스트."""
    if mode == "nometa":
        return prefix
    if mode == "own":
        # ★rstrip 도 개행 추가도 하지 않는다 — 앞부분은 `<meta>` 직전에서 끊었고 블록은 모델이
        #   쓴 그대로이므로, 그냥 이어 붙인 것이 원 롤아웃의 «</meta> 까지»와 바이트 동일하다.
        if not own_meta:
            raise ValueError("own 모드인데 모델 자신의 메타가 비었다(cut 자리엔 own 이 없다).")
        return prefix + own_meta
    if mode == "verify":
        return prefix.rstrip() + "\n" + SEED_VERIFY
    if mode == "redirect":
        return prefix.rstrip() + "\n" + SEED_REDIRECT
    if mode == "meta":
        return prefix.rstrip() + "\n<meta>\n"
    if mode == "donor":
        if not donor_meta:
            raise ValueError("donor 모드인데 기증 메타가 비었다.")
        return prefix.rstrip() + "\n" + donor_meta + "\n"
    raise ValueError(f"unknown mode: {mode!r}")


def _gen_prompt(tok, msgs) -> str:
    """math_rollout.chat 과 같은 생성 프롬프트 — 롤아웃이 만들어진 문맥과 바이트 단위로 같아야
    «같은 자리» 측정이 된다."""
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def gen_request(tok, variant: str, problem: str, fed: str) -> str:
    """생성 프롬프트(system+user, enable_thinking=False) 뒤에 fed 를 **그대로** 잇는다.

    ★0913 수리: continue_final_message 를 쓰지 않는다. Qwen3.5 템플릿은 assistant 본문의 끝
      공백을 지운다 — 자리는 줄 경계라 앞부분이 늘 개행으로 끝나므로 (a) vLLM 토크나이저
      래퍼는 «마지막 메시지가 안 보인다»며 거부해 3회 rc 1, (b) transformers 는 통과시키되
      개행을 **조용히 삭제**해 자리가 한 줄 옮겨진다(실측: 렌더 522 vs 523 바이트). 롤아웃이
      생성된 문맥은 «생성 프롬프트 + 텍스트» 그 자체이므로 그걸 그대로 잇는다.
    math_site_emission_eval 도 이 함수를 써서 같은 문맥을 만든다(그래야 «같은 자리»다).
    """
    # ★0914 감사 수리(버그10): math_rollout 은 build_math_prompt(problem.strip()) 로 생성했다.
    #   같은 함수를 써야 앞·뒤 공백이 있는 문제에서도 프롬프트가 바이트 단위로 같다.
    from src.metacot.math_meta_prompt import build_math_prompt
    return _gen_prompt(tok, build_math_prompt(problem, variant)) + fed


def select_sources(rolls: list[dict], movable_only: bool, keep=None) -> list[dict]:
    """한 문제당 롤아웃 하나만 자리 원천으로(같은 문제가 자리를 독점하지 않게). 잘린 것 제외.

    keep: 원천 후보 술어(own_meta 는 «메타를 낸 롤아웃»만). ★movable 판정은 keep 과 무관하게
      그룹의 **모든** 롤아웃으로 한다 — 발화한 것만으로 p 를 재면 «발화 행 성공률» 편향이 섞인다.

    ★0914 파일럿 교훈(movable_only): 무작위 자리의 92.5%는 이미 결판나 있어(K=8) 판단 대조가
      무승부 93.5%. 원 롤아웃 그룹 정답률이 0<p<1 인 문제만 남긴다. own_meta 원천에도 같은
      필터를 건다 — 자기 멈춤 자리도 결판난 문제에선 Δ̂·판단 라벨이 구조적으로 0/tie 다.
    """
    movable_groups = None
    if movable_only:
        acc: dict[str, list] = {}
        for r in rolls:
            acc.setdefault(r["group_id"], []).append(int(r["r_corr"]))
        movable_groups = {g for g, v in acc.items() if 0 < sum(v) < len(v)}
        print(f"[sites] movable_only: 문제 {len(movable_groups)}/{len(acc)} (0<p<1)", flush=True)
    seen, srcs = set(), []
    for r in rolls:
        if r["group_id"] in seen or r.get("truncated"):
            continue
        if movable_groups is not None and r["group_id"] not in movable_groups:
            continue
        if keep is not None and not keep(r):
            continue
        seen.add(r["group_id"])
        srcs.append(r)
    return srcs


def own_meta_sites(r: dict, n_max: int, min_prefix_chars: int = 1) -> list[dict]:
    """롤아웃 하나에서 모델이 스스로 낸 메타 자리(앞 n_max 개, 보통 하나)를 뽑는다.

    prefix = `<meta>` 태그 **직전**까지(태그 미포함), own_meta = 블록 전체(`</meta>` 까지).
    own_decision/own_confidence 는 parse_meta(form="math") — 수학 메타의 93~97% 가 decision
    줄이 없으므로 결정은 None 일 수 있고, 그 자리는 판단 정확도 분모에서 빠진다.
    ★min_prefix_chars: 앞부분이 그보다 짧은 자리(대개 «메타 먼저» 롤아웃 — Qwen3.5 허용판
      실측 370/450 이 prefix 0자)는 버린다. 아직 접근이 없으니 «내 접근이 옳은가»라는 판단도,
      심는 verify/redirect(«current approach»)도 뜻이 없다. 0 을 주면 전부 받는다.
    rel_pos = 자리의 상대 위치(문자 기준) — cut 의 10~80% 창을 own 에는 강제하지 않되 기록해 둔다.
    """
    out = []
    for m in _META_BLOCK.finditer(r["text"]):
        if m.start() < min_prefix_chars:
            continue
        block = m.group(0)
        got = parse_meta(block, form="math")
        out.append({"site_id": f"{r['group_id']}@{m.start()}", "problem": r["problem"],
                    "gold": r["gold"], "prefix": r["text"][:m.start()],
                    "rel_pos": m.start() / max(1, len(r["text"])),
                    "own_meta": block, "own_decision": got["decision"],
                    "own_confidence": got["confidence"]})
        if len(out) >= n_max:
            break
    return out


def make_sites(srcs: list[dict], source: str, cuts_per_rollout: int, max_sites: int,
               rng: random.Random, min_prefix_chars: int = 1) -> list[dict]:
    sites: list[dict] = []
    for r in srcs:
        if source == "cut":
            new = [{"site_id": f"{r['group_id']}@{c}", "problem": r["problem"],
                    "gold": r["gold"], "prefix": r["text"][:c]}
                   for c in cut_points(r["text"], cuts_per_rollout, rng)]
        elif source == "own_meta":
            new = own_meta_sites(r, cuts_per_rollout, min_prefix_chars)
        else:
            raise ValueError(f"unknown site_source: {source!r}")
        for s in new:
            sites.append(s)
            if len(sites) >= max_sites:
                return sites
    return sites


def own_judgment_correct(own_decision, best_decision):
    """모델 자신의 결정 == 정답 판단. tie/판단 없음/결정 없음이면 None(분모에서 뺀다)."""
    if own_decision is None or best_decision not in ("verify", "redirect"):
        return None
    return int(own_decision == best_decision)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True,
                    help="cut: math_plain 롤아웃 texts.jsonl / own_meta: 허용판(opt) 롤아웃")
    ap.add_argument("--site_source", choices=SITE_SOURCES, default="cut",
                    help="cut=줄 경계 무작위 자리(기본) / own_meta=모델이 스스로 낸 첫 <meta> 자리")
    ap.add_argument("--min_prefix_chars", type=int, default=1,
                    help="own_meta 전용: 앞부분이 이보다 짧은(메타 먼저) 자리를 버린다. 0=전부")
    ap.add_argument("--donor_rollouts", default=None,
                    help="기증 메타를 캘 롤아웃(math_new/opt). 없으면 donor 모드 생략")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--cuts_per_rollout", type=int, default=2)
    ap.add_argument("--max_sites", type=int, default=400)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--gpu_util", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--movable_only", action="store_true",
                    help="★0914 파일럿 교훈: 무작위 자리의 92.5%%는 이미 결판나 있어(K=8) 판단 대조가 "
                         "무승부 93.5%%. 원 롤아웃 그룹 정답률이 0<p<1 인 문제에서만 자리를 뽑는다.")
    ap.add_argument("--modes", default=None,
                    help="쉼표 구분. nometa,meta,donor,verify,redirect,own 중 (donor 는 기증 롤아웃 "
                         "필요, own 은 own_meta 원천 전용). 기본: cut=nometa,meta,donor / "
                         "own_meta=nometa,own,verify,redirect")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    want = [m for m in (a.modes or DEFAULT_MODES[a.site_source]).split(",") if m]
    bad = set(want) - set(ALL_MODES)
    if bad:
        raise SystemExit(f"unknown modes: {sorted(bad)} (choose from {ALL_MODES})")
    if "own" in want and a.site_source != "own_meta":
        raise SystemExit("own 모드는 --site_source own_meta 에서만 뜻이 있다.")
    if "nometa" not in want:
        raise SystemExit("nometa(기준선)가 빠지면 Δ̂ 를 잴 수 없다.")
    base_variant = SOURCE_VARIANT[a.site_source]

    _selftest()
    rng = random.Random(a.seed)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rolls = [json.loads(l) for l in open(a.rollouts)]
    keep = None
    if a.site_source == "own_meta":
        # ★자기 멈춤 자리: 한 문제에서 «메타를 낸» 롤아웃을 원천으로 골라야 한다. 그룹의 첫
        #   롤아웃이 메타 없는 것이면 그 문제가 통째로 빠지므로, 메타 있는 것만 후보로 받는다
        #   (movable 판정은 그룹 전체로 — select_sources 의 keep 참조).
        keep = lambda r: bool(_META_BLOCK.search(r["text"]))  # noqa: E731
        print(f"[sites] own_meta: 메타를 낸 롤아웃 {sum(1 for r in rolls if keep(r))}/{len(rolls)}개",
              flush=True)
    srcs = select_sources(rolls, a.movable_only, keep)
    rng.shuffle(srcs)
    sites = make_sites(srcs, a.site_source, a.cuts_per_rollout, a.max_sites, rng, a.min_prefix_chars)
    n_meta_first = 0
    if a.site_source == "own_meta":
        n_meta_first = sum(1 for r in srcs if _META_BLOCK.search(r["text"]).start() < a.min_prefix_chars)
        print(f"[sites] own_meta: 메타-먼저(prefix<{a.min_prefix_chars}자) 원천 {n_meta_first}개 제외",
              flush=True)
    print(f"[sites] 자리 {len(sites)}개 (문제 {len(srcs)}개에서, source={a.site_source})",
          flush=True)

    donors: list[tuple[str, str]] = []    # (원천 problem 텍스트, 메타 블록)
    if a.donor_rollouts and "donor" in want:
        for l in open(a.donor_rollouts):
            rr = json.loads(l)
            m = _META_BLOCK.search(rr["text"])
            # ★0914 감사 수리(버그6): \boxed 가 든 기증 블록은 채점기에 답을 흘린다 → 제외.
            if m and "\\boxed" not in m.group(0):
                donors.append((rr["problem"], m.group(0)))
        print(f"[sites] 기증 메타 {len(donors)}개", flush=True)
    modes = [m for m in want if m != "donor" or donors]

    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model_path, dtype="bfloat16", seed=a.seed,
              gpu_memory_utilization=a.gpu_util,
              max_model_len=a.max_tokens + 4096, enforce_eager=True)
    tok = llm.get_tokenizer()

    reqs, meta_ix = [], []
    for si, s in enumerate(sites):
        # ★기증 메타는 **다른 문제**에서만 — 같은 문제면 내용 누출이다.
        dm = None
        if donors:
            # ★0913 감사 수리: 예전 검사(앞부분에 문자열 포함 여부)는 같은 문제의 다른 롤아웃에서
            #   온 메타를 통과시켰다(내용 누출). 원천 문제 텍스트가 다른 것만 받는다.
            for _ in range(20):
                src_problem, cand = donors[rng.randrange(len(donors))]
                if src_problem != s["problem"]:
                    dm = cand
                    break
        s["donor_meta"] = dm
        for mode in modes:
            if mode == "donor" and not dm:
                continue
            # ★0914 감사 수리(버그5): 예전엔 meta 모드만 강제 프롬프트(math_new)를 썼다. 그러면
            #   Δ̂ = p(meta) − p(nometa) 가 «태그 삽입 효과»와 «시스템 프롬프트 차이»를 섞는다.
            #   "\n<meta>\n" 심기만으로 블록이 강제되므로 모든 모드가 같은 프롬프트를 쓴다.
            variant = base_variant
            fed = build_fed(mode, s["prefix"], dm, s.get("own_meta"))
            reqs.append(gen_request(tok, variant, s["problem"], fed))
            meta_ix.append((si, mode))

    print(f"[sites] 요청 {len(reqs)}개 x K={a.k}", flush=True)
    outs = llm.generate(reqs, SamplingParams(n=a.k, temperature=1.0, top_p=1.0,
                                             max_tokens=a.max_tokens, seed=a.seed))

    agg: dict[tuple, list] = {}
    rows = []
    for (si, mode), o in zip(meta_ix, outs):
        s = sites[si]
        for x in o.outputs:
            # ★0914 감사 수리(버그6): 채점은 «앞부분 + 이어쓰기»만. 심은 블록(donor/own/seed)이
            #   \boxed 를 담고 있어도 점수에 못 들어간다. meta 모드의 자기 블록은 x.text 안에 있다.
            c = _grade(s["prefix"] + x.text, s["gold"])
            agg.setdefault((si, mode), []).append(c)
            rows.append({"site_id": s["site_id"], "mode": mode, "r_corr": c,
                         "cont": x.text, "truncated": int(x.finish_reason == "length")})

    recs = []
    for si, s in enumerate(sites):
        p = {m: (sum(v) / len(v)) for (i, m), v in agg.items() if i == si}
        if "nometa" not in p:
            continue
        # ★own_meta 원천에선 «모델이 실제 쓴 블록»(own)이 메타 조건이다 — 심은 "<meta>\n" 이
        #   아니라 그 블록의 기여로 SAVE/DERAIL 을 붙인다. meta 도 own 도 없는 실행(판단 파일럿)은 Δ̂=0.
        if "meta" not in p:
            p["meta"] = p.get("own", p["nometa"])
        d = p["meta"] - p["nometa"]
        dd = (p.get("donor") - p["nometa"]) if "donor" in p else None
        do = (p["own"] - p["nometa"]) if "own" in p else None
        if p["nometa"] <= 0.25 and d >= 0.25:
            lab = "SAVE"
        elif p["nometa"] >= 0.75 and d <= -0.25:
            lab = "DERAIL"
        else:
            lab = "NEUTRAL"
        pv, pr = p.get("verify"), p.get("redirect")
        jud = None
        if pv is not None and pr is not None:
            jud = "redirect" if pr - pv >= 0.25 else ("verify" if pv - pr >= 0.25 else "tie")
        recs.append({"site_id": s["site_id"], "problem": s["problem"], "gold": s["gold"],
                     "prefix": s["prefix"], "donor_meta": s.get("donor_meta"),
                     "site_source": a.site_source, "rel_pos": s.get("rel_pos"),
                     "p_nometa": p["nometa"], "p_meta": p["meta"],
                     "p_donor": p.get("donor"), "delta": d, "delta_donor": dd,
                     "p_own": p.get("own"), "delta_own": do,
                     "own_meta": s.get("own_meta"), "own_decision": s.get("own_decision"),
                     "own_confidence": s.get("own_confidence"),
                     "own_judgment_correct": own_judgment_correct(s.get("own_decision"), jud),
                     "p_verify": pv, "p_redirect": pr, "best_decision": jud,
                     "label": lab,
                     "movable": int(0.0 < p["nometa"] < 1.0)})

    with (out / "sites.jsonl").open("w") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out / "continuations.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(recs)
    lab = {k: sum(1 for r in recs if r["label"] == k) for k in ("SAVE", "DERAIL", "NEUTRAL")}
    summ = {
        "n_sites": n, "k": a.k, "modes": modes, "site_source": a.site_source,
        "movable_rate": sum(r["movable"] for r in recs) / max(1, n),
        "labels": lab,
        "save_rate": lab["SAVE"] / max(1, n), "derail_rate": lab["DERAIL"] / max(1, n),
        "mean_delta": sum(r["delta"] for r in recs) / max(1, n),
        "mean_delta_donor": (sum(r["delta_donor"] for r in recs if r["delta_donor"] is not None)
                             / max(1, sum(1 for r in recs if r["delta_donor"] is not None))),
        "mean_p_nometa": sum(r["p_nometa"] for r in recs) / max(1, n),
    }
    # ★판단 반사실 텔레메트리: 전환 심기가 실제로 다른 접근을 내는가(파일럿 판정 지표).
    jd = [r for r in recs if r["best_decision"] is not None]
    if jd:
        _sw = re.compile(r"\b(different|instead|alternative|another (?:approach|method|way)|"
                         r"let'?s try|re-?think|start over)\b", re.I)
        def _switch_rate(mode):
            xs = [r for r in rows if r["mode"] == mode]
            return sum(1 for r in xs if _sw.search(r["cont"][:400])) / max(1, len(xs))
        summ.update({
            "n_judged": len(jd),
            "best_decision": {k: sum(1 for r in jd if r["best_decision"] == k)
                              for k in ("verify", "redirect", "tie")},
            "mean_p_verify": sum(r["p_verify"] for r in jd) / len(jd),
            "mean_p_redirect": sum(r["p_redirect"] for r in jd) / len(jd),
            "switch_phrase_rate_redirect": _switch_rate("redirect"),
            "switch_phrase_rate_verify": _switch_rate("verify"),
            "switch_phrase_rate_nometa": _switch_rate("nometa"),
        })
    if a.site_source == "own_meta":
        summ.update(own_meta_summary(recs))
        summ["n_meta_first_dropped"] = n_meta_first
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2))
    for k, v in summ.items():
        print(f"  {k:18s} {v}")
    print(f"[out] {out}", flush=True)
    return 0


def own_meta_summary(recs: list[dict]) -> dict:
    """자기 멈춤 자리 요약 — «언제 판단»의 정확도(사전등록 §1 마지막 줄).

    own_judgment_acc: 결정을 쓴 자리 중 정답 판단(verify/redirect, tie 제외)과 맞은 비율.
    own_decision_rate: 자리 중 decision 줄을 쓴 비율(수학 메타는 대개 없다 — 분모 크기 경고용).
    delta_own: p(own) − p(nometa) 평균 — 모델이 실제로 쓴 블록의 기여.
    """
    n = len(recs)
    jc = [r["own_judgment_correct"] for r in recs if r.get("own_judgment_correct") is not None]
    do = [r["delta_own"] for r in recs if r.get("delta_own") is not None]
    rp = [r["rel_pos"] for r in recs if r.get("rel_pos") is not None]
    return {
        "own_decision_rate": sum(1 for r in recs if r.get("own_decision")) / max(1, n),
        "mean_rel_pos_own": (sum(rp) / len(rp)) if rp else None,
        "n_own_judged": len(jc),
        "own_judgment_acc": (sum(jc) / len(jc)) if jc else None,
        "delta_own": (sum(do) / len(do)) if do else None,
    }


if __name__ == "__main__":
    sys.exit(main())
