"""메타 값어치 3문 실험 — 한 번의 생성으로 세 질문을 동시에 가른다.

배경(2026-09-01, 페이블 감사 후 확정):
  · confidence → 최종정답 AUC 0.777   (지각은 좋다)
  · novelty    → 최종정답 AUC 0.744   (새 조합을 뒤진 궤적이 훨씬 자주 성공한다)
  · 그런데 «redirect 선언» → «실제 새 탐색» 결합은 약하다 (p=0.54, n=196)
  · 그리고 «고정 문구를 강제 주입»하면 해롭다 (e−f = −3.73pp, 6/6 시드)
관찰(+)과 인과(−)의 어긋남이 「좋은 메타만 도움이 된다」 가설의 자리다.

한 절단점(stem)에서 네 팔을 같은 구조로 — 가지 B개 × 가지당 이어쓰기 M개:
  none     가지 = 그냥 재시작 B번                    (교환가능 → 보정 후 분산 ≈ 0 이어야 함)
  self     가지 = 모델이 «자기 혼잣말»을 새로 씀 B개
  placebo  가지 = ★다른 stem 의 혼잣말 B개를 가져와 붙임
  forced   가지 = 고정 redirect 블록

★placebo 가 왜 필수인가(감사 지적): self 의 가지간 분산은 «혼잣말 품질이 다르다»와
  «다른 글자가 끼면 이어쓰기가 흔들린다»를 섞고 있다. placebo 는 길이·형식·토큰분포가
  같고 stem 특정 내용만 다르므로 후자만 재는 대조가 된다. 그리고 e−f<0 이 이미 확인된
  이상, placebo 없이는 Q1(self−none)이 «내용 이득 − 삽입 손해»로 상쇄돼 해석 불가다.

  Q1 내용 효과     self − placebo        혼잣말 «내용»이 돕는가
     삽입 효과     placebo − none        글자를 끼우는 것 자체의 효과
  Q2 고를 여지     σ²_branch(self) − σ²_branch(placebo)
                   ★max-over-B 는 평균의 단조변환이라 여지를 못 잰다 → 분산성분으로 잰다
                   보정 추정기: σ̂² = Var_b(p̂) − mean_b[p̂(1−p̂)/(M−1)]
  Q3 조건부 인과   forced − none 을 «막힘»으로 층화. 막힘 추정은 none 의 가지 0·1,
                   비교는 가지 2·3 (분리된 표본 — 평균회귀 차단)

★vLLM 씨앗 함정(감사 발견): gpu_model_runner.py:542 는 요청별 salt 없이
  SamplingParams.seed 를 그대로 manual_seed 한다. 같은 프롬프트 + 같은 씨앗 = 같은 토큰열.
  none/forced 는 가지들이 프롬프트가 동일하므로 **B개 가지가 바이트 동일**해진다.
  → 프롬프트마다 다른 씨앗의 SamplingParams 리스트를 넘긴다.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
sys.path.insert(0, os.path.join(os.environ.get("REPO", "/home/jovyan/beomi/splee/metacognition-math"), "scripts"))  # steer_prompts (0902)
sys.path.insert(0, os.environ.get("REPO", "/home/jovyan/beomi/splee/metacognition-math"))
from src.training.countdown_task import PROMPT_VARIANTS, grade  # noqa: E402

MODEL = os.environ.get(
    "PROBE_MODEL",
    "/home/jovyan/.cache/huggingface/hub/models--Qwen--Qwen3-4B/"
    "snapshots/1cfa9a7208912126459214e8b04321603b3df60c")

META_CUE = "\n<meta>\n"
# ★양방향 누락 수정(2026-09-01): 지금까지 «강제 주입» 계열은 redirect 만 넣었다.
#   그런데 프리레지스터한 정책은 «낮으면 전환 · 높으면 검산»의 양방향이고, 실측에서
#   두 방향은 정반대로 행동한다 — clean 사이트 유도가치 L2 가
#     redirect −0.0036 [−0.020,+0.014]  (해롭지 않다)
#     verify   −0.0606 [−0.099,−0.024]  (확실히 해롭다)
#   redirect 만 강제해 놓고 「메타 주입은 해롭다」고 말할 수 없다. 두 팔을 다 넣는다.
#   두 블록은 길이·형식·확신 위치를 맞추고 «방향»만 다르게 한다.
REDIRECT_BLOCK = ("\n<meta>\nconfidence: 0.2\n"
                  "This family of groupings has not reached the target. "
                  "I should abandon it and try a different family.\n"
                  "decision: redirect\n</meta>\n")
VERIFY_BLOCK = ("\n<meta>\nconfidence: 0.8\n"
                "This family of groupings is close to reaching the target. "
                "I should stay with it and check my arithmetic carefully.\n"
                "decision: verify\n</meta>\n")
BLOCKS = {"forced": REDIRECT_BLOCK, "forced_verify": VERIFY_BLOCK}

# ── 전략 주입 팔 (Opus 설계) ────────────────────────────────────────────────
# 지금까지 «조절» 을 단 한 가지 동작(A0)으로만 시험했다. Gandhi 네 습관 중 둘을
# 우리 모델은 사실상 안 한다(거꾸로 풀기 0% / 20,000 롤아웃).
# ★공정성: 문장1 은 «모든 팔에서 동일», 문장2 만 다르다. confidence·decision 은 바이트 동일.
#   숫자·연산자·목표값을 넣지 않아 위약과 공정하다.
# ★C1 은 «구체적 명령이 삽입됐다» 를, C2 는 «아무 절차 규칙이나 도움이 된다» 를 통제한다.
_S1 = "This family of groupings has not reached the target."
_STRAT = {
    "A0_backtrack": "I should abandon it and try a different family of groupings.",
    "A1_backward":  "I should start from the target and undo one operation to reach it.",
    "A2_subgoal":   "I should first build an intermediate value near the target and adjust it.",
    "A3_constraint":"I should place the hardest number first and fit the others around it.",
    "C1_imperative":"I should take a moment to restate the problem clearly and then continue.",
    "C2_misdirect": "I should write the numbers in the order they are given and continue.",
}
for _k, _s2 in _STRAT.items():
    BLOCKS[_k] = f"\n<meta>\nconfidence: 0.2\n{_S1} {_s2}\ndecision: redirect\n</meta>\n"

_META_OK = re.compile(r"confidence\s*:\s*[01]?\.?\d+.*?decision\s*:\s*(verify|redirect)", re.S | re.I)

# ── ★SSI (탐색 유사도 지표) ─────────────────────────────────────────────────
# 문헌 지목 #1 실험 (BroSt, LION 2023 · Ruan/Horvitz/Kautz CP 2002):
#   «K번 재시작이 전부 실패» 는 두 경우를 섞고 있다 —
#     ⓐ 재시작들이 서로 «비슷» 했다 → 다양화가 «일어나지 않았다» → 개입이 크게 도움될 것
#     ⓑ 재시작들이 서로 «달랐다»   → 진짜로 어려운 문제 → 개입은 별 도움 안 되고 예산을 줘야
#   고전 재시작 이론의 핵심 전제(i.i.d. 독립)가 LLM 롤아웃에서 깨지는 지점이 바로 여기다.
#   Countdown 은 «남은 수의 다중집합» 을 해시해 방문 상태를 «정확히» 셀 수 있다 —
#   임베딩 근사가 필요 없다. 이 기준선은 문헌에 대응물이 없다.
_PAIRRE = re.compile(r"(\d{1,4})\s*([+\-*/×÷])\s*(\d{1,4})")


def state_fingerprint(text: str, nums) -> frozenset:
    """이어쓰기가 «방문한 상태» 의 집합. 상태 = (결합한 두 수, 연산)."""
    from collections import Counter as _C
    want = _C(int(v) for v in nums); out = set()
    for a, o, b in _PAIRRE.findall(str(text)):
        a, b = int(a), int(b)
        if all(want.get(k, 0) >= c for k, c in _C((a, b)).items()):
            op = {"×": "*", "÷": "/"}.get(o, o)
            out.add((min(a, b), max(a, b), op) if op in "+*" else (a, b, op))
    return frozenset(out)


def ssi(fps) -> float:
    """재시작들이 서로 얼마나 «비슷» 한가 — 쌍별 자카드 평균. 1=전부 같은 곳을 뒤짐."""
    fps = [f for f in fps if f]
    if len(fps) < 2:
        return float("nan")
    sims = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            u = len(fps[i] | fps[j])
            sims.append(len(fps[i] & fps[j]) / u if u else 0.0)
    return float(np.mean(sims))


def sp_list(count: int, base: int, **kw):
    """프롬프트마다 다른 씨앗 — 같은 프롬프트+같은 씨앗이면 vLLM 이 같은 토큰열을 낸다.

    ★2026-09-02 결함 수정: vLLM v1 은 자식 c 에 seed+c 를 준다(parallel_sampling.py:74).
    보폭이 1 이면 같은 텍스트의 인접 프롬프트(브랜치)가 자식 씨앗을 공유해 연속이 겹쳤다
    (판정 0,1 과 비교 2,3 이 씨앗 3개 공유 → none 기준선 편향). 보폭 = 100 ≥ n 으로 고정."""
    from vllm import SamplingParams
    stride = 100
    assert stride >= max(int(kw.get("n", 1)), 1)
    return [SamplingParams(seed=base + stride * i, **kw) for i in range(count)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--n_traj", type=int, default=3)
    ap.add_argument("--branches", type=int, default=4, help="stem 당 가지 수 B")
    ap.add_argument("--m_cont", type=int, default=4, help="가지당 이어쓰기 수 M (≥2)")
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--arms", default="none,self,placebo,forced,forced_verify")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu_util", type=float, default=0.85)
    ap.add_argument("--judge_n", type=int, default=8, help="judge 팔(별도 판정 실행)의 이어쓰기 수")
    ap.add_argument("--save_text", action="store_true", help="자식별 연속 텍스트를 *_texts.parquet 에 저장")
    ap.add_argument("--ban_meta", action="store_true", help="judge·none 팔에서 <meta> 재발화 금지 (라벨 수정 0902)")
    args = ap.parse_args()
    assert args.m_cont >= 2, "Q2(선택 여지)는 가지 내 분산이 필요하다 — M≥2"

    df = pd.read_parquet(args.data).head(args.limit)
    print(f"[worth] 문제 {len(df)} · 궤적 {args.n_traj} · 가지 {args.branches} × 이어쓰기 "
          f"{args.m_cont} · 팔 {args.arms} · 씨앗 {args.seed}", flush=True)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(model=MODEL, dtype="bfloat16", seed=args.seed, tensor_parallel_size=1,
              gpu_memory_utilization=args.gpu_util, max_model_len=4864, enforce_eager=True)

    def chat(u):
        msgs = [{"role": "system", "content": PROMPT_VARIANTS["new"]},
                {"role": "user", "content": u}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    heads, golds = [], []
    for _, r in df.iterrows():
        p = r["prompt"]
        heads.append(chat(p[-1]["content"] if isinstance(p, (list, np.ndarray)) else str(p)))
        golds.append((list(int(v) for v in r["nums"]), int(r["target"])))

    # ── 1차 궤적 → stem (성공 궤적도 \boxed 직전 절단으로 보존) ──────────────
    first = llm.generate(heads, sp_list(len(heads), args.seed * 100000,
                                        temperature=1.0, top_p=1.0, max_tokens=1200,
                                        n=args.n_traj))
    stems, meta = [], []
    QUANT = (0.25, 0.50, 0.75)
    for i, o in enumerate(first):
        nums, tgt = golds[i]
        for k, g in enumerate(o.outputs):
            txt = g.text
            first_ok = int("\\boxed" in txt and bool(grade(txt, nums, tgt)))
            body = txt[:txt.index("\\boxed")] if "\\boxed" in txt else txt
            if len(body) < 80:
                continue
            q = QUANT[(i + k) % len(QUANT)]
            pos = max(40, int(len(body) * q))
            stems.append(heads[i] + body[:pos])
            meta.append(dict(prob=i, traj=k, first_ok=first_ok, quant=q, cut_chars=pos,
                             pre_meta=int("<meta>" in body[:pos])))
    print(f"[worth] stem {len(stems)}  (성공 프리픽스 {sum(m['first_ok'] for m in meta)} 포함)",
          flush=True)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(stem=i, text=s[-1500:], **meta[i]) for i, s in enumerate(stems)]) \
        .to_parquet(outp.with_name(outp.stem + "_stems.parquet"))

    # ── self 혼잣말 생성 (한 번만; placebo 는 이걸 섞어 쓴다) ────────────────
    arms = [a.strip() for a in args.arms.split(",")]
    self_meta, self_fin = None, None
    if "self" in arms or "placebo" in arms:
        mo = llm.generate([s + META_CUE for s in stems],
                          sp_list(len(stems), args.seed * 100000 + 11,
                                  temperature=1.0, top_p=1.0, max_tokens=160,
                                  n=args.branches, stop=["</meta>"]))
        self_meta = [[g.text for g in o.outputs] for o in mo]
        self_fin = [[g.finish_reason for g in o.outputs] for o in mo]
        nstop = sum(f == "stop" for fs in self_fin for f in fs)
        print(f"[worth] 혼잣말 {sum(len(x) for x in self_meta)}개 "
              f"(</meta> 도달 {nstop})", flush=True)
        # placebo 짝짓기: 같은 quant 층 안에서 stem 을 한 칸 돌린다
        pair = {}
        for q in QUANT:
            idx = [i for i, m in enumerate(meta) if m["quant"] == q]
            for a, b in zip(idx, idx[1:] + idx[:1]):
                pair[a] = b

    rows = []
    texts_out = []                       # ★연속 텍스트 저장 (조작 점검용)
    for arm in arms:
        bprompts, bkeys, btexts, bfin = [], [], [], []
        for si, s in enumerate(stems):
            for b in range(args.branches if arm != "judge" else 1):
                if arm == "judge":       # ★별도 판정 실행: bare stem, 독립 씨앗, n=judge_n
                    body, fin = "", ""
                    bprompts.append(s)
                elif arm in ("truth_ro", "fab_ro"):
                    # ★0902 자기 보고 진실성 A/B: 같은 자리에 «실제로 시도한 묶음»(truth) vs «시도 안 한 묶음»(fab) 을
                    #   ruled_out 에 적은 블록을 끼운다. next·문장·확신·결정은 두 팔 동일. 시도 흔적이 2개 미만인 스템은 건너뜀.
                    nums_, _ = golds[meta[si]["prob"]]
                    fp_ = sorted(state_fingerprint(s, nums_))
                    tried = [f"{a}{o}{b}" for a, b, o in fp_]
                    from itertools import combinations as _cb
                    allp = [f"{min(a,b)}{o}{max(a,b)}" if o in "+*" else f"{a}{o}{b}"
                            for a, b in _cb(list(nums_), 2) for o in "+-*/"]
                    untried = [x for x in allp if x not in tried]
                    if len(tried) < 2 or len(untried) < 3:
                        continue
                    rng_ = np.random.default_rng(si * 7 + 1)
                    ro = list(rng_.choice(tried, 2, replace=False)) if arm == "truth_ro" else list(rng_.choice(untried[:-1], 2, replace=False))
                    nxt = untried[-1]
                    body, fin = "", ""
                    bprompts.append(s + f"\n<meta>\nconfidence: 0.3\nruled_out: {ro[0]}, {ro[1]}\nnext: {nxt} first\n"
                                    f"{_S1} I should abandon it and try a different family of groupings.\ndecision: redirect\n</meta>\n")
                elif arm in ("good_next", "bad_next"):
                    # ★0902 «다음 수 계획» 내용 A/B: next 에 «해를 살리는 첫수»(good) vs «해를 죽이는 첫수»(bad) 를 적는다.
                    #   나머지 문장·확신·결정 동일. 도달가능성은 move_space_probe.solvable 로 완전 열거.
                    import itertools as _it
                    nums_, tgt_ = golds[meta[si]["prob"]]
                    from move_space_probe import solvable as _solv
                    goods, bads = [], []
                    for (i1, a), (i2, b) in _it.combinations(list(enumerate(nums_)), 2):
                        rest = [v for k_, v in enumerate(nums_) if k_ not in (i1, i2)]
                        for o, v in (("+", a + b), ("*", a * b), ("-", abs(a - b)), ("/", (max(a, b) // min(a, b)) if min(a, b) and max(a, b) % min(a, b) == 0 else None)):
                            if v is None or v <= 0: continue
                            hi, lo = max(a, b), min(a, b)
                            mv = f"{lo}{o}{hi}" if o in "+*" else f"{hi}{o}{lo}"
                            (goods if _solv(rest + [v], tgt_) else bads).append(mv)
                    if not goods or not bads:
                        continue
                    rng_ = np.random.default_rng(si * 11 + 3)
                    mv = rng_.choice(goods) if arm == "good_next" else rng_.choice(bads)
                    body, fin = "", ""
                    bprompts.append(s + f"\n<meta>\nconfidence: 0.3\nruled_out: none\nnext: {mv} first\n"
                                    f"{_S1} I should abandon it and try a different family of groupings.\ndecision: redirect\n</meta>\n")
                elif arm == "p5self":    # ★0902: 같은 스템, 시스템 프롬프트만 P5e(전략 어휘+예시)로 교체, 블록 없음.
                    #   «막힌 자리에서 모델이 스스로 backward 를 고르는가»와 그 결과를 잰다 (텍스트 저장 필수).
                    import steer_prompts as _SP
                    assert PROMPT_VARIANTS["new"] in s, "head 에 SOLVE_SYS_NEW 가 없다"
                    body, fin = "", ""
                    bprompts.append(s.replace(PROMPT_VARIANTS["new"], _SP.VARIANTS["P5e"], 1))
                elif arm == "self":
                    body = self_meta[si][b]; fin = self_fin[si][b]
                    bprompts.append(s + META_CUE + body + "</meta>\n")
                elif arm == "placebo":
                    d = pair[si]
                    if d == si:
                        continue
                    body = self_meta[d][b]; fin = self_fin[d][b]
                    bprompts.append(s + META_CUE + body + "</meta>\n")
                else:
                    body, fin = "", ""
                    bprompts.append(s + BLOCKS.get(arm, ""))
                bkeys.append((si, b)); btexts.append(body); bfin.append(fin)
        n_cont = args.judge_n if arm == "judge" else args.m_cont
        # 팔별 씨앗 범위: 10^7 간격 (프롬프트 수 × 보폭 100 보다 크게) — 팔끼리 절대 안 겹친다
        # ★0902 라벨 수정: judge(막힘 판정)·none 은 «메타 없는» 이어쓰기여야 한다 — 실측 40% 가 새 메타를 다시 썼다.
        _ban = {"bad_words": ["<meta", " <meta", "\n<meta", "<meta>", "<META"]} if (args.ban_meta and arm in ("judge", "none")) else {}
        outs = llm.generate(bprompts,
                            sp_list(len(bprompts), args.seed * 10**9 + 10**7 * (arms.index(arm) + 2),
                                    temperature=1.0, top_p=1.0, max_tokens=1400, n=n_cont, **_ban))
        for (si, b), bt, fi, o in zip(bkeys, btexts, bfin, outs):
            nums, tgt = golds[meta[si]["prob"]]
            fps = [state_fingerprint(g.text, nums) for g in o.outputs]
            if args.save_text:
                for ci, g in enumerate(o.outputs):
                    texts_out.append(dict(arm=arm, stem=si, branch=b, child=ci,
                                          ok=int(bool(grade(g.text, nums, tgt))),
                                          fin=g.finish_reason, text=g.text[:2500]))
            rows.append(dict(arm=arm, stem=si, branch=b,
                             ssi=ssi(fps),                       # ★가지 «내부» 유사도
                             n_states=float(np.mean([len(f) for f in fps])),
                             fp=",".join(f"{x[0]}{x[2]}{x[1]}" for x in sorted(fps[0])[:40]),
                             succ=sum(int(bool(grade(g.text, nums, tgt))) for g in o.outputs),
                             m=n_cont,
                             n_tok=float(np.mean([len(g.token_ids) for g in o.outputs])),
                             n_capped=sum(int(g.finish_reason == "length") for g in o.outputs),
                             meta_fin=fi,
                             meta_ok=int(bool(_META_OK.search(bt))) if bt else -1,
                             meta_text=bt[:600], **meta[si]))
        print(f"[worth] 팔 {arm} 완료 ({len(bkeys)} 가지)", flush=True)

    out = pd.DataFrame(rows)
    out.to_parquet(args.out)
    if args.save_text:
        tp = Path(args.out).with_name(Path(args.out).stem + "_texts.parquet")
        pd.DataFrame(texts_out).to_parquet(tp)
        print(f"[worth] wrote {tp} ({len(texts_out)}행)", flush=True)
    print(f"[worth] wrote {args.out} ({len(out)}행)", flush=True)

    # ── 진행 확인용 요약 (헤드라인은 반드시 별도 분석에서 군집 부트스트랩으로) ──
    M = args.m_cont
    def corrected(g):
        v = []
        for _, x in g.groupby("stem"):
            if len(x) < 2:
                continue
            p = (x.succ / M).values
            v.append(p.var(ddof=1) - np.mean(p * (1 - p) / (M - 1)))
        return float(np.mean(v)) if v else np.nan
    print(f"\n{'팔':<9}{'성공률':>9}{'보정분산':>11}{'가지수':>8}{'절단률':>9}")
    for a, g in out.groupby("arm"):
        print(f"{a:<9}{g.succ.sum()/g.m.sum():>9.4f}{corrected(g):>11.5f}"
              f"{len(g):>8}{g.n_capped.sum()/g.m.sum():>9.4f}")   # 분모 = 실제 m (judge 는 judge_n)
    print("\n※ none 의 보정분산은 0 근처여야 정상(가지가 교환가능). 크면 씨앗 배선 재점검.",
          flush=True)


if __name__ == "__main__":
    main()
