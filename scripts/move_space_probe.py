"""수 공간 계측기 — 「메타가 «다음에 둘 수» 의 분포를 정답 쪽으로 옮기나」.

동기: PMI 계열 자 41개가 전멸했다. 원인은 공식이 아니라 «재는 공간» 이었다.
  옛 자   정답식의 로그확률   →  «정답을 알아보는» 능력. 이 과제에선 쉬운 일
  새 자   다음 «수» 의 분포   →  «정답을 찾는» 능력. 이게 진짜 병목
          (실패 궤적은 평균 55회 시도 끝에 소진된다)

★사용자가 짚은 세 함정을 정면으로 처리한다:
  ① 토큰 하나만 보면 안 된다.  실측: '3+7'=3토큰, '100*7'=5토큰 — 수마다 길이가 다르다
     → 수 문자열 «전체 구간» 의 로그확률을 쓰고, 합과 평균을 «둘 다» 기록한다
     → 24개 위에서 정규화해 분포로 만든다 (합 판에서는 공통 접두가 정확히 상쇄되고,
       평균 판은 상쇄되지 않는다 — 평균은 민감도 확인용으로만 쓴다)
     → ★1차 지표는 dPM/dPMT: 후보별 Δ logP 를 G 평균 − B 평균으로 본다.
       같은 토큰열을 두 문맥에서 재므로 길이·서식·빈도가 후보별로 상쇄되고,
       |G| 가 1~3 으로 작은 문제(전체의 62%)에서도 척도가 안 흔들린다
  ② EOS.  ★실측(감사): 중간 문맥에서 <|im_end|> 확률은 사실상 0 이다 — 모델은 EOS 로
     멈추는 게 아니라 «답으로 넘어가며» 탐색을 끝낸다. 그래서 dCont 는 죽은 열로 사전등록하고,
     실제 «끊김» 은 답 시작 구간(STOP_SPAN)의 확률 변화 dStop 으로 잰다.
     ★dG 는 dStop 과 «함께» 읽는다 — 탐색을 멈춘 메타는 dStop≫0·dG≈0 이고 그건 null 이 아니다
  ③ 좋은 수 / 나쁜 수 / 이미 해본 수를 어떻게 정하나 — 아래 정의를 코드로 고정한다

수 공간 정의 (완전 열거, 추측 없음)
  후보수  주어진 네 수에서 «두 개» 를 골라 한 번 연산한 결과.
          + 와 × 는 대칭이라 한 번, − 와 ÷ 는 큰쪽 기준 한 번(중간값 양의 정수 규칙).
          → 문제마다 유효 수의 개수가 다르다 (보통 18~24). 그 집합 전체를 쓴다
  G 좋은 수   그 수를 두고 남은 세 값으로 목표에 «아직 도달 가능» (완전 열거로 판정)
  B 나쁜 수   유효하지만 도달 불가능해짐
  T 해본 수   메타 «앞» 텍스트에 그 결합이 이미 나온 것.
              대칭 연산은 정렬해 비교, 비대칭은 방향까지 비교.
              «식 안에» 들어 있어도 시도한 것으로 센다 ((25*3)-7 → 25*3 은 해봄)
              문서화된 구멍 둘(감사): ⓐ 방향이 틀린 시도("3−25")는 어떤 합법 수의 키도 아니라
              T 에 안 들어간다(합법인 25−3 을 실제로 안 했으므로 타당). ⓑ 파생값이 우연히
              주어진 수와 같으면 파생 연산이 잎 수로 오인될 수 있다 — 드물다.
              ★중간값이 낀 연산(전체의 32%)은 T 를 깨지 않는다 — T 는 «잎» 수의 집합이고,
                잎 연산은 정의상 텍스트에 먼저 나타나므로 부모가 이미 잡힌다

지표 (전부 «메타 뒤 − 메타 앞» 의 이중 차분 — PMI-shift 와 같은 뼈대)
  KL        KL(P_수|메타뒤 ‖ P_수|메타앞)     메타가 정책을 움직이기라도 했나 (방향 없음)
  dH        H(뒤) − H(앞)                     넓히나 좁히나
  dG        P(G|뒤) − P(G|앞)                 ★정답 쪽으로 가이딩하나
  dGT       P(G∖T|뒤) − P(G∖T|앞)             ★★«안 해본 좋은 수» — 정보 이득
  dCont     P(계속|뒤) − P(계속|앞)            메타가 탐색을 끊나
  inv       ★도치판 — 아래

도치(inverted) — self-distill 과 같은 발상을 수 공간으로
  정본  메타를 «주면» 좋은 수 확률이 오르나          P(G|메타) − P(G|무메타)
  도치  좋은 수를 «귀띔하면» 메타 확률이 오르나       logP(메타|힌트) − logP(메타|무힌트)
        → 진짜로 상태를 읽은 메타라면 «정답 방향을 알려줘도» 놀라지 않는다.
          지어낸 메타는 힌트와 충돌한다.
        → 힌트는 «좋은 수 하나» 를 한 줄로 덧붙이는 형태. 정답식은 주지 않는다
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import re
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

REPO = os.environ.get("REPO", "/home/jovyan/beomi/splee/metacognition-math")
sys.path.insert(0, REPO)

MODEL = os.environ.get(
    "PROBE_MODEL",
    "/home/jovyan/.cache/huggingface/hub/models--Qwen--Qwen3-4B/"
    "snapshots/1cfa9a7208912126459214e8b04321603b3df60c")

PAIR = re.compile(r"(\d{1,4})\s*([+\-*/×÷])\s*(\d{1,4})")
# 탐색을 끊고 답으로 넘어가는 구간 — 코퍼스의 최빈 시작 문구
STOP_SPAN = "\n\nSo the final expression is:\n\n\\boxed{"
# 0a 교란: «멈추라» 는 지시만 제거하고 서식 요구는 그대로 둔다 (형식 붕괴 방지)
_STOP_INSTR = "\nThen STOP and answer. Your response MUST end with the final arithmetic "
_NOSTOP_INSTR = "\nThen continue searching. Your response MUST end with the final arithmetic "
_OPSYM = {"×": "*", "÷": "/"}


# ── 수 공간 ────────────────────────────────────────────────────────────────
def moves(nums) -> list[tuple[int, int, str, int]]:
    """유효한 다음 수 전부 → (왼쪽, 오른쪽, 연산, 결과값). Countdown 규칙 준수.

    + 와 × 는 대칭이므로 (작은,큰) 한 번만. − 와 ÷ 는 큰쪽이 왼쪽 (양의 정수 규칙).
    중복 수가 있으면 그 짝도 유효하다 ([5,5,7,8] 의 5+5).
    """
    out, seen = [], set()
    idx = list(range(len(nums)))
    for i, j in itertools.combinations(idx, 2):
        a, b = int(nums[i]), int(nums[j])
        lo, hi = min(a, b), max(a, b)
        for op in ("+", "*", "-", "/"):
            if op == "+":
                l, r, v = lo, hi, lo + hi
            elif op == "*":
                l, r, v = lo, hi, lo * hi
            elif op == "-":
                l, r, v = hi, lo, hi - lo
            else:
                if lo == 0 or hi % lo != 0:
                    continue
                l, r, v = hi, lo, hi // lo
            if v <= 0:
                continue
            key = (l, r, op)
            if key in seen:
                continue
            seen.add(key)
            out.append((l, r, op, v))
    return out


def solvable(vals, target: int) -> bool:
    """남은 값들로 목표 도달 가능한가 — 완전 열거, 중간값 양의 정수 규칙."""
    def go(v):
        if len(v) == 1:
            return v[0] == Fraction(target)
        for i, j in itertools.combinations(range(len(v)), 2):
            a, b = v[i], v[j]
            rest = [v[k] for k in range(len(v)) if k not in (i, j)]
            cands = [a + b, a * b, a - b, b - a]
            if b: cands.append(a / b)
            if a: cands.append(b / a)
            for x in cands:
                if x > 0 and x.denominator == 1 and go(rest + [x]):
                    return True
        return False
    return go([Fraction(int(x)) for x in vals])


def tried_moves(text: str, nums) -> set[tuple[int, int, str]]:
    """메타 «앞» 텍스트에서 이미 시도된 결합. 식 안에 있어도 센다.

    대칭 연산(+,*)은 (작은,큰) 로 정규화. 비대칭(-,/)은 쓰인 방향 그대로.
    피연산자가 주어진 수의 다중집합에서 나올 때만 센다 (중간값끼리의 결합은 별도 층이라 제외).
    """
    want = Counter(int(v) for v in nums)
    got = set()
    for a, o, b in PAIR.findall(str(text)):
        a, b, op = int(a), int(b), _OPSYM.get(o, o)
        if not all(want.get(k, 0) >= c for k, c in Counter((a, b)).items()):
            continue
        got.add((min(a, b), max(a, b), op) if op in "+*" else (a, b, op))
    return got


def remaining_after(nums, mv) -> list[int]:
    l, r, _, v = mv
    rest = [int(x) for x in nums]
    rest.remove(l); rest.remove(r)
    return rest + [int(v)]


# ── 채점 (★수 문자열 «전체 구간» 을 본다. 토큰 하나가 아니다) ─────────────────
def score_spans(model, tok, ctx_ids, cand_strs, batch=8):
    """각 후보 문자열의 (합 로그확률, 평균 로그확률, 토큰수)."""
    import torch
    out = []
    enc = [tok(s, add_special_tokens=False).input_ids for s in cand_strs]
    for lo in range(0, len(enc), batch):
        chunk = enc[lo:lo + batch]
        seqs = [list(ctx_ids) + list(t) for t in chunk]
        L = max(len(s) for s in seqs)
        ids = torch.zeros((len(seqs), L), dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for j, s in enumerate(seqs):
            ids[j, :len(s)] = torch.tensor(s); att[j, :len(s)] = 1
        with torch.no_grad():
            lg = model(input_ids=ids.to(model.device),
                       attention_mask=att.to(model.device)).logits
        for j, t in enumerate(chunk):
            pos = torch.arange(len(ctx_ids) - 1, len(ctx_ids) - 1 + len(t), device=lg.device)
            lsm = torch.log_softmax(lg[j, pos, :].float(), dim=-1)
            tid = torch.tensor(list(t), device=lg.device)
            lp = lsm[torch.arange(len(t), device=lg.device), tid].double().cpu().numpy()
            out.append((float(lp.sum()), float(lp.mean()), len(t)))
        del lg
    return out


def p_continue(model, tok, ctx_ids) -> float:
    """1 − P(EOS 바로 다음). ★메타가 «그만 쓰게» 만드는 효과를 숨기지 않는다."""
    import torch
    with torch.no_grad():
        lg = model(input_ids=torch.tensor([ctx_ids]).to(model.device)).logits
    p = torch.log_softmax(lg[0, -1, :].float(), dim=-1).exp()
    return float(1.0 - p[tok.eos_token_id].item())


def dist(scores, mode="sum"):
    v = np.array([s[0] if mode == "sum" else s[1] for s in scores], dtype=float)
    v = v - v.max()
    e = np.exp(v)
    return e / e.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites_glob", default="cd6_work/probe/sites_shard*.parquet")
    ap.add_argument("--sites_file", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--do_inv", type=int, default=1, help="도치판도 잴지")
    ap.add_argument("--prompt_variant", default="new",
                    help="★0a 교란 검정: 'nostop' 이면 «Then STOP and answer» 를 뺀 프롬프트로 "
                         "문맥을 다시 만든다. dStop 이 «지시 준수» 인지 «모델 성질» 인지 가른다")
    args = ap.parse_args()

    sites = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(args.sites_glob))],
                      ignore_index=True).drop_duplicates("site_id").reset_index(drop=True)
    if args.sites_file:
        keep = {l.strip() for l in open(args.sites_file) if l.strip()}
        sites = sites[sites.site_id.isin(keep)].reset_index(drop=True)
    if args.limit:
        sites = sites.head(args.limit)
    print(f"[move] 사이트 {len(sites)}", flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()

    rows = []
    for n, (_, r) in enumerate(sites.iterrows()):
        nums = [int(v) for v in r["nums"]]; tgt = int(r["target"])
        mvs = moves(nums)
        if len(mvs) < 6:
            continue
        good = np.array([solvable(remaining_after(nums, m), tgt) for m in mvs])
        pre = str(r["response_text"])[:int(r["meta_start"])]
        T = tried_moves(pre, nums)
        key = lambda m: (min(m[0], m[1]), max(m[0], m[1]), m[2]) if m[2] in "+*" else (m[0], m[1], m[2])
        tried = np.array([key(m) in T for m in mvs])

        msgs = json.loads(r["prompt_json"]) if isinstance(r["prompt_json"], str) else r["prompt_json"]
        if args.prompt_variant == "nostop":
            # ★0a — 프롬프트가 «혼잣말 다음에 멈춰라» 를 직접 가르친다(countdown_task.py:373).
            #   그 지시를 빼고 같은 것을 재서, dStop 이 지시 준수인지 모델 성질인지 가른다.
            msgs = [dict(m) for m in msgs]
            for _m in msgs:
                if _m.get("role") == "system":
                    _m["content"] = _m["content"].replace(_STOP_INSTR, _NOSTOP_INSTR)
        try:
            head = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            head = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        text = str(r["response_text"])
        ctx_pre = tok(head + text[:int(r["meta_start"])], add_special_tokens=False).input_ids
        ctx_post = tok(head + text[:int(r["meta_end"])], add_special_tokens=False).input_ids
        # ★공통 접두 '\n' 은 24개 모두에 붙으므로 정규화에서 상쇄된다
        # ★실측: 롤아웃은 공백형 'a + b' 로 쓴다 (공백 18,956 vs 붙임 76). 그 형식을 쓴다
        cands = [f"\n{m[0]} {m[2]} {m[1]} = " for m in mvs]

        rec = dict(site_id=r["site_id"], n_moves=len(mvs), n_good=int(good.sum()),
                   n_tried=int(tried.sum()), n_goodnew=int((good & ~tried).sum()),
                   pos=float(r["pos"]), decision=r["decision"], conf=float(r["confidence"]),
                   ncor=int(r["n_correct_of8"]),
                   solved_in_prefix=int(r.get("solved_in_prefix", 0)))
        # ★같은 입력을 두 번 채점하던 낭비 제거 (런타임 절반)
        pre_sc = score_spans(model, tok, ctx_pre, cands, args.batch)
        post_sc = score_spans(model, tok, ctx_post, cands, args.batch)

        # ★★1차 지표 — 길이중립. 후보마다 «같은 토큰열» 을 두 문맥에서 재므로
        #   길이·토큰경계·서식벌점·유니그램빈도가 «후보별로» 상쇄된다.
        #   그리고 G 평균 − B 평균이라 |G| 가 1~3 으로 작은 문제에서도 척도가 안 흔들린다
        #   (실측: 8000 문제 중 62% 가 |G| ∈ {1,2,3}).
        dlp = np.array([p[0] - q[0] for p, q in zip(post_sc, pre_sc)])
        bad = ~good
        rec["dPM"] = float(dlp[good].mean() - dlp[bad].mean()) if good.any() and bad.any() else np.nan
        gnm = good & ~tried
        rec["dPMT"] = float(dlp[gnm].mean() - dlp[bad].mean()) if gnm.any() and bad.any() else np.nan
        rec["dlp_mean"] = float(dlp.mean()); rec["dlp_sd"] = float(dlp.std())

        for mode in ("sum", "mean"):
            a = dist(pre_sc, mode)
            b = dist(post_sc, mode)
            eps = 1e-12
            rec[f"KL_{mode}"] = float((b * np.log((b + eps) / (a + eps))).sum())
            rec[f"dH_{mode}"] = float(-(b * np.log(b + eps)).sum() + (a * np.log(a + eps)).sum())
            rec[f"dG_{mode}"] = float(b[good].sum() - a[good].sum())
            gn = good & ~tried
            rec[f"dGT_{mode}"] = float(b[gn].sum() - a[gn].sum()) if gn.any() else np.nan
            rec[f"dT_{mode}"] = float(b[tried].sum() - a[tried].sum()) if tried.any() else np.nan
        # ★dCont(im_end 확률)는 중간 문맥에서 사실상 0 이라 죽은 열이다(감사 실측).
        #   탐색을 «끊는» 효과는 «답으로 넘어가는 구간» 의 확률로 잰다.
        #   ★dG 는 dStop 과 «함께» 읽어야 한다 — 탐색을 멈춘 메타는 dStop≫0, dG≈0 이 되고
        #     그건 null 이 아니라 «다른 실제 효과» 다.
        st = score_spans(model, tok, ctx_pre, [STOP_SPAN], 1)[0]
        st2 = score_spans(model, tok, ctx_post, [STOP_SPAN], 1)[0]
        rec["dStop"] = st2[0] - st[0]
        rec["dCont"] = p_continue(model, tok, ctx_post) - p_continue(model, tok, ctx_pre)  # 사전등록: 죽은 열

        if args.do_inv and good.any() and (~good).any():
            # ★도치 — «대조식» 으로만 의미가 있다 (감사 지적).
            #   힌트를 «넣었다» 는 사실 자체가 메타 확률을 바꾸므로, 좋은 힌트와
            #   나쁜 힌트를 짝지어 그 효과를 상쇄한다. 그리고 첫 좋은 수만 쓰면
            #   낮은 인덱스 쌍에 치우치므로 최대 3개를 평균낸다.
            #   ★부호 해석은 decision 조건부로 한다 — 진짜 «막혔다» 메타는 좋은 힌트와
            #     «충돌» 할 수 있다(inv<0). 방향이 아니라 «민감도» 가 진정성의 신호다.
            meta_txt = text[int(r["meta_start"]):int(r["meta_end"])]
            gi = [i for i in range(len(mvs)) if good[i]][:3]
            bi = [i for i in range(len(mvs)) if not good[i]]
            hint_of = lambda i: f"\n(Hint: {mvs[i][0]} {mvs[i][2]} {mvs[i][1]} is worth trying.)\n"
            gs, bs = [], []
            for k, i in enumerate(gi):
                # 나쁜 힌트는 «같은 짝, 다른 연산» 을 우선 — 토큰 길이가 자연히 맞는다
                same = [j for j in bi if {mvs[j][0], mvs[j][1]} == {mvs[i][0], mvs[i][1]}]
                j = same[0] if same else bi[k % len(bi)]
                for lst, idx in ((gs, i), (bs, j)):
                    c = tok(head + text[:int(r["meta_start"])] + hint_of(idx),
                            add_special_tokens=False).input_ids
                    lst.append(score_spans(model, tok, c, [meta_txt], 1)[0][0])
            rec["inv_good"] = float(np.mean(gs)); rec["inv_bad"] = float(np.mean(bs))
            rec["inv_contrast"] = float(np.mean(gs) - np.mean(bs))
        rows.append(rec)
        if (n + 1) % 25 == 0:
            print(f"[move] {n+1}/{len(sites)}", flush=True)

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out)
    print(f"[move] wrote {args.out} ({len(out)}행)\n", flush=True)
    print(f"후보 수 개수 중앙 {out.n_moves.median():.0f} · 좋은 수 {out.n_good.median():.0f} · "
          f"해본 수 {out.n_tried.median():.0f} · 안해본 좋은 수 {out.n_goodnew.median():.0f}")
    for c in ["KL_sum", "dH_sum", "dG_sum", "dGT_sum", "dCont", "inv_sum"]:
        if c in out:
            print(f"  {c:<10} 평균 {out[c].mean():+.4f}  표준편차 {out[c].std():.4f}  "
                  f"|중앙| {out[c].abs().median():.4f}")
    print("\n※ 판정은 감사 후. 여기 숫자는 원자료다.", flush=True)


if __name__ == "__main__":
    main()
