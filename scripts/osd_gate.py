"""출시 전 관문 — OSD(Outcome-Signed Surprisal Drop)가 «좋은 메타» 를 가리는가.

왜 이 스크립트가 따로 있는가. PMI-shift 는 판별력 AUC 0.52(=우연)로 폐기됐다. 그 사실을
**150스텝을 태운 뒤에** 알았다. 같은 실수를 반복하지 않으려면 보상을 배선하기 **전에**
「이 점수가 푼 롤아웃을 위로 매기는가」를 GPU 한 장으로 먼저 재야 한다. 이 파일이 그 관문이다.

읽는 법 (docs/reports/2026-08-17 §1.5 의 교훈):

    수학 세대에서 **가짜 대조군이 우리 공식을 이겼다** — meta_len 이 AUC 0.598,
    PMI-shift 가 0.457. 그래서 절대 임계선 하나로는 아무것도 판정하지 못한다.
    반드시 «길이 대조군» 을 같은 표본·같은 방식으로 나란히 재고, 그것을 이겨야 한다.

    통과 조건:  OSD AUC >= 0.60  AND  OSD AUC >= 길이대조군 AUC + 0.05

여기서 재는 것은 **부호를 곱하기 전의** Δcert 다. R_osd = y * clip(Δcert/c, -1, 1) 의
y 가 곧 AUC 의 라벨(r_corr)이므로, 부호를 곱한 값의 AUC 는 항상 1.000 이 되는 항진명제다
(countdown_rewards.meta_outcome_discrimination 의 B6 감사 주석이 같은 함정을 적는다).
따라서 unsigned Δcert 만이 검정 대상이다.

    Δcert_i = (1/|W|) * [ logP(W | 프롬프트 ⊕ 응답[..t1])      # 메타 «포함» 문맥
                        - logP(W | 프롬프트 ⊕ 응답[..t0)) ]    # 메타 구간만 «제거» 한 문맥

    t0 = <meta> 시작, t1 = </meta> 끝, W = 응답 t1 직후부터 \boxed 끝까지 (최대 200토큰).

핵심은 두 문맥이 **메타 구간 유무만** 다르고 W 는 양쪽에서 **같은 토큰열** 이라는 것이다.
샘플링 없음 — 티처포싱 forward 두 번뿐이다.

산출: `--out` JSON (auc_osd / auc_len / n_pos / n_neg / c_p90 / delta_cert 통계 /
n_leak_blocked / verdict) + 사람이 읽는 요약.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training import countdown_rewards as cdr           # noqa: E402
from src.training.countdown_task import (                    # noqa: E402
    build_prompt, extract_expr, grade,
)


# ══════════════════════════════════════════════════════════════════════════════
# W 구간 잡기
# ══════════════════════════════════════════════════════════════════════════════

def boxed_end_char(text: str) -> int | None:
    r"""마지막 `\boxed{...}` 의 **닫는 중괄호 다음** 문자 오프셋. 없으면 None.

    ⚠countdown_task._last_boxed 와 **같은 중괄호 짝맞추기** 다. 거기는 식 «문자열» 만
    돌려주므로 오프셋을 얻을 수 없어 여기서 한 번 더 훑는다. 로직을 갈라놓지 않으려고
    한 글자도 다르지 않게 옮겼다 — 저쪽이 바뀌면 여기도 같이 바꿔야 한다.
    """
    t = text or ""
    i, out = 0, None
    while True:
        j = t.find("\\boxed{", i)
        if j < 0:
            break
        k, dep = j + 7, 1
        while k < len(t) and dep:
            dep += (t[k] == "{") - (t[k] == "}")
            k += 1
        if dep == 0:
            out = k                    # 닫는 `}` 다음
        i = j + 7
    return out


def shares_ngram(a_ids, b_ids, n: int = 8) -> bool:
    """두 토큰열이 길이 n 이상의 n-gram 을 공유하는가.

    왜 필요한가 — 메타 본문에 «앞으로 쓸 문장을 미리 적어두면» 그 뒤 W 의 logP 가
    통째로 올라간다. Δcert 는 «메타가 확신을 올렸다» 로 읽지만 실제로는 **복붙 캐시**다.
    n-gram 이 겹치면 그 행의 Δcert 를 0 으로 죽여 이 해킹 경로를 닫는다.
    """
    if len(a_ids) < n or len(b_ids) < n:
        return False
    grams = {tuple(a_ids[i:i + n]) for i in range(len(a_ids) - n + 1)}
    return any(tuple(b_ids[i:i + n]) in grams for i in range(len(b_ids) - n + 1))


# ══════════════════════════════════════════════════════════════════════════════
# 티처포싱 forward — pmi_ruler_probe.span_logp 와 같은 방식
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def span_mean_logp(model, tok, ctx_ids, w_ids, device) -> float:
    """ctx 뒤에 W 를 붙였을 때 W 토큰들의 **평균** logprob.

    ctx 와 W 를 각각 따로 인코딩해 이어붙인다(pmi_ruler_probe 와 같은 관례). 그래야
    두 문맥에서 W 가 **완전히 같은 토큰열** 임이 구조적으로 보장된다 — 이어붙인 문자열을
    통째로 재인코딩하면 경계에서 토큰이 갈려 «같은 W» 라는 전제가 깨진다.

    ★평균(=길이로 나눔)인 이유: 정의에 1/|W| 가 들어 있다. 메타 길이는 공식에
    등장하지 않는다 — 길이 대조군이 이미 0.598 을 찍은 전례가 있어, 길이가 뒷문으로
    새어 들어오는 통로를 막아야 한다.
    """
    ids = torch.tensor([list(ctx_ids) + list(w_ids)], device=device)
    logits = model(ids).logits[0]
    s = len(ctx_ids) - 1
    # ★슬라이스를 먼저 하고 log_softmax 를 뒤에 한다. 전체 위치에 대해 float 로 올리면
    #   3800토큰 x 152k vocab 이 그대로 메모리를 먹는다.
    part = logits[s:s + len(w_ids)].float().log_softmax(-1)
    tgt = torch.tensor(list(w_ids), device=device)
    lp = part.gather(1, tgt[:, None])[:, 0]
    return float(lp.mean().item())


# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data", required=True, help="countdown_val_4num.parquet")
    ap.add_argument("--limit", type=int, default=250, help="문제 수")
    ap.add_argument("--n", type=int, default=8, help="문제당 롤아웃")
    # ★3072 = 학습의 data.max_response_length. 1200 으로 줄이면 len_p95=1580 을 잘라
    #   \boxed 가 사라지고 W 가 비어 표본이 통째로 날아간다(실측: 이전 프로브가 그 함정).
    ap.add_argument("--max_tokens", type=int, default=3072)
    # ★4수 세트의 42% 정확도는 «new» 프롬프트로 잰 수다(BASE4_s7: acc 0.418 · 발화 0.411).
    #   학습과 조건을 맞추려면 이 인자를 학습이 쓰는 변형으로 넘겨라.
    ap.add_argument("--prompt_variant", default="new", choices=["new", "old", "shot"])
    ap.add_argument("--meta_form", default="new", choices=["new", "old"])
    ap.add_argument("--w_max_tokens", type=int, default=200, help="W 길이 상한 L")
    ap.add_argument("--ngram_n", type=int, default=8, help="누출 가드 n-gram 길이")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    # ★vLLM 을 먼저 띄우고 HF 모델을 뒤에 올린다. vLLM 은 del 해도 풀 캐시를 완전히
    #   돌려주지 않으므로 0.42 로 낮춰 잡는다(pmi_ruler_probe 의 실측 관례).
    ap.add_argument("--gpu_util", type=float, default=0.42)
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main() -> None:
    a = parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer, AutoModelForCausalLM

    df = pd.read_parquet(a.data).head(a.limit)
    tok = AutoTokenizer.from_pretrained(a.model_path)

    def chat(msgs) -> str:
        try:
            return tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:                       # enable_thinking 없는 토크나이저
            return tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)

    prompts, insts = [], []
    for _, r in df.iterrows():
        inst = {"nums": [int(v) for v in r["nums"]], "target": int(r["target"]),
                "witness": r.get("witness", ""), "decoy": r.get("decoy", "")}
        prompts.append(chat(build_prompt(inst, variant=a.prompt_variant)))
        insts.append(inst)

    print(f"[osd] {len(df)} 문제 x {a.n} 롤아웃 · 프롬프트={a.prompt_variant} "
          f"· 메타형식={a.meta_form}", flush=True)

    # ── 1) 롤아웃 생성 (vLLM) ────────────────────────────────────────────────
    llm = LLM(model=a.model_path, dtype="bfloat16", seed=a.seed,
              gpu_memory_utilization=a.gpu_util,
              max_model_len=a.max_tokens + 1024,
              # 평가는 추론뿐이다. torch.compile 은 CPU 경합에서 죽으므로 건너뛴다.
              enforce_eager=True)
    outs = llm.generate(prompts, SamplingParams(
        n=a.n, temperature=a.temperature, max_tokens=a.max_tokens, seed=a.seed))

    rows = []
    n_roll = n_trunc = n_emit = 0
    for gi, (inst, p, o) in enumerate(zip(insts, prompts, outs)):
        for x in o.outputs:
            n_roll += 1
            n_trunc += int(x.finish_reason == "length")
            text = x.text
            m = cdr.parse_meta(text, form=a.meta_form)
            if not m["emitted"]:
                continue                    # 미발화 → 학습에서 R_osd = 0
            n_emit += 1
            rows.append({
                "group_id": f"g{gi}",
                "prompt": p,
                "text": text,
                "meta": m,
                "r_corr": int(grade(text, inst["nums"], inst["target"])),
                # ⚠`or ""` 필수 — answer_leak 은 None 에 예외를 던진다(의도적).
                "final_expr": extract_expr(text) or "",
                # 형식 위반(블록 2개 이상) → 학습에서 R_osd = 0. 여기서도 같게 죽인다.
                "form_ok": cdr.meta_form_ok(text, a.meta_form),
            })

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[osd] 롤아웃 {n_roll} · 발화 {n_emit} ({n_emit / max(1, n_roll):.3f}) "
          f"· 절단 {n_trunc / max(1, n_roll):.3f}", flush=True)

    # ── 2) Δcert (unsigned) — 티처포싱 forward 두 번 ─────────────────────────
    dev = "cuda"
    model = AutoModelForCausalLM.from_pretrained(
        a.model_path, torch_dtype=torch.bfloat16, device_map=dev).eval()

    scored = []                 # 살아남은 행
    n_leak = n_fmt = n_emptyw = n_drop = 0
    for i, r in enumerate(rows):
        m = r["meta"]
        t0_c, t1_c = m["start"], m["end"]
        text = r["text"]

        meta_ids = tok.encode(m.get("body") or "", add_special_tokens=False)
        meta_n_tok = len(meta_ids)

        # W = </meta> 직후 ~ \boxed 끝. 그 앞이거나 없으면 W 는 빈 구간이다.
        b_end = boxed_end_char(text)
        w_text = text[t1_c:b_end] if (b_end is not None and b_end > t1_c) else ""
        w_ids = tok.encode(w_text, add_special_tokens=False)[:a.w_max_tokens]

        if not r["form_ok"]:
            n_fmt += 1
            scored.append({**r, "delta": 0.0, "meta_n_tok": meta_n_tok, "why": "format"})
            continue
        if not w_ids:
            # W 가 비면 정의상 R_osd = 0 (드롭이 아니다 — 학습에서 0 이 지급된다).
            n_emptyw += 1
            scored.append({**r, "delta": 0.0, "meta_n_tok": meta_n_tok, "why": "empty_W"})
            continue

        # ── 누출 가드 ──────────────────────────────────────────────────────
        #   ① 메타 본문과 W 가 8토큰 n-gram 을 공유  ② 메타가 최종 \boxed 식을 담음
        #   둘 중 하나면 Δcert := 0. 학습 때와 **같은 조건** 으로 재야 하므로 여기서도 건다.
        leaked = (shares_ngram(meta_ids, w_ids, a.ngram_n)
                  or bool(cdr.answer_leak(m["raw"], r["final_expr"])))
        if leaked:
            n_leak += 1
            scored.append({**r, "delta": 0.0, "meta_n_tok": meta_n_tok, "why": "leak"})
            continue

        ctx_with = tok.encode(r["prompt"] + text[:t1_c], add_special_tokens=False)
        ctx_without = tok.encode(r["prompt"] + text[:t0_c], add_special_tokens=False)
        d = (span_mean_logp(model, tok, ctx_with, w_ids, dev)
             - span_mean_logp(model, tok, ctx_without, w_ids, dev))

        if not np.isfinite(d):
            # ★NaN/inf 는 0 으로 채우지 않고 **행을 드롭** 한다 — 기존 PMI 관례와 같다
            #   (pmi_ruler_probe.auc 가 `s == s` 로 걸러낸다). 0 으로 채우면 «쟀는데
            #   신호가 없었다» 로 오독되고, 그것이 이 리포가 반복해서 산 실수다.
            n_drop += 1
            continue
        scored.append({**r, "delta": float(d), "meta_n_tok": meta_n_tok, "why": "ok"})

        if (i + 1) % 100 == 0:
            print(f"[osd] Δcert {i + 1}/{len(rows)}", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ── 3) 두 AUC — 같은 표본·같은 함수·같은 그룹 ────────────────────────────
    #    countdown_rewards.meta_outcome_discrimination 을 **재사용** 한다. 여기서
    #    AUC 를 다시 짜면 관문의 수와 학습 계기판의 수가 갈리고, 그러면 "관문은
    #    통과했는데 학습 중엔 0.5" 를 «측정 방식 차이» 와 구별할 수 없다.
    #    components 에 `meta_pos` 로 넣는 이유: 그 키가 META_TERMS 에 있어 함수가
    #    메타 항으로 읽는다. `meta_mul`/`meta_ctx` 는 sign(adv_corr) 항진명제로 분류돼
    #    auc 를 해석 불가로 만든다 — 절대 쓰면 안 된다.
    gids = [r["group_id"] for r in scored]
    osd = cdr.meta_outcome_discrimination(
        scored, [{"meta_pos": r["delta"]} for r in scored], group_ids=gids)
    lenc = cdr.meta_outcome_discrimination(
        scored, [{"meta_pos": float(r["meta_n_tok"])} for r in scored], group_ids=gids)

    # 그룹 내 비교 가능한 (정답, 오답) 쌍의 수 — AUC 가 실제로 몇 개 위에 서 있는가.
    by_g: dict = {}
    for r in scored:
        by_g.setdefault(r["group_id"], []).append(r["r_corr"])
    n_pairs = sum(sum(v) * (len(v) - sum(v)) for v in by_g.values())

    deltas = [r["delta"] for r in scored]
    # ★정규화 상수 c 는 학습 코드가 쓸 수와 **같은 함수** 로 뽑는다(cdr._quantile).
    #   numpy 로 따로 재면 보간 규약이 갈려 c 가 미세하게 달라진다.
    c_p90 = cdr._quantile([abs(d) for d in deltas], 0.9)

    auc_osd = osd["auc"]
    auc_len = lenc["auc"]
    ok_abs = (auc_osd == auc_osd) and auc_osd >= 0.60
    ok_ctl = (auc_osd == auc_osd) and (auc_len == auc_len) and auc_osd >= auc_len + 0.05
    verdict = "PASS" if (ok_abs and ok_ctl) else "FAIL"

    warns = []
    if osd["n_pos"] < 100:
        warns.append(f"정답 표본 {osd['n_pos']}개 < 100 — 판정 불가 위험(이전 조사가 "
                     "양성 12개로 좌초했다). --limit/--n 을 올려 다시 재라")
    if osd["n_neg"] < 100:
        warns.append(f"오답 표본 {osd['n_neg']}개 < 100 — 같은 이유로 불충분하다")
    if n_pairs < 100:
        warns.append(f"그룹 내 (정답,오답) 쌍 {n_pairs}개 < 100 — AUC 가 너무 얇은 "
                     "표본 위에 서 있다")
    if auc_osd != auc_osd:
        warns.append("OSD AUC 가 NaN — 같은 그룹 안에 (정답, 오답) 쌍이 없다. "
                     "잴 수 없었다는 뜻이지 통과가 아니다")

    fin = [d for d in deltas if np.isfinite(d)]
    res = {
        "verdict": verdict,
        "auc_osd": auc_osd,
        "auc_len": auc_len,
        "margin_over_len": (auc_osd - auc_len) if (auc_osd == auc_osd and auc_len == auc_len)
                           else float("nan"),
        "pass_rule": "auc_osd >= 0.60 AND auc_osd >= auc_len + 0.05",
        "n_pos": osd["n_pos"],
        "n_neg": osd["n_neg"],
        "n_pairs_within_group": n_pairs,
        "c_p90": c_p90,
        "delta_cert": {
            "mean": float(np.mean(fin)) if fin else float("nan"),
            "std": float(np.std(fin)) if fin else float("nan"),
            "p10": cdr._quantile(fin, 0.10), "p50": cdr._quantile(fin, 0.50),
            "p90": cdr._quantile(fin, 0.90),
            "frac_positive": float(np.mean([d > 0 for d in fin])) if fin else float("nan"),
        },
        "n_leak_blocked": n_leak,
        "n_format_blocked": n_fmt,
        "n_empty_w": n_emptyw,
        "n_dropped_nonfinite": n_drop,
        "n_rollouts": n_roll,
        "n_emitted": n_emit,
        "n_scored": len(scored),
        "emit_rate": n_emit / max(1, n_roll),
        "trunc_rate": n_trunc / max(1, n_roll),
        "osd_detail": osd,
        "len_detail": lenc,
        "warnings": warns,
        "spec_version": cdr.SPEC_VERSION,
        "model_path": a.model_path,
        "data": a.data,
        "config": {"limit": a.limit, "n": a.n, "max_tokens": a.max_tokens,
                   "prompt_variant": a.prompt_variant, "meta_form": a.meta_form,
                   "w_max_tokens": a.w_max_tokens, "ngram_n": a.ngram_n,
                   "seed": a.seed},
    }

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))

    # ── 4) 사람이 읽는 판정 ─────────────────────────────────────────────────
    print("\n===== OSD 관문 =====", flush=True)
    print(f"  발화율            {res['emit_rate']:.3f}  (롤아웃 {n_roll} 중 {n_emit})")
    print(f"  절단율            {res['trunc_rate']:.3f}")
    print(f"  채점된 행         {len(scored)}  "
          f"(누출차단 {n_leak} · 형식차단 {n_fmt} · W빈행 {n_emptyw} · 드롭 {n_drop})")
    print(f"  정답/오답 표본    {osd['n_pos']} / {osd['n_neg']}  "
          f"· 그룹내 쌍 {n_pairs}")
    print(f"  Δcert 평균/표준   {res['delta_cert']['mean']:.4f} / "
          f"{res['delta_cert']['std']:.4f}")
    print(f"  ★OSD AUC          {auc_osd:.4f}")
    print(f"  ★길이대조군 AUC   {auc_len:.4f}   (차이 {res['margin_over_len']:+.4f})")
    print(f"  정규화 상수 c     {c_p90:.4f}   (= |Δcert| 90퍼센타일)")
    for w in warns:
        print(f"  ⚠경고             {w}")
    print(f"\n  판정: {verdict}   "
          f"[AUC>=0.60 {'O' if ok_abs else 'X'} · 대조군+0.05 {'O' if ok_ctl else 'X'}]")
    print(f"[osd] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
