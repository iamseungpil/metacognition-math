"""짝 사이트(good/bad) 자 일괄 채점 — HF forward 만 (vLLM 없음), 배치 1, bf16.

사이트마다 (컨텍스트 = 채팅 head + 응답 프리픽스; 창의 중복 프롬프트 꼬리는 마커 뒤로 잘라낸다):
  pmi_shift        PMI_close − PMI_open, PMI = Σ_div logp(\\boxed{gold}) − Σ_div logp(\\boxed{decoy})
                   (countdown_pmi.divergent_spans 의 발산 슬라이스; open = head+prefix, close = head+prefix+meta)
  pmi_shift_full   같은 4 팔, 슬라이스 대신 \\boxed{…} 전체 문자열 합
  osd              (1/|W|)[logP(W|…+meta) − logP(W|… 메타 없음)]  — W 는 대리 후속(아래)
  inv_prose_min    min_t logp(meta_t | HINT ctx) − min_t logp(meta_t | plain ctx), 프로즈 토큰만 (inplace/a2d/min)
  inv_mean         같은 마스크의 mean 차
  dhat_T1_meta/post64   mean_t [log π_stu − log π_T1], T1 = 시스템 메시지에 정답 경로(path_hint) 를 넣은 교사
  dhat_T2_meta/post64   같은 값, T2 = 프리픽스 끝에 «이미 실패한 결합 목록» 을 넣은 탐색 교사
★parquet 에 메타 뒤 후속이 없다(meta_end 뒤는 '\n' 뿐). post64/W 는 대리 텍스트(≤64 토큰)다.
  --w_mode witness_path (기본) : 증인 경로를 푼 문장 — good/bad 짝에서 **바이트 동일**. 짝 차이는 메타만 통해 들어온다.
  --w_mode neutral             : 수를 하나도 안 두는 중립 문장 — 역시 짝 동일.
  --w_mode next_move           : (감사 결과 항진명제) 메타의 'next:' 수를 그대로 두는 문장. good/bad 에서 W 자체가 다르고
                                  코드가 next_move 문자열을 파싱·산술까지 해서 넣으므로 osd/dhat_*_post64 가 메타를 재는 것이
                                  아니라 next 수를 읽는 자가 된다. 진단 비교용으로만 남긴다.
  next_in_witness / next_in_decoy : 항진 진단 — next 수가 증인 경로 / PMI 오답(op-swap) 경로의 한 걸음과 일치하는가.
"""
from __future__ import annotations
import argparse, json, os, random, re, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "scripts"):
    sys.path.insert(0, str(p))
from kl_ruler import MODEL, path_hint, render                     # noqa: E402
from src.training.countdown_pmi import boxed, divergent_spans      # noqa: E402
from src.training.countdown_task import eval_countdown, swap_op_decoy  # noqa: E402

MAX_PRE, W_MAX = 4096, 64
MARKER = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
INV_HINT = "\n\nHint: one valid solution is {w}."
_SKIP_LINE = re.compile(r"^\s*(confidence|decision)\s*:", re.I)
_ATTEMPT = re.compile(r"^\s*(?:\d+[.)]\s*)?\$?\s*([^=$]*\d[^=$]*?)\s*=")
_MOVE = re.compile(r"^\s*(\d+)\s*([+\-*/])\s*(\d+)")


def enc(tok, s):
    return tok.encode(s, add_special_tokens=False)


def resp_prefix(text: str, meta_start: int) -> str:
    """창(프롬프트 꼬리 + 응답)에서 응답만. 마커가 없으면 창 앞이 잘린 응답이므로 그대로 쓴다."""
    i = text.find(MARKER)
    return text[i + len(MARKER):meta_start] if i >= 0 else text[:meta_start]


def hint_head(tok, msgs, target, witness):
    """도치 자의 교사 컨텍스트: 유저 메시지의 'Target: N' 앵커 바로 뒤에 힌트 (countdown_inv 규약)."""
    head = render(tok, msgs)
    anchor = f"Target: {int(target)}"
    i = head.rfind(anchor)
    if i < 0:
        raise ValueError("anchor missing: " + anchor)
    j = i + len(anchor)
    return head[:j] + INV_HINT.format(w=witness) + head[j:]


_FIELD_LINE = re.compile(r"^\s*(next|ruled_out)\s*:", re.I)


def prose_mask(tok, meta_raw: str, skip=_SKIP_LINE, only=None):
    """meta_raw 를 통째로 토큰화하고, 태그/skip 줄이 아닌 문자에 완전히 놓인 토큰만 True.
    ⚠기본 skip 은 confidence/decision 뿐이라 'next:' / 'ruled_out:' 줄이 **프로즈에 포함**된다 — 이 짝에서는
    next 줄이 good/bad 의 유일한 차이이므로 inv_prose_* 는 그 줄을 통해 갈린다. `only=_FIELD_LINE` 은 그 줄만,
    `skip=_FIELD_LINE` 를 겹치면 판단 문장만 남는다(sent 마스크)."""
    flags = []
    for ln in meta_raw.split("\n"):
        keep = bool(ln.strip()) and not ln.strip().startswith("<") and not skip.match(ln)
        if only is not None:
            keep = bool(only.match(ln))
        flags += [keep] * len(ln) + [False]
    flags = flags[:len(meta_raw)]
    e = tok(meta_raw, add_special_tokens=False, return_offsets_mapping=True)
    mask = np.array([b > a and all(flags[a:b]) for a, b in e["offset_mapping"]], dtype=bool)
    return list(e["input_ids"]), mask


_SKIP_SENT = re.compile(r"^\s*(confidence|decision|next|ruled_out)\s*:", re.I)


def failed_attempts(prefix: str, k: int = 8) -> list[str]:
    """프리픽스에서 '=' 와 숫자가 있는 줄의 결합(= 앞부분) 을 뽑아 마지막 k 개."""
    out = []
    for ln in prefix.split("\n"):
        m = _ATTEMPT.match(ln)
        if m and "=" in ln:
            g = re.sub(r"\s+", " ", m.group(1).replace("\\times", "*").replace("\\div", "/")).strip(" $\\")
            if g and any(c in g for c in "+-*/"):
                out.append(g)
    return out[-k:]


def path_steps(expr: str):
    """증인 식 → [(a, op, b, v), …] 안쪽 괄호부터 (kl_ruler.path_hint 와 같은 순서)."""
    steps, e = [], str(expr).replace(" ", "")
    pat = re.compile(r"\((\d+)([+\-*/])(\d+)\)")
    while True:
        m = pat.search(e)
        if not m:
            break
        a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
        v = eval_countdown(f"({a}{op}{b})")
        if v is None:
            break
        steps.append((a, op, b, int(v))); e = e[:m.start()] + str(v) + e[m.end():]
    m = re.fullmatch(r"(\d+)([+\-*/])(\d+)", e)
    if m:
        a, op, b = int(m.group(1)), m.group(2), int(m.group(3)); v = eval_countdown(f"({a}{op}{b})")
        if v is not None:
            steps.append((a, op, b, int(v)))
    return steps


def move_in_path(next_move: str, expr: str) -> bool:
    """next 수 'a op b' 가 expr 경로의 한 걸음과 (교환법칙 포함) 일치하는가 — 항진 진단용."""
    m = _MOVE.match(str(next_move))
    if not m or not expr:
        return False
    a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
    for x, o, y, _ in path_steps(expr):
        if o == op and ((x, y) == (a, b) or (o in "+*" and (x, y) == (b, a))):
            return True
    return False


def witness_path_text(wit: str) -> str:
    """짝 동일 대리 후속: 증인 경로를 한 걸음씩 두고 \\boxed 로 닫는다."""
    st = path_steps(wit)
    if not st:
        return f"\n\nLet me try {wit} directly.\n\\boxed{{{wit}}}"
    lines = [f"Let me try {a} {op} {b} = {v} first." if i == 0 else f"Then {a} {op} {b} = {v}."
             for i, (a, op, b, v) in enumerate(st)]
    return "\n\n" + "\n".join(lines) + f"\nThat reaches the target.\n\\boxed{{{wit}}}"


def witness_nofirst_text(wit: str) -> str:
    """짝 동일 대리 후속인데 첫 걸음 문장을 지운 것 — 메타의 'next:' 줄이 W 첫 줄을 되읽는 항진(감사 §26) 을 막는다."""
    st = path_steps(wit)
    if len(st) < 2:
        return NEUTRAL_W
    lines = [f"Then {a} {op} {b} = {v}." for (a, op, b, v) in st[1:]]
    return "\n\nLet me continue from there.\n" + "\n".join(lines) + f"\nThat reaches the target.\n\\boxed{{{wit}}}"


NEUTRAL_W = "\n\nLet me continue with the next grouping and check whether it reaches the target.\n"


def next_step_text(next_move: str, nums) -> str:
    """⚠항진 경로(--w_mode next_move 전용). 메타의 'next:' 수를 실제로 두는 대리 후속: 값 계산 + 남는 수."""
    m = _MOVE.match(str(next_move))
    if not m:
        return f"\n\nLet me try {next_move} first.\n"
    a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
    v = eval_countdown(f"({a}{op}{b})")
    s = f"\n\nLet me try {a} {op} {b} = {v if v is not None else '?'} first.\n"
    rest = [int(x) for x in nums]
    if a in rest and b in (rest[:rest.index(a)] + rest[rest.index(a) + 1:]) and v is not None:
        rest.remove(a); rest.remove(b)
        s += f"Remaining numbers: {rest + [int(v)]}.\n"
    return s


@torch.no_grad()
def gather_logp(model, ids: list[int], spans: list[tuple[int, int]]) -> list[np.ndarray]:
    """한 시퀀스 forward. spans 의 [s,e) 토큰들을 예측하는 위치의 logp 를 각각 돌려준다."""
    lg = model(input_ids=torch.tensor([ids], device="cuda")).logits[0]
    out = []
    for s, e in spans:
        part = torch.log_softmax(lg[s - 1:e - 1].float(), -1)
        tgt = torch.tensor(ids[s:e], device="cuda")
        out.append(part.gather(1, tgt[:, None])[:, 0].double().cpu().numpy())
    del lg
    return out


def pmi_from(tok, model, ctx: list[int], pair) -> tuple[float, float]:
    """(발산 슬라이스 PMI, 전체 문자열 PMI) at ctx."""
    g, d = list(pair.gold_ids), list(pair.decoy_ids)
    lg = gather_logp(model, ctx + g, [(len(ctx), len(ctx) + len(g))])[0]
    ld = gather_logp(model, ctx + d, [(len(ctx), len(ctx) + len(d))])[0]
    return float(lg[pair.gold_slice].sum() - ld[pair.decoy_slice].sum()), float(lg.sum() - ld.sum())


def score_site(tok, model, r, variants, w_mode="witness_path") -> dict:
    msgs = json.loads(r["prompt_json"]) if isinstance(r["prompt_json"], str) else list(r["prompt_json"])
    text, s, e = str(r["response_text"]), int(r["meta_start"]), int(r["meta_end"])
    nums, target, wit = [int(x) for x in r["nums"]], int(r["target"]), str(r["witness"]).strip()
    prefix, meta = resp_prefix(text, s), text[s:e]
    head = enc(tok, render(tok, msgs))
    pre = enc(tok, prefix)[-MAX_PRE:]
    meta_ids, pmask = prose_mask(tok, meta)
    _, smask = prose_mask(tok, meta, skip=_SKIP_SENT)        # 판단 문장만 (next/ruled_out 제외)
    _, nmask = prose_mask(tok, meta, only=_FIELD_LINE)        # next/ruled_out 줄만
    nm = str(r.get("next_move", ""))
    def self_greedy_text():
        """정답 무관·짝 동일 W: 학생이 메타 없이 프리픽스에서 탐욕 64토큰 이어 쓴 것(감사 §26 제안 ②)."""
        import torch
        with torch.no_grad():
            x = torch.tensor([head + pre], device=model.device)
            g = model.generate(x, max_new_tokens=W_MAX, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(g[0, x.shape[1]:], skip_special_tokens=True)
    w_text = {"witness_path": lambda: witness_path_text(wit), "neutral": lambda: NEUTRAL_W, "self_greedy": self_greedy_text,
              "witness_nofirst": lambda: witness_nofirst_text(wit),
              "next_move": lambda: next_step_text(nm, nums)}[w_mode]()
    w_ids = enc(tok, w_text)[:W_MAX]
    ctx_open, ctx_close = head + pre, head + pre + meta_ids
    m_span = (len(ctx_open), len(ctx_close))
    w_span = (len(ctx_close), len(ctx_close) + len(w_ids))
    out = dict(site_id=r["site_id"], n_pre_tok=len(pre), n_meta_tok=len(meta_ids), n_prose_tok=int(pmask.sum()),
               n_w_tok=len(w_ids), w_mode=w_mode, next_in_witness=move_in_path(nm, wit))

    # ── 학생: 메타 + W 한 번, 메타 없이 W 한 번 (OSD 의 두 팔)
    stu_meta, stu_w = gather_logp(model, ctx_close + w_ids, [m_span, w_span])
    stu_w_open = gather_logp(model, ctx_open + w_ids, [(len(ctx_open), len(ctx_open) + len(w_ids))])[0]
    out["osd"] = float(stu_w.mean() - stu_w_open.mean())

    # ── PMI-shift (4 팔; decoy = swap_op_decoy, countdown_pmi 의 발산 슬라이스)
    decoy = swap_op_decoy(wit, nums, target, random.Random(1))
    pair = divergent_spans(tok, wit, decoy) if decoy else None
    if pair is None:
        out.update(pmi_shift=np.nan, pmi_shift_full=np.nan, pmi_path="none", next_in_decoy=False)
    else:
        po, pfo = pmi_from(tok, model, ctx_open, pair)
        pc, pfc = pmi_from(tok, model, ctx_close, pair)
        out.update(pmi_shift=pc - po, pmi_shift_full=pfc - pfo, pmi_open=po, pmi_close=pc, pmi_path=pair.path,
                   next_in_decoy=move_in_path(nm, decoy))

    # ── 도치: 힌트 head 만 다르고 pre+meta 토큰은 동일
    hh = enc(tok, hint_head(tok, msgs, target, wit))
    hint_meta = gather_logp(model, hh + pre + meta_ids, [(len(hh) + len(pre), len(hh) + len(pre) + len(meta_ids))])[0]
    P, H = stu_meta[pmask], hint_meta[pmask]
    out["inv_prose_min"] = float(H.min() - P.min()) if len(P) >= 3 else np.nan
    out["inv_mean"] = float(H.mean() - P.mean()) if len(P) >= 3 else np.nan
    out["inv_all_mean"] = float(hint_meta.mean() - stu_meta.mean())
    out["n_sent_tok"], out["n_next_tok"] = int(smask.sum()), int(nmask.sum())
    S_, Hs = stu_meta[smask], hint_meta[smask]
    out["inv_sent_min"] = float(Hs.min() - S_.min()) if len(S_) >= 3 else np.nan
    out["inv_sent_mean"] = float(Hs.mean() - S_.mean()) if len(S_) >= 3 else np.nan
    out["inv_next_sum"] = float((hint_meta[nmask] - stu_meta[nmask]).sum()) if nmask.any() else np.nan

    # ── T1: 정답 경로를 시스템 메시지에 (kl_ruler render 규약)
    if "T1" in variants:
        h1 = enc(tok, render(tok, msgs, hint=path_hint(wit, target)))
        b = len(h1) + len(pre)
        t_meta, t_w = gather_logp(model, h1 + pre + meta_ids + w_ids,
                                  [(b, b + len(meta_ids)), (b + len(meta_ids), b + len(meta_ids) + len(w_ids))])
        out["dhat_T1_meta"] = float((stu_meta - t_meta).mean())
        out["dhat_T1_post64"] = float((stu_w - t_w).mean())
        out["dhat_T1_meta_pos"] = float(((stu_meta - t_meta) > 0).mean())
        out["dhat_T1_sent"] = float((stu_meta - t_meta)[smask].mean()) if smask.any() else np.nan
        out["dhat_T1_next"] = float((stu_meta - t_meta)[nmask].sum()) if nmask.any() else np.nan

    # ── T2: 실패한 결합 목록을 프리픽스 끝(메타 직전)에
    if "T2" in variants:
        att = failed_attempts(prefix)
        hint2 = ("\n[Hint: these groupings already failed: " + "; ".join(att) + "]\n") if att \
            else "\n[Hint: no groupings have been tried yet]\n"
        h2 = enc(tok, hint2)
        b = len(ctx_open) + len(h2)
        t_meta, t_w = gather_logp(model, ctx_open + h2 + meta_ids + w_ids,
                                  [(b, b + len(meta_ids)), (b + len(meta_ids), b + len(meta_ids) + len(w_ids))])
        out["dhat_T2_meta"] = float((stu_meta - t_meta).mean())
        out["dhat_T2_post64"] = float((stu_w - t_w).mean())
        out["dhat_T2_sent"] = float((stu_meta - t_meta)[smask].mean()) if smask.any() else np.nan
        out["n_failed_attempts"] = len(att)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--teacher_variants", default="T1,T2")
    ap.add_argument("--w_mode", default="witness_path", choices=["witness_path", "witness_nofirst", "neutral", "next_move", "self_greedy"])
    args = ap.parse_args()
    variants = {v.strip() for v in args.teacher_variants.split(",") if v.strip()}
    from transformers import AutoModelForCausalLM, AutoTokenizer
    S = pd.read_parquet(args.sites)
    if args.limit:
        S = S.head(args.limit)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    rows, t0 = [], time.time()
    for i, (_, r) in enumerate(S.iterrows()):
        rows.append(score_site(tok, model, r, variants, w_mode=args.w_mode))
        if (i + 1) % 25 == 0 or i + 1 == len(S):
            print(f"[pair] {i+1}/{len(S)} {time.time()-t0:.0f}s", flush=True)
    out = pd.DataFrame(rows)
    out.to_parquet(args.out)
    num = out.select_dtypes("number")
    print(f"[pair] wrote {args.out} ({len(out)}행) · non-finite: {int((~np.isfinite(num.to_numpy(float))).sum())}")
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:+.3f}".format):
        print(out.drop(columns=["pmi_path", "w_mode"], errors="ignore").T.to_string() if len(out) <= 10 else num.describe().T)


if __name__ == "__main__":
    main()
