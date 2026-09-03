"""1단계 자 확인 — 교사–학생 정보 비대칭 (RLRT 2605.10781) 을 HF forward 만으로 잰다 (vLLM 없음).

사이트마다:
  D_bar_site  = KL(π(·|prefix) ‖ π(·|hint+prefix))  at 메타 직전 위치 (다음 토큰 분포)
  D_bar_m32   = 메타 앞 32 토큰에 걸친 위치별 KL 평균 (teacher-forced)
  H_site      = 학생 다음 토큰 엔트로피 at 메타 직전
  D_hat_meta  = 메타 토큰별 log π학생 − log π교사 의 평균 (양수 = 교사가 예측 못 한 «학생 주도» 토큰)
  D_hat_pos   = 메타 토큰 중 D_hat > 0 인 비율
  D_hat_post  = 메타 직후 32 토큰의 같은 값 (메타가 연 «다음 걸음» 이 학생 주도인가)
교사 = 같은 모델, 시스템 메시지 끝에 "[Hint: one valid solution is <witness>]" 를 붙인 것. 유저 메시지·프리픽스 동일.
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
import numpy as np, pandas as pd, torch

MODEL = os.environ.get("PROBE_MODEL", "/home/jovyan/.cache/huggingface/hub/models--Qwen--Qwen3-4B/"
                       "snapshots/1cfa9a7208912126459214e8b04321603b3df60c")
MAX_TOK = 4096


def render(tok, msgs, hint=None):
    m = [dict(x) for x in msgs]
    if hint is not None:
        assert m[0]["role"] == "system"; m[0]["content"] = m[0]["content"] + f"\n[Hint: one valid solution is {hint}]"
    try:
        return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)


def path_hint(expr: str, target: int) -> str:
    """정답 식을 안쪽 괄호부터 평가한 단계 목록으로 편다: '(25+4)=29; (29+15)=44; (44+1)=45'."""
    import re as _re
    steps, e = [], expr.replace(" ", "")
    pat = _re.compile(r"\((\d+)([+\-*/])(\d+)\)")
    while True:
        m = pat.search(e)
        if not m: break
        a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
        v = {"+": a + b, "-": a - b, "*": a * b, "/": (a // b if b and a % b == 0 else None)}[op]
        if v is None: break
        steps.append(f"{a}{op}{b}={v}"); e = e[:m.start()] + str(v) + e[m.end():]
    m = _re.fullmatch(r"(\d+)([+\-*/])(\d+)", e)
    if m:
        a, op, b = int(m.group(1)), m.group(2), int(m.group(3)); v = {"+": a + b, "-": a - b, "*": a * b, "/": a // b if b else 0}[op]
        steps.append(f"{a}{op}{b}={v}")
    return "the path " + "; ".join(steps) + f" reaches {target}, i.e. {expr}"


@torch.no_grad()
def logsoftmax_rows(model, ids):
    """ids: list[int] (한 시퀀스). 반환 log-softmax [L, V] (float32, CPU 로 옮기지 않고 GPU 유지)."""
    x = torch.tensor([ids], device="cuda")
    out = model(input_ids=x).logits[0].float()
    return torch.log_softmax(out, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", required=True, help="parquet glob")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--teacher", default="witness", choices=["witness", "path"],
                    help="witness: 정답 식 한 줄 힌트 · path: 정답 식을 단계별로 푼 경로(RLRT 의 «정답 롤아웃 조건»에 가까움)")
    ap.add_argument("--hint_pos", default="system", choices=["system", "prefix_end"],
                    help="system: 시스템 메시지 끝에 힌트 · prefix_end: 메타 «직전» 프리픽스 끝에 힌트 문장 삽입(최근 문맥)")
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    S = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(args.sites))]).drop_duplicates("site_id").reset_index(drop=True)
    if args.limit: S = S.head(args.limit)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
    rows, t0 = [], time.time()
    for i, r in S.iterrows():
        msgs = json.loads(r["prompt_json"]) if isinstance(r["prompt_json"], str) else list(r["prompt_json"])
        text = str(r["response_text"]); s, e = int(r["meta_start"]), int(r["meta_end"])
        pre, meta, post = text[:s], text[s:e], text[e:e + 400]
        hint = str(r["witness"]) if args.teacher == "witness" else path_hint(str(r["witness"]), int(r["target"]))
        if args.hint_pos == "system":
            heads = {"stu": render(tok, msgs), "tea": render(tok, msgs, hint=hint)}
            pre_ids = tok.encode(pre, add_special_tokens=False)[-MAX_TOK:]
            pre_ids_t = pre_ids
        else:   # 힌트를 프리픽스 끝(메타 직전)에 넣는다 — 교사·학생의 head 는 같고 body 앞부분만 다르다
            heads = {"stu": render(tok, msgs), "tea": render(tok, msgs)}
            pre_ids = tok.encode(pre, add_special_tokens=False)[-MAX_TOK:]
            pre_ids_t = pre_ids + tok.encode(f"\n[Hint: one valid solution is {hint}]\n", add_special_tokens=False)
        assert (args.hint_pos != "system") or ("[Hint:" in heads["tea"] and "[Hint:" not in heads["stu"])
        meta_ids = tok.encode(meta, add_special_tokens=False); post_ids = tok.encode(post, add_special_tokens=False)[:32]
        bodies = {"stu": pre_ids + meta_ids + post_ids, "tea": pre_ids_t + meta_ids + post_ids}
        lp = {}
        for k, h in heads.items():
            hid = tok.encode(h, add_special_tokens=False)
            L = logsoftmax_rows(model, hid + bodies[k])
            off = len(hid) - 1 + (len(pre_ids_t) - len(pre_ids) if k == "tea" else 0)
            lp[k] = L[off:]                                   # 두 키 모두: 위치 j 가 (pre_ids+meta+post)[j] 를 예측하도록 정렬
        body = pre_ids + meta_ids + post_ids
        n_pre = len(pre_ids)
        def kl_at(j):  # 학생 ‖ 교사, body[j] 예측 분포
            p, q = lp["stu"][j], lp["tea"][j]; return float((p.exp() * (p - q)).sum())
        def ent_at(j):
            p = lp["stu"][j]; return float(-(p.exp() * p).sum())
        # 메타 첫 토큰을 예측하는 위치 = n_pre (다음 토큰 = meta_ids[0])
        D_site = kl_at(n_pre); H_site = ent_at(n_pre)
        D_m32 = float(np.mean([kl_at(j) for j in range(max(0, n_pre - 32), n_pre)])) if n_pre > 0 else np.nan
        def dhat(rng_):
            v = [float(lp["stu"][j][body[j]] - lp["tea"][j][body[j]]) for j in rng_]
            return (float(np.mean(v)), float(np.mean([x > 0 for x in v]))) if v else (np.nan, np.nan)
        D_meta, D_meta_pos = dhat(range(n_pre, n_pre + len(meta_ids)))
        D_post, D_post_pos = dhat(range(n_pre + len(meta_ids), n_pre + len(meta_ids) + len(post_ids)))
        D_pre32, _ = dhat(range(max(0, n_pre - 32), n_pre))
        rows.append(dict(site_id=r["site_id"], D_bar_site=D_site, D_bar_m32=D_m32, H_site=H_site,
                         D_hat_meta=D_meta, D_hat_meta_pos=D_meta_pos, D_hat_post=D_post, D_hat_post_pos=D_post_pos,
                         D_hat_pre32=D_pre32, n_meta_tok=len(meta_ids), n_pre_tok=n_pre))
        del lp; torch.cuda.empty_cache()
        if (i + 1) % 25 == 0: print(f"[kl] {i+1}/{len(S)} {time.time()-t0:.0f}s", flush=True)
    out = pd.DataFrame(rows); out.to_parquet(args.out)
    print(f"[kl] wrote {args.out} ({len(out)}행) — D_bar_site 평균 {out.D_bar_site.mean():.3f} · D_hat_meta 평균 {out.D_hat_meta.mean():+.3f}", flush=True)


if __name__ == "__main__":
    main()
