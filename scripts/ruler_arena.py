#!/usr/bin/env python
"""ruler_arena — 455 self-meta 사이트를 «메타-블라인드» 분기점/막힘 자 여럿로 채점한다 (1행 = 1사이트).

prefix = chat head + response_text[:meta_start]   (메타를 지운 접두; 최대 4096 토큰, 응답 왼쪽을 자른다)
  R1  KL(student‖teacher) — teacher 는 system 메시지 끝에 정답식 힌트를 붙인 같은 대화 (RLRT 2605.10781).
      site 토큰 KL · 마지막 32 토큰 평균 KL · 그 32 안에서 site KL 의 백분위
  R2  student 엔트로피 — site · 마지막 32 평균 · 메타 뒤 64 토큰 평균 H_post64 (teacher-forced)
  R3  강제 답 — prefix + STOP_SPAN 뒤 8 표본(T=1)의 정답률 V_hat, greedy 완성의 토큰당 평균 logprob C_greedy
  R5  base 대 학습 모델의 site 토큰 JS (병합 체크포인트가 있을 때만, 없으면 NaN)
  기준선  pos · pass8 · deepconf_bottom10 (창 16 슬라이딩 토큰 logprob 의 하위 10% 평균)
  덤프   student 마지막 층 site 은닉 → arena_hidden.npy (행 순서 = arena.parquet)
실행 순서: HF forward(R1·R2·hidden·deepconf) → 모델 해제 → vLLM(R3) → (R5). 결정적.
"""
from __future__ import annotations

import argparse, gc, glob, importlib.util, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")   # 부모 CUDA 컨텍스트 + fork 충돌 회피
W = Path("/home/jovyan/beomi/splee")
REPO = W / "metacognition-math"
sys.path.insert(0, str(REPO))
MODEL = os.environ.get("PROBE_MODEL", "/home/jovyan/.cache/huggingface/hub/models--Qwen--Qwen3-4B/"
                       "snapshots/1cfa9a7208912126459214e8b04321603b3df60c")
CKPT = W / "cd6_work/hf_ckpt/cd6_A_rep_s2_step50"
STOP_SPAN = "\n\nSo the final expression is:\n\n\\boxed{"
HINT = "\n[Hint: one valid solution is {}]"
MAX_PREFIX, WIN, POST, DC_WIN, GEN_TOK = 4096, 32, 64, 16, 40


def grader():
    # venv site-packages 의 `scripts` 패키지가 저장소 scripts/ 를 가리므로 파일 경로로 적재한다
    spec = importlib.util.spec_from_file_location("countdown_gs0_eval", REPO / "scripts/countdown_gs0_eval.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    from src.training.countdown_task import extract_expr
    return mod._solves, extract_expr


def resolve(p):
    p = Path(p); return p if p.is_absolute() else W / p


def chat(tok, msgs):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def build_site(tok, r):
    """→ (head_s, head_t, resp, post) 토큰열. resp 는 «왼쪽» 에서 잘라 head+resp ≤ MAX_PREFIX."""
    msgs = json.loads(r.prompt_json)
    hint = HINT.format(r.witness)
    msgs_t = [dict(m, content=m["content"] + hint) if m["role"] == "system" else m for m in msgs]
    hs, ht = chat(tok, msgs), chat(tok, msgs_t)
    assert hint in ht and hint not in hs                                              # self-check ②
    assert hs.split("<|im_start|>user", 1)[1] == ht.split("<|im_start|>user", 1)[1]  # system 만 다르다
    enc = lambda s: tok.encode(s, add_special_tokens=False)
    head_s, head_t = enc(hs), enc(ht)
    resp = enc(r.response_text[:r.meta_start])
    resp = resp[max(0, len(resp) - (MAX_PREFIX - len(head_s))):]
    post = enc(r.response_text[r.meta_end:])[:POST]
    assert (head_s + resp)[-WIN:] == (head_t + resp)[-WIN:]                           # self-check ③
    return head_s, head_t, resp, post


# ── HF forward 공통부 ──────────────────────────────────────────────────────
def load_hf(path):
    import torch
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                attn_implementation="sdpa").cuda().eval()


def batches(lens, budget):
    """길이순 정렬, B × Lmax ≤ budget 토큰."""
    cur = []
    for i in np.argsort(lens, kind="stable"):
        if cur and (len(cur) + 1) * lens[i] > budget:
            yield cur; cur = []
        cur.append(int(i))
    if cur:
        yield cur


def forward(model, seqs):
    """왼쪽 패딩 배치 forward → 마지막 층(norm 뒤) 은닉 (B, L, D). j행 토큰은 [L-len, L)."""
    import torch
    L = max(map(len, seqs))
    ids = torch.zeros((len(seqs), L), dtype=torch.long); att = torch.zeros_like(ids)
    for j, s in enumerate(seqs):
        ids[j, L - len(s):] = torch.tensor(s); att[j, L - len(s):] = 1
    pos = (att.cumsum(-1) - 1).clamp(min=0)
    with torch.no_grad():
        return model.model(input_ids=ids.cuda(), attention_mask=att.cuda(),
                           position_ids=pos.cuda()).last_hidden_state


def logsm(model, h):
    import torch
    with torch.no_grad():
        return torch.log_softmax(model.lm_head(h).float(), -1)


def entropy(ls):
    return -(ls.exp() * ls).sum(-1)


def deepconf_bottom10(lp):
    if len(lp) == 0:
        return np.nan
    w = np.convolve(lp, np.ones(DC_WIN) / DC_WIN, "valid") if len(lp) >= DC_WIN else np.array([lp.mean()])
    return float(np.sort(w)[:max(1, int(np.ceil(0.1 * len(w))))].mean())


def student_stats(model, h, seq, n_head, n_resp, n_post):
    """h: (L, D) 한 행, seq = head+resp+post. 응답 토큰 logprob · 엔트로피 · site 창 log-softmax · site 은닉."""
    import torch
    off = h.shape[0] - len(seq)
    lo, site = off + n_head - 1, off + n_head + n_resp - 1     # 응답 첫 토큰 예측 위치 / 마지막 prefix 토큰
    tgt = torch.tensor(seq[1:], device=h.device)
    lp, ent = [], []
    for a in range(lo, site + n_post, 1024):                   # 위치 lo … site+n_post-1
        b = min(a + 1024, site + n_post)
        ls = logsm(model, h[a:b])
        lp.append(ls.gather(1, tgt[a - off:b - off, None])[:, 0]); ent.append(entropy(ls)); del ls
    lp, ent = torch.cat(lp).cpu().numpy(), torch.cat(ent).cpu().numpy()
    win = logsm(model, h[site - WIN + 1:site + 1])             # (WIN, V), 마지막 행 = site
    return dict(H_site=float(ent[n_resp]), H_mean32=float(entropy(win).mean()),
                H_post64=float(ent[n_resp:n_resp + n_post].mean()) if n_post else np.nan,
                deepconf_bottom10=deepconf_bottom10(lp[:n_resp]),
                hidden=h[site].float().cpu().numpy().astype(np.float16), win=win)


def phase_hf(model, S, budget, keep_site_lp, t0):
    """R1·R2·deepconf·hidden. student 배치 → teacher 배치 순으로 같은 인덱스를 처리한다."""
    import torch
    N, V, D = len(S), model.config.vocab_size, model.config.hidden_size
    cols = ["kl_site", "kl_mean32", "kl_pct32", "H_site", "H_mean32", "H_post64", "deepconf_bottom10"]
    out = {k: np.full(N, np.nan) for k in cols}
    out["hidden"] = np.zeros((N, D), np.float16)
    out["site_lp"] = np.zeros((N, V), np.float32) if keep_site_lp else None
    seq_s = [hs + r + p for hs, _, r, p in S]; seq_t = [ht + r for _, ht, r, _ in S]
    done = 0
    for idx in batches([len(s) for s in seq_s], budget):
        h, wins = forward(model, [seq_s[i] for i in idx]), {}
        for j, i in enumerate(idx):
            st = student_stats(model, h[j], seq_s[i], len(S[i][0]), len(S[i][2]), len(S[i][3]))
            wins[i] = st.pop("win"); out["hidden"][i] = st.pop("hidden")
            if keep_site_lp:
                out["site_lp"][i] = wins[i][-1].cpu().numpy()
            for k, v in st.items():
                out[k][i] = v
        del h
        h = forward(model, [seq_t[i] for i in idx])
        for j, i in enumerate(idx):
            win_t = logsm(model, h[j, -WIN:])                  # teacher 의 마지막 WIN 위치 = 같은 토큰들
            kl = (wins[i].exp() * (wins[i] - win_t)).sum(-1).cpu().numpy()
            out["kl_site"][i], out["kl_mean32"][i] = kl[-1], kl.mean()
            out["kl_pct32"][i] = float((kl <= kl[-1]).mean())
        del h, wins; torch.cuda.empty_cache()
        if done // 50 != (done + len(idx)) // 50 or done + len(idx) == N:
            print(f"[hf] {done + len(idx)}/{N} {time.time() - t0:.0f}s", flush=True)
        done += len(idx)
    return out


# ── R3 vLLM 강제 답 ───────────────────────────────────────────────────────
def phase_r3(S, sites, args, tok, t0):
    from vllm import LLM, SamplingParams, TokensPrompt
    solves, extract = grader()
    llm = LLM(model=MODEL, dtype="bfloat16", seed=args.seed, gpu_memory_utilization=args.gpu_util,
              max_model_len=MAX_PREFIX + 64, enforce_eager=True)
    stop_ids = tok.encode(STOP_SPAN, add_special_tokens=False)
    N = len(S); V_hat, C, gexpr, gsol = np.full(N, np.nan), np.full(N, np.nan), [""] * N, np.zeros(N, bool)
    for c0 in range(0, N, 50):
        idx = range(c0, min(c0 + 50, N)); prompts, params = [], []
        for i in idx:
            prompts += [TokensPrompt(prompt_token_ids=S[i][0] + S[i][2] + stop_ids)] * 2
            params += [SamplingParams(n=8, temperature=1.0, top_p=1.0, max_tokens=GEN_TOK, stop=["}"],
                                      seed=args.seed + 100 * i),
                       SamplingParams(n=1, temperature=0.0, max_tokens=GEN_TOK, stop=["}"])]   # ★샘플 logprobs 요청은 sampler.gather_logprobs 에서 크래시(0902 실측) → 아래에서 prompt_logprobs 로 재채점
        res = llm.generate(prompts, params, use_tqdm=False)
        for k, i in enumerate(idx):
            r = sites.iloc[i]; nums, tgt = [int(x) for x in r.nums], int(r.target)

            def ok(t):
                e = extract("\\boxed{" + t + "}")
                return bool(e) and solves(e, nums, tgt)
            V_hat[i] = np.mean([ok(o.text) for o in res[2 * k].outputs])
            g = res[2 * k + 1].outputs[0]
            C[i] = np.nan
            if g.token_ids:   # greedy 토큰열을 prompt_logprobs 로 재채점 (move_space_probe 와 같은 경로, 크래시 없음)
                ids = S[i][0] + S[i][2] + stop_ids
                sc = llm.generate([TokensPrompt(prompt_token_ids=ids + list(g.token_ids))],
                                  [SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)], use_tqdm=False)[0]
                lp = sc.prompt_logprobs[len(ids):]
                vals = [d[t].logprob for d, t in zip(lp, g.token_ids) if d is not None and t in d]
                C[i] = float(np.mean(vals)) if vals else np.nan
            gexpr[i], gsol[i] = g.text, ok(g.text)
        print(f"[r3] {idx.stop}/{N} {time.time() - t0:.0f}s", flush=True)
    del llm; gc.collect()
    return V_hat, C, gexpr, gsol


# ── R5 base 대 학습 모델 JS ────────────────────────────────────────────────
def phase_r5(S, site_lp, budget):
    import torch
    model = load_hf(CKPT)
    seqs, js = [hs + r for hs, _, r, _ in S], np.full(len(S), np.nan)
    for idx in batches([len(s) for s in seqs], budget):
        h = forward(model, [seqs[i] for i in idx])
        q, p = logsm(model, h[:, -1]), torch.from_numpy(site_lp[idx]).cuda()
        m = torch.logaddexp(p, q) - float(np.log(2))
        js[idx] = (0.5 * ((p.exp() * (p - m)).sum(-1) + (q.exp() * (q - m)).sum(-1))).cpu().numpy()
        del h, q, p, m
    del model; gc.collect(); torch.cuda.empty_cache()
    return js


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", nargs="+", default=["cd6_work/probe/sites_shard*.parquet"])
    ap.add_argument("--out", default="cd6_work/probe/arena.parquet")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu_util", type=float, default=0.45)
    ap.add_argument("--skip", nargs="*", default=[], choices=["R5"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--token_budget", type=int, default=12288, help="HF 배치 B×Lmax 상한")
    args = ap.parse_args()
    import torch
    from transformers import AutoTokenizer

    files = sorted(f for p in args.sites for f in glob.glob(str(resolve(p))))
    sites = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if args.limit:
        sites = sites.iloc[:args.limit].reset_index(drop=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    S = [build_site(tok, r) for r in sites.itertuples()]
    do_r5 = "R5" not in args.skip and (CKPT / "config.json").exists()
    print(f"sites={len(S)} files={len(files)} R5={'on' if do_r5 else 'off'}", flush=True)

    t0 = time.time(); model = load_hf(MODEL)
    hf = phase_hf(model, S, args.token_budget, do_r5, t0)
    del model; gc.collect(); torch.cuda.empty_cache(); t1 = time.time()
    V_hat, C, gexpr, gsol = phase_r3(S, sites, args, tok, t1); t2 = time.time()
    js = phase_r5(S, hf.pop("site_lp"), args.token_budget) if do_r5 else np.full(len(S), np.nan); t3 = time.time()

    n_head = np.array([len(s[0]) for s in S]); n_resp = np.array([len(s[2]) for s in S])
    df = pd.DataFrame(dict(
        site_id=sites.site_id, prob_idx=sites.prob_idx, rollout_idx=sites.rollout_idx, decision=sites.decision,
        confidence=sites.confidence, solved_in_prefix=sites.solved_in_prefix, pos=sites.pos,
        pass8=sites.n_correct_of8 / 8.0, n_prefix_tok=n_head + n_resp, n_resp_tok=n_resp,
        n_post_tok=[len(s[3]) for s in S], truncated=(n_head + n_resp) >= MAX_PREFIX,
        **{k: hf[k] for k in ["kl_site", "kl_mean32", "kl_pct32", "H_site", "H_mean32", "H_post64",
                              "deepconf_bottom10"]},
        V_hat=V_hat, C_greedy=C, greedy_expr=gexpr, greedy_solves=gsol, js_r5=js))
    out = resolve(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    np.save(out.with_name("arena_hidden.npy"), hf["hidden"])
    N = len(S)
    print(f"wrote {out} ({N} rows) + arena_hidden.npy {hf['hidden'].shape}")
    print(f"time hf={t1 - t0:.0f}s r3={t2 - t1:.0f}s r5={t3 - t2:.0f}s total={t3 - t0:.0f}s"
          f" → est. 455 sites ≈ {(t3 - t0) / N * 455 / 60:.1f} min (vLLM 기동 포함 상한)")


if __name__ == "__main__":
    main()
