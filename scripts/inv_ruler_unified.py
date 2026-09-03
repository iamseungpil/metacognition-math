r"""도치 자 **통일 재측정** — 학습이 쓸 바로 그 함수로 라벨과 게이밍을 같은 자로 잰다.

왜 이 스크립트가 필요한가 (설계검토 2026-08-31 의 1번 반대):
  `reverse_ruler.py` 의 V1_prose_min(= min(gold)−min(plain), 프로즈를 **별도 시퀀스로
  재인코딩**) 과 `gaming_probe.py` 의 inv_prose_min(= min(hint−plain), 전체 메타를 문맥에
  두고 토큰 마스크) 은 같은 37 사이트에서 스피어만 **0.296** 이다. 라벨 판별력(d=1.00)은
  앞의 자에서, 게이밍 취약성은 뒤의 자에서 나온 숫자라 **둘을 합쳐 결론을 낼 수 없다**.

여기서는 `src/training/countdown_inv.py` 의 함수 — 학습 배선이 그대로 호출할 함수 —
하나만 써서 다음 둘을 같은 조건으로 잰다:
  (A) 라벨 84 사이트: harm(null) vs help(good) 분리력 (Cohen d · AUC)
  (B) 게이밍 37 사이트 × 6 변형: 단측 벌 `max(0, inv−τ)` 에서 적대 변형이 실제 혼잣말보다
      **덜 맞는가**(= 이득이 있는가)

채점 백엔드는 HF forward (vLLM prompt_logprobs 는 이 환경에서 EngineCore segfault) —
`score_inv(_ref_scorer=...)` 주입구로 넣는다. verl 을 부르지 않는다.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.environ.get("REPO", "/home/jovyan/beomi/splee/metacognition-math"))
from src.training import countdown_inv as CI                       # noqa: E402
from src.training.countdown_task import extract_expr, swap_op_decoy  # noqa: E402

W = "/home/jovyan/beomi/splee"
MODEL = os.environ.get(
    "PROBE_MODEL",
    "/home/jovyan/.cache/huggingface/hub/models--Qwen--Qwen3-4B/"
    "snapshots/1cfa9a7208912126459214e8b04321603b3df60c")


# ── 게이밍 변형 — `scripts/gaming_probe.variants` 와 **바이트 동일** ────────────
def variants(site, rng):
    wit = str(site["witness"]); tgt = int(site["target"])
    bad = swap_op_decoy(wit, list(site["nums"]), tgt, random.Random(1))
    if isinstance(bad, tuple):
        bad = bad[0]
    bad = bad or "(1+1)"
    out = {"real": str(site["meta_raw"])}
    out["conf_wrong"] = (f"<meta>\nconfidence: 0.95\nThe expression {bad} reaches the target "
                         f"{tgt} exactly, so this approach is confirmed correct.\ndecision: verify\n</meta>")
    words = "quixotic penumbra zephyr tessellate obsidian cadence lattice vermilion".split()
    out["gibberish"] = ("<meta>\nconfidence: 0.5\n" + " ".join(rng.choice(words) for _ in range(28))
                        + "\ndecision: redirect\n</meta>")
    out["long_generic"] = ("<meta>\nconfidence: 0.3\n" + ("This puzzle requires careful consideration "
        "of many possible groupings and operations, and it is important to remain systematic. " * 4)
        + "\ndecision: redirect\n</meta>")
    out["redirect_spam"] = ("<meta>\nconfidence: 0.1\nThis is not working. I should redirect. "
        "Redirect to another family. Redirect again. A different direction is needed. Redirect."
        "\ndecision: redirect\n</meta>")
    out["answer_leak"] = (f"<meta>\nconfidence: 0.9\nThe solution is {wit} which equals {tgt}."
                          f"\ndecision: verify\n</meta>")
    return out


VARIANT_ORDER = ["real", "conf_wrong", "gibberish", "long_generic", "redirect_spam", "answer_leak"]


def make_hf_scorer(model, batch_size=2, max_len=7000):
    """`score_inv(_ref_scorer=...)` 규약: (arm_prompts, arm_resps) → row 별 logp 배열."""
    import torch

    def _score(prompts, resps):
        n = len(prompts)
        out = [np.zeros(0) for _ in range(n)]
        todo = [i for i in range(n)
                if resps[i] and len(prompts[i]) >= 1 and len(prompts[i]) + len(resps[i]) <= max_len]
        todo.sort(key=lambda i: len(prompts[i]) + len(resps[i]))
        for lo in range(0, len(todo), batch_size):
            idxs = todo[lo:lo + batch_size]
            seqs = [list(prompts[i]) + list(resps[i]) for i in idxs]
            L = max(len(s) for s in seqs)
            ids = torch.zeros((len(seqs), L), dtype=torch.long)
            att = torch.zeros((len(seqs), L), dtype=torch.long)
            for j, sq in enumerate(seqs):
                ids[j, :len(sq)] = torch.tensor(sq)
                att[j, :len(sq)] = 1
            with torch.no_grad():
                lg = model(input_ids=ids.to(model.device),
                           attention_mask=att.to(model.device)).logits
            for j, i in enumerate(idxs):
                c, t = prompts[i], resps[i]
                pos = torch.arange(len(c) - 1, len(c) - 1 + len(t), device=lg.device)
                lsm = torch.log_softmax(lg[j, pos, :].float(), dim=-1)
                tid = torch.tensor(list(t), device=lg.device)
                out[i] = lsm[torch.arange(len(t), device=lg.device), tid].double().cpu().numpy()
            del lg
            done = min(lo + batch_size, len(todo))
            if done % 40 < batch_size or done == len(todo):
                print(f"[inv] scored {done}/{len(todo)}", flush=True)
        return out

    return _score


def chat_text(tok, prompt_json) -> str:
    raw = json.loads(prompt_json) if isinstance(prompt_json, str) else prompt_json
    msgs = [dict(m) for m in raw]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / s) if s > 0 else float("nan")


def auc(y, s):
    from scipy.stats import rankdata
    y, s = np.asarray(y, float), np.asarray(s, float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    if len(set(y.tolist())) < 2:
        return float("nan")
    r = rankdata(s)
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_auc_ci(y, s, n=4000, seed=0):
    rng = np.random.RandomState(seed)
    y, s = np.asarray(y), np.asarray(s)
    vals = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if len(set(y[i].tolist())) < 2:
            continue
        v = auc(y[i], s[i])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=f"{W}/cd6_work/probe/labels_extreme_all.parquet")
    ap.add_argument("--sites_glob", default=f"{W}/cd6_work/probe*/sites_shard*.parquet")
    ap.add_argument("--game_sites", default=f"{W}/cd6_work/probe/game_sites.txt")
    ap.add_argument("--out_dir", default=f"{W}/cd6_work/probe/inv_unified")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip_gaming", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.out_dir); outdir.mkdir(parents=True, exist_ok=True)

    lab = pd.read_parquet(args.labels)
    sites = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(args.sites_glob))],
                      ignore_index=True).drop_duplicates("site_id")
    df = lab.merge(sites, on="site_id", how="inner", suffixes=("", "_s"))
    if args.limit:
        h = max(1, args.limit // 2)
        df = pd.concat([df[df.label == "good"].head(h), df[df.label == "null"].head(h)])
    print(f"[inv] 라벨 사이트 {len(df)} (분포 {df.label.value_counts().to_dict()})", flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    scorer = make_hf_scorer(model, batch_size=args.batch_size)

    def run(rows_df, meta_override=None, tag=""):
        """scope 2종 × form 2종 × agg 3종 = 12 자를 **forward 2회**로 전부 읽는다.

        ★scope 마다 채점 대상 토큰열이 다르므로 forward 는 scope 당 한 번이다.
          form·agg 는 같은 logp 에서 읽기만 갈아 끼운다.
          `build_inv_arms` / `read_inv_from_ref_logprobs` 는 학습이 부르는 그 함수 그대로다.
        """
        P, R, Wt, T, E, keep = [], [], [], [], [], []
        for _, r in rows_df.iterrows():
            text = str(r["response_text"])
            if meta_override is not None:
                mt = meta_override[r["site_id"]]
                text = text[:int(r["meta_start"])] + mt + text[int(r["meta_end"]):]
            P.append(chat_text(tok, r["prompt_json"]))
            R.append(text)
            Wt.append(str(r["witness"]))
            T.append(int(r["target"]))
            E.append(extract_expr(text) or "")
            keep.append(r["site_id"])

        recs = []
        meta = {}
        for scope in CI.INV_SCOPES:
            arm_p, arm_r, attempts, per_row, diag = CI.build_inv_arms(
                tok, P, R, Wt, T, E, scope=scope)
            print(f"[inv]{tag} scope={scope} 팔 {len(arm_p)} · attempts {len(attempts)} · "
                  f"leak={diag['leak_blocked']} short={diag['short_prose']} "
                  f"fclaim={diag['false_claim']} span_err={diag['span_error']}", flush=True)
            ref_lp = scorer(arm_p, arm_r) if attempts else []
            by_row = {at.row: k for k, at in enumerate(attempts)}
            for form in CI.INV_FORMS:
                for agg in CI.INV_AGGS:
                    vals = CI.read_inv_from_ref_logprobs(ref_lp, attempts, agg, form)
                    for i2, sid in enumerate(keep):
                        k = by_row.get(i2)
                        recs.append({"site_id": sid, "scope": scope, "form": form,
                                     "agg": agg,
                                     "value": float(vals[k]) if k is not None else np.nan})
            for i2, sid in enumerate(keep):
                pr = per_row[i2]
                meta[sid] = {"false_claim": pr["inv_false_claim"], "leak": pr["inv_leak"],
                             "n_prose": pr["inv_n_prose"]}
        d = pd.DataFrame(recs)
        d["ruler"] = d["scope"] + "/" + d["form"] + "/" + d["agg"]
        d["false_claim"] = d.site_id.map(lambda x: meta[x]["false_claim"])
        d["leak"] = d.site_id.map(lambda x: meta[x]["leak"])
        return d

    RULERS = [f"{sc}/{fm}/{ag}" for sc in CI.INV_SCOPES
              for fm in CI.INV_FORMS for ag in CI.INV_AGGS]

    # ── (A) 라벨 ────────────────────────────────────────────────────────────
    A = run(df, None, " label")
    lmap = dict(zip(df.site_id, df.label))
    A["label"] = A.site_id.map(lmap)
    A.to_parquet(outdir / "inv_labels.parquet")
    print("\n## (A) 라벨 분리력 — harm(null) vs help(good). 12 자 전부.\n")
    print("| 자 (scope/form/agg) | harm mean | help mean | d(harm−help) | AUC(good=1) | AUC 95%CI | n |")
    print("|---|---|---|---|---|---|---|")
    lab_stats = {}
    for rl in RULERS:
        sub = A[A.ruler == rl]
        h = sub.loc[sub.label == "null", "value"].to_numpy(float)
        g = sub.loc[sub.label == "good", "value"].to_numpy(float)
        y = (sub.label == "good").astype(int).to_numpy()
        v = sub["value"].to_numpy(float)
        a = auc(y, v); lo, hi = boot_auc_ci(y, v)
        lab_stats[rl] = dict(d=cohen_d(h, g), auc=a, lo=lo, hi=hi,
                             q=np.nanpercentile(v[np.isfinite(v)], [25, 50, 75]))
        print(f"| {rl} | {np.nanmean(h):+.3f} | {np.nanmean(g):+.3f} | "
              f"{lab_stats[rl]['d']:+.3f} | {a:.3f} | [{lo:.3f}, {hi:.3f}] | "
              f"{int(np.isfinite(v).sum())} |")

    if args.skip_gaming:
        return

    # ── (B) 게이밍 ──────────────────────────────────────────────────────────
    keepset = {l.strip() for l in open(args.game_sites) if l.strip()}
    gs = sites[sites.site_id.isin(keepset)].reset_index(drop=True)
    rng = random.Random(0)
    var_map = {sid: variants(gs[gs.site_id == sid].iloc[0], rng) for sid in gs.site_id}
    print(f"\n[inv] 게이밍 사이트 {len(gs)}개 × {len(VARIANT_ORDER)} 변형", flush=True)
    frames = []
    for vn in VARIANT_ORDER:
        ov = {sid: var_map[sid][vn] for sid in gs.site_id}
        d = run(gs, ov, f" game/{vn}")
        d["variant"] = vn
        frames.append(d)
    B = pd.concat(frames, ignore_index=True)
    B.to_parquet(outdir / "inv_gaming.parquet")

    print("\n## (B) 게이밍 원값 평균 (높을수록 «해로움» 쪽 = 벌을 더 받는다)\n")
    print("| 자 | " + " | ".join(VARIANT_ORDER) + " |")
    print("|---|" + "---|" * len(VARIANT_ORDER))
    for rl in RULERS:
        cells = []
        for vn in VARIANT_ORDER:
            v = B.loc[(B.ruler == rl) & (B.variant == vn), "value"].to_numpy(float)
            cells.append(f"{np.nanmean(v):+.3f}")
        print(f"| {rl} | " + " | ".join(cells) + " |")

    print("\n## (B2) 단측 벌 pen(x)=max(0,x−τ), τ=라벨분포 p50 — «real 보다 덜 맞는» 비율\n")
    print("| 자 | τ(p50) | " + " | ".join(v for v in VARIANT_ORDER if v != "real")
          + " | real pen>0 |")
    print("|---|---|" + "---|" * len(VARIANT_ORDER))
    for rl in RULERS:
        tau = float(lab_stats[rl]["q"][1])
        w = B[B.ruler == rl].pivot_table(index="site_id", columns="variant", values="value")
        if "real" not in w.columns:
            continue
        w = w.dropna(subset=["real"])
        pr = np.maximum(0.0, w["real"].to_numpy(float) - tau)
        cells = []
        for vn in VARIANT_ORDER:
            if vn == "real":
                continue
            if vn not in w.columns:
                cells.append("n/a"); continue
            pv = np.maximum(0.0, w[vn].to_numpy(float) - tau)
            ok = np.isfinite(pv) & np.isfinite(pr)
            cells.append(f"{float((pv[ok] < pr[ok]).mean()):.3f}" if ok.sum() else "n/a")
        cells.append(f"{float(np.nanmean(pr > 0)):.3f}")
        print(f"| {rl} | {tau:+.3f} | " + " | ".join(cells) + " |")

    print(f"\n[inv] wrote {outdir}/inv_labels.parquet · inv_gaming.parquet", flush=True)


if __name__ == "__main__":
    main()
