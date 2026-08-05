"""RQ2 = b3s - b2p on held-out, with NFKC normalization reported both ways (E-138).

b3s writes digit homoglyphs (Khmer, Tamil, fullwidth) that math_verify scores zero even when
the value is right, so an un-normalized number alone cannot separate a reasoning failure from
a script failure. Primary metric stays un-normalized per the pre-registration.
Grading uses the $...$ wrapping (C-027); the canonical grader is reported alongside.
"""
import os, unicodedata, warnings
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore")
R = "iamseungpil/metacot-h200-triobj-dcpo-v3"
ARMS = {
    "b3s":      "eval/rq3v2f_b3s_1030/rq3v2f_b3s_gs300_16k_n8_%s/rq3v2f_b3s_gs300_16k_n8_%s.parquet",
    "b2p":      "eval/rq3v2f_b2p_1030/rq3v2f_b2p_gs300_16k_n8_%s/rq3v2f_b2p_gs300_16k_n8_%s.parquet",
    "b2p_pair": "eval/rq3v2f_b2p_1030_pair/rq3v2f_b2p_gs300_16k_n8_%s/rq3v2f_b2p_gs300_16k_n8_%s.parquet",
    "b0p":      "eval/rq3v2f_b0p_1030/rq3v2f_b0p_gs300_16k_n8_%s/rq3v2f_b0p_gs300_16k_n8_%s.parquet",
}


def _g(args):
    ext, gold, nfkc = args
    if ext is None or (isinstance(ext, float) and np.isnan(ext)):
        return 0
    try:
        from math_verify import parse, verify
        e = unicodedata.normalize("NFKC", str(ext)) if nfkc else str(ext)
        return int(bool(verify(parse("$" + str(gold) + "$"), parse("$" + e + "$"))))
    except Exception:
        return 0


def grade(df, nfkc, nproc=32):
    args = [(a, b, nfkc) for a, b in zip(df["answer_extracted"], df["gold_answer"])]
    with ProcessPoolExecutor(max_workers=nproc) as ex:
        return np.array(list(ex.map(_g, args, chunksize=16)), dtype=float)


def boot(d, n=10000, seed=20260805):
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def homoglyph_rate(df):
    """Fraction of extracted answers that change under NFKC — the E-138 population."""
    e = df["answer_extracted"].astype(str)
    return (e != e.map(lambda s: unicodedata.normalize("NFKC", s))).mean()


def main():
    from datasets import load_dataset
    lvl = {r["problem"]: int(r["level"])
           for r in load_dataset("HuggingFaceH4/MATH-500", split="test")}

    raw, acc = {}, {}
    for a in ARMS:
        for bench in ["math500", "gsm8k", "aime2024"]:
            try:
                df = pd.read_parquet(hf_hub_download(R, ARMS[a] % (bench, bench)))
            except Exception as ex:
                print(f"  {a}/{bench}: 없음 ({type(ex).__name__})")
                continue
            raw[(a, bench)] = df
            for nf in [False, True]:
                v = grade(df, nf)
                acc[(a, bench, nf)] = df.assign(v=v).groupby("question", sort=False)["v"].mean()
            hg = homoglyph_rate(df)
            print(f"  {a:9s} {bench:9s} n={len(df):5d}  무정규화={acc[(a,bench,False)].mean()*100:6.2f}  "
                  f"NFKC={acc[(a,bench,True)].mean()*100:6.2f}  동형이의자={hg*100:5.2f}%  "
                  f"절단={(df['finish_reason']!='stop').mean()*100:5.2f}%", flush=True)

    def cmp(A, B, tag):
        print(f"\n{'='*78}\n{tag}   ({B} - {A})\n{'='*78}")
        for bench in ["math500", "gsm8k", "aime2024"]:
            for nf in [False, True]:
                if (A, bench, nf) not in acc or (B, bench, nf) not in acc:
                    continue
                sa, sb = acc[(A, bench, nf)], acc[(B, bench, nf)]
                idx = sa.index.intersection(sb.index)
                d = (sb.loc[idx].values - sa.loc[idx].values) * 100
                m, lo, hi = boot(d)
                lab = "NFKC   " if nf else "무정규화"
                star = " ★주지표" if (not nf and bench == "math500") else ""
                print(f"  {bench:9s} {lab}  {sa.loc[idx].mean()*100:6.2f} -> {sb.loc[idx].mean()*100:6.2f}  "
                      f"delta={m:+6.2f}pp CI[{lo:+6.2f},{hi:+6.2f}]{star}")
        # MATH500 난이도 기울기 (무정규화, 사전등록)
        if (A, "math500", False) in acc and (B, "math500", False) in acc:
            sa, sb = acc[(A, "math500", False)], acc[(B, "math500", False)]
            idx = sa.index.intersection(sb.index)
            d = (sb.loc[idx].values - sa.loc[idx].values) * 100
            lv = np.array([lvl.get(q, 0) for q in idx])
            e, h = np.isin(lv, [1, 2]), np.isin(lv, [4, 5])
            rng = np.random.default_rng(20260805)
            ie, ih = np.where(e)[0], np.where(h)[0]
            s = np.array([d[rng.choice(ih, len(ih))].mean() - d[rng.choice(ie, len(ie))].mean()
                          for _ in range(10000)])
            print(f"  L1-2(n={e.sum()}) {d[e].mean():+6.2f}pp | L4-5(n={h.sum()}) {d[h].mean():+6.2f}pp | "
                  f"★기울기 {d[h].mean()-d[e].mean():+6.2f}pp "
                  f"CI[{np.percentile(s,2.5):+6.2f},{np.percentile(s,97.5):+6.2f}]")

    cmp("b2p", "b3s", "RQ2  사전등록 주 비교 (cross-job)")
    cmp("b2p_pair", "b3s", "RQ2  중복 b2p 팔로 재계산")
    cmp("b0p", "b3s", "참고: b3s - b0p (프라이밍+보상 합산)")


if __name__ == "__main__":
    main()
