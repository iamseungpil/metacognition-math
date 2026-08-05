"""Paired eval analysis: b0p vs b2p (RQ1) + b2p vs b2p_pair (A-vs-A job drift).

math_verify regrade (stored is_correct is the broken older grader, C-001),
MATH500 level join from the source dataset, truncation rates, paired bootstrap.
"""
import os, sys, json, warnings
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore")
R = "iamseungpil/metacot-h200-triobj-dcpo-v3"
OUT = os.environ["CLAUDE_JOB_DIR"] + "/tmp"

ARMS = {
    "b0p":      "eval/rq3v2f_b0p_1030/rq3v2f_b0p_gs300_16k_n8_%s/rq3v2f_b0p_gs300_16k_n8_%s.parquet",
    "b2p":      "eval/rq3v2f_b2p_1030/rq3v2f_b2p_gs300_16k_n8_%s/rq3v2f_b2p_gs300_16k_n8_%s.parquet",
    "b2p_pair": "eval/rq3v2f_b2p_1030_pair/rq3v2f_b2p_gs300_16k_n8_%s/rq3v2f_b2p_gs300_16k_n8_%s.parquet",
}


def _grade(args):
    """math_verify on one (extracted, gold) pair. Returns 0/1."""
    ext, gold = args
    if ext is None or (isinstance(ext, float) and np.isnan(ext)):
        return 0
    try:
        from math_verify import parse, verify
        g = parse("$" + str(gold) + "$")
        p = parse("$" + str(ext) + "$")
        return int(bool(verify(g, p)))
    except Exception:
        return 0


def regrade(df, nproc=32):
    pairs = list(zip(df["answer_extracted"].tolist(), df["gold_answer"].tolist()))
    with ProcessPoolExecutor(max_workers=nproc) as ex:
        res = list(ex.map(_grade, pairs, chunksize=16))
    return np.array(res, dtype=float)


def load(arm, bench):
    path = ARMS[arm] % (bench, bench)
    return pd.read_parquet(hf_hub_download(R, path))


def levels():
    """MATH500 level per question, from the source dataset."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return {r["problem"]: int(r["level"]) for r in ds}


def per_problem(df, mv):
    """Collapse n=8 samples -> per-question mean accuracy + truncation rate."""
    t = df.assign(mv=mv, trunc=(df["finish_reason"] != "stop").astype(float))
    g = t.groupby("question", sort=False).agg(
        acc=("mv", "mean"), trunc=("trunc", "mean"),
        toks=("completion_length_tokens", "mean"), n=("mv", "size"))
    return g


def boot(a, b, n=10000, seed=20260805):
    """Paired bootstrap on the per-question difference (pp)."""
    d = (b - a) * 100.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return d.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def main():
    lvl = levels()
    store = {}
    for bench in ["math500", "gsm8k", "aime2024"]:
        for arm in ARMS:
            df = load(arm, bench)
            mv = regrade(df)
            store[(arm, bench)] = per_problem(df, mv)
            old = df["is_correct"].astype(float).mean() * 100
            new = mv.mean() * 100
            print(f"{arm:9s} {bench:9s} n={len(df):5d}  stored={old:6.2f}  math_verify={new:6.2f}  "
                  f"trunc={(df['finish_reason']!='stop').mean()*100:5.2f}%  "
                  f"toks={df['completion_length_tokens'].mean():7.1f}", flush=True)

    def compare(a, b, tag):
        print(f"\n{'='*72}\n{tag}   ({b} - {a})\n{'='*72}")
        for bench in ["math500", "gsm8k", "aime2024"]:
            ga, gb = store[(a, bench)], store[(b, bench)]
            common = ga.index.intersection(gb.index)
            ga, gb = ga.loc[common], gb.loc[common]
            m, lo, hi = boot(ga["acc"].values, gb["acc"].values)
            print(f"  {bench:9s} {ga['acc'].mean()*100:6.2f} -> {gb['acc'].mean()*100:6.2f}  "
                  f"delta={m:+6.2f}pp CI[{lo:+6.2f},{hi:+6.2f}]  "
                  f"trunc {ga['trunc'].mean()*100:5.2f}->{gb['trunc'].mean()*100:5.2f}%")
        # MATH500 level slope
        ga, gb = store[(a, "math500")], store[(b, "math500")]
        common = ga.index.intersection(gb.index)
        ga, gb = ga.loc[common], gb.loc[common]
        lv = np.array([lvl.get(q, 0) for q in common])
        easy, hard = np.isin(lv, [1, 2]), np.isin(lv, [4, 5])
        de = (gb["acc"].values[easy] - ga["acc"].values[easy]).mean() * 100
        dh = (gb["acc"].values[hard] - ga["acc"].values[hard]).mean() * 100
        # bootstrap the slope
        rng = np.random.default_rng(20260805)
        d_all = (gb["acc"].values - ga["acc"].values) * 100
        ie, ih = np.where(easy)[0], np.where(hard)[0]
        s = [d_all[rng.choice(ih, len(ih))].mean() - d_all[rng.choice(ie, len(ie))].mean()
             for _ in range(10000)]
        print(f"  L1-2 (n={easy.sum():3d}) {de:+6.2f}pp | L4-5 (n={hard.sum():3d}) {dh:+6.2f}pp | "
              f"slope {dh-de:+6.2f}pp CI[{np.percentile(s,2.5):+6.2f},{np.percentile(s,97.5):+6.2f}]")
        for L in [1, 2, 3, 4, 5]:
            m = lv == L
            if m.sum():
                print(f"    L{L} n={m.sum():3d}  {ga['acc'].values[m].mean()*100:6.2f} -> "
                      f"{gb['acc'].values[m].mean()*100:6.2f}  ({d_all[m].mean():+6.2f}pp)")

    compare("b0p", "b2p", "RQ1 (within-job, paired eval)")
    compare("b2p", "b2p_pair", "A-vs-A  same checkpoint, different eval job")
    compare("b0p", "b2p_pair", "RQ1 using the duplicate b2p arm")


if __name__ == "__main__":
    main()
