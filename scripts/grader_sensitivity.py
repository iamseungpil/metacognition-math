"""Does the grader choice flip RQ1 / A-vs-A?

Three graders on the same parquets:
  stored  — is_correct as written by the eval job (C-001: known broken)
  canon   — src/training/rewards.py convention: parse(str(x), extraction_mode='first_match')
  wrapped — parse('$'+str(x)+'$'), which is what math_verify wants for bare answers
"""
import os, warnings
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore")
R = "iamseungpil/metacot-h200-triobj-dcpo-v3"
ARMS = {
    "b0p":      "eval/rq3v2f_b0p_1030/rq3v2f_b0p_gs300_16k_n8_%s/rq3v2f_b0p_gs300_16k_n8_%s.parquet",
    "b2p":      "eval/rq3v2f_b2p_1030/rq3v2f_b2p_gs300_16k_n8_%s/rq3v2f_b2p_gs300_16k_n8_%s.parquet",
    "b2p_pair": "eval/rq3v2f_b2p_1030_pair/rq3v2f_b2p_gs300_16k_n8_%s/rq3v2f_b2p_gs300_16k_n8_%s.parquet",
}


def _g(args):
    ext, gold, mode = args
    if ext is None or (isinstance(ext, float) and np.isnan(ext)):
        return 0
    try:
        from math_verify import parse, verify
        if mode == "wrapped":
            g, p = parse("$" + str(gold) + "$"), parse("$" + str(ext) + "$")
        else:
            g = parse(str(gold), extraction_mode="first_match", parsing_timeout=None)
            p = parse(str(ext), extraction_mode="first_match", parsing_timeout=None)
        return int(bool(verify(g, p)))
    except Exception:
        return 0


def grade(df, mode, nproc=32):
    args = [(a, b, mode) for a, b in zip(df["answer_extracted"], df["gold_answer"])]
    with ProcessPoolExecutor(max_workers=nproc) as ex:
        return np.array(list(ex.map(_g, args, chunksize=16)), dtype=float)


def boot(d, n=10000, seed=20260805):
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def main():
    from datasets import load_dataset
    lvl = {r["problem"]: int(r["level"]) for r in load_dataset("HuggingFaceH4/MATH-500", split="test")}

    raw = {a: pd.read_parquet(hf_hub_download(R, ARMS[a] % ("math500", "math500"))) for a in ARMS}
    acc = {}
    for a, df in raw.items():
        for mode in ["stored", "canon", "wrapped"]:
            v = df["is_correct"].astype(float).values if mode == "stored" else grade(df, mode)
            acc[(a, mode)] = df.assign(v=v).groupby("question", sort=False)["v"].mean()
        print(f"{a:9s} " + "  ".join(
            f"{m}={acc[(a,m)].mean()*100:6.2f}" for m in ["stored", "canon", "wrapped"]), flush=True)

    print(f"\n{'comparison':28s} {'grader':9s} {'overall':>22s} {'slope L4-5 minus L1-2':>26s}")
    print("-" * 92)
    for tag, (A, B) in [("RQ1  b2p - b0p", ("b0p", "b2p")),
                        ("RQ1  b2p_pair - b0p", ("b0p", "b2p_pair")),
                        ("A-vs-A  pair - orig", ("b2p", "b2p_pair"))]:
        for mode in ["stored", "canon", "wrapped"]:
            sa, sb = acc[(A, mode)], acc[(B, mode)]
            idx = sa.index.intersection(sb.index)
            d = (sb.loc[idx].values - sa.loc[idx].values) * 100
            lv = np.array([lvl.get(q, 0) for q in idx])
            e, h = np.isin(lv, [1, 2]), np.isin(lv, [4, 5])
            m, lo, hi = boot(d)
            rng = np.random.default_rng(20260805)
            ie, ih = np.where(e)[0], np.where(h)[0]
            s = np.array([d[rng.choice(ih, len(ih))].mean() - d[rng.choice(ie, len(ie))].mean()
                          for _ in range(10000)])
            sl = d[h].mean() - d[e].mean()
            print(f"{tag:28s} {mode:9s} {m:+6.2f}pp [{lo:+6.2f},{hi:+6.2f}]  "
                  f"{sl:+6.2f}pp [{np.percentile(s,2.5):+6.2f},{np.percentile(s,97.5):+6.2f}]")
        print()


if __name__ == "__main__":
    main()
