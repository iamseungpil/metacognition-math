"""Is the instruct gain just "the control breaks"?

Decomposes shiftonly - gandhi (the PURE PMI-shift reward axis, C-001 +4.38pp) and
pmishift - gandhi (the full package) into:
  (a) truncation rescue   -- gain that exists only because the control fails to terminate
  (b) terminated-only gap -- gain among responses that BOTH arms finished cleanly

If (a) carries the whole effect, the method is a length/termination fix, not
metacognition. Grading: math_verify with $...$ wrapping (C-027).
"""
import os, warnings
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore")
R = "iamseungpil/metacot-h200-triobj-dcpo-v3"
ARMS = {a: f"eval/{a}_1030_v2/{a}_gs300_16k_n8_%s/{a}_gs300_16k_n8_%s.parquet"
        for a in ["gandhi", "shiftonly", "pmishift"]}


def _g(args):
    ext, gold = args
    if ext is None or (isinstance(ext, float) and np.isnan(ext)):
        return 0
    try:
        from math_verify import parse, verify
        return int(bool(verify(parse("$" + str(gold) + "$"), parse("$" + str(ext) + "$"))))
    except Exception:
        return 0


def grade(df, nproc=32):
    with ProcessPoolExecutor(max_workers=nproc) as ex:
        return np.array(list(ex.map(_g, zip(df["answer_extracted"], df["gold_answer"]),
                                    chunksize=16)), dtype=float)


def boot(d, n=10000, seed=20260805):
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def main():
    for bench in ["math500", "gsm8k"]:
        print(f"\n{'='*84}\n{bench}\n{'='*84}")
        S = {}
        for a, tpl in ARMS.items():
            df = pd.read_parquet(hf_hub_download(R, tpl % (bench, bench)))
            df["mv"] = grade(df)
            df["term"] = (df["finish_reason"] == "stop")
            S[a] = df
            print(f"  {a:10s} n={len(df):5d}  acc={df.mv.mean()*100:6.2f}  "
                  f"trunc={(~df.term).mean()*100:5.2f}%  toks={df.completion_length_tokens.mean():7.1f}  "
                  f"acc|terminated={df.mv[df.term].mean()*100:6.2f}")

        for A, B, tag in [("gandhi", "shiftonly", "PURE REWARD  shiftonly - gandhi"),
                          ("gandhi", "pmishift", "FULL PACKAGE pmishift - gandhi")]:
            a, b = S[A], S[B]
            # per-question, all samples
            ga = a.groupby("question", sort=False).agg(acc=("mv", "mean"), term=("term", "mean"))
            gb = b.groupby("question", sort=False).agg(acc=("mv", "mean"), term=("term", "mean"))
            idx = ga.index.intersection(gb.index)
            ga, gb = ga.loc[idx], gb.loc[idx]
            m, lo, hi = boot((gb.acc.values - ga.acc.values) * 100)
            print(f"\n  --- {tag} ---")
            print(f"    headline            {m:+6.2f}pp [{lo:+6.2f},{hi:+6.2f}]   "
                  f"trunc {(1-ga.term.mean())*100:5.2f}% -> {(1-gb.term.mean())*100:5.2f}%")

            # (b) terminated-only: restrict to SAMPLES both arms terminated, matched by
            # (question, sample_idx) so the comparison stays paired.
            ka = a.set_index(["question", "sample_idx"])
            kb = b.set_index(["question", "sample_idx"])
            common = ka.index.intersection(kb.index)
            ka, kb = ka.loc[common], kb.loc[common]
            both = ka.term.values & kb.term.values
            print(f"    both terminated     n={both.sum():5d}/{len(both)} ({both.mean()*100:5.2f}%)")
            if both.sum():
                t = pd.DataFrame({"q": [i[0] for i in common[both]],
                                  "a": ka.mv.values[both], "b": kb.mv.values[both]})
                gt = t.groupby("q").mean()
                m2, lo2, hi2 = boot((gt.b.values - gt.a.values) * 100)
                print(f"    terminated-only     {m2:+6.2f}pp [{lo2:+6.2f},{hi2:+6.2f}]"
                      f"   <-- (b) real reasoning gain")
                print(f"    rescue = headline - terminated-only = {m-m2:+6.2f}pp"
                      f"   ({(m-m2)/m*100 if m else float('nan'):5.1f}% of headline)")

            # where the control failed to terminate, what did the treated arm do?
            ctrl_trunc = ~ka.term.values
            if ctrl_trunc.sum():
                print(f"    on control-truncated rows (n={ctrl_trunc.sum()}): "
                      f"ctrl acc={ka.mv.values[ctrl_trunc].mean()*100:5.2f}  "
                      f"treat acc={kb.mv.values[ctrl_trunc].mean()*100:5.2f}  "
                      f"treat terminated={kb.term.values[ctrl_trunc].mean()*100:5.2f}%")


if __name__ == "__main__":
    main()
