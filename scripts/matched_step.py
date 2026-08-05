"""Matched-step comparison across the four RQ3v2F arms (in-training metrics).

Arms: b0p (meta-removed twin, vanilla GRPO) / b2p (meta SFT2, vanilla GRPO)
      b3p (triobj, meta_floor 0.0 — invalidated) / b3s (triobj, meta_floor 0.05)

val594 is EXPLICITLY NOT for adjudication (21-38 problems per cell). This is a
health/trend read only. Reward scale -> accuracy via acc = (r+1)/2.
"""
import numpy as np, pandas as pd, wandb, warnings
warnings.filterwarnings("ignore")

RUNS = {"b0p": "rq3v2f-b0p-1", "b2p": "rq3v2f-b2p-2",
        "b3p": "rq3v2f-b3p-1", "b3s": "rq3v2f-b3s-1"}
B2P_ORIG = "rq3v2f-b2p-1"  # second half re-run -> same-arm A-vs-A
STEPS = [50, 100, 150, 200, 250, 300]

TRAIN = ["gdpo/meta_emission/mean", "dcpo/acc_with", "dcpo/acc_without",
         "actor/entropy", "actor/grad_norm", "critic/advantages/mean",
         "response_length/mean", "response_length/clip_ratio",
         "critic/score/mean", "dcpo/cf_text_rate"]


def fetch(api, rid, keys):
    r = api.run(f"gistdslab/metacot-dcpo-v4/{rid}")
    have = {k for k in r.summary.keys() if isinstance(k, str)}
    keys = [k for k in keys if k in have]
    rows = list(r.scan_history(keys=keys + ["_step"], page_size=2000))
    return pd.DataFrame(rows).set_index("_step").sort_index() if rows else pd.DataFrame()


def main():
    api = wandb.Api(timeout=180)
    # discover the val-aux cells present in b3s
    b3s = api.run(f"gistdslab/metacot-dcpo-v4/{RUNS['b3s']}")
    all_keys = [k for k in b3s.summary.keys() if isinstance(k, str)]
    cells = sorted({k.split("/")[1] + ("/" + k.split("/")[2] if k.count("/") > 3 else "")
                    for k in all_keys if k.startswith("val-aux/")})
    heads = sorted({k.rsplit("/", 2)[-2] for k in all_keys if k.startswith("val-aux/")})
    print("val-aux cells :", cells)
    print("val-aux heads :", heads, "\n")

    val_keys = [k for k in all_keys if k.startswith("val-aux/")]
    hist = {}
    for arm, rid in list(RUNS.items()) + [("b2p_orig", B2P_ORIG)]:
        hist[arm] = fetch(api, rid, val_keys + TRAIN)
        print(f"{arm:9s} {rid:14s} rows={len(hist[arm]):4d} "
              f"steps={hist[arm].index.min() if len(hist[arm]) else '-'}"
              f"..{hist[arm].index.max() if len(hist[arm]) else '-'}", flush=True)

    def at(arm, key, step, tol=3):
        df = hist[arm]
        if df.empty or key not in df.columns:
            return np.nan
        s = pd.to_numeric(df[key], errors="coerce").dropna()
        if s.empty:
            return np.nan
        near = s.index[np.abs(s.index - step) <= tol]
        return float(s.loc[near].iloc[-1]) if len(near) else np.nan

    def acc(arm, head, step):
        """Mean over val-aux cells of a reward head, converted to accuracy."""
        ks = [k for k in val_keys if k.endswith(f"/{head}/mean@1")]
        v = [at(arm, k, step) for k in ks]
        v = [x for x in v if not np.isnan(x)]
        return np.nan if not v else (np.mean(v) + 1) / 2 * 100

    def raw(arm, head, step):
        ks = [k for k in val_keys if k.endswith(f"/{head}/mean@1")]
        v = [at(arm, k, step) for k in ks]
        v = [x for x in v if not np.isnan(x)]
        return np.nan if not v else np.mean(v)

    arms = ["b0p", "b2p", "b3p", "b3s"]

    print("\n" + "=" * 88)
    print("val594 CORRECTNESS  (acc%, mean over cells)   -- health read, NOT adjudication")
    print("=" * 88)
    print(f"{'gs':>5s} " + "".join(f"{a:>9s}" for a in arms) +
          f"{'b2p-b0p':>10s}{'b3s-b2p':>10s}{'b3p-b2p':>10s}{'b2pAvA':>9s}")
    for s in STEPS:
        v = {a: acc(a, "correctness", s) for a in arms}
        o = acc("b2p_orig", "correctness", s)
        f = lambda x: f"{x:9.2f}" if not np.isnan(x) else f"{'-':>9s}"
        d = lambda x, y: f"{x-y:+10.2f}" if not (np.isnan(x) or np.isnan(y)) else f"{'-':>10s}"
        print(f"{s:5d} " + "".join(f(v[a]) for a in arms) +
              d(v['b2p'], v['b0p']) + d(v['b3s'], v['b2p']) + d(v['b3p'], v['b2p']) +
              (f"{o-v['b2p']:+9.2f}" if not (np.isnan(o) or np.isnan(v['b2p'])) else f"{'-':>9s}"))

    for head in [h for h in heads if h != "correctness"]:
        vals = {(a, s): raw(a, head, s) for a in arms for s in STEPS}
        if all(np.isnan(x) for x in vals.values()):
            continue
        print(f"\n--- val-aux {head} (raw reward, mean over cells) ---")
        print(f"{'gs':>5s} " + "".join(f"{a:>10s}" for a in arms))
        for s in STEPS:
            print(f"{s:5d} " + "".join(
                f"{vals[(a,s)]:10.4f}" if not np.isnan(vals[(a, s)]) else f"{'-':>10s}"
                for a in arms))

    print("\n" + "=" * 88)
    print("TRAINING-SIDE METRICS at matched steps")
    print("=" * 88)
    for k in TRAIN:
        vals = {(a, s): at(a, k, s) for a in arms for s in STEPS}
        if all(np.isnan(x) for x in vals.values()):
            continue
        print(f"\n--- {k} ---")
        print(f"{'gs':>5s} " + "".join(f"{a:>12s}" for a in arms))
        for s in STEPS:
            print(f"{s:5d} " + "".join(
                f"{vals[(a,s)]:12.4f}" if not np.isnan(vals[(a, s)]) else f"{'-':>12s}"
                for a in arms))

    print("\n" + "=" * 88)
    print("PER-CELL correctness acc% at the last matched step both b2p and b3s have")
    print("=" * 88)
    s = max(x for x in STEPS if not np.isnan(acc("b3s", "correctness", x)))
    print(f"(gs{s})")
    ks = sorted(k for k in val_keys if k.endswith("/correctness/mean@1"))
    print(f"{'cell':46s}" + "".join(f"{a:>9s}" for a in arms) + f"{'b3s-b2p':>10s}")
    for k in ks:
        cell = k[len("val-aux/"):-len("/correctness/mean@1")]
        v = {a: (at(a, k, s) + 1) / 2 * 100 for a in arms}
        f = lambda x: f"{x:9.2f}" if not np.isnan(x) else f"{'-':>9s}"
        dd = v['b3s'] - v['b2p']
        print(f"{cell:46s}" + "".join(f(v[a]) for a in arms) +
              (f"{dd:+10.2f}" if not np.isnan(dd) else f"{'-':>10s}"))


if __name__ == "__main__":
    main()
