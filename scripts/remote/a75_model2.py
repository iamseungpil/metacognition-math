"""수정 67 두 번째 모델(CPU 단계) — a56_hard_class.py 의 hc2 파이프라인을 모델·데이터 꼬리표(TG)만 바꿔 그대로 돈다(같은 hc2_src 문제 묶음,
같은 거르기·무게·관문). 옛 hc(수정 43) 데이터는 2507 전용이라 뺀다. 사용: python a75_model2.py mid TG h | weights TG MODEL | nopfx TG"""
import json, math, os, sys
from pathlib import Path
import os as _os
_ROOT = _os.environ.get("ROOT", "/hdd_data/seungpil")
_REPO = _os.environ.get("REPO") or str(__import__("pathlib").Path(__file__).resolve().parents[2])   # scripts/remote 복사본
_HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)
from a56_hard_class import D, EPS, GATE, MAXP, NCH, RIGHT_N, _load, multi  # noqa: E402


def mid(tg: str, h: str):
    import pandas as pd
    texts = [json.loads(x) for x in open(D + f"data/{tg}_pool_{h}.texts.jsonl")]
    right: dict = {}
    for r in texts:
        right.setdefault(r["problem_idx"], []).append(bool(r["correct"]))
    probe = {str(r["uid"]) for r in json.load(open(D + "eval/probe_commit_selection.json"))}   # 탐침과 같은 문제는 학습에서 뺀다(DAPO 47/9,600)
    uid = {r["problem_idx"]: str(r["uid"]) for r in texts}
    keep = {p for p, v in right.items() if 1 <= sum(v) <= 3 and uid[p] not in probe}
    for r in texts:
        r["selected"] = r["problem_idx"] in keep
    pd.read_parquet(D + f"data/{tg}_pool_{h}.parquet").to_parquet(D + f"data/{tg}_mid_{h}.parquet", index=False)
    with open(D + f"data/{tg}_mid_{h}.texts.jsonl", "w") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in texts)
    print(json.dumps({"chunk": h, "problems": len(right), "mid": len(keep)}))


def weights(tg: str, model: str):
    import pandas as pd
    from transformers import AutoTokenizer
    from mc import context as ctx
    out = D + f"data/mc_pfx_commit_{tg}_screen_plain_le6000.parquet"
    tok = AutoTokenizer.from_pretrained(model)
    new = [(D + f"data/{tg}_pfx_{h}.parquet", D + f"eval/{tg}_sel_{h}.json", [D + f"eval/{tg}_screen_{h}/texts.jsonl"]) for h in range(NCH)]
    have = [x for x in new if Path(x[2][0]).with_name("summary.json").exists()]
    parts = [_load(*x) for x in have]
    df = pd.concat([d for d, *_ in parts], ignore_index=True)
    ps, nxt, banks = (sum((x[j] for x in parts), []) for j in (1, 2, 3))
    ext = [dict(e) for e in df["extra_info"]]
    L = [len(tok.encode(ctx.turn1_prompt(tok, q, "plain") + e["prefix"], add_special_tokens=False)) for q, e in zip(df["problem"], ext)]
    keep = [i for i in range(len(df)) if ps[i] is not None and not (multi(df["problem"][i]) or nxt[i]) and L[i] <= MAXP]
    cls = {c: [i for i in keep if bool(ext[i]["first_wrong"]) == c] for c in (True, False)}
    info = lambda ix: sum(0 < ps[i] < 1 for i in ix)  # noqa: E731
    eps = {True: EPS[True], False: EPS[True] if info(cls[False]) >= RIGHT_N else EPS[False]}
    raw = {i: eps[bool(ext[i]["first_wrong"])] + math.sqrt(ps[i] * (1 - ps[i])) for i in keep}
    res = {"tag": tg, "model": model, "n_rows": len(df), "n_out": len(keep), "wrong": len(cls[True]), "right": len(cls[False]),
           "informative_wrong": info(cls[True]), "informative_right": info(cls[False]), "problems": len({ext[i]["problem_uid"] for i in keep}),
           "chunks_missing": [h for h, x in enumerate(new) if x not in have], "gate": GATE}
    res["gate_pass"] = res["informative_wrong"] >= GATE
    res["forced"] = os.environ.get("FORCE") == "1"   # 수정 69: 관문 미달이어도 «탐색» 으로 데이터를 쓴다(보고 때 표시)
    Path(out.replace(".parquet", ".summary.json")).write_text(json.dumps(res, indent=1))
    print(json.dumps(res))
    if not (res["gate_pass"] or res["forced"]):
        sys.exit(3)
    for c, ix in cls.items():
        tot = sum(raw[i] for i in ix)
        for i in ix:
            ext[i].update(sample_weight=(len(keep) / 2) * raw[i] / tot, p_succ=ps[i], src=tg, **banks[i])
    o = df.iloc[keep].copy()
    o["extra_info"] = [ext[i] for i in keep]
    o.reset_index(drop=True).to_parquet(out, index=False)


def nopfx(tg: str):
    """보통 GRPO 데이터(a69 와 같은 규칙) — 같은 모델이 고른 문제마다 한 행, 앞부분 비움."""
    import pandas as pd
    df = pd.read_parquet(D + f"data/mc_pfx_commit_{tg}_screen_plain_le6000.parquet")
    df = df[~df["extra_info"].map(lambda e: e["problem_uid"]).duplicated()].copy()
    df["extra_info"] = [{**e, "prefix": "", "first_answer": "", "first_wrong": type(e["first_wrong"])(False), "pfx_id": e["problem_uid"] + ":np",
                         "weight": 1.0, "sample_weight": 1.0} for e in df["extra_info"]]
    df.to_parquet(D + f"data/mc_nopfx_{tg}.parquet")
    print(len(df), "problems → nopfx", tg)


if __name__ == "__main__":
    a = sys.argv[1:]
    {"mid": lambda: mid(a[1], a[2]), "weights": lambda: weights(a[1], a[2]), "nopfx": lambda: nopfx(a[1])}[a[0]]()
