"""수정 44 «고칠 수 있는 경계» 교실(CPU 단계 — GPU 단계는 mc.pool·mc.shift_check·mc.probe 를 a56_hard_class.sh 가 부른다).
수정 43(«어려운 교실», 2묶음 합쳐 한 번 준비, k4 거르기, EPS .1)은 /hdd_data/seungpil/tmp/a56_hard_class.*.pre44 판이 만들었다.
sample: DeepScaleR 에서 학습·평가·최종 시험·수정 43 문제와 앞 80자 겹침 없고 여러 문항 규칙(`multi`)에 안 걸리는 9,600문제
        (고정 씨앗) → 800문제 묶음 12개 hc2_src_{h}.parquet.
mid h:  묶음 h 의 base k4 롤아웃 중 4번 중 1–3번 정답 문제만 selected → hc2_mid_{h}.{parquet,texts.jsonl}.
weights: 12묶음 k8 거르기 + 수정 43 k4 거르기를 합쳐 앞부분별 p̂(이어쓰기 최종 정답률) → 여러 문항(문제 규칙 ∪ 성공 이어쓰기 첫머리
        «다음 문항» 표지) 제외 → 무게 ∝ EPS + √(p̂(1−p̂)) 부류 안 · 부류 총량 50:50(가지치기 없음; EPS 틀린 부류만 .1 → .02,
        맞은 부류는 .1 유지 — 정보 있는 맞은 앞부분이 30개 이상일 때만 .02(수정 45 망침 예시 모으기)) ·
        행마다 성공·실패 거르기 이어쓰기 ≤ 2개씩 `succ_bank`·`fail_bank`(수정 45 외톨이 행 이웃) ·
        프롬프트 + 앞부분 ≤ MAXP. 관문: 정보 있는(0<p̂<1) 깨끗한 틀린 첫 답 ≥ GATE 일 때만 OUT 을 쓴다(아니면 요약만, 종료 3)."""
import json, math, random, re, sys
from pathlib import Path
import os as _os
_ROOT = _os.environ.get("ROOT", "/hdd_data/seungpil")
_REPO = _os.environ.get("REPO") or str(__import__("pathlib").Path(__file__).resolve().parents[2])   # scripts/remote 복사본
_HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)
from a53_hard_dev import FINAL, SCREEN, key  # noqa: E402

D = _ROOT + "/scratch/"
NCH, CH, MAXP, GATE = 12, 800, 6000, 100
EPS = {True: 0.02, False: 0.1}                                          # 부류(틀린 첫 답?)별 바닥 무게
RIGHT_N, FORK_CAP = 30, 6000            # 정보 있는 맞은 첫 답 ≥ RIGHT_N 이면 맞은 부류 EPS 도 .02(수정 45) · 이웃 글 상한(train_hook)
OUT = D + "data/mc_pfx_commit_hc2_screen_plain_le6000.parquet"
OLD = (D + "data/mc_pfx_commit_hc_plain.parquet", D + "eval/pfx_hc_screen_sel_all.json",
       [D + f"eval/pfx_hc_screen_{h}/texts.jsonl" for h in (0, 1)])        # 수정 43: 선택 pid = 원 앞부분 행, 두 반 sel[h::2]
NEW = [(D + f"data/hc2_pfx_{h}.parquet", D + f"eval/hc2_sel_{h}.json", [D + f"eval/hc2_screen_{h}/texts.jsonl"]) for h in range(NCH)]
_MULTI = [re.compile(x, re.I) for x in (
    r"\(II\)|\(2\)|\(b\)|\(ii\)|Part\s*2|\bpart\s*\(?2|\(Ⅱ\)|②|(^|\s)2\.\s+[A-Z]",                     # 번호 붙은 둘째 문항
    r"\b(find|determine|calculate|compute|what (is|are)|how (many|much))\b[^.?]{0,200}\band\b[^.?]{0,120}(\?|\.)",  # X 와 Y
    r"\?[^?]+\?", r"\b(both|each condition|each type|each of the)\b")]
_NEXT = re.compile(r"\(\s*(II|ii|2|b)\s*\)|part\s*\(?(2|II|b)\b|second part|next part|\*\*\s*\(?(2|II)[.):]", re.I)


def multi(p: str) -> bool:
    """여러 문항 문제(과탐 허용 — 학습에서 빼기만 한다). 수정 43b 손 라벨 38문제 중 36 재현(hdev_all)."""
    return any(r.search(p) for r in _MULTI) or len(re.findall(r"_{3,}|(?:\\_){2,}", p)) >= 2


def sample():
    import pandas as pd
    from datasets import load_dataset
    from mc.grade import grade_answer
    seen = set()
    for f in ("math_train_math_opt.parquet", "math_val_plain.parquet", "mc_pfx_commit_b10_screen_plain_le8704.parquet",
              "hc_src_0.parquet", "hc_src_1.parquet", *(SCREEN[k].split("/")[-1] for k in SCREEN)):
        d = pd.read_parquet(D + "data/" + f)
        seen |= {key(p) for p in (d["problem"] if "problem" in d else [e["problem"] for e in d["extra_info"]])}
    for hid, sp in FINAL:
        seen |= {key(r["problem"]) for r in load_dataset(hid, split=sp)}
    src = pd.read_parquet(D + "data/math_train_math_opt_deepscaler.parquet")
    ok = [i for i, (p, g) in enumerate(zip(src["problem"], src["gold"]))
          if key(p) not in seen and not multi(p) and 0 < len(str(g)) <= 40 and len(str(p)) <= 2500 and grade_answer(str(g), str(g))]
    pick = sorted(random.Random(20261005).sample(ok, NCH * CH))
    for h in range(NCH):
        src.iloc[pick[h::NCH]].reset_index(drop=True).to_parquet(D + f"data/hc2_src_{h}.parquet")
    print(json.dumps({"clean": len(ok), "excluded_keys": len(seen), "picked": NCH * CH}))


def mid(h: str):
    import pandas as pd
    texts = [json.loads(x) for x in open(D + f"data/hc2_pool_{h}.texts.jsonl")]
    right: dict = {}
    for r in texts:
        right.setdefault(r["problem_idx"], []).append(bool(r["correct"]))
    keep = {p for p, v in right.items() if 1 <= sum(v) <= 3}
    for r in texts:
        r["selected"] = r["problem_idx"] in keep
    pd.read_parquet(D + f"data/hc2_pool_{h}.parquet").to_parquet(D + f"data/hc2_mid_{h}.parquet", index=False)
    with open(D + f"data/hc2_mid_{h}.texts.jsonl", "w") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in texts)
    print(json.dumps({"chunk": h, "problems": len(right), "mid": len(keep)}))


def _load(pfx, sel, scr):
    """(앞부분 parquet, 행별 p̂ — 거르기 없으면 None, 행별 «다음 문항» 표지, 행별 이웃 창고(성공·실패 거르기 이어쓰기 ≤ 2개씩,
    수정 45 `pfx_fork` 외톨이 행용)). 두 판 모두 선택 pid = 앞부분 parquet 행."""
    import pandas as pd
    df, res = pd.read_parquet(pfx), {}
    for f in scr:
        for t in map(json.loads, open(f)):
            res.setdefault(int(t["pid"]), []).append((bool(t["final_correct"]), t["text"][:FORK_CAP]))
    assert len(json.loads(Path(sel).read_text())) == len(df), pfx
    ps = [sum(c for c, _ in res[i]) / len(res[i]) if i in res else None for i in range(len(df))]
    nxt = [any(c and _NEXT.search(x[:400]) for c, x in res.get(i, [])) for i in range(len(df))]
    banks = [{k: [x for c, x in res.get(i, []) if c == ok][:2] for k, ok in (("succ_bank", True), ("fail_bank", False))}
             for i in range(len(df))]
    return df, ps, nxt, banks


def weights():
    import pandas as pd
    from transformers import AutoTokenizer
    from mc import context as ctx
    tok = AutoTokenizer.from_pretrained(D + "models/Qwen3-4B-Instruct-2507")
    have = [x for x in NEW if Path(x[2][0]).with_name("summary.json").exists()]      # 실패한 묶음은 빼고 진행(요약에 기록)
    parts = [(_load(*x), "hc2" if x in NEW else "hc") for x in [OLD] + have]
    df = pd.concat([d for (d, *_), _ in parts], ignore_index=True)
    ps, nxt, banks = (sum((x[j] for x, _ in parts), []) for j in (1, 2, 3))
    src = sum(([s] * len(d) for (d, *_), s in parts), [])
    ext = [dict(e) for e in df["extra_info"]]
    L = [len(tok.encode(ctx.turn1_prompt(tok, q, "plain") + e["prefix"], add_special_tokens=False)) for q, e in zip(df["problem"], ext)]
    mp = [multi(q) or n for q, n in zip(df["problem"], nxt)]
    keep = [i for i in range(len(df)) if ps[i] is not None and not mp[i] and L[i] <= MAXP]
    cls = {c: [i for i in keep if bool(ext[i]["first_wrong"]) == c] for c in (True, False)}
    info = lambda ix: sum(0 < ps[i] < 1 for i in ix)  # noqa: E731
    eps = {True: EPS[True], False: EPS[True] if info(cls[False]) >= RIGHT_N else EPS[False]}   # 망침 예시 충분하면 맞은 부류도 .02
    raw = {i: eps[bool(ext[i]["first_wrong"])] + math.sqrt(ps[i] * (1 - ps[i])) for i in keep}
    res = {"n_rows": len(df), "screened": sum(p is not None for p in ps), "multi_rule": sum(map(multi, df["problem"])),
           "next_part_flag": sum(nxt), "dropped_long": sum(x > MAXP for x in L), "n_out": len(keep),
           "wrong": len(cls[True]), "right": len(cls[False]), "informative_wrong": info(cls[True]), "informative_right": info(cls[False]),
           "informative_wrong_by_src": {s: sum(0 < ps[i] < 1 and src[i] == s for i in cls[True]) for s in ("hc", "hc2")},
           "p_fix_mean": sum(ps[i] for i in cls[True]) / max(1, len(cls[True])),
           "wrong_weight_share_informative": sum(raw[i] for i in cls[True] if 0 < ps[i] < 1) / max(1e-9, sum(raw[i] for i in cls[True])),
           "eps_right": eps[False], "right_weight_share_informative": sum(raw[i] for i in cls[False] if 0 < ps[i] < 1) / max(
               1e-9, sum(raw[i] for i in cls[False])), "bank_rows": {k: sum(bool(banks[i][k]) for i in keep) for k in ("succ_bank", "fail_bank")},
           "problems": len({ext[i]["problem_uid"] for i in keep}), "gate": GATE, "out": OUT,
           "chunks_missing": [h for h, x in enumerate(NEW) if x not in have]}
    res["gate_pass"] = res["informative_wrong"] >= GATE
    Path(OUT.replace(".parquet", ".summary.json")).write_text(json.dumps(res, indent=1))
    print(json.dumps(res))
    if not res["gate_pass"]:
        sys.exit(3)
    for c, ix in cls.items():
        tot = sum(raw[i] for i in ix)
        for i in ix:
            ext[i].update(sample_weight=(len(keep) / 2) * raw[i] / tot, p_succ=ps[i], src=src[i], **banks[i])
    out = df.iloc[keep].copy()
    out["extra_info"] = [ext[i] for i in keep]
    out.reset_index(drop=True).to_parquet(OUT, index=False)


def loopcheck(lineage: str):
    """수정 47 중단 규칙: 잘리지 않은 행의 반복(같은 답 3번째 박스) 몫 ≥ .5 또는 잘림 ≥ .10 이 11–15·16–20 두 구간 연속이면
    종료 1(hdev 32k 평가 생략 — a56.sh 가 probe 만 남김)."""
    from mc.train_hook import loop_char
    rs = [json.loads(x) for x in open(D + f"checkpoints/{lineage}/pfx_rollouts.jsonl")]
    bad = []
    for a, b in ((11, 15), (16, 20)):
        s = [r for r in rs if a <= r["step"] <= b]
        nt = [r for r in s if not r["trunc"]]
        lp, tr = sum(loop_char(r["text"]) is not None for r in nt) / max(1, len(nt)), 1 - len(nt) / max(1, len(s))
        print(json.dumps({"steps": f"{a}-{b}", "loop": round(lp, 3), "trunc": round(tr, 3)}))
        bad.append(lp >= 0.5 or tr >= 0.10)
    sys.exit(1 if all(bad) else 0)


def fbgate(eval_dir: str):
    """수정 48 첫 박스 관문: L5 800×8(12k) 첫 박스 정확도 − base(`mathL5_base_b12k_v4`) 문제 짝 부트스트랩 하한 > −3pp 면 종료 0,
    아니면 1(«첫 풀이 번짐» — a56.sh 가 hdev 32k 를 생략)."""
    import contextlib, io
    from mc.grade import boxed_spans, grade_answer
    def per(d):
        pp: dict = {}
        for r in map(json.loads, open(D + f"eval/{d}/texts.jsonl")):
            sp = boxed_spans(r["text"])
            with contextlib.redirect_stdout(io.StringIO()):
                ok = bool(sp) and bool(grade_answer(sp[0][0], str(r["gold"])))
            pp.setdefault(str(r.get("group_id") or r.get("problem_id")), []).append(ok)
        return {k: sum(v) / len(v) for k, v in pp.items()}
    a, b = per(eval_dir), per("mathL5_base_b12k_v4")
    ks = sorted(set(a) & set(b))
    d = [a[k] - b[k] for k in ks]
    rng = random.Random(0)
    bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(2000))
    res = {"eval": eval_dir, "first_box_diff_pp": round(100 * sum(d) / len(d), 2), "lo": round(100 * bs[50], 2), "hi": round(100 * bs[1949], 2)}
    print(json.dumps(res))
    sys.exit(0 if res["lo"] > -3.0 else 1)


if __name__ == "__main__":
    {"sample": sample, "weights": weights, "mid": lambda: mid(sys.argv[2]), "loopcheck": lambda: loopcheck(sys.argv[2]),
     "fbgate": lambda: fbgate(sys.argv[2])}[sys.argv[1]]()
