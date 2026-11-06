"""SPONT_PFX(0923) — 앞부분 재시작 수정: 빌더(분리·자름·LOO 라벨·균형) · 보상(새 박스 없으면 앞부분 답) ·
계기(`mc.probe.score` 와 같은 자) · 프롬프트(verl 에이전트 루프가 앞부분을 prompt_ids 에 붙인다) · 가드."""
from __future__ import annotations

import asyncio
import math
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

from mc import pool as P
from mc.context import VARIANTS
from mc import train_hook as H
from mc.probe import score

TOK_PATH = Path("/hdd_data/seungpil/scratch/models/Qwen3-4B-Instruct-2507")
PLEN = 2


class _Tok:
    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(int(i)) for i in ids)

    def encode(self, s, add_special_tokens=False):
        return [ord(c) for c in s]


def _texts():
    """(문제, 롤아웃) 합성 — p0: 최종 [3,7,7,7] · p1: 전부 다른 답(라벨 None) · p2: 비선별 · 무박스 1."""
    def row(p, uid, text, sel=True):
        sp = P.boxed_spans(text)
        return {"problem_idx": p, "uid": uid, "problem": f"Q{p}", "gold": "7", "text": text,
                "first_answer": sp[0][0] if sp else "", "final_answer": sp[-1][0] if sp else "",
                "selected": sel}
    return [row(0, "u0", r"a \boxed{3} ok"), row(0, "u0", r"b \boxed{5} hmm \boxed{7}"),
            row(0, "u0", r"c \boxed{7} chk"), row(0, "u0", r"d \boxed{7}"),
            row(1, "u1", r"\boxed{1}"), row(1, "u1", r"\boxed{2}"), row(1, "u1", r"\boxed{3}"),
            row(1, "u1", "no box"),
            row(2, "u2", r"\boxed{9}", False), row(2, "u2", r"\boxed{9}", False)]


def test_builder_cut_loo_label_and_balance():
    recs, s = P.pfx_records(_texts(), seed=0)
    assert {r["problem_idx"] for r in recs} == {0}                   # 비선별·라벨 None 문제 없음
    assert s["n_tied_problems"] == 1 and s["n_nolabel"] == 0          # p1 전부 다른 답 → 문제째 뺀다
    assert s["n_wrong"] == 2 and s["n_right_pool"] == 2
    by = {r["rollout"]: r for r in recs}
    assert by[1]["prefix"] == r"b \boxed{5}"                         # 첫 박스 닫는 괄호까지
    assert by[1]["first_wrong"] and by[0]["first_wrong"]             # LOO: 나머지 [7,7,7]·[3,7,7] → 7
    assert all(r["label"] == "7" for r in recs) and s["n_right"] == 2 and len(recs) == 4
    many = _texts()[:4] + [dict(_texts()[2], text=r"e \boxed{7}")] * 3   # 정답 풀이 오답보다 많다
    recs2, s2 = P.pfx_records(many, seed=1)
    assert s2["n_right"] == s2["n_wrong"] == sum(r["first_wrong"] for r in recs2)
    tie = [dict(_texts()[0], text=t, final_answer=f) for t, f in
           ((r"a \boxed{39}", "39"), (r"b \boxed{29}", "29"), (r"c \boxed{39}", "39"), (r"d \boxed{29}", "29"))]
    assert P.pfx_records(tie)[0] == []                               # K 동률 → LOO 가 반대 답을 줘서 뺀다
    mc = [dict(_texts()[0], text=t, first_answer=a, final_answer=f) for t, a, f in
          ((r"a \boxed{1/2}", "1/2", r"\text{B}"), (r"b \boxed{B}", "B", "B"), (r"c \boxed{B}", "B", "B"))]
    recs3, s3 = P.pfx_records(mc)
    assert recs3 == [] and s3["n_choice_problems"] == 1              # 값 ↔ 선택지 글자 표기 차 = 가짜 «고침»


class _OffTok:     # 오프셋만 흉내 — `}\n` 을 한 토큰으로 낸다
    def __call__(self, text, **_):
        cuts = [0] + [i + 2 for i in range(len(text) - 1) if text[i:i + 2] == "}\n"] + [len(text)]
        return {"offset_mapping": [(a, b) for a, b in zip(cuts, cuts[1:]) if a < b]}


def test_first_box_cut_moves_to_token_end():
    t = "x \\boxed{5}\nmore \\boxed{7}"
    assert P.first_box_cut(t) == t.index("}") + 1                    # 문자 자리
    assert P.first_box_cut(t, _OffTok()) == t.index("}") + 2            # `}\n` 토큰 끝(정책이 서는 경계)


def _build(tmp_path, probe_uids=(), val_problems=("V",)):
    base = tmp_path / "pool"
    Path(str(base) + ".texts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in _texts()))
    pd.DataFrame([{"problem": f"Q{p}", "gold": "7", "prompt": [{"role": "system", "content": VARIANTS["plain"]}],
                   "extra_info": {"level": "Level 5"}} for p in range(3)]).to_parquet(
        str(base) + ".parquet")
    (tmp_path / "sel.json").write_text(json.dumps([{"uid": u} for u in probe_uids]))
    pd.DataFrame({"problem": list(val_problems)}).to_parquet(tmp_path / "val.parquet")
    out = tmp_path / "pfx.parquet"
    P.main(["--pfx_from", str(base) + ".texts.jsonl", "--probe_selection", str(tmp_path / "sel.json"),
            "--val", str(tmp_path / "val.parquet"), "--out", str(out), "--model_path", str(TOK_PATH)])
    return out


def test_builder_disjointness_and_schema(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMPT_VARIANT", "plain")
    out = _build(tmp_path, probe_uids=("u2",))                       # 비선별 문제만 탐침에 → 통과
    df = pd.read_parquet(out)
    e = df["extra_info"][0]
    assert len(df) == 4 and e["level"] == "Level 5" and e["prefix"] == r"a \boxed{3}"
    assert set(e) >= {"pfx_id", "problem_uid", "problem_idx", "label", "first_answer", "first_wrong"}
    assert json.loads(out.with_suffix(".summary.json").read_text())["n_rows"] == 4
    for kw in ({"probe_uids": ("u0",)}, {"val_problems": ("Q0",)}):     # 겹치면 즉사
        with pytest.raises(SystemExit):
            _build(tmp_path, **kw)


def test_pfx_screen_keeps_only_mixed_prefixes(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMPT_VARIANT", "plain")
    src = _build(tmp_path)                                            # 4행(오답 2·정답 2)
    sel = tmp_path / "sel_screen.json"
    P.main(["--pfx_screen", str(src), "--out", str(sel)])
    s = json.loads(sel.read_text())
    e0 = pd.read_parquet(src)["extra_info"][0]
    assert len(s) == 4 and s[0]["gold"] == e0["label"] and s[0]["first_correct"] == (not e0["first_wrong"])
    texts = tmp_path / "texts.jsonl"                                 # pid 0: 섞임 · 1: 전부 1 · 2: 전부 0 · 3: 섞임
    rs = [(0, 1), (0, 0), (1, 1), (1, 1), (2, 0), (2, 0), (3, 0), (3, 1)]
    texts.write_text("".join(json.dumps({"pid": i, "final_correct": bool(c)}) + "\n" for i, c in rs))
    out = tmp_path / "keep.parquet"
    P.main(["--pfx_screen", str(src), "--screen_texts", str(texts), "--out", str(out)])
    kept = pd.read_parquet(out)
    assert len(kept) == 2 and list(kept["extra_info"].map(lambda e: e["prefix"])) == [
        pd.read_parquet(src)["extra_info"][i]["prefix"] for i in (0, 3)]


# ── 보상·계기 ────────────────────────────────────────────────────────────────
#: (앞부분, 라벨, 이어 쓰기) — 한 앞부분 K=2 씩, gold 는 전부 "7".
CASES = [(r"x \boxed{3}", "7", " so the answer stays."),               # 오답·새 박스 없음 → 3 → r 0
         (r"x \boxed{3}", "7", r" wait redo \boxed{7}"),                # 고침 → r 1
         (r"y \boxed{7}", "7", " done."),                               # 정답 유지 → r 1
         (r"y \boxed{7}", "7", r" hmm \boxed{4}"),                      # 깨뜨림 → r 0
         (r"z \boxed{5}", "5", " fine."),                               # 라벨(다수결) 5 ≠ gold 7
         (r"z \boxed{5}", "5", r" check \boxed{5}")]


class _PfxBatch:
    def __init__(self, cases=CASES):
        conts = [c for *_, c in cases]
        w = max(len(t) for t in conts)
        self.batch = {"prompts": torch.zeros(len(conts), PLEN, dtype=torch.long),
                      "responses": torch.as_tensor([[ord(c) for c in t] + [0] * (w - len(t))
                                                    for t in conts], dtype=torch.long),
                      "attention_mask": torch.as_tensor([[1] * (PLEN + len(t)) + [0] * (w - len(t))
                                                         for t in conts], dtype=torch.long)}
        ei = [{"prefix": p, "label": lab, "first_wrong": not P.grade_answer(P.boxed_spans(p)[0][0], lab),
               "problem_uid": f"q{i // 2}", "pfx_id": f"q{i // 2}:0", "problem_idx": i // 2,
               "first_answer": P.boxed_spans(p)[0][0]} for i, (p, lab, _) in enumerate(cases)]
        self.non_tensor_batch = {"uid": [f"g{i // 2}" for i in range(len(cases))],
                                 "gold": ["7"] * len(cases), "extra_info": ei}

    def __len__(self):
        return len(self.batch["responses"])


@pytest.fixture
def pfx_env(monkeypatch, tmp_path):
    for k, v in {"OUTCOME_MODE": "group", "LABEL": "majority", "MC_CKPT_DIR": str(tmp_path)}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("MC_DUMP_ADV", raising=False)
    H.HISTORY.clear()
    yield tmp_path
    H.HISTORY.clear()


def test_reward_final_is_prefix_answer_without_new_box_and_grpo(pfx_env):
    pytest.importorskip("verl")
    import verl.trainer.ppo.ray_trainer as rt

    from mc import trainer as T
    cfg = type("C", (), {"algorithm": type("A", (), {"math_arm": "SPONT_PFX"})()})()
    T.CTX["trainer"] = type("Tr", (), {"config": cfg, "global_steps": 1})()
    T.CTX["tokenizer"] = _Tok()
    d = _PfxBatch()
    T.build_advantage_hook(rt.compute_advantage)(d, adv_estimator="grpo",
                                                  norm_adv_by_std_in_grpo=True, config=None)
    tlr, adv = d.batch["token_level_rewards"], d.batch["advantages"]
    assert tlr.sum(-1).tolist() == [0.0, 1.0, 1.0, 0.0, 1.0, 1.0]     # 새 박스 없음 = 앞부분 답
    L = [len(c) for *_, c in CASES]
    assert all(tlr[i, L[i] - 1] == tlr[i].sum() for i in range(6))     # 마지막 유효 토큰에만
    assert adv[1, 0] > 0 > adv[0, 0] and adv[2, 0] > 0 > adv[3, 0]      # 같은 앞부분 안 GRPO
    assert adv[4].abs().max() == 0 and adv[5].abs().max() == 0          # r 상수 그룹 → 0
    dump = [json.loads(x) for x in (pfx_env / "pfx_rollouts.jsonl").read_text().splitlines()]
    assert [x["r"] for x in dump] == [0.0, 1.0, 1.0, 0.0, 1.0, 1.0] and dump[4]["gold_correct"] is False
    assert dump[1]["uid"] == "q0:0" and dump[1]["text"] == CASES[1][2] and dump[0]["step"] == 1


def test_telemetry_matches_probe_score(pfx_env):
    d = _PfxBatch()
    arm = type("Tr", (), {"config": type("C", (), {"algorithm": type("A", (), {
        "math_arm": "SPONT_PFX"})()})()})()
    _, credit, tel = H.token_rewards(d, _Tok(), arm, 3)
    assert credit == {} and tel["v5_mode"] == "pfx"
    assert tel["pfx_wrong_rows"] == 2 and tel["pfx_right_rows"] == 4
    assert tel["informative_groups"] == pytest.approx(2 / 3)
    s = [score(p, c, lab) for p, lab, c in CASES]
    assert tel["pfx_reopen_wrong"] == 0.5 and tel["pfx_fix_wrong"] == s[1]["final_correct"] / 2
    assert tel["pfx_revise_wrong"] == 0.5 and tel["pfx_break_right"] == pytest.approx(
        (0.5 + 0.0) / 2)                                               # 문제별 평균의 평균(탐침과 같다)
    gs = [score(p, c, "7") for p, _, c in CASES]                        # gold_: 문제 q2 는 첫 답 오답
    assert tel["gold_pfx_fix_wrong"] == pytest.approx(
        (gs[1]["final_correct"] / 2 + (gs[4]["final_correct"] + gs[5]["final_correct"]) / 2) / 2)
    assert tel["pfx_tokens"] == pytest.approx(sum(
        sum(len(c) for *_, c in CASES[2 * q:2 * q + 2]) / 2 for q in range(3)) / 3)


def test_pfx_requires_group_and_majority(pfx_env, monkeypatch):
    monkeypatch.setenv("OUTCOME_MODE", "segment")
    arm = type("Tr", (), {"config": type("C", (), {"algorithm": type("A", (), {
        "math_arm": "SPONT_PFX"})()})()})()
    with pytest.raises(ValueError):
        H.token_rewards(_PfxBatch(), _Tok(), arm, 1)


# ── 중단 가드 ────────────────────────────────────────────────────────────────
def test_stop_guard_break_right_only(tmp_path, monkeypatch):
    crash = [{"gold_r1_acc": 0.9 - 0.1 * i, "destroy": 0.1 * i, "pfx_break_right": 0.02}
             for i in range(12)]
    assert H.stop_reason(crash, "pfx") is None                          # 첫 답 규칙 ①~⑤ 면제
    rise = [{"pfx_break_right": v} for v in (0.01, 0.01, 0.01, 0.08, 0.09, 0.10)]
    assert "pfx_break_right" in H.stop_reason(rise, "pfx")              # 기준 .01 → 하한 .05 초과
    assert H.stop_reason(rise[:2], "pfx") is None                       # 3스텝 전엔 판정 없음
    assert H.stop_reason(rise + [{"pfx_break_right": float("nan")}], "pfx")   # NaN 스텝은 건너뛴다
    hi = [{"pfx_break_right": 0.0, "gold_pfx_break_right": v} for v in (0.12, 0.13, 0.12, 0.14, 0.13, 0.12)]
    assert H.stop_reason(hi, "pfx") is None                             # base 부터 높은 gold 파괴 = 헛발동 없음
    dbl = hi[:3] + [{"pfx_break_right": 0.0, "gold_pfx_break_right": v} for v in (0.30, 0.28, 0.31)]
    assert "gold_pfx_break_right" in H.stop_reason(dbl, "pfx")          # 기준선 2배 초과 → 멈춤
    monkeypatch.setenv("MC_CKPT_DIR", str(tmp_path))
    assert H.stop_reason(hi[:3], "pfx") is None and (tmp_path / "FIRST_REF_PFX.json").is_file()
    assert "gold_pfx_break_right" in H.stop_reason(dbl[3:], "pfx")      # resume: 기준선은 파일에서
    art = [{"pfx_break_right": v, "pfx_break_right_live": 0.0} for v in (0.0, 0.0, 0.0, 0.12, 0.13, 0.12)]
    assert H.stop_reason(art, "pfx") is None                            # 수정 36: 잘린 행 «망침» 만 오르면 헛울리지 않음
    assert "pfx_break_right_live" in H.stop_reason(art[:3] + [{"pfx_break_right_live": 0.1}] * 3, "pfx")


# ── 프롬프트: verl 에이전트 루프가 앞부분을 prompt_ids 에 붙인다 ─────────────────
@pytest.mark.skipif(not TOK_PATH.exists(), reason="토크나이저 없음")
def test_prompt_is_template_plus_prefix(tmp_path, monkeypatch):
    pytest.importorskip("verl")
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from verl.utils.dataset.rl_dataset import RLHFDataset

    from mc import context as ctx
    from mc import trainer as T
    monkeypatch.setenv("PROMPT_VARIANT", "plain")
    tok = AutoTokenizer.from_pretrained(str(TOK_PATH))
    pre = "Let me compute. 2+2 = 4, so\n\\[\n\\boxed{4}"
    pq = tmp_path / "one.parquet"
    pd.DataFrame([{"data_source": "math_meta", "problem": "What is 2+2?", "gold": "4",
                   "prompt": ctx.build_math_prompt("What is 2+2?", "plain"),
                   "extra_info": {"prefix": pre, "label": "4"}}]).to_parquet(pq)
    item = RLHFDataset(str(pq), tok, OmegaConf.create({
        "max_prompt_length": 512, "filter_overlong_prompts": False, "cache_dir": str(tmp_path),
        "apply_chat_template_kwargs": {"enable_thinking": False}}))[0]
    cls = T.prefix_agent_loop_cls()
    agent = cls.__new__(cls)                     # 서버·설정 없이 프롬프트 조립만(진짜 verl 메서드)
    agent.tokenizer, agent.processor, agent.prompt_length = tok, None, 512
    agent.apply_chat_template_kwargs = {"enable_thinking": False}
    agent.rollout_config = type("R", (), {"prompt_length": 512})()
    agent.mc_prefix = str(item["extra_info"]["prefix"])

    async def build():
        agent.loop = asyncio.get_running_loop()
        return await agent.apply_chat_template(list(item["raw_prompt"]))
    ids = asyncio.run(build())
    text = tok.decode(ids)
    assert text.endswith(pre) and text.count("<|im_start|>assistant") == 1
    assert text == ctx.turn1_prompt(tok, "What is 2+2?", "plain") + pre      # 탐침과 같은 자리
    assert T.check_prompt_lengths(tok, str(pq), 512)["max"] == len(ids)


# ── R2(수정 27d 체크 비용)·CH-Fork(수정 28) 공용 배치 ─────────────────────────────
FILL = " z" * 200                                                    # 문자 토크나이저 → 400 토큰
_TR = H.token_rewards                                                # 원본(아래 가로채기가 겹치지 않게)
CASES_U = [(r"x \boxed{3}", "7", FILL + r" \boxed{7}"),              # 틀린 첫 답 → 고침(길게 되짚음)
           (r"x \boxed{3}", "7", FILL + r" \boxed{9}"),              # 틀린 첫 답 → 다른 오답(길게 되짚음)
           (r"y \boxed{7}", "7", FILL + r" \boxed{4}"),              # 맞은 첫 답 → 깨짐(길게 되짚음)
           (r"y \boxed{7}", "7", FILL + " done."),                   # 맞은 첫 답 유지(길게 되짚음 — 멈춤·CH 대상)
           (r"w \boxed{3}", "7", FILL + " so it stays."),            # 틀린 첫 답 유지(길게 되짚음)
           (r"w \boxed{3}", "7", " ok.")]                            # 틀린 첫 답 유지(짧음)


_LAST: dict = {}


def _pfx_run(monkeypatch, weight=None, p=None, **env):
    from mc import trainer as T
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import verl.trainer.ppo.ray_trainer as rt
    cfg = type("C", (), {"algorithm": type("A", (), {"math_arm": "SPONT_PFX"})()})()
    T.CTX["trainer"], T.CTX["tokenizer"] = type("Tr", (), {"config": cfg, "global_steps": 1})(), _Tok()
    d = _LAST["d"] = _PfxBatch(CASES_U)
    if weight is not None:
        for e, w in zip(d.non_tensor_batch["extra_info"], weight):
            e["weight"] = w
    for e, x in zip(d.non_tensor_batch["extra_info"], p or []):
        e["p_wrong"] = x
    box = {}
    monkeypatch.setattr(H, "token_rewards", lambda *a, **k: box.setdefault("o", _TR(*a, **k)))
    T.build_advantage_hook(rt.compute_advantage)(d, adv_estimator="grpo", norm_adv_by_std_in_grpo=True, config=None)
    return d.batch["advantages"].clone(), box["o"]


def test_pfx_row_weight_does_not_scale_loss(pfx_env, monkeypatch):
    """`extra_info.weight`(PFX_WEIGHT_KEY 표집 무게)는 표집기에서만 쓰인다 — 손실 배율은 없다(수정 25)."""
    pytest.importorskip("verl")
    a1, _ = _pfx_run(monkeypatch)
    a2, (_, _, t2) = _pfx_run(monkeypatch, weight=[2.0, 2.0, 0.5, 0.5, 1.0, 1.0])
    assert torch.equal(a2, a1) and "row_weight" not in t2
    assert t2["pfx_informative_n"] == 2 and t2["pfx_groups"] == 3 and t2["pfx_fix_rows"] == 1
    b, (_, cb, tb) = _pfx_run(monkeypatch)                            # PFX-B: 체크 비용·포크 미설정 → 크레딧 없음
    assert cb == {} and "rows" not in tb and b[0, :400].abs().sum() > 0


def test_pfx_sampler_draws_by_weight_seeded(monkeypatch):
    import datasets

    from mc import trainer as T
    ei = [{"prefix": "p", "weight": w} for w in (0.1, 0.1, 1.0, 3.0)]
    ds = type("D", (), {"dataframe": datasets.Dataset.from_dict({"extra_info": ei})})()
    cfg = {"seed": 7}
    monkeypatch.delenv("PFX_WEIGHT_KEY", raising=False)
    sm = T.pfx_sampler(cfg, ds)
    draws = [i for _ in range(500) for i in sm]                      # 에폭마다 생성기가 이어진다
    assert len(list(T.pfx_sampler(cfg, ds))) == 4                    # 에폭 길이 = 행 수(스텝 수 불변)
    share = [draws.count(i) / len(draws) for i in range(4)]
    assert share == pytest.approx([0.1 / 4.2, 0.1 / 4.2, 1 / 4.2, 3 / 4.2], abs=0.02)
    assert min(share) > 0                                            # 가지치기 없음
    assert list(T.pfx_sampler(cfg, ds)) == list(T.pfx_sampler(cfg, ds))   # 고정 시드
    monkeypatch.setenv("PFX_WEIGHT_KEY", "w2")
    assert T.pfx_sampler(cfg, ds) is None                            # 무게 없는 parquet → verl 기본 표집기
    bad = type("D", (), {"dataframe": datasets.Dataset.from_dict({"extra_info": [{"w2": -1.0}, {"w2": 1.0}]})})()
    with pytest.raises(ValueError):
        T.pfx_sampler(cfg, bad)
    assert "PFX_WEIGHT_KEY" in T.WORKER_ENV_KEYS


def test_pfx_a0_is_committed_last_box_not_intermediate(pfx_env, monkeypatch):
    """0925 수리: 앞부분에 중간 박스가 있으면(box_k>0) 첫 답 = **마지막** 박스(약속) — 첫 박스로 계급을 매기면 약속한 오답이
    «맞은 첫 답을 깼다»로 잡혀 파괴 가드가 헛발동했다(B 50스텝·L 스텝 20)."""
    pytest.importorskip("verl")
    cases = [(r"mid \boxed{7} so final \boxed{3}", "7", " kept."),       # 약속 = 3(오답), 중간 박스 7 = 정답
             (r"mid \boxed{7} so final \boxed{3}", "7", r" oops \boxed{7}"),
             (r"mid \boxed{2} so final \boxed{7}", "7", " kept."),       # 약속 = 7(정답), 중간 박스 2
             (r"mid \boxed{2} so final \boxed{7}", "7", r" oops \boxed{2}")]
    monkeypatch.setattr(sys.modules[__name__], "CASES_U", cases)
    _, (_, _, tel) = _pfx_run(monkeypatch)
    assert tel["pfx_wrong_rows"] == 2.0 and tel["pfx_fix_rows"] == 1.0 and tel["pfx_break_right"] == pytest.approx(0.5)


def _fork_setup(monkeypatch, cases, uids):
    monkeypatch.setattr(sys.modules[__name__], "CASES_U", cases)
    real_init = _PfxBatch.__init__

    def init(self, cases=CASES):
        real_init(self, cases)
        self.non_tensor_batch["uid"] = uids
        for e in self.non_tensor_batch["extra_info"]:
            e["problem"] = "Q"
    monkeypatch.setattr(_PfxBatch, "__init__", init)
    from mc import context as ctx
    monkeypatch.setattr(ctx, "turn1_prompt", lambda tok, prob, v: "P:" + prob)


def _heads_str(h):
    return "".join(chr(c) for c in h if c < 0x110000)


def test_pfx_truncated_continuation_masked_under_trunc_mask(pfx_env, monkeypatch):
    """수정 32/32c(DAPO overlong filtering): 잘린 행 = adv 0(칭찬도 벌도 없음) · 고유 uid 라 GRPO 묶음 평균·std 밖 ·
    손실 마스크 0(token-mean 분모·KL 밖) · r 은 실제 결과 기록."""
    pytest.importorskip("verl")
    for k in ("PFX_REP", "PFX_FORK"):
        monkeypatch.delenv(k, raising=False)
    cases = [(r"y \boxed{7}", "7", FILL + FILL),                          # 잘림(가장 긴 행)
             (r"y \boxed{7}", "7", FILL + " done."),                   # 맞은 첫 답 유지(성공)
             (r"y \boxed{7}", "7", FILL + r" \boxed{4}")]              # 깨짐(실패)
    _fork_setup(monkeypatch, cases, ["g0"] * 3)
    adv, (tlr, _, tel) = _pfx_run(monkeypatch, PFX_TRUNC="mask")
    assert tel["pfx_cut_rows"] == 1.0 and max(tlr[0]) == 1.0 and tel["pfx_drop"] == [0]  # 실제 결과(첫 답 유지 = 맞음)
    assert tel["pfx_break_right_live"] == pytest.approx(0.5)                                   # 가드 = 잘리지 않은 행만(수정 36)
    assert float(adv[0].abs().sum()) == 0.0 and float(adv[1, 0]) == pytest.approx(-float(adv[2, 0])) == pytest.approx(.7071, rel=1e-3)
    d = _LAST["d"]                                                                # 두 행 묶음과 같은 adv — 잘린 행은 std 밖
    m = d.batch["response_mask"]                                                  # 첫 토큰만(전부 잘린 미니배치 0/0 방지)
    assert m[0, 1:].sum() == 0 and m[0, 0] == 1 and m[1:].sum() > 0
    assert int(d.batch["attention_mask"][0].sum()) == PLEN + len(cases[0][2])     # 사본에 썼다 — attention_mask 불변
    assert str(d.non_tensor_batch["uid"][0]).endswith("#cut0") and tel["pfx_informative_n"] == 1.0


def test_pfx_rep_penalizes_only_after_third_same_box_even_on_cut_rows(pfx_env, monkeypatch):
    """수정 43 PFX_REP: 같은 답 3번째 박스 뒤 토큰에만 −c(끝낸 행 · 잘린 행 모두) · 그 앞(되짚기·고치는 계산)은 결과 adv 그대로 ·
    mask 로 빠진 잘린 행은 결과 adv 0 이지만 반복 구간은 손실에 남아 벌만 받는다 · 총량 ≤ REP_CAP 몫."""
    pytest.importorskip("verl")
    for k in ("PFX_FORK", "PFX_BREAK_W"):
        monkeypatch.delenv(k, raising=False)
    rep3 = r" \boxed{7} a \boxed{7} b \boxed{7}"
    cases = [(r"x \boxed{3}", "7", FILL + rep3 + " tail tail tail."),     # 고친 뒤 반복 확인(성공) → 꼬리만 벌
             (r"x \boxed{3}", "7", FILL + rep3 + FILL + FILL),            # 반복하다 잘림(가장 긴 행) → 빠지지만 꼬리 벌
             (r"x \boxed{3}", "7", FILL + r" \boxed{9}")]                # 실패, 반복 없음 → 크레딧 없음
    _fork_setup(monkeypatch, cases, ["g0"] * 3)
    base, _ = _pfx_run(monkeypatch, PFX_TRUNC="mask")
    adv, (_, cr, tel) = _pfx_run(monkeypatch, PFX_TRUNC="mask", PFX_REP="0.5")
    j0 = [H.loop_char(c) for *_, c in cases]
    assert j0[2] is None and sorted(cr) == [0, 1] and cr[0][0] == j0[0] and cr[1][0] == j0[1]
    assert tel["pfx_drop"] == [1] and tel["pfx_keep_from"] == {1: j0[1]} and tel["credit_cap"] == H.REP_CAP
    assert torch.allclose(adv[0, :j0[0]], base[0, :j0[0]]) and (adv[0, j0[0]:len(cases[0][2])] < base[0, j0[0]:len(cases[0][2])]).all()
    assert adv[1, :j0[1]].abs().sum() == 0 and (adv[1, j0[1]:] < 0).all()        # 빠진 잘린 행: 반복 구간만 벌
    m = _LAST["d"].batch["response_mask"]
    assert m[1, 1:j0[1]].sum() == 0 and m[1, j0[1]:].all() and torch.allclose(adv[2], base[2])
    assert tel["mass_share"] <= H.REP_CAP + 1e-6


def test_pfx_adv_cap_bounds_spikes_and_leaves_small_rows(pfx_env, monkeypatch):
    """수정 48 PFX_ADV_CAP: CH·BREAK_W 곱 뒤 말 단위 adv 를 ±c 로 자름 — 상한 안 행은 그대로, 계기 adv_absmax."""
    pytest.importorskip("verl")
    for k in ("PFX_FORK", "PFX_REP", "PFX_TRUNC"):
        monkeypatch.delenv(k, raising=False)
    base, _ = _pfx_run(monkeypatch, PFX_BREAK_W="2")
    c = float(base.abs().max()) * 0.6
    adv, (_, _, tel) = _pfx_run(monkeypatch, PFX_BREAK_W="2", PFX_ADV_CAP=str(c))
    small = (base.abs() <= c).all(dim=1)
    assert float(adv.abs().max()) <= c + 1e-6 and tel["adv_absmax"] == pytest.approx(float(base.abs().max()), rel=1e-5)
    assert small.any() and torch.allclose(adv[small], base[small]) and 0 < tel["adv_capped_frac"] < 1
    with pytest.raises(ValueError):
        _pfx_run(monkeypatch, PFX_ADV_CAP="0")


def test_pfx_distill_wrong_first_rows_per_token_credit_and_span_end(pfx_env, monkeypatch):
    """수정 51 PFX_DISTILL: 틀린 첫 답 행만 · 선생님 머리 = 문제 + FACT_TMPL(X)(앞부분 없음) · 구간 [16, 마지막 박스 끝·반복 시작) ·
    말당 β·clip(d) (수정 52 — 행 총량 고정 없음) · 결과 adv 에 가산 · ds_share 계기."""
    pytest.importorskip("verl")
    for k in ("PFX_FORK", "PFX_REP", "PFX_BREAK_W", "PFX_TRUNC"):
        monkeypatch.delenv(k, raising=False)
    rep4 = r" \boxed{9} a \boxed{9} b \boxed{9} c \boxed{9} tail"
    cases = [(r"x \boxed{3}", "7", FILL + r" \boxed{7}"), (r"x \boxed{3}", "7", FILL + rep4),
             (r"y \boxed{7}", "7", FILL + " done."), (r"y \boxed{7}", "7", FILL + r" \boxed{4}"),
             (r"w \boxed{3}", "7", " ok."), (r"w \boxed{3}", "7", FILL)]
    _fork_setup(monkeypatch, cases, ["g0", "g0", "g1", "g1", "g2", "g2"])
    real_init = _PfxBatch.__init__

    def init(self, cases=CASES):
        real_init(self, cases)
        self.batch["old_log_probs"] = torch.full(self.batch["responses"].shape, -1.0)
    monkeypatch.setattr(_PfxBatch, "__init__", init)
    base, _ = _pfx_run(monkeypatch)
    heads = []

    def fake(tr, trees, per_token=False):
        heads.extend(_heads_str(h) for h, _ in trees)
        return [[0.5] * bl[0][2] for _, bl in trees]                   # d_t = .5 − (−1) = 1.5
    from mc import trainer as T
    monkeypatch.setattr(T, "ref_tree_score", fake)
    adv, (_, _, tel) = _pfx_run(monkeypatch, PFX_DISTILL="0.25")
    ds = tel["ds_credit"]
    assert sorted(ds) == [0, 1, 5] and tel["ds_rows"] == 3.0                    # 맞은 첫 답 2·3 · 짧은 4 제외
    assert all("previous attempt at this problem answered 3" in h and "x \\boxed" not in h for h in heads)
    for i, (j0, vals) in ds.items():
        assert j0 == H.DS_FRONT and vals == pytest.approx([0.25 * 1.5] * len(vals))        # 말당 β·d
        assert torch.allclose(adv[i, j0:j0 + len(vals)] - base[i, j0:j0 + len(vals)], torch.tensor(vals, dtype=adv.dtype))
    assert H.DS_FRONT + len(ds[1][1]) == H.loop_char(cases[1][2])          # 반복 시작에서 끊김(꼬리 미포함)
    assert torch.allclose(adv[2:4], base[2:4])
    assert 0 < tel["ds_share"] <= tel["mass_share"]
    monkeypatch.setattr(T, "ref_tree_score", lambda tr, trees, per_token=False:
                        [[-1.0 + 0.1 * (t % 7) for t in range(bl[0][2])] for _, bl in trees])   # 말마다 다른 d
    _, (_, _, t0) = _pfx_run(monkeypatch, PFX_DISTILL="0.25")
    _, (_, _, t1) = _pfx_run(monkeypatch, PFX_DISTILL="0.25", PFX_DISTILL_SHUF="1")   # 수정 55 위약
    for i, (j0, vals) in t0["ds_credit"].items():
        sv = t1["ds_credit"][i][1]
        assert t1["ds_credit"][i][0] == j0 and sorted(sv) == pytest.approx(sorted(vals)) and sv != pytest.approx(vals)
    monkeypatch.delenv("PFX_DISTILL_SHUF")
    _, (_, _, t2) = _pfx_run(monkeypatch, PFX_DISTILL="0.25", PFX_DISTILL_SIGN="1")   # 수정 87 결과 부호
    for i, (j0, vals) in t0["ds_credit"].items():   # 행마다 원래 값 또는 0, 남은 값은 한 부호(맞힘 +·틀림 −)
        kept = t2["ds_credit"][i][1]; nz = [v for v in kept if v]
        assert all(k in (0.0, pytest.approx(v)) for k, v in zip(kept, vals)) and len({v > 0 for v in nz}) <= 1
    assert 0 < t2["ds_sign_cut"] < 1
    with pytest.raises(ValueError, match="함께 쓰지 않는다"):
        _pfx_run(monkeypatch, PFX_DISTILL="0.25", PFX_DISTILL_SIGN="1", PFX_DISTILL_SHUF="1")
    monkeypatch.delenv("PFX_DISTILL_SIGN"); monkeypatch.delenv("PFX_DISTILL_SHUF")
    for k in ("PFX_DISTILL_ROWS", "PFX_REP_HARD", "PFX_ALLOC", "PFX_CONFIRM", "PFX_BRAKE", "PFX_KEEP"):   # 지운 손잡이(59b·46·65·65b·53) = 즉사
        with pytest.raises(ValueError, match="지운 손잡이"):
            _pfx_run(monkeypatch, PFX_DISTILL="0.25", **{k: "1"})
        monkeypatch.delenv(k)


LOOP3 = r" \boxed{3} a \boxed{3} b \boxed{3} c tail"


def _teacher_env(monkeypatch, cases, uids):
    """선생님 시험 공용 — old_log_probs −1 · ref_log_prob 0."""
    _fork_setup(monkeypatch, cases, uids)
    real_init = _PfxBatch.__init__

    def init(self, cases=CASES):
        real_init(self, cases)
        self.batch["old_log_probs"] = torch.full(self.batch["responses"].shape, -1.0)
        self.batch["ref_log_prob"] = torch.zeros(self.batch["responses"].shape)
    monkeypatch.setattr(_PfxBatch, "__init__", init)


def test_pfx_production_combo_mask_breakw_rep_distill(pfx_env, monkeypatch):
    """확정 방법(65b) 조합(mask · BREAK_W 2 · REP · DISTILL): 잘린 행 adv 0 · 리셋은 틀린 첫 답 행만 · 리셋 구간은 반복 자리에서 끝나고
    반복 벌은 거기서 시작 · 맞은 첫 답 행(반복 없음) 무변."""
    pytest.importorskip("verl")
    for k in ("PFX_FORK", "PFX_REP", "PFX_BREAK_W"):
        monkeypatch.delenv(k, raising=False)
    cases = [(r"x \boxed{3}", "7", FILL + r" \boxed{7}"), (r"x \boxed{3}", "7", FILL + LOOP3),   # 틀→맞 · 틀→같은 오답 반복
             (r"y \boxed{7}", "7", FILL + " done."), (r"y \boxed{7}", "7", FILL + r" \boxed{4}"),   # 맞→맞 · 맞→틀
             (r"w \boxed{3}", "7", FILL * 3), (r"w \boxed{3}", "7", FILL)]                        # 잘림(가장 긴 행) · 외톨이
    _teacher_env(monkeypatch, cases, ["g0", "g0", "g1", "g1", "g2", "g2"])
    from mc import trainer as T
    monkeypatch.setattr(T, "ref_tree_score", lambda tr, trees, per_token=False:
                        [[0.3 * (t % 5) for t in range(bl[0][2])] for _, bl in trees])
    base, _ = _pfx_run(monkeypatch, PFX_TRUNC="mask", PFX_BREAK_W="2")
    adv, (_, cr, tel) = _pfx_run(monkeypatch, PFX_TRUNC="mask", PFX_BREAK_W="2", PFX_DISTILL="0.25", PFX_REP="0.25")
    ds, j = tel["ds_credit"], H.loop_char(cases[1][2])
    assert tel["pfx_drop"] == [4] and float(adv[4].abs().sum()) == 0.0
    assert sorted(ds) == [0, 1, 5] and sorted(cr) == [1] and ds[1][0] + len(ds[1][1]) == j == cr[1][0]   # 리셋 끝 = 반복 벌 시작
    assert torch.allclose(adv[2:4], base[2:4])                                                          # 맞은 첫 답 행 무변


def test_pfx_ratio_baselines_rlsd_rlrt_sdpo(pfx_env, monkeypatch):
    """수정 70 비교군: Δ = lp_T − ref_log_prob(=.5 − 0) · λ_1 = .5(1 − 1/D) · RLSD 선생님 = 문제 + 정답, 맞은 행 clip(e^Δ, .8, 1.2)=1.2,
    틀린 행 e^{−Δ}=.61→.8 · RLRT 선생님 = 문제 + 맞힌 형제 풀이(자기 제외), 맞은 행만 e^{−Δ}=.61(ε .5 안), 틀린 행 무게 없음 · DISTILL 과 섞으면 즉사."""
    pytest.importorskip("verl")
    for k in ("PFX_FORK", "PFX_REP", "PFX_BREAK_W", "PFX_TRUNC", "PFX_DISTILL"):
        monkeypatch.delenv(k, raising=False)
    cases = [(r"x \boxed{3}", "7", FILL + r" \boxed{7} sibA"), (r"x \boxed{3}", "7", FILL + r" \boxed{7} sibB"),
             (r"x \boxed{3}", "7", FILL + r" \boxed{9}")]
    _teacher_env(monkeypatch, cases, ["g0", "g0", "g0"])
    heads = []
    from mc import trainer as T
    monkeypatch.setattr(T, "ref_tree_score", lambda tr, trees, per_token=False: (
        heads.extend(_heads_str(h) for h, _ in trees) or [[0.5] * bl[0][2] for _, bl in trees]))
    _, (_, _, t1) = _pfx_run(monkeypatch, PFX_FORK="rlsd")
    w = t1["pfx_weights"]
    lam = 0.5 * (1 - 1 / 50)
    assert sorted(w) == [0, 1, 2] and w[0][0] == pytest.approx(1 - lam + lam * 1.2) and w[2][0] == pytest.approx(1 - lam + lam * 0.8)
    assert all("correct final answer to this problem is 7" in h for h in heads)
    heads.clear()
    _, (_, _, t2) = _pfx_run(monkeypatch, PFX_FORK="rlrt")
    w = t2["pfx_weights"]
    lam = 0.5 * (1 - 1 / 30)
    assert sorted(w) == [0, 1] and w[0][0] == pytest.approx(1 - lam + lam * math.exp(-0.5)) and t2["fork_lam"] == pytest.approx(lam)
    assert "correct solution" in heads[0] and "sibB" in heads[0] and "sibA" in heads[1]
    monkeypatch.setenv("PFX_TEACHER", "policy")         # 수정 74: 학생 쪽 = π_old(−1) → Δ = 1.5, e^{−1.5} → 하한 .5
    assert _pfx_run(monkeypatch, PFX_FORK="rlrt")[1][2]["pfx_weights"][0][0] == pytest.approx(1 - lam + lam * 0.5)
    monkeypatch.delenv("PFX_TEACHER")
    heads.clear()                                       # 수정 72 SDPO 식: 같은 형제 풀이 문맥 · 모든 행 · 가산 .5·(.5 − (−1)) = .75
    adv3, (_, _, t3) = _pfx_run(monkeypatch, PFX_FORK="sdpo")
    cr = t3["ds_credit"]
    assert sorted(cr) == [0, 1, 2] and all(j0 == 0 and v[0] == pytest.approx(0.75) for j0, v in cr.values()) and "pfx_weights" not in t3
    assert "sibB" in heads[0] and "sibA" in heads[1] and "sibA" in heads[2] and t3["fork_d_mean"] == pytest.approx(1.5)
    monkeypatch.delenv("PFX_FORK")                      # 가산은 결과 adv 위에 그대로(응답 말 자리 0 부터)
    base, _ = _pfx_run(monkeypatch)
    assert torch.allclose((adv3 - base)[2, :len(cases[2][2])], torch.full((len(cases[2][2]),), 0.75, dtype=adv3.dtype))
    with pytest.raises(ValueError, match="비교군"):
        _pfx_run(monkeypatch, PFX_FORK="rlsd", PFX_DISTILL="0.5")
    assert "OOM" in H.pfx_guard([{"fork_lam": 0.5, "fork_rows": 5.0, "fork_oom": 2.0}])        # 비교군이 조용히 GRPO 가 되지 않게


def test_ref_tree_score_policy_teacher_routes_to_actor(monkeypatch):
    """수정 74: PFX_TEACHER=policy 면 채점이 얼린 ref 워커가 아니라 actor 워커(policy=True)로 간다."""
    pytest.importorskip("ray")
    from mc import trainer as T
    calls = []
    wg = lambda name: type("W", (), {"mc_tree_score": lambda self, *a: calls.append((name, a)) or [["s"]]})()  # noqa: E731
    tr = type("Tr", (), {"actor_rollout_wg": wg("actor"), "ref_policy_wg": wg("ref")})()
    monkeypatch.delenv("PFX_TEACHER", raising=False)
    T.ref_tree_score(tr, [], per_token=True)
    monkeypatch.setenv("PFX_TEACHER", "policy")
    assert T.ref_tree_score(tr, [], per_token=True) == ["s"] and calls == [("ref", ([], True)), ("actor", ([], True, True))]


STOP_CASES = [(r"x \boxed{3}", "7", FILL + r" \boxed{7}"), (r"x \boxed{3}", "7", FILL + " no."), (r"y \boxed{7}" + "\n", "7", FILL + r" \boxed{7}"), (r"y \boxed{7}", "7", FILL + r" \boxed{4}")]  # 틀→맞·틀→틀|맞→맞(줄 끝)·맞→틀(줄 끝 아님)


def _stop_env(monkeypatch, lp):
    """끝내기·곱 시험 공용 — old_log_probs −1 · ref_log_prob 0 · 선생님 채점 = lp(말 번호)(머리 문맥은 heads 에 모은다)."""
    for k in ("PFX_FORK", "PFX_REP", "PFX_BREAK_W", "PFX_TRUNC", "PFX_VERDICT", "PFX_VERDICT_MIN", "PFX_ROUTE"):
        monkeypatch.delenv(k, raising=False)
    _teacher_env(monkeypatch, STOP_CASES, ["g0", "g0", "g1", "g1"])
    heads = []
    from mc import trainer as T
    monkeypatch.setattr(T, "ref_tree_score", lambda tr, trees, per_token=False: (
        heads.extend(_heads_str(h) for h, _ in trees) or [[lp(t) for t in range(bl[0][2])] for _, bl in trees]))
    return heads


def test_pfx_blind_ratio_reweights_only_wrong_first_window(pfx_env, monkeypatch):
    """수정 82 PFX_FORK=blind(눈 가린 선생님 × 결과 adv, RLSD 식): 성공 행 w = .5 + .5·1.2 · 실패 행 .5 + .5·.8 · 앞 DS_FRONT 말 1 · 맞은 첫 답 행 없음 · 문맥 = 문제 + FACT(X)."""
    pytest.importorskip("verl")
    heads = _stop_env(monkeypatch, lambda t: 0.5)
    _, (_, _, tel) = _pfx_run(monkeypatch, PFX_FORK="blind")
    w = tel["pfx_weights"]
    assert sorted(w) == [0, 1] and tel["fork_mode"] == "blind" and all("answered 3" in h for h in heads)
    assert w[0][:H.DS_FRONT] == [1.0] * H.DS_FRONT and w[0][H.DS_FRONT] == pytest.approx(1.1) and w[1][H.DS_FRONT] == pytest.approx(0.9)
    with pytest.raises(ValueError, match="섞지"):
        _pfx_run(monkeypatch, PFX_FORK="blind", PFX_DISTILL="0.25")


RT_CASES = [(r"y \boxed{7}", "7", FILL + r" \boxed{7} fine." + "\nThe verification text ends here."),   # 맞→같음: 끝내기 교사
            (r"x \boxed{3}", "7", FILL + r" \boxed{7}$$" + "\n" + FILL + r" so \boxed{7}"),                         # 틀→다름: 고치기 교사 · 눈 가린 나 창이 jY 에서 끊김
            (r"y \boxed{7}", "7", FILL + r" \boxed{4}$$" + "\n" + FILL + r" so \boxed{4}"),                         # 맞→다름: 고치기 교사
            (r"x \boxed{3}", "7", "." + FILL + r" \boxed{3}" + FILL + r" so \boxed{3}"),                   # 틀→같음: 판정 교사 없음 · 눈 가린 나 창 그대로
            (r"y \boxed{7}", "7", FILL + " done.")]                                                         # 둘째 박스 없음


def _rt_env(monkeypatch, calls):
    """판정 시험 공용 — old −1 · 선생님 lp(t) = .1(t mod 7) − .5(말마다 달라 어긋남이 드러난다) · 호출마다 머리 문맥을 calls 에 모은다."""
    for k in ("PFX_FORK", "PFX_REP", "PFX_BREAK_W", "PFX_TRUNC", "PFX_VERDICT", "PFX_VERDICT_MIN", "PFX_ROUTE", "PFX_DISTILL"):
        monkeypatch.delenv(k, raising=False)
    _teacher_env(monkeypatch, RT_CASES, ["g0", "g0", "g1", "g1", "g2"])
    from mc import trainer as T
    monkeypatch.setattr(T, "ref_tree_score", lambda tr, trees, per_token=False: (
        calls.append([_heads_str(h) for h, _ in trees]) or [[0.1 * (t % 7) - 0.5 for t in range(bl[0][2])] for _, bl in trees]))


def test_pfx_verdict_reward_signs_cells_and_window(pfx_env, monkeypatch):
    """수정 83 PFX_VERDICT=v: 맞→같음·틀→다름 +v, 맞→다름·틀→같음·틀→안 엶 −v 를 [0, jY)(둘째 박스까지) 모든 말에 · 맞은 줄 안 엶 무변 · 칸 계수."""
    pytest.importorskip("verl")
    _rt_env(monkeypatch, [])
    base, _ = _pfx_run(monkeypatch)
    adv, (_, _, tel) = _pfx_run(monkeypatch, PFX_VERDICT="0.5")
    jy = [len(FILL) + len(r" \boxed{7}")] * 3 + [1 + len(FILL) + len(r" \boxed{3}")]
    assert sorted(tel["ver_credit"]) == [0, 1, 2, 3] and tel["ver_rows"] == 4.0 and tel["ver_none_right"] == 1.0
    for i, sg in enumerate((0.5, 0.5, -0.5, -0.5)):
        assert tel["ver_credit"][i] == (0, [sg] * jy[i])
        assert torch.allclose(adv[i, :jy[i]] - base[i, :jy[i]], torch.full((jy[i],), sg, dtype=adv.dtype), atol=1e-4)
        assert torch.allclose(adv[i, jy[i]:], base[i, jy[i]:])
    assert torch.allclose(adv[4], base[4])
    assert [tel[f"ver_{c}"] for c in ("same_right", "diff_wrong", "diff_right", "same_wrong")] == [1.0] * 4 and 0 < tel["ver_share"] <= tel["mass_share"]
    assert H.verdict(r"a \boxed{7}", r"x \boxed{7.0} y \boxed{4}")[0] == "same" and H.verdict(r"a \boxed{7}", "none") is None
    assert H.verdict(r"a \boxed{7}", r"x \boxed{7.0} y \boxed{4}", c0=3)[0] == "diff" and H.verdict(r"a \boxed{7}", r"x \boxed{7}", c0=3) is None  # 수정 84 답 다시 적기 건너뜀
    assert H.pfx_verdict({0: ("wrong", "none", -1, 3), 1: ("right", "none", -1, 3)}, 0.5)["ver_credit"] == {0: (0, [-0.5] * 3)}   # 놓침 −, 맞은 줄 안 엶 0
    # 수정 90 PFX_VERDICT_SRC=probe: 부호 = 탐침 p_wrong(정답 부류 무시) · v·2(p−τ)·s — 높은 p 에 다름/낮은 p 에 같음·안 엶이 +, 반대 −
    jn = [len(FILL) + len(r" \boxed{7}")] * 3 + [1 + len(FILL) + len(r" \boxed{3}"), len(FILL + " done.")]   # 행 4 = 안 엶 → 행 전체
    for tau, sgn in (("0.5", (-.4, -.4, .4, .4, -.4)), ("0.9", (0, -.8, 0, .8, 0))):
        adv, (_, _, tel) = _pfx_run(monkeypatch, p=[.9, .1, .9, .1, .9], PFX_VERDICT="0.5", PFX_VERDICT_SRC="probe", PFX_PROBE_TAU=tau)
        cr = tel["ver_credit"]
        assert sorted(cr) == [i for i, x in enumerate(sgn) if x] and tel["ver_probe_mean"] == pytest.approx(.58) and tel["ver_none_right"] == 1.0
        assert all(cr[i] == (0, [pytest.approx(x)] * jn[i]) for i, x in enumerate(sgn) if x)   # 행0 맞→같음(p높)·행1 틀→다름(p낮)·행2 맞→다름·행3 틀→같음(p낮)
    with pytest.raises(ValueError, match="p_wrong"):
        _pfx_run(monkeypatch, PFX_VERDICT="0.5", PFX_VERDICT_SRC="probe")           # 데이터에 p_wrong 없음
    for bad in ({"PFX_VERDICT_SRC": "gold", "PFX_VERDICT": "0.5"}, {"PFX_VERDICT_SRC": "probe"}, {"PFX_PROBE_TAU": "0.5", "PFX_VERDICT": "0.5"}):
        for k in ("PFX_VERDICT", "PFX_VERDICT_SRC", "PFX_PROBE_TAU"):
            monkeypatch.delenv(k, raising=False)
        with pytest.raises(ValueError, match="PFX_VERDICT_SRC"):
            _pfx_run(monkeypatch, p=[.9] * 5, **bad)


def test_pfx_route_teachers_follow_own_verdict(pfx_env, monkeypatch):
    """수정 83 PFX_ROUTE: 맞→같음 = 끝내기 교사(둘째 박스 줄 끝에 쪽지 · 그 뒤 STOP_K 말) · 다름 = 고치기 교사([jY+DS_FRONT, j1), gold 면 정답 쪽지) ·
    틀→같음 = 없음 · 다름 줄의 눈 가린 나 창은 jY 에서 끊겨 겹치지 않음 · 트리는 한 번에 채점 · 잘못된 조합은 즉사."""
    pytest.importorskip("verl")
    calls = []
    _rt_env(monkeypatch, calls)
    base, _ = _pfx_run(monkeypatch)
    _, (_, _, t0) = _pfx_run(monkeypatch, PFX_DISTILL="0.25")
    n = [len(c) for *_, c in RT_CASES]
    assert t0["ds_credit"][1][0] + len(t0["ds_credit"][1][1]) == n[1] and t0["ds_credit"][3][0] + len(t0["ds_credit"][3][1]) == n[3]   # 교사 없으면 마지막 박스까지
    calls.clear()
    adv, (_, _, tel) = _pfx_run(monkeypatch, PFX_DISTILL="0.25", PFX_ROUTE="blind")
    ds, rt = tel["ds_credit"], tel["rt_credit"]
    jy = [len(FILL) + len(r" \boxed{7}"), len(FILL) + len(r" \boxed{7}"), len(FILL) + len(r" \boxed{4}"), 1 + len(FILL) + len(r" \boxed{3}")]
    cut, d = RT_CASES[0][2].index("\n"), lambda t: 0.25 * (0.1 * (t % 7) + 0.5)
    assert [len(c) for c in calls] == [2, 3] and sorted(rt) == [0, 1, 2] and sorted(ds) == [1, 3]       # 눈 가린 나 1번 · 판정 교사 1번
    assert (tel["rt_rows"], tel["rt_fin_rows"], tel["rt_fix_rows"], tel["rt_oom"]) == (3.0, 1.0, 2.0, 0.0)
    assert ds[1][0] + len(ds[1][1]) == jy[1] and ds[3][0] + len(ds[3][1]) == n[3] and ds[1][0] + len(ds[1][1]) <= rt[1][0]   # 다름만 jY 에서 끊김
    assert rt[0][0] == cut and rt[0][1] == pytest.approx([d(t) for t in range(H.STOP_K)])
    assert "fine.\n\n[Verification" in calls[1][0] and "finish your response" in calls[1][0] and "answered" not in calls[1][0]
    for i in (1, 2):
        assert rt[i][0] == jy[i] + H.DS_FRONT and rt[i][1] == pytest.approx([d(t) for t in range(H.DS_FRONT, n[i] - jy[i])])
        assert f"answered {'37'[i - 1]}" in calls[1][i] and "correct final answer" not in calls[1][i]
    for i in (0, 1, 2):
        j0, v = rt[i]
        assert torch.allclose(adv[i, j0:j0 + len(v)] - base[i, j0:j0 + len(v)], torch.tensor(v, dtype=adv.dtype), atol=1e-3)
    assert torch.allclose(adv[0, cut + H.STOP_K:], base[0, cut + H.STOP_K:]) and torch.allclose(adv[4], base[4])
    calls.clear()
    _, (_, _, tg) = _pfx_run(monkeypatch, PFX_DISTILL="0.25", PFX_ROUTE="gold")
    assert "correct final answer to this problem is 7" not in calls[1][0] and all("correct final answer to this problem is 7" in h for h in calls[1][1:]) and tg["rt_credit"].keys() == rt.keys()
    calls.clear()                                # 수정 85 tie: 끝내기 없음 · 다름(맞/틀 모두)은 Y 줄 끝부터(닫는 $$ 제외, DS_FRONT 건너뜀 없음) · 머리 = 두 답만(정답 쪽지·풀이 없음)
    adv, (_, _, tt) = _pfx_run(monkeypatch, PFX_DISTILL="0.25", PFX_ROUTE="tie")
    assert sorted(tt["rt_credit"]) == [1, 2] and (tt["rt_fin_rows"], tt["rt_fix_rows"], tt["rt_tie_rows"], tt["rt_oom"]) == (0.0, 0.0, 2.0, 0.0) and sorted(tt["ds_credit"]) == [1, 3]
    for i, xy in ((1, "3 and 7"), (2, "7 and 4")):
        assert tt["rt_credit"][i][0] == jy[i] + 2 and tt["rt_credit"][i][1] == pytest.approx([d(t) for t in range(n[i] - jy[i] - 2)])
        assert f"answered {xy}, which disagree" in calls[1][i - 1] and "correct final answer" not in calls[1][i - 1] and "[Verification" not in calls[1][i - 1]
        assert torch.allclose(adv[i, jy[i] + 2:n[i]] - base[i, jy[i] + 2:n[i]], torch.tensor(tt["rt_credit"][i][1], dtype=adv.dtype), atol=1e-3)
    assert torch.allclose(adv[0], base[0])
    assert "OOM" in H.pfx_guard([{"rt_oom": 2.0, "rt_rows": 1.0}])
    monkeypatch.delenv("PFX_ROUTE")
    for bad, why in (({"PFX_ROUTE": "blind"}, "PFX_ROUTE"), ({"PFX_ROUTE": "x", "PFX_DISTILL": "0.25"}, "PFX_ROUTE"), ({"PFX_VERDICT": "0"}, "PFX_VERDICT"),
                     ({"PFX_FORK": "ch"}, "닫힌 축"), ({"PFX_FIN": "0.25"}, "지운 손잡이"), ({"PFX_OUTCOME": "off"}, "지운 손잡이")):
        monkeypatch.delenv("PFX_DISTILL", raising=False)
        with pytest.raises(ValueError, match=why):
            _pfx_run(monkeypatch, **bad)
        for k in bad:
            monkeypatch.delenv(k, raising=False)


def test_pfx_guard_abs_limit(tmp_path, monkeypatch):
    """PFX_GUARD_ABS(수정 53): 씨앗 무관 절대 한계 — 기준선 2 배 규칙이 놓치는 상승을 잡는다."""
    monkeypatch.setenv("MC_CKPT_DIR", str(tmp_path)); monkeypatch.delenv("PFX_GUARD_ABS", raising=False)
    steps = [{"pfx_break_right_live": v} for v in (.10, .10, .10, .15, .15, .15)]
    assert H.pfx_guard(steps) is None                                           # 2 × 기준선 .10 = .20 > .15
    (tmp_path / "FIRST_REF_PFX.json").unlink()
    monkeypatch.setenv("PFX_GUARD_ABS", "0.12")
    assert "0.1200" in H.pfx_guard(steps)                                       # 절대 한계 .12 < .15
    (tmp_path / "FIRST_REF_PFX.json").unlink()
    assert H.pfx_guard([{"pfx_break_right_live": v} for v in (.20, .10, .10)]) is None   # 수정 85b: 스텝 1(학습 전) 튐만으로는 울리지 않음


def test_pfx_break_w_scales_only_right_first_failures(pfx_env, monkeypatch):
    """수정 43 PFX_BREAK_W: 맞은 첫 답을 뒤집어 실패한 행의 결과 adv 만 × w · 틀린 첫 답 행·성공 행은 그대로."""
    pytest.importorskip("verl")
    for k in ("PFX_FORK", "PFX_REP", "PFX_TRUNC"):
        monkeypatch.delenv(k, raising=False)
    a1, _ = _pfx_run(monkeypatch)
    a2, (_, _, tel) = _pfx_run(monkeypatch, PFX_BREAK_W="2")
    assert tel["pfx_break_rows"] == 1.0 and torch.allclose(a2[2], 2 * a1[2]) and float(a1[2, 0]) < 0   # CASES_U 2 = 깨짐
    assert torch.allclose(a2[[0, 1, 3, 4, 5]], a1[[0, 1, 3, 4, 5]])


def test_loop_char_is_end_of_third_consecutive_equivalent_box():
    t = r"\boxed{7} a \boxed{7} b \boxed{7} c"
    assert H.loop_char(t) == t.index(" c") and H.loop_char(r"\boxed{1/2} \boxed{0.5} \boxed{\frac{1}{2}}") is not None
    assert H.loop_char(r"\boxed{7} \boxed{7} \boxed{4} \boxed{7}") is None and H.loop_char("no box") is None


def test_pfx_truncated_continuation_keep_rewards_and_zero_is_removed(pfx_env, monkeypatch):
    """keep: 상한에서 잘린 이어쓰기도 도중 박스로 채점(옛 동작) · zero(28f, 실패)는 수정 33 에서 삭제 → 즉사."""
    pytest.importorskip("verl")
    cases = [(r"y \boxed{7}", "7", FILL + FILL),                          # 맞은 첫 답, 새 박스 없이 상한까지 = 잘림
             (r"y \boxed{7}", "7", " done.")]                          # 짧게 끝냄
    monkeypatch.setattr(sys.modules[__name__], "CASES_U", cases)
    _, (tlr_keep, _, tk) = _pfx_run(monkeypatch)
    assert tk["pfx_cut_rows"] == 1.0 and tk["pfx_cut_correct"] == 1.0 and max(tlr_keep[0]) == 1.0
    for bad in ("zero", "drop"):
        with pytest.raises(ValueError):
            _pfx_run(monkeypatch, PFX_TRUNC=bad)
