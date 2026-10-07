"""SPONT_PFX(0923) — 앞부분 재시작 수정: 빌더(분리·자름·LOO 라벨·균형) · 보상(새 박스 없으면 앞부분 답) ·
계기(`mc.probe.score` 와 같은 자) · 프롬프트(verl 에이전트 루프가 앞부분을 prompt_ids 에 붙인다) · 가드."""
from __future__ import annotations

import asyncio
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


def _pfx_run(monkeypatch, weight=None, **env):
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


def test_fork_weights_mean_one_and_outcome_direction():
    from mc import credit as C
    sc = [0.0, 0.0, 2.0, 0.0, -2.0, 0.0]
    w = C.fork_weights(sc, 1.0, min_n=2)
    assert sum(w) / len(w) == pytest.approx(1.0) and w[2] == max(w) and w[4] == min(w)   # 성공: + 대조 토큰에 몰림
    v = C.fork_weights(sc, -1.0, min_n=2)
    assert sum(v) / len(v) == pytest.approx(1.0) and v[4] == max(v)                       # 실패: − 대조 토큰에 벌 몰림
    assert C.fork_weights([1.0, 1.0, 1.0], 1.0) == [1.0, 1.0, 1.0] and C.fork_weights([3.0], -1.0) == [1.0]


def test_pfx_fork_reweights_only_mixed_groups_and_preserves_row_mass(pfx_env, monkeypatch):
    """수정 28 CH-Fork: 섞인 묶음 행만 가중(교차 적합 이웃·중립 문구 문맥), 행 결과 adv 총량 보존, 결과 상수 묶음 = B."""
    pytest.importorskip("verl")
    for k in ("PFX_REP", "PFX_FORK"):
        monkeypatch.delenv(k, raising=False)
    real_init = _PfxBatch.__init__

    def init(self, cases=CASES):                                         # 행 0~3 한 묶음(성공 0·3, 실패 1·2) · 4~5 실패만
        real_init(self, cases)
        self.non_tensor_batch["uid"] = ["g0"] * 4 + ["g1"] * 2
        for e in self.non_tensor_batch["extra_info"]:
            e["problem"], e["prior_a0"] = "Q", 0.5
    monkeypatch.setattr(_PfxBatch, "__init__", init)
    base, *_ = _pfx_run(monkeypatch)
    seen = []

    def fake(tr, trees, per_token=False):                              # 성공 이웃 문맥(짝의 앞)이면 앞 3 토큰 +1
        seen.extend(trees)
        return [[(1.0 if q % 2 == 0 else 0.0) if j < 3 else 0.0 for j in range(m)] for q, (_, [(t, toks, m)]) in enumerate(trees)]
    from mc import context as ctx
    from mc import trainer as T
    monkeypatch.setattr(T, "ref_tree_score", fake)
    monkeypatch.setattr(ctx, "turn1_prompt", lambda tok, prob, v: "P:" + prob)
    adv, (_, credit, tel) = _pfx_run(monkeypatch, PFX_FORK="ch")
    w = tel["pfx_weights"]
    assert sorted(w) == [0, 1, 2, 3] and tel["fork_rows"] == 4.0 and len(seen) == 8 and not credit
    for i in range(4):
        n = len(CASES_U[i][2])
        assert float(adv[i, :n].sum()) == pytest.approx(float(base[i, :n].sum()), rel=1e-5)     # 행 총량 보존
        up = abs(float(adv[i, 0])) > abs(float(base[i, 0]))                                   # 성공 행: 선생님이 좋아한
        assert up == (i in (0, 3))                                                             # 앞 토큰에 칭찬↑, 실패 행: 벌↓
    assert torch.allclose(adv[4:], base[4:])                                                   # 결과 상수 묶음 = B
    with pytest.raises(ValueError, match="ch 만"):
        _pfx_run(monkeypatch, PFX_FORK="hsd")


def test_fork_weights_tiny_rows_uniform_and_neighbor_masks_answer():
    """수정 28b: 16 토큰 미만 행은 균등(= B) · 이웃 이어쓰기의 박스 답은 가려서 선생님이 정답을 베끼지 못한다."""
    from mc import credit as C
    assert C.fork_weights([0.0, 5.0, 0.0], 1.0) == [1.0, 1.0, 1.0]
    t = H.fork_neighbor(r"so x=3, hence \boxed{12} and \boxed{\frac{1}{2}}." + "z" * (H.FORK_CAP + 50))
    assert "12" not in t and "frac" not in t and t.count(r"\boxed{...}") == 2 and len(t) <= H.FORK_CAP + 20


def test_pfx_fork_zone_only_lone_success_fallback_and_oom(pfx_env, monkeypatch):
    """수정 28b: 가중은 누설(첫 새 답 진술) 전 구간만 · 외톨이 성공은 s⁺ 대신 «문맥 없음» 으로 대조(ch) · OOM 행은 균등+계수."""
    pytest.importorskip("verl")
    for k in ("PFX_REP", "PFX_FORK"):
        monkeypatch.delenv(k, raising=False)
    cases = [(r"x \boxed{3}", "7", FILL + r" \boxed{7}"),               # 외톨이 성공(고침) — 누설 = 새 박스
             (r"x \boxed{3}", "7", FILL + r" \boxed{9}"),               # 실패
             (r"x \boxed{3}", "7", FILL + " so it stays.")]            # 실패
    monkeypatch.setattr(sys.modules[__name__], "CASES_U", cases)
    real_init = _PfxBatch.__init__

    def init(self, cases=CASES):
        real_init(self, cases)
        self.non_tensor_batch["uid"] = ["g0"] * 3
        for e in self.non_tensor_batch["extra_info"]:
            e["problem"] = "Q"
    monkeypatch.setattr(_PfxBatch, "__init__", init)
    base, *_ = _pfx_run(monkeypatch)
    heads = []

    def fake(tr, trees, per_token=False):
        heads.extend("".join(chr(c) for c in h if c < 0x110000) for h, _ in trees)
        out = [[(1.0 if "<<<" in heads[-len(trees) + q] else 0.0) if j < 3 else 0.0 for j in range(bl[0][2])]
               for q, (h, bl) in enumerate(trees)]
        out[-1] = None                                                 # 마지막 트리 OOM
        return out
    from mc import context as ctx
    from mc import trainer as T
    monkeypatch.setattr(T, "ref_tree_score", fake)
    monkeypatch.setattr(ctx, "turn1_prompt", lambda tok, prob, v: "P:" + prob)
    adv, (_, _, tel) = _pfx_run(monkeypatch, PFX_FORK="ch")
    assert "<<<" not in heads[0] and "<<<" in heads[1]              # 외톨이 성공 행: (없음, s⁻) 대조
    assert tel["fork_oom"] == 1.0 and tel["fork_rows"] == 2.0 and tel["fork_rows_succ"] == 1.0
    w0 = tel["pfx_weights"][0]
    n0 = len(CASES_U[0][2])
    tl = len(FILL) + 1                                                 # 누설 = « \boxed{7}» 앞
    assert w0[tl:] == [1.0] * (len(w0) - tl) and max(w0[:tl]) > 1.0   # 누설 뒤 = 1, 앞에서만 재배분
    assert float(adv[0, :n0].sum()) == pytest.approx(float(base[0, :n0].sum()), rel=1e-5)
    assert torch.allclose(adv[2], base[2])                            # OOM 행 = B


def test_pfx_fork_lone_row_borrows_bank_neighbor(pfx_env, monkeypatch):
    """수정 45: 외톨이 성공 행은 `succ_bank`(base 거르기 이어쓰기)에서 s⁺ 를 빌린다 — 문맥 없음 대조로 약해지지 않게."""
    pytest.importorskip("verl")
    for k in ("PFX_REP", "PFX_FORK"):
        monkeypatch.delenv(k, raising=False)
    cases = [(r"x \boxed{3}", "7", FILL + r" \boxed{7}"), (r"x \boxed{3}", "7", FILL + r" \boxed{9}"),
             (r"x \boxed{3}", "7", FILL + " so it stays.")]
    _fork_setup(monkeypatch, cases, ["g0"] * 3)
    real_init = _PfxBatch.__init__

    def init(self, cases=CASES):
        real_init(self, cases)
        for e in self.non_tensor_batch["extra_info"]:
            e["succ_bank"], e["fail_bank"] = ["BANKTEXT"], []
    monkeypatch.setattr(_PfxBatch, "__init__", init)
    heads = []

    def fake(tr, trees, per_token=False):
        heads.extend(_heads_str(h) for h, _ in trees)
        return [[1.0 if ("BANKTEXT" in _heads_str(h)) == (j % 2 == 0) else 0.0 for j in range(bl[0][2])] for h, bl in trees]
    from mc import trainer as T
    monkeypatch.setattr(T, "ref_tree_score", fake)
    _, (_, _, tel) = _pfx_run(monkeypatch, PFX_FORK="ch")
    assert "BANKTEXT" in heads[0] and "<<<" in heads[1] and tel["fork_bank"] == 1.0 and tel["fork_rows"] == 3.0
    assert all(sum(w) == pytest.approx(len(w)) for w in tel["pfx_weights"].values())


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
    손실 마스크 0(token-mean 분모·KL 밖) · r 은 실제 결과 기록 · CH-Fork 대조·이웃에서도 빠진다."""
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
    from mc import trainer as T
    heads = []
    monkeypatch.setattr(T, "ref_tree_score", lambda tr, trees, per_token=False: (
        heads.extend(trees) or [[0.0] * m for _, [(t, toks, m)] in trees]))
    adv2, (_, _, tel2) = _pfx_run(monkeypatch, PFX_TRUNC="mask", PFX_FORK="ch")
    assert sorted(tel2["pfx_weights"]) == [0, 1, 2] and tel2["pfx_weights"][0] == [0.0] * len(cases[0][2])
    assert len(heads) == 4 and float(adv2[0].abs().sum()) == 0.0                   # CH 는 행 1·2 만(각 2 문맥)


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
    for k in ("PFX_DISTILL_ROWS", "PFX_REP_HARD"):                       # 지운 손잡이(59b·46) = 즉사
        with pytest.raises(ValueError, match="지운 손잡이"):
            _pfx_run(monkeypatch, PFX_DISTILL="0.25", **{k: "1"})
        monkeypatch.delenv(k)


def test_pfx_production_combo_mask_breakw_alloc_distill(pfx_env, monkeypatch):
    """수정 62b — 실제 팔 조합(PFX_TRUNC=mask · BREAK_W 2 · ALLOC · DISTILL): 잘린 행 adv 0 · BREAK_W 는 배분 뒤 곱(맞→틀 행만 ×2) ·
    증류 가산은 틀린 첫 답 행에만 · adv = 결과 adv × 무게 + 증류 크레딧."""
    pytest.importorskip("verl")
    for k in ("PFX_FORK", "PFX_REP", "PFX_KEEP", "PFX_BREAK_W"):
        monkeypatch.delenv(k, raising=False)
    cases = [(r"x \boxed{3}", "7", FILL + r" \boxed{7}"), (r"x \boxed{3}", "7", FILL + r" \boxed{3} again."),   # 틀→맞 · 틀→같은 오답
             (r"y \boxed{7}", "7", FILL + " done."), (r"y \boxed{7}", "7", FILL + r" \boxed{4}"),             # 맞→맞 · 맞→틀
             (r"w \boxed{3}", "7", FILL * 3), (r"w \boxed{3}", "7", FILL)]                                    # 잘림(가장 긴 행) · 외톨이
    _fork_setup(monkeypatch, cases, ["g0", "g0", "g1", "g1", "g2", "g2"])
    real_init = _PfxBatch.__init__

    def init(self, cases=CASES):
        real_init(self, cases)
        self.batch["old_log_probs"] = torch.full(self.batch["responses"].shape, -1.0)
        self.batch["ref_log_prob"] = torch.zeros(self.batch["responses"].shape)
    monkeypatch.setattr(_PfxBatch, "__init__", init)
    from mc import trainer as T
    monkeypatch.setattr(T, "ref_tree_score", lambda tr, trees, per_token=False:
                        [[0.3 * (t % 5) for t in range(bl[0][2])] for _, bl in trees])
    base, _ = _pfx_run(monkeypatch, PFX_TRUNC="mask")
    _, (_, _, t1) = _pfx_run(monkeypatch, PFX_TRUNC="mask", PFX_ALLOC="0.5", PFX_DISTILL="0.25")
    adv, (_, _, t2) = _pfx_run(monkeypatch, PFX_TRUNC="mask", PFX_ALLOC="0.5", PFX_DISTILL="0.25", PFX_BREAK_W="2")
    w1, w2, ds = t1["pfx_weights"], t2["pfx_weights"], t2["ds_credit"]
    assert t2["pfx_drop"] == [4] and float(adv[4].abs().sum()) == 0.0 and sorted(ds) == [0, 1, 5]
    assert w2[3] == pytest.approx([2 * v for v in w1[3]]) and all(w2[i] == pytest.approx(w1[i]) for i in (0, 1, 2))
    for i in range(4):
        n, cr = len(cases[i][2]), torch.zeros(len(cases[i][2]), dtype=adv.dtype)
        if i in ds:
            cr[ds[i][0]:ds[i][0] + len(ds[i][1])] = torch.tensor(ds[i][1], dtype=adv.dtype)
        assert torch.allclose(adv[i, :n], base[i, :n] * torch.tensor(w2[i], dtype=adv.dtype) + cr)


def test_pfx_alloc_mixed_groups_change_vs_keep_tokens(pfx_env, monkeypatch):
    """수정 61/61c PFX_ALLOC: 섞인 묶음 행만(만장일치·짧은 행·틀→다른 틀 제외) · 바꾸기 점수 c = 눈 가린 나 − 어제의 나(ref_log_prob) ·
    답을 바꾼 행은 c 큰 말에, 지킨 행은 c 작은 말에 무게 · 구간 [16, j1) 평균 1 · 밖 1 · 길이 n_tok · CH 와 배타."""
    pytest.importorskip("verl")
    for k in ("PFX_FORK", "PFX_REP", "PFX_BREAK_W", "PFX_TRUNC", "PFX_DISTILL"):
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
        self.batch["ref_log_prob"] = torch.zeros(self.batch["responses"].shape)
    monkeypatch.setattr(_PfxBatch, "__init__", init)
    from mc import trainer as T
    monkeypatch.setattr(T, "ref_tree_score", lambda tr, trees, per_token=False:
                        [[0.3 * (t % 5) for t in range(bl[0][2])] for _, bl in trees])          # c_t = 0, .3, .6, .9, 1.2, …
    base, _ = _pfx_run(monkeypatch)
    adv, (_, _, tel) = _pfx_run(monkeypatch, PFX_ALLOC="0.5")
    w = tel["pfx_weights"]
    assert sorted(w) == [0, 2, 3] and tel["alloc_skip_ww"] == 1.0                    # g2 만장일치 · 4 짧음 · 1 틀→다른 틀(61c) 제외
    assert tel["alloc_changed"] == pytest.approx(2 / 3) and torch.allclose(adv[1], base[1])
    for i, ch in ((0, True), (2, False), (3, True)):
        n = len(w[i])
        assert w[i][:H.DS_FRONT] == [1.0] * H.DS_FRONT and sum(w[i]) == pytest.approx(n)        # 구간 평균 1 · 밖 1 = 합 보존
        a, b = w[i][19], w[i][20]                                                        # 말 19: c 1.2 · 말 20: c 0
        assert (a > b) if ch else (a < b)
        assert torch.allclose(adv[i, :n], base[i, :n] * torch.tensor(w[i], dtype=adv.dtype))
    _, (_, _, ts) = _pfx_run(monkeypatch, PFX_ALLOC="0.5", PFX_ALLOC_SHUF="1")                   # 61e 위약: 같은 행·같은 무게 묶음, 자리만 섞임
    monkeypatch.delenv("PFX_ALLOC_SHUF")
    assert ts["alloc_shuf"] == 1.0 and sorted(ts["pfx_weights"]) == [0, 2, 3]
    assert all(sorted(ts["pfx_weights"][i]) == pytest.approx(sorted(w[i])) and ts["pfx_weights"][i] != w[i] for i in (0, 2, 3))
    with pytest.raises(ValueError, match="PFX_ALLOC"):
        _pfx_run(monkeypatch, PFX_ALLOC="0.5", PFX_FORK="ch")
    monkeypatch.delenv("PFX_FORK")
    with pytest.raises(ValueError, match="양수만"):
        _pfx_run(monkeypatch, PFX_ALLOC="0")


def test_pfx_keep_right_first_rows_only_and_abs_guard(pfx_env, monkeypatch, tmp_path):
    """수정 53 PFX_KEEP: 맞은 첫 답 행만 · 이어쓰기 전체에 γ·clip(ref − old, ±2) 가산(선생님 = 같은 문맥 원래 모델) ·
    틀린 첫 답 행 adv 무변 · keep_share 계기. PFX_GUARD_ABS: 파괴 가드가 «2 × 기준선» 대신 절대 한계를 쓴다."""
    pytest.importorskip("verl")
    for k in ("PFX_FORK", "PFX_REP", "PFX_BREAK_W", "PFX_TRUNC", "PFX_DISTILL"):
        monkeypatch.delenv(k, raising=False)
    cases = [(r"x \boxed{3}", "7", FILL + r" \boxed{7}"), (r"x \boxed{3}", "7", FILL + " no."),
             (r"y \boxed{7}", "7", FILL + " done."), (r"y \boxed{7}", "7", FILL + r" \boxed{4}")]
    _fork_setup(monkeypatch, cases, ["g0", "g0", "g1", "g1"])
    real_init = _PfxBatch.__init__

    def init(self, cases=CASES):
        real_init(self, cases)
        self.batch["old_log_probs"] = torch.full(self.batch["responses"].shape, -1.0)
        self.batch["ref_log_prob"] = torch.full(self.batch["responses"].shape, 2.0)   # d = 3 → clip 2
    monkeypatch.setattr(_PfxBatch, "__init__", init)
    base, _ = _pfx_run(monkeypatch)
    adv, (_, _, tel) = _pfx_run(monkeypatch, PFX_KEEP="0.5")
    kc = tel["keep_credit"]
    assert sorted(kc) == [2, 3] and tel["keep_rows"] == 2.0 and tel["keep_d_mean"] == pytest.approx(H.DS_CLIP)
    for i, (j0, vals) in kc.items():
        n = len(cases[i][2])
        assert j0 == 0 and vals == pytest.approx([0.5 * H.DS_CLIP] * len(vals))
        assert torch.allclose(adv[i, :n] - base[i, :n], torch.full((n,), 0.5 * H.DS_CLIP, dtype=adv.dtype))
    assert torch.allclose(adv[:2], base[:2]) and 0 < tel["keep_share"] <= tel["mass_share"]
    monkeypatch.setenv("MC_CKPT_DIR", str(tmp_path))
    steps = [{"pfx_break_right_live": v} for v in (.10, .10, .10, .15, .15, .15)]
    assert H.pfx_guard(steps) is None                                           # 2 × 기준선 .10 = .20 > .15
    (tmp_path / "FIRST_REF_PFX.json").unlink()
    monkeypatch.setenv("PFX_GUARD_ABS", "0.12")
    assert "0.1200" in H.pfx_guard(steps)                                       # 절대 한계 .12 < .15


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
