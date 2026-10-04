"""scripts/local/regrade_jsonl.py 회귀 시험 — .orig 보존·원자 교체·1→0 차단·라벨 행 보호."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "local"))

import regrade_jsonl as R  # noqa: E402


def _write(p: Path, rows):
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _read(p: Path):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def test_regrade_upgrades_and_preserves_orig(tmp_path):
    p = tmp_path / "texts.jsonl"
    rows = [
        {"problem_id": 0, "gold": "t^7", "text": "so \\boxed{t^7}", "r_corr": 0},
        {"problem_id": 1, "gold": "306", "text": "so \\boxed{153}", "r_corr": 0},
        {"problem_id": 2, "gold": "42", "text": "so \\boxed{42}", "r_corr": 1},
        {"problem_id": 3, "gold": "1", "text": "x", "r_corr": -1},
        {"problem_id": 4, "gold": "1", "text": "x", "r_corr": 0, "mode": "label"},
    ]
    _write(p, rows)
    assert R.main([str(p), "--workers", "1"]) == 0
    out = _read(p)
    assert [r["r_corr"] for r in out] == [1, 0, 1, -1, 0]
    orig = _read(Path(str(p) + ".orig"))
    assert [r["r_corr"] for r in orig] == [0, 0, 1, -1, 0]
    assert not (tmp_path / "texts.jsonl.tmp").exists()


def test_orig_never_overwritten(tmp_path):
    p = tmp_path / "texts.jsonl"
    _write(p, [{"problem_id": 0, "gold": "t^7", "text": "\\boxed{t^7}", "r_corr": 0}])
    assert R.main([str(p), "--workers", "1"]) == 0
    sentinel = Path(str(p) + ".orig")
    sentinel.write_text("SENTINEL\n")
    assert R.main([str(p), "--workers", "1"]) == 0
    assert sentinel.read_text() == "SENTINEL\n"


def test_dry_run_writes_nothing(tmp_path):
    p = tmp_path / "texts.jsonl"
    _write(p, [{"problem_id": 0, "gold": "t^7", "text": "\\boxed{t^7}", "r_corr": 0}])
    before = p.read_text()
    assert R.main([str(p), "--dry_run", "--workers", "1"]) == 0
    assert p.read_text() == before
    assert not Path(str(p) + ".orig").exists()


def test_downgrade_blocks_write_and_exits_2(tmp_path, capsys):
    p = tmp_path / "texts.jsonl"
    _write(p, [{"problem_id": 0, "gold": "306", "text": "\\boxed{153}", "r_corr": 1}])
    before = p.read_text()
    assert R.main([str(p), "--workers", "1"]) == 2
    assert p.read_text() == before
    assert not Path(str(p) + ".orig").exists()
    assert "1→0" in capsys.readouterr().out


def test_gens_gold_lookup_from_texts(tmp_path):
    texts = tmp_path / "texts.jsonl"
    _write(texts, [{"problem_id": 7, "gold": "\\frac{8}{3}", "text": "x", "r_corr": 0}])
    gens = tmp_path / "gens.jsonl"
    _write(gens, [{"problem_id": 7, "text": "so \\boxed{\\frac83}", "r_corr": 0}])
    assert R.main([str(gens), "--gold_from", str(texts), "--workers", "1"]) == 0
    assert _read(gens)[0]["r_corr"] == 1


# ── 게이트 스키마(gen_r_corr) + 1차 시도 r_corr 새로고침 ──────────────────────
def _texts_rows():
    """0: g0 = 재채점으로 정답이 될 1차 시도 · 1,2: 여전히 오답 · 3: 원래 정답."""
    return [
        {"group_id": "g0", "problem_id": 0, "gold": "t^7", "text": "so \\boxed{t^7}",
         "r_corr": 0},
        {"group_id": "g1", "problem_id": 1, "gold": "306", "text": "so \\boxed{153}",
         "r_corr": 0},
        {"group_id": "g2", "problem_id": 2, "gold": "42", "text": "so \\boxed{7}", "r_corr": 0},
        {"group_id": "g3", "problem_id": 3, "gold": "5", "text": "so \\boxed{5}", "r_corr": 1},
    ]


def _gate_gens():
    rows = []
    for gi, (gid, gold, hit) in enumerate([("g0", "t^7", "\\boxed{t^7}"),
                                           ("g1", "306", "\\boxed{306}"),
                                           ("g2", "42", "\\boxed{7}")]):
        for cond in ("blind", "external"):
            for k in range(2):
                txt = hit if (cond == "external" or k == 0) else "\\boxed{999}"
                rows.append({"roll_id": f"{gid}#{gi}", "group_id": gid, "population": "wrong",
                             "cond": cond, "r_corr": 0, "gen_r_corr": 0, "text": txt,
                             "truncated": 0})
    for cond in ("blind", "external"):
        for k in range(2):
            rows.append({"roll_id": "g3#3", "group_id": "g3", "population": "correct",
                         "cond": cond, "r_corr": 1, "gen_r_corr": 1, "text": "\\boxed{5}",
                         "truncated": 0})
    return rows


def test_gate_schema_regrades_gen_and_refreshes_a1(tmp_path, capsys):
    texts = tmp_path / "texts.jsonl"
    _write(texts, _texts_rows())
    assert R.main([str(texts), "--workers", "1"]) == 0        # 1차: texts 재채점
    assert [r["r_corr"] for r in _read(texts)] == [1, 0, 0, 1]

    gens = tmp_path / "gens.jsonl"
    _write(gens, _gate_gens())
    assert R.main([str(gens), "--gold_from", str(texts), "--workers", "1"]) == 0
    out = capsys.readouterr().out
    assert "schema=gate field=gen_r_corr" in out
    assert "n_a1_refreshed=4" in out and "n_a1_unmatched=0" in out
    rows = _read(gens)
    # 생성 정오는 text vs gold 로 다시, 1차 정오는 재채점된 texts 에서
    assert all(r["r_corr"] == (1 if r["group_id"] in ("g0", "g3") else 0) for r in rows)
    g0 = [r for r in rows if r["group_id"] == "g0"]
    assert [r["gen_r_corr"] for r in g0] == [1, 0, 1, 1]
    assert all(r["gen_r_corr"] == 0 for r in rows if r["group_id"] == "g2")


def test_gate_schema_blocks_gen_downgrade(tmp_path, capsys):
    texts = tmp_path / "texts.jsonl"
    _write(texts, _texts_rows())
    gens = tmp_path / "gens.jsonl"
    _write(gens, [{"roll_id": "g1#1", "group_id": "g1", "population": "wrong", "cond": "blind",
                   "r_corr": 0, "gen_r_corr": 1, "text": "\\boxed{999}", "truncated": 0}])
    before = gens.read_text()
    assert R.main([str(gens), "--gold_from", str(texts), "--workers", "1"]) == 2
    assert gens.read_text() == before
    assert "1→0" in capsys.readouterr().out


def test_gate_schema_counts_unmatched_roll_id(tmp_path, capsys):
    texts = tmp_path / "texts.jsonl"
    _write(texts, _texts_rows())
    gens = tmp_path / "gens.jsonl"
    _write(gens, [{"roll_id": "zz#99", "group_id": "g1", "population": "wrong", "cond": "blind",
                   "r_corr": 0, "gen_r_corr": 0, "text": "\\boxed{306}", "truncated": 0}])
    assert R.main([str(gens), "--gold_from", str(texts), "--workers", "1"]) == 0
    assert "n_a1_unmatched=1" in capsys.readouterr().out
    assert _read(gens)[0]["r_corr"] == 0            # 못 찾으면 건드리지 않는다


def test_effort_schema_regrades_r_corr_and_skips_boxless_continuation(tmp_path, capsys):
    texts = tmp_path / "texts.jsonl"
    _write(texts, _texts_rows())
    gens = tmp_path / "gens.jsonl"
    _write(gens, [
        {"cond": "blind", "group_id": "g1", "roll_id": "g1#1", "population": "wrong",
         "text": "\\boxed{306}", "r_corr": 0, "truncated": 0, "n_gen_tokens": 10, "rounds": 1,
         "degenerate": 0},
        {"cond": "wait_forced_1024", "group_id": "g1", "roll_id": "g1#1", "population": "wrong",
         "text": "hmm no box here", "r_corr": 0, "truncated": 0, "n_gen_tokens": 10,
         "rounds": 1, "degenerate": 0},
    ])
    assert R.main([str(gens), "--gold_from", str(texts), "--workers", "1"]) == 0
    out = capsys.readouterr().out
    assert "schema=effort field=r_corr" in out and "n_skipped_cont=1" in out
    assert [r["r_corr"] for r in _read(gens)] == [1, 0]


def test_zero_graded_rows_aborts_loudly(tmp_path, capsys):
    """gold 를 한 행도 못 찾으면 «0행 변경»으로 조용히 성공하면 안 된다 — 안 쓰고 rc 3."""
    p = tmp_path / "texts.jsonl"
    _write(p, [{"problem_id": 0, "text": "so \\boxed{306}", "r_corr": 0},
               {"problem_id": 1, "text": "so \\boxed{42}", "r_corr": 0}])
    assert R.main([str(p), "--workers", "1"]) == 3
    assert "0행" in capsys.readouterr().out
    assert [r["r_corr"] for r in _read(p)] == [0, 0]     # 쓰지 않았다
    assert not (tmp_path / "texts.jsonl.orig").exists()
