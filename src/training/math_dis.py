r"""MATH_DIS — «불일치 진단»(disagreement diagnosis) 메타인지의 단일 진실 원천.

★왜 이 파일인가 (2026-09-14, cd9). 이 4B 비사고(non-thinking) 정책에서 **한 풀이 안의**
모든 언어적 메타 블록(verify / critique / redirect / plan)은 인과적으로 무력했다 —
네 관문 전부에서 own ≤ donor 였다(docs/HYPOTHESIS_LEDGER_cd9.md). 즉 «자기 풀이를 보고
쓴 문장»에는 그 풀이를 고칠 정보가 들어 있지 않다. 반면 **문제 안에서** 유일하게 살아
있는 신호는 자기 샘플들 사이의 불일치다(형제 답 점유율, 문제별 AUC .754).

그래서 M_DIS 는 «메타를 쓰는 습관»을 바로 그 신호 위에서 돌린다:
  프롬프트 — 문제 + 이 정책 **자신의** 후보 풀이 마무리 K=4 개(`[Candidate 1..4]`).
  응답     — `<meta>…</meta>` 한 블록에 DIAGNOSIS(어느 후보가 어디서 갈리는지, 어떤
             가정·계산이 틀렸는지 2~4문장, 후보 번호를 인용)를 쓰고 `Commit: <n>` 으로
             닫은 뒤, 최종 답을 \boxed{} 로 낸다.
  채점     — 답 스팬은 **gold 정오**(모든 팔과 같다). 메타 스팬은 **자기 증류 판단 크레딧**:
             라벨이 gold 가 아니라 **그 GRPO 그룹의 형제 다수결**(이 행 자신은 제외한
             LEAVE-ONE-OUT)에서 온다.

  dis_row_credit = +1  커밋한 후보의 저장된 답 ≡ LOO 형제 다수답(plurality, 자기 자신 제외)
                   −1  다르다
                   미정의(member=False, 0)  커밋 미파싱·범위 밖·메타 없음·블록 다수·
                                            메타 안 \boxed·LOO 형제 2 미만·다수답 미정(동률/무답)

  ★이것은 판단 스팬에 한정한 TTRL 식 «다수결을 라벨로»다. **gold 는 메타 크레딧에
    한 번도 들어가지 않는다**(들어가면 «판단»이 아니라 «정답 복사»를 보상한다).
  ★`cand_correct`(후보들의 gold 정오)는 parquet 에 실리지만 **지표 전용**이다 — 보상
    경로의 어떤 함수도 읽지 않는다(테스트가 gold 없는 행으로 확인한다).

이 파일을 데이터 빌더(scripts/local/build_math_dis_parquet.py)·팔(src/training/math_meta.py)
·평가(scripts/local/math_dis_eval.py)·게이트(scripts/local/math_disagree_gate.py)가 **함께**
쓴다 — 스케치·프롬프트·커밋 파싱·크레딧 정의가 네 군데서 갈리면 학습과 판정을 나란히 읽을
수 없다(M_CRIT 의 critique_leaks/assign_donors 를 이리로 합친 것과 같은 이유).
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from src.training import countdown_rewards as _cdr
from src.training import math_meta as _mm

# 후보 개수와 스케치 길이 — 프롬프트·파서·크레딧이 전부 이 두 상수에 묶인다.
N_CAND = 4
SKETCH_CHARS = 400

DIS_SPEC_VERSION = "math-dis-0914"


# ── 후보 스케치 ────────────────────────────────────────────────────────────────
def strip_meta_blocks(text: str) -> str:
    r"""<meta>…</meta> 를 통째로 지운 본문(`math_meta._meta_block_spans` — 같은 블록 스캐너).
    ★0914 리뷰 D4: `scripts/local/math_disagree_gate.py` 의 `strip_meta` 와 **같은 로직**이다
    (그 파일이 이 함수를 그대로 쓴다) — 후보에 남의 메타 문장이 실려 가면 «후보들의 불일치»가
    아니라 «후보들의 메타 문체»를 진단하게 된다."""
    t = text or ""
    blocks = _mm._meta_block_spans(t)
    if not blocks:
        return t
    keep, prev = [], 0
    for a, b in blocks:
        keep.append(t[prev:a])
        prev = b
    keep.append(t[prev:])
    return "".join(keep)


def sketch_tail(text: str, *, chars: int = SKETCH_CHARS) -> str:
    r"""메타 제거 + 마지막 \boxed{ **앞** chars 자(박스가 없으면 꼬리 chars 자) — «마무리
    스케치»의 저수준 조각(답 줄은 붙이지 않는다).

    ★0914 리뷰 D4: `make_sketch`(아래)와 `math_disagree_gate.candidate_sketch` 가 **공유**한다.
      게이트는 답 줄을 여기 붙이지 않고 후보 렌더링 자리에서 따로 조립한다(그 파일의
      `render_candidates`) — 그래서 «답 줄을 붙이는가»만 두 호출자가 갈리고, 메타 제거·꼬리
      추출 로직은 한 곳(여기)에서만 정의한다.
    """
    t = strip_meta_blocks(text)
    spans = _mm.boxed_spans(t)
    cut = spans[-1][1] if spans else len(t)          # 마지막 \boxed 의 시작 오프셋
    return t[max(0, cut - chars):cut].strip()


def make_sketch(text: str) -> str:
    r"""한 롤아웃 → 프롬프트에 넣을 «마무리 스케치».

    ★메타 제거·꼬리 추출은 `sketch_tail`(공유 조각, D4) — 이 함수가 얹는 것은 답 줄뿐이다.
    ★마지막 줄은 `Final answer: \boxed{…}`(math_meta.last_boxed) — 후보의 답이 **한 자리에**
      고정돼야 모델도 우리도 같은 것을 «그 후보의 답»으로 읽는다.
    ★0914 리뷰 D3(fail-loud): 후보에 \boxed 답이 **없으면 즉사**한다. 예전엔 빈 `\boxed{}` 를
      붙여 넘어갔는데, 그러면 잘린 롤아웃이 «빈 답» 후보가 되어 (a) 자기 군집을 이뤄 다수결
      동점을 만들거나 이기고 (b) 커밋되면 −1(미정의가 아니라 오답)로 채점된다 — 답 없는 후보는
      애초에 후보 목록에 들면 안 된다(build_math_dis_parquet.py 가 그렇게 거른다). 여기서
      raise 하는 것은 그 필터가 새는 것을 다시는 조용히 넘기지 않기 위해서다.
    """
    t = strip_meta_blocks(text)
    ans = _mm.last_boxed(t)
    if not ans:
        raise ValueError("[MATH][DIS] make_sketch: 후보에 \\boxed 답이 없다(잘린 롤아웃?) — "
                         "호출자가 후보로 뽑기 전에 걸러야 한다")
    body = sketch_tail(text, chars=SKETCH_CHARS)
    return f"{body}\nFinal answer: \\boxed{{{ans}}}"


# ── 프롬프트(사용자 턴) ─────────────────────────────────────────────────────────
# ★시스템 프롬프트는 math_opt 와 **바이트 동일**하다(src/metacot/math_meta_prompt.py 의
#   `math_dis` 변형) — 갈리면 «허가된 메타»의 세금이 두 팔에서 달라진다. 갈리는 것은 이
#   사용자 턴 접미뿐이다.
# ★`\boxed{}` 가 들어 있으므로 str.format 을 쓰지 않는다(중괄호가 포맷 자리로 읽힌다) —
#   머리/꼬리를 상수로 두고 후보 블록만 조립한다.
_DIS_ASK_HEAD = "\n\nHere are 4 candidate endings from earlier attempts at this problem:\n"
_DIS_ASK_TAIL = (
    "\n"
    "The candidates may disagree. Inside one <meta>...</meta> block, write a DIAGNOSIS: "
    "which candidates disagree, at which step, and which assumption or computation is "
    "wrong (2-4 sentences, cite candidate numbers), and end the block with "
    "`Commit: <candidate number>`. Then give the final answer in \\boxed{}."
)
# 후보 자리표시자를 채운 «빈» 형태 — 문서·테스트가 지시문 전문을 한 자리에서 읽게 한다.
DIS_ASK = (_DIS_ASK_HEAD
           + "".join(f"[Candidate {i}]\n...\n" for i in range(1, N_CAND + 1))
           + _DIS_ASK_TAIL)

# 변형 어서션·빌더가 공유하는 표식 — 이 문자열이 사용자 턴에 없으면 math_dis 프롬프트가 아니다.
CANDIDATE_MARKER = "[Candidate 1]"
assert CANDIDATE_MARKER in DIS_ASK


def build_dis_user_turn(problem: str, sketches: Sequence[str]) -> str:
    """문제 + DIS_ASK(스케치 4개 삽입). 스케치가 N_CAND 개가 아니면 즉사 —
    후보 수가 조용히 달라지면 `Commit: <n>` 의 범위 검사와 다수 판정이 같이 흔들린다."""
    sk = [str(s or "").strip() for s in sketches]
    if len(sk) != N_CAND:
        raise ValueError(f"[MATH][DIS] 스케치가 {len(sk)} 개다 — N_CAND={N_CAND} 이어야 한다")
    body = "".join(f"[Candidate {i}]\n{s}\n" for i, s in enumerate(sk, start=1))
    return str(problem).strip() + _DIS_ASK_HEAD + body + _DIS_ASK_TAIL


# ── 응답 파싱 ──────────────────────────────────────────────────────────────────
_COMMIT_RE = re.compile(r"commit\s*:\s*(\d+)", re.IGNORECASE)
_CITE_RE = re.compile(r"candidate\s*(\d+)", re.IGNORECASE)
# `Commit: n` 줄은 진단 문장이 아니다 — diag_words 에서 뺀다.
_COMMIT_LINE_RE = re.compile(r"^\s*commit\s*:.*$", re.IGNORECASE | re.MULTILINE)
_META_TAG_RE = re.compile(r"</?meta>", re.IGNORECASE)
_WORD_RE = re.compile(r"\w+")


# TODO(0915): math_disagree_gate.py 는 이 함수를 import 하지 않는다(0914 리뷰 D4) — 그 게이트는
# 의도적으로 **마지막** 매치·더 느슨한 정규식·가변 k 를 쓴다(math_disagree_gate.py 의 parse_commit
# 바로 위 주석에 근거). SKETCH_CHARS·스케치 추출(sketch_tail)만 공유하고 parse_commit 은 갈라
# 둔다 — 합치면 둘 중 하나의 의도된 동작이 깨진다.
def parse_commit(meta_text: str) -> int | None:
    """메타 블록 원문 → 커밋한 후보 번호(1..N_CAND). **첫** 매치만 본다(둘 이상 쓰면 첫 것이
    구속력을 갖는다 — 나중 것으로 바꿔 쓰면 «말 바꾸기»가 공짜가 된다). 범위 밖·없음은 None."""
    m = _COMMIT_RE.search(meta_text or "")
    if not m:
        return None
    try:
        v = int(m.group(1))
    except ValueError:
        return None
    return v if 1 <= v <= N_CAND else None


def cited_candidates(meta_text: str) -> set[int]:
    """메타가 인용한 후보 번호 집합(`Candidate <d>`) — 범위 밖 숫자도 그대로 담는다
    («두 개 이상을 짚었는가»가 관심사이지 번호의 유효성이 아니다)."""
    return {int(m.group(1)) for m in _CITE_RE.finditer(meta_text or "")}


def diag_text(meta_text: str) -> str:
    """진단 문장만 — `<meta>` 태그와 `Commit:` 줄을 뺀 나머지. 정형문 자(countdown_rewards.
    boilerplate_rate)가 읽는 `body` 이기도 하다(행의 `meta` 필드에 이 값을 심는다)."""
    t = _META_TAG_RE.sub(" ", meta_text or "")
    t = _COMMIT_LINE_RE.sub(" ", t)
    return " ".join(t.split())


def diag_words(meta_text: str) -> int:
    """진단 문장의 단어 수. 짧으면 진단이 아니라 커밋 한 줄만 쓴 것이고, 길면 메타 자리에서
    문제를 다시 푼 것이다."""
    return len(_WORD_RE.findall(diag_text(meta_text)))


def plurality_answer(answers: Sequence) -> str | None:
    r"""수학 동치(`math_meta.answers_equivalent`)로 묶은 **최대 군집**의 대표 답.
    빈 답은 세지 않는다. 군집이 없거나 **최대가 둘 이상이면 None**(= 다수 미정) —
    동률에서 하나를 고르면 «형제 다수결»이 사실은 «배치 순서»가 된다."""
    clusters: list[list[str]] = []
    for a in answers:
        s = str(a or "").strip()
        if not s:
            continue
        for c in clusters:
            if _mm.answers_equivalent(c[0], s):
                c.append(s)
                break
        else:
            clusters.append([s])
    if not clusters:
        return None
    top = max(len(c) for c in clusters)
    best = [c for c in clusters if len(c) == top]
    return best[0][0] if len(best) == 1 else None


def all_agree(answers: Sequence) -> int:
    """네(= 주어진) 후보의 답이 **전부 비어 있지 않고 서로 동치**인가 — 빌더의
    `cand_agree_all` 과 평가의 all_agree_frac 이 같은 정의를 쓴다."""
    xs = [str(a or "").strip() for a in answers]
    if not xs or not all(xs):
        return 0
    return int(all(_mm.answers_equivalent(xs[0], a) for a in xs[1:]))


def parse_dis_row(text: str, gold: str, problem: str, cand_answers: Sequence | None = None,
                  cand_correct: Sequence | None = None, *, truncated: int = 0) -> dict:
    r"""M_DIS 한 롤아웃의 원재료 — 학습(math_meta._compute_dis_rows)과 held-out
    (scripts/local/math_dis_eval.py)이 **같은 파서**를 쓴다.

    ★최종 답은 <meta> **밖의** 마지막 \boxed 다(M_RETRY 의 split_attempts 와 같은 규약) —
      메타 안 박스가 답으로 읽히면 boxed_in_meta 가드가 무력해진다.
    ★`cand_correct` 는 **지표 전용**으로만 실린다 — 이 함수도, 크레딧도 그것으로 보상을
      만들지 않는다(gold 는 답 스팬에만 들어간다).
    """
    t = text or ""
    blocks = _mm._meta_block_spans(t)
    raw = t[blocks[0][0]:blocks[0][1]] if blocks else ""
    spans = _mm.boxed_spans(t, exclude=blocks)
    final_ans = spans[-1][0] if spans else None
    cands = [str(x or "") for x in (cand_answers or [])]
    ccorr = [int(bool(x)) for x in (cand_correct or [])]
    commit = parse_commit(raw)
    body = diag_text(raw)
    return {
        "text": t, "gold": str(gold), "problem": problem,
        # ★`meta` 를 직접 싣는 이유: countdown_rewards 의 자(emit_rate/boilerplate_rate/
        #   meta_position_stats)는 행에 `meta` 가 있으면 그것을 읽고, 없으면 parse_meta(form)
        #   로 다시 판다 — M_DIS 블록엔 `confidence:` 줄이 없어 form="math" 파싱은 emitted=0 을
        #   내므로, 심어 주지 않으면 발화율·정형문이 전부 «못 쟀다»로 빠진다.
        "meta": {"emitted": int(bool(blocks)), "form": "dis", "body": body, "raw": raw,
                 "start": blocks[0][0] if blocks else None,
                 "end": blocks[0][1] if blocks else None,
                 "confidence": None, "decision": None, "n_blocks": len(blocks)},
        "diag_text": body,
        "cand_answers": cands, "cand_correct": ccorr,
        "final_answer": final_ans,
        "r_corr": _mm.grade_math(f"\\boxed{{{final_ans}}}", gold) if final_ans else 0,
        "emitted": int(bool(blocks)),      # ★M_DIS 의 «발화» = 블록이 있다(confidence 줄은 안 쓴다)
        "has_meta": int(bool(blocks)),
        "n_blocks": len(blocks),
        "multi_block": int(len(blocks) > 1),
        "meta_start": blocks[0][0] if blocks else None,
        "meta_end": blocks[0][1] if blocks else None,
        "meta_raw": raw,
        "boxed_in_meta": int(any(_mm._BOXED_RE.search(t[a:b]) for a, b in blocks)),
        "commit": commit,
        "commit_answer": (cands[commit - 1] if (commit is not None and commit <= len(cands))
                          else None),
        "diag_words": diag_words(raw),
        "cites": sorted(cited_candidates(raw)),
        "n_chars": len(t),
        "truncated": int(_cdr._bool01(truncated)),
        # ★`confidence`/`decision`/`body` 는 두지 않는다 — M_DIS 블록엔 그 줄이 없고,
        #   있는 척하면 상위 텔레메트리(decision_rate 등)가 «못 쟀다»가 아니라 0 을 읽는다.
    }


# ── 진단 플래그 · 자기증류 크레딧 ───────────────────────────────────────────────
def dis_row_flags(row: Mapping) -> dict:
    r"""중단 규칙·텔레메트리가 읽는 진단 불리언. 행이 파싱 결과(parse_dis_row)든 원문만
    들고 있든 같은 답을 내도록, 없는 키는 `text` 에서 다시 뽑는다.

    has_meta         <meta>…</meta> 가 하나라도 있다
    commit_parsed    `Commit: n` 이 1..N_CAND 로 파싱됐다
    boxed_in_meta    메타 블록 안에 \boxed 가 있다(답 누출 — 항 미정의)
    multi_block      블록이 둘 이상(형식 위반 — 첫 블록만 채점되므로 나머지는 공짜 토큰)
    cites_two_plus   `Candidate <d>` 를 **서로 다른 번호로 둘 이상** 인용했다
                     ★«어느 후보가 갈리는지»를 짚었다는 형식 신호다. 한 번호만 인용하면
                       불일치를 읽은 것이 아니라 하나를 고른 것이다(보상 아님, 관찰용).
    commit_is_minority  커밋한 후보의 답이 **네 후보의 다수답과 다르다**(다수 미정이면 False)
                     ★«소수를 골랐는가» — 이 팔이 다수결 순응으로 붕괴했는지 보는 창이다.
    """
    if "commit" in row and "cand_answers" in row:
        r = row
    else:
        r = parse_dis_row(row.get("text", ""), row.get("gold", ""), row.get("problem", ""),
                          row.get("cand_answers"), row.get("cand_correct"))
    commit = r.get("commit")
    cands = [str(x or "") for x in (r.get("cand_answers") or [])]
    cand_major = plurality_answer(cands)
    ca = r.get("commit_answer")
    if ca is None and commit is not None and commit <= len(cands):
        ca = cands[commit - 1]
    minority = bool(commit is not None and cand_major is not None and ca is not None
                    and not _mm.answers_equivalent(ca, cand_major))
    cites = set(r.get("cites") or cited_candidates(r.get("meta_raw", "")))
    return {
        "has_meta": int(bool(r.get("has_meta", r.get("emitted", 0)))),
        "commit_parsed": int(commit is not None),
        "boxed_in_meta": int(bool(r.get("boxed_in_meta", 0))),
        "multi_block": int(bool(r.get("multi_block", int(r.get("n_blocks", 0)) > 1))),
        "cites_two_plus": int(len(cites) >= 2),
        "commit_is_minority": int(minority),
    }


def dis_row_credit(row: Mapping, group_rows: Sequence[Mapping],
                   self_idx: int | None = None) -> tuple[float, bool]:
    r"""자기 증류 판단 크레딧 → (크레딧, 정의됨).

    라벨 출처는 **그 GRPO 그룹의 LEAVE-ONE-OUT 다수답**이다: `plurality_answer` = 그룹 롤아웃
    중 **이 행 자신을 뺀** 나머지의 최종 \boxed 답을 수학 동치로 묶은 최대 군집(동률·무답이면
    미정). 크레딧은 «커밋한 후보의 저장된 답이 그 LOO 다수답과 같은가»의 ±1 이다.

    ★왜 leave-one-out 인가(0914 리뷰 D1): 이 행 자신의 커밋이 자기 자신의 최종 답을 정한다.
      전체 그룹(자기 포함)으로 다수답을 매기면 이 행이 **자기를 심판하는 다수결에 한 표를
      보탠다** — 8분의 1이지만, 형제끼리 진짜로 갈리는(동률인) 자리에서 자기 표가 캐스팅보트가
      되어 «자기 커밋과 같은 답»이 다수로 뽑히고, 그 다수와 같다며 크레딧 +1 을 받는 자기충족적
      순응 채널이 생긴다. 이 프로젝트의 다른 모든 자기증류 라벨(ledger c1.py 의 형제 답 점유율,
      M_RETRY_SL 의 leave-one-out p_retry)도 LOO 다. M_DIS 만 예외일 이유가 없다.
    ★행 식별은 **self_idx**(호출자가 주면 그 위치를 뺀다)를 우선하고, 없으면 **object identity**
      (`is`)로 `group_rows` 안에서 찾아 뺀다 — `==`(값 비교)로 찾으면 verl 행이 값이 우연히 같은
      딕셔너리(«가짜 쌍둥이»)일 때 엉뚱한 형제를 지운다. self_idx 도 identity 매치도 없으면(=
      호출자가 이미 자기 자신을 뺀 group_rows 를 준 것) 그대로 둔다 — 뺄 것이 없다.
    ★LOO 뒤 **답 있는 형제가 2 미만**이거나 LOO 다수결이 **동률**이면 크레딧은 미정의(0.0, False)
      다 — 형제 하나로는 «다수»를 논할 수 없다.
    ★gold 는 **한 번도 읽지 않는다** — row["gold"]/row["cand_correct"] 를 지우고 불러도 같은
      값이 나와야 한다(tests/test_math_dis.py 가 그렇게 호출한다). 메타 스팬이 gold 를 받으면
      «판단»이 아니라 «정답 복사»가 보상된다.
    ★미정의(0, False)는 «보상 0»이 아니라 «항이 없다»다 — 그룹 중심화의 member 에서 빠진다
      (math_meta._compute_dis_rows → verl_sdc._math_add_meta_region_advantage, 감사 3 과 같은 규약).
    """
    f = dis_row_flags(row)
    if not f["has_meta"] or f["multi_block"] or f["boxed_in_meta"] or not f["commit_parsed"]:
        return 0.0, False
    commit = row.get("commit")
    if commit is None:
        commit = parse_commit(row.get("meta_raw", ""))
    cands = [str(x or "") for x in (row.get("cand_answers") or [])]
    if commit is None or not (1 <= commit <= len(cands)):
        return 0.0, False
    if self_idx is not None:
        rest = [r for i, r in enumerate(group_rows) if i != self_idx]
    else:
        rest = [r for r in group_rows if r is not row]
    rest_answers = [r.get("final_answer") for r in rest]
    n_answered = sum(1 for a in rest_answers if str(a or "").strip())
    if n_answered < 2:
        return 0.0, False
    plur = plurality_answer(rest_answers)
    if plur is None:
        return 0.0, False
    return (1.0 if _mm.answers_equivalent(cands[commit - 1], plur) else -1.0), True
