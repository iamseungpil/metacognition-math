"""cd6a0818 (Countdown 6팔) 감시기.

이 실험에서 확인해야 하는 것은 보통의 학습 감시와 다르다. 이번 판의 목적이 **배선**이라
가장 먼저 볼 것은 손실도 정확도도 아니고 `[COUNTDOWN][WIRED] arm=X` 한 줄이다.
지난 두 판이 전부 "선언은 있고 배선은 없는데 로그는 그럴듯한" 상태로 죽었다.

보는 순서:
  ① 생사      — amlt status (점검기 상태 표시를 믿지 않는다)
  ② ★배선     — 잡마다 WIRED 가 찍혔나, arm= 이 **서로 다른가**
  ③ 보상 분리 — 팔마다 mean 이 다른가 (같으면 배선이 또 무효)
  ④ 중단 조건 — 발화율<0.2 · 정형문>0.5 · 답누출>0.1
  ⑤ 진행      — step/150
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys

EXP = "cd6e0818"
AMLT = ("/tmp/claude-587327809/-home-v-seungplee/"
        "41a99d3b-d246-48cd-b893-68375dc4e059/scratchpad/amlt216/bin/amlt")
ARMS = {"cd6_corr": "A", "cd6_cur": "B", "cd6_mul": "C",
        "cd6_gate": "E", "cd6_full": "F", "cd6_neg": "G"}


def sh(cmd: str, timeout: int = 600) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout)
        return r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return "<timeout>"


def strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", s)


_NAN = float("nan")


def parse_telemetry(blob: str) -> dict:
    """텔레메트리 dict repr 을 판다.

    ⚠`ast.literal_eval` 을 그냥 쓰면 **터진다** — 딕트에 맨 `nan`/`inf` 가 들어 있고
    그건 리터럴이 아니라 이름이라 `ValueError` 가 난다. 초기 스텝의 선택성·정형문은
    거의 항상 nan 이므로, 이걸 안 고치면 텔레메트리가 **영원히 안 보인다**
    (조용히 빈 표 = "지표를 보고 있다"고 착각하는 바로 그 실패).
    """
    safe = re.sub(r"(?<![\w.])(-?)(nan|inf)(?![\w.])", r"None", blob)
    return ast.literal_eval(safe)


def num(v) -> float:
    """None(=원래 nan) 과 숫자를 하나로. 표에서 nan 은 '아직 없다'는 뜻이다."""
    try:
        return _NAN if v is None else float(v)
    except (TypeError, ValueError):
        return _NAN


def status() -> dict:
    out = strip_ansi(sh(f"{AMLT} status {EXP}"))
    st = {}
    for line in out.split("\n"):
        # 칸: # JOB NAME DURATION STATUS ... — DURATION 을 상태로 읽으면
        # 죽은 잡이 "8m" 으로 보인다(0814 무증상 행 오독과 같은 계급).
        m = re.match(r"^:\d+\s+:(\S+)\s+(\S+)\s+(\S+)", line.strip())
        if m:
            st[m.group(1)] = m.group(3)
    return st


def tail_log(job: str, n: int = 4000) -> str:
    """가장 최근 retry 의 로그 꼬리. 옛 retry 를 읽으면 죽은 런을 산 것으로 읽는다."""
    return strip_ansi(sh(f"{AMLT} log {EXP} :{job} -n {n}", timeout=420))


def main() -> None:
    st = status()
    print(f"{'잡':12s} {'상태':11s} {'팔':3s} {'배선':6s} {'보상평균':>9s} "
          f"{'구별':>5s} {'스텝':>8s}  비고")
    print("-" * 82)

    live = {k: v for k, v in st.items()
            if v.lower() in ("running", "preparing", "queued")}
    wired_arms, means = {}, {}
    telem, lens = {}, {}
    aborts = []

    for job, sv in sorted(st.items()):
        want = ARMS.get(job, "-")
        if sv.lower() in ("queued", "preparing"):
            print(f"{job:12s} {sv:11s} {want:3s} {'-':6s} {'-':>9s} {'-':>5s} "
                  f"{'-':>8s}  노드 대기")
            continue

        log = tail_log(job)
        # ② ★배선 증거. 이 줄이 없으면 나머지 수치는 볼 필요가 없다.
        w = re.findall(r"\[COUNTDOWN\]\[WIRED\] arm=(\w+) step=(\d+) n=(\d+) "
                       r"mean=([-\d.]+) distinct=(\d+)", log)
        if w:
            arm, step, n, mean, dist = w[-1]
            wired_arms[job] = arm
            means[job] = float(mean)
            ok = "✅" if arm == want else f"⛔{arm}"
            sm = re.findall(r"step:(\d+)/150|'global_step':\s*(\d+)", log)
            cur = sm[-1][0] or sm[-1][1] if sm else step
            print(f"{job:12s} {sv:11s} {want:3s} {ok:6s} {mean:>9s} "
                  f"{dist:>5s} {cur+'/150':>8s}")
        else:
            fatal = "FATAL WIRING" in log or "[YAML] FATAL" in log
            note = "⛔가드가 창을 죽임" if fatal else "배선 대기(부트스트랩 ~40분)"
            print(f"{job:12s} {sv:11s} {want:3s} {'⏳':6s} {'-':>9s} {'-':>5s} "
                  f"{'-':>8s}  {note}")

        for a in re.findall(r"\[COUNTDOWN\]\[ABORT\] step=\d+ (.+)", log)[-2:]:
            aborts.append(f"{job}: {a[:90]}")
        t = re.findall(r"\[COUNTDOWN\]\[TELEMETRY\] step=(\d+) (\{.+\})", log)
        if t:
            try:
                telem[job] = (int(t[-1][0]), parse_telemetry(t[-1][1]))
            except (ValueError, SyntaxError):
                pass
        # 길이 이상 = eos 리스트 [151645,151643] 를 verl 이 못 받는 경우의 첫 증상
        for ln in re.findall(r"response_length/mean:\s*([\d.]+)", log)[-1:]:
            lens[job] = float(ln)

    print()
    # ③ 팔 정체 + 보상 분리. 이 둘이 이번 판의 판정이다.
    if wired_arms:
        bad = {j: a for j, a in wired_arms.items() if a != ARMS.get(j)}
        print(f"[배선] {len(wired_arms)}/6 잡이 WIRED 를 찍음 · "
              f"팔 문자열 {'전부 일치 ✅' if not bad else f'⛔불일치 {bad}'}")
        uniq = len(set(wired_arms.values()))
        print(f"  서로 다른 arm= : {uniq}/{len(wired_arms)} "
              f"{'✅' if uniq == len(wired_arms) else '⛔같은 팔이 중복'}")
    if len(means) >= 2:
        vs = sorted(means.values())
        spread = vs[-1] - vs[0]
        print(f"[보상 분리] 팔별 평균 { {k: round(v, 3) for k, v in means.items()} }")
        print(f"  퍼짐 {spread:.4f} "
              f"{'✅ 팔이 실제로 갈린다' if spread > 1e-6 else '⛔전부 같다 — 배선 무효'}")
    # ④ 텔레메트리 15종. wandb 로 안 가고 stdout 에만 찍히므로 여기서 읽는다.
    if telem:
        print("\n[텔레메트리]  ★중단 문턱: 발화<0.20 · 정형문>0.50 · 답누출>0.10")
        hdr = (f"  {'잡':11s} {'스텝':>5s} {'발화':>6s} {'정형문':>7s} {'답누출':>7s} "
               f"{'메타먼저':>8s} {'선택성':>7s} {'p̂평균':>7s} {'정답':>6s} "
               f"{'길이p95':>8s} {'redirect':>9s} {'conf종수':>8s} {'save/derail':>12s}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for job, (stp, r) in sorted(telem.items()):
            def g(d, k):
                v = r.get(d, {})
                return num(v.get(k)) if isinstance(v, dict) else _NAN
            sh_ = r.get("shift", {}) or {}
            print(f"  {job:11s} {stp:>5d} {num(r.get('emit_rate')):>6.3f} "
                  f"{g('boilerplate', 'boilerplate_rate'):>7.3f} "
                  f"{num(r.get('answer_leak_rate')):>7.3f} "
                  f"{g('meta_position', 'frac_meta_first'):>8.3f} "
                  f"{g('selectivity', 'selectivity'):>7.3f} "
                  f"{g('phat', 'mean'):>7.3f} {num(r.get('acc')):>6.3f} "
                  f"{g('length', 'len_p95'):>8.1f} "
                  f"{g('decision', 'redirect'):>9.3f} "
                  f"{g('confidence', 'n_unique'):>8.1f} "
                  f"{str(sh_.get('n_save', '-')) + '/' + str(sh_.get('n_derail', '-')):>12s}")
        print("  ⓘ 정형문 nan = 아직 메타를 낸 행이 없다 · 선택성 nan = 난이도 층이 안 갈렸다")
    if lens:
        print(f"\n[응답 길이] {lens}  ⚠cap 3072 에 붙으면 eos 리스트 문제를 의심")
    if aborts:
        print("\n[⛔중단 조건 발동]")
        for a in aborts:
            print(f"  {a}")
    print(f"\n살아있는 잡 {len(live)}/7 · 상태 {dict(sorted(st.items()))}")


if __name__ == "__main__":
    main()
