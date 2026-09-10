"""§13/§14 정직 벌 세금이 스텝에 비례해 커지는 원인 진단.
텔레메트리(응답 길이·잘림)로는 설명이 안 된다는 게 이미 확인됨 — 이 스크립트는
check_row 라벨(fclaim·chk_solved·over_claim·검산 개수·위치)로 더 깊이 판다.
GPU 없음, 기존 eval texts.jsonl 만 읽는다."""
import json, sys, re, argparse
sys.path.insert(0, '/home/ubuntu/seungpil/metacognition-math')
from src.training.countdown_rewards import check_row, parse_checks, _CHECK_RE


def analyze(path):
    n = n_fclaim = n_solved = n_overclaim = n_checks_total = n_rows_with_check = 0
    pos_sum = pos_n = 0
    corr_with_check = corr_n_with = corr_without_check = corr_n_without = 0
    for line in open(path):
        d = json.loads(line)
        n += 1
        text = d['text']
        cr = check_row(text, d['nums'], d['target'], d['r_corr'])
        n_fclaim += cr['fclaim']
        n_solved += cr['chk_solved']
        n_overclaim += cr['over_claim']
        checks = parse_checks(text)
        n_checks_total += len(checks)
        if checks:
            n_rows_with_check += 1
            m = list(_CHECK_RE.finditer(text))[-1]
            pos_sum += m.start() / max(len(text), 1)
            pos_n += 1
            corr_with_check += d['r_corr']; corr_n_with += 1
        else:
            corr_without_check += d['r_corr']; corr_n_without += 1
    return dict(
        n=n, fclaim_rate=n_fclaim / n, chk_solved_rate=n_solved / n, over_claim_rate=n_overclaim / n,
        rows_with_check_rate=n_rows_with_check / n, checks_per_row=n_checks_total / n,
        mean_check_pos_frac=(pos_sum / pos_n) if pos_n else float('nan'),
        acc_with_check=(corr_with_check / corr_n_with) if corr_n_with else float('nan'),
        acc_without_check=(corr_without_check / corr_n_without) if corr_n_without else float('nan'),
        n_with_check=corr_n_with, n_without_check=corr_n_without,
    )


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', nargs='+', default=['TAG0', 'EVCM_CHK'])
    ap.add_argument('--steps', nargs='+', type=int, default=[30, 50, 100])
    ap.add_argument('--eval_root', default='/hdd_data/seungpil/scratch/eval')
    args = ap.parse_args()
    for arm in args.arms:
        for step in args.steps:
            p = f"{args.eval_root}/cd7_{arm}_chk_s1/step_{step}/texts.jsonl"
            try:
                r = analyze(p)
            except FileNotFoundError:
                print(f"[{arm} s{step}] missing: {p}")
                continue
            print(f"[{arm} s{step}] fclaim={r['fclaim_rate']:.3%} chk_solved={r['chk_solved_rate']:.3%} "
                  f"over_claim={r['over_claim_rate']:.3%} rows_w_check={r['rows_with_check_rate']:.3%} "
                  f"checks/row={r['checks_per_row']:.2f} mean_pos={r['mean_check_pos_frac']:.3f} "
                  f"acc|check={r['acc_with_check']:.3f}(n={r['n_with_check']}) "
                  f"acc|no_check={r['acc_without_check']:.3f}(n={r['n_without_check']})")
