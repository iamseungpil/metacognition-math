"""대조학습 가능성 진단: 자연 발생 롤아웃에서 (chk_solved, over_claim/중립) 쌍이
얼마나 자주 같은 그룹(같은 문제 8롤아웃)에 같이 나오는지 센다.
CTSD 교훈: 자연 발생 신호는 변별력(=쌍 형성 가능성)이 없으면 대조학습이 굶는다.
GPU 없음, 기존 eval 로그만 읽는다."""
import json, sys, collections
sys.path.insert(0, '/home/ubuntu/seungpil/metacognition-math')
from src.training.countdown_rewards import check_row

def analyze(path, label):
    groups = collections.defaultdict(list)
    n_rows = 0
    for line in open(path):
        d = json.loads(line)
        n_rows += 1
        cr = check_row(d['text'], d['nums'], d['target'], d['r_corr'])
        groups[d['group_id']].append(cr)
    n_groups = len(groups)
    n_solved = sum(sum(1 for r in rs if r['chk_solved']) for rs in groups.values())
    n_overclaim = sum(sum(1 for r in rs if r['over_claim']) for rs in groups.values())
    n_falsealarm = sum(sum(1 for r in rs if r['false_alarm']) for rs in groups.values())
    n_fclaim = sum(sum(1 for r in rs if r['fclaim']) for rs in groups.values())
    # 같은 그룹 안에 solved 1개 이상 AND (over_claim 또는 미해결) 1개 이상 -> 대조 쌍 형성 가능
    pairable = 0
    for rs in groups.values():
        has_solved = any(r['chk_solved'] for r in rs)
        has_neg = any(r['over_claim'] or (not r['chk_solved'] and r['chk_fixed']==0) for r in rs)
        if has_solved and has_neg:
            pairable += 1
    print(f"[{label}] rows={n_rows} groups={n_groups} chk_solved={n_solved}({n_solved/n_rows:.3%}) "
          f"over_claim={n_overclaim}({n_overclaim/n_rows:.3%}) false_alarm={n_falsealarm} fclaim={n_fclaim} "
          f"pairable_groups={pairable}/{n_groups}({pairable/n_groups:.1%})")
    return dict(rows=n_rows, groups=n_groups, chk_solved=n_solved, over_claim=n_overclaim,
                pairable_groups=pairable, n_groups=n_groups)

if __name__ == '__main__':
    paths = [
        ('/hdd_data/seungpil/scratch/eval/cd7_TAG0_chk_s1/step_30/texts.jsonl', 'TAG0 s30'),
        ('/hdd_data/seungpil/scratch/eval/cd7_TAG0_chk_s1/step_50/texts.jsonl', 'TAG0 s50'),
        ('/hdd_data/seungpil/scratch/eval/cd7_TAG0_chk_s1/step_100/texts.jsonl', 'TAG0 s100'),
        ('/hdd_data/seungpil/scratch/eval/cd7_EVCM_CHK_chk_s1/step_30/texts.jsonl', 'EVCM s30'),
        ('/hdd_data/seungpil/scratch/eval/cd7_EVCM_CHK_chk_s1/step_50/texts.jsonl', 'EVCM s50'),
        ('/hdd_data/seungpil/scratch/eval/cd7_EVCM_CHK_chk_s1/step_100/texts.jsonl', 'EVCM s100'),
    ]
    for p, l in paths:
        try:
            analyze(p, l)
        except FileNotFoundError:
            print(f"[{l}] missing: {p}")
