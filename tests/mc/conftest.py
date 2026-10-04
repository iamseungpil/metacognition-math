"""tests/mc 는 GPU 를 쓰지 않는다 — GPU 0 은 다른 사용자 것(운용 규칙, 0924). CUDA 는 첫 사용 때 이 env 를 읽으므로
어떤 테스트도 암묵적으로 GPU 를 잡지 못한다."""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _k in [k for k in os.environ if k in ("PFX_FORK", "PFX_REP", "PFX_BREAK_W", "PFX_STOP", "PFX_STOP_W", "PFX_TAIL0", "PFX_CHECK_COST", "PFX_CHECK_W", "PFX_RIGHT_SHARE", "W_PMI2") or k.startswith("PMI2_")]:
    os.environ.pop(_k)                 # 셸에 남은 실험 손잡이가 테스트를 오염시키지 않게
