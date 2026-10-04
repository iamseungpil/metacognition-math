"""ref 워커 + `mc_tree_score` — **모듈 최상위** 클래스라 Ray cloudpickle 이 참조로 싣는다(함수 안 클래스는
값으로 실려 verl `register` 의 전역(transfer_queue)까지 끌고 가다 즉사, 0924). 워커는 PYTHONPATH 로 mc 를 읽는다."""
from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.engine_workers import ActorRolloutRefWorker


class MCRefWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def mc_tree_score(self, trees, per_token=False):
        """trees = [(접두 ids, [(t, 토큰열, m)])] → 이어 붙인 블록 점수(`mc.shift_check.score_trees` — 검증과
        같은 채점기, TREE_MAX 나누기·OOM 반분). 장치 = 현재 cuda(매개변수 오프로드면 매개변수는 cpu)."""
        import torch

        from mc.shift_check import score_trees
        eng = self.ref.engine
        with eng.eval_mode(), torch.no_grad():
            return score_trees(eng.module, trees, dev=torch.device("cuda", torch.cuda.current_device()), per_token=per_token)
