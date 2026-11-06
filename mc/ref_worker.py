"""ref 워커 + `mc_tree_score` — **모듈 최상위** 클래스라 Ray cloudpickle 이 참조로 싣는다(함수 안 클래스는
값으로 실려 verl `register` 의 전역(transfer_queue)까지 끌고 가다 즉사, 0924). 워커는 PYTHONPATH 로 mc 를 읽는다."""
from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.engine_workers import ActorRolloutRefWorker


def _chunked_entropy(logits):
    """수정 75: verl 비-rmpad 경로는 엔트로피를 전체 [T, 어휘] 로 한 번에 계산(softmax·곱 임시 ≈ 2 × 16 GiB, 26.7k 말에서 OOM).
    같은 값을 2,048 행씩 계산하는 verl 내장 함수로 바꾼다(모든 팔 공통, 값 동일)."""
    import verl.utils.torch_functional as vf
    return vf.entropy_from_logits_with_chunking(logits.reshape(-1, logits.shape[-1])).reshape(logits.shape[:-1])


def patch_entropy():
    import verl.utils.torch_functional as vf
    vf.entropy_from_logits = _chunked_entropy


patch_entropy()   # 워커가 이 모듈을 실을 때(두 역할 모두 MCRefWorker) 적용


class MCRefWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def mc_tree_score(self, trees, per_token=False, policy=False):
        """trees = [(접두 ids, [(t, 토큰열, m)])] → 이어 붙인 블록 점수(`mc.shift_check.score_trees` — 검증과
        같은 채점기, TREE_MAX 나누기·OOM 반분). 장치 = 현재 cuda(매개변수 오프로드면 매개변수는 cpu).
        policy = 수정 74: 얼린 ref 대신 이 워커의 **학습 중 actor**(advantage 계산 때 = 이번 스텝 π_old 가중치)."""
        import torch

        from mc.shift_check import score_trees
        eng = self.actor.engine if policy else self.ref.engine
        with eng.eval_mode(), torch.no_grad():
            return score_trees(eng.module, trees, dev=torch.device("cuda", torch.cuda.current_device()), per_token=per_token)
