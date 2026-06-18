"""vLLM logits_processor that hard-bans token ids by setting their logit to
-inf at every position. A TRUE hard ban (unlike vLLM's additive/clamped
logit_bias, which a primed model can override when the banned token leads by
> |bias|). Stateless / picklable for Ray workers (mirrors meta_close_processor).

Used by the counterfactual ablation (Stage-C reward `c_without` and the R-B'
eval) to guarantee the `<|switch|>` redirect-decision token never appears in the
suppressed arm — spec 2026-06-18-redirect-priming-counterfactual REV-6 §7,
review round-5 I-2/I-6.
"""


class SwitchBanLogitsProcessor:
    def __init__(self, ban_ids):
        self.ban_ids = [int(i) for i in ban_ids]

    def __call__(self, token_ids, logits):
        for i in self.ban_ids:
            logits[i] = float("-inf")
        return logits
