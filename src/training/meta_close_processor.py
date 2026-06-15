"""vLLM logits_processor that bounds the meta block: after <|meta|>, forbid a 2nd
open, and once `max_open_tokens` have passed without a close, force <|/meta|>.
Stateless across the (token_ids, logits) call contract (reconstructs state from
token_ids each step) so it is picklable for Ray workers (spec §3.2 best-effort)."""
import torch


class MetaCloseLogitsProcessor:
    def __init__(self, meta_open: int, meta_close: int, max_open_tokens: int = 96):
        self.o = meta_open; self.c = meta_close; self.maxn = max_open_tokens

    def __call__(self, token_ids, logits):
        # find last unmatched open
        depth = 0; since = None
        for k, t in enumerate(token_ids):
            if t == self.o:
                depth += 1; since = 0
            elif t == self.c and depth > 0:
                depth -= 1; since = None
            elif since is not None:
                since += 1
        if depth <= 0 or since is None:
            return logits
        if since >= self.maxn:                       # force close
            mask = torch.full_like(logits, float("-inf")); mask[self.c] = logits[self.c]
            return mask
        logits[self.o] = float("-inf")               # within budget: forbid 2nd open
        return logits
