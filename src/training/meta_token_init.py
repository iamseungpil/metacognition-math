"""Transplant the native think-token embeddings into the added meta tokens so the
added (zero-prior) <|meta|>/<|/meta|> inherit the strong open/close pairing prior of
<think>/</think> (spec 2026-06-15-s3b §3.1b). Call AFTER resize_token_embeddings."""
import torch


def transplant_meta_embeddings_from_think(model, tokenizer,
        pairs=(("<|meta|>", "<think>"), ("<|/meta|>", "</think>"))):
    def _id(t):
        i = tokenizer.convert_tokens_to_ids(t)
        if i is None or i < 0:
            raise ValueError(f"token {t!r} not in tokenizer")
        return i
    with torch.no_grad():
        for emb in {id(model.get_input_embeddings()): model.get_input_embeddings(),
                    id(model.get_output_embeddings()): model.get_output_embeddings()}.values():
            if emb is None:
                continue
            W = emb.weight
            for meta_t, think_t in pairs:
                W[_id(meta_t)] = W[_id(think_t)].clone()
    return model
