import torch
from src.training.dcpo_region import compose_dcpo_region_advantage

def _masks(B, T):
    ans = torch.zeros(B, T); ans[:, :2] = 1.0          # answer tokens 0,1
    meta = torch.zeros(B, T); meta[:, 2:4] = 1.0        # meta tokens 2,3
    conf = torch.zeros(B, T)
    rm = torch.ones(B, T)
    return ans, meta, conf, rm

def test_anchor_rescales_aux_to_corr_scale():
    # R_corr scale ~1 (±1), R_meta scale ~0.05 — anchor should lift meta to corr scale.
    B, T = 4, 6
    idx = [0, 0, 1, 1]
    R_corr = [1.0, -1.0, 1.0, -1.0]
    R_meta = [0.05, -0.05, 0.05, -0.05]
    ans, meta, conf, rm = _masks(B, T)
    state = {}
    # warmup=0 -> anchor active immediately; ema=0.0 -> EMA == current batch stats
    A, _ = compose_dcpo_region_advantage(
        response_mask=rm, index=idx, R_corr=R_corr, R_meta=R_meta, R_cal=[0.0]*B,
        answer_mask=ans, meta_content_mask=meta, conf_mask=conf,
        w_corr=1.0, w_meta=1.0, w_cal=0.0,
        anchor_norm=True, anchor_ema_state=state, anchor_ema_decay=0.0,
        anchor_warmup_steps=0,
    )
    # meta-token advantage magnitude should now be ~ corr-token magnitude (within 5%),
    # NOT ~0.05x of it.
    corr_mag = A[:, 0].abs().mean().item()
    meta_mag = A[:, 2].abs().mean().item()
    assert abs(meta_mag - corr_mag) / corr_mag < 0.05, (meta_mag, corr_mag)

def test_anchor_off_is_byte_identical():
    B, T = 4, 6
    idx = [0, 0, 1, 1]
    args = dict(
        response_mask=torch.ones(B, T), index=idx,
        R_corr=[1.0,-1.0,1.0,-1.0], R_meta=[0.05,-0.05,0.05,-0.05], R_cal=[0.0]*B,
        answer_mask=_masks(B,T)[0], meta_content_mask=_masks(B,T)[1],
        conf_mask=_masks(B,T)[2], w_corr=1.0, w_meta=1.0, w_cal=0.0,
    )
    base, _ = compose_dcpo_region_advantage(**args)
    off, _ = compose_dcpo_region_advantage(**args, anchor_norm=False)
    assert torch.allclose(base, off)
