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

def test_anchor_rescales_format_and_emit():
    B, T = 4, 6
    idx = [0, 0, 1, 1]
    ans = torch.zeros(B, T); ans[:, :2] = 1.0
    meta = torch.zeros(B, T); meta[:, 2:4] = 1.0
    fv = torch.zeros(B, T); fv[:, 4] = 1.0
    state = {}
    A, _ = compose_dcpo_region_advantage(
        response_mask=torch.ones(B, T), index=idx,
        R_corr=[1.0,-1.0,1.0,-1.0], R_meta=[0.0]*B, R_cal=[0.0]*B,
        answer_mask=ans, meta_content_mask=meta, conf_mask=torch.zeros(B,T),
        R_format=[0.02,-0.02,0.02,-0.02], format_violation_mask=fv, w_format=1.0,
        w_corr=1.0, w_meta=0.0, w_cal=0.0,
        anchor_norm=True, anchor_ema_state=state, anchor_ema_decay=0.0, anchor_warmup_steps=0,
    )
    corr_mag = A[:, 0].abs().mean().item()
    fmt_mag = A[:, 4].abs().mean().item()
    assert abs(fmt_mag - corr_mag) / corr_mag < 0.05, (fmt_mag, corr_mag)

def test_emit_first_token_routing_clean():
    # Silent row (R_emit 0) must get NEGATIVE emit advantage on token 0 only;
    # answer/meta tokens must be unchanged vs no-emit baseline.
    B, T = 2, 6
    idx = [0, 0]
    ans = torch.zeros(B, T); ans[:, 1:3] = 1.0   # answer tokens 1,2 (NOT token 0)
    meta = torch.zeros(B, T); meta[:, 3:5] = 1.0
    common = dict(
        response_mask=torch.ones(B, T), index=idx,
        R_corr=[1.0, -1.0], R_meta=[0.0]*B, R_cal=[0.0]*B,
        answer_mask=ans, meta_content_mask=meta, conf_mask=torch.zeros(B,T),
        w_corr=1.0, w_meta=0.0, w_cal=0.0,
    )
    base, _ = compose_dcpo_region_advantage(**common)
    routed, _ = compose_dcpo_region_advantage(
        **common, R_emit=[1.0, 0.0], w_emit=0.5,
        emit_route="first_token", emit_first_n=1,
    )
    # token 0 differs (emit landed there); answer tokens 1,2 identical to baseline.
    assert not torch.allclose(routed[:, 0], base[:, 0])
    assert torch.allclose(routed[:, 1:3], base[:, 1:3])
    # silent row (row 1) gets negative emit on token 0
    assert routed[1, 0] < 0

def test_meta_len_cap_limits_floor():
    # A row with many meta tokens: with a cap, only the first `cap` meta tokens
    # share the +floor (row total <= floor*cap/row_n stays bounded); without cap
    # the whole meta span shares it. Assert capped floor total < uncapped.
    B, T = 1, 10
    meta = torch.zeros(B, T); meta[:, 2:9] = 1.0   # 7 meta tokens
    args = dict(
        response_mask=torch.ones(B, T), index=[0],
        R_corr=[1.0], R_meta=[0.0], R_cal=[0.0],
        answer_mask=torch.zeros(B,T), meta_content_mask=meta, conf_mask=torch.zeros(B,T),
        w_corr=0.0, w_meta=0.0, w_cal=0.0,
        meta_floor=0.1, floor_mask=[1.0],
    )
    uncapped, _ = compose_dcpo_region_advantage(**args)
    capped, _ = compose_dcpo_region_advantage(**args, meta_len_cap=3)
    assert capped[:, 2:9].sum().item() < uncapped[:, 2:9].sum().item()
    # capped applies floor to only the first 3 meta tokens
    assert capped[0, 5:9].abs().sum().item() == 0.0
