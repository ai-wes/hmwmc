"""Quick functional test for all HPM architecture changes."""
import torch
import torch.nn.functional as F
from hpm import HPMConfig, HomeostaticPredictiveMemory

h = torch.randn(2, 8, 256)

# 1. Default config (backward compat)
cfg = HPMConfig()
hpm = HomeostaticPredictiveMemory(d_model=256, cfg=cfg)
hpm.train()
out, diag = hpm(h)
print(f"Default: out={out.shape}, keys={sorted(diag.keys())}")
assert "hpm_write_regular_frac" in diag
assert "hpm_write_forced_frac" in diag
assert out.shape == (2, 8, 4 * 256)  # n_slots=4, slot_dim=256, concat

# 2. Surprise subspace
cfg2 = HPMConfig(surprise_dim=64, sigma_floor_adaptive=True, min_surprise_threshold=0.01)
hpm2 = HomeostaticPredictiveMemory(d_model=256, cfg=cfg2)
hpm2.train()
out2, diag2 = hpm2(h)
print(f"Surprise: out={out2.shape}, sigma={diag2['hpm_sigma']:.4f}")
assert hasattr(hpm2, "surprise_proj")

# 3. Content gating
cfg3 = HPMConfig(content_gating=True)
hpm3 = HomeostaticPredictiveMemory(d_model=256, cfg=cfg3)
hpm3.train()
out3, diag3 = hpm3(h)
print(f"Content: out={out3.shape}")
assert hasattr(hpm3, "slot_key")
assert hasattr(hpm3, "content_query")

# 4. Learnable gains
cfg4 = HPMConfig(learnable_gains=True)
hpm4 = HomeostaticPredictiveMemory(d_model=256, cfg=cfg4)
hpm4.train()
out4, diag4 = hpm4(h)
gains = F.softplus(hpm4.gain_logits).tolist()
print(f"Gains: out={out4.shape}, gains={[f'{g:.3f}' for g in gains]}")
assert abs(gains[0] - 1.0) < 0.01  # gain_open init
assert abs(gains[1] - 0.5) < 0.01  # gain_closing init
assert abs(gains[2] - 0.1) < 0.01  # gain_locked init

# 5. Retroactive window (C3)
cfg5 = HPMConfig(retroactive_window=4)
hpm5 = HomeostaticPredictiveMemory(d_model=256, cfg=cfg5)
hpm5.train()
out5, diag5 = hpm5(h)
print(f"Retro: out={out5.shape}")
assert hasattr(hpm5, "retro_mix")

# 6. Multi-timescale (C4)
cfg6 = HPMConfig(slot_timescales=(1.0, 2.0, 4.0, 8.0))
hpm6 = HomeostaticPredictiveMemory(d_model=256, cfg=cfg6)
hpm6.train()
out6, diag6 = hpm6(h)
print(f"Timescale: out={out6.shape}")

# 7. All features combined
cfg7 = HPMConfig(
    surprise_dim=64,
    sigma_floor_adaptive=True,
    min_surprise_threshold=0.01,
    content_gating=True,
    learnable_gains=True,
    retroactive_window=4,
    slot_timescales=(1.0, 2.0, 4.0, 8.0),
)
hpm7 = HomeostaticPredictiveMemory(d_model=256, cfg=cfg7)
hpm7.train()
out7, diag7 = hpm7(h)
print(f"Combined: out={out7.shape}")

# 8. Gradient flow check
cfg8 = HPMConfig(surprise_dim=64, content_gating=True, learnable_gains=True)
hpm8 = HomeostaticPredictiveMemory(d_model=256, cfg=cfg8)
hpm8.train()
out8, _ = hpm8(h)
loss = out8.sum()
loss.backward()
# surprise_proj feeds into detached error stats (by design), so no grad through it.
# content_gating (slot_key, content_query) and learnable gains flow through the gate.
assert hpm8.slot_key.grad is not None, "slot_key should have grad"
assert hpm8.gain_logits.grad is not None, "gain_logits should have grad"
# Write encoder always has grad (core write path).
assert hpm8.write_encoder.weight.grad is not None, "write_encoder should have grad"
print("Gradient flow: OK")

print("\nALL TESTS PASSED")
