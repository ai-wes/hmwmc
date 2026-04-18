"""Focused smoke test for hpm_v2 and the explicit state stack."""
import torch
from hpm_v2 import (
    HPMConfig,
    HomeostaticPredictiveMemory,
    EntityTable,
    EntityTableConfig,
    StructuredStateConfig,
    StructuredStateTable,
    TypedEventLogConfig,
    TypedEventLog,
    StateCheckpointConfig,
    StateCheckpointBank,
)

x = torch.randn(2, 10, 64)

# Test 1: Default config — backward compatibility
cfg = HPMConfig(enabled=True)
hpm = HomeostaticPredictiveMemory(d_model=64, cfg=cfg)
out, diag = hpm(x)
print(f"Test 1 (default): out={out.shape}, diag_keys={sorted(diag.keys())}")
assert out.shape == (2, 10, 4 * 64), f"Expected (2,10,256), got {out.shape}"

# Test 2: Top-k routing
cfg2 = HPMConfig(enabled=True, topk_routing=True, topk_k=2)
hpm2 = HomeostaticPredictiveMemory(d_model=64, cfg=cfg2)
out2, diag2 = hpm2(x)
print(f"Test 2 (topk): out={out2.shape}, has_load_std={'hpm_load_std' in diag2}")
assert "hpm_load_std" in diag2
assert "hpm_age_mean" in diag2

# Test 3: Factorized surprise
cfg3 = HPMConfig(enabled=True, factorized_surprise=True, n_surprise_channels=3)
hpm3 = HomeostaticPredictiveMemory(d_model=64, cfg=cfg3)
out3, diag3 = hpm3(x)
print(f"Test 3 (factorized): out={out3.shape}")

# Test 4: Per-state stats
cfg4 = HPMConfig(enabled=True, per_state_stats=True)
hpm4 = HomeostaticPredictiveMemory(d_model=64, cfg=cfg4)
hpm4.train()
out4, diag4 = hpm4(x)
print(f"Test 4 (per_state_stats): mu_shape={hpm4.mu.shape}")
assert hpm4.mu.shape == (3, 4), f"Expected (3,4), got {hpm4.mu.shape}"

# Test 5: Diversity regularizer
cfg5 = HPMConfig(enabled=True, content_gating=True, slot_diversity_lambda=0.01)
hpm5 = HomeostaticPredictiveMemory(d_model=64, cfg=cfg5)
out5, diag5 = hpm5(x)
has_div = "_diversity_loss" in diag5
print(f"Test 5 (diversity): has_loss={has_div}, slot_key_cos={diag5.get('hpm_slot_key_cos', 'N/A'):.4f}")
assert has_div, "Expected _diversity_loss in diag"

# Test 6: All v2 features together
cfg6 = HPMConfig(
    enabled=True, topk_routing=True, topk_k=2,
    factorized_surprise=True, n_surprise_channels=3,
    per_state_stats=True, slot_diversity_lambda=0.01,
)
hpm6 = HomeostaticPredictiveMemory(d_model=64, cfg=cfg6)
hpm6.train()
out6, diag6 = hpm6(x)
div_loss = diag6.pop("_diversity_loss", None)
print(f"Test 6 (all v2): out={out6.shape}, div_loss_is_tensor={isinstance(div_loss, torch.Tensor)}")
assert div_loss is not None and isinstance(div_loss, torch.Tensor)
# Verify diversity loss has gradients
assert div_loss.requires_grad, "diversity_loss should be differentiable"

# Test 7: EntityTable
et_cfg = EntityTableConfig(enabled=True, n_entities=4, d_entity=32)
et = EntityTable(d_model=64, cfg=et_cfg)
et_out, et_diag = et(x)
print(f"Test 7 (entity): out={et_out.shape}, output_dim={et.output_dim}, diag={et_diag}")
assert et_out.shape == (2, 10, 4 * 32), f"Expected (2,10,128), got {et_out.shape}"

# Test 7b: EntityTable attn read mode
et_cfg_attn = EntityTableConfig(enabled=True, n_entities=4, d_entity=32, read_mode="attn")
et_attn = EntityTable(d_model=64, cfg=et_cfg_attn)
et_attn_out, _ = et_attn(x)
print(f"Test 7b (entity attn): out={et_attn_out.shape}, output_dim={et_attn.output_dim}")
assert et_attn_out.shape == (2, 10, 32)

# Test 8: T=0 edge case
empty = torch.randn(2, 0, 64)
out_empty, _ = hpm(empty)
print(f"Test 8 (T=0): out={out_empty.shape}")
assert out_empty.shape == (2, 0, 4 * 64)

# Test 9: Gradient flow through HPM v2
cfg9 = HPMConfig(
    enabled=True, topk_routing=True, topk_k=2,
    factorized_surprise=True, n_surprise_channels=3,
    slot_diversity_lambda=0.01,
)
hpm9 = HomeostaticPredictiveMemory(d_model=64, cfg=cfg9)
hpm9.train()
x9 = torch.randn(2, 5, 64, requires_grad=True)
out9, diag9 = hpm9(x9)
loss = out9.sum() + diag9.pop("_diversity_loss", torch.tensor(0.0))
loss.backward()
print(f"Test 9 (grad flow): input grad norm={x9.grad.norm():.4f}")
assert x9.grad is not None and x9.grad.norm() > 0

# Test 10: reset_running_stats with topk buffers
hpm6.reset_running_stats()
print(f"Test 10 (reset): sticky={hpm6._sticky.sum():.1f}, load={hpm6._load.sum():.1f}")
assert hpm6._sticky.sum() == 0 and hpm6._load.sum() == 0

# Test 11: Structured state stack
ss = StructuredStateTable(d_model=64, cfg=StructuredStateConfig(enabled=True, n_entities=4, d_state=32))
structured = ss(x)
print(f"Test 11 (structured state): seq={structured.sequence.shape}, mem={structured.memory_tokens.shape}")
assert structured.sequence.shape == (2, 10, 64)
assert structured.memory_tokens.shape == (2, 10, 4, 64)

# Test 12: Typed event log + checkpoints
typed = TypedEventLog(d_model=64, n_entities=4, cfg=TypedEventLogConfig(enabled=True, max_events=6))
typed_out = typed(x, structured, z_per_step=torch.randn(2, 10, 2))
checkpoints = StateCheckpointBank(d_model=64, n_entities=4, cfg=StateCheckpointConfig(enabled=True, n_checkpoints=4))
checkpoint_out = checkpoints(structured, typed_out.event_scores)
print(f"Test 12 (typed events/checkpoints): events={typed_out.entries.shape}, checkpoints={checkpoint_out.entries.shape}")
assert typed_out.entries.shape == (2, 6, 64)
assert checkpoint_out.entries.shape == (2, 4, 64)

print("\nALL 12 TESTS PASSED")
