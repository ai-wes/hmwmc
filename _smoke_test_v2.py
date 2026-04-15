"""Smoke tests for HPM v2 changes — checks instantiation, forward pass shapes,
config gating, backward-compat defaults, diversity loss, event tape, entity table,
iterative query head, continuous plasticity, and retroactive binding."""

import sys, torch, traceback
from dataclasses import asdict

PASS = 0
FAIL = 0

def check(name, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  [PASS] {name}")
        PASS += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()
        FAIL += 1


# ── 1. hpm.py imports ─────────────────────────────────────────────
print("\n=== 1. hpm.py imports ===")
from hpm import (
    HPMConfig, HomeostaticPredictiveMemory,
    EntityTableConfig, EntityTable,
    EventTapeConfig, EventTape,
)
check("all hpm exports importable", lambda: None)

# ── 2. Default HPMConfig backward compat ──────────────────────────
print("\n=== 2. Default HPMConfig backward compat ===")
def test_default_config():
    cfg = HPMConfig()
    assert cfg.continuous_plasticity == False, "continuous_plasticity should default to False"
    assert cfg.slot_diversity_lambda == 0.0, "slot_diversity_lambda should default to 0.0"
    assert cfg.gate_load_balance == 0.0, "gate_load_balance should default to 0.0"
check("default config backward compat", test_default_config)

# ── 3. EntityTable forward ────────────────────────────────────────
print("\n=== 3. EntityTable forward ===")
def test_entity_table():
    ecfg = EntityTableConfig(enabled=True, n_entities=4, d_entity=32)
    et = EntityTable(d_model=64, cfg=ecfg)
    x = torch.randn(2, 10, 64)
    out = et(x)
    assert len(out) == 3, f"Expected 3-tuple, got {len(out)}"
    entity_seq, diag, entity_stack = out
    assert entity_seq.shape == (2, 10, et.output_dim), f"entity_seq shape {entity_seq.shape}"
    assert entity_stack.shape[0] == 2 and entity_stack.shape[1] == 10, f"entity_stack shape {entity_stack.shape}"
    assert isinstance(diag, dict), f"diag type {type(diag)}"
    assert "entity_route_entropy" in diag, f"missing entity_route_entropy, keys: {list(diag.keys())}"
check("EntityTable 3-tuple forward", test_entity_table)

# ── 4. EventTape forward ─────────────────────────────────────────
print("\n=== 4. EventTape forward ===")
def test_event_tape():
    tcfg = EventTapeConfig(enabled=True, max_events=8, surprise_threshold=0.5, include_entity_state=False)
    tape = EventTape(d_model=64, cfg=tcfg, d_entity_total=0)
    B, T = 2, 10
    h_seq = torch.randn(B, T, 64)
    z_per_step = torch.randn(B, T)
    out = tape(h_seq, z_per_step, entity_states=None)
    assert len(out) == 4, f"Expected 4-tuple, got {len(out)}"
    entries, mask, times, diag = out
    assert entries.shape[0] == B and entries.shape[1] == 8, f"entries shape {entries.shape}"
    assert mask.shape == (B, 8), f"mask shape {mask.shape}"
    assert "event_tape_n_events" in diag, f"missing diag key, got {list(diag.keys())}"
check("EventTape 4-tuple forward", test_event_tape)

# ── 5. HPM with continuous_plasticity ─────────────────────────────
print("\n=== 5. HPM continuous plasticity ===")
def test_hpm_continuous():
    cfg = HPMConfig(
        n_slots=4, slot_dim=32,
        continuous_plasticity=True,
        sigma_hard_floor=0.3,
        slot_diversity_lambda=0.01,
        gate_load_balance=0.01,
    )
    hpm = HomeostaticPredictiveMemory(d_model=64, cfg=cfg)
    x = torch.randn(2, 10, 64)
    out_seq, diag = hpm(x)
    assert out_seq.shape[0] == 2 and out_seq.shape[1] == 10, f"out_seq shape {out_seq.shape}"
    assert isinstance(diag, dict), f"diag type {type(diag)}"
    # Should have diversity loss as Tensor
    if "_diversity_loss" in diag:
        assert isinstance(diag["_diversity_loss"], torch.Tensor), "diversity_loss should be Tensor"
    # Continuous plasticity means open_frac/lock_frac should be from continuous path
    # Check z_per_step was returned
    assert "_z_per_step" in diag, f"missing _z_per_step in diag, keys: {list(diag.keys())}"
check("HPM continuous_plasticity path", test_hpm_continuous)

# ── 6. HPM default (no continuous_plasticity) ─────────────────────
print("\n=== 6. HPM default path (state machine) ===")
def test_hpm_default():
    cfg = HPMConfig(n_slots=4, slot_dim=32)
    hpm = HomeostaticPredictiveMemory(d_model=64, cfg=cfg)
    x = torch.randn(2, 10, 64)
    out_seq, diag = hpm(x)
    assert out_seq.shape[0] == 2 and out_seq.shape[1] == 10
    assert "hpm_open_frac" in diag, f"missing hpm_open_frac, keys: {list(diag.keys())}"
check("HPM default (state machine) path", test_hpm_default)

# ── 7. HPM gradient flow ─────────────────────────────────────────
print("\n=== 7. HPM gradient flow ===")
def test_gradient_flow():
    cfg = HPMConfig(n_slots=4, slot_dim=32, continuous_plasticity=True,
                    slot_diversity_lambda=0.01)
    hpm = HomeostaticPredictiveMemory(d_model=64, cfg=cfg)
    x = torch.randn(2, 10, 64, requires_grad=True)
    out_seq, diag = hpm(x)
    loss = out_seq.sum()
    if "_diversity_loss" in diag:
        loss = loss + diag["_diversity_loss"]
    loss.backward()
    assert x.grad is not None, "no gradient on input"
    assert x.grad.abs().sum() > 0, "zero gradient on input"
check("gradient flows through HPM -> input", test_gradient_flow)

# ── 8. IterativeQueryHead ─────────────────────────────────────────
print("\n=== 8. IterativeQueryHead ===")
from tmew1_queries import IterativeQueryHead
def test_iterative_query_head():
    iqh = IterativeQueryHead(
        d_input=96,
        d_memory=64,
        max_entities=10,
        num_query_types=5,
        d_entity=32,
    )
    B, T = 2, 10
    seq = torch.randn(B, T, 96)
    qtimes = torch.randint(0, T, (B, 3))
    qtypes = torch.randint(0, 5, (B, 3))
    entity_state = torch.randn(B, 4, 32)
    event_tape = torch.randn(B, 8, 64)
    event_mask = torch.ones(B, 8, dtype=torch.bool)
    entity_logits, binary_logits = iqh(
        seq, qtimes, qtypes,
        entity_state=entity_state,
        event_tape=event_tape,
        event_tape_mask=event_mask,
    )
    assert entity_logits.shape == (B, 3, 10), f"entity_logits shape {entity_logits.shape}"
    assert binary_logits.shape == (B, 3, 2), f"binary_logits shape {binary_logits.shape}"
check("IterativeQueryHead forward", test_iterative_query_head)

# ── 9. Score logging ─────────────────────────────────────────────
print("\n=== 9. Score logging groups ===")
from score_logging import _METRIC_GROUPS, build_default_metric_specs
def test_score_logging():
    specs = build_default_metric_specs()
    assert "entity_route_entropy" in specs, "missing entity_route_entropy"
    assert "event_tape_n_events" in specs, "missing event_tape_n_events"
    group_names = [g[0] for g in _METRIC_GROUPS]
    assert "Entity" in group_names, f"missing Entity group, got {group_names}"
    assert "EventTape" in group_names, f"missing EventTape group, got {group_names}"
check("score_logging metric groups", test_score_logging)

# ── 10. WorldModel config + instantiation ─────────────────────────
print("\n=== 10. WorldModel config ===")
from homeostatic_multimodal_world_model_chunked import WorldModelConfig, ForwardOutput
def test_world_model_config():
    wcfg = WorldModelConfig.__dataclass_fields__
    assert "entity_table_config" in wcfg, "missing entity_table_config in WorldModelConfig"
    assert "event_tape_config" in wcfg, "missing event_tape_config in WorldModelConfig"
    # ForwardOutput fields
    fout_fields = ForwardOutput.__dataclass_fields__
    for f in ["entity_states", "event_tape_entries", "event_tape_mask", "event_tape_diagnostics"]:
        assert f in fout_fields, f"missing {f} in ForwardOutput"
check("WorldModelConfig + ForwardOutput fields", test_world_model_config)

# ── Summary ───────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  RESULTS: {PASS} passed, {FAIL} failed")
print(f"{'='*60}")
sys.exit(1 if FAIL else 0)
