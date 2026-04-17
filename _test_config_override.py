"""Verify model_config_overrides actually reach WorldModelConfig."""
from tmew1_train import build_model, WorldConfig
from hpm import EntityTableConfig, EventTapeConfig, EntityHistoryConfig

world_cfg = WorldConfig()

# Test 1: baseline — no overrides
model, _, _ = build_model(world_cfg)
assert model.cfg.entity_table_config is None, f"Expected None, got {model.cfg.entity_table_config}"
assert model.cfg.event_tape_config is None
assert model.cfg.entity_history_config is None
assert not hasattr(model, "entity_table") or model.entity_table is None
print("Test 1 PASS: baseline has no entity modules")

# Test 2: with entity_table_config
et_cfg = EntityTableConfig(enabled=True, n_entities=4, d_entity=64)
model2, _, _ = build_model(world_cfg, entity_table_config=et_cfg)
assert model2.cfg.entity_table_config is not None
assert model2.cfg.entity_table_config.enabled
assert model2.entity_table is not None
print(f"Test 2 PASS: entity_table created, output_dim={model2.entity_table.output_dim}")

# Test 3: with all D2 configs
evt_cfg = EventTapeConfig(enabled=True, max_events=32, surprise_threshold=2.0)
eh_cfg = EntityHistoryConfig(enabled=True, n_snapshots=16)
model3, _, _ = build_model(
    world_cfg,
    entity_table_config=et_cfg,
    event_tape_config=evt_cfg,
    entity_history_config=eh_cfg,
)
assert model3.entity_table is not None
assert model3.event_tape is not None
assert model3.entity_history is not None
print(f"Test 3 PASS: all D2 modules created")
print(f"  entity_table: {model3.entity_table}")
print(f"  event_tape: {model3.event_tape}")
print(f"  entity_history: {model3.entity_history}")

# Test 4: ensure baseline model is still clean after building overridden ones
model4, _, _ = build_model(world_cfg)
assert model4.cfg.entity_table_config is None
print("Test 4 PASS: no cross-contamination between builds")

print("\nAll tests PASSED")
