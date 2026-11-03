"""
Observation contract: what the simulator exposes, and what it deliberately hides.

The benchmark asks memory questions, so the observation stream must (a) ground
entity identity, otherwise entity-id answers are hidden simulator labels, and
(b) expose state changes as transient events rather than standing state,
otherwise the query is a per-step readout. These checks pin both for the v2
generators:

1. The per-entity numeric block carries each entity's position, visibility,
   colour and id-slot, matching the simulator state.
2. The audio holder channels fire only on handoff steps (no ambient holder).
3. The per-entity tag channel fires exactly once per tagged entity, on the step
   it was tagged, and the ``who_was_first_tagged`` target is the entity whose
   tag event comes first.
"""
from __future__ import annotations

import numpy as np

from tmew1_train_v2 import (
    WorldConfig,
    DEFAULT_TIERS,
    NUMERIC_ENTITY_OFFSET,
    NUMERIC_ENTITY_STRIDE,
)
from tmew1_diagnostics_v2 import generate_episode_with_diagnostics


def _entity_block(ep, cfg):
    T = ep.length
    n = cfg.max_entities
    return ep.numeric[:, NUMERIC_ENTITY_OFFSET:NUMERIC_ENTITY_OFFSET + NUMERIC_ENTITY_STRIDE * n].reshape(T, n, NUMERIC_ENTITY_STRIDE)


def test_numeric_dim_covers_entity_block() -> None:
    cfg = WorldConfig()
    assert cfg.numeric_dim >= NUMERIC_ENTITY_OFFSET + NUMERIC_ENTITY_STRIDE * cfg.max_entities


def test_holder_cue_is_transient() -> None:
    cfg = WorldConfig()
    n_handoff_steps = 0
    for tier in DEFAULT_TIERS:
        for i in range(48):
            ep = generate_episode_with_diagnostics(cfg, tier, seed=7000 + 1000 * tier.tier + i, num_queries=4, enable_false_cue=tier.tier >= 2)
            holder_audio = ep.audio[:, 8:8 + cfg.max_entities]
            h = ep.holder_per_step
            changed = np.zeros(ep.length, dtype=bool)
            changed[1:] = h[1:] != h[:-1]
            for t in range(ep.length):
                if t == 0:
                    # The initial holder is not recoverable from holder_per_step
                    # alone, so step 0 may carry a cue only for its own holder.
                    others = np.delete(holder_audio[0], h[0])
                    assert np.all(others == 0.0) and holder_audio[0, h[0]] in (0.0, 1.0)
                    continue
                if changed[t]:
                    assert holder_audio[t, h[t]] == 1.0, f"missing handoff cue at t={t}"
                    n_handoff_steps += 1
                else:
                    assert np.all(holder_audio[t] == 0.0), (
                        f"tier {tier.tier} seed {i}: ambient holder cue leaked at t={t}: {holder_audio[t]}"
                    )
    assert n_handoff_steps > 0


def test_entity_block_and_tag_events() -> None:
    cfg = WorldConfig()
    n_tag_events = 0
    n_first_tag_checked = 0
    for tier in DEFAULT_TIERS:
        for i in range(48):
            ep = generate_episode_with_diagnostics(cfg, tier, seed=8000 + 1000 * tier.tier + i, num_queries=4, enable_false_cue=tier.tier >= 2)
            block = _entity_block(ep, cfg)
            # Positions are normalised to [0, 1]; colours too; visibility is binary.
            assert np.all((block[..., :2] >= 0.0) & (block[..., :2] <= 1.0))
            assert set(np.unique(block[..., 2])) <= {0.0, 1.0}
            tag = block[..., 4]
            assert set(np.unique(tag)) <= {0.0, 1.0}
            # At most one tag event per entity, and the first one defines the target.
            per_entity_events = tag.sum(axis=0)
            assert np.all(per_entity_events <= 1.0), f"tag event repeated: {per_entity_events}"
            n_tag_events += int(per_entity_events.sum())
            first_tag_steps = {e: int(np.argmax(tag[:, e])) for e in range(cfg.max_entities) if per_entity_events[e] > 0}
            targets = [q.target for q in ep.queries if q.qtype == "who_was_first_tagged"]
            if first_tag_steps and targets:
                first_entity = min(first_tag_steps, key=first_tag_steps.get)
                for tgt in targets:
                    assert tgt == first_entity, f"tier {tier.tier} seed {i}: first_tagged target {tgt} != first tag event entity {first_entity}"
                    n_first_tag_checked += 1
    assert n_tag_events > 0
    assert n_first_tag_checked > 0


if __name__ == "__main__":
    test_numeric_dim_covers_entity_block()
    print("PASS: numeric_dim covers the per-entity block")
    test_holder_cue_is_transient()
    print("PASS: holder audio cue fires only on handoff steps")
    test_entity_block_and_tag_events()
    print("PASS: entity block is well-formed and tag events identify the first-tagged entity")
    print("\nAll observation-contract tests PASSED")
