"""
Regression test: query labels must be answerable at the time the query is asked.

The world model is a causal scan, so a label that depends on events after
``time_asked`` is unanswerable by construction. This test pins the
``who_holds_token`` target to the simulator's holder at ``time_asked`` for both
v2 episode generators, and checks that ``HandoffState.holder_at`` replays the
transfer history consistently with ``holder_per_step``.
"""
from __future__ import annotations

import numpy as np

from tmew1_train_v2 import WorldConfig, DEFAULT_TIERS
from tmew1_diagnostics_v2 import generate_episode_with_diagnostics
from tmew1_queries_v2 import HandoffState, generate_episode_with_queries


def test_holder_at_replays_history() -> None:
    h = HandoffState(holder_id=3, transfer_history=[(4, 0, 2), (9, 2, 3)])
    assert h.holder_at(0) == 0
    assert h.holder_at(3) == 0
    assert h.holder_at(4) == 2
    assert h.holder_at(8) == 2
    assert h.holder_at(9) == 3
    assert h.holder_at(100) == 3
    assert HandoffState(holder_id=5, transfer_history=[]).holder_at(7) == 5


def test_diagnostics_generator_labels_holder_at_query_time() -> None:
    cfg = WorldConfig()
    n_checked = 0
    n_differs_from_final = 0
    for tier in DEFAULT_TIERS:
        for i in range(96):
            ep = generate_episode_with_diagnostics(
                cfg, tier, seed=9000 + 1000 * tier.tier + i, num_queries=4,
                enable_false_cue=tier.tier >= 2,
            )
            for q in ep.queries:
                if q.qtype != "who_holds_token":
                    continue
                assert q.target == int(ep.holder_per_step[q.time_asked]), (
                    f"tier {tier.tier} seed {i}: target {q.target} != holder "
                    f"{int(ep.holder_per_step[q.time_asked])} at t={q.time_asked}"
                )
                n_checked += 1
                if q.target != int(ep.holder_per_step[-1]):
                    n_differs_from_final += 1
    assert n_checked > 0
    # The point of the fix: a meaningful fraction of queries are asked before
    # the last handoff, so the final holder is the wrong label for them.
    assert n_differs_from_final > 0, "expected some queries before the last handoff"


def test_queries_generator_labels_holder_at_query_time() -> None:
    cfg = WorldConfig()
    n_checked = 0
    for tier in DEFAULT_TIERS:
        for i in range(64):
            ep = generate_episode_with_queries(cfg, tier, seed=5000 + 1000 * tier.tier + i, num_queries=4)
            assert ep.holder_per_step is not None and ep.holder_per_step.shape == (ep.length,)
            for q in ep.queries:
                if q.qtype != "who_holds_token":
                    continue
                assert q.target == int(ep.holder_per_step[q.time_asked]), (
                    f"tier {tier.tier} seed {i}: target {q.target} != holder "
                    f"{int(ep.holder_per_step[q.time_asked])} at t={q.time_asked}"
                )
                n_checked += 1
    assert n_checked > 0


def test_latch_value_targets_agree_with_query_targets() -> None:
    """The per-step register targets must equal the query targets at time_asked."""
    import torch
    from tmew1_diagnostics_v2 import build_latch_targets, episode_to_diag_tensors, collate_diag
    from hpm_v2 import LATCH_REGISTERS

    cfg = WorldConfig()
    reg = {name: i for i, name in enumerate(LATCH_REGISTERS)}
    n_holder = n_tag = n_color = n_gate = 0
    for tier in DEFAULT_TIERS:
        eps = [
            generate_episode_with_diagnostics(cfg, tier, seed=11000 + 1000 * tier.tier + i, num_queries=4, enable_false_cue=tier.tier >= 2)
            for i in range(48)
        ]
        batch = collate_diag([episode_to_diag_tensors(ep) for ep in eps])
        T_in = batch["vision"].size(1) - 1          # inputs are shifted by one
        gate, value = build_latch_targets(batch, T_in, LATCH_REGISTERS)
        assert gate.shape == (len(eps), T_in, len(LATCH_REGISTERS)) and value.shape == gate.shape
        for b, ep in enumerate(eps):
            for q in ep.queries:
                t = min(q.time_asked, T_in - 1)
                if q.qtype == "who_holds_token":
                    assert int(value[b, t, reg["holder"]]) == q.target, (tier.tier, b, q)
                    n_holder += 1
                elif q.qtype == "who_was_first_tagged" and (ep.first_tagged_step < 0 or ep.first_tagged_step <= t):
                    assert int(value[b, t, reg["first_tagged"]]) == q.target, (tier.tier, b, q, ep.first_tagged_step)
                    n_tag += 1
            # colour: after the change step the register target is the entity; before it, 0
            if 0 <= ep.color_change_step < T_in:
                assert int(value[b, ep.color_change_step, reg["color_changed"]]) == ep.color_change_entity_id
                assert int(value[b, T_in - 1, reg["color_changed"]]) == ep.color_change_entity_id
                if ep.color_change_step > 0:
                    assert int(value[b, ep.color_change_step - 1, reg["color_changed"]]) == 0
                assert float(gate[b, ep.color_change_step, reg["color_changed"]]) == 1.0
                assert float(gate[b, :, reg["color_changed"]].sum()) == 1.0
                n_color += 1
            # gate targets sit on the cue step: holder audio one-hot, numeric tag bit
            h = ep.holder_per_step[:T_in]
            for t in range(1, T_in):
                g = float(gate[b, t, reg["holder"]])
                assert g == float(h[t] != h[t - 1])
                if g == 1.0:
                    assert ep.audio[t, 8 + h[t]] == 1.0, "holder gate target off the audio cue step"
                    n_gate += 1
            if 0 <= ep.first_tagged_step < T_in:
                assert float(gate[b, ep.first_tagged_step, reg["first_tagged"]]) == 1.0
                assert ep.numeric[ep.first_tagged_step, 10 + 5 * ep.first_tagged_id + 4] == 1.0, "first_tagged gate target off the numeric cue step"
                assert float(gate[b, :, reg["first_tagged"]].sum()) == 1.0
    assert n_holder > 0 and n_tag > 0 and n_color > 0 and n_gate > 0


if __name__ == "__main__":
    test_holder_at_replays_history()
    print("PASS: HandoffState.holder_at replays transfer history")
    test_diagnostics_generator_labels_holder_at_query_time()
    print("PASS: diagnostics generator labels who_holds_token at query time")
    test_queries_generator_labels_holder_at_query_time()
    print("PASS: queries generator labels who_holds_token at query time")
    test_latch_value_targets_agree_with_query_targets()
    print("PASS: latch register targets agree with query targets at time_asked")
    print("\nAll query-label tests PASSED")
