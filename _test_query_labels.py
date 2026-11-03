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
            # Recover the per-step holder from the ambient audio channel
            # (0.3 at 8 + holder_id each step, 1.0 on the handoff step).
            holder_audio = ep.audio[:, 8:8 + cfg.max_entities]
            holder_per_step = holder_audio.argmax(axis=-1)
            for q in ep.queries:
                if q.qtype != "who_holds_token":
                    continue
                assert np.max(holder_audio[q.time_asked]) > 0.0
                assert q.target == int(holder_per_step[q.time_asked]), (
                    f"tier {tier.tier} seed {i}: target {q.target} != audio holder "
                    f"{int(holder_per_step[q.time_asked])} at t={q.time_asked}"
                )
                n_checked += 1
    assert n_checked > 0


if __name__ == "__main__":
    test_holder_at_replays_history()
    print("PASS: HandoffState.holder_at replays transfer history")
    test_diagnostics_generator_labels_holder_at_query_time()
    print("PASS: diagnostics generator labels who_holds_token at query time")
    test_queries_generator_labels_holder_at_query_time()
    print("PASS: queries generator labels who_holds_token at query time")
    print("\nAll query-label tests PASSED")
