"""
TMEW-1 extension: handoff template + delayed-recall queries + QueryHead.

Designed to slot alongside tmew1_train.py without modifying it. Import the new
pieces from here and pass them into a thin trainer wrapper.

What this adds:
1. A 'handoff' template where one entity transfers a token to another, and the
   identity of the holder must be recalled later.
2. A query schema attached to each episode: (query_type, target, time_asked).
3. A QueryHead module that conditions on the model's sequence output and a
   learned query-type embedding to produce an answer logit over entity IDs or
   binary outcomes.
4. A query loss and a query-step trainer add-on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import math
import random

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from tmew1_train import (
    WorldConfig,
    WorldState,
    Entity,
    CurriculumTier,
    _make_world,
    _render_vision,
    _render_audio,
    _render_numeric,
    _render_text,
    Episode,
)


# -----------------------------------------------------------------------------
# Query schema
# -----------------------------------------------------------------------------
QUERY_TYPES: Tuple[str, ...] = (
    "who_holds_token",          # answer: entity id
    "who_was_first_tagged",     # answer: entity id
    "did_alarm_fire",           # answer: 0/1
    "which_entity_occluded",    # answer: entity id (or -1 if none)
    "did_trigger_before_alarm", # answer: 0/1 — temporal ordering
    "which_entity_first_occluded",  # answer: entity id — who was occluded earliest
    "did_chain2_fire",          # answer: 0/1 — second causal chain
)

QUERY_TYPE_TO_IDX: Dict[str, int] = {q: i for i, q in enumerate(QUERY_TYPES)}
@dataclass
class Query:
    qtype: str
    target: int                 # entity id, or 0/1 for binary
    time_asked: int             # the step index after which the query is posed
    is_binary: bool


@dataclass
class EpisodeWithQueries:
    vision: np.ndarray
    audio: np.ndarray
    numeric: np.ndarray
    text: np.ndarray
    latent_rule: int
    template: str
    length: int
    queries: List[Query]


# -----------------------------------------------------------------------------
# Handoff template — extends the world simulator with a token holder
# -----------------------------------------------------------------------------
@dataclass
class HandoffState:
    holder_id: int
    transfer_history: List[Tuple[int, int, int]]  # (t, from_id, to_id)
    first_tagged_id: int = -1
    color_change_entity_id: int = -1   # entity that changed color, -1 if none
    color_change_step: int = -1        # step at which the change occurred


def _step_world_with_handoff(
    state: WorldState,
    handoff: HandoffState,
    cfg: WorldConfig,
    t: int,
    max_delay: int,
) -> Dict[str, Any]:
    """Drop-in for _step_world that also tracks token transfers."""
    events: Dict[str, Any] = {
        "trigger": False, "alarm_fire": False, "occluded_ids": [],
        "handoff": False, "chain2_trigger": False, "chain2_fire": False,
    }

    # --- A4: rule-dynamic motion modulation ---
    if cfg.rule_dynamic:
        rule = state.active_rule
        if rule <= 1:
            motion_change_prob = 0.15    # baseline
        elif rule <= 3:
            motion_change_prob = 0.05    # sticky — predictable paths
        else:
            motion_change_prob = 0.30    # jittery — erratic paths
    else:
        motion_change_prob = 0.15

    for e in state.entities:
        e.x = (e.x + e.dx) % cfg.grid_w
        e.y = (e.y + e.dy) % cfg.grid_h
        if state.rng.random() < motion_change_prob:
            e.dx = state.rng.choice([-1, 0, 1])
            e.dy = state.rng.choice([-1, 0, 1])

    # --- Color change event: once per episode, ~8% per step ---
    if handoff.color_change_entity_id < 0 and state.rng.random() < 0.08:
        e = state.rng.choice(state.entities)
        candidates = [c for c in range(cfg.vision_channels) if c != e.color]
        if candidates:
            e.color = state.rng.choice(candidates)
            handoff.color_change_entity_id = e.id
            handoff.color_change_step = t
            events["color_change"] = True
            events["color_change_entity_id"] = e.id

    # Token transfer — A4: rule-dynamic handoff probability and range
    if cfg.rule_dynamic:
        _r = state.active_rule
        if _r <= 1:
            _ho_prob, _ho_range = 0.9, 1      # baseline
        elif _r <= 3:
            _ho_prob, _ho_range = 0.5, 1      # sticky token
        else:
            _ho_prob, _ho_range = 0.95, 2     # slippery token
    else:
        _ho_prob, _ho_range = 0.9, 1
    holder = next((e for e in state.entities if e.id == handoff.holder_id), None)
    if holder is not None:
        for e in state.entities:
            if e.id == handoff.holder_id:
                continue
            if abs(holder.x - e.x) + abs(holder.y - e.y) <= _ho_range and state.rng.random() < _ho_prob:
                handoff.transfer_history.append((t, holder.id, e.id))
                handoff.holder_id = e.id
                events["handoff"] = True
                events["new_holder_id"] = e.id
                break

    # --- Conditional trigger rules based on active_rule ---
    def _check_trigger_condition(a: Entity, b: Entity) -> bool:
        rule = state.active_rule
        if rule <= 1:
            return abs(a.x - b.x) + abs(a.y - b.y) <= 1
        elif rule <= 3:
            return a.color == b.color and abs(a.x - b.x) + abs(a.y - b.y) <= 2
        else:  # rule 4-5
            return (a.tagged or b.tagged) and abs(a.x - b.x) + abs(a.y - b.y) <= 1

    # Chain 1: primary alarm
    if not state.alarm_fired and state.alarm_in < 0:
        for i in range(len(state.entities)):
            for j in range(i + 1, len(state.entities)):
                a, b = state.entities[i], state.entities[j]
                if _check_trigger_condition(a, b):
                    # A4: rule-dynamic alarm delay
                    if cfg.rule_dynamic:
                        _r = state.active_rule
                        if _r <= 1:
                            _dlo, _dhi = 2, max_delay
                        elif _r <= 3:
                            _dlo, _dhi = 2, max(2, max_delay - 1)
                        else:
                            _dlo, _dhi = min(3, max_delay), max_delay
                    else:
                        _dlo, _dhi = 2, max_delay
                    state.alarm_in = state.rng.randint(_dlo, _dhi)
                    events["trigger"] = True
                    a.tagged = True
                    if handoff.first_tagged_id < 0:
                        handoff.first_tagged_id = a.id
                    break
            if events["trigger"]:
                break

    if state.alarm_in > 0:
        state.alarm_in -= 1
        if state.alarm_in == 0:
            state.alarm_fired = True
            events["alarm_fire"] = True
            # A4: rule-dynamic post-alarm consequences
            if cfg.rule_dynamic:
                _r = state.active_rule
                if 2 <= _r <= 3:
                    for e in state.entities:
                        if e.tagged:
                            e.dx, e.dy = 0, 0
                elif _r >= 4:
                    for e in state.entities:
                        e.dx = state.rng.choice([-2, -1, 1, 2])
                        e.dy = state.rng.choice([-2, -1, 1, 2])

    # Chain 2 (multi_chain support — always uses simple proximity)
    if not state.chain2_alarm_fired and state.chain2_alarm_in < 0:
        primary_chain_busy = (state.alarm_in > 0) and (not state.alarm_fired)
        if not primary_chain_busy or cfg.chain2_temporal_overlap:
            chain2_radius = 2 if cfg.chain2_frequency_boost > 1.0 else 1
            chain2_delay_hi = max(3, max_delay // 2)
            if cfg.chain2_frequency_boost >= 1.5:
                chain2_delay_hi = max(2, chain2_delay_hi - 1)
            for i in range(len(state.entities) - 1, 0, -1):
                for j in range(i - 1, -1, -1):
                    a, b = state.entities[i], state.entities[j]
                    if abs(a.x - b.x) + abs(a.y - b.y) <= chain2_radius:
                        if cfg.chain2_frequency_boost < 1.5 and events["trigger"]:
                            primary_pair_used = any(entity.id in (a.id, b.id) for entity in state.entities[:1])
                            if primary_pair_used:
                                continue
                        state.chain2_alarm_in = state.rng.randint(2, chain2_delay_hi)
                        state.chain2_trigger_pair = (a.id, b.id)
                        events["chain2_trigger"] = True
                        break
                if events["chain2_trigger"]:
                    break
    if state.chain2_alarm_in > 0:
        state.chain2_alarm_in -= 1
        if state.chain2_alarm_in == 0:
            state.chain2_alarm_fired = True
            events["chain2_fire"] = True

    if state.occluder is not None:
        ox0, oy0, ox1, oy1 = state.occluder
        for e in state.entities:
            inside = ox0 <= e.x <= ox1 and oy0 <= e.y <= oy1
            e.visible = not inside
            if inside:
                events["occluded_ids"].append(e.id)

    return events


def generate_episode_with_queries(
    cfg: WorldConfig,
    tier: CurriculumTier,
    seed: int,
    num_queries: int = 4,
) -> EpisodeWithQueries:
    template = random.Random(seed).choice(tier.template_pool)
    state = _make_world(cfg, template, tier.max_delay, tier.occlusion, seed)
    handoff = HandoffState(holder_id=state.entities[0].id, transfer_history=[])

    T = tier.max_episode_length
    vision = np.zeros((T, cfg.vision_channels, cfg.grid_h, cfg.grid_w), dtype=np.float32)
    audio = np.zeros((T, cfg.audio_dim), dtype=np.float32)
    numeric = np.zeros((T, cfg.numeric_dim), dtype=np.float32)
    text = np.zeros((T, 1), dtype=np.int64)

    last_occluded_id = -1
    first_occluded_id = -1
    fired_at = -1
    trigger_at = -1
    chain2_fired_at = -1
    event_history: List[Dict[str, Any]] = []
    holder_per_step = np.zeros((T,), dtype=np.int64)

    for t in range(T):
        events = _step_world_with_handoff(state, handoff, cfg, t, tier.max_delay)
        event_history.append(dict(events))
        vision[t] = _render_vision(state, cfg)
        audio[t] = _render_audio(state, cfg, events, current_holder_id=handoff.holder_id)
        numeric[t] = _render_numeric(state, cfg)
        text[t] = _render_text(state, cfg, events)
        holder_per_step[t] = handoff.holder_id
        if events["occluded_ids"]:
            last_occluded_id = events["occluded_ids"][0]
            if first_occluded_id < 0:
                first_occluded_id = events["occluded_ids"][0]
        if events.get("alarm_fire") and fired_at < 0:
            fired_at = t
        if events.get("trigger") and trigger_at < 0:
            trigger_at = t
        if events.get("chain2_fire") and chain2_fired_at < 0:
            chain2_fired_at = t

    # Guarantee at least one handoff in handoff-template episodes so the model
    # can't solve the query by memorising the prior "entity 0 always holds it."
    if template == "handoff" and len(handoff.transfer_history) == 0 and len(state.entities) >= 2:
        force_t = max(1, T // 4)
        receiver = next((e for e in state.entities if e.id != handoff.holder_id), state.entities[0])
        handoff.transfer_history.append((force_t, handoff.holder_id, receiver.id))
        handoff.holder_id = receiver.id
        holder_per_step[force_t:] = receiver.id
        # Re-render audio from the forced timestep onward so the ambient holder
        # identity matches the synthetic transfer for the rest of the episode.
        for t in range(force_t, T):
            forced_events = dict(event_history[t])
            if t == force_t:
                forced_events["handoff"] = True
                forced_events["new_holder_id"] = receiver.id
            audio[t] = _render_audio(state, cfg, forced_events, current_holder_id=receiver.id)

    # Build queries. The trainer consumes observations 0..T-2, so keep qtimes
    # in that visible range and answer stateful questions from the prefix rather
    # than from future simulator state.
    rng = random.Random(seed + 7)
    queries: List[Query] = []
    max_query_time = max(0, T - 2)
    back_half_start = min(max_query_time, max(1, T // 2))

    def _last_occluded_at(step: int) -> int:
        last_id = -1
        for tt in range(0, min(step, T - 1) + 1):
            ids = event_history[tt].get("occluded_ids") or []
            if ids:
                last_id = int(ids[0])
        return max(0, last_id)

    def _first_occluded_at(step: int) -> int:
        for tt in range(0, min(step, T - 1) + 1):
            ids = event_history[tt].get("occluded_ids") or []
            if ids:
                return int(ids[0])
        return 0

    for _ in range(num_queries):
        qtype = rng.choice(QUERY_TYPES)
        time_asked = rng.randint(back_half_start, max_query_time)
        if qtype == "who_holds_token":
            target = int(holder_per_step[time_asked])
            queries.append(Query(qtype, target, time_asked, is_binary=False))
        elif qtype == "who_was_first_tagged":
            target = int(handoff.first_tagged_id) if (handoff.first_tagged_id >= 0 and 0 <= trigger_at <= time_asked) else 0
            queries.append(Query(qtype, target, time_asked, is_binary=False))
        elif qtype == "did_alarm_fire":
            target = 1 if fired_at >= 0 and fired_at <= time_asked else 0
            queries.append(Query(qtype, target, time_asked, is_binary=True))
        elif qtype == "which_entity_occluded":
            target = _last_occluded_at(time_asked)
            queries.append(Query(qtype, target, time_asked, is_binary=False))
        elif qtype == "did_trigger_before_alarm":
            # Was the trigger observed before the alarm by query time?
            target = 1 if (trigger_at >= 0 and fired_at >= 0 and trigger_at < fired_at and fired_at <= time_asked) else 0
            queries.append(Query(qtype, target, time_asked, is_binary=True))
        elif qtype == "which_entity_first_occluded":
            target = _first_occluded_at(time_asked)
            queries.append(Query(qtype, target, time_asked, is_binary=False))
        elif qtype == "did_chain2_fire":
            target = 1 if chain2_fired_at >= 0 and chain2_fired_at <= time_asked else 0
            queries.append(Query(qtype, target, time_asked, is_binary=True))

    return EpisodeWithQueries(
        vision=vision,
        audio=audio,
        numeric=numeric,
        text=text,
        latent_rule=state.active_rule,
        template=template,
        length=T,
        queries=queries,
    )


# -----------------------------------------------------------------------------
# QueryHead
# -----------------------------------------------------------------------------
class QueryHead(nn.Module):
    """
    Conditions on (sequence_at_query_time, query_type_embedding) and produces
    a logit vector over entity_id slots plus a binary head for yes/no queries.

    Two heads share a trunk because the model should learn 'how to answer
    questions about its own state', not 'how to answer this specific question'.
    """

    def __init__(self, d_model: int, max_entities: int, num_query_types: int):
        super().__init__()
        self.max_entities = max_entities
        self.num_query_types = num_query_types
        self.qtype_embed = nn.Embedding(num_query_types, d_model)
        self.trunk = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.entity_head = nn.Linear(d_model, max_entities)
        self.binary_head = nn.Linear(d_model, 2)

    def forward(self, sequence: Tensor, query_times: Tensor, query_types: Tensor) -> Tuple[Tensor, Tensor]:
        """
        sequence:    (B, T, D)
        query_times: (B, Q) long, indexes into T
        query_types: (B, Q) long, indexes into num_query_types
        Returns: (entity_logits (B, Q, max_entities), binary_logits (B, Q, 2))
        """
        b, t, d = sequence.shape
        q = query_times.size(1)

        # Gather sequence states at query times: (B, Q, D)
        time_idx = query_times.unsqueeze(-1).expand(-1, -1, d)
        states = torch.gather(sequence, dim=1, index=time_idx)

        type_emb = self.qtype_embed(query_types)              # (B, Q, D)
        fused = self.trunk(torch.cat([states, type_emb], dim=-1))

        entity_logits = self.entity_head(fused)               # (B, Q, max_entities)
        binary_logits = self.binary_head(fused)               # (B, Q, 2)
        return entity_logits, binary_logits


class IterativeQueryHead(nn.Module):
    """
    2-step iterative retrieval head with cross-attention over entity table
    and event tape.

    Step 1: Cross-attend from query to [entity_state + event_tape]
    Step 2: Use result to construct follow-up query, cross-attend again

    This allows the head to trace chains: read entity A's holder → use that
    to look up when the handoff occurred → identify the holder after handoff.
    Designed to close the handoff-count accuracy cliff (0.87 @ 0 → 0.30 @ 6+).

    Memory ablation modes (``memory_ablation``):
      - "fused":    all queries use full memory bank (entity + tape + history).
      - "et_only":  all queries use entity-table-only memory.
      - "tape_only": all queries use event-tape-only memory.
      - "no_aux":   no auxiliary memory — queries fall back to the projected
                    sequence state (cross-attention skipped entirely).

    ET-only per-qtype ablation: when ``et_only_qtypes`` is a non-empty set of
    query-type indices, queries of those types use entity-table-only memory.
    This is orthogonal to ``memory_ablation`` — if ``memory_ablation`` is set
    to anything other than "fused", it overrides per-qtype routing.
    """

    def __init__(
        self,
        d_input: int,          # augmented sequence dim (d_model + max_entities + hpm_dim)
        d_memory: int,         # memory entry dim (d_model, matches event tape output)
        max_entities: int,
        num_query_types: int,
        d_entity: int = 0,     # per-entity dim for projection to d_memory
        n_attn_heads: int = 4,
        n_retrieval_steps: int = 2,
        et_only_qtypes: Optional[set] = None,
        memory_ablation: str = "fused",  # "fused" | "et_only" | "tape_only" | "no_aux"
    ):
        super().__init__()
        d = d_memory
        self.d_memory = d
        self.max_entities = max_entities
        self.num_query_types = num_query_types
        self.n_retrieval_steps = n_retrieval_steps
        self.et_only_qtypes: set = et_only_qtypes or set()
        assert memory_ablation in ("fused", "et_only", "tape_only", "no_aux"), \
            f"Invalid memory_ablation mode: {memory_ablation}"
        self.memory_ablation = memory_ablation

        self.qtype_embed = nn.Embedding(num_query_types, d)

        # Project augmented sequence to working dim.
        self.input_proj = nn.Linear(d_input, d)

        # Fuse projected state + query type.
        self.query_fuse = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.GELU(),
        )

        # Entity state projection (d_entity → d_memory).
        self.entity_proj = nn.Linear(d_entity, d) if d_entity > 0 else None

        # Iterative cross-attention layers.
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(d, n_attn_heads, batch_first=True)
            for _ in range(n_retrieval_steps)
        ])
        self.cross_ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, d * 2),
                nn.GELU(),
                nn.Linear(d * 2, d),
            )
            for _ in range(n_retrieval_steps)
        ])
        self.cross_norm1 = nn.ModuleList([
            nn.LayerNorm(d) for _ in range(n_retrieval_steps)
        ])
        self.cross_norm2 = nn.ModuleList([
            nn.LayerNorm(d) for _ in range(n_retrieval_steps)
        ])

        # Output heads.
        self.entity_head = nn.Linear(d, max_entities)
        self.binary_head = nn.Linear(d, 2)

    def _run_cross_attention(self, query: Tensor, memory: Tensor,
                             key_padding_mask: Tensor) -> Tensor:
        """Run iterative cross-attention over a memory bank."""
        h = query
        for i in range(self.n_retrieval_steps):
            attn_out, _ = self.cross_attn[i](
                h, memory, memory,
                key_padding_mask=key_padding_mask,
            )
            h = self.cross_norm1[i](h + attn_out)
            h = self.cross_norm2[i](h + self.cross_ffn[i](h))
        return h

    def forward(
        self,
        sequence: Tensor,
        query_times: Tensor,
        query_types: Tensor,
        entity_state: Optional[Tensor] = None,       # (B, n_e, d_entity)
        event_tape: Optional[Tensor] = None,          # (B, max_events, d_memory)
        event_tape_mask: Optional[Tensor] = None,     # (B, max_events) bool
        entity_history: Optional[Tensor] = None,      # (B, K, d_memory)
        entity_history_mask: Optional[Tensor] = None,  # (B, K) bool
    ) -> Tuple[Tensor, Tensor]:
        """
        sequence:    (B, T, d_input)  — augmented sequence
        query_times: (B, Q) long
        query_types: (B, Q) long
        entity_state: (B, n_e, d_entity) — current entity table state
        event_tape:   (B, max_events, d_memory) — projected tape entries
        event_tape_mask: (B, max_events) bool — True for valid entries

        Returns: (entity_logits (B, Q, max_entities), binary_logits (B, Q, 2))
        """
        b, t, d_in = sequence.shape
        q = query_times.size(1)
        d = self.d_memory
        device = sequence.device

        # Gather sequence states at query times and project down.
        time_idx = query_times.unsqueeze(-1).expand(-1, -1, d_in)
        states = torch.gather(sequence, dim=1, index=time_idx)  # (B, Q, d_input)
        states_proj = self.input_proj(states)  # (B, Q, d)

        type_emb = self.qtype_embed(query_types)  # (B, Q, d)
        query = self.query_fuse(torch.cat([states_proj, type_emb], dim=-1))  # (B, Q, d)

        # ── Build memory banks ──────────────────────────────────────────
        # Entity-only memory (always available when entity_state is provided).
        et_memory: Optional[Tensor] = None
        et_kpm: Optional[Tensor] = None
        if entity_state is not None and self.entity_proj is not None:
            entity_mem = self.entity_proj(entity_state)  # (B, n_e, d)
            et_memory = entity_mem
            et_kpm = torch.zeros(b, entity_state.size(1),
                                 dtype=torch.bool, device=device)  # all valid

        # Tape-only memory bank (event tape + entity history, no entity table).
        tape_memory: Optional[Tensor] = None
        tape_kpm: Optional[Tensor] = None
        tape_parts: List[Tensor] = []
        tape_mask_parts: List[Tensor] = []
        if event_tape is not None:
            tape_parts.append(event_tape)
            tape_mask_parts.append(
                event_tape_mask if event_tape_mask is not None
                else torch.ones(b, event_tape.size(1), dtype=torch.bool, device=device)
            )
        if entity_history is not None:
            tape_parts.append(entity_history)
            tape_mask_parts.append(
                entity_history_mask if entity_history_mask is not None
                else torch.ones(b, entity_history.size(1), dtype=torch.bool, device=device)
            )
        if tape_parts:
            tape_memory = torch.cat(tape_parts, dim=1)
            tape_kpm = ~torch.cat(tape_mask_parts, dim=1)

        # Full memory: entity + event tape + entity history.
        full_memory: Optional[Tensor] = None
        full_kpm: Optional[Tensor] = None
        memory_parts: List[Tensor] = []
        mask_parts: List[Tensor] = []
        if et_memory is not None:
            memory_parts.append(et_memory)
            mask_parts.append(torch.ones(b, et_memory.size(1),
                                         dtype=torch.bool, device=device))
        if event_tape is not None:
            memory_parts.append(event_tape)
            if event_tape_mask is not None:
                mask_parts.append(event_tape_mask)
            else:
                mask_parts.append(torch.ones(b, event_tape.size(1),
                                             dtype=torch.bool, device=device))
        if entity_history is not None:
            memory_parts.append(entity_history)
            if entity_history_mask is not None:
                mask_parts.append(entity_history_mask)
            else:
                mask_parts.append(torch.ones(b, entity_history.size(1),
                                             dtype=torch.bool, device=device))
        if memory_parts:
            full_memory = torch.cat(memory_parts, dim=1)  # (B, M, d)
            full_kpm = ~torch.cat(mask_parts, dim=1)      # True = ignore

        # ── Memory ablation: global override ────────────────────────────
        # When memory_ablation is not "fused", it overrides all per-qtype
        # routing and forces a single memory source for every query.
        if self.memory_ablation == "no_aux":
            fused = query  # skip cross-attention entirely
        elif self.memory_ablation == "et_only":
            if et_memory is not None:
                fused = self._run_cross_attention(query, et_memory, et_kpm)
            else:
                fused = query
        elif self.memory_ablation == "tape_only":
            if tape_memory is not None:
                fused = self._run_cross_attention(query, tape_memory, tape_kpm)
            else:
                fused = query

        elif self.et_only_qtypes:
            # Mixed per-qtype routing: ET-only for selected queries, fused otherwise.
            use_et = torch.zeros(b, q, dtype=torch.bool, device=device)
            for qtype_idx in self.et_only_qtypes:
                use_et |= (query_types == qtype_idx)

            if use_et.any() and et_memory is not None:
                h_et = self._run_cross_attention(query, et_memory, et_kpm)
            else:
                h_et = query

            if (~use_et).any() and full_memory is not None:
                h_full = self._run_cross_attention(query, full_memory, full_kpm)
            else:
                h_full = query

            fused = torch.where(use_et.unsqueeze(-1), h_et, h_full)
        else:
            # Default fused path: all queries use full memory.
            if full_memory is not None:
                fused = self._run_cross_attention(query, full_memory, full_kpm)
            else:
                fused = query

        entity_logits = self.entity_head(fused)  # (B, Q, max_entities)
        binary_logits = self.binary_head(fused)  # (B, Q, 2)
        return entity_logits, binary_logits


# -----------------------------------------------------------------------------
# Sequence augmentation
# -----------------------------------------------------------------------------
def augment_sequence_with_holder_audio(
    sequence: Tensor,
    audio: Optional[Tensor],
    max_entities: int,
    use_audio: bool = True,
) -> Tensor:
    """
    Append raw holder-identity audio channels to the sequence representation.

    sequence: (B, T, D)
    audio:    (B, T_full, A), typically aligned to the unshifted episode.
    """
    b, t, _ = sequence.shape
    if not use_audio or audio is None:
        holder_audio = torch.zeros(b, t, max_entities, device=sequence.device, dtype=sequence.dtype)
        return torch.cat([sequence, holder_audio], dim=-1)

    holder_start = 8
    holder_end = holder_start + max_entities
    clipped = audio[:, :t, holder_start:min(holder_end, audio.size(-1))].to(sequence.dtype)
    if clipped.size(-1) < max_entities:
        pad = torch.zeros(b, t, max_entities - clipped.size(-1), device=sequence.device, dtype=sequence.dtype)
        clipped = torch.cat([clipped, pad], dim=-1)
    return torch.cat([sequence, clipped], dim=-1)


# -----------------------------------------------------------------------------
# Query loss
# -----------------------------------------------------------------------------
def query_loss(
    entity_logits: Tensor,
    binary_logits: Tensor,
    targets: Tensor,
    is_binary: Tensor,
) -> Tuple[Tensor, Dict[str, float]]:
    """
    targets:   (B, Q) long. For entity queries this is the entity id, for binary
               queries it is 0/1.
    is_binary: (B, Q) bool mask. True where the query is binary.
    """
    binary_mask = is_binary
    entity_mask = ~is_binary

    losses = []
    metrics: Dict[str, float] = {}

    if entity_mask.any():
        ent_l = entity_logits[entity_mask]
        ent_t = targets[entity_mask]
        ent_loss = F.cross_entropy(ent_l, ent_t)
        losses.append(ent_loss)
        with torch.no_grad():
            metrics["entity_acc"] = (ent_l.argmax(-1) == ent_t).float().mean().item()
            metrics["entity_loss"] = float(ent_loss.item())

    if binary_mask.any():
        bin_l = binary_logits[binary_mask]
        bin_t = targets[binary_mask]
        bin_loss = F.cross_entropy(bin_l, bin_t)
        losses.append(bin_loss)
        with torch.no_grad():
            metrics["binary_acc"] = (bin_l.argmax(-1) == bin_t).float().mean().item()
            metrics["binary_loss"] = float(bin_loss.item())

    if not losses:
        return torch.zeros((), device=entity_logits.device), metrics
    return torch.stack(losses).mean(), metrics


# -----------------------------------------------------------------------------
# Collation for query-augmented batches
# -----------------------------------------------------------------------------
def collate_with_queries(batch: Sequence[Dict[str, Any]]) -> Dict[str, Tensor]:
    """Pads queries to a fixed Q per batch — assumes generator emits the same num_queries."""
    out: Dict[str, Tensor] = {}
    keys_static = ("vision", "audio", "numeric", "text", "latent_rule")
    for k in keys_static:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    out["query_times"] = torch.stack([b["query_times"] for b in batch], dim=0)
    out["query_types"] = torch.stack([b["query_types"] for b in batch], dim=0)
    out["query_targets"] = torch.stack([b["query_targets"] for b in batch], dim=0)
    out["query_is_binary"] = torch.stack([b["query_is_binary"] for b in batch], dim=0)
    return out


def episode_to_tensors(ep: EpisodeWithQueries) -> Dict[str, Tensor]:
    """Convert an EpisodeWithQueries to the tensor dict the trainer consumes."""
    qtimes = torch.tensor([q.time_asked for q in ep.queries], dtype=torch.long)
    qtypes = torch.tensor([QUERY_TYPE_TO_IDX[q.qtype] for q in ep.queries], dtype=torch.long)
    qtargets = torch.tensor([q.target for q in ep.queries], dtype=torch.long)
    qbinary = torch.tensor([q.is_binary for q in ep.queries], dtype=torch.bool)
    return {
        "vision": torch.from_numpy(ep.vision),
        "audio": torch.from_numpy(ep.audio),
        "numeric": torch.from_numpy(ep.numeric),
        "text": torch.from_numpy(ep.text).squeeze(-1),
        "latent_rule": torch.tensor(ep.latent_rule, dtype=torch.long),
        "query_times": qtimes,
        "query_types": qtypes,
        "query_targets": qtargets,
        "query_is_binary": qbinary,
    }


# -----------------------------------------------------------------------------
# Trainer add-on
# -----------------------------------------------------------------------------
def query_train_step_addon(
    output_sequence: Tensor,
    query_head: QueryHead,
    batch: Dict[str, Tensor],
    query_type_to_idx: Optional[Dict[str, int]] = None,
    weight: float = 0.5,
    **query_kwargs,
) -> Tuple[Tensor, Dict[str, float]]:
    """
    Call this from inside the main train_step after model.forward and before
    backward(). Returns (weighted_query_loss, metrics) — add the loss to total.

    Wiring example in the main trainer:

        output = model(...)
        losses = criterion(output, ...)
        q_loss, q_metrics = query_train_step_addon(output.sequence, query_head, batch)
        total = losses.total + 0.5 * aux_latent + q_loss
        total.backward()
        optimizer.step()
        model.controller_step(total.detach())   # after optimizer.step(); mutates parameters

    Extra **query_kwargs are forwarded to query_head.forward() — used by
    IterativeQueryHead for entity_state, event_tape, event_tape_mask.
    """
    t_max = output_sequence.size(1) - 1
    qtimes = batch["query_times"].clamp(max=t_max)
    entity_logits, binary_logits = query_head(
        output_sequence, qtimes, batch["query_types"], **query_kwargs,
    )
    qtypes = batch["query_types"]
    targets = batch["query_targets"]
    is_binary = batch["query_is_binary"]

    query_weights = torch.ones_like(targets, dtype=torch.float32, device=targets.device)
    qtype_map = query_type_to_idx or QUERY_TYPE_TO_IDX
    if "who_holds_token" in qtype_map:
        query_weights[qtypes == qtype_map["who_holds_token"]] = 2.0
    if "what_was_true_rule" in qtype_map:
        query_weights[qtypes == qtype_map["what_was_true_rule"]] = 3.0

    losses = []
    metrics: Dict[str, float] = {}

    entity_mask = ~is_binary
    binary_mask = is_binary

    if entity_mask.any():
        ent_logits = entity_logits[entity_mask]
        ent_targets = targets[entity_mask]
        ent_weights = query_weights[entity_mask]
        ent_loss_vec = F.cross_entropy(ent_logits, ent_targets, reduction="none")
        ent_loss = (ent_loss_vec * ent_weights).sum() / ent_weights.sum().clamp_min(1.0)
        losses.append(ent_loss)
        with torch.no_grad():
            metrics["entity_acc"] = (ent_logits.argmax(-1) == ent_targets).float().mean().item()
            metrics["entity_loss"] = float(ent_loss.item())

    if binary_mask.any():
        bin_logits = binary_logits[binary_mask]
        bin_targets = targets[binary_mask]
        bin_weights = query_weights[binary_mask]
        bin_loss_vec = F.cross_entropy(bin_logits, bin_targets, reduction="none")
        bin_loss = (bin_loss_vec * bin_weights).sum() / bin_weights.sum().clamp_min(1.0)
        losses.append(bin_loss)
        with torch.no_grad():
            metrics["binary_acc"] = (bin_logits.argmax(-1) == bin_targets).float().mean().item()
            metrics["binary_loss"] = float(bin_loss.item())

    if not losses:
        return torch.zeros((), device=output_sequence.device), metrics

    return weight * torch.stack(losses).mean(), metrics


__all__ = [
    "QUERY_TYPES",
    "QUERY_TYPE_TO_IDX",
    "Query",
    "EpisodeWithQueries",
    "HandoffState",
    "augment_sequence_with_holder_audio",
    "generate_episode_with_queries",
    "QueryHead",
    "query_loss",
    "collate_with_queries",
    "episode_to_tensors",
    "query_train_step_addon",
]
