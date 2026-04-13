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

    for e in state.entities:
        e.x = (e.x + e.dx) % cfg.grid_w
        e.y = (e.y + e.dy) % cfg.grid_h
        if state.rng.random() < 0.15:
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

    # Token transfer: if the holder is adjacent to another entity, hand off with prob 0.9.
    holder = next((e for e in state.entities if e.id == handoff.holder_id), None)
    if holder is not None:
        for e in state.entities:
            if e.id == handoff.holder_id:
                continue
            if abs(holder.x - e.x) + abs(holder.y - e.y) <= 1 and state.rng.random() < 0.9:
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
                    state.alarm_in = state.rng.randint(2, max_delay)
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

    # Chain 2 (multi_chain support — always uses simple proximity)
    if not state.chain2_alarm_fired and state.chain2_alarm_in < 0:
        for i in range(len(state.entities) - 1, 0, -1):
            for j in range(i - 1, -1, -1):
                a, b = state.entities[i], state.entities[j]
                if abs(a.x - b.x) + abs(a.y - b.y) <= 1:
                    if not events["trigger"] or (a.id != state.entities[0].id):
                        state.chain2_alarm_in = state.rng.randint(2, max(3, max_delay // 2))
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

    for t in range(T):
        events = _step_world_with_handoff(state, handoff, cfg, t, tier.max_delay)
        event_history.append(dict(events))
        vision[t] = _render_vision(state, cfg)
        audio[t] = _render_audio(state, cfg, events, current_holder_id=handoff.holder_id)
        numeric[t] = _render_numeric(state, cfg)
        text[t] = _render_text(state, cfg, events)
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
        # Re-render audio from the forced timestep onward so the ambient holder
        # identity matches the synthetic transfer for the rest of the episode.
        for t in range(force_t, T):
            forced_events = dict(event_history[t])
            if t == force_t:
                forced_events["handoff"] = True
                forced_events["new_holder_id"] = receiver.id
            audio[t] = _render_audio(state, cfg, forced_events, current_holder_id=receiver.id)

    # Build queries. Time-asked is always toward the back half so the model has
    # to actually carry information forward.
    rng = random.Random(seed + 7)
    queries: List[Query] = []
    back_half_start = max(1, T // 2)
    for _ in range(num_queries):
        qtype = rng.choice(QUERY_TYPES)
        time_asked = rng.randint(back_half_start, T - 1)
        if qtype == "who_holds_token":
            target = handoff.holder_id
            queries.append(Query(qtype, target, time_asked, is_binary=False))
        elif qtype == "who_was_first_tagged":
            target = max(0, handoff.first_tagged_id)
            queries.append(Query(qtype, target, time_asked, is_binary=False))
        elif qtype == "did_alarm_fire":
            target = 1 if fired_at >= 0 and fired_at <= time_asked else 0
            queries.append(Query(qtype, target, time_asked, is_binary=True))
        elif qtype == "which_entity_occluded":
            target = max(0, last_occluded_id)
            queries.append(Query(qtype, target, time_asked, is_binary=False))
        elif qtype == "did_trigger_before_alarm":
            # Was the trigger observed before the alarm by query time?
            target = 1 if (trigger_at >= 0 and fired_at >= 0 and trigger_at < fired_at and fired_at <= time_asked) else 0
            queries.append(Query(qtype, target, time_asked, is_binary=True))
        elif qtype == "which_entity_first_occluded":
            target = max(0, first_occluded_id)
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
    """
    t_max = output_sequence.size(1) - 1
    qtimes = batch["query_times"].clamp(max=t_max)
    entity_logits, binary_logits = query_head(output_sequence, qtimes, batch["query_types"])
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
