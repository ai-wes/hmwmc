"""
TMEW-1 diagnostics and false-cue template.

Adds three things on top of tmew1_queries.py without modifying it:
1. False-cue / belief-revision template — early audio cue implies one rule,
   later evidence corrects it. Adds 'what_was_true_rule' query.
2. Per-episode metadata (num_handoffs, had_false_cue, occlusion_steps) so
   diagnostics can bucket accuracy by difficulty.
3. recall_by_difficulty() — buckets who_holds_token accuracy by handoff count
   and logs episodic read entropy alongside, to distinguish write failures
   from read failures.

Usage from tmew1_run.py:
    from tmew1_diagnostics import (
        FALSE_CUE_QUERY_TYPES,
        generate_episode_with_diagnostics,
        recall_by_difficulty,
        log_episodic_entropy,
    )
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import os
import random
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tmew1_train import WorldConfig, CurriculumTier, _make_world, _render_vision, _render_audio, _render_numeric, _render_text
from score_logging import (
    ScoreColorMapper,
    MetricSpec,
    ScoreDirection,
    _Ansi,
)
from tmew1_queries import (
    QUERY_TYPES,
    QUERY_TYPE_TO_IDX,
    Query,
    EpisodeWithQueries,
    HandoffState,
    augment_sequence_with_holder_audio,
    _step_world_with_handoff,
)


# -----------------------------------------------------------------------------
# Extended query type set: adds belief-revision query
# -----------------------------------------------------------------------------
BASE_EXTENDED_QUERY_TYPES: Tuple[str, ...] = QUERY_TYPES + ("what_was_true_rule", "which_entity_changed_color")
BRANCH_QUERY_TYPES: Tuple[str, ...] = (
    "did_occlusion_before_handoff",
    "did_chain2_before_chain1",
    "which_event_was_first",
    "closest_entity_to_holder_at_alarm",
    "entity_sharing_color_with_trigger",
    "which_entity_visible_at_correction",
    "entity_never_occluded",
    "entity_never_tagged",
    "chain_never_fired",
    "holder_if_handoff2_absent",
    "would_alarm_fire_without_correction",
)
_ENV_EXTRA_QUERY_FAMILIES = "TMEW1_EXTRA_QUERY_FAMILIES"
_ENV_REPLACE_QUERY_FAMILIES = "TMEW1_REPLACE_QUERY_FAMILIES"
_FIRST_EVENT_CODES: Dict[str, int] = {
    "trigger": 0,
    "handoff": 1,
    "occlusion": 2,
    "alarm": 3,
    "chain2_fire": 4,
    "correction": 5,
}


def _parse_query_family_env(name: str) -> Tuple[str, ...]:
    raw = os.environ.get(name, "")
    if not raw:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def get_extended_query_types() -> Tuple[str, ...]:
    replace = _parse_query_family_env(_ENV_REPLACE_QUERY_FAMILIES)
    extra = _parse_query_family_env(_ENV_EXTRA_QUERY_FAMILIES)
    supported = set(BASE_EXTENDED_QUERY_TYPES) | set(BRANCH_QUERY_TYPES)
    if replace:
        active = tuple(dict.fromkeys(q for q in replace if q in supported))
        return active or BASE_EXTENDED_QUERY_TYPES
    active = tuple(dict.fromkeys(BASE_EXTENDED_QUERY_TYPES + extra))
    return tuple(q for q in active if q in supported)


def get_extended_query_type_to_idx() -> Dict[str, int]:
    return {q: i for i, q in enumerate(get_extended_query_types())}


EXTENDED_QUERY_TYPES: Tuple[str, ...] = BASE_EXTENDED_QUERY_TYPES
EXTENDED_QUERY_TYPE_TO_IDX: Dict[str, int] = {q: i for i, q in enumerate(EXTENDED_QUERY_TYPES)}


# -----------------------------------------------------------------------------
# Episode + per-episode difficulty metadata
# -----------------------------------------------------------------------------
@dataclass
class EpisodeWithDiagnostics:
    vision: np.ndarray
    audio: np.ndarray
    numeric: np.ndarray
    text: np.ndarray
    holder_per_step: np.ndarray
    latent_rule: int           # the TRUE latent rule, post-correction
    template: str
    length: int
    queries: List[Query]
    # Difficulty metadata
    num_handoffs: int
    had_false_cue: bool
    cue_corrected_at: int      # -1 if no correction
    occlusion_steps: int
    color_change_entity_id: int = -1   # entity that changed color, -1 if none
    color_change_step: int = -1        # step at which color change occurred
    # Event timing for emergent probes
    trigger_step: int = -1
    alarm_step: int = -1
    chain2_fire_step: int = -1
    false_cue_step: int = -1           # step when false cue was injected


# -----------------------------------------------------------------------------
# False-cue template
# -----------------------------------------------------------------------------
def _inject_false_cue(audio_vec: np.ndarray, t: int, false_cue_step: int, correction_step: int) -> np.ndarray:
    """Audio channels 5,6,7 carry the false-cue signal and its correction."""
    if t == false_cue_step:
        audio_vec[5] = 1.0      # initial misleading high-frequency burst
    if correction_step <= t < correction_step + 4:
        audio_vec[6] = 1.0      # durable correction pulse
    if false_cue_step <= t < correction_step:
        audio_vec[7] = 0.4      # ambient "stale belief" hum
    return audio_vec


def generate_episode_with_diagnostics(
    cfg: WorldConfig,
    tier: CurriculumTier,
    seed: int,
    num_queries: int = 4,
    enable_false_cue: bool = False,
) -> EpisodeWithDiagnostics:
    rng = random.Random(seed)
    active_query_types = get_extended_query_types()
    template_pool = list(tier.template_pool)
    if enable_false_cue and "false_cue" not in template_pool:
        template_pool.append("false_cue")
    template = rng.choice(template_pool)

    state = _make_world(cfg, template, tier.max_delay, tier.occlusion, seed)
    handoff = HandoffState(holder_id=state.entities[0].id, transfer_history=[])

    T = tier.max_episode_length

    has_false_cue = template == "false_cue"
    false_cue_step = T // 4 if has_false_cue else -1
    correction_step = (T * 3) // 5 if has_false_cue else -1
    decoy_rule = rng.randint(0, cfg.num_latent_rules - 1) if has_false_cue else -1
    true_rule = state.active_rule

    vision = np.zeros((T, cfg.vision_channels, cfg.grid_h, cfg.grid_w), dtype=np.float32)
    audio = np.zeros((T, cfg.audio_dim), dtype=np.float32)
    numeric = np.zeros((T, cfg.numeric_dim), dtype=np.float32)
    text = np.zeros((T, 1), dtype=np.int64)

    fired_at = -1
    trigger_at = -1
    chain2_fired_at = -1
    first_handoff_at = -1
    last_occluded_id = -1
    first_occluded_id = -1
    first_occlusion_at = -1
    occlusion_steps = 0
    holder_per_step = np.zeros((T,), dtype=np.int64)
    event_history: List[Dict[str, Any]] = []
    active_entity_ids = tuple(e.id for e in state.entities)
    positions = np.full((T, cfg.max_entities, 2), -1, dtype=np.int64)
    visible = np.zeros((T, cfg.max_entities), dtype=np.bool_)
    colors = np.full((T, cfg.max_entities), -1, dtype=np.int64)
    ever_occluded = np.zeros((cfg.max_entities,), dtype=np.bool_)
    ever_tagged = np.zeros((cfg.max_entities,), dtype=np.bool_)

    # Snapshots for B4 counterfactual rollouts
    _cf_correction_snapshot: Optional[Tuple[Any, Any]] = None  # (WorldState, HandoffState) at correction_step

    for t in range(T):
        # Save snapshot right before correction for counterfactual replay
        if has_false_cue and t == correction_step and _cf_correction_snapshot is None:
            _cf_correction_snapshot = (copy.deepcopy(state), copy.deepcopy(handoff))

        if has_false_cue and false_cue_step <= t < correction_step:
            true_rule_saved = state.active_rule
            state.active_rule = decoy_rule

        events = _step_world_with_handoff(state, handoff, cfg, t, tier.max_delay)
        event_history.append(dict(events))
        vision[t] = _render_vision(state, cfg)
        audio[t] = _render_audio(state, cfg, events, current_holder_id=handoff.holder_id)
        audio[t] = _inject_false_cue(audio[t], t, false_cue_step, correction_step)
        numeric[t] = _render_numeric(state, cfg)
        text[t] = _render_text(state, cfg, events)
        holder_per_step[t] = handoff.holder_id

        for e in state.entities:
            positions[t, e.id, 0] = e.x
            positions[t, e.id, 1] = e.y
            visible[t, e.id] = bool(e.visible)
            colors[t, e.id] = e.color
            if e.tagged:
                ever_tagged[e.id] = True

        if has_false_cue and false_cue_step <= t < correction_step:
            state.active_rule = true_rule_saved

        if events.get("alarm_fire") and fired_at < 0:
            fired_at = t
        if events.get("trigger") and trigger_at < 0:
            trigger_at = t
        if events.get("chain2_fire") and chain2_fired_at < 0:
            chain2_fired_at = t
        if events.get("handoff") and first_handoff_at < 0:
            first_handoff_at = t
        if events.get("occluded_ids"):
            last_occluded_id = events["occluded_ids"][0]
            if first_occluded_id < 0:
                first_occluded_id = events["occluded_ids"][0]
            if first_occlusion_at < 0:
                first_occlusion_at = t
            for entity_id in events["occluded_ids"]:
                if 0 <= entity_id < cfg.max_entities:
                    ever_occluded[entity_id] = True
            occlusion_steps += 1

    if template == "handoff" and len(handoff.transfer_history) == 0 and len(state.entities) >= 2:
        force_t = max(1, T // 4)
        receiver = next((e for e in state.entities if e.id != handoff.holder_id), state.entities[0])
        handoff.transfer_history.append((force_t, handoff.holder_id, receiver.id))
        handoff.holder_id = receiver.id
        if first_handoff_at < 0:
            first_handoff_at = force_t
        holder_per_step[force_t:] = receiver.id
        for t in range(force_t, T):
            forced_events = dict(event_history[t])
            if t == force_t:
                forced_events["handoff"] = True
                forced_events["new_holder_id"] = receiver.id
            audio[t] = _render_audio(state, cfg, forced_events, current_holder_id=receiver.id)
            audio[t] = _inject_false_cue(audio[t], t, false_cue_step, correction_step)

    trigger_source_id = handoff.first_tagged_id if handoff.first_tagged_id >= 0 else -1

    # ── B4 counterfactual rollouts ──────────────────────────────────────
    # 1) holder_if_handoff2_absent: who holds the token at episode end if
    #    the 2nd handoff (index 1) never happened?
    cf_holder_no_handoff2: Optional[int] = None
    if len(handoff.transfer_history) >= 2:
        # Replay the handoff chain, skipping entry at index 1
        _cf_holder = handoff.transfer_history[0][2]  # after 1st handoff
        # The 2nd handoff (index 1) is skipped, so holder stays as _cf_holder.
        # For handoffs 3+ (index 2+), they only occur if the holder at that
        # moment matches the from_id. Replay sequentially.
        for _hi in range(2, len(handoff.transfer_history)):
            _ht, _hfrom, _hto = handoff.transfer_history[_hi]
            if _hfrom == _cf_holder:
                _cf_holder = _hto
        cf_holder_no_handoff2 = _cf_holder

    # 2) would_alarm_fire_without_correction: in a false-cue episode, if
    #    the correction never arrived (decoy rule stays active from
    #    correction_step onward), would the alarm still fire?
    cf_alarm_without_correction: Optional[int] = None  # 0 or 1, or None
    if has_false_cue and _cf_correction_snapshot is not None:
        cf_state, cf_handoff = _cf_correction_snapshot
        # Keep the decoy rule active (don't restore true_rule)
        cf_state.active_rule = decoy_rule
        cf_alarm_fired = cf_state.alarm_fired  # may already have fired
        for _ct in range(correction_step, T):
            cf_events = _step_world_with_handoff(cf_state, cf_handoff, cfg, _ct, tier.max_delay)
            if cf_events.get("alarm_fire"):
                cf_alarm_fired = True
        cf_alarm_without_correction = 1 if cf_alarm_fired else 0

    def _first_visible_entity_at(step: int) -> Optional[int]:
        if step < 0 or step >= T:
            return None
        candidates = [entity_id for entity_id in active_entity_ids if visible[step, entity_id]]
        return min(candidates) if candidates else None

    def _closest_entity_to_holder_at_alarm() -> Optional[int]:
        if fired_at < 0 or fired_at >= T:
            return None
        holder_id = int(holder_per_step[fired_at])
        hx, hy = positions[fired_at, holder_id]
        if hx < 0:
            return None
        best: Optional[Tuple[int, int]] = None
        for entity_id in active_entity_ids:
            if entity_id == holder_id:
                continue
            ex, ey = positions[fired_at, entity_id]
            if ex < 0:
                continue
            candidate = (abs(hx - ex) + abs(hy - ey), entity_id)
            if best is None or candidate < best:
                best = candidate
        return best[1] if best is not None else None

    def _entity_sharing_color_with_trigger() -> Optional[int]:
        if trigger_source_id < 0 or trigger_at < 0 or trigger_at >= T:
            return None
        trigger_color = int(colors[trigger_at, trigger_source_id])
        if trigger_color < 0:
            return None
        candidates = [
            entity_id for entity_id in active_entity_ids
            if entity_id != trigger_source_id and int(colors[trigger_at, entity_id]) == trigger_color
        ]
        return min(candidates) if candidates else None

    def _first_event_code() -> Optional[int]:
        event_times = [
            (trigger_at, _FIRST_EVENT_CODES["trigger"]),
            (first_handoff_at, _FIRST_EVENT_CODES["handoff"]),
            (first_occlusion_at, _FIRST_EVENT_CODES["occlusion"]),
            (fired_at, _FIRST_EVENT_CODES["alarm"]),
            (chain2_fired_at, _FIRST_EVENT_CODES["chain2_fire"]),
        ]
        if has_false_cue:
            event_times.append((correction_step, _FIRST_EVENT_CODES["correction"]))
        valid = [(time_idx, code) for time_idx, code in event_times if time_idx >= 0]
        if not valid:
            return None
        valid.sort(key=lambda item: (item[0], item[1]))
        return valid[0][1]

    closest_entity_to_holder_at_alarm = _closest_entity_to_holder_at_alarm()
    visible_at_correction = _first_visible_entity_at(correction_step)
    entity_sharing_color_with_trigger = _entity_sharing_color_with_trigger()
    entity_never_occluded = min((entity_id for entity_id in active_entity_ids if not ever_occluded[entity_id]), default=None)
    entity_never_tagged = min((entity_id for entity_id in active_entity_ids if not ever_tagged[entity_id]), default=None)
    first_event_code = _first_event_code()

    q_rng = random.Random(seed + 7)
    queries: List[Query] = []
    back_half_start = max(1, T // 2)
    _EVAL_ONLY = {"which_entity_changed_color"}

    def _query_available(qtype: str) -> bool:
        if qtype == "what_was_true_rule":
            return has_false_cue
        if qtype == "which_event_was_first":
            return first_event_code is not None
        if qtype == "closest_entity_to_holder_at_alarm":
            return closest_entity_to_holder_at_alarm is not None
        if qtype == "entity_sharing_color_with_trigger":
            return entity_sharing_color_with_trigger is not None
        if qtype == "which_entity_visible_at_correction":
            return has_false_cue and visible_at_correction is not None
        if qtype == "entity_never_occluded":
            return entity_never_occluded is not None
        if qtype == "entity_never_tagged":
            return entity_never_tagged is not None
        if qtype in {"holder_if_handoff2_absent"}:
            return cf_holder_no_handoff2 is not None
        if qtype in {"would_alarm_fire_without_correction"}:
            return cf_alarm_without_correction is not None
        return True

    candidate_pool = active_query_types
    if not has_false_cue:
        candidate_pool = tuple(
            q for q in candidate_pool
            if q not in {"what_was_true_rule", "which_entity_visible_at_correction", "would_alarm_fire_without_correction"}
        )
    pool = [q for q in candidate_pool if q not in _EVAL_ONLY and _query_available(q)]
    if not pool:
        pool = list(QUERY_TYPES)

    for _ in range(num_queries):
        qtype = q_rng.choice(pool)
        time_asked = (T - 1) if qtype in BRANCH_QUERY_TYPES else q_rng.randint(back_half_start, T - 1)
        if qtype == "who_holds_token":
            queries.append(Query(qtype, handoff.holder_id, time_asked, is_binary=False))
        elif qtype == "who_was_first_tagged":
            queries.append(Query(qtype, max(0, handoff.first_tagged_id), time_asked, is_binary=False))
        elif qtype == "did_alarm_fire":
            target = 1 if 0 <= fired_at <= time_asked else 0
            queries.append(Query(qtype, target, time_asked, is_binary=True))
        elif qtype == "which_entity_occluded":
            queries.append(Query(qtype, max(0, last_occluded_id), time_asked, is_binary=False))
        elif qtype == "what_was_true_rule":
            queries.append(Query(qtype, true_rule, time_asked, is_binary=False))
        elif qtype == "did_trigger_before_alarm":
            target = 1 if (trigger_at >= 0 and fired_at >= 0 and trigger_at < fired_at and fired_at <= time_asked) else 0
            queries.append(Query(qtype, target, time_asked, is_binary=True))
        elif qtype == "which_entity_first_occluded":
            queries.append(Query(qtype, max(0, first_occluded_id), time_asked, is_binary=False))
        elif qtype == "did_chain2_fire":
            target = 1 if chain2_fired_at >= 0 and chain2_fired_at <= time_asked else 0
            queries.append(Query(qtype, target, time_asked, is_binary=True))
        elif qtype == "did_occlusion_before_handoff":
            target = 1 if (first_occlusion_at >= 0 and first_handoff_at >= 0 and first_occlusion_at < first_handoff_at) else 0
            queries.append(Query(qtype, target, time_asked, is_binary=True))
        elif qtype == "did_chain2_before_chain1":
            target = 1 if (chain2_fired_at >= 0 and fired_at >= 0 and chain2_fired_at < fired_at) else 0
            queries.append(Query(qtype, target, time_asked, is_binary=True))
        elif qtype == "which_event_was_first":
            queries.append(Query(qtype, int(first_event_code), time_asked, is_binary=False))
        elif qtype == "closest_entity_to_holder_at_alarm":
            queries.append(Query(qtype, int(closest_entity_to_holder_at_alarm), time_asked, is_binary=False))
        elif qtype == "entity_sharing_color_with_trigger":
            queries.append(Query(qtype, int(entity_sharing_color_with_trigger), time_asked, is_binary=False))
        elif qtype == "which_entity_visible_at_correction":
            queries.append(Query(qtype, int(visible_at_correction), time_asked, is_binary=False))
        elif qtype == "entity_never_occluded":
            queries.append(Query(qtype, int(entity_never_occluded), time_asked, is_binary=False))
        elif qtype == "entity_never_tagged":
            queries.append(Query(qtype, int(entity_never_tagged), time_asked, is_binary=False))
        elif qtype == "chain_never_fired":
            target = 1 if ((fired_at >= 0) ^ (chain2_fired_at >= 0)) else 0
            queries.append(Query(qtype, target, time_asked, is_binary=True))
        elif qtype == "holder_if_handoff2_absent":
            queries.append(Query(qtype, int(cf_holder_no_handoff2), time_asked, is_binary=False))
        elif qtype == "would_alarm_fire_without_correction":
            queries.append(Query(qtype, int(cf_alarm_without_correction), time_asked, is_binary=True))

    return EpisodeWithDiagnostics(
        vision=vision,
        audio=audio,
        numeric=numeric,
        text=text,
        holder_per_step=holder_per_step,
        latent_rule=true_rule,
        template=template,
        length=T,
        queries=queries,
        num_handoffs=len(handoff.transfer_history),
        had_false_cue=has_false_cue,
        cue_corrected_at=correction_step,
        occlusion_steps=occlusion_steps,
        color_change_entity_id=handoff.color_change_entity_id,
        color_change_step=handoff.color_change_step,
        trigger_step=trigger_at,
        alarm_step=fired_at,
        chain2_fire_step=chain2_fired_at,
        false_cue_step=false_cue_step,
    )


def episode_to_diag_tensors(ep: EpisodeWithDiagnostics) -> Dict[str, Tensor]:
    qtype_map = get_extended_query_type_to_idx()
    qtimes = torch.tensor([q.time_asked for q in ep.queries], dtype=torch.long)
    qtypes = torch.tensor([qtype_map[q.qtype] for q in ep.queries], dtype=torch.long)
    qtargets = torch.tensor([q.target for q in ep.queries], dtype=torch.long)
    qbinary = torch.tensor([q.is_binary for q in ep.queries], dtype=torch.bool)
    return {
        "vision": torch.from_numpy(ep.vision),
        "audio": torch.from_numpy(ep.audio),
        "numeric": torch.from_numpy(ep.numeric),
        "text": torch.from_numpy(ep.text).squeeze(-1),
        "holder_per_step": torch.from_numpy(ep.holder_per_step),
        "latent_rule": torch.tensor(ep.latent_rule, dtype=torch.long),
        "query_times": qtimes,
        "query_types": qtypes,
        "query_targets": qtargets,
        "query_is_binary": qbinary,
        "num_handoffs": torch.tensor(ep.num_handoffs, dtype=torch.long),
        "had_false_cue": torch.tensor(int(ep.had_false_cue), dtype=torch.long),
        "occlusion_steps": torch.tensor(ep.occlusion_steps, dtype=torch.long),
        "color_change_entity_id": torch.tensor(ep.color_change_entity_id, dtype=torch.long),
        "color_change_step": torch.tensor(ep.color_change_step, dtype=torch.long),
        "trigger_step": torch.tensor(ep.trigger_step, dtype=torch.long),
        "alarm_step": torch.tensor(ep.alarm_step, dtype=torch.long),
        "chain2_fire_step": torch.tensor(ep.chain2_fire_step, dtype=torch.long),
        "cue_corrected_at": torch.tensor(ep.cue_corrected_at, dtype=torch.long),
        "false_cue_step": torch.tensor(ep.false_cue_step, dtype=torch.long),
    }


def collate_diag(batch: Sequence[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0].keys()}


class TMEW1DiagnosticDataset(Dataset):
    def __init__(self, cfg: WorldConfig, tier: CurriculumTier, n: int, base_seed: int, num_queries: int = 4, enable_false_cue: bool = True):
        self.cfg = cfg
        self.tier = tier
        self.n = n
        self.base_seed = base_seed
        self.num_queries = num_queries
        self.enable_false_cue = enable_false_cue

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        ep = generate_episode_with_diagnostics(
            self.cfg, self.tier, seed=self.base_seed + idx,
            num_queries=self.num_queries, enable_false_cue=self.enable_false_cue,
        )
        return episode_to_diag_tensors(ep)


# -----------------------------------------------------------------------------
# Recall accuracy bucketed by handoff count + episodic entropy logging
# -----------------------------------------------------------------------------
@torch.no_grad()
def recall_by_difficulty(
    model,
    query_head,
    loader: DataLoader,
    device: str,
    holder_feature_dim: int,
    enabled: Sequence[str] = ("vision", "numeric", "audio"),
) -> Dict[str, Any]:
    """
    Bucket who_holds_token accuracy by number of handoffs that occurred in
    the episode. Returns per-bucket accuracy plus mean episodic read entropy.

    The bucketing answers the question: does recall accuracy degrade with
    overwrite count? If yes -> episodic write gating is the bottleneck.
    If flat but low -> read attention is the bottleneck.
    """
    model.eval()
    query_head.eval()

    qtype_map = get_extended_query_type_to_idx()
    holds_idx = qtype_map.get("who_holds_token", -1)
    first_tagged_idx = qtype_map.get("who_was_first_tagged", -1)
    true_rule_idx = qtype_map.get("what_was_true_rule", -1)
    trigger_before_alarm_idx = qtype_map.get("did_trigger_before_alarm", -1)
    first_occluded_idx = qtype_map.get("which_entity_first_occluded", -1)
    chain2_fire_idx = qtype_map.get("did_chain2_fire", -1)
    extra_query_names = [
        q for q in get_extended_query_types()
        if q in BRANCH_QUERY_TYPES and q not in {"holder_if_handoff2_absent", "would_alarm_fire_without_correction"}
    ]
    extra_query_idxs = {qtype_map[q]: q for q in extra_query_names if q in qtype_map}

    _handoff_keys = ["0", "1", "2", "3", "4", "5", "6+"]
    bucket_correct: Dict[str, List[int]] = {k: [] for k in _handoff_keys}
    first_tagged_by_handoffs: Dict[str, List[int]] = {k: [] for k in _handoff_keys}
    true_rule_by_falsecue: Dict[str, List[int]] = {"with_cue": [], "without_cue": []}
    temporal_ordering_correct: List[int] = []
    first_occluded_correct: List[int] = []
    chain2_fire_correct: List[int] = []
    extra_query_correct: Dict[str, List[int]] = {q: [] for q in extra_query_names}
    color_change_correct: List[int] = []
    _lag_keys = ["0-5", "6-15", "16-30", "31+"]
    color_change_by_lag: Dict[str, List[int]] = {k: [] for k in _lag_keys}
    entropy_samples: List[float] = []

    def bucket_for(n: int) -> str:
        return "6+" if n >= 6 else str(n)

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        vision_in = batch["vision"][:, :-1]
        audio_in = batch["audio"][:, :-1]
        numeric_in = batch["numeric"][:, :-1]
        text_in = batch["text"][:, :-1]

        output = model(
            text_tokens=text_in if "text" in enabled else None,
            vision=vision_in if "vision" in enabled else None,
            audio=audio_in if "audio" in enabled else None,
            numeric=numeric_in if "numeric" in enabled else None,
        )
        if output.episodic_read_entropy is not None:
            entropy_samples.append(float(output.episodic_read_entropy))

        t_max = output.sequence.size(1) - 1
        qtimes = batch["query_times"].clamp(max=t_max)
        augmented_seq = augment_sequence_with_holder_audio(
            output.sequence,
            batch.get("audio"),
            max_entities=holder_feature_dim,
            use_audio="audio" in enabled,
        )
        if output.hpm_sequence is not None:
            augmented_seq = torch.cat([augmented_seq, output.hpm_sequence.to(augmented_seq.dtype)], dim=-1)
            
        entity_logits, binary_logits = query_head(augmented_seq, qtimes, batch["query_types"])

        bs, q = batch["query_types"].shape
        for bi in range(bs):
            n_h = int(batch["num_handoffs"][bi].item())
            had_cue = bool(batch["had_false_cue"][bi].item())
            bucket = bucket_for(n_h)
            for qi in range(q):
                qtype = int(batch["query_types"][bi, qi].item())
                target = int(batch["query_targets"][bi, qi].item())
                if batch["query_is_binary"][bi, qi]:
                    pred = int(binary_logits[bi, qi].argmax().item())
                else:
                    pred = int(entity_logits[bi, qi].argmax().item())
                correct = int(pred == target)

                if qtype == holds_idx:
                    bucket_correct[bucket].append(correct)
                elif qtype == first_tagged_idx:
                    first_tagged_by_handoffs[bucket].append(correct)
                elif qtype == true_rule_idx and true_rule_idx >= 0:
                    key = "with_cue" if had_cue else "without_cue"
                    true_rule_by_falsecue[key].append(correct)
                elif qtype == trigger_before_alarm_idx and trigger_before_alarm_idx >= 0:
                    temporal_ordering_correct.append(correct)
                elif qtype == first_occluded_idx and first_occluded_idx >= 0:
                    first_occluded_correct.append(correct)
                elif qtype == chain2_fire_idx and chain2_fire_idx >= 0:
                    chain2_fire_correct.append(correct)
                elif qtype in extra_query_idxs:
                    extra_query_correct[extra_query_idxs[qtype]].append(correct)

        # --- Zero-shot eval: color change query (never trained) ---
        # Use a *trained* "which entity" embedding so the probe tests HPM
        # retention, not trunk decoding of a random embedding vector.
        probe_qtype_idx = qtype_map["which_entity_occluded"]
        for bi in range(bs):
            cc_entity = int(batch["color_change_entity_id"][bi].item())
            cc_step = int(batch["color_change_step"][bi].item())
            if cc_entity < 0:
                continue
            lag = t_max - cc_step
            lag_bucket = "0-5" if lag <= 5 else "6-15" if lag <= 15 else "16-30" if lag <= 30 else "31+"
            syn_qtime = torch.tensor([[t_max]], device=device, dtype=torch.long)
            # Use a trained "which entity" embedding slot for the probe. The
            # color_change_idx embedding never receives gradient and would feed
            # the QueryHead trunk random noise.
            probe_qtype_idx = qtype_map["which_entity_occluded"]
            syn_qtype = torch.tensor([[probe_qtype_idx]], device=device, dtype=torch.long)
                
            syn_ent, _ = query_head(augmented_seq[bi:bi+1], syn_qtime, syn_qtype)
            pred = int(syn_ent[0, 0].argmax().item())
            correct = int(pred == cc_entity)
            color_change_correct.append(correct)
            color_change_by_lag[lag_bucket].append(correct)

    model.train()
    query_head.train()

    def summarize(d: Dict[str, List[int]]) -> Dict[str, Any]:
        out = {}
        for k, v in d.items():
            out[k] = {"acc": (sum(v) / len(v)) if v else None, "n": len(v)}
        return out

    return {
        "who_holds_token_by_handoffs": summarize(bucket_correct),
        "who_was_first_tagged_by_handoffs": summarize(first_tagged_by_handoffs),
        "what_was_true_rule_by_falsecue": summarize(true_rule_by_falsecue),
        "did_trigger_before_alarm": {"acc": (sum(temporal_ordering_correct) / len(temporal_ordering_correct)) if temporal_ordering_correct else None, "n": len(temporal_ordering_correct)},
        "which_entity_first_occluded": {"acc": (sum(first_occluded_correct) / len(first_occluded_correct)) if first_occluded_correct else None, "n": len(first_occluded_correct)},
        "did_chain2_fire": {"acc": (sum(chain2_fire_correct) / len(chain2_fire_correct)) if chain2_fire_correct else None, "n": len(chain2_fire_correct)},
        "extra_query_metrics": summarize(extra_query_correct),
        "which_entity_changed_color": {"acc": (sum(color_change_correct) / len(color_change_correct)) if color_change_correct else None, "n": len(color_change_correct)},
        "color_change_by_lag": summarize(color_change_by_lag),
        "mean_episodic_read_entropy": float(np.mean(entropy_samples)) if entropy_samples else None,
    }


# -----------------------------------------------------------------------------
# Emergent capability probes (zero-shot, untrained)
# -----------------------------------------------------------------------------
@torch.no_grad()
def run_emergent_probes(
    model,
    loader: DataLoader,
    device: str,
    enabled: Sequence[str] = ("vision", "numeric", "audio"),
) -> Dict[str, Any]:
    """
    Six zero-shot emergent capability probes. None are trained — they measure
    whether the model spontaneously organizes information in architecturally
    interesting ways.

    1. HPM slot specialization: do slots preferentially respond to different event types?
    2. Causal thread separation: are hidden reps for different causal threads distinct?
    3. Retroactive cause binding: after correction, does h(correction) align with h(cue)?
    4. State reactivation: after surprise, do hidden states briefly mirror earlier states?
    5. Event chunking: does ||Δh|| spike at event boundaries vs non-events?
    6. Surprise-event correlation: is HPM |z| higher at meaningful events?
    """
    model.eval()

    has_hpm = hasattr(model, "hpm") and model.hpm is not None
    if has_hpm:
        model.hpm._probe_mode = True

    # ── accumulators ──
    chunking_ratios: List[float] = []
    surprise_event_ratios: List[float] = []
    thread_sep_scores: List[float] = []
    retro_binding_scores: List[float] = []
    reactivation_scores: List[float] = []
    slot_event_gates: Dict[str, List[Tensor]] = {
        "handoff": [], "trigger": [], "alarm": [], "chain2": [], "correction": [],
    }

    try:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            output = model(
                text_tokens=batch["text"][:, :-1] if "text" in enabled else None,
                vision=batch["vision"][:, :-1] if "vision" in enabled else None,
                audio=batch["audio"][:, :-1] if "audio" in enabled else None,
                numeric=batch["numeric"][:, :-1] if "numeric" in enabled else None,
            )

            seq = output.sequence  # (B, T_seq, d_model)
            B, T_seq, D = seq.shape

            hpm_gates = model.hpm._probe_gates if has_hpm and model.hpm._probe_gates is not None else None
            hpm_z = model.hpm._probe_z if has_hpm and model.hpm._probe_z is not None else None

            for bi in range(B):
                h = seq[bi]  # (T_seq, D)
                holder = batch["holder_per_step"][bi]
                T_eff = min(T_seq, holder.shape[0] - 1)
                h_eff = h[:T_eff]
                holder_eff = holder[: T_eff + 1]

                if T_eff < 10:
                    continue

                # ── detect event steps ──
                handoff_steps: set = set()
                for t in range(1, T_eff):
                    if holder_eff[t] != holder_eff[t - 1]:
                        handoff_steps.add(t)

                trigger_t = int(batch["trigger_step"][bi].item())
                alarm_t = int(batch["alarm_step"][bi].item())
                chain2_t = int(batch["chain2_fire_step"][bi].item())
                correction_t = int(batch["cue_corrected_at"][bi].item())
                false_cue_t = int(batch["false_cue_step"][bi].item())

                all_event_steps: set = set(handoff_steps)
                for s in [trigger_t, alarm_t, chain2_t, correction_t]:
                    if 0 <= s < T_eff:
                        all_event_steps.add(s)
                non_event_steps = set(range(2, T_eff)) - all_event_steps

                if not all_event_steps or not non_event_steps:
                    continue

                # ── Probe 5: Event Chunking ──
                # Does ||Δh|| spike at event boundaries?
                deltas = (h_eff[1:] - h_eff[:-1]).norm(dim=-1)  # (T_eff-1,)
                event_deltas = [deltas[t - 1].item() for t in all_event_steps if 1 <= t < T_eff]
                non_event_deltas = [deltas[t - 1].item() for t in non_event_steps if 1 <= t < T_eff]
                if event_deltas and non_event_deltas:
                    mean_ev = sum(event_deltas) / len(event_deltas)
                    mean_nev = sum(non_event_deltas) / len(non_event_deltas)
                    if mean_nev > 1e-8:
                        chunking_ratios.append(mean_ev / mean_nev)

                # ── Probe 6: Surprise-Event Correlation ──
                # Is HPM |z| higher at meaningful events?
                if hpm_z is not None and hpm_z.shape[1] >= T_eff:
                    z_bi = hpm_z[bi, :T_eff].abs().mean(dim=-1)
                    event_z = [z_bi[t].item() for t in all_event_steps if t < T_eff]
                    non_event_z = [z_bi[t].item() for t in non_event_steps if t < T_eff]
                    if event_z and non_event_z:
                        mean_ez = sum(event_z) / len(event_z)
                        mean_nz = sum(non_event_z) / len(non_event_z)
                        if mean_nz > 1e-8:
                            surprise_event_ratios.append(mean_ez / mean_nz)

                # ── Probe 1: HPM Slot Specialization ──
                # Do individual slots respond preferentially to specific event types?
                if hpm_gates is not None and hpm_gates.shape[1] >= T_eff:
                    gates_bi = hpm_gates[bi, :T_eff]  # (T_eff, n_slots)
                    for evt_type, steps in [
                        ("handoff", handoff_steps),
                        ("trigger", {trigger_t} if 0 <= trigger_t < T_eff else set()),
                        ("alarm", {alarm_t} if 0 <= alarm_t < T_eff else set()),
                        ("chain2", {chain2_t} if 0 <= chain2_t < T_eff else set()),
                        ("correction", {correction_t} if 0 <= correction_t < T_eff else set()),
                    ]:
                        step_list = [s for s in steps if 0 <= s < T_eff]
                        if step_list:
                            mean_gate = gates_bi[step_list].mean(dim=0)
                            slot_event_gates[evt_type].append(mean_gate.cpu())

                # ── Probe 2: Causal Thread Separation ──
                # Thread A: handoff events. Thread B: alarm-related (trigger + alarm).
                thread_a = [t for t in handoff_steps if t < T_eff]
                thread_b = [t for t in [trigger_t, alarm_t] if 0 <= t < T_eff]
                if len(thread_a) >= 1 and len(thread_b) >= 1:
                    h_a = F.normalize(h_eff[thread_a], dim=-1)
                    h_b = F.normalize(h_eff[thread_b], dim=-1)
                    between = (h_a @ h_b.T).mean().item()
                    within_vals: List[float] = []
                    if len(thread_a) > 1:
                        wa = h_a @ h_a.T
                        mask_a = ~torch.eye(len(thread_a), device=device, dtype=torch.bool)
                        within_vals.extend(wa[mask_a].tolist())
                    if len(thread_b) > 1:
                        wb = h_b @ h_b.T
                        mask_b = ~torch.eye(len(thread_b), device=device, dtype=torch.bool)
                        within_vals.extend(wb[mask_b].tolist())
                    if within_vals:
                        within_mean = sum(within_vals) / len(within_vals)
                        thread_sep_scores.append(within_mean - between)

                # ── Probe 3: Retroactive Cause Binding ──
                # After correction, does h(correction) align more with h(cue) than h(random)?
                if 0 < correction_t < T_eff and 0 <= false_cue_t < T_eff:
                    h_corr = F.normalize(h_eff[correction_t: correction_t + 1], dim=-1)
                    h_cue = F.normalize(h_eff[false_cue_t: false_cue_t + 1], dim=-1)
                    binding_sim = (h_corr * h_cue).sum().item()
                    random_steps = [t for t in range(2, correction_t) if t != false_cue_t]
                    if random_steps:
                        h_rand = F.normalize(h_eff[random_steps], dim=-1)
                        rand_sim = (h_corr @ h_rand.T).mean().item()
                        retro_binding_scores.append(binding_sim - rand_sim)

                # ── Probe 4: State Reactivation After Surprise ──
                # After a surprise event (correction), do hidden states briefly
                # mirror specific earlier states more than temporal autocorrelation?
                if correction_t > 5 and correction_t + 3 < T_eff:
                    h_pre = F.normalize(h_eff[2:correction_t], dim=-1)
                    h_post = F.normalize(h_eff[correction_t + 1: correction_t + 4], dim=-1)
                    sims = h_post @ h_pre.T
                    max_sims = sims.max(dim=-1).values.mean().item()
                    if h_pre.shape[0] > 3:
                        baseline_sims = h_pre @ h_pre.T
                        mask_pre = ~torch.eye(h_pre.shape[0], device=device, dtype=torch.bool)
                        baseline_max = baseline_sims[mask_pre].view(h_pre.shape[0], -1).max(dim=-1).values.mean().item()
                        reactivation_scores.append(max_sims - baseline_max)
    finally:
        if has_hpm:
            model.hpm._probe_mode = False
            model.hpm._probe_gates = None
            model.hpm._probe_z = None
        model.train()

    # ── Probe 1 aggregation: slot specialization index ──
    slot_specialization = None
    event_types_with_data = [k for k, v in slot_event_gates.items() if v]
    if len(event_types_with_data) >= 2:
        n_slots = slot_event_gates[event_types_with_data[0]][0].shape[0]
        gate_matrix = torch.zeros(len(event_types_with_data), n_slots)
        for i, etype in enumerate(event_types_with_data):
            gate_matrix[i] = torch.stack(slot_event_gates[etype]).mean(dim=0)
        gate_matrix = gate_matrix / (gate_matrix.sum(dim=0, keepdim=True) + 1e-8)
        log_p = torch.log(gate_matrix + 1e-8)
        entropy_per_slot = -(gate_matrix * log_p).sum(dim=0)
        max_entropy = float(np.log(len(event_types_with_data)))
        if max_entropy > 0:
            normalized_entropy = entropy_per_slot / max_entropy
            slot_specialization = float(1.0 - normalized_entropy.mean().item())

    def _safe_mean(lst: List[float]) -> Optional[float]:
        return float(np.mean(lst)) if lst else None

    return {
        "slot_specialization": slot_specialization,
        "slot_specialization_n": sum(len(v) for v in slot_event_gates.values()),
        "thread_separation": _safe_mean(thread_sep_scores),
        "thread_separation_n": len(thread_sep_scores),
        "retroactive_binding": _safe_mean(retro_binding_scores),
        "retroactive_binding_n": len(retro_binding_scores),
        "state_reactivation": _safe_mean(reactivation_scores),
        "state_reactivation_n": len(reactivation_scores),
        "event_chunking": _safe_mean(chunking_ratios),
        "event_chunking_n": len(chunking_ratios),
        "surprise_event_corr": _safe_mean(surprise_event_ratios),
        "surprise_event_corr_n": len(surprise_event_ratios),
    }


def format_diagnostics(report: Dict[str, Any]) -> str:
    """Pretty-print the diagnostics report as a multi-line string with color."""
    mapper = ScoreColorMapper()
    acc_spec = MetricSpec(ScoreDirection.HIGHER_IS_BETTER, 0.0, 1.0)
    entropy_spec = MetricSpec(ScoreDirection.LOWER_IS_BETTER, 1.5, 3.5)

    def _color_val(value: float, spec: MetricSpec) -> str:
        scored = mapper.evaluate("v", value, spec)
        return f"{scored.band.color}{_Ansi.BOLD}{value:.3f}{_Ansi.RESET}"

    lines = ["", "  ===== Diagnostics ====="]
    ent = report.get("mean_episodic_read_entropy")
    if ent is not None:
        lines.append(f"  mean_episodic_read_entropy: {_color_val(ent, entropy_spec)}")
    else:
        lines.append(f"  mean_episodic_read_entropy: N/A")
    lines.append("  who_holds_token by handoffs:")
    for k, v in report["who_holds_token_by_handoffs"].items():
        if v["n"] > 0:
            lines.append(f"    handoffs={k:>2s}  acc={_color_val(v['acc'], acc_spec)}  n={v['n']}")
    lines.append("  who_was_first_tagged by handoffs:")
    for k, v in report["who_was_first_tagged_by_handoffs"].items():
        if v["n"] > 0:
            lines.append(f"    handoffs={k:>2s}  acc={_color_val(v['acc'], acc_spec)}  n={v['n']}")
    lines.append("  what_was_true_rule (belief revision):")
    for k, v in report["what_was_true_rule_by_falsecue"].items():
        if v["n"] > 0:
            lines.append(f"    {k:>12s}  acc={_color_val(v['acc'], acc_spec)}  n={v['n']}")
    # New query types
    for key, label in [
        ("did_trigger_before_alarm", "temporal ordering (trigger<alarm)"),
        ("which_entity_first_occluded", "first occluded entity"),
        ("did_chain2_fire", "chain2 fire"),
        ("which_entity_changed_color", "color change (zero-shot, untrained)"),
    ]:
        entry = report.get(key)
        if entry and entry.get("n", 0) > 0:
            lines.append(f"  {label}:  acc={_color_val(entry['acc'], acc_spec)}  n={entry['n']}")
    extra_metrics = report.get("extra_query_metrics", {})
    if any(v.get("n", 0) > 0 for v in extra_metrics.values()):
        lines.append("  branch-B query metrics:")
        for key, value in extra_metrics.items():
            if value.get("n", 0) > 0:
                lines.append(f"    {key}:  acc={_color_val(value['acc'], acc_spec)}  n={value['n']}")
    # Color change retention curve by lag
    lag_data = report.get("color_change_by_lag", {})
    if any(v.get("n", 0) > 0 for v in lag_data.values()):
        lines.append("  color change by lag (steps since event):")
        for k, v in lag_data.items():
            if v["n"] > 0:
                lines.append(f"    lag={k:>5s}  acc={_color_val(v['acc'], acc_spec)}  n={v['n']}")
    # ── Emergent capability probes ──
    emergent = report.get("emergent_probes")
    if emergent:
        lines.append("  ===== Emergent Capability Probes (zero-shot) =====")
        _probe_defs = [
            ("slot_specialization", "HPM slot specialization",
             "0=uniform, 1=fully specialized; >0.3 interesting"),
            ("thread_separation", "causal thread separation",
             "within-between cos_sim; >0.1 interesting"),
            ("retroactive_binding", "retroactive cause binding",
             "correction↔cue - correction↔random; >0.05 interesting"),
            ("state_reactivation", "state reactivation (replay)",
             "post-surprise↔pre max_sim - baseline; >0 suggestive"),
            ("event_chunking", "event boundary chunking",
             "||Δh|| event/non-event ratio; >1.5 interesting"),
            ("surprise_event_corr", "surprise-event correlation",
             "HPM |z| event/non-event ratio; >1.5 interesting"),
        ]
        for key, label, hint in _probe_defs:
            val = emergent.get(key)
            n = emergent.get(f"{key}_n", 0)
            if val is not None and n > 0:
                lines.append(f"    {label}:  {val:.3f}  (n={n}, {hint})")
            else:
                lines.append(f"    {label}:  N/A")
    lines.append("")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Convenience: build a diagnostic loader and run a one-shot report
# -----------------------------------------------------------------------------
def run_diagnostic_report(
    model,
    query_head,
    world_cfg: WorldConfig,
    tier: CurriculumTier,
    device: str,
    n_episodes: int = 256,
    base_seed: int = 99000,
) -> Dict[str, Any]:
    ds = TMEW1DiagnosticDataset(world_cfg, tier, n=n_episodes, base_seed=base_seed, enable_false_cue=True)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_diag, num_workers=0)
    report = recall_by_difficulty(model, query_head, loader, device, world_cfg.max_entities, enabled=tier.enabled_modalities)
    # Emergent capability probes (separate forward pass, smaller set)
    probe_ds = TMEW1DiagnosticDataset(world_cfg, tier, n=min(64, n_episodes), base_seed=base_seed + 50000, enable_false_cue=True)
    probe_loader = DataLoader(probe_ds, batch_size=8, shuffle=False, collate_fn=collate_diag, num_workers=0)
    report["emergent_probes"] = run_emergent_probes(model, probe_loader, device, enabled=tier.enabled_modalities)
    print(format_diagnostics(report))
    return report


__all__ = [
    "EXTENDED_QUERY_TYPES",
    "EXTENDED_QUERY_TYPE_TO_IDX",
    "get_extended_query_types",
    "get_extended_query_type_to_idx",
    "EpisodeWithDiagnostics",
    "TMEW1DiagnosticDataset",
    "generate_episode_with_diagnostics",
    "episode_to_diag_tensors",
    "collate_diag",
    "recall_by_difficulty",
    "format_diagnostics",
    "run_diagnostic_report",
    "run_emergent_probes",
]
