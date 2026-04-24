"""
TMEW-1: Temporal Multimodal Episodic World, training implementation.

Pairs with homeostatic_multimodal_world_model_chunked.py.

Scope of this draft:
- Tier 1-2 of the curriculum (single/dual modality, short delays, basic occlusion).
- Trigger-delay and occlusion-identity templates.
- Next-step prediction for vision/audio/numeric/text + an auxiliary latent-state probe.
- Curriculum scheduler that promotes tiers when validation accuracy clears a bar.
- Controller stepped BEFORE optimizer.step() so diagnostics match the parameters.

Designed so harder templates (handoff, alarm, false-cue, multi-entity) and longer
delays drop into the same generator interface without trainer changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import math
import json
import random

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader

from homeostatic_multimodal_world_model_chunked import (
    HomeostaticMultimodalWorldModel,
    MultimodalPredictionLoss,
    LossWeights,
    WorldModelConfig,
    ModalityConfig,
    ControllerConfig,
    ForwardOutput,
    LossOutput,
    summarize_controller_report,
)
from score_logging import ScoreLogger, build_default_metric_specs, log_training_snapshot


# -----------------------------------------------------------------------------
# World definition
# -----------------------------------------------------------------------------
@dataclass
class WorldConfig:
    grid_h: int = 16
    grid_w: int = 16
    vision_channels: int = 3
    audio_dim: int = 24
    # 10 global features + 5 identity-grounded features per default entity.
    # The per-entity block is: x, y, visible, color, tagged. This makes
    # entity-id queries observable instead of asking the model to infer hidden
    # simulator indices from color-only vision.
    numeric_dim: int = 40
    text_vocab_size: int = 64
    text_seq_len: int = 1               # one symbolic token per step
    min_entities: int = 2
    max_entities: int = 6
    episode_length: int = 64
    num_latent_rules: int = 6           # latent_state classification target
    occlusion_prob: float = 0.25
    noise_std: float = 0.02
    chain2_frequency_boost: float = 1.0
    chain2_temporal_overlap: bool = False
    rule_dynamic: bool = False


@dataclass
class CurriculumTier:
    tier: int
    max_episode_length: int
    enabled_modalities: Tuple[str, ...]
    template_pool: Tuple[str, ...]
    max_delay: int
    occlusion: bool
    promote_at_accuracy: float          # validation latent-state acc to promote


DEFAULT_TIERS: Tuple[CurriculumTier, ...] = (
    CurriculumTier(1, 24, ("vision", "numeric"), ("trigger_delay",), 3, False, 0.70),
    CurriculumTier(2, 48, ("vision", "numeric", "audio"), ("trigger_delay", "occlusion_identity", "multi_chain"), 8, True, 0.65),
    CurriculumTier(3, 64, ("vision", "numeric", "audio", "text"), ("trigger_delay", "occlusion_identity", "multi_chain"), 12, True, 1.01),
)


# -----------------------------------------------------------------------------
# Latent world simulator
# -----------------------------------------------------------------------------
@dataclass
class Entity:
    id: int
    x: int
    y: int
    dx: int
    dy: int
    color: int      # 0..vision_channels-1
    visible: bool = True
    tagged: bool = False


@dataclass
class WorldState:
    entities: List[Entity]
    active_rule: int                    # which latent rule is currently armed
    alarm_in: int                       # countdown to delayed effect, -1 if none
    alarm_fired: bool
    occluder: Optional[Tuple[int, int, int, int]]  # x0,y0,x1,y1 or None
    rng: random.Random
    # --- second causal chain (multi_chain template) ---
    chain2_alarm_in: int = -1
    chain2_alarm_fired: bool = False
    chain2_trigger_pair: Optional[Tuple[int, int]] = None  # entity ids that triggered chain 2


def _make_world(cfg: WorldConfig, template: str, max_delay: int, allow_occlusion: bool, seed: int) -> WorldState:
    rng = random.Random(seed)
    # Active entities: 2-4 that participate in causal chains
    n_active = rng.randint(2, min(4, cfg.max_entities))
    # Distractor entities: fill up to max_entities with some randomness,
    # while allowing branches to enforce a denser total-entity regime.
    min_total_entities = max(n_active, min(cfg.min_entities, cfg.max_entities))
    min_distractor = max(0, min_total_entities - n_active)
    n_distractor = rng.randint(min_distractor, cfg.max_entities - n_active)
    n = n_active + n_distractor
    entities: List[Entity] = []
    for i in range(n):
        entities.append(
            Entity(
                id=i,
                x=rng.randint(2, cfg.grid_w - 3),
                y=rng.randint(2, cfg.grid_h - 3),
                dx=rng.choice([-1, 0, 1]),
                dy=rng.choice([-1, 0, 1]),
                color=rng.randint(0, cfg.vision_channels - 1),
            )
        )
    occluder = None
    if allow_occlusion and template in ("occlusion_identity", "multi_chain") and rng.random() < cfg.occlusion_prob + 0.4:
        ox = rng.randint(3, cfg.grid_w - 6)
        oy = rng.randint(3, cfg.grid_h - 6)
        occluder = (ox, oy, ox + 3, oy + 3)
    return WorldState(
        entities=entities,
        active_rule=rng.randint(0, cfg.num_latent_rules - 1),
        alarm_in=-1,
        alarm_fired=False,
        occluder=occluder,
        rng=rng,
    )


def _step_world(state: WorldState, cfg: WorldConfig, t: int, template: str, max_delay: int) -> Dict[str, Any]:
    """Advance the world one step. Returns event dict for the renderers."""
    events: Dict[str, Any] = {
        "trigger": False, "alarm_fire": False, "occluded_ids": [],
        "chain2_trigger": False, "chain2_fire": False,
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

    # --- Conditional trigger rules based on active_rule ---
    # Rule 0-1: proximity trigger (classic) — any two entities within distance 1
    # Rule 2-3: same-color trigger — two entities of the same color within distance 2
    # Rule 4-5: tagged-only trigger — only fires if a tagged entity is involved
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
                    # Freeze tagged entity — predictable halt
                    for e in state.entities:
                        if e.tagged:
                            e.dx, e.dy = 0, 0
                elif _r >= 4:
                    # Scatter — all entities get random velocity kick
                    for e in state.entities:
                        e.dx = state.rng.choice([-2, -1, 1, 2])
                        e.dy = state.rng.choice([-2, -1, 1, 2])

    # Chain 2: secondary causal chain (multi_chain template only)
    if template == "multi_chain":
        if not state.chain2_alarm_fired and state.chain2_alarm_in < 0:
            primary_chain_busy = (state.alarm_in > 0) and (not state.alarm_fired)
            if primary_chain_busy and not cfg.chain2_temporal_overlap:
                return events
            chain2_radius = 2 if cfg.chain2_frequency_boost > 1.0 else 1
            chain2_delay_hi = max(3, max_delay // 2)
            if cfg.chain2_frequency_boost >= 1.5:
                chain2_delay_hi = max(2, chain2_delay_hi - 1)
            # Second chain uses a different pair — scan from the end
            for i in range(len(state.entities) - 1, 0, -1):
                for j in range(i - 1, -1, -1):
                    a, b = state.entities[i], state.entities[j]
                    # chain2 always uses simple proximity, independent of active_rule
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


# -----------------------------------------------------------------------------
# Modality renderers
# -----------------------------------------------------------------------------
def _render_vision(state: WorldState, cfg: WorldConfig) -> np.ndarray:
    img = np.zeros((cfg.vision_channels, cfg.grid_h, cfg.grid_w), dtype=np.float32)
    for e in state.entities:
        if not e.visible:
            continue
        img[e.color, e.y, e.x] = 1.0
        if e.tagged:
            img[e.color, e.y, e.x] = 1.0
    if state.occluder is not None:
        ox0, oy0, ox1, oy1 = state.occluder
        img[:, oy0:oy1 + 1, ox0:ox1 + 1] *= 0.0
        img[0, oy0:oy1 + 1, ox0:ox1 + 1] = 0.3
    img += np.random.normal(0, cfg.noise_std, img.shape).astype(np.float32)
    return img


def _render_audio(
    state: WorldState,
    cfg: WorldConfig,
    events: Dict[str, Any],
    current_holder_id: Optional[int] = None,
) -> np.ndarray:
    vec = np.zeros(cfg.audio_dim, dtype=np.float32)
    # Chain 1 audio signals
    if events.get("trigger"):
        vec[0] = 1.0                    # high tone marks armed state
    if state.alarm_in > 0:
        vec[1] = 0.5 + 0.5 * (1.0 / max(1, state.alarm_in))  # rising countdown
    if events.get("alarm_fire"):
        vec[2] = 1.0
    if events.get("handoff"):
        vec[3] = 1.0                    # token changed hands this step
        vec[4] = 1.0                    # duplicate handoff event bit for a cleaner write cue
    # Chain 2 audio signals (channels 14-15)
    if events.get("chain2_trigger"):
        vec[14] = 1.0
    if events.get("chain2_fire"):
        vec[15] = 1.0
    # Holder identity channels at 8+entity_id
    new_holder_id = events.get("new_holder_id", -1)
    if 0 <= new_holder_id < cfg.max_entities and (8 + new_holder_id) < cfg.audio_dim:
        vec[8 + new_holder_id] = 1.0
    if current_holder_id is not None and 0 <= current_holder_id < cfg.max_entities and (8 + current_holder_id) < cfg.audio_dim:
        vec[8 + current_holder_id] = max(vec[8 + current_holder_id], 0.3)
    # Ambient noise
    vec[5:8] = np.random.normal(0, 0.05, 3).astype(np.float32)
    return vec


def _render_numeric(state: WorldState, cfg: WorldConfig) -> np.ndarray:
    vec = np.zeros(cfg.numeric_dim, dtype=np.float32)
    vec[0] = float(len([e for e in state.entities if e.visible])) / max(1, cfg.max_entities)
    vec[1] = float(state.alarm_in) / 10.0 if state.alarm_in > 0 else 0.0
    vec[2] = 1.0 if state.alarm_fired else 0.0
    vec[3] = float(state.active_rule) / max(1, cfg.num_latent_rules - 1)
    vec[4] = float(any(e.tagged for e in state.entities))
    vec[5] = 1.0 if state.occluder is not None else 0.0
    # Expanded channels for multi-chain + distractor awareness
    vec[6] = float(len(state.entities)) / max(1, cfg.max_entities)   # entity count (includes distractors)
    vec[7] = float(state.chain2_alarm_in) / 10.0 if state.chain2_alarm_in > 0 else 0.0
    vec[8] = 1.0 if state.chain2_alarm_fired else 0.0
    vec[9] = float(sum(1 for e in state.entities if e.tagged)) / max(1, len(state.entities))  # fraction tagged
    
    # Identity-grounded per-entity state. Vision uses color planes, so without
    # these channels the simulator's entity ids are hidden labels. Relational
    # queries such as "closest entity to holder at alarm" require id-addressed
    # positions to be in the observation stream. The block for entity i is:
    #   [x_norm, y_norm, visible, color_norm, tagged]
    entity_offset = 10
    entity_stride = 5
    max_x = max(1, cfg.grid_w - 1)
    max_y = max(1, cfg.grid_h - 1)
    max_color = max(1, cfg.vision_channels - 1)
    for e in state.entities:
        base = entity_offset + entity_stride * int(e.id)
        if base + entity_stride > cfg.numeric_dim:
            continue
        vec[base + 0] = float(e.x) / float(max_x)
        vec[base + 1] = float(e.y) / float(max_y)
        vec[base + 2] = 1.0 if e.visible else 0.0
        vec[base + 3] = float(e.color) / float(max_color)
        vec[base + 4] = 1.0 if e.tagged else 0.0
    return vec


def _render_text(state: WorldState, cfg: WorldConfig, events: Dict[str, Any]) -> np.ndarray:
    # Single token per step. Vocab layout:
    # 0 = pad, 1 = quiet, 2 = trigger, 3 = countdown, 4 = fire, 5 = occluded,
    # 6 = chain2_trigger, 7 = chain2_fire
    if events.get("alarm_fire"):
        tok = 4
    elif events.get("chain2_fire"):
        tok = 7
    elif events.get("trigger"):
        tok = 2
    elif events.get("chain2_trigger"):
        tok = 6
    elif state.alarm_in > 0:
        tok = 3
    elif events.get("occluded_ids"):
        tok = 5
    else:
        tok = 1
    return np.array([tok], dtype=np.int64)


# -----------------------------------------------------------------------------
# Episode generation
# -----------------------------------------------------------------------------
@dataclass
class Episode:
    vision: np.ndarray         # (T, C, H, W)
    audio: np.ndarray          # (T, audio_dim)
    numeric: np.ndarray        # (T, numeric_dim)
    text: np.ndarray           # (T, 1) int64
    latent_rule: int           # episode-level label for the auxiliary probe
    alarm_fired_at: int        # -1 if never
    template: str
    length: int


def generate_episode(cfg: WorldConfig, tier: CurriculumTier, seed: int) -> Episode:
    template = random.Random(seed).choice(tier.template_pool)
    state = _make_world(cfg, template, tier.max_delay, tier.occlusion, seed)
    T = tier.max_episode_length
    vision = np.zeros((T, cfg.vision_channels, cfg.grid_h, cfg.grid_w), dtype=np.float32)
    audio = np.zeros((T, cfg.audio_dim), dtype=np.float32)
    numeric = np.zeros((T, cfg.numeric_dim), dtype=np.float32)
    text = np.zeros((T, 1), dtype=np.int64)
    fired_at = -1
    for t in range(T):
        events = _step_world(state, cfg, t, template, tier.max_delay)
        vision[t] = _render_vision(state, cfg)
        audio[t] = _render_audio(state, cfg, events)
        numeric[t] = _render_numeric(state, cfg)
        text[t] = _render_text(state, cfg, events)
        if events.get("alarm_fire") and fired_at < 0:
            fired_at = t
    return Episode(vision, audio, numeric, text, state.active_rule, fired_at, template, T)


# -----------------------------------------------------------------------------
# Dataset / collation
# -----------------------------------------------------------------------------
class TMEW1Dataset(Dataset):
    def __init__(self, cfg: WorldConfig, tier: CurriculumTier, n_episodes: int, base_seed: int = 0):
        self.cfg = cfg
        self.tier = tier
        self.n = n_episodes
        self.base_seed = base_seed

    def set_tier(self, tier: CurriculumTier) -> None:
        self.tier = tier

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ep = generate_episode(self.cfg, self.tier, seed=self.base_seed + idx)
        return {
            "vision": torch.from_numpy(ep.vision),
            "audio": torch.from_numpy(ep.audio),
            "numeric": torch.from_numpy(ep.numeric),
            "text": torch.from_numpy(ep.text).squeeze(-1),
            "latent_rule": torch.tensor(ep.latent_rule, dtype=torch.long),
        }


def collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Tensor]:
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0].keys()}


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
@dataclass
class TrainConfig:
    batch_size: int = 32
    train_episodes_per_tier: int = 2048
    val_episodes: int = 256
    epochs_per_tier: int = 4
    lr: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    aux_latent_weight: float = 0.5
    log_every: int = 10
    warmup_steps: int = 5
    modality_warmup_steps: int = 200     # ramp new-modality loss weight 0→1 over this many steps
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0
    num_workers: int = 0                         # unused with PreCachedDataset
    # --- Stability controls for Tier 2/3 multimodal load ---
    # Per-tier LR multiplier. E.g. {2: 0.5, 3: 0.3} halves LR at tier 2.
    tier_lr_scale: Optional[Dict[int, float]] = None
    # Weight on holder-tracking CE loss. 0.0 = disabled. Default 0.3.
    holder_loss_weight: float = 0.3
    # When False, HPM slot-diversity regularizer is zeroed out.
    diversity_loss_enabled: bool = True


class LatentRuleProbe(nn.Module):
    """Auxiliary head: predict the episode's active latent rule from the final hidden state."""

    def __init__(self, d_model: int, num_rules: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, num_rules))

    def forward(self, sequence: Tensor) -> Tensor:
        # Mean-pool over time, then classify.
        return self.net(sequence.mean(dim=1))


def build_model(world_cfg: WorldConfig, **model_config_overrides) -> Tuple[HomeostaticMultimodalWorldModel, MultimodalPredictionLoss, LatentRuleProbe]:
    modality = ModalityConfig(
        text_vocab_size=world_cfg.text_vocab_size,
        text_pad_id=0,
        vision_channels=world_cfg.vision_channels,
        vision_height=world_cfg.grid_h,
        vision_width=world_cfg.grid_w,
        audio_dim=world_cfg.audio_dim,
        numeric_dim=world_cfg.numeric_dim,
    )
    cfg = WorldModelConfig(
        modality=modality,
        d_model=256,
        num_layers=6,
        num_cohorts=8,
        num_memory_slots=32,
        num_episodic_slots=64,
        controller=ControllerConfig(
            exploit_budget=20.0,
            unlock_stress_threshold=0.18,
            stress_threshold=0.30,
            intervention_interval=6,
            strategic_unlock_fraction=0.35,
        ),
        enable_online_homeostasis=True,
        **model_config_overrides,
    )
    model = HomeostaticMultimodalWorldModel(cfg)
    criterion = MultimodalPredictionLoss(cfg, LossWeights(text=0.3, numeric=1.0, audio=1.0, vision=1.0))
    probe = LatentRuleProbe(cfg.d_model, world_cfg.num_latent_rules)
    return model, criterion, probe


def shift_targets(x: Tensor) -> Tuple[Tensor, Tensor]:
    """Inputs[:T-1] -> targets[1:T]. Returns (input, target)."""
    return x[:, :-1], x[:, 1:]


def train_step(
    model: HomeostaticMultimodalWorldModel,
    criterion: MultimodalPredictionLoss,
    probe: LatentRuleProbe,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, Tensor],
    tcfg: TrainConfig,
    enabled: Tuple[str, ...],
) -> Dict[str, float]:
    optimizer.zero_grad(set_to_none=True)

    vision_in, vision_tgt = shift_targets(batch["vision"])
    audio_in, audio_tgt = shift_targets(batch["audio"])
    numeric_in, numeric_tgt = shift_targets(batch["numeric"])
    text_in, text_tgt = shift_targets(batch["text"])

    output: ForwardOutput = model(
        text_tokens=text_in if "text" in enabled else None,
        vision=vision_in if "vision" in enabled else None,
        audio=audio_in if "audio" in enabled else None,
        numeric=numeric_in if "numeric" in enabled else None,
    )

    losses: LossOutput = criterion(
        output,
        text_targets=text_tgt if "text" in enabled else None,
        vision_targets=vision_tgt if "vision" in enabled else None,
        audio_targets=audio_tgt if "audio" in enabled else None,
        numeric_targets=numeric_tgt if "numeric" in enabled else None,
    )

    rule_logits = probe(output.sequence)
    aux = nn.functional.cross_entropy(rule_logits, batch["latent_rule"])
    total = losses.total + tcfg.aux_latent_weight * aux

    total.backward()
    if tcfg.grad_clip > 0:
        nn.utils.clip_grad_norm_(list(model.parameters()) + list(probe.parameters()), tcfg.grad_clip)
    optimizer.step()

    # controller_step mutates live Parameters, so defer it until autograd is done.
    if model.cfg.enable_online_homeostasis:
        model.controller_step(total.detach())

    with torch.no_grad():
        rule_acc = (rule_logits.argmax(dim=-1) == batch["latent_rule"]).float().mean().item()

    return {
        "total": float(total.item()),
        "aux_latent": float(aux.item()),
        "latent_acc": rule_acc,
        **{f"loss_{k}": float(v.item()) for k, v in losses.parts.items()},
    }


@torch.no_grad()
def evaluate(
    model: HomeostaticMultimodalWorldModel,
    criterion: MultimodalPredictionLoss,
    probe: LatentRuleProbe,
    loader: DataLoader,
    tcfg: TrainConfig,
    enabled: Tuple[str, ...],
) -> Dict[str, float]:
    model.eval()
    probe.eval()
    totals: Dict[str, float] = {}
    n = 0
    for batch in loader:
        batch = {k: v.to(tcfg.device) for k, v in batch.items()}
        vision_in, vision_tgt = shift_targets(batch["vision"])
        audio_in, audio_tgt = shift_targets(batch["audio"])
        numeric_in, numeric_tgt = shift_targets(batch["numeric"])
        text_in, text_tgt = shift_targets(batch["text"])
        output = model(
            text_tokens=text_in if "text" in enabled else None,
            vision=vision_in if "vision" in enabled else None,
            audio=audio_in if "audio" in enabled else None,
            numeric=numeric_in if "numeric" in enabled else None,
        )
        losses = criterion(
            output,
            text_targets=text_tgt if "text" in enabled else None,
            vision_targets=vision_tgt if "vision" in enabled else None,
            audio_targets=audio_tgt if "audio" in enabled else None,
            numeric_targets=numeric_tgt if "numeric" in enabled else None,
        )
        rule_logits = probe(output.sequence)
        aux = nn.functional.cross_entropy(rule_logits, batch["latent_rule"])
        rule_acc = (rule_logits.argmax(dim=-1) == batch["latent_rule"]).float().mean().item()
        bs = batch["latent_rule"].size(0)
        for k, v in losses.parts.items():
            totals[f"loss_{k}"] = totals.get(f"loss_{k}", 0.0) + float(v.item()) * bs
        totals["aux_latent"] = totals.get("aux_latent", 0.0) + float(aux.item()) * bs
        totals["latent_acc"] = totals.get("latent_acc", 0.0) + rule_acc * bs
        n += bs
    model.train()
    probe.train()
    return {k: v / max(1, n) for k, v in totals.items()}


def run_curriculum(world_cfg: WorldConfig, tcfg: TrainConfig, tiers: Sequence[CurriculumTier] = DEFAULT_TIERS) -> None:
    torch.manual_seed(tcfg.seed)
    np.random.seed(tcfg.seed)
    random.seed(tcfg.seed)

    score_logger = ScoreLogger("tmew_train_scores")
    specs = build_default_metric_specs()

    model, criterion, probe = build_model(world_cfg)
    model = model.to(tcfg.device)
    probe = probe.to(tcfg.device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(probe.parameters()),
        lr=tcfg.lr,
        weight_decay=tcfg.weight_decay,
    )

    for tier in tiers:
        print(f"\n=== Tier {tier.tier} | modalities={tier.enabled_modalities} | T={tier.max_episode_length} ===")
        train_ds = TMEW1Dataset(world_cfg, tier, tcfg.train_episodes_per_tier, base_seed=1000 * tier.tier)
        val_ds = TMEW1Dataset(world_cfg, tier, tcfg.val_episodes, base_seed=9000 + 1000 * tier.tier)
        train_loader = DataLoader(train_ds, batch_size=tcfg.batch_size, shuffle=True, collate_fn=collate, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=tcfg.batch_size, shuffle=False, collate_fn=collate, num_workers=0)

        for epoch in range(tcfg.epochs_per_tier):
            _accum_unlocks = 0
            for step, batch in enumerate(train_loader):
                batch = {k: v.to(tcfg.device) for k, v in batch.items()}
                stats = train_step(model, criterion, probe, optimizer, batch, tcfg, tier.enabled_modalities)
                _step_report = summarize_controller_report(model._last_forward_report)
                _accum_unlocks += _step_report.get("unlocked", 0)
                if epoch == 0 and step < tcfg.warmup_steps:
                    continue
                if step % tcfg.log_every == 0:
                    summary = _step_report
                    step_metrics = {
                        "total": stats["total"],
                        "aux_latent": stats["aux_latent"],
                        "latent_acc": stats["latent_acc"],
                    }
                    stresses = summary.get("stresses") or []
                    if stresses:
                        step_metrics["stress"] = float(np.mean(np.asarray(stresses, dtype=np.float32)))
                    pnn_states = summary.get("pnn_states") or []
                    pnn_str = "/".join(s[0] for s in pnn_states) if pnn_states else ""
                    n_open = sum(1 for s in pnn_states if s == "OPEN") if pnn_states else 0
                    unlock_tag = f" (+{_accum_unlocks} unlocked)" if _accum_unlocks > 0 else ""
                    log_training_snapshot(
                        score_logger,
                        step_label=f"=============================tier{tier.tier} ep{epoch} s{step:04d} | pnn={pnn_str} open={n_open}{unlock_tag}======================================",
                        metrics=step_metrics,
                        specs=specs,
                    )
                    _accum_unlocks = 0

            val = evaluate(model, criterion, probe, val_loader, tcfg, tier.enabled_modalities)
            log_training_snapshot(
                score_logger,
                step_label=f"[val] tier{tier.tier} ep{epoch}",
                metrics=val,
                specs=specs,
            )

            if val["latent_acc"] >= tier.promote_at_accuracy:
                print(f"  -> promoting from tier {tier.tier}")
                break


if __name__ == "__main__":
    run_curriculum(WorldConfig(), TrainConfig())
