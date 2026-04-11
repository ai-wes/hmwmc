from __future__ import annotations

"""
Homeostatic multimodal world model
=================================

A neural architecture draft that takes direct implementation inspiration from the
uploaded MOEA / TE-AI components, but remaps them into a stateful neural system.

Key borrowings by design:
- PerineuronalNet OPEN/CLOSING/LOCKED gating with exploitation budgets
- Circadian phase gating with day/night plasticity multipliers
- RD-style per-island/per-unit stress estimation
- CausalTapestry event logging, time-decayed effect lookup, and directional memory
- Immune cache of harmful intervention signatures

This file is intentionally self-contained and importable, but it will prefer the
uploaded component files when available in the same directory.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple
import hashlib
import math
import sys

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Optional imports from the uploaded MOEA-inspired component files.
# -----------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

try:  # Preferred: user's uploaded implementations
    from causal_tapestry import CausalTapestry  # type: ignore
except Exception:  # pragma: no cover
    CausalTapestry = None  # type: ignore

try:
    from pnn import PerineuronalNet, PNN_STATE, PNNConfig  # type: ignore
except Exception:  # pragma: no cover
    PerineuronalNet = None  # type: ignore
    PNN_STATE = None  # type: ignore
    PNNConfig = None  # type: ignore

try:
    from circadian_gate import CircadianGate, CircadianGateConfig  # type: ignore
except Exception:  # pragma: no cover
    CircadianGate = None  # type: ignore
    CircadianGateConfig = None  # type: ignore

try:
    from rd_stress import (  # type: ignore
        RDStressField,
        RDStressConfig,
        StressImmuneLayer,
        StressImmuneConfig,
    )
except Exception:  # pragma: no cover
    RDStressField = None  # type: ignore
    RDStressConfig = None  # type: ignore
    StressImmuneLayer = None  # type: ignore
    StressImmuneConfig = None  # type: ignore


# -----------------------------------------------------------------------------
# Fallbacks if any optional imports are unavailable.
# -----------------------------------------------------------------------------
class _FallbackPNNState(Enum):
    OPEN = "OPEN"
    CLOSING = "CLOSING"
    LOCKED = "LOCKED"


@dataclass(frozen=True)
class _FallbackPNNConfig:
    lock_stability_window: int = 5
    lock_rel_change: float = 0.05
    closing_generations: int = 5
    min_plasticity: float = 0.2
    time_decay_per_gen: float = 0.2
    stagnation_penalty_threshold: int = 6
    stagnation_penalty_multiplier: float = 2.0
    improvement_eps: float = 1e-9
    drain_success_factor: float = 0.5
    drain_failure_factor: float = 1.5


class _FallbackPerineuronalNet:
    def __init__(self, cell_id: str, exploit_budget: float = 100.0, config: _FallbackPNNConfig = _FallbackPNNConfig()):
        self.cfg = config
        self.cell_id = cell_id
        self.state = _FallbackPNNState.OPEN
        self.generations_in_phase = 0
        self.plasticity_multiplier = 1.0
        self.stability_history: List[float] = []
        self.refractory_until: Optional[int] = None
        self.exploit_budget = float(exploit_budget)
        self.initial_exploit_budget = float(exploit_budget)
        self.fitness_at_lock: Optional[float] = None
        self.best_fitness_in_lock_phase: Optional[float] = None
        self.recent_exploit_evals: float = 0.0
        self.recent_exploit_improvement: float = 0.0

    def note_exploit_outcome(self, evals_used: float, improvement: float) -> None:
        self.recent_exploit_evals = float(max(0.0, evals_used))
        self.recent_exploit_improvement = float(improvement)

    def update(self, current_fitness: float, generation: int, island_median_fitness: Optional[float] = None) -> None:
        if self.refractory_until is not None and generation < self.refractory_until:
            self.generations_in_phase = 0
            return
        if self.refractory_until is not None and generation >= self.refractory_until:
            self.refractory_until = None

        self.generations_in_phase += 1
        self.stability_history.append(float(current_fitness))
        self.stability_history = self.stability_history[-10:]

        if self.state == _FallbackPNNState.OPEN:
            if len(self.stability_history) >= self.cfg.lock_stability_window:
                recent = self.stability_history[-self.cfg.lock_stability_window:]
                mean = float(np.mean(recent))
                rel = abs(float(current_fitness) - mean) / max(1e-8, abs(mean))
                if rel < self.cfg.lock_rel_change:
                    if island_median_fitness is None or current_fitness <= island_median_fitness:
                        self.state = _FallbackPNNState.CLOSING
                        self.generations_in_phase = 0
                    else:
                        self.generations_in_phase = 0
        elif self.state == _FallbackPNNState.CLOSING:
            progress = min(1.0, self.generations_in_phase / float(self.cfg.closing_generations))
            self.plasticity_multiplier = max(self.cfg.min_plasticity, 1.0 - (1.0 - self.cfg.min_plasticity) * progress)
            if progress >= 1.0:
                self.state = _FallbackPNNState.LOCKED
                self.plasticity_multiplier = self.cfg.min_plasticity
                self.generations_in_phase = 0
                self.fitness_at_lock = float(current_fitness)
                self.best_fitness_in_lock_phase = float(current_fitness)
                self.exploit_budget = float(self.initial_exploit_budget)
        else:
            self.plasticity_multiplier = self.cfg.min_plasticity
            if self.best_fitness_in_lock_phase is None:
                self.best_fitness_in_lock_phase = float(current_fitness)
                stagn = 0
            elif current_fitness < self.best_fitness_in_lock_phase:
                self.best_fitness_in_lock_phase = float(current_fitness)
                stagn = 0
            else:
                stagn = self.generations_in_phase
            time_decay = self.cfg.time_decay_per_gen
            if stagn >= self.cfg.stagnation_penalty_threshold:
                stagn_decay = self.cfg.stagnation_penalty_multiplier * (1.0 + (stagn / float(self.cfg.stagnation_penalty_threshold)))
            else:
                stagn_decay = 0.2 * (stagn / float(self.cfg.stagnation_penalty_threshold))
            used = float(self.recent_exploit_evals)
            imp = float(self.recent_exploit_improvement)
            success = imp > self.cfg.improvement_eps
            usage_decay = (self.cfg.drain_success_factor if success else self.cfg.drain_failure_factor) * used
            self.exploit_budget = max(0.0, self.exploit_budget - (time_decay + stagn_decay + usage_decay))
            self.recent_exploit_evals *= 0.5
            self.recent_exploit_improvement *= 0.5

    def force_unlock(self, generation: int, refractory_period: int = 5) -> None:
        recent = self.stability_history[-5:]
        stress_proxy = 0.0
        if recent:
            avg = float(np.mean(recent))
            std = float(np.std(recent))
            if avg > 1e-8:
                stress_proxy = min(1.0, std / avg)
        self.state = _FallbackPNNState.OPEN
        self.plasticity_multiplier = 1.0
        self.generations_in_phase = 0
        self.stability_history = []
        self.fitness_at_lock = None
        self.best_fitness_in_lock_phase = None
        base_ref = int(refractory_period)
        self.refractory_until = generation + max(refractory_period, int(round(base_ref * (1.0 + 1.5 * stress_proxy))))


@dataclass(frozen=True)
class _FallbackCircadianGateConfig:
    period: int = 8
    day_fraction_static: float = 0.5
    day_fraction_highd: float = 0.6
    high_dim_threshold: int = 256
    phase_jitter: float = 0.1
    day_plasticity: float = 1.0
    night_plasticity: float = 0.6


class _FallbackCircadianGate:
    def __init__(self, config: _FallbackCircadianGateConfig, island_id: int, num_islands: int, dimension: Optional[int] = None):
        self.config = config
        self.period = max(1, int(config.period))
        self.phase_offset = (float(island_id) / max(1, int(num_islands))) * float(self.period)
        self.dimension = dimension
        self.is_day = True
        self.plasticity_multiplier = config.day_plasticity

    def update(self, generation: int) -> None:
        frac = self.config.day_fraction_highd if (self.dimension is not None and self.dimension >= self.config.high_dim_threshold) else self.config.day_fraction_static
        t = (float(generation) + self.phase_offset) % float(self.period)
        self.is_day = t < (float(self.period) * frac)
        self.plasticity_multiplier = self.config.day_plasticity if self.is_day else self.config.night_plasticity


@dataclass(frozen=True)
class _FallbackRDStressConfig:
    stress_threshold: float = 0.7
    default_stress: float = 0.3


class _FallbackRDStressField:
    def __init__(self, config: _FallbackRDStressConfig):
        self.cfg = config
        self.stress_values: Dict[str, float] = {}

    def update(self, population_by_island: Dict[int, Sequence[Any]], generation: int) -> None:
        for isl, cells in population_by_island.items():
            fit = []
            for c in cells:
                fh = getattr(c, "fitness_history", None)
                if fh:
                    fit.append(float(fh[-1]))
            if len(fit) >= 2:
                mean = float(np.mean(fit))
                std = float(np.std(fit))
                base_stress = min(std / max(1e-6, abs(mean)), self.cfg.stress_threshold)
            else:
                base_stress = self.cfg.default_stress
            island_mean = float(np.mean(fit)) if fit else None
            for c in cells:
                fh = getattr(c, "fitness_history", None)
                if fh and island_mean is not None:
                    recent = float(fh[-1])
                    denom = max(1e-6, abs(island_mean))
                    relative_stress = 1.0 - min(1.0, max(0.0, recent / denom))
                else:
                    relative_stress = 0.0
                stress = float(np.clip(0.7 * base_stress + 0.3 * relative_stress, 0.0, 1.0))
                setattr(c, "local_stress", stress)
                self.stress_values[f"{isl}_{getattr(c, 'id', 'unknown')}"] = stress


@dataclass(frozen=True)
class _FallbackStressImmuneConfig:
    max_cache: int = 128


class _FallbackStressImmuneLayer:
    def __init__(self, config: _FallbackStressImmuneConfig = _FallbackStressImmuneConfig()):
        self.cfg = config
        self.memory_cells: Dict[str, bool] = {}
        self._order: List[str] = []

    def add_memory(self, signature: str) -> None:
        if signature in self.memory_cells:
            if signature in self._order:
                self._order.remove(signature)
            self._order.append(signature)
            return
        self.memory_cells[signature] = True
        self._order.append(signature)
        while len(self._order) > self.cfg.max_cache:
            old = self._order.pop(0)
            self.memory_cells.pop(old, None)

    def has_memory(self, signature: str) -> bool:
        return signature in self.memory_cells


class _FallbackTapestry:
    def __init__(self):
        self.events: List[Dict[str, Any]] = []

    def add_event_node(self, event_id: str, event_type: str, generation: int, details: Dict[str, Any]) -> None:
        self.events.append({
            "id": event_id,
            "event_type": event_type,
            "generation": generation,
            "details": dict(details),
            "effect": float(details.get("effect", 0.0)),
        })

    def query_action_effect_with_stats(self, action: str, context_filters: Dict[str, Any], generation_window: int = 10, decay_rate: float = 0.0) -> Dict[str, float]:
        vals = []
        for event in self.events:
            details = event.get("details", {})
            if details.get("action") != action:
                continue
            if all(details.get(k) == v for k, v in context_filters.items()):
                vals.append(float(event.get("effect", 0.0)))
        if not vals:
            return {"effect": 0.0, "count": 0, "std": 0.0, "min": 0.0, "max": 0.0}
        arr = np.asarray(vals, dtype=np.float32)
        return {
            "effect": float(arr.mean()),
            "count": int(arr.size),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    def query_action_effect(self, *args: Any, **kwargs: Any) -> float:
        return self.query_action_effect_with_stats(*args, **kwargs).get("effect", 0.0)

    def query_causal_direction(self, action: str, context: Dict[str, Any]) -> Optional[np.ndarray]:
        return None


# Normalized symbols used by the model implementation.
if PerineuronalNet is None:
    PerineuronalNet = _FallbackPerineuronalNet  # type: ignore
if PNN_STATE is None:
    PNN_STATE = _FallbackPNNState  # type: ignore
if PNNConfig is None:
    PNNConfig = _FallbackPNNConfig  # type: ignore
if CircadianGate is None:
    CircadianGate = _FallbackCircadianGate  # type: ignore
if CircadianGateConfig is None:
    CircadianGateConfig = _FallbackCircadianGateConfig  # type: ignore
if RDStressField is None:
    RDStressField = _FallbackRDStressField  # type: ignore
if RDStressConfig is None:
    RDStressConfig = _FallbackRDStressConfig  # type: ignore
if StressImmuneLayer is None:
    StressImmuneLayer = _FallbackStressImmuneLayer  # type: ignore
if StressImmuneConfig is None:
    StressImmuneConfig = _FallbackStressImmuneConfig  # type: ignore
if CausalTapestry is None:
    CausalTapestry = _FallbackTapestry  # type: ignore


# -----------------------------------------------------------------------------
# Configuration dataclasses.
# -----------------------------------------------------------------------------
@dataclass
class ModalityConfig:
    text_vocab_size: int = 512
    text_pad_id: int = 0
    vision_channels: int = 3
    vision_height: int = 16
    vision_width: int = 16
    audio_dim: int = 32
    numeric_dim: int = 16


@dataclass
class ControllerConfig:
    intervention_interval: int = 8
    exploit_budget: float = 64.0
    stress_threshold: float = 0.70
    unlock_stress_threshold: float = 0.85
    intervention_scale: float = 0.05
    action_ucb_c: float = 0.35
    action_discount: float = 0.97
    action_reward_clip: float = 1.0
    strategic_unlock_fraction: float = 0.25
    max_interventions_per_step: int = 2
    harm_effect_threshold: float = 1e-3
    tapestry_generation_window: int = 128
    emergency_residual_floor: float = 0.10
    emergency_forget_ceiling: float = 0.98

    # Layer scoring targets.
    target_density: float = 0.15
    target_attn_entropy_scale: float = 0.60
    residual_ratio_ceiling: float = 0.75
    stale_ratio_ceiling: float = 1.25
    density_penalty_weight: float = 0.20
    residual_penalty_weight: float = 0.20
    stale_penalty_weight: float = 0.20
    entropy_penalty_weight: float = 0.05


@dataclass
class WorldModelConfig:
    d_model: int = 256
    num_layers: int = 6
    latent_multiplier: int = 4
    num_cohorts: int = 8
    num_memory_slots: int = 8
    num_episodic_slots: int = 16
    dropout: float = 0.10
    max_text_tokens_per_step: int = 12
    modality: ModalityConfig = field(default_factory=ModalityConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)

    # Prediction heads
    predict_text: bool = True
    predict_numeric: bool = True
    predict_audio: bool = True
    predict_vision: bool = True

    # Event memory settings
    surprise_event_threshold: float = 0.25
    event_write_min_strength: float = 0.05

    # Runtime behavior
    enable_online_homeostasis: bool = True
    detach_state_between_steps: bool = False
    scan_chunk_size: int = 16


# -----------------------------------------------------------------------------
# Small utilities.
# -----------------------------------------------------------------------------
def _as_float(x: Tensor | float | int) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().float().mean().item())
    return float(x)


def _safe_mean(values: Sequence[float], default: float = 0.0) -> float:
    if not values:
        return default
    return float(np.mean(np.asarray(values, dtype=np.float32)))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


# -----------------------------------------------------------------------------
# Encoders.
# -----------------------------------------------------------------------------
class TextEncoder(nn.Module):
    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.modality.text_vocab_size, cfg.d_model, padding_idx=cfg.modality.text_pad_id)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, text_tokens: Tensor) -> Tensor:
        if text_tokens.dim() == 2:
            return self.proj(self.embedding(text_tokens))
        if text_tokens.dim() != 3:
            raise ValueError(f"text_tokens must have shape [B,T] or [B,T,L], got {tuple(text_tokens.shape)}")
        emb = self.embedding(text_tokens)
        mask = (text_tokens != self.cfg.modality.text_pad_id).float().unsqueeze(-1)
        denom = mask.sum(dim=2).clamp_min(1.0)
        pooled = (emb * mask).sum(dim=2) / denom
        return self.proj(pooled)


class VisionEncoder(nn.Module):
    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        c = cfg.modality.vision_channels
        d = cfg.d_model
        self.net = nn.Sequential(
            nn.Conv2d(c, d // 4, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(d // 4, d // 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(d // 2, d)

    def forward(self, vision: Tensor) -> Tensor:
        if vision.dim() != 5:
            raise ValueError(f"vision must have shape [B,T,C,H,W], got {tuple(vision.shape)}")
        b, t, c, h, w = vision.shape
        x = vision.reshape(b * t, c, h, w)
        feat = self.net(x).flatten(1)
        feat = self.proj(feat)
        return feat.view(b, t, -1)


class AudioEncoder(nn.Module):
    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        d_in = cfg.modality.audio_dim
        d = cfg.d_model
        self.net = nn.Sequential(
            nn.Linear(d_in, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

    def forward(self, audio: Tensor) -> Tensor:
        if audio.dim() != 3:
            raise ValueError(f"audio must have shape [B,T,F], got {tuple(audio.shape)}")
        return self.net(audio)


class NumericEncoder(nn.Module):
    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        d_in = cfg.modality.numeric_dim
        d = cfg.d_model
        self.net = nn.Sequential(
            nn.Linear(d_in, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

    def forward(self, numeric: Tensor) -> Tensor:
        if numeric.dim() != 3:
            raise ValueError(f"numeric must have shape [B,T,F], got {tuple(numeric.shape)}")
        return self.net(numeric)


class GatedModalityFusion(nn.Module):
    MODALITY_ORDER = ("text", "vision", "audio", "numeric")

    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        self.cfg = cfg
        self.modality_embedding = nn.Embedding(len(self.MODALITY_ORDER), cfg.d_model)
        self.gate = nn.Linear(cfg.d_model, 1)
        self.out_norm = RMSNorm(cfg.d_model)

    def forward(self, features: Mapping[str, Optional[Tensor]]) -> Tuple[Tensor, Dict[str, Tensor]]:
        present: List[Tensor] = []
        names: List[str] = []
        for idx, name in enumerate(self.MODALITY_ORDER):
            feat = features.get(name)
            if feat is None:
                continue
            mod_emb = self.modality_embedding.weight[idx].view(1, 1, -1)
            present.append(feat + mod_emb)
            names.append(name)
        if not present:
            raise ValueError("At least one modality must be provided to the fusion module.")

        stack = torch.stack(present, dim=2)  # [B,T,M,D]
        logits = self.gate(torch.tanh(stack)).squeeze(-1)  # [B,T,M]
        weights = torch.softmax(logits, dim=-1)
        fused = (stack * weights.unsqueeze(-1)).sum(dim=2)
        fused = self.out_norm(fused)
        weight_map = {name: weights[..., i] for i, name in enumerate(names)}
        return fused, weight_map


# -----------------------------------------------------------------------------
# Stateful sparse world block.
# -----------------------------------------------------------------------------
class AdaptiveSparseWorldBlock(nn.Module):
    def __init__(self, cfg: WorldModelConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.d_model = cfg.d_model
        self.latent_dim = cfg.d_model * cfg.latent_multiplier
        if self.latent_dim % cfg.num_cohorts != 0:
            raise ValueError("latent_dim must be divisible by num_cohorts")
        self.num_cohorts = cfg.num_cohorts
        self.cohort_dim = self.latent_dim // self.num_cohorts

        d = cfg.d_model
        self.input_norm = RMSNorm(d)
        self.output_norm = RMSNorm(d)
        self.state_norm = RMSNorm(d)
        self.memory_norm = RMSNorm(d)

        self.read_q = nn.Linear(d, d, bias=False)
        self.read_k = nn.Linear(d, d, bias=False)
        self.read_v = nn.Linear(d, d, bias=False)

        self.latent_a = nn.Linear(d, self.latent_dim)
        self.latent_b = nn.Linear(d, self.latent_dim)
        self.decode = nn.Linear(self.latent_dim, d)
        self.transition = nn.Linear(3 * d, d)
        self.state_cell = nn.GRUCell(d, d)
        self.dropout = nn.Dropout(cfg.dropout)

        # Homeostatic control surfaces remain directly learnable by SGD while
        # still being adjustable by the controller under no_grad().
        self.threshold_a = nn.Parameter(torch.zeros(self.num_cohorts))
        self.threshold_b = nn.Parameter(torch.zeros(self.num_cohorts))
        self.gain_a = nn.Parameter(torch.ones(self.num_cohorts))
        self.gain_b = nn.Parameter(torch.ones(self.num_cohorts))
        self.mult_gate = nn.Parameter(torch.ones(self.num_cohorts))
        self.residual_gate = nn.Parameter(torch.ones(1))
        self.write_alpha = nn.Parameter(torch.ones(1) * 0.35)
        self.forget_lambda = nn.Parameter(torch.ones(1) * 0.90)

    def _apply_sparse_controls(self, x: Tensor, threshold: Tensor, gain: Tensor) -> Tuple[Tensor, Tensor]:
        lead_shape = x.shape[:-1]
        x = x.reshape(-1, self.num_cohorts, self.cohort_dim)
        threshold = threshold.view(1, self.num_cohorts, 1)
        gain = gain.view(1, self.num_cohorts, 1)
        y = F.relu(x - threshold) * gain
        density = (y > 0).float().mean(dim=(0, 2))
        return y.reshape(*lead_shape, self.latent_dim), density

    def _read_memory(self, query: Tensor, memory_slots: Tensor) -> Tuple[Tensor, Tensor]:
        q = self.read_q(query).unsqueeze(1)
        mem = self.memory_norm(memory_slots)
        k = self.read_k(mem)
        v = self.read_v(mem)
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_model)
        weights = torch.softmax(logits, dim=-1)
        context = torch.matmul(weights, v).squeeze(1)
        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=-1).mean()
        return context, entropy

    def _update_memory(
        self,
        hidden_state: Tensor,
        memory_slots: Tensor,
        write_alpha_t: Optional[Tensor] = None,
        forget_t: Optional[Tensor] = None,
    ) -> Tensor:
        if write_alpha_t is None:
            write_alpha_t = self.write_alpha.mean()
        if forget_t is None:
            forget_t = self.forget_lambda.mean()
        decayed = memory_slots * forget_t
        new_slot = (F.normalize(hidden_state, dim=-1) * write_alpha_t).unsqueeze(1)
        shifted = decayed[:, :-1, :]
        return torch.cat([new_slot, shifted], dim=1)

    @torch.no_grad()
    def clamp_controls(self) -> None:
        self.threshold_a.clamp_(-2.0, 2.0)
        self.threshold_b.clamp_(-2.0, 2.0)
        self.gain_a.clamp_(0.25, 4.0)
        self.gain_b.clamp_(0.25, 4.0)
        self.mult_gate.clamp_(0.0, 2.0)
        self.residual_gate.clamp_(0.05, 1.50)
        self.write_alpha.clamp_(0.05, 1.00)
        self.forget_lambda.clamp_(0.50, 0.995)

    def get_control_snapshot(self) -> Dict[str, float]:
        return {
            "threshold_a": _as_float(self.threshold_a.mean()),
            "threshold_b": _as_float(self.threshold_b.mean()),
            "gain_a": _as_float(self.gain_a.mean()),
            "gain_b": _as_float(self.gain_b.mean()),
            "mult_gate": _as_float(self.mult_gate.mean()),
            "residual_gate": _as_float(self.residual_gate.mean()),
            "write_alpha": _as_float(self.write_alpha.mean()),
            "forget_lambda": _as_float(self.forget_lambda.mean()),
        }

    def forward_step(self, x_t: Tensor, hidden_state: Tensor, memory_slots: Tensor) -> Tuple[Tensor, Tensor, Tensor, Dict[str, float]]:
        x_norm = self.input_norm(x_t)
        context, attn_entropy = self._read_memory(x_norm, memory_slots)
        ctx = self.state_norm(context + hidden_state)

        a_latent = self.latent_a(x_norm)
        b_latent = self.latent_b(ctx)
        a_sparse, density_a = self._apply_sparse_controls(a_latent, self.threshold_a, self.gain_a)
        b_sparse, density_b = self._apply_sparse_controls(b_latent, self.threshold_b, self.gain_b)

        bsz = x_t.size(0)
        mult = a_sparse.view(bsz, self.num_cohorts, self.cohort_dim)
        mult = mult * b_sparse.view(bsz, self.num_cohorts, self.cohort_dim)
        mult = mult * self.mult_gate.view(1, self.num_cohorts, 1)
        mult_flat = self.dropout(mult.flatten(1))

        delta = self.decode(mult_flat)
        residual_scale = self.residual_gate.mean()
        x_out = self.output_norm(x_t + residual_scale * delta)

        proposal = torch.cat([x_norm, context, hidden_state], dim=-1)
        proposal = torch.tanh(self.transition(proposal))
        candidate_state = self.state_cell(proposal, hidden_state)
        write_alpha_t = self.write_alpha.mean()
        forget_t = self.forget_lambda.mean()
        next_hidden = forget_t * hidden_state + write_alpha_t * candidate_state
        next_memory = self._update_memory(next_hidden, memory_slots, write_alpha_t, forget_t)

        with torch.no_grad():
            residual_ratio = delta.norm(dim=-1).mean() / x_t.norm(dim=-1).mean().clamp_min(1e-6)
            stale_ratio = memory_slots[:, 1:, :].norm(dim=-1).mean() / memory_slots[:, :1, :].norm(dim=-1).mean().clamp_min(1e-6)
            diagnostics = {
                "density_a": _as_float(density_a.mean()),
                "density_b": _as_float(density_b.mean()),
                "attn_entropy": _as_float(attn_entropy),
                "residual_ratio": _as_float(residual_ratio),
                "hidden_norm": _as_float(next_hidden.norm(dim=-1).mean()),
                "stale_ratio": _as_float(stale_ratio),
                "write_alpha": _as_float(write_alpha_t),
                "forget_lambda": _as_float(forget_t),
                "residual_scale": _as_float(residual_scale),
                "mult_gate": _as_float(self.mult_gate.mean()),
            }
        return x_out, next_hidden, next_memory, diagnostics

    def forward_chunk(
        self,
        x_chunk: Tensor,
        hidden_state: Tensor,
        memory_slots: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, float]]]:
        """
        Chunked recurrent scan over a short time window.

        Hoists all input-only projections out of the inner recurrence and keeps
        only the memory/state dependent path sequential. This preserves exact
        recurrence semantics while removing a large amount of Python dispatch and
        repeated projection work from the hot path.
        """
        bsz, steps, _ = x_chunk.shape
        x_norm_chunk = self.input_norm(x_chunk)
        a_latent_chunk = self.latent_a(x_norm_chunk)
        a_sparse_chunk, _ = self._apply_sparse_controls(a_latent_chunk, self.threshold_a, self.gain_a)

        residual_scale = self.residual_gate.mean()
        write_alpha_t = self.write_alpha.mean()
        forget_t = self.forget_lambda.mean()
        mult_gate = self.mult_gate.view(1, self.num_cohorts, 1)

        outputs: List[Tensor] = []
        diagnostics: List[Dict[str, float]] = []
        for step in range(steps):
            x_t = x_chunk[:, step, :]
            x_norm_t = x_norm_chunk[:, step, :]
            a_sparse_t = a_sparse_chunk[:, step, :]

            context, attn_entropy = self._read_memory(x_norm_t, memory_slots)
            ctx = self.state_norm(context + hidden_state)

            b_latent = self.latent_b(ctx)
            b_sparse, density_b = self._apply_sparse_controls(b_latent, self.threshold_b, self.gain_b)

            mult = a_sparse_t.view(bsz, self.num_cohorts, self.cohort_dim)
            mult = mult * b_sparse.view(bsz, self.num_cohorts, self.cohort_dim)
            mult = mult * mult_gate
            mult_flat = self.dropout(mult.flatten(1))

            delta = self.decode(mult_flat)
            x_out = self.output_norm(x_t + residual_scale * delta)

            proposal = torch.cat([x_norm_t, context, hidden_state], dim=-1)
            proposal = torch.tanh(self.transition(proposal))
            candidate_state = self.state_cell(proposal, hidden_state)
            next_hidden = forget_t * hidden_state + write_alpha_t * candidate_state
            next_memory = self._update_memory(next_hidden, memory_slots, write_alpha_t, forget_t)

            with torch.no_grad():
                density_a_t = (a_sparse_t.view(bsz, self.num_cohorts, self.cohort_dim) > 0).float().mean(dim=(0, 2))
                residual_ratio = delta.norm(dim=-1).mean() / x_t.norm(dim=-1).mean().clamp_min(1e-6)
                stale_ratio = memory_slots[:, 1:, :].norm(dim=-1).mean() / memory_slots[:, :1, :].norm(dim=-1).mean().clamp_min(1e-6)
                diagnostics.append({
                    "density_a": _as_float(density_a_t.mean()),
                    "density_b": _as_float(density_b.mean()),
                    "attn_entropy": _as_float(attn_entropy),
                    "residual_ratio": _as_float(residual_ratio),
                    "hidden_norm": _as_float(next_hidden.norm(dim=-1).mean()),
                    "stale_ratio": _as_float(stale_ratio),
                    "write_alpha": _as_float(write_alpha_t),
                    "forget_lambda": _as_float(forget_t),
                    "residual_scale": _as_float(residual_scale),
                    "mult_gate": _as_float(self.mult_gate.mean()),
                })

            outputs.append(x_out)
            hidden_state = next_hidden
            memory_slots = next_memory

        return torch.stack(outputs, dim=1), hidden_state, memory_slots, diagnostics


# -----------------------------------------------------------------------------
# Episodic memory.
# -----------------------------------------------------------------------------
class EpisodicMemoryBank(nn.Module):
    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.write_proj = nn.Linear(2 * d, d)
        self.read_q = nn.Linear(d, d, bias=False)
        self.read_k = nn.Linear(d, d, bias=False)
        self.read_v = nn.Linear(d, d, bias=False)
        self.norm = RMSNorm(d)

    def read(self, query: Tensor, episodic_memory: Tensor) -> Tuple[Tensor, Tensor]:
        q = self.read_q(query).unsqueeze(1)
        k = self.read_k(episodic_memory)
        v = self.read_v(episodic_memory)
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(query.size(-1))
        weights = torch.softmax(logits, dim=-1)
        context = torch.matmul(weights, v).squeeze(1)
        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=-1).mean()
        return context, entropy

    def write(self, state_vec: Tensor, episodic_memory: Tensor, event_strength: Tensor) -> Tensor:
        event_strength = event_strength.clamp(0.0, 1.0).view(state_vec.size(0), 1, 1)
        candidate = self.norm(self.write_proj(torch.cat([state_vec, episodic_memory[:, 0, :]], dim=-1)))
        blended_front = (
            (1.0 - event_strength) * episodic_memory[:, :1, :]
            + event_strength * candidate.unsqueeze(1)
        )
        older = episodic_memory[:, :-1, :]
        return torch.cat([blended_front, older], dim=1)


# -----------------------------------------------------------------------------
# Controller support types.
# -----------------------------------------------------------------------------
@dataclass
class _LayerProxyUnit:
    id: str
    fitness_history: List[float] = field(default_factory=list)
    local_stress: float = 0.0


@dataclass
class PendingIntervention:
    step: int
    layer_idx: int
    action: str
    pre_score: float
    signature: str
    context: Dict[str, Any]


class ContextualOperatorBandit:
    def __init__(self, operators: Sequence[str], exploration_factor: float = 0.35, discount: float = 0.97):
        self.operators = list(operators)
        self.exploration_factor = float(exploration_factor)
        self.discount = float(discount)
        self.state: Dict[Tuple[Any, ...], Dict[str, Dict[str, float]]] = {}
        self.total: Dict[Tuple[Any, ...], float] = {}

    def _ensure(self, key: Tuple[Any, ...]) -> None:
        if key not in self.state:
            self.state[key] = {op: {"count": 0.0, "value": 0.0} for op in self.operators}
            self.total[key] = 0.0

    def _cold_start_order(self, key: Tuple[Any, ...]) -> List[str]:
        def stable_key(op: str) -> str:
            return hashlib.sha256(f"{repr(key)}::{op}".encode("utf-8")).hexdigest()
        return sorted(self.operators, key=stable_key)

    def select(self, key: Tuple[Any, ...], score_overrides: Optional[Mapping[str, float]] = None) -> str:
        self._ensure(key)
        for op in self._cold_start_order(key):
            if self.state[key][op]["count"] < 1.0:
                return op
        total = max(1.0, self.total[key])
        best_op = self.operators[0]
        best_score = -1e18
        for op, rec in self.state[key].items():
            mean = rec["value"] / max(1e-8, rec["count"])
            bonus = self.exploration_factor * math.sqrt(math.log(total + 1.0) / max(1e-8, rec["count"]))
            override = float(score_overrides.get(op, 0.0)) if score_overrides is not None else 0.0
            score = mean + bonus + override
            if score > best_score:
                best_score = score
                best_op = op
        return best_op

    def update(self, key: Tuple[Any, ...], op: str, reward: float) -> None:
        self._ensure(key)
        for candidate in self.operators:
            self.state[key][candidate]["count"] *= self.discount
            self.state[key][candidate]["value"] *= self.discount
        self.state[key][op]["count"] += 1.0
        self.state[key][op]["value"] += float(reward)
        self.total[key] = self.total.get(key, 0.0) * self.discount + 1.0


# -----------------------------------------------------------------------------
# Homeostatic controller.
# -----------------------------------------------------------------------------
class HomeostaticController:
    OPERATORS: Tuple[str, ...] = (
        "noop",
        "raise_threshold_a",
        "lower_threshold_a",
        "raise_threshold_b",
        "lower_threshold_b",
        "increase_gain_a",
        "decrease_gain_a",
        "increase_gain_b",
        "decrease_gain_b",
        "increase_mult_gate",
        "decrease_mult_gate",
        "increase_residual",
        "decrease_residual",
        "increase_write",
        "decrease_write",
        "increase_forget",
        "decrease_forget",
    )

    def __init__(self, blocks: Sequence[AdaptiveSparseWorldBlock], cfg: WorldModelConfig):
        self.cfg = cfg
        self.blocks = list(blocks)
        self.tapestry = CausalTapestry()
        self.bandit = ContextualOperatorBandit(self.OPERATORS, cfg.controller.action_ucb_c, cfg.controller.action_discount)

        pnn_cfg = PNNConfig() if callable(PNNConfig) else None  # type: ignore
        self.pnns = [
            PerineuronalNet(cell_id=f"layer_{i}", exploit_budget=cfg.controller.exploit_budget, config=pnn_cfg) if pnn_cfg is not None else PerineuronalNet(cell_id=f"layer_{i}", exploit_budget=cfg.controller.exploit_budget)  # type: ignore[arg-type]
            for i in range(len(self.blocks))
        ]
        circ_cfg = CircadianGateConfig() if callable(CircadianGateConfig) else None  # type: ignore
        self.circadian = [
            CircadianGate(circ_cfg, island_id=i, num_islands=len(self.blocks), dimension=cfg.d_model) if circ_cfg is not None else CircadianGate(island_id=i, num_islands=len(self.blocks), dimension=cfg.d_model)  # type: ignore[arg-type]
            for i in range(len(self.blocks))
        ]
        rd_cfg = RDStressConfig(stress_threshold=cfg.controller.stress_threshold, default_stress=min(0.3, cfg.controller.stress_threshold)) if callable(RDStressConfig) else None  # type: ignore
        self.stress_field = RDStressField(rd_cfg) if rd_cfg is not None else RDStressField()  # type: ignore[arg-type]
        immune_cfg = StressImmuneConfig(max_cache=128) if callable(StressImmuneConfig) else None  # type: ignore
        self.immune = [StressImmuneLayer(immune_cfg) if immune_cfg is not None else StressImmuneLayer() for _ in range(len(self.blocks))]  # type: ignore[arg-type]

        self.proxies = [_LayerProxyUnit(id=f"layer_{i}") for i in range(len(self.blocks))]
        self.pending: Dict[int, PendingIntervention] = {}
        self.last_scores: List[float] = [0.0 for _ in range(len(self.blocks))]
        self.last_diagnostics: List[Dict[str, float]] = [{} for _ in range(len(self.blocks))]
        self.global_step: int = 0
        self.last_action_report: List[Dict[str, Any]] = []

    def _stress_bin(self, stress: float) -> int:
        return int(np.clip(math.floor(stress * 5.0), 0, 4))

    def _quantize_bin(self, value: float, bins: int = 5, vmax: float = 1.0) -> int:
        value = float(np.clip(value, 0.0, vmax))
        return int(min(bins - 1, math.floor((value / max(vmax, 1e-8)) * bins)))

    def _make_signature(self, layer_idx: int, action: str, diag: Mapping[str, float], stress: float) -> str:
        density = self._quantize_bin(float(diag.get("density_a", 0.0)), bins=5, vmax=0.5)
        residual = self._quantize_bin(float(diag.get("residual_ratio", 0.0)), bins=5, vmax=2.0)
        stale = self._quantize_bin(float(diag.get("stale_ratio", 0.0)), bins=5, vmax=3.0)
        return f"L{layer_idx}:{action}:S{self._stress_bin(stress)}:D{density}:R{residual}:T{stale}"

    def _score_layer(self, diag: Mapping[str, float], global_loss: float) -> float:
        density_a = float(diag.get("density_a", 0.0))
        density_b = float(diag.get("density_b", 0.0))
        residual_ratio = float(diag.get("residual_ratio", 0.0))
        stale_ratio = float(diag.get("stale_ratio", 0.0))
        attn_entropy = float(diag.get("attn_entropy", 0.0))
        ccfg = self.cfg.controller
        density_penalty = abs(density_a - ccfg.target_density) + abs(density_b - ccfg.target_density)
        residual_penalty = max(0.0, residual_ratio - ccfg.residual_ratio_ceiling)
        stale_penalty = max(0.0, stale_ratio - ccfg.stale_ratio_ceiling)
        entropy_target = math.log(max(2, self.cfg.num_memory_slots)) * ccfg.target_attn_entropy_scale
        entropy_penalty = abs(attn_entropy - entropy_target)
        return float(
            global_loss
            + ccfg.density_penalty_weight * density_penalty
            + ccfg.residual_penalty_weight * residual_penalty
            + ccfg.stale_penalty_weight * stale_penalty
            + ccfg.entropy_penalty_weight * entropy_penalty
        )

    def _resolve_pending(self, scores: Sequence[float]) -> None:
        resolved: List[int] = []
        for layer_idx, pending in self.pending.items():
            cur = float(scores[layer_idx])
            effect = cur - float(pending.pre_score)  # negative = improvement
            context_key = (pending.context["island"], pending.context["pnn_state"], pending.context["stress_bin"])
            reward = float(np.clip(-effect, -self.cfg.controller.action_reward_clip, self.cfg.controller.action_reward_clip))
            self.bandit.update(context_key, pending.action, reward)
            self.tapestry.add_event_node(
                event_id=f"INTERVENTION_{self.global_step}_{layer_idx}_{pending.action}",
                event_type="INTERVENTION",
                generation=self.global_step,
                details={
                    "action": pending.action,
                    "effect": float(effect),
                    "island": pending.context["island"],
                    "pnn_state": pending.context["pnn_state"],
                    "stress_bin": pending.context["stress_bin"],
                    "strategy_used": pending.action,
                },
            )
            if effect > self.cfg.controller.harm_effect_threshold:
                self.immune[layer_idx].add_memory(pending.signature)
            resolved.append(layer_idx)
        for idx in resolved:
            self.pending.pop(idx, None)

    def _action_overrides(self, layer_idx: int, context: Dict[str, Any], diag: Mapping[str, float]) -> Dict[str, float]:
        overrides: Dict[str, float] = {}
        for op in self.OPERATORS:
            stats = self.tapestry.query_action_effect_with_stats(
                action=op,
                context_filters={
                    "island": context["island"],
                    "pnn_state": context["pnn_state"],
                    "stress_bin": context["stress_bin"],
                },
                generation_window=self.cfg.controller.tapestry_generation_window,
                decay_rate=0.05,
            )
            predicted_gain = -float(stats.get("effect", 0.0))
            overrides[op] = predicted_gain
        if float(diag.get("stale_ratio", 0.0)) > 1.25:
            overrides["increase_forget"] = overrides.get("increase_forget", 0.0) + 0.15
            overrides["decrease_write"] = overrides.get("decrease_write", 0.0) + 0.10
        if float(diag.get("residual_ratio", 0.0)) > 0.90:
            overrides["decrease_residual"] = overrides.get("decrease_residual", 0.0) + 0.15
        if float(diag.get("density_a", 0.0)) < 0.05:
            overrides["lower_threshold_a"] = overrides.get("lower_threshold_a", 0.0) + 0.10
            overrides["increase_gain_a"] = overrides.get("increase_gain_a", 0.0) + 0.05
        if float(diag.get("density_b", 0.0)) < 0.05:
            overrides["lower_threshold_b"] = overrides.get("lower_threshold_b", 0.0) + 0.10
            overrides["increase_gain_b"] = overrides.get("increase_gain_b", 0.0) + 0.05
        return overrides

    @torch.no_grad()
    def _apply_action(self, layer_idx: int, action: str, stress: float) -> None:
        block = self.blocks[layer_idx]
        pnn = self.pnns[layer_idx]
        circ = self.circadian[layer_idx]
        magnitude = self.cfg.controller.intervention_scale * float(circ.plasticity_multiplier) * float(getattr(pnn, "plasticity_multiplier", 1.0)) * (1.0 + stress)

        if action == "noop":
            return
        if action == "raise_threshold_a":
            block.threshold_a.add_(magnitude)
        elif action == "lower_threshold_a":
            block.threshold_a.sub_(magnitude)
        elif action == "raise_threshold_b":
            block.threshold_b.add_(magnitude)
        elif action == "lower_threshold_b":
            block.threshold_b.sub_(magnitude)
        elif action == "increase_gain_a":
            block.gain_a.add_(magnitude)
        elif action == "decrease_gain_a":
            block.gain_a.sub_(magnitude)
        elif action == "increase_gain_b":
            block.gain_b.add_(magnitude)
        elif action == "decrease_gain_b":
            block.gain_b.sub_(magnitude)
        elif action == "increase_mult_gate":
            block.mult_gate.add_(magnitude)
        elif action == "decrease_mult_gate":
            block.mult_gate.sub_(magnitude)
        elif action == "increase_residual":
            block.residual_gate.add_(magnitude)
        elif action == "decrease_residual":
            block.residual_gate.sub_(magnitude)
        elif action == "increase_write":
            block.write_alpha.add_(magnitude)
        elif action == "decrease_write":
            block.write_alpha.sub_(magnitude)
        elif action == "increase_forget":
            block.forget_lambda.add_(magnitude)
        elif action == "decrease_forget":
            block.forget_lambda.sub_(magnitude)
        else:
            raise ValueError(f"Unknown controller action: {action}")
        block.clamp_controls()
        # Charge exploit budget for non-noop interventions.
        pnn.note_exploit_outcome(evals_used=1.0, improvement=0.0)
        pnn.exploit_budget = max(0.0, float(getattr(pnn, "exploit_budget", 0.0)) - 1.0)

    @torch.no_grad()
    def emergency_stabilize(self) -> None:
        for block in self.blocks:
            block.residual_gate.fill_(max(self.cfg.controller.emergency_residual_floor, float(block.residual_gate.item()) * 0.7))
            block.forget_lambda.fill_(min(self.cfg.controller.emergency_forget_ceiling, float(block.forget_lambda.item()) + 0.05))
            block.clamp_controls()

    def _strategic_unlock(self, scores: Sequence[float]) -> int:
        candidates: List[Tuple[float, int]] = []
        for idx, pnn in enumerate(self.pnns):
            if pnn.state != PNN_STATE.LOCKED:
                continue
            stress = float(self.proxies[idx].local_stress)
            budget_exhausted = float(getattr(pnn, "exploit_budget", 0.0)) <= 0.0
            critical_stress = stress > self.cfg.controller.unlock_stress_threshold
            if budget_exhausted or critical_stress:
                priority = float(scores[idx]) * (1.0 + 0.1 * float(getattr(pnn, "generations_in_phase", 0.0)))
                candidates.append((priority, idx))
        if not candidates:
            return 0
        candidates.sort(key=lambda item: item[0], reverse=True)
        limit = max(1, int(round(len(candidates) * self.cfg.controller.strategic_unlock_fraction)))
        unlocked = 0
        for _, idx in candidates[:limit]:
            self.pnns[idx].force_unlock(self.global_step, refractory_period=4)
            self.tapestry.add_event_node(
                event_id=f"UNLOCK_{self.global_step}_{idx}",
                event_type="UNLOCK",
                generation=self.global_step,
                details={
                    "action": "unlock",
                    "effect": 0.0,
                    "island": f"layer_{idx}",
                    "pnn_state": PNN_STATE.OPEN.value,
                    "stress_bin": self._stress_bin(self.proxies[idx].local_stress),
                },
            )
            unlocked += 1
        return unlocked

    def observe_and_act(self, layer_diagnostics: Sequence[Mapping[str, float]], global_loss: float) -> Dict[str, Any]:
        self.global_step += 1
        self.last_action_report = []
        for gate in self.circadian:
            gate.update(self.global_step)

        scores: List[float] = []
        for idx, diag in enumerate(layer_diagnostics):
            self.last_diagnostics[idx] = dict(diag)
            score = self._score_layer(diag, global_loss)
            self.proxies[idx].fitness_history.append(score)
            self.proxies[idx].fitness_history = self.proxies[idx].fitness_history[-128:]
            scores.append(score)

        self.stress_field.update({0: self.proxies}, self.global_step)
        median_score = float(np.median(np.asarray(scores, dtype=np.float32))) if scores else float(global_loss)

        for idx, score in enumerate(scores):
            self.pnns[idx].update(score, self.global_step, island_median_fitness=median_score)

        self._resolve_pending(scores)
        unlocked = self._strategic_unlock(scores)

        if self.global_step % self.cfg.controller.intervention_interval == 0:
            # Intervene only on the highest-stress layers this step.
            ranked = sorted(range(len(scores)), key=lambda i: self.proxies[i].local_stress, reverse=True)
            for layer_idx in ranked[: self.cfg.controller.max_interventions_per_step]:
                pnn = self.pnns[layer_idx]
                diag = self.last_diagnostics[layer_idx]
                context = {
                    "island": f"layer_{layer_idx}",
                    "pnn_state": pnn.state.value,
                    "stress_bin": self._stress_bin(self.proxies[layer_idx].local_stress),
                }
                context_key = (context["island"], context["pnn_state"], context["stress_bin"])

                allowed_ops = list(self.OPERATORS)
                if pnn.state == PNN_STATE.CLOSING:
                    allowed_ops = [
                        "noop", "increase_forget", "decrease_forget",
                        "increase_residual", "decrease_residual",
                        "increase_write", "decrease_write",
                    ]
                elif pnn.state == PNN_STATE.LOCKED:
                    allowed_ops = ["noop", "increase_forget", "decrease_residual", "decrease_write"]

                overrides = self._action_overrides(layer_idx, context, diag)
                signature_penalties: Dict[str, float] = {}
                for op in allowed_ops:
                    signature = self._make_signature(layer_idx, op, diag, self.proxies[layer_idx].local_stress)
                    if hasattr(self.immune[layer_idx], "has_memory") and self.immune[layer_idx].has_memory(signature):
                        signature_penalties[op] = -1.0
                combined_overrides = {op: overrides.get(op, 0.0) + signature_penalties.get(op, 0.0) for op in allowed_ops}

                bandit = ContextualOperatorBandit(allowed_ops, self.cfg.controller.action_ucb_c, self.cfg.controller.action_discount)
                # warm-start from global bandit state when possible
                if context_key in self.bandit.state:
                    bandit.state[context_key] = {k: dict(v) for k, v in self.bandit.state[context_key].items() if k in allowed_ops}
                    bandit.total[context_key] = self.bandit.total.get(context_key, 0.0)
                action = bandit.select(context_key, combined_overrides)

                signature = self._make_signature(layer_idx, action, diag, self.proxies[layer_idx].local_stress)
                if hasattr(self.immune[layer_idx], "has_memory") and self.immune[layer_idx].has_memory(signature):
                    action = "noop"
                self._apply_action(layer_idx, action, self.proxies[layer_idx].local_stress)
                self.pending[layer_idx] = PendingIntervention(
                    step=self.global_step,
                    layer_idx=layer_idx,
                    action=action,
                    pre_score=scores[layer_idx],
                    signature=signature,
                    context=context,
                )
                self.last_action_report.append({
                    "layer_idx": layer_idx,
                    "action": action,
                    "stress": float(self.proxies[layer_idx].local_stress),
                    "pnn_state": pnn.state.value,
                })

        report = {
            "step": self.global_step,
            "scores": list(scores),
            "stresses": [float(p.local_stress) for p in self.proxies],
            "pnn_states": [pnn.state.value for pnn in self.pnns],
            "unlocked": unlocked,
            "actions": list(self.last_action_report),
        }
        self.last_scores = list(scores)
        return report


# -----------------------------------------------------------------------------
# State and output dataclasses.
# -----------------------------------------------------------------------------
@dataclass
class WorldModelState:
    layer_hidden: List[Tensor]
    layer_memory: List[Tensor]
    episodic_memory: Tensor
    timestep: int = 0

    def detach(self) -> "WorldModelState":
        return WorldModelState(
            layer_hidden=[x.detach() for x in self.layer_hidden],
            layer_memory=[x.detach() for x in self.layer_memory],
            episodic_memory=self.episodic_memory.detach(),
            timestep=self.timestep,
        )


@dataclass
class ForwardOutput:
    sequence: Tensor
    state: WorldModelState
    fusion_weights: Dict[str, Tensor]
    layer_diagnostics: List[Dict[str, float]]
    episodic_read_entropy: float
    controller_report: Optional[Dict[str, Any]] = None
    text_logits: Optional[Tensor] = None
    numeric_pred: Optional[Tensor] = None
    audio_pred: Optional[Tensor] = None
    vision_pred: Optional[Tensor] = None


# -----------------------------------------------------------------------------
# Main model.
# -----------------------------------------------------------------------------
class HomeostaticMultimodalWorldModel(nn.Module):
    """
    Multimodal, stateful world-model architecture with a homeostatic control plane.

    Expected batch shapes:
    - text_tokens: [B, T] or [B, T, L_text]
    - vision:      [B, T, C, H, W]
    - audio:       [B, T, F_audio]
    - numeric:     [B, T, F_numeric]
    - event_mask:  [B, T] optional event-boundary supervision in [0, 1]
    """
    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        self.cfg = cfg
        self.text_encoder = TextEncoder(cfg)
        self.vision_encoder = VisionEncoder(cfg)
        self.audio_encoder = AudioEncoder(cfg)
        self.numeric_encoder = NumericEncoder(cfg)
        self.fusion = GatedModalityFusion(cfg)
        self.blocks = nn.ModuleList([AdaptiveSparseWorldBlock(cfg, i) for i in range(cfg.num_layers)])
        self.episodic_memory = EpisodicMemoryBank(cfg)
        self.final_norm = RMSNorm(cfg.d_model)
        self.event_proj = nn.Linear(2 * cfg.d_model, 1)
        self.controller = HomeostaticController(self.blocks, cfg)

        self.text_head = nn.Linear(cfg.d_model, cfg.modality.text_vocab_size) if cfg.predict_text else None
        self.numeric_head = nn.Linear(cfg.d_model, cfg.modality.numeric_dim) if cfg.predict_numeric else None
        self.audio_head = nn.Linear(cfg.d_model, cfg.modality.audio_dim) if cfg.predict_audio else None
        if cfg.predict_vision:
            self.vision_head = nn.Linear(cfg.d_model, cfg.modality.vision_channels * cfg.modality.vision_height * cfg.modality.vision_width)
        else:
            self.vision_head = None

        self._last_forward_report: Optional[Dict[str, Any]] = None
        self._last_layer_diagnostics: List[Dict[str, float]] = []

    def init_state(self, batch_size: int, device: torch.device | None = None) -> WorldModelState:
        device = device or next(self.parameters()).device
        hidden = [torch.zeros(batch_size, self.cfg.d_model, device=device) for _ in range(self.cfg.num_layers)]
        memory = [torch.zeros(batch_size, self.cfg.num_memory_slots, self.cfg.d_model, device=device) for _ in range(self.cfg.num_layers)]
        episodic = torch.zeros(batch_size, self.cfg.num_episodic_slots, self.cfg.d_model, device=device)
        return WorldModelState(layer_hidden=hidden, layer_memory=memory, episodic_memory=episodic, timestep=0)

    def _validate_inputs(
        self,
        text_tokens: Optional[Tensor],
        vision: Optional[Tensor],
        audio: Optional[Tensor],
        numeric: Optional[Tensor],
        event_mask: Optional[Tensor],
    ) -> None:
        refs: List[Tuple[str, Tuple[int, int]]] = []
        if text_tokens is not None:
            if text_tokens.dim() not in (2, 3):
                raise ValueError(f"text_tokens must have shape [B,T] or [B,T,L], got {tuple(text_tokens.shape)}")
            refs.append(("text", (int(text_tokens.size(0)), int(text_tokens.size(1)))))
        if vision is not None:
            if vision.dim() != 5:
                raise ValueError(f"vision must have shape [B,T,C,H,W], got {tuple(vision.shape)}")
            if vision.size(2) != self.cfg.modality.vision_channels or vision.size(3) != self.cfg.modality.vision_height or vision.size(4) != self.cfg.modality.vision_width:
                raise ValueError(
                    "vision shape mismatch: expected "
                    f"[B,T,{self.cfg.modality.vision_channels},{self.cfg.modality.vision_height},{self.cfg.modality.vision_width}], "
                    f"got {tuple(vision.shape)}"
                )
            refs.append(("vision", (int(vision.size(0)), int(vision.size(1)))))
        if audio is not None:
            if audio.dim() != 3:
                raise ValueError(f"audio must have shape [B,T,F], got {tuple(audio.shape)}")
            if audio.size(-1) != self.cfg.modality.audio_dim:
                raise ValueError(f"audio last dim must be {self.cfg.modality.audio_dim}, got {audio.size(-1)}")
            refs.append(("audio", (int(audio.size(0)), int(audio.size(1)))))
        if numeric is not None:
            if numeric.dim() != 3:
                raise ValueError(f"numeric must have shape [B,T,F], got {tuple(numeric.shape)}")
            if numeric.size(-1) != self.cfg.modality.numeric_dim:
                raise ValueError(f"numeric last dim must be {self.cfg.modality.numeric_dim}, got {numeric.size(-1)}")
            refs.append(("numeric", (int(numeric.size(0)), int(numeric.size(1)))))
        if not refs:
            raise ValueError("At least one modality must be provided.")
        ref_bt = refs[0][1]
        for name, bt in refs[1:]:
            if bt != ref_bt:
                raise ValueError(f"All provided modalities must share the same [B,T]. Expected {ref_bt}, got {name}={bt}")
        if event_mask is not None:
            if event_mask.dim() != 2:
                raise ValueError(f"event_mask must have shape [B,T], got {tuple(event_mask.shape)}")
            if (int(event_mask.size(0)), int(event_mask.size(1))) != ref_bt:
                raise ValueError(f"event_mask must match [B,T]={ref_bt}, got {tuple(event_mask.shape)}")

    def _encode_modalities(
        self,
        text_tokens: Optional[Tensor],
        vision: Optional[Tensor],
        audio: Optional[Tensor],
        numeric: Optional[Tensor],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        features: Dict[str, Optional[Tensor]] = {
            "text": self.text_encoder(text_tokens) if text_tokens is not None else None,
            "vision": self.vision_encoder(vision) if vision is not None else None,
            "audio": self.audio_encoder(audio) if audio is not None else None,
            "numeric": self.numeric_encoder(numeric) if numeric is not None else None,
        }
        fused, weights = self.fusion(features)
        return fused, weights

    def _run_episodic_chunk(
        self,
        x_chunk: Tensor,
        episodic_memory: Tensor,
        prev_output: Tensor,
        event_mask_chunk: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, List[float]]:
        outputs: List[Tensor] = []
        entropies: List[float] = []
        steps = x_chunk.size(1)
        for step in range(steps):
            x_t = x_chunk[:, step, :]
            episodic_context, episodic_entropy = self.episodic_memory.read(x_t, episodic_memory)
            entropies.append(_as_float(episodic_entropy))
            x_t = self.final_norm(x_t + episodic_context)
            if event_mask_chunk is not None:
                strength = event_mask_chunk[:, step].float().clamp(0.0, 1.0)
            else:
                surprise = torch.sigmoid(self.event_proj(torch.cat([x_t, x_t - prev_output], dim=-1))).squeeze(-1)
                strength = surprise.clamp_min(self.cfg.event_write_min_strength) * (surprise > self.cfg.surprise_event_threshold).float()
            episodic_memory = self.episodic_memory.write(x_t, episodic_memory, strength)
            outputs.append(x_t)
            prev_output = x_t
        return torch.stack(outputs, dim=1), episodic_memory, prev_output, entropies

    def forward(
        self,
        *,
        text_tokens: Optional[Tensor] = None,
        vision: Optional[Tensor] = None,
        audio: Optional[Tensor] = None,
        numeric: Optional[Tensor] = None,
        event_mask: Optional[Tensor] = None,
        state: Optional[WorldModelState] = None,
        controller_loss: Optional[Tensor | float] = None,
    ) -> ForwardOutput:
        self._validate_inputs(text_tokens, vision, audio, numeric, event_mask)
        fused, fusion_weights = self._encode_modalities(text_tokens, vision, audio, numeric)
        b, t, _ = fused.shape
        device = fused.device
        state = state or self.init_state(b, device=device)

        outputs: List[Tensor] = []
        per_layer_diagnostics: List[List[Dict[str, float]]] = [[] for _ in range(self.cfg.num_layers)]
        episodic_entropies: List[float] = []
        prev_output = torch.zeros(b, self.cfg.d_model, device=device)
        chunk_size = max(1, int(self.cfg.scan_chunk_size))

        for start in range(0, t, chunk_size):
            end = min(t, start + chunk_size)
            x_chunk = fused[:, start:end, :]
            for layer_idx, block in enumerate(self.blocks):
                x_chunk, state.layer_hidden[layer_idx], state.layer_memory[layer_idx], diag_chunk = block.forward_chunk(
                    x_chunk,
                    state.layer_hidden[layer_idx],
                    state.layer_memory[layer_idx],
                )
                per_layer_diagnostics[layer_idx].extend(diag_chunk)

            event_chunk = event_mask[:, start:end] if event_mask is not None else None
            x_chunk, state.episodic_memory, prev_output, ent_chunk = self._run_episodic_chunk(
                x_chunk,
                state.episodic_memory,
                prev_output,
                event_chunk,
            )
            outputs.append(x_chunk)
            episodic_entropies.extend(ent_chunk)

        sequence = torch.cat(outputs, dim=1)
        state.timestep += int(t)
        if self.cfg.detach_state_between_steps:
            state = state.detach()

        layer_diagnostics = []
        for layer_stats in per_layer_diagnostics:
            if not layer_stats:
                layer_diagnostics.append({})
                continue
            merged: Dict[str, float] = {}
            for key in layer_stats[0].keys():
                merged[key] = _safe_mean([float(item.get(key, 0.0)) for item in layer_stats], default=0.0)
            layer_diagnostics.append(merged)

        controller_report = None
        if self.cfg.enable_online_homeostasis and controller_loss is not None:
            controller_report = self.controller.observe_and_act(layer_diagnostics, _as_float(controller_loss))
        self._last_forward_report = controller_report
        self._last_layer_diagnostics = layer_diagnostics

        text_logits = self.text_head(sequence) if self.text_head is not None else None
        numeric_pred = self.numeric_head(sequence) if self.numeric_head is not None else None
        audio_pred = self.audio_head(sequence) if self.audio_head is not None else None
        vision_pred = None
        if self.vision_head is not None:
            flat = self.vision_head(sequence)
            vision_pred = flat.view(
                b,
                t,
                self.cfg.modality.vision_channels,
                self.cfg.modality.vision_height,
                self.cfg.modality.vision_width,
            )

        return ForwardOutput(
            sequence=sequence,
            state=state,
            fusion_weights=fusion_weights,
            layer_diagnostics=layer_diagnostics,
            episodic_read_entropy=_safe_mean(episodic_entropies),
            controller_report=controller_report,
            text_logits=text_logits,
            numeric_pred=numeric_pred,
            audio_pred=audio_pred,
            vision_pred=vision_pred,
        )

    def controller_step(self, loss_value: Tensor | float) -> Dict[str, Any]:
        if not self._last_layer_diagnostics:
            raise RuntimeError("controller_step called before a forward pass produced diagnostics.")
        report = self.controller.observe_and_act(self._last_layer_diagnostics, _as_float(loss_value))
        # Training loops typically omit controller_loss on forward(); they call this after backward.
        # Mirror forward()'s assignment so summarize_controller_report / logs see real stresses & unlocks.
        self._last_forward_report = report
        return report

    def emergency_stabilize(self) -> None:
        self.controller.emergency_stabilize()


# -----------------------------------------------------------------------------
# Generic multitask loss helper for the synthetic benchmark.
# -----------------------------------------------------------------------------
@dataclass
class LossWeights:
    text: float = 1.0
    numeric: float = 1.0
    audio: float = 1.0
    vision: float = 1.0


@dataclass
class LossOutput:
    total: Tensor
    parts: Dict[str, Tensor]


class MultimodalPredictionLoss(nn.Module):
    def __init__(self, cfg: WorldModelConfig, weights: LossWeights = LossWeights()):
        super().__init__()
        self.cfg = cfg
        self.weights = weights

    def forward(
        self,
        output: ForwardOutput,
        *,
        text_targets: Optional[Tensor] = None,
        numeric_targets: Optional[Tensor] = None,
        audio_targets: Optional[Tensor] = None,
        vision_targets: Optional[Tensor] = None,
    ) -> LossOutput:
        parts: Dict[str, Tensor] = {}
        total = torch.zeros((), device=output.sequence.device)

        if output.text_logits is not None and text_targets is not None:
            logits = output.text_logits.reshape(-1, output.text_logits.size(-1))
            target = text_targets.reshape(-1)
            parts["text"] = F.cross_entropy(logits, target, ignore_index=self.cfg.modality.text_pad_id)
            total = total + self.weights.text * parts["text"]

        if output.numeric_pred is not None and numeric_targets is not None:
            parts["numeric"] = F.mse_loss(output.numeric_pred, numeric_targets)
            total = total + self.weights.numeric * parts["numeric"]

        if output.audio_pred is not None and audio_targets is not None:
            parts["audio"] = F.mse_loss(output.audio_pred, audio_targets)
            total = total + self.weights.audio * parts["audio"]

        if output.vision_pred is not None and vision_targets is not None:
            parts["vision"] = F.mse_loss(output.vision_pred, vision_targets)
            total = total + self.weights.vision * parts["vision"]

        return LossOutput(total=total, parts=parts)


# -----------------------------------------------------------------------------
# Example training step.
# -----------------------------------------------------------------------------
@torch.no_grad()
def summarize_controller_report(report: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if report is None:
        return {}
    return {
        "step": report.get("step"),
        "pnn_states": report.get("pnn_states"),
        "stresses": report.get("stresses"),
        "actions": report.get("actions"),
        "unlocked": report.get("unlocked"),
    }


def example_train_step(
    model: HomeostaticMultimodalWorldModel,
    criterion: MultimodalPredictionLoss,
    optimizer: torch.optim.Optimizer,
    *,
    text_tokens: Optional[Tensor] = None,
    text_targets: Optional[Tensor] = None,
    vision: Optional[Tensor] = None,
    vision_targets: Optional[Tensor] = None,
    audio: Optional[Tensor] = None,
    audio_targets: Optional[Tensor] = None,
    numeric: Optional[Tensor] = None,
    numeric_targets: Optional[Tensor] = None,
    event_mask: Optional[Tensor] = None,
    state: Optional[WorldModelState] = None,
    grad_clip_norm: float = 1.0,
) -> Tuple[ForwardOutput, LossOutput]:
    optimizer.zero_grad(set_to_none=True)
    train_state = state.detach() if state is not None else None
    output = model(
        text_tokens=text_tokens,
        vision=vision,
        audio=audio,
        numeric=numeric,
        event_mask=event_mask,
        state=train_state,
    )
    losses = criterion(
        output,
        text_targets=text_targets,
        numeric_targets=numeric_targets,
        audio_targets=audio_targets,
        vision_targets=vision_targets,
    )
    losses.total.backward()
    if grad_clip_norm > 0:
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()
    for block in model.blocks:
        block.clamp_controls()

    if model.cfg.enable_online_homeostasis:
        probe_state = state.detach() if state is not None else None
        with torch.no_grad():
            probe_output = model(
                text_tokens=text_tokens,
                vision=vision,
                audio=audio,
                numeric=numeric,
                event_mask=event_mask,
                state=probe_state,
            )
            probe_losses = criterion(
                probe_output,
                text_targets=text_targets,
                numeric_targets=numeric_targets,
                audio_targets=audio_targets,
                vision_targets=vision_targets,
            )
        model.controller_step(probe_losses.total.detach())
    return output, losses


__all__ = [
    "ControllerConfig",
    "ForwardOutput",
    "HomeostaticMultimodalWorldModel",
    "LossOutput",
    "LossWeights",
    "ModalityConfig",
    "MultimodalPredictionLoss",
    "WorldModelConfig",
    "WorldModelState",
    "example_train_step",
    "summarize_controller_report",
]
