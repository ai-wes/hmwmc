"""
Homeostatic Predictive Memory (HPM)
====================================

A surprise-gated working-memory module that replaces attention as the retention
primitive for "find the most recent event of type X" queries.

Architectural thesis
--------------------
Plasticity should be governed by second-order statistics of prediction, not
first-order error. HPM implements this directly: memory writes are gated by
z-scored prediction error (surprise relative to expected surprise) rather than
raw magnitude. The memory bootstraps off the prediction mechanism -- a
freshly-trained model has high sigma and writes conservatively; a mature model
has low sigma and is naturally sensitive to rare events; signal-to-noise is
handled by construction.

Slot state (OPEN / CLOSING / LOCKED) follows the same state-machine pattern
used by the PNN regulator in the MOEA codebase, but operates on per-timestep
memory plasticity rather than per-generation synaptic plasticity. A slot that
is frequently rewritten stays OPEN. A slot that has committed to stable
content transitions CLOSING -> LOCKED, lowering its gate gain. An extreme
z-score force-unlocks a LOCKED slot, directly parallel to PNN force_unlock
when local_stress exceeds the critical threshold. Same homeostatic principle,
different timescale.

Update rule
-----------
  pred_t    = predictor_slot(h_{t-1})                    # one-step-ahead prediction
  e_t       = mean((pred_t - h_t.detach())^2)            # per (b, slot) scalar
  z_t       = (e_t - mu_slot) / (sigma_slot + eps)       # normalized surprise
  g_t_raw   = sigmoid( gate_mlp_slot([h_t ; z_t ; state_embed]) )
  g_t       = g_t_raw * state_gain_slot                  # OPEN 1.0, CLOSING 0.5, LOCKED 0.1
  write_t   = write_encoder_slot(h_t)
  w_t       = (1 - g_t) * w_{t-1} + g_t * write_t

  mu, sigma updated by per-slot EMA, detached, training-only.
  Force-unlock: if |z_t| > critical_z, slot state -> OPEN and g_t boosted to ~1.0.

Gradient flow
-------------
gate_mlp and write_encoder carry gradients from downstream loss through the
leaky recurrence w_t = (1-g) w_{t-1} + g write(h_t). The recurrence is the
mechanism by which gradient travels from query time back to the original
surprise event -- this is exactly the "last-event retrieval" property we want.
Running stats, z-scores, and prediction errors are all detached.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
@dataclass
class HPMConfig:
    """Configuration for Homeostatic Predictive Memory."""

    enabled: bool = True
    # Number of memory slots. n_slots=1 is Level 1. n_slots>1 is Level 2.
    n_slots: int = 4
    # Output dimension per slot. None means use d_model.
    slot_dim: Optional[int] = None
    # Aggregation across slots for the HPM read.
    #   "concat": concatenate all slots (hpm_dim = n_slots * slot_dim).
    #   "mean":   mean-pool across slots (hpm_dim = slot_dim).
    #   "attn":   query-conditioned attention over slots (hpm_dim = slot_dim).
    read_mode: str = "concat"
    # Competitive write gating across slots (softmax-based winner-take-all-ish).
    competitive: bool = False
    # EMA decay for per-slot (mu, sigma) of prediction error.
    ema_decay: float = 0.99
    # Force-unlock threshold on |z|. PNN-style critical-stress override.
    critical_z: float = 3.0
    # State machine transition thresholds on gate EMA.
    open_to_closing_rate: float = 0.10
    closing_to_locked_rate: float = 0.03
    unlock_rate: float = 0.25
    # State-dependent gate gains.
    gain_open: float = 1.0
    gain_closing: float = 0.5
    gain_locked: float = 0.1
    # Warmup steps during which force-unlock is disabled.
    warmup_steps: int = 50
    # Floor for sigma.
    sigma_floor: float = 1e-3
    # Gate MLP hidden size.
    gate_hidden: int = 64
    # Dim of state-id embedding fed into gate.
    state_embed_dim: int = 8
    # --- C3: retroactive binding ---
    # Window size for retroactive binding. 0 = disabled (Level 1/2 behaviour).
    # When > 0, writes are a learned mixture over the last `retroactive_window`
    # hidden states instead of the current h_t alone (Level 3).
    retroactive_window: int = 0
    # --- C4: multi-timescale ---
    # Per-slot persistence multipliers. When set, len must equal n_slots.
    # Each value scales the *retention* (1 - g) for that slot:
    #   effective_gate_s = g_s / timescale_s  (higher timescale => slower writes)
    # None means all slots use timescale 1.0 (uniform, default).
    slot_timescales: Optional[Tuple[float, ...]] = None
    # --- Learned surprise subspace (Phase B, critique #1) ---
    # Dimension of the per-slot projection for computing prediction error.
    # 0 = disabled (surprise computed in raw h-space, original behavior).
    surprise_dim: int = 0
    # --- Adaptive sigma floor (Phase B, critique #6) ---
    # When True, sigma floor scales with batch_mu to prevent convergence-collapse.
    sigma_floor_adaptive: bool = False
    sigma_floor_scale: float = 0.1
    # Absolute MSE gate: force-unlock requires e_t > this AND |z| > critical_z.
    min_surprise_threshold: float = 0.0
    # --- Content-conditional write gating (Phase C, critique #2) ---
    # When True, each slot develops content preferences via key-query matching.
    # Subsumes the old `competitive` flag with a content-aware mechanism.
    content_gating: bool = False
    # --- Learnable state gains (Phase D, critique #4) ---
    # When True, the OPEN/CLOSING/LOCKED gain multipliers are learned via softplus.
    learnable_gains: bool = False

    # =====================================================================
    # HPM v2 — architectural overhaul (items 2, 3, 6, 7)
    # =====================================================================

    # --- Top-k routing with stickiness and load-balancing (item #2) ---
    # Replaces sigmoid gate + content_gating + competitive + force-unlock
    # with soft hazard-based routing.
    topk_routing: bool = False
    topk_k: int = 2
    # Surprise bonus weight in routing logit.
    routing_surprise_bonus: float = 1.0    # β
    # Stickiness: bonus for slot that was top-1 last step.
    routing_stickiness: float = 0.5        # η
    # Load-balancing penalty: running write share.
    routing_load_balance: float = 0.1      # γ
    # Age penalty: soft replacement for hard force-unlock.
    routing_age_penalty: float = 0.01      # δ

    # --- Factorized surprise channels (item #3) ---
    # When True, surprise is computed via n_surprise_channels independent heads
    # instead of a single projection. Each produces its own z-score.
    factorized_surprise: bool = False
    n_surprise_channels: int = 3

    # --- Slot diversity regularizer (item #6) ---
    # λ · mean_pairwise_cosine(slot_keys). 0 = disabled.
    # Requires topk_routing=True or content_gating=True (needs slot keys).
    slot_diversity_lambda: float = 0.0

    # --- Per-slot-state running stats (item #7) ---
    # Split μ/σ EMAs by slot state (OPEN/CLOSING/LOCKED) instead of global.
    per_state_stats: bool = False

    # =====================================================================
    # v2: cleanup diff — kill state machine, fix σ floor, fix force-unlock
    # =====================================================================

    # Replace OPEN/CLOSING/LOCKED state machine and force-unlock with a
    # single continuous plasticity scalar per slot:
    #   plasticity = sigmoid(scale * age_factor * content_mismatch * |z| - bias)
    # This is a deletion, not an addition — reduces complexity.
    continuous_plasticity: bool = False
    plasticity_age_tau: float = 10.0   # age saturation timescale
    # Hard sigma floor: clamp sigma ≥ this value. Prevents convergence-collapse
    # (sigma 0.94→0.31) that the adaptive floor failed to catch.
    # 0.5 = 50% of init value (1.0). Set 0 to use only sigma_floor.
    sigma_hard_floor: float = 0.5
    # Load-balance penalty in the non-topk gate path (content_gating=True).
    # Zero = disabled. Penalises overused slots: −γ·running_write_share_s.
    gate_load_balance: float = 0.0


# =========================================================================
# Entity Table configuration
# =========================================================================
@dataclass
class EntityTableConfig:
    """Persistent entity-state table with GRU-style per-entity update."""
    enabled: bool = False
    n_entities: int = 4
    d_entity: int = 64
    # Read aggregation: "concat" or "attn".
    read_mode: str = "concat"


# =========================================================================
# Event Tape configuration
# =========================================================================
@dataclass
class EventTapeConfig:
    """Append-only event tape that snapshots high-surprise boundaries."""
    enabled: bool = False
    max_events: int = 32
    # Minimum total surprise (max |z| across slots) to trigger a tape write.
    surprise_threshold: float = 2.0
    # Whether to include entity table state snapshots in tape entries.
    include_entity_state: bool = True
    # When True, if fewer than min_events pass the threshold, fall back to
    # selecting the top-K most surprising timesteps. Ensures the tape is
    # never empty when there *is* temporal data to record.
    top_k_fallback: bool = True
    min_events: int = 4


# =========================================================================
@dataclass
class EntityHistoryConfig:
    """Time-indexed entity-state snapshots for scene reconstruction."""
    enabled: bool = False
    n_snapshots: int = 16          # K uniformly-spaced entity-state snapshots
    include_time_embed: bool = True
    max_episode_length: int = 512


STATE_OPEN = 0
STATE_CLOSING = 1
STATE_LOCKED = 2
_STATE_NAMES = ("OPEN", "CLOSING", "LOCKED")


# -----------------------------------------------------------------------------
# Batched per-slot linear layer.
# -----------------------------------------------------------------------------
class SlotLinear(nn.Module):
    """
    Independent Linear per slot, vectorized.

    Input  : (B, n_slots, in_dim) for per-slot input, or (B, in_dim) broadcast.
    Output : (B, n_slots, out_dim).
    """

    def __init__(self, n_slots: int, in_dim: int, out_dim: int):
        super().__init__()
        self.n_slots = n_slots
        self.in_dim = in_dim
        self.out_dim = out_dim
        bound = (6.0 / (in_dim + out_dim)) ** 0.5
        self.weight = nn.Parameter(torch.empty(n_slots, in_dim, out_dim).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.zeros(n_slots, out_dim))

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 2:
            return torch.einsum("bi,sio->bso", x, self.weight) + self.bias.unsqueeze(0)
        return torch.einsum("bsi,sio->bso", x, self.weight) + self.bias.unsqueeze(0)


# -----------------------------------------------------------------------------
# Entity Table — persistent entity-state memory (item #1)
# -----------------------------------------------------------------------------
class EntityTable(nn.Module):
    """
    Persistent entity-state table with GRU-style update.

    Maintains (B, n_entities, d_entity) state updated every timestep via a GRU
    cell. Entity slots are routed via learned keys with soft attention over h_t.
    
    This separates slowly-varying per-entity facts (who holds what, where each
    entity is) from HPM's role of storing surprising events/deltas.
    """

    def __init__(self, d_model: int, cfg: EntityTableConfig):
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        self.n_entities = cfg.n_entities
        self.d_entity = cfg.d_entity

        # Learned entity keys for routing h_t → entity slots.
        self.entity_keys = nn.Parameter(torch.randn(cfg.n_entities, d_model) * 0.02)

        # Input projection: d_model → d_entity.
        self.input_proj = nn.Linear(d_model, cfg.d_entity)

        # GRU cell for per-entity state update.
        self.gru = nn.GRUCell(cfg.d_entity, cfg.d_entity)

        # Learnable initial entity state.
        self.e0 = nn.Parameter(torch.zeros(cfg.n_entities, cfg.d_entity))

        # Attention-based read (if read_mode == "attn").
        if cfg.read_mode == "attn":
            self.read_query_proj = nn.Linear(d_model, cfg.d_entity)

    @property
    def output_dim(self) -> int:
        if self.cfg.read_mode == "concat":
            return self.n_entities * self.d_entity
        return self.d_entity

    def forward(self, h_seq: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        """
        h_seq: (B, T, D). Returns (entity_seq, diagnostics).
        entity_seq: (B, T, output_dim).
        """
        B, T, D = h_seq.shape
        device = h_seq.device
        dtype = h_seq.dtype
        n_e = self.n_entities

        if T == 0:
            return (
                torch.zeros(B, 0, self.output_dim, device=device, dtype=dtype),
                {},
            )

        # Precompute input projections: (B, T, d_entity).
        h_proj = self.input_proj(h_seq)

        # Routing: entity_keys dot h_t → soft attention per timestep.
        # (B, T, D) @ (n_e, D).T → (B, T, n_e)
        inv_sqrt_d = 1.0 / math.sqrt(D)
        route_logits = torch.einsum("btd,ed->bte", h_seq, self.entity_keys) * inv_sqrt_d
        route_weights = torch.softmax(route_logits, dim=-1)  # (B, T, n_e)

        # Recurrent GRU update.
        entity_state = self.e0.unsqueeze(0).expand(B, -1, -1).contiguous()  # (B, n_e, d_e)
        per_step: List[Tensor] = []
        route_entropy_sum = 0.0

        for t in range(T):
            # Weighted input per entity: (B, n_e, d_e).
            w_t = route_weights[:, t]  # (B, n_e)
            input_t = w_t.unsqueeze(-1) * h_proj[:, t].unsqueeze(1)  # (B, n_e, d_e)

            # Batched GRU: flatten (B, n_e) → (B*n_e).
            entity_flat = entity_state.reshape(B * n_e, self.d_entity)
            input_flat = input_t.reshape(B * n_e, self.d_entity)
            entity_flat = self.gru(input_flat, entity_flat)
            entity_state = entity_flat.reshape(B, n_e, self.d_entity)

            per_step.append(entity_state)

            # Routing entropy for diagnostics.
            with torch.no_grad():
                ent = -(w_t * (w_t + 1e-8).log()).sum(dim=-1).mean().item()
                route_entropy_sum += ent

        entity_stack = torch.stack(per_step, dim=1)  # (B, T, n_e, d_e)

        # Read aggregation.
        if self.cfg.read_mode == "concat":
            entity_seq = entity_stack.reshape(B, T, n_e * self.d_entity)
        elif self.cfg.read_mode == "attn":
            q = self.read_query_proj(h_seq)  # (B, T, d_e)
            attn_logits = torch.einsum("btd,btnd->btn", q, entity_stack)
            attn = torch.softmax(attn_logits, dim=-1).unsqueeze(-1)  # (B, T, n_e, 1)
            entity_seq = (attn * entity_stack).sum(dim=2)
        else:
            raise ValueError(f"Unknown EntityTable read_mode: {self.cfg.read_mode}")

        diag: Dict[str, float] = {
            "entity_route_entropy": route_entropy_sum / max(T, 1),
            "entity_state_norm": float(entity_state.detach().norm(dim=-1).mean().item()),
        }
        return entity_seq, diag, entity_stack


# -----------------------------------------------------------------------------
# Event Tape — append-only boundary snapshot buffer (item #6)
# -----------------------------------------------------------------------------
class EventTape(nn.Module):
    """
    Append-only event tape that snapshots high-surprise boundaries.

    At each timestep where total surprise (max |z| across slots) exceeds
    a threshold, stores (t, h_t, entity_state_t). At query time, the
    IterativeQueryHead attends over tape entries directly.

    This exploits the strong boundary signal (17.9× event/non-event ratio)
    that HPM v1 surprise detection produces, making it available for
    structured retrieval at query time.
    """

    def __init__(self, d_model: int, cfg: EventTapeConfig, d_entity_total: int = 0):
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        self.d_entity_total = d_entity_total

        entry_dim = d_model + d_entity_total
        self.entry_proj = nn.Linear(entry_dim, d_model)
        self.time_embed = nn.Embedding(512, d_model)

    @property
    def output_dim(self) -> int:
        return self.d_model

    def forward(
        self,
        h_seq: Tensor,
        z_per_step: Tensor,
        entity_states: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Dict[str, float]]:
        """
        Args:
            h_seq:         (B, T, D) — hidden states from block pipeline.
            z_per_step:    (B, T, n_slots) — z-scores from HPM.
            entity_states: (B, T, n_e, d_e) or None — per-step entity table state.

        Returns:
            tape_entries: (B, max_events, D) — projected tape content.
            tape_mask:    (B, max_events) bool — True for valid entries.
            tape_times:   (B, max_events) long — timestep of each entry.
            diagnostics:  dict.
        """
        B, T, D = h_seq.shape
        cfg = self.cfg
        device = h_seq.device
        dtype = h_seq.dtype

        if T == 0:
            empty_e = torch.zeros(B, 0, D, device=device, dtype=dtype)
            empty_m = torch.zeros(B, 0, dtype=torch.bool, device=device)
            empty_t = torch.zeros(B, 0, dtype=torch.long, device=device)
            return empty_e, empty_m, empty_t, {}

        # Total surprise per timestep: max |z| across slots.
        total_surprise = z_per_step.abs().max(dim=-1).values  # (B, T)

        tape_h = torch.zeros(B, cfg.max_events, D, device=device, dtype=dtype)
        tape_times = torch.zeros(B, cfg.max_events, dtype=torch.long, device=device)
        tape_mask = torch.zeros(B, cfg.max_events, dtype=torch.bool, device=device)
        d_ent = self.d_entity_total
        tape_entity = (
            torch.zeros(B, cfg.max_events, d_ent, device=device, dtype=dtype)
            if d_ent > 0 and cfg.include_entity_state and entity_states is not None
            else None
        )

        n_events_total = 0
        for b in range(B):
            event_times_b = (total_surprise[b] > cfg.surprise_threshold).nonzero(as_tuple=False).squeeze(-1)
            n_ev = event_times_b.numel()
            # Fallback: if threshold is too strict, take the top-K most
            # surprising timesteps so the tape is never starved.
            if cfg.top_k_fallback and n_ev < cfg.min_events and T > 0:
                k = min(cfg.max_events, T)
                _, top_idx = total_surprise[b].topk(k)
                event_times_b = top_idx.sort().values
                n_ev = event_times_b.numel()
            if n_ev == 0:
                continue
            if n_ev > cfg.max_events:
                _, top_idx = total_surprise[b, event_times_b].topk(cfg.max_events)
                event_times_b = event_times_b[top_idx].sort().values
                n_ev = cfg.max_events
            tape_h[b, :n_ev] = h_seq[b, event_times_b]
            tape_times[b, :n_ev] = event_times_b
            tape_mask[b, :n_ev] = True
            if tape_entity is not None and entity_states is not None:
                tape_entity[b, :n_ev] = entity_states[b, event_times_b].reshape(n_ev, -1)
            n_events_total += n_ev

        # Project entries: concat h + entity (if available), project to d_model.
        if tape_entity is not None:
            raw_entries = torch.cat([tape_h, tape_entity], dim=-1)
        elif d_ent > 0:
            pad = torch.zeros(B, cfg.max_events, d_ent, device=device, dtype=dtype)
            raw_entries = torch.cat([tape_h, pad], dim=-1)
        else:
            raw_entries = tape_h

        tape_entries = self.entry_proj(raw_entries) + self.time_embed(
            tape_times.clamp(max=511)
        )

        diag: Dict[str, float] = {
            "event_tape_n_events": float(n_events_total) / max(B, 1),
            "event_tape_fill_rate": float(tape_mask.float().mean().item()),
        }
        return tape_entries, tape_mask, tape_times, diag


# -----------------------------------------------------------------------------
# Entity History Bank — uniformly-spaced entity-state snapshots
# -----------------------------------------------------------------------------
class EntityHistoryBank(nn.Module):
    """
    Time-indexed entity-state snapshots for scene reconstruction.

    EntityTable alone only exposes the current state. This bank samples K
    per-timestep entity states across the episode, projects each to d_model,
    and adds a time embedding so the retrieval head can attend over
    (t_k, entity_state_at_t_k) pairs and answer "at alarm" / "at handoff"
    queries even when no surprise boundary was recorded.
    """

    def __init__(self, d_entity_total: int, cfg: EntityHistoryConfig, d_model: int):
        super().__init__()
        self.cfg = cfg
        self.d_entity_total = d_entity_total
        self.d_model = d_model
        self.entry_proj = nn.Linear(d_entity_total, d_model)
        if cfg.include_time_embed:
            self.time_embed = nn.Embedding(cfg.max_episode_length, d_model)

    @property
    def output_dim(self) -> int:
        return self.d_model

    def forward(self, entity_stack: Tensor) -> Tuple[Tensor, Tensor, Tensor, Dict[str, float]]:
        """
        Args:
            entity_stack: (B, T, n_e, d_e) — per-timestep entity states.

        Returns:
            entries: (B, K, d_model) — projected snapshot entries.
            mask:    (B, K) bool — True for valid entries.
            times:   (B, K) long — timestep index of each snapshot.
            diag:    dict.
        """
        B, T, n_e, d_e = entity_stack.shape
        K = min(self.cfg.n_snapshots, T)
        # Uniformly spaced timesteps including t=0 and t=T-1.
        idx = torch.linspace(0, T - 1, steps=K, device=entity_stack.device).long()
        snaps = entity_stack[:, idx].reshape(B, K, n_e * d_e)  # (B, K, d_entity_total)
        entries = self.entry_proj(snaps)
        if self.cfg.include_time_embed:
            entries = entries + self.time_embed(idx.clamp(max=self.cfg.max_episode_length - 1))
        mask = torch.ones(B, K, dtype=torch.bool, device=entity_stack.device)
        times = idx.unsqueeze(0).expand(B, -1).contiguous()
        diag: Dict[str, float] = {"entity_history_snapshots": float(K)}
        return entries, mask, times, diag


# -----------------------------------------------------------------------------
# HPM module
# -----------------------------------------------------------------------------
class HomeostaticPredictiveMemory(nn.Module):
    """Surprise-gated working memory with homeostatic slot state."""

    def __init__(self, d_model: int, cfg: HPMConfig):
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        self.slot_dim = cfg.slot_dim if cfg.slot_dim is not None else d_model
        self.n_slots = cfg.n_slots

        # Per-slot one-step-ahead predictor: h_{t-1} -> h_t prediction.
        self.next_h_predictor = SlotLinear(self.n_slots, d_model, d_model)

        # --- Learned surprise subspace (Phase B) ---
        self._use_surprise_proj = cfg.surprise_dim > 0 and not cfg.factorized_surprise
        if self._use_surprise_proj:
            self.surprise_proj = SlotLinear(self.n_slots, d_model, cfg.surprise_dim)

        # --- Factorized surprise channels (item #3) ---
        self._use_factorized = cfg.factorized_surprise
        if self._use_factorized:
            _fs_dim = cfg.surprise_dim if cfg.surprise_dim > 0 else d_model
            self._fs_dim = _fs_dim
            self.surprise_heads = nn.ModuleList([
                SlotLinear(self.n_slots, d_model, _fs_dim)
                for _ in range(cfg.n_surprise_channels)
            ])

        # Slot state embedding.
        self.state_embed = nn.Embedding(3, cfg.state_embed_dim)

        # Gate MLP: input dimension depends on factorized surprise.
        if cfg.factorized_surprise:
            _z_dim = cfg.n_surprise_channels
        else:
            _z_dim = 1
        gate_in = d_model + _z_dim + cfg.state_embed_dim
        self.gate_fc1 = SlotLinear(self.n_slots, gate_in, cfg.gate_hidden)
        self.gate_fc2 = SlotLinear(self.n_slots, cfg.gate_hidden, 1)

        # Per-slot write encoder.
        self.write_encoder = SlotLinear(self.n_slots, d_model, self.slot_dim)

        # Learnable initial slot state.
        self.w0 = nn.Parameter(torch.zeros(self.n_slots, self.slot_dim))

        if cfg.read_mode == "attn":
            self.read_query = nn.Linear(d_model, self.slot_dim)

        # --- Content-conditional write gating (Phase C) ---
        if cfg.content_gating:
            self.slot_key = nn.Parameter(torch.randn(self.n_slots, self.slot_dim) * 0.02)
            self.content_query = SlotLinear(self.n_slots, d_model, self.slot_dim)

        # --- Learnable state gains (Phase D) ---
        if cfg.learnable_gains:
            # Initialize via inverse-softplus so softplus(logit) ≈ original gain.
            def _inv_softplus(x: float) -> float:
                return math.log(math.exp(x) - 1.0) if x > 0 else 0.0
            self.gain_logits = nn.Parameter(torch.tensor([
                _inv_softplus(cfg.gain_open),
                _inv_softplus(cfg.gain_closing),
                _inv_softplus(cfg.gain_locked),
            ]))

        # --- C3: retroactive binding ---
        if cfg.retroactive_window > 0:
            # Learned mixing weights over the last W hidden states.
            # MLP: h_t -> (W,) softmax weights.  Write becomes weighted
            # combination of recent hidden states, projected through write_encoder.
            self.retro_mix = SlotLinear(self.n_slots, d_model, cfg.retroactive_window)

        # --- C4: multi-timescale ---
        if cfg.slot_timescales is not None:
            assert len(cfg.slot_timescales) == self.n_slots, (
                f"slot_timescales length {len(cfg.slot_timescales)} != n_slots {self.n_slots}"
            )
            self.register_buffer(
                "timescale_vec",
                torch.tensor(cfg.slot_timescales, dtype=torch.float32),
            )

        # =====================================================================
        # HPM v2 — top-k routing (item #2)
        # =====================================================================
        self._use_topk = cfg.topk_routing
        if cfg.topk_routing:
            # Slot keys for content routing (shared with content_gating if both on,
            # but topk_routing subsumes content_gating).
            if not cfg.content_gating:
                self.slot_key = nn.Parameter(torch.randn(self.n_slots, self.slot_dim) * 0.02)
                self.content_query = SlotLinear(self.n_slots, d_model, self.slot_dim)
            # Per-channel surprise bonus weights (learned).
            if cfg.factorized_surprise:
                self.routing_beta = nn.Parameter(
                    torch.full((cfg.n_surprise_channels,), cfg.routing_surprise_bonus)
                )
            # Routing buffers (not parameters — updated in-place).
            self.register_buffer("_sticky", torch.zeros(self.n_slots))
            self.register_buffer("_load", torch.zeros(self.n_slots))
            self.register_buffer("_age", torch.zeros(self.n_slots))

        # =====================================================================
        # HPM v2 — per-state running stats (item #7)
        # =====================================================================
        self._use_per_state_stats = cfg.per_state_stats

        # --- buffers ---
        if cfg.per_state_stats:
            # (3, n_slots) — one mu/sigma per state per slot.
            self.register_buffer("mu", torch.zeros(3, self.n_slots))
            self.register_buffer("sigma", torch.ones(3, self.n_slots))
        else:
            self.register_buffer("mu", torch.zeros(self.n_slots))
            self.register_buffer("sigma", torch.ones(self.n_slots))
        self.register_buffer("gate_ema", torch.full((self.n_slots,), 0.5))
        self.register_buffer("slot_state", torch.zeros(self.n_slots, dtype=torch.long))
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))
        self.register_buffer("force_unlock_count", torch.zeros((), dtype=torch.long))

        # Probe mode: when True, stores per-step gate and z-score data
        # for emergent capability analysis. Only enable during eval.
        self._probe_mode: bool = False
        self._probe_gates: Optional[Tensor] = None  # (B, T, n_slots)
        self._probe_z: Optional[Tensor] = None       # (B, T, n_slots)

        # --- Continuous plasticity (v2: cleanup, replaces state machine) ---
        self._use_continuous_plasticity = cfg.continuous_plasticity
        if cfg.continuous_plasticity:
            self.register_buffer("slot_age", torch.zeros(self.n_slots))
            # Learned scale and bias for the plasticity sigmoid.
            self.plasticity_scale = nn.Parameter(torch.tensor(1.0))
            self.plasticity_bias = nn.Parameter(torch.tensor(1.0))
            # Running write share for load balance in non-topk path.
            if cfg.gate_load_balance > 0 and not cfg.topk_routing:
                self.register_buffer("_gate_load", torch.zeros(self.n_slots))

    # ---- public ----------------------------------------------------------
    @property
    def output_dim(self) -> int:
        if self.cfg.read_mode == "concat":
            return self.n_slots * self.slot_dim
        return self.slot_dim

    def reset_running_stats(self) -> None:
        with torch.no_grad():
            self.mu.zero_()
            self.sigma.fill_(1.0)
            self.gate_ema.fill_(0.5)
            self.slot_state.zero_()
            self.global_step.zero_()
            self.force_unlock_count.zero_()
            if self._use_topk:
                self._sticky.zero_()
                self._load.zero_()
                self._age.zero_()
            if self._use_continuous_plasticity:
                self.slot_age.zero_()
                if hasattr(self, '_gate_load'):
                    self._gate_load.zero_()

    def describe_state(self) -> str:
        return " ".join(
            f"s{s}:{_STATE_NAMES[int(self.slot_state[s].item())]}"
            for s in range(self.n_slots)
        )

    # ---- internal --------------------------------------------------------
    @torch.no_grad()
    def _advance_state_machine(self) -> None:
        cfg = self.cfg
        g = self.gate_ema
        new_state = self.slot_state.clone()
        for s in range(self.n_slots):
            cur = int(self.slot_state[s].item())
            gs = float(g[s].item())
            if cur == STATE_OPEN:
                if gs < cfg.open_to_closing_rate:
                    new_state[s] = STATE_CLOSING
            elif cur == STATE_CLOSING:
                if gs < cfg.closing_to_locked_rate:
                    new_state[s] = STATE_LOCKED
                elif gs > cfg.unlock_rate:
                    new_state[s] = STATE_OPEN
            else:  # LOCKED
                if gs > cfg.unlock_rate:
                    new_state[s] = STATE_OPEN
        self.slot_state.copy_(new_state)

    def _state_gain_vec(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        if self.cfg.learnable_gains:
            gains_all = F.softplus(self.gain_logits)  # (3,)
            return gains_all[self.slot_state].to(device=device, dtype=dtype)
        cfg = self.cfg
        gains = torch.empty(self.n_slots, device=device, dtype=dtype)
        for s in range(self.n_slots):
            cur = int(self.slot_state[s].item())
            if cur == STATE_OPEN:
                gains[s] = cfg.gain_open
            elif cur == STATE_CLOSING:
                gains[s] = cfg.gain_closing
            else:
                gains[s] = cfg.gain_locked
        return gains

    # ---- forward ---------------------------------------------------------
    def forward(self, h_seq: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        """h_seq: (B, T, D). Returns (hpm_seq, diagnostics)."""
        B, T, D = h_seq.shape
        device = h_seq.device
        dtype = h_seq.dtype
        cfg = self.cfg

        if T == 0:
            return (
                torch.zeros(B, 0, self.output_dim, device=device, dtype=dtype),
                {},
            )

        # --- Batched precomputation ---
        h_flat = h_seq.reshape(B * T, D)
        preds_flat = torch.einsum("bi,sio->bso", h_flat.detach(), self.next_h_predictor.weight) \
                     + self.next_h_predictor.bias.unsqueeze(0)
        preds = preds_flat.view(B, T, self.n_slots, D)

        # --- Surprise computation ---
        h_target = h_seq.detach()
        shifted_preds = torch.zeros_like(preds)
        if T > 1:
            shifted_preds[:, 1:, :, :] = preds[:, :-1, :, :]

        # Factorized surprise: n_channels independent prediction heads.
        err_channels_all: Optional[List[Tensor]] = None
        if self._use_factorized:
            err_channels_all = []
            h_target_flat = h_target.reshape(B * T, D)
            for head in self.surprise_heads:
                proj_t = torch.einsum("bi,sio->bso", h_target_flat, head.weight) + head.bias.unsqueeze(0)
                proj_t = proj_t.view(B, T, self.n_slots, self._fs_dim)
                proj_p = torch.einsum("btsd,sdo->btso", shifted_preds, head.weight)
                proj_p = proj_p + head.bias.unsqueeze(0).unsqueeze(0)
                err_c = (proj_p - proj_t).pow(2).mean(dim=-1).detach()
                err_c[:, 0, :] = 0.0
                err_channels_all.append(err_c)
            # Aggregate error for stats tracking: mean across channels.
            err_all = torch.stack(err_channels_all, dim=0).mean(dim=0)  # (B, T, n_slots)
        elif self._use_surprise_proj:
            proj_target = torch.einsum(
                "bi,sio->bso", h_target.reshape(B * T, D), self.surprise_proj.weight
            ) + self.surprise_proj.bias.unsqueeze(0)
            proj_target = proj_target.view(B, T, self.n_slots, cfg.surprise_dim)
            proj_preds = torch.einsum(
                "btsd,sdo->btso", shifted_preds, self.surprise_proj.weight,
            )
            proj_preds = proj_preds + self.surprise_proj.bias.unsqueeze(0).unsqueeze(0)
            err_all = (proj_preds - proj_target).pow(2).mean(dim=-1).detach()
        else:
            err_all = (shifted_preds - h_target.unsqueeze(2)).pow(2).mean(dim=-1).detach()
        err_all[:, 0, :] = 0.0

        # Write candidates precomputed.
        use_retroactive = cfg.retroactive_window > 0
        if not use_retroactive:
            writes_flat = torch.einsum("bi,sio->bso", h_flat, self.write_encoder.weight) \
                          + self.write_encoder.bias.unsqueeze(0)
            writes_all = writes_flat.view(B, T, self.n_slots, self.slot_dim)
        else:
            writes_all = None

        use_timescale = cfg.slot_timescales is not None
        if use_timescale:
            ts_vec = self.timescale_vec.unsqueeze(0)

        # Content routing precomputation (used by both content_gating and topk_routing).
        _has_slot_keys = hasattr(self, 'slot_key')
        if _has_slot_keys:
            cq_flat = torch.einsum("bi,sio->bso", h_flat, self.content_query.weight) \
                      + self.content_query.bias.unsqueeze(0)
            content_queries_all = cq_flat.view(B, T, self.n_slots, self.slot_dim)
            _inv_sqrt_d = 1.0 / math.sqrt(self.slot_dim)

        # --- Recurrent loop setup ---
        w_prev = self.w0.unsqueeze(0).expand(B, -1, -1).contiguous()
        in_warmup = bool(self.global_step.item() < cfg.warmup_steps) if self.training else False
        cur_state = self.slot_state.detach().clone().to(device)
        slot_idx = torch.arange(self.n_slots, device=device)

        # Non-per-state sigma (precomputed once, used when per_state_stats is False).
        if not self._use_per_state_stats:
            _floor = max(cfg.sigma_hard_floor, cfg.sigma_floor)
            sigma_safe = self.sigma.clamp(min=_floor)

        # Accumulators.
        per_step_slot: List[Tensor] = []
        z_per_step_list: List[Tensor] = []
        gate_running_sum = torch.zeros(self.n_slots, device=device)
        z_abs_sum = torch.zeros(self.n_slots, device=device)
        z_abs_max = torch.zeros(self.n_slots, device=device)
        err_sum = torch.zeros(self.n_slots, device=device)
        err_sq_sum = torch.zeros(self.n_slots, device=device)
        write_mag_sum = torch.zeros(self.n_slots, device=device)
        write_regular_sum = torch.zeros(self.n_slots, device=device)
        write_forced_sum = torch.zeros(self.n_slots, device=device)
        force_unlocks_step = 0
        plasticity_sum = torch.zeros(self.n_slots, device=device)
        _probe_gate_list: List[Tensor] = []
        _probe_z_list: List[Tensor] = []

        # Continuous plasticity state (v2: cleanup).
        if self._use_continuous_plasticity:
            cp_slot_age = self.slot_age.clone()
            if hasattr(self, '_gate_load'):
                cp_gate_load = self._gate_load.clone()

        # Per-state stats accumulators (item #7).
        if self._use_per_state_stats:
            err_per_state = torch.zeros(3, self.n_slots, device=device)
            err_sq_per_state = torch.zeros(3, self.n_slots, device=device)
            count_per_state = torch.zeros(3, self.n_slots, device=device)

        # Top-k routing state (item #2).
        if self._use_topk:
            topk_sticky = self._sticky.clone()
            topk_load = self._load.clone()
            topk_age = self._age.clone()

        def _state_gain_from(state_tensor: Tensor, out_dtype: torch.dtype) -> Tensor:
            if cfg.learnable_gains:
                gains_all = F.softplus(self.gain_logits)
                return gains_all[state_tensor].to(dtype=out_dtype)
            gains = torch.empty(self.n_slots, device=device, dtype=out_dtype)
            for s in range(self.n_slots):
                cur = int(state_tensor[s].item())
                if cur == STATE_OPEN:
                    gains[s] = cfg.gain_open
                elif cur == STATE_CLOSING:
                    gains[s] = cfg.gain_closing
                else:
                    gains[s] = cfg.gain_locked
            return gains

        # =================================================================
        # Recurrent loop
        # =================================================================
        for t in range(T):
            h_t = h_seq[:, t]                          # (B, D)
            e_t = err_all[:, t]                        # (B, n_slots)

            # --- Z-score computation ---
            if self._use_per_state_stats:
                mu_sel = self.mu[cur_state, slot_idx]      # (n_slots,)
                sig_sel = self.sigma[cur_state, slot_idx]  # (n_slots,)
                _floor_ps = max(cfg.sigma_hard_floor, cfg.sigma_floor)
                sig_sel = sig_sel.clamp(min=_floor_ps)
                z_t = (e_t - mu_sel.unsqueeze(0)) / sig_sel.unsqueeze(0)
            else:
                z_t = (e_t - self.mu.unsqueeze(0)) / sigma_safe.unsqueeze(0)

            # Per-channel z-scores (factorized surprise, item #3).
            z_multi: Optional[Tensor] = None
            if self._use_factorized and err_channels_all is not None:
                z_list = []
                for c in range(cfg.n_surprise_channels):
                    e_c = err_channels_all[c][:, t]  # (B, n_slots)
                    if self._use_per_state_stats:
                        z_c = (e_c - mu_sel.unsqueeze(0)) / sig_sel.unsqueeze(0)
                    else:
                        z_c = (e_c - self.mu.unsqueeze(0)) / sigma_safe.unsqueeze(0)
                    z_list.append(z_c)
                z_multi = torch.stack(z_list, dim=-1)  # (B, n_slots, n_channels)

            # ============================================================
            # Gate / routing computation
            # ============================================================
            if self._use_topk:
                # --- Top-k routing (item #2) ---
                cq = content_queries_all[:, t]  # (B, n_slots, slot_dim)
                content_logit = (cq * self.slot_key.unsqueeze(0)).sum(dim=-1) * _inv_sqrt_d

                if z_multi is not None:
                    surprise_bonus = (z_multi.abs() * self.routing_beta.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
                else:
                    surprise_bonus = cfg.routing_surprise_bonus * z_t.abs()

                route_logits = (
                    content_logit
                    + surprise_bonus
                    + cfg.routing_stickiness * topk_sticky.unsqueeze(0)
                    - cfg.routing_load_balance * topk_load.unsqueeze(0)
                    - cfg.routing_age_penalty * topk_age.unsqueeze(0)
                )
                route_weights = torch.softmax(route_logits, dim=-1)  # (B, n_slots)

                # Top-k masking: keep only k highest.
                _, topk_idx = route_weights.topk(min(cfg.topk_k, self.n_slots), dim=-1)
                topk_mask = torch.zeros_like(route_weights)
                topk_mask.scatter_(-1, topk_idx, 1.0)
                g_t = route_weights * topk_mask

                # Update routing buffers.
                with torch.no_grad():
                    # Stickiness: which slot was top-1 this step.
                    top1_idx = route_weights.argmax(dim=-1)  # (B,)
                    top1_onehot = torch.zeros(self.n_slots, device=device)
                    for b in range(B):
                        top1_onehot[top1_idx[b]] += 1.0
                    top1_onehot = (top1_onehot / max(B, 1)).clamp(max=1.0)
                    topk_sticky = top1_onehot

                    # Load: EMA of per-slot write share.
                    write_share = topk_mask.mean(dim=0)  # (n_slots,)
                    topk_load = 0.95 * topk_load + 0.05 * write_share

                    # Age: increment for non-written slots, reset for written.
                    was_written = topk_mask.any(dim=0)  # (n_slots,)
                    topk_age = torch.where(was_written, torch.zeros_like(topk_age), topk_age + 1.0)

                g_before_force = g_t.detach().clone()
                # No force-unlock in topk mode — age penalty handles it.

            else:
                # --- Original gate logic (with continuous plasticity option) ---

                # Retroactive binding: compute h_mix BEFORE gate so the
                # cause-window content influences the gate decision (#4 fix).
                h_gate_input = h_t  # default: current hidden state
                if use_retroactive:
                    W = cfg.retroactive_window
                    start = max(0, t - W + 1)
                    h_window = h_seq[:, start:t + 1]
                    W_actual = h_window.shape[1]
                    mix_logits = torch.einsum("bi,sio->bso", h_t, self.retro_mix.weight) \
                                 + self.retro_mix.bias.unsqueeze(0)
                    mix_logits = mix_logits[:, :, :W_actual]
                    mix_weights = torch.softmax(mix_logits, dim=-1)
                    h_mix = torch.einsum("bsw,bwd->bsd", mix_weights, h_window)
                    # Use the retro-mixed representation for gate input.
                    # h_mix is (B, n_slots, D) — average across slots for gate.
                    h_gate_input = h_mix.mean(dim=1)  # (B, D)

                if self._use_continuous_plasticity:
                    # --- Continuous plasticity (v2: cleanup, #2+#3) ---
                    # Gate MLP: use zero-vector for state embed (dims preserved).
                    h_b = h_gate_input.unsqueeze(1).expand(-1, self.n_slots, -1)
                    zero_st = torch.zeros(B, self.n_slots, cfg.state_embed_dim,
                                          device=device, dtype=dtype)
                    if z_multi is not None:
                        z_in = z_multi
                    else:
                        z_in = z_t.unsqueeze(-1)
                    gate_inp = torch.cat([h_b, z_in, zero_st], dim=-1)
                    gh = torch.einsum("bsi,sio->bso", gate_inp, self.gate_fc1.weight) \
                         + self.gate_fc1.bias.unsqueeze(0)
                    gh = F.gelu(gh)
                    gate_logits = (torch.einsum("bsi,sio->bso", gh, self.gate_fc2.weight)
                                   + self.gate_fc2.bias.unsqueeze(0)).squeeze(-1)
                    g_t = torch.sigmoid(gate_logits)

                    # Continuous plasticity scalar: age × content_mismatch × surprise.
                    age_factor = 1.0 - torch.exp(-cp_slot_age / cfg.plasticity_age_tau)  # (n_slots,)
                    # Content mismatch: 1 - cosine_sim(content_query, slot_content).
                    if _has_slot_keys:
                        cq_t = content_queries_all[:, t]  # (B, n_slots, slot_dim)
                        w_norm = F.normalize(w_prev, dim=-1)
                        cq_norm = F.normalize(cq_t, dim=-1)
                        content_mismatch = 1.0 - (cq_norm * w_norm).sum(dim=-1)  # (B, n_slots) in [0, 2]
                    else:
                        # Fallback: L2 distance between write-encoded h_t and slot.
                        w_t_proj = writes_all[:, t]  # (B, n_slots, slot_dim) — already projected
                        content_mismatch = (w_t_proj - w_prev).pow(2).mean(dim=-1)
                    surprise_factor = z_t.abs()  # (B, n_slots)

                    raw_plasticity = (
                        self.plasticity_scale
                        * age_factor.unsqueeze(0)
                        * content_mismatch
                        * surprise_factor
                        - self.plasticity_bias
                    )
                    plasticity = torch.sigmoid(raw_plasticity)  # (B, n_slots) in [0, 1]
                    g_t = g_t * plasticity

                    # Load balance penalty (non-topk path).
                    if cfg.gate_load_balance > 0 and _has_slot_keys:
                        cq_t_ = content_queries_all[:, t] if not _has_slot_keys else cq_t
                        relevance = (cq_t_ * self.slot_key.unsqueeze(0)).sum(dim=-1) * _inv_sqrt_d
                        content_weight = torch.softmax(
                            relevance - cfg.gate_load_balance * cp_gate_load.unsqueeze(0),
                            dim=-1,
                        )
                        g_t = g_t * content_weight

                    g_before_force = g_t.detach().clone()
                    # No force-unlock — continuous plasticity handles it.

                    # Update slot age: increment for all, reset for significantly-written.
                    with torch.no_grad():
                        write_strength = g_t.detach().mean(dim=0)  # (n_slots,)
                        was_written_cp = write_strength > 0.3
                        cp_slot_age = torch.where(was_written_cp,
                                                  torch.zeros_like(cp_slot_age),
                                                  cp_slot_age + 1.0)
                        if hasattr(self, '_gate_load'):
                            cp_gate_load = 0.95 * cp_gate_load + 0.05 * write_strength
                        plasticity_sum += plasticity.detach().mean(dim=0)

                else:
                    # --- Legacy state-machine gate ---
                    state_emb = self.state_embed(cur_state)
                    h_b = h_gate_input.unsqueeze(1).expand(-1, self.n_slots, -1)
                    st_b = state_emb.unsqueeze(0).expand(B, -1, -1)

                    if z_multi is not None:
                        z_in = z_multi
                    else:
                        z_in = z_t.unsqueeze(-1)

                    gate_inp = torch.cat([h_b, z_in, st_b], dim=-1)
                    gh = torch.einsum("bsi,sio->bso", gate_inp, self.gate_fc1.weight) \
                         + self.gate_fc1.bias.unsqueeze(0)
                    gh = F.gelu(gh)
                    gate_logits = (torch.einsum("bsi,sio->bso", gh, self.gate_fc2.weight)
                                   + self.gate_fc2.bias.unsqueeze(0)).squeeze(-1)

                    if cfg.competitive and self.n_slots > 1:
                        base_prob = torch.softmax(gate_logits, dim=-1)
                        urgency = torch.sigmoid(gate_logits.max(dim=-1, keepdim=True).values)
                        g_t = base_prob * urgency
                    else:
                        g_t = torch.sigmoid(gate_logits)

                    state_gain = _state_gain_from(cur_state, out_dtype=g_t.dtype)
                    g_t = g_t * state_gain.unsqueeze(0)

                    g_before_force = g_t.detach().clone()

                    if not in_warmup:
                        z_mag = z_t.abs()
                        if cfg.min_surprise_threshold > 0:
                            force_mask_batch = (z_mag > cfg.critical_z) & (e_t > cfg.min_surprise_threshold)
                        else:
                            force_mask_batch = z_mag > cfg.critical_z
                        if force_mask_batch.any():
                            boost = force_mask_batch.to(g_t.dtype)
                            g_t = torch.clamp(g_t + boost, max=1.0)
                            force_mask_slot = force_mask_batch.any(dim=0)
                            if force_mask_slot.any():
                                cur_state = torch.where(
                                    force_mask_slot,
                                    torch.full_like(cur_state, STATE_OPEN),
                                    cur_state,
                                )
                                force_unlocks_step += int(force_mask_slot.sum().item())

                    # Content gating (Phase C) — skipped when topk handles it.
                    if cfg.content_gating and _has_slot_keys:
                        cq = content_queries_all[:, t]
                        relevance = (cq * self.slot_key.unsqueeze(0)).sum(dim=-1) * _inv_sqrt_d
                        content_weight = torch.softmax(relevance, dim=-1)
                        g_t = g_t * content_weight

            if self._probe_mode:
                _probe_gate_list.append(g_t.detach())
                _probe_z_list.append(z_t.detach())

            if use_timescale:
                g_t = g_t / ts_vec

            # --- Write ---
            if use_retroactive:
                # h_mix was already computed before the gate (retroactive fix).
                writes = torch.einsum("bsi,sio->bso", h_mix, self.write_encoder.weight) \
                         + self.write_encoder.bias.unsqueeze(0)
            else:
                writes = writes_all[:, t]

            z_per_step_list.append(z_t.detach())

            g_exp = g_t.unsqueeze(-1)
            w_new = (1.0 - g_exp) * w_prev + g_exp * writes
            per_step_slot.append(w_new)
            w_prev = w_new

            # --- Accumulate diagnostics ---
            with torch.no_grad():
                gate_running_sum += g_t.mean(dim=0)
                z_abs_sum += z_t.abs().mean(dim=0)
                z_abs_max = torch.maximum(z_abs_max, z_t.abs().amax(dim=0))
                err_sum += e_t.mean(dim=0)
                err_sq_sum += e_t.pow(2).mean(dim=0)
                w_mag = writes.detach().pow(2).mean(dim=-1).mean(dim=0)
                write_mag_sum += w_mag
                g_forced_delta = (g_t.detach() - g_before_force).clamp(min=0.0)
                write_regular_sum += (g_before_force.mean(dim=0) * w_mag)
                write_forced_sum += (g_forced_delta.mean(dim=0) * w_mag)

                # Per-state stats accumulation (item #7).
                if self._use_per_state_stats:
                    for s in range(self.n_slots):
                        st_s = int(cur_state[s].item())
                        err_per_state[st_s, s] += e_t[:, s].mean()
                        err_sq_per_state[st_s, s] += e_t[:, s].pow(2).mean()
                        count_per_state[st_s, s] += 1.0

        # =================================================================
        # Post-loop
        # =================================================================
        slot_seq = torch.stack(per_step_slot, dim=1)  # (B, T, n_slots, slot_dim)

        if self._probe_mode and _probe_gate_list:
            self._probe_gates = torch.stack(_probe_gate_list, dim=1)
            self._probe_z = torch.stack(_probe_z_list, dim=1)
        else:
            self._probe_gates = None
            self._probe_z = None

        # --- Read ---
        if cfg.read_mode == "concat":
            hpm_seq = slot_seq.reshape(B, T, self.n_slots * self.slot_dim)
        elif cfg.read_mode == "mean":
            hpm_seq = slot_seq.mean(dim=2)
        elif cfg.read_mode == "attn":
            q = self.read_query(h_seq)
            attn_logits = torch.einsum("btd,btsd->bts", q, slot_seq)
            attn = torch.softmax(attn_logits, dim=-1).unsqueeze(-1)
            hpm_seq = (attn * slot_seq).sum(dim=2)
        else:
            raise ValueError(f"Unknown read_mode: {cfg.read_mode}")

        # --- Update running stats (training only) ---
        if self.training and T > 0:
            with torch.no_grad():
                decay = cfg.ema_decay
                if self._use_per_state_stats and not self._use_continuous_plasticity:
                    # Update mu/sigma per state (item #7).
                    for st in range(3):
                        mask = count_per_state[st] > 0
                        if mask.any():
                            cnt = count_per_state[st][mask]
                            b_mu = err_per_state[st][mask] / cnt
                            b_var = (err_sq_per_state[st][mask] / cnt) - b_mu.pow(2)
                            b_sigma = b_var.clamp(min=0.0).sqrt() + cfg.sigma_floor
                            self.mu[st][mask] = self.mu[st][mask] * decay + (1 - decay) * b_mu
                            self.sigma[st][mask] = self.sigma[st][mask] * decay + (1 - decay) * b_sigma
                else:
                    batch_mu = err_sum / float(T)
                    batch_var = (err_sq_sum / float(T)) - batch_mu.pow(2)
                    batch_var = batch_var.clamp(min=0.0)
                    batch_sigma = batch_var.sqrt() + cfg.sigma_floor
                    self.mu.mul_(decay).add_((1.0 - decay) * batch_mu)
                    self.sigma.mul_(decay).add_((1.0 - decay) * batch_sigma)

                self.gate_ema.mul_(decay).add_((1.0 - decay) * (gate_running_sum / float(T)))
                self.force_unlock_count += int(force_unlocks_step)
                self.global_step += 1

                if self._use_continuous_plasticity:
                    # Persist continuous plasticity buffers.
                    self.slot_age.copy_(cp_slot_age)
                    if hasattr(self, '_gate_load'):
                        self._gate_load.copy_(cp_gate_load)
                else:
                    # Legacy state machine update.
                    self.slot_state.copy_(cur_state.detach())
                    self._advance_state_machine()

                # Persist top-k routing buffers.
                if self._use_topk:
                    self._sticky.copy_(topk_sticky)
                    self._load.copy_(topk_load)
                    self._age.copy_(topk_age)
        else:
            if force_unlocks_step > 0 and not self._use_continuous_plasticity:
                with torch.no_grad():
                    self.slot_state.copy_(cur_state.detach())

        # Force-unlock write decomposition fractions.
        total_write_contrib = write_regular_sum + write_forced_sum
        safe_total = total_write_contrib.clamp(min=1e-8)
        regular_frac = float((write_regular_sum / safe_total).mean().item())
        forced_frac = float((write_forced_sum / safe_total).mean().item())

        diag: Dict[str, float] = {
            "hpm_gate_mean": float((gate_running_sum / float(T)).mean().item()),
            "hpm_z_abs_mean": float((z_abs_sum / float(T)).mean().item()),
            "hpm_z_abs_max": float(z_abs_max.max().item()),
            "hpm_err_mean": float((err_sum / float(T)).mean().item()),
            "hpm_write_mag": float((write_mag_sum / float(T)).mean().item()),
            "hpm_open_frac": float((self.slot_state == STATE_OPEN).float().mean().item()),
            "hpm_closing_frac": float((self.slot_state == STATE_CLOSING).float().mean().item()),
            "hpm_locked_frac": float((self.slot_state == STATE_LOCKED).float().mean().item()),
            "hpm_force_unlocks_step": float(force_unlocks_step),
            "hpm_mu": float(self.mu.mean().item()),
            "hpm_sigma": float(self.sigma.mean().item()),
            "hpm_write_regular_frac": regular_frac,
            "hpm_write_forced_frac": forced_frac,
        }

        # Continuous plasticity diagnostics.
        if self._use_continuous_plasticity:
            diag["hpm_plasticity_mean"] = float((plasticity_sum / float(T)).mean().item())
            diag["hpm_slot_age_mean"] = float(cp_slot_age.mean().item())

        # Slot diversity regularizer (item #6).
        if cfg.slot_diversity_lambda > 0 and _has_slot_keys:
            keys_norm = F.normalize(self.slot_key, dim=-1)
            cos_sim = keys_norm @ keys_norm.T
            mask = ~torch.eye(self.n_slots, dtype=torch.bool, device=device)
            diversity_loss = cfg.slot_diversity_lambda * cos_sim[mask].mean()
            diag["_diversity_loss"] = diversity_loss  # Tensor — extracted in training loop.
            diag["hpm_slot_key_cos"] = float(cos_sim[mask].mean().item())

        # Top-k routing diagnostics.
        if self._use_topk:
            diag["hpm_load_std"] = float(topk_load.std().item())
            diag["hpm_age_mean"] = float(topk_age.mean().item())

        # Expose per-step z-scores for EventTape consumption.
        z_per_step = torch.stack(z_per_step_list, dim=1) if z_per_step_list else None
        diag["_z_per_step"] = z_per_step  # Tensor or None — consumed by EventTape.

        return hpm_seq, diag


@dataclass
class StructuredStateConfig:
    """Authoritative per-entity world state for identity-style queries."""

    enabled: bool = False
    n_entities: int = 6
    d_state: int = 96
    route_temperature: float = 1.0
    holder_head_hidden: int = 64


@dataclass
class TypedEventLogConfig:
    """Typed transition log for historical and counterfactual queries."""

    enabled: bool = False
    max_events: int = 32
    score_threshold: float = 0.40
    z_score_bonus: float = 0.15
    include_uniform_fallback: bool = True
    n_event_types: int = 7
    max_time_embeddings: int = 512


@dataclass
class StateCheckpointConfig:
    """Checkpointed state snapshots used for temporal reconstruction."""

    enabled: bool = False
    n_checkpoints: int = 16
    max_time_embeddings: int = 512


@dataclass
class StructuredStateOutput:
    sequence: Tensor
    memory_tokens: Tensor
    state_stack: Tensor
    holder_logits: Tensor
    tagged_logits: Tensor
    visible_logits: Tensor
    route_weights: Tensor
    diagnostics: Dict[str, float]


@dataclass
class TypedEventLogOutput:
    entries: Tensor
    mask: Tensor
    times: Tensor
    type_ids: Tensor
    event_type_logits: Tensor
    prev_holder_logits: Tensor
    next_holder_logits: Tensor
    event_scores: Tensor
    diagnostics: Dict[str, float]


@dataclass
class StateCheckpointOutput:
    entries: Tensor
    mask: Tensor
    times: Tensor
    holder_logits: Tensor
    diagnostics: Dict[str, float]


class StructuredStateTable(nn.Module):
    """
    Explicit per-entity state tracker.

    This is the v2 replacement for "entity table as just another memory bank":
    it keeps a persistent per-entity state, emits direct holder/tag/visibility
    logits each step, and exposes entity memory tokens for query-time lookup.
    """

    def __init__(self, d_model: int, cfg: StructuredStateConfig):
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        self.n_entities = cfg.n_entities
        self.d_state = cfg.d_state

        self.entity_keys = nn.Parameter(torch.randn(cfg.n_entities, d_model) * 0.02)
        self.input_proj = nn.Linear(d_model, cfg.d_state)
        self.gru = nn.GRUCell(cfg.d_state, cfg.d_state)
        self.state0 = nn.Parameter(torch.zeros(cfg.n_entities, cfg.d_state))
        self.memory_proj = nn.Linear(cfg.d_state, d_model)
        self.sequence_proj = nn.Linear(cfg.n_entities * d_model, d_model)

        hidden = max(32, int(cfg.holder_head_hidden))
        self.holder_head = nn.Sequential(
            nn.Linear(cfg.d_state, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.tagged_head = nn.Sequential(
            nn.Linear(cfg.d_state, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.visible_head = nn.Sequential(
            nn.Linear(cfg.d_state, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h_seq: Tensor) -> StructuredStateOutput:
        B, T, D = h_seq.shape
        device = h_seq.device
        dtype = h_seq.dtype
        n_e = self.n_entities

        if T == 0:
            zero_state = torch.zeros(B, 0, n_e, self.d_state, device=device, dtype=dtype)
            zero_mem = torch.zeros(B, 0, n_e, D, device=device, dtype=dtype)
            zero_seq = torch.zeros(B, 0, D, device=device, dtype=dtype)
            zero_logits = torch.zeros(B, 0, n_e, device=device, dtype=dtype)
            zero_routes = torch.zeros(B, 0, n_e, device=device, dtype=dtype)
            return StructuredStateOutput(
                sequence=zero_seq,
                memory_tokens=zero_mem,
                state_stack=zero_state,
                holder_logits=zero_logits,
                tagged_logits=zero_logits,
                visible_logits=zero_logits,
                route_weights=zero_routes,
                diagnostics={},
            )

        route_temp = max(1e-4, float(self.cfg.route_temperature))
        route_logits = torch.einsum("btd,ed->bte", h_seq, self.entity_keys) / (
            math.sqrt(D) * route_temp
        )
        route_weights = torch.softmax(route_logits, dim=-1)
        projected = self.input_proj(h_seq)

        state = self.state0.unsqueeze(0).expand(B, -1, -1).contiguous()
        states: List[Tensor] = []
        memory_tokens: List[Tensor] = []
        holder_logits: List[Tensor] = []
        tagged_logits: List[Tensor] = []
        visible_logits: List[Tensor] = []

        for t in range(T):
            routed_in = route_weights[:, t].unsqueeze(-1) * projected[:, t].unsqueeze(1)
            state = self.gru(
                routed_in.reshape(B * n_e, self.d_state),
                state.reshape(B * n_e, self.d_state),
            ).reshape(B, n_e, self.d_state)
            mem_t = self.memory_proj(state)
            states.append(state)
            memory_tokens.append(mem_t)
            holder_logits.append(self.holder_head(state).squeeze(-1))
            tagged_logits.append(self.tagged_head(state).squeeze(-1))
            visible_logits.append(self.visible_head(state).squeeze(-1))

        state_stack = torch.stack(states, dim=1)
        memory_stack = torch.stack(memory_tokens, dim=1)
        holder_stack = torch.stack(holder_logits, dim=1)
        tagged_stack = torch.stack(tagged_logits, dim=1)
        visible_stack = torch.stack(visible_logits, dim=1)
        pooled = self.sequence_proj(memory_stack.reshape(B, T, n_e * self.d_model))

        diag = {
            "structured_state_route_entropy": float(
                (-(route_weights * (route_weights + 1e-8).log()).sum(dim=-1).mean()).item()
            ),
            "structured_state_norm": float(state_stack.detach().norm(dim=-1).mean().item()),
            "structured_holder_entropy": float(
                (-(holder_stack.softmax(dim=-1) * holder_stack.log_softmax(dim=-1)).sum(dim=-1).mean()).item()
            ),
        }
        return StructuredStateOutput(
            sequence=pooled,
            memory_tokens=memory_stack,
            state_stack=state_stack,
            holder_logits=holder_stack,
            tagged_logits=tagged_stack,
            visible_logits=visible_stack,
            route_weights=route_weights,
            diagnostics=diag,
        )


class TypedEventLog(nn.Module):
    """
    Typed event extractor.

    Stores discrete transition candidates rather than raw surprise snapshots.
    The all-step logits are supervised in the trainer; the selected entries are
    used by the v2 query head for temporal lookup and counterfactual questions.
    """

    EVENT_TYPES: Tuple[str, ...] = (
        "none",
        "handoff",
        "trigger",
        "alarm",
        "chain2_fire",
        "correction",
        "state_change",
    )

    def __init__(self, d_model: int, n_entities: int, cfg: TypedEventLogConfig):
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        self.n_entities = n_entities

        self.type_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, cfg.n_event_types),
        )
        self.prev_holder_head = nn.Linear(d_model, n_entities)
        self.next_holder_head = nn.Linear(d_model, n_entities)
        self.type_embed = nn.Embedding(cfg.n_event_types, d_model)
        self.time_embed = nn.Embedding(cfg.max_time_embeddings, d_model)
        self.entry_proj = nn.Linear(d_model * 2 + n_entities, d_model)

    def forward(
        self,
        sequence: Tensor,
        structured: StructuredStateOutput,
        z_per_step: Optional[Tensor] = None,
    ) -> TypedEventLogOutput:
        B, T, D = sequence.shape
        device = sequence.device
        dtype = sequence.dtype

        event_type_logits = self.type_head(sequence)
        prev_holder_logits = self.prev_holder_head(sequence)
        next_holder_logits = self.next_holder_head(sequence)

        non_none_prob = 1.0 - event_type_logits.softmax(dim=-1)[..., 0]
        event_scores = non_none_prob
        if z_per_step is not None:
            z_bonus = z_per_step.abs().max(dim=-1).values
            z_bonus = z_bonus / z_bonus.amax(dim=1, keepdim=True).clamp_min(1.0)
            event_scores = event_scores + self.cfg.z_score_bonus * z_bonus

        max_events = min(self.cfg.max_events, max(1, T))
        entries = torch.zeros(B, max_events, D, device=device, dtype=dtype)
        mask = torch.zeros(B, max_events, dtype=torch.bool, device=device)
        times = torch.zeros(B, max_events, dtype=torch.long, device=device)
        type_ids = torch.zeros(B, max_events, dtype=torch.long, device=device)

        holder_probs = structured.holder_logits.softmax(dim=-1)
        state_summary = structured.sequence

        for b in range(B):
            chosen = (event_scores[b] >= self.cfg.score_threshold).nonzero(as_tuple=False).squeeze(-1)
            if chosen.numel() == 0 and self.cfg.include_uniform_fallback:
                k = max_events
                chosen = torch.arange(min(T, k), device=device, dtype=torch.long)
            elif chosen.numel() > max_events:
                _, top_idx = event_scores[b, chosen].topk(max_events)
                chosen = chosen[top_idx].sort().values
            n = int(chosen.numel())
            if n == 0:
                continue
            cur_types = event_type_logits[b, chosen].argmax(dim=-1)
            raw = torch.cat(
                [sequence[b, chosen], state_summary[b, chosen], holder_probs[b, chosen]],
                dim=-1,
            )
            cur_entries = self.entry_proj(raw)
            cur_entries = cur_entries + self.type_embed(cur_types) + self.time_embed(
                chosen.clamp(max=self.cfg.max_time_embeddings - 1)
            )
            entries[b, :n] = cur_entries
            mask[b, :n] = True
            times[b, :n] = chosen
            type_ids[b, :n] = cur_types

        diag = {
            "typed_event_fill_rate": float(mask.float().mean().item()),
            "typed_event_score_mean": float(event_scores.mean().item()),
            "typed_event_non_none": float((event_type_logits.argmax(dim=-1) != 0).float().mean().item()),
        }
        return TypedEventLogOutput(
            entries=entries,
            mask=mask,
            times=times,
            type_ids=type_ids,
            event_type_logits=event_type_logits,
            prev_holder_logits=prev_holder_logits,
            next_holder_logits=next_holder_logits,
            event_scores=event_scores,
            diagnostics=diag,
        )


class StateCheckpointBank(nn.Module):
    """Checkpointed state snapshots selected from event scores plus uniform fallback."""

    def __init__(self, d_model: int, n_entities: int, cfg: StateCheckpointConfig):
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        self.n_entities = n_entities
        self.entry_proj = nn.Linear(d_model + n_entities, d_model)
        self.time_embed = nn.Embedding(cfg.max_time_embeddings, d_model)

    def forward(
        self,
        structured: StructuredStateOutput,
        event_scores: Optional[Tensor] = None,
    ) -> StateCheckpointOutput:
        state_seq = structured.sequence
        holder_logits = structured.holder_logits
        B, T, D = state_seq.shape
        device = state_seq.device
        dtype = state_seq.dtype
        K = min(self.cfg.n_checkpoints, max(1, T))

        entries = torch.zeros(B, K, D, device=device, dtype=dtype)
        mask = torch.zeros(B, K, dtype=torch.bool, device=device)
        times = torch.zeros(B, K, dtype=torch.long, device=device)
        checkpoint_holders = torch.zeros(B, K, self.n_entities, device=device, dtype=dtype)

        uniform_idx = torch.linspace(0, max(T - 1, 0), steps=K, device=device).long()
        for b in range(B):
            if event_scores is not None:
                k_ev = min(max(1, K // 2), T)
                _, ev_idx = event_scores[b].topk(k_ev)
                chosen = torch.unique(torch.cat([ev_idx, uniform_idx], dim=0), sorted=True)
            else:
                chosen = uniform_idx
            if chosen.numel() > K:
                chosen = chosen[:K]
            n = int(chosen.numel())
            if n == 0:
                continue
            raw = torch.cat(
                [state_seq[b, chosen], holder_logits[b, chosen].softmax(dim=-1)],
                dim=-1,
            )
            cur_entries = self.entry_proj(raw) + self.time_embed(
                chosen.clamp(max=self.cfg.max_time_embeddings - 1)
            )
            entries[b, :n] = cur_entries
            checkpoint_holders[b, :n] = holder_logits[b, chosen]
            mask[b, :n] = True
            times[b, :n] = chosen

        diag = {
            "state_checkpoint_fill_rate": float(mask.float().mean().item()),
            "state_checkpoint_count": float(mask.sum(dim=-1).float().mean().item()),
        }
        return StateCheckpointOutput(
            entries=entries,
            mask=mask,
            times=times,
            holder_logits=checkpoint_holders,
            diagnostics=diag,
        )


__all__ = [
    "HomeostaticPredictiveMemory",
    "HPMConfig",
    "EntityTable",
    "EntityTableConfig",
    "EventTape",
    "EventTapeConfig",
    "EntityHistoryBank",
    "EntityHistoryConfig",
    "StructuredStateConfig",
    "TypedEventLogConfig",
    "StateCheckpointConfig",
    "StructuredStateOutput",
    "TypedEventLogOutput",
    "StateCheckpointOutput",
    "StructuredStateTable",
    "TypedEventLog",
    "StateCheckpointBank",
    "SlotLinear",
    "STATE_OPEN",
    "STATE_CLOSING",
    "STATE_LOCKED",
]
