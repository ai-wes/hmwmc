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

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
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

        # Slot state embedding.
        self.state_embed = nn.Embedding(3, cfg.state_embed_dim)

        # Gate MLP: [h_t ; z ; state_embed] -> scalar per slot.
        gate_in = d_model + 1 + cfg.state_embed_dim
        self.gate_fc1 = SlotLinear(self.n_slots, gate_in, cfg.gate_hidden)
        self.gate_fc2 = SlotLinear(self.n_slots, cfg.gate_hidden, 1)

        # Per-slot write encoder.
        self.write_encoder = SlotLinear(self.n_slots, d_model, self.slot_dim)

        # Learnable initial slot state.
        self.w0 = nn.Parameter(torch.zeros(self.n_slots, self.slot_dim))

        if cfg.read_mode == "attn":
            self.read_query = nn.Linear(d_model, self.slot_dim)

        # --- buffers ---
        self.register_buffer("mu", torch.zeros(self.n_slots))
        self.register_buffer("sigma", torch.ones(self.n_slots))
        self.register_buffer("gate_ema", torch.full((self.n_slots,), 0.5))
        self.register_buffer("slot_state", torch.zeros(self.n_slots, dtype=torch.long))
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))
        self.register_buffer("force_unlock_count", torch.zeros((), dtype=torch.long))

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
        # Predictions: predictor_s(h_{b,t}) -> shape (B, T, n_slots, D).
        h_flat = h_seq.reshape(B * T, D)
        preds_flat = torch.einsum("bi,sio->bso", h_flat.detach(), self.next_h_predictor.weight) \
                     + self.next_h_predictor.bias.unsqueeze(0)
        preds = preds_flat.view(B, T, self.n_slots, D)

        # e_t = error at step t = || predictor_s(h_{t-1}) - h_t ||^2 averaged.
        h_target = h_seq.detach()
        shifted_preds = torch.zeros_like(preds)
        if T > 1:
            shifted_preds[:, 1:, :, :] = preds[:, :-1, :, :]
        err_all = (shifted_preds - h_target.unsqueeze(2)).pow(2).mean(dim=-1).detach()
        err_all[:, 0, :] = 0.0  # no prior at t=0, no surprise

        # Write candidates precomputed: (B, T, n_slots, slot_dim).
        writes_flat = torch.einsum("bi,sio->bso", h_flat, self.write_encoder.weight) \
                      + self.write_encoder.bias.unsqueeze(0)
        writes_all = writes_flat.view(B, T, self.n_slots, self.slot_dim)

        # --- Recurrent loop ---
        w_prev = self.w0.unsqueeze(0).expand(B, -1, -1).contiguous()
        sigma_safe = self.sigma.clamp(min=cfg.sigma_floor)
        in_warmup = bool(self.global_step.item() < cfg.warmup_steps) if self.training else False

        # Work with a local copy of slot_state. Writing back to the buffer happens
        # at the end -- this avoids in-place mutation of tensors that nn.Embedding
        # saved for backward.
        cur_state = self.slot_state.detach().clone().to(device)

        per_step_slot: List[Tensor] = []
        gate_running_sum = torch.zeros(self.n_slots, device=device)
        z_abs_sum = torch.zeros(self.n_slots, device=device)
        z_abs_max = torch.zeros(self.n_slots, device=device)
        err_sum = torch.zeros(self.n_slots, device=device)
        err_sq_sum = torch.zeros(self.n_slots, device=device)
        write_mag_sum = torch.zeros(self.n_slots, device=device)
        force_unlocks_step = 0

        def _state_gain_from(state_tensor: Tensor, out_dtype: torch.dtype) -> Tensor:
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

        for t in range(T):
            h_t = h_seq[:, t]                          # (B, D)
            e_t = err_all[:, t]                        # (B, n_slots), detached
            z_t = (e_t - self.mu.unsqueeze(0)) / sigma_safe.unsqueeze(0)

            # Fresh per-step embedding lookup on the (possibly just-updated) local state.
            state_emb = self.state_embed(cur_state)     # (n_slots, E)

            h_b = h_t.unsqueeze(1).expand(-1, self.n_slots, -1)
            z_in = z_t.unsqueeze(-1)
            st_b = state_emb.unsqueeze(0).expand(B, -1, -1)
            gate_in = torch.cat([h_b, z_in, st_b], dim=-1)

            gh = torch.einsum("bsi,sio->bso", gate_in, self.gate_fc1.weight) \
                 + self.gate_fc1.bias.unsqueeze(0)
            gh = torch.nn.functional.gelu(gh)
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

            if not in_warmup:
                z_mag = z_t.abs()
                force_mask_batch = z_mag > cfg.critical_z
                if force_mask_batch.any():
                    boost = force_mask_batch.to(g_t.dtype)
                    g_t = torch.clamp(g_t + boost, max=1.0)
                    force_mask_slot = force_mask_batch.any(dim=0)
                    if force_mask_slot.any():
                        # Update the LOCAL state tensor, not the buffer yet.
                        cur_state = torch.where(
                            force_mask_slot,
                            torch.full_like(cur_state, STATE_OPEN),
                            cur_state,
                        )
                        force_unlocks_step += int(force_mask_slot.sum().item())

            writes = writes_all[:, t]
            g_exp = g_t.unsqueeze(-1)
            w_new = (1.0 - g_exp) * w_prev + g_exp * writes
            per_step_slot.append(w_new)
            w_prev = w_new

            with torch.no_grad():
                gate_running_sum += g_t.mean(dim=0)
                z_abs_sum += z_t.abs().mean(dim=0)
                z_abs_max = torch.maximum(z_abs_max, z_t.abs().amax(dim=0))
                err_sum += e_t.mean(dim=0)
                err_sq_sum += e_t.pow(2).mean(dim=0)
                write_mag_sum += writes.detach().pow(2).mean(dim=-1).mean(dim=0)

        slot_seq = torch.stack(per_step_slot, dim=1)  # (B, T, n_slots, slot_dim)

        # --- read ---
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

        # --- update running stats (training only) ---
        if self.training and T > 0:
            with torch.no_grad():
                decay = cfg.ema_decay
                batch_mu = err_sum / float(T)
                batch_var = (err_sq_sum / float(T)) - batch_mu.pow(2)
                batch_var = batch_var.clamp(min=0.0)
                batch_sigma = batch_var.sqrt() + cfg.sigma_floor
                self.mu.mul_(decay).add_((1.0 - decay) * batch_mu)
                self.sigma.mul_(decay).add_((1.0 - decay) * batch_sigma)
                self.gate_ema.mul_(decay).add_((1.0 - decay) * (gate_running_sum / float(T)))
                # Commit cur_state (which reflects any force-unlocks) back to the buffer.
                self.slot_state.copy_(cur_state.detach())
                self.force_unlock_count += int(force_unlocks_step)
                self._advance_state_machine()
                self.global_step += 1
        else:
            # eval mode: still commit force-unlocks if any occurred
            if force_unlocks_step > 0:
                with torch.no_grad():
                    self.slot_state.copy_(cur_state.detach())

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
        }
        return hpm_seq, diag


__all__ = [
    "HomeostaticPredictiveMemory",
    "HPMConfig",
    "SlotLinear",
    "STATE_OPEN",
    "STATE_CLOSING",
    "STATE_LOCKED",
]
