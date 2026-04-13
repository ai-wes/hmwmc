"""
TMEW-1 runner: ties tmew1_train.py + tmew1_queries.py together.

Usage:
    python tmew1_run.py --smoke      # 30-second sanity check
    python tmew1_run.py               # full curriculum

Wires the QueryHead and handoff template into the trainer without modifying
either source file. Logs per-query-type accuracy so you can see which query
families are failing in isolation.
"""

from __future__ import annotations
from typing import Optional 
import argparse
import random
from dataclasses import replace
from typing import Dict, Sequence, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader

from tmew1_train import (
    WorldConfig,
    TrainConfig,
    CurriculumTier,
    DEFAULT_TIERS,
    LatentRuleProbe,
    build_model,
    shift_targets,
)
from tmew1_queries import QueryHead, augment_sequence_with_holder_audio, query_train_step_addon
from tmew1_diagnostics import (
    EXTENDED_QUERY_TYPES,
    EXTENDED_QUERY_TYPE_TO_IDX,
    generate_episode_with_diagnostics,
    episode_to_diag_tensors,
    collate_diag,
    run_diagnostic_report,
)

from homeostatic_multimodal_world_model_chunked import (
    HomeostaticMultimodalWorldModel,
    MultimodalPredictionLoss,
    ForwardOutput,
    LossOutput,
    summarize_controller_report,
)
from score_logging import ScoreLogger, build_default_metric_specs, log_training_snapshot
import tmew1_viz_server as viz


# Module-level flag: set True when --viz is passed
_VIZ_ENABLED = False


# -----------------------------------------------------------------------------
# Dataset that emits query-augmented episodes
# -----------------------------------------------------------------------------
class TMEW1QueryDataset(Dataset):
    """Generates episodes on first access and caches them for fast repeated epochs."""
    def __init__(self, cfg: WorldConfig, tier: CurriculumTier, n_episodes: int, base_seed: int, num_queries: int = 4):
        self.cfg = cfg
        self.tier = tier
        self.n = n_episodes
        self.base_seed = base_seed
        self.num_queries = num_queries
        self._cache: Dict[int, Dict[str, Tensor]] = {}

    def set_tier(self, tier: CurriculumTier) -> None:
        self.tier = tier
        self._cache.clear()

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        if idx not in self._cache:
            ep = generate_episode_with_diagnostics(
                self.cfg,
                self.tier,
                seed=self.base_seed + idx,
                num_queries=self.num_queries,
                enable_false_cue=self.tier.tier >= 2,
            )
            self._cache[idx] = episode_to_diag_tensors(ep)
        return self._cache[idx]


class PreCachedDataset(Dataset):
    """Pre-generates all episodes and stacks them into contiguous GPU tensors.

    Eliminates per-step CPU episode generation overhead entirely. The full
    dataset lives on-device so batch fetches are just tensor slicing — no
    host-to-device transfer during training.
    """

    def __init__(
        self,
        cfg: WorldConfig,
        tier: CurriculumTier,
        n_episodes: int,
        base_seed: int,
        num_queries: int = 4,
        device: str = "cuda",
    ):
        import logging
        log = logging.getLogger(__name__)
        log.info("Pre-generating %d episodes for tier %d on CPU...", n_episodes, tier.tier)

        # Generate all episodes on CPU first
        all_tensors: list[Dict[str, Tensor]] = []
        for i in range(n_episodes):
            ep = generate_episode_with_diagnostics(
                cfg, tier, seed=base_seed + i,
                num_queries=num_queries,
                enable_false_cue=tier.tier >= 2,
            )
            all_tensors.append(episode_to_diag_tensors(ep))

        # Stack into contiguous tensors and move to device
        keys = list(all_tensors[0].keys())
        self._data: Dict[str, Tensor] = {}
        for k in keys:
            self._data[k] = torch.stack([t[k] for t in all_tensors], dim=0).to(device)
        self._n = n_episodes
        log.info("Pre-cached %d episodes on %s (%.1f MB)",
                 n_episodes, device,
                 sum(t.nbytes for t in self._data.values()) / 1e6)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        return {k: v[idx] for k, v in self._data.items()}


def precached_collate(batch: Sequence[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    """Collate for PreCachedDataset. Data is already on device, just stack."""
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0].keys()}


# -----------------------------------------------------------------------------
# Auxiliary holder head
# -----------------------------------------------------------------------------
class CurrentHolderHead(nn.Module):
    def __init__(self, d_model: int, max_entities: int):
        super().__init__()
        self.proj = nn.Linear(d_model, max_entities)

    def forward(self, sequence: Tensor) -> Tensor:
        return self.proj(sequence)


# -----------------------------------------------------------------------------
# Full query-input builder: base sequence + audio-holder + HPM sequence
# -----------------------------------------------------------------------------
def build_query_input(
    output: ForwardOutput,
    audio: Optional[Tensor],
    max_entities: int,
    use_audio: bool,
) -> Tensor:
    """Concatenate base sequence, holder-audio channels, and HPM sequence."""
    aug = augment_sequence_with_holder_audio(
        output.sequence,
        audio,
        max_entities=max_entities,
        use_audio=use_audio,
    )
    if output.hpm_sequence is not None:
        aug = torch.cat([aug, output.hpm_sequence.to(aug.dtype)], dim=-1)
    return aug

# -----------------------------------------------------------------------------
# Per-query-type accuracy logging
# -----------------------------------------------------------------------------
@torch.no_grad()
def per_qtype_accuracy(
    output: ForwardOutput,
    query_head: QueryHead,
    batch: Dict[str, Tensor],
    holder_feature_dim: int,
    enabled: Sequence[str],
) -> Dict[str, float]:
    t_max = output.sequence.size(1) - 1
    qtimes = batch["query_times"].clamp(max=t_max)
    augmented_seq = build_query_input(
        output,
        batch.get("audio"),
        max_entities=holder_feature_dim,
        use_audio="audio" in enabled,
    )
    entity_logits, binary_logits = query_head(augmented_seq, qtimes, batch["query_types"])

    targets = batch["query_targets"]
    is_binary = batch["query_is_binary"]
    qtypes = batch["query_types"]

    metrics: Dict[str, float] = {}
    for qtype_name, qtype_idx in EXTENDED_QUERY_TYPE_TO_IDX.items():
        mask = qtypes == qtype_idx
        if not mask.any():
            continue
        if is_binary[mask].all():
            preds = binary_logits[mask].argmax(-1)
        else:
            preds = entity_logits[mask].argmax(-1)
        acc = (preds == targets[mask]).float().mean().item()
        metrics[f"qacc/{qtype_name}"] = acc
    return metrics


# -----------------------------------------------------------------------------
# Train / eval loops
# -----------------------------------------------------------------------------
def train_one_step(
    model: HomeostaticMultimodalWorldModel,
    criterion: MultimodalPredictionLoss,
    probe: LatentRuleProbe,
    query_head: QueryHead,
    holder_head: CurrentHolderHead,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, Tensor],
    tcfg: TrainConfig,
    enabled: Sequence[str],
    holder_feature_dim: int,
    scaler: GradScaler | None = None,
    tier_step: int = 0,
    prev_tier_modalities: Sequence[str] = (),
) -> Dict[str, float]:
    optimizer.zero_grad(set_to_none=True)

    # ── Modality warmup: ramp loss weight for newly-introduced modalities ──
    _new_modalities = set(enabled) - set(prev_tier_modalities)
    _saved_weights: Dict[str, float] = {}
    if _new_modalities and tcfg.modality_warmup_steps > 0:
        warmup_factor = min(1.0, tier_step / tcfg.modality_warmup_steps)
        for mod in _new_modalities:
            _saved_weights[mod] = getattr(criterion.weights, mod)
            setattr(criterion.weights, mod, _saved_weights[mod] * warmup_factor)

    vision_in, vision_tgt = shift_targets(batch["vision"])
    audio_in, audio_tgt = shift_targets(batch["audio"])
    numeric_in, numeric_tgt = shift_targets(batch["numeric"])
    text_in, text_tgt = shift_targets(batch["text"])

    with autocast(device_type="cuda", enabled=tcfg.use_amp):
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

        # Restore base loss weights after computing loss (so warmup is per-call)
        for mod, base_w in _saved_weights.items():
            setattr(criterion.weights, mod, base_w)

        rule_logits = probe(output.sequence)
        aux = nn.functional.cross_entropy(rule_logits, batch["latent_rule"])

        q_loss, q_metrics = query_train_step_addon(
                    build_query_input(
                        output,
                        batch.get("audio"),
                        max_entities=holder_feature_dim,
                        use_audio="audio" in enabled,
                    ),
                    query_head,
                    batch,
                    query_type_to_idx=EXTENDED_QUERY_TYPE_TO_IDX,
                    weight=0.5,
                )

        holder_loss = torch.zeros((), device=output.sequence.device)
        holder_acc = 0.0
        if "audio" in enabled and "holder_per_step" in batch:
            holder_logits = holder_head(output.sequence)
            holder_targets = batch["holder_per_step"][:, :-1]
            holder_loss = nn.functional.cross_entropy(
                holder_logits.reshape(-1, holder_logits.size(-1)),
                holder_targets.reshape(-1),
            )
            with torch.no_grad():
                holder_acc = (holder_logits.argmax(-1) == holder_targets).float().mean().item()

        total = losses.total + tcfg.aux_latent_weight * aux + q_loss + 0.3 * holder_loss

    # ── NaN guard: skip backward + optimizer if loss exploded ──────────
    if torch.isnan(total) or torch.isinf(total):
        import logging as _log
        _log.getLogger(__name__).warning(
            "NaN/Inf loss detected (total=%.4g) — skipping backward, "
            "running emergency_stabilize", float(total.item()) if not torch.isnan(total) else float("nan"),
        )
        optimizer.zero_grad(set_to_none=True)
        if model.cfg.enable_online_homeostasis:
            model.controller.emergency_stabilize()
            # Force-unlock all layers so the model can recover
            for pnn in model.controller.pnns:
                pnn.force_unlock(model.controller.global_step, refractory_period=8)
        with torch.no_grad():
            rule_acc = (rule_logits.argmax(-1) == batch["latent_rule"]).float().mean().item()
        return {
            "total": float("nan"), "next_step": float("nan"),
            "aux_latent": float("nan"), "latent_acc": rule_acc,
            "q_loss": float("nan"), "holder_loss": float("nan"),
            "holder_acc": holder_acc, **q_metrics,
        }

    if scaler is not None:
        scaler.scale(total).backward()
        if tcfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            params = list(model.parameters()) + list(probe.parameters()) + list(query_head.parameters()) + list(holder_head.parameters())
            nn.utils.clip_grad_norm_(params, tcfg.grad_clip)
        else:
            scaler.unscale_(optimizer)
            params = list(model.parameters()) + list(probe.parameters()) + list(query_head.parameters()) + list(holder_head.parameters())

        # ── Fix #3: Pre-step finite-grad check ─────────────────────────
        grads_finite = all(
            p.grad is None or torch.isfinite(p.grad).all()
            for p in params
        )
        if not grads_finite:
            import logging as _log
            _log.getLogger(__name__).warning(
                "Non-finite grads detected — skipping optimizer.step()"
            )
            optimizer.zero_grad(set_to_none=True)
            scaler.update()  # still update scaler so it can back off the scale
            with torch.no_grad():
                rule_acc = (rule_logits.argmax(-1) == batch["latent_rule"]).float().mean().item()
            return {
                "total": float(total.item()), "next_step": float(losses.parts.get("text", losses.parts.get("vision", total)).item()),
                "aux_latent": float(aux.item()), "latent_acc": rule_acc,
                "q_loss": float(q_loss.item()), "holder_loss": float(holder_loss.item()),
                "holder_acc": holder_acc, "skipped_step": True, **q_metrics,
            }
        scaler.step(optimizer)
        scaler.update()
    else:
        total.backward()
        if tcfg.grad_clip > 0:
            params = list(model.parameters()) + list(probe.parameters()) + list(query_head.parameters()) + list(holder_head.parameters())
            nn.utils.clip_grad_norm_(params, tcfg.grad_clip)
        else:
            params = list(model.parameters()) + list(probe.parameters()) + list(query_head.parameters()) + list(holder_head.parameters())

        # ── Fix #3 (non-scaler path): Pre-step finite-grad check ──────
        grads_finite = all(
            p.grad is None or torch.isfinite(p.grad).all()
            for p in params
        )
        if not grads_finite:
            import logging as _log
            _log.getLogger(__name__).warning(
                "Non-finite grads detected — skipping optimizer.step()"
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                rule_acc = (rule_logits.argmax(-1) == batch["latent_rule"]).float().mean().item()
            return {
                "total": float(total.item()), "next_step": float(losses.parts.get("text", losses.parts.get("vision", total)).item()),
                "aux_latent": float(aux.item()), "latent_acc": rule_acc,
                "q_loss": float(q_loss.item()), "holder_loss": float(holder_loss.item()),
                "holder_acc": holder_acc, "skipped_step": True, **q_metrics,
            }
        optimizer.step()

    # ── Post-step NaN check: repair poisoned weights + reset Adam state ──
    _has_nan = False
    for p in model.parameters():
        if p.data.isnan().any() or p.data.isinf().any():
            _has_nan = True
            p.data.nan_to_num_(nan=0.0, posinf=1.0, neginf=-1.0)
            # Fix #4: Reset Adam moment buffers for this param so recovery is possible
            if p in optimizer.state:
                optimizer.state[p] = {}
    if _has_nan:
        import logging as _log
        _log.getLogger(__name__).warning(
            "NaN/Inf detected in model weights after optimizer.step — "
            "zeroed weights AND reset Adam state for affected params"
        )
        if model.cfg.enable_online_homeostasis:
            model.controller.emergency_stabilize()

    # controller_step mutates live Parameters, so defer it until autograd is done.
    if model.cfg.enable_online_homeostasis:
        model.controller_step(total.detach())

    with torch.no_grad():
        rule_acc = (rule_logits.argmax(-1) == batch["latent_rule"]).float().mean().item()
    out = {
        "total": float(total.item()),
        "next_step": float(losses.total.item()),
        "aux_latent": float(aux.item()),
        "latent_acc": rule_acc,
        "q_loss": float(q_loss.item()),
        "holder_loss": float(holder_loss.item()),
        "holder_acc": holder_acc,
        **q_metrics,
    }
    for mod_name, mod_loss in losses.parts.items():
        out[f"loss/{mod_name}"] = float(mod_loss.item())
    return out


@torch.no_grad()
def evaluate(
    model: HomeostaticMultimodalWorldModel,
    criterion: MultimodalPredictionLoss,
    probe: LatentRuleProbe,
    query_head: QueryHead,
    holder_head: CurrentHolderHead,
    loader: DataLoader,
    tcfg: TrainConfig,
    enabled: Sequence[str],
    holder_feature_dim: int,
) -> Dict[str, float]:
    model.eval()
    probe.eval()
    query_head.eval()
    holder_head.eval()

    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    n = 0

    for batch in loader:
        batch = {k: v.to(tcfg.device) for k, v in batch.items()}
        bs = batch["latent_rule"].size(0)

        vision_in, vision_tgt = shift_targets(batch["vision"])
        audio_in, audio_tgt = shift_targets(batch["audio"])
        numeric_in, numeric_tgt = shift_targets(batch["numeric"])
        text_in, text_tgt = shift_targets(batch["text"])

        with autocast(device_type="cuda", enabled=tcfg.use_amp):
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
        rule_acc = (rule_logits.argmax(-1) == batch["latent_rule"]).float().mean().item()
        qtype_metrics = per_qtype_accuracy(output, query_head, batch, holder_feature_dim, enabled)
        holder_acc = 0.0
        if "audio" in enabled and "holder_per_step" in batch:
            holder_logits = holder_head(output.sequence)
            holder_targets = batch["holder_per_step"][:, :-1]
            holder_acc = (holder_logits.argmax(-1) == holder_targets).float().mean().item()

        sums["next_step"] = sums.get("next_step", 0.0) + float(losses.total.item()) * bs
        sums["latent_acc"] = sums.get("latent_acc", 0.0) + rule_acc * bs
        if "audio" in enabled and "holder_per_step" in batch:
            sums["holder_acc"] = sums.get("holder_acc", 0.0) + holder_acc * bs
        # Per-modality loss parts
        for mod_name, mod_loss in losses.parts.items():
            k = f"loss/{mod_name}"
            sums[k] = sums.get(k, 0.0) + float(mod_loss.item()) * bs
        # Empirical text token entropy (marginal distribution over val set)
        if "text" in enabled:
            text_flat = text_tgt.reshape(-1)
            text_flat = text_flat[text_flat != 0]  # exclude pad tokens
            if text_flat.numel() > 0:
                sums["_text_tokens"] = sums.get("_text_tokens", 0.0) + text_flat.numel()
                for tok_id in text_flat.unique():
                    k = f"_text_count_{int(tok_id.item())}"
                    sums[k] = sums.get(k, 0.0) + float((text_flat == tok_id).sum().item())
        for k, v in qtype_metrics.items():
            sums[k] = sums.get(k, 0.0) + v * bs
            counts[k] = counts.get(k, 0) + bs
        n += bs

    model.train()
    probe.train()
    query_head.train()
    holder_head.train()

    out = {"next_step": sums["next_step"] / max(1, n), "latent_acc": sums["latent_acc"] / max(1, n)}
    if "holder_acc" in sums:
        out["holder_acc"] = sums["holder_acc"] / max(1, n)
    # Per-modality val losses
    for mk in ("loss/text", "loss/vision", "loss/audio", "loss/numeric"):
        if mk in sums:
            out[mk] = sums[mk] / max(1, n)
    # Empirical text entropy
    if "_text_tokens" in sums and sums["_text_tokens"] > 0:
        import math as _math
        total_tokens = sums["_text_tokens"]
        entropy = 0.0
        for k, cnt in sums.items():
            if k.startswith("_text_count_") and cnt > 0:
                p = cnt / total_tokens
                entropy -= p * _math.log(p)
        out["text_entropy"] = entropy
    for k in sums:
        if k.startswith("qacc/"):
            out[k] = sums[k] / max(1, counts[k])
    return out


# -----------------------------------------------------------------------------
# Curriculum runner
# -----------------------------------------------------------------------------
def run_curriculum(
    world_cfg: WorldConfig,
    tcfg: TrainConfig,
    tiers: Sequence[CurriculumTier] = DEFAULT_TIERS,
    num_queries: int = 4,
    resume_from: Optional[str] = None,
) -> None:
    torch.manual_seed(tcfg.seed)
    np.random.seed(tcfg.seed)
    random.seed(tcfg.seed)

    score_logger = ScoreLogger("tmew_scores")
    specs = build_default_metric_specs()

    model, criterion, probe = build_model(world_cfg)
    # The shared categorical head must cover both entity-id answers and latent-rule answers.
    num_categorical_answers = max(world_cfg.max_entities, world_cfg.num_latent_rules)

    hpm_dim = model.hpm.output_dim if getattr(model, "hpm", None) is not None else 0
    query_head = QueryHead(model.cfg.d_model + world_cfg.max_entities + hpm_dim, num_categorical_answers, len(EXTENDED_QUERY_TYPES))

    holder_head = CurrentHolderHead(model.cfg.d_model, world_cfg.max_entities)

    model = model.to(tcfg.device)
    probe = probe.to(tcfg.device)
    query_head = query_head.to(tcfg.device)
    holder_head = holder_head.to(tcfg.device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(probe.parameters()) + list(query_head.parameters()) + list(holder_head.parameters()),
        lr=tcfg.lr,
        weight_decay=tcfg.weight_decay,
    )

    scaler = GradScaler("cuda", enabled=tcfg.use_amp)

    # ── Resume from checkpoint ──────────────────────────────────────────
    _resume_after_tier = 0  # 0 means start from tier 1
    if resume_from is not None:
        print(f"Loading checkpoint from {resume_from} ...")
        ckpt = torch.load(resume_from, map_location=tcfg.device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        probe.load_state_dict(ckpt["probe"])
        query_head.load_state_dict(ckpt["query_head"])
        holder_head.load_state_dict(ckpt["holder_head"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and tcfg.use_amp:
            scaler.load_state_dict(ckpt["scaler"])
        _resume_after_tier = ckpt.get("tier", 0)
        print(f"  -> resumed from tier {_resume_after_tier} checkpoint")
        del ckpt

    prev_tier_modalities: Tuple[str, ...] = ()
    for tier in tiers:
        # Skip tiers already completed in the checkpoint
        if tier.tier <= _resume_after_tier:
            prev_tier_modalities = tier.enabled_modalities
            print(f"\n=== Tier {tier.tier} | SKIPPED (already completed in checkpoint) ===")
            continue
        if tier.tier >= 2:
            boosted_templates = list(tier.template_pool)
            if "handoff" not in boosted_templates:
                boosted_templates.append("handoff")
            if "multi_chain" not in boosted_templates:
                boosted_templates.append("multi_chain")
            boosted_templates.extend(["handoff", "handoff", "false_cue", "false_cue", "multi_chain"])
            tier = replace(tier, template_pool=tuple(boosted_templates))

        print(f"\n=== Tier {tier.tier} | modalities={tier.enabled_modalities} | T={tier.max_episode_length} | templates={tier.template_pool} ===")

        # ── Fix #2: Reset Adam state + GradScaler at tier boundary ──────
        new_mods = set(tier.enabled_modalities) - set(prev_tier_modalities)
        if new_mods:
            # Identify params belonging to newly-activated modality encoders/heads
            _new_param_ids: set = set()
            for mod_name in new_mods:
                encoder = getattr(model, f"{mod_name}_encoder", None)
                head = getattr(model, f"{mod_name}_head", None)
                if encoder is not None:
                    for p in encoder.parameters():
                        _new_param_ids.add(id(p))
                if head is not None:
                    # Re-init the head with smaller std to cap initial logit magnitudes
                    nn.init.normal_(head.weight, std=0.02)
                    if head.bias is not None:
                        nn.init.zeros_(head.bias)
                    for p in head.parameters():
                        _new_param_ids.add(id(p))
            # Clear Adam moment buffers for those params
            _cleared = 0
            for p in optimizer.state:
                if id(p) in _new_param_ids and optimizer.state[p]:
                    optimizer.state[p] = {}
                    _cleared += 1
            if _cleared:
                print(f"  -> reset Adam state for {_cleared} params in new modalities: {new_mods}")
            # Reset GradScaler so stale scale factor from previous tier doesn't cause issues
            scaler = GradScaler("cuda", enabled=tcfg.use_amp)

        tier_step = 0  # running step counter within this tier (for modality warmup)

        train_ds = PreCachedDataset(world_cfg, tier, tcfg.train_episodes_per_tier, base_seed=1000 * tier.tier, num_queries=num_queries, device=tcfg.device)
        val_ds = PreCachedDataset(world_cfg, tier, tcfg.val_episodes, base_seed=9000 + 1000 * tier.tier, num_queries=num_queries, device=tcfg.device)
        train_loader = DataLoader(
            train_ds, batch_size=tcfg.batch_size, shuffle=True, collate_fn=precached_collate,
            num_workers=0, pin_memory=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=tcfg.batch_size, shuffle=False, collate_fn=precached_collate,
            num_workers=0, pin_memory=False,
        )

        promoted = False
        for epoch in range(tcfg.epochs_per_tier):
            _accum_unlocks = 0  # accumulate force-unlocks between log events
            for step, batch in enumerate(train_loader):
                # Data is already on device from PreCachedDataset
                stats = train_one_step(model, criterion, probe, query_head, holder_head, optimizer, batch, tcfg, tier.enabled_modalities, world_cfg.max_entities, scaler=scaler, tier_step=tier_step, prev_tier_modalities=prev_tier_modalities)
                tier_step += 1
                # Accumulate force-unlocks every step (not just log steps)
                _step_report = summarize_controller_report(model._last_forward_report)
                _accum_unlocks += _step_report.get("unlocked", 0)
                # Skip logging during warmup steps at the start of each tier
                if epoch == 0 and step < tcfg.warmup_steps:
                    continue
                if step % tcfg.log_every == 0:
                    summary = _step_report
                    step_metrics = {
                        "total": stats["total"],
                        "next_step": stats["next_step"],
                        "aux_latent": stats["aux_latent"],
                        "latent_acc": stats["latent_acc"],
                        "q_loss": stats["q_loss"],
                        "holder_loss": stats["holder_loss"],
                        "holder_acc": stats["holder_acc"],
                        "entity_acc": stats.get("entity_acc", 0.0),
                        "binary_acc": stats.get("binary_acc", 0.0),
                    }
                    for _mk in ("loss/text", "loss/vision", "loss/audio", "loss/numeric"):
                        if _mk in stats:
                            step_metrics[_mk] = stats[_mk]
                    stresses = summary.get("stresses") or []
                    if stresses:
                        step_metrics["stress"] = float(np.mean(np.asarray(stresses, dtype=np.float32)))
                    pnn_states = summary.get("pnn_states") or []
                    pnn_str = "/".join(s[0] for s in pnn_states) if pnn_states else ""
                    n_open = sum(1 for s in pnn_states if s == "OPEN") if pnn_states else 0
                    unlock_tag = f" (+{_accum_unlocks} unlocked)" if _accum_unlocks > 0 else ""
                    log_training_snapshot(
                        score_logger,
                        step_label=(
                            f"t{tier.tier} ep{epoch} s{step:04d}"
                            f" | pnn={pnn_str} open={n_open}{unlock_tag}"
                        ),
                        metrics=step_metrics,
                        specs=specs,
                    )
                    _accum_unlocks = 0  # reset after logging
                    if _VIZ_ENABLED:
                        viz.send_metrics(step=step, tier=tier.tier, epoch=epoch, metrics=step_metrics, pnn_states=pnn_str)

            val = evaluate(model, criterion, probe, query_head, holder_head, val_loader, tcfg, tier.enabled_modalities, world_cfg.max_entities)
            log_training_snapshot(
                score_logger,
                step_label=f"[val] t{tier.tier} ep{epoch}",
                metrics=val,
                specs=specs,
            )

            if val["latent_acc"] >= tier.promote_at_accuracy:
                print(f"  -> promoting from tier {tier.tier}")
                promoted = True
                break

        if not promoted:
            print(f"  -> tier {tier.tier} did not reach promotion threshold; continuing anyway")

        prev_tier_modalities = tier.enabled_modalities

        # Save checkpoint after completing each tier
        ckpt_path = f"checkpoint_tier{tier.tier}.pt"
        torch.save({
            "tier": tier.tier,
            "model": model.state_dict(),
            "query_head": query_head.state_dict(),
            "holder_head": holder_head.state_dict(),
            "probe": probe.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
        }, ckpt_path)
        print(f"  -> saved checkpoint to {ckpt_path}")

        run_diagnostic_report(model, query_head, world_cfg, tier, tcfg.device, n_episodes=256)

        # Send a replay episode to the viz dashboard
        if _VIZ_ENABLED:
            viz.send_episode_replay(
                world_cfg=world_cfg, tier=tier,
                model=model, query_head=query_head, device=tcfg.device,
            )


# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------
def smoke_test() -> None:
    """30-second end-to-end pipeline check. Tiny config, single tier, two batches."""
    print("Running smoke test...")
    world_cfg = WorldConfig(grid_h=8, grid_w=8, episode_length=16, max_entities=4, audio_dim=24, numeric_dim=10, num_latent_rules=6)
    tcfg = TrainConfig(
        batch_size=2,
        train_episodes_per_tier=4,
        val_episodes=4,
        epochs_per_tier=1,
        log_every=1,
    )
    smoke_tier = CurriculumTier(
        tier=1,
        max_episode_length=16,
        enabled_modalities=("vision", "numeric", "audio"),
        template_pool=("trigger_delay", "handoff", "multi_chain"),
        max_delay=3,
        occlusion=False,
        promote_at_accuracy=1.01,
    )
    run_curriculum(world_cfg, tcfg, tiers=(smoke_tier,), num_queries=3)
    print("Smoke test complete.")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def main() -> None:
    global _VIZ_ENABLED
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run a 30-second smoke test")
    parser.add_argument("--viz", action="store_true", help="Launch real-time 3D visualization dashboard (ws://0.0.0.0:8765)")
    parser.add_argument("--viz-port", type=int, default=8765, help="WebSocket port for --viz")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-episodes", type=int, default=2048)
    parser.add_argument("--amp", action="store_true", help="Enable mixed-precision (float16) training")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader worker processes")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt file to resume from")
    args = parser.parse_args()

    if args.viz:
        _VIZ_ENABLED = True
        viz.start_server(port=args.viz_port)

    if args.smoke:
        smoke_test()
        return

    world_cfg = WorldConfig()
    tcfg = TrainConfig(
        batch_size=args.batch_size,
        train_episodes_per_tier=args.train_episodes,
        epochs_per_tier=args.epochs,
        use_amp=args.amp and torch.cuda.is_available(),
        num_workers=args.workers,
    )
    run_curriculum(world_cfg, tcfg, resume_from=args.resume)


if __name__ == "__main__":
    main()
