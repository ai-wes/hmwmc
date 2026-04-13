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

import argparse
import logging
import math
import os
import random
import sys
from dataclasses import dataclass, replace
from enum import Enum
from typing import Dict, Mapping, Optional, Sequence
import numpy as np
import torch
from torch import Tensor, nn
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


class ScoreDirection(str, Enum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


@dataclass(frozen=True)
class ScoreThresholds:
    """
    Thresholds are expressed on a normalized 0.0-1.0 quality scale.

    Interpretation:
    - < bad_max            -> red
    - < low_medium_max     -> orange
    - < medium_max         -> yellow
    - < medium_high_max    -> lime green
    - <= 1.0              -> green
    """

    bad_max: float = 0.20
    low_medium_max: float = 0.40
    medium_max: float = 0.60
    medium_high_max: float = 0.80

    def __post_init__(self) -> None:
        values = [self.bad_max, self.low_medium_max, self.medium_max, self.medium_high_max]
        if any(not isinstance(v, (int, float)) for v in values):
            raise TypeError("All threshold values must be numeric.")
        if any(v < 0.0 or v > 1.0 for v in values):
            raise ValueError("All thresholds must be between 0.0 and 1.0.")
        if not (self.bad_max < self.low_medium_max < self.medium_max < self.medium_high_max):
            raise ValueError(
                "Thresholds must be strictly increasing: "
                "bad_max < low_medium_max < medium_max < medium_high_max."
            )


@dataclass(frozen=True)
class MetricSpec:
    """
    Defines how to normalize a raw metric value to a 0.0-1.0 quality score.

    Examples:
    - Accuracy where higher is better and expected range is 0.0-1.0:
        MetricSpec(direction=ScoreDirection.HIGHER_IS_BETTER, min_value=0.0, max_value=1.0)

    - Loss where lower is better and a useful range is 0.0-2.0:
        MetricSpec(direction=ScoreDirection.LOWER_IS_BETTER, min_value=0.0, max_value=2.0)
    """

    direction: ScoreDirection
    min_value: float
    max_value: float

    def __post_init__(self) -> None:
        if not isinstance(self.min_value, (int, float)) or not isinstance(self.max_value, (int, float)):
            raise TypeError("min_value and max_value must be numeric.")
        if not math.isfinite(self.min_value) or not math.isfinite(self.max_value):
            raise ValueError("min_value and max_value must be finite.")
        if self.max_value <= self.min_value:
            raise ValueError("max_value must be greater than min_value.")

    def normalize(self, value: float) -> float:
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            return 0.0

        span = self.max_value - self.min_value
        if self.direction == ScoreDirection.HIGHER_IS_BETTER:
            normalized = (value - self.min_value) / span
        else:
            normalized = (self.max_value - value) / span
        return max(0.0, min(1.0, normalized))


class _Ansi:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    ORANGE = "\033[38;5;208m"
    YELLOW = "\033[33m"
    LIME = "\033[38;5;154m"
    GREEN = "\033[32m"
    CYAN = "\033[36m"
    DIM = "\033[2m"


@dataclass(frozen=True)
class ScoreBand:
    label: str
    color: str


@dataclass(frozen=True)
class ScoredValue:
    name: str
    raw_value: float
    normalized_score: float
    band: ScoreBand


class ScoreColorMapper:
    def __init__(self, thresholds: Optional[ScoreThresholds] = None) -> None:
        self.thresholds = thresholds or ScoreThresholds()
        self.bad = ScoreBand("bad", _Ansi.RED)
        self.low_medium = ScoreBand("low-medium", _Ansi.ORANGE)
        self.medium = ScoreBand("medium", _Ansi.YELLOW)
        self.medium_high = ScoreBand("medium-high", _Ansi.LIME)
        self.good = ScoreBand("good", _Ansi.GREEN)

    def band_for_score(self, normalized_score: float) -> ScoreBand:
        score = max(0.0, min(1.0, normalized_score))
        if score < self.thresholds.bad_max:
            return self.bad
        if score < self.thresholds.low_medium_max:
            return self.low_medium
        if score < self.thresholds.medium_max:
            return self.medium
        if score < self.thresholds.medium_high_max:
            return self.medium_high
        return self.good

    def evaluate(self, name: str, raw_value: float, metric_spec: MetricSpec) -> ScoredValue:
        normalized = metric_spec.normalize(raw_value)
        return ScoredValue(
            name=name,
            raw_value=float(raw_value),
            normalized_score=normalized,
            band=self.band_for_score(normalized),
        )


class ScoreFormatter(logging.Formatter):
    """
    Formatter that colors log messages based on an attached `scored_value` field.

    Usage:
        logger.info("Validation accuracy", extra={"scored_value": scored_value})
    """

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = "%H:%M:%S",
        use_color: Optional[bool] = None,
    ):
        super().__init__(fmt or "%(asctime)s | %(levelname)s | %(message)s", datefmt=datefmt)
        self.use_color = _supports_color(sys.stdout) if use_color is None else use_color

    def format(self, record: logging.LogRecord) -> str:
        original_msg = record.msg
        try:
            scored_value = getattr(record, "scored_value", None)
            if scored_value is not None:
                record.msg = _format_scored_message(scored_value, use_color=self.use_color)
            return super().format(record)
        finally:
            record.msg = original_msg


class ScoreLogger:
    def __init__(
        self,
        name: str = "score_logger",
        *,
        thresholds: Optional[ScoreThresholds] = None,
        level: int = logging.INFO,
        use_color: Optional[bool] = None,
        stream=None,
    ) -> None:
        self.mapper = ScoreColorMapper(thresholds)
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False

        if not self.logger.handlers:
            handler = logging.StreamHandler(stream or sys.stdout)
            handler.setLevel(level)
            handler.setFormatter(ScoreFormatter(use_color=use_color))
            self.logger.addHandler(handler)

    def score(self, name: str, value: float, metric_spec: MetricSpec) -> ScoredValue:
        return self.mapper.evaluate(name, value, metric_spec)

    def log_score(
        self,
        name: str,
        value: float,
        metric_spec: MetricSpec,
        level: int = logging.INFO,
    ) -> ScoredValue:
        scored = self.score(name, value, metric_spec)
        self.logger.log(level, name, extra={"scored_value": scored})
        return scored

    def log_scores(
        self,
        metrics: Mapping[str, float],
        specs: Mapping[str, MetricSpec],
        level: int = logging.INFO,
    ) -> list[ScoredValue]:
        scored_values: list[ScoredValue] = []
        for metric_name, metric_value in metrics.items():
            spec = specs.get(metric_name)
            if spec is None:
                self.logger.log(level, f"{metric_name}: {metric_value:.6f}")
                continue
            scored_values.append(self.log_score(metric_name, metric_value, spec, level=level))
        return scored_values


def _supports_color(stream) -> bool:
    if os.getenv("NO_COLOR"):
        return False
    if os.getenv("FORCE_COLOR"):
        return True
    return hasattr(stream, "isatty") and stream.isatty()


def _format_scored_message(scored_value: ScoredValue, use_color: bool = True) -> str:
    score_pct = scored_value.normalized_score * 100.0
    base = (
        f"{scored_value.name}: "
        f"raw={scored_value.raw_value:.6f} | "
        f"score={score_pct:6.2f}% | "
        f"band={scored_value.band.label}"
    )
    if not use_color:
        return base
    return f"{scored_value.band.color}{_Ansi.BOLD}{base}{_Ansi.RESET}"


def build_default_metric_specs() -> dict[str, MetricSpec]:
    """
    Reasonable defaults for the metrics in your current training logs.
    Adjust ranges if your observed values move outside these windows.
    """
    return {
        "latent_acc": MetricSpec(ScoreDirection.HIGHER_IS_BETTER, 0.0, 1.0),
        "entity_acc": MetricSpec(ScoreDirection.HIGHER_IS_BETTER, 0.0, 1.0),
        "binary_acc": MetricSpec(ScoreDirection.HIGHER_IS_BETTER, 0.0, 1.0),
        "who_holds_token": MetricSpec(ScoreDirection.HIGHER_IS_BETTER, 0.0, 1.0),
        "who_was_first_tagged": MetricSpec(ScoreDirection.HIGHER_IS_BETTER, 0.0, 1.0),
        "what_was_true_rule": MetricSpec(ScoreDirection.HIGHER_IS_BETTER, 0.0, 1.0),
        "next_step": MetricSpec(ScoreDirection.LOWER_IS_BETTER, 0.0, 0.5),
        "total": MetricSpec(ScoreDirection.LOWER_IS_BETTER, 0.0, 5.0),
        "aux_latent": MetricSpec(ScoreDirection.LOWER_IS_BETTER, 0.0, 2.0),
        "q_loss": MetricSpec(ScoreDirection.LOWER_IS_BETTER, 0.0, 2.0),
        "holder_loss": MetricSpec(ScoreDirection.LOWER_IS_BETTER, 0.0, 2.0),
        "stress": MetricSpec(ScoreDirection.LOWER_IS_BETTER, 0.0, 0.15),
        "episodic_read_entropy": MetricSpec(ScoreDirection.LOWER_IS_BETTER, 1.5, 3.5),
        "holder_acc": MetricSpec(ScoreDirection.HIGHER_IS_BETTER, 0.0, 1.0),
        "qacc/who_holds_token": MetricSpec(ScoreDirection.HIGHER_IS_BETTER, 0.0, 1.0),
        "qacc/who_was_first_tagged": MetricSpec(ScoreDirection.HIGHER_IS_BETTER, 0.0, 1.0),
        "qacc/did_alarm_fire": MetricSpec(ScoreDirection.HIGHER_IS_BETTER, 0.0, 1.0),
        "qacc/which_entity_occluded": MetricSpec(ScoreDirection.HIGHER_IS_BETTER, 0.0, 1.0),
        "qacc/what_was_true_rule": MetricSpec(ScoreDirection.HIGHER_IS_BETTER, 0.0, 1.0),
    }


def log_training_snapshot(
    score_logger: ScoreLogger,
    *,
    step_label: str,
    metrics: Mapping[str, float],
    specs: Optional[Mapping[str, MetricSpec]] = None,
) -> None:
    """
    Convenience helper for your training loop.
    """
    metric_specs = dict(specs or build_default_metric_specs())
    score_logger.logger.info(f"=== {step_label} ===")
    score_logger.log_scores(metrics, metric_specs)


# -----------------------------------------------------------------------------
# Dataset that emits query-augmented episodes
# -----------------------------------------------------------------------------
class TMEW1QueryDataset(Dataset):
    def __init__(self, cfg: WorldConfig, tier: CurriculumTier, n_episodes: int, base_seed: int, num_queries: int = 4):
        self.cfg = cfg
        self.tier = tier
        self.n = n_episodes
        self.base_seed = base_seed
        self.num_queries = num_queries

    def set_tier(self, tier: CurriculumTier) -> None:
        self.tier = tier

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        ep = generate_episode_with_diagnostics(
            self.cfg,
            self.tier,
            seed=self.base_seed + idx,
            num_queries=self.num_queries,
            enable_false_cue=self.tier.tier >= 2,
        )
        return episode_to_diag_tensors(ep)


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

    total.backward()
    if tcfg.grad_clip > 0:
        params = list(model.parameters()) + list(probe.parameters()) + list(query_head.parameters()) + list(holder_head.parameters())
        nn.utils.clip_grad_norm_(params, tcfg.grad_clip)
    optimizer.step()

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
        "hpm_diag": output.hpm_diagnostics or {},
        **q_metrics,
    }
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

    for tier in tiers:
        if tier.tier >= 2:
            boosted_templates = list(tier.template_pool)
            if "handoff" not in boosted_templates:
                boosted_templates.append("handoff")
            boosted_templates.extend(["handoff", "handoff", "false_cue", "false_cue"])
            tier = replace(tier, template_pool=tuple(boosted_templates))

        print(f"\n=== Tier {tier.tier} | modalities={tier.enabled_modalities} | T={tier.max_episode_length} | templates={tier.template_pool} ===")

        train_ds = TMEW1QueryDataset(world_cfg, tier, tcfg.train_episodes_per_tier, base_seed=1000 * tier.tier, num_queries=num_queries)
        val_ds = TMEW1QueryDataset(world_cfg, tier, tcfg.val_episodes, base_seed=9000 + 1000 * tier.tier, num_queries=num_queries)
        train_loader = DataLoader(train_ds, batch_size=tcfg.batch_size, shuffle=True, collate_fn=collate_diag, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=tcfg.batch_size, shuffle=False, collate_fn=collate_diag, num_workers=0)

        promoted = False
        for epoch in range(tcfg.epochs_per_tier):
            for step, batch in enumerate(train_loader):
                batch = {k: v.to(tcfg.device) for k, v in batch.items()}
                stats = train_one_step(model, criterion, probe, query_head, holder_head, optimizer, batch, tcfg, tier.enabled_modalities, world_cfg.max_entities)
                if step % tcfg.log_every == 0:
                    summary = summarize_controller_report(model._last_forward_report)
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
                    hpm_diag = stats.get("hpm_diag") or {}
                    for key in ("hpm_gate_mean", "hpm_z_abs_mean", "hpm_z_abs_max",
                                "hpm_sigma", "hpm_locked_frac", "hpm_force_unlocks_step"):
                        if key in hpm_diag:
                            step_metrics[key] = hpm_diag[key]
                    stresses = summary.get("stresses") or []
                    if stresses:
                        step_metrics["stress"] = float(np.mean(np.asarray(stresses, dtype=np.float32)))
                    pnn_states = summary.get("pnn_states") or []
                    pnn_str = "/".join(s[0] for s in pnn_states) if pnn_states else ""
                    hpm_str = model.hpm.describe_state() if getattr(model, "hpm", None) is not None else ""
                    log_training_snapshot(
                        score_logger,
                        step_label=(
                            f"t{tier.tier} ep{epoch} s{step:04d}"
                            f" | pnn={pnn_str} unlocked={summary.get('unlocked', 0)}"
                            f" | hpm={hpm_str}"
                        ),
                        metrics=step_metrics,
                        specs=specs,
                    )

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

        run_diagnostic_report(model, query_head, world_cfg, tier, tcfg.device, n_episodes=256)


# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------
def smoke_test() -> None:
    """30-second end-to-end pipeline check. Tiny config, single tier, two batches."""
    print("Running smoke test...")
    world_cfg = WorldConfig(grid_h=8, grid_w=8, episode_length=12, max_entities=2)
    tcfg = TrainConfig(
        batch_size=2,
        train_episodes_per_tier=4,
        val_episodes=4,
        epochs_per_tier=1,
        log_every=1,
    )
    smoke_tier = CurriculumTier(
        tier=1,
        max_episode_length=12,
        enabled_modalities=("vision", "numeric", "audio"),
        template_pool=("trigger_delay", "handoff"),
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run a 30-second smoke test")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-episodes", type=int, default=2048)
    args = parser.parse_args()

    if args.smoke:
        smoke_test()
        return

    world_cfg = WorldConfig()
    tcfg = TrainConfig(
        batch_size=args.batch_size,
        train_episodes_per_tier=args.train_episodes,
        epochs_per_tier=args.epochs,
    )
    run_curriculum(world_cfg, tcfg)


if __name__ == "__main__":
    main()
