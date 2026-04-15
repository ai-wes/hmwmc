from __future__ import annotations

import logging
import math
import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping, Optional


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

    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = "%H:%M:%S", use_color: Optional[bool] = None):
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
        stream = None,
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

    def log_score(self, name: str, value: float, metric_spec: MetricSpec, level: int = logging.INFO) -> ScoredValue:
        scored = self.score(name, value, metric_spec)
        self.logger.log(level, name, extra={"scored_value": scored})
        return scored

    def log_scores(self, metrics: Mapping[str, float], specs: Mapping[str, MetricSpec], level: int = logging.INFO) -> list[ScoredValue]:
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
        "loss/text": MetricSpec(ScoreDirection.LOWER_IS_BETTER, 0.0, 2.0),
        "loss/vision": MetricSpec(ScoreDirection.LOWER_IS_BETTER, 0.0, 0.5),
        "loss/audio": MetricSpec(ScoreDirection.LOWER_IS_BETTER, 0.0, 0.5),
        "loss/numeric": MetricSpec(ScoreDirection.LOWER_IS_BETTER, 0.0, 0.5),
        "text_entropy": MetricSpec(ScoreDirection.LOWER_IS_BETTER, 0.0, 2.0),
        "episodic_read_entropy": MetricSpec(ScoreDirection.LOWER_IS_BETTER, 1.5, 3.5),
        "holder_acc": MetricSpec(ScoreDirection.HIGHER_IS_BETTER, 0.0, 1.0),
        "qacc/who_holds_token": MetricSpec(ScoreDirection.HIGHER_IS_BETTER, 0.0, 1.0),
        "qacc/who_was_first_tagged": MetricSpec(ScoreDirection.HIGHER_IS_BETTER, 0.0, 1.0),
        "qacc/did_alarm_fire": MetricSpec(ScoreDirection.HIGHER_IS_BETTER, 0.0, 1.0),
        "qacc/which_entity_occluded": MetricSpec(ScoreDirection.HIGHER_IS_BETTER, 0.0, 1.0),
        "qacc/what_was_true_rule": MetricSpec(ScoreDirection.HIGHER_IS_BETTER, 0.0, 1.0),
        "qacc/did_trigger_before_alarm": MetricSpec(ScoreDirection.HIGHER_IS_BETTER, 0.0, 1.0),
        "qacc/which_entity_first_occluded": MetricSpec(ScoreDirection.HIGHER_IS_BETTER, 0.0, 1.0),
        "qacc/did_chain2_fire": MetricSpec(ScoreDirection.HIGHER_IS_BETTER, 0.0, 1.0),
        # HPM diagnostics
        "hpm_gate_mean": MetricSpec(ScoreDirection.LOWER_IS_BETTER, 0.0, 1.0),
        "hpm_z_abs_mean": MetricSpec(ScoreDirection.LOWER_IS_BETTER, 0.0, 5.0),
        "hpm_z_abs_max": MetricSpec(ScoreDirection.LOWER_IS_BETTER, 0.0, 10.0),
        "hpm_err_mean": MetricSpec(ScoreDirection.LOWER_IS_BETTER, 0.0, 5.0),
        "hpm_write_mag": MetricSpec(ScoreDirection.LOWER_IS_BETTER, 0.0, 3.0),
        "hpm_force_unlocks_step": MetricSpec(ScoreDirection.LOWER_IS_BETTER, 0.0, 10.0),
        "hpm_mu": MetricSpec(ScoreDirection.LOWER_IS_BETTER, -3.0, 3.0),
        "hpm_sigma": MetricSpec(ScoreDirection.HIGHER_IS_BETTER, 0.0, 2.0),
        "hpm_write_regular_frac": MetricSpec(ScoreDirection.HIGHER_IS_BETTER, 0.0, 1.0),
        "hpm_write_forced_frac": MetricSpec(ScoreDirection.LOWER_IS_BETTER, 0.0, 1.0),
    }


# Metric display groups — controls section ordering in log output.
# Each tuple is (section_label, list_of_metric_prefixes_or_names).
# Metrics are matched in order; first match wins. Unmatched go to "Other".
_METRIC_GROUPS: list[tuple[str, list[str]]] = [
    ("Loss",     ["total", "next_step", "aux_latent", "q_loss", "holder_loss",
                  "loss/", "text_entropy", "stress"]),
    ("Accuracy", ["latent_acc", "entity_acc", "binary_acc", "holder_acc",
                  "qacc/", "who_holds_token", "who_was_first_tagged",
                  "what_was_true_rule"]),
    ("PNN",      ["pnn_"]),
    ("HPM",      ["hpm_"]),
]


def _group_metrics(
    metrics: Mapping[str, float],
) -> list[tuple[str, list[str]]]:
    """Partition metric keys into ordered groups. Every key appears exactly once."""
    buckets: dict[str, list[str]] = {label: [] for label, _ in _METRIC_GROUPS}
    buckets["Other"] = []
    assigned: set[str] = set()

    for key in metrics:
        placed = False
        for label, prefixes in _METRIC_GROUPS:
            for pfx in prefixes:
                if key == pfx or key.startswith(pfx):
                    buckets[label].append(key)
                    assigned.add(key)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            buckets["Other"].append(key)

    result: list[tuple[str, list[str]]] = []
    for label, _ in _METRIC_GROUPS:
        if buckets[label]:
            result.append((label, buckets[label]))
    if buckets["Other"]:
        result.append(("Other", buckets["Other"]))
    return result


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
    # Entropy-relative scoring for loss/text: if text_entropy is present,
    # re-anchor min_value so scoring measures excess above the irreducible floor.
    if "text_entropy" in metrics and "loss/text" in metric_specs:
        floor = metrics["text_entropy"]
        metric_specs["loss/text"] = MetricSpec(
            ScoreDirection.LOWER_IS_BETTER, floor, floor + 1.0,
        )
    score_logger.logger.info(f"=== {step_label} ===")

    groups = _group_metrics(metrics)
    use_color = any(
        getattr(h, "formatter", None) and getattr(h.formatter, "use_color", False)
        for h in score_logger.logger.handlers
    )
    for group_label, keys in groups:
        if use_color:
            score_logger.logger.info(f"  {_Ansi.CYAN}{_Ansi.BOLD}── {group_label} ──{_Ansi.RESET}")
        else:
            score_logger.logger.info(f"  ── {group_label} ──")
        group_metrics = {k: metrics[k] for k in keys}
        score_logger.log_scores(group_metrics, metric_specs)


if __name__ == "__main__":
    logger = ScoreLogger("demo_score_logger")
    specs = build_default_metric_specs()
    demo_metrics = {
        "latent_acc": 0.88,
        "who_holds_token": 0.52,
        "what_was_true_rule": 0.21,
        "next_step": 0.18,
        "stress": 0.04,
    }
    log_training_snapshot(logger, step_label="demo", metrics=demo_metrics, specs=specs)
