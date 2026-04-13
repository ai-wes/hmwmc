from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Any, Optional
from collections import OrderedDict
import numpy as np

@dataclass(frozen=True)
class RDStressConfig:
    """
    Configuration for per-layer-group stress.
    - stress_threshold: cap for the coefficient-of-variation based stress
    - default_stress: fallback when a layer group lacks enough history
    """
    stress_threshold: float = 0.70
    default_stress: float = 0.30

    @property
    def score_spread_ceiling(self) -> float:
        return self.stress_threshold

    @property
    def baseline_stress(self) -> float:
        return self.default_stress


class RDStressField:
    """
    Per-layer-group stress calculator.

    Expects each tracked layer/unit to have:
        - id: Any hashable id
        - score_history: List[float] (preferred) or fitness_history: List[float]
    This class writes:
        - unit.layer_stress: float in [0, 1]
        - unit.local_stress: float in [0, 1] as a backward-compatible alias

    Usage:
        cfg = RDStressConfig()
        stress = RDStressField(cfg)
        stress.update(layer_groups, controller_step)
        s = stress.stress_values[f"{group_id}_{unit.id}"]
    """
    def __init__(self, config: RDStressConfig):
        self.cfg = config
        self.stress_values: Dict[str, float] = {}

    @staticmethod
    def _get_score_history(unit: Any) -> Optional[Sequence[float]]:
        history = getattr(unit, "score_history", None)
        if history is not None:
            return history
        return getattr(unit, "fitness_history", None)

    def update(
        self,
        layer_groups: Optional[Dict[Any, Sequence[Any]]] = None,
        controller_step: Optional[int] = None,
        **legacy_kwargs: Any,
    ) -> None:
        if layer_groups is None:
            layer_groups = legacy_kwargs.pop("population_by_island", None)
        if controller_step is None:
            controller_step = legacy_kwargs.pop("generation", None)
        if legacy_kwargs:
            unexpected = ", ".join(sorted(legacy_kwargs.keys()))
            raise TypeError(f"Unexpected RDStressField.update kwargs: {unexpected}")
        if layer_groups is None:
            raise TypeError("RDStressField.update requires layer_groups (or population_by_island).")
        if controller_step is None:
            raise TypeError("RDStressField.update requires controller_step (or generation).")

        for group_id, layer_units in layer_groups.items():
            # Latest score values for this layer group.
            group_scores: List[float] = []
            for unit in layer_units:
                history = self._get_score_history(unit)
                if history:
                    group_scores.append(float(history[-1]))

            # Base stress from coefficient of variation (lower score is better).
            if len(group_scores) >= 2:
                m = float(np.mean(group_scores)); s = float(np.std(group_scores))
                if m > 1e-6:
                    cv = s / m
                    base_stress = float(min(cv, self.cfg.stress_threshold))
                else:
                    base_stress = float(self.cfg.stress_threshold)  # degenerate → stressed
            else:
                base_stress = float(self.cfg.default_stress)

            mean_group_score = float(np.mean(group_scores)) if len(group_scores) >= 2 else None

            for unit in layer_units:
                history = self._get_score_history(unit)
                if history:
                    recent_score = float(history[-1])
                    reference_score = mean_group_score if mean_group_score is not None else recent_score
                    reference_score = max(1e-6, abs(reference_score))
                    # Higher when a layer underperforms its group under minimization.
                    if recent_score > reference_score:
                        relative_stress = 1.0 - min(1.0, reference_score / max(recent_score, 1e-6))
                    else:
                        relative_stress = 0.0
                else:
                    relative_stress = 0.0

                stress = 0.7 * base_stress + 0.3 * relative_stress
                stress = float(max(0.0, min(1.0, stress)))
                setattr(unit, "layer_stress", stress)
                setattr(unit, "local_stress", stress)
                self.stress_values[f"{group_id}_{getattr(unit, 'id', 'unknown')}"] = stress


# ----------------------------
# Lightweight immune cache (LRU)
# ----------------------------
@dataclass(frozen=True)
class StressImmuneConfig:
    max_cache: int = 128


class StressImmuneLayer:
    """
    LRU cache of "threat signatures" seen under stress.
    You can use presence in this cache to veto/alter risky mutations, etc.
    """
    def __init__(self, config: StressImmuneConfig = StressImmuneConfig()):
        self.cfg = config
        self.memory_cells: OrderedDict[str, bool] = OrderedDict()

    def add_memory(self, signature: str) -> None:
        if signature in self.memory_cells:
            self.memory_cells.move_to_end(signature)
        else:
            self.memory_cells[signature] = True
            if len(self.memory_cells) > self.cfg.max_cache:
                self.memory_cells.popitem(last=False)

    def has_memory(self, signature: str) -> bool:
        return signature in self.memory_cells


# ----------------------------
# Optional helper for signatures
# ----------------------------
def compute_threat_signature(
    score_history: Optional[List[float]] = None,
    controller_step: Optional[int] = None,
    **legacy_kwargs: Any,
) -> str:
    """
    Simple signature using short-term trend + average.
    """
    if score_history is None:
        score_history = legacy_kwargs.pop("fitness_history", None)
    if controller_step is None:
        controller_step = legacy_kwargs.pop("generation", None)
    if legacy_kwargs:
        unexpected = ", ".join(sorted(legacy_kwargs.keys()))
        raise TypeError(f"Unexpected compute_threat_signature kwargs: {unexpected}")
    if score_history is None or controller_step is None:
        raise TypeError("compute_threat_signature requires score_history and controller_step.")
    if not score_history:
        return f"unknown_{controller_step}"
    recent = score_history[-5:] if len(score_history) >= 5 else score_history
    avg = float(np.mean(recent))
    trend = "improving" if (len(recent) > 1 and recent[-1] < recent[0]) else "declining"
    return f"{trend}_{avg:.6f}_{controller_step}"
