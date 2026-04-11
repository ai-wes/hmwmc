from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Any
from collections import OrderedDict
import numpy as np

@dataclass(frozen=True)
class RDStressConfig:
    """
    Configuration for per-island stress.
    - stress_threshold: cap for the coefficient-of-variation based stress
    - default_stress: fallback when island lacks enough history
    """
    stress_threshold: float = 0.70
    default_stress: float = 0.30


class RDStressField:
    """
    Per-island stress calculator.

    Expects each "cell" to have:
        - id: Any hashable id
        - fitness_history: List[float] (most recent at the end)
    This class writes:
        - cell.local_stress: float in [0, 1]

    Usage:
        cfg = RDStressConfig()
        stress = RDStressField(cfg)
        stress.update(population_by_island, generation)
        s = stress.stress_values[f"{isl}_{cell.id}"]
    """
    def __init__(self, config: RDStressConfig):
        self.cfg = config
        self.stress_values: Dict[str, float] = {}

    def update(self, population_by_island: Dict[int, Sequence[Any]], generation: int) -> None:
        for isl, cells in population_by_island.items():
            # Latest fitness values for this island
            island_fit: List[float] = []
            for c in cells:
                fh = getattr(c, "fitness_history", None)
                if fh: island_fit.append(float(fh[-1]))

            # Base stress from coefficient of variation (minimization assumed)
            if len(island_fit) >= 2:
                m = float(np.mean(island_fit)); s = float(np.std(island_fit))
                if m > 1e-6:
                    cv = s / m
                    base_stress = float(min(cv, self.cfg.stress_threshold))
                else:
                    base_stress = float(self.cfg.stress_threshold)  # degenerate → stressed
            else:
                base_stress = float(self.cfg.default_stress)

            mean_for_rel = float(np.mean(island_fit)) if len(island_fit) >= 2 else None

            for c in cells:
                fh = getattr(c, "fitness_history", None)
                if fh:
                    recent = float(fh[-1])
                    denom = mean_for_rel if mean_for_rel is not None else recent
                    denom = max(1e-6, abs(denom))
                    # higher when cell underperforms island (minimization)
                    relative_stress = 1.0 - min(1.0, max(0.0, recent / denom))
                else:
                    relative_stress = 0.0

                stress = 0.7 * base_stress + 0.3 * relative_stress
                stress = float(max(0.0, min(1.0, stress)))
                setattr(c, "local_stress", stress)
                self.stress_values[f"{isl}_{getattr(c, 'id', 'unknown')}"] = stress


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
def compute_threat_signature(fitness_history: List[float], generation: int) -> str:
    """
    Simple signature using short-term trend + average.
    """
    if not fitness_history:
        return f"unknown_{generation}"
    recent = fitness_history[-5:] if len(fitness_history) >= 5 else fitness_history
    avg = float(np.mean(recent))
    trend = "improving" if (len(recent) > 1 and recent[-1] < recent[0]) else "declining"
    return f"{trend}_{avg:.6f}_{generation}"