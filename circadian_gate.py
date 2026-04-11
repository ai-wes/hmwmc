# circadian_gate.py
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Optional


@dataclass(frozen=True)
class CircadianGateConfig:
    period: int = 3                        # generations per cycle
    day_fraction_static: float = 0.25      # fraction of the cycle that is "day" in low/med D
    day_fraction_highd: float = 0.45       # wider edit window for high-D
    high_dim_threshold: int = 30
    phase_jitter: float = 0.10             # +/- % of period
    day_plasticity: float = 1.0
    night_plasticity: float = 0.5


def _deterministic_jitter(island_id: int, num_islands: int, period: int, jitter_frac: float) -> float:
    """
    Stable jitter in [-jitter_frac*period/2, +jitter_frac*period/2] based on ids (no RNG/global hash).
    """
    key = f"{island_id}:{num_islands}:{period}:{jitter_frac}".encode("utf-8")
    h = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    # map to [-0.5, 0.5]
    unit = (h / (2**64 - 1)) - 0.5
    return unit * (jitter_frac * period)


class CircadianGate:
    """
    Lightweight circadian rhythm controller.

    Usage:
        cfg = CircadianGateConfig()
        gate = CircadianGate(cfg, island_id=0, num_islands=5, dimension=30)
        gate.update(generation)
        if gate.is_day: ...
        noise_scale *= gate.plasticity_multiplier
    """
    def __init__(self,
                 config: CircadianGateConfig,
                 island_id: int,
                 num_islands: int,
                 dimension: Optional[int] = None):
        self.config = config
        self.period = int(max(1, config.period))
        base_offset = (float(island_id) / max(1, int(num_islands))) * self.period
        self.phase_offset = base_offset + _deterministic_jitter(
            island_id, num_islands, self.period, config.phase_jitter
        )
        self.dimension = int(dimension) if dimension is not None else None
        self.is_day: bool = True
        self.plasticity_multiplier: float = config.day_plasticity

    def update(self, generation: int) -> None:
        time_in_cycle = (float(generation) + float(self.phase_offset)) % float(self.period)
        highd = (self.dimension is not None
                 and self.dimension >= self.config.high_dim_threshold)
        day_frac = self.config.day_fraction_highd if highd else self.config.day_fraction_static
        self.is_day = time_in_cycle < (self.period * day_frac)
        self.plasticity_multiplier = (
            self.config.day_plasticity if self.is_day else self.config.night_plasticity
        )
