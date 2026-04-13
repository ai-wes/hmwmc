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
        gate = CircadianGate(cfg, layer_index=0, num_layers=5, dimension=30)
        gate.update(controller_step)
        if gate.is_day: ...
        noise_scale *= gate.plasticity_multiplier
    """
    def __init__(self,
                 config: CircadianGateConfig,
                 layer_index: Optional[int] = None,
                 num_layers: Optional[int] = None,
                 dimension: Optional[int] = None,
                 **legacy_kwargs):
        if layer_index is None:
            layer_index = legacy_kwargs.pop("island_id", None)
        if num_layers is None:
            num_layers = legacy_kwargs.pop("num_islands", None)
        if legacy_kwargs:
            unexpected = ", ".join(sorted(legacy_kwargs.keys()))
            raise TypeError(f"Unexpected CircadianGate kwargs: {unexpected}")
        if layer_index is None or num_layers is None:
            raise TypeError("CircadianGate requires layer_index and num_layers.")
        self.config = config
        self.period = int(max(1, config.period))
        base_offset = (float(layer_index) / max(1, int(num_layers))) * self.period
        self.phase_offset = base_offset + _deterministic_jitter(
            layer_index, num_layers, self.period, config.phase_jitter
        )
        self.dimension = int(dimension) if dimension is not None else None
        self.is_day: bool = True
        self.plasticity_multiplier: float = config.day_plasticity

    def update(self, controller_step: int) -> None:
        time_in_cycle = (float(controller_step) + float(self.phase_offset)) % float(self.period)
        highd = (self.dimension is not None
                 and self.dimension >= self.config.high_dim_threshold)
        day_frac = self.config.day_fraction_highd if highd else self.config.day_fraction_static
        self.is_day = time_in_cycle < (self.period * day_frac)
        self.plasticity_multiplier = (
            self.config.day_plasticity if self.is_day else self.config.night_plasticity
        )
