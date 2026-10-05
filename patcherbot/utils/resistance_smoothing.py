from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import numpy as np


CAUSAL_LOG_MEDIAN_EMA_METHOD = "causal_log_median_ema"
DEFAULT_RESISTANCE_FLOOR = 1e-3
DEFAULT_MEDIAN_WINDOW = 5
DEFAULT_EMA_ALPHA = 0.3


def _validate_smoothing_params(
    *,
    window: int,
    alpha: float,
    resistance_floor: float,
) -> tuple[int, float, float]:
    window = int(window)
    alpha = float(alpha)
    resistance_floor = float(resistance_floor)
    if window <= 0:
        raise ValueError("median window must be > 0")
    if not np.isfinite(alpha) or alpha <= 0.0 or alpha > 1.0:
        raise ValueError("EMA alpha must be finite and in (0, 1]")
    if not np.isfinite(resistance_floor) or resistance_floor <= 0.0:
        raise ValueError("resistance floor must be finite and > 0")
    return window, alpha, resistance_floor


def log_resistance_with_floor(
    resistance_values: np.ndarray,
    *,
    resistance_floor: float = DEFAULT_RESISTANCE_FLOOR,
) -> np.ndarray:
    """Return natural log resistance after flooring finite values."""

    resistance_floor = float(resistance_floor)
    if not np.isfinite(resistance_floor) or resistance_floor <= 0.0:
        raise ValueError("resistance floor must be finite and > 0")

    raw = np.asarray(resistance_values, dtype=np.float64).reshape(-1)
    log_values = np.full(raw.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(raw)
    log_values[finite] = np.log(np.maximum(raw[finite], resistance_floor))
    return log_values


def causal_log_median_ema(
    resistance_values: np.ndarray,
    *,
    window: int = DEFAULT_MEDIAN_WINDOW,
    alpha: float = DEFAULT_EMA_ALPHA,
    resistance_floor: float = DEFAULT_RESISTANCE_FLOOR,
) -> np.ndarray:
    """Return causal log-resistance smoothed by trailing median then EMA."""

    smoother = CausalLogMedianEmaSmoother(
        window=window,
        alpha=alpha,
        resistance_floor=resistance_floor,
    )
    raw = np.asarray(resistance_values, dtype=np.float64).reshape(-1)
    return np.asarray([smoother.update(value) for value in raw], dtype=np.float64)


@dataclass(slots=True)
class CausalLogMedianEmaSmoother:
    """Incremental causal smoother for live resistance readings."""

    window: int = DEFAULT_MEDIAN_WINDOW
    alpha: float = DEFAULT_EMA_ALPHA
    resistance_floor: float = DEFAULT_RESISTANCE_FLOOR
    _buffer: Deque[float] = field(init=False, repr=False)
    _smooth_log_resistance: float = field(default=np.nan, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.window, self.alpha, self.resistance_floor = _validate_smoothing_params(
            window=self.window,
            alpha=self.alpha,
            resistance_floor=self.resistance_floor,
        )
        self._buffer = deque(maxlen=self.window)

    @property
    def smooth_log_resistance(self) -> float:
        return float(self._smooth_log_resistance)

    def reset(self) -> None:
        self._buffer.clear()
        self._smooth_log_resistance = np.nan
        self._initialized = False

    def update(self, resistance: float) -> float:
        raw = float(resistance)
        if np.isfinite(raw):
            log_value = float(np.log(max(raw, self.resistance_floor)))
        else:
            log_value = np.nan

        self._buffer.append(log_value)
        buffer_values = np.asarray(self._buffer, dtype=np.float64)
        finite = buffer_values[np.isfinite(buffer_values)]
        if finite.size == 0:
            self._smooth_log_resistance = np.nan
            self._initialized = False
            return float(self._smooth_log_resistance)

        median_log_resistance = float(np.median(finite))
        if not self._initialized or not np.isfinite(self._smooth_log_resistance):
            self._smooth_log_resistance = median_log_resistance
            self._initialized = True
        else:
            self._smooth_log_resistance = (
                self.alpha * median_log_resistance
                + (1.0 - self.alpha) * self._smooth_log_resistance
            )
        return float(self._smooth_log_resistance)


__all__ = [
    "CAUSAL_LOG_MEDIAN_EMA_METHOD",
    "DEFAULT_EMA_ALPHA",
    "DEFAULT_MEDIAN_WINDOW",
    "DEFAULT_RESISTANCE_FLOOR",
    "CausalLogMedianEmaSmoother",
    "causal_log_median_ema",
    "log_resistance_with_floor",
]
