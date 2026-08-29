# -*- coding: utf-8 -*-
"""
Gigaseal-specific analysis plots for PatcherBot HDF5 datasets.

This module owns the analyses that are tied to gigaseal pressure/ATM behavior
and resistance-input features. Keep datasetPlotter.py generic.

Examples
--------
Edit the configuration constants below, then run:
    py -3 experiments/Analysis/Rig_Recorder/Reconstruct/GSAnalyzer.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import matplotlib.pyplot as plt
import numpy as np

try:
    from experiments.Analysis.Rig_Recorder.Reconstruct import datasetPlotter as dp
except ImportError:  # Allows running this file directly from its own folder.
    import datasetPlotter as dp


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
FILE_PATH = Path(
    r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent"
    r"\experiments\Datasets\PatcherBot_dataset_v0_966"
    r"\PatcherBot_dataset_v0_966_gigaseal.hdf5"
)

MAX_DEMOS: int | None = None
MAX_SAMPLES: int | None = None
SHOW_PLOTS = True
SAVE_PLOTS = False
OUTPUT_DIR: Path | None = None
PLOT_DPI = 200
PLOT_FORMAT = "png"

ACTION_PATH = "actions"
RESISTANCE_INPUT_PATH = "obs/resistance_input"
RESISTANCE_VALUE_PATH = "obs/resistance"
RESISTANCE_INPUT_ZERO_PADDING_IS_MISSING = True
ACTION_RELATIONSHIP_POINT_SIZE = 11.0
ACTION_RELATIONSHIP_ALPHA = 0.46

PLOT_RESISTANCE_INPUT_ACTION_RELATIONSHIPS = False
PLOT_RESISTANCE_INPUT_DELTA_ACTION_RELATIONSHIPS = False

PLOT_AVERAGE_ACTION_EVENT_FREQUENCY = False
EVENT_FREQUENCY_WINDOW_SAMPLES = 15
EVENT_FREQUENCY_SAMPLE_PERIOD_SECONDS: float | None = None
EVENT_FREQUENCY_TOLERANCE = 0.0
EVENT_FREQUENCY_CONFIDENCE_Z = 1.96

PLOT_FIRST_ATM_APPLIED_WINDOW = False
PLOT_FIRST_ATM_BASELINE_ZEROED_WINDOW = False
PLOT_FIRST_ATM_WINDOWED_DELTA = False
PLOT_FIRST_ATM_SECOND_DERIVATIVE_WINDOW = False
PLOT_FIRST_ATM_ON_OFF_BOXPLOTS = False
PLOT_RESISTANCE_THRESHOLD_TIMES = False
PLOT_RESISTANCE_THRESHOLD_BOXPLOTS = False
PLOT_RESISTANCE_THRESHOLD_TIME_SCATTER = True
PLOT_RESISTANCE_THRESHOLD_INTERVAL_SCATTER = True
PLOT_RESISTANCE_THRESHOLD_ACTION_RATE_SCATTER = True
PLOT_RESISTANCE_THRESHOLD_PEAK_RATE_BOXPLOTS = True
PLOT_RESISTANCE_THRESHOLD_PEAK_RATE_SCATTER = True
PLOT_RESISTANCE_THRESHOLD_WINDOW_SAMPLE_ACTION_BOXPLOTS = True
PLOT_RESISTANCE_DERIVATIVE_TO_FIRST_THRESHOLD = False
PLOT_RESISTANCE_SECOND_DERIVATIVE_TO_FIRST_THRESHOLD = False
PLOT_RESISTANCE_DERIVATIVE_NORMALIZED_TO_FIRST_THRESHOLD = False
ATM_STATE_LABEL = "pressure_atm_state"
ATM_WINDOW_VALUE_PATH = RESISTANCE_VALUE_PATH
ATM_WINDOW_VALUE_CHANNEL = 0
ATM_WINDOW_SAMPLES = 20
ATM_WINDOW_VALUE_YLIM: tuple[float, float] | None = (0.0, 30.0)
ATM_BASELINE_SAMPLES = 15
ATM_BASELINE_ZEROED_VALUE_YLIM: tuple[float, float] | None = (-1.0, 6.0)
ATM_WINDOWED_DELTA_YLIM: tuple[float, float] | None = (-0.5, 1.0)
ATM_SECOND_DERIVATIVE_YLIM: tuple[float, float] | None = (-0.5, 0.5)
ATM_WINDOW_SEARCH_SAMPLES: int | None = 600
ATM_ON_OFF_DURATION_YLIM: tuple[float, float] | None = None
ATM_ON_OFF_RESISTANCE_DELTA_YLIM: tuple[float, float] | None = None

RESISTANCE_THRESHOLD_VALUES = (1.0, 2.0, 5.0, 10.0, 50.0, 80.0, 100.0, 300.0, 500.0)
RESISTANCE_THRESHOLD_GROUPS = (
    (1.0, 2.0, 3.0, 4.0, 5.0),
    (25.0, 50.0, 75.0),
    (100.0, 200.0, 500.0, 1000.0),
)
RESISTANCE_THRESHOLD_BASELINE_SAMPLES = 15
RESISTANCE_THRESHOLD_SEARCH_SAMPLES: int | None = None
RESISTANCE_THRESHOLD_TIME_YLIM: tuple[float, float] | None = None
RESISTANCE_THRESHOLD_SCATTER_X_THRESHOLD = 10.0
RESISTANCE_THRESHOLD_SCATTER_Y_THRESHOLD = 50.0
RESISTANCE_THRESHOLD_ACTION_RATE_THRESHOLD = 50.0
RESISTANCE_THRESHOLD_ACTION_RATE_TOLERANCE = EVENT_FREQUENCY_TOLERANCE
RESISTANCE_THRESHOLD_PEAK_RATE_YLIM: tuple[float, float] | None = None
RESISTANCE_THRESHOLD_WINDOW_BOXPLOT_WINDOWS = (
    (0.0, 10.0),
    (10.0, 50.0),
    (50.0, 150.0),
    (150.0, 450.0),
    (450.0, 1000.0),
)
RESISTANCE_THRESHOLD_WINDOW_ACTION_TOLERANCE = EVENT_FREQUENCY_TOLERANCE
RESISTANCE_THRESHOLD_WINDOW_SAMPLE_YLIM: tuple[float, float] | None = None
RESISTANCE_THRESHOLD_WINDOW_ACTION_YLIM: tuple[float, float] | None = None

RESISTANCE_DERIVATIVE_THRESHOLD = 50
RESISTANCE_DERIVATIVE_BASELINE_SAMPLES = 15
RESISTANCE_DERIVATIVE_SEARCH_SAMPLES: int | None = None
RESISTANCE_DERIVATIVE_TRUNCATE_AT_FIRST_THRESHOLD = True
RESISTANCE_DERIVATIVE_YLIM: tuple[float, float] | None = (-0.5, 0.5)
RESISTANCE_SECOND_DERIVATIVE_YLIM: tuple[float, float] | None = (-0.5, 0.5)
RESISTANCE_DERIVATIVE_NORMALIZED_POINTS = 101
RESISTANCE_DERIVATIVE_NORMALIZED_CONFIDENCE_Z = 1.96
RESISTANCE_DERIVATIVE_NORMALIZED_YLIM: tuple[float, float] | None = (-0.5, 0.5)


@dataclass(frozen=True)
class AtmOnOffSegment:
    demo: str
    x: np.ndarray
    values: np.ndarray
    atm_state: np.ndarray
    on_index: int
    applied_pressure_index: int
    applied_pressure_x: float
    applied_pressure_value: float


@dataclass(frozen=True)
class AtmAppliedWindowSegment:
    demo: str
    x: np.ndarray
    values: np.ndarray
    atm_state: np.ndarray
    on_index: int
    baseline_value: float | None = None


@dataclass(frozen=True)
class ThresholdTimeSummary:
    thresholds: np.ndarray
    threshold_times: tuple[np.ndarray, ...]
    mean_times: np.ndarray
    sem_times: np.ndarray
    reached_counts: np.ndarray
    eligible_demo_count: int


@dataclass(frozen=True)
class ThresholdTimePairSummary:
    x_threshold: float
    y_threshold: float
    x_times: np.ndarray
    y_times: np.ndarray
    demo_names: tuple[str, ...]
    eligible_demo_count: int
    x_reached_count: int
    y_reached_count: int


@dataclass(frozen=True)
class ThresholdActionRateSummary:
    threshold: float
    sample_times: np.ndarray
    action_rates: np.ndarray
    demo_names: tuple[str, ...]
    eligible_demo_count: int
    reached_count: int


@dataclass(frozen=True)
class ThresholdPeakRateSummary:
    x_threshold: float
    y_threshold: float
    first_peak_rates: np.ndarray
    interval_peak_rates: np.ndarray
    demo_names: tuple[str, ...]
    eligible_demo_count: int
    x_reached_count: int
    y_reached_count: int


@dataclass(frozen=True)
class ThresholdWindowSampleActionSummary:
    windows: tuple[tuple[float, float], ...]
    sample_counts: tuple[np.ndarray, ...]
    action_counts: tuple[np.ndarray, ...]
    eligible_demo_count: int
    reached_counts: np.ndarray


@dataclass(frozen=True)
class ScatterRelationshipStats:
    point_count: int
    pearson_r: float
    spearman_rho: float
    slope: float
    intercept: float


@dataclass(frozen=True)
class AtmOnOffBoxplotSummary:
    duration_samples: np.ndarray
    resistance_deltas: np.ndarray
    used_demo_count: int


def _finite_window_values(row: np.ndarray, *, zero_padding_is_missing: bool) -> np.ndarray:
    values = np.asarray(row, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if not zero_padding_is_missing or values.size == 0:
        return values

    nonzero = np.flatnonzero(values != 0.0)
    if nonzero.size == 0:
        return np.asarray([], dtype=float)
    return values[int(nonzero[0]) :]


def _resistance_input_features(
    resistance_input: np.ndarray,
    *,
    zero_padding_is_missing: bool,
) -> dict[str, np.ndarray]:
    windows = dp._as_2d(np.asarray(resistance_input, dtype=float))
    features = {
        "delta": np.zeros(windows.shape[0], dtype=float),
        "sd": np.zeros(windows.shape[0], dtype=float),
    }

    for row_idx, row in enumerate(windows):
        values = _finite_window_values(
            row,
            zero_padding_is_missing=zero_padding_is_missing,
        )
        if values.size >= 2:
            features["delta"][row_idx] = float(values[-1] - values[-2])
            features["sd"][row_idx] = float(np.std(values))

    return features


def _action_event_mask(
    actions: np.ndarray,
    action_labels: Sequence[str],
    *,
    tolerance: float,
) -> np.ndarray:
    actions = dp._as_2d(np.asarray(actions, dtype=float))
    num_rows = actions.shape[0]
    if num_rows == 0 or actions.shape[1] == 0:
        return np.zeros(num_rows, dtype=bool)

    labels = list(action_labels)
    if len(labels) != actions.shape[1]:
        return np.any(np.abs(actions) > tolerance, axis=1)

    action_events = np.zeros(num_rows, dtype=bool)
    for idx, label in enumerate(labels):
        values = np.asarray(actions[:, idx], dtype=float)
        if label in {"commanded_pressure_mbar", "pressure_atm_state", "applied_pressure_mbar"}:
            changes = np.zeros(num_rows, dtype=bool)
            if num_rows > 1:
                changes[1:] = np.abs(np.diff(values)) > tolerance
            action_events |= changes
        else:
            action_events |= np.abs(values) > tolerance
    return action_events


def _rolling_event_rate(
    action_events: np.ndarray,
    *,
    window_samples: int,
    sample_period_seconds: float | None,
) -> np.ndarray:
    events = np.asarray(action_events, dtype=float).reshape(-1)
    if events.size == 0:
        return events

    window = max(1, int(window_samples))
    cumulative = np.concatenate([[0.0], np.cumsum(events)])
    end_indices = np.arange(1, events.size + 1)
    start_indices = np.maximum(0, end_indices - window)
    event_counts = cumulative[end_indices] - cumulative[start_indices]
    sample_counts = end_indices - start_indices

    if sample_period_seconds is None:
        denominators = sample_counts.astype(float)
    else:
        sample_period_seconds = float(sample_period_seconds)
        if sample_period_seconds <= 0.0:
            raise ValueError("sample_period_seconds must be positive or None")
        denominators = sample_counts.astype(float) * sample_period_seconds
    return event_counts / denominators


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return values

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=float)

    start = 0
    while start < sorted_values.size:
        end = start + 1
        while end < sorted_values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * float(start + end - 1) + 1.0
        ranks[order[start:end]] = average_rank
        start = end

    return ranks


def _pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 2:
        return float("nan")

    x = x[mask]
    y = y[mask]
    if np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _theil_sen_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size < 2 or y.size != x.size:
        return float("nan"), float("nan")

    slopes: list[float] = []
    for left_idx in range(x.size - 1):
        dx = x[left_idx + 1 :] - x[left_idx]
        dy = y[left_idx + 1 :] - y[left_idx]
        valid = dx != 0.0
        if np.any(valid):
            slopes.extend((dy[valid] / dx[valid]).tolist())

    if not slopes:
        return float("nan"), float("nan")

    slope = float(np.median(np.asarray(slopes, dtype=float)))
    intercept = float(np.median(y - slope * x))
    return slope, intercept


def _scatter_relationship_stats(x: np.ndarray, y: np.ndarray) -> ScatterRelationshipStats:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    point_count = int(np.sum(mask))
    if point_count < 2:
        return ScatterRelationshipStats(
            point_count,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        )

    x = x[mask]
    y = y[mask]
    pearson_r = _pearson_correlation(x, y)
    spearman_rho = _pearson_correlation(_average_ranks(x), _average_ranks(y))

    slope = float("nan")
    intercept = float("nan")
    if not np.all(x == x[0]):
        slope, intercept = _theil_sen_fit(x, y)

    return ScatterRelationshipStats(point_count, pearson_r, spearman_rho, slope, intercept)


def _format_fit_value(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.3g}"


def _format_scatter_relationship_stats(stats: ScatterRelationshipStats) -> str:
    lines: list[str] = []
    if math.isfinite(stats.pearson_r):
        lines.append(f"Pearson r={stats.pearson_r:.2f}")
    if math.isfinite(stats.spearman_rho):
        lines.append(f"Spearman rho={stats.spearman_rho:.2f}")
    if math.isfinite(stats.slope) and math.isfinite(stats.intercept):
        sign = "+" if stats.intercept >= 0.0 else "-"
        lines.append(
            "robust fit: "
            f"y={_format_fit_value(stats.slope)}x {sign} {_format_fit_value(abs(stats.intercept))}"
        )
    return "\n".join(lines)


def _plot_linear_fit_line(
    ax: plt.Axes,
    stats: ScatterRelationshipStats,
    *,
    x_min: float,
    x_max: float,
    label: str = "robust fit",
) -> None:
    if not (math.isfinite(stats.slope) and math.isfinite(stats.intercept)):
        return
    if not (math.isfinite(x_min) and math.isfinite(x_max)) or x_max <= x_min:
        return

    fit_x = np.asarray([x_min, x_max], dtype=float)
    fit_y = stats.slope * fit_x + stats.intercept
    ax.plot(fit_x, fit_y, color="tab:orange", linewidth=1.6, label=label)


def compute_average_action_event_frequency_with_confidence(
    hdf: h5py.File,
    demos: Sequence[str],
    *,
    action_path: str,
    max_samples: int | None,
    window_samples: int,
    sample_period_seconds: float | None,
    tolerance: float,
    confidence_z: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    confidence_z = float(confidence_z)
    if confidence_z <= 0.0:
        raise ValueError("confidence_z must be positive")

    demo_rates: list[np.ndarray] = []
    for demo in demos:
        demo_group = hdf["data"][demo]
        if action_path not in demo_group:
            continue
        action_ds = demo_group[action_path]
        if not isinstance(action_ds, h5py.Dataset):
            continue

        actions = dp._load_2d(action_ds, max_samples)
        action_events = _action_event_mask(
            actions,
            dp._channel_labels(action_path, action_ds),
            tolerance=tolerance,
        )
        if action_events.size == 0:
            continue
        demo_rates.append(
            _rolling_event_rate(
                action_events,
                window_samples=window_samples,
                sample_period_seconds=sample_period_seconds,
            )
        )

    used_demo_count = len(demo_rates)
    if used_demo_count == 0:
        empty_float = np.asarray([], dtype=float)
        empty_count = np.asarray([], dtype=int)
        return empty_float, empty_float, empty_float, empty_float, empty_count, 0

    max_length = max(rate.shape[0] for rate in demo_rates)
    padded = np.full((used_demo_count, max_length), np.nan, dtype=float)
    for demo_idx, rate in enumerate(demo_rates):
        padded[demo_idx, : rate.shape[0]] = rate

    active_demo_counts = np.sum(np.isfinite(padded), axis=0)
    summed_rates = np.nansum(padded, axis=0)
    mean_rate = np.divide(
        summed_rates,
        active_demo_counts,
        out=np.full(max_length, np.nan, dtype=float),
        where=active_demo_counts > 0,
    )

    centered = padded - mean_rate
    squared_error = np.where(np.isfinite(centered), centered**2, 0.0)
    sample_variance = np.divide(
        np.sum(squared_error, axis=0),
        active_demo_counts - 1,
        out=np.full(max_length, np.nan, dtype=float),
        where=active_demo_counts > 1,
    )
    standard_error = np.divide(
        np.sqrt(sample_variance),
        np.sqrt(active_demo_counts),
        out=np.full(max_length, np.nan, dtype=float),
        where=active_demo_counts > 1,
    )
    confidence_margin = confidence_z * standard_error

    max_rate = 1.0
    if sample_period_seconds is not None:
        max_rate /= float(sample_period_seconds)
    confidence_low = np.clip(mean_rate - confidence_margin, 0.0, max_rate)
    confidence_high = np.clip(mean_rate + confidence_margin, 0.0, max_rate)
    confidence_low[active_demo_counts <= 1] = np.nan
    confidence_high[active_demo_counts <= 1] = np.nan

    time_axis = np.arange(max_length, dtype=float)
    if sample_period_seconds is not None:
        time_axis *= float(sample_period_seconds)
    return (
        time_axis,
        mean_rate,
        confidence_low,
        confidence_high,
        active_demo_counts.astype(int),
        used_demo_count,
    )


def compute_average_action_event_frequency(
    hdf: h5py.File,
    demos: Sequence[str],
    *,
    action_path: str,
    max_samples: int | None,
    window_samples: int,
    sample_period_seconds: float | None,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    time_axis, mean_rate, _low, _high, active_counts, used_count = (
        compute_average_action_event_frequency_with_confidence(
            hdf,
            demos,
            action_path=action_path,
            max_samples=max_samples,
            window_samples=window_samples,
            sample_period_seconds=sample_period_seconds,
            tolerance=tolerance,
            confidence_z=EVENT_FREQUENCY_CONFIDENCE_Z,
        )
    )
    return time_axis, mean_rate, active_counts, used_count


def _is_binary_action(values: np.ndarray) -> bool:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return False
    unique = np.unique(finite)
    return unique.size <= 2 and np.all(np.isin(unique, [0.0, 1.0]))


def _action_delta(actions: np.ndarray) -> np.ndarray:
    actions = dp._as_2d(np.asarray(actions, dtype=float))
    delta = np.zeros_like(actions, dtype=float)
    if actions.shape[0] > 1:
        delta[1:] = np.diff(actions, axis=0)
    return delta


def _continuous_action_indices(
    hdf: h5py.File,
    demos: Sequence[str],
    action_path: str,
    action_width: int,
    action_labels: Sequence[str],
    max_samples: int | None,
) -> list[int]:
    values_by_dim: list[list[np.ndarray]] = [[] for _ in range(action_width)]
    for demo in demos:
        demo_group = hdf["data"][demo]
        if action_path not in demo_group:
            continue
        actions = dp._load_2d(demo_group[action_path], max_samples)
        for action_idx in range(min(action_width, actions.shape[1])):
            values_by_dim[action_idx].append(actions[:, action_idx])

    continuous_indices: list[int] = []
    for action_idx in range(action_width):
        label = action_labels[action_idx] if action_idx < len(action_labels) else f"ch{action_idx}"
        if "atm" in label.lower() or "switch" in label.lower():
            continue
        if not values_by_dim[action_idx]:
            continue
        all_values = np.concatenate(values_by_dim[action_idx])
        if not _is_binary_action(all_values):
            continuous_indices.append(action_idx)
    return continuous_indices


def plot_resistance_input_action_relationships(
    hdf: h5py.File,
    demos: Sequence[str],
    *,
    resistance_input_path: str,
    action_path: str,
    max_samples: int | None,
    output_dir: Path | None,
    save_plots: bool,
    plot_format: str,
    plot_dpi: int,
    zero_padding_is_missing: bool,
    point_size: float,
    alpha: float,
    use_action_delta: bool = False,
) -> None:
    action_ds = dp._first_dataset_for_path(hdf, demos, action_path)
    resistance_ds = dp._first_dataset_for_path(hdf, demos, resistance_input_path)
    if action_ds is None or resistance_ds is None:
        print(
            "Skipping resistance_input action relationship plots: "
            f"missing {action_path!r} or {resistance_input_path!r}."
        )
        return

    action_width = dp._dataset_width(action_ds)
    action_labels = dp._channel_labels(action_path, action_ds)
    if len(action_labels) < action_width:
        action_labels.extend(f"ch{idx}" for idx in range(len(action_labels), action_width))

    selected_action_indices = _continuous_action_indices(
        hdf,
        demos,
        action_path,
        action_width,
        action_labels,
        max_samples,
    )
    if not selected_action_indices:
        print(
            "Skipping resistance_input action relationship plots: "
            "no continuous action dimensions were found."
        )
        return

    feature_specs = (
        ("delta", "Resistance input delta"),
        ("sd", "Resistance input SD"),
    )

    fig, axes = plt.subplots(
        len(selected_action_indices),
        len(feature_specs),
        figsize=(5.2 * len(feature_specs), 3.4 * len(selected_action_indices)),
        squeeze=False,
    )
    plotted_any = False

    for demo_idx, demo in enumerate(demos):
        demo_group = hdf["data"][demo]
        if action_path not in demo_group or resistance_input_path not in demo_group:
            continue

        actions = dp._load_2d(demo_group[action_path], max_samples)
        resistance_input = dp._load_2d(demo_group[resistance_input_path], max_samples)
        num_rows = min(actions.shape[0], resistance_input.shape[0])
        if num_rows == 0:
            continue

        actions = actions[:num_rows]
        plotted_actions = _action_delta(actions) if use_action_delta else actions
        features = _resistance_input_features(
            resistance_input[:num_rows],
            zero_padding_is_missing=zero_padding_is_missing,
        )
        color = dp._plot_color(demo_idx, len(demos))

        for row_idx, action_idx in enumerate(selected_action_indices):
            if action_idx >= plotted_actions.shape[1]:
                continue
            action_values = plotted_actions[:, action_idx]
            for feature_idx, (feature_key, feature_label) in enumerate(feature_specs):
                feature_values = features[feature_key]
                mask = np.isfinite(feature_values) & np.isfinite(action_values)
                if not np.any(mask):
                    continue

                ax = axes[row_idx][feature_idx]
                ax.scatter(
                    feature_values[mask],
                    action_values[mask],
                    s=point_size,
                    color=color,
                    alpha=alpha,
                    edgecolors="none",
                    label=demo if len(demos) <= dp.LEGEND_MAX_DEMOS else None,
                )
                plotted_any = True

    if not plotted_any:
        print(
            "Skipping resistance_input action relationship plots: "
            "no aligned finite samples were found."
        )
        plt.close(fig)
        return

    for row_idx, action_idx in enumerate(selected_action_indices):
        action_label = action_labels[action_idx]
        action_axis_label = f"delta {action_label}" if use_action_delta else action_label
        for feature_idx, (_, feature_label) in enumerate(feature_specs):
            ax = axes[row_idx][feature_idx]
            ax.set_title(f"{action_axis_label} vs {feature_label}")
            ax.set_xlabel(feature_label)
            ax.set_ylabel(f"{action_axis_label} action value")
            ax.grid(True, alpha=0.25)
            if len(demos) <= dp.LEGEND_MAX_DEMOS and plotted_any:
                ax.legend(loc="best", fontsize=7)

    action_title = f"delta {dp._pretty_path(action_path)}" if use_action_delta else dp._pretty_path(action_path)
    fig.suptitle(
        f"{action_title} vs {dp._pretty_path(resistance_input_path)} features",
        fontsize=14,
    )
    fig.tight_layout()

    if save_plots and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        action_slug = f"delta_{dp._slug(action_path)}" if use_action_delta else dp._slug(action_path)
        fig.savefig(
            output_dir / f"{dp._slug(resistance_input_path)}_vs_{action_slug}.{plot_format}",
            dpi=plot_dpi,
            bbox_inches="tight",
        )


def plot_average_action_event_frequency(
    hdf: h5py.File,
    demos: Sequence[str],
    *,
    action_path: str,
    max_samples: int | None,
    output_dir: Path | None,
    save_plots: bool,
    plot_format: str,
    plot_dpi: int,
    window_samples: int,
    sample_period_seconds: float | None,
    tolerance: float,
    confidence_z: float,
) -> None:
    (
        time_axis,
        mean_rate,
        confidence_low,
        confidence_high,
        _active_demo_counts,
        used_demo_count,
    ) = (
        compute_average_action_event_frequency_with_confidence(
            hdf,
            demos,
            action_path=action_path,
            max_samples=max_samples,
            window_samples=window_samples,
            sample_period_seconds=sample_period_seconds,
            tolerance=tolerance,
            confidence_z=confidence_z,
        )
    )
    if used_demo_count == 0:
        print(
            "Skipping average action event frequency plot: "
            f"no demos contained {action_path!r}."
        )
        return

    fig, ax = plt.subplots(figsize=(9, 4.8))
    confidence_mask = np.isfinite(confidence_low) & np.isfinite(confidence_high)
    if np.any(confidence_mask):
        confidence_label = (
            "Approx. 95% CI"
            if math.isclose(float(confidence_z), 1.96, rel_tol=0.0, abs_tol=0.01)
            else f"+/- {float(confidence_z):g} SE"
        )
        ax.fill_between(
            time_axis,
            confidence_low,
            confidence_high,
            where=confidence_mask,
            color="tab:blue",
            alpha=0.18,
            linewidth=0.0,
            label=confidence_label,
        )
        ax.plot(
            time_axis,
            confidence_low,
            color="tab:blue",
            alpha=0.45,
            linestyle="--",
            linewidth=1.0,
        )
        ax.plot(
            time_axis,
            confidence_high,
            color="tab:blue",
            alpha=0.45,
            linestyle="--",
            linewidth=1.0,
        )
    ax.plot(time_axis, mean_rate, color="tab:blue", linewidth=2.0, label="Mean")

    if sample_period_seconds is None:
        ax.set_xlabel("Sample")
        ax.set_ylabel("Events / sample")
    else:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Events / second")

    ax.set_title(
        "Average action event frequency "
        f"({used_demo_count} demos, trailing {max(1, int(window_samples))} samples)"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    if save_plots and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_dir / f"average_action_event_frequency.{plot_format}",
            dpi=plot_dpi,
            bbox_inches="tight",
        )


def find_first_atm_on_to_applied_pressure_indices(
    atm_state: np.ndarray,
    *,
    search_samples: int | None,
) -> tuple[int | None, int | None]:
    values = np.asarray(atm_state, dtype=float).reshape(-1)
    if values.size == 0:
        return None, None

    if search_samples is not None:
        limit = min(values.size, max(0, int(search_samples)))
        values = values[:limit]
    if values.size == 0:
        return None, None

    on_state = values >= 0.5
    if on_state[0]:
        first_on = 0
    else:
        on_transitions = np.flatnonzero((~on_state[:-1]) & on_state[1:]) + 1
        if on_transitions.size == 0:
            return None, None
        first_on = int(on_transitions[0])

    applied_pressure_transitions = np.flatnonzero(on_state[:-1] & (~on_state[1:])) + 1
    applied_pressure_after_on = applied_pressure_transitions[applied_pressure_transitions > first_on]
    if applied_pressure_after_on.size == 0:
        return first_on, None
    return first_on, int(applied_pressure_after_on[0])


def find_first_atm_on_index(
    atm_state: np.ndarray,
    *,
    search_samples: int | None,
) -> int | None:
    first_on, _applied_pressure_idx = find_first_atm_on_to_applied_pressure_indices(
        atm_state,
        search_samples=search_samples,
    )
    return first_on


def compute_first_atm_on_off_boxplot_summary(
    hdf: h5py.File,
    demos: Sequence[str],
    *,
    action_path: str,
    atm_label: str,
    resistance_path: str,
    resistance_channel: int,
    max_samples: int | None,
    search_samples: int | None,
) -> AtmOnOffBoxplotSummary:
    durations: list[float] = []
    deltas: list[float] = []

    for demo in demos:
        demo_group = hdf["data"][demo]
        if action_path not in demo_group or resistance_path not in demo_group:
            continue

        action_ds = demo_group[action_path]
        resistance_ds = demo_group[resistance_path]
        if not isinstance(action_ds, h5py.Dataset) or not isinstance(resistance_ds, h5py.Dataset):
            continue

        atm_idx = _axis_index(action_ds, action_path, atm_label)
        if atm_idx is None:
            continue

        actions = dp._load_2d(action_ds, max_samples)
        resistance_2d = dp._load_2d(resistance_ds, max_samples)
        if resistance_channel < 0 or resistance_channel >= resistance_2d.shape[1]:
            continue

        num_rows = min(actions.shape[0], resistance_2d.shape[0])
        if num_rows == 0 or atm_idx >= actions.shape[1]:
            continue

        on_idx, off_idx = find_first_atm_on_to_applied_pressure_indices(
            actions[:num_rows, atm_idx],
            search_samples=search_samples,
        )
        if on_idx is None or off_idx is None or off_idx >= num_rows:
            continue

        on_resistance = float(resistance_2d[on_idx, resistance_channel])
        off_resistance = float(resistance_2d[off_idx, resistance_channel])
        if not (math.isfinite(on_resistance) and math.isfinite(off_resistance)):
            continue

        durations.append(float(off_idx - on_idx))
        deltas.append(off_resistance - on_resistance)

    return AtmOnOffBoxplotSummary(
        duration_samples=np.asarray(durations, dtype=float),
        resistance_deltas=np.asarray(deltas, dtype=float),
        used_demo_count=len(durations),
    )


def _axis_index(ds: h5py.Dataset, path: str, label: str) -> int | None:
    labels = dp._channel_labels(path, ds)
    if label in labels:
        return labels.index(label)

    label_lower = label.lower()
    for idx, candidate in enumerate(labels):
        if candidate.lower() == label_lower:
            return idx
    return None


def collect_first_atm_on_to_applied_pressure_segments(
    hdf: h5py.File,
    demos: Sequence[str],
    *,
    action_path: str,
    atm_label: str,
    value_path: str,
    value_channel: int,
    max_samples: int | None,
    pre_samples: int,
    post_samples: int,
    search_samples: int | None,
) -> list[AtmOnOffSegment]:
    segments: list[AtmOnOffSegment] = []
    for demo in demos:
        demo_group = hdf["data"][demo]
        if action_path not in demo_group or value_path not in demo_group:
            continue

        action_ds = demo_group[action_path]
        value_ds = demo_group[value_path]
        if not isinstance(action_ds, h5py.Dataset) or not isinstance(value_ds, h5py.Dataset):
            continue

        atm_idx = _axis_index(action_ds, action_path, atm_label)
        if atm_idx is None:
            continue

        actions = dp._load_2d(action_ds, max_samples)
        values_2d = dp._load_2d(value_ds, max_samples)
        if value_channel < 0 or value_channel >= values_2d.shape[1]:
            continue

        num_rows = min(actions.shape[0], values_2d.shape[0])
        if num_rows == 0 or atm_idx >= actions.shape[1]:
            continue

        atm_state = actions[:num_rows, atm_idx]
        on_idx, applied_pressure_idx = find_first_atm_on_to_applied_pressure_indices(
            atm_state,
            search_samples=search_samples,
        )
        if on_idx is None or applied_pressure_idx is None:
            continue

        pre = max(0, int(pre_samples))
        post = max(0, int(post_samples))
        start = max(0, on_idx - pre)
        end = min(num_rows, applied_pressure_idx + post + 1)
        x = np.arange(start, end, dtype=float) - float(on_idx)
        values = values_2d[start:end, value_channel]
        segment_atm = (atm_state[start:end] >= 0.5).astype(float)

        segments.append(
            AtmOnOffSegment(
                demo=demo,
                x=x,
                values=values,
                atm_state=segment_atm,
                on_index=on_idx,
                applied_pressure_index=applied_pressure_idx,
                applied_pressure_x=float(applied_pressure_idx - on_idx),
                applied_pressure_value=float(values_2d[applied_pressure_idx, value_channel]),
            )
        )

    return segments


def collect_first_atm_applied_window_segments(
    hdf: h5py.File,
    demos: Sequence[str],
    *,
    action_path: str,
    atm_label: str,
    value_path: str,
    value_channel: int,
    max_samples: int | None,
    window_samples: int,
    search_samples: int | None,
) -> list[AtmAppliedWindowSegment]:
    segments: list[AtmAppliedWindowSegment] = []
    window = max(1, int(window_samples))

    for demo in demos:
        demo_group = hdf["data"][demo]
        if action_path not in demo_group or value_path not in demo_group:
            continue

        action_ds = demo_group[action_path]
        value_ds = demo_group[value_path]
        if not isinstance(action_ds, h5py.Dataset) or not isinstance(value_ds, h5py.Dataset):
            continue

        atm_idx = _axis_index(action_ds, action_path, atm_label)
        if atm_idx is None:
            continue

        actions = dp._load_2d(action_ds, max_samples)
        values_2d = dp._load_2d(value_ds, max_samples)
        if value_channel < 0 or value_channel >= values_2d.shape[1]:
            continue

        num_rows = min(actions.shape[0], values_2d.shape[0])
        if num_rows == 0 or atm_idx >= actions.shape[1]:
            continue

        atm_state = actions[:num_rows, atm_idx]
        on_idx = find_first_atm_on_index(
            atm_state,
            search_samples=search_samples,
        )
        if on_idx is None:
            continue

        end = min(num_rows, on_idx + window)
        if end <= on_idx:
            continue

        x = np.arange(end - on_idx, dtype=float)
        values = values_2d[on_idx:end, value_channel]
        segment_atm = (atm_state[on_idx:end] >= 0.5).astype(float)

        segments.append(
            AtmAppliedWindowSegment(
                demo=demo,
                x=x,
                values=values,
                atm_state=segment_atm,
                on_index=on_idx,
            )
        )

    return segments


def collect_first_atm_baseline_zeroed_window_segments(
    hdf: h5py.File,
    demos: Sequence[str],
    *,
    action_path: str,
    atm_label: str,
    value_path: str,
    value_channel: int,
    max_samples: int | None,
    window_samples: int,
    baseline_samples: int,
    search_samples: int | None,
) -> list[AtmAppliedWindowSegment]:
    segments: list[AtmAppliedWindowSegment] = []
    window = max(1, int(window_samples))
    baseline_window = max(1, int(baseline_samples))

    for demo in demos:
        demo_group = hdf["data"][demo]
        if action_path not in demo_group or value_path not in demo_group:
            continue

        action_ds = demo_group[action_path]
        value_ds = demo_group[value_path]
        if not isinstance(action_ds, h5py.Dataset) or not isinstance(value_ds, h5py.Dataset):
            continue

        atm_idx = _axis_index(action_ds, action_path, atm_label)
        if atm_idx is None:
            continue

        actions = dp._load_2d(action_ds, max_samples)
        values_2d = dp._load_2d(value_ds, max_samples)
        if value_channel < 0 or value_channel >= values_2d.shape[1]:
            continue

        num_rows = min(actions.shape[0], values_2d.shape[0])
        if num_rows == 0 or atm_idx >= actions.shape[1]:
            continue

        atm_state = actions[:num_rows, atm_idx]
        on_idx = find_first_atm_on_index(
            atm_state,
            search_samples=search_samples,
        )
        if on_idx is None or on_idx < baseline_window:
            continue

        baseline_values = values_2d[on_idx - baseline_window : on_idx, value_channel]
        baseline_values = baseline_values[np.isfinite(baseline_values)]
        if baseline_values.size != baseline_window:
            continue

        end = min(num_rows, on_idx + window)
        if end <= on_idx:
            continue

        baseline_value = float(np.mean(baseline_values))
        x = np.arange(end - on_idx, dtype=float)
        values = values_2d[on_idx:end, value_channel] - baseline_value
        segment_atm = (atm_state[on_idx:end] >= 0.5).astype(float)

        segments.append(
            AtmAppliedWindowSegment(
                demo=demo,
                x=x,
                values=values,
                atm_state=segment_atm,
                on_index=on_idx,
                baseline_value=baseline_value,
            )
        )

    return segments


def collect_first_atm_difference_window_segments(
    hdf: h5py.File,
    demos: Sequence[str],
    *,
    action_path: str,
    atm_label: str,
    value_path: str,
    value_channel: int,
    max_samples: int | None,
    window_samples: int,
    baseline_samples: int,
    search_samples: int | None,
    difference_order: int,
) -> list[AtmAppliedWindowSegment]:
    segments: list[AtmAppliedWindowSegment] = []
    window = max(1, int(window_samples))
    baseline_window = max(1, int(baseline_samples))
    order, _slug, _title, _units = _difference_order_details(difference_order)

    for demo in demos:
        demo_group = hdf["data"][demo]
        if action_path not in demo_group or value_path not in demo_group:
            continue

        action_ds = demo_group[action_path]
        value_ds = demo_group[value_path]
        if not isinstance(action_ds, h5py.Dataset) or not isinstance(value_ds, h5py.Dataset):
            continue

        atm_idx = _axis_index(action_ds, action_path, atm_label)
        if atm_idx is None:
            continue

        actions = dp._load_2d(action_ds, max_samples)
        values_2d = dp._load_2d(value_ds, max_samples)
        if value_channel < 0 or value_channel >= values_2d.shape[1]:
            continue

        num_rows = min(actions.shape[0], values_2d.shape[0])
        if num_rows == 0 or atm_idx >= actions.shape[1]:
            continue

        atm_state = actions[:num_rows, atm_idx]
        on_idx = find_first_atm_on_index(
            atm_state,
            search_samples=search_samples,
        )
        if on_idx is None or on_idx < baseline_window or on_idx < order:
            continue

        baseline_values = values_2d[on_idx - baseline_window : on_idx, value_channel]
        baseline_values = baseline_values[np.isfinite(baseline_values)]
        if baseline_values.size != baseline_window:
            continue

        end = min(num_rows, on_idx + window)
        if end <= on_idx:
            continue

        source_values = values_2d[on_idx - order : end, value_channel]
        if source_values.size < order + 1:
            continue

        difference = np.diff(source_values, n=order)
        if difference.size == 0:
            continue

        x = np.arange(difference.size, dtype=float)
        segment_atm = (atm_state[on_idx : on_idx + difference.size] >= 0.5).astype(float)

        segments.append(
            AtmAppliedWindowSegment(
                demo=demo,
                x=x,
                values=difference,
                atm_state=segment_atm,
                on_index=on_idx,
                baseline_value=float(np.mean(baseline_values)),
            )
        )

    return segments


def collect_first_atm_windowed_delta_segments(
    hdf: h5py.File,
    demos: Sequence[str],
    *,
    action_path: str,
    atm_label: str,
    resistance_input_path: str,
    max_samples: int | None,
    window_samples: int,
    zero_padding_is_missing: bool,
    search_samples: int | None,
) -> list[AtmAppliedWindowSegment]:
    segments: list[AtmAppliedWindowSegment] = []
    window = max(1, int(window_samples))

    for demo in demos:
        demo_group = hdf["data"][demo]
        if action_path not in demo_group or resistance_input_path not in demo_group:
            continue

        action_ds = demo_group[action_path]
        resistance_input_ds = demo_group[resistance_input_path]
        if not isinstance(action_ds, h5py.Dataset) or not isinstance(resistance_input_ds, h5py.Dataset):
            continue

        atm_idx = _axis_index(action_ds, action_path, atm_label)
        if atm_idx is None:
            continue

        actions = dp._load_2d(action_ds, max_samples)
        resistance_input = dp._load_2d(resistance_input_ds, max_samples)
        num_rows = min(actions.shape[0], resistance_input.shape[0])
        if num_rows == 0 or atm_idx >= actions.shape[1]:
            continue

        atm_state = actions[:num_rows, atm_idx]
        on_idx = find_first_atm_on_index(
            atm_state,
            search_samples=search_samples,
        )
        if on_idx is None:
            continue

        end = min(num_rows, on_idx + window)
        if end <= on_idx:
            continue

        features = _resistance_input_features(
            resistance_input[:num_rows],
            zero_padding_is_missing=zero_padding_is_missing,
        )
        x = np.arange(end - on_idx, dtype=float)
        values = features["delta"][on_idx:end]
        segment_atm = (atm_state[on_idx:end] >= 0.5).astype(float)

        segments.append(
            AtmAppliedWindowSegment(
                demo=demo,
                x=x,
                values=values,
                atm_state=segment_atm,
                on_index=on_idx,
            )
        )

    return segments


def compute_resistance_threshold_time_summary(
    hdf: h5py.File,
    demos: Sequence[str],
    *,
    action_path: str,
    atm_label: str,
    resistance_path: str,
    resistance_channel: int,
    thresholds: Sequence[float],
    max_samples: int | None,
    baseline_samples: int,
    threshold_search_samples: int | None,
    atm_search_samples: int | None,
) -> ThresholdTimeSummary:
    threshold_values = np.asarray(thresholds, dtype=float)
    if threshold_values.size == 0:
        raise ValueError("thresholds must contain at least one value")
    if np.any(threshold_values <= 0.0):
        raise ValueError("thresholds must be positive")

    baseline_window = max(1, int(baseline_samples))
    times_by_threshold: list[list[float]] = [[] for _ in threshold_values]
    eligible_demo_count = 0

    for demo in demos:
        demo_group = hdf["data"][demo]
        if action_path not in demo_group or resistance_path not in demo_group:
            continue

        action_ds = demo_group[action_path]
        resistance_ds = demo_group[resistance_path]
        if not isinstance(action_ds, h5py.Dataset) or not isinstance(resistance_ds, h5py.Dataset):
            continue

        atm_idx = _axis_index(action_ds, action_path, atm_label)
        if atm_idx is None:
            continue

        actions = dp._load_2d(action_ds, max_samples)
        resistance_2d = dp._load_2d(resistance_ds, max_samples)
        if resistance_channel < 0 or resistance_channel >= resistance_2d.shape[1]:
            continue

        num_rows = min(actions.shape[0], resistance_2d.shape[0])
        if num_rows == 0 or atm_idx >= actions.shape[1]:
            continue

        atm_state = actions[:num_rows, atm_idx]
        on_idx = find_first_atm_on_index(
            atm_state,
            search_samples=atm_search_samples,
        )
        if on_idx is None or on_idx < baseline_window:
            continue

        resistance = resistance_2d[:num_rows, resistance_channel]
        baseline_values = resistance[on_idx - baseline_window : on_idx]
        baseline_values = baseline_values[np.isfinite(baseline_values)]
        if baseline_values.size != baseline_window:
            continue

        if threshold_search_samples is None:
            end = num_rows
        else:
            end = min(num_rows, on_idx + max(1, int(threshold_search_samples)))
        if end <= on_idx:
            continue

        post_values = resistance[on_idx:end]
        post_delta = post_values - float(np.mean(baseline_values))
        eligible_demo_count += 1

        for threshold_idx, threshold in enumerate(threshold_values):
            crossing = np.flatnonzero(np.isfinite(post_delta) & (post_delta >= threshold))
            if crossing.size == 0:
                continue
            times_by_threshold[threshold_idx].append(float(crossing[0]))

    mean_times = np.full(threshold_values.shape, np.nan, dtype=float)
    sem_times = np.full(threshold_values.shape, np.nan, dtype=float)
    reached_counts = np.zeros(threshold_values.shape, dtype=int)

    for threshold_idx, times in enumerate(times_by_threshold):
        if not times:
            continue
        arr = np.asarray(times, dtype=float)
        reached_counts[threshold_idx] = arr.size
        mean_times[threshold_idx] = float(np.mean(arr))
        if arr.size > 1:
            sem_times[threshold_idx] = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
        else:
            sem_times[threshold_idx] = 0.0

    return ThresholdTimeSummary(
        thresholds=threshold_values,
        threshold_times=tuple(np.asarray(times, dtype=float) for times in times_by_threshold),
        mean_times=mean_times,
        sem_times=sem_times,
        reached_counts=reached_counts,
        eligible_demo_count=eligible_demo_count,
    )


def compute_resistance_threshold_time_pair_summary(
    hdf: h5py.File,
    demos: Sequence[str],
    *,
    action_path: str,
    atm_label: str,
    resistance_path: str,
    resistance_channel: int,
    x_threshold: float,
    y_threshold: float,
    max_samples: int | None,
    baseline_samples: int,
    threshold_search_samples: int | None,
    atm_search_samples: int | None,
) -> ThresholdTimePairSummary:
    x_threshold = float(x_threshold)
    y_threshold = float(y_threshold)
    if x_threshold <= 0.0 or y_threshold <= 0.0:
        raise ValueError("thresholds must be positive")

    baseline_window = max(1, int(baseline_samples))
    x_times: list[float] = []
    y_times: list[float] = []
    paired_demos: list[str] = []
    eligible_demo_count = 0
    x_reached_count = 0
    y_reached_count = 0

    for demo in demos:
        demo_group = hdf["data"][demo]
        if action_path not in demo_group or resistance_path not in demo_group:
            continue

        action_ds = demo_group[action_path]
        resistance_ds = demo_group[resistance_path]
        if not isinstance(action_ds, h5py.Dataset) or not isinstance(resistance_ds, h5py.Dataset):
            continue

        atm_idx = _axis_index(action_ds, action_path, atm_label)
        if atm_idx is None:
            continue

        actions = dp._load_2d(action_ds, max_samples)
        resistance_2d = dp._load_2d(resistance_ds, max_samples)
        if resistance_channel < 0 or resistance_channel >= resistance_2d.shape[1]:
            continue

        num_rows = min(actions.shape[0], resistance_2d.shape[0])
        if num_rows == 0 or atm_idx >= actions.shape[1]:
            continue

        atm_state = actions[:num_rows, atm_idx]
        on_idx = find_first_atm_on_index(
            atm_state,
            search_samples=atm_search_samples,
        )
        if on_idx is None or on_idx < baseline_window:
            continue

        resistance = resistance_2d[:num_rows, resistance_channel]
        baseline_values = resistance[on_idx - baseline_window : on_idx]
        baseline_values = baseline_values[np.isfinite(baseline_values)]
        if baseline_values.size != baseline_window:
            continue

        if threshold_search_samples is None:
            end = num_rows
        else:
            end = min(num_rows, on_idx + max(1, int(threshold_search_samples)))
        if end <= on_idx:
            continue

        post_values = resistance[on_idx:end]
        post_delta = post_values - float(np.mean(baseline_values))
        finite_mask = np.isfinite(post_delta)
        eligible_demo_count += 1

        x_crossing = np.flatnonzero(finite_mask & (post_delta >= x_threshold))
        y_crossing = np.flatnonzero(finite_mask & (post_delta >= y_threshold))
        if x_crossing.size > 0:
            x_reached_count += 1
        if y_crossing.size > 0:
            y_reached_count += 1
        if x_crossing.size == 0 or y_crossing.size == 0:
            continue

        paired_demos.append(demo)
        x_times.append(float(x_crossing[0]))
        y_times.append(float(y_crossing[0]))

    return ThresholdTimePairSummary(
        x_threshold=x_threshold,
        y_threshold=y_threshold,
        x_times=np.asarray(x_times, dtype=float),
        y_times=np.asarray(y_times, dtype=float),
        demo_names=tuple(paired_demos),
        eligible_demo_count=eligible_demo_count,
        x_reached_count=x_reached_count,
        y_reached_count=y_reached_count,
    )


def compute_resistance_threshold_action_rate_summary(
    hdf: h5py.File,
    demos: Sequence[str],
    *,
    action_path: str,
    atm_label: str,
    resistance_path: str,
    resistance_channel: int,
    threshold: float,
    max_samples: int | None,
    baseline_samples: int,
    threshold_search_samples: int | None,
    atm_search_samples: int | None,
    action_tolerance: float,
) -> ThresholdActionRateSummary:
    threshold = float(threshold)
    if threshold <= 0.0:
        raise ValueError("threshold must be positive")

    baseline_window = max(1, int(baseline_samples))
    sample_times: list[float] = []
    action_rates: list[float] = []
    reached_demos: list[str] = []
    eligible_demo_count = 0
    reached_count = 0

    for demo in demos:
        demo_group = hdf["data"][demo]
        if action_path not in demo_group or resistance_path not in demo_group:
            continue

        action_ds = demo_group[action_path]
        resistance_ds = demo_group[resistance_path]
        if not isinstance(action_ds, h5py.Dataset) or not isinstance(resistance_ds, h5py.Dataset):
            continue

        atm_idx = _axis_index(action_ds, action_path, atm_label)
        if atm_idx is None:
            continue

        actions = dp._load_2d(action_ds, max_samples)
        resistance_2d = dp._load_2d(resistance_ds, max_samples)
        if resistance_channel < 0 or resistance_channel >= resistance_2d.shape[1]:
            continue

        num_rows = min(actions.shape[0], resistance_2d.shape[0])
        if num_rows == 0 or atm_idx >= actions.shape[1]:
            continue

        actions = actions[:num_rows]
        atm_state = actions[:, atm_idx]
        on_idx = find_first_atm_on_index(
            atm_state,
            search_samples=atm_search_samples,
        )
        if on_idx is None or on_idx < baseline_window:
            continue

        resistance = resistance_2d[:num_rows, resistance_channel]
        baseline_values = resistance[on_idx - baseline_window : on_idx]
        baseline_values = baseline_values[np.isfinite(baseline_values)]
        if baseline_values.size != baseline_window:
            continue

        if threshold_search_samples is None:
            end = num_rows
        else:
            end = min(num_rows, on_idx + max(1, int(threshold_search_samples)))
        if end <= on_idx:
            continue

        eligible_demo_count += 1
        post_values = resistance[on_idx:end]
        post_delta = post_values - float(np.mean(baseline_values))
        crossing = np.flatnonzero(np.isfinite(post_delta) & (post_delta >= threshold))
        if crossing.size == 0:
            continue

        reached_count += 1
        crossing_offset = int(crossing[0])
        action_events = _action_event_mask(
            actions,
            dp._channel_labels(action_path, action_ds),
            tolerance=action_tolerance,
        )
        reached_demos.append(demo)
        sample_times.append(float(crossing_offset))
        sample_count = crossing_offset + 1
        action_count = float(np.sum(action_events[on_idx : on_idx + sample_count]))
        action_rates.append(action_count / float(sample_count))

    return ThresholdActionRateSummary(
        threshold=threshold,
        sample_times=np.asarray(sample_times, dtype=float),
        action_rates=np.asarray(action_rates, dtype=float),
        demo_names=tuple(reached_demos),
        eligible_demo_count=eligible_demo_count,
        reached_count=reached_count,
    )


def compute_resistance_threshold_window_sample_action_summary(
    hdf: h5py.File,
    demos: Sequence[str],
    *,
    action_path: str,
    atm_label: str,
    resistance_path: str,
    resistance_channel: int,
    windows: Sequence[tuple[float, float]],
    max_samples: int | None,
    baseline_samples: int,
    threshold_search_samples: int | None,
    atm_search_samples: int | None,
    action_tolerance: float,
) -> ThresholdWindowSampleActionSummary:
    window_values = tuple((float(lower), float(upper)) for lower, upper in windows)
    if not window_values:
        raise ValueError("windows must contain at least one interval")
    for lower, upper in window_values:
        if lower < 0.0 or upper <= lower:
            raise ValueError("window thresholds must satisfy 0 <= lower < upper")

    thresholds = sorted(
        {
            threshold
            for window in window_values
            for threshold in window
            if threshold > 0.0
        }
    )
    baseline_window = max(1, int(baseline_samples))
    sample_counts_by_window: list[list[float]] = [[] for _ in window_values]
    action_counts_by_window: list[list[float]] = [[] for _ in window_values]
    reached_counts = np.zeros(len(window_values), dtype=int)
    eligible_demo_count = 0

    for demo in demos:
        demo_group = hdf["data"][demo]
        if action_path not in demo_group or resistance_path not in demo_group:
            continue

        action_ds = demo_group[action_path]
        resistance_ds = demo_group[resistance_path]
        if not isinstance(action_ds, h5py.Dataset) or not isinstance(resistance_ds, h5py.Dataset):
            continue

        atm_idx = _axis_index(action_ds, action_path, atm_label)
        if atm_idx is None:
            continue

        actions = dp._load_2d(action_ds, max_samples)
        resistance_2d = dp._load_2d(resistance_ds, max_samples)
        if resistance_channel < 0 or resistance_channel >= resistance_2d.shape[1]:
            continue

        num_rows = min(actions.shape[0], resistance_2d.shape[0])
        if num_rows == 0 or atm_idx >= actions.shape[1]:
            continue

        actions = actions[:num_rows]
        atm_state = actions[:, atm_idx]
        on_idx = find_first_atm_on_index(
            atm_state,
            search_samples=atm_search_samples,
        )
        if on_idx is None or on_idx < baseline_window:
            continue

        resistance = resistance_2d[:num_rows, resistance_channel]
        baseline_values = resistance[on_idx - baseline_window : on_idx]
        baseline_values = baseline_values[np.isfinite(baseline_values)]
        if baseline_values.size != baseline_window:
            continue

        if threshold_search_samples is None:
            end = num_rows
        else:
            end = min(num_rows, on_idx + max(1, int(threshold_search_samples)))
        if end <= on_idx:
            continue

        baseline_value = float(np.mean(baseline_values))
        post_delta = resistance[on_idx:end] - baseline_value
        finite_mask = np.isfinite(post_delta)
        crossing_offsets: dict[float, int] = {0.0: 0}
        for threshold in thresholds:
            crossing = np.flatnonzero(finite_mask & (post_delta >= threshold))
            if crossing.size > 0:
                crossing_offsets[threshold] = int(crossing[0])

        eligible_demo_count += 1
        action_events = _action_event_mask(
            actions,
            dp._channel_labels(action_path, action_ds),
            tolerance=action_tolerance,
        )

        for window_idx, (lower, upper) in enumerate(window_values):
            if lower not in crossing_offsets or upper not in crossing_offsets:
                continue

            lower_offset = crossing_offsets[lower]
            upper_offset = crossing_offsets[upper]
            if upper_offset < lower_offset:
                continue

            lower_idx = on_idx + lower_offset
            upper_idx = on_idx + upper_offset
            action_start = lower_idx if lower == 0.0 else lower_idx + 1
            action_count = 0.0
            if action_start <= upper_idx:
                action_count = float(np.sum(action_events[action_start : upper_idx + 1]))

            reached_counts[window_idx] += 1
            sample_counts_by_window[window_idx].append(float(upper_offset - lower_offset))
            action_counts_by_window[window_idx].append(action_count)

    return ThresholdWindowSampleActionSummary(
        windows=window_values,
        sample_counts=tuple(np.asarray(values, dtype=float) for values in sample_counts_by_window),
        action_counts=tuple(np.asarray(values, dtype=float) for values in action_counts_by_window),
        eligible_demo_count=eligible_demo_count,
        reached_counts=reached_counts,
    )


def compute_resistance_threshold_peak_rate_summary(
    hdf: h5py.File,
    demos: Sequence[str],
    *,
    action_path: str,
    atm_label: str,
    resistance_path: str,
    resistance_channel: int,
    x_threshold: float,
    y_threshold: float,
    max_samples: int | None,
    baseline_samples: int,
    threshold_search_samples: int | None,
    atm_search_samples: int | None,
) -> ThresholdPeakRateSummary:
    x_threshold = float(x_threshold)
    y_threshold = float(y_threshold)
    if x_threshold <= 0.0 or y_threshold <= 0.0:
        raise ValueError("thresholds must be positive")
    if y_threshold <= x_threshold:
        raise ValueError("y_threshold must be greater than x_threshold")

    baseline_window = max(1, int(baseline_samples))
    first_peak_rates: list[float] = []
    interval_peak_rates: list[float] = []
    paired_demos: list[str] = []
    eligible_demo_count = 0
    x_reached_count = 0
    y_reached_count = 0

    for demo in demos:
        demo_group = hdf["data"][demo]
        if action_path not in demo_group or resistance_path not in demo_group:
            continue

        action_ds = demo_group[action_path]
        resistance_ds = demo_group[resistance_path]
        if not isinstance(action_ds, h5py.Dataset) or not isinstance(resistance_ds, h5py.Dataset):
            continue

        atm_idx = _axis_index(action_ds, action_path, atm_label)
        if atm_idx is None:
            continue

        actions = dp._load_2d(action_ds, max_samples)
        resistance_2d = dp._load_2d(resistance_ds, max_samples)
        if resistance_channel < 0 or resistance_channel >= resistance_2d.shape[1]:
            continue

        num_rows = min(actions.shape[0], resistance_2d.shape[0])
        if num_rows == 0 or atm_idx >= actions.shape[1]:
            continue

        atm_state = actions[:num_rows, atm_idx]
        on_idx = find_first_atm_on_index(
            atm_state,
            search_samples=atm_search_samples,
        )
        if on_idx is None or on_idx < baseline_window:
            continue

        resistance = resistance_2d[:num_rows, resistance_channel]
        baseline_values = resistance[on_idx - baseline_window : on_idx]
        baseline_values = baseline_values[np.isfinite(baseline_values)]
        if baseline_values.size != baseline_window:
            continue

        if threshold_search_samples is None:
            end = num_rows
        else:
            end = min(num_rows, on_idx + max(1, int(threshold_search_samples)))
        if end <= on_idx:
            continue

        baseline_value = float(np.mean(baseline_values))
        post_delta = resistance[on_idx:end] - baseline_value
        finite_mask = np.isfinite(post_delta)
        eligible_demo_count += 1

        x_crossing = np.flatnonzero(finite_mask & (post_delta >= x_threshold))
        y_crossing = np.flatnonzero(finite_mask & (post_delta >= y_threshold))
        if x_crossing.size > 0:
            x_reached_count += 1
        if y_crossing.size > 0:
            y_reached_count += 1
        if x_crossing.size == 0 or y_crossing.size == 0:
            continue

        x_idx = on_idx + int(x_crossing[0])
        y_idx = on_idx + int(y_crossing[0])
        first_derivative = np.diff(resistance[on_idx : x_idx + 1])
        interval_derivative = np.diff(resistance[x_idx : y_idx + 1])
        first_derivative = first_derivative[np.isfinite(first_derivative)]
        interval_derivative = interval_derivative[np.isfinite(interval_derivative)]
        if first_derivative.size == 0 or interval_derivative.size == 0:
            continue

        paired_demos.append(demo)
        first_peak_rates.append(float(np.max(first_derivative)))
        interval_peak_rates.append(float(np.max(interval_derivative)))

    return ThresholdPeakRateSummary(
        x_threshold=x_threshold,
        y_threshold=y_threshold,
        first_peak_rates=np.asarray(first_peak_rates, dtype=float),
        interval_peak_rates=np.asarray(interval_peak_rates, dtype=float),
        demo_names=tuple(paired_demos),
        eligible_demo_count=eligible_demo_count,
        x_reached_count=x_reached_count,
        y_reached_count=y_reached_count,
    )


def _difference_order_details(difference_order: int) -> tuple[int, str, str, str]:
    order = max(1, int(difference_order))
    if order == 1:
        return order, "derivative", "Resistance derivative", "MOhm/sample"
    if order == 2:
        return order, "second_derivative", "Resistance second derivative", "MOhm/sample^2"
    return order, f"difference_order_{order}", f"Resistance difference order {order}", f"MOhm/sample^{order}"


def collect_resistance_derivative_until_threshold_segments(
    hdf: h5py.File,
    demos: Sequence[str],
    *,
    action_path: str,
    atm_label: str,
    resistance_path: str,
    resistance_channel: int,
    threshold: float,
    max_samples: int | None,
    baseline_samples: int,
    threshold_search_samples: int | None,
    atm_search_samples: int | None,
    difference_order: int = 1,
) -> list[AtmAppliedWindowSegment]:
    threshold = float(threshold)
    if threshold <= 0.0:
        raise ValueError("threshold must be positive")

    segments: list[AtmAppliedWindowSegment] = []
    baseline_window = max(1, int(baseline_samples))
    order, _slug, _title, _units = _difference_order_details(difference_order)

    for demo in demos:
        demo_group = hdf["data"][demo]
        if action_path not in demo_group or resistance_path not in demo_group:
            continue

        action_ds = demo_group[action_path]
        resistance_ds = demo_group[resistance_path]
        if not isinstance(action_ds, h5py.Dataset) or not isinstance(resistance_ds, h5py.Dataset):
            continue

        atm_idx = _axis_index(action_ds, action_path, atm_label)
        if atm_idx is None:
            continue

        actions = dp._load_2d(action_ds, max_samples)
        resistance_2d = dp._load_2d(resistance_ds, max_samples)
        if resistance_channel < 0 or resistance_channel >= resistance_2d.shape[1]:
            continue

        num_rows = min(actions.shape[0], resistance_2d.shape[0])
        if num_rows == 0 or atm_idx >= actions.shape[1]:
            continue

        atm_state = actions[:num_rows, atm_idx]
        on_idx = find_first_atm_on_index(
            atm_state,
            search_samples=atm_search_samples,
        )
        if on_idx is None or on_idx < baseline_window or on_idx < order:
            continue

        resistance = resistance_2d[:num_rows, resistance_channel]
        baseline_values = resistance[on_idx - baseline_window : on_idx]
        baseline_values = baseline_values[np.isfinite(baseline_values)]
        if baseline_values.size != baseline_window:
            continue

        if threshold_search_samples is None:
            end = num_rows
        else:
            end = min(num_rows, on_idx + max(1, int(threshold_search_samples)))
        if end <= on_idx:
            continue

        baseline_value = float(np.mean(baseline_values))
        post_delta = resistance[on_idx:end] - baseline_value
        crossing = np.flatnonzero(np.isfinite(post_delta) & (post_delta >= threshold))
        if crossing.size == 0:
            continue

        crossing_idx = on_idx + int(crossing[0])
        diff_source = resistance[on_idx - order : crossing_idx + 1]
        if diff_source.size < order + 1:
            continue

        derivative = np.diff(diff_source, n=order)
        x = np.arange(derivative.size, dtype=float)
        segment_atm = (atm_state[on_idx : crossing_idx + 1] >= 0.5).astype(float)

        segments.append(
            AtmAppliedWindowSegment(
                demo=demo,
                x=x,
                values=derivative,
                atm_state=segment_atm,
                on_index=on_idx,
                baseline_value=baseline_value,
            )
        )

    return segments


def _aligned_mean(
    segments: Sequence[AtmOnOffSegment | AtmAppliedWindowSegment],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not segments:
        empty_float = np.asarray([], dtype=float)
        empty_count = np.asarray([], dtype=int)
        return empty_float, empty_float, empty_count

    min_x = int(min(segment.x[0] for segment in segments))
    max_x = int(max(segment.x[-1] for segment in segments))
    x_axis = np.arange(min_x, max_x + 1, dtype=float)
    aligned = np.full((len(segments), x_axis.size), np.nan, dtype=float)

    for segment_idx, segment in enumerate(segments):
        offset = int(segment.x[0] - min_x)
        aligned[segment_idx, offset : offset + segment.values.size] = segment.values

    counts = np.sum(np.isfinite(aligned), axis=0)
    summed = np.nansum(aligned, axis=0)
    mean = np.divide(
        summed,
        counts,
        out=np.full(x_axis.size, np.nan, dtype=float),
        where=counts > 0,
    )
    return x_axis, mean, counts.astype(int)


def truncate_segments_at_first_threshold(
    segments: Sequence[AtmAppliedWindowSegment],
) -> tuple[list[AtmAppliedWindowSegment], float | None]:
    valid_segments = [segment for segment in segments if segment.x.size > 0]
    if not valid_segments:
        return [], None

    common_endpoint = float(min(segment.x[-1] for segment in valid_segments))
    truncated_segments: list[AtmAppliedWindowSegment] = []
    for segment in valid_segments:
        keep_mask = segment.x <= common_endpoint
        if not np.any(keep_mask):
            continue

        if segment.atm_state.size == segment.x.size:
            atm_state = segment.atm_state[keep_mask]
        else:
            atm_state = segment.atm_state

        truncated_segments.append(
            AtmAppliedWindowSegment(
                demo=segment.demo,
                x=segment.x[keep_mask],
                values=segment.values[keep_mask],
                atm_state=atm_state,
                on_index=segment.on_index,
                baseline_value=segment.baseline_value,
            )
        )

    return truncated_segments, common_endpoint


def compute_normalized_derivative_profile(
    segments: Sequence[AtmAppliedWindowSegment],
    *,
    grid_points: int,
    confidence_z: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    grid = np.linspace(0.0, 1.0, max(2, int(grid_points)), dtype=float)
    rows: list[np.ndarray] = []

    for segment in segments:
        if segment.x.size < 2 or segment.values.size != segment.x.size:
            continue

        duration = float(segment.x[-1] - segment.x[0])
        if duration <= 0.0:
            continue

        normalized_x = (segment.x - segment.x[0]) / duration
        finite_mask = np.isfinite(normalized_x) & np.isfinite(segment.values)
        if np.count_nonzero(finite_mask) < 2:
            continue

        rows.append(np.interp(grid, normalized_x[finite_mask], segment.values[finite_mask]))

    counts = np.zeros(grid.shape, dtype=int)
    mean = np.full(grid.shape, np.nan, dtype=float)
    low = np.full(grid.shape, np.nan, dtype=float)
    high = np.full(grid.shape, np.nan, dtype=float)

    if not rows:
        return grid, mean, low, high, counts, 0

    aligned = np.vstack(rows)
    for grid_idx in range(grid.size):
        values = aligned[:, grid_idx]
        values = values[np.isfinite(values)]
        counts[grid_idx] = values.size
        if values.size == 0:
            continue

        mean[grid_idx] = float(np.mean(values))
        if values.size == 1:
            low[grid_idx] = mean[grid_idx]
            high[grid_idx] = mean[grid_idx]
            continue

        sem = float(np.std(values, ddof=1) / np.sqrt(values.size))
        margin = float(confidence_z) * sem
        low[grid_idx] = mean[grid_idx] - margin
        high[grid_idx] = mean[grid_idx] + margin

    return grid, mean, low, high, counts, len(rows)


def plot_first_atm_applied_window(
    hdf: h5py.File,
    demos: Sequence[str],
    *,
    action_path: str,
    atm_label: str,
    value_path: str,
    value_channel: int,
    max_samples: int | None,
    output_dir: Path | None,
    save_plots: bool,
    plot_format: str,
    plot_dpi: int,
    window_samples: int,
    value_ylim: tuple[float, float] | None,
    search_samples: int | None,
) -> None:
    segments = collect_first_atm_applied_window_segments(
        hdf,
        demos,
        action_path=action_path,
        atm_label=atm_label,
        value_path=value_path,
        value_channel=value_channel,
        max_samples=max_samples,
        window_samples=window_samples,
        search_samples=search_samples,
    )
    if not segments:
        print("Skipping first ATM-applied plot: no demos had an ATM-on sample.")
        return

    skipped_count = len(demos) - len(segments)
    window = max(1, int(window_samples))
    print(
        f"Found {len(segments)} demos with first ATM-on windows "
        f"({skipped_count} skipped, {window} samples requested)."
    )

    value_ds = dp._first_dataset_for_path(hdf, demos, value_path)
    value_labels = dp._channel_labels(value_path, value_ds) if value_ds is not None else []
    value_label = (
        value_labels[value_channel]
        if 0 <= value_channel < len(value_labels)
        else dp._pretty_path(value_path)
    )

    fig, (value_ax, atm_ax) = plt.subplots(
        2,
        1,
        figsize=(9, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0]},
    )

    for segment_idx, segment in enumerate(segments):
        color = dp._plot_color(segment_idx, len(segments))
        value_ax.plot(segment.x, segment.values, color=color, alpha=0.34, linewidth=1.1)
        atm_ax.step(segment.x, segment.atm_state, where="post", color=color, alpha=0.32, linewidth=1.0)

    mean_x, mean_values, mean_counts = _aligned_mean(segments)
    mean_mask = np.isfinite(mean_values) & (mean_counts > 0)
    if np.any(mean_mask):
        value_ax.plot(
            mean_x[mean_mask],
            mean_values[mean_mask],
            color="black",
            linewidth=2.0,
            label="Mean",
        )

    for ax in (value_ax, atm_ax):
        ax.axvline(0.0, color="tab:green", linestyle="--", linewidth=1.4)
        ax.grid(True, alpha=0.25)

    value_ax.plot([], [], color="tab:green", linestyle="--", linewidth=1.4, label="ATM applied")
    value_ax.set_title(f"First {window} samples after ATM is applied ({len(segments)} demos)")
    value_ax.set_ylabel(value_label)
    if value_ylim is not None:
        value_ax.set_ylim(*value_ylim)
    value_ax.legend(loc="best", fontsize=8)

    atm_ax.set_yticks([0.0, 1.0])
    atm_ax.set_yticklabels(["Applied pressure", "ATM"])
    atm_ax.set_ylabel("ATM")
    atm_ax.set_xlabel("Samples after first ATM applied")
    atm_ax.set_xlim(0.0, float(window - 1))

    fig.tight_layout()

    if save_plots and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_dir / f"first_{window}_samples_after_atm_applied_{dp._slug(value_path)}.{plot_format}",
            dpi=plot_dpi,
            bbox_inches="tight",
        )


def plot_first_atm_baseline_zeroed_window(
    hdf: h5py.File,
    demos: Sequence[str],
    *,
    action_path: str,
    atm_label: str,
    value_path: str,
    value_channel: int,
    max_samples: int | None,
    output_dir: Path | None,
    save_plots: bool,
    plot_format: str,
    plot_dpi: int,
    window_samples: int,
    baseline_samples: int,
    value_ylim: tuple[float, float] | None,
    search_samples: int | None,
) -> None:
    segments = collect_first_atm_baseline_zeroed_window_segments(
        hdf,
        demos,
        action_path=action_path,
        atm_label=atm_label,
        value_path=value_path,
        value_channel=value_channel,
        max_samples=max_samples,
        window_samples=window_samples,
        baseline_samples=baseline_samples,
        search_samples=search_samples,
    )
    if not segments:
        print("Skipping baseline-zeroed ATM plot: no demos had enough pre-ATM baseline samples.")
        return

    skipped_count = len(demos) - len(segments)
    window = max(1, int(window_samples))
    baseline_window = max(1, int(baseline_samples))
    print(
        f"Found {len(segments)} demos with {baseline_window}-sample pre-ATM baselines "
        f"({skipped_count} skipped, {window} post-ATM samples requested)."
    )

    value_ds = dp._first_dataset_for_path(hdf, demos, value_path)
    value_labels = dp._channel_labels(value_path, value_ds) if value_ds is not None else []
    value_label = (
        value_labels[value_channel]
        if 0 <= value_channel < len(value_labels)
        else dp._pretty_path(value_path)
    )

    fig, (value_ax, atm_ax) = plt.subplots(
        2,
        1,
        figsize=(9, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0]},
    )

    for segment_idx, segment in enumerate(segments):
        color = dp._plot_color(segment_idx, len(segments))
        value_ax.plot(segment.x, segment.values, color=color, alpha=0.34, linewidth=1.1)
        atm_ax.step(segment.x, segment.atm_state, where="post", color=color, alpha=0.32, linewidth=1.0)

    mean_x, mean_values, mean_counts = _aligned_mean(segments)
    mean_mask = np.isfinite(mean_values) & (mean_counts > 0)
    if np.any(mean_mask):
        value_ax.plot(
            mean_x[mean_mask],
            mean_values[mean_mask],
            color="black",
            linewidth=2.0,
            label="Mean",
        )

    for ax in (value_ax, atm_ax):
        ax.axvline(0.0, color="tab:green", linestyle="--", linewidth=1.4)
        ax.grid(True, alpha=0.25)

    value_ax.axhline(0.0, color="0.25", linestyle=":", linewidth=1.2)
    value_ax.plot([], [], color="tab:green", linestyle="--", linewidth=1.4, label="ATM applied")
    value_ax.set_title(
        f"First {window} samples after ATM, zeroed to prior {baseline_window}-sample mean "
        f"({len(segments)} demos)"
    )
    value_ax.set_ylabel(f"{value_label} change from baseline")
    if value_ylim is not None:
        value_ax.set_ylim(*value_ylim)
    value_ax.legend(loc="best", fontsize=8)

    atm_ax.set_yticks([0.0, 1.0])
    atm_ax.set_yticklabels(["Applied pressure", "ATM"])
    atm_ax.set_ylabel("ATM")
    atm_ax.set_xlabel("Samples after first ATM applied")
    atm_ax.set_xlim(0.0, float(window - 1))

    fig.tight_layout()

    if save_plots and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_dir
            / (
                f"first_{window}_samples_after_atm_applied_"
                f"zeroed_to_prior_{baseline_window}_sample_mean_{dp._slug(value_path)}.{plot_format}"
            ),
            dpi=plot_dpi,
            bbox_inches="tight",
        )


def plot_first_atm_windowed_delta(
    hdf: h5py.File,
    demos: Sequence[str],
    *,
    action_path: str,
    atm_label: str,
    resistance_input_path: str,
    max_samples: int | None,
    output_dir: Path | None,
    save_plots: bool,
    plot_format: str,
    plot_dpi: int,
    window_samples: int,
    zero_padding_is_missing: bool,
    value_ylim: tuple[float, float] | None,
    search_samples: int | None,
) -> None:
    segments = collect_first_atm_windowed_delta_segments(
        hdf,
        demos,
        action_path=action_path,
        atm_label=atm_label,
        resistance_input_path=resistance_input_path,
        max_samples=max_samples,
        window_samples=window_samples,
        zero_padding_is_missing=zero_padding_is_missing,
        search_samples=search_samples,
    )
    if not segments:
        print("Skipping ATM windowed-delta plot: no demos had ATM-on samples and resistance input.")
        return

    skipped_count = len(demos) - len(segments)
    window = max(1, int(window_samples))
    print(
        f"Found {len(segments)} demos with ATM-aligned resistance-input deltas "
        f"({skipped_count} skipped, {window} samples requested)."
    )

    fig, (value_ax, atm_ax) = plt.subplots(
        2,
        1,
        figsize=(9, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0]},
    )

    for segment_idx, segment in enumerate(segments):
        color = dp._plot_color(segment_idx, len(segments))
        value_ax.plot(segment.x, segment.values, color=color, alpha=0.34, linewidth=1.1)
        atm_ax.step(segment.x, segment.atm_state, where="post", color=color, alpha=0.32, linewidth=1.0)

    mean_x, mean_values, mean_counts = _aligned_mean(segments)
    mean_mask = np.isfinite(mean_values) & (mean_counts > 0)
    if np.any(mean_mask):
        value_ax.plot(
            mean_x[mean_mask],
            mean_values[mean_mask],
            color="black",
            linewidth=2.0,
            label="Mean",
        )

    for ax in (value_ax, atm_ax):
        ax.axvline(0.0, color="tab:green", linestyle="--", linewidth=1.4)
        ax.grid(True, alpha=0.25)

    value_ax.axhline(0.0, color="0.25", linestyle=":", linewidth=1.2)
    value_ax.plot([], [], color="tab:green", linestyle="--", linewidth=1.4, label="ATM applied")
    value_ax.set_title(f"Resistance-input window delta after ATM is applied ({len(segments)} demos)")
    value_ax.set_ylabel("Resistance input delta")
    if value_ylim is not None:
        value_ax.set_ylim(*value_ylim)
    value_ax.legend(loc="best", fontsize=8)

    atm_ax.set_yticks([0.0, 1.0])
    atm_ax.set_yticklabels(["Applied pressure", "ATM"])
    atm_ax.set_ylabel("ATM")
    atm_ax.set_xlabel("Samples after first ATM applied")
    atm_ax.set_xlim(0.0, float(window - 1))

    fig.tight_layout()

    if save_plots and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_dir
            / f"first_{window}_samples_after_atm_applied_windowed_delta_{dp._slug(resistance_input_path)}.{plot_format}",
            dpi=plot_dpi,
            bbox_inches="tight",
        )


def _plot_single_boxplot(
    values: np.ndarray,
    *,
    title: str,
    ylabel: str,
    ylim: tuple[float, float] | None,
    output_path: Path | None,
    plot_dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.boxplot(
        [values],
        widths=0.35,
        showmeans=True,
        meanline=True,
        patch_artist=True,
        boxprops={"facecolor": "#d8e8f5", "edgecolor": "0.25"},
        medianprops={"color": "0.1", "linewidth": 1.5},
        meanprops={"color": "tab:red", "linewidth": 1.4},
        whiskerprops={"color": "0.35"},
        capprops={"color": "0.35"},
        flierprops={"marker": "o", "markersize": 4, "markerfacecolor": "0.55", "markeredgecolor": "0.55"},
    )

    if values.size > 1:
        jitter = np.linspace(-0.08, 0.08, values.size)
    else:
        jitter = np.asarray([0.0])
    ax.scatter(
        np.ones(values.size, dtype=float) + jitter,
        values,
        s=18,
        color="black",
        alpha=0.38,
        linewidths=0.0,
        zorder=3,
    )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks([1])
    ax.set_xticklabels([f"n={values.size}"])
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=plot_dpi, bbox_inches="tight")


def plot_first_atm_on_off_boxplots(
    hdf: h5py.File,
    demos: Sequence[str],
    *,
    action_path: str,
    atm_label: str,
    resistance_path: str,
    resistance_channel: int,
    max_samples: int | None,
    output_dir: Path | None,
    save_plots: bool,
    plot_format: str,
    plot_dpi: int,
    search_samples: int | None,
    duration_ylim: tuple[float, float] | None,
    delta_ylim: tuple[float, float] | None,
) -> None:
    summary = compute_first_atm_on_off_boxplot_summary(
        hdf,
        demos,
        action_path=action_path,
        atm_label=atm_label,
        resistance_path=resistance_path,
        resistance_channel=resistance_channel,
        max_samples=max_samples,
        search_samples=search_samples,
    )
    if summary.used_demo_count == 0:
        print("Skipping ATM on/off boxplots: no demos had a first ATM activation and deactivation.")
        return

    skipped_count = len(demos) - summary.used_demo_count
    print(
        f"Found {summary.used_demo_count} demos with first ATM activation/deactivation intervals "
        f"({skipped_count} skipped)."
    )

    duration_path = None
    delta_path = None
    if save_plots and output_dir is not None:
        duration_path = output_dir / f"first_atm_activation_to_deactivation_duration_boxplot.{plot_format}"
        delta_path = output_dir / f"first_atm_activation_to_deactivation_resistance_delta_boxplot.{plot_format}"

    _plot_single_boxplot(
        summary.duration_samples,
        title="First ATM activation to first ATM deactivation duration",
        ylabel="Samples between ATM activation and deactivation",
        ylim=duration_ylim,
        output_path=duration_path,
        plot_dpi=plot_dpi,
    )
    _plot_single_boxplot(
        summary.resistance_deltas,
        title="Resistance change from first ATM activation to deactivation",
        ylabel="Resistance delta (MOhm)",
        ylim=delta_ylim,
        output_path=delta_path,
        plot_dpi=plot_dpi,
    )


def plot_resistance_threshold_times(
    hdf: h5py.File,
    demos: Sequence[str],
    *,
    action_path: str,
    atm_label: str,
    resistance_path: str,
    resistance_channel: int,
    thresholds: Sequence[float],
    max_samples: int | None,
    output_dir: Path | None,
    save_plots: bool,
    plot_format: str,
    plot_dpi: int,
    baseline_samples: int,
    threshold_search_samples: int | None,
    atm_search_samples: int | None,
    time_ylim: tuple[float, float] | None,
) -> None:
    summary = compute_resistance_threshold_time_summary(
        hdf,
        demos,
        action_path=action_path,
        atm_label=atm_label,
        resistance_path=resistance_path,
        resistance_channel=resistance_channel,
        thresholds=thresholds,
        max_samples=max_samples,
        baseline_samples=baseline_samples,
        threshold_search_samples=threshold_search_samples,
        atm_search_samples=atm_search_samples,
    )
    if summary.eligible_demo_count == 0:
        print("Skipping resistance threshold-time plot: no demos had a usable pre-ATM baseline.")
        return

    finite_mask = np.isfinite(summary.mean_times) & (summary.reached_counts > 0)
    if not np.any(finite_mask):
        print("Skipping resistance threshold-time plot: no thresholds were reached.")
        return

    print(
        f"Computed resistance threshold times from {summary.eligible_demo_count} eligible demos."
    )

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    x = np.arange(summary.thresholds.size, dtype=float)
    colors = np.where(finite_mask, "tab:blue", "0.82")
    heights = np.where(finite_mask, summary.mean_times, 0.0)
    yerr = np.where(finite_mask, summary.sem_times, 0.0)

    ax.bar(
        x,
        heights,
        yerr=yerr,
        color=colors,
        alpha=0.86,
        capsize=4,
        edgecolor="0.25",
        linewidth=0.8,
    )

    if time_ylim is not None:
        ax.set_ylim(*time_ylim)
    else:
        finite_top = heights[finite_mask] + yerr[finite_mask]
        if finite_top.size:
            ax.set_ylim(0.0, max(1.0, float(np.nanmax(finite_top)) * 1.18))

    y_min, y_max = ax.get_ylim()
    label_y_offset = 0.025 * max(1.0, y_max - y_min)
    for idx, (mean_time, reached) in enumerate(zip(summary.mean_times, summary.reached_counts)):
        label = f"n={int(reached)}/{summary.eligible_demo_count}"
        if np.isfinite(mean_time) and reached > 0:
            y_pos = min(y_max - label_y_offset, mean_time + yerr[idx] + label_y_offset)
            ax.text(idx, y_pos, label, ha="center", va="bottom", fontsize=8)
        else:
            ax.text(idx, label_y_offset, label, ha="center", va="bottom", fontsize=8, color="0.4")

    baseline_window = max(1, int(baseline_samples))
    ax.set_title(
        "Average samples to reach resistance increase above pre-ATM baseline "
        f"({baseline_window}-sample baseline)"
    )
    ax.set_xlabel("Resistance increase above baseline (MOhm)")
    ax.set_ylabel("Samples after first ATM applied")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{threshold:g}" for threshold in summary.thresholds])
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()

    if save_plots and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_dir / f"resistance_threshold_times_above_baseline.{plot_format}",
            dpi=plot_dpi,
            bbox_inches="tight",
        )


def plot_resistance_threshold_time_scatter(
    hdf: h5py.File,
    demos: Sequence[str],
    *,
    action_path: str,
    atm_label: str,
    resistance_path: str,
    resistance_channel: int,
    x_threshold: float,
    y_threshold: float,
    max_samples: int | None,
    output_dir: Path | None,
    save_plots: bool,
    plot_format: str,
    plot_dpi: int,
    baseline_samples: int,
    threshold_search_samples: int | None,
    atm_search_samples: int | None,
) -> None:
    summary = compute_resistance_threshold_time_pair_summary(
        hdf,
        demos,
        action_path=action_path,
        atm_label=atm_label,
        resistance_path=resistance_path,
        resistance_channel=resistance_channel,
        x_threshold=x_threshold,
        y_threshold=y_threshold,
        max_samples=max_samples,
        baseline_samples=baseline_samples,
        threshold_search_samples=threshold_search_samples,
        atm_search_samples=atm_search_samples,
    )
    if summary.eligible_demo_count == 0:
        print("Skipping resistance threshold scatter: no demos had a usable pre-ATM baseline.")
        return

    paired_count = int(summary.x_times.size)
    if paired_count == 0:
        print(
            "Skipping resistance threshold scatter: "
            f"no demos reached both +{summary.x_threshold:g} and +{summary.y_threshold:g} MOhm."
        )
        return

    print(
        "Computed resistance threshold scatter from "
        f"{paired_count}/{summary.eligible_demo_count} eligible demos."
    )

    stats = _scatter_relationship_stats(summary.x_times, summary.y_times)
    stats_text = _format_scatter_relationship_stats(stats)

    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    ax.scatter(
        summary.x_times,
        summary.y_times,
        s=32,
        color="tab:blue",
        alpha=0.76,
        edgecolors="white",
        linewidths=0.5,
    )

    all_times = np.concatenate([summary.x_times, summary.y_times])
    upper = max(1.0, float(np.nanmax(all_times)) * 1.08)
    ax.plot(
        [0.0, upper],
        [0.0, upper],
        color="0.35",
        linestyle="--",
        linewidth=1.1,
        label="equal sample count",
    )
    _plot_linear_fit_line(
        ax,
        stats,
        x_min=0.0,
        x_max=float(np.nanmax(summary.x_times)),
    )
    ax.set_xlim(0.0, upper)
    ax.set_ylim(0.0, upper)
    ax.set_aspect("equal", adjustable="box")

    baseline_window = max(1, int(baseline_samples))
    count_text = (
        f"+{summary.x_threshold:g} reached: "
        f"{summary.x_reached_count}/{summary.eligible_demo_count}\n"
        f"+{summary.y_threshold:g} reached: "
        f"{summary.y_reached_count}/{summary.eligible_demo_count}"
    )
    if stats_text:
        count_text = f"{count_text}\n{stats_text}"

    ax.set_title(
        "Samples to reach resistance thresholds above baseline\n"
        f"{baseline_window}-sample pre-ATM baseline, {paired_count} paired demos"
    )
    ax.set_xlabel(f"Samples to +{summary.x_threshold:g} MOhm above baseline")
    ax.set_ylabel(f"Samples to +{summary.y_threshold:g} MOhm above baseline")
    ax.text(
        0.04,
        0.96,
        count_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.88},
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    if save_plots and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        pair_slug = _threshold_group_slug((summary.x_threshold, summary.y_threshold))
        fig.savefig(
            output_dir / f"resistance_threshold_time_scatter_{pair_slug}_above_baseline.{plot_format}",
            dpi=plot_dpi,
            bbox_inches="tight",
        )


def plot_resistance_threshold_interval_scatter(
    hdf: h5py.File,
    demos: Sequence[str],
    *,
    action_path: str,
    atm_label: str,
    resistance_path: str,
    resistance_channel: int,
    x_threshold: float,
    y_threshold: float,
    max_samples: int | None,
    output_dir: Path | None,
    save_plots: bool,
    plot_format: str,
    plot_dpi: int,
    baseline_samples: int,
    threshold_search_samples: int | None,
    atm_search_samples: int | None,
) -> None:
    summary = compute_resistance_threshold_time_pair_summary(
        hdf,
        demos,
        action_path=action_path,
        atm_label=atm_label,
        resistance_path=resistance_path,
        resistance_channel=resistance_channel,
        x_threshold=x_threshold,
        y_threshold=y_threshold,
        max_samples=max_samples,
        baseline_samples=baseline_samples,
        threshold_search_samples=threshold_search_samples,
        atm_search_samples=atm_search_samples,
    )
    if summary.eligible_demo_count == 0:
        print("Skipping resistance threshold interval scatter: no demos had a usable pre-ATM baseline.")
        return

    paired_count = int(summary.x_times.size)
    if paired_count == 0:
        print(
            "Skipping resistance threshold interval scatter: "
            f"no demos reached both +{summary.x_threshold:g} and +{summary.y_threshold:g} MOhm."
        )
        return

    interval_times = summary.y_times - summary.x_times
    stats = _scatter_relationship_stats(summary.x_times, interval_times)
    stats_text = _format_scatter_relationship_stats(stats)
    median_interval = float(np.nanmedian(interval_times))
    mean_interval = float(np.nanmean(interval_times))

    print(
        "Computed resistance threshold interval scatter from "
        f"{paired_count}/{summary.eligible_demo_count} eligible demos."
    )

    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    ax.scatter(
        summary.x_times,
        interval_times,
        s=32,
        color="tab:green",
        alpha=0.76,
        edgecolors="white",
        linewidths=0.5,
    )

    x_max = max(1.0, float(np.nanmax(summary.x_times)) * 1.08)
    y_max = max(1.0, float(np.nanmax(interval_times)) * 1.18)
    _plot_linear_fit_line(ax, stats, x_min=0.0, x_max=x_max)
    ax.axhline(
        median_interval,
        color="0.35",
        linestyle="--",
        linewidth=1.1,
        label=f"median extra samples={median_interval:.1f}",
    )
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(0.0, y_max)

    baseline_window = max(1, int(baseline_samples))
    annotation = (
        f"paired demos: {paired_count}/{summary.eligible_demo_count}\n"
        f"mean extra samples={mean_interval:.1f}\n"
        f"median extra samples={median_interval:.1f}"
    )
    if stats_text:
        annotation = f"{annotation}\n{stats_text}"

    ax.set_title(
        "Additional samples from first threshold to second threshold\n"
        f"+{summary.x_threshold:g} to +{summary.y_threshold:g} MOhm, "
        f"{baseline_window}-sample pre-ATM baseline"
    )
    ax.set_xlabel(f"Samples to +{summary.x_threshold:g} MOhm above baseline")
    ax.set_ylabel(
        f"Additional samples from +{summary.x_threshold:g} to +{summary.y_threshold:g} MOhm"
    )
    ax.text(
        0.04,
        0.96,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.88},
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    if save_plots and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        pair_slug = _threshold_group_slug((summary.x_threshold, summary.y_threshold))
        fig.savefig(
            output_dir / f"resistance_threshold_interval_scatter_{pair_slug}_above_baseline.{plot_format}",
            dpi=plot_dpi,
            bbox_inches="tight",
        )


def plot_resistance_threshold_action_rate_scatter(
    hdf: h5py.File,
    demos: Sequence[str],
    *,
    action_path: str,
    atm_label: str,
    resistance_path: str,
    resistance_channel: int,
    threshold: float,
    max_samples: int | None,
    output_dir: Path | None,
    save_plots: bool,
    plot_format: str,
    plot_dpi: int,
    baseline_samples: int,
    threshold_search_samples: int | None,
    atm_search_samples: int | None,
    action_tolerance: float,
) -> None:
    summary = compute_resistance_threshold_action_rate_summary(
        hdf,
        demos,
        action_path=action_path,
        atm_label=atm_label,
        resistance_path=resistance_path,
        resistance_channel=resistance_channel,
        threshold=threshold,
        max_samples=max_samples,
        baseline_samples=baseline_samples,
        threshold_search_samples=threshold_search_samples,
        atm_search_samples=atm_search_samples,
        action_tolerance=action_tolerance,
    )
    if summary.eligible_demo_count == 0:
        print("Skipping resistance threshold action-rate scatter: no demos had a usable pre-ATM baseline.")
        return

    plotted_count = int(summary.sample_times.size)
    if plotted_count == 0:
        print(
            "Skipping resistance threshold action-rate scatter: "
            f"no demos reached +{summary.threshold:g} MOhm."
        )
        return

    print(
        "Computed resistance threshold action-rate scatter from "
        f"{plotted_count}/{summary.eligible_demo_count} eligible demos."
    )

    stats = _scatter_relationship_stats(summary.sample_times, summary.action_rates)
    stats_text = _format_scatter_relationship_stats(stats)

    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    ax.scatter(
        summary.sample_times,
        summary.action_rates,
        s=32,
        color="tab:purple",
        alpha=0.76,
        edgecolors="white",
        linewidths=0.5,
    )

    x_max = max(1.0, float(np.nanmax(summary.sample_times)) * 1.08)
    y_max = max(0.05, float(np.nanmax(summary.action_rates)) * 1.18)
    _plot_linear_fit_line(ax, stats, x_min=0.0, x_max=x_max)
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(0.0, y_max)

    baseline_window = max(1, int(baseline_samples))
    annotation = (
        f"+{summary.threshold:g} reached: {summary.reached_count}/{summary.eligible_demo_count}\n"
        "rate counted from ATM on"
    )
    if stats_text:
        annotation = f"{annotation}\n{stats_text}"

    ax.set_title(
        "Action rate to reach resistance threshold above baseline\n"
        f"+{summary.threshold:g} MOhm, {baseline_window}-sample pre-ATM baseline"
    )
    ax.set_xlabel(f"Samples to +{summary.threshold:g} MOhm above baseline")
    ax.set_ylabel(f"Action events/sample through +{summary.threshold:g} MOhm crossing")
    ax.text(
        0.04,
        0.96,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.88},
    )
    ax.grid(True, alpha=0.25)
    if math.isfinite(stats.slope) and math.isfinite(stats.intercept):
        ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    if save_plots and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        threshold_slug = _threshold_group_slug((summary.threshold,))
        fig.savefig(
            output_dir
            / f"resistance_threshold_action_rate_scatter_{threshold_slug}_above_baseline.{plot_format}",
            dpi=plot_dpi,
            bbox_inches="tight",
        )


def _threshold_window_label(lower: float, upper: float) -> str:
    if lower == 0.0:
        return f"0 to +{upper:g}"
    return f"+{lower:g} to +{upper:g}"


def _plot_threshold_window_boxes(
    ax: plt.Axes,
    values_by_window: Sequence[np.ndarray],
    positions: np.ndarray,
    *,
    facecolor: str,
    ylabel: str,
    ylim: tuple[float, float] | None,
) -> None:
    nonempty_indices = [
        idx for idx, values in enumerate(values_by_window) if values.size > 0
    ]
    if nonempty_indices:
        box = ax.boxplot(
            [values_by_window[idx] for idx in nonempty_indices],
            positions=positions[nonempty_indices],
            widths=0.58,
            showmeans=True,
            meanline=True,
            patch_artist=True,
            boxprops={"facecolor": facecolor, "edgecolor": "0.25"},
            medianprops={"color": "0.1", "linewidth": 1.5},
            meanprops={"color": "tab:red", "linewidth": 1.4},
            whiskerprops={"color": "0.35"},
            capprops={"color": "0.35"},
            flierprops={
                "marker": "o",
                "markersize": 4,
                "markerfacecolor": "0.55",
                "markeredgecolor": "0.55",
                "alpha": 0.45,
            },
        )
        for patch in box["boxes"]:
            patch.set_facecolor(facecolor)

    for idx, values in enumerate(values_by_window):
        if values.size == 0:
            continue
        jitter = np.linspace(-0.12, 0.12, values.size) if values.size > 1 else np.asarray([0.0])
        ax.scatter(
            np.full(values.size, positions[idx], dtype=float) + jitter,
            values,
            s=14,
            color="black",
            alpha=0.28,
            linewidths=0.0,
            zorder=3,
        )

    finite_values = [
        values[np.isfinite(values)]
        for values in values_by_window
        if np.any(np.isfinite(values))
    ]
    if ylim is not None:
        ax.set_ylim(*ylim)
    elif finite_values:
        combined = np.concatenate(finite_values)
        lower = min(0.0, float(np.nanmin(combined)) * 1.05)
        upper = max(1.0, float(np.nanpercentile(combined, 99.0)) * 1.18)
        ax.set_ylim(lower, upper)

    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)


def plot_resistance_threshold_window_sample_action_boxplots(
    hdf: h5py.File,
    demos: Sequence[str],
    *,
    action_path: str,
    atm_label: str,
    resistance_path: str,
    resistance_channel: int,
    windows: Sequence[tuple[float, float]],
    max_samples: int | None,
    output_dir: Path | None,
    save_plots: bool,
    plot_format: str,
    plot_dpi: int,
    baseline_samples: int,
    threshold_search_samples: int | None,
    atm_search_samples: int | None,
    action_tolerance: float,
    sample_ylim: tuple[float, float] | None,
    action_ylim: tuple[float, float] | None,
) -> None:
    summary = compute_resistance_threshold_window_sample_action_summary(
        hdf,
        demos,
        action_path=action_path,
        atm_label=atm_label,
        resistance_path=resistance_path,
        resistance_channel=resistance_channel,
        windows=windows,
        max_samples=max_samples,
        baseline_samples=baseline_samples,
        threshold_search_samples=threshold_search_samples,
        atm_search_samples=atm_search_samples,
        action_tolerance=action_tolerance,
    )
    if summary.eligible_demo_count == 0:
        print("Skipping threshold window sample/action boxplots: no demos had a usable pre-ATM baseline.")
        return

    if not any(values.size > 0 for values in summary.sample_counts):
        print("Skipping threshold window sample/action boxplots: no threshold windows were completed.")
        return

    print(
        "Computed threshold window sample/action boxplots from "
        f"{summary.eligible_demo_count} eligible demos."
    )

    positions = np.arange(1, len(summary.windows) + 1, dtype=float)
    labels = [
        _threshold_window_label(lower, upper)
        for lower, upper in summary.windows
    ]
    baseline_window = max(1, int(baseline_samples))

    fig, (sample_ax, action_ax) = plt.subplots(
        2,
        1,
        figsize=(2.0 * len(summary.windows) + 2.4, 8.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0]},
    )
    _plot_threshold_window_boxes(
        sample_ax,
        summary.sample_counts,
        positions,
        facecolor="#d8e8f5",
        ylabel="Samples in window",
        ylim=sample_ylim,
    )
    _plot_threshold_window_boxes(
        action_ax,
        summary.action_counts,
        positions,
        facecolor="#eadcf4",
        ylabel="Action events in same window",
        ylim=action_ylim,
    )

    sample_ax.set_title(
        "Samples and actions across shared resistance-threshold windows\n"
        f"{baseline_window}-sample pre-ATM baseline"
    )

    sample_y_min, sample_y_max = sample_ax.get_ylim()
    action_y_min, action_y_max = action_ax.get_ylim()
    sample_offset = 0.025 * max(1.0, sample_y_max - sample_y_min)
    action_offset = 0.025 * max(1.0, action_y_max - action_y_min)
    for idx, reached in enumerate(summary.reached_counts):
        sample_values = summary.sample_counts[idx]
        action_values = summary.action_counts[idx]
        label = f"n={int(reached)}/{summary.eligible_demo_count}"
        if sample_values.size > 0:
            sample_y = min(
                sample_y_max - sample_offset,
                float(np.nanmax(sample_values)) + sample_offset,
            )
            sample_ax.text(positions[idx], sample_y, label, ha="center", va="bottom", fontsize=8)
        else:
            sample_ax.text(positions[idx], sample_offset, label, ha="center", va="bottom", fontsize=8)

        if action_values.size > 0:
            action_y = min(
                action_y_max - action_offset,
                float(np.nanmax(action_values)) + action_offset,
            )
            action_ax.text(positions[idx], action_y, label, ha="center", va="bottom", fontsize=8)
        else:
            action_ax.text(positions[idx], action_offset, label, ha="center", va="bottom", fontsize=8)

    action_ax.set_xticks(positions)
    action_ax.set_xticklabels(labels)
    action_ax.set_xlabel("Resistance increase window above baseline (MOhm)")
    fig.tight_layout()

    if save_plots and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        window_slug = "_".join(
            f"{lower:g}_to_{upper:g}".replace(".", "p")
            for lower, upper in summary.windows
        )
        fig.savefig(
            output_dir
            / f"resistance_threshold_window_sample_action_boxplots_{window_slug}_above_baseline.{plot_format}",
            dpi=plot_dpi,
            bbox_inches="tight",
        )


def plot_resistance_threshold_peak_rate_boxplots(
    hdf: h5py.File,
    demos: Sequence[str],
    *,
    action_path: str,
    atm_label: str,
    resistance_path: str,
    resistance_channel: int,
    x_threshold: float,
    y_threshold: float,
    max_samples: int | None,
    output_dir: Path | None,
    save_plots: bool,
    plot_format: str,
    plot_dpi: int,
    baseline_samples: int,
    threshold_search_samples: int | None,
    atm_search_samples: int | None,
    rate_ylim: tuple[float, float] | None,
) -> None:
    summary = compute_resistance_threshold_peak_rate_summary(
        hdf,
        demos,
        action_path=action_path,
        atm_label=atm_label,
        resistance_path=resistance_path,
        resistance_channel=resistance_channel,
        x_threshold=x_threshold,
        y_threshold=y_threshold,
        max_samples=max_samples,
        baseline_samples=baseline_samples,
        threshold_search_samples=threshold_search_samples,
        atm_search_samples=atm_search_samples,
    )
    if summary.eligible_demo_count == 0:
        print("Skipping resistance peak-rate boxplot: no demos had a usable pre-ATM baseline.")
        return

    paired_count = int(summary.first_peak_rates.size)
    if paired_count == 0:
        print(
            "Skipping resistance peak-rate boxplot: "
            f"no demos reached both +{summary.x_threshold:g} and +{summary.y_threshold:g} MOhm "
            "with enough samples to compute both rates."
        )
        return

    print(
        "Computed resistance peak-rate boxplot from "
        f"{paired_count}/{summary.eligible_demo_count} eligible demos."
    )

    baseline_window = max(1, int(baseline_samples))
    plot_specs = (
        (
            "atm_to_first_threshold",
            summary.first_peak_rates,
            f"ATM to +{summary.x_threshold:g} MOhm",
            "Peak resistance rate on the way to first threshold",
            "#d8e8f5",
            f"+{summary.x_threshold:g} reached: {summary.x_reached_count}/{summary.eligible_demo_count}",
        ),
        (
            "first_to_second_threshold",
            summary.interval_peak_rates,
            f"+{summary.x_threshold:g} to +{summary.y_threshold:g} MOhm",
            "Peak resistance rate between thresholds",
            "#d9ead3",
            f"+{summary.y_threshold:g} reached: {summary.y_reached_count}/{summary.eligible_demo_count}",
        ),
    )

    for plot_slug, values, label, title, color, reached_text in plot_specs:
        fig, ax = plt.subplots(figsize=(5.8, 5.2))
        box = ax.boxplot(
            [values],
            widths=0.35,
            showmeans=True,
            meanline=True,
            patch_artist=True,
            boxprops={"facecolor": color, "edgecolor": "0.25"},
            medianprops={"color": "0.1", "linewidth": 1.5},
            meanprops={"color": "tab:red", "linewidth": 1.4},
            whiskerprops={"color": "0.35"},
            capprops={"color": "0.35"},
            flierprops={
                "marker": "o",
                "markersize": 4,
                "markerfacecolor": "0.55",
                "markeredgecolor": "0.55",
                "alpha": 0.45,
            },
        )
        if box["boxes"]:
            box["boxes"][0].set_facecolor(color)

        jitter = np.linspace(-0.08, 0.08, values.size) if values.size > 1 else np.asarray([0.0])
        ax.scatter(
            np.ones(values.size, dtype=float) + jitter,
            values,
            s=18,
            color="black",
            alpha=0.34,
            linewidths=0.0,
            zorder=3,
        )

        if rate_ylim is not None:
            ax.set_ylim(*rate_ylim)
        elif values.size > 0:
            lower = min(0.0, float(np.nanmin(values)) * 1.08)
            upper = max(1.0, float(np.nanmax(values)) * 1.12)
            ax.set_ylim(lower, upper)

        annotation = (
            f"{reached_text}\n"
            f"paired demos: {paired_count}/{summary.eligible_demo_count}\n"
            f"median peak: {float(np.nanmedian(values)):.3g}"
        )
        ax.text(
            0.04,
            0.96,
            annotation,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.88},
        )
        ax.set_title(
            f"{title}\n{label}, {baseline_window}-sample pre-ATM baseline"
        )
        ax.set_ylabel("Peak resistance derivative (MOhm/sample)")
        ax.set_xticks([1.0])
        ax.set_xticklabels([f"{label}\n(n={values.size})"])
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()

        if save_plots and output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            pair_slug = _threshold_group_slug((summary.x_threshold, summary.y_threshold))
            fig.savefig(
                output_dir
                / f"resistance_threshold_peak_rate_boxplot_{plot_slug}_{pair_slug}_above_baseline.{plot_format}",
                dpi=plot_dpi,
                bbox_inches="tight",
            )


def plot_resistance_threshold_peak_rate_scatter(
    hdf: h5py.File,
    demos: Sequence[str],
    *,
    action_path: str,
    atm_label: str,
    resistance_path: str,
    resistance_channel: int,
    x_threshold: float,
    y_threshold: float,
    max_samples: int | None,
    output_dir: Path | None,
    save_plots: bool,
    plot_format: str,
    plot_dpi: int,
    baseline_samples: int,
    threshold_search_samples: int | None,
    atm_search_samples: int | None,
) -> None:
    summary = compute_resistance_threshold_peak_rate_summary(
        hdf,
        demos,
        action_path=action_path,
        atm_label=atm_label,
        resistance_path=resistance_path,
        resistance_channel=resistance_channel,
        x_threshold=x_threshold,
        y_threshold=y_threshold,
        max_samples=max_samples,
        baseline_samples=baseline_samples,
        threshold_search_samples=threshold_search_samples,
        atm_search_samples=atm_search_samples,
    )
    if summary.eligible_demo_count == 0:
        print("Skipping resistance peak-rate scatter: no demos had a usable pre-ATM baseline.")
        return

    paired_count = int(summary.first_peak_rates.size)
    if paired_count == 0:
        print(
            "Skipping resistance peak-rate scatter: "
            f"no demos reached both +{summary.x_threshold:g} and +{summary.y_threshold:g} MOhm "
            "with enough samples to compute both rates."
        )
        return

    print(
        "Computed resistance peak-rate scatter from "
        f"{paired_count}/{summary.eligible_demo_count} eligible demos."
    )

    x_values = summary.first_peak_rates
    y_values = summary.interval_peak_rates
    stats = _scatter_relationship_stats(x_values, y_values)
    stats_text = _format_scatter_relationship_stats(stats)

    x_min = min(0.0, float(np.nanmin(x_values)) * 1.08)
    x_max = max(1.0, float(np.nanmax(x_values)) * 1.12)
    y_min = min(0.0, float(np.nanmin(y_values)) * 1.08)
    y_max = max(1.0, float(np.nanmax(y_values)) * 1.12)

    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    ax.scatter(
        x_values,
        y_values,
        s=34,
        color="tab:cyan",
        alpha=0.78,
        edgecolors="white",
        linewidths=0.5,
    )
    _plot_linear_fit_line(ax, stats, x_min=x_min, x_max=x_max)

    equal_min = max(x_min, y_min)
    equal_max = min(x_max, y_max)
    if equal_max > equal_min:
        ax.plot(
            [equal_min, equal_max],
            [equal_min, equal_max],
            color="0.35",
            linestyle="--",
            linewidth=1.1,
            label="equal peak rate",
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    baseline_window = max(1, int(baseline_samples))
    annotation = (
        f"paired demos: {paired_count}/{summary.eligible_demo_count}\n"
        f"median peaks: {float(np.nanmedian(x_values)):.3g}, "
        f"{float(np.nanmedian(y_values)):.3g}"
    )
    if stats_text:
        annotation = f"{annotation}\n{stats_text}"

    ax.set_title(
        "Peak resistance rate before vs after first threshold\n"
        f"+{summary.x_threshold:g} to +{summary.y_threshold:g} MOhm, "
        f"{baseline_window}-sample pre-ATM baseline"
    )
    ax.set_xlabel(
        f"Peak derivative from ATM to +{summary.x_threshold:g} MOhm (MOhm/sample)"
    )
    ax.set_ylabel(
        f"Peak derivative from +{summary.x_threshold:g} to +{summary.y_threshold:g} MOhm "
        "(MOhm/sample)"
    )
    ax.text(
        0.04,
        0.96,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.88},
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    if save_plots and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        pair_slug = _threshold_group_slug((summary.x_threshold, summary.y_threshold))
        fig.savefig(
            output_dir / f"resistance_threshold_peak_rate_scatter_{pair_slug}_above_baseline.{plot_format}",
            dpi=plot_dpi,
            bbox_inches="tight",
        )


def _threshold_group_slug(thresholds: Sequence[float]) -> str:
    return "_".join(f"{float(threshold):g}".replace(".", "p") for threshold in thresholds)


def plot_resistance_threshold_time_boxplots(
    hdf: h5py.File,
    demos: Sequence[str],
    *,
    action_path: str,
    atm_label: str,
    resistance_path: str,
    resistance_channel: int,
    threshold_groups: Sequence[Sequence[float]],
    max_samples: int | None,
    output_dir: Path | None,
    save_plots: bool,
    plot_format: str,
    plot_dpi: int,
    baseline_samples: int,
    threshold_search_samples: int | None,
    atm_search_samples: int | None,
    time_ylim: tuple[float, float] | None,
) -> None:
    for thresholds in threshold_groups:
        threshold_values = tuple(float(threshold) for threshold in thresholds)
        if not threshold_values:
            continue

        summary = compute_resistance_threshold_time_summary(
            hdf,
            demos,
            action_path=action_path,
            atm_label=atm_label,
            resistance_path=resistance_path,
            resistance_channel=resistance_channel,
            thresholds=threshold_values,
            max_samples=max_samples,
            baseline_samples=baseline_samples,
            threshold_search_samples=threshold_search_samples,
            atm_search_samples=atm_search_samples,
        )
        if summary.eligible_demo_count == 0:
            print(
                "Skipping resistance threshold boxplot "
                f"{threshold_values}: no demos had a usable pre-ATM baseline."
            )
            continue

        nonempty_indices = [
            idx for idx, times in enumerate(summary.threshold_times) if times.size > 0
        ]
        if not nonempty_indices:
            print(f"Skipping resistance threshold boxplot {threshold_values}: no thresholds were reached.")
            continue

        fig, ax = plt.subplots(figsize=(2.2 * len(threshold_values) + 2.0, 5.2))
        positions = np.asarray(nonempty_indices, dtype=float) + 1.0
        box_data = [summary.threshold_times[idx] for idx in nonempty_indices]
        ax.boxplot(
            box_data,
            positions=positions,
            widths=0.62,
            patch_artist=True,
            showmeans=True,
            boxprops={"facecolor": "tab:blue", "alpha": 0.58, "edgecolor": "0.25"},
            medianprops={"color": "black", "linewidth": 1.6},
            meanprops={
                "marker": "o",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "markersize": 5,
            },
            whiskerprops={"color": "0.25"},
            capprops={"color": "0.25"},
            flierprops={
                "marker": "o",
                "markerfacecolor": "tab:blue",
                "markeredgecolor": "none",
                "alpha": 0.3,
                "markersize": 3,
            },
        )

        if time_ylim is not None:
            ax.set_ylim(*time_ylim)
        else:
            finite_times = np.concatenate(box_data)
            if finite_times.size:
                ax.set_ylim(0.0, max(1.0, float(np.nanpercentile(finite_times, 99.0)) * 1.2))

        y_min, y_max = ax.get_ylim()
        label_y_offset = 0.025 * max(1.0, y_max - y_min)
        for idx, reached in enumerate(summary.reached_counts):
            label = f"n={int(reached)}/{summary.eligible_demo_count}"
            if reached > 0:
                times = summary.threshold_times[idx]
                y_pos = min(y_max - label_y_offset, float(np.nanmax(times)) + label_y_offset)
                ax.text(idx + 1, y_pos, label, ha="center", va="bottom", fontsize=8)
            else:
                ax.text(idx + 1, label_y_offset, label, ha="center", va="bottom", fontsize=8, color="0.4")

        baseline_window = max(1, int(baseline_samples))
        threshold_text = ", ".join(f"{threshold:g}" for threshold in summary.thresholds)
        ax.set_title(
            "Samples to resistance increase above baseline\n"
            f"thresholds {threshold_text} MOhm, {baseline_window}-sample baseline"
        )
        ax.set_xlabel("Resistance increase above baseline (MOhm)")
        ax.set_ylabel("Samples after first ATM applied")
        ax.set_xlim(0.4, len(threshold_values) + 0.6)
        ax.set_xticks(np.arange(1, len(threshold_values) + 1))
        ax.set_xticklabels([f"{threshold:g}" for threshold in summary.thresholds])
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()

        if save_plots and output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                output_dir
                / f"resistance_threshold_time_boxplot_{_threshold_group_slug(threshold_values)}.{plot_format}",
                dpi=plot_dpi,
                bbox_inches="tight",
            )


def plot_resistance_derivative_until_threshold(
    hdf: h5py.File,
    demos: Sequence[str],
    *,
    action_path: str,
    atm_label: str,
    resistance_path: str,
    resistance_channel: int,
    threshold: float,
    max_samples: int | None,
    output_dir: Path | None,
    save_plots: bool,
    plot_format: str,
    plot_dpi: int,
    baseline_samples: int,
    threshold_search_samples: int | None,
    atm_search_samples: int | None,
    truncate_at_first_threshold: bool,
    value_ylim: tuple[float, float] | None,
) -> None:
    search_limit = None
    if threshold_search_samples is not None:
        search_limit = max(1, int(threshold_search_samples))

    segments = collect_resistance_derivative_until_threshold_segments(
        hdf,
        demos,
        action_path=action_path,
        atm_label=atm_label,
        resistance_path=resistance_path,
        resistance_channel=resistance_channel,
        threshold=threshold,
        max_samples=max_samples,
        baseline_samples=baseline_samples,
        threshold_search_samples=search_limit,
        atm_search_samples=atm_search_samples,
    )
    search_note = None
    if search_limit is not None:
        search_note = f"within first {search_limit} samples"

    if not segments:
        search_phrase = f" {search_note}" if search_note is not None else ""
        print(
            "Skipping resistance derivative plot: "
            f"no demos reached baseline + {float(threshold):g} MOhm{search_phrase}."
        )
        return

    skipped_count = len(demos) - len(segments)
    threshold = float(threshold)
    baseline_window = max(1, int(baseline_samples))
    search_phrase = f" {search_note}" if search_note is not None else ""
    print(
        f"Found {len(segments)} demos reaching baseline + {threshold:g} MOhm"
        f"{search_phrase} ({skipped_count} skipped)."
    )

    plot_segments = segments
    common_endpoint = None
    if truncate_at_first_threshold:
        plot_segments, common_endpoint = truncate_segments_at_first_threshold(segments)
        if not plot_segments:
            print("Skipping resistance derivative plot: no traces remained after truncation.")
            return
        print(
            "Truncated derivative plot at earliest threshold crossing "
            f"({common_endpoint:g} samples after ATM)."
        )

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    for segment_idx, segment in enumerate(plot_segments):
        color = dp._plot_color(segment_idx, len(plot_segments))
        ax.plot(segment.x, segment.values, color=color, alpha=0.34, linewidth=1.1)

    mean_x, mean_values, mean_counts = _aligned_mean(plot_segments)
    mean_mask = np.isfinite(mean_values) & (mean_counts > 0)
    if np.any(mean_mask):
        ax.plot(
            mean_x[mean_mask],
            mean_values[mean_mask],
            color="black",
            linewidth=2.0,
            label="Mean",
        )

    ax.axvline(0.0, color="tab:green", linestyle="--", linewidth=1.4, label="ATM applied")
    ax.axhline(0.0, color="0.25", linestyle=":", linewidth=1.2)
    title = (
        "Resistance derivative until first threshold crossing\n"
        f"+{threshold:g} MOhm over {baseline_window}-sample pre-ATM baseline ({len(segments)} demos)"
    )
    if search_limit is not None:
        title = f"{title}\nThreshold search limited to first {search_limit} samples after ATM"
    if common_endpoint is not None:
        title = f"{title}\nTruncated where the first trace reaches threshold ({common_endpoint:g} samples)"
    ax.set_title(title)
    ax.set_xlabel("Samples after first ATM applied")
    ax.set_ylabel("Resistance derivative (MOhm/sample)")
    if value_ylim is not None:
        ax.set_ylim(*value_ylim)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    if save_plots and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        limit_slug = ""
        if search_limit is not None:
            limit_slug = f"_first_{search_limit}_samples"
        truncate_slug = ""
        if common_endpoint is not None:
            truncate_slug = "_common_duration"
        fig.savefig(
            output_dir
            / (
                f"resistance_derivative_until_{threshold:g}_mohm_above_baseline"
                f"{limit_slug}{truncate_slug}.{plot_format}"
            ),
            dpi=plot_dpi,
            bbox_inches="tight",
        )


def plot_resistance_derivative_normalized_to_threshold(
    hdf: h5py.File,
    demos: Sequence[str],
    *,
    action_path: str,
    atm_label: str,
    resistance_path: str,
    resistance_channel: int,
    threshold: float,
    max_samples: int | None,
    output_dir: Path | None,
    save_plots: bool,
    plot_format: str,
    plot_dpi: int,
    baseline_samples: int,
    threshold_search_samples: int | None,
    atm_search_samples: int | None,
    value_ylim: tuple[float, float] | None,
    grid_points: int,
    confidence_z: float,
) -> None:
    search_limit = None
    if threshold_search_samples is not None:
        search_limit = max(1, int(threshold_search_samples))

    segments = collect_resistance_derivative_until_threshold_segments(
        hdf,
        demos,
        action_path=action_path,
        atm_label=atm_label,
        resistance_path=resistance_path,
        resistance_channel=resistance_channel,
        threshold=threshold,
        max_samples=max_samples,
        baseline_samples=baseline_samples,
        threshold_search_samples=search_limit,
        atm_search_samples=atm_search_samples,
    )
    grid, mean, low, high, counts, used_count = compute_normalized_derivative_profile(
        segments,
        grid_points=grid_points,
        confidence_z=confidence_z,
    )
    if used_count == 0:
        print(
            "Skipping normalized resistance derivative plot: "
            f"no demos had at least two samples before baseline + {float(threshold):g} MOhm."
        )
        return

    threshold = float(threshold)
    baseline_window = max(1, int(baseline_samples))
    skipped_count = len(demos) - len(segments)
    search_note = None
    if search_limit is not None:
        search_note = f"within first {search_limit} samples"
    search_phrase = f" {search_note}" if search_note is not None else ""
    print(
        f"Normalized {used_count} derivative traces from baseline to +{threshold:g} MOhm"
        f"{search_phrase} ({skipped_count} skipped before normalization)."
    )

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    for segment_idx, segment in enumerate(segments):
        if segment.x.size < 2:
            continue
        duration = float(segment.x[-1] - segment.x[0])
        if duration <= 0.0:
            continue
        normalized_x = (segment.x - segment.x[0]) / duration
        color = dp._plot_color(segment_idx, len(segments))
        ax.plot(
            normalized_x * 100.0,
            segment.values,
            color=color,
            alpha=0.26,
            linewidth=1.0,
        )

    mean_mask = np.isfinite(mean) & (counts > 0)
    ci_mask = np.isfinite(low) & np.isfinite(high) & (counts > 1)
    if np.any(ci_mask):
        ci_label = f"{confidence_z:g}z CI"
        if math.isclose(float(confidence_z), 1.96, rel_tol=0.0, abs_tol=0.005):
            ci_label = "95% CI"
        ax.fill_between(
            grid[ci_mask] * 100.0,
            low[ci_mask],
            high[ci_mask],
            color="black",
            alpha=0.16,
            linewidth=0.0,
            label=ci_label,
        )
    if np.any(mean_mask):
        ax.plot(
            grid[mean_mask] * 100.0,
            mean[mean_mask],
            color="black",
            linewidth=2.0,
            label="Mean",
        )

    ax.axvline(0.0, color="tab:green", linestyle="--", linewidth=1.4, label="ATM applied")
    ax.axvline(100.0, color="tab:red", linestyle="--", linewidth=1.4, label="+1 MOhm")
    ax.axhline(0.0, color="0.25", linestyle=":", linewidth=1.2)
    title = (
        "Resistance derivative normalized to baseline-to-threshold duration\n"
        f"+{threshold:g} MOhm over {baseline_window}-sample pre-ATM baseline ({used_count} demos)"
    )
    if search_limit is not None:
        title = f"{title}\nThreshold search limited to first {search_limit} samples after ATM"
    ax.set_title(title)
    ax.set_xlabel("Normalized duration from ATM baseline to +1 MOhm (%)")
    ax.set_ylabel("Resistance derivative (MOhm/sample)")
    ax.set_xlim(0.0, 100.0)
    if value_ylim is not None:
        ax.set_ylim(*value_ylim)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    if save_plots and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        limit_slug = ""
        if search_limit is not None:
            limit_slug = f"_first_{search_limit}_samples"
        fig.savefig(
            output_dir
            / (
                f"resistance_derivative_normalized_to_{threshold:g}_mohm_above_baseline"
                f"{limit_slug}.{plot_format}"
            ),
            dpi=plot_dpi,
            bbox_inches="tight",
        )


def plot_first_atm_on_to_applied_pressure_window(
    hdf: h5py.File,
    demos: Sequence[str],
    *,
    action_path: str,
    atm_label: str,
    value_path: str,
    value_channel: int,
    max_samples: int | None,
    output_dir: Path | None,
    save_plots: bool,
    plot_format: str,
    plot_dpi: int,
    pre_samples: int,
    post_samples: int,
    search_samples: int | None,
) -> None:
    segments = collect_first_atm_on_to_applied_pressure_segments(
        hdf,
        demos,
        action_path=action_path,
        atm_label=atm_label,
        value_path=value_path,
        value_channel=value_channel,
        max_samples=max_samples,
        pre_samples=pre_samples,
        post_samples=post_samples,
        search_samples=search_samples,
    )
    if not segments:
        print(
            "Skipping first ATM to applied-pressure plot: "
            "no demos had a complete first ATM-on to applied-pressure interval."
        )
        return

    skipped_count = len(demos) - len(segments)
    print(
        "Found "
        f"{len(segments)} demos with complete first ATM-on to applied-pressure intervals "
        f"({skipped_count} skipped)."
    )

    value_ds = dp._first_dataset_for_path(hdf, demos, value_path)
    value_labels = dp._channel_labels(value_path, value_ds) if value_ds is not None else []
    value_label = (
        value_labels[value_channel]
        if 0 <= value_channel < len(value_labels)
        else dp._pretty_path(value_path)
    )

    fig, (value_ax, atm_ax) = plt.subplots(
        2,
        1,
        figsize=(10, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0]},
    )

    for segment_idx, segment in enumerate(segments):
        color = dp._plot_color(segment_idx, len(segments))
        value_ax.plot(segment.x, segment.values, color=color, alpha=0.34, linewidth=1.1)
        atm_ax.step(segment.x, segment.atm_state, where="post", color=color, alpha=0.32, linewidth=1.0)

    applied_x = np.asarray([segment.applied_pressure_x for segment in segments], dtype=float)
    applied_y = np.asarray([segment.applied_pressure_value for segment in segments], dtype=float)
    applied_mask = np.isfinite(applied_x) & np.isfinite(applied_y)
    if np.any(applied_mask):
        value_ax.scatter(
            applied_x[applied_mask],
            applied_y[applied_mask],
            s=16,
            color="tab:red",
            alpha=0.72,
            edgecolors="none",
            label="Applied pressure resumes",
        )

    mean_x, mean_values, mean_counts = _aligned_mean(segments)
    mean_mask = np.isfinite(mean_values) & (mean_counts > 0)
    if np.any(mean_mask):
        value_ax.plot(
            mean_x[mean_mask],
            mean_values[mean_mask],
            color="black",
            linewidth=2.0,
            label="Mean",
        )

    for ax in (value_ax, atm_ax):
        ax.axvline(0.0, color="tab:green", linestyle="--", linewidth=1.4)
        ax.grid(True, alpha=0.25)

    value_ax.plot([], [], color="tab:green", linestyle="--", linewidth=1.4, label="ATM on")
    value_ax.set_title(
        "First ATM-on to applied-pressure window "
        f"({len(segments)} demos, {max(0, int(pre_samples))} pre / {max(0, int(post_samples))} post samples)"
    )
    value_ax.set_ylabel(value_label)
    value_ax.legend(loc="best", fontsize=8)

    atm_ax.set_yticks([0.0, 1.0])
    atm_ax.set_yticklabels(["Applied pressure", "ATM"])
    atm_ax.set_ylabel("ATM")
    atm_ax.set_xlabel("Samples relative to first ATM on")

    fig.tight_layout()

    if save_plots and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_dir / f"first_atm_on_to_applied_pressure_{dp._slug(value_path)}.{plot_format}",
            dpi=plot_dpi,
            bbox_inches="tight",
        )


def analyze_gigaseal_dataset(
    file_path: Path,
    *,
    max_demos: int | None,
    max_samples: int | None,
    save_plots: bool,
    show_plots: bool,
    output_dir: Path | None,
    plot_format: str,
    plot_dpi: int,
) -> None:
    with h5py.File(file_path, "r") as hdf:
        demos = dp._find_demos(hdf, max_demos=max_demos)
        if not demos:
            raise RuntimeError(f"No demos found in {file_path}")

        print(f"Found {len(demos)} demos in {file_path}")

        if PLOT_AVERAGE_ACTION_EVENT_FREQUENCY:
            plot_average_action_event_frequency(
                hdf,
                demos,
                action_path=ACTION_PATH,
                max_samples=max_samples,
                output_dir=output_dir,
                save_plots=save_plots,
                plot_format=plot_format,
                plot_dpi=plot_dpi,
                window_samples=EVENT_FREQUENCY_WINDOW_SAMPLES,
                sample_period_seconds=EVENT_FREQUENCY_SAMPLE_PERIOD_SECONDS,
                tolerance=EVENT_FREQUENCY_TOLERANCE,
                confidence_z=EVENT_FREQUENCY_CONFIDENCE_Z,
            )

        if PLOT_FIRST_ATM_APPLIED_WINDOW:
            plot_first_atm_applied_window(
                hdf,
                demos,
                action_path=ACTION_PATH,
                atm_label=ATM_STATE_LABEL,
                value_path=ATM_WINDOW_VALUE_PATH,
                value_channel=ATM_WINDOW_VALUE_CHANNEL,
                max_samples=max_samples,
                output_dir=output_dir,
                save_plots=save_plots,
                plot_format=plot_format,
                plot_dpi=plot_dpi,
                window_samples=ATM_WINDOW_SAMPLES,
                value_ylim=ATM_WINDOW_VALUE_YLIM,
                search_samples=ATM_WINDOW_SEARCH_SAMPLES,
            )

        if PLOT_FIRST_ATM_BASELINE_ZEROED_WINDOW:
            plot_first_atm_baseline_zeroed_window(
                hdf,
                demos,
                action_path=ACTION_PATH,
                atm_label=ATM_STATE_LABEL,
                value_path=ATM_WINDOW_VALUE_PATH,
                value_channel=ATM_WINDOW_VALUE_CHANNEL,
                max_samples=max_samples,
                output_dir=output_dir,
                save_plots=save_plots,
                plot_format=plot_format,
                plot_dpi=plot_dpi,
                window_samples=ATM_WINDOW_SAMPLES,
                baseline_samples=ATM_BASELINE_SAMPLES,
                value_ylim=ATM_BASELINE_ZEROED_VALUE_YLIM,
                search_samples=ATM_WINDOW_SEARCH_SAMPLES,
            )

        if PLOT_FIRST_ATM_WINDOWED_DELTA:
            plot_first_atm_windowed_delta(
                hdf,
                demos,
                action_path=ACTION_PATH,
                atm_label=ATM_STATE_LABEL,
                resistance_input_path=RESISTANCE_INPUT_PATH,
                max_samples=max_samples,
                output_dir=output_dir,
                save_plots=save_plots,
                plot_format=plot_format,
                plot_dpi=plot_dpi,
                window_samples=ATM_WINDOW_SAMPLES,
                zero_padding_is_missing=RESISTANCE_INPUT_ZERO_PADDING_IS_MISSING,
                value_ylim=ATM_WINDOWED_DELTA_YLIM,
                search_samples=ATM_WINDOW_SEARCH_SAMPLES,
            )

        if PLOT_FIRST_ATM_ON_OFF_BOXPLOTS:
            plot_first_atm_on_off_boxplots(
                hdf,
                demos,
                action_path=ACTION_PATH,
                atm_label=ATM_STATE_LABEL,
                resistance_path=RESISTANCE_VALUE_PATH,
                resistance_channel=ATM_WINDOW_VALUE_CHANNEL,
                max_samples=max_samples,
                output_dir=output_dir,
                save_plots=save_plots,
                plot_format=plot_format,
                plot_dpi=plot_dpi,
                search_samples=ATM_WINDOW_SEARCH_SAMPLES,
                duration_ylim=ATM_ON_OFF_DURATION_YLIM,
                delta_ylim=ATM_ON_OFF_RESISTANCE_DELTA_YLIM,
            )

        if PLOT_RESISTANCE_THRESHOLD_TIMES:
            plot_resistance_threshold_times(
                hdf,
                demos,
                action_path=ACTION_PATH,
                atm_label=ATM_STATE_LABEL,
                resistance_path=RESISTANCE_VALUE_PATH,
                resistance_channel=ATM_WINDOW_VALUE_CHANNEL,
                thresholds=RESISTANCE_THRESHOLD_VALUES,
                max_samples=max_samples,
                output_dir=output_dir,
                save_plots=save_plots,
                plot_format=plot_format,
                plot_dpi=plot_dpi,
                baseline_samples=RESISTANCE_THRESHOLD_BASELINE_SAMPLES,
                threshold_search_samples=RESISTANCE_THRESHOLD_SEARCH_SAMPLES,
                atm_search_samples=ATM_WINDOW_SEARCH_SAMPLES,
                time_ylim=RESISTANCE_THRESHOLD_TIME_YLIM,
            )

        if PLOT_RESISTANCE_THRESHOLD_TIME_SCATTER:
            plot_resistance_threshold_time_scatter(
                hdf,
                demos,
                action_path=ACTION_PATH,
                atm_label=ATM_STATE_LABEL,
                resistance_path=RESISTANCE_VALUE_PATH,
                resistance_channel=ATM_WINDOW_VALUE_CHANNEL,
                x_threshold=RESISTANCE_THRESHOLD_SCATTER_X_THRESHOLD,
                y_threshold=RESISTANCE_THRESHOLD_SCATTER_Y_THRESHOLD,
                max_samples=max_samples,
                output_dir=output_dir,
                save_plots=save_plots,
                plot_format=plot_format,
                plot_dpi=plot_dpi,
                baseline_samples=RESISTANCE_THRESHOLD_BASELINE_SAMPLES,
                threshold_search_samples=RESISTANCE_THRESHOLD_SEARCH_SAMPLES,
                atm_search_samples=ATM_WINDOW_SEARCH_SAMPLES,
            )

        if PLOT_RESISTANCE_THRESHOLD_INTERVAL_SCATTER:
            plot_resistance_threshold_interval_scatter(
                hdf,
                demos,
                action_path=ACTION_PATH,
                atm_label=ATM_STATE_LABEL,
                resistance_path=RESISTANCE_VALUE_PATH,
                resistance_channel=ATM_WINDOW_VALUE_CHANNEL,
                x_threshold=RESISTANCE_THRESHOLD_SCATTER_X_THRESHOLD,
                y_threshold=RESISTANCE_THRESHOLD_SCATTER_Y_THRESHOLD,
                max_samples=max_samples,
                output_dir=output_dir,
                save_plots=save_plots,
                plot_format=plot_format,
                plot_dpi=plot_dpi,
                baseline_samples=RESISTANCE_THRESHOLD_BASELINE_SAMPLES,
                threshold_search_samples=RESISTANCE_THRESHOLD_SEARCH_SAMPLES,
                atm_search_samples=ATM_WINDOW_SEARCH_SAMPLES,
            )

        if PLOT_RESISTANCE_THRESHOLD_ACTION_RATE_SCATTER:
            plot_resistance_threshold_action_rate_scatter(
                hdf,
                demos,
                action_path=ACTION_PATH,
                atm_label=ATM_STATE_LABEL,
                resistance_path=RESISTANCE_VALUE_PATH,
                resistance_channel=ATM_WINDOW_VALUE_CHANNEL,
                threshold=RESISTANCE_THRESHOLD_ACTION_RATE_THRESHOLD,
                max_samples=max_samples,
                output_dir=output_dir,
                save_plots=save_plots,
                plot_format=plot_format,
                plot_dpi=plot_dpi,
                baseline_samples=RESISTANCE_THRESHOLD_BASELINE_SAMPLES,
                threshold_search_samples=RESISTANCE_THRESHOLD_SEARCH_SAMPLES,
                atm_search_samples=ATM_WINDOW_SEARCH_SAMPLES,
                action_tolerance=RESISTANCE_THRESHOLD_ACTION_RATE_TOLERANCE,
            )

        if PLOT_RESISTANCE_THRESHOLD_WINDOW_SAMPLE_ACTION_BOXPLOTS:
            plot_resistance_threshold_window_sample_action_boxplots(
                hdf,
                demos,
                action_path=ACTION_PATH,
                atm_label=ATM_STATE_LABEL,
                resistance_path=RESISTANCE_VALUE_PATH,
                resistance_channel=ATM_WINDOW_VALUE_CHANNEL,
                windows=RESISTANCE_THRESHOLD_WINDOW_BOXPLOT_WINDOWS,
                max_samples=max_samples,
                output_dir=output_dir,
                save_plots=save_plots,
                plot_format=plot_format,
                plot_dpi=plot_dpi,
                baseline_samples=RESISTANCE_THRESHOLD_BASELINE_SAMPLES,
                threshold_search_samples=RESISTANCE_THRESHOLD_SEARCH_SAMPLES,
                atm_search_samples=ATM_WINDOW_SEARCH_SAMPLES,
                action_tolerance=RESISTANCE_THRESHOLD_WINDOW_ACTION_TOLERANCE,
                sample_ylim=RESISTANCE_THRESHOLD_WINDOW_SAMPLE_YLIM,
                action_ylim=RESISTANCE_THRESHOLD_WINDOW_ACTION_YLIM,
            )

        if PLOT_RESISTANCE_THRESHOLD_PEAK_RATE_BOXPLOTS:
            plot_resistance_threshold_peak_rate_boxplots(
                hdf,
                demos,
                action_path=ACTION_PATH,
                atm_label=ATM_STATE_LABEL,
                resistance_path=RESISTANCE_VALUE_PATH,
                resistance_channel=ATM_WINDOW_VALUE_CHANNEL,
                x_threshold=RESISTANCE_THRESHOLD_SCATTER_X_THRESHOLD,
                y_threshold=RESISTANCE_THRESHOLD_SCATTER_Y_THRESHOLD,
                max_samples=max_samples,
                output_dir=output_dir,
                save_plots=save_plots,
                plot_format=plot_format,
                plot_dpi=plot_dpi,
                baseline_samples=RESISTANCE_THRESHOLD_BASELINE_SAMPLES,
                threshold_search_samples=RESISTANCE_THRESHOLD_SEARCH_SAMPLES,
                atm_search_samples=ATM_WINDOW_SEARCH_SAMPLES,
                rate_ylim=RESISTANCE_THRESHOLD_PEAK_RATE_YLIM,
            )

        if PLOT_RESISTANCE_THRESHOLD_PEAK_RATE_SCATTER:
            plot_resistance_threshold_peak_rate_scatter(
                hdf,
                demos,
                action_path=ACTION_PATH,
                atm_label=ATM_STATE_LABEL,
                resistance_path=RESISTANCE_VALUE_PATH,
                resistance_channel=ATM_WINDOW_VALUE_CHANNEL,
                x_threshold=RESISTANCE_THRESHOLD_SCATTER_X_THRESHOLD,
                y_threshold=RESISTANCE_THRESHOLD_SCATTER_Y_THRESHOLD,
                max_samples=max_samples,
                output_dir=output_dir,
                save_plots=save_plots,
                plot_format=plot_format,
                plot_dpi=plot_dpi,
                baseline_samples=RESISTANCE_THRESHOLD_BASELINE_SAMPLES,
                threshold_search_samples=RESISTANCE_THRESHOLD_SEARCH_SAMPLES,
                atm_search_samples=ATM_WINDOW_SEARCH_SAMPLES,
            )

        if PLOT_RESISTANCE_THRESHOLD_BOXPLOTS:
            plot_resistance_threshold_time_boxplots(
                hdf,
                demos,
                action_path=ACTION_PATH,
                atm_label=ATM_STATE_LABEL,
                resistance_path=RESISTANCE_VALUE_PATH,
                resistance_channel=ATM_WINDOW_VALUE_CHANNEL,
                threshold_groups=RESISTANCE_THRESHOLD_GROUPS,
                max_samples=max_samples,
                output_dir=output_dir,
                save_plots=save_plots,
                plot_format=plot_format,
                plot_dpi=plot_dpi,
                baseline_samples=RESISTANCE_THRESHOLD_BASELINE_SAMPLES,
                threshold_search_samples=RESISTANCE_THRESHOLD_SEARCH_SAMPLES,
                atm_search_samples=ATM_WINDOW_SEARCH_SAMPLES,
                time_ylim=RESISTANCE_THRESHOLD_TIME_YLIM,
            )

        if PLOT_RESISTANCE_DERIVATIVE_TO_FIRST_THRESHOLD:
            plot_resistance_derivative_until_threshold(
                hdf,
                demos,
                action_path=ACTION_PATH,
                atm_label=ATM_STATE_LABEL,
                resistance_path=RESISTANCE_VALUE_PATH,
                resistance_channel=ATM_WINDOW_VALUE_CHANNEL,
                threshold=RESISTANCE_DERIVATIVE_THRESHOLD,
                max_samples=max_samples,
                output_dir=output_dir,
                save_plots=save_plots,
                plot_format=plot_format,
                plot_dpi=plot_dpi,
                baseline_samples=RESISTANCE_DERIVATIVE_BASELINE_SAMPLES,
                threshold_search_samples=RESISTANCE_DERIVATIVE_SEARCH_SAMPLES,
                atm_search_samples=ATM_WINDOW_SEARCH_SAMPLES,
                truncate_at_first_threshold=RESISTANCE_DERIVATIVE_TRUNCATE_AT_FIRST_THRESHOLD,
                value_ylim=RESISTANCE_DERIVATIVE_YLIM,
            )

        if PLOT_RESISTANCE_DERIVATIVE_NORMALIZED_TO_FIRST_THRESHOLD:
            plot_resistance_derivative_normalized_to_threshold(
                hdf,
                demos,
                action_path=ACTION_PATH,
                atm_label=ATM_STATE_LABEL,
                resistance_path=RESISTANCE_VALUE_PATH,
                resistance_channel=ATM_WINDOW_VALUE_CHANNEL,
                threshold=RESISTANCE_DERIVATIVE_THRESHOLD,
                max_samples=max_samples,
                output_dir=output_dir,
                save_plots=save_plots,
                plot_format=plot_format,
                plot_dpi=plot_dpi,
                baseline_samples=RESISTANCE_DERIVATIVE_BASELINE_SAMPLES,
                threshold_search_samples=RESISTANCE_DERIVATIVE_SEARCH_SAMPLES,
                atm_search_samples=ATM_WINDOW_SEARCH_SAMPLES,
                value_ylim=RESISTANCE_DERIVATIVE_NORMALIZED_YLIM,
                grid_points=RESISTANCE_DERIVATIVE_NORMALIZED_POINTS,
                confidence_z=RESISTANCE_DERIVATIVE_NORMALIZED_CONFIDENCE_Z,
            )

        if PLOT_RESISTANCE_INPUT_ACTION_RELATIONSHIPS:
            plot_resistance_input_action_relationships(
                hdf,
                demos,
                resistance_input_path=RESISTANCE_INPUT_PATH,
                action_path=ACTION_PATH,
                max_samples=max_samples,
                output_dir=output_dir,
                save_plots=save_plots,
                plot_format=plot_format,
                plot_dpi=plot_dpi,
                zero_padding_is_missing=RESISTANCE_INPUT_ZERO_PADDING_IS_MISSING,
                point_size=ACTION_RELATIONSHIP_POINT_SIZE,
                alpha=ACTION_RELATIONSHIP_ALPHA,
            )

        if PLOT_RESISTANCE_INPUT_DELTA_ACTION_RELATIONSHIPS:
            plot_resistance_input_action_relationships(
                hdf,
                demos,
                resistance_input_path=RESISTANCE_INPUT_PATH,
                action_path=ACTION_PATH,
                max_samples=max_samples,
                output_dir=output_dir,
                save_plots=save_plots,
                plot_format=plot_format,
                plot_dpi=plot_dpi,
                zero_padding_is_missing=RESISTANCE_INPUT_ZERO_PADDING_IS_MISSING,
                point_size=ACTION_RELATIONSHIP_POINT_SIZE,
                alpha=ACTION_RELATIONSHIP_ALPHA,
                use_action_delta=True,
            )

    if show_plots:
        plt.show()
    else:
        plt.close("all")


def main() -> int:
    file_path = FILE_PATH.resolve()
    output_dir = OUTPUT_DIR
    if output_dir is None and SAVE_PLOTS:
        output_dir = file_path.with_suffix("").parent / f"{file_path.stem}_gs_plots"

    analyze_gigaseal_dataset(
        file_path,
        max_demos=MAX_DEMOS,
        max_samples=MAX_SAMPLES,
        save_plots=SAVE_PLOTS,
        show_plots=SHOW_PLOTS,
        output_dir=output_dir,
        plot_format=PLOT_FORMAT,
        plot_dpi=PLOT_DPI,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
