# -*- coding: utf-8 -*-
"""
Analyze resistance-trace noise as baseline resistance grows.

This standalone Rig Recorder reconstruction script reads resistance traces from
each /data/demo_* group in a PatcherBot HDF5 dataset, detrends each trace with a
Savitzky-Golay filter, then bins absolute and percentage-normalized residual
amplitudes by the SG trend baseline resistance.

Edit the configuration constants below, then run:
    py -3 experiments/Analysis/Rig_Recorder/Reconstruct/resistanceAnalyzer.py
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
FILE_PATH = Path(
    r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent"
    r"\experiments\Datasets\PatcherBot_dataset_v0_910"
    r"\PatcherBot_dataset_v0_910_gigaseal.hdf5"
)

RESISTANCE_PATH = "obs/resistance"

MAX_DEMOS: int | None = None
MAX_SAMPLES: int | None = None

SG_WINDOW_LENGTH = 101
SG_POLYORDER = 3

BIN_EDGES = (10.0, 25.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0 , 600.0, 700.0, 800.0, 900.0, 1000.0, 1100.0, np.inf)

PLOT_BIN_DISTRIBUTIONS = True
PLOT_PERCENT_DISTRIBUTIONS = True
PLOT_DIAGNOSTIC_TRACES = True
MAX_DIAGNOSTIC_DEMOS = 4

SHOW_PLOTS = True
SAVE_PLOTS = False
OUTPUT_DIR: Path | None = None
PLOT_DPI = 200
PLOT_FORMAT = "png"


@dataclass(slots=True)
class DemoResidualAnalysis:
    demo: str
    raw: np.ndarray
    trend: np.ndarray
    residual: np.ndarray
    amplitude: np.ndarray
    percent_amplitude: np.ndarray
    finite_mask: np.ndarray
    sg_window: int


@dataclass(slots=True)
class BinStats:
    label: str
    left: float
    right: float
    count: int
    mean: float
    median: float
    rms: float
    percentile95: float


def _natural_key(text: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _slug(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip()).strip("_").lower()
    return cleaned or "plot"


def _find_demos(hdf: h5py.File, max_demos: int | None = None) -> list[str]:
    if "data" not in hdf:
        raise KeyError(f"{hdf.filename} does not contain a /data group")
    demos = sorted(hdf["data"].keys(), key=_natural_key)
    if max_demos is not None:
        demos = demos[:max(0, max_demos)]
    return demos


def _as_resistance_trace(arr: np.ndarray) -> np.ndarray | None:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        trace = arr
    elif arr.ndim == 2 and 1 in arr.shape:
        trace = arr.reshape(-1)
    else:
        return None
    return trace.astype(np.float64, copy=False)


def _load_resistance_trace(
    demo_group: h5py.Group,
    resistance_path: str,
    max_samples: int | None,
) -> np.ndarray | None:
    if resistance_path not in demo_group:
        return None
    obj = demo_group[resistance_path]
    if not isinstance(obj, h5py.Dataset):
        return None
    trace = _as_resistance_trace(np.asarray(obj[()]))
    if trace is None:
        return None
    if max_samples is not None:
        trace = trace[:max(0, max_samples)]
    return trace


def _validate_bin_edges(bin_edges: Iterable[float]) -> np.ndarray:
    edges = np.asarray(tuple(bin_edges), dtype=np.float64)
    if edges.ndim != 1 or edges.shape[0] < 2:
        raise ValueError("BIN_EDGES must contain at least two numeric edges")
    if not np.all(np.diff(edges) > 0):
        raise ValueError("BIN_EDGES must be strictly increasing")
    return edges


def _format_edge(edge: float) -> str:
    if np.isneginf(edge):
        return "-inf"
    if np.isposinf(edge):
        return "inf"
    return f"{edge:g}"


def _bin_labels(bin_edges: Sequence[float]) -> list[str]:
    return [
        f"[{_format_edge(left)}, {_format_edge(right)})"
        for left, right in zip(bin_edges[:-1], bin_edges[1:])
    ]


def _sanitize_savgol_window(sample_count: int, requested_window: int, polyorder: int) -> int | None:
    if sample_count <= polyorder or polyorder < 0:
        return None

    window = max(1, int(requested_window))
    if window % 2 == 0:
        window += 1

    min_window = polyorder + 2
    if min_window % 2 == 0:
        min_window += 1
    window = max(window, min_window)

    max_window = sample_count if sample_count % 2 == 1 else sample_count - 1
    if max_window <= polyorder:
        return None

    window = min(window, max_window)
    if window % 2 == 0:
        window -= 1
    if window <= polyorder:
        return None
    return window


def _interpolate_nonfinite(values: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return None
    if np.all(finite_mask):
        return values.astype(np.float64, copy=True), finite_mask

    x = np.arange(values.shape[0], dtype=np.float64)
    filled = np.interp(x, x[finite_mask], values[finite_mask])
    return filled.astype(np.float64, copy=False), finite_mask


def analyze_resistance_trace(
    demo: str,
    raw: np.ndarray,
    *,
    sg_window_length: int,
    sg_polyorder: int,
) -> DemoResidualAnalysis | None:
    raw = np.asarray(raw, dtype=np.float64).reshape(-1)
    if raw.shape[0] == 0:
        return None

    interpolated = _interpolate_nonfinite(raw)
    if interpolated is None:
        return None
    filled, finite_mask = interpolated

    finite_count = int(np.count_nonzero(finite_mask))
    if finite_count <= sg_polyorder:
        return None

    sg_window = _sanitize_savgol_window(raw.shape[0], sg_window_length, sg_polyorder)
    if sg_window is None:
        return None

    trend = savgol_filter(filled, window_length=sg_window, polyorder=sg_polyorder, mode="interp")
    residual = raw - trend
    residual[~finite_mask] = np.nan
    amplitude = np.abs(residual)
    amplitude[~finite_mask] = np.nan
    percent_amplitude = np.full(raw.shape, np.nan, dtype=np.float64)
    percent_mask = finite_mask & np.isfinite(amplitude) & np.isfinite(trend) & (trend > 0.0)
    percent_amplitude[percent_mask] = (amplitude[percent_mask] / trend[percent_mask]) * 100.0

    return DemoResidualAnalysis(
        demo=demo,
        raw=raw,
        trend=trend,
        residual=residual,
        amplitude=amplitude,
        percent_amplitude=percent_amplitude,
        finite_mask=finite_mask,
        sg_window=sg_window,
    )


def _bin_analysis_values(
    analyses: Sequence[DemoResidualAnalysis],
    bin_edges: Sequence[float],
    *,
    value_name: str,
) -> list[np.ndarray]:
    edges = np.asarray(bin_edges, dtype=np.float64)
    per_bin_parts: list[list[np.ndarray]] = [[] for _ in range(edges.shape[0] - 1)]

    for analysis in analyses:
        values = getattr(analysis, value_name)
        valid = np.isfinite(analysis.trend) & np.isfinite(values)
        if not np.any(valid):
            continue

        bin_resistance = analysis.trend[valid]
        values = values[valid]
        bin_indices = np.searchsorted(edges, bin_resistance, side="right") - 1
        in_range = (bin_indices >= 0) & (bin_indices < edges.shape[0] - 1)

        for bin_idx in range(edges.shape[0] - 1):
            samples = values[in_range & (bin_indices == bin_idx)]
            if samples.size:
                per_bin_parts[bin_idx].append(samples)

    return [
        np.concatenate(parts) if parts else np.asarray([], dtype=np.float64)
        for parts in per_bin_parts
    ]


def bin_residual_amplitudes(
    analyses: Sequence[DemoResidualAnalysis],
    bin_edges: Sequence[float],
) -> list[np.ndarray]:
    return _bin_analysis_values(analyses, bin_edges, value_name="amplitude")


def bin_residual_percentages(
    analyses: Sequence[DemoResidualAnalysis],
    bin_edges: Sequence[float],
) -> list[np.ndarray]:
    return _bin_analysis_values(analyses, bin_edges, value_name="percent_amplitude")


def compute_bin_stats(bin_amplitudes: Sequence[np.ndarray], bin_edges: Sequence[float]) -> list[BinStats]:
    labels = _bin_labels(bin_edges)
    stats: list[BinStats] = []
    for label, left, right, values in zip(labels, bin_edges[:-1], bin_edges[1:], bin_amplitudes):
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            stats.append(
                BinStats(
                    label=label,
                    left=float(left),
                    right=float(right),
                    count=0,
                    mean=np.nan,
                    median=np.nan,
                    rms=np.nan,
                    percentile95=np.nan,
                )
            )
            continue

        stats.append(
            BinStats(
                label=label,
                left=float(left),
                right=float(right),
                count=int(values.size),
                mean=float(np.mean(values)),
                median=float(np.median(values)),
                rms=float(math.sqrt(np.mean(values * values))),
                percentile95=float(np.percentile(values, 95)),
            )
        )
    return stats


def print_bin_summary(stats: Sequence[BinStats], title: str) -> None:
    print(f"\n{title}")
    print("bin\tcount\tmean\tmedian\trms\tp95")
    for stat in stats:
        if stat.count == 0:
            print(f"{stat.label}\t0\tNA\tNA\tNA\tNA")
            continue
        print(
            f"{stat.label}\t{stat.count}\t"
            f"{stat.mean:.6g}\t{stat.median:.6g}\t{stat.rms:.6g}\t{stat.percentile95:.6g}"
        )


def _save_figure(
    fig: plt.Figure,
    output_dir: Path | None,
    stem: str,
    *,
    save_plots: bool,
    plot_format: str,
    plot_dpi: int,
) -> None:
    if not save_plots or output_dir is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{_slug(stem)}.{plot_format}", dpi=plot_dpi, bbox_inches="tight")


def plot_bin_distributions(
    bin_amplitudes: Sequence[np.ndarray],
    stats: Sequence[BinStats],
    *,
    title: str,
    ylabel: str,
    median_label: str,
    rms_label: str,
    output_stem: str,
    output_dir: Path | None,
    save_plots: bool,
    plot_format: str,
    plot_dpi: int,
) -> None:
    labels = [stat.label for stat in stats]
    positions = np.arange(1, len(labels) + 1)
    nonempty_positions = [pos for pos, values in zip(positions, bin_amplitudes) if values.size]
    nonempty_values = [values for values in bin_amplitudes if values.size]

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.1), 6))
    if nonempty_values:
        box = ax.boxplot(
            nonempty_values,
            positions=nonempty_positions,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
        )
        for patch in box["boxes"]:
            patch.set_facecolor("#8ecae6")
            patch.set_alpha(0.75)
        for median in box["medians"]:
            median.set_color("#023047")
            median.set_linewidth(1.6)

    medians = np.asarray([stat.median for stat in stats], dtype=np.float64)
    rms_values = np.asarray([stat.rms for stat in stats], dtype=np.float64)
    ax.plot(positions, medians, color="#1d3557", marker="o", linewidth=1.8, label=median_label)
    ax.plot(positions, rms_values, color="#d62828", marker="s", linewidth=1.8, label=rms_label)

    ax.set_title(title)
    ax.set_xlabel("SG trend resistance bin")
    ax.set_ylabel(ylabel)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()

    _save_figure(
        fig,
        output_dir,
        output_stem,
        save_plots=save_plots,
        plot_format=plot_format,
        plot_dpi=plot_dpi,
    )


def plot_diagnostic_traces(
    analyses: Sequence[DemoResidualAnalysis],
    *,
    max_demos: int,
    output_dir: Path | None,
    save_plots: bool,
    plot_format: str,
    plot_dpi: int,
) -> None:
    selected = list(analyses[:max(0, max_demos)])
    if not selected:
        return

    fig, axes = plt.subplots(len(selected), 2, figsize=(14, 3.4 * len(selected)), squeeze=False)
    for row_idx, analysis in enumerate(selected):
        x = np.arange(analysis.raw.shape[0])
        left_ax = axes[row_idx, 0]
        right_ax = axes[row_idx, 1]

        left_ax.plot(x, analysis.raw, color="#264653", linewidth=1.1, label="Raw resistance")
        left_ax.plot(
            x,
            analysis.trend,
            color="#e76f51",
            linewidth=1.2,
            label=f"SG trend, window={analysis.sg_window}",
        )
        left_ax.set_title(f"{analysis.demo}: raw resistance and SG trend")
        left_ax.set_xlabel("Sample")
        left_ax.set_ylabel("Resistance")
        left_ax.grid(True, alpha=0.25)
        left_ax.legend(loc="best", fontsize=8)

        right_ax.plot(x, analysis.residual, color="#2a9d8f", linewidth=1.0)
        right_ax.axhline(0.0, color="0.35", linestyle=":", linewidth=1.0)
        right_ax.set_title(f"{analysis.demo}: residual")
        right_ax.set_xlabel("Sample")
        right_ax.set_ylabel("Residual")
        right_ax.grid(True, alpha=0.25)

    fig.tight_layout()
    _save_figure(
        fig,
        output_dir,
        "resistance_sg_diagnostic_traces",
        save_plots=save_plots,
        plot_format=plot_format,
        plot_dpi=plot_dpi,
    )


def analyze_hdf5_resistance(
    file_path: Path,
    *,
    resistance_path: str,
    max_demos: int | None,
    max_samples: int | None,
    sg_window_length: int,
    sg_polyorder: int,
    bin_edges: Iterable[float],
    plot_bin_distributions_enabled: bool,
    plot_percent_distributions_enabled: bool,
    plot_diagnostic_traces_enabled: bool,
    max_diagnostic_demos: int,
    save_plots: bool,
    show_plots: bool,
    output_dir: Path | None,
    plot_format: str,
    plot_dpi: int,
) -> tuple[
    list[DemoResidualAnalysis],
    list[np.ndarray],
    list[BinStats],
    list[np.ndarray],
    list[BinStats],
]:
    file_path = Path(file_path).resolve()
    edges = _validate_bin_edges(bin_edges)
    if output_dir is None and save_plots:
        output_dir = file_path.with_suffix("").parent / f"{file_path.stem}_resistance_noise"

    analyses: list[DemoResidualAnalysis] = []
    skipped: list[str] = []

    with h5py.File(file_path, "r") as hdf:
        demos = _find_demos(hdf, max_demos=max_demos)
        if not demos:
            raise RuntimeError(f"No demos found in {file_path}")

        for demo in demos:
            demo_group = hdf["data"][demo]
            raw = _load_resistance_trace(demo_group, resistance_path, max_samples)
            if raw is None:
                skipped.append(f"{demo}: missing or unsupported {resistance_path}")
                continue

            analysis = analyze_resistance_trace(
                demo,
                raw,
                sg_window_length=sg_window_length,
                sg_polyorder=sg_polyorder,
            )
            if analysis is None:
                skipped.append(f"{demo}: too short or invalid for SG detrending")
                continue
            analyses.append(analysis)

    if not analyses:
        raise RuntimeError(f"No usable resistance traces found in {file_path}")

    print(f"Loaded {len(analyses)} usable resistance trace(s) from {file_path}")
    if skipped:
        print(f"Skipped {len(skipped)} demo(s):")
        for reason in skipped:
            print(f"  {reason}")

    bin_amplitudes = bin_residual_amplitudes(analyses, edges)
    stats = compute_bin_stats(bin_amplitudes, edges)
    print_bin_summary(stats, "Residual amplitude by SG trend resistance bin")

    percent_amplitudes = bin_residual_percentages(analyses, edges)
    percent_stats = compute_bin_stats(percent_amplitudes, edges)
    print_bin_summary(percent_stats, "Residual amplitude as percent of SG trend resistance by SG trend bin")

    if plot_bin_distributions_enabled:
        plot_bin_distributions(
            bin_amplitudes,
            stats,
            title="Resistance Noise Residual Amplitude by SG Trend Resistance",
            ylabel="|raw resistance - SG trend|",
            median_label="Median amplitude",
            rms_label="RMS amplitude",
            output_stem="resistance_noise_by_sg_trend_resistance_bin",
            output_dir=output_dir,
            save_plots=save_plots,
            plot_format=plot_format,
            plot_dpi=plot_dpi,
        )

    if plot_percent_distributions_enabled:
        plot_bin_distributions(
            percent_amplitudes,
            percent_stats,
            title="Resistance Noise Residual Percent by SG Trend Resistance",
            ylabel="|raw resistance - SG trend| / SG trend (%)",
            median_label="Median percent",
            rms_label="RMS percent",
            output_stem="resistance_noise_percent_by_sg_trend_resistance_bin",
            output_dir=output_dir,
            save_plots=save_plots,
            plot_format=plot_format,
            plot_dpi=plot_dpi,
        )

    if plot_diagnostic_traces_enabled:
        plot_diagnostic_traces(
            analyses,
            max_demos=max_diagnostic_demos,
            output_dir=output_dir,
            save_plots=save_plots,
            plot_format=plot_format,
            plot_dpi=plot_dpi,
        )

    if show_plots:
        plt.show()
    else:
        plt.close("all")

    return analyses, bin_amplitudes, stats, percent_amplitudes, percent_stats


def main() -> int:
    analyze_hdf5_resistance(
        FILE_PATH,
        resistance_path=RESISTANCE_PATH,
        max_demos=MAX_DEMOS,
        max_samples=MAX_SAMPLES,
        sg_window_length=SG_WINDOW_LENGTH,
        sg_polyorder=SG_POLYORDER,
        bin_edges=BIN_EDGES,
        plot_bin_distributions_enabled=PLOT_BIN_DISTRIBUTIONS,
        plot_percent_distributions_enabled=PLOT_PERCENT_DISTRIBUTIONS,
        plot_diagnostic_traces_enabled=PLOT_DIAGNOSTIC_TRACES,
        max_diagnostic_demos=MAX_DIAGNOSTIC_DEMOS,
        save_plots=SAVE_PLOTS,
        show_plots=SHOW_PLOTS,
        output_dir=OUTPUT_DIR,
        plot_format=PLOT_FORMAT,
        plot_dpi=PLOT_DPI,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
