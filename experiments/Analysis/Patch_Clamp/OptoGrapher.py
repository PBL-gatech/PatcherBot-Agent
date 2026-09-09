#!/usr/bin/env python3
"""
Plot optogenetic traces around each stim window and save WEBP plots.
Voltage is plotted in mV and current in pA.
One plot is saved per data/stim CSV pair.
"""

from pathlib import Path
import re
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import bessel, filtfilt

DEFAULT_FOLDER = Path(
    r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\experiments\Data\patch_clamp_data\2026_01_22-16_16\OptogeneticProtocol"
)
OUTPUT_NAME = "opto_plot.webp"
OUTPUT_PREFIX = "opto_plot"
DATA_PREFIX = "OptogeneticProtocol"
ACTIVE_STATE = "on"
PRE_STIM_S = 1.0
POST_STIM_S = 1.0
DOWN_SAMPLE: Optional[int] = None
CHUNK_ROWS = 1_000_000
BESSEL_CUTOFF_HZ = 3000.0
BESSEL_ORDER = 8
COLOR_ORDER = (
    ("uv", "ultraviolet"),
    ("violet", "purple"),
    ("blue",),
    ("cyan",),
    ("green",),
    ("yellow", "amber"),
    ("orange",),
    ("red",),
    ("infra", "ir"),
)
# Column mapping for this dataset: time, response, command.
TIME_COL = 0
COMMAND_COL = 2
RESPONSE_COL = 1


def detect_separator(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if "," in stripped:
                return ","
            if "\t" in stripped:
                return "\t"
            return "whitespace"
    return "whitespace"


def resolve_downsample(path: Path, requested: Optional[int]) -> int:
    if requested is not None:
        return max(1, int(requested))

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb <= 100:
        return 1
    return max(1, int(round(size_mb / 100)))


def apply_bessel_filter(
    time_s: np.ndarray,
    signal: np.ndarray,
    cutoff_hz: float,
    order: int,
) -> np.ndarray:
    if signal.size < order * 3:
        return signal

    diffs = np.diff(time_s)
    if diffs.size == 0:
        return signal
    dt = float(np.nanmedian(diffs))
    if not np.isfinite(dt) or dt <= 0:
        return signal

    fs = 1.0 / dt
    nyquist = 0.5 * fs
    if not np.isfinite(nyquist) or nyquist <= 0:
        return signal

    norm_cutoff = cutoff_hz / nyquist
    if not np.isfinite(norm_cutoff) or norm_cutoff >= 1.0 or norm_cutoff <= 0:
        return signal

    try:
        b, a = bessel(order, norm_cutoff, btype="low", norm="phase")
        return filtfilt(b, a, signal)
    except Exception:
        return signal


def load_wavelength_trace(
    path: Path,
    downsample: int,
    chunk_rows: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sep = detect_separator(path)
    read_kwargs = {
        "header": None,
        "usecols": sorted({TIME_COL, COMMAND_COL, RESPONSE_COL}),
        "chunksize": chunk_rows,
    }

    if sep == "whitespace":
        read_kwargs["delim_whitespace"] = True
    else:
        read_kwargs["sep"] = sep

    time_parts: List[np.ndarray] = []
    command_parts: List[np.ndarray] = []
    response_parts: List[np.ndarray] = []

    for chunk in pd.read_csv(path, **read_kwargs):
        chunk = chunk.apply(pd.to_numeric, errors="coerce").dropna()
        if chunk.empty:
            continue
        if downsample > 1:
            chunk = chunk.iloc[::downsample]
        time_parts.append(chunk.iloc[:, TIME_COL].to_numpy())
        command_parts.append(chunk.iloc[:, COMMAND_COL].to_numpy())
        response_parts.append(chunk.iloc[:, RESPONSE_COL].to_numpy())

    if not time_parts:
        raise ValueError(f"No numeric data found in {path}")

    return (
        np.concatenate(time_parts),
        np.concatenate(command_parts),
        np.concatenate(response_parts),
    )


def load_stim_windows(
    path: Path, active_state: str
) -> List[Tuple[float, float, str, Optional[float]]]:
    df = pd.read_csv(path)
    if df.empty:
        return []

    columns = {col.lower(): col for col in df.columns}
    start_col = columns.get("start_s") or columns.get("start") or df.columns[0]
    end_col = columns.get("end_s") or columns.get("end") or df.columns[1]
    state_col = columns.get("state") or (df.columns[2] if len(df.columns) > 2 else None)
    wavelength_col = columns.get("wavelength") or (df.columns[3] if len(df.columns) > 3 else None)
    power_col = columns.get("power_percent") or columns.get("power")

    starts = pd.to_numeric(df[start_col], errors="coerce")
    ends = pd.to_numeric(df[end_col], errors="coerce")
    if state_col:
        states = df[state_col].astype(str).str.lower()
    else:
        states = pd.Series(["on"] * len(df), index=df.index)
    if wavelength_col:
        wavelengths = df[wavelength_col].astype(str).str.lower()
    else:
        wavelengths = pd.Series(["stim"] * len(df), index=df.index)
    if power_col:
        powers = pd.to_numeric(df[power_col], errors="coerce")
    else:
        powers = pd.Series([np.nan] * len(df), index=df.index)

    keep = states == active_state.lower()
    windows_df = pd.DataFrame(
        {"start": starts, "end": ends, "wavelength": wavelengths, "power": powers}
    ).loc[keep]
    windows_df = windows_df.dropna(subset=["start", "end"])

    windows: List[Tuple[float, float, str, Optional[float]]] = []
    for row in windows_df.itertuples(index=False):
        start = float(row.start)
        end = float(row.end)
        if end < start:
            start, end = end, start
        power = None if pd.isna(row.power) else float(row.power)
        windows.append((start, end, str(row.wavelength), power))

    return windows


def wavelength_color(name: str) -> str:
    name = name.lower()
    if "uv" in name or "ultraviolet" in name:
        return "#9467bd"
    if "red" in name:
        return "#d62728"
    if "infra" in name or "ir" in name:
        return "#ff7f0e"
    if "blue" in name:
        return "#1f77b4"
    if "green" in name:
        return "#2ca02c"
    if "cyan" in name:
        return "#7ec9f1"
    return "#7f7f7f"


def is_data_csv(path: Path) -> bool:
    name = path.name.lower()
    return (
        name.endswith(".csv")
        and name.startswith(DATA_PREFIX.lower())
        and not name.endswith("_stim.csv")
    )


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip())
    return cleaned.strip("_") or "protocol"


def wavelength_sort_key(name: str) -> int:
    name = name.lower()
    for idx, tokens in enumerate(COLOR_ORDER):
        if any(token in name for token in tokens):
            return idx
    return len(COLOR_ORDER)


Segment = Tuple[Path, np.ndarray, np.ndarray, np.ndarray, float, str, Optional[float]]


def segment_trace(
    path: Path,
    time_s: np.ndarray,
    command_v: np.ndarray,
    response_a: np.ndarray,
    stim_windows: List[Tuple[float, float, str, Optional[float]]],
    apply_filter: bool,
    cutoff_hz: float,
    order: int,
) -> List[Segment]:
    segments: List[Segment] = []
    for start, end, wavelength, power in stim_windows:
        window_start = start - PRE_STIM_S
        window_end = end + POST_STIM_S
        mask = (time_s >= window_start) & (time_s <= window_end)
        if not np.any(mask):
            continue
        time_rel = time_s[mask] - start
        if apply_filter:
            cmd_segment = apply_bessel_filter(
                time_s[mask], command_v[mask], cutoff_hz, order
            )
            resp_segment = apply_bessel_filter(
                time_s[mask], response_a[mask], cutoff_hz, order
            )
        else:
            cmd_segment = command_v[mask]
            resp_segment = response_a[mask]
        baseline_mask = time_rel < 0
        if np.any(baseline_mask):
            cmd_baseline = float(np.nanmean(cmd_segment[baseline_mask]))
            resp_baseline = float(np.nanmean(resp_segment[baseline_mask]))
        else:
            cmd_baseline = float(np.nanmean(cmd_segment[:1]))
            resp_baseline = float(np.nanmean(resp_segment[:1]))

        segments.append(
            (
                path,
                time_rel,
                cmd_segment - cmd_baseline,
                resp_segment - resp_baseline,
                end - start,
                wavelength,
                power,
            )
        )
    return segments


def gather_trace_pairs(
    folder: Path,
    apply_filter: bool,
    cutoff_hz: float,
    order: int,
    order_by_color: bool,
) -> List[Tuple[Path, List[Segment]]]:
    data_files = sorted(path for path in folder.glob("*.csv") if is_data_csv(path))
    if not data_files:
        raise FileNotFoundError(f"No data CSV files found in {folder}")

    pairs: List[Tuple[Path, List[Segment]]] = []
    for data_path in data_files:
        downsample = resolve_downsample(data_path, DOWN_SAMPLE)
        if DOWN_SAMPLE is None and downsample > 1:
            size_mb = data_path.stat().st_size / (1024 * 1024)
            print(
                f"Auto downsample={downsample} for {data_path.name} ({size_mb:.1f} MB). "
                "Set DOWN_SAMPLE to override."
            )

        time_s, command_v, response = load_wavelength_trace(
            data_path, downsample=downsample, chunk_rows=CHUNK_ROWS
        )

        stim_path = data_path.with_name(f"{data_path.stem}_stim.csv")
        stim_windows = load_stim_windows(stim_path, ACTIVE_STATE) if stim_path.exists() else []
        if not stim_windows:
            continue

        segments = segment_trace(
            data_path,
            time_s,
            command_v,
            response,
            stim_windows,
            apply_filter,
            cutoff_hz,
            order,
        )
        if not segments:
            continue
        has_power = any(seg[6] is not None for seg in segments)
        if has_power:
            segments = sorted(
                segments,
                key=lambda seg: (float("inf") if seg[6] is None else seg[6]),
            )
        elif order_by_color:
            segments = sorted(segments, key=lambda seg: wavelength_sort_key(seg[5]))
        pairs.append((data_path, segments))

    return pairs


def plot_opto(
    traces: List[Segment],
    title: Optional[str],
) -> Tuple[plt.Figure, plt.Axes]:
    if not traces:
        raise ValueError("No traces to plot.")

    trace_count = len(traces)
    fig_height = max(6.5, trace_count * 2.6)
    fig, axes = plt.subplots(
        trace_count,
        2,
        sharex=True,
        figsize=(11, fig_height),
        gridspec_kw={"width_ratios": [1, 1]},
    )
    axes = np.atleast_2d(axes)
    xmin = None
    xmax = None

    for _, time_s, _, _, _, _, _ in traces:
        if time_s.size:
            local_min = float(np.nanmin(time_s))
            local_max = float(np.nanmax(time_s))
            xmin = local_min if xmin is None else min(xmin, local_min)
            xmax = local_max if xmax is None else max(xmax, local_max)

    for idx, (_, time_s, command_v, response, duration_s, wavelength, power) in enumerate(traces):
        ax_cmd = axes[idx, 0]
        ax_resp = axes[idx, 1]
        stim_color = wavelength_color(wavelength)

        ax_cmd.plot(time_s, command_v * 1e3, color=stim_color, linewidth=1.0)
        ax_resp.plot(time_s, response * 1e12, color=stim_color, linewidth=1.0)

        for axis in (ax_cmd, ax_resp):
            axis.axvspan(0.0, duration_s, color=stim_color, alpha=0.18, linewidth=0)
            axis.axvline(0.0, color=stim_color, alpha=0.6, linewidth=1.0)
            axis.axvline(duration_s, color=stim_color, alpha=0.6, linewidth=1.0)

        label = f"Stim {wavelength}"
        if power is not None:
            label = f"{label} ({power:g}%)"
        if duration_s > 0:
            label = f"{label} ({duration_s:.4g} s)"
        ax_cmd.set_title(label, fontsize=9)
        ax_cmd.set_ylabel("Command voltage (mV)")
        ax_resp.set_ylabel("Response current (pA)")

        if idx < trace_count - 1:
            ax_cmd.tick_params(labelbottom=False)
            ax_resp.tick_params(labelbottom=False)

    if xmin is not None and xmax is not None:
        for row in axes:
            for axis in row:
                axis.set_xlim(xmin, xmax)

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    if title:
        fig.suptitle(title, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.985])
    else:
        fig.tight_layout()
    return fig, axes[0, 0]


def find_protocol_folders(root: Path) -> List[Path]:
    folders = {path.parent for path in root.rglob("*.csv") if is_data_csv(path)}
    return sorted(folders)


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path = output_path.with_suffix(".webp")
    try:
        fig.savefig(output_path, dpi=300, bbox_inches="tight", format="webp")
    except Exception as exc:
        raise RuntimeError(
            "WEBP save failed. Install Pillow or change OUTPUT_NAME to a supported format."
        ) from exc
    print(f"Saved: {output_path}")


def process_folder(
    folder: Path,
    apply_filter: bool,
    cutoff_hz: float,
    order: int,
    order_by_color: bool,
) -> None:
    trace_pairs = gather_trace_pairs(
        folder, apply_filter, cutoff_hz, order, order_by_color
    )
    multi_pair = len(trace_pairs) > 1

    for data_path, traces in trace_pairs:
        title = data_path.stem if multi_pair else None
        fig, _ = plot_opto(traces, title=title)
        if multi_pair:
            output_name = f"{OUTPUT_PREFIX}_{sanitize_filename(data_path.stem)}.webp"
        else:
            output_name = OUTPUT_NAME
        output_path = folder / output_name
        save_figure(fig, output_path)
        plt.close(fig)


def main(
    folder: Path,
    apply_filter: bool,
    cutoff_hz: float,
    order: int,
    order_by_color: bool,
) -> None:
    if not folder.is_dir():
        raise NotADirectoryError(f"{folder} is not a folder.")

    protocol_folders = find_protocol_folders(folder)
    if not protocol_folders:
        raise FileNotFoundError(f"No OptogeneticProtocol CSVs found in {folder}")

    for protocol_folder in protocol_folders:
        process_folder(protocol_folder, apply_filter, cutoff_hz, order, order_by_color)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot optogenetic traces around each stim window."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=DEFAULT_FOLDER,
        type=Path,
        help="Root folder to search for OptogeneticProtocol CSVs.",
    )
    parser.add_argument(
        "--cutoff-hz",
        type=float,
        default=BESSEL_CUTOFF_HZ,
        help="Bessel low-pass cutoff frequency in Hz.",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=BESSEL_ORDER,
        help="Bessel filter order.",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable the Bessel low-pass filter.",
    )
    parser.add_argument(
        "--color-order",
        action="store_true",
        help="Order stim windows by wavelength color gate instead of stim file order.",
    )
    args = parser.parse_args()

    apply_filter = not args.no_filter
    if apply_filter:
        print(f"Filtering: {args.cutoff_hz} Hz, order={args.order} (Bessel, filtfilt)")
    else:
        print("Filtering disabled.")
    if args.color_order:
        print("Ordering traces by color gate.")

    main(args.folder, apply_filter, args.cutoff_hz, args.order, args.color_order)
