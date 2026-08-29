# -*- coding: utf-8 -*-
"""
Generic PatcherBot HDF5 dataset plotter.

This is the non-movement-specific companion to trajectoryPlotter.py. It walks
each /data/demo_* group, finds numeric 1-D and 2-D time-series datasets, and
plots every scalar/vector channel it can sensibly display. Image tensors and
other high-dimensional datasets are skipped.

Examples
--------
Edit the configuration constants below, then run:
    py -3 experiments/Analysis/Rig_Recorder/Reconstruct/datasetPlotter.py
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
FILE_PATH = Path(
    r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent"
    r"\experiments\Datasets\PatcherBot_dataset_v0_966"
    r"\PatcherBot_dataset_v0_966_hunt_cell.hdf5"
)

ROOTS_TO_PLOT = ("obs", "actions", "dones")
INCLUDE_NEXT_OBS = False

MAX_DEMOS: int | None = None
MAX_SAMPLES: int | None = None
MAX_VECTOR_CHANNELS = 12
MAX_3D_POINTS_PER_DEMO = 2500
LEGEND_MAX_DEMOS = 12

PLOT_3D_POSITION_DATA = True
SHOW_PLOTS = True
SAVE_PLOTS = False
OUTPUT_DIR: Path | None = None
PLOT_DPI = 200
PLOT_FORMAT = "png"
COLORMAP_NAME = "twilight"

# Set to a number if you want a guide line on matching plots.
RESISTANCE_REFERENCE_LINE: float | None = None


def _natural_key(text: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _slug(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip()).strip("_").lower()
    return cleaned or "plot"


def _pretty_path(path: str) -> str:
    return path.replace("_", " ").replace("/", " / ")


def _decode_attr_values(value: object) -> list[str]:
    if value is None:
        return []
    arr = np.asarray(value).reshape(-1)
    labels: list[str] = []
    for item in arr:
        if isinstance(item, bytes):
            labels.append(item.decode("utf-8", errors="replace"))
        else:
            labels.append(str(item))
    return labels


def _find_demos(hdf: h5py.File, max_demos: int | None = None) -> list[str]:
    if "data" not in hdf:
        raise KeyError(f"{hdf.filename} does not contain a /data group")
    demos = sorted(hdf["data"].keys(), key=_natural_key)
    if max_demos is not None:
        demos = demos[:max(0, max_demos)]
    return demos


def _path_root(path: str) -> str:
    return path.split("/", 1)[0]


def _is_numeric_timeseries(ds: h5py.Dataset, max_vector_channels: int) -> bool:
    if not np.issubdtype(ds.dtype, np.number):
        return False
    if ds.ndim not in (1, 2):
        return False
    if not ds.shape or ds.shape[0] == 0:
        return False
    if ds.ndim == 2 and (ds.shape[1] == 0 or ds.shape[1] > max_vector_channels):
        return False
    return True


def _dataset_width(ds: h5py.Dataset) -> int:
    if ds.ndim == 1:
        return 1
    return int(ds.shape[1])


def _as_2d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def _load_2d(ds: h5py.Dataset, max_samples: int | None) -> np.ndarray:
    arr = np.asarray(ds[()])
    arr = _as_2d(arr)
    if max_samples is not None:
        arr = arr[:max(0, max_samples)]
    return arr.astype(float, copy=False)


def _channel_labels(path: str, ds: h5py.Dataset) -> list[str]:
    width = _dataset_width(ds)
    axes = _decode_attr_values(ds.attrs.get("axes"))
    if len(axes) == width:
        return axes
    if width == 1:
        return [path.rsplit("/", 1)[-1]]
    return [f"ch{i}" for i in range(width)]


def _should_include_path(path: str, roots: set[str]) -> bool:
    root = _path_root(path)
    return root in roots


def discover_timeseries_paths(
    hdf: h5py.File,
    demos: Sequence[str],
    roots: Iterable[str],
    max_vector_channels: int,
) -> dict[str, int]:
    roots_set = set(roots)
    paths: dict[str, int] = {}

    for demo in demos:
        demo_group = hdf["data"][demo]

        def visitor(path: str, obj: h5py.Dataset | h5py.Group) -> None:
            if not isinstance(obj, h5py.Dataset):
                return
            if not _should_include_path(path, roots_set):
                return
            if not _is_numeric_timeseries(obj, max_vector_channels):
                return
            paths[path] = max(paths.get(path, 0), _dataset_width(obj))

        demo_group.visititems(visitor)

    return dict(sorted(paths.items(), key=lambda item: _natural_key(item[0])))


def _first_dataset_for_path(
    hdf: h5py.File,
    demos: Sequence[str],
    path: str,
) -> h5py.Dataset | None:
    for demo in demos:
        demo_group = hdf["data"][demo]
        if path in demo_group:
            obj = demo_group[path]
            if isinstance(obj, h5py.Dataset):
                return obj
    return None


def _plot_color(index: int, total: int):
    cmap = plt.get_cmap(COLORMAP_NAME)
    if total <= 1:
        return cmap(0.0)
    return cmap(index / (total - 1))


def plot_time_series_dataset(
    hdf: h5py.File,
    demos: Sequence[str],
    path: str,
    width: int,
    *,
    max_samples: int | None,
    output_dir: Path | None,
    save_plots: bool,
    plot_format: str,
    plot_dpi: int,
    resistance_reference_line: float | None,
) -> None:
    sample_ds = _first_dataset_for_path(hdf, demos, path)
    if sample_ds is None:
        return

    labels = _channel_labels(path, sample_ds)
    ncols = 1 if width == 1 else 2
    nrows = math.ceil(width / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 3.4 * nrows), squeeze=False)
    axes_flat = list(axes.reshape(-1))

    for channel_idx in range(width):
        ax = axes_flat[channel_idx]
        label = labels[channel_idx] if channel_idx < len(labels) else f"ch{channel_idx}"
        for demo_idx, demo in enumerate(demos):
            demo_group = hdf["data"][demo]
            if path not in demo_group:
                continue
            arr = _load_2d(demo_group[path], max_samples)
            if channel_idx >= arr.shape[1]:
                continue
            y = arr[:, channel_idx]
            y = np.where(np.isfinite(y), y, np.nan)
            x = np.arange(y.shape[0])
            ax.plot(
                x,
                y,
                color=_plot_color(demo_idx, len(demos)),
                alpha=0.82,
                linewidth=1.4,
                label=demo if len(demos) <= LEGEND_MAX_DEMOS else None,
            )

        if (
            resistance_reference_line is not None
            and "resistance" in path.lower()
            and "slope" not in path.lower()
            and "log" not in path.lower()
        ):
            ax.axhline(resistance_reference_line, color="0.35", linestyle=":", linewidth=1.5)

        ax.set_title(label)
        ax.set_xlabel("Sample")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
        if len(demos) <= LEGEND_MAX_DEMOS:
            ax.legend(loc="best", fontsize=8)

    for ax in axes_flat[width:]:
        ax.set_visible(False)

    fig.suptitle(_pretty_path(path), fontsize=14)
    fig.tight_layout()

    if save_plots and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_dir / f"{_slug(path)}.{plot_format}",
            dpi=plot_dpi,
            bbox_inches="tight",
        )


def _is_position_path(path: str, width: int) -> bool:
    if width < 3:
        return False
    low = path.lower()
    leaf = low.rsplit("/", 1)[-1]
    return (
        leaf.endswith("_positions")
        or leaf.endswith("_position")
        or "trajectory" in leaf
        or "position" in leaf
    )


def _downsample_for_3d(arr: np.ndarray, max_points: int) -> np.ndarray:
    if arr.shape[0] <= max_points:
        return arr
    step = math.ceil(arr.shape[0] / max_points)
    return arr[::step]


def plot_3d_position_dataset(
    hdf: h5py.File,
    demos: Sequence[str],
    path: str,
    *,
    max_samples: int | None,
    output_dir: Path | None,
    save_plots: bool,
    plot_format: str,
    plot_dpi: int,
) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.grid(False)

    for demo_idx, demo in enumerate(demos):
        demo_group = hdf["data"][demo]
        if path not in demo_group:
            continue
        arr = _load_2d(demo_group[path], max_samples)
        if arr.shape[1] < 3:
            continue
        arr = _downsample_for_3d(arr[:, :3], MAX_3D_POINTS_PER_DEMO)
        if arr.shape[0] == 0:
            continue
        t = np.linspace(0.0, 1.0, arr.shape[0])
        ax.scatter(
            arr[:, 0],
            arr[:, 1],
            arr[:, 2],
            c=t,
            cmap="viridis",
            marker="o",
            s=10,
            alpha=0.82,
        )

    sample_ds = _first_dataset_for_path(hdf, demos, path)
    labels = _channel_labels(path, sample_ds) if sample_ds is not None else ["x", "y", "z"]
    labels = (labels + ["x", "y", "z"])[:3]
    ax.set_title(f"3D {path}")
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    cbar = fig.colorbar(cm.ScalarMappable(cmap="viridis"), ax=ax, pad=0.1, shrink=0.6)
    cbar.set_label("Time within demo")
    fig.tight_layout()

    if save_plots and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_dir / f"{_slug(path)}_3d.{plot_format}",
            dpi=plot_dpi,
            bbox_inches="tight",
        )


def plot_hdf5_dataset(
    file_path: Path,
    *,
    roots: Iterable[str],
    include_next_obs: bool,
    max_demos: int | None,
    max_samples: int | None,
    max_vector_channels: int,
    plot_3d: bool,
    save_plots: bool,
    show_plots: bool,
    output_dir: Path | None,
    plot_format: str,
    plot_dpi: int,
    resistance_reference_line: float | None,
) -> None:
    roots = tuple(roots)
    if include_next_obs and "next_obs" not in roots:
        roots = (*roots, "next_obs")

    with h5py.File(file_path, "r") as hdf:
        demos = _find_demos(hdf, max_demos=max_demos)
        if not demos:
            raise RuntimeError(f"No demos found in {file_path}")

        paths = discover_timeseries_paths(
            hdf,
            demos,
            roots,
            max_vector_channels=max_vector_channels,
        )
        print(f"Found {len(demos)} demos and {len(paths)} plottable numeric datasets in {file_path}")
        for path, width in paths.items():
            print(f"  {path} ({width} channel{'s' if width != 1 else ''})")

        for path, width in paths.items():
            plot_time_series_dataset(
                hdf,
                demos,
                path,
                width,
                max_samples=max_samples,
                output_dir=output_dir,
                save_plots=save_plots,
                plot_format=plot_format,
                plot_dpi=plot_dpi,
                resistance_reference_line=resistance_reference_line,
            )

        if plot_3d:
            for path, width in paths.items():
                if _is_position_path(path, width):
                    plot_3d_position_dataset(
                        hdf,
                        demos,
                        path,
                        max_samples=max_samples,
                        output_dir=output_dir,
                        save_plots=save_plots,
                        plot_format=plot_format,
                        plot_dpi=plot_dpi,
                    )

    if show_plots:
        plt.show()
    else:
        plt.close("all")


def main() -> int:
    file_path = FILE_PATH.resolve()
    output_dir = OUTPUT_DIR
    if output_dir is None and SAVE_PLOTS:
        output_dir = file_path.with_suffix("").parent / f"{file_path.stem}_plots"

    plot_hdf5_dataset(
        file_path,
        roots=ROOTS_TO_PLOT,
        include_next_obs=INCLUDE_NEXT_OBS,
        max_demos=MAX_DEMOS,
        max_samples=MAX_SAMPLES,
        max_vector_channels=MAX_VECTOR_CHANNELS,
        plot_3d=PLOT_3D_POSITION_DATA,
        save_plots=SAVE_PLOTS,
        show_plots=SHOW_PLOTS,
        output_dir=output_dir,
        plot_format=PLOT_FORMAT,
        plot_dpi=PLOT_DPI,
        resistance_reference_line=RESISTANCE_REFERENCE_LINE,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
