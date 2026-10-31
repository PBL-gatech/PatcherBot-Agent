"""Benchmark one scan corner before the complete scan; edit constants, then run.

The legacy and fast corner runs use independent instances and the same frame
prefix. Timed runs include instance initialization, stitching, and image writing.
The full run follows the corner runs, so its device and disk caches may be warm.
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "patcherbot" / "utils"))
SESSION = "2026_09_23-18_05"
DATA_ROOT = REPO_ROOT / "experiments" / "Data" / "rig_recorder_data"
OUTPUT_ROOT = REPO_ROOT / "experiments" / "Data" / "stitch_benchmarks"
RUN_LEGACY_FIRST_CORNER = True
RUN_GPU_FIRST_CORNER = True
RUN_GPU_FULL_SCAN = True
FULL_SCAN_REPEATS = 1  # Set to 3 for repeated confirmation after the first run.
# Set this to an earlier report to retain its baseline when rerunning GPU stages.
PREVIOUS_REPORT = None
TARGET_FULL_SECONDS = 10.0
CORNER_TOLERANCE_TILE_FRACTION = 0.03
AFTER_CORNER_TRAVEL_TILE_FRACTION = 0.5
MAX_LEGACY_CANVAS_GB = 4.0
FAST_OPTIONS = {
    "backend": "cupy",
    "overlap": 0.4,
    "workers": 8,
    "output_downsample": 1,
    "select_frames": True,
    "preserve_coverage": True,
    "blend": "feather",
}
# Optional speed/appearance tradeoff: select_frames=True, blend="feather".
# Spatial selection can omit thin source-covered edge regions; audit coverage.


def load_module(name, filename):
    """Load utility files without importing the hardware/GUI package."""
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "patcherbot" / "utils" / filename
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def new_stitcher(stitcher_class):
    return stitcher_class(
        date_time_folder=SESSION,
        date_time_root=str(DATA_ROOT),
        use_calibration=True,
        flip_x=True,
        flip_y=True,
        stitch_fraction=1,
        downsample=1,
        max_canvas_gb=MAX_LEGACY_CANVAS_GB,
        use_pixel_drift_stitch=True,
    )


def environment_report():
    versions = {}
    for name in ("numpy", "opencv-python", "opencv-python-headless", "cupy-cuda12x", "tifffile"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        gpu = str(exc)
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "package_versions": versions,
        "gpu": gpu,
    }


def first_corner_prefix(stitcher, fast_module):
    timestamps, coords, paths = stitcher._align_all_frames_to_movement()
    xy = stitcher._aligned_coords_from_raw(coords)
    xy -= xy[0]
    sample = stitcher._to_bgr(paths[0])
    height, width = sample.shape[:2]
    tolerance = min(height, width) * CORNER_TOLERANCE_TILE_FRACTION
    corners = fast_module._path_corners(xy, tolerance)
    interior = corners[(corners > 0) & (corners < len(xy) - 1)]
    if not len(interior):
        raise ValueError("No spatial corner found; choose a serpentine scan for this benchmark.")
    first = int(interior[0])
    travel = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))))
    target_travel = travel[first] + min(height, width) * AFTER_CORNER_TRAVEL_TILE_FRACTION
    stop = min(len(paths), int(np.searchsorted(travel, target_travel)) + 1)
    coverage = {}
    for name, points in (("first_corner", xy[:stop]), ("full_scan", xy)):
        selected = (fast_module.select_keyframes(points, (height, width), FAST_OPTIONS["overlap"])
                    if FAST_OPTIONS["select_frames"] else np.arange(len(points)))
        original = fast_module.coverage_report(points, (height, width))
        reduced = fast_module.coverage_report(points[selected], (height, width))
        coverage[name] = {
            "all_frames": original,
            "selected_frames": reduced,
            "bounds_preserved": bool(np.allclose(original["bounds_xyxy"], reduced["bounds_xyxy"], rtol=0, atol=1e-8)),
            "estimated_uncovered_fraction_increase": (
                reduced["estimated_uncovered_fraction"] - original["estimated_uncovered_fraction"]
            ),
            "note": "Geometric footprint estimates; these do not measure seam alignment or image quality.",
        }
    return stop, {
        "source_frame_count": len(paths),
        "frame_shape": list(sample.shape),
        "frame_dtype": str(sample.dtype),
        "first_corner_index": first,
        "corner_prefix_frame_count": stop,
        "corner_tolerance_pixels": tolerance,
        "after_corner_travel_pixels": min(height, width) * AFTER_CORNER_TRAVEL_TILE_FRACTION,
        "prefix_timestamp_start": float(timestamps[0]),
        "prefix_timestamp_end": float(timestamps[stop - 1]),
        "first_frame": str(paths[0]),
        "last_prefix_frame": str(paths[stop - 1]),
        "scan_area_window": stitcher.scan_area_window,
        "movement_file": stitcher.coord_file,
        "movement_file_bytes": Path(stitcher.coord_file).stat().st_size,
        "calibration_file": stitcher.calibration_file,
        "calibration": stitcher.calibration,
        "coverage_comparison": coverage,
    }


def run_stage(name, stitcher_class, fast_module, output_dir, prefix_count=None):
    start = time.perf_counter()
    source = new_stitcher(stitcher_class)
    initialization_seconds = time.perf_counter() - start
    if prefix_count is not None:
        # Each run owns this instance; the repository class defaults are untouched.
        source.frame_catalog = source.frame_catalog[:prefix_count]
    output = output_dir / f"{name}.tif"
    metadata = output_dir / f"{name}.json"
    if name == "legacy_first_corner":
        saved = source.stitch_streaming(
            output_path=str(output), metadata_path=str(metadata), cleanup_memmap=True
        )
    else:
        saved = source.stitch_fast(
            output_path=str(output), metadata_path=str(metadata),
            config=fast_module.FastStitchConfig(**FAST_OPTIONS),
        )
        if FAST_OPTIONS["backend"] == "cupy":
            importlib.import_module("cupy").cuda.Stream.null.synchronize()
    total_seconds = time.perf_counter() - start
    details = json.loads(metadata.read_text(encoding="utf-8"))
    return {
        "output": str(saved),
        "output_bytes": output.stat().st_size,
        "metadata": str(metadata),
        "input_frame_count": len(source.frame_catalog),
        "initialization_seconds": initialization_seconds,
        "stitch_and_write_seconds": total_seconds - initialization_seconds,
        "total_seconds": total_seconds,
        "details": details,
    }


def exact_coverage(offsets, selected, tile_shape, max_pixels=100_000_000):
    """Compare unions of actual rounded tile rectangles on the output pixel grid."""
    import cv2

    tile_h, tile_w = tile_shape
    lower = offsets.min(axis=0)
    upper = offsets.max(axis=0) + [tile_w, tile_h]
    width, height = (int(value) for value in upper - lower)
    if width * height > max_pixels:
        raise MemoryError("Exact audit exceeds its 100 million pixel allocation guard.")
    origins = offsets - lower

    def union(indices):
        delta = np.zeros((height + 1, width + 1), dtype=np.int32)
        x, y = origins[indices].T
        np.add.at(delta, (y, x), 1)
        np.add.at(delta, (y + tile_h, x), -1)
        np.add.at(delta, (y, x + tile_w), -1)
        np.add.at(delta, (y + tile_h, x + tile_w), 1)
        np.cumsum(delta, axis=0, dtype=np.int32, out=delta)
        np.cumsum(delta, axis=1, dtype=np.int32, out=delta)
        return delta[:-1, :-1] > 0

    original = union(np.arange(len(offsets)))
    reduced = union(selected)
    lost = original & ~reduced
    original_count = int(original.sum())
    lost_count = int(lost.sum())
    # Flood background from beyond the image edges to distinguish exterior
    # strips from holes enclosed by the selected tile union (8-connectivity).
    background = np.pad((~reduced).astype(np.uint8), 1, constant_values=1)
    cv2.floodFill(background, None, (0, 0), 2, flags=8)
    interior_loss = int(np.count_nonzero(lost & (background[1:-1, 1:-1] == 1)))
    del background
    core = cv2.erode(original.astype(np.uint8), np.ones((17, 17), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0)
    deep_loss = int(np.count_nonzero(lost & (core != 0)))
    del core
    radius = cv2.distanceTransform(lost.astype(np.uint8), cv2.DIST_C, 3)
    max_radius = float(radius.max())
    del radius
    selected_lower = offsets[selected].min(axis=0)
    selected_upper = offsets[selected].max(axis=0) + [tile_w, tile_h]
    result = {
        "method": "Exact integer pixel rectangle union using the backend's rounded positions; no coarse coverage grid.",
        "source_frame_count": len(offsets), "selected_frame_count": len(selected),
        "canvas_shape_yx": [height, width],
        "source_bounds_xyxy": [int(v) for v in (*lower, *upper)],
        "selected_bounds_xyxy": [int(v) for v in (*selected_lower, *selected_upper)],
        "bounds_preserved": bool(np.array_equal(lower, selected_lower) and np.array_equal(upper, selected_upper)),
        "source_covered_pixels": original_count,
        "selected_covered_pixels": int(reduced.sum()),
        "lost_source_pixels": lost_count,
        "lost_source_fraction": lost_count / original_count,
        "extra_selected_pixels": int(np.count_nonzero(reduced & ~original)),
        "lost_pixels_connected_to_exterior": lost_count - interior_loss,
        "lost_pixels_in_enclosed_holes": interior_loss,
        "lost_pixels_within_8px_of_source_boundary": lost_count - deep_loss,
        "lost_pixels_more_than_8px_from_source_boundary": deep_loss,
        "loss_maximum_chebyshev_inradius_pixels": max_radius,
        "limitation": "Footprint coverage does not establish focus, seam alignment, or registration quality.",
    }
    return result, original, reduced, lost


def audit_outputs(report_path):
    """Write exact geometry evidence and previews after benchmarking, on CPU only."""
    import cv2
    import tifffile

    report_path = Path(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    output_dir = report_path.parent
    legacy = load_module("audit_legacy_scan_stitcher", "scanStitcher.py")
    source = new_stitcher(legacy.ScanStitcher)
    if Path(report["session"]).resolve() != Path(source.dataset_dir).resolve():
        raise ValueError("Audit session differs from the benchmark session.")
    _, raw_coords, paths = source._align_all_frames_to_movement()
    coords = source._aligned_coords_from_raw(raw_coords)
    source_h, source_w = source._to_bgr(paths[0]).shape[:2]
    evidence = {"benchmark": str(report_path.resolve()), "runs": {}}

    def caption(image, title, detail):
        banner = np.full((64, image.shape[1], 3), 28, dtype=np.uint8)
        cv2.putText(banner, title, (16, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.putText(banner, detail, (16, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (205, 205, 205), 1, cv2.LINE_AA)
        return np.vstack((banner, image))

    def preview(path, width=900, fast=True):
        pixels = tifffile.imread(path)[:, :, ::-1] if fast else cv2.imread(str(path))
        if pixels is None:
            raise ValueError(f"Could not decode preview: {path}")
        return cv2.resize(pixels, (width, round(pixels.shape[0] * width / pixels.shape[1])), interpolation=cv2.INTER_AREA)

    for name in ("gpu_first_corner", "gpu_full_scan"):
        run = report["runs"].get(name)
        if run is None:
            continue
        details = json.loads(Path(run["metadata"]).read_text(encoding="utf-8"))
        selected = np.asarray(details["selected_indices"], dtype=np.int64)
        scale = np.asarray(details["pixel_scale_xy"], dtype=float)
        count = run["input_frame_count"]
        offsets = np.rint((coords[:count] - coords[0]) / scale).astype(np.int64)
        shape = (round(source_h / scale[1]), round(source_w / scale[0]))
        metrics, original, reduced, lost = exact_coverage(offsets, selected, shape)
        metrics["output_shape_matches_bounds"] = details["canvas_shape"][:2] == [
            metrics["selected_bounds_xyxy"][3] - metrics["selected_bounds_xyxy"][1],
            metrics["selected_bounds_xyxy"][2] - metrics["selected_bounds_xyxy"][0],
        ]
        evidence["runs"][name] = metrics
        size = (900, round(original.shape[0] * 900 / original.shape[1]))
        geometry = np.full((size[1], size[0], 3), (40, 30, 20), dtype=np.uint8)
        geometry[cv2.resize(reduced.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) != 0] = (215, 215, 215)
        # Preserve thin losses in the preview: any lost-pixel contribution is red.
        geometry[cv2.resize(lost.astype(np.float32), size, interpolation=cv2.INTER_AREA) > 0] = (60, 70, 240)
        geometry = caption(geometry, f"{name}: exact coverage", "Gray: selected coverage | red: lost source pixels (thin losses enlarged)")
        geometry_path = output_dir / f"{name}_coverage.png"
        cv2.imwrite(str(geometry_path), geometry)
        metrics["coverage_preview"] = str(geometry_path)
        del original, reduced, lost
    if "gpu_full_scan" in report["runs"]:
        run = report["runs"]["gpu_full_scan"]
        blend = run.get("details", {}).get("config", {}).get("blend", report["fast_options"]["blend"])
        image = caption(preview(run["output"]), f"Full scan: GPU {blend}", f"{run['total_seconds']:.3f}s including initialization and write | full resolution TIFF; reduced preview")
        cv2.imwrite(str(output_dir / "full_scan_preview.png"), image)
    if all(name in report["runs"] for name in ("legacy_first_corner", "gpu_first_corner")):
        panels = []
        for name, label, is_fast in (("legacy_first_corner", "Legacy streaming", False), ("gpu_first_corner", "Fast GPU", True)):
            run = report["runs"][name]
            if is_fast:
                label += " " + run.get("details", {}).get("config", {}).get("blend", report["fast_options"]["blend"])
            panels.append(caption(preview(run["output"], fast=is_fast), label,
                                  f"Same {run['input_frame_count']} source frames | {run['total_seconds']:.3f}s | reduced preview"))
        height = max(panel.shape[0] for panel in panels)
        panels = [cv2.copyMakeBorder(panel, 0, height - panel.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(28, 28, 28)) for panel in panels]
        cv2.imwrite(str(output_dir / "first_corner_comparison.png"), np.hstack(panels))
    path = output_dir / "coverage.json"
    path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence, indent=2), flush=True)
    return evidence


def audit_full_output(output_path, metadata_path, reference_path, coverage_path):
    """Audit a completed full output and compare it with a prior saved mosaic."""
    import cv2
    import tifffile

    output_path, metadata_path = Path(output_path), Path(metadata_path)
    reference_path, coverage_path = Path(reference_path), Path(coverage_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    legacy = load_module("audit_full_legacy_scan_stitcher", "scanStitcher.py")
    source = new_stitcher(legacy.ScanStitcher)
    _, raw_coords, paths = source._align_all_frames_to_movement()
    coords = source._aligned_coords_from_raw(raw_coords)
    source_h, source_w = source._to_bgr(paths[0]).shape[:2]
    scale = np.asarray(metadata["pixel_scale_xy"], dtype=float)
    offsets = np.rint((coords - coords[0]) / scale).astype(np.int64)
    selected = np.asarray(metadata["selected_indices"], dtype=np.int64)
    metrics, original_mask, selected_mask, lost_mask = exact_coverage(
        offsets, selected, (round(source_h / scale[1]), round(source_w / scale[0]))
    )
    del original_mask, selected_mask, lost_mask
    metrics["output"] = str(output_path.resolve())
    candidate_rgb = tifffile.imread(output_path)
    reference_bgr = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
    if reference_bgr is None:
        raise ValueError(f"Could not read saved reference mosaic: {reference_path}")
    comparison = {
        "reference": str(reference_path.resolve()),
        "candidate_shape": list(candidate_rgb.shape),
        "reference_shape": list(reference_bgr.shape),
        "shapes_match": candidate_rgb.shape == reference_bgr.shape,
        "note": "Saved legacy reference may include the legacy WebP-to-JPEG decode path. Pixel differences are not an image-quality score.",
    }
    if comparison["shapes_match"]:
        absolute_sum, squared_sum = 0.0, 0.0
        max_error = 0.0
        for top in range(0, candidate_rgb.shape[0], 128):
            difference = candidate_rgb[top:top + 128].astype(np.float32)
            difference -= reference_bgr[top:top + 128, :, ::-1]
            absolute_sum += float(np.abs(difference).sum(dtype=np.float64))
            squared_sum += float(np.square(difference).sum(dtype=np.float64))
            max_error = max(max_error, float(np.abs(difference).max()))
        mse = squared_sum / candidate_rgb.size
        comparison.update({
            "mean_absolute_error_8bit": absolute_sum / candidate_rgb.size,
            "mean_squared_error_8bit": mse,
            "psnr_db": float(10 * np.log10(255.0 ** 2 / mse)) if mse else None,
            "maximum_absolute_error_8bit": max_error,
            "pixels_exactly_equal": mse == 0,
        })
    metrics["saved_reference_comparison"] = comparison
    width = 900
    thumbnail = cv2.resize(candidate_rgb[:, :, ::-1],
                           (width, round(candidate_rgb.shape[0] * width / candidate_rgb.shape[1])),
                           interpolation=cv2.INTER_AREA)
    banner = np.full((64, width, 3), 28, dtype=np.uint8)
    cv2.putText(banner, "Full scan: all frames, GPU overwrite", (16, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(banner, f"{len(selected)} source frames | full source footprint preserved: {metrics['lost_source_pixels'] == 0} | reduced preview",
                (16, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (205, 205, 205), 1, cv2.LINE_AA)
    preview_path = output_path.parent / "full_all_frames_preview.png"
    if not cv2.imwrite(str(preview_path), np.vstack((banner, thumbnail))):
        raise RuntimeError(f"Could not write preview: {preview_path}")
    metrics["preview"] = str(preview_path.resolve())
    evidence = json.loads(coverage_path.read_text(encoding="utf-8")) if coverage_path.exists() else {"runs": {}}
    evidence["runs"]["gpu_full_all_frames_overwrite"] = metrics
    coverage_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def main():
    legacy = load_module("benchmark_legacy_scan_stitcher", "scanStitcher.py")
    fast = load_module("scanStitcherFast", "scanStitcherFast.py")
    output_dir = OUTPUT_ROOT / datetime.now().strftime("%Y_%m_%d-%H_%M_%S_%f")
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "benchmark.json"
    report = {
        "created": datetime.now().astimezone().isoformat(),
        "session": str((DATA_ROOT / SESSION).resolve()),
        "environment": environment_report(),
        "fast_options": FAST_OPTIONS,
        "target_full_seconds": TARGET_FULL_SECONDS,
        "timing_scope": "instance initialization + alignment/selection + decode + processing + TIFF/metadata write",
        "cache_note": "Stages run in listed order; later stages can use warm device/filesystem caches.",
        "runs": {},
    }
    if PREVIOUS_REPORT:
        previous = json.loads(Path(PREVIOUS_REPORT).read_text(encoding="utf-8"))
        if previous["session"] != report["session"]:
            raise ValueError("Previous report uses a different scan session.")
        baseline = previous.get("runs", {}).get("legacy_first_corner")
        if baseline:
            report["runs"]["legacy_first_corner"] = baseline
            report["baseline_source_report"] = str(PREVIOUS_REPORT)
    try:
        source = new_stitcher(legacy.ScanStitcher)
        prefix_count, inputs = first_corner_prefix(source, fast)
        report["inputs"] = inputs
        del source
        baseline = report["runs"].get("legacy_first_corner")
        if baseline and baseline["input_frame_count"] != prefix_count:
            raise ValueError("Previous baseline uses a different first-corner prefix.")
        if FULL_SCAN_REPEATS < 1:
            raise ValueError("FULL_SCAN_REPEATS must be at least one.")
        stages = [
            ("legacy_first_corner", RUN_LEGACY_FIRST_CORNER, prefix_count),
            ("gpu_first_corner", RUN_GPU_FIRST_CORNER, prefix_count),
        ]
        stages.extend(
            ("gpu_full_scan" if index == 0 else f"gpu_full_scan_{index + 1}", RUN_GPU_FULL_SCAN, None)
            for index in range(FULL_SCAN_REPEATS)
        )
        for name, enabled, count in stages:
            if not enabled:
                continue
            report["active_stage"] = name
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"\nBenchmark stage: {name}", flush=True)
            report["runs"][name] = run_stage(name, legacy.ScanStitcher, fast, output_dir, count)
            report.pop("active_stage", None)
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        runs = report["runs"]
        if "legacy_first_corner" in runs and "gpu_first_corner" in runs:
            report["first_corner_speedup"] = (
                runs["legacy_first_corner"]["total_seconds"] / runs["gpu_first_corner"]["total_seconds"]
            )
        if "gpu_full_scan" in runs:
            full_times = [run["total_seconds"] for name, run in runs.items() if name.startswith("gpu_full_scan")]
            report["full_scan_times_seconds"] = full_times
            report["full_scan_under_target"] = max(full_times) < TARGET_FULL_SECONDS
            report["full_scan_seconds_over_target"] = max(full_times) - TARGET_FULL_SECONDS
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Benchmark report: {report_path}", flush=True)
    print(json.dumps({key: value for key, value in report.items() if key in (
        "first_corner_speedup", "full_scan_under_target", "full_scan_seconds_over_target"
    )}, indent=2))
    return report


if __name__ == "__main__":
    main()
