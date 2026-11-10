"""Optional spatial-selection stitcher using the existing ScanStitcher geometry.

Tiles use stage coordinates with optional validated image-overlap refinement. CuPy and tifffile are imported only when the fast path is called.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import time
import uuid

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessConfig:
    noise_sigma: float = 3
    background_sigma: float = 45
    local_sigma: float = 50
    epsilon: float = 0.1
    gain: float = 45
    midpoint: float = 128

    def validate(self):
        for name in ('noise_sigma', 'background_sigma', 'local_sigma', 'epsilon'):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if not np.isfinite(self.gain) or self.gain < 0:
            raise ValueError('gain must be nonnegative and finite')
        if not np.isfinite(self.midpoint) or not 0 <= self.midpoint <= 255:
            raise ValueError('midpoint must be between 0 and 255')
        return self


def preprocessing_metadata(config):
    return None if config is None else asdict(config)


def preprocess_bgr(frame, config):
    """Apply the supplied float32 Gaussian/RMS recipe before resizing or blending."""
    if config is None:
        return frame
    config.validate()
    from PIL import Image
    from scipy.ndimage import gaussian_filter

    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError('Preprocessing requires uint8 three-channel BGR frames')
    # Use Pillow's grayscale conversion to match the reference calculation.
    image = Image.fromarray(frame[:, :, ::-1]).convert('L')
    pixels = np.asarray(image, dtype=np.float32)
    detail = (gaussian_filter(pixels, sigma=config.noise_sigma)
              - gaussian_filter(pixels, sigma=config.background_sigma))
    local_scale = np.sqrt(
        gaussian_filter(detail * detail, sigma=config.local_sigma) + config.epsilon)
    normalized = detail / local_scale
    enhanced = np.clip(config.midpoint + config.gain * normalized, 0, 255).astype(np.uint8)
    return np.repeat(enhanced[:, :, None], 3, axis=2)


@dataclass(frozen=True)
class FastStitchConfig:
    overlap: float = 0.4
    workers: int = 8
    output_downsample: int = 1
    backend: str = "cupy"
    select_frames: bool = True
    preserve_coverage: bool = True
    blend: str = "feather"
    batch_size: int = 16
    tile_size: int = 1024
    max_gpu_gb: float = 12.0
    max_output_gb: float = 4.0
    overwrite: bool = False
    preprocessing: PreprocessConfig | None = None
    refine_images: bool = False

    def validate(self):
        if self.preprocessing is not None:
            if not isinstance(self.preprocessing, PreprocessConfig):
                raise TypeError("preprocessing must be a PreprocessConfig or None (OFF).")
            self.preprocessing.validate()
        if self.backend not in ("numpy", "cupy"):
            raise ValueError("backend must be 'numpy' or 'cupy'.")
        if self.blend not in ("feather", "overwrite"):
            raise ValueError("blend must be 'feather' or 'overwrite'.")
        if not np.isfinite(self.overlap) or not 0 <= self.overlap < 1:
            raise ValueError("overlap must satisfy 0 <= overlap < 1.")
        for name in ("workers", "output_downsample", "batch_size", "tile_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.tile_size % 16:
            raise ValueError("tile_size must be a multiple of 16 for tiled TIFF.")
        for name in ("max_gpu_gb", "max_output_gb"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite.")
        if any(not isinstance(value, bool) for value in
               (self.select_frames, self.preserve_coverage, self.overwrite, self.refine_images)):
            raise ValueError("select_frames, preserve_coverage, overwrite and refine_images must be booleans.")
        return self


def _validated(xy, tile_shape):
    points = np.asarray(xy, dtype=np.float64)
    if points.size == 0:
        points = np.empty((0, 2), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("xy must have shape (N, 2).")
    if not np.all(np.isfinite(points)):
        raise ValueError("xy must contain only finite coordinates.")
    shape = np.asarray(tile_shape, dtype=np.float64)
    if shape.shape != (2,) or not np.all(np.isfinite(shape)) or np.any(shape <= 0):
        raise ValueError("tile_shape must be a positive finite (height, width).")
    return points, shape[::-1]


def _path_corners(points, tolerance):
    """Iterative RDP using distance to segments, including collinear reversals."""
    count = len(points)
    keep = np.zeros(count, dtype=bool)
    keep[[0, count - 1]] = True
    pending = [(0, count - 1)]
    while pending:
        first, last = pending.pop()
        if last <= first + 1:
            continue
        delta = points[last] - points[first]
        offsets = points[first + 1:last] - points[first]
        length_sq = float(delta @ delta)
        if length_sq:
            fraction = np.clip((offsets @ delta) / length_sq, 0.0, 1.0)
            residual = offsets - fraction[:, None] * delta
        else:
            residual = offsets
        distance = np.max(np.abs(residual), axis=1)
        local = int(np.argmax(distance))
        if distance[local] > tolerance:
            split = first + 1 + local
            keep[split] = True
            pending.append((first, split))
            pending.append((split, last))
    return np.flatnonzero(keep)


def select_keyframes(xy, tile_shape, overlap=0.4):
    """Return original frame indices selected by displacement, never frame rate.

    The requested overlap bounds each axis separately: retained translations
    should differ by at most ``(1 - overlap) * (width, height)`` along the path
    when source sampling permits it. This is a linear overlap target, not an
    overlap-area guarantee for diagonal motion. Sparse input gaps remain gaps.

    Distinct sweep endpoints, significant turns/reversals, and global spatial
    extrema are retained. Tiny direction changes below 1% of the field of view
    are treated as coordinate noise; pauses within 0.1% are deduplicated. These
    tolerances become smaller for overlap requests very close to one.
    """
    points, size_xy = _validated(xy, tile_shape)
    overlap = float(overlap)
    if not np.isfinite(overlap) or not 0 <= overlap < 1:
        raise ValueError("overlap must satisfy 0 <= overlap < 1.")
    if not len(points):
        return np.empty(0, dtype=np.int64)
    if len(points) == 1:
        return np.array([0], dtype=np.int64)

    normalized = (points - points[0]) / size_xy
    step = 1.0 - overlap
    duplicate_tolerance = min(0.001, step * 0.01)
    corner_tolerance = min(0.01, step * 0.1)

    # Compare with the last distinct point so slow motion cannot disappear just
    # because each consecutive position changes by less than the tolerance.
    distinct = [0]
    for index in range(1, len(points)):
        if np.max(np.abs(normalized[index] - normalized[distinct[-1]])) > duplicate_tolerance:
            distinct.append(index)
    if len(distinct) == 1:
        return np.array([0], dtype=np.int64)
    if np.any(points[-1] != points[distinct[-1]]):
        distinct.append(len(points) - 1)
    extrema = set()
    for axis in range(2):
        if np.ptp(normalized[:, axis]) > 2 * corner_tolerance:
            extrema.add(int(np.argmin(normalized[:, axis])))
            extrema.add(int(np.argmax(normalized[:, axis])))
    distinct = np.asarray(sorted(set(distinct) | extrema), dtype=np.int64)
    path = normalized[distinct]
    mandatory = set(_path_corners(path, corner_tolerance).tolist())
    # Ignore extrema on effectively stationary axes, but retain exact extrema
    # on axes with real movement even when RDP would smooth a shallow bend.
    mandatory.update(int(np.searchsorted(distinct, index)) for index in extrema)

    selected = [0]
    for index in range(1, len(path)):
        displacement = np.max(np.abs(path[index] - path[selected[-1]]))
        if displacement > step + 1e-12:
            # Take the last available sample before the target is exceeded.
            if index - 1 > selected[-1]:
                selected.append(index - 1)
            if np.max(np.abs(path[index] - path[selected[-1]])) > step + 1e-12:
                selected.append(index)  # An unavoidable gap in the source.
        if index in mandatory and selected[-1] != index:
            selected.append(index)
    return distinct[np.asarray(selected, dtype=np.int64)]


def _rectangle_union_mask(x0, y0, x1, y1, shape):
    """Rasterize axis-aligned index ranges with bounded-memory difference sums."""
    height, width = shape
    valid = (x1 > x0) & (y1 > y0)
    x0, y0, x1, y1 = (a[valid] for a in (x0, y0, x1, y1))
    counts = np.zeros((height + 1, width + 1), dtype=np.int64)
    np.add.at(counts, (y0, x0), 1)
    np.add.at(counts, (y1, x0), -1)
    np.add.at(counts, (y0, x1), -1)
    np.add.at(counts, (y1, x1), 1)
    return counts.cumsum(axis=0).cumsum(axis=1)[:-1, :-1] > 0


def coverage_report(xy, tile_shape):
    """Estimate holes inside the tile bounding box with at most 512 squared cells.

    Coordinates are treated as tile top-left translations. Returned values are
    JSON-safe. The center-sampled estimate can miss narrow gaps, so conservative
    lower/upper uncovered-area bounds are also reported. Bounds may be loose:
    a cell counts as certainly covered only if one tile contains the entire
    cell. This reports geometric coverage, not image-registration connectivity,
    and bounding-box corners outside the intended scan count as uncovered.
    """
    points, size_xy = _validated(xy, tile_shape)
    if not len(points):
        return {
            "tile_count": 0,
            "bounds_xyxy": None,
            "grid_shape_yx": [0, 0],
            "grid_cell_size_xy": None,
            "estimated_uncovered_fraction": None,
            "uncovered_fraction_lower_bound": None,
            "uncovered_fraction_upper_bound": None,
        }
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0) + size_xy
    extent = upper - lower
    grid_xy = np.maximum(1, np.ceil(512 * extent / np.max(extent)).astype(int))
    nx, ny = (int(v) for v in grid_xy)
    edges_x = np.linspace(lower[0], upper[0], nx + 1)
    edges_y = np.linspace(lower[1], upper[1], ny + 1)
    end = points + size_xy

    def mask(mode):
        ranges = []
        for axis, edges in enumerate((edges_x, edges_y)):
            if mode == "center":
                centers = (edges[:-1] + edges[1:]) * 0.5
                start = np.searchsorted(centers, points[:, axis], side="left")
                stop = np.searchsorted(centers, end[:, axis], side="right")
            elif mode == "contained":
                start = np.searchsorted(edges[:-1], points[:, axis], side="left")
                stop = np.searchsorted(edges[1:], end[:, axis], side="right")
            else:  # Any positive-area intersection with the cell.
                start = np.searchsorted(edges[1:], points[:, axis], side="right")
                stop = np.searchsorted(edges[:-1], end[:, axis], side="left")
            ranges.append((start, stop))
        return _rectangle_union_mask(
            ranges[0][0], ranges[1][0], ranges[0][1], ranges[1][1], (ny, nx)
        )

    estimate = float(1.0 - np.mean(mask("center")))
    lower_bound = float(1.0 - np.mean(mask("intersects")))
    upper_bound = float(1.0 - np.mean(mask("contained")))
    return {
        "tile_count": int(len(points)),
        "bounds_xyxy": [float(v) for v in (*lower, *upper)],
        "grid_shape_yx": [ny, nx],
        "grid_cell_size_xy": [float(v) for v in extent / grid_xy],
        "estimated_uncovered_fraction": estimate,
        "uncovered_fraction_lower_bound": lower_bound,
        "uncovered_fraction_upper_bound": upper_bound,
    }



def _rectangle_union_area(rectangles):
    """Exact integer area using an edge sweep, without a pixel-sized canvas."""
    events = {}
    for index, (x0, y0, x1, y1) in enumerate(rectangles):
        events.setdefault(int(x0), []).append((index, (int(y0), int(y1))))
        events.setdefault(int(x1), []).append((index, None))
    active = {}
    previous_x = covered_y = area = 0
    for x in sorted(events):
        area += (x - previous_x) * covered_y
        for index, interval in events[x]:
            if interval is None:
                active.pop(index, None)
            else:
                active[index] = interval
        covered_y = 0
        end = None
        for start, stop in sorted(active.values()):
            if end is None or start > end:
                covered_y += stop - start
                end = stop
            elif stop > end:
                covered_y += stop - end
                end = stop
        previous_x = x
    return area


def preserve_source_coverage(offsets, tile_shape, base_indices, max_pixels=100_000_000):
    """Add source frames until their exact integer pixel union is preserved.

    ``offsets`` must be the same rounded source-pixel placements used to render.
    Only input rectangles are added: holes already present in the source remain
    holes. A bounded difference grid and integral image avoid decoding any images;
    larger canvases use exact rectangle sweeps instead of allocating that grid.
    """
    started = time.perf_counter()
    points = np.asarray(offsets)
    shape = np.asarray(tile_shape)
    if points.size == 0:
        points = np.empty((0, 2), dtype=np.int64)
    if (points.ndim != 2 or points.shape[1] != 2 or
            not np.isfinite(points).all() or np.any(points != np.rint(points))):
        raise ValueError("offsets must be a finite integer (N, 2) array.")
    if (shape.shape != (2,) or not np.isfinite(shape).all() or
            np.any(shape <= 0) or np.any(shape != np.rint(shape))):
        raise ValueError("tile_shape must contain positive integer height and width.")
    if (isinstance(max_pixels, bool) or not isinstance(max_pixels, (int, np.integer))
            or max_pixels < 1):
        raise ValueError("max_pixels must be a positive integer.")
    base = np.asarray(base_indices)
    if (base.ndim != 1 or not np.isfinite(base).all() or
            np.any(base != np.rint(base)) or np.any(base < 0) or np.any(base >= len(points))):
        raise ValueError("base_indices must contain valid source frame indices.")
    base = np.unique(base.astype(np.int64))
    count = len(points)
    if not count:
        return base, {"source_covered_pixels": 0, "base_covered_pixels": 0,
                      "missing_pixels_before": 0, "missing_pixels_after": 0,
                      "added_frame_count": 0, "selected_frame_count": 0,
                      "selection_seconds": time.perf_counter() - started,
                      "canvas_shape_yx": [0, 0]}
    if np.max(np.abs(points)) > 2**52 or np.max(shape) > 2**52:
        raise ValueError("Pixel geometry exceeds the supported integer range.")
    points = points.astype(np.int64)
    points = points - points.min(axis=0)
    tile_h, tile_w = (int(v) for v in shape)
    width = int(points[:, 0].max()) + tile_w
    height = int(points[:, 1].max()) + tile_h
    grid_pixels = (height + 1) * (width + 1)
    if grid_pixels > max_pixels:
        rectangles = np.column_stack((points, points + [tile_w, tile_h]))
        source_pixels = _rectangle_union_area(rectangles)
        base_pixels = _rectangle_union_area(rectangles[base])
        selected_mask = np.zeros(count, dtype=bool)
        selected_mask[base] = True
        # Check each remaining rectangle against the current selected union.
        # Source order is deterministic; every added frame contributes coverage.
        for index in np.flatnonzero(~selected_mask):
            x0, y0, x1, y1 = rectangles[index]
            intersects = (selected_mask & (rectangles[:, 0] < x1) &
                          (rectangles[:, 2] > x0) & (rectangles[:, 1] < y1) &
                          (rectangles[:, 3] > y0))
            clipped = rectangles[intersects].copy()
            clipped[:, :2] = np.maximum(clipped[:, :2], [x0, y0])
            clipped[:, 2:] = np.minimum(clipped[:, 2:], [x1, y1])
            if _rectangle_union_area(clipped) < tile_w * tile_h:
                selected_mask[index] = True
        result = np.flatnonzero(selected_mask)
        remaining = source_pixels - _rectangle_union_area(rectangles[result])
        if remaining:
            raise RuntimeError("Source coverage reconstruction failed its exact union check.")
        return result, {
            "source_covered_pixels": source_pixels, "base_covered_pixels": base_pixels,
            "missing_pixels_before": source_pixels - base_pixels,
            "missing_pixels_after": remaining,
            "added_frame_count": len(result) - len(base),
            "selected_frame_count": len(result),
            "selection_seconds": time.perf_counter() - started,
            "canvas_shape_yx": [height, width], "coverage_grid_bytes": 0,
            "coverage_method": "exact_rectangle_sweep",
        }
    sentinel = count + 1
    # One tally encodes source count in the low range and selected coverage in
    # multiples of sentinel. No source-only count can reach that sentinel.
    maximum_count = count + len(base) * sentinel
    dtype = np.int32 if maximum_count <= np.iinfo(np.int32).max else np.int64
    grid = np.zeros((height + 1, width + 1), dtype=dtype)
    x0, y0 = points.T
    x1, y1 = x0 + tile_w, y0 + tile_h

    def add_rectangles(indices, value):
        np.add.at(grid, (y0[indices], x0[indices]), value)
        np.add.at(grid, (y1[indices], x0[indices]), -value)
        np.add.at(grid, (y0[indices], x1[indices]), -value)
        np.add.at(grid, (y1[indices], x1[indices]), value)

    add_rectangles(np.arange(count), 1)
    add_rectangles(base, sentinel)
    np.cumsum(grid, axis=1, dtype=dtype, out=grid)
    np.cumsum(grid, axis=0, dtype=dtype, out=grid)
    interior = grid[:-1, :-1]
    source_pixels = int(np.count_nonzero(interior))
    missing = (interior > 0) & (interior < sentinel)
    missing_before = int(np.count_nonzero(missing))
    selected = set(base.tolist())
    remaining = missing_before
    if remaining:
        # Reuse the tally allocation for a summed-area table of missing pixels.
        # Integral sums cannot exceed grid_pixels; use int64 when required.
        if grid_pixels > np.iinfo(dtype).max:
            raise MemoryError("Coverage integral exceeds the supported counter range.")
        grid[1:, 1:] = missing
        grid[0, :] = 0
        grid[:, 0] = 0
        np.cumsum(grid, axis=1, dtype=dtype, out=grid)
        np.cumsum(grid, axis=0, dtype=dtype, out=grid)
        scores = (grid[y1, x1].astype(np.int64) - grid[y0, x1] -
                  grid[y1, x0] + grid[y0, x0])
        candidates = np.flatnonzero(scores > 0)
        candidates = candidates[np.argsort(-scores[candidates], kind="stable")]
        for index in candidates:
            region = missing[y0[index]:y1[index], x0[index]:x1[index]]
            gain = int(np.count_nonzero(region))
            if gain:
                selected.add(int(index))
                region[:] = False
                remaining -= gain
                if not remaining:
                    break
        if remaining:
            raise RuntimeError("Source coverage reconstruction failed its exact union check.")
    result = np.asarray(sorted(selected), dtype=np.int64)
    return result, {
        "source_covered_pixels": source_pixels,
        "base_covered_pixels": source_pixels - missing_before,
        "missing_pixels_before": missing_before, "missing_pixels_after": remaining,
        "added_frame_count": len(result) - len(base), "selected_frame_count": len(result),
        "selection_seconds": time.perf_counter() - started,
        "canvas_shape_yx": [height, width],
        "coverage_grid_bytes": int(grid.nbytes + missing.nbytes),
    }

def _output_paths(requested, metadata_path):
    """Require a caller-selected directory and contain all derived files in it."""
    if requested is None or not Path(requested).is_absolute():
        raise ValueError("Fast output_path must be an explicit absolute TIFF path.")
    requested = Path(requested)
    directory = requested.parent.resolve()
    output = requested.resolve()
    if output.suffix.lower() not in (".tif", ".tiff"):
        raise ValueError("Fast output must use .tif or .tiff.")
    companion = output.with_name(output.stem + "_downsampled_2x" + output.suffix).resolve()
    metadata = (output.with_name(output.stem + "_meta.json") if metadata_path is None
                else Path(metadata_path))
    if not metadata.is_absolute():
        metadata = directory / metadata
    metadata = metadata.resolve()
    destinations = (output, companion, metadata)
    if any(path.parent != directory for path in destinations):
        raise ValueError("All fast output files must stay in the selected output directory.")
    if len(set(destinations)) != len(destinations):
        raise ValueError("Image, companion and metadata paths must differ.")
    return destinations


def _read_bgr(path):
    """Decode once; avoid the legacy loader's WebP-to-JPEG round trip."""
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Could not decode camera frame: {path}")
    return frame


# Registration calculation recovered from f203c110bb:experiments/restitch_image_refined.py.
def _registration_overlap(a, b, delta):
    dx, dy = np.asarray(delta, dtype=int)
    h, w = a.shape
    return (a[max(0, dy):min(h, h + dy), max(0, dx):min(w, w + dx)],
            b[max(0, -dy):min(h, h - dy), max(0, -dx):min(w, w - dx)])


def _registration_shift(a, b, factor):
    size = (a.shape[1] // factor, a.shape[0] // factor)
    aa = cv2.resize(a, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    bb = cv2.resize(b, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    shift, response = cv2.phaseCorrelate(aa, bb, cv2.createHanningWindow(size, cv2.CV_32F))
    return np.asarray(shift) * np.array([a.shape[1] / size[0], a.shape[0] / size[1]]), response


def _register_pair(item, images, offsets):
    i, j = item
    a, b = _registration_overlap(images[i], images[j], offsets[j] - offsets[i])
    if min(a.shape) < 192:
        return None
    split = a.shape[1] // 2
    shift, response = _registration_shift(a[:, :split], b[:, :split], 4)
    check, check_response = _registration_shift(a[:, :split], b[:, :split], 8)
    if (min(response, check_response) < 0.25 or np.linalg.norm(shift) > 64.0
            or np.linalg.norm(check - shift) > 6):
        return None
    # Validate the unchanged shift on the other half of the _registration_overlap.
    size = (a.shape[1] // 4, a.shape[0] // 4)
    aa = cv2.resize(a, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    bb = cv2.resize(b, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    aa = cv2.GaussianBlur(aa, (0, 0), 0.75) - cv2.GaussianBlur(aa, (0, 0), 7.5)
    bb = cv2.GaussianBlur(bb, (0, 0), 0.75) - cv2.GaussianBlur(bb, (0, 0), 7.5)
    small_shift = shift / np.array([a.shape[1] / size[0], a.shape[0] / size[1]])
    matrix = np.array([[1, 0, -small_shift[0]], [0, 1, -small_shift[1]]], np.float32)
    moved = cv2.warpAffine(bb, matrix, size)
    margin = int(np.ceil(np.max(np.abs(small_shift)))) + 5
    roi = (slice(margin, size[1] - margin), slice(size[0] // 2 + margin, size[0] - margin))
    if aa[roi].size < 300 or min(aa[roi].std(), bb[roi].std()) < 0.1:
        return None
    before = float(np.corrcoef(aa[roi].ravel(), bb[roi].ravel())[0, 1])
    after = float(np.corrcoef(aa[roi].ravel(), moved[roi].ravel())[0, 1])
    if not np.isfinite([before, after]).all() or after < 0.35 or after < before - 0.01:
        return None
    return {"i": i, "j": j, "shift": shift.tolist(), "response": float(min(response, check_response)),
            "heldout_patch_before": before, "heldout_patch_after": after}


def _solve_scan_positions(edges, count):
    from scipy.sparse import coo_matrix, eye, vstack
    from scipy.sparse.linalg import lsqr

    rows = np.repeat(np.arange(len(edges)), 2)
    columns = np.array([[e["i"], e["j"]] for e in edges]).ravel()
    incidence = coo_matrix((np.tile([-1., 1.], len(edges)), (rows, columns)),
                           shape=(len(edges), count)).tocsr()
    target = -np.array([e["shift"] for e in edges])
    confidence = np.sqrt(np.clip([e["response"] for e in edges], 0.1, 1))
    correction = np.zeros((count, 2))
    prior = eye(count, format="csr") * 0.08
    anchor = coo_matrix(([10.], ([0], [0])), shape=(1, count)).tocsr()
    for _ in range(6):
        errors = np.linalg.norm(incidence @ correction - target, axis=1)
        weights = confidence * np.sqrt(np.minimum(1., 5. / np.maximum(errors, 1e-6)))
        design = vstack([incidence.multiply(weights[:, None]), prior, anchor]).tocsr()
        for axis in range(2):
            rhs = np.r_[target[:, axis] * weights, np.zeros(count + 1)]
            correction[:, axis] = lsqr(design, rhs, atol=1e-8, btol=1e-8)[0]
    return correction


def _refine_scan_positions(timestamps, coords, paths, selected):
    """Port of the accepted Nov 3 reference's validated overlap/stage-prior fit."""
    from scipy.spatial import cKDTree

    started = time.perf_counter()
    frame_ids = np.unique(np.r_[selected, 0, len(paths) - 1]).astype(int)

    def reject(reason, **diagnostics):
        reason = f"{reason} Retaining recorded stage positions."
        print(f"[ScanStitcher] {reason}")
        return coords, {
            "method": "Recorded stage positions; image refinement rejected",
            "applied": False, "reason": reason,
            "registration_keyframe_indices": frame_ids.tolist(),
            "elapsed_seconds": time.perf_counter() - started,
            **diagnostics,
        }

    if len(frame_ids) < 2:
        return reject("Image refinement requires overlapping frames.")
    offsets = np.rint(coords[frame_ids] - coords[0]).astype(int)
    with ThreadPoolExecutor(max_workers=6) as pool:
        images = list(pool.map(lambda i: cv2.imread(str(paths[i]), cv2.IMREAD_GRAYSCALE), frame_ids))
    if any(image is None for image in images):
        raise ValueError("Could not decode a registration frame.")
    h, w = images[0].shape
    if any(image.shape != (h, w) for image in images):
        raise ValueError("Registration frames must have matching dimensions.")
    neighbors = cKDTree(offsets).query(offsets, k=min(10, len(frame_ids)))[1]
    pairs = set()
    for i, near in enumerate(neighbors):
        for j in near:
            if i < j:
                dx, dy = np.abs(offsets[j] - offsets[i])
                if dx < w - 192 and dy < h - 192:
                    pairs.add((i, int(j)))
    pairs = sorted(pairs)
    with ThreadPoolExecutor(max_workers=6) as pool:
        edges = [edge for edge in pool.map(
            lambda pair: _register_pair(pair, images, offsets), pairs) if edge is not None]
    if len(edges) < 20:
        return reject(f"Only {len(edges)} reliable overlaps; image refinement rejected.",
                      attempted_pairs=len(pairs), accepted_pairs=len(edges))
    holdout = edges[::5]
    training = [edge for k, edge in enumerate(edges) if k % 5]
    probe = _solve_scan_positions(training, len(frame_ids))
    if not np.isfinite(probe).all():
        return reject("Image refinement validation produced non-finite corrections.",
                      attempted_pairs=len(pairs), accepted_pairs=len(edges))
    before = np.array([np.linalg.norm(edge["shift"]) for edge in holdout])
    after = np.array([np.linalg.norm(probe[edge["j"]] - probe[edge["i"]] + edge["shift"])
                      for edge in holdout])
    validation = {"heldout_pair_count": len(holdout),
                  "median_error_before": float(np.median(before)),
                  "median_error_after": float(np.median(after)),
                  "fraction_improved": float(np.mean(after < before))}
    if np.median(after) >= np.median(before):
        return reject("Held-out overlaps did not improve.", attempted_pairs=len(pairs),
                      accepted_pairs=len(edges), validation=validation)
    corrections = _solve_scan_positions(edges, len(frame_ids))
    finite = bool(np.isfinite(corrections).all())
    maximum_correction = float(np.max(np.linalg.norm(corrections, axis=1))) if finite else None
    if not finite or maximum_correction > 128:
        reason = (f"Image refinement requested a {maximum_correction:.2f}-pixel correction "
                  "above the 128-pixel limit." if finite
                  else "Image refinement produced non-finite corrections.")
        return reject(reason, attempted_pairs=len(pairs), accepted_pairs=len(edges),
                      validation=validation, max_correction_pixels=maximum_correction,
                      correction_limit_pixels=128.)
    all_corrections = np.column_stack([
        np.interp(timestamps, timestamps[frame_ids], corrections[:, axis]) for axis in range(2)])
    return coords + all_corrections, {
        "applied": True,
        "method": "Confidence-filtered overlap translations, robust graph solve with stage prior",
        "registration_keyframe_indices": frame_ids.tolist(), "attempted_pairs": len(pairs),
        "accepted_pairs": len(edges), "validation": validation, "edges": edges,
        "all_frame_corrections": all_corrections.tolist(),
        "max_correction_pixels": float(np.max(np.linalg.norm(corrections, axis=1))),
        "unregistered_frame_policy": "Timestamp interpolation between registered keyframes",
        "elapsed_seconds": time.perf_counter() - started,
    }


def _feather_weights(shape):
    h, w = shape
    y = np.minimum(np.arange(h) + 1, np.arange(h, 0, -1)).astype(np.float32)
    x = np.minimum(np.arange(w) + 1, np.arange(w, 0, -1)).astype(np.float32)
    return (y / y.max())[:, None] * (x / x.max())[None, :]


def stitch_fast(stitcher, output_path=None, metadata_path=None, config=None, frame_limit=None):
    """Write a stage-position mosaic and return its path, retaining no host canvas.

    Spatial selection replaces temporal frame decimation. The existing instance's
    legacy ``downsample`` is deliberately unused; ``output_downsample`` resizes
    both tiles and coordinates. ``blend='overwrite', select_frames=False`` keeps
    the legacy draw order for comparisons at matching output resolution.
    ``output_path`` must be an explicit absolute TIFF path. Metadata and the
    2x area-downsampled companion are restricted to that same directory.
    """
    start = time.perf_counter()
    config = FastStitchConfig() if config is None else config
    if not isinstance(config, FastStitchConfig):
        raise TypeError("config must be a FastStitchConfig.")
    config.validate()
    if not stitcher.use_pixel_drift_stitch:
        raise ValueError("Fast stitching supports only pixel-drift mode.")
    if frame_limit is not None and (isinstance(frame_limit, bool) or
            not isinstance(frame_limit, (int, np.integer)) or frame_limit < 1):
        raise ValueError("frame_limit must be a positive integer or None.")
    output, downsampled_output, metadata = _output_paths(output_path, metadata_path)
    if not config.overwrite and any(path.exists() for path in (output, downsampled_output, metadata)):
        raise FileExistsError("Fast output already exists; choose a new path or set overwrite=True.")
    try:
        import tifffile
    except ImportError as exc:
        raise RuntimeError("Fast stitching requires tifffile in the selected environment.") from exc
    xp = np
    device = "CPU"
    if config.backend == "cupy":
        try:
            import cupy as xp
            if xp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("No CUDA device available.")
            props = xp.cuda.runtime.getDeviceProperties(xp.cuda.Device().id)
            device = props["name"]
            if isinstance(device, bytes):
                device = device.decode()
            xp.zeros(1, dtype=xp.float32).sum().get()
        except Exception as exc:
            raise RuntimeError("CuPy startup failed; use the optional GPU environment or backend='numpy'.") from exc

    timestamps, raw_coords, paths = stitcher._align_all_frames_to_movement()
    source_count = len(paths)
    timestamps, raw_coords, paths = stitcher._sample_aligned_data(timestamps, raw_coords, paths)
    if frame_limit is not None:
        timestamps, raw_coords, paths = timestamps[:frame_limit], raw_coords[:frame_limit], paths[:frame_limit]
    if not len(paths):
        raise ValueError("No frames available for fast stitching.")
    coords = np.asarray(stitcher._aligned_coords_from_raw(raw_coords), dtype=np.float64)
    if coords.shape != (len(paths), 2) or not np.isfinite(coords).all():
        raise ValueError("Aligned coordinates must be finite with shape (N, 2).")
    sample = _read_bgr(paths[0])
    if sample.ndim != 3 or sample.shape[2] != 3 or sample.dtype != np.uint8:
        raise ValueError("Fast stitching requires uint8 three-channel BGR frames.")
    source_h, source_w = sample.shape[:2]
    selected = (select_keyframes(coords, (source_h, source_w), config.overlap)
                if config.select_frames else np.arange(len(paths), dtype=np.int64))
    registration = None
    if config.refine_images:
        coords, registration = _refine_scan_positions(timestamps, coords, paths, selected)
        selected = (select_keyframes(coords, (source_h, source_w), config.overlap)
                    if config.select_frames else np.arange(len(paths), dtype=np.int64))
    base_selected = selected.copy()
    base_selected_set = set(base_selected.tolist())
    tile_h = math.ceil(source_h / config.output_downsample)
    tile_w = math.ceil(source_w / config.output_downsample)
    scale_xy = np.array([source_w / tile_w, source_h / tile_h], dtype=np.float64)
    drift = coords - coords[0]
    offsets_float = np.rint(drift / scale_xy)
    if np.max(np.abs(offsets_float)) > 2**52:
        raise ValueError("Stage offsets exceed the supported integer pixel range.")
    offsets = offsets_float.astype(np.int64)
    selection_coverage = None
    if config.select_frames and config.preserve_coverage:
        selected, selection_coverage = preserve_source_coverage(offsets, (tile_h, tile_w), selected)
    # Compute order on the complete path, then filter: sampling must not alter
    # the legacy corner/underlay interpretation.
    selected_set = set(selected.tolist())
    draw_order = [int(i) for i in stitcher._pixel_drift_draw_order(coords) if int(i) in selected_set]
    origin = offsets[selected].min(axis=0)
    extent = offsets[selected].max(axis=0) - origin + [tile_w, tile_h]
    width, height = (int(v) for v in extent)
    output_bytes = width * height * 3
    if output_bytes > config.max_output_gb * 1024**3:
        raise MemoryError("Output exceeds max_output_gb; explicitly increase output_downsample or the limit.")
    # Include accumulation, normalization and tile-transfer working space.
    has_boundary_repairs = len(selected) > len(base_selected)
    accumulator_bytes = width * height * ((16 if config.blend == "feather" else 3)
                                         + int(has_boundary_repairs))
    working_bytes = tile_h * tile_w * 32 + config.tile_size**2 * 24
    required_bytes = accumulator_bytes + working_bytes
    if required_bytes > config.max_gpu_gb * 1024**3:
        raise MemoryError("Accumulator exceeds max_gpu_gb; explicitly increase output_downsample or the limit.")
    if config.backend == "cupy":
        free_bytes, _ = xp.cuda.runtime.memGetInfo()
        if required_bytes > free_bytes * 0.8:
            raise MemoryError("Insufficient free GPU memory for the mosaic and working buffers.")
    timing = {"prepare_seconds": time.perf_counter() - start}
    coverage = coverage_report(coords[selected], (source_h, source_w))
    if config.blend == "feather":
        canvas = xp.zeros((height, width, 3), dtype=xp.float32)
        weight_canvas = xp.zeros((height, width), dtype=xp.float32)
        weights = xp.asarray(_feather_weights((tile_h, tile_w)))
    else:
        canvas = xp.zeros((height, width, 3), dtype=xp.uint8)
        weight_canvas = weights = None
    # Border-completion frames must not overweight the interior and introduce
    # illumination bands. Their contributions are restricted to new coverage.
    base_coverage = None
    if has_boundary_repairs:
        base_coverage = xp.zeros((height, width), dtype=xp.bool_)
        for index in base_selected:
            x, y = (int(v) for v in offsets[index] - origin)
            base_coverage[y:y + tile_h, x:x + tile_w] = True

    def decode(index):
        frame = sample if index == 0 else _read_bgr(paths[index])
        if frame.shape != sample.shape or frame.dtype != np.uint8:
            raise ValueError(f"Inconsistent frame shape or dtype: {paths[index]}")
        frame = preprocess_bgr(frame, config.preprocessing)
        if (tile_h, tile_w) != (source_h, source_w):
            frame = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        return frame

    blend_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=config.workers) as pool:
        for first in range(0, len(draw_order), config.batch_size):
            batch_indices = draw_order[first:first + config.batch_size]
            # map receives only a bounded batch, avoiding thousands of queued
            # decoded frames on Python versions before Executor.map buffersize.
            for index, frame in zip(batch_indices, pool.map(decode, batch_indices)):
                x, y = (int(v) for v in offsets[index] - origin)
                tile = xp.asarray(frame)
                border_mask = (None if index in base_selected_set else
                               ~base_coverage[y:y + tile_h, x:x + tile_w])
                if config.blend == "feather":
                    tile_weights = weights if border_mask is None else weights * border_mask
                    canvas[y:y + tile_h, x:x + tile_w] += tile * tile_weights[:, :, None]
                    weight_canvas[y:y + tile_h, x:x + tile_w] += tile_weights
                else:
                    target = canvas[y:y + tile_h, x:x + tile_w]
                    target[:] = tile if border_mask is None else xp.where(border_mask[:, :, None], tile, target)
    if config.backend == "cupy":
        xp.cuda.get_current_stream().synchronize()
    timing["decode_fuse_seconds"] = time.perf_counter() - blend_start

    output.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    temporary = output.with_name(f".{output.stem}.{token}.partial{output.suffix}")
    temporary_downsampled = downsampled_output.with_name(
        f".{downsampled_output.stem}.{token}.partial{output.suffix}")
    temporary_pixels = output.with_name(f".{output.stem}.{token}.pixels.partial")
    temporary_metadata = metadata.with_name(f".{metadata.name}.{token}.partial")
    write_start = time.perf_counter()

    def output_tiles():
        side = config.tile_size
        for top in range(0, height, side):
            for left in range(0, width, side):
                block = canvas[top:top + side, left:left + side]
                if config.blend == "feather":
                    denominator = weight_canvas[top:top + side, left:left + side]
                    block = xp.clip(xp.rint(block / xp.maximum(denominator[:, :, None], 1e-12)), 0, 255).astype(xp.uint8)
                host = xp.asnumpy(block) if config.backend == "cupy" else np.asarray(block)
                padded = np.zeros((side, side, 3), dtype=np.uint8)
                padded[:host.shape[0], :host.shape[1]] = host[:, :, ::-1]
                yield padded

    # Classic TIFF is more widely viewable; leave room for headers and metadata.
    classic_tiff_limit = 2**32 - 32 * 1024**2
    tile_count = math.ceil(height / config.tile_size) * math.ceil(width / config.tile_size)
    tiled_tiff_bytes = tile_count * (config.tile_size**2 * 3 + 8)
    try:
        with tifffile.TiffWriter(temporary, bigtiff=tiled_tiff_bytes >= classic_tiff_limit) as writer:
            writer.write(output_tiles(), shape=(height, width, 3), dtype=np.uint8,
                         photometric="rgb", tile=(config.tile_size, config.tile_size),
                         compression=None, metadata={"axes": "YXS"})
        # Decode to disk-backed storage instead of allocating a full host canvas.
        rgb = np.memmap(temporary_pixels, mode="w+", dtype=np.uint8,
                        shape=(height, width, 3))
        try:
            tifffile.imread(temporary, out=rgb)
            downsampled_rgb = cv2.resize(
                rgb, ((width + 1) // 2, (height + 1) // 2), interpolation=cv2.INTER_AREA)
        finally:
            rgb._mmap.close()
        tifffile.imwrite(temporary_downsampled, downsampled_rgb,
                         bigtiff=downsampled_rgb.nbytes >= classic_tiff_limit,
                         photometric="rgb", compression="deflate", metadata={"axes": "YXS"})
        timing["write_seconds"] = time.perf_counter() - write_start
        timing["total_seconds"] = time.perf_counter() - start
        report = {
            "coord_file": stitcher.coord_file, "photo_dir": stitcher.photo_dir,
            "use_calibration": stitcher.use_calibration,
            "calibration_file": stitcher.calibration_file,
            "calibration": stitcher.calibration,
            "scan_time_window": stitcher.scan_area_window,
            "flip_x": stitcher.flip_x, "flip_y": stitcher.flip_y,
            "use_pixel_drift_stitch": stitcher.use_pixel_drift_stitch,
            "stitch_fraction": stitcher.stitch_fraction,
            "downsample": config.output_downsample,
            "legacy_downsample_unused": stitcher.downsample,
            "stitch_mode": "pixel_drift_fast",
            "registration_mode": ("image_refined_stage_prior"
                                  if registration is not None and registration.get("applied", False)
                                  else "stage_only"),
            "registration": registration,
            "canvas_shape": [height, width, 3], "num_frames": len(selected),
            "source_frame_count": source_count, "eligible_frame_count": len(paths),
            "source_frames": [dict(name=Path(path).name, timestamp=float(timestamp),
                                   stage_xy_um=np.asarray(stage).tolist())
                              for path, timestamp, stage in zip(paths, timestamps, raw_coords)],
            "selected_frame_count": len(selected), "decoded_frame_count": len(selected),
            "selected_indices": selected.tolist(), "draw_order": draw_order,
            "base_selected_indices": base_selected.tolist(),
            "coverage_repair_mode": "missing_pixels_only" if has_boundary_repairs else "none",
            "backend": config.backend, "device": device, "config": asdict(config),
            "decoder": "opencv_direct",
            "preprocessing": preprocessing_metadata(config.preprocessing),
            "timing": timing, "coverage": coverage,
            "selection_coverage": selection_coverage,
            "output": str(output), "metadata_path": str(metadata),
            "downsampled_output": str(downsampled_output),
            "downsampled_canvas_shape": list(downsampled_rgb.shape),
            "downsampled_factor": 2,
            "output_origin_xy_pixels": (coords[0] + origin * scale_xy).tolist(),
            "pixel_scale_xy": scale_xy.tolist(),
            "estimated_output_bytes": output_bytes,
            "estimated_accumulator_bytes": accumulator_bytes,
            "streaming_memmap_path": None,
        }
        temporary_metadata.write_text(json.dumps(report, indent=2), encoding="utf-8")
        os.replace(temporary, output)
        os.replace(temporary_downsampled, downsampled_output)
        os.replace(temporary_metadata, metadata)
    finally:
        temporary.unlink(missing_ok=True)
        temporary_downsampled.unlink(missing_ok=True)
        temporary_pixels.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)
    stitcher.canvas = None
    stitcher._streaming_canvas_path = None
    stitcher._streaming_canvas_shape = (height, width, 3)
    stitcher._fast_stitch_report = report
    return str(output)
