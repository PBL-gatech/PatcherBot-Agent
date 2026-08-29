#!/usr/bin/env python3

import argparse
import csv
import datetime
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError

# --- USER CONFIGURATION DEFAULTS ---
DEFAULT_INPUT_DIR = Path(r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\experiments\Data\agent_movement_data\2026_02_18-21_38")
DEFAULT_FPS = 30.0
DEFAULT_RED_THRESHOLDS = (150, 100, 100)  # (r_min, g_max, b_max)
DEFAULT_AXIS_LIMIT = 85
LOCK_WHITE_MODE = True  # Set to True to lock white-dot coordinates to the modal position.
USE_IMAGE_TIME = True  # Set to True to derive timing from filenames instead of constant FPS.
CSV_FIELDNAMES = ["filename", "red_x", "red_y", "white_x", "white_y", "duration"]


def natural_sort_key(path: Path) -> List[object]:
    """
    Generate a natural sorting key for file paths.

    Args:
        path (Path): File path whose name will be used to generate
            the sorting key.

    Returns:
        List[object]: A list containing integers and lowercase strings
            representing the split components of the file name, suitable
            for use as a sorting key.
    """
    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r"(\d+)", path.name)]


def detect_white_dot(bgr: np.ndarray, edge_margin: int = 2) -> Optional[Tuple[float, float]]:
    """
    Detect the small bright dot while ignoring the red square and edge glints.
    Uses white top-hat to emphasize small bright features on bright backgrounds.
    
    Args:
        bgr (np.ndarray): Input image in BGR format.
        edge_margin (int, optional): Minimum distance (in pixels) from the
            image border for valid detections. Components too close to the
            edge are discarded. Defaults to 2.

    Returns:
        Optional[Tuple[float, float]]: (x, y) coordinates of the detected
            dot's centroid in pixel space, or None if no valid candidate
            is found.
    """

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, (0, 80, 80), (8, 255, 255))
    red2 = cv2.inRange(hsv, (170, 80, 80), (180, 255, 255))
    red_mask = cv2.dilate(cv2.bitwise_or(red1, red2), np.ones((5, 5), np.uint8), iterations=1)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opened = cv2.morphologyEx(gray_blur, cv2.MORPH_OPEN, se)
    tophat = cv2.subtract(gray_blur, opened)
    tophat[red_mask > 0] = 0

    thr = max(5, int(np.percentile(tophat, 98)))
    _, mask = cv2.threshold(tophat, thr, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None

    height, width = mask.shape
    best: Optional[Tuple[float, float]] = None
    best_score = float("-inf")

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        cx, cy = centroids[label]

        if not (3 <= area <= 100 and w <= 12 and h <= 12):
            continue
        if (
            x <= edge_margin
            or y <= edge_margin
            or (x + w) >= (width - edge_margin)
            or (y + h) >= (height - edge_margin)
        ):
            continue

        comp_mask = labels == label
        score = float(tophat[comp_mask].mean()) + 0.05 * area
        if score > best_score:
            best_score = score
            best = (float(cx), float(cy))

    return best


def euclidean_err(ax: np.ndarray, bx: np.ndarray, ay: np.ndarray, by: np.ndarray) -> np.ndarray:
    """
    Vectorized Euclidean distance that gracefully propagates NaNs.
    
    Args:
        ax (np.ndarray): X-coordinates of the first set of points.
        bx (np.ndarray): X-coordinates of the second set of points.
        ay (np.ndarray): Y-coordinates of the first set of points.
        by (np.ndarray): Y-coordinates of the second set of points.

    Returns:
        np.ndarray: Array of Euclidean distances between corresponding
            points.
    """

    return np.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


class AgentVisualizer:
    """
    Build outputs (GIF, CSV, trajectory plot) for a directory of agent frames.
    """

    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp"}

    def __init__(
        self,
        input_dir: Path,
        fps: float = DEFAULT_FPS,
        red_thresholds: Sequence[int] = DEFAULT_RED_THRESHOLDS,
        axis_limit: int = DEFAULT_AXIS_LIMIT,
        lock_white_to_mode: bool = False,
        use_image_time: bool = False,
    ) -> None:
        """
        Initialize the AgentVisualizer with input configuration and output paths.

        Args:
            input_dir (Path): Directory containing input image frames. The path
                is expanded and resolved to an absolute path.
            fps (float, optional): Frames per second for GIF generation.
                Defaults to DEFAULT_FPS.
            red_thresholds (Sequence[int], optional): Thresholds for detecting
                red regions in the format (r_min, g_max, b_max). These define
                the minimum red channel value and maximum green/blue channel
                values. Defaults to DEFAULT_RED_THRESHOLDS.
            axis_limit (int, optional): Symmetric limit for x/y axes in trajectory
                plots. Defaults to DEFAULT_AXIS_LIMIT.
            lock_white_to_mode (bool, optional): If True, constrains detected
                white points to a dominant mode across frames to improve
                temporal consistency. Defaults to False.
        """
        self.input_dir = Path(input_dir).expanduser().resolve()
        self.fps = fps
        self.r_min, self.g_max, self.b_max = red_thresholds
        self.axis_limit = axis_limit
        self.lock_white_to_mode = lock_white_to_mode
        self._use_image_time = use_image_time

        self.base_name = self.input_dir.name
        self.output_dir = self.input_dir
        self.output_csv = self.output_dir / f"{self.base_name}.csv"
        self.output_plot = self.output_dir / f"{self.base_name}.png"
        self.output_gif = self.output_dir / f"{self.base_name}.gif"

    def collect_image_paths(self) -> List[Path]:
        """
        Collect and naturally sort valid image file paths from the input directory.

        Returns:
            List[Path]: A list of Path objects corresponding to valid image files,
                sorted in natural (human-readable) order.
        """
        return sorted(
            (p for p in self.input_dir.iterdir() if p.is_file() and p.suffix.lower() in self.IMAGE_EXTS),
            key=natural_sort_key,
        )

    @staticmethod
    def _extract_timestamp_from_filename(path: Path) -> Optional[float]:
        """Parse a POSIX timestamp from camera_image_YYYYMMDD_HHMMSS_micro.png style names."""

        match = re.search(r"(\d{8})_(\d{6})_(\d{1,6})", path.stem)
        if not match:
            return None

        date_part, time_part, micro_part = match.groups()
        try:
            dt_obj = datetime.datetime.strptime(f"{date_part}{time_part}", "%Y%m%d%H%M%S")
            micro = int(micro_part.ljust(6, "0")[:6])
            dt_obj = dt_obj.replace(microsecond=micro)
            return dt_obj.timestamp()
        except ValueError:
            return None

    def find_red_centroid(self, arr: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Compute the centroid of red-colored pixels in an image.

        Args:
            arr (np.ndarray): Input image as a NumPy array in RGB format
                with shape (H, W, 3).

        Returns:
            Optional[Tuple[int, int]]: (x, y) coordinates of the centroid
                rounded to the nearest integer, or None if no red pixels
                are detected.
        """
        red_mask = (arr[:, :, 0] >= self.r_min) & (arr[:, :, 1] <= self.g_max) & (arr[:, :, 2] <= self.b_max)
        coords = np.argwhere(red_mask)
        if coords.size == 0:
            return None
        y, x = coords.mean(axis=0)
        return int(round(x)), int(round(y))

    def _compute_white_mode(
        self, rows: Sequence[Dict[str, object]]
    ) -> Optional[Tuple[Tuple[int, int], int]]:
        """
        Determine the most frequent (mode) white point position from row data.

        Args:
            rows (Sequence[Dict[str, object]]): Iterable of row dictionaries,
                each potentially containing "white_x" and "white_y" entries.

        Returns:
            Optional[Tuple[Tuple[int, int], int]]: A tuple containing:
                - (x, y): The most common integer-rounded white point position
                - count (int): Number of occurrences of that position
                Returns None if no valid white points are found.
        """
        counter: Counter[Tuple[int, int]] = Counter()
        for row in rows:
            wx = row.get("white_x")
            wy = row.get("white_y")
            if wx is None or wy is None:
                continue
            if isinstance(wx, float) and np.isnan(wx):
                continue
            if isinstance(wy, float) and np.isnan(wy):
                continue
            counter[(int(round(float(wx))), int(round(float(wy))))] += 1
        if not counter:
            return None
        position, count = counter.most_common(1)[0]
        return position, count

    def _apply_white_mode(self, rows: Sequence[Dict[str, object]], position: Tuple[int, int]) -> None:
        """
        Overwrite all white point coordinates in the dataset with a fixed mode position.
        
        Args:
            rows (Sequence[Dict[str, object]]): Iterable of row dictionaries to update.
            position (Tuple[int, int]): The (x, y) coordinates to assign as the
                white point for all rows.
        """
        mode_x, mode_y = position
        for row in rows:
            row["white_x"] = mode_x
            row["white_y"] = mode_y

    @staticmethod
    def _sanitize_row(row: Dict[str, object], fieldnames: Sequence[str]) -> Dict[str, object]:
        sanitized: Dict[str, object] = {}
        """
        Normalize a row dictionary to ensure compatibility with CSV writing.
        
        Args:
            row (Dict[str, object]): Input row dictionary containing data fields.
            fieldnames (Sequence[str]): Ordered list of expected field names.

        Returns:
            Dict[str, object]: A sanitized dictionary with all required keys and
                no None or NaN values.
        """
        for key in fieldnames:
            value = row.get(key)
            if value is None:
                sanitized[key] = ""
            elif isinstance(value, float) and np.isnan(value):
                sanitized[key] = ""
            else:
                sanitized[key] = value
        return sanitized

    def _compute_frame_durations_ms(
        self, frame_times: Optional[Sequence[Optional[float]]], n_frames: int
    ) -> Tuple[List[int], str, int]:
        """Compute per-frame durations in ms plus a human-readable timing label."""

        default_duration_ms = max(1, int(round(1000.0 / self.fps))) if self.fps > 0 else 1000
        durations_ms: List[int] = [default_duration_ms] * max(n_frames, 0)

        if n_frames <= 0:
            return durations_ms, "no frames", default_duration_ms

        if not self._use_image_time:
            return durations_ms, "constant FPS", default_duration_ms

        if frame_times is None or len(frame_times) != n_frames:
            return durations_ms, "constant FPS (timestamp length mismatch)", default_duration_ms

        fallback_count = 0
        for idx in range(1, n_frames):
            prev = frame_times[idx - 1]
            curr = frame_times[idx]
            if prev is None or curr is None:
                fallback_count += 1
                continue
            dt = curr - prev
            if np.isfinite(dt) and dt > 0:
                durations_ms[idx] = max(1, int(round(dt * 1000.0)))
            else:
                fallback_count += 1

        if durations_ms:
            timing_label = (
                f"image timestamps (mean {np.mean(durations_ms):.1f} ms, "
                f"range {min(durations_ms)}-{max(durations_ms)} ms, "
                f"fallback frames {fallback_count})"
            )
        else:
            timing_label = "image timestamps"

        return durations_ms, timing_label, default_duration_ms

    def save_csv(self, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
        """
        Write processed row data to a CSV file after sanitization.

        Args:
            rows (Sequence[Dict[str, object]]): Iterable of row dictionaries
                containing data to be written.
            fieldnames (Sequence[str]): Ordered list of column names for the CSV.
        """
        sanitized_rows = [self._sanitize_row(row, fieldnames) for row in rows]
        with open(self.output_csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sanitized_rows)
        print(f"Saved CSV with coordinates to: {self.output_csv}")

    def save_plot(
        self,
        trajectory: Sequence[Tuple[int, int]],
        frame_rows: Sequence[Dict[str, object]],
        frame_times: Optional[Sequence[float]] = None,
    ) -> None:
        """
        Generate and save a multi-panel visualization of agent trajectory and motion statistics.
        
        Args:
            trajectory (Sequence[Tuple[int, int]]): Sequence of (x, y) coordinates
                representing the detected red dot trajectory across frames.
            frame_rows (Sequence[Dict[str, object]]): Per-frame data containing
                keys such as "red_x", "red_y", "white_x", and "white_y" used
                for error computation and time-series analysis.
        """
        if not trajectory:
            print("No red dot detections found; red-specific plots will show 'Not enough data'.")

        xs_arr = np.asarray([pt[0] for pt in trajectory], dtype=float) if trajectory else np.asarray([], dtype=float)
        ys_arr = np.asarray([pt[1] for pt in trajectory], dtype=float) if trajectory else np.asarray([], dtype=float)

        dx = np.diff(xs_arr)
        dy = np.diff(ys_arr)

        if xs_arr.size > 0:
            r = np.hypot(xs_arr, ys_arr)
            theta_rad = np.arctan2(ys_arr, xs_arr)
            theta_unwrapped = np.unwrap(theta_rad)
        else:
            r = np.asarray([], dtype=float)
            theta_unwrapped = np.asarray([], dtype=float)
        dr = np.diff(r)
        dtheta = np.diff(theta_unwrapped)

        fig, axes_grid = plt.subplots(4, 2, figsize=(14, 18), dpi=100)
        axes = axes_grid.ravel()

        def column_to_array(key: str) -> np.ndarray:
            if not frame_rows:
                return np.asarray([], dtype=float)
            values = []
            for row in frame_rows:
                value = row.get(key)
                if value is None:
                    values.append(np.nan)
                else:
                    values.append(float(value))
            return np.asarray(values, dtype=float)

        frames = np.arange(len(frame_rows), dtype=int)
        red_x_series = column_to_array("red_x")
        red_y_series = column_to_array("red_y")
        white_x_series = column_to_array("white_x")
        white_y_series = column_to_array("white_y")
        time_series: Optional[np.ndarray] = None
        if frame_times is not None and len(frame_times) == len(frame_rows):
            time_series = np.asarray(frame_times, dtype=float)

        # XY trajectory
        ax_traj = axes[0]
        if xs_arr.size > 0:
            ax_traj.plot(xs_arr, ys_arr, marker="o")
        else:
            ax_traj.text(0.5, 0.5, "No red detections", ha="center", va="center", transform=ax_traj.transAxes)
        ax_traj.set_title(f"Agent trajectory ({self.base_name})")
        ax_traj.set_xlabel("X coordinate (pixels)")
        ax_traj.set_ylabel("Y coordinate (pixels)")
        ax_traj.set_xlim(0, self.axis_limit)
        ax_traj.set_ylim(self.axis_limit, 0)

        # Histogram of delta X
        ax_hist_dx = axes[1]
        if dx.size > 0:
            ax_hist_dx.hist(dx, bins="auto", color="tab:blue", alpha=0.8)
            ax_hist_dx.set_xlim(-4, 4)
        else:
            ax_hist_dx.text(0.5, 0.5, "Not enough data", ha="center", va="center", transform=ax_hist_dx.transAxes)
        ax_hist_dx.set_title("Histogram of delta X")
        ax_hist_dx.set_xlabel("delta X (pixels)")
        ax_hist_dx.set_ylabel("Count")

        # Histogram of delta Y
        ax_hist_dy = axes[2]
        if dy.size > 0:
            ax_hist_dy.hist(dy, bins="auto", color="tab:orange", alpha=0.8)
            ax_hist_dy.set_xlim(-4, 4)
        else:
            ax_hist_dy.text(0.5, 0.5, "Not enough data", ha="center", va="center", transform=ax_hist_dy.transAxes)
        ax_hist_dy.set_title("Histogram of delta Y")
        ax_hist_dy.set_xlabel("delta Y (pixels)")
        ax_hist_dy.set_ylabel("Count")

        # Histogram of delta r
        ax_hist_dr = axes[3]
        if dr.size > 0:
            ax_hist_dr.hist(dr, bins="auto", color="tab:green", alpha=0.8)
            ax_hist_dr.set_xlim(-4, 4)
        else:
            ax_hist_dr.text(0.5, 0.5, "Not enough data", ha="center", va="center", transform=ax_hist_dr.transAxes)
        ax_hist_dr.set_title("Histogram of delta r")
        ax_hist_dr.set_xlabel("delta r (pixels)")
        ax_hist_dr.set_ylabel("Count")

        # Histogram of delta theta
        ax_hist_dtheta = axes[4]
        if dtheta.size > 0:
            ax_hist_dtheta.hist(dtheta, bins="auto", color="tab:red", alpha=0.8)
            theta_extent = np.max(np.abs(dtheta))
            ax_hist_dtheta.set_xlim(-theta_extent - 1.0, theta_extent + 1.0)
        else:
            ax_hist_dtheta.text(0.5, 0.5, "Not enough data", ha="center", va="center", transform=ax_hist_dtheta.transAxes)
        ax_hist_dtheta.set_title("Histogram of delta theta")
        ax_hist_dtheta.set_xlabel("delta theta (radians)")
        ax_hist_dtheta.set_ylabel("Count")

        # Speed over frames (distance / delta t)
        ax_speed = axes[5]
        step_dist = np.sqrt(np.diff(red_x_series) ** 2 + np.diff(red_y_series) ** 2)
        if time_series is not None:
            delta_t = np.diff(time_series)
        else:
            delta_t = np.full_like(step_dist, fill_value=1.0 / self.fps if self.fps > 0 else 1.0, dtype=float)
        delta_t[delta_t <= 0] = np.nan
        speed = step_dist / delta_t if step_dist.size > 0 else np.asarray([], dtype=float)
        speed_frames = frames[1:] if frames.size > 1 else np.asarray([], dtype=int)
        speed_mask = np.isfinite(speed)
        if speed_mask.any():
            ax_speed.plot(speed_frames[speed_mask], speed[speed_mask], color="tab:olive")
            ax_speed.set_ylabel("Speed (px/sec)")
        else:
            ax_speed.text(0.5, 0.5, "Not enough data", ha="center", va="center", transform=ax_speed.transAxes)
        ax_speed.set_title("6) Speed over frames")
        ax_speed.set_xlabel("Frame")
        ax_speed.grid(True, alpha=0.3)

        # FFT magnitude of delta X and delta Y
        ax_fft = axes[6]
        if time_series is not None and len(time_series) > 1:
            sample_spacing = float(np.nanmean(np.diff(time_series)))
            if not np.isfinite(sample_spacing) or sample_spacing <= 0:
                sample_spacing = 1.0 / self.fps if self.fps > 0 else 1.0
        else:
            sample_spacing = 1.0 / self.fps if self.fps > 0 else 1.0
        if dx.size > 0:
            freq_dx = np.fft.rfftfreq(dx.size, d=sample_spacing)
            fft_dx = np.abs(np.fft.rfft(dx - np.mean(dx))) / max(dx.size, 1)
            ax_fft.plot(freq_dx, fft_dx, label="delta X", color="tab:blue")
        if dy.size > 0:
            freq_dy = np.fft.rfftfreq(dy.size, d=sample_spacing)
            fft_dy = np.abs(np.fft.rfft(dy - np.mean(dy))) / max(dy.size, 1)
            ax_fft.plot(freq_dy, fft_dy, label="delta Y", color="tab:orange", linestyle="--")
        if dx.size == 0 and dy.size == 0:
            ax_fft.text(0.5, 0.5, "Not enough data", ha="center", va="center", transform=ax_fft.transAxes)
        ax_fft.set_title("7) FFT of delta X/Y")
        ax_fft.set_xlabel("Frequency (Hz)")
        ax_fft.set_ylabel("Magnitude")
        if (dx.size > 0) or (dy.size > 0):
            ax_fft.set_xlim(left=0)
            ax_fft.legend()

        # Error subplot
        ax_err = axes[7]
        err = euclidean_err(white_x_series, red_x_series, white_y_series, red_y_series)
        err_label = "Distance white ↔ red (px)"

        if frames.size == 0 or not np.isfinite(err).any():
            ax_err.text(0.5, 0.5, "Not enough data", ha="center", va="center", transform=ax_err.transAxes)
        else:
            ax_err.plot(frames, err, color="tab:purple", label=err_label)

        if frames.size > 0:
            white_missing = (~np.isfinite(white_x_series)) | (~np.isfinite(white_y_series))
            if white_missing.any():
                ax_err.scatter(
                    frames[white_missing],
                    np.zeros(int(white_missing.sum()), dtype=float),
                    marker="x",
                    color="tab:red",
                    label="white detection missing",
                )

        ax_err.set_title("8) Error")
        ax_err.set_xlabel("Frame")
        ax_err.set_ylabel(err_label)
        ax_err.grid(True, alpha=0.3)
        if ax_err.lines or ax_err.collections:
            ax_err.legend()

        fig.tight_layout()
        fig.savefig(self.output_plot, dpi=100)
        plt.close(fig)

        print(f"Saved trajectory plot to: {self.output_plot}")

    def save_gif(
        self,
        frames: Sequence[Image.Image],
        durations_ms: Sequence[int],
        timing_label: str,
        default_duration_ms: int,
    ) -> None:
        if not frames:
            print("No frames available for GIF; skipping GIF creation.", file=sys.stderr)
            return

        if self.fps <= 0:
            raise ValueError("FPS must be a positive number.")

        first_frame, *other_frames = frames
        first_frame.save(
            self.output_gif,
            format="GIF",
            save_all=True,
            append_images=other_frames,
            duration=durations_ms,
            loop=0,
            disposal=2,
        )

        print(
            f"Saved GIF to {self.output_gif} with {len(frames)} frames; timing {timing_label} "
            f"(default frame duration {default_duration_ms} ms)."
        )

    def run(self) -> None:
        """
        Execute the full agent visualization pipeline.

        Raises:
            FileNotFoundError: If the input directory does not exist or contains no supported images.
            RuntimeError: If no valid image data could be processed.
        """
        if not self.input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")

        image_paths = self.collect_image_paths()
        if not image_paths:
            raise FileNotFoundError(f"No supported images found in {self.input_dir}")

        rows: List[Dict[str, object]] = []
        trajectory: List[Tuple[int, int]] = []
        frames: List[Image.Image] = []
        frame_times: List[Optional[float]] = []

        for path in image_paths:
            try:
                with Image.open(path) as img:
                    rgb_img = img.convert("RGB")
                    arr = np.array(rgb_img)
                    centroid = self.find_red_centroid(arr)

                    red_x: Optional[int] = None
                    red_y: Optional[int] = None
                    if centroid:
                        red_x, red_y = centroid
                        trajectory.append((red_x, red_y))

                    bgr_arr = np.ascontiguousarray(arr[:, :, ::-1])
                    white_detection = detect_white_dot(bgr_arr)
                    white_x: Optional[int] = None
                    white_y: Optional[int] = None
                    if white_detection is not None:
                        wx, wy = white_detection
                        white_x, white_y = int(round(wx)), int(round(wy))

                    timestamp = self._extract_timestamp_from_filename(path) if self._use_image_time else None

                    rows.append(
                        {
                            "filename": path.name,
                            "red_x": red_x,
                            "red_y": red_y,
                            "white_x": white_x,
                            "white_y": white_y,
                        }
                    )

                    frames.append(rgb_img.convert("RGBA"))
                    if timestamp is not None:
                        frame_times.append(float(timestamp))
                    elif self._use_image_time:
                        frame_times.append(None)
            except UnidentifiedImageError:
                print(f"Skipping unsupported image file: {path}", file=sys.stderr)
            except OSError as exc:
                print(f"Could not process {path}: {exc}", file=sys.stderr)

        if not rows and not frames:
            raise RuntimeError("No valid image data available after processing.")

        if self.lock_white_to_mode:
            mode_result = self._compute_white_mode(rows)
            if mode_result:
                position, count = mode_result
                self._apply_white_mode(rows, position)
                print(
                    f"White dot mode override enabled: locked to ({position[0]}, {position[1]}) "
                    f"based on {count} detections."
                )
            else:
                print(
                    "White dot mode override requested, but no white dot detections were available to compute the mode.",
                    file=sys.stderr,
                )

        time_series: Optional[List[float]] = None
        if self._use_image_time:
            if frame_times and all(t is not None for t in frame_times):
                time_series = [float(t) for t in frame_times]  # type: ignore[arg-type]
            else:
                print(
                    "Image-time spacing requested, but one or more timestamps were missing; falling back to constant FPS.",
                    file=sys.stderr,
                )

        durations_ms, timing_label, default_duration_ms = self._compute_frame_durations_ms(frame_times, len(frames))

        for idx, row in enumerate(rows):
            row["duration"] = durations_ms[idx] if idx < len(durations_ms) else ""

        if rows:
            self.save_csv(rows, CSV_FIELDNAMES)
        else:
            print("No coordinate data to write; CSV skipped.")

        self.save_plot(trajectory, rows, frame_times=time_series)
        self.save_gif(frames, durations_ms=durations_ms, timing_label=timing_label, default_duration_ms=default_duration_ms)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments for running the agent visualizer.

    Args:
        argv (Optional[Sequence[str]]): List of arguments to parse.
            If None, parses arguments from sys.argv.

    Returns:
        argparse.Namespace: Parsed arguments including:
            - input_dir: Directory containing frame images.
            - fps: Frames per second for GIF output.
            - r_min: Minimum red channel threshold for red centroid detection.
            - g_max: Maximum green channel threshold for red centroid detection.
            - b_max: Maximum blue channel threshold for red centroid detection.
            - axis_limit: Maximum value for X and Y axes in the trajectory plot.
    """
    parser = argparse.ArgumentParser(description="Generate GIF, CSV, and trajectory plot for agent frames.")
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=None,
        help="Directory containing frame images. Defaults to the configured path in the script.",
    )
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="Frames per second for the output GIF.")
    parser.add_argument(
        "--r-min", type=int, default=DEFAULT_RED_THRESHOLDS[0], help="Minimum red channel value for detection."
    )
    parser.add_argument(
        "--g-max", type=int, default=DEFAULT_RED_THRESHOLDS[1], help="Maximum green channel value for detection."
    )
    parser.add_argument(
        "--b-max", type=int, default=DEFAULT_RED_THRESHOLDS[2], help="Maximum blue channel value for detection."
    )
    parser.add_argument(
        "--axis-limit",
        type=int,
        default=DEFAULT_AXIS_LIMIT,
        help="Upper bound for both axes in the trajectory plot.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """
    Entry point for the agent frame visualizer script.

    Args:
        argv (Optional[Sequence[str]]): Optional list of command-line arguments to parse.
            If None, defaults to sys.argv.
    """
    args = parse_args(argv)
    input_dir = args.input_dir or DEFAULT_INPUT_DIR
    if input_dir is None:
        print("An input directory must be provided.", file=sys.stderr)
        sys.exit(1)

    visualizer = AgentVisualizer(
        input_dir=input_dir,
        fps=args.fps,
        red_thresholds=(args.r_min, args.g_max, args.b_max),
        axis_limit=args.axis_limit,
        lock_white_to_mode=LOCK_WHITE_MODE,
        use_image_time=USE_IMAGE_TIME,
    )
    visualizer.run()


if __name__ == "__main__":
    main()
