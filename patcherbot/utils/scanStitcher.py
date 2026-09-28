import os
import glob
import json
from datetime import datetime
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


class ScanStitcher:
    STANDARD_DATA_ROOT = 'experiments/Data/rig_recorder_data'

    def __init__(
        self,
        date_time_folder=None,
        date_time_root=STANDARD_DATA_ROOT,
        use_calibration=False,
        coord_cols=(1, 2),
        flip_x=False,
        flip_y=True,
        prefer_exts=(".webp", ".jpg", ".jpeg"),
        stitch_fraction=1,
        downsample=10,
        max_canvas_gb=1.0,
        use_pixel_drift_stitch=True,
    ):
        print("[ScanStitcher] Initializing...")
        if date_time_folder is None:
            raise ValueError(
                "ScanStitcher now requires a date_time_folder. "
                "Expected standard layout under experiments/Data/rig_recorder_data/<date_time_folder>/"
            )

        self.load_inputs_from_date_time_folder(
            date_time_folder=date_time_folder,
            date_time_root=date_time_root,
        )
        self.prefer_exts = prefer_exts
        self.use_calibration = use_calibration
        self.coord_cols = coord_cols
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.stitch_fraction = float(stitch_fraction)
        self.downsample = self._validate_downsample(downsample)
        self.max_canvas_gb = float(max_canvas_gb)
        self.use_pixel_drift_stitch = bool(use_pixel_drift_stitch)
        self._calibration_reported = False
        self.calibration = self._load_calibration()

        self.coord_data = self._load_coords(self.coord_file)
        self.time_stamps = self.coord_data[:, 0]
        self.scan_area_window = None
        self.scan_area_source_session = None
        self.scan_area_source_attempt = None
        self.scan_area_source_json = None
        self.scan_area_window = self._resolve_scan_area_window()
        self.canvas = None
        self._streaming_canvas_path = None
        self._streaming_canvas_shape = None
        self.frame_catalog = self._build_frame_catalog()
        print(f"[ScanStitcher] Loaded {len(self.coord_data)} movement rows and {len(self.frame_catalog)} frames.")
        print(f"[ScanStitcher] Stitch mode: {'pixel_drift' if self.use_pixel_drift_stitch else 'projection'}")
        print(f"[ScanStitcher] Stitch downsample: 1/{self.downsample} (frame decimation and final-tile decimation)")
        print(f"[ScanStitcher] Source file: {os.path.abspath(__file__)}")

    @staticmethod
    def _validate_downsample(downsample):
        try:
            value = int(downsample)
        except (TypeError, ValueError) as exc:
            raise ValueError("downsample must be a positive integer.") from exc
        if value < 1:
            raise ValueError("downsample must be >= 1.")
        return value

    def _resolve_date_time_folder(self, date_time_folder, date_time_root=STANDARD_DATA_ROOT):
        if os.path.isabs(date_time_folder):
            if os.path.isdir(date_time_folder):
                return os.path.normpath(date_time_folder)
            raise FileNotFoundError(f"date_time folder was not found: {date_time_folder}")

        candidates = [
            os.path.join(date_time_root, date_time_folder),
        ]

        for candidate in candidates:
            resolved = os.path.normpath(candidate)
            if os.path.isdir(resolved):
                return resolved

        searched = "\n  - ".join(os.path.normpath(c) for c in candidates)
        raise FileNotFoundError(
            f"Could not resolve date_time folder '{date_time_folder}'. Searched:\n  - {searched}"
        )

    def load_inputs_from_date_time_folder(self, date_time_folder, date_time_root=STANDARD_DATA_ROOT):
        dataset_dir = self._resolve_date_time_folder(
            date_time_folder=date_time_folder,
            date_time_root=date_time_root,
        )

        movement_path = None
        for name in ("cv_movement_recording.csv", "movement_recording.csv"):
            candidate = os.path.join(dataset_dir, name)
            if os.path.exists(candidate):
                movement_path = candidate
                break
        if movement_path is None:
            raise FileNotFoundError(
                f"Missing movement recording in {dataset_dir}. "
                "Expected cv_movement_recording.csv or movement_recording.csv."
            )

        photo_dir = os.path.join(dataset_dir, "camera_frames")
        if not os.path.isdir(photo_dir):
            raise FileNotFoundError(f"Missing camera_frames folder in dataset directory: {dataset_dir}")

        calibration_file = os.path.join(dataset_dir, "calibration.json")

        self.dataset_dir = dataset_dir
        self.dataset_folder_name = os.path.basename(os.path.normpath(dataset_dir))
        self.coord_file = movement_path
        self.photo_dir = photo_dir
        self.calibration_file = calibration_file
        print(f"[ScanStitcher] Using dataset folder: {dataset_dir}")
        print(f"[ScanStitcher] movement file: {self.coord_file}")
        print(f"[ScanStitcher] camera frames: {self.photo_dir}")
        print(f"[ScanStitcher] calibration file: {self.calibration_file}")
        return dataset_dir

    def _dataset_output_dir(self):
        return os.path.dirname(os.path.abspath(self.photo_dir))

    def _load_calibration(self):
        if not self.use_calibration:
            return None
        if not os.path.exists(self.calibration_file):
            raise FileNotFoundError(
                f"use_calibration=True but calibration file was not found: {self.calibration_file}"
            )

        with open(self.calibration_file, "r") as f:
            payload = json.load(f)

        if "stage" not in payload:
            raise ValueError(
                f"use_calibration=True but 'stage' key is missing in calibration file: {self.calibration_file}"
            )

        stage = payload.get("stage", {})
        M = np.asarray(stage.get("M", []), dtype=float)
        r0 = np.asarray(stage.get("r0", [0, 0]), dtype=float)
        if M.size == 0:
            raise ValueError(
                f"use_calibration=True but stage matrix M is empty in: {self.calibration_file}"
            )

        print(f"[ScanStitcher] Stage calibration enabled from: {self.calibration_file}")
        print(f"[ScanStitcher] stage.M shape={M.shape}, stage.r0={r0.tolist()}")
        return payload

    def _apply_calibration(self, x, y):
        if self.calibration is None:
            return x, y

        manip = self.calibration.get("stage", {})
        M = np.asarray(manip.get("M", []), dtype=float)
        r0 = np.asarray(manip.get("r0", [0, 0]), dtype=float)

        if M.size == 0:
            return x, y

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        pts = np.column_stack((x - r0[0], y - r0[1]))

        if M.shape == (2, 2):
            mapped = pts @ M.T
        elif M.shape == (2, 3):
            mapped = np.column_stack((x - r0[0], y - r0[1], np.ones_like(x))) @ M.T
        elif M.shape == (3, 3):
            mapped = np.column_stack((x - r0[0], y - r0[1], np.ones_like(x))) @ M.T
            mapped = mapped[:, :2]
        else:
            return x, y

        if not self._calibration_reported and mapped.shape[0] > 0:
            dx = mapped[:, 0] - x
            dy = mapped[:, 1] - y
            mad_x = float(np.mean(np.abs(dx)))
            mad_y = float(np.mean(np.abs(dy)))
            print(
                "[ScanStitcher] Calibration applied. "
                f"Mean absolute delta: dx={mad_x:.4f}, dy={mad_y:.4f}"
            )
            self._calibration_reported = True

        return mapped[:, 0], mapped[:, 1]

    def _aligned_coords_from_raw(self, coords):
        coords_arr = np.asarray(coords, dtype=float)
        if coords_arr.ndim != 2 or coords_arr.shape[1] < 2:
            raise ValueError("coords must be an array-like Nx2 of x, y positions.")
        x = coords_arr[:, 0]
        y = coords_arr[:, 1]
        x, y = self._apply_calibration(x, y)
        if self.flip_x:
            x = -x
        if self.flip_y:
            y = -y
        return np.column_stack((x, y))

    def _to_pixels(self, x, y):
        if self.calibration is None:
            return x, y

        manip = self.calibration.get("stage", {})
        M = np.asarray(manip.get("M", []), dtype=float)
        r0 = np.asarray(manip.get("r0", [0, 0]), dtype=float)
        if M.size == 0:
            return x, y

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        pts = np.column_stack((x, y))

        M_h = np.eye(3)
        if M.shape == (2, 2):
            M_h[:2, :2] = M
        elif M.shape == (2, 3):
            M_h[:2, :3] = M
        elif M.shape == (3, 3):
            M_h = M
        else:
            return x, y

        rhs = np.column_stack((pts - r0[:2], np.ones((pts.shape[0], 1))))
        mapped = rhs @ np.linalg.inv(M_h).T

        return mapped[:, 0], mapped[:, 1]

    def _load_coords(self, coord_path):
        print(f"[ScanStitcher] Loading movement coordinates from: {coord_path}")
        ext = os.path.splitext(coord_path)[1].lower()
        if ext == ".csv":
            rows = []

            with open(coord_path, "r") as f:
                for line in f:
                    clean_line = line.strip()
                    if not clean_line:
                        continue

                    if ";" in clean_line:
                        fields = [v.strip() for v in clean_line.split(";")]
                    else:
                        fields = [v.strip() for v in clean_line.split(",")]

                    try:
                        rows.append([float(v) for v in fields if v != ""])
                    except ValueError:
                        continue

            out = np.array(rows, dtype=float)
            print(f"[ScanStitcher] Parsed {len(out)} movement rows from CSV.")
            return out
        out = pd.read_excel(coord_path).values
        print(f"[ScanStitcher] Parsed {len(out)} movement rows from spreadsheet.")
        return out

    @staticmethod
    def _parse_session_folder_datetime(folder_name):
        try:
            return datetime.strptime(folder_name, "%Y_%m_%d-%H_%M")
        except ValueError:
            return None

    @staticmethod
    def _parse_attempt_number(attempt_name):
        if not attempt_name.startswith("attempt_"):
            return None
        suffix = attempt_name.split("_", 1)[1]
        if not suffix.isdigit():
            return None
        return int(suffix)

    @staticmethod
    def _load_scan_area_window_from_json(json_path):
        stem = os.path.splitext(os.path.basename(json_path))[0]
        parts = stem.split("_")
        if len(parts) < 3:
            return None
        state_name = "_".join(parts[1:-1]).lower()
        if state_name != "scan_area":
            return None

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

        started = payload.get("started")
        finished = payload.get("finished")
        if started is None or finished is None:
            return None
        try:
            started = float(started)
            finished = float(finished)
        except (TypeError, ValueError):
            return None
        if finished <= started:
            return None
        return started, finished

    def _select_scan_area_attempt_in_session(self, session_dir):
        attempt_windows = []
        for attempt_name in sorted(os.listdir(session_dir)):
            attempt_path = os.path.join(session_dir, attempt_name)
            if not os.path.isdir(attempt_path):
                continue
            attempt_num = self._parse_attempt_number(attempt_name)
            if attempt_num is None:
                continue

            scan_entries = []
            for json_path in sorted(glob.glob(os.path.join(attempt_path, "*.json"))):
                window = self._load_scan_area_window_from_json(json_path)
                if window is None:
                    continue
                scan_entries.append((window[0], window[1], json_path))

            if not scan_entries:
                continue

            # If multiple scan_area JSON files exist in one attempt, use the latest one.
            scan_entries.sort(key=lambda item: item[0])
            started, finished, json_path = scan_entries[-1]
            attempt_windows.append((attempt_num, started, finished, json_path))

        if not attempt_windows:
            return None

        # If multiple attempts have scan_area logs, use the highest attempt number.
        attempt_windows.sort(key=lambda item: item[0])
        return attempt_windows[-1]

    def _resolve_scan_area_window(self):
        rig_data_root = os.path.dirname(os.path.abspath(self.dataset_dir))
        data_root = os.path.dirname(rig_data_root)
        state_root = os.path.join(data_root, "state_recorder_data")
        if not os.path.isdir(state_root):
            print(f"[ScanStitcher] state_recorder_data not found at {state_root}; frame timestamps will not be cropped.")
            return None

        rig_folder = self.dataset_folder_name
        day_token = rig_folder.split("-", 1)[0]
        movement_ts = np.asarray(self.coord_data[:, 0], dtype=float)
        valid_ts = movement_ts[np.isfinite(movement_ts)]
        if valid_ts.size == 0:
            print("[ScanStitcher] No valid movement timestamps; frame timestamps will not be cropped.")
            return None
        movement_start = float(np.min(valid_ts))
        movement_end = float(np.max(valid_ts))
        movement_center = 0.5 * (movement_start + movement_end)
        rig_dt = self._parse_session_folder_datetime(rig_folder)

        session_candidates = []
        for session_name in sorted(os.listdir(state_root)):
            session_dir = os.path.join(state_root, session_name)
            if not os.path.isdir(session_dir):
                continue
            if not session_name.startswith(day_token):
                continue

            selected = self._select_scan_area_attempt_in_session(session_dir)
            if selected is None:
                continue
            attempt_num, started, finished, json_path = selected
            overlap = max(0.0, min(movement_end, finished) - max(movement_start, started))
            window_center = 0.5 * (started + finished)
            center_delta = abs(window_center - movement_center)
            session_dt = self._parse_session_folder_datetime(session_name)
            if session_dt is not None and rig_dt is not None:
                folder_delta = abs((session_dt - rig_dt).total_seconds())
            else:
                folder_delta = float("inf")
            exact_folder = int(session_name == rig_folder)
            session_candidates.append(
                (
                    exact_folder,
                    overlap,
                    center_delta,
                    folder_delta,
                    attempt_num,
                    session_name,
                    started,
                    finished,
                    json_path,
                )
            )

        if not session_candidates:
            print(
                f"[ScanStitcher] No scan_area state JSON found for day '{day_token}' in {state_root}; "
                "frame timestamps will not be cropped."
            )
            return None

        session_candidates.sort(
            key=lambda item: (
                -item[0],   # exact session-name match first
                -item[1],   # then maximize overlap with movement timestamps
                item[2],    # then closest window center
                item[3],    # then closest folder datetime
                -item[4],   # then highest attempt number
                item[5],    # deterministic tie-break
            )
        )
        best = session_candidates[0]
        self.scan_area_source_session = best[5]
        self.scan_area_source_attempt = int(best[4])
        self.scan_area_source_json = best[8]
        print(
            "[ScanStitcher] Using scan_area window from state recorder: "
            f"session={self.scan_area_source_session}, attempt={self.scan_area_source_attempt}, "
            f"json={os.path.basename(self.scan_area_source_json)}, "
            f"start={best[6]:.6f}, end={best[7]:.6f}"
        )
        return float(best[6]), float(best[7])

    def _build_frame_catalog(self):
        print(f"[ScanStitcher] Indexing frames in: {self.photo_dir}")
        records = []
        skipped_outside_window = 0
        prefer_exts = tuple(ext.lower() for ext in self.prefer_exts)
        time_window = self.scan_area_window
        if time_window is not None:
            print(
                "[ScanStitcher] Cropping frames by scan_area timestamps: "
                f"{time_window[0]:.6f} <= t <= {time_window[1]:.6f}"
            )
        all_files = sorted(glob.glob(os.path.join(self.photo_dir, "*")), key=lambda p: self._camera_order_key(os.path.basename(p)))
        for frame_path in tqdm(all_files, desc="Indexing frame timestamps", unit="file"):
            if not os.path.isfile(frame_path):
                continue
            if os.path.splitext(frame_path)[1].lower() not in prefer_exts:
                continue
            ts = self._extract_timestamp_from_name(frame_path)
            if ts is None:
                continue
            if time_window is not None and not (time_window[0] <= ts <= time_window[1]):
                skipped_outside_window += 1
                continue
            records.append((ts, frame_path))

        if not records:
            if time_window is not None:
                raise FileNotFoundError(
                    "No images found within the scan_area timestamp window "
                    f"[{time_window[0]:.6f}, {time_window[1]:.6f}] in {self.photo_dir} "
                    f"(source JSON: {self.scan_area_source_json})."
                )
            raise FileNotFoundError(f"No images found in {self.photo_dir} for extensions {self.prefer_exts}")

        print(f"[ScanStitcher] Indexed {len(records)} timestamped frames.")
        if time_window is not None:
            print(f"[ScanStitcher] Skipped {skipped_outside_window} frame(s) outside scan_area time window.")
        return sorted(records, key=lambda item: item[0])

    @staticmethod
    def _camera_order_key(name):
        underscore_index = name.find('_')
        dot_index = name.rfind('.')
        if underscore_index == -1:
            return name
        segment = name[underscore_index + 1 : dot_index if dot_index != -1 else None]
        if segment and segment.replace('.', '', 1).isdigit():
            segment = segment.zfill(3)
        suffix = name[dot_index:] if dot_index != -1 else ''
        return f"{name[:underscore_index + 1]}{segment}{suffix}"

    def _extract_timestamp_from_name(self, frame_path):
        filename = os.path.basename(frame_path)
        underscore_index = filename.find("_")
        last_period_index = filename.rfind(".")
        if underscore_index == -1:
            return None
        segment = filename[underscore_index + 1 : last_period_index if last_period_index != -1 else None]
        try:
            return float(segment)
        except (TypeError, ValueError):
            return None

    def _align_all_frames_to_movement(self):
        print("[ScanStitcher] Aligning all frames to nearest movement timestamp...")
        if self.coord_data.size == 0:
            raise ValueError("movement data is empty.")
        if len(self.frame_catalog) == 0:
            raise ValueError("frame catalog is empty.")

        movement = np.asarray(self.coord_data, dtype=float)
        col_x, col_y = self.coord_cols
        if movement.shape[1] <= max(0, col_x, col_y):
            raise ValueError("movement data does not include requested coord_cols.")

        order = np.argsort(movement[:, 0])
        movement = movement[order]
        m_ts = movement[:, 0]
        frame_ts = np.asarray([ts for ts, _ in self.frame_catalog], dtype=float)
        frame_paths = [path for _, path in self.frame_catalog]

        right = np.searchsorted(m_ts, frame_ts)
        left = np.clip(right - 1, 0, len(m_ts) - 1)
        right = np.clip(right, 0, len(m_ts) - 1)
        choose_right = np.abs(m_ts[right] - frame_ts) < np.abs(m_ts[left] - frame_ts)
        indices = np.where(choose_right, right, left)
        matched = movement[indices]
        coords = matched[:, [col_x, col_y]]
        print(f"[ScanStitcher] Aligned {len(frame_ts)} frames to movement samples.")
        return frame_ts, coords, frame_paths

    def _sample_aligned_data(self, timestamps, coords, frame_paths):
        n = len(timestamps)
        if n == 0:
            return timestamps, coords, frame_paths
        frac = min(max(self.stitch_fraction, 0.0), 1.0)
        if frac >= 1.0:
            print(f"[ScanStitcher] stitch_fraction={frac:.3f}; using all {n} frames.")
            return timestamps, coords, frame_paths
        keep = max(1, int(np.ceil(n * frac)))
        print(f"[ScanStitcher] stitch_fraction={frac:.3f}; using first {keep}/{n} frames for stitching.")
        return timestamps[:keep], coords[:keep], frame_paths[:keep]

    def _downsample_aligned_data(self, timestamps, coords, frame_paths):
        n = len(timestamps)
        if n == 0:
            return timestamps, coords, frame_paths
        if self.downsample <= 1:
            print(
                f"[ScanStitcher] downsample=1/{self.downsample}; "
                f"using all {n}/{n} frames (no decimation)."
            )
            return timestamps, coords, frame_paths

        indices = np.arange(0, n, self.downsample, dtype=int)
        sampled_timestamps = np.asarray(timestamps)[indices]
        sampled_coords = np.asarray(coords)[indices]
        sampled_paths = [frame_paths[int(i)] for i in indices]
        print(
            f"[ScanStitcher] downsample=1/{self.downsample}; "
            f"using {len(indices)}/{n} frames for stitching."
        )
        return sampled_timestamps, sampled_coords, sampled_paths

    def _pixel_drift_draw_order(self, coordinates):
        """
        Build frame draw order for pixel-drift stitching.
        - Horizontal-major scans:
          - Left-to-right rows keep original order (later frames over earlier).
          - Right-to-left rows reverse within the row (later frames under earlier).
        - Vertical-major scans:
          - All columns reverse within the column (later frames under earlier),
            regardless of top-to-bottom or bottom-to-top motion.
        """
        coords_arr = np.asarray(coordinates, dtype=float)
        if coords_arr.ndim != 2 or coords_arr.shape[1] < 2:
            raise ValueError("coordinates must be an array-like Nx2 of x, y positions.")
        n = coords_arr.shape[0]
        if n == 0:
            return np.array([], dtype=int)

        step_x = np.abs(np.diff(coords_arr[:, 0]))
        step_y = np.abs(np.diff(coords_arr[:, 1]))
        nonzero_dx = step_x[step_x > 0]
        nonzero_dy = step_y[step_y > 0]
        typical_dx = float(np.median(nonzero_dx)) if nonzero_dx.size else 0.0
        typical_dy = float(np.median(nonzero_dy)) if nonzero_dy.size else 0.0
        horizontal_major = typical_dx >= typical_dy

        if horizontal_major:
            primary_steps = step_x
            secondary_steps = step_y
            secondary_jitter_pool = secondary_steps[secondary_steps <= primary_steps]
            if secondary_jitter_pool.size == 0:
                secondary_jitter_pool = secondary_steps
            secondary_jitter = (
                float(np.median(secondary_jitter_pool)) if secondary_jitter_pool.size else 0.0
            )
            break_ratio = 3.0
            min_secondary_jump = max(2.0, secondary_jitter * 8.0, typical_dx * 0.4)
            direction_axis = 0
            reverse_label = "right-to-left row(s)"
        else:
            primary_steps = step_y
            secondary_steps = step_x
            secondary_jitter_pool = secondary_steps[secondary_steps <= primary_steps]
            if secondary_jitter_pool.size == 0:
                secondary_jitter_pool = secondary_steps
            secondary_jitter = (
                float(np.median(secondary_jitter_pool)) if secondary_jitter_pool.size else 0.0
            )
            # Slightly looser than horizontal mode so vertical serpentine turns are still detected.
            break_ratio = 1.25
            min_secondary_jump = max(2.0, secondary_jitter * 8.0, typical_dy * 0.4)
            direction_axis = 1
            reverse_label = "column(s)"

        row_starts = [0]
        for i in range(1, n):
            primary = primary_steps[i - 1]
            secondary = secondary_steps[i - 1]
            if secondary > (primary * break_ratio) and secondary > min_secondary_jump:
                row_starts.append(i)
        row_starts.append(n)

        draw_order = []
        reversed_segments = 0
        for start, end in zip(row_starts[:-1], row_starts[1:]):
            if end - start <= 1:
                draw_order.extend(range(start, end))
                continue
            if horizontal_major:
                reverse_segment = coords_arr[end - 1, direction_axis] < coords_arr[start, direction_axis]
            else:
                reverse_segment = True
            if reverse_segment:
                reversed_segments += 1
                draw_order.extend(range(end - 1, start - 1, -1))
            else:
                draw_order.extend(range(start, end))

        moved_boundary_entries = 0
        if horizontal_major:
            # Ensure the first frame after each vertical jump is under the prior horizontal frame.
            for start in row_starts[1:-1]:
                prev_idx = int(start - 1)
                start_idx = int(start)
                try:
                    pos_start = draw_order.index(start_idx)
                    pos_prev = draw_order.index(prev_idx)
                except ValueError:
                    continue
                if pos_start > pos_prev:
                    frame_id = draw_order.pop(pos_start)
                    draw_order.insert(pos_prev, frame_id)
                    moved_boundary_entries += 1

        if reversed_segments > 0:
            print(
                "[ScanStitcher] Applied underlay ordering for "
                f"{reversed_segments} {reverse_label}."
            )
        if moved_boundary_entries > 0:
            print(
                "[ScanStitcher] Applied boundary underlay for "
                f"{moved_boundary_entries} vertical-entry frame(s)."
            )

        return np.asarray(draw_order, dtype=int)

    def _build_timestamp_aligned_frame_paths(self, target_timestamps):
        """Align camera frames to timestamps using local-window nearest matching."""
        target_ts = np.asarray(target_timestamps, dtype=float).reshape(-1)
        if target_ts.size == 0:
            raise ValueError("target_timestamps is empty.")

        frame_times = np.array([ts for ts, _ in self.frame_catalog], dtype=float)
        if frame_times.size == 0:
            raise FileNotFoundError(f"No indexed frames available in {self.photo_dir}.")

        aligned_paths = []
        last_index = 0

        for i, target_timestamp in enumerate(target_ts):
            if target_ts.size == 1:
                timestamp_range = float("inf")
            elif i == target_ts.size - 1:
                timestamp_range = abs(target_timestamp - target_ts[i - 1])
            else:
                timestamp_range = abs(target_timestamp - target_ts[i + 1])

            if timestamp_range <= 0:
                timestamp_range = np.finfo(float).eps

            valid_indices = []
            valid_timestamps = []
            for j in range(last_index, len(frame_times)):
                camera_timestamp = frame_times[j]
                if abs(target_timestamp - camera_timestamp) < timestamp_range:
                    valid_indices.append(j)
                    valid_timestamps.append(camera_timestamp)
                elif valid_indices:
                    break

            if not valid_indices:
                raise RuntimeError(
                    f"No camera frame matched timestamp {target_timestamp} in {self.photo_dir}."
                )

            min_idx = valid_indices[0]
            min_diff = float("inf")
            for idx, ts in zip(valid_indices, valid_timestamps):
                diff = abs(target_timestamp - ts)
                if diff < min_diff:
                    min_diff = diff
                    min_idx = idx

            aligned_paths.append(self.frame_catalog[min_idx][1])
            last_index = max(0, min_idx - 1)

        return aligned_paths

    def _find_frame(self, target_time):
        aligned = self._build_timestamp_aligned_frame_paths([target_time])
        return aligned[0]

    def _to_bgr(self, image_path):
        ext = os.path.splitext(image_path)[1].lower()
        frame = cv2.imread(image_path)
        if frame is None:
            raise RuntimeError(f"Unable to decode image: {image_path}")

        if ext == ".webp":
            success, jpg_bytes = cv2.imencode(".jpg", frame)
            if not success:
                raise RuntimeError(f"Failed to convert WebP to JPEG in-memory: {image_path}")
            frame = cv2.imdecode(jpg_bytes, cv2.IMREAD_COLOR)

        return frame

    def _to_rgb(self, image_path):
        return cv2.cvtColor(self._to_bgr(image_path), cv2.COLOR_BGR2RGB)

    def _load_frame_rgb(self, target_time):
        image_path = self._find_frame(target_time)
        return self._to_rgb(image_path)

    def save_projection_image_from_aligned_data(
        self,
        aligned_timestamps,
        aligned_coordinates,
        aligned_frame_paths,
        output_path="projection_image.png",
        coordinate_scale=1.0,
        frame_padding=0,
    ):
        """Build projection_image.png from already timestamp-aligned data."""
        timestamps_arr = np.asarray(aligned_timestamps, dtype=float).reshape(-1)
        coords_arr = np.asarray(aligned_coordinates, dtype=float)
        frame_paths = list(aligned_frame_paths)

        if coords_arr.ndim != 2 or coords_arr.shape[1] < 2:
            raise ValueError("aligned_coordinates must be an array-like Nx2 of x, y positions.")
        if len(timestamps_arr) != len(coords_arr):
            raise ValueError("aligned_timestamps and aligned_coordinates must have the same number of rows.")
        if len(frame_paths) != len(coords_arr):
            raise ValueError("aligned_frame_paths and aligned_coordinates must have the same length.")
        if len(frame_paths) == 0:
            raise ValueError("aligned_frame_paths is empty.")

        print(f"[ScanStitcher] Loading {len(frame_paths)} aligned frames for projection image...")
        camera_frames = [
            self._to_rgb(path)
            for path in tqdm(frame_paths, desc="Loading aligned frames", unit="frame")
        ]
        return self.save_projection_image(
            timestamps=timestamps_arr,
            coordinates=coords_arr[:, :2],
            camera_frames=camera_frames,
            output_path=output_path,
            coordinate_scale=coordinate_scale,
            frame_padding=frame_padding,
        )

    def save_projection_image(
        self,
        timestamps,
        coordinates,
        camera_frames,
        output_path=None,
        coordinate_scale=1.0,
        frame_padding=0,
    ):
        """Save a non-overlapping, block-based projection from timestamp-aligned frames."""
        print("[ScanStitcher] Creating projection canvas...")
        canvas = self._create_projection_canvas(
            timestamps=timestamps,
            coordinates=coordinates,
            camera_frames=camera_frames,
            coordinate_scale=coordinate_scale,
            frame_padding=frame_padding,
        )
        if output_path is None:
            output_path = os.path.join(self._dataset_output_dir(), "projection_image.tif")
        elif not os.path.isabs(output_path):
            output_path = os.path.join(self._dataset_output_dir(), output_path)
        output_path = os.path.normpath(output_path)
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        print(f"[ScanStitcher] Writing projection image: {output_path}")
        if not cv2.imwrite(output_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Failed to write projection image: {output_path}")
        return output_path

    def _create_projection_canvas(
        self,
        timestamps,
        coordinates,
        camera_frames,
        coordinate_scale=1.0,
        frame_padding=0,
        image_downsample=1,
    ):
        print("[ScanStitcher] Building non-overlapping projection canvas...")
        timestamps_arr = np.asarray(timestamps, dtype=float).reshape(-1)
        coords_arr = np.asarray(coordinates, dtype=float)
        frames_list = list(camera_frames)

        if coords_arr.ndim != 2 or coords_arr.shape[1] < 2:
            raise ValueError("coordinates must be an array-like Nx2 of x, y positions.")
        if len(timestamps_arr) != len(coords_arr):
            raise ValueError("timestamps and coordinates must have the same number of rows.")
        if len(frames_list) != len(coords_arr):
            raise ValueError("camera_frames and coordinates must have the same length.")
        if len(frames_list) == 0:
            raise ValueError("camera_frames is empty.")

        valid_xy = np.isfinite(coords_arr[:, 0]) & np.isfinite(coords_arr[:, 1])
        if not np.all(valid_xy):
            timestamps_arr = timestamps_arr[valid_xy]
            coords_arr = coords_arr[valid_xy]
            frames_list = [frame for frame, keep in zip(frames_list, valid_xy) if keep]

        if len(frames_list) == 0:
            raise ValueError("No valid entries after filtering non-finite coordinates.")

        sample = np.asarray(frames_list[0])
        if sample.ndim == 2:
            sample = np.repeat(sample[:, :, None], 3, axis=2)
        elif sample.ndim != 3:
            raise ValueError("camera frames must be 2D grayscale or 3-channel arrays.")
        if sample.shape[2] < 3:
            raise ValueError("camera frames must contain 3 channels.")
        sample = sample[:, :, :3]
        tile_h, tile_w = sample.shape[:2]
        image_ds = self._validate_downsample(image_downsample)
        resize_interp = cv2.INTER_AREA if image_ds > 1 else cv2.INTER_LINEAR
        if image_ds > 1:
            src_h, src_w = tile_h, tile_w
            tile_h = max(1, int(np.ceil(src_h / float(image_ds))))
            tile_w = max(1, int(np.ceil(src_w / float(image_ds))))
            sample = cv2.resize(sample, (tile_w, tile_h), interpolation=resize_interp)
            print(
                "[ScanStitcher] Projection tile decimation: "
                f"1/{image_ds} ({src_w}x{src_h} -> {tile_w}x{tile_h})."
            )

        stride_x = max(1, tile_w + int(max(frame_padding, 0)))
        stride_y = max(1, tile_h + int(max(frame_padding, 0)))

        scale = float(coordinate_scale)
        if scale <= 0:
            raise ValueError("coordinate_scale must be > 0.")

        x = (coords_arr[:, 0] - np.min(coords_arr[:, 0])) * scale
        y = (coords_arr[:, 1] - np.min(coords_arr[:, 1])) * scale
        gx = np.rint(x).astype(int)
        gy = np.rint(y).astype(int)
        gx -= int(np.min(gx))
        gy -= int(np.min(gy))

        canvas_h = int(np.max(gy) + 1) * stride_y + tile_h
        canvas_w = int(np.max(gx) + 1) * stride_x + tile_w
        estimated_bytes = int(canvas_h) * int(canvas_w) * 3
        max_bytes = max(1, int(self.max_canvas_gb * (1024 ** 3)))
        if estimated_bytes > max_bytes:
            shrink = np.sqrt(max_bytes / float(estimated_bytes))
            gx = np.floor(gx * shrink).astype(int)
            gy = np.floor(gy * shrink).astype(int)
            gx -= int(np.min(gx))
            gy -= int(np.min(gy))
            canvas_h = int(np.max(gy) + 1) * stride_y + tile_h
            canvas_w = int(np.max(gx) + 1) * stride_x + tile_w
            estimated_bytes = int(canvas_h) * int(canvas_w) * 3
            print(
                "[ScanStitcher] Canvas too large; auto-downscaled grid "
                f"(shrink={shrink:.4f}, est={estimated_bytes / (1024 ** 3):.2f} GiB)."
            )
        else:
            print(f"[ScanStitcher] Canvas estimate: {estimated_bytes / (1024 ** 3):.2f} GiB.")

        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        occupied = set()

        for idx, frame in enumerate(tqdm(frames_list, desc="Placing frames on canvas", unit="frame")):
            frame_rgb = np.asarray(frame)
            if frame_rgb.ndim == 2:
                frame_rgb = np.repeat(frame_rgb[:, :, None], 3, axis=2)
            elif frame_rgb.ndim != 3 or frame_rgb.shape[2] < 3:
                continue
            frame_rgb = frame_rgb[:, :, :3]
            if frame_rgb.shape[0] == 0 or frame_rgb.shape[1] == 0:
                continue
            if frame_rgb.shape[:2] != (tile_h, tile_w):
                frame_rgb = cv2.resize(frame_rgb, (tile_w, tile_h), interpolation=resize_interp)

            frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)

            left = int(gx[idx]) * stride_x
            top = int(gy[idx]) * stride_y
            while (top, left) in occupied:
                left += stride_x

            right = left + tile_w
            bottom = top + tile_h
            if right > canvas.shape[1]:
                canvas = np.concatenate(
                    [canvas, np.zeros((canvas.shape[0], right - canvas.shape[1], 3), dtype=np.uint8)],
                    axis=1,
                )
            if bottom > canvas.shape[0]:
                canvas = np.concatenate(
                    [canvas, np.zeros((bottom - canvas.shape[0], canvas.shape[1], 3), dtype=np.uint8)],
                    axis=0,
                )

            canvas[top:bottom, left:right] = frame_rgb
            occupied.add((top, left))
        print("[ScanStitcher] Canvas build complete.")
        return canvas

    def _create_pixel_drift_canvas(self, coordinates, camera_frames, image_downsample=1):
        print("[ScanStitcher] Building pixel-drift stitched canvas...")
        coords_arr = np.asarray(coordinates, dtype=float)
        frames_list = list(camera_frames)
        if coords_arr.ndim != 2 or coords_arr.shape[1] < 2:
            raise ValueError("coordinates must be an array-like Nx2 of x, y positions.")
        if len(frames_list) != len(coords_arr):
            raise ValueError("camera_frames and coordinates must have the same length.")
        if len(frames_list) == 0:
            raise ValueError("camera_frames is empty.")

        sample = np.asarray(frames_list[0])
        if sample.ndim == 2:
            sample = np.repeat(sample[:, :, None], 3, axis=2)
        elif sample.ndim != 3 or sample.shape[2] < 3:
            raise ValueError("camera frames must be 2D grayscale or 3-channel arrays.")
        sample = sample[:, :, :3]
        tile_h, tile_w = sample.shape[:2]
        image_ds = self._validate_downsample(image_downsample)
        resize_interp = cv2.INTER_AREA if image_ds > 1 else cv2.INTER_LINEAR
        if image_ds > 1:
            src_h, src_w = tile_h, tile_w
            tile_h = max(1, int(np.ceil(src_h / float(image_ds))))
            tile_w = max(1, int(np.ceil(src_w / float(image_ds))))
            sample = cv2.resize(sample, (tile_w, tile_h), interpolation=resize_interp)
            print(
                "[ScanStitcher] Drift tile decimation: "
                f"1/{image_ds} ({src_w}x{src_h} -> {tile_w}x{tile_h})."
            )

        # Zero at first point so drift is relative to acquisition start.
        drift = coords_arr[:, :2] - coords_arr[0, :2]
        dx = np.rint(drift[:, 0] / float(image_ds)).astype(int)
        dy = np.rint(drift[:, 1] / float(image_ds)).astype(int)

        min_dx = int(np.min(dx))
        min_dy = int(np.min(dy))
        max_dx = int(np.max(dx))
        max_dy = int(np.max(dy))
        canvas_w = (max_dx - min_dx) + tile_w
        canvas_h = (max_dy - min_dy) + tile_h

        estimated_bytes = int(canvas_h) * int(canvas_w) * 3
        max_bytes = max(1, int(self.max_canvas_gb * (1024 ** 3)))
        if estimated_bytes > max_bytes:
            shrink = np.sqrt(max_bytes / float(estimated_bytes))
            dx = np.floor(dx * shrink).astype(int)
            dy = np.floor(dy * shrink).astype(int)
            min_dx = int(np.min(dx))
            min_dy = int(np.min(dy))
            max_dx = int(np.max(dx))
            max_dy = int(np.max(dy))
            canvas_w = (max_dx - min_dx) + tile_w
            canvas_h = (max_dy - min_dy) + tile_h
            estimated_bytes = int(canvas_h) * int(canvas_w) * 3
            print(
                "[ScanStitcher] Drift canvas too large; auto-downscaled offsets "
                f"(shrink={shrink:.4f}, est={estimated_bytes / (1024 ** 3):.2f} GiB)."
            )
        else:
            print(f"[ScanStitcher] Drift canvas estimate: {estimated_bytes / (1024 ** 3):.2f} GiB.")

        draw_order = self._pixel_drift_draw_order(coords_arr)
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        for draw_idx in tqdm(draw_order, desc="Blending frames by drift", unit="frame"):
            frame_rgb = np.asarray(frames_list[int(draw_idx)])
            if frame_rgb.ndim == 2:
                frame_rgb = np.repeat(frame_rgb[:, :, None], 3, axis=2)
            elif frame_rgb.ndim != 3 or frame_rgb.shape[2] < 3:
                continue
            frame_rgb = frame_rgb[:, :, :3]
            if frame_rgb.shape[:2] != (tile_h, tile_w):
                frame_rgb = cv2.resize(frame_rgb, (tile_w, tile_h), interpolation=resize_interp)
            frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)

            left = int(dx[int(draw_idx)] - min_dx)
            top = int(dy[int(draw_idx)] - min_dy)
            right = left + tile_w
            bottom = top + tile_h
            canvas[top:bottom, left:right] = frame_rgb

        print("[ScanStitcher] Pixel-drift canvas build complete.")
        return canvas

    def stitch_streaming(
        self,
        output_path=None,
        metadata_path=None,
        memmap_path=None,
        cleanup_memmap=True,
    ):
        """
        Stream pixel-drift stitching into a disk-backed canvas.
        This avoids loading every frame into RAM at once.
        """
        if not self.use_pixel_drift_stitch:
            raise RuntimeError(
                "stitch_streaming currently supports only pixel-drift stitch mode."
            )

        print("[ScanStitcher] Starting streaming stitch pipeline...")
        frame_timestamps, coords, frame_paths = self._align_all_frames_to_movement()
        n_aligned = len(frame_paths)
        frame_timestamps, coords, frame_paths = self._sample_aligned_data(
            frame_timestamps, coords, frame_paths
        )
        n_sampled = len(frame_paths)
        frame_timestamps, coords, frame_paths = self._downsample_aligned_data(
            frame_timestamps, coords, frame_paths
        )
        n_downsampled = len(frame_paths)
        print(
            "[ScanStitcher] Streaming frame counts: "
            f"aligned={n_aligned}, sampled={n_sampled}, final={n_downsampled}."
        )
        aligned_coords = self._aligned_coords_from_raw(coords)
        if len(frame_paths) == 0:
            raise ValueError("No frames available for streaming stitch.")

        sample = self._to_bgr(frame_paths[0])
        if sample.ndim == 2:
            sample = np.repeat(sample[:, :, None], 3, axis=2)
        elif sample.ndim != 3 or sample.shape[2] < 3:
            raise ValueError("camera frames must be 2D grayscale or 3-channel arrays.")
        sample = sample[:, :, :3]
        tile_h, tile_w = sample.shape[:2]
        image_ds = self.downsample
        resize_interp = cv2.INTER_AREA if image_ds > 1 else cv2.INTER_LINEAR
        if image_ds > 1:
            src_h, src_w = tile_h, tile_w
            tile_h = max(1, int(np.ceil(src_h / float(image_ds))))
            tile_w = max(1, int(np.ceil(src_w / float(image_ds))))
            sample = cv2.resize(sample, (tile_w, tile_h), interpolation=resize_interp)
            print(
                "[ScanStitcher] Streaming drift tile decimation: "
                f"1/{image_ds} ({src_w}x{src_h} -> {tile_w}x{tile_h})."
            )

        # Zero at first point so drift is relative to acquisition start.
        drift = aligned_coords[:, :2] - aligned_coords[0, :2]
        dx = np.rint(drift[:, 0] / float(image_ds)).astype(int)
        dy = np.rint(drift[:, 1] / float(image_ds)).astype(int)

        min_dx = int(np.min(dx))
        min_dy = int(np.min(dy))
        max_dx = int(np.max(dx))
        max_dy = int(np.max(dy))
        canvas_w = (max_dx - min_dx) + tile_w
        canvas_h = (max_dy - min_dy) + tile_h

        estimated_bytes = int(canvas_h) * int(canvas_w) * 3
        max_bytes = max(1, int(self.max_canvas_gb * (1024 ** 3)))
        if estimated_bytes > max_bytes:
            shrink = np.sqrt(max_bytes / float(estimated_bytes))
            dx = np.floor(dx * shrink).astype(int)
            dy = np.floor(dy * shrink).astype(int)
            min_dx = int(np.min(dx))
            min_dy = int(np.min(dy))
            max_dx = int(np.max(dx))
            max_dy = int(np.max(dy))
            canvas_w = (max_dx - min_dx) + tile_w
            canvas_h = (max_dy - min_dy) + tile_h
            estimated_bytes = int(canvas_h) * int(canvas_w) * 3
            print(
                "[ScanStitcher] Drift canvas too large; auto-downscaled offsets "
                f"(shrink={shrink:.4f}, est={estimated_bytes / (1024 ** 3):.2f} GiB)."
            )
        else:
            print(f"[ScanStitcher] Drift canvas estimate: {estimated_bytes / (1024 ** 3):.2f} GiB.")

        if output_path is None:
            output_path = os.path.join(self._dataset_output_dir(), "stitched_scan.tif")
        elif not os.path.isabs(output_path):
            output_path = os.path.join(self._dataset_output_dir(), output_path)
        output_path = os.path.normpath(output_path)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if memmap_path is None:
            stem, _ = os.path.splitext(output_path)
            memmap_path = f"{stem}.memmap"
        elif not os.path.isabs(memmap_path):
            memmap_path = os.path.join(self._dataset_output_dir(), memmap_path)
        memmap_path = os.path.normpath(memmap_path)
        memmap_dir = os.path.dirname(memmap_path)
        if memmap_dir:
            os.makedirs(memmap_dir, exist_ok=True)

        draw_order = self._pixel_drift_draw_order(aligned_coords)
        print(f"[ScanStitcher] Allocating disk-backed canvas: {memmap_path}")
        canvas_mm = np.memmap(memmap_path, dtype=np.uint8, mode="w+", shape=(canvas_h, canvas_w, 3))
        canvas_mm[:] = 0

        for draw_idx in tqdm(draw_order, desc="Streaming stitch frames", unit="frame"):
            draw_idx = int(draw_idx)
            frame_bgr = sample if draw_idx == 0 else self._to_bgr(frame_paths[draw_idx])
            if frame_bgr.ndim == 2:
                frame_bgr = np.repeat(frame_bgr[:, :, None], 3, axis=2)
            elif frame_bgr.ndim != 3 or frame_bgr.shape[2] < 3:
                continue
            frame_bgr = frame_bgr[:, :, :3]
            if frame_bgr.shape[:2] != (tile_h, tile_w):
                frame_bgr = cv2.resize(frame_bgr, (tile_w, tile_h), interpolation=resize_interp)
            frame_bgr = np.clip(frame_bgr, 0, 255).astype(np.uint8)

            left = int(dx[draw_idx] - min_dx)
            top = int(dy[draw_idx] - min_dy)
            right = left + tile_w
            bottom = top + tile_h
            canvas_mm[top:bottom, left:right] = frame_bgr

        canvas_mm.flush()
        print(f"[ScanStitcher] Writing streamed stitched image: {output_path}")
        if not cv2.imwrite(output_path, canvas_mm):
            raise RuntimeError(f"Failed to write streamed stitched image: {output_path}")

        self.canvas = None
        self._streaming_canvas_path = memmap_path
        self._streaming_canvas_shape = tuple(canvas_mm.shape)

        if metadata_path is None and output_path:
            metadata_path = f"{os.path.splitext(output_path)[0]}_meta.json"
        if metadata_path is not None:
            if not os.path.isabs(metadata_path):
                metadata_path = os.path.join(self._dataset_output_dir(), metadata_path)
            metadata_path = os.path.normpath(metadata_path)
            metadata_dir = os.path.dirname(metadata_path)
            if metadata_dir:
                os.makedirs(metadata_dir, exist_ok=True)
            metadata = {
                "coord_file": self.coord_file,
                "photo_dir": self.photo_dir,
                "use_calibration": self.use_calibration,
                "flip_x": self.flip_x,
                "flip_y": self.flip_y,
                "use_pixel_drift_stitch": self.use_pixel_drift_stitch,
                "stitch_fraction": self.stitch_fraction,
                "downsample": self.downsample,
                "stitch_mode": "pixel_drift_streaming",
                "canvas_shape": self._streaming_canvas_shape,
                "num_frames": len(frame_paths),
                "streaming_memmap_path": memmap_path,
            }
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

        del canvas_mm
        if cleanup_memmap:
            try:
                os.remove(memmap_path)
                self._streaming_canvas_path = None
            except OSError as exc:
                print(f"[ScanStitcher] Warning: failed to remove memmap file {memmap_path}: {exc}")

        print("[ScanStitcher] Streaming stitch complete.")
        return output_path

    def stitch(self):
        print("[ScanStitcher] Starting stitch pipeline...")
        frame_timestamps, coords, frame_paths = self._align_all_frames_to_movement()
        n_aligned = len(frame_paths)
        frame_timestamps, coords, frame_paths = self._sample_aligned_data(
            frame_timestamps, coords, frame_paths
        )
        n_sampled = len(frame_paths)
        frame_timestamps, coords, frame_paths = self._downsample_aligned_data(
            frame_timestamps, coords, frame_paths
        )
        n_downsampled = len(frame_paths)
        print(
            "[ScanStitcher] Stitch frame counts: "
            f"aligned={n_aligned}, sampled={n_sampled}, final={n_downsampled}."
        )
        aligned_coords = self._aligned_coords_from_raw(coords)
        print(f"[ScanStitcher] Loading {len(frame_paths)} frames into memory...")
        camera_frames = [
            self._to_rgb(path)
            for path in tqdm(frame_paths, desc="Loading frames", unit="frame")
        ]
        if self.use_pixel_drift_stitch:
            self.canvas = self._create_pixel_drift_canvas(
                coordinates=aligned_coords,
                camera_frames=camera_frames,
                image_downsample=self.downsample,
            )
        else:
            self.canvas = self._create_projection_canvas(
                timestamps=frame_timestamps,
                coordinates=aligned_coords,
                camera_frames=camera_frames,
                coordinate_scale=1.0,
                frame_padding=0,
                image_downsample=self.downsample,
            )
        print("[ScanStitcher] Stitch complete.")
        return self.canvas

    def save_projection_from_dataset(self, output_path=None):
        print("[ScanStitcher] Starting projection-image pipeline...")
        frame_timestamps, coords, frame_paths = self._align_all_frames_to_movement()
        frame_timestamps, coords, frame_paths = self._sample_aligned_data(
            frame_timestamps, coords, frame_paths
        )
        aligned_coords = self._aligned_coords_from_raw(coords)
        if output_path is None:
            output_path = os.path.join(self._dataset_output_dir(), "projection_image.tif")
        elif not os.path.isabs(output_path):
            output_path = os.path.join(self._dataset_output_dir(), output_path)
        saved = self.save_projection_image_from_aligned_data(
            aligned_timestamps=frame_timestamps,
            aligned_coordinates=aligned_coords,
            aligned_frame_paths=frame_paths,
            output_path=output_path,
            coordinate_scale=1.0,
            frame_padding=0,
        )
        print(f"[ScanStitcher] Projection image saved: {saved}")
        return saved

    def save_horizontal_overlap_from_dataset(self, output_path=None, use_sampling=False):
        print("[ScanStitcher] Building horizontal-motion overlap image...")
        frame_timestamps, coords, frame_paths = self._align_all_frames_to_movement()
        if use_sampling:
            frame_timestamps, coords, frame_paths = self._sample_aligned_data(
                frame_timestamps, coords, frame_paths
            )

        aligned_coords = self._aligned_coords_from_raw(coords)

        n = len(aligned_coords)
        if n == 0:
            raise ValueError("No aligned coordinates available.")
        keep = [0]
        for i in range(1, n):
            dx = abs(aligned_coords[i, 0] - aligned_coords[i - 1, 0])
            dy = abs(aligned_coords[i, 1] - aligned_coords[i - 1, 1])
            if dx >= dy and (dx > 0 or dy > 0):
                keep.append(i)
        if len(keep) < 2:
            raise RuntimeError("Not enough horizontal-motion frames to build overlap image.")

        h_timestamps = np.asarray(frame_timestamps)[keep]
        h_coords = aligned_coords[keep]
        h_paths = [frame_paths[i] for i in keep]
        _, h_coords, h_paths = self._downsample_aligned_data(
            h_timestamps, h_coords, h_paths
        )
        if len(h_paths) < 2:
            raise RuntimeError("Not enough horizontal-motion frames to build overlap image after downsampling.")

        print(
            "[ScanStitcher] Horizontal overlap frames selected: "
            f"{len(h_paths)} / {len(keep)} horizontal-motion frames ({n} total aligned)."
        )
        h_frames = [
            self._to_rgb(path)
            for path in tqdm(h_paths, desc="Loading horizontal overlap frames", unit="frame")
        ]
        canvas = self._create_pixel_drift_canvas(
            coordinates=h_coords,
            camera_frames=h_frames,
        )

        if output_path is None:
            output_path = os.path.join(self._dataset_output_dir(), "horizontal_overlap_stitched.tif")
        elif not os.path.isabs(output_path):
            output_path = os.path.join(self._dataset_output_dir(), output_path)
        output_path = os.path.normpath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
        if not cv2.imwrite(output_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Failed to write horizontal overlap image: {output_path}")
        print(f"[ScanStitcher] Horizontal overlap image saved: {output_path}")
        return output_path

    def save_vertical_overlap_from_dataset(self, output_path=None, use_sampling=False):
        print("[ScanStitcher] Building vertical-motion overlap image...")
        frame_timestamps, coords, frame_paths = self._align_all_frames_to_movement()
        if use_sampling:
            frame_timestamps, coords, frame_paths = self._sample_aligned_data(
                frame_timestamps, coords, frame_paths
            )
        frame_timestamps, coords, frame_paths = self._downsample_aligned_data(
            frame_timestamps, coords, frame_paths
        )

        aligned_coords = self._aligned_coords_from_raw(coords)

        n = len(aligned_coords)
        if n == 0:
            raise ValueError("No aligned coordinates available.")
        keep = [0]
        for i in range(1, n):
            dx = abs(aligned_coords[i, 0] - aligned_coords[i - 1, 0])
            dy = abs(aligned_coords[i, 1] - aligned_coords[i - 1, 1])
            if dy > dx and (dx > 0 or dy > 0):
                keep.append(i)
        if len(keep) < 2:
            raise RuntimeError("Not enough vertical-motion frames to build overlap image.")

        v_coords = aligned_coords[keep]
        v_paths = [frame_paths[i] for i in keep]
        print(f"[ScanStitcher] Vertical overlap frames selected: {len(v_paths)} / {n}")
        v_frames = [
            self._to_rgb(path)
            for path in tqdm(v_paths, desc="Loading vertical overlap frames", unit="frame")
        ]
        canvas = self._create_pixel_drift_canvas(
            coordinates=v_coords,
            camera_frames=v_frames,
        )

        if output_path is None:
            output_path = os.path.join(self._dataset_output_dir(), "vertical_overlap_stitched.tif")
        elif not os.path.isabs(output_path):
            output_path = os.path.join(self._dataset_output_dir(), output_path)
        output_path = os.path.normpath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
        if not cv2.imwrite(output_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Failed to write vertical overlap image: {output_path}")
        print(f"[ScanStitcher] Vertical overlap image saved: {output_path}")
        return output_path

    @staticmethod
    def _concat_frames(frames, concat_axis):
        if not frames:
            return None
        base = np.asarray(frames[0])
        if base.ndim == 2:
            base = np.repeat(base[:, :, None], 3, axis=2)
        elif base.ndim != 3 or base.shape[2] < 3:
            return None
        base = base[:, :, :3]
        h, w = base.shape[:2]

        normalized = []
        for frame in frames:
            f = np.asarray(frame)
            if f.ndim == 2:
                f = np.repeat(f[:, :, None], 3, axis=2)
            elif f.ndim != 3 or f.shape[2] < 3:
                continue
            f = f[:, :, :3]
            if f.shape[:2] != (h, w):
                f = cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR)
            normalized.append(np.clip(f, 0, 255).astype(np.uint8))

        if not normalized:
            return None
        return np.concatenate(normalized, axis=concat_axis)

    def save_axis_debug_images(self, horizontal_output_path=None, vertical_output_path=None):
        print("[ScanStitcher] Building axis debug images...")
        frame_timestamps, coords, frame_paths = self._align_all_frames_to_movement()
        frame_timestamps, coords, frame_paths = self._sample_aligned_data(
            frame_timestamps, coords, frame_paths
        )
        frame_timestamps, coords, frame_paths = self._downsample_aligned_data(
            frame_timestamps, coords, frame_paths
        )
        aligned_coords = self._aligned_coords_from_raw(coords)

        n = len(aligned_coords)
        if n == 0:
            raise ValueError("No aligned coordinates available for axis debug images.")

        horizontal_indices = [0]
        vertical_indices = [0]
        for i in range(1, n):
            dx = abs(aligned_coords[i, 0] - aligned_coords[i - 1, 0])
            dy = abs(aligned_coords[i, 1] - aligned_coords[i - 1, 1])
            if dx == 0 and dy == 0:
                continue
            if dx >= dy:
                horizontal_indices.append(i)
            if dy > dx:
                vertical_indices.append(i)

        print(
            "[ScanStitcher] Axis split counts: "
            f"horizontal={len(horizontal_indices)}, vertical={len(vertical_indices)}"
        )

        h_frames = [
            self._to_rgb(frame_paths[i])
            for i in tqdm(horizontal_indices, desc="Loading horizontal frames", unit="frame")
        ]
        v_frames = [
            self._to_rgb(frame_paths[i])
            for i in tqdm(vertical_indices, desc="Loading vertical frames", unit="frame")
        ]

        horizontal_img = self._concat_frames(h_frames, concat_axis=0)
        vertical_img = self._concat_frames(v_frames, concat_axis=1)
        if horizontal_img is None or vertical_img is None:
            raise RuntimeError("Failed to build one or both axis debug images.")

        if horizontal_output_path is None:
            horizontal_output_path = os.path.join(self._dataset_output_dir(), "horizontal_only_debug.tif")
        elif not os.path.isabs(horizontal_output_path):
            horizontal_output_path = os.path.join(self._dataset_output_dir(), horizontal_output_path)

        if vertical_output_path is None:
            vertical_output_path = os.path.join(self._dataset_output_dir(), "vertical_only_debug.tif")
        elif not os.path.isabs(vertical_output_path):
            vertical_output_path = os.path.join(self._dataset_output_dir(), vertical_output_path)

        horizontal_output_path = os.path.normpath(horizontal_output_path)
        vertical_output_path = os.path.normpath(vertical_output_path)
        os.makedirs(os.path.dirname(horizontal_output_path), exist_ok=True) if os.path.dirname(horizontal_output_path) else None
        os.makedirs(os.path.dirname(vertical_output_path), exist_ok=True) if os.path.dirname(vertical_output_path) else None

        if not cv2.imwrite(horizontal_output_path, cv2.cvtColor(horizontal_img, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Failed to write horizontal debug image: {horizontal_output_path}")
        if not cv2.imwrite(vertical_output_path, cv2.cvtColor(vertical_img, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Failed to write vertical debug image: {vertical_output_path}")

        print(f"[ScanStitcher] Horizontal debug image saved: {horizontal_output_path}")
        print(f"[ScanStitcher] Vertical debug image saved: {vertical_output_path}")
        return horizontal_output_path, vertical_output_path

    def plot(self):
        if self.canvas is None:
            raise RuntimeError("Call stitch() before plot().")
        plt.figure(figsize=(10, 10))
        plt.imshow(self.canvas)
        plt.title("Image placement by coordinates")
        plt.axis("off")
        plt.show()

    def save(self, output_path=None, metadata_path=None):
        if self.canvas is None:
            raise RuntimeError("Call stitch() before save().")

        if output_path is None:
            output_path = os.path.join(self._dataset_output_dir(), "stitched_scan.tif")
        elif not os.path.isabs(output_path):
            output_path = os.path.join(self._dataset_output_dir(), output_path)
        print(f"[ScanStitcher] Saving stitched image: {output_path}")
        output_path = os.path.normpath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None

        ext = os.path.splitext(output_path)[1].lower()
        if ext == ".npy":
            np.save(output_path, self.canvas)
        elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]:
            bgr = cv2.cvtColor(self.canvas, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(output_path, bgr):
                raise RuntimeError(f"Failed to write image to {output_path}")
        else:
            raise ValueError(f"Unsupported image extension: {ext}")

        if metadata_path is None and output_path:
            metadata_path = f"{os.path.splitext(output_path)[0]}_meta.json"
        if metadata_path is not None:
            if not os.path.isabs(metadata_path):
                metadata_path = os.path.join(self._dataset_output_dir(), metadata_path)
            metadata_path = os.path.normpath(metadata_path)
            os.makedirs(os.path.dirname(metadata_path), exist_ok=True) if os.path.dirname(metadata_path) else None
            metadata = {
                "coord_file": self.coord_file,
                "photo_dir": self.photo_dir,
                "use_calibration": self.use_calibration,
                "flip_x": self.flip_x,
                "flip_y": self.flip_y,
                "use_pixel_drift_stitch": self.use_pixel_drift_stitch,
                "stitch_fraction": self.stitch_fraction,
                "downsample": self.downsample,
                "canvas_shape": self.canvas.shape,
                "num_frames": len(self.time_stamps),
            }
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

        return output_path


def main():
    use_pixel_drift_stitch = True
    use_streaming_stitch = True
    downsample = 1
    date_time_folder = "2026_03_13-15_28"
    stitcher = ScanStitcher(
        date_time_folder=date_time_folder,
        use_calibration=True,
        flip_x=True,
        downsample=downsample,
        use_pixel_drift_stitch=use_pixel_drift_stitch,
    )
    h_overlap = stitcher.save_horizontal_overlap_from_dataset("horizontal_overlap_stitched.tif")
    print(f"Saved horizontal overlap image: {h_overlap}")
    v_overlap = stitcher.save_vertical_overlap_from_dataset("vertical_overlap_stitched.tif")
    print(f"Saved vertical overlap image: {v_overlap}")
    h_dbg, v_dbg = stitcher.save_axis_debug_images(
        "horizontal_only_debug.tif",
        "vertical_only_debug.tif",
    )
    print(f"Saved horizontal debug image: {h_dbg}")
    print(f"Saved vertical debug image: {v_dbg}")
    # saved_projection = stitcher.save_projection_from_dataset("projection_image.tif")
    # print(f"Saved projection image: {saved_projection}")
    if use_streaming_stitch and use_pixel_drift_stitch:
        saved_scan = stitcher.stitch_streaming(
            output_path="stitched_scan.tif",
            metadata_path="meta.json",
            cleanup_memmap=True,
        )
    else:
        stitcher.stitch()
        saved_scan = stitcher.save("stitched_scan.tif", "meta.json")
    print(f"Saved stitched scan: {saved_scan}")
    # stitcher.plot()



if __name__ == "__main__":
    main()
