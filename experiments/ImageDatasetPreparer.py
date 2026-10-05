"""Build pipette coordinate records from saved rig-recorder frames."""

from __future__ import annotations


import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm isn't installed
    tqdm = None

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from patcherbot.deepLearning.pipetteDetector import PipetteDetectorYOLO1, PipetteDetector2
from patcherbot.deepLearning.pipetteFocuser import PipetteFocuser


from experiments.SimpleDatasetBuilder import (
    ActionSelector,
    AxisToggle,
    ObservationSelector,
    SimpleDatasetBuilder,
)


@dataclass
class FrameRecord:
    timestamp: float
    pi_x: float
    pi_y: float
    pi_z: float


class _EncoderDominantKalman1D:
    """Fuse noisy absolute focus with precise encoder motion.

    State is an encoder-to-focus offset:
      focus_estimate = encoder_z + offset

    The encoder drives step-to-step motion. The focuser updates only the offset.
    """

    def __init__(
        self,
        *,
        process_variance: float,
        measurement_variance: float,
        initial_variance: float,
    ) -> None:
        self.process_variance = max(float(process_variance), 0.0)
        self.measurement_variance = max(float(measurement_variance), 1e-9)
        self.initial_variance = max(float(initial_variance), 1e-9)
        self.offset_estimate: Optional[float] = None
        self.covariance = self.initial_variance

    def step(self, encoder_z: float, focus_measurement: float) -> float:
        encoder_valid = np.isfinite(encoder_z)
        focus_valid = np.isfinite(focus_measurement)
        if not encoder_valid:
            if focus_valid:
                return float(focus_measurement)
            return np.nan

        encoder_z = float(encoder_z)
        if self.offset_estimate is None:
            self.offset_estimate = (
                float(focus_measurement) - encoder_z if focus_valid else 0.0
            )
            self.covariance = self.initial_variance

        self.covariance += self.process_variance

        if focus_valid:
            predicted_focus = encoder_z + self.offset_estimate
            innovation = float(focus_measurement) - predicted_focus
            denom = self.covariance + self.measurement_variance
            gain = self.covariance / denom if denom > 0.0 else 0.0
            self.offset_estimate += gain * innovation
            self.covariance = (1.0 - gain) * self.covariance

        return float(encoder_z + self.offset_estimate)

class _DatasetFilterHelper(SimpleDatasetBuilder):
    """Lightweight SimpleDatasetBuilder adapter to reuse attempt filtering utilities."""

    def __init__(self, rig_data_root: Path) -> None:
        self._rig_data_root = Path(rig_data_root)
        observation_selector = ObservationSelector(
            include_pressure=False,
            include_resistance=False,
            include_current=False,
            include_voltage=False,
            include_stage=False,
            include_pipette=False,
            include_camera=False,
            stage_axes=AxisToggle(False, False, False),
            pipette_axes=AxisToggle(False, False, False),
        )
        action_selector = ActionSelector(
            include_stage=False,
            include_pipette=False,
            include_pressure=False,
            include_high_level=False,
            stage_axes=AxisToggle(False, False, False),
            pipette_axes=AxisToggle(False, False, False),
        )
        super().__init__(
            dataset_name="ImageDatasetPreparer_filter.hdf5",
            val_ratio=0.0,
            omit_stage_movement=False,
            random_seed=0,
            observation_selector=observation_selector,
            action_selector=action_selector,
        )

    def _write_metadata_files(self) -> None:  # pragma: no cover - metadata not needed
        """Skip metadata emission for filtering adapter."""
        return None

    def load_reference_timestamps(self, folder: str) -> Optional[np.ndarray]:
        demo_root = self._rig_data_root / folder
        candidates = (
            demo_root / "graph_recording.csv",
            demo_root / "cv_movement_recording.csv",
            demo_root / "movement_recording.csv",
        )
        for path in candidates:
            if not path.exists():
                continue
            try:
                table = pd.read_csv(path, sep=";")
            except Exception:
                continue
            if table.empty:
                continue
            first_column = table.columns[0]
            try:
                ts = table[first_column].to_numpy(dtype=float)
            except Exception:
                continue
            if ts.size:
                return ts
        return None

    def compute_attempt_windows(
        self,
        folder: str,
        reference_timestamps: np.ndarray,
    ) -> List[Tuple[float, float]]:
        if reference_timestamps.size == 0:
            return []
        experiment_first_timestamp = float(reference_timestamps[0] - 1)
        experiment_last_timestamp = float(reference_timestamps[-1] + 1)
        state_attempts = self.get_timestamps_for_all_successful_state_attempts(
            folder,
            experiment_first_timestamp,
            experiment_last_timestamp,
        )
        all_windows: List[Tuple[float, float]] = []
        for ranges in state_attempts.values():
            all_windows.extend(ranges)
        if not all_windows:
            return []
        return self._merge_windows(all_windows)

    @staticmethod
    def _merge_windows(windows: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
        ordered = sorted((float(start), float(end)) for start, end in windows if start < end)
        if not ordered:
            return []
        merged: List[Tuple[float, float]] = []
        cur_start, cur_end = ordered[0]
        for start, end in ordered[1:]:
            if start <= cur_end:
                cur_end = max(cur_end, end)
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = start, end
        merged.append((cur_start, cur_end))
        return merged




class ImageDatasetPreparer:
    _REQUIRED_STAGE_COLUMNS: Tuple[str, ...] = ("timestamp", "st_x", "st_y", "st_z")
    _OPTIONAL_PIPETTE_COLUMNS: Tuple[str, ...] = ("pi_x", "pi_y", "pi_z")
    _HEADERLESS_COLUMNS: Tuple[str, ...] = (
        "timestamp",
        "st_x",
        "st_y",
        "st_z",
        "pi_x",
        "pi_y",
        "pi_z",
    )
    _COLUMN_ALIASES = {
        "time_stamp": "timestamp",
        "time": "timestamp",
        "stage_x": "st_x",
        "stage_y": "st_y",
        "stage_z": "st_z",
        "pipette_x": "pi_x",
        "pipette_y": "pi_y",
        "pipette_z": "pi_z",
    }

    def __init__(
        self,
        rig_data_root: Path,
        *,
        use_detector1: bool = True,
        filter_images: bool = True,
        focus_with_detector_crop: bool = False,
        focus_crop_size: int = 256,
        use_kalman_focus_fusion: bool = False,
        kalman_process_variance: float = 0.25,
        kalman_measurement_variance: float = 100.0,
        kalman_initial_variance: float = 400.0,
    ) -> None:
        self.rig_data_root = Path(rig_data_root)
        if not self.rig_data_root.exists():
            raise FileNotFoundError(f"Rig-recorder root not found: {self.rig_data_root}")

        self.filter_images = filter_images
        self._filter_helper: Optional[_DatasetFilterHelper]
        if self.filter_images:
            try:
                self._filter_helper = _DatasetFilterHelper(self.rig_data_root)
            except Exception as exc:
                logging.warning("Failed to initialise SimpleDatasetBuilder filter helper: %s", exc)
                self._filter_helper = None
                self.filter_images = False
            else:
                logging.info("Frame filtering enabled for ImageDatasetPreparer")
        else:
            self._filter_helper = None

        self.focus_with_detector_crop = bool(focus_with_detector_crop)
        try:
            self.focus_crop_size = int(focus_crop_size)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"focus_crop_size must be an integer, got {focus_crop_size!r}") from exc
        if self.focus_crop_size <= 0:
            raise ValueError(f"focus_crop_size must be > 0, got {self.focus_crop_size}")

        self.use_kalman_focus_fusion = bool(use_kalman_focus_fusion)
        self.kalman_process_variance = max(float(kalman_process_variance), 0.0)
        self.kalman_measurement_variance = max(float(kalman_measurement_variance), 1e-9)
        self.kalman_initial_variance = max(float(kalman_initial_variance), 1e-9)

        self.detector = PipetteDetectorYOLO1() if use_detector1 else PipetteDetector2()
        self.focuser = PipetteFocuser()

    def build_csv(
        self,
        demo_folder: str,
        *,
        output_name: str = "cv_movement_recording.csv",
        camera_subdir: str = "camera_frames",
    ) -> Path:
        demo_path = self._resolve_demo_path(demo_folder)
        movement_path = demo_path / "movement_recording.csv"
        if not movement_path.exists():
            raise FileNotFoundError(f"movement_recording.csv not found: {movement_path}")

        camera_dir = demo_path / camera_subdir
        if not camera_dir.exists():
            raise FileNotFoundError(f"Camera frame folder not found: {camera_dir}")


        frame_paths = self._collect_frame_paths(camera_dir)
        if not frame_paths:
            raise RuntimeError(f"No image frames found in {camera_dir}")

        if self.filter_images:
            frame_paths = self._apply_frame_filter(frame_paths, demo_path)
            if not frame_paths:
                raise RuntimeError("No image frames remained after applying demonstration filter")

        logging.info("Loaded %d frames from %s", len(frame_paths), camera_dir)

        frame_records = self._infer_pipette_coordinates(frame_paths)
        if not frame_records:
            raise RuntimeError("No valid frames remained after inference preprocessing")
        logging.info("Inference complete for %d frames", len(frame_records))

        movement_df = self._load_movement_dataframe(movement_path)

        frame_df = pd.DataFrame([record.__dict__ for record in frame_records])
        frame_df = frame_df.sort_values("timestamp").reset_index(drop=True)

        movement_merge_columns = ["timestamp", "st_x", "st_y", "st_z"]
        if "pi_z" in movement_df.columns:
            movement_merge_columns.append("pi_z")
        merged = pd.merge_asof(
            frame_df,
            movement_df[movement_merge_columns].sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
            suffixes=("", "_enc"),
        )

        missing_stage = merged[["st_x", "st_y", "st_z"]].isna().any(axis=1).sum()
        if missing_stage:
            logging.warning("Stage data missing for %d frames", missing_stage)

        if self.use_kalman_focus_fusion:
            if "pi_z_enc" not in merged.columns:
                logging.warning(
                    "Kalman focus fusion is enabled, but encoder pi_z is unavailable in "
                    "movement_recording.csv. Falling back to raw focuser pi_z."
                )
            else:
                merged["pi_z"] = self._fuse_focus_with_encoder(
                    focus_values=merged["pi_z"],
                    encoder_values=merged["pi_z_enc"],
                )

        merged = merged[["timestamp", "st_x", "st_y", "st_z", "pi_x", "pi_y", "pi_z"]]
        output_path = demo_path / output_name
        merged.to_csv(output_path, sep=";", index=False, float_format="%.6f")
        logging.info("Wrote %s", output_path)
        return output_path

    def _load_movement_dataframe(self, movement_path: Path) -> pd.DataFrame:
        """Load movement CSV and require canonical stage headers."""

        try:
            movement_df = pd.read_csv(movement_path, sep=";")
        except Exception as exc:
            raise ValueError(f"Failed to read movement CSV: {movement_path}\n{exc}") from exc

        movement_df = self._normalize_movement_columns(movement_df)
        missing_columns = set(self._REQUIRED_STAGE_COLUMNS) - set(movement_df.columns)
        if missing_columns:
            try:
                headerless_df = pd.read_csv(movement_path, sep=";", header=None)
            except Exception:
                headerless_df = pd.DataFrame()

            if headerless_df.shape[1] >= len(self._HEADERLESS_COLUMNS):
                movement_df = headerless_df.iloc[:, : len(self._HEADERLESS_COLUMNS)].copy()
                movement_df.columns = list(self._HEADERLESS_COLUMNS)
            else:
                first_line = movement_path.read_text(encoding="utf-8", errors="replace").splitlines()
                header_preview = first_line[0] if first_line else "<empty file>"
                raise ValueError(
                    "movement_recording.csv missing required columns "
                    f"{sorted(missing_columns)}. Expected header columns include "
                    f"{list(self._REQUIRED_STAGE_COLUMNS)}. Found columns: {list(movement_df.columns)}. "
                    f"Header preview: {header_preview}"
                )

        numeric_columns = list(self._REQUIRED_STAGE_COLUMNS)
        numeric_columns.extend(
            [col for col in self._OPTIONAL_PIPETTE_COLUMNS if col in movement_df.columns]
        )
        for col in numeric_columns:
            movement_df[col] = pd.to_numeric(movement_df[col], errors="coerce")
        movement_df = movement_df.dropna(subset=list(self._REQUIRED_STAGE_COLUMNS))
        if movement_df.empty:
            raise ValueError(
                "movement_recording.csv has no valid numeric rows for required columns "
                f"{list(self._REQUIRED_STAGE_COLUMNS)}: {movement_path}"
            )
        return movement_df.sort_values("timestamp").reset_index(drop=True)

    def _normalize_movement_columns(self, movement_df: pd.DataFrame) -> pd.DataFrame:
        normalized = [str(col).strip().lower() for col in movement_df.columns]
        movement_df = movement_df.copy()
        movement_df.columns = normalized
        for source, target in self._COLUMN_ALIASES.items():
            if source in movement_df.columns and target not in movement_df.columns:
                movement_df = movement_df.rename(columns={source: target})
        return movement_df

    def _fuse_focus_with_encoder(
        self,
        *,
        focus_values: pd.Series,
        encoder_values: pd.Series,
    ) -> np.ndarray:
        focuser = pd.to_numeric(focus_values, errors="coerce").to_numpy(dtype=np.float64, copy=False)
        encoder = pd.to_numeric(encoder_values, errors="coerce").to_numpy(dtype=np.float64, copy=False)
        fused = np.full(focuser.shape[0], np.nan, dtype=np.float64)

        kalman = _EncoderDominantKalman1D(
            process_variance=self.kalman_process_variance,
            measurement_variance=self.kalman_measurement_variance,
            initial_variance=self.kalman_initial_variance,
        )
        focus_update_count = 0
        encoder_only_count = 0
        focus_only_count = 0
        missing_count = 0
        for idx, (focus_z, encoder_z) in enumerate(zip(focuser, encoder)):
            focus_valid = np.isfinite(focus_z)
            encoder_valid = np.isfinite(encoder_z)
            if encoder_valid and focus_valid:
                focus_update_count += 1
            elif encoder_valid:
                encoder_only_count += 1
            elif focus_valid:
                focus_only_count += 1
            else:
                missing_count += 1
            fused[idx] = kalman.step(encoder_z=encoder_z, focus_measurement=focus_z)

        logging.info(
            "Applied Kalman focus fusion (encoder-dominant): %d fused with focus updates, "
            "%d encoder-only, %d focus-only, %d missing",
            focus_update_count,
            encoder_only_count,
            focus_only_count,
            missing_count,
        )
        return fused

    def _resolve_demo_path(self, demo_folder: str) -> Path:
        candidate = Path(demo_folder)
        if candidate.is_dir():
            return candidate
        resolved = self.rig_data_root / demo_folder
        if resolved.is_dir():
            return resolved
        raise FileNotFoundError(f"Rig-recorder demo folder not found: {demo_folder}")

    @staticmethod
    def _collect_frame_paths(camera_dir: Path) -> List[Path]:
        frame_paths = [
            path
            for path in sorted(camera_dir.iterdir())
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        ]
        return frame_paths

    @staticmethod
    def _crop_around_point(img: np.ndarray, point: Sequence[float], crop_size: int) -> Optional[np.ndarray]:
        if img is None or point is None:
            return None
        if len(point) < 2:
            return None
        half = int(crop_size) // 2
        if half <= 0:
            return None

        h, w = img.shape[:2]
        try:
            cx = int(round(float(point[0])))
            cy = int(round(float(point[1])))
        except Exception:
            return None

        x_min = max(cx - half, 0)
        x_max = min(cx + half, w)
        y_min = max(cy - half, 0)
        y_max = min(cy + half, h)
        if x_max <= x_min or y_max <= y_min:
            return None
        return img[y_min:y_max, x_min:x_max]

    def _resolve_focus_input_image(self, img: np.ndarray, detected_xy: Optional[Tuple[int, int]]) -> Optional[np.ndarray]:
        if not self.focus_with_detector_crop:
            return img
        if detected_xy is None:
            return None
        return self._crop_around_point(img, detected_xy, self.focus_crop_size)

    def _apply_frame_filter(self, frame_paths: Sequence[Path], demo_path: Path) -> List[Path]:
        if not self.filter_images or self._filter_helper is None:
            return list(frame_paths)

        helper = self._filter_helper
        timestamps = helper.load_reference_timestamps(demo_path.name)
        if timestamps is None:
            logging.warning("Reference timestamps missing or unreadable for %s; skipping frame filter", demo_path)
            return list(frame_paths)

        windows = helper.compute_attempt_windows(demo_path.name, timestamps)
        if not windows:
            logging.info("No successful attempt windows detected for %s; using all frames", demo_path)
            return list(frame_paths)

        filtered: List[Path] = []
        removed = 0
        for img_path in frame_paths:
            timestamp = self._parse_timestamp(img_path.name)
            if timestamp is None:
                removed += 1
                continue
            if any(start <= timestamp <= end for start, end in windows):
                filtered.append(img_path)
            else:
                removed += 1

        if not filtered:
            logging.warning("Frame filter removed all frames for %s; fallback to full set", demo_path)
            return list(frame_paths)

        if removed:
            logging.info("Filtered out %d frames outside demonstration windows for %s", removed, demo_path)
        return filtered


    def _infer_pipette_coordinates(self, frame_paths: Sequence[Path]) -> List[FrameRecord]:
        records: List[FrameRecord] = []
        skipped_without_timestamp = 0
        failed_to_load = 0
        failed_detection = 0
        failed_focus_crop = 0
        failed_focus_inference = 0

        total_frames = len(frame_paths) if hasattr(frame_paths, "__len__") else None
        iterable = (
            tqdm(frame_paths, desc="Inferring pipette coordinates", unit="frame", total=total_frames)
            if tqdm is not None
            else frame_paths
        )

        try:
            for img_path in iterable:
                timestamp = self._parse_timestamp(img_path.name)
                if timestamp is None:
                    skipped_without_timestamp += 1
                    logging.debug("Skipping frame without timestamp: %s", img_path.name)
                    continue

                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img is None:
                    failed_to_load += 1
                    logging.debug("Failed to load image: %s", img_path)
                    continue

                xy = self.detector.detect_pipette(img)
                if xy is None:
                    failed_detection += 1
                    pi_x = np.nan
                    pi_y = np.nan
                else:
                    pi_x, pi_y = float(xy[0]), float(xy[1])

                focus_img = self._resolve_focus_input_image(img, xy)
                if focus_img is None:
                    failed_focus_crop += 1
                    pi_z = np.nan
                else:
                    try:
                        pi_z = float(self.focuser.get_pipette_focus_value(focus_img))
                    except Exception as exc:
                        failed_focus_inference += 1
                        pi_z = np.nan
                        logging.debug("Focuser inference failed for %s: %s", img_path, exc)
                records.append(FrameRecord(timestamp=timestamp, pi_x=pi_x, pi_y=pi_y, pi_z=pi_z))
        finally:
            if tqdm is not None and hasattr(iterable, "close"):
                iterable.close()

        if skipped_without_timestamp:
            logging.warning("Skipped %d frames without valid timestamps", skipped_without_timestamp)
        if failed_to_load:
            logging.warning("Failed to load %d frames", failed_to_load)
        if failed_detection:
            logging.info("Detector returned no result for %d frames", failed_detection)
        if failed_focus_crop:
            logging.info("Skipped focus inference for %d frames due to missing detector crop", failed_focus_crop)
        if failed_focus_inference:
            logging.warning("Focuser inference failed for %d frames", failed_focus_inference)
        return records

    @staticmethod
    def _parse_timestamp(filename: str) -> Optional[float]:
        stem = Path(filename).stem
        parts = stem.split("_")
        if len(parts) < 2:
            return None
        timestamp_str = "_".join(parts[1:])
        try:
            return float(timestamp_str)
        except ValueError:
            return None


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=level)


def run_preparer(
    rig_data_root: Path,
    rig_recorder_data_folder_set: Sequence[str | Path],
    *,
    output_name: str = "cv_movement_recording.csv",
    use_detector1: bool = False,
    filter_images: bool = False,
    focus_with_detector_crop: bool = False,
    use_kalman_focus_fusion: bool = False,
    verbose: bool = False,
) -> None:
    _configure_logging(verbose)
    preparer = ImageDatasetPreparer(
        rig_data_root,
        use_detector1=use_detector1,
        filter_images=filter_images,
        focus_with_detector_crop=focus_with_detector_crop,
        use_kalman_focus_fusion=use_kalman_focus_fusion,
    )
    total_folders = len(rig_recorder_data_folder_set) if hasattr(rig_recorder_data_folder_set, "__len__") else None
    iterator = (
        tqdm(rig_recorder_data_folder_set, desc="Rig-recorder folders", unit="folder", total=total_folders)
        if tqdm is not None
        else rig_recorder_data_folder_set
    )

    try:
        for folder in iterator:
            logging.info("Processing folder: %s", folder)
            preparer.build_csv(folder, output_name=output_name)
    finally:
        if tqdm is not None and hasattr(iterator, "close"):
            iterator.close()


if __name__ == "__main__":
    default_root = Path(__file__).resolve().parent / "Data" / "rig_recorder_data"
    

    rig_data_root = default_root
    # rig_recorder_data_folder_set = [
    #     "2025_09_25-20_43",
    #     "2025_09_25-21_39",
    #     "2025_10_01-13_15",
    #     "2025_10_01-13_30",
    # ]# training/valid for PipetteFinder 
    # rig_recorder_data_folder_set = [
    # "2025_09_25-22_13"]
     # test dataset for PipetteFinder

    # rig_recorder_data_folder_set = [
    #     "2025_05_20-15_50",
    #     "2025_05_20-15_16",
    #     "2025_05_20-14_05",
    #     "2025_04_10-11_57",
    #     "2025_04_10-12_16",
    # ] # HEK training data (5/20/2025, 4/10/2025)


    # rig_recorder_data_folder_set = ["2025_04_07-14_50"] # HEK testing data

    rig_recorder_data_folder_set = ["2025_10_10-15_12"]

    output_name = "cv_movement_recording.csv"
    use_detector1 = False
    verbose = True
    filter_images = True
    focus_with_detector_crop = False
    use_kalman_focus_fusion = False

    run_preparer(
        rig_data_root,
        rig_recorder_data_folder_set,
        output_name=output_name,
        use_detector1=use_detector1,
        filter_images=filter_images,
        focus_with_detector_crop=focus_with_detector_crop,
        use_kalman_focus_fusion=use_kalman_focus_fusion,
        verbose=verbose,
    )
