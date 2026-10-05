"""SimpleDatasetBuilder
====================

How to use
----------
1.  Import :class:`SimpleDatasetBuilder` (and optionally :class:`DatasetBuilderSettings`)
    from ``experiments.SimpleDatasetBuilder``.
2.  Instantiate a settings object or pass keyword arguments mirroring the
    original ``DatasetBuilder`` signature.  Only the parameters you wish to
    change need to be supplied; everything else falls back to the same
    defaults as ``DatasetBuilder``.
3.  Create ``SimpleDatasetBuilder`` with the settings and call :meth:`add_demo` for
    each rig-recorder folder.  All downstream helper methods retain their
    names and behaviour, so existing scripts can swap the import with minimal
    edits.

Example
:::::::

.. code-block:: python

    from experiments.SimpleDatasetBuilder import SimpleDatasetBuilder

    builder = SimpleDatasetBuilder(
        dataset_name="HEK_inference_set5.hdf5",
        val_ratio=0.0,
        load_next_obs=True,
    )

    for folder in ["2025_04_07-15_50", "2025_05_20-14_05"]:
        builder.add_demo(folder, record_to_file=True)

    builder.write_split_masks()

This produces the same dataset artefacts as the original builder while using a
more modular internal structure that groups related functionality.
"""

from __future__ import annotations

import datetime
import hashlib
import io
import json
import os
import re
import shutil
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import albumentations as A
import h5py
import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "experiments" / "Data"
RIG_RECORDER_DATA_ROOT = DATA_ROOT / "rig_recorder_data"
MAX_CAUSAL_CAMERA_FRAME_AGE_SECONDS = 0.100

# ---------------------------------------------------------------------------
# Configuration containers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FilterSettings:
    enable_random_filter: bool = False # set to true to enable albumentations filtering
    image_filter_prob: float = 0.65
    filter_train_only: bool = False
    filter_same_per_demo: bool = False


@dataclass(slots=True)
class GigasealAugmentationSettings:
    enabled: bool = False
    copies_per_demo: int = 1
    augment_validation: bool = False
    pressure_noise_std: float = 0.05
    pressure_offset_std: float = 0.1
    pressure_min_mbar: float = -1000.0
    pressure_max_mbar: float = 1000.0
    resistance_log_noise_std: float = 0.01
    resistance_floor: float = 1e-9
    sensor_stutter_probability: float = 0.02
    sensor_stutter_max_frames: int = 3
    prefix_hold_probability: float = 0.15
    prefix_hold_max_frames: int = 3
    observations_until_next_action_cap: int = 0


@dataclass(slots=True)
class AxisToggle:
    x: bool = True
    y: bool = True
    z: bool = True

    AXIS_NAMES: ClassVar[Tuple[str, str, str]] = ("x", "y", "z")

    def enabled_indices(self) -> List[int]:
        return [idx for idx, enabled in enumerate((self.x, self.y, self.z)) if enabled]

    def enabled_labels(self) -> List[str]:
        return [label for label, enabled in zip(self.AXIS_NAMES, (self.x, self.y, self.z)) if enabled]


@dataclass(slots=True)
class ObservationSelector:
    include_pressure: bool = True
    include_resistance: bool = True
    include_resistance_slope: bool = False
    include_current: bool = False
    include_voltage: bool = False
    include_stage: bool = True
    include_pipette: bool = True
    include_camera: bool = True
    include_gigaseal_pressure_state: bool = True
    include_gigaseal_effective_pressure: bool = True
    include_gigaseal_resistance_input: bool = False
    include_gigaseal_observations_since_last_action: bool = False
    include_gigaseal_time_since_last_action: bool = True
    stage_axes: AxisToggle = field(default_factory=AxisToggle)
    pipette_axes: AxisToggle = field(default_factory=AxisToggle)

    def stage_indices(self, available: int) -> List[int]:
        if not self.include_stage:
            return []
        return [idx for idx in self.stage_axes.enabled_indices() if idx < available]

    def pipette_indices(self, available: int) -> List[int]:
        if not self.include_pipette:
            return []
        return [idx for idx in self.pipette_axes.enabled_indices() if idx < available]

    def stage_axis_labels(self) -> List[str]:
        return self.stage_axes.enabled_labels() if self.include_stage else []

    def pipette_axis_labels(self) -> List[str]:
        return self.pipette_axes.enabled_labels() if self.include_pipette else []


@dataclass(slots=True)
class ActionSelector:
    include_stage: bool = False
    include_pipette: bool = True
    include_pressure: bool = False
    include_break_in_zap: bool = True
    include_high_level: bool = False
    include_observations_until_next_action: bool = True
    stage_axes: AxisToggle = field(default_factory=AxisToggle)
    pipette_axes: AxisToggle = field(default_factory=AxisToggle)
    combine_gigaseal_pressure_action: bool = False
    binary_gigaseal_pressure_action: bool = False

    def stage_indices(self, available: int) -> List[int]:
        if not self.include_stage:
            return []
        return [idx for idx in self.stage_axes.enabled_indices() if idx < available]

    def pipette_indices(self, available: int) -> List[int]:
        if not self.include_pipette:
            return []
        return [idx for idx in self.pipette_axes.enabled_indices() if idx < available]

    def stage_axis_labels(self) -> List[str]:
        if not self.include_stage:
            return []
        return [f"stage_{axis}" for axis in self.stage_axes.enabled_labels()]

    def pipette_axis_labels(self) -> List[str]:
        if not self.include_pipette:
            return []
        return [f"pipette_{axis}" for axis in self.pipette_axes.enabled_labels()]

    def pressure_axis_labels(self) -> List[str]:
        return ["commanded_pressure_mbar"] if self.include_pressure else []

    def gigaseal_pressure_axis_labels(self) -> List[str]:
        if self.binary_gigaseal_pressure_action:
            return ["gigaseal_delta_neg_5", "gigaseal_delta_pos_5", "gigaseal_reset"]
        if self.combine_gigaseal_pressure_action:
            return ["applied_pressure_mbar"]
        return ["commanded_pressure_mbar", "pressure_atm_state"]

    def timing_axis_labels(self) -> List[str]:
        return ["observations_until_next_action"] if self.include_observations_until_next_action else []


@dataclass(slots=True)
class DatasetBuilderSettings:
    dataset_name: str
    val_ratio: float = 1 / 6 # fraction of demos to reserve for validation
    omit_stage_movement: bool = False # set to true to only record demos when stage is stationary
    random_seed: int = 0
    debug_single_trajectory: bool = False # rewrite output to one random demo copied into train and valid
    include_failed_demos: bool = False # include failed state-recorder attempts; aborted attempts stay excluded
    freq_mask: int = 1
    load_next_obs: bool = False # set to true for goal conditioning
    use_velocities: bool = False # set to true to convert action deltas into per-observation velocities
    prefer_cv_movement: bool = True # set to true to prefer cv_movement_recording.csv over movement_recording.csv
    use_synthetic_position_anchor: bool = False # zero-reference stage/pipette observations to their first positions
    filter: FilterSettings = field(default_factory=FilterSettings)
    gigaseal_augmentation: GigasealAugmentationSettings = field(default_factory=GigasealAugmentationSettings)
    image_resize: int = 85
    pipette_final_pos_color_dot: bool = False # goal-conditioning option: mark final pipette position in camera frames
    output_mode: str = "hdf5" # either "hdf5" or "images"
    locate_cell_end_fraction: float = 0.1 # eligible fraction at the end of each locate-cell attempt
    image_sample_fraction: float = 0.1 # fraction sampled without replacement from each eligible pool

    observation_selector: ObservationSelector = field(default_factory=ObservationSelector)
    action_selector: ActionSelector = field(
        default_factory=lambda: ActionSelector(
            include_stage=False,
            stage_axes=AxisToggle(x=True, y=True, z=True),
            pipette_axes=AxisToggle(x=True, y=True, z=True),
        )
    )

    # Legacy toggles preserved for parity with DatasetBuilder
    center_crop: bool = False # set to true to center crop images around pipette
    inaction: int = 0 # maximum number of consecutive zero-action steps to keep
    inaction_tolerance: float = 0 # per-axis magnitude treated as inactivity
    skip_invalid_observations: bool = True # drop timesteps with NaN/Inf payloads in the selected data
    gigaseal_start_trim_enabled: bool = False # start gigaseal trajectories at the first near--5 mbar command
    gigaseal_event_window_enabled: bool = True # keep gigaseal traces from before first trainable action event
    gigaseal_event_window_radius: int = 15
    gigaseal_resistance_cutoff_enabled: bool = False # stop gigaseal trajectories once resistance reaches the cutoff
    gigaseal_resistance_cutoff: float = 1200.0
    break_in_event_window_enabled: bool = True # keep break-in traces around trainable action events
    break_in_event_window_radius: int = 15
    resistance_slope_window: int = 20
    resistance_input_window: int = 15


@dataclass(slots=True)
class _StateDatasetContext:
    state_name: str
    dataset_name: str
    dataset_dir: Path
    dataset_path: Path
    metadata_filename: str
    split_keys: Dict[str, List[str]]
    processed_folders: List[str] = field(default_factory=list)


@dataclass(slots=True)
class _StateAttemptRecord:
    started: float
    finished: float
    outcome_code: int = 0
    system_mode_code: Optional[int] = None
    attempt_json: Optional[str] = None

    @property
    def outcome_label(self) -> str:
        return _STATE_OUTCOME_LABELS.get(self.outcome_code, f"outcome_{self.outcome_code}")


@dataclass(slots=True)
class _GigasealAugmentedPayload:
    actions: np.ndarray
    dones: np.ndarray
    timestamp_values: Optional[np.ndarray]
    pressure_values: Optional[np.ndarray]
    pressure_state_values: Optional[np.ndarray]
    effective_pressure_values: Optional[np.ndarray]
    resistance_values: Optional[np.ndarray]
    resistance_slope_values: Optional[np.ndarray]
    resistance_input_values: Optional[np.ndarray]
    current_values: Optional[np.ndarray]
    voltage_values: Optional[np.ndarray]
    stage_positions: Optional[np.ndarray]
    pipette_positions: Optional[np.ndarray]
    camera_frames: Optional[np.ndarray]
    observations_since_last_action_values: Optional[np.ndarray]
    time_since_last_action_values: Optional[np.ndarray]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

_LOG_TIMEZONE = ZoneInfo("America/New_York")
_PRESSURE_EVENT_DEDUP_TOLERANCE_SECONDS = 0.1
_STATE_OUTCOME_SUCCESS = 0
_STATE_OUTCOME_FAILURE = 1
_STATE_OUTCOME_ABORTED = 2
_STATE_OUTCOME_LABELS = {
    _STATE_OUTCOME_SUCCESS: "success",
    _STATE_OUTCOME_FAILURE: "failure",
    _STATE_OUTCOME_ABORTED: "aborted",
}
_PRESSURE_RAW_RE = re.compile(
    r"^Setting pressure to (?P<value>[+-]?(?:\d+(?:\.\d+)?|\.\d+)) mbar \(raw: (?P<raw>[^)]+)\)$"
)
_PRESSURE_PLAIN_RE = re.compile(
    r"^Setting pressure to (?P<value>[+-]?(?:\d+(?:\.\d+)?|\.\d+)) mbar$"
)
_ZAP_RE = re.compile(r"^zapping(?:\.\.\.)?(?:\s*\(agent command\))?$", re.IGNORECASE)
_PRESSURE_LOG_EVENT_COLUMNS = [
    "timestamp",
    "event_kind",
    "commanded_pressure_mbar",
    "pressure_atm_state",
    "zap_command",
]

def _slugify_state_name(name: str) -> str:
    """Return a filesystem friendly slug for a state name."""

    cleaned = name.strip().lower().replace(" ", "_")
    slug = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in cleaned)
    slug = slug.strip('_')
    return slug or 'state'


def _is_gigaseal_state_name(name: str) -> bool:
    return "gigaseal" in _slugify_state_name(name)


def _is_break_in_state_name(name: str) -> bool:
    slug = _slugify_state_name(name)
    return "break_in" in slug or "breakin" in slug


def _uses_pressure_command_observations(name: str) -> bool:
    return _is_gigaseal_state_name(name) or _is_break_in_state_name(name)


def _stable_int_seed(*parts: object) -> int:
    """Create a deterministic 32-bit integer seed from arbitrary parts."""
    data = ("||".join(map(str, parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(data).digest()[:4], "big")


def _shift_forward(arr: np.ndarray) -> np.ndarray:
    """Return a copy of `arr` shifted left with the final element repeated."""

    out = np.empty_like(arr)
    out[:-1] = arr[1:]
    out[-1] = arr[-1]
    return out


def _read_csv_with_fallback(
    path: Path,
    *,
    encodings: Sequence[str] = ("utf-8", "utf-8-sig", "cp1252", "latin-1"),
    **kwargs,
) -> pd.DataFrame:
    """Load a CSV trying multiple encodings before replacing undecodable bytes."""

    last_error: Optional[Exception] = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except (UnicodeDecodeError, LookupError) as exc:
            last_error = exc
            continue
    with path.open("rb") as fh:
        buffer = fh.read().decode("utf-8", errors="replace")
    if last_error is not None:
        warnings.warn(
            f"Decoding issues detected while reading {path}; characters outside the fallback encoding were replaced.",
            RuntimeWarning,
        )
    return pd.read_csv(io.StringIO(buffer), **kwargs)


def _normalize_log_message(message: object) -> str:
    """Collapse embedded whitespace so quoted multiline CSV messages are matchable."""

    if pd.isna(message):
        return ""
    return re.sub(r"\s+", " ", str(message)).strip()


def _local_time_columns_to_epoch(
    time_series: pd.Series,
    ms_series: pd.Series,
) -> pd.Series:
    """Convert local log time columns into Unix epoch seconds."""

    ms_text = (
        ms_series.fillna("")
        .astype(str)
        .str.strip()
        .str.extract(r"(\d+)", expand=False)
        .fillna("0")
        .str.zfill(3)
    )
    combined = time_series.fillna("").astype(str).str.strip() + "." + ms_text
    parsed = pd.to_datetime(
        combined,
        format="%Y-%m-%d %H:%M:%S.%f",
        errors="coerce",
    )
    return parsed.apply(
        lambda value: value.to_pydatetime().replace(tzinfo=_LOG_TIMEZONE).timestamp()
        if not pd.isna(value)
        else np.nan
    )


def _positions_to_deltas(positions: np.ndarray) -> np.ndarray:
    """Return per-observation deltas with an initial zero row."""

    if positions.size == 0:
        return np.empty_like(positions, dtype=np.float64)

    positions = np.asarray(positions, dtype=np.float64)
    zeros = np.zeros((1, positions.shape[1]), dtype=np.float64)
    if positions.shape[0] <= 1:
        return zeros
    return np.vstack([zeros, np.diff(positions, axis=0)])


def _deltas_to_observation_velocities(
    deltas: np.ndarray,
    timestamps: np.ndarray,
) -> np.ndarray:
    """Convert deltas into per-observation velocities.

    The first sample uses forward Euler, the last uses backward Euler, and
    interior samples blend forward/backward Euler slopes with dt-based weights.
    """

    deltas = np.asarray(deltas, dtype=np.float64)
    if deltas.size == 0:
        return np.empty_like(deltas)

    num_samples = deltas.shape[0]
    velocities = np.zeros_like(deltas)
    if num_samples <= 1:
        return velocities

    timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if timestamps.shape[0] != num_samples:
        raise ValueError("timestamps and deltas must have matching lengths")

    dt = np.diff(timestamps)
    interval_velocities = np.zeros((num_samples - 1, deltas.shape[1]), dtype=np.float64)
    valid_dt = dt > np.finfo(np.float64).eps
    if np.any(valid_dt):
        interval_velocities[valid_dt] = deltas[1:][valid_dt] / dt[valid_dt, None]

    if not np.any(valid_dt):
        return velocities

    valid_indices = np.flatnonzero(valid_dt)
    velocities[0] = interval_velocities[valid_indices[0]]
    velocities[-1] = interval_velocities[valid_indices[-1]]

    if num_samples > 2:
        interior = velocities[1:-1]
        back_v = interval_velocities[:-1]
        fwd_v = interval_velocities[1:]
        back_dt = dt[:-1]
        fwd_dt = dt[1:]
        back_valid = valid_dt[:-1]
        fwd_valid = valid_dt[1:]

        both_valid = back_valid & fwd_valid
        back_only = back_valid & ~fwd_valid
        fwd_only = ~back_valid & fwd_valid

        if np.any(both_valid):
            denom = (back_dt[both_valid] + fwd_dt[both_valid])[:, None]
            interior[both_valid] = (
                back_dt[both_valid, None] * back_v[both_valid]
                + fwd_dt[both_valid, None] * fwd_v[both_valid]
            ) / denom
        if np.any(back_only):
            interior[back_only] = back_v[back_only]
        if np.any(fwd_only):
            interior[fwd_only] = fwd_v[fwd_only]
    return velocities


def _parse_waveform_column(column: Sequence[str]) -> np.ndarray:
    """Parse JSON-encoded voltage/current columns and pad to equal length."""

    lists = [json.loads(value) for value in column]
    max_len = max(len(lst) for lst in lists)
    padded = [lst + [lst[-1]] * (max_len - len(lst)) for lst in lists]
    return np.asarray(padded, dtype=np.float64)


def _resolve_dataset_paths(dataset_name: str) -> Tuple[Path, Path]:
    """Return dataset directory and HDF5 file path for ``dataset_name``."""

    datasets_root = Path("experiments/Datasets")
    name_path = Path(dataset_name)
    file_name = name_path.name
    folder_name = name_path.stem if name_path.suffix else name_path.name
    dataset_dir = datasets_root / folder_name
    dataset_path = dataset_dir / file_name
    return dataset_dir, dataset_path


def _ensure_dataset_stub(dataset_name: str, create_file: bool = False) -> Tuple[Path, Path]:
    """Ensure dataset directory exists and optionally prepare an empty HDF5 stub."""

    dataset_dir, dataset_path = _resolve_dataset_paths(dataset_name)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    legacy_path = Path("experiments/Datasets") / dataset_name
    if legacy_path.exists() and not dataset_path.exists():
        legacy_path.replace(dataset_path)

    if create_file and not dataset_path.exists():
        with h5py.File(dataset_path, "w") as hf:
            group = hf.create_group("data")
            group.attrs["num_demos"] = 0

    return dataset_dir, dataset_path

class RandomFilterMixin:
    """Albumentations augmentation wrapper kept functionally identical."""

    def __init__(self, settings: DatasetBuilderSettings):
        """Configure Albumentations filters from :class:`DatasetBuilderSettings`."""
        self._filter_settings = settings.filter
        self._rng_seed = settings.random_seed
        self._albu_filter = None
        if self._filter_settings.enable_random_filter:
            self._albu_filter = A.Compose(
                [
                    A.OneOf(
                        [
                            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                            A.HueSaturationValue(
                                hue_shift_limit=20,
                                sat_shift_limit=30,
                                val_shift_limit=20,
                                p=1.0,
                            ),
                            A.Sharpen(p=1.0),
                        ],
                        p=self._filter_settings.image_filter_prob,
                    )
                ],
                seed=settings.random_seed,
            )
        self._filter_active_for_demo = False
        self._albu_replay_comp: Optional[A.ReplayCompose] = None
        self._albu_replay_state = None

    # --- filter context -------------------------------------------------
    def begin_filter_context(self, split_label: str, demo_seed: int) -> None:
        """Prepare Albumentations state for a demo.

        Parameters
        ----------
        split_label:
            Either ``"train"`` or ``"valid"`` to indicate which dataset split
            the demo belongs to.  The flag controls whether filtering is
            enabled when ``filter_train_only`` is set.
        demo_seed:
            Integer seed that keeps per-demo replay filters deterministic when
            ``filter_same_per_demo`` is active.
        """
        cfg = self._filter_settings
        if not cfg.enable_random_filter or (
            cfg.filter_train_only and split_label != "train"
        ):
            self._filter_active_for_demo = False
            self._albu_replay_comp = None
            self._albu_replay_state = None
            return

        self._filter_active_for_demo = True
        if cfg.filter_same_per_demo:
            self._albu_replay_comp = A.ReplayCompose(
                [
                    A.OneOf(
                        [
                            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                            A.HueSaturationValue(
                                hue_shift_limit=20,
                                sat_shift_limit=30,
                                val_shift_limit=20,
                                p=1.0,
                            ),
                            A.Sharpen(p=1.0),
                        ],
                        p=cfg.image_filter_prob,
                    )
                ],
                seed=int(demo_seed),
            )
            self._albu_replay_state = None
        else:
            self._albu_replay_comp = None
            self._albu_replay_state = None

    def end_filter_context(self) -> None:
        """Reset augmentation state after finishing a demo."""
        self._filter_active_for_demo = False
        self._albu_replay_comp = None
        self._albu_replay_state = None

    def apply_albu_filter_to_pil(self, pil_image: Image.Image) -> Image.Image:
        """Apply the configured Albumentations pipeline to a PIL image.

        Parameters
        ----------
        pil_image:
            RGB or grayscale :class:`PIL.Image.Image` frame from the rig
            recorder dataset.

        Returns
        -------
        PIL.Image.Image
            The potentially augmented frame.  If filtering is disabled the
            input image object is returned unchanged.
        """
        if not self._filter_active_for_demo:
            return pil_image
        if self._albu_filter is None and self._albu_replay_comp is None:
            return pil_image

        np_img = np.array(pil_image.convert("RGB"))
        if self._albu_replay_comp is not None:
            if self._albu_replay_state is None:
                out = self._albu_replay_comp(image=np_img)
                self._albu_replay_state = out["replay"]
                np_img = out["image"]
            else:
                np_img = A.ReplayCompose.replay(
                    self._albu_replay_state, image=np_img
                )["image"]
        else:
            np_img = self._albu_filter(image=np_img)["image"]
        return Image.fromarray(np_img)


class SimpleDatasetBuilder(RandomFilterMixin):
    """DatasetBuilder2 variant with calibration and motion transforms removed."""

    def __init__(self, **kwargs):
        """Initialise the builder with keyword arguments from DatasetBuilder.

        Parameters
        ----------
        **kwargs:
            Any parameter accepted by the legacy :class:`DatasetBuilder`
            constructor.  See :class:`DatasetBuilderSettings` for the full list
            and default values.
        """
        settings = DatasetBuilderSettings(**kwargs)
        self.settings = settings
        super().__init__(settings)

        self.dataset_name = settings.dataset_name
        self.center_crop = settings.center_crop
        self.image_resize = settings.image_resize
        self.pipette_final_pos_color_dot = settings.pipette_final_pos_color_dot
        self.inaction = settings.inaction
        self.inaction_tolerance = settings.inaction_tolerance
        self.skip_invalid_observations = settings.skip_invalid_observations
        self.gigaseal_start_trim_enabled = settings.gigaseal_start_trim_enabled
        self.gigaseal_event_window_enabled = settings.gigaseal_event_window_enabled
        self.gigaseal_event_window_radius = max(0, int(settings.gigaseal_event_window_radius))
        self.gigaseal_resistance_cutoff_enabled = settings.gigaseal_resistance_cutoff_enabled
        self.gigaseal_resistance_cutoff = settings.gigaseal_resistance_cutoff
        self.break_in_event_window_enabled = settings.break_in_event_window_enabled
        self.break_in_event_window_radius = max(0, int(settings.break_in_event_window_radius))
        window_limit = self._event_window_radius_limit()
        self._clamp_history_windows_to_trim(settings, window_limit)
        self.gigaseal_augmentation = settings.gigaseal_augmentation
        self.val_ratio = settings.val_ratio
        self.omit_stage_movement = settings.omit_stage_movement
        self.include_failed_demos = settings.include_failed_demos
        self.rng = np.random.default_rng(settings.random_seed)
        self.load_next_obs = settings.load_next_obs
        self.use_velocities = settings.use_velocities
        self.prefer_cv_movement = settings.prefer_cv_movement
        self.use_synthetic_position_anchor = settings.use_synthetic_position_anchor
        self.freq_mask = max(1, int(settings.freq_mask))
        self.observation_selector = settings.observation_selector
        self.action_selector = settings.action_selector
        self._synchronize_selectors()
        self._last_action_labels: List[str] = []
        self._last_action_stage_cols: int = 0
        self._last_stage_motion_detected: bool = False
        self._camera_frame_shape_cache: Dict[str, Tuple[int, int]] = {}
        self._using_cv_movement_file = False
        self._last_pipette_scale: Tuple[float, float] = (1.0, 1.0)
        self._last_action_representation = "velocity" if self.use_velocities else "delta"
        self._pressure_event_cache: Dict[str, pd.DataFrame] = {}

        self.dataset_dir, self.dataset_path = _ensure_dataset_stub(settings.dataset_name, create_file=False)
        self._base_dataset_name = self.dataset_name
        self._metadata_filename = "metadata.json"
        self._state_contexts: Dict[str, _StateDatasetContext] = {}
        self._processed_folders: List[str] = self._load_existing_processed_folders(
            self.dataset_dir / self._metadata_filename
        )
        self._active_state_context: Optional[_StateDatasetContext] = None
        self._full_image_source_counts: Dict[str, int] = {}

        if self.val_ratio == 0:
            self._split_keys = {"train": []}
        else:
            self._split_keys = {"train": [], "valid": []}

        if settings.output_mode not in {"hdf5", "images"}:
            raise ValueError("output_mode must be either hdf5 or images")
        if settings.output_mode == "hdf5":
            self._write_metadata_files()

    @staticmethod
    def _resolve_rig_recorder_folder(rig_recorder_data_folder: str) -> Path:
        """Resolve an explicit demo path or a legacy name under the repository data root."""
        candidate = Path(rig_recorder_data_folder).expanduser()
        if candidate.is_absolute():
            return candidate
        return RIG_RECORDER_DATA_ROOT / candidate

    @staticmethod
    def _rig_recorder_folder_name(rig_recorder_data_folder: str) -> str:
        """Return the demo directory name for metadata, logs, and output filenames."""
        return Path(rig_recorder_data_folder).name

    @classmethod
    def _resolve_state_recorder_root(cls, rig_recorder_data_folder: str) -> Path:
        """Resolve state attempts beside an explicitly selected rig-data root."""
        candidate = Path(rig_recorder_data_folder).expanduser()
        if candidate.is_absolute():
            rig_root = cls._resolve_rig_recorder_folder(rig_recorder_data_folder).parent
            return rig_root.parent / "state_recorder_data"
        return DATA_ROOT / "state_recorder_data"

    @staticmethod
    def _normalize_legacy_epoch_timestamp(timestamp: float) -> float:
        """Convert legacy millisecond epoch timestamps to seconds."""
        if abs(timestamp) >= 100_000_000_000:
            return timestamp / 1000.0
        return timestamp

    @staticmethod
    def _camera_file_timestamp(path: Path) -> Optional[float]:
        """Return the timestamp suffix used by rig-recorder camera filenames."""
        try:
            return float(path.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return None

    def _export_sampled_full_images(
        self,
        rig_recorder_data_folder: str,
        state_attempts: Mapping[str, Sequence[_StateAttemptRecord]],
    ) -> None:
        """Copy deterministic samples from locate-cell and hunt-cell attempts."""
        if self.settings.output_mode != "images":
            return

        folder_name = self._rig_recorder_folder_name(rig_recorder_data_folder)
        camera_dir = self._resolve_rig_recorder_folder(rig_recorder_data_folder) / "camera_frames"
        if not camera_dir.is_dir():
            warnings.warn(f"Camera folder not found for full-image export: {camera_dir}")
            return

        timestamped_files = []
        for path in camera_dir.iterdir():
            if path.is_file():
                timestamp = self._camera_file_timestamp(path)
                if timestamp is not None:
                    timestamped_files.append((timestamp, path))
        timestamped_files.sort(key=lambda item: item[0])

        image_dir = self.dataset_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for state_name, attempts in state_attempts.items():
            state_slug = _slugify_state_name(state_name)
            is_locate = "locate_cell" in state_slug
            is_hunt = "hunt_cell" in state_slug
            if not (is_locate or is_hunt):
                continue
            for attempt_index, attempt in enumerate(attempts):
                candidates = [
                    (timestamp, path)
                    for timestamp, path in timestamped_files
                    if attempt.started <= timestamp <= attempt.finished
                ]
                if not candidates:
                    continue
                sample_rng = np.random.default_rng(
                    _stable_int_seed(
                        self.settings.random_seed,
                        folder_name,
                        state_slug,
                        attempt.started,
                        attempt.finished,
                        "full_images",
                    )
                )
                if is_locate:
                    fraction = min(1.0, max(0.0, self.settings.locate_cell_end_fraction))
                    eligible_count = min(
                        len(candidates),
                        max(1, round(len(candidates) * fraction)),
                    ) if fraction > 0.0 else 0
                    eligible = candidates[-eligible_count:] if eligible_count else []
                else:
                    eligible = candidates

                sample_fraction = min(1.0, max(0.0, self.settings.image_sample_fraction))
                sample_count = min(
                    len(eligible),
                    max(1, round(len(eligible) * sample_fraction)),
                ) if eligible and sample_fraction > 0.0 else 0
                if sample_count:
                    selected_indices = sample_rng.choice(
                        len(eligible),
                        size=sample_count,
                        replace=False,
                    )
                    selected = [eligible[int(index)] for index in selected_indices]
                else:
                    selected = []

                for timestamp, source_path in selected:
                    destination_name = (
                        f"{folder_name}_{state_slug}_{attempt_index:03d}_"
                        f"{timestamp:.6f}{source_path.suffix.lower()}"
                    )
                    shutil.copy2(source_path, image_dir / destination_name)
                    copied += 1

        if copied:
            self._full_image_source_counts[folder_name] = (
                self._full_image_source_counts.get(folder_name, 0) + copied
            )
        total = sum(self._full_image_source_counts.values())
        summary = pd.DataFrame(
            [
                {
                    "source": source,
                    "image_count": count,
                    "percentage": (100.0 * count / total) if total else 0.0,
                }
                for source, count in sorted(self._full_image_source_counts.items())
            ],
            columns=["source", "image_count", "percentage"],
        )
        summary.to_csv(image_dir / "image_sources.csv", index=False)
        print(f"  copied {copied} sampled full-resolution image(s) to {image_dir}")

    def _event_window_radius_limit(self) -> Optional[int]:
        radii: List[int] = []
        if self.gigaseal_event_window_enabled:
            radii.append(max(0, int(self.gigaseal_event_window_radius)))
        if self.break_in_event_window_enabled:
            radii.append(max(0, int(self.break_in_event_window_radius)))
        if not radii:
            return None
        return min(radii)

    @staticmethod
    def _clamp_history_windows_to_trim(
        settings: DatasetBuilderSettings,
        window_limit: Optional[int],
    ) -> None:
        if window_limit is None or window_limit <= 0:
            return
        limit = int(window_limit)
        settings.resistance_input_window = min(
            max(1, int(settings.resistance_input_window)),
            limit,
        )
        settings.resistance_slope_window = min(
            max(1, int(settings.resistance_slope_window)),
            limit,
        )

    def _synchronize_selectors(self) -> None:
        """Ensure observation axes are not broader than the chosen action axes."""

        obs = self.observation_selector
        act = self.action_selector

        def _clamp_axes(obs_toggle: AxisToggle, act_toggle: AxisToggle) -> None:
            for axis in AxisToggle.AXIS_NAMES:
                if not getattr(act_toggle, axis) and getattr(obs_toggle, axis):
                    setattr(obs_toggle, axis, False)

        if not act.include_stage:
            obs.include_stage = False
            obs.stage_axes = AxisToggle(False, False, False)
        elif obs.include_stage:
            _clamp_axes(obs.stage_axes, act.stage_axes)
            if not obs.stage_axes.enabled_indices():
                obs.include_stage = False

        if not act.include_pipette:
            obs.include_pipette = False
            obs.pipette_axes = AxisToggle(False, False, False)
        elif obs.include_pipette:
            _clamp_axes(obs.pipette_axes, act.pipette_axes)
            if not obs.pipette_axes.enabled_indices():
                obs.include_pipette = False

    def _ensure_state_context(self, state_name: str) -> _StateDatasetContext:
        """Create or return cached dataset bookkeeping for ``state_name``."""

        slug = _slugify_state_name(state_name)
        if slug in self._state_contexts:
            return self._state_contexts[slug]

        base_path = Path(self._base_dataset_name)
        suffix = base_path.suffix or ".hdf5"
        stem = base_path.stem if base_path.suffix else base_path.name
        dataset_name = f"{stem}_{slug}{suffix}"
        dataset_dir = self.dataset_dir
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = dataset_dir / dataset_name

        if self.val_ratio == 0:
            split_keys = {"train": []}
        else:
            split_keys = {"train": [], "valid": []}

        metadata_filename = f"metadata_{slug}.json"
        metadata_path = dataset_dir / metadata_filename
        context = _StateDatasetContext(
            state_name=slug,
            dataset_name=dataset_name,
            dataset_dir=dataset_dir,
            dataset_path=dataset_path,
            metadata_filename=metadata_filename,
            split_keys=split_keys,
            processed_folders=self._load_existing_processed_folders(metadata_path),
        )
        self._state_contexts[slug] = context
        return context

    @contextmanager
    def _use_state_context(self, state_name: str):
        """Temporarily switch builder bookkeeping to a state-specific dataset."""

        context = self._ensure_state_context(state_name)
        original_name = self.dataset_name
        original_dir = self.dataset_dir
        original_path = self.dataset_path
        original_split_keys = self._split_keys
        original_metadata = self._metadata_filename
        original_context = self._active_state_context

        self.dataset_name = context.dataset_name
        self.dataset_dir = context.dataset_dir
        self.dataset_path = context.dataset_path
        self._split_keys = context.split_keys
        self._metadata_filename = context.metadata_filename
        self._active_state_context = context
        try:
            yield
        finally:
            context.split_keys = self._split_keys
            self.dataset_name = original_name
            self.dataset_dir = original_dir
            self.dataset_path = original_path
            self._split_keys = original_split_keys
            self._metadata_filename = original_metadata
            self._active_state_context = original_context

    # ------------------------------------------------------------------
    # --- filtering --------------------------------------------------------
    @staticmethod
    def _finite_row_mask(array: np.ndarray) -> np.ndarray:
        """Return a boolean mask marking rows whose payload is entirely finite."""

        array = np.asarray(array)
        if array.ndim == 0:
            raise ValueError("Expected an array with a leading sample dimension")
        if array.shape[0] == 0:
            return np.zeros(0, dtype=bool)
        flattened = array.reshape(array.shape[0], -1)
        return np.isfinite(flattened).all(axis=1)

    def filter_inactive_actions(
        self,
        actions: np.ndarray,
        *arrays: Optional[np.ndarray],
        force_keep_mask: Optional[np.ndarray] = None,
    ) -> tuple:
        """Drop contiguous segments where all action components remain zero."""
        if self.inaction == 0:
            return (actions,) + arrays

        if self.inaction_tolerance > 0.0:
            inactive = (np.all(np.abs(actions) <= self.inaction_tolerance, axis=1)).astype(np.int8)
        else:
            inactive = (actions.sum(axis=1) == 0).astype(np.int8)
        diff = np.diff(np.concatenate(([0], inactive, [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        keep = np.ones_like(inactive, dtype=bool)
        for s, e in zip(starts, ends):
            if e - s >= self.inaction:
                start = max(s, 1)  # always keep the very first sample
                if start < e:
                    keep[start:e] = False
        if force_keep_mask is not None:
            force_keep = np.asarray(force_keep_mask, dtype=bool).reshape(-1)
            if force_keep.shape[0] != keep.shape[0]:
                raise ValueError("force_keep_mask must match the action row count")
            keep |= force_keep

        filtered = (actions[keep],) + tuple(None if arr is None else arr[keep] for arr in arrays)
        return filtered

    def filter_attempt_timesteps(
        self,
        actions: np.ndarray,
        dones: np.ndarray,
        timestamp_values: Optional[np.ndarray],
        pressure_values: Optional[np.ndarray],
        pressure_state_values: Optional[np.ndarray],
        effective_pressure_values: Optional[np.ndarray],
        resistance_values: Optional[np.ndarray],
        resistance_slope_values: Optional[np.ndarray],
        resistance_input_values: Optional[np.ndarray],
        current_values: Optional[np.ndarray],
        voltage_values: Optional[np.ndarray],
        stage_positions: Optional[np.ndarray],
        pipette_positions: Optional[np.ndarray],
        camera_frames: Optional[np.ndarray],
        force_keep_mask: Optional[np.ndarray] = None,
    ):
        """Filter invalid and inactive timesteps while keeping all payloads aligned."""

        payloads: List[Optional[np.ndarray]] = [
            dones,
            timestamp_values,
            pressure_values,
            pressure_state_values,
            effective_pressure_values,
            resistance_values,
            resistance_slope_values,
            resistance_input_values,
            current_values,
            voltage_values,
            stage_positions,
            pipette_positions,
            camera_frames,
        ]

        invalid_removed = 0
        if self.skip_invalid_observations:
            keep = self._finite_row_mask(actions)
            for array in payloads:
                if array is None:
                    continue
                keep &= self._finite_row_mask(array)

            if not np.all(keep):
                invalid_removed = int(np.count_nonzero(~keep))
                actions = actions[keep]
                payloads = [None if array is None else array[keep] for array in payloads]
                if force_keep_mask is not None:
                    force_keep_mask = np.asarray(force_keep_mask, dtype=bool).reshape(-1)[keep]

        before_inactive = actions.shape[0]
        filtered = self.filter_inactive_actions(
            actions,
            *payloads,
            force_keep_mask=force_keep_mask,
        )
        actions = filtered[0]
        payloads = list(filtered[1:])
        inactive_removed = before_inactive - actions.shape[0]

        dones = payloads[0]
        if dones is None:
            dones = np.zeros(actions.shape[0], dtype=np.float64)
        else:
            dones = np.zeros(dones.shape[0], dtype=dones.dtype)
        if dones.size:
            dones[-1] = 1
        payloads[0] = dones

        (
            dones,
            timestamp_values,
            pressure_values,
            pressure_state_values,
            effective_pressure_values,
            resistance_values,
            resistance_slope_values,
            resistance_input_values,
            current_values,
            voltage_values,
            stage_positions,
            pipette_positions,
            camera_frames,
        ) = payloads

        return (
            actions,
            dones,
            timestamp_values,
            pressure_values,
            pressure_state_values,
            effective_pressure_values,
            resistance_values,
            resistance_slope_values,
            resistance_input_values,
            current_values,
            voltage_values,
            stage_positions,
            pipette_positions,
            camera_frames,
            invalid_removed,
            inactive_removed,
        )

    # --- CSV conversion utilities ----------------------------------------
    def convert_graph_recording_csv_to_new_format(self, demo_file_path: str) -> None:
        """Rewrite ``graph_recording.csv`` with semicolon-separated fields.

        Parameters
        ----------
        demo_file_path:
            Name of the rig-recorder folder containing ``graph_recording.csv``.
        """
        file_path = Path("experiments/Data/rig_recorder_data") / demo_file_path / "graph_recording.csv"
        graph_values = pd.read_csv(file_path, delimiter=":")

        converted_file_strings = ["timestamp;pressure;resistance;current;voltage\n"]
        for _, row in graph_values.iterrows():
            timestamp = row[1][:-10]
            pressure = row[2][:-12]
            resistance = row[3][:-9]
            current = row[4][:-8]
            voltage = row[5]
            converted_file_strings.append(
                f"{timestamp};{pressure};{resistance};{current};{voltage}\n"
            )

        with open(file_path, "w") as f:
            f.writelines(converted_file_strings)

        print(pd.read_csv(file_path, delimiter=";"))

    def convert_movement_recording_csv_to_new_format(self, demo_file_path: str) -> None:
        """Rewrite ``cv_movement_recording.csv`` into the new semicolon format."""
        file_path = Path("experiments/Data/rig_recorder_data") / demo_file_path / "cv_movement_recording.csv"
        movement_values = pd.read_csv(file_path, delimiter=":")

        converted_file_strings = ["timestamp;st_x;st_y;st_z;pi_x;pi_y;pi_z\n"]
        for _, row in movement_values.iterrows():
            timestamp = row[1][:-6]
            st_x = row[2][:-6]
            st_y = row[3][:-6]
            st_z = row[4][:-6]
            pi_x = row[5][:-5]
            pi_y = row[6][:-5]
            pi_z = row[7]
            converted_file_strings.append(
                f"{timestamp};{st_x};{st_y};{st_z};{pi_x};{pi_y};{pi_z}\n"
            )

        with open(file_path, "w") as f:
            f.writelines(converted_file_strings)

        print(pd.read_csv(file_path, delimiter=";"))

    # --- Experiment loading ----------------------------------------------
    def load_experiment_data(
        self, rig_recorder_data_folder: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load graph and movement tables for a given experiment folder."""
        base = self._resolve_rig_recorder_folder(rig_recorder_data_folder)
        movement_path = None
        movement_candidates = (
            ("cv_movement_recording.csv", "movement_recording.csv")
            if self.prefer_cv_movement
            else ("movement_recording.csv", "cv_movement_recording.csv")
        )
        for name in movement_candidates:
            candidate = base / name
            if candidate.exists():
                movement_path = candidate
                break
        if movement_path is None:
            expected = ", ".join(movement_candidates)
            raise FileNotFoundError(
                f"Missing movement recording in {base}. Expected one of: {expected}."
            )
        self._using_cv_movement_file = movement_path.name.lower().startswith("cv")
        movement_values = pd.read_csv(movement_path, delimiter=";").to_numpy()
        graph_file = base / "graph_recording.csv"
        selector = self.observation_selector
        graph_required = any(
            (
                selector.include_pressure,
                selector.include_resistance,
                selector.include_current,
                selector.include_voltage,
            )
        )
        if graph_file.exists():
            graph_values = pd.read_csv(graph_file, delimiter=";").to_numpy()
        else:
            if graph_required:
                raise FileNotFoundError(f"Missing graph recording: {graph_file}.")
            timestamps = movement_values[:, 0].astype(np.float64, copy=True)
            # Fall back to timestamps only when graph-dependent features are disabled.
            graph_values = timestamps.reshape(-1, 1)
        if self.freq_mask > 1:
            step = self.freq_mask
            graph_values = graph_values[::step]
            movement_values = movement_values[::step]
        return graph_values, movement_values

    @staticmethod
    def _nearest_timestamp_indices(
        reference_timestamps: np.ndarray,
        target_timestamps: np.ndarray,
    ) -> np.ndarray:
        """Return indices of the nearest reference timestamp for each target timestamp."""

        if target_timestamps.size == 0:
            return np.zeros(0, dtype=np.int64)
        if reference_timestamps.size == 0:
            raise ValueError("reference_timestamps must be non-empty")

        right = np.searchsorted(reference_timestamps, target_timestamps)
        left = np.clip(right - 1, 0, len(reference_timestamps) - 1)
        right = np.clip(right, 0, len(reference_timestamps) - 1)
        choose_right = np.abs(reference_timestamps[right] - target_timestamps) < np.abs(
            reference_timestamps[left] - target_timestamps
        )
        return np.where(choose_right, right, left)

    @staticmethod
    def _filter_plain_pressure_events(
        plain_events: pd.DataFrame,
        raw_timestamps: np.ndarray,
    ) -> pd.DataFrame:
        """Keep fallback pressure logs only when no raw controller event occurred nearby."""

        if plain_events.empty or raw_timestamps.size == 0:
            return plain_events

        plain_timestamps = plain_events["timestamp"].to_numpy(dtype=np.float64)
        right = np.searchsorted(raw_timestamps, plain_timestamps)
        left = np.clip(right - 1, 0, raw_timestamps.size - 1)
        right = np.clip(right, 0, raw_timestamps.size - 1)
        nearest_diff = np.minimum(
            np.abs(raw_timestamps[left] - plain_timestamps),
            np.abs(raw_timestamps[right] - plain_timestamps),
        )
        keep_mask = nearest_diff > _PRESSURE_EVENT_DEDUP_TOLERANCE_SECONDS
        return plain_events.loc[keep_mask].copy()

    def _parse_pressure_log_events(
        self,
        log_values: pd.DataFrame,
        *,
        log_file: Path,
    ) -> pd.DataFrame:
        """Return pressure setpoint and ATM toggle events from a day log."""

        required_columns = {"Time(HH:MM:SS)", "Time(ms)", "Message"}
        missing = sorted(required_columns.difference(log_values.columns))
        if missing:
            raise RuntimeError(f"Missing required log columns in {log_file}: {', '.join(missing)}")

        timestamps = _local_time_columns_to_epoch(log_values["Time(HH:MM:SS)"], log_values["Time(ms)"])
        messages = log_values["Message"].map(_normalize_log_message)
        valid_mask = (~timestamps.isna()) & messages.ne("")
        filtered_logs = pd.DataFrame(
            {
                "timestamp": timestamps.loc[valid_mask].astype(np.float64),
                "message": messages.loc[valid_mask],
            }
        )
        if filtered_logs.empty:
            return pd.DataFrame(columns=_PRESSURE_LOG_EVENT_COLUMNS)

        raw_matches = filtered_logs["message"].str.extract(_PRESSURE_RAW_RE)
        raw_mask = raw_matches["value"].notna()
        raw_events = pd.DataFrame(
            {
                "timestamp": filtered_logs.loc[raw_mask, "timestamp"].to_numpy(dtype=np.float64),
                "event_kind": "pressure",
                "commanded_pressure_mbar": raw_matches.loc[raw_mask, "value"].astype(np.float64).to_numpy(),
                "pressure_atm_state": np.nan,
                "zap_command": np.nan,
            }
        )

        plain_matches = filtered_logs["message"].str.extract(_PRESSURE_PLAIN_RE)
        plain_mask = plain_matches["value"].notna()
        plain_events = pd.DataFrame(
            {
                "timestamp": filtered_logs.loc[plain_mask, "timestamp"].to_numpy(dtype=np.float64),
                "event_kind": "pressure",
                "commanded_pressure_mbar": plain_matches.loc[plain_mask, "value"].astype(np.float64).to_numpy(),
                "pressure_atm_state": np.nan,
                "zap_command": np.nan,
            }
        )
        if raw_events.empty:
            raw_timestamps = np.zeros(0, dtype=np.float64)
        else:
            raw_timestamps = np.sort(raw_events["timestamp"].to_numpy(dtype=np.float64))
        plain_events = self._filter_plain_pressure_events(plain_events, raw_timestamps)

        atm_mask = filtered_logs["message"].str.startswith("Switching to ATM", na=False)
        atm_events = pd.DataFrame(
            {
                "timestamp": filtered_logs.loc[atm_mask, "timestamp"].to_numpy(dtype=np.float64),
                "event_kind": "atm_state",
                "commanded_pressure_mbar": np.nan,
                "pressure_atm_state": np.ones(int(atm_mask.sum()), dtype=np.float64),
                "zap_command": np.nan,
            }
        )

        pressure_mode_mask = filtered_logs["message"].str.startswith("Switching to Pressure", na=False)
        pressure_mode_events = pd.DataFrame(
            {
                "timestamp": filtered_logs.loc[pressure_mode_mask, "timestamp"].to_numpy(dtype=np.float64),
                "event_kind": "atm_state",
                "commanded_pressure_mbar": np.nan,
                "pressure_atm_state": np.zeros(int(pressure_mode_mask.sum()), dtype=np.float64),
                "zap_command": np.nan,
            }
        )

        zap_mask = filtered_logs["message"].str.match(_ZAP_RE, na=False)
        zap_events = pd.DataFrame(
            {
                "timestamp": filtered_logs.loc[zap_mask, "timestamp"].to_numpy(dtype=np.float64),
                "event_kind": "zap",
                "commanded_pressure_mbar": np.nan,
                "pressure_atm_state": np.nan,
                "zap_command": np.ones(int(zap_mask.sum()), dtype=np.float64),
            }
        )

        event_frames = [
            frame
            for frame in (raw_events, plain_events, atm_events, pressure_mode_events, zap_events)
            if not frame.empty
        ]
        if not event_frames:
            return pd.DataFrame(columns=_PRESSURE_LOG_EVENT_COLUMNS)

        pressure_events = pd.concat(event_frames, ignore_index=True)
        pressure_events = pressure_events.drop_duplicates(
            subset=[
                "timestamp",
                "event_kind",
                "commanded_pressure_mbar",
                "pressure_atm_state",
                "zap_command",
            ]
        )
        pressure_events = pressure_events.sort_values("timestamp", kind="stable").reset_index(drop=True)
        return pressure_events

    def _load_pressure_log_events(self, rig_recorder_data_folder: str) -> pd.DataFrame:
        """Load and cache pressure controller events for the recording day."""

        day_token = self._rig_recorder_folder_name(rig_recorder_data_folder).split("-", 1)[0]
        cached = self._pressure_event_cache.get(day_token)
        if cached is not None:
            return cached

        log_file = DATA_ROOT / "log_data" / f"logs_{day_token}.csv"
        if not log_file.exists():
            raise FileNotFoundError(
                f"Day-log pressure events are required, but {log_file} was not found."
            )

        try:
            log_values = _read_csv_with_fallback(log_file, on_bad_lines="skip")
        except Exception as exc:
            raise RuntimeError(f"Failed reading pressure-action log file {log_file}: {exc}") from exc

        pressure_events = self._parse_pressure_log_events(log_values, log_file=log_file)
        if pressure_events.empty:
            raise RuntimeError(
                f"No parsable day-log pressure events were found in {log_file}."
            )

        self._pressure_event_cache[day_token] = pressure_events
        return pressure_events

    def get_attempt_pressure_observation_values(
        self,
        attempt_graph_values: np.ndarray,
        pressure_events: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return command state known at or before each graph timestamp."""

        graph_timestamps = attempt_graph_values[:, 0].astype(np.float64, copy=False)
        if graph_timestamps.size == 0:
            empty = np.zeros(0, dtype=np.float64)
            return empty, empty

        raw_commanded_pressure = np.zeros(graph_timestamps.shape[0], dtype=np.float64)
        atm_state = np.zeros(graph_timestamps.shape[0], dtype=np.float64)

        pressure_changes = pressure_events.loc[
            pressure_events["event_kind"] == "pressure",
            ["timestamp", "commanded_pressure_mbar"],
        ].dropna(subset=["timestamp", "commanded_pressure_mbar"])
        pressure_changes = pressure_changes.sort_values("timestamp", kind="stable")
        if not pressure_changes.empty:
            pressure_timestamps = pressure_changes["timestamp"].to_numpy(dtype=np.float64)
            pressure_values = np.rint(
                pressure_changes["commanded_pressure_mbar"].to_numpy(dtype=np.float64)
            )
            pressure_indices = np.searchsorted(
                pressure_timestamps,
                graph_timestamps,
                side="right",
            ) - 1
            valid_pressure = pressure_indices >= 0
            raw_commanded_pressure[valid_pressure] = pressure_values[
                pressure_indices[valid_pressure]
            ]

        state_changes = pressure_events.loc[
            pressure_events["event_kind"] == "atm_state",
            ["timestamp", "pressure_atm_state"],
        ].dropna(subset=["timestamp", "pressure_atm_state"])
        state_changes = state_changes.sort_values("timestamp", kind="stable")
        if not state_changes.empty:
            state_timestamps = state_changes["timestamp"].to_numpy(dtype=np.float64)
            state_values = state_changes["pressure_atm_state"].to_numpy(dtype=np.float64)
            state_indices = np.searchsorted(
                state_timestamps,
                graph_timestamps,
                side="right",
            ) - 1
            valid_state = state_indices >= 0
            atm_state[valid_state] = state_values[state_indices[valid_state]]

        return raw_commanded_pressure, atm_state

    def get_attempt_pressure_action_values(
        self,
        attempt_graph_values: np.ndarray,
        pressure_events: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return target pressure setpoint and ATM action state after each observation."""

        current_pressure, current_atm_state = self.get_attempt_pressure_observation_values(
            attempt_graph_values,
            pressure_events,
        )
        if current_pressure.size == 0:
            return current_pressure, current_atm_state

        target_pressure = current_pressure.copy()
        target_atm_state = current_atm_state.copy()
        if target_pressure.shape[0] > 1:
            target_pressure[:-1] = current_pressure[1:]
            target_atm_state[:-1] = current_atm_state[1:]
        return (
            target_pressure.astype(np.float64, copy=False),
            target_atm_state.astype(np.float64, copy=False),
        )

    def get_attempt_zap_action_values(
        self,
        attempt_graph_values: np.ndarray,
        pressure_events: pd.DataFrame,
    ) -> np.ndarray:
        """Return a binary zap action for graph rows preceding logged zap events."""

        graph_timestamps = attempt_graph_values[:, 0].astype(np.float64, copy=False)
        zap_actions = np.zeros(graph_timestamps.shape[0], dtype=np.float64)
        if graph_timestamps.size == 0 or "zap_command" not in pressure_events.columns:
            return zap_actions

        zap_events = pressure_events.loc[
            pressure_events["event_kind"] == "zap",
            ["timestamp", "zap_command"],
        ].dropna(subset=["timestamp", "zap_command"])
        if zap_events.empty:
            return zap_actions

        zap_timestamps = zap_events["timestamp"].to_numpy(dtype=np.float64)
        zap_values = zap_events["zap_command"].to_numpy(dtype=np.float64)
        positive_zaps = (
            (zap_values >= 0.5)
            & (zap_timestamps >= graph_timestamps[0])
            & (zap_timestamps <= graph_timestamps[-1])
        )
        zap_indices = np.searchsorted(graph_timestamps, zap_timestamps[positive_zaps], side="right") - 1
        valid_zaps = (zap_indices >= 0) & (zap_indices < graph_timestamps.shape[0])
        zap_actions[zap_indices[valid_zaps]] = 1.0
        return zap_actions

    def get_attempt_pressure_state_observation_values(
        self,
        attempt_graph_values: np.ndarray,
        pressure_events: pd.DataFrame,
    ) -> np.ndarray:
        """Return the pressure controller ATM-state trace aligned to graph rows."""

        _, atm_state = self.get_attempt_pressure_observation_values(
            attempt_graph_values,
            pressure_events,
        )
        return atm_state.astype(np.float64, copy=False)

    @staticmethod
    def get_effective_pressure_observation_values(
        commanded_pressure: np.ndarray,
        atm_state: np.ndarray,
    ) -> np.ndarray:
        """Return pressure applied to the pipette after accounting for ATM mode."""

        pressure = np.asarray(commanded_pressure, dtype=np.float64).reshape(-1)
        atm = np.asarray(atm_state, dtype=np.float64).reshape(-1)
        if pressure.shape[0] != atm.shape[0]:
            raise ValueError("commanded_pressure and atm_state must have matching lengths")
        return np.where(atm >= 0.5, 0.0, pressure).astype(np.float64, copy=False)

    @staticmethod
    def get_applied_pressure_action_values(
        commanded_pressure: np.ndarray,
        atm_state: np.ndarray,
    ) -> np.ndarray:
        """Return the single Gigaseal action: applied pressure, or 0 in ATM mode."""

        pressure = np.asarray(commanded_pressure, dtype=np.float64).reshape(-1)
        atm = np.asarray(atm_state, dtype=np.float64).reshape(-1)
        if pressure.shape[0] != atm.shape[0]:
            raise ValueError("commanded_pressure and atm_state must have matching lengths")
        return np.where(atm > 0.0, 0.0, pressure).astype(np.float64, copy=False)

    @staticmethod
    def get_binary_gigaseal_pressure_action_values(
        current_commanded_pressure: np.ndarray,
        current_atm_state: np.ndarray,
        target_commanded_pressure: np.ndarray,
        target_atm_state: np.ndarray,
    ) -> np.ndarray:
        """Return one-hot Gigaseal pressure commands for -5, +5, and reset."""

        current_applied_pressure = SimpleDatasetBuilder.get_applied_pressure_action_values(
            current_commanded_pressure,
            current_atm_state,
        )
        target_applied_pressure = SimpleDatasetBuilder.get_applied_pressure_action_values(
            target_commanded_pressure,
            target_atm_state,
        )
        target_atm = np.asarray(target_atm_state, dtype=np.float64).reshape(-1)
        if current_applied_pressure.shape[0] != target_applied_pressure.shape[0]:
            raise ValueError("current and target pressure arrays must have matching lengths")
        if target_atm.shape[0] != target_applied_pressure.shape[0]:
            raise ValueError("target_atm_state must match the pressure array length")

        actions = np.zeros((target_applied_pressure.shape[0], 3), dtype=np.float64)
        delta = target_applied_pressure - current_applied_pressure
        reset_mask = target_atm > 0.0
        actions[reset_mask, 2] = 1.0
        non_reset = ~reset_mask
        actions[non_reset & (delta < 0.0), 0] = 1.0
        actions[non_reset & (delta > 0.0), 1] = 1.0
        return actions

    @staticmethod
    def get_action_event_mask(
        actions: np.ndarray,
        action_labels: Sequence[str],
        tolerance: float = 0.0,
    ) -> np.ndarray:
        """Return rows where the final written action payload changes or fires."""

        num_rows = actions.shape[0]
        if num_rows == 0 or actions.shape[1] == 0:
            return np.zeros(num_rows, dtype=bool)

        labels = list(action_labels)
        if len(labels) != actions.shape[1]:
            return np.any(np.abs(actions) > tolerance, axis=1)

        action_events = np.zeros(num_rows, dtype=bool)
        for idx, label in enumerate(labels):
            if label == "observations_until_next_action":
                continue
            values = np.asarray(actions[:, idx], dtype=np.float64)
            if label in {"gigaseal_delta_neg_5", "gigaseal_delta_pos_5", "gigaseal_reset"}:
                action_events |= np.abs(values) > tolerance
            elif label in {"commanded_pressure_mbar", "pressure_atm_state", "applied_pressure_mbar"}:
                changes = np.zeros(num_rows, dtype=bool)
                if num_rows > 1:
                    changes[1:] = np.abs(np.diff(values)) > tolerance
                action_events |= changes
            else:
                action_events |= np.abs(values) > tolerance
        return action_events

    @classmethod
    def get_observations_since_last_action_values(
        cls,
        actions: np.ndarray,
        action_labels: Sequence[str],
        tolerance: float = 0.0,
    ) -> np.ndarray:
        """Count rows since the last action event in the final written sequence."""

        num_rows = actions.shape[0]
        if num_rows == 0:
            return np.zeros(0, dtype=np.float64)

        action_events = cls.get_action_event_mask(actions, action_labels, tolerance=tolerance)
        past_action_events = np.zeros_like(action_events)
        if num_rows > 1:
            past_action_events[1:] = action_events[:-1]

        counts = np.zeros(num_rows, dtype=np.float64)
        since_action = 0
        seen_action = False
        for idx, is_action in enumerate(past_action_events):
            if is_action:
                since_action = 0
                seen_action = True
            elif seen_action:
                since_action += 1
            counts[idx] = since_action
        return counts

    @classmethod
    def get_observations_until_next_action_values(
        cls,
        actions: np.ndarray,
        action_labels: Sequence[str],
        tolerance: float = 0.0,
    ) -> np.ndarray:
        """Count retained observation rows until the next future action event."""

        num_rows = actions.shape[0]
        if num_rows == 0:
            return np.zeros(0, dtype=np.float64)

        action_events = cls.get_action_event_mask(actions, action_labels, tolerance=tolerance)
        future_action_events = np.zeros_like(action_events)
        if num_rows > 1:
            future_action_events[:-1] = action_events[1:]

        counts = np.zeros(num_rows, dtype=np.float64)
        until_action = 0
        seen_action = False
        for idx in range(num_rows - 1, -1, -1):
            if future_action_events[idx]:
                until_action = 0
                seen_action = True
            elif seen_action:
                until_action += 1
            counts[idx] = until_action
        return counts

    @classmethod
    def get_time_since_last_action_values(
        cls,
        actions: np.ndarray,
        action_labels: Sequence[str],
        timestamps: np.ndarray,
        tolerance: float = 0.0,
    ) -> np.ndarray:
        """Return elapsed seconds since the last action event in the final sequence."""

        num_rows = actions.shape[0]
        if num_rows == 0:
            return np.zeros(0, dtype=np.float64)

        times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if times.shape[0] != num_rows:
            raise ValueError("timestamps and actions must have matching lengths")

        action_events = cls.get_action_event_mask(actions, action_labels, tolerance=tolerance)
        past_action_events = np.zeros_like(action_events)
        if num_rows > 1:
            past_action_events[1:] = action_events[:-1]
        elapsed = np.zeros(num_rows, dtype=np.float64)
        last_action_time: Optional[float] = None
        for idx, is_action in enumerate(past_action_events):
            if is_action:
                last_action_time = float(times[idx])
                elapsed[idx] = 0.0
            elif last_action_time is not None:
                elapsed[idx] = max(0.0, float(times[idx]) - last_action_time)
        return elapsed

    @classmethod
    def get_pre_action_event_keep_mask(
        cls,
        actions: np.ndarray,
        action_labels: Sequence[str],
        radius: int,
        tolerance: float = 0.0,
    ) -> np.ndarray:
        """Protect the retained wait rows from later inactive-row pruning."""

        num_rows = actions.shape[0]
        keep = np.zeros(num_rows, dtype=bool)
        if num_rows == 0 or actions.shape[1] == 0:
            return keep

        action_events = cls.get_action_event_mask(
            actions,
            action_labels,
            tolerance=tolerance,
        )
        first_row = np.asarray(actions[0], dtype=np.float64)
        action_events[0] = action_events[0] or bool(
            np.any(np.isfinite(first_row) & (np.abs(first_row) > tolerance))
        )
        event_indices = np.flatnonzero(action_events)
        if event_indices.size == 0:
            return keep

        first_event_idx = int(event_indices[0])
        start_idx = max(0, first_event_idx - max(0, int(radius)))
        keep[start_idx:first_event_idx] = True
        return keep

    def _get_state_attempt_records(
        self,
        rig_recorder_data_folder: str,
        valid_start: float,
        valid_end: float,
        *,
        include_failed: bool = False,
    ) -> Dict[str, List[_StateAttemptRecord]]:
        """Extract selected state attempt windows using state-recorder JSON logs."""

        state_attempts: Dict[str, List[_StateAttemptRecord]] = {}
        state_root = self._resolve_state_recorder_root(rig_recorder_data_folder)
        day_token = self._rig_recorder_folder_name(rig_recorder_data_folder).split('-', 1)[0]
        tolerance = 0.5
        allowed_outcomes = {_STATE_OUTCOME_SUCCESS}
        if include_failed:
            allowed_outcomes.add(_STATE_OUTCOME_FAILURE)

        if state_root.exists():
            for day_dir in sorted(state_root.glob(f"{day_token}*")):
                if not day_dir.is_dir():
                    continue
                for attempt_dir in sorted(day_dir.glob("attempt_*")):
                    if not attempt_dir.is_dir():
                        continue
                    for json_path in sorted(attempt_dir.glob("*.json")):
                        try:
                            with open(json_path, "r", encoding="utf-8") as fh:
                                payload = json.load(fh)
                        except (OSError, json.JSONDecodeError):
                            continue

                        outcome = payload.get("outcome")
                        started = payload.get("started")
                        finished = payload.get("finished")
                        if outcome is None or started is None or finished is None:
                            continue
                        try:
                            outcome_code = int(outcome)
                            started = self._normalize_legacy_epoch_timestamp(float(started))
                            finished = self._normalize_legacy_epoch_timestamp(float(finished))
                        except (TypeError, ValueError):
                            continue
                        if outcome_code not in allowed_outcomes:
                            continue
                        if finished <= started:
                            continue
                        if finished < (valid_start - tolerance) or started > (valid_end + tolerance):
                            continue

                        stem = json_path.stem
                        parts = stem.split("_")
                        if len(parts) >= 3:
                            state_name = "_".join(parts[1:-1])
                        else:
                            state_name = stem
                        slug = _slugify_state_name(state_name)

                        system_mode = payload.get("system_mode")
                        try:
                            system_mode_code = int(system_mode) if system_mode is not None else None
                        except (TypeError, ValueError):
                            system_mode_code = None
                        state_attempts.setdefault(slug, []).append(
                            _StateAttemptRecord(
                                started=started,
                                finished=finished,
                                outcome_code=outcome_code,
                                system_mode_code=system_mode_code,
                                attempt_json=json_path.as_posix(),
                            )
                        )

        for attempts in state_attempts.values():
            attempts.sort(key=lambda record: record.started)

        return state_attempts

    def get_timestamps_for_all_successful_state_attempts(
        self,
        rig_recorder_data_folder: str,
        valid_start: float,
        valid_end: float,
    ) -> Dict[str, List[Tuple[float, float]]]:
        """Extract successful attempt windows for each state using JSON logs."""

        state_attempts = self._get_state_attempt_records(
            rig_recorder_data_folder,
            valid_start,
            valid_end,
            include_failed=False,
        )
        return {
            state_name: [(record.started, record.finished) for record in records]
            for state_name, records in state_attempts.items()
        }

    def _get_state_attempt_records_for_export(
        self,
        rig_recorder_data_folder: str,
        valid_start: float,
        valid_end: float,
    ) -> Dict[str, List[_StateAttemptRecord]]:
        include_failed = bool(getattr(self, "include_failed_demos", False))
        if not include_failed and "get_timestamps_for_all_successful_state_attempts" in self.__dict__:
            legacy_attempts = self.get_timestamps_for_all_successful_state_attempts(
                rig_recorder_data_folder,
                valid_start,
                valid_end,
            )
            return {
                state_name: [
                    _StateAttemptRecord(
                        started=float(started),
                        finished=float(finished),
                    )
                    for started, finished in attempt_ranges
                ]
                for state_name, attempt_ranges in legacy_attempts.items()
            }
        return self._get_state_attempt_records(
            rig_recorder_data_folder,
            valid_start,
            valid_end,
            include_failed=include_failed,
        )

    # --- Graph / movement alignment -------------------------------------
    def truncate_graph_values(
        self, graph_values: np.ndarray, first_timestamp: float, last_timestamp: float
    ) -> np.ndarray:
        """Return graph rows closest to the provided time interval (inclusive)."""
        ts = graph_values[:, 0]
        i0 = np.searchsorted(ts, first_timestamp, side="left")
        if i0 == len(ts):
            i0 = len(ts) - 1
        elif i0 and abs(ts[i0 - 1] - first_timestamp) < abs(ts[i0] - first_timestamp):
            i0 -= 1

        i1 = np.searchsorted(ts, last_timestamp, side="right") - 1
        if i1 < 0:
            i1 = 0
        elif i1 + 1 < len(ts) and abs(ts[i1 + 1] - last_timestamp) < abs(ts[i1] - last_timestamp):
            i1 += 1
        return graph_values[i0 : i1 + 1]

    def truncate_attempt_for_state(
        self,
        attempt_graph_values: np.ndarray,
        attempt_movement_values: np.ndarray,
        state_name: str,
        pressure_events: Optional[pd.DataFrame] = None,
        attempt_first_timestamp: Optional[float] = None,
        attempt_last_timestamp: Optional[float] = None,
        recording_graph_values: Optional[np.ndarray] = None,
        recording_movement_values: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, bool, bool, Optional[str]]:
        """Apply state-specific trimming rules before feature extraction."""

        is_gigaseal_state = _is_gigaseal_state_name(state_name)
        is_break_in_state = _is_break_in_state_name(state_name)
        if attempt_graph_values.shape[0] == 0:
            return attempt_graph_values, attempt_movement_values, False, False, None

        trimmed_for_gigaseal_start = False
        if is_break_in_state and getattr(self, "break_in_event_window_enabled", True):
            if pressure_events is None:
                raise RuntimeError(
                    "Break-in dataset trimming requires parsed day-log pressure events."
                )

            attempt_graph_values, attempt_movement_values, _, skip_reason = (
                self._trim_break_in_attempt_to_action_event_span(
                    attempt_graph_values,
                    attempt_movement_values,
                    pressure_events,
                    recording_graph_values=recording_graph_values,
                    recording_movement_values=recording_movement_values,
                )
            )
            if skip_reason is not None:
                return attempt_graph_values, attempt_movement_values, False, False, skip_reason

        if is_gigaseal_state and getattr(self, "gigaseal_event_window_enabled", True):
            if pressure_events is None:
                raise RuntimeError(
                    "Gigaseal dataset trimming requires parsed day-log pressure events."
                )

            attempt_graph_values, attempt_movement_values, trimmed_for_gigaseal_start, skip_reason = (
                self._trim_gigaseal_attempt_to_first_action_event_window(
                    attempt_graph_values,
                    attempt_movement_values,
                    pressure_events,
                    recording_graph_values=recording_graph_values,
                    recording_movement_values=recording_movement_values,
                )
            )
            if skip_reason is not None:
                return (
                    attempt_graph_values,
                    attempt_movement_values,
                    trimmed_for_gigaseal_start,
                    False,
                    skip_reason,
                )

        elif is_gigaseal_state and getattr(self, "gigaseal_start_trim_enabled", False):
            if pressure_events is None:
                raise RuntimeError(
                    "Gigaseal dataset trimming requires parsed day-log pressure events."
                )
            if attempt_first_timestamp is None or attempt_last_timestamp is None:
                raise RuntimeError(
                    "Gigaseal dataset trimming requires the original state-attempt timestamps."
                )

            attempt_graph_values, attempt_movement_values, trimmed_for_gigaseal_start, skip_reason = (
                self._trim_gigaseal_attempt_start_at_first_atm(
                    attempt_graph_values,
                    attempt_movement_values,
                    pressure_events,
                    attempt_first_timestamp,
                    attempt_last_timestamp,
                )
            )
            if skip_reason is not None:
                return (
                    attempt_graph_values,
                    attempt_movement_values,
                    trimmed_for_gigaseal_start,
                    False,
                    skip_reason,
                )

        if not is_gigaseal_state or not self.gigaseal_resistance_cutoff_enabled:
            return attempt_graph_values, attempt_movement_values, trimmed_for_gigaseal_start, False, None

        resistance_values = attempt_graph_values[:, 2].astype(np.float64)
        cutoff_hits = np.flatnonzero(
            np.isfinite(resistance_values) & (resistance_values >= self.gigaseal_resistance_cutoff)
        )
        if cutoff_hits.size == 0:
            return attempt_graph_values, attempt_movement_values, trimmed_for_gigaseal_start, False, None

        end_idx = int(cutoff_hits[0]) + 1
        trimmed = end_idx < attempt_graph_values.shape[0]
        return (
            attempt_graph_values[:end_idx],
            attempt_movement_values[:end_idx],
            trimmed_for_gigaseal_start,
            trimmed,
            None,
        )

    def _trim_break_in_attempt_to_action_event_span(
        self,
        attempt_graph_values: np.ndarray,
        attempt_movement_values: np.ndarray,
        pressure_events: pd.DataFrame,
        recording_graph_values: Optional[np.ndarray] = None,
        recording_movement_values: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, bool, Optional[str]]:
        """Keep break-in rows from before the first event through after the last."""

        num_rows = attempt_graph_values.shape[0]
        if num_rows == 0:
            return attempt_graph_values, attempt_movement_values, False, None

        _, atm_state = self.get_attempt_pressure_action_values(
            attempt_graph_values,
            pressure_events,
        )
        break_in_components = [atm_state]
        action_labels = ["pressure_atm_state"]
        if self.action_selector.include_break_in_zap:
            zap_command = self.get_attempt_zap_action_values(
                attempt_graph_values,
                pressure_events,
            )
            break_in_components.append(zap_command)
            action_labels.append("zap_command")

        actions = np.column_stack(break_in_components).astype(np.float64, copy=False)
        action_events = self.get_action_event_mask(actions, action_labels, tolerance=0.0)
        event_indices = np.flatnonzero(action_events)
        if event_indices.size == 0:
            return (
                attempt_graph_values,
                attempt_movement_values,
                False,
                "no break-in ATM or zap action event was found in the attempt",
            )

        radius = max(0, int(getattr(self, "break_in_event_window_radius", 15)))
        first_event_idx = int(event_indices[0])
        last_event_idx = int(event_indices[-1])
        if recording_graph_values is not None and recording_movement_values is not None:
            first_event_idx = self._nearest_graph_row_index(
                recording_graph_values,
                float(attempt_graph_values[first_event_idx, 0]),
            )
            last_event_idx = self._nearest_graph_row_index(
                recording_graph_values,
                float(attempt_graph_values[last_event_idx, 0]),
            )
            start_idx = max(0, first_event_idx - radius)
            end_idx = min(recording_graph_values.shape[0], last_event_idx + radius + 1)
            graph_values = recording_graph_values[start_idx:end_idx]
            movement_values = self.associate_attempt_movement_and_graph_values(
                graph_values,
                recording_movement_values,
            )
            trimmed = (
                graph_values.shape[0] != attempt_graph_values.shape[0]
                or graph_values[0, 0] != attempt_graph_values[0, 0]
                or graph_values[-1, 0] != attempt_graph_values[-1, 0]
            )
            return graph_values, movement_values, trimmed, None

        start_idx = max(0, first_event_idx - radius)
        end_idx = min(num_rows, last_event_idx + radius + 1)
        trimmed = start_idx > 0 or end_idx < num_rows
        return (
            attempt_graph_values[start_idx:end_idx],
            attempt_movement_values[start_idx:end_idx],
            trimmed,
            None,
        )

    @staticmethod
    def _nearest_graph_row_index(graph_values: np.ndarray, timestamp: float) -> int:
        """Return the closest graph row index for a timestamp."""

        timestamps = graph_values[:, 0].astype(np.float64, copy=False)
        right = int(np.searchsorted(timestamps, timestamp, side="left"))
        if right <= 0:
            return 0
        if right >= timestamps.shape[0]:
            return timestamps.shape[0] - 1
        left = right - 1
        if abs(timestamps[right] - timestamp) < abs(timestamps[left] - timestamp):
            return right
        return left

    def _get_gigaseal_event_window_actions(
        self,
        attempt_graph_values: np.ndarray,
        attempt_movement_values: np.ndarray,
        pressure_events: pd.DataFrame,
    ) -> Tuple[np.ndarray, List[str]]:
        """Return the action payload used to find the first gigaseal event."""

        selected_components: List[np.ndarray] = []
        action_labels: List[str] = []

        commanded_pressure, atm_state = self.get_attempt_pressure_action_values(
            attempt_graph_values,
            pressure_events,
        )
        selector = self.action_selector
        if selector.binary_gigaseal_pressure_action:
            current_commanded_pressure, current_atm_state = self.get_attempt_pressure_observation_values(
                attempt_graph_values,
                pressure_events,
            )
            binary_pressure_actions = self.get_binary_gigaseal_pressure_action_values(
                current_commanded_pressure,
                current_atm_state,
                commanded_pressure,
                atm_state,
            )
            selected_components.append(binary_pressure_actions.astype(np.float64, copy=False))
        elif selector.combine_gigaseal_pressure_action:
            applied_pressure = self.get_applied_pressure_action_values(
                commanded_pressure,
                atm_state,
            )
            selected_components.append(
                applied_pressure.reshape(-1, 1).astype(np.float64, copy=False)
            )
        else:
            selected_components.append(
                np.column_stack([commanded_pressure, atm_state]).astype(np.float64, copy=False)
            )
        action_labels.extend(selector.gigaseal_pressure_axis_labels())

        timestamps = attempt_movement_values[:, 0].astype(np.float64, copy=False)
        stage_positions = self.get_attempt_stage_positions(attempt_movement_values)
        pipette_positions = self.get_attempt_pipette_positions(attempt_movement_values)
        stage_actions = _positions_to_deltas(stage_positions)
        pip_actions = _positions_to_deltas(pipette_positions)
        if getattr(self, "use_velocities", False):
            stage_actions = _deltas_to_observation_velocities(stage_actions, timestamps)
            pip_actions = _deltas_to_observation_velocities(pip_actions, timestamps)

        stage_indices = selector.stage_indices(stage_actions.shape[1])
        if stage_indices:
            selected_components.append(stage_actions[:, stage_indices])
            action_labels.extend(
                f"stage_{AxisToggle.AXIS_NAMES[idx]}" for idx in stage_indices
            )

        pip_indices = selector.pipette_indices(pip_actions.shape[1])
        if pip_indices:
            selected_components.append(pip_actions[:, pip_indices])
            action_labels.extend(
                f"pipette_{AxisToggle.AXIS_NAMES[idx]}" for idx in pip_indices
            )

        return np.hstack(selected_components), action_labels

    def _trim_gigaseal_attempt_to_first_action_event_window(
        self,
        attempt_graph_values: np.ndarray,
        attempt_movement_values: np.ndarray,
        pressure_events: pd.DataFrame,
        recording_graph_values: Optional[np.ndarray] = None,
        recording_movement_values: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, bool, Optional[str]]:
        """Keep gigaseal rows starting 15 samples before the first action event."""

        num_rows = attempt_graph_values.shape[0]
        if num_rows == 0:
            return attempt_graph_values, attempt_movement_values, False, None

        actions, action_labels = self._get_gigaseal_event_window_actions(
            attempt_graph_values,
            attempt_movement_values,
            pressure_events,
        )
        action_events = self.get_action_event_mask(actions, action_labels, tolerance=0.0)
        if actions.shape[0] > 0 and actions.shape[1] > 0:
            first_row_active = np.any(
                np.isfinite(actions[0].astype(np.float64)) & (np.abs(actions[0]) > 0.0)
            )
            action_events[0] = action_events[0] or first_row_active

        event_indices = np.flatnonzero(action_events)
        if event_indices.size == 0:
            return (
                attempt_graph_values,
                attempt_movement_values,
                False,
                "no gigaseal action event was found in the attempt",
            )

        radius = max(0, int(getattr(self, "gigaseal_event_window_radius", 15)))
        first_event_idx = int(event_indices[0])
        if recording_graph_values is not None and recording_movement_values is not None:
            first_event_idx = self._nearest_graph_row_index(
                recording_graph_values,
                float(attempt_graph_values[first_event_idx, 0]),
            )
            attempt_end_idx = self._nearest_graph_row_index(
                recording_graph_values,
                float(attempt_graph_values[-1, 0]),
            )
            start_idx = max(0, first_event_idx - radius)
            graph_values = recording_graph_values[start_idx : attempt_end_idx + 1]
            movement_values = self.associate_attempt_movement_and_graph_values(
                graph_values,
                recording_movement_values,
            )
            trimmed = (
                graph_values.shape[0] != attempt_graph_values.shape[0]
                or graph_values[0, 0] != attempt_graph_values[0, 0]
                or graph_values[-1, 0] != attempt_graph_values[-1, 0]
            )
            return graph_values, movement_values, trimmed, None

        start_idx = max(0, first_event_idx - radius)
        trimmed = start_idx > 0
        return (
            attempt_graph_values[start_idx:],
            attempt_movement_values[start_idx:],
            trimmed,
            None,
        )

    def _trim_gigaseal_attempt_start_at_first_atm(
        self,
        attempt_graph_values: np.ndarray,
        attempt_movement_values: np.ndarray,
        pressure_events: pd.DataFrame,
        attempt_first_timestamp: float,
        attempt_last_timestamp: float,
    ) -> Tuple[np.ndarray, np.ndarray, bool, Optional[str]]:
        """Drop leading gigaseal rows before the initial near--5 mbar sample."""

        if attempt_graph_values.shape[0] == 0:
            return attempt_graph_values, attempt_movement_values, False, None

        # Recompute the aligned raw pressure trace after each trim so the returned
        # attempt itself starts on the first retained near--5 mbar sample.
        trimmed = False
        graph_values = attempt_graph_values
        movement_values = attempt_movement_values

        while graph_values.shape[0] > 0:
            commanded_pressure, _ = self.get_attempt_pressure_observation_values(
                graph_values,
                pressure_events,
            )
            pressure_ready_indices = np.flatnonzero(
                np.isfinite(np.asarray(commanded_pressure, dtype=np.float64))
                & (np.asarray(commanded_pressure, dtype=np.float64) <= -4.0)
            )
            if pressure_ready_indices.size == 0:
                return (
                    attempt_graph_values,
                    attempt_movement_values,
                    False,
                    "no aligned near--5 mbar pressure sample was found in the attempt",
                )

            start_idx = int(pressure_ready_indices[0])
            if start_idx == 0:
                return graph_values, movement_values, trimmed, None

            graph_values = graph_values[start_idx:]
            movement_values = movement_values[start_idx:]
            trimmed = True

        return (
            attempt_graph_values,
            attempt_movement_values,
            False,
            "no samples remain after trimming to the near--5 mbar start",
        )

    @staticmethod
    def associate_attempt_movement_and_graph_values(
        attempt_graph_values: np.ndarray, movement_values: np.ndarray
    ) -> np.ndarray:
        """Align each graph row with the latest movement sample known at that time."""
        g_ts = attempt_graph_values[:, 0]
        m_ts = movement_values[:, 0]
        indices = np.searchsorted(m_ts, g_ts, side="right") - 1
        indices = np.clip(indices, 0, len(m_ts) - 1)
        return movement_values[indices]

    # --- Attempt level feature extraction --------------------------------
    def get_attempt_dones(self, attempt_graph_values: np.ndarray) -> np.ndarray:
        """Generate a dones vector with a terminal 1 at the final timestep."""
        dones = np.zeros(len(attempt_graph_values))
        dones[-1] = 1
        return dones

    def get_attempt_pressure_values(self, attempt_graph_values: np.ndarray) -> np.ndarray:
        """Return the pressure column for the current attempt."""
        return attempt_graph_values[:, 1].astype(np.float64)

    def get_attempt_resistance_values(self, attempt_graph_values: np.ndarray) -> np.ndarray:
        """Return resistance values for the current attempt."""
        return attempt_graph_values[:, 2].astype(np.float64)

    def _compute_resistance_slope(self, resistance_values: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Return a rolling average resistance slope aligned to each observation."""
        if resistance_values is None:
            return None

        values = np.asarray(resistance_values, dtype=np.float64).reshape(-1)
        slopes = np.zeros(values.shape[0], dtype=np.float64)
        if values.shape[0] < 5:
            return slopes

        diffs = np.diff(values)
        valid = np.isfinite(diffs)
        max_window = min(50, max(1, int(self.settings.resistance_slope_window)))

        for end_idx in range(diffs.shape[0]):
            if end_idx < 3:
                continue
            window_size = min(max_window, end_idx + 1)
            start_idx = end_idx + 1 - window_size
            recent = diffs[start_idx:end_idx + 1]
            recent_valid = valid[start_idx:end_idx + 1]
            if np.any(recent_valid):
                slopes[end_idx + 1] = float(np.mean(recent[recent_valid]))
        return slopes

    def _compute_resistance_input_window(
        self,
        resistance_values: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """Return prior resistance samples aligned to each observation row."""
        if resistance_values is None:
            return None

        values = np.asarray(resistance_values, dtype=np.float64).reshape(-1)
        window_size = max(1, int(getattr(self.settings, "resistance_input_window", 15)))
        resistance_input = np.zeros((values.shape[0], window_size), dtype=np.float64)
        for row_idx in range(values.shape[0]):
            start_idx = max(0, row_idx - window_size)
            prior_values = values[start_idx:row_idx]
            if prior_values.size:
                resistance_input[row_idx, -prior_values.size:] = prior_values
        return resistance_input

    def get_attempt_current_values(self, attempt_graph_values: np.ndarray) -> np.ndarray:
        """Parse JSON-encoded current waveform samples for the attempt."""
        return _parse_waveform_column(attempt_graph_values[:, 3])

    def get_attempt_voltage_values(self, attempt_graph_values: np.ndarray) -> np.ndarray:
        """Parse JSON-encoded voltage waveform samples for the attempt."""
        return _parse_waveform_column(attempt_graph_values[:, 4])

    def get_attempt_stage_positions(self, attempt_movement_values: np.ndarray) -> np.ndarray:
        """Return stage XYZ positions."""
        return attempt_movement_values[:, 1:4].astype(np.float64)

    def get_attempt_pipette_positions(self, attempt_movement_values: np.ndarray) -> np.ndarray:
        """Return pipette XYZ positions."""
        return attempt_movement_values[:, 4:].astype(np.float64)

    def _apply_synthetic_position_anchor(
        self,
        stage_positions: np.ndarray,
        pipette_positions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Optionally express position observations relative to their first rows."""
        if not self.use_synthetic_position_anchor:
            return stage_positions, pipette_positions
        return (
            stage_positions - stage_positions[0],
            pipette_positions - pipette_positions[0],
        )

    # --- Camera helpers --------------------------------------------------
    @staticmethod
    def crop_image_center(pil_image: Image.Image) -> Image.Image:
        """Return a centred half-resolution crop of ``pil_image``."""
        width, height = pil_image.size
        new_width = width // 2
        new_height = height // 2
        left = (width - new_width) // 2
        top = (height - new_height) // 2
        right = left + new_width
        bottom = top + new_height
        return pil_image.crop((left, top, right, bottom))
    
    @staticmethod
    def add_image_color_dot(numpy_image: np.ndarray, color_dot: Optional[Tuple[int, int]]) -> np.ndarray:
        """Add a white dot to ``numpy_image`` at the given coordinates."""
        if color_dot is None or numpy_image.ndim < 2:
            return numpy_image

        height, width = numpy_image.shape[:2]
        x, y = map(int, color_dot)
        if not (0 <= x < width and 0 <= y < height):
            return numpy_image

        radius = 1
        x0 = max(x - radius, 0)
        x1 = min(x + radius + 1, width)
        y0 = max(y - radius, 0)
        y1 = min(y + radius + 1, height)

        if numpy_image.ndim >= 3:
            dot_value = np.array([255, 255, 255], dtype=numpy_image.dtype)
        else:
            dot_value = numpy_image.dtype.type(255)
        numpy_image[y0:y1, x0:x1] = dot_value
        return numpy_image

    @staticmethod
    def _camera_order_key(name: str) -> str:
        underscore_index = name.find('_')
        dot_index = name.rfind('.')
        if underscore_index == -1: return name
        segment = name[underscore_index + 1 : dot_index if dot_index != -1 else None]
        if segment and segment.replace('.', '', 1).isdigit(): segment = segment.zfill(3)
        suffix = name[dot_index:] if dot_index != -1 else ''
        return f"{name[:underscore_index + 1]}{segment}{suffix}"

    @staticmethod
    def _latest_causal_camera_frame_index(
        camera_timestamps: np.ndarray,
        target_timestamp: float,
        max_age_seconds: float = MAX_CAUSAL_CAMERA_FRAME_AGE_SECONDS,
    ) -> Optional[int]:
        """Return the newest non-future camera frame within the causal window."""
        if camera_timestamps.size == 0:
            return None
        index = int(np.searchsorted(camera_timestamps, target_timestamp, side="right")) - 1
        if index < 0:
            return None
        frame_age = float(target_timestamp) - float(camera_timestamps[index])
        if frame_age > max_age_seconds:
            return None
        return index

    def _get_camera_frame_shape(self, rig_recorder_data_folder: str) -> Optional[Tuple[int, int]]:
        cache = self._camera_frame_shape_cache
        if rig_recorder_data_folder in cache:
            return cache[rig_recorder_data_folder]
        camera_dir = self._resolve_rig_recorder_folder(rig_recorder_data_folder) / "camera_frames"
        if not camera_dir.exists():
            return None
        for frame_file in sorted(camera_dir.iterdir(), key=lambda p: self._camera_order_key(p.name)):
            if not frame_file.is_file():
                continue
            try:
                with Image.open(frame_file) as frame:
                    cache[rig_recorder_data_folder] = frame.size
                    return frame.size
            except (OSError, ValueError):
                continue
        return None

    def _camera_xy_transform(
        self, frame_shape: Tuple[int, int]
    ) -> Optional[Tuple[float, float, float, float]]:
        width, height = frame_shape
        if width <= 0 or height <= 0:
            return None
        crop_w = width // 2 if self.center_crop else width
        crop_h = height // 2 if self.center_crop else height
        if crop_w <= 0 or crop_h <= 0:
            return None
        offset_x = (width - crop_w) // 2 if self.center_crop else 0
        offset_y = (height - crop_h) // 2 if self.center_crop else 0
        scale_x = self.image_resize / crop_w
        scale_y = self.image_resize / crop_h
        return float(offset_x), float(offset_y), float(scale_x), float(scale_y)

    def _apply_camera_crop_and_resize(
        self, pipette_positions: np.ndarray, frame_shape: Tuple[int, int]
    ) -> np.ndarray:
        if pipette_positions.ndim < 2 or pipette_positions.shape[1] < 2:
            return pipette_positions
        transform = self._camera_xy_transform(frame_shape)
        if transform is None:
            return pipette_positions
        offset_x, offset_y, scale_x, scale_y = transform
        scaled = pipette_positions.astype(np.float64, copy=True)
        scaled[:, 0] = (scaled[:, 0] - offset_x) * scale_x
        scaled[:, 1] = (scaled[:, 1] - offset_y) * scale_y
        return scaled

    def _pipette_positions_in_image_space(
        self,
        pipette_positions: np.ndarray,
        rig_recorder_data_folder: Optional[str] = None,
    ) -> Tuple[np.ndarray, Tuple[float, float]]:
        """Return pipette positions converted into post-crop, post-resize image space."""
        converted = pipette_positions.astype(np.float64, copy=True)
        scale_xy = (1.0, 1.0)
        if not self._using_cv_movement_file:
            return converted, scale_xy

        frame_shape = None
        if rig_recorder_data_folder:
            frame_shape = self._get_camera_frame_shape(rig_recorder_data_folder)

        if frame_shape is None:
            scale_xy = getattr(self, "_last_pipette_scale", (1.0, 1.0))
            if converted.shape[1] >= 1:
                converted[:, 0] *= scale_xy[0]
            if converted.shape[1] >= 2:
                converted[:, 1] *= scale_xy[1]
            return converted, scale_xy

        transform = self._camera_xy_transform(frame_shape)
        if transform is None:
            return converted, scale_xy
        _, _, scale_x, scale_y = transform
        converted = self._apply_camera_crop_and_resize(converted, frame_shape)
        return converted, (scale_x, scale_y)

    def get_attempt_camera_frames(
        self,
        rig_recorder_data_folder: str,
        attempt_graph_values: np.ndarray,
        rotation_angle: Optional[float] = None,
        pipette_final_pos: Optional[tuple[int, int]] = None
    ) -> np.ndarray:
        """Load rig camera frames aligned to ``attempt_graph_values`` timestamps."""
        base = self._resolve_rig_recorder_folder(rig_recorder_data_folder) / "camera_frames"
        timestamped_camera_files: List[Tuple[float, str]] = []
        for camera_file in os.listdir(base):
            camera_timestamp = self._camera_file_timestamp(base / camera_file)
            if camera_timestamp is not None:
                timestamped_camera_files.append((camera_timestamp, camera_file))
        timestamped_camera_files.sort(key=lambda item: item[0])
        camera_timestamps = np.asarray(
            [item[0] for item in timestamped_camera_files],
            dtype=np.float64,
        )
        frames_list: List[np.ndarray] = []

        for graph_row in attempt_graph_values:
            target_timestamp = float(graph_row[0])
            min_idx = self._latest_causal_camera_frame_index(
                camera_timestamps,
                target_timestamp,
            )
            if min_idx is None:
                warnings.warn(
                    "No causal camera frame within "
                    f"{MAX_CAUSAL_CAMERA_FRAME_AGE_SECONDS * 1000:.0f} ms matched "
                    f"timestamp {target_timestamp} in {rig_recorder_data_folder}; "
                    "skipping attempt.",
                    RuntimeWarning,
                )
                return None

            pil_image = Image.open(base / timestamped_camera_files[min_idx][1])
            pil_image = self.apply_albu_filter_to_pil(pil_image)
            
            if rotation_angle is not None:
                pil_image = pil_image.rotate(rotation_angle, resample=Image.BILINEAR, expand=True)
            if self.center_crop:
                pil_image = self.crop_image_center(pil_image)
            
            resized_image = np.array(pil_image.resize((self.image_resize, self.image_resize)))

            if self.pipette_final_pos_color_dot:
                resized_image = self.add_image_color_dot(resized_image, pipette_final_pos)

            frames_list.append(resized_image)

        return np.array(frames_list)

    # --- Observations ----------------------------------------------------
    def get_attempt_observations(
        self,
        attempt_graph_values: np.ndarray,
        attempt_movement_values: np.ndarray,
        rig_recorder_data_folder: str,
        include_camera: bool = True,
        rotation_angle: Optional[float] = None,
    ):
        """Return selected observation arrays for the current attempt."""
        selector = self.observation_selector

        pressure_values: Optional[np.ndarray]
        if selector.include_pressure:
            pressure_values = self.get_attempt_pressure_values(attempt_graph_values)
        else:
            pressure_values = None

        resistance_values: Optional[np.ndarray]
        if selector.include_resistance:
            resistance_values = self.get_attempt_resistance_values(attempt_graph_values)
        else:
            resistance_values = None

        resistance_slope_values: Optional[np.ndarray]
        if selector.include_resistance and selector.include_resistance_slope and resistance_values is not None:
            resistance_slope_values = self._compute_resistance_slope(resistance_values)
        else:
            resistance_slope_values = None

        current_values: Optional[np.ndarray]
        if selector.include_current:
            current_values = self.get_attempt_current_values(attempt_graph_values)
        else:
            current_values = None

        voltage_values: Optional[np.ndarray]
        if selector.include_voltage:
            voltage_values = self.get_attempt_voltage_values(attempt_graph_values)
        else:
            voltage_values = None

        stage_positions_full = self.get_attempt_stage_positions(attempt_movement_values)
        pipette_positions_raw = self.get_attempt_pipette_positions(attempt_movement_values)
        pipette_positions_full, pipette_scale = self._pipette_positions_in_image_space(
            pipette_positions_raw,
            rig_recorder_data_folder=rig_recorder_data_folder,
        )
        self._last_pipette_scale = pipette_scale

        stage_positions_full, pipette_positions_full = self._apply_synthetic_position_anchor(
            stage_positions_full,
            pipette_positions_full,
        )

        stage_positions: Optional[np.ndarray]
        if selector.include_stage:
            stage_idx = selector.stage_indices(stage_positions_full.shape[1])
            if stage_idx:
                stage_positions = stage_positions_full[:, stage_idx]
            else:
                stage_positions = None
        else:
            stage_positions = None

        pipette_positions: Optional[np.ndarray]
        if selector.include_pipette:
            pipette_idx = selector.pipette_indices(pipette_positions_full.shape[1])
            if pipette_idx:
                pipette_positions = pipette_positions_full[:, pipette_idx]
            else:
                pipette_positions = None
        else:
            pipette_positions = None

        camera_required = include_camera and selector.include_camera
        camera_frames: Optional[np.ndarray] = None
        if camera_required:
            if self.pipette_final_pos_color_dot and pipette_positions_full.shape[1] >= 2:
                pipette_dot = (int(pipette_positions_full[-1][0]), int(pipette_positions_full[-1][1]))
            else:
                pipette_dot = None
            camera_frames = self.get_attempt_camera_frames(
                rig_recorder_data_folder,
                attempt_graph_values,
                rotation_angle=rotation_angle,
                pipette_final_pos=pipette_dot,
            )
            if camera_frames is None:
                return None

        return (
            pressure_values,
            resistance_values,
            resistance_slope_values,
            current_values,
            voltage_values,
            stage_positions,
            pipette_positions,
            camera_frames if camera_required else None,
        )

    def get_attempt_next_observations(
        self,
        pressure_values: Optional[np.ndarray],
        resistance_values: Optional[np.ndarray],
        resistance_slope_values: Optional[np.ndarray],
        current_values: Optional[np.ndarray],
        voltage_values: Optional[np.ndarray],
        stage_positions: Optional[np.ndarray],
        pipette_positions: Optional[np.ndarray],
        camera_frames: Optional[np.ndarray],
        include_next_obs: bool = False,
        include_camera: bool = True,
    ):
        """Compute next-step observation arrays using :func:`_shift_forward`."""
        if not include_next_obs:
            return (None,) * 8

        selector = self.observation_selector

        if selector.include_pressure and pressure_values is not None:
            next_pressure_values: Optional[np.ndarray] = _shift_forward(pressure_values)
        else:
            next_pressure_values = None

        if selector.include_resistance and resistance_values is not None:
            next_resistance_values: Optional[np.ndarray] = _shift_forward(resistance_values)
        else:
            next_resistance_values = None

        if selector.include_resistance_slope and resistance_slope_values is not None:
            next_resistance_slope_values: Optional[np.ndarray] = _shift_forward(resistance_slope_values)
        else:
            next_resistance_slope_values = None

        if selector.include_current and current_values is not None:
            next_current_values: Optional[np.ndarray] = _shift_forward(current_values)
        else:
            next_current_values = None

        if selector.include_voltage and voltage_values is not None:
            next_voltage_values: Optional[np.ndarray] = _shift_forward(voltage_values)
        else:
            next_voltage_values = None

        if selector.include_stage and stage_positions is not None:
            next_stage_positions: Optional[np.ndarray] = _shift_forward(stage_positions)
        else:
            next_stage_positions = None

        if selector.include_pipette and pipette_positions is not None:
            next_pipette_positions: Optional[np.ndarray] = _shift_forward(pipette_positions)
        else:
            next_pipette_positions = None

        camera_required = include_camera and selector.include_camera
        if camera_required and camera_frames is not None:
            next_camera_frames: Optional[np.ndarray] = _shift_forward(camera_frames)
        else:
            next_camera_frames = None

        return (
            next_pressure_values,
            next_resistance_values,
            next_resistance_slope_values,
            next_current_values,
            next_voltage_values,
            next_stage_positions,
            next_pipette_positions,
            next_camera_frames,
        )

    # --- Action computation ----------------------------------------------
    def get_attempt_actions(
        self,
        attempt_movement_values: np.ndarray,
        attempt_graph_values: Optional[np.ndarray] = None,
        rig_recorder_data_folder: Optional[str] = None,
        pressure_events: Optional[pd.DataFrame] = None,
        force_pressure_action: bool = False,
        force_break_in_action: bool = False,
    ) -> np.ndarray:
        """Return low-level per-step deltas or per-observation velocities."""
        timestamps = attempt_movement_values[:, 0].astype(np.float64, copy=False)
        stage_positions = self.get_attempt_stage_positions(attempt_movement_values)
        pipette_positions_raw = self.get_attempt_pipette_positions(attempt_movement_values)
        pipette_positions, pipette_scale = self._pipette_positions_in_image_space(
            pipette_positions_raw,
            rig_recorder_data_folder=rig_recorder_data_folder,
        )
        self._last_pipette_scale = pipette_scale

        stage_delta = _positions_to_deltas(stage_positions)
        pip_delta = _positions_to_deltas(pipette_positions)

        if self.use_velocities:
            stage_actions = _deltas_to_observation_velocities(stage_delta, timestamps)
            pip_actions = _deltas_to_observation_velocities(pip_delta, timestamps)
            self._last_action_representation = "velocity"
        else:
            stage_actions = stage_delta
            pip_actions = pip_delta
            self._last_action_representation = "delta"

        movement_actions = np.hstack([stage_actions, pip_actions])
        stage_dim = stage_actions.shape[1]

        selector = self.action_selector

        # Track whether the raw stage deltas contain any motion so omit_stage_movement
        # decisions do not depend on the current selector configuration.
        self._last_stage_motion_detected = bool(np.any(stage_delta != 0.0))

        if force_break_in_action:
            stage_indices = []
            pip_indices = []
        else:
            stage_indices = selector.stage_indices(stage_dim)
            pip_indices = selector.pipette_indices(pip_actions.shape[1])

        selected_components: List[np.ndarray] = []
        action_labels: List[str] = []

        if force_break_in_action:
            if attempt_graph_values is None:
                raise ValueError("attempt_graph_values is required for break-in action extraction")
            if pressure_events is None:
                raise ValueError("pressure_events are required for break-in action extraction")
            _, atm_state = self.get_attempt_pressure_action_values(
                attempt_graph_values,
                pressure_events,
            )
            break_in_components = [atm_state]
            action_labels.append("pressure_atm_state")
            if selector.include_break_in_zap:
                zap_command = self.get_attempt_zap_action_values(attempt_graph_values, pressure_events)
                break_in_components.append(zap_command)
                action_labels.append("zap_command")
            selected_components.append(
                np.column_stack(break_in_components).astype(np.float64, copy=False)
            )
        elif force_pressure_action:
            if attempt_graph_values is None:
                raise ValueError("attempt_graph_values is required when Pressure Action is enabled")
            if pressure_events is None:
                raise ValueError("pressure_events are required when Pressure Action is enabled")
            commanded_pressure, atm_state = self.get_attempt_pressure_action_values(
                attempt_graph_values,
                pressure_events,
            )
            if selector.binary_gigaseal_pressure_action:
                current_commanded_pressure, current_atm_state = self.get_attempt_pressure_observation_values(
                    attempt_graph_values,
                    pressure_events,
                )
                selected_components.append(
                    self.get_binary_gigaseal_pressure_action_values(
                        current_commanded_pressure,
                        current_atm_state,
                        commanded_pressure,
                        atm_state,
                    )
                )
            elif selector.combine_gigaseal_pressure_action:
                applied_pressure = self.get_applied_pressure_action_values(
                    commanded_pressure,
                    atm_state,
                )
                selected_components.append(
                    applied_pressure.reshape(-1, 1).astype(np.float64, copy=False)
                )
            else:
                selected_components.append(
                    np.column_stack([commanded_pressure, atm_state]).astype(np.float64, copy=False)
                )
            action_labels.extend(selector.gigaseal_pressure_axis_labels())

        if stage_indices:
            selected_components.append(movement_actions[:, stage_indices])
            axis_labels = [f"stage_{AxisToggle.AXIS_NAMES[idx]}" for idx in stage_indices]
            action_labels.extend(axis_labels)

        if pip_indices:
            pip_cols = [stage_dim + idx for idx in pip_indices]
            selected_components.append(movement_actions[:, pip_cols])
            axis_labels = [f"pipette_{AxisToggle.AXIS_NAMES[idx]}" for idx in pip_indices]
            action_labels.extend(axis_labels)

        if selector.include_pressure and not force_pressure_action and not force_break_in_action:
            if attempt_graph_values is None:
                raise ValueError("attempt_graph_values is required when Pressure Action is enabled")
            if pressure_events is None:
                raise ValueError("pressure_events are required when Pressure Action is enabled")
            pressure_action, _ = self.get_attempt_pressure_action_values(
                attempt_graph_values,
                pressure_events,
            )
            selected_components.append(
                pressure_action.reshape(-1, 1).astype(np.float64, copy=False)
            )
            action_labels.extend(selector.pressure_axis_labels())

        if selected_components:
            actions = np.hstack(selected_components)
        else:
            actions = np.zeros((movement_actions.shape[0], 0), dtype=movement_actions.dtype)

        self._last_action_labels = action_labels
        self._last_action_stage_cols = len(stage_indices)
        return actions

    # --- Gigaseal copied-demo augmentation -------------------------------
    @staticmethod
    def _copy_optional_array(array: Optional[np.ndarray]) -> Optional[np.ndarray]:
        return None if array is None else np.array(array, copy=True)

    def _build_gigaseal_augmented_payload(
        self,
        actions: np.ndarray,
        dones: np.ndarray,
        timestamp_values: Optional[np.ndarray],
        pressure_values: Optional[np.ndarray],
        pressure_state_values: Optional[np.ndarray],
        effective_pressure_values: Optional[np.ndarray],
        resistance_values: Optional[np.ndarray],
        resistance_slope_values: Optional[np.ndarray],
        resistance_input_values: Optional[np.ndarray],
        current_values: Optional[np.ndarray],
        voltage_values: Optional[np.ndarray],
        stage_positions: Optional[np.ndarray],
        pipette_positions: Optional[np.ndarray],
        camera_frames: Optional[np.ndarray],
        observations_since_last_action_values: Optional[np.ndarray],
        time_since_last_action_values: Optional[np.ndarray],
    ) -> _GigasealAugmentedPayload:
        """Deep-copy final demo arrays before applying Gigaseal augmentation."""

        return _GigasealAugmentedPayload(
            actions=np.array(actions, copy=True),
            dones=np.array(dones, copy=True),
            timestamp_values=self._copy_optional_array(timestamp_values),
            pressure_values=self._copy_optional_array(pressure_values),
            pressure_state_values=self._copy_optional_array(pressure_state_values),
            effective_pressure_values=self._copy_optional_array(effective_pressure_values),
            resistance_values=self._copy_optional_array(resistance_values),
            resistance_slope_values=self._copy_optional_array(resistance_slope_values),
            resistance_input_values=self._copy_optional_array(resistance_input_values),
            current_values=self._copy_optional_array(current_values),
            voltage_values=self._copy_optional_array(voltage_values),
            stage_positions=self._copy_optional_array(stage_positions),
            pipette_positions=self._copy_optional_array(pipette_positions),
            camera_frames=self._copy_optional_array(camera_frames),
            observations_since_last_action_values=self._copy_optional_array(
                observations_since_last_action_values
            ),
            time_since_last_action_values=self._copy_optional_array(time_since_last_action_values),
        )

    @staticmethod
    def _stutter_measured_sensor_values(
        values: np.ndarray,
        rng: np.random.Generator,
        probability: float,
        max_frames: int,
    ) -> np.ndarray:
        """Repeat the previous measured sensor row for short stale-read runs."""

        if values.shape[0] <= 1 or probability <= 0.0 or max_frames <= 0:
            return values

        result = np.array(values, copy=True)
        row = 1
        while row < result.shape[0]:
            if rng.random() >= probability:
                row += 1
                continue
            run_length = int(rng.integers(1, max_frames + 1))
            end = min(result.shape[0], row + run_length)
            result[row:end] = result[row - 1]
            row = end
        return result

    def _apply_gigaseal_sensor_augmentation(
        self,
        payload: _GigasealAugmentedPayload,
        rng: np.random.Generator,
        cfg: GigasealAugmentationSettings,
    ) -> None:
        """Perturb measured pressure/resistance only, leaving command-state fields intact."""

        stutter_probability = max(0.0, min(1.0, float(cfg.sensor_stutter_probability)))
        stutter_max = min(3, max(0, int(cfg.sensor_stutter_max_frames)))

        if payload.pressure_values is not None:
            pressure = np.asarray(payload.pressure_values, dtype=np.float64).copy()
            if cfg.pressure_offset_std > 0.0:
                pressure += float(rng.normal(0.0, cfg.pressure_offset_std))
            if cfg.pressure_noise_std > 0.0:
                pressure += rng.normal(0.0, cfg.pressure_noise_std, size=pressure.shape)
            pressure = self._stutter_measured_sensor_values(
                pressure,
                rng,
                stutter_probability,
                stutter_max,
            )
            low = min(float(cfg.pressure_min_mbar), float(cfg.pressure_max_mbar))
            high = max(float(cfg.pressure_min_mbar), float(cfg.pressure_max_mbar))
            payload.pressure_values = np.clip(pressure, low, high)

        if payload.resistance_values is not None:
            resistance = np.asarray(payload.resistance_values, dtype=np.float64).copy()
            floor = max(float(cfg.resistance_floor), np.finfo(np.float64).tiny)
            resistance = np.maximum(resistance, floor)
            if cfg.resistance_log_noise_std > 0.0:
                resistance *= np.exp(
                    rng.normal(0.0, cfg.resistance_log_noise_std, size=resistance.shape)
                )
            resistance = self._stutter_measured_sensor_values(
                resistance,
                rng,
                stutter_probability,
                stutter_max,
            )
            payload.resistance_values = np.maximum(resistance, floor)
            if payload.resistance_slope_values is not None:
                payload.resistance_slope_values = self._compute_resistance_slope(
                    payload.resistance_values
                )
            if payload.resistance_input_values is not None:
                payload.resistance_input_values = self._compute_resistance_input_window(
                    payload.resistance_values
                )

    @staticmethod
    def _index_optional_array(array: Optional[np.ndarray], indices: np.ndarray) -> Optional[np.ndarray]:
        return None if array is None else array[indices]

    def _apply_gigaseal_prefix_hold(
        self,
        payload: _GigasealAugmentedPayload,
        rng: np.random.Generator,
        cfg: GigasealAugmentationSettings,
        action_labels: Sequence[str],
    ) -> None:
        """Insert copied pre-transition rows while preserving causal command state."""

        num_rows = payload.actions.shape[0]
        if num_rows <= 1 or cfg.prefix_hold_probability <= 0.0 or cfg.prefix_hold_max_frames <= 0:
            return

        probability = max(0.0, min(1.0, float(cfg.prefix_hold_probability)))
        max_frames = max(1, int(cfg.prefix_hold_max_frames))
        tolerance = self.inaction_tolerance if self.inaction_tolerance > 0.0 else 0.0
        transition_rows = self.get_action_event_mask(
            payload.actions,
            action_labels,
            tolerance=tolerance,
        )

        indices: List[int] = []
        timestamps: List[float] = []
        original_times: Optional[np.ndarray]
        if payload.timestamp_values is None:
            original_times = None
        else:
            original_times = np.asarray(payload.timestamp_values, dtype=np.float64).reshape(-1)

        for row in range(num_rows):
            if row > 0 and transition_rows[row] and rng.random() < probability:
                hold_frames = int(rng.integers(1, max_frames + 1))
                previous_row = row - 1
                for hold_idx in range(hold_frames):
                    indices.append(previous_row)
                    if original_times is not None:
                        previous_time = float(original_times[previous_row])
                        current_time = float(original_times[row])
                        if np.isfinite(previous_time) and np.isfinite(current_time) and current_time > previous_time:
                            fraction = float(hold_idx + 1) / float(hold_frames + 1)
                            timestamps.append(previous_time + (current_time - previous_time) * fraction)
                        else:
                            timestamps.append(previous_time)
            indices.append(row)
            if original_times is not None:
                timestamps.append(float(original_times[row]))

        if len(indices) == num_rows:
            return

        index_array = np.asarray(indices, dtype=np.int64)
        payload.actions = payload.actions[index_array]
        payload.pressure_values = self._index_optional_array(payload.pressure_values, index_array)
        payload.pressure_state_values = self._index_optional_array(
            payload.pressure_state_values,
            index_array,
        )
        payload.effective_pressure_values = self._index_optional_array(
            payload.effective_pressure_values,
            index_array,
        )
        payload.resistance_values = self._index_optional_array(payload.resistance_values, index_array)
        payload.resistance_slope_values = self._index_optional_array(
            payload.resistance_slope_values,
            index_array,
        )
        payload.resistance_input_values = self._index_optional_array(
            payload.resistance_input_values,
            index_array,
        )
        payload.current_values = self._index_optional_array(payload.current_values, index_array)
        payload.voltage_values = self._index_optional_array(payload.voltage_values, index_array)
        payload.stage_positions = self._index_optional_array(payload.stage_positions, index_array)
        payload.pipette_positions = self._index_optional_array(payload.pipette_positions, index_array)
        payload.camera_frames = self._index_optional_array(payload.camera_frames, index_array)
        if original_times is not None:
            payload.timestamp_values = np.asarray(timestamps, dtype=np.float64)

        payload.dones = np.zeros(payload.actions.shape[0], dtype=payload.dones.dtype)
        if payload.dones.size:
            payload.dones[-1] = 1

    @staticmethod
    def _timing_action_index(action_labels: Sequence[str]) -> Optional[int]:
        try:
            return list(action_labels).index("observations_until_next_action")
        except ValueError:
            return None

    def _refresh_gigaseal_timing_features(
        self,
        payload: _GigasealAugmentedPayload,
        cfg: GigasealAugmentationSettings,
        action_labels: Sequence[str],
    ) -> None:
        """Recompute derived counters from the final augmented action sequence."""

        tolerance = self.inaction_tolerance if self.inaction_tolerance > 0.0 else 0.0
        timing_action_idx = self._timing_action_index(action_labels)
        if timing_action_idx is not None and timing_action_idx < payload.actions.shape[1]:
            counts_until = self.get_observations_until_next_action_values(
                payload.actions,
                action_labels,
                tolerance=tolerance,
            )
            cap = int(cfg.observations_until_next_action_cap)
            if cap > 0:
                counts_until = np.minimum(counts_until, float(cap))
            payload.actions[:, timing_action_idx] = counts_until

        if payload.observations_since_last_action_values is not None:
            counts = self.get_observations_since_last_action_values(
                payload.actions,
                action_labels,
                tolerance=tolerance,
            )
            cap = int(cfg.observations_until_next_action_cap)
            if cap > 0:
                counts = np.minimum(counts, float(cap))
            payload.observations_since_last_action_values = counts

        if payload.time_since_last_action_values is not None:
            if payload.timestamp_values is None:
                payload.time_since_last_action_values = np.zeros(
                    payload.actions.shape[0],
                    dtype=np.float64,
                )
            else:
                payload.time_since_last_action_values = self.get_time_since_last_action_values(
                    payload.actions,
                    action_labels,
                    payload.timestamp_values,
                    tolerance=tolerance,
                )

        if payload.resistance_input_values is not None:
            payload.resistance_input_values = self._compute_resistance_input_window(
                payload.resistance_values
            )

    def _make_gigaseal_augmented_payloads(
        self,
        source_seed: int,
        split_label: str,
        actions: np.ndarray,
        dones: np.ndarray,
        timestamp_values: Optional[np.ndarray],
        pressure_values: Optional[np.ndarray],
        pressure_state_values: Optional[np.ndarray],
        effective_pressure_values: Optional[np.ndarray],
        resistance_values: Optional[np.ndarray],
        resistance_slope_values: Optional[np.ndarray],
        resistance_input_values: Optional[np.ndarray],
        current_values: Optional[np.ndarray],
        voltage_values: Optional[np.ndarray],
        stage_positions: Optional[np.ndarray],
        pipette_positions: Optional[np.ndarray],
        camera_frames: Optional[np.ndarray],
        observations_since_last_action_values: Optional[np.ndarray],
        time_since_last_action_values: Optional[np.ndarray],
    ) -> List[Tuple[_GigasealAugmentedPayload, Dict[str, Any]]]:
        """Return configured Gigaseal copied-demo augmentations for one source demo."""

        cfg = getattr(self, "gigaseal_augmentation", GigasealAugmentationSettings())
        if not cfg.enabled:
            return []
        if split_label != "train" and not cfg.augment_validation:
            return []

        copies = max(0, int(cfg.copies_per_demo))
        if copies == 0:
            return []

        augmented: List[Tuple[_GigasealAugmentedPayload, Dict[str, Any]]] = []
        action_labels = list(self._last_action_labels)
        for copy_idx in range(copies):
            aug_seed = _stable_int_seed(source_seed, "gigaseal_augmentation", copy_idx)
            rng = np.random.default_rng(aug_seed)
            payload = self._build_gigaseal_augmented_payload(
                actions,
                dones,
                timestamp_values,
                pressure_values,
                pressure_state_values,
                effective_pressure_values,
                resistance_values,
                resistance_slope_values,
                resistance_input_values,
                current_values,
                voltage_values,
                stage_positions,
                pipette_positions,
                camera_frames,
                observations_since_last_action_values,
                time_since_last_action_values,
            )
            self._apply_gigaseal_sensor_augmentation(payload, rng, cfg)
            self._apply_gigaseal_prefix_hold(payload, rng, cfg, action_labels)
            self._refresh_gigaseal_timing_features(payload, cfg, action_labels)
            attrs = {
                "augmentation_kind": "gigaseal_conservative_copy",
                "augmentation_seed": int(aug_seed),
                "augmentation_copy_index": int(copy_idx),
            }
            augmented.append((payload, attrs))
        return augmented

    def _write_gigaseal_augmented_payload(
        self,
        payload: _GigasealAugmentedPayload,
        split_label: str,
        source_demo_key: str,
        attrs: Mapping[str, Any],
        include_next_obs: bool,
        include_camera: bool,
    ) -> str:
        """Write one augmented copy after regenerating derived next observations."""

        next_obs = self.get_attempt_next_observations(
            payload.pressure_values,
            payload.resistance_values,
            payload.resistance_slope_values,
            payload.current_values,
            payload.voltage_values,
            payload.stage_positions,
            payload.pipette_positions,
            payload.camera_frames,
            include_next_obs=include_next_obs,
            include_camera=include_camera,
        )
        (
            next_pressure_values,
            next_resistance_values,
            next_resistance_slope_values,
            next_current_values,
            next_voltage_values,
            next_stage_positions,
            next_pipette_positions,
            next_camera_frames,
        ) = next_obs

        next_pressure_state_values = (
            _shift_forward(payload.pressure_state_values)
            if include_next_obs and payload.pressure_state_values is not None
            else None
        )
        next_effective_pressure_values = (
            _shift_forward(payload.effective_pressure_values)
            if include_next_obs and payload.effective_pressure_values is not None
            else None
        )
        next_observations_since_last_action_values = (
            _shift_forward(payload.observations_since_last_action_values)
            if include_next_obs and payload.observations_since_last_action_values is not None
            else None
        )
        next_time_since_last_action_values = (
            _shift_forward(payload.time_since_last_action_values)
            if include_next_obs and payload.time_since_last_action_values is not None
            else None
        )
        next_resistance_input_values = (
            _shift_forward(payload.resistance_input_values)
            if include_next_obs and payload.resistance_input_values is not None
            else None
        )

        demo_attrs = dict(attrs)
        demo_attrs["augmentation_source_demo"] = source_demo_key
        return self.add_attempt_demo_to_dataset(
            num_samples=payload.actions.shape[0],
            actions=payload.actions,
            dones=payload.dones,
            pressure_values=payload.pressure_values,
            resistance_values=payload.resistance_values,
            resistance_slope_values=payload.resistance_slope_values,
            current_values=payload.current_values,
            voltage_values=payload.voltage_values,
            stage_positions=payload.stage_positions,
            pipette_positions=payload.pipette_positions,
            camera_frames=payload.camera_frames,
            next_pressure_values=next_pressure_values,
            next_resistance_values=next_resistance_values,
            next_resistance_slope_values=next_resistance_slope_values,
            next_current_values=next_current_values,
            next_voltage_values=next_voltage_values,
            next_stage_positions=next_stage_positions,
            next_pipette_positions=next_pipette_positions,
            next_camera_frames=next_camera_frames,
            include_next_obs=include_next_obs,
            include_camera=include_camera,
            split_label=split_label,
            pressure_state_values=payload.pressure_state_values,
            effective_pressure_values=payload.effective_pressure_values,
            resistance_input_values=payload.resistance_input_values,
            observations_since_last_action_values=payload.observations_since_last_action_values,
            time_since_last_action_values=payload.time_since_last_action_values,
            next_pressure_state_values=next_pressure_state_values,
            next_effective_pressure_values=next_effective_pressure_values,
            next_resistance_input_values=next_resistance_input_values,
            next_observations_since_last_action_values=next_observations_since_last_action_values,
            next_time_since_last_action_values=next_time_since_last_action_values,
            demo_attrs=demo_attrs,
        )

    # --- Dataset writing -------------------------------------------------
    def add_attempt_demo_to_dataset(
        self,
        num_samples: int,
        actions: np.ndarray,
        dones: np.ndarray,
        pressure_values: Optional[np.ndarray],
        resistance_values: Optional[np.ndarray],
        resistance_slope_values: Optional[np.ndarray],
        current_values: Optional[np.ndarray],
        voltage_values: Optional[np.ndarray],
        stage_positions: Optional[np.ndarray],
        pipette_positions: Optional[np.ndarray],
        camera_frames: Optional[np.ndarray],
        next_pressure_values: Optional[np.ndarray],
        next_resistance_values: Optional[np.ndarray],
        next_resistance_slope_values: Optional[np.ndarray],
        next_current_values: Optional[np.ndarray],
        next_voltage_values: Optional[np.ndarray],
        next_stage_positions: Optional[np.ndarray],
        next_pipette_positions: Optional[np.ndarray],
        next_camera_frames: Optional[np.ndarray],
        include_next_obs: bool = False,
        include_camera: bool = True,
        split_label: str = "train",
        pressure_state_values: Optional[np.ndarray] = None,
        effective_pressure_values: Optional[np.ndarray] = None,
        resistance_input_values: Optional[np.ndarray] = None,
        observations_since_last_action_values: Optional[np.ndarray] = None,
        time_since_last_action_values: Optional[np.ndarray] = None,
        next_pressure_state_values: Optional[np.ndarray] = None,
        next_effective_pressure_values: Optional[np.ndarray] = None,
        next_resistance_input_values: Optional[np.ndarray] = None,
        next_observations_since_last_action_values: Optional[np.ndarray] = None,
        next_time_since_last_action_values: Optional[np.ndarray] = None,
        demo_attrs: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """Persist a demo to disk and return the HDF5 key used for the group."""
        selector = self.observation_selector
        effective_include_camera = include_camera and selector.include_camera

        num_samples = actions.shape[0]
        effective_include_camera = effective_include_camera and camera_frames is not None

        def _as_column(arr: np.ndarray) -> np.ndarray:
            return arr.reshape(-1, 1) if arr.ndim == 1 else arr

        with h5py.File(self.dataset_path, "a") as hf:
            if "data" not in hf:
                data_group = hf.create_group("data")
                data_group.attrs["num_demos"] = 0
            else:
                data_group = hf["data"]
            demo_number = data_group.attrs["num_demos"]
            demo_name = f"demo_{demo_number}"
            demo = data_group.create_group(demo_name)
            demo.attrs["num_samples"] = num_samples
            demo.attrs["split"] = split_label
            if demo_attrs:
                for attr_name, attr_value in demo_attrs.items():
                    if attr_value is not None:
                        demo.attrs[attr_name] = attr_value
            self._normalize_demo_outcome_attrs(demo)

            action_ds = demo.create_dataset("actions", data=actions)
            if self._last_action_labels:
                action_ds.attrs["axes"] = np.asarray(self._last_action_labels, dtype="S")
            action_ds.attrs["representation"] = self._last_action_representation
            demo.create_dataset("dones", data=dones)

            observations = demo.create_group("obs")
            if pressure_values is not None:
                observations.create_dataset("pressure", data=_as_column(pressure_values))
            if pressure_state_values is not None:
                observations.create_dataset("pressure_atm_state", data=_as_column(pressure_state_values))
            if effective_pressure_values is not None:
                observations.create_dataset("effective_pressure", data=_as_column(effective_pressure_values))
            if resistance_values is not None:
                observations.create_dataset("resistance", data=_as_column(resistance_values))
            if resistance_slope_values is not None:
                observations.create_dataset("resistance_slope", data=_as_column(resistance_slope_values))
            if resistance_input_values is not None:
                observations.create_dataset("resistance_input", data=_as_column(resistance_input_values))
            if observations_since_last_action_values is not None:
                observations.create_dataset(
                    "observations_since_last_action",
                    data=_as_column(observations_since_last_action_values),
                )
            if time_since_last_action_values is not None:
                observations.create_dataset(
                    "time_since_last_action",
                    data=_as_column(time_since_last_action_values),
                )
            if current_values is not None:
                observations.create_dataset("current", data=current_values)
            if voltage_values is not None:
                observations.create_dataset("voltage", data=voltage_values)
            if stage_positions is not None:
                stage_ds = observations.create_dataset("stage_positions", data=stage_positions)
                stage_axes = self.observation_selector.stage_axis_labels()
                if stage_axes:
                    stage_ds.attrs["axes"] = np.asarray(stage_axes, dtype="S")
            if pipette_positions is not None:
                pip_ds = observations.create_dataset("pipette_positions", data=pipette_positions)
                pip_axes = self.observation_selector.pipette_axis_labels()
                if pip_axes:
                    pip_ds.attrs["axes"] = np.asarray(pip_axes, dtype="S")
            if effective_include_camera and camera_frames is not None:
                observations.create_dataset("camera_image", data=camera_frames)

            if include_next_obs:
                next_obs = demo.create_group("next_obs")
                if next_pressure_values is not None:
                    next_obs.create_dataset("pressure", data=_as_column(next_pressure_values))
                if next_pressure_state_values is not None:
                    next_obs.create_dataset("pressure_atm_state", data=_as_column(next_pressure_state_values))
                if next_effective_pressure_values is not None:
                    next_obs.create_dataset("effective_pressure", data=_as_column(next_effective_pressure_values))
                if next_resistance_values is not None:
                    next_obs.create_dataset("resistance", data=_as_column(next_resistance_values))
                if next_resistance_slope_values is not None:
                    next_obs.create_dataset("resistance_slope", data=_as_column(next_resistance_slope_values))
                if next_resistance_input_values is not None:
                    next_obs.create_dataset("resistance_input", data=_as_column(next_resistance_input_values))
                if next_observations_since_last_action_values is not None:
                    next_obs.create_dataset(
                        "observations_since_last_action",
                        data=_as_column(next_observations_since_last_action_values),
                    )
                if next_time_since_last_action_values is not None:
                    next_obs.create_dataset(
                        "time_since_last_action",
                        data=_as_column(next_time_since_last_action_values),
                    )
                if next_current_values is not None:
                    next_obs.create_dataset("current", data=next_current_values)
                if next_voltage_values is not None:
                    next_obs.create_dataset("voltage", data=next_voltage_values)
                if next_stage_positions is not None:
                    next_stage_ds = next_obs.create_dataset("stage_positions", data=next_stage_positions)
                    stage_axes = self.observation_selector.stage_axis_labels()
                    if stage_axes:
                        next_stage_ds.attrs["axes"] = np.asarray(stage_axes, dtype="S")
                if next_pipette_positions is not None:
                    next_pip_ds = next_obs.create_dataset("pipette_positions", data=next_pipette_positions)
                    pip_axes = self.observation_selector.pipette_axis_labels()
                    if pip_axes:
                        next_pip_ds.attrs["axes"] = np.asarray(pip_axes, dtype="S")
                if effective_include_camera and next_camera_frames is not None:
                    next_obs.create_dataset("camera_image", data=next_camera_frames)

            data_group.attrs["num_demos"] = demo_number + 1
            print(
                f"Added {split_label} {demo_name} to dataset '{self.dataset_name}' with {num_samples} samples."
            )
        self._write_metadata_files()
        return demo_name

    def _load_existing_processed_folders(self, metadata_path: Path) -> List[str]:
        """Return stored processed folder names from an existing metadata file."""
        try:
            with open(metadata_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError, TypeError):
            return []
        folders = data.get("processed_folders")
        if not isinstance(folders, list):
            return []
        return [str(item) for item in folders if isinstance(item, str)]

    def _register_processed_folder(self, folder: str) -> bool:
        """Record a processed folder for the current context and base dataset."""
        added_to_base = False
        if folder not in self._processed_folders:
            self._processed_folders.append(folder)
            added_to_base = True
        if self._active_state_context is not None:
            state_list = self._active_state_context.processed_folders
            if folder not in state_list:
                state_list.append(folder)
        return added_to_base

    def _selector_overview(self) -> Dict[str, Any]:
        """Return a JSON-friendly summary of active observation/action selectors."""

        obs = self.observation_selector
        act = self.action_selector
        return {
            "observations": {
                "include_pressure": obs.include_pressure,
                "include_resistance": obs.include_resistance,
                "include_resistance_slope": obs.include_resistance_slope,
                "include_current": obs.include_current,
                "include_voltage": obs.include_voltage,
                "include_stage": obs.include_stage,
                "include_pipette": obs.include_pipette,
                "include_camera": obs.include_camera,
                "include_gigaseal_pressure_state": obs.include_gigaseal_pressure_state,
                "include_gigaseal_effective_pressure": obs.include_gigaseal_effective_pressure,
                "include_gigaseal_resistance_input": obs.include_gigaseal_resistance_input,
                "include_gigaseal_observations_since_last_action": (
                    obs.include_gigaseal_observations_since_last_action
                ),
                "include_gigaseal_time_since_last_action": obs.include_gigaseal_time_since_last_action,
                "stage_axes": obs.stage_axis_labels(),
                "pipette_axes": obs.pipette_axis_labels(),
            },
            "actions": {
                "representation": "velocity" if self.use_velocities else "delta",
                "include_stage": act.include_stage,
                "include_pipette": act.include_pipette,
                "include_pressure": act.include_pressure,
                "combine_gigaseal_pressure_action": act.combine_gigaseal_pressure_action,
                "binary_gigaseal_pressure_action": act.binary_gigaseal_pressure_action,
                "include_break_in_zap": act.include_break_in_zap,
                "include_high_level": act.include_high_level,
                "include_observations_until_next_action": (
                    act.include_observations_until_next_action
                ),
                "stage_axes": act.stage_axis_labels(),
                "pipette_axes": act.pipette_axis_labels(),
            },
        }

    # --- Dataset bookkeeping --------------------------------------------
    def _collect_metadata(self) -> dict:
        """Aggregate dataset metadata for JSON/CSV export."""

        settings_dict = asdict(self.settings)
        toggles = {
            "center_crop": self.center_crop,
            "image_resize": self.image_resize,
            "pipette_final_pos_color_dot": self.pipette_final_pos_color_dot,
            "inaction": self.inaction,
        }

        split_keys = {key: list(values) for key, values in self._split_keys.items()}
        split_counts = {key: 0 for key in split_keys}
        num_demos = 0

        if self.dataset_path.exists():
            with h5py.File(self.dataset_path, "r") as hf:
                if "data" in hf:
                    data_group = hf["data"]
                    num_demos = int(data_group.attrs.get("num_demos", 0))
                    for demo_name in data_group.keys():
                        split = data_group[demo_name].attrs.get("split")
                        if isinstance(split, bytes):
                            split = split.decode()
                        if split is None:
                            continue
                        split_counts[split] = split_counts.get(split, 0) + 1

        if self._active_state_context is not None:
            processed_folders = list(self._active_state_context.processed_folders)
        else:
            processed_folders = list(self._processed_folders)

        metadata = {
            "dataset_name": self.dataset_name,
            "dataset_directory": str(self.dataset_dir),
            "dataset_path": str(self.dataset_path),
            "num_demos": num_demos,
            "split_counts": split_counts,
            "split_keys": split_keys,
            "settings": settings_dict,
            "toggles": toggles,
            "processed_folders": processed_folders,
            "selectors": self._selector_overview(),
        }

        json_path = self.dataset_dir / self._metadata_filename
        created_at = None
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as fh:
                    existing = json.load(fh)
                created_at = existing.get("created_at")
            except (OSError, json.JSONDecodeError, TypeError):
                created_at = None

        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        metadata["created_at"] = created_at or now_iso
        metadata["updated_at"] = now_iso
        return metadata

    def _write_metadata_files(self) -> None:
        """Persist dataset metadata to JSON beside the HDF5 file."""

        metadata = self._collect_metadata()
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        json_path = self.dataset_dir / self._metadata_filename
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, sort_keys=True)

    @staticmethod
    def _demo_sort_key(demo_key: str) -> Tuple[int, Any]:
        """Sort demo_N keys numerically while tolerating unexpected names."""

        prefix, _, suffix = demo_key.partition("_")
        if prefix == "demo":
            try:
                return (0, int(suffix))
            except ValueError:
                pass
        return (1, demo_key)

    @staticmethod
    def _normalize_demo_outcome_attrs(demo: h5py.Group) -> int:
        """Ensure every demo has explicit success/failure attrs."""

        raw_code = demo.attrs.get("state_outcome_code", _STATE_OUTCOME_SUCCESS)
        try:
            outcome_code = int(raw_code)
        except (TypeError, ValueError):
            outcome_code = _STATE_OUTCOME_SUCCESS

        outcome_label = _STATE_OUTCOME_LABELS.get(
            outcome_code,
            f"outcome_{outcome_code}",
        )
        demo.attrs["state_outcome_code"] = outcome_code
        demo.attrs["state_outcome_label"] = outcome_label
        demo.attrs["state_is_success"] = int(outcome_code == _STATE_OUTCOME_SUCCESS)
        demo.attrs["state_is_failure"] = int(outcome_code == _STATE_OUTCOME_FAILURE)
        return outcome_code

    def _write_mask_group(
        self,
        hf: h5py.File,
        split_keys: Mapping[str, Sequence[str]],
    ) -> Dict[str, List[str]]:
        """Create split and outcome filter masks in robomimic format."""

        if "mask" in hf:
            del hf["mask"]

        data_group = hf["data"]
        demo_keys = sorted(
            [
                demo_key
                for demo_key, demo_obj in data_group.items()
                if isinstance(demo_obj, h5py.Group)
            ],
            key=self._demo_sort_key,
        )

        success_keys: List[str] = []
        failure_keys: List[str] = []
        for demo_key in demo_keys:
            outcome_code = self._normalize_demo_outcome_attrs(data_group[demo_key])
            if outcome_code == _STATE_OUTCOME_SUCCESS:
                success_keys.append(demo_key)
            elif outcome_code == _STATE_OUTCOME_FAILURE:
                failure_keys.append(demo_key)

        success_set = set(success_keys)
        failure_set = set(failure_keys)
        masks: Dict[str, List[str]] = {}
        for split_name in ("train", "valid"):
            if split_name not in split_keys:
                continue
            split_demo_keys = list(split_keys[split_name])
            masks[split_name] = split_demo_keys
            masks[f"{split_name}_success"] = [
                demo_key for demo_key in split_demo_keys if demo_key in success_set
            ]
            masks[f"{split_name}_failure"] = [
                demo_key for demo_key in split_demo_keys if demo_key in failure_set
            ]

        masks["success"] = success_keys
        masks["failure"] = failure_keys

        mask_grp = hf.create_group("mask")
        for mask_name, keys in masks.items():
            mask_grp.create_dataset(mask_name, data=np.asarray(keys, dtype="S"))
        return masks

    def write_debug_single_trajectory_dataset(
        self,
        random_seed: Optional[int] = None,
    ) -> Dict[str, str]:
        """Rewrite each HDF5 output to one compact demo referenced by train and valid."""

        if random_seed is None:
            settings = getattr(self, "settings", None)
            random_seed = int(getattr(settings, "random_seed", 0))
        else:
            random_seed = int(random_seed)

        selected_by_dataset: Dict[str, str] = {}

        def _rewrite_current_dataset() -> Optional[str]:
            if not self.dataset_path.exists():
                return None

            tmp_path = self.dataset_path.with_name(
                f"{self.dataset_path.stem}.debug_tmp{self.dataset_path.suffix}"
            )
            if tmp_path.exists():
                tmp_path.unlink()

            try:
                with h5py.File(self.dataset_path, "r") as src_hf:
                    if "data" not in src_hf:
                        return None

                    source_data = src_hf["data"]
                    demo_names = sorted(source_data.keys())
                    if not demo_names:
                        return None

                    rng = np.random.default_rng(
                        _stable_int_seed(
                            random_seed,
                            self.dataset_name,
                            "debug_single_trajectory",
                        )
                    )
                    selected_demo = demo_names[int(rng.integers(0, len(demo_names)))]

                    with h5py.File(tmp_path, "w") as dst_hf:
                        for attr_key, attr_value in src_hf.attrs.items():
                            dst_hf.attrs[attr_key] = attr_value

                        for top_level_name in src_hf.keys():
                            if top_level_name in {"data", "mask"}:
                                continue
                            src_hf.copy(src_hf[top_level_name], dst_hf, name=top_level_name)

                        dst_data = dst_hf.create_group("data")
                        for attr_key, attr_value in source_data.attrs.items():
                            dst_data.attrs[attr_key] = attr_value
                        dst_data.attrs["num_demos"] = 1

                        src_hf.copy(source_data[selected_demo], dst_data, name="demo_0")
                        demo = dst_data["demo_0"]
                        demo.attrs["split"] = "train"
                        self._normalize_demo_outcome_attrs(demo)
                        self._write_mask_group(
                            dst_hf,
                            {"train": ["demo_0"], "valid": ["demo_0"]},
                        )

                os.replace(tmp_path, self.dataset_path)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()

            self._split_keys = {"train": ["demo_0"], "valid": ["demo_0"]}
            self._write_metadata_files()
            return selected_demo

        selected = _rewrite_current_dataset()
        if selected is not None:
            selected_by_dataset[self.dataset_name] = selected

        for state in sorted(self._state_contexts):
            with self._use_state_context(state):
                selected = _rewrite_current_dataset()
                if selected is not None:
                    selected_by_dataset[self.dataset_name] = selected

        if not selected_by_dataset:
            raise RuntimeError("No written demos found; cannot create debug single-trajectory dataset.")

        return selected_by_dataset

    def write_split_masks(self) -> None:
        """Write train/valid demo names into each dataset's ``mask`` group."""
        if getattr(getattr(self, "settings", None), "debug_single_trajectory", False):
            selected_by_dataset = self.write_debug_single_trajectory_dataset()
            for dataset_name, selected_demo in selected_by_dataset.items():
                print(
                    "debug single-trajectory dataset: "
                    f"{dataset_name} uses {selected_demo} for both train and valid"
            )
            return

        def _write_current_masks() -> None:
            if not self.dataset_path.exists():
                return
            with h5py.File(self.dataset_path, "a") as hf:
                if "data" not in hf:
                    return
                masks = self._write_mask_group(hf, self._split_keys)
            valid_count_local = len(self._split_keys.get("valid", []))
            print(
                f"wrote split masks: {len(self._split_keys['train'])} train | "
                f"{valid_count_local} valid | {len(masks['success'])} success | "
                f"{len(masks['failure'])} failure"
            )
            self._write_metadata_files()

        _write_current_masks()
        for state in sorted(self._state_contexts):
            with self._use_state_context(state):
                _write_current_masks()

    # Maintain backward-compatible private name
    def _write_split_masks(self) -> None:
        """Compatibility shim delegating to :meth:`write_split_masks`."""
        self.write_split_masks()

    # Backward-compatible helper aliases
    def _stable_int_seed(self, *parts: object) -> int:
        """Compatibility shim delegating to module-level :func:`_stable_int_seed`."""
        return _stable_int_seed(*parts)

    def _begin_demo_filter_context(self, split_label: str, demo_seed: int) -> None:
        """Compatibility shim for :meth:`begin_filter_context`."""
        self.begin_filter_context(split_label, demo_seed)

    def _end_demo_filter_context(self) -> None:
        """Compatibility shim for :meth:`end_filter_context`."""
        self.end_filter_context()

    def _apply_albu_filter_to_pil(self, pil_image: Image.Image) -> Image.Image:
        """Compatibility shim for :meth:`apply_albu_filter_to_pil`."""
        return self.apply_albu_filter_to_pil(pil_image)

    # --- High level orchestration ---------------------------------------

    def add_demo(self, rig_recorder_data_folder: str, record_to_file: bool = False) -> None:
        """Parse a rig-recorder folder, extracting selected attempts into per-state datasets."""
        print(f"Adding demos from rig_recorder_data_folder: {rig_recorder_data_folder}")

        include_next_obs = self.load_next_obs
        include_camera = self.observation_selector.include_camera

        if self.action_selector.include_high_level:
            raise RuntimeError(
                "High-level action extraction requires log files, which are no longer processed."
            )

        graph_values, movement_values = self.load_experiment_data(rig_recorder_data_folder)

        graph_start = graph_values[0][0]
        graph_end = graph_values[-1][0]

        state_attempts = self._get_state_attempt_records_for_export(
            rig_recorder_data_folder, graph_start, graph_end
        )

        if not state_attempts:
            print("  no selected state attempts detected; skipping demo export")
            return

        if self.settings.output_mode == "images":
            self._export_sampled_full_images(rig_recorder_data_folder, state_attempts)
            return

        has_gigaseal_state = any(_is_gigaseal_state_name(state_name) for state_name in state_attempts)
        has_break_in_state = any(_is_break_in_state_name(state_name) for state_name in state_attempts)
        has_pressure_command_observation_state = any(
            _uses_pressure_command_observations(state_name) for state_name in state_attempts
        )
        obs_selector = self.observation_selector
        pressure_command_observations_requested = (
            obs_selector.include_gigaseal_pressure_state
            or obs_selector.include_gigaseal_effective_pressure
        )
        requires_pressure_events = (
            self.action_selector.include_pressure
            or has_gigaseal_state
            or has_break_in_state
            or (
                has_pressure_command_observation_state
                and pressure_command_observations_requested
            )
        )

        if requires_pressure_events:
            pressure_events: Optional[pd.DataFrame] = self._load_pressure_log_events(rig_recorder_data_folder)
        else:
            pressure_events = None

        base_metadata_needs_update = False

        for state_name, attempt_ranges in state_attempts.items():
            if not attempt_ranges:
                continue
            is_gigaseal_state = _is_gigaseal_state_name(state_name)
            is_break_in_state = _is_break_in_state_name(state_name)
            uses_pressure_command_observations = _uses_pressure_command_observations(state_name)
            print(f"  processing state '{state_name}' with {len(attempt_ranges)} attempts")
            with self._use_state_context(state_name):
                for attempt_record in attempt_ranges:
                    attempt_first_timestamp = attempt_record.started
                    attempt_last_timestamp = attempt_record.finished
                    outcome_label = attempt_record.outcome_label
                    attempt_graph_values = self.truncate_graph_values(
                        graph_values, attempt_first_timestamp, attempt_last_timestamp
                    )
                    attempt_movement_values = self.associate_attempt_movement_and_graph_values(
                        attempt_graph_values, movement_values
                    )
                    (
                        attempt_graph_values,
                        attempt_movement_values,
                        trimmed_for_gigaseal_start,
                        trimmed_for_gigaseal_cutoff,
                        trim_skip_reason,
                    ) = self.truncate_attempt_for_state(
                        attempt_graph_values,
                        attempt_movement_values,
                        state_name,
                        pressure_events=pressure_events,
                        attempt_first_timestamp=attempt_first_timestamp,
                        attempt_last_timestamp=attempt_last_timestamp,
                        recording_graph_values=graph_values,
                        recording_movement_values=movement_values,
                    )
                    if trim_skip_reason is not None:
                        warning_text = (
                            f"Skipping {outcome_label} '{state_name}' attempt in "
                            f"{rig_recorder_data_folder} "
                            f"({attempt_first_timestamp:.3f} -> {attempt_last_timestamp:.3f}): "
                            f"{trim_skip_reason}."
                        )
                        warnings.warn(warning_text, RuntimeWarning)
                        print(f"    skipped - {warning_text}")
                        continue
                    if attempt_graph_values.shape[0] == 0:
                        print("    skipped - no samples after state trimming")
                        continue
                    if trimmed_for_gigaseal_start:
                        print("    trimmed gigaseal attempt start")
                    if trimmed_for_gigaseal_cutoff:
                        print(
                            "    trimmed gigaseal attempt at resistance >= "
                            f"{self.gigaseal_resistance_cutoff:g}"
                        )
                    dones = self.get_attempt_dones(attempt_graph_values)

                    split_lbl = "valid" if self.rng.random() < self.val_ratio else "train"

                    demo_seed = _stable_int_seed(
                        self.dataset_name,
                        rig_recorder_data_folder,
                        attempt_first_timestamp,
                        attempt_last_timestamp,
                        state_name,
                    )
                    self.begin_filter_context(split_lbl, demo_seed)

                    observations = self.get_attempt_observations(
                        attempt_graph_values,
                        attempt_movement_values,
                        rig_recorder_data_folder,
                        include_camera=include_camera,
                        rotation_angle=None,
                    )
                    if observations is None:
                        print("    skipped - missing camera frames")
                        self.end_filter_context()
                        continue

                    (
                        pressure_values,
                        resistance_values,
                        resistance_slope_values,
                        current_values,
                        voltage_values,
                        stage_positions,
                        pipette_positions,
                        camera_frames,
                    ) = observations

                    timestamp_values: Optional[np.ndarray] = attempt_graph_values[:, 0].astype(
                        np.float64,
                        copy=True,
                    )
                    commanded_pressure_values: Optional[np.ndarray] = None
                    atm_state_values: Optional[np.ndarray] = None
                    pressure_state_values: Optional[np.ndarray] = None
                    effective_pressure_values: Optional[np.ndarray] = None
                    needs_pressure_command_observations = uses_pressure_command_observations and (
                        obs_selector.include_gigaseal_pressure_state
                        or obs_selector.include_gigaseal_effective_pressure
                    )
                    if needs_pressure_command_observations:
                        if pressure_events is None:
                            raise RuntimeError(
                                "Pressure command observations require parsed day-log pressure events."
                            )
                        commanded_pressure_values, atm_state_values = self.get_attempt_pressure_observation_values(
                            attempt_graph_values,
                            pressure_events,
                        )
                    if needs_pressure_command_observations:
                        if commanded_pressure_values is None or atm_state_values is None:
                            raise RuntimeError("Pressure command observations require command traces.")
                        if obs_selector.include_gigaseal_pressure_state:
                            pressure_state_values = atm_state_values
                        if obs_selector.include_gigaseal_effective_pressure:
                            effective_pressure_values = self.get_effective_pressure_observation_values(
                                commanded_pressure_values,
                                atm_state_values,
                            )

                    resistance_input_values: Optional[np.ndarray] = None
                    if (
                        uses_pressure_command_observations
                        and obs_selector.include_gigaseal_resistance_input
                    ):
                        resistance_input_source = (
                            resistance_values
                            if resistance_values is not None
                            else self.get_attempt_resistance_values(attempt_graph_values)
                        )
                        resistance_input_values = self._compute_resistance_input_window(
                            resistance_input_source
                        )

                    actions = self.get_attempt_actions(
                        attempt_movement_values,
                        attempt_graph_values=attempt_graph_values,
                        rig_recorder_data_folder=rig_recorder_data_folder,
                        pressure_events=pressure_events,
                        force_pressure_action=is_gigaseal_state,
                        force_break_in_action=is_break_in_state,
                    )

                    stage_moved = getattr(self, "_last_stage_motion_detected", False)
                    if self.omit_stage_movement and stage_moved:
                        print("    skipped - demo contains stage movement")
                        self.end_filter_context()
                        continue

                    force_keep_mask: Optional[np.ndarray] = None
                    if uses_pressure_command_observations and (
                        (is_gigaseal_state and getattr(self, "gigaseal_event_window_enabled", True))
                        or (is_break_in_state and getattr(self, "break_in_event_window_enabled", True))
                    ):
                        wait_radius = (
                            getattr(self, "gigaseal_event_window_radius", 15)
                            if is_gigaseal_state
                            else getattr(self, "break_in_event_window_radius", 15)
                        )
                        action_tolerance = self.inaction_tolerance if self.inaction_tolerance > 0.0 else 0.0
                        force_keep_mask = self.get_pre_action_event_keep_mask(
                            actions,
                            self._last_action_labels,
                            wait_radius,
                            tolerance=action_tolerance,
                        )

                    (
                        actions,
                        dones,
                        timestamp_values,
                        pressure_values,
                        pressure_state_values,
                        effective_pressure_values,
                        resistance_values,
                        resistance_slope_values,
                        resistance_input_values,
                        current_values,
                        voltage_values,
                        stage_positions,
                        pipette_positions,
                        camera_frames,
                        invalid_removed,
                        inactive_removed,
                    ) = self.filter_attempt_timesteps(
                        actions,
                        dones,
                        timestamp_values,
                        pressure_values,
                        pressure_state_values,
                        effective_pressure_values,
                        resistance_values,
                        resistance_slope_values,
                        resistance_input_values,
                        current_values,
                        voltage_values,
                        stage_positions,
                        pipette_positions,
                        camera_frames,
                        force_keep_mask=force_keep_mask,
                    )
                    if invalid_removed:
                        print(
                            "    removed "
                            f"{invalid_removed} timestep(s) with NaN/Inf values in selected payloads"
                        )
                    if inactive_removed:
                        print(f"    removed {inactive_removed} inactive timestep(s)")
                    if actions.shape[0] == 0:
                        print("    skipped - no samples remain after filtering")
                        self.end_filter_context()
                        continue

                    if (
                        uses_pressure_command_observations
                        and self.action_selector.include_observations_until_next_action
                    ):
                        action_tolerance = self.inaction_tolerance if self.inaction_tolerance > 0.0 else 0.0
                        observations_until_next_action_values = (
                            self.get_observations_until_next_action_values(
                                actions,
                                self._last_action_labels,
                                tolerance=action_tolerance,
                            )
                        )
                        actions = np.hstack(
                            [
                                actions,
                                observations_until_next_action_values.reshape(-1, 1),
                            ]
                        )
                        self._last_action_labels = (
                            list(self._last_action_labels)
                            + self.action_selector.timing_axis_labels()
                        )

                    observations_since_last_action_values: Optional[np.ndarray] = None
                    if (
                        uses_pressure_command_observations
                        and self.observation_selector.include_gigaseal_observations_since_last_action
                    ):
                        action_tolerance = self.inaction_tolerance if self.inaction_tolerance > 0.0 else 0.0
                        observations_since_last_action_values = (
                            self.get_observations_since_last_action_values(
                                actions,
                                self._last_action_labels,
                                tolerance=action_tolerance,
                            )
                        )

                    time_since_last_action_values: Optional[np.ndarray] = None
                    if (
                        uses_pressure_command_observations
                        and obs_selector.include_gigaseal_time_since_last_action
                    ):
                        action_tolerance = self.inaction_tolerance if self.inaction_tolerance > 0.0 else 0.0
                        time_since_last_action_values = self.get_time_since_last_action_values(
                            actions,
                            self._last_action_labels,
                            timestamp_values,
                            tolerance=action_tolerance,
                        )

                    next_obs = self.get_attempt_next_observations(
                        pressure_values,
                        resistance_values,
                        resistance_slope_values,
                        current_values,
                        voltage_values,
                        stage_positions,
                        pipette_positions,
                        camera_frames,
                        include_next_obs=include_next_obs,
                        include_camera=include_camera,
                    )
                    (
                        next_pressure_values,
                        next_resistance_values,
                        next_resistance_slope_values,
                        next_current_values,
                        next_voltage_values,
                        next_stage_positions,
                        next_pipette_positions,
                        next_camera_frames,
                    ) = next_obs

                    next_pressure_state_values = (
                        _shift_forward(pressure_state_values)
                        if include_next_obs and pressure_state_values is not None
                        else None
                    )
                    next_effective_pressure_values = (
                        _shift_forward(effective_pressure_values)
                        if include_next_obs and effective_pressure_values is not None
                        else None
                    )
                    next_observations_since_last_action_values = (
                        _shift_forward(observations_since_last_action_values)
                        if include_next_obs and observations_since_last_action_values is not None
                        else None
                    )
                    next_time_since_last_action_values = (
                        _shift_forward(time_since_last_action_values)
                        if include_next_obs and time_since_last_action_values is not None
                        else None
                    )
                    next_resistance_input_values = (
                        _shift_forward(resistance_input_values)
                        if include_next_obs and resistance_input_values is not None
                        else None
                    )

                    if record_to_file:
                        outcome_code = int(attempt_record.outcome_code)
                        demo_attrs = {
                            "state_outcome_code": outcome_code,
                            "state_outcome_label": outcome_label,
                            "state_is_success": int(outcome_code == _STATE_OUTCOME_SUCCESS),
                            "state_is_failure": int(outcome_code == _STATE_OUTCOME_FAILURE),
                            "state_system_mode_code": attempt_record.system_mode_code,
                            "state_attempt_json": attempt_record.attempt_json,
                        }
                        demo_key = self.add_attempt_demo_to_dataset(
                            num_samples=actions.shape[0],
                            actions=actions,
                            dones=dones,
                            pressure_values=pressure_values,
                            resistance_values=resistance_values,
                            resistance_slope_values=resistance_slope_values,
                            current_values=current_values,
                            voltage_values=voltage_values,
                            stage_positions=stage_positions,
                            pipette_positions=pipette_positions,
                            camera_frames=camera_frames,
                            next_pressure_values=next_pressure_values,
                            next_resistance_values=next_resistance_values,
                            next_resistance_slope_values=next_resistance_slope_values,
                            next_current_values=next_current_values,
                            next_voltage_values=next_voltage_values,
                            next_stage_positions=next_stage_positions,
                            next_pipette_positions=next_pipette_positions,
                            next_camera_frames=next_camera_frames,
                            include_next_obs=include_next_obs,
                            include_camera=include_camera,
                            split_label=split_lbl,
                            pressure_state_values=pressure_state_values,
                            effective_pressure_values=effective_pressure_values,
                            resistance_input_values=resistance_input_values,
                            observations_since_last_action_values=observations_since_last_action_values,
                            time_since_last_action_values=time_since_last_action_values,
                            next_pressure_state_values=next_pressure_state_values,
                            next_effective_pressure_values=next_effective_pressure_values,
                            next_resistance_input_values=next_resistance_input_values,
                            next_observations_since_last_action_values=(
                                next_observations_since_last_action_values
                            ),
                            next_time_since_last_action_values=next_time_since_last_action_values,
                            demo_attrs=demo_attrs,
                        )
                        self._split_keys[split_lbl].append(demo_key)
                        print(f"    added original {outcome_label} {split_lbl} demo")

                        if is_gigaseal_state:
                            augmented_payloads = self._make_gigaseal_augmented_payloads(
                                demo_seed,
                                split_lbl,
                                actions,
                                dones,
                                timestamp_values,
                                pressure_values,
                                pressure_state_values,
                                effective_pressure_values,
                                resistance_values,
                                resistance_slope_values,
                                resistance_input_values,
                                current_values,
                                voltage_values,
                                stage_positions,
                                pipette_positions,
                                camera_frames,
                                observations_since_last_action_values,
                                time_since_last_action_values,
                            )
                            for augmented_payload, augmented_attrs in augmented_payloads:
                                if demo_attrs:
                                    augmented_attrs = {**demo_attrs, **augmented_attrs}
                                augmented_key = self._write_gigaseal_augmented_payload(
                                    augmented_payload,
                                    split_lbl,
                                    demo_key,
                                    augmented_attrs,
                                    include_next_obs,
                                    include_camera,
                                )
                                self._split_keys[split_lbl].append(augmented_key)
                                print(f"    added augmented {split_lbl} demo from {demo_key}")

                        base_updated = self._register_processed_folder(rig_recorder_data_folder)
                        if base_updated:
                            base_metadata_needs_update = True

                        self.end_filter_context()
                        self._write_metadata_files()
                    else:
                        self.end_filter_context()

        if record_to_file and base_metadata_needs_update:
            self._write_metadata_files()


__all__ = [
    "SimpleDatasetBuilder",
    "DatasetBuilderSettings",
    "FilterSettings",
    "GigasealAugmentationSettings",
    "ObservationSelector",
    "ActionSelector",
    "AxisToggle",
]


if __name__ == "__main__":

# ----------------------------------------------------------------------------------------------------------------------------------------
    # dataset_name = "PatcherBot_test_dataset_v0_510.hdf5"
    dataset_name = "PatcherBot_classic_dataset_v0_001.hdf5"


    # rig_recorder_data_folder_set = [
    #     "2025_09_25-20_43",
    #     "2025_09_25-21_39",
    #     "2025_10_01-13_15",# ~ 20 more demos
    #     "2025_10_01-13_30",# ~ 30 more demos
    #     "2025_10_08-23_18" # version 0.200 and beyond. contains random planar endpoints.
    #     ] # version 0.001 training data (9/25/2025) # find pipette data
    # rig_recorder_data_folder_set = ["2025_09_25-22_13"] # version 0.001 test data (9/25/2025) find_pipette test set
    # rig_recorder_data_folder_set = ["2025_10_10-15_12"] # version 300 - version 510

    # rig_recorder_data_folder_set = ["2025_10_09-22_04"] # test_set

    rig_recorder_data_folder_set = ["2025_11_02-19_41",
                                    "2025_11_02-21_00"]

    # ------------------------------------------------------------------------------------------------------------------------------
    # rig_recorder_data_folder_set = [
    #     "2025_05_20-15_50",
    #     "2025_05_20-15_16",
    #     "2025_05_20-14_05",
    #     "2025_04_10-11_57",
    #     "2025_04_10-12_16",
    # ] # HEK training data (5/20/2025, 4/10/2025) ignore

    # rig_recorder_data_folder_set = ["2025_04_07-14_50"] # HEK testing data ignore

    # dataset_name =  "PatcherBot_Dino_dataset_v0_002.hdf5" ignore

    # rig_recorder_data_folder_set = [
    #     "2025_09_25-20_43",
    #     "2025_09_25-21_39",
    #     "2025_10_01-13_15",# ~ 20 more demos
    #     "2025_10_01-13_30", # ~ 30 more demos
    #     "2025_05_20-15_50",
    #     "2025_05_20-15_16",
    #     "2025_05_20-14_05",
    #     "2025_04_10-11_57",
    #     "2025_04_10-12_16",
    # ] ignore

    # # rig_recorder_data_folder_set =  ["2025_03_11-16_32"] # inference test data (3/11/2025), unseen for HEK training ignore

    builder = SimpleDatasetBuilder(
        dataset_name=dataset_name,
        val_ratio=0.1,
        omit_stage_movement=False,
        random_seed=0,
        load_next_obs=False,
    )

    for folder in rig_recorder_data_folder_set:
        print(f"Processing folder: {folder}")
        builder.add_demo(rig_recorder_data_folder=folder, record_to_file=True)

    builder.write_split_masks()











