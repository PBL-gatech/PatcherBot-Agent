"""Post-process an existing PatcherBot HDF5 into anchored hybrid coordinates."""

from __future__ import annotations

import datetime as _datetime
import json
import os
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np


ANCHOR_SIDECAR_VERSION = 1
COORDINATE_REPRESENTATION = "anchored_xy_pixels_z_relative_um"


class AnchorConversionError(RuntimeError):
    """Raised when an HDF5 file cannot be safely anchored."""


@dataclass(frozen=True)
class AnchorSelection:
    frame_index: int
    pixel_xy: Tuple[float, float]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AnchorSelection":
        if not isinstance(value, Mapping):
            raise AnchorConversionError("Each anchor selection must be a mapping.")
        pixel_xy = value.get("pixel_xy")
        if (
            isinstance(pixel_xy, (str, bytes))
            or not isinstance(pixel_xy, Sequence)
            or len(pixel_xy) != 2
        ):
            raise AnchorConversionError("Anchor pixel_xy must contain exactly two values.")
        try:
            selection = cls(
                frame_index=int(value.get("frame_index", -1)),
                pixel_xy=(float(pixel_xy[0]), float(pixel_xy[1])),
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise AnchorConversionError("Anchor frame index and pixel coordinates must be numeric.") from exc
        if (
            selection.frame_index < 0
            or not np.isfinite(selection.pixel_xy).all()
            or any(value < 0.0 for value in selection.pixel_xy)
        ):
            raise AnchorConversionError("Anchor frame index and pixel coordinates must be finite and non-negative.")
        return selection

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "frame_index": int(self.frame_index),
            "pixel_xy": [float(self.pixel_xy[0]), float(self.pixel_xy[1])],
        }


@dataclass(frozen=True)
class HDF5Inspection:
    path: Path
    demo_keys: Tuple[str, ...]
    split_counts: Mapping[str, int]
    observation_keys: Tuple[str, ...]
    camera_shape: Optional[Tuple[int, ...]]
    pipette_axes: Tuple[str, ...]
    stage_axes: Tuple[str, ...]
    action_axes: Tuple[str, ...]
    action_representation: str
    has_next_obs: bool
    anchored: bool
    errors: Tuple[str, ...]


@dataclass(frozen=True)
class CalibrationMatrices:
    stage: np.ndarray
    pipette: np.ndarray


def _decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _decode_axes(dataset: h5py.Dataset) -> Tuple[str, ...]:
    raw = dataset.attrs.get("axes")
    if raw is None:
        return ()
    if isinstance(raw, (bytes, str)):
        return (_decode_text(raw),)
    return tuple(_decode_text(item) for item in np.asarray(raw).reshape(-1))


def _demo_sort_key(name: str) -> Tuple[int, Any]:
    prefix, _, suffix = name.partition("_")
    if prefix == "demo" and suffix.isdigit():
        return (0, int(suffix))
    return (1, name)


def source_fingerprint(path: Path | str) -> Dict[str, int]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {"size": int(stat.st_size), "modified_ns": int(stat.st_mtime_ns)}


def derive_output_path(source: Path | str) -> Path:
    source_path = Path(source).resolve()
    if source_path.suffix.lower() not in {".hdf5", ".h5"}:
        raise AnchorConversionError("The source must be an .hdf5 or .h5 file.")
    if source_path.stem.lower().endswith("_anchored"):
        raise AnchorConversionError("The selected source is already named as an anchored dataset.")
    return source_path.with_name(f"{source_path.stem}_anchored{source_path.suffix}")


def derive_sidecar_path(source: Path | str) -> Path:
    source_path = Path(source).resolve()
    return source_path.with_name(f"{source_path.stem}_anchor_selections.json")


def derive_metadata_path(source: Path | str) -> Path:
    output = derive_output_path(source)
    return output.with_name(f"{output.stem}_metadata.json")


def save_anchor_sidecar(
    source: Path | str,
    anchors: Mapping[str, AnchorSelection],
    sidecar: Optional[Path | str] = None,
) -> Path:
    source_path = Path(source).resolve()
    sidecar_path = Path(sidecar).resolve() if sidecar else derive_sidecar_path(source_path)
    payload = {
        "version": ANCHOR_SIDECAR_VERSION,
        "source_hdf5": source_path.name,
        "source_fingerprint": source_fingerprint(source_path),
        "anchors": {key: value.to_mapping() for key, value in sorted(anchors.items(), key=lambda item: _demo_sort_key(item[0]))},
    }
    temp_path = sidecar_path.with_name(f".{sidecar_path.name}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temp_path, sidecar_path)
    return sidecar_path


def load_anchor_sidecar(
    source: Path | str,
    sidecar: Optional[Path | str] = None,
) -> Dict[str, AnchorSelection]:
    source_path = Path(source).resolve()
    sidecar_path = Path(sidecar).resolve() if sidecar else derive_sidecar_path(source_path)
    if not sidecar_path.exists():
        return {}
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnchorConversionError(f"Could not read anchor sidecar {sidecar_path}: {exc}") from exc
    if payload.get("version") != ANCHOR_SIDECAR_VERSION:
        raise AnchorConversionError(f"Unsupported anchor sidecar version in {sidecar_path}.")
    if payload.get("source_hdf5") != source_path.name:
        raise AnchorConversionError("Anchor sidecar belongs to a different HDF5 filename.")
    if payload.get("source_fingerprint") != source_fingerprint(source_path):
        raise AnchorConversionError("Anchor sidecar fingerprint does not match the selected HDF5.")
    raw_anchors = payload.get("anchors")
    if not isinstance(raw_anchors, Mapping):
        raise AnchorConversionError("Anchor sidecar is missing its anchors mapping.")
    return {str(key): AnchorSelection.from_mapping(value) for key, value in raw_anchors.items()}


def _companion_metadata_uses_cv(path: Path) -> bool:
    candidates = (
        path.parent / "metadata.json",
        path.with_name(f"{path.stem}_metadata.json"),
    )
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        settings = payload.get("settings", {})
        if isinstance(settings, Mapping) and bool(settings.get("prefer_cv_movement", False)):
            return True
    return False


def _inspect_position_dataset(
    demo_key: str,
    dataset_path: str,
    value: Any,
    expected_rows: Optional[int],
    errors: list[str],
) -> Tuple[str, ...]:
    if not isinstance(value, h5py.Dataset):
        errors.append(f"{demo_key} {dataset_path} must be a dataset.")
        return ()
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0:
        errors.append(f"{demo_key} {dataset_path} must be a non-empty 2D dataset, got {value.shape}.")
        return ()
    if expected_rows is not None and value.shape[0] != expected_rows:
        errors.append(
            f"{demo_key} {dataset_path} has {value.shape[0]} rows; expected {expected_rows}."
        )
    if not np.issubdtype(value.dtype, np.floating):
        errors.append(f"{demo_key} {dataset_path} must use a floating-point dtype.")
    axes = _decode_axes(value)
    if not axes:
        errors.append(f"{demo_key} {dataset_path} requires an axes attribute.")
        return ()
    if len(axes) != value.shape[1]:
        errors.append(
            f"{demo_key} {dataset_path} axes do not match dataset width {value.shape[1]}."
        )
        return axes
    prefix = "pipette" if dataset_path.endswith("pipette_positions") else "stage"
    try:
        _output_indices(axes, prefix, allow_plain=True)
    except AnchorConversionError as exc:
        errors.append(f"{demo_key} {dataset_path}: {exc}")
    return axes


def inspect_hdf5(source: Path | str) -> HDF5Inspection:
    path = Path(source).resolve()
    errors = []
    demo_keys: Tuple[str, ...] = ()
    split_counts: Dict[str, int] = {}
    obs_keys: set[str] = set()
    camera_shape: Optional[Tuple[int, ...]] = None
    pipette_axes: Tuple[str, ...] = ()
    stage_axes: Tuple[str, ...] = ()
    action_axes: Tuple[str, ...] = ()
    action_representation = ""
    has_next_obs = False
    anchored = False

    if not path.is_file():
        errors.append(f"HDF5 file not found: {path}")
    elif path.suffix.lower() not in {".hdf5", ".h5"}:
        errors.append("Source file must have an .hdf5 or .h5 extension.")
    elif path.stem.lower().endswith("_anchored"):
        errors.append("Source filename already ends in _anchored.")

    if not errors:
        try:
            with h5py.File(path, "r") as hf:
                anchored = bool(hf.attrs.get("anchored_coordinates", False))
                data = hf.get("data")
                if not isinstance(data, h5py.Group):
                    errors.append("Source HDF5 is missing the data group.")
                else:
                    anchored = anchored or bool(data.attrs.get("anchored_coordinates", False))
                    demo_keys = tuple(
                        sorted(
                            (
                                key
                                for key in data.keys()
                                if key.startswith("demo_") and isinstance(data[key], h5py.Group)
                            ),
                            key=_demo_sort_key,
                        )
                    )
                    if not demo_keys:
                        errors.append("Source HDF5 contains no demonstration groups.")
                    for demo_key in demo_keys:
                        demo = data[demo_key]
                        split = _decode_text(demo.attrs.get("split", "unknown"))
                        split_counts[split] = split_counts.get(split, 0) + 1
                        obs = demo.get("obs")
                        if not isinstance(obs, h5py.Group):
                            errors.append(f"{demo_key} is missing obs.")
                            continue
                        obs_keys.update(obs.keys())
                        required = ("camera_image", "pipette_positions", "stage_positions")
                        missing = [name for name in required if name not in obs]
                        if missing:
                            errors.append(f"{demo_key} is missing obs/{', obs/'.join(missing)}.")
                            continue
                        camera = obs["camera_image"]
                        pipette = obs["pipette_positions"]
                        stage = obs["stage_positions"]
                        sample_count: Optional[int] = None
                        if not isinstance(camera, h5py.Dataset):
                            errors.append(f"{demo_key} obs/camera_image must be a dataset.")
                        elif camera.ndim not in {3, 4} or camera.shape[0] == 0:
                            errors.append(f"{demo_key} has an invalid camera_image shape {camera.shape}.")
                        else:
                            sample_count = int(camera.shape[0])
                            try:
                                _camera_frame_size(camera)
                            except AnchorConversionError as exc:
                                errors.append(f"{demo_key} obs/camera_image: {exc}")
                            if camera_shape is None:
                                camera_shape = tuple(int(value) for value in camera.shape[1:])

                        current_pip_axes = _inspect_position_dataset(
                            demo_key,
                            "obs/pipette_positions",
                            pipette,
                            sample_count,
                            errors,
                        )
                        current_stage_axes = _inspect_position_dataset(
                            demo_key,
                            "obs/stage_positions",
                            stage,
                            sample_count,
                            errors,
                        )
                        pipette_axes = pipette_axes or current_pip_axes
                        stage_axes = stage_axes or current_stage_axes
                        if (
                            current_pip_axes
                            and pipette_axes
                            and current_pip_axes != pipette_axes
                        ) or (
                            current_stage_axes
                            and stage_axes
                            and current_stage_axes != stage_axes
                        ):
                            errors.append("Position axes must be consistent across demonstrations.")

                        actions = demo.get("actions")
                        if actions is not None and not isinstance(actions, h5py.Dataset):
                            errors.append(f"{demo_key} actions must be a dataset.")
                        elif isinstance(actions, h5py.Dataset):
                            current_action_axes = _decode_axes(actions)
                            if (
                                actions.ndim != 2
                                or actions.shape[0] == 0
                                or not current_action_axes
                                or len(current_action_axes) != actions.shape[1]
                            ):
                                errors.append(f"{demo_key} actions require matching axes metadata.")
                            elif sample_count is not None and actions.shape[0] != sample_count:
                                errors.append(
                                    f"{demo_key} actions have {actions.shape[0]} rows; expected {sample_count}."
                                )
                            try:
                                normalized_actions = _normalized_axes(current_action_axes)
                            except AnchorConversionError as exc:
                                errors.append(f"{demo_key} actions: {exc}")
                                normalized_actions = ()
                            if any(
                                axis.startswith(("stage_", "pipette_"))
                                for axis in normalized_actions
                            ) and not np.issubdtype(actions.dtype, np.floating):
                                errors.append(f"{demo_key} movement actions must use a floating-point dtype.")
                            action_axes = action_axes or current_action_axes
                            if current_action_axes and action_axes and current_action_axes != action_axes:
                                errors.append("Action axes must be consistent across demonstrations.")
                            raw_representation = actions.attrs.get("representation")
                            if raw_representation is None:
                                errors.append(
                                    f"{demo_key} actions require a delta or velocity representation attribute."
                                )
                            else:
                                representation = _decode_text(raw_representation).strip().lower()
                                if representation not in {"delta", "velocity"}:
                                    errors.append(
                                        f"{demo_key} actions representation must be delta or velocity, got {representation!r}."
                                    )
                                action_representation = action_representation or representation
                                if representation != action_representation:
                                    errors.append("Action representation must be consistent across demonstrations.")

                        next_obs = demo.get("next_obs")
                        if next_obs is not None:
                            has_next_obs = True
                            if not isinstance(next_obs, h5py.Group):
                                errors.append(f"{demo_key} next_obs must be a group.")
                            else:
                                next_names = ("pipette_positions", "stage_positions")
                                next_missing = [name for name in next_names if name not in next_obs]
                                if next_missing:
                                    errors.append(
                                        f"{demo_key} next_obs is missing next_obs/{', next_obs/'.join(next_missing)}."
                                    )
                                else:
                                    next_pip_axes = _inspect_position_dataset(
                                        demo_key,
                                        "next_obs/pipette_positions",
                                        next_obs["pipette_positions"],
                                        sample_count,
                                        errors,
                                    )
                                    next_stage_axes = _inspect_position_dataset(
                                        demo_key,
                                        "next_obs/stage_positions",
                                        next_obs["stage_positions"],
                                        sample_count,
                                        errors,
                                    )
                                    if current_pip_axes and next_pip_axes != current_pip_axes:
                                        errors.append(
                                            f"{demo_key} next_obs pipette axes must match obs axes."
                                        )
                                    if current_stage_axes and next_stage_axes != current_stage_axes:
                                        errors.append(
                                            f"{demo_key} next_obs stage axes must match obs axes."
                                        )
        except (OSError, ValueError, TypeError, IndexError, AnchorConversionError) as exc:
            errors.append(f"Could not inspect HDF5: {exc}")

    if anchored:
        errors.append("Source HDF5 is already marked as anchored.")
    if path.is_file() and _companion_metadata_uses_cv(path):
        errors.append("Companion metadata indicates CV-defined/pixel coordinates are already in use.")

    return HDF5Inspection(
        path=path,
        demo_keys=demo_keys,
        split_counts=split_counts,
        observation_keys=tuple(sorted(obs_keys)),
        camera_shape=camera_shape,
        pipette_axes=pipette_axes,
        stage_axes=stage_axes,
        action_axes=action_axes,
        action_representation=action_representation,
        has_next_obs=has_next_obs,
        anchored=anchored,
        errors=tuple(dict.fromkeys(errors)),
    )


def _matrix_from_entry(payload: Mapping[str, Any], key: str) -> np.ndarray:
    entry = payload.get(key)
    if not isinstance(entry, Mapping) or entry.get("M") is None:
        raise AnchorConversionError(f"Calibration payload is missing {key}.M.")
    matrix = np.asarray(entry["M"], dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] not in {2, 3}:
        raise AnchorConversionError(f"{key}.M must be a finite 2D matrix with at least two rows and 2 or 3 columns.")
    if not np.isfinite(matrix).all():
        raise AnchorConversionError(f"{key}.M contains NaN or infinite values.")
    return matrix


def load_calibration(path: Path | str) -> CalibrationMatrices:
    calibration_path = Path(path).resolve()
    if not calibration_path.is_file():
        raise AnchorConversionError(f"Calibration file not found: {calibration_path}")
    try:
        if calibration_path.suffix.lower() == ".json":
            payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        else:
            with calibration_path.open("rb") as handle:
                payload = pickle.load(handle)
    except Exception as exc:
        raise AnchorConversionError(f"Could not load calibration file {calibration_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise AnchorConversionError("Calibration payload must be a mapping.")
    return CalibrationMatrices(
        stage=_matrix_from_entry(payload, "stage"),
        pipette=_matrix_from_entry(payload, "manip"),
    )


def _normalized_axes(axes: Sequence[str]) -> Tuple[str, ...]:
    normalized = tuple(str(axis).strip().lower() for axis in axes)
    if any(not axis for axis in normalized) or len(set(normalized)) != len(normalized):
        raise AnchorConversionError(f"Axis labels must be non-empty and unique: {list(axes)}.")
    return normalized


def _axis_indices(
    axes: Sequence[str],
    prefix: str,
    columns: int,
    *,
    allow_plain: bool = False,
) -> Tuple[int, ...]:
    normalized = _normalized_axes(axes)
    suffixes = ("x", "y", "z")[:columns]
    candidates = [tuple(f"{prefix}_{suffix}" for suffix in suffixes)]
    if allow_plain:
        candidates.append(tuple(suffixes))
    matches = [
        tuple(normalized.index(axis) for axis in wanted)
        for wanted in candidates
        if all(axis in normalized for axis in wanted)
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise AnchorConversionError(
            f"{prefix} axes are ambiguous because both plain and prefixed labels are present: {list(axes)}."
        )
    raise AnchorConversionError(
        f"{prefix} matrix needs one of {candidates}, but the dataset provides {list(axes)}."
    )


def _output_indices(
    axes: Sequence[str],
    prefix: str,
    *,
    allow_plain: bool = False,
) -> Tuple[int, int, Optional[int]]:
    normalized = _normalized_axes(axes)
    candidates = [(f"{prefix}_x", f"{prefix}_y", f"{prefix}_z")]
    if allow_plain:
        candidates.append(("x", "y", "z"))
    matches = [
        candidate
        for candidate in candidates
        if candidate[0] in normalized and candidate[1] in normalized
    ]
    if len(matches) != 1:
        detail = (
            "both plain and prefixed X/Y labels are present"
            if len(matches) > 1
            else "labeled X and Y axes are missing"
        )
        raise AnchorConversionError(f"{prefix} positions are ambiguous: {detail}; got {list(axes)}.")
    x_name, y_name, z_name = matches[0]
    return (
        normalized.index(x_name),
        normalized.index(y_name),
        normalized.index(z_name) if z_name in normalized else None,
    )


def _coordinate_units(axes: Sequence[str]) -> np.ndarray:
    normalized = _normalized_axes(axes)
    return np.asarray(
        [
            "pixels" if axis in {"x", "y"} or axis.endswith(("_x", "_y")) else "um"
            for axis in normalized
        ],
        dtype="S",
    )


def transform_positions(
    values: np.ndarray,
    axes: Sequence[str],
    matrix: np.ndarray,
    reference: np.ndarray,
    *,
    prefix: str,
    anchor_xy: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    reference_arr = np.asarray(reference, dtype=np.float64).reshape(-1)
    if result.ndim != 2 or reference_arr.shape[0] != result.shape[1]:
        raise AnchorConversionError(f"Invalid {prefix} position/reference shape.")
    input_indices = _axis_indices(axes, prefix, matrix.shape[1], allow_plain=True)
    x_idx, y_idx, z_idx = _output_indices(axes, prefix, allow_plain=True)
    delta = result - reference_arr
    pixel_xy = delta[:, input_indices] @ matrix[:2, :].T
    if anchor_xy is not None:
        pixel_xy += np.asarray(anchor_xy, dtype=np.float64)
    result[:, x_idx] = pixel_xy[:, 0]
    result[:, y_idx] = pixel_xy[:, 1]
    if z_idx is not None:
        result[:, z_idx] = delta[:, z_idx]
    return result


def transform_actions(
    values: np.ndarray,
    axes: Sequence[str],
    stage_matrix: np.ndarray,
    pipette_matrix: np.ndarray,
) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    normalized = [axis.lower() for axis in axes]
    for prefix, matrix in (("stage", stage_matrix), ("pipette", pipette_matrix)):
        x_name, y_name = f"{prefix}_x", f"{prefix}_y"
        movement_present = any(axis.startswith(f"{prefix}_") for axis in normalized)
        if not movement_present:
            continue
        if x_name not in normalized and y_name not in normalized:
            continue
        input_indices = _axis_indices(axes, prefix, matrix.shape[1])
        transformed = result[:, input_indices] @ matrix[:2, :].T
        if x_name in normalized:
            result[:, normalized.index(x_name)] = transformed[:, 0]
        if y_name in normalized:
            result[:, normalized.index(y_name)] = transformed[:, 1]
    return result


def _camera_frame_size(camera: h5py.Dataset) -> Tuple[int, int]:
    frame_shape = tuple(int(value) for value in camera.shape[1:])
    if len(frame_shape) == 2:
        height, width = frame_shape
    elif len(frame_shape) == 3 and frame_shape[-1] in {1, 3, 4}:
        height, width = frame_shape[:2]
    elif len(frame_shape) == 3 and frame_shape[0] in {1, 3, 4}:
        height, width = frame_shape[1:]
    else:
        raise AnchorConversionError(f"Unsupported camera frame shape: {frame_shape}.")
    if height <= 0 or width <= 0:
        raise AnchorConversionError(f"Camera frames must have positive dimensions, got {frame_shape}.")
    return width, height


def _validate_anchors(
    hf: h5py.File,
    inspection: HDF5Inspection,
    anchors: Mapping[str, AnchorSelection],
) -> None:
    missing = [key for key in inspection.demo_keys if key not in anchors]
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "..." if len(missing) > 8 else ""
        raise AnchorConversionError(f"Missing anchors for {preview}{suffix}")
    unexpected = sorted(set(anchors).difference(inspection.demo_keys), key=_demo_sort_key)
    if unexpected:
        preview = ", ".join(unexpected[:8])
        suffix = "..." if len(unexpected) > 8 else ""
        raise AnchorConversionError(f"Anchors reference unknown demonstrations: {preview}{suffix}")
    data = hf["data"]
    for key in inspection.demo_keys:
        selection = anchors[key]
        camera = data[key]["obs"]["camera_image"]
        frame_count = int(camera.shape[0])
        pixel_xy = np.asarray(selection.pixel_xy, dtype=np.float64)
        if selection.frame_index < 0 or selection.frame_index >= frame_count:
            raise AnchorConversionError(
                f"{key} anchor frame {selection.frame_index} is outside 0..{frame_count - 1}."
            )
        width, height = _camera_frame_size(camera)
        if (
            pixel_xy.shape != (2,)
            or not np.isfinite(pixel_xy).all()
            or not (0.0 <= pixel_xy[0] < width and 0.0 <= pixel_xy[1] < height)
        ):
            raise AnchorConversionError(
                f"{key} anchor pixel {selection.pixel_xy} is outside the {width}x{height} stored image."
            )


def _structure_snapshot(hf: h5py.File) -> Dict[str, Tuple[Any, ...]]:
    snapshot: Dict[str, Tuple[Any, ...]] = {"/": ("group",)}

    def collect(name: str, value: Any) -> None:
        path = f"/{name}"
        if isinstance(value, h5py.Group):
            snapshot[path] = ("group",)
        elif isinstance(value, h5py.Dataset):
            snapshot[path] = (
                "dataset",
                tuple(int(item) for item in value.shape),
                str(value.dtype),
                tuple(value.maxshape) if value.maxshape is not None else None,
                tuple(value.chunks) if value.chunks is not None else None,
                value.compression,
                repr(value.compression_opts),
                bool(value.shuffle),
                bool(value.fletcher32),
                value.scaleoffset,
            )

    hf.visititems(collect)
    return snapshot


def _mask_snapshot(hf: h5py.File) -> Dict[str, np.ndarray]:
    snapshot: Dict[str, np.ndarray] = {}
    mask = hf.get("mask")
    if not isinstance(mask, h5py.Group):
        return snapshot

    def collect(name: str, value: Any) -> None:
        if isinstance(value, h5py.Dataset):
            snapshot[name] = np.asarray(value[...]).copy()

    mask.visititems(collect)
    return snapshot


def _verify_mask_snapshot(expected: Mapping[str, np.ndarray], actual: Mapping[str, np.ndarray]) -> None:
    if set(expected) != set(actual):
        raise AnchorConversionError("Anchored output verification found changed mask datasets.")
    for name in expected:
        if not np.array_equal(expected[name], actual[name]):
            raise AnchorConversionError(f"Anchored output verification found changed mask/{name}.")


def _verify_anchor_coordinates(
    hf: h5py.File,
    inspection: HDF5Inspection,
    anchors: Mapping[str, AnchorSelection],
) -> None:
    data = hf["data"]
    for demo_key in inspection.demo_keys:
        demo = data[demo_key]
        anchor = anchors[demo_key]
        pipette = demo["obs"]["pipette_positions"]
        stage = demo["obs"]["stage_positions"]
        pip_x, pip_y, pip_z = _output_indices(
            _decode_axes(pipette),
            "pipette",
            allow_plain=True,
        )
        stage_x, stage_y, stage_z = _output_indices(
            _decode_axes(stage),
            "stage",
            allow_plain=True,
        )
        pip_anchor = np.asarray(pipette[anchor.frame_index], dtype=np.float64)
        stage_origin = np.asarray(stage[0], dtype=np.float64)
        if not np.allclose(
            pip_anchor[[pip_x, pip_y]],
            np.asarray(anchor.pixel_xy, dtype=np.float64),
            rtol=1e-6,
            atol=1e-4,
        ):
            raise AnchorConversionError(f"{demo_key} pipette anchor XY verification failed.")
        if pip_z is not None and not np.isclose(pip_anchor[pip_z], 0.0, rtol=0.0, atol=1e-6):
            raise AnchorConversionError(f"{demo_key} pipette anchor Z verification failed.")
        origin_indices = [stage_x, stage_y] + ([stage_z] if stage_z is not None else [])
        if not np.allclose(
            stage_origin[origin_indices],
            0.0,
            rtol=0.0,
            atol=1e-6,
        ):
            raise AnchorConversionError(f"{demo_key} stage origin verification failed.")


def _publish_exclusive(temp_path: Path, target_path: Path) -> None:
    if os.name == "nt":
        try:
            os.rename(temp_path, target_path)
        except FileExistsError as exc:
            raise AnchorConversionError(f"Anchored output already exists: {target_path}") from exc
        except OSError as exc:
            raise AnchorConversionError(
                f"Could not publish {target_path.name} without overwriting an existing file: {exc}"
            ) from exc
        return

    try:
        os.link(temp_path, target_path)
    except FileExistsError as exc:
        raise AnchorConversionError(f"Anchored output already exists: {target_path}") from exc
    except OSError as exc:
        raise AnchorConversionError(
            f"Could not publish {target_path.name} without overwriting an existing file: {exc}"
        ) from exc
    try:
        temp_path.unlink()
    except OSError as exc:
        try:
            target_path.unlink()
        except OSError:
            pass
        raise AnchorConversionError(f"Could not finalize {target_path.name}: {exc}") from exc


def _transform_demo(
    demo: h5py.Group,
    anchor: AnchorSelection,
    matrices: CalibrationMatrices,
    calibration_name: str,
) -> None:
    obs = demo["obs"]
    pip_ds = obs["pipette_positions"]
    stage_ds = obs["stage_positions"]
    pip_axes = _decode_axes(pip_ds)
    stage_axes = _decode_axes(stage_ds)
    pip_values = np.asarray(pip_ds[...], dtype=np.float64)
    stage_values = np.asarray(stage_ds[...], dtype=np.float64)
    pip_reference = pip_values[anchor.frame_index].copy()
    stage_reference = stage_values[0].copy()

    pip_ds[...] = transform_positions(
        pip_values,
        pip_axes,
        matrices.pipette,
        pip_reference,
        prefix="pipette",
        anchor_xy=anchor.pixel_xy,
    ).astype(pip_ds.dtype, copy=False)
    stage_ds[...] = transform_positions(
        stage_values,
        stage_axes,
        matrices.stage,
        stage_reference,
        prefix="stage",
    ).astype(stage_ds.dtype, copy=False)
    pip_ds.attrs["units"] = _coordinate_units(pip_axes)
    stage_ds.attrs["units"] = _coordinate_units(stage_axes)

    next_obs = demo.get("next_obs")
    if isinstance(next_obs, h5py.Group):
        if "pipette_positions" in next_obs:
            next_pip = next_obs["pipette_positions"]
            next_pip[...] = transform_positions(
                np.asarray(next_pip[...], dtype=np.float64),
                _decode_axes(next_pip),
                matrices.pipette,
                pip_reference,
                prefix="pipette",
                anchor_xy=anchor.pixel_xy,
            ).astype(next_pip.dtype, copy=False)
            next_pip.attrs["units"] = pip_ds.attrs["units"]
        if "stage_positions" in next_obs:
            next_stage = next_obs["stage_positions"]
            next_stage[...] = transform_positions(
                np.asarray(next_stage[...], dtype=np.float64),
                _decode_axes(next_stage),
                matrices.stage,
                stage_reference,
                prefix="stage",
            ).astype(next_stage.dtype, copy=False)
            next_stage.attrs["units"] = stage_ds.attrs["units"]

    actions = demo.get("actions")
    if isinstance(actions, h5py.Dataset):
        action_axes = _decode_axes(actions)
        actions[...] = transform_actions(
            np.asarray(actions[...], dtype=np.float64),
            action_axes,
            matrices.stage,
            matrices.pipette,
        ).astype(actions.dtype, copy=False)
        actions.attrs["xy_units"] = "pixels"
        actions.attrs["z_units"] = "um"

    demo.attrs["anchor_frame_index"] = int(anchor.frame_index)
    demo.attrs["anchor_pixel_xy"] = np.asarray(anchor.pixel_xy, dtype=np.float64)
    demo.attrs["pipette_encoder_reference"] = pip_reference
    demo.attrs["stage_encoder_reference"] = stage_reference
    demo.attrs["anchor_calibration_file"] = calibration_name
    demo.attrs["pipette_calibration_matrix"] = "manip.M"
    demo.attrs["stage_calibration_matrix"] = "stage.M"


def convert_hdf5(
    source: Path | str,
    calibration_file: Path | str,
    anchors: Mapping[str, AnchorSelection],
    *,
    progress: Optional[Callable[[int, int, str], None]] = None,
    cancelled: Optional[Callable[[], bool]] = None,
) -> Path:
    source_path = Path(source).resolve()
    calibration_path = Path(calibration_file).resolve()
    try:
        captured_source_fingerprint = source_fingerprint(source_path)
    except OSError:
        captured_source_fingerprint = None
    inspection = inspect_hdf5(source_path)
    if inspection.errors:
        raise AnchorConversionError("\n".join(inspection.errors))
    if captured_source_fingerprint is None:
        raise AnchorConversionError(f"Could not fingerprint source HDF5: {source_path}")
    if source_fingerprint(source_path) != captured_source_fingerprint:
        raise AnchorConversionError("Source HDF5 changed while it was being inspected.")
    output_path = derive_output_path(source_path)
    metadata_path = derive_metadata_path(source_path)
    if output_path.exists():
        raise AnchorConversionError(f"Anchored output already exists: {output_path}")
    if metadata_path.exists():
        raise AnchorConversionError(f"Anchored metadata already exists: {metadata_path}")

    matrices = load_calibration(calibration_path)
    # Validate axis/matrix compatibility before copying a potentially large file.
    _axis_indices(
        inspection.stage_axes,
        "stage",
        matrices.stage.shape[1],
        allow_plain=True,
    )
    _output_indices(inspection.stage_axes, "stage", allow_plain=True)
    _axis_indices(
        inspection.pipette_axes,
        "pipette",
        matrices.pipette.shape[1],
        allow_plain=True,
    )
    _output_indices(inspection.pipette_axes, "pipette", allow_plain=True)
    if inspection.action_axes:
        transform_actions(
            np.zeros((1, len(inspection.action_axes)), dtype=np.float64),
            inspection.action_axes,
            matrices.stage,
            matrices.pipette,
        )

    with h5py.File(source_path, "r") as source_hf:
        _validate_anchors(source_hf, inspection, anchors)
        source_structure = _structure_snapshot(source_hf)
        source_masks = _mask_snapshot(source_hf)
    if source_fingerprint(source_path) != captured_source_fingerprint:
        raise AnchorConversionError("Source HDF5 changed while it was being validated.")

    temp_path = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    temp_metadata = metadata_path.with_name(f".{metadata_path.name}.tmp")
    if temp_path.exists():
        raise AnchorConversionError(f"Temporary anchored output already exists: {temp_path}")
    if temp_metadata.exists():
        raise AnchorConversionError(f"Temporary anchored metadata already exists: {temp_metadata}")

    temp_output_owned = False
    temp_metadata_owned = False
    metadata_published = False
    try:
        try:
            with temp_path.open("xb"):
                pass
            temp_output_owned = True
            with temp_metadata.open("xb"):
                pass
            temp_metadata_owned = True
        except FileExistsError as exc:
            raise AnchorConversionError(
                "Another anchoring conversion is already using the temporary output paths."
            ) from exc
        if cancelled and cancelled():
            raise AnchorConversionError("Anchoring cancelled before copy.")
        if progress:
            progress(0, len(inspection.demo_keys), "Copying source HDF5")
        shutil.copy2(source_path, temp_path)
        if cancelled and cancelled():
            raise AnchorConversionError("Anchoring cancelled after copy.")
        if source_fingerprint(source_path) != captured_source_fingerprint:
            raise AnchorConversionError("Source HDF5 changed while it was being copied.")
        with h5py.File(temp_path, "r+") as hf:
            data = hf["data"]
            total = len(inspection.demo_keys)
            for index, demo_key in enumerate(inspection.demo_keys, 1):
                if cancelled and cancelled():
                    raise AnchorConversionError("Anchoring cancelled.")
                _transform_demo(data[demo_key], anchors[demo_key], matrices, calibration_path.name)
                if progress:
                    progress(index, total, demo_key)
            if cancelled and cancelled():
                raise AnchorConversionError("Anchoring cancelled after coordinate transformation.")
            now = _datetime.datetime.now(_datetime.timezone.utc).isoformat()
            hf.attrs["anchored_coordinates"] = True
            hf.attrs["anchor_source_hdf5"] = source_path.name
            hf.attrs["anchor_conversion_timestamp"] = now
            hf.attrs["anchor_calibration_file"] = calibration_path.name
            hf.attrs["coordinate_representation"] = COORDINATE_REPRESENTATION
            hf.attrs["pipette_axis_units"] = "x:pixels,y:pixels,z:um"
            hf.attrs["stage_axis_units"] = "x:pixels,y:pixels,z:um"
            hf.attrs["anchor_sidecar_version"] = ANCHOR_SIDECAR_VERSION
            data.attrs["anchored_coordinates"] = True
            hf.flush()

        verified = inspect_hdf5(temp_path)
        verification_errors = tuple(
            error for error in verified.errors if error != "Source HDF5 is already marked as anchored."
        )
        if verification_errors or not verified.anchored:
            raise AnchorConversionError(
                "Anchored output verification failed: " + "; ".join(verification_errors or ("missing anchored flag",))
            )
        if verified.demo_keys != inspection.demo_keys or verified.split_counts != inspection.split_counts:
            raise AnchorConversionError("Anchored output verification found changed demo or split counts.")
        with h5py.File(temp_path, "r") as verified_hf:
            if _structure_snapshot(verified_hf) != source_structure:
                raise AnchorConversionError("Anchored output verification found changed HDF5 structure.")
            _verify_mask_snapshot(source_masks, _mask_snapshot(verified_hf))
            _verify_anchor_coordinates(verified_hf, inspection, anchors)
        if cancelled and cancelled():
            raise AnchorConversionError("Anchoring cancelled after verification.")
        if source_fingerprint(source_path) != captured_source_fingerprint:
            raise AnchorConversionError("Source HDF5 changed before anchored output publication.")

        metadata = {
            "source_hdf5": source_path.name,
            "output_hdf5": output_path.name,
            "source_fingerprint": captured_source_fingerprint,
            "calibration_file": calibration_path.name,
            "coordinate_representation": COORDINATE_REPRESENTATION,
            "anchor_sidecar_version": ANCHOR_SIDECAR_VERSION,
            "num_demos": len(inspection.demo_keys),
            "split_counts": dict(inspection.split_counts),
            "anchors": {key: anchors[key].to_mapping() for key in inspection.demo_keys},
            "conversion_settings": {
                "calibration_file": calibration_path.name,
                "coordinate_representation": COORDINATE_REPRESENTATION,
                "matrices": {
                    "stage": {
                        "identifier": "stage.M",
                        "shape": list(matrices.stage.shape),
                    },
                    "pipette": {
                        "identifier": "manip.M",
                        "shape": list(matrices.pipette.shape),
                    },
                },
                "calibration_offsets_ignored": True,
                "units": {
                    "pipette": {"x": "pixels", "y": "pixels", "z": "um"},
                    "stage": {"x": "pixels", "y": "pixels", "z": "um"},
                },
                "reference_policies": {
                    "pipette_xy": "selected stored frame and clicked pixel",
                    "pipette_z": "relative to selected stored frame",
                    "stage_xyz": "relative to first stored observation",
                    "actions": "linear transform only; no anchor translation",
                },
            },
            "anchor_summary": {
                "anchor_count": len(anchors),
                "required_demo_count": len(inspection.demo_keys),
                "all_demonstrations_anchored": all(
                    key in anchors for key in inspection.demo_keys
                ),
            },
            "verification_results": {
                "schema_valid": True,
                "demo_keys_preserved": True,
                "split_counts_preserved": True,
                "hdf5_structure_preserved": True,
                "mask_contents_preserved": True,
                "anchor_coordinates_verified": True,
                "source_fingerprint_unchanged": True,
            },
            "verified": True,
            "created_at": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        }
        temp_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        # Publish the HDF5 last. If its exclusive link fails, no final dataset is
        # exposed; the companion metadata is removed by the failure handler.
        _publish_exclusive(temp_metadata, metadata_path)
        metadata_published = True
        _publish_exclusive(temp_path, output_path)
        return output_path
    except Exception:
        for path, owned in (
            (temp_path, temp_output_owned),
            (temp_metadata, temp_metadata_owned),
        ):
            if owned and path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass
        # metadata_published is set only after this conversion exclusively
        # creates the target, so it remains safe to remove even when a competing
        # conversion created the final HDF5 path.
        if metadata_published and metadata_path.exists():
            try:
                metadata_path.unlink()
            except OSError:
                pass
        raise
