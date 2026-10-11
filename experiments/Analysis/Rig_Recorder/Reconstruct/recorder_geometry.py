"""Pure coordinate and target helpers for rig replay.

Movement entries use the replay loader's millimetre coordinates. The configurable
micrometre defaults are visual assumptions; image detections provide no depth.
"""

import bisect
import json
import math
from pathlib import Path
import re


CENTRAL_CELL_STATES = frozenset({
    "hunt", "hunt_cell", "gigaseal", "seal", "break_in", "breakin",
})
CENTERED_PIPETTE_STATES = CENTRAL_CELL_STATES | frozenset({
    "approach", "approach_cell",
})


def normalize_state_name(state):
    """Normalize recorded state names and human-readable labels consistently."""
    return re.sub(r"[^a-z0-9]+", "_", str(state or "").strip().lower()).strip("_")


def uses_central_cell(state):
    """Restrict target-cell parameters to hunt, sealing, and break-in."""
    return normalize_state_name(state) in CENTRAL_CELL_STATES


def uses_centered_pipette(state):
    return normalize_state_name(state) in CENTERED_PIPETTE_STATES


def load_moscow_geometry(rig_config_path=None):
    """Read orientation and inference-model names from Moscow's calibration.

    Only the known top-level YAML scalar fields are needed, so this reader adds
    no YAML, numerical, Qt, or rig hardware dependency to replay geometry.
    """
    repo_root = Path(__file__).resolve().parents[4]
    rig_path = Path(rig_config_path) if rig_config_path is not None else (
        repo_root / "rig_setup" / "rig_configs" / "moscow_rig.json")
    with rig_path.open("r", encoding="utf-8-sig") as stream:
        rig_config = json.load(stream)
    calibration_name = rig_config["calibration_file"]
    calibration_path = Path(calibration_name)
    if not calibration_path.is_absolute():
        calibration_path = rig_path.parent.parent / "cal_configs" / calibration_path
    scalar_fields = {}
    for line in calibration_path.read_text(encoding="utf-8-sig").splitlines():
        match = re.match(r"^(pipette_y_rotation|pipette_z_rotation|pipette_detector_model|pipette_focuser_model):\s*(.*?)\s*$", line)
        if match:
            scalar_fields[match.group(1)] = match.group(2).split(" #", 1)[0].strip().strip("\"'")
    geometry = {
        "pipette_y_rotation": float(scalar_fields["pipette_y_rotation"]),
        "pipette_z_rotation": float(scalar_fields["pipette_z_rotation"]),
        "rig_config_path": str(rig_path.resolve()),
        "calibration_path": str(calibration_path.resolve()),
    }
    if not all(math.isfinite(geometry[key]) for key in ("pipette_y_rotation", "pipette_z_rotation")):
        raise ValueError("Configured pipette rotations must be finite.")
    model_root = repo_root / "patcherbot" / "deepLearning" / "pipetteModel"
    for key in ("pipette_detector_model", "pipette_focuser_model"):
        model_name = scalar_fields.get(key)
        geometry[key] = model_name
        model_path = Path(model_name) if model_name else None
        if model_path is not None and not model_path.is_absolute():
            model_path = model_root / model_path
        geometry[key + "_path"] = str(model_path.resolve()) if model_path is not None else None
    return geometry


def pipette_mesh_rotations(geometry=None):
    """Return cylinder rotations (Y, then stage Z) from configured elevation.

    Cylinders start along +Z. A 25-degree pipette elevation above XY therefore
    needs a 65-degree Y rotation, followed by the configured XY azimuth.
    """
    geometry = load_moscow_geometry() if geometry is None else geometry
    return 90.0 - float(geometry["pipette_y_rotation"]), float(geometry["pipette_z_rotation"])


def phase_reference_for_timestamp(movement_data, state_windows, timestamp):
    """Choose an approach/hunt anchor that survives sealing and break-in.

    State windows contain (start, finish, name, outcome). Each new approach/hunt
    initializes its assumed gap once; later states inherit the preceding phase.
    Computing the reference from timestamps also makes backward seeking stable.
    """
    if not movement_data:
        return None, "hunt_cell"
    anchor_time = None
    phase = "hunt_cell"
    for window in state_windows:
        started = float(window[0])
        state = normalize_state_name(window[2])
        if (started <= timestamp and state in {"approach", "approach_cell", "hunt", "hunt_cell"}
                and (anchor_time is None or started > anchor_time)):
            anchor_time = started
            phase = "approach_cell" if state.startswith("approach") else "hunt_cell"
    if anchor_time is None:
        return movement_data[0], phase
    times = [entry["time"] for entry in movement_data]
    index = bisect.bisect_left(times, anchor_time)
    if index == 0:
        return movement_data[0], phase
    if index == len(times):
        return movement_data[-1], phase
    before, after = movement_data[index - 1], movement_data[index]
    reference = after if after["time"] - anchor_time < anchor_time - before["time"] else before
    return reference, phase


def _positive_scale(pixels_per_um):
    value = float(pixels_per_um)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Pixels per micrometre must be finite and positive.")
    return value


def _image_point(point, width, height):
    try:
        x, y = float(point[0]), float(point[1])
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    if not (0 <= x < width and 0 <= y < height):
        return None
    return x, y


def select_central_cell(detections, width, height):
    """Return (x, y, confidence) nearest image center, skipping invalid points.

    Confidence is retained from the selected detector output and is None when
    that detector does not supply it. Selection does not use confidence ranking.
    """
    if width <= 0 or height <= 0 or detections is None:
        return None
    best = None
    best_distance = math.inf
    for detection in detections:
        point = _image_point(detection, width, height)
        if point is None:
            continue
        x, y = point
        distance = (x - width / 2.0) ** 2 + (y - height / 2.0) ** 2
        if distance >= best_distance:
            continue
        confidence = None
        try:
            value = float(detection[2])
            if math.isfinite(value):
                confidence = value
        except (TypeError, ValueError, IndexError, KeyError):
            pass
        best = (x, y, confidence)
        best_distance = distance
    return best


def cell_inference_parameters(detection, width, height, pixels_per_um=6.0,
                              pipette_point=None):
    """Expose measured detector output for one target without inventing depth."""
    scale = _positive_scale(pixels_per_um)
    point = _image_point(detection, width, height)
    if point is None:
        return {}
    x, y = point
    parameters = {
        "x_px": x,
        "y_px": y,
        "confidence": None,
        "center_distance_um": math.hypot(x - width / 2.0, y - height / 2.0) / scale,
    }
    try:
        confidence = float(detection[2])
        if math.isfinite(confidence):
            parameters["confidence"] = confidence
    except (TypeError, ValueError, IndexError, KeyError):
        pass
    tip = _image_point(pipette_point, width, height)
    if tip is not None:
        parameters["tip_distance_um"] = math.hypot(x - tip[0], y - tip[1]) / scale
    return parameters


def _movement_positions(entry):
    stage = tuple(float(value) for value in entry["stage"])
    pipette = tuple(float(value) for value in entry["pipette"])
    microscope_z = float(entry["microscope_z"])
    if len(stage) != 2 or len(pipette) != 3:
        raise ValueError("Movement must contain stage XY and pipette XYZ positions.")
    if not all(math.isfinite(value) for value in stage + pipette + (microscope_z,)):
        raise ValueError("Movement positions must be finite.")
    return microscope_z, stage, pipette


def reconstruction_positions(movement_entry, reference_entry, *,
                             microscope_offset_enabled=True,
                             microscope_offset_um=1350.0,
                             centered_pipette_enabled=True,
                             pipette_cell_distance_um=20.0,
                             active_state=None, approach_cell_distance_um=75.0,
                             cell_point=None, image_size=None,
                             pixels_per_um=6.0):
    """Apply initial geometry once, preserving recorded relative movement.

    The caller supplies the reference from phase_reference_for_timestamp and
    its phase as active_state. Approach starts at 75 micrometres and hunt at 20
    by default; later sealing and break-in retain the hunt reference. The centered
    assumption places the reference pipette above the cell plane and centers its
    XY position. The microscope offset places its reference Z above the reference
    pipette. Each device then retains its own recorded movement deltas.

    Inputs/outputs are millimetres. Image coordinates use positive X to the right
    and positive Y upwards; cells lie on the assumed Z=0 plane. Observed image
    offsets already reflect stage motion, so stage translation is not added a
    second time. With both defaults disabled, recorded positions are unchanged.
    """
    microscope_z, stage, pipette = _movement_positions(movement_entry)
    reference_microscope, reference_stage, reference_pipette = _movement_positions(reference_entry)
    gap_um = (approach_cell_distance_um if normalize_state_name(active_state) in
              {"approach", "approach_cell"} else pipette_cell_distance_um)
    gap = float(gap_um) / 1000.0
    offset = float(microscope_offset_um) / 1000.0
    if not (math.isfinite(gap) and gap >= 0 and math.isfinite(offset) and offset >= 0):
        raise ValueError("Assumed distances must be finite and nonnegative.")

    if centered_pipette_enabled:
        stage = tuple(value - baseline for value, baseline in zip(stage, reference_stage))
        pipette = tuple(value - baseline for value, baseline in zip(pipette, reference_pipette))
        pipette = (pipette[0], pipette[1], pipette[2] + gap)
    if microscope_offset_enabled:
        reference_tip_z = gap if centered_pipette_enabled else reference_pipette[2]
        microscope_z = microscope_z - reference_microscope + reference_tip_z + offset

    cell_position = (0.0, 0.0, 0.0) if centered_pipette_enabled else None
    if cell_point is not None and image_size is not None:
        width, height = image_size
        point = _image_point(cell_point, width, height)
        if point is not None:
            scale_mm = _positive_scale(pixels_per_um) * 1000.0
            cell_position = ((point[0] - width / 2.0) / scale_mm,
                             -(point[1] - height / 2.0) / scale_mm, 0.0)
    return {"microscope_z": microscope_z, "stage": stage,
            "pipette": pipette, "cell_position": cell_position}
