# coding=utf-8
"""
Calibration configuration parameters for manipulator units.
"""
import numpy as np
import param
from patcherbot.utils.config import Config, NumberWithUnit, Number, Boolean, Tuple

__all__ = ["CalibrationConfig"]


class CalibrationConfig(Config):
    position_update = NumberWithUnit(1000, unit="ms",
                                     doc="dt for updating displayed pos.",
                                     bounds=(0, 10000))

    autofocus_dist = NumberWithUnit(15, unit="um",
                                     doc="z dist to scan for autofocusing.",
                                     bounds=(10, 5000))

    stage_diag_move = NumberWithUnit(500, unit="um",
                                     doc="x, y dist to move for stage cal.",
                                     bounds=(0, 10000))

    frame_lag = NumberWithUnit(4, unit="frames",
                                     doc="number of frames between for computing change with optical flow",
                                     bounds=(1, 20))

    y_delta_scan = NumberWithUnit(750, unit="um",
                                  doc="configured vertical scan distance for stage scans",
                                  bounds=(0, 100000))

    pipette_diag_move = NumberWithUnit(200, unit="um",
                                     doc="x, y dist to move for pipette cal.",
                                     bounds=(50, 10000))
    stage_x_axis_flip = Boolean(False,
                                doc="Flip the x axis of the stage")
    stage_y_axis_flip = Boolean(True,
                                doc="Flip the y axis of the stage")
    
    # Legacy single-pipette angle fallback; multi-pipette rigs should configure
    # pipette_rotation_matrix entries keyed by pipette ID below.
    pipette_z_rotation = NumberWithUnit(-60.75, unit="degrees",
                                doc="Rotation of the pipette in the xy plane (degrees)",
                                bounds=(-360, 360))
    pipette_y_rotation = NumberWithUnit(25, unit="degrees",
                                doc="Rotation of the pipette in the xz plane (degrees)",
                                bounds=(-90, 90))
    clean_move_order = param.List(default=["y", "x", "z"],
                                item_type=str,
                                doc="Axis command order for cleaning approach; safe-space return uses reverse order")
    pipette_k_scale = Number(1.0,
                                doc="Scaling factor for pipette movement",
                                bounds=(-10.0, 10.0))
    pipette_rotation_matrix = param.Dict(default={},
                                        doc="3x3 rotation matrix for each pipette keyed by pipette ID")
    pipette_affine_matrix = param.Dict(default={},
                                       doc="3x4 affine transform for each pipette keyed by pipette ID")

    microscope_units_per_um = Number(5.0,
                                     doc="Microscope controller units per micron",
                                     bounds=(0.001, 1000))
    objective_lift_um = NumberWithUnit(10000, unit="um",
                                     doc="Distance to lift the microscope objective during objective switches",
                                        bounds=(0, 20000))
    home_position_delta_um = NumberWithUnit(-1000, unit="um",
                                           doc="Vertical offset from stage cell surface to pipette home position",
                                             bounds=(-100000, 100000))
    safe_position_delta_um = NumberWithUnit(-8000, unit="um",
                                            doc="Offset from home to safe position along pipette axis",
                                            bounds=(-100000, 100000))
    native_zero = Number(1962,
                         doc="Pressure controller native zero (DAC units at 0 mbar)",
                         bounds=(0, 4096))
    native_per_mbar = Number(2.7836,
                             doc="Pressure controller native units per mbar",
                             bounds=(0.001, 10))
    reader_offset = Number(516.72,
                           doc="Pressure reader offset before scaling to raw units",
                           bounds=(0, 4096))
    reader_scale = Number(0.3923,
                          doc="Pressure reader scale factor to convert to raw units",
                          bounds=(0.001, 10))
    pipette_detector_model = param.String(default="",
                                          doc="Pipette detector model path")
    pipette_focuser_model = param.String(default="",
                                         doc="Pipette focuser model path")
    pipette_focus_crop_feature = Boolean(False,
                                         doc="Whether to crop the image around the pipette tip for the focus model")
    use_ai_features = Boolean(False,
                              doc="Enable AI-based vision features (SAM/LightGlue/robomimic)")

    home_position =  Tuple((0, 0, 0), doc="Home position of the pipette in um")
    home_position_stage =  Tuple((0, 0, 0), doc="Home position of the stage in um")
    safe_position =  Tuple((0, 0, 0), doc="Safe position of the pipette in um")
    safe_position_stage =  Tuple((0, 0, 0), doc="Safe position of the stage in um")
    bath_position =  Tuple((0, 0, 0), doc="Bath position of the pipette in um")

    @staticmethod
    def _normalise_pipette_id(pipette_id):
        if pipette_id is None:
            return None
        pipette_id = str(pipette_id).strip()
        if not pipette_id:
            return None
        if pipette_id.startswith("pipette_"):
            suffix = pipette_id.rsplit("_", 1)[-1]
            return f"pipette_{suffix}" if suffix.isdigit() else pipette_id
        try:
            index = int(pipette_id)
        except (TypeError, ValueError):
            suffix = pipette_id.rsplit("_", 1)[-1]
            return f"pipette_{suffix}" if suffix.isdigit() else pipette_id
        return f"pipette_{index}"

    def _lookup_matrix_for_pipette(self, matrix_store, pipette_id=None, default=None):
        raw_key = str(pipette_id).strip() if pipette_id is not None else None
        if raw_key and isinstance(matrix_store, dict) and raw_key in matrix_store:
            return matrix_store[raw_key]
        key = self._normalise_pipette_id(pipette_id)
        if key is not None and isinstance(matrix_store, dict):
            if key in matrix_store:
                return matrix_store[key]
            if key.startswith("pipette_"):
                numeric_key = key.replace("pipette_", "", 1)
                if numeric_key.isdigit() and int(numeric_key) in matrix_store:
                    return matrix_store[int(numeric_key)]
        return default

    def resolve_pipette_rotation_matrix(self, pipette_id=None):
        configured = self._lookup_matrix_for_pipette(
            getattr(self, "pipette_rotation_matrix", {}),
            pipette_id,
            default=None,
        )
        if configured is not None:
            return np.asarray(configured, dtype=float).reshape(3, 3)

        theta_y = float(self.pipette_y_rotation) * np.pi / 180.0
        theta_z = float(self.pipette_z_rotation) * np.pi / 180.0
        ry = np.array([
            [np.cos(theta_y), 0.0, np.sin(theta_y)],
            [0.0, 1.0, 0.0],
            [-np.sin(theta_y), 0.0, np.cos(theta_y)],
        ], dtype=float)
        rz = np.array([
            [np.cos(theta_z), -np.sin(theta_z), 0.0],
            [np.sin(theta_z), np.cos(theta_z), 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=float)
        return ry @ rz

    def resolve_pipette_axis_direction(self, pipette_id=None):
        configured = self._lookup_matrix_for_pipette(
            getattr(self, "pipette_rotation_matrix", {}),
            pipette_id,
            default=None,
        )
        if configured is None:
            theta = float(self.pipette_y_rotation) * np.pi / 180.0
            return np.array([np.cos(theta), 0.0, np.sin(theta)], dtype=float)
        return self.resolve_pipette_rotation_matrix(pipette_id) @ np.array([1.0, 0.0, 0.0])

    def resolve_pipette_affine_matrix(self, pipette_id=None):
        configured = self._lookup_matrix_for_pipette(
            getattr(self, "pipette_affine_matrix", {}),
            pipette_id,
            default=None,
        )
        if configured is not None:
            matrix = np.asarray(configured, dtype=float)
            if matrix.shape == (2, 3):
                return np.vstack([matrix, [0.0, 0.0, 1.0]])
            if matrix.shape == (3, 3):
                return np.hstack([matrix, np.zeros((3, 1), dtype=float)])
            if matrix.shape == (3, 4):
                return matrix
        return None

    categories = [
        ("Stage Calibration", [
            "autofocus_dist",
            "stage_diag_move",
            "frame_lag",
            "y_delta_scan",
            "stage_x_axis_flip",
            "stage_y_axis_flip",
            "microscope_units_per_um",
            "objective_lift_um",
        ]),
        ("Pipette Calibration", [
            "pipette_diag_move",
            "pipette_z_rotation",
            "pipette_y_rotation",
            "pipette_rotation_matrix",
            "pipette_affine_matrix",
            "clean_move_order",
            "pipette_k_scale",
            "pipette_detector_model",
            "pipette_focuser_model",
            "pipette_focus_crop_feature",
            "use_ai_features",
        ]),
        ("Display", ["position_update"]),
        ("Pressure", ["native_zero", "native_per_mbar", "reader_offset", "reader_scale"]),
        ("Positions", ["home_position", "home_position_stage", "safe_position", "safe_position_stage", "bath_position", "safe_position_delta_um", "home_position_delta_um"]),
    ]
