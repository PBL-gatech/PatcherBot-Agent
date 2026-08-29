# coding=utf-8
"""
Calibration configuration parameters for manipulator units.
"""
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
