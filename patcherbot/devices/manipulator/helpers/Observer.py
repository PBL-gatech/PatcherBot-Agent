"""Collect requested telemetry through the rig's existing methods."""
from copy import deepcopy
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..calibratedunit import CalibratedUnit, CalibratedStage


class Observer:
    default_fields = ("manipulator_position", "pipette_image_xy", "pipette_defocus_um",
                      "stage_positions", "camera_image", "resistance", "deep_learning")
    available_fields = frozenset(default_fields + ("pressure", "commanded_pressure_mbar",
        "pressure_atm_state", "access_resistance", "capacitance", "cell_detections"))

    def __init__(self, controller, attempt_id=None):
        self.controller = controller
        self.unit: "CalibratedUnit" = controller.calibrated_unit
        self.stage: "CalibratedStage" = controller.calibrated_stage
        self.attempt_id, self.attempt_token = attempt_id, object()
        self.deck = []

    def observe(self, *, fields=None, num_measurements=None, interval=None, phase=None,
                evidence=None, raw_resistance=False):
        fields = tuple(dict.fromkeys(self.default_fields if fields is None else fields))
        if evidence is None:
            evidence = self.infer_frame(fields, getattr(phase, "frame_context", None))
        evidence = dict(evidence)
        image = evidence.pop("_source_image", None)
        acquired = evidence.get("frame_retrieval_started_at")
        if acquired is None:
            acquired = evidence.get("frame_available_at")
        age = None if acquired is None else time.monotonic() - acquired
        evidence.update(age_s=age, stale=age is None or not 0 <= age <= 1.)
        controller, unit, stage = self.controller, self.unit, self.stage
        readers = {
            "manipulator_position": unit.position,
            "stage_positions": lambda: np.append(stage.position()[:2],
                unit.microscope.position() / unit.config.microscope_units_per_um),
            "pressure": lambda: controller.pressure.get_last_acquisition_sample()[0],
            "commanded_pressure_mbar": controller.pressure.get_pressure,
            "pressure_atm_state": controller.pressure.get_ATM,
            "resistance": controller.daq.resistance if raw_resistance else lambda: controller.resistanceRamp(
                num_measurements=1 if num_measurements is None else num_measurements,
                interval=.001 if interval is None else interval),
            "access_resistance": lambda: controller.accessRamp(
                num_measurements=5 if num_measurements is None else num_measurements,
                interval=.005 if interval is None else interval),
            "capacitance": lambda: controller.capacitanceRamp(
                num_measurements=5 if num_measurements is None else num_measurements,
                interval=.005 if interval is None else interval),
            "pipette_image_xy": lambda: evidence.get("pipette_position"),
            "pipette_defocus_um": lambda: evidence.get("pipette_focus"),
            "cell_detections": lambda: evidence.get("cell_detections"),
            "camera_image": lambda: None if image is None else image.copy(),
            "deep_learning": lambda: evidence,
        }
        observation = {}
        for name in fields:
            value = readers[name]()
            if value is None and name not in ("camera_image", "deep_learning", "cell_detections"):
                if not (raw_resistance and name == "resistance"):
                    value = np.full(2, np.nan) if name == "pipette_image_xy" else np.nan
            observation[name] = value
        observation.update(observed_at=time.monotonic(), observation_id=len(self.deck), attempt_id=self.attempt_id,
            phase=None if phase is None else type(phase).__name__, phase_run=None if phase is None else phase.observation_run)
        self.deck.append(deepcopy({name: value for name, value in observation.items() if name != "camera_image"}))
        return observation

    def infer_frame(self, fields, frame_context=None, *, confirmation=None, pipette_radius_um=None):
        unit, stage = self.unit, self.stage
        if confirmation is not None:
            helper = stage.focusHelper
            sweep = dict(pipette=True, best_xy=None, depth_advisory=False,
                pixels_to_um_relative=stage.pixels_to_um_relative,
                info=self.controller.info, warning=self.controller.warning, **confirmation)
            evidence = {"confirmed_tip_xy": None}
            helper.start_confirmation(sweep)
            try:
                for _ in range(helper.CONFIRM_MAX_FRAMES):
                    unit.abort_if_requested()
                    evidence = self.infer_frame(fields, frame_context, pipette_radius_um=pipette_radius_um)
                    if helper.update_confirmation(sweep, evidence):
                        return dict(evidence, confirmed_tip_xy=sweep["confirmation"]["confirmed_tip_xy"])
                    unit.sleep(0.05)
                self.controller.warning('Pipette reference not refreshed: no 3 stable confident detections; keeping calibrated marker')
            except Exception as error:
                unit.abort_if_requested()
                self.controller.warning(f'Pipette reference display refresh failed: {error}; keeping calibrated marker')
            return dict(evidence, confirmed_tip_xy=None)

        fields = set(fields)
        visual = fields & {"pipette_image_xy", "pipette_defocus_um", "cell_detections", "deep_learning"}
        if not visual and "camera_image" not in fields:
            return {}
        frame_id, _, image, timing = stage.camera.last_raw_frame_data(include_timing=True) or (None, None, None, {})
        evidence = dict(source_frame=frame_id, frame_retrieval_started_at=timing.get("retrieval_started_at"),
            frame_available_at=timing.get("available_at"), _source_image=image, source_positions={},
            pipette_position=None, pipette_focus=None, cell_detections=None,
            confidence={}, status={}, errors={}, pipette_focus_confidence_supported=None)
        if image is not None and visual:
            image = image.copy()
            evidence["_source_image"] = image
            evidence["source_positions"] = {name: read() for name, read in (frame_context or {}).items()}
            if visual & {"pipette_image_xy", "pipette_defocus_um", "deep_learning"}:
                detector = unit.pipetteCalHelper.pipetteDetector
                options = {}
                if "pipette_expected_xy" in evidence["source_positions"]:
                    options = dict(expected_tip_xy=evidence["source_positions"]["pipette_expected_xy"],
                        pixels_to_um=stage.pixels_to_um_relative,
                        radius_um=(unit.config.pipette_detection_radius_um if pipette_radius_um is None
                                   else pipette_radius_um))
                result = detector.detect_pipette_details(image.copy(), **options)
                evidence["pipette_position"] = result["tip_xy"]
                evidence["confidence"]["pipette_detector"] = result["box_confidence"]
                evidence["status"]["pipette_detector"] = "ok" if result["tip_xy"] is not None else "no_detection"
                if visual & {"pipette_defocus_um", "deep_learning"}:
                    evidence["pipette_focus"] = result["z_um"]
                    evidence["confidence"]["pipette_focuser"] = result["z_confidence"]
                    evidence["status"]["pipette_focuser"] = "ok" if result["z_um"] is not None else "no_detection"
                    evidence["pipette_focus_confidence_supported"] = detector.adapter.model.z_confidence_head is not None
            if visual & {"cell_detections", "deep_learning"}:
                cells = stage.cellDetectHelper._ensure_detector().detect_cells(image.copy())
                evidence["cell_detections"] = cells
                evidence["status"]["cell_detector"] = "ok" if cells is not None else "no_detection"
        acquired = timing.get("retrieval_started_at")
        if acquired is None:
            acquired = timing.get("available_at")
        age = None if acquired is None else time.monotonic() - acquired
        return dict(evidence, age_s=age, stale=age is None or not 0 <= age <= 1.)

    def annotate(self, phase, observation, *, fields=None, calculations=None):
        """Record independent copies of derived fields in the observation and deck."""
        row = self.deck[observation["observation_id"]]
        updates = dict(fields or {})
        if calculations is not None:
            updates["calculations"] = dict(row.get("calculations", {}), **calculations)
        row.update(deepcopy(updates))
        observation.update(deepcopy(updates))
