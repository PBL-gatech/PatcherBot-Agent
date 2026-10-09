"""Collect requested measurements and retain one independent, image-free record."""
from copy import deepcopy
import time
from numbers import Integral, Real
from patcherbot.deepLearning.pipetteFocuser import PipetteFocuser2
import numpy as np

class Observer:
    _visual_fields = {"pipette_image_xy": "pipette_position", "pipette_defocus_um": "pipette_focus",
        "cell_detections": "cell_detections", "deep_learning": None}

    default_fields = ("manipulator_position", "pipette_image_xy", "pipette_defocus_um", "stage_positions",
                      "camera_image", "resistance", "deep_learning")
    available_fields = frozenset(default_fields + ("pressure",
        "commanded_pressure_mbar", "pressure_atm_state", "access_resistance", "capacitance", "cell_detections"))
    def __init__(self, controller):
        self.controller = controller
        self.deck, self._rows, self._next_id = [], {}, 0
        self.reset_attempt(None)

    def reset_attempt(self, attempt_id):
        self.deck.clear()
        self._rows.clear()
        self.attempt_id, self.attempt_token = attempt_id, object()

    def observe(self, *, fields=None, num_measurements=None, interval=None, phase=None, evidence=None,
                raw_resistance=False):
        if isinstance(fields, str):
            raise ValueError("Unknown observation fields")
        requested = tuple(dict.fromkeys(self.default_fields if fields is None else fields))
        if set(requested) - self.available_fields:
            raise ValueError("Unknown observation fields")
        if num_measurements is not None and (isinstance(num_measurements, bool)
                or not isinstance(num_measurements, Integral) or num_measurements < 1):
            raise ValueError("num_measurements must be a positive integer")
        if interval is not None and (isinstance(interval, bool) or not isinstance(interval, Real)
                or not np.isfinite(interval) or interval < 0):
            raise ValueError("interval must be finite and non-negative")
        if evidence is None:
            evidence = self.infer_frame(requested, frame_context=getattr(phase, "frame_context", None))
        evidence = dict(evidence)
        self._update_frame_age(evidence)
        unit, stage = self.controller.calibrated_unit, self.controller.calibrated_stage
        readers = {"manipulator_position": unit.position,
            "stage_positions": lambda: np.append(stage.position()[:2],
                unit.microscope.position() / unit.config.microscope_units_per_um),
            "commanded_pressure_mbar": self.controller.pressure.get_pressure,
            "pressure_atm_state": self.controller.pressure.get_ATM}
        electrical = {"resistance": (self.controller.resistanceRamp, 1, .001),
            "access_resistance": (self.controller.accessRamp, 5, .2), "capacitance": (self.controller.capacitanceRamp, 5, .2)}
        image = evidence.pop("_source_image", None)
        values, metadata = {}, {}
        for name in requested:
            started = time.monotonic()
            detail = dict(source="hardware", acquired_at=None)
            raw_reading = raw_resistance and name == "resistance"
            if raw_reading:
                value = self.controller.daq.resistance()
                detail.update(started_at=started, num_measurements=1, interval_s=0.0)
            elif name in electrical:
                read, count, delay = electrical[name]
                count = count if num_measurements is None else num_measurements
                delay = delay if interval is None else interval
                value = read(num_measurements=count, interval=delay)
                detail.update(started_at=started, num_measurements=count, interval_s=delay)
            elif name == "pressure":
                value, acquired = self.controller.pressure.get_last_acquisition_sample()
                detail.update(source="pressure_acquisition_cache", acquired_at=acquired, read_at=time.monotonic())
            elif name in readers:
                value = readers[name]()
            else:
                if name == "camera_image":
                    value = None if image is None else image.copy()
                else:
                    value = evidence if name == "deep_learning" else evidence.get(self._visual_fields[name])
                detail.update(source="inference", acquired_at=evidence.get("frame_acquired_at"),
                    available_at=evidence.get("frame_available_at"), source_frame=evidence.get("source_frame"),
                    retrieval_started_at=evidence.get("frame_retrieval_started_at"))
            if detail["source"] == "hardware":
                detail["read_completed_at"] = time.monotonic()
                if name in electrical:
                    detail["acquired_at"] = detail["read_completed_at"]
            if not raw_reading:
                if value is None and name not in ("camera_image", "deep_learning", "cell_detections"):
                    value = np.full(2, np.nan) if name == "pipette_image_xy" else np.nan
                if name in electrical or name in ("pressure", "commanded_pressure_mbar", "pressure_atm_state", "pipette_defocus_um"):
                    value = float(value)
            detail["valid"] = bool(evidence.get("source_image_shape")) if name == "deep_learning" else value is not None
            if detail["valid"] and name not in ("camera_image", "deep_learning"):
                try:
                    detail["valid"] = bool(np.isfinite(value).all())
                except (TypeError, ValueError):
                    if not raw_reading:
                        raise
                    detail["valid"] = False
            values[name], metadata[name] = value, detail
        values.update(observed_at=time.monotonic(), field_metadata=metadata, observation_id=self._next_id,
            attempt_id=self.attempt_id, phase=None if phase is None else type(phase).__name__,
            phase_run=None if phase is None else phase.observation_run)
        snapshot = deepcopy({key: value for key, value in values.items() if key != "camera_image"})
        self.deck.append(snapshot)
        self._rows[self._next_id] = phase, snapshot, frozenset(snapshot)
        self._next_id += 1
        return values

    def _update_frame_age(self, evidence):
        at = evidence.get("frame_retrieval_started_at")
        if at is None:
            at = evidence.get("frame_available_at")
        age = None if at is None else time.monotonic() - at
        evidence.update(age_s=age, stale=age is None or not 0 <= age <= 1.)

    def infer_frame(self, fields, frame_context=None, *, confirmation=None):
        """Evaluate requested existing models on one frame; Hunt may schedule this call."""
        if confirmation is not None:
            unit, stage = self.controller.calibrated_unit, self.controller.calibrated_stage
            helper = stage.focusHelper
            sweep = dict(pipette=True, best_xy=None, depth_advisory=False,
                pixels_to_um_relative=stage.pixels_to_um_relative,
                info=self.controller.info, warning=self.controller.warning, **confirmation)
            evidence = {"confirmed_tip_xy": None}
            helper.start_confirmation(sweep)
            try:
                for _ in range(helper.CONFIRM_MAX_FRAMES):
                    unit.abort_if_requested()
                    evidence = self.infer_frame(fields, frame_context)
                    self._update_frame_age(evidence)
                    if helper.update_confirmation(sweep, evidence):
                        evidence["confirmed_tip_xy"] = sweep["confirmation"]["confirmed_tip_xy"]
                        return evidence
                    unit.sleep(0.05)
                self.controller.warning('Pipette reference not refreshed: no 3 stable confident detections; keeping calibrated marker')
            except Exception as error:
                unit.abort_if_requested()
                self.controller.warning(f'Pipette reference display refresh failed: {error}; keeping calibrated marker')
            return dict(evidence, confirmed_tip_xy=None)
        requested = set(fields)
        evidence = {}
        visual = requested & self._visual_fields.keys()
        if not visual and "camera_image" not in requested:
            return evidence
        stage, unit = self.controller.calibrated_stage, self.controller.calibrated_unit
        camera = stage.camera
        frame_id, _, frame, timing = camera.last_raw_frame_data(include_timing=True) or (None, None, None, {})
        evidence = dict(source_frame=frame_id, frame_acquired_at=timing.get("acquired_at"),
            frame_retrieval_started_at=timing.get("retrieval_started_at"),
            frame_available_at=timing.get("available_at"), source_image_shape=None if frame is None else frame.shape,
            _source_image=frame, source_positions={}, source_position_metadata={}, positions_sampled_at=None,
            pipette_position=None, pipette_focus=None, cell_detections=None, status={}, confidence={}, errors={},
            pipette_focus_confidence_supported=None)
        if visual and frame is not None:
            image = np.asarray(frame).copy()
            evidence["_source_image"] = image
            context = frame_context or {}
            if context:
                evidence["positions_sampled_at"] = time.monotonic()
            for name, read in context.items():
                evidence["source_positions"][name] = read()
                evidence["source_position_metadata"][name] = dict(acquired_at=None,
                    read_completed_at=time.monotonic(), timestamp_basis="inference_time_context")
            readers = {}
            if visual & {"pipette_image_xy", "pipette_defocus_um", "deep_learning"}:
                detector = unit.pipetteCalHelper.pipetteDetector
                focus_requested = bool(visual & {"pipette_defocus_um", "deep_learning"})
                focuser = unit.pipetteFocusHelper.pipetteFocuser if focus_requested else None
                shared_depth = (focuser is not None and getattr(focuser, "detector", None) is detector
                    and getattr(focuser.get_pipette_focus_value, "__func__", None)
                    is PipetteFocuser2.get_pipette_focus_value)
                model = getattr(getattr(detector, "adapter", None), "model", None)
                if shared_depth and hasattr(model, "z_confidence_head"):
                    evidence["pipette_focus_confidence_supported"] = model.z_confidence_head is not None
                if visual & {"pipette_image_xy", "deep_learning"} or shared_depth:
                    readers["pipette_detector"] = detector.detect_pipette_details
                if focus_requested and not shared_depth:
                    readers["pipette_focuser"] = focuser.get_pipette_focus_value
                    evidence["confidence"]["pipette_focuser"] = None
            if visual & {"cell_detections", "deep_learning"}:
                readers["cell_detector"] = stage.cellDetectHelper._ensure_detector
            for name, read in readers.items():
                try:
                    if name == "cell_detector":
                        read = read().detect_cells
                    if name == "pipette_detector" and "pipette_expected_xy" in evidence["source_positions"]:
                        result = read(image.copy(),
                            expected_tip_xy=evidence["source_positions"]["pipette_expected_xy"],
                            pixels_to_um=stage.pixels_to_um_relative,
                            radius_um=unit.config.pipette_detection_radius_um)
                    else:
                        result = read(image.copy())
                    if name == "pipette_detector":
                        result = result or {}
                        evidence["pipette_position"] = result.get("tip_xy")
                        evidence["confidence"]["pipette_detector"] = result.get("box_confidence")
                        evidence["pipette_box_xyxy"] = result.get("box_xyxy")
                        evidence["pipette_candidate_counts"] = {key: result.get(key) for key in
                            ("candidate_count", "accepted_candidate_count", "radius_rejected_count")}
                        if shared_depth:
                            evidence["pipette_focus"] = result.get("z_um")
                            evidence["confidence"]["pipette_focuser"] = result.get("z_confidence")
                            evidence["status"]["pipette_focuser"] = "ok" if result.get("z_um") is not None else "no_detection"
                        result = result.get("tip_xy")
                    elif name == "pipette_focuser":
                        evidence["pipette_focus"] = result
                    else:
                        evidence["cell_detections"] = result
                    evidence["status"][name] = "ok" if result is not None else "no_detection"
                except Exception as error:
                    evidence["status"][name], evidence["errors"][name] = "error", str(error)
                    if name == "pipette_focuser":
                        evidence["pipette_focus"] = None
                    elif name == "pipette_detector" and shared_depth:
                        evidence["status"]["pipette_focuser"] = "error"
                        evidence["errors"]["pipette_focuser"] = str(error)
                        evidence["confidence"]["pipette_focuser"] = None
        return evidence


    def annotate(self, phase, observation, *, fields=None, calculations=None, markers=None, source_positions=None):
        saved = self._rows.get(observation.get("observation_id"))
        if saved is None or saved[0] is not phase:
            raise ValueError("Observation does not belong to this phase and attempt")
        _, row, protected = saved
        if any(row[key] != observation.get(key) for key in ("attempt_id", "phase", "phase_run")):
            raise ValueError("Observation identity has changed")
        updates = dict(fields or {})
        if updates.keys() & (protected | {"camera_image", "calculations"}):
            raise ValueError("Cannot replace acquired measurements or observation identity")
        if calculations is not None:
            updates["calculations"] = dict(row.get("calculations", {}), **deepcopy(calculations))
        row.update(deepcopy(updates))
        observation.update(deepcopy(updates))

        display = getattr(self.controller, "alignment_display", None)
        if markers and display is not None:
            source = source_positions or {}
            for name, point in markers.items():
                try:
                    marker = dict(xy=np.asarray(point, dtype=float).copy(),
                        stage_xy=np.asarray(source["stage_xy"] if "stage_xy" in source else self.controller.calibrated_stage.position()).copy())
                    if name == "pipette":
                        marker["pipette_xyz"] = np.asarray(source["pipette_xyz"] if "pipette_xyz" in source else self.controller.calibrated_unit.position()).copy()
                    display = dict(display, **{name: marker})
                except Exception as error:
                    if not display.get("publication_warned"):
                        display = dict(display, publication_warned=True)
                        self.controller.warning(f"Could not display {name} detection marker: {error}. {phase.phase_name} continues.",
                                                extra={"detection_overlay": True})
            self.controller.alignment_display = display
