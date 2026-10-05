"""Collect requested measurements and retain one independent, image-free record."""
from copy import deepcopy
import time
import numpy as np

class Observer:
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
        unit, stage = self.controller.calibrated_unit, self.controller.calibrated_stage
        camera = stage.camera
        requested = tuple(dict.fromkeys(self.default_fields if fields is None else fields))
        readers = {"manipulator_position": unit.position,
            "stage_positions": lambda: np.append(stage.position()[:2],
                unit.microscope.position() / unit.config.microscope_units_per_um),
            "commanded_pressure_mbar": self.controller.pressure.get_pressure,
            "pressure_atm_state": self.controller.pressure.get_ATM}
        electrical = {"resistance": (self.controller.resistanceRamp, 1, .001),
            "access_resistance": (self.controller.accessRamp, 5, .2), "capacitance": (self.controller.capacitanceRamp, 5, .2)}
        visual = {"pipette_image_xy": "pipette_position", "pipette_defocus_um": "pipette_focus",
            "cell_detections": "cell_detections", "deep_learning": None, "camera_image": "_source_image"}
        evidence = dict(evidence or {})
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
                value = evidence if name == "deep_learning" else evidence.get(visual[name])
                detail.update(source="inference", acquired_at=evidence.get("frame_acquired_at"),
                    available_at=evidence.get("frame_available_at"), source_frame=evidence.get("source_frame"),
                    retrieval_started_at=evidence.get("frame_retrieval_started_at"))
                if name == "camera_image":
                    value = image
                    if value is None:
                        frame_id, stamp, value, timing = camera.last_raw_frame_data(include_timing=True) or (None, None, None, {})
                        detail.update(timing, source="camera", source_frame=frame_id)
                    value = None if value is None else value.copy()
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
                if raw_reading:
                    try:
                        detail["valid"] = bool(np.isfinite(value).all())
                    except (TypeError, ValueError):
                        detail["valid"] = False
                else:
                    detail["valid"] = bool(np.isfinite(value).all())
            values[name], metadata[name] = value, detail
        values.update(observed_at=time.monotonic(), field_metadata=metadata, observation_id=self._next_id,
            attempt_id=self.attempt_id, phase=None if phase is None else type(phase).__name__,
            phase_run=None if phase is None else phase.observation_run)
        snapshot = deepcopy({key: value for key, value in values.items() if key != "camera_image"})
        self.deck.append(snapshot)
        self._rows[self._next_id] = phase, snapshot, frozenset(snapshot)
        self._next_id += 1
        return values

    def annotate(self, phase, observation, *, fields=None, calculations=None):
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
