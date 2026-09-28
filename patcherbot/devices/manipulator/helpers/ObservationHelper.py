"""Collect observations and own model caches and phase-scoped history."""

import time
from collections import deque
from copy import deepcopy
from numbers import Integral, Real
from threading import Event, Lock, RLock, Thread, current_thread

import numpy as np

try:
    from .ObservationTracking import ObservationTracking
except ImportError:  # Standalone hardware-free loading used by regression tests.
    import importlib.util
    from pathlib import Path
    _tracking_spec = importlib.util.spec_from_file_location(
        "observation_tracking", Path(__file__).with_name("ObservationTracking.py"))
    _tracking_module = importlib.util.module_from_spec(_tracking_spec)
    _tracking_spec.loader.exec_module(_tracking_module)
    ObservationTracking = _tracking_module.ObservationTracking


class ObservationHelper:
    """Keep acquisition, cached inference, and observation history in one place."""

    def __init__(self, controller):
        self.controller = controller
        self._deep_learning_observation = None
        self._deep_learning_lock = Lock()
        self.deep_learning_max_age = 1.0
        self._producer_lock = Lock()
        self._producer_stop = None
        self._producer_thread = None
        self._producer_generation = 0
        self._producer_session = 0
        self._tracking_session = None
        self._tracking_cell = None
        self._producer_cell = None
        self._cell_models_enabled = True
        self._pipette_models_enabled = False
        self.deep_learning_error = None
        self.observation_decks = {}
        self._history_options = {}
        self._recorded_observations = {}
        self._latest = {}
        self._tracking_lock = RLock()
        self._tracking = ObservationTracking(controller, lambda: time.monotonic())
        self.tracking_settings = self._tracking.pipette_tracking_settings
        self.latest_tracking_observation = None
        self._last_tracking_input = None
        self._selected_tracking_target = None

    @property
    def last_cell_location(self):
        return self._tracking.last_cell_location

    def enrich_tracking(self, observation, target_cell=None):
        """Attach estimates once per observation; raw measurements stay untouched."""
        with self._tracking_lock:
            if target_cell is not self._selected_tracking_target:
                self._tracking._cell_track = None
                self._tracking.last_cell_location = None
                self._selected_tracking_target = target_cell
            calibrations = []
            for name in ("cell", "pipette"):
                try:
                    calibrations.append(self._tracking._calibration_identity(name))
                except (AttributeError, TypeError, ValueError):
                    calibrations.append(None)
            identity = (id(target_cell), None if target_cell is None else
                        tuple(np.asarray(target_cell[0]).reshape(-1)), tuple(calibrations))
            if (self._last_tracking_input is not None and
                    self._last_tracking_input[0] is observation and
                    self._last_tracking_input[1] == identity):
                return observation
            result = self._tracking.enrich(observation, target_cell)
            self._last_tracking_input = (observation, identity)
            if isinstance(result, dict):
                self.latest_tracking_observation = deepcopy(result)
            return result

    def tracking_for_frame(self, frame_id, acquired_at, image_shape, camera=None):
        """Project cached tracking onto a displayed frame; never acquire hardware."""
        with self._tracking_lock:
            result = self._tracking.for_frame(frame_id, acquired_at, image_shape, camera)
            cached = self.latest_tracking_observation or {}
            evidence = cached.get("deep_learning") or {}
            try:
                age = time.monotonic() - float(acquired_at)
                matched = (frame_id is not None and evidence.get("source_frame") == frame_id and
                           evidence.get("frame_acquired_at") == acquired_at and
                           tuple(image_shape[:2]) == tuple(evidence.get("source_image_shape", ())[:2]) and
                           0 <= age <= self.deep_learning_max_age and
                           (camera is None or camera is self.controller.calibrated_stage.camera))
            except (TypeError, ValueError):
                matched = False
            if matched:
                if result is None:
                    result = dict(observed_at=time.monotonic(), field_metadata={"camera_image": dict(
                        source_frame=frame_id, acquired_at=acquired_at, valid=True, stale=False)},
                        pipette_tracking=dict(valid=False, display_valid=False),
                        cell_tracking=dict(valid=False, display_valid=False))
                for field in ("deep_learning", "target_cell_id", "target_cell_image_xy",
                              "target_cell_valid", "visual_context_valid"):
                    if field in cached:
                        result[field] = deepcopy(cached[field])
                result["deep_learning"].update(age_s=age, stale=False)
            return result

    def invalidate_tracking(self, reason, *, pipette=True, cell=False):
        """Notify replacement or encoder re-zeroing without changing calibration."""
        with self._tracking_lock:
            self._tracking.invalidate(reason, pipette=pipette, cell=cell)
            self._last_tracking_input = None
            self.latest_tracking_observation = None

    def fresh_visual(self, observation, previous_frame=None, after=None):
        return self._tracking._fresh_visual(observation, previous_frame, after)

    def usable_tip(self, observation):
        return self._tracking._usable_tip(observation)

    def reset_history(self, phase, maxlen, *, include_images=True):
        """Reset attempt history; None retains all rows, optionally without images."""
        deck = getattr(phase, "observation_deck", None)
        if not isinstance(deck, deque) or deck.maxlen != maxlen:
            deck = deque(maxlen=maxlen)
        else:
            deck.clear()
        self.observation_decks[phase] = deck
        if hasattr(phase, "observation_deck"):
            phase.observation_deck = deck
        self._history_options[phase] = {"include_images": bool(include_images), "fields": set()}
        self._recorded_observations.pop(phase, None)

    def record_observation(self, phase, observation):
        """Store independent measurements; image retention is selected per attempt."""
        deck = self.observation_decks.get(phase)
        if deck is None:
            return
        options = self._history_options.setdefault(phase, {"include_images": True, "fields": set()})
        source = observation
        if not options["include_images"]:
            if isinstance(observation, dict):
                source = dict(observation)
                if "camera_image" in source:
                    source["camera_image"] = None
            elif isinstance(observation, list) and len(observation) >= 3:
                source = list(observation)
                source[2] = None
        snapshot = deepcopy(source)
        if isinstance(snapshot, dict):
            if not options["include_images"] and "camera_image" in snapshot:
                metadata = snapshot.setdefault("field_metadata", {})
                metadata.setdefault("camera_image", {})["history_omitted"] = True
            # Only new field names require revisiting earlier rows. Ordinary
            # recording remains bounded in cost even for whole-attempt history.
            new_fields = set(snapshot) - options["fields"]
            if new_fields:
                for entry in deck:
                    if isinstance(entry, dict):
                        for field in new_fields:
                            entry.setdefault(field, float("nan"))
                options["fields"].update(new_fields)
            for field in options["fields"]:
                snapshot.setdefault(field, float("nan"))
        deck.append(snapshot)
        self._recorded_observations[phase] = (observation, snapshot)

    def record_calculations(self, phase, observation, calculations):
        """Enrich this exact recorded sample without adding a second deck row."""
        recorded = self._recorded_observations.get(phase)
        deck = self.observation_decks.get(phase)
        if (recorded is None or recorded[0] is not observation or not deck
                or deck[-1] is not recorded[1]):
            raise ValueError("Calculations must belong to the latest recorded observation")
        snapshot = recorded[1]
        if not isinstance(observation, dict) or not isinstance(snapshot, dict):
            raise TypeError("Calculation storage requires a dictionary observation")
        # Keep both the returned observation and helper-owned history independent.
        existing = observation.get("calculations")
        merged = deepcopy(existing) if isinstance(existing, dict) else {}
        merged.update(deepcopy(calculations))
        observation["calculations"] = deepcopy(merged)
        snapshot["calculations"] = deepcopy(merged)
        options = self._history_options.setdefault(phase, {"include_images": True, "fields": set()})
        if "calculations" not in options["fields"]:
            for entry in deck:
                if isinstance(entry, dict):
                    entry.setdefault("calculations", float("nan"))
            options["fields"].add("calculations")

    def observation_window(
        self, phase, field, width, *, predicate=None, finite_only=False,
        fill_value=np.nan, dtype=np.float32, shape=(), include_current=False,
    ):
        """Read a configured numeric field window from this phase's deck.

        Values are oldest first, left-padded, and exclude the current snapshot
        unless requested. Scalar and explicitly shaped vector fields use the
        same reader. Selection and formatting never mutate stored observations.
        """
        width, shape = int(width), tuple(shape)
        result = np.full((width, *shape), fill_value, dtype=dtype)
        if width == 0:
            return result
        samples = list(self.observation_decks.get(phase) or ())
        if not include_current:
            samples = samples[:-1]
        values = []
        for sample in samples:
            if not isinstance(sample, dict) or (predicate is not None and not predicate(sample)):
                continue
            try:
                value = sample
                for key in field.split("."):
                    value = value.get(key, np.nan) if isinstance(value, dict) else np.nan
                value = np.asarray(value, dtype=dtype).reshape(shape)
            except (TypeError, ValueError):
                value = np.full(shape, np.nan, dtype=dtype)
            if finite_only and not np.isfinite(value).all():
                continue
            values.append(value)
        values = values[-width:]
        if values:
            result[-len(values):] = np.asarray(values, dtype=dtype)
        return result

    def _cache_reading(self, field, value):
        self._latest[field] = dict(value=deepcopy(value), acquired_at=time.monotonic(),
                                   confidence=None, source="hardware",
                                   timestamp_basis="hardware_read_complete")
        return value

    def get_cached_observation(self, field, *, max_age=1.0):
        """Read a hardware cache entry without sampling or adding history."""
        entry = deepcopy(self._latest.get(field))
        if entry is None:
            return None
        age = None if entry["acquired_at"] is None else time.monotonic() - entry["acquired_at"]
        entry.update(age_s=age, stale=age is None or age < 0 or age > max_age)
        return entry

    def _camera_frame(self):
        queue = self.controller.calibrated_stage.camera.raw_frame_queue
        if not queue:
            return None, None, None
        frame_id, frame_time, _, frame = queue[-1]
        return frame_id, frame_time, None if frame is None else frame.copy()

    def get_camera_image(self, *, _frame=None):
        frame_id, frame_time, image = self._camera_frame() if _frame is None else _frame
        self._cache_reading("camera_image", image)
        stamp = frame_time.timestamp() if hasattr(frame_time, "timestamp") else frame_time
        available_at = None if stamp is None else float(stamp)
        if available_at is not None:
            if not np.isfinite(available_at):
                available_at = None
            elif available_at > 100000000:
                available_at = time.monotonic() - (time.time() - available_at)
        camera = self.controller.calibrated_stage.camera
        timing_reader = getattr(camera, "get_frame_timing", None)
        timing = timing_reader(frame_id, frame_time) if callable(timing_reader) else None
        acquired_at = None if timing is None else timing.get("acquisition_started_at")
        if timing is not None:
            available_at = timing.get("available_at", available_at)
        self._latest["camera_image"].update(
            acquired_at=acquired_at if acquired_at is not None else available_at,
            frame_acquired_at=acquired_at, frame_available_at=available_at,
            source_frame=frame_id, source="camera",
            timestamp_basis="camera_snap_start" if acquired_at is not None else "camera_availability")
        return image

    def get_manipulator_position(self):
        """Hardware manipulator coordinates; never substitute image pixels."""
        return self._cache_reading("manipulator_position", self.controller.calibrated_unit.position())

    def get_stage_positions(self):
        stage = self.controller.calibrated_stage.position()[:2]
        unit = self.controller.calibrated_unit
        z = unit.microscope.position() / unit.config.microscope_units_per_um
        return self._cache_reading("stage_positions", np.append(stage, z))

    def get_motion_state(self):
        """Sample each physical device once; never wait, retry, or decide completion."""
        snapshots, sampled = {}, {}
        for name, device in (("manipulator", self.controller.calibrated_unit),
                             ("stage", self.controller.calibrated_stage),
                             ("microscope", self.controller.microscope)):
            backend = getattr(device, 'dev', device)
            key = id(backend)
            if key not in sampled:
                sampled[key] = dict(device.read_motion_state(), sampled_at=time.monotonic())
            snapshots[name] = dict(sampled[key])
        return self._cache_reading("motion_state", snapshots)

    @staticmethod
    def _validate_sampling(num_measurements, interval):
        if isinstance(num_measurements, bool) or not isinstance(num_measurements, Integral) or num_measurements < 1:
            raise ValueError("num_measurements must be a positive integer")
        if isinstance(interval, bool) or not isinstance(interval, Real) or not np.isfinite(interval) or interval < 0:
            raise ValueError("interval must be a finite non-negative number")
        return int(num_measurements), float(interval)

    def get_resistance(self, num_measurements=1, interval=0.001):
        num_measurements, interval = self._validate_sampling(num_measurements, interval)
        resistance = self.controller.resistanceRamp(num_measurements=num_measurements, interval=interval)
        return self._cache_reading("resistance", resistance)

    def get_access_resistance(self, num_measurements=5, interval=0.200):
        num_measurements, interval = self._validate_sampling(num_measurements, interval)
        access_resistance = self.controller.accessRamp(num_measurements=num_measurements, interval=interval)
        return self._cache_reading("access_resistance", access_resistance)

    def get_capacitance(self, num_measurements=5, interval=0.200):
        num_measurements, interval = self._validate_sampling(num_measurements, interval)
        capacitance = self.controller.capacitanceRamp(num_measurements=num_measurements, interval=interval)
        return self._cache_reading("capacitance", capacitance)

    def get_pressure(self):
        pressure = self.controller.pressure
        sample_reader = getattr(pressure, "get_last_acquisition_sample", None)
        acquired_at = None
        if callable(sample_reader):
            value, acquired_at = sample_reader()
        else:
            read = getattr(pressure, "get_last_acquisition", None)
            value = read() if callable(read) else np.nan
        reading = self._cache_reading("pressure", np.asarray([value], dtype=np.float32))
        metadata = self._latest["pressure"]
        metadata.update(read_at=metadata["acquired_at"], acquired_at=acquired_at,
                        source="pressure_acquisition_cache",
                        timestamp_basis="pressure_acquisition" if acquired_at is not None else "cache_read_complete",
                        measurement_acquired_at=acquired_at, freshness_known=acquired_at is not None)
        return reading

    def get_commanded_pressure_mbar(self):
        read = getattr(self.controller.pressure, "get_pressure", None)
        value = read() if callable(read) else np.nan
        return self._cache_reading("commanded_pressure_mbar", np.asarray([value], dtype=np.float32))

    def get_pressure_atm_state(self):
        read = getattr(self.controller.pressure, "get_ATM", None)
        raw = read() if callable(read) else None
        value = np.nan if raw is None or not np.isfinite(float(raw)) else float(bool(raw))
        return self._cache_reading("pressure_atm_state", np.asarray([value], dtype=np.float32))

    def start_deep_learning(self, cell=None, interval=0.1):
        """Start one independent inference loop; ordinary getters only read its cache.

        The interval is a minimum delay between batches. Slow inference never
        queues another batch. Each camera frame is processed at most once for
        the current model selection and target.
        """
        if isinstance(interval, bool) or not np.isfinite(interval) or interval < 0.1:
            raise ValueError("deep-learning interval must be at least 0.1 seconds")
        self.stop_deep_learning()
        with self._producer_lock:
            self._producer_cell = cell
            self._cell_models_enabled = True
            self._pipette_models_enabled = False
            self.deep_learning_error = None
            stop = Event()
            self._producer_stop = stop
            worker = Thread(target=self._deep_learning_loop, args=(stop, float(interval)),
                            name="observation-inference", daemon=True)
            self._producer_thread = worker
        worker.start()

    def set_deep_learning_models(self, *, cell=True, pipette=False):
        """Select observations for the active action, invalidating older batches."""
        if not isinstance(cell, bool) or not isinstance(pipette, bool):
            raise TypeError("model selections must be booleans")
        with self._producer_lock:
            if (cell == self._cell_models_enabled and
                    pipette == self._pipette_models_enabled):
                return
            self._cell_models_enabled = cell
            self._pipette_models_enabled = pipette
            self._producer_generation += 1
            self._deep_learning_observation = None

    def stop_deep_learning(self):
        """Stop scheduling and invalidate results, including any batch still running."""
        with self._producer_lock:
            stop, worker = self._producer_stop, self._producer_thread
            self._producer_stop = None
            self._producer_thread = None
            self._producer_generation += 1
            self._producer_session += 1
            self._deep_learning_observation = None
            if stop is not None:
                stop.set()
        # Inference cannot be interrupted safely. A late batch is discarded by
        # its generation check, and the inference lock prevents overlapping work.
        if worker is not None and worker is not current_thread():
            worker.join(timeout=1.0)

    def _deep_learning_loop(self, stop, interval):
        previous_frame = None
        while not stop.is_set():
            with self._producer_lock:
                generation = self._producer_generation
                cell = self._producer_cell
                cell_models = self._cell_models_enabled
                pipette_models = self._pipette_models_enabled
            try:
                frame = self._camera_frame()
                frame_key = (generation, frame[0], frame[1])
                if frame_key != previous_frame and (cell_models or pipette_models):
                    self.refresh_deep_learning(
                        cell=cell, _frame=frame, cell_models=cell_models,
                        pipette_models=pipette_models, _generation=generation)
                    previous_frame = frame_key
                    self.deep_learning_error = None
            except Exception as error:
                # Surface producer failures without terminating phase observation.
                self.deep_learning_error = str(error)
            stop.wait(interval)

    def get_deep_learning(self, *, deep_learning=False, cell=None, _include_image=False):
        """False reads the cache; True explicitly refreshes the shared model batch.

        A separate caller loop can call refresh_deep_learning. Cache reads never
        wait for inference or insert repeated model results into measurement history.
        """
        if not isinstance(deep_learning, bool):
            raise TypeError("deep_learning must be a boolean")
        if deep_learning:
            self.refresh_deep_learning(cell=cell)
        cached = self._deep_learning_observation
        accepted = cached is not None and (cell is None or cached[0] is cell)
        evidence = deepcopy({key: value for key, value in cached[1].items()
                             if key != "_source_image"}) if accepted else None
        source_image = cached[1].get("_source_image") if accepted else None
        if evidence is None:
            evidence = dict(source_frame=None, frame_available_at=None, frame_acquired_at=None, confidence=None,
                            status="not_available", pipette_position=None, pipette_focus=None,
                            cell_detections=None, tracked_cell_position=None)
        now = time.monotonic()
        available_at = evidence["frame_available_at"]
        acquired_at = evidence.get("frame_acquired_at")
        freshness_at = acquired_at if acquired_at is not None else available_at
        age = None if freshness_at is None else now - freshness_at
        evidence.update(observed_at=now, age_s=age,
                        stale=age is None or age < 0 or age > self.deep_learning_max_age)
        if _include_image:
            evidence["_source_image"] = None if source_image is None else source_image.copy()
        return evidence

    def get_pipette_positions(self, *, deep_learning=False, cell=None):
        """Image XY and model focus; unavailable results remain NaN."""
        evidence = self.get_deep_learning(deep_learning=deep_learning, cell=cell)
        position, focus = evidence["pipette_position"], evidence["pipette_focus"]
        return np.append([np.nan, np.nan] if position is None else position,
                         np.nan if focus is None else focus)

    def get_pipette_focus(self, *, deep_learning=False, cell=None):
        return self.get_deep_learning(deep_learning=deep_learning, cell=cell)["pipette_focus"]

    def get_cell_detections(self, *, deep_learning=False, cell=None):
        return self.get_deep_learning(deep_learning=deep_learning, cell=cell)["cell_detections"]

    def get_tracked_cell_position(self, *, deep_learning=False, cell=None):
        return self.get_deep_learning(deep_learning=deep_learning, cell=cell)["tracked_cell_position"]

    def refresh_deep_learning(self, *, cell=None, _frame=None,
                              cell_models=True, pipette_models=True, _generation=None):
        """Publish selected models from one owned frame; direct calls default to all."""
        with self._producer_lock:
            generation = self._producer_generation if _generation is None else _generation
            session = self._producer_session
        # Serialize producers only. Cache readers never wait for inference.
        with self._deep_learning_lock:
            with self._producer_lock:
                if generation != self._producer_generation:
                    return
            frame_id, frame_time, frame = self._camera_frame() if _frame is None else _frame
            source_image = None if frame is None else frame.copy()
            started_at = time.monotonic()
            stamp = frame_time.timestamp() if hasattr(frame_time, "timestamp") else frame_time
            available_at = None
            if stamp is not None and np.isfinite(float(stamp)):
                available_at = float(stamp)
                if available_at > 100000000:
                    available_at = started_at - (time.time() - available_at)
            camera = self.controller.calibrated_stage.camera
            timing_reader = getattr(camera, "get_frame_timing", None)
            timing = timing_reader(frame_id, frame_time) if callable(timing_reader) else None
            acquired_at = None if timing is None else timing.get("acquisition_started_at")
            if timing is not None:
                available_at = timing.get("available_at", available_at)
            models = ("pipette_detector", "pipette_focuser", "cell_detector", "cell_tracker")
            result = dict(source_frame=frame_id, frame_available_at=available_at,
                          frame_acquired_at=acquired_at,
                          timestamp_basis="camera_snap_start" if acquired_at is not None else "camera_availability",
                          started_at=started_at,
                          producer_session=session, producer_generation=generation,
                          target_identity=None if cell is None else id(cell),
                          pipette_position=None, pipette_focus=None, cell_detections=None,
                          tracked_cell_position=None, tracking_status=None,
                          confidence={name: None for name in models},
                          status={name: "no_frame" for name in models}, errors={},
                          enabled_models={"cell": cell_models, "pipette": pipette_models},
                          source_image_shape=None if frame is None else tuple(frame.shape),
                          source_positions={}, source_position_metadata={}, positions_sampled_at=started_at,
                          position_timestamp_basis="inference_start")
            for name in models:
                enabled = cell_models if name.startswith("cell") else pipette_models
                if not enabled:
                    result["status"][name] = "disabled"
            # These hardware readings provide frame context, but are sampled at
            # inference start: the camera does not supply synchronized positions.
            unit, stage = self.controller.calibrated_unit, self.controller.calibrated_stage
            position_readers = {
                "manipulator": unit.position,
                "stage": stage.position,
                "microscope": lambda: unit.microscope.position() / unit.config.microscope_units_per_um,
            }
            for name, read in position_readers.items():
                try:
                    position = deepcopy(read())
                    result["source_positions"][name] = position
                    result["source_position_metadata"][name] = dict(
                        acquired_at=time.monotonic(), timestamp_basis="hardware_read_complete",
                        valid=bool(np.isfinite(np.asarray(position, dtype=float)).all()))
                except Exception as error:
                    result["source_positions"][name] = None
                    result["source_position_metadata"][name] = dict(
                        acquired_at=time.monotonic(), timestamp_basis="hardware_read_complete", valid=False)
                    result["errors"][name + "_position"] = str(error)

            def read_model(name, read):
                try:
                    value = read()
                    result["status"][name] = "ok" if value is not None else "no_detection"
                    return value
                except Exception as error:
                    result["status"][name] = "error"
                    result["errors"][name] = str(error)
                    return None

            if frame is not None:
                unit, stage = self.controller.calibrated_unit, self.controller.calibrated_stage
                if pipette_models:
                    detector = unit.pipetteCalHelper.pipetteDetector
                    focuser = unit.pipetteFocusHelper.pipetteFocuser
                    detailed = getattr(detector, "detect_pipette_details", None)
                    if callable(detailed):
                        details = read_model("pipette_detector", lambda: detailed(frame)) or {}
                        result["pipette_position"] = details.get("tip_xy")
                        result["confidence"]["pipette_detector"] = details.get("box_confidence")
                        if result["status"]["pipette_detector"] != "error":
                            result["status"]["pipette_detector"] = ("ok" if result["pipette_position"] is not None else "no_detection")
                        if getattr(focuser, "detector", None) is detector:
                            result["pipette_focus"] = details.get("z_um")
                            result["confidence"]["pipette_focuser"] = details.get("z_confidence")
                            if result["status"]["pipette_detector"] == "error":
                                result["status"]["pipette_focuser"] = "error"
                                result["errors"]["pipette_focuser"] = result["errors"]["pipette_detector"]
                            else:
                                result["status"]["pipette_focuser"] = ("ok" if result["pipette_focus"] is not None else "no_detection")
                        else:
                            result["pipette_focus"] = read_model("pipette_focuser", lambda: focuser.get_pipette_focus_value(frame))
                    else:
                        result["pipette_position"] = read_model("pipette_detector", lambda: detector.detect_pipette(frame))
                        result["pipette_focus"] = read_model("pipette_focuser", lambda: focuser.get_pipette_focus_value(frame))
                if cell_models:
                    # The model accepts a frame; the UI helper acquires its own.
                    result["cell_detections"] = read_model("cell_detector", lambda:
                        stage.cellDetectHelper._ensure_detector().detect_cells(frame))
                    detections = result["cell_detections"]
                    if detections is not None:
                        result["status"]["cell_detector"] = "ok" if len(detections) else "no_detection"
                        result["confidence"]["cell_detector"] = [float(item[2]) for item in detections]
                    if cell is None:
                        result["status"]["cell_tracker"] = "no_target"
                    else:
                        def track_selected_cell():
                            helper = stage._ensure_cell_track_helper()
                            if self._tracking_cell is not cell or self._tracking_session != session:
                                helper.reset_tracking()
                                self._tracking_cell = cell
                                self._tracking_session = session
                            coords, reference, _ = cell
                            point = helper.track_cell(
                                cell, frame,
                                expected_point=np.asarray(stage.reference_position())[:2] - np.asarray(coords)[:2],
                                prompt_point=np.array([reference.shape[1] / 2., reference.shape[0] / 2.]),
                                use_centroid=self.controller.config.use_centroid, mode=self.controller.config.tracking_mode,
                                max_fast_jump_px=self.controller.config.track_max_fast_jump_px)
                            result["tracking_status"] = deepcopy(helper.last_tracking_status)
                            return point
                        result["tracked_cell_position"] = read_model("cell_tracker", track_selected_cell)
            result["completed_at"] = time.monotonic()
            result["inference_latency_s"] = result["completed_at"] - started_at
            # Retain exactly the image used by this batch, privately. Public
            # evidence excludes it; observe exposes one independently owned image.
            published = deepcopy(result)
            published["_source_image"] = source_image
            if published["_source_image"] is not None:
                published["_source_image"].flags.writeable = False
            with self._producer_lock:
                if generation == self._producer_generation:
                    self._deep_learning_observation = (cell, published)

    def _validate_request(self, include_pressure_state, fields, sampling):
        legacy = fields is None
        if legacy:
            if sampling is not None:
                raise ValueError("sampling requires explicit observation fields")
            fields = ("pipette_positions", "stage_positions", "camera_image", "resistance")
            if include_pressure_state:
                fields += ("pressure", "commanded_pressure_mbar", "pressure_atm_state")

        from collections.abc import Mapping

        supported = {
            "pipette_positions", "stage_positions", "camera_image", "resistance",
            "pressure", "commanded_pressure_mbar", "pressure_atm_state",
            "access_resistance", "capacitance", "manipulator_position", "motion_state",
            "deep_learning", "cell_detections", "tracked_cell_position", "pipette_focus",
            "pipette_image_xy", "pipette_defocus_um", "pipette_tracking", "cell_tracking",
        }
        if isinstance(fields, str):
            raise TypeError("fields must be an iterable of names, not a string")
        requested = tuple(dict.fromkeys(fields))
        unknown = set(requested) - supported
        if unknown:
            raise ValueError(f"Unknown observation fields: {unknown}")

        defaults = {
            "resistance": (1, 0.001),
            "access_resistance": (5, 0.200),
            "capacitance": (5, 0.200),
        }
        if sampling is None:
            sampling = {}
        if not isinstance(sampling, Mapping):
            raise TypeError("sampling must map measurement fields to settings")
        sample_options = {}
        for field, options in sampling.items():
            if field not in defaults or field not in requested:
                raise ValueError(f"Sampling requires a requested measurement field: {field}")
            if not isinstance(options, Mapping):
                raise TypeError(f"Sampling settings for {field} must be a mapping")
            unknown_options = set(options) - {"num_measurements", "interval"}
            if unknown_options:
                raise ValueError(f"Unknown sampling settings for {field}: {unknown_options}")
            count, interval = defaults[field]
            count = options.get("num_measurements", count)
            interval = options.get("interval", interval)
            sample_options[field] = self._validate_sampling(count, interval)

        return legacy, requested, sample_options

    def observe(self, include_pressure_state: bool = False, *, fields=None, sampling=None,
                raw_measurements=False, deep_learning=False, cell=None, target_cell=None,
                num_measurements=None, interval=None):
        """Acquire selected measurements explicitly; numeric sampling applies to electrical readings.

        Omitted numeric arguments preserve each measurement's existing defaults.
        The older sampling mapping remains accepted at this compatibility boundary.
        """
        if not isinstance(deep_learning, bool):
            raise TypeError("deep_learning must be a boolean")
        if sampling is not None and (num_measurements is not None or interval is not None):
            raise ValueError("Use numeric sampling arguments or the sampling mapping, not both")
        legacy, requested, sample_options = self._validate_request(include_pressure_state, fields, sampling)
        resistance_count = 1 if num_measurements is None else num_measurements
        averaged_count = 5 if num_measurements is None else num_measurements
        resistance_interval = 0.001 if interval is None else interval
        averaged_interval = 0.200 if interval is None else interval
        resistance_count, resistance_interval = sample_options.get(
            "resistance", (resistance_count, resistance_interval))
        access_count, access_interval = sample_options.get(
            "access_resistance", (averaged_count, averaged_interval))
        capacitance_count, capacitance_interval = sample_options.get(
            "capacitance", (averaged_count, averaged_interval))
        # Validate every requested electrical setting before acquiring any data.
        if "resistance" in requested:
            self._validate_sampling(resistance_count, resistance_interval)
        if "access_resistance" in requested:
            self._validate_sampling(access_count, access_interval)
        if "capacitance" in requested:
            self._validate_sampling(capacitance_count, capacitance_interval)
        if not legacy and "pipette_positions" in requested:
            requested = tuple(dict.fromkeys((*requested, "pipette_image_xy", "pipette_defocus_um")))
        tracking_requested = bool({"pipette_tracking", "cell_tracking"}.intersection(requested))
        if tracking_requested:
            requested = tuple(dict.fromkeys((*[field for field in requested if field not in ("pipette_tracking", "cell_tracking")],
                                             "manipulator_position", "stage_positions", "deep_learning")))
        values, metadata = {}, {}
        if deep_learning:
            self.refresh_deep_learning(cell=cell, _frame=self._camera_frame())
        visual_fields = {"pipette_positions", "pipette_focus", "cell_detections",
                         "tracked_cell_position", "deep_learning", "pipette_image_xy", "pipette_defocus_um"}
        evidence = self.get_deep_learning(cell=cell, _include_image="camera_image" in requested) if visual_fields.intersection(requested) else None
        source_image = None if evidence is None else evidence.pop("_source_image", None)
        visual_models = {
            "pipette_positions": ("pipette_detector", "pipette_focuser"),
            "pipette_focus": ("pipette_focuser",), "cell_detections": ("cell_detector",),
            "pipette_image_xy": ("pipette_detector",), "pipette_defocus_um": ("pipette_focuser",),
            "tracked_cell_position": ("cell_tracker",),
        }
        for field in requested:
            if field == "resistance":
                value = self.get_resistance(resistance_count, resistance_interval)
            elif field == "access_resistance":
                value = self.get_access_resistance(access_count, access_interval)
            elif field == "capacitance":
                value = self.get_capacitance(capacitance_count, capacitance_interval)
            elif field == "manipulator_position":
                value = self.get_manipulator_position()
            elif field == "stage_positions":
                value = self.get_stage_positions()
            elif field == "motion_state":
                value = self.get_motion_state()
            elif field == "camera_image":
                if source_image is not None and not (legacy and not deep_learning):
                    value = source_image
                else:
                    value = self.get_camera_image()
            elif field == "pressure":
                value = self.get_pressure()
            elif field == "commanded_pressure_mbar":
                value = self.get_commanded_pressure_mbar()
            elif field == "pressure_atm_state":
                value = self.get_pressure_atm_state()
            elif field == "pipette_positions":
                point, focus = evidence["pipette_position"], evidence["pipette_focus"]
                value = np.append([np.nan, np.nan] if point is None else point,
                                  np.nan if focus is None else focus)
            elif field == "pipette_image_xy":
                point = evidence["pipette_position"]
                value = np.asarray([np.nan, np.nan] if point is None else point, dtype=float)
            elif field == "pipette_defocus_um":
                focus = evidence["pipette_focus"]
                value = np.nan if focus is None else float(focus)
            elif field == "pipette_focus":
                value = evidence["pipette_focus"]
            elif field == "cell_detections":
                value = evidence["cell_detections"]
            elif field == "tracked_cell_position":
                value = evidence["tracked_cell_position"]
            elif field == "deep_learning":
                value = evidence
            if field in ("resistance", "access_resistance", "capacitance"):
                if not raw_measurements and not (legacy and not include_pressure_state):
                    value = np.asarray([value], dtype=np.float32)
            values[field] = value
            frame_image = field == "camera_image" and value is source_image and source_image is not None
            if field in visual_fields or frame_image:
                statuses = evidence.get("status")
                statuses = statuses if isinstance(statuses, dict) else {}
                models = visual_models.get(field, ())
                if field == "deep_learning":
                    valid = evidence.get("source_frame") is not None and any(
                        status in ("ok", "no_detection") for status in statuses.values())
                elif frame_image:
                    valid = True
                else:
                    try:
                        valid = value is not None and bool(np.isfinite(np.asarray(value, dtype=float)).all())
                    except (TypeError, ValueError):
                        valid = False
                    valid = valid and all(statuses.get(model) in ("ok", "no_detection") for model in models)
                confidences = evidence.get("confidence")
                confidences = confidences if isinstance(confidences, dict) else {}
                metadata[field] = dict(
                    acquired_at=(evidence.get("frame_acquired_at") if evidence.get("frame_acquired_at") is not None
                                 else evidence.get("frame_available_at")), read_at=time.monotonic(),
                    frame_acquired_at=evidence.get("frame_acquired_at"),
                    frame_available_at=evidence.get("frame_available_at"),
                    source="camera" if frame_image else "inference", source_frame=evidence.get("source_frame"),
                    timestamp_basis=evidence.get("timestamp_basis", "camera_availability"), valid=bool(valid),
                    stale=bool(evidence.get("stale", True)), age_s=evidence.get("age_s"),
                    confidence={model: deepcopy(confidences.get(model)) for model in models},
                    status={model: statuses.get(model, "not_available") for model in models},
                    producer_session=evidence.get("producer_session"),
                    producer_generation=evidence.get("producer_generation"))
            else:
                cached = self._latest.get(field) or {}
                entry = deepcopy({key: item for key, item in cached.items() if key != "value"})
                acquired = entry.get("acquired_at")
                now = time.monotonic()
                age = None if acquired is None else now - acquired
                try:
                    valid = value is not None and bool(np.isfinite(np.asarray(value, dtype=float)).all())
                except (TypeError, ValueError):
                    valid = False
                if field == "motion_state":
                    valid = all(isinstance(item.get("busy"), bool) for item in value.values())
                entry.update(read_at=now, valid=bool(valid), source_frame=entry.get("source_frame"),
                             age_s=age, stale=age is None or age < 0 or age > 1.0)
                if field == "pressure" and not entry.get("freshness_known", False):
                    # A legacy controller may expose a value but no acquisition time.
                    entry.update(stale=None, age_s=None)
                metadata[field] = entry
        observed_at = time.monotonic()
        if evidence is not None:
            acquired = evidence.get("frame_acquired_at")
            if acquired is None:
                acquired = evidence.get("frame_available_at")
            age = None if acquired is None else observed_at - acquired
            evidence.update(observed_at=observed_at, age_s=age,
                            stale=age is None or age < 0 or age > self.deep_learning_max_age)
        for entry in metadata.values():
            acquired = entry.get("acquired_at")
            age = None if acquired is None else observed_at - acquired
            limit = self.deep_learning_max_age if entry.get("source") in ("camera", "inference") else 1.0
            entry.update(age_s=age, stale=age is None or age < 0 or age > limit)
            if entry.get("source") == "pressure_acquisition_cache" and not entry.get("freshness_known", False):
                entry.update(age_s=None, stale=None)
        if legacy and not include_pressure_state:
            return [values[field] for field in requested]
        values["observed_at"] = observed_at
        values["field_metadata"] = metadata
        if target_cell is not None or tracking_requested:
            self.enrich_tracking(values, target_cell)
        return values
