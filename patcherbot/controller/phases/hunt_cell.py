"""Sequential Hunt movements with direct calculations and observation history."""

import collections
import math
import sys
import time

import numpy as np

from ..errors import AutopatchError
from ..PhaseController import PhaseController


class HuntCellPhase(PhaseController):
    def __init__(self, controller):
        super().__init__(controller)
        self.contact_candidate = False
        self.cell_lost = None
        self.distance_reached = False
        self.operation = None
        self.operation_error = None
        self.last_observation = None
        self.return_to_cell = False
        self.shift_cell = False
        self.cell_z_um = None
        self.cell_observation = None
        self.spear_descent_um = 0.0

    def run(self, cell=None):
        state = None
        mode = self.controller.config.mode
        cell_type = self.controller.config.cell_type
        try:
            state = self.prepare(cell, mode=mode, cell_type=cell_type)
            while True:
                observation = self.observe(**state["observe"])
                state["readings"].append(state["latest_resistance"])
                state["latest_resistance"] = observation["resistance"]
                self.controller.abort_if_requested()
                self.controller.success_if_requested()
                if self.success_gate(observation, state, mode=mode):
                    self.complete_success()
                    return
                monitor, velocity, relative = self.decide(observation, state, mode=mode, cell_type=cell_type)
                if self.failure_gate(state, mode=mode, cell_type=cell_type):
                    raise AutopatchError("Cell not detected before reaching max hunt distance")
                if self.action_gate(observation):
                    state["completed"] = self.act(
                        monitor, observation=observation, state=state,
                        velocity=velocity, relative=relative, mode=mode)
                if self.success_gate(self.last_observation, state, mode=mode):
                    self.complete_success()
                    return
                state["started"] = True
                self.controller.sleep(0.04)
        finally:
            pending = sys.exc_info()[1]
            try:
                self.act(stop=True)
            except BaseException:
                if pending is None:
                    raise
            finally:
                cleanup_pending = sys.exc_info()[1]
                if state is not None and state["inference_started"]:
                    try:
                        self.controller.observation_helper.stop_deep_learning()
                    except BaseException:
                        if pending is None and cleanup_pending is None:
                            raise

    def prepare(self, cell, *, mode, cell_type):
        """Prepare shared measurements, then initialize the selected patch mode."""
        self.contact_candidate = False
        self.cell_lost = None
        self.distance_reached = False
        self.operation = None
        self.operation_error = None
        self.last_observation = None
        self.return_to_cell = False
        self.shift_cell = False
        self.cell_z_um = None
        self.cell_observation = None
        self.spear_descent_um = 0.0
        self.controller.info("Hunting for cell")
        self.controller.isrigready()
        if not self.controller.rig_ready:
            raise AutopatchError("Rig not ready for cell hunting")
        if cell is None:
            raise AutopatchError("No cell given to patch!")
        self.controller.pressure.set_pressure(self.controller.config.pressure_near)
        self.controller.sleep(3)
        self.controller.first_res = self.controller.resistanceRamp()
        self.controller.observation_helper.reset_history(self, None, include_images=False)
        state = dict(
            cell=cell, inference_started=False,
            started=False, completed=False,
            readings=collections.deque([self.controller.first_res], maxlen=5),
            latest_resistance=self.controller.first_res, start_position=None,
            observe=dict(fields=["manipulator_position", "resistance", "stage_positions",
                                 "deep_learning", "camera_image", "pipette_positions",
                                 "pressure", "commanded_pressure_mbar", "pressure_atm_state"],
                         raw_measurements=True, cell=None, target_cell=cell))
        if mode == "Classic":
            self.controller.info("Classic mode: pipette descent with resistance detection")
        elif mode == "Manual":
            self.controller.info("Manual mode: resistance detection without automatic movement")
        elif mode == "Training":
            self.controller.info("Training mode: waiting for user Success or Abort")
        elif mode == "Agent":
            self.controller.agenthelper.prepare_model("hunt")
        elif mode == "Adaptive":
            if cell_type == "Slice":
                try:
                    state["progress_baseline_um"] = float(self.controller.config.cell_distance)
                except (TypeError, ValueError):
                    raise AutopatchError("Adaptive initial cell distance must be finite and positive")
                if not math.isfinite(state["progress_baseline_um"]) or state["progress_baseline_um"] <= 0:
                    raise AutopatchError("Adaptive initial cell distance must be finite and positive")
                state.update(progress_cycles=0, pipette_z_um=None)
                self.operation = self.spear
                helper = self.controller.observation_helper
                try:
                    helper.start_deep_learning(cell=None)
                    helper.set_deep_learning_models(cell=True, pipette=True)
                    state["inference_started"] = True
                except BaseException:
                    helper.stop_deep_learning()
                    raise
        else:
            raise AutopatchError("Unknown Hunt mode")
        return state

    def observe(self, include_pressure_state=False, *, target_cell=None, **kwargs):
        """Associate the selected cell once and record the complete observation."""
        if include_pressure_state:
            observation = self.controller.observe(include_pressure_state=True, **kwargs)
        else:
            observation = self.controller.observe(**kwargs)
        if not isinstance(observation, dict):
            self.controller.observation_helper.record_observation(self, observation)
            return observation
        evidence = observation.get("deep_learning") or {}
        positions = evidence.get("source_positions") or {}
        reference = np.asarray(target_cell[0], dtype=float) if target_cell is not None else np.asarray([])
        candidates = []
        source_z = float("nan")
        context_valid = False
        try:
            source_xy = np.asarray(positions["stage"], dtype=float).reshape(-1)[:2]
            source_z = float(positions["microscope"])
            frame_at = float(evidence["frame_available_at"])
            position_at = float(evidence["positions_sampled_at"])
            context_valid = (source_xy.size == 2 and np.isfinite(source_xy).all()
                             and all(math.isfinite(v) for v in (source_z, frame_at, position_at))
                             and position_at >= frame_at
                             and reference.size >= 2 and np.isfinite(reference[:2]).all()
                             and evidence.get("source_frame") is not None
                             and not evidence.get("stale", True))
        except (KeyError, TypeError, ValueError):
            pass
        detections = evidence.get("cell_detections")
        association_available = context_valid and detections is not None
        if association_available:
            stage = self.controller.calibrated_stage
            expected = (np.asarray(stage.M) @ source_xy + np.asarray(stage.r0))[:2] - reference[:2]
            for detection in (() if detections is None else detections):
                try:
                    x, y, confidence = (float(v) for v in detection[:3])
                    delta = stage.pixels_to_um_relative([x - expected[0], y - expected[1], 0.0])
                    distance = math.hypot(float(delta[0]), float(delta[1]))
                except (TypeError, ValueError, IndexError):
                    continue
                if all(math.isfinite(v) for v in (x, y, confidence, distance)) and confidence > 0 and distance <= 50.0:
                    candidates.append((x, y))
        valid = association_available and len(candidates) == 1
        observation.update(
            visual_context_valid=context_valid,
            target_cell_id=tuple(float(v) for v in reference),
            target_cell_reference_coordinates=reference.copy(),
            target_cell_image_xy=np.asarray(candidates[0] if valid else [np.nan, np.nan]),
            target_cell_observed_z_um=source_z if valid else float("nan"),
            target_cell_last_known_z_um=self.cell_z_um if self.cell_z_um is not None else float("nan"),
            target_cell_last_known_source_frame=(
                (self.cell_observation.get("deep_learning") or {}).get("source_frame")
                if self.cell_observation is not None else None),
            target_cell_candidate_count=len(candidates), target_cell_valid=valid,
            target_cell_status=("observed" if valid else "ambiguous" if len(candidates) > 1
                                else "not_detected" if association_available else "unavailable"))
        observation.setdefault("field_metadata", {})["target_cell_image_xy"] = {
            "valid": valid, "acquired_at": evidence.get("frame_acquired_at"),
            "available_at": evidence.get("frame_available_at"),
            "source_frame": evidence.get("source_frame"), "source": "selected_cell_association",
            "coordinate_system": "image_pixels", "position_timestamp_basis": evidence.get("position_timestamp_basis")}
        observation["field_metadata"]["target_cell_observed_z_um"] = {
            "valid": valid, "acquired_at": evidence.get("positions_sampled_at"),
            "source_frame": evidence.get("source_frame"), "source": "microscope_context",
            "coordinate_system": "microscope_um", "synchronized_to_frame": False}
        previous_evidence = (self.cell_observation.get("deep_learning") or {}) if self.cell_observation is not None else {}
        observation["field_metadata"]["target_cell_last_known_z_um"] = {
            "valid": self.cell_z_um is not None and math.isfinite(float(self.cell_z_um)),
            "acquired_at": previous_evidence.get("positions_sampled_at"),
            "source_frame": previous_evidence.get("source_frame"), "source": "last_accepted_search",
            "coordinate_system": "microscope_um", "last_known": True, "synchronized_to_frame": False}
        helper = self.controller.observation_helper
        helper.record_observation(self, observation)
        for output, contract in self.observation_windows.items():
            predicate = contract.get("predicate")
            if predicate is None or predicate(observation):
                observation[output] = self.observation_window(**contract)
        return observation

    def action_gate(self, observation=None, state=None, *, monitor=None, command=None,
                    start=None, mode=None, max_distance=None, device=None):
        """Permit only finite commands inside the current operation's limits."""
        if observation is None or self.controller.abort_requested or self.controller.success_requested:
            return False
        if command is None:
            return True
        try:
            if not isinstance(command, dict):
                stopping = np.asarray(command, dtype=float)
                if stopping.shape == (() if monitor is not None else (3,)) and np.all(stopping == 0):
                    expected = (self.controller.microscope if monitor in (self.search, self.scan)
                                else self.controller.calibrated_stage if monitor == self.shift and self.shift_cell
                                else self.controller.calibrated_unit)
                    return device is expected
            if mode in ("Manual", "Training"):
                return False
            if monitor is None:
                vector = np.asarray(command, dtype=float)
                if vector.shape != (3,) or not np.isfinite(vector).all():
                    return False
                if device is not self.controller.calibrated_unit or mode not in ("Classic", "Agent", "Adaptive"):
                    return False
                if mode != "Agent" and np.any(vector[:2] != 0):
                    return False
                limit = float(max_distance)
                z = float(observation["manipulator_position"][2])
                origin = float(state["start_position"][2])
                return bool(math.isfinite(limit) and limit > 0 and math.isfinite(z)
                            and math.isfinite(origin) and abs(z - origin) < limit)
            if mode != "Adaptive":
                return False
            if monitor == self.shift:
                expected = self.controller.calibrated_stage if self.shift_cell else self.controller.calibrated_unit
                if device is not expected or not isinstance(command, dict) or set(command) != {"relative"}:
                    return False
                vector = np.asarray(command["relative"], dtype=float)
                if vector.ndim != 1 or len(vector) not in (2, 3) or not np.isfinite(vector).all():
                    return False
                if len(vector) == 3 and vector[2] != 0:
                    return False
                field = "stage_positions" if self.shift_cell else "manipulator_position"
                current = np.asarray(observation[field][:2], dtype=float)
                origin = np.asarray(start[field][:2], dtype=float)
                return bool(current.shape == (2,) and origin.shape == (2,)
                            and np.isfinite(current).all() and np.isfinite(origin).all()
                            and np.linalg.norm(vector[:2]) <= 50
                            and np.linalg.norm(current - origin) <= 50
                            and np.linalg.norm(current + vector[:2] - origin) <= 50)
            if monitor in (self.search, self.scan):
                if device is not self.controller.microscope:
                    return False
                limit = min(float(state["travel_limit"]), float(max_distance))
                if not math.isfinite(float(max_distance)):
                    return False
                z, origin = float(observation["stage_positions"][2]), float(start["stage_positions"][2])
                if not all(math.isfinite(v) for v in (limit, z, origin)) or limit <= 0 or abs(z - origin) > limit:
                    return False
                if isinstance(command, dict):
                    if monitor != self.scan or set(command) != {"absolute_z"}:
                        return False
                    target = float(command["absolute_z"])
                    return math.isfinite(target) and abs(target - origin) <= limit
                speed = float(command)
                direction = float(device.up_direction) * (-1 if monitor == self.search else 1)
                return (math.isfinite(speed) and direction in (-1, 1) and speed * direction > 0
                        and abs(z - origin) < limit)
            if monitor == self.spear:
                if device is not self.controller.calibrated_unit or isinstance(command, dict):
                    return False
                speed, limit = float(command), float(max_distance)
                z, origin = float(observation["manipulator_position"][2]), float(start["manipulator_position"][2])
                return (all(math.isfinite(v) for v in (speed, limit, z, origin)) and limit > 0
                        and abs(z - origin) < limit
                        and speed * state["descent_direction"] > 0)
        except (KeyError, TypeError, ValueError, IndexError, OverflowError):
            return False
        return False

    def calculate(
        self, operation=None, *, current_z_um=None, start_z_um=None,
        previous_descent_um=0.0, descent_direction=None, max_distance=None,
        delayed_resistance_mohm=None, baseline_resistance_mohm=None,
        cell_R_increase=None, hunt_search_margin=None,
    ):
        """Return calculated values; callers pass current measurements/config values."""
        if operation is None:
            return abs(current_z_um - start_z_um) >= max_distance
        if operation == self.spear:
            signed_descent_um = descent_direction * (current_z_um - start_z_um)
            maximum_descent_um = max(previous_descent_um, signed_descent_um)
            distance_reached = maximum_descent_um >= max_distance
            resistance_increase_mohm = None
            contact_candidate = False
            if delayed_resistance_mohm is not None:
                resistance_increase_mohm = delayed_resistance_mohm - baseline_resistance_mohm
                contact_candidate = resistance_increase_mohm >= cell_R_increase
            return (signed_descent_um, maximum_descent_um, resistance_increase_mohm,
                    distance_reached, contact_candidate)
        if operation in (self.search, self.scan):
            deck = self.observation_deck
            positions = [row["manipulator_position"][2] for row in deck
                         if isinstance(row, dict) and "manipulator_position" in row]
            stage_z = [row["stage_positions"][2] for row in deck
                       if isinstance(row, dict) and "stage_positions" in row]
            if not positions or not stage_z or not all(math.isfinite(float(z)) for z in positions + stage_z):
                raise AutopatchError("Search/Scan require finite position history")
            measured_travel = max(positions) - min(positions) + max(stage_z) - min(stage_z)
            maximum = float(self.controller.config.max_distance)
            margin = float(self.controller.config.hunt_search_margin)
            if not math.isfinite(maximum) or not math.isfinite(margin) or maximum <= 0 or margin < 0:
                raise AutopatchError("Search/Scan distance settings must be finite and nonnegative")
            return min(maximum, measured_travel + margin)
        raise ValueError("Unknown Hunt calculation")

    def decide(self, observation=None, state=None, *, mode, cell_type):
        """Choose the next operation; movement monitoring belongs to act."""
        if state["start_position"] is None:
            state["start_position"] = observation["manipulator_position"]
        if mode == "Adaptive" and cell_type == "Slice":
            cycles = float(self.controller.config.hunt_progress_cycles)
            required_percent = float(self.controller.config.hunt_min_progress_percent)
            if not math.isfinite(cycles) or not cycles.is_integer() or not 1 <= cycles <= 100:
                raise AutopatchError("Adaptive progress cycles must be a whole number from 1 to 100")
            if not math.isfinite(required_percent) or not 0 <= required_percent <= 100:
                raise AutopatchError("Adaptive progress percentage must be between 0 and 100")
            if state["completed"]:
                if self.operation == self.spear:
                    self.operation = self.search
                elif self.operation == self.search:
                    self.shift_cell = True
                    self.operation = self.shift
                elif self.operation == self.shift:
                    if not self.shift_cell:
                        try:
                            cell_z = float(self.cell_z_um)
                            pipette_z = float(state["pipette_z_um"])
                        except (TypeError, ValueError):
                            raise AutopatchError("Adaptive progress requires measured cell and pipette Z")
                        if not math.isfinite(cell_z) or not math.isfinite(pipette_z):
                            raise AutopatchError("Adaptive progress requires finite cell and pipette Z")
                        gap = abs(cell_z - pipette_z)
                        baseline = state["progress_baseline_um"]
                        state["progress_cycles"] += 1
                        progress = 100.0 * (baseline - gap) / baseline if baseline > 0 else 0.0
                        self.controller.observation_helper.record_calculations(self, observation, {
                            "operation": "cycle_complete", "cell_z_um": cell_z,
                            "pipette_z_um": pipette_z, "pipette_cell_gap_um": gap,
                            "progress_baseline_um": state["progress_baseline_um"],
                            "progress_cycles": state["progress_cycles"],
                            "progress_percent": progress, "required_progress_percent": required_percent,
                            "progress_cycle_limit": int(cycles)})
                        if progress >= required_percent:
                            state["progress_baseline_um"] = gap
                            state["progress_cycles"] = 0
                        elif state["progress_cycles"] >= cycles:
                            raise AutopatchError(
                                f"Adaptive Hunt failed: Z gap decreased {progress:.1f}% over "
                                f"{state['progress_cycles']} cycles; required {required_percent:g}% "
                                f"({baseline:g} to {gap:g} um)")
                    self.operation = self.scan if self.shift_cell else self.spear
                elif self.operation == self.scan:
                    self.shift_cell = False
                    self.operation = self.shift
            return self.operation, None, False
        if mode == "Agent":
            action = self.controller.agenthelper.run_inference(observation=observation)
            if action is None:
                return None, [0, 0, 0], True
            if len(action) != 3 or not all(math.isfinite(float(v)) for v in action):
                raise AutopatchError("Hunt agent must return three finite velocity components")
            vx, vy, vz = (float(v) for v in action)
            xy = self.controller.calibrated_unit.pixels_to_um_relative([-vx, -vy, 0.0])
            return None, [float(xy[0]), float(xy[1]), vz], True
        if mode == "Classic" or mode == "Adaptive":
            return None, [0, 0, self.controller.config.max_descent_speed], False
        if mode == "Manual":
            return None, None, False
        if mode == "Training":
            return None, None, False
        raise AutopatchError("Unknown Hunt mode")

    def act(self, monitor=None, *, observation=None, state=None,
            velocity=None, relative=False, stop=False, mode=None):
        """Execute one action, observing and stopping on its low-level conditions."""
        if stop:
            error = None
            for device in (self.controller.calibrated_stage,
                           self.controller.calibrated_unit, self.controller.microscope):
                try:
                    device.stop()
                except BaseException as exc:
                    error = exc
            if error is not None:
                raise error
            return
        if monitor is None:
            if velocity is not None:
                if not all(math.isfinite(float(v)) for v in velocity):
                    raise AutopatchError("Hunt velocity must be finite")
                if not self.action_gate(observation, state, command=velocity, mode=mode,
                                        max_distance=self.controller.config.max_distance,
                                        device=self.controller.calibrated_unit):
                    self.controller.abort_if_requested()
                    self.controller.success_if_requested()
                    raise AutopatchError("Hunt movement rejected by action gate")
                mover = self.controller.calibrated_unit
                if relative:
                    mover.relative_move_group_velocity(velocity)
                elif state.get("applied_velocity") != tuple(velocity):
                    mover.absolute_move_group_velocity(velocity)
                    state["applied_velocity"] = tuple(velocity)
            return False
        start = observation
        state["monitor_state"] = {}
        started_at = time.monotonic()
        if monitor in (self.search, self.scan):
            state["travel_limit"] = self.calculate(self.search)
        self.spear_descent_um = 0.0
        state["descent_direction"] = 0
        applied_speed = None
        self.contact_candidate = False
        self.distance_reached = False
        try:
            while True:
                self.last_observation = observation
                self.controller.abort_if_requested()
                self.controller.success_if_requested()
                if self.success_gate(observation, state, mode=mode):
                    return False
                completed, velocity, device = monitor(observation, start, state)
                if completed:
                    return True
                if time.monotonic() - started_at > 60:
                    raise AutopatchError("Hunt action exceeded its monitoring timeout")
                if velocity is None:
                    raise AutopatchError("Hunt operation has insufficient observation evidence")
                if isinstance(velocity, dict):
                    device.stop()
                    if "relative" in velocity:
                        displacement = np.asarray(velocity["relative"], dtype=float)
                        if (displacement.ndim != 1 or len(displacement) not in (2, 3)
                                or not np.isfinite(displacement).all()):
                            raise AutopatchError("Shift requires finite planar displacement")
                        if len(displacement) == 3:
                            displacement[2] = 0.0
                        length = float(np.linalg.norm(displacement[:2]))
                        if length > 50:
                            displacement[:2] *= 50.0 / length
                        if not self.action_gate(observation, state, monitor=monitor,
                                                command={"relative": displacement}, start=start,
                                                mode=mode, device=device,
                                                max_distance=self.controller.config.max_distance):
                            self.controller.abort_if_requested()
                            self.controller.success_if_requested()
                            raise AutopatchError("Shift movement rejected by action gate")
                        device.relative_move(displacement)
                        device.wait_until_still()
                        state["monitor_state"]["after_move"] = time.monotonic()
                    elif "absolute_z" in velocity:
                        target = float(velocity["absolute_z"])
                        if (not math.isfinite(target) or
                                abs(target - start["stage_positions"][2]) > state["travel_limit"]):
                            raise AutopatchError("Focus return exceeds Scan travel limit")
                        if not self.action_gate(observation, state, monitor=monitor,
                                                command={"absolute_z": target}, start=start,
                                                mode=mode, device=device,
                                                max_distance=self.controller.config.max_distance):
                            self.controller.abort_if_requested()
                            self.controller.success_if_requested()
                            raise AutopatchError("Scan return rejected by action gate")
                        device.absolute_move(target)
                        device.wait_until_still()
                        state["monitor_state"]["after_move"] = time.monotonic()
                    else:
                        raise AutopatchError("Unknown Hunt displacement command")
                    applied_speed = None
                else:
                    if not math.isfinite(float(velocity)):
                        raise AutopatchError("Hunt speed must be finite")
                    if not self.action_gate(observation, state, monitor=monitor, command=velocity,
                                            start=start, mode=mode, device=device,
                                            max_distance=self.controller.config.max_distance):
                        self.controller.abort_if_requested()
                        self.controller.success_if_requested()
                        raise AutopatchError("Hunt velocity rejected by action gate")
                    if velocity != applied_speed:
                        if device == self.controller.calibrated_unit:
                            device.absolute_move_group_velocity([0, 0, velocity])
                        elif device == self.controller.microscope:
                            device.absolute_move_velocity(velocity)
                        elif velocity == 0:
                            device.stop()
                        else:
                            raise AutopatchError("Stage XY requires a gated displacement")
                        applied_speed = velocity
                observation = self.observe(**state["observe"])
                state["readings"].append(state["latest_resistance"])
                state["latest_resistance"] = observation["resistance"]
                self.controller.sleep(0.04)
        finally:
            pending = sys.exc_info()[1]
            try:
                self.act(stop=True)
            except BaseException:
                if pending is None:
                    raise

    def spear(self, observation, start, state):
        """Monitor pipette descent using the same observation as every other mode."""
        speed = float(self.controller.config.max_descent_speed)
        if state["descent_direction"] == 0 and speed != 0:
            state["descent_direction"] = 1 if speed > 0 else -1
        descent = state["descent_direction"] * (
            observation["manipulator_position"][2] - start["manipulator_position"][2])
        self.spear_descent_um = max(self.spear_descent_um, descent)
        self.distance_reached = self.spear_descent_um >= self.controller.config.max_distance
        evidence = observation.get("deep_learning") or {}
        memo = state["monitor_state"]
        if observation.get("target_cell_candidate_count", 0) > 1:
            raise AutopatchError("Spear found multiple plausible cells")
        if observation.get("target_cell_valid", False):
            memo["cell_seen"] = True
            self.cell_lost = False
        elif observation.get("target_cell_status") == "not_detected" and memo.get("cell_seen", False):
            self.cell_lost = True
        elif not memo.get("cell_seen", False):
            self.cell_lost = None
        self.controller.observation_helper.record_calculations(self, observation, {
            "operation": "spear", "maximum_net_descent_um": self.spear_descent_um,
            "distance_reached": self.distance_reached, "cell_lost": self.cell_lost,
            "pipette_visible": not evidence.get("stale", True) and evidence.get("pipette_position") is not None})
        return self.distance_reached or self.cell_lost is True, speed, self.controller.calibrated_unit

    def search(self, observation, start, state):
        """Search downward for the selected cell using fresh calibrated detections."""
        memo = state["monitor_state"]
        device = self.controller.microscope
        z = float(observation["stage_positions"][2])
        origin = float(start["stage_positions"][2])
        limit = float(state["travel_limit"])
        if not all(math.isfinite(v) for v in (z, origin, limit)) or limit <= 0:
            raise AutopatchError("Search requires finite positions and a positive travel limit")
        travelled = abs(z - origin)
        if travelled > limit:
            raise AutopatchError("Search exceeded its observation-derived distance limit")
        evidence = observation.get("deep_learning") or {}
        frame = evidence.get("source_frame")
        acquired = evidence.get("frame_acquired_at")
        sampled = evidence.get("positions_sampled_at")
        fresh = (frame is not None and not evidence.get("stale", True) and frame != memo.get("frame")
                 and isinstance(acquired, (int, float)) and math.isfinite(acquired)
                 and isinstance(sampled, (int, float)) and math.isfinite(sampled)
                 and sampled >= acquired and observation.get("visual_context_valid", False))
        if "after_move" in memo:
            fresh = fresh and acquired >= memo["after_move"]
        if not fresh:
            memo["waiting"] = memo.get("waiting", 0) + 1
            if memo["waiting"] >= 50:
                raise AutopatchError("Search received no fresh visual evidence")
            return False, 0.0, device
        memo.update(frame=frame, waiting=0)
        count = observation.get("target_cell_candidate_count", 0)
        self.controller.observation_helper.record_calculations(self, observation, {
            "operation": "search", "search_distance_um": limit, "travelled_um": travelled,
            "associated_cells": count})
        if count > 1:
            raise AutopatchError("Search found multiple plausible cells")
        if observation.get("target_cell_valid", False):
            cell_z = float(observation["target_cell_observed_z_um"])
            if not math.isfinite(cell_z) or abs(cell_z - origin) > limit:
                raise AutopatchError("Search cell observation lies outside its travel envelope")
            self.cell_z_um = cell_z
            self.cell_observation = observation
            self.cell_lost = False
            return True, 0.0, device
        if travelled >= limit:
            raise AutopatchError("Search reached its observation-derived distance limit")
        speed = abs(float(self.controller.config.max_descent_speed))
        direction = -float(device.up_direction)
        if not math.isfinite(speed) or speed <= 0 or direction not in (-1.0, 1.0):
            raise AutopatchError("Search requires a nonzero finite speed and microscope direction")
        return False, direction * speed, device

    def scan(self, observation, start, state):
        """Scan upward, bracket minimum absolute defocus, and verify its saved Z."""
        memo = state["monitor_state"]
        device = self.controller.microscope
        z = float(observation["stage_positions"][2])
        origin = float(start["stage_positions"][2])
        limit = float(state["travel_limit"])
        if not all(math.isfinite(v) for v in (z, origin, limit)) or limit <= 0:
            raise AutopatchError("Scan requires finite positions and a positive travel limit")
        travelled = abs(z - origin)
        if travelled > limit:
            raise AutopatchError("Scan exceeded its observation-derived distance limit")
        evidence = observation.get("deep_learning") or {}
        frame = evidence.get("source_frame")
        acquired = evidence.get("frame_acquired_at")
        sampled = evidence.get("positions_sampled_at")
        fresh = (frame is not None and not evidence.get("stale", True) and frame != memo.get("frame")
                 and isinstance(acquired, (int, float)) and math.isfinite(acquired)
                 and isinstance(sampled, (int, float)) and math.isfinite(sampled)
                 and sampled >= acquired and observation.get("visual_context_valid", False))
        if "after_move" in memo:
            fresh = fresh and acquired >= memo["after_move"]
        if not fresh:
            memo["waiting"] = memo.get("waiting", 0) + 1
            if memo["waiting"] >= 50:
                raise AutopatchError("Scan received no fresh visual evidence")
            return False, 0.0, device
        memo.update(frame=frame, waiting=0)
        try:
            focus = float(evidence["pipette_focus"])
            point = np.asarray(evidence["pipette_position"], dtype=float).reshape(-1)
            source_z = float((evidence.get("source_positions") or {})["microscope"])
        except (KeyError, TypeError, ValueError):
            focus, source_z, point = float("nan"), float("nan"), np.asarray([])
        valid = (math.isfinite(focus) and math.isfinite(source_z) and abs(source_z - origin) <= limit
                 and point.size >= 2 and np.isfinite(point[:2]).all())
        self.controller.observation_helper.record_calculations(self, observation, {
            "operation": "scan", "scan_distance_um": limit, "travelled_um": travelled,
            "pipette_focus": focus})
        if memo.get("returning"):
            if abs(z - memo["best_z"]) > 1.0:
                raise AutopatchError("Scan failed to return to its measured best-focus position")
            if valid and abs(focus) <= memo["best_focus"] + 1.0:
                state["pipette_z_um"] = memo["best_z"]
                return True, 0.0, device
            raise AutopatchError("Scan could not verify pipette focus after returning")
        if valid:
            previous = memo.get("previous_focus")
            best = memo.get("best_focus", float("inf"))
            bracketed = previous is not None and (previous * focus < 0 or
                         (memo.get("improving", False) and abs(focus) > best + 0.1))
            if abs(focus) < best:
                if previous is not None:
                    memo["improving"] = True
                memo.update(best_focus=abs(focus), best_z=source_z)
            memo["previous_focus"] = focus
            if abs(focus) <= 1.0 or bracketed:
                if abs(z - memo["best_z"]) <= 0.1:
                    state["pipette_z_um"] = memo["best_z"]
                    return True, 0.0, device
                memo["returning"] = True
                return False, {"absolute_z": memo["best_z"]}, device
        if travelled >= limit:
            raise AutopatchError("Scan reached its distance limit without resolving focus")
        speed = abs(float(self.controller.config.max_descent_speed))
        direction = float(device.up_direction)
        if not math.isfinite(speed) or speed <= 0 or direction not in (-1.0, 1.0):
            raise AutopatchError("Scan requires a nonzero finite speed and microscope direction")
        return False, direction * speed, device

    def shift(self, observation, start, state):
        """Center the observed cell or pipette with bounded, verified planar moves."""
        memo = state["monitor_state"]
        device = self.controller.calibrated_stage if self.shift_cell else self.controller.calibrated_unit
        evidence = observation.get("deep_learning") or {}
        frame = evidence.get("source_frame")
        acquired = evidence.get("frame_acquired_at")
        sampled = evidence.get("positions_sampled_at")
        fresh = (frame is not None and not evidence.get("stale", True) and frame != memo.get("frame")
                 and isinstance(acquired, (int, float)) and math.isfinite(acquired)
                 and isinstance(sampled, (int, float)) and math.isfinite(sampled)
                 and sampled >= acquired and observation.get("visual_context_valid", False))
        if "after_move" in memo:
            fresh = fresh and acquired >= memo["after_move"]
        if not fresh:
            memo["waiting"] = memo.get("waiting", 0) + 1
            if memo["waiting"] >= 50:
                raise AutopatchError("Shift received no fresh post-move visual evidence")
            return False, 0.0, device
        memo.update(frame=frame, waiting=0)
        shape = evidence.get("source_image_shape")
        if shape is None:
            image = observation.get("camera_image")
            shape = None if image is None else image.shape
        if shape is None or len(shape) < 2 or min(shape[:2]) <= 0:
            raise AutopatchError("Shift requires the detection image dimensions")
        if self.shift_cell:
            if not observation.get("target_cell_valid", False):
                raise AutopatchError("Shift requires exactly one associated cell")
            point = np.asarray(observation["target_cell_image_xy"], dtype=float)
        else:
            try:
                point = np.asarray(evidence["pipette_position"], dtype=float).reshape(-1)[:2]
            except (KeyError, TypeError, ValueError):
                raise AutopatchError("Shift requires an observed pipette position")
        if point.size != 2 or not np.isfinite(point).all() or not (0 <= point[0] < shape[1] and 0 <= point[1] < shape[0]):
            raise AutopatchError("Shift target lies outside its detection image")
        pixels = [shape[1] / 2.0 - point[0], shape[0] / 2.0 - point[1], 0.0]
        correction = np.asarray(device.pixels_to_um_relative(pixels), dtype=float).reshape(-1)
        if correction.size < 2 or not np.isfinite(correction[:2]).all():
            raise AutopatchError("Shift calibration produced invalid planar displacement")
        distance = math.hypot(float(correction[0]), float(correction[1]))
        field = "stage_positions" if self.shift_cell else "manipulator_position"
        travelled = math.hypot(*(np.asarray(observation[field][:2], dtype=float) - np.asarray(start[field][:2], dtype=float)))
        if not math.isfinite(travelled) or travelled > 50.0:
            raise AutopatchError("Shift exceeded its 50 um planar envelope")
        self.controller.observation_helper.record_calculations(self, observation, {
            "operation": "shift", "max_planar_displacement_um": 50.0,
            "shift_cell": self.shift_cell, "alignment_error_um": distance, "travelled_um": travelled})
        if distance <= 1.0:
            return True, 0.0, device
        if "previous_error" in memo and distance >= memo["previous_error"] - 0.05:
            raise AutopatchError("Shift did not improve alignment after its previous move")
        remaining = 50.0 - memo.get("commanded_distance", 0.0)
        if remaining <= 0 or memo.get("moves", 0) >= 8:
            raise AutopatchError("Shift exhausted its bounded alignment budget")
        scale = min(1.0, remaining / distance)
        movement = [float(correction[0]) * scale, float(correction[1]) * scale]
        if not self.shift_cell:
            movement.append(0.0)
        memo.update(previous_error=distance, moves=memo.get("moves", 0) + 1,
                    commanded_distance=memo.get("commanded_distance", 0.0) + distance * scale)
        return False, {"relative": movement}, device

    def failure_gate(self, state=None, *, mode, cell_type):
        if self.controller.abort_requested:
            return True
        if (mode == "Training" or
                (mode == "Adaptive" and cell_type == "Slice") or
                state["start_position"] is None):
            return False
        deck = self.observation_deck
        return abs(deck[-1]["manipulator_position"][2] - state["start_position"][2]) >= self.controller.config.max_distance

    def success_gate(self, observation=None, state=None, *, mode):
        if self.controller.success_requested:
            return True
        if mode == "Training":
            return False
        return self._resistance_threshold_reached(state["readings"], self.controller.config.cell_R_increase)

    def _resistance_threshold_reached(self, readings, threshold):
        """Preserve the delayed five-reading resistance threshold."""
        return len(readings) >= 5 and readings[4] - self.controller.first_res >= threshold

    def _isCellDetected(self, lastResDeque, cellThreshold=0.15):
        """Compatibility wrapper for callers outside the phase lifecycle."""
        detected = self._resistance_threshold_reached(lastResDeque, cellThreshold)
        if detected:
            self.controller.calibrated_unit.stop()
        return detected


