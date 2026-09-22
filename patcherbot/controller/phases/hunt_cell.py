"""Sequential Hunt movements with direct calculations and observation history."""

import collections
import math
import sys

from ..base import RequestedAbortException, RequestedSuccessException
from ..errors import AutopatchError
from ..PhaseController import PhaseController


class HuntCellPhase(PhaseController):
    def run(self, cell=None):
        self.cell = cell
        self.prepare()
        try:
            if self.operation is not None:
                completed = False
                while True:
                    monitor = self.decide(completed)
                    if monitor is None:
                        return
                    completed = self.act(monitor)
            while not self.controller.abort_requested:
                if self.agent_mode:
                    observation = self.observe()
                    resistance = observation[3]
                    position = self.controller.calibrated_unit.position()
                else:
                    observation = self.observe(
                        fields=["manipulator_position", "resistance"], raw_measurements=True)
                    resistance = observation["resistance"]
                    position = observation["manipulator_position"]
                self.resistance_readings.append(self.latest_resistance)
                self.latest_resistance = resistance
                self.distance_reached = False
                if not self.training_mode:
                    self.distance_reached = self.calculate(
                        current_z_um=position[2], start_z_um=self.start_position[2],
                        max_distance=self.controller.config.max_distance)
                velocity = self.decide(observation)
                if self.cell_detected:
                    return
                self.act(velocity=velocity, relative=self.agent_mode, stop=self.stop_requested)
                self.controller.sleep(0.04)
        finally:
            pending = sys.exc_info()[1]
            try:
                self.act(stop=True)
            except BaseException:
                if pending is None or self.operation is None:
                    raise
        if self.controller.abort_requested:
            self.decide()

    def prepare(self, state=None):
        """Reset attempt variables and establish the original rig/baseline setup."""
        self.started = False
        self.cell_detected = False
        self.contact_candidate = False
        self.cell_lost = None
        self.distance_reached = False
        self.stop_requested = False
        self.operation = None
        self.operation_error = None
        self.last_observation = None
        self.return_to_cell_focus = False
        self.shift_cell = False
        self.cell_focus_z_um = None
        self.cell_focus_observation = None
        self.spear_descent_um = 0.0
        self.controller.info("Hunting for cell")
        self.controller.isrigready()
        if self.controller.rig_ready == False:
            raise AutopatchError("Rig not ready for cell hunting")
        if self.cell is None:
            raise AutopatchError("No cell given to patch!")
        self.controller.info(f"Setting pressure to {self.controller.config.pressure_near} mbar")
        self.controller.pressure.set_pressure(self.controller.config.pressure_near)
        self.controller.sleep(3)
        self.resistance_readings = collections.deque(maxlen=5)
        self.latest_resistance = self.controller.daq.resistance()
        self.resistance_readings.append(self.latest_resistance)
        self.start_position = self.controller.calibrated_unit.position()
        self.controller.first_res = self.controller.resistanceRamp()
        self.controller.info(f"Initial resistance: {self.controller.first_res}")
        self.controller.info(f"{self.controller.config.mode}: starting hunt")
        self.training_mode = self.controller.config.mode == "Training"
        self.agent_mode = self.controller.config.mode == "Agent"
        if self.controller.config.mode == "Adaptive" and self.controller.config.cell_type == "Slice":
            # Setup assumptions only, not measured separation or focus coordinates.
            self.assumed_separation_um = (15.0, 30.0)
            self.assumed_pipette_centered = True
            self.assumed_cell_centered = True
            self.assumed_cell_focused = True
            self.operation = self.spear
            self.controller.observation_helper.reset_history(self, 120)
        if self.agent_mode and not self.controller.abort_requested:
            self.controller.agenthelper.prepare_model("hunt")
        if self.training_mode:
            self.controller.info(
                "Training mode: max hunt distance check disabled; waiting for resistance threshold.")

    def action_gate(self, observation=None, state=None):
        permitted = observation is not None and not self.controller.abort_requested
        if self.operation is not None:
            permitted = permitted and not self.controller.success_requested
        return permitted

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
        if operation == self.search:
            return previous_descent_um + hunt_search_margin
        # Scan/Shift evidence calculations remain unavailable, not negative evidence.
        if operation == self.scan or operation == self.shift:
            return None
        raise ValueError("Unknown Hunt calculation")

    def decide(self, observation=None, state=None):
        """Decide failure, success, or the next movement from the gate results."""
        failed = self.failure_gate()
        succeeded = self.success_gate()
        if failed:
            if self.operation_error is not None:
                if not isinstance(self.operation_error, (RequestedAbortException, RequestedSuccessException)):
                    self.controller.warning(str(self.operation_error))
                raise self.operation_error
            self.controller.abort_if_requested()
            if self.operation is not None and self.contact_candidate:
                self.controller.warning("Contact confirmation is unfinished")
                raise AutopatchError("Contact confirmation is unfinished")
            # Preserve the original precedence of detection over the distance limit.
            if not succeeded:
                raise AutopatchError("Cell not detected before reaching max hunt distance")
        if succeeded:
            if isinstance(self.operation_error, RequestedSuccessException):
                raise self.operation_error
            self.cell_detected = True
            self.act(stop=True)
            if self.training_mode and not self.controller.success_requested:
                self.controller.info("Training mode: waiting for manual Success or Abort.")
                self.wait_for_manual_completion()
            self.complete_success()
            return None
        if self.operation is not None:
            completed = observation
            if self.operation == self.spear:
                if self.cell_lost is True or self.distance_reached:
                    self.return_to_cell_focus = False
                    self.operation = self.search
            elif completed:
                if self.operation == self.search:
                    if self.return_to_cell_focus:
                        self.return_to_cell_focus = False
                        self.operation = self.spear
                    else:
                        try:
                            focus_z_um = float(self.last_observation["stage_positions"][2])
                        except (TypeError, KeyError, IndexError, ValueError):
                            focus_z_um = float("nan")
                        if not math.isfinite(focus_z_um):
                            self.controller.warning("Search found no usable cell-focus reference")
                            raise AutopatchError("Search found no usable cell-focus reference")
                        self.cell_focus_z_um = focus_z_um
                        self.cell_focus_observation = self.last_observation
                        self.shift_cell = True
                        self.operation = self.shift
                elif self.operation == self.shift:
                    if self.shift_cell:
                        self.operation = self.scan
                    else:
                        self.return_to_cell_focus = True
                        self.operation = self.search
                elif self.operation == self.scan:
                    self.shift_cell = False
                    self.operation = self.shift
            return self.operation
        self.stop_requested = self.distance_reached
        if self.stop_requested:
            return None
        if self.controller.config.mode == "Classic":
            if not self.started:
                return [0, 0, self.controller.config.max_descent_speed]
        elif self.controller.config.mode == "Adaptive":
            # Slice uses the monitor sequence above; Plate keeps direct descent.
            if not self.started:
                return [0, 0, self.controller.config.max_descent_speed]
        elif self.controller.config.mode == "Agent":
            if not self.started:
                return None
            action = self.controller.agenthelper.run_inference(observation=observation)
            if action is None:
                self.stop_requested = True
                return None
            try:
                if len(action) != 3:
                    raise ValueError("expected three components")
                vx, vy, vz = (float(value) for value in action)
            except (TypeError, ValueError) as exc:
                raise AutopatchError("Hunt agent must return three velocity components.") from exc
            if not all(math.isfinite(value) for value in (vx, vy, vz)):
                raise AutopatchError("Hunt agent returned a non-finite velocity.")
            xy = self.controller.calibrated_unit.pixels_to_um_relative([-vx, -vy, 0.0])
            velocity = [float(xy[0]), float(xy[1]), vz]
            if not all(math.isfinite(value) for value in velocity):
                raise AutopatchError("Hunt agent velocity calibration returned non-finite values.")
            return velocity
        elif self.controller.config.mode in ("Manual", "Training"):
            return None
        return None

    def act(self, monitor=None, *, velocity=None, relative=False, stop=False):
        """Execute direct movement and monitor calls; always stop before returning."""
        if stop:
            stop_error = None
            for device in (self.controller.calibrated_stage,
                           self.controller.calibrated_unit, self.controller.microscope):
                try:
                    device.stop()
                except BaseException as exc:
                    stop_error = exc
            if stop_error is not None:
                raise stop_error
            return
        if monitor is None:
            if velocity is not None:
                if relative:
                    self.controller.calibrated_unit.relative_move_group_velocity(velocity)
                else:
                    self.controller.calibrated_unit.absolute_move_group_velocity(velocity)
                self.started = True
            elif self.agent_mode:
                self.started = True
            return

        completed = False
        generator = None
        self.operation_error = None
        self.last_observation = None
        self.contact_candidate = False
        self.cell_lost = None
        self.distance_reached = False
        requests = (RequestedAbortException, RequestedSuccessException)
        try:
            failed = self.failure_gate()
            succeeded = self.success_gate()
            if failed or succeeded:
                return False
            if not self.action_gate(monitor):
                raise AutopatchError("Hunt movement is not permitted")
            if monitor == self.search:
                if self.return_to_cell_focus:
                    if (self.cell_focus_z_um is None or self.cell_focus_observation is None
                            or not math.isfinite(self.cell_focus_z_um)):
                        raise AutopatchError("Return to cell focus lacks a usable Search reference")
                    raise AutopatchError("Return to cell focus is unfinished")
                raise AutopatchError("Cell Search is unfinished")
            if monitor == self.scan:
                raise AutopatchError("Pipette Scan is unfinished")
            if monitor == self.shift:
                raise AutopatchError("XY Shift is unfinished")
            if monitor != self.spear:
                raise ValueError("Unknown Hunt monitor")
            self.start_position = self.controller.calibrated_unit.position()
            self.spear_descent_um = 0.0
            requested_speed = float(self.controller.config.max_descent_speed)
            if not math.isfinite(requested_speed):
                raise ValueError("Spear speed must be finite")
            self.spear_direction = 0
            if requested_speed > 0:
                self.spear_direction = 1
            elif requested_speed < 0:
                self.spear_direction = -1
            self.controller.calibrated_unit.absolute_move_group_velocity([0, 0, requested_speed])
            self.applied_descent_speed_um_s = requested_speed
            self.started = True
            generator = monitor()
            for keep_monitoring, completed, requested_speed in generator:
                failed = self.failure_gate()
                succeeded = self.success_gate()
                if failed or succeeded or not keep_monitoring:
                    break
                if not math.isfinite(requested_speed):
                    raise ValueError("Spear speed must be finite")
                if requested_speed != self.applied_descent_speed_um_s:
                    self.controller.calibrated_unit.absolute_move_group_velocity([0, 0, requested_speed])
                    self.applied_descent_speed_um_s = requested_speed
                    if self.spear_direction == 0:
                        if requested_speed > 0:
                            self.spear_direction = 1
                        elif requested_speed < 0:
                            self.spear_direction = -1
            else:
                raise AutopatchError("Hunt monitor ended without a stop condition")
        except Exception as exc:
            self.operation_error = exc
        finally:
            for device in (self.controller.calibrated_stage,
                           self.controller.calibrated_unit, self.controller.microscope):
                try:
                    device.stop()
                except Exception as exc:
                    if self.operation_error is None or (
                            isinstance(exc, requests) and not isinstance(self.operation_error, requests)):
                        self.operation_error = exc
            if generator is not None:
                try:
                    generator.close()
                except Exception as exc:
                    if self.operation_error is None or (
                            isinstance(exc, requests) and not isinstance(self.operation_error, requests)):
                        self.operation_error = exc
            self.started = False
        return completed

    def spear(self):
        while True:
            observation = self.observe(
                fields=["manipulator_position", "resistance", "stage_positions", "deep_learning"],
                raw_measurements=True, cell=self.cell)
            self.last_observation = observation
            measured_resistance = observation["resistance"]
            current_z_um = observation["manipulator_position"][2]
            self.resistance_readings.append(self.latest_resistance)
            self.latest_resistance = measured_resistance
            delayed_resistance = None
            if len(self.resistance_readings) >= 5:
                delayed_resistance = self.resistance_readings[4]
            config = self.controller.config
            requested_speed = config.max_descent_speed
            distance_limit = config.max_distance
            contact_threshold = config.cell_R_increase
            (signed_descent, maximum_descent, resistance_increase,
             distance_reached, contact_candidate) = self.calculate(
                self.spear, current_z_um=current_z_um, start_z_um=self.start_position[2],
                previous_descent_um=self.spear_descent_um, descent_direction=self.spear_direction,
                max_distance=distance_limit, delayed_resistance_mohm=delayed_resistance,
                baseline_resistance_mohm=self.controller.first_res, cell_R_increase=contact_threshold)
            self.spear_descent_um = maximum_descent
            self.distance_reached = distance_reached
            self.contact_candidate = contact_candidate
            # This dictionary is only the deck record; decisions use the variables above.
            self.controller.observation_helper.record_calculations(self, observation, {
                "signed_descent_um": signed_descent, "maximum_net_descent_um": maximum_descent,
                "resistance_increase_mohm": resistance_increase,
                "delayed_resistance_mohm": delayed_resistance,
                "baseline_resistance_mohm": self.controller.first_res,
                "contact_candidate": contact_candidate, "distance_reached": distance_reached,
                "cell_lost": self.cell_lost, "max_distance": distance_limit,
                "cell_R_increase": contact_threshold, "max_descent_speed": requested_speed,
                "applied_descent_speed_um_s": self.applied_descent_speed_um_s,
                "descent_direction": self.spear_direction})
            keep_monitoring = not (contact_candidate or distance_reached or self.cell_lost is True)
            yield keep_monitoring, False, requested_speed
            self.controller.sleep(0.04)

    def search(self):
        while True:
            observation = self.observe(
                fields=["manipulator_position", "resistance", "stage_positions", "deep_learning"],
                raw_measurements=True, cell=self.cell)
            self.last_observation = observation
            self.resistance_readings.append(self.latest_resistance)
            self.latest_resistance = observation["resistance"]
            margin = self.controller.config.hunt_search_margin
            search_distance = self.calculate(
                self.search, previous_descent_um=self.spear_descent_um, hunt_search_margin=margin)
            self.controller.observation_helper.record_calculations(self, observation, {
                "search_distance_um": search_distance, "hunt_search_margin": margin,
                "cell_found": None, "focus_returned": None,
                "return_to_cell_focus": self.return_to_cell_focus})
            # No focus acceptance exists yet; act blocks this unavailable movement.
            yield False, False, None
            self.controller.sleep(0.04)

    def scan(self):
        while True:
            observation = self.observe(
                fields=["manipulator_position", "resistance", "stage_positions", "deep_learning"],
                raw_measurements=True, cell=self.cell)
            self.last_observation = observation
            self.resistance_readings.append(self.latest_resistance)
            self.latest_resistance = observation["resistance"]
            pipette_focused = self.calculate(self.scan)
            self.controller.observation_helper.record_calculations(
                self, observation, {"pipette_focused": pipette_focused})
            yield False, False, None
            self.controller.sleep(0.04)

    def shift(self):
        while True:
            observation = self.observe(
                fields=["manipulator_position", "resistance", "stage_positions", "deep_learning"],
                raw_measurements=True, cell=self.cell)
            self.last_observation = observation
            self.resistance_readings.append(self.latest_resistance)
            self.latest_resistance = observation["resistance"]
            aligned = self.calculate(self.shift)
            self.controller.observation_helper.record_calculations(
                self, observation, {"aligned": aligned, "shift_cell": self.shift_cell})
            yield False, False, None
            self.controller.sleep(0.04)

    def failure_gate(self, state=None):
        """Report failure conditions without logging, moving, or raising."""
        if isinstance(self.operation_error, RequestedSuccessException):
            return False
        if self.operation_error is not None or self.controller.abort_requested:
            return True
        if self.operation is not None:
            return self.contact_candidate
        return self.distance_reached

    def success_gate(self, observation=None, state=None):
        """Report manual or measured success; decide owns completion."""
        if isinstance(self.operation_error, RequestedSuccessException):
            return True
        if self.controller.success_requested:
            return True
        if self.operation is not None:
            # Contact confirmation remains unfinished.
            return False
        return self._resistance_threshold_reached(
            self.resistance_readings, self.controller.config.cell_R_increase)

    def _resistance_threshold_reached(self, readings, threshold):
        """Original five-reading threshold, separate from Adaptive confirmation."""
        return len(readings) >= 5 and readings[4] - self.controller.first_res >= threshold

    def _isCellDetected(self, lastResDeque, cellThreshold=0.15):
        """Compatibility wrapper for callers outside the phase lifecycle."""
        detected = self._resistance_threshold_reached(lastResDeque, cellThreshold)
        if detected:
            self.controller.calibrated_unit.stop()
        return detected

    def track_cell(self, cell):
        '''
        Track the cell during hunting and return its current position in pixels.
        '''
        # TODO will add another condition to check if cell and pipette have moved away from each other based on the mask and original image.
        if not self.controller.config.track_cell:
            return None

        ai_tracking_enabled = bool(self.controller.calibrated_stage.config.use_ai_features)
        if ai_tracking_enabled:
            position, disp = self.controller.calibrated_stage.get_cell_position(
                cell,
                use_centroid=self.controller.config.use_centroid,
                tracking_mode=self.controller.config.tracking_mode,
                track_max_fast_jump_px=self.controller.config.track_max_fast_jump_px,
            )
            status = {}
            cell_track_helper = getattr(self.controller.calibrated_stage, "cellTrackHelper", None)
            if cell_track_helper is not None:
                status = getattr(cell_track_helper, "last_tracking_status", {}) or {}
            method = status.get("method")
            status_name = status.get("status")
            if position is not None and disp is not None:
                status_label = f"{method}/{status_name}" if method else str(status_name)
                self.controller.info(f"cell tracking: {status_label}")
                self.controller.info(f"cell displacement: {disp} px")
                self.controller.info(f"cell position: {position} px")
                self.controller._last_track_cell_status = status_name
                return position
            else:
                if self.controller._last_track_cell_status != status_name:
                    self.controller.info("lost track of cell")
                    self.controller._last_track_cell_status = status_name
                return None

        if not self.controller._track_cell_ai_disabled_logged:
            self.controller.info(
                "Track-cell is enabled, but calibration.use_ai_features is false; skipping AI cell tracking."
            )
            self.controller._track_cell_ai_disabled_logged = True
        return None


