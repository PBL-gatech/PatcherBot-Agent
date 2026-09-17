"""Implementation of the hunt_cell patch phase."""

import collections
import math

from ..errors import AutopatchError

from ..PhaseController import PhaseController


class HuntCellPhase(PhaseController):
    def run(self, cell=None):
        """Coordinate preparation, gated actions, and hunt completion."""
        state = {"cell": cell, "started": False, "cell_detected": False,
                 "termination_reason": None}
        self.prepare(state)
        try:
            command = self.decide(state=state)
            if self.action_gate(command, state):
                self.act(command, state)

            state["cell_detected"] = self.controller._isCellDetected(
                lastResDeque=state["resistance_readings"],
                cellThreshold=self.controller.config.cell_R_increase,
            )
            last_training_threshold_status = state["cell_detected"]
            if state["training_mode"]:
                self.controller.info(
                    f"Training mode: resistance threshold achieved: {state['cell_detected']}"
                )

            while not state["cell_detected"] and not self.controller.abort_requested:
                state["current_position"] = self.controller.calibrated_unit.position()
                state["termination_reason"] = self.failure_gate(state)
                if state["termination_reason"] is not None:
                    break
                observation = self.observe() if state["agent_mode"] else None
                command = self.decide(observation, state)
                if not self.action_gate(command, state):
                    break
                self.act(command, state)
                self.controller.sleep(0.04)
                # Preserve the existing one-sample delay in the detection buffer.
                state["resistance_readings"].append(state["latest_resistance"])
                state["latest_resistance"] = self.controller.daq.resistance()
                state["cell_detected"] = self.controller._isCellDetected(
                    lastResDeque=state["resistance_readings"],
                    cellThreshold=self.controller.config.cell_R_increase,
                )
                if state["training_mode"] and state["cell_detected"] != last_training_threshold_status:
                    self.controller.info(
                        f"Training mode: resistance threshold achieved: {state['cell_detected']}"
                    )
                    last_training_threshold_status = state["cell_detected"]
        finally:
            self.controller.calibrated_stage.stop()
            self.controller.calibrated_unit.stop()
            self.controller.microscope.stop()
        if state["cell_detected"]:
            self.controller.info("Cell Detected")
            if state["training_mode"]:
                self.controller.info("Training mode: waiting for manual Success or Abort.")
                self.wait_for_manual_completion()
            self.complete_success()
        elif state["termination_reason"] is not None:
            self.controller.info(state["termination_reason"])
            raise AutopatchError(state["termination_reason"])
        elif self.controller.abort_requested:
            self.controller.abort_if_requested()

    def prepare(self, state=None):
        """Validate the attempt and establish pressure, position, and baseline."""
        self.controller.info("Hunting for cell")
        self.controller.isrigready()
        if self.controller.rig_ready == False:
            raise AutopatchError("Rig not ready for cell hunting")
        if state["cell"] is None:
            raise AutopatchError("No cell given to patch!")
        self.controller.info(f"Setting pressure to {self.controller.config.pressure_near} mbar")
        self.controller.pressure.set_pressure(self.controller.config.pressure_near)
        self.controller.sleep(3)
        state["resistance_readings"] = collections.deque(maxlen=5)
        state["latest_resistance"] = self.controller.daq.resistance()
        state["resistance_readings"].append(state["latest_resistance"])
        state["start_position"] = self.controller.calibrated_unit.position()
        self.controller.first_res = self.controller.resistanceRamp()
        self.controller.info(f"Initial resistance: {self.controller.first_res}")
        self.controller.info(f"{self.controller.config.mode}: starting hunt")
        state["training_mode"] = self.controller.config.mode == "Training"
        state["agent_mode"] = self.controller.config.mode == "Agent"

        self.controller._track_cell_ai_disabled_logged = False
        self.controller._last_track_cell_status = None
        if self.controller.config.track_cell and bool(self.controller.calibrated_stage.config.use_ai_features):
            cell_track_helper = getattr(self.controller.calibrated_stage, "cellTrackHelper", None)
            if cell_track_helper is not None:
                cell_track_helper.reset_tracking()

        if state["agent_mode"] and not self.controller.abort_requested:
            self.controller.agenthelper.prepare_model("hunt")

        if state["training_mode"]:
            self.controller.info(
                "Training mode: max hunt distance check disabled; waiting for resistance threshold."
            )

    def action_gate(self, observation=None, state=None):
        """Permit the chosen command only while hunting is still active."""
        return (
            observation is not None
            and not state["cell_detected"]
            and not self.controller.abort_requested
        )

    def decide(self, observation=None, state=None):
        """Choose startup motion or infer an Agent velocity alongside tracking."""
        if state["started"]:
            command = {"track_cell": state["cell"]}
            if not state.get("agent_mode", False):
                return command
            action = self.controller.agenthelper.run_inference(observation=observation)
            if action is None:
                return {**command, "stop": True}
            try:
                if len(action) != 3:
                    raise ValueError("expected three components")
                vx, vy, vz = (float(value) for value in action)
            except (TypeError, ValueError) as exc:
                raise AutopatchError("Hunt agent must return three velocity components.") from exc
            if not all(math.isfinite(value) for value in (vx, vy, vz)):
                raise AutopatchError("Hunt agent returned a non-finite velocity.")
            # Match Agent find_pipette: XY px/s -> calibrated um/s; Z is um/s.
            xy = self.controller.calibrated_unit.pixels_to_um_relative([-vx, -vy, 0.0])
            velocity = [float(xy[0]), float(xy[1]), vz]
            if not all(math.isfinite(value) for value in velocity):
                raise AutopatchError("Hunt agent velocity calibration returned non-finite values.")
            command["relative_velocity"] = velocity
            return command
        if self.controller.config.mode == "Classic":
            return {"velocity": [0, 0, self.controller.config.max_descent_speed]}
        elif self.controller.config.mode == "Adaptive":
            return {"velocity": [0, 0, self.controller.config.max_descent_speed]}
        # Agent setup is handled by prepare; manual modes have no startup motion.
        return {}

    def failure_gate(self, state=None):
        """Return a travel-limit failure from the sampled position, or None."""
        if state["training_mode"]:
            return None
        moved_distance = abs(state["current_position"][2] - state["start_position"][2])
        if moved_distance >= int(self.controller.config.max_distance):
            return (
                "Cell not detected before reaching max hunt distance "
                f"({moved_distance:.1f} um >= {float(self.controller.config.max_distance):.1f} um)."
            )
        return None

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


    def act(self, observation=None, state=None):
        """Execute the selected command without choosing mode or checking outcomes."""
        if observation is None:
            return False
        if observation.get("stop"):
            self.controller.calibrated_unit.stop()
        if "relative_velocity" in observation:
            self.controller.calibrated_unit.relative_move_group_velocity(observation["relative_velocity"])
        if "velocity" in observation:
            speed = observation["velocity"]
            self.controller.calibrated_unit.absolute_move_group_velocity(speed)
            self.controller.info(f"moving pipette at: {speed} um/s")
        if state is not None:
            state["started"] = True
        if "track_cell" in observation:
            return self.controller.track_cell(observation["track_cell"])
        return "velocity" in observation or "relative_velocity" in observation

    def success_gate(self, observation=None, state=0.15):
        """Evaluate contact from five buffered readings without moving hardware."""
        if len(observation) < 5:
            return False
        return state <= observation[4] - self.controller.first_res

    def _isCellDetected(self, lastResDeque, cellThreshold=0.15):
        """Preserve the detector helper's immediate stop and argument contract."""
        detected = self.success_gate(lastResDeque, cellThreshold)
        if detected:
            r_delta = lastResDeque[4] - self.controller.first_res
            self.controller.info(f"Cell detected: {detected}; resistance: {r_delta}")
            self.controller.calibrated_unit.stop()
        return detected
