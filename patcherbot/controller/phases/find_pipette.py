"""Find the pipette using the existing phase lifecycle and motion conventions."""
import sys
import numpy as np
from ..PhaseController import PhaseController

class FindPipettePhase(PhaseController):
    def run(self, cell=None):
        try:
            state = self.prepare(cell)
            while True:
                observation = self.observe(fields=["manipulator_position", "pipette_image_xy",
                    "pipette_defocus_um", "stage_positions", "camera_image", "resistance"])
                calculated = self.calculate(observation, state)
                command = self.decide(calculated, state)
                self.act(command, state)
        finally:
            self.finish()

    def prepare(self, cell=None):
        controller = self.controller
        self.begin_observations()
        controller.info("Finding pipette")
        if controller.config.mode == "Agent":
            controller.agenthelper.prepare_model("find_pipette", allow_goal_placeholders=True)
        camera = controller.calibrated_stage.camera
        width, height = getattr(camera, "width", None), getattr(camera, "height", None)
        center = np.array([round(width / 2) if isinstance(width, (int, float)) else 640,
                           round(height / 2) if isinstance(height, (int, float)) else 640,
                           0.0], dtype=float)
        goal = center.astype(np.float32) if controller.goal_needed else None
        if goal is not None and controller.goal_random:
            goal[:2] += np.random.randint(-300, 301, size=2)
            for axis, limit in enumerate((width, height)):
                if isinstance(limit, (int, float)) and limit > 0:
                    goal[axis] = np.clip(goal[axis], 0, max(int(limit) - 1, 0))
        target = center if goal is None else goal.astype(float)
        return {
            "goal": goal,
            "target": target,
            "marker": tuple(int(round(value)) for value in target[:2]),
            "tolerance_um": 2.0,
        }

    def observe(self, *, fields=None, num_measurements=None, interval=None, evidence=None):
        self.failure_gate()
        self.success_gate()
        observation = super().observe(fields=fields, num_measurements=num_measurements,
                                      interval=interval, evidence=evidence)
        self.failure_gate()
        self.success_gate()
        return observation

    def calculate(self, observation=None, state=None):
        self.failure_gate()
        self.success_gate()
        try:
            point = np.r_[observation["pipette_image_xy"],
                          observation["pipette_defocus_um"]].astype(float)
        except (TypeError, ValueError):
            point = np.array([])
        if point.shape != (3,) or not np.isfinite(point).all():
            return None
        scales = np.asarray(self.controller.calibrated_unit.pixel_per_um(), dtype=float)
        if scales.size < 2 or not np.isfinite(scales[:2]).all() or (scales[:2] == 0).any():
            self.failure_gate("Invalid pipette pixel-per-um calibration")
        error = state["target"] - point
        goal_error_um = float(np.sqrt(np.mean(np.r_[error[:2] / scales[:2], error[2]] ** 2)))
        return {"observation": observation, "point": point, "error": error,
                "goal_error_um": goal_error_um,
                "sleep_time": .001 + .004 * float(np.clip(goal_error_um / 20.0, 0, 1))}

    def decide(self, observation=None, state=None):
        self.failure_gate()
        self.success_gate()
        controller = self.controller
        if self.awaiting_operator:
            return None
        if observation is None:
            return {"rejection_reason": "Invalid pipette coordinates; waiting for next observation",
                    "sleep_time": .005}
        mode = controller.config.mode
        if mode not in ("Manual", "Classic", "Adaptive", "Training", "Agent"):
            self.failure_gate(f"Unsupported find_pipette mode: {mode}")
        point = observation["point"]
        error = observation["error"]
        sleep_time = observation["sleep_time"]
        controller.info(f"Goal error (um): {observation['goal_error_um']}")
        if state["goal"] is not None:
            marker_options = {"radius": 15} if mode == "Agent" else {}
            controller.calibrated_stage.camera.show_circle(
                point=state["marker"], color=(255, 255, 255),
                show_center=False, **marker_options)
        complete = self.success_gate(observation["goal_error_um"], state["tolerance_um"])
        if complete or self.goal_event:
            self.finish(success=True, mode=mode)
            return None
        if mode in ("Manual", "Classic", "Adaptive", "Training"):
            xy_um = controller.calibrated_unit.pixels_to_um_relative([error[0], error[1], 0.0])
            return {"move_um": np.array([xy_um[0], xy_um[1], error[2]], dtype=float),
                    "kind": "direct", "mode": mode, "sleep_time": sleep_time}
        helper = controller.agenthelper
        if getattr(helper, "model_type", None) != "find_pipette" or helper.agent is None:
            helper.prepare_model("find_pipette", allow_goal_placeholders=True)
        velocity_prediction = bool(getattr(controller, "velocity_prediction", False))
        agent_goal = None if state["goal"] is None else controller.agenthelper.agent.prepare_coordinate_goal(
            state["goal"], observation["observation"]["camera_image"])
        action = controller.agenthelper.run_inference(
            observation=observation["observation"], goal=agent_goal, is_demo=False)
        try:
            prediction = np.asarray(action, dtype=object)
            if prediction.ndim != 1 or prediction.size not in (2, 3):
                raise ValueError("expected two or three coordinates")
            xy_px = np.asarray(prediction[:2], dtype=float)
            z_supplied = prediction.size == 3 and prediction[2] is not None
            z_um = float(prediction[2]) if z_supplied else (
                0.0 if velocity_prediction else point[2])
            if not np.isfinite(xy_px).all() or not np.isfinite(z_um):
                raise ValueError("non-finite coordinates")
        except (TypeError, ValueError) as exc:
            return {"rejection_reason": f"Invalid policy action: {exc}",
                    "sleep_time": sleep_time}
        if velocity_prediction:
            xy_um = controller.calibrated_unit.pixels_to_um_relative([-xy_px[0], -xy_px[1], 0.0])
            kind = "agent_velocity"
        else:
            screen_target = point[:2] + xy_px
            camera = controller.calibrated_stage.camera
            width, height = getattr(camera, "width", None), getattr(camera, "height", None)
            if width is not None and height is not None and not (
                    0 <= screen_target[0] < width and 0 <= screen_target[1] < height):
                return {"rejection_reason": f"Predicted point not on screen: {screen_target}",
                        "sleep_time": .04}
            xy_um = controller.calibrated_unit.pixels_to_um_relative([xy_px[0], xy_px[1], 0.0])
            kind = "agent_displacement"
        return {"move_um": np.array([xy_um[0], xy_um[1], z_um], dtype=float),
                "kind": kind, "mode": mode, "sleep_time": sleep_time}

    def act(self, observation=None, state=None):
        self.failure_gate()
        self.success_gate()
        if observation is None:
            if self.awaiting_operator:
                self.controller.sleep(0.1)
            return
        command = observation
        unit = self.controller.calibrated_unit
        if "rejection_reason" in command:
            unit.stop()
            self.controller.warning(command["rejection_reason"])
            self.controller.sleep(command["sleep_time"])
            return
        mode = self.controller.config.mode
        kind = command["kind"]
        if command["mode"] != mode or (mode == "Agent" and
                (kind == "agent_velocity") != bool(getattr(self.controller, "velocity_prediction", False))):
            unit.stop()
            self.controller.sleep(command["sleep_time"])
            return
        speed_um_s = abs(float(getattr(self.controller, "find_pipette_velocity_speed_um_s", 200.0)))
        if not np.isfinite(speed_um_s) or speed_um_s == 0:
            self.failure_gate("find_pipette_velocity_speed_um_s must be finite and non-zero")
        move_um = command["move_um"]
        if not self.action_gate(move_um):
            unit.stop()
            self.controller.warning("Invalid or zero movement; waiting for next observation")
        elif kind == "agent_velocity":
            unit.relative_move_group_velocity(move_um.tolist())
        elif kind == "direct" and speed_um_s < 1000.0:
            velocity = -np.asarray(unit.velocity_position_control(move_um, speed_um_s), dtype=float)
            if not np.isfinite(velocity).all():
                raise ValueError("Invalid converted velocity")
            unit.absolute_move_group_velocity(velocity.tolist())
        else:
            unit.relative_move_group(move_um.tolist())
            unit.wait_until_still()
        self.controller.sleep(command["sleep_time"])

    def action_gate(self, observation=None, state=None):
        return np.shape(observation) == (3,) and np.isfinite(observation).all() and np.linalg.norm(observation) != 0

    def failure_gate(self, state=None):
        self.controller.abort_if_requested()
        if state is not None:
            raise ValueError(state)

    def success_gate(self, observation=None, state=None):
        goal = None if observation is None else observation <= state
        return super().success_gate(goal, state)

    def finish(self, *, success=False, mode=None):
        """Stop on every exit without replacing an exception already in flight."""
        primary_error = sys.exc_info()[1]
        try:
            self.controller.calibrated_unit.stop()
        except Exception as stop_error:
            if primary_error is None:
                raise
            if hasattr(primary_error, "add_note"):
                primary_error.add_note(f"Pipette stop also failed: {stop_error}")
        if success:
            self.controller.info("Pipette found")
            self.goal_event = False
            if not self.awaiting_operator:
                self.complete_success()
