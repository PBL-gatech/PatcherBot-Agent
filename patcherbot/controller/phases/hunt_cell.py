"""Historical Hunt behavior within the shared phase lifecycle."""

import collections
import sys

from ..errors import AutopatchError
from ..PhaseController import PhaseController


class HuntCellPhase(PhaseController):
    def run(self, cell=None):
        state = None
        try:
            state = self.prepare(cell)
            while True:
                observation = self.observe(state=state)
                calculated = self.calculate(observation, state)
                command = self.decide(calculated, state)
                self.act(command, state)
        finally:
            self.finish(state)

    def prepare(self, cell=None):
        self.failure_gate()
        self.success_gate()
        controller = self.controller
        controller.isrigready()
        if controller.rig_ready == False:
            self.failure_gate(message="Rig not ready for cell hunting")
        if cell is None:
            self.failure_gate(message="No cell given to patch!")
        self.begin_observations()
        self.observation_windows = {}
        self.frame_context = {}
        controller.info("Hunting for cell")
        controller.pressure.set_pressure(controller.config.pressure_near)
        controller.sleep(3)
        return dict(initialized=False, start_position=None,
                    readings=collections.deque(maxlen=5), last_velocity=None,
                    model_prepared=False, mode=None)

    def observe(self, *, state):
        self.failure_gate()
        self.success_gate()
        controller = self.controller
        mode = controller.config.mode
        if state["initialized"]:
            position = super().observe(fields=["manipulator_position"], evidence={})
            self.failure_gate(state, {**position, "mode": mode})
            controller.sleep(0.04)
        observation = super().observe(
            fields=["resistance", "manipulator_position"],
            evidence={}, raw_resistance=True)
        baseline = (controller.first_res if state["initialized"]
                    else controller.resistanceRamp())
        controller.observer.annotate(self, observation, fields={
            "mode": mode, "baseline_resistance": baseline})
        self.failure_gate()
        self.success_gate()
        return observation

    def calculate(self, observation=None, state=None):
        self.failure_gate()
        self.success_gate()
        mode = observation["mode"]
        if mode != self.controller.config.mode:
            return {"observation": observation, "mode": mode, "stale": True}
        if not state["initialized"]:
            state["start_position"] = observation["manipulator_position"]
            self.controller.first_res = observation["baseline_resistance"]
            state["initialized"] = True
            self.controller.info(f"Initial resistance: {self.controller.first_res}")
        state["readings"].append(observation["resistance"])
        ready = len(state["readings"]) >= 5
        delta = (state["readings"][-1] - self.controller.first_res) if ready else None
        travel = None if mode == "Training" else abs(
            observation["manipulator_position"][2] - state["start_position"][2])
        calculated = {"observation": observation, "mode": mode, "stale": False,
                      "window_ready": ready, "resistance_delta": delta,
                      "travel_um": travel}
        self.controller.observer.annotate(self, observation, calculations={
            "window_ready": ready, "resistance_delta": delta, "travel_um": travel})
        return calculated

    def decide(self, observation=None, state=None):
        self.failure_gate()
        self.success_gate()
        calculated = observation
        mode = calculated["mode"]
        command = {"calculated": calculated, "mode": mode}
        if self.awaiting_operator:
            return None
        if calculated["stale"] or mode != self.controller.config.mode:
            return {**command, "kind": "stop"}
        if mode not in ("Manual", "Classic", "Adaptive", "Training", "Agent"):
            self.failure_gate(message=f"Unsupported hunt mode: {mode}")
        complete = self.success_gate(calculated, state)
        if complete or self.goal_event:
            return {**command, "kind": "complete"}
        self.failure_gate(state, calculated)
        if mode not in ("Classic", "Adaptive") and state["last_velocity"] is not None:
            return {**command, "kind": "stop"}
        if not self.action_gate(calculated, state):
            return None
        if mode in ("Classic", "Adaptive"):
            return {**command, "kind": "descend",
                    "value": [0, 0, self.controller.config.max_descent_speed]}
        if mode == "Agent" and not state["model_prepared"]:
            return {**command, "kind": "prepare_model", "model": "hunt"}
        return None

    def act(self, observation=None, state=None):
        self.failure_gate()
        self.success_gate()
        command = observation
        if command is None:
            return
        if not self.action_gate(command["calculated"], state, command=command):
            self.controller.calibrated_unit.stop()
            state["last_velocity"] = None
            return
        kind = command["kind"]
        if kind == "descend":
            velocity = tuple(command["value"])
            if velocity != state["last_velocity"]:
                self.controller.calibrated_unit.absolute_move_group_velocity(command["value"])
                state["last_velocity"] = velocity
        elif kind == "prepare_model":
            if not state["model_prepared"]:
                self.controller.agenthelper.prepare_model(command["model"])
                state["model_prepared"] = True
        elif kind == "stop":
            self.controller.calibrated_unit.stop()
            state["last_velocity"] = None
        elif kind == "complete":
            self.finish(state, success=True, mode=command["mode"])
        state["mode"] = command["mode"]

    def action_gate(self, observation=None, state=None, *, command=None):
        self.failure_gate()
        self.success_gate()
        if observation is None or state is None:
            return False
        mode = observation["mode"]
        if (observation["stale"] or mode != self.controller.config.mode
                or (command is not None and command["mode"] != mode)):
            return False
        kind = None if command is None else command["kind"]
        if kind == "stop":
            return True
        contact = self.success_gate(observation, state) or self.goal_event
        if contact:
            return kind == "complete"
        if kind == "complete":
            return False
        self.failure_gate(state, observation)
        if mode in ("Classic", "Adaptive"):
            return (kind in (None, "descend") and (command is None or
                    command.get("value") == [0, 0, self.controller.config.max_descent_speed]))
        if mode == "Agent":
            return (kind in (None, "prepare_model") and
                    (command is None or command.get("model") == "hunt"))
        return False

    def failure_gate(self, state=None, observation=None, *, message=None):
        self.controller.abort_if_requested()
        if message is not None:
            raise AutopatchError(message)
        if self.awaiting_operator or state is None or observation is None or not state["initialized"]:
            return
        mode = observation["mode"]
        if mode == "Training" or mode != self.controller.config.mode:
            return
        travel = observation.get("travel_um")
        if travel is None:
            travel = abs(observation["manipulator_position"][2] - state["start_position"][2])
        if travel >= int(self.controller.config.max_distance):
            raise AutopatchError(
                "Cell not detected before reaching max hunt distance "
                f"({travel:.1f} um >= {int(self.controller.config.max_distance)} um).")

    def success_gate(self, observation=None, state=None):
        super().success_gate()
        if self.awaiting_operator or observation is None or state is None:
            return False
        if observation["stale"] or observation["mode"] != self.controller.config.mode:
            return False
        threshold = self.controller.config.cell_R_increase
        if ("cell_detected" not in observation
                or observation["cell_detection_threshold"] != threshold):
            observation["cell_detected"] = self._isCellDetected(state["readings"], threshold)
            observation["cell_detection_threshold"] = threshold
            self.controller.observer.annotate(self, observation["observation"], calculations={
                "cell_detected": observation["cell_detected"], "cell_threshold": threshold})
        return super().success_gate(observation["cell_detected"], state)

    def finish(self, state=None, *, success=False, mode=None):
        controller = self.controller
        primary_error, stop_error = sys.exc_info()[1], None
        for device in (controller.calibrated_stage, controller.calibrated_unit, controller.microscope):
            try:
                device.stop()
            except BaseException as error:
                if stop_error is None:
                    stop_error = error
        if primary_error is not None:
            return
        if stop_error is not None:
            raise stop_error
        if success:
            controller.info("Cell Detected")
            self.goal_event = False
            if not self.awaiting_operator:
                self.complete_success()

    def _isCellDetected(self, lastResDeque, cellThreshold=0.15):
        if len(lastResDeque) < 5:
            return False
        delta = lastResDeque[4] - self.controller.first_res
        detected = cellThreshold <= delta
        if detected:
            self.controller.info(f"Cell detected: {detected}; resistance: {delta}")
            self.controller.calibrated_unit.stop()
        return detected