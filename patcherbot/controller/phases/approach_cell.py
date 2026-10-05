"""Cell-type-dependent approach from the common localization distance."""

import sys
import numpy as np
from ..errors import AutopatchError
from ..PhaseController import PhaseController


class ApproachCellPhase(PhaseController):
    def run(self, cell=None):
        state = None
        try:
            state = self.prepare(cell)
            while True:
                observation = self.observe(fields=state["fields"], evidence={})
                calculated = self.calculate(observation, state)
                command = self.decide(calculated, state)
                self.act(command, state)
        finally:
            self.finish(state)


    def prepare(self, cell=None):
        self.failure_gate()
        self.success_gate()
        self.begin_observations()
        self._adaptive = self.controller.config.mode == "Adaptive"
        self._vision = False
        return {"cell": cell, "fields": [], "step": 0}

    def observe(self, **kwargs):
        kwargs["evidence"] = None if self._vision else kwargs.get("evidence")
        observation = super().observe(**kwargs)
        if self._vision:
            tip = observation["deep_learning"].get("pipette_position")
            self.controller.calibrated_stage.camera.show_circles([] if tip is None else [tip], color=(0, 255, 0), duration=0.2)
        return observation

    def calculate(self, observation=None, state=None):
        if not self._adaptive:
            return super().calculate(observation, state)
        state["observation"] = observation
        if self._vision:
            observation["cell_focus_error"] = observation["stage_positions"][2] - state["cell_focus_z"]
            if state["step"] == 1:
                expected_xy = (self.controller.calibrated_stage.M @ observation["stage_positions"][:2]
                               + self.controller.calibrated_stage.r0)[:2] - state["cell"][0][:2]
                observation["cell_detection"] = min(observation["deep_learning"].get("cell_detections") or [],
                    key=lambda cell: sum((cell[:2] - expected_xy) ** 2), default=None)
            elif state["step"] == 2:
                depth = observation["deep_learning"].get("pipette_focus")
                observation["defocus_error"] = float("inf") if depth is None else abs(depth)
                observation["pipette_z_error"] = abs(observation["stage_positions"][2] - state["pipette_z"])
                observation["up_speed"] = self.controller.microscope.up_direction * self.controller.config.max_locate_speed
                observation["scan_speed"] = observation["up_speed"] * (np.sign(depth) if depth is not None and np.isfinite(depth) else 1)
            elif state["step"] == 3:
                observation["spear_target"] = (observation["manipulator_position"][2]
                    + self.controller.config.slice_start_distance - state["goal_distance"])
            elif state["step"] == 4:
                observation["spear_remaining"] = state["spear_target"] - observation["manipulator_position"][2]
                observation["pipette_plane"] = state["pipette_z"] + observation["manipulator_position"][2] - state["spear_start_z"]
        observation["spear_speed"] = abs(self.controller.config.max_clearing_speed)
        observation["speed"] = self.controller.microscope.up_direction * self.controller.config.max_locate_speed * (-1 if state["step"] in (0, 2, 4) else 1)
        return observation


    def decide(self, observation=None, state=None):
        self.failure_gate()
        self.success_gate()
        if self.awaiting_operator or state.get("finished", False):
            return None
        config = self.controller.config
        command = {"cell": state["cell"],
                   "route_token": (config.cell_type, bool(config.cell_type_toggle) if config.cell_type != "Plate" else None)}
        if config.mode == "Adaptive":
            step = state["step"]
            confidence = observation.get("deep_learning", {}).get("confidence", {})
            cell = observation.get("cell_detection")
            pipette_confident = (step == 2 and (confidence.get("pipette_detector") or 0) >= self.controller.calibrated_unit.pipetteCalHelper.pipetteDetector.adapter.threshold
                                 and (confidence.get("pipette_focuser") or 0) >= 0.50)  # Provisional Z-head cutoff.
            ready = (step == 0 or (step == 1 and observation["cell_focus_error"] >= 0 and cell is not None
                and cell[2] >= self.controller.calibrated_stage.cellDetectHelper.cellDetector.conf_threshold)
                or (pipette_confident and observation["pipette_z_error"] <= 5 and observation["defocus_error"] <= 2.0)
                or (step == 3 and observation["cell_focus_error"] >= 0)
                or (step == 4 and observation["spear_remaining"] <= 0.5))
            if step == 2 and not ready:
                speed = observation["scan_speed"] if pipette_confident and observation["pipette_z_error"] <= 5 else observation["up_speed"]
                return {**command, "route": "adaptive", **self.scan(speed)}
            if not ready:
                return None
            if step == 0:
                move = self.search(observation["speed"])
                move["goal_distance"] = config.cell_distance if config.cell_type == "Plate" or (config.cell_type == "Slice" and config.cell_type_toggle) else config.slice_start_distance
            elif step in (1, 2):
                move = self.shift("cell" if step == 1 else "pipette", self.scan(observation["speed"]))
            elif step == 3:
                move = ({"method": "complete"} if state.get("speared") else
                        self.spear(observation["spear_target"], observation["spear_speed"]))
            else:
                move = {**self.search(observation["speed"]), "pipette_plane": observation["pipette_plane"]}
            return {**command, "route": "adaptive", **move}
        if config.cell_type == "Plate":
            return {**command, "route": "plate"}
        if config.cell_type_toggle and config.cell_type == "Slice":
            return {**command, "route": "slice"}
        return {**command, "route": "initial"}


    def act(self, observation=None, state=None):
        self.failure_gate()
        self.success_gate()
        command = observation
        if (command is not None or self._vision) and not self.action_gate(command, state):
            raise AutopatchError("Approach action is not permitted")
        if command is None:
            self.controller.sleep(0.1)
            return
        if command["route"] == "adaptive" and self._vision:
            if command["method"] != "scan":
                self.controller.microscope.stop()
            if command["method"] == "shift":
                if command["target"] == "cell":
                    state["cell_focus_z"] = state["observation"]["stage_positions"][2]
                    self.controller.calibrated_stage.center_on_cell(state["cell"], use_centroid=self.controller.config.use_centroid)
                else:
                    state["pipette_z"] = state["observation"]["stage_positions"][2]
                    self.controller.calibrated_unit.center_pipette()
                state["step"] += 1
                command = command["next_move"]
            if command["method"] in ("search", "scan"):
                if state["step"] == 4:
                    self.controller.calibrated_unit.stop()
                    state.update(step=1, speared=True, pipette_z=command["pipette_plane"])
                self.controller.microscope.absolute_move_velocity(command["speed"])
                self.controller.sleep(0.1)
            elif command["method"] == "spear":
                state.update(step=4, spear_target=command["target"], spear_start_z=state["observation"]["manipulator_position"][2])
                self.controller.calibrated_unit.absolute_move(command["target"], axis=2, speed=command["speed"])
            else:
                self.finish(state, success=True)
            return
        controller = self.controller
        config = controller.config
        controller.calibrated_stage.set_max_speed(config.max_locate_speed)
        controller.calibrated_unit.set_max_speed(config.max_locate_speed)
        cell = command["cell"]
        if command["route"] == "plate":
            first_alignment_distance = config.cell_distance
            descent_distance = config.slice_start_distance - first_alignment_distance
            controller.move_group_down(descent_distance)
            controller.sleep(0.1)
            controller.fine_calibrate_pipette()
        self.failure_gate()
        self.success_gate()
        if not self.action_gate(command, state):
            raise AutopatchError("Cell type changed during approach")
        controller.amplifier.start_patch()
        if command["route"] == "adaptive":
            state.update(step=1, cell_focus_z=cell[0][2], goal_distance=command["goal_distance"],
                         fields=["stage_positions", "deep_learning", "manipulator_position"],
                         pipette_z=controller.microscope.position() / controller.calibrated_unit.config.microscope_units_per_um)
            self._vision = True
            controller.microscope.absolute_move_velocity(command["speed"])
            return
        distance = first_alignment_distance if command["route"] == "plate" else config.slice_start_distance
        controller.align(cell, distance, config.use_centroid)
        if command["route"] == "slice":
            self.failure_gate()
            self.success_gate()
            if not self.action_gate(command, state):
                raise AutopatchError("Cell type changed during approach")
            controller.clear_to_cell(cell)
            self.failure_gate()
            self.success_gate()
            if not self.action_gate(command, state):
                raise AutopatchError("Cell type changed during approach")
            controller.align(cell, config.cell_distance, config.use_centroid)
        state["fields"] = ["manipulator_position", "resistance"]
        self.finish(state, success=True)


    def search(self, speed):
        return {"method": "search", "speed": speed}

    def shift(self, target, next_move):
        return {"method": "shift", "target": target, "next_move": next_move}

    def scan(self, speed):
        return {"method": "scan", "speed": speed}

    def spear(self, target, speed):
        return {"method": "spear", "target": target, "speed": speed}

    def action_gate(self, observation=None, state=None):
        config = self.controller.config
        if self._vision or (observation and observation["route"] == "adaptive"):
            low, high = sorted([state.get("cell_focus_z", state["cell"][0][2]), state.get("pipette_z", state["cell"][0][2])])
            if observation and observation.get("method") == "spear":
                if not (observation["speed"] > 0 and np.isfinite(observation["target"])
                        and observation["target"] >= state["observation"]["manipulator_position"][2]):
                    return False
            return config.mode == "Adaptive" and (not self._vision or (
                not state["observation"]["deep_learning"].get("stale", True)
                and low - 5 <= state["observation"]["stage_positions"][2] <= high + 5))
        return observation["route_token"] == (config.cell_type, bool(config.cell_type_toggle) if config.cell_type != "Plate" else None)


    def failure_gate(self, state=None):
        self.controller.abort_if_requested()
        if state is not None:
            self.controller.isrigready()
            if self.controller.rig_ready is False:
                raise AutopatchError("Rig not ready for clearing to cell")
            if state["cell"] is None:
                raise AutopatchError("No cell given to patch!")


    def success_gate(self, observation=None, state=None):
        goal = None if observation is None else observation is True
        return super().success_gate(goal, state)


    def finish(self, state=None, *, success=False):
        if state is not None and state.get("finished", False):
            return
        primary_error = sys.exc_info()[1]
        stop_error = None
        devices = [self.controller.calibrated_unit]
        if state is not None and self._vision:
            devices.extend([self.controller.calibrated_stage, self.controller.microscope])
        for device in devices:
            try:
                device.stop()
            except Exception as error:
                if primary_error is not None:
                    if hasattr(primary_error, "add_note"):
                        primary_error.add_note(f"Approach stop also failed: {error}")
                elif stop_error is None:
                    stop_error = error
        if stop_error is not None:
            raise stop_error
        if state is not None:
            state["finished"] = True
        if success:
            self.controller.calibrated_stage.set_max_speed(10000)
            self.controller.calibrated_unit.set_max_speed(100000)
            self.controller.info("Approached Cell")
            self.success_gate(True, state)
            self.goal_event = False
            if not self.awaiting_operator:
                self.complete_success()

    def align(self, cell, cell_distance, use_centroid):
        '''
        Aligns the pipette to the cell using microscope imaging
        '''
        self.controller.info("Aligning pipette to cell using imaging")
        cell_pos, _, _ = cell

        self.controller.microscope.move_to_floor()
        self.controller.microscope.wait_until_still()
        z_pos = self.controller.microscope.position() / self.controller.calibrated_unit.config.microscope_units_per_um
        zdistleft = z_pos - cell_pos[2]
        self.controller.microscope.relative_move(-zdistleft)
        self.controller.microscope.wait_until_still()

        if self.controller.config.cell_type_toggle:
            self.controller.info("centering on cell")
            self.controller.calibrated_stage.center_on_cell(cell,use_centroid)
            self.controller.calibrated_stage.wait_until_still()
            self.controller.info(f"correcting pipette position, moving microscope by {zdistleft} um")
            self.controller.microscope.relative_move(-cell_distance)
            self.controller.microscope.wait_until_still()
            self.controller.calibrated_unit.center_pipette()
            self.controller.calibrated_unit.wait_until_still()
            self.controller.microscope.relative_move(cell_distance)
            self.controller.microscope.wait_until_still()


    def clear_to_cell(self, cell):
        motion_started = False
        try:
            self.failure_gate({"cell": cell})
            self.success_gate()
            controller = self.controller
            config = controller.config
            controller.info("Clearing to cell")
            if not (config.cell_type_toggle and config.cell_type == "Slice"):
                return
            controller.info("Moving pipette to slice position")
            cell_hover_pos = config.cell_distance - config.slice_start_distance
            controller.microscope.relative_move(-config.cell_distance)
            start_pos = controller.calibrated_unit.position()
            motion_started = True
            controller.calibrated_unit.absolute_move_group_velocity([0, 0, config.max_clearing_speed])
            controller.info(f"Cell hover position: {cell_hover_pos} um")
            while start_pos[2] - controller.calibrated_unit.position()[2] > cell_hover_pos:
                self.failure_gate()
                self.success_gate()
                controller.sleep(0.1)
        finally:
            if motion_started:
                self.finish()
