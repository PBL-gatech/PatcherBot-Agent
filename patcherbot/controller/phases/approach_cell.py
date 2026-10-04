"""Cell-type-dependent approach from the common localization distance."""

import sys
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
        return {"cell": cell, "fields": []}


    def decide(self, observation=None, state=None):
        self.failure_gate()
        self.success_gate()
        if self.awaiting_operator or state.get("finished", False):
            return None
        config = self.controller.config
        command = {"cell": state["cell"],
                   "route_token": (config.cell_type, bool(config.cell_type_toggle) if config.cell_type != "Plate" else None)}
        if config.cell_type == "Plate":
            return {**command, "route": "plate"}
        if config.cell_type_toggle and config.cell_type == "Slice":
            return {**command, "route": "slice"}
        return {**command, "route": "initial"}


    def act(self, observation=None, state=None):
        self.failure_gate()
        self.success_gate()
        command = observation
        if command is None:
            self.controller.sleep(0.1)
            return
        if not self.action_gate(command):
            raise AutopatchError("Cell type changed before approach dispatch")
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
        if not self.action_gate(command):
            raise AutopatchError("Cell type changed during approach")
        controller.amplifier.start_patch()
        distance = first_alignment_distance if command["route"] == "plate" else config.slice_start_distance
        controller.align(cell, distance, config.use_centroid)
        if command["route"] == "slice":
            self.failure_gate()
            self.success_gate()
            if not self.action_gate(command):
                raise AutopatchError("Cell type changed during approach")
            controller.clear_to_cell(cell)
            self.failure_gate()
            self.success_gate()
            if not self.action_gate(command):
                raise AutopatchError("Cell type changed during approach")
            controller.align(cell, config.cell_distance, config.use_centroid)
        state["fields"] = ["manipulator_position", "resistance"]
        self.finish(state, success=True)


    def action_gate(self, observation=None, state=None):
        config = self.controller.config
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
        try:
            self.controller.calibrated_unit.stop()
        except Exception as stop_error:
            if primary_error is None:
                raise
            if hasattr(primary_error, "add_note"):
                primary_error.add_note(f"Pipette stop also failed: {stop_error}")
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
