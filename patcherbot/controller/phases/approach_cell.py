"""Cell-type-dependent approach from the common localization distance."""

from ..errors import AutopatchError
from ..PhaseController import PhaseController


class ApproachCellPhase(PhaseController):
    def run(self, cell=None):
        """Prepare, choose, and execute one approach, then signal completion."""
        state = {"cell": cell}
        self.prepare(state)
        command = self.decide(state=state)
        completed = self.act(command)
        if self.success_gate(completed, state):
            self.complete_success()

    def prepare(self, state=None):
        """Set the existing movement speeds for the approach."""
        self.controller.calibrated_stage.set_max_speed(self.controller.config.max_locate_speed)
        self.controller.calibrated_unit.set_max_speed(self.controller.config.max_locate_speed)

    def decide(self, observation=None, state=None):
        """Choose approach distances from the cell type without moving hardware."""
        config = self.controller.config
        command = {
            "cell": state["cell"],
            "cell_distance": config.slice_start_distance,
            "use_centroid": config.use_centroid,
            "move_distance": None,
            "clear_distance": None,
        }
        if config.cell_type == "Plate":
            command["cell_distance"] = config.cell_distance
            command["move_distance"] = config.slice_start_distance - config.cell_distance
        elif config.cell_type_toggle and config.cell_type == "Slice":
            command["clear_distance"] = config.cell_distance
        return command

    def act(self, command):
        """Execute the selected approach without choosing cell-type behavior."""
        cell = command["cell"]
        if command["move_distance"] is not None:
            self.controller.move_group_down(command["move_distance"])
            self.controller.sleep(0.1)
            self.controller.fine_calibrate_pipette()

        self.controller.amplifier.start_patch()
        self.controller.align(cell, command["cell_distance"], command["use_centroid"])
        if command["clear_distance"] is not None:
            self.controller.clear_to_cell(cell)
            self.controller.align(cell, command["clear_distance"], command["use_centroid"])
        return True

    def success_gate(self, observation=None, state=None):
        """Restore speeds and report success after the approach returns normally."""
        if observation is not True:
            return False
        self.controller.calibrated_stage.set_max_speed(10000)
        self.controller.calibrated_unit.set_max_speed(100000)
        self.controller.info("Approached Cell")
        return True

    def failure_gate(self, state=None):
        """Apply the existing rig and cell checks before clearing."""
        self.controller.isrigready()
        if self.controller.rig_ready == False:
            raise AutopatchError("Rig not ready for clearing to cell")
        if state["cell"] is None:
            raise AutopatchError("No cell given to patch!")

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
        '''
        Clears the pipette to the cell by moving down while checking resistance
        '''
        self.controller.info("Clearing to cell")
        self.failure_gate({"cell": cell})

        # if a slice, push pipette into slice from above surface, just about 20um above cell of interest

        if self.controller.config.cell_type_toggle and self.controller.config.cell_type == "Slice":
            self.controller.info("Moving pipette to slice position")
            speed = [0,0,self.controller.config.max_clearing_speed]
            cell_hover_pos =  self.controller.config.cell_distance - self.controller.config.slice_start_distance
            # move the stage up to the cell hover position
            self.controller.microscope.relative_move(-self.controller.config.cell_distance)
            start_pos = self.controller.calibrated_unit.position()
            self.controller.calibrated_unit.absolute_move_group_velocity(speed)
            self.controller.info(f"Cell hover position: {cell_hover_pos} um")
            while start_pos[2] - self.controller.calibrated_unit.position()[2] > cell_hover_pos and not self.controller.abort_requested:
                self.controller.sleep(0.1)
            self.controller.calibrated_unit.stop()

