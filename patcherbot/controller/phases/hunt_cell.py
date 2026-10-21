import collections

from ..errors import AutopatchError
from .approach_cell import ApproachCellPhase


class HuntCellPhase(ApproachCellPhase):
    phase_name = "Hunt"

    def run(self, cell=None):
        try:
            self.prepare(cell)
            while True:
                self.failure_gate()
                self.success_gate()
                self.observe()
                self.calculate()
                self.decide()
                self.act()
        finally:
            self.finish()

    def prepare(self, cell=None):
        """Set near pressure and record the starting position and resistance."""
        self.finished = False
        self.started = False
        self.last_velocity = None
        self.cell_detected = False
        self.cell = cell
        self.failure_gate(validate_cell=True)
        self.success_gate()
        self.begin_observations()
        self.controller.info("Hunting for cell")
        self.controller.pressure.set_pressure(self.controller.config.pressure_near)
        self.controller.sleep(3)
        self.observation = self.controller.observe(
            fields=["resistance", "manipulator_position"], phase=self, raw_resistance=True)
        self.resistance = self.observation["resistance"]
        self.readings = collections.deque([self.resistance], maxlen=5)
        self.start_position = self.observation["manipulator_position"]
        self.controller.first_res = self.controller.resistanceRamp()
        self.controller.info(f"Initial resistance: {self.controller.first_res}")
        self.controller.info(f"{self.controller.config.mode}: starting hunt")
        if self.controller.config.mode == "Training":
            self.controller.info("Training mode: max hunt distance check disabled; waiting for resistance threshold.")

    def observe(self, **kwargs):
        """Collect position and resistance at the existing 40 ms interval."""
        self.failure_gate()
        self.success_gate()
        if not self.started:
            return
        self.failure_gate(position=self.controller.calibrated_unit.position())
        self.controller.sleep(0.04)
        self.readings.append(self.resistance)
        self.observation = self.controller.observe(
            fields=["resistance", "manipulator_position"], phase=self, raw_resistance=True, **kwargs)
        self.resistance = self.observation["resistance"]

    def calculate(self):
        pass

    def decide(self):
        """Request descent only in Classic mode while contact is absent."""
        self.velocity = (0, 0, self.controller.config.max_descent_speed) if (
            self.controller.config.mode == "Classic" and not self.cell_detected) else None

    def act(self):
        """Check the gates, then apply a changed descent velocity."""
        self.failure_gate()
        self.success_gate()
        if self.velocity is not None and self.controller.config.mode != "Classic":
            if self.last_velocity is not None:
                self.controller.calibrated_unit.stop()
                self.last_velocity = None
            return
        if self.velocity != self.last_velocity:
            if self.velocity is None:
                self.controller.calibrated_unit.stop()
            else:
                self.controller.calibrated_unit.absolute_move_group_velocity(list(self.velocity))
                self.controller.info(f"moving pipette at: {self.velocity} um/s")
            self.last_velocity = self.velocity
        self.started = True

    def action_gate(self):
        pass

    def failure_gate(self, *, message=None, recover=False, validate_cell=False, position=None):
        """Check aborts, supported modes, setup requirements, and measured travel."""
        self.controller.abort_if_requested()
        if self.controller.config.mode not in ("Training", "Classic", "Manual"):
            raise AutopatchError(f"Hunt mode is not implemented: {self.controller.config.mode}")
        if recover:
            return False
        if message is not None:
            raise AutopatchError(message)
        if validate_cell:
            self.controller.isrigready()
            if self.controller.rig_ready is False:
                raise AutopatchError("Rig not ready for cell hunting")
            if self.cell is None:
                raise AutopatchError("No cell given to patch!")
        if position is None or not self.started or self.controller.config.mode == "Training":
            return
        travel = abs(position[2] - self.start_position[2])
        if travel >= int(self.controller.config.max_distance):
            raise AutopatchError("Cell not detected before reaching max hunt distance "
                f"({travel:.1f} um >= {float(self.controller.config.max_distance):.1f} um).")

    def success_gate(self, goal=None):
        """Handle manual Success and finish when the resistance threshold is reached."""
        self.controller.success_if_requested()
        if goal is None:
            if not self.started:
                return False
            self.cell_detected = self._isCellDetected(
                self.readings, self.controller.config.cell_R_increase)
        else:
            self.cell_detected = bool(goal)
        if self.cell_detected:
            self.finish(success=True)
        return self.cell_detected

    def finish(self, *, success=False):
        """Stop the devices; Training waits for manual confirmation after contact."""
        if self.finished:
            return
        try:
            self.controller.calibrated_stage.stop()
        finally:
            try:
                self.controller.calibrated_unit.stop()
            finally:
                self.controller.microscope.stop()
        self.finished = True
        if success:
            self.controller.info("Cell Detected")
            if self.controller.config.mode == "Training":
                self.finished = False
                self.controller.info("Training mode: waiting for manual Success or Abort.")
                self.wait_for_manual_completion()
            self.complete_success()

    def _isCellDetected(self, lastResDeque, cellThreshold=0.15):
        """Check five resistance readings and stop the pipette when contact is detected."""
        if len(lastResDeque) < 5:
            return False
        r_delta = lastResDeque[4] - self.controller.first_res
        detected = cellThreshold <= r_delta
        if detected:
            self.controller.info(f"Cell detected: {detected}; resistance: {r_delta}")
            self.controller.calibrated_unit.stop()
        return detected
