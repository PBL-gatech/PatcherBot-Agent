"""Approach movements supervised by resistance and target-mask proximity."""
from collections import deque
from copy import deepcopy
from threading import Event, RLock, Thread
import time

import numpy as np

from ..errors import AutopatchError
from .approach_cell import ApproachCellPhase


class HuntCellPhase(ApproachCellPhase):
    phase_name = "Hunt"
    RESISTANCE_STABILITY_SPAN_MOHM = 0.400

    def run(self, cell=None):
        with self.controller.calibrated_unit.pipetteCalHelper.brain_slice_detector():
            try:
                self.prepare(cell)
                while True:
                    self.failure_gate()
                    self.success_gate()
                    if self.finished or self.awaiting_operator:
                        self.controller.sleep(0.04)
                        continue
                    if self.interrupted:
                        self.check_contact()
                        continue
                    self.observe()
                    self.calculate()
                    self.decide()
                    self.act()
            finally:
                self.finish()

    def prepare(self, cell=None):
        self.finished = self.contact_confirmed = self.checking_contact = False
        self.readings = deque(maxlen=5)
        self.last_packet = self.resistance_error = self.started_at = None
        self.last_checked_rise = 0.0
        self.resistance_lock, self.resistance_stop = RLock(), Event()
        self.resistance_worker = None
        self.resistance_tripped = False
        self.interrupted_sweep = None
        self.search_disabled = False
        self.classic_moving = False
        self.descent_origin = self.search_bounds = self.spear_cell_reference = None
        self.progress_reference_gap, self.progress_cycles = None, 0
        if self.controller.config.mode == "Agent":
            raise AutopatchError("Hunt Agent mode is not implemented")
        if cell is None:
            raise AutopatchError("Hunt requires a selected cell")
        super().prepare(cell)
        if self.mode != "Adaptive":
            self.build_helpers()
        for name in ("cell_R_increase", "hunt_resistance_band_percent", "hunt_contact_distance",
                     "hunt_contact_plane_tolerance_um", "max_descent_speed", "pressure_near",
                     "hunt_timeout", "max_distance", "hunt_search_margin"):
            setattr(self.config, name, getattr(self.controller.config, name))
        self.config.hunt_limit_target_radius = getattr(self.controller.config, "hunt_limit_target_radius", True)
        self.config.hunt_target_radius = getattr(self.controller.config, "hunt_target_radius", 50.0)
        self.config.hunt_progress_cycles = getattr(self.controller.config, 'hunt_progress_cycles', 4)
        self.config.hunt_min_progress_percent = getattr(self.controller.config, 'hunt_min_progress_percent', 25.0)
        self.config.max_clearing_speed = self.config.max_descent_speed
        approach = getattr(self.controller, "approach_cell_phase", None)
        if (approach is not None and approach.completed_successfully and approach.cell is not None
                and approach.cell[0] is cell[0]):
            self.cell_observation = deepcopy(approach.cell_observation)
            self.pipette_observation = deepcopy(approach.pipette_observation)
            self.calibration_observation = deepcopy(getattr(approach, 'calibration_observation', None)
                or getattr(approach, 'initial_observation', None))
            self.cell_search_motor_z = getattr(approach, 'cell_search_motor_z', None)
            source_searcher = getattr(approach, 'searcher', None)
            if source_searcher is not None:
                self.searcher.tracker.copy_state_from(source_searcher.tracker)
        self.controller.pressure.set_pressure(self.config.pressure_near)
        self.controller.sleep(3)
        self.baseline = float(self.controller.resistanceRamp())
        if not np.isfinite(self.baseline):
            raise AutopatchError("Hunt needs a finite resistance baseline")
        self.controller.first_res = self.baseline
        self.start_position = np.asarray(self.unit.position()).copy()
        self.initialized = True
        self.operation = (self.search if self.cell_observation is None else self.spear)             if self.mode == "Adaptive" else self.manual_step
        self.started_at = time.monotonic()
        self.resistance_worker = Thread(target=self.watch_resistance, name="HuntResistance", daemon=True)
        self.resistance_worker.start()

    def observe(self, **kwargs):
        options = dict(cell_inference=True, pipette_inference=True, cell_masks=True, raw_resistance=True)
        options.update(kwargs)
        if self.interrupted or self.awaiting_operator:
            options.update(cell_inference=False, pipette_inference=False, cell_masks=False)
        elif self.search_disabled:
            options.update(cell_inference=False, cell_masks=False)
        observation = super().observe(**options)
        if self.mode != "Adaptive":
            self.searcher.tracker.set_reference_mask(observation)
        return observation

    def calculate_geometry(self):
        super().calculate_geometry()
        self.goal_distance = 0.0
        self.hover_z = self.cell_focus_z

    def calculate_progress(self):
        return max(2 * self.depth_tolerance_um, super().calculate_progress())

    def shift(self):
        self.speared = False
        super().shift()

    def spear(self):
        if self.descent_origin is None:
            if not self.check_cycle_progress():
                return
            self.descent_origin = self.observation["pipette"]["raw_position"][2]
            reference = self.searcher.tracker.last_cell_reference_xy
            self.spear_cell_reference = np.array(self.cell[0][:2] if reference is None else reference, copy=True)
        super().spear()
        if self.operation == self.search:
            travelled = max(0.0, self.unit.position()[2] - self.descent_origin)
            self.search_bounds = (self.cell_focus_z,
                self.cell_focus_z + travelled + self.config.hunt_search_margin)

    def check_cycle_progress(self):
        """Stop repeated adaptive cycles that do not close the estimated Z gap."""
        self.calculate_geometry()
        gap = max(0.0, self.cell_focus_z - self.pipette_z)
        if self.progress_reference_gap is None:
            self.progress_reference_gap = gap
            return True
        self.progress_cycles += 1
        if self.progress_cycles < max(1, int(self.config.hunt_progress_cycles)):
            return True
        previous = self.progress_reference_gap
        required = previous * (1.0 - self.config.hunt_min_progress_percent / 100.0)
        if (previous <= 1e-12 and gap <= 1e-12) or gap > required + 1e-12:
            self.stop_motions()
            self.awaiting_operator = True
            self.controller.warning(f'Hunt stopped repeating: Z gap changed from {previous:.3f} to {gap:.3f} um '
                                    f'over {self.progress_cycles} cycles without sufficient progress')
            return False
        self.progress_reference_gap, self.progress_cycles = gap, 0
        return True

    def _has_calibrated_cell(self):
        try:
            point = self.cell[0]
            return len(point) >= 3 and all(np.isfinite(value) for value in point[:3])
        except (AttributeError, IndexError, TypeError, ValueError):
            return False

    def search(self):
        previous = self.cell_observation
        if self.search_disabled:
            if previous is None and not self._has_calibrated_cell():
                self.stop_motions()
                self.awaiting_operator = True
                return
            self.shift_cell, self.operation = True, self.shift
            self.descent_origin = self.search_bounds = None
            return
        result = super().search(bounds=self.search_bounds)
        if self.operation == self.shift:
            if result == 'not_found' or self.cell_observation is None:
                self.stop_motions()
                self.search_disabled = True
                self.cell_observation = previous
                self.searcher.tracker.using_fallback = True
                if previous is None and not self._has_calibrated_cell():
                    self.awaiting_operator = True
                self.controller.warning("Hunt sweep could not identify the cell; retaining prior evidence and disabling mask retries")
            self.descent_origin = self.search_bounds = None
        self.speared = False

    def on_resistance_packet(self, packet):
        """Stop immediately on a >300 kOhm baseline rise near the selected cell."""
        value = packet.get('totalResistance')
        if value is None or not np.isfinite(value):
            return
        with self.resistance_lock:
            if (self.checking_contact or self.resistance_tripped
                    or float(value) <= self.baseline + 0.300 or not self.planar_near()):
                return
            self.resistance_tripped = self.interrupted = True
            operation = self.operation
            self.interrupted_sweep = operation if operation in (self.search, self.scan) else None
        self.stop_motions()
        self.controller.warning(
            f'Hunt resistance stop near target: baseline={self.baseline:.6f} MOhm, '
            f'sample={float(value):.6f} MOhm; checking contact while stopped')

    def sample_resistance(self):
        # Packet identity is stable; the public accessor returns a copy each poll.
        worker = getattr(self.controller.daq, "_daq_acq_thread", None)
        packets = getattr(worker, "_last_data_queue", ())
        packet = packets[-1] if packets else None
        if packet is None or packet is self.last_packet:
            return
        value = packet["totalResistance"]
        if value is None or not np.isfinite(value):
            raise AutopatchError("Hunt resistance sample is invalid")
        self.last_packet = packet
        self.on_resistance_packet(packet)
        self.readings.append(float(value))

    def resistance_rise(self):
        if len(self.readings) != 5 or np.ptp(self.readings) > self.RESISTANCE_STABILITY_SPAN_MOHM + 1e-12:
            return None
        return float(np.mean(self.readings)) - self.baseline

    def watch_resistance(self):
        try:
            while not self.resistance_stop.wait(0.005):
                self.check_limits()
                with self.resistance_lock:
                    if self.checking_contact:
                        continue
                    self.sample_resistance()
                    if self.interrupted:
                        self.stop_motions()
        except Exception as error:
            self.resistance_error = error
            self.interrupted = True
            self.stop_motions()

    def live_contact_geometry(self):
        """Combine live pose and retained visual evidence with calibration fallback."""
        if self.observation is None or self.pipette_anchor is None:
            return float('inf'), float('nan')
        position = np.asarray(self.unit.position(), dtype=float)
        stage = self.controller.observer.get_stage()
        pipette = dict(self.observation['pipette'], raw_position=position,
                       pixel_position=self.project_pipette_tip(self.pipette_anchor, position),
                       pipette_detection=None)
        observation = dict(self.observation, stage=stage, pipette=pipette)
        tip = self.shifter.pipette_target(self, observation)
        distance = self.searcher.tracker.distance_to_mask(tip, observation, self.cell[0],
            self.stage.pixels_to_um_relative, self.config.approach_cell_max_shift)
        if self.pipette_observation is None:
            reference = observation['reference']
            tip_z = reference['home_stage_position'][2] + position[2] - reference['home_position'][2]
        else:
            saved = self.pipette_observation
            tip_z = saved['stage']['raw_position'][2] + position[2] - saved['pipette']['raw_position'][2]
        return distance, tip_z

    def planar_near(self):
        """Planar proximity alone gates a resistance-triggered motion pause."""
        distance, _ = self.live_contact_geometry()
        return bool(np.isfinite(distance) and distance <= self.config.hunt_contact_distance)

    def contact_close(self):
        """Use the current cell-depth estimate, falling back to selected calibration."""
        distance, tip_z = self.live_contact_geometry()
        cell_z = (self.cell_observation['stage']['raw_position'][2]
                  if self.cell_observation is not None else
                  self.observation['reference']['cell_position'][2] if self.observation is not None else None)
        return bool(cell_z is not None and np.isfinite(cell_z) and np.isfinite(tip_z)
            and np.isfinite(distance) and distance <= self.config.hunt_contact_distance
            and abs(cell_z - tip_z) <= self.config.hunt_contact_plane_tolerance_um)

    def failure_gate(self, *, validate_cell=False):
        super().failure_gate(validate_cell=validate_cell)
        if self.resistance_error is not None:
            raise AutopatchError(f"Hunt resistance monitoring failed: {self.resistance_error}")
        if self.started_at is None or self.finished:
            return
        self.check_limits()
    def check_limits(self):
        if self.started_at is None or self.finished:
            return
        if self.mode == "Adaptive" and time.monotonic() - self.started_at >= self.config.hunt_timeout:
            raise AutopatchError("Adaptive Hunt exceeded hunt_timeout")
        if self.mode in ("Manual", "Classic", "Adaptive"):
            if abs(self.unit.position()[2] - self.start_position[2]) >= self.config.max_distance:
                raise AutopatchError("Hunt reached max_distance without confirmed contact")

    def decide(self):
        self.next_action = None if self.finished or self.awaiting_operator else self.operation

    def act(self):
        if self.interrupted:
            self.check_contact()
        elif self.next_action is not None and self.action_gate():
            self.next_action()

    def check_contact(self):
        with self.resistance_lock:
            self.checking_contact = True
            self.stop_motions()
            self.classic_moving = False
        try:
            self.observe(cell_inference=False, pipette_inference=False, cell_masks=False)
            self.calculate_geometry()
            with self.resistance_lock:
                self.readings.clear()
                worker = getattr(self.controller.daq, "_daq_acq_thread", None)
                packets = getattr(worker, "_last_data_queue", ())
                self.last_packet = packets[-1] if packets else None
            while len(self.readings) < 5:
                self.failure_gate()
                self.controller.success_if_requested()
                with self.resistance_lock:
                    self.sample_resistance()
                self.controller.sleep(0.04)
            rise = self.resistance_rise()
            self.contact_confirmed = bool(rise is not None and rise >= self.config.cell_R_increase
                and self.contact_close())
            with self.resistance_lock:
                if rise is not None:
                    self.last_checked_rise = max(self.last_checked_rise, rise)
                released = rise is not None and all(value < self.baseline + 0.300 for value in self.readings)
                self.readings.clear()
                self.resistance_tripped = not self.contact_confirmed and not released
                self.interrupted = self.resistance_tripped
                if self.resistance_tripped:
                    self.controller.warning("Resistance remains elevated or noisy; keeping Hunt motion stopped")
                elif not self.contact_confirmed and self.interrupted_sweep:
                    if self.mode == 'Adaptive':
                        self._skip_cleared_sweep(self.interrupted_sweep)
                    else:
                        self.interrupted = self.awaiting_operator = True
                        self.controller.warning("Contact unconfirmed during a sweep; waiting for the operator")
                elif not self.contact_confirmed:
                    self.controller.warning("Five fresh readings below the resistance threshold; resuming Hunt after the false alarm")
                if not self.resistance_tripped:
                    self.interrupted_sweep = None
        finally:
            self.checking_contact = False

    def _skip_cleared_sweep(self, sweep):
        """Select the alignment fallback after five stable readings clear the stop."""
        self.shift_cell = sweep == self.search
        if self.shift_cell:
            self.search_disabled = self.searcher.tracker.using_fallback = True
            self.descent_origin = self.search_bounds = None
            if self.cell_observation is None and not self._has_calibrated_cell():
                self.interrupted = self.awaiting_operator = True
                self.controller.warning('Hunt has no cell position for fallback; waiting for the operator')
                return
        else:
            self.pipette_observation = None
        self.speared = False
        self.operation = self.shift
        alignment = 'cell' if self.shift_cell else 'pipette'
        self.controller.info(f'Hunt resistance cleared; skipping interrupted {alignment} sweep and aligning from fallback')

    def manual_step(self):
        if self.mode == "Classic" and not self.classic_moving:
            self.unit.absolute_move_group_velocity([0, 0, self.config.max_descent_speed])
            self.classic_moving = True
        self.controller.sleep(0.04)

    def action_gate(self):
        self.failure_gate()
        self.success_gate()
        return not self.finished and not self.awaiting_operator and not self.interrupted

    def success_gate(self, goal=None):
        if goal is not None:
            return super().success_gate(goal and self.contact_confirmed)
        self.controller.success_if_requested()
        if self.contact_confirmed and not self.finished:
            self.finish(success=True)
        return self.contact_confirmed

    def finish(self, *, success=False):
        self.resistance_stop.set()
        try:
            if self.resistance_worker is not None:
                self.resistance_worker.join(timeout=1.0)
                if self.resistance_worker.is_alive():
                    success = False
                    raise AutopatchError("Hunt resistance watcher did not stop")
        finally:
            super().finish(success=success)
