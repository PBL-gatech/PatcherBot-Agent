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
        self.classic_moving = False
        self.descent_origin = self.search_bounds = self.spear_cell_reference = None
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
        self.config.max_clearing_speed = self.config.max_descent_speed
        approach = getattr(self.controller, "approach_cell_phase", None)
        if (approach is not None and approach.completed_successfully and approach.cell is not None
                and approach.cell[0] is cell[0]):
            self.cell_observation = deepcopy(approach.cell_observation)
            self.pipette_observation = deepcopy(approach.pipette_observation)
            if self.cell_observation is not None:
                tracker = self.searcher.tracker
                tracker.set_reference_mask(self.cell_observation)
                tracker.update(self.cell_observation)
                if approach.searcher.tracker.original_area_px is not None:
                    tracker.original_area_px = approach.searcher.tracker.original_area_px
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
            self.descent_origin = self.observation["pipette"]["raw_position"][2]
            reference = self.searcher.tracker.last_cell_reference_xy
            self.spear_cell_reference = np.array(self.cell[0][:2] if reference is None else reference, copy=True)
        super().spear()
        if self.operation == self.search:
            travelled = max(0.0, self.unit.position()[2] - self.descent_origin)
            self.search_bounds = (self.cell_focus_z,
                self.cell_focus_z + travelled + self.config.hunt_search_margin)

    def search(self):
        super().search(bounds=self.search_bounds)
        if self.operation == self.shift:
            self.descent_origin = self.search_bounds = None
        self.speared = False

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
        self.readings.append(float(value))

    def resistance_rise(self):
        if len(self.readings) != 5 or np.ptp(self.readings) > (self.config.cell_R_increase
                * self.config.hunt_resistance_band_percent / 100.0):
            return None
        return float(np.mean(self.readings)) - self.baseline

    def watch_resistance(self):
        try:
            while not self.resistance_stop.wait(0.04):
                self.check_limits()
                with self.resistance_lock:
                    if self.checking_contact:
                        continue
                    self.sample_resistance()
                    rise = self.resistance_rise()
                    if (rise is not None and not self.interrupted and
                            (rise - self.last_checked_rise >= 0.100 - 1e-12
                             or self.last_checked_rise < self.config.cell_R_increase <= rise)):
                        self.last_checked_rise = max(self.last_checked_rise, rise)
                        self.interrupted = True
                    if self.interrupted:
                        self.stop_motions()
        except Exception as error:
            self.resistance_error = error
            self.interrupted = True
            self.stop_motions()

    def contact_close(self):
        self.calculate_geometry()
        tracker = self.searcher.tracker
        if not self.searcher.searching:
            tracker.update(self.observation)
        tip = self.shifter.pipette_target(self, self.observation)
        distance = tracker.distance_to_mask(tip, self.observation, self.cell[0],
            self.stage.pixels_to_um_relative, self.config.approach_cell_max_shift)
        return (distance <= self.config.hunt_contact_distance
            and abs(self.cell_focus_z - self.pipette_z) <= self.config.hunt_contact_plane_tolerance_um)

    def failure_gate(self, *, validate_cell=False):
        super().failure_gate(validate_cell=validate_cell)
        if self.resistance_error is not None:
            raise AutopatchError(f"Hunt resistance monitoring failed: {self.resistance_error}")
        if self.started_at is None or self.finished:
            return
        self.check_limits()
        # Recheck a previously rejected plateau once geometry becomes eligible.
        with self.resistance_lock:
            rise = self.resistance_rise()
            if (not self.interrupted and rise is not None and rise >= self.config.cell_R_increase
                    and self.observation is not None and self.contact_close()):
                self.interrupted = True
                self.stop_motions()

    def check_limits(self):
        if self.started_at is None or self.finished:
            return
        if self.mode == "Adaptive" and time.monotonic() - self.started_at >= self.config.hunt_timeout:
            raise AutopatchError("Adaptive Hunt exceeded hunt_timeout")
        if self.mode in ("Manual", "Classic"):
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
            self.observe()
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
            if not self.contact_confirmed:
                self.controller.warning("Contact unconfirmed; resuming Hunt")
            with self.resistance_lock:
                if rise is not None:
                    self.last_checked_rise = max(self.last_checked_rise, rise)
                self.readings.clear()
                self.interrupted = False
        finally:
            self.checking_contact = False

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
