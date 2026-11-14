import collections
import time

import numpy as np

from ..errors import AutopatchError
from .approach_cell import ApproachCellPhase


class HuntCellPhase(ApproachCellPhase):
    """Approach's four movement steps, repeated until electrical contact."""

    phase_name = "Hunt"
    RESISTANCE_RISE_MOHM = 0.100

    def prepare(self, cell=None):
        """Prepare the selected mode, near pressure, and initial position and resistance."""
        self.finished = False
        self.resistance_worker = None
        self.resistance_error = None
        self.mode = self.controller.config.mode
        self.started = self.cell_detected = self.contact_checking = False
        self.search_settings = None
        self.resistance_pending = False
        self.active_action = None
        self.started_at = time.monotonic()
        super().prepare(cell)
        self.cell_selection_locked = True
        self.last_velocity = None
        self.contact_resistances = []
        approach = getattr(self.controller, 'approach_cell_phase', None)
        if (self.mode == "Adaptive" and cell is not None and getattr(approach, 'completed_successfully', False)
                and getattr(approach, 'mode', None) == "Adaptive"
                and getattr(approach, 'original_cell_reference', None) is cell[0]
                and np.isfinite([getattr(approach, 'cell_focus_z', np.nan),
                                     getattr(approach, 'pipette_z', np.nan),
                                     getattr(approach, 'manipulator_position', [np.nan] * 3)[2]]).all()
                and getattr(approach, 'pipette_anchor', None) is not None
                and getattr(approach, 'pipette_anchor_homes', (None, None))[0] is self.controller.home_position
                and getattr(approach, 'pipette_anchor_homes', (None, None))[1] is self.controller.home_stage_position
                and getattr(approach, 'pipette_anchor_source', None) is getattr(self.controller.calibrated_unit, '_trusted_tip_reference', None)):
            self.pipette_anchor = tuple(value.copy() for value in approach.pipette_anchor)
            self.pipette_anchor_source = approach.pipette_anchor_source
            self.pipette_anchor_homes = (self.controller.home_position, self.controller.home_stage_position)
            for value in self.pipette_anchor:
                value.setflags(write=False)
        if (self.mode == 'Adaptive'
                and getattr(self.controller.calibrated_unit.pipetteCalHelper.pipetteDetector, 'model_version', None) == 'X1'):
            if self.x1_focus_mode == 'Fixed':
                threshold = self.x1_fixed_focus_threshold
                if not np.isfinite(threshold) or not 0 <= threshold <= 1:
                    raise AutopatchError('X1 fixed focus threshold must be finite and between 0 and 1')
                self.focus_threshold = float(threshold)
                self.controller.info(f'X1 Hunt Fixed focus threshold={self.focus_threshold:.3f}')
            elif self.x1_focus_mode == 'Anchor':
                reference = getattr(approach, 'focus_reference', None)
                # A completed focus reference remains usable after manual Approach termination.
                # Final Approach geometry and origin retain their stricter successful-run handoff above.
                matching_origin = (cell is not None and getattr(approach, 'mode', None) == 'Adaptive'
                        and getattr(approach, 'original_cell_reference', None) is cell[0]
                        and not getattr(approach, 'cell_reference_deleted', False)
                        and getattr(approach, 'pipette_anchor', None) is not None
                        and getattr(approach, 'pipette_anchor_homes', (None, None))[0] is self.controller.home_position
                        and getattr(approach, 'pipette_anchor_homes', (None, None))[1] is self.controller.home_stage_position
                        and getattr(approach, 'pipette_anchor_source', None) is getattr(self.controller.calibrated_unit, '_trusted_tip_reference', None))
                if (matching_origin and isinstance(reference, dict) and reference.get('cell') is cell[0]
                        and reference.get('model') == self.controller.calibrated_unit.pipetteCalHelper.pipetteDetector._active_model):
                    score, threshold = reference.get('score'), reference.get('threshold')
                    if (isinstance(score, (int, float, np.number)) and isinstance(threshold, (int, float, np.number))
                            and np.isfinite([score, threshold]).all() and 0.25 < score <= 1
                            and 0.25 < threshold <= score
                            and np.isclose(threshold, (0.25 + score) / 2)):
                        self.focus_reference = dict(reference, attempt=self.controller.observer.attempt_token)
                        self.focus_threshold = float(threshold)
                if (self.focus_reference is None and matching_origin
                        and getattr(approach, 'focus_reference_unavailable', False)
                        and getattr(approach, 'focus_reference_model', None) == self.controller.calibrated_unit.pipetteCalHelper.pipetteDetector._active_model):
                    self.focus_reference_unavailable = True
                    self.focus_reference_model = approach.focus_reference_model
                if self.focus_reference is None and not self.focus_reference_unavailable:
                    raise AutopatchError('X1 Hunt Anchor mode requires a matching valid Approach focus reference; acquire one or use Fixed mode')
            else:
                raise AutopatchError('X1 focus mode must be Anchor or Fixed')
        self.confirmed_readings = 0
        self.last_confirmation_id = None
        self.last_cell_frame = None
        from threading import Lock
        self._resistance_lock = Lock()
        self.resistance_peak = self.last_checked_rise = 0.0
        self.resistance_pending = False
        self.resume_action = self.active_action = None
        self.contact_ready = False
        self.resistance_rise = False
        self.resistance_armed = True
        self.initialized = True
        self.speared = False
        self.spear_complete = False
        self.spear_remaining = self.spear_target = self.spear_speed = None
        self.next_action = None
        self.goal_distance = 0.0
        self.reference_pending = False
        self.cell_shift_recheck = None
        self.cell_template_point = self.cell_template_evidence = None
        self.tracked_cell_reference = None
        self.stage_position = None
        self.started_at = time.monotonic()
        self.failure_gate(validate_cell=True)
        self.success_gate()
        self.controller.info(f"Hunting for cell (mode: {self.mode})")
        self.controller.pressure.set_pressure(self.controller.config.pressure_near)
        self.controller.sleep(3)
        self.observation = self.controller.observe(
             phase=self, raw_resistance=True)
        self.resistance = self.observation["resistance"]
        self.readings = collections.deque([self.resistance], maxlen=5)
        self.start_position = self.observation["manipulator_position"]
        self.manipulator_position = self.observation["manipulator_position"]
        self.controller.first_res = self.controller.resistanceRamp()
        self.controller.info(f"Initial resistance: {self.controller.first_res}")
        self.controller.info(f"{self.controller.config.mode}: starting hunt")
        if self.controller.config.mode == "Training":
            self.controller.info("Training mode: max hunt distance check disabled; waiting for resistance threshold.")
        if self.controller.config.mode == "Adaptive":
            self.stop_motions(all_devices=True)
            self.search_settings = (
                self.controller.config.hunt_limit_target_radius,
                self.controller.config.hunt_target_radius,
            )
            self.controller.config.hunt_limit_target_radius = True
            self.controller.config.hunt_target_radius = 25.0
            self.cell_focus_z = self.cell[0][2]
            self.pipette_z = (self.controller.home_stage_position[2]
                              + self.start_position[2] - self.controller.home_position[2])
            if (getattr(approach, "completed_successfully", False)
                    and getattr(self.controller.approach_cell_phase, "mode", None) == "Adaptive"
                    and getattr(self.controller.approach_cell_phase, "original_cell_reference", None) is self.cell[0]
                    and np.isfinite([self.controller.approach_cell_phase.cell_focus_z,
                                     self.controller.approach_cell_phase.pipette_z,
                                     self.controller.approach_cell_phase.manipulator_position[2]]).all()):
                self.cell_focus_z = self.controller.approach_cell_phase.cell_focus_z
                self.pipette_z = (self.controller.approach_cell_phase.pipette_z
                                  + self.start_position[2]
                                  - self.controller.approach_cell_phase.manipulator_position[2])
            # Track motor-space contact depth; only confirmed cell searches move this target.
            self.contact_depth_z = self.start_position[2] + max(0.0, self.cell_focus_z - self.pipette_z)
            self.operation = self.spear
            self.at_hover = False
            self.spear_done = None
            self.spear_cell_reference = np.array(self.cell[0], dtype=float, copy=True)
            self.hover_z = self.cell_focus_z
            self.cell_search_bounds = None
            self.controller.calibrated_stage.focusHelper.reset_search()
            self.controller.alignment_phase = self

            from threading import Event, Thread
            self.resistance_stop = Event()
            def watch_resistance():
                try:
                    while not self.resistance_stop.wait(0.04):
                        if time.monotonic() - self.started_at >= self.controller.config.hunt_timeout:
                            raise AutopatchError("Adaptive Hunt exceeded hunt_timeout")
                        self.calculate_contact(monitor=True)
                except Exception as error:
                    self.resistance_error = error
                    for device in (self.controller.calibrated_unit, self.controller.calibrated_stage,
                                   self.controller.microscope):
                        device.stop()
            self.resistance_worker = Thread(target=watch_resistance, name="HuntResistance", daemon=True)
            self.resistance_worker.start()

    def observe(self, **kwargs):
        """Observe the cell, pipette focus, and resistance throughout Hunt."""
        self.failure_gate()
        self.success_gate()
        if not self.started and self.mode != "Adaptive":
            return
        kwargs.setdefault("pipette_inference", self.mode == "Adaptive")
        kwargs.setdefault("cell_inference", self.mode == "Adaptive")
        if self.mode != "Adaptive":
            self.controller.sleep(0.04)
            self.readings.append(self.resistance)
        kwargs["raw_resistance"] = True
        try:
            observation = super().observe(**kwargs)
        except AutopatchError:
            if not self.resistance_pending:
                raise
            self.check_contact()
            observation = super().observe(**kwargs)
        self.resistance = observation["resistance"]
        self.failure_gate(position=self.manipulator_position)
        return observation

    def enrich_observation(self, observation):
        """Retain accepted detector evidence and update the cell reference during Spear."""
        self.cell_detection = None
        if self.controller.config.mode != "Adaptive" or self.search_settings is None:
            return observation
        vision = observation.get("deep_learning", {})
        sources = vision.get("source_positions", {})
        tip = vision.get("pipette_position")
        observation["pipette_visible"] = bool(
            not vision.get("stale", True) and tip is not None
            and (getattr(self, 'focus_threshold', None) is None
                 or self.controller.calibrated_stage.focusHelper.focus_accepted(vision, self.focus_threshold))
            and (vision.get("confidence", {}).get("pipette_detector") or 0)
            >= self.controller.calibrated_unit.pipetteCalHelper.pipetteDetector.adapter.threshold
            and 0 <= tip[0] < self.controller.calibrated_unit.camera.width
            and 0 <= tip[1] < self.controller.calibrated_unit.camera.height)
        stage_xy = sources.get("stage_xy")
        if vision.get("stale", True) or stage_xy is None:
            return observation
        if not np.isfinite(stage_xy).all():
            return observation
        reacquiring = self.operation == self.search and self.cell_search_bounds is not None
        radius_um = self.controller.config.hunt_target_radius
        if self.operation == self.search:
            radius_um = 7.0 if reacquiring else self.controller.config.search_xy_radius_um
        outer_radius = self.controller.config.search_xy_radius_um
        if self.controller.config.hunt_limit_target_radius:
            outer_radius = min(outer_radius, radius_um)
        self.cell_detection = self.select_cell_detection(vision.get("cell_detections"), stage_xy, outer_radius,
            local_reference=self.spear_cell_reference if reacquiring else None,
            local_radius_um=7.0 if reacquiring else None)
        frame = vision.get("source_frame")
        if (not self.contact_checking and not self._focus_recovery_requested()
                and self.operation == self.spear and self.cell_detection is not None
                and frame is not None and frame != self.last_cell_frame):
            reference = np.array(self.cell[0], dtype=float, copy=True)
            target = self.bounded_cell_target(self.cell_detection[:2], stage_xy)
            reference[:2] = (self.controller.calibrated_stage.M @ stage_xy
                            + self.controller.calibrated_stage.r0)[:2] - target
            reference[2] = self.cell_focus_z
            self.cell = (reference, self.cell[1], self.cell[2])
            self.last_cell_frame = frame
        return observation

    def calculate(self):
        """Use Approach's calculations; electrical contact is Hunt's only success."""
        if self.mode != "Adaptive":
            return
        self.cell_detected = self.calculate_contact()
        if not self.contact_checking and not self.cell_detected:
            super().calculate()
            if self.operation == self.search and self.search_confirmed:
                self.contact_depth_z += self.stage_position[2] - self.cell_focus_z
                self.cell_focus_z = self.stage_position[2]
        else:
            self.publish_alignment_snapshot()

    def calculate_spear(self):
        """Use Approach's monitored descent with Hunt's speed and a positive step."""
        if self.operation != self.spear:
            return
        super().calculate_spear()
        self.spear_speed = abs(self.controller.config.max_descent_speed)
        if self.spear_start_z is None:
            self.pipette_plane = self.pipette_z
            self.spear_target = max(self.manipulator_position[2] + 1.0, self.contact_depth_z)

    def calibrated_contact_separation(self, position):
        """Project both anchors in the same frame using the supplied motor position."""
        self.validate_pipette_anchor()
        unit, stage = self.controller.calibrated_unit, self.controller.calibrated_stage
        anchor_xy, anchor_position, anchor_stage = self.pipette_anchor
        position = np.asarray(position, dtype=float)
        tip_xy = anchor_xy + unit.um_to_pixels_relative(position - anchor_position)[:2]
        xy_um = float(np.linalg.norm(stage.pixels_to_um_relative(
            tip_xy - self.expected_cell_xy(anchor_stage))[:2]))
        tip_z = self.pipette_z
        if self.operation == self.spear and self.spear_start_z is not None:
            tip_z += position[2] - self.spear_start_z
        z_um = abs(self.cell_focus_z - tip_z)
        if not np.isfinite([xy_um, z_um]).all():
            raise AutopatchError("Invalid calibrated contact separation")
        return xy_um, z_um

    def calculate_contact(self, *, monitor=False):
        """Share one resistance window between movement monitoring and stopped contact checks."""
        if monitor and (not self.started or self.contact_checking or self.resistance_pending):
            return False
        config = self.controller.config
        with self._resistance_lock:
            observation_id = None if monitor else self.observation["observation_id"]
            if monitor or observation_id != self.last_confirmation_id:
                reading = self.controller.daq.resistance() if monitor else self.observation.get("resistance")
                if reading is None or not np.isfinite(reading):
                    raise AutopatchError("Hunt requires a finite resistance measurement")
                self.contact_resistances.append(float(reading))
                del self.contact_resistances[:-5]
                if not monitor:
                    self.last_confirmation_id = observation_id
            self.confirmed_readings = len(self.contact_resistances)
            rise = float(np.mean(self.contact_resistances)) - self.controller.first_res
            self.resistance_peak = max(self.resistance_peak, rise)
            stable = (self.confirmed_readings == 5 and np.ptp(self.contact_resistances)
                      <= config.cell_R_increase * config.hunt_resistance_band_percent / 100.0)
            further_rise = rise - self.last_checked_rise >= self.RESISTANCE_RISE_MOHM - 1e-12
            reached_goal = self.last_checked_rise < config.cell_R_increase <= rise
            self.resistance_rise = bool(stable and (further_rise or reached_goal))
            if monitor and self.resistance_rise:
                self.last_checked_rise = rise
                self.resume_action = self.active_action or self.operation
                self.resistance_pending = True
        if monitor:
            if self.resistance_pending:
                for device in (self.controller.calibrated_unit, self.controller.calibrated_stage,
                               self.controller.microscope):
                    device.stop()
            return False
        xy_um, z_um = self.calibrated_contact_separation(self.manipulator_position)
        self.contact_aligned = (xy_um <= config.hunt_contact_distance
                                and z_um <= config.hunt_contact_plane_tolerance_um)
        if self.manipulator_position[2] < self.contact_depth_z - 2.0:
            self.contact_ready = False
            return False
        self.contact_ready = bool(stable and self.contact_aligned and rise >= config.cell_R_increase - 1e-12)
        return self.contact_checking and self.contact_ready

    def decide(self):
        """Spear -> Search -> cell Shift -> Scan -> pipette Shift -> Spear."""
        self.next_action = None
        if self.mode != "Adaptive":
            self.velocity = ((0, 0, self.controller.config.max_descent_speed)
                             if self.mode == "Classic" and not self.cell_detected else None)
        elif self.cell_detected:
            return
        elif self.contact_checking:
            if self.confirmed_readings == 5:
                self.contact_checking = False
                self.contact_resistances.clear()
                self.confirmed_readings = 0
                self.next_action = self.resume_action or self.operation
                self.resume_action = None
        elif self.contact_ready or self.resistance_pending or (self.resistance_rise and self.resistance_armed):
            self.next_action = self.check_contact
        elif self._focus_recovery_requested():
            self.next_action = self.scan
        elif self.operation == self.spear:
            if not self.vision_fresh or self.cell_detection is None or self.spear_complete:
                self.next_action = self.search
            elif self.spear_start_z is None:
                self.next_action = self.spear
        else:
            super().decide()

    def act(self):
        """Execute the permitted action, or stop a disallowed velocity command."""
        if not self.action_gate():
            if self.last_velocity is not None:
                self.controller.calibrated_unit.stop()
                self.last_velocity = None
            return
        if self.contact_checking:
            return
        if self.next_action is not None:
            self.started = True
            self.active_action = self.next_action
            try:
                self.next_action()
            except AutopatchError:
                if not self.resistance_pending:
                    raise
                self.check_contact()
            finally:
                self.active_action = None
        elif self.controller.config.mode != "Adaptive":
            if self.velocity != self.last_velocity:
                if self.velocity is None:
                    self.controller.calibrated_unit.stop()
                else:
                    self.controller.calibrated_unit.absolute_move_group_velocity(list(self.velocity))
                    self.controller.info(f"moving pipette at: {self.velocity} um/s")
                self.last_velocity = self.velocity
        self.started = True

    def check_contact(self):
        """Stop every device before collecting five fresh alignment/contact readings."""
        self.resume_action = self.resume_action or self.operation
        self.last_checked_rise = max(self.last_checked_rise,
            float(np.mean(self.contact_resistances)) - self.controller.first_res
            if self.contact_resistances else 0.0)
        self.contact_checking = True
        self.next_action = None
        self.resistance_pending = False
        self.stop_motions(all_devices=True)
        mini = getattr(self, 'mini_scan', None)
        if mini:
            self.operation = self.resume_action = mini['operation']
        self.mini_scan = self.focus_recovery_pending = None
        self.focus_peak_state = None
        if self.spear_start_z is not None:
            self.pipette_z += self.controller.calibrated_unit.position()[2] - self.spear_start_z
            self.spear_start_z = self.spear_motion = self.spear_done = None
        self.controller.calibrated_stage.focusHelper.reset_search()
        self.resistance_armed = False
        self.contact_resistances.clear()
        self.confirmed_readings = 0
        self.controller.info(f"Hunt resistance rise: stopped; increase={self.last_checked_rise:.3f} MOhm; "
                             f"peak={self.resistance_peak:.3f} MOhm; checking calibrated alignment and contact")

    def search(self):
        """Search the cell; after Spear, use actual travel plus the configured margin."""
        if self.operation == self.spear:
            self.cell_detected = False
            self.stop_motions(all_devices=True)
            stopped_z = self.controller.calibrated_unit.position()[2]
            travel = 0.0
            if self.spear_start_z is None:
                self.spear_cell_reference = np.array(self.cell[0], dtype=float, copy=True)
            else:
                travel = max(0.0, stopped_z - self.spear_start_z)
                self.pipette_z += stopped_z - self.spear_start_z
            self.pipette_plane = self.pipette_z
            self.spear_start_z = None
            self.spear_motion = None
            self.spear_done = None
            self.operation = self.search
            self._clear_position_cycle('search')
            self.cell_search_bounds = (self.hover_z, self.hover_z + travel + self.controller.config.hunt_search_margin)
            self.controller.calibrated_stage.focusHelper.reset_search()
            self.search_confirmed = False
            self.scan_confirmed = False
        super().search()
        self.speared = False

    def shift(self):
        """Reuse Approach's two shifts, then return to Spear instead of completing."""
        self.speared = False
        super().shift()

    def spear(self):
        """Focus, retain the original target reference, and start one monitored descent."""
        self.validate_pipette_anchor()
        if not self.at_hover:
            self.controller.calibrated_stage.focusHelper.relocate(self.cell_focus_z)
            return
        self.spear_cell_reference = np.array(self.cell[0], dtype=float, copy=True)
        super().spear()

    def action_gate(self):
        """Keep Approach's movement operations available during contact recovery."""
        self.failure_gate()
        self.success_gate()
        if self.mode != "Adaptive":
            return self.velocity is None or self.mode == "Classic"
        return self.next_action is None or self.next_action in (
            self.search, self.scan, self.shift, self.spear, self.check_contact)

    def failure_gate(self, *, message=None, recover=False, validate_cell=False, position=None):
        """Check operator aborts, setup, search bounds, timeout, and measured travel."""
        self.controller.abort_if_requested()
        if self.resistance_error is not None:
            raise AutopatchError(f"Hunt resistance monitoring failed: {self.resistance_error}")
        if self.resistance_pending:
            if self.active_action is not None:
                raise AutopatchError("Hunt contact calculation requested")
            self.check_contact()
        self.validate_pipette_anchor()
        if self.controller.config.mode not in ("Training", "Classic", "Manual", "Adaptive"):
            raise AutopatchError(f"Hunt mode is not implemented: {self.controller.config.mode}")
        if self.controller.config.mode != self.mode:
            raise AutopatchError("Hunt mode cannot change during a run")
        if recover:
            if (self.controller.config.mode != "Adaptive" or self.search_settings is None
                    or self.operation not in (self.search, self.scan)):
                return False
            return super().failure_gate(message=message, recover=True)
        if message is not None:
            raise AutopatchError(message)
        if validate_cell:
            self.controller.isrigready()
            if self.controller.rig_ready is False:
                raise AutopatchError("Rig not ready for cell hunting")
            if self.cell is None:
                raise AutopatchError("No cell given to patch!")
        if self.controller.config.mode == "Adaptive":
            if time.monotonic() - self.started_at >= self.controller.config.hunt_timeout:
                raise AutopatchError("Adaptive Hunt contact not confirmed before hunt_timeout"
                    if self.contact_checking else "Adaptive Hunt exceeded hunt_timeout")
        # The fixed distance limit from Hunt's starting position excludes Adaptive mode.
        if position is None or not self.started or self.controller.config.mode in ("Training", "Adaptive"):
            return
        travel = abs(position[2] - self.start_position[2])
        if travel >= int(self.controller.config.max_distance):
            raise AutopatchError("Cell not detected before reaching max hunt distance "
                f"({travel:.1f} um >= {float(self.controller.config.max_distance):.1f} um).")

    def success_gate(self, goal=None):
        """Handle operator success and finish when the calculated contact goal is met."""
        self.controller.success_if_requested()
        if self.controller.config.mode == "Adaptive":
            self.cell_detected = bool(
                self.search_settings is not None and self.cell_detected)
        elif self.search_settings is not None:
            return False
        elif goal is None:
            if not self.started:
                return False
            self.cell_detected = self._isCellDetected(self.readings, self.controller.config.cell_R_increase)
        else:
            self.cell_detected = bool(goal)
        if self.cell_detected:
            self.finish(success=True)
        return self.cell_detected

    def finish(self, *, success=False):
        """Cancel owned motions, restore settings, and preserve Training completion."""
        if self.finished:
            return
        self.contact_checking = True
        if self.resistance_worker is not None:
            self.resistance_stop.set()
            self.resistance_worker.join(timeout=1.0)
        try:
            self.stop_motions(all_devices=True)
            if self.search_settings is not None:
                self.controller.calibrated_stage.focusHelper.reset_search()
        finally:
            if getattr(self.controller, 'alignment_phase', None) is self:
                self.controller.alignment_phase = None
            if self.search_settings is not None:
                self.controller.config.hunt_limit_target_radius = self.search_settings[0]
                self.controller.config.hunt_target_radius = self.search_settings[1]
                self.search_settings = None
        self.finished = True
        if success:
            self.controller.info("Cell Detected")
            if self.controller.config.mode == "Training":
                self.finished = False
                self.controller.info("Training mode: waiting for manual Success or Abort.")
                while True:
                    self.failure_gate()
                    self.controller.sleep(0.1)
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
