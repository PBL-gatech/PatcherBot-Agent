import collections
import time

import numpy as np

from ..errors import AutopatchError
from .approach_cell import ApproachCellPhase


class HuntCellPhase(ApproachCellPhase):
    phase_name = "Hunt"

    def run(self, cell=None):
        helper = self.controller.calibrated_unit.pipetteCalHelper
        with helper.brain_slice_detector():
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
        """Prepare the selected mode, near pressure, and initial position and resistance."""
        self.finished = False
        self.mode = self.controller.config.mode
        self.search_settings = None
        self.operation = None
        self.fields = ()
        self.spear_start_z = None
        self.cell_selection_locked = True
        self.started = False
        self.last_velocity = None
        self.cell_detected = False
        self.cell = cell
        self.selected_cell_anchor = None if cell is None else np.array(cell[0], dtype=float, copy=True)
        if self.selected_cell_anchor is not None:
            self.selected_cell_anchor.setflags(write=False)
        self.pipette_anchor = None
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
        self.spear_motion = None
        self.confirmed_readings = 0
        self.last_confirmation_id = None
        self.last_tip_anchor = None
        self.last_cell_frame = None
        self.stage_position = None
        self.started_at = time.monotonic()
        self.cell_confidence_threshold = (
            lambda: self.controller.calibrated_stage.cellDetectHelper.cellDetector.conf_threshold)
        self.pipette_confidence_threshold = (
            lambda: self.controller.calibrated_unit.pipetteCalHelper.pipetteDetector.adapter.threshold)
        self.failure_gate(validate_cell=True)
        self.success_gate()
        self.begin_observations()
        self.controller.info(f"Hunting for cell (mode: {self.mode})")
        self.controller.pressure.set_pressure(self.controller.config.pressure_near)
        self.controller.sleep(3)
        self.observation = self.controller.observe(
            fields=["resistance", "manipulator_position"], phase=self, raw_resistance=True)
        self.resistance = self.observation["resistance"]
        self.readings = collections.deque([self.resistance], maxlen=5)
        self.start_position = self.observation["manipulator_position"]
        self.manipulator_position = self.observation["manipulator_position"]
        self.configure_observations()
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
            if (getattr(self.controller.approach_cell_phase, "completed_successfully", False)
                    and getattr(self.controller.approach_cell_phase, "mode", None) == "Adaptive"
                    and getattr(self.controller.approach_cell_phase, "original_cell_reference", None) is self.cell[0]
                    and np.isfinite([self.controller.approach_cell_phase.cell_focus_z,
                                     self.controller.approach_cell_phase.pipette_z,
                                     self.controller.approach_cell_phase.manipulator_position[2]]).all()):
                self.cell_focus_z = self.controller.approach_cell_phase.cell_focus_z
                self.pipette_z = (self.controller.approach_cell_phase.pipette_z
                                  + self.start_position[2]
                                  - self.controller.approach_cell_phase.manipulator_position[2])
            self.operation = self.search
            self.hover_z = self.cell_focus_z
            self.cell_search_bounds = None
            self.controller.calibrated_stage.focusHelper.reset_search()

    def observe(self, **kwargs):
        """Acquire cell evidence throughout Spear; add tip and resistance in its final 5 um."""
        self.failure_gate()
        self.success_gate()
        if not self.started and self.mode != "Adaptive":
            return
        position = self.controller.calibrated_unit.position()
        self.failure_gate(position=position)
        if self.mode == "Adaptive":
            monitor = (self.spear_start_z is not None and self.spear_target - position[2] <= 5.0
                       if self.operation == self.spear
                       else abs(self.cell_focus_z - self.pipette_z) <= 5.0)
            detectors = (["deep_learning"] if self.operation == self.scan
                         or (self.operation == self.spear and monitor) else ["cell_detections"])
            if kwargs.get("evidence") is None:
                kwargs["evidence"] = self.controller.observer.infer_frame(detectors, self.frame_context,
                    pipette_prefer_anchor=getattr(self, "pipette_prefer_anchor", False),
                    pipette_radius_um=(min(self.controller.calibrated_unit.config.pipette_detection_radius_um, 7.0)
                                       if self.operation == self.spear else None))
            kwargs.setdefault("fields", ["manipulator_position", "stage_positions", "deep_learning"]
                              + (["resistance"] if monitor else []))
        else:
            self.controller.sleep(0.04)
            self.readings.append(self.resistance)
            kwargs.setdefault("fields", ["resistance", "manipulator_position"])
        kwargs["raw_resistance"] = True
        self.observation = super().observe(**kwargs)
        if "resistance" in self.observation:
            self.resistance = self.observation["resistance"]
        if self.mode == "Adaptive":
            self.failure_gate(position=self.manipulator_position)
        return self.observation

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
            and (vision.get("confidence", {}).get("pipette_detector") or 0)
            >= self.controller.calibrated_unit.pipetteCalHelper.pipetteDetector.adapter.threshold
            and 0 <= tip[0] < self.controller.calibrated_unit.camera.width
            and 0 <= tip[1] < self.controller.calibrated_unit.camera.height)
        if (observation["pipette_visible"] and sources.get("pipette_xyz") is not None
                and sources.get("stage_xy") is not None
                and np.isfinite(sources["pipette_xyz"]).all()
                and np.isfinite(sources["stage_xy"]).all()):
            self.last_tip_anchor = (np.asarray(tip)[:2].copy(),
                np.asarray(sources["pipette_xyz"]).copy(), np.asarray(sources["stage_xy"]).copy())
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
        if (self.operation == self.spear and self.cell_detection is not None
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
        """Calculate the active operation, then sustained contact and verification alignment."""
        if self.mode != "Adaptive":
            return
        self.cell_detected = False
        try:
            if self.operation == self.search:
                self.calculate_search()
                if self.search_confirmed:
                    self.cell_focus_z = self.stage_position[2]
            elif self.operation == self.scan:
                self.calculate_scan()
                if self.scan_confirmed:
                    defocus = self.vision.get("pipette_focus")
                    if (not self.vision_fresh or defocus is None or not np.isfinite(defocus)
                            or abs(defocus) > self.controller.config.scan_confirm_defocus_um):
                        self.failure_gate(message=f"Hunt Scan has stable XY but unconfirmed tip focus "
                            f"(defocus_um={defocus}); retrying Scan", recover=True)
                    else:
                        self.pipette_z = self.stage_position[2]
                        self.pipette_plane = self.pipette_z
            elif self.operation == self.spear:
                self.calculate_spear()
        except AutopatchError as error:
            if not str(error).startswith(("cell focus confirmation failed:",
                                          "pipette focus confirmation failed:")):
                raise
            if not self.failure_gate(message=str(error), recover=True):
                raise
        self.cell_detected = self.calculate_contact()

    def calculate_spear(self):
        """Use Approach's descent geometry with Hunt's descent speed."""
        if self.operation != self.spear:
            return
        super().calculate_spear()
        self.spear_speed = abs(self.controller.config.max_descent_speed)
        if self.spear_start_z is None:
            self.pipette_plane = self.pipette_z

    def calculate_contact(self):
        """Count eligible readings and confirm contact after Search and Scan."""
        self.contact_aligned = False
        cell_z = self.cell_focus_z
        tip_z = self.pipette_plane if self.operation == self.spear else self.pipette_z
        eligible = (self.spear_start_z is not None and self.spear_remaining <= 5.0
                    if self.operation == self.spear else abs(cell_z - tip_z) <= 5.0)
        if not eligible:
            self.confirmed_readings = 0
        if self.observation["observation_id"] != self.last_confirmation_id:
            self.last_confirmation_id = self.observation["observation_id"]
            if eligible and "resistance" in self.observation:
                resistance = self.observation["resistance"]
                self.confirmed_readings = (self.confirmed_readings + 1
                    if resistance is not None and np.isfinite(resistance)
                    and resistance - self.controller.first_res >= self.controller.config.cell_R_increase
                    else 0)
        if self.operation == self.spear or not self.vision_fresh or self.cell_detection is None:
            return False
        tip = self.vision.get("pipette_position") if self.observation.get("pipette_visible", False) else None
        if tip is None and self.last_tip_anchor is not None:
            sources = self.vision.get("source_positions", {})
            if sources.get("pipette_xyz") is not None and sources.get("stage_xy") is not None:
                previous_tip, previous_position, previous_stage = self.last_tip_anchor
                tip = (previous_tip
                    + self.controller.calibrated_unit.um_to_pixels_relative(
                        np.asarray(sources["pipette_xyz"]) - previous_position)[:2]
                    + self.controller.calibrated_stage.um_to_pixels_relative(
                        np.asarray(sources["stage_xy"]) - previous_stage)[:2])
        if (tip is None or not np.isfinite(tip[:2]).all()
                or not 0 <= tip[0] < self.controller.calibrated_unit.camera.width
                or not 0 <= tip[1] < self.controller.calibrated_unit.camera.height):
            return False
        separation = self.controller.calibrated_stage.pixels_to_um_relative(
            np.asarray(tip[:2]) - self.cell_detection[:2])[:2]
        self.contact_aligned = bool(
            np.linalg.norm(separation) <= self.controller.config.hunt_contact_distance
            and abs(cell_z - tip_z) <= self.controller.config.hunt_contact_plane_tolerance_um)
        return bool(self.operation == self.scan and self.scan_confirmed
                    and self.contact_aligned and self.confirmed_readings >= 5)

    def decide(self):
        """Select an action from the active operation's result."""
        self.next_action = None
        if self.controller.config.mode == "Adaptive":
            if (self.search_settings is None or "deep_learning" not in self.observation
                    or not self.vision_fresh):
                return
            if self.operation == self.search:
                if self.search_confirmed and self.cell_detection is not None:
                    self.shift_cell = True
                    self.next_action = self.shift
                else:
                    self.next_action = self.search
            elif self.operation == self.scan:
                if self.scan_confirmed:
                    if self.confirmed_readings >= 5 or (self.contact_aligned and (
                            self.confirmed_readings > 0
                            or self.cell_focus_z - self.pipette_z <= 1.0)):
                        return
                    self.shift_cell = False
                    self.next_action = self.shift
                else:
                    self.next_action = self.scan
            elif self.operation == self.spear:
                if self.spear_start_z is None:
                    self.next_action = (self.spear if self.spear_target - self.manipulator_position[2] > 1.0
                                        else self.search)
                elif self.spear_complete:
                    self.next_action = self.search
            return
        self.velocity = (0, 0, self.controller.config.max_descent_speed) if (
            self.controller.config.mode == "Classic" and not self.cell_detected) else None

    def act(self):
        """Execute the permitted action, or stop a disallowed velocity command."""
        if not self.action_gate():
            if self.last_velocity is not None:
                self.controller.calibrated_unit.stop()
                self.last_velocity = None
            return
        if self.next_action is not None:
            self.next_action()
        elif self.controller.config.mode != "Adaptive":
            if self.velocity != self.last_velocity:
                if self.velocity is None:
                    self.controller.calibrated_unit.stop()
                else:
                    self.controller.calibrated_unit.absolute_move_group_velocity(list(self.velocity))
                    self.controller.info(f"moving pipette at: {self.velocity} um/s")
                self.last_velocity = self.velocity
        self.started = True

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
            self.cell_search_bounds = (self.hover_z, self.hover_z + travel + self.controller.config.hunt_search_margin)
            self.controller.calibrated_stage.focusHelper.reset_search()
            self.search_confirmed = False
            self.scan_confirmed = False
        super().search()

    def shift(self):
        """Verify or correct cell alignment, then Scan and correct the pipette before Spear."""
        self.controller.microscope.stop()
        self.validate_pipette_anchor()
        if not self.shift_cell and (not self.scan_confirmed or self.confirmed_tip_xy is None):
            self.failure_gate(message='Pipette SHIFT requires confirmed anchored focus evidence', recover=True)
            return
        if self.shift_cell:
            target = self.bounded_cell_target(self.cell_detection[:2], self.source_stage_xy)
            reference = np.array(self.cell[0], dtype=float, copy=True)
            reference[:2] = (self.controller.calibrated_stage.M @ self.source_stage_xy
                            + self.controller.calibrated_stage.r0)[:2] - target
            reference[2] = self.cell_focus_z
            self.cell = (reference, self.cell[1], self.cell[2])
            if not self.contact_aligned and self.confirmed_readings < 5:
                self.controller.calibrated_stage.center_on_cell(
                    self.cell, detected_px=target + (self.controller.calibrated_stage.M @ (
                        np.asarray(self.controller.calibrated_stage.position())[:2] - self.source_stage_xy))[:2],
                    max_error_px=self.controller.config.approach_cell_max_shift)
                self.confirmed_readings = 0
        else:
            self.last_tip_anchor = (
                np.asarray(self.confirmed_tip_xy).copy(),
                np.asarray(self.vision["source_positions"]["pipette_xyz"]).copy(),
                np.asarray(self.source_stage_xy).copy())
            if not self.contact_aligned and self.confirmed_readings < 5:
                if self.controller.calibrated_unit.center_pipette(
                        speed=self.controller.config.approach_pipette_speed,
                        detected_px=self.projected_confirmed_tip(),
                        expected_px=self.frame_context["pipette_expected_xy"](),
                        max_deviation_um=self.controller.calibrated_unit.config.pipette_detection_radius_um) is False:
                    raise AutopatchError("Hunt pipette centering failed")
                self.start_position = self.controller.calibrated_unit.position()
                self.confirmed_readings = 0
        self.failure_gate()
        self.controller.calibrated_stage.focusHelper.reset_search()
        self.search_confirmed = False
        self.scan_confirmed = False
        self.cell_detected = False
        if self.shift_cell or self.cell_focus_z - self.pipette_z <= 1.0:
            self.operation = self.scan
            self.controller.calibrated_stage.focusHelper.relocate(self.pipette_z)
        else:
            self.operation = self.spear
            self.spear_start_z = None
            self.spear_done = None
            self.spear_motion = None
            self.hover_z = self.cell_focus_z
            self.controller.calibrated_stage.focusHelper.relocate(self.cell_focus_z)
        self.failure_gate()
        self.success_gate()

    def spear(self):
        """Focus, retain the original target reference, and start one monitored descent."""
        self.validate_pipette_anchor()
        if not self.at_hover:
            self.controller.calibrated_stage.focusHelper.relocate(self.cell_focus_z)
            return
        self.spear_cell_reference = np.array(self.cell[0], dtype=float, copy=True)
        self.confirmed_readings = 0
        super().spear()

    def action_gate(self):
        """Permit motion; wait for observations or reject invalid evidence."""
        self.failure_gate()
        self.success_gate()
        if self.controller.config.mode != "Adaptive":
            return self.velocity is None or self.controller.config.mode == "Classic"
        if "deep_learning" not in self.observation:
            return False
        if self.next_action is None:
            return True
        if (not self.vision_fresh or not np.isfinite(self.stage_position).all()
                or not np.isfinite(self.manipulator_position).all()
                or not np.isfinite(self.source_stage_xy).all()
                or not np.isfinite([self.cell_focus_z, self.pipette_z]).all()):
            raise AutopatchError("Hunt action is not permitted")
        self.failure_gate(position=self.controller.calibrated_unit.position())
        if self.next_action in (self.search, self.scan):
            return True
        if self.next_action == self.shift:
            if self.shift_cell:
                permitted = self.search_confirmed and self.cell_detection is not None
            else:
                permitted = (self.scan_confirmed and self.confirmed_tip_xy is not None
                             and np.isfinite(self.confirmed_tip_xy).all())
        elif self.next_action == self.spear:
            permitted = (self.spear_start_z is None
                         and np.isfinite([self.spear_target, self.spear_speed]).all()
                         and self.spear_speed > 0
                         and self.spear_target - self.manipulator_position[2] > 1.0
                         and abs(self.spear_target - self.start_position[2]) < self.controller.config.max_distance)
        else:
            permitted = False
        if not permitted:
            raise AutopatchError("Hunt action is not permitted")
        return True

    def failure_gate(self, *, message=None, recover=False, validate_cell=False, position=None):
        """Check operator aborts, setup, search bounds, timeout, and measured travel."""
        self.controller.abort_if_requested()
        self.validate_pipette_anchor()
        if self.controller.config.mode not in ("Training", "Classic", "Manual", "Adaptive"):
            raise AutopatchError(f"Hunt mode is not implemented: {self.controller.config.mode}")
        if self.controller.config.mode != self.mode:
            raise AutopatchError("Hunt mode cannot change during a run")
        if recover:
            if (self.controller.config.mode != "Adaptive" or self.search_settings is None
                    or self.operation not in (self.search, self.scan)):
                return False
            self.controller.warning(message)
            self.controller.calibrated_stage.focusHelper.reset_search()
            self.search_confirmed = False
            self.scan_confirmed = False
            return True
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
                raise AutopatchError("Adaptive Hunt exceeded hunt_timeout")
            if self.search_settings is not None and self.stage_position is not None:
                lower = min(self.cell_focus_z - self.controller.config.search_z_radius_um,
                            self.pipette_z - self.controller.config.scan_z_radius_um)
                upper = max(self.cell_focus_z + self.controller.config.search_z_radius_um,
                            self.pipette_z + self.controller.config.scan_z_radius_um)
                if self.cell_search_bounds is not None:
                    lower = min(lower, self.cell_search_bounds[0])
                    upper = max(upper, self.cell_search_bounds[1])
                tolerance = (self.controller.calibrated_stage.focusHelper.BOUNDARY_TOLERANCE_UM
                             if self.controller.calibrated_stage.focusHelper.active_search else 0.0)
                if not lower - tolerance <= self.stage_position[2] <= upper + tolerance:
                    raise AutopatchError("Adaptive Hunt exceeded microscope search bounds")
        if position is None or not self.started or self.controller.config.mode == "Training":
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
                self.search_settings is not None and self.operation == self.scan
                and self.cell_detected)
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
        try:
            self.stop_motions(all_devices=True)
            if self.search_settings is not None:
                self.controller.calibrated_stage.focusHelper.reset_search()
        finally:
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
