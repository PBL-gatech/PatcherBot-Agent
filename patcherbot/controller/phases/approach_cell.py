"""Cell-type-dependent approach from the common localization distance."""

import sys
import time
import numpy as np
from ..errors import AutopatchError
from ..PhaseController import PhaseController


class ApproachCellPhase(PhaseController):
    phase_name = "Approach"

    def run(self, cell=None):
        """Run one approach attempt; always release its owned motions."""
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
        """Establish safe defaults, recording, and ownership for this attempt."""
        self.cell = cell
        self.original_cell_reference = None
        self.fields = ()
        self.operation = self._start
        self.next_action = None
        self.finished = False
        self.completed_successfully = False
        self.mode = self.controller.config.mode
        self.initialized = False
        self.speared = False
        self.cell_search_radius_um = 25.0
        self.cell_search_bounds = None
        self.cell_focus_z = None
        self.cell_focus_error = None
        self.pipette_z = None
        self.hover_z = None
        self.goal_distance = None
        self.cell_detection = None
        self.cell_template_point = None
        self.cell_template_evidence = None
        self.cell_shift_recheck = None
        self.tracked_cell_reference = None
        self.confirmed_tip_xy = None
        self.detected_tip_xy = None
        self.calibrated_tip_xy = None
        self.tracked_tip_xy = None
        self.saved_tip_evidence = None
        self.spear_motion = None
        self.spear_done = None
        self.spear_target = None
        self.spear_start_z = None
        self.spear_remaining = None
        self.spear_speed = None
        self.pipette_plane = None
        self.at_hover = False
        self.cell_selection_locked = False
        self.search_confirmed = False
        self.scan_confirmed = False
        self.shift_cell = False
        self.shift_ready = False
        self.spear_complete = False
        self.decision_cell_type = None
        self.decision_cell_type_toggle = None
        self.observation = None
        self.vision = {}
        self.stage_position = None
        self.manipulator_position = None
        self.source_stage_xy = None
        self.cell_detections = ()
        self.vision_fresh = False
        self.alignment_snapshot = None
        self.cell_confidence_threshold = lambda: self.controller.calibrated_stage.cellDetectHelper.cellDetector.conf_threshold
        self.pipette_confidence_threshold = lambda: self.controller.calibrated_unit.pipetteCalHelper.pipetteDetector.adapter.threshold
        self.original_cell_reference = None if cell is None else cell[0]
        self.failure_gate()
        self.success_gate()
        self.begin_observations()
        self.configure_observations()
        self.controller.alignment_phase = self
        self.controller.calibrated_stage.focusHelper.reset_search()
        self.controller.info(f"Approach START: mode={self.mode}")

    def observe(self, **kwargs):
        """Acquire evidence once, retaining frame anchors and settled motor Z.

        The raw record remains available for recording and tracking helpers;
        phase decisions use the named attributes unpacked here.
        """
        kwargs.setdefault("fields", self.fields)
        observation = super().observe(**kwargs)
        self.observation = observation
        self.vision = observation.get("deep_learning", {})
        self.stage_position = observation.get("stage_positions", self.stage_position)
        self.manipulator_position = observation.get("manipulator_position", self.manipulator_position)
        sources = self.vision.get("source_positions", {})
        self.source_stage_xy = sources.get("stage_xy")
        if self.source_stage_xy is None and self.stage_position is not None:
            self.source_stage_xy = self.stage_position[:2]
        detections = self.vision.get("cell_detections")
        self.cell_detections = () if detections is None else detections
        self.vision_fresh = not self.vision.get("stale", True)
        self.detected_tip_xy = self.vision.get("pipette_position")
        self.calibrated_tip_xy = sources.get("pipette_expected_xy")
        self.tracked_tip_xy = self.vision.get("pipette_tracking_position", sources.get("pipette_tracking_xy"))
        if (self.operation == self.spear and self.spear_start_z is not None
                and "stage_positions" in observation and "manipulator_position" in observation):
            try:
                self.spear_done = (self.controller.calibrated_unit.actionHelper.is_complete(self.spear_motion)
                                   if self.spear_motion is not None else None)
            except RuntimeError as error:
                raise AutopatchError(str(error)) from error
            # The monitor can finish after the frame was acquired: reread settled Z.
            if self.spear_done:
                self.manipulator_position = self.controller.calibrated_unit.position()
        return observation

    def configure_observations(self, *, tracking=False):
        """Configure acquisition-time anchors in microscope/manipulator um."""
        controller = self.controller
        unit, stage = controller.calibrated_unit, controller.calibrated_stage
        self.frame_context = dict(stage_xy=stage.position, pipette_xyz=unit.position,
            focus_z=lambda: controller.microscope.position() / unit.config.microscope_units_per_um,
            pipette_expected_xy=lambda: unit.reference_position(
                anchor=getattr(unit, '_trusted_tip_reference', None) or
                (np.array([unit.camera.width / 2., unit.camera.height / 2.]),
                 controller.home_position, np.asarray(controller.home_stage_position)[:2]))[:2])
        if tracking:
            self.frame_context.update(pipette_tracking_xy=lambda: unit.reference_position(anchor=unit._tip_reference)[:2]
                if getattr(unit, '_tip_reference', None) is not None else None)

    def _shift_cell_from_evidence(self):
        """Apply existing cell consensus and centering; False requests a recheck."""
        controller, stage = self.controller, self.controller.calibrated_stage
        options = {}
        fusion = getattr(controller.config, 'cell_shift_fusion', True)
        if fusion:
            options = stage.cellTrackHelper.cell_shift_options(stage, self.observation, self,
                use_centroid=controller.config.use_centroid, info=controller.info, warning=controller.warning)
            if options is False:
                return False
        else:
            controller.info('Cell SHIFT consensus disabled; using existing template centering')
        if 'detected_px' in options:
            # Retain the accepted centering evidence with its own stage anchor.
            self.alignment_snapshot = dict(self.alignment_snapshot or {},
                cell_detection=None,
                accepted_cell=dict(xy=np.asarray(options['detected_px']).copy(),
                                   stage_xy=np.asarray(stage.position())[:2].copy()))
        result = stage.center_on_cell(self.cell, use_centroid=controller.config.use_centroid,
            max_error_px=controller.config.approach_cell_max_shift, **options)
        return None if fusion else result

    def calculate(self):
        """Calculate the four S-method facts and publish copied evidence."""
        if self.mode == "Adaptive" and self.initialized:
            # Search and Scan confirm current evidence before Shift can be chosen.
            self.calculate_search()
            self.calculate_scan()
            # Shift selects its target; Spear measures hover and descent progress.
            self.calculate_shift()
            self.calculate_spear()
            self.controller.observer.annotate(self, self.observation,
                fields={"cell_detection": self.cell_detection},
                calculations={"cell_focus_error": self.cell_focus_error,
                              "spear_target": self.spear_target, "spear_done": self.spear_done,
                              "spear_remaining": self.spear_remaining, "pipette_plane": self.pipette_plane})
        # Publish one copied evidence record at the completed calculation boundary.
        if self.observation is not None:
            from copy import deepcopy
            vision = self.observation.get("deep_learning", {})
            snapshot = deepcopy(dict(
                observation_id=self.observation.get("observation_id"),
                source_frame=vision.get("source_frame"),
                stale=vision.get("stale", True),
                source_positions=vision.get("source_positions", {}),
                stage_position=self.observation.get("stage_positions"),
                manipulator_position=self.observation.get("manipulator_position"),
                cell_detection=self.cell_detection,
                accepted_cell=(self.alignment_snapshot or {}).get("accepted_cell"),
                detected_tip_xy=self.detected_tip_xy,
                calibrated_tip_xy=self.calibrated_tip_xy,
                tracked_tip_xy=self.tracked_tip_xy,
                confirmed_tip_xy=self.confirmed_tip_xy,
                saved_tip_evidence=self.saved_tip_evidence,
                spear_start_z=self.spear_start_z,
            ))
            # Preserve the selected reference and bound operation identities.
            snapshot["cell_reference"] = self.original_cell_reference
            snapshot["operation"] = self.operation
            self.alignment_snapshot = snapshot

    def calculate_search(self):
        """SEARCH: associate the selected cell and confirm settled focus evidence.

        Identity uses confidence and calibrated distance in um. A locked Hunt
        selection is retained; confirmation consumes this observation only.
        """
        self.cell_focus_error = self.stage_position[2] - self.cell_focus_z
        if not self.cell_selection_locked:
            self.cell_detection = None
        if self.operation != self.search:
            return
        stage = self.controller.calibrated_stage
        if not self.cell_selection_locked:
            expected_xy = (stage.M @ self.source_stage_xy + stage.r0)[:2] - self.cell[0][:2]
            threshold = stage.cellDetectHelper.cellDetector.conf_threshold
            candidates = []
            for detection in self.cell_detections:
                if not np.isfinite(detection[:3]).all() or detection[2] < threshold:
                    continue
                offset_px = np.asarray(detection[:2]) - expected_xy
                if np.linalg.norm(stage.pixels_to_um_relative(offset_px)[:2]) <= self.cell_search_radius_um:
                    candidates.append(detection)
            self.cell_detection = min(candidates,
                key=lambda detection: np.linalg.norm(np.asarray(detection[:2]) - expected_xy), default=None)
        try:
            self.search_confirmed = stage.focusHelper.confirm_search(self.vision, self.cell_detection)
        except RuntimeError as error:
            raise AutopatchError(str(error)) from error

    def calculate_scan(self):
        """SCAN: confirm detector XY and retain separately anchored provisional evidence."""
        self.saved_tip_evidence = None
        if self.operation != self.scan:
            return
        focus = self.controller.calibrated_stage.focusHelper
        try:
            self.scan_confirmed = focus.confirm_search(self.vision)
        except RuntimeError as error:
            raise AutopatchError(str(error)) from error
        self.confirmed_tip_xy = focus.confirmed_tip_xy
        self.saved_tip_evidence = focus.saved_tip_evidence

    def calculate_shift(self):
        """SHIFT: select cell after Search or pipette after Scan and its readiness."""
        self.shift_ready = False
        if self.operation == self.search:
            self.shift_cell = True
            self.shift_ready = self.search_confirmed
        elif self.operation == self.scan:
            self.shift_cell = False
            self.shift_ready = self.scan_confirmed

    def calculate_spear(self):
        """SPEAR: measure hover, target, and progress in calibrated um.

        Observation supplies monitor completion and settled motor position.
        Hunt keeps its own descent target, speed, and contact goal.
        """
        if self.operation != self.spear:
            return
        self.spear_speed = abs(self.controller.config.max_clearing_speed)
        self.at_hover = (self.cell_focus_error >= 0 if self.hover_z is None
                         else abs(self.stage_position[2] - self.hover_z) <= 0.5)
        if self.spear_start_z is None:
            distance = (self.controller.config.slice_start_distance - self.goal_distance
                        if self.hover_z is None else self.hover_z - self.pipette_z)
            self.spear_target = self.manipulator_position[2] + distance
            self.spear_complete = False
            return
        pipette_z = self.manipulator_position[2]
        self.spear_remaining = self.spear_target - pipette_z
        self.pipette_plane = self.pipette_z + pipette_z - self.spear_start_z
        self.spear_complete = self.spear_remaining <= 0.5 if self.spear_done is None else self.spear_done

    def decide(self):
        """Choose a named operation using calculated readiness and the live mode."""
        self.next_action = None
        if self.awaiting_operator or self.finished:
            return
        self.decision_cell_type = self.controller.config.cell_type
        self.decision_cell_type_toggle = self.controller.config.cell_type_toggle
        # Classic, Manual, Agent, and Training share the existing cell-type route.
        if self.controller.config.mode != "Adaptive" or not self.initialized:
            self.next_action = self._start
        elif self.operation == self.search:
            self.next_action = self.shift if self.shift_ready else self.search
        elif self.operation == self.scan:
            self.next_action = self.shift if self.shift_ready else self.scan
        elif self.operation == self.spear:
            if self.spear_start_z is None:
                if self.at_hover:
                    self.next_action = self.spear
            elif self.spear_complete:
                self.next_action = self.search

    def act(self):
        """Check permission, then visibly execute the selected S method."""
        if self.next_action is not None or (self.mode == "Adaptive" and self.initialized):
            if not self.action_gate():
                raise AutopatchError("Approach action is not permitted")
        if self.next_action is None:
            self.controller.sleep(0.1)
        elif self.next_action == self._start:
            self._start()
        elif self.next_action == self.search:
            self.search()
        elif self.next_action == self.shift:
            self.shift()
        elif self.next_action == self.scan:
            self.scan()
        elif self.next_action == self.spear:
            self.spear()

    def search(self):
        """Move the microscope to find the selected cell's focal plane.

        After Spear, first stop the descent and return to the saved cell plane.
        Continue the focus search until the cell is ready for centering by Shift."""
        controller, helper = self.controller, self.controller.calibrated_stage.focusHelper
        if self.operation == self.spear:
            controller.microscope.stop()
            # Cancel the monitor before another operation owns the devices.
            if self.spear_motion is not None:
                controller.calibrated_unit.actionHelper.cancel(self.spear_motion)
                self.spear_motion = None
            controller.calibrated_unit.stop()
            self.operation = self.search
            self.speared = True
            self.pipette_z = self.pipette_plane
            self.spear_start_z = None
            helper.relocate(self.cell_focus_z)
            self.failure_gate()
            self.success_gate()
            helper.reset_search()
            self.search_confirmed = False
            return
        try:
            helper.advance_search(actual_z=self.stage_position[2], vision=self.vision,
                cell=self.cell_detection, center_z=self.cell_focus_z,
                speed=controller.config.search_speed, motion=controller.calibrated_stage.actionHelper,
                bounds=self.cell_search_bounds, pipette=False,
                pixels_to_um_relative=controller.calibrated_stage.pixels_to_um_relative,
                confidence_threshold=self.cell_confidence_threshold,
                abort_check=controller.abort_if_requested, info=controller.info, warning=controller.warning)
        except (RuntimeError, ValueError) as error:
            raise AutopatchError(str(error)) from error
        if helper.empty:
            reason = "SEARCH focus sweep found no valid samples"
            if not self.failure_gate(message=reason, recover=True):
                raise AutopatchError(reason)

    def scan(self):
        """Move the microscope to find the pipette's focal plane.

        The confirmed tip position is then used by Shift to center the pipette
        in the camera image."""
        controller = self.controller
        stage, unit = controller.calibrated_stage, controller.calibrated_unit
        try:
            stage.focusHelper.advance_search(actual_z=self.stage_position[2], vision=self.vision,
                cell=self.cell_detection, center_z=self.pipette_z, speed=controller.config.search_speed,
                motion=stage.actionHelper, bounds=None, pipette=True,
                pixels_to_um_relative=stage.pixels_to_um_relative,
                confidence_threshold=self.pipette_confidence_threshold,
                abort_check=controller.abort_if_requested, info=controller.info, warning=controller.warning)
        except (RuntimeError, ValueError) as error:
            raise AutopatchError(str(error)) from error

    def shift(self):
        """Center the cell or pipette in the camera image.

        After Search, move the stage to center the cell, then continue to Scan.
        After Scan, move the manipulator to center the pipette, then start Spear
        or finish if the alignment after descent is complete."""
        controller = self.controller
        controller.microscope.stop()
        if self.shift_cell:
            self.cell_focus_z = self.stage_position[2]
            centered = self._shift_cell_from_evidence()
            if centered is False:
                return
        else:
            self.pipette_z = self.stage_position[2]
            options = {}
            if self.confirmed_tip_xy is not None:
                options = dict(detected_px=self.confirmed_tip_xy,
                    expected_px=self.frame_context["pipette_expected_xy"](),
                    max_deviation_um=controller.calibrated_unit.config.pipette_detection_radius_um)
                self.confirmed_tip_xy = None
            centered = controller.calibrated_unit.center_pipette(
                speed=controller.config.approach_pipette_speed, **options)
        if not self.action_gate():
            raise AutopatchError("Mode changed during approach SHIFT")
        if not self.shift_cell and centered is False:
            reason = getattr(controller.calibrated_unit, "last_pipette_move_failure", None)
            if not isinstance(reason, str) or not reason:
                reason = "pipette centering did not complete"
            next_action = "final alignment incomplete" if self.speared else "SPEAR cancelled"
            raise AutopatchError(f"{self.phase_name} pipette SHIFT failed: {reason}; {next_action}")
        controller.calibrated_stage.focusHelper.reset_search()
        self.search_confirmed = False
        self.scan_confirmed = False
        if self.shift_cell:
            self.operation = self.scan
            next_z = self.pipette_z
        elif self.speared:
            return self.finish(success=True)
        else:
            self.operation = self.spear
            self.spear_start_z = None
            next_z = self.hover_z = self.cell_focus_z - self.goal_distance
        controller.calibrated_stage.focusHelper.relocate(next_z)
        self.failure_gate()
        self.success_gate()

    def spear(self):
        """Start monitored pipette descent to the calculated target depth.

        Return while the pipette moves so the phase can continue observing.
        Later observations check the motion's progress and completion."""
        controller = self.controller
        controller.microscope.stop()
        self.operation = self.spear
        self.spear_start_z = self.manipulator_position[2]
        unit, actions = controller.calibrated_unit, controller.calibrated_unit.actionHelper
        try:
            target, speed = float(self.spear_target), float(self.spear_speed)
            if not np.isfinite([target, speed]).all() or speed <= 0:
                raise ValueError('Spear requires a finite target and positive finite speed')
            controller.abort_if_requested()
            start = float(unit.position()[2])
            if not np.isfinite(start):
                raise RuntimeError('Spear has no valid pipette Z position')
            if target < start - actions.MOTION_ENDPOINT_TOLERANCE_UM:
                raise ValueError('Spear target is behind the current pipette position')
            deadline = time.monotonic() + max(10.0, max(0.0, target - start) / speed * 20.0)
            unit.stop()
            unit.info(f'SPEAR motion: start_z={start}; target_z={target}; '
                      f'velocity_command={-speed}; endpoint monitor active')
            self.spear_motion = actions.start(unit, target=target, deadline=deadline,
                read_position=lambda: float(unit.position()[2]),
                move_velocity=lambda direction: unit.absolute_move_group_velocity([0.0, 0.0, -direction * speed]),
                abort_check=controller.abort_if_requested, label='Spear motion', info=unit.info,
                direction=1.0, reversal_origin=start,
                reversal_tolerance=actions.MOTION_ENDPOINT_TOLERANCE_UM)
        except (RuntimeError, ValueError) as error:
            raise AutopatchError(str(error)) from error

    def _start(self):
        """Initialize Adaptive inputs or execute the existing cell-type route."""
        controller, config = self.controller, self.controller.config
        controller.calibrated_stage.set_max_speed(config.max_locate_speed)
        controller.calibrated_unit.set_max_speed(config.max_locate_speed)
        self.mode = config.mode
        adaptive = self.mode == "Adaptive"
        plate = not adaptive and config.cell_type == "Plate"
        slice_route = not adaptive and config.cell_type == "Slice" and config.cell_type_toggle
        if plate:
            controller.move_group_down(config.slice_start_distance - config.cell_distance)
            controller.sleep(0.1)
            controller.fine_calibrate_pipette()
        if not self.action_gate():
            raise AutopatchError("Cell type changed during approach")
        controller.amplifier.start_patch()
        if adaptive:
            controller.info("Approach WARMUP: 3 inference samples before motion")
            for _ in range(3):
                self.observe(fields=["deep_learning"])
                self.failure_gate()
                self.success_gate()
            if config.mode != self.mode or not self.action_gate():
                raise AutopatchError("Mode changed during approach warmup")
            self.cell_focus_z = self.cell[0][2]
            self.goal_distance = (config.cell_distance if config.cell_type == "Plate" or
                (config.cell_type == "Slice" and config.cell_type_toggle) else config.slice_start_distance)
            self.fields = ["stage_positions", "deep_learning", "manipulator_position", "camera_image"]
            self.pipette_z = controller.microscope.position() / controller.calibrated_unit.config.microscope_units_per_um
            self.operation = self.search
            self.initialized = True
            controller.calibrated_stage.focusHelper.relocate(self.cell_focus_z)
            self.failure_gate()
            self.success_gate()
            return
        self.align(self.cell, config.cell_distance if plate else config.slice_start_distance, config.use_centroid)
        if slice_route:
            if not self.action_gate():
                raise AutopatchError("Cell type changed during approach")
            self.clear_to_cell(self.cell)
            if not self.action_gate():
                raise AutopatchError("Cell type changed during approach")
            self.align(self.cell, config.cell_distance, config.use_centroid)
        self.fields = ["manipulator_position", "resistance"]
        self.initialized = True
        self.finish(success=True)

    def action_gate(self):
        """Validate live selection and Adaptive evidence without losing ownership."""
        self.failure_gate()
        self.success_gate()
        config = self.controller.config
        adaptive = ((self.mode == "Adaptive" and self.initialized) or config.mode == "Adaptive")
        if adaptive:
            if config.mode != "Adaptive":
                return False
            if not self.initialized:
                return True
            low, high = sorted([self.cell_focus_z, self.pipette_z])
            focus = self.controller.calibrated_stage.focusHelper
            tolerance = focus.BOUNDARY_TOLERANCE_UM if focus.active_search else 0.0
            if self.next_action == self.spear:
                if not (self.spear_speed > 0 and np.isfinite(self.spear_target)
                        and self.spear_target >= self.manipulator_position[2]):
                    return False
            return (self.vision_fresh and low - focus.RADIUS_UM - tolerance <= self.stage_position[2]
                    <= high + focus.RADIUS_UM + tolerance)
        return (self.decision_cell_type == config.cell_type and
                (config.cell_type == "Plate" or bool(self.decision_cell_type_toggle) == bool(config.cell_type_toggle)))

    def failure_gate(self, *, message=None, recover=False, validate_cell=False):
        """Check aborts; optionally validate the rig/cell before slice clearing."""
        if recover:
            return False
        self.controller.abort_if_requested()
        if validate_cell:
            self.controller.isrigready()
            if self.controller.rig_ready is False:
                raise AutopatchError("Rig not ready for clearing to cell")
            if self.cell is None:
                raise AutopatchError("No cell given to patch!")

    def success_gate(self, goal=None):
        """Preserve shared operator and Training completion handling."""
        return super().success_gate(None if goal is None else goal is True)

    def stop_motions(self, *, all_devices=False):
        """Cancel monitors and attempt every stop, retaining the original error."""
        primary_error = sys.exc_info()[1]
        stop_error = None
        stops = ([self.controller.calibrated_stage.stop, self.controller.calibrated_unit.stop,
                  self.controller.microscope.stop] if all_devices else [self.controller.calibrated_unit.stop])
        focus = self.controller.calibrated_stage.focusHelper
        if focus.active_search:
            if all_devices:
                stops.insert(0, focus.cancel_search)
            else:
                stops.append(focus.cancel_search)
        if self.spear_motion is not None:
            stops.insert(0, lambda: self.controller.calibrated_unit.actionHelper.cancel(self.spear_motion))
        if not all_devices and self.mode == "Adaptive":
            stops.extend([self.controller.calibrated_stage.stop, self.controller.microscope.stop])
        for stop in stops:
            try:
                stop()
            except BaseException as error:
                if primary_error is not None:
                    if hasattr(primary_error, "add_note"):
                        primary_error.add_note(f"Approach stop also failed: {error}")
                elif stop_error is None:
                    stop_error = error
        if stop_error is not None:
            raise stop_error
        return primary_error is None

    def finish(self, *, success=False):
        """Release owned motions/activity; publish successful Adaptive handoff."""
        try:
            if self.finished:
                return
            self.stop_motions()
            self.finished = True
            if success:
                self.controller.calibrated_stage.set_max_speed(10000)
                self.controller.calibrated_unit.set_max_speed(100000)
                self.controller.info("Approached Cell")
                self.completed_successfully = True
                self.success_gate(True)
                self.goal_event = False
                if not self.awaiting_operator:
                    self.complete_success()
        finally:
            if self.controller.alignment_phase is self:
                self.controller.alignment_phase = None

    def align(self, cell, cell_distance, use_centroid):
        """Align the pipette to the cell using microscope imaging."""
        controller, config = self.controller, self.controller.config
        focus = controller.calibrated_stage.focusHelper
        controller.microscope.move_to_floor()
        focus.relocate(cell[0][2])
        if config.cell_type_toggle:
            controller.calibrated_stage.center_on_cell(
                cell, use_centroid=use_centroid,
                max_error_px=config.approach_cell_max_shift)
            focus.relocate(cell[0][2] - cell_distance)
            controller.calibrated_unit.center_pipette(speed=config.approach_pipette_speed)
            focus.relocate(cell[0][2])

    def clear_to_cell(self, cell):
        """Run existing Slice clearing, stopping descent even when a gate fails."""
        motion_started = False
        try:
            self.failure_gate(validate_cell=True)
            self.success_gate()
            controller = self.controller
            config = controller.config
            controller.info("Clearing to cell")
            if not (config.cell_type_toggle and config.cell_type == "Slice"):
                return
            cell_hover_pos = config.cell_distance - config.slice_start_distance
            controller.microscope.relative_move(-config.cell_distance)
            start_pos = controller.calibrated_unit.position()
            motion_started = True
            controller.calibrated_unit.absolute_move_group_velocity([0, 0, config.max_clearing_speed])
            while start_pos[2] - controller.calibrated_unit.position()[2] > cell_hover_pos:
                self.failure_gate()
                self.success_gate()
                controller.sleep(0.1)
        finally:
            if motion_started:
                self.stop_motions()
