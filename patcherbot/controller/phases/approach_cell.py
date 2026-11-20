"""Cell approach using observation-driven movement helpers."""
from copy import deepcopy
from math import isfinite
from types import SimpleNamespace
import sys
import time

import numpy as np

from ..PhaseController import PhaseController
from ..errors import AutopatchError
from ...devices.manipulator.helpers.ActionHelper import ActionHelper
from ...devices.manipulator.helpers.Tracker import Tracker
from ...devices.manipulator.helpers.Searcher import Searcher
from ...devices.manipulator.helpers.Scanner import Scanner
from ...devices.manipulator.helpers.Shifter import Shifter
from ...devices.manipulator.helpers.Spearer import Spearer



class ApproachCellPhase(PhaseController):
    phase_name = "Approach"

    def __init__(self, controller, *, select_cell=None, rank_cell=None, min_area_px=None,
                 half_life_s=None):
        super().__init__(controller)
        self.select_cell = self._select_cell if select_cell is None else select_cell
        self.rank_cell = self._rank_cell if rank_cell is None else rank_cell
        self.min_area_px, self.half_life_s = min_area_px, half_life_s
        self.alignment_snapshot = None
        self.completed_successfully = False
        self.cell = None

    def build_helpers(self):
        focus = self.stage.focusHelper
        tracker = Tracker(self.select_cell, min_area_px=self.min_area_px,
            min_area_fraction=self.config.cell_mask_min_area_fraction,
            half_life_s=self.config.vision_half_life_s if self.half_life_s is None else self.half_life_s)
        self.searcher = Searcher(focus, ActionHelper(), self.rank_cell, tracker)
        self.scanner = Scanner(focus, ActionHelper())
        self.spearer = Spearer(self.unit, ActionHelper())
        models = self.stage._ensure_cell_track_helper()
        self.shifter = Shifter(segmentor=models.segmentor, matcher=models._matcher)

    @property
    def depth_tolerance_um(self):
        return self.spearer.DEPTH_TOLERANCE_UM

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
                    if self.mode == "Adaptive" and self.initialized:
                        self.observe()
                        self.calculate()
                    self.decide()
                    self.act()
            finally:
                self.finish()

    def prepare(self, cell=None):
        self.finished = self.speared = False
        self.completed_successfully = False
        self.cell = cell
        self.mode = self.controller.config.mode
        self.config = SimpleNamespace(**{name: getattr(self.controller.config, name) for name in (
            "cell_type", "cell_type_toggle", "cell_distance", "slice_start_distance",
            "max_locate_speed", "max_clearing_speed", "use_centroid", "track_cell",
            "search_speed", "search_z_radius_um", "scan_z_radius_um",
            "approach_cell_max_shift", "approach_pipette_speed", "search_xy_radius_um",
            "cell_mask_min_area_fraction", "vision_half_life_s")})
        self.config.x1_focus_mode = getattr(self.controller.config, 'x1_focus_mode', 'Anchor')
        self.config.x1_focus_score_threshold = getattr(self.controller.config, 'x1_focus_score_threshold', 0.5)
        self.config.scan_frame_stride = getattr(self.controller.config, 'scan_frame_stride', 2)
        self.config.final_search_z_radius_um = getattr(self.controller.config, 'final_search_z_radius_um', 10.)
        self.config.final_search_xy_radius_um = getattr(self.controller.config, 'final_search_xy_radius_um', 5.)
        self.config.track_max_missed_detections = getattr(self.controller.config, 'track_max_missed_detections', 3)
        self.config.track_max_unseen_travel_um = getattr(self.controller.config, 'track_max_unseen_travel_um', 3.0)
        self.config.cell_shift_fusion = getattr(self.controller.config, 'cell_shift_fusion', True)
        self.initialized = False
        self.recovering_cell = False
        self.recovery_attempted = False
        self.recovery_target_z = None
        self.visual_fallback = False
        self.final_search_reference_xy = None
        self.endpoint_wide_attempted = False
        self.interrupted = False
        if self.mode not in ("Manual", "Classic", "Adaptive", "Agent", "Training"):
            raise AutopatchError(f"Unsupported Approach mode: {self.mode}")
        if cell is None and self.mode != "Training":
            raise AutopatchError("Approach requires a selected cell")
        self.stage = self.controller.calibrated_stage
        self.unit = self.controller.calibrated_unit
        self.pipette_anchor = None
        if self.cell is not None:
            anchor = getattr(self.unit, '_trusted_tip_reference', None)
            if anchor is None:
                camera = self.stage.camera
                anchor = ((camera.width / 2, camera.height / 2),
                          self.controller.home_position, self.controller.home_stage_position[:2])
            self.pipette_anchor = tuple(np.array(value, dtype=float, copy=True) for value in anchor)
        self.initial_observation = None
        self.cell_observation = self.pipette_observation = None
        self.cell_search_motor_z = None
        self.calibration_observation = None
        self.operation = self._start
        self.next_action = None
        self.observation = self.alignment_snapshot = None
        self.begin_observations()
        self.controller.observer.set_reference(cell)
        self.controller.alignment_phase = self
        self.failure_gate(validate_cell=self.mode != "Training")
        if self.mode == "Adaptive":
            self.build_helpers()  # Fresh per-attempt state; shared models are reused.

    def project_pipette_tip(self, anchor, position):
        """Use the existing calibrated unit and stage anchor transform."""
        delta = np.asarray(position) - anchor[1]
        stage_delta = np.asarray(self.stage.position())[:2] - np.asarray(anchor[2])[:2]
        return (np.asarray(anchor[0])[:2] + self.unit.um_to_pixels_relative(delta)[:2]
                + self.stage.um_to_pixels_relative(stage_delta)[:2])

    def observe(self, **kwargs):
        anchor = self.pipette_anchor
        options = dict(cell_inference=True, pipette_inference=True, cached_ephys=True,
            pipette_anchor=anchor,
            cell_masks=(self.searcher.searching or self.operation == self.search or
                (self.operation == self.spear and self.config.track_cell)))
        options.update(kwargs)
        if getattr(self, 'visual_fallback', False):
            final_look = self.speared and self.operation == self.search
            options['cell_inference'] = final_look and kwargs.get('cell_inference', False)
            options['cell_masks'] = (final_look and self.final_search_reference_xy is not None
                                     and kwargs.get('cell_masks', False))
        self.observation = super().observe(**options)
        if self.initial_observation is None:
            self.initial_observation = deepcopy(self.observation)
        self.publish_alignment_snapshot()
        return self.observation

    def _select_cell(self, observation):
        """Associate masks with the selected cell using calibrated, motion-projected XY."""
        cell = observation.get("cell")
        if cell is None or self.cell is None:
            return None
        expected = (np.asarray(observation["stage"]["pixel_position"])
                    - np.asarray(self.cell[0])[:2])
        searcher = self.searcher
        tracker = searcher.tracker
        radius = self.config.search_xy_radius_um
        if getattr(self, "operation", None) == getattr(self, "spear", object()):
            radius = min(radius, 7.0)
        local_xy = None
        local_radius = 7.0
        if searcher.searching and getattr(self, "search_bounds", None) is not None:
            # Hunt's post-Spear search also follows the cell seen at descent start.
            if getattr(self.config, "hunt_limit_target_radius", True):
                radius = min(radius, 7.0)
            local_reference = getattr(self, "spear_cell_reference", None)
            if local_reference is not None:
                local_xy = np.asarray(observation["stage"]["pixel_position"]) - local_reference
        elif (hasattr(self, "search_bounds") and not searcher.searching and self.operation != self.search
              and getattr(self.config, "hunt_limit_target_radius", False)):
            radius = min(radius, self.config.hunt_target_radius)
        if tracker.last_cell_reference_xy is not None:
            projected = tracker.position(observation, self.cell[0])
            expected = expected + tracker.age_weight(tracker.last_cell_frame_time) * (projected - expected)
        if getattr(self, 'speared', False) and self.phase_name == 'Approach':
            radius = self.config.final_search_xy_radius_um
        if getattr(self, 'final_search_reference_xy', None) is not None:
            local_xy, local_radius = expected, radius
            expected = np.asarray(observation['stage']['pixel_position']) - self.final_search_reference_xy
        candidates = []
        for index, (point, target) in enumerate(zip(cell["cell_seed_points"], cell["cell_targets"])):
            point = np.asarray(point, dtype=float)
            if point.shape != (2,) or not np.isfinite(point).all():
                continue
            if local_xy is not None and np.linalg.norm(
                    self.stage.pixels_to_um_relative(np.r_[point - local_xy, 0.0])[:2]) > local_radius:
                continue
            distance = np.linalg.norm(self.stage.pixels_to_um_relative(np.r_[point - expected, 0.0])[:2])
            if distance <= radius and (
                not searcher.searching or searcher.matches_cell(observation, target, point, expected)
            ):
                candidates.append((distance, index, target))
        return min(candidates, key=lambda row: row[:2])[2] if candidates else None

    def _rank_cell(self, observation):
        """Rank detector evidence against the last cell depth, allowing real travel."""
        tracker = self.searcher.tracker
        if tracker.lost:
            return None
        cell = self.select_cell(observation)
        cells = observation['cell']
        index = next(i for i, target in enumerate(cells['cell_targets']) if target is cell)
        seed = np.asarray(cells['cell_seed_points'][index])
        detections = cells.get('cell_detections')
        rows = [] if detections is None else [row for row in detections
            if len(row) >= 3 and np.isfinite(row[:3]).all()]
        detection = min(rows, key=lambda row: np.linalg.norm(np.asarray(row[:2]) - seed), default=None)
        confidence = 1.0 if detection is None else float(np.clip(detection[2], 0.0, 1.0))
        origin = getattr(self, 'cell_search_motor_z', None)
        if origin is None:
            origin = getattr(self, 'descent_origin', None)
        travelled = 0.0 if origin is None else abs(self.unit.position()[2] - origin)
        scale = max(self.depth_tolerance_um, travelled)
        prior_z = self.cell_focus_z
        if getattr(self, 'final_search_reference_xy', None) is not None:
            prior_z = sum(self.searcher.sweep_bounds) / 2.0
        depth_error = abs(observation['stage']['raw_position'][2] - prior_z)
        return confidence / (1.0 + (depth_error / scale) ** 2)

    def publish_alignment_snapshot(self, observation=None):
        captured_sweep_frame = observation is not None
        if observation is None:
            observation = self.observation
        if observation is None or self.cell is None:
            return
        stage = observation["stage"]
        pipette = observation.get("pipette")
        target = self.select_cell(observation)
        accepted = None
        if target is not None and target["target_points_xy"]:
            accepted = dict(xy=np.asarray(target["target_points_xy"][0]).copy(),
                            stage_xy=np.asarray(stage["raw_position"])[:2].copy())
        detection = pipette["pipette_detection"] if pipette is not None else None
        if detection is not None and not Scanner.tip_matches_anchor(
                detection.get("tip_xy"), pipette["pixel_position"], self.stage,
                self.unit.config.pipette_detection_radius_um):
            detection = None
        pipette_position = np.asarray(pipette["raw_position"]).copy() if pipette is not None else None
        saved_tip = None
        if captured_sweep_frame and self.operation == self.scan and self.scanner.best_observation is not None:
            best = self.scanner.best_observation
            saved_tip = dict(xy=best["pipette"]["pipette_detection"]["tip_xy"],
                             stage_xy=best["stage"]["raw_position"][:2], pipette_xyz=best["pipette"]["raw_position"])
        self.alignment_snapshot = dict(operation=self.operation, cell_reference=self.cell[0],
            saved_tip_evidence=saved_tip,
            source_positions=dict(stage_xy=np.asarray(stage["raw_position"])[:2].copy(),
                                  pipette_xyz=pipette_position),
            stage_position=np.asarray(stage["raw_position"]).copy(),
            manipulator_position=pipette_position,
            accepted_cell=accepted, stale=observation["image"]["frame_number"] is None,
            calibrated_tip_xy=pipette["pixel_position"] if pipette is not None else None,
            detected_tip_xy=None if detection is None else detection.get("tip_xy"))

    def calculate(self):
        self.calculate_geometry()

    def calculate_geometry(self):
        reference = self.observation["reference"]
        motor_z = self.observation["pipette"]["raw_position"][2]
        self.cell_focus_z = (self.cell_observation["stage"]["raw_position"][2]
            if self.cell_observation is not None else reference["cell_position"][2])
        if self.pipette_observation is None:
            self.pipette_z = (reference["home_stage_position"][2] + motor_z
                              - reference["home_position"][2])
        else:
            self.pipette_z = (self.pipette_observation["stage"]["raw_position"][2] + motor_z
                              - self.pipette_observation["pipette"]["raw_position"][2])
        self.goal_distance = (reference["cell_distance"] if reference["cell_type"] == "Plate"
            or (reference["cell_type"] == "Slice" and reference["cell_type_toggle"])
            else reference["slice_start_distance"])
        if not all(isfinite(v) for v in (self.cell_focus_z, self.pipette_z, self.goal_distance)):
            raise AutopatchError("Approach needs finite depth geometry")
        if self.goal_distance < 0:
            raise AutopatchError("Approach separation must be nonnegative")
        self.hover_z = self.cell_focus_z - self.goal_distance

    def calculate_progress(self):
        return self.hover_z - self.pipette_z

    def decide(self):
        self.next_action = None
        if self.finished or self.awaiting_operator:
            return
        if not self.initialized:
            self.next_action = self._start
            return
        if self.mode == "Training":
            return
        self.next_action = self.operation
        if self.operation == self.spear and not getattr(self, 'visual_fallback', False):
            remaining_depth = self.calculate_progress()
            if abs(remaining_depth) <= self.depth_tolerance_um:
                # Already at the stopping distance: do the final alignment pass.
                self.speared = True
                self.next_action = self.search
            elif remaining_depth < 0:
                self.controller.warning("Pipette is past the intended depth; continuing final alignment")
                self.speared = True
                self.next_action = self.search

    def act(self):
        if self.next_action is not None and not self.action_gate():
            raise AutopatchError("Approach action is not permitted")
        if self.next_action == self._start:
            self._start()
        elif self.next_action == self.search:
            self.search()
        elif self.next_action == self.shift:
            self.shift()
        elif self.next_action == self.scan:
            self.scan()
        elif self.next_action == self.spear:
            self.spear()
        else:
            self.controller.sleep(0.1)

    def search(self, *, bounds=None):
        endpoint = getattr(self, 'speared', False) and self.phase_name == 'Approach'
        if endpoint:
            def confirm(z, reference):
                radius = self.config.search_z_radius_um
                self.final_search_reference_xy = reference
                try:
                    return self.searcher.run(self, bounds=(z - radius, z + radius), speed=self.config.search_speed)
                finally:
                    self.final_search_reference_xy = None

            def complete(result):
                if result is None:
                    return None
                if result == 'not_found':
                    self.searcher.tracker.using_fallback = True
                    self.controller.info('Approach endpoint did not confirm the cell; retaining the previous depth')
                    self.finish(success=True)
                    return result
                self.cell_observation = result
                self.cell_search_motor_z = self.observation['pipette']['raw_position'][2]
                self.visual_fallback = False
                self.shift_cell, self.operation = True, self.shift
                return result

            if not getattr(self, 'visual_fallback', False):
                radius = self.config.search_z_radius_um
                local_bounds = bounds if bounds is not None else (self.cell_focus_z - radius, self.cell_focus_z + radius)
                result = self.searcher.run(self, bounds=local_bounds, speed=self.config.search_speed)
                if result != 'not_found':
                    return complete(result)
            else:
                probe = self.observe(cell_inference=True, cell_masks=False, pipette_inference=False)
                if probe is None or not self.motion_checkpoint():
                    return None
                cells = probe.get('cell') or {}
                raw = cells.get('cell_detections')
                rows = [] if raw is None else list(raw)
                nomination = dict(probe, cell=dict(cells,
                    cell_seed_points=[row[:2] for row in rows], cell_targets=rows))
                selected = self._select_cell(nomination) if rows else None
                if selected is not None:
                    reference = np.asarray(probe['stage']['pixel_position']) - np.asarray(selected[:2])
                    result = confirm(float(probe['stage']['raw_position'][2]), reference)
                    if result != 'not_found':
                        return complete(result)
            if getattr(self, 'endpoint_wide_attempted', False):
                return complete('not_found')
            self.endpoint_wide_attempted = True
            radius = getattr(self.config, 'final_search_z_radius_um', 10.)
            detection = self.searcher.look_for_cell(self,
                bounds=(self.cell_focus_z - radius, self.cell_focus_z + radius))
            if detection is None or detection == 'not_found':
                return complete(detection)
            return complete(confirm(*detection))

        if bounds is None:
            radius = self.config.search_z_radius_um
            bounds = (self.cell_focus_z - radius, self.cell_focus_z + radius)
        result = self.searcher.run(self, bounds=bounds, speed=self.config.search_speed)
        if result is None:
            return None
        if result == 'not_found':
            self.searcher.tracker.using_fallback = True
            if (getattr(self, 'recovery_attempted', False) and self.recovering_cell
                    and self.phase_name == 'Approach'):
                self.visual_fallback = True
                self.recovering_cell = False
                self.operation = self.spear
                self.controller.info('Recovery sweep found no cell; continuing to the saved Approach target')
                return result
        else:
            self.cell_observation = result
            self.cell_search_motor_z = self.observation['pipette']['raw_position'][2]
            self.visual_fallback = False
        self.shift_cell, self.operation = True, self.shift
        return result

    def scan(self):
        radius = self.config.scan_z_radius_um
        result = self.scanner.run(self,
            bounds=(self.pipette_z - radius, self.pipette_z + radius),
            speed=self.config.search_speed)
        if result is None:
            return
        self.pipette_observation = None if result == "not_found" else result
        self.shift_cell, self.operation = False, self.shift

    def shift(self):
        helper = self.searcher if self.shift_cell else self.scanner
        viewing_z = self.cell_focus_z if self.shift_cell else self.pipette_z
        positioned = helper.move_to(self, viewing_z, self.config.search_speed)
        if positioned is None:
            return
        if self.shift_cell:
            result = self.shifter.shift_cell(self, target_xy=None,
                max_error_px=self.config.approach_cell_max_shift)
        else:
            center = (self.stage.camera.width / 2, self.stage.camera.height / 2)
            result = self.shifter.shift_pipette(self, desired_xy=center,
                speed=self.config.approach_pipette_speed)
        if result is None:
            return
        if self.shift_cell:
            self.operation = self.scan
            self.recovering_cell = False
        elif self.speared:
            self.finish(success=True)
        else:
            self.operation = self.spear

    def spear(self):
        if getattr(self, 'visual_fallback', False):
            result = self.spearer.run(self, target_z=self.recovery_target_z,
                speed=abs(self.config.max_clearing_speed), tracker=None)
            if result is not None:
                self.speared = True
                self.operation = self.search
            return
        viewing_z = self.cell_focus_z if self.config.track_cell else self.hover_z
        positioned = self.searcher.move_to(self, viewing_z, self.config.search_speed)
        if positioned is None:
            return
        self.calculate_geometry()
        motor_z = self.observation["pipette"]["raw_position"][2]
        target_z = motor_z + max(0.0, self.calculate_progress())
        speed = abs(self.config.max_clearing_speed)
        tracker = self.searcher.tracker if self.config.track_cell else None
        result = self.spearer.run(self, target_z=target_z, speed=speed, tracker=tracker)
        if result == "cell_lost":
            if self.phase_name == 'Approach':
                self.recovery_target_z = target_z
                if self.recovery_attempted:
                    self.visual_fallback = True
                    self.recovering_cell = False
                    self.controller.info('Cell lost again after recovery; continuing to the saved Approach target')
                    return
                self.recovery_attempted = True
            self.recovering_cell = True
            self.operation = self.search
            return
        if result is not None:
            self.speared = True
            self.operation = self.search

    def _start(self):
        config, controller = self.config, self.controller
        if self.mode == "Training":
            controller.amplifier.start_patch()
            self.initialized = True
            self.awaiting_operator = True
            controller.info("Training mode: manual movement. Click Success or Abort to finish.")
            return
        self.stage.set_max_speed(config.max_locate_speed)
        self.unit.set_max_speed(config.max_locate_speed)
        plate = self.mode != "Adaptive" and config.cell_type == "Plate"
        if plate:
            controller.move_group_down(config.slice_start_distance - config.cell_distance)
            controller.sleep(0.1)
            controller.fine_calibrate_pipette()
        controller.amplifier.start_patch()
        if self.mode == "Adaptive":
            self.failure_gate()
            self.success_gate()
            self.unit.microscope.set_max_speed(config.max_locate_speed)
            self.stage.focusHelper.relocate(float(self.cell[0][2]))
            self.failure_gate()
            self.success_gate()
            self.operation = self.search
            self.initialized = True
            return
        self.align(config.cell_distance if plate else config.slice_start_distance)
        if config.cell_type == "Slice" and config.cell_type_toggle:
            self.clear_to_cell()
            self.align(config.cell_distance)
        self.initialized = True
        self.finish(success=True)

    def align(self, cell_distance):
        self.failure_gate()
        self.success_gate()
        focus = self.stage.focusHelper
        self.unit.microscope.move_to_floor()
        focus.relocate(self.cell[0][2])
        if self.config.cell_type_toggle:
            self.stage.center_on_cell(self.cell, use_centroid=self.config.use_centroid,
                max_error_px=self.config.approach_cell_max_shift)
            focus.relocate(self.cell[0][2] - cell_distance)
            self.unit.center_pipette(speed=self.config.approach_pipette_speed)
            focus.relocate(self.cell[0][2])

    def clear_to_cell(self):
        # Preserve the old Slice clearing command and measured-travel condition.
        try:
            self.failure_gate(validate_cell=True)
            self.success_gate()
            config = self.config
            cell_hover_pos = config.cell_distance - config.slice_start_distance
            self.unit.microscope.relative_move(-config.cell_distance)
            start_pos = self.unit.position()
            deadline = time.monotonic() + 45.0
            self.unit.absolute_move_group_velocity([0, 0, config.max_clearing_speed])
            while start_pos[2] - self.unit.position()[2] > cell_hover_pos:
                self.failure_gate()
                self.success_gate()
                if time.monotonic() >= deadline:
                    self.controller.warning("Slice clearing reached 45 seconds; stopping clearing")
                    break
                self.controller.sleep(0.1)
        finally:
            self.stop_motions()

    def action_gate(self):
        self.failure_gate()
        self.success_gate()
        if self.finished or self.awaiting_operator or self.interrupted:
            return False
        if self.mode == "Training":
            return self.next_action == self._start and not self.initialized
        if self.mode != "Adaptive":
            return self.next_action == self._start
        # Helpers validate their own finite positions, speed and endpoint.
        # Missing visual detections are allowed through calibrated fallback.
        return self.next_action in (self._start, self.search, self.scan, self.shift, self.spear)

    def motion_checkpoint(self):
        self.failure_gate()
        self.success_gate()
        return not (self.finished or self.awaiting_operator or self.interrupted)

    def failure_gate(self, *, validate_cell=False):
        self.controller.abort_if_requested()
        if validate_cell:
            self.controller.isrigready()
            if self.controller.rig_ready is False:
                raise AutopatchError("Rig not ready for Approach")
            if self.cell is None:
                raise AutopatchError("Approach requires a selected cell")

    def success_gate(self, goal=None):
        self.controller.success_if_requested()
        if goal is None or self.awaiting_operator:
            return False
        self.goal_event = goal is True
        if self.goal_event and self.mode == "Training":
            self.awaiting_operator = True
            self.controller.info("Training mode: goal reached. Click Success or Abort to finish.")
            return False
        return self.goal_event

    def stop_motions(self):
        error = None
        for device in (self.controller.calibrated_unit, self.controller.calibrated_stage,
                       self.controller.calibrated_unit.microscope):
            try:
                device.stop()
            except Exception as exc:
                error = error or exc
        if error is not None and sys.exc_info()[0] is None:
            raise error

    def finish(self, *, success=False):
        if self.finished:
            return
        error = None
        try:
            self.stop_motions()
        except Exception as exc:
            error = exc
        for device, speed in ((self.controller.calibrated_stage, 10000),
                              (self.controller.calibrated_unit, 100000)):
            try:
                device.set_max_speed(speed)
            except Exception as exc:
                error = error or exc
        self.finished = True
        if self.controller.alignment_phase is self:
            self.controller.alignment_phase = None
        if error is not None and sys.exc_info()[0] is None:
            raise error
        if success and error is None:
            self.completed_successfully = True
            self.success_gate(True)
            if not self.awaiting_operator:
                self.complete_success()
