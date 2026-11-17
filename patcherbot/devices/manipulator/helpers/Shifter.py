import time

import numpy as np
import torch

from patcherbot.deepLearning.cellSegmentor import CellSegmentor2
from patcherbot.deepLearning.PatchMatcher import PatchMatcher
from .StageCalHelper import FocusHelper


class Shifter:
    def __init__(self, segmentor=None, matcher=None):
        self.segmentor = CellSegmentor2() if segmentor is None else segmentor
        self.matcher = PatchMatcher() if matcher is None else matcher
        self.observation = None
        self.cell = None

    def update(self, observation, cell):
        self.observation = observation
        self.cell = cell

    def segment(self, image, point):
        """Segment one prompted cell and return its mask and interior targets."""
        mask = self.segmentor.segment(
            image=self.segmentor._prepare_for_segmentation(image),
            input_point=np.asarray([point], dtype=np.float32),
            input_label=np.asarray([1], dtype=np.int32),
            multimask_output=False,
        )
        return self.segmentor.get_mask_target(mask)

    def _match_image(self, image):
        if not isinstance(image, np.ndarray):
            return image  # Existing PatchMatcher path/tensor inputs stay supported.
        bgr = self.segmentor._prepare_for_segmentation(image)
        rgb = bgr[..., ::-1].copy()
        return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

    def match(self, reference_image, image):
        """Return reference-to-current translation through the existing LightGlue backend."""
        return self.matcher.find_target(
            self._match_image(reference_image), self._match_image(image))["translation"]

    @staticmethod
    def get_error(target_xy, desired_xy):
        """Return the XY correction from the target to the desired image position."""
        return np.asarray(desired_xy, dtype=float) - np.asarray(target_xy, dtype=float)

    @staticmethod
    def _in_frame(point, camera):
        point = np.asarray(point, dtype=float)
        return (point.shape == (2,) and np.isfinite(point).all()
                and np.all(point >= 0) and np.all(point < [camera.width, camera.height]))

    @staticmethod
    def weighted_tip(detected_xy, calibrated_xy, weight, pixels_to_um, max_refinement_um):
        if not 0.0 <= weight <= 1.0 or not np.isfinite(max_refinement_um) or max_refinement_um <= 0:
            raise ValueError("Need a finite anchor limit and a weight in [0, 1]")
        calibrated = np.asarray(calibrated_xy, dtype=float)
        offset = np.asarray(detected_xy, dtype=float) - calibrated
        if calibrated.shape != (2,) or offset.shape != (2,) or not np.isfinite(offset).all():
            raise ValueError("Need finite calibrated and detected XY")
        distance_um = np.linalg.norm(pixels_to_um(np.r_[offset, 0.0])[:2])
        if not np.isfinite(distance_um):
            raise ValueError("Invalid calibrated refinement distance")
        if distance_um > max_refinement_um:
            offset = offset * (max_refinement_um / distance_um)
        return calibrated + weight * offset

    def get_alignment(self, cell, image, reference_point=None):
        """Locate the selected reference point in the current image."""
        if image is None or cell[1] is None:
            raise ValueError("Cell registration needs reference and current images")
        height, width = cell[1].shape[:2]
        point = np.asarray((width / 2, height / 2) if reference_point is None
                           else reference_point, dtype=float)
        target = point + np.asarray(self.match(cell[1], image), dtype=float)
        if target.shape != (2,) or not np.isfinite(target).all():
            raise ValueError("Registration did not produce finite cell XY")
        return target

    @staticmethod
    def _settled_cell_frame(phase, previous):
        settled_wall, settled_tick = time.time(), time.monotonic()
        read_frame = getattr(phase.stage.camera, "last_raw_frame_data", None)
        observation = previous
        while phase.motion_checkpoint():
            # Wait for a retrieval started after encoder settling before paying for inference.
            sample = read_frame(include_timing=True) if read_frame is not None else None
            if read_frame is None or (sample is not None and
                    (sample[3].get("retrieval_started_at") or 0) >= settled_tick):
                observation = phase.observe(cell_masks=True, pipette_inference=False)
                frame = observation["image"]
                if (frame["frame_number"] is not None and frame["time"] is not None
                        and frame["time"] - 0.0005 >= settled_wall
                        and (sample is None or frame["frame_number"] >= sample[0])):
                    return observation, True
            if time.monotonic() - settled_tick >= FocusHelper.CONFIRM_TIMEOUT_S:
                phase.controller.warning("No fresh cell frame after stage settling; keeping the last correction")
                return observation, False
            phase.controller.sleep(0.01)
        return None, False

    @staticmethod
    def _verified_cell_point(phase, observation, winner, winner_point):
        """Use the selected cell's fresh interior point nearest the projected winner."""
        if abs(observation["stage"]["raw_position"][2] - winner["stage"]["raw_position"][2]) > phase.depth_tolerance_um:
            return None
        tracker = phase.searcher.tracker
        cell = tracker.update(observation, retain=False)
        if tracker.lost or not cell["target_points_xy"]:
            return None
        delta = np.asarray(observation["stage"]["pixel_position"]) - winner["stage"]["pixel_position"]
        projected = winner_point + delta
        if not phase.searcher._contains(cell["mask"], projected):
            return None
        point = min(cell["target_points_xy"], key=lambda p: np.linalg.norm(p - projected))
        offset = phase.stage.pixels_to_um_relative(np.r_[point - projected, 0.0])
        return np.asarray(point) if np.linalg.norm(offset[:2]) <= phase.config.search_xy_radius_um else None

    def shift_cell(self, phase, *, target_xy, max_error_px):
        stage = phase.controller.calibrated_stage
        try:
            if not phase.motion_checkpoint():
                return None
            observation = phase.observe()
            if phase.interrupted:
                return None
            stage_xy = np.asarray(observation["stage"]["pixel_position"])
            calibrated = stage_xy - np.asarray(phase.cell[0])[:2]
            radius = phase.config.search_xy_radius_um
            searcher = phase.searcher
            source = 'supplied point'
            winner = phase.cell_observation
            winner_point = None
            if target_xy is None:
                if winner is not None and searcher.best_target_points_xy and not searcher.tracker.using_fallback:
                    winner_stage_xy = np.asarray(winner["stage"]["pixel_position"])
                    winner_point = np.asarray(searcher.best_target_points_xy[0])
                    target_xy = winner_point + stage_xy - winner_stage_xy
                    source = f'Search interior point, frame {winner["image"]["frame_number"]}'
                else:
                    target_xy, source = calibrated, 'calibration'
            target_xy = np.asarray(target_xy, dtype=float)
            if (not self._in_frame(target_xy, stage.camera)
                    or not np.linalg.norm(stage.pixels_to_um_relative(np.r_[target_xy - calibrated, 0.0])[:2]) <= radius):
                phase.controller.warning(f"Cell Shift point outside match radius or frame: target={target_xy}; using calibration")
                target_xy, source, winner_point = calibrated, 'calibration', None
            phase.controller.info(f"Cell Shift target_xy={target_xy}; calibrated_xy={calibrated}; source={source}")
            center = np.array([stage.camera.width / 2, stage.camera.height / 2])
            for attempt in (1, 2):
                if not self._in_frame(target_xy, stage.camera):
                    phase.controller.warning("Cell position outside the frame; skipping XY shift")
                    return observation
                if not phase.action_gate():
                    return None
                stage.center_on_cell(phase.cell, detected_px=target_xy, max_error_px=max_error_px)
                if not phase.motion_checkpoint():
                    return None
                if winner_point is None:
                    return phase.observe()
                observation, fresh = self._settled_cell_frame(phase, observation)
                if observation is None or not phase.motion_checkpoint():
                    return None
                if not fresh:
                    return observation
                target_xy = self._verified_cell_point(phase, observation, winner, winner_point)
                if target_xy is None:
                    phase.controller.warning("Cell Shift could not verify the selected mask; keeping the last correction")
                    return observation
                residual = stage.pixels_to_um_relative(np.r_[center - target_xy, 0])[:2]
                phase.controller.info(f"Cell Shift after move {attempt}: frame={observation['image']['frame_number']}, "
                                      f"interior_xy={target_xy}, residual_um={residual}")
                searcher.tracker.remember(observation, target_xy)
                if np.array_equal(np.rint(target_xy), np.rint(center)):
                    return observation
            phase.controller.warning("Cell remains off center after two stage moves; keeping the last verified position")
            return observation
        finally:
            stage.stop()

    def _project_scan_tip(self, phase, observation):
        """Check the actual sweep measurement against an independent calibration."""
        winner = phase.pipette_observation
        if winner is None:
            return None
        saved = winner["pipette"]
        detection = saved["pipette_detection"]
        stage, unit = phase.stage, phase.unit
        anchor = (detection["tip_xy"], saved["raw_position"], winner["stage"]["raw_position"][:2])
        tip = phase.project_pipette_tip(anchor, observation["pipette"]["raw_position"])
        expected = observation["pipette"]["pixel_position"]
        if not phase.scanner.tip_matches_anchor(tip, expected, stage, unit.config.pipette_detection_radius_um):
            return None
        return np.asarray(tip, dtype=float)

    def pipette_target(self, phase, observation):
        """Estimate tip XY for Hunt's distance-to-mask calculation."""
        expected = np.asarray(observation["pipette"]["pixel_position"], dtype=float)
        if not self._in_frame(expected, phase.stage.camera):
            return None
        saved_observation = phase.pipette_observation
        if saved_observation is None:
            return expected
        detection = saved_observation["pipette"]["pipette_detection"]
        weight = (phase.scanner.weight_for_score(detection["focus_score"])
            * phase.searcher.tracker.age_weight(saved_observation["image"]["time"]))
        if weight <= 0.0:
            return expected
        detected = self._project_scan_tip(phase, observation)
        return expected if detected is None else expected + weight * (detected - expected)

    def shift_pipette(self, phase, *, desired_xy, speed, target_xy=None):
        unit = phase.controller.calibrated_unit
        try:
            if not phase.motion_checkpoint():
                return None
            observation = phase.observe()
            if phase.interrupted:
                return None
            expected_xy = np.asarray(observation["pipette"]["pixel_position"], dtype=float)
            if not self._in_frame(expected_xy, phase.stage.camera):
                phase.controller.warning(f"Pipette calibration outside the frame: {expected_xy}; continuing at calibrated depth")
                phase.pipette_observation = None
                return observation
            target_xy = self._project_scan_tip(phase, observation)
            options = {}
            destination = np.asarray(desired_xy, dtype=float)
            if target_xy is None:
                phase.controller.warning(f"Pipette Shift has no verified tip; using calibrated XY {expected_xy}")
                phase.pipette_observation = None
                target_xy = expected_xy
            else:
                score = phase.pipette_observation["pipette"]["pipette_detection"]["focus_score"]
                weight = phase.scanner.weight_for_score(score)
                # direct_pipette subtracts measured XY; this blends the resulting
                # correction from fully calibrated (weight 0) to fully visual (1).
                destination = destination + (1.0 - weight) * (target_xy - expected_xy)
                options = dict(expected_px=expected_xy, max_deviation_um=unit.config.pipette_detection_radius_um)
            if np.array_equal(target_xy, destination):
                return observation
            if not phase.action_gate():
                if phase.interrupted:
                    return None
                raise RuntimeError("Pipette Shift is not permitted")
            phase.controller.info(f"Pipette Shift tip_xy={target_xy}; calibrated_xy={expected_xy}; "
                                  f"center_xy={desired_xy}; command_xy={destination}")
            completed = unit.direct_pipette(desired_px=destination, detected_px=target_xy, speed=speed, **options)
            if not phase.motion_checkpoint():
                return None
            if completed is False:
                raise RuntimeError(unit.last_pipette_move_failure or "Pipette Shift failed")
            return phase.observe()
        finally:
            unit.stop()
