import time

import numpy as np
import torch

from patcherbot.deepLearning.cellSegmentor import CellSegmentor2
from patcherbot.deepLearning.PatchMatcher import PatchMatcher
from .StageCalHelper import FocusHelper
from .CellTrackHelper import CellTrackHelper


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

    def get_supported_alignment(self, cell, query, pixels_to_um, tolerance_um):
        """Locate the saved cell using a majority of consistent feature translations."""
        reference = cell[1]
        match = self.matcher.match(self._match_image(reference), self._match_image(query), include_overlay=False)
        points = self.matcher._get_matched_keypoints(match['feats0'], match['feats1'], match['matches'])
        if points is None:
            raise ValueError('No LightGlue correspondences for selected cell')
        shifts = (points[1] - points[0]).detach().cpu().numpy()
        shifts = shifts[np.isfinite(shifts).all(axis=1)]
        if len(shifts) < 3:
            raise ValueError('Fewer than three LightGlue correspondences for selected cell')
        basis = np.column_stack([pixels_to_um([1., 0., 0.])[:2], pixels_to_um([0., 1., 0.])[:2]])
        residual_um = (shifts - np.median(shifts, axis=0)) @ basis.T
        supported = shifts[np.linalg.norm(residual_um, axis=1) <= tolerance_um]
        if len(supported) < 3 or 2 * len(supported) <= len(shifts):
            raise ValueError('LightGlue correspondences disagree about cell displacement')
        return np.array(reference.shape[:2][::-1]) / 2.0 + np.median(supported, axis=0)

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
                phase.controller.warning("No fresh cell frame after stage settling; using cell-position fallback")
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

    def _fallback_cell_point(self, phase, observation):
        """Project the last reliable cell anchor, otherwise the original selection."""
        stage = phase.controller.calibrated_stage
        stage_xy = np.asarray(observation["stage"]["pixel_position"], dtype=float)
        calibrated = stage_xy - np.asarray(phase.cell[0], dtype=float)[:2]
        reference = getattr(phase.searcher.tracker, "last_cell_reference_xy", None)
        if reference is not None:
            predicted = stage_xy - np.asarray(reference, dtype=float)
            if self._in_frame(predicted, stage.camera):
                return predicted, "previous reliable prediction"
        return calibrated, "calibration"

    def _recover_cell_shift(self, phase, observation, max_error_px):
        """Make one bounded fallback correction without accepting the failed mask."""
        stage = phase.controller.calibrated_stage
        target, source = self._fallback_cell_point(phase, observation)
        phase.searcher.tracker.using_fallback = True
        phase.controller.warning(f"Cell Shift verification failed; using {source} at {target}")
        center = np.array([stage.camera.width / 2, stage.camera.height / 2])
        if not self._in_frame(target, stage.camera) or np.array_equal(np.rint(target), np.rint(center)):
            return observation
        if not phase.motion_checkpoint() or not phase.action_gate():
            return None
        stage.center_on_cell(phase.cell, detected_px=target, max_error_px=max_error_px)
        return phase.observe() if phase.motion_checkpoint() else None

    def _cell_shift_target(self, phase, observation, *, identity_reference=None):
        """Combine the saved reference match, calibrated anchor and detector evidence."""
        stage = phase.controller.calibrated_stage
        calibrated = np.asarray(observation['stage']['pixel_position']) - np.asarray(phase.cell[0])[:2]
        previous, _ = self._fallback_cell_point(phase, observation)
        if identity_reference is not None:
            previous = np.asarray(observation['stage']['pixel_position']) - identity_reference
        radius = float(np.clip(getattr(phase.config, 'final_search_xy_radius_um', 5.), 3., 5.))
        def compatible(point):
            return (self._in_frame(point, stage.camera) and np.linalg.norm(
                stage.pixels_to_um_relative(np.r_[np.asarray(point)-previous, 0.])[:2]) <= radius)
        fusion_anchor = calibrated if compatible(calibrated) else previous
        calibration_weight = 0.35
        baseline = (getattr(phase, 'calibration_observation', None)
                    or getattr(phase, 'initial_observation', None))
        if baseline is not None:
            calibration_weight *= phase.searcher.tracker.age_weight(
                baseline['image']['time'], now=observation['image']['time'])
            advance = max(0.0, observation['pipette']['raw_position'][2]
                          - baseline['pipette']['raw_position'][2])
            calibration_weight /= 1.0 + (advance / stage.focusHelper.TIP_STABILITY_UM) ** 2
            calibration_weight = max(calibration_weight, 1e-6)
        template = None
        try:
            image = phase.controller.observer.get_inference_image(observation['image']['frame_number'])
            height, width = phase.cell[1].shape[:2]
            image_height, image_width = image.shape[:2]
            if min(height, width) <= 0 or height > image_height or width > image_width:
                raise ValueError('Reference does not fit the current cell image')
            origin = np.clip(np.rint(previous - np.array([width, height])/2).astype(int),
                             [0, 0], [image_width-width, image_height-height])
            x, y = origin
            template = self.get_supported_alignment(phase.cell, image[y:y+height, x:x+width],
                stage.pixels_to_um_relative, radius) + origin
            if not compatible(template):
                phase.controller.warning('Cell Shift rejected reference outside selected-cell identity radius')
                template = None
        except Exception as error:
            phase.controller.info(f'Cell Shift reference unavailable: {error}')
        candidates = []
        if getattr(phase.config, 'cell_shift_fusion', True):
            rows = (observation.get('cell') or {}).get('cell_detections')
            for row in (() if rows is None else rows):
                if (len(row) >= 3 and np.isfinite(row[:3]).all() and row[2] >= 0
                        and compatible(row[:2])):
                    candidates.append(row)
        selected, target = CellTrackHelper.fuse_candidates(stage, candidates, fusion_anchor,
            template_xy=template, previous_xy=previous, expected_weight=calibration_weight)
        if selected is not None and template is None:
            # Keep the prior's weight when registration fails. Weighting both
            # sources by the same support would cancel distance suppression.
            offset = stage.pixels_to_um_relative(np.r_[np.asarray(selected[:2]) - previous, 0.])[:2]
            distance = np.linalg.norm(offset)
            support = 1.0 / (1.0 + (distance / stage.focusHelper.TIP_STABILITY_UM) ** 2)
            confidence = max(float(selected[2]), 1e-6)
            target = np.average([previous, selected[:2]], axis=0,
                                weights=[0.35, confidence * support])
        if target is not None:
            source = 'reference/detector/calibration fusion' if template is not None else 'detector/position-prior fusion'
        elif template is not None:
            target = np.average([fusion_anchor, template], axis=0, weights=[calibration_weight, 1.0])
            source = 'reference/calibration fusion'
        else:
            target, source = previous, 'previous reliable prediction' if identity_reference is not None else self._fallback_cell_point(phase, observation)[1]
        if not compatible(target):
            target, source = previous, 'selected-cell identity fallback'
            template = selected = None
        phase.searcher.tracker.using_fallback = template is None and selected is None
        phase.controller.info(f'Cell Shift evidence: prior_xy={calibrated}; prediction_xy={previous}; template_xy={template}; '
                              f'detector={selected}; fused_xy={target}; source={source}; '
                              f'reference_used={template is not None}; calibration_weight={calibration_weight}')
        return np.asarray(target, dtype=float), source

    def shift_cell(self, phase, *, target_xy, max_error_px):
        stage = phase.controller.calibrated_stage
        try:
            if not phase.motion_checkpoint():
                return None
            observation = phase.observe()
            if phase.interrupted:
                return None
            searcher = phase.searcher
            starting_point, _ = self._fallback_cell_point(phase, observation)
            identity_reference = np.asarray(observation['stage']['pixel_position']) - starting_point
            radius = float(np.clip(getattr(phase.config, 'final_search_xy_radius_um', 5.), 3., 5.))
            if target_xy is None:
                target_xy, _ = self._cell_shift_target(phase, observation, identity_reference=identity_reference)
            target_xy = np.asarray(target_xy, dtype=float)
            center = np.array([stage.camera.width / 2, stage.camera.height / 2])
            for attempt in (1, 2):
                projected = np.asarray(observation['stage']['pixel_position']) - identity_reference
                distance = np.linalg.norm(stage.pixels_to_um_relative(np.r_[target_xy-projected, 0.])[:2])
                if not np.isfinite(distance) or distance > radius:
                    target_xy = projected
                    searcher.tracker.using_fallback = True
                    phase.controller.warning('Cell Shift rejected correction outside selected-cell identity radius')
                if not self._in_frame(target_xy, stage.camera):
                    phase.controller.warning("Cell position outside the frame; skipping XY shift")
                    return observation
                if not phase.action_gate():
                    return None
                stage.center_on_cell(phase.cell, detected_px=target_xy, max_error_px=max_error_px)
                if not phase.motion_checkpoint():
                    return None
                observation, fresh = self._settled_cell_frame(phase, observation)
                if observation is None or not phase.motion_checkpoint():
                    return None
                if not fresh:
                    target_xy = np.asarray(observation['stage']['pixel_position']) - identity_reference
                    searcher.tracker.using_fallback = True
                    continue
                target_xy, _ = self._cell_shift_target(phase, observation, identity_reference=identity_reference)
                residual = stage.pixels_to_um_relative(np.r_[center - target_xy, 0])[:2]
                phase.controller.info(f"Cell Shift after move {attempt}: frame={observation['image']['frame_number']}, "
                                      f"fused_xy={target_xy}, residual_um={residual}")
                if not searcher.tracker.using_fallback:
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
        """Use consistent visual evidence, falling back to calibrated tip XY."""
        expected = np.asarray(observation["pipette"]["pixel_position"], dtype=float)
        if not self._in_frame(expected, phase.stage.camera):
            return None
        saved = phase.pipette_observation
        if saved is None:
            return expected
        detected = self._project_scan_tip(phase, observation)
        if detected is None:
            return expected
        weight = (phase.scanner.weight_for_score(saved["pipette"]["pipette_detection"]["focus_score"])
                  * phase.searcher.tracker.age_weight(saved["image"]["time"]))
        return expected + weight * (detected - expected)

    def shift_pipette(self, phase, *, desired_xy=None, speed, target_xy=None):
        unit = phase.controller.calibrated_unit
        try:
            if not phase.motion_checkpoint():
                return None
            observation = phase.observe()
            if phase.interrupted:
                return None
            if desired_xy is None:
                desired_xy = (phase.stage.camera.width / 2, phase.stage.camera.height / 2)
            desired_xy = np.asarray(desired_xy, dtype=float)
            if not self._in_frame(desired_xy, phase.stage.camera):
                phase.stop_motions()
                phase.awaiting_operator = True
                phase.controller.warning(f"Pipette destination outside the frame or invalid: {desired_xy}; holding motion")
                return None
            expected_xy = np.asarray(observation["pipette"]["pixel_position"], dtype=float)
            if not self._in_frame(expected_xy, phase.stage.camera):
                phase.controller.warning(f"Pipette calibration outside the frame: {expected_xy}; continuing at calibrated depth")
                phase.pipette_observation = None
                return observation
            target_xy = self._project_scan_tip(phase, observation)
            options = dict(expected_px=expected_xy, max_deviation_um=unit.config.pipette_detection_radius_um)
            destination = np.asarray(desired_xy, dtype=float)
            if target_xy is None:
                target_xy = expected_xy
                phase.pipette_observation = None
            else:
                score = phase.pipette_observation["pipette"]["pipette_detection"]["focus_score"]
                weight = phase.scanner.weight_for_score(score)
                # Fuse the position before the motion handoff. The backend must
                # anchor and move the same accepted estimate to the fixed target.
                target_xy = expected_xy + weight * (target_xy - expected_xy)
            if np.array_equal(target_xy, destination):
                return observation
            if not phase.action_gate():
                if phase.interrupted:
                    return None
                raise RuntimeError("Pipette Shift is not permitted")
            phase.controller.info(f"Pipette Shift tip_xy={target_xy}; calibrated_xy={expected_xy}; "
                                  f"destination_xy={desired_xy}; command_xy={destination}")
            completed = unit.direct_pipette(desired_px=destination, detected_px=target_xy, speed=speed, **options)
            if not phase.motion_checkpoint():
                return None
            if completed is False:
                raise RuntimeError(unit.last_pipette_move_failure or "Pipette Shift failed")
            return phase.observe()
        finally:
            unit.stop()
