import math
import time
from collections import deque
from copy import deepcopy

import numpy as np

from .Scanner import FocusCaptureIncomplete, capture_checkpoint, position_focus, sweep_captured_focus
from .StageCalHelper import FocusHelper

class Searcher:
    """Same continuous trajectory, with an explicit selected-cell ranking rule."""

    def __init__(self, focus, actions, rank_cell, tracker):
        self.focus = focus
        self.actions = actions
        self.rank_cell = rank_cell
        self.tracker = tracker
        self.last_best_observation = None
        self.best_observation = None
        self.best_score = None
        self.best_area_px = 0
        self.best_target_points_xy = []
        self.last_frame = None
        self.last_frame_time = None
        self.sweep_sequence = 0
        self.sweep_bounds = None
        self.sweep_observations = deque()

    @property
    def searching(self):
        return self.sweep_bounds is not None

    @staticmethod
    def _contains(mask, point):
        if mask is None or mask.ndim != 2:
            return False
        x, y = np.rint(point).astype(int)
        return 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x] > 0

    def matches_cell(self, observation, target, seed, expected):
        """The phase checks calibrated XY; the sweep owns the accepted Z range."""
        low, high = self.sweep_bounds
        return low - 1e-6 <= observation['stage']['raw_position'][2] <= high + 1e-6

    def calculate(self, observation):
        self.sweep_sequence += 1
        frame = observation["image"]["frame_number"]
        captured = observation['image']['time']
        if (frame is None or frame == self.last_frame or captured is None or not math.isfinite(captured)
                or (self.last_frame_time is not None and captured <= self.last_frame_time)):
            return
        self.last_frame, self.last_frame_time = frame, captured
        cell = self.tracker.update(observation, retain=False)
        if self.tracker.lost:
            return
        score = self.rank_cell(observation)
        if score is None or not math.isfinite(score):
            return
        # Keep the selected mask for each candidate without retaining every
        # full-frame SAM output or other cells from the same image.
        cells = observation["cell"]
        index = next(i for i, candidate in enumerate(cells["cell_targets"]) if candidate is cell)
        mask = np.asarray(cell["mask"], dtype=bool)
        target = {key: value for key, value in cell.items() if key != "mask"}
        self.sweep_observations.append(dict(
            sequence=self.sweep_sequence,
            score=score, area=self.tracker.area_px, stage=observation["stage"], image=observation["image"],
            seed=cells["cell_seed_points"][index], target=target, shape=mask.shape, packed=np.packbits(mask)))

    @staticmethod
    def supported_samples(entries, pixels_to_um):
        """Return actual median-score samples supported by consecutive stable scored frames."""
        window, supported = [], []
        for entry in entries:
            image = entry['image']
            captured, frame = image['time'], image['frame_number']
            if frame is None or captured is None or not math.isfinite(captured):
                window.clear()
                continue
            if window and (entry['sequence'] != window[-1]['sequence'] + 1
                    or captured <= window[-1]['image']['time']
                    or any(frame == previous['image']['frame_number'] for previous in window)):
                window.clear()
            point = np.asarray(entry['relative_seed'], dtype=float)
            if point.shape != (2,) or not np.isfinite(point).all():
                window.clear()
                continue
            window = window[-(FocusHelper.MIN_STABLE_SAMPLES - 1):]
            if any(np.linalg.norm(pixels_to_um(np.r_[
                    point - previous['relative_seed'], 0.0])[:2]) > FocusHelper.TIP_STABILITY_UM
                    for previous in window):
                window.clear()
            window.append(entry)
            window = window[-FocusHelper.MIN_STABLE_SAMPLES:]
            if len(window) == FocusHelper.MIN_STABLE_SAMPLES:
                supported.append(sorted(window, key=lambda sample: sample['score'])[len(window) // 2])
        return supported

    @staticmethod
    def reference_anchor(calibrated, tracker):
        """Blend original calibration with the latest timestamped reliable cell anchor."""
        calibrated = np.asarray(calibrated, dtype=float)[:2]
        previous = getattr(tracker, 'last_cell_reference_xy', None)
        captured = getattr(tracker, 'last_cell_frame_time', None)
        if previous is None or captured is None or not math.isfinite(captured):
            return calibrated
        previous = np.asarray(previous, dtype=float)
        if previous.shape != (2,) or not np.isfinite(previous).all():
            return calibrated
        weight = tracker.age_weight(captured)
        if not math.isfinite(weight):
            return calibrated
        return calibrated + np.clip(weight, 0.0, 1.0) * (previous - calibrated)

    def _confirm_cell(self, phase, winner):
        """Confirm the associated cell in settled frames before promoting its Z."""
        tracker = self.tracker
        count = max(2, int(getattr(self.focus, 'MIN_CELL_SAMPLES', 2)))
        tolerance = float(getattr(self.focus, 'TIP_STABILITY_UM', 5.0))
        if getattr(phase, 'final_search_reference_xy', None) is not None:
            tolerance = min(tolerance, phase.config.final_search_xy_radius_um)
        source_stage = np.asarray(winner['stage']['pixel_position'])
        winner_cell = winner['cell']
        points = winner_cell['cell_targets'][0].get('target_points_xy', ())
        source_point = np.asarray(points[0] if len(points) else winner_cell['cell_seed_points'][0])
        if source_point.shape != (2,) or not np.isfinite(source_point).all():
            return 'not_found'
        source_z = float(winner['stage']['raw_position'][2])
        depth_tolerance = float(getattr(self.focus, 'DEPTH_TOLERANCE_UM', 2.0))
        previous = winner
        last_point = last_stage = None
        last_frame, last_time = None, None
        selected = None
        for _ in range(count):
            if not capture_checkpoint(phase):
                return None
            observation, fresh = phase.shifter._settled_cell_frame(phase, previous)
            if observation is None or not capture_checkpoint(phase):
                return None
            if not fresh:
                return 'not_found'
            settled_z = float(observation['stage']['raw_position'][2])
            if not math.isfinite(settled_z) or abs(settled_z-source_z) > depth_tolerance:
                return 'not_found'
            frame = observation['image'].get('frame_number')
            captured = observation['image'].get('time')
            if (frame is None or captured is None or not math.isfinite(captured)
                    or (last_frame is not None and (frame <= last_frame or captured <= last_time))):
                return 'not_found'
            stage = np.asarray(observation['stage']['pixel_position'])
            expected = source_point + stage - source_stage
            minimum = (getattr(tracker, 'min_area_px', None) or max(1,
                (getattr(tracker, 'original_area_px', None) or 0) * getattr(tracker, 'min_area_fraction', .01)))
            candidates = []
            cell = observation.get('cell') or {}
            for index, (seed, target) in enumerate(zip(cell.get('cell_seed_points', ()), cell.get('cell_targets', ()))):
                mask = target.get('mask')
                if mask is None or np.count_nonzero(mask) < minimum:
                    continue
                target_points = target.get('target_points_xy', ())
                point = np.asarray(min(target_points, key=lambda point: np.linalg.norm(np.asarray(point)-expected))
                                   if len(target_points) else seed, dtype=float)
                if point.shape != (2,) or not np.isfinite(point).all():
                    continue
                distance = np.linalg.norm(phase.stage.pixels_to_um_relative(np.r_[point-expected, 0.])[:2])
                if distance > tolerance:
                    continue
                if last_point is not None and np.linalg.norm(phase.stage.pixels_to_um_relative(
                        np.r_[point-(last_point+stage-last_stage), 0.])[:2]) > tolerance:
                    continue
                candidates.append((distance, index, seed, target, point))
            if not candidates:
                return 'not_found'
            selected = min(candidates, key=lambda row: row[:2])
            last_point, last_stage = selected[4], stage
            last_frame, last_time = frame, captured
            previous = observation
        try:
            image = phase.controller.observer.get_inference_image(last_frame)
            template = np.asarray(phase.cell[1])
            height, width = template.shape[:2]
            image_height, image_width = image.shape[:2]
            if height <= 0 or width <= 0 or height > image_height or width > image_width:
                return 'not_found'
            origin = np.rint(last_point - np.array([width, height]) / 2).astype(int)
            origin = np.clip(origin, [0, 0], [image_width-width, image_height-height])
            x, y = origin
            query = image[y:y+height, x:x+width]
            reference = np.asarray(phase.shifter.get_supported_alignment(
                phase.cell, query, phase.stage.pixels_to_um_relative, tolerance), dtype=float) + origin
        except Exception as error:
            phase.controller.warning(f'Search reference confirmation unavailable: {error}')
            return None if not capture_checkpoint(phase) else 'not_found'
        if not capture_checkpoint(phase):
            return None
        if (reference.shape != (2,) or not np.isfinite(reference).all()
                or not self._contains(selected[3]['mask'], reference)
                or np.linalg.norm(phase.stage.pixels_to_um_relative(np.r_[reference-last_point, 0.])[:2]) > tolerance):
            return 'not_found'
        confirmed = dict(previous)
        target = dict(selected[3])
        if not len(target.get('target_points_xy', ())):
            target['target_points_xy'] = [reference]
        confirmed['cell'] = dict(previous['cell'], cell_seed_points=[selected[2]],
                                 cell_targets=[target], cell_masks=[target['mask']])
        select_cell = getattr(tracker, 'select_cell', None)
        if callable(select_cell) and select_cell(confirmed) is not target:
            return 'not_found'
        return confirmed

    def look_for_cell(self, phase, *, bounds):
        """Nominate a raw detection across one wide pass for independent local proof."""
        candidates = []
        sequence = 0
        low, high = bounds
        expected_z = float(getattr(phase, 'cell_focus_z', (low + high) / 2))
        depth_scale = max((high - low) / 2, 1e-6)

        def capture_pose(z):
            stage = phase.controller.observer.get_stage()
            stage['raw_position'][2] = z
            return stage

        def detect_frame(frame, stage):
            nonlocal sequence
            sequence += 1
            if not capture_checkpoint(phase):
                return
            number, captured, image = frame
            cells = phase.controller.observer.get_cell(image, stage_data=stage,
                include_masks=False, select_detection=phase._select_cell)
            if not capture_checkpoint(phase):
                return
            detections = cells.get('cell_detections')
            if detections is None or len(detections) == 0:
                return
            selected = detections[0]  # get_cell already applied the phase's planar selector.
            point = np.asarray(selected[:2], dtype=float)
            reference = np.asarray(stage['pixel_position'], dtype=float) - point
            z = float(stage['raw_position'][2])
            confidence = float(selected[2]) if len(selected) > 2 else 1.0
            if (point.shape != (2,) or not np.isfinite(reference).all()
                    or not math.isfinite(z) or not math.isfinite(confidence)
                    or number is None or captured is None or not math.isfinite(captured)):
                return
            candidates.append(dict(sequence=sequence, frame=number, time=captured,
                z=z, reference=reference, confidence=float(np.clip(confidence, 0., 1.)), support=1))

        try:
            end_z = sweep_captured_focus(self.focus, self.actions, phase, bounds,
                phase.config.search_speed, capture_pose, detect_frame, 'Final cell look', frame_stride=2)
        except FocusCaptureIncomplete as error:
            phase.controller.warning(f'{error}; final detector pass found no confirmed candidate')
            return 'not_found' if capture_checkpoint(phase) else None
        if end_z is None or not capture_checkpoint(phase):
            return None
        if not candidates:
            return 'not_found'
        for previous, current in zip(candidates, candidates[1:]):
            if (current['sequence'] == previous['sequence'] + 1
                    and current['frame'] > previous['frame'] and current['time'] > previous['time']
                    and np.linalg.norm(phase.stage.pixels_to_um_relative(
                        np.r_[current['reference']-previous['reference'], 0.])[:2])
                        <= phase.config.final_search_xy_radius_um):
                previous['support'] += 1
                current['support'] += 1
        winner = max(candidates, key=lambda candidate: (candidate['support'],
            candidate['confidence'] / (1 + ((candidate['z']-expected_z) / depth_scale) ** 2),
            -abs(candidate['z']-expected_z)))
        phase.controller.info(f"Final detector pass nominated Z={winner['z']:.3f} um, "
            f"support={winner['support']}, confidence={winner['confidence']:.3f}")
        return winner['z'], winner['reference']

    def move_to(self, phase, target_z, speed):
        if not position_focus(self.focus, phase, target_z):
            return None
        observation = phase.observe()
        return observation if capture_checkpoint(phase) else None

    def run(self, phase, *, bounds, speed):
        self.best_observation = self.best_score = self.last_frame = None
        self.best_area_px = 0
        self.best_target_points_xy = []
        self.sweep_observations.clear()
        self.last_frame_time, self.sweep_sequence = None, 0
        self.sweep_bounds = bounds
        accepted = False
        previous_mask_state = (self.tracker.area_px, self.tracker.target_points_xy, self.tracker.lost)
        try:
            observer = phase.controller.observer

            def capture_pose(z):
                stage = observer.get_stage()
                stage["raw_position"][2] = z
                return stage

            def score_frame(frame, stage):
                if not capture_checkpoint(phase):
                    return
                number, timestamp, image = frame
                started = time.monotonic()
                phase.controller.info(f'Search frame={number}: detecting cells and collecting masks')
                cell = observer.get_cell(image, stage_data=stage, include_masks=True,
                                         include_interior_points=False, select_detection=phase._select_cell)
                if not capture_checkpoint(phase):
                    return
                phase.controller.info(f'Search frame={number}: collected {len(cell["cell_targets"])} targets '
                                      f'in {time.monotonic() - started:.3f} s; scoring selected cell')
                observation = dict(image=dict(frame_number=number, time=timestamp), stage=stage, cell=cell)
                self.calculate(observation)
                if not capture_checkpoint(phase):
                    return
                phase.publish_alignment_snapshot(observation)
                phase.controller.info(f'Search frame={number}: score and snapshot complete')

            end_z = sweep_captured_focus(self.focus, self.actions, phase, bounds, speed,
                                         capture_pose, score_frame, 'Search', frame_stride=2)
            if end_z is None:
                return None
            if not self.sweep_observations:
                phase.controller.warning(f'Search ended at z={end_z:.3f} um without a selected-cell mask; retaining prior depth')
                return 'not_found'
            expected_z = phase.cell_focus_z
            # Only stable tracks can define identity; prefer the recent reliable
            # anchor, fading continuously toward the original calibration.
            expected_anchor = self.reference_anchor(phase.cell[0], self.tracker)
            if getattr(phase, 'final_search_reference_xy', None) is not None:
                expected_anchor = phase.final_search_reference_xy
                expected_z = (bounds[0] + bounds[1]) / 2.0
            for entry in self.sweep_observations:
                entry['relative_seed'] = np.asarray(entry['seed']) - entry['stage']['pixel_position']
            supported = self.supported_samples(self.sweep_observations, phase.stage.pixels_to_um_relative)
            if not supported:
                phase.controller.warning('Search found no consecutive stable cell evidence; retaining the previous cell estimate')
                return 'not_found'
            reference = min(supported, key=lambda entry: (
                np.linalg.norm(phase.stage.pixels_to_um_relative(
                    np.r_[entry['relative_seed'] + expected_anchor, 0])[:2]),
                abs(entry['stage']['raw_position'][2] - expected_z)))
            box = reference['target'].get('box_xyxy')
            if box is None:
                footprint = np.unpackbits(reference['packed'], count=math.prod(reference['shape'])).reshape(reference['shape'])
                ys, xs = np.nonzero(footprint)
                box = (xs.min(), ys.min(), xs.max(), ys.max())
            low_xy, high_xy = np.asarray(box).reshape(2, 2) - reference['stage']['pixel_position']
            same_cell = [entry for entry in supported
                         if np.all(entry['relative_seed'] >= low_xy) and np.all(entry['relative_seed'] <= high_xy)]
            if not same_cell:
                phase.controller.warning('Search could not associate a detection with the selected cell; retaining the previous cell estimate')
                return 'not_found'
            selected = max(same_cell, key=lambda entry: (
                entry["score"], -abs(entry["stage"]["raw_position"][2] - expected_z)))
            mask = np.unpackbits(selected["packed"], count=math.prod(selected["shape"])).reshape(selected["shape"]).view(bool)
            target = dict(selected["target"], mask=mask)
            candidate = dict(stage=selected["stage"], image=selected["image"],
                             cell=dict(cell_seed_points=[selected["seed"]], cell_targets=[target], cell_masks=[mask]))
            if not capture_checkpoint(phase):
                return None
            winner_z = selected['stage']['raw_position'][2]
            phase.controller.info(f'Search checking candidate frame={selected["image"]["frame_number"]} at Z={winner_z:.3f} um')
            if not position_focus(self.focus, phase, winner_z):
                return None
            winner = self._confirm_cell(phase, candidate)
            if not capture_checkpoint(phase):
                return None
            if winner is None or winner == 'not_found':
                if winner == 'not_found':
                    phase.controller.warning('Search could not confirm the selected cell at the candidate depth; retaining prior depth')
                return winner
            settled_z = winner['stage']['raw_position'][2]
            target = winner['cell']['cell_targets'][0]
            self.best_observation = winner
            self.best_score = selected['score']
            self.best_area_px = int(np.count_nonzero(target['mask']))
            self.best_target_points_xy = list(target['target_points_xy'])
            phase.publish_alignment_snapshot(winner)
            frame = winner['image']
            phase.controller.info(f'Search ended at z={end_z:.3f} um, expected_z={expected_z:.3f} um, '
                                  f'candidate_frames={len(self.sweep_observations)}, winner_z={winner_z:.3f} um, '
                                  f'winner_frame={frame["frame_number"]}, frame_time_s={frame["time"]}, '
                                  f'settled_z={settled_z:.3f} um, mask_area={self.best_area_px} px, '
                                  f'target_xy={self.best_target_points_xy[:1]}')
            self.tracker.update(winner)
            self.tracker.set_reference_mask(winner, area_px=self.best_area_px)
            self.last_best_observation = deepcopy(winner)
            accepted = True
            return winner
        except FocusCaptureIncomplete as error:
            self.best_observation = self.best_score = None
            self.best_area_px, self.best_target_points_xy = 0, []
            phase.controller.warning(f'{error}; retaining prior cell depth')
            return 'not_found'
        finally:
            self.sweep_bounds = None
            if not accepted:
                self.tracker.area_px, self.tracker.target_points_xy, self.tracker.lost = previous_mask_state
                self.best_observation = self.best_score = None
                self.best_area_px, self.best_target_points_xy = 0, []
                self.sweep_observations.clear()
