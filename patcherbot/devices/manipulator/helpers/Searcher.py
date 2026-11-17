import math
from collections import deque
from copy import deepcopy

import numpy as np

from .Scanner import FocusCaptureIncomplete, position_focus, sweep_captured_focus

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
        frame = observation["image"]["frame_number"]
        if frame is None or frame == self.last_frame:
            return
        self.last_frame = frame
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
            score=score, area=self.tracker.area_px, stage=observation["stage"], image=observation["image"],
            seed=cells["cell_seed_points"][index], target=target, shape=mask.shape, packed=np.packbits(mask)))

    def move_to(self, phase, target_z, speed):
        if not position_focus(self.focus, phase, target_z):
            return None
        observation = phase.observe()
        return observation if phase.motion_checkpoint() else None

    def run(self, phase, *, bounds, speed):
        self.best_observation = self.best_score = self.last_frame = None
        self.best_area_px = 0
        self.best_target_points_xy = []
        self.sweep_observations.clear()
        self.sweep_bounds = bounds
        try:
            observer = phase.controller.observer

            def capture_pose(z):
                stage = observer.get_stage()
                stage["raw_position"][2] = z
                return stage

            def score_frame(frame, stage):
                number, timestamp, image = frame
                cell = observer.get_cell(image, stage_data=stage, include_masks=True)
                observation = dict(image=dict(frame_number=number, time=timestamp), stage=stage, cell=cell)
                self.calculate(observation)
                phase.publish_alignment_snapshot(observation)

            end_z = sweep_captured_focus(self.focus, self.actions, phase, bounds, speed,
                                         capture_pose, score_frame, 'Search')
            if end_z is None:
                return None
            if not self.sweep_observations:
                phase.controller.warning(f'Search ended at z={end_z:.3f} um with no in-range mask matching the selected cell; using calibrated XY and Z')
                return 'not_found'
            expected_z = phase.cell_focus_z
            # Identify the cell across the sweep before comparing its areas. Use the
            # detector footprint of the observation nearest the calibrated selection.
            for entry in self.sweep_observations:
                entry['relative_seed'] = np.asarray(entry['seed']) - entry['stage']['pixel_position']
            reference = min(self.sweep_observations, key=lambda entry: (
                np.linalg.norm(phase.stage.pixels_to_um_relative(
                    np.r_[entry['relative_seed'] + np.asarray(phase.cell[0])[:2], 0])[:2]),
                abs(entry['stage']['raw_position'][2] - expected_z)))
            box = reference['target'].get('box_xyxy')
            if box is None:
                footprint = np.unpackbits(reference['packed'], count=math.prod(reference['shape'])).reshape(reference['shape'])
                ys, xs = np.nonzero(footprint)
                box = (xs.min(), ys.min(), xs.max(), ys.max())
            low_xy, high_xy = np.asarray(box).reshape(2, 2) - reference['stage']['pixel_position']
            same_cell = [entry for entry in self.sweep_observations
                         if np.all(entry['relative_seed'] >= low_xy) and np.all(entry['relative_seed'] <= high_xy)]
            if not same_cell:
                phase.controller.warning('Search could not associate a detection with the selected cell; using calibration')
                return 'not_found'
            selected = max(same_cell, key=lambda entry: (
                entry["score"], -abs(entry["stage"]["raw_position"][2] - expected_z)))
            mask = np.unpackbits(selected["packed"], count=math.prod(selected["shape"])).reshape(selected["shape"]).view(bool)
            target = dict(selected["target"], mask=mask)
            winner = dict(stage=selected["stage"], image=selected["image"],
                          cell=dict(cell_seed_points=[selected["seed"]], cell_targets=[target], cell_masks=[mask]))
            self.best_observation = winner
            self.best_score = selected["score"]
            self.best_area_px = selected["area"]
            self.best_target_points_xy = list(target["target_points_xy"])
            winner_z = winner['stage']['raw_position'][2]
            restored = self.move_to(phase, winner_z, speed)
            if restored is None:
                return None
            settled_z = restored['stage']['raw_position'][2]
            frame = winner['image']
            phase.controller.info(f'Search ended at z={end_z:.3f} um, expected_z={expected_z:.3f} um, '
                                  f'eligible_frames={len(same_cell)}, winner_z={winner_z:.3f} um, '
                                  f'winner_frame={frame["frame_number"]}, frame_time_s={frame["time"]}, '
                                  f'settled_z={settled_z:.3f} um, mask_area={self.best_area_px} px, '
                                  f'target_xy={self.best_target_points_xy[:1]}')
            self.tracker.update(winner)
            self.tracker.set_reference_mask(winner, area_px=self.best_area_px)
            self.last_best_observation = deepcopy(winner)
            return winner
        except FocusCaptureIncomplete as error:
            self.best_observation = self.best_score = None
            self.best_area_px, self.best_target_points_xy = 0, []
            phase.controller.warning(f'{error}; using calibrated cell position')
            return 'not_found'
        finally:
            self.sweep_bounds = None
