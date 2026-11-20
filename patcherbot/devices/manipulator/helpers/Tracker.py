import math
import time
from copy import deepcopy
import numpy as np


class Tracker:
    def __init__(self, select_cell, *, half_life_s, min_area_px=None, min_area_fraction=0.01):
        if min_area_px is not None and (not math.isfinite(min_area_px) or min_area_px < 1):
            raise ValueError("Provide a positive minimum cell-mask area in pixels")
        if not math.isfinite(min_area_fraction) or not 0 < min_area_fraction <= 1:
            raise ValueError("Minimum mask-area fraction must be in (0, 1]")
        if not math.isfinite(half_life_s) or half_life_s <= 0:
            raise ValueError("Provide a positive visual half-life in seconds")
        self.half_life_s = half_life_s
        self.select_cell = select_cell
        self.min_area_px = min_area_px
        self.min_area_fraction = min_area_fraction
        self.reset()

    def reset(self):
        """Forget the previous attempt/cell, including fallback anchors."""
        self.area_px = 0
        self.original_area_px = None
        self.target_points_xy = []
        self.lost = True
        self.using_fallback = False
        self.last_cell_reference_xy = None
        self.last_cell_frame_time = None
        self.last_frame = self.last_frame_time = None
        self.mask_reference = None
        self.largest_area_px = 0

    def copy_state_from(self, other):
        """Carry reliable tracking history into a new phase without its callbacks or policy."""
        for name in ('area_px', 'original_area_px', 'target_points_xy', 'lost',
                     'using_fallback', 'last_cell_reference_xy', 'last_cell_frame_time',
                     'last_frame', 'last_frame_time', 'mask_reference', 'largest_area_px'):
            setattr(self, name, deepcopy(getattr(other, name)))

    def set_reference_mask(self, observation, *, area_px=None):
        """Bind the first completed Search winner's area for this attempt."""
        if self.original_area_px is not None:
            return
        if area_px is None:
            cell = None if observation["cell"] is None else self.select_cell(observation)
            mask = None if cell is None else cell["mask"]
            area_px = 0 if mask is None else int(np.count_nonzero(np.asarray(mask) > 0))
        if area_px > 0:
            self.original_area_px = int(area_px)

    def age_weight(self, frame_time, *, now=None):
        if frame_time is None or not math.isfinite(frame_time):
            return 0.0
        now = time.time() if now is None else now
        age = max(0.0, now - frame_time)
        return 0.5 ** (age / self.half_life_s)

    def remember(self, observation, point_xy):
        self.last_cell_reference_xy = (np.asarray(observation["stage"]["pixel_position"])
            - np.asarray(point_xy))
        self.last_cell_frame_time = observation["image"]["time"]

    def update(self, observation, *, retain=True):
        cell = None if observation["cell"] is None else self.select_cell(observation)
        mask = None if cell is None else cell["mask"]
        self.area_px = 0 if mask is None else int(np.count_nonzero(np.asarray(mask) > 0))
        self.target_points_xy = [] if cell is None else list(cell["target_points_xy"])
        minimum = (self.min_area_px if self.min_area_px is not None else
                   max(1, (self.original_area_px or 0) * self.min_area_fraction))
        self.lost = self.area_px < minimum
        if not retain:
            return cell
        frame, captured = observation["image"]["frame_number"], observation["image"]["time"]
        if (frame is None or captured is None or not math.isfinite(captured)
                or frame == self.last_frame
                or (self.last_frame_time is not None and captured <= self.last_frame_time)):
            return  # Cached/older frames cannot re-arm tracking or replace anchors.
        self.last_frame, self.last_frame_time = frame, captured
        if not self.lost:
            if self.target_points_xy and self.area_px >= self.largest_area_px:
                self.mask_reference = (np.asarray(mask, dtype=bool).copy(),
                    np.asarray(observation["stage"]["pixel_position"]).copy(),
                    np.asarray(self.target_points_xy[0], dtype=float), captured)
                self.largest_area_px = self.area_px
            self.using_fallback = False
            if self.target_points_xy:
                self.remember(observation, self.target_points_xy[0])

    def position(self, observation, calibrated_cell_position):
        """Project the last reliable cell point, or original calibration, into this frame."""
        anchor = (np.asarray(calibrated_cell_position)[:2] if self.last_cell_reference_xy is None
                  else self.last_cell_reference_xy)
        return np.asarray(observation["stage"]["pixel_position"]) - anchor

    def distance_to_mask(self, tip_xy, observation, calibrated_cell_position,
                         pixels_to_um, max_offset_px):
        """Measure distance to the target mask, or to the calibrated cell when no mask exists."""
        stage = np.asarray(observation["stage"]["pixel_position"])
        calibrated = stage - np.asarray(calibrated_cell_position)[:2]
        tip = np.asarray(tip_xy, dtype=float)
        if tip.shape != (2,) or not np.isfinite(tip).all() or not np.isfinite(calibrated).all():
            return float("inf")
        if self.mask_reference is None:
            return float(np.linalg.norm(pixels_to_um(np.r_[tip - calibrated, 0.0])[:2]))
        mask, source_stage, point, captured = self.mask_reference
        offset = np.clip(point + stage - source_stage - calibrated,
                         -max_offset_px, max_offset_px)
        shift = calibrated + self.age_weight(captured) * offset - point
        x, y = np.rint(tip - shift).astype(int)
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
            return 0.0
        points = np.argwhere(mask)[:, ::-1] + shift
        basis = np.column_stack((pixels_to_um([1, 0, 0])[:2], pixels_to_um([0, 1, 0])[:2]))
        return float(np.linalg.norm((points - tip) @ basis.T, axis=1).min())

