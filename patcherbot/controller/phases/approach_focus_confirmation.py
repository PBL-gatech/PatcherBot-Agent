"""Confirm a returned focus plane using fresh stationary camera frames."""
import math
from statistics import median
import time

from ..errors import AutopatchError


class FocusConfirmation:
    TIMEOUT_S = 5.0
    MAX_FRAMES = 8
    DEPTH_TOLERANCE_UM = 2.0
    MIN_PIPETTE_SAMPLES = 3
    MIN_CELL_SAMPLES = 2
    TIP_STABILITY_UM = 5.0

    def __init__(self, controller, kind, selected_tip_xy=None, selected_cell_xy=None):
        self.controller = controller
        self.kind = kind
        self.is_cell = str(kind).lower() in ('cell', 'search')
        self.started_at = time.monotonic()
        self.frames = 0
        self.valid_samples = 0
        self.depths = []
        self.depth_warnings = set()
        self.seen_frames = set()
        self.last_vision = {}
        self.confirmed = False
        self.tip_samples = []
        self.confirmed_tip_xy = None
        self.confirmed_tip_confidence = None
        self.selected_tip_xy = selected_tip_xy
        self.cell_samples = []
        self.selected_cell_xy = selected_cell_xy

    def _fail(self, reason):
        vision = self.last_vision
        depth = median(self.depths) if self.depths else None
        raise AutopatchError(
            f'{self.kind} focus confirmation failed: {reason}; '
            f'fresh_frames={self.frames}; valid_samples={self.valid_samples}; '
            f'median_abs_depth_um={depth}; last_depth_um={vision.get("pipette_focus")}; '
            f'confidence={vision.get("confidence")}; model_status={vision.get("status")}')

    def update(self, observation, valid):
        if self.confirmed:
            return True
        if time.monotonic() - self.started_at >= self.TIMEOUT_S:
            self._fail('timed out waiting for settled camera evidence')
        vision = observation.get('deep_learning', {})
        frame = vision.get('source_frame')
        retrieved = vision.get('frame_retrieval_started_at')
        if (vision.get('stale', True) or frame is None or frame in self.seen_frames
                or retrieved is None or not math.isfinite(retrieved) or retrieved < self.started_at):
            return None
        self.seen_frames.add(frame)
        self.frames += 1
        self.last_vision = vision
        sample_valid = bool(valid)
        expected = None
        if self.is_cell:
            cell = observation.get('cell_detection')
            if sample_valid:
                sample_valid = (cell is not None and len(cell) >= 3
                    and all(math.isfinite(float(value)) for value in cell[:3]))
            if sample_valid:
                xy = tuple(float(value) for value in cell[:2])
                if self.selected_cell_xy is not None:
                    delta = self.controller.calibrated_stage.pixels_to_um_relative(
                        [xy[0] - self.selected_cell_xy[0], xy[1] - self.selected_cell_xy[1]])
                    sample_valid = math.hypot(float(delta[0]), float(delta[1])) <= self.TIP_STABILITY_UM
                    if not sample_valid:
                        self.controller.info(f'{self.kind} FOCUS CONFIRM: cell differs from SEARCH target; '
                            f'selected_xy={self.selected_cell_xy}; current_xy={xy}; reacquiring stable evidence')
            if sample_valid:
                for previous in self.cell_samples:
                    delta = self.controller.calibrated_stage.pixels_to_um_relative(
                        [xy[0] - previous[0], xy[1] - previous[1]])
                    if math.hypot(float(delta[0]), float(delta[1])) > self.TIP_STABILITY_UM:
                        self.cell_samples.clear()
                        self.valid_samples = 0
                        self.controller.info(f'{self.kind} FOCUS CONFIRM: cell XY jumped; restarting stable evidence')
                        break
                self.cell_samples.append(xy)
            else:
                self.cell_samples.clear()
                self.valid_samples = 0
        if not self.is_cell:
            tip = vision.get('pipette_position')
            if sample_valid:
                sample_valid = (tip is not None and len(tip) >= 2
                    and all(math.isfinite(float(value)) for value in tip[:2]))
            if sample_valid:
                tip = tuple(float(value) for value in tip[:2])
                if self.selected_tip_xy is not None:
                    try:
                        expected = tuple(float(value) for value in self.selected_tip_xy)
                    except (TypeError, ValueError):
                        expected = ()
                    sample_valid = (len(expected) == 2
                        and all(math.isfinite(value) for value in expected))
                    distance = math.inf
                    if sample_valid:
                        delta = self.controller.calibrated_stage.pixels_to_um_relative(
                            [tip[0] - expected[0], tip[1] - expected[1]])
                        distance = math.hypot(float(delta[0]), float(delta[1]))
                        sample_valid = math.isfinite(distance) and distance <= self.TIP_STABILITY_UM
                    if not sample_valid:
                        self.controller.info(f'{self.kind} FOCUS CONFIRM: XY rejected; tip={tip}; '
                            f'scan_selected_tip={expected}; selected_tip_error_um={distance}; '
                            f'tolerance_um={self.TIP_STABILITY_UM}')
            if sample_valid:
                for previous, _ in self.tip_samples:
                    delta = self.controller.calibrated_stage.pixels_to_um_relative(
                        [tip[0] - previous[0], tip[1] - previous[1]])
                    if math.hypot(float(delta[0]), float(delta[1])) > self.TIP_STABILITY_UM:
                        self.tip_samples.clear()
                        self.valid_samples = 0
                        self.controller.info(f'{self.kind} FOCUS CONFIRM: XY jumped; restarting stable evidence')
                        break
                self.tip_samples.append((tip, vision.get('confidence', {}).get('pipette_detector')))
            else:
                self.tip_samples.clear()
                self.valid_samples = 0
        if not self.is_cell and sample_valid:
            try:
                depth = float(vision.get('pipette_focus'))
            except (TypeError, ValueError):
                depth = math.nan
            if math.isfinite(depth):
                self.depths.append(abs(depth))
            else:
                self.depth_warnings.add('missing or non-finite Z estimate')
            status = vision.get('status', {}).get('pipette_focuser')
            if status != 'ok':
                self.depth_warnings.add(f'Z model status={status}')
        if sample_valid:
            self.valid_samples += 1
        aggregate = median(self.depths) if self.depths else None
        self.controller.info(
            f'{self.kind} FOCUS CONFIRM: frame={frame}; fresh_frames={self.frames}; '
            f'valid={sample_valid}; valid_samples={self.valid_samples}; '
            f'depth_um={vision.get("pipette_focus")}; median_abs_depth_um={aggregate}; '
            f'confidence={vision.get("confidence")}')
        self.confirmed = (self.valid_samples >= self.MIN_CELL_SAMPLES if self.is_cell else
                          self.valid_samples >= self.MIN_PIPETTE_SAMPLES)
        if self.confirmed and not self.is_cell and expected is not None:
            candidate = tuple(median(tip[axis] for tip, _ in self.tip_samples) for axis in (0, 1))
            delta = self.controller.calibrated_stage.pixels_to_um_relative(
                [candidate[0] - expected[0], candidate[1] - expected[1]])
            distance = math.hypot(float(delta[0]), float(delta[1]))
            if not math.isfinite(distance) or distance > self.TIP_STABILITY_UM:
                self.confirmed = False
                self.valid_samples = 0
                self.tip_samples.clear()
                self.controller.info(f'{self.kind} FOCUS CONFIRM: median XY rejected; '
                    f'selected_tip_error_um={distance}; tolerance_um={self.TIP_STABILITY_UM}')
        if self.confirmed:
            if not self.is_cell:
                self.confirmed_tip_xy = tuple(median(tip[axis] for tip, _ in self.tip_samples)
                                              for axis in (0, 1))
                confidences = [float(confidence) for _, confidence in self.tip_samples
                               if confidence is not None and math.isfinite(float(confidence))]
                self.confirmed_tip_confidence = median(confidences) if confidences else None
                if aggregate is not None and aggregate > self.DEPTH_TOLERANCE_UM:
                    self.depth_warnings.add('median Z estimate exceeds 2 um')
                if self.depth_warnings:
                    self.controller.warning(
                        f'{self.kind} FOCUS CONFIRM: reliable pipette XY accepted; '
                        f'Z focus remains advisory: {"; ".join(sorted(self.depth_warnings))}; '
                        f'median_abs_depth_um={aggregate}; confidence={vision.get("confidence")}')
            self.controller.info(f'{self.kind} FOCUS CONFIRM: accepted settled evidence; '
                                 f'valid_samples={self.valid_samples}; median_abs_depth_um={aggregate}')
            return True
        if self.frames >= self.MAX_FRAMES:
            self._fail('no reliable focus confirmation within the fresh-frame limit')
        return None
