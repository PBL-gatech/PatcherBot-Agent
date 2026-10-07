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

    def __init__(self, controller, kind):
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
            confidence = vision.get('confidence', {}).get('pipette_focuser')
            if confidence is not None:
                try:
                    confidence = float(confidence)
                except (TypeError, ValueError):
                    confidence = math.nan
                if not math.isfinite(confidence) or confidence < 0.5:
                    self.depth_warnings.add('low or invalid Z confidence')
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
        if self.confirmed:
            if not self.is_cell:
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
