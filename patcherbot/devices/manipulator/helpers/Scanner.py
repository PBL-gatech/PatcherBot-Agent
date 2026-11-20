import math
import time
from collections import deque
from contextlib import ExitStack
from copy import deepcopy
from inspect import Parameter, signature
from tempfile import TemporaryFile

import numpy as np

from .StageCalHelper import FocusHelper


class FocusCaptureIncomplete(RuntimeError):
    """Some frames could not be captured, so the scan result is incomplete."""


def capture_checkpoint(phase):
    """Reject queued inference and results as soon as the phase stops."""
    if any(getattr(phase, name, False) for name in
           ('finished', 'failed', 'awaiting_operator', 'interrupted')):
        return False
    return phase.motion_checkpoint()


def position_focus(focus, phase, target_z):
    """Position at locate speed without running inference."""
    if not math.isfinite(target_z):
        raise ValueError('Focus position must be finite')
    if not capture_checkpoint(phase):
        return False
    if not phase.action_gate():
        if phase.interrupted:
            return False
        raise RuntimeError('Focus positioning is not permitted')
    focus.microscope.set_max_speed(phase.config.max_locate_speed)
    focus.relocate(float(target_z))
    return capture_checkpoint(phase)


def _focus_position_at(samples, timestamp):
    """Interpolate the encoder samples bracketing a camera retrieval."""
    while len(samples) > 2 and samples[1][0] <= timestamp:
        samples.popleft()
    first_time, first_z = samples[0]
    if not first_time <= timestamp <= samples[-1][0]:
        return None
    if timestamp == first_time:
        return first_z
    next_time, next_z = samples[1]
    return first_z + (timestamp - first_time) / (next_time - first_time) * (next_z - first_z)


def sweep_captured_focus(focus, motion_device, phase, bounds, speed, snapshot, calculate, label,
                         *, frame_stride=1):
    """Collect distinct published frames during motion and score them in order afterward."""
    low, high = bounds
    if not isinstance(frame_stride, int) or frame_stride < 1:
        raise ValueError('Frame stride must be a positive integer')
    if not all(math.isfinite(value) for value in (low, high, speed)) or low > high or speed <= 0:
        raise ValueError(f'{label} needs finite ordered bounds and positive speed')
    scale = focus._distance_scale()
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f'{label} needs a positive microscope position scale')
    if not position_focus(focus, phase, low):
        return None

    def read_position():
        return float(focus.microscope.position()) / scale

    encoder_history = deque(maxlen=512)
    frames = deque()
    seen_ids = set()
    previous_id = None
    missed_frames = unpaired_frames = 0
    spill = None

    def collect_frame(end_at=None):
        nonlocal previous_id, missed_frames, unpaired_frames, spill
        latest = focus.camera.last_raw_frame_data(include_timing=True)
        encoder_history.append((time.monotonic(), read_position()))
        if latest is None:
            return False
        frame_number, frame_time, image, timing = latest
        frame_at = timing.get('acquired_at')
        if frame_at is None:
            frame_at = timing.get('retrieval_started_at')
        if frame_at is None or not math.isfinite(frame_at):
            if frame_number is not None and frame_number not in seen_ids:
                seen_ids.add(frame_number)
                unpaired_frames += 1
            return False
        if end_at is not None and frame_at > end_at:
            return True  # The latest frame belongs to the period after the sweep.
        if frame_at < started_at or frame_number is None or frame_number in seen_ids:
            return False
        seen_ids.add(frame_number)
        if previous_id is not None:
            missed_frames += max(0, frame_number - previous_id - 1)
        previous_id = frame_number
        z = _focus_position_at(encoder_history, frame_at)
        if z is None or not math.isfinite(z) or frame_time is None or image is None:
            unpaired_frames += 1
        elif low - 1e-6 <= z <= high + 1e-6:
            pose = snapshot(z)
            offset = None
            if len(frames) >= 64:
                try:
                    if spill is None:
                        spill = resources.enter_context(TemporaryFile(mode='w+b', buffering=0))
                    offset = spill.tell()
                    np.save(spill, image, allow_pickle=False)
                except (OSError, TypeError, ValueError) as error:
                    raise FocusCaptureIncomplete(f'{label} could not retain captured frames: {error}') from error
                image = None
            frames.append(((frame_number, frame_time.timestamp(), image), pose, offset))
        return False

    phase.controller.info(f'{label} sweep: bounds_z=({low:.3f}, {high:.3f}) um, '
                          f'settled_start_z={read_position():.3f} um, speed={speed:g} um/s')
    with ExitStack() as resources:
        started_at = time.monotonic()
        encoder_history.append((started_at, read_position()))
        motion = motion_device.start(focus.microscope, target=high,
            deadline=time.monotonic() + max(10.0, (high - low) / speed * 20.0),
            read_position=read_position,
            move_velocity=lambda direction: focus.microscope.absolute_move_velocity(-direction * speed),
            abort_check=phase.controller.abort_if_requested, info=phase.controller.warning,
            label=f'{label} focus sweep', direction=1.0,
            reversal_tolerance=FocusHelper.BOUNDARY_TOLERANCE_UM, reversal_slack=1e-6)
        try:
            while not motion_device.is_complete(motion):
                if not capture_checkpoint(phase):
                    return None
                if not phase.action_gate():
                    if phase.interrupted:
                        return None
                    raise RuntimeError(f'{label} focus sweep is not permitted')
                collect_frame()
                phase.controller.sleep(0.002)
            ended_at = time.monotonic()
        finally:
            motion_device.cancel(motion)
        end_z = read_position()
        if end_z > high + FocusHelper.BOUNDARY_TOLERANCE_UM + 1e-6:
            phase.controller.warning(f'{label} sweep stopped at z={end_z:.3f} um, '
                                     f'{end_z - high:.3f} um beyond the upper edge')

        # Publication can trail retrieval by a frame interval at the endpoint.
        deadline = ended_at + 0.25
        while capture_checkpoint(phase) and not collect_frame(ended_at):
            if time.monotonic() >= deadline:
                break
            phase.controller.sleep(0.002)

        if missed_frames or unpaired_frames:
            phase.controller.warning(f'{label} sweep observed {missed_frames} frame ID gaps and '
                                     f'{unpaired_frames} frames without matching timing; '
                                     f'{len(frames)} timed in-range frames available')
        if not capture_checkpoint(phase):
            return None
        captured_count = len(frames)
        if frame_stride > 1:
            frames = deque(entry for index, entry in enumerate(frames)
                           if index % frame_stride == 0 or index == captured_count - 1)
            phase.controller.info(f'{label} selecting {len(frames)} of {captured_count} captured frames '
                                  f'with stride={frame_stride}; retaining both endpoints')
        frame_count = len(frames)
        phase.controller.info(f'{label} sweep score analysis started ({frame_count} frames)')
        scoring_started = time.monotonic()
        while frames:
            if not capture_checkpoint(phase):
                return None
            frame, pose, offset = frames.popleft()
            phase.controller.info(f'{label} scoring {frame_count - len(frames)}/{frame_count}: frame={frame[0]}')
            if offset is not None:
                try:
                    spill.seek(offset)
                    frame = (*frame[:2], np.load(spill, allow_pickle=False))
                except (OSError, EOFError, ValueError) as error:
                    raise FocusCaptureIncomplete(f'{label} could not read captured frames: {error}') from error
            calculate(frame, pose)
            if not capture_checkpoint(phase):
                return None
        if not frames:
            phase.controller.info(f'{label} sweep score analysis finished ({time.monotonic() - scoring_started:.2f} s)')
        return end_z if capture_checkpoint(phase) else None


class Scanner:
    """Sweep continuously; retain the best existing pipette observation."""

    def __init__(self, focus, actions):
        self.focus = focus
        self.actions = actions
        self.last_best_observation = None
        self.clear_current_scan()

    def clear_current_scan(self):
        self.best_observation = None
        self.best_focus_weight = 0.0
        self.focus_score = None
        self.focus_weight = 0.0
        self.buckets = [[] for _ in range(5)]
        self.last_frame = None
        self.sweep_observations = deque()

    @staticmethod
    def weight_for_score(score):
        t = min(1.0, max(0.0, score / 0.8))
        return t * t * (3.0 - 2.0 * t)

    @staticmethod
    def tip_matches_anchor(tip, expected, stage, radius_um):
        tip, expected = np.asarray(tip, dtype=float), np.asarray(expected, dtype=float)
        bounds = [stage.camera.width, stage.camera.height]
        for point in (tip, expected):
            if (point.shape != (2,) or not np.isfinite(point).all()
                    or np.any(point < 0) or np.any(point >= bounds)):
                return False
        displacement = stage.pixels_to_um_relative(np.r_[tip - expected, 0.0])
        return np.linalg.norm(displacement[:2]) <= radius_um

    def calculate(self, observation, *, stage, radius_um):
        self.focus_score, self.focus_weight = None, 0.0
        frame = observation["image"]["frame_number"]
        if frame is None or frame == self.last_frame:
            return
        self.last_frame = frame
        observation['scan_eligible'] = False
        self.sweep_observations.append(observation)
        pipette = observation["pipette"]
        detection = pipette["pipette_detection"]
        if detection is None:
            return
        score, tip = detection.get("focus_score"), detection.get("tip_xy")
        z = observation["stage"]["raw_position"][2]
        if (score is None or tip is None or not math.isfinite(score)
                or not 0.0 <= score <= 1.0 or not math.isfinite(z)
                or not self.tip_matches_anchor(tip, pipette['pixel_position'], stage, radius_um)):
            return
        self.focus_score = score
        self.focus_weight = self.weight_for_score(score)
        observation['scan_eligible'] = True
        bucket = min(int(score * len(self.buckets)), len(self.buckets) - 1)
        self.buckets[bucket].append({"observation": observation, "weight": self.focus_weight})

    def move_to(self, phase, target_z, speed):
        if not position_focus(self.focus, phase, target_z):
            return None
        z = self.focus.microscope.position() / self.focus._distance_scale()
        return {'stage': {'raw_position': np.r_[phase.controller.calibrated_stage.position()[:2], z]}}

    def run(self, phase, *, bounds, speed):
        self.clear_current_scan()
        stage = phase.controller.calibrated_stage
        unit = phase.controller.calibrated_unit
        radius_um = unit.config.pipette_detection_radius_um
        detector = unit.pipetteCalHelper.pipetteDetector
        detect = detector.detect_pipette_details
        try:
            parameters = signature(detect).parameters
            guided = (any(parameter.kind == Parameter.VAR_KEYWORD for parameter in parameters.values())
                      or all(name in parameters and parameters[name].kind != Parameter.POSITIONAL_ONLY
                             for name in ('expected_tip_xy', 'pixels_to_um', 'radius_um', 'prefer_anchor')))
        except (TypeError, ValueError):
            guided = False
        observer = phase.controller.observer
        expected_z = float(phase.pipette_z)

        def snapshot(z):
            stage_data = observer.get_stage()
            stage_data['raw_position'][2] = z
            pipette = observer.get_pipette(detect=False, pixel_space=False)
            position = np.array(pipette['raw_position'], copy=True)
            pipette['raw_position'] = position
            pipette['pixel_position'] = phase.project_pipette_tip(phase.pipette_anchor, position)
            return stage_data, pipette

        def consider(frame, pose):
            if not capture_checkpoint(phase):
                return
            number, frame_time, image = frame
            stage_data, pipette = pose
            options = (dict(expected_tip_xy=pipette['pixel_position'],
                            pixels_to_um=stage.pixels_to_um_relative,
                            radius_um=radius_um, prefer_anchor=True) if guided else {})
            detection = detect(image.copy(), **options)
            if not capture_checkpoint(phase):
                return
            pipette['pipette_detection'] = detection
            timestamp = round(frame_time, 3)
            observation = dict(time=timestamp,
                image=dict(frame_number=number, time=timestamp,
                           width=image.shape[1], height=image.shape[0]),
                stage=stage_data, pipette=pipette)
            self.calculate(observation, stage=stage, radius_um=radius_um)
            if not capture_checkpoint(phase):
                return
            phase.publish_alignment_snapshot(observation)
            detection = detection or {}
            z = stage_data['raw_position'][2]
            phase.controller.info(f'Scan sample: frame={number}, frame_time_s={timestamp}, '
                                  f'retrieval_z={z:.3f} um, focus_score={detection.get("focus_score")}, '
                                  f'tip_xy={detection.get("tip_xy")}, eligible={self.focus_score is not None}')

        try:
            end_z = sweep_captured_focus(self.focus, self.actions, phase, bounds, speed,
                                         snapshot, consider, 'Scan',
                                         frame_stride=max(1, int(getattr(phase.config, 'scan_frame_stride', 2))))
        except FocusCaptureIncomplete as error:
            self.clear_current_scan()
            phase.controller.warning(f'{error}; using calibrated fallback')
            return 'not_found'
        except Exception:
            self.clear_current_scan()
            raise
        if end_z is None:
            self.clear_current_scan()
            return None

        def rank(observation):
            score = observation['pipette']['pipette_detection']['focus_score']
            z = observation['stage']['raw_position'][2]
            return score, -abs(z - expected_z)

        eligible = [observation for observation in self.sweep_observations if observation['scan_eligible']]
        winner = max(eligible, key=rank, default=None)
        if winner is None:
            phase.controller.warning(f'Scan ended at z={end_z:.3f} um with no eligible tip sample '
                                     f'(evaluated={len(self.sweep_observations)}, expected_z={expected_z:.3f} um, '
                                     f'radius={radius_um:g} um); '
                                     'using calibrated fallback')
            return 'not_found'
        self.best_observation = winner
        self.best_focus_weight = self.weight_for_score(rank(winner)[0])
        winner_z = winner['stage']['raw_position'][2]
        restored = self.move_to(phase, winner_z, speed)
        if restored is None:
            self.clear_current_scan()
            return None
        settled_z = restored['stage']['raw_position'][2]
        detection = winner['pipette']['pipette_detection']
        frame = winner['image']
        if not capture_checkpoint(phase):
            self.clear_current_scan()
            return None
        phase.publish_alignment_snapshot(winner)
        phase.controller.info(f'Scan ended at z={end_z:.3f} um, winner_z={winner_z:.3f} um, '
                              f'winner_frame={frame["frame_number"]}, frame_time_s={frame["time"]}, '
                              f'expected_z={expected_z:.3f} um, settled_z={settled_z:.3f} um, '
                              f'evaluated={len(self.sweep_observations)}, eligible={len(eligible)}, '
                              f'focus_score={detection["focus_score"]:.3f}, '
                              f'tip_xy={detection["tip_xy"]}')
        self.last_best_observation = deepcopy(winner)
        return winner
