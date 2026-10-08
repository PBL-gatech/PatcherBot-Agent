"""Finite objective focus sweeps; the calling phase continues frame inference."""
import math
import threading
import time

from ..errors import AutopatchError


class EmptyFocusEvidence(AutopatchError):
    """A completed sweep had no accepted detection sequence, without a motion fault."""


class FocusSweep:
    RADIUS_UM = 5.0
    ENDPOINT_TOLERANCE_UM = 0.2
    BOUNDARY_TOLERANCE_UM = 1.0
    TIP_STABILITY_UM = 5.0
    MIN_STABLE_SAMPLES = 3
    DEPTH_TOLERANCE_UM = 2.0

    def __init__(self, controller, center_z, kind):
        self.controller = controller
        self.center_z = float(center_z)
        self.kind = kind
        self.low = self.center_z - self.RADIUS_UM
        self.high = self.center_z + self.RADIUS_UM
        self.stage = 'idle'
        self.target = None
        self.best_z = None
        self.best_score = -math.inf
        self.samples = 0
        self.last_frame = None
        self.seen_frames = set()
        self.tip_samples = []
        self.best_tip_xy = None
        self.best_tip_confidence = None
        self.best_tip_defocus_um = None
        self.best_tip_rank = None
        self.best_cell_xy = None
        self.best_cell_confidence = None
        self.deadline = None
        self.speed = None
        self._motion_thread = None
        self._motion_stop = threading.Event()
        self._motion_done = threading.Event()
        self._motion_error = None
        self._endpoint_correction = False

    def start(self):
        self.controller.abort_if_requested()
        speed = float(self.controller.config.search_speed)
        if not math.isfinite(self.center_z) or not math.isfinite(speed) or speed <= 0:
            raise AutopatchError('Focus sweep needs a finite center and positive search speed')
        self.speed = speed
        self.deadline = time.monotonic() + max(60.0, 4.0 * 20.0 / speed * 20.0)
        self.controller.microscope.stop()
        self.controller.info(f'{self.kind} FOCUS SWEEP: bounds=({self.low}, {self.high}); '
                             f'boundary_tolerance_um={self.BOUNDARY_TOLERANCE_UM}; '
                             f'velocity_command={speed}; inference continues during monitored motion')
        self._move('seek_start', self.low)

    def _move(self, stage, target):
        self.stage, self.target = stage, float(target)
        self._endpoint_correction = False
        self.controller.info(f'{self.kind} FOCUS SWEEP: {stage}; target_z={self.target}')
        if stage in ('down', 'up'):
            self._motion_stop.clear()
            self._motion_done.clear()
            self._motion_error = None
            self._motion_thread = threading.Thread(target=self._run_leg, args=(self.target,),
                                                  name='ApproachFocusSweep', daemon=True)
            self._motion_thread.start()
        else:
            self.controller.microscope.absolute_move(self.target)

    def _position(self):
        scale = float(self.controller.calibrated_unit.config.microscope_units_per_um)
        if not math.isfinite(scale) or scale <= 0:
            raise AutopatchError('Focus sweep has an invalid microscope position scale')
        z = float(self.controller.microscope.position()) / scale
        if not math.isfinite(z):
            raise AutopatchError('Focus sweep has no valid objective position')
        return z

    def _run_leg(self, target):
        """Monitor the endpoint independently of potentially slow model inference."""
        try:
            start = self._position()
            direction = 1.0 if target > start else -1.0
            if abs(target - start) <= self.ENDPOINT_TOLERANCE_UM:
                return
            self.controller.abort_if_requested()
            if self._motion_stop.is_set():
                return
            # On this rig VJ Z has the opposite sign to measured position Z.
            self.controller.microscope.absolute_move_velocity(-direction * self.speed)
            drift_logged = False
            while not self._motion_stop.is_set():
                self.controller.abort_if_requested()
                if time.monotonic() > self.deadline:
                    raise AutopatchError(f'{self.kind} focus sweep motion timed out')
                z = self._position()
                if direction * (target - z) <= self.ENDPOINT_TOLERANCE_UM:
                    break
                away = -direction * (z - start)
                if away > self.BOUNDARY_TOLERANCE_UM + 1e-6:
                    raise AutopatchError(f'{self.kind} focus sweep moved in the wrong direction; '
                        f'start_z={start}; actual_z={z}; target_z={target}; '
                        f'away_um={away}; tolerance_um={self.BOUNDARY_TOLERANCE_UM}')
                if away > self.ENDPOINT_TOLERANCE_UM and not drift_logged:
                    self.controller.info(f'{self.kind} FOCUS SWEEP: reversal drift within tolerance; '
                        f'start_z={start}; actual_z={z}; target_z={target}; '
                        f'away_um={away}; tolerance_um={self.BOUNDARY_TOLERANCE_UM}')
                    drift_logged = True
                self._motion_stop.wait(0.01)
        except Exception as error:
            self._motion_error = error
        finally:
            try:
                self.controller.microscope.stop()
            except Exception as error:
                if self._motion_error is None:
                    self._motion_error = error
            self._motion_done.set()

    def _pipette_rank(self, sample):
        """Prefer reliable XY near focus; Z confidence is not a ranking input."""
        score, _, _, depth = sample
        if depth is None:
            return (0, score, 0)
        error = abs(depth)
        if error <= self.DEPTH_TOLERANCE_UM:
            return (2, score, -error)
        return (1, -error, score)

    def _record(self, observation, score):
        vision = observation.get('deep_learning', {})
        frame = vision.get('source_frame')
        z = vision.get('source_positions', {}).get('focus_z')
        if (score is None or z is None or vision.get('stale', True)
                or frame is None or frame in self.seen_frames):
            if frame not in self.seen_frames:
                self.tip_samples.clear()
            return
        self.last_frame = frame
        self.seen_frames.add(frame)
        score, z = float(score), float(z)
        if not math.isfinite(score) or not math.isfinite(z) or not self.low <= z <= self.high:
            self.tip_samples.clear()
            return
        self.samples += 1
        is_cell = str(self.kind).lower() in ('search', 'cell')
        if str(self.kind).lower() in ('scan', 'pipette', 'search', 'cell'):
            tip = observation.get('cell_detection') if is_cell else vision.get('pipette_position')
            if tip is None or len(tip) < 2 or not all(math.isfinite(float(v)) for v in tip[:2]):
                self.tip_samples.clear()
                return
            tip = tuple(float(v) for v in tip[:2])
            try:
                depth = float(vision.get('pipette_focus'))
            except (TypeError, ValueError):
                depth = math.nan
            depth = depth if math.isfinite(depth) else None
            self.controller.info(f'{"SEARCH" if is_cell else "SCAN"} XY sample: frame={frame}; microscope_z={z}; '
                f'box_confidence={score}; detection_xy={tip}'
                + ('' if is_cell else f'; box_xyxy={vision.get("pipette_box_xyxy")}; predicted_defocus_um={depth}'))
            for sample in self.tip_samples:
                previous = sample[2]
                delta = self.controller.calibrated_stage.pixels_to_um_relative(
                    [tip[0] - previous[0], tip[1] - previous[1]])
                if math.hypot(float(delta[0]), float(delta[1])) > self.TIP_STABILITY_UM:
                    self.tip_samples.clear()
                    break
            self.tip_samples.append((score, z, tip) if is_cell else (score, z, tip, depth))
            self.tip_samples = self.tip_samples[-self.MIN_STABLE_SAMPLES:]
            if len(self.tip_samples) < self.MIN_STABLE_SAMPLES:
                return
            # Return to an actually sampled plane supported by a stable sequence,
            # rather than rewarding one isolated high-confidence prediction.
            if not is_cell:
                candidate = sorted(self.tip_samples, key=self._pipette_rank)[1]
                rank = self._pipette_rank(candidate)
                if self.best_tip_rank is None or rank > self.best_tip_rank:
                    score, z, tip, depth = candidate
                    self.best_tip_rank = rank
                    self.best_tip_xy, self.best_tip_confidence = tip, score
                    self.best_tip_defocus_um = depth
                    self.best_score, self.best_z = score, z
                return
            score, z, tip = sorted(self.tip_samples, key=lambda sample: sample[0])[1]
            if score > self.best_score:
                self.best_cell_xy, self.best_cell_confidence = tip, score
        if score > self.best_score:
            self.best_score, self.best_z = score, z

    def update(self, observation, score):
        self.controller.abort_if_requested()
        if self.stage == 'done':
            return True
        if self.stage == 'idle':
            raise AutopatchError('Focus sweep has not started')
        if self._motion_error is not None:
            raise AutopatchError(f'{self.kind} focus sweep motion failed: {self._motion_error}') from self._motion_error
        if time.monotonic() > self.deadline:
            if self._endpoint_correction:
                raise AutopatchError(f'{self.kind} focus sweep endpoint correction timed out; '
                    f'leg={self.stage}; target_z={self.target}; actual_z={observation["stage_positions"][2]}')
            raise AutopatchError(f'{self.kind} focus sweep timed out during {self.stage}')
        if self.stage in ('down', 'up') and not self._endpoint_correction:
            self._record(observation, score)
        z = float(observation['stage_positions'][2])
        if not math.isfinite(z):
            raise AutopatchError(f'{self.kind} focus sweep has no valid objective position')
        if self.stage in ('down', 'up') and not self._motion_done.is_set():
            return False
        tolerance = (self.BOUNDARY_TOLERANCE_UM if self.stage in ('down', 'up') and not self._endpoint_correction
                     else self.ENDPOINT_TOLERANCE_UM)
        if abs(z - self.target) > tolerance + 1e-6:
            direction = 1 if self.stage == 'down' else -1
            if (self.stage in ('down', 'up') and not self._endpoint_correction
                    and direction * (z - self.target) > self.BOUNDARY_TOLERANCE_UM):
                self._endpoint_correction = True
                self.controller.info(f'{self.kind} FOCUS SWEEP: correcting stopped endpoint overshoot; '
                    f'leg={self.stage}; actual_z={z}; target_z={self.target}; '
                    f'error_um={abs(z - self.target):.3f}; one absolute correction')
                self.controller.microscope.absolute_move(self.target)
            return False
        if self._endpoint_correction:
            self.controller.info(f'{self.kind} FOCUS SWEEP: endpoint corrected; '
                f'leg={self.stage}; actual_z={z}; target_z={self.target}; continuing sweep')
        if self.stage in ('down', 'up') and abs(z - self.target) > self.ENDPOINT_TOLERANCE_UM:
            self.controller.info(f'{self.kind} FOCUS SWEEP: endpoint accepted within tolerance; '
                                 f'target_z={self.target}; actual_z={z}; tolerance_um={tolerance}')
        if self.stage == 'seek_start':
            self._move('down', self.high)
        elif self.stage == 'down':
            self._move('up', self.low)
        elif self.stage == 'up':
            if self.best_z is None:
                if str(self.kind).lower() not in ('scan', 'pipette'):
                    raise EmptyFocusEvidence(f'{self.kind} focus sweep found no valid samples')
                self.best_z = self.center_z
                self.controller.warning(f'{self.kind} FOCUS SWEEP: no stable pipette XY samples; '
                    f'returning to saved pipette plane z={self.center_z}; fresh XY detection still required')
            elif str(self.kind).lower() in ('scan', 'pipette') and (
                    self.best_tip_defocus_um is None or abs(self.best_tip_defocus_um) > self.DEPTH_TOLERANCE_UM):
                self.controller.warning(f'{self.kind} FOCUS SWEEP: no stable near-focus XY candidate; '
                    f'using best available XY fallback; predicted_defocus_um={self.best_tip_defocus_um}; '
                    f'box_confidence={self.best_tip_confidence}; target_z={self.best_z}; '
                    'Z confidence ignored; Z estimate alone does not abort')
            self._move('return', self.best_z)
        elif self.stage == 'return':
            self.cancel()
            self.stage = 'done'
            self.controller.info(f'{self.kind} FOCUS SWEEP: finished; best_z={self.best_z}; '
                                 f'score={self.best_score}; valid_samples={self.samples}; '
                                 f'detection_xy={self.best_cell_xy if self.best_cell_xy is not None else self.best_tip_xy}; '
                                 + ('selection=stable XY confidence and predicted defocus; '
                                    f'predicted_defocus_um={self.best_tip_defocus_um}; Z confidence ignored'
                                    if str(self.kind).lower() in ('scan', 'pipette') else
                                    'selection=stable detector confidence'))
            return True
        return False

    def cancel(self):
        """Stop and join the motion monitor before the phase changes tasks."""
        self._motion_stop.set()
        try:
            self.controller.microscope.stop()
        finally:
            if self._motion_thread is not None:
                self._motion_thread.join(timeout=0.5)
                if self._motion_thread.is_alive():
                    raise AutopatchError(f'{self.kind} focus sweep monitor did not stop')

    def stop(self):
        self.cancel()
