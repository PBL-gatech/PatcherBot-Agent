"""Finite objective focus sweeps; the calling phase continues frame inference."""
import math
import threading
import time

from ..errors import AutopatchError


class FocusSweep:
    RADIUS_UM = 5.0
    ENDPOINT_TOLERANCE_UM = 0.2
    BOUNDARY_TOLERANCE_UM = 1.0

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
        self.deadline = None
        self.speed = None
        self._motion_thread = None
        self._motion_stop = threading.Event()
        self._motion_done = threading.Event()
        self._motion_error = None

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

    def _record(self, observation, score):
        vision = observation.get('deep_learning', {})
        frame = vision.get('source_frame')
        z = vision.get('source_positions', {}).get('focus_z')
        if (score is None or z is None or vision.get('stale', True)
                or frame is None or frame == self.last_frame):
            return
        self.last_frame = frame
        score, z = float(score), float(z)
        if not math.isfinite(score) or not math.isfinite(z) or not self.low <= z <= self.high:
            return
        self.samples += 1
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
            raise AutopatchError(f'{self.kind} focus sweep timed out during {self.stage}')
        if self.stage in ('down', 'up'):
            self._record(observation, score)
        z = float(observation['stage_positions'][2])
        if not math.isfinite(z):
            raise AutopatchError(f'{self.kind} focus sweep has no valid objective position')
        if self.stage in ('down', 'up') and not self._motion_done.is_set():
            return False
        tolerance = (self.BOUNDARY_TOLERANCE_UM if self.stage in ('down', 'up')
                     else self.ENDPOINT_TOLERANCE_UM)
        if abs(z - self.target) > tolerance + 1e-6:
            return False
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
                    raise AutopatchError(f'{self.kind} focus sweep found no valid samples')
                self.best_z = self.center_z
                self.controller.warning(f'{self.kind} FOCUS SWEEP: no valid pipette Z samples; '
                    f'returning to saved pipette plane z={self.center_z}; fresh XY detection still required')
            self._move('return', self.best_z)
        elif self.stage == 'return':
            self.cancel()
            self.stage = 'done'
            self.controller.info(f'{self.kind} FOCUS SWEEP: finished; best_z={self.best_z}; '
                                 f'score={self.best_score}; valid_samples={self.samples}')
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
