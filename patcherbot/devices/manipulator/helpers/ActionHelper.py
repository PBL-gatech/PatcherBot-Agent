"""Bounded motor actions composed by calibrated devices and focus helpers."""
import time
import math
import logging
from threading import Event, Thread
from types import SimpleNamespace

class ActionHelper:
    """Execute and monitor velocity actions without owning calibration or phases."""
    MOTION_ENDPOINT_TOLERANCE_UM = 0.2

    def start(self, device, *, target, deadline, read_position,
              move_velocity, abort_check, label, info,
              direction=None, reversal_tolerance=0.2,
              reversal_slack=0.0, reversal_origin=None):
        """Start one cancellable encoder monitor; its thread owns the result."""

        sample_position = read_position
        def read_position():
            position = float(sample_position())
            if not math.isfinite(position):
                raise RuntimeError(f'{label} has no valid motor position')
            return position

        def run():
            try:
                abort_check()
                if motion.stop_requested.is_set():
                    return
                position = read_position()
                sign = direction if direction is not None else (1.0 if target > position else -1.0)
                tolerance = self.MOTION_ENDPOINT_TOLERANCE_UM
                if sign * (target - position) <= tolerance:
                    motion.reached = True
                    return
                abort_check()
                if motion.stop_requested.is_set():
                    return
                move_velocity(sign)
                origin = position if reversal_origin is None else reversal_origin

                def sample(expired):
                    nonlocal position
                    abort_check()
                    now = time.monotonic()
                    if expired(now):
                        raise RuntimeError(f'{label} timed out')
                    position = read_position()
                    return sign * (position - origin), -sign * (target - position), now

                def check_progress(progress, report=False):
                    outside = (position < origin - reversal_tolerance
                               if reversal_origin is not None and sign == 1.0 else
                               -progress > reversal_tolerance + reversal_slack)
                    if outside or report:
                        details = (f'start_z={origin}; actual_z={position}; target_z={target}; '
                                   f'away_um={-progress}; tolerance_um={reversal_tolerance}')
                        if outside:
                            raise RuntimeError(f'{label} moved in the wrong direction; {details}')
                        info(f'{label}: reversal drift within tolerance; {details}')
                    return False

                motion.reached = self.monitor(sample, endpoint=-tolerance, tolerance=tolerance,
                    start_time=0.0, deadline=deadline, cancelled=motion.stop_requested.is_set,
                    pause=lambda: motion.stop_requested.wait(0.01), reversed=check_progress,
                    report_drift=lambda progress: check_progress(progress, report=True),
                    drift_tolerance=tolerance).reached
            except Exception as error:
                motion.error = error
            finally:
                try:
                    device.stop()
                except Exception as error:
                    if motion.error is None:
                        motion.error = error
                motion.done = True

        motion = Thread(target=run, name=label.replace(' ', ''), daemon=True)
        motion.device, motion.abort_check, motion.label = device, abort_check, label
        motion.stop_requested = Event()
        motion.done, motion.reached, motion.error = False, False, None
        motion.start()
        return motion

    def monitor(self, sample, *, endpoint, tolerance, start_time, deadline,
                cancelled, pause, reversed, report_drift, drift_tolerance=0.0,
                progress_timeout=float('inf')):
        """Evaluate encoder progress with caller-owned sampling and motor cleanup."""
        result = SimpleNamespace(reached=False, traveled=0.0, elapsed=0.0, failure=None)
        best, last_progress, warned = 0.0, start_time, False
        expired = lambda now: now - start_time > deadline
        while not cancelled():
            progress, endpoint_progress, now = sample(expired)
            result.traveled, result.elapsed = progress, now - start_time
            if endpoint_progress >= endpoint:
                result.reached = True
                break
            if progress < -drift_tolerance and not warned:
                report_drift(progress)
                warned = True
            if reversed(progress):
                result.failure = 'reverse'
                break
            if progress >= best + tolerance:
                best, last_progress = progress, now
            if expired(now):
                result.failure = f"Overall movement timeout after {result.elapsed:.2f}s"
                break
            if now - last_progress > progress_timeout:
                result.failure = f"No forward progress for {now - last_progress:.2f}s"
                break
            pause()
        return result

    def is_complete(self, motion):
        """Read a nonblocking motion result and surface monitor failures."""
        motion.abort_check()
        done = motion.done
        if motion.error is not None:
            raise RuntimeError(f'{motion.label} failed: {motion.error}') from motion.error
        return done and motion.reached

    def cancel(self, motion):
        """Cancel and join an owned monitor before another operation starts."""
        motion.stop_requested.set()
        try:
            motion.device.stop()
        finally:
            motion.join(timeout=0.5)
            if motion.is_alive():
                raise RuntimeError(f'{motion.label} monitor did not stop')


class ObservationAction:
    """One continuous motion, progressed by values from the phase observation."""

    def __init__(self, device, move_velocity, tolerance=0.2, *, reversal_tolerance=1.0):
        self.device = device
        self.move_velocity = move_velocity
        self.tolerance = tolerance
        self.reversal_tolerance = reversal_tolerance
        self.started = self.running = self.complete = False

    def start(self, *, target, position, now, deadline, velocity):
        if self.running:
            raise RuntimeError("Stop the active action before starting another")
        if not all(math.isfinite(v) for v in
                   (target, position, now, deadline, velocity, self.tolerance, self.reversal_tolerance)):
            raise ValueError("Motion values must be finite")
        if velocity <= 0 or min(self.tolerance, self.reversal_tolerance) < 0 or deadline <= now:
            raise ValueError("Motion needs positive speed and a future deadline")
        self.origin, self.target, self.deadline = position, target, deadline
        self.reversal_warned = False
        self.direction = 1 if target >= position else -1
        self.started = True
        self.complete = abs(target - position) <= self.tolerance
        if not self.complete:
            self.running = True
            try:
                self.move_velocity(self.direction * velocity)
            except Exception:
                self.stop()
                raise

    def update(self, position, now):
        if not self.running or self.complete:
            return
        if not math.isfinite(position) or not math.isfinite(now):
            raise ValueError("Observed position/time must be finite")
        self.complete = self.direction * (self.target - position) <= self.tolerance
        if self.complete:
            return  # A crossed endpoint wins over a late sample's deadline.
        if now > self.deadline:
            raise RuntimeError("Motion timed out")
        away = -self.direction * (position - self.origin)
        if away > self.reversal_tolerance + 1e-6:
            raise RuntimeError(f"Motion moved in the wrong direction; start_z={self.origin}; "
                f"actual_z={position}; target_z={self.target}; away_um={away}; "
                f"tolerance_um={self.reversal_tolerance}")
        if away > 1e-6 and not self.reversal_warned:
            logging.getLogger(__name__).warning(
                "Motion reversal drift within tolerance: start_z=%s; actual_z=%s; "
                "target_z=%s; away_um=%s; tolerance_um=%s",
                self.origin, position, self.target, away, self.reversal_tolerance)
            self.reversal_warned = True

    def is_complete(self):
        return self.complete

    def stop(self):
        if self.running:
            try:
                self.device.stop()
            finally:
                self.running = False

    def reset(self):
        self.stop()
        self.started = self.complete = False
