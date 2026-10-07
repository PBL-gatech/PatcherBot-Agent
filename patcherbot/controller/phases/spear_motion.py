"""Bounded pipette Z velocity motion monitored independently of inference."""
import math
import threading
import time

from ..errors import AutopatchError


class SpearMotion:
    MAX_SPEED = 100.0
    ENDPOINT_TOLERANCE_UM = 0.2

    def __init__(self, controller, target, speed):
        self.controller = controller
        self.unit = controller.calibrated_unit
        self.target = float(target)
        requested_speed = float(speed)
        if not math.isfinite(self.target) or not math.isfinite(requested_speed) or requested_speed <= 0:
            raise AutopatchError('Spear requires a finite target and positive finite speed')
        self.speed = min(requested_speed, self.MAX_SPEED)
        self._thread = None
        self._cancel = threading.Event()
        self._done = threading.Event()
        self._error = None
        self._reached = False
        self.start_z = None
        self.deadline = None

    def _position(self):
        z = float(self.unit.position()[2])
        if not math.isfinite(z):
            raise AutopatchError('Spear has no valid pipette Z position')
        return z

    def start(self):
        self.controller.abort_if_requested()
        if self.start_z is not None:
            raise AutopatchError('Spear motion has already started')
        start = self._position()
        if self.target < start - self.ENDPOINT_TOLERANCE_UM:
            raise AutopatchError('Spear target is behind the current pipette position')
        self.start_z = start
        self.deadline = time.monotonic() + max(10.0, max(0.0, self.target - start) / self.speed * 20.0)
        self.unit.stop()
        self.controller.info(f'SPEAR motion: start_z={start}; target_z={self.target}; '
                             f'velocity_command={-self.speed}; endpoint monitor active')
        self._thread = threading.Thread(target=self._run, name='SpearMotion', daemon=True)
        self._thread.start()

    def _run(self):
        try:
            self.controller.abort_if_requested()
            if self._cancel.is_set():
                return
            z = self._position()
            if self.target - z <= self.ENDPOINT_TOLERANCE_UM:
                self._reached = True
                return
            # The rig's VJ Z sign is opposite to its measured positional Z.
            self.unit.absolute_move_group_velocity([0.0, 0.0, -self.speed])
            while not self._cancel.is_set():
                self.controller.abort_if_requested()
                if time.monotonic() > self.deadline:
                    raise AutopatchError('Spear motion timed out before reaching its target')
                z = self._position()
                if self.target - z <= self.ENDPOINT_TOLERANCE_UM:
                    self._reached = True
                    break
                if z < self.start_z - self.ENDPOINT_TOLERANCE_UM:
                    raise AutopatchError('Spear moved in the wrong Z direction')
                self._cancel.wait(0.01)
        except Exception as error:
            self._error = error
        finally:
            try:
                self.unit.stop()
            except Exception as error:
                if self._error is None:
                    self._error = error
            self._done.set()

    def update(self):
        self.controller.abort_if_requested()
        if self.start_z is None:
            raise AutopatchError('Spear motion has not started')
        if self._error is not None:
            raise AutopatchError(f'Spear motion failed: {self._error}') from self._error
        return self._done.is_set() and self._reached

    def stop(self):
        self._cancel.set()
        try:
            self.unit.stop()
        finally:
            if self._thread is not None:
                self._thread.join(timeout=0.5)
                if self._thread.is_alive():
                    raise AutopatchError('Spear motion monitor did not stop')
