from sensapex import UMP
import numpy as np
from ctypes import c_float, byref
import math
import threading
import time

from patcherbot.devices.manipulator.manipulator import Manipulator


class SensapexManip(Manipulator):
    """
    Sensapex uMp/uMs wrapper that mirrors the callable surface of ScientificaSerial* drivers.

    Sources:
      - UMP.get_ump(), list_devices(), get_device(id) for discovery and selection.
      - device.get_pos(1) returns position list in micrometers.
      - device.goto_pos(target_um, speed_um_per_s) initiates motion and returns a movement handle
        exposing finished_event.
      - device.stop() halts motion.
    """

    DEFAULT_MAX_SPEED = 5000
    DEFAULT_MAX_ACCELERATION = 1

    def __init__(self, deviceID=None, ump: UMP = None, poll_hz: float = 100.0,
                 max_speed=None, max_acceleration=None, tilt_angle_deg=-25.7):
        Manipulator.__init__(self)

        # UMP connection and device selection
        self.ump = ump if ump is not None else UMP.get_ump()
        if deviceID is None:
            ids = self.ump.list_devices()
            assert len(ids) == 1, "must specify sensapex ump device id if there is more than 1 connected!"
            self.device_id = ids[0]
        else:
            self.device_id = int(deviceID)
        self.dev = self.ump.get_device(self.device_id)

        self._lock = threading.RLock()
        self.tilt_angle_deg = float(tilt_angle_deg)

        # Tunables stored for API compatibility
        self.max_speed = self.DEFAULT_MAX_SPEED if max_speed is None else max_speed
        self.max_acceleration = self.DEFAULT_MAX_ACCELERATION if max_acceleration is None else max_acceleration
        self._max_speed = float(self.max_speed)
        self._max_accel = float(self.max_acceleration)
        # Position cache (um)
        self.current_pos = [0.0, 0.0, 0.0]
        # Number of axes
        self.dev.set_n_axes(4)  # force 4-axis mode
        test = self.dev.n_axes()
        self.info("Sensapex device has %d axes", test)
        pos = self.dev.get_pos(1)
        self.info("Initial position readback: %s", str(pos))
        self._n_axes = 3



        # Movement tracking
        self._last_move = None
        self._move_gate = threading.Lock()

        # Velocity emulation state
        self._vel = [0.0, 0.0, 0.0]
        self._vel_enabled = True
        self._vel_dt = 0.05
        self._vel_thread = threading.Thread(target=self._velocity_worker, daemon=True)
        self._vel_thread.start()

        # Start background polling
        self._polling_thread = threading.Thread(target=self.update_pos_continuous, args=(poll_hz,), daemon=True)
        self._polling_thread.start()

    def __del__(self):
        try:
            self._vel_enabled = False
        except Exception:
            pass

    # ---- Speed / accel compatibility ----
    def get_max_speed(self):
        return float(self._max_speed)

    def get_max_accel(self):
        return float(self._max_accel)

    def set_max_speed(self, speed):
        self.max_speed = speed
        self._max_speed = float(speed)

    def set_max_accel(self, accel):
        self.max_acceleration = accel
        self._max_accel = float(accel)

    @staticmethod
    def _convert_coords(xyz, angle, direction):
        """
        Convert between Sensapex motor coords and stage/world coords.
        direction=True: motor -> stage/world
        direction=False: stage/world -> motor
        """
        coords = list(xyz)
        if len(coords) < 3:
            coords.extend([coords[-1] if coords else 0.0] * (3 - len(coords)))

        angle_rad = math.radians(float(angle))
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        if direction:
            mx, my, mz = float(coords[0]), float(coords[1]), float(coords[2])
            wx = mx * cos_a
            wy = my
            wz = mz - mx * sin_a
            out = [wx, wy, wz]
        else:
            wx, wy, wz = float(coords[0]), float(coords[1]), float(coords[2])
            mx = wx / cos_a
            my = wy
            mz = wz + mx * sin_a
            out = [mx, my, mz]

        if len(coords) > 3:
            out.extend(coords[3:])
        return out

    

    # ---- Position helpers ----
    def position(self, axis=None):
        with self._lock:
            raw_pos = list(self.current_pos)
        pos = raw_pos
        if axis is None:
            return pos
        return pos[axis - 1]

    def raw_position(self, axis=None):
        try:
            pos = list(self.dev.get_pos(1))
        except Exception:
            with self._lock:
                pos = list(self.current_pos)
        if axis is None:
            return pos
        return pos[axis - 1]

    def update_pos_continuous(self, freq=100.0):
        """
        Constantly polls device position and updates self.current_pos.
        Uses device.get_pos(1) which returns micrometers.
        """
        period = 1.0 / float(freq) if freq and freq > 0 else 0.01
       
        while True:
            t0 = time.time()
            try:
                pos = list(self.dev.get_pos(1))
               
                with self._lock:
                    self._n_axes = len(pos)
                    if len(pos) < 3:
                        pos.extend([pos[-1]] * (3 - len(pos)))
                    world_pos = self._convert_coords(pos, self.tilt_angle_deg, False)
                    self.current_pos = [float(world_pos[0]), float(world_pos[1]), float(world_pos[2])]
            except Exception:
                pass

            dt = time.time() - t0
            sleep_time = period - dt
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _get_last_move_event(self):
        with self._lock:
            mv = self._last_move
        if mv is None:
            return None
        try:
            return getattr(mv, "finished_event", None)
        except Exception:
            return None

    def _wait_for_last_move_event(self, timeout=None):
        evt = self._get_last_move_event()
        if evt is None:
            return False
        try:
            evt.wait(timeout)
            return evt.is_set()
        except Exception:
            return False

    def _issue_move(self, target, speed, gate=True):
        if gate:
            with self._move_gate:
                self._wait_for_last_move_event(None)
                motor_target = self._convert_coords(target, self.tilt_angle_deg, True)
                mv = self.dev.goto_pos(motor_target, speed)
                with self._lock:
                    self._last_move = mv
                return mv
        motor_target = self._convert_coords(target, self.tilt_angle_deg, True)
        mv = self.dev.goto_pos(motor_target, speed)
        with self._lock:
            self._last_move = mv
        return mv

    # ---- Motion primitives ----
    def absolute_move(self, pos, axis, speed=None):
        self.absolute_move_group([pos], [axis], speed=speed)

    def absolute_move_group(self, x, axes, speed=None):
        x = list(x)
        axes = list(axes)
        
        with self._lock:
            try:
                full_current = list(self.dev.get_pos(1))
            except Exception:
                full_current = list(self.current_pos)

            if len(full_current) < 3:
                full_current.extend([full_current[-1] if full_current else 0.0] * (3 - len(full_current)))
            world_current = list(self.current_pos)
            if len(world_current) < 3:
                world_current.extend([world_current[-1] if world_current else 0.0] * (3 - len(world_current)))
            full_current[:3] = world_current[:3]

            target = list(full_current)
            for val, ax in zip(x, axes):
                ax_i = int(ax) - 1
                if ax_i < 0:
                    continue
                while ax_i >= len(target):
                    target.append(target[-1] if len(target) else 0.0)
                target[ax_i] = float(val)

            sp = float(self._max_speed if speed is None else speed)
            mv = self._issue_move(target[: len(target)], sp, gate=True)
            if 1 in [int(ax) for ax in axes]:
                try:
                    raw = self.raw_position()
                    print(f"[Sensapex X move issued] raw_delta={self._raw_offset(raw)}")
                except Exception:
                    pass
            return mv

    def relative_move_group(self, x, axes, speed=None):
        """
        Supports both call styles:
          - relative_move_group(pos, axis, speed=None)
          - relative_move_group(x_list, axes_list, speed=None)
        """
        if isinstance(axes, (int, np.integer)) and not isinstance(x, (list, tuple, np.ndarray)):
            pos = float(x)
            axis = int(axes)
            return self.relative_move_group([pos], [axis], speed=speed)

        x = list(x)
        axes = list(axes)
       
        with self._lock:
            cur = list(self.current_pos)
        delta = [0.0, 0.0, 0.0]
        for val, ax in zip(x, axes):
            ax_i = int(ax) - 1
            if 0 <= ax_i < 3:
                delta[ax_i] = float(val)

        target = [cur[0] + delta[0], cur[1] + delta[1], cur[2] + delta[2]]
        return self.absolute_move_group(target, [1, 2, 3], speed=speed)

    def absolute_move_group_velocity(self, vel, axes=None):
        """
        Emulates continuous velocity commands by integrating a requested velocity vector.
        """
        vel = list(vel)
        v3 = [0.0, 0.0, 0.0]
        if axes is None:
            if len(vel) >= 3:
                v3 = [float(vel[0]), float(vel[1]), float(vel[2])]
        else:
            axes = list(axes)
            for v, ax in zip(vel, axes):
                ax_i = int(ax) - 1
                if 0 <= ax_i < 3:
                    v3[ax_i] = float(v)
        with self._lock:
            self._vel = v3

    def relative_move_group_velocity(self, vel, axes=None):
        """
        Relative velocity-mode API; Sensapex backend integrates commanded velocities.
        """
        self.absolute_move_group_velocity(vel, axes=axes)

    def wait_until_still(self, axes=None, axis=None):
        """
        Block until the device is no longer moving.

        Prefer querying the SDK for device busy / drive status so this works even if motion was
        initiated outside this wrapper.

        Sensapex umsdk provides:
          - um_is_busy(hndl, dev): "Check if a device is busy." :contentReference[oaicite:2]{index=2}
          - um_get_drive_status(hndl, dev): "Obtain position drive status." :contentReference[oaicite:3]{index=3}

        umsdk also notes function name migration from ump_* -> um_* in the newer branch, so we try both. :contentReference[oaicite:4]{index=4}
        """
        def _call_any(names):
            for fn in names:
                try:
                    return self.ump.call(fn, self.device_id)
                except Exception:
                    pass
            return None

        while True:
            evt = self._get_last_move_event()
            if evt is not None:
                try:
                    if not evt.is_set():
                        evt.wait(0.01)
                        continue
                except Exception:
                    pass
            # 1) Preferred: explicit busy query (um_is_busy / ump_is_busy)
            busy = _call_any(("um_is_busy", "ump_is_busy"))
            if busy is not None:
                try:
                    if int(busy) == 0:
                        return
                except Exception:
                    # if it's already boolean-like
                    if not bool(busy):
                        return
                time.sleep(0.01)
                continue

            # 2) Next: drive status (um_get_drive_status / ump_get_drive_status)
            # libum.h defines: COMPLETED=0, BUSY=1, FAILED=-1 :contentReference[oaicite:5]{index=5}
            st = _call_any(("um_get_drive_status", "ump_get_drive_status"))
            if st is not None:
                try:
                    st_i = int(st)
                    if st_i == 0:
                        return
                    if st_i < 0:
                        # avoid deadlock on failure: treat as "not moving"
                        return
                except Exception:
                    pass
                time.sleep(0.01)
                continue

            # 3) Fallback: wait on the last move handle (older behavior)
            if evt is not None:
                return
            with self._lock:
                mv = self._last_move
            if mv is None:
                return
            try:
                mv.finished_event.wait(None)
            except Exception:
                pass
            return

    def stop(self, axis=None):
        """
        Stops current movements.
        """
        with self._lock:
            self._vel = [0.0, 0.0, 0.0]
        try:
            self.dev.stop()
        except Exception:
            pass

    # ---- Internals ----
    def _get_axis_angle(self):
        angle = c_float()
        try:
            rVal = self.ump.call("ump_get_axis_angle", self.device_id, byref(angle))
        except Exception:
            rVal = None
        return angle.value

    def _velocity_worker(self):
        """
        Integrates the velocity vector into small absolute moves.
        """
        while True:
            if not getattr(self, "_vel_enabled", False):
                return

            with self._lock:
                v = list(self._vel)

            if v[0] == 0.0 and v[1] == 0.0 and v[2] == 0.0:
                time.sleep(0.02)
                continue

            dt = float(self._vel_dt)
            with self._lock:
                cur = list(self.current_pos)
            target = [cur[0] + v[0] * dt, cur[1] + v[1] * dt, cur[2] + v[2] * dt]

            requested = max(abs(v[0]), abs(v[1]), abs(v[2]))
            sp = min(max(requested, 1.0), float(self._max_speed))

            try:
                mv = self._issue_move(target, sp, gate=False)
                try:
                    mv.finished_event.wait(dt)
                except Exception:
                    time.sleep(dt)
            except Exception:
                time.sleep(dt)
