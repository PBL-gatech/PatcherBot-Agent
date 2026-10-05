"""
A fake device useful for development.
It has 9 axes, numbered 1 to 9.
"""

from __future__ import print_function, absolute_import
from .manipulator import Manipulator

from numpy import zeros, clip, pi
import numpy as np
import time
import math

__all__ = ['FakeManipulator']

class FakeManipulator(Manipulator):
    def __init__(self, min=None, max=None):
        Manipulator.__init__(self)
        # Minimum and maximum positions for all axes.
        self.min = min
        self.max = max
        if all([min is not None, max is not None]):
            if len(min) != len(max):
                raise ValueError('min/max needs to be the same length (# of axes)')
        self.num_axes = len(min)

        # Continuous movement values.
        self.x = zeros(self.num_axes)  # current position (um) of each axis
        self.set_max_speed(10000)        # default max speed (as provided)
        self.setpoint = self.x.copy()     # target positions for moves
        self.speeds = zeros(self.num_axes)  # current speeds (in internal units)
        self.cmd_time = [None] * self.num_axes  # last update time for each axis
        self.max_accel = None  # not simulated but provided for API compatibility

    def set_max_speed(self, speed: int):
        """
        Set the maximum speed. (Note that for some reason when you specify
        1000 as the max speed it actually moves at ~82 um/s.)
        """
        self.max_speed = speed / 1000 * 82

    def set_max_accel(self, accel):
        """
        Set the maximum acceleration (not used in this simulation).
        """
        self.max_accel = accel

    def position(self, axis=None):
        """
        Returns the current position(s) in micrometers.
        
        Parameters
        ----------
        axis : int or None
            If None, returns the positions for all axes (as a numpy array);
            otherwise, returns the position for the given axis (numbered starting at 1).
        """
        if axis is None:
            # Update all axes and return a copy of the positions.
            for i in range(self.num_axes):
                self.update_axis(i+1)
            return self.x.copy()
        else:
            self.update_axis(axis)
            return self.x[axis-1]



    def position_group(self, axes):
        """
        Returns the current positions for the given axes.
        
        Parameters
        ----------
        axes : iterable
            List of axis numbers.
        
        Returns
        -------
        np.ndarray
            A NumPy array containing the positions for the specified axes.
        """
        for axis in axes:
            self.update_axis(axis)
        indices = [axis - 1 for axis in axes]
        # Convert self.x to a NumPy array so that advanced indexing works
        return np.array(self.x)[indices]


    def update_axis(self, axis):
        """
        Updates the position on the given axis if a command is being executed.
        
        Returns
        -------
        bool
            True if the move command is still running, False otherwise.
        """
        idx = axis - 1
        # Nothing commanded on this axis.
        if self.cmd_time[idx] is None:
            return False

        now = time.time()
        dt = now - self.cmd_time[idx]
        # Save the update time.
        self.cmd_time[idx] = now

        # Compute the new position.
        new_pos = self.x[idx] + self.speeds[idx] * dt

        # If moving in the positive direction and not yet at setpoint:
        if self.speeds[idx] > 0 and new_pos < self.setpoint[idx]:
            self.x[idx] = new_pos
            return True
        # If moving in the negative direction and not yet at setpoint:
        if self.speeds[idx] < 0 and new_pos > self.setpoint[idx]:
            self.x[idx] = new_pos
            return True

        # Otherwise, we have reached (or exceeded) the setpoint:
        self.x[idx] = self.setpoint[idx]
        self.speeds[idx] = 0
        self.cmd_time[idx] = None
        return False

    def absolute_move(self, x, axis, speed=None):
        """
        Moves the given axis to an absolute position x (in um).

        Parameters
        ----------
        x : float
            Target position in micrometers.
        axis : int
            Axis number (starting at 1).
        speed : int or None
            Speed in the same units as set_max_speed. If None, the max speed is used.
        """
        self.abort_if_requested()
        if self.update_axis(axis):
            raise RuntimeError("Cannot move while another command is running on axis {}".format(axis))

        # If limits are defined, clip the target position.
        if self.min is not None and self.max is not None:
            target = clip(x, self.min[axis-1], self.max[axis-1])
        else:
            target = x

        self.setpoint[axis-1] = target
        self.cmd_time[axis-1] = time.time()
        current_pos = self.x[axis-1]
        # Determine which speed to use.
        if speed is not None:
            used_speed = speed / 1000 * 82
        else:
            used_speed = self.max_speed
        # Set the speed in the proper direction.
        direction = math.copysign(1, target - current_pos) if target != current_pos else 0
        self.speeds[axis-1] = used_speed * direction

    def absolute_move_group(self, x, axes, speed=None):
        """
        Moves a group of axes to the specified absolute positions.
        
        Parameters
        ----------
        x : iterable
            Target positions (in um) for each axis.
        axes : iterable
            List of axis numbers.
        speed : int or None
            Speed to use for all axes (if provided).
        """
        self.abort_if_requested()
        for target, axis in zip(x, axes):
            self.absolute_move(target, axis, speed)

    def relative_move(self, dx, axis, speed=None):
        """
        Moves the given axis by a relative amount.
        
        Parameters
        ----------
        dx : float
            Displacement (in um) to move the axis.
        axis : int
            Axis number (starting at 1).
        speed : int or None
            Speed (if provided, otherwise max_speed is used).
        """
        self.abort_if_requested()
        current_pos = self.position(axis)
        new_target = current_pos + dx
        self.absolute_move(new_target, axis, speed)

    def relative_move_group(self, dx, axes, speed=None):
        """
        Moves a group of axes by a relative displacement.
        
        Parameters
        ----------
        dx : float or iterable
            If a scalar, the same displacement (in um) is applied to all axes.
            Otherwise, dx should be an iterable of displacements.
        axes : iterable
            List of axis numbers.
        speed : int or None
            Speed (if provided).
        """
        # Determine whether dx is scalar or iterable.
        try:
            iter(dx)
            dx_iterable = True
        except TypeError:
            dx_iterable = False

        targets = []
        for i, axis in enumerate(axes):
            current_pos = self.position(axis)
            displacement = dx[i] if dx_iterable else dx
            targets.append(current_pos + displacement)
        self.absolute_move_group(targets, axes, speed)

    def absolute_move_group_velocity(self, vel, axes):
        """
        Moves the given group of axes continuously at the specified velocity.
        
        Parameters
        ----------
        vel : float or iterable
            Velocity in um/s. If a scalar, that velocity is used for all axes.
        axes : iterable
            List of axis numbers.
            
        Notes
        -----
        Because this is a velocity move, the setpoint is set to an
        infinite value (in the correct direction) so that the axis keeps moving
        until a stop command is issued.
        """
        self.abort_if_requested()
        current_time = time.time()
        try:
            iter(vel)
            vel_iterable = True
        except TypeError:
            vel_iterable = False

        for i, axis in enumerate(axes):
            if self.update_axis(axis):
                raise RuntimeError("Cannot move while another command is running on axis {}".format(axis))
            v = vel[i] if vel_iterable else vel
            self.speeds[axis-1] = v / 1000 * 82  # conversion to internal speed units
            self.cmd_time[axis-1] = current_time
            self.setpoint[axis-1] = float('inf') if v > 0 else float('-inf')

    def relative_move_group_velocity(self, vel, axes):
        """
        Relative velocity-mode API; fake backend uses direct velocity setpoints.
        """
        self.absolute_move_group_velocity(vel, axes)

    def stop(self, axis):
        """
        Stops any movement on the specified axis.
        
        Parameters
        ----------
        axis : int
            Axis number (starting at 1).
        """
        self.update_axis(axis)
        idx = axis - 1
        self.speeds[idx] = 0
        self.cmd_time[idx] = None
        self.setpoint[idx] = self.x[idx]

    def wait_until_still(self, axes=None):
        """
        Blocks until the specified axes have finished moving.
        
        Parameters
        ----------
        axes : iterable or None
            List of axis numbers to wait for. If None, waits for all axes.
        """
        if axes is None:
            axes = range(1, self.num_axes + 1)
        elif not hasattr(axes, '__iter__'):
            axes = [axes]

        still_moving = True
        while still_moving:
            still_moving = False
            for axis in axes:
                if self.update_axis(axis):
                    still_moving = True
            time.sleep(0.1)

    def wait_until_reached(self, position, axes=None, precision=0.5, timeout=10):
        """
        Blocks until the given axes reach the target positions within the specified precision.
        
        Parameters
        ----------
        position : float or iterable
            Target position(s) in um.
        axes : iterable or None
            List of axis numbers to check. If None, all axes are used.
        precision : float
            Allowed error (in um) between current and target positions.
        timeout : float
            Maximum time (in seconds) to wait before raising an error.
        """
        start_time = time.time()
        if axes is None:
            axes = list(range(1, self.num_axes + 1))
        elif not hasattr(axes, '__iter__'):
            axes = [axes]

        position = np.array(position)
        while True:
            all_reached = True
            for i, axis in enumerate(axes):
                self.update_axis(axis)
                current_pos = self.x[axis-1]
                target = position[i] if position.size > 1 else position.item()
                if abs(current_pos - target) > precision:
                    all_reached = False
                    break
            if all_reached:
                break
            if time.time() - start_time > timeout:
                raise RuntimeError("Timeout waiting for axes {} to reach target positions.".format(axes))
            time.sleep(0.1)
