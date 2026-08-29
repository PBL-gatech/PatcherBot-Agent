
"""
A class for access to a particular unit managed by a device.
It is essentially a subset of a Manipulator
"""
from __future__ import absolute_import

from numpy import ones, arange
import numpy as np

from .manipulator import Manipulator

__all__ = ['ManipulatorUnit']


class ManipulatorUnit(Manipulator):
    def __init__(self, dev, axes):
        '''
        Initialize a ManipulatorUnit, representing a subset of a device.

        Args:
            dev: The underlying device (Manipulator) to control.
            axes: List of 3 axis indices corresponding to this unit.
        '''
        Manipulator.__init__(self)
        self.dev = dev
        self.axes = axes
        # Motor ranges in um; by default +- one meter
        self.min = -ones(len(axes))*1e6
        self.max = ones(len(axes))*1e6

    def position(self, axis = None):
        '''
        Current position along an axis.

        Args:
            axis: Axis number starting at 0; if None, returns all XYZ axes.

        Returns:
            The current position in micrometers (um), or a NumPy array of positions.
        '''
        if axis is None: # all positions in a vector
            #return array([self.dev.position(self.axes[axis]) for axis in range(len(self.axes))])
            return self.dev.position_group(self.axes)
        else:
            return self.dev.position(self.axes[axis])

    def absolute_move(self, x, axis = None, blocking=False, speed=None):
        '''
        Moves the device axis to position x in um.

        Args:
            x: Target position in um, or list/array for all axes.
            axis: Axis number starting at 0; if None, applies to all axes.
            blocking: If True, waits until movement completes.
            speed: Speed of movement (optional).
        '''

        if axis is None:
            # self.info('Moving axis %s to position %s' % (self.axes[axis], x))
            # then we move all axes
            if blocking:
                for i, axis in enumerate(self.axes):
                    # self.info('Moving axis %s to position %s' % (axis, x[i]))
                    self.dev.absolute_move(x[i], axis, speed)
                    self.dev.wait_until_still([axis])
            else:
                # self.info('Moving axes %s to position %s' % (self.axes, x))
                self.dev.absolute_move_group(x, self.axes, speed)
        else:
            # self.info('Moving axis %s to position %s' % (self.axes[axis], x))
            self.dev.absolute_move(x, self.axes[axis], speed)
            if blocking:
                self.dev.wait_until_still([self.axes[axis]])
        #self.sleep(.05)

    def absolute_move_group(self, x, axes, speed=None):
        '''
        Moves the device axes to positions x in um.

        Args:
            x: Target positions in um (vector/list).
            axes: List of axis indices (relative to this unit).
            speed: Speed of movement (optional).
        '''

        # self.info('Moving axes %s to position %s' % (axes, x))
        self.dev.absolute_move_group(x, np.array(self.axes)[axes], speed)
        #self.sleep(.05)

    def relative_move(self, x, axis = None, speed=None):
        '''
        Moves the device axis by relative amount x in um.

        Args:
            x: Relative displacement in um, or list/array for all axes.
            axis: Axis number starting at 0; if None, applies to all axes.
            speed: Speed of movement (optional).
        '''
        # self.abort_if_requested()
        if axis is None:
            # self.info('Moving axes %s by relative amount %s' % (self.axes, x))
            self.dev.relative_move_group(x, self.axes, speed)
        else:
            # self.info('Moving axis %s by relative amount %s' % (self.axes[axis], x))
            self.dev.relative_move(x, self.axes[axis], speed)
        # self.sleep(.05)

    def relative_move_group(self, x, axis=None, speed=None):
        '''
        Moves the device in um/s by relative amount x in all axes.

        Args:
            x: Relative displacement in um, or list/array for all axes.
            axis: Axis number starting at 0; if None, applies to all axes.
            speed: Speed of movement (optional).
        '''
        self.dev.relative_move_group(x, self.axes,speed)


    def absolute_move_group_velocity(self, vel):
        '''
        Moves the device in um/s.

        Args:
            vel: Velocity in um/s (scalar or iterable).
        '''
        try:
            self.dev.absolute_move_group_velocity(vel)
        except TypeError:
            # Some backends require explicit device axes for velocity commands.
            self.dev.absolute_move_group_velocity(vel, self.axes)
        # self.sleep(.005)

    def relative_move_group_velocity(self, vel):
        '''
        Moves the device in um/s using the relative-velocity API when available.
        '''
        if hasattr(self.dev, "relative_move_group_velocity"):
            try:
                self.dev.relative_move_group_velocity(vel)
                return
            except TypeError:
                self.dev.relative_move_group_velocity(vel, self.axes)
                return
        self.absolute_move_group_velocity(vel)

    def stop(self):
        """
        Stop current movements.
        """
        # self.abort_if_requested()
        self.dev.stop()

    def wait_until_still(self, axes = None):
        """
        Waits for the motors to stop.

        Args:
            axes: Axis indices relative to this unit; if None, waits for all axes.
        """
        if axes is None: # all axes
            axes = arange(len(self.axes))
        if hasattr(axes, '__len__'):  # is that useful?
            for i in axes:
                self.wait_until_still(i)
        else:
            self.dev.wait_until_still([self.axes[axes]])
        # self.sleep(.005)

    def wait_until_reached(self, position, axes=None, precision=0.5, timeout=10):
        """
        Waits until position is reached within precision, and raises an error if the
        target is not reached after the time out, unless the manipulator is still moving.

        Args:
            position: Target position(s) in um.
            axes: Axis indices relative to this unit; if None, applies to all axes.
            precision: Allowed error in um.
            timeout: Maximum wait time in seconds.
        """
        self.dev.wait_until_reached(position, axes, precision, timeout)

    def set_max_speed(self, speed):
        """
        Set maximum speed for the unit.

        Args:
            speed: Maximum speed to set (if supported).
        """
        if speed is None:
            return
        if hasattr(self.dev, "set_max_speed"):
            self.dev.set_max_speed(speed)
    
    def set_max_accel(self, accel):
        """
        Set maximum acceleration for the unit.

        Args:
            accel: Maximum acceleration to set (if supported).
        """
        if accel is None:
            return
        if hasattr(self.dev, "set_max_accel"):
            self.dev.set_max_accel(accel)

    def get_max_speed(self):
        """
        Get maximum speed for the unit.

        Returns:
            Maximum speed if available, else None.
        """
        if hasattr(self.dev, "get_max_speed"):
            return self.dev.get_max_speed()
        return None

    def get_max_accel(self):
        """
        Get maximum acceleration for the unit.

        Returns:
            Maximum acceleration if available, else None.
        """
        if hasattr(self.dev, "get_max_accel"):
            return self.dev.get_max_accel()
        return None
