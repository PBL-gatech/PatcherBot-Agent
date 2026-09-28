
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
        Parameters
        ----------
        dev : underlying device
        axes : list of 3 axis indexes
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

        Parameters
        ----------
        axis : axis number starting at 0; if None, all XYZ axes

        Returns
        -------
        The current position of the device axis in um.
        '''
        if axis is None: # all positions in a vector
            #return array([self.dev.position(self.axes[axis]) for axis in range(len(self.axes))])
            return self.dev.position_group(self.axes)
        else:
            return self.dev.position(self.axes[axis])

    def start_absolute_move(self, target):
        """Dispatch a move and retain its identity without waiting for completion."""
        target = np.asarray(target, dtype=float)
        if target.shape != (len(self.axes),) or not np.isfinite(target).all():
            raise ValueError("Move target must contain one finite value per axis")
        if not all(callable(getattr(self.dev, name, None)) for name in ("start_move", "poll_move")):
            raise NotImplementedError("This backend does not support supervised movement")
        handle = self.dev.start_move(target.tolist(), self.axes)
        if handle is None:
            raise RuntimeError("Backend did not return a supervised movement handle")
        self._supervised_pending = handle

    def start_relative_move(self, delta):
        delta = np.asarray(delta, dtype=float)
        if delta.shape != (len(self.axes),) or not np.isfinite(delta).all():
            raise ValueError("Move delta must contain one finite value per axis")
        self.start_absolute_move(np.asarray(self.position(), dtype=float) + delta)

    def poll_move(self):
        """Return running, settled or failed for this wrapper's own command."""
        handle = getattr(self, "_supervised_pending", None)
        if handle is None:
            return "failed"
        status = self.dev.poll_move(handle)
        if status not in ("running", "settled", "failed"):
            raise RuntimeError("Backend returned an invalid movement status")
        return status

    def absolute_move(self, x, axis = None, blocking=False, speed=None):
        '''
        Moves the device axis to position x in um.

        Parameters
        ----------
        axis : axis number starting at 0; if None, all XYZ axes
        x : target position in um.
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
        '''

        # self.info('Moving axes %s to position %s' % (axes, x))
        self.dev.absolute_move_group(x, np.array(self.axes)[axes], speed)
        #self.sleep(.05)

    def relative_move(self, x, axis = None, speed=None):
        '''
        Moves the device axis by relative amount x in um.

        Parameters
        ----------
        axis : axis number starting at 0; if None, all XYZ axes
        x : position shift in um.
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

        Parameters  
        ----------
        axis : axis number starting at 0; if None, all XYZ axes
        x : position shift in um.
        '''
        self.dev.relative_move_group(x, self.axes,speed)


    def poll_velocity(self):
        """Return the current backend velocity command status."""
        if not callable(getattr(self.dev, "poll_velocity", None)):
            raise NotImplementedError("This backend does not support supervised velocity")
        status = self.dev.poll_velocity()
        if status not in ("running", "failed"):
            raise RuntimeError("Backend returned an invalid velocity status")
        return status

    def start_velocity(self, velocity, *, relative=False):
        """Explicit supervised velocity dispatch; backend failures propagate."""
        velocity = np.asarray(velocity, dtype=float)
        if velocity.shape != (len(self.axes),) or not np.isfinite(velocity).all():
            raise ValueError("Velocity must contain one finite value per axis")
        if not all(callable(getattr(self.dev, name, None)) for name in ("start_velocity", "poll_velocity")):
            raise NotImplementedError("This backend does not support supervised velocity")
        self.dev.start_velocity(velocity.tolist(), self.axes, relative=relative)

    def absolute_move_group_velocity(self, vel):
        '''
        Moves the device in um/s.
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
        self._supervised_pending = None
        self.dev.stop()

    def wait_until_still(self, axes = None):
        """
        Waits for the motors to stop.
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

        Parameters
        ----------
        position : target position in micrometer
        axes : axis number of list of axis numbers
        precision : precision in micrometer
        timeout : time out in second
        """
        self.dev.wait_until_reached(position, axes, precision, timeout)

    def set_max_speed(self, speed):
        if speed is None:
            return
        if hasattr(self.dev, "set_max_speed"):
            self.dev.set_max_speed(speed)
    
    def set_max_accel(self, accel):
        if accel is None:
            return
        if hasattr(self.dev, "set_max_accel"):
            self.dev.set_max_accel(accel)

    def get_max_speed(self):
        if hasattr(self.dev, "get_max_speed"):
            return self.dev.get_max_speed()
        return None

    def get_max_accel(self):
        if hasattr(self.dev, "get_max_accel"):
            return self.dev.get_max_accel()
        return None
