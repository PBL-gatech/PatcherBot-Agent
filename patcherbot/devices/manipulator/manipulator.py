"""
Generic Manipulator class for manipulators.

To make a new device, one must implement at least:
* position
* absolute_move

TODO:
* Add minimum and maximum for each axis
"""
import time
import numpy as np
from numpy import array


from patcherbot.controller import TaskController

__all__ = ['Manipulator', 'ManipulatorError']


class ManipulatorError(Exception):
    def __init__(self, message = 'Device is not calibrated'):
        self.message = message

    def __str__(self):
        return self.message


class Manipulator(TaskController):
    def position(self, axis=None):
        '''
        Current position along an axis.

        Parameters
        ----------
        axis : axis number

        Returns
        -------
        The current position of the device axis in um.
        '''
        return 0. # fake

    def save_state(self):
        self.saved_state = self.position()

    def delete_state(self):
        self.saved_state = None

    def recover_state(self):
        self.absolute_move(self.saved_state)

    def  absolute_move(self, x, axis=None, speed=None):
        '''
        Moves the device axis to position x.

        Parameters
        ----------
        axis: axis number
        x : target position in um.
        '''
        #self.abort_if_requested()
        pass

    def relative_move(self, x, axis, speed=None):
        '''
        Moves the device axis by relative amount x in um.

        Parameters
        ----------
        axis: axis number
        x : position shift in um.
        '''
        if speed is not None:
            ##self.abort_if_requested()
            self.absolute_move(self.position(axis)+x, axis, speed)
        else:
            ##self.abort_if_requested()
            self.absolute_move(self.position(axis)+float(x), axis)

    def position_group(self, axes):
        '''
        Current position along a group of axes.

        Parameters
        ----------
        axes : list of axis numbers

        Returns
        -------
        np.ndarray
            The current position of the device axes in um (vector).
        '''
        return np.array([self.position(axis) for axis in axes])

    def absolute_move_group(self, x, axes, speed=None):
        '''
        Moves the device group of axes to position x.

        Parameters
        ----------
        axes : list of axis numbers
        x : target position in um (vector or list).
        '''
        #self.abort_if_requested()
        # self.info('Moving axes %s to position %s' % (axes, x))
        for xi,axis in zip(x,axes):
            self.absolute_move(xi, axis)

    def relative_move_group(self, x, axes, speed=None):
        '''
        Moves the device group of axes by relative amount x in um.

        Parameters
        ----------
        axes : list of axis numbers
        x : position shift in um (vector or list).
        '''
        self.absolute_move_group(array(self.position_group(axes))+array(x), axes)

    def relative_move_group_velocity(self, vel, axes=None):
        '''
        Moves the device group continuously at velocity vel in um/s.
        By default, this forwards to absolute velocity control.
        '''
        if hasattr(self, "absolute_move_group_velocity"):
            if axes is None:
                return self.absolute_move_group_velocity(vel)
            return self.absolute_move_group_velocity(vel, axes)
        raise NotImplementedError("relative_move_group_velocity is not implemented for this device.")

    def stop(self, axis):
        """
        Stops current movements.
        """

        pass

    def wait_until_still(self, axes = None):
        """
        Waits until motors have stopped.

        Parameters
        ----------
        axes : list of axis numbers
        """
        previous_position = self.position_group(axes)
        new_position = None
        while new_position is None or array(previous_position != new_position).any():
            previous_position = new_position
            new_position = self.position_group(axes)
            self.sleep(0.1)  # 100 ms

    def wait_until_reached(self, position, axes = None, precision = 0.5, timeout = 10):
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
        axes = array(axes)
        position = array(position)

        current_position = position
        previous_position = current_position
        t0 = time.time()
        while (abs(current_position-position)>precision).any():
            if (time.time()-t0>timeout) & (array(previous_position == current_position).all()):
                raise ManipulatorError("Time out while waiting for manipulator to reach target position.")
            previous_position = current_position
            if len(axes) == 1:
                current_position = array([self.position(axes[0])])
            else:
                current_position = self.position_group(axes)
            self.sleep(0.1)  # 100 ms
            

    def get_current_objective(self):
        """
        Return the currently selected objective, or ``None`` if unavailable.
        """
        return None

    def switch_objective(self, target):
        """
        Switch to the requested objective.
        """
        raise RuntimeError("Objective switching is not supported for this device.")

    def get_max_speed(self):
        ''' returns the max speed of the device, (if possible)
        '''
        pass
    
    def get_max_accel(self):
        ''' returns the max acceleration of the device, (if possible)
        '''
        pass
    def set_max_speed(self, speed):
        ''' sets the max speed of the device, (if possible)
        '''
        pass

    def set_max_accel(self, accel):
        ''' sets the max acceleration of the device, (if possible)
        '''
        pass
