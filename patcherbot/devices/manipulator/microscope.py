'''
A microscope is a manipulator with a single axis.
With methods to take a stack of images, autofocus, etc.

TODO:
* a umanager class that autoconfigures with umanager config file
* steps for stack acquisition?
'''
from patcherbot.devices.manipulator import Manipulator
import time
import warnings
try:
    import cv2
except:
    warnings.warn('OpenCV not available')

__all__ = ['Microscope']

class Microscope(Manipulator):
    '''
    A microscope Z axis, obtained here from an axis of a Manipulator.
    '''
    def __init__(self, dev, axis):
        '''
        Parameters
        ----------
        dev : underlying device
        axis : axis index
        '''
        Manipulator.__init__(self)
        self.dev : Manipulator = dev
        self.axis = axis
        self.up_direction = -1.0 # Up is negative on microscope Z by default
        self.floor_Z = None # This is the Z coordinate of the coverslip
        self.config = None
        self.units_per_um = 5.0
        self.objective_lift_um = 10000.0
        # Motor range in um; by default +- one meter
        self.min = -1e6 # This could replace floor_Z
        self.max = 1e6

    def set_max_speed(self, speed):
        self.dev.set_max_speed(speed)

    def position(self):
        '''
        Current position

        Returns
        -------
        The current position of the device axis in um.
        '''
        if self.config is not None:
            units_per_um = float(self.config.microscope_units_per_um)
        else:
            units_per_um = float(self.units_per_um)
        true_position = float(units_per_um * self.dev.position(self.axis))
        
        return true_position

    def absolute_move(self, x):
        '''
        Moves the device axis to position x in um.

        Parameters
        ----------
        x : target position in um.
        '''
        ##self.abort_if_requested()
        self.dev.absolute_move(x, self.axis)
        self.sleep(.05)

    def absolute_move_velocity(self, vel):
        '''
        Moves the device axis at velocity vel in um/s.

        Parameters
        ----------
        vel : velocity in um/s.
        '''
        ###self.abort_if_requested()
        velarr = [0,0,vel]
        self.dev.absolute_move_group_velocity(velarr)

        # self.sleep(.05)

    def move_to_floor(self):
        '''
        Moves the device axis to the floor position.
        '''
        ##self.abort_if_requested()
        self.dev.absolute_move(self.floor_Z, self.axis)
        self.dev.wait_until_still([self.axis])
        print(f"Moved to floor at {self.floor_Z} um")
        # self.dev.absolute_move(self.floor_Z, self.axis)
        # self.dev.wait_until_still([self.axis])

    def fix_backlash(self):
        '''
        Moves the device axis to a position and back to the original position.
        This is to fix backlash.
        '''
        ##self.abort_if_requested()
        curr_pos = self.position()
        self.absolute_move(curr_pos + 200)
        self.wait_until_still()
        self.absolute_move(curr_pos)
        self.wait_until_still()

    def relative_move(self, x):
        '''
        Moves the device axis by relative amount x in um.

        Parameters
        ----------
        x : position shift in um.
        '''
        ##self.abort_if_requested()
        self.dev.relative_move(x, self.axis)
        self.sleep(.05)

    def step_move(self, distance):
        '''
        Moves the device axis by a fixed step distance in um.
        Parameters
        ----------
        distance : step size in um.
        '''
        ###self.abort_if_requested()
        self.dev.step_move(distance, self.axis)

    def stop(self):
        """
        Stop current movements.
        """
        self.dev.stop()

    def wait_until_still(self):
        """
        Waits for the motors to stop.
        """
        self.dev.wait_until_still([self.axis])
        self.sleep(.05)


    def get_current_objective(self):
        return self.dev.get_current_objective()

    def switch_objective(self, target):
        try:
            target = int(target)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Invalid objective target: {target}') from exc

        if target not in (1, 2):
            raise ValueError(f'Objective target must be 1 or 2, got {target}')

        lift_um = getattr(self, 'objective_lift_um', None)
        if lift_um is None:
            lift_um = getattr(self.dev, 'objective_lift_um', None)
        if lift_um is None and self.config is not None and hasattr(self.config, 'objective_lift_um'):
            lift_um = self.config.objective_lift_um
        if lift_um is None:
            lift_um = 10000.0
        lift_um = float(lift_um)

        direction = -1.0
        print(f'[OBJDBG] Microscope.switch_objective start target={target} axis={self.axis} lift_um={lift_um} direction={direction}')
        try:
            print(f'[OBJDBG] pre-lift position={self.dev.position()}')
        except Exception as exc:
            print(f'[OBJDBG] pre-lift position read failed: {exc}')

        if lift_um > 0:
            print(f'[OBJDBG] calling relative_move delta={direction * lift_um} axis={self.axis}')
            self.dev.relative_move(direction * lift_um, self.axis)
            self.dev.wait_until_still([self.axis])
            self.sleep(.05)
            try:
                print(f'[OBJDBG] post-lift position={self.dev.position()}')
            except Exception as exc:
                print(f'[OBJDBG] post-lift position read failed: {exc}')

        try:
            try:
                print(f'[OBJDBG] pre-OBJ position={self.dev.position()}')
            except Exception as exc:
                print(f'[OBJDBG] pre-OBJ position read failed: {exc}')
            print(f'[OBJDBG] sending OBJ switch target={target}')
            result = self.dev.switch_objective(target)
            print(f'[OBJDBG] OBJ switch completed target={target}')
            try:
                print(f'[OBJDBG] post-OBJ position={self.dev.position()}')
            except Exception as exc:
                print(f'[OBJDBG] post-OBJ position read failed: {exc}')
        finally:
            if lift_um > 0:
                print(f'[OBJDBG] calling relative_move delta={-direction * lift_um} axis={self.axis}')
                self.dev.relative_move(-direction * lift_um, self.axis)
                self.dev.wait_until_still([self.axis])
                self.sleep(.05)
                try:
                    print(f'[OBJDBG] post-lower position={self.dev.position()}')
                except Exception as exc:
                    print(f'[OBJDBG] post-lower position read failed: {exc}')

        self._last_objective = target
        return result

    def toggle_objective(self):
        current = self.get_current_objective()
        if current == 1:
            target = 2
        elif current == 2:
            target = 1
        else:
            last = getattr(self, '_last_objective', None)
            if last == 1:
                target = 2
            elif last == 2:
                target = 1
            else:
                target = 1
        return self.switch_objective(target)

    def stack(self, camera, z, preprocessing=lambda img:img, save = None, pause = 0.3):
        '''
        Take a stack of images at the positions given in the z list

        Parameters
        ----------
        camera : a camera, eg with a snap() method
        z : A list of z positions
        preprocessing : a function that processes the images (optional)
        save : saves images to disk if True
        pause : pause in second after each movement
        '''
        position = self.position()
        images = []
        current_z = position
        for k,zi in enumerate(z):
            #self.absolute_move(zi)
            self.relative_move(zi-current_z)
            current_z = zi
            self.wait_until_still()
            # We wait a little bit because there might be mechanical oscillations
            time.sleep(pause) # also make sure the camera is in sync
            img = preprocessing(camera.snap())
            images.append(img)
            if save is not None:
                cv2.imwrite('./screenshots/'+save+'{}.jpg'.format(k), img)
        self.absolute_move(position)
        self.wait_until_still()
        return images

    def save_configuration(self):
        '''
        Outputs configuration in a dictionary.
        '''
        config = {'up_direction' : self.up_direction,
                  'floor_Z' : self.floor_Z}
        return config

    def load_configuration(self, config):
        '''
        Loads configuration from dictionary config.
        Variables not present in the dictionary are untouched.
        '''
        self.up_direction = config.get('up_direction', self.up_direction)
        #self.floor_Z = config.get('floor_Z', self.floor_Z)
