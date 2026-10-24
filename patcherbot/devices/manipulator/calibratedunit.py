# coding=utf-8
"""
A class to handle a manipulator unit with coordinates calibrated to the reference system of a camera.
It contains methods to calibrate the unit.

Should messages be issued?
Also ranges should be taken into account

Should this be in devices/ ? Maybe in a separate calibration folder
"""
from __future__ import print_function
from __future__ import absolute_import
from .manipulatorunit import *
from numpy import array, zeros, dot, isnan
import cv2
import numpy as np
import time
import math
from patcherbot.devices.manipulator import *

from numpy.linalg import pinv, norm
from .helpers.ActionHelper import ActionHelper
from .helpers.StageCalHelper import FocusHelper, StageCalHelper
from .helpers.StageScanHelper import StageScanHelper
from .helpers.PipetteCalHelper import PipetteCalHelper, PipetteFocusHelper
from .helpers.CellDetectHelper import CellDetectHelper

__all__ = ['CalibratedUnit', 'CalibrationError', 'CalibratedStage']

verbose = True

class CalibrationError(Exception):
    def __init__(self, message='Device is not calibrated'):
        self.message = message

    def __str__(self):
        return self.message


class CalibratedUnit(ManipulatorUnit):
    def __init__(self, unit, stage=None, microscope=None, camera=None,
                 config=None):
        '''
        A manipulator unit calibrated to a fixed reference coordinate system.
        The stage refers to a platform on which the unit is mounted, which can
        be None.

        Parameters
        ----------
        unit : ManipulatorUnit for the (XYZ) unit
        stage : CalibratedUnit for the stage
        microscope : ManipulatorUnit for the microscope (single axis)
        camera : a camera, ie, object with a snap() method (optional, for visual calibration)
        '''
        ManipulatorUnit.__init__(self, unit.dev, unit.axes)
        self.saved_state_question = ('Move manipulator and stage back to '
                                     'initial position?')
        if config is None:
            raise ValueError(
                "CalibratedUnit requires an explicit CalibrationConfig instance."
            )
        self.config = config
        if stage is None: # In this case we assume the unit is on a fixed element.
            self.stage = FixedStage()
            self.fixed = True
        else:
            self.stage = stage
            self.fixed = False
        self.microscope = microscope
        self.camera = camera

        self.calibrated = False
        self.z_calibration = None
        self.must_be_recalibrated = False
        self.up_direction = [-1 for _ in range(len(unit.axes))] # Default up direction, determined during calibration
        self.abort_requested = False
        self.pipette_position = None
        self.photos = None
        self.photo_x0 = None
        self.photo_y0 = None

        # Matrices for passing to the camera/microscope system
        self.M = zeros((3,len(unit.axes))) # stage units (in micron) to camera
        self.Minv = zeros((len(unit.axes),3)) # Inverse of M
        self.r0 = zeros(3) # offset for px -> um conversion
        self.r0_inv = zeros(3) # offset for um -> px conversion
        self.unit = unit
        self.actionHelper = ActionHelper()

        self.emperical_offset = np.zeros(3) # offset for pipette position in px based on deep learning model

        #setup pipette calibration helper class
        self.pipetteCalHelper = PipetteCalHelper(unit, self.microscope, camera, stage, config=self.config)
        self.pipetteFocusHelper = PipetteFocusHelper(
            unit, camera, config=self.config, detector=self.pipetteCalHelper.pipetteDetector
        )


    def detect_pipette(self, observer=None):
        evidence = None
        if observer is None:
            frame = self.camera.last_raw_frame_data()
            if frame is None or frame[2] is None:
                self.camera.show_circles([])
                return None
            point = self.pipetteCalHelper.pipetteDetector.detect_pipette(frame[2])
        else:
            anchor = getattr(self, '_tip_reference', None)
            evidence = observer.infer_frame(["pipette_image_xy"], frame_context={
                "pipette_expected_xy": lambda: (self.reference_position(anchor=anchor)[:2] if anchor is not None
                    else np.array([self.camera.width / 2., self.camera.height / 2.])),
                "pipette_xyz": self.position, "stage_xy": self.stage.position},
                confirmation={"confidence_threshold": lambda: 0.5})
            point = evidence.get("confirmed_tip_xy")
            if point is None:
                point = evidence.get("pipette_position")
        if point is None:
            self.camera.show_circles([])
            return None
        point = np.asarray(point, dtype=float).reshape(-1)
        if point.size < 2 or not np.all(np.isfinite(point[:2])):
            self.camera.show_circles([])
            return None
        point = tuple(np.rint(point[:2]).astype(int))
        if evidence is not None and evidence.get("confirmed_tip_xy") is not None:
            source = evidence["source_positions"]
            self.set_tip_reference(evidence["confirmed_tip_xy"], source["pipette_xyz"], source["stage_xy"])
        self.camera.show_circle(point)
        return point



    def set_tip_reference(self, tip, position=None, stage_position=None, trusted=True):
        """Anchor image XY to supplied sampling positions, or current positions."""
        if position is None:
            position = self.position()
        if stage_position is None:
            stage_position = self.stage.position()
        self._tip_reference = (np.asarray(tip, dtype=float)[:2].copy(),
                                np.asarray(position, dtype=float).copy(),
                                np.asarray(stage_position, dtype=float).copy())
        if trusted:
            self._trusted_tip_reference = tuple(value.copy() for value in self._tip_reference)


    def save_state(self):
        if self.stage is not None:
            self.stage.save_state()
        if self.microscope is not None:
            self.microscope.save_state()
        self.saved_state = self.position()

    def delete_state(self):
        if self.stage is not None:
            self.stage.delete_state()
        if self.microscope is not None:
            self.microscope.delete_state()
        self.saved_state = None

    def recover_state(self):
        if self.stage is not None:
            self.stage.recover_state()
        if self.microscope is not None:
            self.microscope.recover_state()
        self.absolute_move(self.saved_state)


    def pixels_to_um(self, pos_pixels):
        '''
        Converts pixel coordinates to pipette um.
        '''
        if self.Minv.shape[1] == 2: #2x2 stage movement
            xy = dot(self.Minv, pos_pixels[0:2]) + self.r0_inv[0:2]
            return np.array([xy[0], xy[1], 0])
        else: #3x3 pipette movement
            return dot(self.Minv, pos_pixels) + self.r0_inv
    
    def pixels_to_um_relative(self, pos_pixels):
        '''
        Converts pixel coordinates to pipette um.
        '''
        if self.Minv.shape[1] == 2: #2x2 stage movement
            xy = dot(self.Minv, pos_pixels[0:2])
            return np.array([xy[0], xy[1], 0])
        else: #3x3 pipette movement
            return dot(self.Minv, pos_pixels)

    
    def um_to_pixels(self, pos_microns):
        '''
        Converts um to pixel coordinates.
        '''
        return dot(self.M, pos_microns) + self.r0 - self.emperical_offset
    
    def um_to_pixels_relative(self, pos_microns):
        '''
        Converts um to pixel coordinates.
        '''
        return dot(self.M, pos_microns)
    



    def reference_position(self, include_offset=True, *, anchor=None):
        '''
        Position of the pipette in pixels (camera coordinate frame)

        Returns
        -------
        The current position in um as an XYZ vector.
        '''
        pos_um = self.position() # position vector (um) in manipulator unit system
        self.debug(f"pipette position: {pos_um}")
        pipette_pos_pixels = self.um_to_pixels(pos_um) 
        self.debug(f"pipette position in pixels: {pipette_pos_pixels}")
        if include_offset:
            pos_pixels = self.um_to_pixels(pos_um) + self.stage.reference_position() + self.emperical_offset
        else:
            pos_pixels = self.um_to_pixels(pos_um) + self.stage.reference_position()
        if anchor is not None:
            tip, position, stage_position = anchor
            pos_pixels = np.asarray(pos_pixels, dtype=float).copy()
            pos_pixels[:2] = (np.asarray(tip, dtype=float)[:2]
                + self.um_to_pixels_relative(pos_um - position)[:2]
                + self.stage.um_to_pixels_relative(self.stage.position() - stage_position)[:2])
        return pos_pixels # position vector (pixels) in camera system

    def reference_move(self, pos_pixels):
        '''
        Moves the unit to position pos_pixels in reference camera system, without moving the stage.

        Parameters
        ----------
        r : XYZ position vector in um

        '''
        self.abort_if_requested()
        if np.isnan(np.array(pos_pixels)).any():
            raise RuntimeError("can not move to nan location.")
        
        if isinstance(self, CalibratedStage) or isinstance(self, FixedStage):
            self.info(f'desired position: {pos_pixels}')
            self.info(f'Stage reference position: {self.stage.reference_position()}')
            pos_micron = self.pixels_to_um(pos_pixels - self.stage.reference_position()) # position vector (um) in manipulator unit system
            self.info(f'Position in um: {pos_micron}')
            self.absolute_move(pos_micron)
            self.wait_until_still()
            return
        else:
            self.info(f'desired position: {pos_pixels}')
            self.info(f'Stage reference position (used for pipette calibration): {self.stage.reference_position()}')
            pos_micron = self.pixels_to_um(pos_pixels - self.stage.reference_position())
            self.info(f'Position in um: {pos_micron}')
            self.absolute_move(pos_micron)
            self.wait_until_still()
            return



    def autofocus_pipette(self):
        '''Use the microscope image to put the pipette in focus
        '''
        if not self.config.pipette_focus_crop_feature:
            self.info('Autofocusing pipette without cropping')
            self.abort_if_requested()
            self.pipetteFocusHelper.focus()
        else:
            self.info('Autofocusing pipette with cropping')
            self.abort_if_requested()
            self.autofocus_cropped_pipette()

    def autofocus_cropped_pipette(self,crop_size=256):
        ''' use pipette detector to crop ROI of pipette tip, then feed directly into focushelpers focuser'''
        _, _, _, img = self.camera.raw_frame_queue[0]
        pipette_px = self.pipetteCalHelper.pipetteDetector.detect_pipette(img)
        if pipette_px is None:
            self.error("No pipette detected in the current frame.")
            return
        pipette_px = np.array(pipette_px)
        h, w = img.shape[:2]
        x_min = max(int(pipette_px[0] - crop_size // 2), 0)
        x_max = min(int(pipette_px[0] + crop_size // 2), w)
        y_min = max(int(pipette_px[1] - crop_size // 2), 0)
        y_max = min(int(pipette_px[1] + crop_size // 2), h)
        cropped_img = img[y_min:y_max, x_min:x_max]
        self.pipetteFocusHelper.focus(cropped_img)


    def safe_move(self, r):
        '''
        Moves the device to position x (an XYZ vector) in a way that minimizes
        interaction with tissue.

        Parameters
        ----------
        r : target position in um, an (X,Y,Z) vector
        yolo_correction : if True, corrects the pipette position using YOLO object detection
        '''
        if not self.calibrated:
            raise CalibrationError
        if self.must_be_recalibrated:
            raise CalibrationError('Pipette offsets must be recalibrated')
        self.abort_if_requested()
        self.reference_move(r) # Or relative move in manipulator coordinates, first axis (faster)

    def pixel_per_um(self, M=None):
        '''
        Returns the objective magnification in pixel per um, calculated for each manipulator axis.
        '''
        if M is None:
            M = self.M
        p = []
        for axis in range(len(self.axes)):
            # The third axis is in um, the first two in pixels, hence the odd formula
            p.append(((M[0,axis]**2 + M[1,axis]**2))**.5) #TODO: is this correct? 
        return p
    
    def rotate(self,coordinates,axis):
        '''
        Rotate the coordinates around the given axis at a specified angle using a rotation matrix.
        '''
        if coordinates is None:
            return None
        # if the stage coordinates need to be flipped do so
        if self.config.stage_x_axis_flip:
            coordinates[0] = -coordinates[0]
        if self.config.stage_y_axis_flip:
            coordinates[1] = -coordinates[1]
        if axis == 0:
            # Rotation matrix around the X-axis.
            R = np.array([[1, 0, 0],
                          [0, np.cos(theta), -np.sin(theta)],
                          [0, np.sin(theta),  np.cos(theta)]])
        elif axis == 1:
            # Rotation matrix around the Y-axis.
            theta = self.config.pipette_y_rotation * np.pi / 180
            R = np.array([[np.cos(theta), 0, np.sin(theta)],
                          [0, 1, 0],
                          [-np.sin(theta), 0, np.cos(theta)]])
        elif axis == 2:
            theta = self.config.pipette_z_rotation * np.pi / 180
            # Rotation matrix around the Z-axis.
            R = np.array([[np.cos(theta), -np.sin(theta), 0],
                          [np.sin(theta),  np.cos(theta), 0],
                          [0, 0, 1]])
        else:
            raise ValueError("Invalid axis. Please choose 0 (X), 1 (Y), or 2 (Z).")
        rotated = np.dot(R, coordinates)
        self.debug(f"Rotated coordinates: {rotated}")
        return rotated
        

    def calibrate_pipette(self):
        '''
        Calibrate the pipette using YOLO object detection and pipette encoders to create a um -> pixels transformation matrix
        '''
        self.abort_if_requested()
        self.pipetteCalHelper.collect_cal_points()
        self.finish_calibration()
        self.center_pipette()
        self.wait_until_still()
        self.autofocus_pipette()
        self.wait_until_still()
        self.center_pipette()
        self.wait_until_still()
        self.autofocus_pipette()
        self.wait_until_still()


    def center_pipette(self, speed=None, detected_px=None, expected_px=None, max_deviation_um=None):
        """
        Moves the pipette so that its detected position in the camera image is centered.
        """
        options = {} if detected_px is None else {"detected_px": detected_px}
        if max_deviation_um is not None:
            options.update(expected_px=expected_px, max_deviation_um=max_deviation_um)
        return self.direct_pipette(speed=speed, **options)

    def direct_pipette(self, desired_px=None, speed=None, detected_px=None, expected_px=None, max_deviation_um=None):
        """
        Moves the pipette so that its detected position matches the requested image coordinates.
        If no coordinates are provided, the pipette is centered in the camera view.
        An explicit speed (um/s) uses velocity control to support slow centering.
        """
        self.last_pipette_move_failure = None
        self.abort_if_requested()
        # (1) Retrieve an image from the raw frame queue.
        _, _, _, img = self.camera.raw_frame_queue[0]
        h, w = img.shape[:2]
        # self.debug("DEBUG: Camera image dimensions: width =", w, "height =", h)
        
        # (2) Get the detected pipette position (in pixels) from the deep-learning detector.
        supplied_detection = detected_px is not None
        # Sample beside image retrieval, before inference can delay the anchor.
        # These are cached device positions, not camera exposure timestamps.
        reference_position = np.asarray(self.position(), dtype=float).copy()
        reference_stage_position = np.asarray(self.stage.position(), dtype=float).copy()
        if not supplied_detection:
            detected_px = self.pipetteCalHelper.pipetteDetector.detect_pipette(img)
        if detected_px is None:
            self.last_pipette_move_failure = "No pipette detected in the current frame."
            self.error(self.last_pipette_move_failure)
            return False
        detected_px = np.array(detected_px)
        if detected_px.shape not in ((2,), (3,)) or not np.isfinite(detected_px).all():
            self.last_pipette_move_failure = 'Invalid confirmed pipette XY'
            return False
        if max_deviation_um is not None:
            expected = np.asarray(expected_px, dtype=float)
            if (expected.shape != (2,) or not np.isfinite(expected).all()
                    or not np.isfinite(max_deviation_um) or max_deviation_um <= 0):
                self.last_pipette_move_failure = 'Missing valid calibrated pipette reference; SHIFT cancelled'
                self.warning(self.last_pipette_move_failure)
                return False
            if not np.linalg.norm(self.stage.pixels_to_um_relative(np.r_[detected_px[:2] - expected, 0.])[:2]) <= max_deviation_um:
                self.last_pipette_move_failure = 'Pipette detection disagrees with calibrated position; SHIFT cancelled'
                self.warning(f'{self.last_pipette_move_failure}; detected_xy={detected_px[:2]}; '
                             f'expected_xy={expected}; tolerance_um={max_deviation_um}')
                return False
        self.set_tip_reference(detected_px[:2], position=reference_position,
                               stage_position=reference_stage_position, trusted=not supplied_detection)
        # Ensure the detected position is expressed as a 3D vector.
        if detected_px.size == 2:
            detected_px = np.append(detected_px, 0)
        # self.debug("DEBUG: Detected pipette position (pixels):", detected_px)

        # (3) Define the desired pipette position.
        if desired_px is None:
            # Default to the image center when no target coordinates are supplied.
            desired_px = np.array([w / 2.0, h / 2.0, 0])
        else:
            if speed is None:
                self.unit.set_max_speed(500)
            desired_px = np.array(desired_px)
            if desired_px.size == 2:
                desired_px = np.append(desired_px, 0)
            elif desired_px.size > 3:
                desired_px = desired_px[:3]
            # For planar calibration, we set missing z entries to 0.
            if desired_px.size < 3:
                desired_px = np.pad(desired_px, (0, 3 - desired_px.size), constant_values=0)
        # self.debug("DEBUG: Desired pipette position (image center):", desired_px)
        
        # (4) Compute the pixel error (desired minus detected).
        error_px = desired_px - detected_px
        # self.debug("DEBUG: Pixel error (desired - detected):", error_px)
        
        # (5) Convert the pixel error into a correction (in microns).
        # pixels_to_um_relative() expects a 3-element vector.
        error_um = self.pixels_to_um_relative(error_px)
        self.info(f'Pipette XY correction: detected={detected_px[:2]}; desired={desired_px[:2]}; '
                  f'displacement_um={error_um}; source={"confirmed scan" if supplied_detection else "fresh detector"}')
        if speed is not None:
            if not np.isfinite(speed) or speed <= 0:
                raise ValueError("Pipette centering speed must be finite and positive.")
            return self._velocity_move_by_displacement(error_um, speed)
        # self.debug("DEBUG: Correction in microns (from pixel error):", error_um)
        
        # (6) Get the current manipulator (pipette) position (in microns) and compute the target.
        current_um = self.position()
        # self.debug("DEBUG: Current manipulator position (um):", current_um)
        target_um = current_um + error_um
        # self.debug("DEBUG: Computed target manipulator position (um):", target_um)
        
        # (7) Command the move and wait until the unit is still.
        self.absolute_move(target_um.tolist())
        self.wait_until_still()
        # self.debug("DEBUG: Centering move complete.")


    def direct_pipette_3D(self, desired_px3D):
        '''
        Moves the pipette so that its detected position matches the requested 3D image coordinates.
        '''
        self.abort_if_requested()
        #(1) get image from raw frame queue
        _, _, _, img = self.camera.raw_frame_queue[0]
        #(2) get detected pipette position from deep learning detector
        detected_px = self.pipetteCalHelper.pipetteDetector.detect_pipette(img)
        #(3) extract planar values from desired_px3D
        if detected_px is None:
            self.error("No pipette detected in the current frame.")
            return
        detected_px = np.asarray(detected_px, dtype=float)  # expected length 2 (x, y)
        desired_px3D = np.asarray(desired_px3D, dtype=float)
        if desired_px3D.size < 3:
            # goal z defaults to 0 defocus if not provided
            desired_px3D = np.pad(desired_px3D, (0, 3 - desired_px3D.size), constant_values=0)
        desired_px = desired_px3D[:2]

        # (4) Compute the pixel error (desired minus detected).
        error_px = desired_px - detected_px
        # self.debug("DEBUG: Pixel error (desired - detected):", error_px)
        
        # (5) Convert the pixel error into a correction (in microns).
        # pixels_to_um_relative() expects a 3-element vector.
        error_um = self.pixels_to_um_relative(np.array([error_px[0], error_px[1], 0]))

        # (5.5) Add z correction from desired_px3D (treat value as defocus to negate)
        z_correction = -float(desired_px3D[2])
        error_um_3D = np.array([error_um[0], error_um[1], z_correction])

        # (6) Get the current manipulator (pipette) position (in microns) and compute the target.
        current_um = self.position()
        # self.debug("DEBUG: Current manipulator position (um):", current_um)
        target_um = current_um + error_um_3D
        # self.debug("DEBUG: Computed target manipulator position (um):", target_um)
        
        # (7) Command the move and wait until the unit is still.
        self.absolute_move(target_um.tolist())
        self.wait_until_still()
        # self.debug("DEBUG: Centering move complete.")


    def record_cal_point(self):
        '''
        records a calibration point for the pipette
        '''
        self.pipetteCalHelper.record_cal_point()

    def finish_calibration(self):
        '''
        Automatic calibration of the pipette manipulator.
        '''
        
        # move the pipette and create a calibration matrix (pix -> um)
        mat = self.pipetteCalHelper.calibrate()
        
        #make matrix 4x4 for inverse
        mat = np.vstack((mat, np.array([0,0,0,1])))

        self.stage.pipette_cal_position = self.stage.position() #update the position the pipette was calibrated at

        # *** Compute the (pseudo-)inverse ***
        mat_inv = pinv(mat)

        self.debug(f'calibration matrix: {mat}')
        self.debug(f'inv :  {mat_inv}')

        # store r0 and r0_inv
        self.r0 = -mat[0:3, 3] #um -> pixels offset
        self.r0_inv = -mat_inv[0:3, 3] #pixels -> um offset

        #just 3x3 portion of M for self.M
        self.M = mat[0:3, 0:3]

        #just 3x3 portion of Minv
        self.Minv = mat_inv[0:3, 0:3]


        #check for nan values (invalid cal)
        if isnan(self.M).any() or isnan(self.Minv).any():
            raise CalibrationError('Matrix contains NaN values')

        self.debug('Calibration Successful!')
        self.debug(f'M: {self.M}')
        self.debug(f'r0: {self.r0}')
        self.debug(f'Minv:  {self.Minv}')
        self.debug(f'r0_inv: {self.r0_inv}')

        self.calibrated = True
        self.must_be_recalibrated = False
    

    def recalibrate_pipette(self):
        '''recalibrate pipette offset while keeping matrix
        '''
        if self.M is None or self.Minv is None:
            raise Exception("initial calibration required for single point recalibration!")
        
        self.debug('recalculating pipette offsets...')
        emperical_poses = []
        for i in range(10):
            _, _, _, frame = self.camera.raw_frame_queue[0]
            pos = self.pipetteCalHelper.pipetteDetector.detect_pipette(frame)
            if pos != None:
                emperical_poses.append([pos[0], pos[1]])
        
        if len(emperical_poses) == 0:
            self.debug("No pipette found in image, can't run correction")
            return
        
        new_offset = np.zeros(3)
        pos_pixels_emperical = np.median(emperical_poses, axis=0)
        pos_pixels_emperical = np.append(pos_pixels_emperical, self.microscope.position())
        new_offset = pos_pixels_emperical - self.um_to_pixels_relative(self.dev.position()) - self.stage.reference_position()
        self.debug(f'Old offsets: {self.r0}, {self.r0_inv}')

        self.r0 = np.array(new_offset)

        #create r0_inv
        homogenous_mat = np.zeros((4,4))
        homogenous_mat[0:3, 0:3] = self.M
        homogenous_mat[0:3, 3] = self.r0
        homogenous_mat[3, 3] = 1

        homogenous_mat_inv = pinv(homogenous_mat)

        #get r0_inv
        self.r0_inv = homogenous_mat_inv[0:3, 3]
        self.calibrated = True
        self.must_be_recalibrated = False

        self.debug(f'New offsets: {self.r0}, {self.r0_inv}')

    def follow_stage(self, movement = 300):
        '''
        Moves the pipette to follow the stage, method used for testing/calibration.
        '''
        #1. move stage by movement in both axes randomly
        movement_vector = np.array([movement * (np.random.rand() - 0.5), movement * (np.random.rand() - 0.5), 0])
        self.stage.relative_move(movement_vector)
        self.stage.wait_until_still()
        #2. rotate movement vector around z axis by pipette_z_rotation
        rotated_vector = self.rotate(movement_vector, 2)
        #3. move pipette by rotated movement vector
        self.relative_move(rotated_vector)
        self.wait_until_still()


    def velocity_position_control(self, position_delta, speed):
        """
        Convert a displacement vector and scalar speed into per-axis velocities.

        Parameters
        ----------
        position_delta : iterable of length 3
            Relative displacement [dx, dy, dz] in um.
        speed : float
            Requested travel speed magnitude in um/s.

        Returns
        -------
        list
            Velocity command [vx, vy, vz] in um/s.
        """
        delta = np.asarray(position_delta, dtype=float).reshape(-1)
        if delta.size != 3:
            raise ValueError("position_delta must be a 3-element vector [dx, dy, dz].")

        distance = float(norm(delta))
        if distance == 0:
            return [0.0, 0.0, 0.0]

        speed = abs(float(speed))
        if speed == 0:
            raise ValueError("Speed must be non-zero for velocity control.")

        unit_direction = delta / distance
        return (unit_direction * speed).tolist()


    def _velocity_move_by_displacement(self, movement_vector, speed, poll_interval=0.01):
        """
        Execute a relative displacement using a continuous velocity command.

        This is used for low-speed motion where firmware clamps can prevent
        `relative_move` from honoring requested speeds.
        """
        self.last_pipette_move_failure = None
        movement_vector = np.asarray(movement_vector, dtype=float)
        distance = float(norm(movement_vector))
        if distance == 0:
            return True

        speed = abs(float(speed))
        if speed == 0:
            raise ValueError("Speed must be non-zero for velocity moves.")

        # Allow Scientifica's 0.1 um readback resolution; short moves still require positive progress.
        position_tolerance = min(0.1, distance * 0.1)
        direction = movement_vector / distance
        # Match FindPipettePhase's displacement-to-velocity convention: the
        # velocity command has the opposite sign to positional displacement.
        velocity = (-np.asarray(self.velocity_position_control(movement_vector, speed), dtype=float)).tolist()
        start_pos = np.asarray(self.position(), dtype=float)
        expected_time = distance / speed
        # VJ command magnitude is not a measured encoder speed. The short
        # watchdog limits time without forward progress, not total move time.
        progress_timeout = max(2.0, expected_time * 5.0)
        overall_timeout = max(10.0, expected_time * 20.0)
        start_time = time.perf_counter()
        self.info(f"Pipette velocity move: target={distance:.2f} um; velocity_command={velocity}; "
                  f"no_progress_timeout_s={progress_timeout:.2f}; overall_timeout_s={overall_timeout:.2f}")

        def sample(expired):
            curr_pos = np.asarray(self.position(), dtype=float)
            traveled = float(np.dot(curr_pos - start_pos, direction))
            return traveled, traveled, time.perf_counter()

        self.absolute_move_group_velocity(velocity)
        try:
            result = self.actionHelper.monitor(sample, endpoint=distance - position_tolerance,
                tolerance=position_tolerance, start_time=start_time,
                deadline=overall_timeout,
                cancelled=lambda: self.abort_requested, pause=lambda: self.sleep(poll_interval),
                reversed=lambda progress: progress <= -distance,
                report_drift=lambda progress: self.warning(
                    "Velocity move is progressing away from the target position."),
                progress_timeout=progress_timeout)
            completed, signed_traveled, elapsed = result.reached, result.traveled, result.elapsed
            self.last_pipette_move_failure = (
                "Pipette moved away from its target" if result.failure == 'reverse' else result.failure)
        finally:
            self.stop()
            self.wait_until_still()
        if not completed:
            reason = self.last_pipette_move_failure or "Pipette movement aborted"
            self.last_pipette_move_failure = (
                f"{reason}; target={distance:.2f} um; traveled={signed_traveled:.2f} um; elapsed={elapsed:.2f}s")
            self.warning(self.last_pipette_move_failure)
        else:
            self.info(f"Pipette velocity move finished: target={distance:.2f} um; "
                      f"traveled={signed_traveled:.2f} um; elapsed={elapsed:.2f}s")
        return completed

    def move_pipette_random_velocity(self, movement = 500, speed = 200):
        '''
        Moves the pipette randomly in xy plane, method used for testing/calibration/data collection.
        For speeds below 1000 um/s, this uses velocity commands instead of
        relative moves to avoid firmware speed clamping.
        '''
        orig = self.get_max_speed()
        self.info(f"Moving pipette randomly for testing/calibration, original speed is: {orig} um/s")

        requested_speed = abs(float(speed))
        if requested_speed == 0:
            raise ValueError("Speed must be non-zero.")

        velocity_threshold = 1000.0
        use_velocity_mode = requested_speed < velocity_threshold

        if use_velocity_mode:
            self.info(
                f"Requested pipette speed {requested_speed:.1f} um/s is below "
                f"{int(velocity_threshold)} um/s; using velocity-command motion."
            )
        else:
            requested_speed_int = int(requested_speed)
            self.set_max_speed(requested_speed_int)
            test_speed = self.get_max_speed()
            if test_speed != requested_speed_int:
                self.warning(
                    f"Requested pipette speed {requested_speed_int} um/s, but device readback is {test_speed} um/s. "
                    "This is likely a Scientifica firmware clamp or device-unit limit."
                )
            else:
                self.info(f"Set pipette speed to {test_speed} um/s for random movement.")

        try:
            # this section is for testing the find pipette method.
            # movement_vector = np.array([movement * (np.random.rand() - 0.5), movement * (np.random.rand() - 0.5), (movement/5) * (np.random.rand() - 0.5)])
            # self.relative_move_group(movement_vector)
            # self.wait_until_still()

            # # the proceeding section is for data collection for focusing and detection models.
            movement_vector = np.array([movement * (np.random.rand() - 0.5), movement * (np.random.rand() - 0.5), 0], dtype=float)
            movement_z_vector = np.array([0,0,movement/5], dtype=float)
            movement_sequence = [
                movement_vector,
                movement_z_vector,
                -movement_z_vector,
                -movement_z_vector,
                movement_z_vector,
                -movement_vector,
            ]

            for step_vector in movement_sequence:
                if use_velocity_mode:
                    self._velocity_move_by_displacement(step_vector, requested_speed)
                else:
                    self.relative_move(step_vector)
                    self.wait_until_still()
        finally:
            if (not use_velocity_mode) and (orig is not None):
                self.set_max_speed(orig)
            self.info(f"Reset pipette speed to {self.get_max_speed()} um/s after random movement.")
            self.info("Finished random pipette movement.")


    def move_pipette_random(self, movement=500):
        '''
        Moves pipette randomly in xyz. This is used for testing find_pipette.
        '''
        movement_vector = np.array([movement * (np.random.rand() - 0.5), movement * (np.random.rand() - 0.5), (movement/5) * (np.random.rand() - 0.5)])
        self.relative_move_group(movement_vector)
        self.wait_until_still()

    @staticmethod
    def _validate_z_calibration(model, matrix):
        """Validate a legacy model or the complete distance bundle before publishing it."""
        def header(record, version, kind):
            if (not isinstance(record, dict) or type(record.get('version')) is not int
                    or record['version'] != version or record.get('mode') != 'group'
                    or record.get('model_kind') != kind or record.get('accepted') is not True
                    or record.get('status') != 'validated'):
                raise ValueError('Unsupported or unaccepted pipette Z calibration')
            jxy = np.asarray(record.get('jxy'), dtype=float)
            if jxy.shape != (2, 2) or not np.isfinite(jxy).all():
                raise ValueError('Invalid pipette Z calibration jxy')
            if not np.allclose(jxy, matrix[:2, :2], rtol=1e-9, atol=1e-9):
                raise ValueError('Pipette Z calibration does not match the XY calibration')

        def single(record):
            header(record, 2, 'group_z_innovations')
            arrays = {}
            for name, shape in (('means_px', (2, 2)), ('gain', (2, 2)),
                                ('innovation_covariance_px2', (2, 2)),
                                ('training_end_state', (2,))):
                value = np.asarray(record.get(name), dtype=float)
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError('Invalid pipette Z calibration ' + name)
                arrays[name] = value
            rho = record.get('rho')
            if (isinstance(rho, (bool, np.bool_))
                    or not isinstance(rho, (int, float, np.integer, np.floating))
                    or not math.isfinite(rho) or abs(rho) >= 1):
                raise ValueError('Invalid pipette Z calibration rho')
            gain = arrays['gain']
            if not np.allclose(gain, np.diag(np.diag(gain)), rtol=0, atol=1e-12):
                raise ValueError('Pipette Z calibration gain must be diagonal')
            if np.max(np.abs(np.linalg.eigvals(rho * np.eye(2) - gain))) >= 1:
                raise ValueError('Unstable pipette Z calibration')
            covariance = arrays['innovation_covariance_px2']
            if not np.allclose(covariance, covariance.T, rtol=1e-9, atol=1e-12):
                raise ValueError('Pipette Z innovation covariance must be symmetric')
            try:
                np.linalg.cholesky(covariance)
            except np.linalg.LinAlgError as error:
                raise ValueError('Pipette Z innovation covariance must be positive definite') from error

        if isinstance(model, dict) and model.get('version') == 3:
            header(model, 3, 'group_z_innovations_by_distance')
            children = model.get('models_by_distance')
            legacy_steps = {'100', '200', '300'}
            full_steps = legacy_steps | {'400', '500', '600'}
            if not isinstance(children, dict) or set(children) not in (legacy_steps, full_steps):
                raise ValueError('Pipette Z calibration requires a complete three- or six-distance bundle')
            if 'requested_steps_um' in model:
                requested = np.asarray(model['requested_steps_um'], dtype=float)
                if (requested.ndim != 1 or not np.isfinite(requested).all()
                        or requested.size != len(children) or np.unique(requested).size != requested.size
                        or set(requested.tolist()) != {float(step) for step in children}):
                    raise ValueError('Pipette Z calibration distance manifest does not match its models')
            for step, child in children.items():
                single(child)
                if 'requested_step_um' in child:
                    distance = child['requested_step_um']
                    if (isinstance(distance, (bool, np.bool_))
                            or not isinstance(distance, (int, float, np.integer, np.floating))
                            or not math.isfinite(distance) or distance != float(step)):
                        raise ValueError('Pipette Z calibration child distance does not match its key')
        else:
            single(model)

    def save_configuration(self):
        '''
        Outputs configuration in a dictionary.
        '''
        import copy

        config = {'up_direction' : self.up_direction,
                  'M' : self.M,
                  'r0' : self.r0}
        if self.M.shape == (3, 3):
            model = getattr(self, 'z_calibration', None)
            compatible = False
            if model is not None:
                try:
                    self._validate_z_calibration(model, self.M)
                    compatible = True
                except (TypeError, ValueError, np.linalg.LinAlgError):
                    pass
            config['z_calibration'] = copy.deepcopy(model) if compatible else None
        return config
    

    def load_configuration(self, config):
        '''
        Loads configuration from dictionary config.
        Missing Z metadata clears a previous model; validate before updating state.
        '''
        import copy

        matrix = np.asarray(config.get('M', self.M), dtype=float)
        model = None
        if matrix.shape == (3, 3):
            model = copy.deepcopy(config.get('z_calibration'))
            if model is not None:
                self._validate_z_calibration(model, matrix)
        inverse = pinv(matrix)
        self.M = matrix
        self.Minv = inverse
        self.z_calibration = model
        self.r0 = np.zeros(self.M.shape[0])
        self.r0_inv = np.zeros(self.M.shape[0])
        if self.M.shape[0] == 3:
            self.must_be_recalibrated = True #the pipette offsets need to be recalibrated upon reboot.
        self.calibrated = True

class CalibratedStage(CalibratedUnit):
    '''
    A horizontal stage calibrated to a fixed reference coordinate system.
    The optional stage refers to a platform on which the unit is mounted, which can
    be None.
    The stage is assumed to be parallel to the focal plane (no autofocus needed)

    Parameters
    ----------
    unit : ManipulatorUnit for this stage
    stage : CalibratedUnit for a stage on which this stage might be mounted
    microscope : ManipulatorUnit for the microscope (single axis)
    camera : a camera, ie, object with a ``snap()`` method (optional, for visual calibration)
    '''
    def __init__(self, unit, stage=None, microscope=None, camera=None,
                 config=None):
        CalibratedUnit.__init__(self, unit, stage, microscope, camera,
                                config=config)
        self.saved_state_question = 'Move stage back to initial position?'

        self.focusHelper = FocusHelper(microscope, camera)
        self.stageCalHelper = StageCalHelper(unit, camera, self.config.frame_lag)
        self.stageScanHelper = StageScanHelper(camera, config=self.config)
        self.cellDetectHelper = CellDetectHelper(camera)
        self.cellTrackHelper = None
        if self.config.use_ai_features:
            from .helpers.CellTrackHelper import CellTrackHelper
            self.cellTrackHelper = CellTrackHelper(self, camera)
        self.pipette_cal_position = np.zeros(2)
        self.unit = unit

        # It should be an XY stage, ie, two axes
        if len(self.axes) != 2:
            raise CalibrationError('The unit should have exactly two axes for horizontal calibration.')

    def detect_cells(self):
        return self.cellDetectHelper.detect_cells()

    def _ensure_cell_track_helper(self):
        if not self.config.use_ai_features:
            raise NotImplementedError(
                "Cell tracking is disabled. Set calibration.use_ai_features to true before use."
            )
        if self.cellTrackHelper is None:
            from .helpers.CellTrackHelper import CellTrackHelper
            self.cellTrackHelper = CellTrackHelper(self, self.camera)
        return self.cellTrackHelper

    def reference_position(self):
        '''Returns the offset (in pixels) of the stage compared to where it was when calibrated
        '''
        #get delta in um
        posDelta = self.unit.position()

        #convert to pixels
        posDelta = dot(self.M, posDelta) + self.r0

        #just get x and y (only concerned with pixels)
        posDelta = posDelta[:2]

        #append 0 for z
        posDelta = np.append(posDelta, 0)
        # self.debug(f'DEBUG: stage reference position: {posDelta}')

        return posDelta

    def safe_move(self, r):
        '''
        Moves the device to position x (an XYZ vector) in a way that minimizes
        interaction with tissue.

        Parameters
        ----------
        r : target position in um, an (X,Y,Z) vector
        yolo_correction : if True, corrects the pipette position using YOLO object detection
        '''
        if not self.calibrated:
            raise CalibrationError
        if self.must_be_recalibrated:
            raise CalibrationError('Pipette offsets must be recalibrated')

        # r from pyQt has origin at the center of the image, move origin to the top left corner (as expected by calibration)
        r = np.array(r)
        r = r + np.array([self.camera.width // 2, self.camera.height // 2, 0])
        self.abort_if_requested()
        self.reference_move(r) # Or relative move in manipulator coordinates, first axis (faster)

    def focus(self):
        ''' focus the stage on object of interest using the microscope
        '''
        self.debug('Focusing stage')
        self.abort_if_requested()
        self.focusHelper.autofocus(dist=self.config.autofocus_dist)

    def reference_relative_move(self, pos_pix):
        '''
        Moves the unit by vector r in reference camera system, without moving the stage.

        Parameters
        ----------
        pos_pix : position in pixels
        '''
        if not self.calibrated:
            raise CalibrationError
        if self.must_be_recalibrated:
            raise CalibrationError('Pipette offsets must be recalibrated')

        self.abort_if_requested()
        pos_microns = dot(self.Minv, pos_pix)
        self.relative_move(pos_microns)

    @property
    def is_collecting_scan_corners(self):
        return self.stageScanHelper.is_collecting

    @property
    def scan_corner_positions(self):
        return self.stageScanHelper.corner_positions

    def start_selecting_scan_corners(self):
        self.stageScanHelper.start_corner_collection(self)

    def reset_scan_corners(self):
        self.stageScanHelper.reset()

    def store_scan_corner(self, click_position):
        return self.stageScanHelper.record_corner_from_click(
            self, self.microscope, click_position
        )

    def move_to_scan_start(self, speed=None):
        self.stageScanHelper.move_to_scan_start(self, speed=speed)

    def scan_area(self, speed=None):
        self.stageScanHelper.scan_area(self, speed=speed)

    def calibrate(self):
        '''
        Automatic calibration for a horizontal XY stage

        '''
        if not self.stage.calibrated:
            self.stage.calibrate()

        self.info('Preparing stage calibration')
        # self.info("auto focusing microscope...")
        # self.focusHelper.autofocus(dist=self.config.autofocus_dist)
        # self.info("Finished focusing.")

        # use LK optical flow to determine transformation matrix
        mat = self.stageCalHelper.calibrate(dist=self.config.stage_diag_move)
        mat = np.append(mat, np.array([[0,0,1]]), axis=0)
        mat_inv = pinv(mat)

        # store r0 and r0_inv
        self.r0 = mat[0:2, 2] #um -> pixels offset
        self.r0_inv = mat_inv[0:2, 2] #pixels -> um offset

        #for M and Minv, we only want the upper 2x2 matrix (b/c assumption that z axis is equivilant), the rest of the matrix is just the identity
        self.M = mat[0:2, 0:2]
        self.Minv = mat_inv[0:2, 0:2]
        self.calibrated = True
        self.must_be_recalibrated = False

        self.info('Stage calibration done')

    def mosaic(self, width = None, height = None):
        '''
        Takes a photo mosaic. Current position corresponds to
        the top left corner of the collated image.
        Stops when the unit's position is out of range, unless
        width and height are specified.

        Parameters
        ----------
        width : total width in pixel (optional)
        height : total height in pixel (optional)

        Returns
        -------
        A large image of the mosaic.
        '''
        u0=self.position()
        if width == None:
            width = self.camera.width * 4
        if height == None:
            height = self.camera.height * 4

        dx, dy = self.camera.width, self.camera.height
        # Number of moves in each direction
        nx = 1+int(width/dx)
        ny = 1+int(height/dy)
        # Big image
        big_image = zeros((ny*dy,nx*dx))

        column = 0
        xdirection = 1 # moving direction along x axis

        try:
            for row in range(ny):
                img, _ = self.camera.snap()
                big_image[row*dy:(row+1)*dy, column*dx:(column+1)*dx] = img
                for _ in range(1,nx):
                    column+=xdirection
                    self.reference_relative_move([-dx*xdirection,0,0]) # sign: it's a compensatory move
                    self.wait_until_still()
                    self.sleep(0.1)
                    img, _ = self.camera.snap()
                    big_image[row * dy:(row + 1) * dy, column * dx:(column + 1) * dx] = img
                if row<ny-1:
                    xdirection = -xdirection
                    self.reference_relative_move([0,-dy,0])
                    self.wait_until_still()
        finally: # move back to initial position
            self.absolute_move(u0)

        cv2.imwrite('mosaic.png', big_image)

        return big_image
    

    def center_on_cell(self, cell, check_same_cell=False, use_centroid = True, max_error_px=None, reference_point=None, detected_px=None):
        """
        Find the cell centroid in pixel space and nudge the stage so the centroid
        is centred in the camera view.
        max_error_px limits each axis; None retains the camera-width / 20 limit.

        Returns `self.wait_until_still` (callable) so the GUI's `execute([...])`
        pipeline keeps working.
        
        """
        _cell_coords, reference_image, _position = cell

        # capture new image
        _, _, _, image = self.camera.raw_frame_queue[0]
        # compute expected cell location in the current camera frame (stage bookkeeping)
        stage_ref_px = np.asarray(self.reference_position(), dtype=np.float32)
        queued_ref_px = np.asarray(_cell_coords, dtype=np.float32)
        expected_px = stage_ref_px[:2] - queued_ref_px[:2]

        # reference thumbnails are cropped around the cell, so default to the crop centre
        ref_h, ref_w = reference_image.shape[:2]
        template_prompt = (np.array([ref_w / 2.0, ref_h / 2.0], dtype=np.float32)
                           if reference_point is None else np.asarray(reference_point, dtype=np.float32))

        self.info(f"Centering on cell at approx. {expected_px} px")
        if detected_px is None:
            cell_track_helper = self._ensure_cell_track_helper()
            centroid = cell_track_helper.find_centroid(
                reference_image, image, use_centroid=use_centroid,
                prompt_point=template_prompt, expected_point=expected_px)
        else:
            centroid = np.asarray(detected_px, dtype=float)
            if centroid.shape != (2,) or not np.isfinite(centroid).all():
                raise ValueError("Supplied cell position must be finite XY coordinates.")
        if centroid is None:
            return self.wait_until_still        # keep call chain consistent

        n_axes = self.Minv.shape[1]             # 2 for XY stage, 3 for XYZ
        centroid   = centroid[:n_axes]
        desired_px = np.array([self.camera.width / 2,
                            self.camera.height / 2])[:n_axes]

        # ------------------------------------------------------------------
        # Pixel error -> stage move (same sign).
        # (Stage motion and image motion are opposite, so this cancels the error.)
        # ------------------------------------------------------------------
        error_px = desired_px - centroid[:n_axes] 

        # -------- debug ---------------------------------------------------
        self.debug(f"centroid_px = {centroid}")
        self.debug(f"desired_px  = {desired_px}")
        self.info(f"error_px    = {error_px}")
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Clamp extreme pixel errors so we do not command huge stage jumps.
        if max_error_px is None:
            max_error_px = self.camera.width / 20
        if not np.isfinite(max_error_px) or max_error_px <= 0:
            raise ValueError("Cell centering limit must be finite and positive.")
        if np.any(np.abs(error_px) > max_error_px):
            self.warning(f"Clamping extreme pixel error (>{max_error_px} px).")
            error_px = np.clip(error_px, -max_error_px, max_error_px)

        # ------------------------------------------------------------------

        self.reference_relative_move(error_px)   # px → µm handled inside
        self.wait_until_still()

        # more debug (final position in µm and px)
        self.info(f"new stage µm position: {self.position()}")
        # self.info(f"new stage px offset  : {self.reference_position()}")


    def get_cell_position(
        self,
        cell,
        use_centroid=True,
        tracking_mode="fast",
        track_max_fast_jump_px=80.0,
    ):
        """
        Find the cell centroid in pixel space.

        Returns the centroid position (in pixels) as a numpy array.
        """
        cell_track_helper = self._ensure_cell_track_helper()

        _cell_coords, reference_image, _position = cell

        # capture new image
        _, _, _, image = self.camera.raw_frame_queue[0]
        # compute expected cell location in the current camera frame (stage bookkeeping)
        stage_ref_px = np.asarray(self.reference_position(), dtype=np.float32)
        queued_ref_px = np.asarray(_cell_coords, dtype=np.float32)
        expected_px = stage_ref_px[:2] - queued_ref_px[:2]

        # reference thumbnails are cropped around the cell, so default to the crop centre
        ref_h, ref_w = reference_image.shape[:2]
        template_prompt = np.array([ref_w / 2.0, ref_h / 2.0], dtype=np.float32)

        self.info(f"Getting position of cell at approx. {expected_px} px")
        centroid = cell_track_helper.track_cell(
            cell,
            image,
            use_centroid=use_centroid,
            prompt_point=template_prompt,
            expected_point=expected_px,
            mode=tracking_mode,
            max_fast_jump_px=track_max_fast_jump_px,
        )
        if centroid is None:
            return None, None

        n_axes = self.Minv.shape[1]
        centroid = centroid[:n_axes].astype(np.float32, copy=False)
        desired_px = np.array(
            [self.camera.width / 2.0, self.camera.height / 2.0],
            dtype=np.float32,
        )[:n_axes]
        error_px = desired_px - centroid

        self.debug(f"centroid_px = {centroid}")
        self.debug(f"desired_px  = {desired_px}")
        self.info(f"error_px    = {error_px}")


        return centroid, error_px
 


class FixedStage(CalibratedUnit):
    '''
    A stage that cannot move. This is used to simplify the code.
    '''
    def __init__(self):
        self.stage = None
        self.microscope = None
        self.r = array([0.,0.,0.]) # position in reference system
        self.u = array([0.,0.]) # position in stage system
        self.calibrated = True

    def position(self):
        return self.u

    def reference_position(self):
        return self.r

    def reference_move(self, r):
        # The fixed stage cannot move: maybe raise an error?
        pass

    def absolute_move(self, x, axis = None):
        pass
