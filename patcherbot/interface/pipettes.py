# coding=utf-8
import pickle
import os
from pathlib import Path

import numpy as np
from datetime import datetime

from patcherbot.interface import TaskInterface, command, blocking_command
from patcherbot.devices.manipulator.calibratedunit import CalibratedUnit, CalibratedStage
from patcherbot.devices.manipulator.CalibrationConfig import CalibrationConfig
from patcherbot.devices.cellsorter import CalibratedCellSorter
import time

from patcherbot.devices.manipulator.microscope import Microscope
from .cell_queue import CellQueueCoordinator


class PipetteInterface(TaskInterface):
    '''
    Controller for the stage, the microscope, a pipette, and the cell sorter.
    '''

    def __init__(self, stage, microscope: Microscope, camera, unit, cellsorterManip, cellsorterController,
                 config_filename='calibration.pickle', calibration_data=None, pipette_id=None):
        """
        Initialize the PipetteInterface with hardware components and calibration configuration.

        Args:
            stage (object): Stage device.
            microscope (Microscope): Microscope device.
            camera (object): Camera device.
            unit (object): Manipulator unit.
            cellsorterManip (object): Cell sorter manipulator.
            cellsorterController (object): Cell sorter controller.
            config_filename (str, optional): Calibration file name.
            calibration_data (dict, optional): Calibration configuration data.
        """
        super().__init__()
        self.pipette_id = str(pipette_id) if pipette_id is not None else "pipette_0"
        suffix = self.pipette_id.rsplit("_", 1)[-1]
        self.pipette_index = int(suffix) if suffix.isdigit() else 0
        self.microscope = microscope
        self.camera = camera
        # Create a common calibration configuration for all stages/manipulators
        self.calibration_config = CalibrationConfig(name='Calibration')
        self.calibration_config.pipette_id = self.pipette_id
        if calibration_data:
            cleaned = {k: v for k, v in calibration_data.items() if v is not None}
            self.calibration_config.from_dict(cleaned)
        self.microscope.config = self.calibration_config
        self.microscope.units_per_um = float(self.calibration_config.microscope_units_per_um)
        self.microscope.objective_lift_um = float(self.calibration_config.objective_lift_um)
        if hasattr(self.microscope, 'dev'):
            self.microscope.dev.objective_lift_um = float(self.calibration_config.objective_lift_um)
        self.calibrated_stage = CalibratedStage(stage, None, microscope, camera,
                                                config=self.calibration_config)
        self.shared_calibrated_stages = [self.calibrated_stage]
        self.calibrated_unit = CalibratedUnit(unit,
                                                self.calibrated_stage,
                                                microscope,
                                                camera,
                                                config=self.calibration_config)
        self.calibrated_unit.pipette_id = self.pipette_id
        self.calibrated_unit.pipette_index = self.pipette_index
        self.calibrated_stage.pipette_id = self.pipette_id
        self.calibrated_cellsorter = CalibratedCellSorter(cellsorterManip, cellsorterController, self.calibrated_stage, microscope, camera)
        self.time_truth = datetime.now()
        self.folder_path = (
            "experiments/Data/calibration_data/"
            + self.time_truth.strftime("%Y_%m_%d-%H_%M")
            + f"/{self.pipette_id}/"
        )
        self.folder_created = False  # Flag to track folder creation

   
        # if config_filename is not None:
        #     #read calibration from file
        #     if os.path.isfile(config_filename):
        #         with open(config_filename, 'rb') as f:
        #             cal = pickle.load(f)
        #             self.calibrated_unit.load_configuration(cal['manip'])
        #             self.calibrated_stage.load_configuration(cal['stage'])

        #             print('Loaded calibration from file!')
        #             print('Manipulator calibration:')
        #             print(cal['manip'])
        #             print('Stage calibration:')
        #             print(cal['stage'])
        #     else:
        #         pass
        #         print('No calibration file found, need to calibrate before usage!')


        self.cleaning_bath_position = None
        self.contact_position = None
        self.rinsing_bath_position = None
        self.paramecium_tank_position = None
        self.timer_t0 = time.time()
        self.pos_before_raise = None
        self.home_position = None
        self.home_stage_position = None
        self.safe_position = None
        self.safe_stage_position = None
        self.tare_pipette = np.array([None, None, None])
        self.tare_stage = np.array([None, None, None])

    def connect(self, main_gui):
        """
        Connect interface signals to the GUI (currently unused).

        Args:
            main_gui (object): Main GUI instance.
        
        Note: This method is currently unimplemented.
        """
        pass #TODO: unused?

    @blocking_command(category='Manipulators',
             description='Move to a repeatable position in the z axis (eliminate backlash)',
             task_description='Moving to a repeatable position in the z axis',
             default_arg=10)
    def fix_backlash(self, none):
        """
        Move the microscope to eliminate backlash and ensure repeatable positioning.

        Args:
            none (object): Unused parameter.
        """
        self.execute(self.microscope.fix_backlash)

    @command(category='Manipulators',
             description='Record a calibration point at the current position',
             default_arg=10)
    def record_cal_point(self, none):
        """
        Record the current position as a calibration point.

        Args:
            none (object): Unused parameter.
        """
        self.calibrated_unit.record_cal_point()
    
    @command(category='Manipulators',
             description='Finish calibration',
             default_arg=10)
    def finish_calibration(self, none):
        """
        Finalize the calibration process for the manipulator.

        Args:
            none (object): Unused parameter.
        """
        self.calibrated_unit.finish_calibration()

    @command(category='Manipulators',
             description='Move pipette in x direction by {:.0f}μm',
             default_arg=10)
    def move_pipette_x(self, distance):
        """
        Move the pipette along the x-axis by a specified distance.

        Args:
            distance (float): Distance to move in micrometers.
        """
        self.calibrated_unit.relative_move(distance, axis=0)


    @command(category='Manipulators',
                description='Write current calibration to file')
    def write_calibration(self):
        """
        Save calibration data and key positions to a file.

        Raises:
            RuntimeError: If calibration or required positions are not set.
            Exception: If writing to file fails.
        """
        if not self.calibrated_stage.calibrated:
            raise RuntimeError('Stage not calibrated')
        if not self.calibrated_unit.calibrated:
            raise RuntimeError('Manipulator not calibrated')
        
        # concatenate home position and home stage position
        if self.home_position is None or self.home_stage_position is None:
            raise RuntimeError('Home position not set')   
        if self.safe_position is None or self.safe_stage_position is None:
            raise RuntimeError('Safe position not set')
        
        self.home_position = np.array(self.home_position)
        self.home_stage_position = np.array(self.home_stage_position)   
        self.safe_position = np.array(self.safe_position)
        self.safe_stage_position = np.array(self.safe_stage_position)
        self.home = np.concatenate((self.home_position, self.home_stage_position))
        self.safe = np.concatenate((self.safe_position, self.safe_stage_position))
    
        try:
            # Build the complete file path for 'calibration.pickle'
            file_path = os.path.join(self.folder_path, 'calibration.pickle')
            if not os.path.exists(self.folder_path):
                os.makedirs(self.folder_path, exist_ok=True)
                self.folder_created = True

            
            # Write calibration data to the file in the specified folder
            with open(file_path, 'wb') as f:
                pickle.dump({
                    'manip': self.calibrated_unit.save_configuration(),
                    'stage': self.calibrated_stage.save_configuration(),
                    'home': self.home,
                    'safe': self.safe,
                    'bath': self.cleaning_bath_position,
                }, f)
            self.info('Calibration written to file')
        except Exception as e:
            self.error(f'Error writing calibration to file: {e}')
            raise

    @command(category='Manipulators',
                description='read most recent calibration from file')
    def read_calibration(self, config_filename=None):
        '''
        Read calibration from file.

        Args:
            config_filename (str, optional): Path to calibration file.

        Raises:
            RuntimeError: If calibration file is not found.
        '''
        if config_filename is None:
            calibration_root = Path("experiments/Data/calibration_data")
            candidates = sorted(
                calibration_root.glob(f"*/{self.pipette_id}/calibration.pickle"),
                key=lambda path: path.stat().st_mtime,
            )
            if not candidates and self.pipette_index == 0:
                candidates = sorted(
                    calibration_root.glob("*/calibration.pickle"),
                    key=lambda path: path.stat().st_mtime,
                )
            config_filename = candidates[-1] if candidates else None
        if config_filename is not None and os.path.isfile(config_filename):
            with open(config_filename, 'rb') as f:
                cal = pickle.load(f)
                self.calibrated_unit.load_configuration(cal['manip'])
                self.calibrated_stage.load_configuration(cal['stage'])
                home = np.asarray(cal['home'], dtype=float)
                safe = np.asarray(cal['safe'], dtype=float)
                self.home_position = home[:3]
                self.home_stage_position = home[3:6]
                self.safe_position = safe[:3]
                self.safe_stage_position = safe[3:6]
                self.cleaning_bath_position = cal['bath']

                print('Loaded calibration from file!')
                print('Manipulator calibration:')
                print(cal['manip'])
                print('Stage calibration:')
                print(cal['stage'])
        else:
            raise RuntimeError('No calibration file found, need to calibrate before usage!')

    @command(category='Manipulators',
             description='write the current tared position to file',
             success_message='Tared position written to file')
    def write_tare(self):
        """
        Save the current tared positions of pipette and stage to a file.

        Raises:
            RuntimeError: If tare positions are not set.
            Exception: If writing to file fails.
        """
        if self.tare_pipette is None or self.tare_stage is None:
            raise RuntimeError('Tare position not set')
        try:
            # Build the complete file path for 'tare.pickle'
            file_path = os.path.join(self.folder_path, 'tare.pickle')
            if not os.path.exists(self.folder_path):
                os.makedirs(self.folder_path, exist_ok=True)
                self.folder_created = True

            # Write tare data to the file in the specified folder
            with open(file_path, 'wb') as f:
                pickle.dump({
                    'pipette': self.tare_pipette,
                    'stage': self.tare_stage,
                }, f)
            self.info('Tare position written to file')
        except Exception as e:
            self.error(f'Error writing tare position to file: {e}')
            raise
    
    @command(category='Manipulators',
                description='recalibrate manipulator offset while preserving matrix')
    def recalibrate_manipulator(self):
        """Recalibrate the manipulator offset while preserving calibration matrix."""
        self.calibrated_unit.recalibrate_pipette()
            
    @command(category='Manipulators',
             description='Move pipette in y direction by {:.0f}μm',
             default_arg=10)
    def move_pipette_y(self, distance):
        """
        Move the pipette along the y-axis by a specified distance.

        Args:
            distance (float): Distance to move in micrometers.
        """
        self.calibrated_unit.relative_move(distance, axis=1)

    @command(category='Manipulators',
             description='Move pipette in z direction by {:.0f}μm',
             default_arg=-1000)
    def move_pipette_z(self, distance):
        """
        Move the pipette along the z-axis by a specified distance.

        Args:
            distance (float): Distance to move in micrometers.
        """
        self.calibrated_unit.relative_move(distance, axis=2)

    @command(category='Manipulators',
             description='Move pipette in xyz direction by {:.0f}μm',
             default_arg=-500)
    def move_pipette_xyz(self, distance):
        """
        Move the pipette in 3D space based on pipette angle configuration.

        Args:
            distance (float): Movement magnitude in micrometers.
        """
        # currently utilized for automatic safe space saving
        direction = self.calibrated_unit.config.resolve_pipette_axis_direction(self.pipette_id)
        distance = distance * direction
        self.calibrated_unit.relative_move(distance)
        
    @command(category='Microscope',
             description='Move microscope by {:.0f}μm',
             default_arg=-500) 
    def move_microscope(self, distance):
        """
        Move the microscope along its axis.

        Args:
            distance (float): Distance to move in micrometers.
        """
        # self.info(f'Moving microscope by {distance}μm')
        self.microscope.relative_move(distance)

    @command(category='Microscope',
             description='Set the position of the floor (cover slip)',
             success_message='Cover slip position stored')
    def set_floor(self):
        """Store the current microscope position as the coverslip floor."""
        z_scale = self.calibrated_unit.config.microscope_units_per_um
        self.microscope.floor_Z = self.microscope.position() / z_scale
        self.info(f'Cell plane position set to {self.microscope.floor_Z}')

    @command(category='Stage',
             description='Move stage vertically by {:.0f}μm',
             default_arg=-50)
    def move_stage_vertical(self, distance):
        """
        Move the stage vertically.

        Args:
            distance (float): Distance to move in micrometers.
        """
        if distance < 0:
            print(f"distance is negative: {distance}")
        self.calibrated_stage.relative_move(distance, axis=1)

    @command(category='Stage',
             description='Move stage horizontally by {:.0f}μm',
             default_arg=10)
    def move_stage_horizontal(self, distance):
        """
        Move the stage horizontally.

        Args:
            distance (float): Distance to move in micrometers.
        """
        if distance < 0:
            print(f"distance is negative: {distance}")
        self.calibrated_stage.relative_move(distance, axis=0)

    @blocking_command(category='Stage',
                      description='Calibrate stage only',
                      task_description='Calibrating stage')
    def calibrate_stage(self):
        """Calibrate the shared stage and update each pipette's stage wrapper."""
        self.execute(
            self.calibrated_stage.calibrate_rig_wide,
            self.shared_calibrated_stages,
        )

    @blocking_command(category='Manipulators',
                      description='Calibrate manipulator',
                      task_description='Calibrating manipulator')
    def calibrate_manipulator(self):
        """Perform calibration of the manipulator."""
        self.execute([self.calibrated_unit.calibrate_pipette])
    @blocking_command(category='Manipulators',
                        description='Home the manipulator',
                        task_description='Homing the manipulator')
    def Home_manipulator(self):
        """Move the manipulator to its home position."""
        self.execute([self.calibrated_unit.home])
        
    @blocking_command(category='Manipulators',
                     description = 'Center the Pipette',
                      task_description='Centering the Pipette')
    def center_pipette(self):
        """Center the pipette in the field of view."""
        self.execute([self.calibrated_unit.center_pipette])
    @blocking_command(category='Manipulators',
                     description = 'direct the Pipette',
                      task_description='Directing the Pipette')
    def direct_pipette(self,desired_px):
        """
        Direct the pipette toward a specified pixel position.

        Args:
            desired_px (array-like): Target pixel coordinates.
        """
        self.execute([self.calibrated_unit.direct_pipette], argument= desired_px)
    @blocking_command(category='Manipulators and Stage',
                      description='Follow stage',
                        task_description='Following the stage')
    def follow_stage(self):
        """Move the pipette to follow stage movements."""
        self.execute([self.calibrated_unit.follow_stage])


    @blocking_command(category='Manipulators and Stage',
                      description='Move pipette randomly in xyz',
                        task_description='displacing pipette randomly in xyz...')
    def move_pipette_random_velocity(self):
        self.execute([self.calibrated_unit.move_pipette_random_velocity])

    @blocking_command(category='Manipulators and Stage',
                      description='Move pipette randomly in xyz',
                        task_description='displacing pipette randomly in xyz...')
    def move_pipette_random(self):
        """Move the pipette randomly in 3D space."""
        self.execute([self.calibrated_unit.move_pipette_random])



    @blocking_command(category='Stage',
                        description='focus the stage',
                        task_description='Focusing the stage')
    def focus_stage(self):
        """Autofocus the stage."""
        self.execute([self.calibrated_stage.focus])
        
    @blocking_command(category='Manipulators',
                      description='Focus the pipette',
                      task_description='Calibrating manipulator')
    def focus_pipette(self):
        """Autofocus the pipette."""
        self.execute([self.calibrated_unit.autofocus_pipette])

    @blocking_command(category='Manipulators',
                     description='Move pipette to position',
                     task_description='Moving to position with safe approach')
    def move_pipette(self, xy_position):
        """
        Move the pipette safely to a specified XY position.

        Args:
            xy_position (tuple): Target (x, y) position.
        """
        x, y = xy_position
        position = np.array([x, y, self.microscope.position()])
        self.debug('asking for safe move to {}'.format(position))
        self.execute(self.calibrated_unit.safe_move, argument=position)

    @blocking_command(category='Manipulators',
                    description='Raise the pipette high enough to insert the coverslip',
                    task_description='Raising the pipette high enough to insert the coverslip')
    def raise_pipette(self, raise_distance = 1000):
        """
        Raise the pipette to allow coverslip insertion.

        Args:
            raise_distance (float, optional): Height to raise (unused directly).

        Raises:
            RuntimeError: If pipette is already raised.
        """
        if self.pos_before_raise is None:
            self.pos_before_raise = self.calibrated_unit.dev.position()
            position = np.array([self.pos_before_raise[0], self.pos_before_raise[1], 0])
            self.execute(self.calibrated_unit.absolute_move, argument=position)
        else:
            raise RuntimeError('Pipette already raised')

    @blocking_command(category='Manipulators',
                description='Lower the pipette after inserting the coverslip',
                task_description='Lowering the pipette after inserting the coverslip')
    def lower_pipette(self):
        """
        Return the pipette to its previous position after raising.

        Raises:
            RuntimeError: If pipette was not previously raised.
        """
        if self.pos_before_raise is not None:
            self.execute(self.calibrated_unit.absolute_move, argument=self.pos_before_raise)
            self.pos_before_raise = None
        else:
            raise RuntimeError('Pipette not raised')
        
    @blocking_command(category='Cell Sorter',
                    description='calibrate the cell sorter',
                    task_description='Calibrating the cell sorter')
    def calibrate_cell_sorter(self):
        """Perform calibration of the cell sorter."""
        self.execute(self.calibrated_cellsorter.calibrate)

    def set_cell_sorter_led(self, enabled: bool, ring: int = 1):
        """
        Enable or disable the LED ring on the cell sorter.

        Args:
            enabled (bool): Whether to enable the LED.
            ring (int, optional): LED ring index.
        """
        self.calibrated_cellsorter.set_led_ring_enabled(enabled, ring)

    @blocking_command(category='Manipulators',
                     description='Move stage to position',
                     task_description='Moving stage to position')
    def move_stage(self, xy_position):
        """
        Move the stage to a specified position.

        Args:
            xy_position (tuple): Target (x, y) position.
        """
        x, y = xy_position
        position = np.array([x, y])
        self.debug('asking for reference move to {}'.format(position))
        self.execute(self.calibrated_stage.reference_relative_move, argument=-position) # compensatory move


    @blocking_command(category='Microscope',
                      description='Go to the floor (cover slip)',
                      task_description='Go to the floor (cover slip)')
    def go_to_floor(self):
        """
        Move the microscope to the stored coverslip floor position.

        Raises:
            RuntimeError: If floor position is not set.
        """
        if self.microscope.floor_Z is None:
            raise RuntimeError("Coverslip floor must be set.")
        self.execute(self.microscope.move_to_floor)
