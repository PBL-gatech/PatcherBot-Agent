# coding=utf-8
import pickle
import os

import numpy as np
from datetime import datetime

from patcherbot.interface import TaskInterface, command, blocking_command
from patcherbot.devices.manipulator.calibratedunit import CalibratedUnit, CalibratedStage
from patcherbot.configs.CalibrationConfig import CalibrationConfig
from patcherbot.devices.cellsorter import CalibratedCellSorter
import time

from patcherbot.devices.manipulator.microscope import Microscope

class PipetteInterface(TaskInterface):
    '''
    Controller for the stage, the microscope, a pipette, and the cell sorter.
    '''

    def __init__(
        self,
        stage,
        microscope: Microscope,
        camera,
        unit,
        cellsorterManip,
        cellsorterController,
        config_filename='calibration.pickle',
        calibration_data=None,
    ):
        super().__init__()
        self.microscope = microscope
        self.camera = camera
        self.display_positions = None
        # Create a common calibration configuration for all stages/manipulators
        self.calibration_config = CalibrationConfig(name='Calibration')
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
        self.calibrated_unit = CalibratedUnit(unit,
                                                self.calibrated_stage,
                                                microscope,
                                                camera,
                                                config=self.calibration_config)
        self.calibrated_cellsorter = CalibratedCellSorter(cellsorterManip, cellsorterController, self.calibrated_stage, microscope, camera)
        self.time_truth = datetime.now()
        self.folder_path = "experiments/Data/calibration_data/" + self.time_truth.strftime("%Y_%m_%d-%H_%M") + "/"
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
        pass #TODO: unused?

    @blocking_command(category='Manipulators',
             description='Move to a repeatable position in the z axis (eliminate backlash)',
             task_description='Moving to a repeatable position in the z axis',
             default_arg=10)
    def fix_backlash(self, none):
        self.execute(self.microscope.fix_backlash)

    @command(category='Manipulators',
             description='Record a calibration point at the current position',
             default_arg=10)
    def record_cal_point(self, none):
        self.calibrated_unit.record_cal_point()
    
    @command(category='Manipulators',
             description='Finish calibration',
             default_arg=10)
    def finish_calibration(self, none):
        self.calibrated_unit.finish_calibration()

    @command(category='Manipulators',
             description='Move pipette in x direction by {:.0f}μm',
             default_arg=10)
    def move_pipette_x(self, distance):
        self.calibrated_unit.relative_move(distance, axis=0)


    @command(category='Manipulators',
                description='Write current calibration to file')
    def write_calibration(self):
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
    def read_calibration(self, config_filename='calibration.pickle'):
        '''
        Read calibration from file.
        '''
        if os.path.isfile(config_filename):
            with open(config_filename, 'rb') as f:
                cal = pickle.load(f)
                home = np.asarray(cal['home'], dtype=float).reshape(-1)
                safe = np.asarray(cal['safe'], dtype=float).reshape(-1)
                if home.shape != (6,) or safe.shape != (6,) or not np.isfinite(np.r_[home, safe]).all():
                    raise ValueError('Saved home and safe anchors must each contain six finite coordinates')
                self.calibrated_unit.load_configuration(cal['manip'])
                self.calibrated_stage.load_configuration(cal['stage'])
                self.home_position = home[:3].copy()
                self.home_stage_position = home[3:].copy()
                self.safe_position = safe[:3].copy()
                self.safe_stage_position = safe[3:].copy()
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
        self.calibrated_unit.recalibrate_pipette()
            
    @command(category='Manipulators',
             description='Move pipette in y direction by {:.0f}μm',
             default_arg=10)
    def move_pipette_y(self, distance):
        self.calibrated_unit.relative_move(distance, axis=1)

    @command(category='Manipulators',
             description='Move pipette in z direction by {:.0f}μm',
             default_arg=-1000)
    def move_pipette_z(self, distance):
        self.calibrated_unit.relative_move(distance, axis=2)

    @command(category='Manipulators',
             description='Move pipette in xyz direction by {:.0f}μm',
             default_arg=-500)
    def move_pipette_xyz(self, distance):
        # currently utilized for automatic safe space saving
        angle = np.deg2rad(self.calibrated_unit.config.pipette_y_rotation)
        distance = np.array([distance*np.cos(angle),0,distance*np.sin(angle)])
        self.calibrated_unit.relative_move(distance)
        
    @command(category='Microscope',
             description='Move microscope by {:.0f}μm',
             default_arg=-1000)
    def move_microscope(self, distance):
        # self.info(f'Moving microscope by {distance}μm')
        self.microscope.relative_move(distance)

    @command(category='Microscope',
             description='Set the position of the floor (cover slip)',
             success_message='Cover slip position stored')
    def set_floor(self):
        z_scale = self.calibrated_unit.config.microscope_units_per_um
        self.microscope.floor_Z = self.microscope.position() / z_scale
        self.info(f'Cell plane position set to {self.microscope.floor_Z}')

    @command(category='Stage',
             description='Move stage vertically by {:.0f}μm',
             default_arg=-50)
    def move_stage_vertical(self, distance):
        if distance < 0:
            print(f"distance is negative: {distance}")
        self.calibrated_stage.relative_move(distance, axis=1)

    @command(category='Stage',
             description='Move stage horizontally by {:.0f}μm',
             default_arg=10)
    def move_stage_horizontal(self, distance):
        if distance < 0:
            print(f"distance is negative: {distance}")
        self.calibrated_stage.relative_move(distance, axis=0)

    @blocking_command(category='Stage',
                      description='Calibrate stage only',
                      task_description='Calibrating stage')
    def calibrate_stage(self):
        self.execute([self.calibrated_stage.calibrate])

    @blocking_command(category='Manipulators',
                      description='Calibrate manipulator',
                      task_description='Calibrating manipulator')
    def calibrate_manipulator(self):
        self.execute([self.calibrated_unit.calibrate_pipette])

    @blocking_command(category='Manipulators',
                      description='Detect the pipette',
                      task_description='Detecting the pipette')
    def detect_pipette(self):
        self.execute(self.calibrated_unit.detect_pipette)
    @blocking_command(category='Manipulators',
                        description='Home the manipulator',
                        task_description='Homing the manipulator')
    def Home_manipulator(self):
        self.execute([self.calibrated_unit.home])
        
    @blocking_command(category='Manipulators',
                     description = 'Center the Pipette',
                      task_description='Centering the Pipette')
    def center_pipette(self):
        self.execute([self.calibrated_unit.center_pipette])
    @blocking_command(category='Manipulators',
                     description = 'direct the Pipette',
                      task_description='Directing the Pipette')
    def direct_pipette(self,desired_px):
        self.execute([self.calibrated_unit.direct_pipette], argument= desired_px)
    @blocking_command(category='Manipulators and Stage',
                      description='Follow stage',
                        task_description='Following the stage')
    def follow_stage(self):
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
        self.execute([self.calibrated_unit.move_pipette_random])



    @blocking_command(category='Stage',
                        description='focus the stage',
                        task_description='Focusing the stage')
    def focus_stage(self):
        self.execute([self.calibrated_stage.focus])
        
    @blocking_command(category='Manipulators',
                      description='Focus the pipette',
                      task_description='Calibrating manipulator')
    def focus_pipette(self):
        self.execute([self.calibrated_unit.autofocus_pipette])

    @blocking_command(category='Manipulators',
                     description='Move pipette to position',
                     task_description='Moving to position with safe approach')
    def move_pipette(self, xy_position):
        x, y = xy_position
        position = np.array([x, y, self.microscope.position()])
        self.debug('asking for safe move to {}'.format(position))
        self.execute(self.calibrated_unit.safe_move, argument=position)

    @blocking_command(category='Manipulators',
                    description='Raise the pipette high enough to insert the coverslip',
                    task_description='Raising the pipette high enough to insert the coverslip')
    def raise_pipette(self, raise_distance = 1000):
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
        if self.pos_before_raise is not None:
            self.execute(self.calibrated_unit.absolute_move, argument=self.pos_before_raise)
            self.pos_before_raise = None
        else:
            raise RuntimeError('Pipette not raised')
        
    @blocking_command(category='Cell Sorter',
                    description='calibrate the cell sorter',
                    task_description='Calibrating the cell sorter')
    def calibrate_cell_sorter(self):
        self.execute(self.calibrated_cellsorter.calibrate)

    @blocking_command(category='Manipulators',
                     description='Move stage to position',
                     task_description='Moving stage to position')
    def move_stage(self, xy_position):
        x, y = xy_position
        position = np.array([x, y])
        self.debug('asking for reference move to {}'.format(position))
        self.execute(self.calibrated_stage.reference_relative_move, argument=-position) # compensatory move


    @blocking_command(category='Microscope',
                      description='Go to the floor (cover slip)',
                      task_description='Go to the floor (cover slip)')
    def go_to_floor(self):
        if self.microscope.floor_Z is None:
            raise RuntimeError("Coverslip floor must be set.")
        self.execute(self.microscope.move_to_floor)
