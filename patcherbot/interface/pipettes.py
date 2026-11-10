# coding=utf-8
import pickle
import os
import math
from numbers import Real

import numpy as np
from datetime import datetime

from patcherbot.interface import TaskInterface, command, blocking_command
from patcherbot.devices.manipulator.calibratedunit import CalibratedUnit, CalibratedStage
from patcherbot.configs.CalibrationConfig import CalibrationConfig
from patcherbot.utils.sessionRegistration import SessionRegistration, calibration_identity, validate_coordinates_bundle
from patcherbot.devices.cellsorter import CalibratedCellSorter
import time

from patcherbot.devices.manipulator.microscope import Microscope
from patcherbot.devices.manipulator.helpers.Observer import Observer

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
        self.session_registration = SessionRegistration()
        self.calibrated_unit.session_registration = self.session_registration
        self._loaded_coordinate_names = set()
        self.session_registration.add_listener(self._publish_registered_coordinates)
        self.calibrated_cellsorter = CalibratedCellSorter(cellsorterManip, cellsorterController, self.calibrated_stage, microscope, camera)
        self.observer = Observer(self)
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

    def _coordinate_calibration_identity(self):
        return calibration_identity(self.calibrated_unit.save_configuration(),
                                    self.calibrated_stage.save_configuration())

    def refresh_coordinate_calibration(self, clear_tip=False):
        identity = self._coordinate_calibration_identity()
        previous = self.session_registration.export_coordinates("pipette").get("calibration_identity")
        if clear_tip and previous != identity:
            self.calibrated_unit._tip_reference = None
            self.calibrated_unit._trusted_tip_reference = None
            self._loaded_display_anchor = None
        self.session_registration.update_calibration("pipette", identity)

    def _measured_home_anchor(self, home):
        trusted = getattr(self.calibrated_unit, "_trusted_tip_reference", None)
        if trusted is None:
            saved = getattr(self, "_loaded_display_anchor", None)
            return saved if saved is not None and np.array_equal(saved["home"], home) else None
        tip, pipette_at_tip, stage_at_tip = trusted
        tip_xy = (np.asarray(tip, dtype=float)
                  + (self.calibrated_unit.M @ (home[:3] - pipette_at_tip))[:2]
                  + self.calibrated_stage.M @ (home[3:5] - stage_at_tip))
        if tip_xy.shape != (2,) or not np.isfinite(tip_xy).all():
            raise ValueError("Measured home tip must contain finite image XY")
        return dict(version=1, objective=40, home=home.tolist(),
                    tip_xy=tip_xy.tolist(), source="measured_tip")

    def record_coordinate_positions(self, positions):
        """Record raw values even when optional map metadata is unavailable."""
        self._store_raw_coordinates(positions)
        self._loaded_coordinate_names.difference_update(positions)
        try:
            home = positions.get("home")
            display = (self._measured_home_anchor(np.asarray(home, dtype=float))
                       if home is not None else getattr(self, "_loaded_display_anchor", None))
            self._loaded_display_anchor = display
            self.session_registration.record_positions(
                "pipette", positions, calibration_identity=self._coordinate_calibration_identity(),
                display_anchor=display)
        except (ValueError, TypeError, KeyError) as error:
            self._loaded_display_anchor = None
            self.session_registration.clear_coordinates("pipette")
            self.calibrated_unit.global_position_helper.clear_pipette_home_anchor()
            self.info(f"Raw coordinates stored without map reference: {error}")

    def _store_raw_coordinates(self, positions):
        """Restore raw calibration coordinates independently of map registration."""
        validate_coordinates_bundle(dict(version=1, positions=positions))
        for name in ("home", "safe"):
            if name in positions:
                value = positions[name]
                value = None if value is None else np.asarray(value, dtype=float)
                setattr(self, name + "_position", None if value is None else value[:3].copy())
                setattr(self, name + "_stage_position", None if value is None else value[3:].copy())
        for name, attr in (("bath", "cleaning_bath_position"), ("rinsing_bath", "rinsing_bath_position")):
            if name in positions:
                value = positions[name]
                setattr(self, attr, None if value is None else np.asarray(value, dtype=float).copy())
        for name in ("floor_Z", "stage_floor"):
            if name in positions:
                setattr(self.microscope, name, positions[name])

    def _publish_registered_coordinates(self):
        """Update the map anchor only; registration never changes raw controls."""
        bundle = self.session_registration.export_coordinates("pipette") or {}
        helper = self.calibrated_unit.global_position_helper
        display = bundle.get("display_anchor")
        if (self.home_position is None or display is None
                or not self.session_registration.positions_ready("pipette", "home")):
            helper.clear_pipette_home_anchor()
        else:
            try:
                helper.restore_pipette_home_anchor(
                    self.home_position, self.home_stage_position, tip_xy=display["tip_xy"],
                    safe_pipette_xyz=self.safe_position, cleaning_bath_xyz=self.cleaning_bath_position)
            except (ValueError, RuntimeError, OSError, KeyError) as error:
                helper.clear_pipette_home_anchor()
                self.info(f"Raw coordinates retained; pipette map display unavailable: {error}")

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
        self.refresh_coordinate_calibration(clear_tip=True)

    @command(category='Manipulators',
             description='Move pipette in x direction by {:.0f}μm',
             default_arg=10)
    def move_pipette_x(self, distance):
        self.calibrated_unit.relative_move(distance, axis=0)


    @command(category='Manipulators',
                description='Write current calibration to file')
    def write_calibration(self):
        from copy import deepcopy
        if not self.calibrated_stage.calibrated:
            raise RuntimeError('Stage not calibrated')
        if not self.calibrated_unit.calibrated:
            raise RuntimeError('Manipulator not calibrated')

        floor_z = self.microscope.floor_Z
        if floor_z is not None:
            if isinstance(floor_z, bool) or not isinstance(floor_z, Real) or not math.isfinite(floor_z):
                raise ValueError('Cell floor must be a finite coordinate or unset')
            floor_z = float(floor_z)
        stage_floor = getattr(self.microscope, 'stage_floor', None)
        if stage_floor is not None:
            if isinstance(stage_floor, bool) or not isinstance(stage_floor, Real) or not math.isfinite(stage_floor):
                raise ValueError('Stage floor must be a finite coordinate or unset')
            stage_floor = float(stage_floor)
        
        if (self.home_position is None or self.home_stage_position is None
                or self.safe_position is None or self.safe_stage_position is None):
            raise RuntimeError("Home and safe positions must be recorded before saving")
        self.home = np.r_[self.home_position, self.home_stage_position].astype(float)
        self.safe = np.r_[self.safe_position, self.safe_stage_position].astype(float)
        bath = None if self.cleaning_bath_position is None else np.asarray(self.cleaning_bath_position).tolist()
        positions = dict(home=self.home.tolist(), safe=self.safe.tolist(), bath=bath,
                         floor_Z=floor_z, stage_floor=stage_floor)
        validate_coordinates_bundle(dict(version=1, positions=positions))
        try:
            display = self._measured_home_anchor(self.home)
            bundle = self.session_registration.export_coordinates("pipette") or {}
            bundle['positions'] = positions
            bundle['display_anchor'] = display
            validate_coordinates_bundle(bundle)
        except (ValueError, TypeError, KeyError) as error:
            self.info(f"Saving raw calibration without map reference: {error}")
            display = None
            bundle = dict(version=1, positions=positions, display_anchor=None)

        try:
            # Build the complete file path for 'calibration.pickle'
            file_path = os.path.join(self.folder_path, 'calibration.pickle')
            if not os.path.exists(self.folder_path):
                os.makedirs(self.folder_path, exist_ok=True)
                self.folder_created = True

            
            calibration = {
                'manip': self.calibrated_unit.save_configuration(),
                'stage': self.calibrated_stage.save_configuration(),
                'home': self.home,
                'safe': self.safe,
                'bath': bath,
                'floor_Z': floor_z,  # Same units as set_floor(); no hardware movement.
                'stage_floor': stage_floor,
                'display_anchor': display,
                'coordinate_reference': bundle,
                'saved_at': datetime.now().astimezone().isoformat(timespec='seconds'),
            }
            snapshot = deepcopy(calibration)
            import tempfile
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode='wb', dir=self.folder_path,
                                                 prefix='calibration.', suffix='.tmp', delete=False) as f:
                    temporary = f.name
                    pickle.dump(calibration, f)
                os.replace(temporary, file_path)
            finally:
                if temporary is not None and os.path.exists(temporary):
                    os.unlink(temporary)
            self._calibration_snapshot = snapshot
            self.info('Calibration written to file')
        except Exception as e:
            self.error(f'Error writing calibration to file: {e}')
            raise

    @command(category='Manipulators',
                description='read most recent calibration from file')
    def read_calibration(self, config_filename='calibration.pickle'):
        from copy import deepcopy
        '''
        Read calibration from file.
        '''
        if os.path.isfile(config_filename):
            with open(config_filename, 'rb') as f:
                cal = pickle.load(f)
                stage_floor = cal.get('stage_floor')
                if stage_floor is not None:
                    if isinstance(stage_floor, bool) or not isinstance(stage_floor, Real) or not math.isfinite(stage_floor):
                        raise ValueError('Saved stage floor must be a finite coordinate or unset')
                    stage_floor = float(stage_floor)
                floor_z = cal.get('floor_Z')
                if floor_z is not None:
                    if isinstance(floor_z, bool) or not isinstance(floor_z, Real) or not math.isfinite(floor_z):
                        raise ValueError('Saved cell floor must be a finite coordinate or unset')
                    floor_z = float(floor_z)
                home = np.asarray(cal['home'], dtype=float).reshape(-1)
                safe = np.asarray(cal['safe'], dtype=float).reshape(-1)
                if home.shape != (6,) or safe.shape != (6,) or not np.isfinite(np.r_[home, safe]).all():
                    raise ValueError('Saved home and safe anchors must each contain six finite coordinates')
                if floor_z is None:
                    # Home contains pipette XYZ followed by stage XYZ, all in micrometres.
                    delta = self.calibration_config.home_position_delta_um
                    if isinstance(delta, bool) or not isinstance(delta, Real) or not math.isfinite(delta):
                        raise ValueError('Home position offset must be finite')
                    floor_z = float(home[5]) - float(delta)
                    if not math.isfinite(floor_z):
                        raise ValueError('Inferred cell floor must be finite')
                # Validate calibrated formats, not the stage's temporary 3x2 startup matrix.
                for key, expected_shape in (('manip', (3, 3)), ('stage', (2, 2))):
                    matrix = np.asarray(cal[key]['M'], dtype=float)
                    if matrix.shape != expected_shape or not np.isfinite(matrix).all():
                        raise ValueError(
                            f'Saved {key} calibration matrix must be finite with shape '
                            f'{expected_shape}; got {matrix.shape}')
                    cal[key]['M'] = matrix
                bath = cal.get('bath')
                if bath is not None:
                    bath = np.asarray(bath, dtype=float)
                    if bath.shape != (3,) or not np.isfinite(bath).all():
                        raise ValueError('Saved bath position must contain three finite coordinates')
                raw_positions = dict(home=home.tolist(), safe=safe.tolist(),
                                     bath=None if bath is None else bath.tolist(),
                                     floor_Z=floor_z, stage_floor=stage_floor)
                # Map metadata is optional and cannot prevent loading valid raw calibration.
                display = cal.get('display_anchor')
                try:
                    if display is not None:
                        validate_coordinates_bundle(dict(version=1, positions=raw_positions,
                                                         display_anchor=display))
                    bundle = cal.get("coordinate_reference") or dict(
                        version=1, binding=None, positions=raw_positions, display_anchor=display)
                    validate_coordinates_bundle(bundle)
                    if any(bundle["positions"].get(name) != value for name, value in raw_positions.items()):
                        raise ValueError("Saved map reference does not match raw calibration positions")
                except (ValueError, TypeError, KeyError) as error:
                    self.info(f"Raw calibration loaded without map reference: {error}")
                    display = None
                    bundle = dict(version=1, binding=None, positions=raw_positions, display_anchor=None)
                self.calibrated_unit.load_configuration(cal['manip'])
                self.calibrated_stage.load_configuration(cal['stage'])
                self.calibrated_unit._tip_reference = None
                self.calibrated_unit._trusted_tip_reference = None
                self._loaded_display_anchor = deepcopy(display) if display is not None else None
                self._loaded_coordinate_names = set(raw_positions)
                self._store_raw_coordinates(raw_positions)
                self.session_registration.stage_coordinates(
                    "pipette", bundle,
                    calibration_identity=calibration_identity(cal['manip'], cal['stage']))
                self._publish_registered_coordinates()

                source_saved_at = cal.get('saved_at')
                try:
                    if not isinstance(source_saved_at, str):
                        source_saved_at = None
                    else:
                        datetime.fromisoformat(source_saved_at)
                except ValueError:
                    source_saved_at = None
                source_file = os.path.abspath(config_filename)
                self.loaded_calibration_reference = {
                    'source_file': source_file,
                    'source_saved_at': source_saved_at,
                    'source_session': os.path.basename(os.path.dirname(source_file)),
                    'reloaded_at': datetime.now().astimezone().isoformat(timespec='seconds'),
                }
                self._calibration_snapshot = deepcopy(cal)
                self.info(
                    f"Calibration reloaded from {source_file}; "
                    f"file saved at: {source_saved_at or 'unknown'}; "
                    f"source session folder: {self.loaded_calibration_reference['source_session']}; "
                    f"reloaded at: {self.loaded_calibration_reference['reloaded_at']}")

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
        self.record_coordinate_positions({'floor_Z': self.microscope.floor_Z})
        self.info(f'Cell plane position set to {self.microscope.floor_Z}')

    def _persist_stage_floor(self, stage_floor):
        """Update session metadata without re-saving reset motion calibration."""
        import tempfile
        from copy import deepcopy
        target = os.path.join(self.folder_path, 'calibration.pickle')
        snapshot = getattr(self, '_calibration_snapshot', None)
        if snapshot is not None:
            calibration = deepcopy(snapshot)
        else:
            reference = getattr(self, 'loaded_calibration_reference', None) or {}
            source = reference.get('source_file') or target
            if not os.path.isfile(source):
                return False  # Initial setup: the normal calibration save includes it.
            with open(source, 'rb') as stream:
                calibration = pickle.load(stream)
        if not isinstance(calibration, dict):
            raise ValueError('Calibration file must contain a calibration dictionary')
        calibration['stage_floor'] = float(stage_floor)
        registration = getattr(self, 'session_registration', None)
        if registration is not None:
            calibration['coordinate_reference'] = registration.export_coordinates('pipette')
        calibration['saved_at'] = datetime.now().astimezone().isoformat(timespec='seconds')
        os.makedirs(self.folder_path, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='wb', dir=self.folder_path,
                                             prefix='calibration.', suffix='.tmp', delete=False) as stream:
                temporary = stream.name
                pickle.dump(calibration, stream)
            os.replace(temporary, target)
            self._calibration_snapshot = deepcopy(calibration)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)
        return True

    @command(category='Microscope',
             description='Set the position of the stage floor (basin)',
             success_message='Stage floor position stored')
    def set_stage_floor(self):
        z_scale = self.calibrated_unit.config.microscope_units_per_um
        stage_floor = self.microscope.position() / z_scale
        if isinstance(stage_floor, bool) or not isinstance(stage_floor, Real) or not math.isfinite(stage_floor):
            raise ValueError('Stage floor must be a finite coordinate')
        previous = self.session_registration.export_coordinates("pipette")
        previous_loaded = self._loaded_coordinate_names.copy()
        previous_floor = self.microscope.stage_floor
        try:
            self.record_coordinate_positions({'stage_floor': float(stage_floor)})
            persisted = self._persist_stage_floor(stage_floor)
        except Exception:
            self.session_registration.restore_coordinates("pipette", previous)
            self._loaded_coordinate_names = previous_loaded
            self.microscope.stage_floor = previous_floor
            raise
        self.microscope.stage_floor = float(stage_floor)
        state = 'saved to session calibration' if persisted else 'will save with the first calibration'
        self.info(f'Stage floor position set to {self.microscope.stage_floor}; {state}')

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
        try:
            self.execute([self.calibrated_stage.calibrate])
        finally:
            self.refresh_coordinate_calibration(clear_tip=True)

    @blocking_command(category='Manipulators',
                      description='Calibrate manipulator',
                      task_description='Calibrating manipulator')
    def calibrate_manipulator(self):
        try:
            self.execute([self.calibrated_unit.calibrate_pipette])
        finally:
            self.refresh_coordinate_calibration()

    @blocking_command(category='Manipulators',
                      description='Detect the pipette',
                      task_description='Detecting the pipette')
    def detect_pipette(self):
        self.execute(self.calibrated_unit.detect_pipette, argument=self.observer)
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

    @blocking_command(category='Microscope',
                      description='Go to the stage floor (basin)',
                      task_description='Go to the stage floor (basin)')
    def go_to_stage_floor(self):
        if self.microscope.stage_floor is None:
            raise RuntimeError("Stage floor must be set.")
        self.execute(self.microscope.move_to_stage_floor)
