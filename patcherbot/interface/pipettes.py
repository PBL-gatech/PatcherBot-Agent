# coding=utf-8
import pickle
import os
import math
from numbers import Real

import numpy as np
from datetime import datetime
from PyQt5 import QtCore

from patcherbot.interface import TaskInterface, command, blocking_command
from patcherbot.devices.manipulator.calibratedunit import CalibratedUnit, CalibratedStage
from patcherbot.configs.CalibrationConfig import CalibrationConfig
from patcherbot.controller.collision import GuardedPipetteDevice, WorkspaceGuard
from patcherbot.devices.cellsorter import CalibratedCellSorter
from patcherbot.utils.multipipette_helper import (
    PipetteConfigView, keyed_pipette_entries, pipette_index, pipette_key_for,
    validate_pipette_geometry,
)
import time

from patcherbot.devices.manipulator.microscope import Microscope
from patcherbot.devices.manipulator.helpers.Observer import Observer

# Positions saved per pipette (manipulator XYZ, or stage XY + microscope Z for *_stage).
_SAVED_POSITION_NAMES = ("home", "home_stage", "safe", "safe_stage", "bath", "rinse")


def _active_position(name, doc):
    """Property reading/writing the active pipette's saved position ``name``."""
    def getter(self):
        return self._saved_positions[self.active_pipette][name]

    def setter(self, value):
        self._saved_positions[self.active_pipette][name] = value
    return property(getter, setter, doc=doc)


class PipetteInterface(TaskInterface):
    '''
    Controller for the stage, the microscope, the pipettes, and the cell sorter.

    Every pipette has its own CalibratedUnit (``calibrated_units[key]``),
    its own view of the calibration config (per-pipette angles and saved
    positions) and its own saved positions. Commands act on the *active*
    pipette (``active_pipette``, default the first); ``calibrated_unit`` is
    the active pipette's unit.
    '''
    active_pipette_changed = QtCore.pyqtSignal(str)
    # A move refused by the collision guard (the readable explanation).
    collision_blocked = QtCore.pyqtSignal(str)

    home_position = _active_position("home", "Active pipette's home position (manipulator um)")
    home_stage_position = _active_position("home_stage", "Stage XY + microscope Z at the active pipette's home")
    safe_position = _active_position("safe", "Active pipette's safe position (manipulator um)")
    safe_stage_position = _active_position("safe_stage", "Stage XY + microscope Z at the active pipette's safe position")
    cleaning_bath_position = _active_position("bath", "Active pipette's cleaning bath position")
    rinsing_bath_position = _active_position("rinse", "Active pipette's rinsing bath position")
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
        self.pipette_units = dict(unit) if isinstance(unit, dict) else {pipette_key_for(0): unit}
        if not self.pipette_units:
            raise ValueError("At least one pipette unit must be selected")
        # One patch workflow runs at a time, on the active pipette.
        self.primary_pipette = next(iter(self.pipette_units))
        self.active_pipette = self.primary_pipette
        # Extra "is something running?" checks (e.g. the patch interface's
        # task); switching pipettes is refused while any returns True.
        self.busy_checks = []
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
        # Every selected pipette needs its own geometry in the calibration file.
        # RigConfigManager already checked (and warned) before the hardware
        # started; this repeats the check for other entry points.
        validate_pipette_geometry(self.calibration_config, list(self.pipette_units))
        if keyed_pipette_entries(self.calibration_config) is not None:
            # Calibration tab: one group per selected pipette with its own
            # angles, k, cleaning order and positions (the unused top-level
            # values are left out).
            self.calibration_config.show_pipettes_in_editor(list(self.pipette_units))
        # Each pipette sees the shared config through a view of its own
        # `pipettes.<key>` entry (angles, saved positions...).
        self.calibrated_units = {}
        for pipette_key, pipette_unit in self.pipette_units.items():
            config = PipetteConfigView(self.calibration_config, pipette_key)
            self.calibrated_units[pipette_key] = CalibratedUnit(
                pipette_unit,
                self.calibrated_stage,
                microscope,
                camera,
                config=config,
            )
        self._saved_positions = {key: dict.fromkeys(_SAVED_POSITION_NAMES)
                                 for key in self.calibrated_units}
        self._pos_before_raise = dict.fromkeys(self.calibrated_units)
        self._tare_pipettes = {key: np.array([None, None, None])
                               for key in self.calibrated_units}
        # Pipette calibrations read from file but not (re)made this session;
        # kept so write_calibration doesn't drop pipettes that aren't selected.
        self._loaded_pipette_calibrations = {}
        # Collision guard: a moving pipette stays on its side of every other tip. Every
        # move command reaches the motors through a guarded device, installed
        # on both the raw unit (used by the calibration helpers) and the
        # calibrated unit.
        self.collision_guard = WorkspaceGuard(
            self.calibrated_units, self.calibrated_stage, self.calibration_config,
            home_anchor=self._home_anchor, camera=camera,
            on_blocked=self.collision_blocked.emit)
        for pipette_key, calibrated_unit in self.calibrated_units.items():
            raw_unit = self.pipette_units[pipette_key]
            guarded = GuardedPipetteDevice(raw_unit.dev, pipette_key, raw_unit.axes,
                                           self.collision_guard)
            raw_unit.dev = guarded
            calibrated_unit.dev = guarded
            if pipette_index(pipette_key) not in (None, 0):
                # Secondary pipettes log under "<logger>.pipette_<n>" so log
                # readers (rig replay) can tell them from pipette_0.
                calibrated_unit.log_name_suffix = pipette_key
                calibrated_unit._logger = None
                calibrated_unit.pipetteCalHelper.log_name = f"patcherbot.PipetteCalibration.{pipette_key}"
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


        self.contact_position = None
        self.paramecium_tank_position = None
        self.timer_t0 = time.time()
        self.tare_stage = np.array([None, None, None])

    # ------------------------------------------------------------------
    # Active pipette
    # ------------------------------------------------------------------
    @property
    def calibrated_unit(self):
        """The active pipette's CalibratedUnit."""
        return self.calibrated_units[self.active_pipette]

    @property
    def primary_calibrated_unit(self):
        return self.calibrated_units[self.primary_pipette]

    @property
    def pos_before_raise(self):
        return self._pos_before_raise[self.active_pipette]

    @pos_before_raise.setter
    def pos_before_raise(self, value):
        self._pos_before_raise[self.active_pipette] = value

    @property
    def tare_pipette(self):
        """Active pipette's display tare (raw manipulator XYZ, entries may be None)."""
        return self._tare_pipettes[self.active_pipette]

    @tare_pipette.setter
    def tare_pipette(self, value):
        self._tare_pipettes[self.active_pipette] = value

    def pipette_tare(self, pipette_key):
        """One pipette's display tare (raw manipulator XYZ, entries may be None)."""
        return self._tare_pipettes[pipette_key]

    def set_pipette_tare(self, pipette_key, values):
        tare = np.asarray(self._tare_pipettes[pipette_key], dtype=object).copy()
        for index, value in enumerate(values):
            if value is not None:
                tare[index] = value
        self._tare_pipettes[pipette_key] = tare

    @command(category='Manipulators',
             description='Switch the collision guard on or off (debugging)')
    def set_collision_guard(self, enabled):
        self.collision_guard.set_enabled(bool(enabled))

    def is_busy(self):
        return self._current_controller is not None or any(check() for check in self.busy_checks)

    @command(category='Manipulators',
             description='Select the pipette that commands act on')
    def set_active_pipette(self, pipette_key):
        if pipette_key not in self.calibrated_units:
            raise KeyError(f"Unknown pipette {pipette_key!r}; known: {list(self.calibrated_units)}")
        if pipette_key != self.active_pipette:
            if self.is_busy():
                self.active_pipette_changed.emit(self.active_pipette)
                raise RuntimeError("Wait for the current task to finish before switching pipettes.")
            self.active_pipette = pipette_key
            # Observer caches the unit it was built with.
            self.observer = Observer(self)
            self.info(f"Active pipette: {pipette_key}")
        self.active_pipette_changed.emit(self.active_pipette)

    @command(category='Manipulators',
             description='Make the next/previous pipette (by index, wrapping) the active one')
    def step_active_pipette(self, step):
        """Active pipette index += ``step``, modulo the number of pipettes."""
        keys = list(self.calibrated_units)
        index = keys.index(self.active_pipette)
        self.set_active_pipette(keys[(index + int(step)) % len(keys)])

    def saved_positions(self, pipette_key=None):
        """Copy of one pipette's saved positions (default: the active pipette)."""
        key = self.active_pipette if pipette_key is None else pipette_key
        return dict(self._saved_positions[key])

    def _home_anchor(self, pipette_key):
        """Tip at the image centre when pipette and stage are at home (or None)."""
        positions = self._saved_positions[pipette_key]
        home, home_stage = positions["home"], positions["home_stage"]
        if home is None or home_stage is None:
            return None
        centre = np.array([self.camera.width / 2., self.camera.height / 2.])
        return centre, np.asarray(home, dtype=float), np.asarray(home_stage, dtype=float)[:2]

    def get_pipette_positions(self):
        """Return current XYZ positions for every selected pipette."""
        return {
            name: np.asarray(pipette_unit.position(), dtype=float).copy()
            for name, pipette_unit in self.pipette_units.items()
        }

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
        # Keyboard jog: manual, never blocked by the collision guard.
        with self.collision_guard.manual():
            self.calibrated_unit.relative_move(distance, axis=0)


    @command(category='Manipulators',
                description='Write current calibration to file')
    def write_calibration(self):
        if not self.calibrated_stage.calibrated:
            raise RuntimeError('Stage not calibrated')

        floor_z = self.microscope.floor_Z
        if floor_z is not None:
            if isinstance(floor_z, bool) or not isinstance(floor_z, Real) or not math.isfinite(floor_z):
                raise ValueError('Cell floor must be a finite coordinate or unset')
            floor_z = float(floor_z)

        # Every calibrated pipette with home and safe positions is saved;
        # pipettes loaded from file but not selected now are carried over.
        pipettes = dict(self._loaded_pipette_calibrations)
        missing = {}
        for key, unit in self.calibrated_units.items():
            positions = self._saved_positions[key]
            if not unit.calibrated:
                missing[key] = 'Manipulator not calibrated'
            elif positions['home'] is None or positions['home_stage'] is None:
                missing[key] = 'Home position not set'
            elif positions['safe'] is None or positions['safe_stage'] is None:
                missing[key] = 'Safe position not set'
            else:
                pipettes[key] = {
                    'manip': unit.save_configuration(),
                    'home': np.concatenate((np.asarray(positions['home'], dtype=float),
                                            np.asarray(positions['home_stage'], dtype=float))),
                    'safe': np.concatenate((np.asarray(positions['safe'], dtype=float),
                                            np.asarray(positions['safe_stage'], dtype=float))),
                    'bath': (None if positions['bath'] is None
                             else np.asarray(positions['bath'], dtype=float)),
                }
        if self.active_pipette in missing:
            raise RuntimeError(f'{missing[self.active_pipette]} ({self.active_pipette})')
        for key, reason in missing.items():
            self.warning(f'{key} not saved to the calibration file: {reason}')

        record = {
            'version': 2,
            'pipettes': pipettes,
            'stage': self.calibrated_stage.save_configuration(),
            'floor_Z': floor_z,  # Same units as set_floor(); no hardware movement.
            'saved_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        }
        # Legacy single-pipette keys (manip/home/safe/bath) are always
        # pipette_0's: the rig replay and dataset builders pair them with
        # pipette_0's movement recording. Without pipette_0 they are left out,
        # so those readers fail loudly instead of using another pipette's matrix.
        legacy_key = pipette_key_for(0)
        if legacy_key in pipettes:
            record.update(pipettes[legacy_key])

        try:
            # Build the complete file path for 'calibration.pickle'
            file_path = os.path.join(self.folder_path, 'calibration.pickle')
            if not os.path.exists(self.folder_path):
                os.makedirs(self.folder_path, exist_ok=True)
                self.folder_created = True

            # Write calibration data to the file in the specified folder
            with open(file_path, 'wb') as f:
                pickle.dump(record, f)
            self.info(f'Calibration written to file for {", ".join(pipettes)}')
        except Exception as e:
            self.error(f'Error writing calibration to file: {e}')
            raise

    @staticmethod
    def _validated_matrix(config, expected_shape, name):
        # Validate calibrated formats, not the stage's temporary 3x2 startup matrix.
        matrix = np.asarray(config['M'], dtype=float)
        if matrix.shape != expected_shape or not np.isfinite(matrix).all():
            raise ValueError(
                f'Saved {name} calibration matrix must be finite with shape '
                f'{expected_shape}; got {matrix.shape}')
        config = dict(config)
        config['M'] = matrix
        return config

    @staticmethod
    def _validated_pipette_entry(entry, key):
        home = np.asarray(entry['home'], dtype=float).reshape(-1)
        safe = np.asarray(entry['safe'], dtype=float).reshape(-1)
        if home.shape != (6,) or safe.shape != (6,) or not np.isfinite(np.r_[home, safe]).all():
            raise ValueError(f'Saved home and safe anchors of {key} must each contain six finite coordinates')
        bath = entry.get('bath')
        if bath is not None:
            bath = np.asarray(bath, dtype=float)
            if bath.shape != (3,) or not np.isfinite(bath).all():
                raise ValueError(f'Saved bath position of {key} must contain three finite coordinates')
        manip = PipetteInterface._validated_matrix(entry['manip'], (3, 3), f'{key} manip')
        return dict(manip=manip, home=home, safe=safe, bath=bath)

    @command(category='Manipulators',
                description='read most recent calibration from file')
    def read_calibration(self, config_filename='calibration.pickle'):
        '''
        Read calibration from file.

        Version-2 files hold one entry per pipette; older files hold one
        pipette, which is loaded as the primary pipette. Returns the keys of
        the pipettes that were loaded.
        '''
        if not os.path.isfile(config_filename):
            raise RuntimeError('No calibration file found, need to calibrate before usage!')
        with open(config_filename, 'rb') as f:
            cal = pickle.load(f)

        # ---- validate everything before changing any state ----
        floor_z = cal.get('floor_Z')
        if floor_z is not None:
            if isinstance(floor_z, bool) or not isinstance(floor_z, Real) or not math.isfinite(floor_z):
                raise ValueError('Saved cell floor must be a finite coordinate or unset')
            floor_z = float(floor_z)
        raw_entries = cal.get('pipettes') or {pipette_key_for(0): cal}  # legacy file = pipette_0
        entries = {key: self._validated_pipette_entry(entry, key)
                   for key, entry in raw_entries.items()}
        stage_config = self._validated_matrix(cal['stage'], (2, 2), 'stage')
        if floor_z is None:
            # Home contains pipette XYZ followed by stage XYZ, all in micrometres.
            reference = entries.get(self.primary_pipette) or next(iter(entries.values()))
            delta = self.calibration_config.home_position_delta_um
            if isinstance(delta, bool) or not isinstance(delta, Real) or not math.isfinite(delta):
                raise ValueError('Home position offset must be finite')
            floor_z = float(reference['home'][5]) - float(delta)
            if not math.isfinite(floor_z):
                raise ValueError('Inferred cell floor must be finite')

        # ---- apply ----
        self.calibrated_stage.load_configuration(stage_config)
        self.microscope.floor_Z = floor_z
        loaded = []
        self._loaded_pipette_calibrations = {}
        for key, entry in entries.items():
            unit = self.calibrated_units.get(key)
            if unit is None:
                # Not selected this session: keep it so the next write keeps it.
                self._loaded_pipette_calibrations[key] = raw_entries[key]
                continue
            unit.load_configuration(entry['manip'])
            unit._tip_reference = None
            unit._trusted_tip_reference = None
            positions = self._saved_positions[key]
            positions['home'] = entry['home'][:3].copy()
            positions['home_stage'] = entry['home'][3:].copy()
            positions['safe'] = entry['safe'][:3].copy()
            positions['safe_stage'] = entry['safe'][3:].copy()
            positions['bath'] = entry['bath']
            loaded.append(key)
        not_loaded = [key for key in self.calibrated_units if key not in entries]
        if not_loaded:
            self.warning(f'Calibration file has no entry for {", ".join(not_loaded)}; '
                         f'calibrate before using {"it" if len(not_loaded) == 1 else "them"}')

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
            'pipettes': list(loaded),
        }
        # Keep this line's format: the rig replay parses
        # "Calibration reloaded from <file>; file saved at:" to find the file.
        self.info(
            f"Calibration reloaded from {source_file}; "
            f"file saved at: {source_saved_at or 'unknown'}; "
            f"source session folder: {self.loaded_calibration_reference['source_session']}; "
            f"reloaded at: {self.loaded_calibration_reference['reloaded_at']}")
        self.info(f"Calibration loaded for: {', '.join(loaded) or 'no selected pipette'}")
        self.check_calibration_geometry()
        return loaded

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

            # tare.pickle keeps exactly {'pipette', 'stage'} (the experiment
            # book only accepts that), with 'pipette' = pipette_0's tare. The
            # other pipettes' display tares go to tare_pipettes.pickle.
            with open(file_path, 'wb') as f:
                pickle.dump({
                    'pipette': self._tare_pipettes.get(pipette_key_for(0), np.array([None] * 3)),
                    'stage': self.tare_stage,
                }, f)
            with open(os.path.join(self.folder_path, 'tare_pipettes.pickle'), 'wb') as f:
                pickle.dump(dict(self._tare_pipettes), f)
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
        # Keyboard jog: manual, never blocked by the collision guard.
        with self.collision_guard.manual():
            self.calibrated_unit.relative_move(distance, axis=1)

    @command(category='Manipulators',
             description='Move pipette in z direction by {:.0f}μm',
             default_arg=-1000)
    def move_pipette_z(self, distance):
        # Keyboard jog: manual, never blocked by the collision guard.
        with self.collision_guard.manual():
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
        self._warn_other_tips_in_view()
        self.execute([self.calibrated_unit.calibrate_pipette])
        self.check_calibration_geometry()

    def check_calibration_geometry(self, angle_tolerance_deg=15.0, scale_tolerance=0.25):
        """Compare each calibrated pipette's measured stage -> manipulator map
        with the one its calibration-file geometry gives (k * Rz * flips, used
        by Follow Stage and locate_cell). Returns the warnings.

        The calibration predicts the map G = inv(M_pipette) @ M_stage, up to an
        overall sign (which depends on conventions that are not verified here).
        That sign doesn't affect the handedness (det of a 2x2 is unchanged by
        negation) or the scale, and leaves the angle known modulo 180 deg, so
        those three are what is checked.
        """
        stage = self.calibrated_stage
        if not getattr(stage, "calibrated", False):
            return []
        warnings = []
        for key, unit in self.calibrated_units.items():
            if not unit.calibrated:
                continue
            try:
                stage_map = np.asarray(stage.M, dtype=float)[:2, :2]
                pipette_map = np.asarray(unit.M, dtype=float)[:2, :2]
                measured = np.linalg.solve(pipette_map, stage_map)
                config = unit.config
                flips = np.diag([-1.0 if config.stage_x_axis_flip else 1.0,
                                 -1.0 if config.stage_y_axis_flip else 1.0])
                angle = float(config.pipette_z_rotation)
                k_scale = float(config.pipette_k_scale)
            except (AttributeError, TypeError, ValueError, IndexError, np.linalg.LinAlgError):
                continue
            det = float(np.linalg.det(measured))
            if not np.isfinite(det) or abs(det) < 1e-12:
                continue
            problems = []
            if np.sign(det) != np.sign(np.linalg.det(flips)):
                problems.append(
                    f"its calibration is {'a mirror image of' if det < 0 else 'a plain rotation of'} the "
                    f"stage axes, but stage_x_axis_flip={bool(config.stage_x_axis_flip)} / "
                    f"stage_y_axis_flip={bool(config.stage_y_axis_flip)} say the opposite: toggle one "
                    f"of pipettes.{key}.stage_x_axis_flip / stage_y_axis_flip (no pipette_z_rotation "
                    f"can make up for it), then re-check the angle with Follow Stage")
            else:
                unflipped = measured @ flips          # = +/- k * Rz(angle)
                measured_angle = math.degrees(math.atan2(unflipped[1, 0] - unflipped[0, 1],
                                                         unflipped[0, 0] + unflipped[1, 1]))
                error = (measured_angle - angle + 90.0) % 180.0 - 90.0
                if abs(error) > angle_tolerance_deg:
                    problems.append(
                        f"its calibration puts the rotation at about {angle + error:.1f} deg "
                        f"(or that +/- 180), but pipettes.{key}.pipette_z_rotation = {angle:g}")
            measured_scale = math.sqrt(abs(det))
            if k_scale and abs(measured_scale / abs(k_scale) - 1.0) > scale_tolerance:
                problems.append(
                    f"its calibration gives about {measured_scale:.2f} pipette um per stage um, but "
                    f"pipettes.{key}.pipette_k_scale = {k_scale:g}")
            if problems:
                message = (f"{key}: " + "; ".join(problems) + ". Follow Stage and locate_cell move "
                           f"{key} with the calibration-file values, so check them with Follow Stage "
                           f"(if this calibration is right).")
                self.warning(message)
                warnings.append(message)
        return warnings

    def _warn_other_tips_in_view(self):
        """Calibration detects one tip with no expected position, so another
        pipette in the field of view can be picked up instead."""
        for key, unit in self.calibrated_units.items():
            if key == self.active_pipette:
                continue
            if not unit.calibrated:
                self.warning(f'Calibrating {self.active_pipette}: make sure {key} is out of view '
                             f'(its position is unknown).')
                continue
            try:
                anchor = (getattr(unit, '_tip_reference', None)
                          or getattr(unit, '_trusted_tip_reference', None)
                          or self._home_anchor(key))
                x, y = unit.reference_position(anchor=anchor)[:2]
            except Exception:
                continue
            if 0 <= x < self.camera.width and 0 <= y < self.camera.height:
                self.warning(f'Calibrating {self.active_pipette}: {key} looks to be in view '
                             f'(~{x:.0f}, {y:.0f} px); move it out of view first.')

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
        with self.collision_guard.manual():  # manual: never blocked by the collision guard
            self._raise_pipette()

    def _raise_pipette(self):
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
        with self.collision_guard.manual():  # manual: never blocked by the collision guard
            self._lower_pipette()

    def _lower_pipette(self):
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
