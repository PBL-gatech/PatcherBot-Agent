import time
import csv

from numbers import Integral, Real
import numpy as np
from patcherbot.devices.amplifier.amplifier import Amplifier
from patcherbot.devices.amplifier.DAQ import DAQ, FakeDAQ, NiDAQ
from patcherbot.devices.manipulator.calibratedunit import CalibratedUnit, CalibratedStage
from patcherbot.devices.manipulator.microscope import Microscope
from patcherbot.devices.pressurecontroller import PressureController
from patcherbot.devices.lamp import Lamp
from patcherbot.devices.laser import Laser
from patcherbot.devices.manipulator.helpers.AgentHelper import AgentHelper
from patcherbot.devices.manipulator.helpers.Observer import Observer
from patcherbot.deepLearning.pipetteFocuser import PipetteFocuser2
from patcherbot.utils.StateMachineLogger import StateMachineLogger, record_state
import collections
import logging
from datetime import datetime
from uuid import uuid4
import pickle
import os
from patcherbot.configs.PatchConfig import PatchConfig
from patcherbot.configs.ProtocolConfig import ProtocolConfig

from .base import TaskController, RequestedSuccessException
from .phases.find_pipette import FindPipettePhase
from .errors import AutopatchError
from .phases.approach_cell import ApproachCellPhase
from .phases.hunt_cell import HuntCellPhase
from .phases.gigaseal import GigasealPhase
from .phases.break_in import BreakInPhase
import threading
# import locking package




class AutoPatcher(TaskController):
    # region Initialization and readiness
    def __init__(
        self,
        amplifier: Amplifier,
        daq: NiDAQ,
        pressure: PressureController,
        calibrated_unit: CalibratedUnit,
        microscope: Microscope,
        calibrated_stage: CalibratedStage,
        lamp: Lamp,
        laser=None,
        config: PatchConfig | None = None,
        protocol_config: ProtocolConfig | None = None,
    ):
        super().__init__()
        self.config = config
        self.protocol_config = protocol_config

    def __init__(
        self,
        amplifier: Amplifier,
        daq: NiDAQ,
        pressure: PressureController,
        calibrated_unit: CalibratedUnit,
        microscope: Microscope,
        calibrated_stage: CalibratedStage,
        lamp: Lamp,
        laser=None,
        config: PatchConfig | None = None,
        protocol_config: ProtocolConfig | None = None,
    ):
        super().__init__()
        self.config = config if config is not None else PatchConfig(name="Patch")
        self.protocol_config = protocol_config if protocol_config is not None else ProtocolConfig(name="Protocols")
        self.amplifier = amplifier
        self.daq = daq
        if isinstance(self.daq, FakeDAQ):
            self.daq.pressureController = pressure
        self.pressure = pressure
        self.calibrated_unit = calibrated_unit
        self.calibrated_stage = calibrated_stage
        self.microscope = microscope
        self.lamp = lamp
        self.laser = laser
        self.laser = laser
        self.safe_position = None
        self.safe_stage_position = None
        self.home_position = None
        self.home_stage_position = None
        self.cleaning_bath_position = None
        self.rinsing_bath_position = None
        self.contact_position = None
        self.initial_resistance = None
        self.vholding = None
        self.iholding = None
        self.rig_ready = False
        self.first_res = None
        self.atm = False
        self.attempt_counter = 0
        self._state_recorder = None
        self._in_patch       = False
        self.agenthelper = AgentHelper(
            use_ai_features=bool(self.calibrated_stage.config.use_ai_features)
        )
        self.current_protocol_graph = None
        self.goal_needed = True
        self.goal_random = True
        self.ninput = None
        self.done = False
        self.find_pipette_velocity_speed_um_s = 1000.0
        # Agent find_pipette toggle:
        # False -> interpret model output as displacement (xy px, z um) and use relative moves.
        # True  -> interpret model output as velocity (xy px/s, z um/s) and stream velocity commands.
        self.velocity_prediction = False
        self.observer = Observer(self)

        # Phases share this controller's hardware, configuration, and request state.
        self.find_pipette_phase = FindPipettePhase(self)
        self.approach_cell_phase = ApproachCellPhase(self)
        self.hunt_cell_phase = HuntCellPhase(self)
        self.gigaseal_phase = GigasealPhase(self)
        self.break_in_phase = BreakInPhase(self)

    def _origin_position(self):
        xy = np.asarray(self.calibrated_stage.position(), dtype=float).reshape(-1)
        scale = float(self.calibrated_unit.config.microscope_units_per_um)
        z = float(self.microscope.position()) / scale
        position = np.array([xy[0], xy[1], z])
        if not np.isfinite(position).all():
            raise ValueError("Stage position is unavailable.")
        return position

    def save_origin(self, request):
        """Validate and save a stationary microscope frame using the snapshot recorder."""
        position = self._origin_position()
        stable_since = time.monotonic()
        while time.monotonic() - stable_since < 0.5:
            self.sleep(0.05)
            if request["movement_busy"]():
                raise ValueError("Wait for the current movement/task to finish.")
            if not np.allclose(position, self._origin_position(), rtol=0, atol=0.1):
                raise ValueError("Wait for the stage to stop before saving.")
        camera_interface = request["camera_interface"]
        try:
            number, captured_at, _, raw = camera_interface.camera.raw_frame_queue[0]
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            raise ValueError("A fresh microscope frame is required.") from exc
        now = datetime.now().astimezone()
        if number is None or raw is None or captured_at is None:
            raise ValueError("A fresh microscope frame is required.")
        age = now.timestamp() - captured_at.timestamp()
        if not 0 <= age <= 1.0:
            raise ValueError("A fresh microscope frame is required.")
        if age > time.monotonic() - stable_since:
            raise ValueError("Wait for a microscope frame after the stage stops.")
        frame = np.asarray(raw).copy()
        if (not frame.size or request["movement_busy"]()
                or not np.allclose(position, self._origin_position(), rtol=0, atol=0.1)):
            raise ValueError("Stage moved during capture; try again when stationary.")
        self.abort_if_requested()
        axis = request["axis"]
        origins = dict(request["origins_um"])
        origins[axis] = float(position[0 if axis == "x" else 1])
        snapshot = camera_interface.save_snapshot(number, captured_at, frame, wait=True)
        record = dict(
            origin_id=uuid4().hex, axis=axis, stage_xyz_um=position.tolist(),
            origins_um=origins, saved_at=now.isoformat(),
            captured_at=captured_at.isoformat(), frame_number=int(number),
            camera_role="main", experiment=dict(request["experiment"]),
            session_id=str(request["logger"].session_dir),
            image_path=snapshot["image_path"])
        record = request["logger"].write_origin(record)
        request["completed"](record, frame)

    def _get_state_recorder(self) -> StateMachineLogger:
        """
        Retrieve or initialize the state machine logger for the current attempt.

        If no recorder exists, a new one is created and the attempt counter
        is incremented.

        Returns:
            StateMachineLogger: Recorder instance for logging state transitions.
        """
        if self._state_recorder is None:
            self.attempt_counter += 1
            self.observer.reset_attempt(self.attempt_counter)
            self._state_recorder = StateMachineLogger(
                base_path="experiments/Data/state_recorder_data/",
                attempt_id=self.attempt_counter
            )
        return self._state_recorder

    def isrigready(self):
        try:
            if not self.calibrated_unit.calibrated:
                # # testing scenario
                # self.calibrated_unit.calibrated = True
                # print("Pipette calibrated for testing")
                raise AutopatchError("Pipette not calibrated")
            if not self.calibrated_stage.calibrated:
                raise AutopatchError("Stage not calibrated")
            if self.safe_position is None:
                raise ValueError('Safe position has not been set')
            if self.home_position is None:
                raise ValueError('Home position has not been set')
            if self.cleaning_bath_position is None:
                raise ValueError('Cleaning bath position has not been set')
            if self.microscope.floor_Z is None:
                raise AutopatchError("Cell Plane not set")
            self.rig_ready = True
        except (AutopatchError, ValueError) as e:
            self.rig_ready = False
            raise e
    # endregion

    # region Patch workflows
    @record_state("patch")
    def patch(self, cell=None):
        """Runs the automatic patch-clamp algorithm, including manipulator movements."""
        self._in_patch = True
        self._get_state_recorder()

        def _run_phase(phase_callable, *phase_args, sleep_after=None):
            """
            Execute a patching phase while ignoring manual success interrupts so
            the full sequence can continue. Any other exception still bubbles up.
            """
            try:
                phase_callable(*phase_args)
            except RequestedSuccessException:
                return
            finally:
                # Reset the flag so follow-up phases do not see a stale request.
                self.success_requested = False
            if sleep_after:
                self.sleep(sleep_after)

        cleanup_performed = False

        try:
            # ------ rig preparation -------------------------------#
            self.isrigready()
            if self.rig_ready is False:
                raise AutopatchError("Rig not ready for patching")

            if cell is None:
                raise AutopatchError("No cell given to patch!")

            self.info("Starting patching process")

            #! Phase 0: locate cell
            _run_phase(self.locate_cell, cell)
            _run_phase(self.approach_cell, cell, sleep_after=5)

            #! Phase 1: hunt for cell
            _run_phase(self.hunt_cell, cell, sleep_after=3)

            #! Phase 2: attempt to form a gigaseal
            _run_phase(self.gigaseal, sleep_after=3)

            #! Phase 3: break into cell
            _run_phase(self.break_in)
            self.info("Whole-cell achieved, resting for 5 seconds")
            self.sleep(5)
            
            if not self.protocol_config.custom_cclamp_protocol:
                    #! Phase 4: run protocols
                    self.info(f"Running protocol")
                    _run_phase(self.run_protocols)


                    #! Phase 5: clean pipette
                    self.info("Data collection complete, cleaning pipette")
                    if self.config.auto_clean_pipette:
                        _run_phase(self.escape)
                        cleanup_performed = True

            self.success_requested = True

        finally:
            if not cleanup_performed:
                try:
                    self.info("Patch attempt interrupted, running escape cleanup")
                    if self.config.auto_clean_pipette:
                        self.escape()
                except RequestedSuccessException:
                    # Escape may also set success; clear it so teardown can finish.
                    self.success_requested = False
                except Exception as cleanup_error:
                    self.warning(f"Cleanup escape failed: {cleanup_error}")
            # ---- teardown so the next call starts a fresh attempt ----
            self._state_recorder = None
            self._in_patch = False

    @record_state("whole_cell")
    def whole_cell(self, cell=None):
        """ method similar to patch, but starts from gigaseal stage
        """
        self._in_patch = True
        self._get_state_recorder()

        def _run_phase(phase_callable, *phase_args, sleep_after=None):
            """
            Execute a patching phase while ignoring manual success interrupts so
            the full sequence can continue. Any other exception still bubbles up.
            """
            try:
                phase_callable(*phase_args)
            except RequestedSuccessException:
                return
            finally:
                # Reset the flag so follow-up phases do not see a stale request.
                self.success_requested = False
            if sleep_after:
                self.sleep(sleep_after)

        cleanup_performed = False

        try:
            # ------ rig preparation -------------------------------#
            self.isrigready()
            if self.rig_ready is False:
                raise AutopatchError("Rig not ready for patching")

            if cell is None:
                raise AutopatchError("No cell given to patch!")

            self.info("Starting patching process")

            #! Phase 2: attempt to form a gigaseal
            _run_phase(self.gigaseal, sleep_after=3)

            #! Phase 3: break into cell
            _run_phase(self.break_in)
            self.info("Whole-cell achieved, resting for 5 seconds")
            self.sleep(5)

            if not self.protocol_config.custom_cclamp_protocol:
                    #! Phase 4: run protocols
                    self.info(f"Running protocol")
                    _run_phase(self.run_protocols)

            self.success_requested = True
            cleanup_performed = True

        finally:
            if not cleanup_performed:
                try:
                    self.info("Patch attempt interrupted, running escape cleanup")
                    if self.config.auto_clean_pipette:
                        self.escape()
                except RequestedSuccessException:
                    # Escape may also set success; clear it so teardown can finish.
                    self.success_requested = False
                except Exception as cleanup_error:
                    self.warning(f"Cleanup escape failed: {cleanup_error}")
            # ---- teardown so the next call starts a fresh attempt ----
            self._state_recorder = None
            self._in_patch = False
    # endregion

    # region Phase entry points
    @record_state("find_pipette")
    def find_pipette(self):
        return self.find_pipette_phase.run()

    @record_state("approach_cell")
    def approach_cell(self, cell):
        """Approach the cell from the common localization distance."""
        return self.approach_cell_phase.run(cell)

    @record_state("hunt_cell")
    def hunt_cell(self, cell=None):
        """Move the pipette toward the cell plane and detect contact by resistance."""
        return self.hunt_cell_phase.run(cell)

    @record_state("gigaseal")
    def gigaseal(self):
        """requires **three consecutive**
        averaged-resistance windows ≥ target to declare success, reducing
        false positives from transient spikes.
        """
        return self.gigaseal_phase.run()
   
    @record_state("break_in")
    def break_in(self):
        """
        Attempts whole-cell break-in.

        NEW LOGIC
        ---------
        * Measure access resistance first each loop.
        * On every sub-threshold reading, **pause** all pressure/zap activity and
        skip the slow resistance & capacitance ramps.
        * Require three consecutive good readings to confirm success.
        * The moment a reading is above threshold, reset the streak and fall back
        to the full pressure/zap/ramp cycle.
        """
        return self.break_in_phase.run()

    # endregion
    
    # region Phase helpers
    def fine_calibrate_pipette(self):
        '''
        Fine calibrates the pipette using microscope imaging
        '''
        self.info("Fine calibrating pipette using imaging")
        self.calibrated_unit.center_pipette()
        self.calibrated_unit.wait_until_still()
        self.calibrated_unit.autofocus_pipette()
        self.calibrated_unit.wait_until_still()
        self.calibrated_unit.center_pipette()
        self.calibrated_unit.wait_until_still()
        self.calibrated_unit.autofocus_pipette()
        self.calibrated_unit.wait_until_still()
        self.calibrated_unit.center_pipette()
        self.calibrated_unit.wait_until_still()
        self.calibrated_unit.autofocus_pipette()
        self.calibrated_unit.wait_until_still()

    def align(self, cell, cell_distance, use_centroid):
        '''
        Aligns the pipette to the cell using microscope imaging
        '''
        return self.approach_cell_phase.align(cell, cell_distance, use_centroid)

    def clear_to_cell(self, cell):
        '''
        Clears the pipette to the cell by moving down while checking resistance
        '''
        return self.approach_cell_phase.clear_to_cell(cell)
        
    def _isCellDetected(self, lastResDeque, cellThreshold = 0.15):
        '''Given a list of three resistance readings, do we think there is a cell where the pipette is?
        '''
        return self.hunt_cell_phase._isCellDetected(lastResDeque, cellThreshold)
   

    # endregion

    # region Escape and cleaning

    @record_state("escape")
    def escape(self):
            self.amplifier.stop_patch()
            self.calibrated_unit.stop()
            self.microscope.stop()
            self.pressure.set_pressure(50)
            self.pressure.set_ATM(atm=False)
            self.daq.setCellMode(False)
            self.sleep(1)
            self.pressure.set_pressure(100)
            self.sleep(1)
            self.move_group_up(20)
            self.sleep(1)
            self.pressure.set_pressure(200)
            self.sleep(1)
            self.move_to_home_space()
            self.clean_pipette()
            self.sleep(1)
            self.move_to_safe_space()
            self.sleep(5)
            self.microscope.move_to_floor()
            self.success_requested = True
            self.success_if_requested()

    def clean_pipette(self):
        if self.cleaning_bath_position is None:
            raise ValueError('Cleaning bath position has not been set')

        if self.safe_position is None:
            raise ValueError('Safe position has not been set')
        # TODO: implement an abort mechanism
        try:
            start_x, start_y, start_z = self.calibrated_unit.position()
            safe_x, safe_y, safe_z = self.safe_position
            clean_move_order = list(self.calibrated_unit.config.clean_move_order)
            if len(clean_move_order) != 3 or set(clean_move_order) != {"x", "y", "z"}:
                self.warning(f"Invalid clean_move_order={clean_move_order}; falling back to ['y', 'x', 'z']")
                clean_move_order = ["y", "x", "z"]
            # Step 1: Move to the safe space
            self.move_to_safe_space()
            clean_x, clean_y, clean_z = self.cleaning_bath_position
            # Step 2 + 3: Move the pipette above and then down into the cleaning bath
            clean_position = {"x": clean_x, "y": clean_y, "z": clean_z}
            axis_map = {"x": 0, "y": 1, "z": 2}
            for axis_name in clean_move_order:
                axis = axis_map[axis_name]
                self.calibrated_unit.absolute_move(clean_position[axis_name], axis=axis)
                self.calibrated_unit.wait_until_still(axis)

            # Step 4: Cleaning
            # Fill up with the Alconox
            self.pressure.set_ATM(atm=False)
            self.pressure.set_pressure(-600)
            self.sleep(1)
            # 5 cycles of tip cleaning
            for i in range(1, 5):
                self.pressure.set_pressure(-600)
                self.sleep(0.75)
                self.pressure.set_pressure(1000)
                self.sleep(0.75)

            # Step 5: Drying
            # move pipette back to safe space in reverse configured axis order
            safe_target = {"x": safe_x, "y": safe_y, "z": safe_z}
            for axis_name in reversed(clean_move_order):
                axis = axis_map[axis_name]
                self.calibrated_unit.absolute_move(safe_target[axis_name], axis=axis)
                self.calibrated_unit.wait_until_still(axis)

            self.pressure.set_pressure(-600)
            self.sleep(1)
            # 5 cycles of tip cleaning
            for i in range(1, 5):
                self.pressure.set_pressure(-600)
                self.sleep(0.75)
                self.pressure.set_pressure(1000)
                self.sleep(0.75)
            self.pressure.set_pressure(50)
  
            # Step 6: Move back to start from safespace
            self.calibrated_unit.absolute_move_group([start_x,safe_y,start_z], [0,1,2])
            self.calibrated_unit.wait_until_still()
            self.calibrated_unit.absolute_move(start_y, axis=1)
            self.calibrated_unit.wait_until_still() # Ensure movement completes
        finally:
            pass

    def clean_pipette_no_move(self):
        '''
        Cleans the pipette without moving to cleaning bath position
        '''
        try:
            # Cleaning
            # Fill up with the Alconox
            self.pressure.set_ATM(atm=False)
            self.pressure.set_pressure(-600)
            self.sleep(1)
            # 5 cycles of tip cleaning
            for i in range(1, 5):
                self.pressure.set_pressure(-600)
                self.sleep(0.75)
                self.pressure.set_pressure(1000)
                self.sleep(0.75)

            self.pressure.set_pressure(50)
  
        finally:
            pass
    # endregion

    # region Recording protocols
    @record_state("run_protocols")
    def run_protocols(self):
        self.daq.setCellMode(True)
        holding = self.getHolding()
        if self.protocol_config.voltage_protocol:
            self.run_voltage_protocol()
            self.sleep(0.25)
        if self.protocol_config.current_protocol:
            self.daq.setCellMode(False)
            self.iholding = holding
            self.run_current_protocol()
            self.sleep(0.25)
            self.daq.setCellMode(True)
        if self.protocol_config.voltage_sweep_protocol:
            self.run_voltage_sweep_protocol()
            self.sleep(0.25)
        if self.protocol_config.holding_protocol:
            self.run_holding_protocol()
            self.sleep(0.25)
        if self.protocol_config.opto_random_wavelength_protocol or self.protocol_config.opto_random_power_protocol:
            self.run_optogenetic_protocol()
            self.sleep(0.25)
        self.success_requested = True
        self.success_if_requested()


    def getHolding(self):
        """Get the holding current as measured by the DAQ."""
        if self.protocol_config.custom_cclamp_protocol:
            holding_current = self.protocol_config.cclamp_hold
            return holding_current
        else:
            holding_current = self.protocol_config.cclamp_hold
            # self.amplifier.voltage_clamp()
            # self.sleep(1)
            # self.amplifier.switch_holding(False) 
            # self.sleep(1)
            # base1a = self.daq.holding_current
            # self.sleep(1)
            # base1b = self.daq.holding_current
            # if base1a and base1b is not None:
            #     base1 = float((base1a + base1b) / 2)
            #     if abs(base1) > 200:
            #         self.info(f'resting membrane current is too high:{base1} pA, setting to default value of 0 pA')
            #         base1 = 0
            # else: 
            #     base1 = None
            # self.amplifier.switch_holding(True)
            # self.sleep(1)
            # base2a = self.daq.holding_current
            # self.sleep(1)
            # base2b = self.daq.holding_current
            # # average base2a and base2b
            # if base2a and base2b is not None:
            #     base2 = float((base2a + base2b) / 2)
            # else:
            #     base2 = None
            # if base1 is None or base2 is None:
            #     self.info("Holding current not set, using default value")
            #     return -50
            # else:
            #     holding_current = (base2 - base1) 
            #     # self.info(f"Base1: {base1}, Base2: {base2}")
            #     self.info(f"Holding current: {holding_current} pA")
            #     if abs(holding_current) > 150:
            #         self.info("Holding current is too high, setting to default value of -50 pA")
            #         holding_current = -50
            return holding_current

    def run_voltage_protocol(self):
        """
        Execute a voltage clamp membrane test protocol, including automatic
        capacitance compensation and data acquisition.
        """
        self.info('Running voltage protocol (membrane test)')
        self.amplifier.voltage_clamp()
        self.sleep(0.25)
        self.amplifier.auto_fast_compensation()
        self.sleep(0.25)
        self.amplifier.auto_slow_compensation()
        self.sleep(0.25)
        self.info('auto capacitance compensation')  
        holding = self.amplifier.get_holding()
        if holding is None:
            holding = -0.070
        self.amplifier.set_holding(holding)
        self.info(f'holding at {holding} mV')
        membrane_hold = float(self.protocol_config.vclamp_hold)
        self.amplifier.set_holding(membrane_hold)
        self.info(f'holding at {membrane_hold * 1e3:.1f} mV for membrane test')
        self.sleep(0.25)
        self.amplifier.switch_holding(True)
        self.info('enabled holding')
        self.sleep(0.25)

        try:
            self.info("Getting data from voltage membrane test")
            self.daq.getDataFromVoltageProtocol(membrane_hold=membrane_hold)
            self.sleep(0.25)

        finally:
            self.amplifier.set_holding(membrane_hold)
            self.amplifier.switch_holding(False)
            self.sleep(0.25)
            self.info(f'holding reset to {membrane_hold * 1e3:.1f} mV after voltage protocol')
            self.info('finished running voltage membrane test')

    def run_voltage_sweep_protocol(self):
        """Execute a voltage clamp sweep protocol with optional P/4 leak subtraction."""
        self.info('Running voltage sweep protocol')
        self.amplifier.voltage_clamp()
        self.sleep(0.25)
        sweep_hold = float(self.protocol_config.vclamp_sweep_hold)
        sweep_step = float(self.protocol_config.vclamp_step)
        sweep_start = float(self.protocol_config.vclamp_start)
        sweep_end = float(self.protocol_config.vclamp_end)
        self.amplifier.set_holding(sweep_hold)
        self.info(f'holding at {sweep_hold * 1e3:.1f} mV for voltage sweep')
        self.sleep(0.25)
        self.amplifier.switch_holding(True)
        self.info('holding enabled for voltage sweep')
        self.sleep(0.25)
        self.info("Executing P/4 leak subtraction series")
        self.daq.getLeakSubtraction(
            start_voltage=sweep_start,
            step_voltage=sweep_step,
            end_voltage=sweep_end,
            holding_voltage=sweep_hold
        )
        self.sleep(0.25)
        self.info("Getting data from voltage clamp sweep")
        self.daq.getVoltageClampSweep(
            start_voltage=sweep_start,
            step_voltage=sweep_step,
            end_voltage=sweep_end,
            holding_voltage=sweep_hold
        )
        self.sleep(0.25)
        self.amplifier.set_holding(self.protocol_config.vclamp_hold)
        self.info('finished running voltage sweep protocol')

    def run_current_protocol(self):
        """
        Execute a current clamp protocol, including capacitance compensation,
        optional neutralization, and bridge balance.
        """
        self.info('Running current protocol (current clamp)')
        self.amplifier.voltage_clamp()
        self.sleep(0.1)
        self.amplifier.auto_fast_compensation()
        self.sleep(0.25)
        self.amplifier.auto_slow_compensation()
        self.sleep(0.25)
        self.info('auto capacitance compensation')  
        cap_c_double = self.amplifier.get_fast_compensation_capacitance()
        cap = float(cap_c_double.value) * 1e12 - 0.5
        cap = cap*1e-12
        self.info(f'fast compensation capacitance: {cap} pF' )
        self.sleep(0.1)
        self.amplifier.current_clamp()
        self.sleep(0.1)
        if self.protocol_config.enable_neutralization_capacitance:
            self.amplifier.set_neutralization_capacitance(cap)
            self.info('set neutralization capacitance')
            self.amplifier.set_neutralization_enable(True)
            self.info('enabled neutralization')
            self.sleep(0.1)
        else:
            self.info('neutralization capacitance disabled')
        if self.protocol_config.enable_bridge_balance:
            self.amplifier.set_bridge_balance(True)
            self.info('auto bridge balance')
            self.amplifier.auto_bridge_balance()
            self.sleep(0.1)
        else:
            self.info('bridge balance disabled')
        if self.iholding is None:
            current = self.protocol_config.cclamp_hold

        else:
            current = (self.iholding)

        current = current * 1e-12
        self.amplifier.set_holding(current)
        self.info(f'holding at {current} pA')
        self.sleep(0.1)
        self.amplifier.switch_holding(True)
        self.info('enabled holding')
        self.sleep(0.1)
        if self.protocol_config.custom_cclamp_protocol:
            self.debug('running custom current protocol')
            self.daq.getDataFromCurrentProtocol(
                custom=self.protocol_config.custom_cclamp_protocol,
                factor=1,
                startCurrentPicoAmp=self.protocol_config.cclamp_start,
                endCurrentPicoAmp=self.protocol_config.cclamp_end,
                stepCurrentPicoAmp=self.protocol_config.cclamp_step,
                recordingTimeMs=self.protocol_config.cclamp_recording_time_ms,
                dutyCycle=self.protocol_config.cclamp_duty_cycle,
            )
        else:
            self.debug('running default current protocol')
            self.daq.getDataFromCurrentProtocol(
                custom=self.protocol_config.custom_cclamp_protocol,
                factor=1,
                startCurrentPicoAmp=None,
                endCurrentPicoAmp=None,
                stepCurrentPicoAmp=10,
                recordingTimeMs=self.protocol_config.cclamp_recording_time_ms,
                dutyCycle=self.protocol_config.cclamp_duty_cycle,
            )
        self.sleep(0.1)
        self.amplifier.switch_holding(False)
        self.info('disabled holding')
        self.sleep(0.1)
        self.amplifier.voltage_clamp()
        self.info('finished running current protocol(current clamp)')

    def run_holding_protocol(self):
        """Execute a holding (E/I PSC) protocol in voltage clamp mode."""
        self.info('Running holding protocol (E/I PSC test)')
        self.amplifier.voltage_clamp()
        self.sleep(0.25)
        holding = float(self.protocol_config.vclamp_hold)
        self.amplifier.set_holding(holding)
        self.info(f'holding at {holding} mV')
        self.sleep(0.25)
        self.daq.getDataFromHoldingProtocol(duration_s=self.protocol_config.hclamp_duration)
        self.sleep(0.25)
        # self.amplifier.set_holding(0)
        self.sleep(0.25)
        self.amplifier.voltage_clamp()
        self.info('finished running holding protocol (E/I PSC test)')

    def run_optogenetic_protocol(self, protocol_params: dict | None = None):
        """
        Run optogenetic protocols based on configuration or explicit parameters.

        Args:
            protocol_params (dict | None):
                Optional dictionary specifying protocol parameters such as:
                wavelengths, powers, timing, randomization, and rate. If None,
                defaults to configuration-based execution.

        Returns:
            list | Any:
                A single result if one protocol is executed, otherwise a list
                of results from multiple protocol runs.

        Raises:
            RuntimeError:
                If the laser device is not available.
            RequestedAbortException:
                If an abort is requested during execution.
            RequestedSuccessException:
                If a success request is triggered during execution.
            Exception:
                Propagates unexpected errors from:
                - laser protocol construction (`build_optogenetic_protocol`)
                - DAQ execution (`getDataFromOptogeneticProtocol`)
                - configuration parsing or timing operations
        """
        if self.laser is None:
            raise RuntimeError("Laser device not available")

        self.info("Running optogenetic protocol")
        self.amplifier.voltage_clamp()
        self.sleep(0.25)
        holding = float(self.protocol_config.vclamp_hold)
        self.amplifier.set_holding(holding)
        self.info(f'holding at {holding} mV')
        self.sleep(0.25)

        results = []

        color_cycle = ["red", "green", "cyan", "uv", "blue", "infrared"]

        def _coerce_wavelength(value):
            if isinstance(value, str):
                return value.strip().lower()
            try:
                idx = int(value)
            except (TypeError, ValueError):
                return value
            if idx == 7:
                return "off"
            if idx <= 0:
                idx = 1
            return color_cycle[(idx - 1) % len(color_cycle)]

        def _run_steps(steps, rate_hz):
            result = self.daq.getDataFromOptogeneticProtocol(
                laser=self.laser,
                protocol_steps=steps,
                rate_hz=rate_hz,
            )
            results.append(result)
            self.sleep(0.25)

        if protocol_params is not None:
            randomize_target = protocol_params.get("randomize_target", protocol_params.get("mode", "wavelength"))
            raw_wavelengths = list(protocol_params.get("wavelengths", ["green"]))
            wavelengths = [_coerce_wavelength(value) for value in raw_wavelengths]
            steps = self.laser.build_optogenetic_protocol(
                wavelengths=wavelengths,
                powers=list(protocol_params.get("powers", [50])),
                randomize_target=randomize_target,
                stabilize_time=float(protocol_params.get("stabilize_time", 1.0)),
                off_time=float(protocol_params.get("off_time", 0.1)),
                on_time=float(protocol_params.get("on_time", 0.01)),
                replicates=int(protocol_params.get("replicates", 1)),
                randomize=bool(protocol_params.get("randomize", True)),
                power_divisor=float(protocol_params.get("power_divisor", 1.0)),
            )
            _run_steps(steps, int(protocol_params.get("rate_hz", 50_000)))
        else:
            cfg = self.protocol_config
            stabilize_time = float(cfg.opto_stabilize_time)
            off_time = float(cfg.opto_off_time)
            on_time = float(cfg.opto_on_time)
            replicates = int(cfg.opto_replicates)
            rate_hz = 50_000

            if cfg.opto_random_wavelength_protocol:
                wavelengths = ["red", "green", "cyan", "uv", "blue", "infrared"]
                powers = [float(cfg.opto_wavelength_power)]
                steps = self.laser.build_optogenetic_protocol(
                    wavelengths=wavelengths,
                    powers=powers,
                    randomize_target="wavelength",
                    stabilize_time=stabilize_time,
                    off_time=off_time,
                    on_time=on_time,
                    replicates=replicates,
                    randomize=True,
                )
                _run_steps(steps, rate_hz)

            if cfg.opto_random_power_protocol:
                wavelengths = [_coerce_wavelength(cfg.opto_power_wavelength)]
                powers = list(range(0, 101, 20))
                steps = self.laser.build_optogenetic_protocol(
                    wavelengths=wavelengths,
                    powers=powers,
                    randomize_target="power",
                    stabilize_time=stabilize_time,
                    off_time=off_time,
                    on_time=on_time,
                    replicates=replicates,
                    randomize=True,
                )
                _run_steps(steps, rate_hz)

        if not results:
            self.warning("No optogenetic protocol flags enabled")

        self.amplifier.voltage_clamp()
        self.info("finished running optogenetic protocol")
        if len(results) == 1:
            return results[0]
        return results
    
    def run_optogenetic_protocol(self, protocol_params: dict | None = None):
        """
        Run optogenetic protocols based on configuration or explicit parameters.
        """
        if self.laser is None:
            self.warning("No laser configured; skipping optogenetic protocol.")
            return None
        if not hasattr(self.daq, "getDataFromOptogeneticProtocol"):
            self.warning("DAQ does not support optogenetic protocol capture.")
            return None

        self.info("Running optogenetic protocol")
        self.amplifier.voltage_clamp()
        self.sleep(0.25)
        holding = float(self.protocol_config.vclamp_hold)
        self.amplifier.set_holding(holding)
        self.info(f'holding at {holding} mV')
        self.sleep(0.25)

        results = []

        color_cycle = ["red", "green", "cyan", "uv", "blue", "infrared"]

        def _coerce_wavelength(value):
            if isinstance(value, str):
                return value.strip().lower()
            try:
                idx = int(value)
            except (TypeError, ValueError):
                return value
            if idx == 7:
                return "off"
            if idx <= 0:
                idx = 1
            return color_cycle[(idx - 1) % len(color_cycle)]

        def _run_steps(steps, rate_hz):
            result = self.daq.getDataFromOptogeneticProtocol(
                laser=self.laser,
                protocol_steps=steps,
                rate_hz=rate_hz,
            )
            results.append(result)
            self.sleep(0.25)

        if protocol_params is not None:
            randomize_target = protocol_params.get("randomize_target", protocol_params.get("mode", "wavelength"))
            raw_wavelengths = list(protocol_params.get("wavelengths", ["green"]))
            wavelengths = [_coerce_wavelength(value) for value in raw_wavelengths]
            steps = self.laser.build_optogenetic_protocol(
                wavelengths=wavelengths,
                powers=list(protocol_params.get("powers", [50])),
                randomize_target=randomize_target,
                stabilize_time=float(protocol_params.get("stabilize_time", 1.0)),
                off_time=float(protocol_params.get("off_time", 0.1)),
                on_time=float(protocol_params.get("on_time", 0.01)),
                replicates=int(protocol_params.get("replicates", 1)),
                randomize=bool(protocol_params.get("randomize", True)),
                power_divisor=float(protocol_params.get("power_divisor", 1.0)),
            )
            _run_steps(steps, int(protocol_params.get("rate_hz", 50_000)))
        else:
            cfg = self.protocol_config
            stabilize_time = float(cfg.opto_stabilize_time)
            off_time = float(cfg.opto_off_time)
            on_time = float(cfg.opto_on_time)
            replicates = int(cfg.opto_replicates)
            rate_hz = 50_000

            if cfg.opto_random_wavelength_protocol:
                wavelengths = ["red", "green", "cyan", "uv", "blue", "infrared"]
                powers = [float(cfg.opto_wavelength_power)]
                steps = self.laser.build_optogenetic_protocol(
                    wavelengths=wavelengths,
                    powers=powers,
                    randomize_target="wavelength",
                    stabilize_time=stabilize_time,
                    off_time=off_time,
                    on_time=on_time,
                    replicates=replicates,
                    randomize=True,
                )
                _run_steps(steps, rate_hz)

            if cfg.opto_random_power_protocol:
                wavelengths = [_coerce_wavelength(cfg.opto_power_wavelength)]
                powers = list(range(0, 101, 20))
                steps = self.laser.build_optogenetic_protocol(
                    wavelengths=wavelengths,
                    powers=powers,
                    randomize_target="power",
                    stabilize_time=stabilize_time,
                    off_time=off_time,
                    on_time=on_time,
                    replicates=replicates,
                    randomize=True,
                )
                _run_steps(steps, rate_hz)

        if not results:
            self.warning("No optogenetic protocol flags enabled")

        self.amplifier.voltage_clamp()
        self.info("finished running optogenetic protocol")
        if len(results) == 1:
            return results[0]
        return results
    # endregion

    # region Observations and measurements
    def observe(self, *, fields=None, num_measurements=None, interval=None, phase=None, evidence=None,
                raw_resistance=False):
        """Call requested models directly, then collect and record requested telemetry."""
        requested = tuple(dict.fromkeys(Observer.default_fields if fields is None else fields))
        if isinstance(fields, str) or set(requested) - Observer.available_fields:
            raise ValueError("Unknown observation fields")
        if num_measurements is not None and (isinstance(num_measurements, bool)
                or not isinstance(num_measurements, Integral) or num_measurements < 1):
            raise ValueError("num_measurements must be a positive integer")
        if interval is not None and (isinstance(interval, bool) or not isinstance(interval, Real)
                or not np.isfinite(interval) or interval < 0):
            raise ValueError("interval must be finite and non-negative")
        if evidence is None:
            evidence = self.infer_frame(requested, frame_context=getattr(phase, "frame_context", None))
        evidence = dict(evidence)
        at = evidence.get("frame_retrieval_started_at")
        if at is None:
            at = evidence.get("frame_available_at")
        age = None if at is None else time.monotonic() - at
        evidence.update(age_s=age, stale=age is None or not 0 <= age <= 1.)
        options = {"raw_resistance": True} if raw_resistance else {}
        return self.observer.observe(fields=requested, num_measurements=num_measurements,
                                     interval=interval, phase=phase, evidence=evidence, **options)

    def infer_frame(self, fields, frame_context=None):
        """Evaluate requested existing models on one frame; Hunt may schedule this call."""
        requested = set(fields)
        evidence = {}
        visual = set(requested) & {"pipette_image_xy", "pipette_defocus_um", "cell_detections", "deep_learning"}
        if visual or "camera_image" in requested:
            stage, unit = self.calibrated_stage, self.calibrated_unit
            camera = stage.camera
            frame_id, stamp, frame, timing = camera.last_raw_frame_data(include_timing=True) or (None, None, None, {})
            evidence = dict(source_frame=frame_id, frame_acquired_at=timing.get("acquired_at"),
                frame_retrieval_started_at=timing.get("retrieval_started_at"),
                frame_available_at=timing.get("available_at"), source_image_shape=None if frame is None else frame.shape,
                _source_image=frame, source_positions={}, source_position_metadata={}, positions_sampled_at=None,
                pipette_position=None, pipette_focus=None, cell_detections=None, status={}, confidence={}, errors={})
            if visual and frame is not None:
                image = np.asarray(frame).copy()
                evidence["_source_image"] = image
                context = frame_context or {}
                if context:
                    evidence["positions_sampled_at"] = time.monotonic()
                for name, read in context.items():
                    evidence["source_positions"][name] = read()
                    evidence["source_position_metadata"][name] = dict(acquired_at=None,
                        read_completed_at=time.monotonic(), timestamp_basis="inference_time_context")
                readers = {}
                if visual & {"pipette_image_xy", "pipette_defocus_um", "deep_learning"}:
                    detector = unit.pipetteCalHelper.pipetteDetector
                    focus_requested = bool(visual & {"pipette_defocus_um", "deep_learning"})
                    focuser = unit.pipetteFocusHelper.pipetteFocuser if focus_requested else None
                    shared_depth = (focuser is not None and getattr(focuser, "detector", None) is detector
                        and getattr(focuser.get_pipette_focus_value, "__func__", None)
                        is PipetteFocuser2.get_pipette_focus_value)
                    if visual & {"pipette_image_xy", "deep_learning"} or shared_depth:
                        readers["pipette_detector"] = detector.detect_pipette_details
                    if focus_requested and not shared_depth:
                        readers["pipette_focuser"] = focuser.get_pipette_focus_value
                        evidence["confidence"]["pipette_focuser"] = None
                if visual & {"cell_detections", "deep_learning"}:
                    readers["cell_detector"] = stage.cellDetectHelper._ensure_detector
                for name, read in readers.items():
                    try:
                        if name == "cell_detector":
                            read = read().detect_cells
                        result = read(image.copy())
                        if name == "pipette_detector":
                            result = result or {}
                            evidence["pipette_position"] = result.get("tip_xy")
                            evidence["confidence"]["pipette_detector"] = result.get("box_confidence")
                            if shared_depth:
                                evidence["pipette_focus"] = result.get("z_um")
                                evidence["confidence"]["pipette_focuser"] = result.get("z_confidence")
                                evidence["status"]["pipette_focuser"] = "ok" if result.get("z_um") is not None else "no_detection"
                            result = result.get("tip_xy")
                        elif name == "pipette_focuser":
                            evidence["pipette_focus"] = result
                        else:
                            evidence["cell_detections"] = result
                        evidence["status"][name] = "ok" if result is not None else "no_detection"
                    except Exception as error:
                        evidence["status"][name], evidence["errors"][name] = "error", str(error)
                        if name == "pipette_focuser":
                            evidence["pipette_focus"] = None
                        elif name == "pipette_detector" and shared_depth:
                            evidence["status"]["pipette_focuser"] = "error"
                            evidence["errors"]["pipette_focuser"] = str(error)
                            evidence["confidence"]["pipette_focuser"] = None
        return evidence

    def resistanceRamp(self, num_measurements=5, interval=0.200):
        return self._safe_average(
            self.daq.resistance, num_measurements, interval
        )

    def accessRamp(self, num_measurements=5, interval=0.200):
        return self._safe_average(
            self.daq.accessResistance, num_measurements, interval
        )

    def capacitanceRamp(self, num_measurements=5, interval=0.200):
        return self._safe_average(
            self.daq.capacitance, num_measurements, interval
        )

    def _safe_average(self, read_fn, num_measurements: int = 5, interval: float = 0.200):
        """Return the mean of *valid* samples from *read_fn*.

        * Skips any reading that is ``None``, ``NaN``, or negative.
        * If **every** reading in a window is invalid, run ``_adjustTrace``
          **once per window** and retry.
        * Retries *max_windows* times (default = 3).  After that, raises.

        Args:
            read_fn (callable):
                Function that returns a measurement value.
            num_measurements (int):
                Number of samples per averaging window.
            interval (float):
                Time interval (seconds) between measurements.

        Returns:
            float:
                Mean of valid readings.

        Raises:
            RuntimeError:
                If all readings are invalid after maximum retries.
        """
        max_windows = 3
        for attempt in range(max_windows):
            readings = []
            for _ in range(num_measurements):
                if self.abort_requested:
                    raise AutopatchError("Seal attempt aborted.")
                val = read_fn()
                # Guard against NaN/None without raising TypeError on None
                if val is None or (isinstance(val, (float, np.floating)) and np.isnan(val)):
                    # Keep log format identical - use debug so existing info/print lines stay untouched
                    self.debug("_safe_average: invalid reading skipped (NaN/None)")
                elif isinstance(val, (int, float, np.integer, np.floating)) and val < 0:
                    self.debug("_safe_average: invalid reading skipped (negative)")
                else:
                    readings.append(val)
                self.sleep(interval)

            if readings:
                return sum(readings) / len(readings)

            # No usable sample → adjust & retry
            self.info("All readings invalid – running _adjustTrace and retrying (_safe_average)")
            self._adjustTrace()

        # Exhausted retries
        raise RuntimeError(f"All measurements from {read_fn.__name__} returned None/NaN after {max_windows} retries")
    
    def _adjustTrace(self):
        '''
        Run capacitance compensation if fitting is failing.
        '''
        self.daq.setCellMode(False)
        self.amplifier.auto_fast_compensation()
        self.sleep(0.5)
        self.amplifier.auto_slow_compensation()
        self.sleep(0.5)
        self.daq.setCellMode(True)
        
    # endregion

    # region Rig movement and scanning
    @record_state("locate_cell")
    def locate_cell(self, cell):
        """Localize the pipette at slice_start_distance above the cell."""
        # move stage and pipette to safe space
        self.info("Moving to safe space")
        self.move_to_safe_space()
        self.info("Setting pressure to 100 mbar")
        self.pressure.set_pressure(100)
        # move to home space
        self.info("Moving to home space")
        self.move_to_home_space()
        # center pipette on cell xy 
        self.info("Centering pipette")
        self.fine_calibrate_pipette()
        
        # Use the same starting distance for every cell type.
        cell_pos, cell_img, pos = cell
        cell_distance = self.config.slice_start_distance

        self.info(f" Moving to Cell position: {cell_pos}") 
        # moving stage to xy position of cell
        # home position
        cell_pos_planar = np.array([cell_pos[0], cell_pos[1], 0])
        self.calibrated_stage.safe_move(np.array(cell_pos_planar))
        self.calibrated_stage.wait_until_still()
        # move pipette to xy position of cell
        stage_pos = self.calibrated_stage.pixels_to_um(self.calibrated_stage.reference_position())
        # print(f"Stage position: {stage_pos}")
        disp = np.zeros(3)
        disp[0] = stage_pos[0] - self.home_stage_position[0]
        disp[1] = stage_pos[1] - self.home_stage_position[1]
        disp[2] = 0
        # print(f"Disp: {disp}")
        # center pipette on cell xy 
        pipette_disp = self.calibrated_unit.rotate(disp,2)
        self.calibrated_unit.relative_move(pipette_disp)
        self.calibrated_unit.wait_until_still() 

       
        # Reduce speed near the cell.
        self.calibrated_stage.set_max_speed(self.config.max_locate_speed)
        self.calibrated_unit.set_max_speed(self.config.max_locate_speed)

        self.fine_calibrate_pipette()
        zdist_cell = self.home_stage_position[2] - cell_pos[2]
        self.move_group_down(-zdist_cell/2)# on real rig
        self.sleep(0.1)
        self.fine_calibrate_pipette()
        second = zdist_cell/2 + cell_distance
        self.move_group_down(-second)
        self.sleep(0.1)
        self.fine_calibrate_pipette()

        self.calibrated_stage.set_max_speed(10000)
        self.calibrated_unit.set_max_speed(100000)
        self.info("Located Cell")
        self.success_requested = True
        self.success_if_requested()

    @record_state("scan_area")
    def scan_area(self, speed=None):
        '''
        Scan the currently selected plate area using stored corners.
        '''
        self.calibrated_stage.scan_area(speed=speed)
        self.success_requested = True
        self.success_if_requested()

    def move_stage_to_cell(self, cell):
        '''
        Moves the stage to the XY position of the target cell.
        '''
        if cell is None:
            raise AutopatchError("No cell given to move stage to")
        if not self.calibrated_stage.calibrated:
            raise AutopatchError("Stage not calibrated")

        cell_pos = None
        cell_array = np.asarray(cell)
        if cell_array.shape == (3,) and np.issubdtype(cell_array.dtype, np.number):
            cell_pos = cell_array
        elif isinstance(cell, (tuple, list)) and len(cell) > 0:
            cell_pos = np.asarray(cell[0])

        if cell_pos is None or cell_pos.size < 2:
            raise AutopatchError("Cell position missing XY coordinates")

        self.info(f" Moving to Cell position: {cell_pos}")
        cell_pos_planar = np.array([cell_pos[0], cell_pos[1], 0])
        self.calibrated_stage.safe_move(np.array(cell_pos_planar))
        self.calibrated_stage.wait_until_still()

    def move_to_safe_space(self):
        '''
        Moves the pipette to the safe space.

        Raises:
            ValueError: If safe_position is not set.
        '''
        if self.safe_position is None:
            raise ValueError('Safe position has not been set')


        try:
            # Extract individual coordinates from the safe position
            safe_x, safe_y, safe_z = self.safe_position
            safe_stage_x, safe_stage_y, safe_microscope_z = self.safe_stage_position
            self.info(f"Moving to safe space: {safe_x}, {safe_y}, {safe_z}")

            # Step 0: Move the microscope to the safe position
            logging.debug(f'Moving microscope to safe position value: Z={safe_microscope_z}')
            self.microscope.absolute_move(safe_microscope_z)
            self.microscope.wait_until_still()  # Ensure movement completes
            # Step 1: Move the stage to the safe position
            self.calibrated_stage.absolute_move([safe_stage_x,safe_stage_y])
            self.calibrated_stage.wait_until_still()
            # # step 1.5: move pipette up if at cleaning position:
            # if self.cleaning_bath_position is not None and self.calibrated_unit.position()[2] == self.cleaning_bath_position[2]:
            #     self.calibrated_unit.relative_move(-500, axis=2)
            # Step 2: Move Y axis first to align with the safe position value
            logging.debug(f'Moving Y axis to safe position value: {safe_y}')
            self.calibrated_unit.absolute_move(safe_y, axis=1)
            self.calibrated_unit.wait_until_still()  # Ensure movement completes

            # Step 3: Simultaneously move X and Z axes to reach the safe position
            logging.debug(f'Moving X and Z axes to safe position values: X={safe_x}, Z={safe_z}')
            self.calibrated_unit.absolute_move_group([safe_x,safe_y,safe_z], [0,1,2])
            self.calibrated_unit.wait_until_still()  # Ensure movement completes

        finally:
            pass
        
    def move_to_home_space(self):
        '''
        Moves the pipette and stage to the home space.

        Raises:
            ValueError: If home_position is not set.
        '''
        if self.home_position is None:
            raise ValueError('Home position has not been set')

        try:
            # # Extract individual coordinates from the home position
            home_x, home_y, home_z = self.home_position
            stage_home_x, stage_home_y, microscope_home_z = self.home_stage_position
            # step 0: move the microscope to the home position
            logging.debug(f'Moving microscope to home position value: Z={microscope_home_z}')
            self.microscope.absolute_move(microscope_home_z)
            self.microscope.wait_until_still()
            # self.sleep(0.5)
            # Step 1: Move the stage to the home position
            logging.debug(f'Moving stage to home position values: X={stage_home_x}, Y={stage_home_y}')
            self.calibrated_stage.absolute_move([stage_home_x,stage_home_y])
            self.calibrated_stage.wait_until_still()
            # # step 1.5: move pipette up if at cleaning position:
            # if self.cleaning_bath_position is not None and self.calibrated_unit.position()[2] == self.cleaning_bath_position[2]:
            #     self.calibrated_unit.relative_move(-500, axis=2)
            # Step 2: Move Y axis first to align with the home position value
            logging.debug(f'Moving Y axis to home position value: {home_y}')
            self.calibrated_unit.absolute_move(home_y, axis=1)
            self.calibrated_unit.wait_until_still()  # Ensure movement completes

            # Step 3: Simultaneously move X and Z axes to reach the home position
            logging.debug(f'Moving X and Z axes to home position values: X={home_x}, Z={home_z}')
            self.calibrated_unit.absolute_move_group([home_x,home_y,home_z], [0,1,2])
            self.calibrated_unit.wait_until_still()  # Ensure movement completes

        finally:
            pass

    def move_group_down(self,dist = 25):
        '''
        Moves the microsope and manipulator down by input distance in the z axis
        
        Args:
            dist (float): Distance to move in micrometers. Defaults to 25 µm.
        '''

        self.info('MOVING GROUP DOWN')

        try:
            self.microscope.relative_move(dist)
            self.microscope.wait_until_still()
            self.calibrated_unit.relative_move(dist, axis=2)
            self.calibrated_unit.wait_until_still(2)
        finally:
            pass
    
    def move_group_up(self,dist = 25):
        '''
        Moves the microscope and manipulator up by input distance in the z axis
        
        Args:
            dist (float): Distance to move in micrometers. Defaults to 25 µm.
        '''
        self.info('MOVING GROUP UP')
    
        try:
            self.microscope.relative_move(-dist)
            self.microscope.wait_until_still()
            self.calibrated_unit.relative_move(-dist, axis=2)
            self.calibrated_unit.wait_until_still(2)
        finally:
            pass

    def move_group_in_x(self,dist = 25):
        '''
        Moves the pipette and stage in x axis by input distance
        '''

        try:
            self.calibrated_unit.relative_move(dist, axis=0)
            self.calibrated_unit.wait_until_still(0)
            self.calibrated_stage.relative_move(dist, axis=0)
            self.calibrated_stage.wait_until_still(0)
        finally:
            pass

    def move_group_in_y(self,dist = 500):
        '''
        Moves the pipette and stage in y axis by input distance
        '''

        try:
            self.calibrated_unit.relative_move(dist, axis=1)
            self.calibrated_unit.wait_until_still(1)
            #rotate for pipette motion in around z
            self.calibrated_stage.relative_move(dist, axis=1)
            self.calibrated_stage.wait_until_still(1)
        finally:
            pass

    def move_group_in_x(self,dist = 25):
        '''
        Moves the pipette and stage in x axis by input distance

        Args:
            dist (float): Distance to move in micrometers. Defaults to 25 µm.
        '''
    
        try:
            self.calibrated_unit.relative_move(dist, axis=0)
            self.calibrated_unit.wait_until_still(0)
            self.calibrated_stage.relative_move(dist, axis=0)
            self.calibrated_stage.wait_until_still(0)
        finally:
            pass
    def move_group_in_y(self,dist = 500):
        '''
        Moves the pipette and stage in y axis by input distance
        
        Args:
            dist (float): Distance to move in micrometers. Defaults to 500 µm.
        '''
    
        try:
            self.calibrated_unit.relative_move(dist, axis=1)
            self.calibrated_unit.wait_until_still(1)
            #rotate for pipette motion in around z
            self.calibrated_stage.relative_move(dist, axis=1)
            self.calibrated_stage.wait_until_still(1)
        finally:
            pass
    
    def move_pipette_up(self, dist = 5000):
        '''
        Moves the pipette up by input distance in the z axis
        
        Args:
            dist (float): Distance to move in micrometers. Defaults to 5000 µm.
        '''
        try:
            self.calibrated_unit.relative_move(-dist, axis=2)
            self.calibrated_unit.wait_until_still(2)
        finally:
            pass
    # endregion

    # region Movement playback
    def test_movement(self, path: str, target_frequency: int = 12):
        """
        Moves the pipette and stage based on parsed data, updating positions every
        ``1/target_frequency`` seconds — all without relying on *pandas*.

        The file must be semicolon‑delimited with a header row::

            timestamp;st_x;st_y;st_z;pi_x;pi_y;pi_z

        Args:
            path (str): Path to semicolon-delimited CSV with columns:
                        'timestamp;st_x;st_y;st_z;pi_x;pi_y;pi_z'.
            target_frequency (int): Frequency in Hz to update positions. Defaults to 12 Hz.
        """
        # --- Step 1 — Rapid parse with csv.DictReader -------------------------
        data = {h: [] for h in (
            'timestamp', 'st_x', 'st_y', 'st_z', 'pi_x', 'pi_y', 'pi_z'
        )}
        with open(path, newline='') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                for key in data:
                    data[key].append(float(row[key]))

        # --- Step 2 — Down‑sample --------------------------------------------
        filtered = self._downsample_data(data, target_frequency)

        # --- Step 3 — Move to the initial position ---------------------------
        self.calibrated_stage.absolute_move([
            filtered['st_x'][0], filtered['st_y'][0], filtered['st_z'][0]
        ])
        self.calibrated_unit.absolute_move_group(
            [filtered['pi_x'][0], filtered['pi_y'][0], filtered['pi_z'][0]], [0, 1, 2]
        )

        self.stop_event = threading.Event()
        self.movement_thread = threading.Thread(
            target=self._movement_loop, args=(filtered,)
        )
        self.info('Movement Test started')
        self.movement_thread.start()

    def stop_movement(self):
        """Requests the movement loop to halt and waits for the thread to finish."""
        if getattr(self, 'stop_event', None):
            self.stop_event.set()
        if getattr(self, 'movement_thread', None):
            self.movement_thread.join()

    def _downsample_data(self, data: dict, target_frequency: int = 12) -> dict:
        """
        Return a *new* dict containing rows sampled at ``target_frequency`` Hz.
        
        Args:
            data (dict): Dictionary with keys 'timestamp', 'st_x', 'st_y', 'st_z', 'pi_x', 'pi_y', 'pi_z'.
            target_frequency (int): Desired sampling frequency in Hz. Defaults to 12 Hz.

        Returns:
            dict: New dictionary containing downsampled data.
        """
        timestamps = data['timestamp']
        t0 = timestamps[0]
        # Normalise to start at 0 seconds
        rel_time = [t - t0 for t in timestamps]

        interval = 1.0 / target_frequency
        max_time = rel_time[-1]
        target_times = [i * interval for i in range(int(max_time // interval) + 1)]

        filtered = {k: [] for k in data}
        idx = 0
        n = len(rel_time)

        for t in target_times:
            # Advance until the first record >= target time ("forward" merge rule)
            while idx < n and rel_time[idx] < t:
                idx += 1
            if idx == n:
                break
            for key in data:
                filtered[key].append(data[key][idx])

        # Replace timestamp column with zero‑based times
        filtered['timestamp'] = [ts - t0 for ts in filtered['timestamp']]
        self.info(f"Filtered data to {len(filtered['timestamp'])} rows")
        return filtered

    def _movement_loop(self, data: dict):
        """
        Executes calibrated moves at each timestamp in *data*.
        
        Args:
            data (dict): Downsampled dictionary of positions and timestamps.
        """
        start = time.perf_counter()
        count = len(data['timestamp'])

        for i in range(count):
            if self.stop_event.is_set():
                self.info('Movement Test stopped')
                break

            # Uncomment if stage moves also needed
            # self.calibrated_stage.absolute_move([
            #     data['st_x'][i], data['st_y'][i], data['st_z'][i]
            # ])
            self.calibrated_unit.absolute_move_group(
                [data['pi_x'][i], data['pi_y'][i], data['pi_z'][i]], [0, 1, 2]
            )

            target_time = data['timestamp'][i]
            while time.perf_counter() < start + target_time:
                if self.stop_event.is_set():
                    break

        self.info('Movement Test completed')
    # endregion

    # region Illumination and laser controls
    def toggle_shutter(self):
        """
        Toggles the lamp shutter open or closed.
        """
        # Toggle the Lamp shutter on or off.
        try:
            current_state = self.lamp.get_shutter_state()
            if current_state == 'open':
                self.lamp.close_shutter()
                self.info("Lamp shutter closed.")
            elif current_state == 'closed':
                self.lamp.open_shutter()
                self.info("Lamp shutter opened.")
            else:
                # If state is None or unexpected, default to opening
                self.warning(f"Unknown shutter state '{current_state}', defaulting to open.")
                self.lamp.open_shutter()
        except Exception as e:
            self.error(f"Error toggling shutter: {e}")

    def toggle_fluorescence(self):
        """
        Toggle a default fluorescence filter cube on or off.
        """
        current = self.lamp.get_filter()
        fluo = int(self.config.lamp)

        # Initialise one-time history store
        if not hasattr(self, "_prev_filter_slot"):
            self._prev_filter_slot = 1          # sensible default

        if current == fluo:
            # On fluorescence → return to previously stored slot
            target = self._prev_filter_slot or 1
        else:
            # Store current (if valid) and move to slot fluo
            if current is not None and current != fluo:
                self._prev_filter_slot = current
            target = fluo

        self.lamp.set_filter(target)
        self.sleep(0.1)

    def move_cube_left(self):
        """
        Moves the filter cube one slot to the left.
        """
        current = self.lamp.get_filter()

        if current is None:
            current = 1
        new_slot = max(1, current - 1)
        self.lamp.set_filter(new_slot)

    def move_cube_right(self):
        """
        Moves the filter cube one slot to the left.
        """
        current = self.lamp.get_filter()
        if current is None:
            current = 1
        new_slot = current + 1
        self.lamp.set_filter(new_slot)

    def toggle_laser_output(self):
        if self.laser is None:
            self.warning("No laser configured; skipping output toggle.")
            return
        try:
            self.laser.excite()
        except Exception as exc:
            self.error(f"Error toggling laser output: {exc}")

    def wavelength_down(self):
        self._step_laser_wavelength(-1)

    def wavelength_up(self):
        self._step_laser_wavelength(1)

    def _step_laser_wavelength(self, step: int):
        if self.laser is None:
            self.warning("No laser configured; skipping wavelength change.")
            return
        current = self.laser.get_wavelength()
        if current is None:
            target = 1
        elif isinstance(current, int):
            target = max(1, current + step)
        else:
            try:
                from enum import Enum
                if isinstance(current, Enum):
                    channels = [c for c in type(current) if getattr(c, "name", "") != "OFF"]
                    if not channels:
                        self.warning("Laser wavelength enum has no selectable channels.")
                        return
                    try:
                        idx = channels.index(current)
                    except ValueError:
                        idx = 0
                    target = channels[max(0, min(len(channels) - 1, idx + step))]
                else:
                    self.warning(f"Unsupported laser wavelength type: {type(current)}")
                    return
            except Exception as exc:
                self.error(f"Unable to step laser wavelength: {exc}")
                return

        self.laser.set_wavelength(target)
    # endregion

