import time
import csv
from enum import Enum
import numpy as np
from patcherbot.devices.amplifier.amplifier import Amplifier
from patcherbot.devices.amplifier.DAQ import DAQ, FakeDAQ, NiDAQ
from patcherbot.devices.manipulator.calibratedunit import CalibratedUnit, CalibratedStage
from patcherbot.devices.manipulator.microscope import Microscope
from patcherbot.devices.pressurecontroller import PressureController
from patcherbot.devices.lamp import Lamp
from patcherbot.devices.laser import Laser
from patcherbot.devices.manipulator.AgentHelper import AgentHelper
from patcherbot.utils.StateMachineLogger import StateMachineLogger, record_state
import collections
import logging
from datetime import datetime
import pickle
import os
from patcherbot.interface.patchConfig import PatchConfig
from patcherbot.interface.protocolConfig import ProtocolConfig

from .base import TaskController, RequestedSuccessException
import threading
# import locking package
from threading import Lock


class AutopatchError(Exception):
    """
    Exception raised for errors during the automatic patching process.
    """
    def __init__(self, message = 'Automatic patching error'):
        """
        Initialize the AutopatchError.

        Args:
            message (str, optional): Error message describing the failure.
        """
        self.message = message

    def __str__(self):
        """
        Return the string representation of the error.

        Returns:
            str: The error message.
        """
        return self.message


class AutoPatcher(TaskController):
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
        self._track_cell_ai_disabled_logged = False
        self._last_track_cell_status = None

        self._last_status_msg = None
        self._last_error_msg = None
        self._last_warning_msg = None
        self.last_log_time = 0
        

    def _microscope_z_um(self) -> float:
        """
        Get the current microscope Z position.

        Returns:
            float: Microscope Z position in micrometers.
        """
        return float(self.microscope.position())

        self._last_status_msg = None
        self._last_error_msg = None
        self._last_warning_msg = None
        self.last_log_time = 0
        

    def _microscope_z_um(self) -> float:
        """
        Get the current microscope Z position.

        Returns:
            float: Microscope Z position in micrometers.
        """
        return float(self.microscope.position())

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
            self._state_recorder = StateMachineLogger(
                base_path="experiments/Data/state_recorder_data/",
                attempt_id=self.attempt_counter
            )
        return self._state_recorder

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

    @record_state("find_pipette")
    def find_pipette(self):
        """
        Locate and center the pipette tip in the camera field of view using either
        a direct control strategy or an agent-based policy.
        """
        self.info("Finding pipette")
        # Only load the agent policy when running in Agent mode.
        if self.config.mode == 'Agent':
            self.agenthelper.prepare_model("find_pipette", allow_goal_placeholders=True)
        elif self.config.mode == 'Adaptive':
            self.info('Adaptive mode detected; skipping agent model load for find_pipette')
        else:
            self.info("Classic/Manual/Training mode detected; skipping agent model load for find_pipette")
        max_sleep_time = 0.005  # seconds (slowest polling)
        min_sleep_time = 0.001  # seconds (fastest polling)
        sleep_time = max_sleep_time
        command_speed_um_s = abs(float(getattr(self, "find_pipette_velocity_speed_um_s", 200.0)))
        if command_speed_um_s == 0:
            raise ValueError("find_pipette_velocity_speed_um_s must be non-zero.")
        non_agent_relative_move_threshold_um_s = 1000.0
        use_non_agent_relative_move = command_speed_um_s >= non_agent_relative_move_threshold_um_s
        use_velocity_prediction = bool(getattr(self, "velocity_prediction", False))
        if self.config.mode == "Agent":
            if use_velocity_prediction:
                self.info(
                    "Agent find_pipette action mode: velocity "
                    "(xy in px/s converted to um/s, z in um/s)."
                )
            else:
                self.info(
                    "Agent find_pipette action mode: displacement "
                    "(xy in px converted to um, z in um)."
                )

        def _log_timing(label: str, duration_s: float) -> None:
            """Lightweight timing logger for find_pipette stages."""
            # self.info(f"[find_pipette timing] {label}: {duration_s * 1000.0:.1f} ms")

        def _adaptive_sleep_time(goal_error_um: float, tol_um: float) -> float:
            """
            Error-scaled polling:
            - near goal   -> faster polling (min_sleep_time)
            - far from goal -> slower polling (max_sleep_time)
            """
            if goal_error_um is None or (not np.isfinite(goal_error_um)):
                return max_sleep_time
            far_error_um = max(float(tol_um) * 10.0, float(tol_um))
            ratio = float(np.clip(float(goal_error_um) / far_error_um, 0.0, 1.0))
            return min_sleep_time + (max_sleep_time - min_sleep_time) * ratio

        goal_needed = bool(self.goal_needed)
        random = bool(self.goal_random)

        camera = self.calibrated_stage.camera
        width = getattr(camera, "width", None)
        height = getattr(camera, "height", None)

        center_x = int(round(width / 2)) if isinstance(width, (int, float)) else 640
        center_y = int(round(height / 2)) if isinstance(height, (int, float)) else 640
        goal_center = np.array([center_x, center_y, 0.0], dtype=float)
        self.info(f"Using goal center at: {goal_center} (px)")

        goal = None
        if goal_needed:
            goal = goal_center.astype(np.float32)
            if random:
                offsets = np.random.randint(-300, 301, size=2)
                goal[:2] = goal[:2] + offsets.astype(np.float32)
                if isinstance(width, (int, float)) and width > 0:
                    max_x = max(int(width) - 1, 0)
                    goal[0] = float(np.clip(goal[0], 0, max_x))
                if isinstance(height, (int, float)) and height > 0:
                    max_y = max(int(height) - 1, 0)
                    goal[1] = float(np.clip(goal[1], 0, max_y))

        goal_display = goal_center if goal is None else np.array(
            [int(round(goal[0])), int(round(goal[1]))],
            dtype=int,
        )
        goal_display_tuple = (int(goal_display[0]), int(goal_display[1]))
        goal_error_target = goal.astype(float) if goal is not None else goal_center.astype(float)

        err = None
        action = None
        target_point = None
        velocity_start_pos_um = None
        velocity_direction = None
        velocity_distance_um = None
        velocity_start_time = None
        velocity_timeout_s = None
        velocity_opposite_direction_warned = False

        def _reset_velocity_motion(stop_motion: bool = False) -> None:
            nonlocal velocity_start_pos_um, velocity_direction, velocity_distance_um
            nonlocal velocity_start_time, velocity_timeout_s, velocity_opposite_direction_warned
            if stop_motion:
                self.calibrated_unit.stop()
            velocity_start_pos_um = None
            velocity_direction = None
            velocity_distance_um = None
            velocity_start_time = None
            velocity_timeout_s = None
            velocity_opposite_direction_warned = False

        while True:
            obs_start = time.perf_counter()
            observation = self.observe()
            # _log_timing("observation", time.perf_counter() - obs_start)
            curr_point = observation[0]

            if curr_point is None:
                self.warning("Pipette detector did not return a location; waiting for next frame")
                self.sleep(sleep_time)
                continue

            if isinstance(curr_point, np.ndarray):
                curr_point = curr_point.tolist()
            if len(curr_point) < 3 or any(value is None for value in curr_point[:3]):
                self.warning("Pipette detector returned incomplete coordinates; waiting for next frame")
                self.sleep(sleep_time)
                continue

            curr_array = np.asarray(curr_point[:3], dtype=float)
            if np.isnan(curr_array).any():
                self.warning("Pipette detector returned NaN coordinates; waiting for next frame")
                self.sleep(sleep_time)
                continue

            curr_point = tuple(float(coord) for coord in curr_array)
            curr_point_np = np.asarray(curr_point, dtype=float)
            camera = self.calibrated_stage.camera
            adaptive_mode = self.config.mode == 'Adaptive'
            should_act = self.config.mode == 'Agent'
            z_weight = 1.0
            tol_um = 2.0
            px_per_um = self.calibrated_unit.pixel_per_um()

            if adaptive_mode:
                xgerr_px = goal_error_target[0] - curr_point_np[0]
                ygerr_px = goal_error_target[1] - curr_point_np[1]
                zerr_um = -curr_point_np[2]  # drive defocus to 0

                dx_um = xgerr_px / px_per_um[0] if px_per_um and px_per_um[0] else np.nan
                dy_um = ygerr_px / px_per_um[1] if px_per_um and px_per_um[1] else np.nan
                gerr_um = float(np.sqrt((dx_um ** 2 + dy_um ** 2 + z_weight * (zerr_um ** 2)) / (2 + z_weight)))
                self.info(f' Goal error (um):{gerr_um}')

                if goal_needed and camera is not None:
                    camera.show_circle(
                        point=goal_display_tuple,
                        color=(255, 255, 255),
                        show_center=False,
                    )

                if gerr_um <= tol_um:
                    self.info('Pipette found')
                    self.calibrated_unit.stop()
                    if self.config.mode == 'Training':
                        self.info('Training mode: goal condition reached. Click Success or Abort to finish.')
                        while True:
                            self.sleep(0.1)
                    self.success_requested = True
                    self.success_if_requested()

                action = None
                target_point = None
                err = None
                act_start = time.perf_counter()
                xy_um = self.calibrated_unit.pixels_to_um_relative([xgerr_px, ygerr_px, 0])
                # target_um = self.calibrated_unit.position() + np.array([xy_um[0], xy_um[1], zerr_um])
                # self.calibrated_unit.absolute_move(target_um.tolist())
                # self.calibrated_unit.wait_until_still()
                move_um = np.array([xy_um[0], xy_um[1], zerr_um], dtype=float)
                move_distance_um = float(np.linalg.norm(move_um))
                if move_distance_um > 0:
                    if use_non_agent_relative_move:
                        self.calibrated_unit.relative_move_group(move_um.tolist())
                        self.calibrated_unit.wait_until_still()
                    else:
                        velocity = self.calibrated_unit.velocity_position_control(move_um, command_speed_um_s)
                        velocity_command_local = -np.asarray(velocity, dtype=float)
                        self.calibrated_unit.absolute_move_group_velocity(velocity_command_local.tolist())
                self.sleep(_adaptive_sleep_time(gerr_um, tol_um))
                continue

            if not should_act:
                xgerr_px = goal_error_target[0] - curr_point_np[0]
                ygerr_px = goal_error_target[1] - curr_point_np[1]
                zerr_um = -curr_point_np[2]  # drive defocus to 0

                dx_um = xgerr_px / px_per_um[0] if px_per_um and px_per_um[0] else np.nan
                dy_um = ygerr_px / px_per_um[1] if px_per_um and px_per_um[1] else np.nan
                gerr_um = float(np.sqrt((dx_um ** 2 + dy_um ** 2 + z_weight * (zerr_um ** 2)) / (2 + z_weight)))
                self.info(f" Goal error (um):{gerr_um}")

                if goal_needed and camera is not None:
                    camera.show_circle(
                        point=goal_display_tuple,
                        color=(255, 255, 255),
                        show_center=False,
                    )

                if gerr_um <= tol_um:
                    self.info("Pipette found")
                    self.calibrated_unit.stop()
                    if self.config.mode == "Training":
                        self.info("Training mode: goal condition reached. Click Success or Abort to finish.")
                        while True:
                            self.sleep(0.1)
                    self.success_requested = True
                    self.success_if_requested()

                action = None
                target_point = None
                err = None
                act_start = time.perf_counter()
                xy_um = self.calibrated_unit.pixels_to_um_relative([xgerr_px, ygerr_px, 0])
                # target_um = self.calibrated_unit.position() + np.array([xy_um[0], xy_um[1], zerr_um])
                # self.calibrated_unit.absolute_move(target_um.tolist())
                # self.calibrated_unit.wait_until_still()
                move_um = np.array([xy_um[0], xy_um[1], zerr_um], dtype=float)
                move_distance_um = float(np.linalg.norm(move_um))
                if move_distance_um > 0:
                    if use_non_agent_relative_move:
                        self.calibrated_unit.relative_move_group(move_um.tolist())
                        self.calibrated_unit.wait_until_still()
                    else:
                        velocity = self.calibrated_unit.velocity_position_control(move_um, command_speed_um_s)
                        velocity_command_local = -np.asarray(velocity, dtype=float)
                        self.calibrated_unit.absolute_move_group_velocity(velocity_command_local.tolist())
                # _log_timing("action_direct_pipette", time.perf_counter() - act_start)
                self.sleep(_adaptive_sleep_time(gerr_um, tol_um))
                continue

            if action is None:
                # preprocess goal by cropping  and rescaling to 85 by 85
                inf_start = time.perf_counter()
                agent_goal = goal
                if goal is not None:
                    agent = getattr(self.agenthelper, "agent", None)
                    if agent is not None:
                        try:
                            image = observation[2]
                            if image is not None and hasattr(agent, "_compute_frame_params"):
                                frame_shape = np.asarray(image).shape[:2]
                                frame_params = agent._compute_frame_params(frame_shape)
                                if frame_params:
                                    goal_array = np.asarray(goal, dtype=np.float32).copy()
                                    if goal_array.ndim >= 1 and goal_array.shape[-1] >= 2:
                                        scale_x = frame_params.get("scale_x", 1.0)
                                        scale_y = frame_params.get("scale_y", 1.0)
                                        offset_x = frame_params.get("offset_x", 0.0)
                                        offset_y = frame_params.get("offset_y", 0.0)
                                        goal_array[..., 0] = (goal_array[..., 0] - offset_x) * scale_x
                                        goal_array[..., 1] = (goal_array[..., 1] - offset_y) * scale_y
                                        if goal_array.shape[-1] < 3:
                                            goal_array = np.pad(goal_array, (0, 3 - goal_array.shape[-1]), constant_values=0)
                                        agent_goal = goal_array
                                        self.info(f"goal scaled: {agent_goal} um")
                        except Exception as exc:
                            self.warning(f"Goal preprocessing failed; using raw goal. Error: {exc}")
                action = self.agenthelper.run_inference(observation=observation, goal=agent_goal, is_demo=False)
                # _log_timing("inference_block", time.perf_counter() - inf_start)
                prediction_units = "velocity (xy px/s, z um/s)" if use_velocity_prediction else "displacement (xy px, z um)"
                self.info(f"pipette prediction: {action} [{prediction_units}]")

                if action is None:
                    self.warning("Model did not return an action; retrying inference")
                    if use_velocity_prediction:
                        _reset_velocity_motion(stop_motion=True)
                    err = None
                    self.sleep(sleep_time)
                    continue

                pred_offset_xy = np.asarray(action[:2], dtype=float)
                if pred_offset_xy.size < 2:
                    self.warning("Predicted offset missing coordinates; retrying inference")
                    if use_velocity_prediction:
                        _reset_velocity_motion(stop_motion=True)
                    action = None
                    target_point = None
                    err = None
                    self.sleep(sleep_time)
                    continue

                if np.isnan(pred_offset_xy).any():
                    self.warning("Predicted offset contains NaNs; retrying inference")
                    if use_velocity_prediction:
                        _reset_velocity_motion(stop_motion=True)
                    action = None
                    target_point = None
                    err = None
                    self.sleep(sleep_time)
                    continue

                if use_velocity_prediction:
                    # Velocity mode: model output is interpreted directly as [vx_px, vy_px, vz_um].
                    # Convert xy into manipulator-frame um/s; pass z through as-is.
                    z_velocity_um_s = 0.0 if len(action) < 3 or action[2] is None else float(action[2])
                    velocity_xy_um_s = self.calibrated_unit.pixels_to_um_relative(
                        [-pred_offset_xy[0], -pred_offset_xy[1], 0.0]
                    )
                    velocity_um_s = np.array(
                        [velocity_xy_um_s[0], velocity_xy_um_s[1], z_velocity_um_s],
                        dtype=float,
                    )
                    if not np.isfinite(velocity_um_s).all():
                        self.warning("Predicted velocity contains invalid values; retrying inference")
                        action = None
                        target_point = None
                        err = None
                        _reset_velocity_motion(stop_motion=True)
                        self.sleep(sleep_time)
                        continue
                    if float(np.linalg.norm(velocity_um_s)) == 0.0:
                        self.info("Predicted velocity is zero; stopping motion and requesting next action.")
                        action = None
                        target_point = None
                        err = None
                        _reset_velocity_motion(stop_motion=True)
                        self.sleep(sleep_time)
                        continue
                    self.calibrated_unit.relative_move_group_velocity(velocity_um_s.tolist())
                    action = None
                    target_point = None
                    err = None
                    _reset_velocity_motion(stop_motion=False)
                else:
                    # Displacement mode: treat model output as [dx_px, dy_px, dz_um], then do a relative move.
                    # Negate agent find_pipette Z output to match coordinate convention used by this rig.
                    z_component = -float(action[2]) if len(action) >= 3 and action[2] is not None else None
                    z_offset_um = -curr_point_np[2] if z_component is None else z_component
                    target_point_float = np.asarray(curr_point[:2], dtype=float) + pred_offset_xy
                    target_point_pixels = (target_point_float[0], target_point_float[1])
                    width = getattr(camera, "width", None)
                    height = getattr(camera, "height", None)
                    if width is not None and height is not None:
                        if not (0 <= target_point_pixels[0] < width and 0 <= target_point_pixels[1] < height):
                            self.warning(f"predicted point not on screen: {target_point_pixels}")
                            action = None
                            target_point = None
                            err = None
                            _reset_velocity_motion(stop_motion=True)
                            self.sleep(0.04)
                            continue
                    move_xy_um = self.calibrated_unit.pixels_to_um_relative(
                        [pred_offset_xy[0], pred_offset_xy[1], 0.0]
                    )
                    move_um = np.array([move_xy_um[0], move_xy_um[1], -z_offset_um], dtype=float)
                    if not np.isfinite(move_um).all():
                        self.warning("Predicted displacement contains invalid values; retrying inference")
                        action = None
                        target_point = None
                        err = None
                        self.sleep(sleep_time)
                        continue
                    move_distance_um = float(np.linalg.norm(move_um))
                    if move_distance_um == 0:
                        self.info("Predicted movement is zero; requesting next action.")
                        action = None
                        target_point = None
                        err = None
                        self.sleep(sleep_time)
                        continue
                    self.calibrated_unit.relative_move_group(move_um.tolist())
                    self.calibrated_unit.wait_until_still()
                    action = None
                    target_point = None
                    err = None
                    _reset_velocity_motion(stop_motion=False)

            xgerr = goal_error_target[0] - curr_point_np[0]
            ygerr = goal_error_target[1] - curr_point_np[1]
            zerr_um_goal = -curr_point_np[2]
            dx_um = xgerr / px_per_um[0] if px_per_um and px_per_um[0] else np.nan
            dy_um = ygerr / px_per_um[1] if px_per_um and px_per_um[1] else np.nan
            gerr = float(np.sqrt((dx_um ** 2 + dy_um ** 2 + z_weight * (zerr_um_goal ** 2)) / (2 + z_weight)))
            self.info(f" Goal error (um):{gerr}")
            loop_sleep_time = _adaptive_sleep_time(gerr, tol_um)

            if goal_needed and camera is not None:
                camera.show_circle(
                    point=goal_display_tuple,
                    color=(255, 255, 255),
                    radius=15,
                    show_center=False,
                )

            if gerr <= tol_um:
                self.info("Pipette found")
                _reset_velocity_motion(stop_motion=True)
                if self.config.mode == "Training":
                    self.info("Training mode: goal condition reached. Click Success or Abort to finish.")
                    while True:
                        self.sleep(0.1)
                self.success_requested = True
                self.success_if_requested()

            if target_point is not None and velocity_start_pos_um is not None and velocity_direction is not None and velocity_distance_um is not None:
                current_position_um = np.asarray(self.calibrated_unit.position(), dtype=float)
                signed_traveled_um = float(np.dot(current_position_um - velocity_start_pos_um, velocity_direction))
                traveled_um = abs(signed_traveled_um)
                if traveled_um >= velocity_distance_um:
                    _reset_velocity_motion(stop_motion=True)
                    action = None
                    target_point = None
                    err = None
                    self.sleep(loop_sleep_time)
                    continue
                if (signed_traveled_um < 0) and (not velocity_opposite_direction_warned):
                    self.warning(
                        "Find-pipette velocity move is progressing opposite commanded direction; "
                        "using absolute displacement criterion."
                    )
                    velocity_opposite_direction_warned = True
                if velocity_start_time is not None and velocity_timeout_s is not None:
                    elapsed_s = time.perf_counter() - velocity_start_time
                    if elapsed_s > velocity_timeout_s:
                        self.warning(
                            f"Find-pipette velocity move timeout after {velocity_timeout_s:.2f}s "
                            f"(target {velocity_distance_um:.2f} um, traveled {signed_traveled_um:.2f} um signed)."
                        )
                        _reset_velocity_motion(stop_motion=True)
                        action = None
                        target_point = None
                        err = None
                        self.sleep(loop_sleep_time)
                        continue

            if target_point is not None:
                xerr = curr_point_np[0] - target_point[0]
                yerr = curr_point_np[1] - target_point[1]
                dx_um = xerr / px_per_um[0] if px_per_um and px_per_um[0] else np.nan
                dy_um = yerr / px_per_um[1] if px_per_um and px_per_um[1] else np.nan
                err = float(np.sqrt((dx_um ** 2 + dy_um ** 2) / 2.0))
                # self.info(f"total XY error (um): {err}")

                if err <= (tol_um / 2):
                    _reset_velocity_motion(stop_motion=True)
                    action = None
                    target_point = None
                    err = None
                    self.sleep(loop_sleep_time)
                    continue

            self.sleep(loop_sleep_time)

    @record_state("run_protocols")
    def run_protocols(self):
        """
        Execute a sequence of electrophysiological protocols based on the
        current protocol configuration.
        """
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
    
    def isrigready(self):
        """
        Validate that all required hardware components and calibration states
        are properly initialized before running an automated patching sequence.
        """
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

    @record_state("scan_area")
    def scan_area(self, speed=None):
        '''
        Scan the currently selected plate area using stored corners.
        '''
        self.calibrated_stage.scan_area(speed=speed)
        self.success_requested = True
        self.success_if_requested()

    @record_state("locate_cell") 
    def locate_cell(self, cell):
        '''
        Performs regional pipette localization to bring pipette above the cell.
        '''
         # regional pipette localization: 
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
        
        # move to cell_distance above cell.
        cell_pos, cell_img,pos = cell
        if self.config.cell_type == "Plate":
            cell_distance = self.config.cell_distance
        elif self.config.cell_type == "Slice":
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

       
 #drop speed to approach cell
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

        self.amplifier.start_patch()

        self.align(cell, cell_distance,self.config.use_centroid)
        
        if self.config.cell_type_toggle and self.config.cell_type == "Slice":
            self.clear_to_cell(cell)
            self.align(cell, self.config.cell_distance,self.config.use_centroid)

        
        self.calibrated_stage.set_max_speed(10000)
        self.calibrated_unit.set_max_speed(100000)
        self.info("Located Cell")
        self.success_requested = True
        self.success_if_requested()

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
        
        Args:
            cell (tuple):
                Cell data containing position and associated metadata.
            cell_distance (float):
                Desired vertical offset from the cell.
            use_centroid (bool):
                Whether to use centroid-based alignment for cell positioning.
        '''
        self.info("Aligning pipette to cell using imaging")
        cell_pos, _, _ = cell

        self.microscope.move_to_floor()
        self.microscope.wait_until_still()
        z_pos = self.microscope.position() / self.calibrated_unit.config.microscope_units_per_um
        zdistleft = z_pos - cell_pos[2]
        self.microscope.relative_move(-zdistleft)
        self.microscope.wait_until_still()

        if self.config.cell_type_toggle:
            self.info("centering on cell")
            self.calibrated_stage.center_on_cell(cell,use_centroid)
            self.calibrated_stage.wait_until_still()
            self.info(f"correcting pipette position, moving microscope by {zdistleft} um")
            self.microscope.relative_move(-cell_distance)
            self.microscope.wait_until_still()
            self.calibrated_unit.center_pipette()
            self.calibrated_unit.wait_until_still()
            self.microscope.relative_move(cell_distance)
            self.microscope.wait_until_still()

    def clear_to_cell(self, cell):
        '''
        Clears the pipette to the cell by moving down while checking resistance
        Moves the pipette down to cell plane and detects a cell using resistance measurements
        
        Args:
            cell (optional):
                Cell data used for tracking or validation.

        Raises:
            AutopatchError:
                If the rig is not ready or no cell is provided.
        '''
        self.info("Clearing to cell")
        self.isrigready()

        if self.rig_ready == False:
            raise AutopatchError("Rig not ready for clearing to cell")
        
        if cell is None:
            raise AutopatchError("No cell given to patch!")
        
        # if a slice, push pipette into slice from above surface, just about 20um above cell of interest

        if self.config.cell_type_toggle and self.config.cell_type == "Slice":
            self.info("Moving pipette to slice position")
            speed = [0,0,self.config.max_clearing_speed]
            cell_hover_pos =  self.config.cell_distance - self.config.slice_start_distance
            # move the stage up to the cell hover position
            self.microscope.relative_move(-self.config.cell_distance)
            start_pos = self.calibrated_unit.position()
            self.calibrated_unit.absolute_move_group_velocity(speed)
            self.info(f"Cell hover position: {cell_hover_pos} um")
            while start_pos[2] - self.calibrated_unit.position()[2] > cell_hover_pos and not self.abort_requested:
                self.sleep(0.1)
            self.calibrated_unit.stop()
        
    @record_state("hunt_cell")
    def hunt_cell(self,cell = None):
        '''
        Moves the pipette down to cell plane and detects a cell using resistance measurements
        
        Args:
            cell (optional):
                Cell data used for tracking or validation.

        Raises:
            AutopatchError:
                If the rig is not ready or no cell is provided.
        '''
        self.info("Hunting for cell")
        
        self.isrigready()

        if self.rig_ready == False:
            raise AutopatchError("Rig not ready for cell hunting")
        
        if cell is None:
            raise AutopatchError("No cell given to patch!")
        
        # # #ensure "near cell" pressure
        self.info(f"Setting pressure to {self.config.pressure_near} mbar")
        self.pressure.set_pressure(self.config.pressure_near)
        self.sleep(3) #let the resistance stabilize

        lastResDeque = collections.deque(maxlen=5)
        # get initial resistance
        daqResistance = self.daq.resistance()
        lastResDeque.append(daqResistance)

        # move pipette down at 10um/s and check resistance every 40 ms
        # get starting position 
        start_pos = self.calibrated_unit.position()

        self.first_res = self.resistanceRamp()

        self.info(f"Initial resistance: {self.first_res}")
        self.info(f"{self.config.mode}: starting hunt")

        if self.config.mode == 'Classic':
            speed = [0, 0, self.config.max_descent_speed]

            self.calibrated_unit.absolute_move_group_velocity(speed)
            self.info(f"moving pipette at: {speed} um/s")
            autoHunt=True
        elif self.config.mode == 'Adaptive':
            speed = [0, 0, self.config.max_descent_speed]

            self.calibrated_unit.absolute_move_group_velocity(speed)
            self.info(f'moving pipette at: {speed} um/s')
            autoHunt=True
        elif self.config.mode == 'Agent':
            #prepare model
            # cell_pos, cell_img,goal_pos = cell
            # goal_res  = self.first_res + self.config.cell_R_increase*1e6
            self.agenthelper.prepare_model("hunt")
            autoHunt = True
        else:
            autoHunt = False

        self._track_cell_ai_disabled_logged = False
        self._last_track_cell_status = None
        if self.config.track_cell and bool(self.calibrated_stage.config.use_ai_features):
            cell_track_helper = getattr(self.calibrated_stage, "cellTrackHelper", None)
            if cell_track_helper is not None:
                cell_track_helper.reset_tracking()
        training_mode = self.config.mode == "Training"
        enforce_max_hunt_distance = self.config.mode != "Training"
        if not enforce_max_hunt_distance:
            self.info(
                "Training mode: max hunt distance check disabled; waiting for resistance threshold."
            )
        hunt_termination_reason = None
        cell_detected = self._isCellDetected(lastResDeque=lastResDeque,cellThreshold = self.config.cell_R_increase)
        last_training_threshold_status = None
        if training_mode:
            last_training_threshold_status = cell_detected
            self.info(f"Training mode: resistance threshold achieved: {cell_detected}")
        while not cell_detected and self.abort_requested == False:
            # if autoHunt:
            #     try: 
            #         model_input = self.observe()
            #         pos = self.agenthelper.run_inference(model_input)
            #         st_pos = pos[:3]
            #         pi_pos = pos[3:]
            #         self.info(f"pipette command: {pi_pos},data type {type(pi_pos)}")          
            #     except: 
            #         self.error("Error in prediction, stopping autopatch")
            #         break
                #     st_pos = [0,0,0]
                #     pi_pos = [0,0,0]
                # # self.info(f"stage command: {st_pos}")
                # self.info(f"pipette command: {pi_pos}")
                # # self.calibrated_stage.relative_move_group(st_pos)
                # # # simple debug
                # # pip_disp = [0,0,1]
                # self.calibrated_unit.relative_move(pi_pos)

            curr_pos = self.calibrated_unit.position()
            if (
                enforce_max_hunt_distance
                and abs(curr_pos[2] - start_pos[2]) >= (int(self.config.max_distance))
            ):
                # we have moved expected um down and still no cell detected
                moved_distance = abs(curr_pos[2] - start_pos[2])
                hunt_termination_reason = (
                    "Cell not detected before reaching max hunt distance "
                    f"({moved_distance:.1f} um >= {float(self.config.max_distance):.1f} um)."
                )
                self.info(hunt_termination_reason)
                self.calibrated_unit.stop()
                self.calibrated_stage.stop()
                self.microscope.stop()
                break
            elif cell_detected:
                if autoHunt:
                    self.calibrated_unit.stop()
                    self.calibrated_stage.stop()
                    self.microscope.stop()

                self.info("Cell Detected")
                break

            self.track_cell(cell)

            self.sleep(0.04)
            lastResDeque.append(daqResistance)
            daqResistance = self.daq.resistance()
            cell_detected = self._isCellDetected(lastResDeque=lastResDeque,cellThreshold=self.config.cell_R_increase)
            if training_mode and cell_detected != last_training_threshold_status:
                self.info(f"Training mode: resistance threshold achieved: {cell_detected}")
                last_training_threshold_status = cell_detected

        self.calibrated_stage.stop()
        self.calibrated_unit.stop()
        self.microscope.stop()
        if cell_detected:
            self.info("Cell Detected")
            if training_mode:
                self.info("Training mode: waiting for manual Success or Abort.")
                while True:
                    self.sleep(0.1)
            self.success_requested = True
            self.success_if_requested()
        elif hunt_termination_reason is not None:
            if self.config.mode == "Training":
                self.info(
                    "Training mode: hunt ended without cell detection. "
                    "Click Success or Abort to finish."
                )
                while True:
                    self.sleep(0.1)
            raise AutopatchError(hunt_termination_reason)
        elif self.abort_requested:
            self.abort_if_requested()

    def track_cell(self, cell):
        '''
        Track the cell during hunting and return its current position in pixels.
        '''
        # TODO will add another condition to check if cell and pipette have moved away from each other based on the mask and original image.
        if not self.config.track_cell:
            return None

        ai_tracking_enabled = bool(self.calibrated_stage.config.use_ai_features)
        if ai_tracking_enabled:
            position, disp = self.calibrated_stage.get_cell_position(
                cell,
                use_centroid=self.config.use_centroid,
                tracking_mode=self.config.tracking_mode,
                track_max_fast_jump_px=self.config.track_max_fast_jump_px,
            )
            status = {}
            cell_track_helper = getattr(self.calibrated_stage, "cellTrackHelper", None)
            if cell_track_helper is not None:
                status = getattr(cell_track_helper, "last_tracking_status", {}) or {}
            method = status.get("method")
            status_name = status.get("status")
            if position is not None and disp is not None:
                status_label = f"{method}/{status_name}" if method else str(status_name)
                self.info(f"cell tracking: {status_label}")
                self.info(f"cell displacement: {disp} px")
                self.info(f"cell position: {position} px")
                self._last_track_cell_status = status_name
                return position
            else:
                if self._last_track_cell_status != status_name:
                    self.info("lost track of cell")
                    self._last_track_cell_status = status_name
                return None

        if not self._track_cell_ai_disabled_logged:
            self.info(
                "Track-cell is enabled, but calibration.use_ai_features is false; skipping AI cell tracking."
            )
            self._track_cell_ai_disabled_logged = True
        return None
    
    @record_state("escape")
    def escape(self):
            """   Safely terminate the patching process and reset hardware to a stable state."""
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
        
    def accessRamp(self, num_measurements=5, interval=0.200):
        """
        Measure access resistance using averaged DAQ readings.

        Args:
            num_measurements (int):
                Number of samples to average.
            interval (float):
                Time interval (seconds) between samples.

        Returns:
            float:
                Averaged access resistance value.
        """
        return self._safe_average(
            self.daq.accessResistance, num_measurements, interval
        )

    def resistanceRamp(self, num_measurements=5, interval=0.200):
        """
        Measure resistance using averaged DAQ readings.

        Args:
            num_measurements (int):
                Number of samples to average.
            interval (float):
                Time interval (seconds) between samples.

        Returns:
            float:
                Averaged resistance value.
        """
        return self._safe_average(
            self.daq.resistance, num_measurements, interval
        )

    def capacitanceRamp(self, num_measurements=5, interval=0.200):
        """
        Measure capacitance using averaged DAQ readings.

        Args:
            num_measurements (int):
                Number of samples to average.
            interval (float):
                Time interval (seconds) between samples.

        Returns:
            float:
                Averaged capacitance value.
        """
        return self._safe_average(
            self.daq.capacitance, num_measurements, interval
        )

    @record_state("gigaseal")
    def gigaseal(self):
        """
        requires **three consecutive**
        averaged-resistance windows ≥ target to declare success, reducing
        false positives from transient spikes.

        Raises:
            AutopatchError: If seal formation fails or deadline is exceeded.
        """
        # TODO: reimplement fake daq simulation thread
        autoPressure = (self.config.mode == 'Classic')
        adaptivePressure = (self.config.mode == 'Adaptive')
        agentPressure = (self.config.mode == 'Agent')
        agent_resistance_input_width = 0
        agent_resistance_history = collections.deque(maxlen=1)
        if agentPressure:
            self.info("Agent gigaseal mode detected; preparing gigaseal policy.")
            self.agenthelper.prepare_model("gigaseal")
            agent_resistance_input_width = self._agent_resistance_input_width()
            agent_resistance_history = collections.deque(
                maxlen=max(1, agent_resistance_input_width)
            )
        self.info(f"{self.config.mode}: Attempting to form gigaseal...")
        self.amplifier.auto_fast_compensation()
        self.sleep(1)
        self.daq.setCellMode(True)
        self.sleep(0.1)
        self.info("Collecting baseline resistance...")

        num_slope_samples = 5
        sample_interval = float(self.config.measurement_speed)

        avg_resistance = self.resistanceRamp(
            num_measurements=num_slope_samples,
            interval=sample_interval,
        )
        consecutive_success = 0

        self.pressure.set_ATM(atm=True)

        self.sleep(3)

        if autoPressure:
            currPressure = -5
            self.pressure.set_pressure(currPressure)
            self.pressure.set_ATM(atm=False)
            prevpressure = currPressure
            speed = 1
            bad_cell_count = 0
            # this is already negative, e.g. -30 mbar
            max_pressure = self.config.pressure_ramp_max
        elif adaptivePressure:
            currPressure = -5
            self.pressure.set_pressure(currPressure)
            self.pressure.set_ATM(atm=False)
            prevpressure = currPressure
            speed = 1
            bad_cell_count = 0
            # this is already negative, e.g. -30 mbar
            max_pressure = self.config.pressure_ramp_max

        holding_switched = False
        last_progress_time = time.time()
        last_agent_action = None
        observations_since_last_action = 0

        while not self.abort_requested:
            # Deadline check
            if time.time() - last_progress_time >= self.config.seal_deadline:
                raise AutopatchError(f"Seal attempt failed: resistance did not improve by at least {self.config.gigaseal_min_delta_R} MegaOhms by the {self.config.seal_deadline} second deadline.")

            prev_resistance = avg_resistance
            avg_resistance = self.resistanceRamp(
                num_measurements=num_slope_samples,
                interval=sample_interval,
            )

            delta_resistance = avg_resistance - prev_resistance
            rate_mohm_per_sec = delta_resistance / (num_slope_samples * sample_interval)

            if delta_resistance >= self.config.gigaseal_min_delta_R:
                last_progress_time = time.time()

            # ---------------------- auto-pressure logic ----------------------
            if agentPressure:
                observation = self.observe(include_pressure_state=True)
                self._attach_agent_resistance_input(
                    observation,
                    agent_resistance_history,
                    agent_resistance_input_width,
                )
                observation["observations_since_last_action"] = np.asarray(
                    [observations_since_last_action],
                    dtype=np.float32,
                )
                self.info(
                    "Gigaseal agent observation collected: "
                    f"resistance={float(observation['resistance'][0]):.3f} MΩ, "
                    f"actual_pressure={float(observation['pressure'][0]):.3f} mbar, "
                    f"setpoint={float(observation['commanded_pressure_mbar'][0]):.3f} mbar, "
                    f"atm={bool(observation['pressure_atm_state'][0])}, "
                    f"observations_since_last_action={int(observation['observations_since_last_action'][0])}"
                )
                action = self.agenthelper.run_inference(observation=observation, is_demo=False)
                self.info(f"Gigaseal agent raw action: {action}")
                if action is None:
                    self.warning("Gigaseal agent did not return an action; skipping pressure update for this iteration.")
                else:
                    action_array = np.asarray(action).reshape(-1)
                    if action_array.size < 1:
                        self.warning(
                            f"Gigaseal agent action must have at least 1 value; received shape {action_array.shape}."
                        )
                    else:
                        first_action_value = float(action_array[0])
                        if action_array.size == 1:
                            target_atm = bool(first_action_value >= 0.0)
                            if target_atm:
                                commanded_pressure = float(self.pressure.get_pressure())
                            else:
                                commanded_pressure = float(
                                    np.clip(first_action_value, float(self.config.pressure_ramp_max), -5.0)
                                )
                        else:
                            commanded_pressure = float(
                                np.clip(first_action_value, float(self.config.pressure_ramp_max), -5.0)
                            )
                            target_atm = bool(float(action_array[1]) >= 0.5)
                        self.info(
                            "Gigaseal agent decoded action: "
                            f"commanded_pressure={commanded_pressure:.3f} mbar, atm={target_atm}"
                        )

                        current_agent_action = (commanded_pressure, target_atm)
                        if current_agent_action != last_agent_action:
                            self.pressure.set_pressure(commanded_pressure)
                            self.pressure.set_ATM(atm=target_atm)
                            observations_since_last_action = 0
                            last_agent_action = current_agent_action
                        else:
                            observations_since_last_action += 1
            elif autoPressure:
                # adjust currPressure by ±5 based on rate_mohm_per_sec, speed, etc.
                increase_gate = self.config.increase_slope_gate
                constant_gate = self.config.constant_slope_gate
                decrease_gate = self.config.decrease_slope_gate

                increase_thresh = self.config.gigaseal_R / increase_gate
                constant_thresh = self.config.gigaseal_R / constant_gate
                decrease_thresh = self.config.gigaseal_R / decrease_gate

                if rate_mohm_per_sec < increase_thresh:
                    currPressure -= 5; speed = 3; max_pressure = self.config.pressure_ramp_max
                elif rate_mohm_per_sec <= constant_thresh:
                    speed = 1  # maintain
                elif rate_mohm_per_sec <= decrease_thresh:
                    max_pressure = self.config.pressure_ramp_max; currPressure += 5; speed = 3

                currPressure = min(currPressure, -5.0)
                currPressure = max(currPressure, self.config.pressure_ramp_max)

                if currPressure != prevpressure:
                    self.pressure.set_pressure(currPressure)
                    prevpressure = currPressure
                    self.sleep(5 / speed)

                if currPressure <= max_pressure:
                    self.pressure.set_ATM(True)
                    self.sleep(5)
                    testresistance = self.resistanceRamp(
                        num_measurements=num_slope_samples,
                        interval=sample_interval,
                    )
                    difference = testresistance - avg_resistance
                    self.info(f"Test resistance: {testresistance} MΩ; difference: {difference} MΩ")
                    if difference < 0:
                        bad_cell_count += 1
                        if bad_cell_count > 5:
                            raise AutopatchError("Bad cell detected")

                    currPressure = -5
                    self.pressure.set_pressure(currPressure)
                    self.pressure.set_ATM(atm=False)
            elif adaptivePressure:
                # adjust currPressure by ±5 based on rate_mohm_per_sec, speed, etc.
                increase_gate = self.config.increase_slope_gate
                constant_gate = self.config.constant_slope_gate
                decrease_gate = self.config.decrease_slope_gate

                increase_thresh = self.config.gigaseal_R / increase_gate
                constant_thresh = self.config.gigaseal_R / constant_gate
                decrease_thresh = self.config.gigaseal_R / decrease_gate

                if rate_mohm_per_sec < increase_thresh:
                    currPressure -= 5; speed = 3; max_pressure = self.config.pressure_ramp_max
                elif rate_mohm_per_sec <= constant_thresh:
                    speed = 1  # maintain
                elif rate_mohm_per_sec <= decrease_thresh:
                    max_pressure = self.config.pressure_ramp_max; currPressure += 5; speed = 3

                currPressure = min(currPressure, -5.0)
                currPressure = max(currPressure, self.config.pressure_ramp_max)

                if currPressure != prevpressure:
                    self.pressure.set_pressure(currPressure)
                    prevpressure = currPressure
                    self.sleep(5 / speed)

                if currPressure <= max_pressure:
                    self.pressure.set_ATM(True)
                    self.sleep(5)
                    testresistance = self.resistanceRamp(
                        num_measurements=num_slope_samples,
                        interval=sample_interval,
                    )
                    difference = testresistance - avg_resistance
                    self.info(f'Test resistance: {testresistance} MΩ; difference: {difference} MΩ')
                    if difference < 0:
                        bad_cell_count += 1
                        if bad_cell_count > 5:
                            raise AutopatchError('Bad cell detected')

                    currPressure = -5
                    self.pressure.set_pressure(currPressure)
                    self.pressure.set_ATM(atm=False)
            # ---------------------------------------------------------------

            # Holding potential switch
            if avg_resistance >= self.config.gigaseal_R / self.config.hold_switch and not holding_switched:
                self.amplifier.set_holding(self.protocol_config.vclamp_hold)
                self.amplifier.switch_holding(True)
                holding_switched = True

            # Success check with consecutive-hit filter
            if avg_resistance >= self.config.gigaseal_R:
                consecutive_success += 1
            else:
                autoPressure = False
            self.info(f"{self.config.mode}: Attempting to form gigaseal...")
            self.amplifier.auto_fast_compensation()
            self.sleep(1)
            self.daq.setCellMode(True)
            self.sleep(0.1)
            self.info("Collecting baseline resistance...")

            if consecutive_success >= 3:
                self.pressure.set_ATM(atm=True)
                self.info("Seal successful!")
                if self.config.mode == "Training":
                    self.info("Training mode: goal condition reached. Click Success or Abort to finish.")
                    while True:
                        self.sleep(0.1)
                self.success_requested = True
                self.success_if_requested()

        # Abort request came in
        # finally:
        #     if sim:
        #         self.daq.stop()
        raise AutopatchError("Seal attempt failed: gigaseal criteria not met.")
   
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
        
        
        Raises:
            AutopatchError: If break-in fails after multiple attempts.
        """
        from collections import deque  # local import keeps patch minimal

        # ---------- initial setup (unchanged) ----------
        self.daq.setCellMode(True)
        autoPressure = (self.config.mode == 'Classic')
        adaptivePressure = (self.config.mode == 'Adaptive')
        agentMode = (self.config.mode == 'Agent')
        agent_resistance_input_width = 0
        agent_resistance_history = deque(maxlen=1)
        if agentMode:
            self.info("Agent break-in mode detected; preparing break-in policy.")
            self.agenthelper.prepare_model("break_in")
            agent_resistance_input_width = self._agent_resistance_input_width()
            agent_resistance_history = deque(maxlen=max(1, agent_resistance_input_width))
        self.info(f"{self.config.mode}: Attempting Break in...")
        self.sleep(3)
        self.pressure.set_pressure(self.config.pulse_pressure_break_in)
        self.amplifier.set_zap_duration(25 * 1e-6)

        measuredAccessResistance = self.accessRamp()
        measuredResistance       = self.resistanceRamp()
        measuredCapacitance      = self.capacitanceRamp()
        self.info(
            f"Initial Resistance: {measuredResistance}; "
            f"Initial Capacitance: {measuredCapacitance},  "
            f"Initial Access Resistance: {measuredAccessResistance}")
        self.info(
            f"Target Resistance: {self.config.max_cell_R}; "
            f"Target Capacitance: {self.config.min_cell_C}, "
            f"Target Access Resistance: {self.config.max_access_R}")

        # ---------- loop variables ----------
        trials        = 0
        speed         = self.config.pulse_pressure_duration
        good_count    = 0
        threshold_AR  = self.config.max_access_R      # adjust here if units differ
        wait_period = 0.50


        # ---------- main loop ----------
        while True:
            # ---- 1) quick access-R check ----
            r_ax = self.accessRamp()
            self.debug(f"Access-R check: {r_ax:.2f} Ohm (good_count={good_count})")

            if r_ax <= threshold_AR:
                good_count += 1
                if good_count >= 3:        # success: 3 good hits in a row
                    measuredAccessResistance = r_ax
                    break
                # pause all actions; go straight to next access-R check
                continue
            else:
                good_count = 0             # reset streak on failure

            # ---- 2) full break-in cycle (runs only after a "bad" access-R) ----
            if agentMode:
                # ---------- Agent mode: let policy control ATM + zap ----------
                trials += 1
                self.debug(f"Trial: {trials} (Agent mode)")

                # Collect observation with pressure state for the agent
                observation = self.observe(include_pressure_state=True)
                self._attach_agent_resistance_input(
                    observation,
                    agent_resistance_history,
                    agent_resistance_input_width,
                )
                action = self.agenthelper.run_inference(observation=observation, is_demo=False)
                self.info(f"Break-in agent raw action: {action}")

                if action is None:
                    self.warning("Break-in agent did not return an action; skipping action application for this iteration.")
                else:
                    try:
                        action_array = np.asarray(action, dtype=float).reshape(-1)
                    except (TypeError, ValueError) as exc:
                        self.warning(f"Break-in agent action could not be converted to a numeric array: {exc}")
                        action_array = None

                    if action_array is not None:
                        if action_array.size < 1:
                            self.warning(
                                f"Break-in agent action must have at least 1 value; received shape {action_array.shape}."
                            )
                        elif not np.isfinite(action_array[0]):
                            self.warning(f"Break-in agent ATM action is invalid: {action_array[0]}")
                        else:
                            # Break-in action dims: ATM state, optional zap command.
                            target_atm = bool(float(action_array[0]) >= 0.5)
                            current_atm = bool(self.pressure.get_ATM())
                            if current_atm != target_atm:
                                self.pressure.set_ATM(atm=target_atm)
                                self.info(f"Break-in agent set ATM: {target_atm}")

                            should_zap = False
                            if action_array.size >= 2:
                                if np.isfinite(action_array[1]):
                                    should_zap = bool(float(action_array[1]) >= 0.5)
                                else:
                                    self.warning(f"Break-in agent zap action is invalid: {action_array[1]}")
                            if should_zap and self.config.zap:
                                self.info("zapping (Agent command)")
                                self.amplifier.zap()
                                self.sleep(0.5)

                            self.info(
                                "Break-in agent decoded action: "
                                f"atm={target_atm}, zap={should_zap}"
                            )

                self.sleep(wait_period * (1 + trials / 2))

            elif autoPressure:
                # ---------- Classic mode: heuristic pressure pulsing + periodic zap ----------
                trials += 1
                self.debug(f"Trial: {trials}")

                speedosc = trials % 5
                if speedosc == 0:
                    speed = 2*self.config.pulse_pressure_duration
                self.pressure.set_ATM(atm=False)
                self.sleep(1 / speed)
                self.pressure.set_ATM(atm=True)
                self.sleep(wait_period*(1 + trials/2))

                osc = trials % 3
                if self.config.zap and osc == 0:
                    self.info("zapping")
                    self.amplifier.zap(); self.sleep(0.5)
                    self.amplifier.zap(); self.sleep(0.5)

                self.sleep(1)

            elif adaptivePressure:
                # ---------- Adaptive mode: cloned heuristic pressure pulsing + periodic zap ----------
                trials += 1
                self.debug(f'Trial: {trials}')

                speedosc = trials % 5
                if speedosc == 0:
                    speed = 2*self.config.pulse_pressure_duration
                self.pressure.set_ATM(atm=False)
                self.sleep(1 / speed)
                self.pressure.set_ATM(atm=True)
                self.sleep(wait_period*(1 + trials/2))

                osc = trials % 3
                if self.config.zap and osc == 0:
                    self.info('zapping')
                    self.amplifier.zap(); self.sleep(0.5)
                    self.amplifier.zap(); self.sleep(0.5)

                self.sleep(1)

            # slow ramps (only if previous access-R was "bad")
            measuredResistance  = self.resistanceRamp()
            measuredCapacitance = self.capacitanceRamp()

            if autoPressure:
                self.info(
                    f"Trial {trials}: Running Avg Membrane Resistance: "
                    f"{measuredResistance}; Membrane Capacitance: "
                    f"{measuredCapacitance}, Access Resistance: {r_ax}")

                if trials > 15:
                    self.info("Break-in failed")
                    raise AutopatchError("Break-in failed")

            elif adaptivePressure:
                self.info(
                    f'Trial {trials}: Running Avg Membrane Resistance: '
                    f'{measuredResistance}; Membrane Capacitance: '
                    f'{measuredCapacitance}, Access Resistance: {r_ax}')

                if trials > 15:
                    self.info('Break-in failed')
                    raise AutopatchError('Break-in failed')

        # ---------- success ----------
        self.pressure.set_pressure(0)
        self.info("Successful break-in, Running Avg Access Resistance = "
                f"{measuredAccessResistance:.2f}")
        if self.config.mode == "Training":
            self.info("Training mode: goal condition reached. Click Success or Abort to finish.")
            while True:
                self.sleep(0.1)
        self.success_requested = True
        self.success_if_requested()

    def _isCellDetected(self, lastResDeque, cellThreshold = 0.15):
        '''
        Given a list of three resistance readings, do we think there is a cell where the pipette is?
        
        Args:
            lastResDeque (collections.deque): Recent resistance measurements.
            cellThreshold (float): Minimum resistance increase to detect a cell.

        Returns:
            bool: True if a cell is detected, False otherwise.
        '''
        # print(lastResDeque)

        # Ensure there are five readings before checking
        if len(lastResDeque) < 5:
            return False

        # Criteria 1: the last three readings must be increasing
        # if not lastResDeque[0] < lastResDeque[1] < lastResDeque[2]:
        #     # show the last three resistances
        #     self.debug(f"Last three resistances: {lastResDeque}")
        #     return False  # Last three resistances must be ascending
        
        # Criteria 2: there must be an increase by at least the cellThreshold
        r_delta = (lastResDeque[4] - self.first_res)

        # self.info(f"Cell detected, resistance: {r_delta}")
        detected = cellThreshold <= r_delta
        if detected:
            self.info(f"Cell detected: {detected}; resistance: {r_delta}")
            self.calibrated_unit.stop()

        return detected
   
    @record_state("patch")
    def patch(self, cell=None):
        """
        Runs the automatic patch-clamp algorithm, including manipulator movements.
        
        Args:
            cell: Target cell data for patching.
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

            #! Phase 0: locate cell
            _run_phase(self.locate_cell, cell, sleep_after=5)

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

    def clean_pipette(self):
        """
        Cleans the pipette using the Alconox bath and pressure pulses.

        Raises:
            ValueError: If cleaning_bath_position or safe_position is not set.
        """
        if self.cleaning_bath_position is None:
            raise ValueError('Cleaning bath position has not been set')

        if self.safe_position is None:
            raise ValueError('Safe position has not been set')
        # TODO: implement an abort mechanism
        try:
            self.info("Cleaning pipette")
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
            self.info("starting cleaning cycle")
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
            self.info("Drying complete, moving back to start position")
            self.calibrated_unit.absolute_move_group([start_x,safe_y,start_z], [0,1,2])
            self.calibrated_unit.wait_until_still()
            self.calibrated_unit.absolute_move(start_y, axis=1)
            self.calibrated_unit.wait_until_still() # Ensure movement completes
        finally:
            self.info("Pipette clean complete")
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

    def stop_movement(self):
        """Requests the movement loop to halt and waits for the thread to finish."""
        if getattr(self, 'stop_event', None):
            self.stop_event.set()
        if getattr(self, 'movement_thread', None):
            self.movement_thread.join()

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

    def toggle_laser_output(self):
        if self.laser is None:
            self.warning("No laser configured; skipping output toggle.")
            return
        try:
            self.laser.excite()
        except Exception as exc:
            self.error(f"Error toggling laser output: {exc}")

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

    def wavelength_down(self):
        self._step_laser_wavelength(-1)

    def wavelength_up(self):
        self._step_laser_wavelength(1)

    def _agent_resistance_input_width(self, default_width: int = 15) -> int:
        """Return the active policy's resistance_input width, or 0 if unused."""
        agent = getattr(self.agenthelper, "agent", None)
        if agent is None:
            return 0

        required_keys = ()
        getter = getattr(agent, "get_required_obs_keys", None)
        if callable(getter):
            try:
                required_keys = tuple(str(key) for key in getter())
            except Exception:
                required_keys = ()
        if not required_keys:
            required_keys = tuple(str(key) for key in getattr(agent, "obs_keys", ()) or ())

        importer = getattr(agent, "importer", None)
        obs_shapes = getattr(importer, "obs_shapes", {}) or {}
        if "resistance_input" not in required_keys and "resistance_input" not in obs_shapes:
            return 0

        shape = obs_shapes.get("resistance_input")
        if shape is not None:
            try:
                shape_tuple = tuple(int(dim) for dim in shape)
            except Exception:
                shape_tuple = ()
            if shape_tuple:
                return max(1, int(shape_tuple[-1]))
        return max(1, int(default_width))

    def _attach_agent_resistance_input(self, observation, history, width: int) -> None:
        """Attach prior resistance samples to a live agent observation."""
        if width <= 0 or not isinstance(observation, dict):
            return

        resistance_input = np.zeros(int(width), dtype=np.float32)
        prior_values = list(history)[-int(width):]
        if prior_values:
            resistance_input[-len(prior_values):] = np.asarray(prior_values, dtype=np.float32)
        observation["resistance_input"] = resistance_input

        try:
            current_resistance = float(np.asarray(observation.get("resistance")).reshape(-1)[0])
        except Exception:
            return
        if np.isfinite(current_resistance):
            history.append(current_resistance)

    def observe(self, include_pressure_state: bool = False):
        import time
        t0 = time.perf_counter()

        _, _, _, img = self.calibrated_stage.camera._last_frame_queue[0]
        t1 = time.perf_counter()

        cvpi = self.calibrated_unit.pipetteCalHelper.pipetteDetector.detect_pipette(img)
        # Pass the current frame to focus estimator (it now requires an image argument)
        cvpiz = self.calibrated_unit.pipetteFocusHelper.pipetteFocuser.get_pipette_focus_value(img)
        cvpi = np.append(cvpi, cvpiz)
        t2 = time.perf_counter()

        pi = self.calibrated_unit.position()
        st = self.calibrated_stage.position()[:2]
        stz = self.calibrated_unit.microscope.position() / self.calibrated_unit.config.microscope_units_per_um
        st = np.append(st, stz)
        t3 = time.perf_counter()

        res = self.resistanceRamp(num_measurements=1, interval=0.001)
        t4 = time.perf_counter()

        # self.info(f"[observe timing] frame={ (t1-t0)*1e3:.1f} ms | detect={ (t2-t1)*1e3:.1f} ms | coords={ (t3-t2)*1e3:.1f} ms | resistanceRamp={ (t4-t3):.3f} s | total={ (t4-t0):.3f} s")
        if not include_pressure_state:
            return [cvpi, st, img, res]

        actual_pressure = self.pressure.get_last_acquisition()
        commanded_pressure = self.pressure.get_pressure()
        pressure_atm_state = float(bool(self.pressure.get_ATM()))

        observation = {
            "pipette_positions": cvpi,
            "stage_positions": st,
            "camera_image": img,
            "resistance": np.asarray([res], dtype=np.float32),
            "pressure": np.asarray([actual_pressure], dtype=np.float32),
            "commanded_pressure_mbar": np.asarray([commanded_pressure], dtype=np.float32),
            "pressure_atm_state": np.asarray([pressure_atm_state], dtype=np.float32),
        }
        return observation
