"""Implementation of the find_pipette phase."""

import time

import numpy as np

from ..PhaseController import PhaseController


class FindPipettePhase(PhaseController):
    def run(self, cell=None):
        self.begin_observations()
        self.controller.info("Finding pipette")
        # Only load the agent policy when running in Agent mode.
        if self.controller.config.mode == 'Agent':
            self.controller.agenthelper.prepare_model("find_pipette", allow_goal_placeholders=True)
        elif self.controller.config.mode == 'Adaptive':
            self.controller.info('Adaptive mode detected; skipping agent model load for find_pipette')
        else:
            self.controller.info("Classic/Manual/Training mode detected; skipping agent model load for find_pipette")
        max_sleep_time = 0.005  # seconds (slowest polling)
        min_sleep_time = 0.001  # seconds (fastest polling)
        sleep_time = max_sleep_time
        command_speed_um_s = abs(float(getattr(self.controller, "find_pipette_velocity_speed_um_s", 200.0)))
        if command_speed_um_s == 0:
            raise ValueError("find_pipette_velocity_speed_um_s must be non-zero.")
        non_agent_relative_move_threshold_um_s = 1000.0
        use_non_agent_relative_move = command_speed_um_s >= non_agent_relative_move_threshold_um_s
        use_velocity_prediction = bool(getattr(self.controller, "velocity_prediction", False))
        if self.controller.config.mode == "Agent":
            if use_velocity_prediction:
                self.controller.info(
                    "Agent find_pipette action mode: velocity "
                    "(xy in px/s converted to um/s, z in um/s)."
                )
            else:
                self.controller.info(
                    "Agent find_pipette action mode: displacement "
                    "(xy in px converted to um, z in um)."
                )

        def _log_timing(label: str, duration_s: float) -> None:
            """Lightweight timing logger for find_pipette stages."""
            # self.controller.info(f"[find_pipette timing] {label}: {duration_s * 1000.0:.1f} ms")

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

        goal_needed = bool(self.controller.goal_needed)
        random = bool(self.controller.goal_random)

        camera = self.controller.calibrated_stage.camera
        width = getattr(camera, "width", None)
        height = getattr(camera, "height", None)

        center_x = int(round(width / 2)) if isinstance(width, (int, float)) else 640
        center_y = int(round(height / 2)) if isinstance(height, (int, float)) else 640
        goal_center = np.array([center_x, center_y, 0.0], dtype=float)
        self.controller.info(f"Using goal center at: {goal_center} (px)")

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
                self.controller.calibrated_unit.stop()
            velocity_start_pos_um = None
            velocity_direction = None
            velocity_distance_um = None
            velocity_start_time = None
            velocity_timeout_s = None
            velocity_opposite_direction_warned = False

        while True:
            obs_start = time.perf_counter()
            observation = self.observe(fields=["manipulator_position", "pipette_image_xy", "pipette_defocus_um",
                                               "stage_positions", "camera_image", "resistance"])
            # _log_timing("observation", time.perf_counter() - obs_start)
            curr_point = np.r_[observation["pipette_image_xy"], observation["pipette_defocus_um"]]

            if curr_point is None:
                self.controller.warning("Pipette detector did not return a location; waiting for next frame")
                self.controller.sleep(sleep_time)
                continue

            if isinstance(curr_point, np.ndarray):
                curr_point = curr_point.tolist()
            if len(curr_point) < 3 or any(value is None for value in curr_point[:3]):
                self.controller.warning("Pipette detector returned incomplete coordinates; waiting for next frame")
                self.controller.sleep(sleep_time)
                continue

            curr_array = np.asarray(curr_point[:3], dtype=float)
            if np.isnan(curr_array).any():
                self.controller.warning("Pipette detector returned NaN coordinates; waiting for next frame")
                self.controller.sleep(sleep_time)
                continue

            curr_point = tuple(float(coord) for coord in curr_array)
            curr_point_np = np.asarray(curr_point, dtype=float)
            camera = self.controller.calibrated_stage.camera
            adaptive_mode = self.controller.config.mode == 'Adaptive'
            should_act = self.controller.config.mode == 'Agent'
            z_weight = 1.0
            tol_um = 2.0
            px_per_um = self.controller.calibrated_unit.pixel_per_um()

            if adaptive_mode:
                xgerr_px = goal_error_target[0] - curr_point_np[0]
                ygerr_px = goal_error_target[1] - curr_point_np[1]
                zerr_um = -curr_point_np[2]  # drive defocus to 0

                dx_um = xgerr_px / px_per_um[0] if px_per_um and px_per_um[0] else np.nan
                dy_um = ygerr_px / px_per_um[1] if px_per_um and px_per_um[1] else np.nan
                gerr_um = float(np.sqrt((dx_um ** 2 + dy_um ** 2 + z_weight * (zerr_um ** 2)) / (2 + z_weight)))
                self.controller.info(f' Goal error (um):{gerr_um}')

                if goal_needed and camera is not None:
                    camera.show_circle(
                        point=goal_display_tuple,
                        color=(255, 255, 255),
                        show_center=False,
                    )

                if self.success_gate(gerr_um, tol_um):
                    self.controller.info('Pipette found')
                    self.controller.calibrated_unit.stop()
                    if self.controller.config.mode == 'Training':
                        self.controller.info('Training mode: goal condition reached. Click Success or Abort to finish.')
                        while True:
                            self.controller.sleep(0.1)
                    self.controller.success_requested = True
                    self.controller.success_if_requested()

                action = None
                target_point = None
                err = None
                act_start = time.perf_counter()
                xy_um = self.controller.calibrated_unit.pixels_to_um_relative([xgerr_px, ygerr_px, 0])
                # target_um = self.controller.calibrated_unit.position() + np.array([xy_um[0], xy_um[1], zerr_um])
                # self.controller.calibrated_unit.absolute_move(target_um.tolist())
                # self.controller.calibrated_unit.wait_until_still()
                move_um = np.array([xy_um[0], xy_um[1], zerr_um], dtype=float)
                self.act(move_um, state={
                    "kind": "direct",
                    "use_relative_move": use_non_agent_relative_move,
                    "command_speed_um_s": command_speed_um_s,
                })
                self.controller.sleep(_adaptive_sleep_time(gerr_um, tol_um))
                continue

            if not should_act:
                xgerr_px = goal_error_target[0] - curr_point_np[0]
                ygerr_px = goal_error_target[1] - curr_point_np[1]
                zerr_um = -curr_point_np[2]  # drive defocus to 0

                dx_um = xgerr_px / px_per_um[0] if px_per_um and px_per_um[0] else np.nan
                dy_um = ygerr_px / px_per_um[1] if px_per_um and px_per_um[1] else np.nan
                gerr_um = float(np.sqrt((dx_um ** 2 + dy_um ** 2 + z_weight * (zerr_um ** 2)) / (2 + z_weight)))
                self.controller.info(f" Goal error (um):{gerr_um}")

                if goal_needed and camera is not None:
                    camera.show_circle(
                        point=goal_display_tuple,
                        color=(255, 255, 255),
                        show_center=False,
                    )

                if self.success_gate(gerr_um, tol_um):
                    self.controller.info("Pipette found")
                    self.controller.calibrated_unit.stop()
                    if self.controller.config.mode == "Training":
                        self.controller.info("Training mode: goal condition reached. Click Success or Abort to finish.")
                        while True:
                            self.controller.sleep(0.1)
                    self.controller.success_requested = True
                    self.controller.success_if_requested()

                action = None
                target_point = None
                err = None
                act_start = time.perf_counter()
                xy_um = self.controller.calibrated_unit.pixels_to_um_relative([xgerr_px, ygerr_px, 0])
                # target_um = self.controller.calibrated_unit.position() + np.array([xy_um[0], xy_um[1], zerr_um])
                # self.controller.calibrated_unit.absolute_move(target_um.tolist())
                # self.controller.calibrated_unit.wait_until_still()
                move_um = np.array([xy_um[0], xy_um[1], zerr_um], dtype=float)
                self.act(move_um, state={
                    "kind": "direct",
                    "use_relative_move": use_non_agent_relative_move,
                    "command_speed_um_s": command_speed_um_s,
                })
                # _log_timing("action_direct_pipette", time.perf_counter() - act_start)
                self.controller.sleep(_adaptive_sleep_time(gerr_um, tol_um))
                continue

            if action is None:
                # preprocess goal by cropping  and rescaling to 85 by 85
                inf_start = time.perf_counter()
                agent_goal = goal
                if goal is not None:
                    agent = getattr(self.controller.agenthelper, "agent", None)
                    if agent is not None:
                        try:
                            image = observation["camera_image"]
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
                                        self.controller.info(f"goal scaled: {agent_goal} um")
                        except Exception as exc:
                            self.controller.warning(f"Goal preprocessing failed; using raw goal. Error: {exc}")
                action = self.controller.agenthelper.run_inference(observation=observation, goal=agent_goal, is_demo=False)
                # _log_timing("inference_block", time.perf_counter() - inf_start)
                prediction_units = "velocity (xy px/s, z um/s)" if use_velocity_prediction else "displacement (xy px, z um)"
                self.controller.info(f"pipette prediction: {action} [{prediction_units}]")

                if action is None:
                    self.controller.warning("Model did not return an action; retrying inference")
                    if use_velocity_prediction:
                        _reset_velocity_motion(stop_motion=True)
                    err = None
                    self.controller.sleep(sleep_time)
                    continue

                pred_offset_xy = np.asarray(action[:2], dtype=float)
                if pred_offset_xy.size < 2:
                    self.controller.warning("Predicted offset missing coordinates; retrying inference")
                    if use_velocity_prediction:
                        _reset_velocity_motion(stop_motion=True)
                    action = None
                    target_point = None
                    err = None
                    self.controller.sleep(sleep_time)
                    continue

                if np.isnan(pred_offset_xy).any():
                    self.controller.warning("Predicted offset contains NaNs; retrying inference")
                    if use_velocity_prediction:
                        _reset_velocity_motion(stop_motion=True)
                    action = None
                    target_point = None
                    err = None
                    self.controller.sleep(sleep_time)
                    continue

                if use_velocity_prediction:
                    # Velocity mode: model output is interpreted directly as [vx_px, vy_px, vz_um].
                    # Convert xy into manipulator-frame um/s; pass z through as-is.
                    z_velocity_um_s = 0.0 if len(action) < 3 or action[2] is None else float(action[2])
                    velocity_xy_um_s = self.controller.calibrated_unit.pixels_to_um_relative(
                        [-pred_offset_xy[0], -pred_offset_xy[1], 0.0]
                    )
                    velocity_um_s = np.array(
                        [velocity_xy_um_s[0], velocity_xy_um_s[1], z_velocity_um_s],
                        dtype=float,
                    )
                    if not np.isfinite(velocity_um_s).all():
                        self.controller.warning("Predicted velocity contains invalid values; retrying inference")
                        action = None
                        target_point = None
                        err = None
                        _reset_velocity_motion(stop_motion=True)
                        self.controller.sleep(sleep_time)
                        continue
                    if float(np.linalg.norm(velocity_um_s)) == 0.0:
                        self.controller.info("Predicted velocity is zero; stopping motion and requesting next action.")
                        action = None
                        target_point = None
                        err = None
                        _reset_velocity_motion(stop_motion=True)
                        self.controller.sleep(sleep_time)
                        continue
                    self.act(velocity_um_s, state={"kind": "agent_velocity"})
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
                            self.controller.warning(f"predicted point not on screen: {target_point_pixels}")
                            action = None
                            target_point = None
                            err = None
                            _reset_velocity_motion(stop_motion=True)
                            self.controller.sleep(0.04)
                            continue
                    move_xy_um = self.controller.calibrated_unit.pixels_to_um_relative(
                        [pred_offset_xy[0], pred_offset_xy[1], 0.0]
                    )
                    move_um = np.array([move_xy_um[0], move_xy_um[1], -z_offset_um], dtype=float)
                    if not np.isfinite(move_um).all():
                        self.controller.warning("Predicted displacement contains invalid values; retrying inference")
                        action = None
                        target_point = None
                        err = None
                        self.controller.sleep(sleep_time)
                        continue
                    move_distance_um = float(np.linalg.norm(move_um))
                    if move_distance_um == 0:
                        self.controller.info("Predicted movement is zero; requesting next action.")
                        action = None
                        target_point = None
                        err = None
                        self.controller.sleep(sleep_time)
                        continue
                    self.act(move_um, state={"kind": "agent_displacement"})
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
            self.controller.info(f" Goal error (um):{gerr}")
            loop_sleep_time = _adaptive_sleep_time(gerr, tol_um)

            if goal_needed and camera is not None:
                camera.show_circle(
                    point=goal_display_tuple,
                    color=(255, 255, 255),
                    radius=15,
                    show_center=False,
                )

            if self.success_gate(gerr, tol_um):
                self.controller.info("Pipette found")
                _reset_velocity_motion(stop_motion=True)
                if self.controller.config.mode == "Training":
                    self.controller.info("Training mode: goal condition reached. Click Success or Abort to finish.")
                    while True:
                        self.controller.sleep(0.1)
                self.controller.success_requested = True
                self.controller.success_if_requested()

            if target_point is not None and velocity_start_pos_um is not None and velocity_direction is not None and velocity_distance_um is not None:
                current_position_um = np.asarray(self.controller.calibrated_unit.position(), dtype=float)
                signed_traveled_um = float(np.dot(current_position_um - velocity_start_pos_um, velocity_direction))
                traveled_um = abs(signed_traveled_um)
                if traveled_um >= velocity_distance_um:
                    _reset_velocity_motion(stop_motion=True)
                    action = None
                    target_point = None
                    err = None
                    self.controller.sleep(loop_sleep_time)
                    continue
                if (signed_traveled_um < 0) and (not velocity_opposite_direction_warned):
                    self.controller.warning(
                        "Find-pipette velocity move is progressing opposite commanded direction; "
                        "using absolute displacement criterion."
                    )
                    velocity_opposite_direction_warned = True
                if velocity_start_time is not None and velocity_timeout_s is not None:
                    elapsed_s = time.perf_counter() - velocity_start_time
                    if elapsed_s > velocity_timeout_s:
                        self.controller.warning(
                            f"Find-pipette velocity move timeout after {velocity_timeout_s:.2f}s "
                            f"(target {velocity_distance_um:.2f} um, traveled {signed_traveled_um:.2f} um signed)."
                        )
                        _reset_velocity_motion(stop_motion=True)
                        action = None
                        target_point = None
                        err = None
                        self.controller.sleep(loop_sleep_time)
                        continue

            if target_point is not None:
                xerr = curr_point_np[0] - target_point[0]
                yerr = curr_point_np[1] - target_point[1]
                dx_um = xerr / px_per_um[0] if px_per_um and px_per_um[0] else np.nan
                dy_um = yerr / px_per_um[1] if px_per_um and px_per_um[1] else np.nan
                err = float(np.sqrt((dx_um ** 2 + dy_um ** 2) / 2.0))
                # self.controller.info(f"total XY error (um): {err}")

                if err <= (tol_um / 2):
                    _reset_velocity_motion(stop_motion=True)
                    action = None
                    target_point = None
                    err = None
                    self.controller.sleep(loop_sleep_time)
                    continue

            self.controller.sleep(loop_sleep_time)

    def act(self, observation=None, state=None):
        """Apply a validated movement using the selected motion convention.

        The run loop retains coordinate conversion, inference retries, and
        velocity bookkeeping so their order relative to observation is explicit.
        """
        move_um = observation
        if state["kind"] == "agent_velocity":
            self.controller.calibrated_unit.relative_move_group_velocity(move_um.tolist())
        elif state["kind"] == "agent_displacement":
            self.controller.calibrated_unit.relative_move_group(move_um.tolist())
            self.controller.calibrated_unit.wait_until_still()
        else:
            move_distance_um = float(np.linalg.norm(move_um))
            if move_distance_um > 0:
                if state["use_relative_move"]:
                    self.controller.calibrated_unit.relative_move_group(move_um.tolist())
                    self.controller.calibrated_unit.wait_until_still()
                else:
                    velocity = self.controller.calibrated_unit.velocity_position_control(
                        move_um, state["command_speed_um_s"]
                    )
                    velocity_command_local = -np.asarray(velocity, dtype=float)
                    self.controller.calibrated_unit.absolute_move_group_velocity(velocity_command_local.tolist())

    def success_gate(self, observation=None, state=None):
        """Compare the measured weighted goal error with the tolerance in um."""
        return observation <= state
