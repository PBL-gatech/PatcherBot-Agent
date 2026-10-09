"""Cell-type-dependent approach from the common localization distance."""

import sys
import time
import numpy as np
from ..errors import AutopatchError
from ..PhaseController import PhaseController


class ApproachCellPhase(PhaseController):
    phase_name = "Approach"

    def run(self, cell=None):
        state = None
        try:
            state = self.prepare(cell)
            while True:
                observation = self.observe(state=state)
                calculated = self.calculate(observation, state)
                command = self.decide(calculated, state)
                self.act(command, state)
        finally:
            display = getattr(self.controller, "alignment_display", None)
            if display is not None and display.get("owner") is self:
                self.controller.alignment_display = None
            self.finish(state)

    def prepare(self, cell=None):
        self.failure_gate()
        self.success_gate()
        self.begin_observations()
        self.controller.alignment_display = dict(owner=self, cell_reference=None if cell is None else cell[0])
        self._adaptive = self.controller.config.mode == "Adaptive"
        self.controller._approach_cell_focus = None
        self._vision = False
        self.controller.info(f"Approach START: mode={self.controller.config.mode}")
        return {"cell": cell, "original_cell_reference": None if cell is None else cell[0],
                "fields": [], "step": 0, "display_detection_markers": True,
                "bounded_focus": True, "cell_search_radius_um": 50.0}


    def observe(self, *, state=None, **kwargs):
        if state is None:
            return super().observe(**kwargs)
        kwargs.setdefault("fields", state["fields"])
        kwargs.setdefault("evidence", {})
        searching = self._vision and state["step"] == 1
        if self._vision:
            self.frame_context["pipette_expected_xy"] = lambda: self.controller.calibrated_unit.reference_position(anchor=getattr(self.controller.calibrated_unit, '_trusted_tip_reference', None) or (np.array([self.controller.calibrated_unit.camera.width / 2., self.controller.calibrated_unit.camera.height / 2.]), self.controller.home_position, np.asarray(self.controller.home_stage_position)[:2]))[:2]
            if state.get("bounded_focus"):
                self.frame_context["focus_z"] = lambda: (self.controller.microscope.position()
                    / self.controller.calibrated_unit.config.microscope_units_per_um)
                if searching:
                    self.frame_context["stage_xy"] = self.controller.calibrated_stage.position
                    kwargs["fields"] = [*kwargs.get("fields", state.get("fields", [])), "camera_image"]
            inference_fields = ["pipette_image_xy", "pipette_defocus_um"]
            if searching:
                inference_fields.append("cell_detections")
            kwargs["evidence"] = self.controller.observer.infer_frame(inference_fields, frame_context=self.frame_context)
        observation = super().observe(**kwargs)
        if self._vision:
            vision = observation.get("deep_learning", {})
            tip = vision.get("pipette_tracking_position", vision.get("pipette_position"))
            if state is not None and state.get("step") == 2:
                sweep = state.get("focus_sweep") or {}
                confirmation = sweep.get("confirmation")
                tip = (confirmation["confirmed_tip_xy"] if confirmation and confirmation["confirmed"]
                       else sweep.get("best_xy"))
                if tip is None:
                    tip = self.frame_context["pipette_expected_xy"]()
            if tip is not None:
                self.controller.observer.annotate(self, observation, markers={"pipette": tip})
        return observation







    def _shift_cell_from_evidence(self, state):
        controller, stage = self.controller, self.controller.calibrated_stage
        options = {}
        fusion = getattr(controller.config, 'cell_shift_fusion', True)
        if fusion:
            from ...devices.manipulator.helpers.CellTrackHelper import CellTrackHelper
            options = CellTrackHelper.cell_shift_options(stage, state['observation'], state,
                use_centroid=controller.config.use_centroid, info=controller.info, warning=controller.warning)
            if options is False:
                return False
            if 'detected_px' in options:
                controller.observer.annotate(self, state['observation'], markers={'cell': options['detected_px']})
        else:
            controller.info('Cell SHIFT consensus disabled; using existing template centering')
        result = stage.center_on_cell(state['cell'], use_centroid=controller.config.use_centroid,
            max_error_px=controller.config.approach_cell_max_shift, **options)
        return None if fusion else result


    def calculate(self, observation=None, state=None):
        if not self._adaptive:
            return super().calculate(observation, state)
        state["observation"] = observation
        if self._vision:
            observation["cell_focus_error"] = observation["stage_positions"][2] - state["cell_focus_z"]
            if state["step"] == 1 and not observation.get("cell_selection_locked", False):
                expected_xy = (self.controller.calibrated_stage.M @ observation["stage_positions"][:2]
                               + self.controller.calibrated_stage.r0)[:2] - state["cell"][0][:2]
                detections = observation["deep_learning"].get("cell_detections") or []
                candidates = detections
                radius_um = state.get("cell_search_radius_um")
                if radius_um is not None:
                    stage = self.controller.calibrated_stage
                    candidates = [cell for cell in detections
                        if np.isfinite(cell[:3]).all()
                        and cell[2] >= stage.cellDetectHelper.cellDetector.conf_threshold
                        and np.linalg.norm(stage.pixels_to_um_relative(
                            np.asarray(cell[:2]) - expected_xy)[:2]) <= radius_um]
                observation["cell_detection"] = min(candidates,
                    key=lambda cell: sum((cell[:2] - expected_xy) ** 2), default=None)
                self.controller.info(f"Cell SEARCH: candidates={len(detections)}; "
                    f"accepted_candidates={len(candidates)}; radius_um={radius_um}; "
                    f"expected_xy={expected_xy}; selected={observation['cell_detection']}; "
                    f"focus_error_um={observation['cell_focus_error']}")
                cell = observation["cell_detection"]
                if (state.get("display_detection_markers") and cell is not None
                        and cell[2] >= self.controller.calibrated_stage.cellDetectHelper.cellDetector.conf_threshold):
                    self.controller.observer.annotate(self, observation, markers={"cell": cell[:2]})
            elif state["step"] == 2:
                depth = observation["deep_learning"].get("pipette_focus")
                observation["defocus_error"] = float("inf") if depth is None else abs(depth)
                observation["pipette_z_error"] = abs(observation["stage_positions"][2] - state["pipette_z"])
                observation["up_speed"] = self.controller.microscope.up_direction * self.controller.config.search_speed
                observation["scan_speed"] = observation["up_speed"] * (np.sign(depth) if depth is not None and np.isfinite(depth) else 1)
            elif state["step"] == 3:
                observation["spear_target"] = (observation["manipulator_position"][2]
                    + (state["hover_z"] - state["pipette_z"] if "hover_z" in state else
                       self.controller.config.slice_start_distance - state["goal_distance"]))
            elif state["step"] == 4:
                motion = state.get("spear_motion")
                try:
                    observation["spear_done"] = self.controller.calibrated_unit.actionHelper.is_complete(motion) if motion is not None else None
                except RuntimeError as error:
                    raise AutopatchError(str(error)) from error
                pipette_z = (self.controller.calibrated_unit.position()[2] if observation["spear_done"]
                             else observation["manipulator_position"][2])
                observation["spear_remaining"] = state["spear_target"] - pipette_z
                observation["pipette_plane"] = state["pipette_z"] + pipette_z - state["spear_start_z"]
        observation["spear_speed"] = abs(self.controller.config.max_clearing_speed)
        observation["speed"] = self.controller.microscope.up_direction * self.controller.config.search_speed * (-1 if state["step"] in (0, 2, 4) else 1)
        return observation


    def decide(self, observation=None, state=None):
        self.failure_gate()
        self.success_gate()
        if self.awaiting_operator or state.get("finished", False):
            return None
        config = self.controller.config
        command = {"cell": state["cell"],
                   "route_token": (config.cell_type, bool(config.cell_type_toggle) if config.cell_type != "Plate" else None)}
        if config.mode == "Adaptive":
            step = state["step"]
            if state.get("bounded_focus") and step in (1, 2) and "confirmation" not in state.get("focus_sweep", {}):
                controller, is_cell = self.controller, step == 1
                stage = controller.calibrated_stage
                focus = dict(actual_z=observation["stage_positions"][2],
                    vision=observation["deep_learning"], cell=observation.get("cell_detection"))
                if state.get("focus_sweep") is None:
                    focus.update(center_z=state["cell_focus_z"] if is_cell else state["pipette_z"],
                        speed=config.search_speed, motion=stage.actionHelper,
                        bounds=state.get("cell_search_bounds") if is_cell else None,
                        label="SEARCH" if is_cell else "SCAN", pipette=not is_cell,
                        pixels_to_um_relative=stage.pixels_to_um_relative,
                        confidence_threshold=lambda: (stage.cellDetectHelper.cellDetector.conf_threshold
                            if is_cell else controller.calibrated_unit.pipetteCalHelper.pipetteDetector.adapter.threshold),
                        abort_check=controller.abort_if_requested, info=controller.info, warning=controller.warning)
                return {**command, "route": "adaptive", "method": "focus", "focus": focus}
            confidence = observation.get("deep_learning", {}).get("confidence", {})
            cell = observation.get("cell_detection")
            vision = observation.get("deep_learning", {})
            depth = vision.get("pipette_focus")
            focus_confidence = confidence.get("pipette_focuser")
            no_focus_confidence_head = vision.get("pipette_focus_confidence_supported") is False
            focus_confident = (np.isfinite(focus_confidence) and focus_confidence >= 0.50 if focus_confidence is not None else
                no_focus_confidence_head and vision.get("status", {}).get("pipette_focuser") == "ok")
            pipette_confident = (step == 2 and (confidence.get("pipette_detector") or 0) >= self.controller.calibrated_unit.pipetteCalHelper.pipetteDetector.adapter.threshold
                                 and depth is not None and np.isfinite(depth) and focus_confident)
            if pipette_confident and focus_confidence is None and not state.get("logged_optional_focus_confidence"):
                self.controller.info("Pipette focus: checkpoint has no Z-confidence head; "
                    f"depth_um={depth}; detector_confidence={confidence.get('pipette_detector')}; "
                    + ("Z confidence is ignored; SCAN uses predicted defocus and XY confidence" if state.get("bounded_focus") else
                       "using valid depth, detector confidence, and existing position/focus tolerances"))
                state["logged_optional_focus_confidence"] = True
            ready = (step == 0 or (step == 1 and observation["cell_focus_error"] >= 0 and cell is not None
                and cell[2] >= self.controller.calibrated_stage.cellDetectHelper.cellDetector.conf_threshold)
                or (pipette_confident and observation["pipette_z_error"] <= 5 and observation["defocus_error"] <= 2.0)
                or (step == 3 and (abs(observation["stage_positions"][2] - state["hover_z"]) <= 0.5
                    if "hover_z" in state else observation["cell_focus_error"] >= 0))
                or (step == 4 and (observation["spear_done"] if observation.get("spear_done") is not None
                                  else observation["spear_remaining"] <= 0.5)))
            if state.get("bounded_focus") and step in (1, 2):
                confirmation = state["focus_sweep"]["confirmation"]
                try:
                    ready = self.controller.calibrated_stage.focusHelper.update_confirmation(
                        state["focus_sweep"], vision, cell)
                except RuntimeError as error:
                    raise AutopatchError(str(error)) from error
                if ready is not True:
                    return None
                if step == 2:
                    state["confirmed_tip_xy"] = confirmation["confirmed_tip_xy"]
            if step == 2 and not ready:
                speed = observation["scan_speed"] if pipette_confident and observation["pipette_z_error"] <= 5 else observation["up_speed"]
                return {**command, "route": "adaptive", **self.scan(speed)}
            if not ready:
                return None
            if step == 0:
                move = self.search(observation["speed"])
                move["goal_distance"] = config.cell_distance if config.cell_type == "Plate" or (config.cell_type == "Slice" and config.cell_type_toggle) else config.slice_start_distance
            elif step in (1, 2):
                move = self.shift("cell" if step == 1 else "pipette", self.scan(observation["speed"]))
            elif step == 3:
                move = ({"method": "complete"} if state.get("speared") else
                        self.spear(observation["spear_target"], observation["spear_speed"]))
            else:
                move = {**self.search(observation["speed"]), "pipette_plane": observation["pipette_plane"]}
            return {**command, "route": "adaptive", **move}
        if config.cell_type == "Plate":
            return {**command, "route": "plate"}
        if config.cell_type_toggle and config.cell_type == "Slice":
            return {**command, "route": "slice"}
        return {**command, "route": "initial"}


    def act(self, observation=None, state=None):
        self.failure_gate()
        self.success_gate()
        command = observation
        controller = self.controller
        if (command is not None or self._vision) and not self.action_gate(command, state):
            raw = state.get("observation", {})
            vision = raw.get("deep_learning", {})
            planes = (state.get("cell_focus_z"), state.get("pipette_z"))
            sweep = state.get("focus_sweep")
            tolerance = self.controller.calibrated_stage.focusHelper.BOUNDARY_TOLERANCE_UM if sweep is not None else 0.0
            limits = None if None in planes else (min(planes) - 5 - tolerance, max(planes) + 5 + tolerance)
            z = raw.get("stage_positions", [None, None, None])[2]
            outside = limits is not None and z is not None and not limits[0] <= z <= limits[1]
            self.controller.info(
                f"Approach BLOCKED: step={state.get('step')}; mode={self.controller.config.mode}; "
                f"method={(command or {}).get('method')}; stale_vision={vision.get('stale')}; "
                f"frame_age_s={vision.get('age_s')}; outside_z_limits={outside}; "
                f"microscope_z={z}; allowed_z={limits}; "
                f"spear_target={(command or {}).get('target')}; speed={(command or {}).get('speed')}; "
                f"pipette_z={state.get('pipette_z')}; pipette_focus={vision.get('pipette_focus')}; "
                f"confidence={vision.get('confidence')}; focus_confidence_supported={vision.get('pipette_focus_confidence_supported')}; "
                f"model_status={vision.get('status')}; model_errors={vision.get('errors')}")
            raise AutopatchError("Approach action is not permitted")
        if command is None:
            self.controller.sleep(0.1)
            return
        if command["route"] == "adaptive" and self._vision:
            if command["method"] == "focus":
                try:
                    state["focus_sweep"] = self.controller.calibrated_stage.focusHelper.update_sweep(
                        state.get("focus_sweep"), **command["focus"])
                except (RuntimeError, ValueError) as error:
                    raise AutopatchError(str(error)) from error
                if state["focus_sweep"].get("empty"):
                    reason = "SEARCH focus sweep found no valid samples"
                    if not self.failure_gate(state, message=reason, recover=True):
                        raise AutopatchError(reason)
                return
            if command["method"] != "scan":
                self.controller.microscope.stop()
            if command["method"] == "shift":
                self.controller.info(f"Approach SHIFT {command['target']}: starting")
                if command["target"] == "cell":
                    state["cell_focus_z"] = state["observation"]["stage_positions"][2]
                    if self._shift_cell_from_evidence(state) is False:
                        return
                else:
                    state["pipette_z"] = state["observation"]["stage_positions"][2]
                    options = (dict(detected_px=state.pop('confirmed_tip_xy'),
                             expected_px= self.frame_context["pipette_expected_xy"](),
                             max_deviation_um=controller.calibrated_unit.config.pipette_detection_radius_um)
                        if 'confirmed_tip_xy' in state else {})
                    if options:
                        controller.observer.annotate(self, state["observation"], markers={"pipette": options["detected_px"]})
                    centered = controller.calibrated_unit.center_pipette(
                        speed=controller.config.approach_pipette_speed, **options)
                    self.failure_gate()
                    self.success_gate()
                    if centered is False:
                        reason = getattr(controller.calibrated_unit, "last_pipette_move_failure", None)
                        if not isinstance(reason, str) or not reason:
                            reason = "pipette centering did not complete"
                        next_action = "final alignment incomplete" if state.get("speared") else "SPEAR cancelled"
                        raise AutopatchError(f"{self.phase_name} pipette SHIFT failed: {reason}; {next_action}")
                self.controller.info(f"Approach SHIFT {command['target']}: finished")
                self.controller.calibrated_stage.focusHelper.reset_state(state)
                state["step"] += 1
                if command["target"] == "cell":
                    self.failure_gate()
                    self.success_gate()
                    self.controller.info(f"Approach RELOCATE pipette: target_z={state['pipette_z']}")
                    self.controller.calibrated_stage.focusHelper.relocate(state["pipette_z"])
                    self.failure_gate()
                    self.success_gate()
                    self.controller.info("Approach RELOCATE pipette: finished; checking focus before local SCAN or SHIFT")
                    return
                if state.get("speared"):
                    self.finish(state, success=True)
                    return
                state["hover_z"] = state["cell_focus_z"] - state["goal_distance"]
                self.controller.info(f"Approach RELOCATE stand-off: target_z={state['hover_z']}; "
                    f"distance_above_cell={state['goal_distance']} um")
                self.controller.calibrated_stage.focusHelper.relocate(state["hover_z"])
                self.failure_gate()
                self.success_gate()
                self.controller.info("Approach RELOCATE stand-off: finished; checking position before SPEAR")
                return
            if command["method"] in ("search", "scan"):
                if state["step"] == 4:
                    self.controller.info("Approach SPEAR complete: starting final alignment; no further Spear")
                    motion = state.pop("spear_motion", None)
                    if motion is not None:
                        self.controller.calibrated_unit.actionHelper.cancel(motion)
                    self.controller.calibrated_unit.stop()
                    state.update(step=1, speared=True, pipette_z=command["pipette_plane"])
                    self.controller.info(f"Approach RELOCATE cell: target_z={state['cell_focus_z']}; final alignment")
                    self.controller.calibrated_stage.focusHelper.relocate(state["cell_focus_z"])
                    self.failure_gate()
                    self.success_gate()
                if state.get("bounded_focus"):
                    self.controller.calibrated_stage.focusHelper.reset_state(state)
                    return
                speed, method = command["speed"], command["method"]
                motion = (state["step"], method, speed)
                if state.get("logged_motion") != motion:
                    detail = ""
                    if state["step"] == 2:
                        raw = state.get("observation", {})
                        vision = raw.get("deep_learning", {})
                        detail = (f"; local pipette focus correction; pipette_z={state.get('pipette_z')}; "
                            f"microscope_z={raw.get('stage_positions', [None, None, None])[2]}; "
                            f"pipette_focus={vision.get('pipette_focus')}; confidence={vision.get('confidence')}")
                    controller.info(f"Approach {method.upper()}: step={state['step']}; "
                        f"speed={speed} um/s; inference continues during motion{detail}")
                    state["logged_motion"] = motion
                controller.microscope.absolute_move_velocity(speed)
                self.controller.sleep(0.1)
            elif command["method"] == "spear":
                state.update(step=4, spear_target=command['target'],
                             spear_start_z=state['observation']['manipulator_position'][2])
                unit, actions = controller.calibrated_unit, controller.calibrated_unit.actionHelper
                try:
                    target, speed = float(command['target']), float(command['speed'])
                    if not np.isfinite([target, speed]).all() or speed <= 0:
                        raise ValueError('Spear requires a finite target and positive finite speed')
                    controller.abort_if_requested()
                    start = float(unit.position()[2])
                    if not np.isfinite(start):
                        raise RuntimeError('Spear has no valid pipette Z position')
                    if target < start - actions.MOTION_ENDPOINT_TOLERANCE_UM:
                        raise ValueError('Spear target is behind the current pipette position')
                    deadline = time.monotonic() + max(10.0, max(0.0, target - start) / speed * 20.0)
                    unit.stop()
                    unit.info(f'SPEAR motion: start_z={start}; target_z={target}; '
                              f'velocity_command={-speed}; endpoint monitor active')
                    state['spear_motion'] = actions.start(unit, target=target, deadline=deadline,
                        read_position=lambda: float(unit.position()[2]),
                        move_velocity=lambda direction: unit.absolute_move_group_velocity([0.0, 0.0, -direction * speed]),
                        abort_check=controller.abort_if_requested, label='Spear motion', info=unit.info,
                        direction=1.0, reversal_origin=start,
                        reversal_tolerance=actions.MOTION_ENDPOINT_TOLERANCE_UM)
                except (RuntimeError, ValueError) as error:
                    raise AutopatchError(str(error)) from error
            else:
                self.finish(state, success=True)
            return
        controller = self.controller
        config = controller.config
        controller.calibrated_stage.set_max_speed(config.max_locate_speed)
        controller.calibrated_unit.set_max_speed(config.max_locate_speed)
        cell = command["cell"]
        if command["route"] == "plate":
            first_alignment_distance = config.cell_distance
            descent_distance = config.slice_start_distance - first_alignment_distance
            controller.move_group_down(descent_distance)
            controller.sleep(0.1)
            controller.fine_calibrate_pipette()
        self.failure_gate()
        self.success_gate()
        if not self.action_gate(command, state):
            raise AutopatchError("Cell type changed during approach")
        controller.amplifier.start_patch()
        if command["route"] == "adaptive":
            # Load models and exercise their slow initial samples before scanning.
            self.controller.info("Approach WARMUP: starting 3 inference samples before motion")
            for sample in range(3):
                warmup = self.observe(fields=["deep_learning"])["deep_learning"]
                self.controller.info(f"Approach WARMUP {sample + 1}/3: finished; "
                    f"stale={warmup.get('stale')}; age_s={warmup.get('age_s')}; "
                    f"focus_confidence_supported={warmup.get('pipette_focus_confidence_supported')}; "
                    f"models={warmup.get('status')}; errors={warmup.get('errors')}")
                self.failure_gate()
                self.success_gate()
            state.update(step=1, cell_focus_z=cell[0][2], goal_distance=command["goal_distance"],
                         fields=["stage_positions", "deep_learning", "manipulator_position"],
                         pipette_z=controller.microscope.position() / controller.calibrated_unit.config.microscope_units_per_um)
            self._vision = True
            self.controller.info(f"Approach RELOCATE: cell_z={state['cell_focus_z']}; "
                f"pipette_z={state['pipette_z']}; goal_distance={state['goal_distance']}")
            self.controller.calibrated_stage.focusHelper.relocate(state["cell_focus_z"])
            self.failure_gate()
            self.success_gate()
            self.controller.info("Approach RELOCATE: finished; starting SEARCH")
            if not state.get("bounded_focus"):
                motion = (state["step"], "search", command["speed"])
                if state.get("logged_motion") != motion:
                    controller.info(f"Approach SEARCH: step={state['step']}; speed={command['speed']} um/s; inference continues during motion")
                    state["logged_motion"] = motion
                controller.microscope.absolute_move_velocity(command["speed"])
            return
        distance = first_alignment_distance if command["route"] == "plate" else config.slice_start_distance
        self.align(cell, distance, config.use_centroid)
        if command["route"] == "slice":
            self.failure_gate()
            self.success_gate()
            if not self.action_gate(command, state):
                raise AutopatchError("Cell type changed during approach")
            self.clear_to_cell(cell)
            self.failure_gate()
            self.success_gate()
            if not self.action_gate(command, state):
                raise AutopatchError("Cell type changed during approach")
            self.align(cell, config.cell_distance, config.use_centroid)
        state["fields"] = ["manipulator_position", "resistance"]
        self.finish(state, success=True)













    def search(self, speed):
        return {"method": "search", "speed": speed}


    def shift(self, target, next_move):
        return {"method": "shift", "target": target, "next_move": next_move}


    def scan(self, speed):
        return {"method": "scan", "speed": speed}


    def spear(self, target, speed):
        return {"method": "spear", "target": target, "speed": speed}


    def action_gate(self, observation=None, state=None):
        config = self.controller.config
        if self._vision or (observation and observation["route"] == "adaptive"):
            low, high = sorted([state.get("cell_focus_z", state["cell"][0][2]), state.get("pipette_z", state["cell"][0][2])])
            sweep = state.get("focus_sweep")
            tolerance = self.controller.calibrated_stage.focusHelper.BOUNDARY_TOLERANCE_UM if sweep is not None else 0.0
            if observation and observation.get("method") == "spear":
                if not (observation["speed"] > 0 and np.isfinite(observation["target"])
                        and observation["target"] >= state["observation"]["manipulator_position"][2]):
                    return False
            return config.mode == "Adaptive" and (not self._vision or (
                not state["observation"]["deep_learning"].get("stale", True)
                and low - 5 - tolerance <= state["observation"]["stage_positions"][2] <= high + 5 + tolerance))
        return observation["route_token"] == (config.cell_type, bool(config.cell_type_toggle) if config.cell_type != "Plate" else None)


    def failure_gate(self, state=None, *, message=None, recover=False):
        if recover:
            return False
        self.controller.abort_if_requested()
        if state is not None:
            self.controller.isrigready()
            if self.controller.rig_ready is False:
                raise AutopatchError("Rig not ready for clearing to cell")
            if state["cell"] is None:
                raise AutopatchError("No cell given to patch!")


    def success_gate(self, observation=None, state=None):
        goal = None if observation is None else observation is True
        return super().success_gate(goal, state)


    def finish(self, state=None, *, success=False):
        if state is not None and state.get("finished", False):
            return
        primary_error = sys.exc_info()[1]
        if primary_error is not None:
            self.controller.info(f"{self.phase_name} FAILED: {primary_error}; stopping motion")
        stop_error = None
        stops = [self.controller.calibrated_unit.stop]
        if state is not None and state.get("spear_motion") is not None:
            stops.insert(0, lambda: self.controller.calibrated_unit.actionHelper.cancel(state["spear_motion"]))
        if state is not None and state.get("focus_sweep") is not None:
            stops.append(lambda: self.controller.calibrated_stage.focusHelper.cancel_sweep(state["focus_sweep"]))
        if state is not None and self._vision:
            stops.extend([self.controller.calibrated_stage.stop, self.controller.microscope.stop])
        for stop in stops:
            try:
                stop()
            except Exception as error:
                if primary_error is not None:
                    if hasattr(primary_error, "add_note"):
                        primary_error.add_note(f"Approach stop also failed: {error}")
                elif stop_error is None:
                    stop_error = error
        if stop_error is not None:
            raise stop_error
        if state is not None:
            state["finished"] = True
        if success:
            if self._vision and state is not None and "cell_focus_z" in state:
                self.controller._approach_cell_focus = {
                    "cell_reference": state.get("original_cell_reference", state["cell"][0]),
                    "z": float(state["cell_focus_z"]), "tracked_cell": state["cell"],
                    "template_point": state.get("cell_template_point")}
            self.controller.calibrated_stage.set_max_speed(10000)
            self.controller.calibrated_unit.set_max_speed(100000)
            self.controller.info("Approached Cell")
            self.success_gate(True, state)
            self.goal_event = False
            if not self.awaiting_operator:
                self.complete_success()


    def align(self, cell, cell_distance, use_centroid):
        '''
        Aligns the pipette to the cell using microscope imaging
        '''
        self.controller.info("Aligning pipette to cell using imaging")
        cell_pos, _, _ = cell

        self.controller.microscope.move_to_floor()
        self.controller.microscope.wait_until_still()
        z_pos = self.controller.microscope.position() / self.controller.calibrated_unit.config.microscope_units_per_um
        zdistleft = z_pos - cell_pos[2]
        self.controller.microscope.relative_move(-zdistleft)
        self.controller.microscope.wait_until_still()

        if self.controller.config.cell_type_toggle:
            self.controller.info("centering on cell")
            self.controller.calibrated_stage.center_on_cell(
                cell, use_centroid=use_centroid,
                max_error_px=self.controller.config.approach_cell_max_shift)
            self.controller.calibrated_stage.wait_until_still()
            self.controller.info(f"correcting pipette position, moving microscope by {zdistleft} um")
            self.controller.microscope.relative_move(-cell_distance)
            self.controller.microscope.wait_until_still()
            self.controller.calibrated_unit.center_pipette(speed=self.controller.config.approach_pipette_speed)
            self.controller.calibrated_unit.wait_until_still()
            self.controller.microscope.relative_move(cell_distance)
            self.controller.microscope.wait_until_still()


    def clear_to_cell(self, cell):
        motion_started = False
        try:
            self.failure_gate({"cell": cell})
            self.success_gate()
            controller = self.controller
            config = controller.config
            controller.info("Clearing to cell")
            if not (config.cell_type_toggle and config.cell_type == "Slice"):
                return
            controller.info("Moving pipette to slice position")
            cell_hover_pos = config.cell_distance - config.slice_start_distance
            controller.microscope.relative_move(-config.cell_distance)
            start_pos = controller.calibrated_unit.position()
            motion_started = True
            controller.calibrated_unit.absolute_move_group_velocity([0, 0, config.max_clearing_speed])
            controller.info(f"Cell hover position: {cell_hover_pos} um")
            while start_pos[2] - controller.calibrated_unit.position()[2] > cell_hover_pos:
                self.failure_gate()
                self.success_gate()
                controller.sleep(0.1)
        finally:
            if motion_started:
                self.finish()
