"""Cell-type-dependent approach from the common localization distance."""

import sys
import logging
import numpy as np
from ..errors import AutopatchError
from ..PhaseController import PhaseController
from .approach_focus import FocusSweep
from .approach_focus_confirmation import FocusConfirmation
from .spear_motion import SpearMotion


class ApproachCellPhase(PhaseController):
    def run(self, cell=None):
        state = None
        camera = self.controller.calibrated_stage.camera
        indicator_owner = object()
        try:
            try:
                camera.begin_pipette_indicator(indicator_owner)
            except Exception as error:
                logging.getLogger("patcherbot.detection_overlay").warning(
                    f"Could not switch Approach pipette indicator: {error}. Approach continues.")
            state = self.prepare(cell)
            while True:
                observation = self.observe(state=state, fields=state["fields"], evidence={})
                calculated = self.calculate(observation, state)
                command = self.decide(calculated, state)
                self.act(command, state)
        finally:
            try:
                if sys.exc_info()[1] is not None:
                    self.controller.info(f"Approach FAILED: {sys.exc_info()[1]}; stopping motion")
                self.finish(state)
            finally:
                try:
                    camera.end_pipette_indicator(indicator_owner)
                except Exception as error:
                    logging.getLogger("patcherbot.detection_overlay").warning(
                        f"Could not restore pipette indicator: {error}. Approach cleanup continues.")


    def prepare(self, cell=None):
        self.failure_gate()
        self.success_gate()
        self.begin_observations()
        self._adaptive = self.controller.config.mode == "Adaptive"
        self.controller._approach_cell_focus = None
        self._vision = False
        self._marker_warning_shown = False
        if self._adaptive:
            try:
                self.controller.calibrated_stage.camera.clear_detection_markers()
            except Exception as error:
                logging.getLogger("patcherbot.detection_overlay").warning(
                    f"Could not reset detection markers: {error}. Approach continues.")
        self.controller.info(f"Approach START: mode={self.controller.config.mode}")
        return {"cell": cell, "original_cell_reference": None if cell is None else cell[0],
                "fields": [], "step": 0, "display_detection_markers": True,
                "bounded_focus": True, "cell_search_radius_um": 50.0}

    def observe(self, *, state=None, **kwargs):
        searching = self._vision and state is not None and state["step"] == 1
        if self._vision:
            if state is not None and state.get("bounded_focus"):
                self.frame_context["pipette_expected_xy"] = self.controller.expected_pipette_xy
                self.frame_context["focus_z"] = lambda: (self.controller.microscope.position()
                    / self.controller.calibrated_unit.config.microscope_units_per_um)
                if searching:
                    self.frame_context["stage_xy"] = self.controller.calibrated_stage.position
                    kwargs["fields"] = [*kwargs.get("fields", state.get("fields", [])), "camera_image"]
            inference_fields = ["pipette_image_xy", "pipette_defocus_um"]
            if searching:
                inference_fields.append("cell_detections")
            kwargs["evidence"] = self.controller.infer_frame(inference_fields, frame_context=self.frame_context)
        else:
            kwargs["evidence"] = kwargs.get("evidence")
        observation = super().observe(**kwargs)
        if self._vision:
            vision = observation["deep_learning"]
            tip = vision.get("pipette_position")
            if state is not None and state.get('step') == 2:
                confirmation = state.get('focus_confirmation')
                sweep = state.get('focus_sweep')
                tip = (confirmation.confirmed_tip_xy if confirmation is not None and confirmation.confirmed
                       else getattr(sweep, 'best_tip_xy', None))
                if tip is None:
                    tip = self.controller.expected_pipette_xy()
            if tip is not None:
                self._show_detection_marker("pipette", tip, (255, 0, 0))
        return observation

    def _show_detection_marker(self, name, point, color):
        """Keep display failures separate from detection and motion decisions."""
        try:
            stage, unit = self.controller.calibrated_stage, self.controller.calibrated_unit
            position_reader = (lambda: stage.reference_position()[:2]) if name == "cell" else (
                lambda: unit.um_to_pixels_relative(unit.position())[:2]
                         + stage.um_to_pixels_relative(stage.position())[:2])
            marker_options = ({"shape": "triangle", "direction": np.asarray(unit.M)[:2, 0]}
                              if name == "pipette" else {})
            stage.camera.show_detection_marker(name, point, color=color, position_reader=position_reader, **marker_options)
        except Exception as error:
            if not getattr(self, "_marker_warning_shown", False):
                self._marker_warning_shown = True
                logging.getLogger("patcherbot.detection_overlay").warning(
                    f"Could not display {name} detection marker: {error}. Approach continues.")

    def _fuse_cell_tracks(self, candidates, expected_xy, template_xy=None, previous_xy=None):
        """Soft spatial support selects a real soma before combining nearby estimates.

        Weights are reliability heuristics, not calibrated model probabilities.
        Calibration is a weak prior; disagreement reduces influence continuously.
        """
        stage = self.controller.calibrated_stage
        scale = FocusConfirmation.TIP_STABILITY_UM
        def support(a, b):
            delta = stage.pixels_to_um_relative(np.asarray(a)[:2] - np.asarray(b)[:2])
            return 1.0 / (1.0 + (np.linalg.norm(delta[:2]) / scale) ** 2)
        def rank(candidate):
            xy = np.asarray(candidate[:2])
            evidence = 0.35 * support(xy, expected_xy)
            if template_xy is not None:
                evidence += support(xy, template_xy)
            if previous_xy is not None:
                evidence += 0.5 * support(xy, previous_xy)
            return float(candidate[2]) * evidence
        selected = max(candidates, key=rank, default=None)
        if selected is None:
            return None, None
        xy = np.asarray(selected[:2], dtype=float)
        confidence = max(float(selected[2]), 1e-6)
        points = [xy, np.asarray(expected_xy, dtype=float)]
        reliability = [confidence, 0.35]
        if template_xy is not None:
            points.append(np.asarray(template_xy, dtype=float))
            reliability.append(1.0)
        # No source supports itself: even a confident detector must earn support.
        weights = [quality * sum(support(point, other)
                                 for j, other in enumerate(points) if j != i)
                   for i, (point, quality) in enumerate(zip(points, reliability))]
        return selected, np.average(points, axis=0, weights=weights)

    def _cell_template_evidence(self, observation, state, expected):
        """Refresh independent identity evidence only on a stationary SEARCH frame."""
        state['cell_template_evidence'] = None
        image = observation.get('camera_image')
        if image is None:
            return
        stage = self.controller.calibrated_stage
        try:
            helper = stage._ensure_cell_track_helper()
            point = helper.find_centroid(state['cell'][1], image,
                use_centroid=self.controller.config.use_centroid,
                prompt_point=state.get('cell_template_point'), expected_point=expected)
            evidence = getattr(helper, 'last_match_evidence', {})
            if point is not None and evidence.get('independent', False):
                source = observation['deep_learning'].get('source_positions', {})
                source_xy = np.asarray(source.get('stage_xy', observation['stage_positions'][:2]))[:2]
                state['cell_template_evidence'] = (stage.M @ source_xy + stage.r0)[:2] - np.asarray(point[:2])
            self.controller.info(f'Cell template: evidence={evidence}; xy={point}; '
                                 'stage fallback is not counted as an independent template match')
        except Exception as error:
            self.controller.info(f'Cell template unavailable: {error}; using calibration and detector evidence')

    def _shift_cell_from_evidence(self, state):
        """Use one accepted detector/template/calibration estimate for marker and motion."""
        controller, raw = self.controller, state['observation']
        stage = controller.calibrated_stage
        if not getattr(controller.config, 'cell_shift_fusion', True):
            controller.info('Cell SHIFT consensus disabled; using existing template centering')
            return stage.center_on_cell(state['cell'], use_centroid=controller.config.use_centroid,
                max_error_px=controller.config.approach_cell_max_shift)
        detection, image = raw.get('cell_detection'), raw.get('camera_image')
        options = {}
        if detection is not None and image is not None:
            source_xy = np.asarray(raw['deep_learning'].get('source_positions', {}).get(
                'stage_xy', raw['stage_positions'][:2]))[:2]
            expected = (stage.M @ source_xy + stage.r0)[:2] - state['cell'][0][:2]
            self._cell_template_evidence(raw, state, expected)
            template = state.get('cell_template_evidence')
            template_xy = None if template is None else (stage.M @ source_xy + stage.r0)[:2] - template
            recheck = state.pop('cell_shift_recheck', None)
            frame = raw['deep_learning'].get('source_frame', raw.get('observation_id'))
            ambiguous = False
            if template_xy is not None:
                points = [np.asarray(detection[:2]), expected, template_xy]
                gaps = [np.linalg.norm(stage.pixels_to_um_relative(points[i] - points[j])[:2])
                        for i in range(3) for j in range(i)]
                ambiguous = min(gaps) > FocusConfirmation.TIP_STABILITY_UM
                if ambiguous and frame is not None and recheck is None:
                    state['cell_shift_recheck'] = (frame, np.asarray(detection[:2]).copy())
                    controller.info('Cell SHIFT: all three estimates disagree; requesting one fresh settled observation before motion')
                    return False
            _, fused = self._fuse_cell_tracks([detection], expected, template_xy)
            if recheck is not None:
                stable = np.linalg.norm(stage.pixels_to_um_relative(
                    np.asarray(detection[:2]) - recheck[1])[:2]) <= FocusConfirmation.TIP_STABILITY_UM
                if frame is None or frame == recheck[0] or raw['deep_learning'].get('stale', False) or not stable:
                    fused = expected.copy()
                    controller.warning('Cell SHIFT: recheck lacks fresh stable evidence for the selected cell; retaining calibration estimate')
            if ambiguous:
                correction = fused - expected
                distance = np.linalg.norm(stage.pixels_to_um_relative(correction)[:2])
                if distance > FocusConfirmation.TIP_STABILITY_UM:
                    fused = expected + correction * (FocusConfirmation.TIP_STABILITY_UM / distance)
                controller.warning('Cell SHIFT: no agreeing pair; model correction limited to 5 um from calibration estimate')
            reference = np.array(state['cell'][0], dtype=float, copy=True)
            reference[:2] = (stage.M @ source_xy + stage.r0)[:2] - fused
            reference[2] = state['cell_focus_z']
            state.setdefault('original_cell_reference', state['cell'][0])
            state['cell'] = (reference, np.asarray(image).copy(), np.append(source_xy, state['cell_focus_z']))
            state['tracked_cell_reference'] = reference[:2].copy()
            state['cell_template_point'] = fused.copy()
            current_xy = fused + (stage.M @ (np.asarray(stage.position())[:2] - source_xy))[:2]
            options['detected_px'] = current_xy
            self._show_detection_marker('cell', current_xy, (0, 0, 255))
            controller.info(f'Cell SHIFT consensus: prior_xy={expected}; template_xy={template_xy}; '
                f'detector_xy={detection[:2]}; detector_confidence={detection[2]}; '
                f'fused_source_xy={fused}; current_xy={current_xy}; z={reference[2]}; '
                'weights=confidence and pairwise spatial support, not calibrated probabilities')
        else:
            if state.pop('cell_shift_recheck', None) is not None:
                options['detected_px'] = (stage.M @ np.asarray(stage.position())[:2] + stage.r0)[:2] - state['cell'][0][:2]
                self._show_detection_marker('cell', options['detected_px'], (0, 0, 255))
                controller.warning('Cell SHIFT: recheck missing detector or source image; retaining calibration estimate')
            else:
                controller.info('Cell SHIFT consensus unavailable: missing detector or source image; using existing template centering')
        stage.center_on_cell(state['cell'], use_centroid=controller.config.use_centroid,
            max_error_px=controller.config.approach_cell_max_shift, **options)

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
                    self._show_detection_marker("cell", cell[:2], (0, 0, 255))
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
                observation["spear_done"] = motion.update() if motion is not None else None
                pipette_z = (self.controller.calibrated_unit.position()[2] if observation["spear_done"]
                             else observation["manipulator_position"][2])
                observation["spear_remaining"] = state["spear_target"] - pipette_z
                observation["pipette_plane"] = state["pipette_z"] + pipette_z - state["spear_start_z"]
        observation["spear_speed"] = min(abs(self.controller.config.max_clearing_speed), 100.0)
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
            if state.get("bounded_focus") and step in (1, 2) and state.get("focus_done") != step:
                return {**command, "route": "adaptive", "method": "focus"}
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
                tip = vision.get("pipette_position")
                detector_confidence = confidence.get("pipette_detector")
                pipette_detected = (step == 2 and tip is not None and np.shape(tip) == (2,)
                    and np.isfinite(tip).all() and detector_confidence is not None
                    and np.isfinite(detector_confidence)
                    and detector_confidence >= self.controller.calibrated_unit.pipetteCalHelper.pipetteDetector.adapter.threshold)
                valid = ((cell is not None and cell[2] >= self.controller.calibrated_stage.cellDetectHelper.cellDetector.conf_threshold)
                         if step == 1 else pipette_detected)
                confirmation = state["focus_confirmation"]
                ready = confirmation.update(observation, valid)
                if ready is not True:
                    return None
                if step == 2:
                    state["confirmed_tip_xy"] = confirmation.confirmed_tip_xy
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
        if (command is not None or self._vision) and not self.action_gate(command, state):
            raw = state.get("observation", {})
            vision = raw.get("deep_learning", {})
            planes = (state.get("cell_focus_z"), state.get("pipette_z"))
            sweep = state.get("focus_sweep")
            tolerance = sweep.BOUNDARY_TOLERANCE_UM if sweep is not None else 0.0
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
                sweep = state.get("focus_sweep")
                if sweep is None:
                    kind = "SEARCH" if state["step"] == 1 else "SCAN"
                    center = state["cell_focus_z"] if state["step"] == 1 else state["pipette_z"]
                    sweep = state["focus_sweep"] = FocusSweep(self.controller, center, kind)
                    sweep.start()
                elif sweep.update(state["observation"], self._focus_score(state)):
                    state["focus_done"] = state["step"]
                    confirmation_options = ({'selected_tip_xy': sweep.best_tip_xy} if state['step'] == 2
                                            else {'selected_cell_xy': sweep.best_cell_xy})
                    state["focus_confirmation"] = FocusConfirmation(
                        self.controller, "cell" if state["step"] == 1 else "pipette", **confirmation_options)
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
                    options = ({"detected_px": state.pop("confirmed_tip_xy")} if "confirmed_tip_xy" in state else {})
                    if options:
                        options.update(expected_px=self.controller.expected_pipette_xy(),
                                       max_deviation_um=50.0)
                        self._show_detection_marker('pipette', options['detected_px'], (255, 0, 0))
                    centered = self.controller.calibrated_unit.center_pipette(speed=self.controller.config.approach_pipette_speed, **options)
                    self.failure_gate()
                    self.success_gate()
                    if centered is False:
                        next_action = "final alignment incomplete" if state.get("speared") else "SPEAR cancelled"
                        reason = getattr(self.controller.calibrated_unit, "last_pipette_move_failure", None)
                        if not isinstance(reason, str) or not reason:
                            reason = "pipette centering did not complete"
                        raise AutopatchError(f"Pipette SHIFT failed: {reason}; {next_action}")
                self.controller.info(f"Approach SHIFT {command['target']}: finished")
                state.pop("focus_sweep", None)
                state.pop("focus_done", None)
                state.pop("focus_confirmation", None)
                state["step"] += 1
                if command["target"] == "cell":
                    self.failure_gate()
                    self.success_gate()
                    self.controller.info(f"Approach RELOCATE pipette: target_z={state['pipette_z']}")
                    self.controller.microscope.absolute_move(state["pipette_z"])
                    self.controller.microscope.wait_until_still()
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
                self.controller.microscope.absolute_move(state["hover_z"])
                self.controller.microscope.wait_until_still()
                self.failure_gate()
                self.success_gate()
                self.controller.info("Approach RELOCATE stand-off: finished; checking position before SPEAR")
                return
            if command["method"] in ("search", "scan"):
                if state["step"] == 4:
                    self.controller.info("Approach SPEAR complete: starting final alignment; no further Spear")
                    motion = state.pop("spear_motion", None)
                    if motion is not None:
                        motion.stop()
                    self.controller.calibrated_unit.stop()
                    state.update(step=1, speared=True, pipette_z=command["pipette_plane"])
                    self.controller.info(f"Approach RELOCATE cell: target_z={state['cell_focus_z']}; final alignment")
                    self.controller.microscope.absolute_move(state["cell_focus_z"])
                    self.controller.microscope.wait_until_still()
                    self.failure_gate()
                    self.success_gate()
                if state.get("bounded_focus"):
                    state.pop("focus_sweep", None)
                    state.pop("focus_done", None)
                    state.pop("focus_confirmation", None)
                    return
                self._move_microscope(command["speed"], state, method=command["method"])
                self.controller.sleep(0.1)
            elif command["method"] == "spear":
                state.update(step=4, spear_target=command["target"], spear_start_z=state["observation"]["manipulator_position"][2])
                motion = state["spear_motion"] = SpearMotion(self.controller, command["target"], command["speed"])
                motion.start()
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
            controller.microscope.absolute_move(state["cell_focus_z"])
            controller.microscope.wait_until_still()
            self.failure_gate()
            self.success_gate()
            self.controller.info("Approach RELOCATE: finished; starting SEARCH")
            if not state.get("bounded_focus"):
                self._move_microscope(command["speed"], state, method="search")
            return
        distance = first_alignment_distance if command["route"] == "plate" else config.slice_start_distance
        controller.align(cell, distance, config.use_centroid)
        if command["route"] == "slice":
            self.failure_gate()
            self.success_gate()
            if not self.action_gate(command, state):
                raise AutopatchError("Cell type changed during approach")
            controller.clear_to_cell(cell)
            self.failure_gate()
            self.success_gate()
            if not self.action_gate(command, state):
                raise AutopatchError("Cell type changed during approach")
            controller.align(cell, config.cell_distance, config.use_centroid)
        state["fields"] = ["manipulator_position", "resistance"]
        self.finish(state, success=True)


    def _focus_score(self, state):
        observation = state["observation"]
        vision = observation["deep_learning"]
        if state["step"] == 2:
            confidence = vision.get("confidence", {})
            score = confidence.get("pipette_detector")
            tip = vision.get("pipette_position")
            if (tip is not None and np.shape(tip) == (2,) and np.isfinite(tip).all()
                    and score is not None and np.isfinite(score)
                    and score >= self.controller.calibrated_unit.pipetteCalHelper.pipetteDetector.adapter.threshold):
                return float(score)
            return None
        cell = observation.get("cell_detection")
        if (cell is None or len(cell) < 3 or not np.isfinite(cell[:3]).all()
                or cell[2] < self.controller.calibrated_stage.cellDetectHelper.cellDetector.conf_threshold):
            return None
        return float(cell[2])

    def _move_microscope(self, speed, state, *, method):
        """Log motion transitions while retaining continuous Search/Scan movement."""
        motion = (state["step"], method, speed)
        if state.get("logged_motion") != motion:
            detail = ""
            if state["step"] == 2:
                raw = state.get("observation", {})
                vision = raw.get("deep_learning", {})
                detail = (f"; local pipette focus correction; pipette_z={state.get('pipette_z')}; "
                    f"microscope_z={raw.get('stage_positions', [None, None, None])[2]}; "
                    f"pipette_focus={vision.get('pipette_focus')}; confidence={vision.get('confidence')}")
            self.controller.info(f"Approach {method.upper()}: step={state['step']}; "
                f"speed={speed} um/s; inference continues during motion{detail}")
            state["logged_motion"] = motion
        self.controller.microscope.absolute_move_velocity(speed)

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
            tolerance = sweep.BOUNDARY_TOLERANCE_UM if sweep is not None else 0.0
            if observation and observation.get("method") == "spear":
                if not (observation["speed"] > 0 and np.isfinite(observation["target"])
                        and observation["target"] >= state["observation"]["manipulator_position"][2]):
                    return False
            return config.mode == "Adaptive" and (not self._vision or (
                not state["observation"]["deep_learning"].get("stale", True)
                and low - 5 - tolerance <= state["observation"]["stage_positions"][2] <= high + 5 + tolerance))
        return observation["route_token"] == (config.cell_type, bool(config.cell_type_toggle) if config.cell_type != "Plate" else None)


    def failure_gate(self, state=None):
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
        stop_error = None
        devices = [self.controller.calibrated_unit]
        if state is not None and state.get("spear_motion") is not None:
            devices.insert(0, state["spear_motion"])
        if state is not None and state.get("focus_sweep") is not None:
            devices.append(state["focus_sweep"])
        if state is not None and self._vision:
            devices.extend([self.controller.calibrated_stage, self.controller.microscope])
        for device in devices:
            try:
                device.stop()
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
