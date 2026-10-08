"""Historical Hunt behavior within the shared phase lifecycle."""

import collections
import logging
import sys
import time
import numpy as np

from ..errors import AutopatchError
from ..PhaseController import PhaseController
from .approach_cell import ApproachCellPhase
from .spear_motion import SpearMotion
from .approach_focus import EmptyFocusEvidence, FocusSweep
from .approach_focus_confirmation import FocusConfirmation


class HuntCellPhase(PhaseController):
    search = ApproachCellPhase.search
    shift = ApproachCellPhase.shift
    scan = ApproachCellPhase.scan
    spear = ApproachCellPhase.spear
    _fuse_cell_tracks = ApproachCellPhase._fuse_cell_tracks
    _cell_template_evidence = ApproachCellPhase._cell_template_evidence
    _shift_cell_from_evidence = ApproachCellPhase._shift_cell_from_evidence
    _focus_score = ApproachCellPhase._focus_score
    _show_detection_marker = ApproachCellPhase._show_detection_marker
    CONTACT_READINGS = 5
    CONTACT_TIMEOUT_S = 5.0
    DEPTH_TOLERANCE_UM = 2.0

    def run(self, cell=None):
        state = None
        camera = None
        indicator_owner = object()
        self._hunt_marker_warning_shown = False
        try:
            try:
                camera = self.controller.calibrated_stage.camera
                camera.begin_pipette_indicator(indicator_owner)
            except Exception as error:
                logging.getLogger('patcherbot.detection_overlay').warning(
                    f'Could not start Hunt pipette indicator: {error}. Hunt continues.')
            state = self.prepare(cell)
            while True:
                observation = self.observe(state=state)
                calculated = self.calculate(observation, state)
                command = self.decide(calculated, state)
                self.act(command, state)
        finally:
            try:
                self.finish(state)
            finally:
                if camera is not None:
                    try:
                        camera.end_pipette_indicator(indicator_owner)
                    except Exception as error:
                        logging.getLogger('patcherbot.detection_overlay').warning(
                            f'Could not restore pipette indicator after Hunt: {error}.')

    def prepare(self, cell=None):
        self.failure_gate()
        self.success_gate()
        controller = self.controller
        controller.isrigready()
        if controller.rig_ready == False:
            self.failure_gate(message="Rig not ready for cell hunting")
        if cell is None:
            self.failure_gate(message="No cell given to patch!")
        self.begin_observations()
        self.observation_windows = {}
        self.frame_context = {}
        controller.info("Hunting for cell")
        controller.pressure.set_pressure(controller.config.pressure_near)
        controller.sleep(3)
        state = dict(initialized=False, start_position=None,
                    readings=collections.deque(maxlen=5), last_velocity=None,
                    model_prepared=False, mode=None, cell=cell, step=0,
                    goal_distance=0, confirming=False, confirmed_readings=0,
                    started_at=time.monotonic(), last_frame=None,
                    last_confirmation_id=None)
        if controller.config.mode == "Adaptive":
            self._adaptive = self._vision = True
            self._marker_warning_shown = False
            controller.info("Hunt WARMUP: 3 inference samples before motion")
            for sample in range(3):
                controller.infer_frame(["pipette_image_xy", "pipette_defocus_um", "cell_detections"])
                self.failure_gate()
                self.success_gate()
                controller.info(f"Hunt WARMUP {sample + 1}/3: finished")
            cell_focus_z = cell[0][2]
            approached = getattr(controller, "_approach_cell_focus", None)
            if (isinstance(approached, dict) and approached.get("cell_reference") is cell[0]
                    and np.isfinite(approached.get("z", np.nan))):
                cell_focus_z = approached["z"]
                if approached.get("tracked_cell") is not None:
                    state["cell"] = approached["tracked_cell"]
                    state["tracked_cell_reference"] = np.asarray(state["cell"][0][:2]).copy()
                    state["cell_template_point"] = approached.get("template_point")
                controller.info(f"Hunt: retaining Approach's confirmed cell plane z={cell_focus_z}")
            state.update(step=1, bounded_focus=True, display_detection_markers=False,
                         cell_focus_z=cell_focus_z,
                         pipette_z=controller.microscope.position() / controller.calibrated_unit.config.microscope_units_per_um)
            controller.info(f"Hunt RELOCATE cell: target_z={state['cell_focus_z']}; pipette_z={state['pipette_z']}")
            controller.microscope.absolute_move(state["cell_focus_z"])
            controller.microscope.wait_until_still()
            self.failure_gate()
            self.success_gate()
            state["started_at"] = time.monotonic()
        return state

    def observe(self, *, state):
        self.failure_gate()
        self.success_gate()
        controller = self.controller
        mode = controller.config.mode
        if state["initialized"]:
            position = super().observe(fields=["manipulator_position"], evidence={})
            self.failure_gate(state, {**position, "mode": mode})
            controller.sleep(0.04)
        fields = ["resistance", "manipulator_position"]
        evidence = {}
        if mode == "Adaptive":
            fields += ["stage_positions", "deep_learning"]
            self.frame_context["pipette_expected_xy"] = controller.expected_pipette_xy
            self.frame_context["pipette_tracking_xy"] = controller.calibrated_unit.tracked_tip_position
            self.frame_context["pipette_xyz"] = controller.calibrated_unit.position
            self.frame_context["stage_xy"] = controller.calibrated_stage.position
            inference_fields = ["pipette_image_xy", "pipette_defocus_um"]
            if state["step"] in (1, 4) or state["confirming"]:
                inference_fields.append("cell_detections")
            if state["step"] == 1:
                fields.append("camera_image")
            self.frame_context["focus_z"] = lambda: (controller.microscope.position()
                / controller.calibrated_unit.config.microscope_units_per_um)
            evidence = controller.infer_frame(inference_fields, frame_context=self.frame_context)
        observation = super().observe(fields=fields,
            evidence=evidence, raw_resistance=True)
        if mode == "Adaptive" and state["step"] == 4:
            self._track_spear_tip(observation)
        baseline = (controller.first_res if state["initialized"]
                    else controller.resistanceRamp())
        controller.observer.annotate(self, observation, fields={
            "mode": mode, "baseline_resistance": baseline})
        self._show_pipette_indicator(observation, state)
        self.failure_gate()
        self.success_gate()
        return observation

    def _track_spear_tip(self, observation):
        """Keep the calibrated trajectory authoritative during an ongoing Spear."""
        vision = observation.get('deep_learning', {})
        source = vision.get('source_positions', {})
        expected = source.get('pipette_expected_xy')
        if expected is None:
            return
        expected = np.asarray(expected, dtype=float)
        if expected.shape != (2,) or not np.isfinite(expected).all():
            return
        tip = vision.get('pipette_position')
        accepted = False
        if tip is not None and not vision.get('stale', True):
            delta = self.controller.calibrated_stage.pixels_to_um_relative(np.asarray(tip)[:2] - expected)
            accepted = np.linalg.norm(delta[:2]) <= 50.0
        if not accepted:
            if tip is not None:
                self.controller.info(f'Hunt SPEAR pipette: rejected detection={tip}; expected_xy={expected}; '
                                     'plausibility_radius_um=50; following calibrated trajectory')
            vision['pipette_position'] = None
        tracking = source.get('pipette_tracking_xy')
        vision['pipette_tracking_position'] = expected if tracking is None else np.asarray(tracking)[:2]
        vision['pipette_tracking_source'] = 'calibrated trajectory from last accepted tip'

    def _show_pipette_indicator(self, observation, state=None):
        """Display existing tip evidence, with calibrated motion between frames."""
        try:
            unit, stage = self.controller.calibrated_unit, self.controller.calibrated_stage
            vision = observation.get('deep_learning', {})
            if vision.get('stale', True) and 'pipette_tracking_position' not in vision:
                return
            tip = vision.get('pipette_tracking_position', vision.get('pipette_position'))
            if state is not None and state.get('step') == 2:
                confirmation = state.get('focus_confirmation')
                sweep = state.get('focus_sweep')
                tip = (confirmation.confirmed_tip_xy if confirmation is not None and confirmation.confirmed
                       else getattr(sweep, 'best_tip_xy', None))
                if tip is None:
                    tip = self.controller.expected_pipette_xy()
            if tip is None:
                return
            tip = np.asarray(tip, dtype=float)
            if tip.shape != (2,) or not np.isfinite(tip).all():
                return
            position_reader = lambda: (unit.um_to_pixels_relative(unit.position())[:2]
                                       + stage.um_to_pixels_relative(stage.position())[:2])
            source = vision.get('source_positions', {})
            if 'pipette_xyz' in source and 'stage_xy' in source:
                tip = (tip + unit.um_to_pixels_relative(unit.position() - np.asarray(source['pipette_xyz']))[:2]
                       + stage.um_to_pixels_relative(stage.position() - np.asarray(source['stage_xy']))[:2])
            stage.camera.show_detection_marker('pipette', tip, color=(255, 0, 0),
                shape='triangle', direction=unit.M[:2, 0], position_reader=position_reader)
        except Exception as error:
            if not getattr(self, '_hunt_marker_warning_shown', False):
                self._hunt_marker_warning_shown = True
                logging.getLogger('patcherbot.detection_overlay').warning(
                    f'Could not display Hunt pipette indicator: {error}. Hunt continues.')

    def calculate(self, observation=None, state=None):
        self.failure_gate()
        self.success_gate()
        mode = observation["mode"]
        if mode != self.controller.config.mode:
            return {"observation": observation, "mode": mode, "stale": True}
        if not state["initialized"]:
            state["start_position"] = observation["manipulator_position"]
            self.controller.first_res = observation["baseline_resistance"]
            state["initialized"] = True
            self.controller.info(f"Initial resistance: {self.controller.first_res}")
        state["readings"].append(observation["resistance"])
        ready = len(state["readings"]) >= 5
        delta = (state["readings"][-1] - self.controller.first_res) if ready else None
        travel = None if mode == "Training" else abs(
            observation["manipulator_position"][2] - state["start_position"][2])
        calculated = {"observation": observation, "mode": mode, "stale": False,
                      "window_ready": ready, "resistance_delta": delta,
                      "travel_um": travel}
        if mode == "Adaptive":
            config = self.controller.config
            unit, stage = self.controller.calibrated_unit, self.controller.calibrated_stage
            if "cell_focus_z" not in state:
                distance = (config.cell_distance if config.cell_type == "Plate"
                            or config.cell_type_toggle else config.slice_start_distance)
                state.update(cell_focus_z=state["cell"][0][2],
                             pipette_z=state["cell"][0][2] - distance)
            self._adaptive = self._vision = True
            vision = observation["deep_learning"]
            confidence = vision.get("confidence", {})
            frame = vision.get("source_frame")
            fresh = (not vision.get("stale", True) and frame is not None
                     and frame != state["last_frame"])
            state["last_frame"] = frame
            source_stage = np.asarray(vision.get('source_positions', {}).get('stage_xy', observation['stage_positions'][:2]))[:2]
            reference = state.get('tracked_cell_reference', state['cell'][0][:2])
            expected = (stage.M @ source_stage + stage.r0)[:2] - reference
            cells = [cell for cell in (vision.get("cell_detections") or [])
                     if np.isfinite(cell[:3]).all()
                     and cell[2] >= stage.cellDetectHelper.cellDetector.conf_threshold]
            if state['step'] == 1 and 'cell_template_evidence' not in state:
                self._cell_template_evidence(observation, state, expected)
            template = state.get('cell_template_evidence')
            template_xy = (None if template is None else
                (stage.M @ source_stage + stage.r0)[:2] - template)
            previous = state.get('cell_detector_reference')
            previous_xy = (None if previous is None else
                (stage.M @ source_stage + stage.r0)[:2] - previous)
            cell, fused = self._fuse_cell_tracks(cells, expected, template_xy, previous_xy)
            displaced_search = state['step'] == 4 or 'cell_reacquisition_origin_z' in state
            if cell is not None and config.hunt_limit_target_radius and not displaced_search:
                offset = unit.pixels_to_um_relative(np.append(cell[:2] - expected, 0))
                if np.linalg.norm(offset) > config.hunt_target_radius:
                    cell = None
            observation["cell_detection"] = cell
            observation['cell_selection_locked'] = True
            observation['cell_fused_xy'] = fused if cell is not None else None
            calculated.update(ApproachCellPhase.calculate(self, observation, state))
            if fresh and cell is not None:
                state['cell_detector_reference'] = (stage.M @ source_stage + stage.r0)[:2] - np.asarray(cell[:2])
                display_xy = np.asarray(cell[:2]) + (stage.M @ (observation['stage_positions'][:2] - source_stage))[:2]
                self._show_detection_marker("cell", display_xy, (0, 0, 255))
                if state['step'] == 1:
                    self.controller.info(f'Hunt Cell SEARCH: candidates={len(cells)}; prior_xy={expected}; '
                        f'template_xy={template_xy}; selected={cell}; fused_xy={fused}; '
                        f'focus_error_um={observation["cell_focus_error"]}')
                if state['step'] == 4:
                    self.controller.info(f'Hunt SPEAR cell: xy={cell[:2]}; confidence={cell[2]}; '
                                         'tracking observed displacement independently of pipette motion')
            tip = vision.get("pipette_position")
            tip_ok = (tip is not None and np.isfinite(tip).all()
                      and (confidence.get("pipette_detector") or 0)
                      >= unit.pipetteCalHelper.pipetteDetector.adapter.threshold)
            gap = float("inf")
            if cell is not None and tip_ok:
                gap = np.linalg.norm(unit.pixels_to_um_relative(np.append(np.asarray(tip)[:2] - cell[:2], 0)))
            aligned = fresh and gap <= config.hunt_contact_distance
            pipette_plane = calculated.get("pipette_plane", state["pipette_z"])
            distance_to_cell = state["cell_focus_z"] - pipette_plane
            plane_tolerance = float(getattr(config, 'hunt_contact_plane_tolerance_um', 2.0))
            contact_depth_ready = abs(distance_to_cell) <= plane_tolerance
            touching = (aligned
                        and contact_depth_ready
                        and abs(calculated["cell_focus_error"]) <= self.DEPTH_TOLERANCE_UM)
            rise = bool(np.isfinite(observation["resistance"])
                        and observation["resistance"] - self.controller.first_res >= config.cell_R_increase)
            if state["confirming"]:
                stopped = np.allclose(observation["manipulator_position"], state["stop_position"], atol=0.01, rtol=0)
                state["stop_position"] = np.array(observation["manipulator_position"], copy=True)
                sample = observation["observation_id"]
                if sample != state["last_confirmation_id"]:
                    state["last_confirmation_id"] = sample
                    good = fresh and stopped and rise and touching
                    state["confirmed_readings"] = state["confirmed_readings"] + 1 if good else 0
                    self.controller.info(f"Hunt CONTACT: fresh={fresh}; stopped={stopped}; "
                        f"resistance_rise={observation['resistance'] - self.controller.first_res:.3f}; "
                        f"aligned={aligned}; at_cell_plane={touching}; "
                        f"estimated_plane_gap_um={distance_to_cell:.3f}; plane_tolerance_um={plane_tolerance:g}; "
                        f"confirmed={state['confirmed_readings']}/{self.CONTACT_READINGS}; "
                        f"pipette_z_focus={vision.get('pipette_focus')} (advisory)")
            calculated.update(cell_detection=cell, vision_fresh=fresh, aligned=aligned,
                              pipette_visible=bool(tip_ok and fresh),
                              contact_depth_ready=contact_depth_ready, distance_to_cell_um=distance_to_cell,
                              touching=touching, resistance_up=rise,
                              adaptive_goal=(state["confirming"] and fresh and stopped
                                             and rise and touching
                                             and state["confirmed_readings"] >= self.CONTACT_READINGS))
            if state["step"] == 3:
                calculated["spear_target"] = observation["manipulator_position"][2] + state["cell_focus_z"] - state["pipette_z"]
            calculated["spear_speed"] = min(abs(config.max_descent_speed), 100.0)
        self.controller.observer.annotate(self, observation, calculations={
            "window_ready": ready, "resistance_delta": delta, "travel_um": travel})
        return calculated

    def decide(self, observation=None, state=None):
        self.failure_gate()
        self.success_gate()
        calculated = observation
        mode = calculated["mode"]
        command = {"calculated": calculated, "mode": mode}
        if self.awaiting_operator:
            return None
        if calculated["stale"] or mode != self.controller.config.mode:
            return {**command, "kind": "stop"}
        if state.get("mode") == "Adaptive" and mode != "Adaptive":
            return {**command, "kind": "stop"}
        if mode not in ("Manual", "Classic", "Adaptive", "Training", "Agent"):
            self.failure_gate(message=f"Unsupported hunt mode: {mode}")
        complete = self.success_gate(calculated, state)
        if complete or self.goal_event:
            return {**command, "kind": "complete"}
        self.failure_gate(state, calculated)
        if mode == "Adaptive":
            if not calculated["vision_fresh"]:
                return None
            if (state['step'] == 3 and np.isfinite(calculated.get('spear_target', np.nan))
                    and calculated['spear_target'] <= calculated['observation']['manipulator_position'][2]):
                return {**command, 'kind': 'adaptive', 'method': 'check_contact'}
            if state["confirming"]:
                if time.monotonic() - state["confirmation_started"] >= self.CONTACT_TIMEOUT_S:
                    self.controller.info('Hunt CONTACT: confirmation window expired; '
                        f"accepted={state['confirmed_readings']}/{self.CONTACT_READINGS}; "
                        'repeating SEARCH, cell SHIFT, SCAN, pipette SHIFT, and SPEAR')
                    return {**command, 'kind': 'adaptive', 'method': 'search',
                            'retry_contact': True, 'pipette_plane': calculated['pipette_plane']}
                return None
            if state["step"] == 4 and calculated["resistance_up"]:
                if calculated["contact_depth_ready"]:
                    return {**command, "kind": "adaptive", "method": "confirm"}
                if not state.get("logged_early_resistance"):
                    self.controller.info("Hunt resistance rise before contact window: "
                        f"remaining_to_cell_um={calculated['distance_to_cell_um']:.2f}; "
                        f"contact_window_um={getattr(self.controller.config, 'hunt_contact_plane_tolerance_um', 2.0)}; "
                        "continuing Spear to the cell plane")
                    state["logged_early_resistance"] = True
            else:
                state["logged_early_resistance"] = False
            if (state['step'] == 4 and calculated.get('pipette_visible')
                    and calculated.get('cell_detection') is not None and not calculated['aligned']
                    and calculated['contact_depth_ready']):
                self.controller.info('Hunt SPEAR: pipette has reached the cell-plane window and '
                    'cell is displaced; stopping for renewed alignment; '
                    f"remaining_to_cell_um={calculated['distance_to_cell_um']:.2f}")
                return {**command, 'kind': 'adaptive', 'method': 'search',
                        'pipette_plane': calculated['pipette_plane']}
            try:
                move = ApproachCellPhase.decide(self, calculated, state)
            except AutopatchError as error:
                if (str(error).startswith('cell focus confirmation failed:')
                        and state['step'] == 1 and state.get('focus_confirmation') is not None
                        and self._retry_cell_search(state, str(error))):
                    return None
                raise
            if move is None:
                return None
            if move["method"] == "complete":
                move = {**move, "method": "confirm"}
            return {**move, **command, "kind": "adaptive"}
        if mode not in ("Classic", "Adaptive") and state["last_velocity"] is not None:
            return {**command, "kind": "stop"}
        if not self.action_gate(calculated, state):
            return None
        if mode == "Classic":
            return {**command, "kind": "descend",
                    "value": [0, 0, self.controller.config.max_descent_speed]}
        if mode == "Agent" and not state["model_prepared"]:
            return {**command, "kind": "prepare_model", "model": "hunt"}
        return None

    def act(self, observation=None, state=None):
        self.failure_gate()
        self.success_gate()
        command = observation
        if command is None:
            return
        if not self.action_gate(command["calculated"], state, command=command):
            if command["mode"] == "Adaptive" or state["mode"] == "Adaptive":
                self.finish(state)
            else:
                self.controller.calibrated_unit.stop()
            state["last_velocity"] = None
            return
        kind = command["kind"]
        if kind == "adaptive":
            controller = self.controller
            method = command["method"]
            if method == "focus":
                sweep = state.get("focus_sweep")
                if sweep is None:
                    kind = "SEARCH" if state["step"] == 1 else "SCAN"
                    center = state["cell_focus_z"] if state["step"] == 1 else state["pipette_z"]
                    controller.info(f"Hunt {kind}: focus sweep around z={center}")
                    sweep = state["focus_sweep"] = FocusSweep(controller, center, kind)
                    if state["step"] == 1 and "cell_search_bounds" in state:
                        sweep.low, sweep.high = state["cell_search_bounds"]
                    sweep.start()
                else:
                    try:
                        finished = sweep.update(state["observation"], self._focus_score(state))
                    except EmptyFocusEvidence as error:
                        if not self._retry_cell_search(state, str(error)):
                            raise
                        finished = False
                    if finished:
                        state["focus_done"] = state["step"]
                        confirmation_options = ({'selected_tip_xy': sweep.best_tip_xy} if state['step'] == 2
                                                else {'selected_cell_xy': sweep.best_cell_xy})
                        state["focus_confirmation"] = FocusConfirmation(
                            controller, "cell" if state["step"] == 1 else "pipette", **confirmation_options)
                state["mode"] = command["mode"]
                return
            if method != "scan":
                controller.microscope.stop()
            if method == "confirm":
                calculated = command["calculated"]
                controller.info(f"Hunt CONFIRM: step={state['step']}; resistance_up={calculated.get('resistance_up')}; "
                                f"touching={calculated.get('touching')}; aligned={calculated.get('aligned')}")
                self.finish(state)
                state.update(confirming=True, confirmed_readings=0,
                             confirmation_started=time.monotonic(),
                             last_confirmation_id=calculated["observation"]["observation_id"],
                             stop_position=np.array(controller.calibrated_unit.position(), copy=True))
            elif method == "shift":
                controller.info(f"Hunt SHIFT {command['target']}: starting")
                if command["target"] == "cell":
                    state["cell_focus_z"] = state["observation"]["stage_positions"][2]
                    self._update_travel_allowance(state)
                    if self._shift_cell_from_evidence(state) is False:
                        return
                else:
                    state["pipette_z"] = state["observation"]["stage_positions"][2]
                    options = ({"detected_px": state.pop("confirmed_tip_xy")} if "confirmed_tip_xy" in state else {})
                    if 'detected_px' in options:
                        options.update(expected_px=controller.expected_pipette_xy(),
                                       max_deviation_um=50.0)
                        self._show_detection_marker('pipette', options['detected_px'], (255, 0, 0))
                    centered = controller.calibrated_unit.center_pipette(speed=controller.config.approach_pipette_speed, **options)
                    self.failure_gate()
                    self.success_gate()
                    if centered is False:
                        reason = getattr(controller.calibrated_unit, "last_pipette_move_failure", None)
                        if not isinstance(reason, str) or not reason:
                            reason = "pipette centering did not complete"
                        raise AutopatchError(f"Hunt pipette SHIFT failed: {reason}; SPEAR cancelled")
                    self._rebase_travel_allowance(state)
                controller.info(f"Hunt SHIFT {command['target']}: finished")
                for key in ("focus_sweep", "focus_done", "focus_confirmation", "cell_template_evidence", "cell_detector_reference"):
                    state.pop(key, None)
                state["step"] += 1
                self.failure_gate()
                self.success_gate()
                if controller.config.mode != "Adaptive":
                    self.finish(state)
                    return
                target = state["pipette_z"] if command["target"] == "cell" else state["cell_focus_z"]
                if command["target"] == "pipette":
                    state["hover_z"] = target
                controller.info(f"Hunt RELOCATE {'pipette' if command['target'] == 'cell' else 'cell contact plane'}: target_z={target}")
                controller.microscope.absolute_move(target)
                controller.microscope.wait_until_still()
                self.failure_gate()
                self.success_gate()
            elif method in ("search", "scan"):
                if state["step"] == 0:
                    state["step"] = 1
                    controller.info(f"Hunt RELOCATE cell: target_z={state['cell_focus_z']}")
                    controller.microscope.absolute_move(state["cell_focus_z"])
                    controller.microscope.wait_until_still()
                    self.failure_gate()
                    self.success_gate()
                elif state["step"] == 4:
                    state.update(confirming=False, confirmed_readings=0, last_confirmation_id=None)
                    state.pop("confirmation_started", None)
                    state.pop("stop_position", None)
                    motion = state.pop("spear_motion", None)
                    if motion is not None:
                        motion.stop()
                    controller.calibrated_unit.stop()
                    actual_z = float(controller.calibrated_unit.position()[2])
                    travel = actual_z - float(state.get("spear_start_z", actual_z))
                    origin = float(state["cell_focus_z"])
                    state.update(cell_reacquisition_origin_z=origin, cell_reacquisition_attempt=0,
                                 cell_search_bounds=(origin - 5.0, origin + 10.0))
                    travel_limit = float(state.get("hunt_travel_limit_um", controller.config.max_distance))
                    remaining = travel_limit - abs(actual_z - state["start_position"][2])
                    controller.info(f"Hunt SEARCH reacquisition: actual_spear_travel_um={travel:.2f}; "
                        f"last_cell_z={origin}; first_bounds={state['cell_search_bounds']}; "
                        f"maximum_deeper_search_um=25; remaining_pipette_travel_um={remaining:.2f}; "
                        f"confirmed_travel_limit_um={travel_limit:g}")
                    stopped_plane = (state['pipette_z'] + travel if 'spear_start_z' in state
                                     else command['pipette_plane'])
                    state.update(step=1, speared=False, pipette_z=stopped_plane)
                    controller.info(f"Hunt SPEAR complete: repeat alignment; cell_z={state['cell_focus_z']}")
                    controller.microscope.absolute_move(state["cell_focus_z"])
                    controller.microscope.wait_until_still()
                    self.failure_gate()
                    self.success_gate()
                if state.get("bounded_focus"):
                    for key in ("focus_sweep", "focus_done", "focus_confirmation"):
                        state.pop(key, None)
                else:
                    controller.microscope.absolute_move_velocity(command["speed"])
            elif method == 'check_contact':
                controller.calibrated_unit.stop()
                actual_z = float(controller.calibrated_unit.position()[2])
                motion = state.pop('spear_motion', None)
                if motion is not None:
                    motion.stop()
                state.update(step=4, spear_start_z=actual_z, spear_target=actual_z)
                controller.info('Hunt SPEAR: no downward distance remains; '
                    f'pipette_plane={state["pipette_z"]}; cell_plane={state["cell_focus_z"]}; '
                    'checking fresh stopped contact evidence, otherwise repeating alignment')
            elif method == "spear":
                controller.info(f"Hunt SPEAR: target_z={command['target']}; speed={command['speed']}")
                state.update(step=4, spear_target=command["target"],
                             spear_start_z=state["observation"]["manipulator_position"][2])
                motion = state["spear_motion"] = SpearMotion(controller, command["target"], command["speed"])
                motion.start()
        elif kind == "descend":
            velocity = tuple(command["value"])
            if velocity != state["last_velocity"]:
                self.controller.calibrated_unit.absolute_move_group_velocity(command["value"])
                state["last_velocity"] = velocity
        elif kind == "prepare_model":
            if not state["model_prepared"]:
                self.controller.agenthelper.prepare_model(command["model"])
                state["model_prepared"] = True
        elif kind == "stop":
            if command["mode"] == "Adaptive" or state["mode"] == "Adaptive":
                self.finish(state)
            else:
                self.controller.calibrated_unit.stop()
            state["last_velocity"] = None
        elif kind == "complete":
            self.finish(state, success=True, mode=command["mode"])
        state["mode"] = command["mode"]

    def action_gate(self, observation=None, state=None, *, command=None):
        self.failure_gate()
        self.success_gate()
        if observation is None or state is None:
            return False
        mode = observation["mode"]
        if (observation["stale"] or mode != self.controller.config.mode
                or (command is not None and command["mode"] != mode)):
            return False
        kind = None if command is None else command["kind"]
        if kind == "stop":
            return True
        contact = self.success_gate(observation, state) or self.goal_event
        if contact:
            return kind == "complete"
        if kind == "complete":
            return False
        self.failure_gate(state, observation)
        if mode == "Adaptive":
            if not observation["vision_fresh"]:
                return False
            if command is None:
                return not state["confirming"]
            if kind != "adaptive":
                return False
            if command["method"] == "confirm":
                return (state["step"] == 4 and observation["resistance_up"]
                        and observation["contact_depth_ready"])
            if command['method'] == 'check_contact':
                return (state['step'] == 3 and not state['confirming']
                        and np.isfinite(observation.get('spear_target', np.nan))
                        and observation['spear_target'] <= observation['observation']['manipulator_position'][2])
            if state["confirming"]:
                return (command["method"] == "search" and command.get("retry_contact") is True
                        and state["step"] == 4
                        and time.monotonic() - state["confirmation_started"] >= self.CONTACT_TIMEOUT_S)
            if command["method"] == "spear":
                position = observation["observation"]["manipulator_position"][2]
                alignment_ready = (state.get("bounded_focus") and state["step"] == 3) or observation["aligned"]
                return (alignment_ready and np.isfinite(command["target"])
                        and command["speed"] > 0 and command["target"] >= position
                        and abs(command["target"] - state["start_position"][2])
                        < float(state.get("hunt_travel_limit_um", self.controller.config.max_distance)))
            return command["method"] in ("search", "shift", "scan", "focus")
        if mode == "Classic":
            return (kind in (None, "descend") and (command is None or
                    command.get("value") == [0, 0, self.controller.config.max_descent_speed]))
        if mode == "Agent":
            return (kind in (None, "prepare_model") and
                    (command is None or command.get("model") == "hunt"))
        return False

    def failure_gate(self, state=None, observation=None, *, message=None):
        self.controller.abort_if_requested()
        if message is not None:
            raise AutopatchError(message)
        if self.awaiting_operator or state is None or observation is None or not state["initialized"]:
            return
        mode = observation["mode"]
        if mode == "Training" or mode != self.controller.config.mode:
            return
        if mode == "Adaptive":
            timeout = float(self.controller.config.hunt_timeout)
            if time.monotonic() - state["started_at"] >= timeout:
                raise AutopatchError(f"Adaptive Hunt timed out after {timeout:g} s (hunt_timeout)")
            raw = observation.get("observation", observation)
            if "stage_positions" in raw and "cell_focus_z" in state:
                low, high = sorted((state["cell_focus_z"], state["pipette_z"]))
                sweep = state.get("focus_sweep")
                tolerance = sweep.BOUNDARY_TOLERANCE_UM if sweep is not None else 0.0
                lower, upper = low - 5 - tolerance, high + 5 + tolerance
                if state.get("step") == 1 and "cell_reacquisition_origin_z" in state:
                    origin = state["cell_reacquisition_origin_z"]
                    lower = min(lower, origin - 5 - tolerance)
                    upper = max(upper, origin + 25 + tolerance)
                if not lower <= raw["stage_positions"][2] <= upper:
                    raise AutopatchError("Adaptive Hunt exceeded microscope search bounds")
        travel = observation.get("travel_um")
        if travel is None:
            travel = abs(observation["manipulator_position"][2] - state["start_position"][2])
        limit = float(state.get("hunt_travel_limit_um", self.controller.config.max_distance)
                      if mode == "Adaptive" else self.controller.config.max_distance)
        if travel >= limit:
            raise AutopatchError(
                "Cell not detected before reaching max hunt distance "
                f"({travel:.1f} um >= {limit:g} um).")

    def _update_travel_allowance(self, state):
        """Extend Hunt travel only to a freshly confirmed cell target."""
        confirmation = state.get("focus_confirmation")
        vision = state.get("observation", {}).get("deep_learning", {})
        frame = vision.get("source_frame")
        if (confirmation is None or confirmation.confirmed is not True
                or not confirmation.is_cell
                or confirmation.valid_samples < confirmation.MIN_CELL_SAMPLES
                or len(confirmation.seen_frames) < confirmation.MIN_CELL_SAMPLES
                or vision.get("stale", True) or frame is None
                or frame != confirmation.last_vision.get("source_frame")
                or frame not in confirmation.seen_frames
                or frame in state.get("travel_allowance_frames", set())):
            return False
        position = float(self.controller.calibrated_unit.position()[2])
        target = position + float(state["cell_focus_z"]) - float(state["pipette_z"])
        start = float(state["start_position"][2])
        if not np.isfinite([position, target, start]).all():
            return False
        previous = float(state.get("hunt_travel_limit_um", self.controller.config.max_distance))
        required = abs(target - start) + SpearMotion.ENDPOINT_TOLERANCE_UM
        state.setdefault("travel_allowance_frames", set()).update(confirmation.seen_frames)
        state["confirmed_travel_cell"] = dict(z=float(state["cell_focus_z"]),
            valid_samples=confirmation.valid_samples)
        state["hunt_travel_limit_um"] = max(previous, required)
        self.controller.info(f"Hunt travel allowance: confirmed_cell_z={state['cell_focus_z']}; "
            f"accepted_samples={confirmation.valid_samples}; target_pipette_z={target}; "
            f"previous_limit_um={previous:g}; limit_um={state['hunt_travel_limit_um']:g}; "
            "fresh confirmed cell evidence")
        return True

    def _rebase_travel_allowance(self, state):
        """Retain cell evidence while refreshing the confirmed pipette plane."""
        cell = state.get("confirmed_travel_cell")
        confirmation = state.get("focus_confirmation")
        vision = state.get("observation", {}).get("deep_learning", {})
        frame = vision.get("source_frame")
        if (cell is None or cell["z"] != state["cell_focus_z"]
                or confirmation is None or confirmation.confirmed is not True
                or confirmation.is_cell
                or confirmation.valid_samples < confirmation.MIN_PIPETTE_SAMPLES
                or len(confirmation.seen_frames) < confirmation.MIN_PIPETTE_SAMPLES
                or vision.get("stale", True) or frame is None
                or frame != confirmation.last_vision.get("source_frame")
                or frame not in confirmation.seen_frames
                or frame in state.get("travel_rebase_frames", set())):
            return False
        position = float(self.controller.calibrated_unit.position()[2])
        target = position + cell["z"] - float(state["pipette_z"])
        start = float(state["start_position"][2])
        if not np.isfinite([position, target, start]).all():
            return False
        previous = float(state.get("hunt_travel_limit_um", self.controller.config.max_distance))
        required = abs(target - start) + SpearMotion.ENDPOINT_TOLERANCE_UM
        state.setdefault("travel_rebase_frames", set()).update(confirmation.seen_frames)
        state["hunt_travel_limit_um"] = max(previous, required)
        self.controller.info(f"Hunt travel allowance: confirmed_cell_z={cell['z']}; "
            f"cell_samples={cell['valid_samples']}; pipette_samples={confirmation.valid_samples}; "
            f"target_pipette_z={target}; previous_limit_um={previous:g}; "
            f"limit_um={state['hunt_travel_limit_um']:g}; "
            "rebased from confirmed pipette and cell")
        return True

    def _retry_cell_search(self, state, reason):
        """Retry missing cell evidence once in the approved deeper search interval."""
        if (state.get("step") != 1 or "cell_reacquisition_origin_z" not in state
                or state.get("cell_reacquisition_attempt", 0) >= 1):
            return False
        sweep = state.pop("focus_sweep", None)
        if sweep is not None:
            sweep.stop()
        origin = state["cell_reacquisition_origin_z"]
        state.update(cell_reacquisition_attempt=1,
                     cell_search_bounds=(origin + 5.0, origin + 25.0))
        state.pop("focus_done", None)
        state.pop("focus_confirmation", None)
        self.controller.info(f"Hunt SEARCH retry: reason={reason}; "
            f"bounds={state['cell_search_bounds']}; maximum_deeper_search_um=25; "
            "last confirmed cell plane retained")
        return True

    def success_gate(self, observation=None, state=None):
        super().success_gate()
        if self.awaiting_operator or observation is None or state is None:
            return False
        if observation["stale"] or observation["mode"] != self.controller.config.mode:
            return False
        if observation["mode"] == "Adaptive":
            return super().success_gate(observation["adaptive_goal"], state)
        threshold = self.controller.config.cell_R_increase
        if ("cell_detected" not in observation
                or observation["cell_detection_threshold"] != threshold):
            observation["cell_detected"] = self._isCellDetected(state["readings"], threshold)
            observation["cell_detection_threshold"] = threshold
            self.controller.observer.annotate(self, observation["observation"], calculations={
                "cell_detected": observation["cell_detected"], "cell_threshold": threshold})
        return super().success_gate(observation["cell_detected"], state)

    def finish(self, state=None, *, success=False, mode=None):
        controller = self.controller
        primary_error, stop_error = sys.exc_info()[1], None
        devices = [controller.calibrated_stage, controller.calibrated_unit, controller.microscope]
        if state is not None and state.get("focus_sweep") is not None:
            devices.insert(0, state["focus_sweep"])
        if state is not None and state.get("spear_motion") is not None:
            devices.insert(0, state["spear_motion"])
        for device in devices:
            try:
                device.stop()
            except BaseException as error:
                if stop_error is None:
                    stop_error = error
        if primary_error is not None:
            return
        if stop_error is not None:
            raise stop_error
        if success:
            controller.info("Cell Detected")
            self.goal_event = False
            if not self.awaiting_operator:
                self.complete_success()

    def _isCellDetected(self, lastResDeque, cellThreshold=0.15):
        if len(lastResDeque) < 5:
            return False
        delta = lastResDeque[4] - self.controller.first_res
        detected = cellThreshold <= delta
        if detected:
            self.controller.info(f"Cell detected: {detected}; resistance: {delta}")
            self.controller.calibrated_unit.stop()
        return detected
