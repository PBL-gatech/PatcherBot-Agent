"""Historical Hunt behavior within the shared phase lifecycle."""

import collections
import sys
import time
import numpy as np

from ..errors import AutopatchError
from .approach_cell import ApproachCellPhase


class HuntCellPhase(ApproachCellPhase):
    phase_name = "Hunt"
    CONTACT_READINGS = 5
    CONTACT_TIMEOUT_S = 5.0
    DEPTH_TOLERANCE_UM = 2.0

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
        controller.alignment_display = dict(owner=self, cell_reference=cell[0])
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
            controller.info("Hunt WARMUP: 3 inference samples before motion")
            for sample in range(3):
                controller.observer.infer_frame(["pipette_image_xy", "pipette_defocus_um", "cell_detections"])
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
            self.controller.calibrated_stage.focusHelper.relocate(state["cell_focus_z"])
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
            self.frame_context["pipette_expected_xy"] = lambda: controller.calibrated_unit.reference_position(anchor=getattr(controller.calibrated_unit, '_trusted_tip_reference', None) or (np.array([controller.calibrated_unit.camera.width / 2., controller.calibrated_unit.camera.height / 2.]), controller.home_position, np.asarray(controller.home_stage_position)[:2]))[:2]
            self.frame_context["pipette_tracking_xy"] = lambda: (controller.calibrated_unit.reference_position(
                anchor=controller.calibrated_unit._tip_reference)[:2]
                if getattr(controller.calibrated_unit, '_tip_reference', None) is not None else None)
            self.frame_context["pipette_xyz"] = controller.calibrated_unit.position
            self.frame_context["stage_xy"] = controller.calibrated_stage.position
            inference_fields = ["pipette_image_xy", "pipette_defocus_um"]
            if state["step"] in (1, 4) or state["confirming"]:
                inference_fields.append("cell_detections")
            if state["step"] == 1:
                fields.append("camera_image")
            self.frame_context["focus_z"] = lambda: (controller.microscope.position()
                / controller.calibrated_unit.config.microscope_units_per_um)
            evidence = controller.observer.infer_frame(inference_fields, frame_context=self.frame_context)
        observation = super().observe(fields=fields,
            evidence=evidence, raw_resistance=True)
        if mode == "Adaptive" and state["step"] == 4:
            vision = observation.get('deep_learning', {})
            source = vision.get('source_positions', {})
            expected = source.get('pipette_expected_xy')
            if expected is not None:
                expected = np.asarray(expected, dtype=float)
                if expected.shape == (2,) and np.isfinite(expected).all():
                    tip = vision.get('pipette_position')
                    accepted = False
                    radius = self.controller.calibrated_unit.config.pipette_detection_radius_um
                    if tip is not None and not vision.get('stale', True):
                        delta = self.controller.calibrated_stage.pixels_to_um_relative(np.asarray(tip)[:2] - expected)
                        accepted = np.linalg.norm(delta[:2]) <= radius
                    if not accepted:
                        if tip is not None:
                            self.controller.info(f'Hunt SPEAR pipette: rejected detection={tip}; expected_xy={expected}; '
                                                 f'plausibility_radius_um={radius}; following calibrated trajectory')
                        vision['pipette_position'] = None
                    tracking = source.get('pipette_tracking_xy')
                    vision['pipette_tracking_position'] = expected if tracking is None else np.asarray(tracking)[:2]
                    vision['pipette_tracking_source'] = 'calibrated trajectory from last accepted tip'
        baseline = (controller.first_res if state["initialized"]
                    else controller.resistanceRamp())
        controller.observer.annotate(self, observation, fields={
            "mode": mode, "baseline_resistance": baseline})
        try:
            vision = observation.get("deep_learning", {})
            if not vision.get("stale", True) or "pipette_tracking_position" in vision:
                tip = vision.get("pipette_tracking_position", vision.get("pipette_position"))
                if state.get("step") == 2:
                    sweep = state.get("focus_sweep") or {}
                    confirmation = sweep.get("confirmation")
                    tip = (confirmation["confirmed_tip_xy"] if confirmation and confirmation["confirmed"]
                           else sweep.get("best_xy"))
                    if tip is None:
                        tip = self.frame_context["pipette_expected_xy"]()
                if tip is not None and np.shape(tip) == (2,) and np.isfinite(tip).all():
                    self.controller.observer.annotate(self, observation, markers={"pipette": tip},
                        source_positions=vision.get("source_positions", {}))
        except Exception as error:
            controller.warning(f"Could not display pipette detection marker: {error}. Hunt continues.", extra={"detection_overlay": True})
        self.failure_gate()
        self.success_gate()
        return observation

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
                from ...devices.manipulator.helpers.CellTrackHelper import CellTrackHelper
                CellTrackHelper.template_evidence(stage, observation, state, expected,
                    use_centroid=config.use_centroid, info=self.controller.info)
            template = state.get('cell_template_evidence')
            template_xy = (None if template is None else
                (stage.M @ source_stage + stage.r0)[:2] - template)
            previous = state.get('cell_detector_reference')
            previous_xy = (None if previous is None else
                (stage.M @ source_stage + stage.r0)[:2] - previous)
            from ...devices.manipulator.helpers.CellTrackHelper import CellTrackHelper
            cell, fused = CellTrackHelper.fuse_candidates(stage, cells, expected, template_xy, previous_xy)
            displaced_search = state['step'] == 4 or 'cell_reacquisition_origin_z' in state
            if cell is not None and config.hunt_limit_target_radius and not displaced_search:
                offset = unit.pixels_to_um_relative(np.append(cell[:2] - expected, 0))
                if np.linalg.norm(offset) > config.hunt_target_radius:
                    cell = None
            observation["cell_detection"] = cell
            observation['cell_selection_locked'] = True
            observation['cell_fused_xy'] = fused if cell is not None else None
            calculated.update(super().calculate(observation, state))
            if fresh and cell is not None:
                state['cell_detector_reference'] = (stage.M @ source_stage + stage.r0)[:2] - np.asarray(cell[:2])
                display_xy = np.asarray(cell[:2]) + (stage.M @ (observation['stage_positions'][:2] - source_stage))[:2]
                self.controller.observer.annotate(self, observation, markers={"cell": display_xy})
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
            calculated["spear_speed"] = abs(config.max_descent_speed)
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
                move = super().decide(calculated, state)
            except AutopatchError as error:
                if (str(error).startswith('cell focus confirmation failed:')
                        and state['step'] == 1 and state.get('focus_sweep', {}).get('confirmation') is not None
                        and self.failure_gate(state, message=str(error), recover=True)):
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
                try:
                    state["focus_sweep"] = controller.calibrated_stage.focusHelper.update_sweep(
                        state.get("focus_sweep"), **command["focus"])
                except (RuntimeError, ValueError) as error:
                    raise AutopatchError(str(error)) from error
                if state["focus_sweep"].get("empty"):
                    reason = "SEARCH focus sweep found no valid samples"
                    if not self.failure_gate(state, message=reason, recover=True):
                        raise AutopatchError(reason)
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
                rebase = command['target'] != 'cell'
                cell = state.get("confirmed_travel_cell") if rebase else None
                if not rebase or (cell is not None and cell["z"] == state["cell_focus_z"]):
                    history = "travel_rebase_frames" if rebase else "travel_allowance_frames"
                    confirmation = controller.calibrated_stage.focusHelper.confirmed_evidence(
                        state.get("focus_sweep", {}), state.get("observation", {}).get("deep_learning", {}),
                        pipette=rebase, used_frames=state.get(history, set()))
                    if confirmation is not None:
                        cell_z = cell["z"] if rebase else float(state["cell_focus_z"])
                        position = float(controller.calibrated_unit.position()[2])
                        target = position + cell_z - float(state["pipette_z"])
                        start = float(state["start_position"][2])
                        if np.isfinite([position, target, start]).all():
                            previous = float(state.get("hunt_travel_limit_um", controller.config.max_distance))
                            required = abs(target - start) + controller.calibrated_unit.actionHelper.MOTION_ENDPOINT_TOLERANCE_UM
                            state.setdefault(history, set()).update(confirmation["seen_frames"])
                            if not rebase:
                                state["confirmed_travel_cell"] = {"z": cell_z}
                            state["hunt_travel_limit_um"] = max(previous, required)
                            source = "rebased from confirmed pipette and cell" if rebase else "fresh confirmed cell evidence"
                            controller.info(f"Hunt travel allowance: confirmed_cell_z={cell_z}; "
                                f"accepted_samples={confirmation['valid_samples']}; target_pipette_z={target}; "
                                f"previous_limit_um={previous:g}; limit_um={state['hunt_travel_limit_um']:g}; {source}")
                if command["target"] == "cell" and self._shift_cell_from_evidence(state) is False:
                    return
                controller.info(f"Hunt SHIFT {command['target']}: finished")
                self.controller.calibrated_stage.focusHelper.reset_state(state, "cell_template_evidence", "cell_detector_reference")
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
                self.controller.calibrated_stage.focusHelper.relocate(target)
                self.failure_gate()
                self.success_gate()
            elif method in ("search", "scan"):
                if state["step"] == 0:
                    state["step"] = 1
                    controller.info(f"Hunt RELOCATE cell: target_z={state['cell_focus_z']}")
                    self.controller.calibrated_stage.focusHelper.relocate(state["cell_focus_z"])
                    self.failure_gate()
                    self.success_gate()
                elif state["step"] == 4:
                    state.update(confirming=False, confirmed_readings=0, last_confirmation_id=None)
                    state.pop("confirmation_started", None)
                    state.pop("stop_position", None)
                    motion = state.pop("spear_motion", None)
                    if motion is not None:
                        self.controller.calibrated_unit.actionHelper.cancel(motion)
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
                    self.controller.calibrated_stage.focusHelper.relocate(state["cell_focus_z"])
                    self.failure_gate()
                    self.success_gate()
                if state.get("bounded_focus"):
                    self.controller.calibrated_stage.focusHelper.reset_state(state)
                else:
                    controller.microscope.absolute_move_velocity(command["speed"])
            elif method == 'check_contact':
                controller.calibrated_unit.stop()
                actual_z = float(controller.calibrated_unit.position()[2])
                motion = state.pop('spear_motion', None)
                if motion is not None:
                    self.controller.calibrated_unit.actionHelper.cancel(motion)
                state.update(step=4, spear_start_z=actual_z, spear_target=actual_z)
                controller.info('Hunt SPEAR: no downward distance remains; '
                    f'pipette_plane={state["pipette_z"]}; cell_plane={state["cell_focus_z"]}; '
                    'checking fresh stopped contact evidence, otherwise repeating alignment')
            elif method == "spear":
                controller.info(f"Hunt SPEAR: target_z={command['target']}; speed={command['speed']}")
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

    def failure_gate(self, state=None, observation=None, *, message=None, recover=False):
        if recover:
            if state is None or state.get('step') != 1 or 'cell_reacquisition_origin_z' not in state:
                return False
            return self.controller.calibrated_stage.focusHelper.retry_search(state,
                state['cell_reacquisition_origin_z'], bounds=(5.0, 25.0), reason=message,
                info=self.controller.info)
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
                tolerance = self.controller.calibrated_stage.focusHelper.BOUNDARY_TOLERANCE_UM if sweep is not None else 0.0
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

    def success_gate(self, observation=None, state=None):
        super().success_gate()
        if self.awaiting_operator or observation is None or state is None:
            return False
        if observation["stale"] or observation["mode"] != self.controller.config.mode:
            return False
        if observation["mode"] == "Adaptive":
            return super().success_gate(bool(observation["adaptive_goal"]), state)
        threshold = self.controller.config.cell_R_increase
        if ("cell_detected" not in observation
                or observation["cell_detection_threshold"] != threshold):
            observation["cell_detected"] = self._isCellDetected(state["readings"], threshold)
            observation["cell_detection_threshold"] = threshold
            self.controller.observer.annotate(self, observation["observation"], calculations={
                "cell_detected": observation["cell_detected"], "cell_threshold": threshold})
        return super().success_gate(bool(observation["cell_detected"]), state)

    def finish(self, state=None, *, success=False, mode=None):
        controller = self.controller
        primary_error, stop_error = sys.exc_info()[1], None
        if primary_error is not None:
            controller.info(f"{self.phase_name} FAILED: {primary_error}; stopping motion")
        stops = [controller.calibrated_stage.stop, controller.calibrated_unit.stop, controller.microscope.stop]
        if state is not None and state.get("focus_sweep") is not None:
            stops.insert(0, lambda: controller.calibrated_stage.focusHelper.cancel_sweep(state["focus_sweep"]))
        if state is not None and state.get("spear_motion") is not None:
            stops.insert(0, lambda: controller.calibrated_unit.actionHelper.cancel(state["spear_motion"]))
        for stop in stops:
            try:
                stop()
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
