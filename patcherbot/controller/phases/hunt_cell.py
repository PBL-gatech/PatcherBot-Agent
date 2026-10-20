"""Hunt for membrane contact using shared alignment and resistance readouts."""

import collections
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
        self.configure_observations(tracking=True)
        controller.alignment_display = dict(owner=self, cell_reference=cell[0])
        controller.info("Hunting for cell")
        controller.pressure.set_pressure(controller.config.pressure_near)
        controller.sleep(3)
        state = dict(initialized=False, start_position=None,
                    readings=collections.deque(maxlen=5), last_velocity=None,
                    model_prepared=False, mode=controller.config.mode, cell=cell, step=0,
                    goal_distance=0, confirming=False, confirmed_readings=0,
                    started_at=time.monotonic(), last_frame=None,
                    last_confirmation_id=None,
                    fields=["resistance", "manipulator_position"])
        if controller.config.mode == "Adaptive":
            self._start_adaptive(state)
        return state

    def _start_adaptive(self, state):
        """Stop the previous route, then start the same bounded alignment sequence."""
        controller, config = self.controller, self.controller.config
        unit, stage = controller.calibrated_unit, controller.calibrated_stage
        self.stop_motions(state, all_devices=True)
        state["started_at"] = time.monotonic()
        cell = state["cell"]
        cell_z = state.get("cell_focus_z", cell[0][2])
        approached = getattr(controller, "_approach_cell_focus", None)
        if ("cell_focus_z" not in state and isinstance(approached, dict)
                and approached.get("cell_reference") is cell[0] and np.isfinite(approached.get("z", np.nan))):
            cell_z = approached["z"]
            if approached.get("tracked_cell") is not None:
                state["cell"] = approached["tracked_cell"]
                state["tracked_cell_reference"] = np.asarray(state["cell"][0][:2]).copy()
                state["cell_template_point"] = approached.get("template_point")
        pipette_z = None
        if state["initialized"]:
            position = unit.position()
            self.failure_gate(state, dict(mode="Adaptive", manipulator_position=position))
            plane, anchor_z = state.get("pipette_plane_reference", (
                cell_z - (config.cell_distance if config.cell_type == "Plate" or config.cell_type_toggle
                          else config.slice_start_distance), state["start_position"][2]))
            pipette_z = plane + position[2] - anchor_z
        stage.focusHelper.reset_state(state, "spear_motion", "confirmation_started", "stop_position",
            "confirmed_tip_xy", "cell_template_evidence", "cell_detector_reference",
            "cell_reacquisition_origin_z", "cell_reacquisition_attempt", "cell_search_bounds", "confirmed_travel_cell")
        self._vision = True
        self.goal_event = False
        state.update(step=1, bounded_focus=True, display_detection_markers=False, cell_focus_z=cell_z,
            pipette_z=controller.microscope.position() / unit.config.microscope_units_per_um if pipette_z is None else pipette_z,
            confirming=False, confirmed_readings=0, last_confirmation_id=None, last_frame=None,
            last_velocity=None, speared=False,
            fields=["resistance", "manipulator_position", "stage_positions", "deep_learning", "camera_image"])
        controller.info("Hunt WARMUP: 3 inference samples before motion")
        for _ in range(3):
            super().observe(fields=["deep_learning"])
            self.failure_gate()
            self.success_gate()
        if controller.config.mode != "Adaptive":
            return
        stage.focusHelper.relocate(cell_z)
        self.failure_gate()
        self.success_gate()
        state["started_at"] = time.monotonic()

    def observe(self, *, state):
        controller = self.controller
        mode = controller.config.mode
        if state["mode"] != mode:
            return dict(mode=mode, mode_changed=True)
        if state["initialized"]:
            position = super().observe(fields=["manipulator_position"], evidence={})
            self.failure_gate(state, {**position, "mode": mode})
            controller.sleep(0.04)
        observation = super().observe(state=state, raw_resistance=True)
        baseline = (controller.first_res if state["initialized"]
                    else controller.resistanceRamp())
        controller.observer.annotate(self, observation, fields={
            "mode": mode, "baseline_resistance": baseline})
        return observation

    def calculate(self, observation=None, state=None):
        mode = observation["mode"]
        if observation.get("mode_changed"):
            return dict(observation=observation, mode=mode, mode_changed=True, stale=False)
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
            vision = observation["deep_learning"]
            source = vision.get("source_positions", {})
            if state["step"] == 4 and source.get("pipette_expected_xy") is not None:
                if vision.get("stale", True):
                    vision["pipette_position"] = None
                tracking = source.get("pipette_tracking_xy")
                vision["pipette_tracking_position"] = np.asarray(
                    source["pipette_expected_xy"] if tracking is None else tracking)[:2]
                vision["pipette_tracking_source"] = "calibrated trajectory from last accepted tip"
            tip = self.pipette_tip(observation, state)
            if tip is not None:
                self.controller.observer.annotate(self, observation, markers={"pipette": tip}, source_positions=source)
            confidence = vision.get("confidence", {})
            frame = vision.get("source_frame")
            fresh = (not vision.get("stale", True) and frame is not None
                     and frame != state["last_frame"])
            state["last_frame"] = frame
            source_stage = np.asarray(source.get('stage_xy', observation['stage_positions'][:2]))[:2]
            source_reference = (stage.M @ source_stage + stage.r0)[:2]
            reference = state.get('tracked_cell_reference', state['cell'][0][:2])
            expected = source_reference - reference
            cells = [cell for cell in (vision.get("cell_detections") or [])
                     if np.isfinite(cell[:3]).all()
                     and cell[2] >= stage.cellDetectHelper.cellDetector.conf_threshold]
            from ...devices.manipulator.helpers.CellTrackHelper import CellTrackHelper
            if state['step'] == 1 and 'cell_template_evidence' not in state:
                CellTrackHelper.template_evidence(stage, observation, state, expected,
                    use_centroid=config.use_centroid, info=self.controller.info)
            template = state.get('cell_template_evidence')
            template_xy = None if template is None else source_reference - template
            previous = state.get('cell_detector_reference')
            previous_xy = None if previous is None else source_reference - previous
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
                state['cell_detector_reference'] = source_reference - np.asarray(cell[:2])
                display_xy = np.asarray(cell[:2]) + (stage.M @ (observation['stage_positions'][:2] - source_stage))[:2]
                self.controller.observer.annotate(self, observation, markers={"cell": display_xy})
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
            stopped = state["confirming"] and np.allclose(
                observation["manipulator_position"], state["stop_position"], atol=0.01, rtol=0)
            contact = fresh and stopped and rise and touching
            if state["confirming"]:
                state["stop_position"] = np.array(observation["manipulator_position"], copy=True)
                sample = observation["observation_id"]
                if sample != state["last_confirmation_id"]:
                    state["last_confirmation_id"] = sample
                    state["confirmed_readings"] = state["confirmed_readings"] + 1 if contact else 0
                    self.controller.info(f"Hunt CONTACT: fresh={fresh}; stopped={stopped}; rise={rise}; "
                        f"touching={touching}; confirmed={state['confirmed_readings']}/{self.CONTACT_READINGS}")
            calculated.update(cell_detection=cell, vision_fresh=fresh, aligned=aligned,
                              pipette_visible=bool(tip_ok and fresh),
                              contact_depth_ready=contact_depth_ready, distance_to_cell_um=distance_to_cell,
                              touching=touching, resistance_up=rise,
                              adaptive_goal=bool(contact and state["confirmed_readings"] >= self.CONTACT_READINGS))
            if state["step"] == 3:
                calculated["spear_target"] = observation["manipulator_position"][2] + state["cell_focus_z"] - state["pipette_z"]
            calculated["spear_speed"] = abs(config.max_descent_speed)
        self.controller.observer.annotate(self, observation, calculations={
            "window_ready": ready, "resistance_delta": delta, "travel_um": travel})
        return calculated

    def decide(self, observation=None, state=None):
        calculated = observation
        mode = calculated["mode"]
        command = {"calculated": calculated, "mode": mode}
        if self.awaiting_operator:
            return None
        if mode not in ("Manual", "Classic", "Adaptive", "Training", "Agent"):
            self.failure_gate(message=f"Unsupported hunt mode: {mode}")
        if calculated.get("mode_changed"):
            return dict(command, kind="initialize" if mode == "Adaptive" else "stop")
        if calculated["stale"] or mode != self.controller.config.mode:
            return {**command, "kind": "stop"}
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
                    self.controller.info("Hunt CONTACT: confirmation timed out; repeating alignment")
                    return {**command, 'kind': 'adaptive', 'method': 'search',
                            'retry_contact': True, 'pipette_plane': calculated['pipette_plane']}
                return None
            if state["step"] == 4 and calculated["resistance_up"] and calculated["contact_depth_ready"]:
                return {**command, "kind": "adaptive", "method": "confirm"}
            if (state['step'] == 4 and calculated.get('pipette_visible')
                    and calculated.get('cell_detection') is not None and not calculated['aligned']
                    and calculated['contact_depth_ready']):
                self.controller.info("Hunt SPEAR: cell displaced at contact plane; repeating alignment")
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
        command = observation
        if command is None:
            if self.awaiting_operator:
                self.controller.sleep(0.04)
            return
        if not self.action_gate(command["calculated"], state, command=command):
            if command["mode"] == "Adaptive" or state["mode"] == "Adaptive":
                self.finish(state)
            else:
                self.controller.calibrated_unit.stop()
            state["last_velocity"] = None
            return
        kind = command["kind"]
        if kind == "initialize":
            self._start_adaptive(state)
        elif kind == "adaptive":
            controller = self.controller
            method = command["method"]
            if method == "focus":
                self.search(command, state)
                state["mode"] = command["mode"]
                return
            if method != "scan":
                controller.microscope.stop()
            if method == "confirm":
                calculated = command["calculated"]
                controller.info("Hunt CONFIRM: stopping for fresh contact readings")
                self.finish(state)
                state.update(confirming=True, confirmed_readings=0,
                             confirmation_started=time.monotonic(),
                             last_confirmation_id=calculated["observation"]["observation_id"],
                             stop_position=np.array(controller.calibrated_unit.position(), copy=True))
            elif method == "shift":
                target = command["target"]
                state["cell_focus_z" if target == "cell" else "pipette_z"] = state["observation"]["stage_positions"][2]
                if target == "pipette":
                    options = (dict(detected_px=state.pop('confirmed_tip_xy'),
                             expected_px= self.frame_context["pipette_expected_xy"](),
                             max_deviation_um=controller.calibrated_unit.config.pipette_detection_radius_um)
                        if 'confirmed_tip_xy' in state else {})
                    if options:
                        controller.observer.annotate(self, state["observation"], markers={"pipette": options["detected_px"]})
                    centered = self.shift("pipette", state, **options)
                    self.failure_gate()
                    self.success_gate()
                    if centered is False:
                        reason = getattr(controller.calibrated_unit, "last_pipette_move_failure", None)
                        if not isinstance(reason, str) or not reason:
                            reason = "pipette centering did not complete"
                        raise AutopatchError(f"{self.phase_name} pipette SHIFT failed: {reason}; SPEAR cancelled")
                self._extend_travel_limit(state, pipette=target == "pipette")
                if target == "cell" and self.shift("cell", state) is False:
                    return
                self.controller.calibrated_stage.focusHelper.reset_state(state, "cell_template_evidence", "cell_detector_reference")
                state["step"] += 1
                self.failure_gate()
                self.success_gate()
                if controller.config.mode != "Adaptive":
                    self.finish(state)
                    return
                next_z = state["pipette_z"] if target == "cell" else state["cell_focus_z"]
                if target == "pipette":
                    state["hover_z"] = next_z
                self.controller.calibrated_stage.focusHelper.relocate(next_z)
                self.failure_gate()
                self.success_gate()
            elif method in ("search", "scan"):
                if state["step"] == 4:
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
                                 cell_search_bounds=(origin - controller.calibrated_stage.focusHelper.RADIUS_UM, origin + 10.0))
                    controller.info(f"Hunt SEARCH reacquisition: bounds={state['cell_search_bounds']}")
                    stopped_plane = (state['pipette_z'] + travel if 'spear_start_z' in state
                                     else command['pipette_plane'])
                    state.update(step=1, speared=False, pipette_z=stopped_plane)
                    controller.info(f"Hunt SPEAR complete: repeat alignment; cell_z={state['cell_focus_z']}")
                    self.controller.calibrated_stage.focusHelper.relocate(state["cell_focus_z"])
                    self.failure_gate()
                    self.success_gate()
                self.search(command, state)
            elif method == 'check_contact':
                controller.calibrated_unit.stop()
                actual_z = float(controller.calibrated_unit.position()[2])
                motion = state.pop('spear_motion', None)
                if motion is not None:
                    self.controller.calibrated_unit.actionHelper.cancel(motion)
                state.update(step=4, spear_start_z=actual_z, spear_target=actual_z)
                controller.info("Hunt SPEAR: at cell plane; checking stopped contact evidence")
            elif method == "spear":
                controller.info(f"Hunt SPEAR: target_z={command['target']}; speed={command['speed']}")
                self.spear(command["target"], command["speed"], state)
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
                if state["mode"] == "Adaptive" and "pipette_z" in state:
                    actual_z = float(self.controller.calibrated_unit.position()[2])
                    plane = state["pipette_z"] + (actual_z - state["spear_start_z"] if state["step"] == 4 else 0)
                    state["pipette_plane_reference"] = (plane, actual_z)
            else:
                self.controller.calibrated_unit.stop()
            state["last_velocity"] = None
            self._vision = False
            state["fields"] = ["resistance", "manipulator_position"]
        elif kind == "complete":
            self.finish(state, success=True)
        state["mode"] = command["mode"]

    def _extend_travel_limit(self, state, *, pipette=False):
        """Extend the descent budget only with fresh, unused confirmed focus evidence."""
        controller, unit = self.controller, self.controller.calibrated_unit
        cell = state.get("confirmed_travel_cell")
        if pipette and (cell is None or cell["z"] != state["cell_focus_z"]):
            return
        history = "travel_rebase_frames" if pipette else "travel_allowance_frames"
        confirmation = controller.calibrated_stage.focusHelper.confirmed_evidence(
            state.get("focus_sweep", {}), state["observation"]["deep_learning"],
            pipette=pipette, used_frames=state.get(history, set()))
        if confirmation is None:
            return
        cell_z = cell["z"] if pipette else float(state["cell_focus_z"])
        position, start = float(unit.position()[2]), float(state["start_position"][2])
        target = position + cell_z - float(state["pipette_z"])
        if not np.isfinite([position, target, start]).all():
            return
        previous = float(state.get("hunt_travel_limit_um", controller.config.max_distance))
        required = abs(target - start) + unit.actionHelper.MOTION_ENDPOINT_TOLERANCE_UM
        state.setdefault(history, set()).update(confirmation["seen_frames"])
        if not pipette:
            state["confirmed_travel_cell"] = {"z": cell_z}
        state["hunt_travel_limit_um"] = max(previous, required)
        controller.info(f"Hunt travel allowance: target_z={target}; "
            f"limit_um={state['hunt_travel_limit_um']:g}; rebase={pipette}")

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
        if kind == "initialize":
            return mode == "Adaptive" and observation.get("mode_changed", False)
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
                return (state["step"] == 3 and np.isfinite(command["target"])
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
                helper = self.controller.calibrated_stage.focusHelper
                tolerance = helper.BOUNDARY_TOLERANCE_UM if sweep is not None else 0.0
                lower, upper = low - helper.RADIUS_UM - tolerance, high + helper.RADIUS_UM + tolerance
                if state.get("step") == 1 and "cell_reacquisition_origin_z" in state:
                    origin = state["cell_reacquisition_origin_z"]
                    lower = min(lower, origin - helper.RADIUS_UM - tolerance)
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
            observation["cell_detected"] = bool(observation["window_ready"]
                and threshold <= observation["resistance_delta"])
            if observation["cell_detected"]:
                self.controller.info(f"Cell detected: True; resistance: {observation['resistance_delta']}")
                self.controller.calibrated_unit.stop()
            observation["cell_detection_threshold"] = threshold
            self.controller.observer.annotate(self, observation["observation"], calculations={
                "cell_detected": observation["cell_detected"], "cell_threshold": threshold})
        return super().success_gate(bool(observation["cell_detected"]), state)

    def finish(self, state=None, *, success=False):
        if not self.stop_motions(state, all_devices=True):
            return
        if success:
            self.controller.info("Cell Detected")
            self.goal_event = False
            if not self.awaiting_operator:
                self.complete_success()

