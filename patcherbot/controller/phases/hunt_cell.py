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


class HuntCellPhase(PhaseController):
    search = ApproachCellPhase.search
    shift = ApproachCellPhase.shift
    scan = ApproachCellPhase.scan
    spear = ApproachCellPhase.spear
    CONTACT_READINGS = 5
    CONTACT_TIMEOUT_S = 5.0
    ADAPTIVE_TIMEOUT_S = 60.0
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
        return dict(initialized=False, start_position=None,
                    readings=collections.deque(maxlen=5), last_velocity=None,
                    model_prepared=False, mode=None, cell=cell, step=0,
                    goal_distance=0, confirming=False, confirmed_readings=0,
                    started_at=time.monotonic(), last_frame=None,
                    last_confirmation_id=None)

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
        if mode == "Adaptive":
            fields += ["stage_positions", "deep_learning"]
        observation = super().observe(fields=fields,
            evidence=None if mode == "Adaptive" else {}, raw_resistance=True)
        baseline = (controller.first_res if state["initialized"]
                    else controller.resistanceRamp())
        controller.observer.annotate(self, observation, fields={
            "mode": mode, "baseline_resistance": baseline})
        self._show_pipette_indicator(observation)
        self.failure_gate()
        self.success_gate()
        return observation

    def _show_pipette_indicator(self, observation):
        """Display existing tip evidence, with calibrated motion between frames."""
        try:
            unit, stage = self.controller.calibrated_unit, self.controller.calibrated_stage
            vision = observation.get('deep_learning', {})
            if vision.get('stale', True):
                return
            tip = vision.get('pipette_position')
            if tip is None:
                return
            tip = np.asarray(tip, dtype=float)
            if tip.shape != (2,) or not np.isfinite(tip).all():
                return
            position_reader = lambda: (unit.um_to_pixels_relative(unit.position())[:2]
                                       + stage.um_to_pixels_relative(stage.position())[:2])
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
            calculated.update(ApproachCellPhase.calculate(self, observation, state))
            vision = observation["deep_learning"]
            confidence = vision.get("confidence", {})
            frame = vision.get("source_frame")
            fresh = (not vision.get("stale", True) and frame is not None
                     and frame != state["last_frame"])
            state["last_frame"] = frame
            expected = (stage.M @ observation["stage_positions"][:2] + stage.r0)[:2] - state["cell"][0][:2]
            cells = [cell for cell in (vision.get("cell_detections") or [])
                     if np.isfinite(cell[:3]).all()
                     and cell[2] >= stage.cellDetectHelper.cellDetector.conf_threshold]
            cell = min(cells, key=lambda item: np.linalg.norm(item[:2] - expected), default=None)
            if cell is not None and config.hunt_limit_target_radius:
                offset = unit.pixels_to_um_relative(np.append(cell[:2] - expected, 0))
                if np.linalg.norm(offset) > config.hunt_target_radius:
                    cell = None
            tip = vision.get("pipette_position")
            tip_ok = (tip is not None and np.isfinite(tip).all()
                      and (confidence.get("pipette_detector") or 0)
                      >= unit.pipetteCalHelper.pipetteDetector.adapter.threshold)
            gap = float("inf")
            if cell is not None and tip_ok:
                gap = np.linalg.norm(unit.pixels_to_um_relative(np.append(np.asarray(tip)[:2] - cell[:2], 0)))
            depth = vision.get("pipette_focus")
            aligned = fresh and gap <= config.hunt_contact_distance
            touching = (aligned and depth is not None and np.isfinite(depth)
                        and (confidence.get("pipette_focuser") or 0) >= 0.50
                        and abs(depth) <= self.DEPTH_TOLERANCE_UM
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
            calculated.update(cell_detection=cell, vision_fresh=fresh, aligned=aligned,
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
                self.failure_gate(message="Adaptive Hunt lost fresh visual evidence")
            if state["confirming"]:
                return None
            if calculated["resistance_up"] or calculated["touching"]:
                return {**command, "kind": "adaptive", "method": "confirm"}
            if state["step"] == 4 and not calculated["aligned"]:
                self.failure_gate(message="Adaptive Hunt lost pipette/cell alignment during Spear")
            move = ApproachCellPhase.decide(self, calculated, state)
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
            if method != "scan":
                controller.microscope.stop()
            if method == "confirm":
                self.finish(state)
                state.update(confirming=True, confirmed_readings=0,
                             confirmation_started=time.monotonic(),
                             last_confirmation_id=command["calculated"]["observation"]["observation_id"],
                             stop_position=np.array(controller.calibrated_unit.position(), copy=True))
            else:
                if method == "shift":
                    if command["target"] == "cell":
                        state["cell_focus_z"] = state["observation"]["stage_positions"][2]
                        controller.calibrated_stage.center_on_cell(state["cell"],
                            use_centroid=controller.config.use_centroid,
                            max_error_px=controller.config.approach_cell_max_shift)
                    else:
                        state["pipette_z"] = state["observation"]["stage_positions"][2]
                        controller.calibrated_unit.center_pipette(speed=controller.config.approach_pipette_speed)
                    state["step"] += 1
                    command = {**command, **command["next_move"]}
                    self.failure_gate()
                    if controller.config.mode != "Adaptive":
                        self.finish(state)
                        return
                if command["method"] in ("search", "scan"):
                    if state["step"] == 0:
                        state["step"] = 1
                    elif state["step"] == 4:
                        motion = state.pop("spear_motion", None)
                        if motion is not None:
                            motion.stop()
                        controller.calibrated_unit.stop()
                        state.update(step=1, speared=True, pipette_z=command["pipette_plane"])
                    controller.microscope.absolute_move_velocity(command["speed"])
                elif command["method"] == "spear":
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
                return True
            if state["confirming"]:
                return False
            if command["method"] == "spear":
                position = observation["observation"]["manipulator_position"][2]
                return (observation["aligned"] and np.isfinite(command["target"])
                        and command["speed"] > 0 and command["target"] >= position
                        and abs(command["target"] - state["start_position"][2]) < int(self.controller.config.max_distance))
            return command["method"] in ("search", "shift", "scan")
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
            if time.monotonic() - state["started_at"] >= self.ADAPTIVE_TIMEOUT_S:
                raise AutopatchError("Adaptive Hunt timed out")
            if state["confirming"] and time.monotonic() - state["confirmation_started"] >= self.CONTACT_TIMEOUT_S:
                raise AutopatchError("Adaptive Hunt contact was not confirmed")
            raw = observation.get("observation", observation)
            if "stage_positions" in raw and "cell_focus_z" in state:
                low, high = sorted((state["cell_focus_z"], state["pipette_z"]))
                if not low - 5 <= raw["stage_positions"][2] <= high + 5:
                    raise AutopatchError("Adaptive Hunt exceeded microscope search bounds")
        travel = observation.get("travel_um")
        if travel is None:
            travel = abs(observation["manipulator_position"][2] - state["start_position"][2])
        if travel >= int(self.controller.config.max_distance):
            raise AutopatchError(
                "Cell not detected before reaching max hunt distance "
                f"({travel:.1f} um >= {int(self.controller.config.max_distance)} um).")

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
