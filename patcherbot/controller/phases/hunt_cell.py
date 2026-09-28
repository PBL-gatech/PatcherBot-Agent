"""Hunt modes and persistent operations supervised by one observation loop."""

import collections
from concurrent.futures import ThreadPoolExecutor
import math
import sys
import time
import numpy as np
from ..errors import AutopatchError
from ..PhaseController import PhaseController


class HuntCellPhase(PhaseController):
    """Keep observing during actions; only run() commits state transitions."""

    adaptive_steps = ("spear", "search", "center_cell", "scan", "center_pipette")
    transitions = dict(spear="search", search="center_cell", center_cell="scan",
                       scan="center_pipette", center_pipette="spear")

    def __init__(self, controller):
        super().__init__(controller)
        self.state = None
        self.last_observation = None
        self.cell_z_um = None
        self.cell_observation = None
        self.last_cell_location = None

    def run(self, cell=None):
        state = None
        mode, cell_type = self.controller.config.mode, self.controller.config.cell_type
        try:
            state = self.prepare(cell, mode=mode, cell_type=cell_type)
            state.update(vision_executor=ThreadPoolExecutor(max_workers=1, thread_name_prefix="hunt-vision"),
                         vision_future=None, vision_result={})
            self.act(state=state)
            while True:
                self.controller.abort_if_requested()
                self.controller.success_if_requested()
                future = state["vision_future"]
                if future is not None and future.done():
                    state["vision_result"] = future.result()
                    state["vision_future"] = None
                observation = self.observe(**state["observe"], evidence=state["vision_result"])
                self.controller.abort_if_requested()
                self.controller.success_if_requested()
                self.calculate(observation, state)
                contact = state["contact"]
                crossing = state["electrical_crossing"]
                if not state["resistance_valid"] and (mode in ("Classic", "Agent", "Adaptive") or (contact["active"] and mode != "Training")):
                    raise AutopatchError("Invalid resistance during supervised Hunt; motion stopped")
                if crossing and not contact["active"]:
                    contact.update(active=True, reference=observation, recovery=None)
                    state["state_context"].pop("queued_command", None)
                    self.act(observation, state, stop=True)
                    state["pending_transition"] = None
                    state["agent_generation"] += 1
                    if mode == "Agent":
                        self.controller.agenthelper.invalidate_inference()
                elif contact["active"] and not crossing and not contact["latched"]:
                    self.act(observation, state, stop=True)
                    contact.update(active=False, recovery=None)
                    state["pending_transition"] = None
                    state["state_context"].pop("returning", None)
                if mode == "Training" and state["electrical_confirmed"]:
                    contact["latched"] = True
                if self.success_gate(observation, state, mode=mode):
                    self.controller.observer.annotate(self, observation, calculations={
                        "hunt_state": state["active_state"], "hunt_state_id": state["state_id"],
                        "hunt_sample_interval_s": state["sample_interval_s"],
                        "hunt_decision": dict(outcome="success", state_id=state["state_id"],
                            reason="requested_success" if self.controller.success_requested else "contact_confirmed")})
                    self.complete_success()
                    return
                if self.failure_gate(state, mode=mode, cell_type=cell_type):
                    raise AutopatchError("Cell not detected before reaching max hunt distance")
                decision = self.decide(observation, state, mode=mode, cell_type=cell_type)
                if decision["state_id"] != state["state_id"]:
                    raise AutopatchError("Decision belongs to an earlier Hunt state")
                state["decision"] = decision
                if decision["outcome"] == "transition":
                    destination = decision["next_state"]
                    pending = state["pending_transition"]
                    if (self.transitions.get(state["active_state"]) != destination or pending is None
                            or pending["state_id"] != state["state_id"]
                            or pending["next_state"] != destination
                            or pending["reason"] != decision["reason"] or not state["settled"]):
                        raise AutopatchError("Unconfirmed or invalid Hunt state transition")
                    if state["active_state"] == "center_pipette":
                        gap = abs(float(self.cell_z_um) - float(state["pipette_z_um"]))
                        if not math.isfinite(gap):
                            raise AutopatchError("Adaptive progress requires finite measured cell and pipette Z")
                        state["progress_cycles"] += 1
                        baseline = state["progress_baseline_um"]
                        progress = 100.0 * (baseline - gap) / baseline if baseline else 0.0
                        self.controller.observer.annotate(self, observation, fields={
                            "pipette_cell_gap_um": gap, "progress_percent": progress})
                        if progress >= state["required_progress"]:
                            state.update(progress_baseline_um=gap, progress_cycles=0)
                        elif state["progress_cycles"] >= state["progress_cycle_limit"]:
                            raise AutopatchError("Adaptive Hunt failed its measured cycle-progress requirement")
                    if state["active_state"] == "search":
                        self.cell_z_um = float(observation["target_cell_observed_z_um"])
                        self.cell_observation = observation
                    elif state["active_state"] == "scan":
                        state["pipette_z_um"] = state["state_context"]["best_z"]
                    transition = dict(previous=state["active_state"], next=destination,
                                      reason=decision["reason"], previous_state_id=state["state_id"],
                                      state_id=state["state_id"] + 1, timestamp=time.monotonic(),
                                      source_frame=(observation.get("deep_learning") or {}).get("source_frame"),
                                      evidence=decision.get("evidence"))
                    state.update(active_state=destination, state_id=state["state_id"] + 1,
                                 state_context={}, pending_transition=None)
                    self.controller.observer.annotate(self, observation, fields={"hunt_transition": transition})
                    self.controller.info(f"Hunt {transition['previous']} -> {destination}: {decision['reason']}")
                elif decision["command"] is not None:
                    if not self.action_gate(observation, state, command=decision["command"]):
                        raise AutopatchError("Hunt movement rejected by action gate")
                    self.act(observation, state, command=decision["command"])
                self.controller.observer.annotate(self, observation, calculations={
                    "hunt_state": state["active_state"], "hunt_state_id": state["state_id"],
                    "hunt_decision": {key: value for key, value in decision.items() if key != "command"},
                    "hunt_transition": observation.get("hunt_transition"),
                    "hunt_sample_interval_s": state["sample_interval_s"]})
                self.act(observation, state)
                self.controller.sleep(0.04)
        except AutopatchError as error:
            if state is not None and isinstance(self.last_observation, dict):
                state["decision"] = dict(outcome="fail", reason=str(error), state_id=state["state_id"],
                                         evidence=dict(source_frame=(self.last_observation.get("deep_learning") or {}).get("source_frame")),
                                         command=None)
                try:
                    self.controller.observer.annotate(self, self.last_observation, calculations={
                        "hunt_state": state["active_state"], "hunt_state_id": state["state_id"],
                        "hunt_decision": state["decision"],
                        "hunt_transition": self.last_observation.get("hunt_transition"),
                        "hunt_sample_interval_s": state["sample_interval_s"]})
                except Exception:
                    pass  # Diagnostic failure must not obscure the original stop reason.
            raise
        finally:
            pending_error, cleanup_error = sys.exc_info()[1], None
            cleanup = [lambda: self.act(state=state, stop=True)]
            if state is not None and state.get("vision_executor") is not None:
                executor = state.pop("vision_executor")
                cleanup.append(lambda: executor.shutdown(wait=False, cancel_futures=True))
            if state is not None and mode == "Agent":
                cleanup.append(self.controller.agenthelper.invalidate_inference)
            for finish in cleanup:
                try:
                    finish()
                except BaseException as error:
                    if cleanup_error is None:
                        cleanup_error = error
            if pending_error is None and cleanup_error is not None:
                raise cleanup_error

    def prepare(self, cell, *, mode, cell_type):
        self.controller.isrigready()
        if not self.controller.rig_ready:
            raise AutopatchError("Rig not ready for cell hunting")
        if cell is None:
            raise AutopatchError("No cell given to patch!")
        if mode not in ("Classic", "Manual", "Training", "Agent", "Adaptive"):
            raise AutopatchError("Unknown Hunt mode")
        self.cell_z_um = self.cell_observation = self.last_cell_location = None
        self.last_observation = None
        self.controller.pressure.set_pressure(self.controller.config.pressure_near)
        self.controller.sleep(3)
        self.controller.first_res = self.controller.resistanceRamp()
        try:
            baseline = float(self.controller.first_res)
            threshold = float(self.controller.config.cell_R_increase)
            if not math.isfinite(baseline) or baseline <= 0 or not math.isfinite(threshold) or threshold <= 0:
                raise ValueError()
        except (ValueError, TypeError, OverflowError):
            raise AutopatchError("Hunt requires a positive finite resistance baseline and contact threshold")
        self.begin_observations()
        state = dict(cell=cell, mode=mode, cell_type=cell_type,
                     active_state="spear" if mode == "Adaptive" and cell_type == "Slice" else mode.lower(),
                     state_id=1, state_context={}, pending_transition=None, decision=None,
                     readings=collections.deque(maxlen=5), start_position=None,
                     contact=dict(active=False, latched=False, recovery=None),
                     applied_command=None, settled=False, stopped_at=None, settled_at=None,
                     previous_pose=None, last_sample_at=None, agent_generation=0,
                     observe=dict(fields=["resistance", "manipulator_position", "stage_positions",
                                         "deep_learning", "camera_image", "pipette_image_xy", "pipette_defocus_um", "pressure",
                                         "commanded_pressure_mbar", "pressure_atm_state"]))
        self.state = state
        if mode == "Agent":
            self.controller.agenthelper.prepare_model("hunt")
        if mode == "Adaptive":
            if cell_type == "Slice":
                config = self.controller.config
                gap, cycles, percent = float(config.cell_distance), float(config.hunt_progress_cycles), float(config.hunt_min_progress_percent)
                if not math.isfinite(gap) or gap <= 0:
                    raise AutopatchError("Adaptive initial cell distance must be finite and positive")
                if not math.isfinite(cycles) or not cycles.is_integer() or not 1 <= cycles <= 100:
                    raise AutopatchError("Adaptive progress cycles must be a whole number from 1 to 100")
                if not math.isfinite(percent) or not 0 <= percent <= 100:
                    raise AutopatchError("Adaptive progress percentage must be between 0 and 100")
                state.update(progress_baseline_um=gap, progress_cycles=0, pipette_z_um=None,
                             progress_cycle_limit=int(cycles), required_progress=percent)
        stage, unit = self.controller.calibrated_stage, self.controller.calibrated_unit
        self.frame_context = {
            "stage": stage.position,
            "microscope": lambda: unit.microscope.position() / unit.config.microscope_units_per_um,
        }
        self.controller.info(f"Hunt starting: mode={mode}, cell_type={cell_type}, baseline_R={self.controller.first_res} MOhm")
        return state

    def calculate(self, observation, state):
        """Update measured context once; neither acquire nor move hardware."""
        now = time.monotonic()
        self.last_observation = observation
        state["sample_interval_s"] = None if state["last_sample_at"] is None else now - state["last_sample_at"]
        state["last_sample_at"] = now
        reading = observation["resistance"]
        state["readings"].append(reading)
        try:
            state["resistance_valid"] = math.isfinite(float(reading)) and float(reading) > 0
        except (ValueError, TypeError):
            state["resistance_valid"] = False
        threshold = self.controller.config.cell_R_increase
        state["electrical_crossing"] = self._resistance_above_threshold(reading, threshold)
        state["electrical_confirmed"] = self._resistance_threshold_reached(state["readings"], threshold)
        position = np.asarray(observation["manipulator_position"], dtype=float)
        stage = np.asarray(observation["stage_positions"], dtype=float)
        if position.shape != (3,) or stage.shape != (3,) or not np.isfinite(np.r_[position, stage]).all():
            raise AutopatchError("Hunt requires finite hardware positions")
        pose = np.r_[position, stage]
        if state["start_position"] is None:
            state["start_position"] = position.copy()
            state["z_bounds"] = [position[2], position[2], stage[2], stage[2]]
        bounds = state["z_bounds"]
        bounds[:] = [min(bounds[0], position[2]), max(bounds[1], position[2]),
                     min(bounds[2], stage[2]), max(bounds[3], stage[2])]
        previous = state["previous_pose"]
        if state["stopped_at"] is not None and previous is not None:
            settled = now > state["stopped_at"] and np.max(np.abs(pose - previous)) <= 0.5
            if settled and not state["settled"]:
                state["settled_at"] = now
            state["settled"] = bool(settled)
            if not settled:
                state["settled_at"] = None
        state["previous_pose"] = pose
        context = state["state_context"]
        if context:
            if state["active_state"] in ("search", "scan"):
                travel = abs(float(stage[2]) - float(context["start"]["stage_positions"][2]))
                if travel > min(context["travel_limit"], float(self.controller.config.max_distance)):
                    raise AutopatchError("Microscope exceeded its operation travel envelope")
            if state["active_state"] in ("center_cell", "center_pipette"):
                field = "stage_positions" if state["active_state"] == "center_cell" else "manipulator_position"
                travel = np.linalg.norm(np.asarray(observation[field][:2]) - np.asarray(context["start"][field][:2]))
                if not math.isfinite(float(travel)) or travel > 50.:
                    raise AutopatchError("Shift exceeded its 50 um planar envelope")
            if time.monotonic() - context["entered_at"] > 60:
                raise AutopatchError("Hunt action exceeded its monitoring timeout")
        return observation

    def decide(self, observation, state, *, mode, cell_type):
        """Evaluate the current process; state transitions are committed by run()."""
        decision = dict(outcome="hold", reason="observing", state_id=state["state_id"], command=None,
                        evidence=dict(observed_at=observation.get("observed_at"),
                                      source_frame=(observation.get("deep_learning") or {}).get("source_frame"),
                                      resistance=observation.get("resistance"),
                                      settled=state["settled"]))
        contact = state["contact"]
        if contact["active"] or contact["latched"]:
            decision["reason"] = "contact_confirmation"
            if mode == "Adaptive" and state["electrical_confirmed"]:
                decision["command"] = self._recover_contact_visuals(observation, state, mode=mode)
            return decision
        if mode in ("Manual", "Training"):
            decision["reason"] = "operator_motion"
            return decision
        if mode == "Agent":
            if not self.usable_tip(observation):
                decision.update(reason="awaiting_visual_evidence",
                    command=dict(kind="stop") if state["applied_command"] is not None else None)
                return decision
            helper = self.controller.agenthelper
            generation = (id(state), state["agent_generation"])
            helper.request_inference(observation, generation)
            ready, action = helper.poll_inference(generation)
            if not ready:
                decision["reason"] = "agent_pending"
                return decision
            action = [0., 0., 0.] if action is None else action
            if len(action) != 3 or not np.isfinite(np.asarray(action, dtype=float)).all():
                raise AutopatchError("Hunt agent must return three finite velocity components")
            vx, vy, vz = map(float, action)
            xy = self.controller.calibrated_unit.pixels_to_um_relative([-vx, -vy, 0.])
            decision.update(outcome="continue", reason="agent_action", command=dict(
                kind="relative_velocity", device=self.controller.calibrated_unit, value=[float(xy[0]), float(xy[1]), vz]))
            return decision
        if mode != "Adaptive" or cell_type != "Slice":
            decision.update(outcome="continue", reason="configured_descent", command=dict(
                kind="velocity", device=self.controller.calibrated_unit, value=[0., 0., self.controller.config.max_descent_speed]))
            return decision
        context = state["state_context"]
        step = state["active_state"]
        if not context:
            bounds = state["z_bounds"]
            maximum, margin = float(self.controller.config.max_distance), float(self.controller.config.hunt_search_margin)
            if not math.isfinite(maximum) or maximum <= 0 or not math.isfinite(margin) or margin < 0:
                raise AutopatchError("Search/Scan distance settings must be finite and nonnegative")
            context.update(start=observation, entered_at=time.monotonic(), waiting=0,
                           travel_limit=min(maximum, bounds[1] - bounds[0] + bounds[3] - bounds[2] + margin))
            self.controller.info(f"Hunt mode={mode}, operation={step}, R={observation['resistance']} MOhm, travel_limit={context['travel_limit']} um")
        queued = context.get("queued_command")
        if queued is not None:
            if not state["settled"]:
                decision["reason"] = "awaiting_stop_before_move"
                return decision
            decision.update(outcome="continue", reason="stopped_move_dispatch", command=context.pop("queued_command"))
            return decision
        target = context.get("move_target")
        if target is not None:
            field = "stage_positions" if step == "center_cell" else "manipulator_position"
            if (time.monotonic() <= context["move_issued_at"]
                    or np.linalg.norm(np.asarray(observation[field][:2]) - target) > 1.0):
                decision["reason"] = "alignment_move_in_progress"
                return decision
            context["after_move"] = time.monotonic()
            decision.update(reason="alignment_target_reached", command=dict(kind="stop"))
            return decision
        if step == "scan" and context.get("returning") and not context.get("return_stopped"):
            error = context["best_z"] - float(observation["stage_positions"][2])
            if abs(error) <= 1.0 or error * context["return_direction"] <= 0:
                context.update(return_stopped=True, after_move=time.monotonic())
                decision.update(reason="focus_return_stopped", command=dict(kind="stop"))
            else:
                decision.update(reason="returning_to_best_focus", command=dict(kind="velocity",
                    device=self.controller.microscope,
                    value=context["return_direction"] * abs(float(self.controller.config.max_descent_speed))))
            return decision
        pending = state["pending_transition"]
        fresh = self.fresh_visual
        if pending is not None:
            if pending["state_id"] != state["state_id"]:
                raise AutopatchError("Pending transition belongs to an earlier state")
            pending["attempts"] += 1
            if pending["attempts"] > 50:
                raise AutopatchError("Operation received no confirming evidence after stopping")
            if not state["settled"]:
                decision["reason"] = "awaiting_stop"
                return decision
            geometric = pending["reason"] == "travel_limit"
            if not geometric and not fresh(observation, pending["frame"], state["settled_at"]):
                decision["reason"] = "awaiting_post_stop_evidence"
                return decision
        if step != "spear":
            after_move = context.get("after_move")
            if after_move is not None and state["stopped_at"] is not None:
                if not state["settled"]:
                    decision["reason"] = "awaiting_stop_after_move"
                    return decision
                after_move = max(after_move, state["settled_at"])
            if not fresh(observation, context.get("frame"), after_move):
                context["waiting"] += 1
                if context["waiting"] >= 50:
                    raise AutopatchError(f"{step} received no fresh visual evidence")
                decision.update(reason="awaiting_visual_evidence", command=dict(kind="stop") if state["applied_command"] is not None else None)
                return decision
            context.update(frame=(observation.get("deep_learning") or {}).get("source_frame"), waiting=0)
        monitor = self.shift if step.startswith("center_") else getattr(self, step)
        completed, command = monitor(observation, state)
        if completed:
            reason = context["exit_reason"]
            if pending is not None and pending["reason"] == reason:
                decision.update(outcome="transition", next_state=self.transitions[step], reason=reason)
            else:
                state["pending_transition"] = dict(state_id=state["state_id"], reason=reason,
                    next_state=self.transitions[step], frame=(observation.get("deep_learning") or {}).get("source_frame"), attempts=0)
                decision.update(reason=reason, command=dict(kind="stop"))
        else:
            if pending is not None:
                state["pending_transition"] = None
            decision.update(outcome="continue" if command else "hold", reason="state_condition_not_met", command=command)
        command = decision["command"]
        if command is not None and command["kind"] == "relative" and not state["settled"]:
            context["queued_command"] = command
            decision.update(outcome="hold", reason="stop_before_positional_move", command=dict(kind="stop"))
        return decision

    def act(self, observation=None, state=None, *, command=None, stop=False):
        """Dispatch once; never acquire observations or wait for movement completion."""
        if (not stop and (command is None or command["kind"] != "stop")
                and state is not None and state.get("vision_executor") is not None
                and state["vision_future"] is None):
            state["vision_future"] = state["vision_executor"].submit(
                self.controller.infer_frame, state["observe"]["fields"], self.frame_context)
        if stop or (command is not None and command["kind"] == "stop"):
            first_error = None
            if state is not None:
                state["state_context"].pop("move_target", None)
                state["state_context"].pop("move_issued_at", None)
            for device in (self.controller.calibrated_stage, self.controller.calibrated_unit, self.controller.microscope):
                try:
                    device.stop()
                except BaseException as error:
                    if first_error is None:
                        first_error = error
            if state is not None:
                state.update(applied_command=None, stopped_at=time.monotonic(), settled=False, settled_at=None)
                state["stop_frame"] = (observation.get("deep_learning") or {}).get("source_frame") if observation else None
            if first_error is not None:
                raise first_error
            return
        if command is None:
            return
        self.controller.abort_if_requested()
        self.controller.success_if_requested()
        kind, device, value = command["kind"], command["device"], command["value"]
        key = (kind, id(device), tuple(np.asarray(value).reshape(-1)))
        if kind == "relative":
            context = state["state_context"]
            field = "stage_positions" if device is self.controller.calibrated_stage else "manipulator_position"
            target = np.asarray(observation[field][:2], dtype=float) + np.asarray(value, dtype=float)[:2]
            device.relative_move(value)
            context.update(move_target=target, move_issued_at=time.monotonic())
            context["moves"] = context.get("moves", 0) + 1
            context["commanded_distance"] = context.get("commanded_distance", 0.) + float(np.linalg.norm(value))
            context["previous_error"] = command["alignment_error"]
        elif kind == "relative_velocity":
            device.relative_move_group_velocity(value)
        elif kind == "velocity":
            if key != state["applied_command"]:
                if device is self.controller.microscope:
                    device.absolute_move_velocity(value)
                else:
                    device.absolute_move_group_velocity(value)
        else:
            raise AutopatchError("Unknown Hunt movement command")
        state.update(applied_command=key, stopped_at=None, settled=False, settled_at=None)

    def action_gate(self, observation=None, state=None, *, command=None):
        """Validate the command against its mode, device and retained movement budget."""
        if observation is None or self.controller.abort_requested or self.controller.success_requested:
            return False
        if command is None or command["kind"] == "stop":
            return True
        try:
            kind, device = command["kind"], command["device"]
            value = np.asarray(command["value"], dtype=float)
            if not np.isfinite(value).all() or state["mode"] in ("Manual", "Training"):
                return False
            recovery = state["contact"].get("recovery")
            if state["contact"]["active"] and recovery is None:
                return False
            step, context = state["active_state"], state["state_context"]
            if recovery is not None:
                context = recovery
                step = recovery["phase"]
            if kind == "relative":
                expected = self.controller.calibrated_stage if step == "center_cell" else self.controller.calibrated_unit
                field = "stage_positions" if step == "center_cell" else "manipulator_position"
                current = np.asarray(observation[field][:2], dtype=float)
                origin = np.asarray(context["start"][field][:2], dtype=float)
                return (state["mode"] == "Adaptive" and step in ("center_cell", "center_pipette") and device is expected
                        and value.shape in ((2,), (3,)) and (value.size == 2 or value[2] == 0)
                        and np.linalg.norm(value[:2]) + context.get("commanded_distance", 0.) <= 50.000001
                        and np.linalg.norm(current + value[:2] - origin) <= 50.000001)
            if device is self.controller.microscope:
                if state["mode"] != "Adaptive" or value.shape != () or step not in ("search", "scan", "return"):
                    return False
                limit = min(float(context["travel_limit"]), float(self.controller.config.max_distance))
                if not math.isfinite(limit) or limit <= 0:
                    return False
                origin = float(context["start"]["stage_positions"][2])
                z = float(observation["stage_positions"][2])
                if abs(z - origin) > limit:
                    return False
                if context.get("returning") or step == "return":
                    target = context["tip_z"] if recovery is not None else context["best_z"]
                    error = float(target) - z
                    return (kind == "velocity" and math.isfinite(float(target))
                            and abs(float(target) - origin) <= limit and value * error > 0)
                up = float(device.up_direction)
                if up not in (-1., 1.):
                    return False
                direction = up * (-1 if step == "search" else 1)
                return kind == "velocity" and value * direction > 0 and abs(z - origin) < limit
            if device is not self.controller.calibrated_unit or kind not in ("velocity", "relative_velocity") or value.shape != (3,):
                return False
            if state["mode"] != "Agent" and np.any(value[:2] != 0):
                return False
            if np.all(value == 0):
                return True
            if step == "spear":
                speed = float(self.controller.config.max_descent_speed)
                if not math.isfinite(speed) or speed == 0 or value[2] * speed <= 0:
                    return False
                origin = context["start"]["manipulator_position"][2]
                limit = float(self.controller.config.max_distance)
            else:
                origin, limit = state["start_position"][2], float(self.controller.config.max_distance)
            return math.isfinite(limit) and limit > 0 and abs(observation["manipulator_position"][2] - origin) < limit
        except (KeyError, TypeError, ValueError, IndexError):
            return False

    def spear(self, observation, state):
        """Descend until the selected cell is lost or the travel limit is reached."""
        context = state["state_context"]
        descent = abs(float(observation["manipulator_position"][2]) - float(context["start"]["manipulator_position"][2]))
        context["maximum_descent_um"] = max(context.get("maximum_descent_um", 0.), descent)
        if observation.get("target_cell_status") == "ambiguous":
            raise AutopatchError("Spear cannot distinguish the selected cell")
        if observation.get("target_cell_valid"):
            context["cell_seen"] = True
        lost = observation.get("target_cell_status") == "not_detected" and context.get("cell_seen", False)
        if descent >= float(self.controller.config.max_distance) or lost:
            context["exit_reason"] = "travel_limit" if descent >= float(self.controller.config.max_distance) else "cell_lost_confirmed"
            return True, None
        return False, dict(kind="velocity", device=self.controller.calibrated_unit,
                           value=[0., 0., float(self.controller.config.max_descent_speed)])

    def search(self, observation, state):
        context = state["state_context"]
        if observation.get("target_cell_status") == "ambiguous":
            raise AutopatchError("Search cannot distinguish the selected cell")
        z = float(observation["stage_positions"][2])
        origin = float(context["start"]["stage_positions"][2])
        if observation.get("target_cell_valid"):
            measured = float(observation["target_cell_observed_z_um"])
            if not math.isfinite(measured) or abs(measured - origin) > context["travel_limit"]:
                raise AutopatchError("Search cell observation lies outside its travel envelope")
            context["candidate_cell_z_um"] = measured
            context["exit_reason"] = "selected_cell_acquired"
            return True, None
        if abs(z - origin) >= context["travel_limit"]:
            raise AutopatchError("Search reached its observation-derived distance limit")
        return False, dict(kind="velocity", device=self.controller.microscope,
                           value=-float(self.controller.microscope.up_direction) * abs(float(self.controller.config.max_descent_speed)))

    def scan(self, observation, state):
        context = state["state_context"]
        evidence = observation.get("deep_learning") or {}
        z = float(observation["stage_positions"][2])
        origin = float(context["start"]["stage_positions"][2])
        try:
            focus = float(evidence["pipette_focus"])
            source_z = float(evidence["source_positions"]["microscope"])
            valid = self.usable_tip(observation) and math.isfinite(focus) and math.isfinite(source_z) and abs(source_z - origin) <= context["travel_limit"]
        except (KeyError, TypeError, ValueError):
            focus, valid = float("nan"), False
        if context.get("returning") or state["pending_transition"] is not None:
            if not valid or abs(z - context["best_z"]) > 1.0 or abs(focus) > context["best_focus"] + 1.0:
                raise AutopatchError("Scan focus was not confirmed after stopping")
            context["exit_reason"] = "focus_verified"
            return True, None
        if valid:
            previous, best = context.get("previous_focus"), context.get("best_focus", float("inf"))
            bracketed = previous is not None and (previous * focus < 0 or (context.get("improving") and abs(focus) > best + 0.1))
            if abs(focus) < best:
                context.update(best_focus=abs(focus), best_z=source_z, improving=previous is not None)
            context["previous_focus"] = focus
            if abs(focus) <= 1.0 or bracketed:
                if abs(z - context["best_z"]) <= 1.0:
                    context["exit_reason"] = "focus_verified"
                    return True, None
                direction = math.copysign(1., context["best_z"] - z)
                context.update(returning=True, return_stopped=False, return_direction=direction)
                return False, dict(kind="velocity", device=self.controller.microscope,
                                   value=direction * abs(float(self.controller.config.max_descent_speed)))
        if abs(z - origin) >= context["travel_limit"]:
            raise AutopatchError("Scan reached its distance limit without resolving focus")
        return False, dict(kind="velocity", device=self.controller.microscope,
                           value=float(self.controller.microscope.up_direction) * abs(float(self.controller.config.max_descent_speed)))

    def shift(self, observation, state):
        context = state["state_context"]
        cell = state["active_state"] == "center_cell"
        device = self.controller.calibrated_stage if cell else self.controller.calibrated_unit
        evidence = observation.get("deep_learning") or {}
        if cell and not observation.get("target_cell_valid"):
            context["missing_target"] = context.get("missing_target", 0) + 1
            if context["missing_target"] >= 50:
                raise AutopatchError("Shift could not associate the selected cell after 50 fresh observations")
            return False, None
        context["missing_target"] = 0
        try:
            point = np.asarray(observation["target_cell_image_xy"] if cell else evidence["pipette_position"], dtype=float).reshape(2)
            shape = evidence["source_image_shape"]
            if not np.isfinite(point).all() or not (0 <= point[0] < shape[1] and 0 <= point[1] < shape[0]):
                raise ValueError()
            correction = np.asarray(device.pixels_to_um_relative([shape[1] / 2. - point[0], shape[0] / 2. - point[1], 0.]), dtype=float)[:2]
            if not np.isfinite(correction).all():
                raise ValueError()
        except (KeyError, TypeError, ValueError, IndexError):
            raise AutopatchError("Shift requires an observed in-image target and finite calibration")
        field = "stage_positions" if cell else "manipulator_position"
        travel = np.linalg.norm(np.asarray(observation[field][:2]) - np.asarray(context["start"][field][:2]))
        if travel > 50:
            raise AutopatchError("Shift exceeded its 50 um planar envelope")
        distance = float(np.linalg.norm(correction))
        if distance <= 1.0:
            context["exit_reason"] = "alignment_verified"
            return True, None
        if "previous_error" in context and distance >= context["previous_error"] - 0.05:
            raise AutopatchError("Shift did not improve alignment after its previous move")
        remaining = 50. - context.get("commanded_distance", 0.)
        if remaining <= 0 or context.get("moves", 0) >= 8:
            raise AutopatchError("Shift exhausted its bounded alignment budget")
        value = correction * min(1., remaining / distance)
        if not cell:
            value = np.r_[value, 0.]
        return False, dict(kind="relative", device=device, value=value, alignment_error=distance)

    def failure_gate(self, state=None, *, mode, cell_type):
        if self.controller.abort_requested:
            return True
        if state is None or state["start_position"] is None or mode == "Training" or (mode == "Adaptive" and cell_type == "Slice"):
            return False
        return abs(self.last_observation["manipulator_position"][2] - state["start_position"][2]) >= self.controller.config.max_distance

    def success_gate(self, observation=None, state=None, *, mode):
        if self.controller.success_requested:
            return True
        if state is None or mode == "Training" or not state.get("electrical_confirmed", False):
            return False
        if mode != "Adaptive":
            return True
        return bool(isinstance(observation, dict) and state["settled"] and self.fresh_visual(
            observation, state.get("stop_frame"), state["settled_at"]) and self._adaptive_contact_confirmed(observation))

    def _recover_contact_visuals(self, observation, state, *, mode):
        """Advance microscope-only contact recovery once; the main loop keeps sampling."""
        if mode != "Adaptive":
            return None
        contact = state["contact"]
        recovery = contact["recovery"]
        if recovery is None:
            bounds = state["z_bounds"]
            limit = min(float(self.controller.config.max_distance), bounds[1] - bounds[0] + bounds[3] - bounds[2] + float(self.controller.config.hunt_search_margin))
            if not math.isfinite(limit) or limit <= 0:
                raise AutopatchError("Contact recovery requires a positive microscope envelope")
            recovery = dict(start=contact["reference"], travel_limit=limit, phase="scan",
                            entered_at=time.monotonic(), waiting=0, frame=None, pending=None, tip_z=None)
            contact["recovery"] = recovery
        if time.monotonic() - recovery["entered_at"] > 60:
            raise AutopatchError("Contact recovery exceeded its monitoring timeout")
        z = float(observation["stage_positions"][2])
        if abs(z - recovery["start"]["stage_positions"][2]) > min(recovery["travel_limit"], float(self.controller.config.max_distance)):
            raise AutopatchError("Contact recovery exceeded its fixed microscope envelope")
        if observation.get("target_cell_id") != recovery["start"].get("target_cell_id") or observation.get("target_cell_status") == "ambiguous":
            raise AutopatchError("Contact recovery target changed or became ambiguous")
        if recovery["phase"] == "return" and not recovery.get("return_stopped"):
            error = recovery["tip_z"] - z
            if abs(error) <= 1.0 or error * recovery["return_direction"] <= 0:
                recovery["return_stopped"] = True
                return dict(kind="stop")
            return dict(kind="velocity", device=self.controller.microscope,
                        value=recovery["return_direction"] * abs(float(self.controller.config.max_descent_speed)))
        after = state["settled_at"] if state["stopped_at"] is not None else None
        if (state["stopped_at"] is not None and not state["settled"]) or not self.fresh_visual(observation, recovery["frame"], after):
            recovery["waiting"] += 1
            if recovery["waiting"] >= 50:
                raise AutopatchError("Contact recovery received no fresh confirming evidence")
            return dict(kind="stop") if state["applied_command"] is not None else None
        recovery.update(waiting=0, frame=(observation.get("deep_learning") or {}).get("source_frame"))
        tip, cell = self.usable_tip(observation), observation.get("target_cell_valid", False)
        if self._adaptive_contact_confirmed(observation):
            return dict(kind="stop") if state["applied_command"] is not None else None
        pending = recovery["pending"]
        if pending == "tip" and tip:
            recovery.update(tip_z=z, phase="search", pending=None)
        elif pending == "cell" and cell:
            if recovery["tip_z"] is None:
                raise AutopatchError("Contact recovery has no verified pipette focus")
            error = recovery["tip_z"] - z
            recovery.update(phase="return", pending="return", return_direction=math.copysign(1., error),
                            return_stopped=abs(error) <= 1.0)
            if recovery["return_stopped"]:
                return dict(kind="stop")
            return dict(kind="velocity", device=self.controller.microscope,
                        value=recovery["return_direction"] * abs(float(self.controller.config.max_descent_speed)))
        elif pending == "return":
            raise AutopatchError("Recovered tip is not within the selected cell contact distance")
        else:
            recovery["pending"] = None
        if recovery["phase"] == "scan" and tip:
            recovery["pending"] = "tip"
            return dict(kind="stop")
        if recovery["phase"] == "search" and cell:
            recovery["pending"] = "cell"
            return dict(kind="stop")
        speed = abs(float(self.controller.config.max_descent_speed))
        direction = float(self.controller.microscope.up_direction) * (-1 if recovery["phase"] == "search" else 1)
        return dict(kind="velocity", device=self.controller.microscope, value=direction * speed)

    def enrich_observation(self, observation):
        """Associate raw detections with this attempt's selected cell after raw recording."""
        evidence = observation.get("deep_learning") or {}
        positions = evidence.get("source_positions") or {}
        target = None if self.state is None else self.state["cell"]
        reference = np.asarray(target[0], dtype=float) if target is not None else np.asarray([])
        identity = tuple(float(value) for value in reference)
        candidates, expected = [], np.asarray([np.nan, np.nan])
        source_z, context_valid = float("nan"), False
        try:
            source_xy = np.asarray(positions["stage"], dtype=float).reshape(-1)[:2]
            source_z = float(positions["microscope"])
            available, sampled = float(evidence["frame_available_at"]), float(evidence["positions_sampled_at"])
            context_valid = (source_xy.size == 2 and np.isfinite(source_xy).all()
                and all(math.isfinite(value) for value in (source_z, available, sampled))
                and sampled >= available and reference.size >= 2 and np.isfinite(reference[:2]).all()
                and evidence.get("source_frame") is not None and not evidence.get("stale", True))
        except (KeyError, TypeError, ValueError):
            pass
        detections = evidence.get("cell_detections")
        association_available = context_valid and detections is not None
        if association_available:
            stage = self.controller.calibrated_stage
            expected = np.asarray(stage.um_to_pixels_relative(source_xy))[:2] + np.asarray(stage.r0)[:2] - reference[:2]
            known = self.last_cell_location
            if known is not None and known["target_cell_id"] == identity:
                expected = known["image_xy"] + np.asarray(stage.um_to_pixels_relative(source_xy - known["stage_xy"]))[:2]
            radius = float(getattr(self.controller.config, "hunt_target_radius", 50.0)) if getattr(self.controller.config, "hunt_limit_target_radius", True) else float("inf")
            if math.isnan(radius) or radius <= 0:
                raise AutopatchError("Target matching radius must be positive")
            for detection in detections:
                try:
                    x, y, confidence = map(float, detection[:3])
                    height, width = evidence["source_image_shape"][:2]
                    delta = stage.pixels_to_um_relative([x - expected[0], y - expected[1], 0.])
                    distance = math.hypot(float(delta[0]), float(delta[1]))
                    if all(math.isfinite(value) for value in (x, y, confidence, distance)) and confidence > 0 and 0 <= x < width and 0 <= y < height and distance <= radius:
                        candidates.append((distance, x, y))
                except (KeyError, TypeError, ValueError, IndexError):
                    continue
            candidates.sort()
        ambiguous = len(candidates) > 1 and candidates[1][0] - candidates[0][0] <= 1.
        valid = association_available and bool(candidates) and not ambiguous
        observation.update(visual_context_valid=context_valid, target_cell_id=identity,
            target_cell_reference_coordinates=reference.copy(), target_cell_expected_xy=expected,
            target_cell_image_xy=np.asarray(candidates[0][1:] if valid else [np.nan, np.nan]),
            target_cell_observed_z_um=source_z if valid else float("nan"),
            target_cell_candidate_count=len(candidates), target_cell_valid=valid,
            target_cell_status="observed" if valid else "ambiguous" if ambiguous else "not_detected" if association_available else "unavailable")
        if valid:
            self.last_cell_location = dict(target_cell_id=identity, image_xy=observation["target_cell_image_xy"].copy(),
                stage_xy=source_xy.copy(), source_frame=evidence.get("source_frame"))
        observation["target_cell_last_known_z_um"] = self.cell_z_um if self.cell_z_um is not None else float("nan")
        accepted = (self.cell_observation.get("deep_learning") or {}) if self.cell_observation is not None else {}
        observation["target_cell_last_known_source_frame"] = accepted.get("source_frame")
        return observation

    @staticmethod
    def fresh_visual(observation, previous_frame=None, after=None):
        """Check retrieval freshness; buffered sensor-frame freshness is not established."""
        evidence = observation.get("deep_learning") or {}
        frame, retrieved, sampled = (evidence.get(key) for key in
                                   ("source_frame", "frame_retrieval_started_at", "positions_sampled_at"))
        return bool(frame is not None and frame != previous_frame and not evidence.get("stale", True)
            and isinstance(retrieved, (int, float)) and math.isfinite(retrieved)
            and isinstance(sampled, (int, float)) and math.isfinite(sampled)
            and sampled >= retrieved and observation.get("visual_context_valid", False)
            and (after is None or retrieved >= after))

    @staticmethod
    def usable_tip(observation):
        evidence = observation.get("deep_learning") or {}
        try:
            tip = np.asarray(evidence["pipette_position"], dtype=float).reshape(-1)
            shape = evidence["source_image_shape"]
            return bool(observation.get("visual_context_valid", False)
                and evidence.get("source_frame") is not None and not evidence.get("stale", True)
                and (evidence.get("status") or {}).get("pipette_detector") not in ("error", "no_detection", "disabled", "no_frame")
                and tip.size >= 2 and np.isfinite(tip[:2]).all()
                and 0 <= tip[0] < shape[1] and 0 <= tip[1] < shape[0])
        except (KeyError, TypeError, ValueError, IndexError):
            return False

    def _adaptive_contact_confirmed(self, observation):
        """Require a fresh tip near this attempt's observed or last-known cell."""
        if not isinstance(observation, dict) or not observation.get("visual_context_valid", False):
            return False
        evidence = observation.get("deep_learning") or {}
        if not self.usable_tip(observation) or observation.get("target_cell_status") == "ambiguous":
            return False
        try:
            tip = np.asarray(evidence["pipette_position"], dtype=float).reshape(-1)[:2]
            current_stage = np.asarray(evidence["source_positions"]["stage"], dtype=float).reshape(-1)[:2]
            if observation.get("target_cell_valid", False):
                cell = np.asarray(observation["target_cell_image_xy"], dtype=float).reshape(-1)[:2]
            else:
                known = self.last_cell_location
                if known is None or known["target_cell_id"] != observation.get("target_cell_id"):
                    return False
                cell = np.asarray(known["image_xy"], dtype=float) + (
                    np.asarray(self.controller.calibrated_stage.M) @ (current_stage - known["stage_xy"]))[:2]
            limit = float(getattr(self.controller.config, "hunt_contact_distance", 5.0))
            if (tip.size != 2 or cell.size != 2 or current_stage.size != 2
                    or not all(np.isfinite(v).all() for v in (tip, cell, current_stage))
                    or (tip < 0).any() or (cell < 0).any()
                    or not math.isfinite(limit) or limit <= 0):
                return False
            delta = self.controller.calibrated_stage.pixels_to_um_relative([*(tip - cell), 0.0])
            distance = math.hypot(float(delta[0]), float(delta[1]))
        except (KeyError, TypeError, ValueError, IndexError):
            return False
        self.controller.observer.annotate(self, observation, fields={
            "hunt_tip_cell_distance_um": distance,
            "hunt_contact_uses_last_known_cell": not observation.get("target_cell_valid", False)})
        return math.isfinite(distance) and distance <= limit

    def _resistance_above_threshold(self, reading, threshold):
        try:
            baseline = float(self.controller.first_res)
            threshold = float(threshold)
            reading = float(reading)
        except (TypeError, ValueError, OverflowError):
            return False
        return (math.isfinite(baseline) and baseline > 0
                and math.isfinite(threshold) and threshold > 0
                and math.isfinite(reading) and reading > 0
                and reading - baseline >= threshold)

    def _resistance_threshold_reached(self, readings, threshold):
        """Require sustained contact evidence across five current readings."""
        return len(readings) >= 5 and all(
            self._resistance_above_threshold(value, threshold)
            for value in list(readings)[-5:])

    def _isCellDetected(self, lastResDeque, cellThreshold=0.15):
        """Compatibility wrapper for callers outside the phase lifecycle."""
        detected = self._resistance_threshold_reached(lastResDeque, cellThreshold)
        if detected:
            self.controller.calibrated_unit.stop()
        return detected

