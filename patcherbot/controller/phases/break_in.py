"""Implementation of the break_in patch phase."""
from dataclasses import dataclass
import numpy as np
from ..errors import AutopatchError
from ..PhaseController import PhaseController


@dataclass
class BreakInState:
    """Attempt progression; operational settings remain owned by config."""
    trials: int = 0
    good_count: int = 0
    electrical_ready: bool = False
    agent_prepared: bool = False
    pulse_active: bool = False


class BreakInPhase(PhaseController):
    def run(self, cell=None):
        try:
            state = self.prepare(cell)
            while True:
                observation = self.observe()
                calculated = self.calculate(observation, state)
                command = self.decide(calculated, state)
                self.act(command, state)
        finally:
            self.finish()

    def prepare(self, cell=None):
        self.begin_observations()
        self.observation_windows = {}
        self._break_in_state = BreakInState()
        config = self.controller.config
        self.controller.info(f"Target Resistance: {config.max_cell_R}; "
                             f"Target Capacitance: {config.min_cell_C}, "
                             f"Target Access Resistance: {config.max_access_R}")
        return self._break_in_state

    def observe(self, *, fields=None, num_measurements=None,
                interval=None, evidence=None):
        controller = self.controller
        state = self._break_in_state
        while True:
            self.failure_gate()
            self.success_gate()
            mode = controller.config.mode
            if mode not in ("Manual", "Training", "Classic", "Adaptive", "Agent"):
                raise AutopatchError(f"Unsupported break_in mode: {mode}")
            if mode != "Training" and not self.awaiting_operator and not state.electrical_ready:
                controller.daq.setCellMode(True)
            requested = ["access_resistance", "resistance", "capacitance"]
            self.observation_windows = {}
            if mode == "Agent" and not self.awaiting_operator:
                helper = controller.agenthelper
                if (not state.agent_prepared or getattr(helper, "model_type", None) != "break_in"
                        or helper.agent is None):
                    helper.prepare_model("break_in")
                    state.agent_prepared = True
                width = helper.observation_input_width("resistance_input")
                if width > 0:
                    self.observation_windows["resistance_input"] = {
                        "field": "resistance", "width": width,
                        "predicate": lambda sample: not float(np.asarray(
                            sample.get("access_resistance", np.nan)
                        ).reshape(-1)[0]) <= controller.config.max_access_R,
                        "finite_only": True, "fill_value": 0.0,
                    }
                requested += ["manipulator_position", "pipette_image_xy",
                              "pipette_defocus_um", "stage_positions", "camera_image",
                              "pressure", "commanded_pressure_mbar", "pressure_atm_state"]
            if mode != "Training" and not self.awaiting_operator and not state.electrical_ready:
                controller.info(f"{mode}: Attempting Break in...")
                controller.sleep(3)
                if controller.config.mode != mode:
                    continue
                controller.pressure.set_pressure(controller.config.pulse_pressure_break_in)
                controller.amplifier.set_zap_duration(25 * 1e-6)
                state.electrical_ready = True
            if controller.config.mode != mode:
                continue
            observation = super().observe(
                fields=requested if fields is None else fields,
                num_measurements=5 if num_measurements is None else num_measurements,
                interval=0.200 if interval is None else interval, evidence=evidence,
            )
            self.failure_gate()
            self.success_gate()
            if controller.config.mode != mode:
                self.success_gate(state=state, reset=True)
                continue
            controller.info(
                f"Pre-action Resistance: {observation['resistance']}; "
                f"Membrane Capacitance: {observation['capacitance']}, "
                f"Access Resistance: {observation['access_resistance']}")
            return {"observation": observation, "mode": mode}

    def calculate(self, observation=None, state=None):
        self.failure_gate()
        self.success_gate()
        trial = state.trials + 1
        return {**observation, "trial": trial, "wait_s": 0.50 * (1 + trial / 2)}

    def decide(self, observation=None, state=None):
        self.failure_gate()
        self.success_gate()
        controller = self.controller
        mode = controller.config.mode
        if observation["mode"] != mode:
            self.success_gate(state=state, reset=True)
            return None
        measured = observation["observation"]
        if self.awaiting_operator:
            return None
        complete = self.success_gate(measured, state)
        if complete or self.goal_event:
            self.finish(success=True, observation=measured)
            return None
        if not self.action_gate(measured, state):
            return None
        if mode in ("Manual", "Training"):
            return None
        trial, wait = observation["trial"], observation["wait_s"]
        command = {"mode": mode, "trial": trial, "wait_s": wait, "observation": measured}
        controller.observer.annotate(self, measured, calculations={
            "access_resistance_threshold": controller.config.max_access_R,
            "consecutive_success": state.good_count, "trial": trial,
            "wait_period_s": 0.50, "wait_s": wait,
            "pressure_pulse_speed": controller.config.pulse_pressure_duration
                * (2 if mode in ("Classic", "Adaptive") and trial >= 5 else 1),
        })
        if mode in ("Classic", "Adaptive"):
            controller.debug(f"Trial: {trial}")
            return {**command, "kind": "pulse"}
        if mode != "Agent":
            raise AutopatchError(f"Unsupported break_in mode: {mode}")
        agent_observation = {key: value for key, value in measured.items()
                             if key not in ("access_resistance", "capacitance", "calculations")}
        controller.debug(f"Trial: {trial} (Agent mode)")
        action = controller.agenthelper.run_inference(
            observation=agent_observation, is_demo=False)
        controller.info(f"Break-in agent raw action: {action}")
        command["kind"] = "agent"
        if action is None:
            controller.warning("Break-in agent did not return an action; skipping action application.")
            return command
        try:
            values = np.asarray(action, dtype=float).reshape(-1)
        except (TypeError, ValueError) as exc:
            controller.warning(f"Break-in agent action is not numeric: {exc}")
            return command
        if values.size < 1 or not np.isfinite(values[0]):
            controller.warning("Break-in agent ATM action is invalid; skipping action application.")
            return command
        target_atm = bool(float(values[0]) >= 0.5)
        command["target_atm"] = target_atm
        should_zap = False
        if values.size >= 2:
            if np.isfinite(values[1]):
                should_zap = bool(float(values[1]) >= 0.5)
            else:
                controller.warning(f"Break-in agent zap action is invalid: {values[1]}")
        return {**command, "should_zap": should_zap,
                "completion_message": f"Break-in agent decoded action: atm={target_atm}, zap={should_zap}"}

    def act(self, observation=None, state=None):
        self.failure_gate()
        self.success_gate()
        if observation is None:
            return
        command = observation
        controller = self.controller
        mode = controller.config.mode
        if command["mode"] != mode or mode not in ("Classic", "Adaptive", "Agent"):
            self.success_gate(state=state, reset=True)
            return
        if not self.action_gate(command["observation"], state):
            return
        state.trials = command["trial"]
        if command["kind"] == "pulse":
            speed = controller.config.pulse_pressure_duration * (2 if state.trials >= 5 else 1)
            pulse_duration = 1 / speed
            controller.pressure.set_pressure(controller.config.pulse_pressure_break_in)
            state.pulse_active = True
            controller.pressure.set_ATM(atm=False)
            controller.sleep(pulse_duration)
            controller.pressure.set_ATM(atm=True)
            state.pulse_active = False
            controller.sleep(command["wait_s"])
            zap_count = 2 if state.trials % 3 == 0 else 0
            zap_message, wait_after = "zapping", 1
        elif command["kind"] == "agent":
            if "target_atm" in command:
                target_atm = command["target_atm"]
                if bool(controller.pressure.get_ATM()) != target_atm:
                    if not target_atm:
                        controller.pressure.set_pressure(controller.config.pulse_pressure_break_in)
                    controller.pressure.set_ATM(atm=target_atm)
                    controller.info(f"Break-in agent set ATM: {target_atm}")
            zap_count = 1 if command.get("should_zap", False) else 0
            zap_message, wait_after = "zapping (Agent command)", command["wait_s"]
        else:
            raise AutopatchError(f"Unsupported break_in command: {command['kind']}")
        for _ in range(zap_count):
            self.failure_gate()
            self.success_gate()
            if controller.config.mode != mode or not controller.config.zap:
                break
            controller.info(zap_message)
            controller.amplifier.zap()
            controller.sleep(0.5)
        if "completion_message" in command:
            controller.info(command["completion_message"])
        controller.sleep(wait_after)
        self.failure_gate(state, after_action=True, action_mode=mode)

    def action_gate(self, observation=None, state=None):
        return not observation["access_resistance"] <= self.controller.config.max_access_R

    def success_gate(self, observation=None, state=None, *, reset=False):
        super().success_gate()
        if reset:
            state.good_count = 0
            return False
        if observation is None:
            return False
        threshold = self.controller.config.max_access_R
        state.good_count = state.good_count + 1 if observation["access_resistance"] <= threshold else 0
        self.controller.observer.annotate(self, observation, calculations={
            "access_resistance_threshold": threshold, "consecutive_success": state.good_count})
        self.controller.debug(f"Access-R check: {observation['access_resistance']:.2f} Ohm "
                              f"(good_count={state.good_count})")
        return super().success_gate(state.good_count >= 3, state)

    def failure_gate(self, state=None, *, after_action=False, action_mode=None):
        self.controller.abort_if_requested()
        if after_action and action_mode in ("Classic", "Adaptive") and state.trials > 15:
            self.controller.info("Break-in failed")
            raise AutopatchError("Break-in failed")

    def finish(self, *, success=False, observation=None):
        if success:
            self.controller.pressure.set_pressure(0)
            self.controller.info("Successful break-in, Running Avg Access Resistance = "
                                 f"{observation['access_resistance']:.2f}")
            self.goal_event = False
            if not self.awaiting_operator:
                self.complete_success()
        # Baseline finalization deliberately performs no new reset on failure.
