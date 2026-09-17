"""Implementation of the break_in patch phase."""

from collections import deque
from dataclasses import dataclass

import numpy as np

from ..errors import AutopatchError

from ..PhaseController import PhaseController


@dataclass
class BreakInState:
    """Working values scoped to one phase run."""
    mode: str = "Manual"
    trials: int = 0
    speed: float = 0.0
    good_count: int = 0
    threshold_AR: float = 0.0
    wait_period: float = 0.50
    wait: float = 0.0


class BreakInPhase(PhaseController):
    def run(self, cell=None):
        """Observe once per loop, then check gates, calculate, decide, and act.

        Every loop reads all required signals before deciding. Three consecutive
        qualifying access readings finish the attempt; qualifying readings never
        trigger an action. The next observation captures the previous action's result.
        Training only observes and logs until manual Success or Abort.
        """
        state = BreakInState()
        self.prepare(state)
        fields = ["access_resistance", "resistance", "capacitance"]
        if state.mode == "Agent":
            fields += [
                "pipette_positions", "stage_positions", "camera_image",
                "pressure", "commanded_pressure_mbar", "pressure_atm_state",
            ]
        sampling = {"resistance": {"num_measurements": 5, "interval": 0.200}}
        self.controller.info(
            f"Target Resistance: {self.controller.config.max_cell_R}; "
            f"Target Capacitance: {self.controller.config.min_cell_C}, "
            f"Target Access Resistance: {self.controller.config.max_access_R}")

        while True:
            observation = self.observe(
                fields=fields, sampling=sampling, raw_measurements=True,
            )
            self.controller.info(
                f"Pre-action Resistance: {observation['resistance']}; "
                f"Membrane Capacitance: {observation['capacitance']}, "
                f"Access Resistance: {observation['access_resistance']}")
            if self.success_gate(observation, state):
                return
            if not self.action_gate(observation, state):
                continue
            observation = self.calculate(observation, state)
            command = self.decide(observation, state)
            if command is None:
                continue
            self.act(command)
            self.failure_gate(state)



    def prepare(self, state):
        """Prepare the rig and agent state for this break-in attempt."""
        self.observation_deck = deque(maxlen=30)
        self.observation_windows = {}
        state.mode = self.controller.config.mode
        if state.mode == "Training":
            self.controller.info("Training mode: observing only. Click Success or Abort to finish.")
            return

        # ---------- initial setup ----------
        self.controller.daq.setCellMode(True)
        if state.mode == "Agent":
            self.controller.info("Agent break-in mode detected; preparing break-in policy.")
            self.controller.agenthelper.prepare_model("break_in")
            width = self.controller.agenthelper.observation_input_width("resistance_input")
            self.observation_deck = deque(maxlen=max(30, width + 1))
            if width > 0:
                self.observation_windows["resistance_input"] = {
                    "field": "resistance",
                    "width": width,
                    "predicate": lambda sample: not float(np.asarray(
                        sample.get("access_resistance", np.nan),
                    ).reshape(-1)[0]) <= state.threshold_AR,
                    "finite_only": True,
                    "fill_value": 0.0,
                }
        self.controller.info(f"{self.controller.config.mode}: Attempting Break in...")
        self.controller.sleep(3)
        self.controller.pressure.set_pressure(self.controller.config.pulse_pressure_break_in)
        self.controller.amplifier.set_zap_duration(25 * 1e-6)
        state.speed = self.controller.config.pulse_pressure_duration
        state.threshold_AR = self.controller.config.max_access_R

    def action_gate(self, observation, state):
        """Allow an action only while access resistance has not qualified."""
        return not observation["access_resistance"] <= state.threshold_AR

    def calculate(self, observation, state):
        """Prepare trial timing and model values without sampling or acting."""
        if state.mode not in ("Classic", "Adaptive", "Agent"):
            return observation
        state.trials += 1
        state.wait = state.wait_period * (1 + state.trials / 2)
        if state.mode in ("Classic", "Adaptive"):
            if state.trials % 5 == 0:
                state.speed = 2 * self.controller.config.pulse_pressure_duration
            return observation

        agent_observation = {
            key: value for key, value in observation.items()
            if key not in ("access_resistance", "capacitance")
        }
        agent_observation["resistance"] = np.asarray(
            observation["resistance"], dtype=np.float32,
        ).reshape(1)
        return agent_observation

    def decide(self, observation, state):
        """Translate the calculated observation and mode into an action command."""
        if state.mode not in ("Classic", "Adaptive", "Agent"):
            return None
        if state.mode in ("Classic", "Adaptive"):
            self.controller.debug(f"Trial: {state.trials}")
            return {
                "pulse_duration": 1 / state.speed,
                "wait_before_zap": state.wait,
                "zap_count": 2 if self.controller.config.zap and state.trials % 3 == 0 else 0,
                "zap_message": "zapping",
                "wait_after": 1,
            }

        self.controller.debug(f"Trial: {state.trials} (Agent mode)")
        action = self.controller.agenthelper.run_inference(observation=observation, is_demo=False)
        self.controller.info(f"Break-in agent raw action: {action}")
        command = {"wait_after": state.wait}
        if action is None:
            self.controller.warning("Break-in agent did not return an action; skipping action application for this iteration.")
            return command
        try:
            action_array = np.asarray(action, dtype=float).reshape(-1)
        except (TypeError, ValueError) as exc:
            self.controller.warning(f"Break-in agent action could not be converted to a numeric array: {exc}")
            return command
        if action_array.size < 1:
            self.controller.warning(
                f"Break-in agent action must have at least 1 value; received shape {action_array.shape}."
            )
            return command
        if not np.isfinite(action_array[0]):
            self.controller.warning(f"Break-in agent ATM action is invalid: {action_array[0]}")
            return command

        target_atm = bool(float(action_array[0]) >= 0.5)
        current_atm = bool(observation["pressure_atm_state"][0])
        if current_atm != target_atm:
            command["set_atm"] = target_atm
        should_zap = False
        if action_array.size >= 2:
            if np.isfinite(action_array[1]):
                should_zap = bool(float(action_array[1]) >= 0.5)
            else:
                self.controller.warning(f"Break-in agent zap action is invalid: {action_array[1]}")
        command.update(
            zap_count=1 if should_zap and self.controller.config.zap else 0,
            zap_message="zapping (Agent command)",
            completion_message=f"Break-in agent decoded action: atm={target_atm}, zap={should_zap}",
        )
        return command

    def act(self, command):
        """Execute the decided command without observations, inference, or mode checks."""
        if "pulse_duration" in command:
            self.controller.pressure.set_ATM(atm=False)
            self.controller.sleep(command["pulse_duration"])
            self.controller.pressure.set_ATM(atm=True)
            self.controller.sleep(command["wait_before_zap"])
        elif "set_atm" in command:
            self.controller.pressure.set_ATM(atm=command["set_atm"])
            self.controller.info(f"Break-in agent set ATM: {command['set_atm']}")
        if command.get("zap_count", 0):
            self.controller.info(command["zap_message"])
            for _ in range(command["zap_count"]):
                self.controller.amplifier.zap()
                self.controller.sleep(0.5)
        if "completion_message" in command:
            self.controller.info(command["completion_message"])
        self.controller.sleep(command["wait_after"])


    def success_gate(self, observation, state):
        """Count qualifying readings and complete success; Training never auto-completes."""
        if state.mode == "Training":
            return False
        if observation["access_resistance"] <= state.threshold_AR:
            state.good_count += 1
        else:
            state.good_count = 0
        self.controller.debug(
            f"Access-R check: {observation['access_resistance']:.2f} Ohm "
            f"(good_count={state.good_count})")
        if state.good_count < 3:
            return False
        self.controller.pressure.set_pressure(0)
        self.controller.info("Successful break-in, Running Avg Access Resistance = "
                f"{observation['access_resistance']:.2f}")
        self.controller.success_requested = True
        self.controller.success_if_requested()
        return True


    def failure_gate(self, state):
        """Log and raise when the existing Classic/Adaptive trial limit is exceeded."""
        if state.mode in ("Classic", "Adaptive") and state.trials > 15:
            self.controller.info("Break-in failed")
            raise AutopatchError("Break-in failed")
