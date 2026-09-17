"""Implementation of the gigaseal patch phase."""

import collections
import time

from dataclasses import dataclass

import numpy as np

from ..errors import AutopatchError

from ..PhaseController import PhaseController


OBSERVATION_HISTORY_SIZE = 30


@dataclass
class GigasealState:
    """Working values scoped to one phase run."""
    autoPressure: bool = False
    adaptivePressure: bool = False
    agentPressure: bool = False
    num_slope_samples: int = 5
    sample_interval: float = 0.0
    avg_resistance: float = 0.0
    rate_mohm_per_sec: float = 0.0
    increase_thresh: float = 0.0
    constant_thresh: float = 0.0
    decrease_thresh: float = 0.0
    consecutive_success: int = 0
    currPressure: float = 0.0
    prevpressure: float = 0.0
    speed: float = 1.0
    bad_cell_count: int = 0
    max_pressure: float = 0.0
    last_agent_action: object = None
    observations_since_last_action: int = 0
    holding_switched: bool = False
    last_progress_time: float = 0.0

class GigasealPhase(PhaseController):
    def run(self, cell=None):
        """Coordinate the original sampling, pressure, holding, and success order."""
        state = GigasealState()
        self.prepare(state)
        sampling = {"resistance": {
            "num_measurements": state.num_slope_samples,
            "interval": state.sample_interval,
        }}
        while not self.controller.abort_requested:
            self.failure_gate(state)
            observation = self.observe(
                fields=["resistance"], sampling=sampling, raw_measurements=True,
            )
            self.calculate(observation, state)
            agent_observation = self.observe(include_pressure_state=True) if state.agentPressure else observation
            if state.agentPressure:
                self.calculate(agent_observation, state)
            command = self.decide(agent_observation, state)
            if command is not None:
                self.act(command, state)
                if command.get("check_pressure_release"):
                    self.act({"atm": True, "wait_after": 5})
                    self.failure_gate(state, self.observe(
                        fields=["resistance"], sampling=sampling, raw_measurements=True,
                    ))
                    state.currPressure = -5
                    self.act({"pressure": state.currPressure, "atm": False})
            self.success_gate(observation["resistance"], state)
        raise AutopatchError("Seal attempt failed: gigaseal criteria not met.")

    def prepare(self, state):
        """Initialize the rig, baseline, mode, and per-attempt state in order."""
        state.autoPressure = (self.controller.config.mode == 'Classic')
        state.adaptivePressure = (self.controller.config.mode == 'Adaptive')
        state.agentPressure = (self.controller.config.mode == 'Agent')
        self.observation_windows = {}
        width = 0
        if state.agentPressure:
            self.controller.info("Agent gigaseal mode detected; preparing gigaseal policy.")
            self.controller.agenthelper.prepare_model("gigaseal")
            width = self.controller.agenthelper.observation_input_width("resistance_input")
            if width:
                self.observation_windows["resistance_input"] = {
                    "field": "resistance", "width": width,
                    "predicate": lambda sample: np.isfinite(
                        np.asarray(sample.get("pressure_atm_state", np.nan), dtype=float)
                    ).all(),
                    "finite_only": True, "fill_value": 0.0,
                }
        # Each Agent input follows a separate averaged loop observation.
        # Retain N prior inputs plus the current one, including interleaving.
        self.observation_deck = collections.deque(maxlen=max(
            OBSERVATION_HISTORY_SIZE, 2 * width + 1,
        ))
        self.controller.info(f"{self.controller.config.mode}: Attempting to form gigaseal...")
        self.controller.amplifier.auto_fast_compensation()
        self.controller.sleep(1)
        self.controller.daq.setCellMode(True)
        self.controller.sleep(0.1)
        self.controller.info("Collecting baseline resistance...")

        state.num_slope_samples = 5
        state.sample_interval = float(self.controller.config.measurement_speed)

        state.avg_resistance = self.observe(
            fields=["resistance"],
            sampling={"resistance": {
                "num_measurements": state.num_slope_samples,
                "interval": state.sample_interval,
            }},
            raw_measurements=True,
        )["resistance"]
        state.consecutive_success = 0

        self.controller.pressure.set_ATM(atm=True)

        self.controller.sleep(3)

        if state.autoPressure:
            state.currPressure = -5
            self.controller.pressure.set_pressure(state.currPressure)
            self.controller.pressure.set_ATM(atm=False)
            state.prevpressure = state.currPressure
            state.speed = 1
            state.bad_cell_count = 0
            # this is already negative, e.g. -30 mbar
            state.max_pressure = self.controller.config.pressure_ramp_max
        elif state.adaptivePressure:
            state.currPressure = -5
            self.controller.pressure.set_pressure(state.currPressure)
            self.controller.pressure.set_ATM(atm=False)
            state.prevpressure = state.currPressure
            state.speed = 1
            state.bad_cell_count = 0
            # this is already negative, e.g. -30 mbar
            state.max_pressure = self.controller.config.pressure_ramp_max

        state.holding_switched = False
        state.last_progress_time = time.time()
        state.last_agent_action = None
        state.observations_since_last_action = 0

    def failure_gate(self, state, observation=None):
        """Check the deadline before a loop, or evaluate a supplied pressure-release test."""
        if observation is not None:
            testresistance = observation["resistance"]
            difference = testresistance - state.avg_resistance
            self.controller.info(f"Test resistance: {testresistance} MΩ; difference: {difference} MΩ")
            if difference < 0:
                state.bad_cell_count += 1
                if state.bad_cell_count > 5:
                    raise AutopatchError("Bad cell detected")
            return
        if time.time() - state.last_progress_time >= self.controller.config.seal_deadline:
            raise AutopatchError(f"Seal attempt failed: resistance did not improve by at least {self.controller.config.gigaseal_min_delta_R} MegaOhms by the {self.controller.config.seal_deadline} second deadline.")

    def calculate(self, observation, state):
        """Calculate progress, slope thresholds, or Agent decision context."""
        if state.agentPressure and "pressure_atm_state" in observation:
            observation["observations_since_last_action"] = np.asarray(
                [state.observations_since_last_action], dtype=np.float32,
            )
            return observation
        delta_resistance = observation["resistance"] - state.avg_resistance
        state.avg_resistance = observation["resistance"]
        state.rate_mohm_per_sec = delta_resistance / (state.num_slope_samples * state.sample_interval)
        if delta_resistance >= self.controller.config.gigaseal_min_delta_R:
            state.last_progress_time = time.time()
        if state.autoPressure or state.adaptivePressure:
            state.increase_thresh = self.controller.config.gigaseal_R / self.controller.config.increase_slope_gate
            state.constant_thresh = self.controller.config.gigaseal_R / self.controller.config.constant_slope_gate
            state.decrease_thresh = self.controller.config.gigaseal_R / self.controller.config.decrease_slope_gate
        return observation

    def decide(self, observation, state):
        """Choose an Agent or Classic/Adaptive command from calculated inputs."""

        # region Agent mode
        if state.agentPressure:
            self.controller.info(
                "Gigaseal agent observation collected: "
                f"resistance={float(observation['resistance'][0]):.3f} MΩ, "
                f"actual_pressure={float(observation['pressure'][0]):.3f} mbar, "
                f"setpoint={float(observation['commanded_pressure_mbar'][0]):.3f} mbar, "
                f"atm={bool(observation['pressure_atm_state'][0])}, "
                f"observations_since_last_action={int(observation['observations_since_last_action'][0])}"
            )
            action = self.controller.agenthelper.run_inference(observation=observation, is_demo=False)
            self.controller.info(f"Gigaseal agent raw action: {action}")
            if action is None:
                self.controller.warning("Gigaseal agent did not return an action; skipping pressure update for this iteration.")
            else:
                action_array = np.asarray(action).reshape(-1)
                if action_array.size < 1:
                    self.controller.warning(
                        f"Gigaseal agent action must have at least 1 value; received shape {action_array.shape}."
                    )
                else:
                    first_action_value = float(action_array[0])
                    if action_array.size == 1:
                        target_atm = bool(first_action_value >= 0.0)
                        if target_atm:
                            commanded_pressure = float(self.controller.pressure.get_pressure())
                        else:
                            commanded_pressure = float(
                                np.clip(first_action_value, float(self.controller.config.pressure_ramp_max), -5.0)
                            )
                    else:
                        commanded_pressure = float(
                            np.clip(first_action_value, float(self.controller.config.pressure_ramp_max), -5.0)
                        )
                        target_atm = bool(float(action_array[1]) >= 0.5)
                    self.controller.info(
                        "Gigaseal agent decoded action: "
                        f"commanded_pressure={commanded_pressure:.3f} mbar, atm={target_atm}"
                    )
                    current_agent_action = (commanded_pressure, target_atm)
                    if current_agent_action != state.last_agent_action:
                        return {"pressure": commanded_pressure, "atm": target_atm,
                                "agent_action": current_agent_action}
                    state.observations_since_last_action += 1
            return None
        # endregion

        # region Manual / Training / inactive modes
        if not (state.autoPressure or state.adaptivePressure):
            return None
        # endregion

        # region Classic / Adaptive modes
        if state.rate_mohm_per_sec < state.increase_thresh:
            state.currPressure -= 5; state.speed = 3; state.max_pressure = self.controller.config.pressure_ramp_max
        elif state.rate_mohm_per_sec <= state.constant_thresh:
            state.speed = 1  # maintain
        elif state.rate_mohm_per_sec <= state.decrease_thresh:
            state.max_pressure = self.controller.config.pressure_ramp_max; state.currPressure += 5; state.speed = 3

        state.currPressure = min(state.currPressure, -5.0)
        state.currPressure = max(state.currPressure, self.controller.config.pressure_ramp_max)
        command = {"check_pressure_release": state.currPressure <= state.max_pressure}
        if state.currPressure != state.prevpressure:
            command.update(pressure=state.currPressure, wait_after=5 / state.speed, remember_pressure=True)
        return command
        # endregion

    def act(self, command, state=None):
        """Execute device writes and waits, recording applied commands without acquiring signals."""
        if "pressure" in command:
            self.controller.pressure.set_pressure(command["pressure"])
            if command.get("remember_pressure"):
                state.prevpressure = command["pressure"]
        if "atm" in command:
            self.controller.pressure.set_ATM(atm=command["atm"])
        if "agent_action" in command:
            state.observations_since_last_action = 0
            state.last_agent_action = command["agent_action"]
        if "holding" in command:
            self.controller.amplifier.set_holding(command["holding"])
            self.controller.amplifier.switch_holding(True)
            state.holding_switched = True
        if "wait_after" in command:
            self.controller.sleep(command["wait_after"])

    def success_gate(self, observation, state):
        """Apply the holding threshold, count qualifying windows, and complete success."""
        if observation >= self.controller.config.gigaseal_R / self.controller.config.hold_switch and not state.holding_switched:
            self.act({"holding": self.controller.protocol_config.vclamp_hold}, state)
        if observation >= self.controller.config.gigaseal_R:
            state.consecutive_success += 1
        else:
            state.consecutive_success = 0
        if state.consecutive_success < 3:
            return False
        self.act({"atm": True})
        self.controller.info("Seal successful!")
        if self.controller.config.mode == "Training":
            self.controller.info("Training mode: goal condition reached. Click Success or Abort to finish.")
            self.wait_for_manual_completion()
        self.complete_success()
        return True
