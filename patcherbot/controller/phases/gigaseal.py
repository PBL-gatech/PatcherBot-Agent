"""Implementation of the gigaseal patch phase."""

import time

from dataclasses import dataclass

import numpy as np

from ..errors import AutopatchError

from ..PhaseController import PhaseController




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
        self.begin_observations()
        state = GigasealState()
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
        self.controller.info(f"{self.controller.config.mode}: Attempting to form gigaseal...")
        self.controller.amplifier.auto_fast_compensation()
        self.controller.sleep(1)
        self.controller.daq.setCellMode(True)
        self.controller.sleep(0.1)
        self.controller.info("Collecting baseline resistance...")

        state.num_slope_samples = 5
        state.sample_interval = float(self.controller.config.measurement_speed)

        baseline_observation = self.observe(
            fields=["resistance"], num_measurements=state.num_slope_samples,
            interval=state.sample_interval,
        )
        state.avg_resistance = baseline_observation["resistance"]
        self.controller.observer.annotate(self, baseline_observation, calculations={
            "num_measurements": state.num_slope_samples,
            "sample_interval_s": state.sample_interval,
            "baseline_resistance_mohm": state.avg_resistance,
        })
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

        while not self.controller.abort_requested:
            self.failure_gate(state)
            state.sample_interval = float(self.controller.config.measurement_speed)
            observation = self.observe(
                fields=["resistance"], num_measurements=state.num_slope_samples,
                interval=state.sample_interval,
            )
            self.calculate(observation, state)
            agent_observation = self.observe(fields=[
                "manipulator_position", "pipette_image_xy", "pipette_defocus_um", "stage_positions", "camera_image",
                "resistance", "pressure", "commanded_pressure_mbar", "pressure_atm_state",
            ]) if state.agentPressure else observation
            if state.agentPressure:
                agent_observation = self.calculate(agent_observation, state)
            command = self.decide(agent_observation, state)
            if command is not None:
                self.act(command, state)
                if command.get("check_pressure_release"):
                    self.act({"atm": True, "wait_after": 5})
                    state.sample_interval = float(self.controller.config.measurement_speed)
                    self.failure_gate(state, self.observe(
                        fields=["resistance"], num_measurements=state.num_slope_samples,
                        interval=state.sample_interval,
                    ))
                    state.currPressure = -5
                    self.act({"pressure": state.currPressure, "atm": False})
            self.success_gate(observation["resistance"], state)
        raise AutopatchError("Seal attempt failed: gigaseal criteria not met.")

    def failure_gate(self, state, observation=None):
        """Check the deadline before a loop, or evaluate a supplied pressure-release test."""
        if observation is not None:
            testresistance = observation["resistance"]
            difference = testresistance - state.avg_resistance
            self.controller.observer.annotate(self, observation, calculations={
                "num_measurements": state.num_slope_samples,
                "sample_interval_s": state.sample_interval,
                "reference_resistance_mohm": state.avg_resistance,
                "resistance_difference_mohm": difference,
            })
            self.controller.info(f"Test resistance: {testresistance} MΩ; difference: {difference} MΩ")
            if difference < 0:
                state.bad_cell_count += 1
                if state.bad_cell_count > 5:
                    raise AutopatchError("Bad cell detected")
            return
        if time.time() - state.last_progress_time >= self.controller.config.seal_deadline:
            raise AutopatchError(f"Seal attempt failed: resistance did not improve by at least {self.controller.config.gigaseal_min_delta_R} MegaOhms by the {self.controller.config.seal_deadline} second deadline.")

    def calculate(self, observation, state):
        """Calculate and record progress, slope thresholds, or Agent context."""
        if state.agentPressure and "pressure_atm_state" in observation:
            observation["observations_since_last_action"] = state.observations_since_last_action
            self.controller.observer.annotate(self, observation, calculations={
                "observations_since_last_action": state.observations_since_last_action,
            })
            return {key: value for key, value in observation.items() if key != "calculations"}
        previous_resistance = state.avg_resistance
        delta_resistance = observation["resistance"] - previous_resistance
        state.avg_resistance = observation["resistance"]
        state.rate_mohm_per_sec = delta_resistance / (state.num_slope_samples * state.sample_interval)
        minimum_delta = self.controller.config.gigaseal_min_delta_R
        if delta_resistance >= minimum_delta:
            state.last_progress_time = time.time()
        calculations = {
            "num_measurements": state.num_slope_samples,
            "sample_interval_s": state.sample_interval,
            "previous_resistance_mohm": previous_resistance,
            "delta_resistance_mohm": delta_resistance,
            "rate_mohm_per_sec": state.rate_mohm_per_sec,
            "gigaseal_min_delta_R": minimum_delta,
        }
        if state.autoPressure or state.adaptivePressure:
            target_resistance = self.controller.config.gigaseal_R
            increase_slope_gate = self.controller.config.increase_slope_gate
            constant_slope_gate = self.controller.config.constant_slope_gate
            decrease_slope_gate = self.controller.config.decrease_slope_gate
            state.increase_thresh = target_resistance / increase_slope_gate
            state.constant_thresh = target_resistance / constant_slope_gate
            state.decrease_thresh = target_resistance / decrease_slope_gate
            calculations.update(
                gigaseal_R=target_resistance,
                increase_slope_gate=increase_slope_gate,
                constant_slope_gate=constant_slope_gate,
                decrease_slope_gate=decrease_slope_gate,
                increase_threshold_mohm_per_sec=state.increase_thresh,
                constant_threshold_mohm_per_sec=state.constant_thresh,
                decrease_threshold_mohm_per_sec=state.decrease_thresh,
            )
        self.controller.observer.annotate(self, observation, calculations=calculations)
        return observation

    def decide(self, observation, state):
        """Choose an Agent or Classic/Adaptive command from calculated inputs."""

        # region Agent mode
        if state.agentPressure:
            self.controller.info(
                "Gigaseal agent observation collected: "
                f"resistance={float(observation['resistance']):.3f} MΩ, "
                f"actual_pressure={float(observation['pressure']):.3f} mbar, "
                f"setpoint={float(observation['commanded_pressure_mbar']):.3f} mbar, "
                f"atm={bool(observation['pressure_atm_state'])}, "
                f"observations_since_last_action={int(observation['observations_since_last_action'])}"
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
