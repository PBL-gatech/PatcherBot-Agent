"""Implementation of the gigaseal patch phase."""

import collections
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
    agent_resistance_input_width: int = 0
    agent_resistance_history: object = None
    num_slope_samples: int = 5
    sample_interval: float = 0.0
    avg_resistance: float = 0.0
    rate_mohm_per_sec: float = 0.0
    consecutive_success: int = 0
    currPressure: float = 0.0
    prevpressure: float = 0.0
    speed: float = 1.0
    bad_cell_count: int = 0
    max_pressure: float = 0.0
    last_agent_action: object = None
    observations_since_last_action: int = 0


class GigasealPhase(PhaseController):
    def run(self, cell=None):
        """requires **three consecutive**
        averaged-resistance windows ≥ target to declare success, reducing
        false positives from transient spikes.
        """
        state = GigasealState()
        state.autoPressure = (self.controller.config.mode == 'Classic')
        state.adaptivePressure = (self.controller.config.mode == 'Adaptive')
        state.agentPressure = (self.controller.config.mode == 'Agent')
        state.agent_resistance_input_width = 0
        state.agent_resistance_history = collections.deque(maxlen=1)
        if state.agentPressure:
            self.controller.info("Agent gigaseal mode detected; preparing gigaseal policy.")
            self.controller.agenthelper.prepare_model("gigaseal")
            state.agent_resistance_input_width = self.controller._agent_resistance_input_width()
            state.agent_resistance_history = collections.deque(
                maxlen=max(1, state.agent_resistance_input_width)
            )
        self.controller.info(f"{self.controller.config.mode}: Attempting to form gigaseal...")
        self.controller.amplifier.auto_fast_compensation()
        self.controller.sleep(1)
        self.controller.daq.setCellMode(True)
        self.controller.sleep(0.1)
        self.controller.info("Collecting baseline resistance...")

        state.num_slope_samples = 5
        state.sample_interval = float(self.controller.config.measurement_speed)

        state.avg_resistance = self.controller.resistanceRamp(
            num_measurements=state.num_slope_samples,
            interval=state.sample_interval,
        )
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

        holding_switched = False
        last_progress_time = time.time()
        state.last_agent_action = None
        state.observations_since_last_action = 0

        while not self.controller.abort_requested:
            # Deadline check
            if time.time() - last_progress_time >= self.controller.config.seal_deadline:
                raise AutopatchError(f"Seal attempt failed: resistance did not improve by at least {self.controller.config.gigaseal_min_delta_R} MegaOhms by the {self.controller.config.seal_deadline} second deadline.")

            prev_resistance = state.avg_resistance
            state.avg_resistance = self.controller.resistanceRamp(
                num_measurements=state.num_slope_samples,
                interval=state.sample_interval,
            )

            delta_resistance = state.avg_resistance - prev_resistance
            state.rate_mohm_per_sec = delta_resistance / (state.num_slope_samples * state.sample_interval)

            if delta_resistance >= self.controller.config.gigaseal_min_delta_R:
                last_progress_time = time.time()

            # ---------------------- auto-pressure logic ----------------------
            self.act(state=state)
            # ---------------------------------------------------------------

            # Holding potential switch
            if state.avg_resistance >= self.controller.config.gigaseal_R / self.controller.config.hold_switch and not holding_switched:
                self.controller.amplifier.set_holding(self.controller.protocol_config.vclamp_hold)
                self.controller.amplifier.switch_holding(True)
                holding_switched = True

            # Success check with consecutive-hit filter
            goal_reached = self.success_gate(state.avg_resistance, state)

            if goal_reached:
                self.controller.pressure.set_ATM(atm=True)
                self.controller.info("Seal successful!")
                if self.controller.config.mode == "Training":
                    self.controller.info("Training mode: goal condition reached. Click Success or Abort to finish.")
                    while True:
                        self.controller.sleep(0.1)
                self.controller.success_requested = True
                self.controller.success_if_requested()

        # Abort request came in
        raise AutopatchError("Seal attempt failed: gigaseal criteria not met.")


    def act(self, observation=None, state=None):
        """Apply this iteration's mode controls; acquire agent input at its original point."""
        if state.agentPressure:
            observation = self.observe(include_pressure_state=True)
            self.controller._attach_agent_resistance_input(
                observation,
                state.agent_resistance_history,
                state.agent_resistance_input_width,
            )
            observation["observations_since_last_action"] = np.asarray(
                [state.observations_since_last_action],
                dtype=np.float32,
            )
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
                        self.controller.pressure.set_pressure(commanded_pressure)
                        self.controller.pressure.set_ATM(atm=target_atm)
                        state.observations_since_last_action = 0
                        state.last_agent_action = current_agent_action
                    else:
                        state.observations_since_last_action += 1
        elif state.autoPressure:
            # adjust currPressure by ±5 based on rate_mohm_per_sec, speed, etc.
            increase_gate = self.controller.config.increase_slope_gate
            constant_gate = self.controller.config.constant_slope_gate
            decrease_gate = self.controller.config.decrease_slope_gate

            increase_thresh = self.controller.config.gigaseal_R / increase_gate
            constant_thresh = self.controller.config.gigaseal_R / constant_gate
            decrease_thresh = self.controller.config.gigaseal_R / decrease_gate

            if state.rate_mohm_per_sec < increase_thresh:
                state.currPressure -= 5; state.speed = 3; state.max_pressure = self.controller.config.pressure_ramp_max
            elif state.rate_mohm_per_sec <= constant_thresh:
                state.speed = 1  # maintain
            elif state.rate_mohm_per_sec <= decrease_thresh:
                state.max_pressure = self.controller.config.pressure_ramp_max; state.currPressure += 5; state.speed = 3

            state.currPressure = min(state.currPressure, -5.0)
            state.currPressure = max(state.currPressure, self.controller.config.pressure_ramp_max)

            if state.currPressure != state.prevpressure:
                self.controller.pressure.set_pressure(state.currPressure)
                state.prevpressure = state.currPressure
                self.controller.sleep(5 / state.speed)

            if state.currPressure <= state.max_pressure:
                self.controller.pressure.set_ATM(True)
                self.controller.sleep(5)
                testresistance = self.controller.resistanceRamp(
                    num_measurements=state.num_slope_samples,
                    interval=state.sample_interval,
                )
                difference = testresistance - state.avg_resistance
                self.controller.info(f"Test resistance: {testresistance} MΩ; difference: {difference} MΩ")
                if difference < 0:
                    state.bad_cell_count += 1
                    if state.bad_cell_count > 5:
                        raise AutopatchError("Bad cell detected")

                state.currPressure = -5
                self.controller.pressure.set_pressure(state.currPressure)
                self.controller.pressure.set_ATM(atm=False)
        elif state.adaptivePressure:
            # adjust currPressure by ±5 based on rate_mohm_per_sec, speed, etc.
            increase_gate = self.controller.config.increase_slope_gate
            constant_gate = self.controller.config.constant_slope_gate
            decrease_gate = self.controller.config.decrease_slope_gate

            increase_thresh = self.controller.config.gigaseal_R / increase_gate
            constant_thresh = self.controller.config.gigaseal_R / constant_gate
            decrease_thresh = self.controller.config.gigaseal_R / decrease_gate

            if state.rate_mohm_per_sec < increase_thresh:
                state.currPressure -= 5; state.speed = 3; state.max_pressure = self.controller.config.pressure_ramp_max
            elif state.rate_mohm_per_sec <= constant_thresh:
                state.speed = 1  # maintain
            elif state.rate_mohm_per_sec <= decrease_thresh:
                state.max_pressure = self.controller.config.pressure_ramp_max; state.currPressure += 5; state.speed = 3

            state.currPressure = min(state.currPressure, -5.0)
            state.currPressure = max(state.currPressure, self.controller.config.pressure_ramp_max)

            if state.currPressure != state.prevpressure:
                self.controller.pressure.set_pressure(state.currPressure)
                state.prevpressure = state.currPressure
                self.controller.sleep(5 / state.speed)

            if state.currPressure <= state.max_pressure:
                self.controller.pressure.set_ATM(True)
                self.controller.sleep(5)
                testresistance = self.controller.resistanceRamp(
                    num_measurements=state.num_slope_samples,
                    interval=state.sample_interval,
                )
                difference = testresistance - state.avg_resistance
                self.controller.info(f'Test resistance: {testresistance} MΩ; difference: {difference} MΩ')
                if difference < 0:
                    state.bad_cell_count += 1
                    if state.bad_cell_count > 5:
                        raise AutopatchError('Bad cell detected')

                state.currPressure = -5
                self.controller.pressure.set_pressure(state.currPressure)
                self.controller.pressure.set_ATM(atm=False)


    def success_gate(self, observation, state):
        """Count consecutive qualifying resistance windows without acquiring a sample."""
        if observation >= self.controller.config.gigaseal_R:
            state.consecutive_success += 1
        else:
            state.consecutive_success = 0
        return state.consecutive_success >= 3
