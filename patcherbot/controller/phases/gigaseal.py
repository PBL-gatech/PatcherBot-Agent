"""Implementation of the gigaseal patch phase."""

import time

from dataclasses import dataclass

import numpy as np

from ..errors import AutopatchError

from ..PhaseController import PhaseController




@dataclass
class GigasealState:
    stage: str = "main"
    num_slope_samples: int = 5
    avg_resistance: float = 0.0
    sample_interval: float = 0.0
    main_window: object = None
    cycle_mode: object = None  # acquisition provenance, never the live routing owner
    consecutive_success: int = 0
    currPressure: float = 0.0
    prevpressure: float = 0.0
    speed: float = 1.0
    bad_cell_count: int = 0
    last_agent_action: object = None
    observations_since_last_action: int = 0
    holding_switched: bool = False
    last_progress_time: float = 0.0
    ramp_initialized: bool = False


class GigasealPhase(PhaseController):
    def run(self, cell=None):
        state = None
        try:
            state = self.prepare(cell)
            while state.stage != "finished":
                observation = self.observe(state=state)
                calculated = self.calculate(observation, state)
                command = self.decide(calculated, state)
                self.act(command, state)
        finally:
            self.finish(state)

    def prepare(self, cell=None):
        controller = self.controller
        self.begin_observations()
        self.observation_windows = {}
        state = GigasealState()
        self._agent_model_prepared = False
        if controller.config.mode == "Agent":
            self._prepare_agent_window(reset=True)
        controller.info(f"{controller.config.mode}: Attempting to form gigaseal...")
        controller.amplifier.auto_fast_compensation()
        controller.sleep(1)
        controller.daq.setCellMode(True)
        controller.sleep(0.1)
        controller.info("Collecting baseline resistance...")
        state.sample_interval = float(controller.config.measurement_speed)
        baseline = self.observe(fields=["resistance"],
            num_measurements=state.num_slope_samples, interval=state.sample_interval)
        state.avg_resistance = baseline["resistance"]
        controller.observer.annotate(self, baseline, calculations={
            "num_measurements": state.num_slope_samples,
            "sample_interval_s": state.sample_interval,
            "baseline_resistance_mohm": state.avg_resistance})
        controller.pressure.set_ATM(atm=True)
        controller.sleep(3)
        if controller.config.mode in ("Classic", "Adaptive"):
            self.act({"kind": "initialize", "mode": controller.config.mode}, state)
        state.last_progress_time = time.time()
        return state

    def _prepare_agent_window(self, *, reset=False):
        helper = self.controller.agenthelper
        if (reset or not getattr(self, "_agent_model_prepared", False)
                or getattr(helper, "model_type", None) != "gigaseal" or helper.agent is None):
            helper.prepare_model("gigaseal")
            self._agent_model_prepared = True
        width = helper.observation_input_width("resistance_input")
        self.observation_windows = {}
        if width:
            self.observation_windows["resistance_input"] = {
                "field": "resistance", "width": width,
                "predicate": lambda sample: np.isfinite(np.asarray(
                    sample.get("pressure_atm_state", np.nan), dtype=float)).all(),
                "finite_only": True, "fill_value": 0.0}

    def observe(self, *, fields=None, num_measurements=None, interval=None,
                evidence=None, state=None):
        self.failure_gate(state if state is not None and state.stage == "main" else None)
        self.success_gate()
        if state is not None:
            if state.stage == "main":
                state.cycle_mode = self.controller.config.mode
            elif state.cycle_mode != self.controller.config.mode:
                return {"stage": "invalid", "mode": state.cycle_mode}
        if state is None:
            result = super().observe(fields=fields, num_measurements=num_measurements,
                                     interval=interval, evidence=evidence)
        elif state.stage == "agent":
            if self.controller.config.mode != "Agent":
                return {"stage": "route", "window": state.main_window, "mode": state.cycle_mode}
            self._prepare_agent_window()
            sample = super().observe(fields=["manipulator_position", "pipette_image_xy",
                "pipette_defocus_um", "stage_positions", "camera_image", "resistance",
                "pressure", "commanded_pressure_mbar", "pressure_atm_state"])
            result = {"stage": "agent", "sample": sample, "mode": state.cycle_mode}
        else:
            state.sample_interval = float(self.controller.config.measurement_speed)
            sample = super().observe(fields=["resistance"],
                num_measurements=state.num_slope_samples, interval=state.sample_interval)
            result = {"stage": state.stage, "sample": sample,
                      "interval": state.sample_interval, "mode": state.cycle_mode}
        self.failure_gate()
        self.success_gate()
        return result

    def calculate(self, observation=None, state=None):
        self.failure_gate()
        self.success_gate()
        stage = observation["stage"]
        if stage == "invalid" or observation.get("mode") != self.controller.config.mode:
            return {"stage": "invalid", "mode": observation.get("mode")}
        if stage == "agent":
            sample = observation["sample"]
            count = state.observations_since_last_action
            self.controller.observer.annotate(self, sample, calculations={
                "observations_since_last_action": count})
            return {"stage": stage, "mode": observation["mode"], "sample": {
                **{key: value for key, value in sample.items() if key != "calculations"},
                "observations_since_last_action": count}}
        if stage == "release":
            sample = observation["sample"]
            difference = sample["resistance"] - state.avg_resistance
            self.controller.observer.annotate(self, sample, calculations={
                "num_measurements": state.num_slope_samples,
                "sample_interval_s": observation["interval"],
                "reference_resistance_mohm": state.avg_resistance,
                "resistance_difference_mohm": difference})
            return {"stage": stage, "mode": observation["mode"],
                    "resistance": sample["resistance"], "difference": difference}
        if stage == "main":
            sample = observation["sample"]
            previous = state.avg_resistance
            delta = sample["resistance"] - previous
            state.avg_resistance = sample["resistance"]
            minimum = self.controller.config.gigaseal_min_delta_R
            if delta >= minimum:
                state.last_progress_time = time.time()
            state.main_window = {"resistance": sample["resistance"],
                "previous_resistance": previous, "interval": observation["interval"],
                "mode": observation["mode"]}
            rate = delta / (state.num_slope_samples * observation["interval"])
            calculations = {
                "num_measurements": state.num_slope_samples,
                "sample_interval_s": observation["interval"],
                "previous_resistance_mohm": previous, "delta_resistance_mohm": delta,
                "rate_mohm_per_sec": rate, "gigaseal_min_delta_R": minimum}
            if self.controller.config.mode in ("Classic", "Adaptive"):
                config = self.controller.config
                target = config.gigaseal_R
                calculations.update(gigaseal_R=target,
                    increase_slope_gate=config.increase_slope_gate,
                    constant_slope_gate=config.constant_slope_gate,
                    decrease_slope_gate=config.decrease_slope_gate,
                    increase_threshold_mohm_per_sec=target / config.increase_slope_gate,
                    constant_threshold_mohm_per_sec=target / config.constant_slope_gate,
                    decrease_threshold_mohm_per_sec=target / config.decrease_slope_gate)
            self.controller.observer.annotate(self, sample, calculations=calculations)
        window = state.main_window
        return {"stage": stage, "mode": observation["mode"],
                "rate": (window["resistance"] -
            window["previous_resistance"]) / (state.num_slope_samples * window["interval"])}

    def decide(self, observation=None, state=None):
        self.failure_gate()
        self.success_gate()
        mode = self.controller.config.mode
        if mode not in ("Manual", "Training", "Classic", "Adaptive", "Agent"):
            raise AutopatchError(f"Unsupported gigaseal mode: {mode}")
        if self.awaiting_operator:
            return None
        stage = observation["stage"]
        if stage == "invalid" or observation.get("mode") != mode:
            return {"kind": "invalidate", "mode": mode}
        if stage == "release":
            self.failure_gate(state, observation)
            return {"kind": "reset" if mode in ("Classic", "Adaptive") else "done",
                    "mode": mode}
        if mode in ("Manual", "Training"):
            return {"kind": "done", "mode": mode}
        if mode == "Agent":
            if stage != "agent":
                return {"kind": "agent_snapshot", "mode": mode}
            sample = observation["sample"]
            self.controller.info("Gigaseal agent observation collected: "
                f"resistance={float(sample['resistance']):.3f} MΩ, "
                f"actual_pressure={float(sample['pressure']):.3f} mbar, "
                f"setpoint={float(sample['commanded_pressure_mbar']):.3f} mbar, "
                f"atm={bool(sample['pressure_atm_state'])}, "
                f"observations_since_last_action={int(sample['observations_since_last_action'])}")
            action = self.controller.agenthelper.run_inference(observation=sample, is_demo=False)
            self.controller.info(f"Gigaseal agent raw action: {action}")
            if action is None:
                self.controller.warning("Gigaseal agent returned no action; skipping pressure update.")
                return {"kind": "done", "mode": mode}
            values = np.asarray(action).reshape(-1)
            if values.size < 1:
                self.controller.warning("Gigaseal agent action has no values; skipping pressure update.")
                return {"kind": "done", "mode": mode}
            value = float(values[0])
            atm = bool(value >= 0.0) if values.size == 1 else bool(float(values[1]) >= 0.5)
            pressure = (float(self.controller.pressure.get_pressure())
                if values.size == 1 and atm else float(np.clip(value,
                    float(self.controller.config.pressure_ramp_max), -5.0)))
            self.controller.info("Gigaseal agent decoded action: "
                f"commanded_pressure={pressure:.3f} mbar, atm={atm}")
            return {"kind": "agent", "mode": mode, "pressure": pressure,
                    "atm": atm, "retain_setpoint": values.size == 1 and atm}
        if stage == "agent":
            return {"kind": "route", "mode": mode}
        if not state.ramp_initialized:
            return {"kind": "initialize", "mode": mode, "continue_window": True}
        target = self.controller.config.gigaseal_R
        rate = observation["rate"]
        pressure, speed = state.currPressure, state.speed
        if rate < target / self.controller.config.increase_slope_gate:
            pressure -= 5
            speed = 3
        elif rate <= target / self.controller.config.constant_slope_gate:
            speed = 1
        elif rate <= target / self.controller.config.decrease_slope_gate:
            pressure += 5
            speed = 3
        return {"kind": "ramp", "mode": mode, "pressure": pressure, "speed": speed}

    def act(self, observation=None, state=None):
        self.failure_gate()
        self.success_gate()
        command = observation
        if command is None:
            return
        if (command["kind"] == "invalidate"
                or command["mode"] != self.controller.config.mode
                or (state.cycle_mode is not None and command["mode"] != state.cycle_mode)):
            self._invalidate_cycle(state)
            return
        kind = command["kind"]
        if command["mode"] != "Agent":
            state.last_agent_action = None
            state.observations_since_last_action = 0
        if command["mode"] not in ("Classic", "Adaptive"):
            state.ramp_initialized = False
        if kind in ("agent_snapshot", "route"):
            state.stage = "agent"
            return
        pressure = self.controller.pressure
        if kind == "initialize":
            pressure.set_pressure(-5)
            pressure.set_ATM(atm=False)
            state.currPressure = state.prevpressure = -5
            state.speed = 1
            state.ramp_initialized = True
            if command.get("continue_window"):
                state.stage = "agent"  # route back using the same main window
            return
        if kind == "ramp":
            limit = self.controller.config.pressure_ramp_max
            target = max(min(command["pressure"], -5.0), limit)
            if not self.action_gate(target):
                raise AutopatchError("Invalid gigaseal pressure command")
            state.currPressure, state.speed = target, command["speed"]
            if target != state.prevpressure:
                pressure.set_pressure(target)
                state.prevpressure = target
                self.controller.sleep(5 / state.speed)
            if command["mode"] != self.controller.config.mode:
                self._invalidate_cycle(state)
                return
            if target <= self.controller.config.pressure_ramp_max:
                pressure.set_ATM(atm=True)
                self.controller.sleep(5)
                if command["mode"] != self.controller.config.mode:
                    self._invalidate_cycle(state)
                    return
                state.stage = "release"
                return
        elif kind == "reset":
            pressure.set_pressure(-5)
            pressure.set_ATM(atm=False)
            state.currPressure = -5  # preserve legacy prevpressure after reset
        elif kind == "agent":
            target = (float(pressure.get_pressure()) if command["retain_setpoint"]
                      else command["pressure"])
            if not command["retain_setpoint"]:
                target = float(np.clip(target,
                    float(self.controller.config.pressure_ramp_max), -5.0))
            if not self.action_gate(target):
                raise AutopatchError("Invalid gigaseal Agent pressure command")
            applied = (target, command["atm"])
            if applied != state.last_agent_action:
                pressure.set_pressure(target)
                pressure.set_ATM(atm=command["atm"])
                state.last_agent_action = applied
                state.observations_since_last_action = 0
            else:
                state.observations_since_last_action += 1
        if command["mode"] != self.controller.config.mode:
            self._invalidate_cycle(state)
            return
        result = self.success_gate(state.main_window["resistance"], state)
        if result["holding"] is not None:
            self.controller.amplifier.set_holding(result["holding"])
            self.controller.amplifier.switch_holding(True)
            state.holding_switched = True
        if command["mode"] != self.controller.config.mode:
            self._invalidate_cycle(state)
            return
        state.stage = "finished" if result["complete"] else "main"
        if self.awaiting_operator and self.goal_event:
            self.finish(state)

    def _invalidate_cycle(self, state):
        state.stage = "main"
        state.cycle_mode = None
        state.main_window = None
        state.consecutive_success = 0
        state.ramp_initialized = False
        state.last_agent_action = None
        state.observations_since_last_action = 0

    def action_gate(self, observation=None, state=None):
        return np.isfinite(observation)

    def failure_gate(self, state=None, observation=None):
        self.controller.abort_if_requested()
        if self.awaiting_operator:
            return
        if observation is not None:
            self.controller.info(f"Test resistance: {observation['resistance']} MΩ; "
                f"difference: {observation['difference']} MΩ")
            if observation["difference"] < 0:
                state.bad_cell_count += 1
                if state.bad_cell_count > 5:
                    raise AutopatchError("Bad cell detected")
        elif state is not None and time.time() - state.last_progress_time >= self.controller.config.seal_deadline:
            raise AutopatchError("Seal attempt failed: resistance did not improve by at least "
                f"{self.controller.config.gigaseal_min_delta_R} MegaOhms by the "
                f"{self.controller.config.seal_deadline} second deadline.")

    def success_gate(self, observation=None, state=None):
        super().success_gate()
        if observation is None:
            return None
        target = self.controller.config.gigaseal_R
        holding = None
        if observation >= target / self.controller.config.hold_switch and not state.holding_switched:
            holding = self.controller.protocol_config.vclamp_hold
        state.consecutive_success = state.consecutive_success + 1 if observation >= target else 0
        complete = super().success_gate(state.consecutive_success >= 3, state)
        return {"holding": holding, "complete": complete}

    def finish(self, state=None):
        if state is None or (state.stage != "finished" and not self.goal_event):
            return
        self.controller.pressure.set_ATM(atm=True)
        self.controller.info("Seal successful!")
        self.goal_event = False
        if not self.awaiting_operator:
            self.complete_success()
