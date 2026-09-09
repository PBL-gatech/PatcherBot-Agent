"""Dual-input adaptive sliding-mode controller for gigaseal formation.

Equations (S1)-(S4) from the supplied supplemental material are implemented
in the paper's membrane-length state. Equation (16) converts measured seal
resistance to length using SI units. The paper's pressure input u1 is treated
as pascals and converted to mbar for the pressure device; u2 is -psi_p and is
converted to the negative amplifier voltage in volts.

The JSON file is the production parameter source. Its initial estimates and
actuator limits include explicit notes where the paper does not specify an
implementation detail.
"""
import copy
import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SealControlCommand:
    """Commands and state returned by one ASMC update."""

    pressure_mbar: float | None = None
    holding_voltage_v: float | None = None
    atmospheric: bool | None = None
    holding_enabled: bool | None = None


class AdaptiveSlidingModeControllerBase(ABC):
    """Abstract base for resistance/length-based ASMC implementations.

    The base class owns the shared parameter loading, model derivation,
    trajectory bookkeeping, state reset, resistance-to-length conversion,
    sign and actuator clipping helpers, and the command container contract.
    A concrete subclass can then define the pressure/voltage control law and
    adaptive estimator policy that is unique to a particular implementation.
    """

    def __init__(self, target_resistance_mohm, parameters=None, parameter_path=None):
        self.parameters = self._load_parameters(parameters, parameter_path)
        trajectory = self.parameters["trajectory"]
        controller = self.parameters["controller"]

        self.target_resistance_mohm = float(target_resistance_mohm)
        self.initial_acceleration_resistance = float(trajectory["initial_acceleration_mohm_per_s2"])
        self.constant_rate_resistance = float(trajectory["constant_rate_mohm_per_s"])

        self.alpha = float(controller["alpha_per_s"])
        self.k1 = float(controller["k1"])
        self.k2 = float(controller["k2"])
        self.gamma = [float(controller[f"gamma{i}"]) for i in range(1, 7)]
        self.lambda1 = float(controller["lambda1"])
        self.lambda2 = float(controller["lambda2"])
        self.pressure_min_mbar = float(controller["pressure_min_mbar"])
        self.pressure_max_mbar = float(controller["pressure_max_mbar"])
        self.voltage_min_v = float(controller["voltage_min_v"])
        self.voltage_max_v = float(controller["voltage_max_v"])
        model = self.parameters["model"]
        self.pipette_radius_um = float(model["pipette_radius_um"])
        self.liquid_layer_thickness_nm = float(model["liquid_layer_thickness_nm"])
        self.seal_media_resistivity_ohm_m = float(model["seal_media_resistivity_ohm_m"])
        self._resistance_to_length_Mohm_to_nm = (
            2.0 * math.pi * self.pipette_radius_um * self.liquid_layer_thickness_nm
            / self.seal_media_resistivity_ohm_m
        )
        self._derive_nominal_model_parameters(model)
        self.reset()

    @staticmethod
    def _load_parameters(parameters, parameter_path):
        if parameters is not None:
            return copy.deepcopy(parameters)
        path = Path(parameter_path) if parameter_path else Path(__file__).with_name("asmcModel") / "asmc_parameters.json"
        with path.open("r", encoding="utf-8") as parameter_file:
            return json.load(parameter_file)

    def _derive_nominal_model_parameters(self, model):
        """Calculate a,b,c,d from the physical model in the paper."""
        epsilon_0 = 8.854e-12
        epsilon_r = float(model["relative_permittivity"])
        mass_g = float(model["membrane_mass_g"])
        elastic_n_per_m = float(model["elastic_coefficient_n_per_m"])
        viscous_kg_per_s = float(model["viscous_coefficient_kg_per_s"])
        adhesion_n_per_m2 = float(model["adhesion_friction_n_per_m2"])

        membrane_potential_mv = float(model["membrane_surface_potential_mv"])
        pipette_potential_mv = float(model["pipette_surface_potential_mv"])
        potential_gradient_mv_per_nm = float(model["surface_potential_gradient_mv_per_nm"])

        self.model_a = -viscous_kg_per_s / (mass_g * 1e-3)
        self.model_b = -(adhesion_n_per_m2 * 2.0 * math.pi * self.pipette_radius_um * 1e-6 + elastic_n_per_m) / (mass_g * 1e-3)
        self.model_c = math.pi * self.pipette_radius_um ** 2 * 1e-12 / (mass_g * 1e-3)
        self.model_d = (
            2.0 * math.pi * self.pipette_radius_um * 1e-6 * epsilon_0 * epsilon_r / (mass_g * 1e-3)
            * ((membrane_potential_mv - pipette_potential_mv) / self.liquid_layer_thickness_nm
               + potential_gradient_mv_per_nm) * 1e-3 * 1e9
        )
        self.nominal_theta = [
            self.model_a / (2.0 * self.model_c),
            self.model_b / (2.0 * self.model_c),
            1.0 / (2.0 * self.model_c),
            self.model_a / (2.0 * self.model_d),
            self.model_b / (2.0 * self.model_d),
            1.0 / (2.0 * self.model_d),
        ]

    def reset(self, initial_resistance_mohm=None, initial_pressure_mbar=-5.0, initial_voltage_v=0.0):
        """Reset estimates, trajectory, and actuator state for a seal attempt."""
        self.elapsed_s = 0.0
        self.previous_resistance_mohm = initial_resistance_mohm
        self.previous_rate_mohm_per_s = 0.0
        self.previous_length_rate_m_per_s = 0.0

        self.desired_length_m = self.resistance_to_length(initial_resistance_mohm or 0.0)
        self.desired_length_rate_m_per_s = 0.0
        self.desired_length_acceleration_m_per_s2 = 0.0
        self.current_pressure_mbar = self._clip_pressure(initial_pressure_mbar)
        self.current_voltage_v = self._clip_voltage(initial_voltage_v)
        initial_theta = self.nominal_theta
        self.theta_hat = [float(value) for value in initial_theta]
        initial_deltas = self.parameters["controller"].get("initial_disturbance_estimates", [0.0, 0.0])
        self.delta_hat_1, self.delta_hat_2 = (float(value) for value in initial_deltas)
        self.inputs_stopped = False

    @abstractmethod
    def update(self, resistance_mohm, measurement_window_s):
        """Advance the model and return bounded pressure/voltage commands."""

    @abstractmethod
    def _update_trajectory(self, dt_s):
        """Generate the accelerated-then-constant-rate trajectory in the paper."""

    @abstractmethod
    def _update_adaptive_estimates(
        self,
        measured_length_m,
        measured_length_rate_m_per_s,
        measured_length_acceleration_m_per_s2,
        sliding_surface,
        measurement_window_s,
    ):
        """Update control estimator state for the implementation."""

    @abstractmethod
    def _remember_measurement(self, resistance, rate, length_rate):
        """Remember the last measurement for derivative estimation."""

    def resistance_to_length(self, resistance_mohm):
        """
        Implement equation (16): L = R * 2*pi*Rp*h / rho.
        The error in for the control signal was given in terms of length,
        but we can only measure resistance.
        """
        return float(resistance_mohm) * 1e-9 * self._resistance_to_length_Mohm_to_nm

    @staticmethod
    def _sign(value):
        if value > 0:
            return 1.0
        if value < 0:
            return -1.0
        return 0.0

    def _clip_pressure(self, pressure_mbar):
        return max(self.pressure_min_mbar, min(self.pressure_max_mbar, float(pressure_mbar)))

    def _clip_voltage(self, voltage_v):
        return max(self.voltage_min_v, min(self.voltage_max_v, float(voltage_v)))


class AdaptiveSlidingModeController(AdaptiveSlidingModeControllerBase):
    """Concrete dual-input ASMC using the current resistance/length control law.

    This class preserves the existing logic in the repository: it uses the
    paper's trajectory, the dual-input control surface, and the adaptive
    estimate-update law encoded by the current `asmc_parameters.json` gains.
    """

    def update(self, resistance_mohm, measurement_window_s):
        """Advance (S1)-(S4) and return bounded pressure/voltage commands.

        ``resistance_mohm`` is used for x, with resistance derivatives in
        MOhm/s. ``measurement_window_s`` is the time represented by the
        averaged measurement window. The controller is called once per
        averaged window; one previous value is enough for the discrete
        derivative. Pressure is returned in mbar and voltage in volts. The
        paper's u2 is -psi_p, so the voltage command applies the corresponding
        negative sign when converting u2 to the amplifier voltage.
        """
        if measurement_window_s <= 0:
            raise ValueError("measurement_window_s must be positive")
        resistance_mohm = float(resistance_mohm)
        measurement_window_s = float(measurement_window_s)
        if self.previous_resistance_mohm is None:
            self.previous_resistance_mohm = resistance_mohm
            self.desired_length_m = self.resistance_to_length(resistance_mohm)

        measured_rate_mohm_per_s = (
            resistance_mohm - self.previous_resistance_mohm
        ) / measurement_window_s
        measured_length_m = self.resistance_to_length(resistance_mohm)
        measured_length_rate_m_per_s = self.resistance_to_length(measured_rate_mohm_per_s)
        measured_length_acceleration_m_per_s2 = (
            measured_length_rate_m_per_s - self.previous_length_rate_m_per_s
        ) / measurement_window_s
        self.elapsed_s += measurement_window_s
        self._update_trajectory(measurement_window_s)

        length_error_m = measured_length_m - self.desired_length_m
        length_rate_error_m_per_s = (
            measured_length_rate_m_per_s - self.desired_length_rate_m_per_s
        )
        sliding_surface = self.alpha * length_error_m + length_rate_error_m_per_s
        trajectory_feedforward_m_per_s2 = (
            self.alpha * length_rate_error_m_per_s
            - self.desired_length_acceleration_m_per_s2
        )

        self._update_adaptive_estimates(
            measured_length_m,
            measured_length_rate_m_per_s,
            measured_length_acceleration_m_per_s2,
            sliding_surface,
            measurement_window_s,
        )

        sign_surface = self._sign(sliding_surface)
        u1 = (
            -self.k1 * sliding_surface
            - self.theta_hat[0] * measured_length_rate_m_per_s
            - self.theta_hat[1] * measured_length_m
            - self.theta_hat[2] * trajectory_feedforward_m_per_s2
            - self.delta_hat_1 * sign_surface
        )
        u2 = (
            -self.k2 * sliding_surface
            - self.theta_hat[3] * measured_length_rate_m_per_s
            - self.theta_hat[4] * measured_length_m
            - self.theta_hat[5] * trajectory_feedforward_m_per_s2
            - self.delta_hat_2 * sign_surface
        )

        self.current_pressure_mbar = self._clip_pressure(-0.01 * u1)
        self.current_voltage_v = self._clip_voltage(-u2)
        self._remember_measurement(resistance_mohm, measured_rate_mohm_per_s, measured_length_rate_m_per_s)
        return SealControlCommand(
            pressure_mbar=self.current_pressure_mbar,
            holding_voltage_v=self.current_voltage_v,
            atmospheric=False,
            holding_enabled=True,
        )

    def _update_trajectory(self, dt_s):
        """Generate the accelerated-then-constant-rate trajectory in the paper."""
        if self.inputs_stopped:
            return
        paper_acceleration_m_per_s2 = self.resistance_to_length(self.initial_acceleration_resistance)
        paper_constant_rate_m_per_s = self.resistance_to_length(self.constant_rate_resistance)
        if self.desired_length_rate_m_per_s < paper_constant_rate_m_per_s:
            self.desired_length_acceleration_m_per_s2 = paper_acceleration_m_per_s2
            self.desired_length_rate_m_per_s = min(
                paper_constant_rate_m_per_s,
                self.desired_length_rate_m_per_s + paper_acceleration_m_per_s2 * dt_s,
            )
        else:
            self.desired_length_acceleration_m_per_s2 = 0.0
        target_length_m = self.resistance_to_length(self.target_resistance_mohm)
        self.desired_length_m = min(
            target_length_m,
            self.desired_length_m + self.desired_length_rate_m_per_s * dt_s,
        )

    def _update_adaptive_estimates(
        self,
        measured_length_m,
        measured_length_rate_m_per_s,
        measured_length_acceleration_m_per_s2,
        sliding_surface,
        measurement_window_s,
    ):
        """
        Update control gains from equation S4.
        Agent made a mistake here, parsing the equations is hard because the dots are not clearly over a particular variable in the supplement.
        I used my best judgement to figure out what derivative is being taken and have the corresponding variable in place.
        """
        regressors = (
            measured_length_rate_m_per_s,
            measured_length_m,
            self.model_c * measured_length_rate_m_per_s
            - self.desired_length_acceleration_m_per_s2,
        )
        for pressure_index, regressor in enumerate(regressors):
            self.theta_hat[pressure_index] += (
                measurement_window_s
                * self.gamma[pressure_index]
                * regressor
                * sliding_surface
            )
            voltage_index = pressure_index + 3
            self.theta_hat[voltage_index] += (
                measurement_window_s
                * self.gamma[voltage_index]
                * regressor
                * sliding_surface
            )
        self.delta_hat_1 = self.lambda1 * abs(sliding_surface)
        self.delta_hat_2 = self.lambda2 * abs(sliding_surface)

    def _remember_measurement(self, resistance, rate, length_rate):
        self.previous_resistance_mohm = resistance
        self.previous_rate_mohm_per_s = rate
        self.previous_length_rate_m_per_s = length_rate
