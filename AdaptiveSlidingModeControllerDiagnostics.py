"""Offline reproduction of the supplemental Figure S4 comparison.

Run from the repository root with::

    python testing/AdaptiveSlidingModeControllerDiagnostics.py

Change ``TIME_STEP_S`` below to inspect different controller update rates.
The synthetic plant follows equation (17) from the paper. The graph contains
only resistance trajectories and resistance tracking errors, matching Figure
S4(a) and Figure S4(b); actuator commands are intentionally not plotted.
"""
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
from simple_pid import PID


# User-facing simulation controls.
TIME_STEP_S = 0.01
SIMULATION_DURATION_S = 60.0
BASELINE_RESISTANCE_MOHM = 10.0
TARGET_RESISTANCE_MOHM = 1200.0

MODULE_PATH = Path(__file__).parents[0] / "patcherbot" / "deepLearning" / "AdaptiveSlidingModeController.py"


def load_controller_class():
    spec = importlib.util.spec_from_file_location("adaptive_sliding_mode_controller", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AdaptiveSlidingModeController


def load_paper_model():
    """Load the physical coefficients calculated by the production ASMC."""
    controller_class = load_controller_class()
    controller = controller_class(target_resistance_mohm=TARGET_RESISTANCE_MOHM)
    return controller


def desired_length_trajectory(time_s, baseline_length_m, target_length_m, acceleration_m_per_s2, rate_m_per_s):
    """Build the paper's accelerated-then-constant-rate reference trajectory."""
    acceleration_time_s = rate_m_per_s / acceleration_m_per_s2
    accelerated_length_m = baseline_length_m + 0.5 * acceleration_m_per_s2 * acceleration_time_s ** 2
    if time_s <= acceleration_time_s:
        return baseline_length_m + 0.5 * acceleration_m_per_s2 * time_s ** 2
    return min(
        target_length_m,
        accelerated_length_m + rate_m_per_s * (time_s - acceleration_time_s),
    )


def make_pid_controllers():
    """Create the paper's hardcoded SI-PID and DI-PID models.

    The PID outputs are physical commands in this diagnostic: pressure is
    mbar and voltage is volts. Negative gains map increasing resistance error
    to the negative-pressure/negative-voltage convention used by this rig.
    """
    si_pid = PID(400, 600, 50, setpoint=0.0, sample_time=None, output_limits=(-90.0, -5.0))
    di_pressure_pid = PID(300, 500, 50, setpoint=0.0, sample_time=None, output_limits=(-90.0, -5.0))
    di_voltage_pid = PID(3, 6, 0.5, setpoint=0.0, sample_time=None, output_limits=(-0.11, 0.0))
    return si_pid, di_pressure_pid, di_voltage_pid


def advance_plant(state, pressure_mbar, voltage_v, disturbance_m_per_s2, model_controller):
    """Advance the stiff linear plant exactly for a zero-order-held command."""
    system_matrix = np.array([
        [0.0, 1.0],
        [model_controller.model_b, model_controller.model_a],
    ])
    input_matrix = np.array([
        [0.0, 0.0, 0.0],
        [model_controller.model_c, model_controller.model_d, 1.0],
    ])
    augmented_matrix = np.block([
        [system_matrix, input_matrix],
        [np.zeros((3, 5))],
    ])
    transition = expm(augmented_matrix * TIME_STEP_S)
    #factor of 100 for the pressure to convert to pascals
    augmented_state = np.concatenate([
        state,
        [pressure_mbar * 100.0, -voltage_v, disturbance_m_per_s2],
    ])
    return transition[:2, :] @ augmented_state


def simulate():
    if TIME_STEP_S <= 0:
        raise ValueError("TIME_STEP_S must be positive")

    model_controller = load_paper_model()
    times = np.arange(0.0, SIMULATION_DURATION_S + TIME_STEP_S, TIME_STEP_S)
    baseline_length_m = model_controller.resistance_to_length(BASELINE_RESISTANCE_MOHM)
    target_length_m = model_controller.resistance_to_length(TARGET_RESISTANCE_MOHM)
    acceleration_m_per_s2 = model_controller.resistance_to_length(4.0)
    constant_rate_m_per_s = model_controller.resistance_to_length(52.0)

    asmc = load_controller_class()(target_resistance_mohm=TARGET_RESISTANCE_MOHM)
    asmc.reset(
        initial_resistance_mohm=BASELINE_RESISTANCE_MOHM,
        initial_pressure_mbar=-5.0,
        initial_voltage_v=0.0,
    )
    si_pid, di_pressure_pid, di_voltage_pid = make_pid_controllers()

    states = {
        "SI-PID": np.array([baseline_length_m, 0.0]),
        "DI-PID": np.array([baseline_length_m, 0.0]),
        "ASMC": np.array([baseline_length_m, 0.0]),
    }
    resistances = {name: [] for name in states}
    desired_resistances = []

    for time_s in times:


        desired_length_m = desired_length_trajectory(
            time_s,
            baseline_length_m,
            target_length_m,
            acceleration_m_per_s2,
            constant_rate_m_per_s,
        )
        desired_resistance_mohm = desired_length_m * 1e9 / model_controller._resistance_to_length_Mohm_to_nm
        desired_resistances.append(desired_resistance_mohm)

        asmc_command = asmc.update(
            resistance_mohm=resistances["ASMC"][-1] if resistances["ASMC"] else BASELINE_RESISTANCE_MOHM,
            measurement_window_s=TIME_STEP_S,
        )
        si_pid.setpoint = desired_length_m* 1e9 / model_controller._resistance_to_length_Mohm_to_nm
        di_pressure_pid.setpoint = desired_length_m* 1e9 / model_controller._resistance_to_length_Mohm_to_nm
        di_voltage_pid.setpoint = desired_length_m* 1e9 / model_controller._resistance_to_length_Mohm_to_nm
        
        si_pressure_mbar = si_pid(states["SI-PID"][0]* 1e9 / model_controller._resistance_to_length_Mohm_to_nm, dt=TIME_STEP_S)
        di_pressure_mbar = di_pressure_pid(states["DI-PID"][0]* 1e9 / model_controller._resistance_to_length_Mohm_to_nm, dt=TIME_STEP_S)
        di_voltage_v = di_voltage_pid(states["DI-PID"][0]* 1e9 / model_controller._resistance_to_length_Mohm_to_nm, dt=TIME_STEP_S)

        commands = {
            "SI-PID": (si_pressure_mbar, 0.0),
            "DI-PID": (di_pressure_mbar, di_voltage_v),
            "ASMC": (asmc_command.pressure_mbar or -5.0, asmc_command.holding_voltage_v or 0.0),
        }

        acceleration_vec = [model_controller.model_a*states["ASMC"][0], model_controller.model_b*states["ASMC"][1], model_controller.model_c*asmc_command.pressure_mbar*100, model_controller.model_d*asmc_command.holding_voltage_v]
        #print(acceleration_vec, np.sum(acceleration_vec))
        #print(states["ASMC"], asmc_command.pressure_mbar, asmc_command.holding_voltage_v)
        #print(model_controller.model_a, model_controller.model_b, model_controller.model_c, model_controller.model_d)
        

        for name, (pressure_mbar, voltage_v) in commands.items():
            pressure_mbar *= -1 #the controllers default output is negative but the diagnostic is working in positive, the paper just has a flipped convention
            disturbance_m_per_s2 = 2e-6 * np.sin(3.14 * time_s)
            states[name] = advance_plant(
                states[name],
                pressure_mbar,
                voltage_v,
                disturbance_m_per_s2,
                model_controller,
            )
            position = states[name][0]
            resistance = position * 1e9 / model_controller._resistance_to_length_Mohm_to_nm
            resistances[name].append(max(0.0, resistance))
        
        

    return times, np.asarray(desired_resistances), resistances


def plot_figure(times, desired_resistances, resistances):
    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 7))
    styles = {
        "SI-PID": {"color": "#2f6f6d", "linestyle": "-"},
        "DI-PID": {"color": "#d08b4f", "linestyle": "--"},
        "ASMC": {"color": "#303b73", "linestyle": "-"},
    }

    axes[0].plot(times, desired_resistances, color="black", linestyle=":", label="desired trajectory")
    for name, values in resistances.items():
        axes[0].plot(times, values, label=name, **styles[name])
    axes[0].set_ylabel("resistance (MOhm)")
    axes[0].set_title("Synthetic gigaseal resistance trajectories")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    for name, values in resistances.items():
        axes[1].plot(times, np.asarray(values) - desired_resistances, label=name, **styles[name])
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("tracking error (MOhm)")
    axes[1].set_title(f"Tracking error, timestep = {TIME_STEP_S:g} s")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.tight_layout()

    output_path = Path(__file__).with_name("supplementary_figure_4_recreation.png")
    figure.savefig(output_path, dpi=150)
    print(f"saved: {output_path}")
    print(f"timestep: {TIME_STEP_S:g} s ({1.0 / TIME_STEP_S:g} Hz)")
    for name, values in resistances.items():
        error = np.asarray(values) - desired_resistances
        print(f"{name}: final={values[-1]:.2f} MOhm, RMSE={np.sqrt(np.mean(error ** 2)):.2f} MOhm")
    plt.show()


def main():
    times, desired_resistances, resistances = simulate()
    plot_figure(times, desired_resistances, resistances)


if __name__ == "__main__":
    main()
