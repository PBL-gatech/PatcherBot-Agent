"""
Generic Laser class for optogenetic systems, including on/off and
wavelength/power-addressable light engines.
"""
__all__ = ["Laser", "FakeLaser"]

import random

from patcherbot.controller import TaskController


def _is_off_wavelength(value):
    if value is None:
        return False
    if isinstance(value, str) and value.strip().lower() == "off":
        return True
    name = getattr(value, "name", None)
    return isinstance(name, str) and name.upper() == "OFF"


class Laser(TaskController):
    """
    Base class for Lasers used in Optogenetic Systems.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize the Laser with any necessary parameters.
        """
        super().__init__(*args, **kwargs)
        self._initialize()

    def _initialize(self):
        """Initialize the Laser with any necessary commands."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def power_on(self):
        """Power on the Laser."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def power_off(self):
        """Power off the Laser."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_power_state(self):
        """Get the current state of the Laser power and its power level."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def set_wavelength(self, wavelength=None):
        """Set the excitation wavelength for the Laser."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_wavelength(self):
        """Get the current excitation wavelength of the Laser."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def set_power_level(self, power_percent: float, wavelength=None):
        """Set the power level of a given excitation wavelength."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_laser_temp(self):
        """Get a temperature reading if supported by the Laser."""
        raise NotImplementedError("This method should be implemented by subclasses.")



    def get_laser_state(self):
        """Get the current state of the Laser (power level, wavelength, temperature)."""
        laser_state =  {
            "power_state": self.get_power_state(),
            "wavelength": self.get_wavelength(),
            "temperature": self.get_laser_temp(),
        }
        self.debug(f"Laser state: {laser_state}")
        return laser_state
    
    def excite(self, power_on=None, excitation_wavelength=None):
        """Enable a specific wavelength or turn it off."""
        laser_state = self.get_laser_state()
        if excitation_wavelength is not None:
            self.set_wavelength(excitation_wavelength)
        if power_on is not None:
            self.set_power_level(power_on)
        if laser_state["power_state"] == "on":
            self.power_off()
        else:
            self.power_on()
        return self.get_laser_state()
    
    def build_optogenetic_protocol(
        self,
        *,
        wavelengths: list[str | int],
        powers: list[int | float],
        randomize_target: str,
        stabilize_time: float,
        off_time: float,
        on_time: float,
        replicates: int,
        randomize: bool = True,
        power_divisor: float = 1.0,
    ):
        """
        Build a randomized optogenetic protocol for wavelengths or power levels.
        """
        if wavelengths is None or len(wavelengths) == 0:
            raise ValueError("wavelengths must contain at least one entry")
        if powers is None or len(powers) == 0:
            raise ValueError("powers must contain at least one entry")
        if replicates < 1:
            raise ValueError("replicates must be >= 1")
        if power_divisor == 0:
            raise ValueError("power_divisor must be non-zero")

        target = str(randomize_target).lower()
        if target not in ("wavelength", "power"):
            raise ValueError("randomize_target must be 'wavelength' or 'power'")

        if target == "wavelength" and len(powers) > 1:
            self.warning("Multiple power values provided; using the first value only.")
        if target == "power" and len(wavelengths) > 1:
            self.warning("Multiple wavelength values provided; using the first value only.")

        raw_on_time = float(on_time)
        raw_off_time = float(off_time)
        raw_stabilize_time = float(stabilize_time)
        # Laser is on for raw_on_time; recording window is 3x longer.
        record_time = raw_on_time * 3.0
        adjusted_off_time = raw_off_time - record_time
        adjusted_stabilize_time = raw_stabilize_time - record_time

        fixed_wavelength = wavelengths[0]
        fixed_power = powers[0]

        sequence = wavelengths if target == "wavelength" else powers
        expanded = []
        for rep in range(int(replicates)):
            values = list(sequence)
            if randomize and len(values) > 1:
                random.shuffle(values)
            expanded.extend(values)
        self.info(f"Optogenetic protocol {target} sequence: {expanded}")

        steps = []
        if adjusted_stabilize_time > 0:
            steps.append({
                "duration_s": float(adjusted_stabilize_time),
                "state": "off",
                "wavelength": "off",
                "power_percent": 0.0,
                "replicate": -1,
                "protocol_type": target,
            })

        total_count = len(expanded)
        for idx, value in enumerate(expanded):
            replicate = idx // len(sequence) if len(sequence) else 0
            if target == "wavelength":
                wavelength = value
                power_percent = float(fixed_power) / power_divisor
            else:
                wavelength = fixed_wavelength
                power_percent = float(value) / power_divisor

            is_off = _is_off_wavelength(wavelength)
            state = "off" if is_off else "on"
            if is_off:
                power_percent = 0.0

            steps.append({
                "duration_s": float(record_time),
                "state": state,
                "wavelength": wavelength,
                "power_percent": power_percent,
                "replicate": replicate,
                "protocol_type": target,
            })

            if raw_off_time > 0 and idx < (total_count - 1):
                off_duration = adjusted_off_time
                if off_duration > 0:
                    steps.append({
                        "duration_s": float(off_duration),
                        "state": "off",
                        "wavelength": wavelength,
                        "power_percent": 0.0,
                        "replicate": replicate,
                        "protocol_type": target,
                    })

        if raw_off_time > 0 and steps:
            last_step = steps[-1]
            last_state = str(last_step.get("state", "on")).lower()
            if last_state != "off":
                steps.append({
                    "duration_s": float(raw_off_time),
                    "state": "off",
                    "wavelength": last_step.get("wavelength"),
                    "power_percent": 0.0,
                    "replicate": last_step.get("replicate", -1),
                    "protocol_type": target,
                })

        return steps

class FakeLaser(Laser):
    """
    Fake Laser that only logs actions.
    """
    def __init__(self, *args, **kwargs):
        """Initialize the fake Laser."""
        super().__init__(*args, **kwargs)
        self.wavelength = None
        self.power_state = "off"
        self.power_levels = 0
        self.info("FakeLaser initialized.")

    def _initialize(self):
        """Skip hardware initialization for the fake Laser."""
        self.debug("FakeLaser: _initialize called (no hardware).")

    def power_on(self):
        """Fake implementation of powering on the Laser."""
        self.power_state = "on"
        self.info("FakeLaser: Power on.")

    def power_off(self):
        """Fake implementation of powering off the Laser."""
        self.power_state = "off"
        self.info("FakeLaser: Power off.")

    def get_power_state(self):
        """Fake implementation of getting the Laser power state."""
        return self.power_state

    def set_wavelength(self, wavelength=None):
        """Fake implementation of setting the excitation wavelength."""
        self.wavelength = wavelength
        self.info(f"FakeLaser: wavelength set to {wavelength}.")

    def get_wavelength(self):
        """Fake implementation of getting the current excitation wavelength."""
        return self.wavelength

    def set_power_level(self, power_percent: float, wavelength=None):
        """Fake implementation of setting the power level for a excitation wavelength."""
        clamped = int(max(0, min(100, round(power_percent))))
        self.power_levels = clamped
        self.info(f"FakeLaser: Power set to {clamped}%.")

    def get_laser_temp(self):
        """Fake implementation returning a placeholder temperature."""
        return 25.0
