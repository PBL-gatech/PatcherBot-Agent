"""Controller for work that spans several pipettes at once.

``MultiPipettePatcher`` owns one ``AutoPatcher`` per pipette and the rig's
single DAQ. It pairs each pipette with its devices (amplifier headstage,
pressure controller, DAQ view) and runs the protocol matrix: per-pipette
amplifier setup, ONE shared DAQ acquisition per step, per-pipette cleanup.
Interfaces only hold the user's selections and ``execute`` this controller.
"""
from concurrent.futures import ThreadPoolExecutor

from patcherbot.utils.multipipette_helper import device_for_pipette

from .base import TaskController
from .patch import AutoPatcher

__all__ = ["MultiPipettePatcher"]


class MultiPipettePatcher(TaskController):
    """One AutoPatcher per pipette plus the shared DAQ."""

    # Protocol selection names, in run order -> AutoPatcher prepare_/finish_ suffix
    PROTOCOL_NAMES = {
        "voltage_protocol": "voltage_protocol",
        "current_protocol": "current_protocol",
        "voltage_sweep_protocol": "voltage_sweep_protocol",
        "holding_protocol": "holding_protocol",
        "opto_random_wavelength_protocol": "optogenetic_protocol",
        "opto_random_power_protocol": "optogenetic_protocol",
    }

    def __init__(self, autopatchers, daq):
        self.autopatchers = dict(autopatchers)
        self.daq = daq
        super().__init__()

    @classmethod
    def from_devices(cls, daq, amplifiers, pressures, calibrated_units, calibrated_stage,
                     lamp, laser=None, config=None, protocol_config=None):
        """Build one AutoPatcher per calibrated unit (keyed ``pipette_<n>``).

        ``amplifiers`` / ``pressures`` are ``{pipette_key: device}``; a single
        device serves every pipette, and a device with ``for_pipette`` (e.g.
        MultiClampAmplifier) hands each pipette its own part. Each pipette
        gets its own view onto the one DAQ.
        """
        autopatchers = {}
        for key, calibrated_unit in calibrated_units.items():
            autopatchers[key] = AutoPatcher(
                device_for_pipette(amplifiers, key),
                daq.for_pipette(key),
                device_for_pipette(pressures, key),
                calibrated_unit,
                calibrated_unit.microscope,
                calibrated_stage=calibrated_stage,
                lamp=lamp,
                laser=laser,
                config=config,
                protocol_config=protocol_config,
            )
        return cls(autopatchers, daq)

    # --------------------------------------------------------------
    # Protocol matrix
    # --------------------------------------------------------------
    @staticmethod
    def _opto_params(protocol_name, config):
        if protocol_name == "opto_random_wavelength_protocol":
            opto_params = {
                "randomize_target": "wavelength",
                "wavelengths": ["red", "green", "cyan", "uv", "blue", "infrared"],
                "powers": [float(config.opto_wavelength_power)],
            }
        else:
            opto_params = {
                "randomize_target": "power",
                "wavelengths": [config.opto_power_wavelength],
                "powers": list(range(0, 101, 20)),
            }
        opto_params.update({
            "stabilize_time": float(config.opto_stabilize_time),
            "off_time": float(config.opto_off_time),
            "on_time": float(config.opto_on_time),
            "replicates": int(config.opto_replicates),
            "randomize": True,
        })
        return opto_params

    def _prepare_protocol(self, pipette_key, protocol_name):
        """Amplifier setup for one pipette; returns its DAQ steps."""
        autopatcher = self.autopatchers[pipette_key]
        runner = self.PROTOCOL_NAMES[protocol_name]
        if protocol_name == "current_protocol":
            autopatcher.iholding = autopatcher.getHolding()
        if runner == "optogenetic_protocol":
            return autopatcher.prepare_optogenetic_protocol(
                self._opto_params(protocol_name, autopatcher.protocol_config))
        return getattr(autopatcher, f"prepare_{runner}")()

    def _finish_protocol(self, pipette_key, protocol_name):
        runner = self.PROTOCOL_NAMES[protocol_name]
        getattr(self.autopatchers[pipette_key], f"finish_{runner}")()

    def _merge_protocol_steps(self, protocol_name, steps_by_pipette):
        """One list of DAQ steps for all pipettes.

        Settings come from the shared protocol config, so the pipettes should
        agree; if not, the first pipette's settings are used. Opto sequences
        are randomized per pipette, but the laser is shared, so the first
        pipette's sequence drives one stimulus run recorded on every pipette.
        """
        reference_key, reference = next(iter(steps_by_pipette.items()))
        methods = [method for method, _ in reference]
        for pipette_key, steps in steps_by_pipette.items():
            if [method for method, _ in steps] != methods:
                raise ValueError(
                    f"{protocol_name}: {pipette_key} needs different DAQ steps than {reference_key}"
                )
            if self.PROTOCOL_NAMES[protocol_name] != "optogenetic_protocol" and steps != reference:
                self.warning(f"{protocol_name}: {pipette_key} settings differ from {reference_key}; "
                             f"using {reference_key}'s for the shared acquisition")
        return reference

    def _run_amplifier_workers(self, selected, operation):
        """Run separate headstages in parallel, sharing a channel sequentially.

        MultiClamp's existing DLL lock protects each channel-selection/call;
        settling waits for different headstages can overlap.

        Returns ``({key: result}, {key: exception})`` for the pipettes that
        succeeded / failed; one failure doesn't stop the others.
        """
        groups = {}
        for key in selected:
            groups.setdefault(id(self.autopatchers[key].amplifier_group), []).append(key)
        if not groups:
            return {}, {}

        def run_group(keys):
            outcome = {}
            for key in keys:
                try:
                    outcome[key] = (True, operation(key))
                except Exception as exc:
                    outcome[key] = (False, exc)
            return outcome

        outcomes = {}
        with ThreadPoolExecutor(max_workers=len(groups)) as executor:
            for group_outcome in executor.map(run_group, groups.values()):
                outcomes.update(group_outcome)
        results = {key: outcomes[key][1] for key in selected if outcomes[key][0]}
        errors = {key: outcomes[key][1] for key in selected if not outcomes[key][0]}
        return results, errors

    def run_protocol_matrix(self, selections):
        """Run each selected protocol for all its pipettes at once.

        ``selections`` is ``{pipette_key: {protocol_name: bool}}``. For each
        protocol in ``PROTOCOL_NAMES`` order: prepare the ticked pipettes'
        amplifiers, ONE DAQ acquisition per step covering all of them, then
        finish every ticked pipette (even if setup failed partway).

        An abort is honoured between protocols and between DAQ steps. It is
        not forwarded to the AutoPatchers, so every finish (amplifier
        cleanup) still runs to completion.
        """
        self.abort_if_requested()
        for protocol_name in self.PROTOCOL_NAMES:
            self.abort_if_requested()
            self.success_if_requested()
            selected = [
                pipette_key for pipette_key, chosen in selections.items()
                if chosen.get(protocol_name, False)
            ]
            if not selected:
                continue

            prepared, prepare_errors = self._run_amplifier_workers(
                selected, lambda key: self._prepare_protocol(key, protocol_name))
            try:
                if prepare_errors:
                    key, error = next(iter(prepare_errors.items()))
                    raise RuntimeError(f"{protocol_name}: amplifier setup failed for {key}") from error
                steps = self._merge_protocol_steps(protocol_name, prepared)
                daq_keys = list(dict.fromkeys(
                    self.autopatchers[key].daq.pipette_key for key in selected))
                for index, (method_name, kwargs) in enumerate(steps):
                    if index:
                        self.sleep(0.25)
                    getattr(self.daq, method_name)(pipettes=daq_keys, **kwargs)
            finally:
                # Every selected pipette is cleaned up, including one whose
                # setup failed partway (it may already be in the protocol's mode).
                _, finish_errors = self._run_amplifier_workers(
                    selected, lambda key: self._finish_protocol(key, protocol_name))
                for key, error in finish_errors.items():
                    self.error(f"{protocol_name}: amplifier cleanup failed for {key}: {error}")
        self.success_if_requested()
