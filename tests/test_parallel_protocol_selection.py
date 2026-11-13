import threading
from types import SimpleNamespace

import patcherbot.interface
from patcherbot.controller.multi_patch import MultiPipettePatcher


class _RecordingDAQ:
    """Stands in for the rig's single DAQ and records every protocol call."""

    def __init__(self, events, event_lock):
        self.events = events
        self.event_lock = event_lock

    def __getattr__(self, method_name):
        def call(pipettes=None, **kwargs):
            with self.event_lock:
                self.events.append(("daq", method_name, tuple(pipettes)))
            return {key: None for key in pipettes}
        return call


class _ProtocolWorker:
    def __init__(self, pipette_key, barriers, events, event_lock):
        self.pipette_key = pipette_key
        self.amplifier_group = object()
        self.daq = SimpleNamespace(pipette_key=pipette_key)
        self.barriers = barriers
        self.events = events
        self.event_lock = event_lock
        self.iholding = None

    def _log(self, *event):
        with self.event_lock:
            self.events.append(event)

    def prepare_voltage_protocol(self):
        self._log("prepare", "voltage", self.pipette_key)
        self.barriers["voltage"].wait(timeout=2)   # fails unless prepares run in parallel
        return [("getDataFromVoltageProtocol", {"membrane_hold": -0.07})]

    def finish_voltage_protocol(self):
        self._log("finish", "voltage", self.pipette_key)

    def getHolding(self):
        return -20

    def prepare_current_protocol(self):
        self._log("prepare", "current", self.pipette_key)
        self.barriers["current"].wait(timeout=2)
        return [("getDataFromCurrentProtocol", {"recordingTimeMs": 500})]

    def finish_current_protocol(self):
        self._log("finish", "current", self.pipette_key)


class _ProtocolRunner:
    PROTOCOL_NAMES = MultiPipettePatcher.PROTOCOL_NAMES
    _prepare_protocol = MultiPipettePatcher._prepare_protocol
    _finish_protocol = MultiPipettePatcher._finish_protocol
    _merge_protocol_steps = MultiPipettePatcher._merge_protocol_steps
    _run_amplifier_workers = MultiPipettePatcher._run_amplifier_workers
    run_protocol_matrix = MultiPipettePatcher.run_protocol_matrix

    def warning(self, message):
        pass

    def abort_if_requested(self):
        pass

    def success_if_requested(self):
        pass

    def sleep(self, seconds):
        pass


def test_selected_pipettes_share_one_daq_acquisition_per_protocol():
    events = []
    event_lock = threading.Lock()
    barriers = {
        "voltage": threading.Barrier(2),
        "current": threading.Barrier(2),
    }
    keys = ["pipette_0", "pipette_1"]
    runner = _ProtocolRunner()
    runner.autopatchers = {
        key: _ProtocolWorker(key, barriers, events, event_lock) for key in keys
    }
    runner.daq = _RecordingDAQ(events, event_lock)
    selections = {
        key: {
            "voltage_protocol": True,
            "current_protocol": True,
            "voltage_sweep_protocol": False,
            "holding_protocol": False,
            "opto_random_wavelength_protocol": False,
            "opto_random_power_protocol": False,
        }
        for key in keys
    }

    runner.run_protocol_matrix(selections)

    daq_calls = [event for event in events if event[0] == "daq"]
    assert daq_calls == [
        ("daq", "getDataFromVoltageProtocol", tuple(keys)),
        ("daq", "getDataFromCurrentProtocol", tuple(keys)),
    ]
    for protocol, call in (("voltage", daq_calls[0]), ("current", daq_calls[1])):
        call_index = events.index(call)
        prepares = [i for i, e in enumerate(events) if e[:2] == ("prepare", protocol)]
        finishes = [i for i, e in enumerate(events) if e[:2] == ("finish", protocol)]
        assert len(prepares) == len(finishes) == 2
        assert max(prepares) < call_index < min(finishes)
    voltage_finishes = [i for i, e in enumerate(events) if e[:2] == ("finish", "voltage")]
    current_prepares = [i for i, e in enumerate(events) if e[:2] == ("prepare", "current")]
    assert max(voltage_finishes) < min(current_prepares)
