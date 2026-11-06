import threading
from types import SimpleNamespace

import patcherbot.interface
from patcherbot.interface.patch import AutoPatchInterface


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
        self.amplifier = object()
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
    _PROTOCOL_RUNNERS = AutoPatchInterface._PROTOCOL_RUNNERS
    _prepare_protocol = AutoPatchInterface._prepare_protocol
    _finish_protocol = AutoPatchInterface._finish_protocol
    _merge_protocol_steps = AutoPatchInterface._merge_protocol_steps
    _run_amplifier_workers = AutoPatchInterface._run_amplifier_workers
    _run_protocol_matrix = AutoPatchInterface._run_protocol_matrix

    def warning(self, message):
        pass


class _ProtocolTaskController:
    def abort_if_requested(self):
        pass

    def success_if_requested(self):
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
    runner.protocol_daQs = {key: SimpleNamespace(pipette_key=key) for key in keys}
    runner.daq = _RecordingDAQ(events, event_lock)
    runner.protocol_selections = {
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

    runner._run_protocol_matrix(_ProtocolTaskController())

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
