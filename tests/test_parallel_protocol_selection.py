import threading

import patcherbot.interface
from patcherbot.interface.patch import AutoPatchInterface


class _ProtocolDAQ:
    def setCellMode(self, mode):
        pass


class _ProtocolWorker:
    def __init__(self, pipette_key, barriers, events, event_lock):
        self.pipette_key = pipette_key
        self.daq = _ProtocolDAQ()
        self.barriers = barriers
        self.events = events
        self.event_lock = event_lock
        self.success_requested = False

    def _run_step(self, name):
        with self.event_lock:
            self.events.append(("start", name, self.pipette_key))
        self.barriers[name].wait(timeout=2)
        with self.event_lock:
            self.events.append(("finish", name, self.pipette_key))

    def run_voltage_protocol(self):
        self._run_step("voltage")

    def getHolding(self):
        return -20

    def run_current_protocol(self):
        self._run_step("current")

    def success_if_requested(self):
        pass


class _ProtocolRunner:
    _run_protocol_step = AutoPatchInterface._run_protocol_step
    _run_protocol_matrix = AutoPatchInterface._run_protocol_matrix


class _ProtocolTaskController:
    def abort_if_requested(self):
        pass

    def success_if_requested(self):
        pass


def test_selected_pipettes_run_each_protocol_in_parallel_before_advancing():
    events = []
    event_lock = threading.Lock()
    barriers = {
        "voltage": threading.Barrier(2),
        "current": threading.Barrier(2),
    }
    workers = {
        "pipette_0": _ProtocolWorker(
            "pipette_0", barriers, events, event_lock
        ),
        "pipette_1": _ProtocolWorker(
            "pipette_1", barriers, events, event_lock
        ),
    }
    runner = _ProtocolRunner()
    runner.autopatchers = workers
    runner.protocol_selections = {
        key: {
            "voltage_protocol": True,
            "current_protocol": True,
            "voltage_sweep_protocol": False,
            "holding_protocol": False,
            "opto_random_wavelength_protocol": False,
            "opto_random_power_protocol": False,
        }
        for key in workers
    }

    runner._run_protocol_matrix(_ProtocolTaskController())

    voltage_finishes = [i for i, event in enumerate(events) if event[:2] == ("finish", "voltage")]
    current_starts = [i for i, event in enumerate(events) if event[:2] == ("start", "current")]
    assert len(voltage_finishes) == len(current_starts) == 2
    assert max(voltage_finishes) < min(current_starts)
