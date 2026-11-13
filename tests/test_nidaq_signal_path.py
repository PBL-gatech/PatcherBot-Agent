"""Norwich input roles and shared protocol tasks, retaining the legacy lifecycle."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import patcherbot.interface
from patcherbot.devices.amplifier.DAQ import DAQ, NiDAQ
from patcherbot.utils.multipipette_helper import build_channel_map

daq_module = importlib.import_module("patcherbot.devices.amplifier.DAQ")
ROOT = Path(__file__).resolve().parents[1]


class Timing:
    def cfg_samp_clk_timing(self, **options):
        self.options = options
        self.samp_clk_rate = options["rate"]


class ModelCellDriver:
    """DAQmx test double: physical AI wiring is independent of the config under test.

    ai19/23 monitor the command, ai17/21 carry the response. V-clamp is a
    10 Mohm access resistor in series with a 500 Mohm / 30 pF membrane,
    with a one-sample current-monitor delay.
    """
    def __init__(self):
        self.tasks = []
        self.starts = []
        self.current_mode = False

    def task(self):
        task = ModelCellTask(self)
        self.tasks.append(task)
        return task

    def response(self, monitor, rate):
        if self.current_mode:
            return (monitor * .04 - .07) * 10  # measured membrane V, amplified tenfold
        step = monitor * .02  # 20 mV command per AO volt
        access, membrane, capacitance = 10e6, 500e6, 30e-12
        tau = access * membrane / (access + membrane) * capacitance
        response = step / (access + membrane)
        for index in np.flatnonzero(np.diff(np.r_[0., step])):
            delta = step[index] - (step[index - 1] if index else 0.)
            response[index:] += delta * (1 / access - 1 / (access + membrane)) * np.exp(
                -np.arange(len(step) - index) / (rate * tau))
        return np.r_[0., response[:-1]] / 2e-9


class ModelCellTask:
    def __init__(self, driver):
        self.driver = driver
        self.inputs, self.outputs, self.writes, self.reads = [], [], [], []
        self.ai_channels = SimpleNamespace(add_ai_voltage_chan=lambda name, **kw: self.inputs.append(name))
        self.ao_channels = SimpleNamespace(add_ao_voltage_chan=self.outputs.append)
        self.timing = Timing()
        self.in_stream = SimpleNamespace(input_buf_size=0, avail_samp_per_chan=10_000_000)
        self.out_stream = SimpleNamespace(output_buf_size=0)
        self.triggers = SimpleNamespace(start_trigger=SimpleNamespace(
            cfg_dig_edge_start_trig=Mock()))
        self.started = self.closed = False
        self.read_position = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def write(self, data, **kw):
        self.writes.append(np.array(data, copy=True))

    def start(self):
        self.started = True
        self.driver.starts.append(self)

    def stop(self):
        self.started = False

    def close(self):
        self.stop()
        self.closed = True

    def read(self, number_of_samples_per_channel, **kw):
        output = next((task for task in reversed(self.driver.tasks)
                       if task.outputs and task.started and not task.closed), None)
        if output is None:
            # DC amplifier outputs during holding / optical protocols.
            levels = {"Dev1/ai19": .25, "Dev1/ai17": .05,
                      "Dev1/ai23": .5, "Dev1/ai21": .075}
            data = np.array([np.full(number_of_samples_per_channel, levels[name])
                             for name in self.inputs])
            self.read_position += number_of_samples_per_channel
            self.reads.append(data.copy())
            return data
        queued = output.out_stream.regen_mode == daq_module.nidaqmx.constants.RegenerationMode.DONT_ALLOW_REGENERATION
        train_size = output.writes[0].shape[-1]
        train, offset = divmod(self.read_position, train_size) if queued else (-1, 0)
        write = output.writes[train]
        waves = np.atleast_2d(write)
        by_output = dict(zip(output.outputs, waves))
        rows = []
        for name in self.inputs:
            headstage = 0 if name in ("Dev1/ai17", "Dev1/ai19") else 1
            monitor = by_output["Dev1/ao0" if headstage == 0 else "Dev1/ao3"]
            response = self.driver.response(monitor, self.timing.samp_clk_rate)
            rows.append(monitor if name in ("Dev1/ai19", "Dev1/ai23") else response)
        data = np.vstack(rows)[:, offset:offset + number_of_samples_per_channel]
        self.read_position += number_of_samples_per_channel
        self.reads.append(data.copy())
        return data


class NIDaqSignalPathTests(unittest.TestCase):
    def setUp(self):
        self.driver = ModelCellDriver()
        self.task_patch = patch.object(daq_module.nidaqmx, "Task", side_effect=self.driver.task)
        self.task_patch.start()
        self.addCleanup(self.task_patch.stop)
        self.sleep_patch = patch.object(daq_module.time, "sleep")
        self.sleep_patch.start()
        self.addCleanup(self.sleep_patch.stop)
        config = json.loads((ROOT / "rig_setup/rig_configs/Norwich_multiple_rig.json").read_text())
        self.daq = NiDAQ.__new__(NiDAQ)
        DAQ.__init__(self.daq, build_channel_map(
            ["pipette_0", "pipette_1"], config["devices"]["daq"]["params"]["channels"]))
        for name in ("info", "debug", "error", "warning"):
            setattr(self.daq, name, Mock())
        self.daq.createSquareWave(40, 50_000, .5, .5, .025)
        self.daq._setupAcquisition()
        for key in self.daq.pipette_keys:
            self.daq.setCellMode(True, pipette_key=key)

    def test_live_physical_inputs_recover_known_model_cell_on_both_headstages(self):
        results = self.daq.getDataFromSquareWave(40, 50_000, .5, .5, .025,
                                                pipettes=self.daq.pipette_keys)
        self.assertEqual(len(self.daq.ai_task.reads), 1)  # one read, both headstages
        for key, ((time, current), (_, voltage), total, membrane, access, cap) in results.items():
            with self.subTest(pipette=key):
                self.assertAlmostEqual(voltage.max(), .010)
                self.assertGreater(current.max(), 900e-12)
                self.assertAlmostEqual(access, 10., delta=1.)
                self.assertAlmostEqual(membrane, 500., delta=5.)
                self.assertAlmostEqual(cap, 30., delta=3.)

    def test_membrane_protocol_uses_one_existing_stream_for_both_headstages(self):
        ai, ao = self.daq.ai_task, self.daq.ao_task
        starts = list(self.driver.starts)
        result = self.daq.getDataFromVoltageProtocol(pipettes=self.daq.pipette_keys)
        self.assertIs(self.daq.ai_task, ai)
        self.assertIs(self.daq.ao_task, ao)
        self.assertEqual(self.driver.starts, starts)  # no added task restart or zeroing
        self.assertEqual(len(ai.reads), 5)  # five simultaneous two-headstage reads
        self.assertEqual(len(ai.inputs), 4)
        self.assertEqual(ao.outputs, ['Dev1/ao0', 'Dev1/ao3'])
        for key, cap in result.items():
            self.assertAlmostEqual(cap, 30., delta=3.)
            trace = self.daq.pipette_state(key).voltage_membrane_test
            self.assertAlmostEqual(trace['command'].max(), .010)
            self.assertGreater(trace['response'].max(), 900e-12)
        self.assertTrue(self.daq._pause_evt.is_set())

    def test_current_protocol_keeps_legacy_waveforms_in_shared_headstage_tasks(self):
        for key, cap in zip(self.daq.pipette_keys, (30., 45.)):
            self.daq.pipette_state(key).voltageMembraneCapacitance = cap
        raw = {key: np.zeros((2, 400)) for key in self.daq.pipette_keys}
        with patch.object(self.daq, '_read_train', return_value=raw) as read:
            result = self.daq.getDataFromCurrentProtocol(
                pipettes=self.daq.pipette_keys, recordingTimeMs=1.)
        self.assertEqual(read.call_count, 14)  # one prime plus 13 shared current steps
        ao = next(task for task in self.driver.tasks if len(task.writes) == 14)
        ai = self.driver.starts[self.driver.starts.index(ao) + 1]
        self.assertEqual(ai.inputs, ['Dev1/ai19', 'Dev1/ai17', 'Dev1/ai23', 'Dev1/ai21'])
        self.assertEqual(ao.outputs, ['Dev1/ao0', 'Dev1/ao3'])
        self.assertEqual(ai.timing.samp_clk_rate, ao.timing.samp_clk_rate)
        self.assertEqual(ao.timing.options['sample_mode'],
                         daq_module.nidaqmx.constants.AcquisitionType.CONTINUOUS)
        self.assertEqual(ao.out_stream.regen_mode,
                         daq_module.nidaqmx.constants.RegenerationMode.ALLOW_REGENERATION)
        for row, (key, (traces, pulses, count)) in enumerate(result.items()):
            self.assertEqual(count, 13)
            self.assertEqual(len(traces), 13)
            for index, pulse in enumerate(pulses):
                generated = ao.writes[index + 1][row]
                np.testing.assert_allclose(generated[:100], 0.)  # baseline
                self.assertAlmostEqual(generated[100], -20e-12 / 400e-12)  # calibrator
                self.assertAlmostEqual(generated[200], pulse * 1e-12 / 400e-12)  # test
                np.testing.assert_allclose(generated[300:], 0.)  # ending baseline
        self.assertTrue(self.daq._pause_evt.is_set())


if __name__ == '__main__':
    unittest.main()
