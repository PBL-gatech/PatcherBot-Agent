"""Shared Norwich amplifier wiring without opening real rig hardware."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

import patcherbot.interface
from patcherbot.devices.amplifier.DAQ import build_channel_map
from patcherbot.interface.patch import AutoPatchInterface
from rig_setup.rig_config import RigConfigError, RigConfigManager


CONFIG_DIR = Path(__file__).resolve().parents[1] / "rig_setup" / "rig_configs"


class SharedAmplifierTests(unittest.TestCase):
    def setUp(self):
        self.manager = RigConfigManager()
        self.config = self.manager.load_config(
            CONFIG_DIR / "Norwich_multiple_rig.json", load_overlays=False
        )

    def test_norwich_uses_one_multiclamp_with_a_channel_per_pipette(self):
        amplifier = self.config["devices"]["amplifier"]
        self.assertIsInstance(amplifier, dict)
        self.assertTrue(amplifier["class"].endswith("multiclamp.MultiClampAmplifier"))
        # pipette_0 -> MultiClamp channel 1, pipette_1 -> channel 2
        self.assertEqual(amplifier["params"], {"channels": [1, 2]})
        self.assertEqual(self.config["devices"]["daq"]["params"]["channels"], [
            {"read": "Dev1/ai17", "resp": "Dev1/ai19", "cmd": "Dev1/ao0"},
            {"read": "Dev1/ai21", "resp": "Dev1/ai23", "cmd": "Dev1/ao3"},
        ])

    def test_build_instantiates_one_amplifier_for_one_or_two_pipettes(self):
        for count in (1, 2):
            with self.subTest(active_pipettes=count):
                selected_amplifier = object()

                def instantiate(slot, spec, instances, default_kwargs=None):
                    if slot == "amplifier":
                        return selected_amplifier
                    if slot == "daq":
                        return SimpleNamespace(channel_map=build_channel_map(
                            pipette_keys=default_kwargs["pipette_keys"],
                            channels=spec["params"]["channels"],
                        ))
                    return SimpleNamespace()

                with patch.object(self.manager, "_instantiate_slot", side_effect=instantiate) as loader:
                    devices = self.manager.build_devices(deepcopy(self.config), count)
                amplifier_calls = [call for call in loader.call_args_list
                                   if call.args[0].startswith("amplifier")]
                self.assertEqual(len(amplifier_calls), 1)
                self.assertIs(devices["amplifier"], selected_amplifier)
                keys = [f"pipette_{index}" for index in range(count)]
                self.assertEqual(list(devices["daq"].channel_map), keys)
                self.assertEqual(list(devices["pipette_unit"]), keys)
                for key in keys:
                    self.assertIs(AutoPatchInterface._device_for_pipette(
                        {"pipette_0": devices["amplifier"]}, key
                    ), selected_amplifier)
                if count == 2:
                    self.assertEqual(devices["daq"].channel_map["pipette_1"].resp, "Dev1/ai23")

    def test_per_pipette_amplifier_lists_still_validate(self):
        config = deepcopy(self.config)
        first = config["devices"]["amplifier"]
        second = deepcopy(first)
        second["params"]["channel"] = 2
        config["devices"]["amplifier"] = [first, second]
        self.manager._validate_config(config)
        self.manager.load_config(CONFIG_DIR / "fake_rig.json", load_overlays=False)
        self.manager.load_config(CONFIG_DIR / "Norwich_rig.json", load_overlays=False)

    def test_incomplete_amplifier_lists_are_rejected(self):
        config = deepcopy(self.config)
        config["devices"]["amplifier"] = [config["devices"]["amplifier"]]
        with self.assertRaisesRegex(RigConfigError, "amplifier"):
            self.manager._validate_config(config)

    def test_other_per_pipette_devices_still_need_one_entry_each(self):
        for slot in ("pipette_controller", "pipette_camera", "pressure"):
            with self.subTest(slot=slot):
                config = deepcopy(self.config)
                config["devices"][slot] = config["devices"][slot][0]
                with self.assertRaisesRegex(RigConfigError, slot):
                    self.manager._validate_config(config)


class AmplifierProtocolWorkersTests(unittest.TestCase):
    def test_shared_amplifier_operations_run_without_overlap(self):
        amplifier = object()
        keys = ["pipette_0", "pipette_1"]
        runner = SimpleNamespace(autopatchers={
            key: SimpleNamespace(amplifier=amplifier) for key in keys
        })
        second_started = threading.Event()
        events = []

        def operation(key):
            events.append(("start", key))
            if key == keys[0]:
                self.assertFalse(second_started.wait(timeout=.05))
            else:
                second_started.set()
            events.append(("finish", key))
            return key

        result, errors = AutoPatchInterface._run_amplifier_workers(runner, keys, operation)
        self.assertEqual(events, [("start", keys[0]), ("finish", keys[0]),
                                  ("start", keys[1]), ("finish", keys[1])])
        self.assertEqual(list(result), keys)
        self.assertEqual(result, dict(zip(keys, keys)))
        self.assertEqual(errors, {})

    def test_separate_amplifiers_can_still_prepare_in_parallel(self):
        keys = ["pipette_0", "pipette_1"]
        runner = SimpleNamespace(autopatchers={
            key: SimpleNamespace(amplifier=object()) for key in keys
        })
        both_running = threading.Barrier(2)

        def operation(key):
            both_running.wait(timeout=2)
            return key

        result, errors = AutoPatchInterface._run_amplifier_workers(runner, keys, operation)
        self.assertEqual(result, dict(zip(keys, keys)))
        self.assertEqual(errors, {})

    def test_headstages_of_one_multiclamp_run_one_after_another(self):
        multiclamp = object()
        keys = ["pipette_0", "pipette_1"]
        runner = SimpleNamespace(autopatchers={
            key: SimpleNamespace(amplifier=SimpleNamespace(shared_amplifier=multiclamp))
            for key in keys
        })
        active = []
        overlaps = []

        def operation(key):
            active.append(key)
            overlaps.append(len(active))
            threading.Event().wait(0.05)
            active.remove(key)
            return key

        result, _ = AutoPatchInterface._run_amplifier_workers(runner, keys, operation)
        self.assertEqual(result, dict(zip(keys, keys)))
        self.assertEqual(max(overlaps), 1)

    def test_one_failure_does_not_stop_the_other_pipette(self):
        keys = ["pipette_0", "pipette_1"]
        runner = SimpleNamespace(autopatchers={
            key: SimpleNamespace(amplifier=object()) for key in keys
        })

        def operation(key):
            if key == keys[0]:
                raise ValueError("amplifier refused")
            return key

        result, errors = AutoPatchInterface._run_amplifier_workers(runner, keys, operation)
        self.assertEqual(result, {keys[1]: keys[1]})
        self.assertIsInstance(errors[keys[0]], ValueError)


class MultiClampAmplifierTests(unittest.TestCase):
    """One 700B, one headstage (MultiClampChannel) per pipette; DLL stubbed out."""

    def _amplifier(self, channels=(1, 2)):
        from patcherbot.devices.amplifier import multiclamp

        class FakeChannel:
            def __init__(self, channel, **identification):
                self.channel = channel
                self.identification = identification

        with patch.object(multiclamp, "MultiClampChannel", FakeChannel):
            return multiclamp.MultiClampAmplifier(channels=channels)

    def test_each_pipette_gets_its_own_channel(self):
        amplifier = self._amplifier()
        self.assertEqual(amplifier.for_pipette("pipette_0").channel, 1)
        self.assertEqual(amplifier.for_pipette("pipette_1").channel, 2)
        self.assertIs(amplifier.for_pipette("pipette_1").shared_amplifier, amplifier)
        with self.assertRaises(KeyError):
            amplifier.for_pipette("pipette_2")

    def test_interface_pairs_each_pipette_with_its_headstage(self):
        amplifier = self._amplifier()
        devices = {"pipette_0": amplifier}        # one amplifier entry in the config
        self.assertEqual(AutoPatchInterface._device_for_pipette(devices, "pipette_0").channel, 1)
        self.assertEqual(AutoPatchInterface._device_for_pipette(devices, "pipette_1").channel, 2)

    def test_dll_calls_are_serialized(self):
        from patcherbot.devices.amplifier import multiclamp

        class Channel:
            selected = None
            inside = 0
            most = 0

            def select_amplifier(self):
                multiclamp.MultiClampChannel.selected_device = self

            @multiclamp.needs_select
            def call(self):
                Channel.inside += 1
                Channel.most = max(Channel.most, Channel.inside)
                threading.Event().wait(0.02)
                Channel.inside -= 1

        a, b = Channel(), Channel()
        threads = [threading.Thread(target=device.call) for device in (a, b, a, b)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(Channel.most, 1)


class ProtocolMatrixCleanupTests(unittest.TestCase):
    def test_failed_setup_still_cleans_up_every_selected_pipette(self):
        keys = ["pipette_0", "pipette_1"]
        finished = []
        daq_calls = []
        runner = SimpleNamespace(
            autopatchers={key: SimpleNamespace(amplifier=object()) for key in keys},
            protocol_selections={key: {name: name == "voltage_protocol"
                                       for name in AutoPatchInterface._PROTOCOL_RUNNERS}
                                 for key in keys},
            _PROTOCOL_RUNNERS=AutoPatchInterface._PROTOCOL_RUNNERS,
            daq=SimpleNamespace(getDataFromVoltageProtocol=lambda **kw: daq_calls.append(kw)),
            error=lambda message: None,
        )

        def prepare(key, protocol_name):
            if key == keys[1]:
                raise RuntimeError("MultiClamp channel 2 not responding")
            return [("getDataFromVoltageProtocol", {})]

        runner._prepare_protocol = prepare
        runner._finish_protocol = lambda key, protocol_name: finished.append(key)
        runner._run_amplifier_workers = lambda selected, operation: \
            AutoPatchInterface._run_amplifier_workers(runner, selected, operation)
        controller = SimpleNamespace(abort_if_requested=lambda: None,
                                     success_if_requested=lambda: None)

        with self.assertRaisesRegex(RuntimeError, "amplifier setup failed for pipette_1"):
            AutoPatchInterface._run_protocol_matrix(runner, controller)
        self.assertEqual(sorted(finished), keys)       # both amplifiers put back
        self.assertEqual(daq_calls, [])                # no acquisition with a half-set-up rig


if __name__ == "__main__":
    unittest.main()
