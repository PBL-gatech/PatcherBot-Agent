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

    def test_norwich_uses_channel_one_with_both_daq_signal_paths(self):
        amplifier = self.config["devices"]["amplifier"]
        self.assertIsInstance(amplifier, dict)
        self.assertEqual(amplifier["params"], {"channel": 1})
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

        result = AutoPatchInterface._run_amplifier_workers(runner, keys, operation)
        self.assertEqual(events, [("start", keys[0]), ("finish", keys[0]),
                                  ("start", keys[1]), ("finish", keys[1])])
        self.assertEqual(list(result), keys)
        self.assertEqual(result, dict(zip(keys, keys)))

    def test_separate_amplifiers_can_still_prepare_in_parallel(self):
        keys = ["pipette_0", "pipette_1"]
        runner = SimpleNamespace(autopatchers={
            key: SimpleNamespace(amplifier=object()) for key in keys
        })
        both_running = threading.Barrier(2)

        def operation(key):
            both_running.wait(timeout=2)
            return key

        result = AutoPatchInterface._run_amplifier_workers(runner, keys, operation)
        self.assertEqual(result, dict(zip(keys, keys)))


if __name__ == "__main__":
    unittest.main()
