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
            {"read": "Dev1/ai19", "resp": "Dev1/ai17", "cmd": "Dev1/ao0"},
            {"read": "Dev1/ai23", "resp": "Dev1/ai21", "cmd": "Dev1/ao3"},
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
                    self.assertEqual(devices["daq"].channel_map["pipette_1"].resp, "Dev1/ai21")

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

    def test_headstages_of_one_multiclamp_overlap_their_settling_waits(self):
        multiclamp = object()
        keys = ["pipette_0", "pipette_1"]
        runner = SimpleNamespace(autopatchers={
            key: SimpleNamespace(amplifier=SimpleNamespace(shared_amplifier=multiclamp))
            for key in keys
        })
        both_running = threading.Barrier(2)

        def operation(key):
            both_running.wait(timeout=2)
            return key

        result, errors = AutoPatchInterface._run_amplifier_workers(runner, keys, operation)
        self.assertEqual(result, dict(zip(keys, keys)))
        self.assertEqual(errors, {})

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

    def test_numpy_channel_array_is_accepted(self):
        import numpy as np
        amplifier = self._amplifier(channels=np.array([1., 2.]))
        self.assertEqual(amplifier.channels, [1, 2])
        self.assertTrue(all(type(channel) is int for channel in amplifier.channels))

    def test_norwich_config_builds_through_the_rig_loader(self):
        """The loader turns ``"channels": [1, 2]`` into a numpy array; the
        amplifier must still build with one headstage per pipette."""
        from patcherbot.devices.amplifier import multiclamp

        class FakeChannel:
            def __init__(self, channel, **identification):
                self.channel = channel
                self.identification = identification

        manager = RigConfigManager()
        config = manager.load_config(CONFIG_DIR / "Norwich_multiple_rig.json", load_overlays=False)
        with patch.object(multiclamp, "MultiClampChannel", FakeChannel):
            amplifier = manager._instantiate_slot("amplifier", config["devices"]["amplifier"], {})
        self.assertEqual(amplifier.channels, [1, 2])
        self.assertEqual(amplifier.for_pipette("pipette_0").channel, 1)
        self.assertEqual(amplifier.for_pipette("pipette_1").channel, 2)
        self.assertIs(type(amplifier.for_pipette("pipette_1").channel), int)

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


class FakeMultiClampDLL:
    """Stands in for AxMultiClampMsg.dll: one 700B with headstages 1 and 2.

    Like the real DLL, it remembers per message handler which Commander
    windows FindFirst/FindNext found, and SelectMultiClamp on a handler that
    never searched fails with MCCMSG_ERROR_MCCNOTOPEN (6002).
    """
    SERIAL = b"00123456"

    def __init__(self):
        self.created = []
        self.destroyed = []
        self.searched = set()
        self.selects = []           # (handle, channel)
        self._pending = {}          # handle -> channels left for FindNext

    def MCCMSG_CreateObject(self, err):
        handle = len(self.created) + 1
        self.created.append(handle)
        err._obj.value = 6000
        return handle

    def MCCMSG_DestroyObject(self, handle):
        self.destroyed.append(handle)

    def _report(self, handle, model, serial, channel, err):
        if not self._pending[handle]:
            err._obj.value = 6002
            return 0
        model._obj.value = 1                        # 700B
        serial.value = self.SERIAL
        channel._obj.value = self._pending[handle].pop(0)
        err._obj.value = 6000
        return 1

    def MCCMSG_FindFirstMultiClamp(self, handle, model, serial, size, port, device, channel, err):
        self.searched.add(handle)
        self._pending[handle] = [1, 2]
        return self._report(handle, model, serial, channel, err)

    def MCCMSG_FindNextMultiClamp(self, handle, model, serial, size, port, device, channel, err):
        return self._report(handle, model, serial, channel, err)

    def MCCMSG_SelectMultiClamp(self, handle, model, serial, port, device, channel, err):
        self.selects.append((handle, channel.value))
        if handle not in self.searched or serial != self.SERIAL:
            err._obj.value = 6002
            return 0
        err._obj.value = 6000
        return 1

    def MCCMSG_BuildErrorText(self, handle, err, buf, size):
        buf.value = b"AXMCCMSG: MultiClamp Commander is not open"

    def __getattr__(self, name):                    # every other MCCMSG_* call succeeds
        if name.startswith("MCCMSG_"):
            return lambda *args: 1
        raise AttributeError(name)


class MultiClampSharedHandleTests(unittest.TestCase):
    """Both headstages must talk to Commander through one searched handler."""

    def setUp(self):
        from patcherbot.devices.amplifier import multiclamp
        self.multiclamp = multiclamp
        self.dll = FakeMultiClampDLL()
        self._reset()
        self._patch = patch.object(multiclamp.ctypes, "WinDLL", create=True,
                                   return_value=self.dll)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._reset()

    def _reset(self):
        cls = self.multiclamp.MultiClampChannel
        cls.all_devices = None
        cls.selected_device = None
        cls._shared_dll = None
        cls._shared_handle = None
        cls._handle_users = 0

    def test_two_headstages_share_the_searched_handler(self):
        amplifier = self.multiclamp.MultiClampAmplifier(channels=[1, 2])
        self.assertEqual(len(self.dll.created), 1)
        handle = self.dll.created[0]
        self.assertEqual({h for h, _ in self.dll.selects}, {handle})
        self.assertEqual([c for _, c in self.dll.selects], [1, 2])

        amplifier.for_pipette("pipette_0").voltage_clamp()
        amplifier.for_pipette("pipette_1").current_clamp()
        self.assertEqual(self.dll.selects[-2:], [(handle, 1), (handle, 2)])

        amplifier.close()
        self.assertEqual(self.dll.destroyed, [handle])
        cls = self.multiclamp.MultiClampChannel
        self.assertIsNone(cls._shared_handle)
        self.assertIsNone(cls.all_devices)
        self.assertEqual(cls._handle_users, 0)

    def test_single_channel_config_still_works(self):
        channel = self.multiclamp.MultiClampChannel(channel=1)
        self.assertEqual(self.dll.selects, [(self.dll.created[0], 1)])
        channel.close()
        self.assertEqual(self.dll.destroyed, self.dll.created)

    def test_failure_names_the_channel_and_releases_the_handler(self):
        with self.assertRaisesRegex(RuntimeError, "MultiClamp channel 3"):
            self.multiclamp.MultiClampAmplifier(channels=[1, 3])
        self.assertEqual(self.dll.destroyed, self.dll.created)
        self.assertEqual(self.multiclamp.MultiClampChannel._handle_users, 0)


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
