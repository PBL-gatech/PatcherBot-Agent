"""multipipette_helper: per-pipette state, key resolution, device lookup and
the DAQ view's pipette-key injection. No hardware or Qt needed."""
import collections
import threading
import unittest
from types import SimpleNamespace

from patcherbot.utils.multipipette_helper import (
    PER_PIPETTE_FIELDS, MultiPipetteDevice, PipetteDAQView, PipetteState,
    build_channel_map, device_for_pipette, multi_pipette_protocol, per_pipette,
    pipette_index, pipette_key_for, pipette_label, takes_pipettes,
)


class KeyTests(unittest.TestCase):
    def test_keys_and_labels(self):
        self.assertEqual(pipette_key_for(2), "pipette_2")
        self.assertEqual(pipette_index("pipette_2"), 2)
        self.assertIsNone(pipette_index("ai17"))
        self.assertEqual(pipette_label("pipette_0"), "Pipette 0")


class PipetteStateTests(unittest.TestCase):
    def test_defaults_and_one_queue_per_pipette(self):
        first, second = PipetteState(), PipetteState()
        self.assertFalse(first.cellMode)
        self.assertIsNone(first.current_protocol_data)
        self.assertIsInstance(first.optogenetic_protocol_queue, collections.deque)
        self.assertIsNot(first.optogenetic_protocol_queue, second.optogenetic_protocol_queue)
        self.assertEqual(set(first.as_dict()), set(PER_PIPETTE_FIELDS))

    def test_unknown_field_is_rejected(self):
        state = PipetteState()
        with self.assertRaises(AttributeError):
            state.cellmode = True          # typo of cellMode

    def test_custom_fields(self):
        state = PipetteState({"pressure_mbar": 0.0, "history": list})
        state.pressure_mbar = 25.0
        self.assertEqual(state.as_dict(), {"pressure_mbar": 25.0, "history": []})


class MultiPipetteDeviceTests(unittest.TestCase):
    def setUp(self):
        self.device = MultiPipetteDevice(["pipette_0", "pipette_1"],
                                         state_fields=PER_PIPETTE_FIELDS)

    def test_default_and_duplicate_keys(self):
        self.assertEqual(MultiPipetteDevice().pipette_keys, ["pipette_0"])
        with self.assertRaises(ValueError):
            MultiPipetteDevice(["pipette_0", "pipette_0"])

    def test_resolve_pipettes(self):
        self.assertEqual(self.device.resolve_pipettes(None), ["pipette_0"])
        self.assertEqual(self.device.resolve_pipettes("pipette_1"), ["pipette_1"])
        self.assertEqual(self.device.resolve_pipettes(["pipette_1", "pipette_0", "pipette_1"]),
                         ["pipette_1", "pipette_0"])
        with self.assertRaises(KeyError):
            self.device.resolve_pipettes(["pipette_2"])

    def test_each_pipette_has_its_own_state(self):
        self.device.pipette_state("pipette_1").cellMode = True
        self.assertFalse(self.device.pipette_state("pipette_0").cellMode)
        self.assertFalse(self.device.pipette_state().cellMode)      # None = primary
        self.assertTrue(self.device.pipette_state("pipette_1").cellMode)


class _FakeDAQ(MultiPipetteDevice):
    """Smallest DAQ-like object the view needs."""

    def __init__(self):
        super().__init__(["pipette_0", "pipette_1"], state_fields=PER_PIPETTE_FIELDS)
        self._protocol_lock = threading.RLock()
        self.calls = []
        self.shared_setting = None

    @multi_pipette_protocol
    def protocol(self, keys, *, amplitude=1.0):
        self.calls.append(("protocol", keys, amplitude))
        return {key: amplitude for key in keys}

    @takes_pipettes
    def square_wave(self, rate, pipettes=None):
        self.calls.append(("square_wave", rate, pipettes))
        return pipettes

    @per_pipette
    def resistance(self, pipette_key=None):
        return self.pipette_state(pipette_key).totalResistance

    @per_pipette
    def set_mode(self, mode, pipette_key=None):
        self.pipette_state(pipette_key).cellMode = mode

    def shared_method(self):
        return "shared"


class PipetteDAQViewTests(unittest.TestCase):
    def setUp(self):
        self.daq = _FakeDAQ()
        self.view = PipetteDAQView(self.daq, "pipette_1")

    def test_rejects_unknown_pipette(self):
        with self.assertRaises(KeyError):
            PipetteDAQView(self.daq, "pipette_5")

    def test_fields_read_and_write_this_pipettes_state(self):
        self.view.totalResistance = 7e6
        self.assertEqual(self.daq.pipette_state("pipette_1").totalResistance, 7e6)
        self.assertIsNone(self.daq.pipette_state("pipette_0").totalResistance)
        self.assertEqual(self.view.totalResistance, 7e6)

    def test_protocol_gets_this_pipette(self):
        self.assertEqual(self.view.protocol(amplitude=2.0), {"pipette_1": 2.0})
        self.assertEqual(self.daq.calls[-1], ("protocol", ["pipette_1"], 2.0))
        # An explicit pipettes= list is left alone.
        self.view.protocol(pipettes=["pipette_0"])
        self.assertEqual(self.daq.calls[-1][1], ["pipette_0"])

    def test_takes_pipettes_method_gets_this_pipette(self):
        self.assertEqual(self.view.square_wave(50_000), ["pipette_1"])

    def test_per_pipette_methods_get_this_key(self):
        self.view.set_mode(True)                       # positional mode, injected key
        self.assertTrue(self.daq.pipette_state("pipette_1").cellMode)
        self.assertFalse(self.daq.pipette_state("pipette_0").cellMode)
        self.daq.pipette_state("pipette_1").totalResistance = 5e6
        self.assertEqual(self.view.resistance(), 5e6)
        self.assertEqual(self.view.resistance(None), 5e6)   # explicit None = this pipette
        self.assertIsNone(self.view.resistance("pipette_0"))

    def test_wrapped_methods_expose_the_shared_daq(self):
        # TaskInterface.execute() finds the TaskController through __self__.
        self.assertIs(self.view.set_mode.__self__, self.daq)
        self.assertIs(self.view.shared_method.__self__, self.daq)
        self.assertEqual(self.view.shared_method(), "shared")

    def test_other_attributes_are_shared(self):
        self.view.shared_setting = 3
        self.assertEqual(self.daq.shared_setting, 3)


class DeviceForPipetteTests(unittest.TestCase):
    def test_exact_key_single_device_and_for_pipette(self):
        first, second = object(), object()
        self.assertIs(device_for_pipette({"pipette_0": first, "pipette_1": second}, "pipette_1"),
                      second)
        self.assertIs(device_for_pipette({"pipette_0": first}, "pipette_3"), first)
        with self.assertRaises(KeyError):
            device_for_pipette({"pipette_0": first, "pipette_1": second}, "pipette_2")
        shared = SimpleNamespace(for_pipette=lambda key: f"part for {key}")
        self.assertEqual(device_for_pipette({"pipette_0": shared}, "pipette_1"),
                         "part for pipette_1")


class ChannelMapTests(unittest.TestCase):
    def test_build_channel_map(self):
        channel_map = build_channel_map(
            ["pipette_0", "pipette_1"],
            channels=[{"read": "Dev1/ai19", "resp": "Dev1/ai17", "cmd": "Dev1/ao0"},
                      {"read": "Dev1/ai23", "resp": "Dev1/ai21", "cmd": "Dev1/ao3"}])
        self.assertEqual(list(channel_map), ["pipette_0", "pipette_1"])
        self.assertEqual(channel_map["pipette_1"].resp, "Dev1/ai21")
        with self.assertRaises(ValueError):
            build_channel_map(["pipette_0", "pipette_1"],
                              channels=[{"read": "ai0", "resp": "ai1", "cmd": "ao0"}] * 2)


if __name__ == "__main__":
    unittest.main()
