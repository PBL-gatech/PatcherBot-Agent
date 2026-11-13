"""Driving the secondary pipette: per-pipette config views, active-pipette
routing, per-pipette calibration files, and the pipette_0 collision rule.

No hardware, models or QApplication needed. Run with
``PYTHONPATH=. python tests/test_multipipette_motion.py``.
"""
import os
import pickle
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from patcherbot.configs.CalibrationConfig import CalibrationConfig, PER_PIPETTE_CALIBRATION_FIELDS
from patcherbot.controller.collision import CollisionError, PrimaryPipetteGuard
from patcherbot.controller.patch import AutoPatcher
from patcherbot.devices.manipulator.calibratedunit import CalibratedUnit
from patcherbot.interface import TaskInterface
from patcherbot.interface.patch import AutoPatchInterface
from patcherbot.interface.pipettes import PipetteInterface, _SAVED_POSITION_NAMES
from patcherbot.utils.multipipette_helper import PipetteConfigView


def rotation_z(degrees):
    theta = np.deg2rad(degrees)
    return np.array([[np.cos(theta), -np.sin(theta), 0.0],
                     [np.sin(theta), np.cos(theta), 0.0],
                     [0.0, 0.0, 1.0]])


class StubUnit:
    """What the guard and interface use of a CalibratedUnit."""

    def __init__(self, M, r0=(0.0, 0.0, 0.0), position=(0.0, 0.0, 0.0), calibrated=True):
        self.M = np.asarray(M, dtype=float)
        self.r0 = np.asarray(r0, dtype=float)
        self.pos = np.asarray(position, dtype=float)
        self.calibrated = calibrated
        self.must_be_recalibrated = False
        self._tip_reference = None
        self._trusted_tip_reference = None
        self.loaded = None

    def position(self):
        return self.pos.copy()

    def save_configuration(self):
        return {"M": self.M.copy(), "r0": self.r0.copy()}

    def load_configuration(self, config):
        self.loaded = config
        self.M = np.asarray(config["M"], dtype=float)
        self.calibrated = True


# Stage frame = image pixels minus stage reference. Stage at the origin,
# reference_position = M @ xy + r0 = 0.
STAGE = SimpleNamespace(M=np.eye(2), r0=np.zeros(2), calibrated=True,
                        save_configuration=lambda: {"M": np.eye(2)},
                        load_configuration=lambda config: None)


def two_pipettes(p0_position=(0.0, 0.0, 0.0), p1_position=(0.0, 0.0, 0.0)):
    """pipette_0: 2 px/um, axes aligned with the image. pipette_1: mirrored in
    x and y (180 deg about z), tip at image (600, 0) px when at its origin."""
    p0 = StubUnit(2 * rotation_z(0), position=p0_position)
    p1 = StubUnit(2 * rotation_z(180), r0=(600.0, 0.0, 0.0), position=p1_position)
    return {"pipette_0": p0, "pipette_1": p1}


class PipetteConfigViewTests(unittest.TestCase):
    def setUp(self):
        self.config = CalibrationConfig(name="Calibration")
        self.config.pipette_z_rotation = -20.0
        self.view0 = PipetteConfigView(self.config, 0, PER_PIPETTE_CALIBRATION_FIELDS)
        self.view1 = PipetteConfigView(self.config, 1, PER_PIPETTE_CALIBRATION_FIELDS)

    def test_falls_back_to_shared_values(self):
        self.assertEqual(self.view1.pipette_z_rotation, -20.0)
        self.assertEqual(self.view1.microscope_units_per_um, self.config.microscope_units_per_um)

    def test_override_is_read_for_that_pipette_only(self):
        self.config.pipettes = [{}, {"pipette_z_rotation": 160.0}]
        self.assertEqual(self.view1.pipette_z_rotation, 160.0)
        self.assertEqual(self.view0.pipette_z_rotation, -20.0)

    def test_writes_to_per_pipette_fields_stay_per_pipette(self):
        self.view1.home_position = (1.0, 2.0, 3.0)
        self.view0.home_position = (4.0, 5.0, 6.0)
        self.assertEqual(self.config.home_position, (4.0, 5.0, 6.0))
        self.assertEqual(self.view1.home_position, (1.0, 2.0, 3.0))
        self.assertEqual(self.config.pipettes[1]["home_position"], (1.0, 2.0, 3.0))

    def test_shared_fields_are_written_to_the_shared_config(self):
        self.view1.autofocus_dist = 42
        self.assertEqual(self.config.autofocus_dist, 42)


class GuardTests(unittest.TestCase):
    def guard(self, units):
        return PrimaryPipetteGuard(units, STAGE, "pipette_0")

    def test_secondary_maps_into_primary_frame(self):
        units = two_pipettes()
        guard = self.guard(units)
        # pipette_1 at its origin sits at image x=600 px = pipette_0 x=300 um.
        np.testing.assert_allclose(guard.to_primary_frame("pipette_1", [0, 0, 0]), [300, 0], atol=1e-9)
        # Moving pipette_1 +x moves it toward pipette_0 (mirrored).
        np.testing.assert_allclose(guard.to_primary_frame("pipette_1", [100, 0, 0]), [200, 0], atol=1e-9)

    def test_cell_left_of_primary_is_refused(self):
        guard = self.guard(two_pipettes(p0_position=(100.0, 0.0, 0.0)))
        # Cells are stored as stage_ref - image px; image x = -cell x.
        left_cell = np.array([-100.0, 0.0, 0.0])    # image x 100 px = p0 x 50 um
        right_cell = np.array([-400.0, 0.0, 0.0])   # image x 400 px = p0 x 200 um
        with self.assertRaises(CollisionError):
            guard.check_cell("pipette_1", left_cell)
        guard.check_cell("pipette_1", right_cell)

    def test_primary_is_never_blocked(self):
        guard = self.guard(two_pipettes(p0_position=(100.0, 0.0, 0.0)))
        guard.check_cell("pipette_0", np.array([-100.0, 0.0, 0.0]))
        guard.check_move("pipette_0", np.zeros(3), np.array([-1e4, 0.0, 0.0]))

    def test_uncalibrated_cell_check_fails_safe_but_moves_are_allowed(self):
        units = two_pipettes(p0_position=(100.0, 0.0, 0.0))
        units["pipette_0"].calibrated = False
        guard = self.guard(units)
        with self.assertRaises(CollisionError):
            guard.check_cell("pipette_1", np.array([-400.0, 0.0, 0.0]))
        guard.check_move("pipette_1", np.zeros(3), np.array([1000.0, 0.0, 0.0]))

    def test_move_into_primary_side_is_refused_and_retreat_allowed(self):
        guard = self.guard(two_pipettes(p0_position=(100.0, 0.0, 0.0)))
        # pipette_1 x=150 -> p0 frame x=150 (right of p0); x=250 -> p0 frame 50 (left).
        guard.check_move("pipette_1", np.zeros(3), np.array([150.0, 0, 0]))
        with self.assertRaises(CollisionError):
            guard.check_move("pipette_1", np.zeros(3), np.array([250.0, 0, 0]))
        # Already on the wrong side (p0 frame 50): backing out toward 100+ is fine,
        # going deeper (p0 frame 0) is not.
        guard.check_move("pipette_1", np.array([250.0, 0, 0]), np.array([200.0, 0, 0]))
        with self.assertRaises(CollisionError):
            guard.check_move("pipette_1", np.array([250.0, 0, 0]), np.array([300.0, 0, 0]))

    def test_tip_anchor_overrides_the_calibration_offset(self):
        units = two_pipettes(p0_position=(100.0, 0.0, 0.0))
        # pipette_0 detected at image (300, 0) px when at x=100 um: an anchor
        # 100 px right of what M/r0 predict (200 px).
        units["pipette_0"]._tip_reference = (np.array([300.0, 0.0]),
                                             np.array([100.0, 0.0, 0.0]), np.zeros(2))
        guard = self.guard(units)
        # Image x 350 px: right of the detected tip -> allowed (M/r0 alone also allow).
        guard.check_cell("pipette_1", np.array([-350.0, 0.0, 0.0]))
        # Image x 250 px: right of the M/r0 tip (200) but left of the detected one.
        with self.assertRaises(CollisionError):
            guard.check_cell("pipette_1", np.array([-250.0, 0.0, 0.0]))

    def test_stage_offset_cancels(self):
        units = two_pipettes(p0_position=(100.0, 0.0, 0.0))
        stage = SimpleNamespace(M=np.eye(2), r0=np.array([500.0, -40.0]))
        guard = PrimaryPipetteGuard(units, stage, "pipette_0")
        units["pipette_0"]._tip_reference = (np.array([700.0, -40.0]),   # 200 px + stage 500
                                             np.array([100.0, 0.0, 0.0]), np.zeros(2))
        with self.assertRaises(CollisionError):
            guard.check_cell("pipette_1", np.array([-100.0, 0.0, 0.0]))
        guard.check_cell("pipette_1", np.array([-400.0, 0.0, 0.0]))


class FakeDevice:
    """Records moves; position_group returns the last commanded position."""

    def __init__(self):
        self.pos = np.zeros(3)
        self.moves = []

    def position_group(self, axes):
        return self.pos.copy()

    def position(self, axis):
        return self.pos[axis - 1]

    def absolute_move(self, x, axis, speed=None):
        self.moves.append(("absolute", x, axis))

    def absolute_move_group(self, x, axes, speed=None):
        self.moves.append(("absolute_group", list(x), list(axes)))

    def relative_move(self, x, axis, speed=None):
        self.moves.append(("relative", x, axis))

    def relative_move_group(self, x, axes, speed=None):
        self.moves.append(("relative_group", list(x), list(axes)))


class MoveGuardHookTests(unittest.TestCase):
    def unit(self, guard):
        unit = CalibratedUnit.__new__(CalibratedUnit)
        unit.dev, unit.axes = FakeDevice(), [1, 2, 3]
        unit.move_guard = guard
        return unit

    def test_every_discrete_move_reports_its_full_target(self):
        seen = []
        unit = self.unit(lambda current, target: seen.append(target.tolist()))
        unit.dev.pos = np.array([1.0, 2.0, 3.0])
        unit.absolute_move(10.0, axis=0)
        unit.absolute_move([4.0, 5.0, 6.0])
        unit.absolute_move_group([7.0, 8.0], [0, 2])
        unit.relative_move(1.0, axis=1)
        unit.relative_move([1.0, 1.0, 1.0])
        unit.relative_move_group([0.0, 0.0, -1.0])
        self.assertEqual(seen, [[10, 2, 3], [4, 5, 6], [7, 2, 8], [1, 3, 3], [2, 3, 4], [1, 2, 2]])
        self.assertEqual(len(unit.dev.moves), 6)

    def test_refused_move_is_not_sent(self):
        def refuse(current, target):
            raise CollisionError("no")
        unit = self.unit(refuse)
        with self.assertRaises(CollisionError):
            unit.relative_move(5.0, axis=0)
        self.assertEqual(unit.dev.moves, [])

    def test_no_guard_passes_straight_through(self):
        unit = self.unit(None)
        unit.absolute_move(3.0, axis=2)
        self.assertEqual(unit.dev.moves, [("absolute", 3.0, 3)])


def make_pipette_interface(units, folder):
    interface = PipetteInterface.__new__(PipetteInterface)
    TaskInterface.__init__(interface)
    interface.calibrated_units = units
    interface.pipette_units = dict(units)
    interface.primary_pipette = next(iter(units))
    interface.active_pipette = interface.primary_pipette
    interface.busy_checks = []
    interface.calibrated_stage = STAGE
    interface.calibration_config = CalibrationConfig(name="Calibration")
    interface.microscope = SimpleNamespace(floor_Z=-50.0)
    interface.camera = SimpleNamespace(width=1000, height=1000)
    interface._saved_positions = {key: dict.fromkeys(_SAVED_POSITION_NAMES) for key in units}
    interface._pos_before_raise = dict.fromkeys(units)
    interface._tare_pipettes = {key: np.array([None] * 3) for key in units}
    interface._loaded_pipette_calibrations = {}
    interface.folder_path = folder + os.sep
    interface.observer = None
    return interface


def set_positions(interface, key, offset):
    positions = interface._saved_positions[key]
    positions.update(home=np.array([offset, 0.0, 0.0]), home_stage=np.array([0.0, 0.0, 10.0]),
                     safe=np.array([offset, 0.0, -100.0]), safe_stage=np.array([0.0, 0.0, 10.0]),
                     bath=np.array([offset, 50.0, 0.0]))


class PipetteInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.units = two_pipettes()
        self.interface = make_pipette_interface(self.units, self.tmp.name)
        import patcherbot.interface.pipettes as pipettes_module
        self._observer = pipettes_module.Observer
        pipettes_module.Observer = lambda controller: ("observer", controller.active_pipette)
        self.addCleanup(setattr, pipettes_module, "Observer", self._observer)

    def test_active_pipette_routes_unit_and_positions(self):
        interface = self.interface
        set_positions(interface, "pipette_0", 1.0)
        set_positions(interface, "pipette_1", 2.0)
        self.assertIs(interface.calibrated_unit, self.units["pipette_0"])
        interface.set_active_pipette("pipette_1")
        self.assertIs(interface.calibrated_unit, self.units["pipette_1"])
        self.assertEqual(interface.home_position[0], 2.0)
        self.assertEqual(interface.observer, ("observer", "pipette_1"))
        interface.cleaning_bath_position = np.array([9.0, 9.0, 9.0])
        self.assertEqual(interface.saved_positions("pipette_0")["bath"][0], 1.0)

    def test_switch_refused_while_busy(self):
        self.interface.busy_checks.append(lambda: True)
        with self.assertRaises(RuntimeError):
            self.interface.set_active_pipette("pipette_1")
        self.assertEqual(self.interface.active_pipette, "pipette_0")

    def test_calibration_round_trip_keeps_each_pipette(self):
        interface = self.interface
        set_positions(interface, "pipette_0", 1.0)
        set_positions(interface, "pipette_1", 2.0)
        interface.write_calibration()
        path = os.path.join(self.tmp.name, "calibration.pickle")
        with open(path, "rb") as stream:
            record = pickle.load(stream)
        self.assertEqual(record["version"], 2)
        self.assertEqual(set(record["pipettes"]), {"pipette_0", "pipette_1"})
        np.testing.assert_allclose(record["home"][:3], [1.0, 0.0, 0.0])  # legacy keys = pipette_0

        fresh_units = two_pipettes()
        for unit in fresh_units.values():
            unit.calibrated = False
        fresh = make_pipette_interface(fresh_units, self.tmp.name)
        self.assertEqual(fresh.read_calibration(path), ["pipette_0", "pipette_1"])
        np.testing.assert_allclose(fresh_units["pipette_1"].M, self.units["pipette_1"].M)
        self.assertEqual(fresh.saved_positions("pipette_1")["home"][0], 2.0)
        self.assertEqual(fresh.saved_positions("pipette_0")["bath"][0], 1.0)

    def test_legacy_file_loads_primary_only(self):
        path = os.path.join(self.tmp.name, "legacy.pickle")
        with open(path, "wb") as stream:
            pickle.dump({"manip": {"M": 2 * np.eye(3)}, "stage": {"M": np.eye(2)},
                         "home": np.zeros(6), "safe": np.ones(6), "bath": None,
                         "floor_Z": -10.0}, stream)
        for unit in self.units.values():
            unit.calibrated = False
        self.assertEqual(self.interface.read_calibration(path), ["pipette_0"])
        self.assertTrue(self.units["pipette_0"].calibrated)
        self.assertFalse(self.units["pipette_1"].calibrated)

    def test_write_requires_the_active_pipette_but_skips_others(self):
        interface = self.interface
        set_positions(interface, "pipette_0", 1.0)       # pipette_1 has no positions
        interface.write_calibration()
        interface.set_active_pipette("pipette_1")
        with self.assertRaises(RuntimeError):
            interface.write_calibration()


class AutoPatchRoutingTests(unittest.TestCase):
    def test_current_autopatcher_and_queue_follow_active_pipette(self):
        pipettes = SimpleNamespace(active_pipette="pipette_0")
        interface = AutoPatchInterface.__new__(AutoPatchInterface)
        interface.pipette_controller = pipettes
        interface.autopatchers = {"pipette_0": "ap0", "pipette_1": "ap1"}
        interface.cell_queues = {"pipette_0": [], "pipette_1": []}
        interface.cells_to_patch.append("cell A")
        pipettes.active_pipette = "pipette_1"
        self.assertEqual(interface.current_autopatcher, "ap1")
        self.assertEqual(interface.cells_to_patch, [])
        interface.cells_to_patch = ["cell B"]
        self.assertEqual(interface.cell_queues, {"pipette_0": ["cell A"], "pipette_1": ["cell B"]})

    def test_autopatcher_checks_cells_with_its_own_key(self):
        guard = PrimaryPipetteGuard(two_pipettes(p0_position=(100.0, 0.0, 0.0)), STAGE)
        autopatcher = AutoPatcher.__new__(AutoPatcher)
        autopatcher.collision_guard, autopatcher.pipette_key = guard, "pipette_1"
        left = (np.array([-100.0, 0.0, 0.0]), None, None)
        with self.assertRaises(CollisionError):
            autopatcher.check_cell_allowed(left)
        autopatcher.pipette_key = "pipette_0"
        autopatcher.check_cell_allowed(left)


if __name__ == "__main__":
    unittest.main()
