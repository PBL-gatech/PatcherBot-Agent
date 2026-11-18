"""Driving several pipettes: per-pipette calibration values, active-pipette
routing, per-pipette calibration files, the collision guard, and
compatibility with tools that read pipette_0's data.

No hardware, models or QApplication needed. Run with
``PYTHONPATH=. python tests/test_multipipette_motion.py``.
"""
import csv
import logging
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from patcherbot.configs.CalibrationConfig import CalibrationConfig
from patcherbot.controller.collision import (
    CollisionError, GuardedPipetteDevice, WorkspaceGuard, boundary_margin, boundary_normals,
    move_allowed,
)
from patcherbot.controller.patch import AutoPatcher
from patcherbot.configs.Config import EditorGrid
from patcherbot.devices.manipulator.calibratedunit import CalibratedUnit
from patcherbot.devices.manipulator.manipulatorunit import ManipulatorUnit
from patcherbot.interface import TaskInterface
from patcherbot.interface.patch import AutoPatchInterface
from patcherbot.interface.pipettes import PipetteInterface, _SAVED_POSITION_NAMES
from patcherbot.utils.experiment_book import ExperimentBookLogger
from patcherbot.utils.log_utils import LoggingObject
from patcherbot.utils.StateMachineLogger import StateMachineLogger
from patcherbot.utils.multipipette_helper import (
    PIPETTE_GEOMETRY_FIELDS, PipetteConfigView, validate_pipette_geometry,
)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments" / "Analysis" / "Rig_Recorder" / "Reconstruct"))
from recorder_geometry import is_secondary_pipette_log_row  # noqa: E402


def rotation_z(degrees):
    theta = np.deg2rad(degrees)
    return np.array([[np.cos(theta), -np.sin(theta), 0.0],
                     [np.sin(theta), np.cos(theta), 0.0],
                     [0.0, 0.0, 1.0]])


GEOMETRY = dict(stage_x_axis_flip=False, stage_y_axis_flip=False, pipette_y_rotation=25,
                pipette_k_scale=1.0, clean_move_order=["y", "x", "z"], safe_position_delta_um=-8000)


class StubUnit:
    """What the guard and interface use of a CalibratedUnit (1 px = 1 um stage)."""

    def __init__(self, M, tip_px, position=(0.0, 0.0, 0.0), calibrated=True, z_rotation=0.0):
        self.M = np.asarray(M, dtype=float)
        self.r0 = np.zeros(3)
        self.pos = np.asarray(position, dtype=float)
        self.calibrated = calibrated
        self.must_be_recalibrated = False
        # Tip seen at image ``tip_px`` when the manipulator was at its origin.
        self._tip_reference = (np.asarray(tip_px, dtype=float), np.zeros(3), np.zeros(2))
        self._trusted_tip_reference = None
        # Geometry consistent with M (stage 1 px = 1 um): G = inv(M) = Rz(-angle of M).
        self.config = SimpleNamespace(pipette_z_rotation=z_rotation, pipette_k_scale=1.0,
                                      stage_x_axis_flip=False, stage_y_axis_flip=False)
        self.loaded = None

    def position(self):
        return self.pos.copy()

    def save_configuration(self):
        return {"M": self.M.copy(), "r0": self.r0.copy()}

    def load_configuration(self, config):
        self.loaded = config
        self.M = np.asarray(config["M"], dtype=float)
        self.calibrated = True


STAGE = SimpleNamespace(M=np.eye(2), r0=np.zeros(2), calibrated=True,
                        save_configuration=lambda: {"M": np.eye(2)},
                        load_configuration=lambda config: None)
CONFIG = SimpleNamespace(collision_tolerance_um=100.0, collision_velocity_lookahead_s=0.5)


def rig(p0=(0.0, 0.0, 0.0), p1=(0.0, 0.0, 0.0), with_y_pipette=False):
    """pipette_0 on the left (world = its xy), pipette_1 on the right, mirrored
    (world = (1000, 0) - its xy). Optional pipette_2 above, coming in from the top."""
    units = {"pipette_0": StubUnit(rotation_z(0), (0, 0), p0, z_rotation=0),
             "pipette_1": StubUnit(rotation_z(180), (1000, 0), p1, z_rotation=180)}
    if with_y_pipette:
        # Comes in from above: manipulator +x points down the image, shaft runs up (+y).
        units["pipette_2"] = StubUnit(rotation_z(-90), (500, 800), z_rotation=90)
    return units


def guard_for(units, **kwargs):
    return WorkspaceGuard(units, STAGE, CONFIG, **kwargs)


def at(x, y=0.0):
    return np.array([x, y, 0.0])


# ------------------------------------------------------------------
# Per-pipette calibration values
# ------------------------------------------------------------------
class PipetteConfigViewTests(unittest.TestCase):
    def setUp(self):
        self.config = CalibrationConfig(name="Calibration")
        self.config.pipette_z_rotation = -60.0     # top-level: unused in keyed files
        self.config.pipettes = {"pipette_0": dict(GEOMETRY, pipette_z_rotation=-20.0),
                                "pipette_1": dict(GEOMETRY, pipette_z_rotation=160.0)}

    def test_each_pipette_reads_only_its_own_entry(self):
        view0 = PipetteConfigView(self.config, "pipette_0")
        view1 = PipetteConfigView(self.config, "pipette_1")
        self.assertEqual(view0.pipette_z_rotation, -20.0)
        self.assertEqual(view1.pipette_z_rotation, 160.0)
        self.assertEqual(view1.microscope_units_per_um, self.config.microscope_units_per_um)

    def test_missing_geometry_never_falls_back(self):
        del self.config.pipettes["pipette_1"]["pipette_y_rotation"]
        with self.assertRaises(AttributeError):
            PipetteConfigView(self.config, "pipette_1").pipette_y_rotation

    def test_positions_are_per_pipette_and_start_unset(self):
        self.config.home_position = (9.0, 9.0, 9.0)
        view0 = PipetteConfigView(self.config, "pipette_0")
        view1 = PipetteConfigView(self.config, "pipette_1")
        self.assertEqual(view1.home_position, (0.0, 0.0, 0.0))
        view1.home_position = (1.0, 2.0, 3.0)
        view0.home_position = (4.0, 5.0, 6.0)
        self.assertEqual(view1.home_position, (1.0, 2.0, 3.0))
        self.assertEqual(view0.home_position, (4.0, 5.0, 6.0))
        self.assertEqual(self.config.home_position, (9.0, 9.0, 9.0))

    def test_editor_shows_pipettes_side_by_side(self):
        self.config.show_pipettes_in_editor(["pipette_0", "pipette_1"])
        categories = dict(self.config.categories)
        self.assertNotIn("Pipette 1 Calibration", categories)
        grids = [field for field in categories["Pipette Calibration"] if isinstance(field, EditorGrid)]
        self.assertEqual(len(grids), 1)
        grid = grids[0]
        self.assertEqual(grid.columns, ["pipette_0", "pipette_1"])
        self.assertEqual(grid.column_labels, ["Pipette 0", "Pipette 1"])
        self.assertIn("stage_y_axis_flip", grid.rows)
        self.assertIn("pipette_1.pipette_z_rotation", grid.names())
        # Shared rows stay; unused top-level per-pipette values don't show anywhere.
        self.assertIn("pipette_diag_move", categories["Pipette Calibration"])
        self.assertNotIn("pipette_z_rotation", categories["Pipette Calibration"])
        self.assertNotIn("stage_y_axis_flip", categories["Stage Calibration"])
        self.assertEqual(self.config.editor_get("pipette_1.pipette_z_rotation"), 160.0)
        self.config.editor_set("pipette_1.pipette_z_rotation", 155.0)
        self.config.editor_set("pipette_0.stage_y_axis_flip", True)
        self.assertEqual(self.config.pipettes["pipette_1"]["pipette_z_rotation"], 155.0)
        self.assertEqual(self.config.pipettes["pipette_0"]["pipette_z_rotation"], -20.0)
        self.assertIs(self.config.pipettes["pipette_0"]["stage_y_axis_flip"], True)
        self.assertIs(self.config.pipettes["pipette_1"]["stage_y_axis_flip"], False)
        self.assertIs(self.config.editor_param("pipette_1.pipette_k_scale"),
                      self.config.param["pipette_k_scale"])
        self.assertEqual(len(self.config.editor_names_for_change("pipettes")),
                         2 * len(grid.rows))

    def test_legacy_file_is_pipette_0_only(self):
        config = CalibrationConfig(name="Calibration")
        config.pipette_z_rotation = -17.0
        self.assertEqual(PipetteConfigView(config, "pipette_0").pipette_z_rotation, -17.0)
        with self.assertRaises(ValueError):
            PipetteConfigView(config, "pipette_1")


class ValidatePipetteGeometryTests(unittest.TestCase):
    keys2 = ["pipette_0", "pipette_1"]

    def entries(self, *keys):
        return {"pipettes": {key: dict(GEOMETRY, pipette_z_rotation=float(i * 180 - 20))
                             for i, key in enumerate(keys)}}

    def assert_message(self, calibration, keys, *fragments):
        with self.assertRaises(ValueError) as caught:
            validate_pipette_geometry(calibration, keys, "Test_cal.yaml")
        for fragment in ("Test_cal.yaml",) + fragments:
            self.assertIn(fragment, str(caught.exception))

    def test_legacy_file_with_two_pipettes(self):
        self.assert_message({"pipette_z_rotation": -20}, self.keys2, "no `pipettes:` section", "pipette_1")

    def test_missing_pipette_entry(self):
        self.assert_message(self.entries("pipette_0"), self.keys2, "only defines pipette_0", "`pipette_1:`")

    def test_missing_fields_are_listed(self):
        calibration = self.entries("pipette_0", "pipette_1")
        del calibration["pipettes"]["pipette_1"]["pipette_k_scale"]
        self.assert_message(calibration, self.keys2, "pipettes.pipette_1", "pipette_k_scale")

    def test_typo_and_bad_key(self):
        calibration = self.entries("pipette_0", "pipette_1")
        calibration["pipettes"]["pipette_1"]["pipette_z_rotaton"] = 1
        self.assert_message(calibration, self.keys2, "pipette_z_rotaton")
        self.assert_message({"pipettes": {"left": GEOMETRY}}, ["pipette_0"], "'left'")

    def test_duplicate_rotation_warns(self):
        calibration = self.entries("pipette_0", "pipette_1")
        calibration["pipettes"]["pipette_1"]["pipette_z_rotation"] = -20.0
        warnings = validate_pipette_geometry(calibration, self.keys2, "Test_cal.yaml")
        self.assertTrue(warnings and "same pipette_z_rotation" in warnings[0])

    def test_rig_calibration_files(self):
        for name, count in (("Norwich", 2), ("fake", 3), ("bo", 1), ("moscow", 1)):
            config = CalibrationConfig(name="Calibration")
            config.from_file(str(REPO / "rig_setup" / "cal_configs" / f"{name}_cal.yaml"))
            validate_pipette_geometry(config, [f"pipette_{i}" for i in range(count)], name)
        with self.assertRaises(ValueError):
            validate_pipette_geometry(config, self.keys2, "moscow_cal.yaml")


# ------------------------------------------------------------------
# Collision guard (line through the other tip)
# ------------------------------------------------------------------
class RuleTests(unittest.TestCase):
    def test_margin_and_symmetry(self):
        # Line through the other tip (x = 1000), 100 um tolerance -> limit at 900.
        self.assertAlmostEqual(boundary_margin(at(0), at(1000), at(400), 100), 500)
        self.assertAlmostEqual(boundary_margin(at(1000), at(0), at(600), 100), 500)
        self.assertTrue(move_allowed(at(0), at(890), at(1000), 100))
        self.assertFalse(move_allowed(at(0), at(910), at(1000), 100))
        self.assertTrue(move_allowed(at(1000), at(110), at(0), 100))
        self.assertFalse(move_allowed(at(1000), at(90), at(0), 100))

    def test_line_is_perpendicular_to_the_tips(self):
        # Sideways moves are fine as long as they stay on the near side of the line.
        self.assertTrue(move_allowed(at(0), at(850, 3000), at(1000), 100))
        # Tilted pair: line through (1000, 1000) perpendicular to the diagonal.
        self.assertTrue(move_allowed(at(0), at(1300, 500), at(1000, 1000), 100))
        self.assertFalse(move_allowed(at(0), at(1300, 800), at(1000, 1000), 100))

    def test_line_never_cuts_the_other_shaft(self):
        # pipette_1's tip at (1000, 1000), shaft running +x back to its manipulator.
        other, shaft = at(1000, 1000), np.array([1.0, 0.0])
        # Active coming from the lower left: the shaft is on the far side, only
        # the plain perpendicular applies.
        normals = boundary_normals(at(0), other, shaft)
        self.assertEqual(len(normals), 1)
        np.testing.assert_allclose(normals[0], [0.7071, 0.7071], atol=1e-4)
        # Active at the lower right: the perpendicular (x - y = 0) would cut the
        # shaft and let the active tip reach (1600, 1400), above the shaft.
        active = at(2000, 0)
        self.assertTrue(move_allowed(active, at(1600, 1400), other, 100))          # plain rule
        normals = boundary_normals(active, other, shaft)
        np.testing.assert_allclose(normals[1], [0.0, 1.0], atol=1e-9)              # shaft line
        self.assertFalse(move_allowed(active, at(1600, 1400), other, 100, shaft))
        self.assertTrue(move_allowed(active, at(1600, 890), other, 100, shaft))
        self.assertFalse(move_allowed(active, at(1600, 910), other, 100, shaft))
        # Only the shaft-side half changes: on the other side of the tip-to-tip
        # line the plain perpendicular still applies, so (500, 400) -- past it --
        # stays refused even though it is well below the shaft.
        self.assertFalse(move_allowed(active, at(500, 400), other, 100))
        self.assertFalse(move_allowed(active, at(500, 400), other, 100, shaft))
        self.assertTrue(move_allowed(active, at(717, 435), other, 100, shaft))
        # Active tip right over the shaft (only after a manual move): it can step off it.
        self.assertTrue(move_allowed(at(1500, 1000), at(1500, 800), other, 100, shaft))

    def test_cannot_get_past_the_other_tip(self):
        active, other = at(0), at(1000)
        for _ in range(5):
            target = at(1000 - 100)                       # furthest allowed point
            self.assertTrue(move_allowed(active, target, other, 100))
            self.assertFalse(move_allowed(target, at(901), other, 100))
            active = target

    def test_already_too_close_may_not_get_worse(self):
        self.assertTrue(move_allowed(at(980), at(970), at(1000), 100))
        self.assertFalse(move_allowed(at(980), at(990), at(1000), 100))


class GuardTests(unittest.TestCase):
    def test_primary_cannot_pass_secondary_and_vice_versa(self):
        guard = guard_for(rig())
        guard.check_move("pipette_0", at(0), at(890))
        with self.assertRaises(CollisionError):
            guard.check_move("pipette_0", at(0), at(910))
        guard.check_move("pipette_1", at(0), at(890))      # world 110
        with self.assertRaises(CollisionError):
            guard.check_move("pipette_1", at(0), at(910))  # world 90

    def test_guard_uses_calibrated_shaft_directions(self):
        guard = guard_for(rig())
        # rig(): pipette_0 on the left (shaft -x), pipette_1 mirrored on the right (shaft +x).
        np.testing.assert_allclose(guard.shaft_direction("pipette_0"), [-1.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(guard.shaft_direction("pipette_1"), [1.0, 0.0], atol=1e-9)
        # pipette_0 at the lower right of pipette_1's tip may not climb over its shaft.
        units = rig(p0=(2000.0, 0.0, 0.0))
        units["pipette_1"]._tip_reference = (np.array([1000.0, 1000.0]), np.zeros(3), np.zeros(2))
        guard = guard_for(units)
        with self.assertRaises(CollisionError) as caught:
            guard.check_move("pipette_0", at(2000), at(1600, 1400))
        self.assertIn("shaft", str(caught.exception))
        guard.check_move("pipette_0", at(2000), at(1600, 800))

    def test_y_aligned_pipette_is_respected(self):
        guard = guard_for(rig(with_y_pipette=True))
        with self.assertRaises(CollisionError):
            guard.check_move("pipette_0", at(0), at(500, 700))    # behind pipette_2's tip
        guard.check_move("pipette_0", at(0), at(300, -200))

    def test_group_move_checks_box_corners(self):
        units = rig()
        units["pipette_1"]._tip_reference = (np.array([500.0, 500.0]), np.zeros(3), np.zeros(2))
        guard = guard_for(units)
        guard.config = SimpleNamespace(collision_tolerance_um=0.0, collision_velocity_lookahead_s=0.5)
        # Limit: x + y < 1000. The target (1100, -200) is inside, the corner (1100, 0) isn't.
        self.assertTrue(move_allowed(at(0), at(1100, -200), at(500, 500), 0))
        with self.assertRaises(CollisionError):
            guard.check_move("pipette_0", at(0), at(1100, -200))

    def test_velocity_lookahead(self):
        guard = guard_for(rig())
        with self.assertRaises(CollisionError):
            guard.check_velocity("pipette_0", at(0), [2000.0, 0.0, 0.0])   # 1000 um in 0.5 s
        guard.check_velocity("pipette_0", at(0), [0.0, 0.0, 50.0])          # Z only
        guard.check_velocity("pipette_0", at(0), [0.0, 0.0, 0.0])           # stop

    def test_unlocated_pipettes(self):
        units = rig()
        units["pipette_1"].calibrated = False
        guard = guard_for(units)
        guard.check_move("pipette_0", at(0), at(2000))      # pipette_1 ignored
        guard.check_move("pipette_1", at(0), at(2000))      # unlocated mover allowed
        self.assertEqual(list(guard.status()["unlocated"]), ["pipette_1"])
        units["pipette_1"].calibrated = True
        units["pipette_1"]._tip_reference = None
        self.assertIn("tip position is unknown", guard.status()["unlocated"]["pipette_1"])

    def test_home_anchor_locates_a_pipette(self):
        units = rig()
        units["pipette_1"]._tip_reference = None
        anchors = {"pipette_1": (np.array([1000.0, 0.0]), np.zeros(3), np.zeros(2))}
        guard = guard_for(units, home_anchor=anchors.get)
        self.assertEqual(guard.status()["unlocated"], {})
        with self.assertRaises(CollisionError):
            guard.check_move("pipette_0", at(0), at(950))

    def test_disabled_and_manual(self):
        guard = guard_for(rig())
        with guard.manual():
            guard.check_move("pipette_0", at(0), at(990))
        guard.set_enabled(False)
        guard.check_move("pipette_0", at(0), at(990))
        self.assertFalse(guard.status()["enabled"])

    def test_message_is_readable_and_reported(self):
        reported = []
        guard = guard_for(rig(), on_blocked=reported.append)
        with self.assertRaises(CollisionError) as caught:
            guard.check_move("pipette_0", at(0), at(950))
        message = str(caught.exception)
        for fragment in ("pipette_0", "pipette_1", "50 um past", "to its right", "100 um short",
                         "Collision Guard in the Movement box"):
            self.assertIn(fragment, message)
        self.assertEqual(reported, [message])

    def test_cells(self):
        guard = guard_for(rig())
        right_cell = np.array([-1100.0, 0.0, 0.0])           # world (1100, 0): past pipette_1
        guard.check_cell("pipette_1", right_cell)
        guard.check_cell("pipette_0", np.array([-800.0, 0.0, 0.0]))   # short of pipette_1: fine
        with self.assertRaises(CollisionError) as caught:
            guard.check_cell("pipette_0", right_cell)
        self.assertIn("past the limit at pipette_1's tip", str(caught.exception))
        autopatcher = AutoPatcher.__new__(AutoPatcher)
        autopatcher.collision_guard, autopatcher.pipette_key = guard, "pipette_0"
        with self.assertRaises(CollisionError):
            autopatcher.check_cell_allowed((right_cell, None, None))


class FakeDevice:
    """Device with Scientifica-like signatures (absolute_move has no speed)."""

    def __init__(self, position=(0.0, 0.0, 0.0)):
        self.pos = np.asarray(position, dtype=float)
        self.moves, self.stopped, self.extra = [], 0, "passthrough"

    def position_group(self, axes):
        return self.pos.copy()

    def absolute_move(self, x, axis):
        self.moves.append(("absolute", x, axis))

    def absolute_move_group(self, x, axes, speed=None):
        self.moves.append(("absolute_group", list(x), list(axes)))

    def relative_move(self, x, axis, speed=None):
        self.moves.append(("relative", x, axis))

    def relative_move_group(self, x, axes, speed=None):
        self.moves.append(("relative_group", list(x), list(axes)))

    def absolute_move_group_velocity(self, vel, axes):
        self.moves.append(("velocity", list(vel), list(axes)))

    def stop(self):
        self.stopped += 1


class GuardedDeviceTests(unittest.TestCase):
    def setUp(self):
        self.units = rig()
        self.devices = {key: FakeDevice() for key in self.units}
        for key, unit in self.units.items():
            # Each stub unit reports its fake device's position.
            unit.position = (lambda dev: (lambda: dev.pos.copy()))(self.devices[key])
        self.guard = guard_for(self.units)
        self.dev0 = GuardedPipetteDevice(self.devices["pipette_0"], "pipette_0", [1, 2, 3], self.guard)

    def test_allowed_moves_are_forwarded(self):
        self.dev0.absolute_move(300.0, 1)
        self.dev0.relative_move_group([100.0, 0.0, 0.0], [1, 2, 3])
        self.assertEqual([move[0] for move in self.devices["pipette_0"].moves], ["absolute", "relative_group"])
        self.assertEqual(self.dev0.extra, "passthrough")

    def test_refused_moves_are_not_sent(self):
        with self.assertRaises(CollisionError):
            self.dev0.absolute_move(950.0, 1)
        with self.assertRaises(CollisionError):
            self.dev0.relative_move(950.0, 1)
        self.assertEqual(self.devices["pipette_0"].moves, [])

    def test_refused_velocity_stops_the_device(self):
        with self.assertRaises(CollisionError):
            self.dev0.absolute_move_group_velocity([2000.0, 0.0, 0.0], [1, 2, 3])
        self.assertEqual(self.devices["pipette_0"].stopped, 1)
        self.assertEqual(self.devices["pipette_0"].moves, [])

    def test_raw_unit_moves_go_through_the_guard(self):
        # PipetteCalHelper moves the raw ManipulatorUnit, not the CalibratedUnit.
        raw_unit = ManipulatorUnit(self.devices["pipette_0"], [1, 2, 3])
        raw_unit.dev = self.dev0
        with self.assertRaises(CollisionError):
            raw_unit.relative_move([950.0, 0.0, 0.0])
        raw_unit.relative_move([10.0, 0.0, 0.0])
        self.assertEqual(len(self.devices["pipette_0"].moves), 1)

    def test_manual_jog_is_never_blocked(self):
        with self.guard.manual():
            self.dev0.relative_move(990.0, 1)
        self.assertEqual(len(self.devices["pipette_0"].moves), 1)


# ------------------------------------------------------------------
# Interfaces
# ------------------------------------------------------------------
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
    interface.collision_guard = guard_for(units)
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
        self.units = rig()
        self.interface = make_pipette_interface(self.units, self.tmp.name)
        import patcherbot.interface.pipettes as pipettes_module
        original = pipettes_module.Observer
        pipettes_module.Observer = lambda controller: ("observer", controller.active_pipette)
        self.addCleanup(setattr, pipettes_module, "Observer", original)

    def test_active_pipette_routes_unit_and_positions(self):
        interface = self.interface
        set_positions(interface, "pipette_0", 1.0)
        set_positions(interface, "pipette_1", 2.0)
        interface.set_active_pipette("pipette_1")
        self.assertIs(interface.calibrated_unit, self.units["pipette_1"])
        self.assertEqual(interface.home_position[0], 2.0)
        self.assertEqual(interface.observer, ("observer", "pipette_1"))

    def test_switch_refused_while_busy(self):
        self.interface.busy_checks.append(lambda: True)
        with self.assertRaises(RuntimeError):
            self.interface.set_active_pipette("pipette_1")
        self.assertEqual(self.interface.active_pipette, "pipette_0")

    def test_collision_guard_toggle(self):
        self.interface.set_collision_guard(False)
        self.assertFalse(self.interface.collision_guard.enabled)
        self.interface.set_collision_guard(True)
        self.assertTrue(self.interface.collision_guard.enabled)

    def test_calibration_round_trip_keeps_each_pipette(self):
        interface = self.interface
        set_positions(interface, "pipette_0", 1.0)
        set_positions(interface, "pipette_1", 2.0)
        interface.write_calibration()
        path = os.path.join(self.tmp.name, "calibration.pickle")
        with open(path, "rb") as stream:
            record = pickle.load(stream)
        self.assertEqual(set(record["pipettes"]), {"pipette_0", "pipette_1"})
        np.testing.assert_allclose(record["home"][:3], [1.0, 0.0, 0.0])  # legacy keys = pipette_0
        fresh_units = rig()
        for unit in fresh_units.values():
            unit.calibrated = False
        fresh = make_pipette_interface(fresh_units, self.tmp.name)
        self.assertEqual(fresh.read_calibration(path), ["pipette_0", "pipette_1"])
        self.assertEqual(fresh.saved_positions("pipette_1")["home"][0], 2.0)

    def test_legacy_keys_are_pipette_0_or_absent(self):
        interface = self.interface
        set_positions(interface, "pipette_1", 2.0)            # only pipette_1 saveable
        interface.set_active_pipette("pipette_1")
        interface.write_calibration()
        with open(os.path.join(self.tmp.name, "calibration.pickle"), "rb") as stream:
            record = pickle.load(stream)
        self.assertEqual(list(record["pipettes"]), ["pipette_1"])
        for legacy in ("manip", "home", "safe"):
            self.assertNotIn(legacy, record)

    def test_reload_log_line_keeps_the_replay_format(self):
        interface = self.interface
        set_positions(interface, "pipette_0", 1.0)
        interface.write_calibration()
        path = os.path.join(self.tmp.name, "calibration.pickle")
        with self.assertLogs(interface.logger, level="INFO") as logs:
            interface.read_calibration(path)
        line = next(record.getMessage() for record in logs.records
                    if "Calibration reloaded from " in record.getMessage())
        # rig_replay3D._read_events extracts the file name like this:
        filename = line.split("Calibration reloaded from ", 1)[1].split("; file saved at:", 1)[0]
        self.assertEqual(os.path.abspath(path), filename)

    def test_tare_file_is_accepted_by_the_experiment_book(self):
        interface = self.interface
        interface._tare_pipettes["pipette_0"] = np.array([1.0, 2.0, 3.0])
        interface._tare_pipettes["pipette_1"] = np.array([4.0, None, 6.0])
        interface.tare_stage = np.array([7.0, 8.0, 9.0])
        interface.write_tare()
        with open(os.path.join(self.tmp.name, "tare.pickle"), "rb") as stream:
            tares = pickle.load(stream)
        self.assertEqual(ExperimentBookLogger._normalize_tares(tares)["pipette"], [1.0, 2.0, 3.0])
        with open(os.path.join(self.tmp.name, "tare_pipettes.pickle"), "rb") as stream:
            self.assertEqual(list(pickle.load(stream)["pipette_1"]), [4.0, None, 6.0])

    def test_sanity_check_compares_each_pipette_with_its_geometry(self):
        self.assertEqual(self.interface.check_calibration_geometry(), [])
        config = self.units["pipette_1"].config
        config.pipette_z_rotation = 0.0        # 180 is also accepted (sign not verified)
        self.assertEqual(self.interface.check_calibration_geometry(), [])
        config.pipette_z_rotation = 150.0
        warnings = self.interface.check_calibration_geometry()
        self.assertTrue(warnings and "pipettes.pipette_1.pipette_z_rotation" in warnings[0])
        config.pipette_z_rotation = 180.0
        config.stage_y_axis_flip = True        # wrong handedness
        warnings = self.interface.check_calibration_geometry()
        self.assertTrue(warnings and "stage_y_axis_flip" in warnings[0]
                        and "pipette_1" in warnings[0])
        config.stage_y_axis_flip = False
        config.pipette_k_scale = 0.5
        warnings = self.interface.check_calibration_geometry()
        self.assertTrue(warnings and "pipettes.pipette_1.pipette_k_scale" in warnings[0])

    def test_sanity_check_accepts_a_mirrored_manipulator(self):
        # Calibration of a manipulator whose map is k * Rz(-20) * flipY (Norwich pipette_0).
        flip_y = np.diag([1.0, -1.0, 1.0])
        G = 0.79 * rotation_z(-20) @ flip_y
        unit = self.units["pipette_0"]
        unit.M = np.linalg.inv(G)
        unit.config.pipette_z_rotation, unit.config.pipette_k_scale = -20.0, 0.79
        self.assertTrue(self.interface.check_calibration_geometry())   # flips not set yet
        unit.config.stage_y_axis_flip = True
        self.assertEqual(self.interface.check_calibration_geometry(), [])


# ------------------------------------------------------------------
# Stage -> pipette map (CalibratedUnit.rotate, Follow Stage)
# ------------------------------------------------------------------
NORWICH_PIPETTE_1_FOLLOW_STAGE = np.array([   # stage x, y -> total pipette x, y (um), 2026-11-17
    [-129.8, -24.6, 101.7, -27.7], [-125.4, 5.7, 126.2, -46.7], [91.6, -118.8, -27.2, 109.1],
    [-65.2, 10.4, 25.4, -30.5], [23.8, -10.7, -35.2, 13.8], [-86.7, -108.6, 145.1, 41.1],
    [-62.1, -140.5, 91.5, 71.2], [99, 70.6, -104, -18.1], [-9.4, 19, -10.5, -16.9],
    [-142, 48.2, 100.9, -85.3], [46, 44.9, -46.4, -14.7], [-22.5, 18.4, 8.9, -19.7],
    [5.3, 14.5, -10.2, -6.7], [-130.4, 100.5, 63.4, -110.3], [46.7, 97.7, -62.6, -46.8],
    [51, -20.8, -29.1, 32.7], [-101.7, -4.6, 73.8, -31.5], [46.3, -62.5, -9.8, 61.1],
    [32.4, 103.7, -59.3, -59.9], [5.4, -140.6, 46.4, 96.6], [-3.3, -97.6, 30.4, 61.8],
    [56.2, -92.3, -9.4, 79.4], [60.2, -59.1, -25.9, 61], [61.1, -41.1, -28.3, 49.7],
    [-137.4, 83.7, 70, -110], [87.7, -46.2, -47.4, 63.5], [-116.7, 27.2, 74.1, -59.8]])


def rotate_with(**geometry):
    unit = SimpleNamespace(config=SimpleNamespace(**dict(GEOMETRY, **geometry)),
                           debug=lambda *args, **kwargs: None)
    return lambda coordinates, axis=2: CalibratedUnit.rotate(unit, coordinates, axis)


def norwich_pipette(key):
    config = CalibrationConfig(name="Calibration")
    config.from_file(str(REPO / "rig_setup" / "cal_configs" / "Norwich_cal.yaml"))
    return SimpleNamespace(config=PipetteConfigView(config, key), debug=lambda *a, **k: None)


class RotateTests(unittest.TestCase):
    def test_flips_rotation_and_k(self):
        rotate = rotate_with(pipette_z_rotation=90.0, pipette_k_scale=0.5, stage_y_axis_flip=True)
        step = np.array([10.0, 20.0, 0.0])
        np.testing.assert_allclose(rotate(step), [10.0, 5.0, 0.0], atol=1e-9)  # 0.5*Rz(90)*(10,-20)
        np.testing.assert_allclose(step, [10.0, 20.0, 0.0])                 # input untouched
        np.testing.assert_allclose(rotate(step, [2]), rotate(step))         # list of axes
        np.testing.assert_allclose(rotate([10.0, 20.0]), rotate(step))      # 2-vector padded

    def test_each_pipette_uses_its_own_geometry(self):
        step = np.array([30.0, -40.0, 0.0])
        flip_y = np.diag([1.0, -1.0, 1.0])
        np.testing.assert_allclose(CalibratedUnit.rotate(norwich_pipette("pipette_0"), step, [2]),
                                   0.79156 * rotation_z(-20.0105) @ flip_y @ step)
        np.testing.assert_allclose(CalibratedUnit.rotate(norwich_pipette("pipette_1"), step, [2]),
                                   0.82 * rotation_z(154.0) @ step)

    def test_norwich_pipette_1_data_fits_its_geometry(self):
        unit = norwich_pipette("pipette_1")
        stage, pipette = NORWICH_PIPETTE_1_FOLLOW_STAGE[:, :2], NORWICH_PIPETTE_1_FOLLOW_STAGE[:, 2:]
        predicted = np.array([CalibratedUnit.rotate(unit, step, [2])[:2] for step in stage])
        rms = np.sqrt(((pipette - predicted) ** 2).mean(axis=0))
        # What's left is mostly x backlash (~25 um deadband) and the x/y scale difference.
        self.assertLess(rms[0], 15.0)
        self.assertLess(rms[1], 10.0)
        # The old setting (158.5 deg, k unused) misses by far more.
        old = rotate_with(pipette_z_rotation=158.5)
        old_rms = np.sqrt(((pipette - np.array([old(step)[:2] for step in stage])) ** 2).mean(axis=0))
        self.assertLess(np.linalg.norm(rms), 15.0)       # ~14 um
        self.assertGreater(np.linalg.norm(old_rms), 20.0)  # ~26 um


class AutoPatchRoutingTests(unittest.TestCase):
    def test_active_and_rig_autopatchers(self):
        pipettes = SimpleNamespace(active_pipette="pipette_0", primary_pipette="pipette_0")
        interface = AutoPatchInterface.__new__(AutoPatchInterface)
        interface.pipette_controller = pipettes
        interface.autopatchers = {"pipette_0": "ap0", "pipette_1": "ap1"}
        interface.cell_queues = {"pipette_0": [], "pipette_1": []}
        interface.cells_to_patch.append("cell A")
        pipettes.active_pipette = "pipette_1"
        self.assertEqual(interface.current_autopatcher, "ap1")
        self.assertEqual(interface.rig_autopatcher, "ap0")
        self.assertEqual(interface.cells_to_patch, [])


# ------------------------------------------------------------------
# Compatibility with tools that read pipette_0's data
# ------------------------------------------------------------------
class CompatibilityTests(unittest.TestCase):
    def test_secondary_attempts_are_out_of_attempt_globs(self):
        with tempfile.TemporaryDirectory() as root:
            StateMachineLogger._session_folder = None
            try:
                StateMachineLogger(base_path=root, attempt_id=1)
                StateMachineLogger(base_path=root, attempt_id=1, subfolder="pipette_1")
                session = Path(StateMachineLogger._session_folder)
            finally:
                StateMachineLogger._session_folder = None
            self.assertEqual([p.name for p in session.glob("attempt_*")], ["attempt_1"])
            self.assertTrue((session / "pipette_1" / "attempt_1").is_dir())

    def test_logger_names(self):
        class Thing(LoggingObject):
            pass
        primary, secondary = Thing(), Thing()
        secondary.log_name_suffix = "pipette_1"
        self.assertFalse(primary.logger.name.endswith("pipette_0"))
        self.assertTrue(secondary.logger.name.endswith(".Thing.pipette_1"))
        autopatcher = AutoPatcher.__new__(AutoPatcher)
        autopatcher.log_name_suffix = "pipette_1"
        self.assertTrue(autopatcher.logger.name.endswith("AutoPatcher.pipette_1"))

    def test_replay_skips_secondary_log_rows(self):
        handler_format = logging.Formatter('%(asctime)s,%(levelname)s,%(message)s,%(name)s,%(thread)d')
        rows = []
        for name in ("patcherbot.controller.patch.AutoPatcher",
                     "patcherbot.controller.patch.AutoPatcher.pipette_1",
                     "patcherbot.PipetteCalibration.pipette_2"):
            record = logging.LogRecord(name, logging.INFO, __file__, 1,
                                       "Moving to Cell position: [1, 2]", None, None)
            rows.append(next(csv.reader([handler_format.format(record)])))
        self.assertEqual([is_secondary_pipette_log_row(row) for row in rows], [False, True, True])


if __name__ == "__main__":
    unittest.main()
