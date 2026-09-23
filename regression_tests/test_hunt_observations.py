"""Hardware-free regression tests for whole Hunt observations.

Run this file with a Python environment containing NumPy. The real helper and
phase classes are loaded without importing GUI, hardware, or model packages.
"""

import ast
import collections
from copy import deepcopy
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from abc import ABC, abstractmethod

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / "patcherbot/devices/manipulator/helpers/ObservationHelper.py"
spec = importlib.util.spec_from_file_location("tested_observation_helper", HELPER_PATH)
helper_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper_module)
ObservationHelper = helper_module.ObservationHelper

PHASE_GLOBALS = dict(ABC=ABC, abstractmethod=abstractmethod, collections=collections,
                     math=math, sys=sys, np=np)


def load_classes(relative_path, names):
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    nodes = [node for node in tree.body
             if isinstance(node, ast.ClassDef) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), PHASE_GLOBALS)


load_classes("patcherbot/controller/base.py", {"RequestedAbortException", "RequestedSuccessException"})
load_classes("patcherbot/controller/errors.py", {"AutopatchError"})
load_classes("patcherbot/controller/PhaseController.py", {"PhaseController"})
load_classes("patcherbot/controller/phases/hunt_cell.py", {"HuntCellPhase"})
HuntCellPhase = PHASE_GLOBALS["HuntCellPhase"]
AutopatchError = PHASE_GLOBALS["AutopatchError"]
RequestedAbortException = PHASE_GLOBALS["RequestedAbortException"]
RequestedSuccessException = PHASE_GLOBALS["RequestedSuccessException"]


class Clock:
    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now

    def time(self):
        return 1_700_000_000.0 + self.now


class Device:
    def __init__(self, controller, name):
        self.controller, self.name = controller, name
        self.xyz = np.zeros(3)
        self.stops = 0
        self.stop_error = None
        self.M = np.eye(2)
        self.Minv = np.eye(2)
        self.r0 = np.zeros(2)
        self.up_direction = 1

    def position(self):
        return float(self.xyz[2]) if self.name == "microscope" else self.xyz.copy()

    def stop(self):
        self.stops += 1
        self.controller.events.append(("stop", self.name))
        if self.stop_error is not None:
            raise self.stop_error

    def absolute_move_group_velocity(self, velocity):
        self.controller.events.append(("absolute", self.name, tuple(velocity)))

    def relative_move_group_velocity(self, velocity):
        self.controller.events.append(("relative", self.name, tuple(velocity)))

    def relative_move(self, displacement):
        displacement = np.asarray(displacement)
        self.xyz[:len(displacement)] += displacement
        self.controller.events.append(("relative_move", self.name, tuple(displacement)))

    def absolute_move(self, value):
        self.xyz[2] = value
        self.controller.events.append(("absolute_z", self.name, value))

    def absolute_move_velocity(self, velocity):
        self.controller.events.append(("focus_velocity", self.name, velocity))

    def wait_until_still(self):
        pass

    def pixels_to_um_relative(self, values):
        return np.asarray(values, dtype=float) * 10.0


class Controller:
    def __init__(self, clock, mode="Classic", cell_type="Plate"):
        self.clock = clock
        self.config = NS(mode=mode, cell_type=cell_type, pressure_near=20.0,
                         max_descent_speed=2.0, max_distance=20.0,
                         cell_R_increase=0.15, hunt_search_margin=5.0,
                         hunt_progress_cycles=4, hunt_min_progress_percent=25.0, cell_distance=20.0,
                         use_centroid=True, tracking_mode="auto", track_max_fast_jump_px=10)
        self.events = []
        self.abort_requested = self.success_requested = False
        self.rig_ready = True
        self.resistance = 5.4
        self.read_count = 0
        self.inference_calls = 0
        self.manual_after_samples = None
        self.calibrated_stage = Device(self, "stage")
        self.calibrated_unit = Device(self, "pipette")
        self.microscope = Device(self, "microscope")
        self.calibrated_unit.microscope = self.microscope
        self.calibrated_unit.config = NS(microscope_units_per_um=1.0)
        self.calibrated_stage.camera = NS(raw_frame_queue=collections.deque())
        self.calibrated_stage.cellDetectHelper = NS(_ensure_detector=lambda: NS(
            detect_cells=lambda frame: [(50.0, 50.0, 0.9)]))
        self.calibrated_unit.pipetteCalHelper = NS(pipetteDetector=NS(
            detect_pipette=lambda frame: np.array([11.0, 22.0])))
        self.calibrated_unit.pipetteFocusHelper = NS(pipetteFocuser=NS(
            get_pipette_focus_value=lambda frame: 3.0))
        self.pressure_value = 17.5
        self.commanded_pressure = 20.0
        self.pressure = NS(get_last_acquisition=lambda: self.pressure_value,
                           get_pressure=lambda: self.commanded_pressure,
                           get_ATM=lambda: False,
                           set_pressure=self.set_pressure)
        self.daq = NS(resistance=lambda: 5.0)
        self.agenthelper = NS(prepare_model=lambda name: self.events.append(("model", name)),
                              run_inference=self.inference)
        self.observation_helper = ObservationHelper(self)

    def set_pressure(self, value):
        self.commanded_pressure = value

    def isrigready(self):
        pass

    def resistanceRamp(self, **kwargs):
        if not kwargs:
            return 5.0
        self.read_count += 1
        if self.read_count > 1000:
            raise AssertionError("Hunt did not terminate")
        return self.resistance

    def abort_if_requested(self):
        if self.abort_requested:
            raise RequestedAbortException()

    def success_if_requested(self):
        if self.success_requested:
            raise RequestedSuccessException()

    def info(self, message):
        self.abort_if_requested()
        self.success_if_requested()

    warning = info

    def sleep(self, seconds):
        self.clock.now += seconds
        if self.manual_after_samples is not None and self.read_count >= self.manual_after_samples:
            self.success_requested = True
        self.abort_if_requested()
        self.success_if_requested()

    def observe(self, *args, **kwargs):
        result = self.observation_helper.observe(*args, **kwargs)
        self.events.append(("observe", self.read_count))
        return result

    def inference(self, observation):
        for field in ("pipette_positions", "stage_positions", "camera_image", "resistance"):
            if field not in observation:
                raise AssertionError("Missing policy field: " + field)
        action = ([1.0, 2.0, 3.0], None, [0.0, 0.0, 0.0])[min(self.inference_calls, 2)]
        self.inference_calls += 1
        self.events.append(("inference", action))
        return action

    def devices(self):
        return self.calibrated_stage, self.calibrated_unit, self.microscope


class HardwareFreeTest(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.time_patch = patch.object(helper_module, "time", self.clock)
        self.time_patch.start()
        self.addCleanup(self.time_patch.stop)
        PHASE_GLOBALS["time"] = self.clock
        self.controller = Controller(self.clock)
        self.helper = self.controller.observation_helper

    def publish(self, frame=11, value=1.0, acquired_at=None, target=None):
        acquired_at = self.clock.now - 0.1 if acquired_at is None else acquired_at
        result = dict(source_frame=frame, frame_available_at=acquired_at,
                      frame_acquired_at=acquired_at - 0.01,
                      started_at=acquired_at + 0.01, completed_at=acquired_at + 0.02,
                      timestamp_basis="camera_availability", producer_session=3,
                      producer_generation=7, target_identity=None if target is None else id(target),
                      pipette_position=np.array([value, value + 1.0]), pipette_focus=value + 2.0,
                      cell_detections=[(value, value + 1.0, 0.9)],
                      tracked_cell_position=np.array([value + 3.0, value + 4.0]),
                      confidence={"cell_detector": [0.9]}, status={"cell_detector": "ok"},
                      source_positions={"stage": np.zeros(3), "microscope": 0.0,
                                        "manipulator": np.zeros(3)},
                      positions_sampled_at=acquired_at + 0.01,
                      source_image_shape=(4, 6, 3),
                      _source_image=np.full((4, 6, 3), value, dtype=np.uint8))
        self.helper._deep_learning_observation = (target, deepcopy(result))
        return result

    def phase(self, mode="Adaptive", cell_type="Slice"):
        self.controller.config.mode = mode
        self.controller.config.cell_type = cell_type
        self.helper.start_deep_learning = lambda **kwargs: None
        self.helper.set_deep_learning_models = lambda **kwargs: None
        self.helper.stop_deep_learning = lambda: None
        return HuntCellPhase(self.controller)


class ObservationTests(HardwareFreeTest):
    def test_one_inference_snapshot_supplies_all_derived_fields(self):
        self.publish(frame=11, value=1.0)
        original = self.helper.get_deep_learning
        calls = []

        def replace_cache_after_read(**kwargs):
            result = original(**kwargs)
            calls.append(result)
            self.publish(frame=12, value=20.0)
            return result

        self.helper.get_deep_learning = replace_cache_after_read
        result = self.helper.observe(fields=["deep_learning", "pipette_positions", "pipette_focus",
                                             "cell_detections", "tracked_cell_position", "camera_image"],
                                     raw_measurements=True)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["deep_learning"]["source_frame"], 11)
        np.testing.assert_array_equal(result["pipette_positions"], [1.0, 2.0, 3.0])
        self.assertEqual(result["pipette_focus"], 3.0)
        self.assertEqual(result["cell_detections"], [(1.0, 2.0, 0.9)])
        np.testing.assert_array_equal(result["tracked_cell_position"], [4.0, 5.0])
        self.assertTrue(np.all(result["camera_image"] == 1))
        self.assertNotIn("_source_image", result["deep_learning"])
        for field in ("pipette_positions", "pipette_focus", "cell_detections", "camera_image"):
            self.assertEqual(result["field_metadata"][field]["source_frame"], 11)


    def test_freshness_is_recomputed_after_slow_hardware_acquisition(self):
        self.publish(acquired_at=self.clock.now - 0.5)
        original = self.controller.resistanceRamp
        def slow_resistance(**kwargs):
            self.clock.now += 0.75
            return original(**kwargs)
        self.controller.resistanceRamp = slow_resistance
        observation = self.helper.observe(fields=["deep_learning", "resistance", "pipette_positions"],
                                          raw_measurements=True)
        self.assertTrue(observation["deep_learning"]["stale"])
        self.assertTrue(observation["field_metadata"]["pipette_positions"]["stale"])
        self.assertGreater(observation["field_metadata"]["pipette_positions"]["age_s"], 1.0)
        np.testing.assert_array_equal(observation["pipette_positions"], [1., 2., 3.])

    def test_missing_pressure_state_is_not_reported_as_atmosphere_off(self):
        self.controller.pressure.get_ATM = lambda: None
        observation = self.helper.observe(fields=["pressure_atm_state"], raw_measurements=True)
        self.assertTrue(np.isnan(observation["pressure_atm_state"]).all())
        self.assertFalse(observation["field_metadata"]["pressure_atm_state"]["valid"])

    def test_missing_and_stale_measurements_stay_in_history_with_provenance(self):
        phase = object()
        self.helper.reset_history(phase, None)
        missing = self.helper.observe(fields=["resistance", "deep_learning", "pipette_positions",
                                              "pressure", "commanded_pressure_mbar", "pressure_atm_state"],
                                      raw_measurements=True)
        self.helper.record_observation(phase, missing)
        self.publish(acquired_at=self.clock.now - 5.0)
        stale = self.helper.observe(fields=["resistance", "deep_learning", "pipette_positions",
                                            "pressure", "commanded_pressure_mbar", "pressure_atm_state"],
                                    raw_measurements=True)
        self.helper.record_observation(phase, stale)
        self.assertEqual(len(self.helper.observation_decks[phase]), 2)
        self.assertTrue(np.isnan(missing["pipette_positions"]).all())
        self.assertTrue(stale["deep_learning"]["stale"])
        self.assertTrue(stale["field_metadata"]["pipette_positions"]["stale"])
        self.assertEqual(stale["pressure"][0], 17.5)
        self.assertIsNone(stale["field_metadata"]["pressure"]["acquired_at"])
        self.assertEqual(stale["field_metadata"]["pressure"]["timestamp_basis"],
                         "cache_read_complete")
        self.assertEqual(stale["field_metadata"]["resistance"]["acquired_at"], self.clock.now)

    def test_unbounded_history_keeps_old_values_and_windows_without_images(self):
        phase = object()
        self.helper.reset_history(phase, None, include_images=False)
        for index in range(150):
            observation = dict(resistance=float(index), camera_image=np.zeros((4, 4, 3)),
                               field_metadata={"camera_image": {"source_frame": index}},
                               deep_learning={"pipette_focus": float(index) / 10})
            self.helper.record_observation(phase, observation)
        deck = self.helper.observation_decks[phase]
        self.assertEqual(len(deck), 150)
        self.assertEqual(deck[0]["resistance"], 0.0)
        self.assertIsNone(deck[0]["camera_image"])
        self.assertEqual(deck[0]["field_metadata"]["camera_image"]["source_frame"], 0)
        np.testing.assert_array_equal(self.helper.observation_window(
            phase, "resistance", 3, include_current=True), [147.0, 148.0, 149.0])
        np.testing.assert_allclose(self.helper.observation_window(
            phase, "deep_learning.pipette_focus", 3, include_current=True), [14.7, 14.8, 14.9])

    def test_calculation_enrichment_merges_without_rewriting_previous_samples(self):
        phase = object()
        self.helper.reset_history(phase, None)
        first = {"resistance": 5.0, "target_cell_image_xy": np.array([10.0, 20.0])}
        self.helper.record_observation(phase, first)
        self.helper.record_calculations(phase, first, {"distance_um": 2.0})
        self.helper.record_calculations(phase, first, {"progress_percent": 25.0})
        first["target_cell_image_xy"][0] = 999.0
        first["calculations"]["distance_um"] = 999.0
        second = {"resistance": 6.0}
        self.helper.record_observation(phase, second)
        snapshot = self.helper.observation_decks[phase][0]
        np.testing.assert_array_equal(snapshot["target_cell_image_xy"], [10.0, 20.0])
        self.assertEqual(snapshot["calculations"], {"distance_um": 2.0, "progress_percent": 25.0})
        with self.assertRaises(ValueError):
            self.helper.record_calculations(phase, first, {"late": True})

    def test_inference_batch_retains_source_frame_session_and_target_identity(self):
        target = object()
        self.helper._producer_session = 13
        self.helper._producer_generation = 17
        self.helper.refresh_deep_learning(cell=target, _frame=(41, 99.0, np.zeros((4, 6, 3))),
                                          cell_models=False, pipette_models=True)
        result = self.helper.get_deep_learning(cell=target)
        self.assertEqual(result["source_frame"], 41)
        self.assertEqual(result["producer_session"], 13)
        self.assertEqual(result["producer_generation"], 17)
        self.assertEqual(result["target_identity"], id(target))
        self.assertEqual(result["frame_available_at"], 99.0)
        self.assertEqual(result["positions_sampled_at"], self.clock.now)
        self.assertEqual(result["position_timestamp_basis"], "inference_start")
        self.assertEqual(self.helper.get_deep_learning(cell=object())["status"], "not_available")


class HuntTests(HardwareFreeTest):
    def make_attempt(self, mode="Adaptive", cell_type="Slice"):
        phase = self.phase(mode, cell_type)
        cell = (np.array([-50., -50.]), np.zeros((100, 100, 3)), np.zeros(3))
        state = phase.prepare(cell, mode=mode, cell_type=cell_type)
        self.publish(value=50.)
        observation = phase.observe(**state["observe"])
        return phase, state, observation

    def complete_cycle(self, phase, state, observation, gap):
        state["completed"] = False
        self.assertEqual(phase.decide(observation, state, mode="Adaptive", cell_type="Slice")[0], phase.spear)
        state["completed"] = True
        for expected in (phase.search, phase.shift, phase.scan, phase.shift):
            self.assertEqual(phase.decide(observation, state, mode="Adaptive", cell_type="Slice")[0], expected)
        self.assertFalse(phase.shift_cell)
        phase.cell_z_um = 100.
        state["pipette_z_um"] = 100. + gap
        return phase.decide(observation, state, mode="Adaptive", cell_type="Slice")



    def test_phase_history_and_helper_share_the_same_deque_after_resets(self):
        phase = self.phase("Manual", "Plate")
        self.helper.reset_history(phase, None, include_images=False)
        deck = phase.observation_deck
        self.assertIs(deck, self.helper.observation_decks[phase])
        self.helper.record_observation(phase, {"resistance": 5.0})
        self.assertEqual(phase.observation_deck[-1]["resistance"], 5.0)
        phase.observation_deck.append({"resistance": 6.0})
        np.testing.assert_array_equal(phase.observation_window(
            "resistance", 2, include_current=True), [5.0, 6.0])
        self.helper.reset_history(phase, None)
        self.assertIs(phase.observation_deck, deck)
        self.assertEqual(len(deck), 0)
        self.helper.reset_history(phase, 2)
        self.assertIs(phase.observation_deck, self.helper.observation_decks[phase])
        self.assertEqual(phase.observation_deck.maxlen, 2)
        for value in (7.0, 8.0, 9.0):
            self.helper.record_observation(phase, {"resistance": value})
        self.assertEqual([row["resistance"] for row in phase.observation_deck], [8.0, 9.0])

    def test_legacy_observe_keeps_list_and_pressure_flag_compatibility(self):
        phase = self.phase("Manual", "Plate")
        self.helper.reset_history(phase, 20)
        self.publish(value=10.)
        current_frame = np.full((4, 6, 3), 7, dtype=np.uint8)
        self.controller.calibrated_stage.camera.raw_frame_queue.append(
            (55, self.clock.now, 0., current_frame))
        legacy = phase.observe()
        self.assertIsInstance(legacy, list)
        self.assertEqual(len(legacy), 4)
        self.assertTrue(np.all(legacy[2] == 7))
        self.assertIsInstance(self.helper.observation_decks[phase][-1], list)
        with_pressure = phase.observe(True)
        self.assertIsInstance(with_pressure, dict)
        for field in ("pressure", "commanded_pressure_mbar", "pressure_atm_state"):
            self.assertIn(field, with_pressure)

    def test_target_association_and_pressure_are_in_the_recorded_sample(self):
        phase, state, _ = self.make_attempt()
        self.publish(frame=31, value=50.)
        self.helper._deep_learning_observation[1]["source_positions"]["microscope"] = 7.
        self.controller.calibrated_stage.xyz[0] = 200.
        observation = phase.observe(**state["observe"])
        self.assertTrue(observation["target_cell_valid"])
        np.testing.assert_array_equal(observation["target_cell_image_xy"], [50., 51.])
        self.assertEqual(observation["target_cell_observed_z_um"], 7.)
        self.assertEqual(observation["stage_positions"][0], 200.)
        snapshot = self.helper.observation_decks[phase][-1]
        self.assertEqual(snapshot["target_cell_id"], (-50., -50.))
        self.assertEqual(snapshot["field_metadata"]["target_cell_image_xy"]["source_frame"], 31)
        self.assertFalse(snapshot["field_metadata"]["target_cell_observed_z_um"]["synchronized_to_frame"])
        self.assertEqual(snapshot["pressure"][0], self.controller.pressure_value)
        self.assertIsNone(snapshot["camera_image"])
        observation["target_cell_image_xy"][0] = -999.
        self.assertEqual(snapshot["target_cell_image_xy"][0], 50.)

    def test_missing_target_evidence_is_unknown_not_confirmed_cell_loss(self):
        phase, state, _ = self.make_attempt()
        self.helper._deep_learning_observation[1]["cell_detections"] = None
        observation = phase.observe(**state["observe"])
        self.assertFalse(observation["target_cell_valid"])
        self.assertEqual(observation["target_cell_status"], "unavailable")
        self.assertTrue(np.isnan(observation["target_cell_observed_z_um"]))
        phase.spear(observation, observation, dict(state, descent_direction=1, monitor_state={}))
        self.assertIsNone(phase.cell_lost)
        self.helper._deep_learning_observation[1]["cell_detections"] = []
        self.assertEqual(phase.observe(**state["observe"])["target_cell_status"], "not_detected")

    def test_hunt_keeps_more_than_120_pressure_and_target_samples(self):
        phase, state, _ = self.make_attempt()
        for index in range(130):
            self.publish(frame=index, value=50.)
            self.controller.pressure_value = float(index)
            phase.observe(**state["observe"])
        deck = self.helper.observation_decks[phase]
        self.assertGreater(len(deck), 120)
        self.assertEqual(deck[1]["pressure"][0], 0.)
        self.assertEqual(deck[1]["target_cell_id"], (-50., -50.))
        np.testing.assert_array_equal(phase.observation_window(
            "pressure", 3, shape=(1,), include_current=True), [[127.], [128.], [129.]])

    def test_old_image_with_new_inference_is_rejected_after_movement(self):
        phase, state, observation = self.make_attempt()
        observation["deep_learning"].update(source_frame=99, frame_acquired_at=90.,
            frame_available_at=101., started_at=110., stale=False, pipette_position=[50., 50.],
            pipette_focus=0., source_image_shape=(100, 100, 3))
        observation["visual_context_valid"] = True
        observation["target_cell_valid"] = True
        for monitor in (phase.search, phase.scan, phase.shift):
            with self.subTest(monitor=monitor.__name__):
                state["monitor_state"] = {"after_move": 100.}
                state["travel_limit"] = 10.
                phase.shift_cell = False
                completed, command, _ = monitor(observation, observation, state)
                self.assertFalse(completed)
                self.assertEqual(command, 0.)

    def test_progress_starts_at_20_counts_first_cycle_and_resets_early(self):
        phase, state, observation = self.make_attempt()
        self.assertEqual(state["progress_baseline_um"], 20.)
        self.complete_cycle(phase, state, observation, 19.)
        self.assertEqual(state["progress_cycles"], 1)
        self.complete_cycle(phase, state, observation, 15.)
        self.assertEqual(state["progress_cycles"], 0)
        self.assertEqual(state["progress_baseline_um"], 15.)
        self.complete_cycle(phase, state, observation, 11.25)
        self.assertEqual(state["progress_cycles"], 0)
        self.assertEqual(state["progress_baseline_um"], 11.25)

    def test_four_unproductive_cycles_fail_and_attempt_reset_clears_progress(self):
        for final_gap in (20., 25., 15.02):
            with self.subTest(final_gap=final_gap):
                phase, state, observation = self.make_attempt()
                for _ in range(20):
                    phase.decide(observation, state, mode="Adaptive", cell_type="Slice")
                self.assertEqual(state["progress_cycles"], 0)
                for _ in range(3):
                    self.complete_cycle(phase, state, observation, 20.)
                with self.assertRaises(AutopatchError):
                    self.complete_cycle(phase, state, observation, final_gap)
                self.assertEqual(phase.operation, phase.shift)
                reset = phase.prepare(state["cell"], mode="Adaptive", cell_type="Slice")
                self.assertEqual(reset["progress_cycles"], 0)
                self.assertEqual(reset["progress_baseline_um"], 20.)


    def test_spear_requires_a_seen_cell_before_declaring_visual_loss(self):
        phase, state, observation = self.make_attempt()
        state["monitor_state"] = {}
        state["descent_direction"] = 1
        observation.update(target_cell_valid=False, target_cell_status="not_detected")
        completed, _, _ = phase.spear(observation, observation, state)
        self.assertFalse(completed)
        self.assertIsNone(phase.cell_lost)
        observation.update(target_cell_valid=True, target_cell_status="observed")
        completed, _, _ = phase.spear(observation, observation, state)
        self.assertFalse(completed)
        self.assertFalse(phase.cell_lost)
        observation.update(target_cell_valid=False, target_cell_status="unavailable")
        completed, _, _ = phase.spear(observation, observation, state)
        self.assertFalse(completed)
        self.assertFalse(phase.cell_lost)
        observation.update(target_cell_status="not_detected")
        completed, _, _ = phase.spear(observation, observation, state)
        self.assertTrue(completed)
        self.assertTrue(phase.cell_lost)

    def test_invalid_adaptive_baseline_fails_before_movement(self):
        for distance in (0., -1., float("nan"), float("inf")):
            with self.subTest(distance=distance):
                phase = self.phase()
                self.controller.config.cell_distance = distance
                with self.assertRaises(AutopatchError):
                    phase.prepare(([-50., -50.], None, None), mode="Adaptive", cell_type="Slice")
                self.assertFalse(any(e[0] in ("absolute", "relative") for e in self.controller.events))

    def test_all_modes_use_whole_observations_and_preserve_policy_behavior(self):
        for mode in ("Classic", "Adaptive", "Manual", "Training", "Agent"):
            for cell_type in ("Plate", "Slice"):
                with self.subTest(mode=mode, cell_type=cell_type):
                    self.controller = Controller(self.clock, mode, cell_type)
                    self.helper = self.controller.observation_helper
                    phase = self.phase(mode, cell_type)
                    if mode == "Training":
                        self.controller.manual_after_samples = 5
                        original_observe = self.controller.observe
                        def distant(*args, **kwargs):
                            self.controller.calibrated_unit.xyz[2] += 100.
                            return original_observe(*args, **kwargs)
                        self.controller.observe = distant
                    self.publish(value=50.)
                    cell = (np.array([-50., -50.]), np.zeros((100, 100, 3)), np.zeros(3))
                    with self.assertRaises(RequestedSuccessException):
                        phase.run(cell)
                    moves = [e for e in self.controller.events if e[0] in ("absolute", "relative")]
                    if mode in ("Manual", "Training"):
                        self.assertFalse(moves)
                    self.assertEqual(self.controller.read_count, 5 if mode == "Training" else 4)
                    if mode == "Agent":
                        self.assertIn(("relative", "pipette", (-10., -20., 3.)), moves)
                        self.assertIn(("relative", "pipette", (0, 0, 0)), moves)
                    self.assertTrue(all(d.stops for d in self.controller.devices()))
                    row = self.helper.observation_decks[phase][-1]
                    for field in ("pressure", "target_cell_id", "field_metadata", "deep_learning"):
                        self.assertIn(field, row)



    def test_prepare_starts_only_the_selected_mode_services(self):
        for mode in ("Classic", "Adaptive", "Manual", "Training", "Agent"):
            for cell_type in ("Plate", "Slice"):
                with self.subTest(mode=mode, cell_type=cell_type):
                    self.controller = Controller(self.clock, mode, cell_type)
                    self.helper = self.controller.observation_helper
                    phase = self.phase(mode, cell_type)
                    calls = []
                    self.helper.start_deep_learning = lambda **kw: calls.append(("start", kw))
                    self.helper.set_deep_learning_models = lambda **kw: calls.append(("models", kw))
                    self.helper.stop_deep_learning = lambda: calls.append(("stop", {}))
                    if mode != "Adaptive" or cell_type != "Slice":
                        self.controller.config.cell_distance = float("nan")
                    phase.prepare((np.array([-50., -50.]), None, None), mode=mode, cell_type=cell_type)
                    self.assertFalse(any(e[0] in ("absolute", "relative") for e in self.controller.events))
                    if mode == "Adaptive" and cell_type == "Slice":
                        self.assertEqual(calls, [("start", {"cell": None}),
                                                 ("models", {"cell": True, "pipette": True})])
                    else:
                        self.assertEqual(calls, [])
                    model_calls = [e for e in self.controller.events if e[0] == "model"]
                    self.assertEqual(model_calls, [("model", "hunt")] if mode == "Agent" else [])

    def test_mode_and_cell_type_changes_during_prepare_do_not_change_current_attempt(self):
        changes = [
            ("Classic", "Plate", "Agent", "Slice"),
            ("Agent", "Plate", "Classic", "Slice"),
            ("Manual", "Slice", "Classic", "Plate"),
            ("Training", "Slice", "Classic", "Plate"),
            ("Adaptive", "Plate", "Adaptive", "Slice"),
            ("Adaptive", "Slice", "Classic", "Plate"),
        ]
        for mode, cell_type, changed_mode, changed_type in changes:
            with self.subTest(mode=mode, cell_type=cell_type):
                self.controller = Controller(self.clock, mode, cell_type)
                self.helper = self.controller.observation_helper
                phase = self.phase(mode, cell_type)
                calls = []
                self.helper.start_deep_learning = lambda **kw: calls.append("start")
                self.helper.set_deep_learning_models = lambda **kw: calls.append("models")
                self.helper.stop_deep_learning = lambda: calls.append("stop")
                original_sleep = self.controller.sleep
                def change_during_settling(seconds):
                    if seconds == 3:
                        self.controller.config.mode = changed_mode
                        self.controller.config.cell_type = changed_type
                    return original_sleep(seconds)
                self.controller.sleep = change_during_settling
                if mode == "Training":
                    self.controller.manual_after_samples = 5
                    original_observe = self.controller.observe
                    def manual_motion(*args, **kwargs):
                        self.controller.calibrated_unit.xyz[2] += 100.
                        return original_observe(*args, **kwargs)
                    self.controller.observe = manual_motion
                cell = (np.array([-50., -50.]), None, None)
                with self.assertRaises(RequestedSuccessException):
                    phase.run(cell)
                moves = [e for e in self.controller.events if e[0] in ("absolute", "relative")]
                if mode in ("Manual", "Training"):
                    self.assertFalse(moves)
                elif mode == "Agent":
                    self.assertIn(("relative", "pipette", (-10., -20., 3.)), moves)
                    self.assertTrue(all(e[0] == "relative" for e in moves))
                else:
                    self.assertIn(("absolute", "pipette", (0, 0, 2.)), moves)
                    self.assertTrue(all(e[0] == "absolute" for e in moves))
                self.assertEqual(calls, ["start", "models", "stop"]
                                 if mode == "Adaptive" and cell_type == "Slice" else [])
                model_calls = [e for e in self.controller.events if e[0] == "model"]
                self.assertEqual(model_calls, [("model", "hunt")] if mode == "Agent" else [])
                self.assertEqual(self.controller.read_count, 5 if mode == "Training" else 4)
                self.assertTrue(all(device.stops for device in self.controller.devices()))

    def test_adaptive_inference_start_and_selection_failures_are_cleaned_up(self):
        for failing_step in ("start", "models"):
            with self.subTest(failing_step=failing_step):
                self.controller = Controller(self.clock, "Adaptive", "Slice")
                self.helper = self.controller.observation_helper
                phase = self.phase()
                calls = []
                pending = RuntimeError(failing_step + " failed")
                def start(**kwargs):
                    calls.append("start")
                    if failing_step == "start":
                        raise pending
                def select(**kwargs):
                    calls.append("models")
                    if failing_step == "models":
                        raise pending
                self.helper.start_deep_learning = start
                self.helper.set_deep_learning_models = select
                self.helper.stop_deep_learning = lambda: calls.append("stop")
                with self.assertRaises(RuntimeError) as caught:
                    phase.run((np.array([-50., -50.]), None, None))
                self.assertIs(caught.exception, pending)
                self.assertEqual(calls, ["start", "stop"] if failing_step == "start"
                                 else ["start", "models", "stop"])
                self.assertTrue(all(device.stops for device in self.controller.devices()))
                self.assertFalse(any(e[0] in ("absolute", "relative") for e in self.controller.events))

    def test_inference_cleanup_cannot_replace_a_pending_abort(self):
        phase = self.phase()
        calls = []
        pending = RequestedAbortException("operator abort")
        def fail_observation(*args, **kwargs):
            raise pending
        def fail_cleanup():
            calls.append("stop")
            raise RuntimeError("inference cleanup failed")
        self.controller.observe = fail_observation
        self.helper.stop_deep_learning = fail_cleanup
        with self.assertRaises(RequestedAbortException) as caught:
            phase.run((np.array([-50., -50.]), None, None))
        self.assertIs(caught.exception, pending)
        self.assertEqual(calls, ["stop"])
        self.assertTrue(all(device.stops for device in self.controller.devices()))

    def test_public_run_stops_after_four_unproductive_cycles_before_fifth_spear(self):
        phase = self.phase()
        self.controller.resistance = 5.0
        self.helper.stop_deep_learning = lambda: self.controller.events.append(("inference_stop",))
        actual_act = phase.act
        operations = []
        def completed_action(monitor=None, **kwargs):
            if kwargs.get("stop"):
                return actual_act(stop=True)
            operations.append((monitor.__name__, phase.shift_cell))
            if monitor == phase.shift and not phase.shift_cell:
                phase.cell_z_um = 100.0
                kwargs["state"]["pipette_z_um"] = 120.0
            return True
        phase.act = completed_action
        with self.assertRaisesRegex(AutopatchError, "4 cycles"):
            phase.run((np.array([-50.0, -50.0]), None, None))
        self.assertEqual(sum(name == "spear" for name, _ in operations), 4)
        self.assertEqual(sum(name == "shift" and not cell for name, cell in operations), 4)
        self.assertEqual(phase.operation, phase.shift)
        self.assertTrue(all(device.stops for device in self.controller.devices()))
        self.assertIn(("inference_stop",), self.controller.events)

    def test_cleanup_preserves_pending_exceptions_in_every_mode(self):
        modes = [(m, "Plate") for m in ("Classic", "Adaptive", "Manual", "Training", "Agent")]
        for mode, cell_type in modes + [("Adaptive", "Slice")]:
            for exception in (RequestedAbortException, RequestedSuccessException, ValueError):
                with self.subTest(mode=mode, cell_type=cell_type, exception=exception.__name__):
                    self.controller = Controller(self.clock, mode, cell_type)
                    self.helper = self.controller.observation_helper
                    phase = self.phase(mode, cell_type)
                    pending = exception("pending observation error")
                    def fail_observation(*args, **kwargs):
                        raise pending
                    self.controller.observe = fail_observation
                    self.controller.calibrated_stage.stop_error = RuntimeError("stop failed")
                    with self.assertRaises(exception) as caught:
                        phase.run((np.array([-50., -50.]), None, None))
                    self.assertIs(caught.exception, pending)
                    self.assertTrue(all(d.stops for d in self.controller.devices()))

class ActionGateTests(HardwareFreeTest):
    def setup_gate(self):
        phase = self.phase()
        observation = {"manipulator_position": np.zeros(3),
                       "stage_positions": np.zeros(3), "resistance": 5.0}
        state = {"start_position": np.zeros(3), "travel_limit": 10.0,
                 "descent_direction": 1, "observe": {},
                 "readings": collections.deque([5.0] * 5), "latest_resistance": 5.0}
        return phase, observation, state

    def test_shift_rejects_unsafe_vectors_and_projected_total_travel(self):
        phase, observation, state = self.setup_gate()
        phase.shift_cell = True
        start = deepcopy(observation)
        def allowed(vector, device=None):
            return phase.action_gate(observation, state, monitor=phase.shift,
                                     command={"relative": vector}, start=start,
                                     mode="Adaptive", max_distance=20.0,
                                     device=device or self.controller.calibrated_stage)
        for vector in ([50.01, 0], [0, 0, 1], [math.nan, 0], [1], [1, 2, 3, 4]):
            with self.subTest(vector=vector):
                self.assertFalse(allowed(vector))
        self.assertTrue(allowed([30, 40, 0]))
        self.assertFalse(allowed([1, 0], self.controller.calibrated_unit))
        observation["stage_positions"][0] = 40.0
        self.assertFalse(allowed([11, 0]))
        self.assertTrue(allowed([10, 0]))
        self.assertTrue(allowed([-10, 0]))

    def test_focus_gate_bounds_target_velocity_direction_and_live_limit(self):
        phase, observation, state = self.setup_gate()
        start = deepcopy(observation)
        def allowed(monitor, command, maximum=20.0):
            return phase.action_gate(observation, state, monitor=monitor,
                                     command=command, start=start, mode="Adaptive",
                                     max_distance=maximum, device=self.controller.microscope)
        self.assertTrue(allowed(phase.scan, {"absolute_z": 10.0}))
        self.assertFalse(allowed(phase.scan, {"absolute_z": 10.01}))
        self.assertFalse(allowed(phase.scan, {"absolute_z": 7.0}, 5.0))
        self.assertFalse(allowed(phase.search, {"absolute_z": 1.0}))
        self.assertTrue(allowed(phase.search, -2.0))
        self.assertFalse(allowed(phase.search, 2.0))
        self.assertTrue(allowed(phase.scan, 2.0))
        self.assertFalse(allowed(phase.scan, -2.0))
        observation["stage_positions"][2] = 10.0
        self.assertFalse(allowed(phase.scan, 2.0))
        observation["stage_positions"][2] = -10.0
        self.assertFalse(allowed(phase.search, -2.0))
        observation["stage_positions"][2] = 100.0
        self.assertTrue(allowed(phase.search, 0.0))

    def test_spear_and_direct_modes_reject_distance_and_passive_motion(self):
        phase, observation, state = self.setup_gate()
        start = deepcopy(observation)
        common = dict(mode="Adaptive", max_distance=20.0,
                      device=self.controller.calibrated_unit)
        self.assertTrue(phase.action_gate(observation, state, monitor=phase.spear,
                                         command=2.0, start=start, **common))
        self.assertFalse(phase.action_gate(observation, state, monitor=phase.spear,
                                          command=-2.0, start=start, **common))
        observation["manipulator_position"][2] = 20.0
        self.assertFalse(phase.action_gate(observation, state, monitor=phase.spear,
                                          command=2.0, start=start, **common))
        for mode in ("Classic", "Agent", "Adaptive", "Manual", "Training"):
            with self.subTest(mode=mode):
                with self.assertRaises(AutopatchError):
                    phase.act(observation=observation, state=state,
                              velocity=[0, 0, 2], mode=mode)
        observation["manipulator_position"][2] = 0.0
        for mode in ("Manual", "Training"):
            with self.assertRaises(AutopatchError):
                phase.act(observation=observation, state=state, velocity=[0, 0, 2], mode=mode)
        self.assertFalse(any(event[0] in ("absolute", "relative") for event in self.controller.events))

    def test_executor_rejects_second_shift_outside_total_envelope(self):
        phase, observation, state = self.setup_gate()
        phase.shift_cell = True
        phase.success_gate = lambda *args, **kwargs: False
        calls = []
        def monitor(observation, start, state):
            calls.append(1)
            return False, {"relative": [40 if len(calls) == 1 else 20, 0, 0]}, self.controller.calibrated_stage
        phase.shift = monitor
        phase.observe = lambda **kwargs: dict(observation, stage_positions=self.controller.calibrated_stage.xyz.copy())
        with self.assertRaisesRegex(AutopatchError, "action gate"):
            phase.act(phase.shift, observation=observation, state=state, mode="Adaptive")
        moves = [event for event in self.controller.events if event[0] == "relative_move"]
        self.assertEqual(moves, [("relative_move", "stage", (40.0, 0.0, 0.0))])
        self.assertTrue(all(device.stops for device in self.controller.devices()))

    def test_unchanged_spear_velocity_rechecks_reduced_live_distance(self):
        phase, observation, state = self.setup_gate()
        phase.success_gate = lambda *args, **kwargs: False
        def monitor(observation, start, state):
            state["descent_direction"] = 1
            return False, 2.0, self.controller.calibrated_unit
        def next_observation(**kwargs):
            self.controller.config.max_distance = 5.0
            return dict(observation, manipulator_position=np.array([0.0, 0.0, 10.0]))
        phase.spear, phase.observe = monitor, next_observation
        with self.assertRaisesRegex(AutopatchError, "action gate"):
            phase.act(phase.spear, observation=observation, state=state, mode="Adaptive")
        moves = [event for event in self.controller.events if event[0] == "absolute"]
        self.assertEqual(moves, [("absolute", "pipette", (0, 0, 2.0))])
        self.assertTrue(all(device.stops for device in self.controller.devices()))

    def test_classic_speed_updates_and_unchanged_velocity_still_rechecks_limit(self):
        phase, observation, state = self.setup_gate()
        state["started"] = True
        for speed in (2.0, 2.0, 3.0):
            self.controller.config.max_descent_speed = speed
            monitor, velocity, relative = phase.decide(observation, state,
                                                       mode="Classic", cell_type="Plate")
            phase.act(monitor, observation=observation, state=state,
                      velocity=velocity, relative=relative, mode="Classic")
        self.assertEqual([event for event in self.controller.events if event[0] == "absolute"],
                         [("absolute", "pipette", (0, 0, 2.0)),
                          ("absolute", "pipette", (0, 0, 3.0))])
        observation["manipulator_position"][2] = 10.0
        self.controller.config.max_distance = 5.0
        monitor, velocity, relative = phase.decide(observation, state,
                                                   mode="Classic", cell_type="Plate")
        with self.assertRaisesRegex(AutopatchError, "action gate"):
            phase.act(monitor, observation=observation, state=state,
                      velocity=velocity, relative=relative, mode="Classic")
        self.assertEqual(len([event for event in self.controller.events if event[0] == "absolute"]), 2)

    def test_request_exceptions_take_precedence_over_rejected_motion(self):
        phase, observation, state = self.setup_gate()
        for abort, success, exception in ((True, False, RequestedAbortException),
                                          (False, True, RequestedSuccessException),
                                          (True, True, RequestedAbortException)):
            with self.subTest(abort=abort, success=success):
                self.controller.abort_requested = abort
                self.controller.success_requested = success
                with self.assertRaises(exception):
                    phase.act(observation=observation, state=state,
                              velocity=[0, 0, 2], mode="Classic")
                phase.act(stop=True)
        self.assertFalse(any(event[0] in ("absolute", "relative") for event in self.controller.events))
        self.assertTrue(all(device.stops == 3 for device in self.controller.devices()))


class AcquisitionTimingTests(HardwareFreeTest):
    def test_camera_marks_capture_start_before_snap_and_preserves_queue_contract(self):
        import datetime
        import threading
        import traceback
        path = ROOT / "patcherbot/devices/camera/camera.py"
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        acquisition = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AcquisitionThread")
        camera = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Camera")
        timing_methods = [n for n in camera.body if isinstance(n, ast.FunctionDef)
                          and n.name in ("_update_frame_pair", "get_frame_timing")]
        timing_class = ast.ClassDef(name="CameraTiming", bases=[], keywords=[], body=timing_methods, decorator_list=[])
        namespace = dict(threading=threading, time=self.clock, datetime=datetime, traceback=traceback)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[acquisition, timing_class], type_ignores=[])),
                     str(path), "exec"), namespace)
        camera = namespace["CameraTiming"]()
        camera._frame_pair_lock = threading.Lock()
        camera._frame_timings = collections.deque(maxlen=8)
        processed_queue, raw_queue = collections.deque(), collections.deque()
        worker = namespace["AcquisitionThread"](camera, [processed_queue], [raw_queue])
        frame = np.ones((4, 6, 3), dtype=np.uint8)
        def snap():
            self.clock.now += 0.3
            worker.running = False
            return frame, frame
        camera.snap = snap
        self.clock.sleep = lambda seconds: setattr(self.clock, "now", self.clock.now + seconds)
        worker.run()
        entry = raw_queue[0]
        self.assertEqual(len(entry), 4)
        timing = camera.get_frame_timing(entry[0], entry[1])
        self.assertEqual(timing["acquisition_started_at"], 100.)
        self.assertAlmostEqual(timing["available_at"], 100.32)
        self.assertEqual(timing["timestamp_basis"], "camera_snap_start")
        self.assertIsNone(camera.get_frame_timing(999, entry[1]))
        timing["acquisition_started_at"] = -1
        self.assertEqual(camera.get_frame_timing(entry[0], entry[1])["acquisition_started_at"], 100.)
        self.controller.calibrated_stage.camera.get_frame_timing = camera.get_frame_timing
        self.helper.refresh_deep_learning(_frame=(entry[0], entry[1], entry[3]),
                                          cell_models=False, pipette_models=True)
        evidence = self.helper.get_deep_learning()
        self.assertEqual(evidence["frame_acquired_at"], 100.)
        self.assertAlmostEqual(evidence["frame_available_at"], 100.32)

    def test_pressure_publishes_value_and_timestamp_as_one_sample(self):
        import threading
        path = ROOT / "patcherbot/devices/pressurecontroller/BasePressureController.py"
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        nodes = [n for n in tree.body if isinstance(n, ast.ClassDef)
                 and n.name in ("PressureAcquisitionThread", "PressureController")]
        namespace = dict(threading=threading, collections=collections, time=self.clock, TaskController=object)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
        pressure = namespace["PressureController"]()
        pressure.measure = lambda: 17.9
        pressure.error = lambda message: self.fail(message)
        callbacks = []
        def measured(value):
            callbacks.append(value)
            worker.running = False
        worker = namespace["PressureAcquisitionThread"](pressure, callback=measured)
        pressure._pressure_acq_thread = worker
        self.clock.sleep = lambda seconds: setattr(self.clock, "now", self.clock.now + seconds)
        worker.run()
        self.assertEqual(pressure.get_last_acquisition(), 17)
        self.assertEqual(pressure.get_last_acquisition_sample(), (17, 100.))
        self.assertEqual(callbacks, [17])
        pressure.set_pressure(20)
        self.controller.pressure = pressure
        self.clock.now = 105.
        sample = self.helper.observe(fields=["pressure", "commanded_pressure_mbar"], raw_measurements=True)
        self.assertEqual(sample["pressure"][0], 17)
        self.assertEqual(sample["field_metadata"]["pressure"]["acquired_at"], 100.)
        self.assertEqual(sample["field_metadata"]["pressure"]["read_at"], 105.)
        self.assertAlmostEqual(sample["field_metadata"]["pressure"]["age_s"], 5.)
        self.assertTrue(sample["field_metadata"]["pressure"]["stale"])
        self.assertEqual(sample["commanded_pressure_mbar"][0], 20)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
