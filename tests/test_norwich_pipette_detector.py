"""Norwich detector routing and compatibility without opening rig hardware."""
import importlib
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

import patcherbot.interface  # Initialize package imports before manipulator helpers.
from patcherbot.configs.CalibrationConfig import CalibrationConfig
from patcherbot.deepLearning.pipetteDetector import PipetteDetectorYOLO1
from rig_setup.rig_config import RigConfigManager


helper_module = importlib.import_module(
    "patcherbot.devices.manipulator.helpers.PipetteCalHelper"
)
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "patcherbot" / "deepLearning" / "pipetteModel"


def norwich_config(filename="Norwich_rig.json"):
    resolved = RigConfigManager().load_config(
        ROOT / "rig_setup" / "rig_configs" / filename
    )["_resolved_calibration"]
    config = CalibrationConfig()
    config.from_dict(resolved)
    return config


class NorwichDetectorRoutingTests(unittest.TestCase):
    def setUp(self):
        for patcher in (
            patch.dict(os.environ, {"PIPETTE_DETECTOR_DEVICE": ""}),
            patch.object(helper_module.torch.cuda, "is_available", return_value=False),
            patch.object(helper_module.torch.cuda, "device_count", return_value=0),
            patch.object(helper_module, "configure_pipette_detector"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_both_norwich_configs_keep_detector_selection_after_overlay(self):
        for filename in ("Norwich_rig.json", "Norwich_multiple_rig.json"):
            with self.subTest(filename=filename):
                config = norwich_config(filename)
                self.assertEqual(config.pipette_detector_type, "yolo1")
                self.assertEqual(config.pipette_detector_device, "auto")
                self.assertEqual(config.pipette_detector_model, "BoRigCombo.pt")
                self.assertEqual(config.pipette_focuser_model, "regression_model2.onnx")

    def test_norwich_loads_matching_detector_and_focus_without_switching(self):
        raw = SimpleNamespace(model_path=MODEL_DIR / "BoRigCombo.pt", threshold=.2)
        focuser = Mock()
        with patch.object(helper_module, "PipetteDetectorYOLO1", return_value=raw) as yolo, \
                patch.object(helper_module, "PipetteFocuser", return_value=focuser) as focus, \
                patch.object(helper_module, "PipetteDetector5") as v5:
            helper = helper_module.PipetteCalHelper(None, None, object(), None, norwich_config())
        yolo.assert_called_once_with(model_path=MODEL_DIR / "BoRigCombo.pt", device="cpu")
        self.assertEqual(Path(focus.call_args.kwargs["model_path"]), MODEL_DIR / "regression_model2.onnx")
        v5.assert_not_called()
        selected = helper.pipetteDetector
        identity = selected._active_model
        self.assertEqual(selected.model_version, "yolo1")
        self.assertIs(selected.adapter, raw)
        with helper.brain_slice_detector():
            self.assertIs(helper.pipetteDetector, selected)
            self.assertEqual(selected._active_model, identity)
        self.assertEqual(selected._active_model, identity)

    def test_norwich_loading_failure_does_not_fall_back_to_r02(self):
        with patch.object(helper_module, "PipetteDetectorYOLO1", side_effect=RuntimeError("bad weights")), \
                patch.object(helper_module, "PipetteDetector5") as v5:
            with self.assertRaisesRegex(RuntimeError, "bad weights"):
                helper_module.PipetteCalHelper(None, None, object(), None, norwich_config())
        v5.assert_not_called()

    def test_default_real_camera_keeps_v5(self):
        with patch.object(helper_module, "PipetteDetector5") as v5, \
                patch.object(helper_module, "PipetteDetectorYOLO1") as yolo:
            helper = helper_module.PipetteCalHelper(None, None, object(), None, CalibrationConfig())
        self.assertIs(helper.pipetteDetector, v5.return_value)
        self.assertEqual(v5.call_args.kwargs["model_version"], "R02")
        yolo.assert_not_called()

    def test_fake_camera_still_skips_learned_models(self):
        camera = SimpleNamespace(get_fake_pipette_tip_xy=lambda pipette: (30, 40))
        with patch.object(helper_module, "PipetteDetector5") as v5, \
                patch.object(helper_module, "PipetteDetectorYOLO1") as yolo, \
                patch.object(helper_module, "PipetteFocuser") as focus:
            helper = helper_module.PipetteCalHelper(None, None, camera, None, norwich_config())
        self.assertEqual(helper.pipetteDetector.detect_pipette(None), (30, 40))
        self.assertEqual(helper.pipetteDetector.get_pipette_z(None), 0.)
        for model in (v5, yolo, focus):
            model.assert_not_called()


class NorwichPredictionTests(unittest.TestCase):
    def setUp(self):
        self.detector = PipetteDetectorYOLO1.__new__(PipetteDetectorYOLO1)
        self.detector.model_path = MODEL_DIR / "BoRigCombo.pt"
        self.detector.pipette_class = 0
        self.detector.imgsz = 640
        self.detector.conf_threshold = .2
        self.detector.device = "cpu"
        self.detector.use_half = False
        boxes = SimpleNamespace(
            cls=torch.tensor([0, 0, 0]),
            conf=torch.tensor([.99, .6, .8]),
            xywhn=torch.tensor([[.8, .8, .1, .1], [.25, .25, .1, .1], [.28, .25, .1, .1]]),
        )
        self.detector.yolo_model = Mock()
        self.detector.yolo_model.predict.return_value = [SimpleNamespace(boxes=boxes)]
        self.image = np.zeros((100, 200), dtype=np.uint8)

    def test_details_filter_radius_and_preserve_actual_confidence(self):
        options = dict(expected_tip_xy=(50, 25), pixels_to_um=lambda delta: delta, radius_um=10.)
        result = self.detector.detect_pipette_details(self.image, **options)
        self.assertEqual(result["tip_xy"], (56, 25))
        self.assertAlmostEqual(result["box_confidence"], .8, places=6)
        self.assertIsNone(result["z_um"])
        self.assertFalse(result["z_confidence_supported"])
        anchored = self.detector.detect_pipette_details(self.image, prefer_anchor=True, **options)
        self.assertEqual(anchored["tip_xy"], (50, 25))
        rejected = self.detector.detect_pipette_details(self.image, **dict(options, radius_um=1., expected_tip_xy=(10, 10)))
        self.assertIsNone(rejected["tip_xy"])

    def test_point_api_retains_highest_confidence_prediction(self):
        self.assertEqual(self.detector.detect_pipette(self.image), (160, 80))
        self.assertIsNone(self.detector.detect_pipette(None))

    def test_focus_adapter_supplies_separate_signed_depth(self):
        focuser = Mock()
        focuser.get_pipette_focus_value.return_value = -3.25
        focused = helper_module._FocusedYOLOPipetteDetector(self.detector, focuser)
        details = focused.detect_pipette_details(self.image)
        self.assertEqual(details["tip_xy"], (160, 80))
        self.assertAlmostEqual(details["box_confidence"], .99, places=6)
        self.assertEqual(details["z_um"], -3.25)
        self.assertEqual(focused.get_pipette_z(self.image), -3.25)


if __name__ == "__main__":
    unittest.main()
