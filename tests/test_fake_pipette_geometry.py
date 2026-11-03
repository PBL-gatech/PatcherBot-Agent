import importlib

import numpy as np
import pytest
from PIL import Image

import patcherbot.interface
from patcherbot.devices.camera.FakeCalCamera import FakePipette, FakePipetteManipulator
from patcherbot.devices.manipulator.fakemanipulator import FakeManipulator


def test_rotated_fake_pipette_moves_in_world_coordinates():
    pipette = FakePipetteManipulator(
        min=[-10000, -10000, -10000],
        max=[10000, 10000, 10000],
        armAngle=0,
    )
    pipette.x = np.array([10.0, 20.0, 30.0])

    np.testing.assert_allclose(pipette.position(), [10, 20, 30], atol=1e-5)
    pipette.absolute_move(50, axis=1)
    pipette.wait_until_still([1, 2, 3])

    np.testing.assert_allclose(pipette.position(), [50, 20, 30], atol=1e-5)


def test_fake_pipette_approach_direction_comes_from_starting_geometry():
    positions = ([560, 320, 100], [260, 320, 100], [260, 500, 100])
    pipettes = []
    for position in positions:
        controller = FakeManipulator(min=[0, 0, 0], max=[4000, 20000, 20000])
        controller.x = np.asarray(position, dtype=float)
        pipettes.append(FakePipette(controller, microscope_pixels_per_micron=1.25))

    assert pipettes[0].geometry_direction[0] > 0
    assert pipettes[0].geometry_direction[1] < 0
    assert pipettes[1].geometry_direction[0] < 0
    assert pipettes[1].geometry_direction[1] < 0
    assert pipettes[2].geometry_direction[0] < 0
    assert pipettes[2].geometry_direction[1] > 0


def test_fake_camera_detector_returns_rendered_tip_without_loading_model(monkeypatch):
    controller = FakeManipulator(min=[0, 0, 0], max=[4000, 20000, 20000])
    controller.x = np.asarray([40.0, 60.0, 80.0])
    pipette = FakePipette(controller, microscope_pixels_per_micron=1.25)
    pipette.add_pipette_to_img(Image.new("L", (1024, 1024)), [0, 0, 0])
    camera = type("FakeCameraStub", (), {
        "get_fake_pipette_tip_xy": lambda self: pipette.last_tip_xy,
    })()
    helper_module = importlib.import_module(
        "patcherbot.devices.manipulator.helpers.PipetteCalHelper"
    )

    def unexpected_model_load(*args, **kwargs):
        pytest.fail("Fake camera must not initialize a learned detector")

    monkeypatch.setattr(helper_module, "PipetteDetector5", unexpected_model_load)
    helper = helper_module.PipetteCalHelper(None, None, camera, None)

    assert helper.pipetteDetector.detect_pipette(None) == pipette.last_tip_xy
    assert helper.pipetteDetector.get_pipette_z(None) == 0.0
    assert helper.pipetteDetector.get_pipette_z(None) == 0.0