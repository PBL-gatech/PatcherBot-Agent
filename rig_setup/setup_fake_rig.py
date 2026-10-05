'''
"Fake setup" for GUI development on a computer without access to a rig LEGACY CODE NOW
'''
import numpy as np
from patcherbot.devices.amplifier.amplifier import FakeAmplifier
from patcherbot.devices.amplifier.DAQ import FakeDAQ
from patcherbot.devices.camera.pcocamera import PcoCamera
from patcherbot.devices.pressurecontroller.BasePressureController import FakePressureController
from patcherbot.devices.camera.camera import FakeCamera
from patcherbot.devices.camera import FakeCalCamera, FakePipetteManipulator
from patcherbot.devices.manipulator import *
from patcherbot.devices.cellsorter import FakeCellSorterController, FakeCellSorterManip
from patcherbot.devices.lamp import FakeLamp

controller = FakeManipulator(min=[-240000, 50000, 280000],
                             max=[-230000, 60000, 290000])
pipetteManip = FakeManipulator(min=[0, 0, 0],
                                      max=[4000, 20000, 20000])
stage = ManipulatorUnit(controller, [1, 2])

cellSorterController = FakeCellSorterController()
cellSorterManip = FakeCellSorterManip()


pipetteManip.x = np.array([200, 300, 400], dtype=np.float64) # start with pipette in frame
controller.x = np.array([-235000, 55000, 285000], dtype=np.float64)
camera = FakeCalCamera(stageManip=controller, pipetteManip=pipetteManip, image_z=100, cellSorterManip=cellSorterManip)
pipette_camera = FakeCamera()
microscope = Microscope(controller, 3)
microscope.up_direction = 1.0

unit = ManipulatorUnit(pipetteManip, [1, 2, 3])

daq = FakeDAQ()
amplifier = FakeAmplifier()
pressure = FakePressureController()

lamp = FakeLamp()