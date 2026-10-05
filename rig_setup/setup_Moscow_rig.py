'''
This script sets up the Moscow rig. It is used to set up the devices and their connections. The devices are then used in the main script. LEGACY CODE NOW
'''
import serial
from patcherbot.devices.amplifier.multiclamp import MultiClampChannel
from patcherbot.devices.amplifier.DAQ import  NiDAQ
from patcherbot.devices.camera.pcocamera import PcoCamera
from patcherbot.devices.camera.PipetteCamera import PipetteCamera
from patcherbot.devices.pressurecontroller import MoscowPressureController
from patcherbot.devices.manipulator import *
from patcherbot.devices.cellsorter import FakeCellSorterController, FakeCellSorterManip
from patcherbot.devices.lamp import OlympusLamp, FakeLamp, lumencor


# set up Camera
camera = PcoCamera()
pipette_camera = PipetteCamera()

# set up Pressure Controller
pressureControllerSerial = serial.Serial(port='COM5', baudrate=9600, timeout=0)
pressureReaderSerial = serial.Serial(port='COM9', baudrate=9600, timeout=0)
pressure = MoscowPressureController(channel=4, controllerSerial=pressureControllerSerial, readerSerial=pressureReaderSerial)


# set up Ephys
amplifier = MultiClampChannel(channel=1)
daq = NiDAQ('cDAQ1Mod1', 'ai0', 'cDaq1Mod4', 'ao0', 'cDaq1Mod1', 'ai3')

# set up movement controllers
print("Setting up stage controllers...")
controllerSerial = serial.Serial('COM6',baudrate=9600)
controller = ScientificaSerialNoEncoder(controllerSerial)
microscope = Microscope(controller, 3)
microscope.up_direction = 1.0

print("Setting up pipette controllers...")
pipetteSerial = serial.Serial('COM3',baudrate=9600)
pipetteManip = ScientificaSerialNoEncoder(pipetteSerial)
stage = ManipulatorUnit(controller, [1, 2])
unit = ManipulatorUnit(pipetteManip, [1, 2, 3])

# set up cell sorter
cellSorterController = FakeCellSorterController()
cellSorterManip = FakeCellSorterManip()

# set up lamp
lamp = OlympusLamp('COM21')  





