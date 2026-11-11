# patch_gui.py
import faulthandler
faulthandler.enable()
# faulthandler.dump_traceback_later(5)

import sys
import atexit
import logging
import os
from pathlib import Path
from time import perf_counter
from importlib.metadata import version, PackageNotFoundError
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMessageBox
import traceback
from patcherbot.utils.exception_handler import set_global_exception_hook

# Set the global exception hook
set_global_exception_hook()


from patcherbot.utils.log_utils import setup_logging
from patcherbot.utils.RecordingStateManager import RecordingStateManager
from patcherbot.interface import AutoPatchInterface
from patcherbot.interface.pipettes import PipetteInterface
from patcherbot.interface.graph import GraphInterface
from patcherbot.gui.graph import EPhysGraph, CurrentProtocolGraph, VoltageProtocolGraph, LeakSubtractionGraph, HoldingProtocolGraph, OptogeneticStimProtocolGraph, OptogeneticWavelengthProtocolGraph
from patcherbot.gui.patch import PatchGui
from rig_setup.rig_config import RigConfigError, RigConfigManager
from rig_setup.rig_selector import RigSelectorDialog
from patcherbot.devices.camera.FakeCalCamera import FakeCalCamera

setup_logging()  # Log to the standard console as well

STARTUP_TRACEBACK_SECONDS = 30
LOGGER = logging.getLogger(__name__)


def _startup_step(label, function, *args, **kwargs):
    started = perf_counter()
    LOGGER.info("Startup: BEGIN %s", label)
    try:
        result = function(*args, **kwargs)
    except BaseException:
        LOGGER.exception("Startup: FAILED %s after %.3f s", label, perf_counter() - started)
        raise
    LOGGER.info("Startup: DONE %s in %.3f s", label, perf_counter() - started)
    return result


def _log_startup_environment():
    LOGGER.info("Startup environment: pid=%s; executable=%s; Python=%s; prefix=%s",
                os.getpid(), sys.executable, sys.version.replace("\n", " "), sys.prefix)
    LOGGER.info("Startup environment: CONDA_DEFAULT_ENV=%s; CONDA_PREFIX=%s; cwd=%s",
                os.environ.get("CONDA_DEFAULT_ENV"), os.environ.get("CONDA_PREFIX"), os.getcwd())
    LOGGER.info("Startup environment: Qt=%s; PyQt=%s", QtCore.qVersion(), QtCore.PYQT_VERSION_STR)
    for name in ("numpy", "torch", "cv2", "PyQt5.QtCore"):
        module = sys.modules.get(name)
        if module is not None:
            LOGGER.info("Startup environment: loaded %s from %s", name, getattr(module, "__file__", None))
    torch_module = sys.modules.get("torch")
    if torch_module is not None:
        LOGGER.info("Startup environment: torch CUDA build=%s",
                    getattr(getattr(torch_module, "version", None), "cuda", None))
    for package in ("numpy", "scipy", "PyQt5", "pyqtgraph", "torch", "torchvision", "opencv-python", "pco"):
        try:
            installed = version(package)
        except PackageNotFoundError:
            installed = "distribution metadata not found"
        except Exception as exc:
            installed = f"unable to read distribution metadata: {exc}"
        LOGGER.info("Startup environment: %s=%s", package, installed)


def _start_startup_watchdog():
    trace_file = None
    try:
        trace_path = Path(__file__).resolve().parent / "experiments" / "Data" / "log_data" / f"startup_traceback_{os.getpid()}.log"
        trace_file = trace_path.open("a", encoding="utf-8")
        trace_file.write(f"Startup diagnostic for PID {os.getpid()}; executable={sys.executable}\n")
        trace_file.flush()
    except OSError:
        if trace_file is not None:
            try:
                trace_file.close()
            except OSError:
                pass
            trace_file = None
        LOGGER.exception("Startup: could not open traceback log; using stderr")
    try:
        faulthandler.dump_traceback_later(
            STARTUP_TRACEBACK_SECONDS, repeat=True, file=trace_file, exit=False)
    except (OSError, RuntimeError, ValueError):
        if trace_file is not None:
            trace_file.close()
        LOGGER.exception("Startup: could not enable traceback watchdog")
        return lambda: None
    LOGGER.info("Startup: stack traces every %s s if startup stalls; output=%s",
                STARTUP_TRACEBACK_SECONDS, trace_file.name if trace_file is not None else "stderr")
    finished = False

    def finish():
        nonlocal finished
        if finished:
            return
        finished = True
        faulthandler.cancel_dump_traceback_later()
        if trace_file is not None:
            trace_file.close()

    atexit.register(finish)
    return finish


def main():
    _log_startup_environment()
    app = _startup_step("QApplication", QApplication, sys.argv)
    manager = _startup_step("RigConfigManager", RigConfigManager)
    _startup_step("ensure default rig configuration", manager.ensure_default_config)

    selector = _startup_step("RigSelectorDialog", RigSelectorDialog, manager)
    LOGGER.info("Startup: waiting for rig selection")
    if selector.exec_() != selector.Accepted:
        LOGGER.info("Startup: rig selection cancelled")
        return
    config_path = selector.selected_path or manager.default_config_path()
    LOGGER.info("Startup: selected rig configuration %s", config_path)
    finish_watchdog = _start_startup_watchdog()

    try:
        config_data = _startup_step("load rig configuration", manager.load_config, config_path)
        rig_devices = _startup_step("build rig devices", manager.build_devices, config_data)
    except RigConfigError as exc:
        finish_watchdog()
        QMessageBox.critical(None, "Rig configuration error", str(exc))
        return
    except Exception:
        finish_watchdog()
        QMessageBox.critical(None, "Rig initialization failed", traceback.format_exc())
        return

    stage = rig_devices["stage"]
    microscope = rig_devices["microscope"]
    camera = rig_devices["camera"]
    # Ensure camera has refs if it needs them
    if isinstance(camera, FakeCalCamera):
        camera.stageManip = rig_devices["stage_controller"]
        camera.pipetteManip = rig_devices["pipette_controller"]
        camera.cellSorterManip = rig_devices["cell_sorter_manipulator"]
    pipette_camera = rig_devices["pipette_camera"]
    unit = rig_devices["pipette_unit"]
    cellSorterManip = rig_devices["cell_sorter_manipulator"]
    cellSorterController = rig_devices["cell_sorter_controller"]
    amplifier = rig_devices["amplifier"]
    daq = rig_devices["daq"]
    pressure = rig_devices["pressure"]
    lamp = rig_devices["lamp"]
    laser = rig_devices["laser"]

    recording_state_manager = _startup_step("RecordingStateManager", RecordingStateManager)
    calibration_data = None
    patch_data = None
    protocol_data = None
    if isinstance(config_data, dict):
        calibration_data = config_data.get("_resolved_calibration", config_data.get("calibration"))
        patch_data = config_data.get("_resolved_patch", config_data.get("patch"))
        protocol_data = config_data.get("_resolved_protocol", config_data.get("protocol"))

    pipette_controller = _startup_step("PipetteInterface", PipetteInterface,
        stage,
        microscope,
        camera,
        unit,
        cellSorterManip,
        cellSorterController,
        calibration_data=calibration_data,
    )
    patch_controller = _startup_step("AutoPatchInterface", AutoPatchInterface,
        amplifier,
        daq,
        pressure,
        pipette_controller,
        recording_state_manager,
        lamp,
        laser=laser,
        config_data=patch_data,
        protocol_data=protocol_data,
    )
    graph_interface = _startup_step("GraphInterface", GraphInterface, amplifier, daq, pressure, recording_state_manager, laser=laser)
    gui = _startup_step("PatchGui", PatchGui, camera, pipette_camera, pipette_controller, patch_controller, recording_state_manager)
    graphs = _startup_step("EPhysGraph", EPhysGraph, graph_interface, recording_state_manager)
    # graphs.location_on_the_screen()
    _startup_step("show EPhysGraph", graphs.show)

    currentProtocolGraph = _startup_step("CurrentProtocolGraph", CurrentProtocolGraph, graph_interface, recording_state_manager)
    voltageProtocolGraph = _startup_step("VoltageProtocolGraph", VoltageProtocolGraph, graph_interface, recording_state_manager)
    leakSubtractionGraph = _startup_step("LeakSubtractionGraph", LeakSubtractionGraph, graph_interface, recording_state_manager)
    holdingProtocolGraph = _startup_step("HoldingProtocolGraph", HoldingProtocolGraph, graph_interface, recording_state_manager)
    optogeneticStimProtocolGraph = _startup_step("OptogeneticStimProtocolGraph", OptogeneticStimProtocolGraph, graph_interface, recording_state_manager)
    optogeneticWavelengthProtocolGraph = _startup_step("OptogeneticWavelengthProtocolGraph", OptogeneticWavelengthProtocolGraph, graph_interface, recording_state_manager)


    _startup_step("initialize PatchGui", gui.initialize)
    _startup_step("show PatchGui", gui.show)

    def startup_ready():
        LOGGER.info("Startup: Qt event loop responsive; stopping startup traceback watchdog")
        finish_watchdog()

    QtCore.QTimer.singleShot(1000, startup_ready)
    LOGGER.info("Startup: entering Qt event loop")
    try:
        ret = app.exec_()
    finally:
        finish_watchdog()
    LOGGER.info("Startup: Qt event loop exited with code %s", ret)
    sys.exit(ret)

if __name__ == "__main__":
    main()
