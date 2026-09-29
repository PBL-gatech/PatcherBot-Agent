# patch_gui.py
import faulthandler

from patcherbot.utils import FileLogger
faulthandler.enable()
# faulthandler.dump_traceback_later(5)

import sys
import atexit
from PyQt5.QtWidgets import QApplication, QMessageBox
import traceback
from patcherbot.utils.exception_handler import set_global_exception_hook

# Set the global exception hook
set_global_exception_hook()


from patcherbot.utils.log_utils import setup_logging
from patcherbot.utils.RecordingStateManager import RecordingStateManager
from patcherbot.interface import AutoPatchInterface
from patcherbot.interface.pipettes import CellQueueCoordinator, PipetteInterface
from patcherbot.interface.graph import GraphInterface
from patcherbot.gui.graph import EPhysGUI, EPhysGraph, CurrentProtocolGraph, VoltageProtocolGraph, LeakSubtractionGraph, HoldingProtocolGraph, OptogeneticStimProtocolGraph, OptogeneticWavelengthProtocolGraph
from patcherbot.gui.patch import PatchGui
from rig_setup.rig_config import RigConfigError, RigConfigManager
from rig_setup.rig_selector import RigSelectorDialog
from patcherbot.devices.camera.FakeCalCamera import FakeCalCamera
from patcherbot.utils.CameraRecordingSession import CameraRecordingSession
from patcherbot.utils.GraphRecorder import GraphRecorder

setup_logging()  # Log to the standard console as well

def create_protocol_graphs(graph_interface, recording_state_manager):
    return {
        "current": CurrentProtocolGraph(
            graph_interface,
            recording_state_manager,
        ),
        "voltage": VoltageProtocolGraph(
            graph_interface,
            recording_state_manager,
        ),
        "leak_subtraction": LeakSubtractionGraph(
            graph_interface,
            recording_state_manager,
        ),
        "holding": HoldingProtocolGraph(
            graph_interface,
            recording_state_manager,
        ),
        "optogenetic_stim": OptogeneticStimProtocolGraph(
            graph_interface,
            recording_state_manager,
        ),
        "optogenetic_wavelength": OptogeneticWavelengthProtocolGraph(
            graph_interface,
            recording_state_manager,
        ),
    }

def main():
    """
    Starts the Patch GUI application and prepares the rig for use.

    This function creates the graphical applications, prompts the user to select a rig configuration, 
    builds all required hardware device objects (stage, microscope, amplifier, etc.), and connects them to
    the appropriate controller and interface classes.

    It then creates the main window and signal graph displays, links all components together, and starts the Qt
    event loop so the program can respond to user input.

    If any configuration or hardware initialization step fails, the error is displayed and the program exists
    safely.

    Raises:
        RigConfigError: If the selected rig configuration is invalid or incomplete.
        Exception: If any other error occurs during initialization.
    """
    app = QApplication(sys.argv)
    manager = RigConfigManager()
    manager.ensure_default_config()

    selector = RigSelectorDialog(manager)
    if selector.exec_() != selector.Accepted:
        return

    config_path = selector.selected_path or manager.default_config_path()
    active_pipette_count = selector.selected_pipette_count

    try:
        config_data = manager.load_config(config_path)

        rig_devices = manager.build_devices(
            config_data,
            active_pipette_count=active_pipette_count,
        )
    except RigConfigError as exc:
        QMessageBox.critical(None, "Rig configuration error", str(exc))
        return
    except Exception:
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

    recording_state_manager = RecordingStateManager()

    rig_recorder = FileLogger(
        recording_state_manager,
        folder_path="experiments/Data/rig_recorder_data/",
        recorder_filename="rig_recording",
    )

    movement_recorder = FileLogger(
        recording_state_manager,
        folder_path="experiments/Data/rig_recorder_data/",
        recorder_filename="movement_recording",
    )

    camera_recording_session = CameraRecordingSession(
        recording_state_manager=recording_state_manager,
        main_camera=camera,
        pipette_cameras=pipette_camera,
    )

    calibration_data = config_data.get("calibration") if isinstance(config_data, dict) else None
    patch_data = config_data.get("patch") if isinstance(config_data, dict) else None
    protocol_data = config_data.get("protocol") if isinstance(config_data, dict) else None
    


    if isinstance(unit, dict):
        patch_controllers = {}
        pipette_controllers = {}
        graph_interface = {}
        protocol_graphs = {}

        for i, id in enumerate(unit.keys()):

            curr_amplifier = list(amplifier.values())[i]
            curr_daq = list(daq.values())[i]
            curr_pressure = list(pressure.values())[i]

            pipette_controllers[id] = PipetteInterface(
                stage, microscope, camera, unit[id], cellSorterManip, cellSorterController,
                calibration_data=calibration_data,
                pipette_id=id,
            )

            patch_controllers[id] = AutoPatchInterface(
                curr_amplifier, curr_daq, curr_pressure, pipette_controllers[id], recording_state_manager, lamp, laser,
                config_data=patch_data,
                protocol_data=protocol_data,
            )

            graph_interface[id] = GraphInterface(curr_amplifier, curr_daq, curr_pressure, recording_state_manager, laser)

            protocol_graphs[id] = create_protocol_graphs(graph_interface[id], recording_state_manager)

            # graph_recorders[id] = FileLogger(
            #     recording_state_manager,
            #     folder_path="experiments/Data/rig_recorder_data/",
            #     recorder_filename=f"graph_recording_{id}",
            # )

        graph_recorder = GraphRecorder(
                        recording_state_manager,
                        pipette_ids=graph_interface.keys(),
                    )
    else:
        graph_recorder = GraphRecorder(
            recording_state_manager,
            pipette_ids=["pipette_0"],
        )
        pipette_controllers = PipetteInterface(
            stage, microscope, camera, unit, cellSorterManip, cellSorterController,
            calibration_data=calibration_data,
            pipette_id="pipette_0",
            )
        
        patch_controllers = AutoPatchInterface(
                amplifier, daq, pressure, pipette_controllers, recording_state_manager, lamp, laser,
                config_data=patch_data,
                protocol_data=protocol_data,
            )
        
        graph_interface = GraphInterface(amplifier, daq, pressure, recording_state_manager, laser)

        protocol_graphs = create_protocol_graphs(graph_interface, recording_state_manager)

        graph_recorder = GraphRecorder(
            recording_state_manager,
            pipette_ids=["pipette_0"]
        )

    pipette_interface_map = (
        pipette_controllers
        if isinstance(pipette_controllers, dict)
        else {"pipette_0": pipette_controllers}
    )
    cell_queue = CellQueueCoordinator(
        pipette_interface_map,
        geometry=(patch_data or {}).get("pipette_geometry", []),
        collision_guard_enabled=(patch_data or {}).get("collision_guard_enabled", True),
    )
    patch_interface_map = (
        patch_controllers
        if isinstance(patch_controllers, dict)
        else {"pipette_0": patch_controllers}
    )
    for patch_interface in patch_interface_map.values():
        patch_interface.set_cell_queue(cell_queue)

    gui = PatchGui(camera, pipette_camera, pipette_controllers, patch_controllers, recording_state_manager, camera_recording_session, movement_recorder, rig_recorder, graph_recorder)
    graphs = EPhysGUI(graph_interface, recording_state_manager, graph_recorder)
    # graphs.location_on_the_screen()
    graphs.show()


    gui.initialize()
    gui.show()
    try:
        ret = app.exec_()
    finally:
        graph_recorder.close()
    sys.exit(ret)

if __name__ == "__main__":
    main()
