# coding=utf-8
'''
Control of automatic patch clamp algorithm
'''
import numpy as np

from patcherbot.interface import TaskInterface, command, blocking_command
from patcherbot.controller import AutoPatcher
from patcherbot.utils import EPhysLogger, RecordingStateManager
from patcherbot.devices.pressurecontroller.BasePressureController import PressureController
from patcherbot.devices.amplifier.amplifier import Amplifier
from patcherbot.interface.pipettes import PipetteInterface
from patcherbot.devices.amplifier.DAQ import DAQ
from patcherbot.devices.lamp import Lamp
from patcherbot.devices.laser import Laser
from .patchConfig import PatchConfig
from .experimentBookConfig import ExperimentBookConfig
from .protocolConfig import ProtocolConfig
from PyQt5 import QtCore
import time

__all__ = ['AutoPatchInterface']

class AutoPatchInterface(TaskInterface):
    '''
    A class to run automatic patch-clamp
    '''
    def __init__(self, amplifier: Amplifier, daq: DAQ, pressure: PressureController, pipette_interface: PipetteInterface,
                 recording_state_manager: RecordingStateManager, lamp: Lamp, laser: Laser, config_data=None, protocol_data=None):
        """
        Initialize the AutoPatchInterface with hardware devices and configuration data.

        Args:
            amplifier (Amplifier): Amplifier device.
            daq (DAQ): Data acquisition device.
            pressure (PressureController): Pressure controller.
            pipette_interfaces (PipetteInterface): Pipette control interfaces.
            recording_state_manager (RecordingStateManager): Recording state manager.
            lamp (Lamp): Lamp device for illumination control.
            laser (Laser): Laser device for optogenetics.
            config_data (dict, optional): Configuration values for patching.
            protocol_data (dict, optional): Configuration values for protocols.
        """
        super().__init__()
        self.config = PatchConfig(name='Patch')
        if config_data:
            cleaned = {k: v for k, v in config_data.items() if v is not None}
            self.config.from_dict(cleaned)
        self.protocol_config = ProtocolConfig(name='Protocols')
        if protocol_data:
            cleaned = {k: v for k, v in protocol_data.items() if v is not None}
            self.protocol_config.from_dict(cleaned)
        self.amplifier = amplifier
        self.daq = daq
        self.pressure = pressure
        self.pipette_controller = pipette_interface
        self.pipette_id = getattr(pipette_interface, "pipette_id", "pipette_0")
        self.pipette_index = getattr(pipette_interface, "pipette_index", 0)
        self.recording_state_manager = recording_state_manager
        self.lamp = lamp
        self.laser = laser
        self.ephys_logger = EPhysLogger(recording_state_manager=self.recording_state_manager, ephys_filename="CellMetadata")
        self.autopatcher = AutoPatcher(amplifier, daq, pressure, self.pipette_controller.calibrated_unit,
                                    self.pipette_controller.calibrated_unit.microscope,
                                    calibrated_stage=self.pipette_controller.calibrated_stage,
                                    lamp=self.lamp,
                                    laser=self.laser,
                                    config=self.config,
                                    protocol_config=self.protocol_config)

        self.is_selecting_cells = False
        self.is_selecting_corners = False
        self._cell_queue = None
        self._local_cells_to_patch = []
        self.movement_file_path = ''

        self._last_status_msg = None
        self._last_error_msg = None
        self._last_warning_msg = None
        self.last_log_time = 0

        #call update_camera_cell_list every 0.05 seconds using a QTimer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_camera_cell_list)
        self.timer.start(50)

    def _microscope_z_um(self) -> float:
        """
        Retrieve the current microscope Z position in micrometers.

        Returns:
            float: Current Z position.
        """
        return float(self.pipette_controller.calibrated_unit.microscope.position())

    def _protocol_holding_parameters(self):
        """
        Extract holding voltage and current parameters from protocol configuration.

        Returns:
            tuple: (voltage_hold, current_hold) values.
        """
        protocol_config = self.autopatcher.protocol_config
        voltage_hold = float("nan")
        current_hold = float("nan")

        if getattr(protocol_config, "voltage_protocol", False):
            try:
                voltage_hold = float(protocol_config.vclamp_hold) * 1e3
            except (TypeError, ValueError):
                voltage_hold = float("nan")

        if getattr(protocol_config, "current_protocol", False):
            try:
                current_hold = float(protocol_config.cclamp_hold)
            except (TypeError, ValueError):
                current_hold = float("nan")

        return voltage_hold, current_hold

    def _wait_for_filter_slot(self, target_slot, timeout_s=2.0, poll_interval_s=0.05):
        """
        Wait until the lamp filter reaches a target slot or timeout occurs.

        Args:
            target_slot (object): Desired filter slot.
            timeout_s (float, optional): Maximum wait time in seconds.
            poll_interval_s (float, optional): Polling interval in seconds.
        """
        if target_slot is None:
            return
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            current_slot = self.autopatcher.lamp.get_filter()
            if current_slot == target_slot:
                return
            time.sleep(poll_interval_s)
        self.warning(f"Timed out waiting for lamp filter to reach slot {target_slot}")

    def _unpack_cell_entry(self, cell_entry):
        if cell_entry is None:
            return None, None, None, None
        if len(cell_entry) >= 4:
            return cell_entry[0], cell_entry[1], cell_entry[2], cell_entry[3]
        if len(cell_entry) == 3:
            return cell_entry[0], cell_entry[1], cell_entry[2], None
        return None, None, None, None

    @property
    def cells_to_patch(self):
        return self._cell_queue.cells if self._cell_queue is not None else self._local_cells_to_patch

    @cells_to_patch.setter
    def cells_to_patch(self, cells):
        self._local_cells_to_patch = list(cells)

    def set_cell_queue(self, queue):
        self._cell_queue = queue

    def _next_cell(self, mark_attempted=False):
        if self._cell_queue is None:
            cell = self._local_cells_to_patch[0] if self._local_cells_to_patch else None
        else:
            cell = self._cell_queue.next_cell(self.pipette_id)
        if cell is not None and mark_attempted and self._cell_queue is not None:
            self._cell_queue.mark_attempted(cell)
        return cell

    def _remove_cell(self, cell):
        if self._cell_queue is not None:
            return self._cell_queue.remove_cell(cell)
        previous_count = len(self._local_cells_to_patch)
        self._local_cells_to_patch = [entry for entry in self._local_cells_to_patch if entry is not cell]
        return len(self._local_cells_to_patch) != previous_count

    
    @blocking_command(category='Patch', description='Break into the cell',
                      task_description='Breaking into the cell')
    
    def break_in(self):
        """Perform membrane break-in for a patched cell."""
        self.recording_state_manager.increment_sample_number()
        self.execute(self.autopatcher.break_in)

    @blocking_command(category='Patch', description='GigaSeal the cell',
                      task_description='GigaSealing the cell')
    def gigaseal(self):
        """Attempt to form a gigaohm seal with the cell membrane."""
        self.recording_state_manager.increment_sample_number()
        self.execute(self.autopatcher.gigaseal)

    def start_selecting_cells(self):
        """Enable cell selection mode for adding cells to the patch queue."""
        self.is_selecting_cells = True

    def start_selecting_corners(self):
        self.is_selecting_corners = True
        self.pipette_controller.calibrated_stage.start_selecting_scan_corners()

    @blocking_command(category='Stage',
                      description='Scan the stored corner coordinates',
                      task_description='Scanning plate area')
    def start_scan(self):
        self.execute(self.autopatcher.scan_area)

    @blocking_command(category='Stage',
                      description='Move stage to the stored scan start coordinate',
                      task_description='Moving to scan start')
    def move_to_scan_start(self):
        self.execute(self.pipette_controller.calibrated_stage.move_to_scan_start)


    @blocking_command(category='Microscope',
                      description='Switch objective',
                      task_description='Switching objective')
    def toggle_objective(self):
        self.execute(self.pipette_controller.microscope.toggle_objective)

    @command(category='Patch',
             description='Select a corner on right-click while Store Corners is active')
    def handle_corner_right_click(self, position):
        if self.is_selecting_corners:
            corners_complete = self.pipette_controller.calibrated_stage.store_scan_corner(position)
            if corners_complete:
                self.is_selecting_corners = False
            return
        self.execute(self.pipette_controller.move_stage, argument=position)

    def remove_last_cell(self):
        """Remove the most recently added cell from the patch queue."""
        if self._cell_queue is not None:
            last_cell = self._cell_queue.remove_last_cell()
            if last_cell is not None:
                self.info(f"Removed last queued cell {last_cell}")
            return
        if self._local_cells_to_patch:
            last_cell = self._local_cells_to_patch[-1]
            self.info(
                f"removing last cell {last_cell}... {self.is_selecting_cells}, at {self._local_cells_to_patch}"
            )
            self._local_cells_to_patch = self._local_cells_to_patch[:-1]
        else:
            self.info("remove_last_cell called but there are no cells to remove")


    @blocking_command(category='Cell Sorter',
            description='Move the cell sorter to a cell',
            task_description='Move the cell sorter to a cell')
    def move_cellsorter_to_cell(self):
        """Move the cell sorter to the first cell in the queue."""
        cell_entry = self._next_cell()
        if cell_entry is None:
            self.warning("No cells assigned to this pipette")
            return
        cellx = cell_entry[0][0]
        celly = cell_entry[0][1]
        cellz = self.pipette_controller.calibrated_unit.microscope.floor_Z

        #move cell sorter to cell
        self.execute(self.pipette_controller.calibrated_cellsorter.center_cellsorter_on_point, argument=[cellx, celly, cellz])

    @command(category='Cell Sorter',
            description='Turn cell sorter LED on')
    def cell_sorter_led_on(self):
        """Turn on the cell sorter LED."""
        self.pipette_controller.calibrated_cellsorter.set_led_status(True, ring=1)

    @command(category='Cell Sorter',
            description='Turn cell sorter LED off')
    def cell_sorter_led_off(self):
        """Turn off the cell sorter LED."""
        self.pipette_controller.calibrated_cellsorter.set_led_status(False, ring=1)

    @blocking_command(category='DAQ',
            description='Run Protocols on the Cell',
            task_description='Run Protocols on the Cell')
    def run_protocols(self):
        """Execute electrophysiology protocols on the selected cell and log metadata."""
        self.recording_state_manager.increment_sample_number()
        index = self.recording_state_manager.sample_number
        cell_entry = self._next_cell()
        if cell_entry is not None:
            stage_coords, img, stage_coords_um, img_fluo = cell_entry
            voltage_hold, current_hold = self._protocol_holding_parameters()
            self.ephys_logger.save_cell_metadata(
                index,
                stage_coords_um,
                img,
                image_fluo=img_fluo,
                voltage_hold=voltage_hold,
                current_hold=current_hold,
            )
        self.execute(self.autopatcher.run_protocols)

    @blocking_command(category='DAQ',
            description='Run Optogenetic Protocol',
            task_description='Running Optogenetic Protocol')
    def run_optogenetic_protocol(self, protocol_params=None):
        """
        Execute an optogenetic stimulation protocol.

        Args:
            protocol_params (object, optional): Parameters for the protocol.
        """
        self.recording_state_manager.increment_sample_number()
        self.execute(self.autopatcher.run_optogenetic_protocol, argument=protocol_params)
    

    @command(category='Patch', description='Add a mouse position to the list of cells to patch')
    def add_cell(self, position):
        """
        Add a selected cell position and associated image data to the patch queue.

        Args:
            position (array-like): Mouse click position in image coordinates.

        Raises:
            RuntimeError: If the extracted image is invalid or near image boundaries.
        """
        #add half the size of the camera image to the position to get the center of the cell
        position = np.array(position)

        position[0] += self.autopatcher.calibrated_unit.camera.width / 2
        position[1] += self.autopatcher.calibrated_unit.camera.height / 2
        # add the z position of the microscope
        z_scale = self.autopatcher.calibrated_unit.config.microscope_units_per_um
        z_pos = self.autopatcher.calibrated_unit.microscope.position() / z_scale
        self.info(f'z position of the microscope: {z_pos}')
        print(f'adding cell... {self.is_selecting_cells}')
        if self.is_selecting_cells:
            self.info(f'Adding cell at {position} to list of cells to patch in pixels')
            stage_pos_pixels = self.autopatcher.calibrated_stage.reference_position()
            stage_pos_pixels[0:2] -= position
            # display stage position
            # add the z_pos to the stage position as a third dimension in the np array

            stage_pos_pixels = np.array([stage_pos_pixels[0], stage_pos_pixels[1], z_pos])
            # get raw currnent stage position in pixels
            stage_pos_plane = self.autopatcher.calibrated_stage.position()
            stage_pos_um = np.array([stage_pos_plane[0], stage_pos_plane[1], z_pos])
            print(f'Stage position in pixels: {stage_pos_pixels}')
            print(f'Stage position in um: {stage_pos_um}')



            # print(f'Stage position dimensions: {np.size(stage_pos_pixels)}')
            # print(f'Stage um position: {stage_pos_pixels}')
            #take a 512x512 image centered on the cell
            img = self.autopatcher.calibrated_unit.camera.get_16bit_image()
            img = img[int(position[1]-256):int(position[1]+256), int(position[0]-256):int(position[0]+256)]
            if img is None or img.shape != (512, 512):
                raise RuntimeError('Cell too Close to edge!')
            
            img_fluo = None
            if self.config.auto_capture_fluo:
                fluo_slot = int(self.config.lamp)
                current_slot = self.autopatcher.lamp.get_filter()
                did_toggle = current_slot != fluo_slot
                if did_toggle:
                    self.autopatcher.toggle_fluorescence()
                    self._wait_for_filter_slot(fluo_slot)
                try:
                    img_fluo = self.autopatcher.calibrated_unit.camera.get_16bit_image()
                    img_fluo = img_fluo[int(position[1]-256):int(position[1]+256), int(position[0]-256):int(position[0]+256)]
                    if img_fluo is None or img_fluo.shape != (512, 512):
                        raise RuntimeError('Cell too Close to edge!')
                finally:
                    if did_toggle:
                        self.autopatcher.toggle_fluorescence()
                        if current_slot is not None:
                            self._wait_for_filter_slot(current_slot)

            cell_entry = (np.array(stage_pos_pixels), img, stage_pos_um, img_fluo)
            if self._cell_queue is not None:
                self._cell_queue.add_cell(cell_entry)
            else:
                self._local_cells_to_patch.append(cell_entry)
            self.is_selecting_cells = False

    # Update the cell list to store both cell coordinates and image.
    def update_camera_cell_list(self) -> None:
        """Update the camera's internal list of detected cells for display."""
        self.autopatcher.calibrated_unit.camera.cell_list = []
        for cell, img, pos, _img_fluo in self.cells_to_patch:
            camera_pos = -cell + self.autopatcher.calibrated_stage.reference_position()
            self.autopatcher.calibrated_unit.camera.cell_list.append((camera_pos[0:2].astype(int), img,pos))
            
    @command(category='Test', 
             description='Send movement file to the autopatcher', 
             success_message='Path sent')
    def send_movement_file(self, path):
        """
        Send a movement file to the autopatcher for execution.

        Args:
            path (str): Path to the movement file.
        """
        self.autopatcher.test_movement(path)
            # Append a tuple of (coordinates, image)
        self.info(f'Movement file path set to {path}')

    @command(category='Patch',
                description='emit the states to logger that are being attempted in manual mode',
                success_message='state emitted')
    def state_emitter(self,state):
        """
        Emit a patching state for logging in manual mode.

        Args:
            state (object): State identifier or data.
        """
        self.execute(self.autopatcher.state_emitter, argument=(state))
                
    @blocking_command(category='Patch', description='Move to cell and patch it',
                      task_description='Moving to cell and patching it')
    def patch(self) -> None:
        """Execute the full patching procedure on the first queued cell."""
        cell_entry = self._next_cell(mark_attempted=True)
        if cell_entry is None:
            self.warning("No cells queued for patching; skipping patch command")
            return

        stage_coords, img, stage_coords_um, img_fluo = cell_entry

        # Allocate a fresh sample index and persist metadata before protocols run
        self.recording_state_manager.increment_sample_number()
        index = self.recording_state_manager.sample_number
        voltage_hold, current_hold = self._protocol_holding_parameters()
        self.ephys_logger.save_cell_metadata(
            index,
            stage_coords_um,
            img,
            image_fluo=img_fluo,
            voltage_hold=voltage_hold,
            current_hold=current_hold,
        )

        success = self.execute(
            self.autopatcher.patch,
            argument=(stage_coords, img, stage_coords_um)
        )
        time.sleep(2)
        if success and not self.autopatcher.config.auto_clean_pipette:
            # Remove the cell from the list after patching if using the default protocol
            self.info("Patch command completed successfully, but auto escape not enabled; leaving cell in of queue for manual follow-up.")
        elif not success and self.autopatcher.config.auto_clean_pipette:
            self.error("Patch command did not complete successfully; cleaning pipette and escaping cell")
            self._remove_cell(cell_entry)
        elif not success and not self.autopatcher.config.auto_clean_pipette:
             self.error("Patch command did not complete and auto escape not enabled; leaving cell in queue for manual follow-up.")
        else:
            self.info("Patch command completed successfully; escaping cell and cleaning pipette")
            self._remove_cell(cell_entry)

    @blocking_command(
        category='Patch',
        description='Attempt gigaseal, break in, and run protocols (optional escape)',
        task_description='Attempting gigaseal, breaking in, running protocols, and optional escape'
    )
    def whole_cell(self) -> None:
        """Perform a full whole-cell patch workflow including sealing and protocols."""
        cell_entry = self._next_cell(mark_attempted=True)
        if cell_entry is None:
            self.warning("No cells queued for patching; skipping whole-cell command")
            return

        stage_coords, img, stage_coords_um, img_fluo = cell_entry

        # Allocate a fresh sample index and persist metadata before protocols run
        self.recording_state_manager.increment_sample_number()
        index = self.recording_state_manager.sample_number
        voltage_hold, current_hold = self._protocol_holding_parameters()
        self.ephys_logger.save_cell_metadata(
            index,
            stage_coords_um,
            img,
            image_fluo=img_fluo,
            voltage_hold=voltage_hold,
            current_hold=current_hold,
        )

        success = self.execute(
            self.autopatcher.whole_cell,
            argument=(stage_coords, img, stage_coords_um)
        )
        time.sleep(2)
        if success and not self.autopatcher.config.auto_clean_pipette:
            self.info("Whole-cell command completed successfully, but auto escape not enabled; leaving cell in queue for manual follow-up.")
        elif not success and self.autopatcher.config.auto_clean_pipette:
            self.error("Whole-cell command did not complete successfully; cleaning pipette and escaping cell")
            self._remove_cell(cell_entry)
        elif not success and not self.autopatcher.config.auto_clean_pipette:
            self.error("Whole-cell command did not complete and auto escape not enabled; leaving cell in queue for manual follow-up.")
        else:
            self.info("Whole-cell command completed successfully; escaping cell and cleaning pipette")
            self._remove_cell(cell_entry)

    @blocking_command(category='Patch',
                        description='Locate the cell',
                        task_description='Moving to the cell')
    def locate_cell(self):
        """Move the system to locate the selected cell."""
        cell_entry = self._next_cell(mark_attempted=True)
        if cell_entry is None:
            self.warning("No cells assigned to this pipette")
            return
        cell, img, pos, img_fluo = cell_entry
        self.recording_state_manager.increment_sample_number()
        self.execute(self.autopatcher.locate_cell,
                      argument = (cell, img, pos))
        time.sleep(2)
 
    @blocking_command(category='Stage',
                     description = 'Center the stage on cell',
                      task_description='Centering the stage on cell')
    def center_on_cell(self):
        """Center the stage on the selected cell."""
        cell_entry = self._next_cell()
        if cell_entry is None:
            self.warning("No cells assigned to this pipette")
            return
        cell, img, pos, img_fluo = cell_entry
        # print( f"patch.py: centering on cell {cell} with image {img.shape}")
        self.execute(self.autopatcher.calibrated_stage.center_on_cell,
                      argument = (cell, img, pos))

    @blocking_command(category='Stage',
                     description='Move stage to cell',
                      task_description='Moving stage to cell')

    def move_stage_to_cell(self):
        """Move the stage to the selected cell position."""
        cell_entry = self._next_cell()
        if cell_entry is None:
            self.warning("No cells queued for stage move; skipping command")
            return
        cell, img, pos, img_fluo = cell_entry
        self.execute(self.autopatcher.move_stage_to_cell, argument=cell)

    @blocking_command(category='Stage',
                     description='Move stage to cell',
                      task_description='Moving stage to cell')
    def move_stage_to_cell(self):
        cell_entry = self._next_cell()
        if cell_entry is None:
            self.warning("No cells queued for stage move; skipping command")
            return
        cell, img, pos, img_fluo = cell_entry
        self.execute(self.autopatcher.move_stage_to_cell, argument=cell)

    @blocking_command(category='Patch',
                        description='Hunt the cell',
                        task_description='Moving to the cell and detecting it ')
    def hunt_cell(self):
        """Move toward the cell and attempt detection."""
        cell_entry = self._next_cell(mark_attempted=True)
        if cell_entry is None:
            self.warning("No cells assigned to this pipette")
            return
        cell, img, pos, img_fluo = cell_entry
        self.recording_state_manager.increment_sample_number()
        self.execute(self.autopatcher.hunt_cell,
                      argument = (cell, img, pos))
        time.sleep(2)
        # self.cells_to_patch = self.cells_to_patch[1:]

    @blocking_command(category='Patch',
                        description='escape the cell',
                        task_description='Moving away from the cell, cleaning pipette and moving to home space')
    def escape_cell(self):
        """Move away from the cell and perform cleanup actions."""
        cell_entry = self._next_cell()
        self.execute(self.autopatcher.escape)
        time.sleep(2)
        # self.cells_to_patch = self.cells_to_patch[1:]
        if cell_entry is not None:
            self._remove_cell(cell_entry)

    @blocking_command(category='Patch',
                        description='find the pipette',
                        task_description='locating pipette')
    def find_pipette(self):
        """Locate the pipette in the system."""
        self.execute(self.autopatcher.find_pipette)
        time.sleep(2)


    @command(category='Patch',
             description='Store the position of the washing bath',
             success_message='Cleaning path position stored')
    def store_cleaning_position(self) -> None:
        """Store the current position as the cleaning bath location."""
        self.autopatcher.cleaning_bath_position = self.pipette_controller.calibrated_unit.position()
        self.autopatcher.calibrated_unit.config.bath_position = tuple(self.autopatcher.cleaning_bath_position)
        self.pipette_controller.cleaning_bath_position = self.autopatcher.cleaning_bath_position
        self._write_calibration_if_ready()

    def _write_calibration_if_ready(self):
        if (
            self.pipette_controller.calibrated_stage.calibrated
            and self.pipette_controller.calibrated_unit.calibrated
            and self.pipette_controller.home_position is not None
            and self.pipette_controller.safe_position is not None
            and self.pipette_controller.home_stage_position is not None
            and self.pipette_controller.safe_stage_position is not None
        ):
            self.pipette_controller.write_calibration()

    @command(category='Patch',
                description='Store the position of the safe space',
                success_message='Safe space position stored')
    def store_safe_position(self) -> None:
        """Store the current position as the safe position."""
        self.autopatcher.safe_position = self.pipette_controller.calibrated_unit.position()
        self.autopatcher.calibrated_unit.config.safe_position = tuple(self.autopatcher.safe_position)
        x,y = self.pipette_controller.calibrated_stage.position()
        z_scale = self.pipette_controller.calibrated_unit.config.microscope_units_per_um
        z = float(self.pipette_controller.calibrated_unit.microscope.position() / z_scale)
        self.autopatcher.safe_stage_position = [x,y,z]
        self.autopatcher.calibrated_stage.config.safe_position_stage = tuple(self.autopatcher.safe_stage_position)
        self.pipette_controller.safe_position = self.autopatcher.safe_position
        self.pipette_controller.safe_stage_position = self.autopatcher.safe_stage_position
        self.info(f'safe space position stored: {self.autopatcher.safe_position} and {self.autopatcher.safe_stage_position}')
        self._write_calibration_if_ready()

    @command(category='Patch',
                description='Store the position of the home space',
                success_message='Home position stored')
    def store_home_position(self) -> None:
        """Store the current position as the home position."""
        # modifying to store the safe position of the and stage as well.
        self.autopatcher.home_position = self.pipette_controller.calibrated_unit.position()
        self.autopatcher.calibrated_unit.config.home_position = tuple(self.autopatcher.home_position)
        x,y = self.pipette_controller.calibrated_stage.position()
        z_scale = self.pipette_controller.calibrated_unit.config.microscope_units_per_um
        z = float(self.pipette_controller.calibrated_unit.microscope.position() / z_scale)
        self.autopatcher.home_stage_position = [x,y,z]
        self.autopatcher.calibrated_stage.config.home_position_stage = tuple(self.autopatcher.home_stage_position)
        self.pipette_controller.home_position = self.autopatcher.home_position
        self.pipette_controller.home_stage_position = self.autopatcher.home_stage_position
        self.info(f'safe home position stored: {self.autopatcher.home_position} and {self.autopatcher.home_stage_position}')
        self._write_calibration_if_ready()


    @command(category='Patch',
                description='Store the position of the home and safe spaces',
                success_message='Home position stored')
    def store_calibration_positions(self) -> None:
        """Compute and store both home and safe positions for calibration."""
        # modifying to store the safe position of the and stage as well.
        self.autopatcher.home_position = self.pipette_controller.calibrated_unit.position()
        direction = self.autopatcher.calibrated_unit.config.resolve_pipette_axis_direction(
            self.pipette_controller.pipette_id
        )
        delta = float(self.autopatcher.calibrated_unit.config.safe_position_delta_um)
        self.autopatcher.safe_position = np.asarray(self.autopatcher.home_position, dtype=float) + delta * direction
        x,y = self.pipette_controller.calibrated_stage.position()
        z_scale = self.pipette_controller.calibrated_unit.config.microscope_units_per_um
        z = float(self.pipette_controller.calibrated_unit.microscope.position() / z_scale)
        self.autopatcher.home_stage_position = [x,y,z]
        self.autopatcher.safe_stage_position = self.autopatcher.home_stage_position
        # save all positions to the config
        self.autopatcher.calibrated_unit.config.home_position = tuple(self.autopatcher.home_position)
        self.autopatcher.calibrated_unit.config.safe_position = tuple(self.autopatcher.safe_position)
        self.autopatcher.calibrated_stage.config.home_position_stage = tuple(self.autopatcher.home_stage_position)
        self.autopatcher.calibrated_stage.config.safe_position_stage = tuple(self.autopatcher.safe_stage_position)
        self.info(f'safe home position stored: {self.autopatcher.home_position} and {self.autopatcher.home_stage_position}')
        self.info(f'safe space position stored: {self.autopatcher.safe_position} and {self.autopatcher.safe_stage_position}')
        # send all positions to the pipette controller
        self.pipette_controller.home_position = self.autopatcher.home_position
        self.pipette_controller.safe_position = self.autopatcher.safe_position
        self.pipette_controller.home_stage_position = self.autopatcher.home_stage_position
        self.pipette_controller.safe_stage_position = self.autopatcher.safe_stage_position
        self.pipette_controller.cleaning_bath_position = self.autopatcher.cleaning_bath_position
        self.pipette_controller.write_calibration()
    

    @command(category='Patch',
             description='read the calibratoin values from file',
                success_message='Calibration values read from file')
    def read_calibration(self) -> None:
        """Load calibration positions from file and update internal state."""
        self.pipette_controller.read_calibration()
        self.autopatcher.home_position = self.pipette_controller.home_position
        self.autopatcher.safe_position = self.pipette_controller.safe_position
        self.autopatcher.home_stage_position = self.pipette_controller.home_stage_position
        self.autopatcher.safe_stage_position = self.pipette_controller.safe_stage_position
        self.autopatcher.cleaning_bath_position = self.pipette_controller.cleaning_bath_position
        self.autopatcher.rinsing_bath_position = self.pipette_controller.rinsing_bath_position
        self.info(f'Calibration values read from file: {self.autopatcher.home_position}, {self.autopatcher.safe_position}, {self.autopatcher.home_stage_position}, {self.autopatcher.safe_stage_position}, {self.autopatcher.cleaning_bath_position}, {self.autopatcher.rinsing_bath_position}')


    @command(category='Patch',
             description='Store the position of the rinsing bath',
             success_message='Rinsing bath position stored')
    def store_rinsing_position(self) -> None:
        """Store the current position as the rinsing bath location."""
        self.autopatcher.rinsing_bath_position = self.pipette_controller.calibrated_unit.position()

    @command(category='Patch',
             description='clear all stored positions',
                success_message='All positions cleared')
    def clear_positions(self) -> None:
        """Clear all stored calibration and position data."""
        self.autopatcher.cleaning_bath_position = None
        self.autopatcher.rinsing_bath_position = None
        self.autopatcher.home_position = None
        self.autopatcher.safe_position = None
        self.autopatcher.home_stage_position = None
        self.autopatcher.safe_stage_position = None
        # set tuple to None in the config
        self.autopatcher.calibrated_unit.config.home_position = None
        self.autopatcher.calibrated_unit.config.safe_position = None
        self.autopatcher.calibrated_stage.config.home_position_stage = None
        self.autopatcher.calibrated_stage.config.safe_position_stage = None
        self.autopatcher.calibrated_unit.config.cleaning_bath_position = None
        # self.autopatcher.calibrated_unit.config.rinsing_bath_position = None    
        self.info('All positions cleared')


    @blocking_command(category='Patch',
                      description='Clean the pipette (wash and rinse)',
                      task_description='Cleaning the pipette')
    def clean_pipette(self):
        """Perform automated pipette cleaning."""
        self.execute(self.autopatcher.clean_pipette)

    @blocking_command(category='Patch',
                      description= 'Move to safe space',
                      task_description='Moving to safe space')
    def move_to_safe_space(self):
        """Move the system to the predefined safe position."""
        self.execute(self.autopatcher.move_to_safe_space)

    @blocking_command(category='Patch',
                        description='Move to home position',
                        task_description='Moving to home position')
    def move_to_home_space(self):
        """Move the system to the predefined home position."""
        self.execute(self.autopatcher.move_to_home_space)

    @blocking_command(category='Patch',
                        description='Move the group down',
                        task_description='Moving the group down')
    def move_group_down(self):
        """Move the manipulator group downward."""
        self.execute(self.autopatcher.move_group_down)

    @blocking_command(category='Patch',
                        description='Move the group up',
                        task_description='Moving the group up')
    def move_group_up(self):
        self.execute(self.autopatcher.move_group_up)

    @blocking_command(category='Patch',
                      description='Move the group in x direction',
                      task_description='Moving the group in x direction')
    def move_group_in_x(self):
        """Move the manipulator group upward."""
        self.execute(self.autopatcher.move_group_in_x)

    @blocking_command(category='Patch',
                      description='Move the group in y direction',
                        task_description='Moving the group in y direction')
    def move_group_in_y(self):
        """Move the manipulator group along the y-axis."""
        self.execute(self.autopatcher.move_group_in_y)

    @blocking_command(category='Patch',
                      description='Move the pipette up',
                      task_description='Moving the pipette up')
    def move_pipette_up(self):
        """Move the pipette upward."""
        self.execute(self.autopatcher.move_pipette_up)

    @blocking_command(category='Patch',
                      description='Sequential patching and cleaning for multiple cells',
                      task_description='Sequential patch clamping')
    def sequential_patching(self):
        """Perform sequential patching and cleaning for multiple cells."""
        self.execute(self.autopatcher.sequential_patching)

    @blocking_command(category='Patch',
                      description='Moving down the calibrated manipulator to detect the contact point with the coverslip',
                      task_description='Contact detection')
    def contact_detection(self):
        """Detect the contact point between the pipette and coverslip."""
        # TODO: Figure out what this should link to
        self.execute(self.autopatcher.contact_detection)

    def set_pressure_near(self) -> None:
        '''Puts the pipette under positive pressure to prevent blockages'''
        self.pressure.set_pressure(self.config.pressure_near)

    @blocking_command(category='Lamp', 
             description='Open or close the shutter',
             task_description='Toggling the shutter')
    def toggle_shutter(self):
        """Toggle the lamp shutter state."""
        self.execute(self.autopatcher.toggle_shutter)

    @blocking_command(category='Lamp', description='Toggle between fluorescence and filterless',
             task_description='Toggling fluorescence')
    def toggle_fluorescence(self):
        """Toggle fluorescence mode on the lamp."""
        self.execute(self.autopatcher.toggle_fluorescence)
    @blocking_command(category='Lamp', 
             description='Move filter cube one slot left',
             task_description='Moving filter cube left')
    def move_cube_left(self):
        """Move the filter cube one position to the left."""
        self.execute(self.autopatcher.move_cube_left)

    @blocking_command(category='Lamp', description='Move filter cube one slot right',
             task_description='Moving filter cube right')
    def move_cube_right(self):
       """Move the filter cube one position to the right."""
       self.execute(self.autopatcher.move_cube_right)

    @blocking_command(category='Laser',
                      description='Toggle laser output',
                      task_description='Toggling laser output')
    def toggle_laser_output(self):
        """Toggle the laser output state."""
        self.execute(self.autopatcher.toggle_laser_output)

    @blocking_command(category='Laser',
                      description='Step wavelength down',
                      task_description='Stepping wavelength down')
    def wavelength_down(self):
        """Decrease the laser wavelength."""
        self.execute(self.autopatcher.wavelength_down)

    @blocking_command(category='Laser',
                      description='Step wavelength up',
                      task_description='Stepping wavelength up')
    def wavelength_up(self):
        """Increase the laser wavelength."""
        self.execute(self.autopatcher.wavelength_up)

    def info(self, msg, *args, **kwargs):
            """Intercept info logs to update the GUI status."""
            self._last_status_msg = str(msg)
            self._last_error_msg = None
            self.last_log_time = time.time()

            super().info(msg, *args, **kwargs)
        
    def error(self, msg, *args, **kwargs):
        """Intercept error logs to update the GUI status."""
        self._last_error_msg = str(msg)
        self._last_log_time = time.time()
        
        super().error(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """Intercept warning logs to update the GUI status."""
        self._last_warning_msg = str(msg)
        self.last_log_time = time.time()

        super().warning(msg, *args, **kwargs)

    @property
    def last_status_msg(self):
        """Returns the newest status message between Interface and Controller."""
        # Get the controller's timestamp safely (defaults to 0 if it doesn't exist)
        ctrl_time = getattr(self.autopatcher, 'last_log_time', 0)
        
        # If the controller logged more recently, return its message
        if ctrl_time > self.last_log_time:
            return getattr(self.autopatcher, '_last_status_msg', "")
            
        # Otherwise, return the interface's message
        return self._last_status_msg

    @property
    def last_error_msg(self):
        """Returns the newest error message between Interface and Controller."""
        ctrl_time = getattr(self.autopatcher, 'last_log_time', 0)
        
        if ctrl_time > self.last_log_time:
            return getattr(self.autopatcher, '_last_error_msg', None)
            
        return self._last_error_msg

    @property
    def last_warning_msg(self):
        """Returns the newest warning message between Interface and Controller."""
        ctrl_time = getattr(self.autopatcher, 'last_log_time', 0)
        
        if ctrl_time > self.last_log_time:
            return getattr(self.autopatcher, '_last_warning_msg', None)
            
        return self._last_error_msg

    @property
    def latest_log_time(self):
        """Returns the newest log time between Interface and Controller."""
        ctrl_time = getattr(self.autopatcher, 'last_log_time', 0)
        
        if ctrl_time > self.last_log_time:
            return getattr(self.autopatcher, 'last_log_time', None)
            
        return self.last_log_time