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
from patcherbot.devices.amplifier.DAQ import NiDAQ
from patcherbot.devices.lamp import Lamp
from patcherbot.configs.PatchConfig import PatchConfig
from patcherbot.configs.ExperimentBookConfig import ExperimentBookConfig
from patcherbot.configs.ProtocolConfig import ProtocolConfig
from PyQt5 import QtCore
import time

__all__ = ['AutoPatchInterface']

class AutoPatchInterface(TaskInterface):
    '''
    A class to run automatic patch-clamp
    '''
    state_press_tally_changed = QtCore.pyqtSignal(object)
    origin_saved = QtCore.pyqtSignal(object, object)

    def __init__(
        self,
        amplifier: Amplifier,
        daq: NiDAQ,
        pressure: PressureController,
        pipette_interface: PipetteInterface,
        recording_state_manager: RecordingStateManager,
        lamp: Lamp,
        laser=None,
        config_data=None,
        protocol_data=None,
    ):
        super().__init__()
        self.config = PatchConfig(name='Patch')
        if config_data:
            cleaned = {k: v for k, v in config_data.items() if v is not None}
            self.config.from_dict(cleaned)
        self.protocol_config = ProtocolConfig(name='Protocols')
        if protocol_data:
            cleaned = {k: v for k, v in protocol_data.items() if v is not None}
            self.protocol_config.from_dict(cleaned)
        self.experiment_book_config = ExperimentBookConfig(name='Experiment Book')
        self.amplifier = amplifier
        self.daq = daq
        self.pressure = pressure
        self.pipette_controller = pipette_interface
        self.recording_state_manager = recording_state_manager
        self.lamp = lamp
        self.laser = laser
        self.ephys_logger = EPhysLogger(recording_state_manager=self.recording_state_manager, ephys_filename="CellMetadata")
        autopatcher = AutoPatcher(amplifier, daq, pressure, self.pipette_controller.calibrated_unit,
                                    self.pipette_controller.calibrated_unit.microscope,
                                    calibrated_stage=self.pipette_controller.calibrated_stage,
                                    lamp=self.lamp,
                                    laser=self.laser,
                                    config=self.config,
                                    protocol_config=self.protocol_config)
        self.current_autopatcher = autopatcher

        self.is_selecting_cells = False
        self.is_selecting_corners = False
        self.cells_to_patch = []
        self.origins = {"x": None, "y": None}
        self.movement_file_path = ''


    @blocking_command(category='Calibration', description='Save X origin',
                      task_description='Saving X origin and microscope image')
    def save_x_origin(self, request):
        self._save_origin("x", request)

    @blocking_command(category='Calibration', description='Save Y origin',
                      task_description='Saving Y origin and microscope image')
    def save_y_origin(self, request):
        self._save_origin("y", request)

    def _save_origin(self, axis, request):
        if (self._current_controller is not None
                or self.pipette_controller._current_controller is not None):
            raise ValueError("Wait for the current movement/task to finish.")
        capture = dict(request)
        capture.update(
            axis=axis, origins_um=dict(self.origins),
            movement_busy=lambda: self.pipette_controller._current_controller is not None,
            completed=self.origin_saved.emit)
        self.execute(self.current_autopatcher.save_origin, argument=capture)

    def accept_origin(self, record):
        """Commit a successful save after the GUI confirms its experiment."""
        self.origins = dict(record["origins_um"])

    def _record_state_press(self, state):
        counts = self.recording_state_manager.increment_state_press(state)
        self.state_press_tally_changed.emit(counts)
        return counts

    def _protocol_holding_parameters(self):
        protocol_config = self.current_autopatcher.protocol_config
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
        if target_slot is None:
            return
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            current_slot = self.current_autopatcher.lamp.get_filter()
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

    
    @blocking_command(category='Patch', description='Break into the cell',
                      task_description='Breaking into the cell')
    
    def break_in(self):
        self._record_state_press("break_in")
        self.recording_state_manager.increment_sample_number()
        self.execute(self.current_autopatcher.break_in)

    @blocking_command(category='Patch', description='GigaSeal the cell',
                      task_description='GigaSealing the cell')
    def gigaseal(self):
        self._record_state_press("gigaseal")
        self.recording_state_manager.increment_sample_number()
        self.execute(self.current_autopatcher.gigaseal)

    def start_selecting_cells(self):
        self.is_selecting_cells = True

    @blocking_command(category='Patch',
                      description='Detect cells',
                      task_description='Detecting cells')
    def detect_cells(self):
        self.execute(self.pipette_controller.calibrated_stage.detect_cells)

    def start_selecting_corners(self):
        self.is_selecting_corners = True
        self.pipette_controller.calibrated_stage.start_selecting_scan_corners()

    @blocking_command(category='Stage',
                      description='Scan the stored corner coordinates',
                      task_description='Scanning plate area')
    def start_scan(self):
        self.execute(self.current_autopatcher.scan_area)

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
        if self.cells_to_patch:
            last_cell = self.cells_to_patch[-1]
            self.info(
                f"removing last cell {last_cell}... {self.is_selecting_cells}, at {self.cells_to_patch}"
            )
            self.cells_to_patch = self.cells_to_patch[:-1]
        else:
            self.info("remove_last_cell called but there are no cells to remove")


    def _remove_first_cell(self) -> None:
        """Remove the first queued cell after escape or automatic cleanup."""
        if self.cells_to_patch:
            first_cell = self.cells_to_patch[0]
            self.info(f"Removing first queued cell {first_cell}")
            self.cells_to_patch = self.cells_to_patch[1:]
        else:
            self.info("_remove_first_cell called but there are no cells to remove")

    @blocking_command(category='Cell Sorter',
            description='Move the cell sorter to a cell',
            task_description='Move the cell sorter to a cell')
    def move_cellsorter_to_cell(self):
        #grab cell from list
        cellx = self.cells_to_patch[0][0][0]
        celly = self.cells_to_patch[0][0][1]
        cellz = self.pipette_controller.calibrated_unit.microscope.floor_Z

        #move cell sorter to cell
        self.execute(self.pipette_controller.calibrated_cellsorter.center_cellsorter_on_point, argument=[cellx, celly, cellz])

    @command(category='Cell Sorter',
            description='Turn cell sorter LED on')
    def cell_sorter_led_on(self):
        self.pipette_controller.calibrated_cellsorter.set_led_status(True, ring=1)

    @command(category='Cell Sorter',
            description='Turn cell sorter LED off')
    def cell_sorter_led_off(self):
        self.pipette_controller.calibrated_cellsorter.set_led_status(False, ring=1)

    @blocking_command(category='DAQ',
            description='Run Protocols on the Cell',
            task_description='Run Protocols on the Cell')
    def run_protocols(self):
        self.recording_state_manager.increment_sample_number()
        index = self.recording_state_manager.sample_number
        if self.cells_to_patch:
            stage_coords, img, stage_coords_um, img_fluo = self.cells_to_patch[0]
            voltage_hold, current_hold = self._protocol_holding_parameters()
            self.ephys_logger.save_cell_metadata(
                index,
                stage_coords_um,
                img,
                image_fluo=img_fluo,
                voltage_hold=voltage_hold,
                current_hold=current_hold,
            )
        self.execute(self.current_autopatcher.run_protocols)

    @blocking_command(category='DAQ',
            description='Run Optogenetic Protocol',
            task_description='Running Optogenetic Protocol')
    def run_optogenetic_protocol(self, protocol_params=None):
        self.recording_state_manager.increment_sample_number()
        self.execute(self.current_autopatcher.run_optogenetic_protocol, argument=protocol_params)
    

    @command(category='Patch', description='Add a mouse position to the list of cells to patch')
    def add_cell(self, position):
        #add half the size of the camera image to the position to get the center of the cell
        position = np.array(position)

        position[0] += self.current_autopatcher.calibrated_unit.camera.width / 2
        position[1] += self.current_autopatcher.calibrated_unit.camera.height / 2
        # add the z position of the microscope
        z_scale = self.current_autopatcher.calibrated_unit.config.microscope_units_per_um
        z_pos = self.current_autopatcher.calibrated_unit.microscope.position() / z_scale
        self.info(f'z position of the microscope: {z_pos}')
        print(f'adding cell... {self.is_selecting_cells}')
        if self.is_selecting_cells:
            print('Adding cell at', position, 'to list of cells to patch in pixels')
            stage_pos_pixels = self.current_autopatcher.calibrated_stage.reference_position()
            stage_pos_pixels[0:2] -= position
            # display stage position
            # add the z_pos to the stage position as a third dimension in the np array

            stage_pos_pixels = np.array([stage_pos_pixels[0], stage_pos_pixels[1], z_pos])
            # get raw currnent stage position in pixels
            stage_pos_plane = self.current_autopatcher.calibrated_stage.position()
            stage_pos_um = np.array([stage_pos_plane[0], stage_pos_plane[1], z_pos])
            print(f'Stage position in pixels: {stage_pos_pixels}')
            print(f'Stage position in um: {stage_pos_um}')



            # print(f'Stage position dimensions: {np.size(stage_pos_pixels)}')
            # print(f'Stage um position: {stage_pos_pixels}')
            #take a 512x512 image centered on the cell
            img = self.current_autopatcher.calibrated_unit.camera.get_16bit_image()
            img = img[int(position[1]-256):int(position[1]+256), int(position[0]-256):int(position[0]+256)]
            if img is None or img.shape != (512, 512):
                raise RuntimeError('Cell too Close to edge!')
            
            img_fluo = None
            if self.config.auto_capture_fluo:
                fluo_slot = int(self.config.lamp)
                current_slot = self.current_autopatcher.lamp.get_filter()
                did_toggle = current_slot != fluo_slot
                if did_toggle:
                    self.current_autopatcher.toggle_fluorescence()
                    self._wait_for_filter_slot(fluo_slot)
                try:
                    img_fluo = self.current_autopatcher.calibrated_unit.camera.get_16bit_image()
                    img_fluo = img_fluo[int(position[1]-256):int(position[1]+256), int(position[0]-256):int(position[0]+256)]
                    if img_fluo is None or img_fluo.shape != (512, 512):
                        raise RuntimeError('Cell too Close to edge!')
                finally:
                    if did_toggle:
                        self.current_autopatcher.toggle_fluorescence()
                        if current_slot is not None:
                            self._wait_for_filter_slot(current_slot)

            self.cells_to_patch.append((np.array(stage_pos_pixels), img, stage_pos_um, img_fluo))
            self.is_selecting_cells = False

    def update_camera_cell_list(self) -> None:
        """Publish calibrated display coordinates from the devices' cached positions."""
        controller = self.current_autopatcher
        unit, stage = controller.calibrated_unit, controller.calibrated_stage
        camera = unit.camera
        self.pipette_controller.display_positions = None
        camera.cell_list = []
        if not stage.calibrated or getattr(stage, "must_be_recalibrated", False):
            return
        try:
            stage_pixels = np.asarray(stage.reference_position(), dtype=float)
            if not np.isfinite(stage_pixels).all():
                return
            camera.cell_list = [
                ((stage_pixels - cell)[:2].astype(int), image, position)
                for cell, image, position, _ in tuple(self.cells_to_patch)]
            if not unit.calibrated or getattr(unit, "must_be_recalibrated", False):
                return
            home = np.asarray(controller.home_position, dtype=float)
            stage_home = np.asarray(controller.home_stage_position, dtype=float)
            if (home.shape != (3,) or stage_home.shape != (3,)
                    or not np.isfinite(home).all() or not np.isfinite(stage_home).all()):
                return
            # HOME aligns the tip with image center; only displacement moves the arrow.
            tip = (np.array([camera.width / 2., camera.height / 2.])
                   + unit.um_to_pixels_relative(unit.position() - home)[:2]
                   + stage.um_to_pixels_relative(stage.position() - stage_home[:2])[:2])
            if np.isfinite(tip).all():
                self.pipette_controller.display_positions = dict(
                    pipette_xy=tip.copy(), at=time.monotonic(),
                    pipette_direction_xy=np.asarray(unit.M, dtype=float)[:2, 0].copy(),
                    image_shape=(camera.height, camera.width))
        except (RuntimeError, ValueError, TypeError, OSError):
            return

    @command(category='Test', 
             description='Send movement file to the autopatcher', 
             success_message='Path sent')
    def send_movement_file(self, path):
        self.current_autopatcher.test_movement(path)
            # Append a tuple of (coordinates, image)
        self.info(f'Movement file path set to {path}')

    @command(category='Patch',
                description='emit the states to logger that are being attempted in manual mode',
                success_message='state emitted')
    def state_emitter(self,state):
        self.execute(self.current_autopatcher.state_emitter, argument=(state))
                
    @blocking_command(category='Patch', description='Move to cell and patch it',
                      task_description='Moving to cell and patching it')
    def patch(self) -> None:
        self._record_state_press("patch")
        if not self.cells_to_patch:
            self.warning("No cells queued for patching; skipping patch command")
            return

        stage_coords, img, stage_coords_um, img_fluo = self.cells_to_patch[0]

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
            self.current_autopatcher.patch,
            argument=(stage_coords, img, stage_coords_um)
        )
        time.sleep(2)
        if success and not self.current_autopatcher.config.auto_clean_pipette:
            # Remove the cell from the list after patching if using the default protocol
            self.info("Patch command completed successfully, but auto escape not enabled; leaving cell in of queue for manual follow-up.")
        elif not success and self.current_autopatcher.config.auto_clean_pipette:
            self.error("Patch command did not complete successfully; cleaning pipette and escaping cell")
            self._remove_first_cell()
        elif not success and not self.current_autopatcher.config.auto_clean_pipette:
             self.error("Patch command did not complete and auto escape not enabled; leaving cell in queue for manual follow-up.")
        else:
            self.info("Patch command completed successfully; escaping cell and cleaning pipette")
            self._remove_first_cell()

    @blocking_command(
        category='Patch',
        description='Attempt gigaseal, break in, and run protocols (optional escape)',
        task_description='Attempting gigaseal, breaking in, running protocols, and optional escape'
    )
    def whole_cell(self) -> None:
        if not self.cells_to_patch:
            self.warning("No cells queued for patching; skipping whole-cell command")
            return

        stage_coords, img, stage_coords_um, img_fluo = self.cells_to_patch[0]

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
            self.current_autopatcher.whole_cell,
            argument=(stage_coords, img, stage_coords_um)
        )
        time.sleep(2)
        if success and not self.current_autopatcher.config.auto_clean_pipette:
            self.info("Whole-cell command completed successfully, but auto escape not enabled; leaving cell in queue for manual follow-up.")
        elif not success and self.current_autopatcher.config.auto_clean_pipette:
            self.error("Whole-cell command did not complete successfully; cleaning pipette and escaping cell")
            self._remove_first_cell()
        elif not success and not self.current_autopatcher.config.auto_clean_pipette:
            self.error("Whole-cell command did not complete and auto escape not enabled; leaving cell in queue for manual follow-up.")
        else:
            self.info("Whole-cell command completed successfully; escaping cell and cleaning pipette")
            self._remove_first_cell()

    @blocking_command(category='Patch',
                        description='Locate the cell',
                        task_description='Moving to the cell')
    def locate_cell(self):
        self._record_state_press("locate_cell")
        cell, img, pos, img_fluo = self.cells_to_patch[0]
        self.recording_state_manager.increment_sample_number()
        self.execute(self.current_autopatcher.locate_cell,
                      argument = (cell, img,pos))
        time.sleep(2)
 
    @blocking_command(category='Patch',
                      description='Approach the cell',
                      task_description='Approaching the cell')
    def approach_cell(self):
        self._record_state_press("approach_cell")
        cell, img, pos, img_fluo = self.cells_to_patch[0]
        self.recording_state_manager.increment_sample_number()
        self.execute(self.current_autopatcher.approach_cell,
                     argument=(cell, img, pos))
        time.sleep(2)

    @blocking_command(category='Stage',
                     description = 'Center the stage on cell',
                      task_description='Centering the stage on cell')
    def center_on_cell(self):
        cell, img, pos, img_fluo = self.cells_to_patch[0]
        # print( f"patch.py: centering on cell {cell} with image {img.shape}")
        self.execute(self.current_autopatcher.calibrated_stage.center_on_cell,
                      argument = (cell, img,pos))

    @blocking_command(category='Stage',
                     description='Move stage to cell',
                      task_description='Moving stage to cell')
    def move_stage_to_cell(self):
        if not self.cells_to_patch:
            self.warning("No cells queued for stage move; skipping command")
            return
        cell, img, pos, img_fluo = self.cells_to_patch[0]
        self.execute(self.current_autopatcher.move_stage_to_cell, argument=cell)

    @blocking_command(category='Patch',
                        description='Hunt the cell',
                        task_description='Moving to the cell and detecting it ')
    def hunt_cell(self):
        self._record_state_press("hunt_cell")
        cell, img, pos, img_fluo = self.cells_to_patch[0]
        self.recording_state_manager.increment_sample_number()
        self.execute(self.current_autopatcher.hunt_cell,
                      argument = (cell, img,pos))
        time.sleep(2)
        # self.cells_to_patch = self.cells_to_patch[1:]

    @blocking_command(category='Patch',
                        description='escape the cell',
                        task_description='Moving away from the cell, cleaning pipette and moving to home space')
    def escape_cell(self):
        self._record_state_press("escape")
        self.execute(self.current_autopatcher.escape)
        time.sleep(2)
        self._remove_first_cell()

    @blocking_command(category='Patch',
                        description='find the pipette',
                        task_description='locating pipette')
    def find_pipette(self):
        self.execute(self.current_autopatcher.find_pipette)
        time.sleep(2)


    @command(category='Patch',
             description='Store the position of the washing bath',
             success_message='Cleaning path position stored')
    def store_cleaning_position(self) -> None:
        self.current_autopatcher.cleaning_bath_position = self.pipette_controller.calibrated_unit.position()
        self.current_autopatcher.calibrated_unit.config.bath_position = tuple(self.current_autopatcher.cleaning_bath_position)
        self.pipette_controller.cleaning_bath_position = self.current_autopatcher.cleaning_bath_position.copy()
        # save calibration to file
        self.pipette_controller.write_calibration()

    @command(category='Patch',
                description='Store the position of the safe space',
                success_message='Safe space position stored')
    def store_safe_position(self) -> None:
        self.current_autopatcher.safe_position = self.pipette_controller.calibrated_unit.position()
        self.current_autopatcher.calibrated_unit.config.safe_position = tuple(self.current_autopatcher.safe_position)
        x,y = self.pipette_controller.calibrated_stage.position()
        z_scale = self.pipette_controller.calibrated_unit.config.microscope_units_per_um
        z = float(self.pipette_controller.calibrated_unit.microscope.position() / z_scale)
        self.current_autopatcher.safe_stage_position = [x,y,z]
        self.current_autopatcher.calibrated_stage.config.safe_position_stage = tuple(self.current_autopatcher.safe_stage_position)
        self.info(f'safe space position stored: {self.current_autopatcher.safe_position} and {self.current_autopatcher.safe_stage_position}')

    @command(category='Patch',
                description='Store the position of the home space',
                success_message='Home position stored')
    def store_home_position(self) -> None:
        # modifying to store the safe position of the and stage as well.
        self.current_autopatcher.home_position = self.pipette_controller.calibrated_unit.position()
        self.current_autopatcher.calibrated_unit.config.home_position = tuple(self.current_autopatcher.home_position)
        x,y = self.pipette_controller.calibrated_stage.position()
        z_scale = self.pipette_controller.calibrated_unit.config.microscope_units_per_um
        z = float(self.pipette_controller.calibrated_unit.microscope.position() / z_scale)
        self.current_autopatcher.home_stage_position = [x,y,z]
        self.current_autopatcher.calibrated_stage.config.home_position_stage = tuple(self.current_autopatcher.home_stage_position)
        self.info(f'safe home position stored: {self.current_autopatcher.home_position} and {self.current_autopatcher.home_stage_position}')


    @command(category='Patch',
                description='Store the position of the home and safe spaces',
                success_message='Home position stored')
    def store_calibration_positions(self) -> None:
        # modifying to store the safe position of the and stage as well.
        self.current_autopatcher.home_position = self.pipette_controller.calibrated_unit.position()
        angle = np.deg2rad(self.current_autopatcher.calibrated_unit.config.pipette_y_rotation)
        delta = float(self.current_autopatcher.calibrated_unit.config.safe_position_delta_um)
        x_pip, y_pip,z_pip  = self.current_autopatcher.home_position
        self.current_autopatcher.safe_position = np.array(
            [x_pip + delta * np.cos(angle), y_pip, z_pip + delta * np.sin(angle)]
        )
        x,y = self.pipette_controller.calibrated_stage.position()
        z_scale = self.pipette_controller.calibrated_unit.config.microscope_units_per_um
        z = float(self.pipette_controller.calibrated_unit.microscope.position() / z_scale)
        self.current_autopatcher.home_stage_position = [x,y,z]
        self.current_autopatcher.safe_stage_position = self.current_autopatcher.home_stage_position
        # save all positions to the config
        self.current_autopatcher.calibrated_unit.config.home_position = tuple(self.current_autopatcher.home_position)
        self.current_autopatcher.calibrated_unit.config.safe_position = tuple(self.current_autopatcher.safe_position)
        self.current_autopatcher.calibrated_stage.config.home_position_stage = tuple(self.current_autopatcher.home_stage_position)
        self.current_autopatcher.calibrated_stage.config.safe_position_stage = tuple(self.current_autopatcher.safe_stage_position)
        self.info(f'safe home position stored: {self.current_autopatcher.home_position} and {self.current_autopatcher.home_stage_position}')
        self.info(f'safe space position stored: {self.current_autopatcher.safe_position} and {self.current_autopatcher.safe_stage_position}')
        # send all positions to the pipette controller
        self.pipette_controller.home_position = self.current_autopatcher.home_position
        self.pipette_controller.safe_position = self.current_autopatcher.safe_position
        self.pipette_controller.home_stage_position = self.current_autopatcher.home_stage_position
        self.pipette_controller.safe_stage_position = self.current_autopatcher.safe_stage_position
        self.pipette_controller.cleaning_bath_position = self.current_autopatcher.cleaning_bath_position
        self.pipette_controller.write_calibration()
    

    @command(category='Patch',
             description='read the calibratoin values from file',
                success_message='Calibration values read from file')
    def read_calibration(self) -> None:
        self.pipette_controller.read_calibration()
        self.current_autopatcher.home_position = self.pipette_controller.home_position
        self.current_autopatcher.safe_position = self.pipette_controller.safe_position
        self.current_autopatcher.home_stage_position = self.pipette_controller.home_stage_position
        self.current_autopatcher.safe_stage_position = self.pipette_controller.safe_stage_position
        self.current_autopatcher.cleaning_bath_position = self.pipette_controller.cleaning_bath_position
        self.current_autopatcher.rinsing_bath_position = self.pipette_controller.rinsing_bath_position
        self.info(f'Calibration values read from file: {self.current_autopatcher.home_position}, {self.current_autopatcher.safe_position}, {self.current_autopatcher.home_stage_position}, {self.current_autopatcher.safe_stage_position}, {self.current_autopatcher.cleaning_bath_position}, {self.current_autopatcher.rinsing_bath_position}')


    @command(category='Patch',
             description='Store the position of the rinsing bath',
             success_message='Rinsing bath position stored')
    def store_rinsing_position(self) -> None:
        self.current_autopatcher.rinsing_bath_position = self.pipette_controller.calibrated_unit.position()

    @command(category='Patch',
             description='clear all stored positions',
                success_message='All positions cleared')
    def clear_positions(self) -> None:
        self.current_autopatcher.cleaning_bath_position = None
        self.current_autopatcher.rinsing_bath_position = None
        self.current_autopatcher.home_position = None
        self.current_autopatcher.safe_position = None
        self.current_autopatcher.home_stage_position = None
        self.current_autopatcher.safe_stage_position = None
        # set tuple to None in the config
        self.current_autopatcher.calibrated_unit.config.home_position = (0., 0., 0.)
        self.current_autopatcher.calibrated_unit.config.safe_position = (0., 0., 0.)
        self.current_autopatcher.calibrated_stage.config.home_position_stage = (0., 0., 0.)
        self.current_autopatcher.calibrated_stage.config.safe_position_stage = (0., 0., 0.)
        self.current_autopatcher.calibrated_unit.config.bath_position = (0., 0., 0.)
        # self.current_autopatcher.calibrated_unit.config.rinsing_bath_position = None    
        self.info('All positions cleared')


    @blocking_command(category='Patch',
                      description='Clean the pipette (wash and rinse)',
                      task_description='Cleaning the pipette')
    def clean_pipette(self):
        self.execute(self.current_autopatcher.clean_pipette)

    @blocking_command(category='Patch',
                      description= 'Move to safe space',
                      task_description='Moving to safe space')
    def move_to_safe_space(self):
        self.execute(self.current_autopatcher.move_to_safe_space)

    @blocking_command(category='Patch',
                        description='Move to home position',
                        task_description='Moving to home position')
    def move_to_home_space(self):
        self.execute(self.current_autopatcher.move_to_home_space)

    @blocking_command(category='Patch',
                        description='Move the group down',
                        task_description='Moving the group down')
    def move_group_down(self):
        self.execute(self.current_autopatcher.move_group_down)

    @blocking_command(category='Patch',
                        description='Move the group up',
                        task_description='Moving the group up')
    def move_group_up(self):
        self.execute(self.current_autopatcher.move_group_up)

    @blocking_command(category='Patch',
                      description='Move the group in x direction',
                      task_description='Moving the group in x direction')
    def move_group_in_x(self):
        self.execute(self.current_autopatcher.move_group_in_x)

    @blocking_command(category='Patch',
                      description='Move the group in y direction',
                        task_description='Moving the group in y direction')
    def move_group_in_y(self):
        self.execute(self.current_autopatcher.move_group_in_y)

    @blocking_command(category='Patch',
                      description='Move the pipette up',
                      task_description='Moving the pipette up')
    def move_pipette_up(self):
        self.execute(self.current_autopatcher.move_pipette_up)

    @blocking_command(category='Patch',
                      description='Sequential patching and cleaning for multiple cells',
                      task_description='Sequential patch clamping')
    def sequential_patching(self):
        self.execute(self.current_autopatcher.sequential_patching)

    @blocking_command(category='Patch',
                      description='Moving down the calibrated manipulator to detect the contact point with the coverslip',
                      task_description='Contact detection')
    def contact_detection(self):
        # TODO: Figure out what this should link to
        self.execute(self.current_autopatcher.contact_detection)

    def set_pressure_near(self) -> None:
        '''puts the pipette under positive pressure to prevent blockages
        '''
        self.pressure.set_pressure(self.config.pressure_near)

    @blocking_command(category='Lamp', 
             description='Open or close the shutter',
             task_description='Toggling the shutter')
    def toggle_shutter(self):
        self.execute(self.current_autopatcher.toggle_shutter)

    @blocking_command(category='Lamp', description='Toggle between fluorescence and filterless',
             task_description='Toggling fluorescence')
    def toggle_fluorescence(self):
        self.execute(self.current_autopatcher.toggle_fluorescence)
    @blocking_command(category='Lamp', 
             description='Move filter cube one slot left',
             task_description='Moving filter cube left')
    def move_cube_left(self):
        self.execute(self.current_autopatcher.move_cube_left)

    @blocking_command(category='Lamp', description='Move filter cube one slot right',
             task_description='Moving filter cube right')
    def move_cube_right(self):
       self.execute(self.current_autopatcher.move_cube_right)

    @blocking_command(category='Laser',
                      description='Toggle laser output',
                      task_description='Toggling laser output')
    def toggle_laser_output(self):
        self.execute(self.current_autopatcher.toggle_laser_output)

    @blocking_command(category='Laser',
                      description='Step wavelength down',
                      task_description='Stepping wavelength down')
    def wavelength_down(self):
        self.execute(self.current_autopatcher.wavelength_down)

    @blocking_command(category='Laser',
                      description='Step wavelength up',
                      task_description='Stepping wavelength up')
    def wavelength_up(self):
        self.execute(self.current_autopatcher.wavelength_up)
