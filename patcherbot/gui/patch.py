from __future__ import absolute_import

from types import MethodType

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal, QObject
import PyQt5.QtGui as QtGui
import numpy as np
import logging
import time

from PyQt5.QtWidgets import QFileDialog, QWidget,QMessageBox

from patcherbot.controller import TaskController
from patcherbot.gui.manipulator import ManipulatorGui
from patcherbot.interface.patch import AutoPatchInterface
from patcherbot.interface.pipettes import PipetteInterface
from patcherbot.utils.RecordingStateManager import RecordingStateManager
from patcherbot.interface.base import command

from patcherbot.utils.FileLogger import FileLogger
from datetime import datetime
import json
import pickle
import os

class PatchGui(ManipulatorGui):

    patch_command_signal = QtCore.pyqtSignal(MethodType, object)
    patch_reset_signal = QtCore.pyqtSignal(TaskController)

    def __init__(self, camera, aux_camera, pipette_interface: PipetteInterface, patch_interface: AutoPatchInterface, recording_state_manager: RecordingStateManager, with_tracking=False):
        super(PatchGui, self).__init__(camera, aux_camera, pipette_interface, with_tracking=with_tracking, recording_state_manager=recording_state_manager)

        self.setWindowTitle("Patch GUI")

        self.patch_interface = patch_interface
        self.pipette_interface = pipette_interface
        self.recording_state_manager = recording_state_manager
        self._cell_list_signature = None
        self.cell_list_window = CellListWindow(self)
        self.cell_list_window.closed.connect(self._cells_window_closed)
        self.show_cells_button = QtWidgets.QPushButton("Show Cells")
        self.show_cells_button.setCheckable(True)
        self.show_cells_button.clicked.connect(self.toggle_cell_list_window)
        self.status_bar.insertPermanentWidget(0, self.show_cells_button)
        self._cell_list_timer = QtCore.QTimer(self)
        self._cell_list_timer.setInterval(500)
        self._cell_list_timer.timeout.connect(self._refresh_cell_list_window)

        self.patch_interface.moveToThread(pipette_interface.thread())
        self.interface_signals[self.patch_interface] = (self.patch_command_signal,
                                                        self.patch_reset_signal)
        self.add_config_gui(self.patch_interface.config)
        self.add_config_gui(self.patch_interface.protocol_config)
        logging.debug("Added config GUI.")
        classic_patching_tab = ClassicPatchButtons(self.patch_interface, pipette_interface, self.start_task,self.interface_signals, self.recording_state_manager)
        self.add_tab(classic_patching_tab, 'Classic Auto Patching', index = 0)

    def register_commands(self):
        super(PatchGui, self).register_commands()
        # self.register_mouse_action(Qt.LeftButton, Qt.ShiftModifier,
        #                            self.patch_interface.patch_with_move)
        self.register_mouse_action(Qt.LeftButton, Qt.NoModifier,
                                   self.patch_interface.add_cell)
        self.register_mouse_action(Qt.RightButton, Qt.ShiftModifier,
                                   self.patch_interface.handle_corner_right_click)
        self.register_key_action(Qt.Key_B, None,
                                 self.patch_interface.break_in)
        self.register_key_action(Qt.Key_F2, None,
                                 self.patch_interface.store_cleaning_position)
        self.register_key_action(Qt.Key_F3, None,
                                 self.patch_interface.store_rinsing_position)
        self.register_key_action(Qt.Key_F4, None,
                                 self.patch_interface.clean_pipette)

    def toggle_cell_list_window(self, checked=None):
        if checked is None:
            checked = self.show_cells_button.isChecked()
        if checked:
            self.show_cells_button.setText("Hide Cells")
            self._refresh_cell_list_window(force=True)
            self.cell_list_window.show()
            self.cell_list_window.raise_()
            self.cell_list_window.activateWindow()
            self._cell_list_timer.start()
        else:
            self.cell_list_window.close()

    def _cells_window_closed(self):
        self._cell_list_timer.stop()
        if self.show_cells_button.isChecked():
            self.show_cells_button.blockSignals(True)
            self.show_cells_button.setChecked(False)
            self.show_cells_button.blockSignals(False)
        self.show_cells_button.setText("Show Cells")

    def _refresh_cell_list_window(self, force=False):
        if not self.cell_list_window.isVisible():
            return
        cells = list(self.patch_interface.cells_to_patch)
        try:
            stage_reference = self.patch_interface.current_autopatcher.calibrated_stage.reference_position()
        except Exception:
            stage_reference = None
        signature = tuple(id(cell) for cell in cells)
        full_refresh = force or (signature != self._cell_list_signature)
        self.cell_list_window.update_cells(cells, stage_reference, full_refresh=full_refresh)
        self._cell_list_signature = signature

class CollapsibleGroupBox(QtWidgets.QGroupBox):
    def __init__(self, title="", parent=None):
        super(CollapsibleGroupBox, self).__init__(parent)
        self.setTitle("")  # Set the group box title to be blank to allow custom styling

        # Apply styles for rounded corners, grey borders, and consistent font
        self.setStyleSheet("""
            QGroupBox {
                border: 1px solid lightgray;  /* Light grey border */
                border-radius: 8px;           /* Rounded corners with 8px radius */
                margin-top: 6px;             /* Adjust top margin for visual separation */
                font-family: Arial, Helvetica, sans-serif;  /* Consistent font family */
                font-size: 14px;              /* Consistent font size for the group box */
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
                font-weight: bold;            /* Bold for the group box title */
            }
            QWidget {
                background-color: #f9f9f9;    /* Light grey background for the content area */
                border-radius: 8px;
                font-family: Arial, Helvetica, sans-serif;  /* Consistent font family */
                font-size: 14px;              /* Consistent font size for content area */
            }
            QPushButton {
                background-color: #ffffff;     /* White background for buttons */
                border: 1px solid lightgray;   /* Light grey border for buttons */
                border-radius: 6px;            /* Slightly rounded corners for buttons */
                padding: 3px;                  /* Padding for a better button look */
                font-family: Arial, Helvetica, sans-serif;  /* Consistent font family */
                font-size: 14px;               /* Adjusted font size for buttons */
                outline: none;                 /* Remove default focus outline */
            }
            QPushButton:hover {
                background-color: rgba(173, 216, 230, 0.5);  /* Light blue with 50% transparency on hover */
                border: 1px solid #87CEEB;       /* Soft blue border on hover */
            }
            QPushButton:pressed {
                background-color: #d1e7ff;     /* Light blue when pressed for a subtle effect */
            }
            QPushButton:focus {
                border: 1px solid #87CEEB;      /* Consistent border color on focus (soft blue) */
                outline: none;                  /* Remove blue edge or highlight on focus */
            }
        """)

        # Create a toggle button (arrow) for expanding/collapsing
        self.toggle_button = QtWidgets.QToolButton()
        self.toggle_button.setStyleSheet("QToolButton { border: none; font-family: Arial, Helvetica, sans-serif; font-size: 14px; }")
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.DownArrow)
        self.toggle_button.setText(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(True)
        self.toggle_button.clicked.connect(self.on_toggle)

        # Layout for the toggle button
        self.header_layout = QtWidgets.QHBoxLayout()
        self.header_layout.addWidget(self.toggle_button, alignment=Qt.AlignLeft)
        self.header_layout.addStretch()

        # Content area
        self.content_area = QtWidgets.QWidget()
        self.content_layout = QtWidgets.QVBoxLayout()
        self.content_area.setLayout(self.content_layout)

        # Main layout of the collapsible group box
        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_layout.addLayout(self.header_layout)
        self.main_layout.addWidget(self.content_area)
        self.main_layout.setContentsMargins(5, 5, 5, 5)  # Add some margin to create spacing inside
        self.setLayout(self.main_layout)

    def on_toggle(self):
        if self.toggle_button.isChecked():
            self.content_area.show()
            self.toggle_button.setArrowType(Qt.DownArrow)
        else:
            self.content_area.hide()
            self.toggle_button.setArrowType(Qt.RightArrow)

    def setContentLayout(self, layout):
        # Remove existing layout if any
        while self.content_layout.count():
            child = self.content_layout.takeAt(0)
            if child.widget():
                child.widget().setParent(None)
        self.content_layout.addLayout(layout)

class CellListWindow(QtWidgets.QDialog):
    closed = QtCore.pyqtSignal()

    def __init__(self, parent=None, thumbnail_size=96):
        super().__init__(parent=parent)
        self.setWindowTitle("Selected Cells")
        self.setWindowFlags(self.windowFlags() | Qt.Tool)
        self.setAttribute(Qt.WA_ShowWithoutActivating)

        self.thumbnail_size = thumbnail_size
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels([
            "Image",
            "Fluo Image",
            "Cell",
            "Stage (px)",
            "Stage (um)",
        ])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(False)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.table.verticalHeader().setDefaultSectionSize(self.thumbnail_size + 12)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.table)
        self.setLayout(layout)

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)

    def update_cells(self, cells, stage_reference=None, full_refresh=True):
        if self.table.rowCount() != len(cells):
            self.table.setRowCount(len(cells))
            full_refresh = True

        for row, cell in enumerate(cells):
            stage_px, img, stage_um, img_fluo = self._unpack_cell(cell)

            if full_refresh:
                self._set_image_cell(row, 0, img)
                self._set_image_cell(row, 1, img_fluo, empty_text="N/A")
                self._set_item(row, 2, str(row + 1))
                self._set_item(row, 3, self._format_vec(stage_px))
                self._set_item(row, 4, self._format_vec(stage_um))

    def _unpack_cell(self, cell):
        if cell is None:
            return None, None, None, None
        if len(cell) >= 4:
            return cell[0], cell[1], cell[2], cell[3]
        if len(cell) == 3:
            return cell[0], cell[1], cell[2], None
        return None, None, None, None

    def _set_item(self, row, col, text):
        item = self.table.item(row, col)
        if item is None:
            item = QtWidgets.QTableWidgetItem()
            item.setFlags(item.flags() ^ Qt.ItemIsEditable)
            self.table.setItem(row, col, item)
        item.setText(text)

    def _set_image_cell(self, row, col, image, empty_text=""):
        if image is None:
            self.table.removeCellWidget(row, col)
            item = QtWidgets.QTableWidgetItem(empty_text)
            item.setFlags(item.flags() ^ Qt.ItemIsEditable)
            self.table.setItem(row, col, item)
            return

        pixmap = self._image_to_pixmap(image)
        label = QtWidgets.QLabel()
        label.setAlignment(Qt.AlignCenter)
        if pixmap is not None:
            label.setPixmap(
                pixmap.scaled(
                    self.thumbnail_size,
                    self.thumbnail_size,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
        self.table.setCellWidget(row, col, label)

    def _image_to_pixmap(self, image):
        if image is None:
            return None
        img = np.array(image)
        if img.ndim == 2:
            img8 = self._normalize_to_uint8(img)
            q_image = QtGui.QImage(
                img8.data,
                img8.shape[1],
                img8.shape[0],
                img8.strides[0],
                QtGui.QImage.Format_Grayscale8,
            ).copy()
        else:
            img8 = self._normalize_to_uint8(img[..., 0])
            q_image = QtGui.QImage(
                img8.data,
                img8.shape[1],
                img8.shape[0],
                img8.strides[0],
                QtGui.QImage.Format_Grayscale8,
            ).copy()
        return QtGui.QPixmap.fromImage(q_image)

    def _normalize_to_uint8(self, img):
        img = img.astype(np.float32)
        min_val = float(np.min(img))
        max_val = float(np.max(img))
        if max_val > min_val:
            img = (img - min_val) / (max_val - min_val) * 255.0
        else:
            img = np.zeros_like(img, dtype=np.float32)
        return img.astype(np.uint8)

    def _format_vec(self, vec):
        if vec is None:
            return "N/A"
        arr = np.array(vec).astype(float).ravel()
        return ", ".join(f"{v:.1f}" for v in arr)

class ButtonTabWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.pos_update_timers = []
        self.pos_labels = []
        self.interface_signals = {}
        self.start_task = None
        self.section_buttons = {}  # Dictionary to store buttons by section
        self.section_button_map = {}  # section -> {button_name: button}
        self.active_buttons_by_section = {}  # section -> set(button_name)
        self.color_change_sections = []  # Sections that should change color on completion
        self.section_colors = {}  # Store custom colors for different sections


    def do_nothing(self):
        pass  # a dummy function for buttons that aren't implemented yet
    
    def run_sequential_commands(self, cmds, button=None, section=None, button_name=None):
        # Ensure cmds is a list
        if not isinstance(cmds, list):
            cmds = [cmds]
        else:
            cmds = self._flatten_sequential_cmds(cmds)
            
        # Have the button immediately lose focus to prevent persistent outline
        if button:
            button.clearFocus()
            
        # Store the command list and reset index
        self._seq_cmds = cmds
        self._seq_index = 0
        self._seq_button = button
        self._seq_section = section
        self._seq_button_name = button_name
        self._seq_active_style = False
        
        # Special case for reset button in any section (assuming it contains "Clear" or "Reset")
        if (section in self.section_buttons and button and 
            ("Clear" in button_name or "Reset" in button_name)):
            # Reset all section button colors before running the command
            self._reset_section_button_colors(section)
        elif (
            button
            and section in self.active_buttons_by_section
            and button_name in self.active_buttons_by_section[section]
        ):
            self._set_button_active_style(button)
            self._seq_active_style = True
        
        self._run_next_seq_command()

    def _flatten_sequential_cmds(self, cmds):
        flat_cmds = []
        for cmd in cmds:
            if isinstance(cmd, list):
                flat_cmds.extend(self._flatten_sequential_cmds(cmd))
            else:
                flat_cmds.append(cmd)
        return flat_cmds

    def _reset_section_button_colors(self, section):
        """Reset colors for all buttons in a section"""
        if section in self.section_buttons:
            for button_info in self.section_buttons[section]:
                button = button_info[0]
                button.setStyleSheet("")  # This will revert to the style from CollapsibleGroupBox

    def _run_next_seq_command(self):
        if self._seq_index >= len(self._seq_cmds):
            # No more commands; sequence complete
            # Update button color if this section should change colors and it's not a reset button
            if (self._seq_section in self.color_change_sections and self._seq_button and 
                not any(reset_term in self._seq_button_name for reset_term in ["Clear", "Reset"])):
                # Get the color for this section, or use default blue
                color = self.section_colors.get(self._seq_section, "rgba(0, 0, 255, 0.3)")
                self._set_button_completion_style(self._seq_button, color)
            elif self._seq_active_style and self._seq_button:
                self._seq_button.setStyleSheet("")
            return

        # Rest of the method implementation unchanged
        cmd = self._seq_cmds[self._seq_index]
        self._seq_index += 1

        # Check if the command is asynchronous (has task_description)
        if hasattr(cmd, 'task_description'):
            interface = cmd.__self__
            # Define a temporary slot that waits for the command to finish
            def on_finished(exit_code, message):
                try:
                    interface.task_finished.disconnect(on_finished)
                except Exception:
                    pass
                if exit_code != 0 and self._seq_active_style and self._seq_button:
                    self._seq_button.setStyleSheet("")
                    self._seq_active_style = False
                # Launch next command after current one finishes
                self._run_next_seq_command()
            # Connect to the task_finished signal
            interface.task_finished.connect(on_finished)
            # Start the task and execute the command
            self.start_task(cmd.task_description, interface)
            if interface in self.interface_signals:
                command_signal, _ = self.interface_signals[interface]
                command_signal.emit(cmd, None)
            else:
                cmd(None)
        else:
            # Synchronous command: run it immediately
            cmd()
            self._run_next_seq_command()


    def run_command(self, cmds):
        if isinstance(cmds, list):
            for cmd in cmds:
                if isinstance(cmd, list):
                    for sub_cmd in cmd:
                        self.execute_command(sub_cmd)
                else:
                    self.execute_command(cmd)
        else:
            self.execute_command(cmds)
    

    def execute_command(self, cmd):
        logging.info(f"Executing command: {cmd}")
        if hasattr(cmd, 'task_description'):
            self.start_task(cmd.task_description, cmd.__self__)
            if cmd.__self__ in self.interface_signals:
                command_signal, _ = self.interface_signals[cmd.__self__]
                command_signal.emit(cmd, None)
            else:
                cmd(None)
        else:
            cmd()

    def _set_button_completion_style(self, button, color="rgba(0, 0, 255, 0.3)"):
        if button is None:
            return
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: {color}; 
                border: 1px solid lightgray;
                border-radius: 6px;
            }}
            QPushButton:hover {{
                background-color: rgba(173, 216, 230, 0.5);
                border: 1px solid #87CEEB;
            }}
            QPushButton:pressed {{
                background-color: #d1e7ff;
            }}
            QPushButton:focus {{
                border: 1px solid lightgray;
                outline: none;
            }}
        """)

    def _set_button_active_style(self, button, color="rgba(173, 216, 230, 0.5)"):
        self._set_button_completion_style(button, color)

    def addPositionBox(self, name: str, layout, update_func, tare_func=None, axes=['x', 'y', 'z']):
        # Use CollapsibleGroupBox instead of QGroupBox
        box = CollapsibleGroupBox(name)
        row = QtWidgets.QHBoxLayout()
        indices = []
        # Create a new row for each position
        for j, axis in enumerate(axes):
            # Create a label for the position
            label = QtWidgets.QLabel(f'{axis}: TODO')
            row.addWidget(label)

            indices.append(len(self.pos_labels))
            self.pos_labels.append(label)
        box.setContentLayout(row)
        layout.addWidget(box)

        if tare_func is not None:
            # Add a button to tare the manipulator
            tare_button = QtWidgets.QPushButton('Tare')
            tare_button.clicked.connect(lambda: tare_func())
            row.addWidget(tare_button)

        # Periodically update the position labels
        pos_timer = QtCore.QTimer()
        pos_timer.timeout.connect(lambda: update_func(indices))
        pos_timer.start(16)
        self.pos_update_timers.append(pos_timer)

    def positionAndTareBox(self, name: str, layout, update_func, tare_funcs, axes=['x', 'y', 'z']):
        # Use CollapsibleGroupBox instead of QGroupBox
        box = CollapsibleGroupBox(name)
        main_layout = QtWidgets.QHBoxLayout()
        indices = []

        for j, axis in enumerate(axes):
            axis_layout = QtWidgets.QVBoxLayout()

            # Create a label for the position
            label = QtWidgets.QLabel(f'{axis}: 0.00')
            axis_layout.addWidget(label)
            indices.append(len(self.pos_labels))
            self.pos_labels.append(label)

            # Add a button to tare the manipulator
            tare_button = QtWidgets.QPushButton(f'Tare {axis}')
            tare_button.clicked.connect(tare_funcs[j])
            axis_layout.addWidget(tare_button)

            main_layout.addLayout(axis_layout)

        box.setContentLayout(main_layout)
        layout.addWidget(box)

        # Periodically update the position labels
        pos_timer = QtCore.QTimer()
        pos_timer.timeout.connect(lambda: update_func(indices))
        pos_timer.start(16)
        self.pos_update_timers.append(pos_timer)

    def addButtonList(self, box_name: str, layout: QtWidgets.QVBoxLayout, buttonNames: list[list[str]], 
                    cmds, sequential=False, change_color_on_complete=False, 
                    completion_color="rgba(0, 0, 255, 0.3)",
                    change_color_during=None):
        # Use CollapsibleGroupBox instead of QGroupBox
        box = CollapsibleGroupBox(box_name)
        rows = QtWidgets.QVBoxLayout()
        
        # Initialize list to store buttons for this section
        section_buttons = []
        
        # Store color change preference and custom color for this section
        if change_color_on_complete:
            self.color_change_sections.append(box_name)
            self.section_colors[box_name] = completion_color
        
        for i, buttons_in_row in enumerate(buttonNames):
            new_row = QtWidgets.QHBoxLayout()
            new_row.setAlignment(Qt.AlignLeft)

            for j, button_name in enumerate(buttons_in_row):
                button = QtWidgets.QPushButton(button_name)
                button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
                button.setMinimumWidth(30)
                button.setMinimumHeight(30)
                
                # Track this button for this section
                section_buttons.append((button, i, j, button_name))
                self.section_button_map.setdefault(box_name, {})[button_name] = button

                # Use a lambda function with default arguments to correctly capture the command
                if i < len(cmds) and j < len(cmds[i]):
                    button_cmd = cmds[i][j]
                    if sequential:
                        button.clicked.connect(lambda state, cmd=button_cmd, btn=button, section=box_name, 
                                            name=button_name: self.run_sequential_commands(cmd, btn, section, name))
                    else:
                        button.clicked.connect(lambda state, cmd=button_cmd: self.run_command(cmd))
                else:
                    button.clicked.connect(self.do_nothing)

                new_row.addWidget(button)
            rows.addLayout(new_row)
        
        # Store buttons for this section
        self.section_buttons[box_name] = section_buttons
        if change_color_during:
            if change_color_during is True:
                active_names = {name for row in buttonNames for name in row}
            else:
                active_names = set(change_color_during)
            self.active_buttons_by_section[box_name] = active_names

        box.setContentLayout(rows)
        layout.addWidget(box)
        return section_buttons

    def get_section_button(self, section: str, name: str):
        return self.section_button_map.get(section, {}).get(name)


class FileSelector(QWidget):
    fileSelected = pyqtSignal(str)  # Signal to emit the selected file path

    def __init__(self):
        super().__init__()

    def open_file_dialog(self):
        # Open the file dialog in non-blocking mode
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, 
                                                   "Select CSV File", 
                                                   "", 
                                                   "CSV Files (*.csv);;All Files (*)", 
                                                   options=options)
        if file_name:
            # Emit the signal with the selected file path
            self.fileSelected.emit(file_name)
class ClassicPatchButtons(ButtonTabWidget):
    def __init__(self, patch_interface: AutoPatchInterface, pipette_interface: PipetteInterface, start_task, interface_signals, recording_state_manager: RecordingStateManager):
        super().__init__()
        self.patch_interface = patch_interface
        self.pipette_interface = pipette_interface

        self.start_task = start_task

        self.interface_signals = interface_signals

        self.recording_state_manager = recording_state_manager

        layout = QtWidgets.QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)

        self.stage_xy = [0, 0]
        self.stage_z = 0
        self.pipette_xyz = [0, 0, 0]
        self.tare_pipette_pos = [0, 0, 0]

        self.currx_stage_pos = [0, 0, 0]
        self.curry_stage_pos = [0, 0, 0]
        self.currz_stage_pos = [0, 0, 0]

        self.file_selector = FileSelector()


        self.recorder = FileLogger(self.recording_state_manager, folder_path="experiments/Data/rig_recorder_data/", recorder_filename="movement_recording")

        self.positionAndTareBox(
            'stage position (um)',
            layout,
            self.update_stage_pos_labels,
            tare_funcs=[self.tare_stage_x, self.tare_stage_y, self.tare_stage_z]
        )
        self.addPositionBox(
            'pipette position (um)',
            layout,
            self.update_pipette_pos_labels,
            tare_func=self.tare_pipette
        )

        self.stage_calibration = [
            self.pipette_interface.set_floor,
            self.pipette_interface.calibrate_stage,
            lambda: self.pipette_interface.move_microscope(
                float(self.pipette_interface.calibrated_unit.config.home_position_delta_um)
            ),
        ]
        self.pipette_calibration = [self.pipette_interface.calibrate_manipulator, self.patch_interface.store_calibration_positions, self.patch_interface.move_to_safe_space]
        self.pipette_calibration_no_move = [self.pipette_interface.calibrate_manipulator, self.patch_interface.store_calibration_positions]
        self.pipette_cleaning_calibration = [self.patch_interface.store_cleaning_position,self.patch_interface.move_pipette_up,self.patch_interface.move_to_safe_space]


        # Add a box for calibration setup
        buttonList = [['Calibrate Stage','Calibrate Pipette'],['Store Cleaning Position','Clear Calibration']]
        # buttonList = [['Calibrate Stage','Calibrate Pipette'],['Store Cleaning Position'],['Load Calibration','Clear Calibration']]
        cmds = [[self.stage_calibration, self.pipette_calibration],
                # [self.patch_interface.store_home_position, self.patch_interface.store_safe_position],
                [[self.pipette_cleaning_calibration],[self.load_calibration, self.patch_interface.clear_positions]]
        ]
        self.addButtonList('calibration', layout, buttonList, cmds, sequential=True, 
                        change_color_on_complete=True, completion_color="rgba(173, 216, 230, 0.5)")

        # Add a box for movement commands 

        
        buttonList = [['move group down','move group up'],['move group in x','move group in y'],
                      ['Move to Safe Position','Move to Home Position'],
                      ['Move to cell plane','Focus Stage'],
                      ['Store corners', 'Start Scan', 'Constant Disturbance'],
                      ['Center Pipette','Clean pipette','Focus Pipette']]

        cmds = [
            [self.patch_interface.move_group_down, self.patch_interface.move_group_up],
            [self.patch_interface.move_group_in_x, self.patch_interface.move_group_in_y],
            [self.patch_interface.move_to_safe_space, self.patch_interface.move_to_home_space],
            [self.pipette_interface.go_to_floor,self.pipette_interface.focus_stage],


            [
                self.patch_interface.start_selecting_corners,
                [[self.patch_interface.move_to_scan_start, self.start_recording, self.patch_interface.start_scan, self.stop_recording]],
                self.toggle_constant_disturbance,
            ],
            [self.pipette_interface.center_pipette,self.patch_interface.clean_pipette,self.pipette_interface.focus_pipette]

        ]
        self.addButtonList('movement', layout, buttonList, cmds, sequential=True)
        self.constant_disturbance_active = False
        self.constant_disturbance_button = self.get_section_button('movement', 'Constant Disturbance')
        if self.constant_disturbance_button is not None:
            self.constant_disturbance_button.setCheckable(True)


        # self.pipette_location = [self.pipette_interface.follow_stage, self.pipette_interface.move_pipette_random,self.rest,self.start_recording,self.patch_interface.find_pipette]
        # self.pipette_location = [self.pipette_interface.follow_stage, self.pipette_interface.move_pipette_random,self.patch_interface.find_pipette]
        self.pipette_location = [self.patch_interface.find_pipette]
        # add a box for testing controllability of the pipette and stage
        buttonList = [['Follow Stage','Move Pipette Random','Find Pipette']]
        cmds = [[self.pipette_interface.follow_stage, self.pipette_interface.move_pipette_random,self.patch_interface.find_pipette]
                ]
        self.addButtonList('testing', layout, buttonList, cmds,sequential=True)

        # Add a box for light controls
        buttonList = [['toggle Light', 'toggle fluorescense'],
                      ['move cube left', 'move cube right']]
        cmds = [[self.toggle_cell_sorter_led, self.patch_interface.toggle_fluorescence],
                [self.patch_interface.move_cube_left, self.patch_interface.move_cube_right]]
        self.addButtonList('Light', layout, buttonList, cmds)

        self.cell_sorter_led_button = self.get_section_button('Light', 'toggle Light')
        if self.cell_sorter_led_button is not None:
            self.cell_sorter_led_button.setCheckable(True)
            self.cell_sorter_led_button.setChecked(False)
            self._update_cell_sorter_led_button_style(False)
            self.toggle_cell_sorter_led(False)

        # Add a box for patching commands
        buttonList = [['Select Cell','Remove Last Cell','Center on Cell','Move Stage to Cell'],
                      ['Locate Cell','Hunt Cell','Gigaseal'],
                      ['Break-in','Escape Cell'],
                      ['Patch Cell','Attempt Whole Cell','Run Protocols']]
        cmds = [[self.patch_interface.start_selecting_cells, self.patch_interface.remove_last_cell, self.patch_interface.center_on_cell, self.patch_interface.move_stage_to_cell],
                [self.patch_interface.locate_cell,
                 [self.start_recording,self.patch_interface.hunt_cell],
                 [self.cell_sorter_led_off, self.patch_interface.gigaseal]],
                [[self.cell_sorter_led_off, self.patch_interface.break_in],
                 [self.stop_recording, self.cell_sorter_led_on, self.patch_interface.escape_cell]],
                [[self.start_recording, self.cell_sorter_led_off, self.patch_interface.patch, self.stop_recording],
                 [self.start_recording, self.cell_sorter_led_off, self.patch_interface.whole_cell, self.stop_recording],
                 [self.stop_recording, self.cell_sorter_led_off, self.patch_interface.run_protocols]]

  
]
        self.addButtonList(
            'patching',
            layout,
            buttonList,
            cmds,
            sequential=True,
            change_color_during={
                'Locate Cell',
                'Hunt Cell',
                'Gigaseal',
                'Break-in',
                'Escape Cell',
                'Patch Cell',
                'Attempt Whole Cell',
                'Run Protocols',
            },
        )

        # Add a box for Rig Recorder
        self.record_button = QtWidgets.QPushButton("Start Recording")
        self.record_button.clicked.connect(self.toggle_recording)
        self.record_button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.record_button.setMinimumWidth(30)
        self.record_button.setMinimumHeight(30)
        layout.addWidget(self.record_button)

        self.setLayout(layout)

    def toggle_constant_disturbance(self):
        if self.constant_disturbance_active:
            self.stop_constant_disturbance()
        else:
            self.start_constant_disturbance()

    def start_constant_disturbance(self):
        self.constant_disturbance_active = True
        self._update_constant_disturbance_button(True)
        self.start_recording()

        cmd = self.patch_interface.constant_disturbance
        interface = cmd.__self__

        def on_finished(exit_code, message):
            try:
                interface.task_finished.disconnect(on_finished)
            except Exception:
                pass
            if self.constant_disturbance_active:
                self.constant_disturbance_active = False
                self._update_constant_disturbance_button(False)
                QtCore.QTimer.singleShot(5000, self.stop_recording)

        interface.task_finished.connect(on_finished)
        self.start_task(cmd.task_description, interface)
        if interface in self.interface_signals:
            command_signal, _ = self.interface_signals[interface]
            command_signal.emit(cmd, None)
        else:
            cmd(None)

    def stop_constant_disturbance(self):
        self.constant_disturbance_active = False
        self._update_constant_disturbance_button(False)
        self.patch_interface.stop_constant_disturbance()
        QtCore.QTimer.singleShot(5000, self.stop_recording)

    def _update_constant_disturbance_button(self, active):
        if self.constant_disturbance_button is None:
            return
        self.constant_disturbance_button.blockSignals(True)
        self.constant_disturbance_button.setChecked(active)
        self.constant_disturbance_button.blockSignals(False)
        if active:
            self.constant_disturbance_button.setText("Stop Disturbance")
            self.constant_disturbance_button.setStyleSheet("background-color: red; color: white;border-radius: 5px; padding: 5px;")
        else:
            self.constant_disturbance_button.setText("Constant Disturbance")
            self.constant_disturbance_button.setStyleSheet("")
    def load_calibration(self):
        self.file_selector.fileSelected.connect(self.load_calibration_file)  # Connect the signal to the slot
        self.file_selector.open_file_dialog()  # Open the file dialog


    def load_calibration_file(self, file_path):
        # call pipette.interface.read_calibration
            logging.info(f"Loading calibration file: {file_path}")
            self.pipette_interface.read_calibration(file_path)


    def test_movement(self):
        # check if recording is enabled
        if self.recording_state_manager.is_recording_enabled():
            # Opens the file selector dialog without blocking the main thread
            self.file_selector.fileSelected.connect(self.load_movement_file)  # Connect the signal to the slot
            self.file_selector.open_file_dialog()
        else:
            # if not recording then start recording
            self.toggle_recording()
            # Opens the file selector dialog without blocking the main thread
            self.file_selector.open_file_dialog()
        
    def load_movement_file(self, file_path):
        logging.info(f"Loading movement file: {file_path}")
        # # send file to the pipette interface
        self.patch_interface.send_movement_file(file_path)

        
    def toggle_recording(self):
        if self.recording_state_manager.is_recording_enabled():
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        self.recording_state_manager.set_recording(True)
        self.record_button.setText("Stop Recording")
        self.record_button.setStyleSheet("background-color: red; color: white;border-radius: 5px; padding: 5px;")
        logging.info("Recording started")

    def stop_recording(self):
        self.recording_state_manager.set_recording(False)
        self.recorder.handle_recording_stopped()
        self.record_button.setText("Start Recording")
        self.record_button.setStyleSheet("")
        logging.info("Recording stopped")

    def _update_cell_sorter_led_button_style(self, enabled: bool):
        if enabled:
            self.cell_sorter_led_button.setStyleSheet("""
                QPushButton {
                    background-color: rgba(173, 216, 230, 0.5);
                    border: 1px solid lightgray;
                    border-radius: 6px;
                }
                QPushButton:hover {
                    background-color: rgba(173, 216, 230, 0.5);
                    border: 1px solid #87CEEB;
                }
                QPushButton:pressed {
                    background-color: #d1e7ff;
                }
                QPushButton:focus {
                    border: 1px solid lightgray;
                    outline: none;
                }
            """)
        else:
            self.cell_sorter_led_button.setStyleSheet("")

    def _set_cell_sorter_led_state(self, enabled: bool):
        if self.cell_sorter_led_button.isChecked() != enabled:
            self.cell_sorter_led_button.blockSignals(True)
            self.cell_sorter_led_button.setChecked(enabled)
            self.cell_sorter_led_button.blockSignals(False)
        self._update_cell_sorter_led_button_style(enabled)
        if enabled:
            self.patch_interface.cell_sorter_led_on()
        else:
            self.patch_interface.cell_sorter_led_off()

    def cell_sorter_led_off(self):
        self._set_cell_sorter_led_state(False)

    def cell_sorter_led_on(self):
        self._set_cell_sorter_led_state(True)

    def toggle_cell_sorter_led(self, checked=None):
        enabled = self.cell_sorter_led_button.isChecked() if checked is None else bool(checked)
        self._set_cell_sorter_led_state(enabled)



    def close(self):
        self.recorder.close()
        super(ClassicPatchButtons, self).close()

    def closeEvent(self, event):
        self.recorder.close()
        super(ClassicPatchButtons, self).closeEvent(event)

    def tare_pipette(self):
        currPos = self.pipette_interface.calibrated_unit.unit.position()
        self.tare_pipette_pos = currPos
        self.pipette_interface.tare_pipette = np.array(self.tare_pipette_pos)
        print("Tare pipette: ", self.tare_pipette_pos)
        self.pipette_interface.write_tare()

    def update_pipette_pos_labels(self, indices):
        # Update the position labels
        # start_time = time.perf_counter_ns()
        # currPos = self.pipette_interface.calibrated_unit.unit.position()
        recPos  = self.pipette_interface.calibrated_unit.unit.position()
        currPos = recPos - self.tare_pipette_pos
        if self.recording_state_manager.is_recording_enabled():
            self.recorder.setBatchMoves(True)
            timestamp = datetime.now().timestamp()
            # logging.info(f"the current time is {timestamp}")
            self.recorder.write_movement_data_batch(
                timestamp,
                self.stage_xy[0],
                self.stage_xy[1],
                self.stage_z,
                recPos[0],
                recPos[1],
                recPos[2]
            )

        self.pipette_xyz = currPos
        # print("Pipette position: ", self.pipette_xyz)

        for i, ind in enumerate(indices):
            label = self.pos_labels[ind]
            label.setText(f'{label.text().split(":")[0]}: {currPos[i]:.2f}')

    def tare_stage_x(self):
        xPos = self.pipette_interface.calibrated_stage.position(0)
        self.currx_stage_pos = [xPos, 0, 0]
        # update pipette controller stage tare at x position as a numpy array
        self.pipette_interface.tare_stage[0] = xPos
        print("Tare stage x: ", self.currx_stage_pos)
        self.pipette_interface.write_tare()

    def tare_stage_y(self):
        yPos = self.pipette_interface.calibrated_stage.position(1)
        self.curry_stage_pos = [0, yPos, 0]
        # update pipette controller stage tare at y position as a numpy array
        self.pipette_interface.tare_stage[1] = yPos
        print("Tare stage y: ", self.curry_stage_pos)
        self.pipette_interface.write_tare()

    def tare_stage_z(self):
        zPos = self.pipette_interface.microscope.position()
        self.currz_stage_pos = [0, 0, zPos]
        # update pipette controller stage tare at z position as a numpy array
        self.pipette_interface.tare_stage[2] = zPos
        print("Tare stage z: ", self.currz_stage_pos)
        self.pipette_interface.write_tare()

    def update_stage_pos_labels(self, indices):
        xyRecPos = self.pipette_interface.calibrated_stage.position()
        zRecPos = self.pipette_interface.microscope.position()
        xyPos = xyRecPos - self.currx_stage_pos[0:2] - self.curry_stage_pos[0:2]
        zPos = zRecPos - self.currz_stage_pos[2]
        self.stage_xy = xyRecPos
        self.stage_z = zRecPos

        for i, ind in enumerate(indices):
            label = self.pos_labels[ind]
            if i < 2:
                label.setText(f'{label.text().split(":")[0]}: {xyPos[i]:.2f}')
            else:
                label.setText(f'{label.text().split(":")[0]}: {zPos:.2f}')


