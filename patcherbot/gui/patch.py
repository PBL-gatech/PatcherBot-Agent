from __future__ import absolute_import

from types import MethodType

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal, QObject
import PyQt5.QtGui as QtGui
import numpy as np
import logging

from PyQt5.QtWidgets import QFileDialog, QTabWidget, QWidget,QMessageBox
import qtawesome as qta

from patcherbot.controller import TaskController
from patcherbot.gui.tabs.experiment_book_tab import ExperimentBookTab
from patcherbot.gui.tabs.atlas_widget import AtlasWindow
from patcherbot.gui.tabs.show_cells import CellListWindow
from patcherbot.gui.manipulator import ManipulatorGui
from patcherbot.interface.experimentBookConfig import ExperimentBookConfig
from patcherbot.interface.patch import AutoPatchInterface
from patcherbot.interface.pipettes import PipetteInterface
from patcherbot.utils.RecordingStateManager import RecordingStateManager
from patcherbot.interface.base import command

from patcherbot.utils.FileLogger import FileLogger
from datetime import datetime
import json
import pickle
import os

import qdarktheme

PATCH_GUI_COLORS = {
    "[light]": {
        # Main application
        "primary": "#000000",
        "background": "#F4F6F8",
        "foreground": "#202428",
        "border": "#D2D7DD",

        # Panels / windows
        "background>panel": "#FFFFFF",

        # Inputs / combo boxes
        "input.background": "#FFFFFF",
        "border>input": "#C8CED6",
        "foreground>input.placeholder": "#858C95",

        # Combo popup
        "background>popup": "#FFFFFF",
        "popupItem.selectionBackground": "#E6EEF9",

        # Lists / tables
        "background>list": "#FFFFFF",
        "list.alternateBackground": "#F7F8FA",
        "list.hoverBackground": "#EEF2F6",

        "background>table": "#FFFFFF",
        "table.alternateBackground": "#F7F8FA",
        "tableSectionHeader.background": "#EEF1F4",

        # Toolbar
        "toolbar.background": "#ECEFF2",
        "toolbar.hoverBackground": "#DDE2E7",
        "toolbar.activeBackground": "#D2D8DE",

        # Tabs
        "tab.activeBackground": "#FFFFFF",
        "tab.hoverBackground": "#E9EDF1",

        # Scrollbars
        "scrollbar.background": "#EFF1F3",
        "scrollbarSlider.background": "#C1C7CE",
        "scrollbarSlider.hoverBackground": "#AEB5BD",
        "scrollbarSlider.activeBackground": "#989FA8",

        # Status bar
        "statusBar.background": "#ECEFF2",
    },

    "[dark]": {
        # Main application
        "primary": "#A3A3A3",
        "background": "#0A0A0A",
        "foreground": "#EDEDED",
        "border": "#2A2A2A",

        # Panels / secondary surfaces
        "background>panel": "#111111",

        # Inputs / combo boxes
        "input.background": "#171717",
        "border>input": "#303030",
        "foreground>input.placeholder": "#777777",

        # Combo popup
        "background>popup": "#141414",
        "popupItem.selectionBackground": "#2A2A2A",

        # Lists
        "background>list": "#111111",
        "list.alternateBackground": "#151515",
        "list.hoverBackground": "#202020",

        # Tables
        "background>table": "#111111",
        "table.alternateBackground": "#151515",
        "tableSectionHeader.background": "#1C1C1C",

        # Toolbar
        "toolbar.background": "#0D0D0D",
        "toolbar.hoverBackground": "#1C1C1C",
        "toolbar.activeBackground": "#282828",

        # Tabs
        "tab.activeBackground": "#202020",
        "tab.hoverBackground": "#191919",

        # Scrollbars
        "scrollbar.background": "#0D0D0D",
        "scrollbarSlider.background": "#333333",
        "scrollbarSlider.hoverBackground": "#484848",
        "scrollbarSlider.activeBackground": "#5C5C5C",

        # Status bar
        "statusBar.background": "#0D0D0D",
    }
}

PATCH_GUI_QSS = """
    QToolBar {
        spacing: 6px;
        padding: 4px;
    }

    QToolBar QLabel {
        margin-left: 6px;
        margin-right: 2px;
    }

    QScrollArea {
        border: none;
    }

    QSplitter::handle:horizontal {
        width: 3px;
    }

    QTabWidget::pane {
        padding: 2px;
    }
    """

class PatchSignals(QtCore.QObject):
    """
    A dedicated container to hold unique signals for a single pipette.
    This prevents multiple tabs from broadcasting on the same channel.
    """
    command = QtCore.pyqtSignal(MethodType, object) 
    reset = QtCore.pyqtSignal(TaskController)

class PatchGui(ManipulatorGui):
    """
    GUI class for controlling the automated patch-clamp system. Inherits from ManipulatorGui.

    Provides cell selection display, integration with pipette and patching interfaces, 
    and configurable controls for manual and automated patching tasks.
    """

    def __init__(self, camera, pipette_cameras, pipette_interfaces, patch_interfaces, recording_state_manager: RecordingStateManager, camera_recording_session=None, rig_recorder=None, movement_recorder=None, graph_recorder=None, with_tracking=False):
        """
        Initialize the patch GUI.

        Args:
            camera: Primary camera object for live imaging.
            aux_camera: Auxiliary camera object.
            pipette_interfaces (PipetteInterface): Interface for controlling pipette manipulations.
            patch_interface (AutoPatchInterface): Interface for automated patching operations.
            recording_state_manager (RecordingStateManager): Manager for recording state and sessions.
            with_tracking (bool, optional): Whether to enable tracking features. Defaults to False.
        """
        super(PatchGui, self).__init__(camera, pipette_cameras, pipette_interfaces, with_tracking=with_tracking, recording_state_manager=recording_state_manager)

        self.setWindowTitle("Patch GUI")
        self.resize(1200, 1000)

        if not isinstance (pipette_interfaces, dict):
            self.pipette_interfaces = {"pipette": pipette_interfaces}
            self.patch_interfaces = {list(self.pipette_interfaces.keys())[0]: patch_interfaces}
        else:
            self.pipette_interfaces = pipette_interfaces
            self.patch_interfaces = patch_interfaces
        self.recording_state_manager = recording_state_manager
        self.camera_recording_sessiong = camera_recording_session
        self.graph_recorder = graph_recorder

        self._cell_list_signature = None
        self.cell_list_window = CellListWindow(self)
        self.cell_list_window.closed.connect(self._cells_window_closed)
        self.show_cells_button = QtWidgets.QPushButton("Show Cells")
        self.show_cells_button.setCheckable(True)
        self.show_cells_button.clicked.connect(self.toggle_cell_list_window)
        self.status_bar.insertPermanentWidget(0, self.show_cells_button)
        self.atlas_window = AtlasWindow(self)
        self.atlas_window.finished.connect(self._atlas_window_closed)
        self.show_atlas_button = QtWidgets.QPushButton("Show Atlas")
        self.show_atlas_button.setCheckable(True)
        self.show_atlas_button.clicked.connect(self.toggle_atlas_window)
        self.status_bar.insertPermanentWidget(1, self.show_atlas_button)
        self._cell_list_timer = QtCore.QTimer(self)
        self._cell_list_timer.setInterval(500)
        self._cell_list_timer.timeout.connect(self._refresh_cell_list_window)

        self.patch_interface.moveToThread(pipette_interface.thread())
        self._display_position_timer = QtCore.QTimer(self)
        self._display_position_timer.setInterval(50)
        self._display_position_timer.timeout.connect(
            lambda: self.patch_interface.update_camera_cell_list())
        self._display_position_timer.start()
        self.interface_signals[self.patch_interface] = (self.patch_command_signal,
                                                        self.patch_reset_signal)
        self.add_config_gui(self.patch_interface.config)
        self.add_config_gui(self.patch_interface.protocol_config)
        self.experiment_book_tab = self.add_config_gui(
            self.patch_interface.experiment_book_config,
            gui_class=ExperimentBookTab,
        )

        self.classic_tab = PipetteStackTab()
        self.calibration_tab = PipetteStackTab()
        self.patch_config_tab = PipetteStackTab()
        self.protocol_tab = PipetteStackTab()

        self.classic_tabs = {}
        self.calibration_guis = {}
        self.patch_config_guis = {}
        self.protocol_guis = {}

        # Experiment book is currently a shared session, so only
        # construct one GUI for it.
        self.experiment_book_tab = ExperimentBookTab(
            self.experiment_book_session
        )

        self._unique_patch_signals = {}

        self.rig_recorder = rig_recorder
        if movement_recorder is None:
            raise ValueError(
                "PatchGui requires a shared movement_recorder"
            )

        self.movement_recorder = movement_recorder

        self.shared_rig_controls = SharedRigControls(
            pipette_interfaces=self.pipette_interfaces,
            patch_interfaces=self.patch_interfaces,
            start_task=self.start_task,
            interface_signals=self.interface_signals,
            recording_state_manager=self.recording_state_manager,
            movement_recorder=self.movement_recorder,
            graph_recorder=self.graph_recorder,
        )

        for id, curr_pipette_interface in self.pipette_interfaces.items():
            curr_patch_interface = self.patch_interfaces[id]

            self.switch_manipulator_box.addItem(f"{id}")
            curr_patch_interface.moveToThread(self.control_threads[id])

            signals = PatchSignals()
            self._unique_patch_signals[id] = signals
            self.interface_signals[curr_patch_interface] = (signals.command, signals.reset)

            logging.debug("Added config GUI.")

            classic_gui = ClassicPatchButtons(
                curr_patch_interface,
                curr_pipette_interface,
                self.start_task,
                self.interface_signals,
                self.recording_state_manager,
                self.movement_recorder,
                self.shared_rig_controls,
            )

            self.classic_tabs[id] = classic_gui

            self.classic_tab.add_pipette_widget(
                id,
                classic_gui,
            )

            calibration_gui = ConfigGui(
                curr_pipette_interface.calibration_config
            )

            self.calibration_guis[id] = calibration_gui

            self.calibration_tab.add_pipette_widget(
                id,
                calibration_gui,
            )

            patch_config_gui = ConfigGui(
                curr_patch_interface.config
            )

            self.patch_config_guis[id] = patch_config_gui

            self.patch_config_tab.add_pipette_widget(
                id,
                patch_config_gui,
            )

            protocol_gui = ConfigGui(
                curr_patch_interface.protocol_config
            )

            self.protocol_guis[id] = protocol_gui

            self.protocol_tab.add_pipette_widget(
                id,
                protocol_gui,
            )

        self.config_tabs.addTab(
            self.classic_tab,
            "Classic Patching",
        )

        self.config_tabs.addTab(
            self.calibration_tab,
            "Calibration",
        )

        self.config_tabs.addTab(
            self.patch_config_tab,
            "Patch Config",
        )

        self.config_tabs.addTab(
            self.protocol_tab,
            "Protocols",
        )

        self.config_tabs.addTab(
            self.experiment_book_tab,
            "Experiment Book",
        )

        self.current_tab = self.config_tabs

        # Container that remains constant when pipettes switch.
        self.config_panel = QtWidgets.QWidget()

        config_panel_layout = QtWidgets.QVBoxLayout(
            self.config_panel
        )

        config_panel_layout.setContentsMargins(
            0, 0, 0, 0
        )

        # Always-visible shared rig controls.
        config_panel_layout.addWidget(
            self.shared_rig_controls
        )

        # Existing per-pipette tab sets temporarily live here.
        config_panel_layout.addWidget(
            self.config_tabs,
            1,
        )

        self.config_scroll_area = QtWidgets.QScrollArea()
        self.config_scroll_area.setWidgetResizable(True)

        self.config_scroll_area.setWidget(
            self.config_panel
        )

        self.config_scroll_area.setMinimumWidth(100) 
        
        self.splitter.addWidget(self.config_scroll_area)
        self.splitter.setSizes([2000, 500])
        self.splitter.setStretchFactor(0, 1) 
        self.splitter.setStretchFactor(1, 0)

        self.switch_manipulator_box.currentTextChanged.connect(self.switch_active_pipette)
        self.switch_manipulator_box.currentIndexChanged.connect(self.set_active_pipette_camera_index)

        self.active_patch_interface = list(self.patch_interfaces.values())[0] if isinstance(patch_interfaces, dict) else patch_interfaces

        self.main_toolbar_default_style = self.main_toolbar.styleSheet()
        self.patch_toolbar_default_style = self.patch_toolbar.styleSheet()
        # self.config_tab_default_style = list(self.config_tabs.values())[0].styleSheet()
        self.cell_list_window_default_style = self.cell_list_window.styleSheet()
        self.pipette_status_window_default_style = self.pipette_status_window.styleSheet()

        self.apply_theme("light")
        self.experiment_book_tab.attach_recording_state_manager(
            self.recording_state_manager
        )
        self.patch_interface.state_press_tally_changed.connect(
            self.experiment_book_tab.handle_state_press_tally
        )
        self.snapshot_captured.connect(self.experiment_book_tab.handle_snapshot)
        self._origin_busy = False
        self.patch_interface.origin_saved.connect(self._origin_saved)
        self.patch_interface.task_finished.connect(self._origin_task_finished)
        logging.debug("Added config GUI.")
        classic_patching_tab = ClassicPatchButtons(self.patch_interface, pipette_interface, self.start_task, self.interface_signals, self.recording_state_manager)
        self.classic_patching_tab = classic_patching_tab
        classic_patching_tab.origin_requested.connect(self._request_origin)
        self.add_tab(classic_patching_tab, 'PatcherBot Agent', index = 0)
        self.record_button.clicked.disconnect(self.toggle_recording)
        self.record_button.clicked.connect(classic_patching_tab.toggle_recording)
        self.record_button.setToolTip('Start/stop recording')
        self._recording_indicator_timer = QtCore.QTimer(self)
        self._recording_indicator_timer.timeout.connect(
            lambda: self.record_button.setChecked(self.recording_state_manager.is_recording_enabled())
        )
        self._recording_indicator_timer.start(100)

    def close(self):
        self._display_position_timer.stop()
        self.pipette_interface.display_positions = None
        return super(PatchGui, self).close()

    def _origin_experiment_details(self):
        book = self.experiment_book_tab
        if book.active_details is not None:
            return dict(book.active_details)
        return {key: str(getattr(book.config, key, "")) for key in
                ("experiment_name", "strain_culture", "gender", "age")}

    @QtCore.pyqtSlot(str)
    def _request_origin(self, axis):
        if self._origin_busy:
            return
        if (getattr(self, "running_task", None) is not None
                or self.patch_interface._current_controller is not None
                or self.pipette_interface._current_controller is not None):
            self.classic_patching_tab.set_origin_status(
                "Wait for the current movement/task to finish.", error=True)
            return
        request = {
            "experiment": self._origin_experiment_details(),
            "logger": self.experiment_book_tab.logger,
            "camera_interface": self.main_interface,
        }
        command = (self.patch_interface.save_x_origin if axis == "x"
                   else self.patch_interface.save_y_origin)
        self._origin_busy = True
        self.classic_patching_tab.set_origin_busy(True)
        self.classic_patching_tab.set_origin_status("Saving origin and microscope image...")
        self.start_task(command.task_description, self.patch_interface)
        self.patch_command_signal.emit(command, request)

    @QtCore.pyqtSlot(object, object)
    def _origin_saved(self, record, frame):
        if self._origin_experiment_details() != record["experiment"]:
            self.classic_patching_tab.set_origin_status(
                "Origin recorded for the previous experiment; current origin unchanged.", error=True)
            return
        self.patch_command_signal.emit(self.patch_interface.accept_origin, record)
        self.experiment_book_tab.add_origin_entry(record, frame)
        self.classic_patching_tab.set_origin_status(
            record["axis"].upper() + " origin saved to Experiment Book.")

    @QtCore.pyqtSlot(int, object)
    def _origin_task_finished(self, exit_code, result):
        if not self._origin_busy:
            return
        self._origin_busy = False
        self.classic_patching_tab.set_origin_busy(False)
        if exit_code:
            self.classic_patching_tab.set_origin_status(
                "Origin not saved; see the task error for details.", error=True)

    def register_commands(self):
        """
        Register GUI mouse and keyboard actions to patching interface commands.
        Overrides parent method to include patch-specific actions.
        """
        super(PatchGui, self).register_commands()
        # self.register_mouse_action(Qt.LeftButton, Qt.ShiftModifier,
        #                            self.active_patch_interface.patch_with_move)
        self.register_mouse_action(Qt.LeftButton, Qt.NoModifier,
                                   self.active_patch_interface.add_cell)
        self.register_mouse_action(Qt.RightButton, Qt.ShiftModifier,
                                   self.active_patch_interface.handle_corner_right_click)
        self.register_key_action(Qt.Key_B, None,
                                 self.active_patch_interface.break_in)
        self.register_key_action(Qt.Key_F2, None,
                                 self.active_patch_interface.store_cleaning_position)
        self.register_key_action(Qt.Key_F3, None,
                                 self.active_patch_interface.store_rinsing_position)
        self.register_key_action(Qt.Key_F4, None,
                                 self.active_patch_interface.clean_pipette)

    def toggle_cell_list_window(self, checked=None):
        """
        Toggle the visibility of the CellListWindow.

        Args:
            checked (bool, optional): If True, shows the window; if False, hides it. 
                If None, uses the current button state.
        """
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
        """
        Slot called when the CellListWindow is closed. Stops the update timer and
        resets the toggle button.
        """
        self._cell_list_timer.stop()
        if self.show_cells_button.isChecked():
            self.show_cells_button.blockSignals(True)
            self.show_cells_button.setChecked(False)
            self.show_cells_button.blockSignals(False)
        self.show_cells_button.setText("Show Cells")

    def _refresh_cell_list_window(self, force=False):
        """
        Update the CellListWindow with current cells from patch_interface.

        Args:
            force (bool, optional): If True, forces a full refresh regardless of previous signature.
        """
        if not self.cell_list_window.isVisible():
            return
        cells = list(self.active_patch_interface.cells_to_patch)
        try:
            stage_reference = self.active_patch_interface.current_autopatcher.calibrated_stage.reference_position()
        except Exception:
            stage_reference = None
        signature = tuple(id(cell) for cell in cells)
        full_refresh = force or (signature != self._cell_list_signature)
        self.cell_list_window.update_cells(cells, stage_reference, full_refresh=full_refresh)
        self._cell_list_signature = signature

    def toggle_pipette_status_window(self, checked=None):
        """
        Toggle the visibility of the PipetteStatusWindow.
    
        Args:
            checked (bool, optional): If True, shows the window; if False, hides it. 
                If None, uses the current button state.
        """
        if checked is None:
            checked = self.show_pipette_status_button.isChecked()
        if checked:
            self.show_pipette_status_button.setText("Close")
            self.pipette_status_window.show()
            self.pipette_status_window.raise_()
            self.pipette_status_window.activateWindow()
            self.pipette_status_timer.start()
        else:
            self.pipette_status_window.close()

    def _pipette_status_window_closed(self):
            """
            Slot called when the PipetteStatusWindow is closed. Stops the update timer and
            resets the toggle button.
            """
            self.pipette_status_timer.stop()
            if self.show_pipette_status_button.isChecked():
                self.show_pipette_status_button.blockSignals(True)
                self.show_pipette_status_button.setChecked(False)
                self.show_pipette_status_button.blockSignals(False)
            self.show_pipette_status_button.setText("Pipette Status")

    def toggle_dark_mode(self):
        """
        Toggle the dark mode for the GUI.
        """
        self.dark_mode = not self.dark_mode

        theme = "dark" if self.dark_mode else "light"
        icon_color = "white" if self.dark_mode else "black"

        self.apply_theme(theme)

        for box in self.findChildren(CollapsibleGroupBox):

            if self.dark_mode:
                if hasattr(box, "dark_style_sheet"):
                    box.setStyleSheet(
                        box.dark_style_sheet
                    )

            else:
                if hasattr(box, "default_style_sheet"):
                    box.setStyleSheet(
                        box.default_style_sheet
                    )

        for config_gui in self.config_tabs.findChildren(ConfigGui):

            if hasattr(config_gui, "save_button"):
                config_gui.save_button.setIcon(
                    qta.icon(
                        "fa.download",
                        color=icon_color,
                    )
                )

            if hasattr(config_gui, "load_button"):
                config_gui.load_button.setIcon(
                    qta.icon(
                        "fa.upload",
                        color=icon_color,
                    )
                )

        if self.dark_mode:
            self.task_progress.setStyleSheet(
                """
                QProgressBar {
                    background-color: #121212;
                    border: 1px solid #444444;
                    border-radius: 3px;
                    text-align: center;
                    color: white;
                }

                QProgressBar::chunk {
                    background-color: #0078D7;
                    border-radius: 2px;
                }
                """
            )

        else:
            self.task_progress.setStyleSheet("")

        self.task_abort_button.setIcon(
            qta.icon(
                "fa.ban",
                color=icon_color,
            )
        )

        self.task_success_button.setIcon(
            qta.icon(
                "fa.check",
                color=icon_color,
            )
        )

        self.help_button.setIcon(
            qta.icon(
                "fa.question-circle",
                color=icon_color,
            )
        )

        self.log_button.setIcon(
            qta.icon(
                "fa.file",
                color=icon_color,
            )
        )

        self.snap_image_button.setIcon(
            qta.icon(
                "fa.camera",
                color=icon_color,
            )
        )

        self.config_button.setIcon(
            qta.icon(
                "fa.cogs",
                color=icon_color,
            )
        )

        if (
            hasattr(self, "shared_rig_controls")
            and hasattr(
                self.shared_rig_controls,
                "record_button",
            )
        ):
            self.shared_rig_controls.record_button.setIcon(
                qta.icon(
                    "fa.video-camera",
                    color=icon_color,
                )
            )

    def apply_theme(self, theme: str):
        qdarktheme.setup_theme(
            theme=theme,
            corner_shape="rounded",
            custom_colors=PATCH_GUI_COLORS,
            additional_qss=PATCH_GUI_QSS,
        )

    def switch_active_pipette(self, id):
        """
        Switch the globally active pipette.

        Shared rig controls are unaffected.
        """
        if id not in self.pipette_interfaces:
            return

        self.active_pipette = (
            self.pipette_interfaces[id]
        )

        self.active_patch_interface = (
            self.patch_interfaces[id]
        )

        self.key_actions.clear()
        self.mouse_actions.clear()

        self.register_commands()

    def complete_task(self):
        """Overrides parent method to target the active patch interface."""
        if self.active_patch_interface is None:
            return
        self.task_success_button.setEnabled(False)
        self.active_patch_interface.complete_task()

    def abort_task(self):
        """Overrides parent method to target the active patch interface."""
        if self.active_patch_interface is None:
            return
        self.task_abort_button.setEnabled(False)
        self.task_success_button.setEnabled(False)
        self.active_patch_interface.abort_task()

    def toggle_atlas_window(self, checked=None):
        if checked is None:
            checked = self.show_atlas_button.isChecked()
        if checked:
            self.show_atlas_button.setText("Hide Atlas")
            self.atlas_window.show()
            self.atlas_window.raise_()
            self.atlas_window.activateWindow()
        else:
            self.atlas_window.close()

    def _atlas_window_closed(self, _result=None):
        self.show_atlas_button.setChecked(False)
        self.show_atlas_button.setText("Show Atlas")

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
    """A QGroupBox subclass with collapsible content area and custom styling."""
    def __init__(self, title="", parent=None):
        """
        Initialize a collapsible group box.

        Args:
            title (str, optional): The title text of the collapsible group. Defaults to "".
            parent (QWidget, optional): Parent widget. Defaults to None.
        """
        super(CollapsibleGroupBox, self).__init__(parent)
        self.setTitle("")  # Set the group box title to be blank to allow custom styling

        # Apply styles for rounded corners, grey borders, and consistent font
        self.default_style_sheet = ("""
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
                background-color: white;     /* white for buttons */
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

        # Apply styles for dark mode
        self.dark_style_sheet = ("""
            QGroupBox {
                border: 1px white;  /* White border */
                border-radius: 8px;           /* Rounded corners with 8px radius */
                margin-top: 6px;             /* Adjust top margin for visual separation */
                font-family: Arial, Helvetica, sans-serif;  /* Consistent font family */
                font-size: 14px;              /* Consistent font size for the group box */
                color: white
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
                font-weight: bold;            /* Bold for the group box title */
                color: white
            }
            QWidget {
                background-color: black;    /* Black background for the content area */
                border-radius: 8px;
                font-family: Arial, Helvetica, sans-serif;  /* Consistent font family */
                font-size: 14px;              /* Consistent font size for content area */
                color: white
            }
            QPushButton {
                background-color: black;     /* black for buttons */
                border: 1px solid white;   /* white border for buttons */
                border-radius: 6px;            /* Slightly rounded corners for buttons */
                padding: 3px;                  /* Padding for a better button look */
                font-family: Arial, Helvetica, sans-serif;  /* Consistent font family */
                font-size: 14px;               /* Adjusted font size for buttons */
                outline: none;                 /* Remove default focus outline */
                color: white;
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
        self.setStyleSheet(self.default_style_sheet)

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
        """
        Slot triggered by toggle_button click to show or hide the content area.
        """
        if self.toggle_button.isChecked():
            self.content_area.show()
            self.toggle_button.setArrowType(Qt.DownArrow)
        else:
            self.content_area.hide()
            self.toggle_button.setArrowType(Qt.RightArrow)

    def setContentLayout(self, layout):
        """
        Set the layout for the collapsible content area.

        Args:
            layout (QtWidgets.QLayout): The layout to set in the content area.
        """
        # Remove existing layout if any
        while self.content_layout.count():
            child = self.content_layout.takeAt(0)
            if child.widget():
                child.widget().setParent(None)
        self.content_layout.addLayout(layout)

    def update_theme(self):
        if not self.dark_mode:
            self.dark_mode = True
            self.setStyleSheet(self.dark_style_sheet)
        else:
            self.dark_mode = False
            self.setStyleSheet(self.dark_style_sheet)

class PipetteStackTab(QtWidgets.QWidget):
    """
    Displays one pipette-specific widget at a time.

    Each pipette widget is constructed once and stored in a
    QStackedWidget. The selector only changes visibility.
    """

    pipette_changed = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.widgets = {}
        self.index_by_id = {}

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        # -------------------------
        # Pipette selector
        # -------------------------

        selector_layout = QtWidgets.QHBoxLayout()

        selector_layout.addWidget(
            QtWidgets.QLabel("Pipette:")
        )

        self.selector = NoWheelComboBox()

        selector_layout.addWidget(
            self.selector
        )

        selector_layout.addStretch()

        layout.addLayout(
            selector_layout
        )

        # -------------------------
        # Pipette-specific contents
        # -------------------------

        self.stack = QtWidgets.QStackedWidget()

        layout.addWidget(
            self.stack,
            1,
        )

        self.selector.currentIndexChanged.connect(
            self._selection_changed
        )

    def add_pipette_widget(
        self,
        pipette_id,
        widget,
    ):
        pipette_id = str(pipette_id)

        if pipette_id in self.widgets:
            raise ValueError(
                f"Duplicate pipette ID: {pipette_id}"
            )

        index = self.stack.addWidget(widget)

        self.widgets[pipette_id] = widget
        self.index_by_id[pipette_id] = index

        self.selector.addItem(
            pipette_id,
            pipette_id,
        )

        if len(self.widgets) == 1:
            self.stack.setCurrentIndex(index)

    def _selection_changed(self, selector_index):
        pipette_id = self.selector.itemData(
            selector_index
        )

        if pipette_id is None:
            return

        pipette_id = str(pipette_id)

        stack_index = self.index_by_id.get(
            pipette_id
        )

        if stack_index is None:
            return

        self.stack.setCurrentIndex(
            stack_index
        )

        self.pipette_changed.emit(
            pipette_id
        )

    def set_pipette(self, pipette_id):
        pipette_id = str(pipette_id)

        index = self.selector.findData(
            pipette_id
        )

        if index < 0:
            return False

        self.selector.setCurrentIndex(index)
        return True

    def current_pipette_id(self):
        return self.selector.currentData()


class PipetteStatusWindow(QtWidgets.QWidget):
    """A collapsible widget that displays the status of all connected pipettes."""
    closed = QtCore.pyqtSignal()
    
    def __init__(self, interfaces, parent=None):
        super().__init__(parent=parent)
        self.interfaces = interfaces
        
        self.setWindowFlags(Qt.Window)
        self.setWindowTitle("Pipette Status")
        self.resize(800, 400)

        self.layout = QtWidgets.QVBoxLayout(self)
        
        
        # 2. The Status Table (Hidden by default)
        self.table = QtWidgets.QTableWidget(len(self.interfaces), 4)
        self.table.setHorizontalHeaderLabels(["Pipette", "Status", "Action", "Last Updated"])
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        
        # Populate initial rows
        for row, (p_id, interface) in enumerate(self.interfaces.items()):
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(p_id)))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem("Initializing..."))
            raw_timestamp = time.time()
            display_timestamp = datetime.fromtimestamp(raw_timestamp).strftime("%H:%M:%S")
            self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(display_timestamp))
            
            # Action buttons
            btn_widget = QtWidgets.QWidget()
            btn_layout = QtWidgets.QHBoxLayout(btn_widget)
            btn_layout.setContentsMargins(0, 0, 0, 0)
            
            abort_btn = QtWidgets.QPushButton("Abort")
            abort_btn.clicked.connect(interface.abort_task)
            btn_layout.addWidget(abort_btn)

            success_btn = QtWidgets.QPushButton("Complete")
            success_btn.clicked.connect(interface.complete_task)
            btn_layout.addWidget(success_btn)
            self.table.setCellWidget(row, 2, btn_widget)

        self.layout.addWidget(self.table)

        self.row_map = {} # ADD THIS
        for row, (p_id, interface) in enumerate(self.interfaces.items()):
            self.row_map[p_id] = row
        
        # self.poll_timer = QtCore.QTimer(self)
        # self.poll_timer.timeout.connect(self.update_status)
        # self.poll_timer.start(500)
        
        # # Start open and polling
        # self.poll_timer.start(500)
        # self.update_status()

    def closeEvent(self, event):
            """
            Overridden close event to emit the 'closed' signal.
    
            Args:
                event (QCloseEvent): Close event.
            """
            self.closed.emit()
            super().closeEvent(event)

    def update_status(self):
        """Polls the global interfaces and updates the table."""
        for p_id, interface in self.interfaces.items():
            if p_id not in self.row_map:
                continue
            row = self.row_map[p_id]

            status_msg = getattr(interface, 'last_status_msg', "Awaiting command")
            error_msg = getattr(interface, 'last_error_msg', None)
            warning_msg = getattr(interface, 'last_warning_msg', None)
            raw_timestamp = getattr(interface, 'latest_log_time', None)
            
            if error_msg:
                display_text = f"ERROR: {error_msg}"
            elif status_msg:
                display_text = str(status_msg)
            elif warning_msg:
                display_text = f"WARNING: {warning_msg}"
            else:
                display_text = "Awaiting command"
            
            item = self.table.item(row, 1)
            if item:
                if error_msg:
                    item.setForeground(QtGui.QBrush(QtCore.Qt.red))
                elif warning_msg:
                    item.setForeground(QtGui.QBrush(QtCore.Qt.darkYellow))
                else:
                    item.setData(QtCore.Qt.ForegroundRole, None)
                item.setText(display_text)

            time_item = self.table.item(row, 3)
            if time_item and raw_timestamp:
                display_timestamp = datetime.fromtimestamp(raw_timestamp).strftime("%H:%M:%S")
                time_item.setText(display_timestamp)

class CellListWindow(QtWidgets.QDialog):
    """Dialog window displaying a list of selected cells with images and stage positions."""
    closed = QtCore.pyqtSignal()

    def __init__(self, parent=None, thumbnail_size=96):
        """
        Initialize a cell list window.

        Args:
            parent (QWidget, optional): Parent widget. Defaults to None.
            thumbnail_size (int, optional): Size of cell image thumbnails in pixels. Defaults to 96.
        """
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
        """
        Overridden close event to emit the 'closed' signal.

        Args:
            event (QCloseEvent): Close event.
        """
        self.closed.emit()
        super().closeEvent(event)

    def update_cells(self, cells, stage_reference=None, full_refresh=True):
        """
        Update the table with current cell information.

        Args:
            cells (list): List of cells to display.
            stage_reference (optional): Reference stage position.
            full_refresh (bool, optional): Whether to force full update. Defaults to True.
        """
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
        """
        Unpack a cell tuple into stage positions and images.

        Args:
            cell (tuple or None): Cell data tuple.

        Returns:
            tuple: (stage_px, img, stage_um, img_fluo)
        """
        if cell is None:
            return None, None, None, None
        if len(cell) >= 4:
            return cell[0], cell[1], cell[2], cell[3]
        if len(cell) == 3:
            return cell[0], cell[1], cell[2], None
        return None, None, None, None

    def _set_item(self, row, col, text):
        """
        Set a text item in the table at specified row and column.

        Args:
            row (int): Row index.
            col (int): Column index.
            text (str): Text to set.
        """
        item = self.table.item(row, col)
        if item is None:
            item = QtWidgets.QTableWidgetItem()
            item.setFlags(item.flags() ^ Qt.ItemIsEditable)
            self.table.setItem(row, col, item)
        item.setText(text)

    def _set_image_cell(self, row, col, image, empty_text=""):
        """
        Set a cell widget in the table with an image or placeholder text.

        Args:
            row (int): Row index.
            col (int): Column index.
            image (ndarray or None): Image to display.
            empty_text (str, optional): Text if image is None. Defaults to "".
        """
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
        """
        Convert a NumPy image array to QPixmap for display.

        Args:
            image (ndarray): Input image.

        Returns:
            QPixmap or None: Pixmap to display, or None if image is invalid.
        """
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
        """
        Normalize a NumPy image to 8-bit range [0, 255].

        Args:
            img (ndarray): Input image.

        Returns:
            ndarray: 8-bit normalized image.
        """
        img = img.astype(np.float32)
        min_val = float(np.min(img))
        max_val = float(np.max(img))
        if max_val > min_val:
            img = (img - min_val) / (max_val - min_val) * 255.0
        else:
            img = np.zeros_like(img, dtype=np.float32)
        return img.astype(np.uint8)

    def _format_vec(self, vec):
        """
        Format a numeric vector for display in the table.

        Args:
            vec (array-like or None): Vector to format.

        Returns:
            str: Comma-separated formatted string or "N/A".
        """
        if vec is None:
            return "N/A"
        arr = np.array(vec).astype(float).ravel()
        return ", ".join(f"{v:.1f}" for v in arr)

class ButtonTabWidget(QtWidgets.QWidget):
    """
    A QWidget subclass for organizing buttons, position displays, and sequential command execution 
    in a GUI. Supports collapsible sections, dynamic button styling, and periodic updates of position labels.
    """
    def __init__(self):
        """
        Initialize a button tab widget.
        """
        super().__init__()
        self.pos_update_timers = []
        self.pos_labels = []
        self.interface_signals = {}
        self.start_task = None
        self.section_buttons = {}  # Dictionary to store buttons by section
        self.section_button_map = {}  # section -> {button_name: button}
        self.active_buttons_by_section = {}  # section -> set(button_name)
        self.section_button_map = {}  # section -> {button_name: button}
        self.active_buttons_by_section = {}  # section -> set(button_name)
        self.color_change_sections = []  # Sections that should change color on completion
        self.section_colors = {}  # Store custom colors for different sections


    def do_nothing(self):
        """Dummy function for buttons that are not yet implemented."""
        pass  # a dummy function for buttons that aren't implemented yet
    
    def run_sequential_commands(self, cmds, button=None, section=None, button_name=None, repeat=1):
        """
        Executes a list of commands sequentially, handling both synchronous and asynchronous commands.

        Args:
            cmds (list or callable): Commands to execute sequentially. Can be nested lists.
            button (QPushButton, optional): The button that triggered the commands.
            section (str, optional): The section name for styling and color logic.
            button_name (str, optional): The name of the button triggering the commands.
        """
        # Ensure cmds is a list
        if not isinstance(cmds, list):
            cmds = [cmds]
        else:
            cmds = self._flatten_sequential_cmds(cmds)
        try:
            repeat_count = max(1, int(repeat))
        except (TypeError, ValueError):
            repeat_count = 1
        if repeat_count > 1:
            cmds = cmds * repeat_count
            
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
        """
        Recursively flattens nested lists of commands into a single list.

        Args:
            cmds (list): A (possibly nested) list of commands.

        Returns:
            list: Flattened list of commands.
        """
        flat_cmds = []
        for cmd in cmds:
            if isinstance(cmd, list):
                flat_cmds.extend(self._flatten_sequential_cmds(cmd))
            else:
                flat_cmds.append(cmd)
        return flat_cmds

    def _reset_section_button_colors(self, section):
        """
        Reset colors for all buttons in a section
        
        Args:
            section (str): The section whose buttons should be reset.
        """
        if section in self.section_buttons:
            for button_info in self.section_buttons[section]:
                button = button_info[0]
                button.setStyleSheet("")  # This will revert to the style from CollapsibleGroupBox

    def _run_next_seq_command(self):
        """
        Executes the next command in a stored sequential command list, handling asynchronous completion
        signals and updating button styles.
        """
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


    def run_command(self, cmds, repeat=1):
        try:
            repeat_count = max(1, int(repeat))
        except (TypeError, ValueError):
            repeat_count = 1

        for _ in range(repeat_count):
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
        """
        Executes a single command, handling asynchronous commands with task_description attribute.

        Args:
            cmd (callable): Command to execute.
        """
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
        """
        Adds a collapsible box displaying position labels for each axis, with optional tare button.

        Args:
            name (str): Title of the box.
            layout (QLayout): Parent layout to add the box to.
            update_func (callable): Function to update position labels, accepts list of label indices.
            tare_func (callable, optional): Function to tare the manipulator.
            axes (list of str, optional): Axes to display. Defaults to ['x', 'y', 'z'].
        """
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
        """
        Adds a collapsible box displaying individual axis positions with separate tare buttons per axis.

        Args:
            name (str): Title of the box.
            layout (QLayout): Parent layout to add the box to.
            update_func (callable): Function to update position labels, accepts list of label indices.
            tare_funcs (list of callables): Tare functions, one per axis.
            axes (list of str, optional): Axes to display. Defaults to ['x', 'y', 'z'].
        """
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
                    cmds, freq=None, sequential=False, change_color_on_complete=False, 
                    completion_color="rgba(0, 0, 255, 0.3)",
                    change_color_during=None):
        """
        Adds a collapsible box containing a list of buttons arranged in rows, with optional sequential execution
        and color-change behavior on completion or during execution.

        Args:
            box_name (str): Title of the collapsible section.
            layout (QVBoxLayout): Parent layout to add the box to.
            buttonNames (list of list of str): Names of buttons arranged by rows.
            cmds (list or list of list of callables): Commands corresponding to each button.
            sequential (bool, optional): Whether to run commands sequentially. Defaults to False.
            change_color_on_complete (bool, optional): Whether to change button color when commands complete. Defaults to False.
            completion_color (str, optional): Color to apply on completion. Defaults to blue overlay.
            change_color_during (list or bool, optional): Button names to style during execution, or True for all.

        Returns:
            list: List of button tuples for the section.
        """
        completion_color=("rgba(0, 0, 255, 0.3)", change_color_during=None, extra_widget=None):
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
                button.setMinimumWidth(50)
                button.setMinimumHeight(30)

                button_freq = 1
                if freq is not None and i < len(freq) and j < len(freq[i]):
                    try:
                        button_freq = max(1, int(freq[i][j]))
                    except (TypeError, ValueError):
                        button_freq = 1
                
                # Track this button for this section
                section_buttons.append((button, i, j, button_name))
                self.section_button_map.setdefault(box_name, {})[button_name] = button
                self.section_button_map.setdefault(box_name, {})[button_name] = button

                # Use a lambda function with default arguments to correctly capture the command
                if i < len(cmds) and j < len(cmds[i]):
                    button_cmd = cmds[i][j]
                    if sequential:
                        button.clicked.connect(lambda state, cmd=button_cmd, btn=button, section=box_name, 
                                            name=button_name, repeat=button_freq: self.run_sequential_commands(cmd, btn, section, name, repeat))
                    else:
                        button.clicked.connect(lambda state, cmd=button_cmd, repeat=button_freq: self.run_command(cmd, repeat))
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

        if extra_widget is not None:
            rows.addWidget(extra_widget)
        box.setContentLayout(rows)
        layout.addWidget(box)
        return section_buttons

    def get_section_button(self, section: str, name: str):
        """
        Retrieves a QPushButton object by section and button name.

        Args:
            section (str): Section name.
            name (str): Button name.

        Returns:
            QPushButton or None: The button object if found, else None.
        """
        return self.section_button_map.get(section, {}).get(name)        

class FileSelector(QWidget):
    """A widget that provides a file selection dialog and emits the selected file path."""
    fileSelected = pyqtSignal(str)  # Signal to emit the selected file path

    def __init__(self):
        """Initializes the FileSelector widget."""
        super().__init__()

    def open_file_dialog(self):
        """Opens a file dialog for selecting a CSV file and emits the selected file path."""
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
    """
    GUI widget that provides grouped controls for calibration, movement,
    testing, lighting, patching, and recording in an automated patch-clamp system.
    """
    def __init__(self, patch_interface: AutoPatchInterface, pipette_interface: PipetteInterface, start_task, interface_signals, recording_state_manager: RecordingStateManager, movement_recorder, shared_rig_controls):
        """
        Initializes the ClassicPatchButtons GUI and sets up all control sections.

        Args:
            patch_interface (AutoPatchInterface): Interface for patching operations.
            pipette_interface (PipetteInterface): Interface for pipette control.
            start_task (callable): Function to start tasks.
            interface_signals (dict): Signals for interfacing with controllers.
            recording_state_manager (RecordingStateManager): Recording state manager.
        """
    origin_requested = QtCore.pyqtSignal(str)

    def __init__(self, patch_interface: AutoPatchInterface, pipette_interface: PipetteInterface, start_task, interface_signals, recording_state_manager: RecordingStateManager):
        super().__init__()
        self.patch_interface = patch_interface
        self.pipette_interface = pipette_interface

        self.start_task = start_task

        self.interface_signals = interface_signals

        self.recording_state_manager = recording_state_manager

        self.shared_rig_controls = shared_rig_controls

        layout = QtWidgets.QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)

        self.pipette_xyz = [0, 0, 0]
        self.tare_pipette_pos = [0, 0, 0]

        self.file_selector = FileSelector()


        self.recorder = movement_recorder

        self.addPositionBox(
            'pipette position (um)',
            layout,
            self.update_pipette_pos_labels,
            tare_func=self.tare_pipette
        )

        self.pipette_calibration = [self.pipette_interface.calibrate_manipulator, self.patch_interface.store_calibration_positions, self.patch_interface.move_to_safe_space]
        self.pipette_calibration_no_move = [self.pipette_interface.calibrate_manipulator, self.patch_interface.store_calibration_positions]
        self.pipette_cleaning_calibration = [self.patch_interface.store_cleaning_position,self.patch_interface.move_pipette_up,self.patch_interface.move_to_safe_space]


        origin_controls = QtWidgets.QWidget()
        origin_layout = QtWidgets.QVBoxLayout(origin_controls)
        origin_layout.setContentsMargins(0, 0, 0, 0)
        origin_row = QtWidgets.QHBoxLayout()
        self.origin_buttons = {}
        for axis in ("x", "y"):
            button = QtWidgets.QPushButton(axis.upper() + " Origin")
            button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            button.setMinimumSize(50, 50)
            button.clicked.connect(lambda checked, a=axis: self.origin_requested.emit(a))
            self.origin_buttons[axis] = button
            origin_row.addWidget(button)
        origin_layout.addLayout(origin_row)
        self.origin_status = QtWidgets.QLabel()
        self.origin_status.setWordWrap(True)
        self.origin_status.hide()
        origin_layout.addWidget(self.origin_status)

        # Add a box for calibration setup
        # buttonList = [['Calibrate Stage','Calibrate Pipette'],['set home space','set safe space'],['Store Cleaning Position'],['Clear Calibration']]
        buttonList = [['Calibrate Stage','Calibrate Pipette'],['Store Cleaning Position'],['Load Calibration','Clear Calibration']]
        cmds = [[self.stage_calibration, self.pipette_calibration],
                # [self.patch_interface.store_home_position, self.patch_interface.store_safe_position],
                [self.pipette_cleaning_calibration],
                [self.load_calibration, self.patch_interface.clear_positions]
        ]
        self.addButtonList('calibration', layout, buttonList, cmds, sequential=True, 
                        change_color_on_complete=True, completion_color="rgba(173, 216, 230, 0.5)",
                        extra_widget=origin_controls)

        # Add a box for movement commands 
        buttonList = [
            ['Move to Safe Position','Move to Home Position'],
            ['Move to cell plane','Focus Stage'],
            ['Store corners', 'Start Scan'],
            ['Move group up', 'Move group down'],
            ['Center Pipette','Clean pipette','Focus Pipette'],
        ]
        cmds = [
            [self.patch_interface.move_to_safe_space, self.patch_interface.move_to_home_space],

            [self.pipette_interface.go_to_floor,self.pipette_interface.focus_stage],
            [
                "Store Cleaning Position",
                "Clear Calibration",
            ],
        ]

        cmds = [
            [
                self.pipette_calibration,
            ],
            [
                [
                    self.pipette_cleaning_calibration
                ],
                [
                    self.load_calibration,
                    self.patch_interface.clear_positions,
                ],
            ],
        ]

        self.addButtonList(
            "Pipette Calibration",
            layout,
            buttonList,
            cmds,
            sequential=True,
            change_color_on_complete=True,
            completion_color="rgba(173, 216, 230, 0.5)",
        )

        # --------------------------------------------------
        # Coordinated Rig Movement
        #
        # Selected pipette matters, but these actions may
        # also move or depend on the shared stage.
        # --------------------------------------------------

        buttonList = [
            [
                "Move to Safe Position",
                "Move to Home Position",
            ],
            [
                "Store Corners",
                "Start Scan",
            ],
            [
                "Move Group Up",
                "Move Group Down",
            ],
        ]

        cmds = [
            [
                self.patch_interface.move_to_safe_space,
                self.patch_interface.move_to_home_space,
            ],
            [
                self.patch_interface.start_selecting_corners,
                [[
                    self.patch_interface.move_to_scan_start,
                    self.shared_rig_controls.start_recording,
                    self.patch_interface.start_scan,
                    self.shared_rig_controls.stop_recording,
                ]],
            ],
            [
                self.patch_interface.move_group_up,
                self.patch_interface.move_group_down,
            ],
        ]

        self.addButtonList(
            "Coordinated Rig Movement",
            layout,
            buttonList,
            cmds,
            sequential=True,
        )

        # --------------------------------------------------
        # Pipette-only movement
        # --------------------------------------------------

        buttonList = [
            [
                "Center Pipette",
                "Clean Pipette",
                "Focus Pipette",
            ],
        ]

        cmds = [
            [
                self.pipette_interface.center_pipette,
                self.patch_interface.clean_pipette,
                self.pipette_interface.focus_pipette,
            ],
        ]

        self.addButtonList(
            "Pipette Movement",
            layout,
            buttonList,
            cmds,
            sequential=True,
        )

        # Find Pipette path
        self.pipette_location = [self.pipette_interface.center_pipette,self.pipette_interface.focus_pipette,self.pipette_interface.move_pipette_random,self.patch_interface.find_pipette]
        # add a box for testing controllability of the pipette and stage
        buttonList = [['Find Pipette','Test Pipette Movement']]
        cmds = [[self.pipette_location,self.pipette_interface.move_pipette_random_velocity]]
        freq = [[1, 3]]
        self.addButtonList('Testing', layout, buttonList, cmds, freq=freq, sequential=True)
        buttonList = [['Find Pipette','Test Pipette Movement'],
                      ['Detect Pipette','Detect Cells']]
        cmds = [[self.pipette_location,self.pipette_interface.move_pipette_random_velocity],
                [self.pipette_interface.detect_pipette, self.patch_interface.detect_cells]]
        freq = [[1, 3],
                [1, 1]]
        self.addButtonList('testing', layout, buttonList, cmds, freq=freq, sequential=True)

        # # Add a box for lamp commands
        buttonList = [['toggle shutter', 'toggle fluorescense'],['move cube left','move cube right']]
        cmds = [[ self.patch_interface.toggle_shutter, self.patch_interface.toggle_fluorescence],
                [self.patch_interface.move_cube_left, self.patch_interface.move_cube_right]
        ]
        self.addButtonList('fluorescence', layout, buttonList, cmds, sequential=True)

        # Add a box for patching commands
        buttonList = [['Select Cell','Remove Last Cell','Center on Cell'],
                      ['Locate Cell','Approach Cell','Hunt Cell'],
                      ['Gigaseal','Break-in','Escape Cell'],
                      ['Patch Cell','Run Protocols']]
        cmds = [[self.patch_interface.start_selecting_cells, self.patch_interface.remove_last_cell, self.patch_interface.center_on_cell],
                [self.patch_interface.locate_cell,
                 [self.shared_rig_controls.start_recording,self.patch_interface.hunt_cell],[self.patch_interface.gigaseal]],
                [[ self.patch_interface.break_in],
                 [self.shared_rig_controls.stop_recording,  self.patch_interface.escape_cell]],
                [[self.shared_rig_controls.start_recording,  self.patch_interface.patch, self.shared_rig_controls.stop_recording],
                 [self.shared_rig_controls.stop_recording,  self.patch_interface.run_protocols]]
                ]
        self.addButtonList('Patching', layout, buttonList, cmds, sequential=True, change_color_during={
                [self.patch_interface.locate_cell, self.patch_interface.approach_cell,
                 [self.start_recording,self.patch_interface.hunt_cell]],
                [[self.patch_interface.gigaseal], [self.patch_interface.break_in],
                 [self.stop_recording,  self.patch_interface.escape_cell]],
                [[self.start_recording,  self.patch_interface.patch, self.stop_recording],
                 [self.stop_recording,  self.patch_interface.run_protocols]]
]
        self.addButtonList('patching', layout, buttonList, cmds, sequential=True, change_color_during={
            'Locate Cell',
            'Approach Cell',
            'Hunt Cell',
            'Gigaseal',
            'Break-in',
            'Escape Cell',
            'Run Protocols',
        })

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
    def set_origin_busy(self, busy):
        for button in self.origin_buttons.values():
            button.setEnabled(not busy)

    def set_origin_status(self, message, error=False):
        self.origin_status.setText(message)
        self.origin_status.setVisible(error)
        for button in self.origin_buttons.values():
            button.setToolTip(message)

    def load_calibration(self):
        """Opens a file dialog and connects selection to calibration loading."""
        self.file_selector.fileSelected.connect(self.load_calibration_file)  # Connect the signal to the slot
        self.file_selector.open_file_dialog()  # Open the file dialog


    def load_calibration_file(self, file_path):
        """
        Loads a calibration file into the pipette interface.

        Args:
            file_path (str): Path to the calibration file.
        """
        # call pipette.interface.read_calibration
        logging.info(f"Loading calibration file: {file_path}")
        self.pipette_interface.read_calibration(file_path)


    def test_movement(self):
        """Opens a movement file for testing. Starts recording if not already enabled."""
        # check if recording is enabled
        if self.recording_state_manager.is_recording_enabled():
            # Opens the file selector dialog without blocking the main thread
            self.file_selector.fileSelected.connect(self.load_movement_file)  # Connect the signal to the slot
            self.file_selector.open_file_dialog()
        else:
            # if not recording then start recording
            self.shared_rig_controls.start_recording()
            # Opens the file selector dialog without blocking the main thread
            self.file_selector.open_file_dialog()
        
    def load_movement_file(self, file_path):
        """
        Loads a movement file and sends it to the patch interface.

        Args:
            file_path (str): Path to the movement file.
        """
        logging.info(f"Loading movement file: {file_path}")
        # # send file to the pipette interface
        self.patch_interface.send_movement_file(file_path)

    def _update_cell_sorter_led_button_style(self, enabled: bool):
        """
        Updates the visual style of the LED toggle button.

        Args:
            enabled (bool): Whether the LED is enabled.
        """
        if self.cell_sorter_led_button is None:
            return
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
        if self.cell_sorter_led_button is not None and self.cell_sorter_led_button.isChecked() != enabled:
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
        if self.cell_sorter_led_button is None:
            return
        enabled = self.cell_sorter_led_button.isChecked() if checked is None else bool(checked)
        self._set_cell_sorter_led_state(enabled)



    def close(self):
        """Closes the widget and releases recorder resources."""
        super(ClassicPatchButtons, self).close()

    def closeEvent(self, event):
        """
        Handles the widget close event and ensures recorder cleanup.

        Args:
            event (QCloseEvent): Close event.
        """
        super(ClassicPatchButtons, self).closeEvent(event)

    def tare_pipette(self):
        """Sets the current pipette position as the zero reference."""
        currPos = self.pipette_interface.calibrated_unit.unit.position()
        self.tare_pipette_pos = currPos
        self.pipette_interface.tare_pipette = np.array(self.tare_pipette_pos)
        print("Tare pipette: ", self.tare_pipette_pos)
        self.pipette_interface.write_tare()

    def update_pipette_pos_labels(self, indices):
        """
        Updates pipette position labels and logs movement data if recording.

        Args:
            indices (list[int]): Indices of label widgets to update.
        """
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
                self.shared_rig_controls.stage_xy[0],
                self.shared_rig_controls.stage_xy[1],
                self.shared_rig_controls.stage_z,
                recPos[0],
                recPos[1],
                recPos[2],
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
        z_scale = self.pipette_interface.calibrated_unit.config.microscope_units_per_um
        self.pipette_interface.tare_stage[2] = zPos / z_scale
        print("Tare stage z: ", self.currz_stage_pos)
        self.pipette_interface.write_tare()

    def update_stage_pos_labels(self, indices):
        """
        Updates stage position labels relative to tare values.

        Args:
            indices (list[int]): Indices of label widgets to update.
        """
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
                z_scale = self.pipette_interface.calibrated_unit.config.microscope_units_per_um
                label.setText(f'{label.text().split(":")[0]}: {zPos / z_scale:.2f}')

class SharedRigControls(ButtonTabWidget):
    """
    Controls hardware/state shared by the entire rig.

    Owns:
        - stage position display / stage tare
        - fluorescence controls
        - global recording state

    Does not depend on the currently selected pipette.
    """

    def __init__(
        self,
        pipette_interfaces,
        patch_interfaces,
        start_task,
        interface_signals,
        recording_state_manager,
        movement_recorder,
        graph_recorder,
    ):
        super().__init__()

        if not isinstance(pipette_interfaces, dict):
            pipette_interfaces = {
                "pipette": pipette_interfaces
            }

        if not isinstance(patch_interfaces, dict):
            patch_interfaces = {
                next(iter(pipette_interfaces)): patch_interfaces
            }

        self.pipette_interfaces = pipette_interfaces
        self.patch_interfaces = patch_interfaces

        # All PipetteInterfaces reference the same physical stage/microscope.
        self.stage_interface = next(
            iter(self.pipette_interfaces.values())
        )

        # All AutoPatchInterfaces reference the same fluorescence hardware.
        self.shared_patch_interface = next(
            iter(self.patch_interfaces.values())
        )

        self.start_task = start_task
        self.interface_signals = interface_signals

        self.recording_state_manager = (
            recording_state_manager
        )
        self.movement_recorder = movement_recorder
        self.graph_recorder = graph_recorder

        self.stage_xy = [0, 0]
        self.stage_z = 0

        self.currx_stage_pos = [0, 0, 0]
        self.curry_stage_pos = [0, 0, 0]
        self.currz_stage_pos = [0, 0, 0]

        layout = QtWidgets.QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)

        title = QtWidgets.QLabel("Shared Rig Controls")
        title.setStyleSheet(
            "font-weight: bold; font-size: 14px;"
        )
        layout.addWidget(title)

        # ---------------------------------
        # Shared stage position
        # ---------------------------------

        self.positionAndTareBox(
            "Stage Position (um)",
            layout,
            self.update_stage_pos_labels,
            tare_funcs=[
                self.tare_stage_x,
                self.tare_stage_y,
                self.tare_stage_z,
            ],
        )

        # --------------------------------------------------
        # Shared Stage Actions
        # --------------------------------------------------

        self.stage_calibration = [
            self.stage_interface.set_floor,
            self.stage_interface.calibrate_stage,
            lambda: self.stage_interface.move_microscope(
                float(
                    self.stage_interface
                    .calibrated_unit
                    .config
                    .home_position_delta_um
                )
            ),
        ]

        stage_buttons = [
            ["Calibrate Stage"],
            ["Move to Cell Plane", "Focus Stage"],
        ]

        stage_cmds = [
            [
                self.stage_calibration,
            ],
            [
                self.stage_interface.go_to_floor,
                self.stage_interface.focus_stage,
            ],
        ]

        self.addButtonList(
            "Stage Controls",
            layout,
            stage_buttons,
            stage_cmds,
            sequential=True,
            change_color_on_complete=True,
            completion_color="rgba(173, 216, 230, 0.5)",
        )

        # ---------------------------------
        # Shared fluorescence
        # ---------------------------------

        button_list = [
            ["Toggle Shutter", "Toggle Fluorescence"],
            ["Move Cube Left", "Move Cube Right"],
        ]

        commands = [
            [
                self.shared_patch_interface.toggle_shutter,
                self.shared_patch_interface.toggle_fluorescence,
            ],
            [
                self.shared_patch_interface.move_cube_left,
                self.shared_patch_interface.move_cube_right,
            ],
        ]

        self.addButtonList(
            "Fluorescence",
            layout,
            button_list,
            commands,
            sequential=True,
        )

        # ---------------------------------
        # Shared recording
        # ---------------------------------

        self.record_button = QtWidgets.QPushButton(
            "Start Recording"
        )

        self.record_button.clicked.connect(
            self.toggle_recording
        )

        self.record_button.setMinimumHeight(30)

        layout.addWidget(self.record_button)

        self.setLayout(layout)
        

    # =========================================================
    # Recording
    # =========================================================

    def toggle_recording(self):
        if self.recording_state_manager.is_recording_enabled():
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        if self.recording_state_manager.is_recording_enabled():
            return

        self.recording_state_manager.set_recording(True)

        self.record_button.setText(
            "Stop Recording"
        )

        self.record_button.setStyleSheet(
            "background-color: red; "
            "color: white; "
            "border-radius: 5px; "
            "padding: 5px;"
        )

        logging.info("Recording started")

    def stop_recording(self):
        if (
            self.recording_state_manager
            .is_recording_enabled()
        ):
            self.recording_state_manager.set_recording(
                False
            )

            if self.graph_recorder is not None:
                self.graph_recorder.handle_recording_stopped()

            if self.movement_recorder is not None:
                self.movement_recorder.handle_recording_stopped()

        self.record_button.setText(
            "Start Recording"
        )

        self.record_button.setStyleSheet("")

        logging.info(
            "Recording stopped"
        )

    # =========================================================
    # Shared stage position
    # =========================================================

    def update_stage_pos_labels(self, indices):
        xy_rec_pos = (
            self.stage_interface.calibrated_stage.position()
        )

        z_rec_pos = (
            self.stage_interface.microscope.position()
        )

        xy_pos = (
            xy_rec_pos
            - self.currx_stage_pos[0:2]
            - self.curry_stage_pos[0:2]
        )

        z_pos = (
            z_rec_pos
            - self.currz_stage_pos[2]
        )

        # Raw stage coordinates used by movement logging.
        self.stage_xy = xy_rec_pos
        self.stage_z = z_rec_pos

        for i, ind in enumerate(indices):
            label = self.pos_labels[ind]
            name = label.text().split(":")[0]

            if i < 2:
                label.setText(
                    f"{name}: {xy_pos[i]:.2f}"
                )
            else:
                z_scale = (
                    self.stage_interface
                    .calibrated_unit
                    .config
                    .microscope_units_per_um
                )

                label.setText(
                    f"{name}: {z_pos / z_scale:.2f}"
                )

    def tare_stage_x(self):
        x_pos = (
            self.stage_interface
            .calibrated_stage
            .position(0)
        )

        self.currx_stage_pos = [
            x_pos,
            0,
            0,
        ]

        self._set_stage_tare_for_all(
            axis=0,
            value=x_pos,
        )

    def tare_stage_y(self):
        y_pos = (
            self.stage_interface
            .calibrated_stage
            .position(1)
        )

        self.curry_stage_pos = [
            0,
            y_pos,
            0,
        ]

        self._set_stage_tare_for_all(
            axis=1,
            value=y_pos,
        )

    def tare_stage_z(self):
        z_pos = (
            self.stage_interface
            .microscope
            .position()
        )

        self.currz_stage_pos = [
            0,
            0,
            z_pos,
        ]

        z_scale = (
            self.stage_interface
            .calibrated_unit
            .config
            .microscope_units_per_um
        )

        self._set_stage_tare_for_all(
            axis=2,
            value=z_pos / z_scale,
        )

    def _set_stage_tare_for_all(
        self,
        axis,
        value,
    ):
        """
        Keep every PipetteInterface synchronized with the
        shared physical stage tare.
        """
        for interface in self.pipette_interfaces.values():
            interface.tare_stage[axis] = value
            interface.write_tare()

        logging.info(
            "Shared stage axis %s tared to %s",
            axis,
            value,
        )

class NoWheelComboBox(QtWidgets.QComboBox):
    """QComboBox that ignores mouse-wheel input."""

    def wheelEvent(self, event):
        event.ignore()


class NoWheelTabBar(QtWidgets.QTabBar):
    """QTabBar that ignores mouse-wheel input."""

    def wheelEvent(self, event):
        event.ignore()
