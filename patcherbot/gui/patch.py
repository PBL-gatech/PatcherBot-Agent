from __future__ import absolute_import

from types import MethodType

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal, QObject
import PyQt5.QtGui as QtGui
import numpy as np
import logging
import time

from PyQt5.QtWidgets import QFileDialog, QTabWidget, QWidget,QMessageBox
import qtawesome as qta

from patcherbot.controller import TaskController
from patcherbot.gui.experiment_book_tab import ExperimentBookTab
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

    def __init__(self, camera, aux_camera, pipette_interfaces, patch_interfaces, recording_state_manager: RecordingStateManager, with_tracking=False):
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
        super(PatchGui, self).__init__(camera, aux_camera, pipette_interfaces, with_tracking=with_tracking, recording_state_manager=recording_state_manager)

        self.setWindowTitle("Patch GUI")
        self.resize(1200, 1000)

        if not isinstance (pipette_interfaces, dict):
            self.pipette_interfaces = {"pipette": pipette_interfaces}
            self.patch_interfaces = {list(self.pipette_interfaces.keys())[0]: patch_interfaces}
        else:
            self.pipette_interfaces = pipette_interfaces
            self.patch_interfaces = patch_interfaces
        self.recording_state_manager = recording_state_manager

        self._cell_list_signature = None
        self.cell_list_window = CellListWindow(self)
        self.cell_list_window.closed.connect(self._cells_window_closed)
        self.show_cells_button = QtWidgets.QPushButton("Show Cells")
        self.show_cells_button.setCheckable(True)
        self.show_cells_button.clicked.connect(self.toggle_cell_list_window)
        # self.status_bar.insertPermanentWidget(1, self.show_cells_button)
        self._cell_list_timer = QtCore.QTimer(self)
        self._cell_list_timer.setInterval(500)
        self._cell_list_timer.timeout.connect(self._refresh_cell_list_window)

        self.patch_toolbar = QtWidgets.QToolBar("Patch Controls")
        self.addToolBar(Qt.TopToolBarArea, self.patch_toolbar)
        self.patch_toolbar.addWidget(self.show_cells_button)

        self.pipette_status_window = PipetteStatusWindow(self.patch_interfaces, self)
        self.pipette_status_window.setWindowFlag(Qt.Tool)
        self.show_pipette_status_button = QtWidgets.QPushButton("Pipette Status")
        self.pipette_status_window.closed.connect(self._pipette_status_window_closed)
        self.show_pipette_status_button.setCheckable(True)
        self.show_pipette_status_button.clicked.connect(self.toggle_pipette_status_window)
        # self.status_bar.insertPermanentWidget(1, self.show_pipette_status_button)
        self.pipette_status_timer = QtCore.QTimer(self)
        self.pipette_status_timer.setInterval(100)
        self.pipette_status_timer.timeout.connect(self.pipette_status_window.update_status)
        self.patch_toolbar.addWidget(self.show_pipette_status_button)

        self.switch_manipulator_box = QtWidgets.QComboBox()
        self.patch_toolbar.addSeparator()
        self.patch_toolbar.addWidget(QtWidgets.QLabel("Active Pipette: "))
        self.patch_toolbar.addWidget(self.switch_manipulator_box)
        # self.status_bar.insertPermanentWidget(1, self.switch_manipulator_box)

        self.classic_tabs = []
        self._unique_patch_signals = {}

        for id, curr_pipette_interface in self.pipette_interfaces.items():
            widget = QtWidgets.QTabWidget()
            self.config_tabs[id] = widget
            curr_config_tab = self.config_tabs[id]

            curr_patch_interface = self.patch_interfaces[id]

            self.switch_manipulator_box.addItem(f"{id}")
            curr_patch_interface.moveToThread(self.control_threads[id])

            signals = PatchSignals()
            self._unique_patch_signals[id] = signals
            self.interface_signals[curr_patch_interface] = (signals.command, signals.reset)

            self.add_config_gui(curr_pipette_interface.calibration_config, curr_config_tab)
            self.add_config_gui(curr_patch_interface.config, curr_config_tab)
            self.add_config_gui(curr_patch_interface.protocol_config, curr_config_tab)
            self.add_config_gui(curr_patch_interface.experiment_book_config, curr_config_tab)
            # self.snapshot_captured.connect(self.experiment_book_tab.handle_snapshot)
            logging.debug("Added config GUI.")
            classic_patching_tab = ClassicPatchButtons(curr_patch_interface, curr_pipette_interface, self.start_task, self.interface_signals, self.recording_state_manager)
            self.classic_tabs.append(classic_patching_tab)
            self.add_tab(classic_patching_tab, 'Classic Auto Patching', curr_config_tab, index = 0)
        
        self.current_tab = list(self.config_tabs.values())[0]

        self.config_scroll_area = QtWidgets.QScrollArea() # Made it 'self.' just in case you need to access it later
        self.config_scroll_area.setWidgetResizable(True)
        self.config_scroll_area.setWidget(self.current_tab)
        self.config_scroll_area.setMinimumWidth(100) 
        
        self.splitter.addWidget(self.config_scroll_area)
        self.splitter.setSizes([2000, 500])
        self.splitter.setStretchFactor(0, 1) 
        self.splitter.setStretchFactor(1, 0)

        self.switch_manipulator_box.currentTextChanged.connect(self.switch_active_pipette)

        self.active_patch_interface = list(self.patch_interfaces.values())[0] if isinstance(patch_interfaces, dict) else patch_interfaces

        self.main_toolbar_default_style = self.main_toolbar.styleSheet()
        self.patch_toolbar_default_style = self.patch_toolbar.styleSheet()
        self.config_tab_default_style = list(self.config_tabs.values())[0].styleSheet()
        self.cell_list_window_default_style = self.cell_list_window.styleSheet()
        self.pipette_status_window_default_style = self.pipette_status_window.styleSheet()

        self.apply_theme("light")

    def register_commands(self):
        """
        Register GUI mouse and keyboard actions to patching interface commands.
        Overrides parent method to include patch-specific actions.
        """
        super(PatchGui, self).register_commands()
        # self.register_mouse_action(Qt.LeftButton, Qt.ShiftModifier,
        #                            self.active_patch_interface.patch_with_move)
        self.register_mouse_action(Qt.LeftButton, Qt.NoModifier,
                                   self.patch_interface.add_cell)
        self.register_mouse_action(Qt.RightButton, Qt.ShiftModifier,
                                   self.patch_interface.handle_corner_right_click)
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
        if not self.dark_mode:
            self.apply_theme("dark")
            self.dark_mode = True
            for config_tab in list(self.config_tabs.values()):
                    for i in range(config_tab.count()):
                        curr_tab = config_tab.widget(i)

                        for box in curr_tab.findChildren(CollapsibleGroupBox):
                            if hasattr(box, 'dark_style_sheet'):
                                box.setStyleSheet(box.dark_style_sheet)
                        if hasattr(curr_tab, "save_button"):
                            curr_tab.save_button.setIcon(qta.icon('fa.download', color='white'))
                            curr_tab.load_button.setIcon(qta.icon('fa.upload', color='white'))

            self.task_progress.setStyleSheet("""
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
            """)

            self.task_abort_button.setIcon(qta.icon('fa.ban', color='white'))
            self.task_success_button.setIcon(qta.icon('fa.check', color='white'))
            self.help_button.setIcon(qta.icon('fa.question-circle', color='white'))
            self.log_button.setIcon(qta.icon('fa.file', color='white'))
            self.record_button.setIcon(qta.icon('fa.video-camera', color='white'))
            self.snap_image_button.setIcon(qta.icon('fa.camera', color='white'))
            self.config_button.setIcon(qta.icon('fa.cogs', color='white'))
            
        else:
            self.apply_theme("light")
            self.dark_mode = False
            
            for config_tab in list(self.config_tabs.values()):
                                for i in range(config_tab.count()):
                                    curr_tab = config_tab.widget(i)
            
                                    for box in curr_tab.findChildren(CollapsibleGroupBox):
                                        if hasattr(box, 'default_style_sheet'):
                                            box.setStyleSheet(box.default_style_sheet)
                                    if hasattr(curr_tab, "save_button"):
                                        curr_tab.save_button.setIcon(qta.icon('fa.download', color='black'))
                                        curr_tab.load_button.setIcon(qta.icon('fa.upload', color='black'))

            self.task_progress.setStyleSheet("")

            self.task_abort_button.setIcon(qta.icon('fa.ban', color='black'))
            self.task_success_button.setIcon(qta.icon('fa.check', color='black'))
            self.help_button.setIcon(qta.icon('fa.question-circle', color='black'))
            self.log_button.setIcon(qta.icon('fa.file', color='black'))
            self.record_button.setIcon(qta.icon('fa.video-camera', color='black'))
            self.snap_image_button.setIcon(qta.icon('fa.camera', color='black'))
            self.config_button.setIcon(qta.icon('fa.cogs', color='black'))

    def apply_theme(self, theme: str):
        qdarktheme.setup_theme(
            theme=theme,
            corner_shape="rounded",
            custom_colors=PATCH_GUI_COLORS,
            additional_qss=PATCH_GUI_QSS,
        )

    def switch_active_pipette(self, id):
        """
        Switch the currently active pipette

        Args:
            pipette (PipetteInterface): The pipette to switch to.
        """
        self.active_pipette = self.pipette_interfaces[id]
        self.active_patch_interface = self.patch_interfaces[id]

        self.key_actions.clear()
        self.mouse_actions.clear()
        self.register_commands()

        old_tab = self.config_scroll_area.takeWidget()
        if old_tab is not None:
            old_tab.hide()

        self.current_tab = self.config_tabs[id]
        self.config_scroll_area.setWidget(self.current_tab)
        self.current_tab.show()

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
    
    def run_sequential_commands(self, cmds, button=None, section=None, button_name=None):
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
                    cmds, sequential=False, change_color_on_complete=False, 
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
                button.setMinimumHeight(50)

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
    def __init__(self, patch_interface: AutoPatchInterface, pipette_interface: PipetteInterface, start_task, interface_signals, recording_state_manager: RecordingStateManager):
        """
        Initializes the ClassicPatchButtons GUI and sets up all control sections.

        Args:
            patch_interface (AutoPatchInterface): Interface for patching operations.
            pipette_interface (PipetteInterface): Interface for pipette control.
            start_task (callable): Function to start tasks.
            interface_signals (dict): Signals for interfacing with controllers.
            recording_state_manager (RecordingStateManager): Recording state manager.
        """
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
                self.patch_interface.start_selecting_corners,
                [[self.patch_interface.move_to_scan_start, self.start_recording, self.patch_interface.start_scan, self.stop_recording]],
            ],
            [self.patch_interface.move_group_up, self.patch_interface.move_group_down],
            [self.pipette_interface.center_pipette,self.patch_interface.clean_pipette,self.pipette_interface.focus_pipette]
        ]
        self.addButtonList('movement', layout, buttonList, cmds, sequential=True)

        # Find Pipette path
        self.pipette_location = [self.pipette_interface.center_pipette,self.pipette_interface.focus_pipette,self.pipette_interface.move_pipette_random,self.patch_interface.find_pipette]
        # add a box for testing controllability of the pipette and stage
        buttonList = [['Find Pipette','Test Pipette Movement']]
        cmds = [[self.pipette_location,self.pipette_interface.move_pipette_random_velocity]]
        freq = [[1, 3]]
        self.addButtonList('testing', layout, buttonList, cmds, freq=freq, sequential=True)

        # # Add a box for lamp commands
        buttonList = [['toggle shutter', 'toggle fluorescense'],['move cube left','move cube right']]
        cmds = [[ self.patch_interface.toggle_shutter, self.patch_interface.toggle_fluorescence],
                [self.patch_interface.move_cube_left, self.patch_interface.move_cube_right]
        ]
        self.addButtonList('fluorescence', layout, buttonList, cmds, sequential=True)

        # Add a box for patching commands
        buttonList = [['Select Cell','Remove Last Cell','Center on Cell'],
                      ['Locate Cell','Hunt Cell','Gigaseal'],
                      ['Break-in','Escape Cell'],
                      ['Patch Cell','Run Protocols']]
        cmds = [[self.patch_interface.start_selecting_cells, self.patch_interface.remove_last_cell, self.patch_interface.center_on_cell],
                [self.patch_interface.locate_cell,
                 [self.start_recording,self.patch_interface.hunt_cell],[self.patch_interface.gigaseal]],
                [[ self.patch_interface.break_in],
                 [self.stop_recording,  self.patch_interface.escape_cell]],
                [[self.start_recording,  self.patch_interface.patch, self.stop_recording],
                 [self.stop_recording,  self.patch_interface.run_protocols]]
]
        self.addButtonList('patching', layout, buttonList, cmds, sequential=True, change_color_during={
            'Locate Cell',
            'Hunt Cell',
            'Gigaseal',
            'Break-in',
            'Escape Cell',
            'Run Protocols',
        })

        # Add a box for Rig Recorder
        self.record_button = QtWidgets.QPushButton("Start Recording")
        self.record_button.clicked.connect(self.toggle_recording)
        self.record_button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.record_button.setMinimumWidth(30)
        self.record_button.setMinimumHeight(30)
        layout.addWidget(self.record_button)

        self.setLayout(layout)

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
            self.toggle_recording()
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

        
    def toggle_recording(self):
        """Toggles the recording state on or off."""
        if self.recording_state_manager.is_recording_enabled():
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        """Enables recording and updates UI state."""
        self.recording_state_manager.set_recording(True)
        self.record_button.setText("Stop Recording")
        self.record_button.setStyleSheet("background-color: red; color: white;border-radius: 5px; padding: 5px;")
        logging.info("Recording started")

    def stop_recording(self):
        """Disables recording, finalizes logging, and updates UI state."""
        self.recording_state_manager.set_recording(False)
        self.recorder.handle_recording_stopped()
        self.record_button.setText("Start Recording")
        self.record_button.setStyleSheet("")
        logging.info("Recording stopped")

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
        self.recorder.close()
        super(ClassicPatchButtons, self).close()

    def closeEvent(self, event):
        """
        Handles the widget close event and ensures recorder cleanup.

        Args:
            event (QCloseEvent): Close event.
        """
        self.recorder.close()
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
        """Sets the current stage X position as the zero reference."""
        xPos = self.pipette_interface.calibrated_stage.position(0)
        self.currx_stage_pos = [xPos, 0, 0]
        # update pipette controller stage tare at x position as a numpy array
        self.pipette_interface.tare_stage[0] = xPos
        print("Tare stage x: ", self.currx_stage_pos)
        self.pipette_interface.write_tare()

    def tare_stage_y(self):
        """Sets the current stage Y position as the zero reference."""
        yPos = self.pipette_interface.calibrated_stage.position(1)
        self.curry_stage_pos = [0, yPos, 0]
        # update pipette controller stage tare at y position as a numpy array
        self.pipette_interface.tare_stage[1] = yPos
        print("Tare stage y: ", self.curry_stage_pos)
        self.pipette_interface.write_tare()

    def tare_stage_z(self):
        """Sets the current stage Z position as the zero reference."""
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


