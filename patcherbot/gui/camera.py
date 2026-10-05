# coding=utf-8
from __future__ import absolute_import

import collections

from patcherbot.interface.experimentBookConfig import ExperimentBookConfig
from patcherbot.utils.FileLogger import FileLogger
# Support older versions of Python
try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

import functools
import logging
import datetime
import os
import traceback
import time
from types import MethodType

import param
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import (QDialog, QPushButton, QDialogButtonBox, QHBoxLayout, QVBoxLayout,
                             QLabel, QLineEdit, QStyle, QFileDialog, QSpinBox)
import qtawesome as qta

from patcherbot.interface.camera import CameraInterface
from patcherbot.controller import TaskController
from patcherbot.configs.Config import NumberWithUnit
from patcherbot.interface.base import command
from .livefeed import LiveFeedQt


class Logger(QtCore.QAbstractTableModel, logging.Handler):
    """
    A Qt table model that also acts as a logging handler.

    Stores log messages in a table format suitable for display in a QTableView.
    Can save messages to a file and color-code by log level.
    """
    def __init__(self):
        """Initialize the Logger model and logging handler."""
        super(Logger, self).__init__()
        # We do not actually use the formatter, but the asctime attribute is
        # available if the formatter requires it
        self.setFormatter(logging.Formatter('%(asctime)s'))
        self.messages = []

    def emit(self, record):
        """
        Process a log record and insert it into the table model.

        Args:
            record (logging.LogRecord): Log record to process.
        """
        self.format(record)
        entry = (record.levelno,
                 datetime.datetime.strptime(record.asctime, '%Y-%m-%d %H:%M:%S,%f'),
                 record.name,
                 record.message,
                 record.exc_info,
                 record.thread)  # Not displayed by default
        self.beginInsertRows(QtCore.QModelIndex(),
                             len(self.messages), len(self.messages))
        self.messages.append(entry)
        self.endInsertRows()

    def rowCount(self, parent=None):
        """
        Return the number of rows in the table model.

        Args:
            parent: Qt parent index (ignored).

        Returns:
            int: Number of log messages.
        """
        return len(self.messages)

    def columnCount(self, parent=None):
        """
        Return the number of rows in the table model.

        Args:
            parent: Qt parent index (ignored).

        Returns:
            int: Number of log messages.
        """
        return 4

    def headerData(self, section, orientation, role):
        """
        Return header text for a column.

        Args:
            section (int): Column index.
            orientation (Qt.Orientation): Horizontal or vertical.
            role (Qt.ItemDataRole): Role requesting the data.

        Returns:
            str or None: Header label if role is DisplayRole, otherwise None.
        """
        if role != Qt.DisplayRole:
            return
        if orientation == Qt.Horizontal:
            return ['', 'time', 'origin', 'message'][section]

    def data(self, index, role):
        """
        Return data for a given table index and role.

        Args:
            index (QModelIndex): Table index.
            role (Qt.ItemDataRole): Data role.

        Returns:
            object: Display value or color depending on role.
        """
        if not index.isValid():
            return None
        if index.row() >= len(self.messages) or index.row() < 0:
            return None

        level, asctime, name, message, exc_info, _ = self.messages[index.row()]
        if role == Qt.DisplayRole or role == Qt.ToolTipRole:
            if index.column() == 0:
                if level == logging.DEBUG:
                    return 'D'
                elif level == logging.INFO:
                    return 'I'
                elif level == logging.WARN:
                    return 'W'
                elif level == logging.ERROR:
                    return 'E'
                else:
                    return None
            elif index.column() == 1:
                return asctime.strftime('%H:%M:%S,%f')[:-3]  # ms instead of us
            elif index.column() == 2:
                return name
            elif index.column() == 3:
                if role == Qt.DisplayRole:
                    return message
                else:
                    if exc_info is None:
                        return message
                    else:
                        return message + '\n' + ''.join(traceback.format_exception(*exc_info))
            else:
                return None
        if role == Qt.ForegroundRole:
            if level == logging.WARN:
                return QtGui.QColor('darkorange')
            elif level == logging.ERROR:
                return QtGui.QColor('darkred')

    def save_to_file(self, filename):
        """
        Save all logged messages to a text file.

        Args:
            filename (str): Path to save file.

        Raises:
            OSError, IOError: If file cannot be written.
        """
        with open(filename, 'w') as f:
            for entry in self.messages:
                level, asctime, name, message, exc_info, thread = entry
                fmt = '{level} {time} {origin} {thread_id}: {message}\n'
                level_name = {logging.DEBUG: 'DEBUG',
                             logging.INFO: 'INFO',
                             logging.WARN: 'WARN',
                             logging.ERROR: 'ERROR'}[level]
                if exc_info is not None:
                    message += '\n' + ''.join(traceback.format_exception(*exc_info))
                f.write(fmt.format(level=level_name,
                                   time=asctime.isoformat(' '),
                                   origin=name,
                                   thread_id=thread,
                                   message=message))


class LogViewerWindow(QtWidgets.QMainWindow):
    """
    A main window displaying logs using a Logger model.

    Provides filtering by log level, saving logs to file, and scrolls
    automatically when new messages are added.
    """
    close_signal = QtCore.pyqtSignal()
    levels = collections.OrderedDict([('DEBUG',logging.DEBUG),
                                      ('INFO', logging.INFO),
                                      ('WARN', logging.WARN),
                                      ('ERROR', logging.ERROR)])

    def __init__(self, parent):
        """
        Initialize the LogViewerWindow.
       
        Args:
            parent: Parent Qt widget.
        """
        super(LogViewerWindow, self).__init__(parent=parent)
        self.setWindowTitle('Log')
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        # self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.log_view = QtWidgets.QTableView()
        self.logger = Logger()
        self.log_view.setModel(self.logger)
        self.log_view.horizontalHeader().setStretchLastSection(True)
        self.log_view.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        # self.log_view.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.log_view.setShowGrid(False)
        self.log_view.setAlternatingRowColors(True)
        self.logger.rowsInserted.connect(self.log_view.scrollToBottom)
        logging.getLogger().addHandler(self.logger)
        logging.getLogger().setLevel(logging.DEBUG)
        self.current_levelno = logging.DEBUG
        central_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        self.level_selection = QtWidgets.QComboBox()
        self.level_selection.insertItems(0, self.levels.keys())
        self.level_selection.currentIndexChanged.connect(self.set_level)
        top_row = QtWidgets.QHBoxLayout()
        top_row.addWidget(self.level_selection)
        self.save_button = QtWidgets.QToolButton(clicked=self.save_log)
        self.save_button.setIcon(qta.icon('fa.download'))
        top_row.addWidget(self.save_button)
        layout.addLayout(top_row)
        layout.addWidget(self.log_view)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def closeEvent(self, event):
        """
        Handle the window close event.

        Emits the `close_signal`.

        Args:
            event (QCloseEvent): Close event.
        """
        self.close_signal.emit()
        super(LogViewerWindow, self).closeEvent(event)

    def set_level(self, level_idx):
        """
        Filter log messages by log level.

        Args:
            level_idx (int): Index of selected level in levels combobox.
        """
        levelno = list(self.levels.values())[level_idx]
        if self.current_levelno == levelno:
            return
        for row in range(self.logger.rowCount()):
            if self.logger.messages[row][0] >= levelno:
                self.log_view.showRow(row)
            else:
                self.log_view.hideRow(row)
        self.current_levelno = levelno

    def save_log(self):
        """Open a file dialog to save the log contents to a file."""
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Log File',
                                                            filter='Text files(*.txt)',
                                                            options=QtWidgets.QFileDialog.DontUseNativeDialog)
        if not filename:
            return
        try:
            self.logger.save_to_file(filename)
        except (OSError, IOError):
            logging.getLogger(__name__).exception('Saving log file to "{}" '
                                                  'failed.'.format(filename))


class KeyboardHelpWindow(QtWidgets.QMainWindow):
    """
    A help window for displaying keyboard and mouse actions.

    Supports registering key actions, mouse actions, and custom actions,
    organized by categories.
    """

    close_signal = QtCore.pyqtSignal()

    def __init__(self, parent):
        """
        Initialize the KeyboardHelpWindow.

        Args:
            parent: Parent Qt widget.
        """
        super(KeyboardHelpWindow, self).__init__(parent=parent)
        self.setWindowTitle('Keyboard/mouse commands')
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setWindowFlags(Qt.Tool | Qt.WindowStaysOnTopHint)
        self.label = QtWidgets.QLabel()
        self.setCentralWidget(self.label)
        self.key_catalog = collections.OrderedDict()
        self.mouse_catalog = collections.OrderedDict()
        self.custom_catalog = collections.OrderedDict()

    def keyPressEvent(self, event):
        """
        Forward key press events to the parent window.

        Args:
            event (QKeyEvent): Key press event.
        """
        # Forward key presses to the parent window
        return self.parent().keyPressEvent(event)

    def register_key_action(self, key, modifier, category, description):
        """
        Register a key action for the help window.

        Args:
            key (int): Qt key code.
            modifier (Qt.KeyboardModifier): Optional modifier.
            category (str): Category of the action.
            description (str): Description of the action.
        """
        if category not in self.key_catalog:
            self.key_catalog[category] = []
        self.key_catalog[category].append((key, modifier, description))
        self.update_text()

    def register_mouse_action(self, click_type, modifier, category, description):
        """
        Register a mouse action for the help window.

        Args:
            click_type (Qt.MouseButton): Type of mouse click.
            modifier (Qt.KeyboardModifier): Optional modifier.
            category (str): Action category.
            description (str): Description of the action.
        """
        if category not in self.mouse_catalog:
            self.mouse_catalog[category] = []
        self.mouse_catalog[category].append((click_type, modifier, description))
        self.update_text()

    def register_custom_action(self, category, action, description):
        """
        Register a custom (non-key/mouse) action.

        Args:
            category (str): Action category.
            action (str): Action name.
            description (str): Description of the action.
        """
        if category not in self.custom_catalog:
            self.custom_catalog[category] = []
        self.custom_catalog[category].append((action, description))

    def update_text(self):
        """Update the displayed HTML table of all registered actions."""
        lines = []
        # Keys
        for category, key_info in self.key_catalog.items():
            # FIXME: The logic below assumes that there is no category that does
            # not have any standard key actions. Instead, we should build a
            # list of all categories first and then go through all catalogs.
            lines.append('<tr><td colspan=2 style="font-size: large; padding-top: 1ex">{}</td></tr>'.format(category))

            mouse_info = self.mouse_catalog.get(category, [])
            for click_type, modifier, description in mouse_info:
                if modifier is not None and modifier != Qt.NoModifier:
                    key_text = QtGui.QKeySequence(int(modifier)).toString()
                else:
                    key_text = ''
                if click_type == Qt.LeftButton:
                    mouse_text = 'Left click'
                elif click_type == Qt.RightButton:
                    mouse_text = 'Right click'
                elif click_type == Qt.MiddleButton:
                    mouse_text = 'Middle click'
                else:
                    mouse_text = '??? click'
                action = key_text + mouse_text
                lines.extend(self._format_action(action, description))

            custom_info = self.custom_catalog.get(category, [])
            for action, description in custom_info:
                lines.extend(self._format_action(action, description))

            for key, modifier, description in key_info:
                if modifier is not None:
                    key_text = QtGui.QKeySequence(int(modifier) + key).toString()
                else:
                    key_text = QtGui.QKeySequence(key).toString()

                lines.extend(self._format_action(key_text, description))
        text = '<table>' +('\n'.join(lines)) + '</table>'

        self.label.setText(text)

    def _format_action(self, action, description):
        """
        Format a single action/description pair as HTML table rows.

        Args:
            action (str): Action key or mouse description.
            description (str): Action description.

        Returns:
            list: List of HTML strings representing the table row.
        """
        lines = ['<tr>',
                 '<td style="font-family: monospace; font-weight: bold; align: center; padding-right: 1ex">{}</td>'
                 '<td>{}</td>'.format(action, description),
                 '</tr>']
        return lines

    def closeEvent(self, event):
        """
        Handle the window close event.

        Emits the `close_signal`.

        Args:
            event (QCloseEvent): Close event.
        """
        self.close_signal.emit()
        super(KeyboardHelpWindow, self).closeEvent(event)


class LogNotifyHandler(logging.Handler):
    """
    A logging handler that emits messages through a Qt signal.
    """
    def __init__(self, signal):
        """
        Initialize the LogNotifyHandler.

        Args:
            signal: Qt signal used to emit log messages.
        """
        super(LogNotifyHandler, self).__init__()
        self.signal = signal

    def emit(self, record):
        """
        Emit a log message through the provided Qt signal.

        Args:
            record (logging.LogRecord): Log record.
        """
        self.format(record)
        if record.exc_info is None:
            message = record.msg
        else:
            _, exc, _ = record.exc_info
            message = '{} ({})'.format(record.msg, str(exc))
        self.signal.emit(message)

class RecordingDialog(QDialog):
    """
    Dialog for configuring video recording settings.
    """
    def __init__(self, base_directory, frame_rate, pixels, settings, parent=None):
        """
        Initialize the RecordingDialog.

        Args:
            base_directory (str): Base directory for saving recordings.
            frame_rate (float): Camera frame rate.
            pixels (int): Number of pixels per frame.
            settings (dict): Existing recording settings.
            parent: Parent Qt widget.
        """
        super(RecordingDialog, self).__init__(parent=parent)

        self.frame_rate = frame_rate
        self.pixels = pixels

        self.setWindowTitle('Recording')
        self.directory_label = QLabel('Directory:')
        self.directory_edit = QLineEdit()
        dir_name = time.strftime('%Y%m%d_%H-%M-%S', time.localtime())
        self.directory_edit.setText(os.path.join(base_directory, dir_name))
        self.directory_button = QPushButton()
        icon = self.style().standardIcon(QStyle.SP_DirIcon)
        self.directory_button.setIcon(icon)
        self.directory_button.clicked.connect(self.directory_clicked)
        directory_layout = QHBoxLayout()
        directory_layout.addWidget(self.directory_label)
        directory_layout.addWidget(self.directory_edit)
        directory_layout.addWidget(self.directory_button)

        skip_label = QLabel('Skip frames:')
        self.skip_spin = QSpinBox()
        self.skip_spin.setRange(0, 10)
        self.skip_spin.valueChanged.connect(self.skip_edited)
        skip_layout = QHBoxLayout()
        skip_layout.addWidget(skip_label)
        skip_layout.addWidget(self.skip_spin)
        self.frame_rate_label = QLabel('')
        self.skip_spin.setValue(settings.get('skip_frames', 0))
        self.skip_edited(self.skip_spin.value())  # trigger even for default value

        memory_label = QLabel('Memory for file queue (MB):')
        self.file_queue_frames = QLabel('')
        self.memory_spin = QSpinBox()
        self.memory_spin.setRange(10, 16000)
        self.memory_spin.valueChanged.connect(self.memory_edited)
        self.memory_spin.setValue(settings.get('memory', 1000))
        self.memory_edited(self.memory_spin.value())  # trigger even for default value
        memory_layout = QHBoxLayout()
        memory_layout.addWidget(memory_label)
        memory_layout.addWidget(self.memory_spin)
        

        self.prefix_label = QLabel('Prefix:')
        self.prefix_edit = QLineEdit()
        self.prefix_edit.textChanged.connect(self.prefix_edited)
        prefix_layout = QHBoxLayout()
        prefix_layout.addWidget(self.prefix_label)
        prefix_layout.addWidget(self.prefix_edit)
        self.prefix_preview = QLabel()
        self.prefix_edit.setText(settings.get('prefix', 'frame'))

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        self.layout = QVBoxLayout()
        self.layout.addLayout(directory_layout)
        self.layout.addLayout(prefix_layout)
        self.layout.addWidget(self.prefix_preview)
        self.layout.addLayout(skip_layout)
        self.layout.addWidget(self.frame_rate_label)
        self.layout.addLayout(memory_layout)
        self.layout.addWidget(self.file_queue_frames)
        self.layout.addWidget(btns)
        self.setLayout(self.layout)
    
    def prefix_edited(self):
        """Update the prefix preview label when the prefix is changed."""
        self.prefix_preview.setText('<i>{}_00000.tiff</i>'.format(self.prefix_edit.text()))

    def skip_edited(self, value):
        """
        Update frame rate label based on skip frames value.
        
        Args:
            value (int): Number of frames to skip.
        """
        if self.frame_rate > 0:
            rate = '~{:.1f}'.format(self.frame_rate / (value + 1))
        else:
            rate = '?'
        self.frame_rate_label.setText('<i>{} frames per second</i>'.format(rate))

    def memory_edited(self, value):
        """
        Update the estimated number of frames in memory queue.

        Args:
            value (int): Memory size in MB.
        """
        self.file_queue_frames.setText('<i>space for ~{} frames in queue'.format(int(value*1e6/self.pixels)))

    def directory_clicked(self):
        """Open a folder selection dialog and update the directory edit."""
        folder = self.select_folder()
        if folder is not None:
            print(folder, folder)
            self.directory_edit.setText(folder)

    def select_folder(self):
        """
        Open a QFileDialog to select a folder.

        Returns:
            str or None: Selected folder path, or None if cancelled.
        """
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.ShowDirsOnly)
        dialog.setWindowTitle('Select recording directory')
        dialog.setDirectory(self.directory_edit.text())
        if dialog.exec_():
            return dialog.selectedFiles()[0]

class CameraSignals(QtCore.QObject):
    command = QtCore.pyqtSignal(MethodType, object)
    reset = QtCore.pyqtSignal(TaskController)

class CameraGui(QtWidgets.QMainWindow):
    '''
    The basic GUI for showing a camera image.

    Parameters
    ----------
    camera : `.Camera`
        The `.Camera` object that will be used for displaying an image via
        `.LiveFeedQt`.
    image_edit : function or list of functions, optional
        A function that will be called with the numpy array returned by the
        camera. Can be used to post-process the image, e.g. to change its
        brightness.
    display_edit : function or list of functions, optional
        A function that will be called with the `.QPixmap` that is based on
        the camera image. Can be used to display additional information on top
        of this image, e.g. a scale bar or text.
    with_tracking : bool, optional
        Whether to activate the object tracking interface. Defaults to
        ``False``.
    '''
    log_signal = QtCore.pyqtSignal('QString')
    camera_signal = QtCore.pyqtSignal(MethodType, object)
    camera_reset_signal = QtCore.pyqtSignal(TaskController)
    snapshot_captured = QtCore.pyqtSignal(object)


    def __init__(self, camera, pipette_cameras=None, recording_state_manager=None,
                 camera_recording_session=None,
                 image_edit=None, display_edit=None,
                 with_tracking=False, base_directory='.'):
        """
        Initialize the CameraGui.

        Args:
            camera: Primary camera object.
            aux_camera: Optional auxiliary camera.
            recording_state_manager: Manager controlling recording state.
            image_edit (callable or list, optional): Image processing functions.
            display_edit (callable or list, optional): Display overlay functions.
            with_tracking (bool): Enable object tracking interface.
            base_directory (str): Base directory for recordings.

        Raises:
            ValueError: If recording_state_manager is not provided.
        """
        super().__init__()
        self.main_camera = camera

        # Normalize pipette cameras to a keyed collection.
        if pipette_cameras is None:
            self.pipette_cameras = {}

        elif isinstance(pipette_cameras, dict):
            self.pipette_cameras = dict(pipette_cameras)

        else:
            # Single-pipette backwards compatibility.
            self.pipette_cameras = {
                "pipette_camera_0": pipette_cameras
            }

        self.active_pipette_camera_id = (
            next(iter(self.pipette_cameras))
            if self.pipette_cameras
            else None
        )

        self.recording_state_manager = recording_state_manager
        self.camera_recording_session = camera_recording_session
        if self.recording_state_manager is None:
            raise ValueError("RecordingStateManager must be provided")
        self.snap_image_recorder = FileLogger(
                    self.recording_state_manager,
                    folder_path="experiments/Data/snap_image_data/",
                    isVideo=True,
                    filetype="csv",
                    recorder_filename="snap_images",
                    frame_batch_size=2,
                )
        self.base_directory = base_directory
        self.with_tracking = with_tracking
        self.show_overlay = True
        self.is_recording = False

        self.main_interface = None
        if self.main_camera is not None:
            self.main_interface = CameraInterface(
                self.main_camera,
                self.snap_image_recorder,
                source_id="main_camera",
                with_tracking=with_tracking,
                status_category="Main Camera",
            )
        self.pipette_camera_interfaces = {}

        for camera_id, pipette_camera in self.pipette_cameras.items():
            self.pipette_camera_interfaces[camera_id] = CameraInterface(
                pipette_camera,
                self.snap_image_recorder,
                source_id=camera_id,
                with_tracking=False,
                status_category=f"Pipette Camera {camera_id}",
            )


        if self.main_camera is not None:
            self.active_camera_role = "main"
            self.active_camera = self.main_camera
            self.active_interface = self.main_interface

        elif self.selected_pipette_camera is not None:
            self.active_camera_role = "pipette"
            self.active_camera = self.selected_pipette_camera
            self.active_interface = self.selected_pipette_camera_interface

        else:
            self.active_camera_role = None
            self.active_camera = None
            self.active_interface = None

        self.camera = self.active_camera
        self.camera_interface = self.active_interface

        self.main_toolbar = QtWidgets.QToolBar("Main Controls")
        self.addToolBar(Qt.TopToolBarArea, self.main_toolbar)

        

        self.status_bar = QtWidgets.QStatusBar()
        self.setStatusBar(self.status_bar)

        self.status_label = QtWidgets.QLabel()
        self.status_bar.addPermanentWidget(self.status_label)

        self.task_abort_button = QtWidgets.QToolButton(clicked=self.abort_task)
        self.task_abort_button.setIcon(qta.icon('fa.ban'))
        self.task_abort_button.setVisible(False)
        self.status_bar.addWidget(self.task_abort_button)
        self.task_progress = QtWidgets.QProgressBar(parent=self)
        self.task_progress.setMaximum(0)
        self.task_progress.setAlignment(Qt.AlignLeft)
        self.task_progress.setTextVisible(False)
        self.task_progress.setVisible(False)
        layout = QtWidgets.QHBoxLayout(self.task_progress)
        self.task_progress_text = QtWidgets.QLabel()
        self.task_progress_text.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.task_progress_text)
        layout.setContentsMargins(0, 0, 0, 0)
        self.status_bar.addWidget(self.task_progress, 1)
        self.task_success_button = QtWidgets.QToolButton(clicked=self.complete_task)
        self.task_success_button.setIcon(qta.icon('fa.check'))
        self.task_success_button.setEnabled(False)
        self.task_success_button.setVisible(False)
        self.status_bar.addWidget(self.task_success_button)
        self.status_label = QtWidgets.QLabel()
        self.status_bar.addPermanentWidget(self.status_label)

        self.help_button = QtWidgets.QToolButton(clicked=self.toggle_help)
        self.help_button.setIcon(qta.icon('fa.question-circle'))
        self.help_button.setCheckable(True)
        self.help_button.setToolTip('Toggle help window display')

        self.log_button = QtWidgets.QToolButton(clicked=self.toggle_log)
        self.log_button.setIcon(qta.icon('fa.file'))
        self.log_button.setCheckable(True)
        self.log_button.setToolTip('Toggle log window display')

        self.overlay_button = QtWidgets.QToolButton(clicked=self.toggle_overlay)
        self.overlay_button.setIcon(qta.icon('fa.bullseye'))
        self.overlay_button.setCheckable(True)
        self.overlay_button.setChecked(self.show_overlay)
        self.overlay_button.setToolTip('Show/hide overlays (does not enable overlay recording)')

        self.record_button = QtWidgets.QToolButton(clicked=self.toggle_recording)
        self.record_button.setIcon(qta.icon('fa.video-camera'))
        self.record_button.setCheckable(True)
        self.record_button.setToolTip('Toggle video recording')
        self.record_button.setStyleSheet('QToolButton:checked {background-color: red;}')

        self.autoexposure_button = QtWidgets.QToolButton(clicked=self.normalize_active_camera)
        self.autoexposure_button.setText('Normalize')
        self.autoexposure_button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.autoexposure_button.setToolTip('Normalize the image')

        self.unnormalize_button = QtWidgets.QToolButton(clicked=self.unnormalize_active_camera)
        self.unnormalize_button.setText('Unnormalize')
        self.unnormalize_button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.unnormalize_button.setToolTip('Unnormalize the image')

        self.snap_image_button = QtWidgets.QToolButton(clicked=self.snap_active_camera_image)
        self.snap_image_button.setIcon(qta.icon('fa.camera'))
        self.snap_image_button.setToolTip('Snap image')

        # create autonormalizatoin checkbox
        self.autonormalize_checkbox = QtWidgets.QCheckBox('Auto-normalize')
        self.autonormalize_checkbox.setChecked(False)
        self.autonormalize_checkbox.stateChanged.connect(self.handle_autonormalize_change)

        self.switch_view_button = QtWidgets.QPushButton()
        self.switch_view_button.clicked.connect(self.toggle_camera_view)
        self.switch_view_button.setEnabled(self.main_camera is not None and bool(self.pipette_cameras))
        self._update_switch_button_text()

        self.setexposure_edit = QtWidgets.QLineEdit()
        self.setexposure_edit.setMaximumWidth(200)

        # convert the text to a float and set the exposure time if the user presses enter and clear the text box
        self.setexposure_edit.returnPressed.connect(self.apply_active_exposure)

        self.status_bar.addPermanentWidget(self.switch_view_button)

        self.dark_mode = False
        self.toggle_dark_mode_button = QtWidgets.QPushButton("Dark mode")
        self.toggle_dark_mode_button.clicked.connect(self.toggle_dark_mode)
        self.status_bar.addPermanentWidget(self.toggle_dark_mode_button)
 
        self.setexposure_edit.setPlaceholderText('Exposure time (ms)')
        self.status_bar.addPermanentWidget(self.setexposure_edit)
        self.status_bar.addPermanentWidget(self.help_button)
        # self.status_bar.addPermanentWidget(self.log_button)
        # self.status_bar.addPermanentWidget(self.record_button)
        self.status_bar.addPermanentWidget(self.log_button)
        self.status_bar.addPermanentWidget(self.overlay_button)
        self.status_bar.addPermanentWidget(self.record_button)
        self.status_bar.addPermanentWidget(self.snap_image_button)
        self.status_bar.addPermanentWidget(self.autoexposure_button)
        self.status_bar.addPermanentWidget(self.unnormalize_button)
        self.status_bar.addPermanentWidget(self.autonormalize_checkbox)

        self.status_bar.setSizeGripEnabled(False)
        self.setStatusBar(self.status_bar)

        self.status_bar.messageChanged.connect(self.status_message_updated)
        self.status_messages = collections.OrderedDict()
        self.key_actions = {}
        self.mouse_actions = {}
        self.help_window = KeyboardHelpWindow(self)
        self.help_window.setFocusPolicy(Qt.NoFocus)
        self.help_window.setVisible(False)
        self.help_window.close_signal.connect(
            lambda: self.help_button.setChecked(False))
        self.log_window = LogViewerWindow(self)
        self.log_window.setFocusPolicy(Qt.NoFocus)
        self.log_window.close_signal.connect(
            lambda: self.log_button.setChecked(False))
        self.running_task = None
        self.running_task_interface = None
        self.config_button = None  # see initialize
        self.setWindowTitle("Camera GUI")

        self.status_bar.addWidget(self.task_abort_button)
        self.status_bar.addWidget(self.task_progress, 1)
        self.status_bar.addWidget(self.task_success_button)
        self.main_toolbar.addWidget(self.switch_view_button)
        self.main_toolbar.addWidget(self.toggle_dark_mode_button)
        self.main_toolbar.addWidget(self.setexposure_edit)
        self.main_toolbar.addSeparator() # Adds a nice vertical line
        self.main_toolbar.addWidget(self.snap_image_button)
        self.main_toolbar.addWidget(self.autoexposure_button)
        self.main_toolbar.addWidget(self.unnormalize_button)
        self.main_toolbar.addWidget(self.autonormalize_checkbox)
        self.main_toolbar.addSeparator()
        self.main_toolbar.addWidget(self.help_button)

        self.display_edit_funcs = []
        if display_edit is None:
            display_edit = [self.draw_circle]
        if isinstance(display_edit, Sequence):
            self.display_edit_funcs.extend(display_edit)
        else:
            self.display_edit_funcs.append(display_edit)

        self.image_edit_funcs = []
        if isinstance(image_edit, Sequence):
            self.image_edit_funcs.extend(image_edit)
        elif image_edit is not None:
            self.image_edit_funcs.append(image_edit)
        self._camera_key_bindings = []

        self.main_video = None
        if self.main_camera is not None:
            self.main_video = LiveFeedQt(self.main_camera,
                                         image_edit=self.image_edit,
                                         display_edit=functools.partial(self.display_edit, camera=self.main_camera),
                                         mouse_handler=self.video_mouse_press,
                                         )
        self.pipette_video = None
        if self.selected_pipette_camera is not None:
            self.pipette_video = LiveFeedQt(self.selected_pipette_camera,
                                        image_edit=self.image_edit,
                                        display_edit=functools.partial(self.display_edit, camera=self.aux_camera),
                                        mouse_handler=self.video_mouse_press,
                                        )

        self.video_stack = QtWidgets.QStackedWidget()
        if self.main_video is not None:
            self.video_stack.addWidget(self.main_video)
        if self.pipette_video is not None:
            self.video_stack.addWidget(self.pipette_video)
        self.active_video = self.main_video if self.active_camera_role == 'main' else self.pipette_video
        if self.active_video is not None:
            self.video_stack.setCurrentWidget(self.active_video)
        self.video_stack.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.video_stack.setMinimumSize(200, 150)

        self.recording_settings = {}
        self.setFocus()  # Need this to handle arrow keys, etc.
        self.interface_signals = {}
        self._pipette_camera_signals = {}

        if self.main_interface is not None:
            self.interface_signals[self.main_interface] = (
                self.camera_signal,
                self.camera_reset_signal,
            )

        for camera_id, interface in (
            self.pipette_camera_interfaces.items()
        ):
            signals = CameraSignals()

            self._pipette_camera_signals[camera_id] = signals

            self.interface_signals[interface] = (
                signals.command,
                signals.reset,
            )
            
        self.splitter = QtWidgets.QSplitter()
        self.splitter.addWidget(self.video_stack)
        self.config_tabs = {}
        self.setCentralWidget(self.splitter)
        self.splitter.setSizes([1, 0])
        self.splitter.splitterMoved.connect(self.splitter_size_changed)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setMinimumSize(200, 300)

        self.config_scroll_area = None

        # Display error messages directly in the status bar
        handler = LogNotifyHandler(self.log_signal)
        handler.setLevel(logging.ERROR)
        logging.getLogger('patcherbot').addHandler(handler)
        self.log_signal.connect(self.error_status)

        self.status_timer = QTimer(self)
        self.status_timer.setSingleShot(True)
        self.status_timer.timeout.connect(self.status_label.clear)

    def _update_switch_button_text(self):
        """Updates the label of the switch view button based on the current active camera."""
        if (
            not hasattr(self, "switch_view_button")
            or self.switch_view_button is None
        ):
            return

        if not self.pipette_cameras:
            self.switch_view_button.setText(
                "Switch View"
            )
            return

        if self.active_camera_role == "main":
            self.switch_view_button.setText(
                "Switch to Pipette View"
            )
        else:
            self.switch_view_button.setText(
                "Switch to Microscope View"
            )

    def normalize_active_camera(self):
        """Normalize the active camera’s image."""
        if self.active_interface is None:
            return
        self.active_interface.normalize()

    def unnormalize_active_camera(self):
        """Unnormalize the active camera’s image."""
        if self.active_interface is None:
            return
        self.active_interface.unnormalize()

    def snap_active_camera_image(self):
        """Take a snapshot from the active camera."""
        if self.active_interface is None:
            return
        snapshot = self.active_interface.snap_image()
        if snapshot is None:
            return
        snapshot = dict(snapshot)
        snapshot["camera_role"] = self.active_camera_role

        snapshot["camera_id"] = (
            self.active_pipette_camera_id
            if self.active_camera_role == "pipette"
            else "main_camera"
        )
        self.snapshot_captured.emit(snapshot)
        return snapshot

    def handle_autonormalize_change(self, state):
        """
        Enable or disable auto-normalization when the checkbox changes.

        Args:
            state (int): Qt checkbox state (0=unchecked, 2=checked).
        """
        if self.active_interface is None:
            return
        self.active_interface.autonormalize(bool(state))

    def apply_active_exposure(self):
        """Apply the exposure time entered in the setexposure_edit box to the active camera."""
        if self.active_interface is None:
            self.setexposure_edit.clear()
            return
        try:
            exposure_value = float(self.setexposure_edit.text())
        except ValueError:
            self.status_bar.showMessage('Invalid exposure value', 2000)
            return
        self.active_interface.set_exposure(exposure_value)
        self.setexposure_edit.clear()

    def toggle_camera_view(self):
        """Switch between main and auxiliary camera views."""
        if not self.pipette_cameras:
            return

        target_role = (
            "pipette"
            if self.active_camera_role == "main"
            else "main"
        )

        self._set_active_camera(target_role)

    def _set_active_camera(self, role):
        """
        Switch between the microscope camera and selected pipette camera.
        """
        if role not in ("main", "pipette"):
            return

        if role == "main":
            camera = self.main_camera
            interface = self.main_interface
            video = self.main_video

        else:
            camera = self.selected_pipette_camera
            interface = self.selected_pipette_camera_interface
            video = self.pipette_video

        if camera is None or interface is None:
            return

        if role == self.active_camera_role:
            return

        self.active_camera_role = role

        self.active_camera = camera
        self.camera = camera

        self.active_interface = interface
        self.camera_interface = interface

        self.active_video = video

        if video is not None:
            self.video_stack.setCurrentWidget(video)

        self._update_switch_button_text()
        self._rebind_camera_key_actions()
        self._sync_interface_activity()

    def _sync_interface_activity(self):
        """
        Enable only the currently selected camera interface.
        """
        if self.main_interface is not None:
            self.main_interface.set_active(
                self.active_camera_role == "main"
            )

        for camera_id, interface in (
            self.pipette_camera_interfaces.items()
        ):
            interface.set_active(
                self.active_camera_role == "pipette"
                and camera_id
                == self.active_pipette_camera_id
            )

    def _rebind_camera_key_actions(self):
        """Rebinds stored camera key actions to the currently active camera interface."""
        if self.active_interface is None:
            return
        for key, modifier, method_name, argument, _ in self._camera_key_bindings:
            command = getattr(self.active_interface, method_name, None)
            if command is None:
                continue
            self.register_key_action(key, modifier, command, argument, default_doc=False)

    def register_camera_key_action(self, key, modifier, method_name, argument=None, default_doc=True):
        """
        Registers a key binding for a camera-specific action.

        Args:
            key (Qt.Key): Key to bind.
            modifier (Qt.Modifier or None): Modifier key required for the binding.
            method_name (str): Name of the method to invoke on the camera interface.
            argument (Any, optional): Argument to pass to the method.
            default_doc (bool, optional): Whether to include in help documentation.
        """
        binding = (key, modifier, method_name, argument, default_doc)
        if binding not in self._camera_key_bindings:
            self._camera_key_bindings.append(binding)
        if self.active_interface is None:
            return
        command = getattr(self.active_interface, method_name, None)
        if command is None:
            return
        self.register_key_action(key, modifier, command, argument, default_doc=default_doc)

    def _current_video_widget(self):
        """
        Retrieves the currently active video widget.

        Returns:
            QWidget or None: The active video widget, if available.
        """
        return getattr(self, 'active_video', None)

    # Add a cross to the display
    def draw_cross(self, pixmap):
        '''
        Draws a cross at the center. Meant to be used as a ``display_edit``
        function.

        Args:
            pixmap (QPixmap): The pixmap to draw on.
        '''
        painter = QtGui.QPainter(pixmap)
        pen = QtGui.QPen(QtGui.QColor(0, 0, 0, 125))
        pen.setWidth(4)
        painter.setPen(pen)
        c_x, c_y = pixmap.width() // 2, pixmap.height() // 2
        painter.drawLine(c_x - 15, c_y, c_x + 15, c_y)
        painter.drawLine(c_x, c_y - 15, c_x, c_y + 15)
        painter.end()

    # add a circle to the display
    def draw_circle(self, pixmap):
        '''
        Draws a circle at the center. Meant to be used as a ``display_edit``
        function.

        Args:
            pixmap (QPixmap): The pixmap to draw on.
        '''
        painter = QtGui.QPainter(pixmap)
        pen = QtGui.QPen(QtGui.QColor(0, 0, 0, 125))
        pen.setWidth(4)
        painter.setPen(pen)
        c_x, c_y = pixmap.width() // 2, pixmap.height() // 2
        painter.drawEllipse(c_x - 15, c_y - 15, 30, 30)
        painter.end()



    def display_edit(self, pixmap, *, camera=None):
        '''
        Applies the functions stored in `~.CameraGui.display_edit_funcs` to the
        video image pixmap.

        Args:
            pixmap (QPixmap): The pixmap to draw on.
        '''
        if camera is not None and camera is not self.active_camera:
            return
        if self.show_overlay:
            for func in self.display_edit_funcs:
                func(pixmap)

    def image_edit(self, image):
        '''
        Applies the functions stored in `~.CameraGui.image_edit_funcs` to the
        video image. Each function works on the result of the previous function

        Args:
            image (~numpy.ndarray):
                The original video image  or the image returned by a previously
                called function.

        Returns:
            new_image (~numpy.ndarray):
                The post-processed image. Should be of the same size and data type
                as the original image.
        '''
        for func in self.image_edit_funcs:
            image = func(image)
        return image

    def closeEvent(self, evt):
       """
       Handles the Qt close event for the window.

        Args:
            evt (QCloseEvent): Close event.
       """
       self.close()
       return super(CameraGui, self).closeEvent(evt)

    @command(category='General',
             description='Exit the application')
    def exit(self):
        """Closes the application."""
        self.close()

    @command(
        category="Camera",
        description="Toggle recording image files to disk",
    )
    def toggle_recording(self, *args):

        if self.camera_recording_session is None:
            logging.error(
                "CameraRecordingSession is not configured."
            )
            return

        if self.is_recording:
            self.camera_recording_session.stop_recording()

            self.is_recording = False
            self.record_button.setChecked(False)
            return

        active_cam = self.active_camera

        if active_cam is None:
            cameras = self._all_cameras()

            if not cameras:
                return

            active_cam = cameras[0]

        frame_rate = getattr(
            active_cam,
            "get_frame_rate",
            lambda: 0,
        )()

        pixels = (
            getattr(active_cam, "width", 0)
            * getattr(active_cam, "height", 0)
        )

        dlg = RecordingDialog(
            self.base_directory,
            frame_rate=frame_rate,
            pixels=pixels,
            settings=self.recording_settings,
            parent=self,
        )

        if not dlg.exec_():
            self.record_button.setChecked(False)
            return

        directory = os.path.abspath(
            dlg.directory_edit.text()
        )

        prefix = dlg.prefix_edit.text()
        memory = dlg.memory_spin.value()
        skip_frames = dlg.skip_spin.value()

        self.recording_settings["prefix"] = prefix
        self.recording_settings["memory"] = memory
        self.recording_settings["skip_frames"] = skip_frames

        self.camera_recording_session.start_recording(
            directory=directory,
            file_prefix=prefix,
            skip_frames=skip_frames,
            memory_mb=memory,
        )

        self.is_recording = True
        self.record_button.setChecked(True)

    def _all_cameras(self):
        cameras = []

        if self.main_camera is not None:
            cameras.append(self.main_camera)

        cameras.extend(
            camera
            for camera in self.pipette_cameras.values()
            if camera is not None
        )

        unique_cameras = []
        seen = set()

        for camera in cameras:
            identity = id(camera)

            if identity in seen:
                continue

            seen.add(identity)
            unique_cameras.append(camera)

        return unique_cameras

    def register_commands(self):
        '''
        Tie keypresses and mouse clicks to commands. Should call
        `.register_key_action` and `.register_mouse_action`. Overriding methods
        in subclass should call the superclass if they want to keep the
        commands registered by the superclass(es).
        '''
        self.register_key_action(Qt.Key_Question, None, self.help_keypress)
        self.register_key_action(Qt.Key_L, None, self.log_keypress)
        self.register_camera_key_action(Qt.Key_N, None, 'normalize')
        self.register_key_action(Qt.Key_Q, Qt.ControlModifier, self.exit)
        self.register_camera_key_action(Qt.Key_Plus, None,
                                        'increase_exposure',
                                        default_doc=False)
        self.register_camera_key_action(Qt.Key_Minus, None,
                                        'decrease_exposure',
                                        default_doc=False)
        self.help_window.register_custom_action('Camera', '+/-',
                                                'Increase/decrease exposure by 2.5ms')
        # self.register_key_action(Qt.Key_I, None,
        #                          self.camera_interface.save_image)
        self.register_key_action(Qt.Key_I, Qt.SHIFT,
                                 self.toggle_recording)

    def close(self):
        logging.info("closing GUI")

        if self.camera_recording_session is not None:
            self.camera_recording_session.close()

        if self.movement_recorder is not None:
            self.movement_recorder.close()

        for cam in self._all_cameras():
            stop_acq = getattr(
                cam,
                "stop_acquisition",
                None,
            )

            if stop_acq is not None:
                stop_acq()

            close_cam = getattr(
                cam,
                "close",
                None,
            )

            if close_cam is not None:
                close_cam()

        self.main_camera = None
        self.pipette_cameras.clear()
        self.pipette_camera_interfaces.clear()

        self.active_pipette_camera_id = None

        self.camera = None
        self.active_camera = None
        self.active_interface = None
        self.camera_interface = None
        self.active_video = None

        super(CameraGui, self).close()

    def register_mouse_action(self, click_type, modifier, command,
                              default_doc=True):
        '''
        Link a mouse click on the camera image to an action.

        Args:
            click_type (.Qt.MouseButton):
                The type of click that should be handled as a ``Qt`` constant, e.g.
                `.Qt.LeftButton` or `.Qt.RightButton`.
            modifier (.Qt.Modifer or `None`):
                The modifier that needs to be pressed at the same time to trigger
                the action. The modifier needs to be given as a ``Qt`` constant,
                e.g. `.Qt.ShiftModifier` or `.Qt.ControlModifier`. Alternatively,
                ``None`` can be used to specify that the mouse click should lead to
                the action independent of the modifier.
            command (method):
                A method implementing the action that has been annotated with the
                `@command <.command>` or `@blocking_command <.blocking_command>`
                decorator.
            default_doc (bool, optional):
                Whether to include the action in the automatically generated help.
                Defaults to ``True``.
        '''
        self.mouse_actions[(click_type, modifier)] = command
        if default_doc:
            self.help_window.register_mouse_action(click_type, modifier,
                                                   command.category,
                                                   command.auto_description())

    def video_mouse_press(self, event):
        """
        Handles mouse press events on the video display and triggers bound actions.

        Args:
            event (QMouseEvent): Mouse event.
        """
        # Look for an exact match first (key + modifier)
        event_tuple = (event.button(), int(event.modifiers()))
        command = self.mouse_actions.get(event_tuple, None)
        # If not found, check for keys that ignore the modifier
        if command is None:
            command = self.mouse_actions.get((event.button(), None), None)

        if command is not None:
            if self.running_task:
                # Another task is running, ignore the mouse click
                return
            # Mouse commands do not have custom arguments, they always get
            # the position in the image (rescaled, i.e. independent of the
            # window size)
            video_widget = self._current_video_widget()
            if video_widget is None or self.active_camera is None:
                return
            pixmap = video_widget.pixmap()
            if pixmap is None:
                return
            x, y = event.x(), event.y()
            xs = x - video_widget.size().width() / 2.
            ys = y - video_widget.size().height() / 2.
            # displayed image is not necessarily the same size as the original camera image
            scale = 1.0 * self.active_camera.width / pixmap.size().width()
            position = (xs * scale, ys * scale)
            if command.is_blocking:
                self.start_task(command.task_description, command.__self__)
            if command.__self__ in self.interface_signals:
                command_signal, _ = self.interface_signals[command.__self__]
                command_signal.emit(command, position)
            else:
                command(position)

    @QtCore.pyqtSlot('QString')
    def status_message_updated(self, message):
        """
        Updates the status bar styling when a new message is set.

        Args:
            message (str): Status message.
        """
        if not message:
            if self.dark_mode:
                self.status_label.setStyleSheet('QLabel{color: white;}')
            else:
                self.status_label.setStyleSheet('QLabel{color: black;}')

    @QtCore.pyqtSlot('QString')
    def error_status(self, message):
        """
        Displays an error message in the status bar.

        Args:
            message (str): Error message.
        """
        self.status_label.setText(f"<font color='red'>{message}</font>")
        self.status_timer.start(5000)

    def initialize(self):
        """Initializes interfaces, connects signals, and registers commands."""
        for interface, (command_signal, reset_signal) in self.interface_signals.items():
            command_signal.connect(interface.command_received)
            reset_signal.connect(interface.reset_requested)
            interface.task_finished.connect(self.task_finished)
            interface.connect(self)
        self._sync_interface_activity()
        self.register_commands()
        # Add a button for the configuration options if necessary
        if len(self.config_tabs) > 0:
            self.config_button = QtWidgets.QToolButton(
                clicked=self.toggle_configuration_display)
            self.config_button.setIcon(qta.icon('fa.cogs'))
            self.config_button.setCheckable(True)
            self.status_bar.addPermanentWidget(self.config_button)

    def register_key_action(self, key, modifier, command, argument=None,
                            default_doc=True):
        '''
        Link a keypress to an action.

        Args:
            key (.Qt.Key):
                The key that should be handled, specified as a ``Qt`` constant, e.g.
                `.Qt.Key_X` or `.Qt.Key_5`.
            modifier (.Qt.Modifer` or `None`):
                The modifier that needs to be pressed at the same time to trigger
                the action. The modifier needs to be given as a ``Qt`` constant,
                e.g. `.Qt.ShiftModifier` or `.Qt.ControlModifier`. Alternatively,
                ``None`` can be used to specify that the keypress should lead to
                the action independent of the modifier.
            command (method):
                A method implementing the action that has been annotated with the
                `@command <.command>` or `@blocking_command <.blocking_command>`
                decorator.
            argument (object, optional):
                An additional argument that should be handled to the method defined
                as ``command``. Can be used to re-use the same action in a
                parametrized way (e.g. steps of different size).
            default_doc (bool, optional):
                Whether to include the action in the automatically generated help.
                Defaults to ``True``.
        '''
        self.key_actions[(key, modifier)] = (command, argument)
        if default_doc:
            self.help_window.register_key_action(key, modifier,
                                                 command.category,
                                                 command.auto_description(argument))

    def start_task(self, task_name, interface):
        """
        Initializes UI state for a running task.

        Args:
            task_name (str): Name of the task.
            interface (TaskController): Interface handling the task.
        """
        self.status_label.clear()
        self.task_progress_text.setText(task_name + '...')
        self.task_progress.setVisible(True)
        self.task_abort_button.setEnabled(True)
        self.task_success_button.setEnabled(True)
        self.task_success_button.setVisible(True)
        self.task_abort_button.setVisible(True)
        self.running_task = task_name
        self.running_task_interface = interface

    def complete_task(self):
        """Forward a manual success request to the running task interface."""
        if self.running_task_interface is None:
            return
        self.task_success_button.setEnabled(False)
        self.running_task_interface.complete_task()

    def abort_task(self):
        """Aborts the currently running task and disables related UI controls."""
        self.task_abort_button.setEnabled(False)
        self.task_success_button.setEnabled(False)
        self.running_task_interface.abort_task()

    @QtCore.pyqtSlot(int, object)
    def task_finished(self, exit_reason, controller_or_message):
        """
        Handles completion of a task, updating UI state and optionally prompting for reset.

        Args:
            exit_reason (int): Status code indicating task result.
            controller_or_message (object): Controller instance or message string.
        """
        if self.running_task is None:
            # This might be a success message for a non-blocking command
            if isinstance(controller_or_message, str):
                if self.dark_mode:
                    self.status_label.setText(f"<font color='white'>{controller_or_message}</font>")
                else:
                    self.status_label.setText(f"<font color='black'>{controller_or_message}</font>")
                self.status_timer.start(1000)
            return  # Nothing else to do

        self.task_progress.setVisible(False)
        self.task_abort_button.setVisible(False)
        self.task_success_button.setVisible(False)
        self.task_success_button.setEnabled(False)
        # 0: correct execution (no need to show a message)
        if exit_reason == 0:
            text = "Task '{}' finished successfully.".format(self.running_task)
            if self.dark_mode:
                self.status_label.setText(f"<font color='white'>{controller_or_message}</font>")
            else:
                self.status_label.setText(f"<font color='black'>{controller_or_message}</font>")
            self.status_timer.start(5000)
        # 1: an error occurred (error will be displayed via `error_status`)
        elif exit_reason == 2:
            text = "Task '{}' aborted.".format(self.running_task)
            if self.dark_mode:
                self.status_label.setText(f"<font color='white'>{controller_or_message}</font>")
            else:
                self.status_label.setText(f"<font color='black'>{controller_or_message}</font>")
            self.status_timer.start(5000)

        # If the task was aborted or failed, and the "controller" object has a
        # saved state (e.g. the position of the pipette), ask the user whether
        # they want to reset the state
        if (exit_reason != 0 and
                controller_or_message is not None and
                controller_or_message.has_saved_state()):
            reply = QtWidgets.QMessageBox.question(self, "Reset",
                                                   controller_or_message.saved_state_question,
                                                   QtWidgets.QMessageBox.Yes |
                                                   QtWidgets.QMessageBox.No)
            if reply == QtWidgets.QMessageBox.Yes:
                _, reset_signal = self.interface_signals[self.running_task_interface]
                reset_signal.emit(controller_or_message)

        self.running_task = None
        self.running_task_interface = None


    def keyPressEvent(self, event):
        """
        Handles key press events and dispatches registered commands.

        Args:
            event (QKeyEvent): Key press event.
        """
        # We remove the keypad modifier, since we do not want to make a
        # difference between key presses as part of the keypad or on the main
        # keyboard (e.g. for the +/- keys). Most importantly, arrow keys always
        # use the keypad modifier on OS X.
        modifiers = event.modifiers() & ~Qt.KeypadModifier

        # Look for an exact match first (key + modifier)
        event_tuple = (event.key(), int(modifiers))
        description = self.key_actions.get(event_tuple, None)
        # If not found, check for keys that ignore the modifier
        if description is None:
            description = self.key_actions.get((event.key(), None), None)

        if description is not None:
            command, argument = description
            if self.running_task and not command.category == 'General':
                # Another task is running, ignore the key press
                # (we allow the "General" category to still allow to see the
                # help, etc.)
                return
            if command.is_blocking:
                self.start_task(command.task_description, command.__self__)
            if command.__self__ in self.interface_signals:
                command_signal, _ = self.interface_signals[command.__self__]
                command_signal.emit(command, argument)
            else:
                command(argument)

    @command(category='General',
             description='Toggle display of keyboard/mouse commands')
    def help_keypress(self):
        """Toggles the help window via keypress."""
        self.help_button.click()

    def toggle_help(self):
        """Shows or hides the help window based on toggle state."""
        if self.help_button.isChecked():
            self.help_window.show()
            # We need to keep the focus
            self.setFocus()
        else:
            self.help_window.setVisible(False)

    @command(category='General',
             description='Toggle display of log output')
    def log_keypress(self):
        """Toggles the log window via keypress."""
        self.log_button.click()

    def toggle_log(self):
        """Shows or hides the log window based on toggle state."""
        if self.log_button.isChecked():
            self.log_window.setVisible(True)
            # We need to keep the focus
            self.setFocus()
        else:
            self.log_window.setVisible(False)

    @QtCore.pyqtSlot('QString', 'QString')
    def set_status_message(self, category, message):
        """
        Updates or removes a categorized status message in the status bar.

        Args:
            category (str): Category of the message.
            message (str or None): Message text or None to remove
        """
        if message is None and category in self.status_messages:
            del self.status_messages[category]
        else:
            self.status_messages[category] = message

        messages = ' | '.join(self.status_messages.values())
        self.status_label.setText(messages)

    @QtCore.pyqtSlot(int, int)
    def splitter_size_changed(self, pos, index):
        """
        Handles splitter movement to update configuration panel visibility.

        Args:
            pos (int): Position of the splitter.
            index (int): Index of the splitter handle.
        """
        if not self.config_button:
            return  # nothing to do
        # If the splitter is moved all the way to the right, get back the focus
        if self.splitter.sizes()[1] == 0:
            self.setFocus()
            self.config_button.setChecked(False)
        else:
            self.config_button.setChecked(True)

    def add_config_gui(self, config, config_tab, gui_class=None):
        """
        Adds a configuration GUI tab for the given config object.

        Args:
            config (Config): Configuration object.
        """
        logging.debug('Adding config GUI for {}'.format(config.name))
        config_gui = ConfigGui(config) if gui_class is None else gui_class(config)
        config_tab.addTab(config_gui, config.name)
        logging.debug('Config GUI added')
        return config_gui

    def add_tab(self, tab, name, config_tab, index=None):
        """
        Adds a new tab to the configuration panel.

        Args:
            tab (QWidget): Tab widget to add.
            name (str): Tab label.
            index (int, optional): Position to insert the tab.
        """
        if index is None:
            config_tab.addTab(tab, name)
        else:
            config_tab.insertTab(index, tab, name)
        config_gui = ParamConfig(config) if gui_class is None else gui_class(config)
        self.add_tab(config_gui, config.name)
        logging.debug('Config GUI added')
        return config_gui

    def add_tab(self, tab, name, index=None):
        scroll = QtWidgets.QScrollArea()
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setWidget(tab)
        if index is None:
            self.config_tab.addTab(scroll, name)
        else:
            self.config_tab.insertTab(index, scroll, name)

    @command(category='General',
             description='Show/hide the configuration pane')
    def configuration_keypress(self):
        """Toggles the configuration panel via keypress."""
        self.config_button.click()

    @command(category='General',
             description='Show/hide the overlay information on the image')
    def toggle_overlay(self):
        """Toggles overlay display on the camera image."""
    def toggle_overlay(self, checked=None):
        self.show_overlay = not self.show_overlay
        self.overlay_button.setChecked(self.show_overlay)

    def toggle_configuration_display(self):
        """Shows or hides the configuration panel by adjusting splitter sizes."""
        if self.config_scroll_area.isVisible():
            self.config_scroll_area.hide()
            self.setFocus()
            self.config_button.setChecked(False)
        else:
            self.config_scroll_area.show()
            self.config_button.setChecked(True)      
            current_sizes = self.splitter.sizes()      
            target_width = max(self.current_tab.sizeHint().width(), 250)
            new_sizes = [current_sizes[0] - target_width, target_width]   
            self.splitter.setSizes(new_sizes)

    def set_active_pipette_camera(self, camera_id):
        """
        Select which pipette camera is represented by the auxiliary view.
        """
        if camera_id not in self.pipette_cameras:
            logging.warning(
                "Unknown pipette camera id: %s",
                camera_id,
            )
            return False

        if camera_id == self.active_pipette_camera_id:
            return True

        self.active_pipette_camera_id = camera_id

        camera = self.selected_pipette_camera
        interface = self.selected_pipette_camera_interface

        # Retarget the single pipette display widget.
        if self.pipette_video is not None:
            self.pipette_video.set_camera_metadata(
                camera
            )

        # If we're currently viewing a pipette camera,
        # immediately switch active control to the newly
        # selected pipette.
        if self.active_camera_role == "pipette":
            self.active_camera = camera
            self.camera = camera

            self.active_interface = interface
            self.camera_interface = interface

            self.active_video = self.pipette_video

        self._rebind_camera_key_actions()
        self._sync_interface_activity()

        return True

    def set_active_pipette_camera_index(self, index):
        camera_ids = list(self.pipette_cameras.keys())

        if not 0 <= index < len(camera_ids):
            return False

        return self.set_active_pipette_camera(
            camera_ids[index]
        )

    @property
    def selected_pipette_camera(self):
        if self.active_pipette_camera_id is None:
            return None

        return self.pipette_cameras.get(
            self.active_pipette_camera_id
        )


    @property
    def selected_pipette_camera_interface(self):
        if self.active_pipette_camera_id is None:
            return None

        return self.pipette_camera_interfaces.get(
            self.active_pipette_camera_id
        )

class ElidedLabel(QtWidgets.QLabel):
    """
    A QLabel subclass that truncates text with an ellipsis if it exceeds
    the available display width.

    Ensures long text strings fit within constrained UI layouts.
    """
    def __init__(self, text, minimum_width=200, *args, **kwds):
        """
        Initializes an ElidedLabel with text truncation behavior.

        Args:
            text (str): Full label text.
            minimum_width (int, optional): Minimum width before eliding.
            *args: Additional positional arguments.
            **kwds: Additional keyword arguments.
        """
        self.minimum_width = minimum_width
        self.text = text
        super(ElidedLabel, self).__init__(*args, **kwds)

    def minimumSizeHint(self):
        """
        Provides the minimum size hint for the label.

        Returns:
            QSize: Minimum size hint.
        """
        return QtCore.QSize(self.minimum_width,
                            super(ElidedLabel, self).minimumSizeHint().height())

    def resizeEvent(self, event):
        """
        Handles resizing and updates displayed text with elision if needed.

        Args:
            event (QResizeEvent): Resize event.
        """
        metric = QtGui.QFontMetrics(self.font())
        elidedText = metric.elidedText(self.text, QtCore.Qt.ElideRight,
                                       self.width())
        self.setText(elidedText)


class ConfigGui(QtWidgets.QWidget):
    """
    A graphical user interface for displaying and interacting with camera feeds.

    Supports multiple cameras, live video display, recording, user input
    handling (keyboard/mouse), and integration with camera control interfaces.
    """
    value_changed_signal = QtCore.pyqtSignal('QString', object)

    def __init__(self, config, show_name=False):
        """
        Initializes the configuration GUI with editable parameter widgets.

        Args:
            config (Config): Configuration object.
            show_name (bool, optional): Whether to display config name.
        """
        super(ConfigGui, self).__init__()
        self.config = config
        self.config._value_changed = self.value_changed
        self.value_changed_signal.connect(self.display_changed_value)
        layout = QtWidgets.QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        top_row = QtWidgets.QHBoxLayout()
        if show_name:
            self.title = QtWidgets.QLabel(config.name)
            self.title.setStyleSheet('font-weight: bold;')
            top_row.addWidget(self.title)
        else:
            top_row.setAlignment(Qt.AlignRight)
        self.load_button = QtWidgets.QToolButton(clicked=self.load_config)
        self.load_button.setIcon(qta.icon('fa.upload'))
        top_row.addWidget(self.load_button)
        self.save_button = QtWidgets.QToolButton(clicked=self.save_config)
        self.save_button.setIcon(qta.icon('fa.download'))
        top_row.addWidget(self.save_button)
        layout.addLayout(top_row)
        all_params = config.param
        self.value_widgets = {}
        for category, params in config.categories:
            box = QtWidgets.QGroupBox(category)
            rows = QtWidgets.QVBoxLayout()
            for param_name in params:
                param_obj = all_params[param_name]
                row = QtWidgets.QHBoxLayout()
                label = ElidedLabel(param_obj.doc)
                label.setToolTip(param_obj.doc)
                if isinstance(param_obj, param.Number):
                    value_widget = QtWidgets.QDoubleSpinBox()
                    value_widget.setMinimum(param_obj.bounds[0])
                    value_widget.setMaximum(param_obj.bounds[1])
                    value_widget.setValue(getattr(config, param_name))
                    value_widget.valueChanged.connect(
                        functools.partial(self.set_numerical_value, param_name))
                if isinstance(param_obj, NumberWithUnit):
                    value_widget = QtWidgets.QDoubleSpinBox()
                    magnitude = param_obj.magnitude
                    value_widget.setMinimum(param_obj.bounds[0] / magnitude)
                    value_widget.setMaximum(param_obj.bounds[1] / magnitude)
                    value_widget.setValue(getattr(config, param_name) / magnitude)
                    value_widget.valueChanged.connect(
                        functools.partial(self.set_numerical_value_with_unit, param_name, magnitude))
                elif isinstance(param_obj, param.Boolean):
                    value_widget = QtWidgets.QCheckBox()
                    value_widget.setChecked(getattr(config, param_name))
                    value_widget.stateChanged.connect(
                        functools.partial(self.set_boolean_value, param_name, value_widget))
                elif isinstance(param_obj, (param.Selector)):
                    value_widget = QtWidgets.QComboBox()
                    value_widget.addItems([str(o) for o in param_obj.objects])
                    current = getattr(config, param_name)
                    if current in param_obj.objects:
                        value_widget.setCurrentIndex(param_obj.objects.index(current))
                    value_widget.currentIndexChanged.connect(
                        functools.partial(self.set_selector_value, param_name, param_obj.objects))
                elif isinstance(param_obj, param.String):
                    value_widget = QtWidgets.QLineEdit()
                    value_widget.setText(str(getattr(config, param_name)))
                    value_widget.textChanged.connect(
                        functools.partial(self.set_string_value, param_name))
                elif isinstance(param_obj, param.Tuple):         
                    value_widget = QtWidgets.QLineEdit()          
                    value_widget.setReadOnly(True)                
                    value_widget.setEnabled(False)               
                    value_widget.setText(str(getattr(config, param_name))) 
                value_widget.setToolTip(param_obj.doc)
                value_widget.setObjectName(param_name)
                self.value_widgets[param_name] = value_widget
                row.addWidget(label, stretch=1)
                row.addWidget(value_widget)
                if isinstance(param_obj, NumberWithUnit):
                    unit_label = QtWidgets.QLabel(param_obj.unit)
                    row.addWidget(unit_label)
                rows.addLayout(row)
            box.setLayout(rows)
            layout.addWidget(box)
        self.setLayout(layout)

    def value_changed(self, key, value):
        """Relay parameter updates coming from the Config object.
        Numeric parameters are scaled by their unit magnitude; non‑numeric
        (e.g. Selector / Boolean) are forwarded unchanged.
        
        Args:
            key (str): Parameter name.
            value (object): Updated value.
        """
        if key not in self.value_widgets:
            return

        param_obj  = self.config.param[key]
        magnitude  = getattr(param_obj, 'magnitude', 1)

        # Only scale numeric values; leave strings / bools intact
        if isinstance(value, (int, float)):
            self.value_changed_signal.emit(key, value / magnitude)
        else:
            self.value_changed_signal.emit(key, value)

    @QtCore.pyqtSlot('QString', object)
    def display_changed_value(self, key, value):
        """
        Updates GUI widgets to reflect new parameter values.

        Args:
            key (str): Parameter name.
            value (object): New value.
        """
        box = self.findChild(QtWidgets.QCheckBox, key)
        if box is not None:
            box.blockSignals(True)
            box.setChecked(bool(value))
            box.blockSignals(False)
            return
        combo = self.findChild(QtWidgets.QComboBox, key)
        if combo is not None:
            index = combo.findText(str(value))
            if index >= 0:
                combo.blockSignals(True)
                combo.setCurrentIndex(index)
                combo.blockSignals(False)
            return
        spin = self.findChild((QtWidgets.QDoubleSpinBox, QtWidgets.QSpinBox), key)
        if spin is not None:
            spin.blockSignals(True)
            spin.setValue(value) 
            spin.blockSignals(False)
            return                                             # (unchanged)

        line = self.findChild(QtWidgets.QLineEdit, key)        
        if line is not None:             
            line.blockSignals(True)                            
            line.setText(str(value))                           
            line.blockSignals(False)                           


    def set_numerical_value(self, name, value):
        """
        Sets a numerical parameter value in the config.

        Args:
            name (str): Parameter name.
            value (float): Value to set.
        """
        setattr(self.config, name, value)

    def set_numerical_value_with_unit(self, name, magnitude, value):
        """
        Sets a numerical parameter value with unit scaling.

        Args:
            name (str): Parameter name.
            magnitude (float): Unit scaling factor.
            value (float): Value to set (pre-scaled).
        """
        setattr(self.config, name, value * magnitude)

    def set_boolean_value(self, name, widget):
        """
        Sets a boolean parameter value based on widget state.

        Args:
            name (str): Parameter name.
            widget (QCheckBox): Checkbox widget.
        """
        setattr(self.config, name, widget.isChecked())

    def set_selector_value(self, name, options, index):
        """
        Sets a selector parameter value based on selected index.

        Args:
            name (str): Parameter name.
            options (list): Available options.
            index (int): Selected index.
        """
        if 0 <= index < len(options):
            setattr(self.config, name, options[index])

    def set_string_value(self, name, value):
        """
        Sets a string parameter value in the config.

        Args:
            name (str): Parameter name.
            value (str): Value to set.
        """
        setattr(self.config, name, value)


    def save_config(self):
        """
        Saves the current configuration to a file.

        Raises:
            Exception: If saving fails.
        """
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save configuration",
                                                            filter='Configuration files (*.yaml)',
                                                            options=QtWidgets.QFileDialog.DontUseNativeDialog)
        if filename:
            try:
                self.config.to_file(filename)
            except Exception as ex:
                err = f'Could not save configuration to file "{filename}"'
                logging.getLogger(__name__).exception(err)
                QtWidgets.QMessageBox.warning(self, 'Saving failed', err + '\n' + str(ex), QtWidgets.QMessageBox.Ok)

    def load_config(self):
        """
        Loads configuration values from a file.

        Raises:
            Exception: If loading fails.
        """
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load configuration",
                                                            filter='Configuration files (*.yaml)',
                                                            options=QtWidgets.QFileDialog.DontUseNativeDialog)
        if filename:
            try:
                self.config.from_file(filename)
            except Exception as ex:
                err = f'Could not load configuration from file "{filename}"'
                logging.getLogger(__name__).exception(err)
                QtWidgets.QMessageBox.warning(self, 'Loading failed', err + '\n' + str(ex), QtWidgets.QMessageBox.Ok)
# Compatibility exports for existing camera-module imports.
from .ParamConfig import ParamConfig, ElidedLabel

ConfigGui = ParamConfig  # Compatibility for pre-extraction imports.
