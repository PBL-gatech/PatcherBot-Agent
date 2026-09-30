# coding=utf-8
from __future__ import absolute_import

import collections
import ast
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
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QDialog, QPushButton, QDialogButtonBox, QHBoxLayout, QVBoxLayout,
                             QLabel, QLineEdit, QStyle, QFileDialog, QSpinBox)
import qtawesome as qta

from patcherbot.interface.camera import CameraInterface
from patcherbot.controller import TaskController
from patcherbot.configs.Config import NumberWithUnit
from patcherbot.interface.base import command
from .livefeed import LiveFeedQt


class Logger(QtCore.QAbstractTableModel, logging.Handler):
    def __init__(self):
        super(Logger, self).__init__()
        # We do not actually use the formatter, but the asctime attribute is
        # available if the formatter requires it
        self.setFormatter(logging.Formatter('%(asctime)s'))
        self.messages = []

    def emit(self, record):
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
        return len(self.messages)

    def columnCount(self, parent=None):
        return 4

    def headerData(self, section, orientation, role):
        if role != Qt.DisplayRole:
            return
        if orientation == Qt.Horizontal:
            return ['', 'time', 'origin', 'message'][section]

    def data(self, index, role):
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
    close_signal = QtCore.pyqtSignal()
    levels = collections.OrderedDict([('DEBUG',logging.DEBUG),
                                      ('INFO', logging.INFO),
                                      ('WARN', logging.WARN),
                                      ('ERROR', logging.ERROR)])

    def __init__(self, parent):
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
        self.close_signal.emit()
        super(LogViewerWindow, self).closeEvent(event)

    def set_level(self, level_idx):
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

    close_signal = QtCore.pyqtSignal()

    def __init__(self, parent):
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
        # Forward key presses to the parent window
        return self.parent().keyPressEvent(event)

    def register_key_action(self, key, modifier, category, description):
        if category not in self.key_catalog:
            self.key_catalog[category] = []
        self.key_catalog[category].append((key, modifier, description))
        self.update_text()

    def register_mouse_action(self, click_type, modifier, category, description):
        if category not in self.mouse_catalog:
            self.mouse_catalog[category] = []
        self.mouse_catalog[category].append((click_type, modifier, description))
        self.update_text()

    def register_custom_action(self, category, action, description):
        if category not in self.custom_catalog:
            self.custom_catalog[category] = []
        self.custom_catalog[category].append((action, description))

    def update_text(self):
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
        lines = ['<tr>',
                 '<td style="font-family: monospace; font-weight: bold; align: center; padding-right: 1ex">{}</td>'
                 '<td>{}</td>'.format(action, description),
                 '</tr>']
        return lines

    def closeEvent(self, event):
        self.close_signal.emit()
        super(KeyboardHelpWindow, self).closeEvent(event)


class LogNotifyHandler(logging.Handler):
    def __init__(self, signal):
        super(LogNotifyHandler, self).__init__()
        self.signal = signal

    def emit(self, record):
        self.format(record)
        if record.exc_info is None:
            message = record.msg
        else:
            _, exc, _ = record.exc_info
            message = '{} ({})'.format(record.msg, str(exc))
        self.signal.emit(message)

class RecordingDialog(QDialog):
    def __init__(self, base_directory, frame_rate, pixels, settings, parent=None):
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
        self.prefix_preview.setText('<i>{}_00000.tiff</i>'.format(self.prefix_edit.text()))

    def skip_edited(self, value):
        if self.frame_rate > 0:
            rate = '~{:.1f}'.format(self.frame_rate / (value + 1))
        else:
            rate = '?'
        self.frame_rate_label.setText('<i>{} frames per second</i>'.format(rate))

    def memory_edited(self, value):
        self.file_queue_frames.setText('<i>space for ~{} frames in queue'.format(int(value*1e6/self.pixels)))

    def directory_clicked(self):
        folder = self.select_folder()
        if folder is not None:
            print(folder, folder)
            self.directory_edit.setText(folder)

    def select_folder(self):
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.ShowDirsOnly)
        dialog.setWindowTitle('Select recording directory')
        dialog.setDirectory(self.directory_edit.text())
        if dialog.exec_():
            return dialog.selectedFiles()[0]


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
    aux_camera_signal = QtCore.pyqtSignal(MethodType, object)
    aux_camera_reset_signal = QtCore.pyqtSignal(TaskController)
    snapshot_captured = QtCore.pyqtSignal(object)


    def __init__(self, camera, aux_camera=None, recording_state_manager=None,
                 image_edit=None, display_edit=None,
                 with_tracking=False, base_directory='.'):
        super().__init__()
        self.main_camera = camera
        self.aux_camera = aux_camera
        self.recording_state_manager = recording_state_manager
        if self.recording_state_manager is None:
            raise ValueError("RecordingStateManager must be provided")
        self.base_directory = base_directory
        self.with_tracking = with_tracking
        self.show_overlay = True
        self.is_recording = False

        self.main_interface = None
        if self.main_camera is not None:
            self.main_interface = CameraInterface(self.main_camera,
                                                  with_tracking=with_tracking,
                                                  status_category='Main Camera')
        self.aux_interface = None
        if self.aux_camera is not None:
            self.aux_interface = CameraInterface(self.aux_camera,
                                                 with_tracking=False,
                                                 status_category='Aux Camera')

        self.active_camera_role = 'main' if self.main_camera is not None else 'aux'
        self.active_camera = self.main_camera if self.active_camera_role == 'main' else self.aux_camera
        self.camera = self.active_camera
        self.active_interface = self.main_interface if self.active_camera_role == 'main' else self.aux_interface
        self.camera_interface = self.active_interface

        self.status_bar = QtWidgets.QStatusBar()
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
        self.switch_view_button.setEnabled(self.aux_camera is not None)
        self._update_switch_button_text()

        self.setexposure_edit = QtWidgets.QLineEdit()
        self.setexposure_edit.setMaximumWidth(200)
        self.setexposure_edit.setPlaceholderText('Exposure time (ms)')
        # convert the text to a float and set the exposure time if the user presses enter and clear the text box
        self.setexposure_edit.returnPressed.connect(self.apply_active_exposure)

        self.status_bar.addPermanentWidget(self.switch_view_button)
        self.status_bar.addPermanentWidget(self.setexposure_edit)
        self.status_bar.addPermanentWidget(self.help_button)
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
                                         recording_state_manager=self.recording_state_manager,
                                         frame_folder_name='camera_frames')
        self.aux_video = None
        if self.aux_camera is not None:
            self.aux_video = LiveFeedQt(self.aux_camera,
                                        image_edit=self.image_edit,
                                        display_edit=functools.partial(self.display_edit, camera=self.aux_camera),
                                        mouse_handler=self.video_mouse_press,
                                        recording_state_manager=self.recording_state_manager,
                                        frame_folder_name='aux_camera_frames')

        self.video_stack = QtWidgets.QStackedWidget()
        if self.main_video is not None:
            self.video_stack.addWidget(self.main_video)
        if self.aux_video is not None:
            self.video_stack.addWidget(self.aux_video)
        self.active_video = self.main_video if self.active_camera_role == 'main' else self.aux_video
        if self.active_video is not None:
            self.video_stack.setCurrentWidget(self.active_video)

        self.recording_settings = {}
        self.setFocus()  # Need this to handle arrow keys, etc.
        self.interface_signals = {}
        if self.main_interface is not None:
            self.interface_signals[self.main_interface] = (self.camera_signal,
                                                           self.camera_reset_signal)
        if self.aux_interface is not None:
            self.interface_signals[self.aux_interface] = (self.aux_camera_signal,
                                                          self.aux_camera_reset_signal)

        self.splitter = QtWidgets.QSplitter()
        self.splitter.addWidget(self.video_stack)
        self.config_tab = QtWidgets.QTabWidget()
        self.splitter.addWidget(self.config_tab)
        self.setCentralWidget(self.splitter)
        self.splitter.setSizes([1, 0])
        self.splitter.splitterMoved.connect(self.splitter_size_changed)

        # Display error messages directly in the status bar
        handler = LogNotifyHandler(self.log_signal)
        handler.setLevel(logging.ERROR)
        logging.getLogger('patcherbot').addHandler(handler)
        self.log_signal.connect(self.error_status)

    def _update_switch_button_text(self):
        if not hasattr(self, 'switch_view_button') or self.switch_view_button is None:
            return
        if self.aux_camera is None:
            self.switch_view_button.setText('Switch View')
            return
        if self.active_camera_role == 'main':
            self.switch_view_button.setText('Switch to Pipette View')
        else:
            self.switch_view_button.setText('Switch to Microscope View')

    def normalize_active_camera(self):
        if self.active_interface is None:
            return
        self.active_interface.normalize()

    def unnormalize_active_camera(self):
        if self.active_interface is None:
            return
        self.active_interface.unnormalize()

    def snap_active_camera_image(self):
        if self.active_interface is None:
            return
        snapshot = self.active_interface.snap_image()
        if snapshot is None:
            return
        snapshot = dict(snapshot)
        snapshot["camera_role"] = self.active_camera_role
        self.snapshot_captured.emit(snapshot)
        return snapshot

    def handle_autonormalize_change(self, state):
        if self.active_interface is None:
            return
        self.active_interface.autonormalize(bool(state))

    def apply_active_exposure(self):
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
        if self.aux_camera is None:
            return
        target_role = 'aux' if self.active_camera_role == 'main' else 'main'
        self._set_active_camera(target_role)

    def _set_active_camera(self, role):
        if role not in ('main', 'aux'):
            return
        if role == 'main' and self.main_camera is None:
            return
        if role == 'aux' and self.aux_camera is None:
            return
        if role == self.active_camera_role:
            return
        self.active_camera_role = role
        self.active_camera = self.main_camera if role == 'main' else self.aux_camera
        self.camera = self.active_camera
        self.active_interface = self.main_interface if role == 'main' else self.aux_interface
        self.camera_interface = self.active_interface
        self.active_video = self.main_video if role == 'main' else self.aux_video
        if self.active_video is not None:
            self.video_stack.setCurrentWidget(self.active_video)
        self._update_switch_button_text()
        self._rebind_camera_key_actions()
        self._sync_interface_activity()

    def _sync_interface_activity(self):
        if self.main_interface is not None:
            self.main_interface.set_active(self.active_camera_role == 'main')
        if self.aux_interface is not None:
            self.aux_interface.set_active(self.active_camera_role == 'aux')

    def _rebind_camera_key_actions(self):
        if self.active_interface is None:
            return
        for key, modifier, method_name, argument, _ in self._camera_key_bindings:
            command = getattr(self.active_interface, method_name, None)
            if command is None:
                continue
            self.register_key_action(key, modifier, command, argument, default_doc=False)

    def register_camera_key_action(self, key, modifier, method_name, argument=None, default_doc=True):
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
        return getattr(self, 'active_video', None)

    # Add a cross to the display
    def draw_cross(self, pixmap):
        '''
        Draws a cross at the center. Meant to be used as a ``display_edit``
        function.

        Parameters
        ----------
        pixmap : `QPixmap`
            The pixmap to draw on.
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

        Parameters
        ----------
        pixmap : `QPixmap`
            The pixmap to draw on.
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

        Parameters
        ----------
        pixmap : `QPixmap`
            The pixmap to draw on.
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

        Parameters
        ----------
        image : `~numpy.ndarray`
            The original video image  or the image returned by a previously
            called function.

        Returns
        -------
        new_image : `~numpy.ndarray`
            The post-processed image. Should be of the same size and data type
            as the original image.
        '''
        for func in self.image_edit_funcs:
            image = func(image)
        return image

    def closeEvent(self, evt):
       self.close()
       return super(CameraGui, self).closeEvent(evt)

    @command(category='General',
             description='Exit the application')
    def exit(self):
        self.close()

    @command(category='Camera',
             description='Toggle recording image files to disk')
    def toggle_recording(self, *args):
        if self.is_recording:
            for cam in filter(None, [self.main_camera, self.aux_camera]):
                stop_method = getattr(cam, 'stop_recording', None)
                if stop_method is not None:
                    stop_method()
            if self.recording_state_manager is not None:
                self.recording_state_manager.set_recording(False)
            if self.main_video is not None:
                self.main_video.recorder.handle_recording_stopped()
            if self.aux_video is not None:
                self.aux_video.recorder.handle_recording_stopped()
            self.is_recording = False
        else:
            active_cam = self.active_camera or self.main_camera or self.aux_camera
            if active_cam is None:
                return
            frame_rate = getattr(active_cam, 'get_frame_rate', lambda: 0)()
            pixels = getattr(active_cam, 'width', 0) * getattr(active_cam, 'height', 0)
            dlg = RecordingDialog(self.base_directory, frame_rate=frame_rate,
                                  pixels=pixels,
                                  settings=self.recording_settings, parent=self)
            if dlg.exec_():
                directory = os.path.abspath(dlg.directory_edit.text())
                prefix = dlg.prefix_edit.text()
                self.recording_settings['prefix'] = prefix
                memory = dlg.memory_spin.value()
                self.recording_settings['memory'] = memory
                skip_frames = dlg.skip_spin.value()
                self.recording_settings['skip_frames'] = skip_frames
                for cam in filter(None, [self.main_camera, self.aux_camera]):
                    width = getattr(cam, 'width', getattr(active_cam, 'width', 1))
                    height = getattr(cam, 'height', getattr(active_cam, 'height', 1))
                    queue_size = int(memory * 1e6 / (width * height)) + 1 if width and height else 1
                    start_method = getattr(cam, 'start_recording', None)
                    if start_method is not None:
                        start_method(directory=directory, file_prefix=prefix,
                                     skip_frames=skip_frames, queue_size=queue_size)
                if self.recording_state_manager is not None:
                    self.recording_state_manager.set_recording(True)
                self.is_recording = True
        self.record_button.setChecked(self.is_recording)

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
        '''
        Close the GUI.
        '''
        logging.info('closing GUI')
        for attr_name in ('main_camera', 'aux_camera'):
            cam = getattr(self, attr_name, None)
            if cam is None:
                continue
            stop_acq = getattr(cam, 'stop_acquisition', None)
            if stop_acq is not None:
                stop_acq()
            stop_recording = getattr(cam, 'stop_recording', None)
            if stop_recording is not None:
                stop_recording()
            close_cam = getattr(cam, 'close', None)
            if close_cam is not None:
                close_cam()
            setattr(self, attr_name, None)
        for video_attr in ('main_video', 'aux_video'):
            video_widget = getattr(self, video_attr, None)
            if video_widget is not None:
                video_widget.recorder.close()
        if self.recording_state_manager is not None:
            self.recording_state_manager.set_recording(False)
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

        Parameters
        ----------
        click_type : `.Qt.MouseButton`
            The type of click that should be handled as a ``Qt`` constant, e.g.
            `.Qt.LeftButton` or `.Qt.RightButton`.
        modifier : `.Qt.Modifer` or ``None``
            The modifier that needs to be pressed at the same time to trigger
            the action. The modifier needs to be given as a ``Qt`` constant,
            e.g. `.Qt.ShiftModifier` or `.Qt.ControlModifier`. Alternatively,
            ``None`` can be used to specify that the mouse click should lead to
            the action independent of the modifier.
        command : method
            A method implementing the action that has been annotated with the
            `@command <.command>` or `@blocking_command <.blocking_command>`
            decorator.
        default_doc : bool, optional
            Whether to include the action in the automatically generated help.
            Defaults to ``True``.
        '''
        self.mouse_actions[(click_type, modifier)] = command
        if default_doc:
            self.help_window.register_mouse_action(click_type, modifier,
                                                   command.category,
                                                   command.auto_description())

    def video_mouse_press(self, event):
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
        if not message:
            self.status_bar.setStyleSheet('QStatusBar{color: black;}')

    @QtCore.pyqtSlot('QString')
    def error_status(self, message):
        self.status_bar.setStyleSheet('QStatusBar{color: red;}')
        self.status_bar.showMessage(message, 5000)

    def initialize(self):
        for interface, (command_signal, reset_signal) in self.interface_signals.items():
            command_signal.connect(interface.command_received)
            reset_signal.connect(interface.reset_requested)
            interface.task_finished.connect(self.task_finished)
            interface.connect(self)
        self._sync_interface_activity()
        self.register_commands()
        # Add a button for the configuration options if necessary
        if self.config_tab.count() > 0:
            self.config_button = QtWidgets.QToolButton(
                clicked=self.toggle_configuration_display)
            self.config_button.setIcon(qta.icon('fa.cogs'))
            self.config_button.setCheckable(True)
            self.status_bar.addPermanentWidget(self.config_button)

    def register_key_action(self, key, modifier, command, argument=None,
                            default_doc=True):
        '''
        Link a keypress to an action.

        Parameters
        ----------
        key : `.Qt.Key`
            The key that should be handled, specified as a ``Qt`` constant, e.g.
            `.Qt.Key_X` or `.Qt.Key_5`.
        modifier : `.Qt.Modifer` or ``None``
            The modifier that needs to be pressed at the same time to trigger
            the action. The modifier needs to be given as a ``Qt`` constant,
            e.g. `.Qt.ShiftModifier` or `.Qt.ControlModifier`. Alternatively,
            ``None`` can be used to specify that the keypress should lead to
            the action independent of the modifier.
        command : method
            A method implementing the action that has been annotated with the
            `@command <.command>` or `@blocking_command <.blocking_command>`
            decorator.
        argument : object, optional
            An additional argument that should be handled to the method defined
            as ``command``. Can be used to re-use the same action in a
            parametrized way (e.g. steps of different size).
        default_doc : bool, optional
            Whether to include the action in the automatically generated help.
            Defaults to ``True``.
        '''
        self.key_actions[(key, modifier)] = (command, argument)
        if default_doc:
            self.help_window.register_key_action(key, modifier,
                                                 command.category,
                                                 command.auto_description(argument))

    def start_task(self, task_name, interface):
        self.status_bar.clearMessage()
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
        self.task_abort_button.setEnabled(False)
        self.task_success_button.setEnabled(False)
        self.running_task_interface.abort_task()

    @QtCore.pyqtSlot(int, object)
    def task_finished(self, exit_reason, controller_or_message):
        if self.running_task is None:
            # This might be a success message for a non-blocking command
            if isinstance(controller_or_message, str):
                self.status_bar.setStyleSheet('QStatusBar{color: black;}')
                self.status_bar.showMessage(controller_or_message, 1000)
            return  # Nothing else to do

        self.task_progress.setVisible(False)
        self.task_abort_button.setVisible(False)
        self.task_success_button.setVisible(False)
        self.task_success_button.setEnabled(False)
        # 0: correct execution (no need to show a message)
        if exit_reason == 0:
            text = "Task '{}' finished successfully.".format(self.running_task)
            self.status_bar.setStyleSheet('QStatusBar{color: black;}')
            self.status_bar.showMessage(text, 5000)
        # 1: an error occurred (error will be displayed via `error_status`)
        elif exit_reason == 2:
            text = "Task '{}' aborted.".format(self.running_task)
            self.status_bar.setStyleSheet('QStatusBar{color: black;}')
            self.status_bar.showMessage(text, 5000)

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
        self.help_button.click()

    def toggle_help(self):
        if self.help_button.isChecked():
            self.help_window.show()
            # We need to keep the focus
            self.setFocus()
        else:
            self.help_window.setVisible(False)

    @command(category='General',
             description='Toggle display of log output')
    def log_keypress(self):
        self.log_button.click()

    def toggle_log(self):
        if self.log_button.isChecked():
            self.log_window.setVisible(True)
            # We need to keep the focus
            self.setFocus()
        else:
            self.log_window.setVisible(False)

    @QtCore.pyqtSlot('QString', 'QString')
    def set_status_message(self, category, message):
        if message is None and category in self.status_messages:
            del self.status_messages[category]
        else:
            self.status_messages[category] = message

        messages = ' | '.join(self.status_messages.values())
        self.status_label.setText(messages)

    @QtCore.pyqtSlot(int, int)
    def splitter_size_changed(self, pos, index):
        if not self.config_button:
            return  # nothing to do
        # If the splitter is moved all the way to the right, get back the focus
        if self.splitter.sizes()[1] == 0:
            self.setFocus()
            self.config_button.setChecked(False)
        else:
            self.config_button.setChecked(True)

    def add_config_gui(self, config, gui_class=None):
        logging.debug('Adding config GUI for {}'.format(config.name))
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
        self.config_button.click()

    @command(category='General',
             description='Show/hide the overlay information on the image')
    def toggle_overlay(self, checked=None):
        self.show_overlay = not self.show_overlay
        self.overlay_button.setChecked(self.show_overlay)

    def toggle_configuration_display(self):
        current_sizes = self.splitter.sizes()
        if current_sizes[1] == 0:
            min_size = self.config_tab.sizeHint().width()
            new_sizes = [current_sizes[0]-min_size, min_size]
            self.config_button.setChecked(True)
        else:
            new_sizes = [current_sizes[0]+current_sizes[1], 0]
            self.setFocus()
            self.config_button.setChecked(False)
        self.splitter.setSizes(new_sizes)


# Compatibility exports for existing camera-module imports.
from .ParamConfig import ParamConfig, ElidedLabel

ConfigGui = ParamConfig  # Compatibility for pre-extraction imports.
