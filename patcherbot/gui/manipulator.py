# coding=utf-8
from types import MethodType
import time

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import numpy as np


from patcherbot.interface.pipettes import PipetteInterface
from patcherbot.utils.RecordingStateManager import RecordingStateManager
from patcherbot.controller import TaskController
from patcherbot.gui import CameraGui
from patcherbot.interface import command, blocking_command
from patcherbot.devices.manipulator.calibratedunit import CalibrationError
import datetime
import cv2

class PipetteSignals(QtCore.QObject):
    """
    A dedicated container to hold unique signals for a single pipette.
    This prevents multiple tabs from broadcasting on the same channel.
    """
    command = QtCore.pyqtSignal(MethodType, object) 
    reset = QtCore.pyqtSignal(TaskController)

class ManipulatorGui(CameraGui):
    """GUI for controlling a manipulator and its associated cameras."""
    pipette_command_signal = QtCore.pyqtSignal(MethodType, object)
    pipette_reset_signal = QtCore.pyqtSignal(TaskController)

    def __init__(self, camera, pipette_cameras, pipette_interfaces, with_tracking=False, recording_state_manager: RecordingStateManager = None):
        """
        Initialize the manipulator GUI.

        Attributes:
            microscope_camera: The main microscope camera.
            pipette_cameras: Auxiliary cameras for pipette view.
            pipette_interfaces: Manipulator interfaces for pipettes.
            control_thread: Thread in which the manipulator interfaces run.
            image_save_number (int): Counter for saved images.
            show_tip_on (bool): Whether the tip is currently being displayed.
            tip_x, tip_y: Coordinates of pipette tip.
            tip_t0: Timestamp when tip display started.
        """
        super(ManipulatorGui, self).__init__(camera, pipette_cameras=pipette_cameras, with_tracking=with_tracking, recording_state_manager=recording_state_manager)
        self.control_threads = {} # Keep track of multiple threads
        self._unique_pipette_signals = {}
        
        self.setWindowTitle("Pipette GUI")

        self.microscope_camera = camera
        self.pipette_camera = aux_camera
        for video in (self.main_video, self.aux_video):
            if video is not None:
                video.recording_config = self.interface.calibration_config

        self.interfaces = pipette_interfaces
        if not isinstance(self.interfaces, dict):
            self.control_threads["pipette"] = QtCore.QThread()
            self.interfaces.moveToThread(self.control_threads["pipette"])
            self.control_threads["pipette"].start()

            self.interface_signals[self.interfaces] = (self.pipette_command_signal,
                                                        self.pipette_reset_signal)
            self.active_pipette = self.interfaces
        else:
            for id, interface in self.interfaces.items():
                thread = QtCore.QThread()
                self.control_threads[id] = thread
                interface.moveToThread(thread)
                thread.start()

                signals = PipetteSignals()
                self._unique_pipette_signals[id] = signals
                self.interface_signals[interface] = (signals.command, signals.reset)

        self.active_pipette = list(self.interfaces.values())[0] if isinstance(self.interfaces, dict) else self.interfaces

        self.display_edit_funcs.append(self.draw_scale_bar)
        self.display_edit_funcs.append(self.display_manipulator)
        self.display_edit_funcs.append(self.show_tip)

        self.show_tip_on = False
        self.tip_x, self.tip_y = None, None
        self.tip_t0 = None

        # Stage position for display
        self._last_stage_measurement = None
        self._stage_position = (None, None, None)

        #number of images we've saved so far.  Allows images to have different names
        self.image_save_number = 0
        self.recording_state_manager = recording_state_manager
        if recording_state_manager is None:
            raise ValueError("RecordingStateManager must be provided")


    def display_manipulator(self, pixmap):
        '''
        Displays the number of the selected manipulator.

        Args:
            pixmap (QPixmap): The image to draw on.
        '''
        if getattr(self, 'active_camera_role', 'main') != 'main':
            return
        painter = QtGui.QPainter(pixmap)
        pen = QtGui.QPen(QtGui.QColor(200, 0, 0, 125))
        painter.setPen(pen)
        painter.setFont(QFont("Arial", int(pixmap.height()/20)))
        c_x, c_y = pixmap.width() *19.0 / 20, pixmap.height() * 19.0 / 20

    def draw_scale_bar(self, pixmap, text=True, autoscale=True,
                       position=True):
        """
        Draw a scale bar on the pixmap to indicate physical distances.

        Args:
            pixmap (QPixmap): Image to draw the scale bar on.
            text (bool, optional): Whether to show length text. Defaults to True.
            autoscale (bool, optional): Whether to automatically scale the bar. Defaults to True.
            position (bool, optional): Whether to consider position adjustments. Defaults to True.

        Raises:
            ValueError: If autoscale is True but text is False.
        """
        if getattr(self, 'active_camera_role', 'main') != 'main':
            return
        if autoscale and not text:
            raise ValueError('Automatic scaling of the bar without showing text '
                             'will not be very helpful...')
        stage = self.active_pipette.calibrated_stage
        camera_pixel_per_um = getattr(self.camera, 'pixel_per_um', None)
        if stage.calibrated or camera_pixel_per_um:
            pen_width = 4
            if camera_pixel_per_um is not None:
                bar_length = camera_pixel_per_um
            else:
                bar_length = stage.pixel_per_um()[0]
            scale = 1.0 * self.camera.width / pixmap.size().width()
            scaled_length = bar_length/scale
            if autoscale:
                lengths = np.array([1, 2, 5, 10, 20, 50, 100])
                if scaled_length*lengths[-1] < pen_width:
                    # even the longest bar is not long enough -- don't show
                    # any scale bar
                    return
                elif scaled_length*lengths[0] > 20*pen_width:
                    # the shortest bar is not short enough (>20x the width)
                    length_in_um = lengths[0]
                else:
                    # Use the length that gives a bar of about 10x its width
                    length_in_um = lengths[np.argmin(np.abs(scaled_length*lengths - 10*pen_width))]
            else:
                length_in_um = 10

            painter = QtGui.QPainter(pixmap)
            pen = QtGui.QPen(QtGui.QColor(0, 0, 0, 255))
            pen.setWidth(pen_width)
            painter.setPen(pen)
            c_x, c_y = pixmap.width() / 20, pixmap.height() * 19.0 / 20
            c_x = int(c_x)
            c_y = int(c_y)
            painter.drawLine(int(c_x), c_y,
                             int(c_x + round(length_in_um*scaled_length)), c_y)
            if text:
                painter.drawText(c_x, c_y - 10, '{}µm'.format(length_in_um))
            painter.end()

    def register_commands(self, manipulator_keys = True):
        """
        Register keyboard and mouse commands for controlling the manipulator,
        stage, and cameras.

        Args:
            manipulator_keys (bool, optional): Whether to register manipulator keybindings. Defaults to True.
        """
        super(ManipulatorGui, self).register_commands()

        if manipulator_keys:
            # Commands to move the stage
            # Note that we do not use the automatic documentation mechanism here,
            # as we one entry for every possible keypress
            modifiers = [Qt.NoModifier, Qt.AltModifier, Qt.ShiftModifier]
            distances = [0.4, 0.1, 1.0]
            self.help_window.register_custom_action('Stage',  'Arrows',
                                                    'Move stage')
            self.help_window.register_custom_action('Stage',
                                                    '/'.join(QtGui.QKeySequence(mod).toString()
                                                                 if mod is not Qt.NoModifier else 'No modifier'
                                                             for mod in modifiers),
                                                    'Move stage by ' + '/'.join(str(x) for x in distances) + ' µm')
            self.help_window.register_custom_action('Manipulators', 'A/S/W/D',
                                                    'Move pipette by in x/y direction')
            self.help_window.register_custom_action('Manipulators', 'Q/E',
                                                    'Move pipette by in z direction')
            self.help_window.register_custom_action('Manipulators',
                                                    '/'.join(QtGui.QKeySequence(mod).toString()
                                                                 if mod is not Qt.NoModifier else 'No modifier'
                                                             for mod in modifiers),
                                                    'Move pipette by ' + '/'.join(str(x) for x in distances) + ' µm')

            for modifier, distance in zip(modifiers, distances):
                self.register_key_action(Qt.Key_Up, modifier,
                                         self.active_pipette.move_stage_vertical,
                                         argument=-distance, default_doc=False)
                self.register_key_action(Qt.Key_Down, modifier,
                                         self.active_pipette.move_stage_vertical,
                                         argument=distance, default_doc=False)
                self.register_key_action(Qt.Key_Left, modifier,
                                         self.active_pipette.move_stage_horizontal,
                                         argument=-distance, default_doc=False)
                self.register_key_action(Qt.Key_Right, modifier,
                                         self.active_pipette.move_stage_horizontal,
                                         argument=distance, default_doc=False)
                self.register_key_action(Qt.Key_W, modifier,
                                         self.active_pipette.move_pipette_y,
                                         argument=distance, default_doc=False)
                self.register_key_action(Qt.Key_S, modifier,
                                         self.active_pipette.move_pipette_y,
                                         argument=-distance, default_doc=False)
                self.register_key_action(Qt.Key_A, modifier,
                                         self.active_pipette.move_pipette_x,
                                         argument=distance, default_doc=False)
                self.register_key_action(Qt.Key_D, modifier,
                                         self.active_pipette.move_pipette_x,
                                         argument=-distance, default_doc=False)
                self.register_key_action(Qt.Key_Q, modifier,
                                         self.active_pipette.move_pipette_z,
                                         argument=distance, default_doc=False)
                self.register_key_action(Qt.Key_E, modifier,
                                         self.active_pipette.move_pipette_z,
                                         argument=-distance, default_doc=False)

        # #save image command
        # self.register_key_action(Qt.Key_I, Qt.NoModifier,
        #                          self.save_image)

        # Show the tip
        self.register_key_action(Qt.Key_T, Qt.NoModifier,
                                 self.show_tip_switch)

        # Calibration commands
        self.register_key_action(Qt.Key_C, Qt.ControlModifier,
                                 self.active_pipette.calibrate_stage)
        self.register_key_action(Qt.Key_C, Qt.NoModifier,
                                 self.active_pipette.calibrate_manipulator)
        self.register_key_action(Qt.Key_F, Qt.ControlModifier,
                                 self.active_pipette.focus_pipette)

        # Move pipette by clicking
        self.register_mouse_action(Qt.LeftButton, Qt.ShiftModifier,
                                   self.active_pipette.move_pipette)

        # Move stage by clicking
        self.register_mouse_action(Qt.RightButton, Qt.NoModifier,
                                   self.active_pipette.move_stage)

        # Microscope control
        self.register_key_action(Qt.Key_PageUp, None,
                                 self.active_pipette.move_microscope,
                                 argument=10, default_doc=False)
        self.register_key_action(Qt.Key_PageDown, None,
                                 self.active_pipette.move_microscope,
                                 argument=-10, default_doc=False)
        key_string = (QtGui.QKeySequence(Qt.Key_PageUp).toString() + '/' +
                      QtGui.QKeySequence(Qt.Key_PageDown).toString())
        self.help_window.register_custom_action('Microscope', key_string,
                                                'Move microscope up/down by 10µm')
        self.register_key_action(Qt.Key_F, None,
                                 self.active_pipette.set_floor)
        self.register_key_action(Qt.Key_G, None,
                                 self.active_pipette.go_to_floor)

        # Show configuration pane
        self.register_key_action(Qt.Key_P, None,
                                 self.configuration_keypress)

        # Toggle overlays
        self.register_key_action(Qt.Key_O, None,
                                 self.toggle_overlay)

    @command(category='Manipulators',
             description='Show the tip of selected manipulator')
    def show_tip_switch(self):
        """
        Activate display of the pipette tip for one second.

        Raises:
            CalibrationError: If the manipulator unit is not calibrated.
        """
        try:
            self.tip_x, self.tip_y, _ = self.active_pipette.calibrated_unit.reference_position()
            self.tip_t0 = time.time()
            self.show_tip_on = True
        except CalibrationError:  # not yet calibrated
            return

    def show_tip(self, pixmap):
        """Draw supplied calibrated coordinates and the existing temporary tip marker."""
        if (getattr(self, 'active_camera_role', 'main') != 'main'
                or not getattr(self, 'show_overlay', True)):
            return
        width, height = self.camera.width, self.camera.height
        painter = QtGui.QPainter(pixmap)
        try:
            painter.scale(pixmap.width() / width, pixmap.height() / height)
            positions = getattr(self.interface, "display_positions", None)
            if (positions is not None
                    and positions["image_shape"] == (height, width)
                    and 0 <= time.monotonic() - positions["at"] <= 1.):
                point = np.asarray(positions["pipette_xy"], dtype=float).copy()
                direction = np.asarray(positions.get("pipette_direction_xy"), dtype=float).copy()
                if direction.shape != (2,) or not np.isfinite(direction).all():
                    direction = np.zeros(2)
                if self.camera.flipped:
                    point[0] = width - 1 - point[0]
                    direction[0] = -direction[0]
                length = np.linalg.norm(direction)
                # Match cell overlays: hide when the position leaves the image.
                if (np.isfinite(point).all() and np.isfinite(length) and length > 0
                        and 0 <= point[0] < width and 0 <= point[1] < height):
                    unit = direction / length
                    side = np.array([-unit[1], unit[0]])
                    pen = QtGui.QPen(QtGui.QColor("#ffba45"), 2)
                    pen.setCosmetic(True)
                    painter.setPen(pen)
                    painter.drawLine(QtCore.QPointF(*(point - 24 * unit)), QtCore.QPointF(*point))
                    painter.drawPolyline(QtGui.QPolygonF([
                        QtCore.QPointF(*(point - 9 * unit + 5 * side)), QtCore.QPointF(*point),
                        QtCore.QPointF(*(point - 9 * unit - 5 * side))]))
            if self.show_tip_on:
                if self.tip_x is not None and self.tip_y is not None:
                    x = width - 1 - self.tip_x if self.camera.flipped else self.tip_x
                    pen = QtGui.QPen(QtGui.QColor(0, 0, 200, 125), 3)
                    pen.setCosmetic(True)
                    painter.setPen(pen)
                    scale = width / pixmap.width()
                    painter.drawRect(QtCore.QRectF(x - 10 * scale, self.tip_y - 10 * scale,
                                                  10 * scale, 10 * scale))
                if time.time() > self.tip_t0 + 1.:
                    self.show_tip_on = False
        finally:
            painter.end()

    @command(category='Camera',
             description='Save the current image to the outputs folder')
    def save_image(self):
        """
        Capture and save the current image from the camera to the outputs folder.
        Image filenames are indexed using image_save_number.
        """
        #get the current image
        currImg = self.camera.get_16bit_image()

        #save the image
        cv2.imwrite(f'outputs/{self.image_save_number}.png', currImg)
        print(f'Saved image as outputs/{self.image_save_number}.png')
        self.image_save_number += 1


    def display_timer(self, pixmap):
        """
        Draw a runtime timer on the provided pixmap, based on interface timer.

        Args:
            pixmap (QPixmap): Image to draw the timer on.
        """
        interface = self.active_pipette
        painter = QtGui.QPainter(pixmap)
        pen = QtGui.QPen(QtGui.QColor(200, 0, 0, 125))
        pen.setWidth(1)
        painter.setPen(pen)
        c_x, c_y = pixmap.width() / 20, pixmap.height() / 20
        t = int(time.time() - interface(0).timer_t0)
        hours = t//3600
        minutes = (t-hours*3600)//60
        seconds = t-hours*3600-minutes*60
        painter.drawText(c_x, c_y, '{}'.format(datetime.time(hours,minutes,seconds)))
        painter.end()
    
# def make_dynamic_command(gui_instance, method_name):
#     """
#     Creates a proxy function that dynamically calls the active pipette's method,
#     but preserves the @command metadata required by the help menu generator.
#     """
#     def proxy_command(*args, **kwargs):
#         # Dynamically grab the method from whichever pipette is currently active
#         method = getattr(gui_instance.active_pipette, method_name)
#         return method(*args, **kwargs)
        
#     # Copy the decorator metadata from the first pipette to satisfy the GUI
#     sample_method = getattr(gui_instance.active_pipette, method_name)
#     if hasattr(sample_method, 'category'):
#         proxy_command.category = sample_method.category
#     if hasattr(sample_method, 'description'):
#         proxy_command.description = sample_method.description
#     if hasattr(sample_method, 'auto_description'):
#         proxy_command.auto_description = sample_method.auto_description
#     if hasattr(sample_method, 'is_blocking'):
#         proxy_command.is_blocking = sample_method.is_blocking
#     return proxy_command