from __future__ import print_function
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

import traceback
import numpy as np

import logging
import time

from patcherbot.devices.camera.camera import Camera


__all__ = ['LiveFeedQt']


class LiveFeedQt(QtWidgets.QLabel):
    """
    A QLabel-based widget that displays live camera feed, optionally logs frames,
    and allows custom image and display editing functions.
    """
    def __init__(
            self,
            camera: Camera,
            image_edit=None,
            display_edit=None,
            mouse_handler=None,
            parent=None,
        ):        
        """
        Args:
            camera (Camera): The camera object providing frames.
            image_edit (callable, optional): Function to modify raw numpy frames before display. Defaults to identity.
            display_edit (callable, optional): Function to modify QPixmap frames for GUI overlay. Defaults to identity.
            mouse_handler (callable, optional): Function to handle mouse events on the widget. Defaults to None.
            parent (QWidget, optional): Parent Qt widget. Defaults to None.
        """
        super(LiveFeedQt, self).__init__(parent=parent)
        # The image_edit function (does nothing by default) gets the raw
        # unscaled image (i.e. a numpy array), while the display_edit
        # function gets a QPixmap and is meant to draw GUI elements in
        # "display space" (by default, a red cross in the middle of the
        # screen).
        if image_edit is None:
            image_edit = lambda frame: frame
        self.image_edit = image_edit

        if display_edit is None:
            display_edit = lambda img: img
        self.display_edit = display_edit

        self.mouse_handler = mouse_handler
        self.mouse_handler = mouse_handler

# ------------------------------------------------------------
# Camera handling
# ------------------------------------------------------------

        if isinstance(camera, dict):
            if not camera:
                raise ValueError(
                    "LiveFeedQt received an empty camera dictionary."
                )

            self.cameras = camera
            self.active_camera_index = 0

            self.camera = list(
                self.cameras.values()
            )[self.active_camera_index]

        else:
            # Backwards-compatible single-camera behavior.
            self.cameras = None
            self.active_camera_index = 0
            self.camera = camera

        self.width, self.height = (
            self.camera.width,
            self.camera.height,
        )

        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignCenter)

        self.recording_state_manager = recording_state_manager
        self.recorder = FileLogger(recording_state_manager, folder_path="experiments/Data/rig_recorder_data/", isVideo=True, filetype="csv", recorder_filename="camera_frames", frame_folder_name=frame_folder_name)
        self._uses_aux_folder = frame_folder_name == self.recorder.aux_frame_folder_name
        self.log_processed_frames = bool(log_processed_frames)
        self.recording_config = None

        # Remember the last frame that we displayed, to not unnecessarily
        # process/show the same frame for slow input sources
        self._last_frameno = None
        self._last_edited_frame = None
        
        self.last_frame_time = None
        self.fps = 0

        self.update_image()

        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.update_image)
        timer.start(33) # 30 fps -> 31.5fps
        # timer.start(16) # 60 fps --> but actually 64 fps
        # timer.start(33) # 30 fps --> but avctually 21.5 fps

    def mousePressEvent(self, event):
        """
        Handle mouse press events on the widget.

        Args:
            event (QMouseEvent): The mouse event object.
        """
        # Ignore clicks that are not on the image
        xs = event.x() - self.size().width() * 0.5
        ys = event.y() - self.size().height() * 0.5
        pixmap = self.pixmap()
        if abs(xs) > pixmap.width() * 0.5 or abs(ys) > pixmap.height() * 0.5:
            self.setFocus()
            return

        if self.mouse_handler is not None:
            self.mouse_handler(event)

    def log_frame_rate(self):
        """Calculate and log the current frame rate based on the last processed frame."""
    # Calculate and log the frame rate at which images are processed
        current_time = time.time()
        if self.last_frame_time is not None:
            time_diff = current_time - self.last_frame_time
            self.fps = 1.0 / time_diff
            logging.info(f"FPS in LIVEFEED: {self.fps:.2f}")
        self.last_frame_time = current_time

    def set_active_camera(self, index: int) -> None:
        """
        Switch the live feed to another configured pipette camera.

        Args:
            index: Zero-based pipette/camera index.
        """
        new_camera = self._get_camera_by_index(index)

        self.active_camera_index = index
        self.camera = new_camera

        self.width = self.camera.width
        self.height = self.camera.height

        self._last_frameno = None
        self._last_edited_frame = None

    def set_camera_metadata(self, camera) -> None:
        """
        Update the widget with a new camera object and its metadata.

        Args:
            camera (Camera): New camera object to use.
        """
        self.camera = camera
        self.width = camera.width
        self.height = camera.height

        # New physical camera = new frame-number namespace.
        self._last_frameno = None
        self._last_edited_frame = None

    def _get_camera_by_index(self, index: int):
        """
        Return a configured pipette camera by zero-based index.
        """
        if self.cameras is None:
            if index == 0:
                return self.camera

            raise IndexError(
                "Single-camera rig only has camera index 0."
            )

        key = f"pipette_camera_{index}"

        if key not in self.cameras:
            raise IndexError(
                f"Camera '{key}' is not configured."
            )

        return self.cameras[key]

    @QtCore.pyqtSlot()
    def update_image(self):
        """
        Fetch the latest frame from the camera, apply edits, log the frame, 
        convert to QImage, scale it to widget size, and display it.

        Raises:
            Exception: Any exception raised during frame processing is caught and printed.
        """
        try:
            # get last frame from camera
            frame_info = self.camera.last_frame_pair()
            if frame_info is None:
                return
            frameno, frame_time, processed_frame, raw_frame = frame_info
            if processed_frame is None:
                return

            record_overlays = bool(getattr(self.recording_config, "record_overlays", self.log_processed_frames))
            source_frame = raw_frame if raw_frame is not None else processed_frame
            frame_to_log = source_frame.copy()
            if self._last_frameno is None or self._last_frameno != frameno:
                frame = self.image_edit(processed_frame.copy())
                self._last_edited_frame = frame
                self._last_frameno = frameno
            else:
                frame = self._last_edited_frame

            if frame is None:
                return
            annotate_recording = record_overlays and self.recording_state_manager.is_recording_enabled()
            write_frame = self.recorder.write_aux_camera_frames if self._uses_aux_folder else self.recorder.write_camera_frames
            if not annotate_recording:
                write_frame(frame_time.timestamp(), frame_to_log, frameno)

            if len(frame.shape) == 2:
                # Grayscale image via MicroManager
                if frame.dtype == np.dtype('uint32'):
                    bytesPerLine = self.width*4
                    format = QtGui.QImage.Format_RGB32
                else:
                    bytesPerLine = self.width
                    format = QtGui.QImage.Format_Indexed8
            else:
                # Color image via OpenCV
                bytesPerLine = 3 * self.width
                format = QtGui.QImage.Format_RGB888
            
            # ? So why frame.data and not frame? frame.data points to a memory location. I think both work, but check https://doc.qt.io/qt-6/qimage.html#QImage-5
            q_image = QtGui.QImage(frame.data, self.width, self.height,
                                   bytesPerLine, format)
            # q_image = QtGui.QImage(frame, self.width, self.height,
            #                        bytesPerLine, format)
            
            
            if format == QtGui.QImage.Format_RGB888:
                # OpenCV returns images as 24bit BGR (and not RGB), but there is no
                # direct support for this format in QImage
                q_image = q_image.rgbSwapped()

            pixmap = QtGui.QPixmap.fromImage(q_image)
            size = self.size()
            width, height = size.width(), size.height()
            scaled_pixmap = pixmap.scaled(width, height,
                                          Qt.KeepAspectRatio,
                                          Qt.SmoothTransformation)
            if self.display_edit is not None:
                self.display_edit(scaled_pixmap)
            if annotate_recording:
                # Paint at camera resolution so recording does not depend on window size.
                recorded_pixmap = pixmap.copy()
                if self.display_edit is not None:
                    self.display_edit(recorded_pixmap)
                recorded_image = recorded_pixmap.toImage().convertToFormat(QtGui.QImage.Format_RGB888)
                pixels = recorded_image.bits()
                pixels.setsize(recorded_image.byteCount())
                frame_to_log = np.frombuffer(pixels, dtype=np.uint8).reshape(
                    recorded_image.height(), recorded_image.bytesPerLine())[:, :recorded_image.width() * 3]
                frame_to_log = frame_to_log.reshape(recorded_image.height(), recorded_image.width(), 3).copy()
            if annotate_recording:
                write_frame(frame_time.timestamp(), frame_to_log, frameno)
            self.setPixmap(scaled_pixmap)

        except Exception:
            print(traceback.format_exc())
