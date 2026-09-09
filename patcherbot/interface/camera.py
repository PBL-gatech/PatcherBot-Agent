from __future__ import print_function
import numpy as np
from PyQt5 import QtCore, QtWidgets
import warnings
import cv2
from numpy import *
from patcherbot.interface import TaskInterface, command, blocking_command
import os
import logging



class CameraInterface(TaskInterface):
    """
    Interface for controlling camera operations, including exposure,
    normalization, and image capture, with optional object tracking support.
    """
    updated_exposure = QtCore.pyqtSignal('QString', 'QString')

    def __init__(
        self,
        camera,
        snap_image_recorder,
        source_id,
        with_tracking=False,
        status_category="Camera",
    ):
        super().__init__()

        if snap_image_recorder is None:
            raise ValueError(
                "CameraInterface requires a shared snap_image_recorder"
            )

        self.camera = camera
        self.snap_image_recorder = snap_image_recorder
        self.source_id = str(source_id)
        self.with_tracking = with_tracking
        self.status_category = status_category
        self._is_active = False

    def connect(self, main_gui):
        """
        Connect camera-related signals and optional tracking functions to the GUI.

        Args:
            main_gui (CameraGui): Main GUI instance to connect signals to.
        """
        self.updated_exposure.connect(main_gui.set_status_message)
        self.signal_updated_exposure()
        if self.with_tracking:
            main_gui.image_edit_funcs.append(self.show_tracked_objects)
            #main_gui.image_edit_funcs.append(self.show_tracked_paramecium)
            #main_gui.image_edit_funcs.append(self.pipette_contact_detection)

    def signal_updated_exposure(self):
        """Emit a signal to update the GUI with the current exposure value."""
        # Should be called by subclasses that actually support setting the exposure
        if not self._is_active:
            return
        exposure = self.camera.get_exposure()
        if exposure > 0:
            self.updated_exposure.emit(self.status_category, 'Exposure: %.1f ms' % exposure)

    def set_active(self, is_active: bool):
        """
        Enable or disable the camera interface and update exposure display accordingly.

        Args:
            is_active (bool): Whether the camera interface should be active.
        """
        was_active = self._is_active
        self._is_active = bool(is_active)
        if self._is_active:
            self.signal_updated_exposure()
        elif was_active:
            self.updated_exposure.emit(self.status_category, None)

    def set_status_category(self, category: str) -> None:
        """
        Set the status category label used for GUI exposure updates.

        Args:
            category (str): New status category label.
        """
        self.status_category = category
        if self._is_active:
            self.signal_updated_exposure()

    def set_camera(self, camera) -> None:
        """
        Update the camera object and refresh exposure information if active.

        Args:
            camera (object): New camera instance.
        """
        self.camera = camera
        if self._is_active:
            self.signal_updated_exposure()
    
    @blocking_command(category='Camera',
                      description='Auto exposure',
                      task_description='Adjusting exposure')
    def auto_exposure(self,args):
        """
        Automatically adjust the camera exposure.

        Args:
            args (object): Optional argument (unused).
        """
        self.camera.auto_exposure()
        self.signal_updated_exposure()

    @command(category='Camera',
             description='Increase exposure time by {:.1f}ms',
             default_arg=2.5)
    def increase_exposure(self, increase):
        """
        Increase the camera exposure time by a specified amount.

        Args:
            increase (float): Amount of exposure time to increase (in ms).
        """
        self.camera.change_exposure(increase)
        self.signal_updated_exposure()
    @command(category='Camera',
                description='Set exposure time to {:.1f}ms',
                default_arg=2.5)
    def set_exposure(self, exposure):     
        """
        Set the camera exposure to a specific value by adjusting relative change.

        Args:
            exposure (float): Target exposure time (in ms).
        """
        currexpos = self.camera.get_exposure()
        change = exposure - currexpos
        if change > 0:
            self.increase_exposure(change)
        elif change < 0:
            self.decrease_exposure(-change)
        # logging.info('Current exposure time is: {}'.format(currexpos))
        # logging.info("difference is: {}".format(change))
        logging.info('New exposure time is: {}'.format(exposure))
        # self.camera.set_exposure(exposure)
        self.signal_updated_exposure()
    
    @command(category='Camera',
             description='Normalize the image',
             )
    def normalize(self, param=None):
        """
        Enable image normalization on the camera.

        Args:
            param (object, optional): Unused parameter.
        """
        self.camera.normalize()

    @command(category='Camera',
             description='Unnormalize the image',
             )
    def unnormalize(self, param=None):
        """
        Disable image normalization on the camera.

        Args:
            param (object, optional): Unused parameter.
        """
        self.camera.unnormalize()

    @command(category='Camera',
             description='Snap image',
             )
    def snap_image(self, param=None):
        """
        Capture the current frame from the camera and save it to disk.

        Args:
            param (object, optional): Unused parameter.
        """
        try:
            frameno, frame_time, _, raw_frame = self.camera.raw_frame_queue[0]
        except (AttributeError, IndexError, TypeError, ValueError):
            return
        if frameno is None or raw_frame is None or frame_time is None:
            return
        try:
            time_value = frame_time.timestamp()
            frame_to_save = raw_frame.copy() if hasattr(raw_frame, "copy") else raw_frame
        except (AttributeError, TypeError, ValueError, OverflowError):
            return
        if frame_to_save is None:
            return
        recorder = self.snap_image_recorder
        if not recorder.folder_created:
            try:
                recorder.create_folder()
            except OSError as exc:
                logging.error(
                    "Error creating snap image folder: %s",
                    exc,
                )
                return
        image_path = os.path.join(
            recorder.camera_folder_path,
            f"{self.source_id}_{frameno}_{time_value}.{recorder.image_type}",
        )

        recorder._save_image(frame_to_save, image_path)

        return {
            "source_id": self.source_id,
            "frame_number": frameno,
            "captured_at": frame_time,
            "image_path": image_path,
            "frame": frame_to_save,
        }

    @command(category='Camera',
             description='AutoNormalize the image',
             )
    def autonormalize(self, state):
        """
        Enable or disable automatic image normalization.

        Args:
            state (bool): Whether to enable autonormalization.
        """
        # if state: 
        #     print("AutoNormalizing")
        # else:
        #     print("Not AutoNormalizing")
        self.camera.autonormalize(state)

    @command(category='Camera',
             description='Decrease exposure time by {:.1f}ms',
             default_arg=2.5)
    def decrease_exposure(self, decrease):
        """
        Decrease the camera exposure time by a specified amount.

        Args:
            decrease (float): Amount of exposure time to decrease (in ms).
        """
        self.camera.change_exposure(-decrease)
        self.signal_updated_exposure()


