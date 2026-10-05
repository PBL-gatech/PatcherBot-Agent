from __future__ import print_function
import numpy as np
from PyQt5 import QtCore, QtWidgets
import warnings
import cv2
from numpy import *
from patcherbot.interface import TaskInterface, command, blocking_command
import os
import logging
from patcherbot.utils.RecordingStateManager import RecordingStateManager
from patcherbot.utils.FileLogger import FileLogger


class CameraInterface(TaskInterface):
    updated_exposure = QtCore.pyqtSignal('QString', 'QString')

    def __init__(self, camera, with_tracking=False, status_category='Camera'):
        super().__init__()
        self.camera = camera
        self.with_tracking = with_tracking
        self.recording_state_manager = RecordingStateManager()
        self.snap_image_recorder = FileLogger(
            self.recording_state_manager,
            folder_path="experiments/Data/snap_image_data/",
            isVideo=True,
            filetype="csv",
            recorder_filename="snap_images",
            frame_batch_size=2,
        )
        self.status_category = status_category
        self._is_active = False

    def connect(self, main_gui):
        self.updated_exposure.connect(main_gui.set_status_message)
        self.signal_updated_exposure()
        if self.with_tracking:
            main_gui.image_edit_funcs.append(self.show_tracked_objects)
            #main_gui.image_edit_funcs.append(self.show_tracked_paramecium)
            #main_gui.image_edit_funcs.append(self.pipette_contact_detection)

    def signal_updated_exposure(self):
        # Should be called by subclasses that actually support setting the exposure
        if not self._is_active:
            return
        exposure = self.camera.get_exposure()
        if exposure > 0:
            self.updated_exposure.emit(self.status_category, 'Exposure: %.1f ms' % exposure)

    def set_active(self, is_active: bool):
        was_active = self._is_active
        self._is_active = bool(is_active)
        if self._is_active:
            self.signal_updated_exposure()
        elif was_active:
            self.updated_exposure.emit(self.status_category, None)

    def set_status_category(self, category: str) -> None:
        self.status_category = category
        if self._is_active:
            self.signal_updated_exposure()

    def set_camera(self, camera) -> None:
        self.camera = camera
        if self._is_active:
            self.signal_updated_exposure()
    
    @blocking_command(category='Camera',
                      description='Auto exposure',
                      task_description='Adjusting exposure')
    def auto_exposure(self,args):
        self.camera.auto_exposure()
        self.signal_updated_exposure()

    @command(category='Camera',
             description='Increase exposure time by {:.1f}ms',
             default_arg=2.5)
    def increase_exposure(self, increase):
        self.camera.change_exposure(increase)
        self.signal_updated_exposure()
    @command(category='Camera',
                description='Set exposure time to {:.1f}ms',
                default_arg=2.5)
    def set_exposure(self, exposure):     
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
        self.camera.normalize()

    @command(category='Camera',
             description='Unnormalize the image',
             )
    def unnormalize(self, param=None):
        self.camera.unnormalize()

    @command(category='Camera',
             description='Snap image',
             )
    def snap_image(self, param=None):
        try:
            frameno, frame_time, _, raw_frame = self.camera.raw_frame_queue[0]
        except (AttributeError, IndexError, TypeError, ValueError):
            return
        if frameno is None or raw_frame is None or frame_time is None:
            return
        try:
            return self.save_snapshot(frameno, frame_time, raw_frame)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return
        except OSError as exc:
            logging.error("Error creating snap image folder: %s", exc)
            return

    def save_snapshot(self, frame_number, captured_at, frame, wait=False):
        """Save a supplied frame using the normal snapshot path and image format.

        With ``wait=True``, writing completes here and failures reach the caller.
        """
        if frame_number is None or captured_at is None or frame is None:
            raise ValueError("Snapshot requires a frame number, timestamp, and frame.")
        time_value = captured_at.timestamp()
        frame_to_save = frame.copy() if hasattr(frame, "copy") else frame
        if frame_to_save is None:
            raise ValueError("Snapshot frame cannot be empty.")
        recorder = self.snap_image_recorder
        if not recorder.folder_created:
            os.makedirs(recorder.camera_folder_path, exist_ok=True)
            os.makedirs(recorder.aux_camera_folder_path, exist_ok=True)
            recorder.folder_created = True
        image_path = os.path.join(
            recorder.camera_folder_path,
            f"{frame_number}_{time_value}.{recorder.image_type}",
        )
        recorder._save_image(frame_to_save, image_path, wait=wait)
        return {
            "frame_number": frame_number,
            "captured_at": captured_at,
            "image_path": image_path,
            "frame": frame_to_save,
        }

    @command(category='Camera',
             description='AutoNormalize the image',
             )
    def autonormalize(self, state):
        # if state: 
        #     print("AutoNormalizing")
        # else:
        #     print("Not AutoNormalizing")
        self.camera.autonormalize(state)

    @command(category='Camera',
             description='Decrease exposure time by {:.1f}ms',
             default_arg=2.5)
    def decrease_exposure(self, decrease):
        self.camera.change_exposure(-decrease)
        self.signal_updated_exposure()


