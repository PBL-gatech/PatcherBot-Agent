import logging
from datetime import datetime
import threading
from collections import deque
import imageio
import os
import logging
import time

import pandas as pd


class FileLogger(threading.Thread):
    """
    Threaded logger for recording experimental data including graph values,
    movement data, and camera frames. Supports batching, asynchronous writes,
    and optional video frame handling.
    """
    def __init__(self, recording_state_manager, folder_path="experiments/Data/", recorder_filename="recording", filetype="csv", isVideo=False, frame_batch_size=500, frame_folder_name="camera_frames", aux_frame_folder_name="aux_camera_frames"):
        """
        Initialize the FileLogger.

        Args:
            recording_state_manager: Object controlling recording state.
            folder_path (str, optional): Base directory for saving data.
            recorder_filename (str, optional): Name of the output file.
            filetype (str, optional): File extension for the main log file.
            isVideo (bool, optional): Whether video frame saving is enabled.
            frame_batch_size (int, optional): Maximum number of frames to batch.
            frame_folder_name (str, optional): Folder name for primary camera frames.
            aux_frame_folder_name (str, optional): Folder name for auxiliary camera frames.
        """
        super().__init__()
        self.recording_state_manager = recording_state_manager
        self.time_truth = datetime.now()

        self.image_type  = "webp"

        testMode = self.recording_state_manager.testMode
        if testMode:
            folder_path = folder_path.replace("Data/", "Data/TEST_")

        self.frame_folder_name = frame_folder_name
        self.aux_frame_folder_name = aux_frame_folder_name

        self.folder_path = os.path.join(folder_path, self.time_truth.strftime("%Y_%m_%d-%H_%M"))
        self.camera_folder_path = os.path.join(self.folder_path, self.frame_folder_name)
        self.aux_camera_folder_path = os.path.join(self.folder_path, self.aux_frame_folder_name)
        self.filename = os.path.join(self.folder_path, recorder_filename + "." + filetype)
        self.file = None
        self.last_frameno = 0
        self.last_aux_frameno = 0
        self.frame_batch_size = frame_batch_size
        self.frame_batch_limit = int(frame_batch_size * 0.8)

        self.batch_mode_movements = False
        self.batch_mode_graph = False

        self.write_event = threading.Event()
        self.write_event.set()
        self.is_video = isVideo
        self.write_frame = threading.Event() if isVideo else None
        if self.write_frame:
            self.write_frame.set()

        self.batch_frames = deque(maxlen=frame_batch_size)
        # self.graph_contents = deque(maxlen=frame_batch_size)
        self.movement_contents = deque(maxlen=frame_batch_size)

        self.last_movement_time = None
        self.last_graph_time = None
        self.folder_created = False  # Flag to track folder creation

        logging.info(f"FileLogger initialized. Folder path set to: {self.folder_path}")

    def create_folder(self):
        """
        Create required directories for recording if recording is enabled.

        Raises:
            OSError: If directory creation fails.
        """
        # Check if the recording is enabled before creating the folder
        if self.recording_state_manager.is_recording_enabled() and not self.folder_created:
            try:
                os.makedirs(self.camera_folder_path, exist_ok=True)
                os.makedirs(self.aux_camera_folder_path, exist_ok=True)
                self.folder_created = True  # Set the flag to True once folder is created
                print(f"Created folder at: {self.folder_path}")
            except OSError as exc:
                logging.error("Error creating folder for recording: %s", exc)


    def open(self):
        """Open the main log file and write headers if the file is empty."""
        self.file = open(self.filename, "a+")
        self.file.seek(0, os.SEEK_END)
        is_empty = self.file.tell() == 0

        if is_empty:
            headers = {
                "movement_recording.csv": "timestamp;st_x;st_y;st_z;pi_x;pi_y;pi_z",
                "graph_recording.csv": "timestamp;pressure;resistance;current;voltage",
            }
            header = headers.get(os.path.basename(self.filename))
            if header:
                self.file.write(f"{header}\n")
                self.file.flush()


        print(f"Opened file at: {self.filename}")

    def _write_to_file(self, contents):
        """
        Write a single string of data to the log file.

        Args:
            contents (str): Data string to write.
        """
        if self.file is None:
            self.open()
        self.file.write(contents)
        self.file.flush()
        self.write_event.set()  # Signal that writing is done
        # print("Wrote file contents at path: ", self.filename)

    def _write_to_file_batch(self, contents):
        """
        Write multiple lines of data to the log file.

        Args:
            contents (list[str]): List of data strings.
        """
        if self.file is None:
            self.open()
        self.file.writelines(contents)
        self.file.flush()
        self.write_event.set()  # Signal that writing is done
        # print("Wrote file contents at path: ", self.filename)

    def write_graph_data(self, time_value, pressure: float, resistance: float, current, voltage):
        """
        Log a single graph data point asynchronously.

        Args:
            time_value: Timestamp value.
            pressure (float): Pressure measurement.
            resistance (float): Resistance measurement.
            current: Current measurement.
            voltage: Voltage measurement.
        """
        # ? time_current is probably not necessary, will remove in a future commit when confirmed.
        # def write_graph_data(self, time_value, pressure: float, resistance: float, time_current, current):
        if not self.recording_state_manager.is_recording_enabled():
            return
        if time_value == self.last_graph_time:
            return
        self.last_graph_time = time_value
        self.create_folder()  # Create the folder if recording is enabled and it's the first time
        # content = f"timestamp:{time_value}  pressure:{pressure}  resistance:{resistance}  / current:{current}\n"
        content = f"{time_value};{pressure};{resistance};{current};{voltage}\n"
        self.write_event.clear()
        threading.Thread(target=self._write_to_file, args=(content,)).start()

    def write_movement_data_batch(self, time_value, stage_x, stage_y, stage_z, pipette_x, pipette_y, pipette_z):
        """
        Buffer movement data and write in batches.

        Args:
            time_value: Timestamp value.
            stage_x, stage_y, stage_z: Stage coordinates.
            pipette_x, pipette_y, pipette_z: Pipette coordinates.
        """
        # start_time = time.perf_counter_ns()
        if not self.recording_state_manager.is_recording_enabled():
            return
        if time_value == self.last_movement_time:
            return
        self.last_movement_time = time_value
        self.create_folder()  # Create the folder if recording is enabled and it's the first time
        content = f"{time_value};{stage_x};{stage_y};{stage_z};{pipette_x};{pipette_y};{pipette_z}\n"

        #print('New Pos: ' + content)

        self.movement_contents.append(content)
        if len(self.movement_contents) >= self.frame_batch_limit:
            # logging.info(f"Batch size reached for MOVEMENT. Writing to disk at {datetime.now() - self.time_truth} seconds after start")
            self._flush_contents(self.movement_contents)
        # end_time = time.perf_counter_ns()
        # print(f"Time taken to write movement data: {(end_time - start_time)/1e6} ms")

    def _flush_contents(self, data):
        """
        Flush buffered data to disk asynchronously.

        Args:
            data (collections.deque): Buffered data.
        """
        if data:
            contents = data.copy()
            data.clear()
            self.write_event.clear()
            threading.Thread(target=self._write_to_file_batch, args=(contents,)).start()

    def _save_image(self, frame, path):
        """
        Buffer an image frame for batched disk writing.

        Args:
            frame: Image data.
            path (str): File path to save the image.
        """
        self.batch_frames.append((frame, path))
        if len(self.batch_frames) >= self.frame_batch_limit:
            # logging.info(f"Batch size reached for FRAMES. Writing to disk at {datetime.now() - self.time_truth} seconds after start")
            self.write_frame.clear()
            threading.Thread(target=self._write_batch_to_disk).start()
    
    def _save_image_sleep(self):
        """Flush any buffered frames to disk."""
        if self.batch_frames:
            self.write_frame.clear()
            threading.Thread(target=self._write_batch_to_disk).start()

    def _write_batch_to_disk(self):
        """Write all buffered image frames to disk."""
        while self.batch_frames:
            frame, path = self.batch_frames.popleft()
            # imwrite(path, frame)
            imageio.imwrite(path, frame, format=self.image_type)
            # qoi.write(path, frame)
        self.write_frame.set()  # Signal that image saving is done

    def write_camera_frames(self, time_value, frame, frameno):
        """
        Save primary camera frames with batching.

        Args:
            time_value: Timestamp value.
            frame: Image data.
            frameno (int): Frame number.
        """
        if not self.recording_state_manager.is_recording_enabled():
            self._save_image_sleep()
            return

        # * Add this back in if you change where this function is called within the update_image function in the LiveFeedQT class. 
        # if frameno is None:
        #     logging.info("No frame number detected. Closing the camera recorder")
        #     self.close()
        #     return

        if frameno <= self.last_frameno:
            return
        self.create_folder()  # Create the folder if recording is enabled and it's the first time
        image_path = os.path.join(self.camera_folder_path, f"{frameno}_{time_value}.{self.image_type}")
        self._save_image(frame, image_path)
        self.last_frameno = frameno

    def write_aux_camera_frames(self, time_value, frame, frameno):
        """
        Save auxiliary camera frames with batching.

        Args:
            time_value: Timestamp value.
            frame: Image data.
            frameno (int): Frame number.
        """
        if not self.recording_state_manager.is_recording_enabled():
            self._save_image_sleep()
            return

        if frameno is None or frameno <= self.last_aux_frameno:
            return
        self.create_folder()
        image_path = os.path.join(self.aux_camera_folder_path, f"{frameno}_{time_value}.{self.image_type}")
        self._save_image(frame, image_path)
        self.last_aux_frameno = frameno

    def setBatchGraph(self, value=True):
        """
        Enable or disable batch mode for graph data.

        Args:
            value (bool, optional): Batch mode flag.
        """
        self.batch_mode_graph = value
    def setBatchMoves(self, value=True):
        """
        Enable or disable batch mode for movement data.

        Args:
            value (bool, optional): Batch mode flag.
        """
        self.batch_mode_movements = value

    def flush_movement_data(self):
        """Flush any remaining buffered movement data to disk."""
        if not self.write_event.is_set():
            self.write_event.wait()
        if not self.movement_contents:
            return

        try:
            os.makedirs(self.folder_path, exist_ok=True)
        except OSError as exc:
            logging.error("Error ensuring movement folder exists: %s", exc)

        if self.file is None:
            self.open()

        contents = list(self.movement_contents)
        self.movement_contents.clear()

        self.write_event.clear()
        self.file.writelines(contents)
        self.file.flush()
        self.write_event.set()

    def handle_recording_stopped(self):
        """Handle cleanup when recording stops, ensuring all data is written."""
        if self.is_video and self.write_frame:
            self._save_image_sleep()
            self.write_frame.wait()
        self.flush_movement_data()

    def close(self):
        """Close the logger, ensuring all pending writes and frames are saved."""
        if self.file is not None:
            logging.info("Closing file: %s", self.filename)
            # if self.batch_mode_graph and self.graph_contents:
            #     self._flush_contents(self.graph_contents)
            if self.batch_mode_movements and self.movement_contents:
                self._flush_contents(self.movement_contents)
            self.write_event.wait()  # Wait for the last task to complete

            self.file.close()
            self.file = None

        self.write_frame.wait()
        logging.info("Closing CSV recorder writing thread")
        if self.batch_frames:
            self._write_batch_to_disk()
        if self.is_video:
            logging.info("Closing frame saving thread")

