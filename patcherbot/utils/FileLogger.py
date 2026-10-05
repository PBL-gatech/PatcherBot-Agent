from __future__ import annotations

import logging
import os
import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import imageio.v2 as imageio


class FileLogger:
    """
    Thread-safe, queue-backed low-level recorder.

    FileLogger intentionally knows only how to:
        - append text/CSV rows to one file
        - save camera frames asynchronously
        - flush pending writes deterministically

    Higher-level recording semantics (multi-pipette graph logging,
    rig-state snapshots, movement streams, Excel export, etc.) should live
    in dedicated recorder classes built on top of this class.
    """

    DEFAULT_HEADERS = {
        "movement_recording.csv": (
            "timestamp;st_x;st_y;st_z;pi_x;pi_y;pi_z"
        ),
        "graph_recording.csv": (
            "timestamp;pressure;resistance;current;voltage"
        ),
    }

    _STOP = object()

    def __init__(
        self,
        recording_state_manager,
        folder_path="experiments/Data/",
        recorder_filename="recording",
        filetype="csv",
        isVideo=False,
        frame_batch_size=500,
        frame_folder_name="camera_frames",
        aux_frame_folder_name="aux_camera_frames",
        *,
        header: Optional[str] = None,
        session_id: Optional[str] = None,
        image_type="webp",
        text_batch_size=128,
    ):
        self.recording_state_manager = recording_state_manager
        self.time_truth = datetime.now()

        self.image_type = image_type
        self.is_video = bool(isVideo)
        self.frame_batch_size = max(1, int(frame_batch_size))
        self.text_batch_size = max(1, int(text_batch_size))

        test_mode = bool(
            getattr(self.recording_state_manager, "testMode", False)
        )
        if test_mode:
            folder_path = folder_path.replace("Data/", "Data/TEST_")

        # Preserve the old behavior where loggers created during the same
        # startup minute land in the same session directory. A future
        # RecordingCoordinator can pass an explicit session_id instead.
        if session_id is None:
            session_id = self.time_truth.strftime("%Y_%m_%d-%H_%M")

        self.base_folder_path = Path(folder_path)
        self.folder_path = str(self.base_folder_path / session_id)

        self.frame_folder_name = frame_folder_name
        self.aux_frame_folder_name = aux_frame_folder_name
        self.camera_folder_path = os.path.join(
            self.folder_path,
            self.frame_folder_name,
        )
        self.aux_camera_folder_path = os.path.join(
            self.folder_path,
            self.aux_frame_folder_name,
        )

        self.filename = os.path.join(
            self.folder_path,
            f"{recorder_filename}.{filetype}",
        )

        self.header = (
            header
            if header is not None
            else self.DEFAULT_HEADERS.get(os.path.basename(self.filename))
        )

        # Kept public for compatibility with old code that may inspect it.
        self.file = None

        self.folder_created = False
        self._closed = False

        self.last_graph_time = None
        self.last_movement_time = None
        self.last_frameno = 0
        self.last_aux_frameno = 0
        self._last_frame_by_source = {}

        # Legacy flags retained so old callers do not break. Queue-backed
        # writing makes explicit batch-mode toggles unnecessary.
        self.batch_mode_graph = False
        self.batch_mode_movements = False

        self._state_lock = threading.Lock()

        self._text_queue = queue.Queue()
        self._image_queue = queue.Queue() if self.is_video else None

        self._text_thread = threading.Thread(
            target=self._text_worker,
            name=f"FileLogger-text-{recorder_filename}",
            daemon=True,
        )
        self._text_thread.start()

        self._image_thread = None
        if self.is_video:
            self._image_thread = threading.Thread(
                target=self._image_worker,
                name=f"FileLogger-image-{recorder_filename}",
                daemon=True,
            )
            self._image_thread.start()

        logging.info(
            "FileLogger initialized. Folder path set to: %s",
            self.folder_path,
        )

    # ------------------------------------------------------------------
    # Filesystem / file lifecycle
    # ------------------------------------------------------------------

    def create_folder(self):
        """Create the session directory if needed."""
        if self.folder_created:
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
        if self.file is None:
            self.open()
        self.file.write(contents)
        self.file.flush()
        self.write_event.set()  # Signal that writing is done
        # print("Wrote file contents at path: ", self.filename)

    def _write_to_file_batch(self, contents):
        if self.file is None:
            self.open()
        self.file.writelines(contents)
        self.file.flush()
        self.write_event.set()  # Signal that writing is done
        # print("Wrote file contents at path: ", self.filename)

    def write_graph_data(self, time_value, pressure: float, resistance: float, current, voltage):
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
        if data:
            contents = data.copy()
            data.clear()
            self.write_event.clear()
            threading.Thread(target=self._write_to_file_batch, args=(contents,)).start()

    def _save_image(self, frame, path, wait=False):
        if wait:
            self._write_image(frame, path)
            return
        self.batch_frames.append((frame, path))
        if len(self.batch_frames) >= self.frame_batch_limit:
            # logging.info(f"Batch size reached for FRAMES. Writing to disk at {datetime.now() - self.time_truth} seconds after start")
            self.write_frame.clear()
            threading.Thread(target=self._write_batch_to_disk).start()
    
    def _save_image_sleep(self):
        if self.batch_frames:
            self.write_frame.clear()
            threading.Thread(target=self._write_batch_to_disk).start()

    def _write_image(self, frame, path):
        imageio.imwrite(path, frame, format=self.image_type)

    def _write_batch_to_disk(self):
        while self.batch_frames:
            frame, path = self.batch_frames.popleft()
            # imwrite(path, frame)
            self._write_image(frame, path)
            # qoi.write(path, frame)
        self.write_frame.set()  # Signal that image saving is done

    def write_camera_frames(self, time_value, frame, frameno):
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
        self.batch_mode_graph = value
    def setBatchMoves(self, value=True):
        self.batch_mode_movements = value

    def flush_movement_data(self):
        if not self.write_event.is_set():
            self.write_event.wait()
        if not self.movement_contents:
            return

        try:
            os.makedirs(self.folder_path, exist_ok=True)
            self.folder_created = True
        except OSError:
            logging.exception(
                "Error creating recording folder: %s",
                self.folder_path,
            )
            raise

    def _ensure_file_open(self):
        """Open the text log file from the text writer thread."""
        if self.file is not None:
            return

        self.create_folder()

        self.file = open(
            self.filename,
            "a+",
            encoding="utf-8",
            newline="",
        )
        self.file.seek(0, os.SEEK_END)

        if self.file.tell() == 0 and self.header:
            self.file.write(f"{self.header}\n")
            self.file.flush()

        logging.info("Opened recording file: %s", self.filename)

    def open(self):
        """
        Compatibility wrapper.

        Normal callers should not need to call open(); the worker opens the
        file on the first queued write.
        """
        self._ensure_not_closed()
        self._ensure_file_open()

    def _ensure_not_closed(self):
        if self._closed:
            raise RuntimeError("Cannot write to a closed FileLogger")

    # ------------------------------------------------------------------
    # Queue-backed text writing
    # ------------------------------------------------------------------

    def _enqueue_text(self, content: str):
        self._ensure_not_closed()
        self._text_queue.put(content)

    def _text_worker(self):
        """Single owner of the text file handle."""
        try:
            while True:
                item = self._text_queue.get()

                if item is self._STOP:
                    self._text_queue.task_done()
                    break

                batch = [item]

                # Opportunistically batch queued rows into one disk write.
                while len(batch) < self.text_batch_size:
                    try:
                        next_item = self._text_queue.get_nowait()
                    except queue.Empty:
                        break

                    if next_item is self._STOP:
                        # Put the sentinel back so it is handled after this
                        # batch has been written.
                        self._text_queue.task_done()
                        self._text_queue.put(self._STOP)
                        break

                    batch.append(next_item)

                try:
                    self._ensure_file_open()
                    self.file.writelines(batch)
                    self.file.flush()
                except Exception:
                    logging.exception(
                        "Failed writing recorder data to %s",
                        self.filename,
                    )
                finally:
                    for _ in batch:
                        self._text_queue.task_done()
        finally:
            if self.file is not None:
                try:
                    self.file.flush()
                    self.file.close()
                finally:
                    self.file = None

    def write_row(self, values: Iterable, delimiter=";"):
        """Append one delimited row while recording is enabled."""
        if not self.recording_state_manager.is_recording_enabled():
            return

        content = delimiter.join(
            "" if value is None else str(value)
            for value in values
        ) + "\n"

        self._enqueue_text(content)

    # ------------------------------------------------------------------
    # Backward-compatible graph / movement API
    # ------------------------------------------------------------------

    def write_graph_data(
        self,
        time_value,
        pressure: float,
        resistance: float,
        current,
        voltage,
    ):
        """
        Compatibility method for the existing single-stream graph recorder.

        Multi-pipette graph recording will move to GraphRecorder rather than
        extending this method with workbook semantics.
        """
        if not self.recording_state_manager.is_recording_enabled():
            return

        with self._state_lock:
            if time_value == self.last_graph_time:
                return
            self.last_graph_time = time_value

        self.write_row(
            (
                time_value,
                pressure,
                resistance,
                current,
                voltage,
            )
        )

    def write_movement_data_batch(
        self,
        time_value,
        stage_x,
        stage_y,
        stage_z,
        pipette_x,
        pipette_y,
        pipette_z,
    ):
        """
        Compatibility method for the existing movement logger.

        Multi-pipette movement recording will move to MovementRecorder so
        each pipette gets its own identifiable stream/file.
        """
        if not self.recording_state_manager.is_recording_enabled():
            return

        with self._state_lock:
            if time_value == self.last_movement_time:
                return
            self.last_movement_time = time_value

        self.write_row(
            (
                time_value,
                stage_x,
                stage_y,
                stage_z,
                pipette_x,
                pipette_y,
                pipette_z,
            )
        )

    def setBatchGraph(self, value=True):
        """Legacy compatibility; queue-backed writing is always batched."""
        self.batch_mode_graph = bool(value)

    def setBatchMoves(self, value=True):
        """Legacy compatibility; queue-backed writing is always batched."""
        self.batch_mode_movements = bool(value)

    def flush_movement_data(self):
        """Legacy compatibility wrapper."""
        self.flush()

    # ------------------------------------------------------------------
    # Camera frame writing
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_source_name(source_id) -> str:
        value = str(source_id or "camera").strip()
        safe = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_"
            for ch in value
        )
        return safe or "camera"

    def _frame_folder_for_source(self, source_id):
        source = self._safe_source_name(source_id)

        if source in {"main", "main_camera", "primary", "camera"}:
            folder_name = self.frame_folder_name
        elif source in {"aux", "aux_camera"}:
            folder_name = self.aux_frame_folder_name
        else:
            # Multi-pipette cameras naturally become separate folders such as
            # pipette_camera_0/, pipette_camera_1/, ...
            folder_name = source

        return os.path.join(self.folder_path, folder_name)

    def write_camera_frame(self, source_id, timestamp, frame, frameno):
        """Queue one camera frame for asynchronous saving."""
        if not self.is_video:
            return

        if not self.recording_state_manager.is_recording_enabled():
            return

        if frameno is None or frame is None:
            return

        source = self._safe_source_name(source_id)

        with self._state_lock:
            last_frame = self._last_frame_by_source.get(source)
            if last_frame is not None and frameno <= last_frame:
                return
            self._last_frame_by_source[source] = frameno

            if source in {"main", "main_camera", "primary", "camera"}:
                self.last_frameno = frameno
            elif source in {"aux", "aux_camera"}:
                self.last_aux_frameno = frameno

        folder = self._frame_folder_for_source(source)
        filename = f"{frameno}_{timestamp}.{self.image_type}"
        path = os.path.join(folder, filename)

        # Camera backends may reuse their frame buffer after this method
        # returns, so queue an owned copy where possible.
        queued_frame = frame.copy() if hasattr(frame, "copy") else frame
        self._image_queue.put((queued_frame, path))

    # Legacy method names used by older LiveFeedQt implementations.
    def write_camera_frames(self, time_value, frame, frameno):
        self.write_camera_frame(
            "main_camera",
            time_value,
            frame,
            frameno,
        )

    def write_aux_camera_frames(self, time_value, frame, frameno):
        self.write_camera_frame(
            "aux_camera",
            time_value,
            frame,
            frameno,
        )

    def _image_worker(self):
        while True:
            item = self._image_queue.get()

            if item is self._STOP:
                self._image_queue.task_done()
                break

            frame, path = item

            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                imageio.imwrite(
                    path,
                    frame,
                    format=self.image_type,
                )
            except Exception:
                logging.exception(
                    "Failed saving camera frame to %s",
                    path,
                )
            finally:
                self._image_queue.task_done()

    # ------------------------------------------------------------------
    # Flush / shutdown
    # ------------------------------------------------------------------

    def flush(self):
        """Block until all currently queued writes are complete."""
        if self._closed:
            return

        self._text_queue.join()

        if self._image_queue is not None:
            self._image_queue.join()

        if self.file is not None:
            self.file.flush()

    def handle_recording_stopped(self):
        """
        Finish all writes belonging to the recording that just stopped.

        The worker threads remain alive so the same logger can be used for a
        subsequent recording period until close() is called.
        """
        self.flush()

    def close(self):
        """Flush pending data and shut down writer threads exactly once."""
        if self._closed:
            return

        self.flush()
        self._closed = True

        self._text_queue.put(self._STOP)
        self._text_thread.join()

        if self._image_queue is not None:
            self._image_queue.put(self._STOP)
            self._image_thread.join()

        logging.info("Closed FileLogger: %s", self.filename)
