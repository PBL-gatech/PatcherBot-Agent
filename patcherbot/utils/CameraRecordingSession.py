import logging


class CameraRecordingSession:
    """
    Own recording state for all cameras in one rig session.

    Camera objects perform the actual frame writing.
    GUI views do not own recording resources.
    """

    def __init__(
        self,
        recording_state_manager,
        main_camera,
        pipette_cameras=None,
    ):
        self.recording_state_manager = recording_state_manager
        self.cameras = {}
        self._recording = False
        self._closed = False

        self._add_camera(
            "main_camera",
            main_camera,
        )

        if pipette_cameras is None:
            pipette_cameras = {}

        elif not isinstance(pipette_cameras, dict):
            pipette_cameras = {
                "pipette_camera_0": pipette_cameras
            }

        for source_id, camera in pipette_cameras.items():
            self._add_camera(
                source_id,
                camera,
            )

        logging.info(
            "CameraRecordingSession initialized with sources: %s",
            list(self.cameras.keys()),
        )

    def _add_camera(self, source_id, camera):
        if camera is None:
            return

        # Don't register the exact same camera object twice.
        for existing_id, existing_camera in self.cameras.items():
            if existing_camera is camera:
                logging.warning(
                    "Camera '%s' duplicates '%s'; skipping.",
                    source_id,
                    existing_id,
                )
                return

        self.cameras[str(source_id)] = camera

    def start_recording(
        self,
        directory,
        file_prefix,
        skip_frames=0,
        memory_mb=1000,
    ):
        if self._closed:
            raise RuntimeError(
                "Cannot start a closed CameraRecordingSession."
            )

        if self._recording:
            return

        started_cameras = []

        try:
            for source_id, camera in self.cameras.items():
                start_method = getattr(
                    camera,
                    "start_recording",
                    None,
                )

                if start_method is None:
                    logging.debug(
                        "Camera '%s' does not support start_recording.",
                        source_id,
                    )
                    continue

                width = getattr(camera, "width", 0)
                height = getattr(camera, "height", 0)

                if width and height:
                    queue_size = (
                        int(memory_mb * 1e6 / (width * height))
                        + 1
                    )
                else:
                    queue_size = 1

                # Give each physical camera a unique file prefix.
                source_prefix = (
                    f"{file_prefix}_{source_id}"
                )

                start_method(
                    directory=directory,
                    file_prefix=source_prefix,
                    skip_frames=skip_frames,
                    queue_size=queue_size,
                )

                started_cameras.append(camera)

        except Exception:
            # Roll back cameras that already started.
            for camera in started_cameras:
                stop_method = getattr(
                    camera,
                    "stop_recording",
                    None,
                )

                if stop_method is not None:
                    try:
                        stop_method()
                    except Exception:
                        logging.exception(
                            "Error rolling back camera recording."
                        )

            raise

        if self.recording_state_manager is not None:
            self.recording_state_manager.set_recording(True)

        self._recording = True

    def stop_recording(self):
        if not self._recording:
            if self.recording_state_manager is not None:
                self.recording_state_manager.set_recording(False)
            return

        for source_id, camera in self.cameras.items():
            stop_method = getattr(
                camera,
                "stop_recording",
                None,
            )

            if stop_method is None:
                continue

            try:
                stop_method()
            except Exception:
                logging.exception(
                    "Error stopping camera '%s'.",
                    source_id,
                )

        if self.recording_state_manager is not None:
            self.recording_state_manager.set_recording(False)

        self._recording = False

    def close(self):
        if self._closed:
            return

        self.stop_recording()
        self._closed = True