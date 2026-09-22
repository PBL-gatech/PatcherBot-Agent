import numpy as np


class StageScanHelper:
    """Helper for selecting stage scan corners and executing serpentine scans."""

    CORNER_LABELS = ("Top Left", "Bottom Right")
    SCAN_MAX_SPEED = 2000
    NORMAL_MAX_SPEED = 10000

    def __init__(self, camera, config=None):
        self.camera = camera
        self.config = config
        self.reset()

    def reset(self):
        self._is_collecting = False
        self._corner_positions = []

    @property
    def is_collecting(self):
        return self._is_collecting

    @property
    def corner_positions(self):
        return list(self._corner_positions)

    @property
    def corner_count(self):
        return len(self._corner_positions)

    def _scan_camera(self, stage):
        stage_camera = getattr(stage, "camera", None)
        if stage_camera is not None:
            return stage_camera
        return self.camera

    def start_corner_collection(self, stage):
        self.reset()
        self._is_collecting = True
        stage.info("Store Corners armed. Shift-right click Top Left, then Bottom Right.")

    def _stage_xy_from_pixels(self, stage, stage_pixels):
        """Convert a stage-reference pixel coordinate into stage XY in um."""
        stage_pixels = np.asarray(stage_pixels, dtype=float).reshape(-1)
        if stage_pixels.size < 2:
            raise ValueError("stage_pixels must contain at least x and y values.")
        if stage_pixels.size < 3:
            stage_pixels = np.append(stage_pixels, 0.0)
        stage_um = np.asarray(stage.pixels_to_um(stage_pixels), dtype=float).reshape(-1)
        if stage_um.size < 2:
            raise ValueError("pixels_to_um did not return a valid XY coordinate.")
        return np.array([float(stage_um[0]), float(stage_um[1])], dtype=float)

    def _store_corner(self, stage, microscope, click_position):
        camera = self._scan_camera(stage)
        if camera is None:
            raise RuntimeError("No camera available for stage scan corner conversion.")

        click_position = np.asarray(click_position, dtype=float).reshape(-1)
        if click_position.size < 2:
            raise ValueError("click_position must contain x and y values.")

        click_position = click_position.copy()
        click_position[0] += float(camera.width) / 2.0
        click_position[1] += float(camera.height) / 2.0

        stage_pos_pixels = np.asarray(stage.reference_position(), dtype=float).reshape(-1)
        if stage_pos_pixels.size < 2:
            raise RuntimeError("Stage reference_position() must return at least x and y.")
        stage_pos_pixels[0:2] -= click_position[0:2]
        if stage_pos_pixels.size < 3:
            stage_pos_pixels = np.append(stage_pos_pixels, 0.0)

        stage_xy_um = self._stage_xy_from_pixels(stage, stage_pos_pixels)
        z_um = 0.0
        if microscope is not None:
            z_um = float(microscope.position() / 5.0)
        stage_pos_um = np.array([stage_xy_um[0], stage_xy_um[1], z_um], dtype=float)

        description = self.CORNER_LABELS[self.corner_count]
        self._corner_positions.append(
            {
                "description": description,
                "stage_pixels": stage_pos_pixels,
                "stage_um": stage_pos_um,
            }
        )
        stage.info(
            f"{description} selected: stage pixels={stage_pos_pixels}, "
            f"stage um={stage_pos_um}"
        )

    def record_corner_from_click(self, stage, microscope, click_position):
        """Record one corner and return True once both corners are captured."""
        if not self._is_collecting:
            stage.info(
                "Store corners is not active. Click 'Store corners' before "
                "Shift-right-clicking points."
            )
            return False

        if self.corner_count >= len(self.CORNER_LABELS):
            self._is_collecting = False
            stage.warning("Corner list already full; restarting collection.")
            return True

        self._store_corner(stage, microscope, click_position)
        if self.corner_count == len(self.CORNER_LABELS):
            self._is_collecting = False
            top_left, bottom_right = self._corner_positions
            stage.info("Corner capture complete.")
            stage.info(f"Top Left stage pixels: {top_left['stage_pixels']}")
            stage.info(f"Top Left stage um: {top_left['stage_um']}")
            stage.info(f"Bottom Right stage pixels: {bottom_right['stage_pixels']}")
            stage.info(f"Bottom Right stage um: {bottom_right['stage_um']}")
            return True
        return False

    def _move_to_scan_start(self, stage, top_left_xy):
        """Move stage to the scan starting point and wait for motion to finish."""
        stage.abort_if_requested()
        stage.absolute_move(top_left_xy)
        stage.abort_if_requested()
        stage.wait_until_still()

    def move_to_scan_start(self, stage, speed=None):
        """Move to the stored Top Left corner for scan initialization."""
        if self.corner_count < len(self.CORNER_LABELS):
            stage.warning(
                "Move to scan start requires both Top Left and Bottom Right "
                "corners from Select Corners."
            )
            return

        top_left_pixels = np.asarray(self._corner_positions[0]["stage_pixels"], dtype=float)
        top_left_xy = self._stage_xy_from_pixels(stage, top_left_pixels)
        scan_speed = self.SCAN_MAX_SPEED if speed is None else float(speed)

        prior_speed = None
        try:
            prior_speed = stage.get_max_speed()
        except Exception:
            prior_speed = None

        try:
            try:
                stage.set_max_speed(scan_speed)
            except Exception:
                stage.warning("Could not set stage max speed for move-to-start.")
            self._move_to_scan_start(stage, top_left_xy)
        finally:
            try:
                restore_speed = prior_speed if prior_speed is not None else self.NORMAL_MAX_SPEED
                stage.set_max_speed(restore_speed)
            except Exception:
                stage.warning("Failed to restore stage max speed after move-to-start.")

    def _scan_y_delta_um(self, stage):
        if self.config is not None and hasattr(self.config, "y_delta_scan"):
            return float(self.config.y_delta_scan)
        stage_config = getattr(stage, "config", None)
        if stage_config is not None and hasattr(stage_config, "y_delta_scan"):
            return float(stage_config.y_delta_scan)
        return 1340.0

    def scan_area(self, stage, speed=None):
        """Scan the selected rectangle in a serpentine pattern."""
        stage.abort_if_requested()
        if self.corner_count < len(self.CORNER_LABELS):
            stage.warning(
                "Scan area requires both Top Left and Bottom Right corners from "
                "Select Corners."
            )
            return

        camera = self._scan_camera(stage)
        if camera is None:
            stage.warning("No camera available for scan step calculation.")
            return

        top_left_px = np.asarray(self._corner_positions[0]["stage_pixels"], dtype=float)
        bottom_right_px = np.asarray(self._corner_positions[1]["stage_pixels"], dtype=float)

        top_left_xy = self._stage_xy_from_pixels(stage, top_left_px)
        left_x_px = float(top_left_px[0])
        right_x_px = float(bottom_right_px[0])

        target_delta_y_px = float(bottom_right_px[1] - top_left_px[1])
        if np.isclose(target_delta_y_px, 0.0):
            stage.warning(
                "Top Left and Bottom Right have the same Y coordinate; no vertical scan range."
            )
            return
        target_y_px = float(bottom_right_px[1])
        target_direction = 1.0 if target_delta_y_px > 0 else -1.0

        configured_y_delta_um = self._scan_y_delta_um(stage)
        if configured_y_delta_um <= 0:
            stage.warning("Configured y_delta_scan must be positive.")
            return

        y_delta_px_vec = np.asarray(
            stage.um_to_pixels_relative(
                np.array([0.0, target_direction * configured_y_delta_um], dtype=float)
            ),
            dtype=float,
        ).reshape(-1)
        if y_delta_px_vec.size < 2:
            stage.warning("Could not convert y_delta_scan to pixel space for scanning.")
            return

        row_step_mag_px = abs(float(y_delta_px_vec[1]))
        if np.isclose(row_step_mag_px, 0.0):
            stage.warning("Configured y_delta_scan converted to zero in pixel space.")
            return

        row_step_y_px = row_step_mag_px if target_direction > 0 else -row_step_mag_px
        reached_target_y = lambda y_px: (y_px - target_y_px) * target_direction >= 0

        prior_speed = None
        try:
            prior_speed = stage.get_max_speed()
        except Exception:
            prior_speed = None

        scan_speed = self.SCAN_MAX_SPEED if speed is None else float(speed)

        try:
            try:
                stage.set_max_speed(scan_speed)
            except Exception:
                stage.warning(
                    "Could not set stage max speed for scan; speed override may not apply."
                )

            self._move_to_scan_start(stage, top_left_xy)

            def move_to_pixel(px_x, px_y):
                target_xy = self._stage_xy_from_pixels(
                    stage, np.array([px_x, px_y, 0.0], dtype=float)
                )
                stage.absolute_move(target_xy)
                stage.wait_until_still()

            current_y_px = float(top_left_px[1])
            move_toward_right = True
            while True:
                stage.abort_if_requested()
                target_x_px = right_x_px if move_toward_right else left_x_px
                move_to_pixel(target_x_px, current_y_px)

                stage.abort_if_requested()
                if reached_target_y(current_y_px):
                    break

                next_y_px = current_y_px + row_step_y_px
                if reached_target_y(next_y_px):
                    current_y_px = target_y_px
                    move_to_pixel(target_x_px, current_y_px)
                    move_toward_right = not move_toward_right
                    final_target_x_px = right_x_px if move_toward_right else left_x_px
                    move_to_pixel(final_target_x_px, current_y_px)
                    break

                stage.abort_if_requested()
                current_y_px = next_y_px
                move_to_pixel(target_x_px, current_y_px)
                move_toward_right = not move_toward_right
        finally:
            try:
                restore_speed = prior_speed if prior_speed is not None else self.NORMAL_MAX_SPEED
                stage.set_max_speed(restore_speed)
            except Exception:
                stage.warning("Failed to restore stage max speed after scan.")
