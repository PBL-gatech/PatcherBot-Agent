from __future__ import absolute_import

import numpy as np
from patcherbot.controller import TaskController


class PlateScanner(TaskController):
    """Collect two plate corner positions from image-space right clicks."""

    CORNER_LABELS = ("Top Left", "Bottom Right")

    def __init__(self, patch_interface):
        super().__init__()
        self.patch_interface = patch_interface
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

    def _iter_speed_controllers(self, stage):
        """Yield candidate objects that may expose max-speed getters/setters."""
        seen = set()
        for obj in (stage, getattr(stage, "unit", None), getattr(stage, "dev", None), getattr(getattr(stage, "unit", None), "dev", None)):
            if obj is None:
                continue
            ident = id(obj)
            if ident in seen:
                continue
            seen.add(ident)
            if hasattr(obj, "set_max_speed") or hasattr(obj, "get_max_speed"):
                yield obj

    def _get_max_speed(self, stage):
        """Return first reachable max speed from available speed controllers."""
        for ctrl in self._iter_speed_controllers(stage):
            getter = getattr(ctrl, "get_max_speed", None)
            if not callable(getter):
                continue
            try:
                speed = getter()
            except Exception:
                continue
            if speed is not None:
                return speed, ctrl
        return None, None

    def _set_max_speed(self, stage, speed):
        """Try to apply max speed on the first controller that supports it."""
        for ctrl in self._iter_speed_controllers(stage):
            setter = getattr(ctrl, "set_max_speed", None)
            if not callable(setter):
                continue
            try:
                setter(speed)
                return True
            except Exception:
                continue
        return False

    def _stage_xy_from_pixels(self, stage, stage_pixels):
        """Convert a stage reference pixel coordinate into stage XY in um."""
        stage_pixels = np.asarray(stage_pixels, dtype=float).reshape(-1)
        if stage_pixels.size < 2:
            raise ValueError("stage_pixels must contain at least x and y values.")
        if stage_pixels.size < 3:
            stage_pixels = np.append(stage_pixels, 0.0)
        stage_um = np.asarray(stage.pixels_to_um(stage_pixels), dtype=float).reshape(-1)
        if stage_um.size < 2:
            raise ValueError("pixels_to_um did not return a valid XY coordinate.")
        return np.array([float(stage_um[0]), float(stage_um[1])], dtype=float)

    def _store_position(self, click_position):
        click_position = np.array(click_position, dtype=float)

        camera = self.patch_interface.pipette_controller.calibrated_unit.camera
        click_position[0] += camera.width / 2.0
        click_position[1] += camera.height / 2.0

        stage = self.patch_interface.pipette_controller.calibrated_stage
        stage_pos_pixels = np.array(stage.reference_position(), dtype=float)
        stage_pos_pixels[0:2] -= click_position

        stage_xy_um = self._stage_xy_from_pixels(stage, stage_pos_pixels)
        z_um = float(self.patch_interface.pipette_controller.calibrated_unit.microscope.position() / 5.0)
        stage_pos_um = np.array([stage_xy_um[0], stage_xy_um[1], z_um], dtype=float)

        description = self.CORNER_LABELS[self.corner_count]
        self._corner_positions.append(
            {
                "description": description,
                "stage_pixels": stage_pos_pixels,
                "stage_um": stage_pos_um,
            }
        )
        print(
            f"{description}: stage pixels {stage_pos_pixels}, "
            f"stage um {stage_pos_um}"
        )
        self.patch_interface.info(
            f"{description} selected: stage pixels={stage_pos_pixels}, stage um={stage_pos_um}"
        )

    def SelectCorners(self, click_position=None):
        """Collect the Top Left and Bottom Right clicks from the GUI."""
        if click_position is None:
            self.reset()
            self._is_collecting = True
            print("Store Corners armed. Shift-right click Top Left, then Bottom Right.")
            return

        if not self._is_collecting:
            print("Store corners is not active. Click 'Store corners' before Shift-right-clicking points.")
            return

        self._store_position(click_position)
        if self.corner_count == len(self.CORNER_LABELS):
            self._is_collecting = False
            top_left, bottom_right = self._corner_positions
            print("Corner capture complete.")
            print(f"Top Left stage pixels: {top_left['stage_pixels']}")
            print(f"Top Left stage um: {top_left['stage_um']}")
            print(f"Bottom Right stage pixels: {bottom_right['stage_pixels']}")
            print(f"Bottom Right stage um: {bottom_right['stage_um']}")

    def _move_to_scan_start(self, stage, top_left_xy, speed):
        """Move the stage to the scan starting point and wait for motion to finish."""
        self.abort_if_requested()
        stage.absolute_move(top_left_xy, speed=speed)
        self.abort_if_requested()
        stage.wait_until_still()

    def move_to_scan_start(self, speed=70):
        """Move to the stored Top Left corner for scan initialization."""
        if self.corner_count < len(self.CORNER_LABELS):
            self.patch_interface.warning(
                "Move to scan start requires both Top Left and Bottom Right corners from Select Corners."
            )
            return

        stage = self.patch_interface.pipette_controller.calibrated_stage
        top_left_pixels = np.array(self._corner_positions[0]["stage_pixels"], dtype=float)
        top_left_xy = self._stage_xy_from_pixels(stage, top_left_pixels)
        self._move_to_scan_start(stage, top_left_xy, speed)

    def ScanArea(self):
        """Scan the selected rectangular area from top-left to bottom-right at constant velocity."""
        self.abort_if_requested()
        if self.corner_count < len(self.CORNER_LABELS):
            self.patch_interface.warning(
                "ScanArea requires both Top Left and Bottom Right corners from Select Corners."
            )
            return

        stage = self.patch_interface.pipette_controller.calibrated_stage
        camera = self.patch_interface.pipette_controller.calibrated_unit.camera
        top_left_px = np.array(self._corner_positions[0]["stage_pixels"], dtype=float)
        bottom_right_px = np.array(self._corner_positions[1]["stage_pixels"], dtype=float)

        row_step_px = float(camera.height) / 2.0
        if row_step_px <= 0:
            self.patch_interface.warning("Camera height is too small for scan step calculation.")
            return

        scan_speed_um_per_sec = 1000
        top_left_xy = self._stage_xy_from_pixels(stage, top_left_px)

        target_delta_y_px = float(bottom_right_px[1] - top_left_px[1])
        if np.isclose(target_delta_y_px, 0.0):
            self.patch_interface.warning("Top Left and Bottom Right have the same Y coordinate; no vertical scan range.")
            return

        target_y_px = float(bottom_right_px[1])
        target_direction = 1.0 if target_delta_y_px > 0 else -1.0
        row_step_y_px = row_step_px if target_direction > 0 else -row_step_px
        reached_target_y = lambda y_px: (y_px - target_y_px) * target_direction >= 0
        left_x_px = float(top_left_px[0])
        right_x_px = float(bottom_right_px[0])

        prior_speed = None
        prior_speed_controller = None
        try:
            prior_speed, prior_speed_controller = self._get_max_speed(stage)
            if prior_speed is None:
                # Last-resort fallback: use the configured default max speed on the
                # underlying controller, if available.
                for ctrl in self._iter_speed_controllers(stage):
                    default_speed = getattr(ctrl, "DEFAULT_MAX_SPEED", None)
                    if default_speed is not None:
                        prior_speed = default_speed
                        prior_speed_controller = ctrl
                        break

            if prior_speed is not None and not self._set_max_speed(stage, scan_speed_um_per_sec):
                self.patch_interface.warning("Could not set stage max speed for scan; speed override may not apply.")

            self._move_to_scan_start(stage, top_left_xy, scan_speed_um_per_sec)

            def move_to_pixel(px_x, px_y):
                target_xy = self._stage_xy_from_pixels(stage, np.array([px_x, px_y, 0.0], dtype=float))
                stage.absolute_move(target_xy, speed=scan_speed_um_per_sec)
                stage.wait_until_still()

            current_y_px = float(top_left_px[1])
            move_toward_right = True
            while True:
                self.abort_if_requested()
                target_x_px = right_x_px if move_toward_right else left_x_px
                move_to_pixel(target_x_px, current_y_px)

                self.abort_if_requested()
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

                self.abort_if_requested()
                current_y_px = next_y_px
                move_to_pixel(target_x_px, current_y_px)

                move_toward_right = not move_toward_right
        finally:
            if prior_speed is not None:
                try:
                    if prior_speed_controller is not None:
                        prior_speed_set = getattr(prior_speed_controller, "set_max_speed", None)
                        if callable(prior_speed_set):
                            prior_speed_set(prior_speed)
                        else:
                            raise RuntimeError("No speed setter available on saved speed controller.")
                    elif not self._set_max_speed(stage, prior_speed):
                        raise RuntimeError("No speed setter available for this stage.")
                except Exception:
                    self.patch_interface.warning(
                        "Failed to restore stage max speed after scan."
                    )
