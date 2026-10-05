import time
import cv2
import numpy as np
import os
from pathlib import Path
from patcherbot.devices.manipulator.microscope import Microscope
from patcherbot.devices.manipulator import Manipulator
from patcherbot.devices.camera import Camera
from patcherbot.deepLearning.pipetteDetector import PipetteDetector, PipetteDetectorYOLO1, PipetteDetector4
from patcherbot.deepLearning.pipetteFocuser import PipetteFocuser2
from threading import Thread
import logging
from .pipette_flow import track_pipette_displacement


CALIBRATION_FRAME_COUNT = 5
CALIBRATION_TIMEOUT_SECONDS = 3.0

def _resolve_model_path(model_name):
    """
    Resolves a pipette model path string into an absolute Path object.

    If `model_name` is None or an empty string, returns None. If it is relative,
    the path is assumed to be relative to the project's `deepLearning/pipetteModel` folder.

    Args:
        model_name (str or None): Name or path of the model file.

    Returns:
        Path or None: Absolute Path to the model if given, else None.
    """
    if model_name is None:
        return None
    model_text = str(model_name).strip()
    if not model_text:
        return None

    # Normalize accidental wrappers like r'...', "...", and stray quote tails.
    while model_text:
        previous = model_text
        if len(model_text) >= 2 and model_text[0].lower() == "r" and model_text[1] in ("'", '"'):
            model_text = model_text[1:].strip()
        if len(model_text) >= 2 and model_text[0] in ("'", '"') and model_text[-1] == model_text[0]:
            model_text = model_text[1:-1].strip()
        model_text = model_text.strip(" '\"")
        if model_text == previous:
            break

    if not model_text:
        return None

    path = Path(model_text).expanduser()
    if path.is_absolute():
        return path

    model_dir = Path(__file__).resolve().parents[3] / "deepLearning" / "pipetteModel"
    return model_dir / path


class PipetteCalHelper():
    """
    A helper class to aid with 2D pipette calibration.

    Overview:
      - Calibration points pair raw image coordinates with pipette encoders.
      - Only the (x, y) components are used, ignoring the z-axis.
      - Ten calibration points are gathered so that the field of view is well‐sampled.
      - A 2D affine transformation is computed from pipette encoder (x, y) positions to
        image (x, y) positions, using OpenCV.
      - The 2×3 matrix is then embedded in a 3×4 homogeneous transformation matrix.
    
    Summary:
      1. Record calibration points as tuples: (image_x, image_y, encoder_x, encoder_y).
      2. Subtract the stage's reference position from the pipette’s detected image location.
      3. Use cv2.estimateAffine2D on the collected points.
      4. Convert the resulting 2×3 matrix into a 3×4 matrix for homogeneous representation.
    """

    CAL_MAX_SPEED = 1000
    NORMAL_MAX_SPEED = 1000

    def __init__(self, pipette: Manipulator, microscope: Microscope, camera: Camera, calibrated_stage, config=None):
        """
        Initializes the PipetteCalHelper.

        Args:
            pipette (Manipulator): The pipette manipulator to calibrate.
            microscope (Microscope): Microscope used to image the pipette.
            camera (Camera): Camera capturing the pipette.
            calibrated_stage: Reference stage for relative positioning.
            config: Optional configuration object with model paths.
        """
        self.pipette: Manipulator = pipette
        self.microscope: Microscope = microscope
        self.camera = camera
        self.config = config
        device = os.getenv("PIPETTE_DETECTOR_DEVICE")
        model_name = getattr(self.config, "pipette_detector_model", None) if self.config is not None else None
        model_path = _resolve_model_path(model_name)
        try:
            # self.pipetteDetector: PipetteDetector = PipetteDetectorYOLO1(
            #     model_path=model_path,
            #     device=device,
            # )
            self.pipetteDetector: PipetteDetector = PipetteDetector4(
                model_path=model_path,
                device=device,
            )
        except Exception as exc:
            logging.warning(
                "Failed to initialize PipetteDetector4 with model '%s' (%s); using default model path",
                model_name,
                exc,
            )
            self.pipetteDetector = PipetteDetector4(device=device)
        self.calibrated_stage = calibrated_stage
        # Each calibration point will be a tuple:
        #   (image_x, image_y, encoder_x, encoder_y)
        self.cal_points = []
        self.calibration_samples = []
        self._flow_reference = None

    def collect_cal_points(self, num_points=10, xy_step=5, max_retries=5):
        """Apply the stage-style calibration speed cap and restore the prior limit."""
        log = logging.getLogger("patcherbot.PipetteCalibration")
        previous_speed = self.pipette.get_max_speed()
        if previous_speed is None or not np.isfinite(previous_speed) or previous_speed <= 0:
            raise RuntimeError("Cannot read pipette speed; calibration was not started")
        calibration_speed = min(float(previous_speed), self.CAL_MAX_SPEED)
        try:
            self.pipette.set_max_speed(calibration_speed)
            applied_speed = self.pipette.get_max_speed()
            if (applied_speed is None or not np.isfinite(applied_speed)
                    or applied_speed <= 0 or applied_speed > calibration_speed):
                raise RuntimeError("Pipette calibration speed limit was not confirmed; sampling was not started")
            log.info("Pipette calibration speed limit: %s -> %s (controller units)",
                     previous_speed, applied_speed)
            return self._collect_cal_points_at_calibration_speed(num_points, xy_step, max_retries)
        finally:
            self.pipette.set_max_speed(previous_speed)
            restored_speed = self.pipette.get_max_speed()
            if restored_speed is None or restored_speed != previous_speed:
                raise RuntimeError("Pipette speed restoration was not confirmed")
            log.info("Pipette calibration speed limit restored to %s (controller units)", restored_speed)

    def _collect_cal_points_at_calibration_speed(self, num_points=10, xy_step=5, max_retries=5):
        """
        Collects 'num_points' calibration points.
        
        For each point:
         - The pipette’s current (x, y) image position (median of five fresh detections)
           is paired with the pipette’s encoder (x, y) coordinates.
         - A small move (with added randomness) is commanded between points so that the
           calibration data covers a larger area.

        Args:
            num_points (int): Number of calibration points to collect. Default is 10.
            xy_step (float): Nominal step size (in microns) between consecutive points. Default is 5.
            max_retries (int): Maximum retries per point if pipette detection fails. Default is 5.

        Returns:
            bool: True if the requested number of points were successfully collected, False otherwise.
        """
        self.cal_points = []
        self.calibration_samples = []
        self._flow_reference = None
        step = min(2.0 * abs(float(xy_step)), 10.0)
        logging.getLogger("patcherbot.PipetteCalibration").info(
            "Pipette calibration sampling: X then Y, %d points, %.1f um steps, Z unchanged",
            num_points, step)
        for i in range(num_points):
            self._record_point_with_retries(max_retries)
            if i < num_points - 1:
                # Separate axis measurements; default steps stay within the old 0-10 um range.
                delta = np.zeros(3)
                delta[0 if i < (num_points - 1) // 2 else 1] = step
                self.pipette.relative_move(delta.tolist())
                self.pipette.wait_until_still()
                self.pipette.sleep(0.5)  # Allow time for camera to update
        return len(self.cal_points) >= num_points

    def _record_point_with_retries(self, max_retries=5):
        """
        Attempt to record a calibration point. If no pipette is detected in the image,
        jitter the pipette and retry.

        Args:
            max_retries (int): Maximum number of attempts before giving up. Default is 5.

        Returns:
            bool: True if a point was successfully recorded, False if all retries failed.
        """
        for _ in range(max_retries):
            before = len(self.cal_points)
            self.record_cal_point()
            if len(self.cal_points) > before:
                return True
            # Jitter the pipette if detection failed.
            self.pipette.relative_move([
                np.random.uniform(-15, 15),
                np.random.uniform(-15, 15),
                0
            ])
            self.pipette.wait_until_still()
            time.sleep(1)
        return False

    def record_cal_point(self):
        """Record the median of five distinct newly retrieved stationary frames."""
        log = logging.getLogger("patcherbot.PipetteCalibration")
        started = time.monotonic()
        encoder_pos = np.asarray(self.pipette.position(), dtype=float).copy()
        seen, detections, observations = set(), [], []
        sample = dict(encoder=encoder_pos.tolist(), sampling_started_at=started,
                      frames=[], status="timeout")
        self.calibration_samples.append(sample)
        while time.monotonic() - started < CALIBRATION_TIMEOUT_SECONDS:
            data = self.camera.last_raw_frame_data(include_timing=True)
            if data is None:
                time.sleep(0.01)
                continue
            frame_no, _, frame, timing = data
            retrieved = timing.get("retrieval_started_at")
            if frame_no in seen or retrieved is None or retrieved < started:
                time.sleep(0.01)
                continue
            seen.add(frame_no)
            if not np.allclose(self.pipette.position(), encoder_pos, atol=0.1, rtol=0):
                sample["status"] = "encoder_moved"
                log.warning("Pipette calibration sample rejected: encoder moved before detection")
                return
            if hasattr(self.pipetteDetector, "detect_pipette_tracking"):
                tracking = self.pipetteDetector.detect_pipette_tracking(frame.copy())
                tip = tracking.get("tip_xy")
                flow_frame, mask = tracking.get("frame"), tracking.get("mask")
            else:
                tip = self.pipetteDetector.detect_pipette(frame.copy())
                flow_frame, mask = frame.copy(), None
            if not np.allclose(self.pipette.position(), encoder_pos, atol=0.1, rtol=0):
                sample["status"] = "encoder_moved"
                log.warning("Pipette calibration sample rejected: encoder moved during detection")
                return
            if tip is None:
                continue
            tip = np.asarray(tip, dtype=float)
            if tip.shape != (2,) or not np.isfinite(tip).all():
                continue
            detections.append(tip)
            observations.append((flow_frame, mask))
            sample["frames"].append(dict(frame_no=int(frame_no),
                                         retrieval_started_at=float(retrieved), tip_xy=tip.tolist()))
            if len(detections) == CALIBRATION_FRAME_COUNT:
                median = np.median(detections, axis=0)
                representative = int(np.argmin(np.linalg.norm(np.asarray(detections) - median, axis=1)))
                position, source = self._hybrid_position(median, *observations[representative])
                self.cal_points.append(tuple(position) + tuple(encoder_pos[:2]))
                spread = float(np.max(np.linalg.norm(np.asarray(detections) - median, axis=1)))
                sample.update(status="complete", median_xy=median.tolist(), max_spread_px=spread,
                              fitted_xy=position.tolist(), source=source)
                log.info("Pipette calibration point %d: %d frames, median=(%.2f, %.2f) px, "
                         "max spread=%.2f px, encoder=(%.3f, %.3f) um",
                         len(self.cal_points), len(detections), *median, spread, *encoder_pos[:2])
                self.camera.show_circle(tuple(int(round(v)) for v in position))
                return
        log.warning("Pipette calibration sample timed out: %d/%d valid new frames; point rejected",
                    len(detections), CALIBRATION_FRAME_COUNT)

    def _hybrid_position(self, detected, frame, mask):
        """Use validated foreground motion, with detection anchoring and fallback."""
        log = logging.getLogger("patcherbot.PipetteCalibration")
        previous = getattr(self, "_flow_reference", None)
        position, source = detected.copy(), "detection"
        if previous is not None and frame is not None and mask is not None and previous["mask"] is not None:
            try:
                delta, quality = track_pipette_displacement(previous["frame"], frame,
                                                           previous["mask"], mask,
                                                           initial_displacement=detected - previous["detected"])
                if delta is not None:
                    disagreement = float(np.linalg.norm(delta - (detected - previous["detected"])))
                    if disagreement <= 12.0:
                        predicted = previous["position"] + delta
                        position = 0.85 * predicted + 0.15 * detected
                        source = "flow+detector"
                    log.info("Pipette calibration flow: %s; detector disagreement=%.2f px; %s",
                             source, disagreement, quality)
                else:
                    log.info("Pipette calibration flow rejected; using detection: %s", quality)
            except (cv2.error, ValueError, TypeError):
                log.warning("Pipette calibration flow failed; using detection", exc_info=True)
        else:
            log.info("Pipette calibration flow: detection anchor (no usable frame/mask pair)")
        self._flow_reference = dict(frame=frame, mask=mask, detected=detected.copy(), position=position.copy())
        return position, source

    def _log_calibration_quality(self, matrix, inliers):
        """Log fit residuals for the matrix selected by normal calibration."""
        points = np.asarray(self.cal_points, dtype=float)
        design = np.column_stack((points[:, 2:4], np.ones(len(points))))
        errors = np.linalg.norm(design @ matrix[:2, [0, 1, 3]].T - points[:, :2], axis=1)
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        self.last_calibration_quality = dict(
            mode="active_calibration", status="fitted", matrix=matrix.tolist(),
            points=points.tolist(), samples=self.calibration_samples,
            timestamp_basis="camera_retrieval_not_sensor_exposure",
            inliers=None if inliers is None else inliers.ravel().tolist(),
            training_errors_px=errors.tolist(), training_rmse_px=rmse)
        log = logging.getLogger("patcherbot.PipetteCalibration")
        singular = np.linalg.svd(points[:, 2:4] - points[:, 2:4].mean(axis=0), compute_uv=False)
        condition = float(singular[0] / singular[-1]) if singular[-1] > 1e-9 else float("inf")
        flow_count = sum(sample.get("source") == "flow+detector" for sample in self.calibration_samples)
        log.info("Pipette calibration geometry condition=%.2f; flow-assisted points=%d/%d",
                 condition, flow_count, len(points))
        log.info("Pipette calibration fit: %d hybrid points, %d inliers, "
                 "fit RMSE=%.3f px, maximum fit error=%.3f px (training residuals)",
                 len(points), 0 if inliers is None else int(np.count_nonzero(inliers)),
                 rmse, float(errors.max()))
        log.info("Pipette calibration selected matrix: %s", matrix.tolist())
        log.info("Pipette calibration errors by point (px): %s", np.round(errors, 3).tolist())

    def calibrate(self):
        """
        Computes a 2D affine transformation that maps pipette encoder (x, y)
        positions to image (x, y) positions in the camera image.
        
        It then embeds the resulting 2×3 matrix into a 3×4 homogeneous transformation matrix.
        
        Returns:
            A 3×4 transformation matrix.
            (The calibrated unit's finish_calibration routine may then convert this
             into a full 4×4 matrix as needed.)
        """
        if len(self.cal_points) < 3:
            print("Not enough calibration points for affine transformation.")
            return None

        # Prepare the data arrays (each is N×2):
        # encoder_points: [ [encoder_x, encoder_y], ... ]
        # image_points:   [ [image_x, image_y], ... ]
        encoder_points = np.array([[pt[2], pt[3]] for pt in self.cal_points], dtype=np.float64)
        image_points   = np.array([[pt[0], pt[1]] for pt in self.cal_points], dtype=np.float64)

        # Compute the 2D affine transformation (a 2×3 matrix)
        M2x3, inliers = cv2.estimateAffine2D(encoder_points, image_points)
        if M2x3 is None:
            print("Failed to compute affine transformation.")
            return None

        print("Raw 2D affine transformation matrix (2×3):")
        print(M2x3)

        # Convert the 2×3 matrix to a 3×4 homogeneous transformation matrix.
        # We assume the pipette moves only in x and y (z=0).
        # The resulting matrix has the form:
        #   [ A  B  0  t_x ]
        #   [ C  D  0  t_y ]
        #   [ 0  0  1   0  ]
        mat3x4 = np.zeros((3, 4), dtype=np.float64)
        mat3x4[0, 0:2] = M2x3[0, 0:2]
        mat3x4[0, 3]   = M2x3[0, 2]
        mat3x4[1, 0:2] = M2x3[1, 0:2]
        mat3x4[1, 3]   = M2x3[1, 2]
        mat3x4[2, 2]   = 1.0

        print("Converted 3×4 homogeneous transformation matrix:")
        print(mat3x4)

        # Save the calibration matrix for later use (e.g., for centering the pipette).
        self.calibration_matrix = mat3x4
        try:
            self._log_calibration_quality(mat3x4, inliers)
        except Exception:
            logging.exception("Could not log pipette calibration fit details")
        # Clear the calibration points after computing the matrix.
        self.cal_points = []
        return mat3x4

class PipetteFocusHelper():
    def __init__(self, pipette: Manipulator, camera: Camera, config=None, *, detector):
        """
        Initializes the PipetteFocusHelper.

        Args:
            pipette (Manipulator): The pipette manipulator to focus.
            camera (Camera): Camera used for image capture.
            config: Optional configuration object with model paths.
        """

        self.pipette = pipette
        self.camera = camera
        self.config = config
        self.pipetteFocuser = PipetteFocuser2(detector)
    
    def focus(self,frame=None):
        """
        Adjusts the pipette focus by capturing an image,
        predicting the defocus value, and commanding a relative move
        using that value.
        """
        if frame is None:
            # Get the latest frame from the camera if not provided.
            _, _, _, frame = self.camera.raw_frame_queue[0]
        # convert to 8-bit for display
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        defocus_value = self.pipetteFocuser.get_pipette_focus_value(frame)
        # print(f"Defocus value: {defocus_value:.2f} µm")
        
        # Use the defocus value directly in the relative move command.
        self.pipette.relative_move([0, 0, -defocus_value])
        self.pipette.wait_until_still()
