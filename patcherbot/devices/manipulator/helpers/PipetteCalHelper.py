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
      - Calibration points are collected relative to the stage's base reference.
      - Only the (x, y) components are used, ignoring the z-axis.
      - Ten calibration points are gathered so that the field of view is well‐sampled.
      - A 2D affine transformation is computed from pipette encoder (x, y) positions to
        image (x, y) positions (with stage reference subtracted), using OpenCV.
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

    def collect_cal_points(self, num_points=10, xy_step=5, max_retries=5):
        """
        Collects 'num_points' calibration points.
        
        For each point:
         - The pipette’s current (x, y) image position (with the stage's reference subtracted)
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
        for i in range(num_points):
            self._record_point_with_retries(max_retries)
            if i < num_points - 1:
                # Move pipette slightly in the plane to spread out calibration data.
                dx = xy_step + np.random.uniform(-5, 5)
                dy = xy_step + np.random.uniform(-5, 5)
                self.pipette.relative_move([dx, dy, 0])
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
        """
        Records a calibration point as follows:
         - Retrieves the current camera frame.
         - Uses the pipetteDetector to detect the pipette’s (x, y) position.
         - Subtracts the stage’s reference position so that the result is in the stage’s coordinate system.
         - Pairs the (x, y) image position with the pipette’s encoder (x, y) coordinates.
        """
        # Get the latest frame from the camera.
        _, _, _, frame = self.camera.raw_frame_queue[0]
        pos_pix = self.pipetteDetector.detect_pipette(frame)
        if pos_pix is not None:
            # Optionally, display the detected pipette on the frame.
            frame = cv2.circle(frame, pos_pix, 10, 0, 2)
            image_x = pos_pix[0]
            image_y = pos_pix[1]
            # Retrieve the pipette's encoder (x, y) positions; ignore z.
            encoder_pos = self.pipette.position()  # assumed format: [x, y, z]
            encoder_x = encoder_pos[0]
            encoder_y = encoder_pos[1]
            self.cal_points.append((image_x, image_y, encoder_x, encoder_y))
            self.camera.show_circle(pos_pix)
            print("Recorded calibration point:", self.cal_points[-1])
        else:
            print("No pipette detected in current frame [here].")

    def calibrate(self):
        """
        Computes a 2D affine transformation that maps pipette encoder (x, y)
        positions to image (x, y) positions (already relative to the stage's base).
        
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
