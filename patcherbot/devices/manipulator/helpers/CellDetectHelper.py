import logging
from typing import List, Tuple

import numpy as np


class CellDetectHelper:
    """Run cell detection on the latest camera frame and display the results."""

    def __init__(self, camera):
        self.camera = camera
        self.cellDetector = None

    def _ensure_detector(self):
        if self.cellDetector is None:
            from patcherbot.deepLearning.CellDetector import CellDetector2

            self.cellDetector = CellDetector2(model_type="pidnet")
        return self.cellDetector

    def _latest_raw_frame(self):
        try:
            frame = self.camera.raw_frame_queue[0][3]
        except (AttributeError, IndexError, TypeError):
            return None
        if frame is None:
            return None
        return frame.copy()

    def detect_cells(self) -> List[Tuple[int, int, float]]:
        frame = self._latest_raw_frame()
        if frame is None:
            logging.info("Cell detector: no raw camera frame is available.")
            self.camera.show_circles([])
            return []

        detections = self._ensure_detector().detect_cells(frame)
        height, width = frame.shape[:2]
        valid_detections: List[Tuple[int, int, float]] = []

        for detection in detections or []:
            try:
                x, y, confidence = detection
                x = float(x)
                y = float(y)
                confidence = float(confidence)
            except (TypeError, ValueError):
                continue
            if not np.all(np.isfinite([x, y, confidence])):
                continue

            x_int = int(round(x))
            y_int = int(round(y))
            if not (0 <= x_int < width and 0 <= y_int < height):
                continue
            valid_detections.append((x_int, y_int, confidence))

        self.camera.show_circles([(x, y) for x, y, _ in valid_detections])
        if valid_detections:
            logging.info("Cell detector locations: %s", valid_detections)
        else:
            logging.info("Cell detector: no cells detected.")
        return valid_detections
