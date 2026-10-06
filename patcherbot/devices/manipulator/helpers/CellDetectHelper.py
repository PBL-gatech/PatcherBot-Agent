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

    def detect_cells(self) -> List[Tuple[int, int, float]]:
        frame = self.camera.last_raw_frame_data()
        if frame is None or frame[2] is None:
            self.camera.show_circles([])
            return []
        detections = self._ensure_detector().detect_cells(frame[2])
        height, width = frame[2].shape[:2]
        valid_detections = []
        for detection in detections if detections is not None else []:
            try:
                x, y, confidence = map(float, detection)
            except (TypeError, ValueError):
                continue
            if np.isfinite([x, y, confidence]).all() and 0 <= round(x) < width and 0 <= round(y) < height:
                valid_detections.append((int(round(x)), int(round(y)), confidence))
        self.camera.show_circles([(x, y) for x, y, _ in valid_detections])
        if valid_detections:
            logging.info("Cell detector locations: %s", valid_detections)
        else:
            logging.info("Cell detector: no cells detected.")
        return valid_detections
