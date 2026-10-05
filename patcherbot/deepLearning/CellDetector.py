import sys
import time
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
import torch
import importlib.util


logger = logging.getLogger(__name__)


def _import_module_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return module


class CellDetector(ABC):
    """Abstract base class for cell detectors."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def detect_cell(self, img: np.ndarray) -> Optional[Tuple[int, int]]:
        """Locate the cell tip in the provided image."""
        raise NotImplementedError

    @staticmethod
    def _ensure_color(img: np.ndarray) -> np.ndarray:
        """Ensure the image has three channels for models that expect color input."""
        if img is None:
            return img
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img

    @staticmethod
    def _ensure_grayscale(img: np.ndarray) -> np.ndarray:
        """Convert a color image to grayscale if required."""
        if img is None:
            return img
        if len(img.shape) == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _test_detector(self, img: np.ndarray,repititions: int) -> Tuple[List[float], float]:
        """Run inference on the same image a certain number of  times and return per-run and average durations (seconds)."""
        timings: List[float] = []
        for _ in range(repititions):
            start = time.perf_counter()
            _ = self.detect_cell(img)
            end = time.perf_counter()
            timings.append(end - start)

        avg_time = sum(timings) / len(timings) if timings else float("nan")
        return timings, avg_time

class CellDetector1(CellDetector):
    """ONNX-based cell detector (original implementation)."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        super().__init__()
        cur_file = Path(__file__).parent.absolute()
        default_model = cur_file / "cellModel" / "cellDetectorNet4.onnx"
        self.model_path = Path(model_path) if model_path is not None else default_model
        self.yolo_net = cv2.dnn.readNetFromONNX(str(self.model_path))
        layer_names = self.yolo_net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.yolo_net.getUnconnectedOutLayers()]
        self.cell_class = 0

    def detect_cell(self, img: np.ndarray) -> Optional[Tuple[int, int]]:
        """Return the (x, y) position of the cell tip or None if not detected."""
        img = self._ensure_color(img)
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (640, 640), swapRB=True, crop=False)
        outs = self._forward(blob)

        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                idx = np.argmax(detection[4, :])
                detection = detection[:, idx]
                x, y, width, height, objectness = tuple(detection)
                if objectness < 0.20:
                    continue

                boxes.append([x, y])
                confidences.append(float(objectness))

        if len(boxes) == 0:
            return None

        confidences = np.array(confidences)
        best_x, best_y = boxes[confidences.argmax()]
        if np.isnan(best_x) or np.isnan(best_y):
            return None

        best_x = (best_x / 640) * img.shape[1]
        best_y = (best_y / 640) * img.shape[0]

        return int(best_x), int(best_y)

    def _forward(self, blob: np.ndarray):
        self.yolo_net.setInput(blob)
        return self.yolo_net.forward(self.output_layers)

class CellDetectorYOLO1(CellDetector):
    """YOLO (.pt)-based cell detector with an API matching CellDetector1."""

    def __init__(self, model_path: Optional[str] = None,
                 device: Optional[str] = None,
                 imgsz: int = 1024,
                 conf: float = 0.20) -> None:
        super().__init__()
        from ultralytics import YOLO

        cur_file = Path(__file__).parent.absolute()
        default_model = cur_file / "cellModel" / "cellDetectorNet2.pt"
        self.model_path = Path(model_path) if model_path is not None else default_model

        self.yolo_model = YOLO(str(self.model_path))

        # Match CellDetector1 behavior
        self.cell_class = 0
        self.imgsz = imgsz
        self.conf_threshold = conf

        # ---- Speed flags
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.use_half = (self.device != "cpu")
        # Optional: only enable half if GPU supports fast FP16 (most do)
        if self.device.startswith("cuda"):
            try:
                major, _ = torch.cuda.get_device_capability(0)
                self.use_half = self.use_half and (major >= 7)
            except Exception:
                pass

            torch.backends.cudnn.benchmark = True  # fixed-size 640x640

        # Put model on device (Ultralytics will do it, but we make it explicit)
        try:
            self.yolo_model.to(self.device)
        except Exception:
            pass

        # ---- Warmup once to compile kernels / allocate memory
        if self.device.startswith("cuda"):
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            _ = self.yolo_model.predict(
                source=dummy,
                imgsz=self.imgsz,
                conf=0.01,
                device=self.device,
                half=self.use_half,
                verbose=False,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    def detect_cell(self, img: np.ndarray) -> Optional[Tuple[int, int]]:
        detections = self.detect_cells(img, max_outputs=1)
        if not detections:
            return None
        x_pix, y_pix, _ = detections[0]
        return x_pix, y_pix

    def detect_cells(
        self, img: np.ndarray, max_outputs: Optional[int] = None
    ) -> List[Tuple[int, int, float]]:
        """Return up to max_outputs detections as (x, y, confidence), sorted by confidence."""
        if img is None:
            return []

        img = self._ensure_color(img)
        img = np.ascontiguousarray(img)  # avoid extra copies in preprocessing
        h, w = img.shape[:2]

        predict_kwargs = dict(
            source=img,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            device=self.device,
            half=self.use_half,
            verbose=False,
        )
        if max_outputs is not None and max_outputs > 0:
            predict_kwargs["max_det"] = int(max_outputs)

        try:
            results = self.yolo_model.predict(**predict_kwargs)
        except Exception as exc:
            logger.warning("YOLO inference failed: %s", exc)
            return []

        if not results or results[0] is None or results[0].boxes is None:
            return []

        boxes = results[0].boxes
        try:
            cls = boxes.cls.detach().cpu().numpy().astype(int)
            conf = boxes.conf.detach().cpu().numpy()
            xywhn = boxes.xywhn.detach().cpu().numpy()
        except Exception:
            return []

        mask = (cls == self.cell_class) & (conf >= self.conf_threshold)
        if not np.any(mask):
            return []

        conf_sel = conf[mask]
        xywhn_sel = xywhn[mask]
        order = np.argsort(conf_sel)[::-1]
        if max_outputs is not None:
            order = order[:max(0, int(max_outputs))]

        detections: List[Tuple[int, int, float]] = []
        for idx in order:
            cx_n = float(xywhn_sel[idx, 0])
            cy_n = float(xywhn_sel[idx, 1])
            if np.isnan(cx_n) or np.isnan(cy_n):
                continue

            x_pix = int(round(cx_n * w))
            y_pix = int(round(cy_n * h))
            if not (0 <= x_pix < w and 0 <= y_pix < h):
                continue

            detections.append((x_pix, y_pix, float(conf_sel[idx])))

        return detections


class CellDetector2(CellDetector):
    """Factory-backed dense detector with the CellDetector centroid API."""

    def __init__(
        self,
        model_type: str = "segformer",
        model_path: Optional[str | Path] = None,
        encoder_config_path: Optional[str | Path] = None,
        device: Optional[str] = None,
        conf: float = 0.20,
    ) -> None:
        super().__init__()
        cur_file = Path(__file__).parent.absolute()
        model_root = cur_file / "cellModel"
        self.model_type = str(model_type).strip().lower()
        default_models = {
            "segformer": model_root / "segformer_best_map50.pt",
            "pidnet": model_root / "pidnet_best_map50.pt",
        }
        if self.model_type not in default_models:
            supported = ", ".join(default_models)
            raise ValueError(
                f"Unsupported model_type {model_type!r}; expected one of: {supported}"
            )
        self.model_path = (
            Path(model_path)
            if model_path is not None
            else default_models[self.model_type]
        )
        self.encoder_config_path = (
            Path(encoder_config_path)
            if encoder_config_path is not None
            else model_root / "config.json"
            if self.model_type == "segformer"
            else None
        )
        adapter_module = _import_module_from_path(
            "cell_detector_model_factory", model_root / "CDModelFactory.py"
        )
        self.adapter = adapter_module.CDModelFactory.create(
            model_type=self.model_type,
            model_path=self.model_path,
            encoder_config_path=self.encoder_config_path,
            device=device,
            conf=conf,
        )
        self.cell_class = 0
        self.conf_threshold = float(conf)
        self.device = str(self.adapter.device)

    def detect_cell(self, img: np.ndarray) -> Optional[Tuple[int, int]]:
        detections = self.detect_cells(img, max_outputs=1)
        if not detections:
            return None
        x_pix, y_pix, _ = detections[0]
        return x_pix, y_pix

    def detect_cells(
        self, img: np.ndarray, max_outputs: Optional[int] = None
    ) -> List[Tuple[int, int, float]]:
        """Return detections as (x, y, confidence), sorted by confidence."""
        if img is None:
            return []
        try:
            return self.adapter.predict(img, max_outputs=max_outputs)
        except Exception as exc:
            logger.warning("%s inference failed: %s", self.model_type, exc)
            return []
    
if __name__ == '__main__':
    # detector = CellDetectorYOLO1()
    # detector = CellDetector2(device="cuda:0")
    detector = CellDetector2(model_type="pidnet", device="cuda:0")
    # path = r"C:\Users\sa-forest\GaTech Dropbox\Benjamin Magondu\YOLOretrainingdata\Cell CNN Training Data\20191016\3654098923.png"
    # path = r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\experiments\Data\rig_recorder_data\2026_02_25-14_56\camera_frames\26197_1772050291.581619.webp"
    # path = r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\experiments\Data\patch_clamp_data\2026_02_25-14_56\CellMetadata\cell_5.webp"
    # path = r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\experiments\Data\patch_clamp_data\2025_10_29-18_56\CellMetadata\cell_9.webp"
    # path = r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\experiments\Data\patch_clamp_data\2025_10_16-19_24\CellMetadata\cell_8.webp"
    path = r"C:\Users\sa-forest\Downloads\image (3).png"
    img = cv2.imread(path)

    values = detector._test_detector(img,10)
    print(f'Test timings (s): {values[0]}, average: {values[1]}')

    start = time.time()
    results = detector.detect_cells(img, max_outputs=20)
    end = time.time()

    if results:
        print(f'framerate: {1 / (end - start)}')
        for x, y, conf in results:
            cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
            cv2.putText(
                img,
                f"{conf:.2f}",
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        cv2.imshow("cell detection test", img)
        cv2.waitKey(0)
    else:
        print("No cell detected")
