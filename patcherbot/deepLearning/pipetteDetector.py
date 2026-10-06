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


class PipetteDetector(ABC):
    """Abstract base class for pipette detectors."""

    def __init__(self) -> None:
        """
        Initialize the base pipette detector.
        """
        super().__init__()

    @abstractmethod
    def detect_pipette(self, img: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Locate the pipette tip in the provided image.
        
        Args:
            img (np.ndarray): Input image array (H x W x C or H x W).

        Returns:
            Optional[Tuple[int, int]]: (x, y) pixel coordinates of the pipette tip,
            or None if no tip is detected.
        """
        raise NotImplementedError

    @staticmethod
    def _ensure_color(img: np.ndarray) -> np.ndarray:
        """
        Ensure the image has three channels for models that expect color input.
        
        Args:
            img (np.ndarray): Input image.

        Returns:
            np.ndarray: Image with 3 channels (H x W x 3).
        """
        if img is None:
            return img
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img

    @staticmethod
    def _ensure_grayscale(img: np.ndarray) -> np.ndarray:
        """
        Convert a color image to grayscale if required.
        
        Args:
            img (np.ndarray): Input image.

        Returns:
            np.ndarray: Grayscale image (H x W).
        """
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
            _ = self.detect_pipette(img)
            end = time.perf_counter()
            timings.append(end - start)

        avg_time = sum(timings) / len(timings) if timings else float("nan")
        return timings, avg_time



class PipetteDetector1(PipetteDetector):
    """ONNX-based pipette detector (original implementation)."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        """
        Initialize the ONNX pipette detector.

        Args:
            model_path (Optional[str]): Path to the ONNX model file. Defaults to
                'pipette-small.onnx' in the pipetteModel folder.
        """
        super().__init__()
        cur_file = Path(__file__).parent.absolute()
        default_model = cur_file / "pipetteModel" / "BoPipetteDetectorNet.onnx"
        self.model_path = Path(model_path) if model_path is not None else default_model
        self.yolo_net = cv2.dnn.readNetFromONNX(str(self.model_path))
        layer_names = self.yolo_net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.yolo_net.getUnconnectedOutLayers()]
        self.pipette_class = 0

    def detect_pipette(self, img: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Return the (x, y) position of the pipette tip or None if not detected.
        
        Args:
            img (np.ndarray): Input image array (H x W x C or H x W).

        Returns:
            Optional[Tuple[int, int]]: (x, y) pixel coordinates of the pipette tip,
            or None if not detected.
        """
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
        """
        Run a forward pass through the ONNX DNN model.

        Args:
            blob (np.ndarray): Preprocessed image blob.

        Returns:
            List[np.ndarray]: Model outputs for pipette detection.
        """
        self.yolo_net.setInput(blob)
        return self.yolo_net.forward(self.output_layers)


class PipetteDetectorCuda1(PipetteDetector):
    """Pipette detector that uses onnxruntime-gpu and falls back to OpenCV DNN."""

    _GPU_PROVIDERS = (
        "CUDAExecutionProvider",
        "ROCMExecutionProvider",
        "DirectMLExecutionProvider",
        "DmlExecutionProvider",
    )

    def __init__(self, model_path: Optional[str] = None) -> None:
        """
        Initialize the GPU pipette detector.

        Args:
            model_path (Optional[str]): Path to the ONNX model file.
        """
        super().__init__()
        cur_file = Path(__file__).parent.absolute()
        default_model = cur_file / "pipetteModel" / "pipetteDetectorNet4.onnx"
        self.model_path = Path(model_path) if model_path is not None else default_model

        self.pipette_class = 0
        self._ort_session = None
        self._ort_input_name: Optional[str] = None
        self._ort_output_names: Tuple[str, ...] = ()
        self._fallback: Optional[PipetteDetector1] = None
        self.compute_device = "unknown"

        if not self._init_onnxruntime():
            self._ensure_fallback()

    def _init_onnxruntime(self) -> bool:
        """
        Initialize the onnxruntime session with GPU providers.

        Returns:
            bool: True if GPU session initialized successfully, False otherwise.
        """
        try:
            import onnxruntime as ort
        except ImportError:
            logger.info("onnxruntime is not installed; using OpenCV fallback")
            return False

        available = ort.get_available_providers()
        providers = [provider for provider in self._GPU_PROVIDERS if provider in available]
        if not providers:
            logger.info("No onnxruntime GPU providers detected; using OpenCV fallback")
            return False
        if "CPUExecutionProvider" in available:
            providers.append("CPUExecutionProvider")

        try:
            session = ort.InferenceSession(str(self.model_path), providers=providers)
        except Exception as exc:
            logger.warning("Failed to create onnxruntime session with providers %s: %s", providers, exc)
            return False

        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if not inputs:
            logger.warning("onnxruntime session has no inputs; using OpenCV fallback")
            return False

        self._ort_session = session
        self._ort_input_name = inputs[0].name
        self._ort_output_names = tuple(out.name for out in outputs if out.name)
        active_provider = session.get_providers()[0] if session.get_providers() else "onnxruntime"
        self.compute_device = active_provider
        logger.info("PipetteDetectorCuda1 using onnxruntime provider %s", active_provider)
        return True

    def _ensure_fallback(self) -> None:
        """
        Initialize the OpenCV fallback detector if GPU inference fails.
        """
        if self._fallback is None:
            logger.info("Initializing OpenCV fallback for PipetteDetectorCuda1")
            self._fallback = PipetteDetector1(model_path=str(self.model_path))
            self.compute_device = "opencv"

    def detect_pipette(self, img: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Detect the pipette tip using GPU-accelerated onnxruntime or fallback.

        Args:
            img (np.ndarray): Input image array (H x W x C or H x W).

        Returns:
            Optional[Tuple[int, int]]: (x, y) pixel coordinates of the pipette tip,
            or None if detection fails.
        """
        if self._ort_session is None:
            self._ensure_fallback()
            return self._fallback.detect_pipette(img) if self._fallback else None

        img = self._ensure_color(img)
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (640, 640), swapRB=True, crop=False)

        try:
            outs = self._ort_session.run(self._ort_output_names or None, {self._ort_input_name: blob})
        except Exception as exc:
            logger.warning("onnxruntime inference failed; switching to OpenCV fallback: %s", exc)
            self._ort_session = None
            self._ort_input_name = None
            self._ort_output_names = ()
            self._ensure_fallback()
            return self._fallback.detect_pipette(img) if self._fallback else None

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


class PipetteDetector2(PipetteDetector):
    """DINO-based pipette detector using the transformer pipeline."""

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None) -> None:
        """
        Initialize the DINO transformer-based detector.

        Args:
            model_path (Optional[str]): Path to the PyTorch model file (.pt).
            device (Optional[str]): Torch device to run inference on (e.g., 'cuda' or 'cpu').
        """
        super().__init__()
        src_dir = Path(__file__).parent / "pipetteModel" / "holypipette_pipette_detection" / "src"
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        model_module = _import_module_from_path("holypipette_pipette_detection.model", src_dir / "model.py")
        pipeline_module = _import_module_from_path("holypipette_pipette_detection.pipeline", src_dir / "pipeline.py")

        DINOPipetteDetector = model_module.DINOPipetteDetector
        get_pipeline = pipeline_module.get_pipeline

        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DINOPipetteDetector()

        default_model = (
            Path(__file__).parent
            / "pipetteModel"
            / "holypipette_pipette_detection"
            / "models"
            / "DINOPipetteDetector.pt"
        )
        self.model_path = Path(model_path) if model_path is not None else default_model

        state_dict = torch.load(self.model_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

        self.pipeline = get_pipeline(self.model, self.device)
        self._last_z: Optional[float] = None

    def detect_pipette(self, img: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Return the (x, y) position of the pipette tip or None if not detected.
        
        Args:
            img (np.ndarray): Input image array (H x W x C or H x W).

        Returns:
            Optional[Tuple[int, int]]: (x, y) pixel coordinates of the pipette tip,
            or None if detection fails.
        """
        if img is None:
            return None

        gray = self._ensure_grayscale(img)
        try:
            xy_pred, z_coord_pred = self.pipeline.get_model_prediction(gray)
        except Exception:
            return None

        xy_tensor = xy_pred[0].detach().cpu().numpy()
        x_offset, y_offset = float(xy_tensor[0]), float(xy_tensor[1])
        height, width = gray.shape

        x_pred = int(round(x_offset + width // 2))
        y_pred = int(round(y_offset + height // 2))

        z_tensor = z_coord_pred.detach().cpu().numpy()
        self._last_z = float(z_tensor[0]) if z_tensor.size else None

        if not (0 <= x_pred < width and 0 <= y_pred < height):
            return None

        return x_pred, y_pred

    @property
    def last_depth_prediction(self) -> Optional[float]:
        """
        Return the most recent z-coordinate prediction, if available.
        
        Returns:
            Optional[float]: Predicted depth in pixels, or None if unavailable.
        """
        return self._last_z


class PipetteDetectorYOLO1(PipetteDetector):
    """YOLO (.pt)-based pipette detector with an API matching PipetteDetector1."""

    def __init__(self, model_path: Optional[str] = None,
                 device: Optional[str] = None,
                 imgsz: int = 640,
                 conf: float = 0.20) -> None:
        """
        Initialize the YOLO-based pipette detector.

        Args:
            model_path (Optional[str]): Path to the YOLO .pt model.
            device (Optional[str]): Torch device to run inference on.
            imgsz (int): Input image size for YOLO (square).
            conf (float): Confidence threshold for detection.
        """
        super().__init__()
        from ultralytics import YOLO

        cur_file = Path(__file__).parent.absolute()
        default_model = cur_file / "pipetteModel" / "BoRigCombo.pt"
        self.model_path = Path(model_path) if model_path is not None else default_model

        self.yolo_model = YOLO(str(self.model_path))

        # Match PipetteDetector1 behavior
        self.pipette_class = 0
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

    def detect_pipette(self, img: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Detect the pipette tip using YOLOv8.

        Args:
            img (np.ndarray): Input image array (H x W x C or H x W).

        Returns:
            Optional[Tuple[int, int]]: (x, y) pixel coordinates of the pipette tip,
            or None if detection fails.
        """
        if img is None:
            return None

        img = self._ensure_color(img)
        img = np.ascontiguousarray(img)  # avoid extra copies in preprocessing
        h, w = img.shape[:2]

        try:
            results = self.yolo_model.predict(
                source=img,
                imgsz=self.imgsz,
                conf=self.conf_threshold,
                device=self.device,
                half=self.use_half,
                verbose=False,
            )
        except Exception as exc:
            logger.warning("YOLO inference failed: %s", exc)
            return None

        if not results or results[0] is None or results[0].boxes is None:
            return None

        boxes = results[0].boxes
        try:
            cls = boxes.cls.detach().cpu().numpy().astype(int)
            conf = boxes.conf.detach().cpu().numpy()
            xywhn = boxes.xywhn.detach().cpu().numpy()
        except Exception:
            return None

        mask = (cls == self.pipette_class) & (conf >= self.conf_threshold)
        if not np.any(mask):
            return None

        conf_sel = conf[mask]
        xywhn_sel = xywhn[mask]
        best_idx = int(np.argmax(conf_sel))
        cx_n, cy_n = float(xywhn_sel[best_idx, 0]), float(xywhn_sel[best_idx, 1])

        if np.isnan(cx_n) or np.isnan(cy_n):
            return None

        x_pix = int(round(cx_n * w))
        y_pix = int(round(cy_n * h))
        if not (0 <= x_pix < w and 0 <= y_pix < h):
            return None
        return x_pix, y_pix
    
class PipetteDetector4(PipetteDetector):
    """Canonical 384px U-Net + MobileNetV4 boxes pipette detector.

    Loads the canonical 384px grayscale+Sobel pipette checkpoint through
    PDModelFactory (loaded by path, mirroring the CellModel pattern). The
    default model path is the verified best checkpoint at the repository root:
    training/Pipette_Fresh_384_001_Patience10/Fresh_384_pipettes_Patience10/epoch_0030.pt
    (schema canonical_checkpoint_v1, family unet_mobilenetv4_boxes,
    input_mode grayscale_sobel_magnitude, resolution 384).

    CUDA uses BF16 and a captured model/decoder-preparation graph by default;
    CPU uses FP32. Pass cuda_graph=False for eager execution.
    """

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None,
                 *, threshold: float = 0.001, cuda_graph: bool = True,
                 precision: Optional[str] = None) -> None:
        super().__init__()
        model_root = Path(__file__).parent.absolute()
        if model_path is None:
            model_path = str(model_root / "pipetteModel" / "PipetteDetectorNetX.pt")
        adapter_module = _import_module_from_path(
            "pipette_detector_model_factory", model_root / "pipetteModel" / "PDModelFactory.py"
        )
        self.adapter = adapter_module.PDModelFactory.create(
            model_type="unet_mobilenetv4_boxes",
            model_path=model_path,
            device=device,
            threshold=threshold,
            cuda_graph=cuda_graph,
            precision=precision,
        )

    @staticmethod
    def _to_uint8_grayscale(img: np.ndarray) -> np.ndarray:
        """Normalize family inputs (2D/1ch/3-4ch, float or uint8) to uint8 grayscale."""
        if img.ndim == 3:
            if img.shape[2] == 1:
                img = img[:, :, 0]
            else:
                img = PipetteDetector._ensure_grayscale(img)
        if img.dtype != np.uint8:
            img = np.asarray(img, dtype=np.float64)
            if img.size == 0:
                raise ValueError("Expected a nonempty image")
            peak = float(img.max())
            if peak > 1.0:
                if peak > 255.0:
                    raise ValueError("Image intensity exceeds uint8 range")
                img = img
            else:
                img = img * 255.0
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def detect_pipette_tracking(self, img: np.ndarray) -> dict:
        """Return an independent native-size mask, grayscale frame and subpixel tip."""
        prediction = dict(tip_xy=None, mask=None, frame=None)
        if img is None:
            return prediction
        try:
            gray = self._to_uint8_grayscale(img)
            if gray.ndim != 2 or gray.size == 0:
                raise ValueError("Expected a nonempty grayscale image")
            result = self.adapter.predict(gray, native_mask=True)
            prediction["frame"] = gray.copy()
            tip = result.get("tip_xy")
            if tip is not None:
                tip = np.asarray(tip, dtype=float)
                if tip.shape == (2,) and np.isfinite(tip).all():
                    prediction["tip_xy"] = tuple(float(value) for value in tip)
                else:
                    logger.warning("PipetteDetector4 tracking tip has invalid shape or values")
            mask = result.get("mask")
            if mask is not None:
                mask = np.asarray(mask, dtype=np.float32)
                if mask.shape == gray.shape and np.isfinite(mask).all():
                    prediction["mask"] = mask.copy()
                else:
                    logger.warning("PipetteDetector4 tracking mask has invalid shape or values")
            return prediction
        except Exception as exc:
            logger.warning("PipetteDetector4 tracking inference failed: %s", exc)
            return dict(tip_xy=None, mask=None, frame=None)

    def detect_pipette_details(self, img: np.ndarray) -> dict:
        """Return XY, depth in microns, box confidence and optional Z confidence.

        Uses one inference pass. Missing/invalid predictions are None; box
        confidence is not a substitute for the optional Z-confidence head.
        """
        prediction = dict(tip_xy=None, z_um=None, box_confidence=None, z_confidence=None)
        if img is None:
            return prediction
        try:
            gray = self._to_uint8_grayscale(img)
            result = self.adapter.predict(gray, native_mask=False)
            tip = result["tip_xy"]
            if tip is None:
                return prediction
            prediction["tip_xy"] = tuple(int(round(v)) for v in tip)
            for key in ("z_um", "box_confidence", "z_confidence"):
                value = result.get(key)
                if value is not None and np.isfinite(value):
                    prediction[key] = float(value)
            if result.get("z_um") is not None and prediction["z_um"] is None:
                logger.warning("PipetteDetector4 Z prediction is nonfinite; returning tip only")
            return prediction
        except Exception as exc:
            logger.warning("PipetteDetector4 inference failed: %s", exc)
            return dict(tip_xy=None, z_um=None, box_confidence=None, z_confidence=None)

    def detect_pipette(self, img: np.ndarray) -> Optional[Tuple[int, int]]:
        """Preserve the existing integer XY-only detection interface."""
        return self.detect_pipette_details(img)["tip_xy"]

    def get_pipette_z(self, img: np.ndarray) -> Optional[float]:
        """Infer depth in microns, or None when unavailable."""
        return self.detect_pipette_details(img)["z_um"]

    def get_pipette_confidence(self, img: np.ndarray) -> Optional[float]:
        """Infer box detection confidence, or None when unavailable."""
        return self.detect_pipette_details(img)["box_confidence"]

    def get_pipette_z_confidence(self, img: np.ndarray) -> Optional[float]:
        """Infer Z confidence, or None if the checkpoint has no such head."""
        return self.detect_pipette_details(img)["z_confidence"]


_DEFAULT_DETECTOR = None


def configure_pipette_detector(detector):
    """Select a loaded detector for the optional module-level point API."""
    if not callable(getattr(detector, "detect_pipette", None)):
        raise TypeError("detector must implement detect_pipette(image)")
    global _DEFAULT_DETECTOR
    _DEFAULT_DETECTOR = detector


def detect_pipette(img):
    """Return a tip or None using the detector selected at application startup."""
    if _DEFAULT_DETECTOR is None:
        raise RuntimeError("Call configure_pipette_detector with a loaded detector first")
    return _DEFAULT_DETECTOR.detect_pipette(img)



if __name__ == '__main__':
    detector = PipetteDetector4()
    # path = r"C:\Users\sa-forest\Documents\GitHub\Neuron_Detection\codex_training\personal_training\combined_net6\test\images\2026_08_26-16_43__2298_1787777086.380385.webp"
    # path = r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\experiments\Data\snap_image_data\2026_08_27-17_08\camera_frames\3333_1787864986.209094.webp"
    # path = r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\experiments\Data\snap_image_data\2026_08_27-17_08\camera_frames\8352_1787865160.058934.webp"
    path = r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\experiments\Data\snap_image_data\2026_08_27-18_41\camera_frames\1864_1787870571.912273.webp"
    # detector = PipetteDetectorYOLO1()
    # path = r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\experiments\Data\snap_image_data\2026_01_27-14_57\camera_frames\22276_1769544568.447358.webp"
    # # detector = PipetteDetector4()
    # # # path = r"C:\Users\sa-forest\Documents\GitHub\Neuron_Detection\codex_training\personal_training\combined_net6\test\images\2026_08_26-16_43__2298_1787777086.380385.webp"
    # # # path = r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\experiments\Data\snap_image_data\2026_08_27-17_08\camera_frames\3333_1787864986.209094.webp"
    # # # path = r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\experiments\Data\snap_image_data\2026_08_27-17_08\camera_frames\8352_1787865160.058934.webp"
    # # path = r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\experiments\Data\snap_image_data\2026_08_27-18_41\camera_frames\1864_1787870571.912273.webp"
    
    img = cv2.imread(path)

    values = detector._test_detector(img,10)
    print(f'Test timings (s): {values[0]}, average: {values[1]}')

    start = time.time()
    result = detector.detect_pipette(img)
    end = time.time()

    if result is not None:
        x, y = result
        print(f'framerate: {1 / (end - start)}')
        cv2.circle(img, (x, y), 3, (0, 255, 0))
        cv2.imshow("pipette detection test", img)
        cv2.waitKey(0)
    else:
        print("No pipette detected")
