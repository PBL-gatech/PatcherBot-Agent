import sys
import time
import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
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
        super().__init__()

    @abstractmethod
    def detect_pipette(self, img: np.ndarray) -> Optional[Tuple[int, int]]:
        """Locate the pipette tip in the provided image."""
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
            _ = self.detect_pipette(img)
            end = time.perf_counter()
            timings.append(end - start)

        avg_time = sum(timings) / len(timings) if timings else float("nan")
        return timings, avg_time



class PipetteDetector1(PipetteDetector):
    """ONNX-based pipette detector (original implementation)."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        super().__init__()
        cur_file = Path(__file__).parent.absolute()
        default_model = cur_file / "pipetteModel" / "pipetteDetectorNet4.onnx"
        self.model_path = Path(model_path) if model_path is not None else default_model
        from neuron_detection.models import ModelFactory
        predictor = ModelFactory.create("onnx_centroid", {"source": self.model_path})
        self.yolo_net, self.output_layers = predictor.net, predictor.output_layers
        self.pipette_class = 0

    def detect_pipette(self, img: np.ndarray) -> Optional[Tuple[int, int]]:
        from neuron_detection.models import OnnxCentroidPredictor
        return OnnxCentroidPredictor.predict(img, self._forward)

    def _forward(self, blob: np.ndarray):
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
        if self._fallback is None:
            logger.info("Initializing OpenCV fallback for PipetteDetectorCuda1")
            self._fallback = PipetteDetector1(model_path=str(self.model_path))
            self.compute_device = "opencv"

    def detect_pipette(self, img: np.ndarray) -> Optional[Tuple[int, int]]:
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

        from neuron_detection.models import OnnxCentroidPredictor
        return OnnxCentroidPredictor.decode(outs, img.shape)


class PipetteDetector2(PipetteDetector):
    """DINO-based pipette detector using the transformer pipeline."""

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None) -> None:
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
        """Return the (x, y) position of the pipette tip or None if not detected."""
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
        """Return the most recent z-coordinate prediction, if available."""
        return self._last_z


class PipetteDetectorYOLO1(PipetteDetector):
    """YOLO (.pt)-based pipette detector with an API matching PipetteDetector1."""

    def __init__(self, model_path: Optional[str] = None,
                 device: Optional[str] = None,
                 imgsz: int = 640,
                 conf: float = 0.20) -> None:
        super().__init__()
        from neuron_detection.models import ModelFactory

        repository_root = Path(__file__).resolve().parents[1]
        default_model = repository_root / "training" / "PipetteDetectorNet7.pt"
        self.model_path = Path(model_path) if model_path is not None else default_model

        self.yolo_model = ModelFactory.create("yolo", {"source": str(self.model_path)})

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


class PipetteDetector3(PipetteDetector):
    """U-Net/LiteHRNet consumer with optional spatial outputs from one pass.

    By default returns the native-image integer (x, y), or None. With either
    output flag, returns a dictionary containing tip, diagnostics and the
    requested maps. Maps are independent float32 NumPy arrays in model space:
    heatmap is the raw decoder score map (not calibrated confidence), and mask
    is the 560x560 U-Net foreground probability map. Both span the full input
    image; array indexing is [y, x]. Map pixel centers map to native coordinates
    as (coordinate + 0.5) * native_size / map_size - 0.5.

    Invalid input returns None maps; a rejected tip can still have valid maps.
    Image-only checkpoints return mask=None because no U-Net mask was used.
    Like the parent, instances retain last_diagnostics and are not thread-safe.
    """

    def __init__(self, model_path=None, device=None, codec_options=None):
        super().__init__()
        import hashlib
        from neuron_detection.models.unet_memory import create_unet2d
        from neuron_detection.models import ModelFactory, PipetteHeatmapCodec
        if model_path is None:
            raise ValueError('Select a tip heatmap checkpoint in the application config')
        self.device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                                   if device is None or device == 'auto' else device)
        checkpoint = Path(model_path).resolve()
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        memory_payload = payload if payload.get('schema') in {'pipette_unet_memory_v1', 'pipette_unet_memory_v2'} else None
        self.uses_memory = memory_payload is not None
        if self.uses_memory:
            head = Path(memory_payload['head_checkpoint'])
            if not head.is_absolute():
                head = checkpoint.parent / head
            if hashlib.sha256(head.read_bytes()).hexdigest() != memory_payload['head_sha256']:
                raise ValueError('Frozen LiteHRNet checkpoint hash mismatch')
            payload = torch.load(head, map_location='cpu', weights_only=True)
            if payload.get('unet_sha256') != memory_payload['base_sha256']:
                raise ValueError('Memory base does not match the frozen LiteHRNet source UNet')
        if payload.get('schema') != 'pipette_tip_heatmap_v1':
            raise ValueError('Expected pipette_tip_heatmap_v1 or versioned pipette_unet_memory checkpoint')
        options = dict(payload['model_options'])
        options['initialize_pretrained'] = False
        self.model = ModelFactory.create('pipette_tip_heatmap', options)
        self.model.load_state_dict(payload['model'], strict=True)
        self.model.to(self.device).eval().requires_grad_(False)
        settings = dict(payload.get('codec_options', {}))
        settings.update(codec_options or {})
        self.codec = PipetteHeatmapCodec(**settings)
        self.input_mode = options.get('input_mode', 'image_mask')
        self.native_boundary_epsilon = float(payload.get('config', {}).get('native_boundary_epsilon', 1e-4))
        source = Path(memory_payload['base_checkpoint'] if self.uses_memory else payload['unet_checkpoint'])
        if not source.is_absolute():
            source = checkpoint.parent / source
        if hashlib.sha256(source.read_bytes()).hexdigest() != payload['unet_sha256']:
            raise ValueError('Original UNet source checkpoint hash mismatch')
        if self.uses_memory:
            from neuron_detection.models.unet_memory import PipetteMemoryUNet
            if self.input_mode != 'image_mask':
                raise ValueError('Memory checkpoints require an image_mask LiteHRNet head')
            self.unet = PipetteMemoryUNet.from_checkpoint(memory_payload, training_scope='inference')
            if memory_payload['schema'] == 'pipette_unet_memory_v2' and memory_payload['lineage']['stage'] == 'B':
                lineage = memory_payload['lineage']
                parent_path = Path(lineage['stage_a_checkpoint'])
                if not parent_path.is_absolute():
                    parent_path = checkpoint.parent / parent_path
                if hashlib.sha256(parent_path.read_bytes()).hexdigest() != lineage['stage_a_sha256']:
                    raise ValueError('Selected stage A checkpoint hash mismatch')
                parent = torch.load(parent_path, map_location='cpu', weights_only=True)
                if (parent.get('schema') != 'pipette_unet_memory_v2'
                        or parent.get('lineage', {}).get('stage') != 'A'
                        or parent.get('base_sha256') != memory_payload['base_sha256']
                        or parent.get('head_sha256') != memory_payload['head_sha256']):
                    raise ValueError('Selected stage A lineage differs from stage B')
                PipetteMemoryUNet._validate_lineage(parent)
                child_weights, parent_weights = memory_payload['unet_state_dict'], parent['unet_state_dict']
                if (child_weights.keys() != parent_weights.keys()
                        or any(value.shape != parent_weights[key].shape for key, value in child_weights.items())):
                    raise ValueError('Stage B changed the selected stage A single-image architecture')
                scope = memory_payload.get('training_scope', 'memory')
                decoder_prefixes = ('up_blocks.', 'conv_norm_out.', 'conv_out.')
                frozen_weights = (() if scope == 'all' else
                    (key for key in child_weights if not key.startswith(decoder_prefixes))
                    if scope == 'decoder_memory' else child_weights)
                if any(not torch.equal(child_weights[key], parent_weights[key]) for key in frozen_weights):
                    raise ValueError('Stage B changed the selected stage A single-image weights outside its declared training scope')
            if self.unet.config.sample_size not in (560, (560, 560), [560, 560]):
                raise ValueError('Memory checkpoints require 560-pixel full-frame inference')
        else:
            unet_payload = torch.load(source, map_location='cpu', weights_only=True)
            mask_options = payload.get('unet_options') or unet_payload['config']
            if mask_options.get('_class_name') == 'PipetteSegFormerB0':
                if self.codec.native_crop_size != 640 or mask_options != unet_payload['config']:
                    raise ValueError('SegFormer checkpoint requires matching 640px crop metadata')
                self.unet = ModelFactory.create('segformer', mask_options, task='amodal_mask', topology='pipette_b0')
            else:
                self.unet = create_unet2d(mask_options)
            self.unet.load_state_dict(unet_payload['model'], strict=True)
        self.unet.to(self.device).eval().requires_grad_(False)
        self.input_size = 560
        self.last_diagnostics = None

    def detect_pipette(self, img: np.ndarray, *, return_heatmap: bool = False,
                       return_mask: bool = False):
        if not return_heatmap and not return_mask:
            return self._detect_pipette(img)
        outputs = {}
        if return_heatmap:
            outputs['heatmap'] = None
        if return_mask:
            outputs['mask'] = None
        tip = self._detect_pipette(img, outputs)
        return dict(tip=tip, **outputs, diagnostics=dict(self.last_diagnostics))


    def detect_pipette_frame(self, img: np.ndarray, state=None, *,
                            return_heatmap: bool = False, return_mask: bool = False):
        """Return (ordinary detection result, next state) without retaining history.

        Pass None to start a new stream or reset memory. Keep a separate state
        for each stream. Invalid input leaves its supplied state unchanged;
        a rejected tip can still update memory from a valid image. Legacy
        checkpoints accept only state=None and return next_state=None.
        """
        if state is not None and not getattr(self, 'uses_memory', False):
            raise ValueError('This checkpoint does not support recurrent state')
        outputs = {}
        if return_heatmap:
            outputs['heatmap'] = None
        if return_mask:
            outputs['mask'] = None
        context = dict(state=state)
        tip = self._detect_pipette(img, outputs or None, state_context=context)
        result = (dict(tip=tip, **outputs, diagnostics=dict(self.last_diagnostics))
                  if return_heatmap or return_mask else tip)
        return result, context['state']

    @torch.inference_mode()
    def _detect_pipette(self, img: np.ndarray, outputs=None,
                        state_context=None, stage_timings=None) -> Optional[Tuple[int, int]]:
        # Used only in a separate diagnostic pass, never the primary latency loop.
        import time
        stage_started = None
        if stage_timings is not None:
            if self.device.type=='cuda':torch.cuda.synchronize(self.device)
            stage_started=time.perf_counter()
        def mark_stage(name):
            nonlocal stage_started
            if stage_timings is not None:
                if self.device.type=='cuda':torch.cuda.synchronize(self.device)
                now=time.perf_counter();stage_timings[name]=(now-stage_started)*1000;stage_started=now
        self.last_diagnostics = dict(valid=False, reason='invalid_input', tip_xy=None)
        if not isinstance(img, np.ndarray) or img.size == 0:
            return None
        if img.ndim == 2:
            gray = img
        elif img.ndim == 3 and img.shape[2] == 1:
            gray = img[..., 0]
        elif img.ndim == 3 and img.shape[2] in (3, 4):
            if not np.issubdtype(img.dtype, np.number):
                return None
            color = img if img.dtype in (np.uint8, np.uint16, np.float32) else img.astype(np.float32)
            gray = self._ensure_grayscale(color)
        else:
            return None
        if not (np.issubdtype(gray.dtype, np.integer) or np.issubdtype(gray.dtype, np.floating)):
            return None
        source = np.array(gray, dtype=np.float32, copy=True, order='C')
        if not np.isfinite(source).all() or source.min() < 0:
            return None
        if np.issubdtype(gray.dtype, np.floating) and source.max() <= 1:
            source *= 255
        if source.max() > 255:
            return None
        native_height, native_width = source.shape
        view_size, view_origin = self.codec.view_geometry((native_height,native_width))
        vx, vy = view_origin; vw, vh = view_size
        view = source[vy:vy+vh,vx:vx+vw].astype(np.uint8)
        resized = cv2.resize(view, (560, 560), interpolation=cv2.INTER_AREA)
        mark_stage('native_cpu_preprocessing')
        condition = torch.from_numpy(resized).to(self.device, dtype=torch.float32)[None, None] / 127.5 - 1
        mark_stage('input_transfer_normalization')
        if self.input_mode == 'image_only':
            mask = torch.zeros_like(condition)
        else:
            network_size = (self.unet.config.get('sample_size') if isinstance(self.unet.config,dict)
                            else getattr(self.unet.config, 'sample_size', None))
            network_shape = ((network_size, network_size) if isinstance(network_size, int)
                             else tuple(network_size) if network_size is not None else condition.shape[-2:])
            unet_input = (condition if tuple(condition.shape[-2:]) == tuple(network_shape) else
                          torch.nn.functional.interpolate(condition, size=network_shape, mode='area'))
            if network_shape==(768,768) or (isinstance(self.unet.config,dict) and self.unet.config.get('_class_name')=='PipetteSegFormerB0'):
                unet_input=torch.from_numpy(cv2.resize(view,network_shape[::-1],interpolation=cv2.INTER_AREA)).to(
                    self.device,dtype=torch.float32)[None,None]/127.5-1
            mark_stage('mask_input_preparation_transfer')
            if state_context is not None and getattr(self, 'uses_memory', False):
                logits, next_state = self.unet.forward_frame(unet_input, state_context['state'])
                state_context['state'] = next_state.detach()
            else:
                logits = self.unet(unet_input, torch.zeros(1, device=self.device)).sample
            mark_stage('mask_model_inference')
            if tuple(logits.shape[-2:]) != tuple(condition.shape[-2:]):
                logits = torch.nn.functional.interpolate(logits, size=condition.shape[-2:],
                                                         mode='bilinear', align_corners=False)
            probability = logits.sigmoid()
            mask = probability * 2 - 1
        pair=torch.cat((condition,mask),dim=1)
        mark_stage('mask_resize_and_pair')
        heatmaps = self.model(pair)
        mark_stage('litehrnet_inference')
        if outputs is not None:
            if 'heatmap' in outputs:
                outputs['heatmap'] = heatmaps[0, 0].detach().float().cpu().numpy().copy()
            if 'mask' in outputs and self.input_mode != 'image_only':
                outputs['mask'] = probability[0, 0].detach().float().cpu().numpy().copy()
        model_xy, diagnostics = self.codec.decode(heatmaps)
        native_xy = self.codec.model_to_native(model_xy[0],(native_height,native_width)).detach().cpu().numpy()
        diagnostic = dict(diagnostics[0])
        diagnostic.update(native_shape=[native_height, native_width],
                          input_mode=self.input_mode, unet_used=self.input_mode != 'image_only')
        if not np.isfinite(native_xy).all():
            diagnostic.update(valid=False, tip_xy=None)
            self.last_diagnostics = diagnostic
            mark_stage('heatmap_decode_native_extraction')
            return None
        public_xy, valid = self.codec.public_from_model(
            model_xy, (native_height, native_width), self.native_boundary_epsilon)
        if not bool(valid[0]):
            diagnostic.update(valid=False, reason='outside_native_image', tip_xy=None,
                              native_xy_unrounded=native_xy.tolist())
            self.last_diagnostics = diagnostic
            mark_stage('heatmap_decode_native_extraction')
            return None
        point = tuple(int(value) for value in public_xy[0].detach().cpu().tolist())
        diagnostic.update(valid=True, reason='ok', tip_xy=list(point), native_xy_unrounded=native_xy.tolist())
        self.last_diagnostics = diagnostic
        mark_stage('heatmap_decode_native_extraction')
        return point


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
        if model_path is None:
            repository_root = Path(__file__).resolve().parents[3]
            model_path = str(repository_root / "training" / "Pipette_Fresh_384_001_Patience10"
                             / "Fresh_384_pipettes_Patience10" / "epoch_0030.pt")
        adapter_module = _import_module_from_path(
            "pipette_detector_model_factory", self._model_factory_path()
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
    def _model_factory_path() -> Path:
        model_root = Path(__file__).resolve().parent
        adjacent = model_root / "PDModelFactory.py"
        return adjacent if adjacent.is_file() else model_root / "pipetteModel" / "PDModelFactory.py"

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

    def detect_pipette_details(self, img: np.ndarray, *, expected_tip_xy=None,
                               pixels_to_um=None, radius_um=50.0) -> dict:
        """Return XY, depth in microns, box confidence and optional Z confidence.

        Uses one inference pass. Missing/invalid predictions are None; box
        confidence is not a substitute for the optional Z-confidence head.
        """
        empty_prediction = dict(tip_xy=None, z_um=None, box_confidence=None, z_confidence=None,
                                box_xyxy=None, candidate_count=0, accepted_candidate_count=0,
                                radius_rejected_count=0)
        if img is None:
            return empty_prediction
        prediction = empty_prediction.copy()
        try:
            gray = self._to_uint8_grayscale(img)
            if expected_tip_xy is None:
                result = self.adapter.predict(gray, native_mask=False)
                prediction["candidate_count"] = int(result.get("tip_xy") is not None)
                prediction["accepted_candidate_count"] = prediction["candidate_count"]
            else:
                expected = np.asarray(expected_tip_xy, dtype=float)
                if expected.shape != (2,) or not np.isfinite(expected).all() or not callable(pixels_to_um):
                    raise ValueError("A finite expected XY and calibrated pixel-to-micron converter are required")
                if not np.isfinite(radius_um) or radius_um <= 0:
                    raise ValueError("Pipette candidate radius must be positive and finite")
                decoded = self.adapter.predict(gray, native_mask=False, max_detections=100)
                candidates = decoded.get("candidates", [])
                prediction["candidate_count"] = len(candidates)
                accepted = []
                for candidate in candidates:
                    tip = np.asarray(candidate.get("tip_xy"), dtype=float)
                    confidence = candidate.get("box_confidence")
                    if tip.shape != (2,) or not np.isfinite(tip).all() or confidence is None or not np.isfinite(confidence):
                        continue
                    offset = np.asarray(pixels_to_um(tip - expected), dtype=float).reshape(-1)[:2]
                    if offset.size != 2 or not np.isfinite(offset).all():
                        continue
                    if np.linalg.norm(offset) > radius_um:
                        prediction["radius_rejected_count"] += 1
                        continue
                    accepted.append(candidate)
                prediction["accepted_candidate_count"] = len(accepted)
                if not accepted:
                    return prediction
                result = max(accepted, key=lambda candidate: candidate["box_confidence"])
            tip = result["tip_xy"]
            if tip is None:
                return prediction
            prediction["tip_xy"] = tuple(int(round(v)) for v in tip)
            box = result.get("box_xyxy")
            if box is not None:
                box = np.asarray(box, dtype=float)
                if box.shape == (4,) and np.isfinite(box).all():
                    prediction["box_xyxy"] = tuple(float(value) for value in box)
            for key in ("z_um", "box_confidence", "z_confidence"):
                value = result.get(key)
                if value is not None and np.isfinite(value):
                    prediction[key] = float(value)
            if result.get("z_um") is not None and prediction["z_um"] is None:
                logger.warning("PipetteDetector4 Z prediction is nonfinite; returning tip only")
            return prediction
        except Exception as exc:
            logger.warning("PipetteDetector4 inference failed: %s", exc)
            return empty_prediction

    def detect_pipette(self, img: np.ndarray) -> Optional[Tuple[int, int]]:
        if img is None:
            return None
        try:
            gray = self._to_uint8_grayscale(img)
            result = self.adapter.predict(gray, native_mask=False)
            tip = result["tip_xy"]
            if tip is None:
                return None
            z = result["z_um"]
            if z is not None and not np.isfinite(z):
                logger.warning("PipetteDetector4 Z prediction is nonfinite; returning tip only")
            return tuple(int(round(v)) for v in tip)
        except Exception as exc:
            logger.warning("PipetteDetector4 inference failed: %s", exc)
            return None


class PipetteDetector5(PipetteDetector4):
    """Configurable R02/R12 detector with shared point and detailed output APIs.

    model_version="R02": full-frame AREA resize to 384, no contrast stretch.
    model_version="R12" (default): center crop 896, p1/p99 stretch, AREA to 512.
    Pass the matching checkpoint; resolution/schema mismatches raise on load.
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        *,
        model_version: str = "R12",
        threshold: float = 0.001,
        cuda_graph: bool = True,
        precision: Optional[str] = None,
    ) -> None:
        PipetteDetector.__init__(self)
        self._model_lock = RLock()
        self._adapter_options = dict(
            device=device, threshold=threshold,
            cuda_graph=cuda_graph, precision=precision,
        )
        adapter_module = _import_module_from_path(
            "pipette_detector_model_factory",
            self._model_factory_path(),
        )
        self._factory = adapter_module.PDModelFactory
        self.adapter = None
        self._active_model = None
        self.reload_model(model_path, model_version=model_version)

    def reload_model(self, model_path, *, model_version):
        """Load a checkpoint while retaining this detector's shared identity."""
        version = str(model_version).strip().upper()
        if version not in self._factory.MODEL_VERSIONS:
            raise ValueError(f"Unsupported model version: {model_version!r}")
        if model_path is None:
            if version != "R02":
                raise ValueError("An R12 checkpoint path is required")
            # Preserve Detector4's existing general-model fallback.
            model_path = (
                Path(__file__).resolve().parents[3] / "training"
                / "Pipette_Fresh_384_001_Patience10"
                / "Fresh_384_pipettes_Patience10" / "epoch_0030.pt"
            )
        selection = (Path(model_path).expanduser().resolve(), version)
        with self._model_lock:
            if self.adapter is not None and selection == self._active_model:
                return
            adapter = self._factory.create(
                model_type=self._factory.MODEL_VERSIONS[version],
                model_path=str(selection[0]), **self._adapter_options,
            )
            self.adapter = adapter
            self.model_version = version
            self._active_model = selection
            logger.info("Pipette model loaded: %s (%s)", *selection)

    @contextmanager
    def use_model(self, model_path, *, model_version):
        """Use a checkpoint until the caller exits, then restore its predecessor."""
        with self._model_lock:
            self._require_model()
            previous = self.adapter, self.model_version, self._active_model
            self.reload_model(model_path, model_version=model_version)
        try:
            yield self
        finally:
            with self._model_lock:
                self.adapter, self.model_version, self._active_model = previous

    def _require_model(self):
        if self.adapter is None:
            raise RuntimeError("Pipette detector has no usable model")

    def detect_pipette(self, img):
        with self._model_lock:
            self._require_model()
            return super().detect_pipette(img)

    def detect_pipette_details(
        self, img, *, expected_tip_xy=None, pixels_to_um=None, radius_um=50.0
    ):
        with self._model_lock:
            self._require_model()
            result = super().detect_pipette_details(
                img, expected_tip_xy=expected_tip_xy,
                pixels_to_um=pixels_to_um, radius_um=radius_um,
            )
            result["z_confidence_supported"] = (
                self.adapter.model.z_confidence_head is not None
            )
            return result

    def get_pipette_z(self, img):
        return self.detect_pipette_details(img)["z_um"]

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
