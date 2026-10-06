import os
import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
from abc import ABC, abstractmethod

# ================= BASE SEGMENTOR =================
class BaseSegmentor(ABC):
    """Common functionality for cell image segmentation across different SAM backends."""

    def __init__(self, device: str | None = None, cache_image_embeddings: bool = True):
        """
        Initialize the segmentor, resolving device and loading the model.

        args:
            device (str | None): Explicit device string (e.g., "cuda", "cpu"). If None, auto-detect.
        """
        self.device = self._resolve_device(device)
        self.cache_image_embeddings = cache_image_embeddings
        self._raw_image_pil = None
        self._image_embeddings = None
        self.model = None
        self.processor = None
        self.image = None
        self._load_model()

    @staticmethod
    def _resolve_device(explicit: str | None):
        """
        Determine the torch device to use.

        args:
            explicit (str | None): Explicit device string.

        returns:
            torch.device: Selected device.
        """
        if explicit is not None:
            return torch.device(explicit)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @abstractmethod
    def _load_model(self):
        """Load the SAM model and predictor. Must be implemented by subclasses."""

    @abstractmethod
    def _prepare_inputs(self, input_point, input_label, input_box, multimask_output):
        """Child class prepares model inputs (including any wrapping or embeddings)."""
        ...

    @abstractmethod
    def _forward(self, inputs, multimask_output):
        """Child class runs the model and returns (pred_masks, scores)."""
        ...

    @abstractmethod
    def _post_process(self, pred_masks: torch.Tensor, inputs: dict):
        """Child class converts raw model masks to image space."""
        ...

    @abstractmethod
    def _normalize_masks(self, masks):
        """Child class normalizes masks to numpy arrays."""
        ...

    @abstractmethod
    def _normalize_scores(self, iou_scores):
        """Child class normalizes scores to numpy arrays."""
        ...

    @abstractmethod
    def _cache_image_embeddings(self):
        """Child class optionally computes and stores image embeddings."""
        ...

    @staticmethod
    def _to_pil(rgb_image):
        from PIL import Image

        if isinstance(rgb_image, Image.Image):
            return rgb_image.convert("RGB")
        if isinstance(rgb_image, np.ndarray):
            if rgb_image.dtype != np.uint8:
                rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)
            return Image.fromarray(rgb_image)
        raise TypeError(f"Unsupported image type: {type(rgb_image)}. Expected numpy RGB array or PIL.Image.")

    def _require_ready(self):
        if self.model is None or self.processor is None:
            raise RuntimeError("Transformers model/processor not loaded. Check your segmentor _load_model().")

    def _predict(self, *, input_points=None, input_labels=None, input_boxes=None, multimask_output=True):
        if self.image is None or self._raw_image_pil is None:
            raise ValueError("Image has not been set. Please call set_image() before predicting.")

        self._require_ready()

        inputs = self._prepare_inputs(input_points, input_labels, input_boxes, multimask_output)
        pred_masks, iou_scores = self._forward(inputs, multimask_output)
        masks_pp = self._post_process(pred_masks, inputs)
        masks_np = self._normalize_masks(masks_pp)
        scores_np = self._normalize_scores(iou_scores)
        return masks_np, scores_np

    @abstractmethod
    def _prepare_inputs(self, input_point, input_label, input_box, multimask_output):
        """Child class prepares model inputs (including any wrapping or embeddings)."""
        ...

    @abstractmethod
    def _forward(self, inputs, multimask_output):
        """Child class runs the model and returns (pred_masks, scores)."""
        ...

    @abstractmethod
    def _post_process(self, pred_masks: torch.Tensor, inputs: dict):
        """Child class converts raw model masks to image space."""
        ...

    @abstractmethod
    def _normalize_masks(self, masks):
        """Child class normalizes masks to numpy arrays."""
        ...

    @abstractmethod
    def _normalize_scores(self, iou_scores):
        """Child class normalizes scores to numpy arrays."""
        ...

    @abstractmethod
    def _cache_image_embeddings(self):
        """Child class optionally computes and stores image embeddings."""
        ...

    @staticmethod
    def _to_pil(rgb_image):
        from PIL import Image

        if isinstance(rgb_image, Image.Image):
            return rgb_image.convert("RGB")
        if isinstance(rgb_image, np.ndarray):
            if rgb_image.dtype != np.uint8:
                rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)
            return Image.fromarray(rgb_image)
        raise TypeError(f"Unsupported image type: {type(rgb_image)}. Expected numpy RGB array or PIL.Image.")

    def _require_ready(self):
        if self.model is None or self.processor is None:
            raise RuntimeError("Transformers model/processor not loaded. Check your segmentor _load_model().")

    def _predict(self, *, input_points=None, input_labels=None, input_boxes=None, multimask_output=True):
        if self.image is None or self._raw_image_pil is None:
            raise ValueError("Image has not been set. Please call set_image() before predicting.")

        self._require_ready()

        inputs = self._prepare_inputs(input_points, input_labels, input_boxes, multimask_output)
        pred_masks, iou_scores = self._forward(inputs, multimask_output)
        masks_pp = self._post_process(pred_masks, inputs)
        masks_np = self._normalize_masks(masks_pp)
        scores_np = self._normalize_scores(iou_scores)
        return masks_np, scores_np

    def load_image(self, image_path=None, image=None):
        """
        Load an image from file path or array and store as RGB.

        args:
            image_path (str | None): Path to image file.
            image (np.ndarray | None): Image array.

        raises:
            FileNotFoundError: If image_path is provided but file cannot be read.
            ValueError: If neither image_path nor image is provided.
        """
        if image_path is not None:
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Error: Could not load image from {image_path}. Please check the file path and ensure the file exists.")
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image is not None:
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Either image_path or image must be provided.")

    def set_image(self):
        """
        Set the current image for the predictor.

        raises:
            ValueError: If no image is loaded.
        """
        if self.image is not None:
            self.predictor.set_image(self.image)
        else:
            raise ValueError("No image loaded. Please load an image first.")
        self._require_ready()
        self._raw_image_pil = self._to_pil(self.image)
        self._image_embeddings = None
        if self.cache_image_embeddings:
            self._image_embeddings = self._cache_image_embeddings()

    def predict_mask(self, input_point, input_label, multimask_output=True):
        """
        Predict masks given points and labels.

        args:
            input_point (np.ndarray): Array of points to guide segmentation.
            input_label (np.ndarray): Array of labels corresponding to points.
            multimask_output (bool): Whether to output multiple masks.

        returns:
            tuple: (masks, scores) predicted by the model.

        raises:
            ValueError: If image has not been set.
        if self.image is None:
            raise ValueError("Image has not been set. Please call set_image() before predicting.")
        """
        return self._predict(
            input_points=input_point,
            input_labels=input_label,
            input_boxes=None,
            multimask_output=multimask_output,
        )

    def predict_mask_box(self, input_box, multimask_output=True):
        """
        Predict masks given a bounding box.

        args:
            input_box (array-like): [x0, y0, x1, y1] bounding box coordinates.
            multimask_output (bool): Whether to output multiple masks.

        returns:
            tuple: (masks, scores) predicted by the model.

        raises:
            ValueError: If image has not been set.
        """
        return self._predict(
            input_points=None,
            input_labels=None,
            input_boxes=input_box,
            multimask_output=multimask_output,
        )

    def single_prediction(self, input_point, input_label, multimask_output=False):
        """
        Return a single mask prediction (first mask).

        args:
            input_point (np.ndarray): Array of points.
            input_label (np.ndarray): Array of labels.
            multimask_output (bool): Whether to output multiple masks.

        returns:
            np.ndarray: Single predicted mask.

        raises:
            ValueError: If image has not been set.
        """
        if self.image is None:
            raise ValueError("Image has not been set. Please call set_image() before predicting.")

        masks, scores = self.predict_mask(input_point, input_label, multimask_output)
        return masks[0]  # Return the first mask

    def show_image(self):
        """
        Display the loaded image using matplotlib.

        raises:
            ValueError: If no image is loaded.
        """
        if self.image is not None:
            plt.figure(figsize=(10, 10))
            plt.imshow(self.image)
            plt.axis('on')
            plt.show()
        else:
            raise ValueError("No image loaded. Please load an image first.")

    @staticmethod
    def show_mask(mask, ax, random_color=False, borders=True):
        """
        Display a mask on a matplotlib axis.

        args:
            mask (np.ndarray): Binary mask to display.
            ax (matplotlib.axes.Axes): Axis to draw on.
            random_color (bool): Whether to use random color for mask.
            borders (bool): Whether to draw borders around mask.
        """
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

        if borders:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)

        ax.imshow(mask_image)

    @staticmethod
    def show_circles(coords, labels, ax, marker_size=375):
        """
        Show points as colored circles.

        args:
            coords (np.ndarray): Coordinates of points.
            labels (np.ndarray): Labels for each point (0 or 1).
            ax (matplotlib.axes.Axes): Axis to draw on.
            marker_size (int): Size of markers.
        """
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size,
                   edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size,
                   edgecolor='white', linewidth=1.25)

    @staticmethod
    def show_box(box, ax):
        """
        Draw a bounding box on a matplotlib axis.

        args:
            box (array-like): [x0, y0, x1, y1]
            ax (matplotlib.axes.Axes): Axis to draw on.
        """
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

    def visualize_prediction(self, input_point=None, input_label=None, input_box=None, multimask_output=False, borders=True):
        """        
        Visualize segmentation on the loaded image.

        args:
            input_point (np.ndarray | None): Points for segmentation.
            input_label (np.ndarray | None): Labels for points.
            input_box (array-like | None): Bounding box for segmentation.
            multimask_output (bool): Whether to predict multiple masks.
            borders (bool): Whether to draw mask borders.

        raises:
            ValueError: If neither or both of input_point and input_box are provided.
        """
        if input_point is None and input_box is None:
            raise ValueError("Please provide either input_point or input_box.")
        if input_point is not None and input_box is not None:
            raise ValueError("Please provide only one of input_point or input_box.")

        if input_point is not None:
            masks, scores = self.predict_mask(input_point, input_label, multimask_output)
            for i, (mask, score) in enumerate(zip(masks, scores)):
                plt.figure(figsize=(10, 10))
                plt.imshow(self.image)
                self.show_mask(mask, plt.gca(), borders=borders)
                self.show_circles(input_point, input_label, plt.gca())
                plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
                plt.axis('off')
                plt.show()
        else:
            masks, scores = self.predict_mask_box(input_box, multimask_output)
            for i, (mask, score) in enumerate(zip(masks, scores)):
                plt.figure(figsize=(10, 10))
                plt.imshow(self.image)
                self.show_mask(mask, plt.gca(), borders=borders)
                self.show_box(input_box, plt.gca())
                plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
                plt.axis('off')
                plt.show()

    def segment(self, image=None, input_point=None, input_label=None, input_box=None, multimask_output=False):
        """
        Perform segmentation and return mask.

        args:
            image (np.ndarray | None): Optional image to load before segmenting.
            input_point (np.ndarray | None): Points for segmentation.
            input_label (np.ndarray | None): Labels for points.
            input_box (array-like | None): Bounding box for segmentation.
            multimask_output (bool): Whether to predict multiple masks.

        returns:
            np.ndarray | None: Mask array, or None if no mask found.

        raises:
            ValueError: If no image is loaded or neither/both of input_point and input_box are provided.
        """
        if image is not None:
            self.load_image(image=image)
            self.set_image()
        else:
            if self.image is None:
                raise ValueError("No image loaded. Please load an image first.")
        if input_point is None and input_box is None:
            raise ValueError("Please provide either input_point or input_box.")
        if input_point is not None and input_box is not None:
            raise ValueError("Please provide only one of input_point or input_box.")

        if input_point is not None:
            mask = self.single_prediction(input_point, input_label, multimask_output)
            if mask is None:
                logging.error("No mask found")
                return None
            else:
                return mask
        else:
            masks, scores = self.predict_mask_box(input_box, multimask_output)
            if masks is None:
                logging.error("No masks found")
                return None
            else:
                return masks[0]


# ================= MOBILE SAM SEGMENTOR (CellSegmentor1) =================
class CellSegmentor1(BaseSegmentor):
    """Segmentor powered by MobileSAM (Transformers SamModel-compatible checkpoint)."""

    def __init__(self, sam_checkpoint=None, model_type="vit_t", device=None, cache_image_embeddings: bool = True):
        """
        Initialize MobileSAM segmentor.

        args:
            sam_checkpoint (str | None): Path to MobileSAM checkpoint. Defaults to repo weight.
            model_type (str): Model type string (e.g., 'vit_t').
            device (str | None): Torch device. Auto-resolved if None.
        """
        # Keep signature for compatibility with your existing code.
        # sam_checkpoint now means "HF model id or local HF-exported folder".
        #
        # MobileSAM that is SamModel-compatible example:
        # - bhllx/mobilesam has "architectures": ["SamModel"] and "model_type": "sam"
        #   in config.json.
        self.model_id_or_path = sam_checkpoint or "bhllx/mobilesam"

        # model_type kept only so existing callers don't break; not used by Transformers loader.
        self.model_type = model_type

        super().__init__(device=device, cache_image_embeddings=cache_image_embeddings)

    def _load_model(self):
        from mobile_sam import sam_model_registry, SamPredictor
        self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)


# ================= SAM2 SEGMENTOR (CellSegmentor2) =================
class CellSegmentor2(BaseSegmentor):
    """Segmentor powered by SAM2 (v2.1 Hiera‑Tiny)."""

    def __init__(self, sam_checkpoint=None, model_cfg=None, device=None):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(current_dir, ".."))
        default_ckpt = os.path.join(repo_root, "deepLearning", "cellModel", "sam2", "checkpoints", "sam2.1_hiera_tiny.pt")
        default_cfg = os.path.join(repo_root, "deepLearning", "cellModel", "sam2", "sam2", "configs", "sam2.1", "sam2.1_hiera_t.yaml")
        self.sam_checkpoint = sam_checkpoint or default_ckpt
        self.model_cfg = model_cfg or default_cfg
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        super().__init__(device)

    def _enable_cuda_tricks(self):
        if self.device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

    def _load_model(self):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self._enable_cuda_tricks()
        try:
            with open(self.model_cfg, "r") as f:
                yaml.safe_load(f)
            self.sam = build_sam2(self.model_cfg, self.sam_checkpoint, device=self.device)
        except Exception as exc:
            logging.error(f"Error loading SAM2 model: {exc}")
            raise
        self.predictor = SAM2ImagePredictor(self.sam)

import os
import numpy as np
import onnxruntime as ort
import cv2
import matplotlib.pyplot as plt
import torch
from typing import Optional

class CellSegmentor3(BaseSegmentor):
    """ONNX-based SAM2 segmentor, refactored to match BaseSegmentor interface and sam-cpp-macos reference."""
    def __init__(self, encoder_path: Optional[str] = None, decoder_path: Optional[str] = None, device: Optional[str] = None):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(current_dir, ".."))
        default_dir = os.path.join(repo_root, "deepLearning", "cellModel")

        if model_type == "sam2_video":
            self.model = Sam2VideoModel.from_pretrained(self.model_id_or_path).to(self.device)
            try:
                self.processor = Sam2VideoProcessor.from_pretrained(self.model_id_or_path, use_fast=True)
            except OSError:
                self.processor = Sam2VideoProcessor.from_pretrained("facebook/sam2.1-hiera-tiny", use_fast=True)
        else:
            raise ValueError("Either image_path or image must be provided.")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]
        self._orig_shape = (orig_h, orig_w)
        scale = min(1024 / orig_h, 1024 / orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)
        resized = cv2.resize(img_rgb, (new_w, new_h))
        canvas = np.zeros((1024, 1024, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        norm_img = (canvas.astype(np.float32) / 255.0 - mean) / std
        self.image = norm_img
        self.input_size = (1024, 1024)
        self._pad_shape = (new_h, new_w)
        self._pad_offset = (0, 0)
        # Debug info
        # print(f"[DEBUG] Original image shape: {orig_h}x{orig_w}")
        # print(f"[DEBUG] Resized image shape: {new_h}x{new_w}")
        # print(f"[DEBUG] Padding offset: {self._pad_offset}")

    def set_image(self):
        if self.image is None:
            raise ValueError("No image loaded. Please load an image first.")
        inp = self.image.transpose(2, 0, 1)[None].astype(np.float32)
        result = self.enc_session.run(None, {self.enc_input_name: inp})
        self.image_embeddings = result[0]
        self.high_res_features1 = result[1]
        self.high_res_features2 = result[2]
        # Debug info
        # print(f"[DEBUG] Encoder output shapes: {[x.shape for x in result]}")

    def _cache_image_embeddings(self):
        if hasattr(self.model, "get_image_embeddings"):
            inputs = self.processor(images=self._raw_image_pil, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                return self.model.get_image_embeddings(inputs["pixel_values"])
        return None


# ================= SAM2 SEGMENTOR (CellSegmentor2) =================
class CellSegmentor2(BaseSegmentor):
    """Segmentor powered by SAM2 via Transformers.

    Note: Meta's public SAM2 checkpoints on the Hugging Face Hub are `sam2_video` models. For single-image
    segmentation we run `Sam2VideoModel._single_frame_forward(...)` under the hood (same prompts / outputs),
    which matches the behavior of the upstream SAM2 repo for interactive image segmentation.
    """

    def __init__(self, sam_checkpoint=None, model_cfg=None, device=None, cache_image_embeddings: bool = True):
        """
        Initialize SAM2 segmentor.

        args:
            sam_checkpoint (str | None): Path to SAM2 checkpoint. Defaults to repo weight.
            model_cfg (str | None): Path to YAML config. Defaults to repo config.
            device (str | None): Torch device. Auto-resolved if None.
        """
        # Keep signature for compatibility.
        # sam_checkpoint now means "HF model id or local HF-exported folder".
        # The official Facebook/Meta checkpoints are `sam2_video`, but they support single-image interactive
        # segmentation as well (we call the model's single-frame forward).
        self.model_id_or_path = sam_checkpoint or "facebook/sam2.1-hiera-tiny"

        # model_cfg kept only so existing callers don't break; not used by Transformers loader.
        self.model_cfg = model_cfg

        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        super().__init__(device=device, cache_image_embeddings=cache_image_embeddings)

    def _load_model(self):
        """
        Load the SAM2 model and predictor from checkpoint and config.

        raises:
            Exception: If loading model or config fails.
        """
        from transformers import AutoConfig, Sam2Model, Sam2Processor, Sam2VideoModel, Sam2VideoProcessor

        cfg = AutoConfig.from_pretrained(self.model_id_or_path)
        model_type = getattr(cfg, "model_type", None)

        if model_type == "sam2_video":
            self.model = Sam2VideoModel.from_pretrained(self.model_id_or_path).to(self.device)
            try:
                self.processor = Sam2VideoProcessor.from_pretrained(self.model_id_or_path, use_fast=True)
            except OSError:
                self.processor = Sam2VideoProcessor.from_pretrained("facebook/sam2.1-hiera-tiny", use_fast=True)
        else:
            raise ValueError("No image loaded. Please load an image first.")

    @staticmethod
    def _wrap_box(input_box):
        box = np.asarray(input_box)
        if box.shape != (4,):
            raise ValueError(f"input_box must be shape (4,) as [x1,y1,x2,y2]. Got {box.shape}.")
        return [[box.astype(float).tolist()]]

    def _prepare_inputs(self, input_point, input_label, input_box, multimask_output):
        proc_kwargs = {"images": self._raw_image_pil, "return_tensors": "pt"}
        if input_point is not None and input_label is not None:
            pts, lbls = self._wrap_points_labels(input_point, input_label)
            proc_kwargs["input_points"] = pts
            proc_kwargs["input_labels"] = lbls
        if input_box is not None:
            proc_kwargs["input_boxes"] = self._wrap_box(input_box)

        inputs = self.processor(**proc_kwargs).to(self.device)

        if self._image_embeddings is not None:
            inputs.pop("pixel_values", None)
            inputs["image_embeddings"] = self._image_embeddings

        return inputs

    def _forward(self, inputs, multimask_output):
        with torch.inference_mode():
            if hasattr(self.model, "_single_frame_forward"):
                outputs = self.model._single_frame_forward(**inputs, multimask_output=multimask_output)
            else:
                outputs = self.model(**inputs, multimask_output=multimask_output)
        return outputs.pred_masks, outputs.iou_scores

    def _post_process(self, pred_masks: torch.Tensor, inputs: dict):
        original_sizes = inputs.get("original_sizes", None)

        if isinstance(original_sizes, torch.Tensor):
            original_sizes = original_sizes.detach().cpu()

        pred_masks = pred_masks.detach().cpu()
        # The SAM2 image post-processor expects a 5D tensor: (B, point_batch, num_masks, H, W).
        # Sam2VideoModel single-frame output is 4D (B, point_batch, H, W), so add the singleton mask dim.
        if pred_masks.ndim == 4:
            pred_masks = pred_masks.unsqueeze(2)

        if hasattr(self.processor, "post_process_masks"):
            # Sam2Processor.post_process_masks signature is (masks, original_sizes, mask_threshold=...),
            # so passing reshaped_input_sizes positionally will break.
            return self.processor.post_process_masks(pred_masks, original_sizes)

        return self.processor.image_processor.post_process_masks(pred_masks, original_sizes, reshaped_input_sizes)

    @staticmethod
    def _normalize_masks(masks):
        if isinstance(masks, (list, tuple)):
            masks = masks[0]
        if not isinstance(masks, torch.Tensor):
            masks = torch.as_tensor(masks)
        while masks.ndim > 3:
            masks = masks[0]
        masks = masks > 0
        return masks.detach().cpu().numpy()

    @staticmethod
    def _normalize_scores(iou_scores):
        if not isinstance(iou_scores, torch.Tensor):
            iou_scores = torch.as_tensor(iou_scores)
        scores = iou_scores.detach().cpu()
        if scores.ndim == 3:
            scores = scores[0, 0]
        elif scores.ndim == 2:
            scores = scores[0]
        return scores.numpy()

    def _cache_image_embeddings(self):
        if hasattr(self.model, "get_image_embeddings"):
            inputs = self.processor(images=self._raw_image_pil, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                return self.model.get_image_embeddings(inputs["pixel_values"])
        return None


# ================= SAM2 SEGMENTOR (CellSegmentor2) =================
class CellSegmentor2(BaseSegmentor):
    """Segmentor powered by SAM2 via Transformers.

    Note: Meta's public SAM2 checkpoints on the Hugging Face Hub are `sam2_video` models. For single-image
    segmentation we run `Sam2VideoModel._single_frame_forward(...)` under the hood (same prompts / outputs),
    which matches the behavior of the upstream SAM2 repo for interactive image segmentation.
    """

    def __init__(self, sam_checkpoint=None, model_cfg=None, device=None, cache_image_embeddings: bool = True):
        # Keep signature for compatibility.
        # sam_checkpoint now means "HF model id or local HF-exported folder".
        # The official Facebook/Meta checkpoints are `sam2_video`, but they support single-image interactive
        # segmentation as well (we call the model's single-frame forward).
        self.model_id_or_path = sam_checkpoint or "facebook/sam2.1-hiera-tiny"

        # model_cfg kept only so existing callers don't break; not used by Transformers loader.
        self.model_cfg = model_cfg
        self._image_original_sizes = None

        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        super().__init__(device=device, cache_image_embeddings=cache_image_embeddings)

    def set_image(self):
        # Sizes and embeddings belong to the same image, even when a camera reuses its buffer.
        self._image_original_sizes = None
        return super().set_image()

    def _load_model(self):
        from transformers import AutoConfig, Sam2Model, Sam2Processor, Sam2VideoModel, Sam2VideoProcessor

        cfg = AutoConfig.from_pretrained(self.model_id_or_path)
        model_type = getattr(cfg, "model_type", None)

        if model_type == "sam2_video":
            self.model = Sam2VideoModel.from_pretrained(self.model_id_or_path).to(self.device)
            try:
                self.processor = Sam2VideoProcessor.from_pretrained(self.model_id_or_path, use_fast=True)
            except OSError:
                self.processor = Sam2VideoProcessor.from_pretrained("facebook/sam2.1-hiera-tiny", use_fast=True)
        else:
            self.model = Sam2Model.from_pretrained(self.model_id_or_path).to(self.device)
            try:
                self.processor = Sam2Processor.from_pretrained(self.model_id_or_path, use_fast=True)
            except OSError:
                # Some exported checkpoints ship weights/config only (no processor files). Fall back to a
                # compatible processor from the official checkpoint hub.
                self.processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-tiny", use_fast=True)

        self.model.eval()

    @staticmethod
    def _wrap_points_labels(input_point, input_label):
        pts = np.asarray(input_point)
        lbl = np.asarray(input_label)

        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"input_point must be shape (N, 2). Got {pts.shape}.")
        if lbl.ndim != 1 or lbl.shape[0] != pts.shape[0]:
            raise ValueError(f"input_label must be shape (N,), matching input_point. Got {lbl.shape} vs {pts.shape}.")

        input_points = [[pts.tolist()]]
        input_labels = [[lbl.astype(int).tolist()]]
        return input_points, input_labels

    @staticmethod
    def _wrap_box(input_box):
        box = np.asarray(input_box)
        if box.shape != (4,):
            raise ValueError(f"input_box must be shape (4,) as [x1,y1,x2,y2]. Got {box.shape}.")
        return [[box.astype(float).tolist()]]

    def _prepare_inputs(self, input_point, input_label, input_box, multimask_output):
        proc_kwargs = {"return_tensors": "pt"}
        if self._image_embeddings is not None and self._image_original_sizes is not None:
            proc_kwargs["original_sizes"] = self._image_original_sizes
        else:
            proc_kwargs["images"] = self._raw_image_pil
        if input_point is not None and input_label is not None:
            pts, lbls = self._wrap_points_labels(input_point, input_label)
            proc_kwargs["input_points"] = pts
            proc_kwargs["input_labels"] = lbls
        if input_box is not None:
            proc_kwargs["input_boxes"] = self._wrap_box(input_box)

        inputs = self.processor(**proc_kwargs).to(self.device)

        if self._image_embeddings is not None:
            inputs.pop("pixel_values", None)
            inputs["image_embeddings"] = self._image_embeddings

        return inputs

    def _forward(self, inputs, multimask_output):
        with torch.inference_mode():
            if hasattr(self.model, "_single_frame_forward"):
                outputs = self.model._single_frame_forward(**inputs, multimask_output=multimask_output)
            else:
                outputs = self.model(**inputs, multimask_output=multimask_output)
        return outputs.pred_masks, outputs.iou_scores

    def _post_process(self, pred_masks: torch.Tensor, inputs: dict):
        original_sizes = inputs.get("original_sizes", None)

        if isinstance(original_sizes, torch.Tensor):
            original_sizes = original_sizes.detach().cpu()

        pred_masks = pred_masks.detach().cpu()
        # The SAM2 image post-processor expects a 5D tensor: (B, point_batch, num_masks, H, W).
        # Sam2VideoModel single-frame output is 4D (B, point_batch, H, W), so add the singleton mask dim.
        if pred_masks.ndim == 4:
            pred_masks = pred_masks.unsqueeze(2)

        if hasattr(self.processor, "post_process_masks"):
            # Sam2Processor.post_process_masks signature is (masks, original_sizes, mask_threshold=...),
            # so passing reshaped_input_sizes positionally will break.
            return self.processor.post_process_masks(pred_masks, original_sizes)

        return self.processor.image_processor.post_process_masks(pred_masks, original_sizes)

    @staticmethod
    def _normalize_masks(masks):
        if isinstance(masks, (list, tuple)):
            masks = masks[0]
        if not isinstance(masks, torch.Tensor):
            masks = torch.as_tensor(masks)
        while masks.ndim > 3:
            masks = masks[0]
        masks = masks > 0
        return masks.detach().cpu().numpy()

    @staticmethod
    def _normalize_scores(iou_scores):
        if not isinstance(iou_scores, torch.Tensor):
            iou_scores = torch.as_tensor(iou_scores)
        scores = iou_scores.detach().cpu()
        if scores.ndim == 3:
            scores = scores[0, 0]
        elif scores.ndim == 2:
            scores = scores[0]
        return scores.numpy()

    def _cache_image_embeddings(self):
        if hasattr(self.model, "get_image_embeddings"):
            inputs = self.processor(images=self._raw_image_pil, return_tensors="pt")
            original_sizes = inputs.get("original_sizes")
            if isinstance(original_sizes, torch.Tensor):
                original_sizes = original_sizes.detach().cpu().clone()
            inputs = inputs.to(self.device)
            with torch.inference_mode():
                embeddings = self.model.get_image_embeddings(inputs["pixel_values"])
            self._image_original_sizes = original_sizes
            return embeddings
        return None



import sys, time
import numpy as np
import cv2
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, 
    QHBoxLayout, QPushButton, QFileDialog  # <-- NEW IMPORTS
)
from PyQt5.QtGui import QPixmap, QImage

class CameraWorker(QThread):
    """Thread that simulates camera feed and emits segmented frames."""
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, fps=30, parent=None):
        """
        Initialize the camera worker.

        args:
            fps (int): Frames per second.
            parent (QObject | None): Parent Qt object.
        """
        super().__init__(parent)
        self.fps = fps
        self.keep_running = True
        # self.image_path = r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\patcherbot\deepLearning\cellModel\example pictures\before.tiff"
        # self.image_path = r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\patcherbot\deepLearning\cellModel\example pictures\square.png"
        self.image_path = r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\patcherbot\devices\camera\FakeMicroscopeImgs\cellsegtest.png"
        bgr = cv2.imread(self.image_path)
        if bgr is None:
            raise FileNotFoundError(f"Could not load {self.image_path}")
        self.rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        self.points = []
        self.labels = []
        self.segmentor = CellSegmentor2()

    def run(self):
        """
        Main loop of thread: apply segmentation and emit frame.
        """
        while self.keep_running:
            display_img = self.rgb.copy()
            if self.points:
                mask = self.segmentor.segment(
                    image=display_img,
                    input_point=np.array(self.points),
                    input_label=np.array(self.labels),
                    multimask_output=False
                )
                if mask is not None:
                    mask_bool = mask.astype(bool)
                    display_img[mask_bool] = [255, 0, 0]
                    for (x, y) in self.points:
                        cv2.circle(display_img, (x, y), 5, (0, 255, 0), -1)

            self.frame_ready.emit(display_img)
            time.sleep(1 / self.fps)

    def stop(self):
        """
        Stop the camera thread safely.
        """
        self.keep_running = False
        self.quit()
        self.wait()

    def add_point(self, x, y):
        """
        Add a segmentation point.

        args:
            x (int): X coordinate.
            y (int): Y coordinate.
        """
        self.points.append([x, y])
        self.labels.append(1)

    def clear_points(self):
        """
        Clear all current points.
        """
        self.points.clear()
        self.labels.clear()

class SamCAM(QMainWindow):
    """GUI application for point-based segmentation with live preview."""
    def __init__(self):
        """
        Initialize GUI and camera worker.
        """
        super().__init__()
        self.setWindowTitle("SamCAM")
        self.image_label = QLabel(alignment=Qt.AlignCenter)
        
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)

        # ----------------------------------------------------
        # 1) Create a bottom-right button layout and button
        button_layout = QHBoxLayout()
        button_layout.addStretch()  # pushes the button to the right
        self.save_button = QPushButton("Save segmentation?")
        button_layout.addWidget(self.save_button)
        layout.addLayout(button_layout)
        # 2) Connect button to a method that saves the image
        self.save_button.clicked.connect(self.save_segmentation)
        # ----------------------------------------------------
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Create worker thread for “camera”
        self.worker = CameraWorker(fps=5)
        self.worker.frame_ready.connect(self.update_image)
        self.worker.start()

        # Will store the latest displayed frame here
        self.latest_frame = None

    def update_image(self, rgb):
        """
        Receive new frame and display it.

        args:
            rgb (np.ndarray): RGB frame from CameraWorker.
        """
        # Store the latest image so we can save it
        self.latest_frame = rgb.copy()
        
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg))

    def mousePressEvent(self, event):
        """
        Handle mouse clicks to add or clear points.

        args:
            event (QMouseEvent): Qt mouse event.
        """
        if event.button() == Qt.LeftButton:
            label_pos = self.image_label.mapFromParent(event.pos())
            pm = self.image_label.pixmap()
            if pm:
                scale_w = pm.width() / self.image_label.width()
                scale_h = pm.height() / self.image_label.height()
                x_img = int(label_pos.x() * scale_w)
                y_img = int(label_pos.y() * scale_h)
                self.worker.add_point(x_img, y_img)
        elif event.button() == Qt.RightButton:
            self.worker.clear_points()

    def closeEvent(self, event):
        """
        Handle window close event to stop camera thread.

        args:
            event (QCloseEvent): Qt close event.
        """
        self.worker.stop()
        super().closeEvent(event)

    # ----------------------------------------------------
    def save_segmentation(self):
        """
        Re-run segmentation on original image and save only the cut-out region (RGBA).      

        raises:
            FileNotFoundError: If image cannot be loaded.
        """
        if not self.worker.points:
            print("No points set, nothing to segment.")
            return

        # Re-run segmentation on the original
        rgb_copy = self.worker.rgb.copy()
        mask = self.worker.segmentor.segment(
            image=rgb_copy,
            input_point=np.array(self.worker.points),
            input_label=np.array(self.worker.labels),
            multimask_output=False
        )
        if mask is None:
            print("Segmentation returned None, nothing to save.")
            return

        # Convert mask to boolean
        mask_bool = (mask > 0)

        # Build an RGBA image (4 channels)
        h, w, _ = rgb_copy.shape
        segmented_rgba = np.zeros((h, w, 4), dtype=np.uint8)

        # Copy over the original RGB where the mask is True
        segmented_rgba[mask_bool, 0:3] = rgb_copy[mask_bool]

        # Set alpha channel to 255 where the mask is True
        segmented_rgba[mask_bool, 3] = 255

        # Let user pick the save path
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Segmented PNG", "", 
            "PNG Files (*.png);;All Files (*)"
        )
        if not file_path:
            return  # user canceled

        # Finally save as PNG (4-channels) - the alpha channel is preserved
        cv2.imwrite(file_path, segmented_rgba)
        print(f"Saved RGBA cut-out to: {file_path}")

    # ----------------------------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = SamCAM()
    w.show()
    sys.exit(app.exec_())

    # #### test segmentor 
    # # obj_seg = CellSegmentor()
    # obj_seg = CellSegmentor2()

    # # Load an example image
    # # # image_path = r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\patcherbot\deepLearning\cellModel\sam2\notebooks\images\truck.jpg"
    # # # image_path = r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\experiments\Data\rig_recorder_data\2024_12_10-15_17\camera_frames\178605_1733864203.246073.webp" # pipette

    # # image_path = r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\patcherbot\deepLearning\cellModel\example pictures\before.tiff" # cell
    # image_path = r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\patcherbot\devices\camera\FakeMicroscopeImgs\background.tif"

     
    # # load and set image
    # # obj_seg.load_image(r"C:\Users\sa-forest\Documents\GitHub\patchability\code\segment-anything\notebooks\images\truck.jpg")
    # obj_seg.load_image(image_path)
    # obj_seg.set_image()
    # # obj_seg.show_image()

    # # # Predict using a single point
    # # input_point = np.array([[500, 375]]) # cell or truck
    # # # input_point = np.array([[400, 900]]) # pipette
  
    # input_point = np.array([600, 400])

    # input_label = np.array([1])
    # obj_seg.visualize_prediction(input_point=input_point, input_label=input_label)

    # # # # # Predict using a bounding box
    # # input_box = np.array([425, 600, 700, 875])
    # # obj_seg.visualize_prediction(input_box=input_box)

    # # perform a  single prediction
    # #obj_seg.single_prediction(input_point, input_label)

    # # # test segment speed
    # # start_time = time.time()

    # # image = cv2.imread(image_path)
    # # total_time = 0
    # # num  = 1000
    # # time_array = np.zeros(num)
    # # for i in range(num):
    # #     start_time = time.time()
    # #     mask = obj_seg.segment(image=image, input_point=input_point, input_label=input_label)
    # #     elapsed_time = time.time() - start_time
    # #     elapsed_time_ms = elapsed_time * 1000
    # #     # total_time += elapsed_time
    # #     time_array[i] = elapsed_time_ms
    
    # # time_array = time_array[3:]
    # # average_time = np.mean(time_array)
    # # print(f'Average segment time: {average_time:.4f}  milliseconds')
    # # # remove first few points

    # # # plot time_array
    # # plt.plot(time_array)
    # # plt.xlabel('Iteration')
    # # plt.ylabel('Time (ms)')
    # # plt.title('Segment Time')
    # # plt.show()
