import os
import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from torch import nn
from torch.nn import functional as F

# ================= BASE SEGMENTOR =================
class BaseSegmentor(ABC):
    """Shared image, device and target utilities; SAM prediction hooks are optional."""

    def __init__(self, device: str | None = None, cache_image_embeddings: bool = True):
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
        if explicit is not None:
            return torch.device(explicit)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @abstractmethod
    def _load_model(self):
        ...

    def _prepare_inputs(self, input_point, input_label, input_box, multimask_output):
        """Prepare inputs for backends that use the shared SAM prediction path."""
        raise NotImplementedError("This backend does not implement SAM prompt prediction")

    def _forward(self, inputs, multimask_output):
        """Child class runs the model and returns (pred_masks, scores)."""
        raise NotImplementedError("This backend does not implement SAM prompt prediction")

    def _post_process(self, pred_masks: torch.Tensor, inputs: dict):
        """Child class converts raw model masks to image space."""
        raise NotImplementedError("This backend does not implement SAM mask postprocessing")

    def _normalize_masks(self, masks):
        """Child class normalizes masks to numpy arrays."""
        raise NotImplementedError("This backend does not implement SAM mask normalization")

    def _normalize_scores(self, iou_scores):
        """Child class normalizes scores to numpy arrays."""
        raise NotImplementedError("This backend does not implement SAM score normalization")

    def _cache_image_embeddings(self):
        """Child class optionally computes and stores image embeddings."""
        return None

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

    @staticmethod
    def _ensure_bgr(image: np.ndarray) -> np.ndarray:
        """Ensure *image* is a BGR array (H, W, 3)."""
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 1:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 3:
            return image
        raise ValueError(f"Unsupported image shape for BGR conversion: {image.shape}")

    @staticmethod
    def _prepare_for_segmentation(image: np.ndarray) -> np.ndarray:
        """Return a uint8 BGR image suitable for SAM2."""
        bgr = BaseSegmentor._ensure_bgr(image)
        if bgr.dtype == np.uint16:
            bgr = (bgr / 257.0).astype(np.uint8)
        elif bgr.dtype != np.uint8:
            arr = bgr.astype(np.float32)
            if arr.max() > 1.0:
                arr /= arr.max()
            bgr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return bgr

    def load_image(self, image_path=None, image=None):
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
        if self.image is None:
            raise ValueError("No image loaded. Please load an image first.")
        self._require_ready()
        self._raw_image_pil = self._to_pil(self.image)
        self._image_embeddings = None
        if self.cache_image_embeddings:
            self._image_embeddings = self._cache_image_embeddings()

    def predict_mask(self, input_point, input_label, multimask_output=True):
        return self._predict(
            input_points=input_point,
            input_labels=input_label,
            input_boxes=None,
            multimask_output=multimask_output,
        )

    def predict_mask_box(self, input_box, multimask_output=True):
        return self._predict(
            input_points=None,
            input_labels=None,
            input_boxes=input_box,
            multimask_output=multimask_output,
        )

    def single_prediction(self, input_point, input_label, multimask_output=False):
        if self.image is None:
            raise ValueError("Image has not been set. Please call set_image() before predicting.")

        masks, scores = self.predict_mask(input_point, input_label, multimask_output)
        return masks[0]  # Return the first mask

    @staticmethod
    def get_mask_target(mask):
        """Return the mask, up to three maximum-clearance interior points, and clearance.

        Positive pixels are foreground. Image borders count as background.
        Empty masks return an empty list. Coordinates are (x, y); tied points are spread apart spatially.
        """
        result = dict(mask=mask, target_points_xy=[], edge_clearance_px=0.0)
        if mask is None:
            return result
        binary = np.asarray(mask)
        while binary.ndim > 2 and binary.shape[0] == 1:
            binary = binary[0]
        if binary.ndim != 2 or not np.isfinite(binary).all():
            raise ValueError("Expected one finite two-dimensional cell mask")
        binary = (binary > 0).astype(np.uint8)
        if not np.any(binary):
            return result
        padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        distances = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        clearance = distances.max()
        result["target_points_xy"] = [
            (int(x) - 1, int(y) - 1)
            for y, x in np.argwhere(distances == clearance)
        ]
        candidates = result["target_points_xy"]
        if len(candidates) > 3:
            selected = [candidates[0]]
            for _ in range(2):
                selected.append(max(candidates, key=lambda p: min(
                    (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 for q in selected)))
            result["target_points_xy"] = selected
        result["edge_clearance_px"] = float(clearance)
        return result

    def show_image(self):
        if self.image is not None:
            plt.figure(figsize=(10, 10))
            plt.imshow(self.image)
            plt.axis('on')
            plt.show()
        else:
            raise ValueError("No image loaded. Please load an image first.")

    @staticmethod
    def show_mask(mask, ax, random_color=False, borders=True):
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
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size,
                   edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size,
                   edgecolor='white', linewidth=1.25)

    @staticmethod
    def show_box(box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

    def visualize_prediction(self, input_point=None, input_label=None, input_box=None, multimask_output=False, borders=True):
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
        from transformers import SamModel, SamProcessor

        self.model = SamModel.from_pretrained(self.model_id_or_path).to(self.device)
        self.processor = SamProcessor.from_pretrained(self.model_id_or_path)
        self.model.eval()

    @staticmethod
    def _wrap_points_labels(input_point, input_label):
        pts = np.asarray(input_point)
        lbl = np.asarray(input_label)

        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"input_point must be shape (N, 2). Got {pts.shape}.")
        if lbl.ndim != 1 or lbl.shape[0] != pts.shape[0]:
            raise ValueError(f"input_label must be shape (N,), matching input_point. Got {lbl.shape} vs {pts.shape}.")

        # Nested list convention expected by HF SAM/SAM2: [batch][object][points][xy]
        input_points = [[pts.tolist()]]
        input_labels = [[lbl.astype(int).tolist()]]
        return input_points, input_labels

    @staticmethod
    def _wrap_box(input_box):
        box = np.asarray(input_box)
        if box.shape != (4,):
            raise ValueError(f"input_box must be shape (4,) as [x1,y1,x2,y2]. Got {box.shape}.")
        return [[box.astype(float).tolist()]]  # [batch][box][coords]

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
            outputs = self.model(**inputs, multimask_output=multimask_output)
        return outputs.pred_masks, outputs.iou_scores

    def _post_process(self, pred_masks: torch.Tensor, inputs: dict):
        original_sizes = inputs.get("original_sizes", None)
        reshaped_input_sizes = inputs.get("reshaped_input_sizes", None)

        if isinstance(original_sizes, torch.Tensor):
            original_sizes = original_sizes.detach().cpu()
        if isinstance(reshaped_input_sizes, torch.Tensor):
            reshaped_input_sizes = reshaped_input_sizes.detach().cpu()

        pred_masks = pred_masks.detach().cpu()

        if hasattr(self.processor, "post_process_masks"):
            try:
                if reshaped_input_sizes is not None:
                    return self.processor.post_process_masks(pred_masks, original_sizes, reshaped_input_sizes)
                return self.processor.post_process_masks(pred_masks, original_sizes)
            except TypeError:
                return self.processor.post_process_masks(pred_masks, original_sizes, reshaped_input_sizes)

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



def _resize(value, size):
    return F.interpolate(value, size=size, mode='bilinear', align_corners=False)


class TinyBoxUNet(nn.Module):
    """99,981-parameter crop U-Net; frozen detector pixels are explicit inputs.

    Input channels are grayscale intensity, the predicted-box prompt and the
    frozen PIDNet foreground probability. The output is one mask-logit plane;
    SAM2 targets never enter this forward path.
    """
    def __init__(self):
        super().__init__()
        def block(in_channels, channels):
            return nn.Sequential(nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
                nn.GroupNorm(2, channels), nn.SiLU(),
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.GroupNorm(2, channels), nn.SiLU())
        self.encoders = nn.ModuleList([block(3,8), block(8,16), block(16,32), block(32,50)])
        self.decoders = nn.ModuleList([block(82,32), block(48,16), block(24,8)])
        self.mask = nn.Conv2d(8,1,1)
        self.parameter_count = sum(value.numel() for value in self.parameters())

    def forward(self, inputs):
        if inputs.ndim != 4 or inputs.shape[1] != 3 or min(inputs.shape[-2:]) < 8:
            raise ValueError('Refinement requires Bx3xHxW grayscale/box/foreground crops')
        skips = []
        value = inputs
        for index, encoder in enumerate(self.encoders):
            value = encoder(value if index == 0 else F.max_pool2d(value,2))
            skips.append(value)
        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            value = decoder(torch.cat((_resize(value,skip.shape[-2:]),skip),dim=1))
        return self.mask(value)


def prepare_refinement_tensors(image_bgr, boxes_xyxy, foreground_logits, geometry, *, device='cuda',
                               crop_size=64, context=1.25, image_normalization='none'):
    """Keep the native foreground cascade and sample all ROI inputs on one device.

    Geometry uses the CPU reference's float64 floor/ceil equations. Bilinear ROI
    coordinates clamp to each integer crop before sampling; prompt sampling uses
    OpenCV INTER_NEAREST's floor indices, not bilinear coordinates.
    """
    import cv2
    import numpy as np
    image=np.asarray(image_bgr);boxes=np.asarray(boxes_xyxy,np.float64).reshape(-1,4)
    if image.dtype!=np.uint8 or image.ndim not in (2,3) or (image.ndim==3 and image.shape[2]!=3):
        raise ValueError('Expected native uint8 grayscale or BGR image')
    height,width=image.shape[:2]
    if min(height,width)<1 or not np.isfinite(boxes).all() or np.any(boxes[:,2:]<=boxes[:,:2]):
        raise ValueError('Expected nonempty image and finite positive-area XYXY boxes')
    if type(crop_size) is not int or crop_size<8 or not np.isfinite(context) or context<1:
        raise ValueError('Positive crop size and context >=1 required')
    if image_normalization not in ('none','local_zscore_clip3'):raise ValueError('Unsupported image normalization')
    if (int(geometry['height']),int(geometry['width']))!=(height,width):raise ValueError('Detector/native geometry differs')
    value=torch.as_tensor(foreground_logits,device=device).detach().float().reshape(1,1,*foreground_logits.shape[-2:])
    value=F.interpolate(value.sigmoid(),size=(640,640),mode='bilinear',align_corners=False)
    left,top=int(geometry['left']),int(geometry['top'])
    rw,rh=round(width*geometry['scale_x']),round(height*geometry['scale_y'])
    foreground=F.interpolate(value[:,:,top:top+rh,left:left+rw],size=(height,width),mode='bilinear',align_corners=False)[0,0]
    centers=(boxes[:,:2]+boxes[:,2:])*.5;half=(boxes[:,2:]-boxes[:,:2])*context*.5
    lower=np.maximum(0,np.minimum([width,height],np.floor(centers-half))).astype(np.int64)
    upper=np.maximum(0,np.minimum([width,height],np.ceil(centers+half))).astype(np.int64)
    crop_edges=np.concatenate((lower,upper),axis=1);ready=np.all(upper>lower,axis=1)
    geometries=[dict(native_hw=[height,width],crop_xyxy=edge.tolist(),box_xyxy_native=box.tolist(),
        status='ready' if permitted else 'outside_image',crop_size=crop_size,context=float(context))
        for edge,box,permitted in zip(crop_edges,boxes,ready)]
    if not len(boxes):return dict(inputs=torch.empty((0,3,crop_size,crop_size),device=device),geometries=geometries,foreground=foreground)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    gray=torch.from_numpy(np.ascontiguousarray(gray)).to(device=device,non_blocking=True).float()/255.
    packed=np.concatenate((crop_edges,boxes,ready[:,None]),axis=1)
    packed=torch.from_numpy(packed).to(device=device,non_blocking=True)
    edges=packed[:,:4].float()
    sizes=(edges[:,2:]-edges[:,:2]).clamp_min(1)
    offsets=torch.arange(crop_size,device=device,dtype=torch.float32)
    local_x=((offsets[None,:]+.5)*sizes[:,0:1]/crop_size-.5).clamp_min(0)
    local_y=((offsets[None,:]+.5)*sizes[:,1:2]/crop_size-.5).clamp_min(0)
    xs=edges[:,0:1]+torch.minimum(local_x,sizes[:,0:1]-1)
    ys=edges[:,1:2]+torch.minimum(local_y,sizes[:,1:2]-1)
    gx=((2*xs+1)/width-1)[:,None,:].expand(-1,crop_size,-1)
    gy=((2*ys+1)/height-1)[:,:,None].expand(-1,-1,crop_size)
    grid=torch.stack((gx,gy),dim=-1).reshape(1,len(boxes)*crop_size,crop_size,2)
    sampled=F.grid_sample(torch.stack((gray,foreground))[None],grid,mode='bilinear',padding_mode='border',align_corners=False)
    sampled=sampled[0].reshape(2,len(boxes),crop_size,crop_size).permute(1,0,2,3)
    raw=packed[:,4:8]
    nearest_x=edges[:,0:1]+torch.floor(offsets[None,:]*sizes[:,0:1]/crop_size)+.5
    nearest_y=edges[:,1:2]+torch.floor(offsets[None,:]*sizes[:,1:2]/crop_size)+.5
    prompt=((nearest_x[:,None,:]>=raw[:,0,None,None])&(nearest_x[:,None,:]<raw[:,2,None,None])&
            (nearest_y[:,:,None]>=raw[:,1,None,None])&(nearest_y[:,:,None]<raw[:,3,None,None])).float()
    inputs=torch.stack((sampled[:,0],prompt,sampled[:,1]),dim=1)
    if image_normalization=='local_zscore_clip3':
        plane=inputs[:,0];mean=plane.mean((-2,-1),keepdim=True);std=plane.std((-2,-1),unbiased=False,keepdim=True).clamp_min(1/255)
        inputs[:,0]=((plane-mean)/std).clamp(-3,3)/6+.5
    inputs*=packed[:,8].float()[:,None,None,None]
    return dict(inputs=inputs,geometries=geometries,foreground=foreground)


def _reconstruct_refinement_tensors(probabilities, foreground, geometries):
    """Shared native reconstruction for eager and captured unchanged-head paths."""
    union=torch.zeros_like(foreground)
    for value,geometry in zip(probabilities,geometries):
        if geometry['status']!='ready':continue
        x1,y1,x2,y2=geometry['crop_xyxy']
        restored=F.interpolate(value[None],size=(y2-y1,x2-x1),mode='bilinear',align_corners=False)[0,0]
        union[y1:y2,x1:x2]=torch.maximum(union[y1:y2,x1:x2],restored)
    return dict(probabilities=probabilities,geometries=geometries,foreground=foreground,mask_probability=union)


class CudaBoxRefinementRuntime:
    """Opt-in runtime: same outputs, GPU crops and padded32 captured head.

    Each crop is independent (GroupNorm), so zero padding leaves real outputs
    unchanged. More than32 boxes retain the eager, chunked all-box path. The
    first nonempty CUDA call includes graph warmup/capture and is not a warm call.
    """
    def __init__(self, head, *, crop_size=64, context=1.25, image_normalization='none', batch_size=128):
        self.head=head;self.crop_size=crop_size;self.context=context
        self.image_normalization=image_normalization;self.batch_size=batch_size
        self.device=next(head.parameters()).device
        self._head_graph=None;self._head_input=None;self._head_output=None

    def _probabilities(self, inputs):
        count=len(inputs)
        if not count:return inputs.new_empty((0,1,*inputs.shape[-2:]))
        if self.device.type!='cuda' or count>32 or self.batch_size<32:
            return torch.cat([self.head(batch).sigmoid() for batch in inputs.split(self.batch_size)])
        if self._head_graph is None:
            self._head_input=inputs.new_zeros((32,*inputs.shape[1:]))
            current=torch.cuda.current_stream(self.device);stream=torch.cuda.Stream(device=self.device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(3):self.head(self._head_input).sigmoid()
            current.wait_stream(stream)
            self._head_graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._head_graph,stream=stream):
                self._head_output=self.head(self._head_input).sigmoid()
            current.wait_stream(stream)
        self._head_input.zero_();self._head_input[:count].copy_(inputs)
        self._head_graph.replay()
        return self._head_output[:count]

    def __call__(self, image_bgr, boxes, foreground_logits, detector_geometry):
        with torch.inference_mode():
            prepared=prepare_refinement_tensors(image_bgr,boxes,foreground_logits,detector_geometry,device=self.device,
                crop_size=self.crop_size,context=self.context,image_normalization=self.image_normalization)
            probabilities=self._probabilities(prepared['inputs'])
            result=_reconstruct_refinement_tensors(probabilities,prepared['foreground'],prepared['geometries'])
            if self.device.type!='cuda':
                return {name:value if name=='geometries' else value.float().cpu().numpy() for name,value in result.items()}
            names=[name for name in result if name!='geometries']
            packed=torch.cat([result[name].float().reshape(-1) for name in names])
            host=torch.empty(packed.shape,dtype=torch.float32,device='cpu',pin_memory=True)
            host.copy_(packed,non_blocking=True)
            torch.cuda.current_stream(self.device).synchronize()
            array=host.numpy();offset=0;materialized={'geometries':result['geometries']}
            for name in names:
                value=result[name];count=value.numel()
                materialized[name]=array[offset:offset+count].reshape(tuple(value.shape));offset+=count
            return materialized

class BoxRefiner:
    """Load one local trained head; every segmentation call needs explicit PID inputs."""
    def __init__(self, checkpoint, device='cuda'):
        path=Path(checkpoint).expanduser().resolve()
        if not path.is_file():raise FileNotFoundError(path)
        requested=torch.device(device)
        if requested.type not in ('cpu','cuda'):raise ValueError('Choose cpu or cuda explicitly')
        if requested.type=='cuda' and (not torch.cuda.is_available() or
                (requested.index is not None and requested.index>=torch.cuda.device_count())):
            raise ValueError('Requested CUDA device is unavailable')
        payload=torch.load(path,map_location='cpu',weights_only=True)
        metadata=payload.get('metadata',{}) if isinstance(payload,dict) else {}
        if metadata.get('schema')!='pidnet_box_refinement_v1' or metadata.get('parameter_count')!=99981:
            raise ValueError('Expected the trained99,981-parameter box-refinement checkpoint')
        if metadata.get('input_channels')!=['grayscale_0_1','box_prompt','pidnet_foreground_probability']:
            raise ValueError('Checkpoint image/box/PID foreground input contract differs')
        if metadata.get('crop_size')!=64 or metadata.get('context')!=1.25 or metadata.get('image_normalization','none')!='none':
            raise ValueError('Checkpoint crop or normalization contract differs')
        with torch.random.fork_rng(devices=[]):self.head=TinyBoxUNet()
        self.head.load_state_dict(payload['state_dict'],strict=True)
        self.head=self.head.to(requested).eval().requires_grad_(False)
        self.metadata=metadata;self.device=next(self.head.parameters()).device
        self.runtime=CudaBoxRefinementRuntime(self.head,crop_size=64,context=1.25,image_normalization='none')

    def segment(self, image_bgr, boxes_xyxy_native, *, foreground_logits, detector_geometry):
        if image_bgr is None or boxes_xyxy_native is None or foreground_logits is None or detector_geometry is None:
            raise ValueError('image, boxes, PID foreground_logits and detector_geometry are all required')
        image=np.asarray(image_bgr)
        if image.dtype!=np.uint8 or image.ndim not in (2,3) or (image.ndim==3 and image.shape[2]!=3):
            raise ValueError('image_bgr must be a native uint8 grayscale or BGR frame')
        height,width=image.shape[:2]
        if min(height,width)<1:raise ValueError('image_bgr must not be empty')
        boxes=np.asarray(boxes_xyxy_native)
        if boxes.ndim!=2 or boxes.shape[1]!=4 or boxes.dtype.kind not in 'iuf':
            raise ValueError('boxes_xyxy_native must be an Nx4 numeric XYXY array; point prompts are unsupported')
        if not np.isfinite(boxes).all() or np.any(boxes[:,2:]<=boxes[:,:2]):
            raise ValueError('Boxes must be finite with positive XYXY area')
        if not isinstance(detector_geometry,dict):raise ValueError('detector_geometry must be a letterbox geometry dictionary')
        keys=('height','width','left','top','scale_x','scale_y')
        try:values=np.array([detector_geometry[k] for k in keys],np.float64)
        except (KeyError,TypeError,ValueError) as error:raise ValueError('Missing or invalid PID letterbox geometry') from error
        if not np.isfinite(values).all() or tuple(values[:2])!=(height,width):
            raise ValueError('PID geometry must match the current native image dimensions')
        left,top,sx,sy=values[2:];rw,rh=round(width*sx),round(height*sy)
        if left!=int(left) or top!=int(top) or min(left,top)<0 or min(sx,sy)<=0 or min(rw,rh)<1 or left+rw>640 or top+rh>640:
            raise ValueError('PID geometry must describe valid640-pixel letterbox content')
        if not isinstance(foreground_logits,torch.Tensor) or foreground_logits.dtype!=torch.float32:
            raise ValueError('foreground_logits must be the raw FP32 PID tensor, not a probability mask')
        if foreground_logits.device!=self.device:
            raise ValueError('PID foreground_logits must already be on the segmenter device')
        if tuple(foreground_logits.shape) not in ((160,160),(1,160,160),(1,1,160,160)):
            raise ValueError('Expected one raw160x160 PID foreground-logit plane')
        if not bool(torch.isfinite(foreground_logits).all()):raise ValueError('PID foreground_logits must be finite')
        result=self.runtime(image,boxes,foreground_logits,detector_geometry)
        result['masks']=result['probabilities'][:,0]>=.5
        return result


class CellSegmenter4(BaseSegmentor):
    """Produce crop outputs or native per-cell masks from same-frame PIDNet inputs."""

    def __init__(self, checkpoint=None, device="cuda"):
        self.checkpoint = (Path(__file__).resolve().parent / "cellModel" / "cell_refiner_100k.pt"
                           if checkpoint is None else checkpoint)
        super().__init__(device=device, cache_image_embeddings=False)

    def _load_model(self):
        self._refiner = BoxRefiner(self.checkpoint, device=self.device)
        self.model = self._refiner.head

    def _require_ready(self):
        if self.model is None:
            raise RuntimeError("Box-refinement model has not been loaded")

    def segment(self, image_bgr, boxes_xyxy_native, *, foreground_logits, detector_geometry):
        return self._refiner.segment(image_bgr, boxes_xyxy_native,
            foreground_logits=foreground_logits, detector_geometry=detector_geometry)

    @torch.inference_mode()
    def segment_boxes(self, image, boxes, *, foreground_logits, detector_geometry):
        """Return native bool masks in box order, thresholding after resizing.

        Restore each crop's probabilities independently so overlapping cells
        keep separate masks. Outside-image boxes produce empty native masks.
        """
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        result = self.segment(image, boxes,
            foreground_logits=foreground_logits, detector_geometry=detector_geometry)
        height, width = np.asarray(image).shape[:2]
        probabilities = result["probabilities"]
        geometries = result["geometries"]
        if len(probabilities) != len(geometries) or len(geometries) != len(boxes):
            raise ValueError("Segmenter outputs must preserve detector box order")
        masks = []
        for probability, geometry in zip(probabilities, geometries):
            if tuple(geometry["native_hw"]) != (height, width):
                raise ValueError("Segmenter geometry differs from the current image")
            mask = np.zeros((height, width), dtype=bool)
            if geometry["status"] == "ready":
                x1, y1, x2, y2 = geometry["crop_xyxy"]
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    raise ValueError("Segmenter crop lies outside the native image")
                value = torch.as_tensor(probability, dtype=torch.float32)
                restored = torch.nn.functional.interpolate(
                    value[None], size=(y2-y1, x2-x1), mode="bilinear",
                    align_corners=False)[0, 0]
                mask[y1:y2, x1:x2] = (restored >= .5).cpu().numpy()
            masks.append(mask)
        return masks


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
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, fps=30, parent=None):
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
        self.keep_running = False
        self.quit()
        self.wait()

    def add_point(self, x, y):
        self.points.append([x, y])
        self.labels.append(1)

    def clear_points(self):
        self.points.clear()
        self.labels.clear()

class SamCAM(QMainWindow):
    def __init__(self):
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
        """Receive new frames and display them."""
        # Store the latest image so we can save it
        self.latest_frame = rgb.copy()
        
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg))

    def mousePressEvent(self, event):
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
        self.worker.stop()
        super().closeEvent(event)

    # ----------------------------------------------------
    def save_segmentation(self):
        """Re-run segmentation on original image and save only the cut-out region (RGBA)."""
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
