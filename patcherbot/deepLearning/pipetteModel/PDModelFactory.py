"""Pipette detection model factory for the canonical 384px U-Net + MobileNetV4 boxes model.

Mirrors the CellModel CDModelFactory pattern: private model internals live here,
and consumer classes load this file by path instead of importing a package.

Checkpoint contract (validated on load):
    schema == "canonical_checkpoint_v1"
    experiment.cohort == "pipettes"
    experiment.family == "unet_mobilenetv4_boxes"
    experiment.resolution == 384
    experiment.input_mode == "grayscale_sobel_magnitude"

Inference: uint8 grayscale/BGR/BGRA image -> 384x384 resize -> normalize to
[-1, 1] -> optional BF16 CUDA graph capture/replay or FP32 CPU eager ->
prepare_decode(geometry=True) -> decode(threshold, max_detections=1) ->
pixel-center geometry mapped to native pixels.

The mask is foreground probability, not a binary mask. The box is the trained
tip square, not a bounding box around the entire shaft. Z is signed micrometers.
Scores are uncalibrated. Instances are not thread-safe. Returned NumPy arrays
own their data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import timm
from diffusers import UNet2DModel
from PIL import Image

_SOBEL_INPUT_FEATURES = "grayscale_sobel_magnitude"
_SOBEL_ONLY_FEATURES = ("sobel_magnitude", "sobel_xy")


class _SobelInputConv2d(nn.Conv2d):
    """Use opt-in Sobel maps, preserving grayscale-plus-magnitude by default.

    Inputs remain normalized B,1,H,W grayscale in [-1,1]. The raw 3x3
    derivatives operate on [0,1] intensities with reflect borders. Dividing
    their magnitude by sqrt(32) uses a fixed upper bound, never image statistics.
    """

    def __init__(self, original, input_features=_SOBEL_INPUT_FEATURES):
        if input_features not in (_SOBEL_INPUT_FEATURES, *_SOBEL_ONLY_FEATURES):
            raise ValueError("Unsupported Sobel input_features: " + str(input_features))
        if not isinstance(original, nn.Conv2d) or original.in_channels != 1 or original.groups != 1:
            raise ValueError("Sobel features require a one-channel, ungrouped input convolution")
        channels = 1 if input_features == "sobel_magnitude" else 2
        super().__init__(channels, original.out_channels, original.kernel_size,
                         stride=original.stride, padding=original.padding,
                         dilation=original.dilation, groups=1,
                         bias=original.bias is not None, padding_mode=original.padding_mode,
                         device=original.weight.device, dtype=original.weight.dtype)
        self.input_features = input_features
        kernels = torch.tensor([[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                                [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]],
                               device=original.weight.device, dtype=torch.float32)[:, None]
        self.register_buffer("sobel_kernels", kernels, persistent=False)
        with torch.no_grad():
            self.weight.zero_()
            self.weight[:, :1].copy_(original.weight)
            if input_features == "sobel_xy":
                self.weight.copy_(original.weight.expand_as(self.weight) * .5)
            if self.bias is not None:
                self.bias.copy_(original.bias)
        self.weight.requires_grad_(original.weight.requires_grad)
        if self.bias is not None:
            self.bias.requires_grad_(original.bias.requires_grad)
        self.train(original.training)

    def sobel_derivatives(self, image):
        """Raw FP32 Gx/Gy on [0,1] intensity with reflect borders."""
        if image.ndim != 4 or image.shape[1] != 1 or min(image.shape[-2:]) < 2:
            raise ValueError("Sobel features require B,1,H,W grayscale with H,W >= 2")
        with torch.autocast(device_type=image.device.type, enabled=False):
            gray = (image.float() + 1.) * .5
            derivatives = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"),
                                   self.sobel_kernels.float())
            return derivatives

    def sobel_magnitude(self, image):
        """FP32 features with a finite, zero gradient at an exactly flat input."""
        with torch.autocast(device_type=image.device.type, enabled=False):
            return torch.linalg.vector_norm(self.sobel_derivatives(image), dim=1, keepdim=True) / (32. ** .5)

    def forward(self, image):
        if self.input_features == "sobel_xy":
            features = (self.sobel_derivatives(image) / 4.).to(dtype=image.dtype)
        else:
            features = self.sobel_magnitude(image).to(dtype=image.dtype)
            if self.input_features == _SOBEL_INPUT_FEATURES:
                features = torch.cat((image, features), dim=1)
        return self._conv_forward(features, self.weight, self.bias)


def _enable_unet_sobel(model, input_features=_SOBEL_INPUT_FEATURES):
    """Opt in after loading grayscale weights; the legacy added channel starts at zero."""
    if model.config.in_channels != 1 or model.config.center_input_sample:
        raise ValueError("Sobel features require external grayscale [-1,1] without input centering")
    if isinstance(model.conv_in, _SobelInputConv2d):
        raise ValueError("Sobel input features are already enabled")
    original_shape = list(model.conv_in.weight.shape)
    model.conv_in = _SobelInputConv2d(model.conv_in, input_features=input_features)
    model.register_to_config(input_features=input_features)
    return dict(input_features=input_features, external_in_channels=1,
                convolution_in_channels=model.conv_in.in_channels, source_weight_shape=original_shape,
                feature_weight_shape=list(model.conv_in.weight.shape),
                grayscale_weights="copied" if input_features == _SOBEL_INPUT_FEATURES else "not_used",
                sobel_weights=("zero_initialized" if input_features == _SOBEL_INPUT_FEATURES else
                               "grayscale_copied" if input_features == "sobel_magnitude" else "grayscale_copied_half_per_axis"),
                scaling="gx/4,gy/4" if input_features == "sobel_xy" else "sqrt(gx^2+gy^2)/sqrt(32)", border="reflect",
                derivative_dtype="float32", kernel_size=3)


def _create_unet2d(options):
    """Restore repository input-feature configuration before loading model weights."""
    options = dict(options)
    feature = options.pop("input_features", None)
    if feature not in (None, "none", _SOBEL_INPUT_FEATURES, *_SOBEL_ONLY_FEATURES):
        raise ValueError("Unsupported UNet input_features: " + str(feature))
    if feature in (_SOBEL_INPUT_FEATURES, *_SOBEL_ONLY_FEATURES) and options.get("in_channels", 3) != 1:
        raise ValueError("Sobel features require in_channels=1 for external grayscale input")
    model = UNet2DModel.from_config(options)
    if feature in (_SOBEL_INPUT_FEATURES, *_SOBEL_ONLY_FEATURES):
        _enable_unet_sobel(model, input_features=feature)
    return model


class _UNetMobileNetV4Boxes(nn.Module):
    """Opt-in shared U-Net features -> MobileNetV4 boxes and optional signed Z.

    External inputs are normalized grayscale; input_features on the U-Net may
    select Sobel-only maps. Z predictions are signed micrometers. The default
    initialization is random and never downloads pretrained weights.
    """

    def __init__(self, unet_options=None, head_channels=32, predict_z=False, pretrained=False):
        super().__init__()
        if not isinstance(head_channels, int) or head_channels < 1:
            raise ValueError("head_channels must be a positive integer")
        options = dict(sample_size=448, in_channels=1, out_channels=1,
                       block_out_channels=(32, 64, 128), layers_per_block=1,
                       down_block_types=("DownBlock2D",) * 3,
                       up_block_types=("UpBlock2D",) * 3,
                       add_attention=False, norm_num_groups=8)
        options.update(unet_options or {})
        self.unet = _create_unet2d(options)
        config = self.unet.config
        if (config.in_channels != 1 or config.out_channels != 1
                or config.time_embedding_type != "positional"
                or self.unet.class_embedding is not None or config.center_input_sample
                or any(hasattr(block, "skip_conv") for block in (*self.unet.down_blocks, *self.unet.up_blocks))):
            raise ValueError("Box model requires an unconditioned single-channel positional-time U-Net")
        self.predict_z = bool(predict_z)
        self.head_channels = head_channels
        self.mask_head = self.unet.conv_out
        self.unet.conv_out = nn.Identity()
        self.register_buffer("timestep_zero", torch.zeros(1, dtype=torch.long), persistent=False)
        self.backbone = timm.create_model("mobilenetv4_conv_small", pretrained=pretrained,
            in_chans=self.mask_head.in_channels, features_only=True, out_indices=(1, 4))
        low_channels, high_channels = self.backbone.feature_info.channels()
        self.detail = nn.Conv2d(low_channels, head_channels, 1)
        self.context = nn.Conv2d(high_channels, head_channels, 1)
        combined = 2 * head_channels
        self.fusion = nn.Sequential(nn.Conv2d(combined, combined, 3, padding=1, groups=combined),
                                    nn.SiLU(), nn.Conv2d(combined, head_channels, 1), nn.SiLU())
        self.box_head = nn.Conv2d(head_channels, 5, 1)
        nn.init.normal_(self.box_head.weight, std=.001)
        nn.init.zeros_(self.box_head.bias)
        with torch.no_grad():
            self.box_head.bias[0] = -4.595
        self.z_head = nn.Conv2d(head_channels, 1, 1) if self.predict_z else None
        if self.z_head is not None:
            nn.init.normal_(self.z_head.weight, std=.001)
            nn.init.zeros_(self.z_head.bias)
        # Optional head, injected post-construction by the adapter when the
        # checkpoint contains z_confidence_head weights.
        self.z_confidence_head: Optional[nn.Conv2d] = None
        self.focus_head: Optional[nn.Conv2d] = None

    def forward(self, image):
        if image.ndim != 4 or image.shape[0] < 1 or image.shape[1] != 1 or min(image.shape[-2:]) < 32:
            raise ValueError("Box model expects normalized Bx1xHxW images with H,W >= 32")
        if not image.is_floating_point():
            raise TypeError("Box model input must be floating point")
        features = self.unet(image, timestep=self.timestep_zero).sample
        low, high = self.backbone(features)
        context = F.interpolate(self.context(high), size=low.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.fusion(torch.cat((self.detail(low), context), dim=1))
        boxes = self.box_head(fused).float()
        output = dict(mask_logits=self.mask_head(features), quality=boxes[:, :1],
                      ltrb_raw=boxes[:, 1:], input_hw=tuple(image.shape[-2:]))
        if self.z_head is not None:
            output["z_um"] = self.z_head(fused).float()
        if self.z_confidence_head is not None:
            output['z_confidence_logits'] = self.z_confidence_head(fused.detach()).float()
        if self.focus_head is not None:
            output['focus_logits'] = self.focus_head(features).float()
        return output

    @staticmethod
    def prepare_decode(output, *, geometry=False):
        """Fixed-shape decoder arithmetic; ephemeral graph outputs, never predictions."""
        quality, raw = output["quality"].float(), output["ltrb_raw"].float()
        finite = torch.isfinite(quality) & torch.isfinite(raw).all(dim=1, keepdim=True)
        if output.get("z_um") is not None:
            finite = finite & torch.isfinite(output["z_um"])
        probability = quality.sigmoid().masked_fill(~finite, -1.)
        maxima = probability.eq(F.max_pool2d(probability, 3, stride=1, padding=1))
        scores = probability.masked_fill(~maxima | ~finite, -1.)
        prepared = dict(finite=finite, scores=scores, distances=F.softplus(raw) + 1e-3,
                        ranked=[torch.topk(score.flatten(), score.numel()) for score in scores[:, 0]])
        if geometry:
            height, width = output["input_hw"]
            stride_y, stride_x = height / quality.shape[-2], width / quality.shape[-1]
            prepared["geometry"] = []
            prepared["invalid_counts"] = (~finite).flatten(1).sum(1)
            for index, ((_, positions), ltrb) in enumerate(zip(prepared["ranked"], prepared["distances"])):
                y = torch.div(positions, scores.shape[-1], rounding_mode="floor")
                x = positions.remainder(scores.shape[-1])
                cx, cy = (x.float() + .5) * stride_x, (y.float() + .5) * stride_y
                extent = ltrb[:, y, x]
                boxes = torch.stack(((cx - extent[0] * stride_x).clamp(0, width),
                                     (cy - extent[1] * stride_y).clamp(0, height),
                                     (cx + extent[2] * stride_x).clamp(0, width),
                                     (cy + extent[3] * stride_y).clamp(0, height)), dim=1)
                valid = torch.isfinite(boxes).all(1) & (boxes[:, 2:] > boxes[:, :2]).all(1)
                z = output["z_um"][index, 0, y, x] if output.get("z_um") is not None else None
                prepared["geometry"].append((boxes, valid, z))
        return prepared

    @staticmethod
    def decode(output, *, threshold=.3, max_detections=100, nms_iou=.7, prepared=None):
        """Return boxes in input pixels and optional signed Z; count invalid candidates."""
        from torchvision.ops import nms
        if not 0 < threshold < 1 or not isinstance(max_detections, int) or max_detections < 1 or not 0 <= nms_iou <= 1:
            raise ValueError("Invalid box decoding settings")
        quality = output["quality"].float()
        raw = output["ltrb_raw"].float()
        if quality.ndim != 4 or quality.shape[1] != 1 or raw.shape != (len(quality), 4, *quality.shape[-2:]):
            raise ValueError("Inconsistent quality/box map geometry")
        height, width = output["input_hw"]
        if height < 1 or width < 1 or min(quality.shape[-2:]) < 1:
            raise ValueError("Input and quality-map dimensions must be positive")
        z = output.get("z_um")
        if z is not None and z.shape != quality.shape:
            raise ValueError("Z map must match quality-map geometry")
        prepared = _UNetMobileNetV4Boxes.prepare_decode(output) if prepared is None else prepared
        finite, scores, distances = prepared["finite"], prepared["scores"], prepared["distances"]
        stride_y, stride_x = height / quality.shape[-2], width / quality.shape[-1]
        rows = []
        for index, (score, ltrb) in enumerate(zip(scores[:, 0], distances)):
            confidence, positions = prepared["ranked"][index]
            selected = confidence >= threshold
            confidence, positions = confidence[selected], positions[selected]
            if "geometry" in prepared:
                boxes, valid, ranked_z = prepared["geometry"][index]
                boxes, valid = boxes[selected], valid[selected]
                selected_z = ranked_z[selected][valid] if ranked_z is not None else None
            else:
                y = torch.div(positions, score.shape[1], rounding_mode="floor")
                x = positions.remainder(score.shape[1])
                cx, cy = (x.float() + .5) * stride_x, (y.float() + .5) * stride_y
                extent = ltrb[:, y, x]
                boxes = torch.stack(((cx - extent[0] * stride_x).clamp(0, width),
                                     (cy - extent[1] * stride_y).clamp(0, height),
                                     (cx + extent[2] * stride_x).clamp(0, width),
                                     (cx + extent[3] * stride_y).clamp(0, height)), dim=1)
                valid = torch.isfinite(boxes).all(1) & (boxes[:, 2:] > boxes[:, :2]).all(1)
                selected_z = z[index, 0, y, x][valid] if z is not None else None
            invalid_map = prepared["invalid_counts"][index] if "invalid_counts" in prepared else (~finite[index]).sum()
            invalid = int(invalid_map) + int((~valid).sum())
            boxes, confidence = boxes[valid], confidence[valid]
            # NMS cannot suppress its highest-score first box. Match the
            # native descending sort (including ties) while avoiding its full
            # pairwise suppression when the public contract needs one tip.
            keep = (torch.argsort(confidence, descending=True)[:1] if max_detections == 1
                    else nms(boxes, confidence, nms_iou)[:max_detections])
            boxes, confidence = boxes[keep], confidence[keep]
            row = dict(xy=(boxes[:, :2] + boxes[:, 2:]) * .5, boxes=boxes,
                       boxes_xyxy=boxes, confidence=confidence, invalid_candidates=invalid,
                       failure_reasons=["nonfinite_or_invalid_box_candidates"] if invalid else [])
            if selected_z is not None:
                row["z_um"] = selected_z[keep]
            if 'z_confidence_logits' in output:
                row['z_confidence'] = output['z_confidence_logits'][index, 0].flatten()[positions][valid][keep].sigmoid()
            rows.append(row)
        return rows


_UNET_OPTIONS = {'sample_size': 384,
    'in_channels': 1,
    'out_channels': 1,
    'center_input_sample': False,
    'time_embedding_type': 'positional',
    'time_embedding_dim': None,
    'freq_shift': 0,
    'flip_sin_to_cos': True,
    'down_block_types': ['DownBlock2D',
                         'DownBlock2D',
                         'DownBlock2D',
                         'DownBlock2D'],
    'mid_block_type': 'UNetMidBlock2D',
    'up_block_types': ['UpBlock2D', 'UpBlock2D', 'UpBlock2D', 'UpBlock2D'],
    'block_out_channels': [32, 64, 128, 128],
    'layers_per_block': 1,
    'mid_block_scale_factor': 1,
    'downsample_padding': 1,
    'downsample_type': 'conv',
    'upsample_type': 'conv',
    'dropout': 0.0,
    'act_fn': 'silu',
    'attention_head_dim': 8,
    'norm_num_groups': 8,
    'attn_norm_num_groups': None,
    'norm_eps': 1e-05,
    'resnet_time_scale_shift': 'default',
    'add_attention': True,
    'class_embed_type': None,
    'num_class_embeds': None,
    'num_train_timesteps': None,
    'input_features': 'grayscale_sobel_magnitude'}


class _PipetteDetector4Adapter:
    """Self-contained mask/box/Z adapter for canonical 384px pipette weights.

    CUDA uses BF16 and a captured model/decoder-preparation graph by default.
    CPU uses FP32. Pass cuda_graph=False for eager execution. Graph capture
    errors are raised, never silently replaced with another execution path.
    """

    INPUT_SIZE = 384
    FOCUS_MODEL = False

    def _prepare_image(self, image):
        h, w = image.shape
        resized = cv2.resize(
            image, (self.INPUT_SIZE, self.INPUT_SIZE),
            interpolation=cv2.INTER_AREA,
        )
        return resized, (0, 0, w, h)

    def __init__(self, model_path, device=None, *, threshold=0.001,
                 cuda_graph=True, precision=None):
        if not 0 < threshold < 1:
            raise ValueError("threshold must be between zero and one")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Supported devices: cpu, cuda")
        self.precision = precision or ("bfloat16" if self.device.type == "cuda" else "float32")
        if self.precision not in ("float32", "bfloat16") or (self.device.type == "cpu" and self.precision != "float32"):
            raise ValueError("Use float32 CPU or float32/bfloat16 CUDA")
        if self.device.type == "cuda" and self.precision == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("BF16 unavailable; select precision='float32'")
        payload = torch.load(Path(model_path), map_location="cpu", weights_only=True)
        exp = payload.get("experiment", {})
        state = payload.get("model", {})
        if self.FOCUS_MODEL:
            if ("focus_head.weight" not in state or "focus_head.bias" not in state
                    or any(key.startswith(("z_head.", "z_confidence_head.")) for key in state)):
                raise ValueError("Expected X1 checkpoint with a focal head and no Z head")
        elif (payload.get("schema") != "canonical_checkpoint_v1" or
            exp.get("cohort") != "pipettes" or
            exp.get("family") != "unet_mobilenetv4_boxes" or
            exp.get("resolution") != self.INPUT_SIZE or
            exp.get("input_mode") != "grayscale_sobel_magnitude"):
            raise ValueError(
                f"Expected canonical {self.INPUT_SIZE}px grayscale+Sobel pipette checkpoint"
            )
        options = dict(_UNET_OPTIONS, sample_size=self.INPUT_SIZE)
        self.model = _UNetMobileNetV4Boxes(
            unet_options=options, predict_z=not self.FOCUS_MODEL, pretrained=False,
        )
        if self.FOCUS_MODEL:
            self.model.focus_head = nn.Conv2d(self.model.mask_head.in_channels, 1, 1)
        if "z_confidence_head.weight" in state:
            self.model.z_confidence_head = nn.Conv2d(self.model.head_channels, 1, 1)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval().requires_grad_(False)
        if self.FOCUS_MODEL:
            self.model.to(memory_format=torch.channels_last)
        self.threshold = float(threshold)
        self.epoch = payload.get("epoch")
        self._graph = None
        self._captured = None
        self._static = torch.zeros(
            (1, 1, self.INPUT_SIZE, self.INPUT_SIZE), device=self.device,
        )
        if self.FOCUS_MODEL:
            self._static = self._static.contiguous(memory_format=torch.channels_last)
        if cuda_graph and self.device.type == "cuda":
            self._capture()

    def _autocast(self):
        return torch.autocast(self.device.type, dtype=torch.bfloat16,
                              enabled=self.precision == "bfloat16", cache_enabled=False)

    def _execute(self, image):
        output = self.model(image)
        prepared = self.model.prepare_decode(output, geometry=True)
        prepared["mask_probability"] = output["mask_logits"][0, 0].float().sigmoid()
        if "focus_logits" in output:
            prepared["focus_probability"] = output["focus_logits"][0, 0].float().sigmoid()
        return output, prepared

    def _capture(self):
        with torch.cuda.device(self.device):
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream), torch.inference_mode(), self._autocast():
                for _ in range(3):
                    self._execute(self._static)
            torch.cuda.current_stream(self.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.inference_mode(), self._autocast(), torch.cuda.graph(graph):
                self._captured = self._execute(self._static)
            self._graph = graph

    @torch.inference_mode()
    def predict(self, image, *, native_mask=True, max_detections=1):
        """Return camera-coordinate detections and a native or model-size mask."""
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.size == 0:
            raise ValueError("Expected a nonempty uint8 image")
        if image.ndim == 3 and image.shape[2] in (3, 4):
            image = cv2.cvtColor(
                image, cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_BGRA2GRAY,
            )
        elif image.ndim == 3 and image.shape[2] == 1:
            image = image[:, :, 0]
        if image.ndim != 2:
            raise ValueError("Expected grayscale, BGR or BGRA image")

        if not isinstance(max_detections, int) or max_detections < 1:
            raise ValueError("max_detections must be a positive integer")
        h, w = image.shape
        resized, crop = self._prepare_image(image)
        left, top, right, bottom = crop
        scale = np.array([
            (right - left) / self.INPUT_SIZE,
            (bottom - top) / self.INPUT_SIZE,
        ])
        offset = np.array([left, top])
        box_scale, box_offset = np.tile(scale, 2), np.tile(offset, 2)
        resized = resized.astype(np.float32) / 127.5 - 1.
        inputs = torch.from_numpy(resized)[None, None].to(self.device)
        if self.FOCUS_MODEL:
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        if self._graph is not None:
            self._static.copy_(inputs)
            self._graph.replay()
            output, prepared = self._captured
        else:
            with self._autocast():
                output, prepared = self._execute(inputs)

        row = self.model.decode(
            output, threshold=self.threshold, max_detections=max_detections, prepared=prepared,
        )[0]
        fields = [
            row["boxes_xyxy"].float(),
            row["confidence"].float()[:, None],
            (row["z_um"].float() if "z_um" in row else
             torch.full_like(row["confidence"], float("nan")))[:, None],
        ]
        has_z_confidence = "z_confidence" in row
        if has_z_confidence:
            fields.append(row["z_confidence"].float()[:, None])
        values = torch.cat(fields, dim=1).cpu().numpy()
        probability = prepared["mask_probability"].cpu().numpy().copy()
        focus = (prepared["focus_probability"].cpu().numpy().copy()
                 if "focus_probability" in prepared else None)
        mask_valid = bool(np.isfinite(probability).all())
        mask = None
        mask_score = None
        if mask_valid:
            foreground = probability >= .5
            if foreground.any():
                mask_score = float(np.mean(probability, where=foreground))
            mask = probability
            if native_mask:
                mask = np.asarray(Image.fromarray(probability, mode="F").resize(
                    (right - left, bottom - top), Image.Resampling.BILINEAR,
                )).copy()
                if crop != (0, 0, w, h):
                    native = np.zeros((h, w), dtype=np.float32)
                    native[top:bottom, left:right] = mask
                    mask = native

        candidates = []
        for value in values:
            box = (value[:4].astype(np.float64) + .5) * box_scale - .5 + box_offset
            box[0::2] = np.clip(box[0::2], 0, w)
            box[1::2] = np.clip(box[1::2], 0, h)
            center = (value[:2] + value[2:4]) * .5
            point = (center + .5) * scale - .5 + offset
            tip = None
            if np.isfinite(point).all() and (point >= 0).all() and (point <= [w - 1, h - 1]).all():
                tip = tuple(float(v) for v in point)
            z_score = None
            if has_z_confidence and np.isfinite(value[6]):
                z_score = float(value[6])
            candidates.append(dict(
                box_xyxy=box, tip_xy=tip, z_um=float(value[5]) if "z_um" in row else None,
                box_confidence=float(value[4]), z_confidence=z_score,
            ))
            if self.FOCUS_MODEL:
                candidates[-1]["focus_score"] = self._focus_score(focus, center) if tip is not None else None
        selected = candidates[0] if candidates else dict(
            box_xyxy=None, tip_xy=None, z_um=None, box_confidence=None, z_confidence=None,
        )
        if self.FOCUS_MODEL and not candidates:
            selected["focus_score"] = None
        return dict(
            **selected, mask=mask, mask_valid=mask_valid, candidates=candidates,
            mask_foreground_confidence=mask_score,
            invalid_candidates=row["invalid_candidates"], epoch=self.epoch,
        )


class _PipetteDetector5Adapter(_PipetteDetector4Adapter):
    """R12 August preprocessing with shared model execution and decoding."""

    INPUT_SIZE = 512
    CROP_SIZE = 896

    def _prepare_image(self, image):
        h, w = image.shape
        if min(h, w) < self.CROP_SIZE:
            raise ValueError("R12 requires an image at least 896 pixels in each dimension")
        if (h, w) not in ((1280, 1280), (1024, 1280)):
            raise ValueError(f"Native image shape is outside R12's verified protocol: {(h, w)}")

        left = (w - self.CROP_SIZE) // 2
        top = (h - self.CROP_SIZE) // 2
        right, bottom = left + self.CROP_SIZE, top + self.CROP_SIZE
        crop = image[top:bottom, left:right]
        p1, p99 = map(float, np.percentile(crop, [1, 99], method="linear"))
        stretched = crop if p99 <= p1 else np.rint(np.clip(
            (crop.astype(np.float64) - p1) * 255.0 / (p99 - p1), 0, 255,
        )).astype(np.uint8)
        resized = cv2.resize(
            stretched, (self.INPUT_SIZE, self.INPUT_SIZE),
            interpolation=cv2.INTER_AREA,
        )
        return resized, (left, top, right, bottom)


class _X1Adapter(_PipetteDetector5Adapter):
    """X1 box localization with its native focal score; no Z or implicit gate."""

    FOCUS_MODEL = True

    @staticmethod
    def _focus_score(field, point):
        """Gaussian mean at subpixel model XY: sigma 2, radius 6 model pixels.

        Matches model_analysis.tip_focus_score.local_mean without importing
        training code. Work in prepared coordinates for either native crop.
        """
        if field is None or not np.isfinite(field).all():
            return None
        height, width = field.shape
        point = np.asarray(point, dtype=np.float64)
        if not np.isfinite(point).all() or not ((point >= 0).all() and (point <= [width-1, height-1]).all()):
            return None
        offsets = np.arange(-6, 7, dtype=np.float64)
        yy, xx = np.meshgrid(offsets, offsets, indexing="ij")
        use = xx * xx + yy * yy <= 36
        x, y = point[0] + xx[use], point[1] + yy[use]
        weights = np.exp(-(xx[use] ** 2 + yy[use] ** 2) / 8.)
        valid = (x >= 0) & (x <= width-1) & (y >= 0) & (y <= height-1)
        x, y, weights = x[valid], y[valid], weights[valid]
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        x1, y1 = np.minimum(x0 + 1, width-1), np.minimum(y0 + 1, height-1)
        dx, dy = x - x0, y - y0
        values = ((1-dx)*(1-dy)*field[y0,x0] + dx*(1-dy)*field[y0,x1]
                  + (1-dx)*dy*field[y1,x0] + dx*dy*field[y1,x1])
        return float(np.sum(values * weights) / np.sum(weights))


class PDModelFactory:
    """Construct a supported pipette inference backend."""

    MODEL_VERSIONS = {
        "R02": "unet_mobilenetv4_boxes",
        "R12": "unet_mobilenetv4_boxes_v5",
        "X1": "unet_mobilenetv4_boxes_x1",
    }

    SUPPORTED_MODELS = (
        "unet_mobilenetv4_boxes",
        "unet_mobilenetv4_boxes_v5",
        "unet_mobilenetv4_boxes_x1",
    )

    @classmethod
    def create(
        cls,
        model_type: str,
        model_path: str | Path,
        device: Optional[str] = None,
        threshold: float = 0.001,
        cuda_graph: bool = True,
        precision: Optional[str] = None,
    ) -> _PipetteDetector4Adapter:
        normalized = str(model_type).strip().lower()
        adapters = {
            "unet_mobilenetv4_boxes": _PipetteDetector4Adapter,
            "unet_mobilenetv4_boxes_v5": _PipetteDetector5Adapter,
            "unet_mobilenetv4_boxes_x1": _X1Adapter,
        }
        adapter_type = adapters.get(normalized)
        if adapter_type is None:
            supported = ", ".join(cls.SUPPORTED_MODELS)
            raise ValueError(
                f"Unsupported model_type {model_type!r}; expected one of: {supported}"
            )
        return adapter_type(
            model_path=model_path, device=device, threshold=threshold,
            cuda_graph=cuda_graph, precision=precision,
        )
