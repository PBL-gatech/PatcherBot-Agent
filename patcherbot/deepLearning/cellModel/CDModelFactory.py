"""Inference-only model factory for the custom neuron detectors."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.ops import nms


INPUT_SIZE = 640
STRIDE = 4
OUTPUT_SIZE = INPUT_SIZE // STRIDE
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


class _ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )


class _SegFormerNeuronDetector(nn.Module):
    """Inference-only copy of the architecture used by Idea 20."""

    def __init__(self, encoder_config_path: Path, decoder_channels: int = 96) -> None:
        super().__init__()
        from transformers import SegformerConfig, SegformerModel

        config = SegformerConfig.from_json_file(str(encoder_config_path))
        self.encoder = SegformerModel(config)
        hidden_sizes = list(config.hidden_sizes)
        self.projections = nn.ModuleList(
            nn.Conv2d(channels, decoder_channels, 1) for channels in hidden_sizes
        )
        self.fuse = nn.Sequential(
            _ConvNormAct(len(hidden_sizes) * decoder_channels, decoder_channels),
            _ConvNormAct(decoder_channels, decoder_channels, 3),
        )
        self.mask_head = nn.Sequential(
            _ConvNormAct(decoder_channels, 48, 3), nn.Conv2d(48, 1, 1)
        )
        self.distance_head = nn.Sequential(
            _ConvNormAct(decoder_channels, 48, 3), nn.Conv2d(48, 1, 1)
        )
        self.detection_stem = nn.Sequential(
            _ConvNormAct(decoder_channels + 2, decoder_channels, 3),
            _ConvNormAct(decoder_channels, decoder_channels, 3),
        )
        self.center_head = nn.Conv2d(decoder_channels, 1, 1)
        self.offset_head = nn.Conv2d(decoder_channels, 2, 1)
        self.log_size_head = nn.Conv2d(decoder_channels, 2, 1)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encoder(images, output_hidden_states=True)
        hidden_states = encoded.hidden_states
        if hidden_states is None or len(hidden_states) != 4:
            raise RuntimeError("MiT-B1 did not return four feature maps")
        target_size = hidden_states[0].shape[-2:]
        projected = [
            F.interpolate(layer(feature), target_size, mode="bilinear", align_corners=False)
            for layer, feature in zip(self.projections, hidden_states)
        ]
        features = self.fuse(torch.cat(projected, dim=1))
        mask = self.mask_head(features)
        distance = self.distance_head(features)
        detection = self.detection_stem(
            torch.cat((features, mask.sigmoid(), distance.sigmoid()), dim=1)
        )
        return {
            "center": self.center_head(detection),
            "offset": self.offset_head(detection),
            "log_size": self.log_size_head(detection),
        }


class _SegFormerAdapter:
    """Load the trained SegFormer detector and return neuron centroids."""

    def __init__(
        self,
        model_path: str | Path,
        encoder_config_path: str | Path,
        device: Optional[str] = None,
        conf: float = 0.20,
        topk: int = 100,
        nms_iou: float = 0.80,
    ) -> None:
        self.device = torch.device(
            device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.conf = float(conf)
        self.topk = int(topk)
        self.nms_iou = float(nms_iou)

        checkpoint = torch.load(
            Path(model_path), map_location="cpu", weights_only=True
        )
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
            raise ValueError("SegFormer checkpoint does not contain a model state dictionary")
        decoder_channels = int(
            checkpoint.get("config", {}).get("model", {}).get("decoder_channels", 96)
        )
        self.model = _SegFormerNeuronDetector(
            Path(encoder_config_path), decoder_channels
        )
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.to(self.device).eval()

    @staticmethod
    def _prepare(img: np.ndarray) -> tuple[torch.Tensor, dict[str, float]]:
        if img.ndim == 2:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 3:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Expected a grayscale or three-channel image")

        height, width = rgb.shape[:2]
        scale = min(INPUT_SIZE / width, INPUT_SIZE / height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        left = (INPUT_SIZE - resized_width) // 2
        top = (INPUT_SIZE - resized_height) // 2
        resized = Image.fromarray(rgb).resize(
            (resized_width, resized_height), Image.Resampling.BILINEAR
        )
        canvas = np.full(
            (INPUT_SIZE, INPUT_SIZE, 3), 114.0 / 255.0, dtype=np.float32
        )
        canvas[top : top + resized_height, left : left + resized_width] = (
            np.asarray(resized, dtype=np.float32) / 255.0
        )
        tensor = torch.from_numpy(canvas).permute(2, 0, 1)
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        meta = {
            "width": float(width),
            "height": float(height),
            "scale_x": float(resized_width / width),
            "scale_y": float(resized_height / height),
            "left": float(left),
            "top": float(top),
        }
        return tensor, meta

    @torch.inference_mode()
    def predict(
        self, img: np.ndarray, max_outputs: Optional[int] = None, *, include_boxes: bool = False
    ) -> List[Tuple[int, int, float] | Tuple[int, int, float, Tuple[float, float, float, float]]]:
        tensor, meta = self._prepare(img)
        tensor = tensor.unsqueeze(0).to(self.device)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            outputs = self.model(tensor)

        heatmap = outputs["center"].sigmoid()
        heatmap *= heatmap.eq(F.max_pool2d(heatmap, 3, 1, 1))
        scores, indices = heatmap[0, 0].flatten().topk(
            min(self.topk, heatmap[0, 0].numel())
        )
        ys = torch.div(indices, OUTPUT_SIZE, rounding_mode="floor")
        xs = indices.remainder(OUTPUT_SIZE)
        offset = outputs["offset"][0, :, ys, xs].sigmoid().transpose(0, 1)
        log_size = outputs["log_size"][0, :, ys, xs].transpose(0, 1).clamp(-6.0, 0.0)
        center_x = ((xs.to(offset.dtype) + offset[:, 0]) * STRIDE - meta["left"]) / meta["scale_x"]
        center_y = ((ys.to(offset.dtype) + offset[:, 1]) * STRIDE - meta["top"]) / meta["scale_y"]
        widths = log_size[:, 0].exp() * INPUT_SIZE / meta["scale_x"]
        heights = log_size[:, 1].exp() * INPUT_SIZE / meta["scale_y"]
        inside = (
            (center_x >= 0)
            & (center_x <= meta["width"])
            & (center_y >= 0)
            & (center_y <= meta["height"])
            & (scores >= self.conf)
        )
        center_x, center_y = center_x[inside], center_y[inside]
        widths, heights, scores = widths[inside], heights[inside], scores[inside]
        if scores.numel() == 0:
            return []

        boxes = torch.stack(
            (
                center_x - widths * 0.5,
                center_y - heights * 0.5,
                center_x + widths * 0.5,
                center_y + heights * 0.5,
            ),
            dim=1,
        )
        boxes[:, (0, 2)].clamp_(0.0, meta["width"])
        boxes[:, (1, 3)].clamp_(0.0, meta["height"])
        keep = nms(boxes.float(), scores.float(), self.nms_iou)
        if max_outputs is not None:
            keep = keep[: max(0, int(max_outputs))]

        detections = []
        for index in keep.tolist():
            x = int(round(float(center_x[index])))
            y = int(round(float(center_y[index])))
            if 0 <= x < int(meta["width"]) and 0 <= y < int(meta["height"]):
                detection = (x, y, float(scores[index]))
                if include_boxes:
                    detection += (tuple(boxes[index].tolist()),)
                detections.append(detection)
        return detections


class _PIDConvModule(nn.Module):
    """Conv/BatchNorm/ReLU block used by the trained PIDNet-S backbone."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        activation: bool = True,
        order: tuple[str, ...] = ("conv", "bn", "act"),
    ) -> None:
        super().__init__()
        self.order = order
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        norm_channels = (
            in_channels if order.index("bn") < order.index("conv") else out_channels
        )
        self.bn = nn.BatchNorm2d(norm_channels, eps=1e-5, momentum=0.1)
        self.act = nn.ReLU(inplace=True) if activation else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = inputs
        for operation in self.order:
            if operation == "conv":
                output = self.conv(output)
            elif operation == "bn":
                output = self.bn(output)
            elif operation == "act" and self.act is not None:
                output = self.act(output)
        return output


class _PIDBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        act_out: bool = True,
    ) -> None:
        super().__init__()
        self.conv1 = _PIDConvModule(
            in_channels, channels, 3, stride=stride, padding=1
        )
        self.conv2 = _PIDConvModule(
            channels, channels, 3, padding=1, activation=False
        )
        self.downsample = downsample
        self.act = nn.ReLU(inplace=True) if act_out else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs if self.downsample is None else self.downsample(inputs)
        output = self.conv2(self.conv1(inputs)) + residual
        return self.act(output) if self.act is not None else output


class _PIDBottleneck(nn.Module):
    expansion = 2

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        act_out: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = _PIDConvModule(in_channels, channels, 1)
        self.conv2 = _PIDConvModule(
            channels, channels, 3, stride=stride, padding=1
        )
        self.conv3 = _PIDConvModule(
            channels, channels * self.expansion, 1, activation=False
        )
        self.downsample = downsample
        self.act = nn.ReLU(inplace=True) if act_out else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs if self.downsample is None else self.downsample(inputs)
        output = self.conv3(self.conv2(self.conv1(inputs))) + residual
        return self.act(output) if self.act is not None else output


def _pid_make_layer(
    block: type[_PIDBasicBlock] | type[_PIDBottleneck],
    in_channels: int,
    channels: int,
    num_blocks: int,
    stride: int = 1,
) -> nn.Sequential:
    downsample = None
    if stride != 1 or in_channels != channels * block.expansion:
        downsample = _PIDConvModule(
            in_channels,
            channels * block.expansion,
            1,
            stride=stride,
            activation=False,
        )
    layers: list[nn.Module] = [block(in_channels, channels, stride, downsample)]
    in_channels = channels * block.expansion
    for index in range(1, num_blocks):
        layers.append(block(in_channels, channels, act_out=index != num_blocks - 1))
    return nn.Sequential(*layers)


def _pid_make_single_layer(
    block: type[_PIDBasicBlock] | type[_PIDBottleneck],
    in_channels: int,
    channels: int,
    stride: int = 1,
) -> nn.Module:
    downsample = None
    if stride != 1 or in_channels != channels * block.expansion:
        downsample = _PIDConvModule(
            in_channels,
            channels * block.expansion,
            1,
            stride=stride,
            activation=False,
        )
    return block(in_channels, channels, stride, downsample, act_out=False)


class _PagFM(nn.Module):
    def __init__(self, in_channels: int, channels: int) -> None:
        super().__init__()
        self.f_i = _PIDConvModule(in_channels, channels, 1, activation=False)
        self.f_p = _PIDConvModule(in_channels, channels, 1, activation=False)

    def forward(self, x_p: torch.Tensor, x_i: torch.Tensor) -> torch.Tensor:
        f_i = F.interpolate(
            self.f_i(x_i), size=x_p.shape[-2:], mode="bilinear", align_corners=False
        )
        f_p = self.f_p(x_p)
        sigma = torch.sigmoid(torch.sum(f_p * f_i, dim=1, keepdim=True))
        x_i = F.interpolate(
            x_i, size=x_p.shape[-2:], mode="bilinear", align_corners=False
        )
        return sigma * x_i + (1.0 - sigma) * x_p


class _PAPPM(nn.Module):
    def __init__(
        self, in_channels: int, branch_channels: int, out_channels: int
    ) -> None:
        super().__init__()
        preactivation = ("bn", "act", "conv")
        self.scales = nn.ModuleList(
            [_PIDConvModule(in_channels, branch_channels, 1, order=preactivation)]
        )
        for kernel, stride, padding in ((5, 2, 2), (9, 4, 4), (17, 8, 8)):
            self.scales.append(
                nn.Sequential(
                    nn.AvgPool2d(
                        kernel_size=kernel, stride=stride, padding=padding
                    ),
                    _PIDConvModule(
                        in_channels, branch_channels, 1, order=preactivation
                    ),
                )
            )
        self.scales.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                _PIDConvModule(
                    in_channels, branch_channels, 1, order=preactivation
                ),
            )
        )
        self.processes = _PIDConvModule(
            branch_channels * 4,
            branch_channels * 4,
            3,
            padding=1,
            groups=4,
            order=preactivation,
        )
        self.compression = _PIDConvModule(
            branch_channels * 5, out_channels, 1, order=preactivation
        )
        self.shortcut = _PIDConvModule(
            in_channels, out_channels, 1, order=preactivation
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = self.scales[0](inputs)
        features = []
        for scale in self.scales[1:]:
            feature = F.interpolate(
                scale(inputs),
                size=inputs.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            features.append(feature + base)
        processed = self.processes(torch.cat(features, dim=1))
        return (
            self.compression(torch.cat((base, processed), dim=1))
            + self.shortcut(inputs)
        )


class _LightBag(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.f_p = _PIDConvModule(in_channels, out_channels, 1, activation=False)
        self.f_i = _PIDConvModule(in_channels, out_channels, 1, activation=False)

    def forward(
        self, x_p: torch.Tensor, x_i: torch.Tensor, x_d: torch.Tensor
    ) -> torch.Tensor:
        sigma = torch.sigmoid(x_d)
        return self.f_p((1.0 - sigma) * x_i + x_p) + self.f_i(
            x_i + sigma * x_p
        )


class _PIDNetSBackbone(nn.Module):
    """Exact PIDNet-S topology required by the trained detector checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            _PIDConvModule(3, 32, 3, stride=2, padding=1),
            _PIDConvModule(32, 32, 3, stride=2, padding=1),
            _pid_make_layer(_PIDBasicBlock, 32, 32, 2),
            nn.ReLU(),
            _pid_make_layer(_PIDBasicBlock, 32, 64, 2, stride=2),
            nn.ReLU(),
        )
        self.relu = nn.ReLU()
        self.i_branch_layers = nn.ModuleList(
            [
                _pid_make_layer(_PIDBasicBlock, 64, 128, 3, stride=2),
                _pid_make_layer(_PIDBasicBlock, 128, 256, 3, stride=2),
                _pid_make_layer(_PIDBottleneck, 256, 256, 2, stride=2),
            ]
        )
        self.p_branch_layers = nn.ModuleList(
            [
                _pid_make_layer(_PIDBasicBlock, 64, 64, 2),
                _pid_make_layer(_PIDBasicBlock, 64, 64, 2),
                _pid_make_layer(_PIDBottleneck, 64, 64, 1),
            ]
        )
        self.compression_1 = _PIDConvModule(128, 64, 1, activation=False)
        self.compression_2 = _PIDConvModule(256, 64, 1, activation=False)
        self.pag_1 = _PagFM(64, 32)
        self.pag_2 = _PagFM(64, 32)
        self.d_branch_layers = nn.ModuleList(
            [
                _pid_make_single_layer(_PIDBasicBlock, 64, 32),
                _pid_make_layer(_PIDBottleneck, 32, 32, 1),
                _pid_make_layer(_PIDBottleneck, 64, 64, 1),
            ]
        )
        self.diff_1 = _PIDConvModule(
            128, 32, 3, padding=1, activation=False
        )
        self.diff_2 = _PIDConvModule(
            256, 64, 3, padding=1, activation=False
        )
        self.spp = _PAPPM(512, 96, 128)
        self.dfm = _LightBag(128, 128)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        height, width = inputs.shape[-2] // 8, inputs.shape[-1] // 8
        x = self.stem[0](inputs)
        x = self.stem[1](x)
        stem_s4 = self.stem[2](x)
        x = self.stem[5](self.stem[4](self.stem[3](stem_s4)))

        x_i = self.relu(self.i_branch_layers[0](x))
        x_p = self.p_branch_layers[0](x)
        x_d = self.d_branch_layers[0](x)
        x_p = self.pag_1(x_p, self.compression_1(x_i))
        x_d = x_d + F.interpolate(
            self.diff_1(x_i),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        proportional_s8 = x_p

        x_i = self.relu(self.i_branch_layers[1](x_i))
        x_p = self.p_branch_layers[1](self.relu(x_p))
        x_d = self.d_branch_layers[1](self.relu(x_d))
        x_p = self.pag_2(x_p, self.compression_2(x_i))
        x_d = x_d + F.interpolate(
            self.diff_2(x_i),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        derivative_s8 = x_d

        x_i = self.i_branch_layers[2](x_i)
        x_p = self.p_branch_layers[2](self.relu(x_p))
        x_d = self.d_branch_layers[2](self.relu(x_d))
        context = F.interpolate(
            self.spp(x_i),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        fused_s8 = self.dfm(x_p, context, x_d)
        return {
            "stem_s4": stem_s4,
            "proportional_s8": proportional_s8,
            "derivative_s8": derivative_s8,
            "fused_s8": fused_s8,
        }


class _BasePIDHead(nn.Module):
    def __init__(self, in_channels: int, channels: int) -> None:
        super().__init__()
        self.conv = _PIDConvModule(
            in_channels,
            channels,
            3,
            padding=1,
            order=("bn", "act", "conv"),
        )
        self.norm = nn.BatchNorm2d(channels, eps=1e-5, momentum=0.1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(inputs)))


class _RetainedPIDHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.i_head = _BasePIDHead(128, 128)
        self.p_head = _BasePIDHead(64, 128)
        self.d_head = _BasePIDHead(64, 32)
        self.d_cls_seg = nn.Conv2d(32, 1, 1)


class _PIDNetDenseLTRB(nn.Module):
    """Inference-only PIDNet-S plus the trained boundary-gated box head."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = _PIDNetSBackbone()
        self.decode_head = _RetainedPIDHead()
        self.stem_projection = _ConvNormAct(32, 128, 1)
        self.foreground_head = nn.Sequential(
            _ConvNormAct(128, 64, 3), nn.Conv2d(64, 1, 1)
        )
        self.detection_stem = nn.Sequential(
            _ConvNormAct(258, 96, 3), _ConvNormAct(96, 96, 3)
        )
        self.quality_head = nn.Conv2d(96, 1, 1)
        self.ltrb_head = nn.Conv2d(96, 4, 1)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(inputs)
        output_size = features["stem_s4"].shape[-2:]
        detail = self.decode_head.p_head(features["proportional_s8"])
        context = self.decode_head.i_head(features["fused_s8"])
        derivative = self.decode_head.d_head(features["derivative_s8"])
        boundary_s8 = self.decode_head.d_cls_seg(derivative)
        gate = torch.sigmoid(boundary_s8)
        gated_s8 = gate * detail + (1.0 - gate) * context
        ungated_s8 = 0.5 * (detail + context)
        gated = F.interpolate(
            gated_s8, size=output_size, mode="bilinear", align_corners=False
        )
        ungated = F.interpolate(
            ungated_s8, size=output_size, mode="bilinear", align_corners=False
        )
        ungated = ungated + self.stem_projection(features["stem_s4"])
        boundary = F.interpolate(
            boundary_s8, size=output_size, mode="bilinear", align_corners=False
        )
        foreground = self.foreground_head(gated)
        detector = self.detection_stem(
            torch.cat(
                (
                    ungated,
                    gated,
                    torch.sigmoid(foreground),
                    torch.sigmoid(boundary),
                ),
                dim=1,
            )
        )
        return {
            "quality": self.quality_head(detector),
            "ltrb_raw": self.ltrb_head(detector),
            "foreground": foreground,
            "boundary": boundary,
        }


class _PIDNetAdapter:
    """Load the trained PIDNet EMA checkpoint and return neuron centroids."""

    def __init__(
        self,
        model_path: str | Path,
        device: Optional[str] = None,
        conf: float = 0.34,
        pre_nms_topk: int = 300,
        topk: int = 100,
        nms_iou: float = 0.70,
    ) -> None:
        self.device = torch.device(
            device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.conf = float(conf)
        self.pre_nms_topk = int(pre_nms_topk)
        self.topk = int(topk)
        self.nms_iou = float(nms_iou)

        checkpoint = torch.load(
            Path(model_path), map_location="cpu", weights_only=True
        )
        if not isinstance(checkpoint, dict) or not isinstance(
            checkpoint.get("ema"), dict
        ):
            raise ValueError("PIDNet checkpoint does not contain an EMA state dictionary")
        self.model = _PIDNetDenseLTRB()
        self.model.load_state_dict(checkpoint["ema"], strict=True)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(
        self, img: np.ndarray, max_outputs: Optional[int] = None, *, include_boxes: bool = False
    ) -> List[Tuple[int, int, float] | Tuple[int, int, float, Tuple[float, float, float, float]]]:
        tensor, meta = _SegFormerAdapter._prepare(img)
        tensor = tensor.unsqueeze(0).to(self.device)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            outputs = self.model(tensor)

        quality = outputs["quality"].float().sigmoid()
        quality *= quality.eq(F.max_pool2d(quality, 3, stride=1, padding=1))
        ltrb = F.softplus(outputs["ltrb_raw"].float()) + 1e-3
        height, width = quality.shape[-2:]
        count = min(self.pre_nms_topk, height * width)
        scores, flat = torch.topk(quality[0, 0].flatten(), count)
        y = torch.div(flat, width, rounding_mode="floor")
        x = flat.remainder(width)
        distances = ltrb[0, :, y, x] * STRIDE
        center_x = (x.float() + 0.5) * STRIDE
        center_y = (y.float() + 0.5) * STRIDE
        boxes = torch.stack(
            (
                center_x - distances[0],
                center_y - distances[1],
                center_x + distances[2],
                center_y + distances[3],
            ),
            dim=1,
        )
        boxes.clamp_(0.0, float(INPUT_SIZE))
        boxes[:, (0, 2)] = (
            boxes[:, (0, 2)] - float(meta["left"])
        ) / float(meta["scale_x"])
        boxes[:, (1, 3)] = (
            boxes[:, (1, 3)] - float(meta["top"])
        ) / float(meta["scale_y"])
        boxes[:, (0, 2)].clamp_(0.0, float(meta["width"]))
        boxes[:, (1, 3)].clamp_(0.0, float(meta["height"]))

        keep = nms(boxes, scores, self.nms_iou)[: self.topk]
        keep = keep[scores[keep] >= self.conf]
        if max_outputs is not None:
            keep = keep[: max(0, int(max_outputs))]
        centers = (boxes[keep, :2] + boxes[keep, 2:]) * 0.5

        detections = []
        selected_boxes = boxes[keep].tolist() if include_boxes else None
        for index, (center, score) in enumerate(zip(centers.tolist(), scores[keep].tolist())):
            x_pixel = int(round(center[0]))
            y_pixel = int(round(center[1]))
            if (
                0 <= x_pixel < int(meta["width"])
                and 0 <= y_pixel < int(meta["height"])
            ):
                detection = (x_pixel, y_pixel, float(score))
                if include_boxes:
                    detection += (tuple(selected_boxes[index]),)
                detections.append(detection)
        return detections


class CDModelFactory:
    """Construct a supported inference backend without exposing model internals."""

    SUPPORTED_MODELS = ("segformer", "pidnet")

    @classmethod
    def create(
        cls,
        model_type: str,
        model_path: str | Path,
        device: Optional[str] = None,
        conf: float = 0.20,
        encoder_config_path: Optional[str | Path] = None,
    ) -> _SegFormerAdapter | _PIDNetAdapter:
        normalized = str(model_type).strip().lower()
        if normalized == "segformer":
            if encoder_config_path is None:
                raise ValueError("SegFormer requires encoder_config_path")
            return _SegFormerAdapter(
                model_path=model_path,
                encoder_config_path=encoder_config_path,
                device=device,
                conf=conf,
            )
        if normalized == "pidnet":
            return _PIDNetAdapter(
                model_path=model_path,
                device=device,
                conf=conf,
            )
        supported = ", ".join(cls.SUPPORTED_MODELS)
        raise ValueError(
            f"Unsupported model_type {model_type!r}; expected one of: {supported}"
        )
