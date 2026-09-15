import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
except ImportError:  # matplotlib is optional
    plt = None

try:
    from .cellModel.lightglue_backend_transformers import TransformersLightGlueBackend
except ImportError:  # pragma: no cover - allow running as a script
    from cellModel.lightglue_backend_transformers import TransformersLightGlueBackend

_FEATURE_MODEL_MAP = {
    "superpoint": "ETH-CVG/lightglue_superpoint",
    "disk": "ETH-CVG/lightglue_disk",
}

PathLike = Union[str, Path]


class PointMatcher:
    """High-level interface to measure LightGlue matching latency."""

    def __init__(
        self,
        *,
        features: str = "superpoint",
        device: Optional[str] = None,
        extractor_conf: Optional[Dict] = None,
        matcher_conf: Optional[Dict] = None,
    ) -> None:
        self.features = features.lower()
        if self.features not in _FEATURE_MODEL_MAP:
            supported = ", ".join(sorted(_FEATURE_MODEL_MAP))
            raise ValueError(
                f"Unsupported features '{features}'. Transformers LightGlue supports: {supported}."
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        extractor_kwargs = dict(extractor_conf or {})
        if "max_num_keypoints" not in extractor_kwargs and self.features == "superpoint":
            extractor_kwargs["max_num_keypoints"] = 2048

        matcher_kwargs = dict(matcher_conf or {})
        model_id = _FEATURE_MODEL_MAP[self.features]
        self.backend = TransformersLightGlueBackend(
            model_id=model_id,
            device=self.device,
            extractor_conf=extractor_kwargs,
            matcher_conf=matcher_kwargs,
        )

    def match(
        self,
        image0: Union[PathLike, torch.Tensor],
        image1: Union[PathLike, torch.Tensor],
        *,
        load_conf: Optional[Dict] = None,
        include_overlay: bool = True,
        **preprocess,
    ) -> Dict[str, object]:
        """Match two images and report latency.

        Args:
            image0/image1: Either file paths or pre-loaded ``torch.Tensor`` images.
            load_conf: Extra kwargs forwarded to the internal image loader when the
                inputs are file paths. Example: ``{\"resize\": 1024}``.
            include_overlay: Render the aligned image overlay; otherwise return None.
            **preprocess: Additional preprocessing configuration for the extractor.

        Returns:
            Dict containing ``feats0``, ``feats1``, ``matches`` and ``latency``.
        """
        image0_tensor = self._prepare_image(image0, load_conf)
        image1_tensor = self._prepare_image(image1, load_conf)

        start = time.perf_counter()
        feats0, feats1, matches = self.backend.match_pair(image0_tensor, image1_tensor)
        latency = time.perf_counter() - start

        keypoints_pair = self._get_matched_keypoints(feats0, feats1, matches)
        center_shift = self._calculate_shift(keypoints_pair, feats0, feats1)
        overlay = None
        if include_overlay:
            overlay = self._match_patch(
                image0_tensor.to(self.device), image1_tensor.to(self.device), keypoints_pair
            )

        return {
            "feats0": feats0,
            "feats1": feats1,
            "matches": matches,
            "latency": latency,
            "center_shift": center_shift,
            "overlay": overlay,
        }

    def _prepare_image(
        self, image: Union[PathLike, torch.Tensor], load_conf: Optional[Dict]
    ) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            tensor = image
        else:
            load_kwargs = dict(load_conf or {})
            tensor = self._load_image(str(image), **load_kwargs)

        if tensor.dtype == torch.uint8:
            tensor = tensor.to(dtype=torch.float32) / 255.0
        return tensor

    @staticmethod
    def _load_image(path: str, resize: Optional[Union[int, Tuple[int, int]]] = None, **kwargs: object) -> torch.Tensor:
        """Load an image from disk as a (C,H,W) float tensor in [0,1].

        This mirrors the previous LightGlue loader behavior (RGB + optional resize).
        """
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image at {path}.")
        img = img[..., ::-1]  # BGR -> RGB

        if resize is not None:
            fn = str(kwargs.get("fn", "max"))
            interp = str(kwargs.get("interp", "area"))
            img = PointMatcher._resize_image(img, resize, fn=fn, interp=interp)

        img = img.transpose((2, 0, 1))  # HWC -> CHW
        return torch.tensor(img / 255.0, dtype=torch.float32)

    @staticmethod
    def _resize_image(
        image: np.ndarray,
        size: Union[int, Tuple[int, int]],
        *,
        fn: str = "max",
        interp: str = "area",
    ) -> np.ndarray:
        h, w = image.shape[:2]
        fn_op = {"max": max, "min": min}.get(fn)
        if fn_op is None:
            raise ValueError(f"Unsupported resize fn '{fn}'.")

        if isinstance(size, int):
            scale = float(size) / float(fn_op(h, w))
            h_new = int(round(h * scale))
            w_new = int(round(w * scale))
        else:
            h_new, w_new = int(size[0]), int(size[1])

        mode = {
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "nearest": cv2.INTER_NEAREST,
            "area": cv2.INTER_AREA,
        }.get(interp)
        if mode is None:
            raise ValueError(f"Unsupported resize interp '{interp}'.")

        return cv2.resize(image, (w_new, h_new), interpolation=mode)

    def _get_matched_keypoints(
        self,
        feats0: Dict[str, torch.Tensor],
        feats1: Dict[str, torch.Tensor],
        matches: Dict[str, torch.Tensor],
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        match_indices = matches.get("matches")
        if match_indices is None or match_indices.ndim != 2 or match_indices.shape[1] != 2:
            return None
        if match_indices.numel() == 0:
            return None

        valid = (match_indices >= 0).all(dim=1)
        match_indices = match_indices[valid].to(device=feats0["keypoints"].device, dtype=torch.long)
        if match_indices.numel() == 0:
            return None

        points0 = feats0["keypoints"][match_indices[:, 0]]
        points1 = feats1["keypoints"][match_indices[:, 1]]
        if points0.shape[0] == 0 or points1.shape[0] == 0:
            return None

        return points0, points1

    def _calculate_shift(
        self,
        keypoints_pair: Optional[Tuple[torch.Tensor, torch.Tensor]],
        feats0: Dict[str, torch.Tensor],
        feats1: Dict[str, torch.Tensor],
    ) -> Optional[Dict[str, object]]:
        """Estimate translation between image centers from matched keypoints."""
        if keypoints_pair is None:
            return None

        points0, points1 = keypoints_pair
        translation = (points1 - points0).mean(dim=0)
        if not torch.isfinite(translation).all():
            return None

        size0 = feats0.get("image_size")
        size1 = feats1.get("image_size")
        if size0 is not None and size1 is not None:
            size0 = size0.to(translation)
            size1 = size1.to(translation)
            center0 = (size0 - 1.0) / 2.0
            center1 = (size1 - 1.0) / 2.0
        else:
            center0 = center1 = None

        result = {
            "center_dx": float(translation[0].item()),
            "center_dy": float(translation[1].item()),
            "num_matches": int(points0.shape[0]),
        }

        if center0 is not None and center1 is not None:
            aligned_center0 = center0 + translation
            displacement_vec = center1 - aligned_center0
            displacement = {
                "dx": float(displacement_vec[0].item()),
                "dy": float(displacement_vec[1].item()),
            }
            center0_tuple = tuple(map(float, center0.tolist()))
            center1_tuple = tuple(map(float, center1.tolist()))
            result.update(
                {
                    "center0": center0_tuple,
                    "center1": center1_tuple,
                    "center_displacement": displacement,
                }
            )

        return result

    def _match_patch(
        self,
        image0: torch.Tensor,
        image1: torch.Tensor,
        keypoints_pair: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Optional[torch.Tensor]:
        """Overlay image0 onto image1 by aligning matched keypoints via translation."""
        if keypoints_pair is None:
            return None

        points0, points1 = keypoints_pair
        translation = (points1 - points0).mean(dim=0)
        if not torch.isfinite(translation).all():
            return None

        if image0.shape[0] != image1.shape[0]:
            return None

        target_hw = image1.shape[-2], image1.shape[-1]
        aligned_image0 = self._apply_translation(image0, translation, target_hw=target_hw)
        overlay = 0.5 * aligned_image0 + 0.5 * image1.to(aligned_image0)
        return overlay.clamp(0.0, 1.0).detach().cpu()

    def _apply_translation(
        self,
        image: torch.Tensor,
        translation: torch.Tensor,
        *,
        target_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Translate `image` by `translation` (dx, dy) using bilinear sampling."""
        if image.dim() != 3:
            raise ValueError("Expected image tensor with shape (C, H, W).")

        _, src_h, src_w = image.shape
        tgt_h, tgt_w = target_hw if target_hw is not None else (src_h, src_w)

        if src_h <= 1 or src_w <= 1:
            return image.clone()

        device = image.device
        dtype = image.dtype

        # Create target pixel coordinate grid.
        ys = torch.arange(tgt_h, device=device, dtype=dtype)
        xs = torch.arange(tgt_w, device=device, dtype=dtype)
        if hasattr(torch, "meshgrid"):
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        else:
            grid_y, grid_x = torch.meshgrid(ys, xs)

        # Shift coordinates by translation to sample from source image.
        src_x = grid_x - translation[0]
        src_y = grid_y - translation[1]

        if src_w > 1:
            src_x_norm = (src_x / (src_w - 1)) * 2 - 1
        else:
            src_x_norm = src_x * 0
        if src_h > 1:
            src_y_norm = (src_y / (src_h - 1)) * 2 - 1
        else:
            src_y_norm = src_y * 0

        grid = torch.stack((src_x_norm, src_y_norm), dim=-1).unsqueeze(0)
        image_batch = image.unsqueeze(0)

        warped = F.grid_sample(
            image_batch,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return warped.squeeze(0)

if __name__ == "__main__":
    # IMAGE0_PATH = Path(r"C:\Users\sa-forest\Documents\GitHub\LightGlue\ex_data\cell_4.webp")
    # IMAGE1_PATH = Path(r"C:\Users\sa-forest\Documents\GitHub\LightGlue\ex_data\36712_1760473097.769556.webp")
    IMAGE0_PATH = Path(r"C:\Users\sa-forest\Documents\GitHub\LightGlue\ex_data\88602_1760469071.915733.webp")
    IMAGE1_PATH = Path(r"C:\Users\sa-forest\Documents\GitHub\LightGlue\ex_data\106826_1760469696.866317.webp")
    matcher = PointMatcher(features="superpoint")
    result = matcher.match(IMAGE0_PATH, IMAGE1_PATH)

    latency_ms = result["latency"] * 1e3
    print(f"Latency: {latency_ms:.2f} ms")
    shift = result.get("center_shift")
    if shift is None:
        print("Center shift: unavailable (no valid matches).")
    else:
        print(
            f"Center shift: dx={shift['center_dx']:.2f} px, dy={shift['center_dy']:.2f} px "
            f"(from {shift['num_matches']} matches)"
        )
        displacement = shift.get("center_displacement")
        if displacement is not None:
            print(
                f"Aligned center displacement: dx={displacement['dx']:.2f} px, "
                f"dy={displacement['dy']:.2f} px"
            )

    overlay = result.get("overlay")
    if overlay is None:
        print("Overlay: unavailable.")
    else:
        print(f"Overlay image shape: {tuple(overlay.shape)}")
        if plt is None:
            print("Matplotlib not installed; cannot display overlay.")
        else:
            overlay_np = overlay.permute(1, 2, 0).numpy()
            plt.figure("LightGlue Overlay")
            plt.imshow(overlay_np)
            plt.title("Overlay: image0 warped onto image1")
            plt.axis("off")
            plt.show()
