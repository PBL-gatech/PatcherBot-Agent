from __future__ import annotations

"""
Homography-based mask warping built on top of PointMatcher (LightGlue).

Usage (copy-paste friendly):
    warper = MaskWarper(features="superpoint")
    result = warper.warp(
        template_image,        # np.ndarray or torch.Tensor or path
        current_image,         # np.ndarray or torch.Tensor or path
        template_mask,         # np.ndarray (0/255 or bool)
        load_conf=None,        # optional LightGlue load config
        preprocess={},         # optional extractor preprocess kwargs
    )
    warped_mask = result["warped_mask"]
    warped_template = result["warped_template"]
    centroid_current = result["centroid_current"]
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch

try:
    from .PointMatcher import PointMatcher
except ImportError:  # pragma: no cover - allow running as script
    from PointMatcher import PointMatcher

ImageInput = Union[str, Path, np.ndarray, torch.Tensor]
MaskInput = np.ndarray


class MaskWarper:
    """Fit a homography from LightGlue matches and warp the template mask."""

    def __init__(
        self,
        *,
        matcher: Optional[PointMatcher] = None,
        features: str = "superpoint",
        device: Optional[str] = None,
        extractor_conf: Optional[Dict] = None,
        matcher_conf: Optional[Dict] = None,
    ) -> None:
        self._matcher = matcher or PointMatcher(
            features=features,
            device=device,
            extractor_conf=extractor_conf,
            matcher_conf=matcher_conf,
        )

    # ------------------------------------------------------------------ #
    def warp(
        self,
        template_image: ImageInput,
        current_image: ImageInput,
        template_mask: MaskInput,
        *,
        load_conf: Optional[Dict] = None,
        ransac_reproj_threshold: float = 3.0,
        max_iters: int = 10000,
        confidence: float = 0.999,
        **preprocess: object,
    ) -> Dict[str, object]:
        """
        Warp `template_mask` into the `current_image` using a homography fit.

        Returns a dict with:
            - warped_mask: np.ndarray uint8
            - warped_template: np.ndarray uint8 (template warped to current frame)
            - homography: 3x3 np.ndarray float32
            - inliers: np.ndarray bool mask of matches used
            - centroid_template: np.ndarray [cx, cy] or None
            - centroid_current: np.ndarray [cx, cy] or None (template centroid transformed)
            - matches_template: np.ndarray Nx2 float32 matched points in template
            - matches_current: np.ndarray Nx2 float32 matched points in current
            - overlay: np.ndarray uint8 visualization
        """
        tmpl_img = template_image
        cur_img = current_image
        tmask = np.asarray(template_mask)
        if tmask.ndim == 3:
            tmask = tmask[0]
        tmask = tmask.astype(np.uint8)

        res = self._matcher.match(
            self._to_lightglue_tensor(tmpl_img),
            self._to_lightglue_tensor(cur_img),
            load_conf=load_conf,
            **preprocess,
        )
        mkpts0, mkpts1 = self._matched_points_from_result(res)
        if mkpts0 is None or mkpts1 is None:
            raise RuntimeError("Not enough valid matches to estimate a homography.")

        filtered0, filtered1 = self._filter_points_inside_mask(mkpts0, mkpts1, tmask)
        if filtered0.shape[0] >= 8:
            mkpts0, mkpts1 = filtered0, filtered1

        H, inliers = self._fit_homography(
            mkpts0,
            mkpts1,
            ransac_reproj_threshold,
            max_iters,
            confidence,
        )
        warped_template, warped_mask = self._warp_images(
            tmpl_img,
            tmask,
            cur_img,
            H,
        )
        centroid_template = self._compute_centroid(tmask)
        centroid_current = None
        if centroid_template is not None:
            centroid_current = self._transform_point(centroid_template, H)

        overlay = self._build_overlay(cur_img, warped_template, warped_mask, centroid_current)

        return {
            "warped_mask": warped_mask,
            "warped_template": warped_template,
            "homography": H,
            "inliers": inliers,
            "centroid_template": centroid_template,
            "centroid_current": centroid_current,
            "matches_template": mkpts0,
            "matches_current": mkpts1,
            "overlay": overlay,
        }

    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_lightglue_tensor(image: ImageInput) -> torch.Tensor:
        """Convert input to (3,H,W) float32 tensor in [0,1]."""
        if isinstance(image, torch.Tensor):
            tensor = image
            if tensor.dim() == 3 and tensor.shape[0] in (1, 3):
                return tensor
            if tensor.dim() == 3 and tensor.shape[-1] in (1, 3):
                return tensor.permute(2, 0, 1)
            raise ValueError("Unsupported tensor shape for LightGlue input.")
        if isinstance(image, (str, Path)):
            # PointMatcher.load_image handles path inputs internally.
            return torch.from_numpy(np.array([]))  # placeholder; PointMatcher will load paths
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Unsupported image shape for LightGlue input: {arr.shape}")
        arr_f = arr.astype(np.float32)
        if arr_f.max() > 1.0:
            arr_f /= 255.0
        arr_f = np.clip(arr_f, 0.0, 1.0)
        return torch.from_numpy(arr_f.transpose(2, 0, 1))

    @staticmethod
    def _matched_points_from_result(result: Dict[str, object]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract matched keypoints (template->current) from PointMatcher.match()."""
        feats0 = result.get("feats0")
        feats1 = result.get("feats1")
        matches = result.get("matches")
        if not isinstance(feats0, dict) or not isinstance(feats1, dict) or not isinstance(matches, dict):
            return None, None
        mi = matches.get("matches", None)
        if mi is None or mi.ndim != 2 or mi.shape[1] != 2 or mi.numel() == 0:
            return None, None
        valid = (mi >= 0).all(dim=1)
        if not valid.any().item():
            return None, None
        mi = mi[valid].to(device=feats0["keypoints"].device, dtype=torch.long)
        p0 = feats0["keypoints"][mi[:, 0]]  # (N,2)
        p1 = feats1["keypoints"][mi[:, 1]]  # (N,2)
        if p0.shape[0] < 4:
            return None, None
        mkpts0 = p0.detach().cpu().numpy().astype(np.float32)
        mkpts1 = p1.detach().cpu().numpy().astype(np.float32)
        return mkpts0, mkpts1

    @staticmethod
    def _filter_points_inside_mask(
        pts0: np.ndarray,
        pts1: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Keep only matches whose template points fall inside mask."""
        mask_2d = mask if mask.ndim == 2 else mask.squeeze()
        x = np.rint(pts0[:, 0]).astype(int)
        y = np.rint(pts0[:, 1]).astype(int)
        ok = (x >= 0) & (x < mask_2d.shape[1]) & (y >= 0) & (y < mask_2d.shape[0]) & (mask_2d[y, x] > 0)
        return pts0[ok], pts1[ok]

    @staticmethod
    def _fit_homography(
        pts0: np.ndarray,
        pts1: np.ndarray,
        ransac_reproj_threshold: float,
        max_iters: int,
        confidence: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if pts0.shape[0] < 4:
            raise RuntimeError("At least 4 matches are required to fit a homography.")
        method = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC
        H, inliers = cv2.findHomography(
            pts0,
            pts1,
            method,
            ransac_reproj_threshold,
            maxIters=int(max_iters),
            confidence=float(confidence),
        )
        if H is None:
            raise RuntimeError("Homography fit failed.")
        if inliers is None:
            inliers = np.ones((pts0.shape[0], 1), dtype=np.uint8)
        return H.astype(np.float32), inliers.astype(bool).reshape(-1)

    @staticmethod
    def _warp_images(
        template_image: ImageInput,
        template_mask: np.ndarray,
        current_image: ImageInput,
        H: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Warp template image and mask into current frame using homography."""
        tmpl = MaskWarper._ensure_bgr(template_image)
        curr = MaskWarper._ensure_bgr(current_image)
        Hc, Wc = curr.shape[:2]
        tmpl_warp = cv2.warpPerspective(tmpl, H, (Wc, Hc), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask_warp = cv2.warpPerspective(template_mask, H, (Wc, Hc), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask_warp = mask_warp.astype(np.uint8)
        return tmpl_warp, mask_warp

    @staticmethod
    def _compute_centroid(mask: np.ndarray) -> Optional[np.ndarray]:
        """Compute centroid of a binary mask."""
        mask_2d = mask if mask.ndim == 2 else mask.squeeze()
        m = (mask_2d > 0).astype(np.uint8)
        M = cv2.moments(m)
        if M["m00"] == 0:
            return None
        return np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]], dtype=np.float32)

    @staticmethod
    def _transform_point(point: np.ndarray, H: np.ndarray) -> Optional[np.ndarray]:
        """Apply homography to a single (cx, cy) point."""
        if point is None:
            return None
        pt = np.asarray(point, dtype=np.float32).reshape(1, 1, 2)
        out = cv2.perspectiveTransform(pt, H)
        return out[0, 0]

    @staticmethod
    def _build_overlay(
        current_image: ImageInput,
        warped_template: np.ndarray,
        warped_mask: np.ndarray,
        centroid_current: Optional[np.ndarray],
    ) -> np.ndarray:
        """Blend warped template onto current image and mark mask + centroid."""
        cur_bgr = MaskWarper._ensure_bgr(current_image).astype(np.float32)
        tmpl_bgr = warped_template.astype(np.float32)
        overlay = np.clip(0.5 * cur_bgr + 0.5 * tmpl_bgr, 0, 255).astype(np.uint8)
        mask_bool = warped_mask > 0
        overlay[mask_bool] = (0, 255, 0)
        if centroid_current is not None:
            cx, cy = int(round(centroid_current[0])), int(round(centroid_current[1]))
            cv2.drawMarker(
                overlay,
                (cx, cy),
                (0, 255, 255),
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=18,
                thickness=2,
            )
        return overlay

    # ------------------------------------------------------------------ #
    @staticmethod
    def _ensure_bgr(image: ImageInput) -> np.ndarray:
        """Return a BGR uint8 image."""
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Could not load image from {image}")
            return img
        if isinstance(image, torch.Tensor):
            arr = image.detach().cpu()
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = arr.permute(1, 2, 0)
            elif arr.ndim == 3 and arr.shape[2] in (1, 3):
                pass
            else:
                raise ValueError("Unsupported tensor shape for BGR conversion.")
            arr_np = arr.numpy()
        else:
            arr_np = np.asarray(image)
        if arr_np.ndim == 2:
            bgr = cv2.cvtColor(arr_np, cv2.COLOR_GRAY2BGR)
        elif arr_np.ndim == 3 and arr_np.shape[2] == 1:
            bgr = cv2.cvtColor(arr_np, cv2.COLOR_GRAY2BGR)
        elif arr_np.ndim == 3 and arr_np.shape[2] == 3:
            bgr = arr_np
        else:
            raise ValueError(f"Unsupported image shape: {arr_np.shape}")
        if bgr.dtype != np.uint8:
            bgr_f = bgr.astype(np.float32)
            if bgr_f.max() > 1.0:
                bgr_f /= bgr_f.max() if bgr_f.max() > 0 else 255.0
            bgr = np.clip(bgr_f * 255.0, 0, 255).astype(np.uint8)
        return bgr


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Warp template mask to current image using LightGlue homography.")
    parser.add_argument("template", type=str, help="Path to template image")
    parser.add_argument("current", type=str, help="Path to current image")
    parser.add_argument("template_mask", type=str, help="Path to template mask (0/255)")
    parser.add_argument("--features", type=str, default="superpoint", help="LightGlue feature extractor")
    parser.add_argument("--out_overlay", type=str, default="overlay_homography.png", help="Output overlay path")
    parser.add_argument("--out_mask", type=str, default="warped_mask.png", help="Output warped mask path")
    args = parser.parse_args()

    warper = MaskWarper(features=args.features)
    result = warper.warp(
        args.template,
        args.current,
        cv2.imread(args.template_mask, cv2.IMREAD_GRAYSCALE),
    )
    cv2.imwrite(args.out_overlay, result["overlay"])
    cv2.imwrite(args.out_mask, result["warped_mask"])
    print(f"Wrote {args.out_overlay} and {args.out_mask}")
