import cv2
import numpy as np
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Deque, Dict, List, Optional, Tuple

@dataclass
class TrackingResult:
    """
    Output of one per-frame update.

    flow_px: Dense optical flow in pixels (H, W, 2), mapping prev -> curr.
    points_px: Tracked point locations in pixels (N, 2), if points were provided; else None.
    disp_px: Per-point displacement vectors in pixels (N, 2), if points were provided; else None.
    scores: Quality / diagnostic scores for this update.
    meta: Any extra metadata you want to pass through (frame_no, timestamp, etc).
    """
    flow_px: Optional[np.ndarray]
    points_px: Optional[np.ndarray]
    disp_px: Optional[np.ndarray]
    scores: Dict[str, Any]
    meta: Dict[str, Any]


class PointTracker(ABC):
    """
    Abstract per-frame (online) point / tissue motion tracker.

    Design goals:
      - Swap flow models easily (Farneback now, deep model later).
      - Single-call per incoming frame: update(image, points=...)
      - Optionally persist a sparse point set via set_points(...).
      - Optional ROI and mask to ignore pipette / background.

    Coordinate conventions:
      - Pixel coordinates: (x right, y down).
      - Flow returned is (dx, dy) in pixels from prev->curr.
    """

    def __init__(
        self,
        roi: Optional[Tuple[int, int, int, int]] = None,  # (x, y, w, h)
        mask: Optional[np.ndarray] = None,                # same size as incoming frame (or ROI frame)
        normalize: bool = True,
        blur_ksize: int = 3,
        expect_gray: bool = True,
        compute_scores_enabled: bool = True,
        debug_timing: bool = False,
        debug_interval: int = 30,
        debug_label: str = "PointTracker",
    ):
        self.roi = roi
        self.mask = mask
        self.normalize = normalize
        self.blur_ksize = blur_ksize
        self.expect_gray = expect_gray
        self.compute_scores_enabled = compute_scores_enabled
        self.debug_timing = bool(debug_timing)
        self.debug_interval = int(max(1, debug_interval))
        self.debug_label = str(debug_label)

        self.prev_proc: Optional[np.ndarray] = None
        self.prev_raw_shape: Optional[Tuple[int, int]] = None  # (H, W) of processed frame
        self.frame_index: int = 0
        self._persistent_points_px: Optional[np.ndarray] = None

    # ----------------------------
    # Public API
    # ----------------------------
    def reset(self) -> None:
        """Forget previous frame state."""
        self.prev_proc = None
        self.prev_raw_shape = None
        self.frame_index = 0

    def set_roi(self, roi: Optional[Tuple[int, int, int, int]]) -> None:
        """Set ROI as (x, y, w, h). If None, use full frame."""
        self.roi = roi
        # Mask might no longer match; user can re-set mask explicitly.

    def set_mask(self, mask: Optional[np.ndarray]) -> None:
        """
        Set a mask (uint8 or bool) where non-zero/True indicates pixels to USE.
        Mask should match the processed image size (full frame or ROI).
        """
        self.mask = mask

    def set_points(self, points: Optional[Any]) -> None:
        """
        Set persistent full-frame points to track on every update when update(points=None).

        Accepts:
          - Nx2 array-like
          - single (x, y) point
          - Python set/list/tuple of points
        """
        if points is None:
            self._persistent_points_px = None
            return
        self._persistent_points_px = self._normalize_points_input(points).copy()

    def clear_points(self) -> None:
        """Clear persistent points previously set with set_points()."""
        self._persistent_points_px = None

    def update(
        self,
        image: np.ndarray,
        points: Optional[Any] = None,  # (N,2) points in PIXELS, full-frame coordinates
        meta: Optional[Dict[str, Any]] = None,
    ) -> TrackingResult:
        """
        Process one frame online.

        image: current frame (grayscale or BGR).
        points: optional points to track (in full-frame pixel coordinates).
                Accepts Nx2 array-like, single (x, y), or set/list/tuple of (x, y).
                If None, tracker uses points previously set with set_points(...).
                If ROI is set, points are internally shifted into ROI coordinates.

        Returns a TrackingResult. On the very first frame, no flow can be computed,
        so flow_* and point outputs will be None, but you still get scores/meta.
        """
        t_update_start = time.perf_counter()
        if meta is None:
            meta = {}

        t_pre = time.perf_counter()
        curr_proc, point_shift = self._preprocess_with_roi(image)
        preprocess_ms = (time.perf_counter() - t_pre) * 1e3

        # First frame: just prime the tracker
        if self.prev_proc is None:
            self.prev_proc = curr_proc
            self.prev_raw_shape = curr_proc.shape[:2]
            self.frame_index += 1
            total_ms = (time.perf_counter() - t_update_start) * 1e3
            if self.debug_timing and (
                self.frame_index <= 3 or (self.frame_index % self.debug_interval == 0)
            ):
                frame_token = meta.get("frame_no", self.frame_index)
                print(
                    f"[{self.debug_label}] frame={frame_token} "
                    f"preprocess={preprocess_ms:.2f}ms flow=0.00ms scores=0.00ms "
                    f"points=0.00ms total={total_ms:.2f}ms status=primed"
                )
            return TrackingResult(
                flow_px=None,
                points_px=None,
                disp_px=None,
                scores={"status": "primed", "frame_index": self.frame_index},
                meta=dict(meta),
            )

        # Estimate dense flow prev -> curr
        t_flow = time.perf_counter()
        flow_px, conf = self.estimate_flow(self.prev_proc, curr_proc, mask=self.mask)
        flow_ms = (time.perf_counter() - t_flow) * 1e3

        # Postprocess & score
        t_scores = time.perf_counter()
        if self.compute_scores_enabled:
            scores = self.compute_scores(self.prev_proc, curr_proc, flow_px, conf)
        else:
            scores = {"status": "ok"}
        scores_ms = (time.perf_counter() - t_scores) * 1e3

        # Track points if provided
        t_points = time.perf_counter()
        points_px_out = disp_px_out = None
        points_src = points
        use_persistent_points = False
        if points_src is None and self._persistent_points_px is not None:
            points_src = self._persistent_points_px
            use_persistent_points = True

        if points_src is not None:
            points = self._normalize_points_input(points_src)

            # Convert full-frame points -> ROI coords (if ROI exists)
            points_roi = points.copy()
            points_roi[:, 0] -= point_shift[0]
            points_roi[:, 1] -= point_shift[1]

            disp_px = self.sample_flow_at_points(flow_px, points_roi)  # (N,2)
            points_roi_new = points_roi + disp_px

            # Convert back ROI coords -> full-frame coords
            points_new = points_roi_new.copy()
            points_new[:, 0] += point_shift[0]
            points_new[:, 1] += point_shift[1]

            points_px_out = points_new
            disp_px_out = disp_px
            if use_persistent_points:
                # Keep persistent points aligned with tracked positions.
                self._persistent_points_px = points_new.copy()
        points_ms = (time.perf_counter() - t_points) * 1e3

        # Advance state
        self.prev_proc = curr_proc
        self.frame_index += 1
        scores["frame_index"] = self.frame_index
        total_ms = (time.perf_counter() - t_update_start) * 1e3

        if self.debug_timing and (
            self.frame_index <= 3 or (self.frame_index % self.debug_interval == 0)
        ):
            frame_token = meta.get("frame_no", self.frame_index)
            print(
                f"[{self.debug_label}] frame={frame_token} "
                f"preprocess={preprocess_ms:.2f}ms flow={flow_ms:.2f}ms "
                f"scores={scores_ms:.2f}ms points={points_ms:.2f}ms "
                f"total={total_ms:.2f}ms"
            )

        return TrackingResult(
            flow_px=flow_px,
            points_px=points_px_out,
            disp_px=disp_px_out,
            scores=scores,
            meta=dict(meta),
        )

    @staticmethod
    def _normalize_points_input(points: Any) -> np.ndarray:
        """
        Normalize point inputs to float32 Nx2.
        """
        if isinstance(points, set):
            points = list(points)

        pts = np.asarray(points, dtype=np.float32)
        if pts.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        if pts.ndim == 1:
            if pts.shape[0] != 2:
                raise ValueError("points must be shape (2,) or (N,2).")
            pts = pts.reshape(1, 2)
        elif pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("points must be shape (2,) or (N,2).")
        return pts.astype(np.float32, copy=False)

    # ----------------------------
    # Model interface (swap these)
    # ----------------------------
    @abstractmethod
    def estimate_flow(
        self,
        prev_proc: np.ndarray,
        curr_proc: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Return:
          flow_px: (H, W, 2) float32, dx/dy in pixels mapping prev->curr
          confidence: optional (H, W) float32 or None
        """
        raise NotImplementedError

    # ----------------------------
    # Default helpers (override if desired)
    # ----------------------------
    def preprocess(self, image_gray: np.ndarray) -> np.ndarray:
        """
        Default preprocessing for optical flow:
          - optional blur (reduce noise)
          - optional normalization to [0,255] uint8 or float32 in [0,1]
        Subclasses can override if needed (e.g. deep net expects specific scaling).
        """
        img = image_gray

        if self.blur_ksize and self.blur_ksize > 1:
            k = int(self.blur_ksize)
            if k % 2 == 0:
                k += 1
            img = cv2.GaussianBlur(img, (k, k), 0)

        # Farneback is fine with uint8 or float32; deep models often want float32.
        if self.normalize:
            # Normalize contrast robustly
            img_f = img.astype(np.float32)
            mn, mx = float(np.min(img_f)), float(np.max(img_f))
            if mx - mn > 1e-6:
                img_f = (img_f - mn) / (mx - mn)
            else:
                img_f = img_f * 0.0
            # Keep float32 [0,1] (works with Farneback too)
            return img_f.astype(np.float32)

        return img.astype(np.float32)

    def compute_scores(
        self,
        prev_proc: np.ndarray,
        curr_proc: np.ndarray,
        flow_px: np.ndarray,
        conf: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """
        Produce lightweight diagnostics each frame (customize as you like).
        """
        mag = np.sqrt(flow_px[..., 0] ** 2 + flow_px[..., 1] ** 2)
        scores: Dict[str, Any] = {
            "status": "ok",
            "flow_mag_mean_px": float(np.mean(mag)),
            "flow_mag_median_px": float(np.median(mag)),
            "flow_mag_p95_px": float(np.percentile(mag, 95)),
        }

        # Simple "texture" proxy: average gradient magnitude (helps detect low-texture failure)
        gx = cv2.Sobel(prev_proc, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(prev_proc, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx * gx + gy * gy)
        scores["texture_mean"] = float(np.mean(grad_mag))

        if conf is not None:
            scores["conf_mean"] = float(np.mean(conf))
            scores["conf_median"] = float(np.median(conf))

        return scores

    def sample_flow_at_points(self, flow_px: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
        """
        Bilinear sample dense flow at point locations.
        points_xy are in the SAME coordinate frame as flow (ROI if ROI is used).

        Returns per-point displacement (dx, dy) in pixels.
        """
        H, W = flow_px.shape[:2]
        pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)

        x = np.clip(pts[:, 0], 0, W - 1)
        y = np.clip(pts[:, 1], 0, H - 1)

        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, W - 1)
        y1 = np.clip(y0 + 1, 0, H - 1)

        wx = (x - x0).astype(np.float32)
        wy = (y - y0).astype(np.float32)

        f00 = flow_px[y0, x0]
        f10 = flow_px[y0, x1]
        f01 = flow_px[y1, x0]
        f11 = flow_px[y1, x1]

        top = f00 * (1 - wx)[:, None] + f10 * wx[:, None]
        bot = f01 * (1 - wx)[:, None] + f11 * wx[:, None]
        f = top * (1 - wy)[:, None] + bot * wy[:, None]
        return f.astype(np.float32)

    def _preprocess_with_roi(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
        """
        Returns:
          processed_image (ROI-cropped if roi is set)
          point_shift: (x_shift, y_shift) to convert full-frame points -> ROI points
        """
        # Convert to grayscale
        if image.ndim == 3:
            # assume BGR
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        if self.roi is not None:
            x, y, w, h = self.roi
            gray_roi = gray[int(y):int(y + h), int(x):int(x + w)]
            proc = self.preprocess(gray_roi)
            return proc, (float(x), float(y))

        proc = self.preprocess(gray)
        return proc, (0.0, 0.0)


class PointTracker1(PointTracker):
    """
    PointTracker1: Dense optical flow using OpenCV Farneback.

    Swap this class later with a deep model by implementing estimate_flow().
    """

    def __init__(
        self,
        *args,
        farneback_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Reasonable defaults (tune per dataset)
        defaults = dict(
            pyr_scale=0.5,
            levels=4,
            winsize=25,
            iterations=5,
            poly_n=7,
            poly_sigma=1.5,
            flags=0,
        )
        if farneback_params:
            defaults.update(farneback_params)
        self.fb = defaults

    def estimate_flow(
        self,
        prev_proc: np.ndarray,
        curr_proc: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Returns flow in pixels (dx, dy) for each pixel: prev -> curr.

        Notes:
          - Farneback expects single-channel images; float32 in [0,1] is fine.
          - No native confidence map; we return None (you can build one from texture).
          - If mask is provided, we zero out flow outside the mask for convenience.
        """
        prev = self._as_u8_for_farneback(prev_proc)
        curr = self._as_u8_for_farneback(curr_proc)

        flow = self._estimate_flow_core(prev, curr)

        if mask is not None:
            m = mask.astype(bool)
            if m.shape != flow.shape[:2]:
                raise ValueError("Mask shape must match processed image shape (ROI if ROI is set).")
            flow[~m] = 0.0

        return flow, None

    def _estimate_flow_core(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        return cv2.calcOpticalFlowFarneback(
            prev,
            curr,
            None,
            self.fb["pyr_scale"],
            self.fb["levels"],
            self.fb["winsize"],
            self.fb["iterations"],
            self.fb["poly_n"],
            self.fb["poly_sigma"],
            self.fb["flags"],
        ).astype(np.float32)

    @staticmethod
    def _as_u8_for_farneback(image: np.ndarray) -> np.ndarray:
        """
        Farneback is most stable with uint8 grayscale input.
        """
        if image.dtype == np.uint8:
            return image

        arr = np.asarray(image, dtype=np.float32)
        mn = float(np.min(arr))
        mx = float(np.max(arr))
        if mn >= -1e-3 and mx <= 1.0 + 1e-3:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0)
        return arr.astype(np.uint8)


class PointTracker2(PointTracker):
    """
    PointTracker2: Sparse point tracking with TAPNext++.

    This tracker follows the PointTracker update contract but returns sparse point
    displacements instead of dense optical flow.
    """

    def __init__(
        self,
        *args,
        tapnext_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        try:
            from patcherbot.deepLearning.trackModel.tapnet.SimpleTapNext import SimpleTapNext
        except ImportError:
            import importlib.util
            import sys

            module_path = Path(__file__).resolve().parent / "tapnet" / "SimpleTapNext.py"
            spec = importlib.util.spec_from_file_location("_patcherbot_simple_tapnext", module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load SimpleTapNext from {module_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            SimpleTapNext = module.SimpleTapNext

        self.tapnext = SimpleTapNext(**(tapnext_kwargs or {}))

    def reset(self) -> None:
        super().reset()
        if hasattr(self, "tapnext"):
            self.tapnext.reset()

    def set_points(self, points: Optional[Any]) -> None:
        super().set_points(points)
        if hasattr(self, "tapnext"):
            self.tapnext.set_points(self._persistent_points_px)

    def clear_points(self) -> None:
        super().clear_points()
        if hasattr(self, "tapnext"):
            self.tapnext.set_points(None)

    def update(
        self,
        image: np.ndarray,
        points: Optional[Any] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> TrackingResult:
        t_update_start = time.perf_counter()
        if meta is None:
            meta = {}

        frame_roi, point_shift = self._frame_with_roi(image)
        points_src = points
        use_persistent_points = False
        if points_src is None and self._persistent_points_px is not None:
            points_src = self._persistent_points_px
            use_persistent_points = True

        points_roi = None
        if points_src is not None:
            points_full = self._normalize_points_input(points_src)
            points_roi = points_full.copy()
            points_roi[:, 0] -= point_shift[0]
            points_roi[:, 1] -= point_shift[1]

        result = self.tapnext.track(frame_roi, points_xy=points_roi)

        points_px_out = None
        disp_px_out = None
        if result.points_px is not None:
            points_new = result.points_px.copy()
            points_new[:, 0] += point_shift[0]
            points_new[:, 1] += point_shift[1]
            points_px_out = points_new.astype(np.float32, copy=False)
            disp_px_out = result.disp_px
            if use_persistent_points:
                self._persistent_points_px = points_px_out.copy()

        self.prev_raw_shape = frame_roi.shape[:2]
        self.frame_index += 1

        scores: Dict[str, Any] = {
            "status": result.status,
            "frame_index": self.frame_index,
            "tapnext_frame_index": result.frame_index,
            "device": result.meta.get("device"),
            "inference_ms": result.meta.get("inference_ms"),
            "total_ms": (time.perf_counter() - t_update_start) * 1e3,
        }
        if result.visible is not None:
            scores["visible_count"] = int(np.count_nonzero(result.visible))
            scores["occluded_count"] = int(result.visible.shape[0] - np.count_nonzero(result.visible))
            scores["visible"] = result.visible
        if result.occluded is not None:
            scores["occluded"] = result.occluded
        if result.visible_logits is not None:
            scores["visible_logits"] = result.visible_logits
        if result.certainty is not None:
            scores["certainty"] = result.certainty

        if self.debug_timing and (
            self.frame_index <= 3 or (self.frame_index % self.debug_interval == 0)
        ):
            frame_token = meta.get("frame_no", self.frame_index)
            print(
                f"[{self.debug_label}] frame={frame_token} status={result.status} "
                f"inference={scores['inference_ms']:.2f}ms total={scores['total_ms']:.2f}ms"
            )

        return TrackingResult(
            flow_px=None,
            points_px=points_px_out,
            disp_px=disp_px_out,
            scores=scores,
            meta=dict(meta),
        )

    def estimate_flow(
        self,
        prev_proc: np.ndarray,
        curr_proc: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        raise NotImplementedError("PointTracker2 uses TAPNext++ sparse point tracking, not dense flow.")

    def _frame_with_roi(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
        if self.roi is None:
            return image, (0.0, 0.0)
        x, y, w, h = self.roi
        return image[int(y):int(y + h), int(x):int(x + w)], (float(x), float(y))


class FrameStreamer:
    """
    Stream saved frames like an online feed and visualize sparse tracked points with trails.
    """

    IMAGE_EXTENSIONS = (".webp", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    def __init__(
        self,
        directory: str,
        tracker: Optional[PointTracker] = None,
        *,
        max_points: int = 10,
        min_distance: int = 18,
        quality_level: float = 0.01,
        block_size: int = 7,
        trail_length: int = 30,
        trail_thickness: int = 3,
        grid_center_bias: float = 1.0,
        middle_quarter_fraction: Optional[float] = None,
        point_bounds_fraction: float = 0.125,
        use_random_sparse_points: bool = False,
        random_seed: Optional[int] = None,
        random_candidate_oversample: int = 8,
        use_feature_sparse_points: bool = False,
        window_name: str = "PointTracker Stream",
        playback_fps: float = 30.0,
        min_wait_ms: int = 1,
        max_wait_ms: int = 10,
        inference_scale: float = 0.35,
        auto_roi_middle_quarter: bool = True,
        debug_timing: bool = False,
        debug_interval: int = 30,
    ):
        self.directory = Path(directory)
        self.frame_dir = self._resolve_frame_dir(self.directory)
        self.frame_paths, self.frame_indices, self.timestamps = self._load_frame_paths(self.frame_dir)

        if tracker is None:
            tracker = PointTracker1(
                compute_scores_enabled=False,
                farneback_params={
                    "pyr_scale": 0.5,
                    "levels": 3,
                    "winsize": 15,
                    "iterations": 2,
                    "poly_n": 5,
                    "poly_sigma": 1.1,
                    "flags": 0,
                },
                debug_timing=debug_timing,
                debug_interval=debug_interval,
            )
        self.tracker = tracker

        self.max_points = int(max(1, max_points))
        self.min_distance = int(max(1, min_distance))
        self.quality_level = float(max(1e-6, quality_level))
        self.block_size = int(max(3, block_size))
        if self.block_size % 2 == 0:
            self.block_size += 1
        self.trail_length = int(max(1, trail_length))
        self.trail_thickness = int(max(1, trail_thickness))
        self.grid_center_bias = float(max(1.0, grid_center_bias))
        if middle_quarter_fraction is not None:
            # Legacy knob: 1.0 means default center bounds size.
            point_bounds_fraction = 0.125 * float(np.clip(middle_quarter_fraction, 0.0, 1.0))
        self.point_bounds_fraction = float(np.clip(point_bounds_fraction, 1e-3, 1.0))
        # Backward-compatible alias used by older call-sites.
        self.middle_quarter_fraction = self.point_bounds_fraction
        self.use_random_sparse_points = bool(use_random_sparse_points)
        self.use_feature_sparse_points = bool(use_feature_sparse_points)
        self.random_candidate_oversample = int(max(2, random_candidate_oversample))
        self._rng = np.random.default_rng(random_seed)
        self.window_name = window_name
        self.playback_fps = float(max(0.1, playback_fps))
        self.min_wait_ms = int(max(1, min_wait_ms))
        self.max_wait_ms = int(max(self.min_wait_ms, max_wait_ms))
        self.inference_scale = float(min(1.0, max(0.1, inference_scale)))
        self.auto_roi_middle_quarter = bool(auto_roi_middle_quarter)
        self.debug_timing = bool(debug_timing)
        self.debug_interval = int(max(1, debug_interval))

        self.points_px = np.empty((0, 2), dtype=np.float32)
        self.trails: List[Deque[Tuple[float, float]]] = []

    def stream(self) -> None:
        """
        Iterate frames in timestamp order, run tracker inference per frame, and display overlays.
        Keys:
          - q / esc: quit
          - space: pause/resume
        """
        self.tracker.reset()
        self.points_px = np.empty((0, 2), dtype=np.float32)
        self.trails = []
        debug_window_start = time.perf_counter()
        debug_window_frames = 0
        auto_set_roi = self.auto_roi_middle_quarter and (self.tracker.roi is None)

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        try:
            for i, frame_path in enumerate(self.frame_paths):
                t_frame_start = time.perf_counter()

                t_read = time.perf_counter()
                frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                read_ms = (time.perf_counter() - t_read) * 1e3
                if frame is None:
                    continue

                t_prep = time.perf_counter()
                inference_frame = self._prepare_inference_frame(frame)
                prep_ms = (time.perf_counter() - t_prep) * 1e3

                if auto_set_roi:
                    roi = self._middle_quarter_roi(inference_frame.shape[1], inference_frame.shape[0])
                    self.tracker.set_roi(roi)
                    auto_set_roi = False
                    if self.debug_timing:
                        print(f"[FrameStreamer] Auto ROI set to middle quarter: roi={roi}")

                if self.points_px.shape[0] == 0:
                    self._reseed_points(inference_frame)

                query_points = self.points_px if self.points_px.shape[0] > 0 else None
                t_track = time.perf_counter()
                result = self.tracker.update(
                    inference_frame,
                    points=query_points,
                    meta={
                        "frame_index": int(i),
                        "frame_no": int(self.frame_indices[i]),
                        "timestamp": float(self.timestamps[i]),
                        "image_path": str(frame_path),
                    },
                )
                track_ms = (time.perf_counter() - t_track) * 1e3

                if result.points_px is not None:
                    tracked_points = np.asarray(result.points_px, dtype=np.float32).reshape(-1, 2)
                else:
                    tracked_points = self.points_px.copy()

                t_post = time.perf_counter()
                tracked_points = self._keep_points_in_frame(
                    tracked_points, inference_frame.shape[1], inference_frame.shape[0]
                )
                self.points_px = tracked_points
                self._append_points_to_trails(self.points_px)
                self._top_up_points(inference_frame)
                post_ms = (time.perf_counter() - t_post) * 1e3

                t_draw = time.perf_counter()
                overlay = self._draw_overlay(
                    frame.copy(),
                    self.points_px,
                    scale=(1.0 / self.inference_scale),
                )
                cv2.putText(
                    overlay,
                    f"Points: {self.points_px.shape[0]}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(self.window_name, overlay)
                draw_ms = (time.perf_counter() - t_draw) * 1e3

                t_wait = time.perf_counter()
                key = cv2.waitKey(self._frame_delay_ms(i)) & 0xFF
                wait_ms = (time.perf_counter() - t_wait) * 1e3
                total_ms = (time.perf_counter() - t_frame_start) * 1e3

                if self.debug_timing:
                    debug_window_frames += 1
                    should_report = ((i + 1) <= 3) or ((i + 1) % self.debug_interval == 0)
                    if should_report:
                        window_elapsed = max(time.perf_counter() - debug_window_start, 1e-9)
                        window_fps = debug_window_frames / window_elapsed
                        print(
                            f"[FrameStreamer] frame={i + 1}/{len(self.frame_paths)} "
                            f"read={read_ms:.2f}ms prep={prep_ms:.2f}ms track={track_ms:.2f}ms "
                            f"post={post_ms:.2f}ms draw={draw_ms:.2f}ms wait={wait_ms:.2f}ms "
                            f"total={total_ms:.2f}ms pts={self.points_px.shape[0]} "
                            f"proc={inference_frame.shape[1]}x{inference_frame.shape[0]} "
                            f"fps_window={window_fps:.2f}"
                        )
                        debug_window_start = time.perf_counter()
                        debug_window_frames = 0

                if key in (27, ord("q")):
                    break
                if key == ord(" "):
                    if self._pause_loop():
                        break
        finally:
            cv2.destroyWindow(self.window_name)

    def _resolve_frame_dir(self, directory: Path) -> Path:
        if not directory.exists():
            raise FileNotFoundError(f"Directory does not exist: {directory}")
        if directory.is_file():
            raise FileNotFoundError(f"Expected a directory, got file: {directory}")

        camera_frames = directory / "camera_frames"
        if camera_frames.is_dir():
            return camera_frames

        has_images = any(
            path.suffix.lower() in self.IMAGE_EXTENSIONS
            for path in directory.iterdir()
            if path.is_file()
        )
        if has_images:
            return directory
        raise FileNotFoundError(
            f"No frames found in {directory}. Expected image files or a camera_frames subfolder."
        )

    def _load_frame_paths(self, frame_dir: Path) -> Tuple[List[Path], List[int], List[float]]:
        parsed: List[Tuple[float, int, Path]] = []
        for path in frame_dir.iterdir():
            if not path.is_file() or path.suffix.lower() not in self.IMAGE_EXTENSIONS:
                continue
            idx, ts = self._extract_image_data(path)
            parsed.append((ts, idx, path))

        if not parsed:
            raise FileNotFoundError(f"No frame images found in {frame_dir}")

        parsed.sort(key=lambda item: (item[0], item[1]))
        paths = [item[2] for item in parsed]
        indices = [item[1] for item in parsed]
        timestamps = [item[0] for item in parsed]
        return paths, indices, timestamps

    @staticmethod
    def _middle_quarter_roi(width: int, height: int) -> Tuple[int, int, int, int]:
        width = int(max(2, width))
        height = int(max(2, height))
        x = width // 4
        y = height // 4
        w = max(2, width // 2)
        h = max(2, height // 2)
        if x + w > width:
            w = width - x
        if y + h > height:
            h = height - y
        return x, y, w, h

    @staticmethod
    def _extract_image_data(path: Path) -> Tuple[int, float]:
        """
        Parse frame metadata from <index>_<timestamp>.<ext>. Falls back to lexical order keys on parse failure.
        """
        stem = path.stem
        if "_" not in stem:
            return 0, 0.0
        first, second = stem.split("_", 1)
        try:
            return int(first), float(second)
        except (TypeError, ValueError):
            return 0, 0.0

    def _prepare_inference_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.inference_scale >= 0.999:
            return frame
        h, w = frame.shape[:2]
        out_w = max(2, int(round(w * self.inference_scale)))
        out_h = max(2, int(round(h * self.inference_scale)))
        return cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

    def _gray(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _center_bias_axis(values_01: np.ndarray, power: float) -> np.ndarray:
        centered = values_01 * 2.0 - 1.0
        warped = np.sign(centered) * (np.abs(centered) ** power)
        return (warped + 1.0) * 0.5

    def _grid_in_box(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        n_points: int,
        *,
        bias_power: float,
    ) -> np.ndarray:
        if n_points <= 0:
            return np.empty((0, 2), dtype=np.float32)

        w = float(max(1.0, xmax - xmin))
        h = float(max(1.0, ymax - ymin))
        aspect = w / max(1.0, h)

        cols = int(np.ceil(np.sqrt(float(n_points) * aspect)))
        cols = max(1, cols)
        rows = int(np.ceil(float(n_points) / float(cols)))
        rows = max(1, rows)

        u = np.linspace(0.0, 1.0, cols, dtype=np.float32)
        v = np.linspace(0.0, 1.0, rows, dtype=np.float32)
        if bias_power > 1.0:
            u = self._center_bias_axis(u, bias_power)
            v = self._center_bias_axis(v, bias_power)

        x = float(xmin) + (float(xmax) - float(xmin)) * u
        y = float(ymin) + (float(ymax) - float(ymin)) * v
        gx, gy = np.meshgrid(x, y)
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
        if pts.shape[0] <= int(n_points):
            return pts

        sample_idx = np.linspace(0, pts.shape[0] - 1, int(n_points), dtype=np.int32)
        return pts[sample_idx]

    def _middle_quarter_bounds(self, width: int, height: int) -> Tuple[float, float, float, float]:
        width = int(max(2, width))
        height = int(max(2, height))
        frac = float(np.clip(self.point_bounds_fraction, 1e-3, 1.0))

        if self.tracker.roi is not None:
            x, y, w, h = self.tracker.roi
            x0 = float(np.clip(x, 0, width - 1))
            y0 = float(np.clip(y, 0, height - 1))
            x1 = float(np.clip(x + max(1, w) - 1, 0, width - 1))
            y1 = float(np.clip(y + max(1, h) - 1, 0, height - 1))
            center_x = 0.5 * (x0 + x1)
            center_y = 0.5 * (y0 + y1)
            half_w = 0.5 * frac * max(1.0, x1 - x0)
            half_h = 0.5 * frac * max(1.0, y1 - y0)
            xmin = max(0.0, center_x - half_w)
            xmax = min(float(width - 1), center_x + half_w)
            ymin = max(0.0, center_y - half_h)
            ymax = min(float(height - 1), center_y + half_h)
            return (xmin, xmax, ymin, ymax)

        max_x = float(width - 1)
        max_y = float(height - 1)
        # Centered region spanning `point_bounds_fraction` of width/height.
        half_w = 0.5 * frac
        half_h = 0.5 * frac
        xmin = (0.5 - half_w) * max_x
        xmax = (0.5 + half_w) * max_x
        ymin = (0.5 - half_h) * max_y
        ymax = (0.5 + half_h) * max_y
        return (xmin, xmax, ymin, ymax)

    def _make_grid_points(self, width: int, height: int, n_points: int) -> np.ndarray:
        if n_points <= 0:
            return np.empty((0, 2), dtype=np.float32)

        mid_xmin, mid_xmax, mid_ymin, mid_ymax = self._middle_quarter_bounds(width, height)
        points = self._grid_in_box(
            mid_xmin,
            mid_xmax,
            mid_ymin,
            mid_ymax,
            n_points,
            bias_power=self.grid_center_bias,
        )
        return points.astype(np.float32)

    def _make_random_sparse_points(
        self,
        width: int,
        height: int,
        n_points: int,
        existing: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if n_points <= 0:
            return np.empty((0, 2), dtype=np.float32)

        xmin, xmax, ymin, ymax = self._middle_quarter_bounds(width, height)
        refs = np.empty((0, 2), dtype=np.float32)
        if existing is not None and np.size(existing) > 0:
            refs = np.asarray(existing, dtype=np.float32).reshape(-1, 2)

        min_allowed2 = (0.5 * float(self.min_distance)) ** 2
        selected: List[np.ndarray] = []
        attempts = 0
        max_attempts = int(max(64, n_points * self.random_candidate_oversample * 10))

        while len(selected) < n_points and attempts < max_attempts:
            batch = int(max(8, (n_points - len(selected)) * self.random_candidate_oversample))
            xs = self._rng.uniform(xmin, xmax, size=batch).astype(np.float32)
            ys = self._rng.uniform(ymin, ymax, size=batch).astype(np.float32)
            candidates = np.stack((xs, ys), axis=1)

            for point in candidates:
                attempts += 1
                if refs.shape[0] > 0:
                    diff = refs - point[None, :]
                    if float(np.min(np.sum(diff * diff, axis=1))) < min_allowed2:
                        if attempts >= max_attempts:
                            break
                        continue

                point32 = point.astype(np.float32)
                selected.append(point32)
                if refs.shape[0] > 0:
                    refs = np.vstack((refs, point32[None, :]))
                else:
                    refs = point32[None, :]

                if len(selected) >= n_points or attempts >= max_attempts:
                    break

        if not selected:
            return np.empty((0, 2), dtype=np.float32)
        return np.asarray(selected, dtype=np.float32)

    def _reseed_points(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        if self.use_random_sparse_points:
            points = self._make_random_sparse_points(w, h, self.max_points)
        elif self.use_feature_sparse_points:
            points = self._make_feature_sparse_points(frame, self.max_points)
            if points.shape[0] < self.max_points:
                fallback = self._make_grid_points(w, h, self.max_points)
                points = self._merge_candidate_points(points, fallback, self.max_points)
        else:
            points = self._make_grid_points(w, h, self.max_points)
        self.points_px = points[: self.max_points]
        self.trails = [deque(maxlen=self.trail_length) for _ in range(self.points_px.shape[0])]
        self._append_points_to_trails(self.points_px)

    def _top_up_points(self, frame: np.ndarray) -> None:
        missing = self.max_points - int(self.points_px.shape[0])
        if missing <= 0:
            return

        h, w = frame.shape[:2]
        if self.use_random_sparse_points:
            candidates = self._make_random_sparse_points(
                w,
                h,
                missing,
                existing=self.points_px if self.points_px.size else None,
            )
        elif self.use_feature_sparse_points:
            candidates = self._make_feature_sparse_points(
                frame,
                missing,
                existing=self.points_px if self.points_px.size else None,
            )
            if candidates.shape[0] < missing:
                fallback = self._make_grid_points(w, h, self.max_points)
                if self.points_px.size:
                    existing = self.points_px.astype(np.float32, copy=False)
                    diffs = fallback[:, None, :] - existing[None, :, :]
                    min_d2 = np.min(np.sum(diffs * diffs, axis=2), axis=1)
                    min_allowed = (0.5 * float(self.min_distance)) ** 2
                    fallback = fallback[min_d2 >= min_allowed]
                candidates = self._merge_candidate_points(candidates, fallback, missing)
        else:
            candidates = self._make_grid_points(w, h, self.max_points)
            if self.points_px.size:
                existing = self.points_px.astype(np.float32, copy=False)
                diffs = candidates[:, None, :] - existing[None, :, :]
                min_d2 = np.min(np.sum(diffs * diffs, axis=2), axis=1)
                min_allowed = (0.5 * float(self.min_distance)) ** 2
                candidates = candidates[min_d2 >= min_allowed]

        if candidates.size == 0:
            return

        if candidates.shape[0] > missing:
            candidates = candidates[:missing]

        if self.points_px.size:
            self.points_px = np.vstack((self.points_px, candidates)).astype(np.float32)
        else:
            self.points_px = candidates

        for point in candidates:
            trail: Deque[Tuple[float, float]] = deque(maxlen=self.trail_length)
            trail.append((float(point[0]), float(point[1])))
            self.trails.append(trail)

    def _make_feature_sparse_points(
        self,
        frame: np.ndarray,
        n_points: int,
        existing: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if n_points <= 0:
            return np.empty((0, 2), dtype=np.float32)

        h, w = frame.shape[:2]
        xmin, xmax, ymin, ymax = self._middle_quarter_bounds(w, h)
        gray = self._gray(frame)
        mask = np.zeros(gray.shape[:2], dtype=np.uint8)
        x0 = int(np.floor(xmin))
        x1 = int(np.ceil(xmax))
        y0 = int(np.floor(ymin))
        y1 = int(np.ceil(ymax))
        mask[y0 : y1 + 1, x0 : x1 + 1] = 255

        if existing is not None and np.size(existing) > 0:
            refs = np.asarray(existing, dtype=np.float32).reshape(-1, 2)
            for point in refs:
                cv2.circle(
                    mask,
                    (int(round(float(point[0]))), int(round(float(point[1])))),
                    max(1, int(self.min_distance)),
                    0,
                    -1,
                )

        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=int(max(1, n_points)),
            qualityLevel=self.quality_level,
            minDistance=float(self.min_distance),
            mask=mask,
            blockSize=self.block_size,
            useHarrisDetector=False,
        )
        if corners is None:
            return np.empty((0, 2), dtype=np.float32)
        return corners.reshape(-1, 2).astype(np.float32)

    @staticmethod
    def _merge_candidate_points(
        primary: np.ndarray,
        fallback: np.ndarray,
        max_points: int,
    ) -> np.ndarray:
        primary = np.asarray(primary, dtype=np.float32).reshape(-1, 2)
        fallback = np.asarray(fallback, dtype=np.float32).reshape(-1, 2)
        if primary.shape[0] >= int(max_points):
            return primary[: int(max_points)]
        if fallback.size == 0:
            return primary
        merged = np.vstack((primary, fallback))
        return merged[: int(max_points)].astype(np.float32, copy=False)

    def _keep_points_in_frame(self, points: np.ndarray, width: int, height: int) -> np.ndarray:
        if points.size == 0:
            self.trails = []
            return np.empty((0, 2), dtype=np.float32)

        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        finite = np.isfinite(points).all(axis=1)
        in_bounds = (
            (points[:, 0] >= 0.0)
            & (points[:, 0] <= float(width - 1))
            & (points[:, 1] >= 0.0)
            & (points[:, 1] <= float(height - 1))
        )
        mid_xmin, mid_xmax, mid_ymin, mid_ymax = self._middle_quarter_bounds(width, height)
        in_middle_quarter = (
            (points[:, 0] >= mid_xmin)
            & (points[:, 0] <= mid_xmax)
            & (points[:, 1] >= mid_ymin)
            & (points[:, 1] <= mid_ymax)
        )
        keep = finite & in_bounds & in_middle_quarter

        filtered = points[keep]
        self.trails = [trail for trail, should_keep in zip(self.trails, keep) if bool(should_keep)]
        return filtered

    def _append_points_to_trails(self, points: np.ndarray) -> None:
        if points.size == 0:
            return

        if len(self.trails) < points.shape[0]:
            self.trails.extend(deque(maxlen=self.trail_length) for _ in range(points.shape[0] - len(self.trails)))

        for trail, point in zip(self.trails, points):
            trail.append((float(point[0]), float(point[1])))

    def _draw_overlay(self, frame: np.ndarray, points: np.ndarray, scale: float = 1.0) -> np.ndarray:
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        subpixel_shift = 4
        fixed_scale = float(scale) * float(1 << subpixel_shift)

        for trail in self.trails:
            if len(trail) < 2:
                continue
            pts = np.array(trail, dtype=np.float32)
            pts *= fixed_scale
            pts = pts.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(
                frame,
                [pts],
                isClosed=False,
                color=(60, 220, 60),
                thickness=self.trail_thickness,
                lineType=cv2.LINE_AA,
                shift=subpixel_shift,
            )

        for point in points:
            cv2.circle(
                frame,
                (
                    int(round(float(point[0]) * fixed_scale)),
                    int(round(float(point[1]) * fixed_scale)),
                ),
                6 << subpixel_shift,
                (0, 0, 255),
                -1,
                lineType=cv2.LINE_AA,
                shift=subpixel_shift,
            )
        return frame

    def _frame_delay_ms(self, index: int) -> int:
        if index + 1 < len(self.timestamps):
            dt = float(self.timestamps[index + 1] - self.timestamps[index])
            if np.isfinite(dt) and dt > 0:
                delay_ms = int(max(1, round(dt * 1000.0)))
                return int(np.clip(delay_ms, self.min_wait_ms, self.max_wait_ms))
        fallback_ms = int(max(1, round(1000.0 / self.playback_fps)))
        return int(np.clip(fallback_ms, self.min_wait_ms, self.max_wait_ms))

    def _pause_loop(self) -> bool:
        while True:
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                return True
            if key == ord(" "):
                return False


if __name__ == "__main__":
    tracker = PointTracker2(
        normalize=True,
        blur_ksize=3,
        debug_timing=False,
        debug_interval=10,
    )

    path = r"D:\holypipette_data_backup\experiments\Data\rig_recorder_data\2026_02_20-13_28\camera_frames" # tissue deform 
    # path  = r"D:\holypipette_data_backup\experiments\Data\rig_recorder_data\2026_02_20-13_28"
    # path = r"D:\holypipette_data_backup\experiments\Data\rig_recorder_data\2025_10_29-20_35\aux_camera_frames" # pipette approach
    # path = r"D:\holypipette_data_backup\experiments\Data\rig_recorder_data\2026_02_25-14_26\camera_frames" # cultured plate
    # path = r"C:\Users\sa-forest\Documents\GitHub\pipetteFindingCNN\pipettedata\SlicePipetteData\2025_10_30-17_29" # pipette movement
    streamer = FrameStreamer(
        path,
        tracker=tracker,
        max_points=24,
        trail_thickness=3,
        point_bounds_fraction=0.10,
        use_random_sparse_points=False,
        use_feature_sparse_points=False,
        inference_scale=0.5,
        debug_timing=False,
        debug_interval=10,
    )
    streamer.stream()
    pass
