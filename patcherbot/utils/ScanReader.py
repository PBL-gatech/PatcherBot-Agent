"""Normalize 40x microscope scans and read an upright etched number.

Normalization: numpy, OpenCV, Pillow. OCR additionally requires torch,
torchvision, transformers>=5.8.0, scipy and scikit-image. No CLI is required.

    enhanced = ScanReader.normalize("scan_40x.tif")
    enhanced.save("scan_40x_enhanced.tif", compression="tiff_deflate")
    reader = ScanReader(crop=number_rectangle)  # x, y, width, height
    number = reader.read(original_scan)  # Normalize, rotate, flip, crop, ridge filter, OCR.
    number = reader.read(upright_scan, prepared=True)  # Skip image preparation.
    number = reader.read_crop(number_image)  # Skip preparation and crop selection.

The 40x preparation rotates 16 degrees clockwise, flips vertically, and fits
within 1400 pixels (configurable). Crop coordinates are in that prepared image,
not in the original scan. No crop rectangle or expected number is hardcoded.
read() requires a caller-selected number rectangle; it does not locate all
numbered squares in an entire mosaic. Outputs
are strings preserving leading zeros; strict=True raises for nonnumeric output.
strict=False returns an empty string instead. Scores are raw model scores,
not calibrated probabilities.
Model weights download from Hugging Face on first OCR use, then are cached.
Images are saved only when debug_dir is supplied; source arrays are not modified.
"""

from __future__ import annotations

import math
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class ScanReader:
    MODEL_ID = "PaddlePaddle/PP-OCRv6_medium_rec_safetensors"

    def __init__(self, crop=None, device=None, rotation_clockwise=16.0,
                 flip_vertical=True, working_max_dimension=1400):
        if crop is not None:
            if len(crop) != 4 or any(
                not isinstance(v, (int, np.integer)) for v in crop
            ):
                raise ValueError("crop must contain integers: x, y, width, height")
            crop = tuple(int(v) for v in crop)
            if min(crop[:2]) < 0 or min(crop[2:]) <= 0:
                raise ValueError("Crop origin must be nonnegative and size positive")
        self.crop = crop
        if not math.isfinite(rotation_clockwise):
            raise ValueError("rotation_clockwise must be finite")
        if (not isinstance(working_max_dimension, int)
                or isinstance(working_max_dimension, bool) or working_max_dimension <= 0):
            raise ValueError("working_max_dimension must be a positive integer")
        self.rotation_clockwise = float(rotation_clockwise)
        self.flip_vertical = bool(flip_vertical)
        self.working_max_dimension = working_max_dimension
        self.device = device
        self.processor = self.model = None
        self.last_text = self.last_score = self.last_filtered = None
        self.last_prepared = self.last_region = self.last_sobel = None
        self.last_canny = self.last_sobel_mask = None
        self.last_uncleaned = None

    @staticmethod
    def _log(message):
        print(f"[ScanReader] {message}", flush=True)

    def _reset_last(self):
        self.last_text = self.last_score = self.last_filtered = None
        self.last_prepared = self.last_region = self.last_sobel = None
        self.last_canny = self.last_sobel_mask = self.last_uncleaned = None

    @staticmethod
    def _load_rgb(image):
        """Accept uint8 grayscale/RGB/RGBA; convert OpenCV BGR before calling."""
        if isinstance(image, (str, Path)):
            with Image.open(image) as source:
                return ScanReader._load_rgb(source)
        if isinstance(image, Image.Image):
            if image.mode not in ("L", "RGB", "RGBA", "P"):
                raise ValueError("Scale non-8-bit images explicitly before reading")
            return np.array(image.convert("RGB"), dtype=np.uint8)
        if not isinstance(image, np.ndarray):
            raise TypeError("Expected a path, PIL image, or uint8 NumPy array")
        if image.dtype != np.uint8:
            raise ValueError("Array input must be uint8; scale other depths first")
        if image.ndim == 3 and image.shape[2] == 1:
            image = image[:, :, 0]
        if not (image.ndim == 2 or (image.ndim == 3 and image.shape[2] in (3, 4))):
            raise ValueError("Expected HxW grayscale or HxWx3/4 RGB/RGBA")
        if not image.size:
            raise ValueError("Image must not be empty")
        return np.array(Image.fromarray(image).convert("RGB"), dtype=np.uint8)

    def select_crop(self, image):
        """Select the configured rectangle from an already prepared image."""
        rgb = self._load_rgb(image)
        if self.crop is not None:
            x, y, width, height = self.crop
            if x + width > rgb.shape[1] or y + height > rgb.shape[0]:
                raise ValueError("Crop extends outside the image")
            rgb = rgb[y:y + height, x:x + width]
        return Image.fromarray(rgb)

    def prepare(self, image, normalize=True):
        """Prepare an original native 40x scan before selecting a number region.

        Uses the previously applied native 40x normalization settings. Rotate
        on an expanded canvas, flip vertically, then reduce for ridge filtering.
        Call select_crop(last_prepared) to inspect the selected number region.
        """
        if normalize:
            gray = self.normalize(image, noise_sigma=1.5, background_sigma=80,
                                  local_sigma=80, rms_floor=2, gain=40, midpoint=128)
        else:
            gray = Image.fromarray(self._load_rgb(image)).convert("L")
        pixels = np.asarray(gray)
        height, width = pixels.shape
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2),
                                        -self.rotation_clockwise, 1.0)
        cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
        output_width = math.ceil(width * cosine + height * sine)
        output_height = math.ceil(height * cosine + width * sine)
        matrix[0, 2] += (output_width - width) / 2
        matrix[1, 2] += (output_height - height) / 2
        pixels = cv2.warpAffine(pixels, matrix, (output_width, output_height),
                                flags=cv2.INTER_LINEAR, borderValue=0)
        if self.flip_vertical:
            pixels = pixels[::-1]
        scale = min(1.0, self.working_max_dimension / max(output_width, output_height))
        if scale < 1:
            pixels = cv2.resize(pixels, (round(output_width * scale), round(output_height * scale)),
                                interpolation=cv2.INTER_AREA)
        self.last_prepared = Image.fromarray(np.ascontiguousarray(pixels))
        return self.last_prepared

    @staticmethod
    def normalize(image, noise_sigma=1.0, background_sigma=30.0,
                  local_sigma=20.0, rms_floor=1.0, gain=42.5, midpoint=127.5):
        """Reproduce the saved 4x scan's masked Gaussian/local RMS normalization.

        Sigmas are native input pixels. Broad filters run at quarter resolution
        with area downsampling and bilinear upsampling, as in that enhancement.
        All-zero source pixels mark uncovered canvas and remain black. This
        increases visible contrast, including artifacts; it adds no grid lines.
        """
        for value in (noise_sigma, background_sigma, local_sigma, rms_floor):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Sigmas and RMS floor must be finite and positive")
        if not math.isfinite(gain) or gain < 0:
            raise ValueError("gain must be finite and nonnegative")
        if not math.isfinite(midpoint) or not 0 <= midpoint <= 255:
            raise ValueError("midpoint must be between 0 and 255")
        rgb = ScanReader._load_rgb(image)
        mask = np.any(rgb > 0, axis=2).astype(np.float32)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        height, width = gray.shape
        work_size = (max(1, math.ceil(width / 4)), max(1, math.ceil(height / 4)))

        def masked_blur(values, sigma, wide=False):
            weights = mask
            weighted = values * mask
            if wide:
                weights = cv2.resize(weights, work_size, interpolation=cv2.INTER_AREA)
                weighted = cv2.resize(weighted, work_size, interpolation=cv2.INTER_AREA)
                sx, sy = sigma * work_size[0] / width, sigma * work_size[1] / height
            else:
                sx = sy = sigma
            numerator = cv2.GaussianBlur(weighted, (0, 0), sx, sigmaY=sy)
            denominator = cv2.GaussianBlur(weights, (0, 0), sx, sigmaY=sy)
            result = numerator / np.maximum(denominator, 1e-6)
            if wide:
                result = cv2.resize(result, (width, height), interpolation=cv2.INTER_LINEAR)
            return result

        smoothed = masked_blur(gray, noise_sigma)
        detail = (smoothed - masked_blur(gray, background_sigma, wide=True)) * mask
        rms = np.sqrt(np.maximum(masked_blur(detail * detail, local_sigma, wide=True), 0))
        result = np.clip(midpoint + gain * detail / np.maximum(rms, rms_floor), 0, 255)
        result[mask == 0] = 0
        return Image.fromarray(result.astype(np.uint8))

    def filter_contours(self, image, normalize=True):
        """Filter an already upright image, applying the configured crop."""
        region = self.select_crop(image)
        gray = self.normalize(region) if normalize else region.convert("L")
        return self.filter_ridges(gray)

    def filter_ridges(self, region):
        """Reproduce the earlier successful dual-polarity ridge input.

        No geometry changes or normalization: sigma 1.5, 80th-percentile
        threshold, skeletonization, and removal of components under 50 pixels.
        Returns black lines on white. Parameters were tested on the supplied
        aligned scan; they are not validated across arbitrary scans or scales.
        """
        from scipy import ndimage as ndi
        from skimage.filters import frangi
        from skimage.morphology import skeletonize

        gray = Image.fromarray(self._load_rgb(region)).convert("L")
        values = np.asarray(gray, dtype=np.float64) / 255.0
        ridges = np.maximum(
            frangi(values, sigmas=[1.5], black_ridges=True),
            frangi(values, sigmas=[1.5], black_ridges=False),
        )
        if not np.isfinite(ridges).all():
            raise ValueError("Ridge filtering produced non-finite values")
        contours = skeletonize(ridges > np.percentile(ridges, 80))
        labels, _ = ndi.label(contours, structure=np.ones((3, 3), dtype=np.uint8))
        sizes = np.bincount(labels.ravel())
        contours &= sizes[labels] >= 50
        self.last_filtered = Image.fromarray(np.where(contours, 0, 255).astype(np.uint8))
        return self.last_filtered

    def filter_edges(self, region, min_component_area=8, min_component_span=18):
        """Keep strong Sobel strokes supported by nearby Canny edge pixels.

        Input is the already normalized, rotated, flipped and cropped region.
        Otsu sets the high threshold from gradient magnitudes; the low threshold
        is 40% of it. Expand Canny support by one pixel and intersect it with
        the strong Sobel mask, retaining stroke width before the OCR resize.
        These are related filters, not independent evidence of lettering.
        No expected glyphs are supplied or synthesized.
        Remove isolated components below the supplied area or maximum span
        (in prepared-image pixels), retaining extended strokes. This removes
        pixels only; faint disconnected character fragments can also be lost.
        """
        if min_component_area < 1 or min_component_span < 1:
            raise ValueError("Component area and span must be positive")
        gray = cv2.cvtColor(self._load_rgb(region), cv2.COLOR_RGB2GRAY)
        smooth = cv2.GaussianBlur(gray, (0, 0), 1.0)
        dx = cv2.Sobel(smooth, cv2.CV_16S, 1, 0, ksize=3)
        dy = cv2.Sobel(smooth, cv2.CV_16S, 0, 1, ksize=3)
        magnitude = cv2.magnitude(dx.astype(np.float32), dy.astype(np.float32))
        peak = float(magnitude.max())
        if peak == 0:
            scaled = np.zeros(gray.shape, dtype=np.uint8)
            edges = np.zeros_like(scaled)
            strong = np.zeros(gray.shape, dtype=bool)
        else:
            scaled = np.clip(magnitude * (255.0 / peak), 0, 255).astype(np.uint8)
            threshold, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            high = max(1.0, threshold * peak / 255.0)
            edges = cv2.Canny(dx, dy, high * 0.4, high, L2gradient=True)
            strong = magnitude >= high
        support = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8)) != 0
        overlap = support & strong
        self.last_uncleaned = Image.fromarray(np.where(overlap, 0, 255).astype(np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            overlap.astype(np.uint8), connectivity=8)
        keep = np.zeros(count, dtype=bool)
        keep[1:] = ((stats[1:, cv2.CC_STAT_AREA] >= min_component_area)
                    & (np.maximum(stats[1:, cv2.CC_STAT_WIDTH],
                                  stats[1:, cv2.CC_STAT_HEIGHT]) >= min_component_span))
        overlap = keep[labels]
        self.last_sobel = Image.fromarray(scaled)
        self.last_canny = Image.fromarray(255 - edges)
        self.last_sobel_mask = Image.fromarray(np.where(strong, 0, 255).astype(np.uint8))
        self.last_filtered = Image.fromarray(np.where(overlap, 0, 255).astype(np.uint8))
        return self.last_filtered

    def _load_model(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoImageProcessor, AutoModelForTextRecognition

        device = torch.device(self.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use device='cpu' or CUDA-enabled PyTorch")
        self._log(f"Loading {self.MODEL_ID} on {device}...")
        processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
        model = AutoModelForTextRecognition.from_pretrained(
            self.MODEL_ID, use_safetensors=True
        ).to(device).eval()
        self.processor, self.model, self.device = processor, model, device
        self._log("Model ready")

    def read(self, image, normalize=True, contours=True, *, prepared=False,
             filter_method="ridge", strict=True, print_result=True, debug_dir=None):
        """Read the configured crop from a scan, printing and returning digits.

        prepared=True bypasses normalization, rotation, flip AND resizing.
        Otherwise prepare(image, normalize=normalize) runs as before; crop
        coordinates must refer to its output. normalize=False alone only skips
        contrast normalization. contours=False skips the optional ridge/edge
        filter; filter_method='sobel_canny' selects the supplied Sobel/Canny path.
        The model's required resize and normalization still run in all modes.
        """
        self._reset_last()
        if self.crop is None:
            raise ValueError("Set crop=(x, y, width, height) in prepared image pixels; "
                             "or use read_crop(image) for an already cropped number")
        if prepared:
            self._log("Using supplied prepared image; skipping normalization and geometry changes")
            self.last_prepared = Image.fromarray(self._load_rgb(image))
        else:
            self._log("Preparing original scan...")
            self.last_prepared = self.prepare(image, normalize=normalize)
        self._log(f"Prepared size: {self.last_prepared.size}; crop xywh: {self.crop}")
        region = self.select_crop(self.last_prepared)
        return self._recognize_region(region, contours, filter_method, strict,
                                      print_result, debug_dir)

    def read_crop(self, image, contours=True, *, filter_method="ridge", strict=True,
                  print_result=True, debug_dir=None):
        """Read a number crop without preparing or cropping it again.

        This ignores self.crop. Use contours=False for an OCR-ready image;
        only the model's own required resize/normalization will then run.
        """
        self._reset_last()
        region = Image.fromarray(self._load_rgb(image))
        self._log(f"Using supplied number crop unchanged: {region.size}")
        return self._recognize_region(region, contours, filter_method, strict,
                                      print_result, debug_dir)

    def _recognize_region(self, region, contours, filter_method, strict,
                          print_result, debug_dir):
        self.last_region = region
        if contours:
            self._log(f"Applying {filter_method} filter...")
            if filter_method == "ridge":
                filtered = self.filter_ridges(region)
            elif filter_method == "sobel_canny":
                filtered = self.filter_edges(region)
            else:
                raise ValueError("filter_method must be 'ridge' or 'sobel_canny'")
        else:
            filtered = region
            self.last_filtered = filtered
        debug_path = Path(debug_dir) if debug_dir is not None else None
        if debug_path is not None:
            debug_path.mkdir(parents=True, exist_ok=True)
            # Do not leave a previous run's successful result beside new inputs.
            (debug_path / "diagnostics.json").unlink(missing_ok=True)
            if self.last_prepared is not None:
                self.last_prepared.save(debug_path / "prepared.png")
            else:
                (debug_path / "prepared.png").unlink(missing_ok=True)
            region.save(debug_path / "crop.png")
            filtered.save(debug_path / "ocr_input.png")
            self._log(f"Saved OCR input to {debug_path.resolve()}")
        self._load_model()
        import torch

        inputs = self.processor(images=filtered.convert("RGB"), return_tensors="pt").to(self.device)
        self._log(f"Running OCR on {tuple(inputs['pixel_values'].shape)}")
        with torch.inference_mode():
            outputs = self.model(**inputs)
        result = self.processor.post_process_text_recognition(outputs)[0]
        self.last_text = str(result["text"])
        score = float(result.get("score", float("nan")))
        self.last_score = score if math.isfinite(score) else None
        self._log(f"Raw OCR text: {self.last_text!r}; score: {self.last_score}")
        if debug_path is not None:
            from importlib.metadata import version
            diagnostics = {
                "model_id": self.MODEL_ID, "device": str(self.device),
                "torch": torch.__version__, "transformers": version("transformers"),
                "torchvision": version("torchvision"), "opencv": cv2.__version__,
                "filter_method": filter_method if contours else "none",
                "region_size": list(region.size),
                "input_shape": list(inputs["pixel_values"].shape),
                "raw_text": self.last_text, "score": self.last_score,
            }
            (debug_path / "diagnostics.json").write_text(
                json.dumps(diagnostics, indent=2, allow_nan=False), encoding="utf-8")
        text = "".join(self.last_text.split())
        if not text or any(char not in "0123456789" for char in text):
            if strict:
                raise ValueError(f"OCR returned {self.last_text!r} (score={self.last_score}), "
                                 "not a complete number. Inspect last_region/last_filtered "
                                 "or pass debug_dir='scan_debug' to save the inputs.")
            return ""
        if print_result:
            print(text, flush=True)
        return text
