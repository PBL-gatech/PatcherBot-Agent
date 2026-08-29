#!/usr/bin/env python
import os
import json
import time
import cv2
import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path
import logging


class PipetteFocuser:
    def __init__(
        self,
        model_path=None,
        device=None,
        model_factory=None,
        enable_contrast_stretch: bool = False,
        preprocess_config_path: str | None = None,
    ):
        """
        model_path: path to an ONNX (.onnx) or PyTorch (.pt/.pth) file.
                    Defaults to pipetteModel/regression_model2.onnx next to this file.
        device: optional torch.device override (used only for .pt/.pth inference).
        model_factory: optional callable returning a torch.nn.Module when the .pt file
                       is a state_dict (fallback if it is NOT TorchScript). Supply this
                       from the consuming repository; otherwise expect a TorchScript file.
        """
        cur_dir = Path(__file__).parent.resolve()
        model_path = Path(model_path) if model_path is not None else cur_dir / "pipetteModel"/ "PipetteFocuserNet.pt"

        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.backend = None

        # Input image size (as used in training transforms)
        self.imgSize = 224

        # Default to ImageNet statistics; may be overwritten when loading .pt/.pth stats
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.z_mean = None
        self.z_std = None
        self.enable_contrast_stretch = bool(enable_contrast_stretch)
        self._load_preprocess_config(model_path.parent, preprocess_config_path=preprocess_config_path)

        suffix = model_path.suffix.lower()
        if suffix == ".onnx":
            self._load_onnx(model_path)
        elif suffix in {".pt", ".pth"}:
            self._load_torch(model_path, model_factory=model_factory)
        else:
            raise ValueError(f"Unsupported model file extension: {model_path.suffix}")

    def _load_onnx(self, model_path: Path):
        """Initialize ONNX Runtime session (keeps legacy behavior)."""
        self.session = ort.InferenceSession(str(model_path))
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.backend = "onnx"

    def _load_torch(self, model_path: Path, model_factory=None):
        """
        Load a PyTorch model.
        - First tries TorchScript (works when you export scripted/traced .pt).
        - If that fails and a model_factory is provided, builds the model and loads a state_dict.
        """
        self._load_stats(model_path.parent)
        self.backend = "torch"

        # Try TorchScript first for maximum portability
        try:
            self.model = torch.jit.load(str(model_path), map_location=self.device)
            self.model.eval()
            return
        except (RuntimeError, ValueError):
            pass

        if model_factory is None:
            raise RuntimeError(
                "The .pt/.pth file is not TorchScript. Provide a TorchScript export or pass model_factory "
                "that returns the correct torch.nn.Module to load the state_dict."
            )

        model = model_factory()
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        self.model = model

    def _load_stats(self, stats_dir: Path):
        """
        Load channel_norm.json and z_norm.json if they exist beside the model.
        These are used only for .pt/.pth inference.
        """
        channel_path = stats_dir / "channel_norm.json"
        if channel_path.is_file():
            try:
                with open(channel_path, "r", encoding="utf-8") as f:
                    stats = json.load(f)
                if "mean" in stats and "std" in stats:
                    self.mean = np.array(stats["mean"], dtype=np.float32)
                    self.std = np.array(stats["std"], dtype=np.float32)
            except Exception as exc:
                logging.warning(f"Failed to load {channel_path}: {exc}")

        z_path = stats_dir / "z_norm.json"
        if z_path.is_file():
            try:
                with open(z_path, "r", encoding="utf-8") as f:
                    z_stats = json.load(f)
                if "z_mean" in z_stats and "z_std" in z_stats:
                    self.z_mean = float(z_stats["z_mean"])
                    self.z_std = float(z_stats["z_std"] if z_stats["z_std"] != 0 else 1.0)
            except Exception as exc:
                logging.warning(f"Failed to load {z_path}: {exc}")

    def _load_preprocess_config(
        self,
        stats_dir: Path,
        preprocess_config_path: str | None = None,
    ):
        """
        Load preprocessing config if present.
        Expected file: preprocess_config.json (beside the model) or explicit path.
        """
        cfg_path = Path(preprocess_config_path) if preprocess_config_path else (stats_dir / "preprocess_config.json")
        if not cfg_path.is_file():
            return
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if "enable_contrast_stretch" in cfg:
                self.enable_contrast_stretch = bool(cfg["enable_contrast_stretch"])
        except Exception as exc:
            logging.warning(f"Failed to load {cfg_path}: {exc}")

    @staticmethod
    def contrast_stretch_mu_2sigma_uint8(image: np.ndarray) -> np.ndarray:
        """
        Per-image contrast stretch:
          low = mean - 2*std
          high = mean + 2*std
          out = clip((x-low)/(high-low), 0, 1)
        Returns uint8 image in [0, 255].
        """
        img = np.asarray(image)
        if img.size == 0:
            return img.copy()

        if img.dtype != np.uint8:
            if np.issubdtype(img.dtype, np.floating):
                max_val = float(np.nanmax(img)) if img.size else 0.0
                scale = 255.0 if max_val <= 1.0 else 1.0
                img_uint8 = np.clip(img * scale, 0, 255).astype(np.uint8)
            else:
                img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
        else:
            img_uint8 = img

        img_float = img_uint8.astype(np.float32)
        mu = float(np.mean(img_float))
        sigma = float(np.std(img_float))

        if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 1e-6:
            return img_uint8.copy()

        low = mu - 2.0 * sigma
        high = mu + 2.0 * sigma
        if high <= low + 1e-6:
            return img_uint8.copy()

        stretched = (img_float - low) / (high - low)
        stretched = np.clip(stretched, 0.0, 1.0)
        return np.round(stretched * 255.0).astype(np.uint8)

    def preprocess(self, img):
        """
        Preprocess the image:
          - Optional contrast stretching (mu +/- 2*sigma) when enabled.
          - Resize to self.imgSize x self.imgSize.
          - Convert BGR (OpenCV) to RGB.
          - Scale pixel values to [0, 1].
          - Normalize with mean and std.
          - Rearrange dimensions from HWC to CHW and add a batch dimension.
        
        Args:
            img (np.ndarray): Input image in BGR format.

        Returns:
            np.ndarray: Preprocessed image of shape (1, 3, imgSize, imgSize).
        """
        working_img = img
        if working_img.ndim == 2:
            working_img = cv2.cvtColor(working_img, cv2.COLOR_GRAY2BGR)
        if self.enable_contrast_stretch:
            working_img = self.contrast_stretch_mu_2sigma_uint8(working_img)

        img_resized = cv2.resize(working_img, (self.imgSize, self.imgSize))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0
        img_normalized = (img_float - self.mean) / self.std
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        img_batch = np.expand_dims(img_transposed, axis=0)
        return img_batch

    def get_pipette_focus_value(self, img):
        """
        Runs the ONNX model on a preprocessed image and returns the defocus value in microns.
        
        Args:
            img (np.ndarray): Input image in BGR format.

        Returns:
            float: Predicted defocus value in microns.
        """
        input_tensor = self.preprocess(img)
        start_time = time.time()

        if self.backend == "onnx":
            outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
            output_array = outputs[0]
            norm_pred = float(output_array.flatten()[0])
        elif self.backend == "torch":
            torch_input = torch.from_numpy(input_tensor).to(self.device)
            with torch.no_grad():
                output = self.model(torch_input)
            norm_pred = float(output.reshape(-1)[0].item())
        else:
            raise RuntimeError("PipetteFocuser backend not initialized.")

        inference_time = time.time() - start_time
        # print(f"Inference time: {inference_time:.4f} seconds")

        pred_microns = norm_pred
        if self.backend == "torch" and self.z_mean is not None and self.z_std is not None:
            pred_microns = norm_pred * self.z_std + self.z_mean
        return pred_microns

if __name__ == '__main__':
    focuser = PipetteFocuser()
    
    # Adjust image path as needed
    # cur_dir = Path(__file__).parent.absolute()
    # image_path = os.path.join(cur_dir, "", "neg7_focus.png")
    image_path = r"C:\Users\sa-forest\GaTech Dropbox\Benjamin Magondu\YOLOretrainingdata\Pipette CNN Training Data\20191016\3654098923.png"
    # image_path = r"C:\Users\sa-forest\Documents\GitHub\pipetteFindingCNN\pipettedata\3DPrelimData\compiled\cropped_camera_frames\2051_1770343660.202234.webp" # cropped example
    # image_path = r"C:\Users\sa-forest\Documents\GitHub\pipetteFindingCNN\pipettedata\3DPrelimData\compiled\cropped_camera_frames\2051_1769809540.821779.webp" # cropped example, different focus
    image_path = r"C:\Users\sa-forest\Documents\GitHub\pipetteFindingCNN\pipettedata\3DPrelimData\compiled\cropped_camera_frames\2070_1770056951.261588.webp" # cropped example, different focus
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        exit(1)
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image: {image_path}")
        exit(1)
    
    focus_value = focuser.get_pipette_focus_value(img)
    # print(f"Predicted pipette focus value: {focus_value:.2f} microns")
    
    label = f"Focus: {focus_value:.2f} um"
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Pipette Focus Inference", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
