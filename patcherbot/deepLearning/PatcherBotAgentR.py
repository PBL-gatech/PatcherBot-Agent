from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw

# Make the vendored robomimic package importable without requiring installation.
ROBO_ROOT = Path(__file__).resolve().parent / "patchModel" / "robomimic"
if str(ROBO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROBO_ROOT))

from robomimic.envs.env_patcher_online import EnvPatcherOnline
from robomimic.envs.wrappers import FrameStackWrapper
from robomimic.utils import obs_utils as ObsUtils
from robomimic.utils.file_utils import config_from_checkpoint, policy_from_checkpoint
from robomimic.utils.torch_utils import get_torch_device


class ModelImporter:
    """
    Load a robomimic checkpoint and prepare an EnvPatcherOnline instance plus
    the policy, action dimension, frame stack, and observation keys.
    """
    def __init__(
        self,
        ckpt_path: Optional[Union[str, Path]] = None,
        *,
        success_epsilon: float = 0.10,
        frame_stack: Optional[int] = None,
        model_type: Optional[str] = None,
    ) -> None:
        self.ckpt_path = ckpt_path
        self.model_type = model_type
        self.success_epsilon = float(success_epsilon)
        self.frame_stack_override = frame_stack

        self.policy = None
        self.cfg = None
        self.ckpt_dict = None
        self.obs_keys: Sequence[str] = []
        self.action_dim: Optional[int] = None
        self.frame_stack: int = 1
        self.goal_required: bool = False
        self.env = None
        self.image_key: Optional[str] = None
        self.pipette_key: Optional[str] = None
        self.stage_key: Optional[str] = None
        self.resistance_key: Optional[str] = None
        self.required_obs_keys: Sequence[str] = []
        self.obs_shapes: Dict[str, Any] = {}

    # ---------------------- helpers ----------------------
    @staticmethod
    def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, Mapping):
            return obj.get(key, default)
        return getattr(obj, key, default)

    @staticmethod
    def _to_list(value: Optional[Sequence]) -> list:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    @staticmethod
    def _ensure_key_in_modalities(modalities: Dict[str, list], key: Optional[str], modality: str) -> None:
        if key is None:
            return
        entries = modalities.setdefault(modality, [])
        if key not in entries:
            entries.append(key)

    def _extract_obs_modalities(self, cfg) -> Dict[str, list]:
        obs_cfg = self._safe_get(self._safe_get(cfg.observation, "modalities"), "obs")
        if obs_cfg is None:
            return {}
        if hasattr(obs_cfg, "to_dict"):
            obs_dict = obs_cfg.to_dict()
        elif isinstance(obs_cfg, Mapping):
            obs_dict = dict(obs_cfg)
        else:
            obs_dict = {}
        modalities: Dict[str, list] = {}
        for modality, keys in obs_dict.items():
            modalities[str(modality)] = [str(k) for k in self._to_list(keys)]
        return modalities

    @staticmethod
    def _extract_policy_action(output: Any) -> np.ndarray:
        if isinstance(output, np.ndarray):
            arr = output
        elif isinstance(output, (list, tuple)):
            arr = np.asarray(output)
        elif hasattr(output, "detach") and hasattr(output, "cpu"):
            arr = output.detach().cpu().numpy()
        elif isinstance(output, Mapping):
            for key in ("actions", "action", "pred_actions", "ac"):
                if key in output:
                    return ModelImporter._extract_policy_action(output[key])
            first_val = next(iter(output.values()), None)
            if first_val is not None:
                return ModelImporter._extract_policy_action(first_val)
            arr = np.array([], dtype=np.float32)
        else:
            arr = np.asarray(output)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 0:
            return arr.reshape(1)
        if arr.ndim > 1:
            arr = arr.reshape(arr.shape[0], -1)
            arr = arr[0]
        return arr.reshape(-1)

    @staticmethod
    def _infer_action_dim(ckpt_dict: Mapping[str, Any]) -> Optional[int]:
        meta = ckpt_dict.get("shape_metadata") if isinstance(ckpt_dict, Mapping) else None
        if not meta:
            return None
        if "ac_dim" in meta:
            return int(meta["ac_dim"])
        action_entry = meta.get("action") if isinstance(meta, Mapping) else None
        shape = None
        if isinstance(action_entry, Mapping):
            if "shape" in action_entry:
                shape = action_entry.get("shape")
            else:
                for value in action_entry.values():
                    if isinstance(value, Mapping) and "shape" in value:
                        shape = value.get("shape")
                        break
        if shape is None:
            return None
        try:
            dims = np.asarray(shape, dtype=int).reshape(-1)
        except Exception:
            return None
        if dims.size == 0:
            return None
        return int(np.prod(dims))

    def _find_default_checkpoint(self) -> Path:
        agent_name = (self.model_type or self.__class__.__name__).upper()
        env_keys = [
            f"PATCHERBOT_ROBO_CKPT_{agent_name}",
            "PATCHERBOT_ROBO_CKPT",
        ]
        for key in env_keys:
            val = os.environ.get(key)
            if val:
                candidate = Path(val).expanduser()
                if candidate.exists():
                    return candidate
        training_dir = Path(__file__).resolve().parent / "patchModel" / "robomimic" / "training"
        candidates = sorted(training_dir.rglob("*.pth"))
        if candidates:
            return candidates[0]
        raise FileNotFoundError(
            "No robomimic checkpoint found. Set PATCHERBOT_ROBO_CKPT[_<AGENT>] "
            "or pass ckpt_path explicitly."
        )

    def _resolve_checkpoint(self, path: Union[str, Path]) -> Path:
        candidate = Path(path).expanduser()
        if candidate.is_file():
            return candidate
        if candidate.is_dir():
            ckpts = sorted(candidate.glob("*.pth"))
            if not ckpts:
                raise FileNotFoundError(f"No .pth checkpoints found in {candidate}")
            return ckpts[0]
        globbed = sorted(Path().glob(str(candidate)))
        for entry in globbed:
            if entry.is_file() and entry.suffix == ".pth":
                return entry
        raise FileNotFoundError(f"Checkpoint not found for {candidate}")

    # ---------------------- load ----------------------
    def load(self) -> "ModelImporter":
        ckpt_path = self.ckpt_path or self._find_default_checkpoint()
        ckpt_path = self._resolve_checkpoint(ckpt_path)
        self.resolved_ckpt_path = ckpt_path

        device = get_torch_device(try_to_use_cuda=True)
        policy, ckpt_dict = policy_from_checkpoint(ckpt_path=str(ckpt_path), device=device, verbose=False)
        cfg, _ = config_from_checkpoint(ckpt_path=str(ckpt_path), ckpt_dict=ckpt_dict)
        self.policy_class = policy.__class__.__name__
        self.experiment_name = getattr(getattr(cfg, "experiment", None), "name", None)
        # Lightweight visibility into which model is in use.
        print(f"[PatcherBotAgentR] using checkpoint: {ckpt_path}")
        print(f"[PatcherBotAgentR] policy class: {self.policy_class}")
        if self.experiment_name:
            print(f"[PatcherBotAgentR] experiment name: {self.experiment_name}")

        obs_modalities = self._extract_obs_modalities(cfg)
        rgb_keys = obs_modalities.get("rgb", [])
        low_dim_keys = obs_modalities.get("low_dim", [])
        image_key = rgb_keys[0] if rgb_keys else None
        pipette_key = next((k for k in low_dim_keys if "pipette" in k.lower()), None)
        stage_key = next((k for k in low_dim_keys if "stage" in k.lower()), None)
        resistance_key = next((k for k in low_dim_keys if "resist" in k.lower()), None)
        self._ensure_key_in_modalities(obs_modalities, image_key, "rgb")
        self._ensure_key_in_modalities(obs_modalities, pipette_key, "low_dim")
        self._ensure_key_in_modalities(obs_modalities, stage_key, "low_dim")
        self._ensure_key_in_modalities(obs_modalities, resistance_key, "low_dim")

        goal_modalities = self._safe_get(self._safe_get(cfg.observation, "modalities"), "goal") or {}
        self.goal_required = any(bool(v) for v in goal_modalities.values())
        self.image_key = image_key
        self.pipette_key = pipette_key
        self.stage_key = stage_key
        self.resistance_key = resistance_key

        # Frame stack: override -> policy -> train cfg -> default 1
        frame_stack = self.frame_stack_override
        if frame_stack is None:
            policy_frame_stack = None
            policy_impl = getattr(policy, "policy", None)
            if policy_impl is not None:
                cfg_policy = getattr(policy_impl, "global_config", None)
                if cfg_policy is not None:
                    try:
                        cfg_stack = getattr(cfg_policy.train, "frame_stack", None)
                        if cfg_stack is not None:
                            policy_frame_stack = int(cfg_stack)
                    except AttributeError:
                        policy_frame_stack = None
            if policy_frame_stack is None:
                cfg_stack = getattr(cfg.train, "frame_stack", None)
                if cfg_stack is not None:
                    policy_frame_stack = int(cfg_stack)
            if policy_frame_stack is None and policy_impl is not None:
                try:
                    policy_frame_stack = int(policy_impl.algo_config.horizon.observation_horizon)
                except Exception:
                    policy_frame_stack = None
            frame_stack = policy_frame_stack
        self.frame_stack = max(int(frame_stack or 1), 1)

        self.policy = policy
        self.cfg = cfg
        self.ckpt_dict = ckpt_dict
        self.obs_keys = self._collect_obs_keys(
            obs_modalities,
            image_key=image_key,
            pipette_key=pipette_key,
            stage_key=stage_key,
            resistance_key=resistance_key,
        )
        self.required_obs_keys = list(self.obs_keys)
        self.action_dim = self._infer_action_dim(ckpt_dict)
        shape_meta = ckpt_dict.get("shape_metadata") if isinstance(ckpt_dict, Mapping) else None
        if isinstance(shape_meta, Mapping):
            self.obs_shapes = dict(shape_meta.get("all_shapes", {}) or {})

        base_env = EnvPatcherOnline(
            obs_keys=self.obs_keys,
            action_dim=self.action_dim,
            success_epsilon=self.success_epsilon,
            modalities={k: v for k, v in obs_modalities.items()},
        )
        self.env = FrameStackWrapper(base_env, num_frames=self.frame_stack) if self.frame_stack > 1 else base_env
        return self

    def _collect_obs_keys(
        self,
        modalities: Mapping[str, Sequence[str]],
        image_key: Optional[str],
        pipette_key: Optional[str],
        stage_key: Optional[str],
        resistance_key: Optional[str],
    ) -> Sequence[str]:
        ordered: list[str] = []
        seen = set()
        for keys in modalities.values():
            for key in keys:
                if key not in seen:
                    ordered.append(key)
                    seen.add(key)
        for key in (image_key, pipette_key, stage_key, resistance_key):
            if key and key not in seen:
                ordered.append(key)
                seen.add(key)
        return ordered


class ModelInferencer:
    """
    Mirror the ONNX ModelInferencer API while routing inference through robomimic.
    """

    def __init__(self, model_importer: ModelImporter) -> None:
        self.crop_size = 1.0
        self.image_resize = (85, 85)
        self.image_layout = "CHW"
        self._last_frame_params: Optional[Dict[str, float]] = None
        self._pipette_action_dim: Optional[int] = None
        # Keep a per-run folder name for debugging artifacts (mirrors ONNX agent behavior)
        self._debug_run_uid = datetime.now().strftime("%Y_%m_%d-%H_%M")
        # When True, missing goal fields will be auto‑filled from the latest observation
        # (or zeros shaped like the observation) instead of raising.
        self.allow_goal_placeholders: bool = True
        self.importer = model_importer.load()
        self.policy = self.importer.policy
        self.env = self.importer.env
        self.obs_keys = self.importer.obs_keys
        self.goal_required = bool(self.importer.goal_required)
        self.goal = None
        self._started = False
        self._last_obs: Optional[Dict[str, Any]] = None
        self.using_robomimic = True
        self.image_key = getattr(self.importer, "image_key", None) or "camera_image"
        self.pipette_key = getattr(self.importer, "pipette_key", None) or "pipette_positions"
        self.stage_key = getattr(self.importer, "stage_key", None) or "stage_positions"
        self.resistance_key = getattr(self.importer, "resistance_key", None) or "resistance"

        obs_shapes = getattr(self.importer, "obs_shapes", {}) or {}
        camera_shape = obs_shapes.get(self.image_key) or obs_shapes.get("camera_image")
        if camera_shape is not None:
            try:
                camera_shape = tuple(int(s) for s in camera_shape)
            except Exception:
                camera_shape = None
        if camera_shape and 3 in camera_shape and camera_shape.index(3) == len(camera_shape) - 1:
            self.image_layout = "HWC"
        target_height = target_width = None
        if camera_shape:
            if self.image_layout == "CHW" and len(camera_shape) >= 3:
                target_height = camera_shape[-2]
                target_width = camera_shape[-1]
            elif self.image_layout == "HWC" and len(camera_shape) >= 3:
                target_height = camera_shape[0]
                target_width = camera_shape[1]
        if isinstance(target_height, (int, float)) and isinstance(target_width, (int, float)):
            if target_height > 0 and target_width > 0:
                self.image_resize = (int(target_width), int(target_height))
        self.image_resize = (int(self.image_resize[0]), int(self.image_resize[1]))

    def _get_required_obs_keys(self) -> Sequence[str]:
        required = getattr(self.importer, "required_obs_keys", None)
        if required:
            return list(required)
        policy_impl = getattr(self.policy, "policy", None)
        cfg = getattr(policy_impl, "global_config", None)
        keys = getattr(cfg, "all_obs_keys", None)
        if keys:
            return list(keys)
        return list(self.obs_keys) if self.obs_keys else []

    def _infer_stack_dim(self, reference_obs: Optional[Mapping[str, Any]]) -> Optional[int]:
        if self.importer.frame_stack <= 1 or not reference_obs:
            return None
        for value in reference_obs.values():
            arr = np.asarray(value)
            if arr.ndim >= 1 and arr.shape[0] == self.importer.frame_stack:
                return int(self.importer.frame_stack)
        return None

    def _key_is_image(self, key: str) -> bool:
        if key == self.image_key:
            return True
        if ObsUtils.OBS_KEYS_TO_MODALITIES is not None:
            modality = ObsUtils.OBS_KEYS_TO_MODALITIES.get(key)
            if modality in ("rgb", "depth"):
                return True
        return "image" in key.lower()

    @staticmethod
    def _raw_image_shape_from_processed(shape: Tuple[int, ...]) -> Tuple[int, ...]:
        if len(shape) == 3:
            if shape[0] in (1, 3) and shape[-1] not in (1, 3):
                return (shape[1], shape[2], shape[0])
            if shape[-1] in (1, 3):
                return shape
        return shape

    def _placeholder_for_key(
        self,
        key: str,
        *,
        reference_obs: Optional[Mapping[str, Any]] = None,
    ) -> np.ndarray:
        if reference_obs is not None:
            if key in reference_obs:
                return np.zeros_like(reference_obs[key])
            if self._key_is_image(key):
                for ref_key in (self.image_key, "camera_image"):
                    if ref_key in reference_obs:
                        return np.zeros_like(reference_obs[ref_key], dtype=np.uint8)

        shape_meta = self.importer.obs_shapes.get(key) if self.importer.obs_shapes else None
        if shape_meta is not None:
            try:
                shape = tuple(int(s) for s in shape_meta)
            except Exception:
                shape = None
        else:
            shape = None

        if self._key_is_image(key):
            if shape is not None:
                shape = self._raw_image_shape_from_processed(shape)
            else:
                height = int(self.image_resize[1])
                width = int(self.image_resize[0])
                shape = (height, width, 3)
            dtype = np.uint8
        else:
            if shape is None:
                shape = (1,)
            dtype = np.float32

        stack_dim = self._infer_stack_dim(reference_obs)
        if stack_dim is not None and shape and shape[0] != stack_dim:
            shape = (stack_dim,) + tuple(shape)
        return np.zeros(shape, dtype=dtype)

    def _complete_obs_dict(
        self,
        obs: Mapping[str, Any],
        *,
        reference_obs: Optional[Mapping[str, Any]] = None,
        allow_placeholders: bool = False,
    ) -> Dict[str, Any]:
        required_keys = self._get_required_obs_keys()
        if not required_keys:
            return dict(obs)
        missing = [k for k in required_keys if k not in obs]
        if missing:
            if not allow_placeholders:
                raise RuntimeError(
                    f"Observation missing required keys {missing}. "
                    "Provide these fields instead of relying on zero placeholders."
                )
            completed = dict(obs)
            # First try to copy values from a reference observation (e.g., the latest live frame).
            if reference_obs:
                for key in missing:
                    if key in reference_obs:
                        ref_val = reference_obs[key]
                        completed[key] = ref_val.copy() if isinstance(ref_val, np.ndarray) else ref_val
            # Fill anything still missing with zero-shaped placeholders.
            for key in required_keys:
                if key not in completed:
                    completed[key] = self._placeholder_for_key(key, reference_obs=reference_obs)
            return completed
        return dict(obs)

    @staticmethod
    def _assemble_obs_dict(
        image_payload: Optional[np.ndarray],
        pipette_payload: Optional[np.ndarray],
        stage_payload: Optional[np.ndarray],
        resistance_payload: Optional[np.ndarray],
        extra_payload: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        obs_dict: Dict[str, Any] = {}
        if image_payload is not None:
            obs_dict["camera_image"] = image_payload
        if pipette_payload is not None:
            obs_dict["pipette_positions"] = pipette_payload
        if stage_payload is not None:
            obs_dict["stage_positions"] = stage_payload
        if resistance_payload is not None:
            obs_dict["resistance"] = resistance_payload
        if extra_payload:
            obs_dict.update(extra_payload)
        return obs_dict

    def set_goal(self, goal: Any) -> None:
        self.goal = goal if self.goal_required else None

    def get_goal(self) -> Any:
        return self.goal if self.goal_required else None

    def get_required_obs_keys(self) -> Sequence[str]:
        return tuple(self._get_required_obs_keys())

    def _compute_frame_params(self, frame_shape: Tuple[int, int]) -> Optional[Dict[str, float]]:
        if not frame_shape or len(frame_shape) < 2:
            return None
        h, w = int(frame_shape[0]), int(frame_shape[1])
        if h <= 0 or w <= 0:
            return None
        if self.crop_size >= 1.0:
            crop_h = h
            crop_w = w
        else:
            crop_h = max(int(h * self.crop_size), 1)
            crop_w = max(int(w * self.crop_size), 1)
        crop_h = min(crop_h, h)
        crop_w = min(crop_w, w)
        offset_y = max((h - crop_h) // 2, 0)
        offset_x = max((w - crop_w) // 2, 0)
        resize_w, resize_h = self.image_resize
        scale_x = resize_w / crop_w if crop_w else 1.0
        scale_y = resize_h / crop_h if crop_h else 1.0
        return {
            "crop_h": float(crop_h),
            "crop_w": float(crop_w),
            "offset_x": float(offset_x),
            "offset_y": float(offset_y),
            "scale_x": float(scale_x),
            "scale_y": float(scale_y),
        }

    def _prepare_image(
        self,
        image: np.ndarray,
        frame_params: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        frame = np.asarray(image, dtype=np.uint8)
        h, w = frame.shape[:2]
        params = frame_params or self._compute_frame_params((h, w))
        if params is None:
            return frame
        crop_h = int(params["crop_h"])
        crop_w = int(params["crop_w"])
        y0 = int(params["offset_y"])
        x0 = int(params["offset_x"])
        cropped = frame[y0:y0 + crop_h, x0:x0 + crop_w]
        return np.array(
            Image.fromarray(cropped).resize(self.image_resize, Image.BILINEAR),
            dtype=np.uint8,
        )

    def _ensure_rgb_channels(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return np.repeat(frame[..., None], 3, axis=-1)
        if frame.ndim == 3:
            if frame.shape[-1] == 1:
                return np.repeat(frame, 3, axis=-1)
            if frame.shape[-1] > 3:
                return frame[..., :3]
            if frame.shape[0] == 1 and frame.shape[-1] != 3:
                base = frame[0]
                return np.repeat(base[..., None], 3, axis=-1)
        return frame

    def _scale_pipette_for_model(
        self,
        pipette_positions: Optional[np.ndarray],
        frame_params: Optional[Dict[str, float]],
    ) -> Optional[np.ndarray]:
        if pipette_positions is None:
            return None
        scaled = np.asarray(pipette_positions, dtype=np.float32).copy()
        if frame_params is None or scaled.size == 0:
            return scaled
        if scaled.ndim == 0:
            scaled = scaled.reshape(1)
        if scaled.shape[-1] >= 2:
            scale_x = frame_params.get("scale_x", 1.0)
            scale_y = frame_params.get("scale_y", 1.0)
            offset_x = frame_params.get("offset_x", 0.0)
            offset_y = frame_params.get("offset_y", 0.0)
            scaled[..., 0] = (scaled[..., 0] - offset_x) * scale_x
            scaled[..., 1] = (scaled[..., 1] - offset_y) * scale_y
        return scaled

    def _restore_pipette_from_model(
        self,
        pipette_components: np.ndarray,
        frame_params: Optional[Dict[str, float]],
    ) -> np.ndarray:
        restored = np.asarray(pipette_components, dtype=np.float32).copy()
        if frame_params is None or restored.size == 0:
            return restored
        if restored.ndim == 0:
            restored = restored.reshape(1)
        if restored.shape[-1] >= 2:
            scale_x = frame_params.get("scale_x", 1.0)
            scale_y = frame_params.get("scale_y", 1.0)
            offset_x = frame_params.get("offset_x", 0.0)
            offset_y = frame_params.get("offset_y", 0.0)
        if scale_x != 0:
            restored[..., 0] = restored[..., 0] / scale_x + offset_x
        if scale_y != 0:
            restored[..., 1] = restored[..., 1] / scale_y + offset_y
        return restored

    def _save_observation(self, frame_rgb: np.ndarray, pipette_arr: Optional[np.ndarray]) -> None:
        """Persist the current frame (with pipette overlays) under Agent_movement_data for debugging."""
        debug_save_dir = Path(r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\experiments\Data\agent_movement_data")
        run_dir = debug_save_dir / self._debug_run_uid
        debug_save_dir.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "model_config.json"
        if not config_path.exists():
            try:
                importer = getattr(self, "importer", None)
                model_info = {
                    "checkpoint_path": str(getattr(importer, "resolved_ckpt_path", None))
                    if getattr(importer, "resolved_ckpt_path", None) is not None
                    else None,
                    "policy_class": getattr(importer, "policy_class", None),
                    "experiment_name": getattr(importer, "experiment_name", None),
                    "obs_keys": list(getattr(importer, "obs_keys", []) or []),
                    "action_dim": getattr(importer, "action_dim", None),
                    "frame_stack": getattr(importer, "frame_stack", None),
                    "goal_required": getattr(importer, "goal_required", None),
                    "image_key": getattr(importer, "image_key", None),
                    "pipette_key": getattr(importer, "pipette_key", None),
                    "stage_key": getattr(importer, "stage_key", None),
                    "resistance_key": getattr(importer, "resistance_key", None),
                    "obs_shapes": getattr(importer, "obs_shapes", None),
                    "image_resize": list(self.image_resize) if self.image_resize else None,
                    "image_layout": self.image_layout,
                    "crop_size": self.crop_size,
                    "debug_run_uid": self._debug_run_uid,
                    "created_at": datetime.now().isoformat(),
                }
                config_path.write_text(json.dumps(model_info, indent=2, default=str))
            except Exception:
                # Config export is best-effort; failures should not block saving frames.
                pass
        debug_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        save_frame = np.clip(frame_rgb, 0, 255).astype(np.uint8)
        debug_image = Image.fromarray(save_frame)
        overlay_points = []

        pipette_debug = pipette_arr
        if pipette_debug is not None:
            pipette_debug = np.asarray(pipette_debug, dtype=np.float32)
            if pipette_debug.size > 0:
                if pipette_debug.ndim == 1:
                    pipette_debug = pipette_debug.reshape(1, -1)
                elif pipette_debug.ndim > 2:
                    pipette_debug = pipette_debug.reshape(-1, pipette_debug.shape[-1])
                if pipette_debug.ndim >= 2 and pipette_debug.shape[-1] >= 2:
                    height, width = save_frame.shape[:2]
                    max_x = max(width - 1.0, 0.0)
                    max_y = max(height - 1.0, 0.0)
                    for point in pipette_debug:
                        if point.shape[0] < 2 or not np.all(np.isfinite(point[:2])):
                            continue
                        x = float(np.clip(point[0], 0.0, max_x))
                        y = float(np.clip(point[1], 0.0, max_y))
                        overlay_points.append((x, y))

        if overlay_points:
            draw = ImageDraw.Draw(debug_image)
            radius = max(2, int(min(debug_image.size) * 0.02))
            for x, y in overlay_points:
                bbox = (x - radius, y - radius, x + radius, y + radius)
                draw.ellipse(bbox, fill=(255, 0, 0), outline=(255, 255, 255))

        debug_image.save(run_dir / f"camera_image_{debug_timestamp}.png")

    def _prepare_observation(
        self,
        observation: Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Mapping[str, Any]],
        *,
        is_demo: bool = False,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[str, Any]]]:
        extras: Dict[str, Any] = {}
        if isinstance(observation, Mapping):
            obs_map = dict(observation)
            pipette = obs_map.get(self.pipette_key, obs_map.get("pipette_positions"))
            stage = obs_map.get(self.stage_key, obs_map.get("stage_positions"))
            image = obs_map.get(self.image_key, obs_map.get("camera_image"))
            resistance = obs_map.get(self.resistance_key, obs_map.get("resistance"))
            reserved = {
                self.pipette_key,
                "pipette_positions",
                self.stage_key,
                "stage_positions",
                self.image_key,
                "camera_image",
                self.resistance_key,
                "resistance",
            }
            extras.update({k: v for k, v in obs_map.items() if k not in reserved})
        else:
            pipette, stage, image, resistance = observation
        frame_params: Optional[Dict[str, float]] = None
        self._last_frame_params = None
        required_keys = set(self._get_required_obs_keys())
        uses_image = self.image_key in required_keys

        image_payload: Optional[np.ndarray] = None
        image_arr = None if image is None else np.asarray(image)
        if image_arr is not None:
            if not is_demo and uses_image:
                frame_params = self._compute_frame_params(image_arr.shape[:2])
                prepared_image = self._prepare_image(image_arr, frame_params) if frame_params else image_arr
                self._last_frame_params = frame_params
            else:
                prepared_image = image_arr
            frame_rgb = self._ensure_rgb_channels(prepared_image)
            if self.image_key == "camera_image":
                image_payload = frame_rgb.astype(np.uint8, copy=False)
            elif self.image_key in self.obs_keys:
                extras[self.image_key] = frame_rgb.astype(np.uint8, copy=False)

        pipette_payload: Optional[np.ndarray] = None
        pipette_arr = None if pipette is None else np.asarray(pipette, dtype=np.float32)
        if pipette_arr is not None:
            if frame_params is not None:
                pipette_arr = self._scale_pipette_for_model(pipette_arr, frame_params)
            if pipette_arr is not None and pipette_arr.ndim >= 1 and pipette_arr.shape[-1] > 0:
                self._pipette_action_dim = int(pipette_arr.shape[-1])
            if self.pipette_key == "pipette_positions":
                pipette_payload = np.asarray(pipette_arr, dtype=np.float32)
            elif self.pipette_key in self.obs_keys:
                extras[self.pipette_key] = np.asarray(pipette_arr, dtype=np.float32)

        # Save debug frames with pipette overlays to Agent_movement_data (parity with ONNX agent)
        if image_arr is not None and not is_demo:
            try:
                self._save_observation(frame_rgb, pipette_arr)
            except Exception:
                # Debug saving should never break inference flow
                pass

        stage_payload: Optional[np.ndarray] = None
        if stage is not None:
            stage_arr = np.asarray(stage, dtype=np.float32)
            if self.stage_key == "stage_positions":
                stage_payload = stage_arr
            elif self.stage_key in self.obs_keys:
                extras[self.stage_key] = stage_arr

        resistance_payload: Optional[np.ndarray] = None
        if resistance is not None:
            res_arr = np.asarray(resistance, dtype=np.float32)
            if self.resistance_key == "resistance":
                resistance_payload = res_arr
            elif self.resistance_key in self.obs_keys:
                extras[self.resistance_key] = res_arr

        extra_payload = extras or None
        return image_payload, pipette_payload, stage_payload, resistance_payload, extra_payload

    def _process_action(self, action: np.ndarray) -> np.ndarray:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if self._last_frame_params and self._pipette_action_dim:
            pip_dim = min(int(self._pipette_action_dim), arr.shape[0])
            if pip_dim > 0:
                pip_slice = slice(arr.shape[0] - pip_dim, arr.shape[0])
                pipette_components = np.asarray(arr[pip_slice], dtype=np.float32)
                restored = self._restore_pipette_from_model(pipette_components, self._last_frame_params)
                arr[pip_slice] = restored.reshape(pip_dim)
        return arr.astype(np.float32, copy=False)

    def _format_goal(
        self,
        goal: Any,
        *,
        reference_obs: Optional[Mapping[str, Any]] = None,
        is_demo: bool = False,
    ) -> Optional[Dict[str, Any]]:
        if not self.goal_required or goal is None:
            return None
        goal_dict: Optional[Dict[str, Any]] = None
        if isinstance(goal, Mapping):
            goal_dict = dict(goal)
        elif isinstance(goal, (tuple, list)) and len(goal) == 4:
            image_payload, pipette_payload, stage_payload, resistance_payload, extra_payload = self._prepare_observation(
                goal, is_demo=is_demo
            )
            goal_dict = self._assemble_obs_dict(
                image_payload,
                pipette_payload,
                stage_payload,
                resistance_payload,
                extra_payload,
            )
        else:
            arr = np.asarray(goal, dtype=np.float32).reshape(-1)
            key = self.pipette_key or "pipette_positions"
            goal_dict = {key: arr}
        return self._complete_obs_dict(
            goal_dict,
            reference_obs=reference_obs,
            allow_placeholders=self.allow_goal_placeholders,
        )

    def inference(self, observation, goal=None, is_demo: bool = False):
        image_payload, pipette_payload, stage_payload, resistance_payload, extra_payload = self._prepare_observation(
            observation, is_demo=is_demo
        )
        self.env.update_observation(
            camera_image=image_payload,
            pipette_positions=pipette_payload,
            stage_positions=stage_payload,
            resistance=resistance_payload,
            extra=extra_payload,
        )
        obs_ready: Dict[str, Any]
        if not self._started:
            obs_ready = self.env.reset()
            if hasattr(self.policy, "start_episode"):
                self.policy.start_episode()
            self._started = True
        else:
            # If frame stacking is enabled, maintain wrapper history with the latest push
            try:
                from robomimic.envs.wrappers import FrameStackWrapper
                if isinstance(self.env, FrameStackWrapper):
                    latest = self.env.env.get_observation()
                    if self.env.obs_history is None:
                        self.env.obs_history = self.env._get_initial_obs_history(latest)
                    else:
                        for k in latest:
                            self.env.obs_history[k].append(latest[k][None])
                    obs_ready = self.env._get_stacked_obs_from_history()
                else:
                    obs_ready = self.env.get_observation()
            except Exception:
                obs_ready = self.env.get_observation()
        obs_ready = self._complete_obs_dict(obs_ready, reference_obs=obs_ready)
        self._last_obs = obs_ready
        goal_payload = self._format_goal(goal, reference_obs=obs_ready, is_demo=is_demo)
        if goal_payload is None:
            goal_payload = self._format_goal(self.goal, reference_obs=obs_ready, is_demo=is_demo)
        if self.goal_required and goal_payload is None:
            raise RuntimeError("Goal is required by the loaded policy. Call set_goal() or pass goal to inference().")
        act_raw = self.policy(ob=obs_ready, goal=goal_payload)
        action = ModelImporter._extract_policy_action(act_raw)
        processed_action = self._process_action(action)
        self._last_obs, _, _, _ = self.env.step(processed_action)
        return processed_action.astype(np.float32, copy=False)


class DemoReplayAgent:
    """
    Replay a cached sequence of actions without using live observations.
    """

    def __init__(self, actions: Optional[np.ndarray] = None) -> None:
        self._actions: Optional[np.ndarray] = None
        self._cursor: int = 0
        self._image_size: Optional[Tuple[int, int]] = None  # (height, width)
        if actions is not None:
            self.load_actions(actions)

    def load_actions(self, actions: np.ndarray) -> None:
        replay = np.asarray(actions, dtype=np.float32)
        if replay.ndim == 1:
            replay = replay.reshape(1, -1)
        if replay.size == 0:
            raise ValueError("DemoReplayAgent received an empty action array")
        self._actions = replay.astype(np.float32, copy=False)
        self._cursor = 0

    def set_image_size(self, image_shape: Tuple[int, int]) -> None:
        if image_shape is None or len(image_shape) != 2:
            raise ValueError("Image shape must be a (height, width) tuple")
        self._image_size = (int(image_shape[0]), int(image_shape[1]))

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        if self._image_size is None:
            return action.astype(np.float32, copy=False)
        height, width = self._image_size
        scaled = np.asarray(action, dtype=np.float32).copy()
        if scaled.size >= 2:
            scaled[0] = scaled[0] * (width / 85.0)
            scaled[1] = scaled[1] * (height / 85.0)
        if scaled.size >= 4:
            scaled[2] = scaled[2] * (width / 85.0)
            scaled[3] = scaled[3] * (height / 85.0)
        return scaled.astype(np.float32, copy=False)

    def inference(self, observation, goal=None, is_demo: bool = False):
        if self._actions is None:
            raise RuntimeError("DemoReplayAgent requires actions to be loaded before inference")
        index = min(self._cursor, self._actions.shape[0] - 1)
        action = self._actions[index]
        if self._cursor < self._actions.shape[0]:
            self._cursor += 1
        return self._scale_action(action)


class PipetteFinder(ModelInferencer):
    def __init__(self, model_path: Optional[Union[str, Path]] = r"patcherbot\deepLearning\patchModel\Agents\PipetteFinder", **kwargs) -> None:
        kwargs.setdefault("model_type", "PipetteFinder")
        importer = ModelImporter(model_path, **kwargs)
        super().__init__(importer)


class CellHunter(ModelInferencer):
    def __init__(self, model_path: Optional[Union[str, Path]] = r"patcherbot\deepLearning\patchModel\Agents\CellHunter", **kwargs) -> None:
        kwargs.setdefault("model_type", "CellHunter")
        importer = ModelImporter(model_path, **kwargs)
        super().__init__(importer)


class GigaSealer(ModelInferencer):
    def __init__(self, model_path: Optional[Union[str, Path]] = r"patcherbot\deepLearning\patchModel\Agents\GigaSealer", **kwargs) -> None:
        kwargs.setdefault("model_type", "GigaSealer")
        importer = ModelImporter(model_path, **kwargs)
        super().__init__(importer)


class Burglar(ModelInferencer):
    def __init__(self, model_path: Optional[Union[str, Path]] = r"patcherbot\deepLearning\patchModel\Agents\Burglar", **kwargs) -> None:
        kwargs.setdefault("model_type", "Burglar")
        importer = ModelImporter(model_path, **kwargs)
        super().__init__(importer)
