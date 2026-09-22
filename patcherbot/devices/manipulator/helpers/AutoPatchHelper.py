import time
import json
import pickle
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from patcherbot.deepLearning.autoPatcher import CellHunter, GigaSealer, Burglar, PipetteFinder


DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().parents[4] / "experiments" / "Datasets" / "average_calibration_full.pickle"



class AutoPatchHelper:
    """
    A helper class to aid with different stages of the auto patching process.
    Neuron Hunting
    Gigasealing
    Break in
    """

    def __init__(
        self,
        *,
        calibration_enabled: bool = False,
        calibrate_inputs: Optional[bool] = None,
        calibrate_outputs: Optional[bool] = None,
    ):
        """
        Initialize the AutoPatchHelper with model instances and calibration settings.

        Args:
            calibration_enabled (bool):
                Global default flag to enable calibration for both inputs and outputs.
                Used when specific input/output flags are not provided.

            calibrate_inputs (Optional[bool]):
                Whether to apply calibration to model inputs (microns → pixels).
                If None, inherits from calibration_enabled.

            calibrate_outputs (Optional[bool]):
                Whether to apply calibration to model outputs (pixels → microns).
                If None, inherits from calibration_enabled.
        """
        self.hunter = CellHunter()
        self.finder = PipetteFinder()
        self.gigasealer = GigaSealer()
        self.burglar = Burglar()
        self.poslist = []
        self.hunterh0 = None
        self.hunterc0 = None
        self.finderh0 = None
        self.finderc0 = None
        self._hunter_state_snapshot = None
        self._finder_state_snapshot = None
        self.calibration_path: Optional[Path] = None
        self._calibration = {"pipette": None, "stage": None}
        default_cal = bool(calibration_enabled)
        inputs_flag = default_cal if calibrate_inputs is None else bool(calibrate_inputs)
        outputs_flag = default_cal if calibrate_outputs is None else bool(calibrate_outputs)
        self._calibration_usage = {"inputs": inputs_flag, "outputs": outputs_flag}
        self.cal_enabled = inputs_flag or outputs_flag
        self._model_meta = {}
        self._axis_meta = {}

    def _actor_for(self, which: str):
        """
        Return the model instance corresponding to a given stage name.

        Args:
            which (str):
                Model identifier ("hunt", "find_pipette", "gigaseal", "break_in").

        Returns:
            Optional[Any]:
                The corresponding model instance, or None if not found.
        """
        return {
            "hunt": self.hunter,
            "find_pipette": self.finder,
            "gigaseal": self.gigasealer,
            "break_in": self.burglar,
        }.get(which)

    def _model_dims(self, which: str) -> Tuple[int, int, Optional[int]]:
        """
        Retrieve and cache axis dimension information for a model.

        Args:
            which (str):
                Model identifier.

        Returns:
            Tuple[int, int, Optional[int]]:
                Tuple containing (pipette_dim, stage_dim, optional_extra_dim).
        """
        cached = self._axis_meta.get(which)
        if cached is not None:
            return cached
        actor = self._actor_for(which)
        if actor is None:
            return (3, 3, None)
        dims = actor._axis_dims()
        self._axis_meta[which] = dims
        return dims

    def _trim_axes_for(self, pip, stage, which: str):
        """
        Resize and pad pipette and stage vectors to match model input dimensions.

        Args:
            pip (Sequence[float]):
                Pipette position vector.
            stage (Sequence[float]):
                Stage position vector.
            which (str):
                Model identifier.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                Trimmed and padded pipette and stage vectors.
        """
        pip_dim, stage_dim, _ = self._model_dims(which)
        pip_arr = np.asarray(pip, np.float32).reshape(-1)
        stage_arr = np.asarray(stage, np.float32).reshape(-1)
        if pip_dim:
            if pip_arr.size < pip_dim:
                pip_arr = np.pad(pip_arr, (0, pip_dim - pip_arr.size))
            pip_arr = pip_arr[:pip_dim]
        else:
            pip_arr = np.zeros((0,), dtype=np.float32)
        if stage_dim:
            if stage_arr.size < stage_dim:
                stage_arr = np.pad(stage_arr, (0, stage_dim - stage_arr.size))
            stage_arr = stage_arr[:stage_dim]
        else:
            stage_arr = np.zeros((0,), dtype=np.float32)
        return pip_arr.astype(np.float32), stage_arr.astype(np.float32)

    def _pad_output(self, stage_vec: np.ndarray, pip_vec: np.ndarray) -> np.ndarray:
        """
        Combine stage and pipette outputs into a fixed-length vector.

        Args:
            stage_vec (np.ndarray):
                Stage output vector.
            pip_vec (np.ndarray):
                Pipette output vector.

        Returns:
            np.ndarray:
                Combined output vector of length at least 6.
        """
        combined = np.concatenate([stage_vec, pip_vec])
        if combined.size < 6:
            combined = np.pad(combined, (0, 6 - combined.size), constant_values=0.0)
        return combined

    def load_calibration(self, path: Optional[Union[str, Path]] = None):
        """
        Load calibration data and cache the affine transforms.
        
        Args:
            path (Optional[Union[str, Path]]):
                Path to calibration file (JSON or pickle). Uses default if None.

        Returns:
            Dict[str, Any]:
                Dictionary containing calibration matrices and offsets.

        Raises:
            FileNotFoundError:
                If the calibration file does not exist.
            ValueError:
                If the calibration format is invalid.
        """
        candidate = path or self.calibration_path or DEFAULT_CALIBRATION_PATH
        candidate = Path(candidate).expanduser()
        if not candidate.exists():
            raise FileNotFoundError(f"Calibration file not found: {candidate}")

        if candidate.suffix.lower() == ".json":
            with open(candidate, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        else:
            with open(candidate, "rb") as fh:
                payload = pickle.load(fh)

        if not isinstance(payload, dict):
            raise ValueError(f"Unsupported calibration format in {candidate}")

        def _coerce(entry):
            if entry is None or "M" not in entry:
                return None
            matrix = np.asarray(entry["M"], dtype=np.float64)
            if matrix.ndim != 2 or matrix.size == 0:
                raise ValueError("Calibration matrix must be a non-empty 2-D array")
            offset = np.asarray(entry.get("r0", np.zeros(matrix.shape[0])), dtype=np.float64).reshape(-1)
            if offset.size < matrix.shape[0]:
                offset = np.pad(offset, (0, matrix.shape[0] - offset.size), constant_values=0.0)
            elif offset.size > matrix.shape[0]:
                offset = offset[:matrix.shape[0]]
            matrix_inv = np.linalg.pinv(matrix)
            offset_inv = -matrix_inv @ offset
            return {"M": matrix, "Minv": matrix_inv, "r0": offset, "r0_inv": offset_inv}

        manip_entry = payload.get("manip") or payload.get("pipette") or payload
        stage_entry = payload.get("stage")

        self._calibration["pipette"] = _coerce(manip_entry)
        self._calibration["stage"] = _coerce(stage_entry) if stage_entry is not None else None
        self.calibration_path = candidate

        return {
            "pipette": {"M": self._calibration["pipette"]["M"], "r0": self._calibration["pipette"]["r0"]} if self._calibration["pipette"] else None,
            "stage": {"M": self._calibration["stage"]["M"], "r0": self._calibration["stage"]["r0"]} if self._calibration["stage"] else None,
            "path": candidate,
        }

    def set_calibration_usage(self, *, inputs: Optional[bool] = None, outputs: Optional[bool] = None) -> None:
        """
        Toggle calibration independently for model inputs and outputs.
        
        Args:
            inputs (Optional[bool]):
                Whether to apply calibration to inputs.
            outputs (Optional[bool]):
                Whether to apply calibration to outputs.
        """
        if inputs is not None:
            self._calibration_usage["inputs"] = bool(inputs)
        if outputs is not None:
            self._calibration_usage["outputs"] = bool(outputs)
        self.cal_enabled = self._calibration_usage["inputs"] or self._calibration_usage["outputs"]

    @property
    def use_calibrated_inputs(self) -> bool:
        """
        Check if input calibration is enabled.

        Returns:
            bool:
                True if calibration is applied to inputs.
        """
        return self._calibration_usage["inputs"]

    @use_calibrated_inputs.setter
    def use_calibrated_inputs(self, enabled: bool) -> None:
        """
        Enable or disable input calibration.

        Args:
            enabled (bool):
                Desired calibration state.
        """
        self.set_calibration_usage(inputs=enabled)

    @property
    def use_calibrated_outputs(self) -> bool:
        """
        Check if output calibration is enabled.

        Returns:
            bool:
                True if calibration is applied to outputs.
        """
        return self._calibration_usage["outputs"]

    @use_calibrated_outputs.setter
    def use_calibrated_outputs(self, enabled: bool) -> None:
        """
        Enable or disable output calibration.

        Args:
            enabled (bool):
                Desired calibration state.
        """
        self.set_calibration_usage(outputs=enabled)

    def apply_calibration(self, values: Union[Sequence[float], Tuple[Sequence[float], Sequence[float]]], *, direction: str = "to_pixels", split: bool = False, ignore_offsets: Optional[dict] = None):
        """
        Convert coordinates between microns and pixels using cached calibration matrices.
        
        Args:
            values (Union[Sequence[float], Tuple[Sequence[float], Sequence[float]]]):
                Combined or separate stage and pipette coordinates.
            direction (str):
                Conversion direction ("to_pixels" or "to_microns").
            split (bool):
                Whether to return stage and pipette separately.
            ignore_offsets (Optional[dict]):
                Flags to ignore offsets for specific components.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
                Converted coordinates.

        Raises:
            ValueError:
                If direction is invalid or input format is incorrect.
        """
        mode = direction.lower()
        if mode in {"to_pixels", "microns_to_pixels", "um_to_pixels", "forward"}:
            forward = True
        elif mode in {"to_microns", "pixels_to_microns", "pixel_to_um", "pixels_to_um", "inverse"}:
            forward = False
        else:
            raise ValueError(f"Unsupported direction '{direction}'. Expected 'to_pixels' or 'to_microns'.")

        def _split(vals):
            if isinstance(vals, (tuple, list)) and len(vals) == 2:
                stage_vec = np.asarray(vals[0], dtype=np.float64).reshape(-1)
                pip_vec = np.asarray(vals[1], dtype=np.float64).reshape(-1)
                return stage_vec, pip_vec, None, True
            arr = np.asarray(vals, dtype=np.float64).reshape(-1)
            if arr.size < 6:
                raise ValueError("Expected at least six values combining stage and pipette coordinates.")
            return arr[:3], arr[3:6], arr, False

        def _merge(stage_vec, pip_vec, combined, want_split):
            if combined is None:
                if want_split:
                    return stage_vec.astype(np.float32), pip_vec.astype(np.float32)
                return np.concatenate([stage_vec, pip_vec]).astype(np.float32)
            merged = combined.copy()
            merged[:stage_vec.shape[0]] = stage_vec
            merged[3:3 + pip_vec.shape[0]] = pip_vec
            if want_split:
                return merged[:stage_vec.shape[0]].astype(np.float32), merged[3:3 + pip_vec.shape[0]].astype(np.float32)
            return merged.astype(np.float32)

        stage_vec, pip_vec, combined, paired = _split(values)
        stage_ignore = bool(ignore_offsets.get("stage")) if ignore_offsets else False
        pip_ignore = bool(ignore_offsets.get("pipette")) if ignore_offsets else False
        if not self.cal_enabled:
            return _merge(stage_vec, pip_vec, combined, split or paired)
        if self._calibration["pipette"] is None:
            try:
                self.load_calibration()
            except FileNotFoundError:
                return _merge(stage_vec, pip_vec, combined, split or paired)
        pip_entry = self._calibration.get("pipette")
        stage_entry = self._calibration.get("stage")

        def _apply(entry, vec, ignore_offset):
            if entry is None:
                return vec.astype(np.float32)
            vec64 = np.asarray(vec, dtype=np.float64).reshape(-1)
            if vec64.size == 0:
                return vec64.astype(np.float32)
            out = vec64.copy()
            if forward:
                matrix = entry["M"]
                offset = np.zeros(matrix.shape[0], dtype=np.float64) if ignore_offset else entry["r0"]
                out_dim = matrix.shape[0]
                in_dim = min(matrix.shape[1], vec64.size)
                out[:out_dim] = matrix @ vec64[:in_dim] + offset[:out_dim]
            else:
                matrix = entry["Minv"]
                offset = np.zeros(matrix.shape[0], dtype=np.float64) if ignore_offset else entry["r0_inv"]
                rows = min(entry["M"].shape[0], vec64.size)
                out_dim = matrix.shape[0]
                out[:out_dim] = matrix @ vec64[:rows] + offset[:out_dim]
            return out.astype(np.float32)

        stage_result = _apply(stage_entry, stage_vec, stage_ignore)
        pip_result = _apply(pip_entry, pip_vec, pip_ignore)
        return _merge(stage_result, pip_result, combined, split or paired)

    def hunt(self, model_input):
        """
        Preprocess + handshake:
          • Wrapper models (obs::/goal::): build {"obs": (...), "goal": partial or omitted}
            and pass RAW HWC image + raw numerics (normalization handled inside ONNX).
          • Legacy models: pass (pip, stage, img, res); CellHunter will normalize/crop/stack.
        
        Args:
        model_input (Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
            Tuple containing (pipette_positions, stage_positions, image, resistance).

        Returns:
            List[float]:
                Predicted movement vector in microns.
        """
        # Ensure model is loaded and wrapper flags are known
        if self.hunter.session is None:
            self.prepare_model("hunt")

        info = self._model_meta.get("hunt")
        if info is None:
            info = self.hunter.identify_model()
            self._model_meta["hunt"] = info
        uses_wrapper = info.get("uses_wrapper", False)
        has_goal = info.get("has_goal", False)

        # Coerce observation (mild normalization of types/shapes only)
        pip_raw, stage_raw, img, res = model_input
        pip_vec, stage_vec = self._trim_axes_for(pip_raw, stage_raw, "hunt")
        if self._calibration_usage.get("inputs", False):
            stage_px, pip_px = self.apply_calibration((stage_vec, pip_vec), direction="to_pixels", split=True)
        else:
            stage_px = np.asarray(stage_vec, dtype=np.float32)
            pip_px = np.asarray(pip_vec, dtype=np.float32)
        res = np.asarray(res, np.float32).reshape(-1)
        img = np.asarray(img)
        if img.ndim == 2:  # gray → 3‑chan
            img = np.stack([img]*3, axis=-1)

        if uses_wrapper:
            payload = {"obs": (pip_px, stage_px, img, res)}
            if has_goal and hasattr(self, "_goal") and len(self._goal) > 0:
                # Partial goal OK (e.g., no image) — CellHunter fills missing keys from obs
                payload["goal"] = self._goal
            model_payload = payload
        else:
            # Legacy model — give tuple; CellHunter does CHW+resize+[0,1] and stacking
            model_payload = (pip_px, stage_px, img, res)
        if self._hunter_state_snapshot is not None:
            self.hunter.set_state_snapshot(self._hunter_state_snapshot)
        # print(f"model payload prepared {type(model_payload)}")
        pos, h0_out, c0_out = self.hunter.inference(model_payload, self.hunterh0, self.hunterc0)
        snapshot = self.hunter.get_state_snapshot()
        self._hunter_state_snapshot = snapshot
        if snapshot:
            self.hunterh0 = snapshot.get("h0")
            self.hunterc0 = snapshot.get("c0")
        else:
            self.hunterh0, self.hunterc0 = h0_out, c0_out
        # print(f"model inference returned pos {pos}")
        pos = np.asarray(pos, np.float32).reshape(-1)
        pip_dim, stage_dim, _ = self._model_dims("hunt")
        stage_part = pos[:stage_dim] if stage_dim else np.zeros((0,), dtype=np.float32)
        pip_part = pos[stage_dim:stage_dim + pip_dim]
        if self._calibration_usage.get("outputs", False):
            stage_um, pip_um = self.apply_calibration(
                (stage_part, pip_part),
                direction="to_microns",
                split=True,
                ignore_offsets={"pipette": True},
            )
        else:
            stage_um = np.asarray(stage_part, dtype=np.float32)
            pip_um = np.asarray(pip_part, dtype=np.float32)
        pos_um = self._pad_output(stage_um, pip_um)
        return self.clamp_positions(pos_um)


    def find_pipette(self, model_input):
        """
        Preprocess + handshake for pipette localisation.
          - Wrapper models (obs::/goal::): build {"obs": (...)} and pass raw HWC image.
          - Legacy models: pass (pip, stage, img); PipetteFinder normalises internally.
        
        Args:
            model_input (Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
                Tuple containing (pipette_positions, stage_positions, image, resistance).

        Returns:
            List[float]:
                Predicted pipette position vector in microns.
        """
        if self.finder.session is None:
            self.prepare_model("find_pipette")

        info = self._model_meta.get("find_pipette")
        if info is None:
            info = self.finder.identify_model()
            self._model_meta["find_pipette"] = info
        uses_wrapper = info.get("uses_wrapper", False)
        has_goal = info.get("has_goal", False)

        pip_raw, stage_raw, img, res = model_input
        pip_vec, stage_vec = self._trim_axes_for(pip_raw, stage_raw, "find_pipette")
        if self._calibration_usage.get("inputs", False):
            stage_px, pip_px = self.apply_calibration((stage_vec, pip_vec), direction="to_pixels", split=True)
        else:
            stage_px = np.asarray(stage_vec, dtype=np.float32)
            pip_px = np.asarray(pip_vec, dtype=np.float32)
        
        img = np.asarray(img)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)

        if uses_wrapper:
            payload = {"obs": (pip_px, stage_px, img)}
            goal = getattr(self, "_finder_goal", None)
            if has_goal and goal:
                payload["goal"] = goal
            model_payload = payload
        else:
            model_payload = (pip_px, stage_px, img)

            # print(f"model st payload:{stage_px}")
        if self._finder_state_snapshot is not None:
            self.finder.set_state_snapshot(self._finder_state_snapshot)
        # print(f"pipette finder payload prepared {type(model_payload)}")
        # print(f"model pip payload:{pip_px}")
        # print(f"model stage payload:{stage_px}")
        # save_path = r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\patcherbot\temp\test_image"
        # plot and save image
        # plt.imshow(img)
        # plt.savefig(save_path)
        # cv2.imwrite(save_path, img)
        pos, h0_out, c0_out = self.finder.inference(model_payload, self.finderh0, self.finderc0)
        snapshot = self.finder.get_state_snapshot()
        self._finder_state_snapshot = snapshot
        if snapshot:
            self.finderh0 = snapshot.get("h0")
            self.finderc0 = snapshot.get("c0")
        else:
            self.finderh0, self.finderc0 = h0_out, c0_out
        # print(f"pipette finder inference returned pos in pixels {pos}")
        pos = np.asarray(pos, np.float32).reshape(-1)
        pip_dim, stage_dim, _ = self._model_dims("find_pipette")
        stage_part = pos[:stage_dim] if stage_dim else np.zeros((0,), dtype=np.float32)
        pip_part = pos[stage_dim:stage_dim + pip_dim]
        if self._calibration_usage.get("outputs", False):
            stage_um, pip_um = self.apply_calibration(
                (stage_part, pip_part),
                direction="to_microns",
                split=True,
                ignore_offsets={"pipette": True},
            )
        else:
            stage_um = np.asarray(stage_part, dtype=np.float32)
            pip_um = np.asarray(pip_part, dtype=np.float32)
        pos_um = self._pad_output(stage_um, pip_um)
        # print(f"pipette finder inference returned pos in microns {pos_um}")
        pip_disp = pip_um - pip_vec[:pip_dim]
        # print(f"pipette finder inference displacement pos in microns {pip_disp}")
        return self.clamp_positions(pos_um)

    def gigaseal(self,mode,type,input):
       """
        Placeholder for gigaseal stage inference.

        Args:
            mode (Any):
                Execution mode.
            type (Any):
                Model or operation type.
            input (Any):
                Input data.
        """
       pass
    def breakin(self,mode,type,input):
        """
        Placeholder for break-in stage inference.

        Args:
            mode (Any):
                Execution mode.
            type (Any):
                Model or operation type.
            input (Any):
                Input data.
        """
        pass

    def pathplan(self, pos):
        """
        Convert a list of position samples (e.g. [[x1, y1, z1], [x2, y2, z2], …])
        into a flat velocity profile suitable for the manipulator.

        The method:
        • Uses a fixed time step (dt = 10 ms).
        • Computes velocities with central differences (forward/backward for the edges).
        • Duplicates the 3‑D velocity into the 6‑output format.
        • Limits each 3‑D velocity vector to `max_speed`.
        
        Args:
            pos (Sequence[Sequence[float]]):
                List of 3D positions over time.

        Returns:
            List[float]:
                Flattened velocity vector sequence.
        """
        # Guard against too few points
        if len(pos) < 2:
            return [0, 0, 0, 0, 0, 0]

        pos_arr = np.array(pos, dtype=float)
        dt = 0.01  # 10 ms per sample

        # Initialise velocity array
        vel_arr = np.zeros((len(pos_arr), 3))

        # Central difference for middle points
        vel_arr[1:-1] = (pos_arr[2:] - pos_arr[:-2]) / (2 * dt)

        # Forward difference for the first point
        vel_arr[0] = (pos_arr[1] - pos_arr[0]) / dt

        # Backward difference for the last point
        vel_arr[-1] = (pos_arr[-1] - pos_arr[-2]) / dt

        # Flatten and duplicate each 3‑D vector into the 6‑output format
        vel_flat = []
        for v in vel_arr:
            vel_flat.extend(v.tolist())

        # Clamp each 3‑D velocity to a maximum speed
        max_speed = 5.0  # adjust as needed
        for i in range(0, len(vel_flat), 3):
            v3 = np.array(vel_flat[i:i + 3])
            speed = np.linalg.norm(v3)
            if speed > max_speed:
                scale = max_speed / speed
                vel_flat[i:i + 3] = (v3 * scale).tolist()

        return vel_flat

    def clamp_positions(self, positions, max_distance=2000.0):
        """
        Clamp position values to a maximum magnitude.

        Args:
            positions (Sequence[float]):
                Position vector.
            max_distance (float):
                Maximum allowed absolute value.

        Returns:
            List[float]:
                Clamped position values.
        """
        # ensure 1-D float array
        arr = np.asarray(positions, dtype=float).ravel()
        # clamp both positive and negative to ±max_distance
        arr = np.clip(arr, -max_distance, max_distance)
        return arr.tolist()  # <-- returns a plain Python list


    def prime_model(self,model,model_input):
        """
        Prepare a given model for accurate inferencing. requires loading and preloading inputs to provide an accurate prediction set

        Args:
            model (str):
                Model identifier.
            model_input (Sequence[Any]):
                Sequence of input observations.
        """
        if model == 'hunt':
            self.hunter.load_model()
            modelactor = self.hunter
        elif model == 'find_pipette':
             self.finder.load_model()
             modelactor = self.finder
        elif model == 'gigaseal':
             self.gigasealer.load_model()
             modelactor=self.gigasealer
        elif model == 'break_in':
             self.burglar.load_model()
             modelactor = self.burglar

        modelactor.reset_state()

        predlist = []
        h0list = []
        c0list = []
        for i in range(len(model_input)):
            if i == 0:
                h0 = None
                c0 = None
            else:
                h0 = h0list[i-1]
                c0 = c0list[i-1]

            pred, h0, c0 = modelactor.inference(model_input[i], h0, c0)

            predlist.append(pred)
            h0list.append(h0)
            c0list.append(c0)
            # print(f"observation primed {i}")
        snapshot = modelactor.get_state_snapshot()
        if model == 'find_pipette':
            self._finder_state_snapshot = snapshot
            self.finderh0 = snapshot.get("h0") if snapshot else h0
            self.finderc0 = snapshot.get("c0") if snapshot else c0
        elif model == 'hunt':
            self._hunter_state_snapshot = snapshot
            self.hunterh0 = snapshot.get("h0") if snapshot else h0
            self.hunterc0 = snapshot.get("c0") if snapshot else c0


    def prepare_model(self, which="hunt", *, onnx_path=None, providers=None):
        """
        Loads the selected model and records whether it uses new wrapper IO (obs::/goal::).
        Compatible with your call site: self.autopatchhelper.prepare_model("hunt")
        You may optionally pass an explicit `onnx_path` to control which policy is loaded.
        
        Args:
            which (str):
                Model identifier.
            onnx_path (Optional[str]):
                Path to ONNX model file.
            providers (Optional[Any]):
                Execution providers.

        Returns:
            AutoPatchHelper:
                Instance with initialized model.

        Raises:
            ValueError:
                If model identifier is unknown.
        """
        if which == "hunt":
            self.hunter.load_model(onnx_path, providers=providers)
            self.hunter.reset_state()
            self._hunter_state_snapshot = self.hunter.get_state_snapshot()
            self.hunterh0 = self._hunter_state_snapshot.get("h0") if self._hunter_state_snapshot else None
            self.hunterc0 = self._hunter_state_snapshot.get("c0") if self._hunter_state_snapshot else None
            info = self.hunter.identify_model()
            self._model_meta["hunt"] = info
            self._axis_meta["hunt"] = self.hunter._axis_dims()
        elif which == "find_pipette":
            self.finder.load_model(onnx_path, providers=providers)
            self.finder.reset_state()
            self._finder_state_snapshot = self.finder.get_state_snapshot()
            self.finderh0 = self._finder_state_snapshot.get("h0") if self._finder_state_snapshot else None
            self.finderc0 = self._finder_state_snapshot.get("c0") if self._finder_state_snapshot else None
            info = self.finder.identify_model()
            self._model_meta["find_pipette"] = info
            self._axis_meta["find_pipette"] = self.finder._axis_dims()
        elif which == "gigaseal":
            self.gigasealer.load_model(onnx_path, providers=providers)
            self.gigasealer.reset_state()
            info = self.gigasealer.identify_model()
            self._model_meta["gigaseal"] = info
            self._axis_meta["gigaseal"] = self.gigasealer._axis_dims()
        elif which == "break_in":
            self.burglar.load_model(onnx_path, providers=providers)
            self.burglar.reset_state()
            info = self.burglar.identify_model()
            self._model_meta["break_in"] = info
            self._axis_meta["break_in"] = self.burglar._axis_dims()
        else:
            raise ValueError(f"Unknown model '{which}'")
        return self


    def set_goal(self, *, pip=None, stage=None, resistance=None, image=None):
        """
        Store a (possibly partial) goal. Any missing keys (e.g., image) will be mirrored
        from the current observation by the wrapper-aware path in CellHunter.
        
        Args:
            pip (Optional[Sequence[float]]):
                Target pipette position.
            stage (Optional[Sequence[float]]):
                Target stage position.
            resistance (Optional[Sequence[float]]):
                Target resistance values.
            image (Optional[np.ndarray]):
                Target image.

        Returns:
            AutoPatchHelper:
                Instance with stored goal state.
        """
        g = {}
        if pip is not None:
            g["pipette_positions"] = np.asarray(pip, np.float32).reshape(-1)
        if stage is not None:
            s = np.asarray(stage, np.float32).reshape(-1)
            if s.shape[0] == 2:
                s = np.concatenate([s, [0.0]]).astype(np.float32)
            else:
                s = s[:3].astype(np.float32)
            g["stage_positions"] = s
        if resistance is not None:
            g["resistance"] = np.asarray(resistance, np.float32).reshape(-1)
        if image is not None:
            im = np.asarray(image)
            if im.ndim == 2:  # gray → 3‑chan
                im = np.stack([im]*3, axis=-1)
            g["camera_image"] = im
        self._goal = g
        return self














