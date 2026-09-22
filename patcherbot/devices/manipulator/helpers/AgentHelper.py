import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import h5py

_AI_FEATURE_DISABLED_MESSAGE = (
    "AI features need to be enabled in calibration config before use. "
    "Set calibration.use_ai_features to true."
)

_DEFAULT_REPLAY_OBS_KEYS = ("camera_image", "pipette_positions", "stage_positions", "resistance")
_REPLAY_MODEL_TYPES = {"find_pipette_replay", "hunt_replay"}


class AgentHelper:

    def __init__(self, use_ai_features: bool = True):
        """Track the active agent instance and its configuration."""
        self._use_ai_features = bool(use_ai_features)
        self._agent_classes: Optional[Dict[str, Any]] = None
        self.agent = None
        self.requires_goal = False
        self._demo_actions: Optional[np.ndarray] = None
        self._last_demo_dataset: Optional[Dict[str, Any]] = None
        self._default_demo_sources: Dict[str, Dict[str, Any]] = {
            "find_pipette_replay": {
                "path": Path(
                    r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\experiments\Datasets\PatcherBot_dataset_v0_435\PatcherBot_dataset_v0_435_find_pipette.hdf5"
                ),
                "demo_id": "demo_140",
            },
        }

    def _load_agent_classes(self) -> Dict[str, Any]:
        if self._agent_classes is not None:
            return self._agent_classes

        if not self._use_ai_features:
            raise NotImplementedError(_AI_FEATURE_DISABLED_MESSAGE)
        try:
            from patcherbot.deepLearning.PatcherBotAgentR import (
                Burglar,
                CellHunter,
                DemoReplayAgent,
                GigaSealer,
                PipetteFinder,
            )
        except Exception as exc:
            raise NotImplementedError(f"Agent models are not available: {exc}") from exc

        self._agent_classes = {
            "Burglar": Burglar,
            "CellHunter": CellHunter,
            "DemoReplayAgent": DemoReplayAgent,
            "GigaSealer": GigaSealer,
            "PipetteFinder": PipetteFinder,
        }
        return self._agent_classes

    def prepare_model(self, model_type, *, allow_goal_placeholders: Optional[bool] = None):
        """Instantiate one of the supported agent subclasses."""
        agent_classes = self._load_agent_classes()
        demo_replay_cls = agent_classes["DemoReplayAgent"]
        self.model_type = model_type
        if model_type == "find_pipette":
            self.agent = agent_classes["PipetteFinder"]()
        elif model_type == "hunt":
            self.agent = agent_classes["CellHunter"]()
        elif model_type == "gigaseal":
            self.agent = agent_classes["GigaSealer"]()
        elif model_type == "break_in":
            self.agent = agent_classes["Burglar"]()
        elif model_type in {"find_pipette_replay", "hunt_replay"}:
            self.agent = demo_replay_cls()
            if self._demo_actions is None:
                source = self._default_demo_sources.get(model_type)
                if source and source.get("path"):
                    try:
                        self.load_demo_from_hdf5(source["path"], demo_id=source.get("demo_id"))
                    except Exception as exc:
                        raise RuntimeError(f"Failed to load demo actions for '{model_type}': {exc}") from exc
                else:
                    raise RuntimeError(
                        f"No demo actions loaded for '{model_type}'. "
                        "Call load_demo()/load_demo_from_hdf5() or register a demo source via set_demo_source()."
                    )
            else:
                self.agent.load_actions(self._demo_actions)
        else:
            raise ValueError(f"Model type '{model_type}' not supported")
        if isinstance(self.agent, demo_replay_cls):
            image_size = self._infer_image_size(self._last_demo_dataset)
            if image_size is not None:
                self.agent.set_image_size(image_size)
        if allow_goal_placeholders is not None and hasattr(self.agent, "allow_goal_placeholders"):
            self.agent.allow_goal_placeholders = bool(allow_goal_placeholders)
        self.requires_goal = bool(getattr(self.agent, "goal_required", False))

    def observation_input_width(self, input_name: str, default_width: int = 15) -> int:
        """Return an active policy input's final dimension, or zero if unused."""
        if self.agent is None:
            return 0
        required_keys = ()
        getter = getattr(self.agent, "get_required_obs_keys", None)
        if callable(getter):
            try:
                required_keys = tuple(str(key) for key in getter())
            except Exception:
                required_keys = ()
        if not required_keys:
            required_keys = tuple(str(key) for key in getattr(self.agent, "obs_keys", ()) or ())

        importer = getattr(self.agent, "importer", None)
        obs_shapes = getattr(importer, "obs_shapes", {}) or {}
        if input_name not in required_keys and input_name not in obs_shapes:
            return 0
        shape = obs_shapes.get(input_name)
        if shape is not None:
            try:
                shape_tuple = tuple(int(dim) for dim in shape)
            except Exception:
                shape_tuple = ()
            if shape_tuple:
                return max(1, shape_tuple[-1])
        return max(1, int(default_width))

    def load_demo(self, actions: np.ndarray) -> None:
        """Store demo actions for later replay and pass them to an active replay agent."""
        replay = np.asarray(actions, dtype=np.float32)
        if replay.ndim == 1:
            replay = replay.reshape(1, -1)
        if replay.size == 0:
            raise ValueError("Demo action array must contain at least one action")
        self._demo_actions = replay.astype(np.float32, copy=False)
        if self.agent is not None and type(self.agent).__name__ == "DemoReplayAgent":
            self.agent.load_actions(self._demo_actions)

    def has_demo_actions(self) -> bool:
        """Return True when a demo action sequence has been cached."""
        return self._demo_actions is not None and self._demo_actions.size > 0

    def load_demo_from_hdf5(self, data_path: Union[str, Path], *, demo_id: Optional[str] = None, model_type: Optional[str] = None) -> Dict[str, Any]:
        """Load demo data from disk and cache its action sequence for replay use."""
        dataset = self._load_hdf5_sequence(Path(data_path), demo_id=demo_id, model_type=model_type or getattr(self, "model_type", None))
        self._last_demo_dataset = dataset
        self.load_demo(dataset["actions"])
        return dataset

    def _get_active_obs_keys(self, model_type: Optional[str] = None) -> Tuple[str, ...]:
        """Return observation keys from the active model instead of hardcoded model presets."""
        if self.agent is not None and type(self.agent).__name__ != "DemoReplayAgent":
            getter = getattr(self.agent, "get_required_obs_keys", None)
            keys = getter() if callable(getter) else getattr(self.agent, "obs_keys", None)
            if keys:
                return tuple(str(key) for key in keys)

        model_key = str(model_type or getattr(self, "model_type", "")).lower()
        if model_key in _REPLAY_MODEL_TYPES:
            return _DEFAULT_REPLAY_OBS_KEYS

        raise RuntimeError("Prepare the target model before loading HDF5 so its config can define observation keys.")

    def set_demo_source(
        self,
        model_type: str,
        *,
        path: Optional[Union[str, Path]],
        demo_id: Optional[str] = None,
    ) -> None:
        """Register or remove a default demo dataset for a replay-capable model."""
        if path is None:
            self._default_demo_sources.pop(model_type, None)
            return
        self._default_demo_sources[model_type] = {
            "path": Path(path),
            "demo_id": demo_id,
        }

    def _infer_image_size(self, dataset: Optional[Dict[str, Any]]) -> Optional[Tuple[int, int]]:
        """Extract the (height, width) from the cached dataset if available."""
        if not dataset:
            return None
        images = dataset.get("images")
        if images is None or getattr(images, "shape", None) is None:
            return None
        if images.ndim == 4:  # (N, H, W, C)
            return int(images.shape[1]), int(images.shape[2])
        if images.ndim == 3:  # (N, H, W)
            return int(images.shape[1]), int(images.shape[2])
        return None

    def _load_hdf5_sequence(
        self,
        data_path: Path,
        *,
        demo_id: Optional[str] = None,
        model_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load a single demonstration sequence from an HDF5 file."""
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {data_path}")

        with h5py.File(str(data_path), "r") as h5:
            if "data" not in h5:
                raise ValueError(f"Expected group 'data' in {data_path}")
            keys = sorted(h5["data"].keys())
            if not keys:
                raise ValueError(f"No demonstrations stored in {data_path}")
            demo_key = demo_id or keys[0]
            if demo_key not in h5["data"]:
                raise ValueError(f"Demo '{demo_key}' not found in {data_path}")

            obs_root = f"data/{demo_key}/obs"
            act_root = f"data/{demo_key}"

            obs_group = h5[f"{obs_root}"]
            actions = h5[f"{act_root}/actions"][:]
            obs_keys = self._get_active_obs_keys(model_type)
            missing_obs = [key for key in obs_keys if key not in obs_group]
            if missing_obs:
                raise ValueError(
                    f"Demo '{demo_key}' in {data_path} is missing required observation keys: {missing_obs}"
                )

            def _read_obs_value(key: str) -> np.ndarray:
                value = obs_group[key][:]
                if "image" in key.lower():
                    return np.asarray(value)
                return np.asarray(value, dtype=np.float32)

            obs_values = {key: _read_obs_value(key) for key in obs_keys}
            num_frames = int(actions.shape[0])
            image_key = next((key for key in obs_keys if "image" in key.lower()), None)
            pipette_key = next((key for key in obs_keys if "pipette" in key.lower()), None)
            stage_key = next((key for key in obs_keys if "stage" in key.lower()), None)
            resistance_key = next((key for key in obs_keys if "resist" in key.lower()), None)

            images = obs_values.get(image_key)
            resistance = obs_values.get(resistance_key)
            pipette_positions = obs_values.get(pipette_key)
            stage_positions = obs_values.get(stage_key)

        return {
            "demo_id": demo_key,
            "images": None if images is None else np.asarray(images),
            "resistance": None if resistance is None else np.asarray(resistance, dtype=np.float32),
            "pipette_positions": None if pipette_positions is None else np.asarray(pipette_positions, dtype=np.float32),
            "stage_positions": None if stage_positions is None else np.asarray(stage_positions, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.float32),
            "obs": obs_values,
            "obs_keys": obs_keys,
        }

    def run_inference(
        self,
        observation: Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Dict[str, np.ndarray]],
        goal: Optional[np.ndarray] = None,
        *,
        is_demo: bool = False,
    ) -> np.ndarray:
        """Execute inference on the prepared agent."""
        if self.agent is None:
            raise RuntimeError("Call prepare_model before run_inference")
        active_goal = goal if self.requires_goal else None
        return self.agent.inference(observation=observation, goal=active_goal, is_demo=is_demo)


class AgentTester:
    def __init__(self) -> None:
        """Prepare the agent for testing."""
        self.agent_helper = AgentHelper()
        self.predictions: List[np.ndarray] = []
        self.latencies_ms: List[float] = []
        self.errors: List[float] = []
        self.last_results: Optional[Dict[str, Any]] = None
        self.last_dataset: Optional[Dict[str, Any]] = None

    def compute_errors(self, predictions: np.ndarray, actions: np.ndarray) -> List[float]:
        """Compute the L2 error between the predicted and actual actions per frame."""
        pred = np.asarray(predictions, dtype=np.float32)
        gt = np.asarray(actions, dtype=np.float32)

        if pred.ndim == 1:
            pred = pred.reshape(1, -1)
        if gt.ndim == 1:
            gt = gt.reshape(1, -1)
        if pred.shape[0] != gt.shape[0]:
            raise ValueError(
                f"Prediction count ({pred.shape[0]}) does not match action count ({gt.shape[0]})"
            )

        pred_flat = pred.reshape(pred.shape[0], -1)
        gt_flat = gt.reshape(gt.shape[0], -1)
        min_dim = min(pred_flat.shape[1], gt_flat.shape[1])
        if min_dim == 0:
            return [0.0] * pred_flat.shape[0]

        diff = pred_flat[:, :min_dim] - gt_flat[:, :min_dim]
        return np.linalg.norm(diff, axis=1).astype(float).tolist()

    def visualize(
        self,
        pred: Optional[np.ndarray] = None,
        *,
        fps: int = 30,
        save_video_path: Optional[Union[str, Path]] = None,
        save_plot_path: Optional[Union[str, Path]] = None,
        show_animation: bool = True,
        show_plot: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Overlay predicted vs. ground-truth deltas on frames and plot per-axis comparisons."""
        if self.last_results is None or self.last_dataset is None:
            raise RuntimeError("No cached results available. Run test_model before visualize.")

        try:
            import matplotlib.pyplot as plt
            from matplotlib import animation
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure
            from matplotlib.patches import Circle
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("matplotlib is required for visualization but is not installed.") from exc

        predictions_src = pred if pred is not None else self.last_results.get("predictions")
        if predictions_src is None:
            raise RuntimeError("Predictions are not available; provide `pred` or run test_model first.")
        predictions = np.asarray(predictions_src, dtype=np.float32)
        ground_truth_src = self.last_results.get("ground_truth")
        if ground_truth_src is None:
            raise RuntimeError("Ground-truth actions are missing from the cached results.")
        ground_truth = np.asarray(ground_truth_src, dtype=np.float32)
        frames = np.asarray(self.last_dataset.get("images"))

        if predictions.ndim == 1:
            predictions = predictions[:, np.newaxis]
        if ground_truth.ndim == 1:
            ground_truth = ground_truth[:, np.newaxis]

        num_frames = min(len(frames), len(predictions), len(ground_truth))
        if num_frames == 0:
            raise RuntimeError("No frames available for visualization.")
        predictions = predictions[:num_frames]
        ground_truth = ground_truth[:num_frames]
        frames = frames[:num_frames]

        def _ensure_rgb(image: np.ndarray) -> np.ndarray:
            arr = np.asarray(image)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            elif arr.ndim == 3 and arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            arr = arr.astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0
            arr = np.clip(arr, 0.0, 1.0)
            return (arr * 255.0).astype(np.uint8)

        def _vector_xy(vec: np.ndarray) -> np.ndarray:
            if vec.size >= 2:
                return vec[:2]
            if vec.size == 1:
                return np.array([vec[0], 0.0], dtype=np.float32)
            return np.zeros(2, dtype=np.float32)

        def _canvas_to_rgb(canvas: FigureCanvasAgg) -> np.ndarray:
            width, height = canvas.get_width_height()
            try:
                buffer = canvas.tostring_rgb()
            except AttributeError:
                rgba = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
                return rgba.reshape(height, width, 4)[..., :3].copy()
            array = np.frombuffer(buffer, dtype=np.uint8)
            return array.reshape(height, width, 3).copy()

        frame_height = frames.shape[1]
        frame_width = frames.shape[2] if frames.ndim >= 3 else frames.shape[1]

        pipette_positions: Optional[np.ndarray] = None
        pipette_positions_src = self.last_dataset.get("pipette_positions")
        if pipette_positions_src is not None:
            pipette_positions_arr = np.asarray(pipette_positions_src, dtype=np.float32)
            if pipette_positions_arr.ndim == 1:
                pipette_positions_arr = pipette_positions_arr[:, np.newaxis]
            if (
                pipette_positions_arr.shape[1] >= 2
                and pipette_positions_arr.shape[0] >= num_frames
            ):
                pipette_positions = pipette_positions_arr[:num_frames, :2]

        overlay_frames = []
        for idx in range(num_frames):
            frame_rgb = _ensure_rgb(frames[idx])
            height, width = frame_rgb.shape[:2]

            if pipette_positions is not None and pipette_positions.shape[0] > idx:
                current_coords = pipette_positions[idx]
            else:
                current_coords = np.array([width / 2.0, height / 2.0], dtype=np.float32)
            if np.any(np.isnan(current_coords)):
                current_coords = np.array([width / 2.0, height / 2.0], dtype=np.float32)
            current_x = float(np.clip(current_coords[0], 0.0, max(width - 1.0, 0.0)))
            current_y = float(np.clip(current_coords[1], 0.0, max(height - 1.0, 0.0)))
            circle_radius = max(3.0, min(width, height) * 0.03)
            trail_radius = max(2.0, circle_radius * 0.7)
            future_steps = max(0, min(5, num_frames - idx - 1))

            def _clamp_point(coords: np.ndarray) -> np.ndarray:
                point = np.array(coords, dtype=np.float32)
                point[0] = float(np.clip(point[0], 0.0, max(width - 1.0, 0.0)))
                point[1] = float(np.clip(point[1], 0.0, max(height - 1.0, 0.0)))
                return point

            fig = Figure(figsize=(width / 100.0, height / 100.0), dpi=100)
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
            ax.imshow(frame_rgb)
            ax.axis("off")

            def _future_points_from_deltas(
                vecs: np.ndarray,
                *,
                start_anchor: np.ndarray,
            ) -> List[np.ndarray]:
                points: List[np.ndarray] = []
                base = np.array(start_anchor, dtype=np.float32)
                for step_idx in range(future_steps):
                    action_idx = idx + step_idx
                    if action_idx >= vecs.shape[0]:
                        break
                    delta = _vector_xy(vecs[action_idx])
                    if np.any(np.isnan(delta)):
                        continue
                    offset = np.array([float(delta[0]), -float(delta[1])], dtype=np.float32)
                    base = _clamp_point(base + offset)
                    points.append(base.copy())
                return points

            def _future_ground_truth_points() -> List[np.ndarray]:
                points: List[np.ndarray] = []
                if pipette_positions is not None:
                    for step_idx in range(future_steps):
                        future_idx = idx + step_idx + 1
                        if future_idx >= pipette_positions.shape[0]:
                            break
                        points.append(_clamp_point(pipette_positions[future_idx, :2]))
                    if points:
                        return points
                current_point = np.array([current_x, current_y], dtype=np.float32)
                return _future_points_from_deltas(ground_truth, start_anchor=current_point)

            def _future_prediction_points() -> List[np.ndarray]:
                current_point = np.array([current_x, current_y], dtype=np.float32)
                return _future_points_from_deltas(predictions, start_anchor=current_point)

            def _draw_trail(points: List[np.ndarray], color: str) -> None:
                for step_idx, point in enumerate(points):
                    alpha = max(0.3, 1.0 - step_idx * 0.15)
                    ax.add_patch(
                        Circle(
                            (float(point[0]), float(point[1])),
                            radius=trail_radius,
                            facecolor=color,
                            edgecolor="none",
                            alpha=alpha,
                        )
                    )

            if future_steps:
                gt_points = _future_ground_truth_points()
                pred_points = _future_prediction_points()
                _draw_trail(gt_points, "tab:blue")
                _draw_trail(pred_points, "tab:orange")

            ax.add_patch(
                Circle(
                    (current_x, current_y),
                    radius=circle_radius,
                    facecolor="white",
                    edgecolor="black",
                    linewidth=0.6,
                    alpha=0.95,
                )
            )
            canvas.draw()
            image = _canvas_to_rgb(canvas)
            overlay_frames.append(image)
            plt.close(fig)

        overlay_array = np.stack(overlay_frames, axis=0)

        time_axis = np.arange(num_frames)
        dims = max(predictions.shape[1], ground_truth.shape[1])
        fig_plot, axes = plt.subplots(dims, 1, sharex=True, figsize=(10, 3 * dims))
        if dims == 1:
            axes = [axes]

        for axis_idx in range(dims):
            ax_plot = axes[axis_idx]
            gt_series = ground_truth[:, axis_idx] if axis_idx < ground_truth.shape[1] else None
            pred_series = predictions[:, axis_idx] if axis_idx < predictions.shape[1] else None
            if gt_series is not None:
                ax_plot.plot(time_axis, gt_series, label="Ground truth", color="tab:blue")
            if pred_series is not None:
                ax_plot.plot(time_axis, pred_series, label="Prediction", color="tab:orange", linestyle="--")
            ax_plot.set_ylabel(f"Axis {axis_idx}")
        axes[0].set_title("Action delta comparison")
        axes[-1].set_xlabel("Frame index")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(handles, labels, loc="upper right")
        fig_plot.tight_layout()

        canvas_plot = FigureCanvasAgg(fig_plot)
        canvas_plot.draw()
        plot_array = _canvas_to_rgb(canvas_plot)

        if save_plot_path is not None:
            save_plot_path = Path(save_plot_path)
            save_plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig_plot.savefig(save_plot_path, dpi=200, bbox_inches="tight")

        if save_video_path is not None:
            save_video_path = Path(save_video_path)
            save_video_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                import imageio.v3 as iio
            except ImportError:
                try:
                    import imageio as iio  # type: ignore[no-redef]
                except ImportError as exc:
                    raise RuntimeError("imageio is required to save the visualization video.") from exc
            if save_video_path.suffix.lower() == ".gif":
                iio.imwrite(save_video_path, overlay_array, fps=fps, loop=0)
            else:
                with iio.get_writer(save_video_path, fps=fps) as writer:
                    for frame in overlay_array:
                        writer.append_data(frame)

        if not show_plot:
            plt.close(fig_plot)

        if show_animation:
            fig_anim, ax_anim = plt.subplots(
                figsize=(overlay_array.shape[2] / 100.0, overlay_array.shape[1] / 100.0), dpi=100
            )
            ax_anim.axis("off")
            image_artist = ax_anim.imshow(overlay_array[0])

            def _update(frame: np.ndarray):
                image_artist.set_data(frame)
                return (image_artist,)

            anim = animation.FuncAnimation(  # keep reference to avoid garbage collection
                fig_anim,
                _update,
                frames=overlay_array,
                interval=max(1, int(1000 / max(fps, 1))),
                repeat=True,
                blit=True,
            )
            plt.show()
            plt.close(fig_anim)
            if show_plot:
                plt.close(fig_plot)
        else:
            if show_plot:
                plt.show()
                plt.close(fig_plot)

        return overlay_array, plot_array

    def test_model(
        self,
        model_type: str,
        data_path: Union[str, Path],
        demo_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run inference over a dataset and collect error/latency metrics."""
        self.agent_helper.prepare_model(model_type, allow_goal_placeholders=True)
        dataset = self.agent_helper.load_demo_from_hdf5(data_path, demo_id=demo_id, model_type=model_type)
        self.last_dataset = dataset

        goal = None
        if self.agent_helper.requires_goal:
            if dataset.get("pipette_positions") is None:
                raise RuntimeError("Goal-required model dataset is missing pipette position observations.")
            goal = np.asarray(dataset["pipette_positions"][-1], dtype=np.float32)

        self.predictions.clear()
        self.latencies_ms.clear()
        self.errors.clear()

        num_frames = dataset["actions"].shape[0]
        obs_keys = tuple(dataset.get("obs_keys", dataset["obs"].keys()))
        for idx in range(num_frames):
            observation = {
                key: np.asarray(dataset["obs"][key][idx])
                for key in obs_keys
            }

            start = time.perf_counter()
            predicted_action = self.agent_helper.run_inference(
                observation=observation,
                goal=goal,
                is_demo=True,
            )
            latency_ms = (time.perf_counter() - start) * 1000.0

            self.latencies_ms.append(float(latency_ms))
            self.predictions.append(np.asarray(predicted_action, dtype=np.float32).reshape(-1))

        if not self.predictions:
            raise RuntimeError("No predictions were generated during testing")

        predictions_arr = np.asarray(self.predictions, dtype=np.float32)
        actions_arr = np.asarray(dataset["actions"], dtype=np.float32)

        min_count = min(predictions_arr.shape[0], actions_arr.shape[0])
        predictions_arr = predictions_arr[:min_count]
        actions_arr = actions_arr[:min_count]

        self.errors = self.compute_errors(predictions_arr, actions_arr)

        summary = {
            "demo_id": dataset["demo_id"],
            "predictions": predictions_arr,
            "ground_truth": actions_arr,
            "errors": self.errors,
            "mean_error": float(np.mean(self.errors)) if self.errors else float("nan"),
            "max_error": float(np.max(self.errors)) if self.errors else float("nan"),
            "latencies_ms": self.latencies_ms,
            "mean_latency_ms": float(np.mean(self.latencies_ms)) if self.latencies_ms else float("nan"),
        }
        self.last_results = summary


        return summary

    def main(
        self,
        model_type: str,
        data_path: Union[str, Path],
        demo_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Entry point used by the CLI stub below."""
        results = self.test_model(model_type=model_type, data_path=data_path, demo_id=demo_id)
    
        mean_error = results.get("mean_error", float("nan"))
        mean_latency = results.get("mean_latency_ms", float("nan"))
        print(
            f"[RESULT] Demo {results['demo_id']}: mean error={mean_error:.4f}, "
            f"max error={results['max_error']:.4f}, mean latency={mean_latency:.2f} ms"
        )
        self.visualize()
        return results


if __name__ == "__main__":
    agenttester = AgentTester()

    model_type = "find_pipette"
    # data_path = r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\experiments\Datasets\PatcherBot_test_dataset_v0_300\PatcherBot_test_dataset_v0_300_find_pipette.hdf5"
    data_path = r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\experiments\Datasets\PatcherBot_dataset_v0_435\PatcherBot_dataset_v0_435_find_pipette.hdf5"
    # data_path = r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\experiments\Datasets\PatcherBot_dataset_v0_400\PatcherBot_dataset_v0_400_find_pipette.hdf5"
    demo_id = "demo_140"
    agenttester.main(model_type=model_type, data_path=data_path, demo_id=demo_id)

