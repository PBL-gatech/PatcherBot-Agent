import sys
import h5py
import numpy as np
from typing import List
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QGraphicsScene, QGraphicsView, QPushButton, QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap, QPen, QPainter, QBrush
from PyQt5.QtCore import Qt, QTimer, QPointF


class DemoPlayer(QWidget):
    def __init__(self, hdf5_path):
        super().__init__()
        self.hdf5_path = hdf5_path
        self.playing = True  # video plays by default
        self.observed_pipette_positions = np.empty((0, 2), dtype=np.float32)
        self.reconstructed_pipette_positions = np.empty((0, 2), dtype=np.float32)
        self._pipette_frame_count = 0
        self.pipette_tail_length = 25  # number of upcoming points to render
        self.action_axes: List[str] = []

        # Try opening the HDF5 file.
        try:
            self.hdf5_file = h5py.File(self.hdf5_path, 'r')
        except OSError as e:
            QMessageBox.critical(self, "File Error", f"Could not open file {hdf5_path}:\n{e}")
            sys.exit(1)

        if 'data' not in self.hdf5_file:
            QMessageBox.critical(self, "File Error", f"The file {hdf5_path} does not contain a 'data' group.")
            sys.exit(1)

        self.demo_keys = sorted(self.hdf5_file['data'].keys())
        if not self.demo_keys:
            QMessageBox.critical(self, "Data Error", "No demos found in the HDF5 file!")
            sys.exit(1)

        self.current_demo_idx = 0
        self.current_frame = 0

        self.init_ui()
        self.load_demo(self.current_demo_idx)

    # ------------------------------------------------------------------
    # 1. init_ui() — removed hard-coded 500 × 500, added centring
    # ------------------------------------------------------------------
    def init_ui(self):
        self.setWindowTitle("Demo Viewer")
        self.resize(800, 500)

        main_layout = QHBoxLayout(self)

        # Left side: video display and navigation buttons.
        left_layout = QVBoxLayout()

        self.video_label = QLabel(self)
        # self.video_label.setFixedSize(500, 500)   # ← removed per request
        self.video_label.setAlignment(Qt.AlignCenter)     # keep pixmap centred
        left_layout.addWidget(self.video_label)

        # Button layout for navigation.
        button_layout = QHBoxLayout()

        self.prev_button = QPushButton("Previous Demo", self)
        self.prev_button.clicked.connect(self.prev_demo)
        button_layout.addWidget(self.prev_button)

        self.play_pause_button = QPushButton("Pause", self)
        self.play_pause_button.clicked.connect(self.toggle_play_pause)
        button_layout.addWidget(self.play_pause_button)

        self.next_button = QPushButton("Next Demo", self)
        self.next_button.clicked.connect(self.next_demo)
        button_layout.addWidget(self.next_button)

        left_layout.addLayout(button_layout)
        main_layout.addLayout(left_layout)

        # Right side: resistance plot area (fixed to 400x400).
        self.scene = QGraphicsScene(self)
        self.plot_view = QGraphicsView(self.scene, self)
        self.plot_view.setFixedSize(500, 500)
        main_layout.addWidget(self.plot_view)


        self.action_scene = QGraphicsScene(self)
        self.action_view  = QGraphicsView(self.action_scene, self)
        self.action_view.setFixedSize(500, 400)      # tweak height as you like
        main_layout.addWidget(self.action_view)



        self.setLayout(main_layout)

        # Timer for video frame updates (~33 fps).
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    # ------------------------------------------------------------------
    # 2. load_demo() — resize label to native image size (once)
    # ------------------------------------------------------------------
    def load_demo(self, idx):
        """Load images (video) and resistance data for the given demo."""
        self.current_frame = 0
        demo_key = self.demo_keys[idx]
        demo_path = f'data/{demo_key}/obs'
        demo_path_act = f'data/{demo_key}/actions'
        print(f"Loading demo: {demo_key}")  # Debug message

        self.actions = np.empty((0, 0), dtype=np.float32)
        self.action_axes = []
        self.observed_pipette_positions = np.empty((0, 2), dtype=np.float32)
        self.reconstructed_pipette_positions = np.empty((0, 2), dtype=np.float32)
        self._pipette_frame_count = 0

        for name, attr in (('camera_image', 'images'), ('resistance', 'resistance')):
            try:
                setattr(self, attr, self.hdf5_file[f'{demo_path}/{name}'][:])
            except KeyError as e:
                print(f"Warning: missing '{name}' in {demo_path}: {e}")
                setattr(self, attr, np.array([]))

        try:
            action_ds = self.hdf5_file[f'data/{demo_key}/actions']
            data = np.asarray(action_ds[:], dtype=np.float32)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            elif data.ndim > 2:
                data = data.reshape(data.shape[0], -1)
            self.actions = data
            axes_attr = action_ds.attrs.get("axes")
            if axes_attr is None:
                self.action_axes = []
            else:
                axes_array = np.atleast_1d(axes_attr)
                self.action_axes = [
                    axis.decode("utf-8") if isinstance(axis, bytes) else str(axis)
                    for axis in axes_array
                ]
        except KeyError:
            print(f"Warning: missing 'actions' dataset in data/{demo_key}")



        # Resize the label *once* to the first frame’s native size
        if self.images.size:
            h_img, w_img = self.images[0].shape[:2]
            self.video_label.setFixedSize(w_img, h_img)

        # Validate the resistance data.
        # if self.resistance.ndim != 1:
        #     QMessageBox.critical(
        #         self,
        #         "Data Error",
        #         f"Expected 'resistance' to be 1D, got shape {self.resistance.shape}"
        #     )
        #     self.resistance = np.array([])
        if self.resistance.ndim != 1:
            # reshape if possible
            if self.resistance.ndim == 2 and 1 in self.resistance.shape:
                self.resistance = self.resistance.flatten()
            else:
                print(f"Warning: expected 1D resistance, got {self.resistance.shape}")
                self.resistance = np.array([])

        self._load_observed_pipette_positions(demo_path)
        self._build_reconstructed_pipette_path()
        self.plot_resistance()
        self.plot_actions()

    def _load_observed_pipette_positions(self, demo_path):
        """Load pipette positions for the current demo if available."""
        self.observed_pipette_positions = np.empty((0, 2), dtype=np.float32)

        try:
            dataset = self.hdf5_file[f'{demo_path}/pipette_positions']
        except KeyError:
            return

        data = np.asarray(dataset[:], dtype=np.float32)

        if data.ndim == 1:
            if data.size % 3 == 0:
                cols = 3
            elif data.size % 2 == 0:
                cols = 2
            else:
                cols = 1
            data = data.reshape(-1, cols)

        if data.size == 0:
            return

        if data.ndim != 2 or data.shape[1] == 0:
            print(f"Warning: '{demo_path}/pipette_positions' has unexpected shape {data.shape}")
            return

        dims = min(2, data.shape[1])
        if dims < 2:
            print(f"Warning: '{demo_path}/pipette_positions' has only {data.shape[1]} column(s); expected at least 2 for image overlay")
            return
        positions = np.asarray(data[:, :dims], dtype=np.float32)

        frame_dims = None
        if self.images.size:
            frame_count = min(len(self.images), positions.shape[0])
            if frame_count != positions.shape[0]:
                print(f"Warning: trimming pipette positions from {positions.shape[0]} to {frame_count} to match frames")
            positions = positions[:frame_count]
            h_img, w_img = self.images[0].shape[:2]
            frame_dims = np.array([w_img, h_img], dtype=np.float32)
        else:
            frame_count = positions.shape[0]

        if frame_count == 0:
            return

        def _in_frame(arr: np.ndarray, dims: np.ndarray, margin: float = 0.5) -> bool:
            if arr.size == 0 or dims is None:
                return True
            x_ok = np.logical_and(arr[:, 0] >= -margin, arr[:, 0] <= dims[0] - 1.0 + margin)
            y_ok = np.logical_and(arr[:, 1] >= -margin, arr[:, 1] <= dims[1] - 1.0 + margin)
            return bool(np.all(x_ok) and np.all(y_ok))

        if frame_dims is not None and not _in_frame(positions, frame_dims):
            base_scale = np.where(frame_dims > 0.0, frame_dims / 1280.0, 0.0)
            scaled = positions * base_scale
            if _in_frame(scaled, frame_dims):
                positions = scaled
                print(f"Info: scaled pipette positions to match {int(frame_dims[0])}x{int(frame_dims[1])} frames")
            else:
                upper = np.maximum(frame_dims - 1.0, 0.0)
                outside_mask = np.logical_or(np.any(scaled < 0.0, axis=1), np.any(scaled > upper, axis=1))
                if np.any(outside_mask):
                    count = int(outside_mask.sum())
                    print(f"Warning: clipped {count} pipette position rows to image bounds ({int(frame_dims[0])}x{int(frame_dims[1])})")
                positions = np.clip(scaled, [0.0, 0.0], upper)

        finite_mask = np.all(np.isfinite(positions), axis=1)
        if not np.all(finite_mask):
            dropped = int(finite_mask.size - finite_mask.sum())
            print(f"Warning: dropping {dropped} pipette position rows with non-finite values")
            positions = positions[finite_mask]

        self.observed_pipette_positions = positions

        if positions.size and frame_dims is not None and not _in_frame(positions, frame_dims):
            print(f"Warning: pipette positions fall outside image bounds ({int(frame_dims[0])}x{int(frame_dims[1])}); overlay may not be visible")

    def _resolve_pipette_action_columns(self, pipette_dim: int) -> List[int]:
        """Return indices corresponding to pipette deltas within the action matrix."""
        if self.actions.size == 0 or self.actions.ndim != 2:
            return []

        indices: List[int] = []
        if self.action_axes:
            indices = [
                idx for idx, name in enumerate(self.action_axes)
                if isinstance(name, str) and name.startswith("pipette_")
            ]

        if not indices and pipette_dim > 0:
            total = self.actions.shape[1]
            if total >= pipette_dim:
                indices = list(range(total - pipette_dim, total))

        return indices[:pipette_dim] if indices else []

    def _build_reconstructed_pipette_path(self):
        """Integrate pipette action deltas to obtain absolute positions for overlay."""
        self.reconstructed_pipette_positions = np.empty((0, 2), dtype=np.float32)
        self._pipette_frame_count = 0

        if self.actions.size == 0:
            return

        if self.observed_pipette_positions.size:
            pipette_dim = self.observed_pipette_positions.shape[1]
        else:
            pipette_dim = sum(
                1 for name in self.action_axes if name.startswith("pipette_")
            ) or min(2, self.actions.shape[1])

        pipette_dim = max(0, min(pipette_dim, self.actions.shape[1]))
        pipette_dim = min(2, pipette_dim)
        if pipette_dim == 0:
            return

        pip_cols = self._resolve_pipette_action_columns(pipette_dim)
        if not pip_cols:
            return

        deltas = self.actions[:, pip_cols].astype(np.float32, copy=False)
        if deltas.ndim != 2:
            deltas = deltas.reshape(-1, pipette_dim)

        cumulative = np.cumsum(deltas, axis=0)
        if self.observed_pipette_positions.size:
            start = self.observed_pipette_positions[0]
        else:
            start = np.zeros((pipette_dim,), dtype=np.float32)
        cumulative += start

        frame_limit = cumulative.shape[0]
        if self.images.size:
            frame_limit = min(frame_limit, len(self.images))
        if self.observed_pipette_positions.size:
            frame_limit = min(frame_limit, self.observed_pipette_positions.shape[0])

        self.reconstructed_pipette_positions = cumulative[:frame_limit]
        if self.observed_pipette_positions.size:
            self.observed_pipette_positions = self.observed_pipette_positions[:frame_limit]
        self._pipette_frame_count = self.reconstructed_pipette_positions.shape[0]

    # ------------------------------------------------------------------
    # Resistance-plotting code (unchanged)
    # ------------------------------------------------------------------
    def plot_resistance(self):
        self.scene.clear()
        if self.resistance.size == 0:
            self.scene.addText("No resistance data available")
            return

        plot_w, plot_h = 400, 350
        max_res = float(np.max(self.resistance))
        min_res = float(np.min(self.resistance))
        points = len(self.resistance)

        if points < 2:
            self.scene.addText("Not enough data points for plot")
            return

        pen = QPen(Qt.blue)
        step = plot_w / (points - 1)
        diff = max_res - min_res if max_res != min_res else 1e-6

        for i in range(points - 1):
            x1 = i * step
            y1 = plot_h - ((self.resistance[i] - min_res) / diff) * plot_h
            x2 = (i + 1) * step
            y2 = plot_h - ((self.resistance[i + 1] - min_res) / diff) * plot_h
            self.scene.addLine(x1, y1, x2, y2, pen)

        # y-axis labels
        for i in range(6):
            y = plot_h - (i / 5) * plot_h
            value = min_res + (i / 5) * diff
            t = self.scene.addText(f"{value:.2f}")
            t.setPos(-50, y - t.boundingRect().height() / 2)

        # x-axis labels: first and last
        t0 = self.scene.addText("0")
        t0.setPos(0, plot_h + 5)
        tN = self.scene.addText(f"{points - 1}")
        tN.setPos(plot_w - tN.boundingRect().width(), plot_h + 5)

    def plot_actions(self):
        """Draw a bar chart of non-zero action counts for up to six axes."""
        self.action_scene.clear()

        if self.actions.size == 0 or self.actions.ndim != 2:
            self.action_scene.addText("No action data available")
            return

        max_cols = min(6, self.actions.shape[1])
        if max_cols == 0:
            self.action_scene.addText("No action data available")
            return

        counts = np.count_nonzero(self.actions[:, :max_cols], axis=0)
        labels = []
        for idx in range(max_cols):
            if self.action_axes and idx < len(self.action_axes):
                labels.append(self.action_axes[idx])
            else:
                labels.append(str(idx))

        plot_w, plot_h = 400, 200
        bar_w = plot_w / max_cols
        max_ct = counts.max() or 1

        pen = QPen(Qt.black)
        brush = Qt.gray

        for i, c in enumerate(counts):
            x = i * bar_w
            h = (c / max_ct) * plot_h
            self.action_scene.addRect(x, plot_h - h, bar_w * 0.8, h, pen, brush)
            label_item = self.action_scene.addText(str(int(c)))
            label_item.setPos(
                x + bar_w * 0.4 - label_item.boundingRect().width() / 2,
                plot_h - h - label_item.boundingRect().height() - 2,
            )

        for i in range(max_cols):
            axis_label = labels[i][:12]
            lbl_item = self.action_scene.addText(axis_label)
            lbl_item.setPos(
                i * bar_w + bar_w * 0.4 - lbl_item.boundingRect().width() / 2,
                plot_h + 5,
            )

    def _draw_pipette_overlay(self, pixmap, frame_idx):
        if self._pipette_frame_count == 0:
            return

        if frame_idx >= self._pipette_frame_count:
            frame_idx = self._pipette_frame_count - 1
        if frame_idx < 0:
            return

        point = self.reconstructed_pipette_positions[frame_idx]
        if not np.all(np.isfinite(point)):
            return

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        pen_width = max(2, max(pixmap.width(), pixmap.height()) // 180)
        radius = max(3, min(pixmap.width(), pixmap.height()) // 50)

        if self.observed_pipette_positions.shape[0] > frame_idx:
            obs_point = self.observed_pipette_positions[frame_idx]
            if np.all(np.isfinite(obs_point)):
                obs_pen = QPen(Qt.green)
                obs_pen.setWidth(max(1, pen_width // 2))
                obs_pen.setStyle(Qt.DashLine)
                painter.setPen(obs_pen)
                painter.setBrush(Qt.NoBrush)
                obs_radius = max(2, radius // 2)
                painter.drawEllipse(
                    QPointF(float(obs_point[0]), float(obs_point[1])),
                    obs_radius,
                    obs_radius,
                )
                if self.pipette_tail_length > 0:
                    end_idx = min(
                        self.observed_pipette_positions.shape[0],
                        frame_idx + self.pipette_tail_length + 1,
                    )
                    obs_future = self.observed_pipette_positions[frame_idx:end_idx]
                    finite_obs = obs_future[np.all(np.isfinite(obs_future), axis=1)]
                    if finite_obs.shape[0] > 1:
                        q_obs_points = [QPointF(float(p[0]), float(p[1])) for p in finite_obs]
                        for p0, p1 in zip(q_obs_points[:-1], q_obs_points[1:]):
                            painter.drawLine(p0, p1)

        pen = QPen(Qt.red)
        pen.setWidth(int(pen_width))
        painter.setPen(pen)
        painter.setBrush(QBrush(Qt.red))

        painter.drawEllipse(QPointF(float(point[0]), float(point[1])), radius, radius)

        if self.pipette_tail_length > 0:
            end_idx = min(
                self._pipette_frame_count,
                frame_idx + self.pipette_tail_length + 1,
            )
            future_path = self.reconstructed_pipette_positions[frame_idx:end_idx]
            finite_future = future_path[np.all(np.isfinite(future_path), axis=1)]
            if finite_future.shape[0] > 1:
                q_points = [QPointF(float(p[0]), float(p[1])) for p in finite_future]
                for p0, p1 in zip(q_points[:-1], q_points[1:]):
                    painter.drawLine(p0, p1)

        painter.end()
    # ------------------------------------------------------------------
    # 3. update_frame() — shrink oversized frames, never upscale
    # ------------------------------------------------------------------
    def update_frame(self):
        if self.images.size == 0:
            return

        if self.current_frame >= len(self.images):
            self.current_frame = 0

        frame_idx = self.current_frame

        try:
            img_array = self.images[frame_idx]
        except IndexError:
            self.current_frame = 0
            return

        # Handle grayscale vs. colour frames.
        if img_array.ndim == 2:
            h_img, w_img = img_array.shape
            bytes_per_line = w_img
            q_img = QImage(img_array.data, w_img, h_img, bytes_per_line, QImage.Format_Grayscale8)
        elif img_array.ndim == 3 and img_array.shape[-1] in (3, 4):
            h_img, w_img, channels = img_array.shape
            bytes_per_line = w_img * channels
            q_format = QImage.Format_RGB888 if channels == 3 else QImage.Format_RGBA8888
            q_img = QImage(img_array.data, w_img, h_img, bytes_per_line, q_format)
        else:
            return  # unexpected format

        pixmap = QPixmap.fromImage(q_img)
        self._draw_pipette_overlay(pixmap, frame_idx)

        # Only shrink if the frame exceeds the label dimensions
        if (pixmap.width() > self.video_label.width() or
                pixmap.height() > self.video_label.height()):
            pixmap = pixmap.scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

        self.video_label.setPixmap(pixmap)
        self.current_frame = frame_idx + 1

    # ------------------------------------------------------------------
    # Control / navigation handlers (unchanged)
    # ------------------------------------------------------------------
    def toggle_play_pause(self):
        if self.playing:
            self.timer.stop()
            self.play_pause_button.setText("Play")
        else:
            self.timer.start(30)
            self.play_pause_button.setText("Pause")
        self.playing = not self.playing

    def next_demo(self):
        self.current_demo_idx = (self.current_demo_idx + 1) % len(self.demo_keys)
        self.load_demo(self.current_demo_idx)

    def prev_demo(self):
        self.current_demo_idx = (self.current_demo_idx - 1) % len(self.demo_keys)
        self.load_demo(self.current_demo_idx)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Right:
            self.next_demo()
        elif event.key() == Qt.Key_Left:
            self.prev_demo()
        elif event.key() == Qt.Key_Space:
            self.toggle_play_pause()

    def closeEvent(self, event):
        if hasattr(self, 'hdf5_file'):
            self.hdf5_file.close()
        event.accept()


# ----------------------------------------------------------------------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    # data_path = r"experiments/Datasets/PatcherBot_test_dataset_v0_201/PatcherBot_test_dataset_v0_201_find_pipette.hdf5"

    # data_path = r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\experiments\Datasets\PatcherBot_dataset_v0_740\PatcherBot_dataset_v0_740_find_pipette.hdf5"
    data_path = r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\experiments\Datasets\PatcherBotViewer_test_dataset_v0_001\PatcherBotViewer_test_dataset_v0_001_hunt_cell.hdf5"
    viewer = DemoPlayer(data_path)
    viewer.show()
    sys.exit(app.exec_())
