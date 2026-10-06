import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt5.QtCore import QPointF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGraphicsEllipseItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QSlider,
)


IMAGE_EXTENSIONS = {".webp", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
DEFAULT_DATA_ROOT = Path("experiments/Data/rig_recorder_data")


@dataclass(frozen=True)
class FrameRecord:
    path: Path
    frame_number: int
    timestamp: Optional[float]


def frame_sort_key(path: Path):
    match = re.match(r"(\d+)(?:_([0-9]+(?:\.[0-9]+)?))?", path.stem)
    if match:
        frame_no = int(match.group(1))
        timestamp = float(match.group(2)) if match.group(2) is not None else float(frame_no)
        return frame_no, timestamp, path.name
    return 10**18, 0.0, path.name


def load_frame_records(camera_dir: Path) -> List[FrameRecord]:
    paths = [
        path
        for path in camera_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    paths.sort(key=frame_sort_key)
    records = []
    for fallback_index, path in enumerate(paths):
        frame_no, timestamp, _ = frame_sort_key(path)
        if frame_no == 10**18:
            frame_no = fallback_index
            timestamp = None
        records.append(FrameRecord(path=path, frame_number=frame_no, timestamp=timestamp))
    return records


class ImageSceneView(QGraphicsView):
    imageClicked = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.pixmap_item: Optional[QGraphicsPixmapItem] = None
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)

    def set_pixmap(self, pixmap: QPixmap):
        scene = self.scene()
        scene.clear()
        self.pixmap_item = scene.addPixmap(pixmap)
        scene.setSceneRect(pixmap.rect())
        self.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)

    def add_marker(self, x: float, y: float, color: QColor, label: str = ""):
        radius = 5
        item = QGraphicsEllipseItem(x - radius, y - radius, radius * 2, radius * 2)
        item.setPen(QPen(color, 2))
        item.setBrush(color)
        self.scene().addItem(item)
        if label:
            text = self.scene().addText(label)
            text.setDefaultTextColor(color)
            text.setPos(QPointF(x + 7, y + 7))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.scene() and not self.scene().sceneRect().isNull():
            self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.pixmap_item is not None:
            scene_point = self.mapToScene(event.pos())
            rect = self.pixmap_item.boundingRect()
            if rect.contains(scene_point):
                self.imageClicked.emit(scene_point.x(), scene_point.y())
                return
        super().mousePressEvent(event)


class ReplayAnnotator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rig Replay Annotator")
        self.resize(1450, 900)

        self.dataset_dir: Optional[Path] = None
        self.camera_dir: Optional[Path] = None
        self.frames: List[FrameRecord] = []
        self.current_index = 0
        self.cuts: List[Dict[str, Any]] = []
        self.annotations: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.clip_start = 0
        self.clip_end = 0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.advance_frame)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.video_tab = QWidget()
        self.track_tab = QWidget()
        self.tabs.addTab(self.video_tab, "Video Cuts")
        self.tabs.addTab(self.track_tab, "Point Tracking")

        self._build_video_tab()
        self._build_track_tab()

    def _build_video_tab(self):
        layout = QVBoxLayout(self.video_tab)

        top = QHBoxLayout()
        self.load_button = QPushButton("Select rig recording")
        self.load_button.clicked.connect(self.select_recording)
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_playback)
        self.prev_button = QPushButton("Prev")
        self.prev_button.clicked.connect(lambda: self.set_frame(self.current_index - 1))
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(lambda: self.set_frame(self.current_index + 1))
        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(1.0, 120.0)
        self.fps_spin.setValue(30.0)
        self.fps_spin.setSuffix(" fps")
        self.fps_spin.valueChanged.connect(self.update_timer_interval)
        top.addWidget(self.load_button)
        top.addWidget(self.prev_button)
        top.addWidget(self.play_button)
        top.addWidget(self.next_button)
        top.addWidget(QLabel("Playback"))
        top.addWidget(self.fps_spin)
        top.addStretch()
        layout.addLayout(top)

        middle = QHBoxLayout()
        self.video_view = ImageSceneView()
        self.video_view.imageClicked.connect(lambda _x, _y: None)
        middle.addWidget(self.video_view, stretch=4)

        cut_panel = QVBoxLayout()
        self.frame_label = QLabel("No recording loaded")
        cut_panel.addWidget(self.frame_label)
        self.add_cut_button = QPushButton("Add cut at frame")
        self.add_cut_button.clicked.connect(self.add_cut)
        self.delete_cut_button = QPushButton("Delete selected cut")
        self.delete_cut_button.clicked.connect(self.delete_selected_cut)
        self.save_cuts_button = QPushButton("Save cuts")
        self.save_cuts_button.clicked.connect(self.save_cuts)
        self.load_cuts_button = QPushButton("Load cuts")
        self.load_cuts_button.clicked.connect(self.load_cuts)
        self.cut_list = QListWidget()
        self.cut_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.cut_list.itemDoubleClicked.connect(self.jump_to_cut)
        cut_panel.addWidget(self.add_cut_button)
        cut_panel.addWidget(self.delete_cut_button)
        cut_panel.addWidget(self.save_cuts_button)
        cut_panel.addWidget(self.load_cuts_button)
        cut_panel.addWidget(QLabel("Event cuts"))
        cut_panel.addWidget(self.cut_list, stretch=1)
        middle.addLayout(cut_panel, stretch=1)
        layout.addLayout(middle, stretch=1)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.slider_changed)
        layout.addWidget(self.slider)

    def _build_track_tab(self):
        layout = QVBoxLayout(self.track_tab)
        controls = QHBoxLayout()

        self.clip_start_spin = QSpinBox()
        self.clip_start_spin.valueChanged.connect(self.clip_spin_changed)
        self.clip_end_spin = QSpinBox()
        self.clip_end_spin.valueChanged.connect(self.clip_spin_changed)
        self.set_clip_start_button = QPushButton("Set clip start")
        self.set_clip_start_button.clicked.connect(self.set_clip_start_to_current)
        self.set_clip_end_button = QPushButton("Set clip end")
        self.set_clip_end_button.clicked.connect(self.set_clip_end_to_current)
        self.point_name_button = QPushButton("Select point name")
        self.point_name_button.clicked.connect(self.select_point_name)
        self.point_name_label = QLabel("Point: point_1")
        self.delete_point_button = QPushButton("Delete point on frame")
        self.delete_point_button.clicked.connect(self.delete_current_point)
        self.save_annotations_button = QPushButton("Save annotations")
        self.save_annotations_button.clicked.connect(self.save_annotations)
        self.load_annotations_button = QPushButton("Load annotations")
        self.load_annotations_button.clicked.connect(self.load_annotations)
        self.export_csv_button = QPushButton("Export CSV")
        self.export_csv_button.clicked.connect(self.export_annotations_csv)

        form = QFormLayout()
        form.addRow("Start", self.clip_start_spin)
        form.addRow("End", self.clip_end_spin)
        controls.addLayout(form)
        controls.addWidget(self.set_clip_start_button)
        controls.addWidget(self.set_clip_end_button)
        controls.addWidget(self.point_name_button)
        controls.addWidget(self.point_name_label)
        controls.addWidget(self.delete_point_button)
        controls.addWidget(self.save_annotations_button)
        controls.addWidget(self.load_annotations_button)
        controls.addWidget(self.export_csv_button)
        controls.addStretch()
        layout.addLayout(controls)

        body = QHBoxLayout()
        self.track_view = ImageSceneView()
        self.track_view.imageClicked.connect(self.annotate_current_frame)
        body.addWidget(self.track_view, stretch=4)
        side = QVBoxLayout()
        self.annotation_info = QLabel("Click the image to annotate the selected point.")
        self.annotation_info.setWordWrap(True)
        self.point_list = QListWidget()
        side.addWidget(self.annotation_info)
        side.addWidget(QLabel("Annotated points on this frame"))
        side.addWidget(self.point_list, stretch=1)
        body.addLayout(side, stretch=1)
        layout.addLayout(body, stretch=1)

        nav = QHBoxLayout()
        self.track_prev_button = QPushButton("Prev in clip")
        self.track_prev_button.clicked.connect(lambda: self.set_frame(max(self.clip_start, self.current_index - 1)))
        self.track_next_button = QPushButton("Next in clip")
        self.track_next_button.clicked.connect(lambda: self.set_frame(min(self.clip_end, self.current_index + 1)))
        nav.addWidget(self.track_prev_button)
        nav.addWidget(self.track_next_button)
        nav.addStretch()
        layout.addLayout(nav)

    def select_recording(self):
        start_dir = DEFAULT_DATA_ROOT if DEFAULT_DATA_ROOT.exists() else Path.cwd()
        directory = QFileDialog.getExistingDirectory(self, "Select rig_recorder_data folder or camera_frames", str(start_dir))
        if not directory:
            return
        selected = Path(directory)
        camera_dir = selected if selected.name == "camera_frames" else selected / "camera_frames"
        if not camera_dir.exists():
            QMessageBox.warning(self, "Missing camera_frames", f"No camera_frames folder found in:\n{selected}")
            return
        frames = load_frame_records(camera_dir)
        if not frames:
            QMessageBox.warning(self, "No frames", f"No image frames found in:\n{camera_dir}")
            return
        self.dataset_dir = camera_dir.parent
        self.camera_dir = camera_dir
        self.frames = frames
        self.current_index = 0
        self.clip_start = 0
        self.clip_end = len(self.frames) - 1
        self.annotations = {}
        self.cuts = []
        self._configure_ranges()
        self.refresh_cut_list()
        self.set_frame(0)
        self.try_load_sidecars()

    def _configure_ranges(self):
        max_index = max(0, len(self.frames) - 1)
        self.slider.blockSignals(True)
        self.slider.setRange(0, max_index)
        self.slider.blockSignals(False)
        for spin in (self.clip_start_spin, self.clip_end_spin):
            spin.blockSignals(True)
            spin.setRange(0, max_index)
            spin.blockSignals(False)
        self.clip_start_spin.setValue(self.clip_start)
        self.clip_end_spin.setValue(self.clip_end)

    def try_load_sidecars(self):
        cuts_path = self.default_cuts_path()
        annotations_path = self.default_annotations_path()
        if cuts_path and cuts_path.exists():
            self._read_cuts(cuts_path)
        if annotations_path and annotations_path.exists():
            self._read_annotations(annotations_path)

    def set_frame(self, index: int):
        if not self.frames:
            return
        self.current_index = max(0, min(index, len(self.frames) - 1))
        self.slider.blockSignals(True)
        self.slider.setValue(self.current_index)
        self.slider.blockSignals(False)
        self.update_views()

    def slider_changed(self, value: int):
        self.set_frame(value)

    def update_views(self):
        if not self.frames:
            return
        pixmap = QPixmap(str(self.frames[self.current_index].path))
        if pixmap.isNull():
            self.frame_label.setText(f"Unable to load frame {self.current_index}")
            return
        self.video_view.set_pixmap(pixmap)
        self.track_view.set_pixmap(pixmap)
        self.draw_current_overlays()
        record = self.frames[self.current_index]
        timestamp = "unknown" if record.timestamp is None else f"{record.timestamp:.6f}"
        self.frame_label.setText(
            f"Frame index {self.current_index} / {len(self.frames) - 1} | "
            f"file frame {record.frame_number} | timestamp {timestamp}"
        )
        self.refresh_annotation_list()

    def draw_current_overlays(self):
        self.draw_cut_overlay()
        self.draw_annotation_overlay()

    def draw_cut_overlay(self):
        for cut in self.cuts:
            if cut.get("frame_index") == self.current_index:
                self.video_view.add_marker(18, 18, QColor("#e41a1c"), cut.get("label", "cut"))

    def draw_annotation_overlay(self):
        frame_key = str(self.current_index)
        frame_annotations = self.annotations.get(frame_key, {})
        colors = [QColor("#1f77b4"), QColor("#e41a1c"), QColor("#4daf4a"), QColor("#984ea3"), QColor("#ff7f00")]
        for idx, (name, point) in enumerate(sorted(frame_annotations.items())):
            self.track_view.add_marker(point["x"], point["y"], colors[idx % len(colors)], name)

    def toggle_playback(self):
        if not self.frames:
            QMessageBox.information(self, "No recording", "Select a recording first.")
            return
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("Play")
        else:
            self.update_timer_interval()
            self.timer.start()
            self.play_button.setText("Pause")

    def update_timer_interval(self):
        self.timer.setInterval(max(1, int(1000 / self.fps_spin.value())))

    def advance_frame(self):
        if not self.frames:
            return
        next_index = self.current_index + 1
        if next_index >= len(self.frames):
            self.timer.stop()
            self.play_button.setText("Play")
            return
        self.set_frame(next_index)

    def add_cut(self):
        if not self.frames:
            QMessageBox.information(self, "No recording", "Select a recording first.")
            return
        label, ok = QInputDialog.getText(self, "Cut label", "Event label:")
        if not ok:
            return
        record = self.frames[self.current_index]
        self.cuts.append(
            {
                "frame_index": self.current_index,
                "frame_number": record.frame_number,
                "timestamp": record.timestamp,
                "label": label.strip() or f"cut_{len(self.cuts) + 1}",
                "image": record.path.name,
            }
        )
        self.cuts.sort(key=lambda item: item["frame_index"])
        self.refresh_cut_list()
        self.update_views()

    def refresh_cut_list(self):
        self.cut_list.clear()
        for cut in self.cuts:
            text = f'{cut["frame_index"]:06d} | {cut.get("label", "")} | {cut.get("image", "")}'
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, cut["frame_index"])
            self.cut_list.addItem(item)

    def jump_to_cut(self, item: QListWidgetItem):
        self.set_frame(int(item.data(Qt.UserRole)))

    def delete_selected_cut(self):
        row = self.cut_list.currentRow()
        if row < 0:
            return
        del self.cuts[row]
        self.refresh_cut_list()
        self.update_views()

    def default_cuts_path(self) -> Optional[Path]:
        return self.dataset_dir / "rig_replay_cuts.json" if self.dataset_dir else None

    def default_annotations_path(self) -> Optional[Path]:
        return self.dataset_dir / "rig_replay_annotations.json" if self.dataset_dir else None

    def save_cuts(self):
        path = self.default_cuts_path()
        if path is None:
            QMessageBox.information(self, "No recording", "Select a recording first.")
            return
        with path.open("w", encoding="utf-8") as handle:
            json.dump({"camera_frames": str(self.camera_dir), "cuts": self.cuts}, handle, indent=2)
        QMessageBox.information(self, "Cuts saved", f"Saved cuts to:\n{path}")

    def load_cuts(self):
        start = str(self.dataset_dir or Path.cwd())
        path, _ = QFileDialog.getOpenFileName(self, "Load cuts", start, "JSON files (*.json)")
        if path:
            self._read_cuts(Path(path))

    def _read_cuts(self, path: Path):
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.cuts = list(payload.get("cuts", []))
            self.cuts.sort(key=lambda item: item.get("frame_index", 0))
            self.refresh_cut_list()
            self.update_views()
        except Exception as exc:
            QMessageBox.warning(self, "Could not load cuts", str(exc))

    def clip_spin_changed(self):
        if not self.frames:
            return
        start = min(self.clip_start_spin.value(), self.clip_end_spin.value())
        end = max(self.clip_start_spin.value(), self.clip_end_spin.value())
        self.clip_start = start
        self.clip_end = end
        if self.current_index < start or self.current_index > end:
            self.set_frame(start)
        self.update_annotation_info()

    def set_clip_start_to_current(self):
        self.clip_start_spin.setValue(self.current_index)

    def set_clip_end_to_current(self):
        self.clip_end_spin.setValue(self.current_index)

    def select_point_name(self):
        name, ok = QInputDialog.getText(self, "Point name", "Point name:", text=self.current_point_name())
        if ok and name.strip():
            self.point_name_label.setText(f"Point: {name.strip()}")

    def current_point_name(self) -> str:
        return self.point_name_label.text().replace("Point:", "", 1).strip() or "point_1"

    def annotate_current_frame(self, x: float, y: float):
        if not self.frames:
            return
        if self.current_index < self.clip_start or self.current_index > self.clip_end:
            QMessageBox.information(self, "Outside clip", "Move to a frame inside the selected clip.")
            return
        frame_key = str(self.current_index)
        self.annotations.setdefault(frame_key, {})[self.current_point_name()] = {
            "x": round(float(x), 3),
            "y": round(float(y), 3),
            "frame_number": self.frames[self.current_index].frame_number,
            "timestamp": self.frames[self.current_index].timestamp,
            "image": self.frames[self.current_index].path.name,
        }
        self.update_views()

    def delete_current_point(self):
        frame_key = str(self.current_index)
        points = self.annotations.get(frame_key, {})
        points.pop(self.current_point_name(), None)
        if not points and frame_key in self.annotations:
            del self.annotations[frame_key]
        self.update_views()

    def refresh_annotation_list(self):
        self.point_list.clear()
        frame_annotations = self.annotations.get(str(self.current_index), {})
        for name, point in sorted(frame_annotations.items()):
            self.point_list.addItem(f'{name}: x={point["x"]:.1f}, y={point["y"]:.1f}')
        self.update_annotation_info()

    def update_annotation_info(self):
        if not self.frames:
            self.annotation_info.setText("Select a recording, then choose a clip.")
            return
        self.annotation_info.setText(
            f"Clip {self.clip_start}-{self.clip_end}. "
            f"Current frame {self.current_index}. Click the image to set {self.current_point_name()}."
        )

    def save_annotations(self):
        path = self.default_annotations_path()
        if path is None:
            QMessageBox.information(self, "No recording", "Select a recording first.")
            return
        payload = {
            "camera_frames": str(self.camera_dir),
            "clip": {"start": self.clip_start, "end": self.clip_end},
            "annotations": self.annotations,
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        QMessageBox.information(self, "Annotations saved", f"Saved annotations to:\n{path}")

    def load_annotations(self):
        start = str(self.dataset_dir or Path.cwd())
        path, _ = QFileDialog.getOpenFileName(self, "Load annotations", start, "JSON files (*.json)")
        if path:
            self._read_annotations(Path(path))

    def _read_annotations(self, path: Path):
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.annotations = dict(payload.get("annotations", {}))
            clip = payload.get("clip", {})
            self.clip_start = int(clip.get("start", self.clip_start))
            self.clip_end = int(clip.get("end", self.clip_end))
            self._configure_ranges()
            self.update_views()
        except Exception as exc:
            QMessageBox.warning(self, "Could not load annotations", str(exc))

    def export_annotations_csv(self):
        if not self.dataset_dir:
            QMessageBox.information(self, "No recording", "Select a recording first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export annotations CSV",
            str(self.dataset_dir / "rig_replay_annotations.csv"),
            "CSV files (*.csv)",
        )
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["point_name", "frame_index", "frame_number", "timestamp", "x", "y", "image"])
            for frame_key in sorted(self.annotations, key=lambda value: int(value)):
                for point_name, point in sorted(self.annotations[frame_key].items()):
                    writer.writerow(
                        [
                            point_name,
                            frame_key,
                            point.get("frame_number", ""),
                            point.get("timestamp", ""),
                            point.get("x", ""),
                            point.get("y", ""),
                            point.get("image", ""),
                        ]
                    )
        QMessageBox.information(self, "CSV exported", f"Exported annotations to:\n{path}")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self.set_frame(self.current_index - 1)
        elif event.key() == Qt.Key_Right:
            self.set_frame(self.current_index + 1)
        elif event.key() == Qt.Key_Space:
            self.toggle_playback()
        else:
            super().keyPressEvent(event)


def main():
    app = QApplication(sys.argv)
    window = ReplayAnnotator()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
