"""Lazy PyQt5 editor for selecting pipette anchors inside an HDF5 dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import h5py
import numpy as np
from PyQt5.QtCore import QPointF, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QKeySequence, QPen, QPixmap
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QShortcut,
    QSpinBox,
    QVBoxLayout,
)

from experiments.HDF5AnchorConverter import (
    AnchorConversionError,
    AnchorSelection,
    inspect_hdf5,
    load_anchor_sidecar,
    save_anchor_sidecar,
)


class AnchorScene(QGraphicsScene):
    """Graphics scene that reports clicks in stored-image coordinates."""

    image_clicked = pyqtSignal(float, float)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        point = event.scenePos()
        if self.sceneRect().contains(point):
            self.image_clicked.emit(float(point.x()), float(point.y()))
            event.accept()
            return
        super().mousePressEvent(event)


class HDF5AnchorDialog(QDialog):
    """Select one retained image-space pipette anchor per HDF5 demonstration."""

    FILTERS = ("All", "Unanchored", "Anchored", "Train", "Validation")

    def __init__(self, source_hdf5: Path | str, parent=None) -> None:
        super().__init__(parent)
        self.source_path = Path(source_hdf5).resolve()
        self.inspection = inspect_hdf5(self.source_path)
        if self.inspection.errors:
            raise AnchorConversionError("\n".join(self.inspection.errors))
        self.anchors: Dict[str, AnchorSelection] = load_anchor_sidecar(self.source_path)
        unknown = sorted(set(self.anchors).difference(self.inspection.demo_keys))
        if unknown:
            raise AnchorConversionError(
                "Anchor sidecar contains demonstrations not present in the source: "
                + ", ".join(unknown)
            )
        self._hf = h5py.File(self.source_path, "r")
        self._visible_demo_keys = list(self.inspection.demo_keys)
        self._demo_index = 0
        self._frame_index = 0
        self._pixmap_item = None
        self._marker_item = None

        self.setWindowTitle(f"Anchor HDF5 — {self.source_path.name}")
        self.resize(1100, 850)
        self._build_ui()
        self._connect()
        self._apply_filter()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        header = QHBoxLayout()
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(self.FILTERS)
        self.demo_combo = QComboBox()
        self.demo_status = QLabel()
        header.addWidget(QLabel("Filter:"))
        header.addWidget(self.filter_combo)
        header.addWidget(QLabel("Demonstration:"))
        header.addWidget(self.demo_combo, 1)
        header.addWidget(self.demo_status)
        root.addLayout(header)

        self.scene = AnchorScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHints(self.view.renderHints())
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        root.addWidget(self.view, 1)

        frame_row = QHBoxLayout()
        self.previous_demo_btn = QPushButton("Previous Demo")
        self.next_demo_btn = QPushButton("Next Demo")
        self.previous_frame_btn = QPushButton("Previous Frame")
        self.next_frame_btn = QPushButton("Next Frame")
        self.frame_spin = QSpinBox()
        self.frame_spin.setMinimum(0)
        self.frame_label = QLabel()
        self.clear_anchor_btn = QPushButton("Clear Anchor")
        frame_row.addWidget(self.previous_demo_btn)
        frame_row.addWidget(self.next_demo_btn)
        frame_row.addSpacing(16)
        frame_row.addWidget(self.previous_frame_btn)
        frame_row.addWidget(self.frame_spin)
        frame_row.addWidget(self.next_frame_btn)
        frame_row.addWidget(self.frame_label, 1)
        frame_row.addWidget(self.clear_anchor_btn)
        root.addLayout(frame_row)

        help_label = QLabel(
            "Navigate to a frame where the pipette is fully in focus, then click the pipette tip. "
            "Selections are saved immediately."
        )
        help_label.setWordWrap(True)
        root.addWidget(help_label)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.accept)
        root.addWidget(buttons)

        self.left_shortcut = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.right_shortcut = QShortcut(QKeySequence(Qt.Key_Right), self)

    def _connect(self) -> None:
        self.filter_combo.currentTextChanged.connect(self._apply_filter)
        self.demo_combo.currentIndexChanged.connect(self._select_demo)
        self.frame_spin.valueChanged.connect(self._select_frame)
        self.previous_demo_btn.clicked.connect(lambda: self._move_demo(-1))
        self.next_demo_btn.clicked.connect(lambda: self._move_demo(1))
        self.previous_frame_btn.clicked.connect(lambda: self._move_frame(-1))
        self.next_frame_btn.clicked.connect(lambda: self._move_frame(1))
        self.clear_anchor_btn.clicked.connect(self._clear_anchor)
        self.scene.image_clicked.connect(self._set_anchor)
        self.left_shortcut.activated.connect(lambda: self._move_frame(-1))
        self.right_shortcut.activated.connect(lambda: self._move_frame(1))

    def _split_for(self, demo_key: str) -> str:
        value = self._hf["data"][demo_key].attrs.get("split", "unknown")
        return value.decode() if isinstance(value, bytes) else str(value)

    def _matches_filter(self, demo_key: str, selected_filter: str) -> bool:
        if selected_filter == "Unanchored":
            return demo_key not in self.anchors
        if selected_filter == "Anchored":
            return demo_key in self.anchors
        if selected_filter == "Train":
            return self._split_for(demo_key).lower() == "train"
        if selected_filter == "Validation":
            return self._split_for(demo_key).lower() in {"valid", "validation"}
        return True

    def _apply_filter(self) -> None:
        current = self.current_demo_key
        selected_filter = self.filter_combo.currentText() or "All"
        self._visible_demo_keys = [
            key for key in self.inspection.demo_keys if self._matches_filter(key, selected_filter)
        ]
        self.demo_combo.blockSignals(True)
        self.demo_combo.clear()
        self.demo_combo.addItems(self._visible_demo_keys)
        if current in self._visible_demo_keys:
            self._demo_index = self._visible_demo_keys.index(current)
        else:
            self._demo_index = 0
        if self._visible_demo_keys:
            self.demo_combo.setCurrentIndex(self._demo_index)
        self.demo_combo.blockSignals(False)
        self._load_demo()

    @property
    def current_demo_key(self) -> Optional[str]:
        if not self._visible_demo_keys:
            return None
        return self._visible_demo_keys[min(self._demo_index, len(self._visible_demo_keys) - 1)]

    def _select_demo(self, index: int) -> None:
        if index < 0:
            return
        self._demo_index = index
        self._load_demo()

    def _load_demo(self) -> None:
        key = self.current_demo_key
        enabled = key is not None
        for widget in (
            self.previous_demo_btn,
            self.next_demo_btn,
            self.previous_frame_btn,
            self.next_frame_btn,
            self.frame_spin,
            self.clear_anchor_btn,
        ):
            widget.setEnabled(enabled)
        if key is None:
            self.scene.clear()
            self.demo_status.setText("No demonstrations match this filter.")
            self.frame_label.clear()
            return
        frame_count = int(self._hf["data"][key]["obs"]["camera_image"].shape[0])
        saved = self.anchors.get(key)
        self._frame_index = min(saved.frame_index if saved else 0, frame_count - 1)
        self.frame_spin.blockSignals(True)
        self.frame_spin.setRange(0, frame_count - 1)
        self.frame_spin.setValue(self._frame_index)
        self.frame_spin.blockSignals(False)
        self._display_frame()

    def _select_frame(self, index: int) -> None:
        self._frame_index = index
        self._display_frame()

    def _move_frame(self, offset: int) -> None:
        if self.current_demo_key is None:
            return
        self.frame_spin.setValue(max(self.frame_spin.minimum(), min(self.frame_spin.maximum(), self._frame_index + offset)))

    def _move_demo(self, offset: int) -> None:
        if not self._visible_demo_keys:
            return
        self.demo_combo.setCurrentIndex(max(0, min(len(self._visible_demo_keys) - 1, self._demo_index + offset)))

    @staticmethod
    def _array_to_qimage(frame: np.ndarray) -> QImage:
        image = np.asarray(frame)
        if image.ndim == 3 and image.shape[0] in {1, 3, 4} and image.shape[-1] not in {1, 3, 4}:
            image = np.moveaxis(image, 0, -1)
        if image.ndim == 3 and image.shape[2] == 1:
            image = image[:, :, 0]
        image = np.ascontiguousarray(image)
        if image.dtype != np.uint8:
            finite = np.nan_to_num(image.astype(np.float64))
            minimum, maximum = float(finite.min()), float(finite.max())
            image = np.zeros_like(finite, dtype=np.uint8) if maximum <= minimum else np.clip(
                (finite - minimum) * 255.0 / (maximum - minimum), 0, 255
            ).astype(np.uint8)
        if image.ndim == 2:
            height, width = image.shape
            return QImage(image.data, width, height, image.strides[0], QImage.Format_Grayscale8).copy()
        if image.ndim != 3 or image.shape[2] not in {3, 4}:
            raise AnchorConversionError(f"Unsupported camera frame shape: {image.shape}")
        height, width, channels = image.shape
        fmt = QImage.Format_RGB888 if channels == 3 else QImage.Format_RGBA8888
        return QImage(image.data, width, height, image.strides[0], fmt).copy()

    def _display_frame(self) -> None:
        key = self.current_demo_key
        if key is None:
            return
        try:
            frame = self._hf["data"][key]["obs"]["camera_image"][self._frame_index]
            pixmap = QPixmap.fromImage(self._array_to_qimage(frame))
        except Exception as exc:
            QMessageBox.critical(self, "Frame error", str(exc))
            return
        self.scene.clear()
        self._pixmap_item = self.scene.addPixmap(pixmap)
        self._pixmap_item.setZValue(-1)
        self.scene.setSceneRect(0.0, 0.0, float(pixmap.width()), float(pixmap.height()))
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        saved = self.anchors.get(key)
        if saved and saved.frame_index == self._frame_index:
            self._draw_marker(QPointF(*saved.pixel_xy))
        status = "anchored" if saved else "unanchored"
        self.demo_status.setText(
            f"{self._split_for(key)} · {status} · "
            f"{len(self.anchors)}/{len(self.inspection.demo_keys)} complete"
        )
        total = int(self._hf["data"][key]["obs"]["camera_image"].shape[0])
        self.frame_label.setText(f"Frame {self._frame_index + 1} of {total}")
        self.clear_anchor_btn.setEnabled(saved is not None)

    def _draw_marker(self, point: QPointF) -> None:
        radius = 4.0
        pen = QPen(Qt.red)
        pen.setWidth(2)
        self._marker_item = self.scene.addEllipse(
            point.x() - radius,
            point.y() - radius,
            radius * 2,
            radius * 2,
            pen,
        )
        self._marker_item.setZValue(2)

    def _set_anchor(self, x: float, y: float) -> None:
        key = self.current_demo_key
        if key is None:
            return
        rect = self.scene.sceneRect()
        if not (0.0 <= x < rect.width() and 0.0 <= y < rect.height()):
            return
        previous = self.anchors.get(key)
        self.anchors[key] = AnchorSelection(self._frame_index, (x, y))
        try:
            save_anchor_sidecar(self.source_path, self.anchors)
        except Exception as exc:
            if previous is None:
                del self.anchors[key]
            else:
                self.anchors[key] = previous
            QMessageBox.critical(self, "Could not save anchor", str(exc))
            return
        if self.filter_combo.currentText() in {"Anchored", "Unanchored"}:
            self._apply_filter()
        else:
            self._display_frame()

    def _clear_anchor(self) -> None:
        key = self.current_demo_key
        if key is None or key not in self.anchors:
            return
        previous = self.anchors.pop(key)
        try:
            save_anchor_sidecar(self.source_path, self.anchors)
        except Exception as exc:
            self.anchors[key] = previous
            QMessageBox.critical(self, "Could not save anchor", str(exc))
            return
        if self.filter_combo.currentText() in {"Anchored", "Unanchored"}:
            self._apply_filter()
        else:
            self._display_frame()

    def _close_hdf5(self) -> None:
        if self._hf is None:
            return
        if self._hf.id.valid:
            self._hf.close()
        self._hf = None

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._close_hdf5()
        super().closeEvent(event)

    def done(self, result: int) -> None:
        self._close_hdf5()
        super().done(result)
