"""Saved scan display. Image reads run off the GUI thread; no images are written."""
from pathlib import Path
import json

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from patcherbot.configs.Config import Config
from patcherbot.gui.ParamConfig import ParamConfig


REFERENCE_MANIFEST = Path(__file__).resolve().parents[2] / "utils" / "scan_reference.json"
MAX_DECODE_BYTES = 512 * 1024 ** 2
OVERVIEW_EDGE = 2048


def resolve_scan_image(path):
    """Resolve a saved image or a reference manifest without copying its pixels."""
    path = Path(path).resolve()
    if path.suffix.lower() != ".json":
        return path
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    image_path = document.get("image_path") or document.get("output_path") or document.get("output")
    if not image_path:
        raise ValueError("The manifest does not contain an image_path, output_path or output.")
    image_path = Path(image_path)
    if not image_path.is_absolute():
        image_path = path.parent / image_path
    return image_path.resolve()


def read_scan_image(path, native=False, enhance=False):
    """Read one bounded display level; TIFF dimensions always use native pixels."""
    import cv2
    import tifffile
    from PIL import Image

    path = resolve_scan_image(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            width, height = page.imagewidth, page.imagelength
        companion = path.with_name(path.stem + "_downsampled_2x" + path.suffix)
        source = path
        if not native and companion.is_file():
            try:
                with tifffile.TiffFile(companion) as tif:
                    page = tif.pages[0]
                    if (page.imagewidth, page.imagelength) == ((width + 1) // 2, (height + 1) // 2):
                        source = companion
            except (OSError, tifffile.TiffFileError):
                pass  # The original remains usable without a readable overview.
        with tifffile.TiffFile(source) as tif:
            page = tif.pages[0]
            decode_bytes = int(np.prod(page.shape, dtype=np.int64)) * page.dtype.itemsize
            display_bytes = page.imagewidth * page.imagelength * 3
            if max(decode_bytes, display_bytes) > MAX_DECODE_BYTES:
                raise ValueError("This display level exceeds the 512 MiB viewer limit.")
            pixels = page.asarray()
    else:
        with Image.open(path) as im:
            width, height = im.size
            if width * height * 4 > MAX_DECODE_BYTES:
                raise ValueError("This image exceeds the 512 MiB viewer limit.")
            if not native:
                im.thumbnail((OVERVIEW_EDGE, OVERVIEW_EDGE))
            pixels = np.asarray(im.convert("RGB")).copy()
    if pixels.ndim == 2:
        pixels = np.repeat(pixels[:, :, None], 3, axis=2)
    if pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
        raise ValueError("Expected a grayscale, RGB, or RGBA scan image.")
    if pixels.dtype != np.uint8:
        if np.issubdtype(pixels.dtype, np.integer):
            pixels = (pixels.astype(np.float32) * (255 / np.iinfo(pixels.dtype).max))
        pixels = np.clip(pixels, 0, 255).astype(np.uint8)
    pixels = pixels[:, :, :3]
    if not native and max(pixels.shape[:2]) > OVERVIEW_EDGE:
        ratio = OVERVIEW_EDGE / max(pixels.shape[:2])
        pixels = cv2.resize(pixels, (max(1, round(pixels.shape[1] * ratio)),
                                     max(1, round(pixels.shape[0] * ratio))),
                            interpolation=cv2.INTER_AREA)
    if enhance:
        # Viewing only: preserve the source file and its native coordinate system.
        gray = cv2.cvtColor(np.ascontiguousarray(pixels), cv2.COLOR_RGB2GRAY)
        valid = np.any(pixels != 0, axis=2)
        values = gray.astype(np.float32)
        sigma = max(0.1, 30.0 * pixels.shape[1] / width)
        mask = valid.astype(np.float32)
        weight = cv2.GaussianBlur(mask, (0, 0), sigma)
        background = cv2.GaussianBlur(values * mask, (0, 0), sigma)
        values -= background / np.maximum(weight, 1e-6)
        values += 127.5
        if valid.any():
            low, high = np.percentile(values[valid], (1, 99))
            if high > low:
                values = (values - low) * (255.0 / (high - low))
        gray = np.ascontiguousarray(np.clip(values, 0, 255).astype(np.uint8))
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        gray[~valid] = 0
        pixels = np.repeat(gray[:, :, None], 3, axis=2)
    return {"path": str(path), "width": width, "height": height,
            "pixels": np.ascontiguousarray(pixels), "native": native,
            "enhanced": enhance}


class _ImageReader(QtCore.QThread):
    loaded = QtCore.pyqtSignal(int, object)
    failed = QtCore.pyqtSignal(int, str)

    def __init__(self, token, path, native, enhance, parent):
        super().__init__(parent)
        self.token, self.path, self.native = token, path, native
        self.enhance = enhance

    def run(self):
        try:
            self.loaded.emit(self.token, read_scan_image(self.path, self.native, self.enhance))
        except Exception as error:
            self.failed.emit(self.token, str(error))


class _ScanView(QtWidgets.QGraphicsView):
    zoom_changed = QtCore.pyqtSignal(float)
    pixel_hovered = QtCore.pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setDragMode(self.ScrollHandDrag)
        self.setTransformationAnchor(self.AnchorUnderMouse)
        self.setResizeAnchor(self.AnchorViewCenter)
        self.setMouseTracking(True)
        self.setBackgroundBrush(QtGui.QColor("#262626"))

    def wheelEvent(self, event):
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        scale = self.transform().m11() * factor
        if 0.005 <= scale <= 16:
            self.scale(factor, factor)
            self.zoom_changed.emit(scale)
        event.accept()

    def mouseMoveEvent(self, event):
        position = self.mapToScene(event.pos())
        self.pixel_hovered.emit(position.x(), position.y())
        super().mouseMoveEvent(event)


class ScanViewerTab(ParamConfig):
    """Saved scan image tab using the existing configuration-tab framework."""

    def __init__(self, config=None, parent=None):
        super().__init__(config if config is not None else Config(name="Scan"),
                         parent=parent, build_ui=False)
        self.image_path = None
        self._reader = None
        self._pending = None
        self._token = 0
        self._displayed_token = None
        self._native = False
        self._native_requested = False
        self._overview_ratio = 1.0
        self._item = None
        self._fit_pending = False
        layout = QtWidgets.QVBoxLayout(self)
        toolbar = QtWidgets.QHBoxLayout()
        for label, action in (("Open scan…", self.open_image),
                              ("Open reference", self.open_reference),
                              ("Fit", self.fit_image),
                              ("Native detail", self.load_native)):
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(action)
            toolbar.addWidget(button)
        self.enhance_detail = QtWidgets.QCheckBox("Enhance detail")
        self.enhance_detail.setToolTip(
            "Viewing only: subtract background, stretch levels, and enhance local contrast")
        self.enhance_detail.toggled.connect(self._enhancement_changed)
        toolbar.addWidget(self.enhance_detail)
        toolbar.addStretch()
        self.coordinates = QtWidgets.QLabel("Image pixels")
        toolbar.addWidget(self.coordinates)
        layout.addLayout(toolbar)
        self.view = _ScanView(self)
        self.view.zoom_changed.connect(self._zoom_changed)
        self.view.pixel_hovered.connect(self._pixel_hovered)
        layout.addWidget(self.view, 1)
        self.status = QtWidgets.QLabel("Open a completed scan or the saved reference.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

    def open_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open saved scan", "", "Scans (*.tif *.tiff *.png *.webp *.json)")
        if path:
            self.load_image(path)

    def open_reference(self):
        self.load_image(REFERENCE_MANIFEST)

    def load_image(self, path):
        self._token += 1
        self._fit_pending = True
        self._native = self._native_requested = False
        self.image_path = str(path)
        self.status.setText(f"Loading overview: {Path(path).name}")
        self._request_read(False)

    def _request_read(self, native):
        request = (self._token, self.image_path, native, self.enhance_detail.isChecked())
        if self._reader is not None:
            self._pending = request
            return
        self._start_reader(request)

    def _enhancement_changed(self, enabled):
        if self.image_path:
            self._token += 1
            native = self._native or self._native_requested
            self.status.setText("Updating saved-image detail…")
            self._request_read(native)

    def _start_reader(self, request):
        self._reader = _ImageReader(*request, self)
        self._reader.loaded.connect(self._display_image)
        self._reader.failed.connect(self._read_failed)
        self._reader.finished.connect(self._reader_finished)
        self._reader.start()

    def _reader_finished(self):
        reader = self._reader
        self._reader = None
        reader.deleteLater()
        if self._pending is not None:
            pending, self._pending = self._pending, None
            self._start_reader(pending)

    def _display_image(self, token, result):
        if token != self._token:
            return
        pixels = result["pixels"]
        height, width = pixels.shape[:2]
        image = QtGui.QImage(pixels.data, width, height, pixels.strides[0],
                            QtGui.QImage.Format_RGB888).copy()
        pixmap = QtGui.QPixmap.fromImage(image)
        if self._item is None:
            self._item = self.view.scene().addPixmap(pixmap)
        else:
            self._item.setPixmap(pixmap)
        self._item.setTransform(QtGui.QTransform.fromScale(
            result["width"] / width, result["height"] / height))
        self.view.setSceneRect(0, 0, result["width"], result["height"])
        self.image_path = result["path"]
        self._displayed_token = token
        self._native = result["native"]
        if not self._native:
            self._overview_ratio = width / result["width"]
        if self._fit_pending:
            self.fit_image()
            self._fit_pending = False
        level = "Native pixels" if self._native else "Downsampled overview; zoom for native detail"
        if result["enhanced"]:
            level += "; enhanced display"
        self.status.setText(f"{Path(self.image_path).name} — {result['width']} × "
                            f"{result['height']} — {level}")

    def _read_failed(self, token, error):
        if token == self._token:
            self._native_requested = False
            self.status.setText(f"Image could not be loaded: {error}")

    def _zoom_changed(self, scale):
        if scale > self._overview_ratio * 1.25:
            self.load_native()

    def _pixel_hovered(self, x, y):
        if self.view.sceneRect().contains(x, y):
            self.coordinates.setText(f"Image x={x:.0f}, y={y:.0f} px")

    def load_native(self):
        if (self.image_path and self._displayed_token == self._token
                and not self._native and not self._native_requested):
            self._native_requested = True
            self.status.setText("Loading native detail…")
            self._request_read(True)

    def fit_image(self):
        if self._item is not None:
            self.view.fitInView(self.view.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def shutdown(self):
        self._pending = None
        if self._reader is not None:
            self._reader.wait()
