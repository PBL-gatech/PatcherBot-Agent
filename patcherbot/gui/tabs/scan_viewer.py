"""Saved scan display with rotation and registration sidecars beside each scan."""
from pathlib import Path
import json

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from patcherbot.configs.Config import Config
from patcherbot.gui.ParamConfig import ParamConfig


REFERENCE_MANIFEST = Path(__file__).resolve().parents[2] / "deepLearning" / "globalModel" / "reference" / "scan_reference.json"
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
    original = path.stat()
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
    identity = path.stat()
    if (identity.st_size, identity.st_mtime_ns) != (original.st_size, original.st_mtime_ns):
        raise ValueError("The scan changed while loading; reopen it.")
    return {"path": str(path), "width": width, "height": height,
            "source_identity": (identity.st_size, identity.st_mtime_ns),
            "pixels": np.ascontiguousarray(pixels), "native": native,
            "enhanced": enhance}


# Approximate AP reference convention, Takatoh et al. (2021), Fig. 5 source data:
# https://pmc.ncbi.nlm.nih.gov/articles/PMC8137149/
BREGMA_CCF_AP_UM = 5400.0


def rotate_scan(pixels, degrees):
    """Return upright RGB pixels and native-edge -> upright-edge affine.

    Scene coordinates remain in the original mosaic; this separate transform
    preserves their relationship to the recorded 4x calibration and tile offsets.
    """
    import cv2
    height, width = pixels.shape[:2]
    matrix = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), -degrees, 1)
    corners = np.array([[-.5, -.5], [width-.5, -.5],
                        [width-.5, height-.5], [-.5, height-.5]])
    bounds = corners @ matrix[:, :2].T + matrix[:, 2]
    low, high = bounds.min(axis=0), bounds.max(axis=0)
    size = np.ceil(high - low - 1e-9).astype(int)
    matrix[:, 2] -= low + .5
    rotated = cv2.warpAffine(pixels, matrix, tuple(size),
                             flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    edges = np.eye(3)
    edges[:2] = matrix
    edges[:2, 2] += .5 - matrix[:, :2] @ np.array([.5, .5])
    return rotated, edges


def atlas_page_summary(plates, bounds):
    """Approximate reference pages; this is not a cross-atlas anatomical fit."""
    coronal = [plate for plate in plates if plate.plane == "coronal"]
    if bounds is None or not coronal:
        return "Approximate reference pages unavailable"
    lo, hi = sorted((BREGMA_CCF_AP_UM - np.asarray(bounds)[0]) / 1000)
    if hi < min(p.depth for p in coronal) or lo > max(p.depth for p in coronal):
        return "Registered AP range is outside the reference atlas"
    endpoints = [min(coronal, key=lambda p: abs(p.depth - value)) for value in (lo, hi)]
    selected = [p.index for p in coronal if lo <= p.depth <= hi] + [p.index for p in endpoints]
    return (f"Approximate reference pages {min(selected)}-{max(selected)} "
            f"(bregma AP {lo:.2f} to {hi:.2f} mm)")


class _AtlasWorker(QtCore.QThread):
    loaded = QtCore.pyqtSignal(int, object)
    failed = QtCore.pyqtSignal(int, str)

    def __init__(self, token, path, angle, registrar, parent):
        super().__init__(parent)
        self.token, self.path, self.angle = token, path, angle
        self.registrar = registrar

    def run(self):
        try:
            from PIL import Image
            from patcherbot.deepLearning.AtlasRegistrar import AtlasRegistrar, save_scan_registration
            source = resolve_scan_image(self.path)
            before = source.stat()
            source_identity = (before.st_size, before.st_mtime_ns)
            scan = read_scan_image(source, native=True, enhance=False)
            pixels, transform = rotate_scan(scan["pixels"], self.angle)
            if self.registrar is None:
                self.registrar = AtlasRegistrar()
            result = self.registrar.register(Image.fromarray(pixels), upright=True)
            if result is None:
                raise ValueError("No atlas registration was returned.")
            overlay = self.registrar.render_overlay(result)
            data = {"result": result, "transform": transform,
                    "source_size": (scan["width"], scan["height"]),
                    "rotation_degrees_clockwise": self.angle, **overlay}
            try:
                current = source.stat()
                if (current.st_size, current.st_mtime_ns) != source_identity:
                    raise ValueError("The scan changed during registration; run mapping again.")
                data["saved_path"] = str(save_scan_registration(
                    scan["path"], result, self.angle, transform, overlay["rgba"],
                    overlay["ccf_bounds_um_ap_dv_ml"], (scan["width"], scan["height"])))
            except Exception as error:
                data["save_error"] = str(error)
            self.loaded.emit(self.token, data)
        except Exception as error:
            self.failed.emit(self.token, str(error))


class _AutoRotateWorker(_AtlasWorker):
    """Estimate orientation without saving or publishing an atlas registration."""

    def run(self):
        try:
            from PIL import Image
            from patcherbot.deepLearning.AtlasRegistrar import AtlasRegistrar
            source = resolve_scan_image(self.path)
            before = source.stat()
            scan = read_scan_image(source, native=True, enhance=False)
            if self.registrar is None:
                self.registrar = AtlasRegistrar()
            angle = self.registrar.estimate_upright_rotation(Image.fromarray(scan["pixels"]))
            after = source.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("The scan changed during auto rotation; try again.")
            self.loaded.emit(self.token, angle)
        except Exception as error:
            self.failed.emit(self.token, str(error))


class _ImageReader(QtCore.QThread):
    loaded = QtCore.pyqtSignal(int, object)
    failed = QtCore.pyqtSignal(int, str)

    def __init__(self, token, path, native, enhance, restore_atlas, parent):
        super().__init__(parent)
        self.token, self.path, self.native = token, path, native
        self.enhance = enhance
        self.restore_atlas = restore_atlas

    def run(self):
        try:
            scan = read_scan_image(self.path, self.native, self.enhance)
            if self.restore_atlas:
                source = Path(scan["path"])
                sidecar = source.with_name(source.stem + "_atlas_registration.npz")
                if sidecar.is_file():
                    try:
                        from patcherbot.deepLearning.AtlasRegistrar import load_scan_registration
                        scan["atlas"] = load_scan_registration(source)
                    except Exception as error:
                        scan["atlas_error"] = str(error)
            self.loaded.emit(self.token, scan)
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
        scale = np.hypot(self.transform().m11(), self.transform().m12()) * factor
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

    scanLoaded = QtCore.pyqtSignal(object)

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
        self._global_overlay_items = []
        self._fit_pending = False
        self._atlas_worker = None
        self._registrar = None
        self._atlas_generation = 0
        self._atlas_result = None
        self._atlas_overlay = None
        self._scan_to_upright = None
        self._atlas_bounds = None
        self._atlas_plates = []
        self._atlas_provider = None
        self._atlas_storage_status = ""
        self._restore_atlas_pending = False
        self.scan_metadata = {}
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
        mapping = QtWidgets.QHBoxLayout()
        mapping.addWidget(QtWidgets.QLabel("Upright rotation"))
        self.rotation = QtWidgets.QDoubleSpinBox()
        self.rotation.setRange(-180, 180)
        self.rotation.setDecimals(2)
        self.rotation.setSingleStep(.5)
        self.rotation.setSuffix(" deg clockwise")
        self.rotation.valueChanged.connect(self._rotation_changed)
        mapping.addWidget(self.rotation)
        self.auto_rotate_button = QtWidgets.QPushButton("Auto rotate")
        self.auto_rotate_button.setEnabled(False)
        self.auto_rotate_button.setToolTip("Estimate the tilt and upside-down orientation, then review before mapping")
        self.auto_rotate_button.clicked.connect(self.auto_rotate)
        mapping.addWidget(self.auto_rotate_button)
        self.map_button = QtWidgets.QPushButton("Map upright scan")
        self.map_button.setEnabled(False)
        self.map_button.clicked.connect(self.map_atlas)
        mapping.addWidget(self.map_button)
        self.show_atlas = QtWidgets.QCheckBox("Atlas overlay")
        self.show_atlas.setChecked(True)
        self.show_atlas.toggled.connect(self._overlay_visibility)
        mapping.addWidget(self.show_atlas)
        self.show_global_overlay = QtWidgets.QCheckBox("Show 40x / pipette")
        self.show_global_overlay.setChecked(True)
        self.show_global_overlay.toggled.connect(self._global_overlay_visibility)
        mapping.addWidget(self.show_global_overlay)
        mapping.addStretch()
        layout.addLayout(mapping)
        self.atlas_status = QtWidgets.QLabel("Rotate the scan upright, then map it.")
        self.atlas_status.setWordWrap(True)
        layout.addWidget(self.atlas_status)
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
        self.set_global_overlay(None)
        self._invalidate_atlas()
        self._restore_atlas_pending = True
        with QtCore.QSignalBlocker(self.rotation):
            self.rotation.setValue(0)
        self.view.resetTransform()
        self.map_button.setEnabled(False)
        self.auto_rotate_button.setEnabled(False)
        self.scan_metadata = {}
        try:
            source = Path(path)
            metadata = source if source.suffix.lower() == ".json" else source.with_name(source.stem + "_meta.json")
            if metadata.is_file():
                self.scan_metadata = json.loads(metadata.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            pass  # Image viewing remains available without calibration metadata.
        self._token += 1
        self._fit_pending = True
        self._native = self._native_requested = False
        self.image_path = str(path)
        self.status.setText(f"Loading overview: {Path(path).name}")
        self._request_read(False)

    def _request_read(self, native):
        request = (self._token, self.image_path, native, self.enhance_detail.isChecked(),
                   self._restore_atlas_pending)
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
        self.map_button.setEnabled(self._atlas_worker is None)
        self.auto_rotate_button.setEnabled(self._atlas_worker is None)
        self._native = result["native"]
        if not self._native:
            self._overview_ratio = width / result["width"]
        if self._restore_atlas_pending:
            self._restore_atlas_pending = False
            saved = result.get("atlas")
            if saved is not None:
                with QtCore.QSignalBlocker(self.rotation):
                    self.rotation.setValue(saved["rotation_degrees_clockwise"])
                self.view.setTransform(QtGui.QTransform().rotate(self.rotation.value()))
                self._atlas_loaded(self._atlas_generation, saved)
            elif result.get("atlas_error"):
                self.atlas_status.setText("Saved registration could not be loaded: " + result["atlas_error"])
        if self._fit_pending:
            self.fit_image()
            self._fit_pending = False
        level = "Native pixels" if self._native else "Downsampled overview; zoom for native detail"
        if result["enhanced"]:
            level += "; enhanced display"
        self.status.setText(f"{Path(self.image_path).name} — {result['width']} × "
                            f"{result['height']} — {level}")
        metadata = dict(self.scan_metadata) if isinstance(self.scan_metadata, dict) else {}
        self.scanLoaded.emit(dict(result, metadata=metadata))

    def set_global_overlay(self, corners_px, pipette_px=None, pipette_direction_px=None):
        """Display anchored positions in native scan pixels without moving the rig."""
        scene = self.view.scene()
        if corners_px is None or self._item is None:
            for item in self._global_overlay_items:
                scene.removeItem(item)
            self._global_overlay_items = []
            return
        corners = np.asarray(corners_px, dtype=float)
        if corners.shape != (4, 2) or not np.isfinite(corners).all():
            raise ValueError("The 40x footprint must contain four finite pixel coordinates.")
        pen = QtGui.QPen(QtGui.QColor("#00e5ff"), 2)
        pen.setCosmetic(True)
        polygon = QtGui.QPolygonF([QtCore.QPointF(float(x), float(y)) for x, y in corners])
        if self._global_overlay_items:
            self._global_overlay_items[0].setPolygon(polygon)
        else:
            self._global_overlay_items.append(scene.addPolygon(polygon, pen))
        marker = self._global_overlay_items[1] if len(self._global_overlay_items) > 1 else None
        expected_type = QtWidgets.QGraphicsEllipseItem if pipette_direction_px is None else QtWidgets.QGraphicsPolygonItem
        if marker is not None and (pipette_px is None or not isinstance(marker, expected_type)):
            scene.removeItem(marker)
            self._global_overlay_items.pop()
            marker = None
        if pipette_px is not None:
            point = np.asarray(pipette_px, dtype=float)
            if point.shape != (2,) or not np.isfinite(point).all():
                raise ValueError("The pipette marker must contain two finite pixel coordinates.")
            if pipette_direction_px is None:
                if marker is None:
                    marker = QtWidgets.QGraphicsEllipseItem(-4, -4, 8, 8)
            else:
                direction = np.asarray(pipette_direction_px, dtype=float)
                if direction.shape != (2,) or not np.isfinite(direction).all() or np.linalg.norm(direction) == 0:
                    raise ValueError("Pipette direction must contain two finite nonzero pixel components.")
                if marker is None:
                    marker = QtWidgets.QGraphicsPolygonItem()
                marker.setData(0, direction / np.linalg.norm(direction))
            marker.setPen(QtGui.QPen(QtGui.QColor("#ff0000"), 1))
            marker.setBrush(QtGui.QBrush(QtGui.QColor("#ff0000")))
            marker.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations)
            marker.setPos(float(point[0]), float(point[1]))
            if len(self._global_overlay_items) == 1:
                scene.addItem(marker)
                self._global_overlay_items.append(marker)
        self._orient_pipette_arrow()
        for item in self._global_overlay_items:
            item.setZValue(20)
        self._global_overlay_visibility(self.show_global_overlay.isChecked())

    def _orient_pipette_arrow(self):
        """Keep the arrow screen-sized while its direction follows scan rotation."""
        transform = self.view.transform()
        origin = transform.map(QtCore.QPointF(0., 0.))
        for marker in self._global_overlay_items:
            direction = marker.data(0)
            if direction is None:
                continue
            endpoint = transform.map(QtCore.QPointF(*direction)) - origin
            direction = np.array([endpoint.x(), endpoint.y()])
            direction /= np.linalg.norm(direction)
            side = np.array([-direction[1], direction[0]])
            marker.setPolygon(QtGui.QPolygonF([QtCore.QPointF(0., 0.),
                QtCore.QPointF(*(-12 * direction + 5 * side)),
                QtCore.QPointF(*(-12 * direction - 5 * side))]))

    def _global_overlay_visibility(self, visible):
        for item in self._global_overlay_items:
            item.setVisible(visible)

    def _read_failed(self, token, error):
        if token == self._token:
            self._native_requested = False
            self.status.setText(f"Image could not be loaded: {error}")

    def _zoom_changed(self, scale):
        if scale > self._overview_ratio * 1.25:
            self.load_native()

    def _pixel_hovered(self, x, y):
        if self.view.sceneRect().contains(x, y):
            text = f"Image x={x:.0f}, y={y:.0f} px"
            if self._atlas_result is not None:
                point = self._scan_to_upright @ np.array([x, y, 1.])
                px, py = point[:2] - .5
                result = self._atlas_result
                if 0 <= px < result.width and 0 <= py < result.height:
                    ap, dv, ml = result.pixel_to_ccf_um(px, py)
                    text += f" | Allen CCF AP {ap:.0f}, DV {dv:.0f}, ML {ml:.0f} um"
            self.coordinates.setText(text)

    def load_native(self):
        if (self.image_path and self._displayed_token == self._token
                and not self._native and not self._native_requested):
            self._native_requested = True
            self.status.setText("Loading native detail…")
            self._request_read(True)

    def fit_image(self):
        if self._item is not None:
            self.view.fitInView(self.view.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def _invalidate_atlas(self):
        self._atlas_generation += 1
        self._atlas_storage_status = ""
        self._atlas_result = self._atlas_bounds = self._scan_to_upright = None
        if self._atlas_overlay is not None:
            self.view.scene().removeItem(self._atlas_overlay)
            self._atlas_overlay = None
        self.atlas_status.setText("Rotate the scan upright, then map it.")

    def _rotation_changed(self, angle):
        self._restore_atlas_pending = False
        self._invalidate_atlas()
        self.view.setTransform(QtGui.QTransform().rotate(angle))
        self._orient_pipette_arrow()
        self.fit_image()

    def _overlay_visibility(self, visible):
        if self._atlas_overlay is not None:
            self._atlas_overlay.setVisible(visible)

    def auto_rotate(self):
        if self._atlas_worker is not None or self._displayed_token != self._token:
            return
        self._restore_atlas_pending = False
        self._atlas_generation += 1
        self.map_button.setEnabled(False)
        self.auto_rotate_button.setEnabled(False)
        self.atlas_status.setText("Estimating upright rotation; first use may download atlas/model assets...")
        worker = self._atlas_worker = _AutoRotateWorker(
            self._atlas_generation, self.image_path, 0, self._registrar, self)
        worker.loaded.connect(self._auto_rotation_loaded)
        worker.failed.connect(self._auto_rotation_failed)
        worker.finished.connect(self._atlas_finished)
        worker.start()

    def _auto_rotation_loaded(self, token, angle):
        if token != self._atlas_generation:
            return
        self.rotation.setValue(angle)
        self.atlas_status.setText(f"Auto rotation: {angle:.2f} deg clockwise. Review, then click Map upright scan.")

    def _auto_rotation_failed(self, token, error):
        if token == self._atlas_generation:
            self.atlas_status.setText(f"Auto rotation unavailable: {error}")

    def map_atlas(self):
        if self._atlas_worker is not None or self._displayed_token != self._token:
            return
        self._restore_atlas_pending = False
        self._invalidate_atlas()
        self.map_button.setEnabled(False)
        self.auto_rotate_button.setEnabled(False)
        self.atlas_status.setText("Registering upright scan; first use may download atlas/model assets...")
        worker = self._atlas_worker = _AtlasWorker(
            self._atlas_generation, self.image_path, self.rotation.value(), self._registrar, self)
        worker.loaded.connect(self._atlas_loaded)
        worker.failed.connect(self._atlas_failed)
        worker.finished.connect(self._atlas_finished)
        worker.start()

    def _atlas_loaded(self, token, data):
        if token != self._atlas_generation:
            return
        if data.get("save_error"):
            self._atlas_storage_status = "Registration not saved: " + data["save_error"]
        elif data.get("saved_path"):
            self._atlas_storage_status = "Saved: " + str(data["saved_path"])
        self._atlas_result = result = data["result"]
        self._scan_to_upright = data["transform"]
        self._atlas_bounds = data["ccf_bounds_um_ap_dv_ml"]
        rgba = np.ascontiguousarray(data["rgba"])
        height, width = rgba.shape[:2]
        image = QtGui.QImage(rgba.data, width, height, rgba.strides[0], QtGui.QImage.Format_RGBA8888).copy()
        item = self._atlas_overlay = self.view.scene().addPixmap(QtGui.QPixmap.fromImage(image))
        matrix = np.linalg.inv(self._scan_to_upright) @ np.diag([result.width/width, result.height/height, 1.])
        item.setTransform(QtGui.QTransform(matrix[0, 0], matrix[1, 0], matrix[0, 1],
                                          matrix[1, 1], matrix[0, 2], matrix[1, 2]))
        item.setZValue(1)
        item.setVisible(self.show_atlas.isChecked())
        self._update_atlas_status()
        if self._atlas_provider is None:
            from patcherbot.utils.reference.scraper import AtlasProvider
            self._atlas_provider = AtlasProvider(self)
            self._atlas_provider.metadataReady.connect(self._atlas_pages_ready)
            self._atlas_provider.failed.connect(self._atlas_pages_failed)
        if not self._atlas_plates:
            self._atlas_provider.load_metadata()

    def _atlas_pages_ready(self, plates):
        self._atlas_plates = plates
        self._update_atlas_status()

    def _atlas_pages_failed(self, message, request_id):
        if self._atlas_result is not None:
            self._update_atlas_status()
            self.atlas_status.setText(self.atlas_status.text() + "; reference pages unavailable: " + message)

    def _update_atlas_status(self):
        if self._atlas_result is None:
            return
        summary = atlas_page_summary(self._atlas_plates, self._atlas_bounds)
        self.atlas_status.setText(f"Atlas fit Dice {self._atlas_result.silhouette_dice:.3f}. {summary}. "
                                 "Hover for Allen CCF coordinates. " + self._atlas_storage_status)

    def _atlas_failed(self, token, error):
        if token == self._atlas_generation:
            self.atlas_status.setText(f"Atlas mapping unavailable: {error}")

    def _atlas_finished(self):
        worker, self._atlas_worker = self._atlas_worker, None
        self._registrar = worker.registrar
        worker.deleteLater()
        self.map_button.setEnabled(self._displayed_token == self._token)
        self.auto_rotate_button.setEnabled(self._displayed_token == self._token)

    def shutdown(self):
        self._pending = None
        self._atlas_generation += 1
        if self._atlas_worker is not None:
            self._atlas_worker.wait()
        if self._atlas_provider is not None:
            self._atlas_provider.shutdown()
        if self._reader is not None:
            self._reader.wait()


class _AtlasStorageWorker(QtCore.QThread):
    loaded = QtCore.pyqtSignal(int, object)
    failed = QtCore.pyqtSignal(int, str)

    def __init__(self, token, path, identity, data, parent):
        super().__init__(parent)
        self.token, self.path, self.identity, self.data = token, path, identity, data

    def run(self):
        try:
            from patcherbot.deepLearning.AtlasRegistrar import load_scan_registration, save_scan_registration
            stat = Path(self.path).stat()
            if self.identity is None or (stat.st_size, stat.st_mtime_ns) != tuple(self.identity):
                raise ValueError("The source scan changed; reopen it before saving or reloading alignment.")
            if self.data is None:
                data = load_scan_registration(self.path)
                if data is None:
                    raise ValueError("No saved atlas alignment exists for this scan.")
            else:
                data = dict(self.data)
                data["saved_path"] = str(save_scan_registration(
                    self.path, data["result"], data["rotation_degrees_clockwise"],
                    data["transform"], data["rgba"], data["ccf_bounds_um_ap_dv_ml"], data["source_size"]))
                data.pop("save_error", None)
            self.loaded.emit(self.token, data)
        except Exception as error:
            self.failed.emit(self.token, str(error))


class LinkedScanViewerTab(ScanViewerTab):
    """Publish existing Atlas display state for the registered 3D scene."""

    atlasOverlayChanged = QtCore.pyqtSignal(object)

    def __init__(self, *args, **kwargs):
        self._published_atlas = None
        self._alignment_base = self._alignment_data = None
        self._source_identity = None
        self._storage_worker = None
        super().__init__(*args, **kwargs)
        row = QtWidgets.QHBoxLayout()
        for name, label, low, high, default, step in (
                ("tx", "Atlas X px", -100000., 100000., 0., 1.),
                ("ty", "Y px", -100000., 100000., 0., 1.),
                ("angle", "Rotation", -180., 180., 0., .5),
                ("scale", "Scale", .01, 100., 1., .01)):
            row.addWidget(QtWidgets.QLabel(label))
            control = QtWidgets.QDoubleSpinBox()
            control.setRange(low, high)
            control.setDecimals(3)
            control.setSingleStep(step)
            control.setValue(default)
            control.valueChanged.connect(self._alignment_changed)
            setattr(self, "atlas_" + name, control)
            row.addWidget(control)
        self.save_alignment_button = QtWidgets.QPushButton("Save alignment")
        self.reload_alignment_button = QtWidgets.QPushButton("Reload alignment")
        self.save_alignment_button.clicked.connect(self.save_alignment)
        self.reload_alignment_button.clicked.connect(self.reload_alignment)
        row.addWidget(self.save_alignment_button)
        row.addWidget(self.reload_alignment_button)
        self.layout().insertLayout(2, row)
        self.atlas_position_status = QtWidgets.QLabel("Atlas position: load or save an alignment.")
        self.atlas_position_status.setWordWrap(True)
        self.layout().insertWidget(4, self.atlas_position_status)

    def _display_image(self, token, result):
        if token == self._token:
            self._source_identity = result.get("source_identity")
        super()._display_image(token, result)

    def _atlas_loaded(self, token, data):
        if token != self._atlas_generation:
            return
        if data.get("saved_path"):
            with QtCore.QSignalBlocker(self.rotation):
                self.rotation.setValue(data["rotation_degrees_clockwise"])
            self.view.setTransform(QtGui.QTransform().rotate(self.rotation.value()))
            self._orient_pipette_arrow()
        if self._atlas_overlay is not None:
            self.view.scene().removeItem(self._atlas_overlay)
            self._atlas_overlay = None
        super()._atlas_loaded(token, data)
        self._alignment_base = np.asarray(data["transform"], dtype=float).copy()
        self._alignment_data = dict(data)
        for name, value in (("tx", 0), ("ty", 0), ("angle", 0), ("scale", 1)):
            control = getattr(self, "atlas_" + name)
            with QtCore.QSignalBlocker(control):
                control.setValue(value)
        self._publish_alignment()

    def _publish_alignment(self):
        data = self._alignment_data
        result = data["result"]
        self._published_atlas = {
            "rgba": np.asarray(data["rgba"]), "width": result.width,
            "height": result.height, "scan_to_upright": self._scan_to_upright.copy(),
            "transform": self._scan_to_upright.copy(), "result": result,
            "source_size": data.get("source_size", (int(self.view.sceneRect().width()), int(self.view.sceneRect().height()))),
            "scan_path": self.image_path,
            "visible": self.show_atlas.isChecked(),
        }
        self.atlasOverlayChanged.emit(self._published_atlas)

    def _alignment_changed(self, _value=None):
        if self._alignment_base is None:
            return
        center = np.array([self.view.sceneRect().width(), self.view.sceneRect().height()]) / 2
        angle = np.deg2rad(self.atlas_angle.value())
        linear = self.atlas_scale.value() * np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        adjustment = np.eye(3)
        adjustment[:2, :2] = linear
        adjustment[:2, 2] = center + [self.atlas_tx.value(), self.atlas_ty.value()] - linear @ center
        self._scan_to_upright = self._alignment_base @ np.linalg.inv(adjustment)
        result = self._atlas_result
        height, width = self._alignment_data["rgba"].shape[:2]
        matrix = np.linalg.inv(self._scan_to_upright) @ np.diag([result.width/width, result.height/height, 1.])
        self._atlas_overlay.setTransform(QtGui.QTransform(matrix[0, 0], matrix[1, 0], matrix[0, 1],
                                                          matrix[1, 1], matrix[0, 2], matrix[1, 2]))
        self._atlas_storage_status = "Alignment changed; save to keep it. Original fit score is unchanged."
        self._update_atlas_status()
        self._publish_alignment()

    def save_alignment(self):
        if self._alignment_data is None:
            return
        data = dict(self._alignment_data, transform=self._scan_to_upright.copy(),
                    source_size=self._published_atlas["source_size"],
                    rotation_degrees_clockwise=self.rotation.value())
        self._start_storage(data)

    def reload_alignment(self):
        self._start_storage(None)

    def _start_storage(self, data):
        if self.image_path is None or self._storage_worker is not None or self._atlas_worker is not None:
            return
        worker = self._storage_worker = _AtlasStorageWorker(
            self._atlas_generation, self.image_path, self._source_identity, data, self)
        worker.loaded.connect(self._atlas_loaded)
        worker.failed.connect(self._atlas_failed)
        worker.finished.connect(self._storage_finished)
        self.save_alignment_button.setEnabled(False)
        self.reload_alignment_button.setEnabled(False)
        for name in ("tx", "ty", "angle", "scale"):
            getattr(self, "atlas_" + name).setEnabled(False)
        worker.start()

    def _storage_finished(self):
        worker, self._storage_worker = self._storage_worker, None
        worker.deleteLater()
        self.save_alignment_button.setEnabled(True)
        self.reload_alignment_button.setEnabled(True)
        for name in ("tx", "ty", "angle", "scale"):
            getattr(self, "atlas_" + name).setEnabled(True)

    def _invalidate_atlas(self):
        super()._invalidate_atlas()
        self._alignment_base = self._alignment_data = None
        self._published_atlas = None
        self.atlasOverlayChanged.emit(None)

    def shutdown(self):
        if self._storage_worker is not None:
            self._storage_worker.wait()
        super().shutdown()

    def _overlay_visibility(self, visible):
        super()._overlay_visibility(visible)
        if self._published_atlas is not None:
            self._published_atlas = dict(self._published_atlas, visible=bool(visible))
            self.atlasOverlayChanged.emit(self._published_atlas)


class ScanWorkspace(ParamConfig):
    """Independent saved-bottom and session views using the existing scan widget."""

    scanLoaded = QtCore.pyqtSignal(object)
    layerLoaded = QtCore.pyqtSignal(str, object)
    atlasOverlayChanged = QtCore.pyqtSignal(str, object)

    def __init__(self, config=None, parent=None):
        super().__init__(config if config is not None else Config(name="Scan"),
                         parent=parent, build_ui=False)
        self.tabs = QtWidgets.QTabWidget(self)
        self.reference_view = LinkedScanViewerTab(parent=self.tabs)
        self.session_view = LinkedScanViewerTab(parent=self.tabs)
        self.tabs.addTab(self.reference_view, "Reference bottom")
        self.tabs.addTab(self.session_view, "Session scan / Allen")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs)
        self._results = {}
        self._atlas_results = {}
        self.reference_view.scanLoaded.connect(lambda result: self._loaded("reference", result))
        self.session_view.scanLoaded.connect(lambda result: self._loaded("session", result))
        self.reference_view.atlasOverlayChanged.connect(lambda result: self._atlas_changed("reference", result))
        self.session_view.atlasOverlayChanged.connect(lambda result: self._atlas_changed("session", result))
        self.tabs.currentChanged.connect(self._selection_changed)
        QtCore.QTimer.singleShot(0, self.load_default_reference)

    @property
    def active_view(self):
        return self.tabs.currentWidget()

    @property
    def active_role(self):
        return "reference" if self.active_view is self.reference_view else "session"

    @property
    def image_path(self):
        return self.active_view.image_path

    @property
    def scan_metadata(self):
        return self.active_view.scan_metadata

    def load_default_reference(self):
        """Load the saved map only; viewing never invokes a donor rebuild."""
        try:
            from patcherbot.deepLearning.ObjectiveRegistrar import ObjectiveRegistrar
            source = ObjectiveRegistrar.shared().reference_image_path
            if source is None or not Path(source).is_file():
                raise ValueError("Saved reference image is unavailable")
            self.reference_view.load_image(source)
        except (ImportError, OSError, ValueError, KeyError) as error:
            self.reference_view.status.setText("Reference could not be loaded: " + str(error))

    def _loaded(self, role, result):
        result = dict(result)
        result["metadata"] = dict(result.get("metadata", {}), registration_role=role)
        self._results[role] = result
        self.layerLoaded.emit(role, result)
        if role == self.active_role:
            self.scanLoaded.emit(result)

    def _atlas_changed(self, role, result):
        self._atlas_results[role] = result
        self.atlasOverlayChanged.emit(role, result)

    def _selection_changed(self, index):
        result = self._results.get(self.active_role)
        if result is not None:
            self.scanLoaded.emit(result)
        else:
            self.active_view.set_global_overlay(None)

    def open_image(self):
        self.tabs.setCurrentWidget(self.session_view)
        self.session_view.open_image()

    def load_image(self, path):
        """Newly completed scans go to the session without replacing the bottom."""
        self.tabs.setCurrentWidget(self.session_view)
        self.session_view.load_image(path)

    def open_reference(self):
        self.tabs.setCurrentWidget(self.reference_view)
        self.load_default_reference()

    def set_global_overlay(self, corners_px, pipette_px=None, pipette_direction_px=None):
        self.active_view.set_global_overlay(corners_px, pipette_px, pipette_direction_px)

    def shutdown(self):
        self.reference_view.shutdown()
        self.session_view.shutdown()
