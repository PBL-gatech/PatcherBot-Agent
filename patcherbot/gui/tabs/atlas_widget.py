"""Native, calibrated mouse atlas reference viewer."""
from PyQt5 import QtCore, QtGui, QtWidgets
from patcherbot.utils.reference.scraper import (
    AtlasCoordinates, AtlasProvider, marker_position, select_plate,
)


class AtlasCanvas(QtWidgets.QWidget):
    """Paint image and marker in the same source-pixel coordinate system."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = QtGui.QImage()
        self.marker = None
        self.setMinimumSize(180, 180)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    def sizeHint(self):
        return QtCore.QSize(420, 300)

    def image_rect(self):
        if self.image.isNull():
            return QtCore.QRectF()
        size = self.image.size().scaled(self.size(), QtCore.Qt.KeepAspectRatio)
        return QtCore.QRectF((self.width() - size.width()) / 2,
                             (self.height() - size.height()) / 2,
                             size.width(), size.height())

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor("white"))
        if self.image.isNull():
            return
        rect = self.image_rect()
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        painter.drawImage(rect, self.image)
        if self.marker is None:
            return
        x, y = self.marker
        if not (0 <= x < self.image.width() and 0 <= y < self.image.height()):
            return
        point = QtCore.QPointF(rect.x() + x * rect.width() / self.image.width(),
                              rect.y() + y * rect.height() / self.image.height())
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(QtGui.QPen(QtGui.QColor("#d7191c"), 2))
        painter.drawLine(point + QtCore.QPointF(-9, 0), point + QtCore.QPointF(9, 0))
        painter.drawLine(point + QtCore.QPointF(0, -9), point + QtCore.QPointF(0, 9))
        painter.setBrush(QtGui.QColor("#d7191c"))
        painter.drawEllipse(point, 2.5, 2.5)


class AtlasWidget(QtWidgets.QGroupBox):
    """Independent reference viewer; coordinates never command rig movement."""

    selectionChanged = QtCore.pyqtSignal(object, str)

    def __init__(self, parent=None, provider=None):
        super().__init__("Mouse Brain Atlas", parent)
        self.provider = provider if provider is not None else AtlasProvider(parent=self)
        self.plates = []
        self.plane = "coronal"
        self.coordinates = AtlasCoordinates()
        self._request_id = 0
        self._started = False
        self._displayed_plate = None
        layout = QtWidgets.QVBoxLayout(self)
        buttons = QtWidgets.QHBoxLayout()
        self.view_group = QtWidgets.QButtonGroup(self)
        self.view_group.setExclusive(True)
        self.view_buttons = {}
        for plane in ("coronal", "sagittal"):
            button = QtWidgets.QPushButton(plane.title())
            button.setCheckable(True)
            button.setChecked(plane == self.plane)
            button.clicked.connect(lambda checked, p=plane: self._set_plane(p))
            self.view_group.addButton(button)
            self.view_buttons[plane] = button
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.canvas = AtlasCanvas(self)
        layout.addWidget(self.canvas, 1)
        self.status = QtWidgets.QLabel("Atlas loads when this window is opened.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.sliders = {}
        self.readouts = {}
        for axis, bounds in (("ml", (-4, 372)), ("ap", (-824, 428)), ("dv", (-800, 0))):
            row = QtWidgets.QHBoxLayout()
            row.addWidget(QtWidgets.QLabel(axis.upper()))
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setObjectName("atlas_" + axis)
            slider.setRange(*bounds)
            slider.setSingleStep(1)
            slider.setPageStep(10)
            slider.setValue(0)
            slider.setTracking(False)
            slider.setEnabled(False)
            value = QtWidgets.QLabel("0.00 mm")
            value.setMinimumWidth(65)
            slider.sliderMoved.connect(lambda n, a=axis: self._preview(a, n))
            slider.valueChanged.connect(self._commit)
            self.sliders[axis] = slider
            self.readouts[axis] = value
            row.addWidget(slider, 1)
            row.addWidget(value)
            layout.addLayout(row)
        self.retry = QtWidgets.QPushButton("Retry")
        self.retry.clicked.connect(self._retry)
        self.retry.hide()
        layout.addWidget(self.retry)
        reference = QtWidgets.QLabel(
            'Coordinates in mm, relative to skull/Bregma; negative DV is depth.<br>'
            '<a href="https://labs.gaidi.ca/mouse-brain-atlas/">Atlas: Matt Gaidica</a>'
            ' · Paxinos &amp; Franklin (2001)')
        reference.setWordWrap(True)
        reference.setOpenExternalLinks(True)
        layout.addWidget(reference)
        self.provider.metadataReady.connect(self._metadata_ready)
        self.provider.imageReady.connect(self._image_ready)
        self.provider.failed.connect(self._failed)

    def showEvent(self, event):
        super().showEvent(event)
        if not self._started:
            self._started = True
            self.status.setText("Loading atlas calibration…")
            self.provider.load_metadata()

    def selection(self):
        """Return committed coordinates and selected plane for future consumers."""
        return self.coordinates, self.plane

    def _preview(self, axis, value):
        self.readouts[axis].setText(f"{value / 100:.2f} mm")

    def _metadata_ready(self, plates):
        self.plates = plates
        for axis, plane in (("ml", "sagittal"), ("ap", "coronal")):
            depths = [p.depth for p in plates if p.plane == plane]
            slider = self.sliders[axis]
            with QtCore.QSignalBlocker(slider):
                slider.setRange(round(min(depths) * 100), round(max(depths) * 100))
        for slider in self.sliders.values():
            slider.setEnabled(True)
        self._commit(force=True)

    def _set_plane(self, plane):
        if plane != self.plane:
            self.plane = plane
            self._commit(force=True)

    def _commit(self, _value=None, force=False):
        if not self.plates:
            return
        coords = AtlasCoordinates(**{a: s.value() / 100 for a, s in self.sliders.items()})
        for axis, slider in self.sliders.items():
            self._preview(axis, slider.value())
        if not force and coords == self.coordinates:
            return
        self.coordinates = coords
        self._request_id += 1
        self.selectionChanged.emit(coords, self.plane)
        plate = select_plate(self.plates, self.plane, coords)
        self.retry.hide()
        if self._displayed_plate == plate and not self.canvas.image.isNull():
            self._image_ready(plate, self.canvas.image, self._request_id)
            return
        self.canvas.image = QtGui.QImage()
        self.canvas.marker = None
        self.canvas.update()
        self.status.setText("Loading selected atlas section…")
        self.provider.load_plate(plate, self._request_id)

    def _image_ready(self, plate, image, request_id):
        if request_id != self._request_id:
            return
        self._displayed_plate = plate
        self.canvas.image = image
        self.canvas.marker = marker_position(plate, self.coordinates)
        x, y = self.canvas.marker
        c = self.coordinates
        axis = "AP" if self.plane == "coronal" else "ML"
        message = (f"Selected ML {c.ml:.2f}, AP {c.ap:.2f}, DV {c.dv:.2f} mm. "
                   f"{self.plane.title()} plate {axis} {plate.depth:.2f} mm.")
        if not (0 <= x < image.width() and 0 <= y < image.height()):
            message += " Selected point is outside this image."
        self.status.setText(message)
        self.retry.hide()
        self.canvas.update()

    def _failed(self, message, request_id):
        if request_id not in (-1, self._request_id):
            return
        self.status.setText("Atlas unavailable: " + message)
        self.retry.show()

    def _retry(self):
        self.retry.hide()
        if not self.plates:
            self.status.setText("Loading atlas calibration…")
            self.provider.load_metadata()
        else:
            self._commit(force=True)


class AtlasWindow(QtWidgets.QDialog):
    """Persistent, resizable atlas popup; closing preserves its selection."""

    def __init__(self, parent=None, provider=None):
        super().__init__(parent)
        self.setWindowTitle("Mouse Brain Atlas")
        self.setWindowFlag(QtCore.Qt.WindowMaximizeButtonHint, True)
        self.setSizeGripEnabled(True)
        layout = QtWidgets.QVBoxLayout(self)
        self.atlas_widget = AtlasWidget(self, provider=provider)
        layout.addWidget(self.atlas_widget)
        self.resize(1000, 800)
