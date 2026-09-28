"""On-demand, cached access to the Gaidica mouse brain atlas.

Calibration and image decoding follow https://labs.gaidi.ca/mouse-brain-atlas/.
Importing this module performs no filesystem or network operations.
"""
from dataclasses import dataclass
import csv
import io
import math
from pathlib import Path

from PyQt5 import QtCore, QtGui, QtNetwork, sip

BASE_URL = "https://labs.gaidi.ca/mouse-brain-atlas/"
CSV_URL = BASE_URL + "mouse-brain-atlas.csv"
IMAGE_KEY = bytes.fromhex(
    "6b1e9f3a8c2d4057e8a1f4c9d2b6e305a7f8c1d4e9b2a60853f7e1c9d4a2b6e8"
)


@dataclass(frozen=True)
class AtlasCoordinates:
    ml: float = 0.0
    ap: float = 0.0
    dv: float = 0.0


@dataclass(frozen=True)
class AtlasPlate:
    index: int
    plane: str
    depth: float
    x0: float
    y0: float
    pxx: float
    pxy: float


def parse_calibration(text):
    """Parse the site's headerless CSV, retaining one-based image indexes."""
    plates = []
    for index, row in enumerate(csv.reader(io.StringIO(text.lstrip("\ufeff"))), 1):
        if not row or not any(value.strip() for value in row):
            continue
        if len(row) != 6:
            raise ValueError("Atlas calibration has an unexpected column count")
        plane = row[0].strip().lower()
        if plane not in ("coronal", "sagittal"):
            raise ValueError("Atlas calibration has an unknown plane")
        values = tuple(float(value) for value in row[1:])
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Atlas calibration contains non-finite values")
        if values[-2] <= 0 or values[-1] <= 0:
            raise ValueError("Atlas calibration has an invalid pixel scale")
        plates.append(AtlasPlate(index, plane, *values))
    if not plates:
        raise ValueError("Atlas calibration is empty")
    return plates


def select_plate(plates, plane, coordinates):
    """Match nearest slice, retaining CSV order for ties and signed ML."""
    if plane not in ("coronal", "sagittal"):
        raise ValueError("Unknown atlas plane")
    depth = coordinates.ap if plane == "coronal" else coordinates.ml
    candidates = [plate for plate in plates if plate.plane == plane]
    if not candidates:
        raise ValueError("Atlas calibration has no " + plane + " plates")
    return min(candidates, key=lambda plate: abs(plate.depth - depth))


def marker_position(plate, coordinates):
    horizontal = coordinates.ml if plate.plane == "coronal" else coordinates.ap
    return plate.x0 - horizontal * plate.pxx, plate.y0 + abs(coordinates.dv) * plate.pxy


def decode_atlas_image(data):
    """Decode the repeating XOR used by the public atlas viewer."""
    return bytes(value ^ IMAGE_KEY[index % len(IMAGE_KEY)] for index, value in enumerate(data))


class AtlasProvider(QtCore.QObject):
    metadataReady = QtCore.pyqtSignal(object)
    imageReady = QtCore.pyqtSignal(object, object, int)
    failed = QtCore.pyqtSignal(str, int)

    def __init__(self, parent=None, cache_dir=None):
        super().__init__(parent)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else Path(
            QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.CacheLocation)
        ) / "mouse_atlas"
        self.network = QtNetwork.QNetworkAccessManager(self)
        self._pending = {}
        self._replies = {}
        self._cached_deliveries = []
        self._closed = False
        # A child timer is destroyed with its owner; a static singleShot is not.
        self._cache_timer = QtCore.QTimer(self)
        self._cache_timer.setSingleShot(True)
        self._cache_timer.timeout.connect(self._deliver_cached)
        app = QtCore.QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.shutdown)

    def load_metadata(self):
        def accept(data):
            plates = parse_calibration(data.decode("utf-8-sig"))
            if not all(any(p.plane == plane for p in plates) for plane in ("coronal", "sagittal")):
                raise ValueError("Atlas calibration is missing an image plane")
            return plates

        self._load(CSV_URL, "mouse-brain-atlas.csv", accept,
                   self.metadataReady.emit, -1)

    def load_plate(self, plate, request_id):
        filename = "Mouse_Brain_Atlas_{}.atlasbin".format(plate.index)

        def accept(data):
            image = QtGui.QImage.fromData(decode_atlas_image(data))
            if image.isNull():
                raise ValueError("Atlas image could not be decoded")
            return image

        self._load(BASE_URL + "images/" + filename, filename, accept,
                   lambda image: self.imageReady.emit(plate, image, request_id), request_id)

    @QtCore.pyqtSlot()
    def _deliver_cached(self):
        deliveries, self._cached_deliveries = self._cached_deliveries, []
        for deliver, result in deliveries:
            if sip.isdeleted(self) or self._closed:
                return
            deliver(result)

    def _load(self, url, filename, validate, deliver, request_id):
        if self._closed:
            return
        path = self.cache_dir / filename
        try:
            result = validate(path.read_bytes())
        except (OSError, ValueError, UnicodeError):
            pass
        else:
            self._cached_deliveries.append((deliver, result))
            self._cache_timer.start(0)
            return
        if url in self._pending:
            self._pending[url].append((deliver, request_id))
            return
        self._pending[url] = [(deliver, request_id)]
        request = QtNetwork.QNetworkRequest(QtCore.QUrl(url))
        request.setAttribute(QtNetwork.QNetworkRequest.RedirectPolicyAttribute,
                             QtNetwork.QNetworkRequest.NoLessSafeRedirectPolicy)
        reply = self.network.get(request)
        timeout = QtCore.QTimer(reply)
        timeout.setSingleShot(True)
        timeout.timeout.connect(reply.abort)
        timeout.start(10000)
        self._replies[reply] = (url, path, validate, timeout)
        # A bound Qt slot is disconnected automatically when this provider dies.
        reply.finished.connect(self._download_finished)

    @QtCore.pyqtSlot()
    def _download_finished(self):
        reply = self.sender()
        context = self._replies.pop(reply, None)
        if context is None or sip.isdeleted(reply):
            return
        url, path, validate, timeout = context
        timeout.stop()
        listeners = self._pending.pop(url, [])
        error = None
        result = None
        try:
            if reply.error() != QtNetwork.QNetworkReply.NoError:
                raise ValueError("Atlas download failed: " + reply.errorString())
            data = bytes(reply.readAll())
            result = validate(data)
            self._cache(path, data)
        except (ValueError, UnicodeError) as exc:
            error = str(exc)
        # Schedule disposal before signals: a receiver may delete this whole tree.
        reply.deleteLater()
        for callback, identifier in listeners:
            if sip.isdeleted(self) or self._closed:
                return
            if error is None:
                callback(result)
            else:
                self.failed.emit(error, identifier)

    @QtCore.pyqtSlot()
    def shutdown(self):
        """Cancel pending work before application teardown."""
        if self._closed:
            return
        self._closed = True
        self._cache_timer.stop()
        self._cached_deliveries.clear()
        replies = list(self._replies)
        self._replies.clear()
        self._pending.clear()
        for reply in replies:
            if not sip.isdeleted(reply):
                reply.finished.disconnect(self._download_finished)
                reply.abort()
                reply.deleteLater()

    @staticmethod
    def _cache(path, data):
        # Cache failure must not prevent displaying a valid downloaded image.
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            output = QtCore.QSaveFile(str(path))
            if output.open(QtCore.QIODevice.WriteOnly):
                if output.write(data) == len(data):
                    output.commit()
                else:
                    output.cancelWriting()
        except OSError:
            pass
