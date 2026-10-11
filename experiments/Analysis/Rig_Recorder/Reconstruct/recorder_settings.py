"""Persistent replay controls and display assumptions for the rig recorder."""

import math

from PyQt5 import QtCore, QtWidgets


DEFAULT_SETTINGS = {
    "tracking_states": True,
    "enable_pipette_detection": True,
    "enable_pipette_focus": True,
    "enable_cell_detection": True,
    "cell_detector_model": "pidnet",
    "pipette_detector_model": "netx",
    "apply_microscope_offset": True,
    "microscope_offset_um": 1350.0,
    "apply_cell_distance": True,
    # A configurable visual assumption, not a measured cell depth.
    "cell_distance_um": 20.0,
    "approach_cell_distance_um": 75.0,
    "centered_cell_only": True,
    "export_aspect_ratio": "16:9",
    "export_inference": True,
    "export_fps": 30,
}

CELL_MODELS = (("PIDNet", "pidnet"), ("YOLO", "yolo"))
PIPETTE_MODELS = (
    ("YOLO", "yolo"),
    ("ONNX", "onnx"),
    ("DINO", "dino"),
    ("NetX", "netx"),
)


def normalize_settings(values=None):
    """Return known settings with valid types and safe fallbacks for old stores."""
    values = values or {}
    result = dict(DEFAULT_SETTINGS)
    for key, default in DEFAULT_SETTINGS.items():
        value = values.get(key, default)
        if isinstance(default, bool):
            if isinstance(value, str):
                normalized = value.strip().lower()
                value = normalized in ("true", "1", "yes", "on") if normalized in (
                    "true", "1", "yes", "on", "false", "0", "no", "off"
                ) else default
            result[key] = bool(value) if value is not None else default
        elif isinstance(default, int):
            try:
                number = int(value)
                if 1 <= number <= 120:
                    result[key] = number
            except (TypeError, ValueError, OverflowError):
                pass
        elif isinstance(default, float):
            try:
                number = float(value)
                if math.isfinite(number) and 0.0 <= number <= 100000.0:
                    result[key] = number
            except (TypeError, ValueError):
                pass
        else:
            result[key] = str(value)
    choices = {
        "cell_detector_model": {model for _, model in CELL_MODELS},
        "pipette_detector_model": {model for _, model in PIPETTE_MODELS},
        "export_aspect_ratio": {"16:9", "4:3"},
    }
    for key, allowed in choices.items():
        if result[key] not in allowed:
            result[key] = DEFAULT_SETTINGS[key]
    return result


def _settings_store():
    return QtCore.QSettings("PatcherBot", "RigRecorderReplay")


def load_settings(store=None):
    """Read replay preferences without depending on application initialization."""
    store = _settings_store() if store is None else store
    store.beginGroup("replay")
    try:
        values = {key: store.value(key, default) for key, default in DEFAULT_SETTINGS.items()}
    finally:
        store.endGroup()
    return normalize_settings(values)


def save_settings(values, store=None):
    """Persist only supported preferences and return the normalized values."""
    normalized = normalize_settings(values)
    store = _settings_store() if store is None else store
    store.beginGroup("replay")
    try:
        for key, value in normalized.items():
            store.setValue(key, value)
    finally:
        store.endGroup()
    store.sync()
    if store.status() != QtCore.QSettings.NoError:
        raise OSError("Unable to save rig recorder settings.")
    return normalized


class RecorderSettingsDialog(QtWidgets.QDialog):
    """Edit replay preferences; get_settings() returns accepted values."""

    def __init__(self, settings=None, parent=None, store=None):
        super().__init__(parent)
        self.setWindowTitle("Rig Recorder Settings")
        self.resize(530, 430)
        self._store = store
        self.settings = normalize_settings(load_settings(store) if settings is None else settings)
        self._controls = {}
        layout = QtWidgets.QVBoxLayout(self)
        tabs = QtWidgets.QTabWidget(self)
        layout.addWidget(tabs)

        inference = QtWidgets.QWidget()
        inference_layout = QtWidgets.QFormLayout(inference)
        self._add_checkbox(inference_layout, "tracking_states", "Track recorded states")
        self._add_checkbox(inference_layout, "enable_pipette_detection", "Detect pipette tip")
        self._add_combo(inference_layout, "pipette_detector_model", "Pipette model", PIPETTE_MODELS)
        self._add_checkbox(inference_layout, "enable_pipette_focus", "Estimate pipette focus")
        self._add_checkbox(inference_layout, "enable_cell_detection", "Detect cells")
        self._add_combo(inference_layout, "cell_detector_model", "Cell model", CELL_MODELS)
        self._add_checkbox(inference_layout, "centered_cell_only", "Show only the cell nearest the image center")
        self._controls["centered_cell_only"].setToolTip(
            "Show one cell in replay and export, including recordings without state labels. "
            "Cell parameters are shown during hunt, seal, and break in."
        )
        note = QtWidgets.QLabel("Use the replay inference button to start or stop all selected models.")
        note.setWordWrap(True)
        inference_layout.addRow(note)
        tabs.addTab(inference, "States and inference")

        geometry = QtWidgets.QWidget()
        geometry_layout = QtWidgets.QFormLayout(geometry)
        self._add_checkbox(geometry_layout, "apply_microscope_offset", "Apply microscope Z offset")
        self._add_distance(geometry_layout, "microscope_offset_um", "Microscope above pipette", "apply_microscope_offset")
        self._add_checkbox(geometry_layout, "apply_cell_distance", "Assume distance to cells during approach / hunt")
        self._add_distance(geometry_layout, "approach_cell_distance_um", "Approach distance to cells", "apply_cell_distance")
        self._add_distance(geometry_layout, "cell_distance_um", "Hunt distance to cells", "apply_cell_distance")
        geometry_note = QtWidgets.QLabel(
            "Distances are in microns. The 1350 micron microscope offset, 75 micron approach "
            "distance, and 20 micron hunt distance are configurable display assumptions. "
            "Recording coordinates are preserved."
        )
        geometry_note.setWordWrap(True)
        geometry_layout.addRow(geometry_note)
        tabs.addTab(geometry, "3D view")

        export = QtWidgets.QWidget()
        export_layout = QtWidgets.QFormLayout(export)
        self._add_checkbox(export_layout, "export_inference", "Include selected inference models in exported videos")
        self._add_combo(export_layout, "export_aspect_ratio", "Video aspect ratio", (("16:9", "16:9"), ("4:3", "4:3")))
        fps = QtWidgets.QSpinBox()
        fps.setRange(1, 120)
        fps.setSuffix(" fps")
        fps.setValue(self.settings["export_fps"])
        self._controls["export_fps"] = fps
        export_layout.addRow("Video frame rate", fps)
        export_note = QtWidgets.QLabel(
            "Exports include a state label when available, a 5 micron scale bar at "
            "6 pixels per micron, the movement reconstruction, the auxiliary camera "
            "when available, and electrophysiology."
        )
        export_note.setWordWrap(True)
        export_layout.addRow(export_note)
        tabs.addTab(export, "Video export")

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _add_checkbox(self, layout, key, text):
        control = QtWidgets.QCheckBox(text)
        control.setChecked(self.settings[key])
        self._controls[key] = control
        layout.addRow(control)

    def _add_combo(self, layout, key, label, choices):
        control = QtWidgets.QComboBox()
        for text, value in choices:
            control.addItem(text, value)
        control.setCurrentIndex(control.findData(self.settings[key]))
        self._controls[key] = control
        layout.addRow(label, control)

    def _add_distance(self, layout, key, label, enabled_key):
        control = QtWidgets.QDoubleSpinBox()
        control.setRange(0.0, 100000.0)
        control.setDecimals(2)
        control.setSuffix(" µm")
        control.setValue(self.settings[key])
        control.setEnabled(self.settings[enabled_key])
        self._controls[enabled_key].toggled.connect(control.setEnabled)
        self._controls[key] = control
        layout.addRow(label, control)

    def get_settings(self):
        """Return a copy so applying preferences cannot mutate the dialog's state."""
        return dict(self.settings)

    def accept(self):
        values = {}
        for key, control in self._controls.items():
            if isinstance(control, QtWidgets.QCheckBox):
                values[key] = control.isChecked()
            elif isinstance(control, QtWidgets.QComboBox):
                values[key] = control.currentData()
            else:
                values[key] = control.value()
        try:
            self.settings = save_settings(values, self._store)
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, "Settings could not be saved", str(exc))
            return
        super().accept()
