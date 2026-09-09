"""Experiment metadata and timeline widgets for the patching GUI."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import logging

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from patcherbot.utils.experiment_book import ExperimentBookLogger

class ExperimentBookSession(QtCore.QObject):
    """Shared state and behavior for one Experiment Book session."""

    event_added = QtCore.pyqtSignal(object)
    status_changed = QtCore.pyqtSignal(str, bool)
    active_changed = QtCore.pyqtSignal(bool)

    def __init__(
        self,
        config,
        logger=None,
        storage_root=None,
        session_time=None,
        parent=None,
    ):
        super().__init__(parent)

        self.config = config
        self.logger = logger if logger is not None else ExperimentBookLogger(
            folder_path=storage_root,
            session_time=session_time,
        )

        self.book_active = False
        self.active_details = None
        self.timeline_events = []

    def save_details(self):
        details = {
            "experiment_name": str(self.config.experiment_name).strip(),
            "strain_culture": str(self.config.strain_culture).strip(),
            "gender": str(self.config.gender).strip(),
            "age": str(self.config.age).strip(),
        }

        if not details["experiment_name"]:
            self.status_changed.emit(
                "Experiment name is required.",
                True,
            )
            return False

        timestamp = datetime.now().astimezone()

        try:
            normalized = self.logger.write_details(
                details,
                timestamp,
            )
        except OSError:
            logging.getLogger(__name__).exception(
                "Unable to save experiment details"
            )
            self.status_changed.emit(
                "Experiment details could not be saved.",
                True,
            )
            return False

        self.active_details = normalized

        if not self.book_active:
            self.book_active = True
            self.active_changed.emit(True)

        detail_text = "\n".join([
            f"Experiment name: {normalized['experiment_name']}",
            f"Strain/Culture: {normalized['strain_culture']}",
            f"Gender: {normalized['gender']}",
            f"Age: {normalized['age']}",
        ])

        self._add_event({
            "type": "text",
            "entry_type": "Details",
            "text": detail_text,
            "timestamp": timestamp,
        })

        self.status_changed.emit(
            "Experiment details saved.",
            False,
        )

        return True

    def send_note(self):
        text = self.config.general_notes.strip()

        if not self.book_active:
            self.status_changed.emit(
                "Save experiment details before adding notes.",
                True,
            )
            return False

        if not text:
            self.status_changed.emit(
                "Enter a note before sending.",
                True,
            )
            return False

        timestamp = datetime.now().astimezone()

        try:
            cleaned = self.logger.write_note(
                text,
                timestamp,
            )
        except OSError:
            logging.getLogger(__name__).exception(
                "Unable to save experiment note"
            )
            self.status_changed.emit(
                "The note could not be saved.",
                True,
            )
            return False

        self._add_event({
            "type": "text",
            "entry_type": "Note",
            "text": cleaned,
            "timestamp": timestamp,
        })

        # Shared config update automatically clears every view.
        self.config.general_notes = ""

        self.status_changed.emit(
            "Note saved.",
            False,
        )

        return True

    @QtCore.pyqtSlot(object)
    def handle_snapshot(self, payload):
        if not self.book_active:
            return False

        if not isinstance(payload, Mapping):
            return False

        image_path = payload.get("image_path")
        camera_role = payload.get("camera_role")
        frame = payload.get("frame")
        frame_number = payload.get("frame_number")
        captured_at = payload.get("captured_at")
        camera_id = payload.get("camera_id")

        if (
            not image_path
            or camera_role not in ("main", "pipette")
            or frame is None
            or frame_number is None
            or not isinstance(captured_at, datetime)
        ):
            return False

        try:
            self.logger.write_snapshot(
                image_path,
                camera_role,
                captured_at,
            )
            
            display_camera = (
                camera_id
                if camera_id is not None
                else camera_role
            )

            self._add_snapshot_card(
                frame,
                display_camera,
                captured_at,
            )
        except OSError:
            logging.getLogger(__name__).exception(
                "Unable to log experiment snapshot"
            )
            self.status_changed.emit(
                "The snapshot could not be added to the experiment log.",
                True,
            )
            return False

        self._add_event({
            "type": "snapshot",
            "frame": frame,
            "image_path": image_path,
            "camera_role": camera_role,
            "frame_number": frame_number,
            "timestamp": captured_at,
        })

        self.status_changed.emit(
            "Snapshot added to the experiment timeline.",
            False,
        )

        return True

    def _add_event(self, event):
        self.timeline_events.append(event)
        self.event_added.emit(event)

class ExperimentBookTab(QtWidgets.QWidget):
    """Experiment detail form with an append-only chat-style timeline."""

    THUMBNAIL_SIZE = QtCore.QSize(160, 120)
    config_value_changed_signal = QtCore.pyqtSignal(str, object)

    def __init__(
        self,
        session,
        parent=None,
    ):
        super().__init__(parent=parent)

        self.session = session
        self.config = session.config

        # Widgets themselves still belong to this individual view.
        self.timeline_cards = []

        self._build_ui()

        self.config_value_changed_signal.connect(
            self._display_config_value
        )

        for param_name in (
            "experiment_name",
            "strain_culture",
            "gender",
            "age",
            "general_notes",
        ):
            self.config.param.watch(
                self._config_param_changed,
                param_name,
            )

        self.session.event_added.connect(
            self._display_event
        )
        self.session.status_changed.connect(
            self._set_status
        )
        self.session.active_changed.connect(
            self.send_button.setEnabled
        )

        self.send_button.setEnabled(
            self.session.book_active
        )

        # Allows a newly-created view to catch up to the current book.
        for event in self.session.timeline_events:
            self._display_event(event)

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        details_group = QtWidgets.QGroupBox("Experiment Details")
        details_layout = QtWidgets.QFormLayout(details_group)
        self.experiment_name_edit = QtWidgets.QLineEdit(self.config.experiment_name)
        self.strain_culture_edit = QtWidgets.QLineEdit(self.config.strain_culture)
        self.gender_edit = QtWidgets.QLineEdit(self.config.gender)
        self.age_edit = QtWidgets.QLineEdit(self.config.age)
        self.detail_edits = {
            "experiment_name": self.experiment_name_edit,
            "strain_culture": self.strain_culture_edit,
            "gender": self.gender_edit,
            "age": self.age_edit,
        }
        for name, edit in self.detail_edits.items():
            edit.setObjectName(name)
            edit.textChanged.connect(
                lambda value, config_name=name: self._set_config_value(
                    config_name,
                    value,
                )
            )
        details_layout.addRow("Experiment Name:", self.experiment_name_edit)
        details_layout.addRow("Strain/Culture:", self.strain_culture_edit)
        details_layout.addRow("Gender:", self.gender_edit)
        details_layout.addRow("Age:", self.age_edit)

        self.save_details_button = QtWidgets.QPushButton("Save Details")
        self.save_details_button.clicked.connect(self.save_details)
        details_layout.addRow(self.save_details_button)

        self.status_label = QtWidgets.QLabel()
        self.status_label.setWordWrap(True)
        details_layout.addRow(self.status_label)
        layout.addWidget(details_group)

        timeline_label = QtWidgets.QLabel("Experiment Timeline")
        timeline_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(timeline_label)

        self.timeline_scroll = QtWidgets.QScrollArea()
        self.timeline_scroll.setWidgetResizable(True)
        self.timeline_scroll.setMinimumHeight(220)
        self.timeline_widget = QtWidgets.QWidget()
        self.timeline_layout = QtWidgets.QVBoxLayout(self.timeline_widget)
        self.timeline_layout.setAlignment(QtCore.Qt.AlignTop)
        self.timeline_layout.setContentsMargins(6, 6, 6, 6)
        self.timeline_layout.setSpacing(8)
        self.timeline_scroll.setWidget(self.timeline_widget)
        scrollbar = self.timeline_scroll.verticalScrollBar()
        scrollbar.rangeChanged.connect(
            lambda _minimum, maximum: scrollbar.setValue(maximum)
        )
        layout.addWidget(self.timeline_scroll, 1)

        notes_group = QtWidgets.QGroupBox("General Notes")
        notes_layout = QtWidgets.QVBoxLayout(notes_group)
        self.notes_edit = QtWidgets.QPlainTextEdit()
        self.notes_edit.setPlaceholderText("Type a note for this experiment...")
        self.notes_edit.setMaximumHeight(100)
        self.notes_edit.setObjectName("general_notes")
        self.notes_edit.setPlainText(self.config.general_notes)
        self.notes_edit.textChanged.connect(self._notes_changed)
        notes_layout.addWidget(self.notes_edit)
        self.send_button = QtWidgets.QPushButton("Send")
        self.send_button.setEnabled(False)
        self.send_button.clicked.connect(self.send_note)
        notes_layout.addWidget(self.send_button, alignment=QtCore.Qt.AlignRight)
        layout.addWidget(notes_group)

    def save_details(self):
        return self.session.save_details()

    def send_note(self):
        return self.session.send_note()

    def _set_config_value(self, name, value):
        if getattr(self.config, name) != value:
            setattr(self.config, name, value)

    def _notes_changed(self):
        self._set_config_value("general_notes", self.notes_edit.toPlainText())

    def _config_param_changed(self, event):
        self.config_value_changed_signal.emit(
            event.name,
            event.new,
        )

    @QtCore.pyqtSlot(str, object)
    def _display_config_value(self, name, value):
        if name == "general_notes":
            widget = self.notes_edit
            new_value = str(value)
            if widget.toPlainText() == new_value:
                return
            widget.blockSignals(True)
            widget.setPlainText(new_value)
            widget.blockSignals(False)
            return
        widget = self.detail_edits.get(name)
        if widget is None or widget.text() == str(value):
            return
        widget.blockSignals(True)
        widget.setText(str(value))
        widget.blockSignals(False)

    def handle_snapshot(self, payload):
        if not self.book_active:
            return False
        if not isinstance(payload, Mapping):
            return False

        image_path = payload.get("image_path")
        camera_role = payload.get("camera_role")
        frame = payload.get("frame")
        frame_number = payload.get("frame_number")
        captured_at = payload.get("captured_at")
        if (
            not image_path
            or camera_role not in ("main", "aux")
            or frame is None
            or frame_number is None
            or not isinstance(captured_at, datetime)
        ):
            return False

        try:
            self.logger.write_snapshot(image_path, camera_role, captured_at)
        except OSError:
            logging.getLogger(__name__).exception("Unable to log experiment snapshot")
            self._set_status("The snapshot could not be added to the experiment log.", error=True)
            return False

        self._add_snapshot_card(frame, camera_role, captured_at)
        self._set_status("Snapshot added to the experiment timeline.")
        return True

    def _add_text_card(self, entry_type, text, timestamp):
        card, card_layout = self._new_card(entry_type, timestamp)
        body = QtWidgets.QLabel(text)
        body.setTextFormat(QtCore.Qt.PlainText)
        body.setWordWrap(True)
        body.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        card_layout.addWidget(body)
        self._append_card(card)

    def _add_snapshot_card(self, frame, camera_role, timestamp):
        card, card_layout = self._new_card(f"Snapshot - {camera_role}", timestamp)
        preview = QtWidgets.QLabel()
        preview.setObjectName("snapshot_thumbnail")
        preview.setAlignment(QtCore.Qt.AlignCenter)
        try:
            pixmap = self._frame_to_pixmap(frame)
            preview.setPixmap(pixmap.scaled(
                self.THUMBNAIL_SIZE,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            ))
        except Exception:
            logging.getLogger(__name__).warning(
                "Experiment snapshot was logged but its preview could not be rendered",
                exc_info=True,
            )
            preview.setText("Preview unavailable")
        card_layout.addWidget(preview)
        self._append_card(card)

    def _new_card(self, entry_type, timestamp):
        card = QtWidgets.QFrame()
        card.setProperty("entry_type", entry_type.lower())
        card.setFrameShape(QtWidgets.QFrame.StyledPanel)
        card.setStyleSheet(
            "QFrame { background: #f4f4f4; border: 1px solid #d4d4d4; "
            "border-radius: 7px; } QLabel { border: none; background: transparent; }"
        )
        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.setContentsMargins(9, 7, 9, 7)
        header = QtWidgets.QLabel(
            f"{entry_type}  |  {timestamp.strftime('%H:%M:%S')}"
        )
        header.setStyleSheet("font-size: 11px; color: #666666;")
        card_layout.addWidget(header)
        return card, card_layout

    def _append_card(self, card):
        self.timeline_layout.addWidget(card)
        self.timeline_cards.append(card)
        QtCore.QTimer.singleShot(0, self._scroll_to_latest)

    def _scroll_to_latest(self):
        scrollbar = self.timeline_scroll.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _set_status(self, message, error=False):
        self.status_label.setText(message)
        color = "#b00020" if error else "#2e7d32"
        self.status_label.setStyleSheet(f"color: {color};")

    def _config_param_changed(self, event):
        self.config_value_changed_signal.emit(
            event.name,
            event.new,
        )

    @QtCore.pyqtSlot(object)
    def _display_event(self, event):
        event_type = event.get("type")

        if event_type == "text":
            self._add_text_card(
                event["entry_type"],
                event["text"],
                event["timestamp"],
            )

        elif event_type == "snapshot":
            self._add_snapshot_card(
                event["frame"],
                event["camera_role"],
                event["timestamp"],
            )

    @classmethod
    def _frame_to_pixmap(cls, frame):
        array = np.asarray(frame)
        if array.ndim == 3 and array.shape[2] == 1:
            array = array[:, :, 0]
        if array.ndim not in (2, 3) or array.size == 0:
            raise ValueError("Unsupported snapshot frame shape")

        array = cls._normalize_to_uint8(array)
        if array.ndim == 2:
            array = np.ascontiguousarray(array)
            height, width = array.shape
            image_format = QtGui.QImage.Format_Grayscale8
        else:
            if array.shape[2] not in (3, 4):
                raise ValueError("Unsupported snapshot channel count")
            array = np.ascontiguousarray(array)
            height, width, channels = array.shape
            image_format = (
                QtGui.QImage.Format_RGB888
                if channels == 3
                else QtGui.QImage.Format_RGBA8888
            )

        image = QtGui.QImage(
            array.data,
            width,
            height,
            array.strides[0],
            image_format,
        ).copy()
        if array.ndim == 3:
            image = image.rgbSwapped()
        if image.isNull():
            raise ValueError("Unable to create snapshot preview")
        return QtGui.QPixmap.fromImage(image)

    @staticmethod
    def _normalize_to_uint8(array):
        if array.dtype == np.uint8:
            return array
        numeric = array.astype(np.float32)
        finite = np.isfinite(numeric)
        if not finite.any():
            return np.zeros(array.shape, dtype=np.uint8)
        minimum = float(numeric[finite].min())
        maximum = float(numeric[finite].max())
        normalized = np.zeros(numeric.shape, dtype=np.float32)
        if maximum > minimum:
            normalized[finite] = (
                (numeric[finite] - minimum) / (maximum - minimum) * 255.0
            )
        return normalized.astype(np.uint8)

