"""Experiment metadata and timeline widgets for the patching GUI."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import logging

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from patcherbot.utils.experiment_book import ExperimentBookLogger
from patcherbot.gui.ParamConfig import ParamConfig

class ExperimentBookSession(QtCore.QObject):
    """Shared state and behavior for one Experiment Book session."""

    event_added = QtCore.pyqtSignal(object)
    status_changed = QtCore.pyqtSignal(str, bool)
    active_changed = QtCore.pyqtSignal(bool)
    
class ExperimentBookTab(ParamConfig):
    """Experiment detail form with an append-only chat-style timeline."""

    THUMBNAIL_SIZE = QtCore.QSize(160, 120)

    def __init__(
        self,
        config,
        logger=None,
        storage_root=None,
        session_time=None,
        parent=None,
    ):
        super().__init__(config, parent=parent, build_ui=False)
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
        self.recording_state_manager = None
        self.state_tally_card = None
        self.state_tally_header = None
        self.state_tally_body = None

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

        category, names = next(
            (category, names) for category, names in self.config.categories
            if category == "Experiment Details"
        )
        details_group = self._create_config_group(category, names)
        details_layout = details_group.layout()
        self.detail_edits = {name: self.value_widgets[name] for name in names}
        for name, widget in self.detail_edits.items():
            setattr(self, f"{name}_edit", widget)

        self.save_details_button = QtWidgets.QPushButton("Save Details")
        self.save_details_button.clicked.connect(self.save_details)
        details_layout.addWidget(self.save_details_button)

        self.status_label = QtWidgets.QLabel()
        self.status_label.setWordWrap(True)
        details_layout.addWidget(self.status_label)
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
        self.notes_edit = self._create_value_widget("general_notes", multiline=True)
        self.notes_edit.setPlaceholderText("Type a note for this experiment...")
        self.notes_edit.setMaximumHeight(100)
        notes_layout.addWidget(self.notes_edit)
        self.send_button = QtWidgets.QPushButton("Send")
        self.send_button.setEnabled(False)
        self.send_button.clicked.connect(self.send_note)
        notes_layout.addWidget(self.send_button, alignment=QtCore.Qt.AlignRight)
        layout.addWidget(notes_group)

    def attach_recording_state_manager(self, recording_state_manager):
        """Attach the shared manager that owns the experiment's press counts."""
        self.recording_state_manager = recording_state_manager

    def save_details(self):
        was_active = self.book_active
        details = {
            name: str(getattr(self.config, name)).strip()
            for name in self.detail_edits
        }
        if not details["experiment_name"]:
            self._set_status("Experiment name is required.", error=True)
            return False
        for name, value in details.items():
            if getattr(self.config, name) != value:
                setattr(self.config, name, value)

        timestamp = datetime.now().astimezone()
        try:
            normalized = self.logger.write_details(details, timestamp)
        except OSError:
            logging.getLogger(__name__).exception("Unable to save experiment details")
            self._set_status("Experiment details could not be saved.", error=True)
            return False

        self.active_details = normalized
        self.book_active = True
        self.send_button.setEnabled(True)
        detail_text = "\n".join([
            f"Experiment name: {normalized['experiment_name']}",
            f"Strain/Culture: {normalized['strain_culture']}",
            f"Gender: {normalized['gender']}",
            f"Age: {normalized['age']}",
        ])
        self._add_text_card("Details", detail_text, timestamp)
        tally_initialized = True
        if not was_active:
            tally_initialized = self._initialize_state_press_tally(timestamp)
        if tally_initialized:
            self._set_status("Experiment details saved.")
        return True

    def _initialize_state_press_tally(self, timestamp):
        if self.recording_state_manager is None:
            return True
        try:
            snapshot = self.recording_state_manager.reset_state_press_counts()
        except Exception:
            logging.getLogger(__name__).exception("Unable to reset state press tally")
            self._set_status("The state tally could not be initialized.", error=True)
            return False
        return self._record_state_press_tally(snapshot, timestamp)

    @QtCore.pyqtSlot(object)
    def handle_state_press_tally(self, snapshot):
        """Persist a manager snapshot and refresh the one live tally card."""
        return self._record_state_press_tally(
            snapshot,
            datetime.now().astimezone(),
        )

    def _record_state_press_tally(self, snapshot, timestamp):
        if not self.book_active or self.recording_state_manager is None:
            return False
        state_labels = getattr(
            self.recording_state_manager,
            "STATE_PRESS_LABELS",
            None,
        )
        if not isinstance(snapshot, Mapping) or not isinstance(state_labels, Mapping):
            return False
        try:
            normalized = self.logger.write_state_tally(
                snapshot,
                state_labels,
                timestamp,
            )
        except (TypeError, ValueError):
            return False
        except OSError:
            logging.getLogger(__name__).exception("Unable to save state press tally")
            self._set_status("The state tally could not be saved.", error=True)
            return False

        self._update_state_tally_card(normalized, state_labels, timestamp)
        self._set_status("State tally updated.")
        return True

    def _update_state_tally_card(self, counts, state_labels, timestamp):
        tally_text = "\n".join(
            f"{state_labels[state_name]}: {counts[state_name]}"
            for state_name in state_labels
        )
        if self.state_tally_card is None:
            card, card_layout = self._new_card("State Tally", timestamp)
            body = QtWidgets.QLabel(tally_text)
            body.setTextFormat(QtCore.Qt.PlainText)
            body.setWordWrap(True)
            body.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            card_layout.addWidget(body)
            self.state_tally_card = card
            self.state_tally_header = card_layout.itemAt(0).widget()
            self.state_tally_body = body
            self._append_card(card)
            return

        self.state_tally_header.setText(
            f"State Tally  |  {timestamp.strftime('%H:%M:%S')}"
        )
        self.state_tally_body.setText(tally_text)

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

    def add_origin_entry(self, record, frame):
        """Append one already-persisted origin card without further disk writes."""
        timestamp = datetime.fromisoformat(record["saved_at"])
        axis = record["axis"].upper()
        card, card_layout = self._new_card(f"Origin {axis}", timestamp)
        xyz = record["stage_xyz_um"]
        origins = record["origins_um"]
        origin_text = ", ".join(
            f"{name.upper()}: {origins[name]:.2f} um"
            if origins.get(name) is not None else f"{name.upper()}: not saved"
            for name in ("x", "y")
        )
        body = QtWidgets.QLabel(
            f"Saved {axis} origin\n"
            f"Stage X: {xyz[0]:.2f}, Y: {xyz[1]:.2f}, Z: {xyz[2]:.2f} um\n"
            f"Origins: {origin_text}\n"
            f"Image: {record['image_path']}"
        )
        body.setTextFormat(QtCore.Qt.PlainText)
        body.setWordWrap(True)
        body.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        card_layout.addWidget(body)
        preview = QtWidgets.QLabel()
        preview.setObjectName("origin_thumbnail")
        preview.setAlignment(QtCore.Qt.AlignCenter)
        try:
            preview.setPixmap(self._frame_to_pixmap(frame).scaled(
                self.THUMBNAIL_SIZE,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            ))
        except Exception:
            logging.getLogger(__name__).warning(
                "Origin snapshot saved but preview unavailable", exc_info=True
            )
            preview.setText("Preview unavailable")
        card_layout.addWidget(preview)
        self._append_card(card)

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

