import ast
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
import pytest
from PyQt5 import QtWidgets

from patcherbot.interface.camera import CameraInterface
from patcherbot.interface.experimentBookConfig import ExperimentBookConfig
from patcherbot.gui.camera import CameraGui
from patcherbot.gui.experiment_book_tab import ExperimentBookTab
from patcherbot.utils.experiment_book import ExperimentBookLogger


ROOT = Path(__file__).resolve().parent
EASTERN = timezone(timedelta(hours=-4))
SESSION_TIME = datetime(2026, 8, 24, 14, 32, 10, tzinfo=EASTERN)


@pytest.fixture(scope="session")
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app
    app.processEvents()


def _timeline_widget_count(tab):
    return sum(
        tab.timeline_layout.itemAt(index).widget() is not None
        for index in range(tab.timeline_layout.count())
    )


def _all_label_text(tab):
    return "\n".join(label.text() for label in tab.findChildren(QtWidgets.QLabel))


def _new_tab(tmp_path):
    logger = ExperimentBookLogger(
        folder_path=tmp_path / "experiment_book_data",
        session_time=SESSION_TIME,
    )
    config = ExperimentBookConfig(name="Experiment Book")
    return ExperimentBookTab(config=config, logger=logger), logger


def _activate(tab, name="Membrane response study"):
    tab.experiment_name_edit.setText(name)
    tab.strain_culture_edit.setText("HEK293")
    tab.gender_edit.setText("N/A")
    tab.age_edit.setText("3 days")
    tab.save_details()
    assert tab.book_active is True


def test_store_is_lazy_and_writes_trimmed_human_readable_details(tmp_path):
    root = tmp_path / "experiment_book_data"
    store = ExperimentBookLogger(folder_path=root, session_time=SESSION_TIME)
    expected_session_dir = root / "2026_08_24-14_32"

    assert Path(store.session_dir) == expected_session_dir
    assert Path(store.log_path) == expected_session_dir / "experiment_book.log"
    assert store.folder_created is False
    assert not expected_session_dir.exists()

    store.write_details(
        {
            "experiment_name": "  Membrane response study  ",
            "strain_culture": "  HEK293  ",
            "gender": "   ",
            "age": "  3 days ",
        },
        SESSION_TIME,
    )

    assert expected_session_dir.is_dir()
    assert store.folder_created is True
    text = Path(store.log_path).read_text(encoding="utf-8")
    assert text == (
        "[2026-08-24T14:32:10-04:00] DETAILS\n"
        "Experiment name: Membrane response study\n"
        "Strain/Culture: HEK293\n"
        "Gender: N/A\n"
        "Age: 3 days\n\n"
    )


def test_store_appends_updates_multiline_notes_and_original_snapshot_path(tmp_path):
    store = ExperimentBookLogger(
        folder_path=tmp_path / "experiment_book_data",
        session_time=SESSION_TIME,
    )
    store.write_details(
        {
            "experiment_name": "First name",
            "strain_culture": "",
            "gender": "",
            "age": "",
        },
        SESSION_TIME,
    )
    details_update_time = datetime(2026, 8, 24, 14, 33, tzinfo=EASTERN)
    store.write_details(
        {
            "experiment_name": "Updated name",
            "strain_culture": "Primary culture",
            "gender": "female",
            "age": "P14",
        },
        details_update_time,
    )
    note_time = datetime(2026, 8, 24, 14, 34, 2, tzinfo=EASTERN)
    store.write_note("  First line\nsecond line  ", note_time)
    snapshot_time = datetime(2026, 8, 24, 14, 35, 19, tzinfo=EASTERN)
    image_path = Path("experiments/Data/snap_image_data/session/camera_frames/42.webp")
    store.write_snapshot(image_path, "aux", snapshot_time)

    text = Path(store.log_path).read_text(encoding="utf-8")
    assert text.count("] DETAILS") == 2
    assert "Experiment name: First name" in text
    assert "Experiment name: Updated name" in text
    assert "[2026-08-24T14:34:02-04:00] NOTE\nFirst line\nsecond line\n\n" in text
    assert "[2026-08-24T14:35:19-04:00] SNAPSHOT\nCamera: aux\n" in text
    assert (
        "Saved: experiments/Data/snap_image_data/session/camera_frames/42.webp\n"
        in text.replace("\\", "/")
    )


def test_widget_requires_name_then_keeps_editable_details(qapp, tmp_path):
    tab, store = _new_tab(tmp_path)
    initial_cards = _timeline_widget_count(tab)

    tab.experiment_name_edit.setText("   ")
    tab.save_details()

    assert tab.book_active is False
    assert tab.send_button.isEnabled() is False
    assert not Path(store.session_dir).exists()
    assert _timeline_widget_count(tab) == initial_cards
    assert "name" in tab.status_label.text().lower()

    tab.experiment_name_edit.setText("  Membrane response study  ")
    tab.strain_culture_edit.setText("HEK293")
    tab.gender_edit.setText("female")
    tab.age_edit.setText("3 days")
    tab.save_details()

    assert tab.book_active is True
    assert tab.send_button.isEnabled() is True
    assert tab.experiment_name_edit.text().strip() == "Membrane response study"
    assert tab.strain_culture_edit.text() == "HEK293"
    assert tab.gender_edit.text() == "female"
    assert tab.age_edit.text() == "3 days"
    assert tab.config.experiment_name == "Membrane response study"
    assert tab.config.strain_culture == "HEK293"
    assert tab.config.gender == "female"
    assert tab.config.age == "3 days"
    assert _timeline_widget_count(tab) == initial_cards + 1
    assert Path(store.log_path).is_file()

    tab.experiment_name_edit.clear()
    tab.save_details()
    assert tab.book_active is True
    assert tab.send_button.isEnabled() is True
    assert _timeline_widget_count(tab) == initial_cards + 1

    tab.close()


def test_widget_sends_ordered_notes_and_scrolls_to_latest(qapp, tmp_path):
    tab, store = _new_tab(tmp_path)
    tab.resize(360, 320)
    tab.show()
    _activate(tab)
    cards_after_details = _timeline_widget_count(tab)

    tab.notes_edit.setPlainText("  \n  ")
    tab.send_note()
    assert _timeline_widget_count(tab) == cards_after_details
    assert tab.notes_edit.toPlainText() == "  \n  "

    for index in range(10):
        tab.notes_edit.setPlainText(
            "Note {}\nA deliberately long second line that makes the timeline scroll.".format(
                index
            )
        )
        assert tab.config.general_notes.startswith(f"Note {index}")
        tab.send_note()

    qapp.processEvents()
    qapp.processEvents()

    assert tab.notes_edit.toPlainText() == ""
    assert tab.config.general_notes == ""
    assert _timeline_widget_count(tab) == cards_after_details + 10
    text = Path(store.log_path).read_text(encoding="utf-8")
    assert text.count("] NOTE") == 10
    assert text.index("Note 0") < text.index("Note 9")
    scroll_area = tab.findChild(QtWidgets.QScrollArea)
    assert scroll_area is not None
    scroll_bar = scroll_area.verticalScrollBar()
    assert scroll_bar.value() == scroll_bar.maximum()

    tab.close()


def test_widget_ignores_inactive_or_invalid_snapshots_and_logs_one_valid_card(
    qapp, tmp_path
):
    tab, store = _new_tab(tmp_path)
    source_image = tmp_path / "snap_image_data" / "session" / "camera_frames" / "7.webp"
    source_image.parent.mkdir(parents=True)
    source_image.write_bytes(b"existing full-resolution image")
    payload = {
        "frame_number": 7,
        "captured_at": SESSION_TIME,
        "image_path": str(source_image),
        "frame": np.zeros((20, 40, 3), dtype=np.uint8),
        "camera_role": "main",
    }
    initial_cards = _timeline_widget_count(tab)

    tab.handle_snapshot(payload)
    assert _timeline_widget_count(tab) == initial_cards
    assert not Path(store.session_dir).exists()

    _activate(tab)
    cards_after_details = _timeline_widget_count(tab)
    tab.handle_snapshot({"camera_role": "main", "frame": payload["frame"]})
    assert _timeline_widget_count(tab) == cards_after_details
    text_before_snapshot = Path(store.log_path).read_text(encoding="utf-8")

    tab.handle_snapshot(payload)

    assert _timeline_widget_count(tab) == cards_after_details + 1
    text = Path(store.log_path).read_text(encoding="utf-8")
    assert text.startswith(text_before_snapshot)
    assert text.count("] SNAPSHOT") == 1
    assert str(source_image).replace("\\", "/") in text.replace("\\", "/")
    assert "main" in _all_label_text(tab).lower()
    assert source_image.read_bytes() == b"existing full-resolution image"
    assert [path.name for path in Path(store.session_dir).iterdir()] == [
        "experiment_book.log"
    ]

    tab.close()


def test_widget_keeps_snapshot_record_when_preview_is_unavailable(qapp, tmp_path):
    tab, store = _new_tab(tmp_path)
    _activate(tab)
    cards_before = _timeline_widget_count(tab)

    tab.handle_snapshot(
        {
            "frame_number": 9,
            "captured_at": SESSION_TIME,
            "image_path": "experiments/Data/snap_image_data/session/9.webp",
            "frame": object(),
            "camera_role": "aux",
        }
    )

    assert _timeline_widget_count(tab) == cards_before + 1
    assert Path(store.log_path).read_text(encoding="utf-8").count("] SNAPSHOT") == 1
    labels = _all_label_text(tab).lower()
    assert "aux" in labels
    assert "preview unavailable" in labels

    tab.close()


def test_widget_storage_failure_retains_input_and_adds_no_false_entry(
    qapp, tmp_path, monkeypatch
):
    tab, store = _new_tab(tmp_path)
    _activate(tab)
    cards_before = _timeline_widget_count(tab)
    tab.notes_edit.setPlainText("Keep this note")

    def fail_write(*_args, **_kwargs):
        raise OSError("read-only storage")

    monkeypatch.setattr(store, "write_note", fail_write)
    tab.send_note()

    assert tab.notes_edit.toPlainText() == "Keep this note"
    assert _timeline_widget_count(tab) == cards_before
    assert tab.status_label.text()

    monkeypatch.setattr(store, "write_snapshot", fail_write)
    tab.handle_snapshot(
        {
            "frame_number": 10,
            "captured_at": SESSION_TIME,
            "image_path": "experiments/Data/snap_image_data/session/10.webp",
            "frame": np.zeros((10, 10), dtype=np.uint8),
            "camera_role": "main",
        }
    )
    assert _timeline_widget_count(tab) == cards_before

    tab.close()


class _StubSnapRecorder:
    def __init__(self, root):
        self.folder_created = False
        self.camera_folder_path = str(root / "camera_frames")
        self.aux_camera_folder_path = str(root / "aux_camera_frames")
        self.image_type = "webp"
        self.saved = []

    def _save_image(self, frame, path):
        self.saved.append((frame, path))


def _camera_interface(tmp_path, queue):
    camera = type("FakeCamera", (), {"raw_frame_queue": queue})()
    interface = CameraInterface(camera)
    interface.snap_image_recorder = _StubSnapRecorder(tmp_path / "snap_session")
    return interface


def test_camera_interface_returns_copied_snapshot_payload(tmp_path):
    original = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    expected = original.copy()
    interface = _camera_interface(
        tmp_path,
        [(42, SESSION_TIME, "processed frame is ignored", original)],
    )

    payload = interface.snap_image()

    recorder = interface.snap_image_recorder
    assert payload["frame_number"] == 42
    assert payload["captured_at"] is SESSION_TIME
    assert payload["frame"] is recorder.saved[0][0]
    assert payload["frame"] is not original
    assert payload["image_path"] == recorder.saved[0][1]
    assert Path(payload["image_path"]).parent == Path(recorder.camera_folder_path)
    assert Path(payload["image_path"]).name == "42_1787596330.0.webp"
    assert Path(recorder.camera_folder_path).is_dir()
    assert Path(recorder.aux_camera_folder_path).is_dir()
    assert recorder.folder_created is True

    original.fill(0)
    np.testing.assert_array_equal(payload["frame"], expected)


@pytest.mark.parametrize(
    "queue",
    [
        [],
        [None],
        [(1, SESSION_TIME, None)],
        [(None, SESSION_TIME, None, np.zeros((2, 2), dtype=np.uint8))],
        [(1, None, None, np.zeros((2, 2), dtype=np.uint8))],
        [(1, SESSION_TIME, None, None)],
    ],
)
def test_camera_interface_rejects_missing_or_malformed_frames(tmp_path, queue):
    interface = _camera_interface(tmp_path, queue)

    assert interface.snap_image() is None
    assert interface.snap_image_recorder.saved == []
    assert interface.snap_image_recorder.folder_created is False


def test_camera_interface_returns_none_when_snapshot_folder_creation_fails(
    tmp_path, monkeypatch
):
    interface = _camera_interface(
        tmp_path,
        [(1, SESSION_TIME, None, np.zeros((2, 2), dtype=np.uint8))],
    )

    def fail_makedirs(*_args, **_kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr("patcherbot.interface.camera.os.makedirs", fail_makedirs)

    assert interface.snap_image() is None
    assert interface.snap_image_recorder.saved == []
    assert interface.snap_image_recorder.folder_created is False


class _MinimalCameraGui(CameraGui):
    def __init__(self, interface, camera_role):
        QtWidgets.QMainWindow.__init__(self)
        self.active_interface = interface
        self.active_camera_role = camera_role
        self.recording_state_manager = None


class _ReturningInterface:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def snap_image(self):
        self.calls += 1
        return self.payload


def test_camera_gui_emits_one_copied_payload_with_active_role(qapp):
    original_payload = {
        "frame_number": 11,
        "captured_at": SESSION_TIME,
        "image_path": "experiments/Data/snap_image_data/session/11.webp",
        "frame": np.zeros((2, 2), dtype=np.uint8),
    }
    interface = _ReturningInterface(original_payload)
    gui = _MinimalCameraGui(interface, "aux")
    emitted = []
    gui.snapshot_captured.connect(emitted.append)

    gui.snap_active_camera_image()

    assert interface.calls == 1
    assert len(emitted) == 1
    assert emitted[0] is not original_payload
    assert emitted[0]["camera_role"] == "aux"
    assert "camera_role" not in original_payload

    interface.payload = None
    gui.snap_active_camera_image()
    assert interface.calls == 2
    assert len(emitted) == 1

    gui.close()


def test_experiment_book_config_has_expected_structure():
    config = ExperimentBookConfig(name="Experiment Book")

    assert config.to_dict() == {
        "experiment_name": "",
        "strain_culture": "",
        "gender": "",
        "age": "",
        "general_notes": "",
    }
    assert config.categories == [
        (
            "Experiment Details",
            ["experiment_name", "strain_culture", "gender", "age"],
        ),
        ("Notes", ["general_notes"]),
    ]


def test_patch_gui_places_experiment_book_immediately_after_protocols_and_connects_snapshots():
    module = ast.parse((ROOT / "patcherbot/gui/patch.py").read_text(encoding="utf-8-sig"))
    patch_gui = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "PatchGui"
    )
    init = next(
        node
        for node in patch_gui.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    tab_additions = sorted(
        (
            call
            for call in ast.walk(init)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr in {"add_config_gui", "add_tab"}
        ),
        key=lambda call: (call.lineno, call.col_offset),
    )

    protocol_index = next(
        index
        for index, call in enumerate(tab_additions)
        if call.func.attr == "add_config_gui"
        and call.args
        and isinstance(call.args[0], ast.Attribute)
        and call.args[0].attr == "protocol_config"
    )
    experiment_index = next(
        index
        for index, call in enumerate(tab_additions)
        if call.func.attr == "add_config_gui"
        and call.args
        and isinstance(call.args[0], ast.Attribute)
        and call.args[0].attr == "experiment_book_config"
    )
    assert experiment_index == protocol_index + 1
    experiment_call = tab_additions[experiment_index]
    gui_class = next(
        keyword.value
        for keyword in experiment_call.keywords
        if keyword.arg == "gui_class"
    )
    assert isinstance(gui_class, ast.Name)
    assert gui_class.id == "ExperimentBookTab"

    connect_calls = [
        call
        for call in ast.walk(init)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "connect"
    ]
    assert any(
        isinstance(call.func.value, ast.Attribute)
        and call.func.value.attr == "snapshot_captured"
        and call.args
        and isinstance(call.args[0], ast.Attribute)
        and call.args[0].attr == "handle_snapshot"
        for call in connect_calls
    )

    interface_source = (
        ROOT / "patcherbot/interface/patch.py"
    ).read_text(encoding="utf-8-sig")
    assert "self.experiment_book_config = ExperimentBookConfig(" in interface_source
