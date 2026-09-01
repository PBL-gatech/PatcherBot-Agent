import ast
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest
from PyQt5 import QtWidgets

from patcherbot.interface.camera import CameraInterface
from patcherbot.gui.experiment_book_tab import ExperimentBookTab
from patcherbot.interface.experimentBookConfig import ExperimentBookConfig
from patcherbot.utils.RecordingStateManager import RecordingStateManager
from patcherbot.utils.experiment_book import ExperimentBookLogger


ROOT = Path(__file__).resolve().parent
EXPECTED_STATE_PRESS_LABELS = {
    "locate_cell": "Locate Cell",
    "hunt_cell": "Hunt Cell",
    "gigaseal": "Gigaseal",
    "break_in": "Break-in",
    "escape": "Escape Cell",
    "patch": "Patch Cell",
}
ZERO_TALLY = {state: 0 for state in EXPECTED_STATE_PRESS_LABELS}


@pytest.fixture(scope="session")
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app
    app.processEvents()


@pytest.fixture
def tally_tab(qapp, tmp_path):
    manager = RecordingStateManager()
    logger = ExperimentBookLogger(folder_path=tmp_path / "experiment_book_data")
    config = ExperimentBookConfig(name="Experiment Book")
    tab = ExperimentBookTab(config=config, logger=logger)
    tab.attach_recording_state_manager(manager)
    yield tab, manager, logger
    tab.close()


def _activate(tab, name="State press study"):
    tab.experiment_name_edit.setText(name)
    assert tab.save_details() is True
    assert tab.book_active is True


def _state_tally_cards(tab):
    return [
        card
        for card in tab.timeline_cards
        if card.property("entry_type") == "state tally"
    ]


def _expected_tally_text(snapshot):
    return "\n".join(
        f"{label}: {snapshot[state]}"
        for state, label in EXPECTED_STATE_PRESS_LABELS.items()
    )


def _last_tally_body(log_text):
    return log_text.rsplit("] STATE TALLY\n", 1)[1].split("\n\n", 1)[0]


def test_recording_state_manager_tally_contract_and_preserves_existing_state():
    manager = RecordingStateManager()

    assert manager.STATE_PRESS_LABELS == EXPECTED_STATE_PRESS_LABELS
    assert manager.get_state_press_counts() == ZERO_TALLY
    assert list(manager.get_state_press_counts()) == list(EXPECTED_STATE_PRESS_LABELS)

    increment_snapshot = manager.increment_state_press("hunt_cell")
    assert increment_snapshot == {**ZERO_TALLY, "hunt_cell": 1}
    increment_snapshot["hunt_cell"] = 99
    assert manager.get_state_press_counts()["hunt_cell"] == 1

    read_snapshot = manager.get_state_press_counts()
    read_snapshot["locate_cell"] = 99
    assert manager.get_state_press_counts()["locate_cell"] == 0

    before_invalid = manager.get_state_press_counts()
    with pytest.raises(ValueError):
        manager.increment_state_press("not_a_patch_state")
    assert manager.get_state_press_counts() == before_invalid

    manager.sample_number = 17
    manager.testMode = True
    manager.set_recording(True)
    reset_snapshot = manager.reset_state_press_counts()
    assert reset_snapshot == ZERO_TALLY
    reset_snapshot["patch"] = 99
    assert manager.get_state_press_counts() == ZERO_TALLY
    assert manager.sample_number == 17
    assert manager.testMode is True
    assert manager.is_recording_enabled() is True


def test_recording_state_manager_tally_increments_are_thread_safe():
    manager = RecordingStateManager()
    repetitions = 200

    def increment_many(state):
        for _ in range(repetitions):
            manager.increment_state_press(state)

    with ThreadPoolExecutor(max_workers=len(EXPECTED_STATE_PRESS_LABELS)) as executor:
        list(executor.map(increment_many, EXPECTED_STATE_PRESS_LABELS))

    assert manager.get_state_press_counts() == {
        state: repetitions for state in EXPECTED_STATE_PRESS_LABELS
    }


def test_experiment_book_ignores_inactive_tally_then_resets_on_first_activation(
    tally_tab,
):
    tab, manager, logger = tally_tab
    pre_activation = manager.increment_state_press("locate_cell")

    assert tab.handle_state_press_tally(pre_activation) is False
    assert _state_tally_cards(tab) == []
    assert not Path(logger.session_dir).exists()

    _activate(tab)

    assert manager.get_state_press_counts() == ZERO_TALLY
    assert len(_state_tally_cards(tab)) == 1
    assert tab.state_tally_card is _state_tally_cards(tab)[0]
    assert tab.state_tally_body.text() == _expected_tally_text(ZERO_TALLY)
    log_text = Path(logger.log_path).read_text(encoding="utf-8")
    assert log_text.count("] DETAILS") == 1
    assert log_text.count("] STATE TALLY") == 1
    assert log_text.index("] DETAILS") < log_text.index("] STATE TALLY")
    assert _last_tally_body(log_text) == _expected_tally_text(ZERO_TALLY)


def test_experiment_book_updates_one_live_card_appends_each_tally_and_preserves_on_resave(
    tally_tab,
):
    tab, manager, logger = tally_tab
    _activate(tab)
    original_card = tab.state_tally_card

    snapshots = [
        manager.increment_state_press("locate_cell"),
        manager.increment_state_press("patch"),
        manager.increment_state_press("patch"),
    ]
    for snapshot in snapshots:
        assert tab.handle_state_press_tally(snapshot) is True

    expected = {
        **ZERO_TALLY,
        "locate_cell": 1,
        "patch": 2,
    }
    assert manager.get_state_press_counts() == expected
    assert tab.state_tally_card is original_card
    assert _state_tally_cards(tab) == [original_card]
    assert tab.state_tally_body.text() == _expected_tally_text(expected)

    log_text = Path(logger.log_path).read_text(encoding="utf-8")
    assert log_text.count("] STATE TALLY") == 1 + len(snapshots)
    assert _last_tally_body(log_text) == _expected_tally_text(expected)

    tab.age_edit.setText("P14")
    assert tab.save_details() is True
    assert manager.get_state_press_counts() == expected
    assert tab.state_tally_card is original_card
    assert _state_tally_cards(tab) == [original_card]
    updated_log_text = Path(logger.log_path).read_text(encoding="utf-8")
    assert updated_log_text.count("] DETAILS") == 2
    assert updated_log_text.count("] STATE TALLY") == 1 + len(snapshots)


def test_experiment_book_tally_write_failure_is_isolated(
    tally_tab,
    monkeypatch,
):
    tab, manager, logger = tally_tab
    _activate(tab)
    original_card = tab.state_tally_card
    original_body = tab.state_tally_body.text()
    original_log_text = Path(logger.log_path).read_text(encoding="utf-8")

    def fail_write(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(logger, "write_state_tally", fail_write)
    snapshot = manager.increment_state_press("gigaseal")

    assert tab.handle_state_press_tally(snapshot) is False
    assert tab.book_active is True
    assert manager.get_state_press_counts()["gigaseal"] == 1
    assert tab.state_tally_card is original_card
    assert _state_tally_cards(tab) == [original_card]
    assert tab.state_tally_body.text() == original_body
    assert "could not" in tab.status_label.text().lower()
    assert Path(logger.log_path).read_text(encoding="utf-8") == original_log_text


def test_six_interface_commands_record_canonical_states_and_patch_gui_wires_tally():
    interface_module = ast.parse(
        (ROOT / "patcherbot/interface/patch.py").read_text(encoding="utf-8-sig")
    )
    interface_class = next(
        node
        for node in interface_module.body
        if isinstance(node, ast.ClassDef) and node.name == "AutoPatchInterface"
    )
    methods = {
        node.name: node
        for node in interface_class.body
        if isinstance(node, ast.FunctionDef)
    }
    expected_method_states = {
        "locate_cell": "locate_cell",
        "hunt_cell": "hunt_cell",
        "gigaseal": "gigaseal",
        "break_in": "break_in",
        "escape_cell": "escape",
        "patch": "patch",
    }

    assert any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "state_press_tally_changed"
            for target in node.targets
        )
        for node in interface_class.body
    )
    for method_name, canonical_state in expected_method_states.items():
        first_statement = methods[method_name].body[0]
        assert isinstance(first_statement, ast.Expr)
        assert isinstance(first_statement.value, ast.Call)
        call = first_statement.value
        assert isinstance(call.func, ast.Attribute)
        assert call.func.attr == "_record_state_press"
        assert len(call.args) == 1
        assert isinstance(call.args[0], ast.Constant)
        assert call.args[0].value == canonical_state

    gui_module = ast.parse(
        (ROOT / "patcherbot/gui/patch.py").read_text(encoding="utf-8-sig")
    )
    patch_gui_class = next(
        node
        for node in gui_module.body
        if isinstance(node, ast.ClassDef) and node.name == "PatchGui"
    )
    init = next(
        node
        for node in patch_gui_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    calls = [node for node in ast.walk(init) if isinstance(node, ast.Call)]
    assert any(
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "attach_recording_state_manager"
        for call in calls
    )
    assert any(
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "connect"
        and isinstance(call.func.value, ast.Attribute)
        and call.func.value.attr == "state_press_tally_changed"
        and call.args
        and isinstance(call.args[0], ast.Attribute)
        and call.args[0].attr == "handle_state_press_tally"
        for call in calls
    )
