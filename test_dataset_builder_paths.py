import importlib
from pathlib import Path

import pytest

from experiments.SimpleDatasetBuilder import ObservationSelector, SimpleDatasetBuilder

builder_module = importlib.import_module("experiments.SimpleDatasetBuilder")


def _builder(*, prefer_cv_movement=False):
    builder = object.__new__(SimpleDatasetBuilder)
    builder.prefer_cv_movement = prefer_cv_movement
    builder.observation_selector = ObservationSelector(
        include_pressure=False,
        include_resistance=False,
        include_current=False,
        include_voltage=False,
    )
    builder.freq_mask = 1
    return builder


def _write_movement(path: Path, value: int) -> None:
    path.write_text(f"timestamp;value\n1.0;{value}\n", encoding="utf-8")


def test_load_experiment_data_uses_explicit_folder_and_preference(tmp_path):
    demo = tmp_path / "2025_07_31-12_29"
    demo.mkdir()
    _write_movement(demo / "movement_recording.csv", 10)
    _write_movement(demo / "cv_movement_recording.csv", 20)

    builder = _builder(prefer_cv_movement=True)
    _, movement = builder.load_experiment_data(str(demo))

    assert movement[0, 1] == 20
    assert builder._using_cv_movement_file is True


def test_load_experiment_data_resolves_legacy_name_from_repository_root(
    tmp_path, monkeypatch
):
    demo = tmp_path / "2025_07_31-12_29"
    demo.mkdir()
    _write_movement(demo / "movement_recording.csv", 10)
    monkeypatch.setattr(builder_module, "RIG_RECORDER_DATA_ROOT", tmp_path)

    _, movement = _builder().load_experiment_data(demo.name)

    assert movement[0, 1] == 10


def test_missing_movement_error_reports_explicit_path_and_candidates(tmp_path):
    demo = tmp_path / "missing-demo"

    with pytest.raises(FileNotFoundError) as exc_info:
        _builder().load_experiment_data(str(demo))

    message = str(exc_info.value)
    assert str(demo) in message
    assert "movement_recording.csv" in message
    assert "cv_movement_recording.csv" in message


def test_legacy_epoch_timestamp_normalizes_milliseconds_only():
    assert SimpleDatasetBuilder._normalize_legacy_epoch_timestamp(
        1753981158544
    ) == pytest.approx(1753981158.544)
    assert SimpleDatasetBuilder._normalize_legacy_epoch_timestamp(
        1759339846.007
    ) == pytest.approx(1759339846.007)


def test_state_recorder_root_follows_explicit_rig_data_folder(tmp_path):
    demo = tmp_path / "Data" / "rig_recorder_data" / "2025_07_31-12_29"

    state_root = SimpleDatasetBuilder._resolve_state_recorder_root(str(demo))

    assert state_root == tmp_path / "Data" / "state_recorder_data"


def test_state_recorder_root_keeps_repository_default_for_legacy_name():
    state_root = SimpleDatasetBuilder._resolve_state_recorder_root(
        "2025_07_31-12_29"
    )

    assert state_root == builder_module.DATA_ROOT / "state_recorder_data"


def test_camera_matching_uses_latest_causal_frame_within_100_ms():
    timestamps = builder_module.np.asarray([9.90, 9.95, 10.01])

    index = SimpleDatasetBuilder._latest_causal_camera_frame_index(
        timestamps,
        10.0,
    )

    assert index == 1


def test_camera_matching_rejects_causal_frame_older_than_100_ms():
    timestamps = builder_module.np.asarray([9.90])

    index = SimpleDatasetBuilder._latest_causal_camera_frame_index(
        timestamps,
        10.001,
    )

    assert index is None
