import warnings
from contextlib import contextmanager

import h5py
import numpy as np
import pandas as pd

from experiments.SimpleDatasetBuilder import (
    ActionSelector,
    DatasetBuilderSettings,
    GigasealAugmentationSettings,
    ObservationSelector,
    SimpleDatasetBuilder,
)


def _pressure_builder(action_selector=None):
    builder = object.__new__(SimpleDatasetBuilder)
    builder.action_selector = action_selector or ActionSelector(
        include_stage=False,
        include_pipette=False,
        include_pressure=True,
    )
    return builder


def _pressure_events():
    return pd.DataFrame(
        {
            "timestamp": [0.4, 1.4, 1.5],
            "event_kind": ["pressure", "atm_state", "zap"],
            "commanded_pressure_mbar": [-5.0, np.nan, np.nan],
            "pressure_atm_state": [np.nan, 1.0, np.nan],
            "zap_command": [np.nan, np.nan, 1.0],
        }
    )


def _break_in_event_window_builder(include_break_in_zap=True, radius=15):
    builder = _pressure_builder(
        ActionSelector(
            include_stage=True,
            include_pipette=True,
            include_pressure=False,
            include_break_in_zap=include_break_in_zap,
        )
    )
    builder.break_in_event_window_enabled = True
    builder.break_in_event_window_radius = radius
    return builder


def _gigaseal_event_window_builder(radius=15, action_selector=None):
    builder = _pressure_builder(
        action_selector
        or ActionSelector(
            include_stage=False,
            include_pipette=False,
            include_pressure=False,
        )
    )
    builder.gigaseal_start_trim_enabled = False
    builder.gigaseal_event_window_enabled = True
    builder.gigaseal_event_window_radius = radius
    builder.gigaseal_resistance_cutoff_enabled = False
    builder.gigaseal_resistance_cutoff = 1200.0
    builder.use_velocities = False
    return builder


def _break_in_graph_and_movement(num_rows):
    timestamps = np.arange(num_rows, dtype=np.float64)
    graph_values = np.column_stack(
        [
            timestamps,
            np.full(num_rows, 1.0, dtype=np.float64),
            np.linspace(100.0, 200.0, num_rows, dtype=np.float64),
        ]
    )
    movement_values = np.column_stack(
        [
            timestamps,
            np.zeros((num_rows, 6), dtype=np.float64),
        ]
    )
    return graph_values, movement_values


def _break_in_events_for_action_index(action_index):
    return _break_in_events_for_action_indices([action_index])


def _break_in_events_for_action_indices(action_indices):
    atm_values = [1.0 if idx % 2 == 0 else 0.0 for idx, _ in enumerate(action_indices)]
    return pd.DataFrame(
        {
            "timestamp": [float(action_index) + 0.5 for action_index in action_indices],
            "event_kind": ["atm_state"] * len(action_indices),
            "commanded_pressure_mbar": [np.nan] * len(action_indices),
            "pressure_atm_state": atm_values,
            "zap_command": [np.nan] * len(action_indices),
        }
    )


def _empty_pressure_events():
    return pd.DataFrame(
        columns=[
            "timestamp",
            "event_kind",
            "commanded_pressure_mbar",
            "pressure_atm_state",
            "zap_command",
        ]
    )


def test_pressure_command_observation_trace_does_not_use_future_events():
    builder = _pressure_builder()
    graph_values = np.asarray(
        [
            [0.0, 1.1, 100.0],
            [1.0, 1.2, 101.0],
            [2.0, 1.3, 102.0],
        ],
        dtype=np.float64,
    )

    commanded_pressure, atm_state = builder.get_attempt_pressure_observation_values(
        graph_values,
        _pressure_events(),
    )

    np.testing.assert_array_equal(commanded_pressure, np.asarray([0.0, -5.0, -5.0]))
    np.testing.assert_array_equal(atm_state, np.asarray([0.0, 0.0, 1.0]))


def test_pressure_action_targets_command_and_atm_after_observation():
    builder = _pressure_builder()
    graph_values = np.asarray(
        [
            [0.0, 1.1, 100.0],
            [1.0, 1.2, 101.0],
            [2.0, 1.3, 102.0],
        ],
        dtype=np.float64,
    )

    target_pressure, target_atm_state = builder.get_attempt_pressure_action_values(
        graph_values,
        _pressure_events(),
    )

    np.testing.assert_array_equal(target_pressure, np.asarray([-5.0, -5.0, -5.0]))
    np.testing.assert_array_equal(target_atm_state, np.asarray([0.0, 1.0, 1.0]))


def test_applied_pressure_action_zeroes_atm_rows():
    applied_pressure = SimpleDatasetBuilder.get_applied_pressure_action_values(
        np.asarray([-5.0, -10.0, -15.0, -20.0]),
        np.asarray([0.0, 1.0, -1.0, 0.0]),
    )

    np.testing.assert_array_equal(
        applied_pressure,
        np.asarray([-5.0, 0.0, -15.0, -20.0]),
    )


def test_binary_gigaseal_pressure_action_maps_negative_delta_to_neg_5():
    actions = SimpleDatasetBuilder.get_binary_gigaseal_pressure_action_values(
        np.asarray([0.0]),
        np.asarray([0.0]),
        np.asarray([-2.0]),
        np.asarray([0.0]),
    )

    np.testing.assert_array_equal(actions, np.asarray([[1.0, 0.0, 0.0]]))


def test_binary_gigaseal_pressure_action_maps_positive_delta_to_pos_5():
    actions = SimpleDatasetBuilder.get_binary_gigaseal_pressure_action_values(
        np.asarray([-2.0]),
        np.asarray([0.0]),
        np.asarray([0.0]),
        np.asarray([0.0]),
    )

    np.testing.assert_array_equal(actions, np.asarray([[0.0, 1.0, 0.0]]))


def test_binary_gigaseal_pressure_action_maps_target_atm_to_reset():
    actions = SimpleDatasetBuilder.get_binary_gigaseal_pressure_action_values(
        np.asarray([-5.0]),
        np.asarray([0.0]),
        np.asarray([-5.0]),
        np.asarray([1.0]),
    )

    np.testing.assert_array_equal(actions, np.asarray([[0.0, 0.0, 1.0]]))


def test_binary_gigaseal_pressure_action_maps_no_change_to_zero_row():
    actions = SimpleDatasetBuilder.get_binary_gigaseal_pressure_action_values(
        np.asarray([-5.0]),
        np.asarray([0.0]),
        np.asarray([-5.0]),
        np.asarray([0.0]),
    )

    np.testing.assert_array_equal(actions, np.asarray([[0.0, 0.0, 0.0]]))


def test_pressure_log_parser_includes_zap_events():
    builder = object.__new__(SimpleDatasetBuilder)
    log_values = pd.DataFrame(
        {
            "Time(HH:MM:SS)": [
                "2026-05-04 12:00:00",
                "2026-05-04 12:00:01",
            ],
            "Time(ms)": ["000", "500"],
            "Message": ["Setting pressure to -5 mbar", "zapping (Agent command)"],
        }
    )

    events = builder._parse_pressure_log_events(log_values, log_file="fake-log.csv")
    zap_events = events.loc[events["event_kind"] == "zap"]

    assert len(zap_events) == 1
    assert float(zap_events.iloc[0]["zap_command"]) == 1.0


def test_break_in_zap_actions_ignore_future_day_log_zaps():
    builder = _pressure_builder(
        ActionSelector(
            include_stage=False,
            include_pipette=False,
            include_pressure=False,
            include_break_in_zap=True,
        )
    )
    graph_values, _ = _break_in_graph_and_movement(3)
    pressure_events = pd.DataFrame(
        {
            "timestamp": [50.0],
            "event_kind": ["zap"],
            "commanded_pressure_mbar": [np.nan],
            "pressure_atm_state": [np.nan],
            "zap_command": [1.0],
        }
    )

    zap_actions = builder.get_attempt_zap_action_values(graph_values, pressure_events)

    np.testing.assert_array_equal(zap_actions, np.zeros(3, dtype=np.float64))


def test_resistance_input_window_uses_prior_samples_with_zero_padding():
    builder = object.__new__(SimpleDatasetBuilder)
    builder.settings = DatasetBuilderSettings(
        dataset_name="resistance_input_test.hdf5",
        resistance_input_window=3,
    )

    resistance_input = builder._compute_resistance_input_window(
        np.asarray([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    )

    np.testing.assert_array_equal(
        resistance_input,
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 10.0],
                [0.0, 10.0, 20.0],
                [10.0, 20.0, 30.0],
            ],
            dtype=np.float64,
        ),
    )


def test_history_windows_are_not_larger_than_event_trim_radius():
    settings = DatasetBuilderSettings(
        dataset_name="window_clamp_test.hdf5",
        resistance_input_window=30,
        resistance_slope_window=20,
    )

    SimpleDatasetBuilder._clamp_history_windows_to_trim(settings, window_limit=15)

    assert settings.resistance_input_window == 15
    assert settings.resistance_slope_window == 15


def _gigaseal_augmentation_builder(cfg):
    builder = object.__new__(SimpleDatasetBuilder)
    builder.gigaseal_augmentation = cfg
    builder.inaction_tolerance = 0.0
    builder.use_velocities = False
    builder.settings = DatasetBuilderSettings(
        dataset_name="augmentation_test.hdf5",
        resistance_slope_window=5,
        gigaseal_augmentation=cfg,
    )
    builder.observation_selector = ObservationSelector(
        include_pressure=True,
        include_resistance=True,
        include_resistance_slope=True,
        include_current=False,
        include_voltage=False,
        include_stage=False,
        include_pipette=False,
        include_camera=False,
        include_gigaseal_pressure_state=True,
        include_gigaseal_effective_pressure=True,
        include_gigaseal_observations_since_last_action=False,
        include_gigaseal_time_since_last_action=True,
    )
    builder.action_selector = ActionSelector(
        include_stage=False,
        include_pipette=False,
        include_pressure=False,
    )
    base_action_labels = ["commanded_pressure_mbar", "pressure_atm_state"]
    builder._last_action_labels = base_action_labels + ["observations_until_next_action"]
    builder._last_action_representation = "delta"
    builder._write_metadata_files = lambda: None
    return builder


def _augmentation_arrays(builder):
    base_actions = np.asarray(
        [[0.0, 0.0], [5.0, 0.0], [5.0, 1.0], [5.0, 1.0], [5.0, 1.0]],
        dtype=np.float64,
    )
    base_action_labels = ["commanded_pressure_mbar", "pressure_atm_state"]
    actions = np.column_stack(
        [
            base_actions,
            SimpleDatasetBuilder.get_observations_until_next_action_values(
                base_actions,
                base_action_labels,
            ),
        ]
    )
    timestamps = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    resistance = np.asarray([100.0, 120.0, 140.0, 160.0, 180.0], dtype=np.float64)
    return dict(
        actions=actions,
        dones=np.asarray([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        timestamp_values=timestamps,
        pressure_values=np.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64),
        pressure_state_values=np.asarray([0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64),
        effective_pressure_values=np.asarray([0.0, 5.0, 0.0, 0.0, 0.0], dtype=np.float64),
        resistance_values=resistance,
        resistance_slope_values=builder._compute_resistance_slope(resistance),
        resistance_input_values=builder._compute_resistance_input_window(resistance),
        current_values=None,
        voltage_values=None,
        stage_positions=None,
        pipette_positions=None,
        camera_frames=None,
        observations_since_last_action_values=None,
        time_since_last_action_values=SimpleDatasetBuilder.get_time_since_last_action_values(
            actions,
            builder._last_action_labels,
            timestamps,
        ),
    )


def test_non_gigaseal_pressure_actions_write_raw_commanded_pressure_only():
    builder = _pressure_builder(
        ActionSelector(
            include_stage=False,
            include_pipette=False,
            include_pressure=True,
        )
    )
    builder.use_velocities = False
    builder._pipette_positions_in_image_space = (
        lambda pipette_positions, rig_recorder_data_folder=None: (pipette_positions, (1.0, 1.0))
    )
    graph_values = np.asarray(
        [
            [0.0, 1.1, 100.0],
            [1.0, 1.2, 101.0],
            [2.0, 1.3, 102.0],
        ],
        dtype=np.float64,
    )
    movement_values = np.column_stack(
        [
            graph_values[:, 0],
            np.zeros((graph_values.shape[0], 6), dtype=np.float64),
        ]
    )

    actions = builder.get_attempt_actions(
        movement_values,
        attempt_graph_values=graph_values,
        pressure_events=_pressure_events(),
    )

    np.testing.assert_array_equal(
        actions,
        np.asarray(
            [
                [-5.0],
                [-5.0],
                [-5.0],
            ]
        ),
    )
    assert builder._last_action_labels == ["commanded_pressure_mbar"]


def test_gigaseal_force_pressure_actions_prepend_pressure_columns():
    builder = _pressure_builder(
        ActionSelector(
            include_stage=False,
            include_pipette=True,
            include_pressure=False,
        )
    )
    builder.use_velocities = False
    builder._pipette_positions_in_image_space = (
        lambda pipette_positions, rig_recorder_data_folder=None: (pipette_positions, (1.0, 1.0))
    )
    graph_values = np.asarray(
        [
            [0.0, 1.1, 100.0],
            [1.0, 1.2, 101.0],
            [2.0, 1.3, 102.0],
        ],
        dtype=np.float64,
    )
    movement_values = np.column_stack(
        [
            graph_values[:, 0],
            np.zeros((graph_values.shape[0], 6), dtype=np.float64),
        ]
    )

    actions = builder.get_attempt_actions(
        movement_values,
        attempt_graph_values=graph_values,
        pressure_events=_pressure_events(),
        force_pressure_action=True,
    )

    np.testing.assert_array_equal(actions[:, :2], np.asarray([[-5.0, 0.0], [-5.0, 1.0], [-5.0, 1.0]]))
    assert builder._last_action_labels[:2] == ["commanded_pressure_mbar", "pressure_atm_state"]


def test_gigaseal_force_pressure_actions_can_use_applied_pressure_column():
    builder = _pressure_builder(
        ActionSelector(
            include_stage=False,
            include_pipette=False,
            include_pressure=False,
            combine_gigaseal_pressure_action=True,
        )
    )
    builder.use_velocities = False
    builder._pipette_positions_in_image_space = (
        lambda pipette_positions, rig_recorder_data_folder=None: (pipette_positions, (1.0, 1.0))
    )
    graph_values = np.asarray(
        [
            [0.0, 1.1, 100.0],
            [1.0, 1.2, 101.0],
            [2.0, 1.3, 102.0],
        ],
        dtype=np.float64,
    )
    movement_values = np.column_stack(
        [
            graph_values[:, 0],
            np.zeros((graph_values.shape[0], 6), dtype=np.float64),
        ]
    )

    actions = builder.get_attempt_actions(
        movement_values,
        attempt_graph_values=graph_values,
        pressure_events=_pressure_events(),
        force_pressure_action=True,
    )

    np.testing.assert_array_equal(actions, np.asarray([[-5.0], [0.0], [0.0]]))
    assert builder._last_action_labels == ["applied_pressure_mbar"]


def test_gigaseal_force_pressure_actions_can_use_binary_one_hot_columns():
    builder = _pressure_builder(
        ActionSelector(
            include_stage=False,
            include_pipette=False,
            include_pressure=False,
            binary_gigaseal_pressure_action=True,
        )
    )
    builder.use_velocities = False
    builder._pipette_positions_in_image_space = (
        lambda pipette_positions, rig_recorder_data_folder=None: (pipette_positions, (1.0, 1.0))
    )
    graph_values = np.asarray(
        [
            [0.0, 1.1, 100.0],
            [1.0, 1.2, 101.0],
            [2.0, 1.3, 102.0],
        ],
        dtype=np.float64,
    )
    movement_values = np.column_stack(
        [
            graph_values[:, 0],
            np.zeros((graph_values.shape[0], 6), dtype=np.float64),
        ]
    )

    actions = builder.get_attempt_actions(
        movement_values,
        attempt_graph_values=graph_values,
        pressure_events=_pressure_events(),
        force_pressure_action=True,
    )

    np.testing.assert_array_equal(
        actions,
        np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    )
    assert builder._last_action_labels == [
        "gigaseal_delta_neg_5",
        "gigaseal_delta_pos_5",
        "gigaseal_reset",
    ]


def test_break_in_force_actions_write_atm_and_zap_only():
    builder = _pressure_builder(
        ActionSelector(
            include_stage=True,
            include_pipette=True,
            include_pressure=False,
        )
    )
    builder.use_velocities = False
    builder._pipette_positions_in_image_space = (
        lambda pipette_positions, rig_recorder_data_folder=None: (pipette_positions, (1.0, 1.0))
    )
    graph_values = np.asarray(
        [
            [0.0, 1.1, 100.0],
            [1.0, 1.2, 101.0],
            [2.0, 1.3, 102.0],
        ],
        dtype=np.float64,
    )
    movement_values = np.column_stack(
        [
            graph_values[:, 0],
            np.ones((graph_values.shape[0], 6), dtype=np.float64),
        ]
    )

    actions = builder.get_attempt_actions(
        movement_values,
        attempt_graph_values=graph_values,
        pressure_events=_pressure_events(),
        force_break_in_action=True,
    )

    np.testing.assert_array_equal(
        actions,
        np.asarray(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=np.float64,
        ),
    )
    assert builder._last_action_labels == [
        "pressure_atm_state",
        "zap_command",
    ]


def test_break_in_force_actions_can_omit_sparse_zap_dimension():
    builder = _pressure_builder(
        ActionSelector(
            include_stage=True,
            include_pipette=True,
            include_pressure=False,
            include_break_in_zap=False,
        )
    )
    builder.use_velocities = False
    builder._pipette_positions_in_image_space = (
        lambda pipette_positions, rig_recorder_data_folder=None: (pipette_positions, (1.0, 1.0))
    )
    graph_values = np.asarray(
        [
            [0.0, 1.1, 100.0],
            [1.0, 1.2, 101.0],
            [2.0, 1.3, 102.0],
        ],
        dtype=np.float64,
    )
    movement_values = np.column_stack(
        [
            graph_values[:, 0],
            np.ones((graph_values.shape[0], 6), dtype=np.float64),
        ]
    )

    actions = builder.get_attempt_actions(
        movement_values,
        attempt_graph_values=graph_values,
        pressure_events=_pressure_events(),
        force_break_in_action=True,
    )

    np.testing.assert_array_equal(
        actions,
        np.asarray([[0.0], [1.0], [1.0]], dtype=np.float64),
    )
    assert builder._last_action_labels == ["pressure_atm_state"]


def test_break_in_event_window_returns_31_aligned_rows():
    builder = _break_in_event_window_builder(radius=15)
    graph_values, movement_values = _break_in_graph_and_movement(50)

    trimmed_graph, trimmed_movement, gigaseal_start, gigaseal_cutoff, skip_reason = (
        builder.truncate_attempt_for_state(
            graph_values,
            movement_values,
            "break_in",
            pressure_events=_break_in_events_for_action_index(25),
        )
    )

    assert skip_reason is None
    assert not gigaseal_start
    assert not gigaseal_cutoff
    assert trimmed_graph.shape[0] == 31
    assert trimmed_movement.shape[0] == 31
    np.testing.assert_array_equal(trimmed_graph[:, 0], np.arange(10.0, 41.0))
    np.testing.assert_array_equal(trimmed_movement[:, 0], trimmed_graph[:, 0])


def test_break_in_event_window_spans_first_through_last_event_command():
    builder = _break_in_event_window_builder(radius=15)
    graph_values, movement_values = _break_in_graph_and_movement(60)

    trimmed_graph, trimmed_movement, _, _, skip_reason = builder.truncate_attempt_for_state(
        graph_values,
        movement_values,
        "break_in",
        pressure_events=_break_in_events_for_action_indices([20, 35]),
    )

    assert skip_reason is None
    assert trimmed_graph.shape[0] == 46
    np.testing.assert_array_equal(trimmed_graph[:, 0], np.arange(5.0, 51.0))
    np.testing.assert_array_equal(trimmed_movement[:, 0], trimmed_graph[:, 0])


def test_break_in_event_window_clamps_near_attempt_edges():
    builder = _break_in_event_window_builder(radius=15)

    for action_index, expected_start, expected_stop in (
        (4, 0, 20),
        (28, 13, 30),
    ):
        graph_values, movement_values = _break_in_graph_and_movement(30)
        trimmed_graph, trimmed_movement, _, _, skip_reason = builder.truncate_attempt_for_state(
            graph_values,
            movement_values,
            "break_in",
            pressure_events=_break_in_events_for_action_index(action_index),
        )

        assert skip_reason is None
        np.testing.assert_array_equal(
            trimmed_graph[:, 0],
            np.arange(float(expected_start), float(expected_stop)),
        )
        np.testing.assert_array_equal(trimmed_movement[:, 0], trimmed_graph[:, 0])


def test_break_in_event_window_reaches_before_state_attempt_when_recording_has_context():
    builder = _break_in_event_window_builder(radius=15)
    graph_values, movement_values = _break_in_graph_and_movement(50)
    attempt_graph_values = graph_values[20:40]
    attempt_movement_values = movement_values[20:40]

    trimmed_graph, trimmed_movement, _, _, skip_reason = builder.truncate_attempt_for_state(
        attempt_graph_values,
        attempt_movement_values,
        "break_in",
        pressure_events=_break_in_events_for_action_index(25),
        recording_graph_values=graph_values,
        recording_movement_values=movement_values,
    )

    assert skip_reason is None
    np.testing.assert_array_equal(trimmed_graph[:, 0], np.arange(10.0, 41.0))
    np.testing.assert_array_equal(trimmed_movement[:, 0], trimmed_graph[:, 0])


def test_gigaseal_event_window_keeps_15_rows_before_first_pressure_event():
    builder = _gigaseal_event_window_builder(radius=15)
    graph_values, movement_values = _break_in_graph_and_movement(50)
    pressure_events = pd.DataFrame(
        {
            "timestamp": [25.5],
            "event_kind": ["pressure"],
            "commanded_pressure_mbar": [-5.0],
            "pressure_atm_state": [np.nan],
            "zap_command": [np.nan],
        }
    )

    trimmed_graph, trimmed_movement, gigaseal_start, gigaseal_cutoff, skip_reason = (
        builder.truncate_attempt_for_state(
            graph_values,
            movement_values,
            "gigaseal",
            pressure_events=pressure_events,
        )
    )

    assert skip_reason is None
    assert gigaseal_start
    assert not gigaseal_cutoff
    np.testing.assert_array_equal(trimmed_graph[:, 0], np.arange(10.0, 50.0))
    np.testing.assert_array_equal(trimmed_movement[:, 0], trimmed_graph[:, 0])


def test_gigaseal_event_window_uses_binary_pressure_action_representation():
    builder = _gigaseal_event_window_builder(
        radius=15,
        action_selector=ActionSelector(
            include_stage=False,
            include_pipette=False,
            include_pressure=False,
            binary_gigaseal_pressure_action=True,
        ),
    )
    graph_values, movement_values = _break_in_graph_and_movement(50)
    pressure_events = pd.DataFrame(
        {
            "timestamp": [25.5],
            "event_kind": ["pressure"],
            "commanded_pressure_mbar": [-2.0],
            "pressure_atm_state": [np.nan],
            "zap_command": [np.nan],
        }
    )

    actions, labels = builder._get_gigaseal_event_window_actions(
        graph_values,
        movement_values,
        pressure_events,
    )
    action_events = SimpleDatasetBuilder.get_action_event_mask(actions, labels)

    assert labels == [
        "gigaseal_delta_neg_5",
        "gigaseal_delta_pos_5",
        "gigaseal_reset",
    ]
    np.testing.assert_array_equal(actions[25], np.asarray([1.0, 0.0, 0.0]))
    assert action_events[25]
    assert not action_events[24]
    assert not action_events[26]

    trimmed_graph, trimmed_movement, _, _, skip_reason = builder.truncate_attempt_for_state(
        graph_values,
        movement_values,
        "gigaseal",
        pressure_events=pressure_events,
    )

    assert skip_reason is None
    np.testing.assert_array_equal(trimmed_graph[:, 0], np.arange(10.0, 50.0))
    np.testing.assert_array_equal(trimmed_movement[:, 0], trimmed_graph[:, 0])


def test_gigaseal_event_window_reaches_before_state_attempt_when_recording_has_context():
    builder = _gigaseal_event_window_builder(radius=15)
    graph_values, movement_values = _break_in_graph_and_movement(50)
    attempt_graph_values = graph_values[20:]
    attempt_movement_values = movement_values[20:]
    pressure_events = pd.DataFrame(
        {
            "timestamp": [25.5],
            "event_kind": ["pressure"],
            "commanded_pressure_mbar": [-5.0],
            "pressure_atm_state": [np.nan],
            "zap_command": [np.nan],
        }
    )

    trimmed_graph, trimmed_movement, _, _, skip_reason = builder.truncate_attempt_for_state(
        attempt_graph_values,
        attempt_movement_values,
        "gigaseal",
        pressure_events=pressure_events,
        recording_graph_values=graph_values,
        recording_movement_values=movement_values,
    )

    assert skip_reason is None
    np.testing.assert_array_equal(trimmed_graph[:, 0], np.arange(10.0, 50.0))
    np.testing.assert_array_equal(trimmed_movement[:, 0], trimmed_graph[:, 0])


def test_gigaseal_event_window_uses_first_event_from_any_action_dimension():
    builder = _gigaseal_event_window_builder(
        radius=15,
        action_selector=ActionSelector(
            include_stage=False,
            include_pipette=True,
            include_pressure=False,
        ),
    )
    graph_values, movement_values = _break_in_graph_and_movement(50)
    movement_values[25:, 4] = 1.0

    trimmed_graph, trimmed_movement, _, _, skip_reason = builder.truncate_attempt_for_state(
        graph_values,
        movement_values,
        "gigaseal",
        pressure_events=_empty_pressure_events(),
    )

    assert skip_reason is None
    np.testing.assert_array_equal(trimmed_graph[:, 0], np.arange(10.0, 50.0))
    np.testing.assert_array_equal(trimmed_movement[:, 0], trimmed_graph[:, 0])


def test_break_in_event_window_skips_attempt_without_action_event():
    builder = _break_in_event_window_builder(radius=15)
    graph_values, movement_values = _break_in_graph_and_movement(30)
    pressure_events = pd.DataFrame(
        {
            "timestamp": [0.5],
            "event_kind": ["pressure"],
            "commanded_pressure_mbar": [-5.0],
            "pressure_atm_state": [np.nan],
            "zap_command": [np.nan],
        }
    )

    trimmed_graph, trimmed_movement, _, _, skip_reason = builder.truncate_attempt_for_state(
        graph_values,
        movement_values,
        "break_in",
        pressure_events=pressure_events,
    )

    assert trimmed_graph.shape[0] == graph_values.shape[0]
    assert trimmed_movement.shape[0] == movement_values.shape[0]
    assert skip_reason == "no break-in ATM or zap action event was found in the attempt"

    builder = _gigaseal_augmentation_builder(GigasealAugmentationSettings(enabled=False))
    builder.dataset_name = "break_in_no_event.hdf5"
    builder.load_next_obs = False
    builder.gigaseal_start_trim_enabled = False
    builder.gigaseal_resistance_cutoff_enabled = False
    builder.gigaseal_resistance_cutoff = 1200.0
    builder.break_in_event_window_enabled = True
    builder.break_in_event_window_radius = 15
    builder.val_ratio = 0.0
    builder.omit_stage_movement = False
    builder.skip_invalid_observations = False
    builder.inaction = 0
    builder.rng = np.random.default_rng(0)
    builder._split_keys = {"train": []}
    builder._last_stage_motion_detected = False
    builder.load_experiment_data = lambda folder: (graph_values, movement_values)
    builder.get_timestamps_for_all_successful_state_attempts = (
        lambda folder, start, end: {"break_in": [(0.0, 29.0)]}
    )
    builder._load_pressure_log_events = lambda folder: pressure_events

    @contextmanager
    def state_context(state_name):
        yield

    captured = []
    builder._use_state_context = state_context
    builder.add_attempt_demo_to_dataset = lambda **kwargs: captured.append(kwargs) or "demo_0"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        builder.add_demo("fake-folder", record_to_file=True)

    assert captured == []
    assert builder._split_keys["train"] == []
    assert any("no break-in ATM or zap action event" in str(item.message) for item in caught)


def test_gigaseal_since_last_action_observations_ignore_current_action_target():
    actions = np.asarray(
        [[0.0, 0.0], [-5.0, 0.0], [-5.0, 1.0], [-5.0, 1.0], [-5.0, 1.0]],
        dtype=np.float64,
    )
    labels = ["commanded_pressure_mbar", "pressure_atm_state"]
    timestamps = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    counts = SimpleDatasetBuilder.get_observations_since_last_action_values(
        actions,
        labels,
    )
    elapsed = SimpleDatasetBuilder.get_time_since_last_action_values(
        actions,
        labels,
        timestamps,
    )

    np.testing.assert_array_equal(counts, np.asarray([0.0, 0.0, 0.0, 0.0, 1.0]))
    np.testing.assert_array_equal(elapsed, np.asarray([0.0, 0.0, 0.0, 0.0, 1.0]))


def test_gigaseal_since_last_action_treats_applied_pressure_as_change_event():
    actions = np.asarray([[0.0], [-5.0], [0.0], [0.0], [-10.0]], dtype=np.float64)
    labels = ["applied_pressure_mbar"]
    timestamps = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    counts = SimpleDatasetBuilder.get_observations_since_last_action_values(
        actions,
        labels,
    )
    elapsed = SimpleDatasetBuilder.get_time_since_last_action_values(
        actions,
        labels,
        timestamps,
    )

    np.testing.assert_array_equal(counts, np.asarray([0.0, 0.0, 0.0, 0.0, 1.0]))
    np.testing.assert_array_equal(elapsed, np.asarray([0.0, 0.0, 0.0, 0.0, 1.0]))


def test_gigaseal_binary_one_hot_rows_drive_action_timing_events():
    actions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    labels = ActionSelector(binary_gigaseal_pressure_action=True).gigaseal_pressure_axis_labels()

    action_events = SimpleDatasetBuilder.get_action_event_mask(actions, labels)
    counts = SimpleDatasetBuilder.get_observations_since_last_action_values(
        actions,
        labels,
    )

    np.testing.assert_array_equal(
        action_events,
        np.asarray([False, True, False, True, True, False]),
    )
    np.testing.assert_array_equal(counts, np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]))


def test_gigaseal_until_next_action_counts_future_wait_rows():
    actions = np.asarray(
        [[0.0, 0.0], [0.0, 0.0], [-5.0, 0.0], [-5.0, 0.0], [-5.0, 1.0]],
        dtype=np.float64,
    )
    labels = ["commanded_pressure_mbar", "pressure_atm_state"]

    counts = SimpleDatasetBuilder.get_observations_until_next_action_values(
        actions,
        labels,
    )

    np.testing.assert_array_equal(counts, np.asarray([1.0, 0.0, 1.0, 0.0, 0.0]))


def test_gigaseal_pressure_observation_keeps_continuous_graph_pressure():
    builder = object.__new__(SimpleDatasetBuilder)
    builder.dataset_name = "pressure_regression.hdf5"
    builder.load_next_obs = False
    builder.gigaseal_start_trim_enabled = False
    builder.gigaseal_resistance_cutoff_enabled = False
    builder.gigaseal_resistance_cutoff = 1200.0
    builder.val_ratio = 0.0
    builder.omit_stage_movement = False
    builder.skip_invalid_observations = False
    builder.inaction = 0
    builder.inaction_tolerance = 0.0
    builder.rng = np.random.default_rng(0)
    builder._split_keys = {"train": []}
    builder._last_action_labels = []
    builder.settings = DatasetBuilderSettings(
        dataset_name="pressure_regression.hdf5",
        resistance_input_window=3,
    )

    builder.observation_selector = ObservationSelector(
        include_pressure=True,
        include_resistance=False,
        include_resistance_slope=False,
        include_current=False,
        include_voltage=False,
        include_stage=False,
        include_pipette=False,
        include_camera=False,
        include_gigaseal_pressure_state=True,
        include_gigaseal_effective_pressure=True,
        include_gigaseal_resistance_input=True,
        include_gigaseal_observations_since_last_action=False,
        include_gigaseal_time_since_last_action=False,
    )
    builder.action_selector = ActionSelector(
        include_stage=False,
        include_pipette=False,
        include_pressure=False,
    )

    graph_values = np.asarray(
        [
            [0.0, 1.1, 100.0],
            [1.0, 1.2, 101.0],
            [2.0, 1.3, 102.0],
        ],
        dtype=np.float64,
    )
    movement_values = np.column_stack(
        [
            graph_values[:, 0],
            np.zeros((graph_values.shape[0], 6), dtype=np.float64),
        ]
    )
    pressure_events = pd.DataFrame(
        {
            "timestamp": [0.4],
            "event_kind": ["pressure"],
            "commanded_pressure_mbar": [50.0],
            "pressure_atm_state": [np.nan],
        }
    )

    builder.load_experiment_data = lambda folder: (graph_values, movement_values)
    builder.get_timestamps_for_all_successful_state_attempts = (
        lambda folder, start, end: {"gigasealing": [(0.0, 2.0)]}
    )
    builder._load_pressure_log_events = lambda folder: pressure_events

    @contextmanager
    def state_context(state_name):
        yield

    builder._use_state_context = state_context
    builder.begin_filter_context = lambda split_label, demo_seed: None
    builder.end_filter_context = lambda: None
    builder._register_processed_folder = lambda folder: False
    builder._write_metadata_files = lambda: None
    builder._pipette_positions_in_image_space = (
        lambda pipette_positions, rig_recorder_data_folder=None: (pipette_positions, (1.0, 1.0))
    )

    def fake_actions(attempt_movement_values, attempt_graph_values=None, **kwargs):
        builder._last_action_labels = []
        builder._last_stage_motion_detected = False
        return np.zeros((attempt_graph_values.shape[0], 0), dtype=np.float64)

    builder.get_attempt_actions = fake_actions

    captured = {}

    def capture_demo(**kwargs):
        captured.update(kwargs)
        return "demo_0"

    builder.add_attempt_demo_to_dataset = capture_demo

    builder.add_demo("fake-folder", record_to_file=True)

    np.testing.assert_array_equal(captured["pressure_values"], graph_values[:, 1])
    np.testing.assert_array_equal(captured["effective_pressure_values"], np.asarray([0.0, 50.0, 50.0]))
    assert captured["resistance_values"] is None
    np.testing.assert_array_equal(
        captured["resistance_input_values"],
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 100.0],
                [0.0, 100.0, 101.0],
            ],
            dtype=np.float64,
        ),
    )


def test_gigaseal_pressure_observations_stay_current_while_actions_target_future():
    builder = object.__new__(SimpleDatasetBuilder)
    builder.dataset_name = "pressure_alignment.hdf5"
    builder.load_next_obs = False
    builder.gigaseal_start_trim_enabled = False
    builder.gigaseal_resistance_cutoff_enabled = False
    builder.gigaseal_resistance_cutoff = 1200.0
    builder.val_ratio = 0.0
    builder.omit_stage_movement = False
    builder.skip_invalid_observations = False
    builder.inaction = 0
    builder.inaction_tolerance = 0.0
    builder.use_velocities = False
    builder.rng = np.random.default_rng(0)
    builder._split_keys = {"train": []}
    builder._last_action_labels = []

    builder.observation_selector = ObservationSelector(
        include_pressure=True,
        include_resistance=True,
        include_resistance_slope=False,
        include_current=False,
        include_voltage=False,
        include_stage=False,
        include_pipette=False,
        include_camera=False,
        include_gigaseal_pressure_state=True,
        include_gigaseal_effective_pressure=True,
        include_gigaseal_observations_since_last_action=False,
        include_gigaseal_time_since_last_action=False,
    )
    builder.action_selector = ActionSelector(
        include_stage=False,
        include_pipette=False,
        include_pressure=False,
    )

    graph_values = np.asarray(
        [
            [0.0, 1.1, 100.0],
            [1.0, 1.2, 101.0],
            [2.0, 1.3, 102.0],
            [3.0, 1.4, 103.0],
        ],
        dtype=np.float64,
    )
    movement_values = np.column_stack(
        [
            graph_values[:, 0],
            np.zeros((graph_values.shape[0], 6), dtype=np.float64),
        ]
    )
    pressure_events = pd.DataFrame(
        {
            "timestamp": [0.4, 1.4],
            "event_kind": ["pressure", "atm_state"],
            "commanded_pressure_mbar": [50.0, np.nan],
            "pressure_atm_state": [np.nan, 1.0],
        }
    )

    builder.load_experiment_data = lambda folder: (graph_values, movement_values)
    builder.get_timestamps_for_all_successful_state_attempts = (
        lambda folder, start, end: {"gigasealing": [(0.0, 3.0)]}
    )
    builder._load_pressure_log_events = lambda folder: pressure_events

    @contextmanager
    def state_context(state_name):
        yield

    builder._use_state_context = state_context
    builder.begin_filter_context = lambda split_label, demo_seed: None
    builder.end_filter_context = lambda: None
    builder._register_processed_folder = lambda folder: False
    builder._write_metadata_files = lambda: None
    builder._pipette_positions_in_image_space = (
        lambda pipette_positions, rig_recorder_data_folder=None: (pipette_positions, (1.0, 1.0))
    )

    captured = {}

    def capture_demo(**kwargs):
        captured.update(kwargs)
        return "demo_0"

    builder.add_attempt_demo_to_dataset = capture_demo

    builder.add_demo("fake-folder", record_to_file=True)

    np.testing.assert_array_equal(captured["effective_pressure_values"], np.asarray([0.0, 50.0, 0.0, 0.0]))
    np.testing.assert_array_equal(captured["pressure_state_values"], np.asarray([0.0, 0.0, 1.0, 1.0]))
    assert captured["observations_since_last_action_values"] is None
    np.testing.assert_array_equal(
        captured["actions"],
        np.asarray(
            [
                [50.0, 0.0, 0.0],
                [50.0, 1.0, 0.0],
                [50.0, 1.0, 0.0],
                [50.0, 1.0, 0.0],
            ]
        ),
    )
    assert builder._last_action_labels == [
        "commanded_pressure_mbar",
        "pressure_atm_state",
        "observations_until_next_action",
    ]


def test_gigaseal_sensor_augmentation_leaves_sources_actions_and_command_state_unchanged():
    cfg = GigasealAugmentationSettings(
        enabled=True,
        copies_per_demo=1,
        pressure_noise_std=0.0,
        pressure_offset_std=1.0,
        resistance_log_noise_std=0.05,
        sensor_stutter_probability=0.0,
        prefix_hold_probability=0.0,
    )
    builder = _gigaseal_augmentation_builder(cfg)
    arrays = _augmentation_arrays(builder)
    originals = {key: None if value is None else value.copy() for key, value in arrays.items()}

    augmented = builder._make_gigaseal_augmented_payloads(
        17,
        "train",
        **arrays,
    )

    assert len(augmented) == 1
    payload, attrs = augmented[0]
    assert attrs["augmentation_kind"] == "gigaseal_conservative_copy"
    for key, original in originals.items():
        if original is not None:
            np.testing.assert_array_equal(arrays[key], original)
    np.testing.assert_array_equal(payload.actions, arrays["actions"])
    np.testing.assert_array_equal(payload.pressure_state_values, arrays["pressure_state_values"])
    np.testing.assert_array_equal(payload.effective_pressure_values, arrays["effective_pressure_values"])
    assert not np.array_equal(payload.pressure_values, arrays["pressure_values"])
    assert np.all(payload.resistance_values > 0.0)
    np.testing.assert_allclose(
        payload.resistance_slope_values,
        builder._compute_resistance_slope(payload.resistance_values),
    )
    np.testing.assert_allclose(
        payload.resistance_input_values,
        builder._compute_resistance_input_window(payload.resistance_values),
    )


def test_break_in_demo_gets_pressure_command_observations_but_is_not_augmented():
    cfg = GigasealAugmentationSettings(enabled=True, copies_per_demo=2)
    builder = _gigaseal_augmentation_builder(cfg)
    builder.dataset_name = "non_gigaseal.hdf5"
    builder.load_next_obs = False
    builder.gigaseal_start_trim_enabled = False
    builder.gigaseal_resistance_cutoff_enabled = False
    builder.gigaseal_resistance_cutoff = 1200.0
    builder.val_ratio = 0.0
    builder.omit_stage_movement = False
    builder.skip_invalid_observations = False
    builder.inaction = 0
    builder.rng = np.random.default_rng(0)
    builder._split_keys = {"train": []}
    builder._last_stage_motion_detected = False
    builder.settings.resistance_input_window = 2
    builder.observation_selector.include_gigaseal_resistance_input = True

    graph_values = np.asarray(
        [
            [0.0, 1.1, 100.0],
            [1.0, 1.2, 101.0],
            [2.0, 1.3, 102.0],
        ],
        dtype=np.float64,
    )
    movement_values = np.column_stack(
        [
            graph_values[:, 0],
            np.zeros((graph_values.shape[0], 6), dtype=np.float64),
        ]
    )
    builder.load_experiment_data = lambda folder: (graph_values, movement_values)
    builder.get_timestamps_for_all_successful_state_attempts = (
        lambda folder, start, end: {"break_in": [(0.0, 2.0)]}
    )
    builder._load_pressure_log_events = lambda folder: _pressure_events()

    @contextmanager
    def state_context(state_name):
        yield

    builder._use_state_context = state_context
    builder.begin_filter_context = lambda split_label, demo_seed: None
    builder.end_filter_context = lambda: None
    builder._register_processed_folder = lambda folder: False
    builder._write_metadata_files = lambda: None
    builder._pipette_positions_in_image_space = (
        lambda pipette_positions, rig_recorder_data_folder=None: (pipette_positions, (1.0, 1.0))
    )

    captured = []

    def capture_demo(**kwargs):
        captured.append(kwargs)
        return f"demo_{len(captured) - 1}"

    builder.add_attempt_demo_to_dataset = capture_demo

    builder.add_demo("fake-folder", record_to_file=True)

    assert len(captured) == 1
    assert "demo_attrs" not in captured[0] or captured[0]["demo_attrs"] is None
    assert builder._last_action_labels == [
        "pressure_atm_state",
        "zap_command",
        "observations_until_next_action",
    ]
    np.testing.assert_array_equal(captured[0]["pressure_state_values"], np.asarray([0.0, 0.0, 1.0]))
    np.testing.assert_array_equal(captured[0]["effective_pressure_values"], np.asarray([0.0, -5.0, 0.0]))
    np.testing.assert_array_equal(
        captured[0]["resistance_input_values"],
        np.asarray(
            [
                [0.0, 0.0],
                [0.0, 100.0],
                [100.0, 101.0],
            ],
            dtype=np.float64,
        ),
    )
    np.testing.assert_array_equal(
        captured[0]["actions"],
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
    )
    assert captured[0]["observations_since_last_action_values"] is None
    np.testing.assert_array_equal(
        captured[0]["time_since_last_action_values"],
        np.asarray([0.0, 0.0, 0.0]),
    )
    assert builder._split_keys["train"] == ["demo_0"]


def test_gigaseal_prefix_hold_recomputes_counters_and_writes_augmented_metadata(tmp_path):
    cfg = GigasealAugmentationSettings(
        enabled=True,
        copies_per_demo=1,
        pressure_noise_std=0.0,
        pressure_offset_std=0.0,
        resistance_log_noise_std=0.0,
        sensor_stutter_probability=0.0,
        prefix_hold_probability=1.0,
        prefix_hold_max_frames=1,
    )
    builder = _gigaseal_augmentation_builder(cfg)
    builder.dataset_name = "prefix_hold.hdf5"
    builder.dataset_path = tmp_path / "prefix_hold.hdf5"
    arrays = _augmentation_arrays(builder)

    augmented = builder._make_gigaseal_augmented_payloads(
        23,
        "train",
        **arrays,
    )

    assert len(augmented) == 1
    payload, attrs = augmented[0]
    assert payload.actions.shape[0] == arrays["actions"].shape[0] + 2
    np.testing.assert_array_equal(
        payload.actions[:5],
        np.asarray(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [5.0, 0.0, 1.0],
                [5.0, 0.0, 0.0],
                [5.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
    )
    assert payload.observations_since_last_action_values is None
    np.testing.assert_array_equal(
        payload.resistance_input_values,
        builder._compute_resistance_input_window(payload.resistance_values),
    )
    np.testing.assert_array_equal(payload.pressure_state_values[:5], np.asarray([0.0, 0.0, 0.0, 0.0, 1.0]))
    np.testing.assert_array_equal(payload.effective_pressure_values[:5], np.asarray([0.0, 0.0, 5.0, 5.0, 0.0]))

    source_key = builder.add_attempt_demo_to_dataset(
        num_samples=arrays["actions"].shape[0],
        actions=arrays["actions"],
        dones=arrays["dones"],
        pressure_values=arrays["pressure_values"],
        resistance_values=arrays["resistance_values"],
        resistance_slope_values=arrays["resistance_slope_values"],
        current_values=None,
        voltage_values=None,
        stage_positions=None,
        pipette_positions=None,
        camera_frames=None,
        next_pressure_values=None,
        next_resistance_values=None,
        next_resistance_slope_values=None,
        next_current_values=None,
        next_voltage_values=None,
        next_stage_positions=None,
        next_pipette_positions=None,
        next_camera_frames=None,
        include_next_obs=False,
        include_camera=False,
        split_label="train",
        pressure_state_values=arrays["pressure_state_values"],
        effective_pressure_values=arrays["effective_pressure_values"],
        observations_since_last_action_values=arrays["observations_since_last_action_values"],
        time_since_last_action_values=arrays["time_since_last_action_values"],
    )
    augmented_key = builder._write_gigaseal_augmented_payload(
        payload,
        "train",
        source_key,
        attrs,
        include_next_obs=True,
        include_camera=False,
    )

    with h5py.File(builder.dataset_path, "r") as hf:
        source = hf[f"data/{source_key}"]
        augmented_group = hf[f"data/{augmented_key}"]
        np.testing.assert_array_equal(source["obs/pressure"][:, 0], arrays["pressure_values"])
        assert "augmentation_kind" not in source.attrs
        assert augmented_group.attrs["augmentation_kind"] == "gigaseal_conservative_copy"
        assert augmented_group.attrs["augmentation_source_demo"] == source_key
        assert augmented_group.attrs["augmentation_seed"] == attrs["augmentation_seed"]
        assert augmented_group.attrs["num_samples"] == payload.actions.shape[0]
        np.testing.assert_array_equal(augmented_group["actions"][:], payload.actions)
        assert [
            axis.decode("utf-8")
            for axis in augmented_group["actions"].attrs["axes"]
        ] == builder._last_action_labels
        assert "observations_since_last_action" not in source["obs"]
        assert "observations_since_last_action" not in augmented_group["obs"]
        np.testing.assert_array_equal(
            augmented_group["next_obs/pressure"][:, 0],
            np.r_[payload.pressure_values[1:], payload.pressure_values[-1]],
        )
        assert "observations_since_last_action" not in augmented_group["next_obs"]
        np.testing.assert_array_equal(
            augmented_group["next_obs/resistance_input"][:],
            np.concatenate(
                [payload.resistance_input_values[1:], payload.resistance_input_values[-1:]],
                axis=0,
            ),
        )
