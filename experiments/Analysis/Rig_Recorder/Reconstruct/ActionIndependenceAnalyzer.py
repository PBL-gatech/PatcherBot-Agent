"""Find action rows that activate more than one movement dimension."""

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import h5py
import numpy as np


# Edit this path when analyzing a different dataset from an IDE.
HDF5_FILE_PATH = (
    r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent"
    r"\experiments\Datasets\PatcherBotViewer_test_dataset_v0_001"
    r"\PatcherBotViewer_test_dataset_v0_001_hunt_cell.hdf5"
)

EXCLUDED_ACTION_AXES = ("commanded_pressure_mbar",)


@dataclass(frozen=True)
class ActionViolation:
    """One action row containing multiple active movement dimensions."""

    row_index: int
    active_dimensions: tuple[str, ...]
    active_values: tuple[float, ...]


@dataclass(frozen=True)
class DemoAnalysis:
    """Action-independence results for one demonstration."""

    demo_name: str
    row_count: int
    action_dimensions: tuple[str, ...]
    excluded_dimensions: tuple[str, ...]
    violations: tuple[ActionViolation, ...]

    @property
    def violation_count(self) -> int:
        return len(self.violations)

    @property
    def is_independent(self) -> bool:
        return self.violation_count == 0


@dataclass(frozen=True)
class ActionIndependenceResult:
    """Complete action-independence result for an HDF5 dataset."""

    file_path: Path
    tolerance: float
    demo_count: int
    total_rows: int
    action_dimensions: tuple[str, ...]
    excluded_dimensions: tuple[str, ...]
    demos: tuple[DemoAnalysis, ...]
    total_violations: int

    @property
    def is_independent(self) -> bool:
        return self.total_violations == 0


def _decode_axis_label(raw_label: object, index: int) -> str:
    if isinstance(raw_label, bytes):
        label = raw_label.decode("utf-8")
    else:
        label = str(raw_label)
    return label if label else f"action_{index}"


def _get_action_dimensions(
    actions_dataset: h5py.Dataset,
    demo_name: str,
) -> tuple[str, ...]:
    action_width = actions_dataset.shape[1]
    raw_axes = actions_dataset.attrs.get("axes")

    if raw_axes is None:
        return tuple(f"action_{index}" for index in range(action_width))

    flattened_axes = np.asarray(raw_axes).reshape(-1)
    if len(flattened_axes) != action_width:
        raise ValueError(
            f"data/{demo_name}/actions has width {action_width}, but its "
            f"'axes' attribute contains {len(flattened_axes)} labels."
        )

    return tuple(
        _decode_axis_label(raw_label, index)
        for index, raw_label in enumerate(flattened_axes)
    )


def _load_actions(
    demo_group: h5py.Group,
    demo_name: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if "actions" not in demo_group:
        raise KeyError(f"Missing required dataset: data/{demo_name}/actions")

    actions_dataset = demo_group["actions"]
    if not isinstance(actions_dataset, h5py.Dataset):
        raise TypeError(f"data/{demo_name}/actions is not an HDF5 dataset.")
    if actions_dataset.ndim != 2:
        raise ValueError(
            f"data/{demo_name}/actions must be two-dimensional; "
            f"found shape {actions_dataset.shape}."
        )
    if actions_dataset.shape[1] == 0:
        raise ValueError(f"data/{demo_name}/actions has no action dimensions.")
    if not np.issubdtype(actions_dataset.dtype, np.number):
        raise TypeError(
            f"data/{demo_name}/actions must be numeric; "
            f"found dtype {actions_dataset.dtype}."
        )
    if np.issubdtype(actions_dataset.dtype, np.complexfloating):
        raise TypeError(
            f"data/{demo_name}/actions must contain real values; "
            f"found dtype {actions_dataset.dtype}."
        )

    actions = np.asarray(actions_dataset[...])
    if not np.all(np.isfinite(actions)):
        invalid_rows = np.flatnonzero(~np.all(np.isfinite(actions), axis=1))
        preview = ", ".join(str(int(row)) for row in invalid_rows[:10])
        suffix = "..." if len(invalid_rows) > 10 else ""
        raise ValueError(
            f"data/{demo_name}/actions contains non-finite values in "
            f"row(s) {preview}{suffix}."
        )

    return actions, _get_action_dimensions(actions_dataset, demo_name)


def _analyze_demo(
    demo_group: h5py.Group,
    demo_name: str,
    tolerance: float,
) -> DemoAnalysis:
    actions, action_dimensions = _load_actions(demo_group, demo_name)
    excluded_indices = tuple(
        index
        for index, label in enumerate(action_dimensions)
        if label in EXCLUDED_ACTION_AXES
    )
    included_indices = tuple(
        index
        for index in range(actions.shape[1])
        if index not in excluded_indices
    )

    if not included_indices:
        raise ValueError(
            f"data/{demo_name}/actions has no dimensions left after excluding "
            f"{EXCLUDED_ACTION_AXES}."
        )

    included_actions = actions[:, included_indices]
    active_mask = np.abs(included_actions) > tolerance
    violation_rows = np.flatnonzero(np.count_nonzero(active_mask, axis=1) > 1)

    violations = []
    for row_index in violation_rows:
        active_local_indices = np.flatnonzero(active_mask[row_index])
        active_global_indices = tuple(
            included_indices[int(local_index)]
            for local_index in active_local_indices
        )
        violations.append(
            ActionViolation(
                row_index=int(row_index),
                active_dimensions=tuple(
                    action_dimensions[index] for index in active_global_indices
                ),
                active_values=tuple(
                    float(actions[row_index, index])
                    for index in active_global_indices
                ),
            )
        )

    return DemoAnalysis(
        demo_name=demo_name,
        row_count=int(actions.shape[0]),
        action_dimensions=action_dimensions,
        excluded_dimensions=tuple(
            action_dimensions[index] for index in excluded_indices
        ),
        violations=tuple(violations),
    )


def _print_result(result: ActionIndependenceResult) -> None:
    print("Action Independence Analysis")
    print(f"File: {result.file_path}")
    print(f"Tolerance: {result.tolerance:g}")
    print(
        "Excluded action dimensions: "
        + ", ".join(result.excluded_dimensions)
    )
    print(f"Discovered action dimensions: {', '.join(result.action_dimensions)}")
    print(f"Demos analyzed: {result.demo_count}")
    print(f"Action rows analyzed: {result.total_rows}")
    print()
    print("Per-demo results:")

    for demo in result.demos:
        print(
            f"  {demo.demo_name}: rows={demo.row_count}, "
            f"violations={demo.violation_count}"
        )
        for violation in demo.violations:
            active_entries = ", ".join(
                f"{dimension}={value:.12g}"
                for dimension, value in zip(
                    violation.active_dimensions,
                    violation.active_values,
                )
            )
            print(f"    row {violation.row_index}: {active_entries}")

    print()
    print(f"Global violations: {result.total_violations}")
    if result.is_independent:
        print("Result: PASS - all analyzed movement actions are independent.")
    else:
        print(
            "Result: FAIL - at least one action row activates multiple "
            "movement dimensions."
        )


def analyze_action_independence(
    file_path: Union[str, Path],
    tolerance: float = 0.0,
) -> ActionIndependenceResult:
    """Analyze non-pressure action dimensions within each stored action row."""

    try:
        numeric_tolerance = float(tolerance)
    except (TypeError, ValueError) as error:
        raise TypeError("tolerance must be a finite, non-negative number.") from error
    if not np.isfinite(numeric_tolerance) or numeric_tolerance < 0.0:
        raise ValueError("tolerance must be a finite, non-negative number.")

    resolved_path = Path(file_path).expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"HDF5 file does not exist: {resolved_path}")

    try:
        with h5py.File(resolved_path, "r") as hdf5_file:
            if "data" not in hdf5_file:
                raise KeyError("Missing required HDF5 group: data")
            data_group = hdf5_file["data"]
            if not isinstance(data_group, h5py.Group):
                raise TypeError("The HDF5 object 'data' is not a group.")

            demo_names = sorted(data_group.keys())
            if not demo_names:
                raise ValueError("The HDF5 'data' group contains no demos.")

            demos = tuple(
                _analyze_demo(
                    data_group[demo_name],
                    demo_name,
                    numeric_tolerance,
                )
                for demo_name in demo_names
            )
    except OSError as error:
        raise OSError(f"Could not read HDF5 file '{resolved_path}': {error}") from error

    discovered_dimensions = tuple(
        dict.fromkeys(
            dimension
            for demo in demos
            for dimension in demo.action_dimensions
        )
    )
    result = ActionIndependenceResult(
        file_path=resolved_path,
        tolerance=numeric_tolerance,
        demo_count=len(demos),
        total_rows=sum(demo.row_count for demo in demos),
        action_dimensions=discovered_dimensions,
        excluded_dimensions=EXCLUDED_ACTION_AXES,
        demos=demos,
        total_violations=sum(demo.violation_count for demo in demos),
    )
    _print_result(result)
    return result


def main() -> None:
    analyze_action_independence(HDF5_FILE_PATH)


if __name__ == "__main__":
    main()
