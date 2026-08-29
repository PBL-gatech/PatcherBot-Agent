"""PyQt5 GUI wrapper for experiments.SimpleDatasetBuilder."""

from __future__ import annotations

import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Iterable

from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.SimpleDatasetBuilder import (  # noqa: E402
    ActionSelector,
    AxisToggle,
    FilterSettings,
    GigasealAugmentationSettings,
    ObservationSelector,
    SimpleDatasetBuilder,
)
from experiments.HDF5AnchorConverter import (  # noqa: E402
    AnchorConversionError,
    convert_hdf5,
    derive_metadata_path,
    derive_output_path,
    inspect_hdf5,
    load_calibration,
    load_anchor_sidecar,
)
from experiments.HDF5AnchorEditor import HDF5AnchorDialog  # noqa: E402


class AnchorConversionWorker(QObject):
    """Run a potentially large HDF5 conversion outside the GUI thread."""

    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, source: str, calibration: str, anchors) -> None:
        super().__init__()
        self.source = source
        self.calibration = calibration
        self.anchors = anchors
        self.cancel_requested = False

    def run(self) -> None:
        try:
            output = convert_hdf5(
                self.source,
                self.calibration,
                self.anchors,
                progress=lambda current, total, label: self.progress.emit(current, total, label),
                cancelled=lambda: self.cancel_requested,
            )
        except Exception:
            self.failed.emit(traceback.format_exc())
            return
        self.finished.emit(str(output))


class DatasetBuilderGUI(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DatasetBuilderGUI")
        self.resize(1200, 900)
        self._build_ui()
        self._connect()
        self._update_dataset_name_preview()
        self._sync_cv_generation_controls()
        self._sync_gigaseal_cutoff_controls()
        self._sync_gigaseal_event_window_controls()
        self._sync_break_in_event_window_controls()
        self._sync_gigaseal_augmentation_controls()
        self._sync_selector_constraints()

    def _build_ui(self) -> None:
        outer = QHBoxLayout(self)
        self.page_list = QListWidget()
        self.page_list.setFixedWidth(205)
        self.page_stack = QStackedWidget()
        outer.addWidget(self.page_list)
        outer.addWidget(self.page_stack, 1)

        # The existing builders still create and configure every legacy widget.
        # Their widgets are then reparented into smaller workflow pages below.
        settings_template = self._build_settings_group()
        selector_group = self._build_selector_group()

        workflow_page = QWidget()
        workflow_layout = QVBoxLayout(workflow_page)
        workflow_layout.addWidget(self._page_heading("Choose a Workflow"))
        workflow_box = QGroupBox("Workflow")
        workflow_form = QVBoxLayout(workflow_box)
        self.workflow_build = QRadioButton("Build Dataset")
        self.workflow_anchor = QRadioButton("Anchor Existing HDF5")
        self.workflow_build.setChecked(True)
        self.workflow_group = QButtonGroup(self)
        self.workflow_group.addButton(self.workflow_build)
        self.workflow_group.addButton(self.workflow_anchor)
        workflow_form.addWidget(self.workflow_build)
        workflow_form.addWidget(self.workflow_anchor)
        workflow_layout.addWidget(workflow_box)
        workflow_layout.addStretch(1)

        meta_group = QGroupBox("Metadata")
        meta_row = QHBoxLayout(meta_group)
        self.metadata_path = QLineEdit()
        self.metadata_path.setReadOnly(True)
        self.load_metadata_btn = QPushButton("Load Metadata...")
        meta_row.addWidget(QLabel("File:"))
        meta_row.addWidget(self.metadata_path, 1)
        meta_row.addWidget(self.load_metadata_btn)

        folders_group = QGroupBox("Rig Recorder Folders")
        folders_layout = QVBoxLayout(folders_group)
        root_row = QHBoxLayout()
        self.data_root = QLineEdit(str(REPO_ROOT / "experiments" / "Data" / "rig_recorder_data"))
        self.browse_root_btn = QPushButton("Browse Root...")
        root_row.addWidget(QLabel("Data Root:"))
        root_row.addWidget(self.data_root, 1)
        root_row.addWidget(self.browse_root_btn)
        folders_layout.addLayout(root_row)

        btn_row = QHBoxLayout()
        self.import_folder_btn = QPushButton("Import Folder...")
        self.import_parent_btn = QPushButton("Import Parent...")
        self.add_existing_dirs_btn = QPushButton("Add Date Folders...")
        self.remove_folder_btn = QPushButton("Remove Selected")
        self.clear_folders_btn = QPushButton("Clear")
        btn_row.addWidget(self.import_folder_btn)
        btn_row.addWidget(self.import_parent_btn)
        btn_row.addWidget(self.add_existing_dirs_btn)
        btn_row.addWidget(self.remove_folder_btn)
        btn_row.addWidget(self.clear_folders_btn)
        btn_row.addStretch(1)
        folders_layout.addLayout(btn_row)

        self.folder_list = QListWidget()
        self.folder_list.setSelectionMode(QListWidget.ExtendedSelection)
        folders_layout.addWidget(self.folder_list)
        data_page = self._scroll_page("Data Sources", [meta_group, folders_group])

        output_page = self._form_page(
            "Output",
            [
                ("section", "Output / Split"),
                ("Dataset Name:", self.dataset_name),
                (None, self.append_test_name),
                ("Effective Name:", self.dataset_name_preview),
                ("Validation Ratio:", self.val_ratio),
                ("Random Seed:", self.random_seed),
                (None, self.debug_single_trajectory),
                (None, self.include_failed_demos),
                ("section", "Output Type"),
                (None, self.output_hdf5),
                (None, self.output_images),
                ("Locate End Fraction:", self.locate_cell_end_fraction),
                ("Image Sampling Percentage:", self.image_sample_fraction),
            ],
        )
        images_page = self._form_page(
            "Images",
            [
                ("section", "Camera / Goal Conditioning"),
                ("Image Resize:", self.image_resize),
                (None, self.center_crop),
                (None, self.pipette_dot),
                ("section", "CV Coordinate Generation"),
                (None, self.use_cv_defined_coords),
                (None, self.cv_filter_images),
                (None, self.cv_focus_with_detector_crop),
                (None, self.cv_use_kalman_focus_fusion),
            ],
        )
        sampling_page = self._form_page(
            "Sampling & Events",
            [
                ("section", "Sampling / Cleanup"),
                ("Frequency Mask:", self.freq_mask),
                ("Inaction Steps:", self.inaction),
                ("Inaction Tolerance:", self.inaction_tolerance),
                (None, self.skip_invalid_observations),
                ("section", "Trajectory Representation"),
                (None, self.load_next_obs),
                (None, self.use_velocities),
                (None, self.omit_stage_movement),
                ("section", "Gigaseal Trimming"),
                (None, self.gigaseal_start_trim_enabled),
                (None, self.gigaseal_event_window_enabled),
                ("Gigaseal Event Radius:", self.gigaseal_event_window_radius),
                (None, self.gigaseal_cutoff_enabled),
                ("Gigaseal Resistance Cutoff:", self.gigaseal_cutoff_value),
                ("section", "Break-in Trimming"),
                (None, self.break_in_event_window_enabled),
                ("Break-in Event Radius:", self.break_in_event_window_radius),
                ("section", "Gigaseal Augmentation"),
                (None, self.gigaseal_aug_enabled),
                ("Copies per Demo:", self.gigaseal_aug_copies),
                (None, self.gigaseal_aug_validation),
                ("Pressure Noise Std:", self.gigaseal_aug_pressure_noise),
                ("Pressure Offset Std:", self.gigaseal_aug_pressure_offset),
                ("Resistance Log Noise Std:", self.gigaseal_aug_resistance_log_noise),
                ("Sensor Stutter Probability:", self.gigaseal_aug_stutter_probability),
                ("Sensor Stutter Max Frames:", self.gigaseal_aug_stutter_max),
                ("Prefix Hold Probability:", self.gigaseal_aug_prefix_probability),
                ("Prefix Hold Max Frames:", self.gigaseal_aug_prefix_max),
                ("Timing Action Cap (0 off):", self.gigaseal_aug_counter_cap),
                ("section", "Random Image Filtering"),
                (None, self.enable_filter),
                ("Filter Probability:", self.filter_prob),
                (None, self.filter_train_only),
                (None, self.filter_same_demo),
            ],
        )
        signals_page = self._scroll_page("Signals", [selector_group])

        self.build_btn = QPushButton("Build Dataset")
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.build_review = QTextEdit()
        self.build_review.setReadOnly(True)
        build_review_page = QWidget()
        build_review_layout = QVBoxLayout(build_review_page)
        build_review_layout.addWidget(self._page_heading("Review & Build"))
        build_review_layout.addWidget(QLabel("Effective settings:"))
        build_review_layout.addWidget(self.build_review, 1)
        build_review_layout.addWidget(self.build_btn)
        build_review_layout.addWidget(QLabel("Build log:"))
        build_review_layout.addWidget(self.log, 2)

        hdf5_input_page = self._build_hdf5_input_page()
        anchor_config_page = self._build_anchor_config_page()
        anchor_review_page = self._build_anchor_review_page()

        settings_template.setParent(self)
        settings_template.hide()
        self._pages = {
            "Workflow": workflow_page,
            "Data Sources": data_page,
            "Output": output_page,
            "Images": images_page,
            "Sampling & Events": sampling_page,
            "Signals": signals_page,
            "Review & Build": build_review_page,
            "HDF5 Input": hdf5_input_page,
            "Calibration & Anchors": anchor_config_page,
            "Review & Convert": anchor_review_page,
        }
        for page in self._pages.values():
            self.page_stack.addWidget(page)
        self._refresh_page_navigation()
        self._apply_toggle_help()

    def _page_heading(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet("font-size: 18px; font-weight: 700; margin-bottom: 8px;")
        return label

    def _scroll_page(self, title: str, widgets: list[QWidget]) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.addWidget(self._page_heading(title))
        for widget in widgets:
            layout.addWidget(widget)
        layout.addStretch(1)
        scroll.setWidget(content)
        return scroll

    def _form_page(self, title: str, rows: list[tuple]) -> QScrollArea:
        group = QGroupBox(title)
        form = QFormLayout(group)
        for label, widget in rows:
            if label == "section":
                form.addRow(self._form_section(str(widget)))
            elif label is None:
                form.addRow(widget)
            else:
                form.addRow(label, widget)
        return self._scroll_page(title, [group])

    def _build_hdf5_input_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(self._page_heading("HDF5 Input"))
        source_group = QGroupBox("Existing Dataset")
        source_form = QFormLayout(source_group)
        source_row = QHBoxLayout()
        self.anchor_source_path = QLineEdit()
        self.anchor_source_path.setReadOnly(True)
        self.browse_anchor_source_btn = QPushButton("Load HDF5...")
        source_row.addWidget(self.anchor_source_path, 1)
        source_row.addWidget(self.browse_anchor_source_btn)
        source_form.addRow("Source:", source_row)
        self.anchor_output_path = QLineEdit()
        self.anchor_output_path.setReadOnly(True)
        source_form.addRow("Output:", self.anchor_output_path)
        layout.addWidget(source_group)
        self.anchor_inspection = QTextEdit()
        self.anchor_inspection.setReadOnly(True)
        layout.addWidget(self.anchor_inspection, 1)
        return page

    def _build_anchor_config_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(self._page_heading("Calibration & Anchors"))
        calibration_group = QGroupBox("Calibration")
        calibration_form = QFormLayout(calibration_group)
        calibration_row = QHBoxLayout()
        self.anchor_calibration_path = QLineEdit()
        self.anchor_calibration_path.setReadOnly(True)
        self.browse_anchor_calibration_btn = QPushButton("Load Calibration...")
        calibration_row.addWidget(self.anchor_calibration_path, 1)
        calibration_row.addWidget(self.browse_anchor_calibration_btn)
        calibration_form.addRow("File:", calibration_row)
        layout.addWidget(calibration_group)
        self.edit_anchors_btn = QPushButton("Set / Edit Demonstration Anchors...")
        self.anchor_status = QLabel("Load an HDF5 file to begin.")
        self.anchor_status.setWordWrap(True)
        layout.addWidget(self.edit_anchors_btn)
        layout.addWidget(self.anchor_status)
        layout.addStretch(1)
        return page

    def _build_anchor_review_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(self._page_heading("Review & Convert"))
        self.anchor_review = QTextEdit()
        self.anchor_review.setReadOnly(True)
        self.anchor_progress = QProgressBar()
        self.anchor_progress.setRange(0, 1)
        self.anchor_progress.setValue(0)
        button_row = QHBoxLayout()
        self.convert_anchor_btn = QPushButton("Create Anchored HDF5")
        self.cancel_anchor_btn = QPushButton("Cancel")
        self.cancel_anchor_btn.setEnabled(False)
        button_row.addWidget(self.convert_anchor_btn)
        button_row.addWidget(self.cancel_anchor_btn)
        self.anchor_log = QTextEdit()
        self.anchor_log.setReadOnly(True)
        layout.addWidget(self.anchor_review, 1)
        layout.addWidget(self.anchor_progress)
        layout.addLayout(button_row)
        layout.addWidget(self.anchor_log, 1)
        return page

    def _form_section(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet("font-weight: 600; margin-top: 8px; color: #333;")
        return label

    def _build_settings_group(self) -> QGroupBox:
        g = QGroupBox("Dataset / Build Settings")
        form = QFormLayout(g)

        self.dataset_name = QLineEdit("PatcherBot_dataset_gui.hdf5")
        self.append_test_name = QCheckBox("Use test dataset naming")
        self.append_test_name.setToolTip(
            "If checked, names like 'PatcherBot_dataset_v0_510' become 'PatcherBot_test_dataset_v0_510'."
        )
        self.dataset_name_preview = QLabel("")
        self.dataset_name_preview.setStyleSheet("color: #666;")
        self.val_ratio = QDoubleSpinBox()
        self.val_ratio.setRange(0.0, 1.0)
        self.val_ratio.setDecimals(3)
        self.val_ratio.setSingleStep(0.01)
        self.val_ratio.setValue(0.1)

        self.random_seed = QSpinBox()
        self.random_seed.setRange(0, 999999999)
        self.debug_single_trajectory = QCheckBox("Debug: one random trajectory as full dataset")
        self.debug_single_trajectory.setToolTip(
            "After building demos, keep one random trajectory and copy it to train and validation. "
            "Uses Random Seed; Validation Ratio is ignored."
        )
        self.include_failed_demos = QCheckBox("Include failed demonstrations")
        self.include_failed_demos.setToolTip(
            "When checked, include failed state-recorder attempts while still excluding aborted attempts. "
            "Output demos are tagged and filterable through success/failure HDF5 masks."
        )
        self.freq_mask = QSpinBox()
        self.freq_mask.setRange(1, 1000)
        self.freq_mask.setValue(1)

        self.image_resize = QSpinBox()
        self.image_resize.setRange(1, 4096)
        self.image_resize.setValue(85)

        self.output_hdf5 = QRadioButton("HDF5 dataset")
        self.output_images = QRadioButton("Image folder")
        self.output_hdf5.setChecked(True)
        self.output_mode_group = QButtonGroup(self)
        self.output_mode_group.addButton(self.output_hdf5)
        self.output_mode_group.addButton(self.output_images)
        self.output_images.setToolTip(
            "Create only an images folder containing sampled, unmodified camera files."
        )
        self.locate_cell_end_fraction = QDoubleSpinBox()
        self.locate_cell_end_fraction.setRange(0.001, 1.0)
        self.locate_cell_end_fraction.setDecimals(3)
        self.locate_cell_end_fraction.setValue(0.1)
        self.image_sample_fraction = QDoubleSpinBox()
        self.image_sample_fraction.setRange(0.0, 1.0)
        self.image_sample_fraction.setDecimals(3)
        self.image_sample_fraction.setValue(0.1)
        self.image_sample_fraction.setToolTip(
            "Randomly copy this fraction without replacement: from the final locate-cell "
            "window or from the complete hunt-cell attempt."
        )

        self.inaction = QSpinBox()
        self.inaction.setRange(0, 100000)
        self.inaction_tolerance = QDoubleSpinBox()
        self.inaction_tolerance.setRange(0, 1000000)
        self.inaction_tolerance.setDecimals(4)
        self.inaction_tolerance.setValue(0.0)
        self.skip_invalid_observations = QCheckBox("Skip timesteps with NaN/Inf in selected data")
        self.skip_invalid_observations.setChecked(True)
        self.skip_invalid_observations.setToolTip(
            "Drop any timestep whose selected observation or action payload contains NaN or Inf values."
        )
        self.gigaseal_start_trim_enabled = QCheckBox("Gigaseal: start at first near--5 mbar command")
        self.gigaseal_start_trim_enabled.setChecked(False)
        self.gigaseal_start_trim_enabled.setToolTip(
            "When enabled, discard leading gigaseal samples before the first aligned near--5 mbar command."
        )
        self.gigaseal_cutoff_enabled = QCheckBox("Gigaseal: stop at resistance cutoff")
        self.gigaseal_cutoff_enabled.setToolTip(
            "When enabled, gigaseal trajectories stop once resistance reaches the configured cutoff."
        )
        self.gigaseal_cutoff_value = QDoubleSpinBox()
        self.gigaseal_cutoff_value.setRange(0.0, 1000000.0)
        self.gigaseal_cutoff_value.setDecimals(3)
        self.gigaseal_cutoff_value.setSingleStep(10.0)
        self.gigaseal_cutoff_value.setValue(1200.0)

        self.gigaseal_event_window_enabled = QCheckBox("Gigaseal: window before first action event")
        self.gigaseal_event_window_enabled.setChecked(True)
        self.gigaseal_event_window_enabled.setToolTip(
            "When enabled, gigaseal trajectories keep rows from before the first event in any action dimension."
        )
        self.gigaseal_event_window_radius = QSpinBox()
        self.gigaseal_event_window_radius.setRange(5, 100000)
        self.gigaseal_event_window_radius.setValue(15)
        self.gigaseal_event_window_radius.setToolTip(
            "Rows to keep before the first gigaseal action event."
        )

        self.break_in_event_window_enabled = QCheckBox("Break-in: window around action events")
        self.break_in_event_window_enabled.setChecked(True)
        self.break_in_event_window_enabled.setToolTip(
            "When enabled, break-in trajectories keep rows from before the first ATM/zap event through after the last."
        )
        self.break_in_event_window_radius = QSpinBox()
        self.break_in_event_window_radius.setRange(5, 100000)
        self.break_in_event_window_radius.setValue(15)
        self.break_in_event_window_radius.setToolTip(
            "Rows to keep before the first break-in action event and after the last."
        )

        self.gigaseal_aug_enabled = QCheckBox("Gigaseal: enable copied-demo augmentation")
        self.gigaseal_aug_enabled.setChecked(False)
        self.gigaseal_aug_enabled.setToolTip(
            "When enabled, add conservative copied Gigaseal training demos without changing originals."
        )
        self.gigaseal_aug_copies = QSpinBox()
        self.gigaseal_aug_copies.setRange(1, 20)
        self.gigaseal_aug_copies.setValue(1)
        self.gigaseal_aug_validation = QCheckBox("Augment validation demos")
        self.gigaseal_aug_validation.setChecked(False)
        self.gigaseal_aug_pressure_noise = QDoubleSpinBox()
        self.gigaseal_aug_pressure_noise.setRange(0.0, 100.0)
        self.gigaseal_aug_pressure_noise.setDecimals(4)
        self.gigaseal_aug_pressure_noise.setSingleStep(0.01)
        self.gigaseal_aug_pressure_noise.setValue(0.05)
        self.gigaseal_aug_pressure_offset = QDoubleSpinBox()
        self.gigaseal_aug_pressure_offset.setRange(0.0, 100.0)
        self.gigaseal_aug_pressure_offset.setDecimals(4)
        self.gigaseal_aug_pressure_offset.setSingleStep(0.01)
        self.gigaseal_aug_pressure_offset.setValue(0.1)
        self.gigaseal_aug_resistance_log_noise = QDoubleSpinBox()
        self.gigaseal_aug_resistance_log_noise.setRange(0.0, 1.0)
        self.gigaseal_aug_resistance_log_noise.setDecimals(4)
        self.gigaseal_aug_resistance_log_noise.setSingleStep(0.001)
        self.gigaseal_aug_resistance_log_noise.setValue(0.01)
        self.gigaseal_aug_stutter_probability = QDoubleSpinBox()
        self.gigaseal_aug_stutter_probability.setRange(0.0, 1.0)
        self.gigaseal_aug_stutter_probability.setDecimals(4)
        self.gigaseal_aug_stutter_probability.setSingleStep(0.01)
        self.gigaseal_aug_stutter_probability.setValue(0.02)
        self.gigaseal_aug_stutter_max = QSpinBox()
        self.gigaseal_aug_stutter_max.setRange(1, 3)
        self.gigaseal_aug_stutter_max.setValue(3)
        self.gigaseal_aug_prefix_probability = QDoubleSpinBox()
        self.gigaseal_aug_prefix_probability.setRange(0.0, 1.0)
        self.gigaseal_aug_prefix_probability.setDecimals(4)
        self.gigaseal_aug_prefix_probability.setSingleStep(0.01)
        self.gigaseal_aug_prefix_probability.setValue(0.15)
        self.gigaseal_aug_prefix_max = QSpinBox()
        self.gigaseal_aug_prefix_max.setRange(1, 10)
        self.gigaseal_aug_prefix_max.setValue(3)
        self.gigaseal_aug_counter_cap = QSpinBox()
        self.gigaseal_aug_counter_cap.setRange(0, 100000)
        self.gigaseal_aug_counter_cap.setValue(0)

        self.obs_resistance_slope = QCheckBox("Resistance Slope")
        self.obs_resistance_slope.setChecked(False)
        self.obs_resistance_slope.setToolTip("Compute windowed numerical average slope of resistance per sample.")
        self.slope_window_spin = QSpinBox()
        self.slope_window_spin.setRange(5, 50)
        self.slope_window_spin.setValue(20)
        self.slope_window_spin.setEnabled(False)
        self.slope_window_spin.setToolTip("Window size for resistance slope computation (5–50).")

        self.resistance_input_window_spin = QSpinBox()
        self.resistance_input_window_spin.setRange(1, 500)
        self.resistance_input_window_spin.setValue(15)
        self.resistance_input_window_spin.setEnabled(False)
        self.resistance_input_window_spin.setToolTip(
            "Number of preceding resistance samples to provide as obs/resistance_input."
        )

        self.load_next_obs = QCheckBox("Load next observations")
        self.use_velocities = QCheckBox("Use velocities")
        self.use_velocities.setToolTip(
            "Convert deltas to per-observation velocities using forward/backward Euler."
        )
        self.omit_stage_movement = QCheckBox("Omit stage movement attempts")
        self.center_crop = QCheckBox("Center crop camera")
        self.pipette_dot = QCheckBox("Add final pipette dot")
        self.pipette_dot.setToolTip(
            "Goal-conditioning option: draw the final pipette position into camera frames."
        )
        self.use_cv_defined_coords = QCheckBox(
            "Use CV-defined coordinates (generate cv_movement_recording.csv)"
        )
        self.use_cv_defined_coords.setToolTip(
            "When enabled, the GUI runs ImageDatasetPreparer before building demos."
        )
        self.cv_filter_images = QCheckBox(
            "CV generation: filter frames to successful-attempt windows"
        )
        self.cv_filter_images.setChecked(True)
        self.cv_filter_images.setToolTip(
            "If unchecked, CV generation uses all frames in each camera_frames folder."
        )
        self.cv_focus_with_detector_crop = QCheckBox(
            "CV generation: crop focus input around detected pipette"
        )
        self.cv_focus_with_detector_crop.setChecked(True)
        self.cv_focus_with_detector_crop.setToolTip(
            "If checked, pipette focus inference runs on a detector-centered crop instead of the full frame."
        )
        self.cv_use_kalman_focus_fusion = QCheckBox(
            "CV generation: Kalman fuse encoder Z + focuser Z"
        )
        self.cv_use_kalman_focus_fusion.setChecked(False)
        self.cv_use_kalman_focus_fusion.setToolTip(
            "Encoder Z dominates motion while focuser Z gradually realigns focal-plane reference."
        )

        self.enable_filter = QCheckBox("Enable random image filter")
        self.filter_prob = QDoubleSpinBox()
        self.filter_prob.setRange(0.0, 1.0)
        self.filter_prob.setDecimals(3)
        self.filter_prob.setSingleStep(0.05)
        self.filter_prob.setValue(0.65)
        self.filter_train_only = QCheckBox("Filter train split only")
        self.filter_same_demo = QCheckBox("Same filter per demo")

        form.addRow(self._form_section("Output / Split"))
        form.addRow("Dataset Name:", self.dataset_name)
        form.addRow(self.append_test_name)
        form.addRow("Effective Name:", self.dataset_name_preview)
        form.addRow("Validation Ratio:", self.val_ratio)
        form.addRow("Random Seed:", self.random_seed)
        form.addRow(self.debug_single_trajectory)
        form.addRow(self.include_failed_demos)

        form.addRow(self._form_section("Sampling / Cleanup"))
        form.addRow("Frequency Mask:", self.freq_mask)
        form.addRow("Inaction Steps:", self.inaction)
        form.addRow("Inaction Tolerance:", self.inaction_tolerance)
        form.addRow(self.skip_invalid_observations)

        form.addRow(self._form_section("Gigaseal Trimming"))
        form.addRow(self.gigaseal_start_trim_enabled)
        form.addRow(self.gigaseal_event_window_enabled)
        form.addRow("Gigaseal Event Radius:", self.gigaseal_event_window_radius)
        form.addRow(self.gigaseal_cutoff_enabled)
        form.addRow("Gigaseal Resistance Cutoff:", self.gigaseal_cutoff_value)

        form.addRow(self._form_section("Break-in Trimming"))
        form.addRow(self.break_in_event_window_enabled)
        form.addRow("Break-in Event Radius:", self.break_in_event_window_radius)

        form.addRow(self._form_section("Gigaseal Augmentation"))
        form.addRow(self.gigaseal_aug_enabled)
        form.addRow("Copies per Demo:", self.gigaseal_aug_copies)
        form.addRow(self.gigaseal_aug_validation)
        form.addRow("Pressure Noise Std:", self.gigaseal_aug_pressure_noise)
        form.addRow("Pressure Offset Std:", self.gigaseal_aug_pressure_offset)
        form.addRow("Resistance Log Noise Std:", self.gigaseal_aug_resistance_log_noise)
        form.addRow("Sensor Stutter Probability:", self.gigaseal_aug_stutter_probability)
        form.addRow("Sensor Stutter Max Frames:", self.gigaseal_aug_stutter_max)
        form.addRow("Prefix Hold Probability:", self.gigaseal_aug_prefix_probability)
        form.addRow("Prefix Hold Max Frames:", self.gigaseal_aug_prefix_max)
        form.addRow("Timing Action Cap (0 off):", self.gigaseal_aug_counter_cap)

        form.addRow(self._form_section("Trajectory Representation"))
        form.addRow(self.load_next_obs)
        form.addRow(self.use_velocities)
        form.addRow(self.omit_stage_movement)

        form.addRow(self._form_section("Camera / Goal Conditioning"))
        form.addRow("Image Resize:", self.image_resize)
        form.addRow(self.center_crop)
        form.addRow(self.pipette_dot)

        form.addRow(self._form_section("Output Type"))
        form.addRow(self.output_hdf5)
        form.addRow(self.output_images)
        form.addRow("Locate End Fraction:", self.locate_cell_end_fraction)
        form.addRow("Image Sampling Percentage:", self.image_sample_fraction)

        form.addRow(self._form_section("CV Coordinate Generation"))
        form.addRow(self.use_cv_defined_coords)
        form.addRow(self.cv_filter_images)
        form.addRow(self.cv_focus_with_detector_crop)
        form.addRow(self.cv_use_kalman_focus_fusion)

        form.addRow(self._form_section("Random Image Filtering"))
        form.addRow(self.enable_filter)
        form.addRow("Filter Probability:", self.filter_prob)
        form.addRow(self.filter_train_only)
        form.addRow(self.filter_same_demo)
        return g

    def _build_selector_group(self) -> QGroupBox:
        g = QGroupBox("Observation / Action Selection")
        root = QVBoxLayout(g)

        obs_box = QGroupBox("Observations")
        obs_grid = QGridLayout(obs_box)

        self.obs_pressure = QCheckBox("Pressure")
        self.obs_pressure.setChecked(True)
        self.obs_resistance = QCheckBox("Resistance")
        self.obs_resistance.setChecked(True)
        self.obs_current = QCheckBox("Current")
        self.obs_voltage = QCheckBox("Voltage")
        self.obs_stage = QCheckBox("Stage Position")
        self.obs_stage.setChecked(True)
        self.obs_pipette = QCheckBox("Pipette Position")
        self.obs_pipette.setChecked(True)
        self.use_synthetic_position_anchor = QCheckBox("Synthetic Zero Position Anchor")
        self.use_synthetic_position_anchor.setToolTip(
            "Subtract each demonstration's first stage and pipette positions from all subsequent coordinates."
        )
        self.obs_camera = QCheckBox("Camera Image")
        self.obs_camera.setChecked(True)
        self.obs_gigaseal_pressure_state = QCheckBox("Gigaseal/Break-in Pressure State")
        self.obs_gigaseal_pressure_state.setChecked(True)
        self.obs_gigaseal_pressure_state.setToolTip(
            "For gigaseal and break-in demos, add obs/pressure_atm_state from pressure/ATM command logs."
        )
        self.obs_gigaseal_effective_pressure = QCheckBox("Gigaseal/Break-in Effective Pressure")
        self.obs_gigaseal_effective_pressure.setChecked(True)
        self.obs_gigaseal_effective_pressure.setToolTip(
            "For gigaseal and break-in demos, add obs/effective_pressure as commanded pressure unless ATM is active."
        )
        self.obs_gigaseal_time_since_action = QCheckBox("Gigaseal/Break-in Time Since Last Action")
        self.obs_gigaseal_time_since_action.setChecked(True)
        self.obs_gigaseal_time_since_action.setToolTip(
            "For gigaseal and break-in demos, add obs/time_since_last_action in seconds after filtering."
        )
        self.obs_gigaseal_resistance_input = QCheckBox("Gigaseal/Break-in Resistance Input")
        self.obs_gigaseal_resistance_input.setChecked(False)
        self.obs_gigaseal_resistance_input.setToolTip(
            "For gigaseal and break-in demos, add obs/resistance_input as a zero-padded window of prior resistance samples."
        )

        self.obs_stage_x = QCheckBox("Stage X")
        self.obs_stage_x.setChecked(True)
        self.obs_stage_y = QCheckBox("Stage Y")
        self.obs_stage_y.setChecked(True)
        self.obs_stage_z = QCheckBox("Stage Z")
        self.obs_stage_z.setChecked(True)
        self.obs_pip_x = QCheckBox("Pipette X")
        self.obs_pip_x.setChecked(True)
        self.obs_pip_y = QCheckBox("Pipette Y")
        self.obs_pip_y.setChecked(True)
        self.obs_pip_z = QCheckBox("Pipette Z")
        self.obs_pip_z.setChecked(True)

        obs_grid.addWidget(self.obs_pressure, 0, 0)
        obs_grid.addWidget(self.obs_resistance, 0, 1)
        obs_grid.addWidget(self.obs_resistance_slope, 0, 2)
        obs_grid.addWidget(self.obs_current, 0, 3)
        obs_grid.addWidget(self.obs_voltage, 1, 0)
        obs_grid.addWidget(self.obs_stage, 1, 1)
        obs_grid.addWidget(self.obs_pipette, 1, 2)
        obs_grid.addWidget(self.obs_camera, 1, 3)
        obs_grid.addWidget(self.use_synthetic_position_anchor, 2, 0, 1, 2)
        obs_grid.addWidget(self.obs_gigaseal_pressure_state, 3, 0, 1, 2)
        obs_grid.addWidget(self.obs_gigaseal_effective_pressure, 3, 2, 1, 2)
        obs_grid.addWidget(self.obs_gigaseal_time_since_action, 4, 0, 1, 2)
        obs_grid.addWidget(self.obs_gigaseal_resistance_input, 5, 0, 1, 2)
        obs_grid.addWidget(QLabel("Resistance Window:"), 5, 2)
        obs_grid.addWidget(self.resistance_input_window_spin, 5, 3)
        obs_grid.addWidget(QLabel("Slope Window:"), 6, 0)
        obs_grid.addWidget(self.slope_window_spin, 6, 1)

        obs_grid.addWidget(QLabel("Stage Axes:"), 7, 0)
        obs_grid.addWidget(self.obs_stage_x, 7, 1)
        obs_grid.addWidget(self.obs_stage_y, 7, 2)
        obs_grid.addWidget(self.obs_stage_z, 7, 3)

        obs_grid.addWidget(QLabel("Pipette Axes:"), 8, 0)
        obs_grid.addWidget(self.obs_pip_x, 8, 1)
        obs_grid.addWidget(self.obs_pip_y, 8, 2)
        obs_grid.addWidget(self.obs_pip_z, 8, 3)

        act_box = QGroupBox("Actions")
        act_grid = QGridLayout(act_box)

        self.act_stage = QCheckBox("Stage Action")
        self.act_pipette = QCheckBox("Pipette Action")
        self.act_pipette.setChecked(True)
        self.act_pressure = QCheckBox("Pressure Action")
        self.act_pressure.setToolTip(
            "Uses target commanded pressure. Gigaseal writes pressure + ATM unless applied-pressure mode is enabled; break-in writes ATM and optional zap."
        )
        self.act_gigaseal_combined_pressure = QCheckBox("Gigaseal Applied Pressure Action")
        self.act_gigaseal_combined_pressure.setToolTip(
            "For gigaseal demos, replace pressure + ATM actions with one applied_pressure_mbar action; ATM maps to 0."
        )
        self.act_gigaseal_binary_pressure = QCheckBox("Gigaseal Binary Pressure Commands")
        self.act_gigaseal_binary_pressure.setChecked(False)
        self.act_gigaseal_binary_pressure.setToolTip(
            "Writes three one-hot command columns for -5, 5, and reset."
        )
        self.act_break_in_zap = QCheckBox("Break-in Zap Action")
        self.act_break_in_zap.setChecked(True)
        self.act_break_in_zap.setToolTip(
            "Adds the sparse zap command as a break-in action dimension. Disable to train break-in on ATM switching only."
        )
        self.act_observations_until_next_action = QCheckBox(
            "Gigaseal/Break-in Observations Until Next Action"
        )
        self.act_observations_until_next_action.setChecked(True)
        self.act_observations_until_next_action.setToolTip(
            "Adds observations_until_next_action as an action dimension computed after filtering."
        )

        self.act_stage_x = QCheckBox("Stage X")
        self.act_stage_x.setChecked(True)
        self.act_stage_y = QCheckBox("Stage Y")
        self.act_stage_y.setChecked(True)
        self.act_stage_z = QCheckBox("Stage Z")
        self.act_stage_z.setChecked(True)
        self.act_pip_x = QCheckBox("Pipette X")
        self.act_pip_x.setChecked(True)
        self.act_pip_y = QCheckBox("Pipette Y")
        self.act_pip_y.setChecked(True)
        self.act_pip_z = QCheckBox("Pipette Z")
        self.act_pip_z.setChecked(True)

        act_grid.addWidget(self.act_stage, 0, 0)
        act_grid.addWidget(self.act_pipette, 0, 1)
        act_grid.addWidget(self.act_pressure, 0, 2)
        act_grid.addWidget(self.act_break_in_zap, 0, 3)
        act_grid.addWidget(self.act_gigaseal_combined_pressure, 1, 0, 1, 3)
        act_grid.addWidget(self.act_observations_until_next_action, 1, 3)
        act_grid.addWidget(self.act_gigaseal_binary_pressure, 2, 0, 1, 3)

        act_grid.addWidget(QLabel("Stage Axes:"), 3, 0)
        act_grid.addWidget(self.act_stage_x, 3, 1)
        act_grid.addWidget(self.act_stage_y, 3, 2)
        act_grid.addWidget(self.act_stage_z, 3, 3)

        act_grid.addWidget(QLabel("Pipette Axes:"), 4, 0)
        act_grid.addWidget(self.act_pip_x, 4, 1)
        act_grid.addWidget(self.act_pip_y, 4, 2)
        act_grid.addWidget(self.act_pip_z, 4, 3)

        root.addWidget(obs_box)
        root.addWidget(act_box)
        return g

    def _refresh_page_navigation(self) -> None:
        item = self.page_list.currentItem()
        current = item.data(Qt.UserRole) if item is not None else None
        if self.workflow_anchor.isChecked():
            names = ["Workflow", "HDF5 Input", "Calibration & Anchors", "Review & Convert"]
        else:
            names = ["Workflow", "Data Sources", "Output", "Images", "Sampling & Events", "Signals", "Review & Build"]
        self.page_list.blockSignals(True)
        self.page_list.clear()
        for name in names:
            page_item = QListWidgetItem(name)
            page_item.setData(Qt.UserRole, name)
            self.page_list.addItem(page_item)
        selected = names.index(current) if current in names else 0
        self.page_list.setCurrentRow(selected)
        self.page_list.blockSignals(False)
        self._show_page(selected)

    def _show_page(self, row: int) -> None:
        item = self.page_list.item(row)
        if item is None:
            return
        name = item.data(Qt.UserRole)
        self.page_stack.setCurrentWidget(self._pages[name])
        if name == "Review & Build":
            self._update_build_review()
        elif name == "Review & Convert":
            self._update_anchor_review()

    def _select_page(self, name: str) -> None:
        for row in range(self.page_list.count()):
            if self.page_list.item(row).data(Qt.UserRole) == name:
                self.page_list.setCurrentRow(row)
                return

    def _update_build_review(self) -> None:
        folders = [str(value) for value in self._selected_folders() if isinstance(value, str)]
        lines = [
            f"Dataset: {self._effective_dataset_name(self.dataset_name.text())}",
            f"Output: {'Image folder' if self.output_images.isChecked() else 'HDF5'}",
            f"Folders: {len(folders)}",
        ]
        lines.extend(f"  • {folder}" for folder in folders)
        lines.extend([
            f"Validation ratio: {self.val_ratio.value():g}",
            f"Image size: {self.image_resize.value()} × {self.image_resize.value()}",
            f"CV coordinates: {'enabled' if self.use_cv_defined_coords.isChecked() else 'disabled'}",
        ])
        self.build_review.setPlainText("\n".join(lines))

    def _apply_toggle_help(self) -> None:
        definitions = {
            "workflow_build": "Build a new dataset from selected raw rig-recorder folders. All existing sampling and signal settings apply.",
            "workflow_anchor": "Load a completed HDF5, label retained pipette-tip frames, and create a separate _anchored HDF5.",
            "output_hdf5": "Write demonstrations, observations, actions, masks, and metadata to HDF5 files.",
            "gigaseal_aug_validation": "Also create augmented copies in the validation split; normally augmentation is training-only.",
            "load_next_obs": "Write next_obs datasets by shifting each selected observation one row forward for goal-conditioned training.",
            "omit_stage_movement": "Exclude attempts containing nonzero stage movement from the built dataset.",
            "center_crop": "Center-crop each camera frame to half its width and height before resizing it for the dataset.",
            "enable_filter": "Apply the configured random image augmentation pipeline while loading dataset camera frames.",
            "filter_train_only": "Apply random image filters only to training demonstrations and leave validation images unchanged.",
            "filter_same_demo": "Replay one sampled image filter consistently across every frame in a demonstration.",
            "obs_pressure": "Include measured pressure as an observation value.",
            "obs_resistance": "Include measured seal resistance as an observation value.",
            "obs_current": "Include the recorded current waveform as an observation.",
            "obs_voltage": "Include the recorded voltage waveform as an observation.",
            "obs_stage": "Include the selected stage position axes as observations.",
            "obs_pipette": "Include the selected pipette position axes as observations.",
            "use_synthetic_position_anchor": "Subtract each demonstration's first stage and pipette positions from every coordinate row so each trajectory starts at zero.",
            "obs_camera": "Include resized camera images in each demonstration's observations.",
            "act_stage": "Include movement deltas or velocities for the selected stage axes in actions.",
            "act_pipette": "Include movement deltas or velocities for the selected pipette axes in actions.",
        }
        named_widgets = {id(value): name for name, value in vars(self).items() if isinstance(value, (QCheckBox, QRadioButton))}
        toggles = self.findChildren(QCheckBox) + self.findChildren(QRadioButton)
        for widget in toggles:
            name = named_widgets.get(id(widget), "")
            behavior = widget.toolTip().strip() or definitions.get(name, "")
            if not behavior and ("stage_" in name or "pip_" in name):
                kind = "observation" if name.startswith("obs_") else "action"
                behavior = f"Include {widget.text()} in the {kind}."
            if not behavior:
                behavior = f"Enable or disable the {widget.text()} dataset setting."

            if name.startswith("obs_"):
                affected = "The named observation dataset in every generated demonstration."
            elif name.startswith("act_") or name == "use_velocities":
                affected = "The actions dataset in every generated demonstration."
            elif name.startswith("workflow_"):
                affected = "The available Dataset Builder pages and active workflow only."
            elif name.startswith("output_"):
                affected = "The generated dataset artifact and its companion metadata."
            elif name.startswith("cv_") or name == "use_cv_defined_coords":
                affected = "Generated CV movement coordinates and, where stated, source camera images."
            elif name.startswith("filter_") or name == "enable_filter":
                affected = "Camera observations selected for random image augmentation."
            elif name.startswith("gigaseal_aug"):
                affected = "Synthetic gigaseal demonstrations and their selected signals."
            else:
                affected = "Raw-build sampling, trimming, image, split, or naming behavior as described."

            if "pressure" in name:
                units = "Pressure values are mbar; boolean state or command columns are dimensionless."
            elif "resistance" in name:
                units = "Resistance values use the source dataset's resistance units."
            elif "stage_" in name or "pip_" in name or name in {"obs_stage", "obs_pipette", "act_stage", "act_pipette"}:
                units = "Raw builds keep source units; anchored X/Y is pixels and anchored Z remains µm."
            elif name in {"obs_camera", "center_crop", "pipette_dot"} or name.startswith("cv_") or name.startswith("filter_") or name == "enable_filter":
                units = "Image geometry is measured in stored-image pixels; image samples are otherwise unitless."
            else:
                units = "This toggle is dimensionless; numeric companion controls retain their displayed units."

            dependency = "No additional toggle dependency."
            if name.startswith("obs_stage_"):
                dependency = "Requires Stage Position observation to be enabled."
            elif name.startswith("obs_pip_"):
                dependency = "Requires Pipette Position observation to be enabled."
            elif name.startswith("act_stage_"):
                dependency = "Requires Stage Movement action to be enabled."
            elif name.startswith("act_pip_"):
                dependency = "Requires Pipette Movement action to be enabled."
            elif name.startswith("cv_"):
                dependency = "Requires CV-defined coordinates to be enabled."
            elif name.startswith("filter_"):
                dependency = "Requires random image filtering to be enabled."
            elif name.startswith("gigaseal_aug_") and name != "gigaseal_aug_enabled":
                dependency = "Requires gigaseal augmentation to be enabled."
            elif name == "gigaseal_cutoff_enabled":
                dependency = "Enables the gigaseal resistance-cutoff value control."
            elif name == "gigaseal_event_window_enabled":
                dependency = "Enables the gigaseal event-window radius control."
            elif name == "break_in_event_window_enabled":
                dependency = "Enables the break-in event-window radius control."
            elif name == "debug_single_trajectory":
                dependency = "Uses Random Seed and ignores Validation Ratio after selecting one demonstration."
            elif name == "use_synthetic_position_anchor":
                dependency = "Applies only to enabled Stage Position and Pipette Position observations."

            exclusion = "Can be combined with other compatible settings."
            if name in {"workflow_build", "workflow_anchor"}:
                exclusion = "Mutually exclusive with the other workflow choice."
            elif name in {"output_hdf5", "output_images"}:
                exclusion = "Mutually exclusive with the other output type."
            elif name in {"act_gigaseal_combined_pressure", "act_gigaseal_binary_pressure"}:
                exclusion = "Mutually exclusive with the other gigaseal pressure representation."

            definition = (
                f"Behavior: {behavior}\n"
                f"Affected data: {affected}\n"
                f"Units: {units}\n"
                f"Dependencies: {dependency}\n"
                f"Mutual exclusions: {exclusion}"
            )
            widget.setToolTip(definition)
            widget.setAccessibleDescription(definition)
            widget.setStatusTip(definition)

    def browse_anchor_source(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Existing HDF5", str(REPO_ROOT / "experiments" / "Datasets"), "HDF5 Files (*.hdf5 *.h5)")
        if path:
            self.anchor_source_path.setText(str(Path(path).resolve()))
            self._inspect_anchor_source()

    def _inspect_anchor_source(self) -> None:
        source = self.anchor_source_path.text().strip()
        if not source:
            self.anchor_inspection.clear()
            self.anchor_output_path.clear()
            return
        inspection = inspect_hdf5(source)
        try:
            self.anchor_output_path.setText(str(derive_output_path(source)))
        except AnchorConversionError:
            self.anchor_output_path.clear()
        lines = [
            f"Demonstrations: {len(inspection.demo_keys)}",
            f"Splits: {dict(inspection.split_counts)}",
            f"Observation keys: {', '.join(inspection.observation_keys) or 'none'}",
            f"Camera shape: {inspection.camera_shape}",
            f"Pipette axes: {inspection.pipette_axes}",
            f"Stage axes: {inspection.stage_axes}",
            f"Action axes: {inspection.action_axes}",
            f"Action representation: {inspection.action_representation or 'unknown'}",
            f"Next observations: {'yes' if inspection.has_next_obs else 'no'}",
            f"Already anchored: {'yes' if inspection.anchored else 'no'}",
        ]
        if inspection.errors:
            lines.append("\nValidation errors:")
            lines.extend(f"  • {error}" for error in inspection.errors)
        else:
            lines.append("\nSource schema is ready for anchor selection.")
        self.anchor_inspection.setPlainText("\n".join(lines))
        self._refresh_anchor_status()

    def browse_anchor_calibration(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Calibration", str(REPO_ROOT / "experiments" / "Data" / "Calibration_data"), "Calibration Files (*.json *.pickle *.pkl);;All Files (*)")
        if path:
            self.anchor_calibration_path.setText(str(Path(path).resolve()))
            self._refresh_anchor_status()

    def edit_hdf5_anchors(self) -> None:
        source = self.anchor_source_path.text().strip()
        if not source:
            self._select_page("HDF5 Input")
            QMessageBox.warning(self, "HDF5 input", "Load a valid HDF5 file first.")
            return
        try:
            HDF5AnchorDialog(source, self).exec_()
        except Exception as exc:
            QMessageBox.critical(self, "Anchor editor", str(exc))
        self._refresh_anchor_status()

    def _refresh_anchor_status(self) -> None:
        source = self.anchor_source_path.text().strip()
        if not source:
            self.anchor_status.setText("Load an HDF5 file to begin.")
            return
        inspection = inspect_hdf5(source)
        try:
            anchors = load_anchor_sidecar(source)
            anchor_error = ""
        except (AnchorConversionError, OSError) as exc:
            anchors, anchor_error = {}, str(exc)
        complete = sum(1 for key in inspection.demo_keys if key in anchors)
        status = f"Anchors: {complete}/{len(inspection.demo_keys)} complete."
        if anchor_error:
            status += f" Sidecar error: {anchor_error}"
        if inspection.errors:
            status += " Source validation must be resolved."
        if not self.anchor_calibration_path.text().strip():
            status += " Select a calibration file."
        self.anchor_status.setText(status)
        self._update_anchor_review()

    def _update_anchor_review(self) -> None:
        source = self.anchor_source_path.text().strip()
        calibration = self.anchor_calibration_path.text().strip()
        if not source:
            self.anchor_review.setPlainText("Load an HDF5 file on the HDF5 Input page.")
            return
        inspection = inspect_hdf5(source)
        try:
            anchors, anchor_error = load_anchor_sidecar(source), None
        except (AnchorConversionError, OSError) as exc:
            anchors, anchor_error = {}, str(exc)
        missing = [key for key in inspection.demo_keys if key not in anchors]
        complete = sum(1 for key in inspection.demo_keys if key in anchors)
        lines = [f"Source: {source}", f"Output: {self.anchor_output_path.text().strip()}", f"Calibration: {calibration or 'not selected'}", f"Anchors: {complete}/{len(inspection.demo_keys)}"]
        if missing:
            lines.append(f"Missing: {', '.join(missing[:12])}{' ...' if len(missing) > 12 else ''}")
        if anchor_error:
            lines.append(f"Sidecar error: {anchor_error}")
        if inspection.errors:
            lines.append("Validation errors:")
            lines.extend(f"  • {error}" for error in inspection.errors)
        self.anchor_review.setPlainText("\n".join(lines))

    def start_anchor_conversion(self) -> None:
        source = self.anchor_source_path.text().strip()
        calibration = self.anchor_calibration_path.text().strip()
        if not source:
            self._select_page("HDF5 Input")
            QMessageBox.warning(self, "HDF5 input", "Load an HDF5 file first.")
            return
        inspection = inspect_hdf5(source)
        if inspection.errors:
            self._select_page("HDF5 Input")
            QMessageBox.warning(self, "Invalid HDF5", "\n".join(inspection.errors))
            return
        if not calibration:
            self._select_page("Calibration & Anchors")
            QMessageBox.warning(self, "Calibration", "Select a calibration file.")
            return
        try:
            output_path = derive_output_path(source)
            metadata_path = derive_metadata_path(source)
            load_calibration(calibration)
        except AnchorConversionError as exc:
            self._select_page("Calibration & Anchors")
            QMessageBox.warning(self, "Calibration", str(exc))
            return
        existing_targets = [path for path in (output_path, metadata_path) if path.exists()]
        if existing_targets:
            self._select_page("HDF5 Input")
            QMessageBox.warning(
                self,
                "Anchored output exists",
                "Conversion will not overwrite an existing target:\n"
                + "\n".join(str(path) for path in existing_targets),
            )
            return
        try:
            anchors = load_anchor_sidecar(source)
        except (AnchorConversionError, OSError) as exc:
            self._select_page("Calibration & Anchors")
            QMessageBox.warning(self, "Anchors", str(exc))
            return
        missing = [key for key in inspection.demo_keys if key not in anchors]
        if missing:
            self._select_page("Calibration & Anchors")
            QMessageBox.warning(self, "Anchors", f"Missing anchors for {len(missing)} demonstrations.")
            return
        self._set_conversion_busy(True)
        self.anchor_log.append("Starting safe HDF5 copy and anchor conversion...")
        self._anchor_thread = QThread(self)
        self._anchor_worker = AnchorConversionWorker(source, calibration, anchors)
        self._anchor_worker.moveToThread(self._anchor_thread)
        self._anchor_thread.started.connect(self._anchor_worker.run)
        self._anchor_worker.progress.connect(self._on_anchor_progress)
        self._anchor_worker.finished.connect(self._on_anchor_finished)
        self._anchor_worker.failed.connect(self._on_anchor_failed)
        self._anchor_worker.finished.connect(self._anchor_thread.quit)
        self._anchor_worker.failed.connect(self._anchor_thread.quit)
        self._anchor_thread.finished.connect(self._anchor_worker.deleteLater)
        self._anchor_thread.finished.connect(self._anchor_thread.deleteLater)
        self._anchor_thread.start()

    def cancel_anchor_conversion(self) -> None:
        worker = getattr(self, "_anchor_worker", None)
        if worker is not None:
            worker.cancel_requested = True
            self.cancel_anchor_btn.setEnabled(False)
            self.anchor_log.append("Cancellation requested; the current safe step will finish first.")

    def _set_conversion_busy(self, busy: bool) -> None:
        self.page_list.setEnabled(not busy)
        self.workflow_build.setEnabled(not busy)
        self.workflow_anchor.setEnabled(not busy)
        self.convert_anchor_btn.setEnabled(not busy)
        self.cancel_anchor_btn.setEnabled(busy)

    def _on_anchor_progress(self, current: int, total: int, label: str) -> None:
        self.anchor_progress.setRange(0, max(1, total))
        self.anchor_progress.setValue(current)
        self.anchor_progress.setFormat(f"{label} — {current}/{total}")

    def _on_anchor_finished(self, output: str) -> None:
        self._set_conversion_busy(False)
        self.anchor_log.append(f"Anchored dataset created: {output}")
        QMessageBox.information(self, "Anchoring complete", f"Created:\n{output}")

    def _on_anchor_failed(self, details: str) -> None:
        self._set_conversion_busy(False)
        self.anchor_log.append(details)
        message = details.strip().splitlines()[-1] if details.strip() else "Unknown conversion error"
        display_message = message.split(": ", 1)[-1]
        lower_message = display_message.lower()
        if "cancelled" in lower_message:
            QMessageBox.information(self, "Anchoring cancelled", "No anchored output was published.")
            return
        if any(token in lower_message for token in ("calibration", ".m", "matrix", "anchor")):
            self._select_page("Calibration & Anchors")
        elif any(token in lower_message for token in ("axis", "obs/", "actions", "hdf5")):
            self._select_page("HDF5 Input")
        QMessageBox.critical(self, "Anchoring failed", display_message)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        thread = getattr(self, "_anchor_thread", None)
        if thread is not None and thread.isRunning():
            self.cancel_anchor_conversion()
            QMessageBox.warning(
                self,
                "Conversion running",
                "Cancellation was requested. Wait for the current safe step to finish "
                "before closing the Dataset Builder.",
            )
            event.ignore()
            return
        super().closeEvent(event)

    def _connect(self) -> None:
        self.load_metadata_btn.clicked.connect(self.load_metadata)
        self.browse_root_btn.clicked.connect(self.browse_root)
        self.import_folder_btn.clicked.connect(self.import_folder)
        self.import_parent_btn.clicked.connect(self.import_parent)
        self.add_existing_dirs_btn.clicked.connect(self.add_existing_date_folders)
        self.remove_folder_btn.clicked.connect(self.remove_selected)
        self.clear_folders_btn.clicked.connect(self.folder_list.clear)
        self.build_btn.clicked.connect(self.build_dataset)
        self.page_list.currentRowChanged.connect(self._show_page)
        self.workflow_build.toggled.connect(self._refresh_page_navigation)
        self.workflow_anchor.toggled.connect(self._refresh_page_navigation)
        self.browse_anchor_source_btn.clicked.connect(self.browse_anchor_source)
        self.browse_anchor_calibration_btn.clicked.connect(self.browse_anchor_calibration)
        self.edit_anchors_btn.clicked.connect(self.edit_hdf5_anchors)
        self.convert_anchor_btn.clicked.connect(self.start_anchor_conversion)
        self.cancel_anchor_btn.clicked.connect(self.cancel_anchor_conversion)

        self.dataset_name.textChanged.connect(self._update_dataset_name_preview)
        self.append_test_name.toggled.connect(self._update_dataset_name_preview)
        self.use_cv_defined_coords.toggled.connect(self._sync_cv_generation_controls)
        self.gigaseal_cutoff_enabled.toggled.connect(self._sync_gigaseal_cutoff_controls)
        self.gigaseal_event_window_enabled.toggled.connect(self._sync_gigaseal_event_window_controls)
        self.break_in_event_window_enabled.toggled.connect(self._sync_break_in_event_window_controls)
        self.gigaseal_event_window_radius.valueChanged.connect(self._sync_event_window_bounds)
        self.break_in_event_window_radius.valueChanged.connect(self._sync_event_window_bounds)
        self.gigaseal_aug_enabled.toggled.connect(self._sync_gigaseal_augmentation_controls)

        self.obs_resistance.toggled.connect(self._sync_selector_constraints)
        self.obs_resistance_slope.toggled.connect(self._sync_selector_constraints)
        self.obs_gigaseal_resistance_input.toggled.connect(self._sync_selector_constraints)
        self.act_stage.toggled.connect(self._sync_selector_constraints)
        self.act_pipette.toggled.connect(self._sync_selector_constraints)
        self.act_pressure.toggled.connect(self._sync_selector_constraints)
        self.act_gigaseal_binary_pressure.toggled.connect(self._sync_selector_constraints)
        self.obs_stage.toggled.connect(self._sync_selector_constraints)
        self.obs_pipette.toggled.connect(self._sync_selector_constraints)
        self.obs_stage_x.toggled.connect(self._sync_selector_constraints)
        self.obs_stage_y.toggled.connect(self._sync_selector_constraints)
        self.obs_stage_z.toggled.connect(self._sync_selector_constraints)
        self.obs_pip_x.toggled.connect(self._sync_selector_constraints)
        self.obs_pip_y.toggled.connect(self._sync_selector_constraints)
        self.obs_pip_z.toggled.connect(self._sync_selector_constraints)
        self.act_stage_x.toggled.connect(self._sync_selector_constraints)
        self.act_stage_y.toggled.connect(self._sync_selector_constraints)
        self.act_stage_z.toggled.connect(self._sync_selector_constraints)
        self.act_pip_x.toggled.connect(self._sync_selector_constraints)
        self.act_pip_y.toggled.connect(self._sync_selector_constraints)
        self.act_pip_z.toggled.connect(self._sync_selector_constraints)

    def _sync_cv_generation_controls(self) -> None:
        enabled = self.use_cv_defined_coords.isChecked()
        self.cv_filter_images.setEnabled(enabled)
        self.cv_focus_with_detector_crop.setEnabled(enabled)
        self.cv_use_kalman_focus_fusion.setEnabled(enabled)

    def _sync_gigaseal_cutoff_controls(self) -> None:
        self.gigaseal_cutoff_value.setEnabled(self.gigaseal_cutoff_enabled.isChecked())

    def _sync_gigaseal_event_window_controls(self) -> None:
        self.gigaseal_event_window_radius.setEnabled(
            self.gigaseal_event_window_enabled.isChecked()
        )
        self._sync_event_window_bounds()

    def _sync_break_in_event_window_controls(self) -> None:
        self.break_in_event_window_radius.setEnabled(
            self.break_in_event_window_enabled.isChecked()
        )
        self._sync_event_window_bounds()

    def _event_window_radius_limit(self) -> int | None:
        radii = []
        if self.gigaseal_event_window_enabled.isChecked():
            radii.append(int(self.gigaseal_event_window_radius.value()))
        if self.break_in_event_window_enabled.isChecked():
            radii.append(int(self.break_in_event_window_radius.value()))
        return min(radii) if radii else None

    def _sync_event_window_bounds(self, *_args) -> None:
        limit = self._event_window_radius_limit()
        slope_max = 50 if limit is None else min(50, max(5, int(limit)))
        resistance_max = 500 if limit is None else min(500, max(1, int(limit)))
        self.slope_window_spin.setMaximum(slope_max)
        self.resistance_input_window_spin.setMaximum(resistance_max)

    def _sync_gigaseal_augmentation_controls(self) -> None:
        enabled = self.gigaseal_aug_enabled.isChecked()
        for control in (
            self.gigaseal_aug_copies,
            self.gigaseal_aug_validation,
            self.gigaseal_aug_pressure_noise,
            self.gigaseal_aug_pressure_offset,
            self.gigaseal_aug_resistance_log_noise,
            self.gigaseal_aug_stutter_probability,
            self.gigaseal_aug_stutter_max,
            self.gigaseal_aug_prefix_probability,
            self.gigaseal_aug_prefix_max,
            self.gigaseal_aug_counter_cap,
        ):
            control.setEnabled(enabled)

    def _with_blocked_signals(self, *widgets):
        class _Blocker:
            def __init__(self, controls):
                self.controls = controls
                self.states = []

            def __enter__(self):
                for control in self.controls:
                    self.states.append(control.blockSignals(True))
                return self

            def __exit__(self, exc_type, exc, tb):
                for control, state in zip(self.controls, self.states):
                    control.blockSignals(state)
                return False

        return _Blocker(widgets)

    def _link_include(self, obs_cb: QCheckBox, act_cb: QCheckBox) -> None:
        if obs_cb.isChecked() and not act_cb.isChecked():
            with self._with_blocked_signals(act_cb):
                act_cb.setChecked(True)

    def _link_axis(self, obs_cb: QCheckBox, act_cb: QCheckBox) -> None:
        if obs_cb.isChecked() and not act_cb.isChecked():
            with self._with_blocked_signals(act_cb):
                act_cb.setChecked(True)

    def _sync_selector_constraints(self) -> None:
        binary_gigaseal_pressure = self.act_gigaseal_binary_pressure.isChecked()
        if binary_gigaseal_pressure and self.act_gigaseal_combined_pressure.isChecked():
            with self._with_blocked_signals(self.act_gigaseal_combined_pressure):
                self.act_gigaseal_combined_pressure.setChecked(False)
        self.act_gigaseal_combined_pressure.setEnabled(not binary_gigaseal_pressure)

        self._link_include(self.obs_stage, self.act_stage)
        self._link_include(self.obs_pipette, self.act_pipette)

        self._link_axis(self.obs_stage_x, self.act_stage_x)
        self._link_axis(self.obs_stage_y, self.act_stage_y)
        self._link_axis(self.obs_stage_z, self.act_stage_z)
        self._link_axis(self.obs_pip_x, self.act_pip_x)
        self._link_axis(self.obs_pip_y, self.act_pip_y)
        self._link_axis(self.obs_pip_z, self.act_pip_z)

        for control in (self.obs_stage_x, self.obs_stage_y, self.obs_stage_z):
            control.setEnabled(self.obs_stage.isChecked())
        for control in (self.obs_pip_x, self.obs_pip_y, self.obs_pip_z):
            control.setEnabled(self.obs_pipette.isChecked())
        self.obs_resistance_slope.setEnabled(self.obs_resistance.isChecked())
        self.slope_window_spin.setEnabled(
            self.obs_resistance.isChecked() and self.obs_resistance_slope.isChecked()
        )
        self.resistance_input_window_spin.setEnabled(
            self.obs_gigaseal_resistance_input.isChecked()
        )
        for control in (self.act_stage_x, self.act_stage_y, self.act_stage_z):
            control.setEnabled(self.act_stage.isChecked())
        for control in (self.act_pip_x, self.act_pip_y, self.act_pip_z):
            control.setEnabled(self.act_pipette.isChecked())
    def _append(self, text: str) -> None:
        self.log.append(text)

    def _effective_dataset_name(self, raw_name: str) -> str:
        name = raw_name.strip()
        if not name:
            return ""
        if not name.lower().endswith((".h5", ".hdf5")):
            name = f"{name}.hdf5"

        if self.append_test_name.isChecked():
            stem = Path(name).stem
            suffix = Path(name).suffix
            if "_test_dataset_" in stem:
                return f"{stem}{suffix}"
            if "_dataset_" in stem:
                stem = stem.replace("_dataset_", "_test_dataset_", 1)
            elif not stem.endswith("_test"):
                stem = f"{stem}_test"
            name = f"{stem}{suffix}"
        return name

    def _update_dataset_name_preview(self) -> None:
        effective = self._effective_dataset_name(self.dataset_name.text())
        self.dataset_name_preview.setText(effective or "(empty)")

    def browse_root(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select rig_recorder_data root", self.data_root.text())
        if d:
            self.data_root.setText(d)

    def import_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select rig-recorder folder", self.data_root.text())
        if d:
            self._import_one(Path(d), ask_on_conflict=True)

    def import_parent(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select parent folder", self.data_root.text())
        if not d:
            return
        parent = Path(d)
        candidates = [p for p in sorted(parent.iterdir()) if p.is_dir() and self._looks_like_demo(p)]
        if not candidates:
            QMessageBox.information(self, "No folders", "No child folders with movement CSV files found.")
            return
        count = 0
        for p in candidates:
            if self._import_one(p, ask_on_conflict=False):
                count += 1
        self._append(f"Imported {count} folder(s) from {parent}")

    def add_existing_date_folders(self) -> None:
        root = Path(self.data_root.text().strip())
        if not root.is_dir():
            QMessageBox.warning(self, "Data root", f"Not a directory:\n{root}")
            return

        candidates = [p for p in sorted(root.iterdir()) if p.is_dir() and self._looks_like_demo(p)]
        if not candidates:
            QMessageBox.information(
                self,
                "No date folders",
                "No date folders containing movement CSV files were found under the current data root.",
            )
            return

        selected = self._pick_multiple_directories(candidates)
        if not selected:
            return

        count = 0
        for directory in selected:
            if self._add_folder(directory.name, directory):
                count += 1
        self._append(f"Added {count} existing folder(s) from data root.")

    def _pick_multiple_directories(self, directories: Iterable[Path]) -> list[Path]:
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Date Folders")
        dialog.resize(640, 520)
        layout = QVBoxLayout(dialog)

        info = QLabel(
            "Select one or more date directories to add to the build list. "
            "You can click multiple rows directly."
        )
        layout.addWidget(info)

        list_widget = QListWidget()
        list_widget.setSelectionMode(QAbstractItemView.MultiSelection)
        for directory in directories:
            item = QListWidgetItem(directory.name)
            item.setData(Qt.UserRole, str(directory))
            item.setToolTip(str(directory))
            list_widget.addItem(item)
        layout.addWidget(list_widget, 1)

        utility_row = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        clear_btn = QPushButton("Clear")
        utility_row.addWidget(select_all_btn)
        utility_row.addWidget(clear_btn)
        utility_row.addStretch(1)
        layout.addLayout(utility_row)

        select_all_btn.clicked.connect(list_widget.selectAll)
        clear_btn.clicked.connect(list_widget.clearSelection)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        if dialog.exec_() != QDialog.Accepted:
            return []

        selected_paths: list[Path] = []
        for item in list_widget.selectedItems():
            raw = item.data(Qt.UserRole)
            if isinstance(raw, str):
                selected_paths.append(Path(raw))
        return selected_paths

    def _import_one(self, src: Path, ask_on_conflict: bool) -> bool:
        if not self._looks_like_demo(src):
            QMessageBox.warning(self, "Invalid folder", f"Missing movement CSV in:\n{src}")
            return False

        root = Path(self.data_root.text().strip())
        root.mkdir(parents=True, exist_ok=True)
        dst = root / src.name

        if src.resolve() != dst.resolve():
            if dst.exists():
                if ask_on_conflict:
                    ans = QMessageBox.question(
                        self,
                        "Folder exists",
                        f"{dst} exists. Reuse existing folder?",
                        QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                        QMessageBox.Yes,
                    )
                    if ans == QMessageBox.Cancel:
                        return False
                    if ans == QMessageBox.No:
                        shutil.rmtree(dst)
                        shutil.copytree(src, dst)
                else:
                    pass
            else:
                shutil.copytree(src, dst)

        return self._add_folder(dst.name, dst)

    def _looks_like_demo(self, p: Path) -> bool:
        return (p / "movement_recording.csv").exists() or (p / "cv_movement_recording.csv").exists()

    def _add_folder(self, name: str, path: Path) -> bool:
        normalized_path = str(path.resolve())
        existing = {self.folder_list.item(i).data(Qt.UserRole) for i in range(self.folder_list.count())}
        if normalized_path in existing:
            return False
        item = QListWidgetItem(name)
        item.setData(Qt.UserRole, normalized_path)
        item.setToolTip(normalized_path)
        self.folder_list.addItem(item)
        self._append(f"Added folder: {name}")
        return True

    def remove_selected(self) -> None:
        for item in self.folder_list.selectedItems():
            self.folder_list.takeItem(self.folder_list.row(item))

    def load_metadata(self) -> None:
        p, _ = QFileDialog.getOpenFileName(
            self, "Select metadata.json", str(REPO_ROOT / "experiments" / "Datasets"), "JSON (*.json)"
        )
        if not p:
            return
        try:
            payload = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception as exc:
            QMessageBox.critical(self, "Metadata error", str(exc))
            return

        self.metadata_path.setText(p)
        settings = payload.get("settings", {}) if isinstance(payload, dict) else {}
        selectors = payload.get("selectors", {}) if isinstance(payload, dict) else {}
        loaded_name = settings.get("dataset_name") or payload.get("dataset_name")
        self._set_if(self.dataset_name, loaded_name)
        if isinstance(loaded_name, str) and loaded_name.strip():
            stem = Path(loaded_name).stem.lower()
            is_test_name = "_test_dataset_" in stem or stem.endswith("_test")
            self.append_test_name.setChecked(is_test_name)
        self._set_if(self.val_ratio, settings.get("val_ratio"))
        self._set_if(self.random_seed, settings.get("random_seed"))
        self._set_if(self.debug_single_trajectory, settings.get("debug_single_trajectory"))
        self._set_if(self.include_failed_demos, settings.get("include_failed_demos"))
        self._set_if(self.freq_mask, settings.get("freq_mask", settings.get("frequency_mod")))
        self._set_if(self.image_resize, settings.get("image_resize"))
        output_mode = settings.get("output_mode")
        if output_mode == "images" or (
            output_mode is None and settings.get("export_full_images") is True
        ):
            self.output_images.setChecked(True)
        elif output_mode == "hdf5":
            self.output_hdf5.setChecked(True)
        self._set_if(self.locate_cell_end_fraction, settings.get("locate_cell_end_fraction"))
        self._set_if(
            self.image_sample_fraction,
            settings.get(
                "image_sample_fraction",
                settings.get("hunt_cell_sample_probability"),
            ),
        )
        self._set_if(self.inaction, settings.get("inaction"))
        self._set_if(self.inaction_tolerance, settings.get("inaction_tolerance"))
        self._set_if(self.skip_invalid_observations, settings.get("skip_invalid_observations"))
        self._set_if(self.gigaseal_start_trim_enabled, settings.get("gigaseal_start_trim_enabled"))
        self._set_if(self.gigaseal_event_window_enabled, settings.get("gigaseal_event_window_enabled"))
        self._set_if(self.gigaseal_event_window_radius, settings.get("gigaseal_event_window_radius"))
        self._set_if(self.gigaseal_cutoff_enabled, settings.get("gigaseal_resistance_cutoff_enabled"))
        self._set_if(self.gigaseal_cutoff_value, settings.get("gigaseal_resistance_cutoff"))
        self._set_if(self.break_in_event_window_enabled, settings.get("break_in_event_window_enabled"))
        self._set_if(self.break_in_event_window_radius, settings.get("break_in_event_window_radius"))
        aug_cfg = settings.get("gigaseal_augmentation", {})
        if isinstance(aug_cfg, dict):
            self._set_if(self.gigaseal_aug_enabled, aug_cfg.get("enabled"))
            self._set_if(self.gigaseal_aug_copies, aug_cfg.get("copies_per_demo"))
            self._set_if(self.gigaseal_aug_validation, aug_cfg.get("augment_validation"))
            self._set_if(self.gigaseal_aug_pressure_noise, aug_cfg.get("pressure_noise_std"))
            self._set_if(self.gigaseal_aug_pressure_offset, aug_cfg.get("pressure_offset_std"))
            self._set_if(
                self.gigaseal_aug_resistance_log_noise,
                aug_cfg.get("resistance_log_noise_std"),
            )
            self._set_if(
                self.gigaseal_aug_stutter_probability,
                aug_cfg.get("sensor_stutter_probability"),
            )
            self._set_if(self.gigaseal_aug_stutter_max, aug_cfg.get("sensor_stutter_max_frames"))
            self._set_if(
                self.gigaseal_aug_prefix_probability,
                aug_cfg.get("prefix_hold_probability"),
            )
            self._set_if(self.gigaseal_aug_prefix_max, aug_cfg.get("prefix_hold_max_frames"))
            self._set_if(
                self.gigaseal_aug_counter_cap,
                aug_cfg.get(
                    "observations_until_next_action_cap",
                    aug_cfg.get("observations_since_last_action_cap"),
                ),
            )
        self._set_if(self.slope_window_spin, settings.get("resistance_slope_window"))
        self._set_if(self.resistance_input_window_spin, settings.get("resistance_input_window"))
        self._set_if(self.load_next_obs, settings.get("load_next_obs"))
        self._set_if(self.use_velocities, settings.get("use_velocities"))
        self._set_if(self.use_cv_defined_coords, settings.get("prefer_cv_movement"))
        self._set_if(self.cv_filter_images, settings.get("cv_filter_images"))
        self._set_if(self.cv_focus_with_detector_crop, settings.get("cv_focus_with_detector_crop"))
        self._set_if(
            self.cv_use_kalman_focus_fusion,
            settings.get("cv_use_kalman_focus_fusion"),
        )
        self._set_if(self.omit_stage_movement, settings.get("omit_stage_movement"))
        self._set_if(self.center_crop, settings.get("center_crop"))
        self._set_if(self.pipette_dot, settings.get("pipette_final_pos_color_dot"))

        fcfg = settings.get("filter", {})
        if isinstance(fcfg, dict):
            self._set_if(self.enable_filter, fcfg.get("enable_random_filter"))
            self._set_if(self.filter_prob, fcfg.get("image_filter_prob"))
            self._set_if(self.filter_train_only, fcfg.get("filter_train_only"))
            self._set_if(self.filter_same_demo, fcfg.get("filter_same_per_demo"))

        ocfg = settings.get("observation_selector", selectors.get("observations", {}))
        acfg = settings.get("action_selector", selectors.get("actions", {}))
        if isinstance(ocfg, dict):
            self._set_if(self.obs_pressure, ocfg.get("include_pressure"))
            self._set_if(self.obs_resistance, ocfg.get("include_resistance"))
            self._set_if(self.obs_resistance_slope, ocfg.get("include_resistance_slope"))
            self._set_if(self.obs_current, ocfg.get("include_current"))
            self._set_if(self.obs_voltage, ocfg.get("include_voltage"))
            self._set_if(self.obs_stage, ocfg.get("include_stage"))
            self._set_if(self.obs_pipette, ocfg.get("include_pipette"))
            self._set_if(self.obs_camera, ocfg.get("include_camera"))
            self._set_if(
                self.obs_gigaseal_pressure_state,
                ocfg.get("include_gigaseal_pressure_state"),
            )
            self._set_if(
                self.obs_gigaseal_effective_pressure,
                ocfg.get("include_gigaseal_effective_pressure"),
            )
            self._set_if(
                self.obs_gigaseal_time_since_action,
                ocfg.get("include_gigaseal_time_since_last_action"),
            )
            self._set_if(
                self.obs_gigaseal_resistance_input,
                ocfg.get("include_gigaseal_resistance_input"),
            )
            self._set_axis(self.obs_stage_x, self.obs_stage_y, self.obs_stage_z, ocfg.get("stage_axes"), "stage")
            self._set_axis(self.obs_pip_x, self.obs_pip_y, self.obs_pip_z, ocfg.get("pipette_axes"), "pipette")
        self._set_if(
            self.use_synthetic_position_anchor,
            settings.get("use_synthetic_position_anchor"),
        )
        if isinstance(acfg, dict):
            self._set_if(self.act_stage, acfg.get("include_stage"))
            self._set_if(self.act_pipette, acfg.get("include_pipette"))
            self._set_if(self.act_pressure, acfg.get("include_pressure"))
            self._set_if(
                self.act_gigaseal_binary_pressure,
                acfg.get("binary_gigaseal_pressure_action"),
            )
            self._set_if(
                self.act_gigaseal_combined_pressure,
                acfg.get("combine_gigaseal_pressure_action"),
            )
            self._set_if(self.act_break_in_zap, acfg.get("include_break_in_zap"))
            timing_enabled = acfg.get("include_observations_until_next_action")
            if timing_enabled is None and isinstance(ocfg, dict):
                timing_enabled = ocfg.get("include_gigaseal_observations_since_last_action")
            self._set_if(self.act_observations_until_next_action, timing_enabled)
            self._set_axis(self.act_stage_x, self.act_stage_y, self.act_stage_z, acfg.get("stage_axes"), "stage")
            self._set_axis(self.act_pip_x, self.act_pip_y, self.act_pip_z, acfg.get("pipette_axes"), "pipette")

        folders = payload.get("processed_folders", [])
        if isinstance(folders, list) and folders:
            if QMessageBox.question(
                self, "Load folders", "Load processed_folders into the list?", QMessageBox.Yes | QMessageBox.No
            ) == QMessageBox.Yes:
                root = Path(self.data_root.text().strip())
                for f in folders:
                    if isinstance(f, str) and (root / f).is_dir():
                        self._add_folder(f, root / f)
        self._update_dataset_name_preview()
        self._sync_cv_generation_controls()
        self._sync_gigaseal_cutoff_controls()
        self._sync_gigaseal_event_window_controls()
        self._sync_break_in_event_window_controls()
        self._sync_gigaseal_augmentation_controls()
        self._sync_selector_constraints()
        self._append(f"Loaded metadata: {p}")

    def _set_if(self, widget, value) -> None:
        if isinstance(widget, QLineEdit):
            if isinstance(value, str) and value.strip():
                widget.setText(value.strip())
            return
        if isinstance(widget, QCheckBox):
            if isinstance(value, bool):
                widget.setChecked(value)
            return
        if value is None:
            return
        try:
            widget.setValue(value)
        except Exception:
            pass

    def _set_axis(self, xcb, ycb, zcb, axis_value, prefix: str) -> None:
        if isinstance(axis_value, dict):
            xcb.setChecked(bool(axis_value.get("x", False)))
            ycb.setChecked(bool(axis_value.get("y", False)))
            zcb.setChecked(bool(axis_value.get("z", False)))
            return
        if isinstance(axis_value, list):
            vals = {str(v).lower() for v in axis_value}
            xcb.setChecked("x" in vals or f"{prefix}_x" in vals)
            ycb.setChecked("y" in vals or f"{prefix}_y" in vals)
            zcb.setChecked("z" in vals or f"{prefix}_z" in vals)

    def _selected_folders(self):
        return [self.folder_list.item(i).data(Qt.UserRole) for i in range(self.folder_list.count())]

    def _collect_kwargs(self):
        name = self._effective_dataset_name(self.dataset_name.text())
        event_window_limit = self._event_window_radius_limit()
        resistance_slope_window = int(self.slope_window_spin.value())
        resistance_input_window = int(self.resistance_input_window_spin.value())
        binary_gigaseal_pressure = self.act_gigaseal_binary_pressure.isChecked()
        if event_window_limit is not None:
            resistance_slope_window = min(resistance_slope_window, int(event_window_limit))
            resistance_input_window = min(resistance_input_window, int(event_window_limit))
        return dict(
            dataset_name=name,
            val_ratio=float(self.val_ratio.value()),
            omit_stage_movement=self.omit_stage_movement.isChecked(),
            random_seed=int(self.random_seed.value()),
            debug_single_trajectory=self.debug_single_trajectory.isChecked(),
            include_failed_demos=self.include_failed_demos.isChecked(),
            freq_mask=int(self.freq_mask.value()),
            load_next_obs=self.load_next_obs.isChecked(),
            use_velocities=self.use_velocities.isChecked(),
            prefer_cv_movement=self.use_cv_defined_coords.isChecked(),
            use_synthetic_position_anchor=self.use_synthetic_position_anchor.isChecked(),
            filter=FilterSettings(
                enable_random_filter=self.enable_filter.isChecked(),
                image_filter_prob=float(self.filter_prob.value()),
                filter_train_only=self.filter_train_only.isChecked(),
                filter_same_per_demo=self.filter_same_demo.isChecked(),
            ),
            gigaseal_augmentation=GigasealAugmentationSettings(
                enabled=self.gigaseal_aug_enabled.isChecked(),
                copies_per_demo=int(self.gigaseal_aug_copies.value()),
                augment_validation=self.gigaseal_aug_validation.isChecked(),
                pressure_noise_std=float(self.gigaseal_aug_pressure_noise.value()),
                pressure_offset_std=float(self.gigaseal_aug_pressure_offset.value()),
                resistance_log_noise_std=float(self.gigaseal_aug_resistance_log_noise.value()),
                sensor_stutter_probability=float(self.gigaseal_aug_stutter_probability.value()),
                sensor_stutter_max_frames=int(self.gigaseal_aug_stutter_max.value()),
                prefix_hold_probability=float(self.gigaseal_aug_prefix_probability.value()),
                prefix_hold_max_frames=int(self.gigaseal_aug_prefix_max.value()),
                observations_until_next_action_cap=int(self.gigaseal_aug_counter_cap.value()),
            ),
            image_resize=int(self.image_resize.value()),
            output_mode="images" if self.output_images.isChecked() else "hdf5",
            locate_cell_end_fraction=float(self.locate_cell_end_fraction.value()),
            image_sample_fraction=float(self.image_sample_fraction.value()),
            pipette_final_pos_color_dot=self.pipette_dot.isChecked(),
            center_crop=self.center_crop.isChecked(),
            inaction=int(self.inaction.value()),
            inaction_tolerance=float(self.inaction_tolerance.value()),
            skip_invalid_observations=self.skip_invalid_observations.isChecked(),
            gigaseal_start_trim_enabled=self.gigaseal_start_trim_enabled.isChecked(),
            gigaseal_event_window_enabled=self.gigaseal_event_window_enabled.isChecked(),
            gigaseal_event_window_radius=int(self.gigaseal_event_window_radius.value()),
            gigaseal_resistance_cutoff_enabled=self.gigaseal_cutoff_enabled.isChecked(),
            gigaseal_resistance_cutoff=float(self.gigaseal_cutoff_value.value()),
            break_in_event_window_enabled=self.break_in_event_window_enabled.isChecked(),
            break_in_event_window_radius=int(self.break_in_event_window_radius.value()),
            resistance_slope_window=resistance_slope_window,
            resistance_input_window=resistance_input_window,
            observation_selector=ObservationSelector(
                include_pressure=self.obs_pressure.isChecked(),
                include_resistance=self.obs_resistance.isChecked(),
                include_resistance_slope=self.obs_resistance_slope.isChecked(),
                include_current=self.obs_current.isChecked(),
                include_voltage=self.obs_voltage.isChecked(),
                include_stage=self.obs_stage.isChecked(),
                include_pipette=self.obs_pipette.isChecked(),
                include_camera=self.obs_camera.isChecked(),
                include_gigaseal_pressure_state=self.obs_gigaseal_pressure_state.isChecked(),
                include_gigaseal_effective_pressure=self.obs_gigaseal_effective_pressure.isChecked(),
                include_gigaseal_resistance_input=self.obs_gigaseal_resistance_input.isChecked(),
                include_gigaseal_observations_since_last_action=False,
                include_gigaseal_time_since_last_action=self.obs_gigaseal_time_since_action.isChecked(),
                stage_axes=AxisToggle(self.obs_stage_x.isChecked(), self.obs_stage_y.isChecked(), self.obs_stage_z.isChecked()),
                pipette_axes=AxisToggle(self.obs_pip_x.isChecked(), self.obs_pip_y.isChecked(), self.obs_pip_z.isChecked()),
            ),
            action_selector=ActionSelector(
                include_stage=self.act_stage.isChecked(),
                include_pipette=self.act_pipette.isChecked(),
                include_pressure=self.act_pressure.isChecked(),
                combine_gigaseal_pressure_action=(
                    self.act_gigaseal_combined_pressure.isChecked()
                    and not binary_gigaseal_pressure
                ),
                binary_gigaseal_pressure_action=binary_gigaseal_pressure,
                include_break_in_zap=self.act_break_in_zap.isChecked(),
                include_observations_until_next_action=(
                    self.act_observations_until_next_action.isChecked()
                ),
                include_high_level=False,
                stage_axes=AxisToggle(self.act_stage_x.isChecked(), self.act_stage_y.isChecked(), self.act_stage_z.isChecked()),
                pipette_axes=AxisToggle(self.act_pip_x.isChecked(), self.act_pip_y.isChecked(), self.act_pip_z.isChecked()),
            ),
        )

    def _generate_cv_movement_files(self, folders: list[str]) -> None:
        from experiments.ImageDatasetPreparer import ImageDatasetPreparer

        rig_root = Path(self.data_root.text().strip())
        if not rig_root.is_dir():
            raise FileNotFoundError(f"Rig data root not found: {rig_root}")
        filter_images = self.cv_filter_images.isChecked()
        focus_with_detector_crop = self.cv_focus_with_detector_crop.isChecked()
        use_kalman_focus_fusion = self.cv_use_kalman_focus_fusion.isChecked()
        preparer = ImageDatasetPreparer(
            rig_root,
            use_detector1=False,
            filter_images=filter_images,
            focus_with_detector_crop=focus_with_detector_crop,
            use_kalman_focus_fusion=use_kalman_focus_fusion,
        )

        self._append(
            "Generating cv_movement_recording.csv files "
            f"(filter_images={'on' if filter_images else 'off'}, "
            f"focus_crop={'on' if focus_with_detector_crop else 'off'}, "
            f"kalman={'on' if use_kalman_focus_fusion else 'off'})..."
        )
        for i, folder in enumerate(folders, 1):
            self._append(f"[cv {i}/{len(folders)}] {folder}")
            QApplication.processEvents()
            preparer.build_csv(folder, output_name="cv_movement_recording.csv")
        self._append("CV coordinate generation complete.")

    def build_dataset(self) -> None:
        folders = [f for f in self._selected_folders() if isinstance(f, str)]
        if not folders:
            self._select_page("Data Sources")
            QMessageBox.warning(self, "No folders", "Import at least one folder.")
            return
        if not self.dataset_name.text().strip():
            self._select_page("Output")
            QMessageBox.warning(self, "Dataset name", "Dataset name is required.")
            return

        try:
            kwargs = self._collect_kwargs()
        except Exception as exc:
            self._append(str(exc))
            self._select_page("Sampling & Events")
            QMessageBox.warning(self, "Builder settings", str(exc))
            return
        self._append(f"Building dataset: {kwargs['dataset_name']}")
        self._append(f"Folders: {', '.join(folders)}")
        if self.debug_single_trajectory.isChecked():
            self._append(
                "Debug single-trajectory mode: final HDF5 outputs will contain one compact demo "
                "referenced by both train and valid masks."
            )

        self.setEnabled(False)
        try:
            if self.use_cv_defined_coords.isChecked():
                self._generate_cv_movement_files(folders)

            image_output = kwargs["output_mode"] == "images"
            builder = SimpleDatasetBuilder(**kwargs)
            processed_count = 0
            skipped_folders: list[tuple[str, str]] = []
            for i, folder in enumerate(folders, 1):
                self._append(f"[{i}/{len(folders)}] {folder}")
                QApplication.processEvents()
                try:
                    builder.add_demo(folder, record_to_file=not image_output)
                except FileNotFoundError as exc:
                    reason = str(exc)
                    if not reason.startswith(
                        ("Missing movement recording", "Missing graph recording")
                    ):
                        raise
                    skipped_folders.append((folder, reason))
                    self._append(f"  skipped: {reason}")
                    continue
                processed_count += 1

            if skipped_folders:
                self._append(
                    f"Skipped {len(skipped_folders)} folder(s) missing required recording files."
                )
            if processed_count == 0:
                QMessageBox.warning(
                    self,
                    "No folders built",
                    "All folders were skipped because required recording files were missing.",
                )
                return

            summary = (
                f"\n\nProcessed: {processed_count}\nSkipped: {len(skipped_folders)}"
            )
            if image_output:
                output_path = builder.dataset_dir / "images"
                self._append(f"Done. Image folder: {output_path}")
                QMessageBox.information(
                    self, "Completed", f"Image dataset built:\n{output_path}{summary}"
                )
            else:
                builder.write_split_masks()
                self._append("Outcome masks written: success/failure and split-specific filters.")
                self._append(f"Done. Dataset path: {builder.dataset_path}")
                QMessageBox.information(
                    self, "Completed", f"Dataset built:\n{builder.dataset_path}{summary}"
                )
        except Exception:
            err = traceback.format_exc()
            self._append(err)
            QMessageBox.critical(self, "Build failed", err)
        finally:
            self.setEnabled(True)


def main() -> int:
    app = QApplication(sys.argv)
    win = DatasetBuilderGUI()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
