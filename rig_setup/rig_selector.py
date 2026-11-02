from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union
from copy import deepcopy

from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QFormLayout,
    QGroupBox,
    QWidget,
    QSpinBox,
    QScrollArea,
)

from .rig_config import (
    DEVICE_SLOTS,
    SHARED_DEVICE_SLOTS,
    PIPETTE_DEVICE_SLOTS,
    RigConfigError,
    RigConfigManager,
)


def _get_nested(params: Dict[str, Any], dotted: str, default=None):
    """
    Retrieve a nested value from a dictionary using a dotted key path.

    Args:
        params: Dictionary to search within.
        dotted: Dot-separated key string (e.g., "a.b.c").
        default: Value to return if the key path does not exist.

    Returns:
        The value at the specified nested key, or default if not found.
    """
    cur = params
    parts = dotted.split(".")
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _set_nested(params: Dict[str, Any], dotted: str, value: Any) -> None:
    """
    Set a nested value in a dictionary using a dotted key path.

    Args:
        params: Dictionary to modify.
        dotted: Dot-separated key string (e.g., "a.b.c").
        value: Value to assign at the nested location.
    """
    cur = params
    parts = dotted.split(".")
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _coerce(text: str, typ: str) -> Any:
    """
    Convert a string to a specified type.

    Args:
        text: Input string value.
        typ: Target type ("int", "float", "bool", or "str").

    Returns:
        The converted value.

    Raises:
        ValueError: If conversion fails for int or float.
    """
    if typ == "int":
        return int(text)
    if typ == "float":
        return float(text)
    if typ == "bool":
        return text.strip().lower() in ("1", "true", "yes", "on")
    return text


class SettingsDialog(QDialog):
    """Dialog for editing device parameter fields."""
    def __init__(self, slot: str, fields: List[Dict[str, Any]], params: Dict[str, Any], parent=None):
        """
        Initialize the settings dialog UI.

        Args:
            slot: Device slot name.
            fields: Field definitions for editable parameters.
            params: Existing parameter values.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle(f"{slot} settings")
        self.fields = fields
        self.params = deepcopy(params)
        self.inputs: Dict[str, QLineEdit] = {}
        layout = QVBoxLayout()
        form = QFormLayout()
        layout.addLayout(form)

        if not fields:
            label = QLabel("No configurable settings for this device.")
            label.setStyleSheet("color: gray;")
            layout.addWidget(label)
        else:
            for field in fields:
                key = field["key"]
                label_text = field.get("label", key)
                typ = field.get("type", "str")
                readonly = field.get("readonly", False)
                val = _get_nested(self.params, key, "")
                line = QLineEdit(str(val) if val not in (None, "") else "")
                line.setReadOnly(readonly)
                line.setPlaceholderText(typ)
                form.addRow(QLabel(label_text), line)
                self.inputs[key] = line

        btn_row = QHBoxLayout()
        save = QPushButton("OK")
        cancel = QPushButton("Cancel")
        save.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(save)
        btn_row.addWidget(cancel)
        layout.addLayout(btn_row)
        self.setLayout(layout)

    def get_params(self) -> Dict[str, Any]:
        """
        Retrieve updated parameters from user input.

        Returns:
            Dictionary of updated parameters.
        """
        params = deepcopy(self.params)
        for key, widget in self.inputs.items():
            text = widget.text().strip()
            field = next((f for f in self.fields if f["key"] == key), {})
            typ = field.get("type", "str")
            if text == "" and field.get("optional"):
                continue
            try:
                val = _coerce(text, typ)
            except Exception:
                val = text
            _set_nested(params, key, val)
        return params


class RigBuilderDialog(QDialog):
    """Dialog for composing a new rig configuration."""

    def __init__(self, manager: RigConfigManager, parent=None, initial_config: Dict[str, Any] | None = None, save_path: Path | None = None):
        """
        Initialize the rig builder dialog.

        Args:
            manager: Configuration manager instance.
            parent: Optional parent widget.
            initial_config: Existing config to load into the UI.
            save_path: Path where config will be saved.
        """
        super().__init__(parent)
        self.manager = manager
        self.setWindowTitle("Create Rig Configuration")
        self.resize(900, 600)
        self.saved_path: Path | None = None
        self.edit_path = save_path
        self.options = manager.get_device_options()
        self.slot_rows: Dict[str, List[Dict[str, Any]]] = {
            slot: [] for slot in DEVICE_SLOTS
        }
        self.pipette_row_cache: Dict[str, List[Dict[str, Any]]] = {
            slot: [] for slot in PIPETTE_DEVICE_SLOTS
        }
        self.initial_config = initial_config
        self.pipette_count = self._get_pipette_count(initial_config)
        self._build_ui()
        if self.initial_config:
            self._load_config(self.initial_config)

    def _get_pipette_count(self, config: Dict[str, Any] | None) -> int:
        """
        Gets maximum number of pipettes available on rig.

        Args:
            config (Dict[str, Any]): Rig config.
        
        Returns:
            Number of pipettes available on given rig.
        """
        if not config:
            return 1

        try:
            return max(
                1,
                int(config.get("pipette_count", 1)),
            )
        except (TypeError, ValueError):
            return 1

    def _build_ui(self) -> None:
        """Construct the dialog user interface."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # ------------------------------------------------------------
        # Scrollable content
        # ------------------------------------------------------------

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        scroll_area.setWidget(scroll_widget)

        layout.addWidget(scroll_area)

        # ------------------------------------------------------------
        # Rig name
        # ------------------------------------------------------------

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Rig name:"))

        self.name_edit = QLineEdit(
            self.initial_config.get("name", "Custom Rig")
            if self.initial_config
            else "Custom Rig"
        )

        name_row.addWidget(self.name_edit)
        scroll_layout.addLayout(name_row)

        # ------------------------------------------------------------
        # Pipette count
        # ------------------------------------------------------------

        pipette_count_row = QHBoxLayout()

        pipette_count_row.addWidget(
            QLabel("Rig pipette capacity:")
        )

        self.pipette_count_label = QLabel(
            str(self.pipette_count)
        )

        pipette_count_row.addWidget(
            self.pipette_count_label
        )

        pipette_count_row.addStretch()

        scroll_layout.addLayout(pipette_count_row)

        # ------------------------------------------------------------
        # Shared devices
        # ------------------------------------------------------------

        shared_group = QGroupBox("Shared Devices")
        shared_layout = QVBoxLayout(shared_group)

        for slot in SHARED_DEVICE_SLOTS:
            self._add_device_row(
                shared_layout,
                slot,
                instance_index=0,
            )

        scroll_layout.addWidget(shared_group)

        # ------------------------------------------------------------
        # Pipette-specific devices
        # ------------------------------------------------------------

        self.pipette_container = QWidget()
        self.pipette_layout = QVBoxLayout(
            self.pipette_container
        )

        scroll_layout.addWidget(self.pipette_container)

        self._rebuild_pipette_sections(
            preserve_existing=False
        )
  
        scroll_layout.addStretch()
        # ------------------------------------------------------------
        # Save / cancel
        # ------------------------------------------------------------

        btns = QHBoxLayout()

        save_btn = QPushButton("Save")
        cancel_btn = QPushButton("Cancel")

        save_btn.clicked.connect(self._on_save)
        cancel_btn.clicked.connect(self.reject)

        btns.addStretch()
        btns.addWidget(save_btn)
        btns.addWidget(cancel_btn)

        layout.addLayout(btns)

    def _add_device_row(self, parent_layout: QVBoxLayout, slot: str, instance_index: int) -> None:
        """
        Add one configurable device row to the UI.
        """
        row = QHBoxLayout()

        label = QLabel(
            slot.replace("_", " ").title()
        )

        combo = QComboBox()

        for opt in self.options.get(slot, []):
            combo.addItem(
                opt.get(
                    "label",
                    opt.get("class", "unknown"),
                ),
                opt,
            )

        settings_btn = QPushButton("Settings")

        row.addWidget(label)
        row.addWidget(combo, stretch=3)
        row.addWidget(settings_btn, stretch=1)

        parent_layout.addLayout(row)

        widgets = {
            "combo": combo,
            "params": {},
        }

        self.slot_rows[slot].append(widgets)

        combo.currentIndexChanged.connect(
            lambda _,
            s=slot,
            i=instance_index,
            c=combo:
            self._apply_option_defaults(s, i, c)
        )

        settings_btn.clicked.connect(
            lambda _,
            s=slot,
            i=instance_index:
            self._open_settings(s, i)
        )

        self._apply_option_defaults(
            slot,
            instance_index,
            combo,
        )

    def _apply_option_defaults(self, slot: str, instance_index: int, combo: QComboBox) -> None:
        """
        Apply default parameters for a selected device option.
        """
        opt = combo.currentData()

        if not isinstance(opt, dict):
            return

        params = opt.get("params", {})

        self.slot_rows[slot][instance_index]["params"] = (
            deepcopy(params)
        )

    def _open_settings(self, slot: str, instance_index: int) -> None:
        """
        Open the settings dialog for a device instance.
        """
        widgets = self.slot_rows[slot][instance_index]

        opt = widgets["combo"].currentData()

        if not isinstance(opt, dict):
            QMessageBox.warning(
                self,
                "No device",
                "Select a device first.",
            )
            return

        fields = opt.get("fields", [])

        if slot in PIPETTE_DEVICE_SLOTS:
            title = (
                f"Pipette {instance_index + 1} - "
                f"{slot.replace('_', ' ').title()}"
            )
        else:
            title = slot.replace("_", " ").title()

        dlg = SettingsDialog(
            title,
            fields,
            widgets["params"],
            self,
        )

        if dlg.exec_() == QDialog.Accepted:
            widgets["params"] = dlg.get_params()

    def _on_save(self) -> None:
        """
        Save the current rig configuration.
        """
        name = (
            self.name_edit.text().strip()
            or "Custom Rig"
        )

        devices: Dict[str, Any] = {}

        try:

            # --------------------------------------------------------
            # Shared devices
            # --------------------------------------------------------

            for slot in SHARED_DEVICE_SLOTS:

                if not self.slot_rows[slot]:
                    raise RigConfigError(
                        f"No device configured for '{slot}'."
                    )

                widgets = self.slot_rows[slot][0]

                opt = widgets["combo"].currentData()

                if not opt or "class" not in opt:
                    raise RigConfigError(
                        f"No class selected for slot '{slot}'."
                    )

                devices[slot] = {
                    "class": opt["class"],
                    "params": deepcopy(
                        widgets["params"]
                    ),
                }

            # --------------------------------------------------------
            # Pipette-specific devices
            # --------------------------------------------------------

            for slot in PIPETTE_DEVICE_SLOTS:

                specs = []

                for pipette_index in range(
                    self.pipette_count
                ):

                    widgets = self.slot_rows[slot][
                        pipette_index
                    ]

                    opt = widgets["combo"].currentData()

                    if not opt or "class" not in opt:
                        raise RigConfigError(
                            f"No class selected for '{slot}' "
                            f"on Pipette {pipette_index + 1}."
                        )

                    specs.append(
                        {
                            "class": opt["class"],
                            "params": deepcopy(
                                widgets["params"]
                            ),
                        }
                    )

                if self.pipette_count == 1:
                    devices[slot] = specs[0]
                else:
                    devices[slot] = specs

        except (RigConfigError, json.JSONDecodeError) as exc:

            QMessageBox.critical(
                self,
                "Invalid configuration",
                str(exc),
            )

            return

        if self.initial_config:
            config = deepcopy(self.initial_config)
        else:
            config = self.manager.build_empty_template()

        config["name"] = name
        config["schema_version"] = 2
        config["pipette_count"] = self.pipette_count
        config["devices"] = devices

        path = (
            self.edit_path
            or self.manager.make_config_path(name)
        )

        try:
            self.manager.save_config(
                path,
                config,
            )

        except Exception as exc:

            QMessageBox.critical(
                self,
                "Save failed",
                str(exc),
            )

            return

        self.saved_path = path
        self.accept()

    def _load_config(self, config: Dict[str, Any]) -> None:
        """
        Load an existing configuration into the UI.
        """
        devices = config.get("devices", {})

        if not isinstance(devices, dict):
            return

        # ------------------------------------------------------------
        # Shared devices
        # ------------------------------------------------------------

        for slot in SHARED_DEVICE_SLOTS:
            spec = devices.get(slot)

            if isinstance(spec, dict):
                self._set_row_spec(
                    slot,
                    0,
                    spec,
                )

        # ------------------------------------------------------------
        # Pipette-specific devices
        # ------------------------------------------------------------

        for slot in PIPETTE_DEVICE_SLOTS:

            spec = devices.get(slot)

            if isinstance(spec, list):
                specs = spec

            elif isinstance(spec, dict):
                specs = [spec]

            else:
                continue

            # Store the complete configs, including parameters that may
            # not have visible SettingsDialog fields.
            self.pipette_row_cache[slot] = deepcopy(specs)

            for index, device_spec in enumerate(specs):

                if index >= len(self.slot_rows[slot]):
                    break

                self._set_row_spec(
                    slot,
                    index,
                    device_spec,
                )

    def _rebuild_pipette_sections(self, preserve_existing: bool = True) -> None:
        """
        Rebuild pipette-specific controls using the rig's configured
        maximum pipette count.
        """
        if preserve_existing:
            self._cache_pipette_rows()

        self._clear_layout(self.pipette_layout)

        for slot in PIPETTE_DEVICE_SLOTS:
            self.slot_rows[slot] = []

        for pipette_index in range(self.pipette_count):

            group = QGroupBox(
                f"Pipette {pipette_index + 1}"
            )

            group_layout = QVBoxLayout(group)

            for slot in PIPETTE_DEVICE_SLOTS:
                self._add_device_row(
                    group_layout,
                    slot,
                    instance_index=pipette_index,
                )

            self.pipette_layout.addWidget(group)

        if preserve_existing:
            self._restore_pipette_rows()

    def _clear_layout(self, layout) -> None:
        """Recursively remove widgets and child layouts."""
        while layout.count():
            item = layout.takeAt(0)

            widget = item.widget()

            if widget is not None:
                widget.deleteLater()
                continue

            child_layout = item.layout()

            if child_layout is not None:
                self._clear_layout(child_layout)

    def _cache_pipette_rows(self) -> None:
        """
        Capture current pipette-specific selections and parameters.
        """
        for slot in PIPETTE_DEVICE_SLOTS:

            rows = self.slot_rows.get(slot, [])

            for index, widgets in enumerate(rows):

                opt = widgets["combo"].currentData()

                spec = {
                    "class": (
                        opt.get("class")
                        if isinstance(opt, dict)
                        else None
                    ),
                    "params": deepcopy(
                        widgets["params"]
                    ),
                }

                # Replace an existing cached pipette.
                if index < len(self.pipette_row_cache[slot]):
                    self.pipette_row_cache[slot][index] = spec

                # Or add a newly created pipette.
                else:
                    self.pipette_row_cache[slot].append(spec)

    def _restore_pipette_rows(self) -> None:
        """
        Restore cached pipette configurations into currently visible rows.
        """
        for slot in PIPETTE_DEVICE_SLOTS:

            cached_specs = self.pipette_row_cache.get(
                slot,
                []
            )

            for index, spec in enumerate(cached_specs):

                if index >= len(self.slot_rows[slot]):
                    break

                self._set_row_spec(
                    slot,
                    index,
                    spec,
                )

    def _set_row_spec(self, slot: str, instance_index: int, spec: Dict[str, Any]) -> None:
        """
        Apply a saved device specification to one editor row.
        """
        if instance_index >= len(self.slot_rows[slot]):
            return

        widgets = self.slot_rows[slot][instance_index]

        combo: QComboBox = widgets["combo"]

        target_class = spec.get("class")

        match_idx = -1

        for i in range(combo.count()):
            data = combo.itemData(i)

            if (
                isinstance(data, dict)
                and data.get("class") == target_class
            ):
                match_idx = i
                break

        if match_idx >= 0:
            combo.setCurrentIndex(match_idx)

        widgets["params"] = deepcopy(
            spec.get("params", {})
        )

class RigSelectorDialog(QDialog):
    """Dialog to select or create a rig configuration before launching the GUI."""

    def __init__(self, manager: RigConfigManager, parent=None):
        """
        Initialize the rig selector dialog.

        Args:
            manager: Configuration manager instance.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self.manager = manager
        self.setWindowTitle("Select Rig Configuration")
        self.resize(500, 200)
        self.selected_path: Path | None = None
        self.manager.ensure_default_config()
        self.selected_pipette_count: int = 1
        self._build_ui()
        self._reload_configs()

    def _build_ui(self) -> None:
        """Construct the dialog user interface."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        layout.addWidget(QLabel("Choose a rig configuration JSON to load:"))
        self.config_combo = QComboBox()
        self.config_combo.currentIndexChanged.connect(self._update_path_display)
        layout.addWidget(self.config_combo)

        pipette_row = QHBoxLayout()

        pipette_row.addWidget(
            QLabel("Pipettes to use:")
        )

        self.pipette_count_combo = QComboBox()

        pipette_row.addWidget(
            self.pipette_count_combo
        )

        layout.addLayout(pipette_row)

        btn_row = QHBoxLayout()
        browse_btn = QPushButton("Browse...")
        new_btn = QPushButton("New config")
        edit_btn = QPushButton("Edit selected")
        refresh_btn = QPushButton("Refresh list")
        browse_btn.clicked.connect(self._browse_for_config)
        new_btn.clicked.connect(self._create_config)
        edit_btn.clicked.connect(self._edit_config)
        refresh_btn.clicked.connect(self._reload_configs)
        btn_row.addWidget(browse_btn)
        btn_row.addWidget(new_btn)
        btn_row.addWidget(edit_btn)
        btn_row.addWidget(refresh_btn)
        layout.addLayout(btn_row)

        action_row = QHBoxLayout()
        load_btn = QPushButton("Load")
        cancel_btn = QPushButton("Cancel")
        load_btn.clicked.connect(self._on_accept)
        cancel_btn.clicked.connect(self.reject)
        action_row.addStretch()
        action_row.addWidget(load_btn)
        action_row.addWidget(cancel_btn)
        layout.addLayout(action_row)

    def _reload_configs(self) -> None:
        """Reload available configuration files into the dropdown."""
        current = self.config_combo.currentData()
        self.config_combo.blockSignals(True)
        self.config_combo.clear()
        for path in self.manager.list_configs():
            self.config_combo.addItem(path.name, path)
        self.config_combo.blockSignals(False)

        if current:
            idx = self.config_combo.findData(current)
            if idx >= 0:
                self.config_combo.setCurrentIndex(idx)
        if self.config_combo.count() and self.config_combo.currentIndex() < 0:
            self.config_combo.setCurrentIndex(0)
        self._update_path_display()

    def _update_path_display(self) -> None:
        path = self.config_combo.currentData()

        if not isinstance(path, Path):
            return

        try:
            config = self.manager.load_config(
                path,
                load_overlays=False,
            )
        except Exception:
            return

        try:
            max_pipettes = max(
                1,
                int(config.get("pipette_count", 1)),
            )
        except (TypeError, ValueError):
            max_pipettes = 1

        self.pipette_count_combo.clear()

        for count in range(
            1,
            max_pipettes + 1,
        ):
            self.pipette_count_combo.addItem(
                str(count),
                count,
            )

    def _browse_for_config(self) -> None:
        """Open file dialog to manually select a configuration file."""
        path_str, _ = QFileDialog.getOpenFileName(self, "Select rig config", str(self.manager.config_dir), "JSON (*.json)")
        if path_str:
            self.selected_path = Path(path_str)

    def _create_config(self) -> None:
        """Launch the rig builder dialog to create a new configuration."""
        builder = RigBuilderDialog(self.manager, self)
        if builder.exec_() == QDialog.Accepted and builder.saved_path:
            self.selected_path = builder.saved_path
            self._reload_configs()
            idx = self.config_combo.findData(builder.saved_path)
            if idx >= 0:
                self.config_combo.setCurrentIndex(idx)

    def _edit_config(self) -> None:
        """Open the selected configuration for editing."""
        path = self.config_combo.currentData()
        if not isinstance(path, Path):
            QMessageBox.warning(self, "No configuration", "Select a config to edit.")
            return
        try:
            config = self.manager.load_config(
            path,
            load_overlays=False,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return
        builder = RigBuilderDialog(self.manager, self, initial_config=config, save_path=path)
        if builder.exec_() == QDialog.Accepted and builder.saved_path:
            self.selected_path = builder.saved_path
            self._reload_configs()
            idx = self.config_combo.findData(builder.saved_path)
            if idx >= 0:
                self.config_combo.setCurrentIndex(idx)

    def _on_accept(self) -> None:
        """Confirm selection of a configuration and close the dialog."""
        if self.selected_path is None:
            data = self.config_combo.currentData()
            if isinstance(data, Path):
                self.selected_path = data
        if not self.selected_path:
            QMessageBox.warning(self, "No configuration", "Please select or create a rig configuration.")
            return
        selected_count = (
            self.pipette_count_combo.currentData()
        )

        if selected_count is None:
            selected_count = 1

        self.selected_pipette_count = int(
            selected_count
        )
        self.accept()

