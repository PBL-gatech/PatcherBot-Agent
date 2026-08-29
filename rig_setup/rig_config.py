"""
Rig configuration schema and loader utilities.

Schema (schema_version=2):
{
  "name": "<friendly name>",
  "schema_version": 2,
  "calibration_file": "<calibration config file>",
  "patch_file": "<patch config file>",
  "protocol_file": "<protocol config file>",
  "devices": {
    "<slot>": {
      "class": "module.ClassName",
      "params": { ... }   # simple kwargs; extra keys are applied as attributes
    }
  }
}

Special value helpers allowed in args/kwargs/post_init:
- "$ref:<slot_name>" or {"__ref__": "<slot_name>"} : injects another device instance.
- Lists of numbers auto-convert to numpy arrays when numpy is available.
- {"__type__": "serial", ...} : creates serial.Serial(**params) (optional, only needed for serial ports).
"""
from __future__ import annotations

import importlib
import json
import logging
import re
from pathlib import Path
import inspect
from typing import Any, Dict, List, Tuple
import yaml

LOGGER = logging.getLogger(__name__)

SCHEMA_VERSION = 2
CONFIG_DIR = Path(__file__).parent / "rig_configs"
CAL_CONFIG_DIR = Path(__file__).parent / "cal_configs"
PATCH_CONFIG_DIR = Path(__file__).parent / "patch_configs"
PROTOCOL_CONFIG_DIR = Path(__file__).parent / "protocol_configs"
DEFAULT_CONFIG_NAME = "fake_rig.json"

# Core device slots only; derived pieces (stage, pipette_unit, microscope) are built automatically.
SHARED_DEVICE_SLOTS: List[str] = [
    "stage_controller",
    "camera",
    "cell_sorter_controller",
    "cell_sorter_manipulator",
    "lamp",
    "laser",
]

PIPETTE_DEVICE_SLOTS: List[str] = [
    "pipette_controller",
    "pipette_camera",
    "pressure",
    "daq",
    "amplifier",
]

DEVICE_SLOTS: List[str] = [
    *PIPETTE_DEVICE_SLOTS,
    *SHARED_DEVICE_SLOTS,
]


class RigConfigError(Exception):
    """Raised when a configuration cannot be loaded or instantiated."""


def _slugify(text: str) -> str:
    """
    Convert text into a filesystem-safe slug.

    Args:
        text (str): Input string.

    Returns:
        str: Slugified string.
    """
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_")
    return slug or "rig_config"


def _default_devices() -> Dict[str, Dict[str, Any]]:
    """
    Default fake rig mirrors rig_setup/setup_fake_rig.py.
    
    Returns:
        Dict[str, Dict[str, Any]]: Default device configuration mapping.
    """
    return {
        "stage_controller": {
            "class": "patcherbot.devices.manipulator.fakemanipulator.FakeManipulator",
            "params": {"min": [-240000, 50000, 280000], "max": [-230000, 60000, 290000], "x": [-235000, 55000, 285000]},
        },
        "pipette_controller": {
            "class": "patcherbot.devices.manipulator.fakemanipulator.FakeManipulator",
            "params": {"min": [0, 0, 0], "max": [4000, 20000, 20000], "x": [200, 300, 400]},
        },
        "cell_sorter_controller": {
            "class": "patcherbot.devices.cellsorter.CellSorter.FakeCellSorterController",
        },
        "cell_sorter_manipulator": {
            "class": "patcherbot.devices.cellsorter.CellSorter.FakeCellSorterManip",
        },
        "camera": {
            "class": "patcherbot.devices.camera.FakeCalCamera.FakeCalCamera",
            "params": {"image_z": 100},
        },
        "pipette_camera": {
            "class": "patcherbot.devices.camera.camera.FakeCamera",
            "params": {},
        },
        "daq": {
            "class": "patcherbot.devices.amplifier.DAQ.FakeDAQ",
            "params": {},
        },
        "amplifier": {
            "class": "patcherbot.devices.amplifier.amplifier.FakeAmplifier",
            "params": {},
        },
        "pressure": {
            "class": "patcherbot.devices.pressurecontroller.BasePressureController.FakePressureController",
            "params": {},
        },
        "lamp": {
            "class": "patcherbot.devices.lamp.lamp.FakeLamp",
            "params": {},
        },
        "laser": {
            "class": "patcherbot.devices.laser.laser.FakeLaser",
            "params": {},
        },
    }


def _empty_calibration() -> Dict[str, Any]:
    """
    Create an empty calibration configuration.

    Returns:
        Dict[str, Any]: Empty calibration dictionary.
    """
    return {
        "pipette_detector_model": None,
        "pipette_focuser_model": None,
        "use_ai_features": False,
    }

def _default_ai_features() -> Dict[str, Any]:
    """
    Create an empty calibration configuration.

    Returns:
        Dict[str, Any]: Empty calibration dictionary.
    """
    return {
        "enabled": True,
    }


# Available device options (used by the builder UI).
DEVICE_OPTIONS: Dict[str, List[Dict[str, Any]]] = {
    "stage_controller": [
        {
            "label": "FakeManipulator (stage)",
            "class": "patcherbot.devices.manipulator.fakemanipulator.FakeManipulator",
            "params": {"min": [-240000, 50000, 280000], "max": [-230000, 60000, 290000], "x": [-235000, 55000, 285000]},
            "fields": [],
        },
        {
            "label": "ScientificaSerialNoEncoder",
            "class": "patcherbot.devices.manipulator.scientificaSerial.ScientificaSerialNoEncoder",
            "params": {
                "serial": {"port": "COM6", "baudrate": 9600},
                "max_speed": 100000,
                "max_accel": 1000,
                "polling_freq": 100,
            },
            "fields": [
                {"key": "serial.port", "label": "Port", "type": "str"},
                {"key": "serial.baudrate", "label": "Baudrate", "type": "int"},
                {"key": "serial.timeout", "label": "Timeout", "type": "float", "optional": True},
                {"key": "max_speed", "label": "Max Speed", "type": "int", "optional": True},
                {"key": "max_accel", "label": "Max Accel", "type": "int", "optional": True},
                {"key": "polling_freq", "label": "Polling Hz", "type": "int", "optional": True},
            ],
        },
    ],
    "pipette_controller": [
        {
            "label": "FakeManipulator (pipette)",
            "class": "patcherbot.devices.manipulator.fakemanipulator.FakeManipulator",
            "params": {"min": [0, 0, 0], "max": [4000, 20000, 20000], "x": [200, 300, 400]},
            "fields": [],
        },
        {
            "label": "ScientificaSerialNoEncoder",
            "class": "patcherbot.devices.manipulator.scientificaSerial.ScientificaSerialNoEncoder",
            "params": {
                "serial": {"port": "COM3", "baudrate": 9600},
                "max_speed": 100000,
                "max_accel": 1000,
                "polling_freq": 100,
            },
            "fields": [
                {"key": "serial.port", "label": "Port", "type": "str"},
                {"key": "serial.baudrate", "label": "Baudrate", "type": "int"},
                {"key": "serial.timeout", "label": "Timeout", "type": "float", "optional": True},
                {"key": "max_speed", "label": "Max Speed", "type": "int", "optional": True},
                {"key": "max_accel", "label": "Max Accel", "type": "int", "optional": True},
                {"key": "polling_freq", "label": "Polling Hz", "type": "int", "optional": True},
            ],
        },
        {
            "label": "SensapexManip",
            "class": "patcherbot.devices.manipulator.sensapexWrapper.SensapexManip",
            "params": {"deviceID": None, "max_speed": 5000, "max_acceleration": 1, "tilt_angle_deg": 25.0},
            "fields": [
                {"key": "deviceID", "label": "Device ID", "type": "int", "optional": True},
                {"key": "max_speed", "label": "Max Speed", "type": "int", "optional": True},
                {"key": "max_acceleration", "label": "Max Accel", "type": "float", "optional": True},
                {"key": "tilt_angle_deg", "label": "Tilt Angle (deg)", "type": "float", "optional": True},
            ],
        },
    ],
    "cell_sorter_controller": [
        {
            "label": "FakeCellSorterController",
            "class": "patcherbot.devices.cellsorter.CellSorter.FakeCellSorterController",
            "params": {},
            "fields": [],
        },
        {
            "label": "CellSorterController (serial)",
            "class": "patcherbot.devices.cellsorter.CellSorter.CellSorterController",
            "params": {"comPort": {"port": "COM12", "baudrate": 115200, "timeout": 2, "write_timeout": 1}},
            "fields": [
                {"key": "comPort.port", "label": "Port", "type": "str"},
                {"key": "comPort.baudrate", "label": "Baudrate", "type": "int"},
                {"key": "comPort.timeout", "label": "Timeout", "type": "float"},
                {"key": "comPort.write_timeout", "label": "Write Timeout", "type": "float", "optional": True},
            ],
        },
    ],
    "cell_sorter_manipulator": [
        {
            "label": "FakeCellSorterManip",
            "class": "patcherbot.devices.cellsorter.CellSorter.FakeCellSorterManip",
            "params": {},
            "fields": [],
        },
        {
            "label": "CellSorterManip (serial)",
            "class": "patcherbot.devices.cellsorter.CellSorter.CellSorterManip",
            "params": {
                "comPort": {
                    "port": "COM10",
                    "baudrate": 57600,
                    "timeout": 2,
                    "write_timeout": 1,
                    "stopbits": 2,
                }
            },
            "fields": [
                {"key": "comPort.port", "label": "Port", "type": "str"},
                {"key": "comPort.baudrate", "label": "Baudrate", "type": "int"},
                {"key": "comPort.timeout", "label": "Timeout", "type": "float"},
                {"key": "comPort.write_timeout", "label": "Write Timeout", "type": "float", "optional": True},
                {"key": "comPort.stopbits", "label": "Stop Bits", "type": "int", "optional": True},
            ],
        },
    ],
    "camera": [
        {
            "label": "FakeCalCamera",
            "class": "patcherbot.devices.camera.FakeCalCamera.FakeCalCamera",
            "params": {"image_z": 100},
            "fields": [
                {"key": "image_z", "label": "Image Z", "type": "float"},
            ],
        },
        {
            "label": "PcoCamera",
            "class": "patcherbot.devices.camera.pcocamera.PcoCamera",
            "params": {},
            "fields": [],
        },
    ],
    "pipette_camera": [
        {
            "label": "FakeCamera",
            "class": "patcherbot.devices.camera.camera.FakeCamera",
            "params": {},
            "fields": [],
        },
        {
            "label": "PipetteCamera",
            "class": "patcherbot.devices.camera.PipetteCamera.PipetteCamera",
            "params": {},
            "fields": [],
        },
    ],
    "daq": [
        {
            "label": "FakeDAQ",
            "class": "patcherbot.devices.amplifier.DAQ.FakeDAQ",
            "params": {},
            "fields": [],
        },
        {
            "label": "NiDAQ (Moscow example)",
            "class": "patcherbot.devices.amplifier.DAQ.NiDAQ",
            "params": {"readDev": "cDAQ1Mod1", "readChannel": "ai0", "cmdDev": "cDaq1Mod4", "cmdChannel": "ao0", "respDev": "cDaq1Mod1", "respChannel": "ai3"},
            "fields": [
                {"key": "readDev", "label": "Read Dev", "type": "str"},
                {"key": "readChannel", "label": "Read Channel", "type": "str"},
                {"key": "cmdDev", "label": "Command Dev", "type": "str"},
                {"key": "cmdChannel", "label": "Command Channel", "type": "str"},
                {"key": "respDev", "label": "Response Dev", "type": "str"},
                {"key": "respChannel", "label": "Response Channel", "type": "str"},
            ],
        },
        {
            "label": "NiDAQ (IBB example)",
            "class": "patcherbot.devices.amplifier.DAQ.NiDAQ",
            "params": {"readDev": "cDAQ1Mod3", "readChannel": "ai0", "cmdDev": "cDaq1Mod1", "cmdChannel": "ao1", "respDev": "cDaq1Mod3", "respChannel": "ai1"},
            "fields": [
                {"key": "readDev", "label": "Read Dev", "type": "str"},
                {"key": "readChannel", "label": "Read Channel", "type": "str"},
                {"key": "cmdDev", "label": "Command Dev", "type": "str"},
                {"key": "cmdChannel", "label": "Command Channel", "type": "str"},
                {"key": "respDev", "label": "Response Dev", "type": "str"},
                {"key": "respChannel", "label": "Response Channel", "type": "str"},
            ],
        },
    ],
    "amplifier": [
        {
            "label": "FakeAmplifier",
            "class": "patcherbot.devices.amplifier.amplifier.FakeAmplifier",
            "params": {},
            "fields": [],
        },
        {
            "label": "MultiClampChannel (channel 1)",
            "class": "patcherbot.devices.amplifier.multiclamp.MultiClampChannel",
            "params": {"channel": 1},
            "fields": [
                {"key": "channel", "label": "Channel", "type": "int"},
            ],
        },
    ],
    "pressure": [
        {
            "label": "FakePressureController",
            "class": "patcherbot.devices.pressurecontroller.BasePressureController.FakePressureController",
            "params": {},
            "fields": [],
        },
        {
            "label": "MoscowPressureController",
            "class": "patcherbot.devices.pressurecontroller.MoscowPressureController.MoscowPressureController",
            "params": {
                "channel": 4,
                "controllerSerial": {"port": "COM5", "baudrate": 9600, "timeout": 0},
                "readerSerial": {"port": "COM9", "baudrate": 9600, "timeout": 0},
                "serial_cmd_timeout": 1,
            },
            "fields": [
                {"key": "channel", "label": "Channel", "type": "int"},
                {"key": "controllerSerial.port", "label": "Controller Port", "type": "str"},
                {"key": "controllerSerial.baudrate", "label": "Controller Baud", "type": "int"},
                {"key": "controllerSerial.timeout", "label": "Controller Timeout", "type": "float", "optional": True},
                {"key": "readerSerial.port", "label": "Reader Port", "type": "str"},
                {"key": "readerSerial.baudrate", "label": "Reader Baud", "type": "int"},
                {"key": "readerSerial.timeout", "label": "Reader Timeout", "type": "float", "optional": True},
                {"key": "serial_cmd_timeout", "label": "Command Timeout", "type": "float", "optional": True},
            ],
        },
        {
            "label": "WaynesBoroPressureController",
            "class": "patcherbot.devices.pressurecontroller.WaynesboroPressureController.WaynesBoroPressureController",
            "params": {
                "channel": 1,
                "controllerSerial": {"port": "COM5", "baudrate": 9600, "timeout": 0},
                "readerSerial": {"port": "COM4", "baudrate": 9600, "timeout": 0},
                "serial_cmd_timeout": 1,
            },
            "fields": [
                {"key": "channel", "label": "Channel", "type": "int"},
                {"key": "controllerSerial.port", "label": "Controller Port", "type": "str"},
                {"key": "controllerSerial.baudrate", "label": "Controller Baud", "type": "int"},
                {"key": "controllerSerial.timeout", "label": "Controller Timeout", "type": "float", "optional": True},
                {"key": "readerSerial.port", "label": "Reader Port", "type": "str"},
                {"key": "readerSerial.baudrate", "label": "Reader Baud", "type": "int"},
                {"key": "readerSerial.timeout", "label": "Reader Timeout", "type": "float", "optional": True},
                {"key": "serial_cmd_timeout", "label": "Command Timeout", "type": "float", "optional": True},
            ],
        },
        {
            "label": "IBBPressureController",
            "class": "patcherbot.devices.pressurecontroller.IBBPressureController.IBBPressureController",
            "params": {
                "channel": 1,
                "arduinoSerial": {"port": "COM5", "baudrate": 9600, "timeout": 0},
                "serial_cmd_timeout": 1,
                "startup_pressure": 20,
            },
            "fields": [
                {"key": "channel", "label": "Channel", "type": "int"},
                {"key": "arduinoSerial.port", "label": "Arduino Port", "type": "str"},
                {"key": "arduinoSerial.baudrate", "label": "Arduino Baud", "type": "int"},
                {"key": "arduinoSerial.timeout", "label": "Arduino Timeout", "type": "float", "optional": True},
                {"key": "serial_cmd_timeout", "label": "Command Timeout", "type": "float", "optional": True},
                {"key": "startup_pressure", "label": "Startup Pressure", "type": "float", "optional": True},
            ],
        },
    ],
    "lamp": [
        {
            "label": "FakeLamp",
            "class": "patcherbot.devices.lamp.lamp.FakeLamp",
            "params": {},
            "fields": [],
        },
        {
            "label": "OlympusLamp",
            "class": "patcherbot.devices.lamp.olympus.OlympusLamp",
            "params": {"port": "COM21"},
            "fields": [
                {"key": "port", "label": "Port", "type": "str"},
            ],
        },
        {
            "label": "Lumencore",
            "class": "patcherbot.devices.lamp.lumencor.Lumencore",
            "params": {"com": {"port": "COM6", "baudrate": 9600, "timeout": 1}},
            "fields": [
                {"key": "com.port", "label": "Port", "type": "str"},
                {"key": "com.baudrate", "label": "Baudrate", "type": "int"},
                {"key": "com.timeout", "label": "Timeout", "type": "float", "optional": True},
            ],
        },
    ],
    "laser": [
        {
            "label": "FakeLaser",
            "class": "patcherbot.devices.laser.laser.FakeLaser",
            "params": {},
            "fields": [],
        },
        {
            "label": "LumencorLaser",
            "class": "patcherbot.devices.laser.lumencor.LumencorLaser",
            "params": {"com": {"port": "COM6", "baudrate": 9600, "timeout": 1}},
            "fields": [
                {"key": "com.port", "label": "Port", "type": "str"},
                {"key": "com.baudrate", "label": "Baudrate", "type": "int"},
                {"key": "com.timeout", "label": "Timeout", "type": "float", "optional": True},
            ],
        },
    ],
}


class RigConfigManager:
    """Manager for loading, validating, and instantiating rig configurations."""
    def __init__(self, config_dir: Path | None = None,
                 cal_config_dir: Path | None = None,
                 patch_config_dir: Path | None = None,
                 protocol_config_dir: Path | None = None):
        """
        Initialize RigConfigManager with directory paths.

        Args:
            config_dir (Path, optional): Directory for rig configs.
            cal_config_dir (Path, optional): Directory for calibration configs.
            patch_config_dir (Path, optional): Directory for patch configs.
            protocol_config_dir (Path, optional): Directory for protocol configs.
        """
        self.config_dir = config_dir or CONFIG_DIR
        self.cal_config_dir = cal_config_dir or CAL_CONFIG_DIR
        self.patch_config_dir = patch_config_dir or PATCH_CONFIG_DIR
        self.protocol_config_dir = protocol_config_dir or PROTOCOL_CONFIG_DIR
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.cal_config_dir.mkdir(parents=True, exist_ok=True)
        self.patch_config_dir.mkdir(parents=True, exist_ok=True)
        self.protocol_config_dir.mkdir(parents=True, exist_ok=True)

    def default_config_path(self) -> Path:
        """
        Get default configuration file path.

        Returns:
            Path: Path to default config file.
        """
        return self.config_dir / DEFAULT_CONFIG_NAME

    def ensure_default_config(self) -> Path:
        """
        Get default configuration file path.

        Returns:
            Path: Path to default config file.
        """
        path = self.default_config_path()
        if not path.exists():
            self.save_config(
                path,
                {
                    "name": "Fake Rig",
                    "schema_version": SCHEMA_VERSION,
                    "pipette_count": 1,
                    "calibration_file": "fake_cal.yaml",
                    "patch_file": "fake_patch.yaml",
                    "protocol_file": "fake_protocol.yaml",
                    "calibration": _empty_calibration(),
                    "ai_features": _default_ai_features(),
                    "devices": _default_devices(),
                },
            )
        return path

    def list_configs(self) -> List[Path]:
        """
        List all available configuration files.

        Returns:
            List[Path]: Sorted list of config file paths.
        """
        return sorted(self.config_dir.glob("*.json"))

    def save_config(self, path: Path, data: Dict[str, Any]) -> None:
        """
        Save configuration data to a file.

        Args:
            path (Path): File path to save config.
            data (Dict[str, Any]): Configuration data.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        sanitized = {
            key: value for key, value in data.items()
            if not (isinstance(key, str) and key.startswith("_"))
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(sanitized, f, indent=2)

    def load_config(self, path: Path, load_overlays: bool = True) -> Dict[str, Any]:
        """
        Load and validate a configuration file.

        Args:
            path (Path): Path to configuration file.

        Returns:
            Dict[str, Any]: Loaded configuration.

        Raises:
            RigConfigError: If file does not exist or validation fails.
        """
        if not path.exists():
            raise RigConfigError(f"Configuration file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        self._validate_config(config)

        if load_overlays:
            self._load_overlay_configs(config)

        return config

    def build_devices_from_file(self, path: Path, active_pipette_count: int | None = None) -> Dict[str, Any]:
        """
        Load config and instantiate devices.

        Args:
            path (Path): Path to configuration file.

        Returns:
            Dict[str, Any]: Instantiated devices.
        """
        config = self.load_config(path)
        return self.build_devices(config, active_pipette_count=active_pipette_count)

    def build_devices(self, config: Dict[str, Any], active_pipette_count: int | None = None) -> Dict[str, Any]:
        """
        Instantiate all devices from configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary.

        Returns:
            Dict[str, Any]: All instantiated devices.
        """
        max_pipette_count = int(
            config.get("pipette_count", 1)
        )

        if active_pipette_count is None:
            active_pipette_count = max_pipette_count

        if not 1 <= active_pipette_count <= max_pipette_count:
            raise RigConfigError(
                f"Requested {active_pipette_count} pipettes, "
                f"but this rig supports {max_pipette_count}."
            )
        devices_cfg = config.get("devices", {})
        self._apply_pressure_calibration(config, devices_cfg)
        base_instances: Dict[str, Any] = {}
        for slot in DEVICE_SLOTS:
            if slot not in devices_cfg:
                raise RigConfigError(f"Missing device slot '{slot}' in configuration.")
            curr_device = devices_cfg[slot]
            
            if isinstance(curr_device, list):
                if slot not in PIPETTE_DEVICE_SLOTS:
                    raise RigConfigError(f"Shared device slot '{slot}' cannot contain multiple devices.")
                if not curr_device:
                    raise RigConfigError(f"Device slot '{slot}' contains an empty device list.")
                devices_to_initialize = curr_device[:active_pipette_count]
                instance_dict: Dict[str, Any] = {}
                for i, dev_cfg in enumerate(devices_to_initialize):

                    if not isinstance(dev_cfg, dict):
                        raise RigConfigError(
                            f"Invalid configuration for '{slot}[{i}]'. "
                            "Expected a device mapping."
                        )

                    instance_key = f"{slot}_{i}"

                    instance = self._instantiate_slot(
                        instance_key,
                        dev_cfg,
                        base_instances,
                    )

                    instance_dict[instance_key] = instance

                base_instances[slot] = instance_dict

            # Single-instance device slot.
            elif isinstance(curr_device, dict):
                base_instances[slot] = self._instantiate_slot(
                    slot,
                    curr_device,
                    base_instances,
                )
            else:
                raise RigConfigError(
                f"Invalid configuration for device slot '{slot}'. "
                "Expected a device mapping or list of device mappings."
            )

        # Auto-wire derived components
        from patcherbot.devices.manipulator.manipulatorunit import ManipulatorUnit
        from patcherbot.devices.manipulator.microscope import Microscope

        stage_controller = base_instances["stage_controller"]
        pipette_controllers = base_instances["pipette_controller"]

        pipette_units = {}
        if isinstance(pipette_controllers, dict):
            for key, controller in pipette_controllers.items():
                pipette_units[key] = ManipulatorUnit(controller, [1, 2, 3])
        else:
            pipette_units = ManipulatorUnit(pipette_controllers, [1, 2, 3])

        derived = {
            "stage": ManipulatorUnit(stage_controller, [1, 2]),
            "pipette_unit": pipette_units,
            "microscope": Microscope(stage_controller, 3),
        }
        derived["microscope"].up_direction = 1.0
        calibration = config.get("calibration")
        if isinstance(calibration, dict):
            units = calibration.get("microscope_units_per_um")
            if units is not None:
                derived["microscope"].set_units_per_um(units)

        # Inject refs into cameras if they declare matching kwargs
        camera = base_instances.get("camera")
        if camera and hasattr(camera, "__dict__"):
            if hasattr(camera, "stageManip"):
                camera.stageManip = stage_controller
            if hasattr(camera, "pipetteManip"):
                if isinstance(pipette_controllers, dict):
                    camera.pipetteManip = list(pipette_controllers.values())[0]
                else:
                    camera.pipetteManip = pipette_controllers
            if hasattr(camera, "cellSorterManip") and "cell_sorter_manipulator" in base_instances:
                camera.cellSorterManip = base_instances["cell_sorter_manipulator"]
            calibration_cfg = config.get("_resolved_calibration")
            if not isinstance(calibration_cfg, dict):
                calibration_cfg = config.get("calibration")
            ai_enabled = False
            if isinstance(calibration_cfg, dict) and "use_ai_features" in calibration_cfg:
                ai_enabled = bool(calibration_cfg.get("use_ai_features"))
            setattr(camera, "use_ai_features", ai_enabled)

        all_devices = {**base_instances, **derived}
        LOGGER.info("Initialized devices: %s", ", ".join(sorted(all_devices.keys())))
        return all_devices

    def _instantiate_slot(self, slot: str, cfg: Dict[str, Any], instances: Dict[str, Any]) -> Any:
        """
        Instantiate a device for a given slot.

        Args:
            slot (str): Device slot name.
            cfg (Dict[str, Any]): Slot configuration.
            instances (Dict[str, Any]): Existing instances for reference resolution.

        Returns:
            Any: Instantiated device.

        Raises:
            RigConfigError: If import or instantiation fails.
        """
        class_path = cfg.get("class") or cfg.get("class_path")
        if not class_path or "." not in class_path:
            raise RigConfigError(f"Invalid class for slot '{slot}'.")
        module_name, class_name = class_path.rsplit(".", 1)
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
        except Exception as exc:
            raise RigConfigError(f"Failed to import {class_path} for slot '{slot}': {exc}") from exc

        params_cfg = cfg.get("params", {}) or {}
        ctor_kwargs, post_attrs = self._split_ctor_params(cls, params_cfg, instances)
        try:
            instance = cls(**ctor_kwargs)
        except Exception as exc:
            raise RigConfigError(f"Failed to instantiate {class_path} for slot '{slot}': {exc}") from exc

        for attr, value in post_attrs.items():
            try:
                setattr(instance, attr, value)
            except Exception as exc:
                raise RigConfigError(f"Failed to set attribute '{attr}' for slot '{slot}': {exc}") from exc

        LOGGER.info("Initialized slot %s with %s", slot, class_path)
        return instance

    def _split_ctor_params(self, cls: Any, params_cfg: Dict[str, Any], instances: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Split parameters into constructor args and post-init attributes.

        Args:
            cls (Any): Target class.
            params_cfg (Dict[str, Any]): Parameter configuration.
            instances (Dict[str, Any]): Existing instances.

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: Constructor kwargs and post-init attributes.
        """
        resolved = {k: self._resolve_value(v, instances) for k, v in (params_cfg or {}).items()}
        sig = inspect.signature(cls.__init__)
        has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

        # Auto-map common aliases (e.g., serial -> comPort) to match ctor signatures.
        if "serial" in resolved and "comPort" in sig.parameters and "comPort" not in resolved:
            resolved["comPort"] = resolved.pop("serial")

        ctor_kwargs: Dict[str, Any] = {}
        post_attrs: Dict[str, Any] = {}
        for key, val in resolved.items():
            if has_kwargs or key in sig.parameters:
                ctor_kwargs[key] = val
            else:
                post_attrs[key] = val
        return ctor_kwargs, post_attrs

    def _resolve_value(self, value: Any, instances: Dict[str, Any]) -> Any:
        """
        Resolve special configuration values.

        Args:
            value (Any): Input value.
            instances (Dict[str, Any]): Existing instances.

        Returns:
            Any: Resolved value.

        Raises:
            RigConfigError: If reference resolution fails or serial creation fails.
        """
        if isinstance(value, str):
            if value.startswith("$ref:"):
                ref = value.split(":", 1)[1]

                # ============================================================
                # NEW: Handle indexed references such as "$ref:pressure[0]"
                # ============================================================
                match = re.fullmatch(
                    r"([A-Za-z_][A-Za-z0-9_]*)\[(\d+)\]",
                    ref
                )

                if match:
                    slot, index = match.groups()
                    index = int(index)

                    # Make sure the slot exists
                    if slot not in instances:
                        raise RigConfigError(
                            f"Reference '{slot}' not available yet."
                        )

                    # Multi-device slots are stored as dictionaries
                    device_group = instances[slot]

                    if not isinstance(device_group, dict):
                        raise RigConfigError(
                            f"Reference '{slot}[{index}]' is not a multi-device slot."
                        )

                    devices = list(device_group.values())

                    if index >= len(devices):
                        raise RigConfigError(
                            f"Reference '{slot}[{index}]' does not exist. "
                            f"Available devices: {len(devices)}"
                        )

                    return devices[index]

                # ============================================================
                # EXISTING: Handle normal references such as "$ref:pressure"
                # ============================================================
                if ref not in instances:
                    raise RigConfigError(
                        f"Reference '{ref}' not available yet."
                    )

                return instances[ref]

            return value
        if isinstance(value, list):
            # Auto-convert flat numeric lists to numpy arrays (keeps JSON simple).
            if value and all(isinstance(v, (int, float)) for v in value):
                try:
                    import numpy as np  # type: ignore
                    return np.array(value, dtype=float)
                except Exception:
                    pass
            return [self._resolve_value(v, instances) for v in value]
        if isinstance(value, dict):
            if "__ref__" in value:
                ref = value["__ref__"]
                if ref not in instances:
                    raise RigConfigError(f"Reference '{ref}' not available yet.")
                return instances[ref]
            # Detect serial configs (simple dict with port + baudrate)
            if "port" in value and ("baudrate" in value or "baud" in value):
                try:
                    import serial  # type: ignore
                    serial_kwargs = {k: v for k, v in value.items()}
                    return serial.Serial(**serial_kwargs)
                except Exception as exc:  # pragma: no cover
                    raise RigConfigError(f"pyserial is required for serial configs: {exc}") from exc
            return {k: self._resolve_value(v, instances) for k, v in value.items()}
        return value

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration structure.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
            
        Raises:
            RigConfigError: If configuration is invalid.
        """
        if config.get("schema_version") not in (None, SCHEMA_VERSION):
            raise RigConfigError("Unsupported schema_version in rig configuration.")
        devices = config.get("devices")
        if not isinstance(devices, dict):
            raise RigConfigError("Configuration must contain a 'devices' mapping.")
        try:
            pipette_count = max(
                1,
                int(config.get("pipette_count", 1)),
            )
        except (TypeError, ValueError):
            raise RigConfigError(
                "Invalid pipette_count in rig configuration."
            )
        for slot in DEVICE_SLOTS:
            if slot not in devices:
                raise RigConfigError(
                    f"Configuration missing required slot '{slot}'."
                )
            if slot in PIPETTE_DEVICE_SLOTS:
                spec = devices[slot]
                if pipette_count > 1:
                    if not isinstance(spec, list):
                        raise RigConfigError(
                            f"'{slot}' must contain {pipette_count} "
                            "device configurations."
                        )

                    if len(spec) != pipette_count:
                        raise RigConfigError(
                            f"'{slot}' contains {len(spec)} devices, "
                            f"but this rig declares {pipette_count} pipettes."
                        )

    def make_config_path(self, name: str) -> Path:
        """
        Generate a file path for a configuration name.

        Args:
            name (str): Configuration name.

        Returns:
            Path: Generated file path.
        """
        return self.config_dir / f"{_slugify(name)}.json"

    def get_device_options(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get available device options.

        Returns:
            Dict[str, List[Dict[str, Any]]]: Device options mapping.
        """
        return DEVICE_OPTIONS

    def build_empty_template(self) -> Dict[str, Any]:
        """
        Get available device options.

        Returns:
            Dict[str, List[Dict[str, Any]]]: Device options mapping.
        """
        return {
            "name": "New Rig",
            "schema_version": SCHEMA_VERSION,
            "pipette_count": 1,
            "calibration_file": None,
            "patch_file": None,
            "protocol_file": None,
            "calibration": _empty_calibration(),
            "ai_features": _default_ai_features(),
            "devices": {slot: {} for slot in DEVICE_SLOTS},
        }

    def _load_overlay_configs(self, config: Dict[str, Any]) -> None:
        calibration = None
        calibration_fallback: Dict[str, Any] = {}
        try:
            from patcherbot.devices.manipulator.CalibrationConfig import CalibrationConfig
            calibration = CalibrationConfig(name="Calibration")
        except Exception as exc:
            LOGGER.warning("CalibrationConfig import failed; using YAML fallback: %s", exc)

        PatchConfig = None
        try:
            from patcherbot.interface.patchConfig import PatchConfig as PatchConfigType
            PatchConfig = PatchConfigType
        except Exception as exc:
            LOGGER.warning("Patch overlay disabled because imports failed: %s", exc)

        ProtocolConfig = None
        try:
            from patcherbot.interface.protocolConfig import ProtocolConfig as ProtocolConfigType
            ProtocolConfig = ProtocolConfigType
        except Exception as exc:
            LOGGER.warning("Protocol overlay disabled because imports failed: %s", exc)

        has_pressure_overrides = False
        cal_file = config.get("calibration_file")
        if cal_file:
            cal_path = self.cal_config_dir / cal_file
            if cal_path.exists():
                try:
                    if calibration is not None:
                        calibration.from_file(str(cal_path))
                    else:
                        with cal_path.open("r", encoding="utf-8") as f:
                            loaded = yaml.safe_load(f) or {}
                        if isinstance(loaded, dict):
                            calibration_fallback.update({k: v for k, v in loaded.items() if v is not None})
                    has_pressure_overrides = True
                except Exception as exc:
                    LOGGER.warning("Failed to load calibration file %s: %s", cal_path, exc)
            else:
                LOGGER.warning("Calibration file not found: %s (using defaults)", cal_path)

        inline_cal = config.get("calibration")
        if isinstance(inline_cal, dict):
            cleaned = {k: v for k, v in inline_cal.items() if v is not None}
            if calibration is not None:
                calibration.from_dict(cleaned)
            else:
                calibration_fallback.update(cleaned)
            if any(
                key in cleaned
                for key in ("native_zero", "native_per_mbar", "reader_offset", "reader_scale")
            ):
                has_pressure_overrides = True

        if "ai_features" in config:
            LOGGER.warning(
                "Rig-level 'ai_features' is ignored. Set calibration.use_ai_features in the calibration config file."
            )

        has_patch_overlay = False
        patch = PatchConfig(name="Patch") if PatchConfig is not None else None
        if patch is not None:
            patch_file = config.get("patch_file")
            if patch_file:
                patch_path = self.patch_config_dir / patch_file
                if patch_path.exists():
                    try:
                        patch.from_file(str(patch_path))
                        has_patch_overlay = True
                    except Exception as exc:
                        LOGGER.warning("Failed to load patch file %s: %s", patch_path, exc)
                else:
                    LOGGER.warning("Patch file not found: %s (using defaults)", patch_path)

            inline_patch = config.get("patch")
            if isinstance(inline_patch, dict):
                cleaned = {k: v for k, v in inline_patch.items() if v is not None}
                patch.from_dict(cleaned)
                if cleaned:
                    has_patch_overlay = True

        protocol = ProtocolConfig(name="Protocols") if ProtocolConfig is not None else None
        if protocol is not None:
            protocol_file = config.get("protocol_file")
            if protocol_file:
                protocol_path = self.protocol_config_dir / protocol_file
                if protocol_path.exists():
                    try:
                        protocol.from_file(str(protocol_path))
                    except Exception as exc:
                        LOGGER.warning("Failed to load protocol file %s: %s", protocol_path, exc)
                else:
                    LOGGER.warning("Protocol file not found: %s (using defaults)", protocol_path)
            else:
                LOGGER.warning("No protocol_file specified; using defaults")

            inline_protocol = config.get("protocol")
            if isinstance(inline_protocol, dict):
                cleaned = {k: v for k, v in inline_protocol.items() if v is not None}
                protocol.from_dict(cleaned)

        if calibration is not None:
            resolved_calibration = calibration.to_dict()
        else:
            resolved_calibration = _empty_calibration()
            resolved_calibration.update(calibration_fallback)
            resolved_calibration.setdefault("use_ai_features", False)
        config["_resolved_calibration"] = resolved_calibration
        if has_patch_overlay and patch is not None:
            config["_resolved_patch"] = patch.to_dict()
        if protocol is not None:
            config["_resolved_protocol"] = protocol.to_dict()
        config.pop("ai_features", None)
        config["_has_pressure_calibration_overrides"] = has_pressure_overrides

    def _apply_pressure_calibration(self, config: Dict[str, Any], devices_cfg: Dict[str, Any]) -> None:
        if not bool(config.get("_has_pressure_calibration_overrides", False)):
            return
        calibration = config.get("_resolved_calibration")
        if not isinstance(calibration, dict):
            calibration = config.get("calibration")
        if not isinstance(calibration, dict):
            return
        pressure_cfg = devices_cfg.get("pressure")
        if not isinstance(pressure_cfg, dict):
            return
        params = pressure_cfg.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        class_path = pressure_cfg.get("class") or pressure_cfg.get("class_path") or ""

        updates: Dict[str, Any] = {}
        if calibration.get("native_zero") is not None:
            updates["native_zero"] = calibration["native_zero"]
        if calibration.get("native_per_mbar") is not None:
            updates["native_per_mbar"] = calibration["native_per_mbar"]
        if "IBBPressureController" not in str(class_path):
            if calibration.get("reader_offset") is not None:
                updates["sensor_offset"] = calibration["reader_offset"]
            if calibration.get("reader_scale") is not None:
                updates["sensor_scale"] = calibration["reader_scale"]

        if updates:
            merged = dict(params)
            merged.update(updates)
            pressure_cfg["params"] = merged
