import sys
import os
import csv
import bisect
import json
import logging
from pathlib import Path
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QComboBox,
    QFileDialog, QShortcut, QHBoxLayout, QFrame, QSlider, QMessageBox, QSizePolicy, QStackedWidget,
    QLineEdit, QGraphicsDropShadowEffect, QDialog, QProgressBar
)
from PyQt5.QtGui import QPixmap, QColor, QKeySequence
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
from collections import deque
import re

try:
    import cv2
except Exception:  # pragma: no cover - dependency may be unavailable in some environments
    cv2 = None

logger = logging.getLogger(__name__)


def _bootstrap_repo_import_path():
    """Ensure the repository root is importable when running this file directly."""
    try:
        repo_root = Path(__file__).resolve().parents[4]
    except Exception:
        return
    candidate_paths = [
        repo_root,
        repo_root / "patcherbot" / "deepLearning",
    ]
    for candidate in candidate_paths:
        try:
            if not candidate.exists():
                continue
        except Exception:
            continue
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


_bootstrap_repo_import_path()

CELL_DETECTOR_PIDNET = "pidnet"
CELL_DETECTOR_YOLO = "yolo"


def _create_cell_detector(model_name):
    """Create the selected cell detector without changing the PIDNet path."""
    normalized_name = str(model_name).strip().lower()
    if normalized_name == CELL_DETECTOR_PIDNET:
        from patcherbot.deepLearning.CellDetector import CellDetector2
        return CellDetector2(model_type="pidnet")
    if normalized_name == CELL_DETECTOR_YOLO:
        from patcherbot.deepLearning.CellDetector import CellDetectorYOLO1
        return CellDetectorYOLO1()
    raise ValueError(f"Unsupported cell detector model: {model_name!r}")


def _cell_detector_display_name(model_name):
    normalized_name = str(model_name).strip().lower()
    if normalized_name == CELL_DETECTOR_PIDNET:
        return "CellDetector2 PIDNet"
    if normalized_name == CELL_DETECTOR_YOLO:
        return "CellDetectorYOLO1"
    return str(model_name)


# ------------------- 3D Mesh Creation Functions -------------------

def create_cylinder_mesh(radius=0.5, height=5.0, sectors=32):
    """
    Create vertices and faces for a cylinder.
    """
    vertices = []
    faces = []

    # Bottom circle
    for i in range(sectors):
        angle = 2 * np.pi * i / sectors
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = 0
        vertices.append([x, y, z])

    # Top circle
    for i in range(sectors):
        angle = 2 * np.pi * i / sectors
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = height
        vertices.append([x, y, z])

    # Center points
    vertices.append([0, 0, 0])      # Bottom center
    vertices.append([0, 0, height]) # Top center

    bottom_center = len(vertices) - 2
    top_center = len(vertices) - 1

    # Side faces
    for i in range(sectors):
        next_i = (i + 1) % sectors
        # Bottom to top
        faces.append([i, next_i, sectors + next_i])
        faces.append([i, sectors + next_i, sectors + i])

    # Bottom faces
    for i in range(sectors):
        next_i = (i + 1) % sectors
        faces.append([bottom_center, next_i, i])

    # Top faces
    for i in range(sectors):
        next_i = (i + 1) % sectors
        faces.append([top_center, sectors + i, sectors + next_i])

    vertices = np.array(vertices)
    faces = np.array(faces)

    return vertices, faces

def create_box_mesh(width=5.0, depth=10.0, height=0.5):
    """
    Create vertices and faces for a rectangular prism.
    Origin is set to the top center of the box.
    """
    w = width / 2
    d = depth / 2
    h = height

    vertices = np.array([
        [-w, -d, -h],  # 0: Bottom Front Left
        [w, -d, -h],   # 1: Bottom Front Right
        [w, d, -h],    # 2: Bottom Back Right
        [-w, d, -h],   # 3: Bottom Back Left
        [-w, -d, 0],    # 4: Top Front Left
        [w, -d, 0],     # 5: Top Front Right
        [w, d, 0],      # 6: Top Back Right
        [-w, d, 0],     # 7: Top Back Left
    ])

    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # Bottom
        [4, 5, 6], [4, 6, 7],  # Top
        [0, 1, 5], [0, 5, 4],  # Front
        [1, 2, 6], [1, 6, 5],  # Right
        [2, 3, 7], [2, 7, 6],  # Back
        [3, 0, 4], [3, 4, 7],  # Left
    ])

    return vertices, faces

def create_sphere_mesh(radius=0.5, sectors=32, stacks=16):
    """
    Create vertices and faces for a sphere.
    """
    vertices = []
    faces = []

    for stack in range(stacks + 1):
        phi = np.pi / 2 - stack * np.pi / stacks  # from pi/2 to -pi/2
        y = radius * np.sin(phi)
        r = radius * np.cos(phi)

        for sector in range(sectors + 1):
            theta = 2 * np.pi * sector / sectors
            x = r * np.cos(theta)
            z = r * np.sin(theta)
            vertices.append([x, y, z])

    for stack in range(stacks):
        for sector in range(sectors):
            first = stack * (sectors + 1) + sector
            second = first + sectors + 1

            faces.append([first, second, first + 1])
            faces.append([second, second + 1, first + 1])

    vertices = np.array(vertices)
    faces = np.array(faces)

    return vertices, faces

def rotate_vertices(vertices, angle_degrees, axis='x'):
    """
    Rotate vertices by a specified angle around a given axis.

    :param vertices: Nx3 numpy array of vertex coordinates.
    :param angle_degrees: Angle in degrees to rotate.
    :param axis: Axis to rotate around ('x', 'y', or 'z').
    :return: Rotated vertices as Nx3 numpy array.
    """
    angle_rad = np.deg2rad(angle_degrees)
    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad), np.cos(angle_rad)]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad)],
            [0, 1, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad)]
        ])
    elif axis == 'z':
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Invalid axis. Choose from 'x', 'y', or 'z'.")

    rotated_vertices = vertices.dot(rotation_matrix.T)
    return rotated_vertices

# ------------------- 3D Visualization Widget -------------------

class GLViewWidgetWithGrid(gl.GLViewWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackgroundColor('k')  # Set background to black

        # Set initial isometric view
        self.restore_isometric_view()

        # Add grid to XY plane
        self.add_grid()

        # Add XYZ axes
        self.add_axes()

        # Initialize objects
        self.init_objects()

    def add_grid(self):
        # Create grid lines on the XY plane with millimeter squares
        grid_size = 100  # Define the extent of the grid (100 mm in each direction)
        step = 1  # 1 mm squares

        # Create lines parallel to X-axis
        for y in np.arange(-grid_size, grid_size + step, step):
            pts = np.array([[-grid_size, y, 0], [grid_size, y, 0]])
            plt = gl.GLLinePlotItem(pos=pts, color=(1, 1, 1, 1), width=1, antialias=True)
            self.addItem(plt)

        # Create lines parallel to Y-axis
        for x in np.arange(-grid_size, grid_size + step, step):
            pts = np.array([[x, -grid_size, 0], [x, grid_size, 0]])
            plt = gl.GLLinePlotItem(pos=pts, color=(1, 1, 1, 1), width=1, antialias=True)
            self.addItem(plt)

    def add_axes(self):
        # Define axis length
        axis_length = 50  # 50 mm

        # X-axis (Red)
        x_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [axis_length, 0, 0]]),
                                color=(1, 0, 0, 1), width=2, antialias=True)
        self.addItem(x_axis)
        
        # X-axis label
        x_label = gl.GLTextItem(text='X', color=(1, 0, 0, 1))
        x_label.translate(axis_length + 2, 0, 0)  # Slightly beyond the axis tip
        self.addItem(x_label)

        # Y-axis (Green)
        y_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, axis_length, 0]]),
                                color=(0, 1, 0, 1), width=2, antialias=True)
        self.addItem(y_axis)
        
        # Y-axis label
        y_label = gl.GLTextItem(text='Y', color=(0, 1, 0, 1))
        y_label.translate(0, axis_length + 2, 0)  # Slightly beyond the axis tip
        self.addItem(y_label)

        # Z-axis (Blue)
        z_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 0, axis_length]]),
                                color=(0, 0, 1, 1), width=2, antialias=True)
        self.addItem(z_axis)
        
        # Z-axis label
        z_label = gl.GLTextItem(text='Z', color=(0, 0, 1, 1))
        z_label.translate(0, 0, axis_length + 2)  # Slightly beyond the axis tip
        self.addItem(z_label)


    def restore_isometric_view(self):
        # Set camera elevation and azimuth for isometric view
        self.opts['azimuth'] = 45  # Rotate 45 degrees around Z-axis
        self.opts['elevation'] = 30  # Tilt 30 degrees from horizontal
        self.update()

    def init_objects(self):
        # Create Microscope as a cylinder
        cyl_vertices, cyl_faces = create_cylinder_mesh(radius=0.5, height=5.0, sectors=32)
        self.microscope_mesh = gl.GLMeshItem(vertexes=cyl_vertices, faces=cyl_faces,
                                            smooth=True, color=(1, 0, 0, 1), shader='shaded', drawEdges=False)
        self.addItem(self.microscope_mesh)

        # Create Stage as a box (rectangular prism) with origin at top center
        box_vertices, box_faces = create_box_mesh(width=5.0, depth=10.0, height=0.5)
        self.stage_mesh = gl.GLMeshItem(vertexes=box_vertices, faces=box_faces,
                                       smooth=False, color=(0, 1, 0, 1), shader='shaded', drawEdges=False)
        self.addItem(self.stage_mesh)

        # Create Pipette as a rotated cylinder
        pipette_vertices, pipette_faces = create_cylinder_mesh(radius=0.2, height=3.0, sectors=32)
        # Rotate 25 degrees with respect to the XY plane (i.e., 65 degrees away from the Z-axis)
        pipette_vertices = rotate_vertices(pipette_vertices, 65, 'y')
        self.pipette_mesh = gl.GLMeshItem(vertexes=pipette_vertices, faces=pipette_faces,
                                         smooth=True, color=(0, 0, 1, 1), shader='shaded', drawEdges=False)
        # Adjust pipette position so that its bottom is at the origin
        pipette_change = np.array([3.0 * np.cos(np.radians(65)), 0, 0])
        self.pipette_mesh.translate(*pipette_change)
        pipette_change = np.array([3.0 * (np.cos(np.radians(65))-np.sin(np.radians(65))), 0, 0])
        self.pipette_mesh.translate(*pipette_change)
        self.addItem(self.pipette_mesh)

    def update_objects(self, microscope_z, stage_pos, pipette_pos):
        """
        Update the positions of Stage and Pipette.

        :param microscope_z: Z-coordinate for Microscope (fixed X and Y at 0).
        :param stage_pos: (x, y) position for Stage in the XY-plane.
        :param pipette_pos: (x, y, z) position for Pipette.
        """
        # Update Microscope position (fixed X and Y, variable Z)
        self.microscope_mesh.resetTransform()
        self.microscope_mesh.translate(0, 0, microscope_z)

        # Update Stage position (only in XY-plane, Z is fixed at 0)
        self.stage_mesh.resetTransform()
        self.stage_mesh.translate(stage_pos[0], stage_pos[1], 0)

        # Update Pipette position (in XYZ)
        self.pipette_mesh.resetTransform()
        self.pipette_mesh.translate(pipette_pos[0], pipette_pos[1], pipette_pos[2])

# ------------------- Data Manager -------------------

class DataManager:
    def __init__(self, scaling_factor=1/1000):
        self.image_paths = []
        self.timestamps = []
        self.image_index = []
        self.aux_image_paths = []
        self.aux_timestamps = []
        self.aux_image_index = []
        self.movement_data = []
        self.graph_data = []
        self.directory = None
        self.scaling_factor = scaling_factor  # micrometers to millimeters
        self.init_image_time = 0.0
        self.state_windows = []
        self._state_window_starts = []

    def load_directory(self, directory):
        self.directory = directory
        camera_frames_dir = os.path.join(directory, 'camera_frames')
        aux_camera_frames_dir = os.path.join(directory, 'aux_camera_frames')
        movement_file_path = os.path.join(directory, 'movement_recording.csv')
        graph_file_path = os.path.join(directory, 'graph_recording.csv')

        # Load images
        self.load_images(camera_frames_dir)
        self.load_aux_images(aux_camera_frames_dir)

        # Load movement data
        if os.path.exists(movement_file_path):
            self.load_movement_data(movement_file_path)
        else:
            raise FileNotFoundError("movement_recording.csv not found in the selected directory.")

        # Load graph data
        if os.path.exists(graph_file_path):
            self.load_graph_data(graph_file_path)
        else:
            raise FileNotFoundError("graph_recording.csv not found in the selected directory.")

        valid_start, valid_end = self._get_valid_recording_window()
        folder_name = os.path.basename(os.path.normpath(directory))
        self.load_state_attempts(folder_name, valid_start, valid_end)

    def _get_valid_recording_window(self):
        if self.graph_data:
            return self.graph_data[0]['time'], self.graph_data[-1]['time']
        if self.timestamps:
            return self.timestamps[0], self.timestamps[-1]
        return 0.0, 0.0

    @staticmethod
    def _slugify_state_name(name):
        cleaned = name.strip().lower().replace(" ", "_")
        slug = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in cleaned)
        slug = slug.strip('_')
        return slug or 'state'

    def load_state_attempts(self, rig_recorder_data_folder, valid_start, valid_end):
        self.state_windows = []
        self._state_window_starts = []

        day_token = rig_recorder_data_folder.split('-', 1)[0]
        tolerance = 0.5

        state_roots = self._candidate_state_roots()
        if not state_roots:
            return

        seen_json_paths = set()
        for state_root in state_roots:
            for day_dir in sorted(state_root.glob(f"{day_token}*")):
                if not day_dir.is_dir():
                    continue
                for attempt_dir in sorted(day_dir.glob("attempt_*")):
                    if not attempt_dir.is_dir():
                        continue
                    for json_path in sorted(attempt_dir.glob("*.json")):
                        try:
                            json_key = str(json_path.resolve())
                        except OSError:
                            json_key = str(json_path)
                        if json_key in seen_json_paths:
                            continue
                        seen_json_paths.add(json_key)
                        try:
                            with open(json_path, "r", encoding="utf-8") as fh:
                                payload = json.load(fh)
                        except (OSError, json.JSONDecodeError):
                            continue

                        started = payload.get("started")
                        finished = payload.get("finished")
                        if started is None or finished is None:
                            continue
                        try:
                            started = float(started)
                            finished = float(finished)
                        except (TypeError, ValueError):
                            continue
                        if finished <= started:
                            continue
                        if finished < (valid_start - tolerance) or started > (valid_end + tolerance):
                            continue

                        stem = json_path.stem
                        parts = stem.split("_")
                        if len(parts) >= 3:
                            state_name = "_".join(parts[1:-1])
                        else:
                            state_name = stem
                        state_name = self._slugify_state_name(state_name)
                        outcome = payload.get("outcome")
                        try:
                            outcome_code = int(outcome) if outcome is not None else None
                        except (TypeError, ValueError):
                            outcome_code = None

                        self.state_windows.append((started, finished, state_name, outcome_code))

        self.state_windows.sort(key=lambda window: window[0])
        self._state_window_starts = [window[0] for window in self.state_windows]

    def _candidate_state_roots(self):
        """Return candidate state_recorder_data roots, preferring paths near the selected recording."""
        roots = []
        seen = set()

        def add_root(path):
            try:
                resolved = path.resolve()
            except OSError:
                resolved = path
            key = str(resolved)
            if key in seen:
                return
            if path.exists() and path.is_dir():
                roots.append(path)
                seen.add(key)

        if self.directory:
            recording_path = Path(self.directory)
            for ancestor in recording_path.parents:
                add_root(ancestor / "state_recorder_data")

        add_root(Path("experiments/Data/state_recorder_data"))
        return roots

    def get_active_state_for_timestamp(self, timestamp):
        if not self.state_windows:
            return None, None

        idx = bisect.bisect_right(self._state_window_starts, timestamp)
        for j in range(idx - 1, -1, -1):
            started, finished, state_name, outcome_code = self.state_windows[j]
            if started <= timestamp <= finished:
                return state_name, outcome_code
        return None, None

    def _parse_image_directory(self, directory, allow_empty=False, empty_message=None):
        if not os.path.exists(directory):
            if allow_empty:
                return [], [], []
            raise FileNotFoundError(f"{directory} does not exist.")

        image_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.webp')]
        if not image_files:
            if allow_empty:
                return [], [], []
            raise FileNotFoundError(empty_message or f"No .webp images found in {directory}.")

        parsed_images = []
        for img_path in image_files:
            idx, ts = self.extract_image_data(img_path)
            parsed_images.append((ts, idx, img_path))
        parsed_images.sort(key=lambda item: (item[0], item[1]))
        paths = [item[2] for item in parsed_images]
        indices = [item[1] for item in parsed_images]
        timestamps = [item[0] for item in parsed_images]
        return paths, indices, timestamps

    def load_images(self, directory):
        paths, indices, timestamps = self._parse_image_directory(
            directory,
            allow_empty=False,
            empty_message="No .webp images found in the camera_frames directory."
        )
        self.image_paths = paths
        self.image_index = indices
        self.timestamps = timestamps
        self.init_image_time = self.timestamps[0] if self.timestamps else 0.0

    def load_aux_images(self, directory):
        self.aux_image_paths = []
        self.aux_image_index = []
        self.aux_timestamps = []
        if not os.path.isdir(directory):
            return

        paths, indices, timestamps = self._parse_image_directory(directory, allow_empty=True)
        if paths:
            self.aux_image_paths = paths
            self.aux_image_index = indices
            self.aux_timestamps = timestamps

    @staticmethod
    def _parse_numeric_list(value):
        """
        Parse a string representation of a numeric list, handling optional np.float64 wrappers.
        """
        if value is None:
            return []
        value = value.strip()
        if value.startswith('[') and value.endswith(']'):
            value = value[1:-1]
        if not value:
            return []
        cleaned = re.sub(r'np\.float64\((.*?)\)', r'\1', value)
        return [float(item) for item in cleaned.split(',') if item.strip()]

    def has_aux_images(self):
        return bool(self.aux_image_paths)

    def get_aux_image_for_timestamp(self, timestamp):
        if not self.aux_timestamps:
            return None

        idx = bisect.bisect_left(self.aux_timestamps, timestamp)
        if idx == 0:
            return self.aux_image_paths[0]
        if idx >= len(self.aux_timestamps):
            return self.aux_image_paths[-1]

        before_ts = self.aux_timestamps[idx - 1]
        after_ts = self.aux_timestamps[idx]
        if abs(timestamp - before_ts) <= abs(after_ts - timestamp):
            return self.aux_image_paths[idx - 1]
        return self.aux_image_paths[idx]

    def load_movement_data(self, file_path):
        self.movement_data.clear()
        try:
            with open(file_path, mode='r') as file:
                first_line = file.readline().strip()
                if ';' in first_line:
                    file.seek(0)
                    reader = csv.reader(file, delimiter=';')
                    rows = list(reader)
                    if not rows:
                        raise ValueError("No data found in movement file.")

                    header_candidates = {'timestamp', 'st_x', 'st_y', 'st_z', 'pi_x', 'pi_y', 'pi_z'}
                    row_lower = [cell.strip().lower() for cell in rows[0]]
                    has_header = any(cell in header_candidates for cell in row_lower)
                    if has_header:
                        header = row_lower
                        data_rows = rows[1:]
                        idx_map = {name: idx for idx, name in enumerate(header)}
                        required_fields = ['timestamp', 'st_x', 'st_y', 'st_z', 'pi_x', 'pi_y', 'pi_z']
                        if not all(field in idx_map for field in required_fields):
                            raise ValueError("Movement file header missing required columns.")
                        for row in data_rows:
                            if len(row) < len(header):
                                continue
                            try:
                                time_value = float(row[idx_map['timestamp']])
                                st_x = float(row[idx_map['st_x']]) * self.scaling_factor
                                st_y = float(row[idx_map['st_y']]) * self.scaling_factor
                                st_z = float(row[idx_map['st_z']]) * self.scaling_factor
                                pi_x = float(row[idx_map['pi_x']]) * self.scaling_factor
                                pi_y = float(row[idx_map['pi_y']]) * self.scaling_factor
                                pi_z = float(row[idx_map['pi_z']]) * self.scaling_factor
                            except Exception:
                                continue
                            self.movement_data.append({
                                'time': time_value,
                                'microscope_z': -(st_z),
                                'stage': (-st_x, -st_y),
                                'pipette': (-pi_x, -pi_y, -(pi_z - 1.365))
                            })
                    else:
                        for row in rows:
                            if len(row) < 7:
                                continue
                            try:
                                time_value = float(row[0])
                                st_x = float(row[1]) * self.scaling_factor
                                st_y = float(row[2]) * self.scaling_factor
                                st_z = float(row[3]) * self.scaling_factor
                                pi_x = float(row[4]) * self.scaling_factor
                                pi_y = float(row[5]) * self.scaling_factor
                                pi_z = float(row[6]) * self.scaling_factor
                            except Exception:
                                continue
                            self.movement_data.append({
                                'time': time_value,
                                'microscope_z': -(st_z),
                                'stage': (-st_x, -st_y),
                                'pipette': (-pi_x, -pi_y, -(pi_z - 1.365))
                            })
                else:
                    # Assume old format with space-delimited values and colon-split key-value pairs
                    file.seek(0)
                    reader = csv.reader(file, delimiter=' ')
                    for line in reader:
                        parts = [part for part in line if part]
                        if len(parts) < 7:
                            continue  # Skip invalid lines
                        try:
                            time_value = float(parts[0].split(':')[1])
                            st_x = float(parts[1].split(':')[1]) * self.scaling_factor
                            st_y = float(parts[2].split(':')[1]) * self.scaling_factor
                            st_z = float(parts[3].split(':')[1]) * self.scaling_factor
                            pi_x = float(parts[4].split(':')[1]) * self.scaling_factor
                            pi_y = float(parts[5].split(':')[1]) * self.scaling_factor
                            pi_z = float(parts[6].split(':')[1]) * self.scaling_factor

                            self.movement_data.append({
                                'time': time_value,
                                'microscope_z': -(st_z),
                                'stage': (-st_x, -st_y),
                                'pipette': (-pi_x, -pi_y, -(pi_z - 1.365))
                            })
                        except Exception:
                            continue

            if not self.movement_data:
                raise ValueError("No valid movement data found in the selected file.")

            # Sort movement data by time
            self.movement_data.sort(key=lambda x: x['time'])
            # Align movement data's time with image data's time
            if self.timestamps:
                initial_image_time = self.timestamps[0]
                # Shift movement_data's time to align with image_data's initial timestamp
                # Assuming that movement_data starts after image_data's start
                self.movement_data = [
                    {k: v if k != 'time' else v - 0 for k, v in entry.items()}
                    for entry in self.movement_data
                ]

        except Exception as e:
            raise IOError(f"Error loading movement data: {e}")

    def get_closest_movement_entry(self, timestamp):
        """
        Retrieve the movement data entry closest to the given timestamp.
        """
        if not self.movement_data:
            return None

        times = [entry['time'] for entry in self.movement_data]
        index = bisect.bisect_left(times, timestamp)

        if index == 0:
            return self.movement_data[0]
        if index == len(times):
            return self.movement_data[-1]

        before = self.movement_data[index - 1]
        after = self.movement_data[index]

        if abs(after['time'] - timestamp) < abs(before['time'] - timestamp):
            return after
        else:
            return before

    def load_graph_data(self, file_path):
        self.graph_data.clear()
        try:
            with open(file_path, mode='r') as file:
                first_line = file.readline().strip()
                if ';' in first_line:
                    file.seek(0)
                    reader = csv.reader(file, delimiter=';')
                    rows = list(reader)
                    if not rows:
                        raise ValueError("No data found in graph file.")

                    header_candidates = {'timestamp', 'pressure', 'resistance', 'current', 'voltage'}
                    row_lower = [cell.strip().lower() for cell in rows[0]]
                    has_header = any(cell in header_candidates for cell in row_lower)
                    if has_header:
                        header = row_lower
                        data_rows = rows[1:]
                        idx_map = {name: idx for idx, name in enumerate(header)}
                        required_fields = ['timestamp', 'pressure', 'resistance', 'current', 'voltage']
                        if not all(field in idx_map for field in required_fields):
                            raise ValueError("Graph file header missing required columns.")
                        for row in data_rows:
                            if len(row) < len(header):
                                continue
                            try:
                                time_value = float(row[idx_map['timestamp']])
                                pressure_value = float(row[idx_map['pressure']])
                                resistance_value = float(row[idx_map['resistance']])
                                current_value = self._parse_numeric_list(row[idx_map['current']])
                                voltage_value = self._parse_numeric_list(row[idx_map['voltage']])
                            except Exception:
                                continue
                            self.graph_data.append({
                                'time': time_value,
                                'pressure': pressure_value,
                                'resistance': resistance_value,
                                'current': current_value,
                                'voltage': voltage_value
                            })
                    else:
                        for row in rows:
                            if len(row) < 5:
                                continue
                            try:
                                time_value = float(row[0])
                                pressure_value = float(row[1])
                                resistance_value = float(row[2])
                                current_value = self._parse_numeric_list(row[3])
                                voltage_value = self._parse_numeric_list(row[4])
                            except Exception:
                                continue
                            self.graph_data.append({
                                'time': time_value,
                                'pressure': pressure_value,
                                'resistance': resistance_value,
                                'current': current_value,
                                'voltage': voltage_value
                            })
                else:
                    # Assume old format with colon-separated values
                    file.seek(0)
                    for line in file:
                        parts = [part for part in line.strip().split(':') if part]
                        if len(parts) < 6:
                            continue  # Skip invalid lines
                        try:
                            time_value = float(parts[1].split(' ')[0])
                            pressure_value = float(parts[2].split(' ')[0])
                            resistance_value = float(parts[3].split(' ')[0])
                            current_data = parts[4].split('[')[-1].split(']')[0]
                            current_value = [float(x) for x in current_data.split(',') if x]
                            voltage_data = parts[5].split('[')[-1].split(']')[0]
                            voltage_value = [float(x) for x in voltage_data.split(',') if x]
                            self.graph_data.append({
                                'time': time_value,
                                'pressure': pressure_value,
                                'resistance': resistance_value,
                                'current': current_value,
                                'voltage': voltage_value
                            })
                        except Exception:
                            continue

            self.graph_data.sort(key=lambda x: x['time'])

        except Exception as e:
            raise IOError(f"Error loading graph data: {e}")

    def extract_image_data(self, image_path):
        """
        Extract the image index and timestamp from the image filename.
        Assumes filenames are in the format 'index_timestamp.webp'.
        """
        try:
            filename = os.path.basename(image_path)
            index_str = int(filename.split('_')[0])
            timestamp_str = float(filename.split('_')[1].rsplit('.', 1)[0])
        except Exception as e:
            print(f"Error extracting index and timestamp from {image_path}: {e}")
            index_str, timestamp_str = 0, 0.0  # Default values in case of error
        return index_str, timestamp_str


class DirectoryLoadWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str, str)
    progress = QtCore.pyqtSignal(str)

    def __init__(self, directory, scaling_factor):
        super().__init__()
        self.directory = directory
        self.scaling_factor = scaling_factor

    @QtCore.pyqtSlot()
    def run(self):
        try:
            self.progress.emit("Loading camera, movement, and graph recordings...")
            manager = DataManager(scaling_factor=self.scaling_factor)
            manager.load_directory(self.directory)
            self.finished.emit(manager)
        except FileNotFoundError as exc:
            self.error.emit("File Not Found", str(exc))
        except IOError as exc:
            self.error.emit("IO Error", str(exc))
        except Exception as exc:
            self.error.emit("Error", f"An unexpected error occurred: {exc}")


class InferenceEnableWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(str)

    def __init__(
        self,
        *,
        enable_pipette_detector,
        enable_cell_detector,
        enable_pipette_focuser,
        cell_detector_model=CELL_DETECTOR_PIDNET,
    ):
        super().__init__()
        self.enable_pipette_detector = bool(enable_pipette_detector)
        self.enable_cell_detector = bool(enable_cell_detector)
        self.enable_pipette_focuser = bool(enable_pipette_focuser)
        self.cell_detector_model = str(cell_detector_model).strip().lower()

    @QtCore.pyqtSlot()
    def run(self):
        result = {
            "pipette_detector": None,
            "cell_detector": None,
            "pipette_focuser": None,
            "pipette_detector_init_failed": False,
            "cell_detector_init_failed": False,
            "pipette_focuser_init_failed": False,
            "warnings": [],
        }
        try:
            if self.enable_pipette_detector:
                self.progress.emit("Initializing pipette detector...")
                try:
                    from patcherbot.deepLearning.pipetteDetector import PipetteDetectorYOLO1
                    result["pipette_detector"] = PipetteDetectorYOLO1()
                except Exception as exc:
                    result["pipette_detector_init_failed"] = True
                    result["warnings"].append(
                        f"PipetteDetectorYOLO1 init failed; disabling pipette overlay: {exc}"
                    )

            if self.enable_cell_detector:
                detector_name = _cell_detector_display_name(self.cell_detector_model)
                self.progress.emit(f"Initializing {detector_name}...")
                try:
                    result["cell_detector"] = _create_cell_detector(self.cell_detector_model)
                except Exception as exc:
                    result["cell_detector_init_failed"] = True
                    result["warnings"].append(
                        f"{detector_name} init failed; disabling cell overlay: {exc}"
                    )

            if self.enable_pipette_focuser:
                self.progress.emit("Initializing pipette focuser...")
                try:
                    from patcherbot.deepLearning.pipetteFocuser import PipetteFocuser
                    result["pipette_focuser"] = PipetteFocuser()
                except Exception as exc:
                    result["pipette_focuser_init_failed"] = True
                    result["warnings"].append(
                        f"PipetteFocuser init failed; disabling focus readout: {exc}"
                    )

            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(f"Inference initialization failed: {exc}")

# ------------------- Main Integrated Window -------------------

class IntegratedTimeline(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Integrated Timeline and 3D Visualization")
        self.setGeometry(50, 50, 1600, 900)  # Adjusted size for better visibility

        # Central Widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Main Layout
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)

        # Top Layout: Image and Graphs/3D Plot
        self.top_layout = QHBoxLayout()
        self.top_layout.setSpacing(10)
        self.main_layout.addLayout(self.top_layout, stretch=8)  # Allocate most space to top_layout

        # Image Display
        self.image = QLabel("No Image loaded yet!")
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.image_frame = QFrame()
        image_layout = QVBoxLayout(self.image_frame)
        image_layout.addWidget(self.image)
        self.image_frame.setFrameShape(QFrame.StyledPanel)
        self.image_frame.setFrameShadow(QFrame.Sunken)
        self.image_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.top_layout.addWidget(self.image_frame, stretch=1)  # Image takes 1 part

        # Graphs and 3D Plot
        self.graph_stack = QStackedWidget()
        self.graph_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 2D Graphs Widget
        self.graph_frame = QFrame()
        self.graph_frame.setFrameShape(QFrame.StyledPanel)
        self.graph_frame.setFrameShadow(QFrame.Sunken)
        self.graph_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.graph_frame.setMinimumSize(600, 400)  # Set minimum size for graph_frame

        self.graph_layout = QVBoxLayout(self.graph_frame)
        self.graph_layout.setContentsMargins(5, 5, 5, 5)
        self.graph_layout.setSpacing(5)

        # GraphicsLayoutWidget for 2x2 plots
        self.graphics_layout = pg.GraphicsLayoutWidget()
        self.graphics_layout.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.graph_layout.addWidget(self.graphics_layout)

        # Create 2x2 grid of plots
        self.plots = [
            self.graphics_layout.addPlot(row=0, col=0),
            self.graphics_layout.addPlot(row=0, col=1),
            self.graphics_layout.addPlot(row=1, col=0),
            self.graphics_layout.addPlot(row=1, col=1)
        ]
        self.plots[0].setTitle("Voltage")
        self.plots[0].setLabels(left='Voltage (V)', bottom='Time (s)')
        self.plots[1].setTitle("Current")
        self.plots[1].setLabels(left='Current (A)', bottom='Time (s)')
        self.plots[2].setTitle("Pressure")
        self.plots[2].setLabels(left='Pressure (mBAR)', bottom='Time (s)')
        self.plots[3].setTitle("Resistance")
        self.plots[3].setLabels(left='Resistance (Ohm)', bottom='Time (s)')

        # Initialize plot data objects
        self.plot_voltage = self.plots[0].plot([], pen=pg.mkPen(color='b', width=2))
        self.plot_current = self.plots[1].plot([], pen=pg.mkPen(color='r', width=2))
        self.plot_pressure = self.plots[2].plot([], pen=pg.mkPen(color='g', width=2))
        self.plot_resistance = self.plots[3].plot([], pen=pg.mkPen(color='m', width=2))

        self.graph_stack.addWidget(self.graph_frame)  # Add 2D graphs to stack

        # 3D Plot Widget
        self.three_d_view = GLViewWidgetWithGrid()
        self.graph_stack.addWidget(self.three_d_view)  # Add 3D view to stack

        # Aux Camera Widget (populated when data is available)
        self.aux_frame = QFrame()
        self.aux_frame.setFrameShape(QFrame.StyledPanel)
        self.aux_frame.setFrameShadow(QFrame.Sunken)
        self.aux_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        aux_layout = QVBoxLayout(self.aux_frame)
        aux_layout.setContentsMargins(5, 5, 5, 5)
        aux_layout.setSpacing(5)
        self.aux_image_label = QLabel("Aux camera data not loaded.")
        self.aux_image_label.setAlignment(Qt.AlignCenter)
        self.aux_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        aux_layout.addWidget(self.aux_image_label)
        self.graph_stack.addWidget(self.aux_frame)  # Aux camera view

        self.view_widgets = {
            'graphs': self.graph_frame,
            'three_d': self.three_d_view,
            'aux': self.aux_frame
        }
        self.view_labels = {
            'graphs': "2D Graphs",
            'three_d': "3D View",
            'aux': "Aux Camera"
        }
        self.view_order = ['graphs', 'three_d']
        self.current_view_key = 'graphs'
        self.graph_stack.setCurrentWidget(self.graph_frame)

        self.top_layout.addWidget(self.graph_stack, stretch=1)  # Graphs take 1 part

        # Slider Frame
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMaximumHeight(30)
        self.slider.setTickPosition(QSlider.NoTicks)
        self.slider.valueChanged.connect(self.slider_changed)
        self.slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.slider_frame = QFrame()
        slider_layout = QHBoxLayout(self.slider_frame)
        slider_layout.setContentsMargins(0, 0, 0, 0)
        slider_layout.addWidget(self.slider)
        self.slider_frame.setFrameShape(QFrame.StyledPanel)
        self.slider_frame.setFrameShadow(QFrame.Sunken)
        self.slider_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.main_layout.addWidget(self.slider_frame, stretch=1)  # Allocate less space

        # Middle Layout: Info and Buttons
        self.middle_layout = QVBoxLayout()
        self.middle_layout.setSpacing(5)

        # Info Frame
        self.info_frame = QFrame()
        info_frame_layout = QHBoxLayout(self.info_frame)
        info_frame_layout.setContentsMargins(5, 5, 5, 5)
        info_frame_layout.setSpacing(10)

        # Info Label
        self.info = QLabel("No info yet!")
        self.info.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.info.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.info.setStyleSheet("color: white;")  # Ensure text is visible on dark background

        info_frame_layout.addWidget(self.info, stretch=1)

        self.chapter_label = QLabel("Chapter:")
        info_frame_layout.addWidget(self.chapter_label)

        self.chapter_selector = QComboBox()
        self.chapter_selector.setMinimumWidth(240)
        self.chapter_selector.setMaximumWidth(360)
        self.chapter_selector.addItem("Entire Replay", (0, -1))
        self.chapter_selector.currentIndexChanged.connect(self._on_chapter_changed)
        info_frame_layout.addWidget(self.chapter_selector)

        self.state_indicator = QLineEdit("No state")
        self.state_indicator.setReadOnly(True)
        self.state_indicator.setFocusPolicy(Qt.NoFocus)
        self.state_indicator.setAlignment(Qt.AlignCenter)
        self.state_indicator.setMinimumWidth(180)
        self.state_indicator.setMaximumWidth(260)
        self.state_glow_effect = None
        self._last_state_indicator_key = ("__unset__", "__unset__")
        self._set_state_indicator(None)
        info_frame_layout.addWidget(self.state_indicator)

        # Buttons Layout
        self.buttons_layout = QHBoxLayout()
        self.buttons_layout.setSpacing(10)

        # Previous Timepoint Button
        self.prev_image_button = QPushButton("Previous Timepoint")
        self.prev_image_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.prev_image_button.clicked.connect(self.show_previous_timepoint)
        self.buttons_layout.addWidget(self.prev_image_button)

        # Next Timepoint Button
        self.next_image_button = QPushButton("Next Timepoint")
        self.next_image_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.next_image_button.clicked.connect(self.show_next_timepoint)
        self.buttons_layout.addWidget(self.next_image_button)

        # Play/Pause Button
        self.toggle_video_button = QPushButton("Play")
        self.toggle_video_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.toggle_video_button.clicked.connect(self.toggle_video)
        self.buttons_layout.addWidget(self.toggle_video_button)

        # Select Data Directory Button
        self.select_image_dir_button = QPushButton("Select Data Directory")
        self.select_image_dir_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.select_image_dir_button.clicked.connect(self.open_directory)
        self.buttons_layout.addWidget(self.select_image_dir_button)

        # Toggle View Button
        self.toggle_view_button = QPushButton("")
        self.toggle_view_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.toggle_view_button.clicked.connect(self.toggle_view)
        self.buttons_layout.addWidget(self.toggle_view_button)
        self.update_toggle_button_text()

        # Toggle Inference Button
        self.toggle_inference_button = QPushButton("")
        self.toggle_inference_button.setCheckable(True)
        self.toggle_inference_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.toggle_inference_button.clicked.connect(self.toggle_replay_inference)
        self.buttons_layout.addWidget(self.toggle_inference_button)

        self.cell_detector_model_label = QLabel("Cell model:")
        self.cell_detector_model_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.buttons_layout.addWidget(self.cell_detector_model_label)

        self.cell_detector_model_selector = QComboBox()
        self.cell_detector_model_selector.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.cell_detector_model_selector.addItem("PIDNet", CELL_DETECTOR_PIDNET)
        self.cell_detector_model_selector.addItem("YOLO", CELL_DETECTOR_YOLO)
        self.cell_detector_model_selector.setToolTip(
            "Choose the cell detector used when replay inference is enabled."
        )
        self.cell_detector_model_selector.currentIndexChanged.connect(
            self._on_cell_detector_model_changed
        )
        self.buttons_layout.addWidget(self.cell_detector_model_selector)

        # Add a spacer to push buttons to the left
        self.buttons_layout.addStretch()

        # Add buttons layout to info_frame_layout
        info_frame_layout.addLayout(self.buttons_layout)

        # Add info_frame to middle_layout
        self.middle_layout.addWidget(self.info_frame)

        # Add middle_layout to main_layout
        self.main_layout.addLayout(self.middle_layout, stretch=1)  # Allocate less space

        # Initialize Data Manager
        self.data_manager = DataManager()

        # Inference/overlay settings (configure in-file as needed)
        self.enable_replay_inference = False
        self.enable_pipette_detector = True
        self.enable_cell_detector = True
        self.enable_pipette_focuser = True
        self.cell_detector_model = self.cell_detector_model_selector.currentData()
        self.overlay_point_radius = 6
        self.overlay_text_margin = 10

        # Lazy-loaded model holders and runtime state
        self._pipette_detector = None
        self._cell_detector = None
        self._pipette_focuser = None

        self._pipette_detector_init_failed = False
        self._cell_detector_init_failed = False
        self._pipette_focuser_init_failed = False
        self._frame_overlay_results = None
        self._cv2_unavailable_warned = False
        self._inference_disabled_notice_printed = False
        self._loading_dialog = None
        self._loading_elapsed_timer = None
        self._load_thread = None
        self._load_worker = None
        self._pending_directory = None
        self._inference_enable_thread = None
        self._inference_enable_worker = None

        self.toggle_inference_button.setChecked(self.enable_replay_inference)
        self.update_inference_toggle_button_text()

        # Initialize variables
        self.current_index = 0
        self._chapter_start_index = 0
        self._chapter_end_index = -1
        self.directory = None

        # Deques for real-time graphing
        self.pressure_deque = deque(maxlen=100)
        self.resistance_deque = deque(maxlen=100)

        # Timer for playback
        self.timer = QTimer()
        self.interval = 33  # 33ms for ~30 frames per second
        self.timer.setInterval(self.interval)
        self.timer.timeout.connect(self.advance_timepoint)

        # Keyboard Shortcuts
        self.shortcut_left = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.shortcut_left.activated.connect(self.show_previous_timepoint)
        self.shortcut_right = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.shortcut_right.activated.connect(self.show_next_timepoint)

        # Styling
        self.apply_styles()

    def apply_styles(self):
        """Apply dark theme styling to the application."""
        self.setStyleSheet("""
            QWidget {
                background-color: #2e2e2e;
                color: white;
            }
            QPushButton {
                background-color: #4a4a4a;
                color: white;
                border: 1px solid #5a5a5a;
                padding: 5px 10px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #4a4a4a;
                border: 1px solid #5a5a5a;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #ffffff;
                border: 1px solid #5a5a5a;
                width: 14px;
                margin: -3px 0;
                border-radius: 7px;
            }
        """)

    def _ensure_state_glow_effect(self):
        effect = self.state_indicator.graphicsEffect()
        if effect is None or not isinstance(effect, QGraphicsDropShadowEffect):
            effect = QGraphicsDropShadowEffect(self.state_indicator)
            effect.setOffset(0, 0)
            effect.setBlurRadius(18)
            self.state_indicator.setGraphicsEffect(effect)
        self.state_glow_effect = effect
        return effect

    def _set_state_indicator(self, state_name, outcome_code=None):
        glow_effect = self._ensure_state_glow_effect()
        if state_name:
            # Outcome code colors:
            # 0 -> green (success), 1 -> red (failure), 2 -> yellow (abort), 3+ -> orange
            if outcome_code == 0:
                bg_color = "#0f3a1d"
                text_color = "#9dffb6"
                border_color = "#2ee66f"
                glow_color = QColor(46, 230, 111)
            elif outcome_code == 1:
                bg_color = "#3b1414"
                text_color = "#ffb0b0"
                border_color = "#ff5c5c"
                glow_color = QColor(255, 92, 92)
            elif outcome_code == 2:
                bg_color = "#3b320f"
                text_color = "#ffe88f"
                border_color = "#ffd84d"
                glow_color = QColor(255, 216, 77)
            else:
                bg_color = "#3a240f"
                text_color = "#ffc991"
                border_color = "#ff9e42"
                glow_color = QColor(255, 158, 66)

            self.state_indicator.setText(state_name)
            self.state_indicator.setStyleSheet(
                "QLineEdit { "
                f"background-color: {bg_color}; color: {text_color}; "
                f"border: 2px solid {border_color}; border-radius: 4px; padding: 4px 8px; font-weight: 600; }}"
            )
            glow_effect.setColor(glow_color)
            glow_effect.setEnabled(True)
        else:
            self.state_indicator.setText("No state")
            self.state_indicator.setStyleSheet(
                "QLineEdit { background-color: #3a3a3a; color: #d8d8d8; "
                "border: 1px solid #5a5a5a; border-radius: 4px; padding: 4px 8px; }"
            )
            glow_effect.setEnabled(False)
        self.state_indicator.update()

    def toggle_view(self):
        """Cycle through available right-side views."""
        next_view = self._get_next_view_key()
        if not next_view:
            return

        self.current_view_key = next_view
        target_widget = self.view_widgets[next_view]
        self.graph_stack.setCurrentWidget(target_widget)

        if next_view == 'aux':
            self.update_aux_image()

        self.update_toggle_button_text()

    def _get_next_view_key(self):
        if not self.view_order:
            return None
        try:
            current_index = self.view_order.index(self.current_view_key)
        except ValueError:
            self.current_view_key = self.view_order[0]
            current_index = 0
        return self.view_order[(current_index + 1) % len(self.view_order)]

    def update_toggle_button_text(self):
        next_view = self._get_next_view_key()
        if not next_view:
            self.toggle_view_button.setText("Switch View")
            return
        self.toggle_view_button.setText(f"Switch to {self.view_labels[next_view]}")

    def update_inference_toggle_button_text(self):
        if self.enable_replay_inference:
            self.toggle_inference_button.setText("Disable Inference")
        else:
            self.toggle_inference_button.setText("Enable Inference")

    def _show_loading_popup(self, message):
        self._close_loading_popup()
        dialog = QDialog(self)
        dialog.setModal(True)
        dialog.setWindowTitle("Loading Replay")
        dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        dialog.setMinimumWidth(420)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        title_label = QLabel("Loading data into view...")
        title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        title_label.setStyleSheet("color: #7dff9b; font-weight: 700; font-size: 14px;")
        layout.addWidget(title_label)

        self._loading_message_label = QLabel(message)
        self._loading_message_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._loading_message_label.setStyleSheet("color: #d9ffe3;")
        layout.addWidget(self._loading_message_label)

        self._loading_elapsed_label = QLabel("Elapsed: 0.0 s")
        self._loading_elapsed_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._loading_elapsed_label.setStyleSheet("color: #9dffb6;")
        layout.addWidget(self._loading_elapsed_label)

        bar = QProgressBar(dialog)
        bar.setRange(0, 0)
        bar.setTextVisible(False)
        bar.setFixedHeight(14)
        bar.setStyleSheet(
            "QProgressBar {"
            "border: 1px solid #2e7d44; border-radius: 6px; background-color: #132318;"
            "}"
            "QProgressBar::chunk {"
            "background-color: #2ee66f;"
            "}"
        )
        layout.addWidget(bar)

        self._loading_dialog = dialog
        self._loading_elapsed_timer = QtCore.QElapsedTimer()
        self._loading_elapsed_timer.start()

        self._loading_tick_timer = QTimer(self)
        self._loading_tick_timer.setInterval(100)
        self._loading_tick_timer.timeout.connect(self._update_loading_popup_elapsed)
        self._loading_tick_timer.start()

        dialog.show()
        QApplication.processEvents()

    def _update_loading_popup_elapsed(self):
        if self._loading_dialog is None or self._loading_elapsed_timer is None:
            return
        try:
            elapsed_s = self._loading_elapsed_timer.elapsed() / 1000.0
        except Exception:
            elapsed_s = 0.0
        if hasattr(self, "_loading_elapsed_label") and self._loading_elapsed_label is not None:
            self._loading_elapsed_label.setText(f"Elapsed: {elapsed_s:.1f} s")
        QApplication.processEvents()

    def _set_loading_popup_message(self, message):
        if self._loading_dialog is None:
            return
        if hasattr(self, "_loading_message_label") and self._loading_message_label is not None:
            self._loading_message_label.setText(message)
        QApplication.processEvents()

    def _close_loading_popup(self):
        tick_timer = getattr(self, "_loading_tick_timer", None)
        if tick_timer is not None:
            try:
                tick_timer.stop()
            except Exception:
                pass
            self._loading_tick_timer = None
        dialog = self._loading_dialog
        self._loading_dialog = None
        self._loading_elapsed_timer = None
        self._loading_message_label = None
        self._loading_elapsed_label = None
        if dialog is not None:
            try:
                dialog.close()
            except Exception:
                pass

    def _start_directory_load_worker(self, directory):
        if self._load_thread is not None and self._load_thread.isRunning():
            self._close_loading_popup()
            QMessageBox.warning(self, "Loading In Progress", "A directory load is already running.")
            return

        self._pending_directory = directory
        self._load_thread = QtCore.QThread(self)
        self._load_worker = DirectoryLoadWorker(
            directory=directory,
            scaling_factor=self.data_manager.scaling_factor,
        )
        self._load_worker.moveToThread(self._load_thread)

        self._load_thread.started.connect(self._load_worker.run)
        self._load_worker.progress.connect(self._set_loading_popup_message)
        self._load_worker.finished.connect(self._on_directory_load_finished)
        self._load_worker.error.connect(self._on_directory_load_error)

        self._load_worker.finished.connect(self._load_thread.quit)
        self._load_worker.error.connect(self._load_thread.quit)
        self._load_worker.finished.connect(self._load_worker.deleteLater)
        self._load_worker.error.connect(self._load_worker.deleteLater)
        self._load_thread.finished.connect(self._load_thread.deleteLater)
        self._load_thread.finished.connect(self._on_directory_load_thread_finished)

        self._load_thread.start()

    def _on_directory_load_thread_finished(self):
        self._load_thread = None
        self._load_worker = None
        self._pending_directory = None

    def _on_directory_load_finished(self, loaded_manager):
        try:
            self.data_manager = loaded_manager
            self.directory = loaded_manager.directory or self._pending_directory
            self.info.setText(f"Loaded data from {self.directory}")
            self._last_state_indicator_key = ("__unset__", "__unset__")
            self.current_view_key = 'graphs'
            self.graph_stack.setCurrentWidget(self.view_widgets[self.current_view_key])
            self._reset_inference_state()
            self.refresh_view_order()

            self._set_loading_popup_message("Preparing timeline chapters...")
            self._populate_chapters()

            self._set_loading_popup_message("Rendering first frame...")
            self.update_view()
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"An unexpected error occurred: {exc}")
        finally:
            self._close_loading_popup()

    def _on_directory_load_error(self, title, message):
        self._close_loading_popup()
        QMessageBox.critical(self, title, message)

    def _start_inference_enable_worker(self):
        if self._inference_enable_thread is not None and self._inference_enable_thread.isRunning():
            return

        self._inference_enable_thread = QtCore.QThread(self)
        self._inference_enable_worker = InferenceEnableWorker(
            enable_pipette_detector=self.enable_pipette_detector,
            enable_cell_detector=self.enable_cell_detector,
            enable_pipette_focuser=self.enable_pipette_focuser,
            cell_detector_model=self.cell_detector_model,
        )
        self._inference_enable_worker.moveToThread(self._inference_enable_thread)

        self._inference_enable_thread.started.connect(self._inference_enable_worker.run)
        self._inference_enable_worker.progress.connect(self._set_loading_popup_message)
        self._inference_enable_worker.finished.connect(self._on_inference_enable_worker_finished)
        self._inference_enable_worker.error.connect(self._on_inference_enable_worker_error)

        self._inference_enable_worker.finished.connect(self._inference_enable_thread.quit)
        self._inference_enable_worker.error.connect(self._inference_enable_thread.quit)
        self._inference_enable_worker.finished.connect(self._inference_enable_worker.deleteLater)
        self._inference_enable_worker.error.connect(self._inference_enable_worker.deleteLater)
        self._inference_enable_thread.finished.connect(self._inference_enable_thread.deleteLater)
        self._inference_enable_thread.finished.connect(self._on_inference_enable_thread_finished)

        self._inference_enable_thread.start()

    def _on_inference_enable_thread_finished(self):
        self._inference_enable_thread = None
        self._inference_enable_worker = None

    def _on_inference_enable_worker_finished(self, result):
        try:
            self._pipette_detector = result.get("pipette_detector")
            self._cell_detector = result.get("cell_detector")
            self._pipette_focuser = result.get("pipette_focuser")
            self._pipette_detector_init_failed = bool(result.get("pipette_detector_init_failed", False))
            self._cell_detector_init_failed = bool(result.get("cell_detector_init_failed", False))
            self._pipette_focuser_init_failed = bool(result.get("pipette_focuser_init_failed", False))
            for warning_text in result.get("warnings", []):
                logger.warning(warning_text)
        except Exception as exc:
            logger.error("Failed to apply inference initialization results: %s", exc)
            self._on_inference_enable_worker_error(f"Inference initialization failed: {exc}")
            return

        self._complete_enable_inference()

    def _on_inference_enable_worker_error(self, message):
        logger.error("%s", message)
        self.enable_replay_inference = False
        self.toggle_inference_button.setChecked(False)
        self.update_inference_toggle_button_text()
        self.toggle_inference_button.setEnabled(True)
        self.cell_detector_model_selector.setEnabled(True)
        self._close_loading_popup()
        QMessageBox.critical(self, "Inference Error", message)

    def _complete_enable_inference(self):
        try:
            logger.info("Replay deep-learning inference enabled.")
            if self.data_manager.image_paths and 0 <= self.current_index < len(self.data_manager.image_paths):
                self._set_loading_popup_message("Rendering first inferred frame...")
                self.update_view()
        except Exception as exc:
            logger.error("Failed while enabling replay inference: %s", exc)
            self.enable_replay_inference = False
            self.toggle_inference_button.setChecked(False)
            self.update_inference_toggle_button_text()
            self.cell_detector_model_selector.setEnabled(True)
            QMessageBox.critical(self, "Inference Error", f"Failed to enable inference: {exc}")
        finally:
            self.toggle_inference_button.setEnabled(True)
            self._close_loading_popup()

    def toggle_replay_inference(self, checked):
        self.enable_replay_inference = bool(checked)
        self.update_inference_toggle_button_text()
        self.cell_detector_model_selector.setEnabled(not self.enable_replay_inference)

        if self.enable_replay_inference:
            self._inference_disabled_notice_printed = False
            self._show_loading_popup("Enabling deep-learning inference...")
            self._set_loading_popup_message("Initializing inference and loading models...")
            self.toggle_inference_button.setEnabled(False)
            self._start_inference_enable_worker()
            return

        logger.error("Replay deep-learning inference disabled by toggle; continuing without inference overlays.")
        self._frame_overlay_results = None
        if self.data_manager.image_paths and 0 <= self.current_index < len(self.data_manager.image_paths):
            self.display_image(self.data_manager.image_paths[self.current_index])

    def _on_cell_detector_model_changed(self):
        selected_model = self.cell_detector_model_selector.currentData()
        if selected_model is None:
            return
        self.cell_detector_model = str(selected_model).strip().lower()
        self._cell_detector = None
        self._cell_detector_init_failed = False
        self._reset_inference_state()

    def refresh_view_order(self):
        self.view_order = ['graphs', 'three_d']
        if self.data_manager.has_aux_images():
            self.view_order.append('aux')
        else:
            self.aux_image_label.setText("Aux camera data not loaded.")
            self.aux_image_label.setPixmap(QPixmap())

        if self.current_view_key not in self.view_order:
            self.current_view_key = 'graphs'
            self.graph_stack.setCurrentWidget(self.view_widgets[self.current_view_key])

        if self.current_view_key == 'aux':
            self.update_aux_image()

        self.update_toggle_button_text()

    def _reset_inference_state(self):
        self._frame_overlay_results = None

    def _ensure_pipette_detector(self):
        if self._pipette_detector is not None:
            return self._pipette_detector
        if self._pipette_detector_init_failed:
            return None
        try:
            from patcherbot.deepLearning.pipetteDetector import PipetteDetectorYOLO1
            self._pipette_detector = PipetteDetectorYOLO1()
        except Exception as exc:
            self._pipette_detector_init_failed = True
            logger.warning("PipetteDetectorYOLO1 init failed; disabling pipette overlay: %s", exc)
            self._pipette_detector = None
        return self._pipette_detector

    def _ensure_cell_detector(self):
        if self._cell_detector is not None:
            return self._cell_detector
        if self._cell_detector_init_failed:
            return None
        detector_name = _cell_detector_display_name(self.cell_detector_model)
        try:
            self._cell_detector = _create_cell_detector(self.cell_detector_model)
        except Exception as exc:
            self._cell_detector_init_failed = True
            logger.warning("%s init failed; disabling cell overlay: %s", detector_name, exc)
            self._cell_detector = None
        return self._cell_detector

    def _ensure_pipette_focuser(self):
        if self._pipette_focuser is not None:
            return self._pipette_focuser
        if self._pipette_focuser_init_failed:
            return None
        try:
            from patcherbot.deepLearning.pipetteFocuser import PipetteFocuser
            self._pipette_focuser = PipetteFocuser()
        except Exception as exc:
            self._pipette_focuser_init_failed = True
            logger.warning("PipetteFocuser init failed; disabling focus readout: %s", exc)
            self._pipette_focuser = None
        return self._pipette_focuser

    @staticmethod
    def _normalize_point(point, width, height):
        if point is None:
            return None
        try:
            x = float(point[0])
            y = float(point[1])
        except Exception:
            return None
        if not np.isfinite([x, y]).all():
            return None
        x_i = int(round(x))
        y_i = int(round(y))
        if x_i < 0 or y_i < 0 or x_i >= int(width) or y_i >= int(height):
            return None
        return x_i, y_i

    def _run_frame_inference(self, image_path, active_state):
        self._frame_overlay_results = {
            "image_path": image_path,
            "pipette_point": None,
            "cell_points": [],
            "focus_value": None,
        }

        if not self.enable_replay_inference:
            if not self._inference_disabled_notice_printed:
                logger.error("Replay deep-learning inference is disabled; continuing without inference overlays.")
                self._inference_disabled_notice_printed = True
            return
        self._inference_disabled_notice_printed = False
        if cv2 is None:
            if not self._cv2_unavailable_warned:
                logger.warning("OpenCV (cv2) is unavailable; replay inference overlays are disabled.")
                self._cv2_unavailable_warned = True
            return

        try:
            frame_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        except Exception as exc:
            logger.warning("Failed to read replay frame for inference %s: %s", image_path, exc)
            return

        if frame_bgr is None:
            logger.warning("Failed to read replay frame for inference: %s", image_path)
            return

        h, w = frame_bgr.shape[:2]

        if self.enable_pipette_detector:
            detector = self._ensure_pipette_detector()
            if detector is not None:
                try:
                    point = detector.detect_pipette(frame_bgr)
                    self._frame_overlay_results["pipette_point"] = self._normalize_point(point, w, h)
                except Exception as exc:
                    logger.warning("Pipette detector inference failed on replay frame: %s", exc)

        if self.enable_cell_detector:
            detector = self._ensure_cell_detector()
            if detector is not None:
                try:
                    detections = detector.detect_cells(frame_bgr, max_outputs=20)
                    self._frame_overlay_results["cell_points"] = [
                        point
                        for detection in detections
                        if (point := self._normalize_point(detection, w, h)) is not None
                    ]
                except Exception as exc:
                    logger.warning("Cell detector inference failed on replay frame: %s", exc)

        if self.enable_pipette_focuser:
            focuser = self._ensure_pipette_focuser()
            if focuser is not None:
                try:
                    focus_value = float(focuser.get_pipette_focus_value(frame_bgr))
                    if np.isfinite(focus_value):
                        self._frame_overlay_results["focus_value"] = focus_value
                except Exception as exc:
                    logger.warning("Pipette focus inference failed on replay frame: %s", exc)

    def _draw_overlay_point(self, painter, point, color):
        if point is None:
            return
        try:
            x, y = int(point[0]), int(point[1])
        except Exception:
            return
        painter.setPen(QtGui.QPen(color, 2))
        painter.setBrush(QtGui.QBrush(color))
        painter.drawEllipse(QtCore.QPointF(float(x), float(y)), float(self.overlay_point_radius), float(self.overlay_point_radius))

    def _apply_frame_overlays(self, pixmap, image_path):
        if pixmap is None or pixmap.isNull():
            return pixmap
        if not self._frame_overlay_results:
            return pixmap
        if self._frame_overlay_results.get("image_path") != image_path:
            return pixmap

        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        self._draw_overlay_point(
            painter,
            self._frame_overlay_results.get("pipette_point"),
            QtGui.QColor(255, 0, 0),
        )
        for cell_point in self._frame_overlay_results.get("cell_points", []):
            self._draw_overlay_point(painter, cell_point, QtGui.QColor(0, 255, 0))
        focus_value = self._frame_overlay_results.get("focus_value")
        if focus_value is not None:
            text_rect = pixmap.rect().adjusted(
                self.overlay_text_margin,
                self.overlay_text_margin,
                -self.overlay_text_margin,
                -self.overlay_text_margin,
            )
            font = painter.font()
            font.setPointSize(12)
            font.setBold(True)
            painter.setFont(font)
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0)))
            painter.drawText(
                text_rect,
                Qt.AlignTop | Qt.AlignRight,
                f"Focus: {focus_value:.2f}",
            )

        painter.end()
        return pixmap

    def _populate_chapters(self):
        """Build selectable replay ranges from recorded state windows."""
        frame_count = len(self.data_manager.timestamps)
        full_end = frame_count - 1
        self._chapter_start_index = 0
        self._chapter_end_index = full_end
        self.current_index = 0

        self.chapter_selector.blockSignals(True)
        self.slider.blockSignals(True)
        try:
            self.chapter_selector.clear()
            self.chapter_selector.addItem("Entire Replay", (0, full_end))

            occurrences = {}
            for started, finished, state_name, outcome_code in self.data_manager.state_windows:
                start_index = bisect.bisect_left(self.data_manager.timestamps, started)
                end_index = bisect.bisect_right(self.data_manager.timestamps, finished) - 1
                if start_index >= frame_count or end_index < start_index:
                    continue

                occurrences[state_name] = occurrences.get(state_name, 0) + 1
                display_name = state_name.replace("_", " ").title()
                label = f"{display_name} #{occurrences[state_name]}"
                if outcome_code is not None:
                    label += f" — outcome {outcome_code}"
                self.chapter_selector.addItem(label, (start_index, min(end_index, full_end)))

            self.chapter_selector.setCurrentIndex(0)
            self.slider.setMinimum(0)
            self.slider.setMaximum(max(0, full_end))
            self.slider.setValue(0)
        finally:
            self.slider.blockSignals(False)
            self.chapter_selector.blockSignals(False)

    def _on_chapter_changed(self, chapter_index):
        """Activate a chapter and constrain all replay navigation to its frames."""
        if not self.data_manager.image_paths:
            return
        chapter_range = self.chapter_selector.itemData(chapter_index)
        if not chapter_range or len(chapter_range) != 2:
            return

        start_index, end_index = (int(chapter_range[0]), int(chapter_range[1]))
        if end_index < start_index:
            return

        self.timer.stop()
        self.toggle_video_button.setText("Play")
        self._chapter_start_index = start_index
        self._chapter_end_index = end_index
        self.current_index = start_index

        self.slider.blockSignals(True)
        try:
            self.slider.setMinimum(start_index)
            self.slider.setMaximum(end_index)
            self.slider.setValue(start_index)
        finally:
            self.slider.blockSignals(False)
        self.update_view()

    def open_directory(self):
        """Open a directory dialog to select data directory."""
        directory = QFileDialog.getExistingDirectory(self, "Open Directory", "")
        if directory:
            self._show_loading_popup("Scanning files...")
            self._set_loading_popup_message("Loading camera, movement, and graph recordings...")
            self._start_directory_load_worker(directory)

    def check_data_loaded(self):
        """Check if data is loaded before allowing navigation."""
        if not self.data_manager.image_paths:
            QMessageBox.warning(self, "No Data Loaded", "Please load a directory with images first.")
            return False
        return True

    def show_previous_timepoint(self):
        """Show the previous timepoint."""
        if not self.check_data_loaded():
            return

        if self.current_index > self._chapter_start_index:
            self.current_index -= 1
            self.slider.setValue(self.current_index)

    def show_next_timepoint(self):
        """Show the next timepoint."""
        if not self.check_data_loaded():
            return

        if self.current_index < self._chapter_end_index:
            self.current_index += 1
            self.slider.setValue(self.current_index)

    def toggle_video(self):
        """Toggle play/pause functionality."""
        if not self.check_data_loaded():
            return

        if self.timer.isActive():
            self.toggle_video_button.setText("Play")
            self.timer.stop()
        else:
            self.toggle_video_button.setText("Pause")
            self.timer.start()

    def advance_timepoint(self):
        """Advance to the next timepoint during playback."""
        if not self.check_data_loaded():
            return

        if self.current_index < self._chapter_end_index:
            self.current_index += 1
            self.slider.setValue(self.current_index)
        else:
            self.toggle_video_button.setText("Play")
            self.timer.stop()

    def slider_changed(self, value):
        """Update view based on slider's position."""
        if not self.check_data_loaded():
            return

        self.current_index = value
        self.update_view()

    def update_view(self):
        """Update image, 2D graphs, and 3D plot based on current index."""
        image_path = self.data_manager.image_paths[self.current_index]
        current_timestamp = self.data_manager.timestamps[self.current_index]
        active_state, outcome_code = self.data_manager.get_active_state_for_timestamp(current_timestamp)
        self._run_frame_inference(image_path, active_state)
        self.display_image(image_path)
        indicator_key = (active_state, outcome_code)
        if indicator_key != self._last_state_indicator_key:
            self._set_state_indicator(active_state, outcome_code)
            self._last_state_indicator_key = indicator_key
        if self.data_manager.has_aux_images():
            self.update_aux_image()
        self.update_graphs()
        self.update_3d_view()

    def display_image(self, image_path):
        """Display the image on the QLabel."""
        if not os.path.exists(image_path):
            QMessageBox.warning(self, "Error", f"Image file {image_path} does not exist.")
            return

        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            QMessageBox.warning(self, "Error", "Unable to load image.")
        else:
            pixmap = self._apply_frame_overlays(pixmap, image_path)
            # Scale the pixmap to fit the label while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image.setPixmap(scaled_pixmap)
            self.image.show()

    def update_aux_image(self):
        """Display the aux camera frame that matches the current timestamp."""
        if not self.data_manager.has_aux_images():
            return
        if not self.data_manager.timestamps:
            return
        if not (0 <= self.current_index < len(self.data_manager.timestamps)):
            return

        aux_image_path = self.data_manager.get_aux_image_for_timestamp(
            self.data_manager.timestamps[self.current_index]
        )
        if not aux_image_path or not os.path.exists(aux_image_path):
            self.aux_image_label.clear()
            self.aux_image_label.setText("Aux frame missing for this timepoint.")
            return

        pixmap = QPixmap(aux_image_path)
        if pixmap.isNull():
            self.aux_image_label.clear()
            self.aux_image_label.setText("Unable to load aux camera image.")
            return

        target_size = self.aux_image_label.size()
        if target_size.width() > 0 and target_size.height() > 0:
            pixmap = pixmap.scaled(
                target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )

        self.aux_image_label.setText("")
        self.aux_image_label.setPixmap(pixmap)
        self.aux_image_label.show()

    def resizeEvent(self, event):
        """Handle window resize events to scale the image appropriately."""
        if self.data_manager.image_paths and 0 <= self.current_index < len(self.data_manager.image_paths):
            self.display_image(self.data_manager.image_paths[self.current_index])
        if self.data_manager.has_aux_images() and self.current_view_key == 'aux':
            self.update_aux_image()
        super().resizeEvent(event)  # Ensure the base class resizeEvent is also called

    def update_graphs(self):
        """Update the 2D graphs with the current timepoint data."""
        if not self.check_data_loaded():
            return

        # Find the closest graph_data entry for the current timestamp
        current_timestamp = self.data_manager.timestamps[self.current_index]
        graph_index = next((i for i, data in enumerate(self.data_manager.graph_data) if data['time'] >= current_timestamp), None)

        if graph_index is not None:
            graph_data = self.data_manager.graph_data[graph_index]

            # Update deques
            self.pressure_deque.clear()
            self.resistance_deque.clear()
            start_index = max(0, graph_index - 99)
            for i in range(start_index, graph_index + 1):
                data = self.data_manager.graph_data[i]
                self.pressure_deque.append(data['pressure'])
                self.resistance_deque.append(data['resistance'])

            # Update plot data
            self.plot_voltage.setData(graph_data['voltage'])
            self.plot_current.setData(graph_data['current'])
            self.plot_pressure.setData(list(self.pressure_deque))
            self.plot_resistance.setData(list(self.resistance_deque))

            # Retrieve the closest movement data entry based on current_timestamp
            movement_entry = self.data_manager.get_closest_movement_entry(current_timestamp)

            if movement_entry:
                stage_x, stage_y = movement_entry['stage']       # Correctly unpack stage
                microscope_z = movement_entry['microscope_z']     # Retrieve microscope_z separately
                pipette_x, pipette_y, pipette_z = movement_entry['pipette']
                time = current_timestamp - self.data_manager.init_image_time
                info_text = (
                    f"Time: {time:.2f} s, "
                    f"Stage: (X: {stage_x:.2f} mm, Y: {stage_y:.2f} mm, Z: {microscope_z:.2f} mm), "
                    f"Pipette: (X: {pipette_x:.2f} mm, Y: {pipette_y:.2f} mm, Z: {pipette_z:.2f} mm)<br>"
                    f"<span style='font-size: 24px; color: red;'>X "
                    f"<span style='font-size: 24px; color: green;'>Y "
                    f"<span style='font-size: 24px; color: blue;'>Z"
                )
            else:
                info_text = f"Time: {current_timestamp:.2f} s, No movement data available."

            self.info.setText(info_text)
        else:
            self.info.setText("No graph data available for this timestamp.")

    def update_3d_view(self):
        """Update the 3D visualization based on the current timepoint."""
        if not self.check_data_loaded():
            return

        if self.current_index >= len(self.data_manager.image_paths):
            return

        # Get the current image timestamp
        current_timestamp = self.data_manager.timestamps[self.current_index]

        # Retrieve the closest movement data entry based on current_timestamp
        movement_entry = self.data_manager.get_closest_movement_entry(current_timestamp)

        if movement_entry:
            time = movement_entry['time']
            stage_x, stage_y = movement_entry['stage']       # Only 2 elements
            microscope_z = movement_entry['microscope_z']     # Separate key
            pipette_x, pipette_y, pipette_z = movement_entry['pipette']

            # Update 3D objects
            self.three_d_view.update_objects(
                microscope_z=microscope_z,  # Corrected mapping
                stage_pos=(stage_x, stage_y),
                pipette_pos=(pipette_x, pipette_y, pipette_z)
            )
        else:
            # Optionally, handle cases where no movement data is available
            pass

# ------------------- Main Function -------------------

def main():
    app = QApplication(sys.argv)
    timeline = IntegratedTimeline()
    timeline.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
