"""Global System View tab; anchored snapshots use millimetres.

Supply set_anchor_state(dict) from the anchoring system, or load its JSON:
  scan_pixel_to_world_mm: 4x4 affine mapping native scan [x,y,0,1] to mm
  objective_xyz_mm: focus-plane centre [x,y,z], fov_size_mm: [width,height]
  pipette_xyz_mm: optional anchored tip [x,y,z]
  objective_transform / chamber_transform: optional CAD-local -> world matrices
The 40x rectangle is parallel to world XY. Snapshots never command hardware.
"""
import json
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtWidgets


class GlobalViewerTab(QtWidgets.QWidget):
    def __init__(self, scan_viewer, parent=None):
        super().__init__(parent)
        self.scan_viewer = scan_viewer
        self._scan = self._anchor = self.scene = None
        self._layout = QtWidgets.QVBoxLayout(self)
        buttons = QtWidgets.QHBoxLayout()
        for label, action in (("Open scan", scan_viewer.open_image),
                              ("Load anchoring", self.load_anchoring),
                              ("Reset 3D", lambda: self._view("reset_view")),
                              ("Top", lambda: self._view("top_view")),
                              ("Focus 40x", lambda: self._view("focus_view"))):
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(action)
            buttons.addWidget(button)
        self._layout.addLayout(buttons)
        self.status = QtWidgets.QLabel("Load a scan and its anchoring snapshot. Coordinates are in mm.")
        self.status.setWordWrap(True)
        self._layout.addWidget(self.status)
        scan_viewer.scanLoaded.connect(self._scan_loaded)

    @staticmethod
    def _matrix(value, name):
        matrix = np.asarray(value, dtype=float)
        if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                or not np.allclose(matrix[3], [0, 0, 0, 1])
                or np.linalg.matrix_rank(matrix[:3, :3]) != 3):
            raise ValueError(name + " must be a finite, invertible 4x4 affine matrix")
        return matrix

    @QtCore.pyqtSlot(object)
    def set_anchor_state(self, state):
        """Accept a complete snapshot; callers may connect their queued Qt signal."""
        state = dict(state)
        state["scan_pixel_to_world_mm"] = self._matrix(
            state["scan_pixel_to_world_mm"], "scan_pixel_to_world_mm")
        for name, size in (("objective_xyz_mm", 3), ("fov_size_mm", 2),
                           ("pipette_xyz_mm", 3)):
            if name == "pipette_xyz_mm" and state.get(name) is None:
                continue
            value = np.asarray(state[name], dtype=float)
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError(name + " has invalid coordinates")
            if name == "fov_size_mm" and np.any(value <= 0):
                raise ValueError("Field-of-view dimensions must be positive")
            state[name] = value
        for name in ("objective_transform", "chamber_transform"):
            if state.get(name) is not None:
                state[name] = self._matrix(state[name], name)
                rotation = state[name][:3, :3]
                if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
                    raise ValueError(name + " must preserve CAD dimensions")
        self._anchor = state
        self._update_overlay()
        self._render()

    def load_anchoring(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load anchoring snapshot (mm)", "", "Anchoring (*.json)")
        if path:
            try:
                self.set_anchor_state(json.loads(Path(path).read_text(encoding="utf-8-sig")))
            except (OSError, ValueError, KeyError, TypeError) as error:
                self.status.setText("Anchoring could not be loaded: " + str(error))

    def _scan_loaded(self, result):
        changed = self._scan is not None and self._scan["path"] != result["path"]
        self._scan = dict(result)
        pixels = result["pixels"]
        step = max(1, int(np.ceil(max(pixels.shape[:2]) / 256)))
        self._scan["pixels"] = pixels[::step, ::step].copy()
        if changed:
            self._anchor = None
        supplied = result.get("metadata", {}).get("global_system_anchor")
        if supplied is not None:
            try:
                self.set_anchor_state(supplied)
                return
            except (ValueError, KeyError, TypeError) as error:
                self._anchor = None
                self.status.setText("Invalid scan anchoring: " + str(error))
        self._update_overlay()
        self._render()

    def _update_overlay(self):
        if self._anchor is None or self._scan is None:
            self.scan_viewer.set_global_overlay(None)
            return
        state = self._anchor
        centre = state["objective_xyz_mm"]
        half = state["fov_size_mm"] / 2
        corners = centre + np.array([[-half[0], -half[1], 0],
                                      [half[0], -half[1], 0],
                                      [half[0], half[1], 0],
                                      [-half[0], half[1], 0]])
        transform = state["scan_pixel_to_world_mm"]
        def project(points):
            # Orthogonal projection onto the supplied scan plane, in native pixels.
            return np.linalg.lstsq(transform[:3, :2],
                                   (np.asarray(points) - transform[:3, 3]).T,
                                   rcond=None)[0].T
        tip = state.get("pipette_xyz_mm")
        direction = state.get("pipette_marker_direction_xyz")
        direction = None if direction is None else np.linalg.lstsq(
            transform[:3, :2], direction, rcond=None)[0]
        self.scan_viewer.set_global_overlay(project(corners),
                                           None if tip is None else project([tip])[0], direction)

    def showEvent(self, event):
        super().showEvent(event)
        if self.scene is None:
            try:
                from matplotlib.figure import Figure
                from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
                from patcherbot.deepLearning.globalModel.GlobalSystemViewer import GlobalSystemViewer
                self.figure = Figure(figsize=(7, 6))
                self.canvas = FigureCanvasQTAgg(self.figure)
                self._layout.addWidget(NavigationToolbar2QT(self.canvas, self))
                self._layout.addWidget(self.canvas, 1)
                self.scene = GlobalSystemViewer(self.figure.add_subplot(111, projection="3d"))
            except (ImportError, OSError, ValueError) as error:
                self.status.setText("Global viewer could not start: " + str(error))
                return
        self._render()

    def _render(self):
        if self._anchor is None:
            self.status.setText("Awaiting anchoring snapshot; CAD dimensions are nominal estimates.")
            if self.scene is not None:
                self.scene.ax.clear()
                self.canvas.draw_idle()
            return
        self.status.setText("Anchored snapshot in mm | cyan: 40x field | magenta: pipette tip. "
                            "CAD mounting pose requires supplied transforms; chamber dimensions are estimated.")
        if self.scene is None or not self.isVisible():
            return
        if self._scan is not None:
            self.scene.set_scan(self._scan["pixels"], self._scan["width"],
                                self._scan["height"], self._anchor["scan_pixel_to_world_mm"])
        self.scene.set_state(**{key: self._anchor[key] for key in (
            "objective_xyz_mm", "fov_size_mm", "pipette_xyz_mm", "objective_transform",
            "chamber_transform") if key in self._anchor})
        self.canvas.draw_idle()

    def _view(self, method):
        if self.scene is not None and self._anchor is not None:
            getattr(self.scene, method)()
            self.canvas.draw_idle()


class RegisteredGlobalViewerTab(GlobalViewerTab):
    """Viewer connection to the shared registrar and read-only unit helper."""

    snapshot_ready = QtCore.pyqtSignal(int, object, object)
    snapshot_failed = QtCore.pyqtSignal(int, str)
    atlas_position_ready = QtCore.pyqtSignal(object, object)

    def __init__(self, scan_viewer, parent=None, position_helper=None):
        super().__init__(scan_viewer, parent)
        legacy_toolbar = self._layout.takeAt(0).layout()
        while legacy_toolbar.count():
            widget = legacy_toolbar.takeAt(0).widget()
            if widget is not None:
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()
        legacy_toolbar.deleteLater()
        self.position_helper = position_helper
        if self.position_helper is None and parent is not None:
            self.position_helper = parent.pipette_interface.calibrated_unit.global_position_helper
        self._scan_transform = None
        self._scan_registration_valid = True
        self._snapshot_busy = False
        self._snapshot_generation = 0
        self._scan_render_key = None
        self._layers = {}
        self._layer_errors = {}
        self._initial_system_fit = False
        self._last_position_key = None
        self._atlas_states = {}
        self._atlas_scan_info = {}
        self._atlas_query_busy = False
        self._atlas_query_key = self._current_atlas_key = None
        self._atlas_position_text = self._position_text = ""
        from collections import deque
        import threading
        self._producer_stop = threading.Event()
        self._producer_active = threading.Event()
        self._producer_input = None
        self._snapshot_mailbox = deque(maxlen=1)
        self._atlas_jobs = deque(maxlen=1)
        self._atlas_mailbox = deque(maxlen=1)
        self._objective = 40
        self._view_settings = QtCore.QSettings("PatcherBot", "RegisteredGlobalViewer")
        self._cad_angles = {"objective": (0., 0.), "chamber": (0., 0.)}
        self._view_name = self._view_settings.value("view", "isometric")
        if self._view_name not in ("top", "side", "isometric"):
            self._view_name = "isometric"
        self._closed = False
        self.position_box = QtWidgets.QLabel("Global XY: awaiting position | reference-plane Z: 0 µm")
        self.position_box.setObjectName("globalPositionBox")
        self.position_box.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self._layout.insertWidget(0, self.position_box)
        controls = QtWidgets.QHBoxLayout()
        for label, method in (("Top", "top_view"), ("Side", "side_view"), ("Isometric", "isometric_view")):
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(lambda checked=False, name=method: self._view(name))
            controls.addWidget(button)
        self.lock_view = QtWidgets.QCheckBox("Lock view")
        self.lock_view.setChecked(self._view_settings.value("view_locked", True, type=bool))
        self.lock_view.toggled.connect(self._lock_changed)
        controls.addWidget(self.lock_view)
        for label, factor in (("Zoom +", .8), ("Zoom −", 1.25)):
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(lambda checked=False, value=factor: self.scene.zoom(value) if self.scene else None)
            controls.addWidget(button)
        self.fit_system_button = QtWidgets.QPushButton("Fit system")
        self.fit_system_button.clicked.connect(lambda: self.scene.reset_view() if self.scene else None)
        controls.addWidget(self.fit_system_button)
        self._layout.insertLayout(0, controls)
        self.snapshot_ready.connect(self._receive_snapshot)
        self.snapshot_failed.connect(self._receive_failure)
        self.atlas_position_ready.connect(self._receive_atlas_position)
        scan_viewer.scanLoaded.connect(self._registered_scan_loaded)
        if hasattr(scan_viewer, "layerLoaded"):
            scan_viewer.layerLoaded.connect(self._layer_loaded)
        if hasattr(scan_viewer, "atlasOverlayChanged"):
            scan_viewer.atlasOverlayChanged.connect(self._atlas_changed)
        self._refresh_timer = QtCore.QTimer(self)
        self._refresh_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self._refresh_timer.setInterval(24)
        self._refresh_timer.timeout.connect(self.refresh_position)
        self._refresh_timer.start()
        self.destroyed.connect(self._stop_refresh)
        self._position_producer = threading.Thread(target=self._produce_positions,
                                                   name="GlobalPositionProducer", daemon=True)
        self._atlas_producer = threading.Thread(target=self._produce_atlas,
                                                name="AtlasPositionProducer", daemon=True)
        self._position_producer.start()
        self._atlas_producer.start()

    def _produce_positions(self):
        """Independent cache consumer; publication replaces, never queues, old state."""
        while not self._producer_stop.is_set():
            if self._producer_active.is_set() and self._producer_input is not None:
                generation, helper, transform, objective = self._producer_input
                try:
                    state, stage_xy = helper.cached_snapshot(transform, objective=objective)
                    scope_um = state.get("microscope_z_um")
                    if scope_um is not None:
                        if not hasattr(self, "_scope_reference_um"):
                            self._scope_reference_um = float(scope_um)
                        state["rig_focus_z_mm"] = -(float(scope_um) - self._scope_reference_um) / 1000.
                    self._snapshot_mailbox.append((generation, state, stage_xy, None))
                except Exception as error:
                    self._snapshot_mailbox.append((generation, None, None, str(error)))
            self._producer_stop.wait(.024)

    def _produce_atlas(self):
        """Atlas labels run separately and never delay the geometry producer."""
        while not self._producer_stop.is_set():
            if self._producer_active.is_set():
                try:
                    key, helper, role, points = self._atlas_jobs.pop()
                except IndexError:
                    pass
                else:
                    values = {}
                    for name, xy in points.items():
                        try:
                            values[name] = helper.global_to_atlas(xy, role=role)
                        except Exception as error:
                            values[name] = {"error": str(error)}
                    self._atlas_mailbox.append((key, values))
            self._producer_stop.wait(.2)

    def _registered_scan_loaded(self, result):
        self._snapshot_generation += 1
        self._scan_render_key = None
        if self.position_helper is None:
            return
        metadata = result.get("metadata", {})
        role = metadata.get("registration_role", "reference")
        try:
            self._scan_transform = self._scan_to_global(role, result)
            self._scan_registration_valid = True
            if role not in self._layers or not hasattr(self.scan_viewer, "layerLoaded"):
                self._layer_loaded(role, result)
            self.set_anchor_state(self.position_helper.snapshot(self._scan_transform, objective=self._objective))
        except (OSError, ValueError, KeyError, TypeError) as error:
            self._scan_transform = None
            self._scan_registration_valid = False
            self.scan_viewer.set_global_overlay(None)
            self.status.setText("Scan needs a compatible registration: " + str(error))

    def _scan_to_global(self, role, result):
        if role == "reference":
            return np.array(self.position_helper.registrar.reference_pixel_to_global_um)
        metadata = result.get("metadata", {})
        shape = metadata.get("canvas_shape")
        factor = 1. if shape is None else np.array([float(shape[1]) / result["width"],
                                                   float(shape[0]) / result["height"]])
        return self.position_helper.registered_session_mosaic(metadata, downsample_factor=factor)

    def _layer_loaded(self, role, result):
        if role not in ("reference", "session") or self.position_helper is None:
            return
        self._atlas_scan_info[role] = {key: result[key] for key in ("path", "width", "height", "metadata")
                                      if key in result}
        try:
            transform = self._scan_to_global(role, result)
            state = self.position_helper.snapshot(transform, objective=self._objective)
            self._layers[role] = dict(width=result["width"], height=result["height"],
                                     transform=self._matrix(state["scan_pixel_to_world_mm"], role))
            self._layer_errors.pop(role, None)
        except (OSError, ValueError, KeyError, TypeError, np.linalg.LinAlgError) as error:
            self._layers.pop(role, None)
            self._layer_errors[role] = str(error)
        self._atlas_changed(role, self._atlas_states.get(role))
        self._render()

    def _atlas_changed(self, role, data):
        """Bind the displayed atlas fit to this scan's existing global transform."""
        if role not in ("reference", "session") or self.position_helper is None:
            return
        self._atlas_states[role] = data
        self.position_helper.set_atlas_alignment(None, role=role)
        self._snapshot_generation += 1
        self._atlas_query_key = self._current_atlas_key = None
        self._atlas_position_text = ""
        self.position_box.setText(self._position_text)
        label = getattr(getattr(self.scan_viewer, role + "_view", None), "atlas_position_status", None)
        if label is not None:
            label.setText("Atlas position: awaiting a valid alignment and current position.")
        source = self._atlas_scan_info.get(role)
        if data is None or source is None or "result" not in data:
            return
        try:
            if Path(data["scan_path"]).resolve() != Path(source["path"]).resolve():
                return
            self.position_helper.set_atlas_alignment(data, self._scan_to_global(role, source), role=role)
        except (ValueError, KeyError, OSError, np.linalg.LinAlgError) as error:
            self.status.setText("Atlas coordinates unavailable: " + str(error))

    def _refresh_atlas_position(self, state):
        role = getattr(self.scan_viewer, "active_role", "reference")
        helper = self.position_helper
        if not helper.has_atlas_alignment(role):
            return
        points = {"Field": np.asarray(state["objective_xyz_mm"])[:2] * 1000}
        if state.get("pipette_xyz_mm") is not None:
            points["Pipette"] = np.asarray(state["pipette_xyz_mm"])[:2] * 1000
        key = (self._snapshot_generation, role, tuple((name, tuple(xy)) for name, xy in points.items()))
        self._current_atlas_key = key
        if key == self._atlas_query_key:
            return
        self._atlas_query_busy = True
        self._atlas_query_key = key
        self._atlas_jobs.append((key, helper, role, points))

    def _receive_atlas_position(self, key, values):
        self._atlas_query_busy = False
        if self._closed or key != self._current_atlas_key or key[0] != self._snapshot_generation:
            return
        labels = []
        for name, value in values.items():
            if "error" in value:
                labels.append(name + ": " + value["error"])
            else:
                ap, dv, ml = value["ccf_um_ap_dv_ml"]
                labels.append(f"{name}: Allen AP {ap:.0f}, DV {dv:.0f}, ML {ml:.0f} µm; region {value.get('region_id', 'unavailable')}")
        self._atlas_position_text = " | ".join(labels)
        self.position_box.setText(self._position_text + "\n" + self._atlas_position_text)
        label = getattr(getattr(self.scan_viewer, "active_view", None), "atlas_position_status", None)
        if label is not None:
            label.setText(self._atlas_position_text)

    def _update_overlay(self):
        if not self.scan_viewer.isVisible():
            return
        if not self._scan_registration_valid or self._anchor is None or self._scan is None:
            self.scan_viewer.set_global_overlay(None)
            return
        corners = self._anchor.get("fov_corners_mm")
        if corners is None:
            return super()._update_overlay()
        transform = self._anchor["scan_pixel_to_world_mm"]
        def project(points):
            return np.linalg.lstsq(transform[:3, :2],
                                   (np.asarray(points) - transform[:3, 3]).T, rcond=None)[0].T
        tip = self._anchor.get("pipette_xyz_mm")
        direction = self._anchor.get("pipette_marker_direction_xyz")
        direction = None if direction is None else np.linalg.lstsq(
            transform[:3, :2], direction, rcond=None)[0]
        self.scan_viewer.set_global_overlay(project(corners),
                                           None if tip is None else project([tip])[0], direction)

    def refresh_position(self):
        if (self._closed or not (self.isVisible() or self.scan_viewer.isVisible()) or self.position_helper is None):
            self._producer_active.clear()
            return
        transform = None if self._scan_transform is None else self._scan_transform.copy()
        self._producer_input = (self._snapshot_generation, self.position_helper, transform, self._objective)
        self._producer_active.set()
        try:
            generation, state, stage_xy, error = self._snapshot_mailbox.pop()
        except IndexError:
            pass
        else:
            if error is None:
                self._receive_snapshot(generation, state, stage_xy)
            else:
                self._receive_failure(generation, error)
        try:
            key, values = self._atlas_mailbox.pop()
        except IndexError:
            pass
        else:
            self._receive_atlas_position(key, values)

    def _receive_snapshot(self, generation, state, stage_xy):
        self._snapshot_busy = False
        if (self._closed or generation != self._snapshot_generation
                or not (self.isVisible() or self.scan_viewer.isVisible())):
            return
        self._refresh_atlas_position(state)
        fields = ("scan_pixel_to_world_mm", "objective_xyz_mm", "fov_size_mm", "fov_corners_mm",
                  "pipette_xyz_mm", "pipette_direction_xyz", "objective_transform", "chamber_transform", "stage_bottom_xyz_mm",
                  "objective_rotation_yz_degrees", "chamber_rotation_yz_degrees",
                  "rig_focus_z_mm", "pipette_rig_origin_mm", "pipette_rig_xyz_mm",
                  "pipette_marker_direction_xyz", "global_to_rig_rotation")
        key = (generation, self.isVisible(), self.scan_viewer.isVisible(),
               np.asarray(stage_xy, dtype=float).tobytes(),
               tuple(None if state.get(name) is None else np.asarray(state[name], dtype=float).tobytes()
                     for name in fields),
               tuple((name, np.asarray(xyz, dtype=float).tobytes())
                     for name, xyz in sorted(state.get("named_anchors_mm", {}).items())))
        if key == self._last_position_key:
            return
        self.set_anchor_state(state)
        self._last_position_key = key
        x, y, z = np.asarray(state["objective_xyz_mm"]) * 1000
        text = (f"Global X {x:.2f}, Y {y:.2f} µm | reference-plane Z {z:.2f} µm\n"
                f"Stage X {stage_xy[0]:.2f}, Y {stage_xy[1]:.2f} µm")
        if state.get("pipette_xyz_mm") is not None:
            px, py, pz = np.asarray(state["pipette_xyz_mm"]) * 1000
            text += f" | Pipette X {px:.2f}, Y {py:.2f}, Z {pz:.2f} µm"
        self._position_text = text
        self.position_box.setText(text + ("\n" + self._atlas_position_text if self._atlas_position_text else ""))

    def _receive_failure(self, generation, message):
        self._snapshot_busy = False
        self._last_position_key = None
        if generation == self._snapshot_generation and not self._closed:
            self.position_box.setText("Global position unavailable: " + message)

    def _stop_refresh(self, *_):
        self._closed = True
        self._refresh_timer.stop()
        self._producer_active.clear()
        self._producer_stop.set()

    def _lock_changed(self, locked):
        self._view_settings.setValue("view_locked", bool(locked))
        if self.scene is not None:
            self.scene.set_view_lock(locked)

    def _view(self, method):
        choices = {"top_view": "top", "side_view": "side", "isometric_view": "isometric"}
        if method in choices:
            self._view_name = choices[method]
            self._view_settings.setValue("view", self._view_name)
        if self.scene is not None and self._anchor is not None:
            getattr(self.scene, method)()
            self.canvas.update()

    def set_anchor_state(self, state):
        self._last_position_key = None
        corners = state.get("fov_corners_mm")
        if corners is not None and (np.asarray(corners).shape != (4, 3) or not np.isfinite(corners).all()):
            raise ValueError("Field-of-view corners must be four finite XYZ coordinates in mm")
        for asset in self._cad_angles:
            angles = state.get(asset + "_rotation_yz_degrees")
            if angles is not None:
                if np.asarray(angles).shape != (2,) or not np.isfinite(angles).all():
                    raise ValueError("CAD rotation must contain finite Y/Z degrees")
                self._cad_angles[asset] = tuple(map(float, angles))
        super().set_anchor_state(state)

    def showEvent(self, event):
        QtWidgets.QWidget.showEvent(self, event)
        if self.scene is None:
            try:
                from patcherbot.deepLearning.globalModel.GlobalSystemViewer import GpuGlobalSystemViewer
                from patcherbot.configs.CalibrationConfig import CalibrationConfig
                config = getattr(getattr(self.position_helper, "unit", None), "config", None)
                chamber_z = getattr(config, "chamber_z_rotation",
                                    CalibrationConfig.param["chamber_z_rotation"].default)
                self.scene = GpuGlobalSystemViewer(parent=self, chamber_z_rotation=chamber_z)
            except (ImportError, OSError, RuntimeError, ValueError) as error:
                self.status.setText("GPU 3D viewer could not start: " + str(error))
                return
            self.canvas = self.scene.canvas
            self._layout.addWidget(self.canvas, 1)
            self.scene.set_view(self._view_name)
            self.scene.set_view_lock(self.lock_view.isChecked())
            self.scene.set_layer_visible("cad", True)
            self.scene.set_layer_visible("chamber", True)
            self.scene.mark_origin()
        for role, result in getattr(self.scan_viewer, "_results", {}).items():
            if role not in self._layers:
                self._layer_loaded(role, result)
        self._render()
        self.refresh_position()

    def _render(self):
        if self.scene is None or not self.isVisible():
            return
        if self._anchor is None:
            return
        self.scene.cad_angles.update(self._cad_angles)
        basis = np.asarray(self._anchor.get("global_to_rig_rotation", np.eye(3)))
        # The scan uses sample coordinates; hardware is shown relative to the
        # fixed objective XY axis. Recorder stage motion is negative raw XY.
        offset = np.r_[np.asarray(self._anchor["objective_xyz_mm"])[:2], 0.]
        pipette_origin = np.asarray(self._anchor.get("pipette_rig_origin_mm")
                                    if self._anchor.get("pipette_rig_origin_mm") is not None else [0., 0., 0.])
        tip = self._anchor.get("pipette_xyz_mm")
        tip = None if tip is None else np.asarray(tip) - pipette_origin
        if self._anchor.get("pipette_rig_xyz_mm") is not None:
            tip = np.asarray(self._anchor["pipette_rig_xyz_mm"])
        tip = None if tip is None else basis @ tip
        if self._anchor.get("stage_bottom_xyz_mm") is not None:
            self.scene.set_stage_bottom(basis @ (np.asarray(self._anchor["stage_bottom_xyz_mm"]) - offset))
        anchors = self._anchor.get("named_anchors_mm", {})
        for name in set(getattr(self, "_displayed_anchor_names", ())) - set(anchors):
            self.scene.clear_named_anchor(name)
        for name, xyz in anchors.items():
            position = (tip if name == "pipette" and tip is not None else basis @ (np.asarray(xyz) -
                        (pipette_origin if name.startswith("pipette") or name == "cleaning_bath" else offset)))
            self.scene.set_named_anchor(name, position)
        self._displayed_anchor_names = set(anchors)
        corners = self._anchor.get("fov_corners_mm")
        self.scene.set_fov_corners(None if corners is None else (np.asarray(corners) - offset) @ basis.T)
        state = {key: self._anchor[key] for key in ("fov_size_mm", "objective_transform", "chamber_transform")
                 if key in self._anchor}
        state.update(objective_xyz_mm=[0., 0., self._anchor.get("rig_focus_z_mm", 0.)], pipette_xyz_mm=tip)
        self.scene.set_state(**state)
        self.scene.items["origin"].setData(pos=np.asarray([basis @ -offset]))
        direction = self._anchor.get("pipette_direction_xyz")
        self.scene.set_pipette_pose(tip if direction is not None else None,
                                    direction)
        if not self._initial_system_fit:
            self.scene.reset_view()
            self._initial_system_fit = True
        status = ("Objective fixed in XY; stage follows recorded movement | "
                  "Basin floor at slope bottom (provisional) | "
                  f"Objective working distance: {self.scene.OBJECTIVE_WORKING_DISTANCE_MM:.3f} mm")
        if "session" in self._layers:
            status += " | Session placement uses recorded stage mapping"
        if self._layer_errors:
            status += " | " + "; ".join(name + ": " + error for name, error in self._layer_errors.items())
        self.status.setText(status)
