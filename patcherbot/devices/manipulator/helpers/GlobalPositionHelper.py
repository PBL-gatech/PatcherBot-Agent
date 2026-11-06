"""Read-only bridge between a calibrated unit and the image-defined global origin.

All public conversion inputs/outputs are native pixels or microns. Only viewer
snapshots convert to millimetres. No method calibrates, records, or moves hardware.
"""
import json
from pathlib import Path

import numpy as np


class GlobalPositionHelper:
    def __init__(self, calibrated_unit, registrar=None):
        self.unit = calibrated_unit
        self._registrar = registrar
        self._named_anchors_um = {}
        self._pipette_home_anchor = None
        self._pending_pipette_positions = {}

    @property
    def registrar(self):
        if self._registrar is None:
            from patcherbot.deepLearning.ObjectiveRegistrar import ObjectiveRegistrar
            self._registrar = ObjectiveRegistrar.shared(config=getattr(self.unit, "config", None))
        return self._registrar

    @property
    def stage(self):
        if len(self.unit.axes) == 2:
            return self.unit
        return self.unit.stage

    @staticmethod
    def _xy(value):
        value = np.asarray(value, dtype=float)
        if value.shape[-1:] != (2,) or not np.isfinite(value).all():
            raise ValueError("Expected finite XY coordinates")
        return value

    def _reference(self, objective):
        if objective not in (4, 40):
            raise ValueError("objective must be 4 or 40")
        reference = self.registrar.reference_frame(objective)
        size = np.asarray(reference["image_size"], dtype=float)
        center = (size - 1) / 2
        global_center = self.registrar.pixel_to_global(center, objective)
        return reference, np.asarray(global_center, dtype=float)

    def _camera_size(self, objective):
        camera = getattr(self.unit, "camera", None)
        if camera is not None:
            size = np.array([camera.width, camera.height], dtype=float)
        else:
            size = np.asarray(self._reference(objective)[0]["image_size"], dtype=float)
        if size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0):
            raise ValueError("Camera dimensions must be positive native pixels")
        return size

    def stage_to_global(self, stage_xy, objective=40):
        reference, center = self._reference(objective)
        return self._xy(stage_xy) - self._xy(reference["stage_xy_um"]) + center

    def global_to_stage(self, global_xy, objective=40):
        reference, center = self._reference(objective)
        return self._xy(global_xy) - center + self._xy(reference["stage_xy_um"])

    def current_global_position(self, objective=40):
        """Explicit hardware read through the existing stage.position() API."""
        return self.stage_to_global(np.asarray(self.stage.position(), dtype=float)[:2], objective)

    def pixel_to_global(self, pixels, objective=40, *, stage_xy=None, frame_index=None):
        points = self._xy(pixels)
        if frame_index is not None and stage_xy is not None:
            raise ValueError("Choose a saved frame or a stage position, not both")
        if stage_xy is None:
            return self.registrar.pixel_to_global(points, objective, frame_index=frame_index)
        center = (self._camera_size(objective) - 1) / 2
        matrix = np.asarray(self.registrar.pixel_to_micron_matrix(objective), dtype=float)
        return (points - center) @ matrix.T + self.stage_to_global(stage_xy, objective)

    def global_to_pixel(self, global_xy, objective=40, *, stage_xy=None, frame_index=None):
        points = self._xy(global_xy)
        if frame_index is not None and stage_xy is not None:
            raise ValueError("Choose a saved frame or a stage position, not both")
        if stage_xy is None:
            return self.registrar.global_to_pixel(points, objective, frame_index=frame_index)
        matrix = np.asarray(self.registrar.pixel_to_micron_matrix(objective), dtype=float)
        return ((points - self.stage_to_global(stage_xy, objective)) @ np.linalg.inv(matrix).T
                + (self._camera_size(objective) - 1) / 2)

    def register_pair(self, image4, image40):
        return self.registrar.register_pair(image4, image40)

    @staticmethod
    def _xyz(value):
        point = np.asarray(value, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("Expected three finite XYZ coordinates in microns")
        return point

    @staticmethod
    def pipette_shaft_direction(elevation_degrees, azimuth_degrees, zero_azimuth_degrees):
        """Tip-to-holder direction from the configured azimuth reference.

        Zero azimuth is measured from world +X (-90 degrees means -Y).
        Positive Z rotation is counterclockwise; motor signs do not apply.
        """
        angles = np.asarray([elevation_degrees, azimuth_degrees, zero_azimuth_degrees], dtype=float)
        if not np.isfinite(angles).all():
            raise ValueError("Finite shaft rotations are required")
        elevation, azimuth, zero = np.deg2rad(angles)
        azimuth += zero
        pitch = np.array([[np.cos(elevation), 0., -np.sin(elevation)], [0., 1., 0.],
                          [np.sin(elevation), 0., np.cos(elevation)]])
        yaw = np.array([[np.cos(azimuth), -np.sin(azimuth), 0.],
                        [np.sin(azimuth), np.cos(azimuth), 0.], [0., 0., 1.]])
        return yaw @ pitch @ np.array([1., 0., 0.])

    def set_pipette_home_anchor(self, home_pipette_xyz, home_stage_xyz, *, home_z_um,
                                tip_xy=None, safe_pipette_xyz=None, cleaning_bath_xyz=None,
                                objective=40):
        """Bind newly recorded home/safe positions to the image-defined XY map.

        home_z_um explicitly sets home height in the global reference; zero
        declares a home reference plane and does not measure the basin floor.
        Existing pipette M supplies image XY; its identity Z row is raw motor
        microns. up_direction converts those microns to positive-up world Z.
        Rig rotation is already measured in M, so it is never applied twice.
        """
        if len(self.unit.axes) != 3 or not getattr(self.unit, "calibrated", False):
            raise ValueError("A calibrated three-axis pipette is required")
        if objective != 40:
            raise ValueError("The existing pipette calibration uses the 40x image frame")
        home, stage = self._xyz(home_pipette_xyz), self._xyz(home_stage_xyz)
        matrix = np.asarray(self.unit.M, dtype=float)
        if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
                or not np.allclose(matrix[2], [0, 0, 1]) or not np.isfinite(home_z_um)):
            raise ValueError("Pipette anchoring needs finite planar M, raw-micron Z, and explicit home Z")
        up = float(self.unit.up_direction[2])
        if up not in (-1., 1.):
            raise ValueError("Pipette Z up_direction must be -1 or 1")
        composed = np.empty((3, 3))
        composed[:2] = np.asarray(self.registrar.pixel_to_micron_matrix(objective)) @ matrix[:2]
        composed[2] = [0., 0., up]
        if np.linalg.matrix_rank(composed) != 3:
            raise ValueError("The composed pipette-to-global calibration must be invertible")
        source = "supplied_tip"
        if tip_xy is None:
            trusted = getattr(self.unit, "_trusted_tip_reference", None)
            if trusted is not None:
                tip, pipette_at_tip, stage_at_tip = trusted
                tip_xy = (self._xy(tip) + (matrix @ (home - self._xyz(pipette_at_tip)))[:2]
                          + np.asarray(self.stage.M)[:2, :2] @ (stage[:2] - self._xy(stage_at_tip)))
                source = "trusted_tip_at_home"
            else:
                # Same calibrated projection as reference_position(), evaluated
                # at the recorded home instead of reading the current motors.
                tip_xy = ((matrix @ home + np.asarray(self.unit.r0))[:2]
                          + np.asarray(self.stage.M)[:2, :2] @ stage[:2]
                          + np.asarray(self.stage.r0)[:2])
                source = "calibrated_home_projection"
        tip_xy = self._xy(tip_xy)
        if tip_xy.shape != (2,):
            raise ValueError("Home tip must be one image XY point")
        origin = np.r_[self.pixel_to_global(tip_xy, objective, stage_xy=stage[:2]), float(home_z_um)]
        if safe_pipette_xyz is None:
            safe_pipette_xyz = self._pending_pipette_positions.get("safe")
        if cleaning_bath_xyz is None:
            cleaning_bath_xyz = self._pending_pipette_positions.get("bath")
        safe = None if safe_pipette_xyz is None else self._xyz(safe_pipette_xyz).copy()
        bath = None if cleaning_bath_xyz is None else self._xyz(cleaning_bath_xyz).copy()
        # Configuration describes physical shaft orientation, independently of
        # the measured image-coordinate matrix used for actual tip positions.
        # Applying these angles to composed would rotate the calibration twice.
        angles = np.asarray([self.unit.config.pipette_y_rotation,
                             self.unit.config.pipette_z_rotation], dtype=float)
        direction = self.pipette_shaft_direction(*angles, self.unit.config.pipette_zero_azimuth)
        self._pipette_home_anchor = dict(home=home.copy(), global_home=origin,
            matrix=composed, safe=safe, bath=bath, direction=direction / np.linalg.norm(direction),
            objective=objective, source=source, home_z_um=float(home_z_um), rotations=angles.copy(),
            home_view_center_mm=np.r_[self.stage_to_global(stage[:2], objective) / 1000, 0.])
        self._pending_pipette_positions = {name: point.copy() for name, point in
                                            (("safe", safe), ("bath", bath)) if point is not None}

    def clear_pipette_home_anchor(self):
        self._pipette_home_anchor = None
        self._pending_pipette_positions.clear()

    def set_pipette_safe_position(self, safe_pipette_xyz):
        """Update a saved safe marker without moving the home reference."""
        safe = self._xyz(safe_pipette_xyz).copy()
        self._pending_pipette_positions["safe"] = safe
        anchor = self._pipette_home_anchor
        if anchor is None:
            return
        anchor["safe"] = safe

    def set_pipette_cleaning_bath_position(self, cleaning_bath_xyz):
        """Convert the existing saved manipulator bath position through home."""
        bath = self._xyz(cleaning_bath_xyz).copy()
        self._pending_pipette_positions["bath"] = bath
        if self._pipette_home_anchor is None:
            return
        self._pipette_home_anchor["bath"] = bath

    def pipette_to_global(self, pipette_xyz):
        """Convert raw pipette XYZ microns using the recorded home anchor."""
        anchor = self._pipette_home_anchor
        if anchor is None:
            raise RuntimeError("Record a pipette home anchor before global conversion")
        return anchor["global_home"] + anchor["matrix"] @ (self._xyz(pipette_xyz) - anchor["home"])

    def global_to_pipette(self, global_xyz):
        anchor = self._pipette_home_anchor
        if anchor is None:
            raise RuntimeError("Record a pipette home anchor before global conversion")
        return anchor["home"] + np.linalg.solve(anchor["matrix"], self._xyz(global_xyz) - anchor["global_home"])

    @staticmethod
    def _anchor_name(name):
        if name not in ("stage", "pipette", "cleaning_bath"):
            raise ValueError("Anchor name must be stage, pipette, or cleaning_bath")
        return name

    def set_named_anchor(self, name, global_xyz_um):
        """Store an explicitly supplied global XYZ marker; never moves hardware.

        Coordinates must already use the image-defined origin, including a
        measured or explicitly chosen global Z. Raw motor coordinates are not
        converted here. A stage marker does not replace the saved rim centre.
        """
        name = self._anchor_name(name)
        point = np.asarray(global_xyz_um, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("A named anchor needs three finite global coordinates in microns")
        self._named_anchors_um[name] = point.copy()

    def clear_named_anchor(self, name):
        """Remove an explicit marker; stage then uses the saved rim centre."""
        self._named_anchors_um.pop(self._anchor_name(name), None)

    @property
    def named_anchors_mm(self):
        """Viewer markers in millimetres, returned as independent copies."""
        anchors = {}
        bottom = self.stage_bottom_xyz_mm
        if bottom is not None:
            anchors["stage"] = bottom
        anchors.update({name: (point / 1000).tolist()
                        for name, point in self._named_anchors_um.items()})
        anchor = self._pipette_home_anchor
        if anchor is not None:
            anchors["home"] = (anchor["global_home"] / 1000).tolist()
            for name, raw in (("safe", anchor["safe"]), ("cleaning_bath", anchor["bath"])):
                if raw is not None:
                    anchors[name] = (self.pipette_to_global(raw) / 1000).tolist()
        return anchors

    @property
    def stage_bottom_xyz_mm(self):
        """Saved second-rim centre in the image anchor's reference Z plane."""
        registration = getattr(self.registrar, "registration", None)
        if not registration or not registration.get("frame4"):
            return None
        cache = getattr(self, "_stage_bottom_cache", None)
        if cache is not None and cache[0] is registration:
            return cache[1].copy()
        path = getattr(self.registrar, "reference_manifest_path", None)
        if path is None:
            path = Path(__file__).resolve().parents[3] / "deepLearning" / "globalModel" / "reference" / "scan_reference.json"
        path = Path(path)
        if not path.is_file():
            return None
        observations = json.loads(path.read_text(encoding="utf-8"))["local_frame_observations"]
        frames = {Path(frame["path"]).name: frame for frame in registration["frame4"]}
        points, weights = [], []
        for observation in observations:
            frame = frames.get(observation["frame_name"])
            if frame is None:
                continue
            weight = float(observation["feather_weight_unnormalized"])
            if not np.isfinite(weight) or weight <= 0:
                continue
            points.append(self.registrar.pixel_to_global(
                observation["feature_pixels_from_frame_top_left"], 4,
                frame_index=int(frame["index"])))
            weights.append(weight)
        if not points:
            return None
        center = np.average(np.asarray(points, dtype=float), axis=0, weights=weights)
        result = [float(center[0]) / 1000, float(center[1]) / 1000, 0.0]
        self._stage_bottom_cache = (registration, result)
        return result.copy()

    def registered_session_mosaic(self, metadata, objective=4, downsample_factor=1):
        """Native scan pixels to global microns for the standard scan convention.

        The scan and reference must share the recorded stage coordinate origin.
        A later remount requires reanchoring before this transform can be reused.
        """
        if not metadata.get("flip_x") or not metadata.get("flip_y"):
            raise ValueError("Mosaic conversion requires the registered flip_x/flip_y scan convention")
        matrix = np.asarray(metadata["calibration"]["stage"]["M"], dtype=float)
        if matrix.shape != (2, 2) or not np.isfinite(matrix).all():
            raise ValueError("Scan metadata must contain a finite 2x2 stage matrix")
        pixel_to_stage = np.linalg.inv(-matrix)
        factor = np.asarray(downsample_factor, dtype=float)
        scalar_factor = factor.ndim == 0
        if (not scalar_factor and factor.shape != (2,)) or not np.isfinite(factor).all() or np.any(factor <= 0):
            raise ValueError("Downsample factor must be a positive finite scalar or XY pair")
        resize_ratio = np.full(2, float(factor)) if scalar_factor else factor.copy()
        if (scalar_factor and float(factor) != 1 and float(factor) == metadata.get("downsampled_factor")
                and "canvas_shape" in metadata and "downsampled_canvas_shape" in metadata):
            full_shape = np.asarray(metadata["canvas_shape"][:2], dtype=float)
            small_shape = np.asarray(metadata["downsampled_canvas_shape"][:2], dtype=float)
            if (full_shape.shape != (2,) or small_shape.shape != (2,)
                    or not np.isfinite(full_shape).all() or not np.isfinite(small_shape).all()
                    or np.any(full_shape <= 0) or np.any(small_shape <= 0)):
                raise ValueError("Scan canvas dimensions must be positive and finite")
            # ceil(width/2) and ceil(height/2) make odd-sized scans slightly
            # anisotropic. Use the actual resize, not its nominal 2x label.
            resize_ratio = full_shape[::-1] / small_shape[::-1]
        scale = self._xy(metadata["pixel_scale_xy"]) * resize_ratio
        if not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("Scan pixel scales must be positive")
        origin = self._xy(metadata["output_origin_xy_pixels"])
        reference, global_center = self._reference(objective)
        center = (np.asarray(reference["image_size"], dtype=float) - 1) / 2
        transform = np.eye(3)
        transform[:2, :2] = pixel_to_stage @ np.diag(scale)
        # Native pixel centres obey source=(output+.5)*scale-.5 for both
        # tile and final-image resizes; the stored origin is a tile origin.
        transform[:2, 2] = (pixel_to_stage @ (origin + (scale - 1) / 2)
                             - self._xy(reference["stage_xy_um"])
                             + global_center - pixel_to_stage @ center)
        return transform

    def snapshot(self, scan_pixel_to_global_um=None, objective=40, z_um=0., stage_xy=None,
                 pipette_position=None, pipette_reference_xy=None):
        """Viewer state in mm; defaults to the recorded reference, with no live read.

        z_um is an explicitly supplied focus-plane height, defaulting to the
        reference plane zero. No pipette position or basin floor is invented.
        """
        reference, reference_center = self._reference(objective)
        center = reference_center if stage_xy is None else self.stage_to_global(stage_xy, objective)
        matrix = np.asarray(self.registrar.pixel_to_micron_matrix(objective), dtype=float)
        size = self._camera_size(objective)
        fov = np.linalg.norm(matrix * size[None, :], axis=0)
        # Outer pixel boundaries, in TL/TR/BR/BL order, preserve the full
        # calibrated orientation and shear instead of replacing it by a box.
        pixel_corners = np.array([[-.5, -.5], [size[0] - .5, -.5],
                                  [size[0] - .5, size[1] - .5], [-.5, size[1] - .5]])
        corners_xy = self.pixel_to_global(pixel_corners, objective, stage_xy=stage_xy)
        transform = np.asarray(self.registrar.reference_pixel_to_global_um if
                               scan_pixel_to_global_um is None else scan_pixel_to_global_um, dtype=float)
        if transform.shape != (3, 3) or not np.isfinite(transform).all() or not np.allclose(transform[2], [0, 0, 1]):
            raise ValueError("Scan transform must be a finite 3x3 XY affine matrix")
        if not np.isfinite(z_um):
            raise ValueError("z_um must be finite")
        viewer_transform = np.eye(4)
        viewer_transform[:2, :2] = transform[:2, :2] / 1000
        viewer_transform[:2, 3] = transform[:2, 2] / 1000
        viewer_transform[2, 3] = float(z_um) / 1000
        named_anchors = self.named_anchors_mm
        pipette = named_anchors.get("pipette")
        direction = None
        anchor = self._pipette_home_anchor
        pipette_xy_source = None
        pipette_rig_xyz = None
        marker_direction = None
        if anchor is not None:
            raw = anchor["home"] if pipette_position is None else pipette_position
            pipette = self.pipette_to_global(raw) / 1000
            pipette_xy_source = "home_motor_projection"
            # Reuse CalibratedUnit.reference_position(), not a painted marker.
            # Camera XY and anchored motor Z are distinct coordinate sources.
            if objective == 40 and stage_xy is not None and pipette_reference_xy is not None:
                point = self._xy(pipette_reference_xy)
                if point.shape != (2,):
                    raise ValueError("Pipette reference must be one camera XY point")
                pipette[:2] = self.pixel_to_global(point, 40, stage_xy=stage_xy) / 1000
                pipette_rig_xyz = np.r_[matrix @ (point - (size - 1) / 2) / 1000,
                                       pipette[2]].tolist()
                pipette_xy_source = "calibrated_reference_position"
            pipette = pipette.tolist()
            named_anchors["pipette"] = pipette
            direction = anchor["direction"].tolist()
            marker_direction = np.r_[self.registrar.pixel_to_micron_matrix(40)
                                      @ np.asarray(self.unit.M)[:2, 0], 0.].tolist()
        return dict(scan_pixel_to_world_mm=viewer_transform.tolist(),
                    objective_xyz_mm=[float(center[0]) / 1000, float(center[1]) / 1000, float(z_um) / 1000],
                    fov_size_mm=(fov / 1000).tolist(),
                    fov_corners_mm=(np.column_stack([corners_xy, np.full(4, z_um)]) / 1000).tolist(),
                    pipette_xyz_mm=pipette,
                    pipette_direction_xyz=direction,
                    pipette_marker_direction_xyz=marker_direction,
                    pipette_rotation_yz_degrees=None if anchor is None else anchor["rotations"].tolist(),
                    pipette_anchor_source=None if anchor is None else anchor["source"],
                    pipette_xy_source=pipette_xy_source,
                    pipette_rig_xyz_mm=pipette_rig_xyz,
                    pipette_rig_origin_mm=None if anchor is None else anchor["home_view_center_mm"].tolist(),
                    named_anchors_mm=named_anchors,
                    stage_bottom_xyz_mm=self.stage_bottom_xyz_mm,
                    objective=objective, position_source="recorded_reference" if stage_xy is None else "supplied_stage")
