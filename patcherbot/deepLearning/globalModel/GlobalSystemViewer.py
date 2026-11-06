"""Millimetre-scale global scene, rendered by Matplotlib without OpenGL.

Anchors supply world coordinates, never raw motor coordinates. CAD files retain
native millimetres. The chamber is an estimated CAD reconstruction. Unless an
explicit objective_transform is supplied, the objective's lowest native Z face
is placed at the focus point as a *preview assumption*, not a working distance.
"""
from pathlib import Path
import struct

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.collections import PolyCollection


ASSET_ROOT = Path(__file__).resolve().parent


def read_stl(path):
    """Read binary or ASCII STL triangles without a CAD runtime dependency."""
    data = Path(path).read_bytes()
    if len(data) >= 84:
        count = struct.unpack_from('<I', data, 80)[0]
        if len(data) == 84 + count * 50:
            records = np.frombuffer(data, dtype=np.dtype([
                ('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)),
                ('attribute', '<u2')]), offset=84, count=count)
            triangles = records['vertices'].astype(float)
        else:
            triangles = None
    else:
        triangles = None
    if triangles is None:
        vertices = [list(map(float, line.split()[1:]))
                    for line in data.decode('ascii').splitlines()
                    if line.strip().startswith('vertex ')]
        triangles = np.asarray(vertices, dtype=float).reshape(-1, 3, 3)
    if not triangles.size or not np.isfinite(triangles).all():
        raise ValueError(f'Invalid or empty STL: {path}')
    return triangles


def transform_points(points, matrix):
    """Transform points by a finite affine 4x4 matrix, in millimetres."""
    points = np.asarray(points, dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1])):
        raise ValueError('Expected a finite affine 4x4 transform')
    return points @ matrix[:3, :3].T + matrix[:3, 3]


class TriangleMeshCollection(Poly3DCollection):
    """Vectorized projection of unchanged STL triangles for interactive CAD."""

    def do_3d_projection(self):
        # Non-triangle/colormapped modes retain Matplotlib's standard behavior.
        if (self._A is not None or self._codes3d is not None or self._axlim_clip
                or self._sort_zpos is not None
                or any(s.stop - s.start != 3 for s in self._segslices)):
            return super().do_3d_projection()
        projected = self.axes.M @ self._vec
        projected = projected[:3] / projected[3]
        depth = projected[2].reshape(-1, 3)
        order = np.argsort(-depth.mean(axis=1), kind="stable")
        polygons = projected[:2].T.reshape(-1, 3, 2)[order]
        count = len(order)
        face = self._facecolor3d
        edge = self._edgecolor3d
        self._facecolors2d = np.repeat(face, count, axis=0) if len(face) == 1 else face[order]
        self._edgecolors2d = (np.repeat(edge, count, axis=0) if len(edge) == 1
                              else edge[order] if len(edge) else edge)
        PolyCollection.set_verts(self, polygons, self._closed)
        return float(depth.min()) if depth.size else float("nan")


class GlobalSystemViewer:
    """Scene generator attached to a Matplotlib 3D axes instance."""

    def __init__(self, ax):
        self.ax = ax
        self.chamber_mesh = read_stl(ASSET_ROOT / 'chamber_cad' / 'chamber.stl')
        self.objective_mesh = read_stl(ASSET_ROOT / '40_objective_cad' / 'step_34557.stl')
        self.scan = None
        self.state = None
        self.fov_corners_mm = None
        self.show_cad = True
        self.show_scan = True
        self.show_pipette = True
        self._limits = None
        self._render()
        self.reset_view()

    def set_scan(self, pixels, width, height, pixel_to_world):
        """Set RGB preview and native pixel extents; [px,py,0,1] maps to mm."""
        pixels = np.asarray(pixels)
        if pixels.ndim == 2:
            pixels = np.repeat(pixels[..., None], 3, axis=2)
        if (pixels.ndim != 3 or pixels.shape[2] not in (3, 4)
                or not pixels.shape[0] or not pixels.shape[1]
                or not np.isfinite([width, height]).all()
                or width <= 0 or height <= 0):
            raise ValueError('Scan needs RGB pixels and positive native dimensions')
        matrix = np.asarray(pixel_to_world, dtype=float)
        transform_points(np.zeros((1, 3)), matrix)
        if np.linalg.matrix_rank(matrix[:3, :2]) < 2:
            raise ValueError('Scan transform must define two independent pixel axes')
        stride_y = max(1, int(np.ceil(pixels.shape[0] / 80)))
        stride_x = max(1, int(np.ceil(pixels.shape[1] / 80)))
        colors = pixels[::stride_y, ::stride_x, :3].astype(float)
        if np.issubdtype(pixels.dtype, np.integer):
            colors /= np.iinfo(pixels.dtype).max
        colors = np.clip(colors, 0, 1)
        rows, cols = colors.shape[:2]
        xx, yy = np.meshgrid(np.linspace(0, width, cols + 1),
                             np.linspace(0, height, rows + 1))
        points = transform_points(np.stack([xx, yy, np.zeros_like(xx)], -1), matrix)
        self.scan = (points, colors)
        self._render()

    def clear_scan(self):
        self.scan = None
        self._render()

    def set_state(self, objective_xyz_mm, fov_size_mm, pipette_xyz_mm=None,
                  objective_transform=None, chamber_transform=None):
        """Supply anchored focal centre, FOV width/height and optional tip XYZ.

        Optional CAD transforms map native CAD coordinates directly to world mm.
        The FOV is an XY rectangle; an inclined imaging plane needs a future
        explicit optical-frame transform. Pipette is shown as a tip marker.
        """
        centre = np.asarray(objective_xyz_mm, dtype=float)
        size = np.asarray(fov_size_mm, dtype=float)
        tip = None if pipette_xyz_mm is None else np.asarray(pipette_xyz_mm, dtype=float)
        if centre.shape != (3,) or not np.isfinite(centre).all():
            raise ValueError('Objective focus must be finite XYZ in millimetres')
        if size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0):
            raise ValueError('40x FOV must have positive width and height in millimetres')
        if tip is not None and (tip.shape != (3,) or not np.isfinite(tip).all()):
            raise ValueError('Pipette tip must be finite XYZ in millimetres')
        chamber = np.eye(4) if chamber_transform is None else np.asarray(chamber_transform, float)
        objective = np.eye(4) if objective_transform is None else np.asarray(objective_transform, float)
        if objective_transform is None:
            bounds = self.objective_mesh.reshape(-1, 3)
            native_face = (bounds.min(0) + bounds.max(0)) / 2
            native_face[2] = bounds[:, 2].min()
            objective[:3, 3] = centre - native_face
        transform_points(np.zeros((1, 3)), chamber)
        transform_points(np.zeros((1, 3)), objective)
        corners = np.array([[-.5, -.5, 0], [.5, -.5, 0], [.5, .5, 0], [-.5, .5, 0]])
        self.fov_corners_mm = centre + corners * np.r_[size, 1]
        self.state = (centre, size, tip, objective, chamber)
        self._render()

    def _mesh(self, vertices, color, alpha=1):
        item = Poly3DCollection(vertices, facecolors=color, edgecolors='none',
                                alpha=alpha, rasterized=True)
        self.ax.add_collection3d(item)

    def _render(self):
        ax = self.ax
        elev, azim = ax.elev, ax.azim
        limits = (ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()) if self._limits else None
        ax.clear()
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        points = []
        if self.show_cad:
            chamber_transform = np.eye(4) if self.state is None else self.state[4]
            chamber = transform_points(self.chamber_mesh, chamber_transform)
            self._mesh(chamber, '#91a4b5', .28)
            points.append(chamber.reshape(-1, 3))
        if self.scan is not None and self.show_scan:
            grid, colors = self.scan
            ax.plot_surface(grid[..., 0], grid[..., 1], grid[..., 2],
                            facecolors=colors, rstride=1, cstride=1,
                            shade=False, linewidth=0, antialiased=False)
            points.append(grid.reshape(-1, 3))
        if self.state is not None:
            centre, size, tip, objective_transform, _ = self.state
            if self.show_cad:
                mesh = transform_points(self.objective_mesh, objective_transform)
                self._mesh(mesh, '#e6bd73', .3)
                points.append(mesh.reshape(-1, 3))
            rectangle = np.vstack([self.fov_corners_mm, self.fov_corners_mm[0]])
            ax.plot(*rectangle.T, color='#00a9c9', linewidth=2.5, label='40x FOV')
            ax.text(*centre, '40x', color='#007e96')
            points.append(rectangle)
            if tip is not None and self.show_pipette:
                ax.scatter(*tip, color='#d52489', s=40, depthshade=False)
                ax.text(*tip, ' Pipette tip', color='#d52489')
                points.append(tip[None, :])
            ax.legend(loc='upper left')
        self._bounds = np.concatenate(points) if points else np.array([[-1, -1, -1], [1, 1, 1]])
        ax.view_init(elev=elev, azim=azim)
        if limits is not None:
            ax.set_xlim3d(*limits[0]); ax.set_ylim3d(*limits[1]); ax.set_zlim3d(*limits[2])
        else:
            self._fit(self._bounds)
        ax.figure.canvas.draw_idle()

    def redraw(self):
        """Apply show_cad/show_scan/show_pipette flags without changing anchors."""
        self._render()

    def _fit(self, points):
        low, high = points.min(0), points.max(0)
        centre = (low + high) / 2
        radius = max(float((high - low).max()) * .55, .05)
        self.ax.set_xlim3d(centre[0] - radius, centre[0] + radius)
        self.ax.set_ylim3d(centre[1] - radius, centre[1] + radius)
        self.ax.set_zlim3d(centre[2] - radius, centre[2] + radius)
        self.ax.set_box_aspect((1, 1, 1))
        self._limits = True

    def reset_view(self):
        self._fit(self._bounds)
        self.ax.view_init(elev=30, azim=-60)
        self.ax.figure.canvas.draw_idle()

    def top_view(self):
        self.ax.view_init(elev=90, azim=-90)
        self.ax.figure.canvas.draw_idle()

    def focus_view(self):
        if self.fov_corners_mm is not None:
            self._fit(self.fov_corners_mm)
            self.ax.figure.canvas.draw_idle()


class RegisteredGlobalSystemViewer(GlobalSystemViewer):
    """Registered scene with rigid CAD rotations and optional locked camera views."""

    # User-specified objective working distance: focal plane to optical nose.
    # It is not a guaranteed clearance from the pipette shaft or chamber.
    OBJECTIVE_WORKING_DISTANCE_MM = 3.300

    def __init__(self, ax):
        self.cad_angles = {"objective": (0., 0.), "chamber": (0., 0.)}
        self.view_locked = False
        self.view_name = "isometric"
        self._input_state = None
        self.scan_layers = {}
        self._layer_visibility = {"chamber": True}
        self.stage_bottom_xyz_mm = np.zeros(3)
        self._scene_artists = {}
        self._cad_poses = {}
        self._cad_world_vertices = {}
        self._origin_visible = True
        self._scene_initialized = False
        self._explicit_fov_corners = None
        self._last_cad_angles = None
        super().__init__(ax)
        self._scroll_connection = ax.figure.canvas.mpl_connect("scroll_event", self._scroll_zoom)

    def set_scan_layer(self, name, pixels, width, height, pixel_to_world):
        """Place a named RGB/RGBA texture using native pixel centres and world mm.

        All layers retain their supplied physical Z. Rendering order, rather
        than an invented Z offset, keeps translucent atlas overlays visible.
        Texture geometry is cached until that particular layer changes.
        """
        if not isinstance(name, str) or not name or name == "cad":
            raise ValueError("A scan layer needs a nonempty name other than cad")
        pixels = np.asarray(pixels)
        if pixels.ndim == 2:
            pixels = np.repeat(pixels[..., None], 3, axis=2)
        if (pixels.ndim != 3 or pixels.shape[2] not in (3, 4)
                or min(pixels.shape[:2]) == 0 or not np.isfinite(pixels).all()
                or not np.isfinite([width, height]).all() or min(width, height) <= 0):
            raise ValueError("Texture needs finite RGB/RGBA pixels and positive native dimensions")
        matrix = np.array(pixel_to_world, dtype=float, copy=True)
        transform_points(np.zeros((1, 3)), matrix)
        if np.linalg.matrix_rank(matrix[:3, :2]) < 2:
            raise ValueError("Texture transform needs two independent pixel axes")
        sy = max(1, int(np.ceil(pixels.shape[0] / 128)))
        sx = max(1, int(np.ceil(pixels.shape[1] / 128)))
        colors = pixels[::sy, ::sx].astype(float)
        if np.issubdtype(pixels.dtype, np.integer):
            colors /= np.iinfo(pixels.dtype).max
        colors = np.clip(colors, 0, 1)
        old = self.scan_layers.get(name)
        if (old is not None and old["width"] == width and old["height"] == height
                and np.array_equal(old["pixel_to_world"], matrix)
                and np.array_equal(old["colors"], colors)):
            return
        rows, cols = colors.shape[:2]
        xx, yy = np.meshgrid(np.linspace(-.5, width-.5, cols+1),
                             np.linspace(-.5, height-.5, rows+1))
        grid = transform_points(np.stack([xx, yy, np.zeros_like(xx)], axis=-1), matrix)
        quads = np.stack([grid[:-1, :-1], grid[:-1, 1:], grid[1:, 1:], grid[1:, :-1]], axis=2)
        if old is not None:
            old["artist"].remove()
        priority = 4 if name.startswith("atlas") else (3 if name == "session" else 2)
        artist = Poly3DCollection(quads.reshape(-1, 4, 3), facecolors=colors.reshape(-1, colors.shape[2]),
                                  edgecolors="none", linewidths=0, antialiased=False,
                                  rasterized=True, zorder=priority)
        self.ax.add_collection3d(artist)
        self.scan_layers[name] = dict(points=grid, colors=colors, artist=artist,
            pixel_to_world=matrix, width=float(width), height=float(height))
        self._render()

    def set_scan(self, pixels, width, height, pixel_to_world):
        """Compatibility entry point: a newly opened scan is the session layer."""
        self.set_scan_layer("session", pixels, width, height, pixel_to_world)

    def clear_scan_layer(self, name):
        layer = self.scan_layers.pop(name, None)
        if layer is None:
            return
        layer["artist"].remove()
        self._render()

    def clear_scan(self):
        self.clear_scan_layer("session")

    def set_layer_visible(self, name, visible):
        visible = bool(visible)
        current = self.show_cad if name == "cad" else self._layer_visibility.get(name, True)
        if current == visible:
            return
        if name == "cad":
            self.show_cad = visible
        else:
            self._layer_visibility[name] = visible
        self._render()

    def set_fov_corners(self, corners_mm):
        """Use the supplied four world-space corners, including rotation/tilt."""
        if corners_mm is None:
            if self._explicit_fov_corners is None:
                return
            self._explicit_fov_corners = None
            if self.state is None:
                self.fov_corners_mm = None
            else:
                centre, size = self.state[:2]
                self.fov_corners_mm = centre + np.array([[-.5,-.5,0],[.5,-.5,0],[.5,.5,0],[-.5,.5,0]])*np.r_[size,1]
            self._render()
            return
        corners = np.asarray(corners_mm, dtype=float)
        if corners.shape != (4, 3) or not np.isfinite(corners).all():
            raise ValueError("FOV corners must be four finite XYZ coordinates in millimetres")
        if self._explicit_fov_corners is not None and np.array_equal(corners, self._explicit_fov_corners):
            return
        self._explicit_fov_corners = corners.copy()
        self.fov_corners_mm = corners.copy()
        self._render()

    def mark_origin(self):
        if self._origin_visible:
            return
        self._origin_visible = True
        self._render()

    def set_stage_bottom(self, xyz_mm):
        """Align the STL opening's underside to the saved coverslip bottom."""
        xyz = np.asarray(xyz_mm, dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError("Stage bottom must be finite XYZ in millimetres")
        if np.array_equal(xyz, self.stage_bottom_xyz_mm):
            return
        self.stage_bottom_xyz_mm = xyz.copy()
        self._last_cad_angles = None
        if self._input_state is not None:
            self.set_state(*self._input_state)

    def fit_scans(self):
        """Fit visible image layers independently of the much larger CAD models."""
        points = [layer["points"].reshape(-1, 3) for name, layer in self.scan_layers.items()
                  if self.show_scan and self._layer_visibility.get(name, True)]
        if points:
            self._fit(np.concatenate(points))
            self.ax.figure.canvas.draw_idle()

    def _update_cad_artist(self, name, vertices, pose, color, alpha):
        artist = self._scene_artists.get(name)
        changed = name not in self._cad_poses or not np.array_equal(pose, self._cad_poses[name])
        if changed:
            world = transform_points(vertices, pose)
            self._cad_world_vertices[name] = world
            self._cad_poses[name] = np.array(pose, copy=True)
            if artist is None:
                artist = TriangleMeshCollection(world, facecolors=color, edgecolors="none", alpha=alpha,
                                          rasterized=True, zorder=1)
                self.ax.add_collection3d(artist)
                self._scene_artists[name] = artist
            else:
                artist.set_verts(world)
        artist.set_visible(self.show_cad and self._layer_visibility.get(name, True))
        return self._cad_world_vertices[name].reshape(-1, 3)

    def _render(self):
        """Update dynamic artists without clearing or rebuilding image textures."""
        ax = self.ax
        if not self._scene_initialized:
            ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
            ax.computed_zorder = False
            self._scene_artists["fov"], = ax.plot([], [], [], color="#00a9c9", linewidth=2.5, zorder=6)
            self._scene_artists["pipette"], = ax.plot([], [], [], marker="o", color="#d52489", linestyle="none", zorder=7)
            self._scene_artists["origin"], = ax.plot([0], [0], [0], marker="+", markersize=12,
                                                    markeredgewidth=2, color="#ffc400", linestyle="none", zorder=8)
            self._scene_artists["origin_label"] = ax.text(0, 0, 0, " Image zero", color="#9a6900", zorder=8)
            self._scene_initialized = True
        points = []
        chamber_pose = np.eye(4) if self.state is None else self.state[4]
        chamber = self._update_cad_artist("chamber", self.chamber_mesh, chamber_pose, "#91a4b5", .28)
        if self.show_cad and self._layer_visibility.get("chamber", True):
            points.append(chamber)
        for name, layer in self.scan_layers.items():
            visible = self.show_scan and self._layer_visibility.get(name, True)
            layer["artist"].set_visible(visible)
            if visible:
                points.append(layer["points"].reshape(-1, 3))
        if self.state is not None:
            _, _, tip, objective_pose, _ = self.state
            objective = self._update_cad_artist("objective", self.objective_mesh, objective_pose, "#e6bd73", .3)
            if self.show_cad and self._layer_visibility.get("objective", True):
                points.append(objective)
            pipette = self._scene_artists["pipette"]
            pipette.set_visible(tip is not None and self.show_pipette)
            if tip is not None:
                pipette.set_data_3d([tip[0]], [tip[1]], [tip[2]])
                if self.show_pipette:
                    points.append(tip[None, :])
        fov = self._scene_artists["fov"]
        if self._explicit_fov_corners is not None:
            self.fov_corners_mm = self._explicit_fov_corners.copy()
        fov.set_visible(self.fov_corners_mm is not None)
        if self.fov_corners_mm is not None:
            rectangle = np.vstack([self.fov_corners_mm, self.fov_corners_mm[0]])
            fov.set_data_3d(*rectangle.T)
            points.append(rectangle)
        self._scene_artists["origin"].set_visible(self._origin_visible)
        self._scene_artists["origin_label"].set_visible(self._origin_visible)
        self._bounds = np.concatenate(points) if points else np.array([[-1., -1., -1.], [1., 1., 1.]])
        if not self._limits:
            self._fit(self._bounds)
        ax.figure.canvas.draw_idle()

    @staticmethod
    def rotation_yz(y_degrees, z_degrees):
        """Native local Y rotation followed by world Z rotation; no scaling."""
        y, z = np.deg2rad([y_degrees, z_degrees])
        ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
        rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
        return rz @ ry

    def set_state(self, objective_xyz_mm, fov_size_mm, pipette_xyz_mm=None,
                  objective_transform=None, chamber_transform=None):
        proposed = tuple(None if value is None else np.array(value, dtype=float, copy=True)
                         for value in (objective_xyz_mm, fov_size_mm, pipette_xyz_mm,
                                       objective_transform, chamber_transform))
        if (self._input_state is not None and self._last_cad_angles == self.cad_angles
                and all((a is None and b is None) or (a is not None and b is not None and np.array_equal(a,b))
                        for a,b in zip(proposed,self._input_state))):
            return
        tip = (self.objective_mesh.reshape(-1, 3).min(0) +
               self.objective_mesh.reshape(-1, 3).max(0)) / 2
        tip[2] = self.objective_mesh[..., 2].min()
        objective = np.eye(4) if objective_transform is None else np.array(objective_transform, float, copy=True)
        if objective_transform is None:
            objective[:3, 3] = np.asarray(objective_xyz_mm) - tip + [0., 0., self.OBJECTIVE_WORKING_DISTANCE_MM]
        rotation = np.eye(4)
        rotation[:3, :3] = self.rotation_yz(*self.cad_angles["objective"])
        rotation[:3, 3] = tip - rotation[:3, :3] @ tip
        objective = objective @ rotation
        if objective_transform is None:
            lowest_z = transform_points(self.objective_mesh, objective)[..., 2].min()
            objective[2, 3] += (float(objective_xyz_mm[2])
                                + self.OBJECTIVE_WORKING_DISTANCE_MM - lowest_z)
        chamber = np.eye(4) if chamber_transform is None else np.array(chamber_transform, float, copy=True)
        rotation = np.eye(4)
        rotation[:3, :3] = self.rotation_yz(*self.cad_angles["chamber"])
        floor = np.array([0., 0., self.chamber_mesh[..., 2].min()])
        if chamber_transform is None:
            chamber[:3, 3] = self.stage_bottom_xyz_mm - rotation[:3, :3] @ floor
        super().set_state(objective_xyz_mm, fov_size_mm, pipette_xyz_mm,
                          objective, chamber @ rotation)
        self._input_state = proposed
        self._last_cad_angles = dict(self.cad_angles)
        self.set_view_lock(self.view_locked)

    def set_cad_rotation(self, asset, y_degrees=0., z_degrees=0.):
        if asset not in self.cad_angles or not np.isfinite([y_degrees, z_degrees]).all():
            raise ValueError("Choose objective or chamber and finite Y/Z angles")
        self.cad_angles[asset] = (float(y_degrees), float(z_degrees))
        if self._input_state is not None:
            self.set_state(*self._input_state)

    def set_view_lock(self, locked):
        self.view_locked = bool(locked)
        self.ax.mouse_init(rotate_btn=[] if self.view_locked else 1, pan_btn=2, zoom_btn=3)

    def set_view(self, name):
        angles = {"top": (90, -90), "side": (0, -90), "isometric": (30, -60)}
        if name not in angles:
            raise ValueError("View must be top, side, or isometric")
        self.view_name = name
        self.ax.set_proj_type("ortho")
        self.ax.view_init(*angles[name])
        self.set_view_lock(self.view_locked)
        self.ax.figure.canvas.draw_idle()

    def top_view(self):
        self.set_view("top")

    def side_view(self):
        self.set_view("side")

    def isometric_view(self):
        self.set_view("isometric")

    def zoom(self, factor):
        factor = float(factor)
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("Zoom factor must be positive")
        for get, put in ((self.ax.get_xlim3d, self.ax.set_xlim3d),
                         (self.ax.get_ylim3d, self.ax.set_ylim3d),
                         (self.ax.get_zlim3d, self.ax.set_zlim3d)):
            lo, hi = get()
            centre, half = (lo + hi) / 2, (hi - lo) * factor / 2
            put(centre - half, centre + half)
        self._limits = True
        self.ax.figure.canvas.draw_idle()

    def _scroll_zoom(self, event):
        if event.inaxes is self.ax:
            self.zoom(.8 if event.button == "up" else 1.25)


class GpuGlobalSystemViewer(RegisteredGlobalSystemViewer):
    """OpenGL scene: static CAD buffers and matrix-only updates during movement.

    Requires pyqtgraph.opengl and PyOpenGL. No Matplotlib canvas is created.
    Millimetres and supplied anchors retain the same meaning as the CPU viewer.
    """

    # Display dimensions only: no measured pipette CAD asset is available.
    PIPETTE_SCHEMATIC_LENGTH_MM = 30.0
    PIPETTE_SCHEMATIC_RADIUS_MM = .4
    PIPETTE_SCHEMATIC_TAPER_MM = 8.0
    AXIS_GUIDE_LENGTH_MM = 3.0

    def __init__(self, parent=None, chamber_z_rotation=0.):
        self.chamber_base_z_rotation = float(chamber_z_rotation)
        if not np.isfinite(self.chamber_base_z_rotation):
            raise ValueError("Chamber base Z rotation must be finite degrees")
        import json
        from PyQt5 import QtCore, QtGui
        from OpenGL.GL import GL_DEPTH_TEST
        import pyqtgraph.opengl as gl
        self._gl, self._QtGui = gl, QtGui
        class LockedView(gl.GLViewWidget):
            rotation_locked = False
            def mouseMoveEvent(widget, event):
                if widget.rotation_locked and event.buttons() & QtCore.Qt.LeftButton:
                    widget.mousePos = event.pos()
                    return
                super().mouseMoveEvent(event)
        self.canvas = self.view = LockedView(parent=parent)
        self.canvas.setBackgroundColor((232, 237, 242))
        self.chamber_mesh = read_stl(ASSET_ROOT/'chamber_cad'/'chamber.stl')
        self.objective_mesh = read_stl(ASSET_ROOT/'40_objective_cad'/'step_34557.stl')
        self.cad_angles = {"objective": (0.,0.), "chamber": (0.,0.)}
        self.view_locked = False
        self.view_name = "isometric"
        self.state = self._input_state = None
        self._last_cad_angles = None
        self._explicit_fov_corners = self.fov_corners_mm = None
        self.stage_bottom_xyz_mm = np.zeros(3)
        self.show_cad = self.show_pipette = True
        self.show_scan = True
        self._origin_visible = True
        self._layer_visibility = {"chamber": False}
        self.scan_layers, self.items = {}, {}
        self._cad_poses, self._orientation_cache = {}, {}
        self._mesh_bounds = {}
        self._layer_visibility["chamber"] = True
        self._named_anchors = {}
        self.pipette_pose_matrix = None
        self._pipette_pose_managed = False
        self.pipette_geometry_kind = "schematic_display_only_not_collision_geometry"
        self._bounds = np.array([[-5.,-5.,0.],[5.,5.,5.]])
        self._limits = True
        vertices = self.objective_mesh.reshape(-1,3)
        self._objective_tip = (vertices.min(0)+vertices.max(0))/2
        self._objective_tip[2] = vertices[:,2].max()
        chamber_parameters = json.loads((ASSET_ROOT/'chamber_cad'/'parameters.json').read_text())
        # User-selected provisional basin datum: bottom of the funnel slope,
        # not the lowest exterior face. The CAD has no coverslip solid.
        basin_floor_z = (chamber_parameters['base_thickness']
                         + chamber_parameters['underside_height']
                         - chamber_parameters['funnel_depth'])
        self._chamber_floor = np.array([0.,0.,basin_floor_z])
        self.chamber_floor_datum_status = "provisional_user_selected_bottom_of_funnel_slope"
        for name,mesh,color in (("chamber",self.chamber_mesh,(.57,.65,.71,1.)),
                                ("objective",self.objective_mesh,(.90,.74,.45,1.))):
            points = mesh.reshape(-1, 3)
            low, high = points.min(0), points.max(0)
            self._mesh_bounds[name] = np.array([[x, y, z] for x in (low[0], high[0])
                for y in (low[1], high[1]) for z in (low[2], high[2])])
            item = gl.GLMeshItem(vertexes=mesh.astype(np.float32), smooth=False,
                                shader="shaded", color=color, drawEdges=False, glOptions="opaque")
            self.items[name] = item
            self.canvas.addItem(item)
        self.items["fov"] = gl.GLLinePlotItem(pos=np.zeros((5,3)), color=(0,.8,1,1), width=2, mode="line_strip")
        self.items["pipette"] = gl.GLScatterPlotItem(pos=np.zeros((1,3)), color=(1,.1,.55,1), size=9, pxMode=True)
        self.items["origin"] = gl.GLScatterPlotItem(pos=np.zeros((1,3)), color=(1,.8,0,1), size=10, pxMode=True)
        for name in ("fov","pipette","origin"):
            self.canvas.addItem(self.items[name])
        # Fixed world axes, not motor directions or an additional calibration.
        # At top view: +X is screen right, +Y is screen up, +Z faces the viewer.
        length = self.AXIS_GUIDE_LENGTH_MM
        self.axis_guide_points = np.array([[0,0,0],[length,0,0],
                                          [0,0,0],[0,length,0],
                                          [0,0,0],[0,0,length]],dtype=np.float32)
        axis_colors = np.repeat(np.array([[.85,.12,.12,1.],[.05,.55,.22,1.],[.10,.30,.95,1.]],np.float32),2,axis=0)
        self.items["world_axes"] = gl.GLLinePlotItem(pos=self.axis_guide_points,
            color=axis_colors,width=2,mode="lines",glOptions={GL_DEPTH_TEST:False})
        # An annotation must remain readable when the objective covers origin;
        # bypass depth occlusion without moving the guide's world coordinates.
        self.items["world_axes"].setDepthValue(100)
        self.canvas.addItem(self.items["world_axes"])
        for label,position,color in (("X+",[length+.25,0,0],(190,25,25)),
                                     ("Y+",[0,length+.25,0],(10,125,45)),
                                     ("Z+",[.15,.15,length+.2],(25,65,210))):
            item = gl.GLTextItem(pos=np.array(position),text=label,color=color,font=QtGui.QFont("Arial",10))
            item.setDepthValue(101)
            self.items["axis_"+label[0].lower()] = item
            self.canvas.addItem(item)
        self._pipette_vertices = self._schematic_pipette_mesh()
        self.items["pipette_shaft"] = gl.GLMeshItem(vertexes=self._pipette_vertices,
            smooth=False, shader="shaded", color=(.94,.48,.69,1.), drawEdges=False, glOptions="opaque")
        self.items["pipette_label"] = gl.GLTextItem(
            pos=np.array([0.,0.,self.PIPETTE_SCHEMATIC_LENGTH_MM]), text="Pipette (schematic)",
            color=(135,25,75), font=QtGui.QFont("Arial",10))
        for name in ("pipette_shaft","pipette_label"):
            self.items[name].setVisible(False)
            self.canvas.addItem(self.items[name])
        radius = self.PIPETTE_SCHEMATIC_RADIUS_MM
        length = self.PIPETTE_SCHEMATIC_LENGTH_MM
        self._pipette_bounds = np.array([[x,y,z] for x in (-radius,radius)
                                        for y in (-radius,radius) for z in (0.,length)])
        self._scene_artists = self.items
        self._render()
        self.reset_view()

    def _qmatrix(self, matrix):
        return self._QtGui.QMatrix4x4(*np.asarray(matrix,dtype=float).ravel().tolist())

    @classmethod
    def _schematic_pipette_mesh(cls):
        """Small static taper/cylinder; local origin is the anchored tip."""
        count = 24
        theta = np.arange(count)*2*np.pi/count
        ring = np.column_stack([np.cos(theta),np.sin(theta)])*cls.PIPETTE_SCHEMATIC_RADIUS_MM
        lower = np.column_stack([ring,np.full(count,cls.PIPETTE_SCHEMATIC_TAPER_MM)])
        upper = np.column_stack([ring,np.full(count,cls.PIPETTE_SCHEMATIC_LENGTH_MM)])
        tip,cap = np.zeros(3),np.array([0.,0.,cls.PIPETTE_SCHEMATIC_LENGTH_MM])
        faces = []
        for i in range(count):
            j = (i+1)%count
            faces.extend([[tip,lower[j],lower[i]],[lower[i],lower[j],upper[j]],
                          [lower[i],upper[j],upper[i]],[cap,upper[i],upper[j]]])
        return np.asarray(faces,dtype=np.float32)

    def set_pipette_pose(self, tip_xyz_mm, direction_xyz):
        """Anchor the schematic shaft; direction points away from the tip.

        A missing pose hides both shaft and tip. This supplies no physical
        pipette dimensions or collision envelope, and never commands motion.
        """
        if tip_xyz_mm is None and direction_xyz is None:
            changed = not self._pipette_pose_managed or self.pipette_pose_matrix is not None
            self._pipette_pose_managed = True
            self.pipette_pose_matrix = None
            if changed:
                self._render()
            return
        tip = np.asarray(tip_xyz_mm,dtype=float)
        direction = np.asarray(direction_xyz,dtype=float)
        if (tip.shape != (3,) or direction.shape != (3,) or not np.isfinite(tip).all()
                or not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-12):
            raise ValueError("Pipette pose needs finite XYZ and a nonzero direction away from the tip")
        z = direction/np.linalg.norm(direction)
        reference = np.array([0.,1.,0.]) if abs(z[1]) < .9 else np.array([1.,0.,0.])
        x = np.cross(reference,z);x /= np.linalg.norm(x)
        y = np.cross(z,x)
        matrix = np.eye(4)
        matrix[:3,:3] = np.column_stack([x,y,z])
        matrix[:3,3] = tip
        if self._pipette_pose_managed and self.pipette_pose_matrix is not None and np.array_equal(matrix,self.pipette_pose_matrix):
            return
        self._pipette_pose_managed = True
        self.pipette_pose_matrix = matrix
        transform = self._qmatrix(matrix)
        self.items["pipette_shaft"].setTransform(transform)
        self.items["pipette_label"].setTransform(transform)
        self._render()

    def _orientation(self):
        key = tuple(self.cad_angles["objective"])
        if key not in self._orientation_cache:
            # The supplied CAD has its optical nose at native MAX Z.
            rotation = self.rotation_yz(*key) @ self.rotation_yz(180,0)
            lowest = float((self.objective_mesh.reshape(-1,3) @ rotation.T)[:,2].min())
            self._orientation_cache[key] = (rotation, lowest)
        return self._orientation_cache[key]

    def set_state(self, objective_xyz_mm, fov_size_mm, pipette_xyz_mm=None,
                  objective_transform=None, chamber_transform=None):
        proposed = tuple(None if value is None else np.array(value,dtype=float,copy=True)
                         for value in (objective_xyz_mm,fov_size_mm,pipette_xyz_mm,objective_transform,chamber_transform))
        if (self._input_state is not None and self._last_cad_angles == self.cad_angles
                and all((a is None and b is None) or (a is not None and b is not None and np.array_equal(a,b))
                        for a,b in zip(proposed,self._input_state))):
            return
        centre = np.asarray(objective_xyz_mm,dtype=float)
        rotation, lowest = self._orientation()
        if objective_transform is None:
            objective = np.eye(4)
            objective[:3,:3] = rotation
            objective[:3,3] = centre-rotation@self._objective_tip
            objective[2,3] = centre[2]+self.OBJECTIVE_WORKING_DISTANCE_MM-lowest
        else:
            objective = np.array(objective_transform,dtype=float,copy=True)
            offset = np.eye(4)
            offset[:3,:3] = self.rotation_yz(*self.cad_angles["objective"])
            offset[:3,3] = self._objective_tip-offset[:3,:3]@self._objective_tip
            objective = objective@offset
        chamber_rotation = self.rotation_yz(*self.cad_angles["chamber"])
        if chamber_transform is None:
            # The caller supplies the rig's base orientation. Interactive
            # CAD angles remain offsets; this renderer has no rig defaults.
            chamber_rotation = chamber_rotation @ self.rotation_yz(0.,self.chamber_base_z_rotation)
            chamber = np.eye(4)
            chamber[:3,:3] = chamber_rotation
            chamber[:3,3] = self.stage_bottom_xyz_mm-chamber_rotation@self._chamber_floor
        else:
            chamber = np.array(chamber_transform,dtype=float,copy=True)
            offset = np.eye(4);offset[:3,:3] = chamber_rotation
            chamber = chamber@offset
        GlobalSystemViewer.set_state(self,centre,fov_size_mm,pipette_xyz_mm,objective,chamber)
        self._input_state = proposed
        self._last_cad_angles = dict(self.cad_angles)

    def _render(self):
        if self.state is not None:
            _,_,tip,objective,chamber = self.state
        else:
            tip,objective,chamber = None,np.eye(4),np.eye(4)
            chamber[:3,:3] = self.rotation_yz(*self.cad_angles["chamber"]) @ self.rotation_yz(0.,self.chamber_base_z_rotation)
            chamber[:3,3] = self.stage_bottom_xyz_mm-chamber[:3,:3]@self._chamber_floor
        bounds = []
        for name,pose in (("objective",objective),("chamber",chamber)):
            if name not in self._cad_poses or not np.array_equal(pose,self._cad_poses[name]):
                self.items[name].setTransform(self._qmatrix(pose))
                self._cad_poses[name] = pose.copy()
            self.items[name].setVisible(self.show_cad and self._layer_visibility.get(name,True)
                                        and (name != "objective" or self.state is not None))
            if self.items[name].visible():
                bounds.append(transform_points(self._mesh_bounds[name], pose))
        if self._explicit_fov_corners is not None:
            self.fov_corners_mm = self._explicit_fov_corners.copy()
        self.items["fov"].setVisible(self.fov_corners_mm is not None)
        if self.fov_corners_mm is not None:
            self.items["fov"].setData(pos=np.vstack([self.fov_corners_mm,self.fov_corners_mm[0]]).astype(np.float32))
        if self._pipette_pose_managed:
            tip = None if self.pipette_pose_matrix is None else self.pipette_pose_matrix[:3,3]
        pipette_visible = self.show_pipette and self._layer_visibility.get("pipette",True)
        self.items["pipette"].setVisible(tip is not None and pipette_visible)
        if tip is not None:
            self.items["pipette"].setData(pos=np.asarray(tip,np.float32).reshape(1,3))
            if pipette_visible:
                bounds.append(np.asarray(tip).reshape(1,3))
        shaft_visible = self.pipette_pose_matrix is not None and pipette_visible
        self.items["pipette_shaft"].setVisible(shaft_visible)
        self.items["pipette_label"].setVisible(shaft_visible)
        if shaft_visible:
            bounds.append(transform_points(self._pipette_bounds,self.pipette_pose_matrix))
        if bounds:
            self._bounds = np.concatenate(bounds)
        self.items["origin"].setVisible(self._origin_visible)
        for name,layer in self.scan_layers.items():
            layer["artist"].setVisible(self.show_scan and self._layer_visibility.get(name,True))
        for name,item in self._named_anchors.items():
            item.setVisible(self._layer_visibility.get(name,True))
        self.canvas.update()

    def set_stage_bottom(self, xyz_mm):
        """Place the provisional basin floor (funnel-slope bottom) at world XYZ."""
        previous = self.stage_bottom_xyz_mm.copy()
        RegisteredGlobalSystemViewer.set_stage_bottom(self,xyz_mm)
        if self.state is None and not np.array_equal(previous,self.stage_bottom_xyz_mm):
            self._render()

    def set_named_anchor(self, name, xyz_mm):
        """Show a supplied location; no absent stage/pipette/bath anchor is invented."""
        if xyz_mm is None:
            self.clear_named_anchor(name)
            return
        xyz = np.asarray(xyz_mm,dtype=float)
        if not isinstance(name,str) or not name or xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError("Named anchor needs a name and finite world XYZ millimetres")
        if name not in self._named_anchors:
            item = self._gl.GLScatterPlotItem(pos=xyz.astype(np.float32).reshape(1,3),color=(.2,1,.45,1),size=8,pxMode=True)
            self._named_anchors[name] = item
            self.canvas.addItem(item)
        else:
            self._named_anchors[name].setData(pos=xyz.astype(np.float32).reshape(1,3))
        self.canvas.update()

    def clear_named_anchor(self, name):
        item = self._named_anchors.pop(name, None)
        if item is not None:
            self.canvas.removeItem(item)
            self.canvas.update()

    def set_scan_layer(self,name,pixels,width,height,pixel_to_world):
        from OpenGL.GL import GL_DEPTH_TEST,GL_BLEND,GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA,GL_LEQUAL
        pixels = np.asarray(pixels)
        if pixels.ndim == 2:
            pixels = np.repeat(pixels[...,None],3,axis=2)
        if pixels.ndim != 3 or pixels.shape[2] not in (3,4) or min(pixels.shape[:2]) == 0:
            raise ValueError("Expected RGB/RGBA texture")
        matrix = np.asarray(pixel_to_world,dtype=float)
        transform_points(np.zeros((1,3)),matrix)
        if min(width,height) <= 0 or np.linalg.matrix_rank(matrix[:3,:2]) < 2:
            raise ValueError("Invalid texture dimensions or transform")
        if pixels.dtype != np.uint8:
            pixels = np.clip(pixels*255,0,255).astype(np.uint8)
        if pixels.shape[2] == 3:
            pixels = np.concatenate([pixels,np.full((*pixels.shape[:2],1),255,np.uint8)],axis=2)
        old = self.scan_layers.get(name)
        if old is not None and old["width"] == width and old["height"] == height and np.array_equal(old["pixel_to_world"],matrix) and np.array_equal(old["colors"],pixels):
            return
        if old is not None:
            self.canvas.removeItem(old["artist"])
        item = self._gl.GLImageItem(np.ascontiguousarray(pixels.transpose(1,0,2)),smooth=True)
        item.setGLOptions({GL_DEPTH_TEST:True,GL_BLEND:True,"glBlendFunc":(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA),"glDepthFunc":(GL_LEQUAL,)})
        item.setDepthValue(4 if name.startswith("atlas") else (3 if name == "session" else 2))
        local = np.eye(4);local[0,0]=width/pixels.shape[1];local[1,1]=height/pixels.shape[0];local[:2,3]=-.5
        item.setTransform(self._qmatrix(matrix@local))
        self.canvas.addItem(item)
        corners = transform_points([[-.5,-.5,0],[width-.5,-.5,0],[width-.5,height-.5,0],[-.5,height-.5,0]],matrix)
        self.scan_layers[name] = dict(artist=item,points=corners,colors=pixels.copy(),pixel_to_world=matrix.copy(),width=width,height=height)
        self._render()

    def clear_scan_layer(self,name):
        layer = self.scan_layers.pop(name,None)
        if layer is not None:
            self.canvas.removeItem(layer["artist"])
            self.canvas.update()

    def set_view_lock(self,locked):
        self.view_locked = bool(locked)
        self.canvas.rotation_locked = self.view_locked

    def set_view(self,name):
        angles = {"top":(90,-90),"side":(0,-90),"isometric":(30,-60)}
        if name not in angles:
            raise ValueError("View must be top, side, or isometric")
        self.view_name = name
        elevation,azimuth = angles[name]
        self.canvas.setCameraPosition(elevation=elevation,azimuth=azimuth)

    def _fit(self,points):
        points = np.asarray(points,dtype=float).reshape(-1,3)
        low,high = points.min(0),points.max(0)
        center = (low+high)/2
        span = max(float(np.ptp(points,axis=0).max()),.1)
        self.canvas.setCameraPosition(pos=self._QtGui.QVector3D(*center),distance=span*1.7)

    def reset_view(self):
        self._fit(self._bounds)
        self.set_view("isometric")

    def fit_scans(self):
        points = [layer["points"] for name,layer in self.scan_layers.items() if self.show_scan and self._layer_visibility.get(name,True)]
        if points:
            self._fit(np.concatenate(points))

    def focus_view(self):
        if self.fov_corners_mm is not None:
            self._fit(self.fov_corners_mm)

    def zoom(self,factor):
        factor = float(factor)
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("Zoom factor must be positive")
        self.canvas.setCameraPosition(distance=self.canvas.opts["distance"]*factor)

    def redraw(self):
        self._render()
