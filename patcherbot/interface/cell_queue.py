"""Geometry-based cell assignment and round-robin queues for pipette interfaces."""
import math
import threading
from dataclasses import dataclass

import numpy as np


@dataclass
class _QueuedCell:
    cell: tuple
    pipette_id: str
    attempted: bool = False
    queue_order: int = 0


class CellQueueCoordinator:
    """Shared cell assignment and round-robin queue for active pipette interfaces."""

    def __init__(self, pipette_interfaces, geometry=None, collision_guard_enabled=True):
        self.pipette_interfaces = dict(pipette_interfaces)
        self.pipette_ids = sorted(self.pipette_interfaces, key=self._pipette_sort_key)
        self.geometry = geometry or []
        self.collision_guard_enabled = bool(collision_guard_enabled)
        self._entries = []
        self._ordered_entries = []
        self._workspace_center = np.zeros(2, dtype=float)
        self._has_attempted = False
        self._lock = threading.RLock()
        self._angles = self._build_workspace_angles()

    @staticmethod
    def _pipette_sort_key(pipette_id):
        suffix = str(pipette_id).rsplit("_", 1)[-1]
        return (0, int(suffix)) if suffix.isdigit() else (1, str(pipette_id))

    @staticmethod
    def _canonical_pipette_id(pipette_id):
        suffix = str(pipette_id).rsplit("_", 1)[-1]
        return f"pipette_{suffix}" if suffix.isdigit() else str(pipette_id)

    @staticmethod
    def _cell_xy(cell):
        return np.asarray(cell[0], dtype=float).reshape(-1)[:2]

    def _build_workspace_angles(self):
        angles = {}
        geometry_by_id = {}
        geometry_items = self.geometry.values() if isinstance(self.geometry, dict) else self.geometry
        for item in geometry_items:
            if isinstance(item, dict):
                key = item.get("id", item.get("pipette_id"))
                if key is not None:
                    geometry_by_id[str(key)] = item
        if isinstance(self.geometry, dict):
            for key, item in self.geometry.items():
                if isinstance(item, dict):
                    geometry_by_id.setdefault(str(key), item)

        for pipette_id in self.pipette_ids:
            item = geometry_by_id.get(str(pipette_id))
            if item is None:
                item = geometry_by_id.get(self._canonical_pipette_id(pipette_id), {})
            x_um = float(item.get("x_um", 0.0))
            y_um = float(item.get("y_um", 0.0))
            if x_um or y_um:
                angles[pipette_id] = math.atan2(y_um, x_um)
                continue
            if "angle_deg" in item or "theta_deg" in item:
                angles[pipette_id] = math.radians(float(item.get("angle_deg", item.get("theta_deg", 0.0))))
                continue
            config = self.pipette_interfaces[pipette_id].calibration_config
            rotation = config.resolve_pipette_rotation_matrix(pipette_id)
            direction = np.asarray(rotation, dtype=float) @ np.array([1.0, 0.0, 0.0])
            angles[pipette_id] = math.atan2(direction[1], direction[0])
        return angles

    def _validate_workspace_geometry(self):
        if not self.collision_guard_enabled or len(self.pipette_ids) < 2:
            return
        for index, first_id in enumerate(self.pipette_ids):
            for second_id in self.pipette_ids[index + 1:]:
                delta = abs(
                    (self._angles[first_id] - self._angles[second_id] + math.pi)
                    % (2.0 * math.pi) - math.pi
                )
                if delta < math.radians(1.0):
                    raise RuntimeError(
                        "Cannot assign cells safely: each active pipette needs a distinct "
                        "pipette_geometry direction or pipette-specific rotation matrix."
                    )

    def _workspace_pipette(self, xy):
        if len(self.pipette_ids) == 1:
            return self.pipette_ids[0]
        angle = math.atan2(float(xy[1]), float(xy[0]))
        candidates = []
        for pipette_id in self.pipette_ids:
            delta = abs((angle - self._angles[pipette_id] + math.pi) % (2.0 * math.pi) - math.pi)
            candidates.append((delta, self._pipette_sort_key(pipette_id), pipette_id))
        return min(candidates)[2]

    def add_cell(self, cell):
        with self._lock:
            self._validate_workspace_geometry()
            entry = _QueuedCell(tuple(cell), "")
            if self._entries and self._has_attempted:
                entry.pipette_id = self._workspace_pipette(
                    self._cell_xy(entry.cell) - self._workspace_center
                )
                existing_orders = [queued.queue_order for queued in self._entries
                                   if queued.pipette_id == entry.pipette_id]
                entry.queue_order = max(existing_orders, default=-1) + 1
                self._entries.append(entry)
                self._rebuild_order()
            else:
                self._entries.append(entry)
                self._assign_all()
            return entry.cell

    def _assign_all(self):
        if not self._entries:
            self._ordered_entries = []
            self._workspace_center = np.zeros(2, dtype=float)
            self._has_attempted = False
            return
        points = [self._cell_xy(entry.cell) for entry in self._entries]
        centroid = np.mean(points, axis=0)
        center_index = min(range(len(points)), key=lambda i: float(np.linalg.norm(points[i] - centroid)))
        center_point = points[center_index]
        self._workspace_center = center_point.copy()
        counts = {pipette_id: 0 for pipette_id in self.pipette_ids}
        for index, entry in enumerate(self._entries):
            if index == center_index:
                continue
            entry.pipette_id = self._workspace_pipette(points[index] - center_point)
            counts[entry.pipette_id] += 1
        center_owner = min(self.pipette_ids, key=lambda pid: (counts[pid], self._pipette_sort_key(pid)))
        self._entries[center_index].pipette_id = center_owner
        grouped = {pipette_id: [] for pipette_id in self.pipette_ids}
        for entry in self._entries:
            grouped[entry.pipette_id].append(entry)
        for queue in grouped.values():
            queue.sort(key=lambda entry: float(np.linalg.norm(self._cell_xy(entry.cell) - center_point)))
            for queue_order, entry in enumerate(queue):
                entry.queue_order = queue_order
        self._rebuild_order()

    def _rebuild_order(self):
        grouped = {pipette_id: [] for pipette_id in self.pipette_ids}
        for entry in self._entries:
            grouped[entry.pipette_id].append(entry)
        for queue in grouped.values():
            queue.sort(key=lambda entry: entry.queue_order)

        self._ordered_entries = []
        queue_index = 0
        while any(queue_index < len(grouped[pipette_id]) for pipette_id in self.pipette_ids):
            for pipette_id in self.pipette_ids:
                if queue_index < len(grouped[pipette_id]):
                    self._ordered_entries.append(grouped[pipette_id][queue_index])
            queue_index += 1

    @property
    def cells(self):
        with self._lock:
            return [entry.cell for entry in self._ordered_entries]

    def next_cell(self, pipette_id):
        with self._lock:
            return next((entry.cell for entry in self._ordered_entries
                         if entry.pipette_id == pipette_id), None)

    def mark_attempted(self, cell):
        with self._lock:
            entry = self._find(cell)
            if entry is not None:
                entry.attempted = True
                self._has_attempted = True
                self._rebuild_order()

    def remove_cell(self, cell):
        with self._lock:
            entry = self._find(cell)
            if entry is None:
                return False
            self._entries.remove(entry)
            if not self._entries:
                self._has_attempted = False
                self._assign_all()
            elif not self._has_attempted:
                self._assign_all()
            else:
                self._rebuild_order()
            return True

    def remove_last_cell(self):
        with self._lock:
            if not self._ordered_entries:
                return None
            cell = self._ordered_entries[-1].cell
            self.remove_cell(cell)
            return cell

    def metadata(self, cell):
        with self._lock:
            entry = self._find(cell)
            if entry is None:
                return None
            local_queue = [candidate for candidate in self._ordered_entries if candidate.pipette_id == entry.pipette_id]
            pipette = self.pipette_interfaces[entry.pipette_id]
            return {
                "pipette_id": entry.pipette_id,
                "pipette_index": getattr(
                    pipette,
                    "pipette_index",
                    self.pipette_ids.index(entry.pipette_id),
                ),
                "pipette_queue_index": local_queue.index(entry),
                "overall_queue_index": self._ordered_entries.index(entry),
            }

    def _find(self, cell):
        return next((entry for entry in self._entries if entry.cell is cell), None)