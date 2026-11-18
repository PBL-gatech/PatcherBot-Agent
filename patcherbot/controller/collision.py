"""Collision guard: the active pipette stays on its own side of every other tip.

The rule
--------
Seen from above (XY only, Z ignored), for the pipette being moved (the
*active* pipette) and each other pipette:

1. draw the line joining the two tips;
2. draw the line perpendicular to it through the *other* pipette's tip;
3. the active tip is on one side of that line; it may be moved
   *automatically* to any point on that same side, keeping at least
   ``collision_tolerance_um`` from the line.

With several other pipettes, the target must satisfy this for each of them.
Because the line goes through the other tip, the active pipette can come up
to ``collision_tolerance_um`` of the other tip but never get beyond it.

The limit may not cut through the other pipette's shaft (the axis running
from its tip back to its manipulator). The perpendicular through the other
tip has two halves, one on each side of the line joining the tips. If the
shaft points back towards the active pipette, the half on the shaft's side
is replaced by the shaft itself; the other half is left unchanged. The limit
is then a bent line (perpendicular on one side, shaft on the other) and the
active tip must keep ``collision_tolerance_um`` from both parts. The shaft
direction comes from the pipette's calibration (its manipulator x axis,
which the pipette lies along), i.e. the same frame the tips are placed in.

* A pipette that is already past a line (e.g. after a manual jog) may still
  be moved as long as the move doesn't take it further past; retreating
  always works.
* Manual moves (keyboard jogs, raise/lower) run inside ``guard.manual()``
  and are never checked.
* A pipette the guard can't place (not calibrated, or no known tip
  position yet) is left out: its own moves aren't checked and it doesn't
  restrict the others. ``status()`` lists such pipettes so the GUI can warn.

Where pipettes are
------------------
Positions are compared in the *stage frame*, in um: camera pixels minus the
stage reference position, converted with the stage calibration. The
manipulators ride on the stage, so a fixed tip keeps its coordinates when
the stage moves. A tip is placed the same way the camera overlay places it:
from an anchor (latest confirmed tip detection, then trusted detection,
then "tip at the image centre when pipette and stage are at home") plus the
pipette calibration matrix for the displacement since. The calibration
offset ``r0`` is never used, because it is unreliable after calibration.

Where it is enforced
--------------------
``GuardedPipetteDevice`` wraps each pipette's manipulator device; every
discrete and velocity move command, from any code path (interface
commands, patch phases, calibration helpers, cleaning, home/safe moves),
passes through it.
"""
import contextlib
import logging
import threading

import numpy as np

from .errors import AutopatchError

__all__ = ["CollisionError", "WorkspaceGuard", "GuardedPipetteDevice", "boundary_normals",
           "boundary_margin", "move_allowed"]

LOGGER = logging.getLogger(__name__)


class CollisionError(AutopatchError):
    """A pipette was asked to move past another pipette's tip."""


# ------------------------------------------------------------------
# The rule (pure functions)
# ------------------------------------------------------------------
def boundary_normals(active, other, other_shaft=None):
    """Unit normals of the straight pieces of the limit through ``other``'s
    tip, each pointing from the active side to the far side.

    * The perpendicular to the line joining the tips (always).
    * The shaft's line, when ``other_shaft`` (unit XY vector from ``other``'s
      tip towards its manipulator) points back towards the active side, i.e.
      when the perpendicular's half on the shaft's side would cut the shaft.
      Requiring the active side of both lines replaces just that half of the
      perpendicular with the shaft and leaves the other half as it is.

    Empty if the tips coincide.
    """
    active, other = (np.asarray(v, dtype=float)[:2] for v in (active, other))
    offset = other - active
    distance = float(np.linalg.norm(offset))
    if distance < 1e-9:
        return []
    normal = offset / distance
    normals = [normal]
    if other_shaft is not None:
        shaft = np.asarray(other_shaft, dtype=float)[:2]
        if float(np.dot(normal, shaft)) < 0:
            across = np.array([-shaft[1], shaft[0]])
            if float(np.dot(across, normal)) < 0:
                across = -across          # point away from the active tip
            normals.append(across)
    return normals


def _margins(active, other, point, tolerance, other_shaft):
    other, point = (np.asarray(v, dtype=float)[:2] for v in (other, point))
    normals = boundary_normals(active, other, other_shaft)
    if not normals:
        # Coincident tips: no direction to draw the line along; fall back to
        # plain distance from the other tip.
        return [float(np.linalg.norm(point - other)) - tolerance]
    return [float(np.dot(other - point, normal)) - tolerance for normal in normals]


def boundary_margin(active, other, point, tolerance, other_shaft=None):
    """Signed distance (um) from ``point`` to the limit set by ``other``.

    The limit is the line through ``other``'s tip perpendicular to the line
    joining the tips, with its half on the shaft side replaced by ``other``'s
    shaft when needed (``boundary_normals``), moved ``tolerance`` towards the
    active side. Positive = on the active side with room to spare; negative
    = that far past the limit.
    """
    return min(_margins(active, other, point, tolerance, other_shaft))


def move_allowed(active, target, other, tolerance, other_shaft=None):
    """True if moving the active tip from ``active`` to ``target`` stays on its
    side of each piece of ``other``'s limit, or at least doesn't go further
    past any piece it is already past."""
    start = _margins(active, other, active, tolerance, other_shaft)
    end = _margins(active, other, target, tolerance, other_shaft)
    return all(after >= 0 or (before < 0 and after >= before - 1e-9)
               for before, after in zip(start, end))


def _usable(unit):
    return bool(getattr(unit, "calibrated", False)) and not getattr(unit, "must_be_recalibrated", False)


class WorkspaceGuard:
    """Keeps the active pipette on its side of every other tip (see the module docstring).

    calibrated_units: ``{pipette_key: CalibratedUnit}``.
    calibrated_stage: the shared ``CalibratedStage``.
    config: the shared calibration config (collision_tolerance_um,
        collision_velocity_lookahead_s).
    home_anchor: optional ``f(key) -> (image_xy, home_um, home_stage_xy) or None``.
    camera: optional, only to word directions ("to its right") as displayed.
    on_blocked: optional ``f(message)`` called before a CollisionError is raised.
    """

    def __init__(self, calibrated_units, calibrated_stage, config, home_anchor=None,
                 camera=None, on_blocked=None):
        self.units = calibrated_units
        self.stage = calibrated_stage
        self.config = config
        self.home_anchor = home_anchor
        self.camera = camera
        self.on_blocked = on_blocked
        # On/off lives in config.collision_guard_enabled when the config has
        # it (Calibration tab > Collision, saved in the YAML); else here.
        self._enabled = True
        self._local = threading.local()
        self._warned = set()

    # ---------------- settings ----------------
    @property
    def tolerance(self):
        return float(getattr(self.config, "collision_tolerance_um", 100.0))

    @property
    def lookahead_s(self):
        return float(getattr(self.config, "collision_velocity_lookahead_s", 0.5))

    @property
    def enabled(self):
        return bool(getattr(self.config, "collision_guard_enabled", self._enabled))

    def set_enabled(self, enabled):
        self._enabled = bool(enabled)
        if hasattr(self.config, "collision_guard_enabled"):
            self.config.collision_guard_enabled = self._enabled
        self._warned.discard("disabled")
        if not self.enabled:
            LOGGER.warning("Collision guard switched OFF: automatic pipette moves are not checked")
        else:
            LOGGER.info("Collision guard switched ON")

    @contextlib.contextmanager
    def manual(self):
        """Moves made inside this block (keyboard jogs etc.) are not checked."""
        self._local.depth = getattr(self._local, "depth", 0) + 1
        try:
            yield
        finally:
            self._local.depth -= 1

    def is_manual(self):
        return getattr(self._local, "depth", 0) > 0

    def should_check(self, key):
        """Quick test (no device access) whether moves of ``key`` need checking."""
        if len(self.units) < 2 or self.is_manual():
            return False
        if not self.enabled:
            self._warn_once("disabled", "Collision guard is OFF; automatic pipette moves are not checked")
            return False
        return True

    # ---------------- locating pipettes ----------------
    def _stage_matrix(self):
        return np.asarray(self.stage.M, dtype=float)[:2, :2]

    def _stage_pixels(self, stage_xy):
        """Stage reference position (pixels, XY) when the stage is at ``stage_xy`` um."""
        stage_xy = np.asarray(stage_xy, dtype=float).reshape(-1)[:2]
        return self._stage_matrix() @ stage_xy + np.asarray(self.stage.r0, dtype=float).reshape(-1)[:2]

    def _anchor(self, key):
        """``(stage-frame tip XY px, manipulator um)`` for one known tip position, or None."""
        unit = self.units[key]
        candidates = [getattr(unit, "_tip_reference", None),
                      getattr(unit, "_trusted_tip_reference", None)]
        if self.home_anchor is not None:
            candidates.append(self.home_anchor(key))
        for anchor in candidates:
            if anchor is None:
                continue
            tip, position, stage_xy = anchor
            tip = np.asarray(tip, dtype=float).reshape(-1)[:2]
            position = np.asarray(position, dtype=float).reshape(-1)
            stage_xy = np.asarray(stage_xy, dtype=float).reshape(-1)
            if (tip.shape == (2,) and position.size >= 2 and stage_xy.size >= 2
                    and np.isfinite(tip).all() and np.isfinite(position).all()
                    and np.isfinite(stage_xy[:2]).all()):
                return tip - self._stage_pixels(stage_xy), position
        return None

    def unlocated_reason(self, key):
        """Why ``key`` can't be placed (None if it can). No device access."""
        if not getattr(self.stage, "calibrated", False):
            return "the stage is not calibrated"
        unit = self.units[key]
        if not getattr(unit, "calibrated", False):
            return "it is not calibrated"
        matrix = np.asarray(unit.M, dtype=float)
        if matrix.shape[0] < 2 or matrix.shape[1] < 2 or not np.isfinite(matrix[:2, :2]).all():
            return "its calibration matrix is invalid"
        if self._anchor(key) is None:
            return "its tip position is unknown (detect the pipette or store its home position)"
        return None

    def locate(self, key, position_um):
        """Tip of ``key`` at manipulator ``position_um``, in stage-frame um (XY), or None."""
        if self.unlocated_reason(key) is not None:
            return None
        tip_px, anchor_um = self._anchor(key)
        position_um = np.asarray(position_um, dtype=float).reshape(-1)
        matrix = np.asarray(self.units[key].M, dtype=float)[:2, :2]
        stage_frame_px = tip_px + matrix @ (position_um[:2] - anchor_um[:2])
        return np.linalg.solve(self._stage_matrix(), stage_frame_px)

    def shaft_direction(self, key):
        """Unit XY vector (stage-frame um) from ``key``'s tip back towards its
        manipulator, or None. The pipette lies along its manipulator's x axis
        (+x drives the tip forward), so the shaft points along -x as measured
        by the calibration."""
        if self.unlocated_reason(key) is not None:
            return None
        x_axis_px = np.asarray(self.units[key].M, dtype=float)[:2, 0]
        x_axis = np.linalg.solve(self._stage_matrix(), x_axis_px)
        length = float(np.linalg.norm(x_axis))
        return None if length < 1e-12 or not np.isfinite(length) else -x_axis / length

    def cell_location(self, cell):
        """A queued cell (``stage_reference_px - image_px``) in stage-frame um (XY)."""
        cell = np.asarray(cell, dtype=float).reshape(-1)
        if cell.size < 2 or not np.isfinite(cell[:2]).all():
            raise CollisionError(f"Cell position {cell} needs finite x and y")
        return np.linalg.solve(self._stage_matrix(), -cell[:2])

    def status(self):
        """``{"enabled", "unlocated": {key: reason}}``; no device access."""
        unlocated = {}
        if len(self.units) > 1:
            for key in self.units:
                reason = self.unlocated_reason(key)
                if reason is not None:
                    unlocated[key] = reason
        return {"enabled": self.enabled, "unlocated": unlocated}

    def _seeds(self, mover):
        """``{key: (tip XY, shaft direction)}`` for every located pipette except ``mover``."""
        seeds = {}
        for key, unit in self.units.items():
            if key == mover or self.unlocated_reason(key) is not None:
                continue
            seeds[key] = (self.locate(key, unit.position()), self.shaft_direction(key))
        return seeds

    # ---------------- checks ----------------
    def _warn_once(self, tag, message):
        if tag not in self._warned:
            self._warned.add(tag)
            LOGGER.warning(message)

    def _mover_seed(self, key, current_um):
        reason = self.unlocated_reason(key)
        if reason is not None:
            self._warn_once(("unlocated", key, reason),
                            f"Collision guard can't check {key}'s moves: {reason}")
            return None
        self._warned = {tag for tag in self._warned if not (isinstance(tag, tuple) and tag[1] == key)}
        return self.locate(key, current_um)

    def _direction_word(self, from_um, to_um):
        """Where ``to_um`` is relative to ``from_um`` in the camera image."""
        delta_px = self._stage_matrix() @ (np.asarray(to_um, dtype=float) - np.asarray(from_um, dtype=float))
        dx, dy = float(delta_px[0]), float(delta_px[1])
        if getattr(self.camera, "flipped", False):
            dx = -dx
        if abs(dx) >= abs(dy):
            return "to its right" if dx > 0 else "to its left"
        return "below it" if dy > 0 else "above it"

    def _block(self, message):
        LOGGER.error(message)
        if self.on_blocked is not None:
            try:
                self.on_blocked(message)
            except Exception:
                LOGGER.exception("Collision guard could not report a blocked move")
        raise CollisionError(message)

    def _check_points(self, key, current_um, targets_um, what):
        seed = self._mover_seed(key, current_um)
        if seed is None:
            return
        tolerance = self.tolerance
        worst = None
        for target_um in targets_um:
            target = self.locate(key, target_um)
            for other_key, (other, shaft) in self._seeds(key).items():
                if move_allowed(seed, target, other, tolerance, shaft):
                    continue
                overshoot = -boundary_margin(seed, other, target, tolerance, shaft)
                if worst is None or overshoot > worst[0]:
                    worst = (overshoot, other_key, other, target)
        if worst is None:
            return
        overshoot, other_key, other, target = worst
        self._block(
            f"Collision guard stopped {key}: {what} would take it {overshoot:.0f} um past "
            f"the limit at {other_key}'s tip, which is {self._direction_word(target, other)} in "
            f"the camera image. {key} must stay on its own side of the line through {other_key}'s "
            f"tip (never across {other_key}'s shaft), at least {tolerance:.0f} um short of it. "
            f"Move {other_key} away first, or switch off the collision guard (status-bar shield) "
            f"(debugging only).")

    def check_move(self, key, current_um, target_um):
        """Check a discrete move of ``key`` from ``current_um`` to ``target_um``.

        Manipulator axes don't move in a straight line, so every corner of the
        XY box between start and target is checked (the path stays inside it).
        """
        if not self.should_check(key):
            return
        current = np.asarray(current_um, dtype=float).reshape(-1)
        target = np.asarray(target_um, dtype=float).reshape(-1)
        corners = [target]
        for x, y in ((current[0], target[1]), (target[0], current[1])):
            corner = target.copy()
            corner[0], corner[1] = x, y
            corners.append(corner)
        self._check_points(key, current, corners, "this move")

    def check_velocity(self, key, current_um, velocity_um_s):
        """Check a velocity command by projecting it ``collision_velocity_lookahead_s`` ahead."""
        if not self.should_check(key):
            return
        velocity = np.asarray(velocity_um_s, dtype=float).reshape(-1)
        if velocity.size < 2 or not np.any(velocity[:2]):
            return  # stops and pure Z motion are always allowed
        current = np.asarray(current_um, dtype=float).reshape(-1)
        ahead = current.copy()
        ahead[:2] = current[:2] + velocity[:2] * self.lookahead_s
        self._check_points(key, current, [ahead], "this velocity command")

    def check_cell(self, key, cell):
        """Raise CollisionError if ``cell`` lies past another pipette's tip."""
        if not self.should_check(key):
            return
        seed = self._mover_seed(key, self.units[key].position())
        if seed is None:
            return
        point = self.cell_location(cell)
        tolerance = self.tolerance
        for other_key, (other, shaft) in self._seeds(key).items():
            margin = boundary_margin(seed, other, point, tolerance, shaft)
            if margin < 0:
                self._block(
                    f"{key} may not patch this cell: it is {-margin:.0f} um past the limit at "
                    f"{other_key}'s tip ({other_key} is {self._direction_word(point, other)} in the "
                    f"camera image; tolerance {tolerance:.0f} um). Patch it with {other_key}, move "
                    f"{other_key} away first, or switch off the collision guard (status-bar shield) "
                    f"(debugging only).")


class GuardedPipetteDevice:
    """A pipette's manipulator device with every move checked by a WorkspaceGuard.

    Installed in place of ``unit.dev`` on both the raw ManipulatorUnit and the
    CalibratedUnit of a pipette, so calibration helpers, phases and interface
    commands all go through it. ``raw`` is the wrapped device; anything that
    isn't a move command is passed straight to it.
    """

    def __init__(self, dev, pipette_key, axes, guard):
        object.__setattr__(self, "raw", dev)
        object.__setattr__(self, "pipette_key", pipette_key)
        object.__setattr__(self, "axes", [int(axis) for axis in axes])
        object.__setattr__(self, "guard", guard)

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def __setattr__(self, name, value):
        setattr(self.raw, name, value)

    def __repr__(self):
        return f"GuardedPipetteDevice({self.raw!r}, {self.pipette_key!r})"

    # ---- helpers ----
    def _current(self):
        return np.asarray(self.raw.position_group(self.axes), dtype=float).reshape(-1)

    def _index(self, axis):
        try:
            return self.axes.index(int(axis))
        except (TypeError, ValueError):
            return None

    def _check(self, make_target):
        if not self.guard.should_check(self.pipette_key):
            return
        current = self._current()
        self.guard.check_move(self.pipette_key, current, make_target(current.copy()))

    def _check_velocity(self, velocity, axes):
        if not self.guard.should_check(self.pipette_key):
            return
        velocity = np.asarray(velocity, dtype=float).reshape(-1)
        unit_velocity = np.zeros(len(self.axes))
        if axes is None:
            unit_velocity[:min(len(velocity), len(self.axes))] = velocity[:len(self.axes)]
        else:
            for value, axis in zip(velocity, np.atleast_1d(axes)):
                index = self._index(axis)
                if index is not None:
                    unit_velocity[index] = value
        try:
            self.guard.check_velocity(self.pipette_key, self._current(), unit_velocity)
        except CollisionError:
            self._stop()
            raise

    def _stop(self):
        """Stop this pipette's axes. Devices differ: Scientifica's ``stop()``
        takes no axis, fake/base manipulators take one axis at a time."""
        try:
            self.raw.stop()
        except TypeError:
            for axis in self.axes:
                self.raw.stop(axis)

    # ---- discrete moves ----
    def absolute_move(self, x, axis, *args, **kwargs):
        def target(current):
            index = self._index(axis)
            if index is not None:
                current[index] = float(x)
            return current
        self._check(target)
        return self.raw.absolute_move(x, axis, *args, **kwargs)

    def absolute_move_group(self, x, axes, *args, **kwargs):
        def target(current):
            for value, axis in zip(np.asarray(x, dtype=float).reshape(-1), np.atleast_1d(axes)):
                index = self._index(axis)
                if index is not None:
                    current[index] = value
            return current
        self._check(target)
        return self.raw.absolute_move_group(x, axes, *args, **kwargs)

    def relative_move(self, x, axis, *args, **kwargs):
        def target(current):
            index = self._index(axis)
            if index is not None:
                current[index] += float(x)
            return current
        self._check(target)
        return self.raw.relative_move(x, axis, *args, **kwargs)

    def relative_move_group(self, x, axes, *args, **kwargs):
        def target(current):
            for value, axis in zip(np.asarray(x, dtype=float).reshape(-1), np.atleast_1d(axes)):
                index = self._index(axis)
                if index is not None:
                    current[index] += value
            return current
        self._check(target)
        return self.raw.relative_move_group(x, axes, *args, **kwargs)

    # ---- velocity moves ----
    def absolute_move_group_velocity(self, vel, *args, **kwargs):
        axes = kwargs.get("axes", args[0] if args else None)
        self._check_velocity(vel, axes)
        return self.raw.absolute_move_group_velocity(vel, *args, **kwargs)

    def relative_move_group_velocity(self, vel, *args, **kwargs):
        axes = kwargs.get("axes", args[0] if args else None)
        self._check_velocity(vel, axes)
        return self.raw.relative_move_group_velocity(vel, *args, **kwargs)
