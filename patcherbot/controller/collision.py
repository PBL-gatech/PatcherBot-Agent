"""Minimal collision rule between the primary pipette and the others.

Geometry (operator's view): pipette_0's manipulator is on the left, the
others come from the right. In pipette_0's manipulator frame, -x points
back along pipette_0, away from the centre, so the region with x smaller
than pipette_0's current x is where pipette_0's body is. The rule is:

    a secondary pipette may not go to (or be sent to patch) a point whose
    x, expressed in pipette_0's manipulator frame, is smaller than
    pipette_0's current x.

How points move between frames
------------------------------
Everything is compared in the *stage frame*: camera pixels minus the stage
reference position (``CalibratedStage.reference_position``). The pipettes
and the slice ride on the stage, so a point fixed to the slice or to a
manipulator position keeps its stage-frame coordinates wherever the stage
goes.

A pipette's tip is placed in that frame the same way the camera overlay
(``AutoPatchInterface.update_camera_cell_list``) and the patch phases place
it: from an *anchor* ``(tip_px, manipulator_um, stage_um)`` plus the
calibrated matrix ``M`` for the displacement since the anchor. The anchor
is the latest confirmed tip detection (``_tip_reference``, then
``_trusted_tip_reference``), else "tip at the image centre when the
pipette and stage are at their home positions". The absolute calibration
offset ``r0`` is used only when no anchor exists. So the guard draws the
boundary where the overlay shows pipette_0.

Only XY matters: the calibration maps encoder XY to image XY and leaves Z
separate, so a pipette's x in pipette_0's frame doesn't depend on height.
Only pipette_0 vs each secondary is checked; nothing is checked between
secondaries, and velocity moves are not checked.
"""
import logging

import numpy as np

from .errors import AutopatchError

__all__ = ["CollisionError", "PrimaryPipetteGuard"]

LOGGER = logging.getLogger(__name__)


class CollisionError(AutopatchError):
    """A secondary pipette was asked to enter pipette_0's side."""


def _usable(unit):
    return bool(getattr(unit, "calibrated", False)) and not getattr(unit, "must_be_recalibrated", False)


def _finite(*arrays):
    return all(np.isfinite(np.asarray(a, dtype=float)).all() for a in arrays)


class PrimaryPipetteGuard:
    """Keeps every secondary pipette on its side of pipette_0.

    calibrated_units: ``{pipette_key: CalibratedUnit}``; the primary is
        ``primary_key`` (default: the first key).
    calibrated_stage: the shared ``CalibratedStage``.
    home_anchor: optional ``f(key) -> (image_xy, home_um, home_stage_xy)``
        or None, the "tip at image centre at home" fallback anchor.
    """

    def __init__(self, calibrated_units, calibrated_stage, primary_key=None, home_anchor=None):
        self.units = calibrated_units
        self.stage = calibrated_stage
        self.primary_key = primary_key if primary_key is not None else next(iter(calibrated_units))
        self.home_anchor = home_anchor
        self._warned_uncalibrated = set()

    @property
    def primary(self):
        return self.units[self.primary_key]

    # ---------------- frame mapping ----------------
    def _stage_pixels(self, stage_xy):
        """Stage reference position (pixels, XY) when the stage is at ``stage_xy`` um."""
        stage_xy = np.asarray(stage_xy, dtype=float).reshape(-1)[:2]
        return np.dot(np.asarray(self.stage.M, dtype=float)[:2, :2], stage_xy) \
            + np.asarray(self.stage.r0, dtype=float).reshape(-1)[:2]

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
            if tip.shape == (2,) and position.size >= 2 and _finite(tip, position, stage_xy):
                return tip - self._stage_pixels(stage_xy), position
        return None

    def _matrix_xy(self, key):
        return np.asarray(self.units[key].M, dtype=float)[:2, :2]

    def stage_frame_xy(self, key, position_um):
        """Tip of pipette ``key`` at manipulator ``position_um``, in stage-frame pixels (XY)."""
        position_um = np.asarray(position_um, dtype=float).reshape(-1)
        anchor = self._anchor(key)
        if anchor is not None:
            tip_px, anchor_um = anchor
            return tip_px + self._matrix_xy(key) @ (position_um[:2] - anchor_um[:2])
        unit = self.units[key]
        return (np.dot(np.asarray(unit.M, dtype=float), position_um)
                + np.asarray(unit.r0, dtype=float))[:2]

    def primary_xy_from_stage_frame(self, stage_xy_px):
        """Pipette_0 manipulator XY (um) whose tip sits at stage-frame pixel ``stage_xy_px``."""
        stage_xy_px = np.asarray(stage_xy_px, dtype=float).reshape(-1)[:2]
        anchor = self._anchor(self.primary_key)
        matrix = self._matrix_xy(self.primary_key)
        if anchor is not None:
            tip_px, anchor_um = anchor
            return anchor_um[:2] + np.linalg.solve(matrix, stage_xy_px - tip_px)
        unit = self.primary
        r0 = np.asarray(unit.r0, dtype=float).reshape(-1)[:2]
        return np.linalg.solve(matrix, stage_xy_px - r0)

    def to_primary_frame(self, key, position_um):
        """XY of pipette ``key`` at ``position_um`` in pipette_0's manipulator um."""
        if key == self.primary_key:
            return np.asarray(position_um, dtype=float).reshape(-1)[:2]
        return self.primary_xy_from_stage_frame(self.stage_frame_xy(key, position_um))

    def cell_in_primary_frame(self, cell):
        """A queued cell's XY in pipette_0's manipulator um.

        Cells are stored as ``stage_reference_px - image_px`` (XY) plus z: at
        stage reference S the cell is at image pixel ``S - cell``, so its
        stage-frame position is ``-cell``.
        """
        cell = np.asarray(cell, dtype=float).reshape(-1)
        if cell.size < 2 or not np.isfinite(cell[:2]).all():
            raise CollisionError(f"Cell position {cell} needs finite x and y")
        return self.primary_xy_from_stage_frame(-cell[:2])

    def primary_x(self):
        return float(np.asarray(self.primary.position(), dtype=float).reshape(-1)[0])

    # ---------------- checks ----------------
    def _require_calibrated(self, key):
        for name in (self.primary_key, key):
            if not _usable(self.units[name]):
                raise CollisionError(
                    f"Collision check for {key} needs {name} calibrated; "
                    f"calibrate (or recalibrate) {name} first")

    def check_cell(self, key, cell):
        """Raise CollisionError if pipette ``key`` may not patch ``cell``.

        Fail-safe: a secondary pipette needs both itself and pipette_0
        calibrated, otherwise the cell is refused.
        """
        if key == self.primary_key:
            return
        self._require_calibrated(key)
        cell_x = float(self.cell_in_primary_frame(cell)[0])
        primary_x = self.primary_x()
        if cell_x < primary_x:
            raise CollisionError(
                f"{key} may not patch this cell: it is on {self.primary_key}'s side "
                f"(cell x = {cell_x:.1f} um < {self.primary_key} x = {primary_x:.1f} um, "
                f"in {self.primary_key}'s coordinates)")

    def check_move(self, key, current_um, target_um):
        """Raise CollisionError if moving pipette ``key`` to ``target_um`` enters pipette_0's side.

        A move that doesn't go further into the forbidden side (e.g. a
        retreat after pipette_0 moved) is always allowed. Without both
        calibrations the move is allowed, so the secondary can still be
        jogged and calibrated.
        """
        if key == self.primary_key:
            return
        if not (_usable(self.primary) and _usable(self.units[key])):
            if key not in self._warned_uncalibrated:
                self._warned_uncalibrated.add(key)
                LOGGER.warning("Collision guard inactive for %s until %s and %s are calibrated",
                               key, self.primary_key, key)
            return
        self._warned_uncalibrated.discard(key)
        target_x = float(self.to_primary_frame(key, target_um)[0])
        current_x = float(self.to_primary_frame(key, current_um)[0])
        primary_x = self.primary_x()
        if target_x < primary_x and target_x < current_x:
            raise CollisionError(
                f"Refusing to move {key}: target x = {target_x:.1f} um is on "
                f"{self.primary_key}'s side (x = {primary_x:.1f} um, "
                f"in {self.primary_key}'s coordinates)")
