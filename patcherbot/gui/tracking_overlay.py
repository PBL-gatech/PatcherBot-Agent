"""Frame-bound tracking overlays; geometry is usable without importing Qt."""
import math
import time
import numpy as np


def _finite_number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def build_tracking_overlay(observation, frame_id, acquired_at, image_shape, *, now=None,
                           show_raw=False, show_prediction=False, max_age=1.0):
    now = time.monotonic() if now is None else now
    output = {"markers": [], "labels": []}
    if not isinstance(observation, dict):
        return output
    height, width = image_shape[:2]
    if width <= 0 or height <= 0:
        return output
    evidence = observation.get("deep_learning") or {}
    for name in ("pipette", "cell"):
        track = observation.get(name + "_tracking") or {}
        if not track.get("valid"):
            output["labels"].append(name + ": tracking unavailable")
            raw = evidence.get("pipette_position") if name == "pipette" else (
                observation.get("target_cell_image_xy") if observation.get("target_cell_valid") else None)
            try:
                raw_match = (raw is not None and not evidence.get("stale", True)
                             and evidence.get("source_frame") == frame_id
                             and acquired_at is not None and evidence.get("frame_acquired_at") == acquired_at
                             and 0 <= now - float(acquired_at) <= max_age
                             and (evidence.get("status") or {}).get(name + "_detector") not in
                                 ("error", "no_detection", "disabled", "no_frame"))
            except (TypeError, ValueError):
                raw_match = False
            if not raw_match:
                continue
            track = dict(valid=True, display_valid=True, display_acquired_at=acquired_at,
                         source_frame=frame_id, image_shape=evidence.get("source_image_shape", ()),
                         image_xy=raw, status="raw")
        stamp = track.get("display_acquired_at")
        try:
            age = now - float(stamp)
            matched = (track.get("display_valid") and acquired_at is not None
                       and frame_id == track.get("source_frame")
                       and float(acquired_at) == float(stamp)
                       and tuple(track["image_shape"][:2]) == tuple(image_shape[:2])
                       and 0 <= age <= max_age)
        except (KeyError, TypeError, ValueError, OverflowError):
            matched, age = False, float("inf")
        if not matched:
            output["labels"].append(name + ": tracking stale / frame unavailable")
            continue
        status = str(track.get("status") or "prediction_only")
        color = ("#ff7080" if status == "raw" else "#ffba45" if status == "prediction_only"
                 else "#44e08a" if name == "pipette" else "#64caff")
        sigma = _finite_number(track.get("uncertainty_um"))
        label = f"{name}: {status.replace('_', ' ')} | image age {age:.2f}s"
        correction_age = _finite_number(track.get("last_correction_age_s"))
        if correction_age is not None:
            label += f" | visual age {correction_age:.2f}s"
        elif status == "prediction_only":
            label += " | uncorrected"
        if sigma is not None:
            label += f" | sigma {sigma:.1f} um"
        if track.get("uncertainty_provisional", False):
            label += " (provisional)"
        depth = _finite_number(track.get("estimated_defocus_um"))
        if depth is not None:
            label += f" | defocus {depth:+.1f} um"
        output["labels"].append(label)
        candidates = [(track.get("image_xy"), status, color, track.get("covariance_pixels"))]
        if show_prediction and track.get("source_predicted_xy") is not None:
            candidates.append((track["source_predicted_xy"], "calibration", "#c893ff", None))
        if (show_raw and status != "raw" and not evidence.get("stale", True)
                and evidence.get("source_frame") == frame_id
                and evidence.get("frame_acquired_at") == acquired_at
                and (evidence.get("status") or {}).get(name + "_detector") not in
                    ("error", "no_detection", "disabled", "no_frame")):
            point = evidence.get("pipette_position") if name == "pipette" else track.get("measurement_xy")
            if name == "cell" and point is None and observation.get("target_cell_valid"):
                point = observation.get("target_cell_image_xy")
            if point is not None:
                candidates.append((point, "raw", "#ff7080", None))
        for point, style, color, covariance in candidates:
            try:
                point = np.asarray(point, dtype=float).reshape(2)
                if not np.isfinite(point).all():
                    continue
                outside = not (0 <= point[0] < width and 0 <= point[1] < height)
                if outside and style == "raw":
                    continue
                visible = point.copy()
                arrow = None
                if outside:
                    center = np.array([width / 2., height / 2.])
                    direction = point - center
                    distance = np.linalg.norm(direction)
                    unit = direction / distance
                    fraction = min(max(0., center[i] - 12.) / abs(direction[i])
                                   for i in range(2) if direction[i] != 0)
                    visible = center + fraction * direction
                    side = np.array([-unit[1], unit[0]])
                    arrow = [visible, visible - 12 * unit + 6 * side, visible - 12 * unit - 6 * side]
                ellipse = None
                if covariance is not None and not outside:
                    try:
                        covariance = np.asarray(covariance, dtype=float).reshape(2, 2)
                        if np.isfinite(covariance).all():
                            values, vectors = np.linalg.eigh(covariance)
                            if min(values) >= 0:
                                ellipse = (np.sqrt(values), math.degrees(math.atan2(vectors[1, 0], vectors[0, 0])))
                    except (TypeError, ValueError, np.linalg.LinAlgError):
                        pass
                output["markers"].append(dict(name=name, xy=point.copy(), display_xy=visible,
                                               style=style, color=color, arrow=arrow, ellipse=ellipse))
            except (TypeError, ValueError, IndexError, np.linalg.LinAlgError):
                continue
    return output


def paint_tracking_overlay(pixmap, overlay, image_shape):
    from PyQt5 import QtCore, QtGui
    painter = QtGui.QPainter(pixmap)
    try:
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.save()
        painter.scale(pixmap.width() / image_shape[1], pixmap.height() / image_shape[0])
        for marker in overlay["markers"]:
            pen = QtGui.QPen(QtGui.QColor(marker["color"]), 2)
            pen.setCosmetic(True)
            pen.setStyle(QtCore.Qt.DashLine if marker["style"] == "prediction_only" else QtCore.Qt.SolidLine)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            point = marker["display_xy"]
            if marker["arrow"] is not None:
                painter.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(*p) for p in marker["arrow"]]))
            elif marker["style"] in ("raw", "calibration"):
                painter.drawLine(QtCore.QPointF(point[0]-5, point[1]), QtCore.QPointF(point[0]+5, point[1]))
                painter.drawLine(QtCore.QPointF(point[0], point[1]-5), QtCore.QPointF(point[0], point[1]+5))
            else:
                painter.drawEllipse(QtCore.QPointF(*point), 6, 6)
            if marker["ellipse"] is not None:
                radii, angle = marker["ellipse"]
                painter.save()
                painter.translate(*point)
                painter.rotate(angle)
                painter.drawEllipse(QtCore.QPointF(0, 0), float(radii[0]), float(radii[1]))
                painter.restore()
        painter.restore()
        painter.setPen(QtGui.QColor("#eeeeee"))
        for index, label in enumerate(overlay["labels"]):
            painter.drawText(10, 20 + index * 20, label)
    finally:
        painter.end()
