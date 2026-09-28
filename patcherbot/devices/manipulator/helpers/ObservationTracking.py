"""Persistent affine observation fusion; no hardware commands or phase lifecycle."""

import collections
import math
import numpy as np


class ObservationTracking:
    pipette_tracking_settings = dict(pose_max_age=1.0, pose_skew=0.1,
        stationary_um=0.5, initial_variance=16.0, drift_variance_per_s=0.04,
        drift_variance_per_um=0.0025, measurement_sigma=1.0,
        unknown_confidence_sigma=8.0, min_confidence=0.25, gate_sigma=3.0,
        max_residual_um=30.0, max_tracking_uncertainty_um=8.0,
        max_alignment_error_um=10.0, focus_near_um=3.0, focus_far_um=5.0,
        unknown_focus_sigma_scale=2.0, defocused_sigma_scale=4.0)

    def __init__(self, controller, clock):
        self.controller = controller
        self.clock = clock
        self.pipette_tracking_settings = dict(type(self).pipette_tracking_settings)
        self._pipette_track = None
        self._cell_track = None
        self.last_cell_location = None
        self._invalidated = {}

    def enrich(self, observation, target_cell=None):
        if not isinstance(observation, dict):
            return observation
        evidence = observation.get("deep_learning") or {}
        positions = evidence.get("source_positions") or {}
        reference = np.asarray(target_cell[0], dtype=float) if target_cell is not None else np.asarray([])
        observation["target_cell_reference_coordinates"] = reference.copy()
        self._update_tracking(observation)
        candidates = []
        expected = np.asarray([np.nan, np.nan])
        source_z = float("nan")
        context_valid = False
        try:
            source_xy = np.asarray(positions["stage"], dtype=float).reshape(-1)[:2]
            source_z = float(positions["microscope"])
            frame_at = float(evidence["frame_available_at"])
            position_at = float(evidence["positions_sampled_at"])
            context_valid = (source_xy.size == 2 and np.isfinite(source_xy).all()
                             and all(math.isfinite(v) for v in (source_z, frame_at, position_at))
                             and position_at >= frame_at
                             and reference.size >= 2 and np.isfinite(reference[:2]).all()
                             and evidence.get("source_frame") is not None
                             and not evidence.get("stale", True))
        except (KeyError, TypeError, ValueError):
            pass
        detections = evidence.get("cell_detections")
        tracked = observation["cell_tracking"]
        association_available = (context_valid and detections is not None and
                                 (not tracked["valid"] or tracked["frame_context_valid"]))
        ambiguous = False
        if association_available:
            stage = self.controller.calibrated_stage
            limit_radius = getattr(self.controller.config, "hunt_limit_target_radius", True)
            radius_um = float(getattr(self.controller.config, "hunt_target_radius", 50.0)) if limit_radius else float("inf")
            if limit_radius and (not math.isfinite(radius_um) or radius_um <= 0):
                raise ValueError("Target matching radius must be finite and positive")
            expected = (np.asarray(stage.M) @ source_xy + np.asarray(stage.r0))[:2] - reference[:2]
            if tracked["valid"]:
                # The filter selects once. Keep the selected RAW point for search,
                # centering and contact; prediction alone never confirms visibility.
                expected = tracked["source_expected_xy"]
                point = tracked.get("measurement_xy")
                if point is not None:
                    distance = float(np.linalg.norm(tracked["detector_residual_um"]))
                    if distance <= radius_um:
                        candidates.append((distance, *point))
                ambiguous = tracked.get("last_measurement_reason") == "ambiguous"
            else:
                # Legacy observations can lack the poses needed for calibrated fusion.
                known = self.last_cell_location
                if known is not None and known["target_cell_id"] == tuple(float(v) for v in reference):
                    expected = np.asarray(known["image_xy"], dtype=float) + (
                        np.asarray(stage.M) @ (source_xy - known["stage_xy"]))[:2]
                for detection in detections:
                    try:
                        x, y, confidence = (float(v) for v in detection[:3])
                        height, width = evidence["source_image_shape"][:2]
                        if not (0 <= x < width and 0 <= y < height):
                            continue
                        delta = stage.pixels_to_um_relative([x - expected[0], y - expected[1], 0.0])
                        distance = math.hypot(float(delta[0]), float(delta[1]))
                    except (KeyError, TypeError, ValueError, IndexError):
                        continue
                    if all(math.isfinite(v) for v in (x, y, confidence, distance)) and confidence > 0 and distance <= radius_um:
                        candidates.append((distance, x, y))
                candidates.sort()
                ambiguous = len(candidates) > 1 and candidates[1][0] - candidates[0][0] <= 1.0
        valid = association_available and bool(candidates) and not ambiguous
        observation.update(
            visual_context_valid=context_valid,
            target_cell_id=tuple(float(v) for v in reference),
            target_cell_reference_coordinates=reference.copy(),
            target_cell_image_xy=np.asarray(candidates[0][1:] if valid else [np.nan, np.nan]),
            target_cell_expected_xy=expected.copy(),
            target_cell_match_distance_um=candidates[0][0] if candidates else float("nan"),
            target_cell_observed_z_um=source_z if valid else float("nan"),
            target_cell_candidate_count=tracked.get("candidate_count", 0) if tracked["valid"] else len(candidates),
            target_cell_valid=valid,
            target_cell_status=("observed" if valid else "ambiguous" if ambiguous
                                else "not_detected" if association_available else "unavailable"))
        if valid and (not observation["cell_tracking"]["valid"] or
                      observation["cell_tracking"]["measurement_accepted"]):
            self.last_cell_location = dict(
                target_cell_id=observation["target_cell_id"],
                image_xy=observation["target_cell_image_xy"].copy(),
                stage_xy=source_xy.copy(), source_frame=evidence.get("source_frame"),
                frame_acquired_at=evidence.get("frame_acquired_at"))
        observation.setdefault("field_metadata", {})["target_cell_image_xy"] = {
            "valid": valid, "acquired_at": evidence.get("frame_acquired_at"),
            "available_at": evidence.get("frame_available_at"),
            "source_frame": evidence.get("source_frame"), "source": "selected_cell_association",
            "coordinate_system": "image_pixels", "position_timestamp_basis": evidence.get("position_timestamp_basis")}
        observation["field_metadata"]["target_cell_observed_z_um"] = {
            "valid": valid, "acquired_at": evidence.get("positions_sampled_at"),
            "source_frame": evidence.get("source_frame"), "source": "microscope_context",
            "coordinate_system": "microscope_um", "synchronized_to_frame": False}
        return observation

    def _calibration_identity(self, name):
        """Read calibration attributes only; never sample hardware from a GUI call."""
        stage, unit = self.controller.calibrated_stage, self.controller.calibrated_unit
        camera = stage.camera
        devices = (stage,) if name == "cell" else (stage, unit)
        arrays = [stage.M, stage.r0]
        if name == "pipette":
            arrays += [unit.M, unit.r0, self.controller.home_position, self.controller.home_stage_position]
        return (tuple((id(device), getattr(device, "calibrated", False),
                       getattr(device, "must_be_recalibrated", False)) for device in devices),
                id(camera), getattr(camera, "width", None), getattr(camera, "height", None),
                getattr(camera, "flipped", False),
                tuple(tuple(np.asarray(value, dtype=float).reshape(-1)) for value in arrays))

    def invalidate(self, reason, *, pipette=True, cell=False):
        """Hold invalidated anchors unavailable until their calibration identity changes."""
        for name, selected in (("pipette", pipette), ("cell", cell)):
            if selected:
                try:
                    identity = self._calibration_identity(name)
                except (AttributeError, TypeError, ValueError):
                    identity = None
                self._invalidated[name] = (identity, str(reason))
                setattr(self, "_" + name + "_track", None)
        if cell:
            self.last_cell_location = None

    def for_frame(self, frame_id, acquired_at, image_shape, camera=None):
        """Return detached display estimates only for a sampled stationary bracket."""
        settings, now = self.pipette_tracking_settings, self.clock()
        actual_camera = self.controller.calibrated_stage.camera
        try:
            acquired = float(acquired_at)
            shape = tuple(int(value) for value in image_shape)
            if (frame_id is None or not math.isfinite(acquired) or
                    not 0 <= now - acquired <= settings["pose_max_age"] or
                    len(shape) < 2 or min(shape[:2]) <= 0 or
                    shape[:2] != (actual_camera.height, actual_camera.width) or
                    (camera is not None and camera is not actual_camera)):
                return None
        except (TypeError, ValueError):
            return None
        observation = dict(observed_at=now, field_metadata={"camera_image": dict(
            source_frame=frame_id, acquired_at=acquired, valid=True, stale=False)})
        any_valid = False
        for name in ("cell", "pipette"):
            result = dict(valid=False, display_valid=False, reason="unavailable_frame_pose")
            observation[name + "_tracking"] = result
            track = getattr(self, "_" + name + "_track")
            if track is None or track["identity"][1] != id(actual_camera):
                continue
            try:
                if track["calibration_identity"] != self._calibration_identity(name):
                    continue
            except (AttributeError, TypeError, ValueError):
                continue
            # A later visual correction must not be painted onto an earlier frame.
            if track["corrected_at"] is not None and track["corrected_at"] > acquired:
                continue
            following = next(((stamp, pose) for stamp, pose in track["samples"]
                              if acquired <= stamp <= acquired + settings["pose_skew"]), None)
            if following is None or not self._frame_pose_valid(
                    track, following[1], [following[0]], acquired, now):
                continue
            predicted = track["pixel_origin"] + track["matrix"] @ (following[1] - track["origin"])
            fused = predicted + track["stage_matrix"] @ track["correction"]
            variance = track["variance"] + settings["drift_variance_per_s"] * max(0., now - track["at"])
            age = None if track["corrected_at"] is None else now - track["corrected_at"]
            status = "fused" if (track["accepted_frame"] is not None and
                                 track["accepted_frame"][2] == frame_id and
                                 track["accepted_frame"][3] == acquired) else "prediction_only"
            result.update(valid=True, display_valid=True, display_pose_current=True, reason=None,
                          status=status, predicted_xy=predicted.copy(), fused_xy=fused.copy(),
                          image_xy=fused.copy(), source_predicted_xy=predicted.copy(),
                          display_frame=frame_id, source_frame=frame_id,
                          display_acquired_at=acquired, image_shape=shape,
                          uncertainty_um=math.sqrt(variance), uncertainty_provisional=True,
                          covariance_pixels=variance * track["stage_matrix"] @ track["stage_matrix"].T,
                          last_correction_age_s=age, estimated_defocus_um=None,
                          anchor_identity=track["identity"])
            any_valid = True
        return observation if any_valid else None

    def _correct_tracking(self, track, residual, confidence, sigma_scale=1.0):
        """Apply the same confidence and innovation rule to either XY track."""
        settings = self.pipette_tracking_settings
        confidence = None if confidence is None else float(confidence)
        if confidence is not None and (not math.isfinite(confidence) or not settings["min_confidence"] <= confidence <= 1):
            raise ValueError("low_confidence")
        sigma = (settings["unknown_confidence_sigma"] if confidence is None
                 else settings["measurement_sigma"] / confidence)
        noise = (sigma * sigma_scale) ** 2
        gate = min(settings["max_residual_um"], settings["gate_sigma"] * math.sqrt(track["variance"] + noise))
        if not np.isfinite(residual).all() or np.linalg.norm(residual) > gate:
            raise ValueError("inconsistent_detection")
        gain = track["variance"] / (track["variance"] + noise)
        track["correction"] += gain * residual
        track["variance"] *= 1 - gain

    def _fresh_visual(self, observation, previous_frame=None, after=None):
        """Check visual freshness; pose synchronization is a separate tracking test."""
        evidence = observation.get("deep_learning") or {}
        frame, acquired, sampled = (evidence.get(key) for key in
                                     ("source_frame", "frame_acquired_at", "positions_sampled_at"))
        return bool(frame is not None and frame != previous_frame and not evidence.get("stale", True)
                    and isinstance(acquired, (int, float)) and math.isfinite(acquired)
                    and isinstance(sampled, (int, float)) and math.isfinite(sampled)
                    and sampled >= acquired and observation.get("visual_context_valid", False)
                    and (after is None or acquired >= after))

    def _usable_tip(self, observation):
        """Require a fresh, successful, in-image raw tip for visual confirmation."""
        evidence = observation.get("deep_learning") or {}
        try:
            tip = np.asarray(evidence["pipette_position"], dtype=float).reshape(-1)
            shape = evidence["source_image_shape"]
            return bool(observation.get("visual_context_valid", False)
                        and evidence.get("source_frame") is not None and not evidence.get("stale", True)
                        and (evidence.get("status") or {}).get("pipette_detector") not in
                        ("error", "no_detection", "disabled", "no_frame")
                        and tip.size >= 2 and np.isfinite(tip[:2]).all()
                        and 0 <= tip[0] < shape[1] and 0 <= tip[1] < shape[0])
        except (KeyError, TypeError, ValueError, IndexError):
            return False

    def _update_tracking(self, observation):
        """Project both anchors through their affine maps, then fuse visual residuals.

        Cell: stage.M @ stage_xy + stage.r0 - stored_cell_reference.
        Tip: home_pixel + unit.M[:2] @ (pipette - home)
             + stage.M @ (stage_xy - home_stage_xy).
        Corrections live in stage microns; neither calibration nor anchors change.
        """
        stage, unit = self.controller.calibrated_stage, self.controller.calibrated_unit
        camera = stage.camera
        for name in ("cell", "pipette"):
            result = dict(valid=False, status="unavailable", reason="missing_anchor_or_calibration",
                          predicted_xy=None, fused_xy=None, uncertainty_um=None,
                          uncertainty_provisional=True, estimated_defocus_um=None,
                          cell_plane_separation_um=None, measurement_accepted=False,
                          measurement_reason="unavailable", detector_residual_um=None,
                          anchor_identity=None, display_valid=False, image_xy=None,
                          motion_xy_um=None, cell_xy=None, tip_minus_cell_um=None,
                          relative_uncertainty_um=None, source_frame=None, last_correction_age_s=None)
            observation[name + "_tracking"] = result
            metadata = dict(valid=False, source="affine_tracking", coordinate_system="image_pixels")
            observation.setdefault("field_metadata", {})[name + "_tracking"] = metadata
            try:
                stage_matrix = np.asarray(stage.M, dtype=float).reshape(2, 2)
                stage_offset = np.asarray(stage.r0, dtype=float).reshape(2)
                inverse = np.linalg.inv(stage_matrix)
                devices = (stage,) if name == "cell" else (stage, unit)
                if any(not getattr(device, "calibrated", False) or
                       getattr(device, "must_be_recalibrated", False) for device in devices):
                    raise ValueError("invalid_calibration")
                fields = (("stage_positions", "stage", 2),)
                if name == "cell":
                    reference = np.asarray(observation["target_cell_reference_coordinates"], dtype=float).reshape(-1)
                    if reference.size < 2:
                        raise ValueError("invalid_cell_reference")
                    origin, matrix = np.zeros(2), stage_matrix
                    pixel_origin = stage_offset - reference[:2]
                    anchor = reference
                else:
                    home = np.asarray(self.controller.home_position, dtype=float).reshape(3)
                    stage_home = np.asarray(self.controller.home_stage_position, dtype=float).reshape(3)
                    pipette_matrix = np.asarray(unit.M, dtype=float).reshape(3, 3)
                    offset = np.asarray(unit.r0, dtype=float).reshape(3)
                    origin = np.r_[home, stage_home[:2]]
                    matrix = np.column_stack((pipette_matrix[:2], stage_matrix))
                    pixel_origin = (pipette_matrix @ home + offset)[:2] + stage_matrix @ stage_home[:2] + stage_offset
                    anchor = np.r_[home, stage_home, pipette_matrix.ravel(), offset]
                    fields = (("manipulator_position", "manipulator", 3),) + fields
                geometry = np.r_[origin, pixel_origin, matrix.ravel(), anchor, stage_offset]
                if pixel_origin.shape != (2,) or not np.isfinite(geometry).all():
                    raise ValueError("invalid_anchor")
                identity = (tuple(id(device) for device in devices), id(camera),
                            getattr(camera, "width", None), getattr(camera, "height", None),
                            getattr(camera, "flipped", False), tuple(geometry))
            except (AttributeError, KeyError, TypeError, ValueError, np.linalg.LinAlgError):
                setattr(self, "_" + name + "_track", None)
                continue
            calibration_identity = self._calibration_identity(name)
            invalidated = self._invalidated.get(name)
            if invalidated is not None:
                if invalidated[0] == calibration_identity:
                    result["reason"] = "anchor_invalidated: " + invalidated[1]
                    continue
                self._invalidated.pop(name, None)
            self._track_position(observation, name, result, identity, fields,
                                 origin, pixel_origin, matrix, stage_matrix, inverse)
            metadata.update(valid=result["valid"], acquired_at=result.get("acquired_at"),
                            stale=not result["valid"], synchronized_to_frame=False)
        cell, tip = observation["cell_tracking"], observation["pipette_tracking"]
        if cell["valid"] and tip["valid"]:
            tip.update(cell_xy=cell["fused_xy"],
                       tip_minus_cell_um=inverse @ (tip["fused_xy"] - cell["fused_xy"]),
                       relative_uncertainty_um=math.hypot(tip["uncertainty_um"], cell["uncertainty_um"]),
                       image_cell_xy=cell["image_xy"] if cell["display_valid"] else None)

    def _frame_pose_valid(self, track, pose, stamps, acquired, now):
        """Require a stationary pose bracket around acquisition, not around inference completion."""
        settings = self.pipette_tracking_settings
        if (not np.isfinite(np.r_[pose, stamps, acquired]).all()
                or not 0 <= now - acquired <= settings["pose_max_age"]
                or any(not acquired <= stamp <= min(now, acquired + settings["pose_skew"]) for stamp in stamps)):
            return False
        preceding = next(((stamp, sample) for stamp, sample in reversed(track["samples"])
                          if stamp <= acquired), None)
        if preceding is None or acquired - preceding[0] > settings["pose_max_age"]:
            return False
        bracket = [preceding[1], *(sample for stamp, sample in track["samples"]
                                  if acquired < stamp <= max(stamps))]
        return all(np.max(np.abs(pose - sample)) <= settings["stationary_um"] for sample in bracket)

    def _track_position(self, observation, name, result, identity, fields,
                        origin, pixel_origin, matrix, stage_matrix, inverse):
        """One prediction/correction rule for the stage-linked cell and home-linked tip."""
        now, settings = self.clock(), self.pipette_tracking_settings
        metadata, evidence = observation["field_metadata"], observation.get("deep_learning") or {}
        result["anchor_identity"] = identity
        try:
            pose = np.concatenate([np.asarray(observation[field], dtype=float).reshape(-1)[:size]
                                   for field, _, size in fields])
            stamps = [float(metadata[field]["acquired_at"]) for field, _, _ in fields]
            if (pose.shape != origin.shape or not np.isfinite(np.r_[pose, stamps]).all()
                    or any(not 0 <= now - stamp <= settings["pose_max_age"] for stamp in stamps)
                    or max(stamps) - min(stamps) > settings["pose_skew"]
                    or any(metadata[field].get("valid") is False for field, _, _ in fields)):
                raise ValueError("unreliable_pose")
        except (KeyError, TypeError, ValueError):
            result["reason"] = "missing_or_stale_pose"
            return
        track = getattr(self, "_" + name + "_track")
        if track is None or track["identity"] != identity:
            track = dict(identity=identity, correction=np.zeros(2), variance=settings["initial_variance"],
                         pose=pose.copy(), at=now, frame=None, corrected_at=None, previous_xy=None,
                         frames=collections.deque(maxlen=64), frame_at=-float("inf"),
                         accepted_frame=None, accepted_xy=None, accepted_residual=None,
                         measurement_reason="prediction_only", candidate_count=0, samples=collections.deque(maxlen=64))
            setattr(self, "_" + name + "_track", track)
        track.update(calibration_identity=self._calibration_identity(name),
                     origin=origin.copy(), pixel_origin=pixel_origin.copy(),
                     matrix=matrix.copy(), stage_matrix=stage_matrix.copy())
        track["variance"] += (settings["drift_variance_per_s"] * max(0.0, now - track["at"])
                              + settings["drift_variance_per_um"] * np.linalg.norm(pose - track["pose"]))
        track["samples"].append((max(stamps), pose.copy()))
        predicted = pixel_origin + matrix @ (pose - origin)
        frame = (evidence.get("producer_session"), evidence.get("producer_generation"),
                 evidence.get("source_frame"), evidence.get("frame_acquired_at"))
        source_prediction, context_valid = None, False
        try:
            source_pose = np.concatenate([np.asarray(evidence["source_positions"][source], dtype=float).reshape(-1)[:size]
                                          for _, source, size in fields])
            source_meta = evidence["source_position_metadata"]
            source_stamps = [float(source_meta[source]["acquired_at"]) for _, source, _ in fields]
            acquired = float(evidence["frame_acquired_at"])
            context_valid = (source_pose.shape == origin.shape and not evidence.get("stale", True)
                             and all(source_meta[source].get("valid") is not False for _, source, _ in fields)
                             and self._frame_pose_valid(track, source_pose, source_stamps, acquired, now))
            if context_valid:
                source_prediction = pixel_origin + matrix @ (source_pose - origin)
                result["source_expected_xy"] = source_prediction + stage_matrix @ track["correction"]
        except (KeyError, TypeError, ValueError):
            pass
        result.update(frame_context_valid=bool(context_valid), measurement_reason="prediction_only")
        if evidence.get("source_frame") is not None:
            if frame in track["frames"]:
                result["measurement_reason"] = "duplicate"
            else:
                track["frames"].append(frame)
                track["candidate_count"] = 0
                try:
                    if not context_valid:
                        raise ValueError("unreliable_frame_pose")
                    if acquired < track["frame_at"]:
                        raise ValueError("out_of_order")
                    track["frame_at"] = acquired
                    height, width = evidence["source_image_shape"][:2]
                    confidence = (evidence.get("confidence") or {}).get("pipette_detector")
                    detections = evidence.get("cell_detections") if name == "cell" else None
                    if name == "pipette" and evidence.get("pipette_position") is not None:
                        detections = [(*evidence["pipette_position"], confidence)]
                    candidates = []
                    radius = (float(getattr(self.controller.config, "hunt_target_radius", 50.0))
                              if name == "cell" and getattr(self.controller.config, "hunt_limit_target_radius", True)
                              else float("inf"))
                    if name == "cell" and getattr(self.controller.config, "hunt_limit_target_radius", True):
                        if not math.isfinite(radius) or radius <= 0:
                            raise ValueError("invalid_target_radius")
                    for detection in (() if detections is None else detections):
                        try:
                            point = np.asarray(detection[:2], dtype=float).reshape(2)
                            confidence = None if len(detection) < 3 or detection[2] is None else float(detection[2])
                            if (not np.isfinite(point).all() or not (0 <= point[0] < width and 0 <= point[1] < height)
                                    or (name == "cell" and confidence is not None and
                                        not settings["min_confidence"] <= confidence <= 1)):
                                continue
                            residual = inverse @ (point - source_prediction) - track["correction"]
                            distance = float(np.linalg.norm(residual))
                            if distance <= radius:
                                candidates.append((distance, point, confidence, residual))
                        except (TypeError, ValueError, IndexError):
                            continue
                    candidates.sort(key=lambda item: item[0])
                    track["candidate_count"] = len(candidates)
                    if (evidence.get("status") or {}).get(name + "_detector") in ("error", "no_detection", "disabled", "no_frame") or not candidates:
                        raise ValueError("no_usable_detection")
                    if len(candidates) > 1 and candidates[1][0] - candidates[0][0] <= 1.0:
                        raise ValueError("ambiguous")
                    _, point, confidence, residual = candidates[0]
                    result.update(detector_residual_um=residual.copy(), measurement_confidence=confidence)
                    sigma_scale = 1.0
                    if name == "pipette":
                        # This is a measured focus hint, not calibrated depth. Only
                        # the focus head's own confidence may increase its influence.
                        sigma_scale = settings["unknown_focus_sigma_scale"]
                        try:
                            focus = float(evidence["pipette_focus"])
                            focus_confidence = float((evidence.get("confidence") or {})["pipette_focuser"])
                            if (math.isfinite(focus) and settings["min_confidence"] <= focus_confidence <= 1
                                    and (evidence.get("status") or {}).get("pipette_focuser") not in
                                    ("error", "no_detection", "disabled", "no_frame")):
                                fraction = max(0.0, (abs(focus) - settings["focus_near_um"]) /
                                               (settings["focus_far_um"] - settings["focus_near_um"]))
                                sigma_scale = 1.0 + (settings["defocused_sigma_scale"] - 1.0) * min(fraction, 2.0)
                                result.update(measured_defocus_um=focus, focus_confidence=focus_confidence)
                        except (KeyError, TypeError, ValueError):
                            pass
                    result["measurement_sigma_scale"] = sigma_scale
                    self._correct_tracking(track, residual, confidence, sigma_scale)
                    track.update(corrected_at=acquired, accepted_frame=frame,
                                 accepted_xy=point.copy(), accepted_residual=residual.copy())
                    result.update(measurement_accepted=True, measurement_reason="accepted", measurement_xy=point.copy())
                except (KeyError, TypeError, ValueError, IndexError) as error:
                    result["measurement_reason"] = str(error) if isinstance(error, ValueError) else "no_detection"
                track["measurement_reason"] = result["measurement_reason"]
        result.update(last_measurement_reason=track["measurement_reason"], candidate_count=track["candidate_count"])
        if context_valid and frame == track["accepted_frame"]:
            result.update(measurement_xy=track["accepted_xy"].copy(), detector_residual_um=track["accepted_residual"].copy())
        fused = predicted + stage_matrix @ track["correction"]
        result.update(valid=True, reason=None, predicted_xy=predicted, fused_xy=fused,
                      status="fused" if context_valid and frame == track["accepted_frame"] else "prediction_only",
                      uncertainty_um=math.sqrt(track["variance"]), covariance_pixels=track["variance"] * stage_matrix @ stage_matrix.T,
                      acquired_at=min(stamps), age_s=now - min(stamps), source_frame=evidence.get("source_frame"),
                      last_correction_age_s=None if track["corrected_at"] is None else now - track["corrected_at"],
                      motion_xy_um=None if track["previous_xy"] is None else inverse @ (fused - track["previous_xy"]))
        image_meta = metadata.get("camera_image") or {}
        try:
            image_at = float(image_meta["acquired_at"])
            if not image_meta.get("stale", True) and image_meta.get("valid", False):
                if context_valid and image_meta.get("source_frame") == evidence.get("source_frame") and image_at == acquired:
                    result.update(display_valid=True, image_xy=source_prediction + stage_matrix @ track["correction"],
                                  display_pose_current=bool(np.max(np.abs(source_pose - pose)) <= settings["stationary_um"]),
                                  display_acquired_at=image_at, image_shape=evidence["source_image_shape"])
                elif self._frame_pose_valid(track, pose, stamps, image_at, now):
                    result.update(display_valid=True, display_pose_current=True, image_xy=fused.copy(), display_acquired_at=image_at,
                                  image_shape=observation["camera_image"].shape)
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
        if result["display_valid"]:
            result["display_frame"] = image_meta.get("source_frame")
        track.update(pose=pose.copy(), at=now, frame=frame, previous_xy=fused.copy())

