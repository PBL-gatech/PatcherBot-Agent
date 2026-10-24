import time
import cv2
import numpy as np
import os
from pathlib import Path
from patcherbot.devices.manipulator.microscope import Microscope
from patcherbot.devices.manipulator import Manipulator
from patcherbot.devices.camera import Camera
from patcherbot.deepLearning.pipetteDetector import (
    PipetteDetector, PipetteDetector5, configure_pipette_detector,
)
from patcherbot.deepLearning.pipetteFocuser import PipetteFocuser2
from threading import Thread
import logging
from .pipette_flow import track_pipette_displacement


CALIBRATION_FRAME_COUNT = 5
CALIBRATION_TIMEOUT_SECONDS = 3.0

def _resolve_model_path(model_name):
    if model_name is None:
        return None
    model_text = str(model_name).strip()
    if not model_text:
        return None

    # Normalize accidental wrappers like r'...', "...", and stray quote tails.
    while model_text:
        previous = model_text
        if len(model_text) >= 2 and model_text[0].lower() == "r" and model_text[1] in ("'", '"'):
            model_text = model_text[1:].strip()
        if len(model_text) >= 2 and model_text[0] in ("'", '"') and model_text[-1] == model_text[0]:
            model_text = model_text[1:-1].strip()
        model_text = model_text.strip(" '\"")
        if model_text == previous:
            break

    if not model_text:
        return None

    path = Path(model_text).expanduser()
    if path.is_absolute():
        return path

    model_dir = Path(__file__).resolve().parents[3] / "deepLearning" / "pipetteModel"
    return model_dir / path


class PipetteCalHelper():
    """
    A helper class to aid with 2D pipette calibration.

    Overview:
      - Calibration points pair raw image coordinates with pipette encoders.
      - Only the (x, y) components are used, ignoring the z-axis.
      - Ten calibration points are gathered so that the field of view is well‐sampled.
      - A 2D affine transformation is computed from pipette encoder (x, y) positions to
        image (x, y) positions, using OpenCV.
      - The 2×3 matrix is then embedded in a 3×4 homogeneous transformation matrix.
    
    Summary:
      1. Record calibration points as tuples: (image_x, image_y, encoder_x, encoder_y).
      2. Subtract the stage's reference position from the pipette’s detected image location.
      3. Use cv2.estimateAffine2D on the collected points.
      4. Convert the resulting 2×3 matrix into a 3×4 matrix for homogeneous representation.
    """

    CAL_MAX_SPEED = 1000
    NORMAL_MAX_SPEED = 1000

    def __init__(self, pipette: Manipulator, microscope: Microscope, camera: Camera, calibrated_stage, config=None):
        self.pipette: Manipulator = pipette
        self.microscope: Microscope = microscope
        self.camera = camera
        self.config = config
        device = os.getenv("PIPETTE_DETECTOR_DEVICE", "cuda:0")
        model_name = getattr(self.config, "pipette_detector_model", None)
        model_path = _resolve_model_path(model_name)
        model_version = getattr(self.config, "pipette_detector_model_version", "R02")
        try:
            self.pipetteDetector: PipetteDetector = PipetteDetector5(
                model_path=model_path,
                device=device,
                model_version=model_version,
            )
        except Exception as exc:
            logging.warning(
                "Failed to load general pipette model '%s' (%s); using the existing default R02 checkpoint",
                model_name,
                exc,
            )
            self.pipetteDetector = PipetteDetector5(
                model_path=None, device=device, model_version="R02",
            )
        configure_pipette_detector(self.pipetteDetector)
        self.calibrated_stage = calibrated_stage
        # Each calibration point will be a tuple:
        #   (image_x, image_y, encoder_x, encoder_y)
        self.cal_points = []
        self.calibration_samples = []
        self._flow_reference = None

    def brain_slice_detector(self):
        """Temporarily load the configured specialist into the shared detector."""
        model_path = _resolve_model_path(getattr(
            self.config, "pipette_brain_slice_model", "PipetteDetectorNetB.pt",
        ))
        if model_path is None:
            raise ValueError("A brain-slice checkpoint path is required")
        return self.pipetteDetector.use_model(
            model_path,
            model_version=getattr(self.config, "pipette_brain_slice_model_version", "R12"),
        )

    def collect_cal_points(self, num_points=10, xy_step=5, max_retries=5):
        """Apply the stage-style calibration speed cap and restore the prior limit."""
        log = logging.getLogger("patcherbot.PipetteCalibration")
        previous_speed = self.pipette.get_max_speed()
        if previous_speed is None or not np.isfinite(previous_speed) or previous_speed <= 0:
            raise RuntimeError("Cannot read pipette speed; calibration was not started")
        calibration_speed = min(float(previous_speed), self.CAL_MAX_SPEED)
        try:
            self.pipette.set_max_speed(calibration_speed)
            applied_speed = self.pipette.get_max_speed()
            if (applied_speed is None or not np.isfinite(applied_speed)
                    or applied_speed <= 0 or applied_speed > calibration_speed):
                raise RuntimeError("Pipette calibration speed limit was not confirmed; sampling was not started")
            log.info("Pipette calibration speed limit: %s -> %s (controller units)",
                     previous_speed, applied_speed)
            return self._collect_cal_points_at_calibration_speed(num_points, xy_step, max_retries)
        finally:
            self.pipette.set_max_speed(previous_speed)
            restored_speed = self.pipette.get_max_speed()
            if restored_speed is None or restored_speed != previous_speed:
                raise RuntimeError("Pipette speed restoration was not confirmed")
            log.info("Pipette calibration speed limit restored to %s (controller units)", restored_speed)

    def _collect_cal_points_at_calibration_speed(self, num_points=10, xy_step=5, max_retries=5):
        """
        Collects 'num_points' calibration points.
        
        For each point:
         - The pipette’s current (x, y) image position (median of five fresh detections)
           is paired with the pipette’s encoder (x, y) coordinates.
         - A small move (with added randomness) is commanded between points so that the
           calibration data covers a larger area.
        """
        self.cal_points = []
        self.calibration_samples = []
        self._flow_reference = None
        step = min(2.0 * abs(float(xy_step)), 10.0)
        logging.getLogger("patcherbot.PipetteCalibration").info(
            "Pipette calibration sampling: X then Y, %d points, %.1f um steps, Z unchanged",
            num_points, step)
        for i in range(num_points):
            self._record_point_with_retries(max_retries)
            if i < num_points - 1:
                # Separate axis measurements; default steps stay within the old 0-10 um range.
                delta = np.zeros(3)
                delta[0 if i < (num_points - 1) // 2 else 1] = step
                self.pipette.relative_move(delta.tolist())
                self.pipette.wait_until_still()
        return len(self.cal_points) >= num_points

    def _record_point_with_retries(self, max_retries=5):
        """
        Attempt to record a calibration point. If no pipette is detected in the image,
        jitter the pipette and retry.
        """
        for _ in range(max_retries):
            before = len(self.cal_points)
            self.record_cal_point()
            if len(self.cal_points) > before:
                return True
            # Jitter the pipette if detection failed.
            self.pipette.relative_move([
                np.random.uniform(-15, 15),
                np.random.uniform(-15, 15),
                0
            ])
            self.pipette.wait_until_still()
            time.sleep(0.5)
        return False

    def record_cal_point(self):
        """Record the median of five distinct newly retrieved stationary frames."""
        log = logging.getLogger("patcherbot.PipetteCalibration")
        started = time.monotonic()
        encoder_pos = np.asarray(self.pipette.position(), dtype=float).copy()
        seen, detections, observations = set(), [], []
        sample = dict(encoder=encoder_pos.tolist(), sampling_started_at=started,
                      frames=[], status="timeout")
        self.calibration_samples.append(sample)
        while time.monotonic() - started < CALIBRATION_TIMEOUT_SECONDS:
            data = self.camera.last_raw_frame_data(include_timing=True)
            if data is None:
                time.sleep(0.01)
                continue
            frame_no, _, frame, timing = data
            retrieved = timing.get("retrieval_started_at")
            if frame_no in seen or retrieved is None or retrieved < started:
                time.sleep(0.01)
                continue
            seen.add(frame_no)
            if not np.allclose(self.pipette.position(), encoder_pos, atol=0.1, rtol=0):
                sample["status"] = "encoder_moved"
                log.warning("Pipette calibration sample rejected: encoder moved before detection")
                return
            if hasattr(self.pipetteDetector, "detect_pipette_tracking"):
                tracking = self.pipetteDetector.detect_pipette_tracking(frame.copy())
                tip = tracking.get("tip_xy")
                flow_frame, mask = tracking.get("frame"), tracking.get("mask")
            else:
                tip = self.pipetteDetector.detect_pipette(frame.copy())
                flow_frame, mask = frame.copy(), None
            if not np.allclose(self.pipette.position(), encoder_pos, atol=0.1, rtol=0):
                sample["status"] = "encoder_moved"
                log.warning("Pipette calibration sample rejected: encoder moved during detection")
                return
            if tip is None:
                continue
            tip = np.asarray(tip, dtype=float)
            if tip.shape != (2,) or not np.isfinite(tip).all():
                continue
            detections.append(tip)
            observations.append((flow_frame, mask))
            sample["frames"].append(dict(frame_no=int(frame_no),
                                         retrieval_started_at=float(retrieved), tip_xy=tip.tolist()))
            if len(detections) == CALIBRATION_FRAME_COUNT:
                median = np.median(detections, axis=0)
                representative = int(np.argmin(np.linalg.norm(np.asarray(detections) - median, axis=1)))
                position, source = self._hybrid_position(median, *observations[representative])
                self.cal_points.append(tuple(position) + tuple(encoder_pos[:2]))
                spread = float(np.max(np.linalg.norm(np.asarray(detections) - median, axis=1)))
                sample.update(status="complete", median_xy=median.tolist(), max_spread_px=spread,
                              fitted_xy=position.tolist(), source=source)
                log.info("Pipette calibration point %d: %d frames, median=(%.2f, %.2f) px, "
                         "max spread=%.2f px, encoder=(%.3f, %.3f) um",
                         len(self.cal_points), len(detections), *median, spread, *encoder_pos[:2])
                self.camera.show_circle(tuple(int(round(v)) for v in position))
                return
        log.warning("Pipette calibration sample timed out: %d/%d valid new frames; point rejected",
                    len(detections), CALIBRATION_FRAME_COUNT)

    def observe_z_calibration(self, abort_check):
        """Collect five fresh detections after the caller's move-completion wait.

        Readback variation is measurement evidence, not a movement interlock.
        Scientifica positions are cached asynchronously, so use the readbacks
        accompanying accepted frames rather than a potentially stale entry value.
        """
        abort_check()
        started = time.monotonic()
        readbacks = []

        def read_position():
            xyz = np.asarray(self.pipette.position(), dtype=float).copy()
            stage = np.asarray(self.calibrated_stage.position(), dtype=float)[:2].copy()
            scope = float(self.microscope.position())
            if (xyz.shape != (3,) or stage.shape != (2,)
                    or not np.isfinite(np.r_[xyz, stage, scope]).all()):
                raise RuntimeError("Z calibration requires finite positions")
            value = dict(pipette_xyz=xyz.tolist(), stage_xy=stage.tolist(),
                         microscope_z=scope, read_at=time.monotonic())
            readbacks.append(value)
            return value

        read_position()

        seen, points, frames = set(), [], []
        while time.monotonic() - started < CALIBRATION_TIMEOUT_SECONDS:
            abort_check()
            data = self.camera.last_raw_frame_data(include_timing=True)
            if data is None:
                time.sleep(0.01)
                continue
            frame_id, _, image, timing = data
            retrieved = timing.get("retrieval_started_at")
            if (frame_id in seen or image is None or retrieved is None
                    or not np.isfinite(retrieved) or not started <= retrieved <= time.monotonic()):
                time.sleep(0.01)
                continue
            seen.add(frame_id)
            tip = self.pipetteDetector.detect_pipette(image.copy())
            abort_check()
            position = read_position()
            if time.monotonic() - started >= CALIBRATION_TIMEOUT_SECONDS:
                break
            if tip is None:
                continue
            tip = np.asarray(tip, dtype=float)
            if tip.shape != (2,) or not np.isfinite(tip).all():
                continue
            points.append(tip)
            available = timing.get("available_at")
            frames.append(dict(frame_no=int(frame_id), retrieval_started_at=float(retrieved),
                available_at=None if available is None else float(available), tip_xy=tip.tolist(),
                **position))
            if len(points) == CALIBRATION_FRAME_COUNT:
                median = np.median(points, axis=0)
                xyz = np.median([f["pipette_xyz"] for f in frames], axis=0)
                stage = np.median([f["stage_xy"] for f in frames], axis=0)
                scope = float(np.median([f["microscope_z"] for f in frames]))
                return dict(tip_xy=median.tolist(), pipette_xyz=xyz.tolist(), stage_xy=stage.tolist(),
                    microscope_z=scope, frames=frames, sampling_started_at=started,
                    sampling_finished_at=time.monotonic(),
                    position_readbacks=readbacks,
                    readback_span=dict(
                        pipette_xyz_um=np.ptp([r["pipette_xyz"] for r in readbacks], axis=0).tolist(),
                        stage_xy_um=np.ptp([r["stage_xy"] for r in readbacks], axis=0).tolist(),
                        microscope_z_native=float(np.ptp([r["microscope_z"] for r in readbacks]))),
                    spread_px=float(np.max(np.linalg.norm(np.asarray(points) - median, axis=1))))
        raise RuntimeError("Z calibration timed out before five fresh valid detections")

    @staticmethod
    def fit_z_calibration(samples, jxy, continuous=False):
        """Fit group errors; keep centering and autofocus as separate evidence."""
        jxy = np.asarray(jxy, dtype=float)
        if (jxy.shape != (2, 2) or not np.isfinite(jxy).all()
                or np.linalg.cond(jxy) > 1e8):
            raise ValueError("A nonsingular finite XY calibration is required")
        if not samples:
            raise ValueError("No group calibration samples")
        requested_steps = [sample.get("requested_step_um") for sample in samples]
        step_values = []
        if any(step is not None for step in requested_steps):
            if any(step is None or isinstance(step, (bool, np.bool_)) for step in requested_steps):
                raise ValueError("Every sample must specify its requested group distance")
            step_values = [float(step) for step in requested_steps]
            if not np.isfinite(step_values).all() or any(step <= 0 for step in step_values):
                raise ValueError("Requested group distances must be finite and positive")
        distances = list(dict.fromkeys(step_values))
        if len(distances) > 1 and not continuous:
            if len({sample["id"] for sample in samples}) != len(samples):
                raise ValueError("Sample IDs must be unique across distance blocks")
            models = {}
            for distance in distances:
                subset = [sample for sample, step in zip(samples, step_values) if step == distance]
                models[format(distance, "g")] = PipetteCalHelper.fit_z_calibration(subset, jxy)
            accepted = all(model["status"] == "validated" for model in models.values())
            return dict(version=3, mode="group", model_kind="group_z_innovations_by_distance",
                status="validated" if accepted else "unsupported", accepted=accepted,
                jxy=jxy.tolist(), requested_steps_um=distances, models_by_distance=models,
                training_count=sum(model["training_count"] for model in models.values()),
                validation=dict(by_distance={key: model["validation"] for key, model in models.items()}),
                reason="; ".join(f"{key} um: {model['reason']}" for key, model in models.items()))
        reference = np.asarray(samples[0]["before"]["stage_xy"], dtype=float)
        positions, scopes, observations, directions, splits = [], [], [], [], []
        pair_keys, ids = [], set()

        def check(observation):
            tip = np.asarray(observation["tip_xy"], dtype=float)
            xyz = np.asarray(observation["pipette_xyz"], dtype=float)
            stage = np.asarray(observation["stage_xy"], dtype=float)
            scope = float(observation["microscope_z"])
            if (tip.shape != (2,) or xyz.shape != (3,) or stage.shape != (2,)
                    or not np.isfinite(np.r_[tip, xyz, stage, scope]).all()
                    or not np.allclose(stage, reference, atol=0.1, rtol=0)):
                raise ValueError("Invalid or moving-stage calibration observation")
            positions.append(float(xyz[2]))
            scopes.append(scope)
            return tip, xyz

        for sample in samples:
            if sample["id"] in ids or sample["direction"] not in ("up", "down"):
                raise ValueError("Duplicate sample or invalid direction")
            ids.add(sample["id"])
            before_tip, before_xyz = check(sample["before"])
            after_tip, after_xyz = check(sample["after"])
            check(sample["final"])
            delta = after_xyz - before_xyz
            sign = -1 if sample["direction"] == "up" else 1
            dz = float(sample["dz_um"])
            if not np.isfinite(dz) or dz * sign <= 0 or not np.isclose(delta[2], dz, atol=1e-6, rtol=0):
                raise ValueError("Actual Z disagrees with group direction")
            error = after_tip - before_tip - jxy @ delta[:2]
            sample["group_error_px"] = error.tolist()
            for event in sample["correction_events"]:
                old_tip, old_xyz = check(event["before"])
                new_tip, new_xyz = check(event["after"])
                if event["kind"] not in ("center", "focus"):
                    raise ValueError("Unknown fine-calibration event")
                event["achieved_xyz_um"] = (new_xyz - old_xyz).tolist()
                event["image_displacement_px"] = (new_tip - old_tip).tolist()
                if event["kind"] == "center":
                    event["centering_response_error_px"] = (new_tip - old_tip - jxy @ (new_xyz - old_xyz)[:2]).tolist()
            observations.append(error)
            directions.append(sign)
            splits.append(sample["split"])
            pair_keys.append((sample["split"], sample["pair_id"]))
        if len(samples) % 2:
            raise ValueError("Complete up/down pairs are required")
        for index in range(0, len(samples), 2):
            if (pair_keys[index] != pair_keys[index + 1]
                    or directions[index:index + 2] != [-1, 1]
                    or pair_keys[index] in pair_keys[:index]):
                raise ValueError("Calibration pairs must be complete, ordered and unique")
        if continuous:
            return PipetteCalHelper._fit_z_continuous(
                samples, observations, directions, jxy, positions, scopes)
        fitted = PipetteCalHelper._fit_z_innovations(observations, directions, splits)
        metrics, status = PipetteCalHelper._z_validation_metrics(samples, observations, fitted, jxy)
        reasons = []
        worse = [direction for direction in ("up", "down")
                 if metrics[direction]["ssm_rmse_um"] > metrics[direction]["mean_rmse_um"]]
        if worse:
            reasons.append("SSM held-out RMSE exceeds the directional-mean baseline for " + ", ".join(worse))
        interval = metrics["paired_mse_benefit_interval_um2"]
        if interval[0] <= 0:
            reasons.append(f"paired MSE-benefit interval [{interval[0]:.6g}, {interval[1]:.6g}] um^2 is not wholly positive")
        if metrics.get("cross_cycle_dependence", {}).get("significant", False):
            reasons.append("held-out innovations retain significant cross-cycle dependence")
        fitted.update(version=2, mode="group", model_kind="group_z_innovations", status=status,
            accepted=status == "validated", jxy=jxy.tolist(), validation=metrics,
            reason="; ".join(reasons) if reasons else "Held-out SSM improvement passes the paired and directional checks",
            fit_sample_ids=[s["id"] for s in samples if s["split"] == "fit"],
            validation_sample_ids=[s["id"] for s in samples if s["split"] == "validation"],
            measured_coverage=dict(steps_um=[float(s["dz_um"]) for s in samples],
                pipette_z_um=[min(positions), max(positions)],
                microscope_z_native=[min(scopes), max(scopes)]))
        if distances:
            fitted["requested_step_um"] = distances[0]
        return fitted

    @staticmethod
    def _fit_z_continuous(samples, observations, directions, jxy, positions, scopes):
        """Report-only continuous predictors; reserved targets never reach a fit or score."""
        splits = np.asarray([sample["split"] for sample in samples])
        counts = {name: int(np.sum(splits == name))
                  for name in ("fit", "validation", "reserved")}
        expected = [name for name in counts for _ in range(counts[name])]
        if (not all(counts.values()) or not np.array_equal(splits, expected)
                or any(count % 2 for count in counts.values())):
            raise ValueError("Continuous data require chronological fit, validation and reserved pairs")
        n, stop = counts["fit"], counts["fit"] + counts["validation"]
        active = samples[:stop]
        labels = np.asarray(directions[:stop])
        if any(np.sum(labels[:n] == sign) < 6 for sign in (-1, 1)):
            raise ValueError("At least six fitting moves per direction are required")
        # Slice before conversion: reserved target values do not enter any estimator.
        errors_px = np.asarray(observations[:stop], dtype=float)
        errors = np.linalg.solve(jxy, errors_px.T).T
        distance = np.abs([sample["dz_um"] for sample in active]) / 100.0
        validation_labels = labels[n:]

        def score(predicted):
            residual = errors[n:] - predicted
            result = {}
            for name, sign in (("all", None), ("up", -1), ("down", 1)):
                mask = np.ones(len(residual), dtype=bool) if sign is None else validation_labels == sign
                if not mask.any():
                    raise ValueError("Both validation directions are required")
                selected = residual[mask]
                result[name] = dict(count=int(mask.sum()),
                    rmse_xy_um=np.sqrt(np.mean(selected ** 2, axis=0)).tolist(),
                    rmse_um=float(np.sqrt(np.mean(np.sum(selected ** 2, axis=1)))),
                    bias_um=selected.mean(axis=0).tolist())
            return result, residual.tolist()

        models, predictions = {}, {}
        bases = {
            "zero": np.ones((stop, 1)),
            "directional_mean": np.ones((stop, 1)),
            "affine": np.column_stack((np.ones(stop), distance)),
            "quadratic": np.column_stack((np.ones(stop), distance, distance ** 2)),
            "cubic": np.column_stack((np.ones(stop), distance, distance ** 2, distance ** 3)),
            "hinge300": np.column_stack((np.ones(stop), distance, np.maximum(0, distance - 3))),
        }
        for name, design in bases.items():
            predicted = np.zeros_like(errors)
            coefficients = {}
            for direction, sign in (("up", -1), ("down", 1)):
                train_mask = labels[:n] == sign
                if name == "zero":
                    coef = np.zeros((1, 2))
                else:
                    coef, _, rank, _ = np.linalg.lstsq(design[:n][train_mask], errors[:n][train_mask], rcond=None)
                    if rank < design.shape[1] or not np.isfinite(coef).all():
                        raise ValueError("Insufficient training distance variation for " + name)
                coefficients[direction] = coef.tolist()
                predicted[labels == sign] = design[labels == sign] @ coef
            metrics, residuals = score(predicted[n:])
            models[name] = dict(status="fitted", coefficients_um=coefficients,
                basis=("[1,d,max(0,d-3)]" if name == "hinge300" else
                       "ascending powers of d; d=abs(actual dz_um)/100"),
                validation=metrics, validation_predictions_um=predicted[n:].tolist(),
                validation_residuals_um=residuals)
            predictions[name] = predicted

        # The existing estimator operates in image pixels. Its mean is a residual
        # offset, and its state updates only after recording each prediction.
        affine = predictions["affine"]
        residual_px = (errors - affine) @ jxy.T
        try:
            innovations = PipetteCalHelper._fit_z_innovations(
                residual_px, labels, splits[:stop])
            ssm_prediction = affine[n:] + np.linalg.solve(
                jxy, np.asarray(innovations["validation_predictions_px"]).T).T
            metrics, residuals = score(ssm_prediction)
            models["affine_ssm"] = dict(status="fitted", prediction_mode="detector-assisted one-step",
                innovations=innovations, validation=metrics,
                validation_predictions_um=ssm_prediction.tolist(),
                validation_residuals_um=residuals,
                affine_rmse_um=models["affine"]["validation"]["all"]["rmse_um"],
                improvement_um=models["affine"]["validation"]["all"]["rmse_um"] - metrics["all"]["rmse_um"])
        except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            models["affine_ssm"] = dict(status="unsupported", reason=str(error))

        for index, sample in enumerate(active):
            sample["group_error_um"] = errors[index].tolist()
            sample["predictions_um"] = {name: value[index].tolist() for name, value in predictions.items()}
            if index >= n and models["affine_ssm"]["status"] == "fitted":
                sample["predictions_um"]["affine_ssm"] = models["affine_ssm"]["validation_predictions_um"][index - n]
            sample["proposed_corrections_um"] = {
                name: (-np.asarray(prediction)).tolist()
                for name, prediction in sample["predictions_um"].items()}
        static_names = ("affine", "quadratic", "cubic", "hinge300")
        best = min(static_names, key=lambda name: models[name]["validation"]["all"]["rmse_um"])
        ssm_ok = models["affine_ssm"]["status"] == "fitted"
        return dict(version=4, mode="group", model_kind="group_z_continuous_characterization",
            report_only=True, accepted=False, status="analyzed" if ssm_ok else "unsupported",
            reason=("Report-only model comparison; reserved observations remain unevaluated" if ssm_ok else
                    "Static analysis completed; residual SSM unsupported: " + models["affine_ssm"]["reason"]),
            jxy=jxy.tolist(), counts=counts, training_count=n, validation_count=counts["validation"],
            reserved_count=counts["reserved"], models=models, best_static_model=best,
            distance_scale_um=100.0, hinge_breakpoint_um=300.0,
            fit_sample_ids=[sample["id"] for sample in samples[:n]],
            validation_sample_ids=[sample["id"] for sample in samples[n:stop]],
            reserved_sample_ids=[sample["id"] for sample in samples[stop:]],
            measured_coverage=dict(steps_um=[float(sample["dz_um"]) for sample in samples],
                pipette_z_um=[min(positions), max(positions)], microscope_z_native=[min(scopes), max(scopes)]))

    @staticmethod
    def _fit_z_innovations(errors_xy, directions, splits):
        from scipy.optimize import minimize

        errors = np.asarray(errors_xy, dtype=float)
        direction, split = np.asarray(directions), np.asarray(splits)
        if (errors.ndim != 2 or errors.shape[1] != 2
                or direction.shape != (len(errors),) or split.shape != (len(errors),)
                or not np.isfinite(errors).all()
                or not np.isin(direction, [-1, 1]).all()
                or not np.isin(split, ["fit", "validation"]).all()):
            raise ValueError("Finite XY errors, signed directions and valid splits required")
        n = int(np.sum(split == "fit"))
        if (n == 0 or n == len(errors) or not np.all(split[:n] == "fit")
                or not np.all(split[n:] == "validation")):
            raise ValueError("Training must precede nonempty held-out data")
        labels = (direction > 0).astype(int)
        train, train_labels = errors[:n], labels[:n]
        if any(np.sum(train_labels == i) < 6 for i in (0, 1)):
            raise ValueError("Too few training movements in either direction")
        if any(np.sum(labels[n:] == i) == 0 for i in (0, 1)):
            raise ValueError("Held-out data must contain both directions")
        baseline = np.asarray([train[train_labels == i].mean(axis=0) for i in (0, 1)])
        scale = max(1.0, float(np.max(np.abs(train))))
        floor = (scale * 1e-6) ** 2
        bounds = [(-10 * scale, 10 * scale)] * 4 + [
            (-0.98, 0.98), (-1.96, 1.96), (-1.96, 1.96)]

        def replay(p, values, indices, state=None):
            state = np.zeros(2) if state is None else state.copy()
            predictions, innovations = [], []
            for value, label in zip(values, indices):
                prediction = p[:4].reshape(2, 2)[label] + state
                predictions.append(prediction.copy())
                innovation = value - prediction
                innovations.append(innovation.copy())
                state = p[4] * state + p[5:7] * innovation
            return np.asarray(predictions), np.asarray(innovations), state

        def profile(p):
            _, residuals, _ = replay(p, train, train_labels)
            covariance = residuals.T @ residuals / n + np.eye(2) * floor
            sign, logdet = np.linalg.slogdet(covariance)
            if sign <= 0 or not np.isfinite(covariance).all():
                return float("inf"), covariance
            quadratic = np.sum(residuals * np.linalg.solve(covariance, residuals.T).T)
            return float(0.5 * (n * (2 * np.log(2 * np.pi) + logdet) + quadratic)), covariance

        constraint = {"type": "ineq", "fun": lambda p: 0.98 - np.abs(p[4] - p[5:7])}
        candidates = []
        for rho in (0.0, -0.5, 0.5, -0.9, 0.9):
            result = minimize(lambda p: profile(p)[0],
                              np.r_[baseline.ravel(), rho, 0.0, 0.0],
                              method="SLSQP", bounds=bounds, constraints=[constraint],
                              options={"maxiter": 400, "ftol": 1e-8})
            if (result.success and np.isfinite(result.x).all()
                    and np.isfinite(result.fun)
                    and np.min(constraint["fun"](result.x)) >= -1e-7):
                candidates.append(result)
        if not candidates:
            raise RuntimeError("No converged stable movement-error model")
        best = min(candidates, key=lambda candidate: candidate.fun)
        p = best.x
        _, _, state = replay(p, train, train_labels)
        predicted, residuals, _ = replay(p, errors[n:], labels[n:], state)
        return {
            "means_px": p[:4].reshape(2, 2).tolist(),
            "direction_order": ["up", "down"], "rho": float(p[4]),
            "gain": np.diag(p[5:7]).tolist(),
            "innovation_covariance_px2": profile(p)[1].tolist(),
            "covariance_floor_px2": floor,
            "training_end_state": state.tolist(), "training_count": n,
            "training_negative_log_likelihood": float(best.fun),
            "optimizer_message": str(best.message),
            "optimizer_converged": bool(best.success),
            "optimizer_iterations": int(best.nit),
            "boundary_parameter_indices": [i for i, (lower, upper) in enumerate(bounds)
                if min(abs(p[i] - lower), abs(p[i] - upper)) <= 1e-6 * max(1.0, abs(p[i]))],
            "error_dynamics_radius": float(np.max(np.abs(p[4] - p[5:7]))),
            "validation_predictions_px": predicted.tolist(),
            "validation_residuals_px": residuals.tolist(),
            "baseline_predictions_px": baseline[labels[n:]].tolist(),
        }

    @staticmethod
    def _z_validation_metrics(samples, observations, fitted, jxy):
        start = fitted["training_count"]
        rows = samples[start:]
        actual = np.asarray(observations, dtype=float)[start:]
        predicted = np.asarray(fitted["validation_predictions_px"])
        baseline = np.asarray(fitted["baseline_predictions_px"])
        mean_error = np.linalg.solve(jxy, (actual - baseline).T).T
        ssm_error = np.linalg.solve(jxy, (actual - predicted).T).T
        result = {}
        for name in ("all", "up", "down"):
            mask = np.array([name == "all" or row["direction"] == name for row in rows])
            if not mask.any():
                raise ValueError("Missing validation direction")
            result[name] = {
                "count": int(mask.sum()),
                "mean_rmse_um": float(np.sqrt(np.mean(np.sum(mean_error[mask] ** 2, axis=1)))),
                "ssm_rmse_um": float(np.sqrt(np.mean(np.sum(ssm_error[mask] ** 2, axis=1)))),
                "ssm_bias_um": ssm_error[mask].mean(axis=0).tolist(),
            }
        for row, prediction in zip(rows, predicted):
            row["predicted_error_px"] = prediction.tolist()
            row["predicted_correction_um"] = (-np.linalg.solve(jxy, prediction)).tolist()
        pair_ids = sorted({row["pair_id"] for row in rows})
        benefits = []
        for pair_id in pair_ids:
            mask = np.array([row["pair_id"] == pair_id for row in rows])
            benefits.append(float(np.mean(np.sum(mean_error[mask] ** 2, axis=1)
                                         - np.sum(ssm_error[mask] ** 2, axis=1))))
        rng = np.random.default_rng(0)
        draws = rng.choice(benefits, size=(2000, len(benefits)), replace=True).mean(axis=1)
        interval = np.quantile(draws, [0.025, 0.975]).tolist()
        result["paired_mse_benefit_interval_um2"] = interval
        no_direction_worse = all(result[d]["ssm_rmse_um"] <= result[d]["mean_rmse_um"]
                                 for d in ("up", "down"))
        from scipy.stats import pearsonr
        pair_residuals = np.asarray([np.concatenate([ssm_error[i] for i, row in enumerate(rows)
            if row["pair_id"] == pair_id]) for pair_id in pair_ids])
        correlations, probabilities = [], []
        for component in pair_residuals.T:
            if (len(component) < 4 or np.std(component[:-1]) < 1e-12
                    or np.std(component[1:]) < 1e-12):
                correlations.append(0.0)
                probabilities.append(1.0)
            else:
                correlation, probability = pearsonr(component[:-1], component[1:])
                correlations.append(float(correlation))
                probabilities.append(float(probability))
        dependent = any(p < 0.05 / len(probabilities) for p in probabilities)
        result["cross_cycle_dependence"] = dict(lag_pairs=1, correlations=correlations,
            p_values=probabilities, familywise_alpha=0.05, significant=dependent,
            method="lag-one Pearson per direction/coordinate; Bonferroni")
        result["pair_mse_benefits_um2"] = benefits
        for name in ("all", "up", "down"):
            mask = np.array([name == "all" or row["direction"] == name for row in rows])
            result[name].update(
                mean_rmse_px=float(np.sqrt(np.mean(np.sum((actual[mask] - baseline[mask]) ** 2, axis=1)))),
                ssm_rmse_px=float(np.sqrt(np.mean(np.sum((actual[mask] - predicted[mask]) ** 2, axis=1)))),
                ssm_bias_px=(actual[mask] - predicted[mask]).mean(axis=0).tolist())
        supported = interval[0] > 0 and no_direction_worse and not dependent
        return result, ("validated" if supported else "unsupported")

    def _hybrid_position(self, detected, frame, mask):
        """Use validated foreground motion, with detection anchoring and fallback."""
        log = logging.getLogger("patcherbot.PipetteCalibration")
        previous = getattr(self, "_flow_reference", None)
        position, source = detected.copy(), "detection"
        if previous is not None and frame is not None and mask is not None and previous["mask"] is not None:
            try:
                delta, quality = track_pipette_displacement(previous["frame"], frame,
                                                           previous["mask"], mask,
                                                           initial_displacement=detected - previous["detected"])
                if delta is not None:
                    disagreement = float(np.linalg.norm(delta - (detected - previous["detected"])))
                    if disagreement <= 12.0:
                        predicted = previous["position"] + delta
                        position = 0.85 * predicted + 0.15 * detected
                        source = "flow+detector"
                    log.info("Pipette calibration flow: %s; detector disagreement=%.2f px; %s",
                             source, disagreement, quality)
                else:
                    log.info("Pipette calibration flow rejected; using detection: %s", quality)
            except (cv2.error, ValueError, TypeError):
                log.warning("Pipette calibration flow failed; using detection", exc_info=True)
        else:
            log.info("Pipette calibration flow: detection anchor (no usable frame/mask pair)")
        self._flow_reference = dict(frame=frame, mask=mask, detected=detected.copy(), position=position.copy())
        return position, source

    def _log_calibration_quality(self, matrix, inliers):
        """Log fit residuals for the matrix selected by normal calibration."""
        points = np.asarray(self.cal_points, dtype=float)
        design = np.column_stack((points[:, 2:4], np.ones(len(points))))
        errors = np.linalg.norm(design @ matrix[:2, [0, 1, 3]].T - points[:, :2], axis=1)
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        self.last_calibration_quality = dict(
            mode="active_calibration", status="fitted", matrix=matrix.tolist(),
            points=points.tolist(), samples=self.calibration_samples,
            timestamp_basis="camera_retrieval_not_sensor_exposure",
            inliers=None if inliers is None else inliers.ravel().tolist(),
            training_errors_px=errors.tolist(), training_rmse_px=rmse)
        log = logging.getLogger("patcherbot.PipetteCalibration")
        singular = np.linalg.svd(points[:, 2:4] - points[:, 2:4].mean(axis=0), compute_uv=False)
        condition = float(singular[0] / singular[-1]) if singular[-1] > 1e-9 else float("inf")
        flow_count = sum(sample.get("source") == "flow+detector" for sample in self.calibration_samples)
        log.info("Pipette calibration geometry condition=%.2f; flow-assisted points=%d/%d",
                 condition, flow_count, len(points))
        log.info("Pipette calibration fit: %d hybrid points, %d inliers, "
                 "fit RMSE=%.3f px, maximum fit error=%.3f px (training residuals)",
                 len(points), 0 if inliers is None else int(np.count_nonzero(inliers)),
                 rmse, float(errors.max()))
        log.info("Pipette calibration selected matrix: %s", matrix.tolist())
        log.info("Pipette calibration errors by point (px): %s", np.round(errors, 3).tolist())

    def calibrate(self):
        """
        Computes a 2D affine transformation that maps pipette encoder (x, y)
        positions to image (x, y) positions in the camera image.
        
        It then embeds the resulting 2×3 matrix into a 3×4 homogeneous transformation matrix.
        
        Returns:
            A 3×4 transformation matrix.
            (The calibrated unit's finish_calibration routine may then convert this
             into a full 4×4 matrix as needed.)
        """
        if len(self.cal_points) < 3:
            print("Not enough calibration points for affine transformation.")
            return None

        # Prepare the data arrays (each is N×2):
        # encoder_points: [ [encoder_x, encoder_y], ... ]
        # image_points:   [ [image_x, image_y], ... ]
        encoder_points = np.array([[pt[2], pt[3]] for pt in self.cal_points], dtype=np.float64)
        image_points   = np.array([[pt[0], pt[1]] for pt in self.cal_points], dtype=np.float64)

        # Compute the 2D affine transformation (a 2×3 matrix)
        M2x3, inliers = cv2.estimateAffine2D(encoder_points, image_points)
        if M2x3 is None:
            print("Failed to compute affine transformation.")
            return None

        print("Raw 2D affine transformation matrix (2×3):")
        print(M2x3)

        # Convert the 2×3 matrix to a 3×4 homogeneous transformation matrix.
        # We assume the pipette moves only in x and y (z=0).
        # The resulting matrix has the form:
        #   [ A  B  0  t_x ]
        #   [ C  D  0  t_y ]
        #   [ 0  0  1   0  ]
        mat3x4 = np.zeros((3, 4), dtype=np.float64)
        mat3x4[0, 0:2] = M2x3[0, 0:2]
        mat3x4[0, 3]   = M2x3[0, 2]
        mat3x4[1, 0:2] = M2x3[1, 0:2]
        mat3x4[1, 3]   = M2x3[1, 2]
        mat3x4[2, 2]   = 1.0

        print("Converted 3×4 homogeneous transformation matrix:")
        print(mat3x4)

        # Save the calibration matrix for later use (e.g., for centering the pipette).
        self.calibration_matrix = mat3x4
        try:
            self._log_calibration_quality(mat3x4, inliers)
        except Exception:
            logging.exception("Could not log pipette calibration fit details")
        # Clear the calibration points after computing the matrix.
        self.cal_points = []
        return mat3x4

class PipetteFocusHelper():
    def __init__(self, pipette: Manipulator, camera: Camera, config=None, *, detector):
        self.pipette = pipette
        self.camera = camera
        self.config = config
        self.pipetteFocuser = PipetteFocuser2(detector)
    
    def focus(self,frame=None):
        """
        Adjusts the pipette focus by capturing an image,
        predicting the defocus value, and commanding a relative move
        using that value.
        """
        if frame is None:
            # Get the latest frame from the camera if not provided.
            _, _, _, frame = self.camera.raw_frame_queue[0]
        # convert to 8-bit for display
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        defocus_value = self.pipetteFocuser.get_pipette_focus_value(frame)
        # print(f"Defocus value: {defocus_value:.2f} µm")
        
        # Use the defocus value directly in the relative move command.
        self.pipette.relative_move([0, 0, -defocus_value])
        self.pipette.wait_until_still()
