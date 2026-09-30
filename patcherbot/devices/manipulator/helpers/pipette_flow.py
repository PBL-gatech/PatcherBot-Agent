"""Image-only pipette displacement measurement; never sends hardware commands."""
import cv2
import numpy as np


MIN_TRACKS = 6
MAX_FEATURES = 80
MAX_FB_ERROR_PX = 1.0
MAX_SPREAD_PX = 2.0


def track_pipette_displacement(previous_frame, current_frame, previous_mask, current_mask,
                               *, initial_displacement=None):
    """Return native-pixel translation and quality, or ``(None, quality)``.

    Frames must be matching uint8 grayscale images. Masks must be matching native
    foreground probabilities. Only tracks inside both foreground masks qualify.
    A detection displacement seeds tracking; image evidence still determines the result.
    """
    quality = dict(status="invalid_input", features=0, valid_tracks=0, inliers=0)

    def reject(status):
        quality["status"] = status
        return None, quality

    if previous_mask is None or current_mask is None:
        return reject("missing_mask")
    try:
        prior = None
        if initial_displacement is not None:
            prior = np.asarray(initial_displacement, dtype=np.float32)
            if prior.shape != (2,) or not np.isfinite(prior).all():
                return reject("invalid_initial_displacement")
            quality["initial_displacement_px"] = prior.tolist()
        previous = np.asarray(previous_frame)
        current = np.asarray(current_frame)
        old_probability = np.asarray(previous_mask, dtype=float)
        new_probability = np.asarray(current_mask, dtype=float)
        if (previous.ndim != 2 or previous.size == 0 or min(previous.shape) < 3
                or previous.dtype != np.uint8 or current.dtype != np.uint8
                or current.shape != previous.shape
                or old_probability.shape != previous.shape
                or new_probability.shape != previous.shape
                or not np.isfinite(old_probability).all()
                or not np.isfinite(new_probability).all()):
            return reject("invalid_input")
        old_foreground = (old_probability >= .5).astype(np.uint8) * 255
        new_foreground = new_probability >= .5
        if not old_foreground.any() or not new_foreground.any():
            return reject("empty_mask")
        previous = np.ascontiguousarray(previous)
        current = np.ascontiguousarray(current)
        points = cv2.goodFeaturesToTrack(previous, maxCorners=MAX_FEATURES,
                                         qualityLevel=.01, minDistance=5,
                                         mask=old_foreground, blockSize=7)
        if points is None:
            return reject("insufficient_features")
        quality["features"] = int(len(points))
        if len(points) < MIN_TRACKS:
            return reject("insufficient_features")
        options = dict(winSize=(21, 21), maxLevel=3,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
        initial_forward = None
        if prior is not None:
            initial_forward = points + prior.reshape(1, 1, 2)
            options["flags"] = cv2.OPTFLOW_USE_INITIAL_FLOW
        forward, status_forward, _ = cv2.calcOpticalFlowPyrLK(
            previous, current, points, initial_forward, **options)
        if forward is None or status_forward is None:
            return reject("tracking_failed")
        source = points.reshape(-1, 2)
        destination = forward.reshape(-1, 2)
        forward_valid = status_forward.reshape(-1).astype(bool) & np.isfinite(destination).all(axis=1)
        if int(forward_valid.sum()) < MIN_TRACKS:
            return reject("insufficient_tracks")
        source = source[forward_valid]
        destination = destination[forward_valid]
        initial_backward = None if prior is None else (destination - prior).reshape(-1, 1, 2)
        backward, status_backward, _ = cv2.calcOpticalFlowPyrLK(
            current, previous, destination.reshape(-1, 1, 2), initial_backward, **options)
        if backward is None or status_backward is None:
            return reject("tracking_failed")
        returned = backward.reshape(-1, 2)
        fb_error = np.linalg.norm(returned - source, axis=1)
        height, width = previous.shape
        # Clip only for safe mask indexing; the separate bounds predicate rejects
        # endpoints outside the image instead of clamping their measurements.
        bounds = ((destination[:, 0] >= 0) & (destination[:, 0] < width)
                  & (destination[:, 1] >= 0) & (destination[:, 1] < height))
        x = np.rint(np.clip(destination[:, 0], 0, width - 1)).astype(np.int64)
        y = np.rint(np.clip(destination[:, 1], 0, height - 1)).astype(np.int64)
        foreground = new_foreground[y, x]
        valid = (status_backward.reshape(-1).astype(bool) & np.isfinite(fb_error)
                 & (fb_error <= MAX_FB_ERROR_PX) & bounds & foreground)
        quality["valid_tracks"] = int(valid.sum())
        if prior is not None:
            valid &= np.linalg.norm(destination - source - prior, axis=1) <= 12.0
            quality["guided_tracks"] = int(valid.sum())
        if int(valid.sum()) < MIN_TRACKS:
            return reject("insufficient_tracks")
        displacements = destination[valid] - source[valid]
        delta = np.median(displacements, axis=0)
        distances = np.linalg.norm(displacements - delta, axis=1)
        mad = float(np.median(np.abs(distances - np.median(distances))))
        median_distance = float(np.median(distances))
        cutoff = min(MAX_SPREAD_PX, max(.5, median_distance + 3. * 1.4826 * mad))
        quality.update(median_track_spread_px=median_distance, consensus_cutoff_px=cutoff)
        inliers = distances <= cutoff
        quality["inliers"] = int(inliers.sum())
        if quality["inliers"] < MIN_TRACKS:
            return reject("inconsistent_tracks")
        delta = np.median(displacements[inliers], axis=0)
        spread = float(np.max(np.linalg.norm(displacements[inliers] - delta, axis=1)))
        quality.update(max_spread_px=spread,
                       median_fb_error_px=float(np.median(fb_error[valid][inliers])))
        if not np.isfinite(delta).all() or spread > MAX_SPREAD_PX:
            return reject("inconsistent_tracks")
        quality["displacement_px"] = delta.tolist()
        quality["status"] = "ok"
        return delta.astype(float), quality
    except (cv2.error, ValueError, TypeError, OverflowError):
        return reject("tracking_failed")
