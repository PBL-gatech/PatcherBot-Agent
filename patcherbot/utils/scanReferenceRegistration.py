"""Offline image registration of a session mosaic to the saved reference.

Transforms map native pixel centres. Rejected proposals never authorize overlays.
"""
import json
import os
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

VERSION = 1


def image_identity(scan):
    path = Path(scan['path']).resolve()
    identity = scan.get('source_identity')
    if identity is None:
        stat = path.stat()
        identity = (stat.st_size, stat.st_mtime_ns)
    return dict(path=os.path.normcase(str(path)), stat=list(identity),
                size=[int(scan['width']), int(scan['height'])])


def overview_to_native(scan):
    height, width = scan['pixels'].shape[:2]
    scale = np.array([scan['width'] / width, scan['height'] / height])
    matrix = np.eye(3)
    matrix[:2, :2] = np.diag(scale)
    matrix[:2, 2] = (scale - 1) / 2
    return matrix


def _features(scan):
    pixels = np.asarray(scan['pixels'])
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY) if pixels.ndim == 3 else pixels
    gray = np.ascontiguousarray(gray, dtype=np.uint8)
    mask = cv2.erode((gray > 2).astype(np.uint8) * 255, np.ones((9, 9), np.uint8))
    enhanced = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8)).apply(gray)
    keypoints, descriptors = cv2.SIFT_create(nfeatures=6000).detectAndCompute(enhanced, mask)
    return keypoints, descriptors


def _spread(points, shape):
    if len(points) < 3:
        return 0.
    return float(cv2.contourArea(cv2.convexHull(np.float32(points))) / (shape[0] * shape[1]))


def _register_sift(scan, reference):
    """Fit a similarity transform with spatial support and held-out checks.

    SIFT handles rotation and scale; reciprocal ratio matches and RANSAC reject
    changed tissue and harp features. Acceptance measures geometric consistency,
    not anatomical correctness. No hardware or source-image edits occur here.
    """
    result = dict(version=VERSION, accepted=False, reason='', metrics={},
                  scan_identity=image_identity(scan), reference_identity=image_identity(reference),
                  method='sift_reciprocal_similarity_ransac_heldout')
    def reject(reason):
        result['reason'] = reason
        return result
    source_keys, source_desc = _features(scan)
    target_keys, target_desc = _features(reference)
    metrics = result['metrics']
    metrics.update(source_features=len(source_keys), reference_features=len(target_keys))
    if source_desc is None or target_desc is None or min(len(source_keys), len(target_keys)) < 20:
        return reject('Too few image features to register this scan.')
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    def ratio_matches(a, b):
        return {pair[0].queryIdx: pair[0].trainIdx for pair in matcher.knnMatch(a, b, k=2)
                if len(pair) == 2 and pair[0].distance < .72 * pair[1].distance}
    forward = ratio_matches(source_desc, target_desc)
    backward = ratio_matches(target_desc, source_desc)
    pairs = [(i, j) for i, j in forward.items() if backward.get(j) == i]
    metrics['matches'] = len(pairs)
    if len(pairs) < 20:
        return reject('Too few reciprocal feature matches to the reference.')
    source = np.float32([source_keys[i].pt for i, _ in pairs])
    target = np.float32([target_keys[j].pt for _, j in pairs])
    return _fit_correspondences(scan, reference, source, target, result)


def _fit_correspondences(scan, reference, source, target, result, tolerance=4.):
    metrics = result['metrics']
    metrics['matches'] = len(source)
    def reject(reason):
        result['accepted'] = False
        result['reason'] = reason
        return result
    if len(source) < 20:
        return reject('Too few feature matches to the reference.')
    order = np.random.default_rng(0).permutation(len(source))
    heldout, training = order[::4], np.delete(order, np.arange(0, len(order), 4))
    proposal, _ = cv2.estimateAffinePartial2D(source[training], target[training],
        method=cv2.RANSAC, ransacReprojThreshold=.75 * tolerance, maxIters=5000, confidence=.999, refineIters=10)
    if proposal is None:
        return reject('No consistent image transform was found.')
    errors = np.linalg.norm(source @ proposal[:, :2].T + proposal[:, 2] - target, axis=1)
    heldout_good = errors[heldout] <= tolerance
    metrics.update(heldout_matches=len(heldout), heldout_inliers=int(heldout_good.sum()),
                   heldout_inlier_fraction=float(heldout_good.mean()))
    if heldout_good.sum() < 4 or heldout_good.mean() < .35:
        return reject('Transform failed independent held-out feature checks.')
    good = errors <= tolerance
    if good.sum() < 16 or good.mean() < .3:
        return reject('Insufficient consistent feature support for this transform.')
    final, _ = cv2.estimateAffinePartial2D(source[good], target[good],
        method=cv2.RANSAC, ransacReprojThreshold=.75 * tolerance, maxIters=5000, confidence=.999, refineIters=10)
    if final is None or not np.isfinite(final).all():
        return reject('The refined image transform is invalid.')
    errors = np.linalg.norm(source @ final[:, :2].T + final[:, 2] - target, axis=1)
    good = errors <= tolerance
    source_coverage = _spread(source[good], scan['pixels'].shape)
    reference_coverage = _spread(target[good], reference['pixels'].shape)
    scale = float(np.linalg.norm(final[:, 0]))
    metrics.update(inliers=int(good.sum()), inlier_fraction=float(good.mean()),
                   median_error_overview_px=float(np.median(errors[good])),
                   source_coverage=source_coverage, reference_coverage=reference_coverage,
                   overview_scale=scale, rotation_degrees=float(np.degrees(np.arctan2(final[1, 0], final[0, 0]))))
    if min(source_coverage, reference_coverage) < .015:
        return reject('Matches cover too little of the images to establish a reliable alignment.')
    if not .2 <= scale <= 5.:
        return reject('Estimated scale is outside the supported range.')
    matrix = np.eye(3)
    matrix[:2] = final
    native = overview_to_native(reference) @ matrix @ np.linalg.inv(overview_to_native(scan))
    if not np.isfinite(native).all() or np.linalg.det(native[:2, :2]) <= 0:
        return reject('Native-coordinate transform is invalid.')
    result.update(accepted=True, reason='Image alignment passed geometric checks.',
                  scan_to_reference_native=native.tolist())
    return result


def register_scan_to_reference(scan, reference):
    result = _register_sift(scan, reference)
    if result['accepted']:
        return result
    try:
        from patcherbot.utils.scanReferenceFeatures import learned_correspondences
        source, target, scores = learned_correspondences(scan['pixels'], reference['pixels'])
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        result['metrics']['fallback_unavailable'] = str(error)
        return result
    learned = dict(result, accepted=False, method='lightglue_superpoint_similarity_ransac_heldout',
                   metrics={'sift_matches': result['metrics'].get('matches', 0),
                            'feature_method': 'SuperPoint / LightGlue'})
    tolerance = max(4., max(reference['pixels'].shape[:2]) / 96.)
    learned['metrics']['validation_tolerance_overview_px'] = tolerance
    learned = _fit_correspondences(scan, reference, source, target, learned, tolerance=tolerance)
    if learned['accepted']:
        learned['quality'] = 'coarse'
        error = learned['metrics']['median_error_overview_px']
        learned['reason'] = f'Coarse reference alignment; median residual {error:.2f} overview pixels.'
    if not learned['accepted']:
        result['metrics']['fallback_reason'] = learned['reason']
        result['metrics']['fallback_matches'] = learned['metrics']['matches']
        return result
    return learned


def registration_path(scan_path):
    path = Path(scan_path)
    return path.with_name(path.stem + '_reference_registration.json')


def save_registration(result, scan_path):
    destination = registration_path(scan_path)
    temporary = destination.with_name(destination.name + '.' + uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return str(destination)


def load_registration(scan, reference):
    """Reuse only an accepted result bound to both unchanged source images."""
    try:
        result = json.loads(registration_path(scan['path']).read_text(encoding='utf-8'))
        matrix = np.asarray(result.get('scan_to_reference_native'), dtype=float)
        if (result.get('version') != VERSION or not result.get('accepted')
                or result.get('scan_identity') != image_identity(scan)
                or result.get('reference_identity') != image_identity(reference)
                or matrix.shape != (3, 3) or not np.isfinite(matrix).all()
                or not np.allclose(matrix[2], [0, 0, 1])
                or np.linalg.det(matrix[:2, :2]) <= 0):
            return None
        return result
    except (OSError, ValueError, TypeError, KeyError):
        return None
