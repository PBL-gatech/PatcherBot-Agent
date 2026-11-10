"""Image-level session/reference registration and identity-bound fit caches.

Transforms map native pixel centres. Rejected proposals never authorize overlays.
"""
import json
import os
from pathlib import Path
from uuid import uuid4
from functools import lru_cache
from hashlib import sha256
from time import perf_counter

import cv2
import numpy as np

VERSION = 2


@lru_cache(maxsize=32)
def _image_hash(path, size, mtime_ns):
    digest = sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def image_identity(scan):
    path = Path(scan['path']).resolve()
    stat = path.stat()
    identity = (stat.st_size, stat.st_mtime_ns)
    loaded_identity = scan.get('source_identity', identity)
    metadata = scan.get('metadata', {})
    geometry = {key: metadata[key] for key in (
        'calibration', 'pixel_scale_xy', 'output_origin_xy_pixels', 'flip_x', 'flip_y',
        'canvas_shape', 'downsampled_canvas_shape', 'capture_frame_id', 'scan_id', 'objective',
        'source_frames', 'registration', 'registration_role', 'source_manifest_sha256')
        if key in metadata}
    return dict(path=os.path.normcase(str(path)), stat=list(identity),
                loaded_stat=list(loaded_identity),
                size=[int(scan['width']), int(scan['height'])],
                sha256=_image_hash(str(path), *identity),
                geometry_sha256=sha256(json.dumps(geometry, sort_keys=True,
                    separators=(',', ':'), allow_nan=False).encode()).hexdigest())


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


def _fit_correspondences(scan, reference, source, target, result, tolerance=4., best_effort=False):
    metrics = result['metrics']
    metrics['matches'] = len(source)
    warnings=[]
    def reject(reason):
        result['accepted'] = False
        result['reason'] = reason
        return result
    if len(source) < (3 if best_effort else 20):
        return reject('Too few feature matches to the reference.')
    if best_effort and len(source)<20:
        warnings.append('Fewer than20 feature matches')
    order = np.random.default_rng(0).permutation(len(source))
    heldout, training = order[::4], np.delete(order, np.arange(0, len(order), 4))
    proposal, _ = cv2.estimateAffinePartial2D(source[training], target[training],
        method=cv2.RANSAC, ransacReprojThreshold=.75 * tolerance, maxIters=5000, confidence=.999, refineIters=10)
    if proposal is None and best_effort:
        proposal,_=cv2.estimateAffinePartial2D(source,target,method=cv2.RANSAC,
            ransacReprojThreshold=.75*tolerance,maxIters=5000,confidence=.999,refineIters=10)
    if proposal is None:
        return reject('No consistent image transform was found.')
    errors = np.linalg.norm(source @ proposal[:, :2].T + proposal[:, 2] - target, axis=1)
    heldout_good = errors[heldout] <= tolerance
    metrics.update(heldout_matches=len(heldout), heldout_inliers=int(heldout_good.sum()),
                   heldout_inlier_fraction=float(heldout_good.mean()))
    if heldout_good.sum() < 4 or heldout_good.mean() < .35:
        if not best_effort:return reject('Transform failed independent held-out feature checks.')
        warnings.append('Limited held-out feature agreement')
    good = errors <= tolerance
    if good.sum() < 16 or good.mean() < .3:
        if not best_effort:return reject('Insufficient consistent feature support for this transform.')
        warnings.append('Limited consistent feature support')
    final=proposal
    if good.sum()>=2:
        refined, _ = cv2.estimateAffinePartial2D(source[good], target[good],
            method=cv2.RANSAC, ransacReprojThreshold=.75 * tolerance, maxIters=5000, confidence=.999, refineIters=10)
        if refined is not None:final=refined
        elif not best_effort:return reject('The refined image transform is invalid.')
    if final is None or not np.isfinite(final).all():
        return reject('The refined image transform is invalid.')
    errors = np.linalg.norm(source @ final[:, :2].T + final[:, 2] - target, axis=1)
    good = errors <= tolerance
    source_coverage = _spread(source[good], scan['pixels'].shape)
    reference_coverage = _spread(target[good], reference['pixels'].shape)
    scale = float(np.linalg.norm(final[:, 0]))
    metrics.update(inliers=int(good.sum()), inlier_fraction=float(good.mean()),
                   median_error_overview_px=float(np.median(errors[good] if good.any() else errors)),
                   source_coverage=source_coverage, reference_coverage=reference_coverage,
                   overview_scale=scale, rotation_degrees=float(np.degrees(np.arctan2(final[1, 0], final[0, 0]))))
    if min(source_coverage, reference_coverage) < .015:
        if not best_effort:return reject('Matches cover too little of the images to establish a reliable alignment.')
        warnings.append('Limited spatial feature coverage')
    if not .2 <= scale <= 5.:
        if not best_effort:return reject('Estimated scale is outside the supported range.')
        warnings.append('Unusual estimated image scale')
    matrix = np.eye(3)
    matrix[:2] = final
    native = overview_to_native(reference) @ matrix @ np.linalg.inv(overview_to_native(scan))
    if not np.isfinite(native).all() or np.linalg.det(native[:2, :2]) <= 0:
        return reject('Native-coordinate transform is invalid.')
    result.update(accepted=True, reason='Image alignment passed geometric checks.',
                  scan_to_reference_native=native.tolist())
    if best_effort:
        result.update(geometric_validation_passed=not warnings,quality='unverified' if warnings else 'coarse',
            image_fit=dict(available=True,acceptance_policy='best_available'),quality_warnings=warnings)
    return result


def _register_mosaic_proposal(scan, reference):
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


def register_scan_to_reference(scan, reference):
    """Fit session mosaic pixels in the worker; leave stage calibration intact."""
    if scan.get('metadata', {}).get('registration_role') not in ('session', 'atlas'):
        return _register_mosaic_proposal(scan, reference)
    started = perf_counter()
    result = dict(version=VERSION, accepted=False, reason='', method='scan_image_fit', metrics={},
                  registration_id=uuid4().hex,
                  image_validation=dict(accepted=False, method='bounded_scan_image_validation',
                      raw_frames_used=False, hardware_coordinates_changed=False))
    try:
        from patcherbot.utils.scanImageFitter import register_scan_images, ImageFitConfig
        identities = dict(scan_identity=image_identity(scan), reference_identity=image_identity(reference))
        for identity in identities.values():
            if identity['loaded_stat'] != identity['stat']:
                raise ValueError('A source image changed after loading; reopen it.')
        scale = expected_native_image_scale(scan, reference)
        config = None if scale is None else ImageFitConfig(expected_native_scale=scale)
        fitted = register_scan_images(scan, reference, config=config)
        if fitted.get('accepted') is not True:
            fitted = _register_bounded_learned(scan, reference, fitted)
        if fitted.get('scan_to_reference_native') is None and fitted.get('candidate_scan_to_reference_native') is not None:
            fitted['scan_to_reference_native']=fitted['candidate_scan_to_reference_native']
            fitted['method']=fitted.get('candidate_method',fitted.get('method','image_candidate'))
        matrix=np.asarray(fitted.get('scan_to_reference_native'),dtype=float)
        if matrix.shape==(3,3) and np.isfinite(matrix).all() and np.allclose(matrix[2],[0,0,1]) and np.linalg.det(matrix[:2,:2])>1e-12:
            fitted.update(accepted=True,image_fit=dict(available=True,acceptance_policy='best_available'))
            if not _validated_image_result(fitted):
                fitted.update(quality='unverified',reason='Applied best available image alignment; quality is unverified. '+fitted.get('reason',''))
        result.update(fitted, **identities, version=VERSION)
        if (identities['scan_identity'] != image_identity(scan)
                or identities['reference_identity'] != image_identity(reference)):
            raise ValueError('A source image changed during image fitting; reopen it.')
        if result.get('accepted') is True and not _usable_image_result(result):
            raise ValueError('Image fitter did not return accepted image validation.')
    except (ImportError, OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        result.update(accepted=False, reason=f'Scan image registration unavailable: {error}')
        result['image_validation'] = dict(accepted=False, method='bounded_scan_image_validation',
            raw_frames_used=False, hardware_coordinates_changed=False)
        result.pop('scan_to_reference_native', None)
    result.setdefault('metrics', {})['total_elapsed_seconds'] = perf_counter() - started
    return result


def _validated_image_result(result):
    evidence = result.get('image_validation')
    return (isinstance(evidence, dict) and evidence.get('accepted') is True
            and evidence.get('method') == 'bounded_scan_image_validation'
            and evidence.get('raw_frames_used') is False
            and evidence.get('hardware_coordinates_changed') is False)


def _usable_image_result(result):
    fit=result.get('image_fit',{})
    return _validated_image_result(result) or (isinstance(fit,dict) and fit.get('available') is True
        and fit.get('acceptance_policy')=='best_available')


def _register_bounded_learned(scan, reference, initial):
    """Optional local-weight fallback; fixed768px bound, spatial/held-out checks."""
    def bounded(image):
        pixels=image['pixels'];height,width=pixels.shape[:2]
        ratio=min(1.,768/max(height,width))
        return dict(image,pixels=cv2.resize(pixels,(max(1,round(width*ratio)),max(1,round(height*ratio))),interpolation=cv2.INTER_AREA))
    try:
        from patcherbot.utils.scanReferenceFeatures import learned_correspondences
        source,target=bounded(scan),bounded(reference)
        a,b,_=learned_correspondences(source['pixels'],target['pixels'])
        proposal=dict(accepted=False,reason='',method='image_lightglue_similarity',quality='coarse',
            metrics=dict(initial_reason=initial.get('reason',''),learned_input_max_dimension=768),
            image_validation=dict(accepted=False,method='bounded_scan_image_validation',
                raw_frames_used=False,hardware_coordinates_changed=False))
        proposal=_fit_correspondences(source,target,a,b,proposal,tolerance=8.,best_effort=True)
        proposal['image_validation']['accepted']=bool(proposal.get('geometric_validation_passed'))
        proposal['image_validation']['validation_kind']='learned_features_spatial_heldout'
        if proposal['accepted']:
            proposal['reason']=('Coarse scan-image alignment passed spatial and held-out feature checks.'
                if proposal['image_validation']['accepted'] else 'Best available feature fit; '+', '.join(proposal.get('quality_warnings',[])))
            return proposal
        initial.setdefault('metrics',{}).update(learned_fallback_reason=proposal['reason'],learned_fallback_matches=len(a))
    except (ImportError,OSError,RuntimeError,ValueError,TypeError,KeyError) as error:
        initial.setdefault('metrics',{})['learned_fallback_unavailable']=str(error)
    return initial


def reference_image_metadata(reference, manifest_path):
    """Read the declared reference downsampling; never infer scale from content."""
    path = Path(manifest_path)
    if not path.is_file():
        return {}
    payload = path.read_bytes()
    manifest = json.loads(payload)
    declared = (path.parent / manifest['image_path']).resolve()
    if declared != Path(reference['path']).resolve():
        return {}
    if list(manifest.get('image_shape_yxc', [])[:2]) != [reference['height'], reference['width']]:
        raise ValueError('Reference manifest dimensions disagree with the image')
    geometry = manifest['coordinate_transform']
    return dict(canvas_shape=geometry['full_shape_yxc'],pixel_scale_xy=geometry['pixel_scale_xy'],
                source_manifest_sha256=sha256(payload).hexdigest())


def expected_native_image_scale(scan, reference):
    """Convert declared mosaic output sampling into a scalar image-scale prior."""
    scales=[]
    for image in (scan,reference):
        metadata=image.get('metadata',{})
        if 'canvas_shape' not in metadata or 'pixel_scale_xy' not in metadata:
            return None
        full=np.asarray(metadata['canvas_shape'][:2],dtype=float)[::-1]
        pixel=np.asarray(metadata['pixel_scale_xy'],dtype=float)
        native=np.array([image['width'],image['height']],dtype=float)
        if full.shape!=(2,) or pixel.shape!=(2,) or not np.isfinite(np.r_[full,pixel,native]).all() or np.any(np.r_[full,pixel,native]<=0):
            raise ValueError('Image sampling metadata must be finite positive XY dimensions')
        scales.append(pixel*full/native)
    ratio=scales[0]/scales[1]
    if max(ratio)/min(ratio)>1.02:
        return None  # Similarity fallback cannot honor a genuinely anisotropic prior.
    return float(np.sqrt(np.prod(ratio)))


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
    """Reuse accepted image fits only for the exact current images and geometry."""
    if scan.get('metadata', {}).get('registration_role') == 'atlas':
        return None  # An atlas cache requires the current session chain context.
    try:
        result = json.loads(registration_path(scan['path']).read_text(encoding='utf-8'))
        matrix = np.asarray(result.get('scan_to_reference_native'), dtype=float)
        if (result.get('version') != VERSION or not result.get('accepted')
                or (scan.get('metadata', {}).get('registration_role') == 'session'
                    and not _usable_image_result(result))
                or result.get('scan_identity') != image_identity(scan)
                or result.get('reference_identity') != image_identity(reference)
                or matrix.shape != (3, 3) or not np.isfinite(matrix).all()
                or not np.allclose(matrix[2], [0, 0, 1])
                or np.linalg.det(matrix[:2, :2]) <= 0):
            return None
        return result
    except (ImportError, OSError, RuntimeError, ValueError, TypeError, KeyError):
        return None


def _native_matrix(value):
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[2], [0, 0, 1]) or np.linalg.det(matrix[:2, :2]) <= 1e-12):
        raise ValueError('Image transform is not a finite invertible native affine matrix.')
    return matrix


def _session_chain_context(session, reference, session_registration):
    if session is None or reference is None or not isinstance(session_registration, dict):
        raise ValueError('A session scan and its accepted reference alignment are required.')
    if session_registration.get('accepted') is not True:
        raise ValueError('The session scan has no available reference alignment.')
    session_identity, reference_identity = image_identity(session), image_identity(reference)
    if (session_identity['loaded_stat'] != session_identity['stat']
            or reference_identity['loaded_stat'] != reference_identity['stat']
            or session_registration.get('scan_identity') != session_identity
            or session_registration.get('reference_identity') != reference_identity):
        raise ValueError('The session alignment does not match the current session and reference images.')
    matrix = _native_matrix(session_registration.get('scan_to_reference_native'))
    signature = sha256(json.dumps(session_registration, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return session_identity, reference_identity, matrix, signature


def register_atlas_to_session(atlas, session, reference, session_registration):
    """Fit atlas pixels to session pixels, then compose the session reference map."""
    result = dict(version=VERSION, accepted=False, pending=True, method='atlas_to_session_to_reference',
        registration_chain='atlas_to_session_to_reference', chain_version=1, metrics={})
    try:
        session_identity, reference_identity, session_matrix, signature = _session_chain_context(
            session, reference, session_registration)
        fitted = register_scan_to_reference(atlas, session)
        result.update(fitted, method='atlas_to_session_to_reference', pending=False,
            registration_chain='atlas_to_session_to_reference', chain_version=1,
            reference_identity=reference_identity, session_identity=session_identity,
            session_registration_signature=signature, atlas_to_session_method=fitted.get('method'))
        if fitted.get('accepted') is not True:
            result['reason'] = 'Atlas-to-session image fit unavailable. ' + fitted.get('reason', '')
            return result
        atlas_matrix = _native_matrix(fitted.get('scan_to_reference_native'))
        composed = _native_matrix(session_matrix @ atlas_matrix)
        current = _session_chain_context(session, reference, session_registration)
        if current[0] != session_identity or current[1] != reference_identity or current[3] != signature:
            raise ValueError('The session/reference alignment changed while fitting the atlas.')
        warnings = list(fitted.get('quality_warnings', []))
        if session_registration.get('quality') == 'unverified':
            warnings.append('The session-to-reference image alignment is unverified.')
        quality = 'unverified' if ('unverified' in (fitted.get('quality'), session_registration.get('quality'))) else fitted.get('quality', 'coarse')
        reason = 'Atlas aligned to the session scan and composed with its reference alignment.'
        if quality == 'unverified':
            reason += ' Quality unverified: ' + (warnings[-1] if warnings else 'One or both image fits are unverified.')
        result.update(accepted=True, atlas_to_session_native=atlas_matrix.tolist(),
            session_to_reference_native=session_matrix.tolist(), scan_to_reference_native=composed.tolist(),
            quality=quality, quality_warnings=warnings, session_quality=session_registration.get('quality'),
            chain_validation_passed=_validated_image_result(fitted) and _validated_image_result(session_registration),
            reason=reason)
    except (ImportError, OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
        result.update(accepted=False, pending=True, reason=f'Atlas alignment is waiting for a valid session: {error}')
        result.pop('scan_to_reference_native', None)
    return result


def load_atlas_registration(atlas, session, reference, session_registration):
    """Load only a chain bound to the exact current session fit and all three images."""
    try:
        session_identity, reference_identity, session_matrix, signature = _session_chain_context(
            session, reference, session_registration)
        result = json.loads(registration_path(atlas['path']).read_text(encoding='utf-8'))
        if (result.get('version') != VERSION or result.get('accepted') is not True
                or result.get('registration_chain') != 'atlas_to_session_to_reference'
                or result.get('chain_version') != 1 or result.get('scan_identity') != image_identity(atlas)
                or result.get('reference_identity') != reference_identity
                or result.get('session_identity') != session_identity
                or result.get('session_registration_signature') != signature):
            return None
        atlas_matrix = _native_matrix(result.get('atlas_to_session_native'))
        composed = _native_matrix(result.get('scan_to_reference_native'))
        if (not np.array_equal(_native_matrix(result.get('session_to_reference_native')), session_matrix)
                or not np.allclose(composed, session_matrix @ atlas_matrix, rtol=1e-12, atol=1e-10)):
            return None
        if result['scan_identity']['loaded_stat'] != result['scan_identity']['stat']:
            return None
        return result
    except (ImportError, OSError, RuntimeError, ValueError, TypeError, KeyError):
        return None
