"""Session image registration and map-display validity, without hardware I/O."""
from copy import deepcopy
from hashlib import sha256
import json
from threading import RLock
from uuid import uuid4

import numpy as np


class CoordinateReferenceError(ValueError):
    """A map position has no confirmed active session reference."""


def fingerprint(value):
    def encode(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                             allow_nan=False, default=encode).encode()).hexdigest()


def calibration_identity(manip, stage):
    # r0 is deliberately reset by the existing loader; M is the saved calibration.
    return fingerprint({'manip_M': manip['M'], 'stage_M': stage['M']})


def validate_coordinates_bundle(bundle):
    """Validate the complete candidate before any calibration device is changed."""
    if not isinstance(bundle, dict) or bundle.get('version') != 1:
        raise ValueError('Unsupported coordinate reference bundle')
    positions, records = bundle.get('positions'), bundle.get('records', {})
    if not isinstance(positions, dict) or not isinstance(records, dict):
        raise ValueError('Coordinate reference must contain position records')
    for name, value in positions.items():
        if value is None:
            continue
        array = np.asarray(value, dtype=float)
        shape = {'home': (6,), 'safe': (6,), 'bath': (3,), 'rinsing_bath': (3,),
                 'floor_Z': (), 'stage_floor': ()}.get(name)
        if (not np.isfinite(array).all() or isinstance(value, (bool, np.bool_))
                or (shape is not None and array.shape != shape)):
            raise ValueError(f'Invalid saved {name} coordinates')
        record = records.get(name)
        if record is not None:
            if not isinstance(record, dict) or not np.array_equal(record.get('value'), array):
                raise ValueError(f'Saved {name} provenance does not match its coordinates')
            if record.get('binding') is not None and not record.get('calibration_identity'):
                raise ValueError(f'Saved {name} binding has no calibration identity')
    display = bundle.get('display_anchor')
    if display is not None:
        if (not isinstance(display, dict) or display.get('version') != 1
                or display.get('objective') != 40):
            raise ValueError('Unsupported saved measured tip anchor')
        tip = np.asarray(display.get('tip_xy'), dtype=float)
        home = np.asarray(display.get('home'), dtype=float)
        if (tip.shape != (2,) or home.shape != (6,) or not np.isfinite(tip).all()
                or not np.isfinite(home).all()
                or not np.array_equal(home, positions.get('home'))):
            raise ValueError('Saved measured tip anchor does not match home')
    return deepcopy(bundle)


def compose_session_correction(scan_to_reference_native, reference_pixel_to_global_um,
                               session_pixel_to_uncorrected_global_um):
    matrices = [np.array(value, dtype=float, copy=True) for value in (
        scan_to_reference_native, reference_pixel_to_global_um,
        session_pixel_to_uncorrected_global_um)]
    for matrix in matrices:
        if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
                or not np.allclose(matrix[2], [0., 0., 1.])
                or abs(np.linalg.det(matrix[:2, :2])) < 1e-12):
            raise ValueError('Session registration requires finite invertible XY affines')
    return matrices[1] @ matrices[0] @ np.linalg.inv(matrices[2])


class SessionRegistration:
    """Share one accepted map fit; this object does not authorize hardware movement."""
    def __init__(self):
        self.capture_frame_id = uuid4().hex
        self.state = 'missing'
        self.reason = 'No accepted session alignment'
        self.generation = 0
        self._transform = None
        self._binding = None
        self._owners = {}
        self._listeners = []
        self._lock = RLock()

    @property
    def transform(self):
        with self._lock:
            return None if self._transform is None else self._transform.copy()

    @property
    def binding(self):
        with self._lock:
            return deepcopy(self._binding)

    def add_listener(self, callback):
        if callback not in self._listeners:
            self._listeners.append(callback)

    def _notify(self):
        for callback in tuple(self._listeners):
            callback()

    def invalidate(self, reason='Session alignment invalidated'):
        with self._lock:
            self.generation += 1
            self._transform = self._binding = None
            self.state, self.reason = 'invalid', str(reason)
        self._notify()

    def accept_registration(self, result, metadata, reference_pixel_to_global_um,
                            session_pixel_to_uncorrected_global_um):
        if result.get('accepted') is not True:
            raise CoordinateReferenceError('Session image alignment was not accepted')
        evidence = result.get('raw_frame_validation')
        raw_accepted = isinstance(evidence, dict) and evidence.get('accepted') is True
        image_evidence = result.get('image_validation')
        image_accepted = (isinstance(image_evidence, dict) and image_evidence.get('accepted') is True
            and image_evidence.get('method') == 'bounded_scan_image_validation'
            and image_evidence.get('raw_frames_used') is False
            and image_evidence.get('hardware_coordinates_changed') is False)
        proposal=result.get('image_fit',{})
        image_available=(isinstance(proposal,dict) and proposal.get('available') is True
            and proposal.get('acceptance_policy')=='best_available')
        if not (raw_accepted or image_accepted or image_available):
            raise CoordinateReferenceError('Session map placement requires accepted image validation or raw-frame validation')
        if not result.get('scan_identity') or not result.get('reference_identity'):
            raise CoordinateReferenceError('Registration is missing scan/reference identity')
        if metadata.get('registration_role') != 'session':
            raise CoordinateReferenceError('Only a session scan can reference live coordinates')
        if metadata.get('objective') not in (None, 4, '4', '4x'):
            raise CoordinateReferenceError('Session registration requires 4x scan geometry')
        transform = compose_session_correction(result['scan_to_reference_native'],
            reference_pixel_to_global_um, session_pixel_to_uncorrected_global_um)
        binding = dict(scan_identity=deepcopy(result['scan_identity']),
                       reference_identity=deepcopy(result['reference_identity']),
                       scan_to_reference_native=np.asarray(result['scan_to_reference_native']).tolist(),
                       reference_geometry=np.asarray(reference_pixel_to_global_um).tolist(),
                       stage_geometry=np.asarray(session_pixel_to_uncorrected_global_um).tolist(),
                       metadata_identity=fingerprint(metadata))
        binding['registration_id'] = fingerprint(binding)
        with self._lock:
            self._transform, self._binding = transform, binding
            self.state, self.reason = 'accepted', result.get('reason', 'Session alignment accepted')
            frame = metadata.get('capture_frame_id')
            for data in self._owners.values():
                for record in data['records'].values():
                    if (not record['loaded'] and record.get('calibration_identity')
                            and record['capture_frame_id'] == self.capture_frame_id
                            and (frame is None or frame == self.capture_frame_id)):
                        record['binding'] = deepcopy(binding)
        self._notify()
        return self.binding

    def record_positions(self, owner, positions, calibration_identity=None, display_anchor=None):
        with self._lock:
            data = deepcopy(self._owners.get(owner, dict(records={}, calibration_identity=None,
                                                        display_anchor=None)))
            if calibration_identity is not None:
                data['calibration_identity'] = calibration_identity
            if display_anchor is not None or 'home' in positions:
                data['display_anchor'] = deepcopy(display_anchor)
            for name, value in positions.items():
                if value is None:
                    data['records'].pop(name, None)
                    continue
                array = np.asarray(value, dtype=float)
                if not np.isfinite(array).all():
                    raise ValueError(f'{name} must contain finite coordinates')
                data['records'][name] = dict(value=array.tolist(), loaded=False,
                    capture_frame_id=self.capture_frame_id,
                    binding=deepcopy(self._binding) if data['calibration_identity'] else None,
                    calibration_identity=data['calibration_identity'])
            validate_coordinates_bundle(dict(version=1, positions={k: v['value'] for k, v
                in data['records'].items()}, **data))
            self._owners[owner] = data
        self._notify()

    def export_coordinates(self, owner):
        with self._lock:
            data = deepcopy(self._owners.get(owner, dict(records={}, calibration_identity=None,
                                                         display_anchor=None)))
            return dict(version=1, positions={k: deepcopy(v['value']) for k, v in data['records'].items()},
                        binding=deepcopy(self._binding), capture_frame_id=self.capture_frame_id, **data)

    def stage_coordinates(self, owner, bundle, calibration_identity=None):
        candidate = validate_coordinates_bundle(bundle)
        records = {}
        for name, value in candidate.get('positions', {}).items():
            if value is None:
                continue
            array = np.asarray(value, dtype=float)
            if not np.isfinite(array).all():
                raise ValueError(f'{name} must contain finite coordinates')
            saved = candidate.get('records', {}).get(name, {})
            records[name] = dict(value=array.tolist(), loaded=True,
                capture_frame_id=saved.get('capture_frame_id', candidate.get('capture_frame_id')),
                binding=deepcopy(saved.get('binding')) if candidate.get('version') == 1 else None,
                calibration_identity=saved.get('calibration_identity'))
        # Caller publishes after the entire calibration candidate has loaded.
        with self._lock:
            self._owners[owner] = dict(records=records, calibration_identity=calibration_identity,
                                      display_anchor=deepcopy(candidate.get('display_anchor')))

    def clear_coordinates(self, owner):
        with self._lock:
            self._owners.pop(owner, None)
        self._notify()

    def restore_coordinates(self, owner, snapshot):
        """Roll back an in-memory transaction without reclassifying fresh captures."""
        candidate = validate_coordinates_bundle(snapshot)
        with self._lock:
            self._owners[owner] = {key: deepcopy(candidate.get(key)) for key in
                ('records', 'calibration_identity', 'display_anchor')}
        self._notify()

    def update_calibration(self, owner, identity):
        """Publish a calibration change without replacing the accepted scan fit."""
        with self._lock:
            data = self._owners.get(owner)
            if data is None or data.get('calibration_identity') == identity:
                return False
            data['calibration_identity'] = identity
        self._notify()
        return True

    def check_calibration(self, owner, identity):
        with self._lock:
            if self._owners.get(owner, {}).get('calibration_identity') != identity:
                raise CoordinateReferenceError('Saved positions use a different calibration')

    def require_positions(self, owner, *names):
        with self._lock:
            if self.state != 'accepted' or self._binding is None:
                raise CoordinateReferenceError(f'Map placement unconfirmed: {self.reason}')
            data = self._owners.get(owner, {})
            for name in names:
                record = data.get('records', {}).get(name)
                if record is None:
                    raise CoordinateReferenceError(f'{name} is unreferenced: no coordinate record')
                if record.get('binding') != self._binding:
                    raise CoordinateReferenceError(f'{name} is unreferenced: session registration does not match')
                if record.get('calibration_identity') != data.get('calibration_identity'):
                    raise CoordinateReferenceError(f'{name} is unreferenced: calibration does not match')
                if (name in ('home', 'safe') and
                        (data.get('display_anchor') or {}).get('source') != 'measured_tip'):
                    raise CoordinateReferenceError(f'{name} is unreferenced: measured 40x tip anchor missing')

    def positions_ready(self, owner, *names):
        try:
            self.require_positions(owner, *names)
            return True
        except CoordinateReferenceError:
            return False

    def transform_state(self, state):
        transform = self.transform
        if transform is None:
            return state
        result = deepcopy(state)
        linear, shift = transform[:2, :2], transform[:2, 2] / 1000.
        def point(value):
            value = np.asarray(value, dtype=float).copy()
            value[..., :2] = value[..., :2] @ linear.T + shift
            return value
        for name in ('objective_xyz_mm', 'pipette_xyz_mm', 'fov_corners_mm'):
            if state.get(name) is not None:
                result[name] = point(state[name])
        for name in ('pipette_marker_direction_xyz', 'pipette_direction_xyz'):
            if state.get(name) is not None:
                value = np.asarray(state[name], dtype=float).copy()
                value[:2] = linear @ value[:2]
                result[name] = value
        if state.get('named_anchors_mm') is not None:
            result['named_anchors_mm'] = {name: point(value) if name in
                ('home', 'safe', 'cleaning_bath', 'pipette') else value
                for name, value in state['named_anchors_mm'].items()}
        if result.get('fov_corners_mm') is not None:
            corners = np.asarray(result['fov_corners_mm'])
            result['fov_size_mm'] = np.linalg.norm(corners[[1, 3]] - corners[0], axis=1)
        return result
