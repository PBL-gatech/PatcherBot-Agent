import time
from copy import deepcopy
import cv2
import numpy as np
from patcherbot.devices.manipulator.microscope import Microscope
from patcherbot.devices.manipulator import Manipulator
from patcherbot.devices.camera import Camera
from threading import Thread
import math
from statistics import median


class FocusHelper():
    FOCUSING_MAX_SPEED = 25
    NORMAL_MAX_SPEED = 10000
    RADIUS_UM = 5.0
    ENDPOINT_TOLERANCE_UM = 0.2
    BOUNDARY_TOLERANCE_UM = 1.0
    TIP_STABILITY_UM = 5.0
    DEPTH_TOLERANCE_UM = 2.0
    MIN_STABLE_SAMPLES = 3
    MIN_PIPETTE_SAMPLES = 3
    MIN_CELL_SAMPLES = 2
    CONFIRM_TIMEOUT_S = 5.0
    CONFIRM_MAX_FRAMES = 8

    def __init__(self, microscope: Microscope, camera: Camera):
        """Bind focus devices and own at most one phase Search or Scan record."""
        self.microscope = microscope
        self.camera = camera
        self._search = None

    @property
    def active_search(self):
        """Whether this helper owns a Search or Scan record."""
        return self._search is not None

    @property
    def confirmed(self):
        """Whether the active search has enough settled confirming evidence."""
        if self._search is None:
            return False
        confirmation = self._search.get('confirmation')
        return confirmation is not None and confirmation['confirmed']

    @property
    def confirmed_tip_xy(self):
        """Return confirmed detector XY without selecting a display marker."""
        if not self.confirmed:
            return None
        return self._search['confirmation']['confirmed_tip_xy']

    @property
    def saved_tip_evidence(self):
        """Return copied provisional Scan evidence with its original frame anchors."""
        if self._search is None or not self._search['pipette']:
            return None
        evidence = self._search.get('best_tip_evidence')
        if evidence is None or evidence['stage_xy'] is None or evidence['pipette_xyz'] is None:
            return None
        return deepcopy(evidence)

    @property
    def empty(self):
        """Whether the active cell search finished without usable evidence."""
        return self._search is not None and self._search['empty']

    def confirm_search(self, vision, cell=None):
        """Consume settled evidence without starting or advancing motor motion."""
        if self._search is None or 'confirmation' not in self._search:
            return False
        return self.update_confirmation(self._search, vision, cell) is True

    def advance_search(
        self, *, actual_z, vision, cell, center_z, speed, motion, bounds,
        pipette, pixels_to_um_relative, confidence_threshold,
        abort_check, info, warning,
    ):
        """Advance Search or Scan while retaining ownership until explicit reset.

        A cell search and a pipette scan cannot replace one another's active
        trajectory; the owning phase must reset between those operations.
        """
        if self._search is not None:
            if self._search['pipette'] != pipette:
                raise RuntimeError('Reset the active focus search before changing its target')
            if 'confirmation' in self._search:
                return None
        self._search = self.update_sweep(
            self._search, actual_z, vision, cell,
            center_z=center_z,
            speed=speed,
            motion=motion,
            bounds=bounds,
            label='SCAN' if pipette else 'SEARCH',
            pipette=pipette,
            pixels_to_um_relative=pixels_to_um_relative,
            confidence_threshold=confidence_threshold,
            abort_check=abort_check,
            info=info,
            warning=warning,
        )
        return None

    def cancel_search(self):
        """Stop the owned search, retaining its record until explicit reset."""
        if self._search is not None:
            self.cancel_sweep(self._search)

    def reset_search(self):
        """Release owned evidence only after its focus motion has been stopped."""
        self.cancel_search()
        self._search = None

    @staticmethod
    def reset_state(state, *extra_keys):
        for key in ("focus_sweep", *extra_keys):
            state.pop(key, None)


    def relocate(self, target):
        self.microscope.absolute_move(target)
        self.microscope.wait_until_still()

    def retry_search(self, state, origin, *, bounds, reason, info):
        """Restart a bounded sweep; the phase owns when recovery is permitted."""
        sweep = state.pop('focus_sweep', None)
        if sweep is not None:
            self.cancel_sweep(sweep)
        state.update(cell_reacquisition_attempt=state.get('cell_reacquisition_attempt', 0) + 1,
                     cell_search_bounds=tuple(origin + offset for offset in bounds))
        info(f"Hunt SEARCH retry {state['cell_reacquisition_attempt']}: reason={reason}; "
             f"bounds={state['cell_search_bounds']}; maximum_deeper_search_um={bounds[1]:g}; "
             "last confirmed cell plane retained")
        return True



    def _distance_scale(self):
        if getattr(self.microscope, "config", None) is not None:
            return float(self.microscope.config.microscope_units_per_um)
        return float(getattr(self.microscope, "units_per_um", 5.0))

    def autofocusContinuous(self, distance, timeout=1):


        focusThread = FocusUpdater(self.microscope, self.camera)
        focusThread.start()
        scale = self._distance_scale()
        commandedPos = self.microscope.position() / scale + distance
        self.microscope.relative_move(distance)
        start_time = time.time()
        while abs(self.microscope.position() / scale - commandedPos) > 0.3:
            if time.time() - start_time > timeout:
                print("Timeout reached waiting for microscope movement")
                break
            time.sleep(0.05)

        focusThread.stop()
        focusThread.join()

        if len(focusThread.posFocusList) == 0:
            raise RuntimeError("No focus data collected!")

        bestIndex = np.argmax(focusThread.posFocusList[:, 1])
        bestPos = focusThread.posFocusList[bestIndex, 0]
        bestScore = focusThread.posFocusList[bestIndex, 1]

        return bestPos, bestScore

    def autofocus(self, dist=100):
        self.microscope.set_max_speed(self.FOCUSING_MAX_SPEED)
        scale = self._distance_scale()
        initPos = self.microscope.position() / scale
        # print("Moving downward to collect forward scores..")
        bestForwardPos, bestForwardScore = self.autofocusContinuous(-dist)
        # print(f"Focus values:{bestForwardPos/scale} um, {bestForwardScore} units")

        self.microscope.set_max_speed(self.NORMAL_MAX_SPEED)
        # self.microscope.absolute_move(initPos)

        # print ("moving up to reset")
        self.microscope.relative_move(dist)
        self.microscope.wait_until_still()
        self.microscope.set_max_speed(self.FOCUSING_MAX_SPEED)
        # print("Moving upward to collect backwards scores..")
        bestBackwardPos, bestBackwardScore = self.autofocusContinuous(dist)
        # print(f"Focus values:{bestBackwardPos/scale} um, {bestBackwardScore} units")

        self.microscope.set_max_speed(self.NORMAL_MAX_SPEED)

        finalPos = bestForwardPos if bestForwardScore >= bestBackwardScore else bestBackwardPos
        finalPos = finalPos / scale
        # print(f"FinalPosition: {finalPos}")
        self.microscope.absolute_move(finalPos)
        self.microscope.wait_until_still()


    def read_search_frame(self, state, stage, detector, *, expected_xy=None, radius_um=50.0):
        """Read one frame, its positions, and the selected detector's result."""
        frame, _, image, timing = self.camera.last_raw_frame_data(include_timing=True) or (None, None, None, {})
        image = None if image is None else image.copy()
        focus_z = self.microscope.position() / self._distance_scale()
        stage_xy = np.asarray(stage.position(), dtype=float)[:2].copy()
        vision = dict(source_frame=frame, frame_retrieval_started_at=timing.get('retrieval_started_at'),
            source_positions=dict(focus_z=focus_z, stage_xy=stage_xy),
            pipette_position=None, pipette_focus=None, cell_detections=[], confidence={}, status={})
        cell = None
        if image is not None:
            if state['pipette']:
                result = detector.detect_pipette_details(image.copy(), expected_tip_xy=expected_xy,
                    pixels_to_um=stage.pixels_to_um_relative, radius_um=radius_um)
                vision.update(pipette_position=result['tip_xy'], pipette_focus=result['z_um'])
                vision['confidence'] = dict(pipette_detector=result['box_confidence'],
                                            pipette_focuser=result['z_confidence'])
                vision['status']['pipette_focuser'] = 'ok' if result['z_um'] is not None else 'no_detection'
            else:
                cells = detector.detect_cells(image.copy())
                vision['cell_detections'] = cells
                candidates = [candidate for candidate in cells
                    if np.isfinite(candidate[:3]).all() and candidate[2] >= state['confidence_threshold']()]
                if expected_xy is None:
                    cell = max(candidates, key=lambda candidate: candidate[2], default=None)
                else:
                    candidates = [candidate for candidate in candidates
                        if np.linalg.norm(stage.pixels_to_um_relative(
                            np.asarray(candidate[:2]) - expected_xy)[:2]) <= radius_um]
                    cell = min(candidates, key=lambda candidate: np.linalg.norm(
                        np.asarray(candidate[:2]) - expected_xy), default=None)
        acquired = timing.get('retrieval_started_at')
        age = None if acquired is None else time.monotonic() - acquired
        vision.update(age_s=age, stale=age is None or not 0 <= age <= 1.0)
        actual_z = self.microscope.position() / self._distance_scale()
        return dict(camera_image=image, deep_learning=vision, cell_detection=cell,
                    stage_positions=np.append(stage.position()[:2], actual_z))

    def search(self, state, stage, detector, *, expected_xy=None, radius_um=50.0):
        """Advance a start_sweep() state once; result holds confirmed XY and Z."""
        if state['empty'] or state.get('result') is not None:
            return state
        try:
            state['abort_check']()
            observation = self.read_search_frame(state, stage, detector,
                expected_xy=expected_xy, radius_um=radius_um)
            state['observation'] = observation
            vision, cell = observation['deep_learning'], observation['cell_detection']
            if 'confirmation' not in state:
                self.update_sweep(state, observation['stage_positions'][2], vision, cell)
            elif self.update_confirmation(state, vision, cell):
                state['result'] = dict(xy=np.median(state['confirmation']['xy_samples'], axis=0),
                                       z=state['best_z'])
            return state
        except Exception:
            self.cancel_sweep(state)
            raise

    def start_sweep(self, center_z, speed, *, motion, pixels_to_um_relative,
                    abort_check, info, warning, confidence_threshold, bounds=None, label='SEARCH', pipette=False):
        """Return per-attempt focus state; the caller owns confidence and retry policy."""
        abort_check()
        center_z, speed = float(center_z), float(speed)
        if not math.isfinite(center_z) or not math.isfinite(speed) or speed <= 0:
            raise ValueError('Focus sweep needs a finite center and positive search speed')
        low, high = bounds if bounds is not None else (center_z-self.RADIUS_UM, center_z+self.RADIUS_UM)
        state = dict(center_z=center_z, low=low, high=high, label=label, pipette=pipette,
            stage='idle', target=None, best_z=None, best_score=-math.inf, best_rank=None,
            samples=0, seen_frames=set(), tip_samples=[], empty=False,
            best_xy=None, best_depth=None, best_tip_evidence=None,
            deadline=time.monotonic()+max(60.0, 4.0*20.0/speed*20.0), speed=speed,
            endpoint_correction=False, motion=None, motion_device=motion,
            pixels_to_um_relative=pixels_to_um_relative, confidence_threshold=confidence_threshold,
            abort_check=abort_check, info=info, warning=warning)
        self.microscope.stop()
        info(f'{label} FOCUS SWEEP: bounds=({low}, {high}); speed={speed}; '
             f'boundary_tolerance_um={self.BOUNDARY_TOLERANCE_UM}')
        self._move_focus(state, 'seek_start', low)
        return state

    def _move_focus(self, state, stage, target):
        """Advance the objective trajectory without taking over motor monitoring."""
        state.update(stage=stage, target=float(target), endpoint_correction=False)
        if stage in ('down', 'up'):
            speed, deadline = float(state['speed']), float(state['deadline'])
            if not all(math.isfinite(value) for value in (target, speed, deadline)) or speed <= 0:
                raise ValueError('Focus motion requires a finite target, deadline and positive speed')
            reversal_tolerance = float(self.BOUNDARY_TOLERANCE_UM)
            if not math.isfinite(reversal_tolerance) or reversal_tolerance < 0:
                raise ValueError('Focus reversal tolerance must be finite and nonnegative')

            def read_position():
                scale = self._distance_scale()
                if not math.isfinite(scale) or scale <= 0:
                    raise RuntimeError('Focus sweep has an invalid microscope position scale')
                z = float(self.microscope.position()) / scale
                if not math.isfinite(z):
                    raise RuntimeError('Focus sweep has no valid objective position')
                return z

            state['motion'] = state['motion_device'].start(
                self.microscope, target=float(target), deadline=deadline,
                read_position=read_position,
                move_velocity=lambda direction: self.microscope.absolute_move_velocity(-direction * speed),
                abort_check=state['abort_check'], info=state['info'],
                label=f'{state["label"]} focus sweep',
                reversal_tolerance=reversal_tolerance, reversal_slack=1e-6)
        else:
            self.microscope.absolute_move(float(target))

    @classmethod
    def _focus_sample_rank(cls, sample):
        """Prefer confident pipette XY near focus without consulting Z confidence."""
        score, _, _, depth = sample[:4]
        if depth is None:
            return (0, score, 0)
        error = abs(depth)
        return (2, score, -error) if error <= cls.DEPTH_TOLERANCE_UM else (1, -error, score)

    def _record_focus_sample(self, state, *, frame_id, focus_z, xy, score, stale,
                             depth=None, source_positions=None):
        """Select an actually sampled plane supported by three stable XY detections."""
        seen, samples = state['seen_frames'], state['tip_samples']
        if score is None or focus_z is None or stale or frame_id is None or frame_id in seen:
            if frame_id not in seen:
                samples.clear()
            return
        seen.add(frame_id)
        score, focus_z = float(score), float(focus_z)
        if not math.isfinite(score) or not math.isfinite(focus_z) or not state['low'] <= focus_z <= state['high']:
            samples.clear()
            return
        state['samples'] += 1
        if xy is None or len(xy) < 2 or not all(math.isfinite(float(value)) for value in xy[:2]):
            samples.clear()
            return
        xy = tuple(float(value) for value in xy[:2])
        try:
            depth = float(depth)
        except (TypeError, ValueError):
            depth = math.nan
        depth = depth if math.isfinite(depth) else None
        if any(not np.linalg.norm(state['pixels_to_um_relative'](np.r_[np.subtract(xy, previous[2]), 0.])[:2]) <= self.TIP_STABILITY_UM for previous in samples):
            samples.clear()
        # The winning sample may precede this frame; keep its own copied anchors.
        sources = source_positions or {}
        evidence = deepcopy(dict(xy=xy, source_frame=frame_id,
            stage_xy=sources.get('stage_xy'), pipette_xyz=sources.get('pipette_xyz')))
        samples.append((score, focus_z, xy, depth, evidence))
        del samples[:-self.MIN_STABLE_SAMPLES]
        if len(samples) < self.MIN_STABLE_SAMPLES:
            return
        rank_sample = self._focus_sample_rank if state['pipette'] else lambda sample: (sample[0],)
        candidate = sorted(samples, key=rank_sample)[len(samples)//2]
        rank = rank_sample(candidate)
        if state['best_rank'] is None or rank > state['best_rank']:
            score, focus_z, xy, depth = candidate[:4]
            state.update(best_rank=rank, best_score=score, best_z=focus_z,
                         best_xy=xy, best_depth=depth if state['pipette'] else None,
                         best_tip_evidence=candidate[4] if state['pipette'] else None)

    @staticmethod
    def _focus_evidence(state, vision, cell):
        """Read detector evidence using the caller's current acceptance threshold."""
        if state['pipette']:
            xy, score = vision.get('pipette_position'), vision.get('confidence', {}).get('pipette_detector')
            if (xy is not None and np.shape(xy) == (2,) and np.isfinite(xy).all()
                    and score is not None and np.isfinite(score)
                    and score >= state['confidence_threshold']()):
                return xy, float(score)
            return xy, None
        xy = None if cell is None else cell[:2]
        if (cell is None or len(cell) < 3 or not np.isfinite(cell[:3]).all()
                or cell[2] < state['confidence_threshold']()):
            return xy, None
        return xy, float(cell[2])

    def update_sweep(self, state, actual_z, vision, cell=None, **start_options):
        """Consume one scored frame and advance the bounded focus trajectory."""
        if state is None:
            return self.start_sweep(**start_options)
        frame_id, focus_z = vision.get('source_frame'), vision.get('source_positions', {}).get('focus_z')
        xy, score = self._focus_evidence(state, vision, cell)
        stale, depth = vision.get('stale', True), vision.get('pipette_focus')
        state['abort_check']()
        stage, label = state['stage'], state['label']
        if stage in ('done', 'empty'):
            return state
        if stage == 'idle':
            raise RuntimeError('Focus sweep has not started')
        motion_done = (state['motion_device'].is_complete(state['motion'])
                       if state['motion'] is not None else True)
        if time.monotonic() > state['deadline']:
            if state['endpoint_correction']:
                raise RuntimeError(f'{label} focus sweep endpoint correction timed out; '
                    f'leg={stage}; target_z={state["target"]}; actual_z={actual_z}')
            raise RuntimeError(f'{label} focus sweep timed out during {stage}')
        if stage in ('down', 'up') and not state['endpoint_correction']:
            self._record_focus_sample(state, frame_id=frame_id, focus_z=focus_z, xy=xy,
                                      score=score, stale=stale, depth=depth,
                                      source_positions=vision.get('source_positions'))
        z = float(actual_z)
        if not math.isfinite(z):
            raise RuntimeError(f'{label} focus sweep has no valid objective position')
        if stage in ('down', 'up') and not motion_done:
            return state
        tolerance = (self.BOUNDARY_TOLERANCE_UM if stage in ('down', 'up') and not state['endpoint_correction']
                     else self.ENDPOINT_TOLERANCE_UM)
        target = state['target']
        if abs(z-target) > tolerance+1e-6:
            direction = 1 if stage == 'down' else -1
            if stage in ('down', 'up') and not state['endpoint_correction'] and direction*(z-target) > self.BOUNDARY_TOLERANCE_UM:
                state['endpoint_correction'] = True
                state['info'](f'{label} FOCUS SWEEP: correcting stopped endpoint overshoot; '
                    f'leg={stage}; actual_z={z}; target_z={target}; error_um={abs(z-target):.3f}; one absolute correction')
                self.microscope.absolute_move(target)
            return state
        if stage in ('seek_start', 'down'):
            next_stage, endpoint = ('down', 'high') if stage == 'seek_start' else ('up', 'low')
            self._move_focus(state, next_stage, state[endpoint])
        elif stage == 'up':
            if state['best_z'] is None:
                if not state['pipette']:
                    state.update(empty=True, stage='empty')
                    return state
                state['best_z'] = state['center_z']
                state['warning'](f'{label} FOCUS SWEEP: no stable pipette XY samples; '
                    f'returning to saved pipette plane z={state["center_z"]}; fresh XY detection still required')
            elif state['pipette'] and (state['best_depth'] is None
                    or abs(state['best_depth']) > self.DEPTH_TOLERANCE_UM):
                state['warning'](f'{label} FOCUS SWEEP: no stable near-focus XY candidate; '
                    f'using best available XY fallback; predicted_defocus_um={state["best_depth"]}; '
                    f'box_confidence={state["best_score"]}; target_z={state["best_z"]}; Z confidence ignored')
            self._move_focus(state, 'return', state['best_z'])
        elif stage == 'return':
            self.cancel_sweep(state)
            state['stage'] = 'done'
            state['info'](f'{label} FOCUS SWEEP: finished; best_z={state["best_z"]}; '
                          f'score={state["best_score"]}; valid_samples={state["samples"]}')
            self.start_confirmation(state)
        return state

    def cancel_sweep(self, state):
        """Stop an active focus leg before its owning phase changes tasks."""
        if state.get('motion') is not None:
            state['motion_device'].cancel(state['motion'])
        else:
            self.microscope.stop()

    def start_confirmation(self, sweep):
        """Create a stationary-frame accumulator after returning to a focus plane."""
        pipette, selected_xy = sweep['pipette'], sweep.get('best_xy')
        if pipette and selected_xy is not None:
            try:
                selected_xy = tuple(float(value) for value in selected_xy)
            except (TypeError, ValueError):
                selected_xy = ()
            if len(selected_xy) != 2 or not all(math.isfinite(value) for value in selected_xy):
                selected_xy = ()
        sweep['confirmation'] = dict(label='pipette' if pipette else 'cell', pipette=pipette, selected_xy=selected_xy,
            pixels_to_um_relative=sweep['pixels_to_um_relative'], info=sweep['info'], warning=sweep['warning'],
            started_at=time.monotonic(), valid_samples=0, depths=[], depth_advisory=sweep.get('depth_advisory', True),
            depth_warnings=set(), seen_frames=set(), last_frame=None, last_vision={}, confirmed=False,
            xy_samples=[], confirmed_tip_xy=None)

    @staticmethod
    def _confirmation_failed(state, reason):
        """Report settled-evidence failure without deciding the phase's retry policy."""
        vision = state['last_vision']
        depth = median(state['depths']) if state['depths'] else None
        raise RuntimeError(f'{state["label"]} focus confirmation failed: {reason}; '
            f'fresh_frames={len(state["seen_frames"])}; valid_samples={state["valid_samples"]}; '
            f'median_abs_depth_um={depth}; last_depth_um={vision.get("pipette_focus")}; '
            f'confidence={vision.get("confidence")}; model_status={vision.get("status")}')

    def update_confirmation(self, sweep, vision, cell=None):
        """Accept distinct settled frames with stable XY; pipette depth stays advisory."""
        xy, score = self._focus_evidence(sweep, vision, cell)
        valid, state = score is not None, sweep['confirmation']
        frame_id, retrieved_at = vision.get('source_frame'), vision.get('frame_retrieval_started_at')
        stale, confidence = vision.get('stale', True), vision.get('confidence', {})
        depth, focus_status = vision.get('pipette_focus'), vision.get('status', {}).get('pipette_focuser')
        if state['confirmed']:
            return True
        if time.monotonic()-state['started_at'] >= self.CONFIRM_TIMEOUT_S:
            self._confirmation_failed(state, 'timed out waiting for settled camera evidence')
        if (stale or frame_id is None or frame_id in state['seen_frames'] or retrieved_at is None
                or not math.isfinite(retrieved_at) or retrieved_at < state['started_at']):
            return None
        state['seen_frames'].add(frame_id)
        state['last_frame'] = frame_id
        state['last_vision'] = dict(pipette_focus=depth, confidence=confidence,
                                   status={'pipette_focuser': focus_status})
        sample_valid = (bool(valid) and xy is not None and len(xy) >= 2
                        and all(math.isfinite(float(value)) for value in xy[:2]))
        expected, samples = state['selected_xy'], state['xy_samples']
        if sample_valid:
            xy = tuple(float(value) for value in xy[:2])
            if expected is not None:
                distance = np.linalg.norm(state['pixels_to_um_relative'](np.r_[np.subtract(xy, expected), 0.])[:2]) if len(expected) >= 2 else math.inf
                sample_valid = math.isfinite(distance) and distance <= self.TIP_STABILITY_UM
                if not sample_valid:
                    state['info'](f'{state["label"]} FOCUS CONFIRM: XY rejected; current_xy={xy}; '
                        f'selected_xy={expected}; selected_error_um={distance}; tolerance_um={self.TIP_STABILITY_UM}')
        if sample_valid:
            if any(not np.linalg.norm(state['pixels_to_um_relative'](np.r_[np.subtract(xy, previous), 0.])[:2]) <= self.TIP_STABILITY_UM for previous in samples):
                samples.clear()
            samples.append(xy)
        else:
            samples.clear()
        state['valid_samples'] = len(samples)
        if state['pipette'] and sample_valid and state['depth_advisory']:
            try:
                numeric_depth = float(depth)
            except (TypeError, ValueError):
                numeric_depth = math.nan
            if math.isfinite(numeric_depth):
                state['depths'].append(abs(numeric_depth))
            else:
                state['depth_warnings'].add('missing or non-finite Z estimate')
            if focus_status != 'ok':
                state['depth_warnings'].add(f'Z model status={focus_status}')
        aggregate = median(state['depths']) if state['depths'] else None
        state['confirmed'] = state['valid_samples'] >= (self.MIN_PIPETTE_SAMPLES if state['pipette'] else self.MIN_CELL_SAMPLES)
        if state['confirmed'] and state['pipette']:
            candidate = tuple(median(point[axis] for point in samples) for axis in (0, 1))
            distance = np.linalg.norm(state['pixels_to_um_relative'](np.r_[np.subtract(candidate, expected), 0.])[:2]) if expected is not None else 0.0
            if not math.isfinite(distance) or distance > self.TIP_STABILITY_UM:
                state['confirmed'] = False
                state['valid_samples'] = 0
                samples.clear()
                state['info'](f'{state["label"]} FOCUS CONFIRM: median XY rejected; selected_tip_error_um={distance}')
            else:
                state['confirmed_tip_xy'] = candidate
                if aggregate is not None and aggregate > self.DEPTH_TOLERANCE_UM:
                    state['depth_warnings'].add('median Z estimate exceeds 2 um')
                if state['depth_warnings']:
                    state['warning'](f'{state["label"]} FOCUS CONFIRM: reliable pipette XY accepted; '
                        f'Z focus remains advisory: {"; ".join(sorted(state["depth_warnings"]))}; '
                        f'median_abs_depth_um={aggregate}; confidence={confidence}')
        if state['confirmed']:
            return True
        if len(state['seen_frames']) >= self.CONFIRM_MAX_FRAMES:
            self._confirmation_failed(state, 'no reliable focus confirmation within the fresh-frame limit')
        return None


    def confirmed_evidence(self, sweep, vision, *, pipette=False, used_frames=()):
        """Return settled confirmation only when this frame is fresh and unused."""
        confirmation, frame = sweep.get('confirmation'), vision.get('source_frame')
        if confirmation is None or confirmation['confirmed'] is not True:
            return None
        minimum = self.MIN_PIPETTE_SAMPLES if pipette else self.MIN_CELL_SAMPLES
        if (bool(not confirmation['pipette']) == pipette
                or confirmation['valid_samples'] < minimum or len(confirmation['seen_frames']) < minimum
                or vision.get('stale', True) or frame is None or frame != confirmation.get('last_frame')
                or frame not in confirmation['seen_frames'] or frame in used_frames):
            return None
        return confirmation


class FocusUpdater(Thread):
    def __init__(self, microscope: Microscope, camera: Camera):
        super().__init__()
        self.isRunning = True
        self.camera = camera
        self.microscope = microscope
        self.posFocusList = []
        self.lastFrame = -1

    def run(self):
        while self.isRunning:
            if len(self.camera.raw_frame_queue) == 0:
                time.sleep(0.01)
                continue

            frame_no, frametime, _, img = self.camera.raw_frame_queue[-1]
            if frame_no == self.lastFrame:
                time.sleep(0.005)
                continue

            self.lastFrame = frame_no
            score = self._getFocusScore(img)
            self.posFocusList.append([self.microscope.position(), score])

        self.posFocusList = np.array(self.posFocusList)

    def _getFocusScore(self, image):
        focusSize = 512
        x = image.shape[1] / 2 - focusSize / 2
        y = image.shape[0] / 2 - focusSize / 2
        crop_img = image[int(y):int(y + focusSize), int(x):int(x + focusSize)]

        xEdges = cv2.norm(cv2.Sobel(src=crop_img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=7))
        yEdges = cv2.norm(cv2.Sobel(src=crop_img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=7))

        score = xEdges ** 2 + yEdges ** 2
        return score

    def stop(self):
        self.isRunning = False

class StageCalHelper():
    '''A helper class to aid with Stage Calibration
    '''
    
    CAL_MAX_SPEED = 1000
    NORMAL_MAX_SPEED = 10000

    def __init__(self, stage: Manipulator, camera: Camera, frameLag: int):
        self.stage : Manipulator = stage
        self.camera : Camera = camera
        self.lastFrameNo : int = None
        self.frameLag = frameLag

    def calibrateContinuous(self, distance, video=False):
        '''Tell the stage to go a certain distance at a low max speed.
           Take a bunch of pictures and run optical flow. Use optical flow information
           to create a linear transform from stage microns to image pixels.
           if set, video creates an mp4 of the optical flow running in the project directory.
        '''
        #move the microscope a certain distance forward and up
        currPos = self.stage.position()
        commandedPos = np.array([currPos[0] + distance, currPos[1] - distance])
        axes = np.array([0, 1], dtype=int)
        self.stage.absolute_move_group(commandedPos, axes)

        #wait for the microscope to reach the pos, recording frames
        framesAndPoses = []
        currPos = self.stage.position()
        startPos = currPos
        _, _, _, firstFrame = self.camera.raw_frame_queue[0]
        p0 = self.calcOpticalFlowP0(firstFrame)
        while abs(currPos[0] - commandedPos[0]) > 0.3 or abs(currPos[1] - commandedPos[1]) > 0.3:
            while self.lastFrameNo == self.camera.get_frame_no():
                time.sleep(0.05) #wait for a new frame to be read from the camera
            self.lastFrameNo = self.camera.get_frame_no()
            currPos = self.stage.position()

            #get latest img
            _, _, _, frame = self.camera.raw_frame_queue[0]

            framesAndPoses.append([frame.copy(), currPos[0] - startPos[0], currPos[1] - startPos[1]])

        #run optical flow on the recorded frames
        print('running optical flow...')
        imgPosStagePosList = []
        x_pix_total = 0
        y_pix_total = 0

        if video:
            out = cv2.VideoWriter('opticalFlow.mp4', -1, 10.0, (1024,1024))

        #calculate the average image
        avgImg = np.zeros_like(framesAndPoses[0][0], dtype=np.float64)
        for frame, _, _ in framesAndPoses:
            avgImg += frame
        avgImg = avgImg / len(framesAndPoses)
        avgImg = avgImg.astype(np.uint8)

        #subtract average from all frames
        for frame, _, _ in framesAndPoses:
            frame -= avgImg

        for i in range(len(framesAndPoses) - 1):
            currFrame, x_microns, y_microns = framesAndPoses[i + 1]
            lastFrame, last_x_microns, last_y_microns = framesAndPoses[i]

            p0 = self.calcOpticalFlowP0(lastFrame)

            x_pix, y_pix = self.calcOpticalFlow(lastFrame, currFrame, p0)
            x_pix_total += x_pix
            y_pix_total += y_pix

            if math.isnan(x_pix) or math.isnan(y_pix): #if no corners can be found with optical flow, nan could be returned.  Don't add this to the list
                continue

            if video:
                vidFrame = cv2.cvtColor(currFrame.copy(), cv2.COLOR_GRAY2BGR)
                cv2.line(vidFrame, (512,512), (512 + int(x_pix_total), 512 + int(y_pix_total)), (255,0,0), 3)
                cv2.line(vidFrame, (600,100), (600 + int(x_pix_total), 100 + int(y_pix_total)), (255,0,0), 3)
                cv2.line(vidFrame, (900,400), (900 + int(x_pix_total), 400 + int(y_pix_total)), (255,0,0), 3)
                out.write(vidFrame)


            imgPosStagePosList.append([x_pix_total, y_pix_total, x_microns, y_microns])
        imgPosStagePosList = np.array(imgPosStagePosList)
        

        if video:
            out.release()
        
        #for some reason, estimateAffinePartial2D only works with int64
        #we can multiply by 100, to preserve 2 decimal places without affecting rotation / scaling portion of affline transform
        imgPosStagePosList = (imgPosStagePosList).astype(np.int64)
        print(imgPosStagePosList)

        #compute affine transformation matrix
        mat, inVsOut = cv2.estimateAffinePartial2D(imgPosStagePosList[:,2:4], imgPosStagePosList[:,0:2])

        #fix intercept - set image center --> stage center
        mat[0,2] = 0
        mat[1,2] = 0

        print('completed optical flow. matrix:')
        print(mat)

        #return transformation matrix
        return mat

    def calcOpticalFlowP0(self, firstFrame):
        #params for corner detector
        feature_params = dict(maxCorners = 100,
                                qualityLevel = 0.1,
                                minDistance = 10,
                                blockSize = 10)
        p0 = cv2.goodFeaturesToTrack(firstFrame, 70, 0.05, 25)
        # p0 = cv2.goodFeaturesToTrack(firstFrame, mask = None, **feature_params)

        return p0
    
    def calcMotionTranslation(self, lastFrame, currFrame):
        warp_mode = cv2.MOTION_TRANSLATION
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000,  1e-10)

        #compute transformation for image translation
        (cc, warp_matrix) = cv2.findTransformECC (lastFrame, currFrame,warp_matrix, warp_mode, criteria)

        #get x, y translation (pixels)
        print(warp_matrix)
        x_pix = warp_matrix[0,2]
        y_pix = warp_matrix[1,2]

        print(x_pix, y_pix)

        return x_pix, y_pix

    def calcOpticalFlow(self, lastFrame, currFrame, p0):
        #params for optical flow
        lk_params = dict(winSize  = (20, 20),
                    maxLevel = 20)

        # calculate optical flow from first frame
        p1, st, err = cv2.calcOpticalFlowPyrLK(lastFrame, currFrame, p0, None, **lk_params)

        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
        
         #find median movement vector
        dMovement = good_new - good_old
        medianVect = np.median(dMovement, axis=0)
        
        return medianVect[0], medianVect[1]


    def calibrate(self, dist=500):
        '''Calibrates the microscope stage using optical flow and stage encoders to create a um -> pixels transformation matrix
        '''

        self.stage.set_max_speed(self.CAL_MAX_SPEED)
        # self.stage.set_max_accel(10)

        initPos = self.stage.position()
        print('starting optical flow')
        mat = self.calibrateContinuous(dist)
        # commandedPos = np.array([initPos[0] + 200, initPos[1] - 200])
        # axes = np.array([0, 1], dtype=int)
        # self.stage.absolute_move_group(commandedPos, axes)
        self.stage.wait_until_still()
        currPos = self.stage.position()
        
        self.stage.set_max_speed(self.NORMAL_MAX_SPEED)
        self.stage.absolute_move(initPos)
        self.stage.wait_until_still()

        return mat
