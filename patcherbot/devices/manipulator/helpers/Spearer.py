import math
import time


class Spearer:
    """Track the cell while the existing encoder monitor owns descent."""

    DEPTH_TOLERANCE_UM = 1.0

    def __init__(self, unit, actions):
        self.unit = unit
        self.actions = actions

    def run(self, phase, *, target_z, speed, tracker=None):
        if not all(math.isfinite(value) for value in (target_z, speed)) or speed <= 0:
            raise ValueError("Spear needs a finite target and positive speed")
        motion = None
        read_z = lambda: float(self.unit.position()[2])
        seen_frames = set()
        missed_detections = 0
        last_good_z = loss_origin_z = None
        max_misses = max(1, int(getattr(phase.config, "track_max_missed_detections", 3)))
        max_unseen_travel = float(getattr(phase.config, "track_max_unseen_travel_um", 3.0))
        try:
            while phase.motion_checkpoint():
                if motion is not None and self.actions.is_complete(motion):
                    phase.controller.info(f"Spear complete: target_z={target_z:.2f}, actual_z={read_z():.2f} um")
                    return phase.observation
                observation = phase.observe()
                if not phase.action_gate():
                    if phase.interrupted:
                        return None
                    raise RuntimeError("Spear is not permitted")
                z = read_z()
                if z >= target_z - self.DEPTH_TOLERANCE_UM:
                    return observation
                if tracker is not None and not getattr(phase, "search_disabled", False):
                    frame = observation.get("image", {}).get("frame_number")
                    fresh = frame is not None and frame not in seen_frames
                    if fresh:
                        seen_frames.add(frame)
                        tracker.update(observation)
                        if not tracker.lost:
                            missed_detections = 0
                            last_good_z, loss_origin_z = z, None
                        else:
                            missed_detections += 1
                            if loss_origin_z is None:
                                loss_origin_z = z if last_good_z is None else last_good_z
                    elif loss_origin_z is None:
                        # A repeated/missing frame supplies no new detection evidence.
                        loss_origin_z = z if last_good_z is None else last_good_z
                    unseen_travel = 0.0 if loss_origin_z is None else abs(z - loss_origin_z)
                    if missed_detections >= max_misses or unseen_travel >= max_unseen_travel:
                        phase.controller.warning(
                            f"Spear cell tracking lost: consecutive_fresh_misses={missed_detections}/{max_misses}; "
                            f"unseen_travel_um={unseen_travel:.3f}/{max_unseen_travel:g}; reacquiring cell")
                        return "cell_lost"
                if motion is None:
                    phase.controller.info(f"Spear: start_z={z:.2f}, target_z={target_z:.2f} um, speed={speed:g} um/s")
                    motion = self.actions.start(self.unit, target=target_z,
                        deadline=time.monotonic() + max(10., (target_z - z) / speed * 20.),
                        read_position=read_z,
                        move_velocity=lambda direction: self.unit.absolute_move_group_velocity([0., 0., -direction * speed]),
                        abort_check=phase.controller.abort_if_requested, label="Spear motion",
                        info=phase.controller.info, direction=1.,
                        reversal_tolerance=self.DEPTH_TOLERANCE_UM, reversal_slack=1e-6)
                phase.controller.sleep(0.01)
        finally:
            if motion is not None:
                self.actions.cancel(motion)
