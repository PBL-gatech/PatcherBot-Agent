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
                if tracker is not None:
                    tracker.update(observation)
                    if tracker.lost and not tracker.using_fallback:
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
