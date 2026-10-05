import threading


class RecordingStateManager:
    STATE_PRESS_LABELS = {
        "locate_cell": "Locate Cell",
        "approach_cell": "Approach Cell",
        "hunt_cell": "Hunt Cell",
        "gigaseal": "Gigaseal",
        "break_in": "Break-in",
        "escape": "Escape Cell",
        "patch": "Patch Cell",
    }

    def __init__(self):
        self._recording_enabled = False
        self._lock = threading.Lock()
        self.sample_number = 0
        self.testMode = False
        self._state_press_counts = {
            state: 0 for state in self.STATE_PRESS_LABELS
        }
    
    def increment_sample_number(self):
        with self._lock:
            self.sample_number += 1
            print("Sample number incremented to:", self.sample_number)

    def increment_state_press(self, state):
        if not isinstance(state, str) or state not in self.STATE_PRESS_LABELS:
            raise ValueError(f"Unsupported patch state: {state!r}")
        with self._lock:
            self._state_press_counts[state] += 1
            return dict(self._state_press_counts)

    def reset_state_press_counts(self):
        with self._lock:
            for state in self._state_press_counts:
                self._state_press_counts[state] = 0
            return dict(self._state_press_counts)

    def get_state_press_counts(self):
        with self._lock:
            return dict(self._state_press_counts)

    def toggle_recording(self):
        with self._lock:
            self._recording_enabled = not self._recording_enabled
            # print("Recording state toggled to:", self._recording_enabled)
            return self._recording_enabled

    def set_recording(self, state: bool) -> None:
        with self._lock:
            self._recording_enabled = state
            # print("Recording state set to:", self._recording_enabled)

    def is_recording_enabled(self) -> bool:
        with self._lock:
            return self._recording_enabled
