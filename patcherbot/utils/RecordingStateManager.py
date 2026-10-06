import threading


class RecordingStateManager:
    """Thread-safe manager for controlling recording state and tracking sample numbers."""
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
        """Initialize the RecordingStateManager."""
        self._recording_enabled = False
        self._lock = threading.Lock()
        self.sample_number = 0
        self.testMode = False
        self._state_press_counts = {
            state: 0 for state in self.STATE_PRESS_LABELS
        }
    
    def increment_sample_number(self):
        """Increment the sample number in a thread-safe manner."""
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
        """
        Toggle the recording state between enabled and disabled.

        Returns:
            bool: The new recording state.
        """
        with self._lock:
            self._recording_enabled = not self._recording_enabled
            # print("Recording state toggled to:", self._recording_enabled)
            return self._recording_enabled

    def set_recording(self, state: bool) -> None:
        """
        Set the recording state explicitly.

        Args:
            state (bool): True to enable recording, False to disable.
        """
        with self._lock:
            self._recording_enabled = state
            # print("Recording state set to:", self._recording_enabled)

    def is_recording_enabled(self) -> bool:
        """
        Check if recording is currently enabled.

        Returns:
            bool: True if recording is enabled, False otherwise.
        """
        with self._lock:
            return self._recording_enabled
