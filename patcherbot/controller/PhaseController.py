"""Common interface for phases that share an AutoPatcher controller."""

from abc import ABC, abstractmethod
from uuid import uuid4
import numpy as np


class PhaseController(ABC):
    def __init__(self, controller):
        """Store shared dependencies; attempt setup belongs at the start of run()."""
        self.controller = controller
        self.observation_windows = {}
        self.frame_context = {}
        self.observation_run = None
        self._observation_attempt = None
        self.goal_event = False
        self.awaiting_operator = False

    @abstractmethod
    def run(self, cell=None):
        """Execute this phase; concrete phases retain their input contracts."""
        raise NotImplementedError

    def action_gate(self, observation=None, state=None):
        """Determine whether the observation permits an action."""
        pass

    def failure_gate(self, state=None):
        """Handle a phase's failure condition."""
        pass

    def calculate(self, observation=None, state=None):
        """Prepare phase calculations for decide without acquiring or acting."""
        return observation

    def decide(self, observation=None, state=None):
        """Choose a command based on the observation for act."""
        pass

    @abstractmethod
    def act(self, observation=None, state=None):
        """Apply this phase's actions using its observation and decide's command."""
        raise NotImplementedError

    def success_gate(self, observation=None, state=None):
        """Apply the shared completion policy to a phase's Boolean goal."""
        self.controller.success_if_requested()
        if observation is None or self.awaiting_operator:
            return False
        self.goal_event = bool(observation)
        if not self.goal_event:
            return False
        if self.controller.config.mode == "Training":
            self.awaiting_operator = True
            self.controller.info("Training mode: goal reached. Click Success or Abort to finish.")
            return False
        return True

    def begin_observations(self):
        self.observation_run = uuid4().hex
        self._observation_attempt = self.controller.observer.attempt_token
        self.goal_event = False
        self.awaiting_operator = False

    def enrich_observation(self, observation):
        return observation

    def observation_window(
        self, field, width, *, predicate=None, finite_only=False,
        fill_value=np.nan, dtype=np.float32, shape=(), include_current=False,
    ):
        """Read a configured numeric field window from this phase's deck.

        Values are oldest first, left-padded, and exclude the current snapshot
        unless requested. Scalar and explicitly shaped vector fields use the
        same reader. Selection and formatting never mutate stored observations.
        """
        width, shape = int(width), tuple(shape)
        result = np.full((width, *shape), fill_value, dtype=dtype)
        if width == 0 or self._observation_attempt is not self.controller.observer.attempt_token:
            return result
        samples = [row for row in self.controller.observer.deck
                   if row.get("phase_run") == self.observation_run]
        if not include_current:
            samples = samples[:-1]
        values = []
        for sample in samples:
            if not isinstance(sample, dict) or (predicate is not None and not predicate(sample)):
                continue
            try:
                value = sample
                for key in field.split("."):
                    value = value.get(key, np.nan) if isinstance(value, dict) else np.nan
                value = np.asarray(value, dtype=dtype).reshape(shape)
            except (TypeError, ValueError):
                value = np.full(shape, np.nan, dtype=dtype)
            if finite_only and not np.isfinite(value).all():
                continue
            values.append(value)
        values = values[-width:]
        if values:
            result[-len(values):] = np.asarray(values, dtype=dtype)
        return result

    def observe(self, *, fields=None, num_measurements=None, interval=None, evidence=None,
                raw_resistance=False):
        if self._observation_attempt is not self.controller.observer.attempt_token:
            self.begin_observations()
        options = {"raw_resistance": True} if raw_resistance else {}
        observation = self.controller.observe(fields=fields, num_measurements=num_measurements,
                                              interval=interval, phase=self, evidence=evidence, **options)
        acquired_fields = set(observation)
        self.enrich_observation(observation)
        derived = {key: value for key, value in observation.items() if key not in acquired_fields}
        if derived:
            self.controller.observer.annotate(self, observation, fields=derived)
        for output, contract in self.observation_windows.items():
            predicate = contract.get("predicate")
            if predicate is None or predicate(observation):
                observation[output] = self.observation_window(**contract)
        return observation

    def complete_success(self):
        """Signal completion through the controller's request mechanism."""
        self.controller.success_requested = True
        return self.controller.success_if_requested()

    def wait_for_manual_completion(self):
        """Wait at a manual-confirmation point while allowing task interrupts."""
        while True:
            self.controller.sleep(0.1)
