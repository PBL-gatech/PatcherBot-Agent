"""Common interface for phases that share an AutoPatcher controller."""

from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np


class PhaseController(ABC):
    def __init__(self, controller):
        self.controller = controller
        self.observation_deck = None
        self.observation_windows = {}

    @abstractmethod
    def run(self, cell=None):
        """Execute this phase; concrete phases retain their input contracts."""
        raise NotImplementedError

    def prepare(self, state=None):
        """Prepare a phase attempt; override when preparation is needed."""
        pass

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

    @abstractmethod
    def success_gate(self, observation=None, state=None):
        """Evaluate this phase's completion condition."""
        raise NotImplementedError

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
        if width == 0:
            return result
        samples = list(self.observation_deck or ())
        if not include_current:
            samples = samples[:-1]
        values = []
        for sample in samples:
            if not isinstance(sample, dict) or (predicate is not None and not predicate(sample)):
                continue
            try:
                value = np.asarray(sample.get(field, np.nan), dtype=dtype).reshape(shape)
            except (TypeError, ValueError):
                value = np.full(shape, np.nan, dtype=dtype)
            if finite_only and not np.isfinite(value).all():
                continue
            values.append(value)
        values = values[-width:]
        if values:
            result[-len(values):] = np.asarray(values, dtype=dtype)
        return result

    def observe(self, include_pressure_state: bool = False, *, fields=None, sampling=None, raw_measurements=False):
        """Read through controller overrides and snapshot into an enabled phase deque."""
        if raw_measurements:
            observation = self.controller.observe(
                include_pressure_state, fields=fields, sampling=sampling,
                raw_measurements=True,
            )
        elif fields is None and sampling is None:
            if include_pressure_state:
                observation = self.controller.observe(include_pressure_state=True)
            else:
                observation = self.controller.observe()
        else:
            observation = self.controller.observe(
                include_pressure_state, fields=fields, sampling=sampling
            )
        if self.observation_deck is not None:
            snapshot = deepcopy(observation)
            if isinstance(snapshot, dict):
                # Learn fields from this run; missing measurements are never
                # carried forward from an earlier observation.
                previous = [entry for entry in self.observation_deck if isinstance(entry, dict)]
                fields = set(snapshot)
                for entry in previous:
                    fields.update(entry)
                for entry in [*previous, snapshot]:
                    for field in fields:
                        entry.setdefault(field, float("nan"))
            self.observation_deck.append(snapshot)
        if isinstance(observation, dict):
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
