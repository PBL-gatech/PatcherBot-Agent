"""Common interface for phases that share an AutoPatcher controller."""

from abc import ABC, abstractmethod
import collections


class PhaseController(ABC):
    def __init__(self, controller):
        """Store shared dependencies; attempt setup belongs at the start of run()."""
        self.controller = controller
        self.observation_windows = {}
        self.observation_deck = collections.deque()

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

    @abstractmethod
    def success_gate(self, observation=None, state=None):
        """Evaluate this phase's completion condition."""
        raise NotImplementedError

    def observation_window(self, field, width, **options):
        """Delegate this phase's numeric history window to the shared helper."""
        return self.controller.observation_helper.observation_window(self, field, width, **options)

    def observe(self, include_pressure_state: bool = False, *, fields=None, sampling=None, raw_measurements=False,
                deep_learning=False, cell=None, num_measurements=None, interval=None):
        """Read through controller overrides and record into helper-owned history."""
        if num_measurements is not None or interval is not None:
            observation = self.controller.observe(
                include_pressure_state, fields=fields, sampling=sampling,
                raw_measurements=raw_measurements, deep_learning=deep_learning, cell=cell,
                num_measurements=num_measurements, interval=interval,
            )
        elif deep_learning or cell is not None:
            observation = self.controller.observe(
                include_pressure_state, fields=fields, sampling=sampling,
                raw_measurements=raw_measurements, deep_learning=deep_learning, cell=cell,
            )
        elif raw_measurements:
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
        self.controller.observation_helper.record_observation(self, observation)
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
