"""Common interface for phases that share an AutoPatcher controller."""

from abc import ABC, abstractmethod


class PhaseController(ABC):
    def __init__(self, controller):
        """Store shared dependencies; attempt setup belongs at the start of run()."""
        self.controller = controller
        self.observation_window = 0
        self.observation_count = 0
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
        self.observation_count = 0
        self.goal_event = False
        self.awaiting_operator = False

    def enrich_observation(self, observation):
        return observation

    def observe(self, *, pipette_inference=False, cell_inference=False, cell_masks=False,
                num_measurements=None, interval=None, raw_resistance=False, **options):
        observation = self.controller.observe(pipette_inference=pipette_inference,
            cell_inference=cell_inference, cell_masks=cell_masks, num_measurements=num_measurements,
            interval=interval, raw_resistance=raw_resistance, **options)
        self.observation_count += 1
        return self.enrich_observation(observation)

    def complete_success(self):
        """Signal completion through the controller's request mechanism."""
        self.controller.success_requested = True
        return self.controller.success_if_requested()

    def wait_for_manual_completion(self):
        """Wait at a manual-confirmation point while allowing task interrupts."""
        while True:
            self.controller.sleep(0.1)
