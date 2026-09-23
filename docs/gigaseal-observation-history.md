# Phase observation decks and calculations

## Shared interface, separate phase histories

- `ObservationHelper` owns separate bounded decks for Gigaseal and BreakIn; each phase resets its history at the start of `run()`.
- The deck stores deep copies of acquired observation dictionaries. New fields automatically appear in retained rows; missing values are scalar NaN, including missing vectors/images. A new run resets field discovery.
- `PhaseController.observation_windows` maps output names to generic field-window contracts. `observe()` records the acquisition once, then adds contracted windows to its returned observation. Derived windows are not copied back into the deck.
- `observation_window(field, width, ...)` supports oldest-first scalar or explicitly shaped vector windows. Contracts specify selection (`predicate`), finite-value filtering, padding, dtype, shape, and whether to include the current observation. Reading a window does not acquire, average, consume, or mutate measurements.
- Agent preparation reads model input dimensions through `AgentHelper.observation_input_width(input_name)`. The controller's old `_agent_resistance_input_width` and `_attach_agent_resistance_input` methods are removed.

For example, another numeric field can use the same observation system:

```python
self.observation_windows["capacitance_window"] = {
    "field": "capacitance",
    "width": 10,
    "include_current": True,
}
```

## Phase contracts

- Both Agent contracts expose `resistance_input` as prior finite resistance values, oldest first, float32, left-zero-padded to the model width. Current readings remain separate. Model frame stacking is unchanged.
- Gigaseal selects observations with a finite pressure-state value, excluding baseline, loop-average, and pressure-release observations. Its capacity is max(30, 2*N + 1), covering normal interleaving of loop averages and Agent observations.
- BreakIn selects observations whose access-resistance result does not qualify for its action gate, matching its existing decision eligibility. Its capacity is max(30, N + 1). Good access checks remain recorded but do not enter the Agent window.
- Training also records observations, while preserving its existing completion and action behavior.

## calculate before decide

- `PhaseController.calculate(observation, state)` is a default pass-through hook. It performs phase calculations without acquiring measurements, invoking inference, or issuing hardware commands.
- Gigaseal moves resistance delta, slope, progress-time updates and Classic/Adaptive thresholds into `calculate()`. For Agent observations it adds `observations_since_last_action`. Calls retain the original placement of progress updates before the separate Agent acquisition.
- BreakIn calls `calculate()` after success/action gates pass. It updates trial count, wait duration and periodic pulse speed, and prepares the Agent observation's fields and resistance dtype. `decide()` consumes these calculated inputs and selects commands.
- Sampling, averaging, hardware-command order and the current BreakIn acquisition sequence are preserved. The pre-refactor committed BreakIn sequence is not restored by this change.

## Direct acquisition and calculation records

Electrical getters validate numeric `num_measurements` and `interval` arguments,
then call `resistanceRamp`, `accessRamp`, or `capacitanceRamp` directly. Observation
selection uses explicit getter calls in the requested acquisition order. Legacy
list/dictionary output and scalar/one-element-array formats are preserved.

Phases pass numeric sampling arguments through `observe()`. Omitted arguments keep
each electrical measurement's existing defaults. The older `sampling` mapping is
accepted for compatibility but cannot be combined with numeric sampling arguments.
Gigaseal reads the live `measurement_speed` before each averaged loop observation
and uses that same interval for its slope calculation. Retests also read the current
configured interval.

Gigaseal records resistance changes, slope, applicable thresholds, and their inputs
against the observation used for the calculation. Agent action-count calculations
are stored against their own observation. BreakIn records calculated trial count,
wait duration, and pulse speed before constructing its Agent input. Recording enriches
the existing copied deck row; it does not acquire again or append another sample.
Calculation metadata is excluded from Agent payloads, which retain numeric inputs.

This simplification was checked by source review and syntax compilation only. No
hardware, model inference, or cell acquisition was executed for this change.

## Deferred critiques and limits

- A deck window counts observations, not elapsed time or unique DAQ acquisitions. DAQ getters read cached estimates; separate electrical averages can describe different times.
- Bounded retention can evict older eligible/finite readings during long invalid bursts or many excluded observations. Contracts use retained data and their configured padding; they cannot recover evicted values.
- Gigaseal's pressure-state selection and BreakIn's access-gate selection reflect current phase observation semantics. New observation roles may require updating those phase contracts.
- Deep-copying frames adds memory and latency. Entry count is bounded, but live-rig memory and latency budgets remain unmeasured.
- Future VLM consumers and acquisition-quality metadata are outside this change.

## Validation

Hardware-free tests cover deck ownership/reset, dynamic NaN fields, scalar/vector windows, padding, prior/current selection, model-width resolution, phase calculations, gate eligibility and existing Agent actions. Existing mode checks are included. No live-rig test was run.
