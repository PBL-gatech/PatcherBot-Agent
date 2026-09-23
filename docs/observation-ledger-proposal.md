# Proposal: shared observation history

**Status:** Proposed; implementation deferred at the user's request.
**Date:** 2026-09-15
**Scope:** Shared observation acquisition, with gigaseal as the first integration point.

## Execution brief

- **Premise:** Gigaseal and BreakIn currently manage Agent resistance history locally. The proposed shared ledger would retain recent observations for every metric, with a default history of 30 entries per metric.
- **Problem:** History collection, model-window sizing, and phase logic are coupled. Generalizing history must preserve the observation contract and the meaning of existing model inputs.
- **Directives:** Keep the current application unchanged for now. On a future implementation, preserve the original gigaseal snapshot, existing hardware-command order, sampling, deadlines, controller overrides, and mode behavior. Avoid extra phase helpers and unrelated changes. Do not introduce a CLI or argparse.
- **Requirements:** Separate retained history from the model's requested view. Let Agent preparation select and validate window sizes. Bound memory, preserve observation return values, and record only measurements actually acquired.
- **Objective:** Provide a shared, bounded history that phases and models can consume without duplicating acquisition or embedding decision logic in observe.
- **Acceptance checks:** Listed below; implementation is not authorized by this document.
- **Working assumption:** An observation means a value returned by observe, which may already be an average. Recording every raw DAQ sample or changing the sampling rate is outside this proposal unless explicitly selected later.

## Current behavior and evidence

Source locations were checked on 2026-09-15; method names identify the relevant code if line numbers later move.

| Area | Existing behavior | Source |
|---|---|---|
| Phase observation contract | Delegates to the controller, preserving overrides and observation options. | [PhaseController.py](../patcherbot/controller/PhaseController.py), lines 41-54 |
| Selected observations | Reads requested fields with explicit sampling options; preserves electrical scalars when requested. | [patch.py](../patcherbot/controller/patch.py), lines 799-891 |
| Legacy observations | Returns a list without pressure state and a dictionary with pressure state. | [patch.py](../patcherbot/controller/patch.py), lines 893-932 |
| Gigaseal acquisition | Uses five-reading averages for its baseline, loop, and pressure retest, plus a separate Agent observation. | [gigaseal.py](../patcherbot/controller/phases/gigaseal.py), lines 43-65 and 92-102; [patch.py](../patcherbot/controller/patch.py), line 912 |
| Agent window size | Uses the model's resistance_input shape when available, otherwise a default width of 15; returns zero when the input is unused. | [patch.py](../patcherbot/controller/patch.py), lines 993-1022 |
| Agent history semantics | Builds a zero-padded input from prior readings, then appends the finite current reading. | [patch.py](../patcherbot/controller/patch.py), lines 1024-1040 |
| Camera ownership | Returns the queued frame directly rather than copying it. | [patch.py](../patcherbot/controller/patch.py), lines 858-861 and 897-926 |

## Proposed responsibilities

### Shared observation ledger

- One shared owner per controller, with bounded histories and a default retention target of **30 observations per metric**.
- Append each acquired metric once. Unrequested metrics receive no new entries, and the ledger does not trigger acquisition to fill gaps.
- Initial coverage means metrics acquired through observe. Direct acquisitions elsewhere, such as [hunt_cell.py](../patcherbot/controller/phases/hunt_cell.py), line 35, currently bypass that boundary. Covering every acquisition throughout the application would require a separately reviewed expansion.
- Keep the current observe signature, returned values, shapes, and scalar/list/dictionary conventions. Any normalization for storage remains internal.
- Preserve controller overrides. The recording boundary must see the observation actually returned by an override; hooking only the default controller implementation would leave overridden observations unrecorded. Select one recording boundary to avoid double appends through phase/controller delegation.
- Each entry should identify its value, metric, monotonic acquisition timestamp or sampling interval, observation identity, attempt, and sampling provenance. Reuse batch metadata where appropriate. Exact storage representation remains an implementation choice.
- For averaged readings, document whether the timestamp denotes the beginning or completion of the window; storing both would make the averaging interval explicit.
- Histories are not synchronized merely because their lengths match. Consumers must use timestamps or observation identities when aligning metrics.
- The ledger performs acquisition bookkeeping only. Slope calculations, progress timers, mode decisions, success conditions, and hardware actions stay in their respective phase responsibilities.

A bounded deque is a suitable candidate: appends are approximately constant-time and its maximum length bounds retention. [Python deque documentation](https://docs.python.org/3/library/collections.html#collections.deque).

### Agent prepare

- Resolve each requested metric's window width from the model contract or explicit configuration once during prepare. Preparation validates the model's expected dimensions; it does not choose an arbitrary incompatible width.
- Record the selected metric streams, oldest-to-newest ordering, dtype, missing-history policy, and whether the current observation is included.
- Ensure capacity is sufficient before acquisition begins. If the ledger already contains the current reading and a model requires N prior readings, the selected stream needs at least N + 1 retained entries. A default of 30 is not a hard cap for larger model requirements.
- Do not resize histories on every inference call. Growing a buffer later cannot recover already-evicted samples.
- Preserve per-attempt model history. A new attempt must not silently consume the previous attempt's readings; use a reset or an attempt-filtered view.

### Agent decide

- Read the configured window without acquiring more measurements or maintaining another phase-owned deque.
- Preserve the existing resistance_input convention: prior readings only, finite values, oldest-to-newest ordering, float32 formatting, and zero padding while history is genuinely incomplete.
- Obtain a stable view of the required entries for one decision. Reading a window must not consume or reorder the ledger.
- Continue using the existing current-observation inputs alongside historical inputs.

## Sampling semantics: decision required before implementation

A five-sample resistance average and a one-sample Agent resistance reading must not silently become interchangeable history entries. Baselines and pressure-release retests also have different roles in the existing algorithm.

Two designs remain possible:

1. **Retain distinguishable streams within each metric.** Give each sampling profile/role sufficient bounded retention, and let prepare select the model-compatible stream. This is the preferred compatibility option, but makes the 30-entry default apply to a metric stream and increases total retention when several streams exist.
2. **Retain one mixed deque per metric.** Tag entries and filter model windows. Capacity must account for intervening ineligible observations; max(30, N + 1) alone does not guarantee N eligible prior readings. Evicted history must not be disguised as ordinary startup padding.

A third option, deliberately unifying sampling for all consumers, would change the algorithm and model inputs. It requires a separate decision and validation plan.

## Images and mutable values

The all-metrics objective includes camera_image, but its retention policy remains unresolved. Numeric measurements and position vectors are small; full image payloads dominate the measured memory use below.

- Copy small mutable vectors when necessary so later mutation cannot rewrite history.
- Retain image references only if frame ownership and immutability are established. The current direct frame return does not establish that guarantee.
- Alternatives include owned full-frame copies, a separately bounded shared frame store referenced by ledger entries, or model-compatible reduced representations.
- Do not silently downsample images, change their dtype, or exclude image history. Decide the policy before an implementation claiming coverage of all metrics.

## Measured overhead

A standalone synthetic benchmark was run locally with Python 3.11.15 and NumPy 2.4.6. It used six scalar metrics and three three-element position vectors, capacity 30, and five timed repeats. Numeric timings used 20,000 operations per repeat; image timings used 100.

| Operation | Median measured cost |
|---|---:|
| Append nine metric references | 0.67 microseconds |
| Append nine timestamped metrics, copying small arrays | 1.84 microseconds |
| Materialize 10-entry NumPy windows for all nine metrics | 15.38 microseconds |
| Materialize 30-entry NumPy windows for all nine metrics | 29.22 microseconds |
| Retained memory for the populated numeric ledger | 44,004 bytes, approximately 43 KiB |
| Copy and append one 1024 x 1024 uint16 grayscale frame | 0.34 milliseconds |
| Payload for 30 such frames | 60 MiB |
| Copy and append one 2048 x 2048 uint16 grayscale frame | 1.51 milliseconds |
| Payload for 30 such frames | 240 MiB |

Evidence: [benchmark script](../experiments/benchmark_observation_ledger.py), lines 1-100; [recorded results](../experiments/benchmark_observation_ledger_results.json), lines 1-61.

**Interpretation:** the measured numeric append plus full-window conversion totals about 0.031 milliseconds. This supports proceeding with numeric-ledger design when work resumes. Image storage requires an explicit memory/copy budget.

**Limits:** these are batch-average microbenchmark medians, not live-rig latency or worst-case guarantees. The benchmark includes a timestamp and small-array copies, but not the full proposed provenance metadata, filtering, locking, image stacking, GPU transfer, inference, or device acquisition. The retained-memory estimate describes benchmark objects rather than whole-process memory. Adding acquisitions to fill histories could dominate the bookkeeping cost and is not proposed.

## Future implementation sequence

Implementation remains deferred. If resumed:

1. Resolve the sampling-stream, image-retention, and raw-versus-averaged observation decisions above.
2. Specify the ledger entry and read-window contract, capacity rules, invalid-value policy, and per-attempt lifecycle.
3. Add ledger recording at one shared observation boundary while preserving overrides and all existing return contracts.
4. Migrate gigaseal's Agent history selection into prepare and history consumption into decide. Remove its redundant local resistance deque only after equivalent model inputs are demonstrated.
5. Migrate BreakIn and other consumers in bounded steps after gigaseal verification.
6. Measure integrated latency and memory on representative workloads before making performance claims about the live rig.

## Acceptance checks

- Default retention is 30 per metric under the agreed stream policy; memory remains bounded during long runs.
- Agent window widths are resolved once in prepare and match model input shapes, including widths above 30.
- Prior-only versus current-inclusive windows, startup padding, invalid samples, ordering, and dtype match the selected contract.
- Mixed sampling profiles cannot silently change model input distributions or evict required history unnoticed.
- Only acquired metrics are appended; a resistance-only observation does not read cameras, pressure, or other devices.
- Default controller observations and custom overrides are recorded exactly once and retain their existing return forms.
- Entries remain stable when source arrays or camera buffers change, according to the agreed ownership policy.
- New attempts do not inherit stale model history unless explicitly configured.
- Concurrent recording and window access, if used, provide a coherent view without holding locks during device I/O or inference.
- Gigaseal preserves device-command order, measurement cadence, pressure-release retests, deadline checkpoint placement, holding transitions, success filtering, and Training completion behavior.
- Compare against the preserved [original gigaseal snapshot](../snapshots/gigaseal_original_20260915_113125_790.py), lines 1-295, and extend the existing snapshot regression tests to compare model input arrays as well as acquisition/action traces.
- Run the focused phase, observation, and regression suites. Measure integrated overhead separately from acquisition and inference; establish an acceptable latency budget before rollout.

## Decisions left open

- One mixed history per metric or separate sampling-profile/role streams?
- Observe outputs only, or an explicitly requested raw-sample ledger?
- Full camera frames, owned references, or a model-compatible alternative?
- Reset per attempt or retain a bounded cross-attempt ledger with isolated model views?
- Record invalid values with validity metadata, or omit them from storage while preserving model filtering?
- Which shared recording boundary covers every intended caller and controller override without duplicate entries?

These are recorded for later work. No application change, rollout, or further refactor is requested now.
