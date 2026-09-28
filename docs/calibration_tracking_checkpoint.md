# Calibration-assisted tracking and supervised Hunt: implementation audit

The software rebuild is implemented. Offline checks exercise the real Hunt supervisor, shared tracking, asynchronous inference, rendering, and simulated movement backends. Physical stop latency and recording-based spatial accuracy remain unvalidated.

## Ownership and algorithm

- ObservationHelper owns one persistent ObservationTracking instance per controller/pipette. Home and home-stage positions remain the anchor. The tracker predicts tip XY from their calibrated displacement and predicts the selected cell from its stage reference. Separate local residual corrections and uncertainty are retained for the two tracks.
- Fresh, pose-consistent detector measurements correct those residuals once. Confidence and innovation gates reject outliers; missing confidence remains unknown. Raw detector fields remain available. Defocus prediction and cell-plane separation remain unavailable; measured focus is separate.
- Hunt repeatedly acquires one observation, updates its quantities, evaluates contact and limits, evaluates the active mode/state, dispatches a command, and sleeps 40 ms. It records actual observation intervals.
- Hunt alone commits S-state transitions. Each decision carries its state ID, reason, and evidence reference. A proposed transition stops movement, retains the state context, and requires applicable settled/fresh confirmation. Contact holds preserve travel budgets and best measurements.
- Positional moves and velocity commands use explicit supervised dispatch, polling, and cancellation. Agent inference has one outstanding request and discards invalidated results.
- Camera/livefeed render a matching cached frame estimate; they do not read hardware. Prediction-only markers, uncertainty, measurement age, raw/calibration toggles, and offscreen arrows live in presentation code.

## Hunt method audit

Final Hunt size: **763 lines, 19 methods**, compared with 1,546 lines and 32 methods at the start of this rebuild.

| Method | Start-end lines |
|---|---|
| `__init__` | 19-25 |
| `run` | 27-137 |
| `prepare` | 139-195 |
| `calculate` | 197-250 |
| `decide` | 252-377 |
| `act` | 379-414 |
| `action_gate` | 416-476 |
| `spear` | 478-514 |
| `search` | 516-532 |
| `scan` | 534-565 |
| `shift` | 567-604 |
| `failure_gate` | 606-611 |
| `success_gate` | 613-621 |
| `_recover_contact_visuals` | 623-683 |
| `observe` | 685-705 |
| `_adaptive_contact_confirmed` | 707-737 |
| `_resistance_above_threshold` | 739-749 |
| `_resistance_threshold_reached` | 751-755 |
| `_isCellDetected` | 757-762 |

The three added Hunt-specific methods separate immediate electrical stopping, Adaptive visual confirmation, and single-observation microscope recovery. The original compatibility method `_isCellDetected` has a real caller in `controller/patch.py:350-353`; it is not a retained private forwarding name with no caller.

The six estimator/evidence additions were consolidated in observation infrastructure; two marker additions moved to presentation; five lifecycle additions were absorbed by the existing observation/supervision/state methods. ObservationTracking has no hardware movement commands, mode sequencing, or phase lifecycle.

## Edit audit

These are current line ranges for the changed areas, not a claim that every line inside each range is new. Unrelated pre-existing working-tree changes are retained.

| File | Start-end lines | Change / requirement |
|---|---|---|
| `patcherbot/controller/phases/hunt_cell.py` | 1-763 | One continuously supervised loop, persistent states, guarded transitions and decisions, distinct mode contracts, supervised movement, async Agent, raw contact confirmation. |
| `patcherbot/devices/manipulator/helpers/ObservationTracking.py` | 1-487 | Shared home/stage prediction, residual fusion, uncertainty, association, timing checks, frame projection and invalidation. |
| `patcherbot/devices/manipulator/helpers/ObservationHelper.py` | 7-20; 46-121; 533-554; 605-610; 644-646; 676-679; 802-804 | Persistent ownership/cache, target context, one-pass detailed heads, tracking fields and independent focuser compatibility. |
| `patcherbot/controller/PhaseController.py` | 49-57 | Forward target_cell independently of learned tracker cell. |
| `patcherbot/controller/patch.py` | 805-824; 948 | Shared observation forwarding, explicit anchor invalidation hook and settled Locate sample. |
| `patcherbot/controller/phases/approach_cell.py` | 49 | Settled Approach sample through shared observation helper. |
| `patcherbot/interface/pipettes.py` | 170-179 | Validate both six-coordinate records before loading; split XYZ/XYZ for home and safe. |
| `patcherbot/devices/manipulator/helpers/AgentHelper.py` | 3-4; 29-34; 77; 271-344 | Single outstanding asynchronous request, generation invalidation, locked model calls, preserved synchronous interface. |
| `patcherbot/devices/manipulator/manipulatorunit.py` | 49-75; 144-160; 186-192 | Positional and velocity start/poll wrappers; cancel before stopping. |
| `patcherbot/devices/manipulator/microscope.py` | 11; 62-85; 99-115; 126-130; 179-184 | Equivalent microscope API, correct axis dispatch and cancellation. |
| `patcherbot/devices/manipulator/scientificaSerial.py` | 194-195; 319-416; 501-526; 555-558; 599-600; 720-725; 831-834 | Pollable bounded encoder correction, explicit velocity failure propagation, cancellation serialized with dispatch; NoEncoder reuse. |
| `patcherbot/devices/manipulator/sensapexWrapper.py` | 66-67; 213-268; 334-359; 377-379; 459-473; 492-526 | No prior-move wait on supervised start; poll SDK failures; generations prevent commands after cancellation and stale velocity errors. |
| `patcherbot/devices/manipulator/fakemanipulator.py` | 13; 26-28; 130-165; 265-280; 299-314; 322-332 | Time-spanning supervised fake movement, finite zero/change velocities and stop-all behavior. |
| `patcherbot/gui/livefeed.py` | 20; 35; 156-162 | Pass displayed frame ID/time/image shape to optional overlay callback. |
| `patcherbot/gui/camera.py` | 32; 572; 581; 763-790; 890-891 | Main-camera tracking callback and Ctrl+Alt+R/P raw/calibration toggles. |
| `patcherbot/gui/tracking_overlay.py` | 1-159 | Fused/predicted/raw markers, uncertainty ellipse, ages, defocus, offscreen arrows and stale handling. |

Tests changed or added:

| File under `regression_tests/` | Start-end lines | Coverage |
|---|---|---|
| `test_agent_async.py` | 1-179 | Single outstanding request, snapshot isolation, epoch invalidation, concurrency and failures. |
| `test_hunt_supervisor.py` | 1-170 | Persistent state, allowed/fresh transitions, duplicates, stop on failed velocity, electrical baseline, method/loop audit. |
| `test_hunt_supervision_motion.py` | 1-299 | Movement spanning observations, first contact/abort/invalid resistance cancellation, ongoing travel bounds and recovery return. |
| `test_supervised_movement.py` | 1-491 | 28 wrapper/backend tests including velocity dispatch and asynchronously reported SDK failures. |
| `test_observation_tracking.py` | 1-386 | Affine signs/scales, drift/dropout/outliers, confidence, timing, invalidation, frame projection and cell bounds. |
| `test_observation_tracking_ownership.py` | 1-140 | Single-pass heads, independent focusers, raw fields, saved-anchor round trips and ownership. |
| `test_hunt_tracking_markers.py` | 1-182 | 17 frame-aware overlay geometry/style/age/uncertainty tests and existing preprocessing behavior. |
| `test_hunt_contact_recovery.py` | 1-185 | Nine recovery scenarios exercised through run(), including resistance polling while returning. |
| `test_hunt_operation_confirmation.py` | 1-185 | Search/Scan/Shift stop, arrival, fresh confirmation and failure scenarios through run(). |

Migrated existing test areas: `test_hunt_observations.py` 63-73, 102-128, 170-176, 227-241, 257-318, 452-496, 537-646, 813-842, 862-999; `test_hunt_adaptive_contact.py` 10-18 and 32-44; `test_hunt_cell_association.py` 70-99 and 126-138; `test_hunt_contact_confirmation.py` 52-61; `test_hunt_spear_checkpoint.py` 64-119; `test_hunt_spear_tracking.py` 29-73; `test_hunt_training_monitor.py` 24-63.

Existing lifecycle, Training, electrical contact and resistance replay regressions were included in the complete suite. Migrated S-state tests execute the actual supervisor instead of reconstructing the removed nested loops; they retain the original safety and progress assertions.

## Offline validation

- **239 tests passed**, no failures, errors or skips, using unittest discovery from a temporary working directory. [Final test log](C:/Users/SA-FOR~1/AppData/Local/Temp/patcherbot_full_validation_rqlxd3nr/regressions.log).
- **553 Python files parsed and compiled**. Flat temporary bytecode destinations avoid Windows path-length failures in vendored DINOv2 directories.
- Refreshed source-only wheel built successfully; archive integrity and new tracking/Hunt/render modules were checked. Application-order imports pass from its extracted installation. [Final build report](C:/Users/SA-FOR~1/AppData/Local/Temp/patcherbot_build_validation_dwey07x5/final_verified.json).
- **1,100 seeded backend scenarios**: 600 encoder cases (433 settled, 51 expected bounded retry failures, 116 cancelled), 300 Sensapex coordinate/extra-axis cases, and 200 handle/cancel cases. [Stress results](C:/Users/SA-FOR~1/AppData/Local/Temp/patcherbot-motion-audit-vy_72bn3/results.txt).
- **480 repeated asynchronous Agent checks** passed, and a real offscreen Qt overlay was rendered and inspected. [GUI/Agent artifacts](C:/Users/SA-FOR~1/AppData/Local/Temp/patcherbot-overlay-smoke-se_1awta).
- No new noise fit or spatial accuracy score is claimed from these synthetic and control regressions.

No physical movement, real model loading, package installation, or network download was required for these checks. The build is a source-only Python wheel check, not a packaged-model deployment test.

## Remaining validation and explicit boundaries

- Uncertainty settings remain provisional. Raw recordings exist under `experiments/Data/rig_recorder_data`, but matching calibration and independent manually identified tips have not been established. The requested detector-only/calibration-only/fused accuracy comparison and noise fitting on held-out recordings are still pending that evidence.
- Predicted physical depth stays unavailable. The calibration's identity Z row is not used as verified defocus or cell-plane separation.
- Replacement/re-zeroing requires calling `AutoPatcher.invalidate_tracking_anchor(reason, stage=False)`. No existing physical replacement/re-zero command was found to wire automatically. Calibration and camera geometry changes invalidate/reset the cached estimate through their identity.
- Serial/SDK requests themselves remain synchronous and subject to transport timeout. Motion-completion waits were removed from Hunt's command path; real device response and stop latency still need measurement.
- The repository has an existing circular import when Hunt is imported before interface initialization. The same isolated check fails at HEAD. The normal interface-first import sequence succeeds for both HEAD and the rebuilt package; that unrelated package initialization was preserved.
