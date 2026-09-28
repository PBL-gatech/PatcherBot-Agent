# Calibration-assisted tracking and supervised Hunt: implementation audit

The software rebuild is implemented. Offline checks exercise the real Hunt supervisor, shared tracking, asynchronous inference, rendering, and simulated movement backends. Physical stop latency and recording-based spatial accuracy remain unvalidated.

## Ownership and algorithm

- ObservationHelper owns one persistent ObservationTracking instance per controller/pipette. Home and home-stage positions remain the anchor. The tracker predicts tip XY from their calibrated displacement and predicts the selected cell from its stage reference. Separate local residual corrections and uncertainty are retained for the two tracks.
- Fresh, pose-consistent detector measurements correct those residuals once. Confidence and innovation gates reject outliers; missing confidence remains unknown. Raw detector fields remain available. Defocus prediction and cell-plane separation remain unavailable; measured focus is separate.
- Hunt repeatedly acquires one observation, updates its quantities, evaluates contact and limits, evaluates the active mode/state, dispatches a command, and sleeps 40 ms. It records actual observation intervals.
- Hunt alone commits S-state transitions. Each decision carries its state ID, reason, and evidence reference. A proposed transition stops movement, retains the state context, and requires applicable settled/fresh confirmation. Contact holds preserve travel budgets and best measurements.
- ObservationHelper samples passive device status once per observation, deduplicating shared backends. Hunt owns movement completion, encoder correction attempts, and cancellation. Commands use the observed position and return after one dispatch; no new device polling loops are introduced. Existing device acquisition and legacy blocking APIs remain unchanged. Agent inference has one outstanding request and discards invalidated results.
- Camera/livefeed render a matching cached frame estimate; they do not read hardware. Prediction-only markers, uncertainty ellipses, raw/calibration toggles, and offscreen arrows live in PatchGui. Tracking status text has been removed. Calibration Display now controls whether rig recordings include overlays.

## Hunt method audit

Final Hunt size: **824 lines, 19 methods**, compared with 1,546 lines and 32 methods at the start of this rebuild.

| Method | Start-end lines |
|---|---|
| `__init__` | 19-25 |
| `run` | 27-137 |
| `prepare` | 139-195 |
| `calculate` | 197-285 |
| `decide` | 287-415 |
| `act` | 417-467 |
| `action_gate` | 469-534 |
| `spear` | 536-572 |
| `search` | 574-590 |
| `scan` | 592-623 |
| `shift` | 625-662 |
| `failure_gate` | 664-669 |
| `success_gate` | 671-679 |
| `_recover_contact_visuals` | 681-744 |
| `observe` | 746-766 |
| `_adaptive_contact_confirmed` | 768-798 |
| `_resistance_above_threshold` | 800-810 |
| `_resistance_threshold_reached` | 812-816 |
| `_isCellDetected` | 818-823 |

The three added Hunt-specific methods separate immediate electrical stopping, Adaptive visual confirmation, and single-observation microscope recovery. The original compatibility method `_isCellDetected` has a real caller in `controller/patch.py:350-353`; it is not a retained private forwarding name with no caller.

The six estimator/evidence additions were consolidated in observation infrastructure; two marker additions moved to presentation; five lifecycle additions were absorbed by the existing observation/supervision/state methods. ObservationTracking has no hardware movement commands, mode sequencing, or phase lifecycle.

## Edit audit

These are current line ranges for the changed areas, not a claim that every line inside each range is new. Unrelated pre-existing working-tree changes are retained.

| File | Start-end lines | Change / requirement |
|---|---|---|
| `patcherbot/controller/phases/hunt_cell.py` | 168-171; 245-284; 347-360; 417-467; 482-487; 702-714 | Existing supervisor now owns pending targets, completion, encoder freshness/retries and cancellation; 19 methods and one while loop. |
| `patcherbot/devices/manipulator/helpers/ObservationTracking.py` | 1-487 | Shared home/stage prediction, residual fusion, uncertainty, association, timing checks, frame projection and invalidation. |
| `patcherbot/devices/manipulator/helpers/ObservationHelper.py` | 284-295; 621-624; 717-718; 792-793 | Collect passive motion snapshots once per physical backend in each observation; no completion or retry policy. |
| `patcherbot/controller/PhaseController.py` | 49-57 | Forward target_cell independently of learned tracker cell. |
| `patcherbot/controller/patch.py` | 805-824; 948 | Shared observation forwarding, explicit anchor invalidation hook and settled Locate sample. |
| `patcherbot/controller/phases/approach_cell.py` | 49 | Settled Approach sample through shared observation helper. |
| `patcherbot/interface/pipettes.py` | 170-179 | Validate both six-coordinate records before loading; split XYZ/XYZ for home and safe. |
| `patcherbot/devices/manipulator/helpers/AgentHelper.py` | 3-4; 29-34; 77; 271-344 | Single outstanding asynchronous request, generation invalidation, locked model calls, preserved synchronous interface. |
| `patcherbot/devices/manipulator/manipulatorunit.py` | 49-65; 135-142; 168-173 | Commands use supplied observation context; passive status forwarding; remove pending motion and completion polling. |
| `patcherbot/devices/manipulator/microscope.py` | 62-75; 90-97; 161-165 | Same passive command/status boundary for the microscope; no command-side position sampling. |
| `patcherbot/devices/manipulator/scientificaSerial.py` | 317-356; 442-454; 484-485; 646; 753-754 | One-command starts and passive busy/encoder snapshots; remove newly introduced correction polling and retained targets. |
| `patcherbot/devices/manipulator/sensapexWrapper.py` | 213-241; 308-320; 420-433 | Single SDK dispatch and passive snapshot; remove device-side completion/target state; preserve cancellation generations. |
| `patcherbot/devices/manipulator/fakemanipulator.py` | 127-143; 244-252; 294-302 | Match passive command/status API without retaining supervision policy. |
| `patcherbot/gui/livefeed.py` | 48; 105-118; 151-176 | Read recording toggle; record raw frames or paint existing callbacks at camera resolution; copy source/output buffers. |
| `patcherbot/gui/camera.py` | 569-580; 764-778 | Forward optional frame renderer and emitting camera; guard inherited painters against the wrong camera; remove patch-specific rendering ownership. |
| `patcherbot/gui/patch.py` | 3; 75-76; 93-236 | Own tracking markers and optional raw/calibration shortcuts beside the autopatcher interface. Remove tracking text; delete standalone tracking_overlay.py. |

| `patcherbot/devices/manipulator/CalibrationConfig.py` | 15; 114 | Add Record overlays under Display, default off. |
| `patcherbot/gui/manipulator.py` | 31-33 | Share the calibration config with both recording views. |
| `regression_tests/test_hunt_observations.py` | 117-129; 308-312 | Update retained fixtures for passive motion snapshots; no new tests or test files. |

Follow-up validation: **35 retained regression tests passed**; all changed Python modules compile. Inline checks cover controller-owned retries and cancellation, status sampling, Qt marker equivalence, live recording-toggle changes, raw-buffer preservation, camera routing and source-resolution output. No test files were created. Annotated recordings use RGB8; raw recordings preserve the available source array. Real-device response timing remains unmeasured.

Historical tests changed or added (the generated test files and temporary validation artifacts below were subsequently removed at the user's request):

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

## Historical offline validation before test cleanup

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
