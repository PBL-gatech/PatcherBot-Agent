# Pipette Z calibration and actuation-error compensation

Status: narrowed proposal; runtime code has not been changed. Updated 2026-10-22.

Scope: add pipette Z calibration after XY calibration, using existing group moves and detector-driven fine calibration. Collect, fit, validate and save the resulting correction model. No new production modules and no changes to locate_cell, approach or hunt behavior are proposed. Offline/online filtering remains a research comparison; deployment into patching workflows is a separate scope decision.

## Execution brief

**Premise:** The Scientifica PatchStar uses true Z. Existing XY calibration and the detector-driven `fine_calibrate_pipette` sequence work. The new Z stage follows the complete XY stage and uses the existing group-up/group-down operations.

**Problem:** Executing movement commands produces lateral tip displacement, requiring detector-based recentering. This happens at the user's operating speeds. Detection becomes less reliable in tissue.

**Directives:**

1. Characterize actuation-induced XY error accompanying Z movement. Do not introduce an elapsed-time/idle-drift compensation project or propose slower movement as the remedy.
2. Complete existing XY calibration first; retain its fitted coefficients.
3. Use `move_group_up`, `move_group_down`, and detector-driven fine calibration to collect and correct positioning errors in a clear region.
4. Record error before correction, each achieved correction and autofocus movement, and the final residual. Keep correction active during calibration.
5. Fit the correction offline first. Evaluate whether an online estimator would improve prediction using recorded data; do not install an online controller in this change.
6. Preserve existing raw movement semantics, detector checks, XY coefficients and patching workflows. Reuse existing files and classes. Test candidate corrections inside the Z calibration validation sequence only.
7. Treat group movement and pipette-only descent as distinct operating modes until transfer is measured.
8. Use existing Python/configuration patterns, no argparse or required terminal arguments. Obtain implementation approval after this plan; this document does not authorize hardware execution.

**Requirements:** A bounded Z acquisition routine, recorded evidence, fitted movement model and uncertainty, independent validation, and versioned persistence in existing files. Numerical travel and acceptance limits must come from the rig configuration, not arbitrary universal defaults.

**Objective:** Deliver a validated Z correction model and evidence of its effect on detector correction demand in the calibration sequence, while retaining final detector-based alignment.

**Acceptance:** Compare complete held-out movement cycles against existing fine calibration. Compensation must reduce pre-correction XY error and correction demand, meet the configured final-centering tolerance, and preserve existing behavior when no accepted Z model is available.

## Repository evidence and call chain

Line ranges below refer to the source inspected while preparing this plan; they are edit anchors, not a claim that changes have already been made.

| Source | Exact inspected range | Finding |
| --- | --- | --- |
| `patcherbot/gui/patch.py` | 825-827, 850-858 | GUI runs pipette calibration actions sequentially: XY calibration, store calibration positions, move to safe space. |
| `patcherbot/interface/pipettes.py` | 339-343 | Existing manipulator calibration action dispatches `CalibratedUnit.calibrate_pipette`. |
| `patcherbot/devices/manipulator/calibratedunit.py` | 352-366 | XY calibration includes fitting and subsequent centering/autofocus; finish this entire sequence before Z acquisition. |
| `patcherbot/devices/manipulator/helpers/PipetteCalHelper.py` | 152-164, 317-340 | Existing fitting samples XY with fixed Z and constructs zero Z-to-image-XY terms. |
| `patcherbot/controller/patch.py` | 369-385, 952-960 | `AutoPatcher.fine_calibrate_pipette` performs three center/autofocus cycles; `locate_cell` invokes it around group-down movements. |
| `patcherbot/controller/patch.py` | 1086-1111 | Group operations move microscope Z first and pipette Z second, with waits. |
| `patcherbot/devices/manipulator/helpers/PipetteCalHelper.py` | 362-378 | Pipette autofocus actuates pipette Z. It must be recorded, not assumed to be observation-only. |
| `patcherbot/devices/manipulator/calibratedunit.py` | 407-420, 443-466 | Existing detection gate and pixel-to-motor correction. Record inputs/results while retaining this behavior. |
| `patcherbot/devices/manipulator/calibratedunit.py` | 788-810 | Existing calibration persistence; add optional Z-model metadata here. |

Proposed orchestration: GUI calibration sequence -> existing XY action -> new AutoPatchInterface Z action -> AutoPatcher Z runner -> existing PipetteCalHelper sampling/fitting methods and existing group moves/fine calibration. Do not make CalibratedUnit depend on AutoPatcher.

Failure propagation must prevent the sequential GUI workflow from storing positions as if Z calibration succeeded. The runner must honor the owning AutoPatcher cancellation state and stop both devices on failure. Check microscope readback normalization against `devices/manipulator/microscope.py` lines 45-70 and `controller/phases/approach_cell.py` line 153; normalize logged units once rather than assuming raw readback and command units are interchangeable.

## Files to edit and degree of change

Six existing production files would be touched: four small wiring/configuration/persistence edits and two localized additions for acquisition and fitting. No new production files are required. This is a proposed scope, not implemented code.

| File | Current edit anchor | Degree | Proposed change |
| --- | --- | --- | --- |
| `patcherbot/gui/patch.py` | 826-826 | Small, one sequence entry | Insert Z calibration after XY and before storing positions. |
| `patcherbot/interface/patch.py` | insertion adjacent to 766-776 | Small, one action | Dispatch Z calibration to the existing AutoPatcher. |
| `patcherbot/controller/patch.py` | insertion beside 369-385 | Moderate, localized | Add the bounded Z sweep using existing group methods and fine calibration; expose optional records from each fine-calibration substep. Existing calls retain their behavior. |
| `patcherbot/devices/manipulator/helpers/PipetteCalHelper.py` | 186-243; insertion after 353; 362-378 if focus records require it | Moderate, localized | Reuse fresh-frame sampling; add Z records, numerical fitting and validation methods to the existing helper. Record autofocus motion only as needed by the acquisition loop. |
| `patcherbot/devices/manipulator/calibratedunit.py` | 788-810 | Small, persistence | Save/load optional validated Z model metadata. Missing legacy metadata clears stale Z state. XY matrices retain their existing values. |
| `patcherbot/configs/CalibrationConfig.py` | 77-95, 114-128 | Small, configuration | Add sweep bounds, movement levels, repeats and centering acceptance limits to the existing configuration. |

Keep group primitives at `controller/patch.py:1086-1111` unchanged. Keep the existing XY action, `interface/pipettes.py`, and locate/approach/hunt call sites unchanged. The existing unit loader is the correct place to manage Z metadata; an additional interface reset is unnecessary.

Extend `tests/test_calibration_reload.py` for optional Z metadata and legacy-load behavior when implementation occurs. Numerical fitting and acquisition need hardware-free checks described below; existing recovery tests are regression checks, not evidence that Z calibration works. No new test files are prescribed by this plan.

## Collection protocol

1. Finish XY calibration in a clear, verified working region. Save reference pipette XYZ, microscope Z, stage XY, XY matrix, detector model, and objective configuration.
2. Use the configured bounded up/down sequence with several movement magnitudes and both repeated-direction and reversal cases. Use the existing operating speed; record it as context.
3. At each event boundary, obtain fresh valid detector observations. Keep stage XY fixed. Execute the existing group primitive and wait for completion.
4. Measure post-group error before correction, then run the existing fine-calibration sequence with event recording. Record each XY centering action and each pipette autofocus separately, in order. Obtain a final fresh observation after the last autofocus.
5. Keep detector corrections active. Log rejected detections and unsuccessful samples; do not silently replace missing evidence with the predicted center. A calibration-specific acquisition window may need to be wider than the usual anchor gate, with tip identity independently checked.
6. Do not reuse the XY calibration retry that introduces random XY jitter. Repeat or reject a failed Z sample without unrecorded changes to the trajectory.
7. Group up/down must be labeled by measured coordinate changes as well as helper name: existing callers can supply signed distances. Record controller units and converted micrometers explicitly.
8. Reserve entire cycles for validation before fitting. A proposed initial acquisition budget is six fitting cycles and three validation cycles, with multiple transitions per cycle; these are pilot counts, not a guarantee of statistical sufficiency. Repeat a validation block later in the session. Increase collection only if estimates remain unstable.
9. On success, restore the reference group position and verify focus/centering before storing calibration positions. On failure/cancellation, stop movement, retain the prior accepted model, report actual positions, and restore settings; any return movement must use the existing validated path and cancellation policy.

Each event record should contain: cycle ID and fit/validation split; parent group ID; operation type; signed requested and achieved motion; before/after pipette XYZ, microscope Z and stage XY; acquisition timestamps/frame IDs; tip coordinates and quality; detector correction requested and achieved; focus correction; success/rejection reason; raw frame references. Record evidence timing honestly: current calibration uses camera retrieval timing, not verified sensor-exposure timestamps.

For estimator bookkeeping, use exactly one process-noise increment per modeled physical movement. Group records summarize children; do not charge uncertainty for both the parent and its children. Repeated detector frames are measurement updates, not new movement events.

## Parameters: collection, fitting, and acceptance

| Parameter | Evidence | Proposed estimation/test |
| --- | --- | --- |
| Existing XY Jacobian `Jxy` | Completed XY calibration | Freeze; test conversion/sign consistency. |
| Mean group Z coupling `beta_up`, `beta_down` (pixels/um) | Paired detector positions around group-only events, actual Z and XY movement | Initial directional linear fit; evaluate by held-out cycle. |
| Depth/reversal dependence | Repeated depths, same-direction moves, reversals | Add terms only if held-out causal prediction improves. |
| Observation uncertainty `R` | Fresh observations at event boundaries; independently checked image subset | Estimate aggregate-observation variability under stated assumptions; do not divide by frame count without checking dependence. No idle-drift characterization is required. |
| Movement uncertainty `Q_event` | Full event traces including detector corrections | Structured innovation-likelihood fit with constrained/bounded R. Compare predicted versus measured validation errors. |
| Persistent coupling `rho`, `Q_bias` | Residual sequence across fitting moves | Fit only if correlation remains after the mean model; retain only if online replay outperforms offline-only. |
| Acceptance tolerances | Rig's actual centering requirement | Configure in um; do not use detector search radius as a centering-accuracy tolerance. |

Observed pre/post residual covariance includes detector uncertainty at both ends. Do not assign all of it to Q and then add R again. Input/output logs alone generally do not uniquely identify physical process versus detector noise; if constraints cannot identify them, save empirical prediction uncertainty and limit physical interpretations [S1].

Because corrections depend on detections, these are closed-loop records. Use the initial regression to initialize a structured prediction-error/maximum-likelihood fit; do not assume arbitrary feedback regressions are unbiased [S2]. Keep the known XY mapping fixed and parameter count small. EM with Kalman smoothing is a possible offline optimizer [S3]; validation must use causal predictions, never future-informed smoothed states.

## Model and offline/online research decision

Define detector error as `detected_xy - target_xy`. The first deliverable is an offline map `g(move)` for expected lateral displacement. For valid XY mapping Jxy, the correction in manipulator micrometers is `-solve(Jxy, predicted_error_px)`.

The initial fitted output is a direction-dependent movement-error map plus measured residual variability. Validation applies its proposed XY correction inside the calibration routine, then uses the detector/fine-calibration loop for the remaining error.

For the requested SSM/Kalman assessment, compare offline-only predictions with a movement-indexed state-space candidate on recorded cycles:

`e_next = e + Jxy @ actual_dxy + g(move) + actual_dz * b + w`

`b_next = rho * b + eta; detector_error = e + measurement_noise`

Here `b` is a two-dimensional residual coupling state and the index advances on movement, not idle time. Known inputs make this a linear-in-state model [S4]. Test whether its causal predictions improve on the offline map. Independent random per-move error cannot be predicted from preceding movements; an online filter is justified for next-move compensation only if predictive structure and validation gains are demonstrated. This plan does not implement the estimator in the live patching workflow.

Group primitives move both microscope and pipette. Keep the first model mode-specific. Do not insert the combined group coefficient directly into the global pipette-only Z column of M; that would silently change the meaning of other coordinate transforms.

## Concrete code suggestions

These are proposed additions/replacements for implementation review. They are not installed in runtime modules. Numerical examples are self-contained; hardware integration still requires the recorder, validation checks and failure handling specified above.

### 1. GUI sequence replacement

Replace only the pipette-calibration assignment at `gui/patch.py:826`:

```python
self.pipette_calibration = [
    self.pipette_interface.calibrate_manipulator,
    self.patch_interface.calibrate_pipette_z,
    self.patch_interface.store_calibration_positions,
    self.patch_interface.move_to_safe_space,
]
```

### 2. Interface action addition

Add beside the existing group actions in `interface/patch.py`. This class already uses `blocking_command`:

```python
@blocking_command(category='Patch',
                  description='Calibrate pipette Z movement',
                  task_description='Calibrating pipette Z')
def calibrate_pipette_z(self):
    self.execute(self.current_autopatcher.calibrate_pipette_z)
```

The new AutoPatcher method should own the sweep and cancellation, call its existing group/fine-calibration methods, and send records to its existing calibrated unit's PipetteCalHelper for fitting. Add methods to that existing helper; do not create another helper module, controller or device connection.

### 3. Offline directional mean initializer

Suggested pure numerical function in the existing `helpers/PipetteCalHelper.py` module, called by its new Z-fit method. Each row must be one accepted group-only event. `delta_image_xy` is post-minus-pre detector position; `actual_dxy` is the achieved manipulator XY input during that event. Stage XY must be fixed. Fit separately for physical upward/downward movements and retain microscope-mode metadata.

```python
import numpy as np


def fit_group_z_slope(dz_um, delta_image_xy, actual_dxy_um, jxy):
    dz = np.asarray(dz_um, dtype=float)
    dp = np.asarray(delta_image_xy, dtype=float)
    dq = np.asarray(actual_dxy_um, dtype=float)
    jxy = np.asarray(jxy, dtype=float)
    if dz.ndim != 1 or dp.shape != (dz.size, 2) or dq.shape != dp.shape:
        raise ValueError("Expected N Z moves and N-by-2 image/XY displacements")
    if dz.size < 3 or jxy.shape != (2, 2):
        raise ValueError("Need at least three moves and a 2-by-2 XY calibration")
    if not all(np.isfinite(a).all() for a in (dz, dp, dq, jxy)):
        raise ValueError("Calibration inputs must be finite")
    if np.any(dz == 0) or not (np.all(dz > 0) or np.all(dz < 0)):
        raise ValueError("Fit one physical direction using nonzero Z moves")
    residual = dp - dq @ jxy.T
    beta = np.linalg.lstsq(dz[:, None], residual, rcond=None)[0][0]
    errors = residual - dz[:, None] * beta
    return {
        "beta_px_per_um": beta,
        "training_residuals_px": errors,
        "training_rmse_px": float(np.sqrt(np.mean(np.sum(errors ** 2, axis=1)))),
    }
```

This produces an initializer and training diagnostics, not evidence of held-out accuracy and not physical Q/R estimates. If group-only detections are out of focus, reject these rows and fit the full logged sequence with explicit autofocus events instead of silently combining motions.

## Testing and parameter-selection procedure

1. Test the fit with known synthetic up/down coupling, reversed camera axes, zero coupling and invalid input. Measure error on separate generated samples, not only fitted samples.
2. Use fake devices to verify that XY finishes before Z; group/fine operations run in order; actual corrections and autofocus movements are recorded; cancellation stops collection and prevents publishing a model.
3. Extend existing calibration reload tests for Z metadata round-trip, malformed metadata and legacy files. Check that no Z calibration changes existing XY coefficients.
4. Reserve complete real movement cycles before fitting. Measure pre-correction XY error, mean bias, RMS error, final residual, achieved XY correction distance and rejected detections. Use actual corrections and autofocus movements when reconstructing the trajectory.
5. Within the clear-region calibration routine, interleave existing group-plus-fine-calibration trials with matched trials that apply the fitted correction before detector residual correction. Require improvement beyond run-to-run variability and the configured final-centering tolerance.
6. Evaluate the optional SSM/Kalman candidate offline on these records using causal predictions. Do not credit future-informed smoothing or smoother-looking traces as proof of improved movement correction.
7. Save the traces, cycle split, model parameters, validity envelope and validation report alongside existing calibration data. A failed validation does not replace an accepted model.
8. Production use in locate/approach/hunt, pipette-only descent, and tissue is outside this change. Those integrations require their own agreed scope and validation.

## Research basis

- [S1: MathWorks, separate process/measurement noise identification](https://www.mathworks.com/help/ident/ug/identifying-state-space-models-with-independent-process-and-measurement-noise.html): unrestricted input/output identification does not uniquely determine physical Q/R; use structured assumptions and an innovations predictor.
- [S2: Gustavsson, Ljung and Soderstrom, closed-loop identification](https://portal.research.lu.se/en/publications/identification-of-processes-in-closed-loopidentifiability-and-acc/): prediction-error approaches can identify systems under feedback when their assumptions and information requirements are met.
- [S3: Ghahramani and Hinton, Parameter Estimation for Linear Dynamical Systems](https://www.cs.toronto.edu/~hinton/absps/tr-96-2.pdf): offline EM estimation uses Kalman filtering and backward smoothing for latent-state estimates.
- [S4: MathWorks, linear Kalman tracking model](https://www.mathworks.com/help/fusion/ref/trackingkf.html): linear state transition with known controls and separate process/measurement noise supports prediction and measurement updates.
- [S5: MathWorks, residual analysis](https://www.mathworks.com/help/ident/ug/what-is-residual-analysis.html): test residual whiteness and independence from past inputs to assess missing dynamics. Use these checks to justify additional movement-memory states.
- [S6: PI, dynamic error compensation](https://www.pi-usa.us/en/tech-blog/improving-stage-accuracy-with-acs-dynamic-error-compensation): measured error maps support trajectory correction, with performance limited by repeatability.

## Remaining decisions, risks and audit

- Rig-specific Z envelope, movement levels, repeat budget and centering tolerance remain configuration decisions before hardware execution. This plan does not invent safe distances.
- Detector errors and actuator errors are partially confounded; the initial result is a predictive movement calibration, not identification of a thermal or mechanical root cause.
- Collection, fitting and real-rig validation exceed a five-minute implementation pass. Implement in separately reviewable increments within the agreed calibration scope: acquisition/persistence, then fitting/validation. Respect the working agreement's stop/report rule for each implementation pass; do not claim the full feature finished after a partial pass.
- Current edit audit: only this existing plan document is revised; new production modules and live-workflow integration have been removed from the proposal. Runtime implementation, production tests and hardware validation remain unperformed.
- Document verification: the retained Python examples are checked for syntax after this revision. A numerical smoke check was attempted but could not execute because the environment's imported `numpy` module has no `array` attribute. Numerical example correctness has therefore not been confirmed by execution in this environment.
