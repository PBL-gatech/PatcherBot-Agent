# Hunt functionality audit ? 2026-09-23

## Scope and result

Compared the pre-phase implementation at `c52bcff778283c64a2f33c581e59920be007ec02` (`d51bdd25de^`) with phase introduction `d51bdd25de`, observation rewrite `f31a36d451`, committed code `38f81ab037`, and the current local correction. Inspected the entire current Hunt phase, the legacy hunt/detection/tracking functions, directly interacting observation/controller/interface code, and regression tests. This is a code-path audit with simulated execution, not hardware certification.

The code is NOT behaviorally equivalent to the pre-phase implementation. The single-sample false-positive detector predates the phase refactor; subsequent changes also enabled Agent motion, replaced Adaptive Slice movement, bypassed legacy tracking, and removed diagnostic evidence. The local correction addresses resistance confirmation only; those other differences remain.

## Exact premature-success mechanism

Before the local fix, Hunt initialized a five-entry deque with the averaged baseline, then appended the previous resistance before updating the current resistance. Once length reached five, the detector checked only entry 4. The buffer length was a warm-up check, not confirmation across five values.

Reproduction through `HuntCellPhase.run`: baseline 5.0 MOhm, threshold 0.15 MOhm, observations `[5.0, 5.0, 5.4, 5.0, 5.0]`. On observation 4, the evaluated delayed reading is 5.4 although the current reading is 5.0. The old code raises RequestedSuccessException. It reproduces in Classic, Adaptive, Manual and Agent for both Plate and Slice. Training excludes automatic success.

Sources: pre-fix `38f81ab037:patcherbot/controller/phases/hunt_cell.py`, lines 35?42, 95?103, 509?511, 741?750. The buffer is seeded with the baseline and the baseline is appended again at the first observation, so it can reach length five after only four actual observations.

The same single-index predicate exists before phases in `6e41911f4c:patcherbot/controller/patch.py:1698?1722`, `f361d6730c:patcherbot/controller/patch.py:1557?1581`, and `0dace7bba4:patcherbot/controller/patch.py:1602?1626`. The historical increasing-readings condition is commented out. It would be incorrect to claim the phase introduction removed an active five-reading confirmation rule.

## Functionality comparison

Historical line references below use `d51bdd25de^:patcherbot/controller/patch.py` unless another commit/file is named. Current phase references use `patcherbot/controller/phases/hunt_cell.py` after the local correction.

| Area | Before phases | Current behavior and impact | Sources |
|---|---|---|---|
| GUI Hunt entry | Packages selected cell, increments recording sample, executes controller hunt | Exactly identical | Current `patcherbot/interface/patch.py:441?450` |
| Task dispatch / success flags | Resets flags at task start; success exception logs success | Entire interface/base.py is exactly identical | Current `patcherbot/interface/base.py:222?263` |
| Readiness / selected cell | Requires ready rig and non-null cell | Preserved | Historical 974?982; current 85?90 |
| Pressure / stabilization | Near-cell pressure, 3-second wait | Preserved | Historical 984?987; current 91?92 |
| Baseline | Five DAQ samples at 200 ms spacing, averaged | Baseline helper is AST-exactly identical | Historical resistanceRamp 1236?1239 / safe average 1186?1217; current `controller/patch.py:822?825,837?868` |
| Live resistance | Direct DAQ cached value each loop | ObservationHelper calls resistanceRamp with one sample / 1 ms; invalid-value handling and possible trace adjustment now occur inside acquisition | Historical 991,1089?1092; current `ObservationHelper.py:205?208,552?557,584?586`; changed in f31a36d451 |
| Sample buffer / detection | Delayed sample; length five; compare only index 4 | Committed refactor retained this weakness. Local fix uses current samples, empty initial buffer, five valid elevated readings | Historical 989?998,1089?1092,1698?1722; current 35?42,95?103,509?511,748?761 |
| Classic movement | Starts constant Z descent before loop | Starts after first observation/decision; retains absolute Z velocity, now explicitly gated | Historical 1003?1008; current 35?50,387?388,409?425 |
| Adaptive Plate | Same Z descent as Classic | Still follows direct Classic-like descent | Historical 1009?1014; current 332,387?388 |
| Adaptive Slice | Same direct descent as Classic | New spear -> search -> stage shift -> scan -> pipette shift cycle | Historical 1009?1014; current 112?129,339?377,521?729 |
| Adaptive automatic success | Uses common resistance detector | f31a36d451 temporarily disabled success when an operation existed (lines 432?446: unfinished contact confirmation). 38f81ab037 re-enabled the shared detector for these operations | Current 438?445,741?746 |
| Agent motion | Prepares model; inference/movement block commented out | Phase introduction enabled inference and relative calibrated velocity; current retains it | Historical 1015?1020,1043?1060; d51bdd25de phase 119?138; current 378?386,419?424 |
| Manual | No automatic descent; resistance can complete hunt; distance limit applies | Preserved mode behavior, subject to changed sampling/detection | Historical 1021?1022,1030?1037,1062?1077; current 106?107,389?390,731?746 |
| Training | Reports resistance threshold, stops devices at threshold, then waits for manual Success/Abort; ignores travel limit | Ignores resistance entirely and waits for operator; threshold-triggered stop/status reporting removed | Historical 1030?1041,1093?1105; current 108?109,391?392,734?745 |
| Global distance | All non-Training modes use absolute displacement from initial Z; limit cast to int | Classic/Manual/Agent/Adaptive Plate use numeric limit without int truncation; Adaptive Slice bypasses this global failure gate | Historical 1062?1077; current 731?739 |
| Adaptive limits | No separate operation limits/progress cycles | Spear limits reset per action; bounded search/scan travel, 50 um shift envelope, 60-second operation timeout, progress checks across cycles | Current 213?290,312?325,333?372,426?447,521?595,597?729. Not equivalent to total hunt travel bound |
| Configurable tracking | Calls track_cell every loop, honors track_cell/use_ai_features and tracking settings | Hunt no longer calls it. Producer gets cell=None; helper explicitly marks tracker no_target. Selected-cell detection association is a replacement with different semantics | Historical 1024?1029,1087,1120?1160; current 98?103,123?125,145?189; `ObservationHelper.py:458?476` |
| AI enablement | Tracking gated by use_ai_features | Adaptive Slice starts model producer without this gate; other modes do not start that producer in Hunt | Historical 1026,1128; current 112?129 |
| Visual target association | Tracker tied to supplied target/reference image | Fresh frame/context checks; exactly one detection within 50 um of projected selected position; ambiguous matches fail during Adaptive operations | Current 143?204,532?540,563?588. Different behavior from tracker identity continuity |
| Mode changes during attempt | Repeated config reads could change behavior within one attempt | Mode and cell type captured at run start; numerical config still read during execution | Historical 1003?1031; current 31?32,333?338,387?393,523,746 |
| Exception cleanup | Normal tail stops devices; no encompassing finally for entire hunt | Finally stops stage, pipette and microscope; stops started inference; preserves pending exceptions | Historical 1097?1118; current 56?70,395?408,513?519 |
| Observations / history | Local five-entry resistance deque and direct hardware reads | Full phase observation history, metadata and calculations; images omitted from stored history | Current 94?103,134?211; `ObservationHelper.py:36?68` |
| Config additions | Existing distance/speed/resistance/tracking settings | New search margin, progress-cycle count and minimum progress percentage | Current `patcherbot/interface/patchConfig.py:26?31,65` |
| Diagnostic logging | Logs baseline, mode, pressure, detected resistance delta | Baseline/delta and training threshold logs removed; supplied success line cannot identify automatic versus manual success | Historical 985,1000?1001,1714?1719; current 85?109,763?768; interface/base.py233?239 |

## Local correction and edit audit

Only resistance confirmation behavior was changed in production. No historical movement or tracking behavior was restored without a separate implementation decision.

| File | Start?end lines | Change / directive satisfied |
|---|---|---|
| patcherbot/controller/phases/hunt_cell.py | 37?38 | Append current observation; correct stale-sample completion |
| patcherbot/controller/phases/hunt_cell.py | 98?98 | Empty initial buffer; count actual hunt observations |
| patcherbot/controller/phases/hunt_cell.py | 510?511 | Same current-sample handling within Adaptive action loop |
| patcherbot/controller/phases/hunt_cell.py | 748?761 | Require all five readings above threshold; reject nonfinite/nonpositive baseline, threshold or readings |
| regression_tests/test_hunt_observations.py | 557?557; 639?639 | Update two legacy timing expectations from four actual readings to five |
| regression_tests/test_hunt_contact_confirmation.py | 1?87 | Contact criterion, invalid input and public-run noise coverage |
| regression_tests/test_hunt_lifecycle_regression.py | 1?101 | Public lifecycle: stale startup spike, confirmation, distance, abort, Training |
| regression_tests/test_hunt_recorded_replay.py | 1?60 | Replay measured noise through full Hunt and append synthetic sustained contact |
| regression_tests/fixtures/hunt_resistance_2026_09_23.csv | 1?32 | Preserve 31 measured timestamp/resistance pairs for reproducible tests |
| docs/hunt_functionality_audit.md | 1?95 | Historical functionality comparison, evidence, validation and remaining decisions |

The five-consecutive-reading rule is a deliberate correction to a longstanding weakness, not a claim that this was the historical rule. Manual success still bypasses automatic contact confirmation, and Training still awaits operator completion.

## New tests and regression proof

1. Startup spike sequence `[5.0,5.0,5.4,5.0,5.0]`: run the public phase; require no success, then terminate the finite test stream by abort; verify every motion device is stopped. Pre-fix code incorrectly succeeds in all eight non-Training mode/type combinations.
2. Genuine sustained rise after noise: require success only after five elevated current observations, with all devices stopped.
3. Distance exhaustion: simulate displacement reaching max_distance and require AutopatchError, not success, for direct-motion modes.
4. Operator abort: require abort and all devices stopped in all mode/type combinations.
5. Training: elevated resistance and large manual displacement must not auto-complete; operator abort still works.
6. Recorded replay: extract unchanged resistance values from `experiments/Data/rig_recorder_data/2026_09_23-14_10/graph_recording.csv`, epoch timestamps 1790187404.013?1790187405.533, immediately preceding reported success. Values span 13.017618?13.392163 MOhm. At test baseline 13.0 and threshold 0.3, there are six crossings with a maximum run of four; at test baseline 13.05 there are two isolated crossings. Neither constitutes five confirmations. Exercise Classic, Manual, Agent, Adaptive Plate and Adaptive Slice, then test subsequent sustained contact.

Recorded replay baselines 13.0/13.05 are explicit injected test assumptions. The runtime baseline was not logged; these tests reproduce a plausible failure mechanism, not the exact acquisition stream or baseline of the event. Recorded values are not rescaled or synthesized. Synthetic sustained-contact suffixes are separately identified in the test.

Loaded `git show HEAD:patcherbot/controller/phases/hunt_cell.py` into an isolated AST test namespace without changing the working tree. Recorded replay produced 11 unexpected-success errors and one premature-completion assertion failure across its parameterized cases. The startup-spike lifecycle case independently failed in all eight non-Training combinations. Current code passes both.

Validation command: `C:/Users/sa-forest/.conda/envs/agent/python.exe -m unittest discover -s regression_tests -p 'test_hunt*.py'`. Result: **49 tests passed**, including all existing Hunt tests. `git diff --check` passed with only Git line-ending conversion notices.

## Remaining risks and next decisions

- A resistance rise alone is not proof of contact with the selected cell. The fix rejects short transients; sustained noise, obstruction or a bad baseline can still satisfy it. No hardware contact outcome was validated.
- Five observations are five controller reads, not proven distinct DAQ acquisitions. DAQ.resistance returns the cached totalResistance (`patcherbot/devices/amplifier/DAQ.py:649?654`); unique electrical acquisition identifiers are not checked by this gate.
- Automatic success still applies during every Adaptive operation. This audit does not establish that resistance confirmation should be permitted during search, scan and alignment, versus descent only.
- Adaptive Slice has no equivalent cumulative distance bound. Its per-operation limits/progress checks are a different policy requiring explicit acceptance and physical verification.
- Tracking configuration no longer delivers the pre-phase behavior; restoring it requires deciding whether the legacy tracker or the newer detection association is intended.
- Training threshold-triggered stop/reporting differs from legacy. Current tests document today's behavior; they do not prove this change was desired.
- Numeric threshold/baseline validation rejects invalid automatic confirmation, but does not diagnose the invalid setting at startup. The gate has no acquired-baseline/contact-detail log.
- For exact incident attribution, capture the actual first_res, cell_R_increase, timestamped evaluated readings and whether operator Success was requested. Those facts cannot be reconstructed conclusively from the provided success line.

Recommended next implementation: decide intended Adaptive travel/contact/target-tracking contracts against the documented legacy behavior, add acceptance tests for those chosen contracts, then change those paths separately. Do not treat the 49 passing tests as proof that all legacy functionality was preserved.

## Follow-up: cell association and logger correction

The comparison above records the earlier audit snapshot; its current-code line references predate this follow-up. Mode/cell-type and Adaptive-operation output now uses controller.info, not stdout print. Hunt now ranks detections by calibrated distance to the selected cell's projected position within the existing 50 um association radius. Multiple detections are allowed; nearest distances within 1 um of one another remain ambiguous. Spear/search reject ambiguity rather than the presence of multiple candidates. Shift pauses on missing/ambiguous target evidence and retries up to 50 fresh observations, then reports status/count; the existing motion bounds remain. Geometric association does not establish visual identity if the selected cell disappears and a neighbor is the sole nearby detection.

Edit audit: patcherbot/controller/phases/hunt_cell.py lines 34?34 and 435?436 (logger); 148?148 and 177?197 (association); 543?544 and 590?591 (monitor checks); 706?718 (shift retry). Regression coverage is in regression_tests/test_hunt_cell_association.py.

## Follow-up: stop first, confirm while stationary

Resistance now stops stage, pipette and microscope at the first qualifying reading. Five consecutive readings confirm success while stationary; a valid dip resumes the same operation, while invalid confirmation resistance or abort fails/stops. Spear, Search, Scan and Shift now stop all devices at candidate completion and require a new observation before advancing. Visual confirmations require a different frame acquired after stopping. Search/Shift may resume after a rejected confirmation; Shift does not consume displacement budget for an unexecuted confirmation command. Unconfirmed Spear travel or Scan focus fails explicitly. Confirmation waits are bounded to 50 observations. Training remains operator-controlled.

Logging is once per hunt/operation start with mode-specific conditions (resistance baseline/target, pressure, speed, travel, selected target or focus as relevant); per-observation spearing/searching/scanning/shifting messages were removed.

Edit audit in patcherbot/controller/phases/hunt_cell.py: 35?51 (startup logging/contact pause), 108?113 and 132?150 (mode/start conditions), 466?509 (operation conditions and stop/confirm dispatch), 572?578 (stop on newly acquired resistance before sleeping), 587?646 (stationary confirmation), 648?651 / 674?677 / 724?727 / 794?797 (remove repeated monitor logs), 887?922 (candidate stop versus sustained success). Test fixture edits: regression_tests/test_hunt_observations.py120,158,542,559,605,642,733 model initial clear readings and a prepared baseline. New tests: regression_tests/test_hunt_contact_stop.py1?137; regression_tests/test_hunt_operation_confirmation.py1?183.

Validation: 71 Hunt tests passed. New tests assert device stop calls precede the next measurement; no movement occurs during confirmation; transient resistance and rejected visual confirmation resume correctly; stale and pre-stop frames cannot confirm; Shift budget counts only commanded moves; lost Scan focus and abort do not advance. The new tests reproduced continued motion/repeated logging and immediate unconfirmed operation completion in the pre-change code. These are simulated-hardware checks, not a physical-rig validation.

## Follow-up: Adaptive contact identification

Adaptive automatic success now requires sustained resistance plus a fresh identified pipette tip within the user-approved 5 um image-plane distance of the selected cell. Current detection or this attempt's last-known cell location is accepted; saved coordinates are projected with the detector frame's stage position. Ambiguous matches, stale/missing tip evidence, mismatched target identity or excessive distance cannot report success. Contact confirmation requires a new tip frame acquired after the stop. Missing visual confirmation is bounded to 50 unique observations while resistance remains confirmed; motion stays stopped and failure is explicit. Both Adaptive Plate and Slice start/stop detection. Manual Success remains an explicit override.

The new Hunt setting hunt_contact_distance is independent of the optional target-matching radius. Tests include calibrated geometry, source-frame stage compensation, last-known identity, missing/stale tip, resistance-only rejection and post-stop freshness. The updated suite passes 84 tests. This proximity rule is XY, not a reconstructed 3D contact measurement.
