# Adaptive Hunt and observation contract

## Mode behavior

Run captures mode and Slice/Plate once per attempt and passes them explicitly to
Prepare and the decision gates. Prepare has five explicit mode branches. Speeds
and thresholds are read from current config; state holds attempt measurements and
progress. Run uses one observe / decide / action-gate / act loop. All modes request
the same observation fields; observations inside act use that same contract.

- Classic uses configured pipette Z descent (default -10 micrometres/s).
- Manual issues no movement and completes on the resistance threshold or user success.
- Training issues no movement, ignores distance/resistance completion, and waits
  for user success or abort.
- Agent supplies the full observation dictionary to the imitation policy and
  retains calibrated relative-velocity actions and electrical completion.
- Adaptive Plate retains direct descent. Adaptive Slice repeats Spear, Search,
  cell Shift, Scan, pipette Shift, then Spear. Slice/Plate are separate from mode.

Resistance completion retains the delayed five-reading window. Cleanup attempts
all three device stops and preserves an already pending error or user request.

## Adaptive actions

Decide selects the next high-level action. Act applies gated commands, collects
observations, and asks the active S method whether to continue or stop.

- Spear moves pipette Z at the configured speed and monitors the selected cell
  and pipette. An initially absent cell is unknown, not evidence of losing a cell
  already observed during that Spear.
- Search moves microscope Z downward to reacquire the selected cell by calibrated
  image geometry. Multiple plausible candidates are rejected.
- Scan moves microscope Z upward, brackets minimum absolute pipette defocus,
  returns to the best observed Z, and verifies that result.
- Shift centers the observed cell or pipette. Commands have zero pipette Z and a
  planar norm of at most 50 micrometres. A Shift has a total 50 micrometre travel
  budget and requires fresh observations showing improvement after each move.

Search and Scan freeze a travel allowance derived from recorded manipulator and
microscope travel plus hunt_search_margin, capped by max_distance. Missing/stale
vision, invalid coordinates, exhausted travel, or nonconvergence stop the action.
Before dispatch, action_gate validates the concrete command and receiving device.
Shift is checked against both a 50 micrometre command limit and a 50 micrometre
envelope from its starting XY, with zero pipette Z. Search/Scan commands and Scan
return targets must stay within the recorded allowance and current max_distance.
Spear checks direction and distance from its starting pipette Z. Active velocities
are checked again on subsequent observations even when their speed is unchanged.
Manual and Training cannot dispatch automatic motion. Stop commands remain
available at the travel boundary; rejected motion ends the attempt with cleanup.
Association uses a 50 micrometre candidate radius; alignment and pipette-focus
acceptance use 1 micrometre tolerances. These assumptions require rig validation.
There is no cell-focus decision and no CellTracker integration in Hunt.

## Progress failure

The initial gap baseline uses existing cell_distance, which defaults to 20
micrometres. This is the user-approved starting assumption, not a measured gap.
Every completed cycle counts immediately. The measured gap is the absolute
difference between the last accepted Search cell Z and Scan pipette Z, both in
microscope coordinates. Raw manipulator Z is never subtracted from microscope Z.

hunt_min_progress_percent defaults to 25. As soon as a completed cycle achieves
that reduction, including the first cycle, its gap becomes the new baseline and
the counter resets. hunt_progress_cycles defaults to 4: four completed cycles
without the required improvement raise failure before another Spear starts.
Partial operations do not count. Missing/nonfinite Z values fail explicitly;
zero baseline has zero percentage improvement. Cycle results and thresholds are
recorded in the deck. Classic, Manual, Training, Agent, and Adaptive Plate do not
use this progress guard.

## Observation fields and history

Every Hunt observation includes the following fields and field_metadata:

| Fields | Meaning |
| --- | --- |
| manipulator_position | Hardware manipulator XYZ, in its own coordinate system |
| stage_positions | Stage XY plus normalized microscope Z |
| pipette_positions | Image XY plus model defocus; not hardware XYZ |
| target_cell_id, target_cell_reference_coordinates | Stable selected coordinate anchor for this attempt |
| target_cell_image_xy, target_cell_valid, target_cell_status | Current selected-cell association, explicitly unavailable or ambiguous when necessary |
| target_cell_observed_z_um | Microscope context for the associated detection; not a cell-focus estimate |
| target_cell_last_known_z_um, target_cell_last_known_source_frame | Last accepted Search result, explicitly distinguished from current evidence |
| resistance | Electrical resistance reading |
| pressure | Measured pressure, independently timestamped when supported |
| commanded_pressure_mbar, pressure_atm_state | Controller command and ATM state, separate from measured pressure |
| camera_image, deep_learning | Image and model outputs from one inference batch when available |
| observed_at, field_metadata | Read times, source frames, validity, freshness, confidence, and timestamp basis |

The collector snapshots one inference batch for all derived visual fields. It
retains that batch's source image so a later camera frame is not paired with older
model results. With no inference batch, a current camera image is still available
and missing model measurements remain invalid. The legacy four-item observation
list keeps its existing interface.

The phase's self.observation_deck and the helper's deck refer to the same deque.
Hunt retains numeric observations and metadata for the whole attempt, without
copying image arrays into every history row. The current observation still has
its image. Other phases retain their existing bounded histories and default image
retention. Snapshots are independent, calculation enrichment merges into the
matching row, and unavailable values do not silently inherit older measurements.

Decisions can read the deck directly or use observation_window for top-level and
dotted numeric fields, including target coordinates, resistance, pressure, and
calculated progress. Validity/freshness predicates must be supplied when a policy
requires fresh measurements; finite_only alone does not establish freshness.

## Timing and validation boundary

Camera acquisition request start is recorded before snap(), separately from
frame availability and inference start. Post-move visual acceptance requires a
request that started after the finite movement completed. This is not a hardware
exposure timestamp: device buffering and image-to-position synchronization still
require rig validation. Source hardware positions are sampled at inference time
and explicitly labelled as not synchronized to the image.

Pressure acquisition publishes the value and sampling time atomically. Existing
numeric getters remain unchanged. Older controllers without sample timestamps
have explicitly unknown freshness rather than an invented acquisition time.

Hardware-free regressions cover shared observation fields, coherent inference
batches, independent whole-attempt histories, pressure timing, target validity,
post-move freshness, progress thresholds/resets, movement gates, and mode/cleanup behavior. These
checks do not validate cell identity, model accuracy, physical motion, or contact
on the rig.
