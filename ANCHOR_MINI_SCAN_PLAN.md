# Set Anchor using a 40× mini scan

## Execution brief

- **Premise.** Set Anchor currently takes one settled 40× frame and searches the loaded, aligned 4× Session scan. [Controller, lines 174–190](patcherbot/controller/patch.py#L174-L190)
- **Problem.** One 40× field may show too little shared detail to locate it reliably. The operator should be able to start at any suitable field covered by the 4× scan.
- **Objective.** Use the surrounding 40× fields to locate the starting field on the 4× map, without asking the operator to choose a nearby reference anchor.
- **Directives.** Keep the work within Set Anchor, reuse existing methods where practical, and preserve Save Origin, full scans, calibration, and Undo Anchor. Use the existing GUI command with no new terminal arguments. Keep investigation and implementation to approximately five minutes.
- **Initial settings.** Try a 3×3 grid with about 30% overlap in each direction. Treat this as a starting point for testing on the rig.

## What to reuse

| Reuse | Change needed |
| --- | --- |
| [Settled capture, lines 126–154](patcherbot/controller/patch.py#L126-L154) and [stage movement, lines 120–125](patcherbot/devices/manipulator/helpers/StageScanHelper.py#L120-L125) | Capture the center and eight neighbors. Record the measured XY for each settled frame and use the existing move and wait methods. |
| [Mosaic movement, lines 1097–1149](patcherbot/devices/manipulator/calibratedunit.py#L1097-L1149) and [stitch placement, lines 225–236](patcherbot/utils/scanStitcher.py#L225-L236) | Follow the existing placement sign convention. Add a small in-memory tile assembler. The existing mosaic uses abutting tiles, and the [full stitcher, lines 43–75](patcherbot/devices/manipulator/helpers/ScanStitchHelper.py#L43-L75) reads a completed recording from disk. |
| [Session matching, lines 247–334](patcherbot/deepLearning/ObjectiveRegistrar.py#L247-L334) | Keep the calibrated rescale and whole-map search. Allow the caller to supply the starting frame's center in mosaic pixels. Keep the current image-center default for existing callers. |
| [Anchor saving and undo, lines 191–243](patcherbot/controller/patch.py#L191-L243) and [result dialog, lines 223–265](patcherbot/gui/patch.py#L223-L265) | Keep the starting frame as the Experiment Book snapshot. Show the scaled mosaic and matched Session crop in the existing preview. |

**Reuse effort.** Moderate. Most behavior is already in place. The main work is collecting the nine frames and keeping track of the starting field as they are assembled and matched.

## Implementation sequence

1. **Capture the starting field.** Check the Session alignment and the 40× stage calibration before moving. Save the starting XY and capture a settled frame there.
2. **Collect the neighboring fields.** Convert the grid spacing from camera pixels to stage motion using the 40× calibration. Visit the eight neighbors, wait for each move to finish, and record actual XY with every frame. Return to the starting XY when capture completes, fails, or is canceled. Check that the stage reached its starting position before matching.
3. **Build the mosaic.** Place tiles from the measured stage offsets, blend their overlap, and crop incomplete outer edges. Reject a mosaic with gaps in the matching area. Carry the starting frame's center through the crop so the correction uses the same physical point that the stage started at.
4. **Find the anchor.** Search the loaded 4× map using the existing matcher. Convert the starting frame's center to Session coordinates and pass it to the existing correction and save path. If the scan, stage return, or match fails, keep the previous anchor and show the failure. If the match succeeds, show the preview and keep Undo Anchor available.

## Acceptance checks

- **Tile placement.** Simulated captures with measured positions different from the commanded positions still assemble correctly. Check both stage axes and verify that an asymmetric crop preserves the starting point.
- **Stage return.** Simulate a successful scan, a camera failure after a move, and cancellation. Confirm the controller commands and verifies the return to the starting XY. A failed return or failed match leaves the previous anchor intact.
- **Workflow.** Verify that Set Anchor shows scan progress, saves the original 40× snapshot, shows the mosaic and match preview, and supports undo. Save Origin and the full scan retain their existing behavior.
- **Matching on the rig.** Compare the center frame alone with the mosaic from the same capture at manually confirmed locations, including fields with few distinguishing features. Record how often each method finds the correct location and how far each match is from the confirmed position. Treat improved reliability as unverified until this comparison is done.
