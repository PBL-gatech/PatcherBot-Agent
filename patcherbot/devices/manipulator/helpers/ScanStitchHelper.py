"""Connect the existing patch-controller scan to the fast stitcher."""
from pathlib import Path
import time
from uuid import uuid4


class ScanStitchHelper:
    def __init__(self, stage):
        self.stage = stage
        self.record = None

    def begin(self):
        if not self.stage.calibrated:
            raise RuntimeError("Load or calibrate the stage matrix before scanning.")
        if len(self.stage.scan_corner_positions) != 2:
            raise RuntimeError("Store both scan corners before scanning.")
        self.record = {
            "start": time.time(), "end": None,
            "stage": {"M": self.stage.M.tolist(), "r0": self.stage.r0.tolist()},
            # Use the validated reference's tile-placement convention.
            "flip_x": True,
            "flip_y": True,
        }

    def complete(self):
        self.record["end"] = time.time()

    def cancel(self):
        self.record = None

    def stitch(self, recording_folder):
        if not self.record or self.record["end"] is None:
            raise RuntimeError("Complete a scan before stitching.")
        if not recording_folder:
            raise ValueError("A recording folder is required for stitching.")
        source = Path(recording_folder).resolve()
        from patcherbot.utils.scanStitcher import ScanStitcher
        from patcherbot.utils.scanStitcherFast import FastStitchConfig

        destination = Path(__file__).resolve().parents[4] / "experiments" / "Data" / "scan_stitch_data" / source.name
        prefix = "scan_" + uuid4().hex
        stitcher = ScanStitcher(
            date_time_folder=str(source), use_calibration=False,
            scan_time_window=(self.record["start"], self.record["end"]),
            flip_x=self.record["flip_x"], flip_y=self.record["flip_y"],
            downsample=1, use_pixel_drift_stitch=True,
        )
        stitcher.calibration = {"stage": self.record["stage"]}
        stitcher.use_calibration = True
        stitcher.calibration_file = None
        destination.mkdir(parents=True, exist_ok=True)
        return stitcher.stitch_fast(output_path=str(destination / (prefix + ".tif")),
                                    config=FastStitchConfig(refine_images=True))
