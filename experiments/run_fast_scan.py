"""Run the full scan with GPU feather blending; edit constants, no CLI required.

Use the cloned patcherbot-stitch-gpu Conda environment. Outputs go to a new
file in the existing session's flat derived-output directory. Choose a new
OUTPUT_NAME for another run; existing output files are protected from overwrite.
"""
import time
STARTED = time.perf_counter()

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "patcherbot" / "utils"))
from scanStitcher import ScanStitcher
from scanStitcherFast import FastStitchConfig

SESSION = "2026_09_23-18_05"
DATA_ROOT = ROOT / "experiments" / "Data" / "rig_recorder_data"
OUTPUT_ROOT = ROOT / "experiments" / "Data" / "scan_stitch_data"
OUTPUT_NAME = "scan_0001.tif"  # Choose a new prefix for another run in this session.
CONFIG = FastStitchConfig(backend="cupy", select_frames=True, preserve_coverage=True, blend="feather", preprocessing=None)


def main():
    output = OUTPUT_ROOT / SESSION
    stitcher = ScanStitcher(
        date_time_folder=SESSION, date_time_root=str(DATA_ROOT),
        use_calibration=True, flip_x=True, flip_y=True, downsample=1,
    )
    saved = stitcher.stitch_fast(output_path=str(output / OUTPUT_NAME), config=CONFIG)
    elapsed = time.perf_counter() - STARTED
    print(f"Saved: {saved}")
    print(f"Total including Python imports: {elapsed:.3f} seconds")
    return saved


if __name__ == "__main__":
    main()
