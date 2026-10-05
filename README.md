# PatcherBot-Agent


This code has its basis in [HolyPipette](https://github.com/romainbrette/holypipette). Many 
Thanks to the original contributors.

See the [documentation](docs/index.rst) for installation and usage.

## Conda setup and install order
1. `conda create -n patcherbot python=3.10.11`
2. `conda activate patcherbot`
3. `pip install -r requirements_runtime.txt` (core runtime deps)
4. Ensure robomimic is already cloned into `patcherbot/deepLearning/patchModel/robomimic`.
5. `pip install -e patcherbot/deepLearning/patchModel/robomimic`
6. `pip install -e .`

Notes:
- SAM2 segmentation and LightGlue matching are loaded via Hugging Face Transformers (`transformers>=4.56.0`; models download on first use).

## Fake pipette calibration

Run `python testing/fake_rig_console.py` in the `patcher_bot` environment to move the fake rig without the GUI. Commands use micrometers, with one-based pipette indices and `x`, `y`, `z` axes; for example, `move stage z 100` and `move pipette 2 z -25`. Use `set` instead of `move` for an absolute position. The fake stage begins at Z offset 0 relative to the camera's reference; the configured focus plane is Z=100 um.

The active fake-rig geometry is in `rig_setup/rig_configs/fake_rig.json`: each pipette's initial controller position determines its tip location and approach direction relative to the image center. The stage start position is configured in the JSON and must match `FAKE_STAGE_REFERENCE_POSITION` in `FakeCalCamera.py`.

To inspect the detector result on any image, run `python testing/preview_pipette_detector.py path/to/image.png`. `PipetteDetectorYOLO1.detect_pipette()` returns the highest-confidence detected tip; YOLO does not know which pipette is selected, so with multiple tips in frame it may return another pipette. The repository contains synthetic camera assets in `patcherbot/devices/camera/FakeMicroscopeImgs`, but no detector test-photo set; pass a real or mirrored image as the input.
