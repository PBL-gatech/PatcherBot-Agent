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
