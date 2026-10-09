from __future__ import absolute_import

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

__all__ = [
    "PipetteDetector",
    "PipetteDetector1",
    "PipetteDetectorYOLO1",
    "PipetteDetector2",
    "PipetteFocuser",
    "CellSegmentor2",
    "CellSegmentor3",
]

if TYPE_CHECKING:
    from .cellSegmentor import CellSegmentor2  # pragma: no cover
    from .pipetteDetector import PipetteDetector, PipetteDetector1, PipetteDetector2, PipetteDetectorYOLO1  # pragma: no cover
    from .pipetteFocuser import PipetteFocuser  # pragma: no cover


def __getattr__(name: str):
    if name in {"PipetteDetector", "PipetteDetector1", "PipetteDetector2", "PipetteDetectorYOLO1"}:
        from .pipetteDetector import PipetteDetector, PipetteDetector1, PipetteDetector2, PipetteDetectorYOLO1

        globals().update({
            "PipetteDetector": PipetteDetector,
            "PipetteDetector1": PipetteDetector1,
            "PipetteDetector2": PipetteDetector2,
            "PipetteDetectorYOLO1": PipetteDetectorYOLO1,
        })
        return globals()[name]
    if name == "PipetteFocuser":
        from .pipetteFocuser import PipetteFocuser

        globals()["PipetteFocuser"] = PipetteFocuser
        return globals()[name]
    if name in {"CellSegmentor2", "CellSegmentor3"}:
        from .cellSegmentor import CellSegmentor2
        globals().update({
            "CellSegmentor2": CellSegmentor2,
            "CellSegmentor3": CellSegmentor2,
        })
        return globals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def _import_module_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return module

