"""
Module providing pipette detection, focuser utilities, and lazy-loading cell segmentors.

This module exports the following classes:

- PipetteDetector, PipetteDetector1, PipetteDetector2: Classes for detecting pipette positions in images.
- PipetteFocuser: Class for controlling pipette focus.
- CellSegmentor2, CellSegmentor3: Lazily loaded classes for cell segmentation (imported on demand).

Lazy loading of CellSegmentor2 and CellSegmentor3 is implemented via __getattr__ to reduce
startup overhead and avoid circular imports.
"""
from __future__ import absolute_import

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
