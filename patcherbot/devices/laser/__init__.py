from __future__ import absolute_import

from .laser import FakeLaser, Laser
try:
    from .lumencor import LumencorLaser
except Exception:  # pragma: no cover - optional serial dependency
    LumencorLaser = None

__all__ = ["Laser", "FakeLaser", "LumencorLaser"]
