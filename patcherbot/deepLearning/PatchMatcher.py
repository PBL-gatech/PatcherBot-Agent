from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch

try:
    from .PointMatcher import PointMatcher
except ImportError:  # pragma: no cover - allow running as a script
    from PointMatcher import PointMatcher

ImageInput = Union[str, Path, torch.Tensor]
MatcherConfig = Dict[str, object]


class PatchMatcher:
    """LightGlue wrapper that exposes the target point derived from center alignment."""

    def __init__(self, **matcher_kwargs: object) -> None:
        """
        Forward kwargs to ``PointMatcher`` for configuration.
        
        Args:
            **matcher_kwargs: Keyword arguments forwarded to ``PointMatcher`` 
                for configuration (e.g., device, model parameters).
        """
        self._matcher = PointMatcher(**matcher_kwargs)

    @staticmethod
    def _to_float_pair(value: Union[Sequence[Union[int, float]], torch.Tensor]) -> Tuple[float, float]:
        """
        Convert a sequence or tensor into a 2D float tuple.

        Args:
            value: A length-2 sequence or tensor representing (x, y).

        Returns:
            tuple[float, float]: Converted (x, y) pair.

        Raises:
            TypeError: If the input is not a sequence or tensor.
            ValueError: If the input does not have exactly 2 elements.
        """
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().tolist()
        if not isinstance(value, Sequence):
            raise TypeError("Expected a length-2 sequence or tensor for a point.")
        if len(value) != 2:
            raise ValueError("Expected a length-2 sequence for a point.")
        return float(value[0]), float(value[1])

    def find_target(
        self,
        reference_image: ImageInput,
        current_image: ImageInput,
        *,
        current_center: Optional[Union[Sequence[Union[int, float]], torch.Tensor]] = None,
        load_conf: Optional[MatcherConfig] = None,
        **preprocess: object,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Return the target point computed from the center displacement.
        
        Args:
            reference_image: Reference image (path or tensor).
            current_image: Current image (path or tensor).
            current_center: Optional (x, y) center of the current image.
                If not provided, the method attempts to use the center
                returned by the matcher.
            load_conf: Optional configuration dictionary passed to the matcher.
            **preprocess: Additional preprocessing arguments forwarded to
                ``PointMatcher.match``.

        Returns:
            dict: Dictionary containing:
                - "target_point": (x, y) computed target location
                - "current_center": (x, y) center used
                - "translation": (dx, dy) rigid translation
                - "displacement": (dx, dy) raw displacement estimate

        Raises:
            RuntimeError:
                - If the matcher does not return a "center_shift".
                - If translation information is missing from the result.
            ValueError:
                - If ``current_center`` is not provided and cannot be inferred.
                - If ``current_center`` is not a valid length-2 coordinate.
        """
        """Return the target point computed from the center displacement."""
        preprocess.setdefault("include_overlay", False)
        result = self._matcher.match(reference_image, current_image, load_conf=load_conf, **preprocess)
        shift = result.get("center_shift")
        if not shift:
            raise RuntimeError("LightGlue did not return a center shift; cannot compute target point.")

        displacement = shift.get("center_displacement")
        translation_dx = shift.get("center_dx")
        translation_dy = shift.get("center_dy")

        if translation_dx is None or translation_dy is None:
            if displacement is None:
                raise RuntimeError("LightGlue centre shift did not include translation information.")
            translation_pair = (float(displacement["dx"]), float(displacement["dy"]))
        else:
            translation_pair = (float(translation_dx), float(translation_dy))

        if displacement is None:
            displacement_pair = translation_pair
        else:
            displacement_pair = (float(displacement["dx"]), float(displacement["dy"]))

        dx, dy = translation_pair

        if current_center is None:
            current_center = shift.get("center1")
            if current_center is None:
                raise ValueError(
                    "The current center must be provided when LightGlue does not expose the second image size."
                )

        cx, cy = self._to_float_pair(current_center)
        target_point = (cx + dx, cy + dy)

        return {
            "target_point": target_point,
            "current_center": (cx, cy),
            "translation": translation_pair,
            "displacement": displacement_pair,
        }


if __name__ == "__main__":
    import pprint
    reference_image = Path(r"C:\Users\sa-forest\Documents\GitHub\LightGlue\ex_data\cell_4.webp")
    current_image = Path(r"C:\Users\sa-forest\Documents\GitHub\LightGlue\ex_data\36712_1760473097.769556.webp")
    # reference_image = Path(r"C:\Users\sa-forest\Documents\GitHub\LightGlue\ex_data\88602_1760469071.915733.webp")
    # current_image = Path(r"C:\Users\sa-forest\Documents\GitHub\LightGlue\ex_data\106826_1760469696.866317.webp")

    patch_matcher = PatchMatcher()
    match_info = patch_matcher.find_target(reference_image, current_image)
    pprint.pprint(match_info)
