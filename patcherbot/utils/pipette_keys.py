"""Global pipette keys.

Every per-pipette device and every piece of per-pipette data is keyed by
``pipette_<n>`` (n counts from 0 in rig order). The DAQ maps these keys to
its own analog channel names internally.
"""
import re

__all__ = ["pipette_key_for", "pipette_index", "pipette_label"]

_KEY_PATTERN = re.compile(r"pipette_(\d+)")


def pipette_key_for(index):
    """Key for the pipette at position ``index`` (0-based)."""
    return f"pipette_{int(index)}"


def pipette_index(key):
    """Position of ``key`` (``"pipette_2"`` -> 2), or None if it isn't a pipette key."""
    match = _KEY_PATTERN.fullmatch(str(key))
    return int(match.group(1)) if match else None


def pipette_label(key):
    """Display name for ``key`` (``"pipette_0"`` -> ``"Pipette 0"``)."""
    index = pipette_index(key)
    return f"Pipette {index}" if index is not None else str(key)
