"""Multi-pipette bookkeeping shared by devices, controllers and interfaces.

Every per-pipette device and every piece of per-pipette data is keyed by
``pipette_<n>`` (n counts from 0 in rig order). This module owns everything
that maps those keys onto hardware or state, so device classes only have to
talk to their hardware:

* keys and labels (``pipette_key_for``, ``pipette_index``, ``pipette_label``)
* picking a pipette's device out of a per-pipette slot (``device_for_pipette``)
* per-pipette state (``PER_PIPETTE_FIELDS``, ``PipetteState``)
* ``MultiPipetteDevice``: base class for one device serving several pipettes
* the DAQ channel model (``PipetteChannels``, ``build_channel_map``)
* DAQ method tags (``multi_pipette_protocol``, ``per_pipette``) and
  ``PipetteDAQView``, one pipette's handle on the shared DAQ

Nothing here imports device modules; devices are used duck-typed.
"""
import collections
import functools
import inspect
import re
from dataclasses import dataclass

__all__ = [
    "pipette_key_for", "pipette_index", "pipette_label", "validate_pipette_keys",
    "device_for_pipette",
    "PER_PIPETTE_FIELDS", "PipetteState",
    "MultiPipetteDevice",
    "PipetteChannels", "build_channel_map", "enumerated_channels",
    "multi_pipette_protocol", "takes_pipettes", "per_pipette", "PipetteDAQView",
]


# ========================================================
#  Keys
# ========================================================
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


def validate_pipette_keys(pipette_keys=None):
    """Pipette keys as a list of strings in rig order (default ``[pipette_0]``).

    Raises ValueError on duplicates.
    """
    keys = [str(key) for key in pipette_keys] if pipette_keys is not None else []
    if not keys:
        keys = [pipette_key_for(0)]
    if len(set(keys)) != len(keys):
        raise ValueError(f"Duplicate pipette keys: {keys}")
    return keys


# ========================================================
#  Per-pipette devices
# ========================================================
def device_for_pipette(devices, key):
    """The device for pipette ``key`` out of a per-pipette slot.

    ``devices`` is ``{pipette_key: device}``; a single device serves every
    pipette. A device that serves several pipettes through separate parts
    (e.g. a MultiClampAmplifier with one headstage per pipette) provides
    ``for_pipette(key)``, which returns this pipette's part.
    """
    if key in devices:
        device = devices[key]
    elif len(devices) == 1:
        device = next(iter(devices.values()))
    else:
        raise KeyError(f"No device configured for {key}; available: {list(devices)}")
    for_pipette = getattr(device, "for_pipette", None)
    return for_pipette(key) if callable(for_pipette) else device


# ========================================================
#  Per-pipette state
# ========================================================
# State the DAQ keeps separately for each pipette. A callable default is a
# factory, called once per pipette (so each pipette gets its own queue).
PER_PIPETTE_FIELDS = {
    "cellMode": False,
    "totalResistance": None,
    "latestAccessResistance": None,
    "latestMembraneResistance": None,
    "latestMembraneCapacitance": None,
    "holding_current": None,
    "holding_voltage": None,
    "pulses": None,
    "pulseRange": None,
    "current_protocol_data": None,
    "voltage_protocol_data": None,
    "voltage_command_data": None,
    "voltage_membrane_test": None,
    "voltageTotalResistance": None,
    "voltageMembraneResistance": None,
    "voltageAccessResistance": None,
    "voltageMembraneCapacitance": None,
    "vclamp_steps": None,
    "vclamp_hold_value": None,
    "leak_subtraction_data": None,
    "leak_subtraction_meta": None,
    "holding_protocol_data": None,
    "optogenetic_protocol_data": None,
    "optogenetic_stim_data": None,
    "optogenetic_protocol_type": None,
    "optogenetic_pulses": None,
    "optogenetic_pulseRange": None,
    "optogenetic_protocol_queue": collections.deque,
}


class PipetteState:
    """One pipette's state: a fixed set of attributes with defaults.

    ``fields`` maps attribute name -> default (a callable default is called
    once to build this pipette's own value). Setting an attribute that isn't
    a declared field raises AttributeError, so a typo can't silently create
    new state.
    """

    def __init__(self, fields=None):
        fields = PER_PIPETTE_FIELDS if fields is None else fields
        object.__setattr__(self, "_fields", tuple(fields))
        for name, default in fields.items():
            object.__setattr__(self, name, default() if callable(default) else default)

    @property
    def fields(self):
        return self._fields

    def __setattr__(self, name, value):
        if name not in self._fields:
            raise AttributeError(f"{name!r} is not a per-pipette field")
        object.__setattr__(self, name, value)

    def as_dict(self):
        return {name: getattr(self, name) for name in self._fields}

    def __repr__(self):
        return f"PipetteState({len(self._fields)} fields)"


class MultiPipetteDevice:
    """Base class for one device that serves several pipettes.

    Holds the pipette keys the device serves (rig order) and, optionally, one
    ``PipetteState`` per pipette. Subclasses implement ``for_pipette(key)``,
    returning the part or handle that drives that pipette.

    ``__init__`` does not call ``super().__init__``; subclasses that also
    inherit another base (e.g. TaskController) call each base explicitly.
    """

    def __init__(self, pipette_keys=None, state_fields=None):
        self._pipette_keys = validate_pipette_keys(pipette_keys)
        self._pipette_states = (
            {key: PipetteState(state_fields) for key in self._pipette_keys}
            if state_fields is not None else {}
        )

    @property
    def pipette_keys(self):
        return list(self._pipette_keys)

    @property
    def primary_pipette_key(self):
        return self._pipette_keys[0]

    def require_pipette(self, key):
        """Return ``key`` if this device serves it, else raise KeyError."""
        if key not in self._pipette_keys:
            raise KeyError(f"{type(self).__name__} has no pipette {key!r}; "
                           f"known pipettes: {self._pipette_keys}")
        return key

    def resolve_pipettes(self, pipettes=None):
        """Normalize a ``pipettes`` argument to a validated list of keys
        (duplicates dropped, order kept). None means the primary pipette."""
        if pipettes is None:
            return [self.primary_pipette_key]
        keys = [pipettes] if isinstance(pipettes, str) else list(dict.fromkeys(pipettes))
        for key in keys:
            self.require_pipette(key)
        return keys

    def pipette_state(self, key=None):
        """The PipetteState of ``key`` (default: the primary pipette)."""
        key = self.primary_pipette_key if key is None else key
        try:
            return self._pipette_states[key]
        except KeyError:
            raise KeyError(f"{type(self).__name__} keeps no state for pipette {key!r}; "
                           f"known pipettes: {list(self._pipette_states)}") from None

    def for_pipette(self, key):
        raise NotImplementedError


# ========================================================
#  DAQ channel model
# ========================================================
@dataclass(frozen=True)
class PipetteChannels:
    """DAQ channel names used by one pipette.

    read: analog input carrying the amplifier's command monitor
    resp: analog input carrying the amplifier's response
    cmd:  analog output that drives the amplifier's command input
    """
    read: str
    resp: str
    cmd: str

    @classmethod
    def from_spec(cls, spec):
        if isinstance(spec, cls):
            return spec
        if not isinstance(spec, dict):
            raise TypeError(f"Channel spec must be a mapping with read/resp/cmd, got {spec!r}")
        missing = {"read", "resp", "cmd"} - set(spec)
        if missing:
            raise ValueError(f"Channel spec {spec!r} is missing {sorted(missing)}")
        return cls(read=str(spec["read"]), resp=str(spec["resp"]), cmd=str(spec["cmd"]))


def build_channel_map(pipette_keys=None, channels=None, default_channels=None):
    """Return an ordered ``{pipette_key: PipetteChannels}`` mapping.

    pipette_keys: pipette names in rig order (defaults to ``["pipette_0"]``).
    channels: optional list of ``{"read", "resp", "cmd"}`` specs, one per pipette.
    default_channels: ``f(index) -> PipetteChannels`` used when ``channels`` is None.
    """
    keys = validate_pipette_keys(pipette_keys)
    if channels is not None:
        specs = list(channels)
        if len(specs) < len(keys):
            raise ValueError(
                f"{len(keys)} pipettes selected but only {len(specs)} channel sets configured"
            )
        specs = [PipetteChannels.from_spec(spec) for spec in specs[:len(keys)]]
    elif default_channels is not None:
        specs = [default_channels(index) for index in range(len(keys))]
    else:
        raise ValueError("Either channels or default_channels must be provided")

    channel_map = collections.OrderedDict(zip(keys, specs))
    names = [name for spec in specs for name in (spec.read, spec.resp, spec.cmd)]
    duplicates = sorted(name for name, count in collections.Counter(names).items() if count > 1)
    if duplicates:
        raise ValueError(f"Channels assigned to more than one role/pipette: {duplicates}")
    return channel_map


def enumerated_channels(index):
    """Default channel names for pipette ``index``: ai{2i} (read), ai{2i+1} (resp), ao{i} (cmd)."""
    return PipetteChannels(f"ai{2 * index}", f"ai{2 * index + 1}", f"ao{index}")


# ========================================================
#  DAQ method tags and the per-pipette view
# ========================================================
def multi_pipette_protocol(method):
    """Protocol that acquires every requested pipette in the same reads.

    The decorated method is written as ``method(self, keys, **kwargs)`` and
    returns ``{pipette_key: result}``. Callers pass ``pipettes=[...]``
    (default: the primary pipette). One protocol call runs at a time; a
    call may record several pipettes simultaneously.
    """
    @functools.wraps(method)
    def wrapper(self, *args, pipettes=None, **kwargs):
        keys = self.resolve_pipettes(pipettes)
        with self._protocol_lock:
            return method(self, keys, *args, **kwargs)
    return takes_pipettes(wrapper)


def takes_pipettes(method):
    """Mark a method that takes ``pipettes=[...]`` (None = primary pipette).

    PipetteDAQView fills in ``[<its pipette>]`` when the caller leaves it None.
    """
    method._pipettes_arg = True
    return method


def per_pipette(method):
    """Mark a method that takes ``pipette_key=`` (None = primary pipette).

    PipetteDAQView fills in its own key when the caller leaves it None.
    """
    method._pipette_key_arg = True
    return method


def _call_with_default(method, name, value, args, kwargs):
    """Call ``method`` with argument ``name`` set to ``value`` if it is missing or None."""
    # follow_wrapped=False: a multi_pipette_protocol wrapper's own signature
    # has ``pipettes=``; the wrapped body's has ``keys`` instead.
    bound = inspect.signature(method, follow_wrapped=False).bind_partial(*args, **kwargs)
    if bound.arguments.get(name) is None:
        bound.arguments[name] = value
    return method(*bound.args, **bound.kwargs)


class PipetteDAQView:
    """One pipette's handle on the single shared DAQ.

    Holds a reference (never a copy). Per-pipette fields (``view.cellMode``,
    ``view.current_protocol_data``...) read and write this pipette's state.
    Methods tagged ``@per_pipette`` get ``pipette_key=<this pipette>`` and
    ``@multi_pipette_protocol`` methods get ``pipettes=[<this pipette>]``
    when the caller doesn't pass them. Everything else is the shared DAQ's.
    Call the DAQ itself with ``pipettes=[...]`` to acquire several pipettes
    together.
    """

    def __init__(self, daq, pipette_key):
        daq.require_pipette(pipette_key)
        object.__setattr__(self, "daq", daq)
        object.__setattr__(self, "pipette_key", pipette_key)

    def __getattr__(self, name):
        daq = self.daq
        key = self.pipette_key
        if name in PER_PIPETTE_FIELDS:
            return getattr(daq.pipette_state(key), name)
        value = getattr(daq, name)
        if getattr(value, "_pipettes_arg", False):
            argument, default = "pipettes", [key]
        elif getattr(value, "_pipette_key_arg", False):
            argument, default = "pipette_key", key
        else:
            return value

        def call_for_pipette(*args, **kwargs):
            return _call_with_default(value, argument, default, args, kwargs)

        call_for_pipette.__name__ = name
        # TaskInterface.execute() runs only methods of a TaskController and
        # finds the owner through __self__, so expose the shared DAQ.
        call_for_pipette.__self__ = daq
        return call_for_pipette

    def __setattr__(self, name, value):
        if name in PER_PIPETTE_FIELDS:
            setattr(self.daq.pipette_state(self.pipette_key), name, value)
        else:
            setattr(self.daq, name, value)

    def __repr__(self):
        return f"PipetteDAQView({type(self.daq).__name__}, {self.pipette_key!r})"
