'''
Support for configuration objects (based on the param package)
'''
import warnings
import yaml
import logging
import param
from yaml.constructor import ConstructorError
from param import Number, Boolean, Selector, Tuple, List  # to make it available for import

try:
    import numpy as _np
except ImportError:
    _np = None


def _yaml_safe_value(value):
    """
    Convert values to YAML-safe types (no numpy scalars/arrays or tuples).
    """
    if isinstance(value, dict):
        return {k: _yaml_safe_value(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_yaml_safe_value(v) for v in value]
    if isinstance(value, list):
        return [_yaml_safe_value(v) for v in value]
    if _np is not None:
        if isinstance(value, _np.ndarray):
            return [_yaml_safe_value(v) for v in value.tolist()]
        if isinstance(value, _np.generic):
            return value.item()
    return value


def _coerce_parameter_value(parameter, value):
    """
    Coerce deserialized values back to the expected param.Parameter type.
    """
    if isinstance(parameter, param.Tuple) and isinstance(value, list):
        return tuple(value)
    if isinstance(parameter, param.List) and isinstance(value, tuple):
        return list(value)
    return value

class NumberWithUnit(param.Number):
    __slots__ = ['unit', 'magnitude']

    def __init__(self, default, unit, magnitude=1.0, *args, **kwds):
        super(NumberWithUnit, self).__init__(default=default, *args, **kwds)
        self.unit = unit
        self.magnitude = magnitude


class Config(param.Parameterized):
    def __init__(self, value_changed=None, *args, **kwds):
        super(Config, self).__init__(*args, **kwds)
        self._value_changed = value_changed

    def __setattr__(self, key, value):
        super(Config, self).__setattr__(key, value)
        if not key.startswith('_') and getattr(self, '_value_changed', None) is not None:
            self._value_changed(key, value)
            logging.debug('Config value changed: %s = %s', key, value)

    def to_dict(self):
        return {name: getattr(self, name) for name in self.param
                if name != 'name'}

    def from_dict(self, config_dict):
        if config_dict is None:
            return
        for name, value in config_dict.items():
            parameter = self.param[name] if name in self.param else None
            coerced_value = _coerce_parameter_value(parameter, value) if parameter else value
            setattr(self, name, coerced_value)

    def to_file(self, filename):
        config_dict = self.to_dict()
        yaml_ready = _yaml_safe_value(config_dict)
        with open(filename, 'w') as f:
            yaml.safe_dump(yaml_ready, f, sort_keys=False)

    def from_file(self, filename):
        with open(filename, 'r') as f:
            try:
                config_dict = yaml.safe_load(f)
            except ConstructorError:
                f.seek(0)
                unsafe_loader = getattr(yaml, 'UnsafeLoader', yaml.Loader)
                config_dict = yaml.load(f, Loader=unsafe_loader)
                config_dict = _yaml_safe_value(config_dict)
                self.from_dict(config_dict)
                self.to_file(filename)
                return
        self.from_dict(config_dict)
