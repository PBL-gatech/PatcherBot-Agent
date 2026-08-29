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

    Args:
        value: The input value to be converted.

    Returns:
        A YAML-safe version of the input value with compatible Python types.
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

    Args:
        parameter (param.Parameter): The parameter definition associated with the value.
        value: The value to be coerced.

    Returns:
        The value converted to the appropriate type if necessary.
    """
    if isinstance(parameter, param.Tuple) and isinstance(value, list):
        return tuple(value)
    if isinstance(parameter, param.List) and isinstance(value, tuple):
        return list(value)
    return value

class NumberWithUnit(param.Number):
    """Extension of param.Number that includes unit and magnitude metadata."""
    __slots__ = ['unit', 'magnitude']

    def __init__(self, default, unit, magnitude=1.0, *args, **kwds):
        """
        Initialize a NumberWithUnit parameter with an associated unit and magnitude.

        Args:
            default (float): The default numeric value.
            unit (str): The unit associated with the value.
            magnitude (float, optional): Scaling factor applied to the value. Defaults to 1.0.
            *args: Additional positional arguments passed to param.Number.
            **kwds: Additional keyword arguments passed to param.Number.
        """
        super(NumberWithUnit, self).__init__(default=default, *args, **kwds)
        self.unit = unit
        self.magnitude = magnitude


class Config(param.Parameterized):
    """
    Configuration container based on param.Parameterized with support for
    change callbacks and YAML serialization.
    """
    def __init__(self, value_changed=None, *args, **kwds):
        """
        Initialize the Config object.

        Args:
            value_changed (callable, optional): Callback triggered on value changes.
            *args: Additional positional arguments.
            **kwds: Additional keyword arguments.
        """
        super(Config, self).__init__(*args, **kwds)
        self._value_changed = value_changed

    def __setattr__(self, key, value):
        """
        Set an attribute and trigger the value_changed callback if applicable.

        Args:
            key (str): Attribute name.
            value: Value to assign.
        """
        super(Config, self).__setattr__(key, value)
        if not key.startswith('_') and getattr(self, '_value_changed', None) is not None:
            self._value_changed(key, value)
            logging.debug('Config value changed: %s = %s', key, value)

    def to_dict(self):
        """
        Convert configuration parameters to a dictionary.

        Returns:
            dict: Dictionary of parameter names and their values.
        """
        return {name: getattr(self, name) for name in self.param
                if name != 'name'}

    def from_dict(self, config_dict):
        """
        Load configuration values from a dictionary.

        Args:
            config_dict (dict): Dictionary containing configuration values.
        """
        if config_dict is None:
            return
        for name, value in config_dict.items():
            parameter = self.param[name] if name in self.param else None
            coerced_value = _coerce_parameter_value(parameter, value) if parameter else value
            setattr(self, name, coerced_value)

    def to_file(self, filename):
        """
        Save configuration to a YAML file.

        Args:
            filename (str): Path to the output file.
        """
        config_dict = self.to_dict()
        yaml_ready = _yaml_safe_value(config_dict)
        with open(filename, 'w') as f:
            yaml.safe_dump(yaml_ready, f, sort_keys=False)

    def from_file(self, filename):
        """
        Load configuration from a YAML file. Falls back to unsafe loading if needed,
        then rewrites the file in a safe format.

        Args:
            filename (str): Path to the input file.

        Raises:
            yaml.constructor.ConstructorError: If YAML parsing fails and cannot be recovered.
        """
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
