"""
Configuration module.

Configuration in this module is taken from yaml files.

Functions wrapped by :py:func:`parse_config` have their kwargs automatically patched by
values in the config.

The ``CONFIG`` dictionary contains all values, structured by submodules and functions. For example,
for function ``function`` defined in ``module.py`` of this package, the config would look like:

.. code-block:: yaml

    module:
        function:
            ...kwargs...


The :py:func:`load_config` function fills the ``CONFIG`` dict from yaml files.
It always updates the dictionary, so the latest file read has the highest priority.

At calling time, the priority order is always (from highest to lowest priority):

1. Explicitly passed keyword-args
2. Values in the loaded config
3. Function's default values.

Special sections
~~~~~~~~~~~~~~~~
After parsing the files, :py:func:`load_config` will look into the config and perform some
extra actions when finding the following special sections:

- ``logging``:
  The content of this section will be sent directly to :py:func:`logging.config.dictConfig`.
- ``xarray``:
  The content of this section will be sent directly to :py:func:`xarray.set_options`.
- ``xclim``:
  The content of this section will be sent directly to :py:func:`xclim.set_options`.
  Here goes `metadata_locales: - fr` to activate the automatic translation of added attributes, for example.
- ``warnings``:
  The content of this section must be a simple mapping. The keys are understood as python
  warning categories (types) and the values as an action to add to the filter. The key "all"
  applies the filter to any warnings. Only built-in warnings are supported.
"""

import ast
import builtins
import collections.abc
import inspect
import logging.config
import types
import warnings
from copy import deepcopy
from functools import wraps
from pathlib import Path
from typing import Any

import xarray as xr
import xclim as xc
import yaml


logger = logging.getLogger(__name__)
EXTERNAL_MODULES = ["logging", "xarray", "xclim", "warnings"]

__all__ = [
    "CONFIG",
    "args_as_str",
    "load_config",
    "parse_config",
    "recursive_update",
]


class ConfigDict(dict):
    """A special dictionary that returns a copy on getitem."""

    def __getitem__(self, key):
        value = super().__getitem__(key)
        if isinstance(value, collections.abc.Mapping):
            return ConfigDict(deepcopy(value))
        return value

    def set(self, key, value):
        parts = key.split(".")
        d = self
        for part in parts[:-1]:
            d = d.setdefault(part, {})
            if not isinstance(d, collections.abc.Mapping):
                raise ValueError(f"Key {key} points to an invalid config section ({part} if not a mapping).")
        d[parts[-1]] = value

    def update_from_list(self, pairs):
        for key, valstr in pairs:
            try:
                val = ast.literal_eval(valstr)
            except (SyntaxError, ValueError):
                val = valstr
            self.set(key, val)


CONFIG = ConfigDict()


def recursive_update(d, other):
    """
    Update a dictionary recursively with another dictionary.

    Values that are Mappings are updated recursively as well.
    """
    for k, v in other.items():
        if isinstance(v, collections.abc.Mapping):
            old_v = d.get(k)
            if isinstance(old_v, collections.abc.Mapping):
                d[k] = recursive_update(old_v, v)
            else:
                d[k] = v
        else:
            d[k] = v
    return d


def args_as_str(*args: tuple[Any, ...]) -> tuple[str, ...]:
    """Return arguments as strings."""
    new_args = []
    for _i, arg in enumerate(*args):
        if isinstance(arg, Path):
            new_args.append(str(arg))
        else:
            new_args.append(arg)
    return tuple(new_args)


def load_config(
    *elements,
    reset: bool = False,
    encoding: str | None = None,
    verbose: bool = False,
):
    """
    Load configuration from given files or key=value pairs.

    Once all elements are loaded, special sections are dispatched to their module, but only if
    the section was changed by the loaded elements. These special sections are:

    * `locales` : The locales to use when writing metadata in xscen, xclim and figanos. This section must be a list of 2-char strings.
    * `logging` : Everything passed to :py:func:`logging.config.dictConfig`.
    * `xarray` : Passed to :py:func:`xarray.set_options`.
    * `xclim` : Passed to :py:func:`xclim.set_options`.
    * `warning` : Mappings where the key is a Warning category (or "all") and the value an action to pass to :py:func:`warnings.simplefilter`.

    Parameters
    ----------
    elements : str
        Files or values to add into the config.
        If a directory is passed, all `.yml` files of this directory are added, in alphabetical order.
        If a "key=value" string, "key" is a dotted name and value will be evaluated if possible.
        "key=value" pairs are set last, after all files are being processed.
    reset : bool
        If True, erases the current config before loading files.
    encoding : str, optional
        The encoding to use when reading files.
    verbose: bool
        If True, each element triggers a INFO log line.

    Example
    -------
    .. code-block:: python

        load_config("my_config.yml", "config_dir/", "logging.loggers.xscen.level=DEBUG")

    Will load configuration from `my_config.yml`, then from all yml files in `config_dir`
    and then the logging level of xscen's logger will be set to DEBUG.
    """
    if reset:
        CONFIG.clear()

    old_external = [deepcopy(CONFIG.get(module, {})) for module in EXTERNAL_MODULES]

    # Use of map(Path, ...) ensures that "file" is a Path, no matter if a Path or a str was given.
    for element in elements:
        if "=" in element:
            key, value = element.split("=")
            CONFIG.update_from_list([(key, value)])
            if verbose:
                msg = f"Updated the config with {element}."
                logger.info(msg)
        else:
            file = Path(element)
            if file.is_dir():
                # Get all yml files, sort by name
                configfiles = sorted(file.glob("*.yml"), key=lambda p: p.name)
            else:
                configfiles = [file]

            for configfile in configfiles:
                with configfile.open(encoding=encoding) as f:
                    recursive_update(CONFIG, yaml.safe_load(f))
                    if verbose:
                        msg = f"Updated the config with {configfile}."
                        logger.info(msg)

    for module, old in zip(EXTERNAL_MODULES, old_external, strict=False):
        if old != CONFIG.get(module, {}):
            _setup_external(module, CONFIG.get(module, {}))


def parse_config(func_or_cls):  # noqa: D103
    module = ".".join(func_or_cls.__module__.split(".")[1:])

    if isinstance(func_or_cls, type):
        func = func_or_cls.__init__
    else:
        func = func_or_cls

    @wraps(func)
    def _wrapper(*args, **kwargs):
        # Get dotted module name, excluding the main package name.

        from_config = CONFIG.get(module, {}).get(func.__name__, {})
        sig = inspect.signature(func)
        if CONFIG.get("print_it_all"):
            msg = f"For func {func}, found config {from_config}.\nOriginal kwargs : {kwargs}"
            logger.debug(msg)
        for k, v in from_config.items():
            if k in sig.parameters:
                kwargs.setdefault(k, v)
        if CONFIG.get("print_it_all"):
            msg = f"Modified kwargs : {kwargs}"
            logger.debug(msg)

        return func(*args, **kwargs)

    _wrapper.configurable = True
    if isinstance(func_or_cls, type):
        func_or_cls.__init__ = _wrapper
        return func_or_cls
    _wrapper: func_or_cls  # Fix a decorator bug in Pycharm 2022
    return _wrapper


def _setup_external(module, config):
    if module == "logging":
        config.update(version=1)
        logging.config.dictConfig(config)
    elif module == "xclim":
        xc.set_options(**config)
    elif module == "xarray":
        xr.set_options(**config)
    elif module == "warning":
        for category, action in config.items():
            if category == "all":
                warnings.simplefilter(action)
            elif issubclass(getattr(builtins, category), builtins.Warning):
                warnings.simplefilter(action, category=getattr(builtins, category))


def get_configurable():
    """Return a dictionary of all configurable functions and classes of xscen."""
    import xscen as xs

    configurable = {}
    for module in dir(xs):
        modobj = getattr(xs, module)
        if isinstance(modobj, types.ModuleType):
            for func in dir(modobj):
                funcobj = getattr(modobj, func)
                if getattr(funcobj, "configurable", False) or getattr(getattr(funcobj, "__init__", None), "configurable", False):
                    configurable[f"xscen.{module}.{func}"] = funcobj
    return configurable
