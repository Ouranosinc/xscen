import logging
from functools import partial
from pathlib import Path
from types import ModuleType

import xclim as xc
from intake_esm import DerivedVariableRegistry
from xclim.core.indicator import Indicator
from yaml import safe_load

logger = logging.getLogger(__name__)


__all__ = []


def ensure_list(x):
    if not isinstance(x, (list, tuple)):
        return [x]
    return x


def load_xclim_module(filename, reload=False) -> ModuleType:
    """Return the xclim module described by the yaml file (or group of yaml, jsons and py).

    Parameters
    ----------
    filename : pathlike
      The filepath to the yaml file of the module or to the stem of yaml, jsons and py files.
    reload : bool
      If False (default) and the module already exists in `xclim.indicators`, it is not re-build.

    Returns
    -------
    ModuleType
    """
    if not reload:
        # Same code as in xclim to get the module name.
        filepath = Path(filename)

        if not filepath.suffix:
            # A stem was passed, try to load files
            ymlpath = filepath.with_suffix(".yml")
        else:
            ymlpath = filepath

        # Read YAML file
        with ymlpath.open() as f:
            yml = safe_load(f)

        name = yml.get("module", filepath.stem)
        if hasattr(xc.indicators, name):
            return getattr(xc.indicators, name)

    return xc.build_indicator_module_from_yaml(filename)


def derived_func(ind: xc.core.indicator.Indicator, nout: int) -> partial:
    def func(ds, *, ind, nout):
        out = ind(ds=ds)
        if isinstance(out, tuple):
            out = out[nout]
        ds[out.name] = out
        return ds

    func.__name__ = ind.identifier
    return partial(func, ind=ind, nout=nout)


def registry_from_module(
    module, registry=None, variable_column="variable"
) -> DerivedVariableRegistry:
    """Converts a xclim virtual indicators module to an intake_esm Derived Variable Registry.

    Parameters
    ----------
    module : ModuleType
      A module of xclim.
    registry : DerivedVariableRegistry, optional
      If given, this registry is extended, instead of creating a new one.
    variable_column : str
      The name of the variable column (the name used in the query).

    Returns
    -------
    DerivedVariableRegistry
      A variable registry where each indicator and each of its output has been registered.
      If an indicator returns multiple values, each of them is mapped individually, as
      the DerivedVariableRegistry only supports single output function.
      Each indicator was wrapped into a new function that only accepts a dataset and
      returns it with the extra variable appended. This means all other parameters are
      given their defaults.
    """
    dvr = registry or DerivedVariableRegistry()
    for name, ind in module.iter_indicators():
        query = {
            variable_column: [p.default for p in ind.parameters.values() if p.kind == 0]
        }
        for i, attrs in enumerate(ind.cf_attrs):
            dvr.register(variable=attrs["var_name"], query=query)(derived_func(ind, i))
    return dvr
