import logging
from functools import partial
from pathlib import Path, PosixPath
from types import ModuleType
from typing import Sequence, Tuple, Union

import pandas as pd
import xarray as xr
import xclim as xc
from intake_esm import DerivedVariableRegistry
from xclim.core.indicator import Indicator
from yaml import safe_load

from . import CV
from .config import parse_config

logger = logging.getLogger(__name__)


def ensure_list(x):
    if not isinstance(x, (list, tuple)):
        return [x]
    return x


def load_xclim_module(filename, reload=False):
    """Return the xclim module described by the yaml file (or group of yaml, jsons and py).

    Parameters
    ----------
    filename : pathlike
      The filepath to the yaml file of the module or to the stem of yaml, jsons and py files.
    reload : bool
      If False (default) and the module already exists in `xclim.indicators`, it is not re-build.
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


@parse_config
def compute_indicators(
    ds: xr.Dataset,
    indicators: Union[
        str, PosixPath, Sequence[Indicator], Sequence[Tuple[str, Indicator]], ModuleType
    ],
    *,
    to_level: str = "indicators",
) -> Union[dict, xr.Dataset]:
    """
    Calculates variables and indicators based on a YAML call to xclim.

    Parameters
    ----------
    ds : xr.Dataset
      Dataset to use for the indicators.
    indicators : Union[str, PosixPath, Sequence[Indicator], Sequence[Tuple[str, Indicator]]]
      Path to a YAML file that instructs on how to calculate missing variables.
      Can also be only the "stem", if translations and custom indices are implemented.
      Can be the indicator module directly, or a sequence of indicators or a sequence of
      tuples (indicator name, indicator) as returned by `iter_indicators()`.
    to_level : str, optional
      The processing level to assign to the output.
      If None, the processing level of the inputs is preserved.


    Returns
    -------
    dict, xr.Dataset
      Dictionary (keys = timedeltas) with indicators separated by temporal resolution.
      If there is a single timestep, a Dataset will be returned instead.

    """
    if isinstance(indicators, (str, Path)):
        logger.debug("Loading indicator module.")
        module = load_xclim_module(indicators)
        indicators = module.iter_indicators()
    elif hasattr(indicators, "iter_indicators"):
        indicators = indicators.iter_indicators()

    try:
        N = len(indicators)
    except TypeError:
        N = None
    else:
        logger.info(f"Computing {N} indicators.")

    out_dict = dict()
    for i, ind in enumerate(indicators, 1):
        if isinstance(ind, tuple):
            iden, ind = ind
        else:
            iden = ind.identifier
        logger.info(f"{i} - Computing {iden}.")

        # Make the call to xclim
        out = ind(ds=ds)

        # Infer the indicator's frequency
        freq = xr.infer_freq(out.time) if "time" in out.dims else "fx"

        # Create the dictionary key
        key = freq
        if key not in out_dict:
            if isinstance(out, tuple):  # In the case of multiple outputs
                out_dict[key] = xr.merge(o for o in out if o.name in indicators)
            else:
                out_dict[key] = out.to_dataset()

            # TODO: Double-check History, units, attrs, add missing variables (grid_mapping), etc.
            out_dict[key].attrs = ds.attrs
            out_dict[key].attrs.pop("cat/variable")
            out_dict[key].attrs["cat/xrfreq"] = freq
            out_dict[key].attrs["cat/frequency"] = CV.xrfreq_to_frequency(freq, None)
            out_dict[key].attrs["cat/timedelta"] = pd.to_timedelta(
                CV.xrfreq_to_timedelta(freq, None)
            )
            if to_level is not None:
                out_dict[key].attrs["cat/processing_level"] = to_level

        else:
            if isinstance(out, tuple):  # In the case of multiple outputs
                for o in out:
                    if o.name in indicators:
                        out_dict[key][o.name] = o
            else:
                out_dict[key][out.name] = out

    return out_dict


def derived_func(ind: xc.core.indicator.Indicator, nout: int):
    def func(ds, *, ind, nout):
        out = ind(ds=ds)
        if isinstance(out, tuple):
            out = out[nout]
        ds[out.name] = out
        return ds

    func.__name__ = ind.identifier
    return partial(func, ind=ind, nout=nout)


def registry_from_module(module, registry=None, variable_column="variable"):
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
