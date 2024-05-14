"""Functions to compute xclim indicators."""

import logging
import os
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Optional, Union

import xarray as xr
import xclim as xc
from intake_esm import DerivedVariableRegistry
from xclim.core.indicator import Indicator
from yaml import safe_load

from xscen.config import parse_config

from .catutils import parse_from_ds
from .utils import CV, standardize_periods

logger = logging.getLogger(__name__)

__all__ = ["compute_indicators", "load_xclim_module", "registry_from_module"]


def load_xclim_module(
    filename: Union[str, os.PathLike], reload: bool = False
) -> ModuleType:
    """Return the xclim module described by the yaml file (or group of yaml, jsons and py).

    Parameters
    ----------
    filename : str or os.PathLike
        The filepath to the yaml file of the module or to the stem of yaml, jsons and py files.
    reload : bool
        If False (default) and the module already exists in `xclim.indicators`, it is not re-build.

    Returns
    -------
    ModuleType
        The xclim module.
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
def compute_indicators(  # noqa: C901
    ds: xr.Dataset,
    indicators: Union[
        str,
        os.PathLike,
        Sequence[Indicator],
        Sequence[tuple[str, Indicator]],
        ModuleType,
    ],
    *,
    periods: Optional[Union[list[str], list[list[str]]]] = None,
    restrict_years: bool = True,
    to_level: Optional[str] = "indicators",
) -> dict:
    """Calculate variables and indicators based on a YAML call to xclim.

    The function cuts the output to be the same years as the inputs.
    Hence, if an indicator creates a timestep outside the original year range (e.g. the first DJF for QS-DEC),
    it will not appear in the output.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to use for the indicators.
    indicators : Union[str, os.PathLike, Sequence[Indicator], Sequence[tuple[str, Indicator]], ModuleType]
        Path to a YAML file that instructs on how to calculate missing variables.
        Can also be only the "stem", if translations and custom indices are implemented.
        Can be the indicator module directly, or a sequence of indicators or a sequence of
        tuples (indicator name, indicator) as returned by `iter_indicators()`.
    periods : list of str or list of lists of str, optional
        Either [start, end] or list of [start, end] of continuous periods over which to compute the indicators.
        This is needed when the time axis of ds contains some jumps in time.
        If None, the dataset will be considered continuous.
    restrict_years : bool
        If True, cut the time axis to be within the same years as the input.
        This is mostly useful for frequencies that do not start in January, such as QS-DEC.
        In that instance, `xclim` would start on previous_year-12-01 (DJF), with a NaN.
        `restrict_years` will cut that first timestep.
        This should have no effect on YS and MS indicators.
    to_level : str, optional
        The processing level to assign to the output.
        If None, the processing level of the inputs is preserved.

    Returns
    -------
    dict
        Dictionary (keys = timedeltas) with indicators separated by temporal resolution.

    See Also
    --------
    xclim.indicators, xclim.core.indicator.build_indicator_module_from_yaml
    """
    if isinstance(indicators, (str, os.PathLike)):
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

    def _infer_freq_from_meta(ind):
        return (
            ind.injected_parameters["freq"]
            if "freq" in ind.injected_parameters
            else (
                ind.parameters["freq"].default
                if "freq" in ind.parameters
                else ind.src_freq
            )
        )

    periods = standardize_periods(periods)

    out_dict = dict()
    for i, ind in enumerate(indicators, 1):
        if isinstance(ind, tuple):
            iden, ind = ind
        else:
            iden = ind.identifier
        logger.info(f"{i} - Computing {iden}.")

        if periods is None:
            # Make the call to xclim
            out = ind(ds=ds)

            # In the case of multiple outputs, merge them into a single dataset
            if isinstance(out, tuple):
                out = xr.merge(out)
                out.attrs = {}
            else:
                out = out.to_dataset()

            # Infer the indicator's frequency
            if "time" in out.dims:
                if len(out.time) < 3:
                    freq = _infer_freq_from_meta(ind)
                else:
                    freq = xr.infer_freq(out.time)
            else:
                freq = "fx"
            if freq == "YS":
                freq = "YS-JAN"

        else:
            # Multiple time periods to concatenate
            concats = []
            for period in periods:
                # Make the call to xclim
                ds_subset = ds.sel(time=slice(period[0], period[1]))
                tmp = ind(ds=ds_subset)

                # In the case of multiple outputs, merge them into a single dataset
                if isinstance(tmp, tuple):
                    tmp = xr.merge(tmp)
                    tmp.attrs = {}
                else:
                    tmp = tmp.to_dataset()

                # Infer the indicator's frequency
                if "time" in tmp.dims:
                    if len(tmp.time) < 3:
                        freq = _infer_freq_from_meta(ind)
                    else:
                        freq = xr.infer_freq(tmp.time)
                else:
                    freq = "fx"

                if freq == "YS":
                    freq = "YS-JAN"
                # In order to concatenate time periods, the indicator still needs a time dimension
                if freq == "fx":
                    tmp = tmp.assign_coords({"time": ds_subset.time[0]})

                concats.append(tmp)
            out = xr.concat(concats, dim="time")

        # Make sure that attributes have been kept for the dimensions and coordinates. Fixes a bug in xarray.
        for c in set(list(out.coords) + list(out.dims)).intersection(
            set(list(ds.coords) + list(ds.dims))
        ):
            if (out[c].attrs != ds[c].attrs) and (out[c].sizes == ds[c].sizes):
                out[c].attrs = ds[c].attrs

        if restrict_years and "time" in out.dims:
            # cut the time axis to be within the same years as the input
            # for QS-DEC, xclim starts on DJF with time previous_year-12-01 with a nan as values. We want to cut this.
            # this should have no effect on YS and MS indicators
            out = out.sel(
                time=slice(
                    str(ds.time[0].dt.year.values), str(ds.time[-1].dt.year.values)
                )
            )

        # Create the dictionary key
        key = freq
        if key not in out_dict:
            out_dict[key] = out
            # TODO: Double-check History, units, attrs, add missing variables (grid_mapping), etc.
            out_dict[key].attrs = ds.attrs
            out_dict[key].attrs["cat:variable"] = parse_from_ds(
                out_dict[key], ["variable"]
            )["variable"]
            out_dict[key].attrs["cat:xrfreq"] = freq
            out_dict[key].attrs["cat:frequency"] = CV.xrfreq_to_frequency(freq, None)
            if to_level is not None:
                out_dict[key].attrs["cat:processing_level"] = to_level

        else:
            for v in out.data_vars:
                out_dict[key][v] = out[v]

    return out_dict


def registry_from_module(
    module: ModuleType,
    registry: Optional[DerivedVariableRegistry] = None,
    variable_column: str = "variable",
) -> DerivedVariableRegistry:
    """Convert a xclim virtual indicators module to an intake_esm Derived Variable Registry.

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
            dvr.register(variable=attrs["var_name"], query=query)(_derived_func(ind, i))
    return dvr


def _ensure_list(x):
    if not isinstance(x, (list, tuple)):
        return [x]
    return x


def _derived_func(ind: xc.core.indicator.Indicator, nout: int) -> partial:
    def func(ds, *, ind, nout):
        out = ind(ds=ds)
        if isinstance(out, tuple):
            out = out[nout]
        ds[out.name] = out
        return ds

    func.__name__ = ind.identifier
    return partial(func, ind=ind, nout=nout)


def select_inds_for_avail_vars(
    ds: xr.Dataset,
    indicators: Union[
        str,
        os.PathLike,
        Sequence[Indicator],
        Sequence[tuple[str, Indicator]],
        ModuleType,
    ],
) -> ModuleType:
    """Filter the indicators for which the necessary variables are available.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to use for the indicators.
    indicators : Union[str, os.PathLike, Sequence[Indicator], Sequence[Tuple[str, Indicator]]]
        Path to a YAML file that instructs on how to calculate indicators.
        Can also be only the "stem", if translations and custom indices are implemented.
        Can be the indicator module directly, or a sequence of indicators or a sequence of
        tuples (indicator name, indicator) as returned by `iter_indicators()`.

    Returns
    -------
    ModuleType – An indicator module of 'length' ∈ [0, n].

    See Also
    --------
    xclim.indicators, xclim.core.indicator.build_indicator_module_from_yaml
    """
    # Transform the 'indicators' input into a list of tuples (name, indicator)
    is_list_of_tuples = isinstance(indicators, list) and all(
        isinstance(i, tuple) for i in indicators
    )
    if isinstance(indicators, (str, os.PathLike)):
        logger.debug("Loading indicator module.")
        indicators = load_xclim_module(indicators, reload=True)
    if hasattr(indicators, "iter_indicators"):
        indicators = [(name, ind) for name, ind in indicators.iter_indicators()]
    elif isinstance(indicators, (list, tuple)) and not is_list_of_tuples:
        indicators = [(ind.base, ind) for ind in indicators]

    available_vars = {
        var for var in ds.data_vars if var in xc.core.utils.VARIABLES.keys()
    }
    available_inds = [
        (name, ind)
        for var in available_vars
        for name, ind in indicators
        if var in ind.parameters.keys()
    ]
    return xc.core.indicator.build_indicator_module(
        "inds_for_avail_vars", available_inds, reload=True
    )
