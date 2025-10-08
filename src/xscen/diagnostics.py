"""Functions to perform diagnostics on datasets."""

import logging
import os
import warnings
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import numpy as np
import xarray as xr
import xclim as xc
import xclim.core.dataflags
from xclim.core import ValidationError
from xclim.core.formatting import (  # noqa: F401
    _merge_attrs_drop_conflicts as merge_attrs,
)
from xclim.core.indicator import Indicator

from .config import parse_config
from .indicators import load_xclim_module
from .utils import (
    add_attr,
    change_units,
    clean_up,
    date_parser,
    standardize_periods,
    unstack_fill_nan,
    update_attr,
    xclim_convert_units_to,
)


logger = logging.getLogger(__name__)

__all__ = [
    "health_checks",
    "measures_heatmap",
    "measures_improvement",
    "properties_and_measures",
]


# Dummy function to make gettext aware of translatable-strings
def _(s):
    return s


@parse_config
def health_checks(  # noqa: C901
    ds: xr.Dataset | xr.DataArray,
    *,
    structure: dict | None = None,
    calendar: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    variables_and_units: dict | None = None,
    strict_units: bool = False,
    cfchecks: dict | None = None,
    freq: str | None = None,
    missing: dict | str | list | None = None,
    flags: dict | None = None,
    flags_kwargs: dict | None = None,
    return_flags: bool = False,
    raise_on: list | None = None,
) -> None | xr.Dataset:
    """
    Perform a series of health checks on the dataset. Be aware that missing data checks and flag checks can be slow.

    Parameters
    ----------
    ds: xr.Dataset or xr.DataArray
        Dataset to check.
    structure: dict, optional
        Dictionary with keys "dims" and "coords" containing the expected dimensions and coordinates.
        This check will fail is extra dimensions or coordinates are found.
    calendar: str, optional
        Expected calendar. Synonyms should be detected correctly (e.g. "standard" and "gregorian").
    start_date: str, optional
        To check if the dataset starts at least at this date.
    end_date: str, optional
        To check if the dataset ends at least at this date.
    variables_and_units: dict, optional
        Dictionary containing the expected variables and units.
    strict_units: bool
        If True, the units in variables_and_units need to match exactly.
    cfchecks: dict, optional
        Dictionary where the key is the variable to check and the values are the cfchecks.
        The cfchecks themselves must be a dictionary with the keys being the cfcheck names
        and the values being the arguments to pass to the cfcheck.
        See `xclim.core.cfchecks` for more details.
    freq: str, optional
        Expected frequency, written as the result of xr.infer_freq(ds.time).
    missing: dict or str or list of str, optional
        String, list of strings, or dictionary where the key is the method to check for missing data
        and the values are the arguments to pass to the method.
        The methods are: "missing_any", "at_least_n_valid", "missing_pct", "missing_wmo".
        See :py:func:`xclim.core.missing` for more details.
    flags: dict, optional
        Dictionary where the key is the variable to check and the values are the flags.
        The flags themselves must be a dictionary with the keys being the data_flags names
        and the values being the arguments to pass to the data_flags.
        If `None` is passed instead of a dictionary, then xclim's default flags for the given variable are run.
        See :py:data:`xclim.core.utils.VARIABLES`.
        See also :py:func:`xclim.core.dataflags.data_flags` for the list of possible flags.
    flags_kwargs: dict, optional
        Additional keyword arguments to pass to the data_flags ("dims" and "freq").
    return_flags: bool
        Whether to return the Dataset created by data_flags.
    raise_on: list of str, optional
        Whether to raise an error if a check fails, else there will only be a warning.
        The possible values are the names of the checks.
        Use ["all"] to raise on all checks.

    Returns
    -------
    xr.Dataset or None
        Dataset containing the flags if return_flags is True & raise_on is False for the "flags" check.
    """
    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset()
    raise_on = raise_on or []
    if "all" in raise_on:
        raise_on = [
            "structure",
            "calendar",
            "start_date",
            "end_date",
            "variables_and_units",
            "cfchecks",
            "freq",
            "missing",
            "flags",
        ]

    warns = []
    errs = []

    def _error(msg, check):
        if check in raise_on:
            errs.append(msg)
        else:
            warns.append(msg)

    def _message():
        base = "The following health checks failed:"
        if len(warns) > 0:
            msg = "\n  - ".join([base] + warns)
            warnings.warn(msg, UserWarning, stacklevel=2)
        if len(errs) > 0:
            msg = "\n  - ".join([base] + errs)
            raise ValueError(msg)

    # Check the dimensions and coordinates
    if structure is not None:
        if "dims" in structure:
            for dim in structure["dims"]:
                if dim not in ds.dims:
                    _error(f"The dimension '{dim}' is missing.", "structure")
            extra_dims = [dim for dim in ds.dims if dim not in structure["dims"]]
            if len(extra_dims) > 0:
                _error(
                    f"Extra dimensions found: {extra_dims}.",
                    "structure",
                )
        if "coords" in structure:
            for coord in structure["coords"]:
                if coord not in ds.coords:
                    if coord in ds.data_vars:
                        _error(
                            f"'{coord}' is detected as a data variable, not a coordinate.",
                            "structure",
                        )
                    else:
                        _error(f"The coordinate '{coord}' is missing.", "structure")
            extra_coords = [coord for coord in ds.coords if coord not in structure["coords"]]
            if len(extra_coords) > 0:
                _error(f"Extra coordinates found: {extra_coords}.", "structure")

    # Check the calendar
    if calendar is not None:
        cal = ds.time.dt.calendar
        if xc.core.calendar.common_calendar([calendar]).replace("default", "standard") != xc.core.calendar.common_calendar([cal]).replace(
            "default", "standard"
        ):
            _error(f"The calendar is not '{calendar}'. Received '{cal}'.", "calendar")

    # Check the start/end dates
    if (start_date is not None) or (end_date is not None):
        ds_start = date_parser(ds.time.min().dt.floor("D").item())
        ds_end = date_parser(ds.time.max().dt.floor("D").item())
    if start_date is not None:
        # Create cf_time objects to compare the dates
        start_date = date_parser(start_date)
        if not ((ds_start <= start_date) and (ds_end > start_date)):
            _error(
                f"The start date is not at least {start_date}. Received {ds_start}.",
                "start_date",
            )
    if end_date is not None:
        # Create cf_time objects to compare the dates
        end_date = date_parser(end_date)
        if not ((ds_start < end_date) and (ds_end >= end_date)):
            _error(
                f"The end date is not at least {end_date}. Received {ds_end}.",
                "end_date",
            )

    # Check variables
    if variables_and_units is not None:
        for v in variables_and_units:
            if v not in ds:
                _error(f"The variable '{v}' is missing.", "variables_and_units")
            elif ds[v].attrs.get("units", None) != variables_and_units[v]:
                with xc.set_options(data_validation="raise"):
                    try:
                        xc.core.units.check_units(ds[v], variables_and_units[v])
                    except ValidationError as e:
                        _error(f"'{v}' ValidationError: {e}", "variables_and_units")
                # are they technically the same
                close_enough = xc.units.str2pint(ds[v].attrs["units"]) == xc.units.str2pint(variables_and_units[v])

                if strict_units or not close_enough:
                    _error(
                        f"The variable '{v}' does not have the expected units '{variables_and_units[v]}'. Received '{ds[v].attrs['units']}'.",
                        "variables_and_units",
                    )

    # Check CF conventions
    if cfchecks is not None:
        cfchecks = deepcopy(cfchecks)
        for v in cfchecks:
            for check in cfchecks[v]:
                if check == "check_valid":
                    cfchecks[v][check]["var"] = ds[v]
                elif check == "cfcheck_from_name":
                    cfchecks[v][check].setdefault("varname", v)
                    cfchecks[v][check]["vardata"] = ds[v]
                else:
                    raise ValueError(f"Check '{check}' is not in xclim.")
                with xc.set_options(cf_compliance="raise"):
                    try:
                        getattr(xc.core.cfchecks, check)(**cfchecks[v][check])
                    except ValidationError as e:
                        _error(f"'{v}' ValidationError: {e}", "cfchecks")

    if freq is not None:
        inferred_freq = xr.infer_freq(ds.time)
        if inferred_freq is None:
            _error("The timesteps are irregular or cannot be inferred by xarray.", "freq")
        elif freq.replace("YS", "YS-JAN") != inferred_freq:
            _error(f"The frequency is not '{freq}'. Received '{inferred_freq}'.", "freq")

    if missing is not None:
        inferred_freq = xr.infer_freq(ds.time)
        if inferred_freq not in ["M", "MS", "D", "H"]:
            warnings.warn(
                f"Frequency {inferred_freq} is not supported for missing data checks. That check will be skipped.",
                UserWarning,
                stacklevel=1,
            )
        else:
            if isinstance(missing, str):
                missing = {missing: {}}
            elif isinstance(missing, list):
                missing = {m: {} for m in missing}
            for method, kwargs in missing.items():
                kwargs.setdefault("freq", "YS")
                for v in ds.data_vars:
                    if "time" in ds[v].dims:
                        ms = getattr(xc.core.missing, method)(ds[v], **kwargs)
                        if ms.any():
                            _error(
                                f"The variable '{v}' has missing values according to the '{method}' method.",
                                "missing",
                            )
                    else:
                        msg = (f"Variable '{v}' has no time dimension. The missing data check will be skipped.",)
                        logger.info(msg)

    if flags is not None:
        if return_flags:
            out = xr.Dataset()
        for v in flags:
            dsflags = xc.core.dataflags.data_flags(
                ds[v],
                ds,
                flags=flags[v],
                raise_flags=False,
                **(flags_kwargs or {}),
            )
            if np.any([dsflags[dv] for dv in dsflags.data_vars]):
                bad_checks = [dv for dv in dsflags.data_vars if dsflags[dv].any()]
                _error(
                    f"'{v}' has suspicious values according to the following flags: {bad_checks}.",
                    "flags",
                )
            if return_flags:
                dsflags = dsflags.rename({dv: f"{v}_{dv}" for dv in dsflags.data_vars})
                out = xr.merge([out, dsflags])

    _message()
    if return_flags and flags is not None:
        return out


@parse_config
def properties_and_measures(  # noqa: C901
    ds: xr.Dataset,
    properties: (str | os.PathLike | Sequence[Indicator] | Sequence[tuple[str, Indicator]] | ModuleType),
    period: list[str] | None = None,
    unstack: bool = False,
    rechunk: dict | None = None,
    dref_for_measure: xr.Dataset | None = None,
    change_units_arg: dict | None = None,
    to_level_prop: str = "diag-properties",
    to_level_meas: str = "diag-measures",
) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Calculate properties and measures of a dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    properties : str | os.PathLike | Sequence[Indicator] | Sequence[tuple[str, Indicator]] | ModuleType
        Path to a YAML file that instructs on how to calculate properties.
        Can be the indicator module directly, or a sequence of indicators or a sequence of
        tuples (indicator name, indicator) as returned by `iter_indicators()`.
    period : list of str, optional
        [start, end] of the period to be evaluated. The period will be selected on ds
        and dref_for_measure if it is given.
    unstack : bool
        Whether to unstack ds before computing the properties.
    rechunk : dict, optional
        Dictionary of chunks to use for a rechunk before computing the properties.
    dref_for_measure : xr.Dataset, optional
        Dataset of properties to be used as the ref argument in the computation of the measure.
        Ideally, this is the first output (prop) of a previous call to this function.
        Only measures on properties that are provided both in this dataset and in the properties list will be computed.
        If None, the second output of the function (meas) will be an empty Dataset.
    change_units_arg : dict, optional
        If not None, calls `xscen.utils.change_units` on ds before computing properties using
        this dictionary for the `variables_and_units` argument.
        It can be useful to convert units before computing the properties, because it is sometimes
        easier to convert the units of the variables than the units of the properties (e.g. variance).
    to_level_prop : str
        processing_level to give the first output (prop)
    to_level_meas : str
        processing_level to give the second output (meas)

    Returns
    -------
    prop : xr.Dataset
        Dataset of properties of ds
    meas : xr.Dataset
        Dataset of measures between prop and dref_for_meas

    See Also
    --------
    xsdba.properties, xsdba.measures, xclim.core.indicator.build_indicator_module_from_yaml
    """
    if isinstance(properties, str | Path):
        logger.debug("Loading properties module.")
        module = load_xclim_module(properties)
        properties = module.iter_indicators()
    elif hasattr(properties, "iter_indicators"):
        properties = properties.iter_indicators()

    try:
        N = len(properties)
    except TypeError:
        N = None
    else:
        msg = f"Computing {N} properties."
        logger.info(msg)

    period = standardize_periods(period, multiple=False)
    # select period for ds
    if period is not None and "time" in ds:
        ds = ds.sel({"time": slice(period[0], period[1])})

    # select periods for ref_measure
    if dref_for_measure is not None and period is not None and "time" in dref_for_measure:
        dref_for_measure = dref_for_measure.sel({"time": slice(period[0], period[1])})

    if unstack:
        ds = unstack_fill_nan(ds)

    if rechunk:
        ds = ds.chunk(rechunk)

    if change_units_arg:
        ds = change_units(ds, variables_and_units=change_units_arg)

    prop = xr.Dataset()  # dataset with all properties
    meas = xr.Dataset()  # dataset with all measures
    for i, ind in enumerate(properties, 1):
        if isinstance(ind, tuple):
            iden, ind = ind
        else:
            iden = ind.identifier
        # Make the call to xclim
        msg = f"{i} - Computing {iden}."
        logger.info(msg)
        with xclim_convert_units_to():
            out = ind(ds=ds)
        vname = out.name
        prop[vname] = out

        if period is not None:
            prop[vname].attrs["period"] = f"{period[0]}-{period[1]}"

        # calculate the measure if a reference dataset is given for the measure
        if dref_for_measure and vname in dref_for_measure:
            with xclim_convert_units_to():
                meas[vname] = ind.get_measure()(sim=prop[vname], ref=dref_for_measure[vname])
            # create a merged long_name
            update_attr(
                meas[vname],
                "long_name",
                "{attr1} {attr}",
                others=[prop[vname]],
            )

    for ds1 in [prop, meas]:
        ds1.attrs = ds.attrs
        ds1.attrs["cat:xrfreq"] = "fx"
        ds1.attrs.pop("cat:variable", None)
        ds1.attrs["cat:frequency"] = "fx"

        # to be able to save in zarr, convert object to string
        if "season" in ds1:
            ds1["season"] = ds1.season.astype("str")

    prop.attrs["cat:processing_level"] = to_level_prop
    meas.attrs["cat:processing_level"] = to_level_meas

    return prop, meas


def measures_heatmap(meas_datasets: list[xr.Dataset] | dict, to_level: str = "diag-heatmap") -> xr.Dataset:
    """
    Create a heatmap to compare the performance of the different datasets.

    The columns are properties and the rows are datasets.
    Each point is the absolute value of the mean of the measure over the whole domain.
    Each column is normalized from 0 (best) to 1 (worst).

    Parameters
    ----------
    meas_datasets : list of xr.Dataset or dict
        List or dictionary of datasets of measures of properties.
        If it is a dictionary, the keys will be used to name the rows.
        If it is a list, the rows will be given a number.
    to_level : str
        The processing_level to assign to the output.

    Returns
    -------
    xr.Dataset
        Dataset containing the heatmap.
    """
    name_of_datasets = None
    if isinstance(meas_datasets, dict):
        name_of_datasets = list(meas_datasets.keys())
        meas_datasets = list(meas_datasets.values())

    hmap = []
    for meas in meas_datasets:
        row = []
        # iterate through all available properties
        for var_name in meas:
            da = meas[var_name]
            # mean the absolute value of the bias over all positions and add to heat map
            # TODO: check this indeed works with xsdba
            if "xsdba.measures.RATIO" in da.attrs["history"]:
                # if ratio, best is 1, this moves "best to 0 to compare with bias
                row.append(abs(da - 1).mean().values)
            else:
                row.append(abs(da).mean().values)
        # append all properties
        hmap.append(row)

    # plot heatmap of biases (1 column per properties, 1 row per dataset)
    hmap = np.array(hmap)
    # normalize to 0-1 -> best-worst
    hmap = np.array([((c - np.min(c)) / (np.max(c) - np.min(c)) if np.max(c) != np.min(c) else [0.5] * len(c)) for c in hmap.T]).T

    name_of_datasets = name_of_datasets or list(range(1, hmap.shape[0] + 1))
    ds_hmap = xr.DataArray(
        hmap,
        coords={
            "realization": name_of_datasets,
            "properties": list(meas_datasets[0].data_vars),
        },
        dims=["realization", "properties"],
    )
    ds_hmap = ds_hmap.to_dataset(name="heatmap")

    ds_hmap.attrs = merge_attrs(*[ds for ds in meas_datasets])
    ds_hmap = clean_up(
        ds=ds_hmap,
        common_attrs_only=meas_datasets,
    )
    ds_hmap.attrs["cat:processing_level"] = to_level
    ds_hmap.attrs.pop("cat:variable", None)
    add_attr(ds_hmap["heatmap"], "long_name", _("Ranking of measure performance"))

    return ds_hmap


def measures_improvement(
    meas_datasets: list[xr.Dataset] | dict,
    dim: str | Sequence[str] | None = None,
    to_level: str = "diag-improved",
) -> xr.Dataset:
    """
    Calculate the fraction of improved grid points for each property between two datasets of measures.

    Parameters
    ----------
    meas_datasets: list[xr.Dataset] | dict
        List of 2 datasets: Initial dataset of measures and final (improved) dataset of measures.
        Both datasets must have the same variables.
        It is also possible to pass a dictionary where the values are the datasets and the key are not used.
    dim : str or sequence of str, optional
        Dimension(s) on which to compute the percentage of improved grid points. Default is `None`, which reduces all dimensions.
    to_level: str
        processing_level to assign to the output dataset

    Returns
    -------
    xr.Dataset
        Dataset containing information on the fraction of improved grid points for each property.

    Notes
    -----
    If `dim` is specified, it should be present in every variable of  `meas_datasets`.
    """
    if isinstance(meas_datasets, dict):
        meas_datasets = list(meas_datasets.values())

    if len(meas_datasets) != 2:
        warnings.warn("meas_datasets has more than 2 datasets. Only the first 2 will be compared.", stacklevel=2)
    ds1 = meas_datasets[0]
    ds2 = meas_datasets[1]
    if dim is not None:
        dims = [dim] if isinstance(dim, str) else dim
        for v in ds1.data_vars:
            if set(dims).issubset(set(ds1[v].dims)) is False:
                raise ValueError(f"Dimension provided `dim` ({dim}) must be in every variable of `meas_datasets`")

    percent_better = []
    for var in ds2.data_vars:
        if dim is None:
            # reduce all dimensions (which may be variable dependent)
            dims = ds2[var].dims
        if "xsdba.measures.RATIO" in ds1[var].attrs["history"]:
            diff_bias = abs(ds1[var] - 1) - abs(ds2[var] - 1)
        else:
            diff_bias = abs(ds1[var]) - abs(ds2[var])
        total_improved = (diff_bias >= 0).sum(dim=dims)
        total_notnull = diff_bias.notnull().sum(dim=dims)
        percent_better_var = total_improved / total_notnull
        percent_better.append(percent_better_var.expand_dims({"properties": [var]}))

    ds_better = xr.concat(percent_better, dim="properties")

    ds_better = ds_better.to_dataset(name="improved_grid_points")

    add_attr(
        ds_better["improved_grid_points"],
        "long_name",
        _("Fraction of improved grid cells"),
    )
    ds_better.attrs = ds2.attrs
    ds_better.attrs["cat:processing_level"] = to_level
    ds_better.attrs.pop("cat:variable", None)

    return ds_better


def measures_improvement_2d(dict_input: dict, to_level: str = "diag-improved-2d") -> xr.Dataset:
    """
    Create a 2D dataset with dimension `realization` showing the fraction of improved grid cell.

    Parameters
    ----------
    dict_input: dict
      If dict of datasets, the datasets should be the output of `measures_improvement`.
      If dict of dict/list, the dict/list should be the input `meas_datasets` to `measures_improvement`.
      The keys will be the values of the dimension `realization`.
    to_level: str
      Processing_level to assign to the output dataset.

    Returns
    -------
    xr.Dataset
      Dataset with extra `realization` coordinates.
    """
    merge = {}
    for name, value in dict_input.items():
        # if dataset, assume the value is already the output of `measures_improvement`
        if isinstance(value, xr.Dataset):
            out = value.expand_dims(dim={"realization": [name]})
        # else, compute the `measures_improvement`
        else:
            out = measures_improvement(value).expand_dims(dim={"realization": [name]})
        merge[name] = out
    # put everything in one dataset with dim datasets
    ds_merge = xr.concat(list(merge.values()), dim="realization")
    ds_merge["realization"] = ds_merge["realization"].astype(str)
    ds_merge = clean_up(
        ds=ds_merge,
        common_attrs_only=merge,
    )
    ds_merge.attrs["cat:processing_level"] = to_level

    return ds_merge
