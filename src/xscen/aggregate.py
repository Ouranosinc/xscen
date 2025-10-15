"""Functions to aggregate data over time and space."""

import datetime
import logging
import os
import warnings
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy
import shapely
import xarray as xr
import xclim as xc
import xclim.core.calendar


try:
    import xesmf as xe
except (ImportError, KeyError) as e:
    if isinstance(e, KeyError):
        if e.args[0] == "Author":
            warnings.warn(
                "The xesmf package could not be imported due to a known KeyError bug that occurs with some "
                "older versions of ESMF and specific execution setups (such as debugging on a Windows machine). "
                "As a workaround, try installing 'importlib-metadata <8.0.0' and/or updating ESMF. If you do not "
                "need 'xesmf' functionalities (e.g. regridding), you can ignore this warning.",
                stacklevel=2,
            )
        else:
            raise e
    xe = None
from xclim.core.indicator import Indicator
from xclim.core.units import pint2cfattrs, units2pint

from .config import parse_config
from .extract import subset_warming_level
from .indicators import compute_indicators
from .spatial import get_grid_mapping, subset
from .utils import standardize_periods, unstack_dates, update_attr


logger = logging.getLogger(__name__)

__all__ = [
    "climatological_op",
    "compute_deltas",
    "produce_horizon",
    "spatial_mean",
]


# Dummy function to make gettext aware of translatable-strings
def _(s):
    return s


@parse_config
def climatological_op(  # noqa: C901
    ds: xr.Dataset,
    *,
    op: str | dict = "mean",
    window: int | None = None,
    min_periods: int | float | None = None,
    stride: int = 1,
    periods: list[str] | list[list[str]] | None = None,
    rename_variables: bool = True,
    to_level: str = "climatology",
    horizons_as_dim: bool = False,
) -> xr.Dataset:
    """
    Perform an operation 'op' over time, for given time periods, respecting the temporal resolution of ds.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to use for the computation.
    op : str or dict
        Operation to perform over time.
        The operation can be any method name of xarray.core.rolling.DatasetRolling, 'linregress', or a dictionary.
        If 'op' is a dictionary, the key is the operation name and the value is a dict of kwargs
        accepted by the equivalent NumPy function. See the Notes for more information.
        While other operations are technically possible, the following are recommended and tested:
        ['max', 'mean', 'median', 'min', 'std', 'sum', 'var', 'linregress'].
        Operations beyond methods of xarray.core.rolling.DatasetRolling include:

            - 'linregress' : Computes the linear regression over time, using
              scipy.stats.linregress and employing years as regressors.
              The output will have a new dimension 'linreg_param' with coordinates:
              ['slope', 'intercept', 'rvalue', 'pvalue', 'stderr', 'intercept_stderr'].

        Only one operation per call is supported, so len(op)==1 if a dict.
    window : int, optional
        Number of years to use for the rolling operation.
        If left at None and periods is given, window will be the size of the first period. Hence, if periods are of
        different lengths, the shortest period should be passed first.
        If left at None and periods is not given, the window will be the size of the input dataset.
    min_periods : int or float, optional
        For the rolling operation, minimum number of years required for a value to be computed.
        If left at None and the xrfreq is either QS or AS and doesn't start in January,
        min_periods will be one less than window.
        Otherwise, if left at None, it will be deemed the same as 'window'.
        If passed as a float value between 0 and 1, this will be interpreted as the floor of the fraction of the window size.
    stride : int
        Stride (in years) at which to provide an output from the rolling window operation.
    periods : list of str or list of lists of str, optional
        Either [start, end] or list of [start, end] of continuous periods to be considered.
        This is needed when the time axis of ds contains some jumps in time.
        If None, the dataset will be considered continuous.
    rename_variables : bool
        If True, '_clim_{op}' will be added to variable names.
    to_level : str, optional
        The processing level to assign to the output.
        If None, the processing level of the inputs is preserved.
    horizons_as_dim : bool
        If True, the output will have 'horizon' and the frequency as 'month', 'season' or 'year' as
        dimensions and coordinates. The 'time' coordinate will be unstacked to horizon and frequency dimensions.
        Horizons originate from periods and/or windows and their stride in the rolling operation.

    Returns
    -------
    xr.Dataset
        Dataset with the results from the climatological operation.

    Notes
    -----
    xarray.core.rolling.DatasetRolling functions do not support kwargs other than 'keep_attrs'. In order to pass
    additional arguments to the operation, we instead use the 'reduce' method and pass the operation as a numpy function.
    If possible, a function that handles NaN values will be used (e.g. op='mean' will use `np.nanmean`), as the
    'min_periods' argument already decides how many NaN values are acceptable.
    """
    # Daily data is not supported
    if len(ds.time) > 3 and xr.infer_freq(ds.time) == "D":
        raise NotImplementedError("xs.climatological_op does not currently support daily data.")
    # more than one operation per call is not supported (yet), case for dict
    if isinstance(op, dict) and len(op) > 1:
        raise NotImplementedError("xs.climatological_op does not currently support more than one operation per call.")

    # unstack 1D time in coords (day, month, and year) to make climatological mean faster
    mindex_coords = xr.Coordinates.from_pandas_multiindex(
        pd.MultiIndex.from_arrays(
            [
                ds.time.dt.year.values,
                ds.time.dt.month.values,
                ds.time.dt.day.values,
            ],
            names=["year", "month", "day"],
        ),
        dim="time",
    )
    ds_unstack = ds.assign_coords(coords=mindex_coords).unstack("time")

    # Rolling will ignore gaps in time, so raise an exception beforehand
    if (not all(ds_unstack.year.diff(dim="year", n=1) == 1)) & (periods is None):
        raise ValueError("Data is not continuous. Use the 'periods' argument.")

    # define periods, windows, and min_periods
    periods = standardize_periods(periods or [[int(ds_unstack.year[0]), int(ds_unstack.year[-1])]])
    window = window or int(periods[0][1]) - int(periods[0][0]) + 1

    # there is one less occurrence when a period crosses years
    freq_across_year = [f"{f}-{mon}" for mon in xr.coding.cftime_offsets._MONTH_ABBREVIATIONS.values() for f in ["AS", "QS"] if mon != "JAN"]
    if (
        any(
            x in freq_across_year
            for x in [
                ds.attrs.get("cat:xrfreq"),
                (xr.infer_freq(ds.time) if len(ds.time) > 3 else None),
            ]
        )
        and min_periods is None
    ):
        min_periods = window - 1

    # unpack min_periods as fraction of window
    if isinstance(min_periods, float):
        if 0 < min_periods <= 1:
            min_periods = int(np.floor(min_periods * window))
        else:
            raise ValueError(f"When 'min_periods' is passed as a 'float', it must be between 0 and 1. Got {min_periods}.")

    # set min_periods
    min_periods = min_periods or window
    if min_periods > window:
        raise ValueError("'min_periods' should be smaller or equal to 'window'")

    # if op is a dict, unpack it
    if isinstance(op, dict):
        op, op_kwargs = list(op.items())[0]
        op_kwargs.setdefault("keep_attrs", True)
    else:
        op_kwargs = {"keep_attrs": True}

    # special case for averaging standard deviations: need to convert to variance before averaging
    ds_has_std = False
    std_v = []
    if op == "mean":
        for vv in ds_unstack.data_vars:
            if "std" in vv or "standard deviation" in ds_unstack[vv].attrs.get("description", "").lower():
                ds_unstack[vv] = np.square(ds_unstack[vv])
                ds_has_std = True
                std_v.extend([vv])

    # Compute the climatological operation
    concats = []
    for period in periods:
        # Rolling average
        ds_rolling = ds_unstack.sel(year=slice(period[0], period[1])).rolling(year=window, min_periods=min_periods)

        # apply operation on rolling object
        if hasattr(ds_rolling, op) and callable(getattr(ds_rolling, op)):
            if op not in ["max", "mean", "median", "min", "std", "sum", "var"]:
                warnings.warn(f"The requested operation '{op}' has not been tested and may produce unexpected results.", stacklevel=2)
            if len(set(op_kwargs.keys()).difference(["keep_attrs"])) > 0:
                # Reduce allows for kwargs to be passed to the operation, but it is less efficient
                try:
                    numpy_op = getattr(np, f"nan{op}")
                except AttributeError:
                    warnings.warn(
                        f"The requested operation '{op}' has no NaN handling in numpy. Results may ignore the 'min_periods' argument.", stacklevel=2
                    )
                    numpy_op = getattr(np, op)
                ds_rolling = ds_rolling.reduce(numpy_op, **op_kwargs)
            else:
                ds_rolling = getattr(ds_rolling, op)(**op_kwargs)

            # revert variance to std, where applicable
            if op == "mean" and ds_has_std:
                for vv in std_v:
                    ds_rolling[vv] = np.sqrt(ds_rolling[vv])

            # Select the windows at provided stride, starting from the first full window's operation result
            ds_rolling = ds_rolling.isel(year=slice(window - 1, None, stride))

        elif op == "linregress":

            def _ulinregress(x, y, **kwargs):
                # Wrapper for scipy.stats.linregress to unpack multiple return values in xr.apply_ufunc
                valid_x = ~np.isnan(x)
                valid_y = ~np.isnan(y)
                mask = valid_x & valid_y
                if np.sum(mask) >= kwargs.get("min_periods", 1):
                    x = x[mask]
                    y = y[mask]
                    reg = scipy.stats.linregress(x, y, alternative=kwargs.get("alternative", "two-sided"))
                    out = np.array(
                        [
                            reg.slope,
                            reg.intercept,
                            reg.rvalue,
                            reg.pvalue,
                            reg.stderr,
                            reg.intercept_stderr,
                        ]
                    )
                else:
                    out = np.full(6, np.nan)
                return out

            # prepare kwargs
            linreg_kwargs = {k: v for k, v in op_kwargs.items() if "keep_attrs" not in k}
            linreg_kwargs["min_periods"] = min_periods

            # unwrap DatasetRolling object and select years subset
            dsr_construct = ds_rolling.construct(window_dim="window", keep_attrs=True)
            dsr_construct = dsr_construct.isel(year=slice(window - 1, None, stride))

            # construct array to use years as x values (==regressors) in xr.apply_ufunc
            years_as_x_values = xr.DataArray(
                np.arange(dsr_construct.window.size).repeat(dsr_construct.year.size).reshape(dsr_construct.window.size, dsr_construct.year.size)
                + dsr_construct.year.values
                - window
                + 1,
                dims=["window", "year"],
                coords={
                    "window": dsr_construct.window.values,
                    "year": dsr_construct.year.values,
                },
            )

            # apply linregress along windows
            ds_rolling = xr.apply_ufunc(
                _ulinregress,
                years_as_x_values,
                dsr_construct,
                input_core_dims=[["window"], ["window"]],
                output_core_dims=[["linreg_param"]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=["float32"],
                dask_gufunc_kwargs={"output_sizes": {"linreg_param": 6}},
                keep_attrs="no_conflicts",
                kwargs=linreg_kwargs,
            )
            # label new coords
            ds_rolling.coords["linreg_param"] = [
                "slope",
                "intercept",
                "rvalue",
                "pvalue",
                "stderr",
                "intercept_stderr",
            ]

        else:
            raise ValueError(f"Operation '{op}' not implemented.")

        # build horizons
        horizons = xr.DataArray(
            [f"{yr - (window - 1)}-{yr}" for yr in ds_rolling.year.values],
            dims=dict(year=ds_rolling.year),
        ).astype(str)
        ds_rolling = ds_rolling.assign_coords(horizon=horizons)

        # revert to 1D time, rebuilding time coord
        ds_rolling = ds_rolling.stack(time=("year", "month", "day"))
        if isinstance(ds.indexes["time"], pd.core.indexes.datetimes.DatetimeIndex):
            time_coord = pd.to_datetime(
                {
                    "year": ds_rolling.year.values - window + 1,
                    "month": ds_rolling.month.values,
                    "day": ds_rolling.day.values,
                }
            ).to_list()
        elif isinstance(ds.indexes["time"], xr.coding.cftimeindex.CFTimeIndex):
            time_coord = [
                xclim.core.calendar.datetime_classes[ds.time.dt.calendar](y - window + 1, m, d)
                for y, m, d in zip(
                    ds_rolling.year.values,
                    ds_rolling.month.values,
                    ds_rolling.day.values,
                    strict=False,
                )
            ]
        else:
            raise ValueError("The type of 'time' was not understood.")
        ds_rolling = ds_rolling.drop_vars({"month", "year", "time", "day"})
        ds_rolling = ds_rolling.assign_coords(time=time_coord).transpose("time", ...)

        # append to list of results
        concats.extend([ds_rolling])
        # end loop over periods

    # concatenate results
    ds_rolling = xr.concat(concats, dim="time", data_vars="minimal")

    # update data_vars names, attrs, history
    if rename_variables:
        ds_rolling = ds_rolling.rename_vars({vv: f"{vv}_clim_{op}" for vv in ds_rolling.data_vars})

    for vv in ds_rolling.data_vars:
        for a in ["description", "long_name"]:
            try:
                op_format = dict.fromkeys(("mean", "std", "var", "sum"), "adj") | dict.fromkeys(("max", "min"), "noun")
                operation = xc.core.formatting.default_formatter.format_field(op, op_format[op])
            except (KeyError, ValueError):
                operation = op
            update_attr(
                ds_rolling[vv],
                a,
                _("{window}-year climatological {operation} of {attr}."),
                window=window,
                operation=operation,
            )

        new_history = (
            f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {window}-year climatological {operation} "
            f"over window (non-centered), with a minimum of {min_periods} years of data - xarray v{xr.__version__}"
        )
        history = new_history + " \n " + ds_rolling[vv].attrs["history"] if "history" in ds_rolling[vv].attrs else new_history
        ds_rolling[vv].attrs["history"] = history

    # update processing level
    if to_level is not None:
        ds_rolling.attrs["cat:processing_level"] = to_level

    if horizons_as_dim:
        # restructure output to have horizons as a dimension instead of stacked horizons per year/season/month
        new_coords = {1: "year", 4: "season", 12: "month"}
        new_coord_len = len(np.unique(ds_rolling.time.dt.month))

        if new_coord_len == 1:
            all_horizons = [
                ds_rolling.sel(time=ds_rolling.horizon == horizon)
                .swap_dims({"time": "horizon"})
                .assign(
                    time_2D=xr.DataArray(
                        ds_rolling.time.sel(time=ds_rolling.horizon == horizon).values.copy(),
                        dims=["time"],
                    )
                )
                .drop_vars("time")
                .rename({"time_2D": "time"})
                .set_coords("time")
                .squeeze(dim="time")
                for horizon in np.unique(ds_rolling.horizon.values)
            ]
        else:
            all_horizons = [
                unstack_dates(
                    ds_rolling.sel(time=ds_rolling.horizon == horizon).assign(
                        time_2D=xr.DataArray(
                            ds_rolling.time.sel(time=ds_rolling.horizon == horizon).values.copy(),
                            dims=["time"],
                        )
                    ),
                    new_dim=new_coords[new_coord_len],
                )
                .drop_vars("horizon")
                .assign_coords(horizon=("time", [horizon]))
                .swap_dims({"time": "horizon"})
                .drop_vars("time")
                .rename({"time_2D": "time"})
                .set_coords("time")
                for horizon in np.unique(ds_rolling.horizon.values)
            ]

        return xr.concat(all_horizons, dim="horizon")

    else:
        return ds_rolling


@parse_config
def compute_deltas(  # noqa: C901
    ds: xr.Dataset,
    reference_horizon: str | xr.Dataset,
    *,
    kind: str | dict = "+",
    rename_variables: bool = True,
    to_level: str | None = "deltas",
) -> xr.Dataset:
    """
    Compute deltas in comparison to a reference time period, respecting the temporal resolution of ds.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to use for the computation.
    reference_horizon : str or xr.Dataset
        Either a YYYY-YYYY string corresponding to the 'horizon' coordinate of the reference period,
        or a xr.Dataset containing the climatological mean.
    kind : str or dict
        ['+', '/', '%'] Whether to provide absolute, relative, or percentage deltas.
        Can also be a dictionary separated per variable name.
    rename_variables : bool
        If True, '_delta_YYYY-YYYY' will be added to variable names.
    to_level : str, optional
        The processing level to assign to the output.
        If None, the processing level of the inputs is preserved.

    Returns
    -------
    xr.Dataset
        Returns a Dataset with the requested deltas.
    """
    if isinstance(reference_horizon, str):
        if reference_horizon not in ds.horizon:
            raise ValueError(f"The reference horizon {reference_horizon} is not in the dataset.")
        # Separate the reference from the other horizons
        if xc.core.utils.uses_dask(ds["horizon"]):
            ds["horizon"].load()
        ref = ds.where(ds.horizon == reference_horizon, drop=True)
    elif isinstance(reference_horizon, xr.Dataset):
        ref = reference_horizon
        if "horizon" in ref:
            reference_horizon = np.unique(ref["horizon"])
            if len(reference_horizon) != 1:
                raise ValueError("The reference dataset appears to contain multiple horizons.")
            reference_horizon = reference_horizon[0]
        else:
            reference_horizon = "unknown_horizon"
    else:
        raise ValueError("reference_horizon should be either a string or an xarray.Dataset.")

    if "time" in ds:
        if (len(ds.time) >= 3) and (xr.infer_freq(ds.time) == "D"):
            raise NotImplementedError("xs.compute_deltas does not currently support daily data.")

        # Remove references to 'year' in REF
        mindex_coords_1 = xr.Coordinates.from_pandas_multiindex(
            pd.MultiIndex.from_arrays(
                [ref.time.dt.month.values, ref.time.dt.day.values],
                names=["month", "day"],
            ),
            dim="time",
        )
        ref = ref.assign_coords(coords=mindex_coords_1).unstack("time")

        mindex_coords_2 = xr.Coordinates.from_pandas_multiindex(
            pd.MultiIndex.from_arrays(
                [
                    ds.time.dt.year.values,
                    ds.time.dt.month.values,
                    ds.time.dt.day.values,
                ],
                names=["year", "month", "day"],
            ),
            dim="time",
        )
        other_hz = ds.assign_coords(coords=mindex_coords_2).unstack("time")

    else:
        other_hz = ds
        ref = ref.squeeze()
    deltas = xr.Dataset(coords=other_hz.coords, attrs=other_hz.attrs)
    # Calculate deltas
    for vv in list(ds.data_vars):
        v_name = vv if rename_variables is False else f"{vv}_delta_{reference_horizon.replace('-', '_')}"

        with xr.set_options(keep_attrs=True):
            if (isinstance(kind, dict) and kind[vv] == "+") or kind == "+":
                _kind = "abs."
                deltas[v_name] = other_hz[vv] - ref[vv]
                unit = pint2cfattrs(units2pint(other_hz[vv].attrs["units"]), is_difference=True)
                deltas[v_name].attrs.update(unit)
            elif (isinstance(kind, dict) and kind[vv] == "/") or kind == "/":
                _kind = "rel."
                deltas[v_name] = other_hz[vv] / ref[vv]
                deltas[v_name].attrs["units"] = ""
            elif (isinstance(kind, dict) and kind[vv] == "%") or kind == "%":
                _kind = "pct."
                deltas[v_name] = 100 * (other_hz[vv] - ref[vv]) / ref[vv]
                deltas[v_name].attrs["units"] = "%"
            else:
                raise ValueError(f"Delta 'kind' not understood. Should be '+', '/' or '%', received {kind}.")

        # modify attrs and history
        deltas[v_name].attrs["delta_kind"] = _kind
        deltas[v_name].attrs["delta_reference"] = reference_horizon

        for a in ["description", "long_name"]:
            update_attr(
                deltas[v_name],
                a,
                _("{attr1}: {kind} delta compared to {refhoriz}."),
                others=[other_hz[vv]],
                refhoriz=reference_horizon,
                kind=_kind,
            )

        new_history = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {_kind} delta vs. {reference_horizon} - xarray v{xr.__version__}"
        history = new_history + " \n " + deltas[v_name].attrs["history"] if "history" in deltas[v_name].attrs else new_history
        deltas[v_name].attrs["history"] = history

    if "time" in ds:
        # get back to 1D time
        deltas = deltas.stack(time=("year", "month", "day"))
        # rebuild time coord
        if isinstance(ds.indexes["time"], pd.core.indexes.datetimes.DatetimeIndex):
            time_coord = list(
                pd.to_datetime(
                    {
                        "year": deltas.year.values,
                        "month": deltas.month.values,
                        "day": deltas.day.values,
                    }
                ).values
            )
        elif isinstance(ds.indexes["time"], xr.coding.cftimeindex.CFTimeIndex):
            time_coord = [
                xclim.core.calendar.datetime_classes[ds.time.dt.calendar](y, m, d)
                for y, m, d in zip(deltas.year.values, deltas.month.values, deltas.day.values, strict=False)
            ]
        else:
            raise ValueError("The type of 'time' could not be understood.")

        deltas = deltas.drop_vars({"year", "day", "month", "time"}).assign(time=time_coord).transpose("time", ...)
        deltas = deltas.reindex_like(ds)

    if to_level is not None:
        deltas.attrs["cat:processing_level"] = to_level

    return deltas


@parse_config
def spatial_mean(  # noqa: C901
    ds: xr.Dataset,
    method: str,
    *,
    spatial_subset: bool | None = None,
    region: dict | str | None = None,
    simplify_tolerance: float | None = None,
    kwargs: dict | None = None,
    to_domain: str | None = None,
    to_level: str | None = None,
) -> xr.Dataset:
    r"""
    Compute the spatial mean using a variety of available methods.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to use for the computation.
    method : str
        'cos-lat' will weight the area covered by each pixel using an approximation based on latitude.
        'xesmf' will make use of xESMF's SpatialAverager. This will typically be more precise,
        especially for irregular regions, but can be much slower than other methods.
    spatial_subset : bool, optional
        If True, xscen.spatial.subset will be called prior to the other operations. This requires the 'region' argument.
        If None, this will automatically become True if 'region' is provided and the subsetting method is 'cos-lat'.
    region : dict or str, optional
        Description of the region and the subsetting method (required fields listed in the Notes).
        If method=='xesmf', the bounding box or shapefile is given to SpatialAverager.
        Can also be "global", for global averages.
        The latter is simply a shortcut for `{'name': 'global', 'method': 'bbox', 'lon_bnds' [-180, 180], 'lat_bnds': [-90, 90]}`.
    simplify_tolerance : float, optional
        Only used with 'xesmf' method. Precision (in degree) used to simplify a shapefile before sending it to SpatialAverager().
        The simpler the polygons, the faster the averaging, but it will lose some precision.
    kwargs : dict, optional
        Arguments to send to SpatialAverager().
        In addition, also for SpatialAverager, one can give `skipna` or  `output_chunks` here, to be passed to the averager call itself.
        Additionally, for all methods, the 'latitude_name' and 'longitude_name' can be passed here to specify the names of those coordinates.
    to_domain : str, optional
        The domain to assign to the output.
        If None, the domain of the inputs is preserved.
    to_level : str, optional
        The processing level to assign to the output.
        If None, the processing level of the inputs is preserved.

    Returns
    -------
    xr.Dataset
        Returns a Dataset with the spatial dimensions averaged.

    Notes
    -----
    'region' required fields:
        name: str
            Region name used to overwrite domain in the catalog.
        method: str
            ['gridpoint', 'bbox', shape', 'sel']
        tile_buffer: float, optional
            Multiplier to apply to the model resolution. Only used if spatial_subset==True.
        \*\*kwargs
            Arguments specific to the method used.

    See Also
    --------
    xesmf.SpatialAverager
    """
    kwargs = (kwargs or {}).copy()

    # Determine the coordinates
    i = 0
    while ("latitude" not in ds.cf.coordinates or "longitude" not in ds.cf.coordinates) and i < 2:
        if i == 0:
            # First, try to guess the latitude coordinate using CF conventions
            ds = ds.cf.guess_coord_axis()
        elif i == 1:
            missing_coords = [f"{coord}_name" for coord in ["latitude", "longitude"] if coord not in ds.cf.coordinates]
            # If the coordinates are still not found, use the 'kwargs' argument
            if any(coord not in kwargs for coord in missing_coords):
                raise ValueError(
                    "Could not determine the spatial coordinate names using CF conventions. "
                    "Use kwargs = {latitude_name: str, longitude_name: str} to specify them."
                )
            for coord in missing_coords:
                ds[kwargs[coord]].attrs["standard_name"] = coord.split("_")[0]
        i += 1
    kwargs.pop("latitude_name", None)
    kwargs.pop("longitude_name", None)

    if region == "global":
        region = {
            "name": "global",
            "method": "bbox",
            "lat_bnds": [-90 + 1e-5, 90 - 1e-5],
        }
        # `spatial_subset` won't wrap coords on the bbox, we need to fit the system used on ds.
        if ds.cf["longitude"].min() >= 0:
            region["lon_bnds"] = [0, 360]
        else:
            region["lon_bnds"] = [-180, 180]

    if (region is not None) and (spatial_subset is None) and (method == "cos-lat"):
        logger.info("Automatically turning spatial_subset to True based on inputs.")
        spatial_subset = True

    # If requested, call xscen.spatial.subset prior to averaging
    if spatial_subset:
        ds = subset(ds, **region)

    if method == "cos-lat":
        if ds.cf["latitude"].attrs.get("units", "") != "degrees_north":
            warnings.warn(
                f"Latitude units are '{ds.cf['latitude'].attrs.get('units', '')}', "
                "expected 'degrees_north'. Make sure that the computation is right.",
                stacklevel=2,
            )

        if ((ds.cf["longitude"].min() < -160) & (ds.cf["longitude"].max() > 160)) or (
            (ds.cf["longitude"].min() < 20) & (ds.cf["longitude"].max() > 340)
        ):
            warnings.warn(
                "The region appears to be crossing the -180/180° meridian. Bounds computation is currently bugged in cf_xarray. "
                "Make sure that the computation is right.",
                stacklevel=2,
            )

        weights = np.cos(np.deg2rad(ds.cf["latitude"]))
        if ds.cf["longitude"].ndim == 1:
            dims = ds.cf["longitude"].dims + ds.cf["latitude"].dims
        else:
            if "longitude" not in ds.cf.bounds:
                ds = ds.cf.add_bounds(["longitude", "latitude"])
            # Weights the weights by the cell area (in °²)
            weights = weights * xr.DataArray(
                shapely.area(shapely.polygons(shapely.linearrings(ds.lon_bounds, ds.lat_bounds))),
                dims=ds.cf["longitude"].dims,
                coords=ds.cf["longitude"].coords,
            )
            dims = ds.cf["longitude"].dims
        ds_agg = ds.weighted(weights).mean(dims, keep_attrs=True)

        # Prepare the History field
        new_history = (
            f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"weighted mean(dim={dims}) using a 'cos-lat' approximation of areacella (in deg2)"
        )

    elif method == "interp_centroid":
        # FIXME: No backward compatibility for this method since it produced incorrect results, but we want a specific error message for now.
        raise ValueError("Method 'interp_centroid' has been removed, since it produced incorrect results.")

    # Uses xesmf.SpatialAverager
    elif method == "xesmf":
        if xe is None:
            raise ImportError("Method xesmf requires xESMF to work, please install that package.")

        # If the region is a bounding box, call shapely and geopandas to transform it into an input compatible with xesmf
        if region["method"] == "bbox":
            polygon_geom = shapely.box(
                region["lon_bnds"][0],
                region["lat_bnds"][0],
                region["lon_bnds"][1],
                region["lat_bnds"][1],
            )
            polygon = gpd.GeoDataFrame(index=[0], geometry=[polygon_geom])

            # Prepare the History field
            new_history = (
                f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"xesmf.SpatialAverager over {region['lon_bnds']}{region['lat_bnds']} - xESMF v{xe.__version__}"
            )

        # If the region is a shapefile, open with geopandas
        elif region["method"] == "shape":
            if not isinstance(region["shape"], gpd.GeoDataFrame):
                polygon = gpd.read_file(region["shape"])
                name = Path(region["shape"]).name
            else:
                polygon = region["shape"]
                name = f"{len(polygon)} polygons"

            # Simplify the geometries to a given tolerance, if needed.
            # The simpler the polygons, the faster the averaging, but it will lose some precision.
            if simplify_tolerance is not None:
                polygon = polygon.copy()
                polygon["geometry"] = polygon.simplify(tolerance=simplify_tolerance, preserve_topology=True)

            # Prepare the History field
            new_history = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] xesmf.SpatialAverager over {name} - xESMF v{xe.__version__}"

        else:
            raise ValueError("'method' should be one of [bbox, shape].")

        kwargs_copy = deepcopy(kwargs)
        call_kwargs = {"skipna": kwargs_copy.pop("skipna", False)}
        if "output_chunks" in kwargs:
            call_kwargs["output_chunks"] = kwargs_copy.pop("output_chunks")

        # Preemptive segmentization. Same threshold as xESMF, but there isn't strong analysis behind this choice
        geoms = shapely.segmentize(polygon.geometry, 1)

        if ds.cf["longitude"].ndim == 2 and "longitude" not in ds.cf.bounds and "rotated_pole" in ds:
            from .regrid import create_bounds_gridmapping

            ds = ds.assign_coords(**create_bounds_gridmapping(ds))

        savg = xe.SpatialAverager(ds, geoms, **kwargs_copy)
        ds_agg = savg(ds, keep_attrs=True, **call_kwargs)

        geom_dim_name = kwargs_copy.pop("geom_dim_name", "geom")

        extra_coords = {col: xr.DataArray(polygon[col], dims=(geom_dim_name,)) for col in polygon.columns if col != "geometry"}
        extra_coords[geom_dim_name] = xr.DataArray(polygon.index, dims=(geom_dim_name,))
        ds_agg = ds_agg.assign_coords(**extra_coords)
        if len(polygon) == 1:
            ds_agg = ds_agg.squeeze(geom_dim_name)

    else:
        raise ValueError("Subsetting method should be ['cos-lat', 'interp_centroid', 'xesmf']")

    # If the dataset had a projection, remove it
    grid_mapping = get_grid_mapping(ds)
    if grid_mapping:
        ds_agg = ds_agg.drop_vars(grid_mapping)
        for v in ds_agg.data_vars:
            ds_agg[v].attrs.pop("grid_mapping", None)

    # History
    history = new_history + " \n " + ds_agg.attrs["history"] if "history" in ds_agg.attrs else new_history
    ds_agg.attrs["history"] = history

    # Attrs
    if to_domain is not None:
        ds_agg.attrs["cat:domain"] = to_domain
    if to_level is not None:
        ds_agg.attrs["cat:processing_level"] = to_level

    return ds_agg


@parse_config
def produce_horizon(  # noqa: C901
    ds: xr.Dataset,
    *,
    indicators: (str | os.PathLike | Sequence[Indicator] | Sequence[tuple[str, Indicator]] | ModuleType | None) = None,
    periods: list[str] | list[list[str]] | None = None,
    warminglevels: dict | None = None,
    op: str | dict = "mean",
    to_level: str | None = "horizons",
) -> xr.Dataset:
    """
    Compute indicators, then the climatological mean, and finally unstack dates in order
    to have a single dataset with all indicators of different frequencies.

    Once this is done, the function drops 'time' in favor of 'horizon'.
    This function computes the indicators and does an interannual mean.
    It stacks the season and month in different dimensions and adds a dimension `horizon` for the period or the warming level, if given.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with a time dimension.
        If 'indicators' is None, the dataset should contain the precomputed indicators.
    indicators :  str | os.PathLike | Sequence[Indicator] | Sequence[Tuple[str, Indicator]] | ModuleType, optional
        Indicators to compute. It will be passed to the `indicators` argument of `xs.compute_indicators`.
    periods : list of str or list of lists of str, optional
        Either [start, end] or list of [start_year, end_year] for the period(s) to be evaluated.
        If both periods and warminglevels are None, the full time series will be used.
    warminglevels : dict, optional
        Dictionary of arguments to pass to `py:func:xscen.subset_warming_level`.
        If 'wl' is a list, the function will be called for each value and produce multiple horizons.
        If both periods and warminglevels are None, the full time series will be used.
    op : str or dict
        Operation to perform over the time dimension. See `py:func:xscen.climatological_op` for details. Default is 'mean'.
    to_level : str, optional
        The processing level to assign to the output.
        If there is only one horizon, you can use "{wl}", "{period0}" and "{period1}" in the string to dynamically
        include that information in the processing level.

    Returns
    -------
    xr.Dataset
        Horizon dataset.
    """
    if "warminglevel" in ds and len(ds.warminglevel) != 1:
        raise ValueError(
            "Input dataset should only have `warminglevel` dimension of length 1. "
            "If you want to use produce_horizon for multiple warming levels, "
            "extract the full time series and use the `warminglevels` argument instead."
        )

    all_periods = []
    if periods is not None:
        all_periods.extend(standardize_periods(periods))
    if warminglevels is not None:
        if isinstance(warminglevels["wl"], int | float):
            all_periods.append(warminglevels)
        elif isinstance(warminglevels["wl"], list):
            template = deepcopy(warminglevels)
            for wl in warminglevels["wl"]:
                all_periods.append({**template, "wl": wl})
        else:
            raise ValueError(f"Could not understand the format of warminglevels['wl']: {warminglevels['wl']}")
    if len(all_periods) == 0:
        all_periods = standardize_periods([[int(ds.time.dt.year[0]), int(ds.time.dt.year[-1])]])

    out = []
    for period in all_periods:
        if isinstance(period, list):
            if ds.time.dt.year[0] <= int(period[0]) and ds.time.dt.year[-1] >= int(period[1]):
                ds_sub = ds.sel(time=slice(period[0], period[1]))
            else:
                warnings.warn(
                    f"The requested period {period} is not fully covered by the input dataset. The requested period will be skipped.", stacklevel=2
                )
                ds_sub = None
        else:
            ds_sub = subset_warming_level(ds, **period)

        if ds_sub is not None:
            if indicators is not None:
                # compute indicators
                ind_dict = compute_indicators(
                    ds=(ds_sub.squeeze(dim="warminglevel") if "warminglevel" in ds_sub.dims else ds_sub),
                    indicators=indicators,
                )
            else:
                ind_dict = {"skipped": ds_sub}

            # Compute the window-year mean
            ds_merge = xr.Dataset()
            for freq, ds_ind in ind_dict.items():
                if freq != "fx":
                    ds_mean = climatological_op(
                        ds_ind,
                        op=op,
                        rename_variables=False,
                        horizons_as_dim=True,
                    ).drop_vars("time")
                else:
                    ds_mean = ds_ind.expand_dims(dim={"horizon": [f"{ds.time.dt.year[0].item()}-{ds.time.dt.year[-1].item()}"]})
                    ds_mean["horizon"] = ds_mean["horizon"].astype(str)

                if "warminglevel" in ds_mean.coords:
                    wl = np.array([ds_mean["warminglevel"].item()])
                    wl_attrs = ds_mean["warminglevel"].attrs
                    if "warminglevel" in ds_mean.dims:
                        ds_mean = ds_mean.squeeze("warminglevel")
                    ds_mean = ds_mean.drop_vars("warminglevel")
                    ds_mean["horizon"] = wl
                    ds_mean["horizon"].attrs.update(wl_attrs)

                # put all indicators in one dataset
                for var in ds_mean.data_vars:
                    ds_merge[var] = ds_mean[var]
                ds_merge.attrs.update(ds_mean.attrs)

            out.append(ds_merge)

    # If automatic attributes are not the same for all indicators, warn the user.
    if len(out) > 0:
        for v in out[0].data_vars:
            if not all(
                [all([out[0][v].attrs.get(attr) == out[i][v].attrs.get(attr) for i in range(1, len(out))]) for attr in ["long_name", "description"]]
            ):
                warnings.warn(
                    f"The attributes for variable {v} are not the same for all horizons, "
                    "probably because the periods were not of the same length. "
                    "Attributes will be kept from the first horizon, but this might not be the most appropriate.",
                    stacklevel=2,
                )

        out = xr.concat(out, dim="horizon")

        out.attrs["cat:xrfreq"] = "fx"
        out.attrs["cat:frequency"] = "fx"
        if to_level:
            if len(all_periods) == 1:
                if (isinstance(all_periods[0], dict)) or ("warminglevel" in ds.dims and warminglevels is None):
                    to_level = to_level.format(wl=(ds_sub.warminglevel.values[0] if isinstance(all_periods[0], dict) else ds.warminglevel.values[0]))
                else:
                    to_level = to_level.format(period0=all_periods[0][0], period1=all_periods[0][1])
            out.attrs["cat:processing_level"] = to_level

        return out

    else:
        raise ValueError("No horizon could be computed. Check your inputs.")
