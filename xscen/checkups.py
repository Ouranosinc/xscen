import logging
from pathlib import Path, PosixPath
from types import ModuleType
from typing import Optional, Sequence, Tuple, Union

import matplotlib as mpl
import numpy as np
import xarray as xr
import xclim as xc
from cartopy import crs
from matplotlib import pyplot as plt
from xclim import sdba
from xclim.core.indicator import Indicator
from xclim.core.units import convert_units_to
from xclim.sdba import measures

from .catalog import DataCatalog
from .common import maybe_unstack, unstack_fill_nan
from .indicators import load_xclim_module
from .io import save_to_zarr

logger = logging.getLogger(__name__)


# TODO: Implement logging, warnings, etc.
# TODO: Change all paths to PosixPath objects, including in the catalog?


def fix_unphysical_values(
    catalog: DataCatalog,
):
    """
    Basic checkups for impossible values such as tasmin > tasmax, or negative precipitation

    Parameters
    ----------
    catalog : DataCatalog

    """

    if len(catalog.unique("processing_level")) > 1:
        raise NotImplementedError

    for domain in catalog.unique("domain"):
        for id in catalog.unique("id"):
            sim = catalog.search(id=id, domain=domain)

            # tasmin > tasmax
            if len(sim.search(variable=["tasmin", "tasmax"]).catalog) == 2:
                # TODO: Do something with the last output
                ds_tn = xr.open_zarr(
                    sim.search(variable="tasmin").catalog.df.iloc[0]["path"]
                )
                ds_tx = xr.open_zarr(
                    sim.search(variable="tasmax").catalog.df.iloc[0]["path"]
                )

                tn, tx, _ = _invert_unphysical_temperatures(
                    ds_tn["tasmin"], ds_tx["tasmax"]
                )
                ds_tn["tasmin"] = tn
                ds_tx["tasmax"] = tx
                # TODO: History, attrs, etc.

                save_to_zarr(
                    ds=ds_tn,
                    filename=sim.search(variable="tasmin").catalog.df.iloc[0]["path"],
                    zarr_kwargs={"mode": "w"},
                )
                save_to_zarr(
                    ds=ds_tx,
                    filename=sim.search(variable="tasmax").catalog.df.iloc[0]["path"],
                    zarr_kwargs={"mode": "w"},
                )

            # TODO: "clip" function for pr, sfcWind, huss/hurs, etc. < 0 (and > 1/100, when applicable).
            #   Can we easily detect the variables that need to be clipped ? From the units ?


def _invert_unphysical_temperatures(
    tasmin: xr.DataArray, tasmax: xr.Dataset
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Invert tasmin and tasmax points where tasmax <  tasmin.

    Returns
    -------
    tasmin : xr.DataArray
      New tasmin.
    tasmax : xr.DataArray
      New tasmax
    switched : xr.DataArray
      A scalar DataArray with the number of switched data points.
    """
    is_bad = tasmax < tasmin
    mask = tasmax.isel(time=0).notnull()
    switch = is_bad & mask
    tn = xr.where(switch, tasmax, tasmin)
    tx = xr.where(switch, tasmin, tasmax)

    tn.attrs.update(tasmin.attrs)
    tx.attrs.update(tasmax.attrs)
    return tn, tx, switch.sum()


# TODO: just measures?


def properties_and_measures(
    ds: xr.Dataset,
    properties: Union[
        str, PosixPath, Sequence[Indicator], Sequence[Tuple[str, Indicator]], ModuleType
    ],
    period: list = None,
    unstack: bool = False,
    dref_for_measure: Optional[xr.Dataset] = None,
    unit_conversion: Optional[dict] = None,
    to_level_prop: str = "diag-properties",
    to_level_meas: str = "diag-measures",
):
    """
    Calculate properties and measures of a dataset.

    Parameters
    ----------
    ds: xr.Dataset
        Input dataset.
    properties: Union[str, PosixPath, Sequence[Indicator], Sequence[Tuple[str, Indicator]]]
      Path to a YAML file that instructs on how to calculate properties.
      Can be the indicator module directly, or a sequence of indicators or a sequence of
      tuples (indicator name, indicator) as returned by `iter_indicators()`.
    period: lst
        [start, end] of the period to be evaluated. The period will be selected on ds
        and dref_for_measure it it is given.
    unstack: bool
        Whether to unstack ds before computing the properties.
    dref_for_measure: xr.Dataset
        Dataset of properties to be used as the ref argument in the computation of the measure.
        Ideally, this is the first output (prop) of a previous call to this function.
        Only measures on properties that are provided both in this dataset and in the properties list will be computed.
        If None, the second output of the function (meas) will be an empty Dataset.
    unit_conversion: dict
        Dictionary of unit conversion to apply to ds before computing the properties.
        It is useful to be able to convert units, because, for the measure, sim and ref need to have similar units.
    to_level_prop: str
        processing_level to give the first output (prop)
    to_level_meas
        processing_level to give the second output (meas)

    Returns
    -------
    prop: xr.Dataset
        Dataset of properties of ds
    meas: xr.Dataset
        Dataset of measures between prop and dref_for_meas

    """

    if isinstance(properties, (str, Path)):
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
        logger.info(f"Computing {N} indicators.")

    # select periods for ds
    if period is not None and "time" in ds:
        ds = ds.sel({"time": slice(str(period[0]), str(period[1]))})
        date_start = ds.time.values[0]
        date_end = ds.time.values[-1]
    # select periods for ref_measure
    if (
        dref_for_measure is not None
        and period is not None
        and "time" in dref_for_measure
    ):
        dref_for_measure = dref_for_measure.sel(
            {"time": slice(str(period[0]), str(period[1]))}
        )

    if unstack:
        ds = unstack_fill_nan(ds)

    unit_conversion = unit_conversion or {}
    for var, unit in unit_conversion.items():
        ds[var] = convert_units_to(ds[var], unit)

    prop = xr.Dataset()  # dataset with all properties
    meas = xr.Dataset()  # dataset with all measures
    for i, ind in enumerate(properties, 1):
        if isinstance(ind, tuple):
            iden, ind = ind
        else:
            iden = ind.identifier
        logger.info(f"{i} - Computing {iden}.")

        # Make the call to xclim
        prop[iden] = ind(ds=ds)

        # calculate the measure if a reference dataset is given for the measure
        if dref_for_measure and iden in dref_for_measure:
            meas[iden] = ind.get_measure()(sim=prop[iden], ref=dref_for_measure[iden])

    for ds1 in [prop, meas]:
        ds1.attrs = ds.attrs
        ds1.attrs["cat/xrfreq"] = "fx"
        ds1.attrs.pop("cat/variable", None)
        ds1.attrs["cat/frequency"] = "fx"
        ds1.attrs["cat/timedelta"] = "NAN"
        if period:
            ds1.attrs["cat/date_start"] = date_start
            ds1.attrs["cat/date_end"] = date_end
        # not erasing the date_start/end to keep track of what time range was collapsed

    prop.attrs["cat/processing_level"] = to_level_prop
    meas.attrs["cat/processing_level"] = to_level_meas

    return prop, meas


def heatmap(
    meas_datasets: list, name_of_datasets: list = None, to_level: str = "diag-heatmap"
):
    """
    Create a heat map to compare the performance of the different datasets.
    The columns are properties and the rows are datasets.
    Each point is the absolute value of the mean of the measure over the whole domain.
    Each column is normalized from 0 (best) to 1 (worst).

    Parameters
    ----------
    meas_datasets: list
        List of datasets of measures of properties.

    name_of_datasets: list
        List of names for meas_datasets
        If None, they will be given a number.
    to_level
        processing_level to assign to the output

    Returns
    -------
        xr.DataArray
    """
    hmap = []
    for meas in meas_datasets:
        row = []
        # iterate through all available properties
        for var_name in meas:
            da = meas[var_name]
            # mean the absolute value of the bias over all positions and add to heat map
            if "xclim.sdba.measures.RATIO" in da.attrs["history"]:
                # if ratio, best is 1, this moves "best to 0 to compare with bias
                row.append(abs(da - 1).mean().values)
            else:
                row.append(abs(da).mean().values)
        # append all properties
        hmap.append(row)

    # plot heat map of biases ( 1 column per properties, 1 column for sim , 1 column for scen)
    hmap = np.array(hmap)
    # normalize to 0-1 -> best-worst
    hmap = np.array(
        [
            (c - min(c)) / (max(c) - min(c)) if max(c) != min(c) else [0.5] * len(c)
            for c in hmap.T
        ]
    ).T

    name_of_datasets = name_of_datasets or list(range(1, hmap.shape[0] + 1))

    ds_hmap = xr.DataArray(
        hmap,
        coords={
            "datasets": name_of_datasets,
            "properties": list(meas.data_vars),
        },
        dims=["datasets", "properties"],
    )
    ds_hmap = ds_hmap.to_dataset(name="heatmap")

    ds_hmap.attrs = meas_datasets[0].attrs
    ds_hmap.attrs["processing_level"] = to_level
    ds_hmap.attrs.pop("cat/variable", None)

    return ds_hmap


def improved_grid_points(ds1, ds2, to_level: str = "diag-improvedgridpoints"):
    """
    Calculate the fraction of improved grid points for each properties between two datasets of measures.

    Parameters
    ----------
    ds1: xr.Dataset
        Initial dataset of measures
    ds2: xr.Dataset
        Final dataset of measures. Must have the same variables as ds1.
    to_level:
        processing_level to assign to the output dataset

    Returns
    -------
    xr.Dataset

    """

    percent_better = []
    for var in ds2.data_vars:
        if "xclim.sdba.measures.RATIO" in ds1.attrs["history"]:
            diff_bias = abs(ds1[var] - 1) - abs(ds2[var] - 1)
        else:
            diff_bias = abs(ds1[var]) - abs(ds2[var])
        diff_bias = diff_bias.values.ravel()
        diff_bias = diff_bias[~np.isnan(diff_bias)]

        total = ds2[var].values.ravel()
        total = total[~np.isnan(total)]

        improved = diff_bias >= 0
        percent_better.append(np.sum(improved) / len(total))

    ds_better = xr.DataArray(
        percent_better, coords={"properties": list(ds2.data_vars)}, dims="properties"
    )

    ds_better = ds_better.to_dataset(name="improved_grid_points")

    ds_better.attrs = ds2.attrs
    ds_better.attrs["cat/processing_level"] = to_level
    ds_better.attrs.pop("cat/variable", None)

    return ds_better


#


# fonction with all diags

# """
# Diagnostics for bias-adjustment
# -------------------------------
#
# This module defines graphic diagnostics for bias-adjustment outputs. It works like so:
#
# The `checkup_plots` function that will iterate over so-call `checkup_plot` functions for
# each variable and each site.
# Those `checkup_plot` are functions plotting one figure for one set of 1D timeseries.
# Their signature must look like:
#
#     ..code:
#
#     @register_checkup_plot(period)
#     def plotting_func(fig, obs, raw, sce, **kwargs):
#         # plotting code
#         return fig
#
# Where `period` is one of "present", "future" or None. Inputs `obs`, `raw` and `sce` are 1D daily timeseries of the reference, un-adjusted and adjusted data. Their name is the variable name.
#
# - If period is "present", all three inputs are defined on the reference period.
# - If period is "future", `obs` is missing and `raw` and `sce` are on a future period.
# - If period is ``None``, the inputs are not subsetted, so they might not be the same length (`obs` should be shorter).
# """
# import itertools
# from pathlib import Path
# import warnings
#
# from clisops.core import subset
# import numpy as np
# import xarray as xr
# import xclim as xc
#
# import matplotlib.pyplot as plt
#
#
# from .config import parse_config
# _checkup_plot_funcs = {}
#
#
# def register_checkup_plot(period=None):
#     assert period in [None, 'present', 'future']
#
#     def _register_func(func):
#         func.__dict__['period'] = period
#         _checkup_plot_funcs[func.__name__] = func
#         return func
#     return _register_func
#
#
# def _get_coord_str(da):
#     return ', '.join(f"{name}={crd.item():.2f}" for name, crd in da.coords.items() if name != 'time')
#
#
# @register_checkup_plot('present')
# @parse_config
# def annual_cycle_monthly_present(
#     fig, obs, raw, sce,
#     *,
#     labels=['obs', 'raw', 'scen'], colors='krb', alpha=0.1
# ):
#     """Plot of the annual cycle (monthly agg), for the present period"""
#     var = obs.name
#     aggfunc = {
#         'pr': xc.atmos.precip_accumulation,
#         'tasmin': xc.atmos.tn_mean,
#         'tasmax': xc.atmos.tx_mean
#     }[var]
#
#     obs = aggfunc(obs, freq='MS')
#     raw = aggfunc(raw, freq='MS')
#     sce = aggfunc(sce, freq='MS')
#
#     ax = fig.add_subplot(1, 1, 1)
#
#     for da, label, color in zip([obs, raw, sce], labels, colors):
#         g = da.groupby('time.month')
#         mn = g.min()
#         mx = g.max()
#         ax.fill_between(mn.month, mn, mx, color=color, alpha=alpha)
#         g.mean().plot(ax=ax, color=color, linestyle='-', label=label)
#
#     ax.set_title(f"Annual cycle (monthly means), {var} at {_get_coord_str(obs)} - present")
#     ax.legend()
#     fig.tight_layout()
#     return fig
#
#
# @register_checkup_plot('present')
# @parse_config
# def annual_cycle_daily_present(
#     fig, obs, raw, sce,
#     *,
#     labels=['obs', 'raw', 'scen'], colors='krb', alpha=0.1
# ):
#     """Plot of the annual cycle (day-of-year agg), for the present period."""
#     var = obs.name
#
#     ax = fig.add_subplot(1, 1, 1)
#     for da, label, color in zip([obs, raw, sce], labels, colors):
#         g = da.groupby('time.dayofyear')
#         mn = g.min()
#         mx = g.max()
#         ax.fill_between(mn.dayofyear, mn, mx, color=color, alpha=alpha)
#         g.mean().plot(ax=ax, color=color, linestyle='-', label=label)
#
#     ax.set_title(f"Annual cycle, {var} at {_get_coord_str(obs)} - present")
#     ax.legend()
#     fig.tight_layout()
#     return fig
#
#
# @register_checkup_plot('future')
# @parse_config
# def annual_cycle_monthly_future(
#     fig, raw, sce,
#     *,
#     labels=['raw', 'scen'], colors='rb', alpha=0.1
# ):
#     """Plot of the annual cycle (monthly agg), for the future period (no obs)"""
#     var = raw.name
#     aggfunc = {
#         'pr': xc.atmos.precip_accumulation,
#         'tasmin': xc.atmos.tn_mean,
#         'tasmax': xc.atmos.tx_mean
#     }[var]
#
#     raw = aggfunc(raw, freq='MS')
#     sce = aggfunc(sce, freq='MS')
#
#     ax = fig.add_subplot(1, 1, 1)
#
#     for da, label, color in zip([raw, sce], labels, colors):
#         g = da.groupby('time.month')
#         mn = g.min()
#         mx = g.max()
#         ax.fill_between(mn.month, mn, mx, color=color, alpha=alpha)
#         g.mean().plot(ax=ax, color=color, linestyle='-', label=label)
#
#     ax.set_title(f"Annual cycle (monthly means), {var} at {_get_coord_str(raw)} - future")
#     ax.legend()
#     fig.tight_layout()
#     return fig
#
#
# @register_checkup_plot('future')
# @parse_config
# def annual_cycle_daily_future(
#     fig, raw, sce,
#     *,
#     labels=['raw', 'scen'], colors='rb', alpha=0.1
# ):
#     """Plot of the annual cycle (day-of-year agg), for the future period (no obs)"""
#     var = raw.name
#
#     ax = fig.add_subplot(1, 1, 1)
#     for da, label, color in zip([raw, sce], labels, colors):
#         g = da.groupby('time.dayofyear')
#         mn = g.min()
#         mx = g.max()
#         ax.fill_between(mn.dayofyear, mn, mx, color=color, alpha=alpha)
#         g.mean().plot(ax=ax, color=color, linestyle='-', label=label)
#
#     ax.set_title(f"Annual cycle, {var} at {_get_coord_str(raw)} - future")
#     ax.legend()
#     fig.tight_layout()
#     return fig
#
#
# @register_checkup_plot()
# @parse_config
# def annual_mean_timeseries(
#     fig, obs, raw, sce,
#     *,
#     labels=['obs', 'raw', 'scen'], colors='krb',
# ):
#     """Plot the timeseries of the annual means."""
#     var = obs.name
#     aggfunc = {
#         'pr': xc.atmos.precip_accumulation,
#         'tasmin': xc.atmos.tn_mean,
#         'tasmax': xc.atmos.tx_mean
#     }[var]
#
#     obs = aggfunc(obs, freq='YS')
#     raw = aggfunc(raw, freq='YS')
#     sce = aggfunc(sce, freq='YS')
#
#     ax = fig.add_subplot(1, 1, 1)
#
#     for da, label, color in zip([obs, raw, sce], labels, colors):
#         da.plot(ax=ax, color=color, linestyle='-', label=label)
#
#     ax.set_title(f"Annual mean timeseries. {var} at {_get_coord_str(obs)}")
#     ax.legend()
#     fig.tight_layout()
#     return fig
#
#
# @register_checkup_plot()
# @parse_config
# def annual_maximum_timeseries(
#     fig, obs, raw, sce,
#     *,
#     labels=['obs', 'raw', 'scen'], colors='krb',
# ):
#     """Plot the timeseries of the annual maximums."""
#     var = obs.name
#     aggfunc = {
#         'pr': xc.atmos.max_1day_precipitation_amount,
#         'tasmin': xc.atmos.tn_max,
#         'tasmax': xc.atmos.tx_max
#     }[var]
#
#     obs = aggfunc(obs, freq='YS')
#     raw = aggfunc(raw, freq='YS')
#     sce = aggfunc(sce, freq='YS')
#
#     ax = fig.add_subplot(1, 1, 1)
#
#     for da, label, color in zip([obs, raw, sce], labels, colors):
#         da.plot(ax=ax, color=color, linestyle='-', label=label)
#
#     ax.set_title(f"Annual maximum timeseries. {var} at {_get_coord_str(obs)}")
#     ax.legend()
#     fig.tight_layout()
#     return fig
#
#
# @register_checkup_plot()
# @parse_config
# def annual_minimum_timeseries(
#     fig, obs, raw, sce,
#     *,
#     labels=['obs', 'raw', 'scen'], colors='krb',
# ):
#     """Plot the timeseries of the annual minimums."""
#     var = obs.name
#     aggfunc = {
#         'pr': lambda pr, freq: pr.resample(time=freq).min(),
#         'tasmin': xc.atmos.tn_min,
#         'tasmax': xc.atmos.tx_min
#     }[var]
#
#     obs = aggfunc(obs, freq='YS')
#     raw = aggfunc(raw, freq='YS')
#     sce = aggfunc(sce, freq='YS')
#
#     ax = fig.add_subplot(1, 1, 1)
#
#     for da, label, color in zip([obs, raw, sce], labels, colors):
#         da.plot(ax=ax, color=color, linestyle='-', label=label)
#
#     ax.set_title(f"Annual minimum timeseries. {var} at {_get_coord_str(obs)}")
#     ax.legend()
#     fig.tight_layout()
#     return fig
#
#
# @parse_config
# def checkup_plots(
#     obs, raw, sce,
#     points=None,
#     *,
#     checkups='all',
#     variables=['tasmin', 'tasmax', 'pr'],
#     num_points={'lat': 3, 'lon': 2},
#     present=('1981', '2010'),
#     future=('2071', '2100'),
#     labels=['obs', 'raw', 'sce'],
#     colors='krb',
#     figsize=(6, 4),
#     output_format='png',
#     output_dir='.',
#     output_points=None,
#     verbose=False
# ):
#     """Call checkup plotting functions iteratively.
#
#     Parameters
#     ----------
#     obs : xr.Dataset
#       The reference dataset for the bias-adjustment.
#       Must have the same calendar as `raw` and `sce`, but could be on a different grid.
#     raw: xr.Dataset
#       The simulated, non-adjusted data.
#     sce: xr.Dataset
#       The simulated and adjusted data.
#     points : xr.Dataset, optional
#       Sites to select for plotting. should have one variable per spatial dimension, all along dimension 'site'.
#       If not given, random points are selected using :py:func:`random_site_selection` and ``num_points``.
#     checkups : sequence of strings or "all"
#       Which plotting function to call. If "all", all the registered functions are called.
#     variables: sequence of strings
#       Which variables to plot.
#     num_points : int or sequence of ints or dict
#       How many points to choose, see arg `num` of ``random_site_selection``.
#     present : 2-tuple
#       The start and end of the reference period.
#     future : 2-tuple
#       The start and end of the future period.
#     labels : 3-tuple of strings
#       The name to give to each dataset in the plots.
#     colors : 3-tuple of colors
#       The colors to associate with each dataset in the plots. Any matplotlib-understandable object is fine.
#     figsize : 2-tuple of numbers
#       The size of the figure. See arg `figsize` of `matplotlib.pyplot.figure`.
#     output_format : string
#       The file extension for output figures.
#     output_dir : Pathlike
#       A path to the folder where to save all figures.
#     output_points : string, optional
#       A filename where to save the selected points are saved to a netCDF in the output_dir.
#       If None (default), they are not saved.
#     verbose : bool
#       If True, something is printed for each step of the work.
#     """
#     output_dir = Path(output_dir)
#
#     if points is None:
#         if verbose:
#             print('Creating mask and selecting points')
#         # Select middle point to avoid NaN slices on the sides.
#         # points on SCE because with the stack_drop_nans option, this is the only 2D ds.
#         mask = sce[variables[0]].isel(time=int(len(sce.time) / 2), drop=True).notnull().load()
#         points = random_site_selection(mask, num_points)
#
#     if output_points is not None:
#         points.to_netcdf(output_dir / output_points)
#
#     obs = subset.subset_gridpoint(obs, lon=points.lon, lat=points.lat)
#     raw = subset.subset_gridpoint(raw, lon=points.lon, lat=points.lat)
#     sce = subset.subset_gridpoint(sce, lon=points.lon, lat=points.lat)
#
#     # All checkup plot function MUST accept these.
#     kwargs = dict(labels=labels, colors=colors)
#     fut_kwargs = dict(labels=labels[1:], colors=colors[1:])
#
#     if checkups == 'all':
#         checkups = _checkup_plot_funcs.keys()
#
#     for var in variables:
#         for site in obs.site:
#             # We only load here to minimize memory use.
#             ob = obs[var].sel(site=site, drop=True).load()
#             ra = raw[var].sel(site=site, drop=True).load()
#             sc = sce[var].sel(site=site, drop=True).load()
#
#             # Ensure homogeneous units
#             ra = xc.core.units.convert_units_to(ra, ob)
#             sc = xc.core.units.convert_units_to(sc, ob)
#
#             for checkup in checkups:
#                 func = _checkup_plot_funcs[checkup]
#
#                 if verbose:
#                     print(f"Plotting {var}, site {site.item()}, {checkup} ...", end=" ")
#
#                 # Get the correct period for this plot
#                 period = {'present': present, 'future': future, None: (None, None)}[func.period]
#                 if func.period != 'future':
#                     o = ob.sel(time=slice(*period))
#                 r = ra.sel(time=slice(*period))
#                 s = sc.sel(time=slice(*period))
#
#                 fig = plt.figure(figsize=figsize)
#
#                 with warnings.catch_warnings():
#                     warnings.simplefilter('ignore')
#                     if func.period == 'future':
#                         fig = func(fig, r, s, **fut_kwargs)
#                     else:
#                         fig = func(fig, o, r, s, **kwargs)
#
#                 fig.savefig(output_dir / f"{var}_site{site.item()}_{checkup}.{output_format}")
#                 plt.close()
#                 if verbose:
#                     print("done.")
#
#
# def random_site_selection(mask, num=4):
#     """Select points (semi-)randomly from a grid.
#
#     Parameters
#     ----------
#     mask : xr.DataArray
#       Array with coords, from which points are taken.
#       Only points where mask is True are considered.
#     num : int or sequence of integers or dict
#       The number of points to consider. If a sequence or a dict, then the grid is divided in
#       as many zones and one point is selected in each. When a dict is given, it is a mapping from
#       dimension name to number of subzones along that dimension.
#
#     Return
#     ------
#     xarray.Dataset
#       One variable per coord of mask, along dimension "site".
#     """
#
#     if isinstance(num, (int, float)):
#         indexes = np.where(mask.values)
#
#         if len(indexes[0]) == 0:
#             raise ValueError('There are no True values in the mask.')
#
#         randindexes = np.random.default_rng().integers(
#             0, high=len(indexes[0]), size=(num,))
#
#         return xr.Dataset(
#             data_vars={
#                 name: mask[name].isel(
#                     {name: xr.DataArray(index[randindexes], dims=('site',), name='site')})
#                 for name, index in zip(mask.dims, indexes)
#             }
#         )
#     # else : divide in zones
#
#     if isinstance(num, dict):
#         num = [num[dim] for dim in mask.dims]
#
#     slices = [
#         [slice(i * (L // N), (i + 1) * (L // N)) for i in range(N)]
#         for L, N in zip(mask.shape, num)
#     ]
#
#     points = []
#     for zone_slices in itertools.product(*slices):
#         zone = mask.isel({dim: slic for dim, slic in zip(mask.dims, zone_slices)})
#         points.append(random_site_selection(zone, 1))
#
#     return xr.concat(points, 'site')


def fig_compare_and_diff(
    sim: xr.DataArray, scen: xr.DataArray, op: str = "difference", title: str = ""
) -> plt.figure:
    """Plot sim and scen, as well the their difference, improvement or distance.

    Parameters
    ----------
    sim: xr.DataArray
        First data to plot (normally an xclim.sbda.properties of an unajusted simulation, but could be anything)
    scen: xr.DataArray
        Second data to plot (normally an xclim.sbda.properties of an ajusted simulation, but could be anything)
    op: str
        Operation for the third panel
        If 'distance': abs(sim - scen)
        If 'improvement': abs(sim) - abs(scen)
        If 'difference': sim - scen
    title: str
        Title of the figure
    """
    cmbias = mpl.cm.BrBG.copy()
    cmbias.set_bad("gray")
    cmimpr = mpl.cm.RdBu.copy()
    cmimpr.set_bad("gray")

    fig = plt.figure(figsize=(15, 5))
    gs = mpl.gridspec.GridSpec(6, 2, hspace=2)
    axsm = plt.subplot(gs[3:, 0], projection=crs.PlateCarree())
    axsc = plt.subplot(gs[3:, 1], projection=crs.PlateCarree())
    axim = plt.subplot(gs[:3, 1], projection=crs.PlateCarree())

    vmin = min(
        min(sim.quantile(0.05), scen.quantile(0.05)),
        -max(sim.quantile(0.95), scen.quantile(0.95)),
    )

    ps = sim.plot(
        ax=axsm,
        vmin=vmin,
        vmax=-vmin,
        cmap=cmbias,
        add_colorbar=False,
        transform=crs.PlateCarree(),
    )
    scen.plot(
        ax=axsc,
        vmin=vmin,
        vmax=-vmin,
        cmap=cmbias,
        add_colorbar=False,
        transform=crs.PlateCarree(),
    )
    if op == "distance":
        diff = abs(sim - scen)
    elif op == "improvement":
        diff = abs(sim) - abs(scen)
    elif op == "difference":  # op == 'diff':
        diff = sim - scen
    else:
        raise NotImplementedError(
            "Operation not implemented. Try 'distance', 'improvement' or 'difference' instead."
        )
    pc = diff.plot(
        ax=axim,
        robust=True,
        cmap=cmimpr,
        center=0,
        add_colorbar=False,
        transform=crs.PlateCarree(),
    )

    fig.suptitle(title, fontsize="x-large")
    axsm.set_title(sim.name.replace("_", " ").capitalize())
    axsc.set_title(scen.name.replace("_", " ").capitalize())
    axim.set_title(op.capitalize())
    fig.tight_layout()
    fig.colorbar(
        ps,
        cax=plt.subplot(gs[1, 0]),
        shrink=0.5,
        label=sim.attrs.get("long_name", sim.name),
        orientation="horizontal",
    )
    fig.colorbar(
        pc,
        cax=plt.subplot(gs[0, 0]),
        shrink=0.5,
        label=op.capitalize(),
        orientation="horizontal",
    )
    return fig


def fig_bias_compare_and_diff(ref, sim, scen, measure: str = "bias", **kwargs):
    """Like fig_compare_and_diff, but calculates xclim.sdba measure first."""
    bsim = getattr(measures, measure)(sim, ref).rename(f"sim_{measure}")
    bscen = getattr(measures, measure)(scen, ref).rename(f"scen_{measure}")
    kwargs.setdefault("op", "improvement")
    kwargs.setdefault("title", f"Comparing {measure} of {ref.name}, sim vs scen")
    return fig_compare_and_diff(bsim, bscen, **kwargs)
