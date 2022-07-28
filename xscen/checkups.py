from typing import Tuple

import matplotlib as mpl
import xarray as xr
from cartopy import crs
from matplotlib import pyplot as plt
from xclim.sdba import measures

from .catalog import DataCatalog
from .io import save_to_zarr

# TODO: Implement logging, warnings, etc.
# TODO: Change all paths to PosixPath objects, including in the catalog?


__all__ = ["fig_bias_compare_and_diff", "fig_compare_and_diff", "fix_unphysical_values", ]


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
