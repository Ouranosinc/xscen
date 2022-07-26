import datetime
import logging
from pathlib import Path
from typing import Union

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
from shapely.geometry import Polygon

from .config import parse_config
from .extraction import clisops_subset

logger = logging.getLogger(__name__)


@parse_config
def climatological_mean(
    ds: xr.Dataset,
    *,
    window: int = None,
    min_periods: int = None,
    interval: int = 1,
    periods: list = None,
    to_level: str = None,
) -> xr.Dataset:
    """
    Computes the mean over 'year' for given time periods, respecting the temporal resolution of ds.

    Parameters
    ----------
    ds : xr.Dataset
      Dataset to use for the computation.
    window: int
      Number of years to use for the time periods.
    min_periods: int
      For the rolling operation, minimum number of years required for a value to be computed.
      If left at None, it will be deemed the same as 'window'
    interval: int
      Interval (in years) at which to provide an output.
    periods: list
      list of [start, end] of the contiguous periods to be evaluated, in the case of disjointed datasets.
      If left at None, the dataset will be considered continuous.
    to_level : str, optional
      The processing level to assign to the output.
      If None, the processing level of the inputs is preserved.

    Returns
    -------
    xr.Dataset
      Returns a Dataset with the requested operation applied over time.

    """

    min_periods = min_periods or window

    # separate 1d time in coords (day, month, and year) to make climatological mean faster
    ind = pd.MultiIndex.from_arrays(
        [ds.time.dt.year.values, ds.time.dt.month.values, ds.time.dt.day.values],
        names=["year", "month", "day"],
    )
    ds_unstack = ds.assign(time=ind).unstack("time")

    # Rolling will ignore jumps in time, so we want to raise an exception beforehand
    if (not all(ds_unstack.year.diff(dim="year", n=1) == 1)) & (periods is None):
        raise ValueError("Data is not contiguous. Use the 'periods' argument.")

    # Compute temporal means
    concats = []
    periods = periods or [[int(ds_unstack.year[0]), int(ds_unstack.year[-1])]]
    for period in periods:
        # Rolling average
        ds_rolling = (
            ds_unstack.sel(year=slice(str(period[0]), str(period[1])))
            .rolling(year=window, min_periods=min_periods)
            .mean()
        )

        # Select every horizons in 'x' year intervals, starting from the first full windowed mean
        ds_rolling = ds_rolling.isel(
            year=slice(window - 1, None)
        )  # Select from the first full windowed mean
        intervals = ds_rolling.year.values % interval
        ds_rolling = ds_rolling.sel(year=(intervals - intervals[0] == 0))
        horizons = xr.DataArray(
            [f"{yr - (window - 1)}-{yr}" for yr in ds_rolling.year.values],
            dims=dict(year=ds_rolling.year),
        ).astype(str)
        ds_rolling = ds_rolling.assign_coords(horizon=horizons)

        # get back to 1D time
        ds_rolling = ds_rolling.stack(time=("year", "month", "day"))
        # rebuild time coord
        time_coord = [
            pd.to_datetime(f"{y - window + 1}, {m}, {d}")
            for y, m, d in zip(
                ds_rolling.year.values, ds_rolling.month.values, ds_rolling.day.values
            )
        ]
        ds_rolling = ds_rolling.assign_coords(time=time_coord).transpose(
            "time", "lat", "lon"
        )

        concats.extend([ds_rolling])
    ds_rolling = xr.concat(concats, dim="time", data_vars="minimal")

    # modify attrs and history
    for vv in ds_rolling.data_vars:
        for a in ["description", "long_name"]:
            if hasattr(ds_rolling[vv], a):
                ds_rolling[vv].attrs[
                    a
                ] = f"{window}-year mean of {ds_rolling[vv].attrs[a]}"

        new_history = (
            f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {window}-year rolling average (non-centered) "
            f"with a minimum of {min_periods} years of data - xarray v{xr.__version__}"
        )
        history = (
            new_history + " \n " + ds_rolling[vv].attrs["history"]
            if "history" in ds_rolling[vv].attrs
            else new_history
        )
        ds_rolling[vv].attrs["history"] = history

    if to_level is not None:
        ds_rolling.attrs["cat/processing_level"] = to_level

    return ds_rolling


@parse_config
def compute_deltas(
    ds: xr.Dataset,
    reference_horizon: str,
    *,
    kind: Union[str, dict] = "+",
    to_level: str = None,
) -> xr.Dataset:
    """
    Computes deltas in comparison to a reference time period, respecting the temporal resolution of ds.

    Parameters
    ----------
    ds : xr.Dataset
      Dataset to use for the computation.
    reference_horizon: str
      YYYY-YYYY string corresponding to the 'horizon' coordinate of the reference period.
    kind: str
      ['+', '/'] Whether to provide absolute or relative deltas.
      Can also be a dictionary separated per variable name.
    to_level : str, optional
      The processing level to assign to the output.
      If None, the processing level of the inputs is preserved.

    Returns
    -------
    xr.Dataset
      Returns a Dataset with the requested deltas.

    """

    # Separate the reference from the other horizons
    ref = ds.where(ds.horizon == reference_horizon, drop=True)
    # Remove references to 'year' in REF
    ind = pd.MultiIndex.from_arrays(
        [ref.time.dt.month.values, ref.time.dt.day.values], names=["month", "day"]
    )
    ref = ref.assign(time=ind).unstack("time")

    other_hz = ds.where(ds.horizon != reference_horizon, drop=True)
    ind = pd.MultiIndex.from_arrays(
        [
            other_hz.time.dt.year.values,
            other_hz.time.dt.month.values,
            other_hz.time.dt.day.values,
        ],
        names=["year", "month", "day"],
    )
    other_hz = other_hz.assign(time=ind).unstack("time")

    deltas = xr.Dataset(coords=other_hz.coords, attrs=other_hz.attrs)
    # Calculate deltas
    for vv in list(ds.data_vars):
        if (isinstance(kind, dict) and kind[vv] == "+") or kind == "+":
            _kind = "absolute"
        elif (isinstance(kind, dict) and kind[vv] == "/") or kind == "/":
            _kind = "relative"
        else:
            raise ValueError("Delta 'kind' not understood.")

        with xr.set_options(keep_attrs=True):
            if _kind == "absolute":
                deltas[f"{vv}_delta_{reference_horizon.replace('-', '_')}"] = (
                    other_hz[vv] - ref[vv]
                )
            else:
                deltas[f"{vv}_delta_{reference_horizon.replace('-', '_')}"] = (
                    other_hz[vv] / ref[vv]
                )
                deltas[f"{vv}_delta_{reference_horizon.replace('-', '_')}"].attrs[
                    "units"
                ] = ""

        # modify attrs and history
        deltas[f"{vv}_delta_{reference_horizon.replace('-', '_')}"].attrs[
            "delta_kind"
        ] = _kind
        deltas[f"{vv}_delta_{reference_horizon.replace('-', '_')}"].attrs[
            "delta_reference"
        ] = reference_horizon

        for a in ["description", "long_name"]:
            if hasattr(other_hz[vv], a):
                deltas[f"{vv}_delta_{reference_horizon.replace('-', '_')}"].attrs[
                    a
                ] = f"{other_hz[vv].attrs[a]}: {_kind} delta compared to {reference_horizon.replace('-', '_')}."

        new_history = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {_kind} delta vs. {reference_horizon} - xarray v{xr.__version__}"
        history = (
            new_history
            + " \n "
            + deltas[f"{vv}_delta_{reference_horizon.replace('-', '_')}"].attrs[
                "history"
            ]
            if "history"
            in deltas[f"{vv}_delta_{reference_horizon.replace('-', '_')}"].attrs
            else new_history
        )
        deltas[f"{vv}_delta_{reference_horizon.replace('-', '_')}"].attrs[
            "history"
        ] = history

    # get back to 1D time
    deltas = deltas.stack(time=("year", "month", "day"))
    # rebuild time coord
    time_coord = [
        pd.to_datetime(f"{y}, {m}, {d}")
        for y, m, d in zip(deltas.year.values, deltas.month.values, deltas.day.values)
    ]
    deltas = deltas.assign_coords(time=time_coord).transpose("time", "lat", "lon")

    if to_level is not None:
        deltas.attrs["cat/processing_level"] = to_level

    return deltas


def spatial_mean(
    ds: xr.Dataset,
    method: str,
    *,
    call_clisops: bool = False,
    region: dict = None,
    kwargs: dict = None,
    simplify_tolerance: float = None,
    to_domain: str = None,
    to_level: str = None,
) -> xr.Dataset:
    """
    Computes the spatial mean, using a variety of available methods.

    Parameters
    ----------
    ds: xr.Dataset
      Dataset to use for the computation.
    method: str
      'mean' will perform a .mean() over the spatial dimensions of the Dataset.
      'interp_coord' will (optionally) find the region's centroid, then perform a .interp() over the spatial dimensions of the Dataset.
      The coordinate can also be directly fed to .interp() through the 'kwargs' argument below.
      'xesmf' will make use of xESMF's SpatialAverager. Note that this can be much slower than other methods.
    call_clisops: bool
      If True, xscen.extraction.clisops_subset will be called prior to the other operations. This requires the 'region' argument.
    region: dict
      Description of the region and the subsetting method (required fields listed in the Notes).
      If method=='interp_coord', this is used to find the region's centroid.
      If method=='xesmf', the bounding box or shapefile is given to SpatialAverager.
    kwargs: dict
      Arguments to send to either interp() or SpatialAverager().
    simplify_tolerance: float
      Precision (in degree) used to simplify a shapefile before sending it to SpatialAverager().
      The simpler the polygons, the faster the averaging, but it will lose some precision.
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
        method: str
            ['gridpoint', 'bbox', shape']
        <method>: dict
            Arguments specific to the method used.
        buffer: float, optional
            Multiplier to apply to the model resolution. Only used if call_clisops==True.
    """

    kwargs = kwargs or {}

    # If requested, call xscen.extraction.clisops_subset prior to averaging
    if call_clisops:
        ds = clisops_subset(ds, region)

    # This simply calls .mean() over the spatial dimensions
    if method == "mean":
        # Determine the X and Y names
        spatial_dims = {}
        for d in ds.dims:
            if "axis" in ds[d].attrs and ds[d].attrs["axis"] in ["X", "x", "Y", "y"]:
                spatial_dims.update({ds[d].attrs["axis"].upper(): d})

        ds_agg = ds.mean(dim=list(spatial_dims.values()), keep_attrs=True)

        # Prepare the History field
        new_history = (
            f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"xarray.mean(dim={list(spatial_dims.values())}) - xarray v{xr.__version__}"
        )

    # This calls .interp() to a pair of coordinates
    elif method == "interp_coord":

        # Find the centroid
        if region is not None:
            if region["method"] == "gridpoint":
                if len(region["gridpoint"]["lon"] != 1):
                    raise ValueError(
                        "Only a single location should be used with interp_centroid."
                    )
                centroid = {
                    "lon": region["gridpoint"]["lon"],
                    "lat": region["gridpoint"]["lat"],
                }

            elif region["method"] == "bbox":
                centroid = {
                    "lon": np.mean(region["bbox"]["lon_bnds"]),
                    "lat": np.mean(region["bbox"]["lat_bnds"]),
                }

            elif region["method"] == "shape":
                s = gpd.read_file(region["shape"]["shape"])
                if len(s != 1):
                    raise ValueError(
                        "Only a single polygon should be used with interp_centroid."
                    )
                centroid = {"lon": s.centroid[0].x, "lat": s.centroid[0].y}
            else:
                raise ValueError("'method' not understood.")
            kwargs.update(centroid)

        ds_agg = ds.interp(**kwargs)

        new_history = (
            f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"xarray.interp(**{kwargs}) - xarray v{xr.__version__}"
        )

    # Uses xesmf.SpatialAverager
    elif method == "xesmf":

        # If the region is a bounding box, call shapely and geopandas to transform it into an input compatible with xesmf
        if region["method"] == "bbox":
            lon_point_list = [
                region["bbox"]["lon_bnds"][0],
                region["bbox"]["lon_bnds"][0],
                region["bbox"]["lon_bnds"][1],
                region["bbox"]["lon_bnds"][1],
            ]
            lat_point_list = [
                region["bbox"]["lat_bnds"][0],
                region["bbox"]["lat_bnds"][1],
                region["bbox"]["lat_bnds"][1],
                region["bbox"]["lat_bnds"][0],
            ]

            polygon_geom = Polygon(zip(lon_point_list, lat_point_list))
            polygon = gpd.GeoDataFrame(index=[0], geometry=[polygon_geom])

            # Prepare the History field
            new_history = (
                f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"xesmf.SpatialAverager over {region['bbox']['lon_bnds']}{region['bbox']['lat_bnds']} - xESMF v{xe.__version__}"
            )

        # If the region is a shapefile, open with geopandas
        elif region["method"] == "shape":
            polygon = gpd.read_file(region["shape"]["shape"])
            if len(polygon != 1):
                raise NotImplementedError(
                    "spatial_mean currently accepts only single polygons."
                )

            # Simplify the geometries to a given tolerance, if needed.
            # The simpler the polygons, the faster the averaging, but it will lose some precision.
            if simplify_tolerance is not None:
                polygon["geometry"] = polygon.simplify(
                    tolerance=simplify_tolerance, preserve_topology=True
                )

            # Prepare the History field
            new_history = (
                f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"xesmf.SpatialAverager over {Path(region['shape']['shape']).name} - xESMF v{xe.__version__}"
            )

        else:
            raise ValueError("'method' not understood.")

        savg = xe.SpatialAverager(ds, polygon.geometry, **kwargs)
        ds_agg = savg(ds, keep_attrs=True).isel(geom=0)

    else:
        raise ValueError(
            "Subsetting method should be ['mean', 'interp_coord', 'xesmf']"
        )

    # History
    history = (
        new_history + " \n " + ds_agg.attrs["history"]
        if "history" in ds_agg.attrs
        else new_history
    )
    ds_agg.attrs["history"] = history

    # Attrs
    if to_domain is not None:
        ds_agg.attrs["cat/domain"] = to_domain
    if to_level is not None:
        ds_agg.attrs["cat/processing_level"] = to_level

    return ds_agg
