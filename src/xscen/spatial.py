"""Spatial tools."""

import datetime
import itertools
import logging
import warnings
from collections.abc import Sequence
from pathlib import Path

import cartopy.crs


try:
    import clisops  # For version info
    import clisops.core as cl
except KeyError as e:
    if e.args[0] == "Author":
        warnings.warn(
            "The clisops package could not be imported due to a known KeyError bug that occurs with some "
            "older versions of ESMF and specific execution setups (such as debugging on a Windows machine). "
            "As a workaround, try installing 'importlib-metadata <8.0.0' and/or updating ESMF. If you do not "
            "need 'clisops.core' functionalities (e.g. spatial subsetting), you can ignore this warning.",
            stacklevel=2,
        )
    else:
        raise e
    cl = None
    clisops = None
import dask
import geopandas as gpd
import numpy as np
import sparse as sp
import xarray as xr
import xclim as xc
from pyproj.crs import CRS

from .config import parse_config


logger = logging.getLogger(__name__)

__all__ = [
    "creep_fill",
    "creep_weights",
    "get_crs",
    "get_grid_mapping",
    "rotate_vectors",
    "subset",
]


@parse_config
def creep_weights(mask: xr.DataArray, n: int = 1, steps: int = 1, mode: str = "clip") -> xr.DataArray:
    """
    Compute weights for the creep fill.

    The output is a sparse matrix with the same dimensions as `mask`, twice.
    If `steps` is larger than 1, the output has a "step" dimension of that length.

    Parameters
    ----------
    mask : DataArray
      A boolean DataArray. False values are candidates to the filling.
      Usually they represent missing values (`mask = da.notnull()`).
      All dimensions are creep filled.
    n : int
      The order of neighbouring to use. 1 means only the adjacent grid cells are used.
    steps: int
      Apply the algorithm this number of times, creeping `n` neighbours at each step.
    mode : {'clip', 'wrap'}
      If a cell is on the edge of the domain, `mode='wrap'` will wrap around to find neighbours.

    Returns
    -------
    DataArray
       Weights. The dot product must be taken over the last N dimensions, in sequence for each step.
    """
    if mode not in ["clip", "wrap"]:
        raise ValueError("mode must be either 'clip' or 'wrap'")

    if steps > 1:
        da = xr.ones_like(mask).where(mask)
        weights = []
        for i in range(steps):
            w = creep_weights(da.notnull())
            weights.append(w)
            if i < steps - 1:  # no need to do it one the last step
                da = creep_fill(da, w)
        # TODO: If we treated the nan differently we could maybe collapse all weights with dot instead of having to apply them iteratively
        return xr.concat(weights, "step")

    da = mask
    mask = da.values
    neighbors = np.array(list(itertools.product(*[np.arange(-n, n + 1) for j in range(mask.ndim)]))).T
    src = []
    dst = []
    w = []
    it = np.nditer(mask, flags=["f_index", "multi_index"], order="C")
    for i in it:
        if not i:
            neigh_idx_2d = np.atleast_2d(it.multi_index).T + neighbors
            neigh_idx_1d = np.ravel_multi_index(neigh_idx_2d, mask.shape, order="C", mode=mode)
            if mode == "clip":
                neigh_idx = np.unravel_index(np.unique(neigh_idx_1d), mask.shape, order="C")
            elif mode == "wrap":
                neigh_idx = np.unravel_index(neigh_idx_1d, mask.shape, order="C")
            neigh = mask[neigh_idx]
            N = (neigh).sum()
            if N > 0:
                src.extend([it.multi_index] * N)
                dst.extend(np.stack(neigh_idx)[:, neigh].T)
                w.extend([1 / N] * N)
            else:
                src.extend([it.multi_index])
                dst.extend([it.multi_index])
                w.extend([np.nan])
        else:
            src.extend([it.multi_index])
            dst.extend([it.multi_index])
            w.extend([1])
    crds = np.concatenate((np.array(src).T, np.array(dst).T), axis=0)
    return xr.DataArray(
        sp.COO(crds, w, (*da.shape, *da.shape)),
        dims=[f"{d}_out" for d in da.dims] + list(da.dims),
        coords=da.coords,
        name="creep_fill_weights",
    )


@parse_config
def creep_fill(da: xr.DataArray, w: xr.DataArray) -> xr.DataArray:
    """
    Creep fill using pre-computed weights.

    Parameters
    ----------
    da: DataArray
      A DataArray sharing the dimensions with the one used to compute the weights.
      It can have other dimensions.
      Dask is supported as long as there are no chunks over the creeped dims.
    w: DataArray
      The result of `creep_weights`.

    Returns
    -------
    xarray.DataArray, same shape as `da`, but values filled according to `w`.

    Examples
    --------
    >>> w = creep_weights(da.isel(time=0).notnull(), n=1)
    >>> da_filled = creep_fill(da, w)
    """
    if "step" in w.dims:
        out = da
        for step in w.step:
            out = creep_fill(out, w.sel(step=step))
        return out

    def _dot(arr, wei):
        N = wei.ndim // 2
        extra_dim = arr.ndim - N
        return np.tensordot(arr, wei, axes=(np.arange(N) + extra_dim, np.arange(N) + N))

    N = w.ndim // 2
    return xr.apply_ufunc(
        _dot,
        da,
        w,
        input_core_dims=[w.dims[N:], w.dims],
        output_core_dims=[w.dims[N:]],
        dask="parallelized",
        output_dtypes=["float64"],
    )


def rotate_vectors(
    uu: xr.DataArray,
    vv: xr.DataArray,
    crs: cartopy.crs.Projection | None = None,
    reverse: bool = False,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Rotate a vector field defined in its grid space to the real south-north / west-east axes.

    Parameters
    ----------
    uu : xr.DataArray
      The u component of the vector (along the X grid axis).
      Must be CF-compliant, so its coordinates are correctly found.
    vv : xr.DataArray
      The v component of the vector (along the Y grid axis)
      Must be CF-compliant, so its coordinates are correctly found.
    crs : pyproj.CRS, optional
      The projection of the UU and VV grid. If not given, the grid mapping attribute from UU is read.
    reverse: bool
      If True, the opposite rotation is performed : inputs are understood as in the real S-N, W-E axes and are
      rotated back to the grid axes.

    Returns
    -------
    uuc : xr.DataArray
      The west-east component of the vector. Attributes from uu are copied.
    vvc : xr.DataArray
      The south-north component of the vector. Attributes from vv are copied.
    """
    if crs is None:
        crs = get_crs(uu.rename("uu").to_dataset())

    lalo = cartopy.crs.PlateCarree(globe=crs.globe)
    geod = lalo.get_geod()

    X = uu.cf["X"]
    Y = uu.cf["Y"]
    # ensure everything has the same dim order
    YY, XX = xr.broadcast(Y, X)
    XXarr = XX.values
    YYarr = YY.values
    # a very small increment along the y axis
    if X.dims == Y.dims:
        # we got a list of points, can't infer the grid resolution
        dy = 1e-5
    else:
        dy = (Y[1] - Y[0]).item() / 1000

    # Get lat lon of center points
    crds_c = lalo.transform_points(crs, XXarr, YYarr)
    # Get lat lon of points just slightly above along the y axis
    crds_y = lalo.transform_points(crs, XXarr, YYarr + dy)
    # The azimuth of this short vector
    az, _, _ = geod.inv(crds_c[..., 0], crds_c[..., 1], crds_y[..., 0], crds_y[..., 1])
    if reverse:
        az = -az
    # The rotation angle : opposite of the azimuth, in rad
    th = xr.DataArray(-np.deg2rad(az), dims=YY.dims, coords=YY.coords)
    # Rotation matrix like in Algèbre linéaire et géométrie vectorielle
    c = np.cos(th)
    s = np.sin(th)
    uuc = uu * c - vv * s
    vvc = uu * s + vv * c
    return uuc.assign_attrs(uu.attrs), vvc.assign_attrs(vv.attrs)


def subset(
    ds: xr.Dataset,
    method: str,
    *,
    name: str | None = None,
    tile_buffer: float = 0,
    **kwargs,
) -> xr.Dataset:
    r"""
    Subset the data to a region.

    Either creates a slice and uses the .sel() method, or customizes a call to
    clisops.subset() that allows for an automatic buffer around the region.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to be subsetted.
    method : {'gridpoint', 'bbox', shape', 'sel'}
        If the method is 'sel', this is not a call to clisops but only a subsetting with the xarray `.sel()` function.
    name : str, optional
        Used to rename the 'cat:domain' attribute.
    tile_buffer : float
        For ['bbox', shape'], uses an approximation of the grid cell size to add a buffer around the requested region.
        This differs from clisops' 'buffer' argument in subset_shape().
    \*\*kwargs : dict
        Arguments to be sent to clisops. See relevant function for details. Depending on the method, required kwargs are:
        - gridpoint: lon, lat
        - bbox: lon_bnds, lat_bnds
        - shape: shape
        - sel: slices for each dimension

    Returns
    -------
    xr.Dataset
        Subsetted Dataset.

    See Also
    --------
    clisops.core.subset.subset_gridpoint, clisops.core.subset.subset_bbox, clisops.core.subset.subset_shape
    """
    if cl is None and method in ["gridpoint", "bbox", "shape"]:
        raise ImportError("The clisops package is required for the 'gridpoint', 'bbox' and 'shape' methods.")
    if tile_buffer > 0 and method in ["gridpoint", "sel"]:
        warnings.warn(
            f"tile_buffer is not used for the '{method}' method. Ignoring the argument.",
            UserWarning,
            stacklevel=2,
        )

    if "latitude" not in ds.cf or "longitude" not in ds.cf:
        ds = ds.cf.guess_coord_axis()

    if method == "gridpoint":
        ds_subset = _subset_gridpoint(ds, name=name, **kwargs)
    elif method == "bbox":
        ds_subset = _subset_bbox(ds, name=name, tile_buffer=tile_buffer, **kwargs)
    elif method == "shape":
        ds_subset = _subset_shape(ds, name=name, tile_buffer=tile_buffer, **kwargs)
    elif method == "sel":
        ds_subset = _subset_sel(ds, name=name, **kwargs)
    else:
        raise ValueError("Subsetting type not recognized. Use 'gridpoint', 'bbox', 'shape' or 'sel'.")

    return ds_subset


def _subset_gridpoint(
    ds: xr.Dataset,
    lon: float | Sequence[float] | xr.DataArray,
    lat: float | Sequence[float] | xr.DataArray,
    *,
    name: str | None = None,
    **kwargs,
) -> xr.Dataset:
    r"""
    Subset the data to a gridpoint.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to be subsetted.
    lon : float or Sequence[float] or xr.DataArray
        Longitude coordinate(s). Must be of the same length as lat.
    lat : float or Sequence[float] or xr.DataArray
        Latitude coordinate(s). Must be of the same length as lon.
    name: str, optional
        Used to rename the 'cat:domain' attribute.
    \*\*kwargs : dict
        Other arguments to be sent to clisops. Possible kwargs are:
        - start_date (str): Start date for the subset in the format 'YYYY-MM-DD'.
        - end_date (str): End date for the subset in the format 'YYYY-MM-DD'.
        - first_level (int or float): First level of the subset.
        - last_level (int or float): Last level of the subset.
        - tolerance (float): Masks values if the distance to the nearest gridpoint is larger than tolerance in meters.
        - add_distance (bool): If True, adds a variable with the distance to the nearest gridpoint.

    Returns
    -------
    xr.Dataset
        Subsetted Dataset.
    """
    ds = _load_lon_lat(ds)
    if not hasattr(lon, "__iter__"):
        lon = [lon]
    if not hasattr(lat, "__iter__"):
        lat = [lat]

    ds_subset = cl.subset_gridpoint(ds, lon=lon, lat=lat, **kwargs)
    new_history = (
        f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        f"gridpoint spatial subsetting on {len(lon)} coordinates - clisops v{clisops.__version__}"
    )

    return update_history_and_name(ds_subset, new_history, name)


def _subset_bbox(
    ds: xr.Dataset,
    lon_bnds: tuple[float, float] | list[float],
    lat_bnds: tuple[float, float] | list[float],
    *,
    name: str | None = None,
    tile_buffer: float = 0,
    **kwargs,
) -> xr.Dataset:
    r"""
    Subset the data to a bounding box.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to be subsetted.
    lon_bnds : tuple or list of two floats
        Longitude boundaries of the bounding box.
    lat_bnds : tuple or list of two floats
        Latitude boundaries of the bounding box.
    name: str, optional
        Used to rename the 'cat:domain' attribute.
    tile_buffer: float
        Uses an approximation of the grid cell size to add a dynamic buffer around the requested region.
    \*\*kwargs : dict
        Other arguments to be sent to clisops. Possible kwargs are:
        - start_date (str): Start date for the subset in the format 'YYYY-MM-DD'.
        - end_date (str): End date for the subset in the format 'YYYY-MM-DD'.
        - first_level (int or float): First level of the subset.
        - last_level (int or float): Last level of the subset.
        - time_values (Sequence[str]): A list of datetime strings to subset.
        - level_values (Sequence[int or float]): A list of levels to subset.

    Returns
    -------
    xr.Dataset
        Subsetted Dataset.
    """
    ds = _load_lon_lat(ds)

    if tile_buffer > 0:
        lon_res, lat_res = _estimate_grid_resolution(ds)
        lon_bnds = (
            lon_bnds[0] - lon_res * tile_buffer,
            lon_bnds[1] + lon_res * tile_buffer,
        )
        lat_bnds = (
            lat_bnds[0] - lat_res * tile_buffer,
            lat_bnds[1] + lat_res * tile_buffer,
        )

    ds_subset = cl.subset_bbox(ds, lon_bnds=lon_bnds, lat_bnds=lat_bnds, **kwargs)
    new_history = (
        f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        f"bbox spatial subsetting with {'buffer=' + str(tile_buffer) if tile_buffer > 0 else 'no buffer'}"
        f", lon_bnds={np.array(lon_bnds)}, lat_bnds={np.array(lat_bnds)}"
        f" - clisops v{clisops.__version__}"
    )

    return update_history_and_name(ds_subset, new_history, name)


def _subset_shape(
    ds: xr.Dataset,
    shape: str | Path | gpd.GeoDataFrame,
    *,
    name: str | None = None,
    tile_buffer: float = 0,
    **kwargs,
) -> xr.Dataset:
    r"""
    Subset the data to a shape.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to be subsetted.
    shape : str or gpd.GeoDataFrame
        Path to the shapefile or GeoDataFrame.
    name: str, optional
        Used to rename the 'cat:domain' attribute.
    tile_buffer: float
        Uses an approximation of the grid cell size to add a buffer around the requested region.
    \*\*kwargs : dict
        Other arguments to be sent to clisops. Possible kwargs are:
        - raster_crs (str or int): EPSG number or PROJ4 string.
        - shape_crs (str or int): EPSG number or PROJ4 string.
        - buffer (float): Buffer size to add around the shape. Units are based on the shape degrees/metres.
        - start_date (str): Start date for the subset in the format 'YYYY-MM-DD'.
        - end_date (str): End date for the subset in the format 'YYYY-MM-DD'.
        - first_level (int or float): First level of the subset.
        - last_level (int or float): Last level of the subset.

    Returns
    -------
    xr.Dataset
        Subsetted Dataset.
    """
    ds = _load_lon_lat(ds)

    if tile_buffer > 0:
        if kwargs.get("buffer") is not None:
            raise ValueError("Both tile_buffer and clisops' buffer were requested. Use only one.")
        lon_res, lat_res = _estimate_grid_resolution(ds)

        # The buffer argument needs to be in the same units as the shapefile, so it's simpler to always project the shapefile to WGS84.
        if isinstance(shape, str | Path):
            shape = gpd.read_file(shape)

        try:
            shape_crs = shape.crs
        except AttributeError:
            shape_crs = None
        if shape_crs is None:
            warnings.warn(
                "Shapefile does not have a CRS. Compatibility with the dataset is not guaranteed.",
                stacklevel=2,
                category=UserWarning,
            )
        elif shape_crs != CRS(4326):  # WGS84
            warnings.warn(
                "Shapefile is not in EPSG:4326. Reprojecting to this CRS.",
                UserWarning,
                stacklevel=2,
            )
            shape = shape.to_crs(4326)

        kwargs["buffer"] = np.max([lon_res, lat_res]) * tile_buffer

    ds_subset = cl.subset_shape(ds, shape=shape, **kwargs)
    new_history = (
        f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        f"shape spatial subsetting with {'buffer=' + str(tile_buffer) if tile_buffer > 0 else 'no buffer'}"
        f", shape={Path(shape).name if isinstance(shape, str | Path) else 'gpd.GeoDataFrame'}"
        f" - clisops v{clisops.__version__}"
    )

    return update_history_and_name(ds_subset, new_history, name)


def _subset_sel(ds: xr.Dataset, *, name: str | None = None, **kwargs) -> xr.Dataset:
    r"""
    Subset the data using the .sel() method.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to be subsetted.
    name: str, optional
        Used to rename the 'cat:domain' attribute.
    \*\*kwargs : dict
        The keys are the dimensions to subset and the values are turned into a slice.

    Returns
    -------
    xr.Dataset
        Subsetted Dataset.
    """
    # Create a dictionary with slices for each dimension
    arg_sel = {dim: slice(*map(float, bounds)) for dim, bounds in kwargs.items()}

    # Subset the dataset
    ds_subset = ds.sel(**arg_sel)

    # Update the history attribute
    new_history = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] sel subsetting with arguments {arg_sel}"

    return update_history_and_name(ds_subset, new_history, name)


def _load_lon_lat(ds: xr.Dataset) -> xr.Dataset:
    """Load longitude and latitude for more efficient subsetting."""
    if xc.core.utils.uses_dask(ds.cf["longitude"]):
        logger.info("Loading longitude for more efficient subsetting.")
        (ds[ds.cf["longitude"].name],) = dask.compute(ds[ds.cf["longitude"].name])
    if xc.core.utils.uses_dask(ds.cf["latitude"]):
        logger.info("Loading latitude for more efficient subsetting.")
        (ds[ds.cf["latitude"].name],) = dask.compute(ds[ds.cf["latitude"].name])

    return ds


def get_grid_mapping(ds: xr.Dataset) -> str:
    """Get the grid_mapping attribute from the dataset."""
    gridmap = [ds[v].attrs["grid_mapping"] for v in ds.data_vars if "grid_mapping" in ds[v].attrs]
    gridmap += [c for c in ds.coords if ds[c].attrs.get("grid_mapping_name", None)]
    gridmap = list(np.unique(gridmap))

    if len(gridmap) > 1:
        warnings.warn(f"There are conflicting grid_mapping attributes in the dataset. Assuming {gridmap[0]}.", stacklevel=2)

    return gridmap[0] if gridmap else ""


def get_crs(gridmap: xr.Dataset | xr.DataArray) -> cartopy.crs.Projection:
    """
    Get the cartopy CRS.

    Parameters
    ----------
    gridmap: xr.Dataset or xr.DataArray
      Either a dataset that has a grid mapping variable or that grid mapping variable directly.

    Returns
    -------
    cartopy.crs.Projection
      The cartopy crs. Only RotatedPole and ObliqueMercator are supported.
    """
    if isinstance(gridmap, xr.Dataset):
        gridmap = gridmap[get_grid_mapping(gridmap)]
    cf_params = gridmap.attrs
    globe = cartopy.crs.Globe(
        datum=cf_params.get("horizontal_datum_name"),
        ellipse=cf_params.get("reference_ellipsoid_name", "WGS84"),
        semimajor_axis=(cf_params.get("earth_radius") or cf_params.get("semi_major_axis")),
        semiminor_axis=cf_params.get("semi_minor_axis"),
        inverse_flattening=cf_params.get("inverse_flattening"),
        towgs84=cf_params.get("towgs84"),
    )
    if cf_params["grid_mapping_name"] == "rotated_latitude_longitude":
        crs = cartopy.crs.RotatedPole(
            pole_longitude=cf_params["grid_north_pole_longitude"],
            pole_latitude=cf_params["grid_north_pole_latitude"],
            central_rotated_longitude=cf_params.get("north_pole_grid_longitude", 0),
            globe=globe,
        )
    elif cf_params["grid_mapping_name"] == "oblique_mercator":
        crs = cartopy.crs.ObliqueMercator(
            central_longitude=cf_params["longitude_of_projection_origin"],
            central_latitude=cf_params["latitude_of_projection_origin"],
            scale_factor=cf_params["scale_factor_at_projection_origin"],
            azimuth=cf_params["azimuth_of_central_line"],
            false_easting=cf_params.get("false_easting", 0),
            false_northing=cf_params.get("false_northing", 0),
            globe=globe,
        )
    else:
        raise NotImplementedError(f"Grid mapping {cf_params['grid_mapping_name']} not implemented.")
    return crs


def _estimate_grid_resolution(ds: xr.Dataset) -> tuple[float, float]:
    # Since this is to compute a buffer, we take the maximum difference as an approximation.
    # Estimate the grid resolution
    if len(ds.lon.dims) == 1:  # 1D lat-lon
        lon_res = np.abs(ds.lon.diff("lon").max().values)
        lat_res = np.abs(ds.lat.diff("lat").max().values)
    else:
        lon_res = np.abs(ds.lon.diff(ds.cf["X"].name).max().values)
        lat_res = np.abs(ds.lat.diff(ds.cf["Y"].name).max().values)

    return lon_res, lat_res


def update_history_and_name(ds_subset, new_history, name):
    history = new_history + " \n " + ds_subset.attrs["history"] if "history" in ds_subset.attrs else new_history
    ds_subset.attrs["history"] = history
    if name is not None:
        ds_subset.attrs["cat:domain"] = name
    return ds_subset
