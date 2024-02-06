"""Functions to regrid datasets."""

import datetime
import operator
import os
import warnings
from copy import deepcopy
from typing import Optional, Union

import cartopy.crs as ccrs
import cf_xarray as cfxr
import numpy as np
import xarray as xr

try:
    import xesmf as xe
    from xesmf.frontend import Regridder
except ImportError:
    xe = None
    Regridder = "xesmf.Regridder"

from .config import parse_config

# TODO: Implement logging, warnings, etc.
# TODO: Add an option to call xesmf.util.grid_2d or xesmf.util.grid_global
# TODO: Implement support for an OBS2SIM kind of interpolation


__all__ = ["create_mask", "regrid_dataset"]


@parse_config
def regrid_dataset(  # noqa: C901
    ds: xr.Dataset,
    ds_grid: xr.Dataset,
    weights_location: Union[str, os.PathLike],
    *,
    regridder_kwargs: Optional[dict] = None,
    intermediate_grids: Optional[dict] = None,
    to_level: str = "regridded",
) -> xr.Dataset:
    """Regrid a dataset according to weights and a reference grid.

    Based on an intake_esm catalog, this function performs regridding on Zarr files.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to regrid. The Dataset needs to have lat/lon coordinates.
        Supports a 'mask' variable compatible with ESMF standards.
    weights_location : Union[str, os.PathLike]
        Path to the folder where weight file is saved.
    ds_grid : xr.Dataset
        Destination grid. The Dataset needs to have lat/lon coordinates.
        Supports a 'mask' variable compatible with ESMF standards.
    regridder_kwargs : dict, optional
        Arguments to send xe.Regridder(). If it contains `skipna` or `output_chunks`, those
        are passed to the regridder call directly.
    intermediate_grids : dict, optional
        This argument is used to do a regridding in many steps, regridding to regular
        grids before regridding to the final ds_grid.
        This is useful when there is a large jump in resolution between ds and ds grid.
        The format is a nested dictionary shown in Notes.
        If None, no intermediary grid is used, there is only a regrid from ds to ds_grid.
    to_level : str
        The processing level to assign to the output.
        Defaults to 'regridded'

    Returns
    -------
    xarray.Dataset
        Regridded dataset

    Notes
    -----
    intermediate_grids =
      {'name_of_inter_grid_1': {'cf_grid_2d': {arguments for util.cf_grid_2d },'regridder_kwargs':{arguments for xe.Regridder}},
        'name_of_inter_grid_2': dictionary_as_above}

    See Also
    --------
    xesmf.regridder, xesmf.util.cf_grid_2d
    """
    if xe is None:
        raise ImportError(
            "xscen's regridding functionality requires xESMF to work, please install that package."
        )

    regridder_kwargs = regridder_kwargs or {}

    ds_grids = []  # list of target grids
    reg_arguments = []  # list of accompanying arguments for xe.Regridder()
    if intermediate_grids:
        for name_inter, dict_inter in intermediate_grids.items():
            reg_arguments.append(dict_inter["regridder_kwargs"])
            ds_grids.append(xe.util.cf_grid_2d(**dict_inter["cf_grid_2d"]))

    ds_grids.append(ds_grid)  # add final ds_grid
    reg_arguments.append(regridder_kwargs)  # add final regridder_kwargs

    out = None

    # Whether regridding is required
    if ds["lon"].equals(ds_grid["lon"]) & ds["lat"].equals(ds_grid["lat"]):
        out = ds
        if "mask" in out:
            out = out.where(out.mask == 1)
            out = out.drop_vars(["mask"])

    else:
        for i, (ds_grid, regridder_kwargs) in enumerate(zip(ds_grids, reg_arguments)):
            # if this is not the first iteration (out != None),
            # get result from last iteration (out) as input
            ds = out or ds

            kwargs = deepcopy(regridder_kwargs)
            # if weights_location does no exist, create it
            if not os.path.exists(weights_location):
                os.makedirs(weights_location)
            id = ds.attrs["cat:id"] if "cat:id" in ds.attrs else "weights"
            # give unique name to weights file
            weights_filename = os.path.join(
                weights_location,
                f"{id}_regrid{i}"
                f"{'_'.join(kwargs[k] for k in kwargs if isinstance(kwargs[k], str))}.nc",
            )

            # Re-use existing weight file if possible
            if os.path.isfile(weights_filename) and not (
                ("reuse_weights" in kwargs) and (kwargs["reuse_weights"] is False)
            ):
                kwargs["weights"] = weights_filename
                kwargs["reuse_weights"] = True

            # Extract args that are to be given at call time.
            # output_chunks is only valid for xesmf >= 0.8, so don't add it be default to the call_kwargs
            call_kwargs = {"skipna": regridder_kwargs.pop("skipna", False)}
            if "output_chunks" in regridder_kwargs:
                call_kwargs["output_chunks"] = regridder_kwargs.pop("output_chunks")

            regridder = _regridder(
                ds_in=ds, ds_grid=ds_grid, filename=weights_filename, **regridder_kwargs
            )

            # The regridder (when fed Datasets) doesn't like if 'mask' is present.
            if "mask" in ds:
                ds = ds.drop_vars(["mask"])

            out = regridder(ds, keep_attrs=True, **call_kwargs)

            # double-check that grid_mapping information is transferred
            gridmap_out = any(
                "grid_mapping" in ds_grid[da].attrs for da in ds_grid.data_vars
            )
            if gridmap_out:
                gridmap = np.unique(
                    [
                        ds_grid[da].attrs["grid_mapping"]
                        for da in ds_grid.data_vars
                        if "grid_mapping" in ds_grid[da].attrs
                        and ds_grid[da].attrs["grid_mapping"] in ds_grid
                    ]
                )
                if len(gridmap) != 1:
                    warnings.warn(
                        "Could not determine and transfer grid_mapping information."
                    )
                else:
                    # Add the grid_mapping attribute
                    for v in out.data_vars:
                        out[v].attrs["grid_mapping"] = gridmap[0]
                    # Add the grid_mapping coordinate
                    if gridmap[0] not in out:
                        out = out.assign_coords({gridmap[0]: ds_grid[gridmap[0]]})
                    # Regridder seems to seriously mess up the rotated dimensions
                    for d in out.lon.dims:
                        out[d] = ds_grid[d]
                        if d not in out.coords:
                            out = out.assign_coords({d: ds_grid[d]})
            else:
                gridmap = np.unique(
                    [
                        ds[da].attrs["grid_mapping"]
                        for da in ds.data_vars
                        if "grid_mapping" in ds[da].attrs
                    ]
                )
                # Remove the original grid_mapping attribute
                for v in out.data_vars:
                    if "grid_mapping" in out[v].attrs:
                        out[v].attrs.pop("grid_mapping")
                # Remove the original grid_mapping coordinate if it is still in the output
                out = out.drop_vars(set(gridmap).intersection(out.variables))

            # History
            kwargs_for_hist = deepcopy(regridder_kwargs)
            kwargs_for_hist.setdefault("method", regridder.method)
            if intermediate_grids and i < len(intermediate_grids):
                name_inter = list(intermediate_grids.keys())[i]
                cf_grid_2d_args = intermediate_grids[name_inter]["cf_grid_2d"]
                new_history = (
                    f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"regridded with regridder arguments {kwargs_for_hist} to a xesmf"
                    f" cf_grid_2d with arguments {cf_grid_2d_args}  - xESMF v{xe.__version__}"
                )
            else:
                new_history = (
                    f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"regridded with arguments {kwargs_for_hist} - xESMF v{xe.__version__}"
                )
            history = (
                f"{new_history}\n{out.attrs['history']}"
                if "history" in out.attrs
                else new_history
            )
            out.attrs["history"] = history

    out = out.drop_vars("latitude_longitude", errors="ignore")
    # Attrs
    out.attrs["cat:processing_level"] = to_level
    out.attrs["cat:domain"] = (
        ds_grid.attrs["cat:domain"] if "cat:domain" in ds_grid.attrs else None
    )
    return out


@parse_config
def create_mask(ds: Union[xr.Dataset, xr.DataArray], mask_args: dict) -> xr.DataArray:
    """Create a 0-1 mask based on incoming arguments.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Dataset or DataArray to be evaluated
    mask_args : dict
        Instructions to build the mask (required fields listed in the Notes).

    Note
    ----
    'mask' fields:
        variable: str, optional
            Variable on which to base the mask, if ds_mask is not a DataArray.
        where_operator: str, optional
            Conditional operator such as '>'
        where_threshold: str, optional
            Value threshold to be used in conjunction with where_operator.
        mask_nans: bool
            Whether to apply a mask on NaNs.

    Returns
    -------
    xr.DataArray
        Mask array.
    """
    # Prepare the mask for the destination grid
    ops = {
        "<": operator.lt,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
        ">=": operator.ge,
        ">": operator.gt,
    }

    def cmp(arg1, op, arg2):
        operation = ops.get(op)
        return operation(arg1, arg2)

    mask_args = mask_args or {}
    if isinstance(ds, xr.DataArray):
        mask = ds
    elif isinstance(ds, xr.Dataset) and "variable" in mask_args:
        mask = ds[mask_args["variable"]]
    else:
        raise ValueError("Could not determine what to base the mask on.")

    if "time" in mask.dims:
        mask = mask.isel(time=0)

    if "where_operator" in mask_args:
        mask = xr.where(
            cmp(mask, mask_args["where_operator"], mask_args["where_threshold"]), 1, 0
        )
    else:
        mask = xr.ones_like(mask)
    if ("mask_nans" in mask_args) & (mask_args["mask_nans"] is True):
        mask = mask.where(np.isreal(mask), other=0)

    # Attributes
    if "where_operator" in mask_args:
        mask.attrs["where_threshold"] = (
            f"{mask_args['variable']} {mask_args['where_operator']} {mask_args['where_threshold']}"
        )
    mask.attrs["mask_nans"] = f"{mask_args['mask_nans']}"

    return mask


def _regridder(
    ds_in: xr.Dataset,
    ds_grid: xr.Dataset,
    filename: Union[str, os.PathLike],
    *,
    method: str = "bilinear",
    unmapped_to_nan: Optional[bool] = True,
    **kwargs,
) -> Regridder:
    """Call to xesmf Regridder with a few default arguments.

    Parameters
    ----------
    ds_in : xr.Dataset
        Incoming grid. The Dataset needs to have lat/lon coordinates.
    ds_grid : xr.Dataset
        Destination grid. The Dataset needs to have lat/lon coordinates.
    filename : str or os.PathLike
        Path to the NetCDF file with weights information.
    method : str
        Interpolation method.
    unmapped_to_nan : bool, optional
        Arguments to send xe.Regridder().
    regridder_kwargs : dict
        Arguments to send xe.Regridder().

    Returns
    -------
    xe.frontend.Regridder
        Regridder object
    """
    if method.startswith("conservative"):
        if (
            ds_in.cf["longitude"].ndim == 2
            and "longitude" not in ds_in.cf.bounds
            and "rotated_pole" in ds_in
        ):
            ds_in = ds_in.update(create_bounds_rotated_pole(ds_in))
        if (
            ds_grid.cf["longitude"].ndim == 2
            and "longitude" not in ds_grid.cf.bounds
            and "rotated_pole" in ds_grid
        ):
            ds_grid = ds_grid.update(create_bounds_rotated_pole(ds_grid))

    regridder = xe.Regridder(
        ds_in=ds_in,
        ds_out=ds_grid,
        method=method,
        unmapped_to_nan=unmapped_to_nan,
        **kwargs,
    )
    if ~os.path.isfile(filename):
        regridder.to_netcdf(filename)

    return regridder


def create_bounds_rotated_pole(ds: xr.Dataset):
    """Create bounds for rotated pole datasets."""
    ds = ds.cf.add_bounds(["rlat", "rlon"])

    # In "vertices" format then expand to 2D. From (N, 2) to (N+1,) to (N+1, M+1)
    rlatv1D = cfxr.bounds_to_vertices(ds.rlat_bounds, "bounds")
    rlonv1D = cfxr.bounds_to_vertices(ds.rlon_bounds, "bounds")
    rlatv = rlatv1D.expand_dims(rlon_vertices=rlonv1D).transpose(
        "rlon_vertices", "rlat_vertices"
    )
    rlonv = rlonv1D.expand_dims(rlat_vertices=rlatv1D).transpose(
        "rlon_vertices", "rlat_vertices"
    )

    # Get cartopy's crs for the projection
    RP = ccrs.RotatedPole(
        pole_longitude=ds.rotated_pole.grid_north_pole_longitude,
        pole_latitude=ds.rotated_pole.grid_north_pole_latitude,
        central_rotated_longitude=ds.rotated_pole.north_pole_grid_longitude,
    )
    PC = ccrs.PlateCarree()

    # Project points
    pts = PC.transform_points(RP, rlonv.values, rlatv.values)
    lonv = rlonv.copy(data=pts[..., 0]).rename("lon_vertices")
    latv = rlatv.copy(data=pts[..., 1]).rename("lat_vertices")

    # Back to CF bounds format. From (N+1, M+1) to (4, N, M)
    lonb = cfxr.vertices_to_bounds(lonv, ("bounds", "rlon", "rlat")).rename(
        "lon_bounds"
    )
    latb = cfxr.vertices_to_bounds(latv, ("bounds", "rlon", "rlat")).rename(
        "lat_bounds"
    )

    # Create dataset, set coords and attrs
    ds_bnds = xr.merge([lonb, latb]).assign(
        lon=ds.lon, lat=ds.lat, rotated_pole=ds.rotated_pole
    )
    ds_bnds["rlat"] = ds.rlat
    ds_bnds["rlon"] = ds.rlon
    ds_bnds.lat.attrs["bounds"] = "lat_bounds"
    ds_bnds.lon.attrs["bounds"] = "lon_bounds"
    return ds_bnds.transpose(*ds.lon.dims, "bounds")
