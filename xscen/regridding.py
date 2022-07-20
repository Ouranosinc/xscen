import datetime
import operator
import os
from copy import deepcopy
from pathlib import PosixPath
from typing import Optional, Union

import numpy as np
import xarray
import xarray as xr
import xesmf as xe

from .config import parse_config

# TODO: Implement logging, warnings, etc.
# TODO: Change all paths to PosixPath objects, including in the catalog?
# TODO: Add an option to call xesmf.util.grid_2d or xesmf.util.grid_global
# TODO: Implement support for an OBS2SIM kind of interpolation


@parse_config
def regrid(
    ds: xarray.Dataset,
    weights_location: Union[str, PosixPath],
    ds_grid: xr.Dataset,
    *,
    regridder_kwargs: Optional[dict] = None,
    to_level: str = "regridded",
) -> xarray.Dataset:
    """
    Based on an intake_esm catalog, this function regrids Zarr files.

    Parameters
    ----------
    ds : xarray.Dataset
      Dataset to regrid. The Dataset needs to have lat/lon coordinates.
      Supports a 'mask' variable compatible with ESMF standards.
    weights_location : Union[str, PosixPath]
      Path to the folder where weight file is saved.
    ds_grid : xr.Dataset
      Destination grid. The Dataset needs to have lat/lon coordinates.
      Supports a 'mask' variable compatible with ESMF standards.
    regridder_kwargs : dict
      Arguments to send xe.Regridder().
    to_level: str
      The processing level to assign to the output.
      Defaults to 'regridded'

    Returns
    -------
    out: xarray.Dataset
      Regridded dataset

    """
    regridder_kwargs = regridder_kwargs or {}

    # Whether or not regridding is required
    if ds["lon"].equals(ds_grid["lon"]) & ds["lat"].equals(ds_grid["lat"]):
        out = ds
        if "mask" in out:
            out = out.where(out.mask == 1)
            out = out.drop_vars(["mask"])

    else:
        kwargs = deepcopy(regridder_kwargs)
        # if weights_location does no exist, create it
        if not os.path.exists(weights_location):
            os.makedirs(weights_location)
        # give unique name to weights file
        weights_filename = os.path.join(
            weights_location,
            f"{ds.attrs['cat/id']}_"
            f"{'_'.join(kwargs[k] for k in kwargs if isinstance(kwargs[k], str))}.nc",
        )

        # TODO: Support for conservative regridding (use xESMF to add corner information), Locstreams, etc.

        # Re-use existing weight file if possible
        if os.path.isfile(weights_filename) and not (
            ("reuse_weights" in kwargs) and (kwargs["reuse_weights"] is False)
        ):
            kwargs["weights"] = weights_filename
            kwargs["reuse_weights"] = True
        regridder = _regridder(
            ds_in=ds, ds_grid=ds_grid, filename=weights_filename, **regridder_kwargs
        )

        # The regridder (when fed Datasets) doesn't like if 'mask' is present.
        if "mask" in ds:
            ds = ds.drop_vars(["mask"])
        out = regridder(ds, keep_attrs=True)

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
                ]
            )
            if len(gridmap) != 1:
                raise ValueError("Could not determine grid_mapping information.")
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
        new_history = (
            f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"regridded with arguments {kwargs_for_hist} - xESMF v{xe.__version__}"
        )
        history = (
            new_history + " \n " + out.attrs["history"]
            if "history" in out.attrs
            else new_history
        )
        out.attrs["history"] = history

    # Attrs
    out.attrs["cat/processing_level"] = to_level
    out.attrs["cat/domain"] = (
        ds_grid.attrs["cat/domain"] if "cat/domain" in ds_grid.attrs else None
    )

    return out


@parse_config
def create_mask(ds: Union[xr.Dataset, xr.DataArray], mask_args: dict) -> xr.DataArray:
    """
    Creates a 0-1 mask based on incoming arguments.

    Parameters
    ----------
    ds : [xr.Dataset, xr.DataArray]
      Dataset or DataArray to be evaluated
    mask_args : dict
      Instructions to build the mask (required fields listed in the Notes).

    Note
    ----------
    'mask' fields:
        variable: str, optional
            Variable on which to base the mask, if ds_mask is not a DataArray.
        where_operator: str, optional
            Conditional operator such as '>'
        where_threshold: str, optional
            Value threshold to be used in conjunction with where_operator.
        mask_nans: bool
            Whether or not to apply a mask on NaNs.

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
        mask = xr.where(np.isreal(mask), mask, 0)

    # Attributes
    if "where_operator" in mask_args:
        mask.attrs[
            "where_threshold"
        ] = f"{mask_args['variable']} {mask_args['where_operator']} {mask_args['where_threshold']}"
    mask.attrs["mask_nans"] = f"{mask_args['mask_nans']}"

    return mask


def _regridder(
    ds_in: xr.Dataset,
    ds_grid: xr.Dataset,
    filename: str,
    *,
    method: Optional[str] = "bilinear",
    unmapped_to_nan: Optional[bool] = True,
    **kwargs,
) -> xe.frontend.Regridder:
    """
    Simple call to xe.Regridder with a few default arguments

    Parameters
    ----------
    ds_in : xr.Dataset
      Incoming grid. The Dataset needs to have lat/lon coordinates.
    ds_grid : xr.Dataset
      Destination grid. The Dataset needs to have lat/lon coordinates.
    filename : str
      Path to the NetCDF file with weights information.
    method, unmapped_to_nan
      Arguments to send xe.Regridder().
    regridder_kwargs : dict
      Arguments to send xe.Regridder().

    Returns
    -------
    xe.frontend.Regridde
      Regridder object

    """

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
