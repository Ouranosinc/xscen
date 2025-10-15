"""Functions to regrid datasets."""

import datetime
import operator
import os
import random
import string
import warnings
from copy import deepcopy
from pathlib import Path

import cartopy.crs as ccrs
import cf_xarray as cfxr
import xarray as xr
from xclim.core.units import convert_units_to


try:
    import xesmf as xe
    from xesmf.frontend import Regridder
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
    Regridder = "xesmf.Regridder"

from .config import parse_config
from .spatial import get_crs, get_grid_mapping


__all__ = ["create_bounds_gridmapping", "create_mask", "regrid_dataset"]


@parse_config
def regrid_dataset(  # noqa: C901
    ds: xr.Dataset,
    ds_grid: xr.Dataset,
    *,
    weights_location: str | os.PathLike | None = None,
    regridder_kwargs: dict | None = None,
    intermediate_grids: dict | None = None,
    to_level: str = "regridded",
) -> xr.Dataset:
    """
    Regrid a dataset according to weights and a reference grid.

    Based on an intake_esm catalog, this function performs regridding on Zarr files.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to regrid. The Dataset needs to have lat/lon coordinates.
        Supports a 'mask' variable compatible with ESMF standards.
    ds_grid : xr.Dataset
        Destination grid. The Dataset needs to have lat/lon coordinates.
        Supports a 'mask' variable compatible with ESMF standards.
    weights_location : Union[str, os.PathLike], optional
        Path to the folder where weight file is saved. Leave as None to force re-computation of weights.
        Note that in order to reuse the weights, ds and ds_grid should both have the 'cat:id' and 'cat:domain' attributes.
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
        raise ImportError("xscen's regridding functionality requires xESMF to work, please install that package.")
    ds = ds.copy()
    regridder_kwargs = regridder_kwargs or {}

    # We modify the dataset later, so we need to keep track of whether it had lon_bounds and lat_bounds to begin with
    has_lon_bounds = "lon_bounds" in ds
    has_lat_bounds = "lat_bounds" in ds

    # Generate unique IDs to name the weights file, but remove the members and experiment from the dataset ID
    if weights_location is not None:
        dsid = (
            ds.attrs.get("cat:id", _generate_random_string(15))
            .replace(ds.attrs.get("cat:member", ""), "")
            .replace(ds.attrs.get("cat:driving_member", ""), "")
            .replace(ds.attrs.get("cat:experiment", ""), "")
        )
        dsid = f"{dsid}_{ds.attrs.get('cat:domain', _generate_random_string(15))}"
        gridid = f"{ds_grid.attrs.get('cat:id', _generate_random_string(15))}_{ds_grid.attrs.get('cat:domain', _generate_random_string(15))}"

    ds_grids = []  # List of target grids
    reg_arguments = []  # List of accompanying arguments for xe.Regridder()
    if intermediate_grids:
        for dict_inter in intermediate_grids.values():
            reg_arguments.append(dict_inter["regridder_kwargs"])
            ds_grids.append(xe.util.cf_grid_2d(**dict_inter["cf_grid_2d"]))

    ds_grids.append(ds_grid)  # Add the final ds_grid
    reg_arguments.append(regridder_kwargs)  # Add the final regridder_kwargs

    out = None

    # If the grid is the same, skip the call to xESMF
    if ds["lon"].equals(ds_grid["lon"]) & ds["lat"].equals(ds_grid["lat"]):
        out = ds
        if "mask" in out:
            out = out.where(out.mask == 1)
            out = out.drop_vars(["mask"])
        if "mask" in ds_grid:
            out = out.where(ds_grid.mask == 1)

    else:
        for i, (ds_grid, regridder_kwargs) in enumerate(zip(ds_grids, reg_arguments, strict=False)):
            # If this is not the first iteration (out != None),
            # get the result from the last iteration (out) as input
            ds = out or ds
            kwargs = deepcopy(regridder_kwargs)

            # Prepare the weight file
            if weights_location is not None:
                Path(weights_location).mkdir(parents=True, exist_ok=True)
                weights_filename = Path(
                    weights_location,
                    f"{dsid}_{gridid}_regrid{i}{'_'.join(kwargs[k] for k in kwargs if isinstance(kwargs[k], str))}.nc",
                )

                # Reuse existing weight file if possible
                if Path(weights_filename).is_file() and not (("reuse_weights" in kwargs) and (kwargs["reuse_weights"] is False)):
                    kwargs["weights"] = weights_filename
                    kwargs["reuse_weights"] = True
            else:
                weights_filename = None

            # Extract args that are to be given at call time.
            # output_chunks is only valid for xesmf >= 0.8, so don't add it be default to the call_kwargs
            call_kwargs = {"skipna": kwargs.pop("skipna", False)}
            if "output_chunks" in kwargs:
                call_kwargs["output_chunks"] = kwargs.pop("output_chunks")

            regridder = _regridder(ds_in=ds, ds_grid=ds_grid, filename=weights_filename, **kwargs)

            # The regridder (when fed Datasets) doesn't like if 'mask' is present.
            if "mask" in ds:
                ds = ds.drop_vars(["mask"])

            out = regridder(ds, keep_attrs=True, **call_kwargs)

            # Double-check that grid_mapping information is transferred
            gridmap_out = get_grid_mapping(ds_grid)
            if gridmap_out:
                # Regridder seems to seriously mess up the rotated dimensions
                for d in out.lon.dims:
                    out[d] = ds_grid[d]
                    if d not in out.coords:
                        out = out.assign_coords({d: ds_grid[d]})
                # Add the grid_mapping attribute
                for v in out.data_vars:
                    if any(d in out[v].dims for d in [out.cf.axes["X"][0], out.cf.axes["Y"][0]]):
                        out[v].attrs["grid_mapping"] = gridmap_out
                # Add the grid_mapping coordinate
                if gridmap_out not in out:
                    out = out.assign_coords({gridmap_out: ds_grid[gridmap_out]})
            else:
                gridmap_in = get_grid_mapping(ds)
                # Remove the original grid_mapping attribute
                for v in out.data_vars:
                    if "grid_mapping" in out[v].attrs:
                        out[v].attrs.pop("grid_mapping")
                # Remove the original grid_mapping coordinate if it is still in the output
                out = out.drop_vars(gridmap_in, errors="ignore")

            # cf_grid_2d adds temporary variables that we don't want to keep
            if "lon_bounds" in out and has_lon_bounds is False:
                out = out.drop_vars("lon_bounds")
            if "lat_bounds" in out and has_lat_bounds is False:
                out = out.drop_vars("lat_bounds")

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
                    f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] regridded with arguments {kwargs_for_hist} - xESMF v{xe.__version__}"
                )
            history = f"{new_history}\n{out.attrs['history']}" if "history" in out.attrs else new_history
            out.attrs["history"] = history

    out = out.drop_vars("latitude_longitude", errors="ignore")
    # Attrs
    out.attrs["cat:processing_level"] = to_level
    out.attrs["cat:domain"] = ds_grid.attrs["cat:domain"] if "cat:domain" in ds_grid.attrs else None
    return out


@parse_config
def create_mask(
    ds: xr.Dataset | xr.DataArray,
    *,
    variable: str | None = None,
    where_operator: str | None = None,
    where_threshold: float | str | None = None,
    mask_nans: bool = True,
) -> xr.DataArray:
    """
    Create a 0-1 mask based on incoming arguments.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Dataset or DataArray to be evaluated. If a time dimension is present, the first time step will be used.
    variable : str, optional
        If using a Dataset, the variable on which to base the mask.
    where_operator : str, optional
        Operator to use for the threshold comparison. One of "<", "<=", "==", "!=", ">=", ">".
        Needs to be used with `where_threshold`.
    where_threshold : float or str, optional
        Threshold value to use for the comparison. A string can be used to reference units, e.g. "10 mm/day".
        Needs to be used with `where_operator`.
    mask_nans : bool, optional
        Whether to mask NaN values in the mask array. Default is True.

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

    if isinstance(ds, xr.Dataset):
        if variable is None:
            raise ValueError("A variable needs to be specified when passing a Dataset.")
        ds = ds[variable].copy()
    else:
        ds = ds.copy()
    if "time" in ds.dims:
        ds = ds.isel(time=0)

    mask = xr.ones_like(ds)
    mask.attrs = {"long_name": "Mask"}
    mask.name = "mask"

    # Create the mask based on the threshold
    if (where_operator is not None and where_threshold is None) or (where_operator is None and where_threshold is not None):
        raise ValueError("'where_operator' and 'where_threshold' must be used together.")
    if where_threshold is not None:
        mask.attrs["where_threshold"] = f"{variable} {where_operator} {where_threshold}"
        if isinstance(where_threshold, str):
            ds = convert_units_to(ds, where_threshold.split(" ")[1])
            where_threshold = float(where_threshold.split(" ")[0])

        mask = xr.where(cmp(ds, where_operator, where_threshold), mask, 0, keep_attrs=True)

    # Mask NaNs
    if mask_nans:
        mask = xr.where(ds.notnull(), mask, 0, keep_attrs=True)
        mask.attrs["mask_NaNs"] = "True"
    else:
        # The where clause above will mask NaNs, so we need to revert that
        attrs = mask.attrs
        mask = xr.where(ds.isnull(), 1, mask)
        mask.attrs = attrs
        mask.attrs["mask_NaNs"] = "False"

    return mask


def _regridder(
    ds_in: xr.Dataset,
    ds_grid: xr.Dataset,
    *,
    filename: str | os.PathLike | None = None,
    method: str = "bilinear",
    unmapped_to_nan: bool | None = True,
    **kwargs,
) -> Regridder:
    """
    Call to xesmf Regridder with a few default arguments.

    Parameters
    ----------
    ds_in : xr.Dataset
        Incoming grid. The Dataset needs to have lat/lon coordinates.
    ds_grid : xr.Dataset
        Destination grid. The Dataset needs to have lat/lon coordinates.
    filename : str or os.PathLike, optional
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
        gridmap_in = get_grid_mapping(ds_in)
        gridmap_grid = get_grid_mapping(ds_grid)

        if ds_in.cf["longitude"].ndim == 2 and "longitude" not in ds_in.cf.bounds and gridmap_in in ds_in:
            ds_in = ds_in.assign_coords(**create_bounds_gridmapping(ds_in, gridmap_in))
        if ds_grid.cf["longitude"].ndim == 2 and "longitude" not in ds_grid.cf.bounds and gridmap_grid in ds_grid:
            ds_grid = ds_grid.assign_coords(**create_bounds_gridmapping(ds_grid, gridmap_grid))

    regridder = xe.Regridder(
        ds_in=ds_in,
        ds_out=ds_grid,
        method=method,
        unmapped_to_nan=unmapped_to_nan,
        **kwargs,
    )
    if filename is not None and not Path(filename).is_file():
        regridder.to_netcdf(filename)

    return regridder


def create_bounds_rotated_pole(ds: xr.Dataset) -> xr.Dataset:
    warnings.warn(
        "This function is deprecated and will be removed in xscen v0.12.0. Use create_bounds_gridmapping instead.",
        FutureWarning,
        stacklevel=2,
    )
    return create_bounds_gridmapping(ds, "rotated_pole")


def create_bounds_gridmapping(ds: xr.Dataset, gridmap: str | None = None) -> xr.Dataset:
    """Create bounds for rotated pole datasets."""
    if gridmap is None:
        gridmap = get_grid_mapping(ds)
        if gridmap == "":
            raise ValueError("Grid mapping could not be inferred from the dataset.")
    if gridmap not in ds:
        raise ValueError("Input `gridmap`={gridmap} is not a coordinate of ds.")

    xname = ds.cf.axes["X"][0]
    yname = ds.cf.axes["Y"][0]

    ds = ds.cf.add_bounds([yname, xname])

    # In "vertices" format then expand to 2D. From (N, 2) to (N+1,) to (N+1, M+1)
    yv1D = cfxr.bounds_to_vertices(ds[f"{yname}_bounds"], "bounds")
    xv1D = cfxr.bounds_to_vertices(ds[f"{xname}_bounds"], "bounds")
    yv = yv1D.expand_dims(dict([(f"{xname}_vertices", xv1D)])).transpose(f"{xname}_vertices", f"{yname}_vertices")
    xv = xv1D.expand_dims(dict([(f"{yname}_vertices", yv1D)])).transpose(f"{xname}_vertices", f"{yname}_vertices")

    crs = get_crs(ds[gridmap])
    PC = ccrs.PlateCarree(globe=crs.globe)

    # Project points
    pts = PC.transform_points(crs, xv.values, yv.values)
    lonv = xv.copy(data=pts[..., 0]).rename("lon_vertices")
    latv = yv.copy(data=pts[..., 1]).rename("lat_vertices")

    # Back to CF bounds format. From (N+1, M+1) to (4, N, M)
    lonb = cfxr.vertices_to_bounds(lonv, ("bounds", xname, yname)).rename("lon_bounds")
    latb = cfxr.vertices_to_bounds(latv, ("bounds", xname, yname)).rename("lat_bounds")

    # Create dataset, set coords and attrs
    ds_bnds = xr.merge([lonb, latb]).assign(dict([("lon", ds.lon), ("lat", ds.lat), (gridmap, ds[gridmap])]))
    ds_bnds[yname] = ds[yname]
    ds_bnds[xname] = ds[xname]
    ds_bnds.lat.attrs["bounds"] = "lat_bounds"
    ds_bnds.lon.attrs["bounds"] = "lon_bounds"
    return ds_bnds.transpose(*ds.lon.dims, "bounds")


def _generate_random_string(length: int):
    characters = string.ascii_letters + string.digits

    # Random seed based on the current time
    random.seed(datetime.datetime.now().timestamp())
    random_string = "".join(
        random.choice(characters)  # noqa: S311
        for _ in range(length)
    )
    return random_string
