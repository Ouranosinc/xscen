import logging
import os
import shutil as sh
from pathlib import Path, PosixPath
from types import ModuleType
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr
import zarr
from rechunker import rechunk as _rechunk
from xclim.core.calendar import get_calendar
from xclim.core.indicator import Indicator

from xscen.config import parse_config
from xscen.indicators import load_xclim_module
from xscen.utils import CV, translate_time_chunk

logger = logging.getLogger(__name__)


def save_to_netcdf(
    ds: xr.Dataset,
    filename: str,
    *,
    rechunk: Optional[dict] = None,
    netcdf_kwargs: Optional[dict] = None,
) -> None:
    """Saves a Dataset to NetCDF, rechunking if requested.

    Parameters
    ----------
    ds : xr.Dataset
      Dataset to be saved.
    filename : str
      Name of the NetCDF file to be saved.
    rechunk : dict, optional
      This is a mapping from dimension name to new chunks (in any format understood by dask).
      Rechunking is only done on *data* variables sharing dimensions with this argument.
    netcdf_kwargs : dict, optional
      Additional arguments to send to_netcdf()

    Returns
    -------
    None
    """

    if rechunk:
        for rechunk_var in ds.data_vars:
            # Support for chunks varying per variable
            if rechunk_var in rechunk:
                rechunk_dims = rechunk[rechunk_var]
            else:
                rechunk_dims = rechunk

            ds[rechunk_var] = ds[rechunk_var].chunk(
                {
                    d: chnks
                    for d, chnks in rechunk_dims.items()
                    if d in ds[rechunk_var].dims
                }
            )
            ds[rechunk_var].encoding.pop("chunksizes", None)
            ds[rechunk_var].encoding.pop("chunks", None)

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare to_netcdf kwargs
    netcdf_kwargs = netcdf_kwargs or {}
    netcdf_kwargs.setdefault("engine", "h5netcdf")
    netcdf_kwargs.setdefault("format", "NETCDF4")

    # Ensure no funky objects in attrs:
    def coerce_attrs(attrs):
        for k in attrs.keys():
            if not (
                isinstance(attrs[k], (str, float, int, np.ndarray))
                or isinstance(attrs[k], (tuple, list))
                and isinstance(attrs[k][0], (str, float, int))
            ):
                attrs[k] = str(attrs[k])

    coerce_attrs(ds.attrs)
    for var in ds.variables.values():
        coerce_attrs(var.attrs)

    ds.to_netcdf(filename, **netcdf_kwargs)


def save_to_zarr(
    ds: xr.Dataset,
    filename: str,
    *,
    rechunk: Optional[dict] = None,
    zarr_kwargs: Optional[dict] = None,
    compute: bool = True,
    encoding: dict = None,
    mode: str = "f",
    itervar: bool = False,
) -> None:
    """
    Saves a Dataset to Zarr, rechunking if requested.
    According to mode, removes variables that we don't want to re-compute in ds.

    Parameters
    ----------
    ds : xr.Dataset
      Dataset to be saved.
    filename : str
      Name of the Zarr file to be saved.
    rechunk : dict, optional
      This is a mapping from dimension name to new chunks (in any format understood by dask).
      Rechunking is only done on *data* variables sharing dimensions with this argument.
    zarr_kwargs : dict, optional
      Additional arguments to send to_zarr()
    compute : bool
      Whether to start the computation or return a delayed object.
    mode: {'f', 'o', 'a'}
      If 'f', fails if any variable already exists.
      if 'o', removes the existing variables.
      if 'a', skip existing variables, writes the others.
    encoding : dict, optional
      If given, skipped variables are popped in place.
    itervar : bool
      If True, (data) variables are written one at a time, appending to the zarr.
      If False, this function computes, no matter what was passed to kwargs.

    Returns
    -------
    None
    """

    if rechunk:
        for rechunk_var in ds.data_vars:
            # Support for chunks varying per variable
            if rechunk_var in rechunk:
                rechunk_dims = rechunk[rechunk_var]
            else:
                rechunk_dims = rechunk

            ds[rechunk_var] = ds[rechunk_var].chunk(
                {
                    d: chnks
                    for d, chnks in rechunk_dims.items()
                    if d in ds[rechunk_var].dims
                }
            )
            ds[rechunk_var].encoding.pop("chunksizes", None)
            ds[rechunk_var].encoding.pop("chunks", None)

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_dir():
        tgtds = zarr.open(str(path), mode="r")
    else:
        tgtds = {}

    if encoding:
        encoding = encoding.copy()

    # Prepare to_zarr kwargs
    if zarr_kwargs is None:
        zarr_kwargs = {}

    def _skip(var):
        exists = var in tgtds

        if mode == "f" and exists:
            raise ValueError(f"Variable {var} exists in dataset {path}.")

        if mode == "o":
            if exists:
                var_path = path / var
                print(f"Removing {var_path} to overwrite.")
                sh.rmtree(var_path)
            return False

        if mode == "a":
            if "append_dim" not in zarr_kwargs:
                return exists
            return False

    for var in ds.data_vars.keys():
        if _skip(var):
            logger.info(f"Skipping {var} in {path}.")
            ds = ds.drop_vars(var)
            if encoding:
                encoding.pop(var)

    if len(ds.data_vars) == 0:
        return None

    # Ensure no funky objects in attrs:
    def coerce_attrs(attrs):
        for k in list(attrs.keys()):
            if not (
                isinstance(attrs[k], (str, float, int, np.ndarray))
                or isinstance(attrs[k], (tuple, list))
                and isinstance(attrs[k][0], (str, float, int))
            ):
                attrs[k] = str(attrs[k])

    coerce_attrs(ds.attrs)
    for var in ds.variables.values():
        coerce_attrs(var.attrs)

    if itervar:
        zarr_kwargs["compute"] = True
        allvars = set(ds.data_vars.keys())
        if mode == "f":
            dsbase = ds.drop_vars(allvars)
            dsbase.to_zarr(path, **zarr_kwargs)
        if mode == "o":
            dsbase = ds.drop_vars(allvars)
            dsbase.to_zarr(path, **zarr_kwargs, mode="w")
        for i, (name, var) in enumerate(ds.data_vars.items()):
            logger.debug(f"Writing {name} ({i + 1} of {len(ds.data_vars)}) to {path}")
            dsvar = ds.drop_vars(allvars - {name})
            dsvar.to_zarr(
                path,
                mode="a",
                encoding={k: v for k, v in (encoding or {}).items() if k in dsvar},
                **zarr_kwargs,
            )
    else:
        logger.debug(f"Writing {list(ds.data_vars.keys())} for {filename}.")
        ds.to_zarr(
            filename, compute=compute, mode="a", encoding=encoding, **zarr_kwargs
        )


@parse_config
def rechunk(
    path_in: Union[os.PathLike, str, xr.Dataset],
    path_out: Union[os.PathLike, str],
    *,
    chunks_over_var: Optional[dict] = None,
    chunks_over_dim: Optional[dict] = None,
    worker_mem: str,
    temp_store: Optional[Union[os.PathLike, str]] = None,
    overwrite: bool = False,
) -> None:
    """Rechunk a dataset into a new zarr.

    Parameters
    ----------
    path_in : path, str or xr.Dataset
      Input to rechunk.
    path_out : path or str
      Path to the target zarr.
    chunks_over_var: dict
      Mapping from variables to mappings from dimension name to size. Give this argument or `chunks_over_dim`.
    chunks_over_dim: dict
      Mapping from dimension name to size that will be used for all variables in ds.
      Give this argument or `chunks_over_var`.
    worker_mem : str
      The maximal memory usage of each task.
      When using a distributed Client, this an approximate memory per thread.
      Each worker of the client should have access to 10-20% more memory than this times the number of threads.
    temp_store: path, str, optional
      A path to a zarr where to store intermediate results.
    overwrite: bool
      If True, it will delete whatever is in path_out before doing the rechunking.

    Returns
    -------
    None
    """
    if Path(path_out).is_dir() and overwrite:
        sh.rmtree(path_out)

    if isinstance(path_in, os.PathLike) or isinstance(path_in, str):
        path_in = Path(path_in)
        if path_in.suffix == ".zarr":
            ds = xr.open_zarr(path_in)
        else:
            ds = xr.open_dataset(path_in)
    else:
        ds = path_in
    variables = list(ds.data_vars)
    if chunks_over_var:
        chunks = chunks_over_var
    elif chunks_over_dim:
        chunks = {v: {d: chunks_over_dim[d] for d in ds[v].dims} for v in variables}
        chunks.update(time=None, lat=None, lon=None)
        cal = get_calendar(ds)
        Nt = ds.time.size
        chunks = translate_time_chunk(chunks, cal, Nt)
    else:
        raise ValueError(
            "No chunks given. Need to give at `chunks_over_var` or `chunks_over_dim`."
        )

    plan = _rechunk(ds, chunks, worker_mem, str(path_out), temp_store=str(temp_store))

    plan.execute()
    zarr.consolidate_metadata(path_out)

    if temp_store is not None:
        sh.rmtree(temp_store)


@parse_config
def compute_indicators(
    ds: xr.Dataset,
    indicators: Union[
        str, PosixPath, Sequence[Indicator], Sequence[Tuple[str, Indicator]], ModuleType
    ],
    *,
    periods: list = None,
    to_level: str = "indicators",
) -> Union[dict, xr.Dataset]:
    """
    Calculates variables and indicators based on a YAML call to xclim.

    Parameters
    ----------
    ds : xr.Dataset
      Dataset to use for the indicators.
    indicators : Union[str, PosixPath, Sequence[Indicator], Sequence[Tuple[str, Indicator]]]
      Path to a YAML file that instructs on how to calculate missing variables.
      Can also be only the "stem", if translations and custom indices are implemented.
      Can be the indicator module directly, or a sequence of indicators or a sequence of
      tuples (indicator name, indicator) as returned by `iter_indicators()`.
    periods : list
      list of [start, end] of the contiguous periods to be evaluated, in the case of disjointed datasets.
      If left at None, the dataset will be considered continuous.
    to_level : str, optional
      The processing level to assign to the output.
      If None, the processing level of the inputs is preserved.


    Returns
    -------
    dict
      Dictionary (keys = timedeltas) with indicators separated by temporal resolution.

    """
    if isinstance(indicators, (str, Path)):
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

            # Infer the indicator's frequency
            freq = xr.infer_freq(out.time) if "time" in out.dims else "fx"

        else:
            # Multiple time periods to concatenate
            concats = []
            for period in periods:
                # Make the call to xclim
                ds_subset = ds.sel(time=slice(str(period[0]), str(period[1])))
                tmp = ind(ds=ds_subset)

                # Infer the indicator's frequency
                freq = xr.infer_freq(tmp.time) if "time" in tmp.dims else "fx"

                # In order to concatenate time periods, the indicator still needs a time dimension
                if freq == "fx":
                    tmp = tmp.assign_coords({"time": ds_subset.time[0]})

                concats.extend(tmp)
            out = xr.concat(concats, dim="time")

        # Create the dictionary key
        key = freq
        if key not in out_dict:
            if isinstance(out, tuple):  # In the case of multiple outputs
                out_dict[key] = xr.merge(o for o in out if o.name in indicators)
            else:
                out_dict[key] = out.to_dataset()

            # TODO: Double-check History, units, attrs, add missing variables (grid_mapping), etc.
            out_dict[key].attrs = ds.attrs
            out_dict[key].attrs.pop("cat/variable", None)
            out_dict[key].attrs["cat/xrfreq"] = freq
            out_dict[key].attrs["cat/frequency"] = CV.xrfreq_to_frequency(freq, None)
            if to_level is not None:
                out_dict[key].attrs["cat/processing_level"] = to_level

        else:
            if isinstance(out, tuple):  # In the case of multiple outputs
                for o in out:
                    if o.name in indicators:
                        out_dict[key][o.name] = o
            else:
                out_dict[key][out.name] = out

    return out_dict
