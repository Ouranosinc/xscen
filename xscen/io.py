# noqa: D100
import logging
import os
import shutil as sh
from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Union

import h5py
import netCDF4
import numpy as np
import xarray as xr
import zarr
from rechunker import rechunk as _rechunk
from xclim.core.calendar import get_calendar

from .config import parse_config
from .scripting import TimeoutException
from .utils import translate_time_chunk

logger = logging.getLogger(__name__)


__all__ = [
    "clean_incomplete",
    "estimate_chunks",
    "get_engine",
    "rechunk",
    "save_to_netcdf",
    "save_to_zarr",
    "subset_maxsize",
    "rechunk_for_saving",
]


def get_engine(file: str) -> str:
    """Use functionality of h5py to determine if a NetCDF file is compatible with h5netcdf.

    Parameters
    ----------
    file : str
        Path to the file.

    Returns
    -------
    str
        Engine to use with xarray
    """
    # find the ideal engine for xr.open_mfdataset
    if Path(file).suffix == ".zarr":
        engine = "zarr"
    elif h5py.is_hdf5(file):
        engine = "h5netcdf"
    else:
        engine = "netcdf4"

    return engine


def estimate_chunks(
    ds: Union[str, xr.Dataset],
    dims: list,
    target_mb: float = 50,
    chunk_per_variable: bool = False,
) -> dict:
    """Return an approximate chunking for a file or dataset.

    Parameters
    ----------
    ds : xr.Dataset, str
        Either a xr.Dataset or the path to a NetCDF file. Existing chunks are not taken into account.
    dims : list
        Dimension(s) on which to estimate the chunking. Not implemented for more than 2 dimensions.
    target_mb : float, optional
        Roughly the size of chunks (in Mb) to aim for.
    chunk_per_variable : bool
        If True, the output will be separated per variable. Otherwise, a common chunking will be found.

    Returns
    -------
    dict
        dictionary of estimated chunks

    """

    def _estimate_chunks(ds, target_mb, size_of_slice, rechunk_dims):
        # Approximate size of the chunks (equal across dims)
        approx_chunks = np.power(target_mb / size_of_slice, 1 / len(rechunk_dims))

        chunks_per_dim = dict()
        if len(rechunk_dims) == 1:
            rounding = (
                1
                if ds[rechunk_dims[0]].shape[0] <= 15
                else 5
                if ds[rechunk_dims[0]].shape[0] <= 250
                else 10
            )
            chunks_per_dim[rechunk_dims[0]] = np.max(
                [
                    np.min(
                        [
                            int(rounding * np.round(approx_chunks / rounding)),
                            ds[rechunk_dims[0]].shape[0],
                        ]
                    ),
                    1,
                ]
            )
        elif len(rechunk_dims) == 2:
            # Adjust approx_chunks based on the ratio of the rectangle sizes
            for d in rechunk_dims:
                rounding = (
                    1 if ds[d].shape[0] <= 15 else 5 if ds[d].shape[0] <= 250 else 10
                )
                adjusted_chunk = int(
                    rounding
                    * np.round(
                        approx_chunks
                        * (
                            ds[d].shape[0]
                            / np.prod(
                                [
                                    ds[dd].shape[0]
                                    for dd in rechunk_dims
                                    if dd not in [d]
                                ]
                            )
                        )
                        / rounding
                    )
                )
                chunks_per_dim[d] = np.max(
                    [np.min([adjusted_chunk, ds[d].shape[0]]), 1]
                )
        else:
            raise NotImplementedError(
                "estimating chunks on more than 2 dimensions is not implemented yet."
            )

        return chunks_per_dim

    out = {}
    # If ds is the path to a file, use NetCDF4
    if isinstance(ds, str):
        ds = netCDF4.Dataset(ds, "r")

        # Loop on variables
        for v in ds.variables:
            # Find dimensions to chunk
            rechunk_dims = list(set(dims).intersection(ds.variables[v].dimensions))
            if not rechunk_dims:
                continue

            dtype_size = ds.variables[v].datatype.itemsize
            num_elem_per_slice = np.prod(
                [ds[d].shape[0] for d in ds[v].dimensions if d not in rechunk_dims]
            )

            size_of_slice = (num_elem_per_slice * dtype_size) / 1024**2

            estimated_chunks = _estimate_chunks(
                ds, target_mb, size_of_slice, rechunk_dims
            )
            for other in set(ds[v].dimensions).difference(dims):
                estimated_chunks[other] = -1

            if chunk_per_variable:
                out[v] = estimated_chunks
            else:
                for d in estimated_chunks:
                    if (d not in out) or (out[d] > estimated_chunks[d]):
                        out[d] = estimated_chunks[d]

    # Else, use xarray
    else:
        for v in ds.data_vars:
            # Find dimensions to chunk
            rechunk_dims = list(set(dims).intersection(ds[v].dims))
            if not rechunk_dims:
                continue

            dtype_size = ds[v].dtype.itemsize
            num_elem_per_slice = np.prod(
                [ds[d].shape[0] for d in ds[v].dims if d not in rechunk_dims]
            )
            size_of_slice = (num_elem_per_slice * dtype_size) / 1024**2

            estimated_chunks = _estimate_chunks(
                ds, target_mb, size_of_slice, rechunk_dims
            )
            for other in set(ds[v].dims).difference(dims):
                estimated_chunks[other] = -1

            if chunk_per_variable:
                out[v] = estimated_chunks
            else:
                for d in estimated_chunks:
                    if (d not in out) or (out[d] > estimated_chunks[d]):
                        out[d] = estimated_chunks[d]

    return out


def subset_maxsize(
    ds: xr.Dataset,
    maxsize_gb: float,
) -> list:
    """Estimate a dataset's size and, if higher than the given limit, subset it alongside the 'time' dimension.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to be saved.
    maxsize_gb : float
        Target size for the NetCDF files.
        If the dataset is bigger than this number, it will be separated alongside the 'time' dimension.

    Returns
    -------
    list
        list of xr.Dataset subsetted alongside 'time' to limit the filesize to the requested maximum.
    """
    # Estimate the size of the dataset
    size_of_file = 0
    for v in ds:
        dtype_size = ds[v].dtype.itemsize
        varsize = np.prod(list(ds[v].sizes.values()))
        size_of_file = size_of_file + (varsize * dtype_size) / 1024**3

    if size_of_file < maxsize_gb:
        logger.info(f"Dataset is already smaller than {maxsize_gb} Gb.")
        return [ds]

    elif "time" in ds:
        years = np.unique(ds.time.dt.year)
        ratio = int(len(years) / (size_of_file / maxsize_gb))
        ds_sub = []
        for y in range(years[0], years[-1], ratio):
            ds_sub.extend([ds.sel({"time": slice(str(y), str(y + ratio - 1))})])
        return ds_sub

    else:
        raise NotImplementedError(
            f"Size of the NetCDF file exceeds the {maxsize_gb} Gb target, but the dataset does not contain a 'time' variable."
        )


def clean_incomplete(path: Union[str, os.PathLike], complete: Sequence[str]) -> None:
    """Delete un-catalogued variables from a zarr folder.

    The goal of this function is to clean up an incomplete calculation.
    It will remove any variable in the zarr that is neither in the `complete` list
    nor in the  `coords`.

    Parameters
    ----------
    path : str, Path
        A path to a zarr folder.
    complete : sequence of strings
        Name of variables that were completed.

    Returns
    -------
    None
    """
    path = Path(path)
    with xr.open_zarr(path) as ds:
        complete = set(complete).union(ds.coords.keys())

    for fold in filter(lambda p: p.is_dir(), path.iterdir()):
        if fold.name not in complete:
            logger.warning(f"Removing {fold} from disk")
            sh.rmtree(fold)


@parse_config
def save_to_netcdf(
    ds: xr.Dataset,
    filename: str,
    *,
    rechunk: Optional[dict] = None,
    netcdf_kwargs: Optional[dict] = None,
) -> None:
    """Save a Dataset to NetCDF, rechunking if requested.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to be saved.
    filename : str
        Name of the NetCDF file to be saved.
    rechunk : dict, optional
        This is a mapping from dimension name to new chunks (in any format understood by dask).
        Spatial dimensions can be generalized as 'X' and 'Y', which will be mapped to the actual grid type's
        dimension names.
        Rechunking is only done on *data* variables sharing dimensions with this argument.
    netcdf_kwargs : dict, optional
        Additional arguments to send to_netcdf()

    Returns
    -------
    None

    See Also
    --------
    xarray.Dataset.to_netcdf
    """
    if rechunk:
        ds = rechunk_for_saving(ds, rechunk)

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


@parse_config
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
    timeout_cleanup: bool = True,
) -> None:
    """Save a Dataset to Zarr format, rechunking if requested.

    According to mode, removes variables that we don't want to re-compute in ds.

    Parameters
    ----------
    ds : xr.Dataset
      Dataset to be saved.
    filename : str
      Name of the Zarr file to be saved.
    rechunk : dict, optional
      This is a mapping from dimension name to new chunks (in any format understood by dask).
      Spatial dimensions can be generalized as 'X' and 'Y' which will be mapped to the actual grid type's
      dimension names.
      Rechunking is only done on *data* variables sharing dimensions with this argument.
    zarr_kwargs : dict, optional
      Additional arguments to send to_zarr()
    compute : bool
      Whether to start the computation or return a delayed object.
    mode : {'f', 'o', 'a'}
      If 'f', fails if any variable already exists.
      if 'o', removes the existing variables.
      if 'a', skip existing variables, writes the others.
    encoding : dict, optional
      If given, skipped variables are popped in place.
    itervar : bool
      If True, (data) variables are written one at a time, appending to the zarr.
      If False, this function computes, no matter what was passed to kwargs.
    timeout_cleanup : bool
      If True (default) and a :py:class:`xscen.scripting.TimeoutException` is raised during the writing,
      the variable being written is removed from the dataset as it is incomplete.
      This does nothing if `compute` is False.

    Returns
    -------
    dask.delayed object if compute=False, None otherwise.

    See Also
    --------
    xarray.Dataset.to_zarr
    """
    # to address this issue https://github.com/pydata/xarray/issues/3476
    for v in list(ds.coords.keys()):
        if ds.coords[v].dtype == object:
            ds[v].encoding.clear()

    if rechunk:
        ds = rechunk_for_saving(ds, rechunk)

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
            try:
                dsvar.to_zarr(
                    path,
                    mode="a",
                    encoding={k: v for k, v in (encoding or {}).items() if k in dsvar},
                    **zarr_kwargs,
                )
            except TimeoutException:
                if timeout_cleanup:
                    logger.info(f"Removing incomplete {name}.")
                    sh.rmtree(path / name)
                raise

    else:
        logger.debug(f"Writing {list(ds.data_vars.keys())} for {filename}.")
        try:
            return ds.to_zarr(
                filename, compute=compute, mode="a", encoding=encoding, **zarr_kwargs
            )
        except TimeoutException:
            if timeout_cleanup:
                logger.info(
                    f"Removing incomplete {list(ds.data_vars.keys())} for {filename}."
                )
                for name in ds.data_vars:
                    sh.rmtree(path / name)
            raise


def rechunk_for_saving(ds, rechunk):
    """Rechunk before saving to .zarr or .nc, generalized as Y/X for different axes lat/lon, rlat/rlon.

    Parameters
    ----------
    ds : xr.Dataset
        The xr.Dataset to be rechunked.
    rechunk : dict
        A dictionary with the dimension names of ds and the new chunk size. Spatial dimensions
        can be provided as X/Y.

    Returns
    -------
    xr.Dataset
        The dataset with new chunking.
    """
    for rechunk_var in ds.data_vars:
        # Support for chunks varying per variable
        if rechunk_var in rechunk:
            rechunk_dims = rechunk[rechunk_var].copy()
        else:
            rechunk_dims = rechunk.copy()

        # get actual axes labels
        if "X" in rechunk_dims and "X" not in ds.dims:
            rechunk_dims[ds.cf.axes["X"][0]] = rechunk_dims.pop("X")
        if "Y" in rechunk_dims and "Y" not in ds.dims:
            rechunk_dims[ds.cf.axes["Y"][0]] = rechunk_dims.pop("Y")

        ds[rechunk_var] = ds[rechunk_var].chunk(
            {d: chnks for d, chnks in rechunk_dims.items() if d in ds[rechunk_var].dims}
        )
        ds[rechunk_var].encoding.pop("chunksizes", None)
        ds[rechunk_var].encoding.pop("chunks", None)

    return ds


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
    chunks_over_var : dict
        Mapping from variables to mappings from dimension name to size. Give this argument or `chunks_over_dim`.
    chunks_over_dim : dict
        Mapping from dimension name to size that will be used for all variables in ds.
        Give this argument or `chunks_over_var`.
    worker_mem : str
        The maximal memory usage of each task.
        When using a distributed Client, this an approximate memory per thread.
        Each worker of the client should have access to 10-20% more memory than this times the number of threads.
    temp_store : path, str, optional
        A path to a zarr where to store intermediate results.
    overwrite : bool
        If True, it will delete whatever is in path_out before doing the rechunking.

    Returns
    -------
    None

    See Also
    --------
    rechunker.rechunk
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
