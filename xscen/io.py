import logging
import os
import shutil as sh
from pathlib import Path
from typing import Sequence, Union

import h5py
import netCDF4
import numpy as np
import xarray as xr
from xclim.core.calendar import get_calendar

logger = logging.getLogger(__name__)

__all__ = [
    "clean_incomplete",
    "estimate_chunks",
    "get_calendar",
    "get_engine",
    "subset_maxsize",
]


def get_engine(file: str) -> str:
    """
    Uses a h5py functionality to determine if a NetCDF file is compatible with h5netcdf

    Parameters
    ----------
    file: str
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
    """
    Returns an approximate chunking for a file or dataset.

    Parameters
    ----------
    ds : xr.Dataset, str
      Either a xr.Dataset or the path to a NetCDF file. Existing chunks are not taken into account.
    dims : list
      Dimension(s) on which to estimate the chunking. Not implemented for more than 2 dimensions.
    target_mb: float, optional
      Roughly the size of chunks (in Mb) to aim for.
    chunk_per_variable: bool
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
    """
    Estimate a dataset's size and, if higher than the given limit, subset it alongside the 'time' dimension

    Parameters
    ----------
    ds : xr.Dataset
      Dataset to be saved.
    maxsize_gb : float
      Target size for the NetCDF files. If the dataset is bigger than this number, it will be separated alongside the 'time' dimension.

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
