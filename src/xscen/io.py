"""Input/Output functions for xscen."""

import datetime
import json
import logging
import os
import shutil as sh
from collections import defaultdict
from collections.abc import Sequence
from inspect import signature
from pathlib import Path
from zipfile import ZipFile

import h5py
import netCDF4
import numpy as np
import pandas as pd
import sparse as sp
import xarray as xr
import zarr
from numcodecs.bitround import BitRound
from rechunker import rechunk as _rechunk
from xclim.core.options import METADATA_LOCALES
from xclim.core.options import OPTIONS as XC_OPTIONS

from .config import parse_config
from .scripting import TimeoutException
from .utils import TRANSLATOR, season_sort_key, strip_cat_attrs, translate_time_chunk


logger = logging.getLogger(__name__)
KEEPBITS = defaultdict(lambda: 12)


__all__ = [
    "clean_incomplete",
    "estimate_chunks",
    "get_engine",
    "load_sparse",
    "make_toc",
    "rechunk",
    "rechunk_for_saving",
    "round_bits",
    "save_sparse",
    "save_to_netcdf",
    "save_to_table",
    "save_to_zarr",
    "subset_maxsize",
    "to_table",
    "unzip_directory",
    "zip_directory",
]


def get_engine(file: str | os.PathLike) -> str:
    """
    Determine which Xarray engine should be used to open the given file.

    The .zarr, .zarr.zip and .zip extensions are recognized as Zarr datasets,
    the rest is seen as a netCDF. If the file is HDF5, the h5netcdf engine is used.

    Parameters
    ----------
    file : str or os.PathLike
        Path to the file.

    Returns
    -------
    str
        Engine to use with xarray
    """
    # find the ideal engine for xr.open_mfdataset
    if Path(file).suffix in [".zarr", ".zip", ".zarr.zip"]:
        engine = "zarr"
    elif h5py.is_hdf5(file):
        engine = "h5netcdf"
    else:
        engine = "netcdf4"

    return engine


def estimate_chunks(  # noqa: C901
    ds: str | os.PathLike | xr.Dataset,
    dims: list,
    target_mb: float = 50,
    chunk_per_variable: bool = False,
) -> dict:
    """
    Return an approximate chunking for a file or dataset.

    Parameters
    ----------
    ds : xr.Dataset or os.PathLike or str
        Either a xr.Dataset or the path to a NetCDF file. Existing chunks are not taken into account.
    dims : list
        Dimension(s) on which to estimate the chunking. Not implemented for more than 2 dimensions.
    target_mb : float
        Roughly the size of chunks (in Mb) to aim for.
    chunk_per_variable : bool
        If True, the output will be separated per variable. Otherwise, a common chunking will be found.

    Returns
    -------
    dict
        A dictionary mapping dimensions to chunk sizes.
    """

    def _estimate_chunks(da, target_mb, size_of_slice, rechunk_dims):
        # Divide the dimensions by the smallest dimension
        min_dim = np.min([da[d].shape[0] for d in rechunk_dims])
        ratio = {d: da[d].shape[0] / min_dim for d in rechunk_dims}

        # Get the approximate number of chunks, supposing the chunks are cubes
        approx_chunks = np.power(target_mb / size_of_slice, 1 / len(rechunk_dims))

        # Redistribute the chunks based on the ratio of the dimensions
        x = (approx_chunks ** len(rechunk_dims) / np.prod(list(ratio.values()))) ** (1 / len(rechunk_dims))
        rounding_per_dim = {d: 1 if da[d].shape[0] <= 15 else 5 if da[d].shape[0] <= 250 else 10 for d in rechunk_dims}
        chunks_per_dim = {d: int(rounding_per_dim[d] * np.round(x * ratio[d] / rounding_per_dim[d])) for d in rechunk_dims}
        chunks_per_dim = {d: np.max([np.min([chunks_per_dim[d], da[d].shape[0]]), 1]) for d in rechunk_dims}

        return chunks_per_dim

    out = {}
    # If ds is the path to a file, use NetCDF4
    if isinstance(ds, str | os.PathLike):
        ds = netCDF4.Dataset(ds, "r")

        # Loop on variables
        for v in ds.variables:
            # Find dimensions to chunk
            rechunk_dims = list(set(dims).intersection(ds.variables[v].dimensions))
            if not rechunk_dims or v in ds.dimensions:
                continue

            dtype_size = ds.variables[v].datatype.itemsize
            num_elem_per_slice = np.prod([ds[d].shape[0] for d in ds[v].dimensions if d not in rechunk_dims])

            size_of_slice = (num_elem_per_slice * dtype_size) / 1024**2

            estimated_chunks = _estimate_chunks(ds, target_mb, size_of_slice, rechunk_dims)
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
            num_elem_per_slice = np.prod([ds[d].shape[0] for d in ds[v].dims if d not in rechunk_dims])
            size_of_slice = (num_elem_per_slice * dtype_size) / 1024**2

            estimated_chunks = _estimate_chunks(ds, target_mb, size_of_slice, rechunk_dims)
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
) -> list[xr.Dataset]:
    """
    Estimate a dataset's size and, if higher than the given limit, subset it alongside the 'time' dimension.

    Parameters
    ----------
    ds : xr.Dataset
        The Dataset to be saved.
    maxsize_gb : float
        Target size for the NetCDF files.
        If the dataset is bigger than this number, it will be separated alongside the 'time' dimension.

    Returns
    -------
    list of xr.Dataset
        List of xr.Dataset subsetted alongside 'time' to limit the filesize to the requested maximum.
    """
    # Estimate the size of the dataset
    size_of_file = 0
    for v in ds:
        dtype_size = ds[v].dtype.itemsize
        varsize = np.prod(list(ds[v].sizes.values()))
        size_of_file = size_of_file + (varsize * dtype_size) / 1024**3

    if size_of_file < maxsize_gb:
        msg = f"Dataset is already smaller than {maxsize_gb} Gb."
        logger.info(msg)
        return [ds]

    elif "time" in ds.dims:
        years = np.unique(ds.time.dt.year)
        ratio = np.max([int(len(years) / (size_of_file / maxsize_gb)), 1])
        ds_sub = []
        for y in range(years[0], years[-1] + 1, ratio):
            ds_sub.extend([ds.sel({"time": slice(str(y), str(y + ratio - 1))})])
        return ds_sub

    else:
        raise NotImplementedError(f"Size of the NetCDF file exceeds the {maxsize_gb} Gb target, but the dataset does not contain a 'time' variable.")


def clean_incomplete(
    path: str | os.PathLike,
    complete: Sequence[str] | None = None,
    incomplete: Sequence[str] | None = None,
) -> None:
    """
    Delete un-catalogued variables from a zarr folder.

    The goal of this function is to clean up an incomplete calculation.
    It will remove any variable in the zarr that is neither in the `complete` list
    nor in the  `coords`.

    Parameters
    ----------
    path : str, Path
        A path to a zarr folder.
    complete : sequence of strings, optional
        Name of variables that were completed. All other variables (except coordinates) will be removed.
        Use either `complete` or `incomplete`, not both.
    incomplete : sequence of strings, optional
        Name of variables that should be removed. Coordinates and dimensions will never be removed through this function.
        Use either `complete` or `incomplete`, not both.

    Returns
    -------
    None
    """
    path = Path(path)

    if complete is not None and incomplete is not None:
        raise ValueError("Use either `complete` or `incomplete`, not both.")
    v_to_rm = []

    if complete is not None:
        with xr.open_zarr(path) as ds:
            complete = set(complete).union(ds.coords.keys())

        for fold in filter(lambda p: p.is_dir(), path.iterdir()):
            if fold.name not in complete:
                v_to_rm.append(fold.name)
                msg = f"Removing {fold} from disk"
                logger.warning(msg)
                sh.rmtree(fold)

    elif incomplete is not None:
        with xr.open_zarr(path) as ds:
            incomplete = [v for v in incomplete if (v not in ds.coords) and (v not in ds.dims)]

        for fold in filter(lambda p: p.is_dir(), path.iterdir()):
            if fold.name in incomplete:
                v_to_rm.append(fold.name)
                msg = f"Removing {fold} from disk"
                logger.warning(msg)
                sh.rmtree(fold)

    # Update the .zmetadata file
    with (path / ".zmetadata").open("r") as f:
        metadata = json.load(f)
    for v in v_to_rm:
        [metadata["metadata"].pop(k) for k in list(metadata["metadata"].keys()) if k.startswith(f"{v}/.")]
    with (path / ".zmetadata").open("w") as f:
        json.dump(metadata, f, indent=2)


def _coerce_attrs(attrs):
    """Ensure no funky objects in attrs."""
    for k in list(attrs.keys()):
        if not (
            isinstance(attrs[k], str | float | int | np.ndarray) or isinstance(attrs[k], tuple | list) and isinstance(attrs[k][0], str | float | int)
        ):
            attrs[k] = str(attrs[k])


def _np_bitround(array: xr.DataArray, keepbits: int):
    """Bitround for Arrays."""
    codec = BitRound(keepbits=keepbits)
    data = array.copy()  # otherwise overwrites the input
    encoded = codec.encode(data)
    return codec.decode(encoded)


def round_bits(da: xr.DataArray, keepbits: int):
    """
    Round floating point variable by keeping a given number of bits in the mantissa, dropping the rest. This allows for a much better compression.

    Parameters
    ----------
    da : xr.DataArray
        Variable to be rounded.
    keepbits : int
        The number of bits of the mantissa to keep.
    """
    encoding = da.encoding
    da = xr.apply_ufunc(_np_bitround, da, keepbits, dask="parallelized", keep_attrs=True)
    da.encoding = encoding
    da.attrs["_QuantizeBitRoundNumberOfSignificantDigits"] = keepbits
    new_history = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Data compressed with BitRound by keeping {keepbits} bits."
    history = new_history + " \n " + da.attrs["history"] if "history" in da.attrs else new_history
    da.attrs["history"] = history
    return da


def _get_keepbits(bitround: bool | int | dict, varname: str, vartype):
    # Guess the number of bits to keep depending on how bitround was passed, the var dtype and the var name.
    if not np.issubdtype(vartype, np.floating) or bitround is False:
        if isinstance(bitround, dict) and varname in bitround:
            raise ValueError(f"A keepbits value was given for variable {varname} even though it is not of a floating dtype.")
        return None
    if bitround is True:
        return KEEPBITS[varname]
    if isinstance(bitround, int):
        return bitround
    if isinstance(bitround, dict):
        return bitround.get(varname, KEEPBITS[varname])
    return None


@parse_config
def save_to_netcdf(
    ds: xr.Dataset,
    filename: str | os.PathLike,
    *,
    rechunk: dict | None = None,
    bitround: bool | int | dict = False,
    compute: bool = True,
    netcdf_kwargs: dict | None = None,
    strip_cat_metadata: bool = False,
):
    """
    Save a Dataset to NetCDF, rechunking or compressing if requested.

    Parameters
    ----------
    ds : xr.Dataset
        The Dataset to be saved.
    filename : str or os.PathLike
        Name of the NetCDF file to be saved.
    rechunk : dict, optional
        This is a mapping from dimension name to new chunks (in any format understood by dask).
        Spatial dimensions can be generalized as 'X' and 'Y', which will be mapped to the actual grid type's
        dimension names.
        Rechunking is only done on *data* variables sharing dimensions with this argument.
    bitround : bool or int or dict
        If not False, float variables are bit-rounded by dropping a certain number of bits from their mantissa,
        allowing for a much better compression.
        If an int, this is the number of bits to keep for all float variables.
        If a dict, a mapping from variable name to the number of bits to keep.
        If True, the number of bits to keep is guessed based on the variable's name, defaulting to 12,
        which yields a relative error below 0.013%.
    compute : bool
        Whether to start the computation or return a delayed object.
    netcdf_kwargs : dict, optional
        Additional arguments to send to_netcdf()
    strip_cat_metadata : bool
        If True (default), strips all catalog-added attributes before saving the dataset.

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

    for var in list(ds.data_vars.keys()):
        if keepbits := _get_keepbits(bitround, var, ds[var].dtype):
            ds = ds.assign({var: round_bits(ds[var], keepbits)})
        # Remove original_shape from encoding, since it can cause issues with some engines.
        ds[var].encoding.pop("original_shape", None)

    if strip_cat_metadata:
        ds = strip_cat_attrs(ds)

    _coerce_attrs(ds.attrs)
    for var in ds.variables.values():
        _coerce_attrs(var.attrs)

    return ds.to_netcdf(filename, compute=compute, **netcdf_kwargs)


@parse_config
def save_to_zarr(  # noqa: C901
    ds: xr.Dataset,
    filename: str | os.PathLike,
    *,
    rechunk: dict | None = None,
    zarr_kwargs: dict | None = None,
    compute: bool = True,
    encoding: dict | None = None,
    bitround: bool | int | dict = False,
    mode: str = "f",
    itervar: bool = False,
    timeout_cleanup: bool = True,
    strip_cat_metadata: bool = False,
):
    """
    Save a Dataset to Zarr format, rechunking and compressing if requested.

    According to mode, removes variables that we don't want to re-compute in ds.

    Parameters
    ----------
    ds : xr.Dataset
        The Dataset to be saved.
    filename : str
        Name of the Zarr file to be saved.
    rechunk : dict, optional
        This is a mapping from dimension name to new chunks (in any format understood by dask).
        Spatial dimensions can be generalized as 'X' and 'Y' which will be mapped to the actual grid type's
        dimension names.
        Rechunking is only done on *data* variables sharing dimensions with this argument.
    zarr_kwargs : dict, optional
        Additional arguments to send `to_zarr()`.
    compute : bool
        Whether to start the computation or return a delayed object.
    mode : {'f', 'o', 'a'}
        If 'f', fails if any variable already exists.
        if 'o', removes the existing variables.
        if 'a', skip existing variables, writes the others.
    encoding : dict, optional
        If given here instead of 'zarr_kwargs', encoding will only be applied to the variables that are being written,
        skipping those that are already in the zarr.
    bitround : bool or int or dict
        If not False, float variables are bit-rounded by dropping a certain number of bits from their mantissa,
        allowing for a much better compression.
        If an int, this is the number of bits to keep for all float variables.
        If a dict, a mapping from variable name to the number of bits to keep.
        If True, the number of bits to keep is guessed based on the variable's name, defaulting to 12,
        which yields a relative error of 0.012%.
    itervar : bool
        If True, (data) variables are written one at a time, appending to the zarr.
        If False, this function computes, no matter what was passed to kwargs.
    timeout_cleanup : bool
        If True (default) and a :py:class:`xscen.scripting.TimeoutException` is raised during the writing,
        the variable being written is removed from the dataset as it is incomplete.
        This does nothing if `compute` is False.
    strip_cat_metadata : bool
        If True (default), strips all catalog-added attributes before saving the dataset.

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
                clean_incomplete(path, incomplete=[var])
            return False

        if mode == "a":
            # In all cases, we need to skip the encoding of existing variables.
            if exists:
                if encoding:
                    encoding.pop(var, None)

            # If we are not appending, we need to skip the writing of existing variables.
            if "append_dim" not in zarr_kwargs:
                return exists

            # If we are appending, we need to raise an error if there are new variables.
            elif exists is False:
                raise ValueError("When 'append_dim' is set in zarr_kwargs, all variables must already exist in the dataset.")

            return False

    for var in list(ds.data_vars.keys()):
        if _skip(var):
            msg = f"Skipping {var} in {path}."
            logger.info(msg)
            ds = ds.drop_vars(var)
            continue
        if keepbits := _get_keepbits(bitround, var, ds[var].dtype):
            ds = ds.assign({var: round_bits(ds[var], keepbits)})
        # Remove original_shape from encoding, since it can cause issues with some engines.
        ds[var].encoding.pop("original_shape", None)

    if len(ds.data_vars) == 0:
        return None

    if strip_cat_metadata:
        ds = strip_cat_attrs(ds)

    _coerce_attrs(ds.attrs)
    for var in ds.variables.values():
        _coerce_attrs(var.attrs)

    if itervar:
        zarr_kwargs["compute"] = True
        allvars = set(ds.data_vars.keys())
        if mode == "f":
            dsbase = ds.drop_vars(allvars)
            dsbase.to_zarr(path, **zarr_kwargs)
        if mode == "o":
            dsbase = ds.drop_vars(allvars)
            dsbase.to_zarr(path, **zarr_kwargs, mode="w")
        for i, name in enumerate(ds.data_vars.keys()):
            msg = f"Writing {name} ({i + 1} of {len(ds.data_vars)}) to {path}"
            logger.debug(msg)
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
                    clean_incomplete(path, incomplete=[name])
                raise

    else:
        msg = f"Writing {list(ds.data_vars.keys())} for {filename}."
        logger.debug(msg)
        try:
            return ds.to_zarr(filename, compute=compute, mode="a", encoding=encoding, **zarr_kwargs)
        except TimeoutException:
            if timeout_cleanup:
                clean_incomplete(path, incomplete=list(ds.data_vars.keys()))
            raise


def _to_dataframe(
    data: xr.DataArray,
    row: list[str],
    column: list[str],
    coords: list[str],
    coords_dims: dict,
):
    """Convert a DataArray to a DataFrame with support for MultiColumn."""
    df = data.to_dataframe()
    if not column:
        # Fast track for the easy case where xarray's default is already what we want.
        return df
    df_data = df[[data.name]].reset_index().pivot(index=row, columns=column).droplevel(None, axis=1)
    dfs = []
    for v in coords:
        drop_cols = [c for c in column if c not in coords_dims[v]]
        cols = [c for c in column if c in coords_dims[v]]
        dfc = df[[v]].reset_index().drop(columns=drop_cols).pivot(index=row, columns=cols)
        cols = dfc.columns
        # The "None" level has the aux coord name we want it either at the same level as variable, or at lowest missing level otherwise.
        varname_lvl = "variable" if "variable" in drop_cols else drop_cols[-1]
        cols = cols.rename(varname_lvl if not isinstance(cols, pd.MultiIndex) else [nm or varname_lvl for nm in cols.name])
        if isinstance(df_data.columns, pd.MultiIndex) or isinstance(cols, pd.MultiIndex):
            # handle different depth of multicolumns, expand MultiCol of coord with None for missing levels.
            cols = pd.MultiIndex.from_arrays(
                [cols.get_level_values(lvl) if lvl in cols.names else [None] for lvl in df_data.columns.names],
                names=df_data.columns.names,
            )
        dfc.columns = cols
        dfs.append(dfc[~dfc.index.duplicated()])  # We dropped columns thus the index is not unique anymore
    dfs.append(df_data)
    return pd.concat(dfs, axis=1).sort_index(level=row, key=season_sort_key)


def to_table(
    ds: xr.Dataset | xr.DataArray,
    *,
    row: str | Sequence[str] | None = None,
    column: str | Sequence[str] | None = None,
    sheet: str | Sequence[str] | None = None,
    coords: bool | str | Sequence[str] = True,
) -> pd.DataFrame | dict:
    """
    Convert a dataset to a pandas DataFrame with support for multicolumns and multisheet.

    This function will trigger a computation of the dataset.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Dataset or DataArray to be saved.
        If a Dataset with more than one variable is given, the dimension "variable"
        must appear in one of `row`, `column` or `sheet`.
    row : str or sequence of str, optional
        Name of the dimension(s) to use as indexes (rows).
        Default is all data dimensions.
    column : str or sequence of str, optional
        Name of the dimension(s) to use as columns.
        Default is "variable", i.e. the name of the variable(s).
    sheet : str or sequence of str, optional
        Name of the dimension(s) to use as sheet names.
    coords: bool or str or sequence of str
        A list of auxiliary coordinates to add to the columns (as would variables).
        If True, all (if any) are added.

    Returns
    -------
    pd.DataFrame or dict
        DataFrame with a MultiIndex with levels `row` and MultiColumn with levels `column`.
        If `sheet` is given, the output is dictionary with keys for each unique "sheet" dimensions tuple, values are DataFrames.
        The DataFrames are always sorted with level priority as given in `row` and in ascending order.
    """
    if isinstance(ds, xr.Dataset):
        da = ds.to_array(name="data")
        if len(ds) == 1:
            da = da.isel(variable=0)
            da.name = str(da["variable"].values)
            da = da.drop_vars("variable")
    else:
        da = ds

    def _ensure_list(seq):
        if isinstance(seq, str):
            return [seq]
        return list(seq)

    passed_dims = set().union(_ensure_list(row or []), _ensure_list(column or []), _ensure_list(sheet or []))
    if row is None:
        row = [d for d in da.dims if d != "variable" and d not in passed_dims]
    row = _ensure_list(row)
    if column is None:
        column = ["variable"] if isinstance(ds, xr.Dataset) and len(ds) > 1 and "variable" not in passed_dims else []
    column = _ensure_list(column)
    if sheet is None:
        sheet = []
    sheet = _ensure_list(sheet)

    needed_dims = row + column + sheet
    if len(set(needed_dims)) != len(needed_dims):
        raise ValueError(f"Repeated dimension names. Got row={row}, column={column} and sheet={sheet}.Each dimension should appear only once.")
    if set(needed_dims) != set(da.dims):
        raise ValueError(f"Passed row, column and sheet do not match available dimensions. Got {needed_dims}, data has {da.dims}.")

    if coords is not True:
        coords = _ensure_list(coords or [])
        drop = set(da.coords.keys()) - set(da.dims) - set(coords)
        da = da.drop_vars(drop)
    else:
        coords = list(set(da.coords.keys()) - set(da.dims))
    if len(coords) > 1 and ("variable" in row or "variable" in sheet):
        raise NotImplementedError(
            "Keeping auxiliary coords is not implemented when 'variable' is in the row or in the sheets."
            "Pass `coords=False` or put 'variable' in `column` instead."
        )

    table_kwargs = dict(
        row=row,
        column=column,
        coords=coords,
        coords_dims={c: ds[c].dims for c in coords},
    )
    if sheet:
        out = {}
        das = da.stack(sheet=sheet)
        for elem in das.sheet:
            out[elem.item()] = _to_dataframe(das.sel(sheet=elem, drop=True), **table_kwargs)
        return out
    return _to_dataframe(da, **table_kwargs)


def make_toc(ds: xr.Dataset | xr.DataArray, loc: str | None = None) -> pd.DataFrame:
    """
    Make a table of content describing a dataset's variables.

    This return a simple DataFrame with variable names as index, the long_name as "description" and units.
    Column names and long names are taken from the activated locale if found, otherwise the english version is taken.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Dataset or DataArray from which to extract the relevant metadata.
    loc : str, optional
        The locale to use. If None, either the first locale in the list of activated xclim locales is used, or "en" if none is activated.

    Returns
    -------
    pd.DataFrame
      A DataFrame with variables as index, and columns "description" and "units".
    """
    if loc is None:
        loc = (XC_OPTIONS[METADATA_LOCALES] or ["en"])[0]
    locsuf = "" if loc == "en" else f"_{loc}"
    _ = TRANSLATOR[loc]  # Combine translation and gettext parsing (like it usually is)

    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset()

    toc = pd.DataFrame.from_records(
        [
            {
                _("Variable"): vv,
                _("Description"): da.attrs.get(f"long_name{locsuf}", da.attrs.get("long_name")),
                _("Units"): da.attrs.get("units"),
            }
            for vv, da in ds.data_vars.items()
        ],
    ).set_index(_("Variable"))

    # Add global attributes by using a fake variable and description
    if len(ds.attrs) > 0:
        globattr = pd.DataFrame.from_records(
            [
                {
                    _("Variable"): vv,
                    _("Description"): da,
                    _("Units"): "",
                }
                for vv, da in ds.attrs.items()
            ],
        ).set_index(_("Variable"))
        globattr.attrs["name"] = _("Global attributes")

        # Empty row to separate global attributes from variables
        toc = pd.concat([toc, pd.DataFrame(index=[""])])
        toc = pd.concat([toc, pd.DataFrame(index=[_("Global attributes")])])
        toc = pd.concat([toc, globattr])
    toc.attrs["name"] = _("Content")

    return toc


TABLE_FORMATS = {".csv": "csv", ".xls": "excel", ".xlsx": "excel"}


def save_to_table(  # noqa: C901
    ds: xr.Dataset | xr.DataArray,
    filename: str | os.PathLike,
    output_format: str | None = None,
    *,
    row: str | Sequence[str] | None = None,
    column: None | str | Sequence[str] = None,
    sheet: str | Sequence[str] | None = None,
    coords: bool | Sequence[str] = True,
    col_sep: str = "_",
    row_sep: str | None = None,
    add_toc: bool | pd.DataFrame = False,
    **kwargs,
):
    r"""
    Save the dataset to a tabular file (csv, excel, ...).

    This function will trigger a computation of the dataset.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Dataset or DataArray to be saved.
        If a Dataset with more than one variable is given, the dimension "variable"
        must appear in one of `row`, `column` or `sheet`.
    filename : str or os.PathLike
        Name of the file to be saved.
    output_format: {'csv', 'excel', ...}, optional
        The output format. If None (default), it is inferred
        from the extension of `filename`. Not all possible output format are supported for inference.
        Valid values are any that matches a :py:class:`pandas.DataFrame` method like "df.to_{format}".
    row : str or sequence of str, optional
        Name of the dimension(s) to use as indexes (rows).
        Default is all data dimensions.
    column : str or sequence of str, optional
        Name of the dimension(s) to use as columns.
        When using a Dataset with more than 1 variable, default is "variable", i.e. the name of the variable(s).
        When using a DataArray, default is None.
    sheet : str or sequence of str, optional
        Name of the dimension(s) to use as sheet names.
        Only valid if the output format is excel.
    coords: bool or sequence of str
        A list of auxiliary coordinates to add to the columns (as would variables).
        If True, all (if any) are added.
    col_sep : str,
        Multi-columns (except in Excel) and sheet names are concatenated with this separator.
    row_sep : str, optional
        Multi-index names are concatenated with this separator, except in Excel.
        If None (default), each level is written in its own column.
    add_toc : bool or DataFrame
        A table of content to add as the first sheet. Only valid if the output format is excel.
        If True, :py:func:`make_toc` is used to generate the toc.
        The sheet name of the toc can be given through the "name" attribute of the DataFrame, otherwise "Content" is used.
    \*\*kwargs:
        Other arguments passed to the pandas function.
        If the output format is excel, kwargs to :py:class:`pandas.ExcelWriter` can be given here as well.
    """
    filename = Path(filename)

    if output_format is None:
        output_format = TABLE_FORMATS.get(filename.suffix)
    if output_format is None:
        raise ValueError(f"Output format could not be inferred from filename {filename.name}. Please pass `output_format`.")

    if column is None and isinstance(ds, xr.Dataset) and len(ds.data_vars) > 1:
        column = "variable"

    if sheet is not None and output_format != "excel":
        raise ValueError(f"Argument `sheet` is only valid with excel as the output format. Got {output_format}.")
    if add_toc is not False and output_format != "excel":
        raise ValueError(f"A TOC was requested, but the output format is not Excel. Got {output_format}.")

    out = to_table(ds, row=row, column=column, sheet=sheet, coords=coords)

    if add_toc is not False:
        if not sheet:
            out = {("data",): out}
        if add_toc is True:
            add_toc = make_toc(ds)
        out = {(add_toc.attrs.get("name", "Content"),): add_toc, **out}

    # Get engine_kwargs
    if output_format == "excel":
        engine_kwargs = {}  # Extract engine kwargs
        for arg in signature(pd.ExcelWriter).parameters:
            if arg in kwargs:
                engine_kwargs[arg] = kwargs.pop(arg)
    else:
        engine_kwargs = {}

    if sheet or (add_toc is not False):
        with pd.ExcelWriter(filename, **engine_kwargs) as writer:
            for sheet_name, df in out.items():
                df.to_excel(writer, sheet_name=col_sep.join(sheet_name), **kwargs)
    elif len(engine_kwargs) > 0:
        with pd.ExcelWriter(filename, **engine_kwargs) as writer:
            out.to_excel(writer, **kwargs)
    else:
        if output_format != "excel" and isinstance(out.columns, pd.MultiIndex):
            out.columns = out.columns.map(lambda lvls: col_sep.join(map(str, lvls)))
        if output_format != "excel" and row_sep is not None and isinstance(out.index, pd.MultiIndex):
            new_name = row_sep.join(out.index.names)
            out.index = out.index.map(lambda lvls: row_sep.join(map(str, lvls)))
            out.index.name = new_name
        getattr(out, f"to_{output_format}")(filename, **kwargs)


def rechunk_for_saving(ds: xr.Dataset, rechunk: dict):
    """
    Rechunk before saving to .zarr or .nc, generalized as Y/X for different axes lat/lon, rlat/rlon.

    Parameters
    ----------
    ds : xr.Dataset
        The xr.Dataset to be rechunked.
    rechunk : dict
        A dictionary with the dimension names of ds and the new chunk size.
        Spatial dimensions can be provided as X/Y.

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

        ds[rechunk_var] = ds[rechunk_var].chunk({d: chnks for d, chnks in rechunk_dims.items() if d in ds[rechunk_var].dims})
        ds[rechunk_var].encoding["chunksizes"] = tuple(rechunk_dims[d] if d in rechunk_dims else ds[d].shape[0] for d in ds[rechunk_var].dims)
        ds[rechunk_var].encoding.pop("chunks", None)
        ds[rechunk_var].encoding["preferred_chunks"] = rechunk_dims

    return ds


@parse_config
def rechunk(
    path_in: os.PathLike | str | xr.Dataset,
    path_out: os.PathLike | str,
    *,
    chunks_over_var: dict | None = None,
    chunks_over_dim: dict | None = None,
    worker_mem: str,
    temp_store: os.PathLike | str | None = None,
    overwrite: bool = False,
) -> None:
    """
    Rechunk a dataset into a new zarr.

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
    temp_store : path or str, optional
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
    # Remove all input chunks information, avoids an error with rechunker and xarray >= 2024.10
    # TODO: Remove this and pin rechunker when https://github.com/pangeo-data/rechunker/pull/156 is merged and released
    for var in ds.variables.values():
        var.encoding.pop("chunks", None)

    variables = list(ds.data_vars)
    if chunks_over_var:
        chunks = chunks_over_var
    elif chunks_over_dim:
        chunks = {v: {d: chunks_over_dim[d] for d in ds[v].dims} for v in variables}
        chunks.update(time=None, lat=None, lon=None)
        cal = ds.time.dt.calendar
        Nt = ds.time.size
        chunks = translate_time_chunk(chunks, cal, Nt)
    else:
        raise ValueError("No chunks given. Need to give at `chunks_over_var` or `chunks_over_dim`.")
    plan = _rechunk(ds, chunks, worker_mem, str(path_out), temp_store=str(temp_store))

    plan.execute()
    zarr.consolidate_metadata(path_out)

    if temp_store is not None:
        sh.rmtree(temp_store)


def zip_directory(
    root: str | os.PathLike,
    zipfile: str | os.PathLike,
    delete: bool = False,
    **zip_args,
):
    r"""
    Make a zip archive of the content of a directory.

    Parameters
    ----------
    root : path
        The directory with the content to archive.
    zipfile : path
        The zip file to create.
    delete : bool
        If True, the original directory is deleted after zipping.
    \*\*zip_args
        Any other arguments to pass to :py:mod:`zipfile.ZipFile`, such as "compression".
        The default is to make no compression (``compression=ZIP_STORED``).
    """
    root = Path(root)

    def _add_to_zip(zf, path, root):
        zf.write(path, path.relative_to(root))
        if path.is_dir():
            for subpath in path.iterdir():
                _add_to_zip(zf, subpath, root)

    with ZipFile(zipfile, "w", **zip_args) as zf:
        for file in root.iterdir():
            _add_to_zip(zf, file, root)

    if delete:
        sh.rmtree(root)


def unzip_directory(zipfile: str | os.PathLike, root: str | os.PathLike):
    r"""
    Unzip an archive to a directory.

    This function is the exact opposite of :py:func:`xscen.io.zip_directory`.

    Parameters
    ----------
    zipfile : path
        The zip file to read.
    root : path
        The directory where to put the content to archive.
        If it doesn't exist, it will be created (and all its parents).
        If it does exist, it should be empty.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    with ZipFile(zipfile, "r") as zf:
        zf.extractall(root)


# TODO: Implement as CF compression-by-gathering in cf-xarray instead
def save_sparse(
    da: xr.DataArray,
    filename: str | os.PathLike,
):
    """
    Utility to save a sparse array to disk.

    This is useful for saving :py:func:`~xscen.spatial.creep_weights` to disk, for example.
    Code inspired by xESMF. The output dataset preserves all coordinates, dimensions and attributes
    of the input dataarray, but in a syntax specific to xscen.

    Parameters
    ----------
    da : xr.DataArray
        Must be wrapping a :py:class:`sparse.COO` array.
    filename : pathlike
        Path of the netCDF to write.

    See Also
    --------
    load_sparse
    """
    arr = da.data
    ds = xr.Dataset(
        {
            "sp_values": ("nnz", arr.data),
            "sp_coords": (("sp_dims", "nnz"), arr.coords),
            "sp_names": (("sp_dims",), list(da.dims)),
            "sp_shape": (("sp_dims",), list(da.shape)),
        },
        coords=da.coords,
        attrs={"original_name": da.name or "weights", **da.attrs},
    )
    ds.to_netcdf(filename)


def load_sparse(filename: str | os.PathLike):
    """
    Load a sparse array that was saved with :py:func:`~xscen.io.save_sparse`.

    Parameters
    ----------
    filename: pathlike
        Path of the netCDF to read.

    Returns
    -------
    xr.DataArray: DataArray wrapping a sparse.COO array.
    """
    ds = xr.open_dataset(filename)
    return xr.DataArray(
        sp.COO(ds.sp_coords.values, ds.sp_values.values, ds.sp_shape.values),
        dims=ds.sp_names.values,
        coords=ds.coords,
        name=ds.attrs.pop("original_name"),
        attrs=ds.attrs,
    )
