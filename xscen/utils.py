"""Common utilities to be used in many places."""
import json
import logging
import os
import re
from collections.abc import Sequence
from datetime import datetime
from io import StringIO
from itertools import chain
from pathlib import Path
from types import ModuleType
from typing import Optional, TextIO, Union

import cftime
import flox.xarray
import numpy as np
import pandas as pd
import xarray as xr
from xclim.core import units
from xclim.core.calendar import convert_calendar, get_calendar, parse_offset
from xclim.core.utils import uses_dask
from xclim.testing.utils import show_versions as _show_versions

from .config import parse_config

logger = logging.getLogger(__name__)

__all__ = [
    "change_units",
    "clean_up",
    "date_parser",
    "get_cat_attrs",
    "maybe_unstack",
    "minimum_calendar",
    "natural_sort",
    "publish_release_notes",
    "stack_drop_nans",
    "standardize_periods",
    "translate_time_chunk",
    "unstack_fill_nan",
    "unstack_dates",
]


def date_parser(
    date,
    *,
    end_of_period: bool = False,
    out_dtype: str = "datetime",
    strtime_format: str = "%Y-%m-%d",
    freq: str = "H",
) -> Union[str, pd.Period, pd.Timestamp]:
    """Return a datetime from a string.

    Parameters
    ----------
    date : str, cftime.datetime, pd.Timestamp, datetime.datetime, pd.Period
        Date to be converted
    end_of_period : bool, str, optional
        If 'Y' or 'M', the returned date will be the end of the year or month that contains the received date.
        If True, the period is inferred from the date's precision, but `date` must be a string, otherwise nothing is done.
    out_dtype : str, optional
        Choices are 'datetime', 'period' or 'str'
    strtime_format : str, optional
        If out_dtype=='str', this sets the strftime format
    freq : str
        If out_dtype=='period', this sets the frequency of the period.

    Returns
    -------
    pd.Timestamp, pd.Period, str
        Parsed date
    """
    # Formats, ordered depending on string length
    fmts = {
        4: ["%Y"],
        6: ["%Y%m"],
        7: ["%Y-%m"],
        8: ["%Y%m%d"],
        10: ["%Y%m%d%H", "%Y-%m-%d"],
        12: ["%Y%m%d%H%M"],
        16: ["%Y-%m-%d %H:%M"],
        19: ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"],
    }

    def _parse_date(date, fmts):
        for fmt in fmts:
            try:
                # `pd.to_datetime` fails with out-of-bounds
                s = datetime.strptime(date, fmt)
            except ValueError:
                pass
            else:
                match = fmt
                break
        else:
            raise ValueError(f"Can't parse date {date} with formats {fmts}.")
        return s, match

    fmt = None
    # Timestamp can parse a few date formats by default, but not the ones without spaces
    # So we try a few known formats first, then a plain call
    # Also we need "fmt" to know the precision of the string (if end_of_period is True)
    if isinstance(date, str):
        try:
            date, fmt = _parse_date(date, fmts[len(date)])
        except (KeyError, ValueError):
            try:
                date = pd.Timestamp(date)
            except pd._libs.tslibs.parsing.DateParseError:
                date = pd.NaT
    elif isinstance(date, cftime.datetime):
        for n in range(3):
            try:
                date = pd.Timestamp((date - pd.Timedelta(n, "D")).isoformat())
            except ValueError:  # We are NOT catching OutOfBoundsDatetime.
                pass
            else:
                break
        else:
            raise ValueError(
                "Unable to parse cftime date {date}, even when moving back 2 days."
            )
    elif isinstance(date, pd.Period):
        # Pandas, you're a mess: Period.to_timestamp() fails for out-of-bounds dates (<1677, > 2242), but not when parsing a string...
        date = pd.Timestamp(date.strftime("%Y-%m-%dT%H:%M:%S"))

    if not isinstance(date, pd.Timestamp):
        date = pd.Timestamp(date)

    if isinstance(end_of_period, str) or (end_of_period is True and fmt):
        quasiday = (pd.Timedelta(1, "d") - pd.Timedelta(1, "s")).as_unit(date.unit)
        if end_of_period == "Y" or "m" not in fmt:
            date = (
                pd.tseries.frequencies.to_offset("A-DEC").rollforward(date) + quasiday
            )
        elif end_of_period == "M" or "d" not in fmt:
            date = pd.tseries.frequencies.to_offset("M").rollforward(date) + quasiday
        # TODO: Implement subdaily ?

    if out_dtype == "str":
        return date.strftime(strtime_format)

    if out_dtype == "period":
        return date.to_period(freq)
    return date


def minimum_calendar(*calendars) -> str:
    """Return the minimum calendar from a list.

    Uses the hierarchy: 360_day < noleap < standard < all_leap,
    and returns one of those names.
    """
    if "360_day" in calendars:
        return "360_day"

    if "noleap" in calendars or "365_day" in calendars:
        return "noleap"

    if all(cal in ["all_leap", "366_day"] for cal in calendars):
        return "all_leap"

    return "standard"


def translate_time_chunk(chunks: dict, calendar: str, timesize) -> dict:
    """Translate chunk specification for time into a number.

    -1 translates to `timesize`
    'Nyear' translates to N times the number of days in a year of calendar `calendar`.
    """
    for k, v in chunks.items():
        if isinstance(v, dict):
            chunks[k] = translate_time_chunk(v.copy(), calendar, timesize)
        elif k == "time" and v is not None:
            if isinstance(v, str) and v.endswith("year"):
                n = int(chunks["time"].split("year")[0])
                Nt = n * {"noleap": 365, "360_day": 360, "all_leap": 366}.get(
                    calendar, 365.25
                )
                chunks[k] = int(Nt)
            elif v == -1:
                chunks[k] = timesize
    return chunks


@parse_config
def stack_drop_nans(
    ds: xr.Dataset,
    mask: xr.DataArray,
    *,
    new_dim: str = "loc",
    to_file: Optional[str] = None,
) -> xr.Dataset:
    """Stack dimensions into a single axis and drops indexes where the mask is false.

    Parameters
    ----------
    ds : xr.Dataset
      A dataset with the same coords as `mask`.
    mask : xr.DataArray
      A boolean DataArray with True on the points to keep.
      Mask will be loaded within this function.
    new_dim : str
      The name of the new stacked dim.
    to_file : str, optional
      A netCDF filename where to write the stacked coords for use in `unstack_fill_nan`.
      If given a string with {shape} and {domain}, the formatting will fill them with
      the original shape of the dataset and the global attributes 'cat:domain'.
      If None (default), nothing is written to disk.
      It is recommended to fill this argument in the config. It will be parsed automatically.
      E.g.:

          utils:
            stack_drop_nans:
                to_file: /some_path/coords/coords_{domain}_{shape}.nc
            unstack_fill_nan:
                coords: /some_path/coords/coords_{domain}_{shape}.nc

    Returns
    -------
    xr.Dataset
      Same as `ds`, but all dimensions of mask have been stacked to a single `new_dim`.
      Indexes where mask is False have been dropped.

    See Also
    --------
    unstack_fill_nan : The inverse operation.
    """
    original_shape = "x".join(map(str, mask.shape))

    mask_1d = mask.stack({new_dim: mask.dims})
    out = ds.stack({new_dim: mask.dims}).where(mask_1d, drop=True).reset_index(new_dim)
    for dim in mask.dims:
        out[dim].attrs.update(ds[dim].attrs)

    if to_file is not None:
        # set default path to store the information necessary to unstack
        # the name includes the domain and the original shape to uniquely identify the dataset
        domain = ds.attrs.get("cat:domain", "unknown")
        to_file = to_file.format(domain=domain, shape=original_shape)
        if not Path(to_file).parent.exists():
            os.makedirs(Path(to_file).parent, exist_ok=True)
        mask.coords.to_dataset().to_netcdf(to_file)

    # carry information about original shape to be able to unstack properly
    for dim in mask.dims:
        out[dim].attrs["original_shape"] = original_shape

        # this is needed to fix a bug in xarray '2022.6.0'
        out[dim] = xr.DataArray(
            out[dim].values,
            dims=out[dim].dims,
            coords=out[dim].coords,
            attrs=out[dim].attrs,
        )

    return out


@parse_config
def unstack_fill_nan(
    ds: xr.Dataset, *, dim: str = "loc", coords: Optional[Sequence[str]] = None
):
    """Unstack a Dataset that was stacked by :py:func:`stack_drop_nans`.

    Parameters
    ----------
    ds : xr.Dataset
      A dataset with some dims stacked by `stack_drop_nans`.
    dim : str
      The dimension to unstack, same as `new_dim` in `stack_drop_nans`.
    coords : Sequence of strings, Mapping of str to array, str, optional
      If a sequence : if the dataset has coords along `dim` that are not original
      dimensions, those original dimensions must be listed here.
      If a dict : a mapping from the name to the array of the coords to unstack
      If a str : a filename to a dataset containing only those coords (as coords).
      If given a string with {shape} and {domain}, the formatting will fill them with
      the original shape of the dataset (that should have been store in the
      attributes of the stacked dimensions) by `stack_drop_nans` and the global attributes 'cat:domain'.
      It is recommended to fill this argument in the config. It will be parsed automatically.
      E.g.:

          utils:
            stack_drop_nans:
                to_file: /some_path/coords/coords_{domain}_{shape}.nc
            unstack_fill_nan:
                coords: /some_path/coords/coords_{domain}_{shape}.nc

      If None (default), all coords that have `dim` a single dimension are used as the
      new dimensions/coords in the unstacked output.
      Coordinates will be loaded within this function.

    Returns
    -------
    xr.Dataset
      Same as `ds`, but `dim` has been unstacked to coordinates in `coords`.
      Missing elements are filled according to the defaults of `fill_value` of :py:meth:`xarray.Dataset.unstack`.
    """
    if coords is None:
        logger.info("Dataset unstacked using no coords argument.")

    if isinstance(coords, (str, os.PathLike)):
        # find original shape in the attrs of one of the dimension
        original_shape = "unknown"
        for c in ds.coords:
            if "original_shape" in ds[c].attrs:
                original_shape = ds[c].attrs["original_shape"]
        domain = ds.attrs.get("cat:domain", "unknown")
        coords = coords.format(domain=domain, shape=original_shape)
        logger.info(f"Dataset unstacked using {coords}.")
        coords = xr.open_dataset(coords)
        # separate coords that are dims or not
        coords_and_dims = {
            name: x for name, x in coords.coords.items() if name in coords.dims
        }
        coords_not_dims = {
            name: x for name, x in coords.coords.items() if name not in coords.dims
        }

        dims, crds = zip(
            *[
                (name, crd.load().values)
                for name, crd in ds.coords.items()
                if crd.dims == (dim,) and name in coords_and_dims
            ]
        )
        out = (
            ds.drop_vars(dims)
            .assign_coords({dim: pd.MultiIndex.from_arrays(crds, names=dims)})
            .unstack(dim)
        )

        # only reindex with the dims
        out = out.reindex(**coords_and_dims)
        # add back the coords that arent dims
        for c in coords_not_dims:
            out[c] = coords[c]
    else:
        if isinstance(coords, (list, tuple)):
            dims, crds = zip(*[(name, ds[name].load().values) for name in coords])
        else:
            dims, crds = zip(
                *[
                    (name, crd.load().values)
                    for name, crd in ds.coords.items()
                    if crd.dims == (dim,)
                ]
            )

        out = (
            ds.drop_vars(dims)
            .assign_coords({dim: pd.MultiIndex.from_arrays(crds, names=dims)})
            .unstack(dim)
        )

        if not isinstance(coords, (list, tuple)) and coords is not None:
            out = out.reindex(**coords.coords)

    for dim in dims:
        out[dim].attrs.update(ds[dim].attrs)

    return out


def natural_sort(_list: list):
    """
    For strings of numbers. alternative to sorted() that detects a more natural order.

    e.g. [r3i1p1, r1i1p1, r10i1p1] is sorted as [r1i1p1, r3i1p1, r10i1p1] instead of [r10i1p1, r1i1p1, r3i1p1]
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()  # noqa: E731
    alphanum_key = lambda key: [  # noqa: E731
        convert(c) for c in re.split("([0-9]+)", key)
    ]
    return sorted(_list, key=alphanum_key)


def get_cat_attrs(
    ds: Union[xr.Dataset, dict], prefix: str = "cat:", var_as_str=False
) -> dict:
    """Return the catalog-specific attributes from a dataset or dictionary.

    Parameters
    ----------
    ds: xr.Dataset
        Dataset to be parsed.
    prefix: str
        Prefix automatically generated by intake-esm. With xscen, this should be 'cat:'
    var_as_str: bool
        If True, variable will be returned as a string if there is only one.

    Returns
    -------
    dict
        Compilation of all attributes in a dictionary.

    """
    if isinstance(ds, (xr.Dataset, xr.DataArray)):
        attrs = ds.attrs
    else:
        attrs = ds
    facets = {
        k[len(prefix) :]: v for k, v in attrs.items() if k.startswith(f"{prefix}")
    }

    # to be usable in a path
    if (
        var_as_str
        and "variable" in facets
        and not isinstance(facets["variable"], str)
        and len(facets["variable"]) == 1
    ):
        facets["variable"] = facets["variable"][0]
    return facets


@parse_config
def maybe_unstack(
    ds: xr.Dataset,
    coords: str = None,
    rechunk: bool = None,
    stack_drop_nans: bool = False,
):
    """If stack_drop_nans is True, unstack and rechunk."""
    if stack_drop_nans:
        ds = unstack_fill_nan(ds, coords=coords)
        if rechunk is not None:
            ds = ds.chunk(rechunk)
    return ds


# Read CVs and fill a virtual module
CV = ModuleType(
    "CV",
    (
        """
        Mappings of (controlled) vocabulary. This module is generated automatically
        from json files in xscen/CVs. Functions are essentially mappings, most of
        which are meant to provide translations between columns.\n\n
        Json files must be shallow dictionaries to be supported. If the json file
        contains a ``is_regex: True`` entry, then the keys are automatically
        translated as regex patterns and the function returns the value of the first
        key that matches the pattern. Otherwise the function essentially acts like a
        normal dictionary. The 'raw' data parsed from the json file is added in the
        ``dict`` attribute of the function.
        Example:

        .. code-block:: python

            xs.utils.CV.frequency_to_timedelta.dict

        .. literalinclude:: ../xscen/CVs/frequency_to_timedelta.json
           :language: json
           :caption: frequency_to_timedelta

        .. literalinclude:: ../xscen/CVs/frequency_to_xrfreq.json
           :language: json
           :caption: frequency_to_xrfreq

        .. literalinclude:: ../xscen/CVs/infer_resolution.json
           :language: json
           :caption: infer_resolution

        .. literalinclude:: ../xscen/CVs/resampling_methods.json
           :language: json
           :caption: resampling_methods

        .. literalinclude:: ../xscen/CVs/variable_names.json
           :language: json
           :caption: variable_names

        .. literalinclude:: ../xscen/CVs/xrfreq_to_frequency.json
           :language: json
           :caption: xrfreq_to_frequency

        .. literalinclude:: ../xscen/CVs/xrfreq_to_timedelta.json
           :language: json
           :caption: xrfreq_to_timedelta


        """
    ),
)


def __read_CVs(cvfile):
    with cvfile.open("r") as f:
        cv = json.load(f)
    is_regex = cv.pop("is_regex", False)
    doc = """Controlled vocabulary mapping from {name}.

    The raw dictionary can be accessed by the dict attribute of this function.

    Parameters
    ----------
    key: str
      The value to translate.{regex}
    default : 'pass', 'error' or Any
      If the key is not found in the mapping, default controls the behaviour.

      - "error", a KeyError is raised (default).
      - "pass", the key is returned.
      - another value, that value is returned.
"""

    def cvfunc(key, default="error"):
        if is_regex:
            for cin, cout in cv.items():
                try:
                    if re.fullmatch(cin, key):
                        return cout
                except TypeError:
                    pass
        else:
            if key in cv:
                return cv[key]
        if isinstance(default, str):
            if default == "pass":
                return key
            if default == "error":
                raise KeyError(key)
        return default

    cvfunc.__name__ = cvfile.stem
    cvfunc.__doc__ = doc.format(
        name=cvfile.stem.replace("_", " "),
        regex=" The key will be matched using regex" if is_regex else "",
    )
    cvfunc.__dict__["dict"] = cv
    cvfunc.__module__ = "xscen.CV"
    return cvfunc


for cvfile in (Path(__file__).parent / "CVs").glob("*.json"):
    try:
        CV.__dict__[cvfile.stem] = __read_CVs(cvfile)
    except Exception as err:
        raise ValueError(f"While reading {cvfile} got {err}")


def change_units(ds: xr.Dataset, variables_and_units: dict) -> xr.Dataset:
    """Change units of Datasets to non-CF units.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to use
    variables_and_units : dict
        Description of the variables and units to output

    Returns
    -------
    xr.Dataset

    See Also
    --------
    xclim.core.units.convert_units_to, xclim.core.units.rate2amount
    """
    with xr.set_options(keep_attrs=True):
        for v in variables_and_units:
            if (v in ds) and (
                units.units2pint(ds[v]) != units.units2pint(variables_and_units[v])
            ):
                time_in_ds = units.units2pint(ds[v]).dimensionality.get("[time]")
                time_in_out = units.units2pint(
                    variables_and_units[v]
                ).dimensionality.get("[time]")

                if time_in_ds == time_in_out:
                    ds[v] = units.convert_units_to(ds[v], variables_and_units[v])
                elif time_in_ds - time_in_out == 1:
                    # ds is an amount
                    ds[v] = units.amount2rate(ds[v], out_units=variables_and_units[v])
                elif time_in_ds - time_in_out == -1:
                    # ds is a rate
                    ds[v] = units.rate2amount(ds[v], out_units=variables_and_units[v])
                else:
                    raise NotImplementedError(
                        f"No known transformation between {ds[v].units} and {variables_and_units[v]} (temporal dimensionality mismatch)."
                    )

    return ds


def clean_up(
    ds: xr.Dataset,
    *,
    variables_and_units: Optional[dict] = None,
    convert_calendar_kwargs: Optional[dict] = None,
    missing_by_var: Optional[dict] = None,
    maybe_unstack_dict: Optional[dict] = None,
    round_var: Optional[dict] = None,
    common_attrs_only: Union[dict, list] = None,
    common_attrs_open_kwargs: dict = None,
    attrs_to_remove: Optional[dict] = None,
    remove_all_attrs_except: Optional[dict] = None,
    add_attrs: Optional[dict] = None,
    change_attr_prefix: Optional[str] = None,
    to_level: Optional[str] = None,
):
    """Clean up of the dataset.

    It can:
     - convert to the right units using xscen.finalize.change_units
     - convert the calendar and interpolate over missing dates
     - call the xscen.common.maybe_unstack function
     - remove a list of attributes
     - remove everything but a list of attributes
     - add attributes
     - change the prefix of the catalog attrs

    in that order.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset to clean up
    variables_and_units : dict
        Dictionary of variable to convert. eg. {'tasmax': 'degC', 'pr': 'mm d-1'}
    convert_calendar_kwargs : dict
        Dictionary of arguments to feed to xclim.core.calendar.convert_calendar. This will be the same for all variables.
        If missing_by_vars is given, it will override the 'missing' argument given here.
        Eg. {target': default, 'align_on': 'random'}
    missing_by_var : list
        Dictionary where the keys are the variables and the values are the argument to feed the `missing`
        parameters of the xclim.core.calendar.convert_calendar for the given variable with the `convert_calendar_kwargs`.
        If missing_by_var == 'interpolate', the missing will be filled with NaNs, then linearly interpolated over time.
    maybe_unstack_dict : dict
        Dictionary to pass to xscen.common.maybe_unstack function.
        The format should be: {'coords': path_to_coord_file, 'rechunk': {'time': -1 }, 'stack_drop_nans': True}.
    round_var : dict
        Dictionary where the keys are the variables of the dataset and the values are the number of decimal places to round to
    common_attrs_only : dict, list
        Dictionnary of datasets or list of datasets, or path to NetCDF or Zarr files.
        Keeps only the global attributes that are the same for all datasets and generates a new id.
    common_attrs_open_kwargs : dict
        Dictionary of arguments for xarray.open_dataset(). Used with common_attrs_only if given paths.
    attrs_to_remove : dict
        Dictionary where the keys are the variables and the values are a list of the attrs that should be removed.
        For global attrs, use the key 'global'.
        The element of the list can be exact matches for the attributes name
        or use the same substring matching rules as intake_esm:
        - ending with a '*' means checks if the substring is contained in the string
        - starting with a '^' means check if the string starts with the substring.
        eg. {'global': ['unnecessary note', 'cell*'], 'tasmax': 'old_name'}
    remove_all_attrs_except : dict
        Dictionary where the keys are the variables and the values are a list of the attrs that should NOT be removed,
        all other attributes will be deleted. If None (default), nothing will be deleted.
        For global attrs, use the key 'global'.
        The element of the list can be exact matches for the attributes name
        or use the same substring matching rules as intake_esm:
        - ending with a '*' means checks if the substring is contained in the string
        - starting with a '^' means check if the string starts with the substring.
        eg. {'global': ['necessary note', '^cat:'], 'tasmax': 'new_name'}
    add_attrs : dict
        Dictionary where the keys are the variables and the values are a another dictionary of attributes.
        For global attrs, use the key 'global'.
        eg. {'global': {'title': 'amazing new dataset'}, 'tasmax': {'note': 'important info about tasmax'}}
    change_attr_prefix : str
        Replace "cat:" in the catalog global attrs by this new string
    to_level : str
        The processing level to assign to the output.

    Returns
    -------
    xr.Dataset
        Cleaned up dataset

    See Also
    --------
    xclim.core.calendar.convert_calendar
    """
    if variables_and_units:
        logger.info(f"Converting units: {variables_and_units}")
        ds = change_units(ds=ds, variables_and_units=variables_and_units)

    # convert calendar
    if convert_calendar_kwargs:
        ds_copy = ds.copy()
        # create mask of grid point that should always be nan
        ocean = ds_copy.isnull().all("time")

        # if missing_by_var exist make sure missing data are added to time axis
        if missing_by_var:
            if not all(k in missing_by_var.keys() for k in ds.data_vars):
                raise ValueError(
                    "All variables must be in 'missing_by_var' if using this option."
                )
            convert_calendar_kwargs["missing"] = -9999

        # make default `align_on`='`random` when the initial calendar is 360day
        if get_calendar(ds) == "360_day" and "align_on" not in convert_calendar_kwargs:
            convert_calendar_kwargs["align_on"] = "random"

        logger.info(f"Converting calendar with {convert_calendar_kwargs} ")
        ds = convert_calendar(ds, **convert_calendar_kwargs).where(~ocean)

        # convert each variable individually
        if missing_by_var:
            # remove 'missing' argument to be replace by `missing_by_var`
            del convert_calendar_kwargs["missing"]
            for var, missing in missing_by_var.items():
                logging.info(f"Filling missing {var} with {missing}")
                if missing == "interpolate":
                    ds_with_nan = ds[var].where(ds[var] != -9999)
                    converted_var = ds_with_nan.interpolate_na("time", method="linear")
                else:
                    var_attrs = ds[var].attrs
                    converted_var = xr.where(ds[var] == -9999, missing, ds[var])
                    converted_var.attrs = var_attrs
                ds[var] = converted_var

    # unstack nans
    if maybe_unstack_dict:
        ds = maybe_unstack(ds, **maybe_unstack_dict)

    if round_var:
        for var, n in round_var.items():
            ds[var] = ds[var].round(n)

    def _search(a, b):
        if a[-1] == "*":  # check if a is contained in b
            return a[:-1] in b
        elif a[0] == "^":
            return b.startswith(a[1:])
        else:
            return a == b

    if common_attrs_only:
        from .catalog import generate_id

        common_attrs_open_kwargs = common_attrs_open_kwargs or {}
        if isinstance(common_attrs_only, dict):
            common_attrs_only = list(common_attrs_only.values())

        for i in range(len(common_attrs_only)):
            if isinstance(common_attrs_only[i], (str, Path)):
                dataset = xr.open_dataset(
                    common_attrs_only[i], **common_attrs_open_kwargs
                )
            else:
                dataset = common_attrs_only[i]
            attributes = ds.attrs.copy()
            for a_key, a_val in attributes.items():
                if (
                    (a_key not in dataset.attrs)
                    or (a_key in ["cat:date_start", "cat:date_end"])
                    or (a_val != dataset.attrs[a_key])
                ):
                    del ds.attrs[a_key]

        # generate a new id
        try:
            ds.attrs["cat:id"] = generate_id(ds).iloc[0]
        except IndexError as err:
            logger.warning(f"Unable to generate a new id for the dataset. Got {err}.")
            if "cat:id" in ds.attrs:
                del ds.attrs["cat:id"]

    if to_level:
        ds.attrs["cat:processing_level"] = to_level

    # remove attrs
    if attrs_to_remove:
        for var, list_of_attrs in attrs_to_remove.items():
            obj = ds if var == "global" else ds[var]
            for ds_attr in list(obj.attrs.keys()):  # iter over attrs in ds
                for list_attr in list_of_attrs:  # check if we want to remove attrs
                    if _search(list_attr, ds_attr):
                        del obj.attrs[ds_attr]

    # delete all attrs, but the ones in the list
    if remove_all_attrs_except:
        for var, list_of_attrs in remove_all_attrs_except.items():
            obj = ds if var == "global" else ds[var]
            for ds_attr in list(obj.attrs.keys()):  # iter over attrs in ds
                delete = True  # assume we should delete it
                for list_attr in list_of_attrs:
                    if _search(list_attr, ds_attr):
                        delete = (
                            False  # if attr is on the list to not delete, don't delete
                        )
                if delete:
                    del obj.attrs[ds_attr]

    if add_attrs:
        for var, attrs in add_attrs.items():
            obj = ds if var == "global" else ds[var]
            for attrname, attrtmpl in attrs.items():
                obj.attrs[attrname] = attrtmpl

    if change_attr_prefix:
        for ds_attr in list(ds.attrs.keys()):
            new_name = ds_attr.replace("cat:", change_attr_prefix)
            if new_name:
                ds.attrs[new_name] = ds.attrs.pop(ds_attr)

    return ds


def publish_release_notes(
    style: str = "md", file: Optional[Union[os.PathLike, StringIO, TextIO]] = None
) -> Optional[str]:
    """Format release history in Markdown or ReStructuredText.

    Parameters
    ----------
    style: {"rst", "md"}
      Use ReStructuredText formatting or Markdown. Default: Markdown.
    file: {os.PathLike, StringIO, TextIO}, optional
      If provided, prints to the given file-like object. Otherwise, returns a string.

    Returns
    -------
    str, optional

    Notes
    -----
    This function exists solely for development purposes.
    Adapted from xclim.testing.utils.publish_release_notes.
    """
    history_file = Path(__file__).parent.parent.joinpath("HISTORY.rst")

    if not history_file.exists():
        raise FileNotFoundError("History file not found in xscen file tree.")

    with open(history_file) as hf:
        history = hf.read()

    if style == "rst":
        hyperlink_replacements = {
            r":issue:`([0-9]+)`": r"`GH/\1 <https://github.com/Ouranosinc/xscen/issues/\1>`_",
            r":pull:`([0-9]+)`": r"`PR/\1 <https://github.com/Ouranosinc/xscen/pull/\>`_",
            r":user:`([a-zA-Z0-9_.-]+)`": r"`@\1 <https://github.com/\1>`_",
        }
    elif style == "md":
        hyperlink_replacements = {
            r":issue:`([0-9]+)`": r"[GH/\1](https://github.com/Ouranosinc/xscen/issues/\1)",
            r":pull:`([0-9]+)`": r"[PR/\1](https://github.com/Ouranosinc/xscen/pull/\1)",
            r":user:`([a-zA-Z0-9_.-]+)`": r"[@\1](https://github.com/\1)",
        }
    else:
        raise NotImplementedError()

    for search, replacement in hyperlink_replacements.items():
        history = re.sub(search, replacement, history)

    if style == "md":
        history = history.replace("=======\nHistory\n=======", "# History")

        titles = {r"\n(.*?)\n([\-]{1,})": "-", r"\n(.*?)\n([\^]{1,})": "^"}
        for title_expression, level in titles.items():
            found = re.findall(title_expression, history)
            for grouping in found:
                fixed_grouping = (
                    str(grouping[0]).replace("(", r"\(").replace(")", r"\)")
                )
                search = rf"({fixed_grouping})\n([\{level}]{'{' + str(len(grouping[1])) + '}'})"
                replacement = f"{'##' if level=='-' else '###'} {grouping[0]}"
                history = re.sub(search, replacement, history)

        link_expressions = r"[\`]{1}([\w\s]+)\s<(.+)>`\_"
        found = re.findall(link_expressions, history)
        for grouping in found:
            search = rf"`{grouping[0]} <.+>`\_"
            replacement = f"[{str(grouping[0]).strip()}]({grouping[1]})"
            history = re.sub(search, replacement, history)

    if not file:
        return history
    print(history, file=file)


def unstack_dates(
    ds: xr.Dataset,
    seasons: dict[int, str] = None,
    new_dim: str = "season",
    winter_starts_year: bool = False,
):
    """Unstack a multi-season timeseries into a yearly axis and a season one.

    Parameters
    ----------
    ds: xr.Dataset or DataArray
      The xarray object with a "time" coordinate.
      Only supports monthly or coarser frequencies.
      The time axis must be complete and regular (`xr.infer_freq(ds.time)` doesn't fail).
    seasons: dict, optional
      A dictionary from month number to a season name.
      If not given, it is guessed from the time coord's frequency.
      See notes.
    new_dim: str
      The name of the new dimension.
    winter_starts_year: bool
      If True, the winter season (DJF) is associated with the year of January, instead of December.

    Returns
    -------
    xr.Dataset or DataArray
      Same as ds but the time axis is now yearly (AS-JAN) and the seasons are along the new dimension.

    Notes
    -----
    When `season` is None, the inferred frequency determines the new coordinate:

    - For MS, the coordinates are the month abbreviations in english (JAN, FEB, etc.)
    - For ?QS-? and other ?MS frequencies, the coordinates are the initials of the months in each season.
      Ex: QS-DEC (with winter_starts_year=True) : DJF, MAM, JJA, SON.
    - For YS or AS-JAN, the new coordinate has a single value of "annual".
    - For ?AS-? frequencies, the new coordinate has a single value of "annual-{anchor}", were "anchor"
      is the abbreviation of the first month of the year. Ex: AS-JUL -> "annual-JUL".
    """
    # Get some info about the time axis
    freq = xr.infer_freq(ds.time)
    if freq is None:
        raise ValueError(
            "The data must have a clean time coordinate. If you know the "
            "data's frequency, please pass `ds.resample(time=freq).first()` "
            "to pad missing dates and reset the time coordinate."
        )
    first, last = ds.indexes["time"][[0, -1]]
    use_cftime = xr.coding.times.contains_cftime_datetimes(ds.time.variable)
    calendar = ds.time.dt.calendar
    mult, base, isstart, anchor = parse_offset(freq)

    if base not in "YAQM":
        raise ValueError(
            f"Only monthly frequencies or coarser are supported. Got: {freq}."
        )

    # Fast track for annual
    if base == "A":
        if seasons:
            seaname = seasons[first.month]
        elif anchor == "JAN":
            seaname = "annual"
        else:
            seaname = f"annual-{anchor}"
        dso = ds.expand_dims({new_dim: [seaname]})
        dso["time"] = xr.date_range(
            f"{first.year}-01-01",
            f"{last.year}-01-01",
            freq="YS",
            calendar=calendar,
            use_cftime=use_cftime,
        )
        return dso

    if base == "M" and 12 % mult != 0:
        raise ValueError(
            f"Only periods that divide the year evenly are supported. Got {freq}."
        )

    # Guess the new season coordinate
    if seasons is None:
        if base == "Q" or (base == "M" and mult > 1):
            # Labels are the month initials
            months = np.array(list("JFMAMJJASOND"))
            n = mult * {"M": 1, "Q": 3}[base]
            seasons = {
                m: "".join(months[np.array(range(m - 1, m + n - 1)) % 12])
                for m in np.unique(ds.time.dt.month)
            }
        else:  # M or MS
            seasons = xr.coding.cftime_offsets._MONTH_ABBREVIATIONS
    # The ordered season names
    seas_list = [seasons[month] for month in sorted(seasons.keys())]

    # Multi-month seasons that isn't synced with january
    if winter_starts_year and 1 not in seasons:
        # The last label is the period that overlaps the year
        winter_month = max(seasons.keys())
        # Put it back in the beginning
        seas_list = [seasons[winter_month]] + seas_list[:-1]
        # The year associated with each timestamp (add 1 in winter)
        years = ds.time.dt.year + xr.where(ds.time.dt.month == winter_month, 1, 0)
    else:  # Monthly or aligned seasons
        years = ds.time.dt.year

    # The goal here is to use `reshape()` instead of `unstack` to limit the number of dask operations.
    # Thus, the time axis must be properly constructed so that reshapes fits the final size.
    # We pad on both sides to ensure full years
    pad_left = seas_list.index(seasons[first.month])
    pad_right = len(seas_list) - (seas_list.index(seasons[last.month]) + 1)
    dsp = ds.pad(time=(pad_left, pad_right))  # pad with NaN
    # Similarly pad our "group labels".
    years = years.pad(time=(pad_left, pad_right), constant_values=(years[0], years[-1]))

    # New coords
    new_time = xr.date_range(  # New time axis (YS)
        f"{years[0].item()}-01-01",
        f"{years[-1].item()}-01-01",
        freq="YS",
        calendar=calendar,
        use_cftime=use_cftime,
    )

    def reshape_da(da):
        if "time" not in da.dims:
            return da
        # Replace (A,'time',B) by (A,'time', 'season',B) in both the new shape and the new dims
        new_dims = list(
            chain.from_iterable(
                [d] if d != "time" else ["time", new_dim] for d in da.dims
            )
        )
        new_shape = [len(new_coords[d]) for d in new_dims]
        # Use dask or numpy's algo.

        if uses_dask(da):
            # This is where it happens. Flox will minimally rechunk
            # so the reshape operation can be performed blockwise
            da = flox.xarray.rechunk_for_blockwise(da, "time", years)
        return xr.DataArray(da.data.reshape(new_shape), dims=new_dims)

    new_coords = dict(ds.coords)
    new_coords.update({"time": new_time, new_dim: seas_list})

    # put horizon in the right time dimension
    if "horizon" in new_coords:
        new_coords["horizon"] = reshape_da(new_coords["horizon"])

    if isinstance(ds, xr.Dataset):
        dso = dsp.map(reshape_da, keep_attrs=True)
    else:
        dso = reshape_da(dsp)
    return dso.assign_coords(**new_coords)


def show_versions(
    file: Optional[Union[os.PathLike, StringIO, TextIO]] = None,
    deps: Optional[list] = None,
) -> Optional[str]:
    """Print the versions of xscen and its dependencies.

    Parameters
    ----------
    file : {os.PathLike, StringIO, TextIO}, optional
        If provided, prints to the given file-like object. Otherwise, returns a string.
    deps : list, optional
        A list of dependencies to gather and print version information from. Otherwise, prints `xscen` dependencies.

    Returns
    -------
    str or None
    """
    if deps is None:
        deps = [
            "zarr",
            "yaml",
            "xesmf",
            "xcollection",
            "xclim",
            "xarray",
            "sparse",
            "shapely",
            "requests",
            "rechunker",
            "pydantic",
            "pyarrow",
            "parse",
            "pandas",
            "netCDF4",
            "nc_time_axis",
            "matplotlib",
            "intake_esm",
            "intake",
            "h5py",
            "h5netcdf",
            "geopandas",
            "fsspec",
            "fastprogress",
            "dask",
            "clisops",
            "cftime",
            "cartopy",
            "xscen",
        ]

    return _show_versions(file=file, deps=deps)


def ensure_correct_time(ds: xr.Dataset, xrfreq: str) -> xr.Dataset:
    """
    Ensure a dataset has the correct time coordinate, as expected for the given frequency.

    Daily or finer datasets are "floored" even if `xr.infer_freq` succeeds.
    Errors are raised if the number of data points per period is not 1.
    The dataset is modified in-place, but returned nonetheless.
    """
    # Check if we got the expected freq (skip for too short timeseries)
    inffreq = xr.infer_freq(ds.time) if ds.time.size > 2 else None
    if inffreq == xrfreq:
        # Even when the freq is correct, we ensure the correct "anchor" for daily and finer
        if xrfreq in "DHTMUL":
            ds["time"] = ds.time.dt.floor(xrfreq)
    else:
        # We can't infer it, there might be a problem
        counts = ds.time.resample(time=xrfreq).count()
        if (counts > 1).any().item():
            raise ValueError(
                "Dataset is labelled as having a sampling frequency of "
                f"{xrfreq}, but some periods have more than one data point."
            )
        if (counts.isnull()).any().item():
            raise ValueError(
                "The resampling count contains nans. There might be some missing data."
            )
        ds["time"] = counts.time
    return ds


def standardize_periods(periods, multiple=True):
    """Reformats 'periods' to a list of strings [['start', 'end'], ['start', 'end']]."""
    if periods is None:
        return periods

    if not isinstance(periods[0], list):
        periods = [periods]

    for i in range(len(periods)):
        if len(periods[i]) != 2:
            raise ValueError(
                "Each instance of 'periods' should be comprised of two elements: [start, end]."
            )
        if int(periods[i][0]) > int(periods[i][1]):
            raise ValueError(
                f"'periods' should be in chronological order, received {periods[i]}."
            )
        periods[i] = [str(p) for p in periods[i]]

    if multiple:
        return periods
    else:
        if len(periods) > 1:
            raise ValueError(
                f"'period' should be a single instance of [start, end], received {len(periods)}."
            )
        return periods[0]
