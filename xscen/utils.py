"""
Common things to be used at many places
"""
import json
import logging
import os
import re
from pathlib import Path
from types import ModuleType
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import xarray as xr
from xclim.core import units
from xclim.core.calendar import convert_calendar, get_calendar

from .config import parse_config

logger = logging.getLogger(__name__)

__all__ = [
    "maybe_unstack",
    "minimum_calendar",
    "natural_sort",
    "stack_drop_nans",
    "translate_time_chunk",
    "unstack_fill_nan",
]


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
      If None (default), nothing is written to disk.

    Returns
    -------
    xr.Dataset
      Same as `ds`, but all dimensions of mask have been stacked to a single `new_dim`.
      Indexes where mask is False have been dropped.

    See also
    --------
    unstack_fill_nan : The inverse operation.
    """
    mask_1d = mask.stack({new_dim: mask.dims})
    out = ds.stack({new_dim: mask.dims}).where(mask_1d, drop=True).reset_index(new_dim)
    for dim in mask.dims:
        out[dim].attrs.update(ds[dim].attrs)

    if to_file is not None:
        mask.coords.to_dataset().to_netcdf(to_file)
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
      If None (default), all coords that have `dim` a single dimension are used as the
      new dimensions/coords in the unstacked output.
      Coordinates will be loaded within this function.

    Returns
    -------
    xr.Dataset
      Same as `ds`, but `dim` has been unstacked to coordinates in `coords`.
      Missing elements are filled according to the defaults of `fill_value` of :py:meth:`xarray.Dataset.unstack`.
    """
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
        if isinstance(coords, (str, os.PathLike)):
            coords = xr.open_dataset(coords)
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


def maybe_unstack(
    ds: xr.Dataset, coords, rechunk: bool = None, stack_drop_nans: bool = False
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
        "Mappings of (controlled) vocabulary. This module is generated automatically "
        "from json files in xscen/CVs. Functions are essentially mappings, most of "
        "which are meant to provide translations between columns.\n\n"
        "Json files must be shallow dictionaries to be supported. If the json file "
        "contains a ``is_regex: True`` entry, then the keys are automatically "
        "translated as regex patterns and the function returns the value of the first "
        "key that matches the pattern. Otherwise the function essentially acts like a "
        "normal dictionary. The 'raw' data parsed from the json file is added in the "
        "``dict`` attribute of the function."
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


@parse_config
def change_units(ds: xr.Dataset, variables_and_units: dict) -> xr.Dataset:
    """Changes units of Datasets to non-CF units.

    Parameters
    ----------
    ds : xr.Dataset
      Dataset to use
    variables_and_units : dict
      Description of the variables and units to output

    Returns
    -------
    xr.Dataset
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
    variables_and_units: Optional[dict] = None,
    convert_calendar_kwargs: Optional[dict] = None,
    missing_by_var: Optional[dict] = None,
    maybe_unstack_dict: Optional[dict] = None,
    attrs_to_remove: Optional[dict] = None,
    remove_all_attrs_except: Optional[dict] = None,
    add_attrs: Optional[dict] = None,
    change_attr_prefix: Optional[str] = None,
    to_level: Optional[str] = "cleanedup",
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
    ds: xr.Dataset
        Input dataset to clean up
    variables_and_units: dict
        Dictionary of variable to convert. eg. {'tasmax': 'degC', 'pr': 'mm d-1'}
    convert_calendar_kwargs: dict
        Dictionary of arguments to feed to xclim.core.calendar.convert_calendar. This will be the same for all variables.
        If missing_by_vars is given, it will override the 'missing' argument given here.
        Eg. {target': default, 'align_on': 'random'}
    missing_by_var: list
        Dictionary where the keys are the variables and the values are the argument to feed the `missing`
        parameters of the xclim.core.calendar.convert_calendar for the given variable with the `convert_calendar_kwargs`.
        If missing_by_var == 'interpolate', the missing will be filled with NaNs, then linearly interpolated over time.
    maybe_unstack_dict: dict
        Dictionary to pass to xscen.common.maybe_unstack function.
        The format should be: {'coords': path_to_coord_file, 'rechunk': {'time': -1 }, 'stack_drop_nans': True}.
    attrs_to_remove: dict
        Dictionary where the keys are the variables and the values are a list of the attrs that should be removed.
        For global attrs, use the key 'global'.
        The element of the list can be exact matches for the attributes name
        or use the same substring matching rules as intake_esm:
        - ending with a '*' means checks if the substring is contained in the string
        - starting with a '^' means check if the string starts with the substring.
        eg. {'global': ['unnecessary note', 'cell*'], 'tasmax': 'old_name'}
    remove_all_attrs_except: dict
        Dictionary where the keys are the variables and the values are a list of the attrs that should NOT be removed,
        all other attributes will be deleted. If None (default), nothing will be deleted.
        For global attrs, use the key 'global'.
        The element of the list can be exact matches for the attributes name
        or use the same substring matching rules as intake_esm:
        - ending with a '*' means checks if the substring is contained in the string
        - starting with a '^' means check if the string starts with the substring.
        eg. {'global': ['necessary note', '^cat/'], 'tasmax': 'new_name'}
    add_attrs: dict
        Dictionary where the keys are the variables and the values are a another dictionary of attributes.
        For global attrs, use the key 'global'.
        eg. {'global': {'title': 'amazing new dataset'}, 'tasmax': {'note': 'important info about tasmax'}}
    change_attr_prefix: str
        Replace "cat/" in the catalogue global attrs by this new string
    to_level: str
        The processing level to assign to the output.

    Returns
    -------
    xr.Dataset
        Cleaned up dataset
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
            convert_calendar_kwargs.setdefault("missing", np.nan)

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
                    converted_var = convert_calendar(
                        ds_copy[var], **convert_calendar_kwargs, missing=np.nan
                    )
                    converted_var = converted_var.interpolate_na(
                        "time", method="linear"
                    )
                else:
                    ocean_var = ds_copy[var].isnull().all("time")
                    converted_var = convert_calendar(
                        ds_copy[var], **convert_calendar_kwargs, missing=missing
                    ).where(~ocean_var)
                ds[var] = converted_var

    # unstack nans
    if maybe_unstack_dict:
        ds = maybe_unstack(ds, **maybe_unstack_dict)

    def _search(a, b):
        if a[-1] == "*":  # check if a is contained in b
            return a[:-1] in b
        elif a[0] == "^":
            return b.startswith(a[1:])
        else:
            return a == b

    ds.attrs["cat/processing_level"] = to_level

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
            new_name = ds_attr.replace("cat/", change_attr_prefix)
            if new_name:
                ds.attrs[new_name] = ds.attrs.pop(ds_attr)

    return ds
