"""
Common things to be used at many places
"""
import json
import os
import re
from pathlib import Path
from types import ModuleType
from typing import Optional

import pandas as pd
import xarray as xr

from .config import parse_config


def minimum_calendar(*calendars):
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


def translate_time_chunk(chunks, calendar, timesize):
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
):
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
def unstack_fill_nan(ds: xr.Dataset, *, dim: str = "loc", coords=None):
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


def maybe_unstack(ds, coords, rechunk: bool = None, stack_drop_nans: bool = False):
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
