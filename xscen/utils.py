"""Common utilities to be used in many places."""

import fnmatch
import gettext
import json
import logging
import os
import re
from collections import defaultdict
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
from xarray.coding import cftime_offsets as cfoff
from xclim.core import units
from xclim.core.calendar import convert_calendar, get_calendar, parse_offset
from xclim.core.options import METADATA_LOCALES
from xclim.core.options import OPTIONS as XC_OPTIONS
from xclim.core.utils import uses_dask
from xclim.testing.utils import show_versions as _show_versions

from .config import parse_config

logger = logging.getLogger(__name__)

__all__ = [
    "add_attr",
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
    "unstack_dates",
    "unstack_fill_nan",
    "update_attr",
]

TRANSLATOR = defaultdict(lambda: lambda s: s)
"""Dictionary of translating objects.

Each key is a two letter locale code and values are functions that return the translated message as compiled in the gettext catalogs.
If a language is not defined or a message not translated, the function will return the raw message.
"""

try:
    for loc in (Path(__file__).parent / "data").iterdir():
        if loc.is_dir() and len(loc.name) == 2:
            TRANSLATOR[loc.name] = gettext.translation(
                "xscen", localedir=loc.parent, languages=[loc.name]
            ).gettext
except FileNotFoundError as err:
    raise ImportError(
        "Your xscen installation doesn't have compiled translations. Run `make translate` from the source directory to fix."
    ) from err


def update_attr(
    ds: Union[xr.Dataset, xr.DataArray],
    attr: str,
    new: str,
    others: Optional[Sequence[Union[xr.Dataset, xr.DataArray]]] = None,
    **fmt,
) -> Union[xr.Dataset, xr.DataArray]:
    """Format an attribute referencing itself in a translatable way.

    Parameters
    ----------
    ds: Dataset or DataArray
        The input object with the attribute to update.
    attr : str
        Attribute name.
    new : str
        New attribute as a template string. It may refer to the old version
        of the attribute with the "{attr}" field.
    others: Sequence of Datasets or DataArrays
        Other objects from which we can extract the attribute `attr`.
        These can be referenced as "{attrXX}" in `new`, where XX is the based-1 index of the other source in `others`.
        If they don't have the `attr` attribute, an empty string is sent to the string formatting.
        See notes.
    fmt:
        Other formatting data.

    Returns
    -------
    `ds`, but updated with the new version of `attr`, in each of the activated languages.

    Notes
    -----
    This is meant for constructing attributes by extending a previous version
    or combining it from different sources. For example, given a `ds` that has `long_name="Variability"`:

    >>> update_attr(ds, "long_name", _("Mean of {attr}"))

    Will update the "long_name" of `ds` with `long_name="Mean of Variability"`.
    The use of `_(...)` allows the detection of this string by the translation manager. The function
    will be able to add a translatable version of the string for each activated language, for example adding
    a `long_name_fr="Moyenne de Variabilité"` (assuming a `long_name_fr` was present on the initial `ds`).

    If the new attribute is an aggregation from multiple sources, these can be passed in `others`.

    >>> update_attr(
    ...     ds0,
    ...     "long_name",
    ...     _("Addition of {attr} and {attr1}, divided by {attr2}"),
    ...     others=[ds1, ds2],
    ... )

    Here, `ds0` will have it's `long_name` updated with the passed string, where  `attr1` is the `long_name` of `ds1`
    and `attr2` the `long_name` of `ds2`. The process will be repeated for each localized `long_name` available on `ds0`.
    For example, if `ds0` has a `long_name_fr`, the template string is translated and
    filled with the `long_name_fr` attributes of  `ds0`, `ds1` and `ds2`.
    If the latter don't exist, the english version is used instead.
    """
    others = others or []
    # .strip(' .') removes trailing and leading whitespaces and dots
    if attr in ds.attrs:
        others = {
            f"attr{i}": dso.attrs.get(attr, "").strip(" .")
            for i, dso in enumerate(others, 1)
        }
        ds.attrs[attr] = new.format(attr=ds.attrs[attr].strip(" ."), **others, **fmt)
    # All existing locales
    for key in fnmatch.filter(ds.attrs.keys(), f"{attr}_??"):
        loc = key[-2:]
        others = {
            f"attr{i}": dso.attrs.get(key, dso.attrs.get(attr, "")).strip(" .")
            for i, dso in enumerate(others, 1)
        }
        ds.attrs[key] = TRANSLATOR[loc](new).format(
            attr=ds.attrs[key].strip(" ."), **others, **fmt
        )


def add_attr(ds: Union[xr.Dataset, xr.DataArray], attr: str, new: str, **fmt):
    """Add a formatted translatable attribute to a dataset."""
    ds.attrs[attr] = new.format(**fmt)
    for loc in XC_OPTIONS[METADATA_LOCALES]:
        ds.attrs[f"{attr}_{loc}"] = TRANSLATOR[loc](new).format(**fmt)


def date_parser(  # noqa: C901
    date: Union[str, cftime.datetime, pd.Timestamp, datetime, pd.Period],
    *,
    end_of_period: Union[bool, str] = False,
    out_dtype: str = "datetime",
    strtime_format: str = "%Y-%m-%d",
    freq: str = "H",
) -> Union[str, pd.Period, pd.Timestamp]:
    """Return a datetime from a string.

    Parameters
    ----------
    date : str, cftime.datetime, pd.Timestamp, datetime.datetime, pd.Period
        Date to be converted
    end_of_period : bool or str
        If 'Y' or 'M', the returned date will be the end of the year or month that contains the received date.
        If True, the period is inferred from the date's precision, but `date` must be a string, otherwise nothing is done.
    out_dtype : str
        Choices are 'datetime', 'period' or 'str'
    strtime_format : str
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
            except (ValueError, pd._libs.tslibs.parsing.DateParseError):
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
    elif out_dtype == "period":
        return date.to_period(freq)
    else:
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
    ds: xr.Dataset,
    *,
    dim: str = "loc",
    coords: Optional[
        Union[str, os.PathLike, Sequence[Union[str, os.PathLike]], dict]
    ] = None,
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


def natural_sort(_list: list[str]):
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
    ds: Union[xr.Dataset, xr.DataArray, dict], prefix: str = "cat:", var_as_str=False
) -> dict:
    """Return the catalog-specific attributes from a dataset or dictionary.

    Parameters
    ----------
    ds: xr.Dataset, dict
        Dataset to be parsed. If a dictionary, it is assumed to be the attributes of the dataset (ds.attrs).
    prefix: str
        Prefix automatically generated by intake-esm. With xscen, this should be 'cat:'
    var_as_str: bool
        If True, 'variable' will be returned as a string if there is only one.

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
    coords: Optional[str] = None,
    rechunk: Optional[dict] = None,
    stack_drop_nans: bool = False,
) -> xr.Dataset:
    """If stack_drop_nans is True, unstack and rechunk.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to unstack.
    coords : str, optional
        Path to a dataset containing the coords to unstack (and only those).
    rechunk : dict, optional
        If not None, rechunk the dataset after unstacking.
    stack_drop_nans : bool
        If True, unstack the dataset and rechunk it.
        If False, do nothing.

    Returns
    -------
    xr.Dataset
        Unstacked dataset.
    """
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


def clean_up(  # noqa: C901
    ds: xr.Dataset,
    *,
    variables_and_units: Optional[dict] = None,
    convert_calendar_kwargs: Optional[dict] = None,
    missing_by_var: Optional[dict] = None,
    maybe_unstack_dict: Optional[dict] = None,
    round_var: Optional[dict] = None,
    common_attrs_only: Optional[
        Union[dict, list[Union[xr.Dataset, str, os.PathLike]]]
    ] = None,
    common_attrs_open_kwargs: Optional[dict] = None,
    attrs_to_remove: Optional[dict] = None,
    remove_all_attrs_except: Optional[dict] = None,
    add_attrs: Optional[dict] = None,
    change_attr_prefix: Optional[str] = None,
    to_level: Optional[str] = None,
) -> xr.Dataset:
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
    variables_and_units : dict, optional
        Dictionary of variable to convert. eg. {'tasmax': 'degC', 'pr': 'mm d-1'}
    convert_calendar_kwargs : dict, optional
        Dictionary of arguments to feed to xclim.core.calendar.convert_calendar. This will be the same for all variables.
        If missing_by_vars is given, it will override the 'missing' argument given here.
        Eg. {target': default, 'align_on': 'random'}
    missing_by_var : dict, optional
        Dictionary where the keys are the variables and the values are the argument to feed the `missing`
        parameters of the xclim.core.calendar.convert_calendar for the given variable with the `convert_calendar_kwargs`.
        When the value of an entry is 'interpolate', the missing values will be filled with NaNs, then linearly interpolated over time.
    maybe_unstack_dict : dict, optional
        Dictionary to pass to xscen.common.maybe_unstack function.
        The format should be: {'coords': path_to_coord_file, 'rechunk': {'time': -1 }, 'stack_drop_nans': True}.
    round_var : dict, optional
        Dictionary where the keys are the variables of the dataset and the values are the number of decimal places to round to
    common_attrs_only : dict, list of datasets, or list of paths, optional
        Dictionnary of datasets or list of datasets, or path to NetCDF or Zarr files.
        Keeps only the global attributes that are the same for all datasets and generates a new id.
    common_attrs_open_kwargs : dict, optional
        Dictionary of arguments for xarray.open_dataset(). Used with common_attrs_only if given paths.
    attrs_to_remove : dict, optional
        Dictionary where the keys are the variables and the values are a list of the attrs that should be removed.
        For global attrs, use the key 'global'.
        The element of the list can be exact matches for the attributes name
        or use the same substring matching rules as intake_esm:
        - ending with a '*' means checks if the substring is contained in the string
        - starting with a '^' means check if the string starts with the substring.
        eg. {'global': ['unnecessary note', 'cell*'], 'tasmax': 'old_name'}
    remove_all_attrs_except : dict, optional
        Dictionary where the keys are the variables and the values are a list of the attrs that should NOT be removed,
        all other attributes will be deleted. If None (default), nothing will be deleted.
        For global attrs, use the key 'global'.
        The element of the list can be exact matches for the attributes name
        or use the same substring matching rules as intake_esm:
        - ending with a '*' means checks if the substring is contained in the string
        - starting with a '^' means check if the string starts with the substring.
        eg. {'global': ['necessary note', '^cat:'], 'tasmax': 'new_name'}
    add_attrs : dict, optional
        Dictionary where the keys are the variables and the values are a another dictionary of attributes.
        For global attrs, use the key 'global'.
        eg. {'global': {'title': 'amazing new dataset'}, 'tasmax': {'note': 'important info about tasmax'}}
    change_attr_prefix : str, optional
        Replace "cat:" in the catalog global attrs by this new string
    to_level : str, optional
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
            if isinstance(common_attrs_only[i], (str, os.PathLike)):
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
    style: str = "md",
    file: Optional[Union[os.PathLike, StringIO, TextIO]] = None,
    changes: Union[str, os.PathLike] = None,
) -> Optional[str]:
    """Format release history in Markdown or ReStructuredText.

    Parameters
    ----------
    style : {"rst", "md"}
        Use ReStructuredText (`rst`) or Markdown (`md`) formatting. Default: Markdown.
    file : {os.PathLike, StringIO, TextIO, None}
        If provided, prints to the given file-like object. Otherwise, returns a string.
    changes : {str, os.PathLike}, optional
        If provided, manually points to the file where the changelog can be found.
        Assumes a relative path otherwise.

    Returns
    -------
    str, optional

    Notes
    -----
    This function exists solely for development purposes.
    Adapted from xclim.testing.utils.publish_release_notes.
    """
    if isinstance(changes, (str, Path)):
        changes_file = Path(changes).absolute()
    else:
        changes_file = Path(__file__).absolute().parents[2].joinpath("CHANGES.rst")

    if not changes_file.exists():
        raise FileNotFoundError("Changes file not found in xscen file tree.")

    with open(changes_file) as f:
        changes = f.read()

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
        changes = re.sub(search, replacement, changes)

    if style == "md":
        changes = changes.replace("=========\nChangelog\n=========", "# Changelog")

        titles = {r"\n(.*?)\n([\-]{1,})": "-", r"\n(.*?)\n([\^]{1,})": "^"}
        for title_expression, level in titles.items():
            found = re.findall(title_expression, changes)
            for grouping in found:
                fixed_grouping = (
                    str(grouping[0]).replace("(", r"\(").replace(")", r"\)")
                )
                search = rf"({fixed_grouping})\n([\{level}]{'{' + str(len(grouping[1])) + '}'})"
                replacement = f"{'##' if level=='-' else '###'} {grouping[0]}"
                changes = re.sub(search, replacement, changes)

        link_expressions = r"[\`]{1}([\w\s]+)\s<(.+)>`\_"
        found = re.findall(link_expressions, changes)
        for grouping in found:
            search = rf"`{grouping[0]} <.+>`\_"
            replacement = f"[{str(grouping[0]).strip()}]({grouping[1]})"
            changes = re.sub(search, replacement, changes)

    if not file:
        return changes
    if isinstance(file, (Path, os.PathLike)):
        file = Path(file).open("w")
    print(changes, file=file)


def unstack_dates(
    ds: xr.Dataset,
    seasons: Optional[dict[int, str]] = None,
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
      A dictionary from month number (as int) to a season name.
      If not given, it is guessed from the time coord's frequency.
      See notes.
    new_dim: str
      The name of the new dimension.
    winter_starts_year: bool
      If True, the year of winter (DJF) is built from the year of January, not December.
      i.e. DJF made from [Dec 1980, Jan 1981, and Feb 1981] will be associated with the year 1981, not 1980.

    Returns
    -------
    xr.Dataset or DataArray
      Same as ds but the time axis is now yearly (YS-JAN) and the seasons are along the new dimension.

    Notes
    -----
    When `season` is None, the inferred frequency determines the new coordinate:

    - For MS, the coordinates are the month abbreviations in english (JAN, FEB, etc.)
    - For ?QS-? and other ?MS frequencies, the coordinates are the initials of the months in each season.
      Ex: QS-DEC (with winter_starts_year=True) : DJF, MAM, JJA, SON.
    - For YS or YS-JAN, the new coordinate has a single value of "annual".
    - For ?YS-? frequencies, the new coordinate has a single value of "annual-{anchor}", were "anchor"
      is the abbreviation of the first month of the year. Ex: YS-JUL -> "annual-JUL".
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
            "xscen",
            # Main packages
            "cartopy",
            "cftime",
            "cf_xarray",
            "clisops",
            "dask",
            "flox",
            "fsspec",
            "geopandas",
            "h5netcdf",
            "h5py",
            "intake_esm",
            "matplotlib",
            "netCDF4",
            "numpy",
            "pandas",
            "parse",
            "pyyaml",
            "rechunker",
            "shapely",
            "sparse",
            "toolz",
            "xarray",
            "xclim",
            "xesmf",
            "zarr",
            # For translations
            "babel",
            # Opt
            "nc-time-axis",
            "pyarrow",
            # Extras specific to this function
            "fastprogress",
            "intake",
            "pydantic",
            "requests",
            "xcollection",
            "yaml",
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


def standardize_periods(
    periods: Optional[Union[list[str], list[list[str]]]], multiple: bool = True
) -> Optional[Union[list[str], list[list[str]]]]:
    """Reformats the input to a list of strings, ['start', 'end'], or a list of such lists.

    Parameters
    ----------
    periods : list of str or list of lists of str, optional
      The period(s) to standardize. If None, return None.
    multiple : bool
        If True, return a list of periods, otherwise return a single period.
    """
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


def season_sort_key(idx: pd.Index, name: Optional[str] = None):
    """Get a proper sort key for a "season"  or "month" index to avoid alphabetical sorting.

    If any of the values in the index is not recognized as a 3-letter
    season code or a 3-letter month abbreviation, the operation is
    aborted and the index is returned untouched.
    DJF is the first season of the year.

    Parameters
    ----------
    idx : pd.Index
      Any array that implements a `map` method.
      If name is "month", index elements are expected to be 3-letter month abbreviations, uppercase (JAN, FEB, etc).
      If name is "season", index elements are expected to be 3-letter season abbreviations, uppercase (DJF, AMJ, OND, etc.)
      If anything else, the index is returned untouched.
    name : str, optional
      The index name. By default, the `name` attribute of the index is used, if present.

    Returns
    -------
    idx : Integer sort key for months and seasons, the input index untouched otherwise.
    """
    try:
        if (name or getattr(idx, "name", None)) == "season":
            m = "DJFMAMJJASONDJ"
            return idx.map(m.index)
        if (name or getattr(idx, "name", None)) == "month":
            m = list(xr.coding.cftime_offsets._MONTH_ABBREVIATIONS.values())
            return idx.map(m.index)
    except (TypeError, ValueError):
        # ValueError if string not in seasons, or value not in months
        # TypeError if season element was not a string.
        pass
    return idx


def xrfreq_to_timedelta(freq: str):
    """Approximate the length of a period based on its frequency offset."""
    N, B, _, _ = parse_offset(freq)
    return N * pd.Timedelta(CV.xrfreq_to_timedelta(B, "NaT"))


def ensure_new_xrfreq(freq: str) -> str:  # noqa: C901
    """Convert the frequency string to the newer syntax (pandas >= 2.2) if needed."""
    # Copied from xarray xr.coding.cftime_offsets._legacy_to_new_freq
    # https://github.com/pydata/xarray/pull/8627/files
    if not isinstance(freq, str):
        # For when freq is NaN or None in a catalog
        return freq
    try:
        freq_as_offset = cfoff.to_offset(freq, warn=False)
    except ValueError:
        # freq may be valid in pandas but not in xarray
        return freq

    if isinstance(freq_as_offset, cfoff.MonthEnd) and "ME" not in freq:
        freq = freq.replace("M", "ME")
    elif isinstance(freq_as_offset, cfoff.QuarterEnd) and "QE" not in freq:
        freq = freq.replace("Q", "QE")
    elif isinstance(freq_as_offset, cfoff.YearBegin) and "YS" not in freq:
        freq = freq.replace("AS", "YS")
    elif isinstance(freq_as_offset, cfoff.YearEnd):
        if "A-" in freq:
            # Check for and replace "A-" instead of just "A" to prevent
            # corrupting anchored offsets that contain "Y" in the month
            # abbreviation, e.g. "A-MAY" -> "YE-MAY".
            freq = freq.replace("A-", "YE-")
        elif "Y-" in freq:
            freq = freq.replace("Y-", "YE-")
        elif freq.endswith("A"):
            # the "A-MAY" case is already handled above
            freq = freq.replace("A", "YE")
        elif "YE" not in freq and freq.endswith("Y"):
            # the "Y-MAY" case is already handled above
            freq = freq.replace("Y", "YE")
    elif isinstance(freq_as_offset, cfoff.Hour):
        freq = freq.replace("H", "h")
    elif isinstance(freq_as_offset, cfoff.Minute):
        freq = freq.replace("T", "min")
    elif isinstance(freq_as_offset, cfoff.Second):
        freq = freq.replace("S", "s")
    elif isinstance(freq_as_offset, cfoff.Millisecond):
        freq = freq.replace("L", "ms")
    elif isinstance(freq_as_offset, cfoff.Microsecond):
        freq = freq.replace("U", "us")

    return freq
