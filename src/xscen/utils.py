"""Common utilities to be used in many places."""

import fnmatch
import gettext
import json
import logging
import os
import re
import warnings
from collections import defaultdict
from collections.abc import Sequence
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from itertools import chain
from pathlib import Path
from types import ModuleType

import cftime
import flox.xarray
import numpy as np
import pandas as pd
import xarray as xr
import xsdba
from xarray.coding import cftime_offsets as cfoff
from xclim.core import units
from xclim.core.calendar import parse_offset
from xclim.core.options import METADATA_LOCALES
from xclim.core.options import OPTIONS as XC_OPTIONS
from xclim.core.utils import uses_dask

from .config import parse_config


logger = logging.getLogger(__name__)

__all__ = [
    "CV",
    "add_attr",
    "change_units",
    "clean_up",
    "date_parser",
    "ensure_correct_time",
    "ensure_new_xrfreq",
    "get_cat_attrs",
    "maybe_unstack",
    "minimum_calendar",
    "natural_sort",
    "stack_drop_nans",
    "standardize_periods",
    "translate_time_chunk",
    "unstack_dates",
    "unstack_fill_nan",
    "update_attr",
    "xclim_convert_units_to",
    "xrfreq_to_timedelta",
]

TRANSLATOR = defaultdict(lambda: lambda s: s)
"""Dictionary of translating objects.

Each key is a two letter locale code and values are functions that return the translated message
as compiled in the gettext catalogs. If a language is not defined or a message not translated,
the function will return the raw message.
"""

try:
    for loc in (Path(__file__).parent / "data").iterdir():
        if loc.is_dir() and len(loc.name) == 2:
            TRANSLATOR[loc.name] = gettext.translation("xscen", localedir=loc.parent, languages=[loc.name]).gettext
except FileNotFoundError as err:
    raise ImportError("Your xscen installation doesn't have compiled translations. Run `make translate` from the source directory to fix.") from err


def update_attr(
    ds: xr.Dataset | xr.DataArray,
    attr: str,
    new: str,
    others: Sequence[xr.Dataset | xr.DataArray] | None = None,
    **fmt,
) -> xr.Dataset | xr.DataArray:
    r"""
    Format an attribute referencing itself in a translatable way.

    Parameters
    ----------
    ds : Dataset or DataArray
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
    \*\*fmt
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
        others_attrs = {f"attr{i}": dso.attrs.get(attr, "").strip(" .") for i, dso in enumerate(others, 1)}
        ds.attrs[attr] = new.format(attr=ds.attrs[attr].strip(" ."), **others_attrs, **fmt)
    # All existing locales
    for key in fnmatch.filter(ds.attrs.keys(), f"{attr}_??"):
        loc = key[-2:]
        others_attrs = {f"attr{i}": dso.attrs.get(key, dso.attrs.get(attr, "")).strip(" .") for i, dso in enumerate(others, 1)}
        ds.attrs[key] = TRANSLATOR[loc](new).format(attr=ds.attrs[key].strip(" ."), **others_attrs, **fmt)


def add_attr(ds: xr.Dataset | xr.DataArray, attr: str, new: str, **fmt):
    """Add a formatted translatable attribute to a dataset."""
    ds.attrs[attr] = new.format(**fmt)
    for loc in XC_OPTIONS[METADATA_LOCALES]:
        ds.attrs[f"{attr}_{loc}"] = TRANSLATOR[loc](new).format(**fmt)


def date_parser(  # noqa: C901
    date: str | cftime.datetime | pd.Timestamp | datetime | pd.Period,
    *,
    end_of_period: bool | str = False,
    out_dtype: str = "datetime",
    strtime_format: str = "%Y-%m-%d",
    freq: str = "h",
) -> str | pd.Period | pd.Timestamp:
    """
    Return a datetime from a string.

    Parameters
    ----------
    date : str, cftime.datetime, pd.Timestamp, datetime.datetime, pd.Period
        Date to be converted
    end_of_period : bool or str
        If 'YE' or 'ME', the returned date will be the end of the year or month that contains the received date.
        If True, the period is inferred from the date's precision, but `date` must be a string,
        otherwise nothing is done.
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
            raise ValueError("Unable to parse cftime date {date}, even when moving back 2 days.")
    elif isinstance(date, pd.Period):
        # Pandas, you're a mess: Period.to_timestamp() fails for out-of-bounds dates (<1677, > 2242), but not when parsing a string...
        date = pd.Timestamp(date.strftime("%Y-%m-%dT%H:%M:%S"))

    if not isinstance(date, pd.Timestamp):
        date = pd.Timestamp(date)

    if isinstance(end_of_period, str) or (end_of_period is True and fmt):
        quasiday = (pd.Timedelta(1, "d") - pd.Timedelta(1, "s")).as_unit(date.unit)
        if end_of_period in ["Y", "YE"] or "m" not in fmt:
            date = pd.tseries.frequencies.to_offset("YE-DEC").rollforward(date) + quasiday
        elif end_of_period in ["M", "ME"] or "d" not in fmt:
            date = pd.tseries.frequencies.to_offset("ME").rollforward(date) + quasiday
        # TODO: Implement subdaily ?

    if out_dtype == "str":
        return date.strftime(strtime_format)
    elif out_dtype == "period":
        return date.to_period(freq)
    else:
        return date


def minimum_calendar(*calendars) -> str:
    """
    Return the minimum calendar from a list.

    Uses the hierarchy: 360_day < noleap < standard < all_leap, and returns one of those names.
    """
    # Unwrap any lists or tuples given in the input, but without destroying strings.
    calendars = [[cal] if isinstance(cal, str) else cal for cal in calendars]
    calendars = list(chain(*calendars))

    # Raise an error if the calendars are not recognized
    unknowns = set(calendars).difference(
        [
            "360_day",
            "365_day",
            "noleap",
            "standard",
            "default",
            "all_leap",
            "366_day",
            "gregorian",
            "proleptic_gregorian",
        ]
    )
    if unknowns:
        warnings.warn(
            f"These calendars are not recognized: {unknowns}. Results may be incorrect.",
            stacklevel=2,
        )

    if "360_day" in calendars:
        out = "360_day"
    elif "noleap" in calendars or "365_day" in calendars:
        out = "noleap"
    elif all(cal in ["all_leap", "366_day"] for cal in calendars):
        out = "all_leap"
    else:
        out = "standard"

    return out


def translate_time_chunk(chunks: dict, calendar: str, timesize: int) -> dict:
    """
    Translate chunk specification for time into a number.

    Parameters
    ----------
    chunks : dict
        Dictionary specifying the chunk sizes for each dimension. The time dimension can be specified as:
        -1 : translates to `timesize`
        'Nyear' : translates to N times the number of days in a year of the given calendar.
    calendar : str
        The calendar type (e.g., 'noleap', '360_day', 'all_leap').
    timesize : int
        The size of the time dimension.

    Returns
    -------
    dict
        The updated chunks dictionary with the time dimension translated to a number.

    Notes
    -----
    -1 translates to `timesize`
    'Nyear' translates to N times the number of days in a year of calendar `calendar`.
    """
    for k, v in chunks.items():
        if isinstance(v, dict):
            chunks[k] = translate_time_chunk(v.copy(), calendar, timesize)
        elif k == "time" and v is not None:
            if isinstance(v, str) and v.endswith("year"):
                n = int(chunks["time"].split("year")[0])
                nt = n * {
                    "noleap": 365,
                    "365_day": 365,
                    "360_day": 360,
                    "all_leap": 366,
                    "366_day": 366,
                }.get(calendar, 365.25)
                if nt != int(nt):
                    warnings.warn(
                        f"The number of days in {chunks['time']} for calendar {calendar} is not an integer. "
                        f"Chunks will not align perfectly with year ends.",
                        stacklevel=2,
                    )
                chunks[k] = int(nt)
            elif v == -1:
                chunks[k] = timesize
    return chunks


@parse_config
def stack_drop_nans(
    ds: xr.Dataset,
    mask: xr.DataArray | list[str],
    *,
    new_dim: str = "loc",
    to_file: str | None = None,
) -> xr.Dataset:
    """
    Stack dimensions into a single axis and drops indexes where the mask is false.

    Parameters
    ----------
    ds : xr.Dataset
      A dataset with the same coords as `mask`.
    mask : xr.DataArray or list of str
      A boolean DataArray with True on the points to keep. The mask will be loaded within this function, but not the dataset.
      Alternatively, a list of dimension names to stack. In this case, a mask will be created by loading all data and checking for NaNs.
      The latter is not recommended for large datasets.
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
    if isinstance(mask, xr.DataArray):
        mask_1d = mask.stack({new_dim: mask.dims})
        out = ds.stack({new_dim: mask.dims}).where(mask_1d, drop=True)
    else:
        mask = ds.coords.to_dataset().drop_vars([v for v in ds.coords if not any(d in mask for d in ds[v].dims)])
        mask = xr.DataArray(
            np.ones(list(mask.sizes.values())), dims=mask.dims, coords=mask.coords
        )  # Make it a DataArray to fit the rest of the function
        out = ds.stack({new_dim: mask.dims}).dropna(new_dim, how="all")
    out = out.reset_index(new_dim)
    for dim in mask.dims:
        out[dim].attrs.update(ds[dim].attrs)

    original_shape = "x".join(map(str, mask.shape))

    if to_file is not None:
        # Set default path to store the information necessary to unstack
        # The name includes the domain and the original shape to uniquely identify the dataset
        domain = ds.attrs.get("cat:domain", "unknown")
        to_file = to_file.format(domain=domain, shape=original_shape)
        if not Path(to_file).parent.exists():
            Path(to_file).parent.mkdir(exist_ok=True)
        # Add all coordinates that might have been affected by the stack
        mask = mask.assign_coords({c: ds[c] for c in ds.coords if any(d in mask.dims for d in ds[c].dims)})
        mask.coords.to_dataset().to_netcdf(to_file)

    # Carry information about original shape to be able to unstack properly
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
    coords: None | (str | os.PathLike | Sequence[str | os.PathLike] | dict[str, xr.DataArray]) = None,
):
    """
    Unstack a Dataset that was stacked by :py:func:`stack_drop_nans`.

    Parameters
    ----------
    ds : xr.Dataset
      A dataset with some dimensions stacked by `stack_drop_nans`.
    dim : str
      The dimension to unstack, same as `new_dim` in `stack_drop_nans`.
    coords : string or os.PathLike or Sequence or dict, optional
        Additional information used to reconstruct coordinates that might have been lost in the stacking (e.g., if a lat/lon grid was all NaNs).
        If a string or os.PathLike : Path to a dataset containing only those coordinates, such as the output of `to_file` in `stack_drop_nans`.
        This is the recommended option.
        If a dictionary : A mapping from the name of the coordinate that was stacked to a DataArray. Better alternative if no file is available.
        If a sequence : The names of the original dimensions that were stacked. Worst option.
        If None (default), same as a sequence, but all coordinates that have `dim` as a single dimension are used as the new dimensions.
        See Notes for more information.

    Returns
    -------
    xr.Dataset
      Same as `ds`, but `dim` has been unstacked to coordinates in `coords`.
      Missing elements are filled according to the defaults of `fill_value` of :py:meth:`xarray.Dataset.unstack`.

    Notes
    -----
    Some information might have been completely lost in the stacking process, for example, if a longitude is NaN across all latitudes.
    It is impossible to recover that information when using `coords` as a list, which is why it is recommended to use a file or a dictionary instead.

    If a dictionary is used, the keys must be the names of the coordinates that were stacked and the values must be the DataArrays.
    This method can recover both dimensions and additional coordinates that were not dimensions in the original dataset, but were stacked.

    If the original stacking was done with `stack_drop_nans` and the `to_file` argument was used, the `coords` argument should be a string with
    the path to the file. Additionally, the file name can contain the formatting fields {shape} and {domain}, which will be automatically  filled
    with the original shape of the dataset and the global attribute 'cat:domain'. If using that dynamic path, it is recommended to fill the
    argument in the xscen config.
    E.g.:

          utils:
            stack_drop_nans:
                to_file: /some_path/coords/coords_{domain}_{shape}.nc
            unstack_fill_nan:
                coords: /some_path/coords/coords_{domain}_{shape}.nc
    """
    if coords is None:
        logger.info("Dataset unstacked using no coords argument.")
        coords = [d for d in ds.coords if ds[d].dims == (dim,)]

    if isinstance(coords, str | os.PathLike):
        # find original shape in the attrs of one of the dimension
        original_shape = "unknown"
        for c in ds.coords:
            if "original_shape" in ds[c].attrs:
                original_shape = ds[c].attrs["original_shape"]
        domain = ds.attrs.get("cat:domain", "unknown")
        coords = coords.format(domain=domain, shape=original_shape)
        msg = f"Dataset unstacked using {coords}."
        logger.info(msg)
        coords = xr.open_dataset(coords)
        # separate coords that are dims or not
        coords_and_dims = {name: x for name, x in coords.coords.items() if name in coords.dims}
        coords_not_dims = {name: x for name, x in coords.coords.items() if name not in coords.dims}

        dims, crds = zip(
            *[(name, crd.load().values) for name, crd in ds.coords.items() if crd.dims == (dim,) and name in coords_and_dims], strict=False
        )

        mindex_obj = pd.MultiIndex.from_arrays(crds, names=dims)
        mindex_coords = xr.Coordinates.from_pandas_multiindex(mindex_obj, dim)

        out = ds.drop_vars(dims).assign_coords(mindex_coords).unstack(dim)

        # only reindex with the dims
        out = out.reindex(**coords_and_dims)
        # add back the coords that aren't dims
        for c in coords_not_dims:
            out[c] = coords[c]
    else:
        coord_not_dim = {}
        # Special case where the dictionary contains both dimensions and other coordinates
        if isinstance(coords, dict):
            coord_not_dim = {k: v for k, v in coords.items() if len(set(v.dims).intersection(list(coords))) != 1}
            coords = deepcopy(coords)
            coords = {k: v for k, v in coords.items() if k in set(coords).difference(coord_not_dim)}

        dims, crds = zip(
            *[(name, crd.load().values) for name, crd in ds.coords.items() if (crd.dims == (dim,) and name in set(coords))], strict=False
        )

        # Reconstruct the dimensions
        mindex_obj = pd.MultiIndex.from_arrays(crds, names=dims)
        mindex_coords = xr.Coordinates.from_pandas_multiindex(mindex_obj, dim)

        out = ds.drop_vars(dims).assign_coords(mindex_coords).unstack(dim)

        if isinstance(coords, dict):
            # Reindex with the coords that were dimensions
            out = out.reindex(**coords)
            # Add back the coordinates that aren't dimensions
            for c in coord_not_dim:
                out[c] = coord_not_dim[c]

        # Reorder the dimensions to match the CF conventions
        order = [out.cf.axes.get(d, [""])[0] for d in ["T", "Z", "Y", "X"]]
        order = [d for d in order if d] + [d for d in out.dims if d not in order]
        out = out.transpose(*order)

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


def get_cat_attrs(ds: xr.Dataset | xr.DataArray | dict, prefix: str = "cat:", var_as_str=False) -> dict:
    """
    Return the catalog-specific attributes from a dataset or dictionary.

    Parameters
    ----------
    ds : xr.Dataset, dict
        Dataset to be parsed. If a dictionary, it is assumed to be the attributes of the dataset (ds.attrs).
    prefix : str
        Prefix automatically generated by intake-esm. With xscen, this should be 'cat:'
    var_as_str : bool
        If True, 'variable' will be returned as a string if there is only one.

    Returns
    -------
    dict
        Compilation of all attributes in a dictionary.
    """
    if isinstance(ds, xr.Dataset | xr.DataArray):
        attrs = ds.attrs
    else:
        attrs = ds
    facets = {k[len(prefix) :]: v for k, v in attrs.items() if k.startswith(f"{prefix}")}

    # to be usable in a path
    if var_as_str and "variable" in facets and not isinstance(facets["variable"], str) and len(facets["variable"]) == 1:
        facets["variable"] = facets["variable"][0]
    return facets


def strip_cat_attrs(ds: xr.Dataset, prefix: str = "cat:"):
    """Remove attributes added from the catalog by `to_dataset` or `extract_dataset`."""
    dsc = ds.copy()
    for k in list(dsc.attrs):
        if k.startswith(prefix):
            del dsc.attrs[k]
    return dsc


@parse_config
def maybe_unstack(
    ds: xr.Dataset,
    dim: str | None = "loc",
    coords: str | None = None,
    rechunk: dict | None = None,
    stack_drop_nans: bool = False,
) -> xr.Dataset:
    """
    If stack_drop_nans is True, unstack and rechunk.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to unstack.
    dim : str, optional
        Dimension to unstack.
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
        ds = unstack_fill_nan(ds, dim=dim, coords=coords)
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

        .. literalinclude:: ../src/xscen/CVs/frequency_to_timedelta.json
           :language: json
           :caption: frequency_to_timedelta

        .. literalinclude:: ../src/xscen/CVs/frequency_to_xrfreq.json
           :language: json
           :caption: frequency_to_xrfreq

        .. literalinclude:: ../src/xscen/CVs/infer_resolution.json
           :language: json
           :caption: infer_resolution

        .. literalinclude:: ../src/xscen/CVs/resampling_methods.json
           :language: json
           :caption: resampling_methods

        .. literalinclude:: ../src/xscen/CVs/variable_names.json
           :language: json
           :caption: variable_names

        .. literalinclude:: ../src/xscen/CVs/xrfreq_to_frequency.json
           :language: json
           :caption: xrfreq_to_frequency

        .. literalinclude:: ../src/xscen/CVs/xrfreq_to_timedelta.json
           :language: json
           :caption: xrfreq_to_timedelta


        """
    ),
)


def __read_CVs(cvfile):  # noqa: N802
    with cvfile.open("r") as f:
        cv = json.load(f)
    is_regex = cv.pop("is_regex", False)
    doc = """Controlled vocabulary mapping from {name}.

    The raw dictionary can be accessed by the dict attribute of this function.

    Parameters
    ----------
    key : str
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


for cvfile in Path(__file__).parent.joinpath("CVs").glob("*.json"):
    try:
        CV.__dict__[cvfile.stem] = __read_CVs(cvfile)
    # FIXME: This is a catch-all, but we should be more specific
    except Exception as err:  # noqa: BLE001
        raise ValueError(f"Unable to process CV file: {cvfile}.") from err


def change_units(ds: xr.Dataset, variables_and_units: dict) -> xr.Dataset:
    """
    Change units of Datasets to non-CF units.

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
            if v in ds:
                if units.units2pint(ds[v]) != units.units2pint(variables_and_units[v]):
                    time_in_ds = units.units2pint(ds[v]).dimensionality.get("[time]")
                    time_in_out = units.units2pint(variables_and_units[v]).dimensionality.get("[time]")

                    if time_in_ds == time_in_out:
                        ds = ds.assign({v: units.convert_units_to(ds[v], variables_and_units[v])})
                    elif time_in_ds - time_in_out == 1:
                        # ds is an amount
                        ds = ds.assign({v: units.amount2rate(ds[v], out_units=variables_and_units[v])})
                    elif time_in_ds - time_in_out == -1:
                        # ds is a rate
                        ds = ds.assign({v: units.rate2amount(ds[v], out_units=variables_and_units[v])})
                    else:
                        raise ValueError(
                            f"No known transformation between {ds[v].units} and {variables_and_units[v]} (temporal dimensionality mismatch)."
                        )
                # update unit name if physical units are equal but not their name (ex. degC vs °C)
                if (units.units2pint(ds[v]) == units.units2pint(variables_and_units[v])) and (ds[v].units != variables_and_units[v]):
                    ds = ds.assign({v: ds[v].assign_attrs(units=variables_and_units[v])})

    return ds


def _convert_units_to_infer(source, target):
    return units.convert_units_to(source, target, context="infer")


@contextmanager
def xclim_convert_units_to():
    """
    Patch xsdba with xclim's units converter.

    Yields
    ------
    None
        In this context, ``xsdba.units.convert_units_to`` is replaced with
        ``xclim.core.units.convert_units_to`` with `context="infer"` activated.
    """
    original_function = xsdba.units._convert_units_to
    new_function = _convert_units_to_infer
    try:
        xsdba.units._convert_units_to = new_function
        yield
    finally:
        xsdba.units._convert_units_to = original_function


def clean_up(  # noqa: C901
    ds: xr.Dataset,
    *,
    variables_and_units: dict | None = None,
    convert_calendar_kwargs: dict | None = None,
    missing_by_var: dict | None = None,
    maybe_unstack_dict: dict | None = None,
    round_var: dict | None = None,
    clip_var: dict | None = None,
    common_attrs_only: None | (dict | list[xr.Dataset | str | os.PathLike]) = None,
    common_attrs_open_kwargs: dict | None = None,
    attrs_to_remove: dict | None = None,
    remove_all_attrs_except: dict | None = None,
    add_attrs: dict | None = None,
    change_attr_prefix: str | dict | None = None,
    to_level: str | None = None,
) -> xr.Dataset:
    """
    Clean up of the dataset.

    It can:
     - convert to the right units using xscen.utils.change_units
     - convert the calendar and interpolate over missing dates
     - call the xscen.utils.maybe_unstack function
     - round variables
     - clip variables
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
        Dictionary of variable to convert. e.g. {'tasmax': 'degC', 'pr': 'mm d-1'}
    convert_calendar_kwargs : dict, optional
        Dictionary of arguments to feed to xarray.Dataset.convert_calendar. This will be the same for all variables.
        If missing_by_vars is given, it will override the 'missing' argument given here.
        Eg. {'calendar': 'standard', 'align_on': 'random'}
    missing_by_var : dict, optional
        Dictionary where the keys are the variables and the values are the argument to feed the `missing`
        parameters of xarray.Dataset.convert_calendar for the given variable with the
        `convert_calendar_kwargs`. When the value of an entry is 'interpolate', the missing values will be filled
        with NaNs, then linearly interpolated over time.
    maybe_unstack_dict : dict, optional
        Dictionary to pass to xscen.common.maybe_unstack function.
        The format should be: {'coords': path_to_coord_file, 'rechunk': {'time': -1 }, 'stack_drop_nans': True}.
    round_var : dict, optional
        Dictionary where the keys are the variables of the dataset and the values are the number of
        decimal places to round to.
    clip_var : dict, optional
        Dictionary where the keys are the variables of the dataset and the values are the arguments to give ``.clip()``
    common_attrs_only : dict, list of datasets, or list of paths, optional
        Dictionary of datasets or list of datasets, or path to NetCDF or Zarr files.
        Keeps only the global attributes that are the same for all datasets and generates a new id.
    common_attrs_open_kwargs : dict, optional
        Dictionary of arguments for xarray.open_dataset(). Used with common_attrs_only if given paths.
    attrs_to_remove : dict, optional
        Dictionary where the keys are the variables and the values are a list of the attrs that should be removed.
        The match is done using re.fullmatch, so the strings can be regex patterns but don't need to contain '^' or '$'.
        For global attrs, use the key 'global'.
        e.g. {'global': ['unnecessary note', 'cell.*'], 'tasmax': 'old_name'}
    remove_all_attrs_except : dict, optional
        Dictionary where the keys are the variables and the values are a list of the attrs that should NOT be removed.
        The match is done using re.fullmatch, so the strings can be regex patterns but don't need to contain '^' or '$'.
        All other attributes will be deleted. For global attrs, use the key 'global'.
        e.g. {'global': ['necessary note', '^cat:'], 'tasmax': 'new_name'}
    add_attrs : dict, optional
        Dictionary where the keys are the variables and the values are a another dictionary of attributes.
        For global attrs, use the key 'global'.
        e.g. {'global': {'title': 'amazing new dataset'}, 'tasmax': {'note': 'important info about tasmax'}}
    change_attr_prefix : str or dict, optional
        If a string, replace "cat:" in the catalog global attributes by this new string.
        If a dictionary, the key is the old prefix and the value is the new prefix.
    to_level : str, optional
        The processing level to assign to the output.

    Returns
    -------
    xr.Dataset
        Cleaned up dataset

    See Also
    --------
    xarray.Dataset.convert_calendar, xarray.DataArray.round, xarray.DataArray.clip
    """
    ds = ds.copy()

    if variables_and_units:
        msg = f"Converting units: {variables_and_units}"
        logger.info(msg)
        ds = change_units(ds=ds, variables_and_units=variables_and_units)
    # convert calendar
    if convert_calendar_kwargs:
        vars_with_no_time = [v for v in ds.data_vars if "time" not in ds[v].dims]
        # create mask of grid point that should always be nan
        ocean = ds.isnull().all("time")
        # if missing_by_var exist make sure missing data are added to time axis
        if missing_by_var:
            if not all(k in missing_by_var.keys() for k in ds.data_vars):
                raise ValueError("All variables must be in 'missing_by_var' if using this option.")
            convert_calendar_kwargs["missing"] = -9999

        # make default `align_on`='`random` when the initial calendar is 360day
        if any(cal == "360_day" for cal in [ds.time.dt.calendar, convert_calendar_kwargs["calendar"]]) and "align_on" not in convert_calendar_kwargs:
            convert_calendar_kwargs["align_on"] = "random"

        msg = f"Converting calendar with {convert_calendar_kwargs}."
        logger.info(msg)
        ds = ds.convert_calendar(**convert_calendar_kwargs).where(~ocean)

        # FIXME: Fix for xarray <= 2025.04.0: https://github.com/pydata/xarray/issues/10266
        for vv in vars_with_no_time:
            if "time" in ds[vv].dims:
                ds[vv] = ds[vv].isel(time=0).drop_vars("time")

        # convert each variable individually
        if missing_by_var:
            # remove 'missing' argument to be replaced by `missing_by_var`
            del convert_calendar_kwargs["missing"]
            for var, missing in missing_by_var.items():
                msg = f"Filling missing {var} with {missing}"
                logging.info(msg)
                if missing == "interpolate":
                    ds_with_nan = ds[var].where(ds[var] != -9999)
                    converted_var = ds_with_nan.chunk({"time": -1}).interpolate_na("time", method="linear")
                else:
                    converted_var = ds[var].where(ds[var] != -9999, other=missing)
                ds = ds.assign({var: converted_var})

    # unstack nans
    if maybe_unstack_dict:
        ds = maybe_unstack(ds, **maybe_unstack_dict)

    if round_var:
        for var, n in round_var.items():
            ds[var] = ds[var].round(n)
            new_history = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Rounded '{var}' to {n} decimals."
            history = f"{new_history}\n{ds[var].attrs['history']}" if "history" in ds[var].attrs else new_history
            ds[var].attrs["history"] = history

    if clip_var:
        for var, c in clip_var.items():
            ds[var] = ds[var].clip(*c, keep_attrs=True)
            new_history = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Clipped '{var}' to {c}."
            history = f"{new_history}\n{ds[var].attrs['history']}" if "history" in ds[var].attrs else new_history
            ds[var].attrs["history"] = history

    if common_attrs_only:
        from .catalog import generate_id

        common_attrs_open_kwargs = common_attrs_open_kwargs or {}
        if isinstance(common_attrs_only, dict):
            common_attrs_only = list(common_attrs_only.values())

        for i in range(len(common_attrs_only)):
            if isinstance(common_attrs_only[i], str | os.PathLike):
                dataset = xr.open_dataset(common_attrs_only[i], **common_attrs_open_kwargs)
            else:
                dataset = common_attrs_only[i]
            attributes = ds.attrs.copy()
            for a_key, a_val in attributes.items():
                if (a_key not in dataset.attrs) or (a_key in ["cat:date_start", "cat:date_end"]) or (a_val != dataset.attrs[a_key]):
                    del ds.attrs[a_key]

        # generate a new id
        try:
            ds.attrs["cat:id"] = generate_id(ds).iloc[0]
        except IndexError as err:
            msg = f"Unable to generate a new id for the dataset. Got {err}."
            logger.warning(msg)

    if to_level:
        ds.attrs["cat:processing_level"] = to_level

    # remove attrs
    if attrs_to_remove:
        for var, list_of_attrs in attrs_to_remove.items():
            obj = ds if var == "global" else ds[var]
            to_remove = list(chain.from_iterable([list(filter(re.compile(attr).fullmatch, list(obj.attrs.keys()))) for attr in list_of_attrs]))
            for attr in to_remove:
                del obj.attrs[attr]

    # delete all attrs, but the ones in the list
    if remove_all_attrs_except:
        for var, list_of_attrs in remove_all_attrs_except.items():
            obj = ds if var == "global" else ds[var]
            to_keep = list(chain.from_iterable([list(filter(re.compile(attr).fullmatch, list(obj.attrs.keys()))) for attr in list_of_attrs]))
            to_remove = list(set(obj.attrs.keys()).difference(to_keep))
            for attr in to_remove:
                del obj.attrs[attr]

    if add_attrs:
        for var, attrs in add_attrs.items():
            obj = ds if var == "global" else ds[var]
            for attrname, attrtmpl in attrs.items():
                obj.attrs[attrname] = attrtmpl

    if change_attr_prefix:
        if isinstance(change_attr_prefix, str):
            change_attr_prefix = {"cat:": change_attr_prefix}
        # Make sure that the prefixes are in the right format
        chg_attr_prefix = {}
        for old_prefix, new_prefix in change_attr_prefix.items():
            if not old_prefix.endswith(":"):
                old_prefix += ":"
            if not new_prefix.endswith(":"):
                new_prefix += ":"
            chg_attr_prefix[old_prefix] = new_prefix

        # Change the prefixes, but keep the order of the keys
        attrs = {}
        for ds_attr in list(ds.attrs.keys()):
            changed = False
            for old_prefix, new_prefix in chg_attr_prefix.items():
                if ds_attr.startswith(old_prefix):
                    new_name = ds_attr.replace(old_prefix, new_prefix)
                    attrs[new_name] = ds.attrs[ds_attr]
                    changed = True
            if not changed:
                attrs[ds_attr] = ds.attrs[ds_attr]
        ds.attrs = attrs

    return ds


def unstack_dates(  # noqa: C901
    ds: xr.Dataset,
    seasons: dict[int, str] | None = None,
    new_dim: str | None = None,
    winter_starts_year: bool = False,
):
    """
    Unstack a multi-season timeseries into a yearly axis and a season one.

    Parameters
    ----------
    ds : xr.Dataset or DataArray
      The xarray object with a "time" coordinate.
      Only supports monthly or coarser frequencies.
      The time axis must be complete and regular (`xr.infer_freq(ds.time)` doesn't fail).
    seasons : dict, optional
      A dictionary from month number (as int) to a season name.
      If not given, it is guessed from the time coordinate frequency.
      See notes.
    new_dim : str, optional
      The name of the new dimension.
      If None, the name is inferred from the frequency of the time axis.
      See notes.
    winter_starts_year : bool
      If True, the year of winter (DJF) is built from the year of January, not December.
      i.e. DJF made from [Dec 1980, Jan 1981, and Feb 1981] will be associated with the year 1981, not 1980.

    Returns
    -------
    xr.Dataset or DataArray
      Same as ds but the time axis is now yearly (YS-JAN) and the seasons are along the new dimension.

    Notes
    -----
    When `seasons` is None, the inferred frequency determines the new coordinate:
    - For MS, the coordinates are the month abbreviations in english (JAN, FEB, etc.)
    - For ?QS-? and other ?MS frequencies, the coordinates are the initials of the months in each season. Ex: QS -> DJF, MAM, JJA, SON.
    - For YS or YS-JAN, the new coordinate has a single value of "annual".
    - For ?YS-? frequencies, the new coordinate has a single value of "annual-{anchor}". Ex: YS-JUL -> "annual-JUL".

    When `new_dim` is None, the new dimension name is inferred from the frequency:
    - For ?YS, ?QS frequencies or ?MS with mult > 1, the new dimension is "season".
    - For MS, the new dimension is "month".

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
        raise ValueError(f"Only monthly frequencies or coarser are supported. Got: {freq}.")

    if new_dim is None:
        if base == "M" and mult == 1:
            new_dim = "month"
        else:
            new_dim = "season"

    if base in "YA":
        if seasons:
            seaname = f"{seasons[first.month]}"
        elif anchor == "JAN":
            seaname = "annual"
        else:
            seaname = f"annual-{anchor}"
        if mult > 1:
            seaname = f"{mult}{seaname}"
        # Fast track for annual, if nothing more needs to be done.
        if winter_starts_year is False:
            dso = ds.expand_dims({new_dim: [seaname]})
            dso["time"] = xr.date_range(
                f"{first.year}-01-01",
                f"{last.year}-01-01",
                freq=f"{mult}YS",
                calendar=calendar,
                use_cftime=use_cftime,
            )
            return dso
        else:
            seasons = seasons or {}
            seasons.update({first.month: seaname})

    if base == "M" and 12 % mult != 0:
        raise ValueError(f"Only periods that divide the year evenly are supported. Got {freq}.")

    # Guess the new season coordinate
    if seasons is None:
        if base == "Q" or (base == "M" and mult > 1):
            # Labels are the month initials
            months = np.array(list("JFMAMJJASOND"))
            n = mult * {"M": 1, "Q": 3}[base]
            seasons = {m: "".join(months[np.array(range(m - 1, m + n - 1)) % 12]) for m in np.unique(ds.time.dt.month)}
        else:  # M or MS
            seasons = xr.coding.cftime_offsets._MONTH_ABBREVIATIONS
    else:
        # Only keep the entries for the months in the data
        seasons = {m: seasons[m] for m in np.unique(ds.time.dt.month)}
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
        # Replace (A,'time',B) by (A,'time', 'season',B) in both the new shape and the new dims
        new_dims = list(chain.from_iterable([d] if d != "time" else ["time", new_dim] for d in da.dims))
        new_shape = [len(new_coords[d]) for d in new_dims]
        # Use dask or numpy's algo.

        if uses_dask(da):
            # This is where it happens. Flox will minimally rechunk
            # so the reshape operation can be performed blockwise
            da = flox.xarray.rechunk_for_blockwise(da, "time", years)
        return xr.DataArray(da.data.reshape(new_shape), dims=new_dims)

    new_coords = dict(ds.coords)
    new_coords.update({"time": new_time, new_dim: seas_list})

    # put other coordinates that depend on time in the new shape
    for coord in new_coords:
        if (coord not in ["time", new_dim]) and ("time" in ds[coord].dims):
            new_coords[coord] = reshape_da(dsp[coord])

    if isinstance(ds, xr.Dataset):
        dso = dsp.map(reshape_da, keep_attrs=True)
    else:
        dso = reshape_da(dsp)
    return dso.assign_coords(**new_coords)


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
            raise ValueError(f"Dataset is labelled as having a sampling frequency of {xrfreq}, but some periods have more than one data point.")
        if (counts.isnull() | (counts == 0)).any().item():
            raise ValueError("The resampling count contains NaNs or 0s. There might be some missing data.")
        ds["time"] = counts.time
    return ds


def standardize_periods(
    periods: list[str | pd.Timestamp] | list[list[str | pd.Timestamp]] | None,
    multiple: bool = True,
    end_of_periods: bool = True,
    out_dtype: str = "str",
) -> list[str] | list[list[str]] | None:
    """
    Reformats the input to a list of strings or Timestamps, ['start', 'end'], or a list of such lists. Does not modify in-place.

    Parameters
    ----------
    periods : list of str or pd.Timestamp, or list of lists of str or pd.Timestamp, optional
      The period(s) to standardize. If None, return None.
    multiple : bool
        If True, return a list of periods, otherwise return a single period.
    end_of_periods: bool or str
        If 'YE' or 'ME', the returned date will be the end of the year or month that contains the received date.
        If True (default), standardizes yearly and monthly periods to end on the last second of the last day of the year/month.
        This parameter is only used for str periods that do not specify the month/day.
    out_dtype : str
        Choices are 'datetime', 'period' or 'str'. Defaults to 'str', which will only output the year.
    """
    if periods is None:
        return periods

    periods = deepcopy(periods)
    if not isinstance(periods[0], list):
        periods = [periods]

    for i in range(len(periods)):
        if len(periods[i]) != 2:
            raise ValueError("Each instance of 'periods' should be comprised of two elements: [start, end].")
        period = periods[i]
        if isinstance(period[0], int) or isinstance(period[0], str):
            period[0] = date_parser(str(period[0]), out_dtype="datetime")
        if isinstance(period[1], int) or isinstance(period[1], str):
            period[1] = date_parser(str(period[1]), out_dtype="datetime", end_of_period=end_of_periods)
        if period[0] > period[1]:
            raise ValueError(f"'periods' should be in chronological order, received {periods[i]}.")
        # TODO: allow more than year in periods for out_dtype = str
        periods[i] = [
            date_parser(period[0], out_dtype=out_dtype, strtime_format="%Y"),
            date_parser(period[1], out_dtype=out_dtype, strtime_format="%Y"),
        ]
    if multiple:
        return periods
    else:
        if len(periods) > 1:
            raise ValueError(f"'period' should be a single instance of [start, end], received {len(periods)}.")
        return periods[0]


def season_sort_key(idx: pd.Index, name: str | None = None):
    """
    Get a proper sort key for a "season" or "month" index to avoid alphabetical sorting.

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
    except (TypeError, ValueError) as err:
        # ValueError if string not in seasons, or value not in months
        # TypeError if season element was not a string.
        logging.error(err)
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


def _xarray_defaults(**kwargs):
    """Translate from xscen's extract names to intake-esm names and put better defaults."""
    if "xr_open_kwargs" in kwargs:
        kwargs["xarray_open_kwargs"] = kwargs.pop("xr_open_kwargs")
    if "xr_combine_kwargs" in kwargs:
        kwargs["xarray_combine_by_coords_kwargs"] = kwargs.pop("xr_combine_kwargs")

    kwargs.setdefault("xarray_open_kwargs", {}).setdefault("chunks", {})
    kwargs.setdefault("xarray_combine_by_coords_kwargs", {}).setdefault("data_vars", "minimal")
    return kwargs


def rechunk_for_resample(obj: xr.DataArray | xr.Dataset, **resample_kwargs):
    if not uses_dask(obj):
        return obj

    res = obj.resample(**resample_kwargs)
    return flox.xarray.rechunk_for_blockwise(obj, res._dim, res._codes)


def publish_release_notes(*args, **kwargs):
    """Backward compatibility for the old function."""
    warnings.warn(
        "'xscen.utils.publish_release_notes' has been moved to 'xscen.testing.publish_release_notes'."
        "Support for this function will be removed in xscen v0.12.0.",
        FutureWarning,
        stacklevel=2,
    )

    from .testing import publish_release_notes as prn

    return prn(*args, **kwargs)


def show_versions(*args, **kwargs):
    """Backward compatibility for the old function."""
    warnings.warn(
        "'xscen.utils.show_versions' has been moved to 'xscen.testing.show_versions'.Support for this function will be removed in xscen v0.12.0.",
        FutureWarning,
        stacklevel=2,
    )

    from .testing import show_versions as sv

    return sv(*args, **kwargs)
